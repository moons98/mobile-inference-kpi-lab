package com.example.kpilab

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer

/**
 * SD v1.5 Inpainting pipeline with 9ch UNet.
 * Sessions: VAE Encoder / Text Encoder / Inpainting UNet / VAE Decoder.
 * YOLO-seg session is managed separately by the caller (BenchmarkRunner).
 *
 * Pipeline:
 * ROI image + mask ??VAE Enc (횞2) ??Add Noise ??UNet loop (9ch) ??VAE Dec ??output ROI
 */
class InpaintPipeline(
    private val context: Context,
    private val config: BenchmarkConfig
) {
    companion object {
        private const val TAG = "InpaintPipeline"
        private const val LATENT_CHANNELS = 4
        private const val VAE_SCALE_FACTOR = 0.18215f
        private const val SEED = 42L
    }

    private var ortEnv: OrtEnvironment? = null

    // 4 SD sessions (YOLO-seg??蹂꾨룄 愿由?
    private var vaeEncoder: OrtRunner? = null
    private var textEncoder: OrtRunner? = null
    private var unet: OrtRunner? = null
    private var vaeDecoder: OrtRunner? = null

    private var tokenizer: Tokenizer? = null
    private var scheduler: Scheduler? = null

    // Precomputed text embeddings (when skipTextEncode = true)
    private var cachedTextEmbeddings: FloatArray? = null

    /** Cold start timing per SD component */
    data class ColdStartTiming(
        val vaeEncLoadMs: Long = 0,
        val textEncLoadMs: Long = 0,
        val unetLoadMs: Long = 0,
        val vaeDecLoadMs: Long = 0
    ) {
        val totalLoadMs: Long get() = vaeEncLoadMs + textEncLoadMs + unetLoadMs + vaeDecLoadMs
    }

    /** Stage timing for one inpainting run */
    data class StageTiming(
        val roiCropMs: Float = 0f,
        val tokenizeMs: Float = 0f,
        val textEncMs: Float = 0f,
        val vaeEncMs: Float = 0f,
        val maskedImgPrepMs: Float = 0f,
        val addNoiseMs: Float = 0f,
        val unetTotalMs: Float = 0f,
        val vaeDecMs: Float = 0f,
        val compositeMs: Float = 0f,
        val schedulerOverheadMs: Float = 0f
    ) {
        val inpaintE2eMs: Float
            get() = roiCropMs + tokenizeMs + textEncMs + vaeEncMs + maskedImgPrepMs +
                    addNoiseMs + unetTotalMs + vaeDecMs + compositeMs
    }

    /** Per UNet step detail */
    data class StepDetail(
        val stepIndex: Int,
        val inputCreateMs: Float,
        val sessionRunMs: Float,
        val outputCopyMs: Float,
        val schedulerStepMs: Float,
        val stepTotalMs: Float
    )

    /** Full inpainting result */
    data class InpaintResult(
        val outputImage: Bitmap,
        val stageTiming: StageTiming,
        val stepDetails: List<StepDetail>,
        val actualSteps: Int
    )

    /** Progress callback */
    interface ProgressListener {
        fun onStageStart(stage: String)
        fun onUnetStep(step: Int, totalSteps: Int)
    }

    var coldStartTiming: ColdStartTiming? = null
        private set

    /**
     * Initialize 4 SD sessions. Call once at benchmark start.
     */
    fun initialize(): Boolean {
        Log.i(TAG, "=== Initializing Inpaint Pipeline ===")
        Log.i(TAG, "Config: $config")

        try {
            tokenizer = Tokenizer(context)
        } catch (e: Exception) {
            Log.e(TAG, "Tokenizer init failed: ${e.message}. " +
                    "Run scripts/sd/extract_tokenizer_assets.py to generate vocab.json + merges.txt")
            return false
        }
        scheduler = Scheduler()

        val ep = config.sdBackend
        val fp16 = config.useNpuFp16
        val perf = config.htpPerformanceMode
        val modelDir = config.modelDir

        val enableProfiling = config.phase == BenchmarkPhase.SINGLE_ERASE

        val env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
        ortEnv = env

        // VAE Encoder
        vaeEncoder = OrtRunner(context).also {
            val prec = config.sdPrecisionFor(SdComponent.VAE_ENCODER)
            val path = "$modelDir/${SdComponent.VAE_ENCODER.filename(prec)}"
            if (!it.initialize(path, ep, fp16, true, perf, false, env)) {
                Log.e(TAG, "VAE Encoder init failed: ${it.lastError}")
                return false
            }
        }

        // Text Encoder — skip if precomputed embeddings exist
        if (config.skipTextEncode) {
            val npyPath = "$modelDir/text_embeddings.npy"
            val loaded = loadNpyFloatArray(npyPath)
            if (loaded != null) {
                cachedTextEmbeddings = loaded
                Log.i(TAG, "Loaded cached text embeddings from $npyPath (${loaded.size} floats)")
            } else {
                Log.w(TAG, "skipTextEncode=true but $npyPath not found, falling back to model")
                textEncoder = OrtRunner(context).also {
                    val prec = config.sdPrecisionFor(SdComponent.TEXT_ENCODER)
                    val path = "$modelDir/${SdComponent.TEXT_ENCODER.filename(prec)}"
                    if (!it.initialize(path, ep, fp16, true, perf, false, env)) {
                        Log.e(TAG, "Text Encoder init failed: ${it.lastError}")
                        return false
                    }
                }
            }
        } else {
            textEncoder = OrtRunner(context).also {
                val prec = config.sdPrecisionFor(SdComponent.TEXT_ENCODER)
                val path = "$modelDir/${SdComponent.TEXT_ENCODER.filename(prec)}"
                if (!it.initialize(path, ep, fp16, true, perf, false, env)) {
                    Log.e(TAG, "Text Encoder init failed: ${it.lastError}")
                    return false
                }
            }
        }

        // Inpainting UNet (9ch)
        unet = OrtRunner(context).also {
            val prec = config.sdPrecisionFor(SdComponent.INPAINT_UNET)
            val path = "$modelDir/${SdComponent.INPAINT_UNET.filename(prec)}"
            if (!it.initialize(path, ep, fp16, true, perf, enableProfiling, env)) {
                Log.e(TAG, "Inpainting UNet init failed: ${it.lastError}")
                return false
            }
        }

        // VAE Decoder
        vaeDecoder = OrtRunner(context).also {
            val prec = config.sdPrecisionFor(SdComponent.VAE_DECODER)
            val path = "$modelDir/${SdComponent.VAE_DECODER.filename(prec)}"
            if (!it.initialize(path, ep, fp16, true, perf, false, env)) {
                Log.e(TAG, "VAE Decoder init failed: ${it.lastError}")
                return false
            }
        }

        val textEncTiming = if (textEncoder != null) {
            textEncoder!!.modelLoadMs + textEncoder!!.sessionCreateMs
        } else 0L
        coldStartTiming = ColdStartTiming(
            vaeEncLoadMs = vaeEncoder!!.modelLoadMs + vaeEncoder!!.sessionCreateMs,
            textEncLoadMs = textEncTiming,
            unetLoadMs = unet!!.modelLoadMs + unet!!.sessionCreateMs,
            vaeDecLoadMs = vaeDecoder!!.modelLoadMs + vaeDecoder!!.sessionCreateMs
        )
        Log.i(TAG, "Cold start: ${coldStartTiming!!.totalLoadMs}ms total")
        Log.i(TAG, "=== Inpaint Pipeline Ready ===")
        return true
    }

    /**
     * Run one full inpainting.
     * @param roiBitmap ROI image (already cropped, any size ??will be resized to 512짼)
     * @param roiMask ROI mask (same size as roiBitmap, 255=inpaint area)
     * @param listener Progress callback
     * @return InpaintResult or null on failure
     */
    fun inpaint(
        roiBitmap: Bitmap,
        roiMask: Bitmap,
        listener: ProgressListener? = null
    ): InpaintResult? {
        val vaeEnc = vaeEncoder ?: run { Log.e(TAG, "inpaint: vaeEncoder is null"); return null }
        val unetRunner = unet ?: run { Log.e(TAG, "inpaint: unet is null"); return null }
        val vaeDec = vaeDecoder ?: run { Log.e(TAG, "inpaint: vaeDecoder is null"); return null }
        val sched = scheduler ?: run { Log.e(TAG, "inpaint: scheduler is null"); return null }

        val resolution = config.resolution
        val latentSize = config.latentSize
        val stepDetails = mutableListOf<StepDetail>()

        // === Stage: Text Encode (cached or live) ===
        val textEmbeddings: FloatArray
        val tokenizeMs: Float
        val textEncMs: Float

        if (cachedTextEmbeddings != null) {
            // Use precomputed embeddings — skip tokenize + text encoder
            listener?.onStageStart("text_encode (cached)")
            val t0 = System.nanoTime()
            textEmbeddings = cachedTextEmbeddings!!
            tokenizeMs = 0f
            textEncMs = nsToMs(System.nanoTime() - t0)
            Log.i(TAG, "Using cached text embeddings (${textEmbeddings.size} floats)")
        } else {
            val textEnc = textEncoder ?: return null
            val tok = tokenizer ?: return null

            listener?.onStageStart("tokenize")
            val tokStart = System.nanoTime()
            val (tokenIds, tokenShape) = tok.tokenize(config.prompt)
            tokenizeMs = nsToMs(System.nanoTime() - tokStart)

            listener?.onStageStart("text_encode")
            val textEncStart = System.nanoTime()
            val textEncInputName = textEnc.inputNames.firstOrNull() ?: "input_ids"
            val textEncResult = textEnc.runMixed(
                mapOf(textEncInputName to Pair(tokenIds as Any, tokenShape))
            ) ?: run { Log.e(TAG, "inpaint: textEncoder.run returned null"); return null }
            textEmbeddings = textEncResult.outputs.values.first() as FloatArray
            textEncMs = nsToMs(System.nanoTime() - textEncStart)
        }

        // === Stage: VAE Encode (image_latent) ===
        listener?.onStageStart("vae_encode")
        val vaeEncStart = System.nanoTime()
        val (imageInput, imageShape) = ImagePreprocessor.preprocess(roiBitmap, resolution)
        val vaeEncResult = vaeEnc.run(imageInput, imageShape) ?: run { Log.e(TAG, "inpaint: vaeEncoder.run returned null"); return null }
        // Note: VAE Encoder ONNX wrapper already includes * scaling_factor internally.
        val imageLatent = vaeEncResult.outputs.values.first() as FloatArray
        val vaeEncMs = nsToMs(System.nanoTime() - vaeEncStart)

        // === Stage: Masked Image Prep (masked_image_latent) ===
        listener?.onStageStart("masked_image_prep")
        val maskedPrepStart = System.nanoTime()
        // Create masked image: ROI 횞 (1 ??mask)
        val roiResized = Bitmap.createScaledBitmap(roiBitmap, resolution, resolution, true)
        val maskResized = Bitmap.createScaledBitmap(roiMask, resolution, resolution, true)
        val maskedImage = ImagePreprocessor.createMaskedImage(roiResized, maskResized)
        roiResized.recycle()

        // VAE Encode the masked image (session 3 ?ъ궗??
        val (maskedInput, maskedShape) = ImagePreprocessor.preprocess(maskedImage, resolution)
        maskedImage.recycle()
        val maskedEncResult = vaeEnc.run(maskedInput, maskedShape) ?: run { Log.e(TAG, "inpaint: maskedImage vaeEncoder.run returned null"); return null }
        val maskedImageLatent = maskedEncResult.outputs.values.first() as FloatArray
        val maskedImgPrepMs = nsToMs(System.nanoTime() - maskedPrepStart)

        // === Mask ??latent space (64횞64) ===
        val (maskLatentBuffer, _) = ImagePreprocessor.resizeMaskToLatent(maskResized, latentSize)
        maskResized.recycle()

        // === Stage: Add Noise ===
        listener?.onStageStart("add_noise")
        val addNoiseStart = System.nanoTime()
        sched.setTimesteps(config.steps, config.strength)
        val actualSteps = sched.getActualSteps()
        val noisyLatent = sched.addNoise(imageLatent, SEED)
        var currentLatent = noisyLatent
        val addNoiseMs = nsToMs(System.nanoTime() - addNoiseStart)

        // === Stage: UNet Denoising Loop (9ch) ===
        listener?.onStageStart("unet_loop")
        val unetLoopStart = System.nanoTime()

        // Prepare reusable buffers
        val textEmbShape = longArrayOf(1, 77, 768) // SD v1.5 CLIP output
        val textEmbBuffer = FloatBuffer.wrap(textEmbeddings)
        val planeSize = latentSize * latentSize
        val concat9ch = FloatArray(9 * planeSize)
        val latentBuffer = FloatBuffer.wrap(concat9ch)
        val latentShape = longArrayOf(1, 9, latentSize.toLong(), latentSize.toLong())
        val timestepShape = longArrayOf(1)

        // Detect timestep input type (precompiled models may use int32 instead of float)
        val unetNames = unetRunner.inputNames
        val sampleName = unetNames.getOrElse(0) { "sample" }
        val timestepName = unetNames.getOrElse(1) { "timestep" }
        val condName = unetNames.getOrElse(2) { "encoder_hidden_states" }
        val timestepType = unetRunner.inputTypes[timestepName] ?: OnnxJavaType.FLOAT
        Log.i(TAG, "UNet timestep input type: $timestepType")

        // Allocate typed timestep buffer
        val timestepFloatBuf = if (timestepType == OnnxJavaType.FLOAT) FloatBuffer.allocate(1) else null
        val timestepIntBuf = if (timestepType == OnnxJavaType.INT32) IntBuffer.allocate(1) else null

        for (step in 0 until actualSteps) {
            listener?.onUnetStep(step, actualSteps)
            val stepStart = System.nanoTime()

            // Scale input
            val scaledLatent = sched.scaleModelInput(currentLatent)

            // === 9ch concat: noisy_latent(4) + masked_image_latent(4) + mask(1) ===
            System.arraycopy(scaledLatent, 0, concat9ch, 0, 4 * planeSize)
            System.arraycopy(maskedImageLatent, 0, concat9ch, 4 * planeSize, 4 * planeSize)
            // Copy mask latent (1ch)
            maskLatentBuffer.rewind()
            maskLatentBuffer.get(concat9ch, 8 * planeSize, planeSize)

            latentBuffer.rewind()
            textEmbBuffer.rewind()

            // Set timestep in the correct type
            val timestepValue = sched.getCurrentTimestep()
            val timestepPair: Pair<Any, LongArray> = when {
                timestepIntBuf != null -> {
                    timestepIntBuf.put(0, timestepValue.toInt())
                    Pair(timestepIntBuf, timestepShape)
                }
                else -> {
                    timestepFloatBuf!!.put(0, timestepValue.toFloat())
                    Pair(timestepFloatBuf, timestepShape)
                }
            }

            val unetInputs = mapOf(
                sampleName to Pair(latentBuffer as Any, latentShape),
                timestepName to timestepPair,
                condName to Pair(textEmbBuffer as Any, textEmbShape)
            )

            val unetResult = unetRunner.runMixed(unetInputs) ?: run { Log.e(TAG, "inpaint: unet.run returned null at step $step"); return null }
            val modelOutput = unetResult.outputs.values.first() as FloatArray

            // Scheduler step
            val schedStart = System.nanoTime()
            currentLatent = sched.step(currentLatent, modelOutput)
            val schedulerStepMs = nsToMs(System.nanoTime() - schedStart)

            val stepTotalMs = nsToMs(System.nanoTime() - stepStart)

            stepDetails.add(StepDetail(
                stepIndex = step,
                inputCreateMs = unetResult.inputCreateMs,
                sessionRunMs = unetResult.sessionRunMs,
                outputCopyMs = unetResult.outputCopyMs,
                schedulerStepMs = schedulerStepMs,
                stepTotalMs = stepTotalMs
            ))
        }
        val unetTotalMs = nsToMs(System.nanoTime() - unetLoopStart)

        // === Stage: VAE Decode ===
        listener?.onStageStart("vae_decode")
        val vaeDecStart = System.nanoTime()
        // Note: VAE Decoder ONNX wrapper already includes / scaling_factor internally.
        // Do NOT divide again here.
        val finalLatent = currentLatent
        val decResult = vaeDec.run(
            FloatBuffer.wrap(finalLatent),
            longArrayOf(1, LATENT_CHANNELS.toLong(), latentSize.toLong(), latentSize.toLong())
        ) ?: run { Log.e(TAG, "inpaint: vaeDecoder.run returned null"); return null }
        val outputImage = ImagePreprocessor.postprocess(
            decResult.outputs.values.first() as FloatArray,
            resolution, resolution
        )
        val vaeDecMs = nsToMs(System.nanoTime() - vaeDecStart)

        val schedulerOverheadMs = stepDetails.sumOf { it.schedulerStepMs.toDouble() }.toFloat()

        val stageTiming = StageTiming(
            tokenizeMs = tokenizeMs,
            textEncMs = textEncMs,
            vaeEncMs = vaeEncMs,
            maskedImgPrepMs = maskedImgPrepMs,
            addNoiseMs = addNoiseMs,
            unetTotalMs = unetTotalMs,
            vaeDecMs = vaeDecMs,
            schedulerOverheadMs = schedulerOverheadMs
        )

        Log.i(TAG, "Inpainting complete: E2E=${stageTiming.inpaintE2eMs}ms, " +
                "UNet=${unetTotalMs}ms (${actualSteps} steps)")

        return InpaintResult(
            outputImage = outputImage,
            stageTiming = stageTiming,
            stepDetails = stepDetails,
            actualSteps = actualSteps
        )
    }

    /**
     * Run warmup inpainting (results discarded).
     */
    fun warmup(roiBitmap: Bitmap, roiMask: Bitmap, count: Int = 2) {
        Log.i(TAG, "Running $count warmup inpaintings...")
        for (i in 0 until count) {
            inpaint(roiBitmap, roiMask)
            Log.i(TAG, "Warmup ${i + 1}/$count complete")
        }
    }

    fun getUnetProfilingSummary(): OrtRunner.ProfilingSummary? {
        return unet?.analyzeProfilingData()
    }

    fun release() {
        vaeEncoder?.release()
        textEncoder?.release()
        unet?.release()
        vaeDecoder?.release()
        vaeEncoder = null
        textEncoder = null
        unet = null
        vaeDecoder = null
        ortEnv?.close()
        ortEnv = null
        tokenizer = null
        scheduler = null
        cachedTextEmbeddings = null
        Log.i(TAG, "Pipeline released")
    }

    /**
     * Load a NumPy .npy file containing a float32 array.
     * Supports simple dense arrays (not structured/object arrays).
     */
    private fun loadNpyFloatArray(path: String): FloatArray? {
        val file = File(path)
        if (!file.exists()) return null

        return try {
            val bytes = file.readBytes()
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

            // NPY magic: \x93NUMPY
            val magic = ByteArray(6)
            buf.get(magic)
            if (magic[0] != 0x93.toByte() || String(magic, 1, 5) != "NUMPY") {
                Log.e(TAG, "Invalid .npy magic: $path")
                return null
            }

            val major = buf.get().toInt()
            val minor = buf.get().toInt()

            // Header length
            val headerLen = if (major >= 2) {
                buf.int
            } else {
                buf.short.toInt() and 0xFFFF
            }

            // Parse header (e.g. "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 77, 768), }")
            val headerBytes = ByteArray(headerLen)
            buf.get(headerBytes)
            val header = String(headerBytes).trim()
            Log.i(TAG, "NPY header: $header")

            // Verify float32
            if (!header.contains("<f4") && !header.contains("float32")) {
                Log.e(TAG, "Expected float32 .npy, got: $header")
                return null
            }

            // Read remaining data as float32
            val dataStart = 6 + 2 + (if (major >= 2) 4 else 2) + headerLen
            val floatCount = (bytes.size - dataStart) / 4
            val result = FloatArray(floatCount)
            val dataBuf = ByteBuffer.wrap(bytes, dataStart, bytes.size - dataStart)
                .order(ByteOrder.LITTLE_ENDIAN)
            dataBuf.asFloatBuffer().get(result)

            Log.i(TAG, "Loaded .npy: $floatCount floats from $path")
            result
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load .npy: ${e.message}", e)
            null
        }
    }

    private fun nsToMs(ns: Long): Float = (ns / 1_000_000.0).toFloat()
}

