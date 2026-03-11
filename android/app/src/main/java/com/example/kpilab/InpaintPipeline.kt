package com.example.kpilab

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.os.Debug
import android.util.Log
import java.nio.FloatBuffer

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
        val cache = config.useContextCache
        val perf = config.htpPerformanceMode
        val modelDir = config.modelDir

        val enableProfiling = config.phase == BenchmarkPhase.SINGLE_ERASE

        val env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
        ortEnv = env

        // VAE Encoder
        vaeEncoder = OrtRunner(context).also {
            val prec = config.sdPrecisionFor(SdComponent.VAE_ENCODER)
            val path = "$modelDir/${SdComponent.VAE_ENCODER.filename(prec)}"
            if (!it.initialize(path, ep, fp16, cache, perf, false, env)) {
                Log.e(TAG, "VAE Encoder init failed: ${it.lastError}")
                return false
            }
        }

        // Text Encoder
        textEncoder = OrtRunner(context).also {
            val prec = config.sdPrecisionFor(SdComponent.TEXT_ENCODER)
            val path = "$modelDir/${SdComponent.TEXT_ENCODER.filename(prec)}"
            if (!it.initialize(path, ep, fp16, cache, perf, false, env)) {
                Log.e(TAG, "Text Encoder init failed: ${it.lastError}")
                return false
            }
        }

        // Inpainting UNet (9ch)
        unet = OrtRunner(context).also {
            val prec = config.sdPrecisionFor(SdComponent.INPAINT_UNET)
            val path = "$modelDir/${SdComponent.INPAINT_UNET.filename(prec)}"
            if (!it.initialize(path, ep, fp16, cache, perf, enableProfiling, env)) {
                Log.e(TAG, "Inpainting UNet init failed: ${it.lastError}")
                return false
            }
        }

        // VAE Decoder
        vaeDecoder = OrtRunner(context).also {
            val prec = config.sdPrecisionFor(SdComponent.VAE_DECODER)
            val path = "$modelDir/${SdComponent.VAE_DECODER.filename(prec)}"
            if (!it.initialize(path, ep, fp16, cache, perf, false, env)) {
                Log.e(TAG, "VAE Decoder init failed: ${it.lastError}")
                return false
            }
        }

        coldStartTiming = ColdStartTiming(
            vaeEncLoadMs = vaeEncoder!!.modelLoadMs + vaeEncoder!!.sessionCreateMs,
            textEncLoadMs = textEncoder!!.modelLoadMs + textEncoder!!.sessionCreateMs,
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
        val vaeEnc = vaeEncoder ?: return null
        val textEnc = textEncoder ?: return null
        val unetRunner = unet ?: return null
        val vaeDec = vaeDecoder ?: return null
        val tok = tokenizer ?: return null
        val sched = scheduler ?: return null

        val resolution = config.resolution
        val latentSize = config.latentSize
        val stepDetails = mutableListOf<StepDetail>()

        // === Stage: Tokenize (怨좎젙 prompt) ===
        listener?.onStageStart("tokenize")
        val tokStart = System.nanoTime()
        val (tokenIds, tokenShape) = tok.tokenize(config.prompt)
        val tokenizeMs = nsToMs(System.nanoTime() - tokStart)

        // === Stage: Text Encode ===
        listener?.onStageStart("text_encode")
        val textEncStart = System.nanoTime()
        val textEncInputName = textEnc.inputNames.firstOrNull() ?: "input_ids"
        val textEncResult = textEnc.runMixed(
            mapOf(textEncInputName to Pair(tokenIds as Any, tokenShape))
        ) ?: return null
        val textEmbeddings = textEncResult.outputs.values.first() as FloatArray
        val textEncMs = nsToMs(System.nanoTime() - textEncStart)

        // === Stage: VAE Encode (image_latent) ===
        listener?.onStageStart("vae_encode")
        val vaeEncStart = System.nanoTime()
        val (imageInput, imageShape) = ImagePreprocessor.preprocess(roiBitmap, resolution)
        val vaeEncResult = vaeEnc.run(imageInput, imageShape) ?: return null
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
        val maskedEncResult = vaeEnc.run(maskedInput, maskedShape) ?: return null
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
        val timestepBuffer = FloatBuffer.allocate(1)
        val timestepShape = longArrayOf(1)

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
            timestepBuffer.put(0, sched.getCurrentTimestep().toFloat())
            textEmbBuffer.rewind()

            // UNet input names
            val unetNames = unetRunner.inputNames
            val sampleName = unetNames.getOrElse(0) { "sample" }
            val timestepName = unetNames.getOrElse(1) { "timestep" }
            val condName = unetNames.getOrElse(2) { "encoder_hidden_states" }

            val unetInputs = mapOf(
                sampleName to Pair(latentBuffer, latentShape),
                timestepName to Pair(timestepBuffer, timestepShape),
                condName to Pair(textEmbBuffer, textEmbShape)
            )

            val unetResult = unetRunner.run(unetInputs) ?: return null
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
        ) ?: return null
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

    /**
     * Build QNN context cache for all 4 SD models sequentially.
     * Loads one model at a time → compiles → saves .bin cache → releases → next.
     * This avoids LMK kill from simultaneous compilation memory pressure.
     *
     * @param onProgress callback with (componentName, index, total, status)
     * @return map of component name → cache file path (or error message)
     */
    fun buildContextCache(
        onProgress: (String, Int, Int, String) -> Unit = { _, _, _, _ -> }
    ): Map<String, String> {
        val ep = config.sdBackend
        val fp16 = config.useNpuFp16
        val perf = config.htpPerformanceMode
        val modelDir = config.modelDir
        // Build smallest models first to maximize chance of success
        val components = SdComponent.values().sortedBy { component ->
            val prec = config.sdPrecisionFor(component)
            val file = java.io.File("$modelDir/${component.filename(prec)}")
            if (file.exists()) file.length() else Long.MAX_VALUE
        }
        val results = mutableMapOf<String, String>()

        for ((index, component) in components.withIndex()) {
            val name = component.displayName
            val prec = config.sdPrecisionFor(component)
            val cacheFile = getCacheFile(component, prec, fp16)
            if (cacheFile.exists() && cacheFile.length() > 0L) {
                results[name] = "OK (cached)"
                onProgress(name, index + 1, components.size, "skip")
                Log.i(TAG, "=== Cache skip [${index + 1}/${components.size}]: $name (${cacheFile.name}) ===")
                continue
            }

            // Log memory state before compilation
            val modelFile = java.io.File("$modelDir/${component.filename(prec)}")
            val modelSizeMb = if (modelFile.exists()) modelFile.length() / (1024 * 1024) else -1
            logMemoryState("BEFORE $name (model=${modelSizeMb}MB)")

            onProgress(name, index + 1, components.size, "compiling")
            Log.i(TAG, "=== Cache build [${index + 1}/${components.size}]: $name (${modelSizeMb}MB) ===")

            // Each component gets its own OrtEnvironment to fully release native memory
            val env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
            val runner = OrtRunner(context)
            val path = "$modelDir/${component.filename(prec)}"
            val success = runner.initialize(
                modelPath = path,
                executionProvider = ep,
                useNpuFp16 = fp16,
                useContextCache = true,
                htpPerformanceMode = perf,
                enableProfiling = false,
                sharedEnv = env
            )

            if (success) {
                results[name] = "OK (${runner.sessionCreateMs}ms)"
                Log.i(TAG, "$name cache built in ${runner.sessionCreateMs}ms")
            } else {
                results[name] = "FAILED: ${runner.lastError}"
                Log.e(TAG, "$name cache build failed: ${runner.lastError}")
            }

            // Aggressively release native memory before next model
            runner.release()
            env.close()
            System.gc()
            Runtime.getRuntime().gc()
            // Give OS time to reclaim native memory pages
            Thread.sleep(2000)

            logMemoryState("AFTER $name")
            onProgress(name, index + 1, components.size, if (success) "done" else "failed")
        }

        Log.i(TAG, "=== Context cache build complete ===")
        return results
    }

    private fun logMemoryState(label: String) {
        val runtime = Runtime.getRuntime()
        val javaUsedMb = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
        val javaMaxMb = runtime.maxMemory() / (1024 * 1024)
        val nativeMb = Debug.getNativeHeapAllocatedSize() / (1024 * 1024)
        val nativeTotalMb = Debug.getNativeHeapSize() / (1024 * 1024)

        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memInfo)
        val availMb = memInfo.availMem / (1024 * 1024)
        val totalMb = memInfo.totalMem / (1024 * 1024)
        val lowMemory = memInfo.lowMemory

        Log.w(TAG, "=== MEM [$label] Java=${javaUsedMb}/${javaMaxMb}MB | " +
                "Native=${nativeMb}/${nativeTotalMb}MB | " +
                "System=${availMb}/${totalMb}MB avail | lowMem=$lowMemory ===")
    }

    /**
     * Check which QNN context cache files exist.
     * @return map of component name → cached (true/false)
     */
    fun checkContextCache(): Map<String, Boolean> {
        val fp16 = config.useNpuFp16

        return SdComponent.values().associate { component ->
            val prec = config.sdPrecisionFor(component)
            val cacheFile = getCacheFile(component, prec, fp16)
            component.displayName to (cacheFile.exists() && cacheFile.length() > 0L)
        }
    }

    private fun getCacheFile(component: SdComponent, precision: SdPrecision, fp16: Boolean): java.io.File {
        val precStr = if (fp16) "fp16" else "fp32"
        val modelName = java.io.File(component.filename(precision)).nameWithoutExtension
        val cachePath = "${context.cacheDir.absolutePath}/qnn_${modelName}_${precStr}.bin"
        return java.io.File(cachePath)
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
        Log.i(TAG, "Pipeline released")
    }

    private fun nsToMs(ns: Long): Float = (ns / 1_000_000.0).toFloat()
}

