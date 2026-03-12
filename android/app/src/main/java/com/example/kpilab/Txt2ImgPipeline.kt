package com.example.kpilab

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer
import java.nio.IntBuffer

/**
 * SD v1.5 / LCM-LoRA text-to-image pipeline with 4ch UNet.
 * Sessions: Text Encoder / UNet / VAE Decoder.
 *
 * Pipeline:
 * Prompt → Text Encode → Initial Noise → UNet loop (4ch) → VAE Dec → output image
 *
 * CFG (Classifier-Free Guidance):
 * - SD v1.5 (guidanceScale > 1): runs UNet twice per step (conditional + unconditional)
 * - LCM-LoRA (guidanceScale = 1.0): single UNet pass per step
 */
class Txt2ImgPipeline(
    private val context: Context,
    private val config: BenchmarkConfig
) {
    companion object {
        private const val TAG = "Txt2ImgPipeline"
        private const val LATENT_CHANNELS = 4
        private const val VAE_SCALE_FACTOR = 0.18215f
        private const val SEED = 42L
    }

    private var ortEnv: OrtEnvironment? = null

    // 3 SD sessions
    private var textEncoder: OrtRunner? = null
    private var unet: OrtRunner? = null
    private var vaeDecoder: OrtRunner? = null

    private var tokenizer: Tokenizer? = null
    private var scheduler: Scheduler? = null

    var lastError: String? = null
        private set

    /** Cold start timing per SD component */
    data class ColdStartTiming(
        val textEncLoadMs: Long = 0,
        val unetLoadMs: Long = 0,
        val vaeDecLoadMs: Long = 0
    ) {
        val totalLoadMs: Long get() = textEncLoadMs + unetLoadMs + vaeDecLoadMs
    }

    /** Stage timing for one generation run */
    data class StageTiming(
        val tokenizeMs: Float = 0f,
        val textEncMs: Float = 0f,
        val noiseGenMs: Float = 0f,
        val unetTotalMs: Float = 0f,
        val vaeDecMs: Float = 0f,
        val schedulerOverheadMs: Float = 0f,
        val pipelineWallClockMs: Float = 0f
    ) {
        val generateE2eMs: Float
            get() = tokenizeMs + textEncMs + noiseGenMs + unetTotalMs + vaeDecMs
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

    /** Full generation result */
    data class GenerateResult(
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
     * Initialize 3 SD sessions. Call once at benchmark start.
     */
    fun initialize(): Boolean {
        Log.i(TAG, "=== Initializing Txt2Img Pipeline ===")
        Log.i(TAG, "Config: $config")

        try {
            tokenizer = Tokenizer(context)
        } catch (e: Exception) {
            lastError = "Tokenizer init failed: ${e.message}"
            Log.e(TAG, "$lastError. Run scripts/deploy/extract_tokenizer_assets.py to generate vocab.json + merges.txt")
            return false
        }
        scheduler = Scheduler()

        val ep = config.sdBackend
        val fp16 = config.useNpuFp16
        val perf = config.htpPerformanceMode
        val modelDir = config.modelDir

        val enableProfiling = config.phase == BenchmarkPhase.SINGLE_GENERATE

        val env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
        ortEnv = env

        // Text Encoder
        textEncoder = createTextEncoder(ep, fp16, perf, env) ?: return false

        // UNet (4ch, variant-aware filename)
        unet = OrtRunner(context).also {
            val path = "$modelDir/${config.unetFilename()}"
            if (!it.initialize(path, ep, fp16, true, perf, enableProfiling, env)) {
                lastError = "UNet init failed: ${it.lastError}"
                Log.e(TAG, lastError!!)
                return false
            }
        }

        // VAE Decoder
        vaeDecoder = OrtRunner(context).also {
            val prec = config.sdPrecisionFor(SdComponent.VAE_DECODER)
            val path = "$modelDir/${SdComponent.VAE_DECODER.filename(prec)}"
            if (!it.initialize(path, ep, fp16, true, perf, false, env)) {
                lastError = "VAE Decoder init failed: ${it.lastError}"
                Log.e(TAG, lastError!!)
                return false
            }
        }

        coldStartTiming = ColdStartTiming(
            textEncLoadMs = textEncoder!!.modelLoadMs + textEncoder!!.sessionCreateMs,
            unetLoadMs = unet!!.modelLoadMs + unet!!.sessionCreateMs,
            vaeDecLoadMs = vaeDecoder!!.modelLoadMs + vaeDecoder!!.sessionCreateMs
        )
        Log.i(TAG, "Cold start: ${coldStartTiming!!.totalLoadMs}ms total")
        Log.i(TAG, "=== Txt2Img Pipeline Ready ===")
        return true
    }

    private fun createTextEncoder(
        ep: ExecutionProvider, fp16: Boolean, perf: String, env: OrtEnvironment
    ): OrtRunner? {
        return OrtRunner(context).also {
            val prec = config.sdPrecisionFor(SdComponent.TEXT_ENCODER)
            val path = "${config.modelDir}/${SdComponent.TEXT_ENCODER.filename(prec)}"
            if (!it.initialize(path, ep, fp16, true, perf, false, env)) {
                lastError = "Text Encoder init failed: ${it.lastError}"
                Log.e(TAG, lastError!!)
                return null
            }
        }
    }

    /**
     * Run one full text-to-image generation.
     * @param listener Progress callback
     * @return GenerateResult or null on failure
     */
    fun generate(listener: ProgressListener? = null): GenerateResult? {
        val wallClockStart = System.nanoTime()

        val textEnc = textEncoder ?: run { Log.e(TAG, "generate: textEncoder is null"); return null }
        val unetRunner = unet ?: run { Log.e(TAG, "generate: unet is null"); return null }
        val vaeDec = vaeDecoder ?: run { Log.e(TAG, "generate: vaeDecoder is null"); return null }
        val sched = scheduler ?: run { Log.e(TAG, "generate: scheduler is null"); return null }
        val tok = tokenizer ?: run { Log.e(TAG, "generate: tokenizer is null"); return null }

        val latentSize = config.latentSize
        val stepDetails = mutableListOf<StepDetail>()
        val useCfg = config.guidanceScale > 1.0f

        // === Stage: Tokenize ===
        listener?.onStageStart("tokenize")
        val tokStart = System.nanoTime()
        val (tokenIds, tokenShape) = tok.tokenize(config.prompt)
        val tokenizeMs = nsToMs(System.nanoTime() - tokStart)

        // === Stage: Text Encode ===
        listener?.onStageStart("text_encode")
        val textEncStart = System.nanoTime()
        val textEncInputName = textEnc.inputNames.firstOrNull() ?: "input_ids"

        // Precompiled models (QAI Hub) may expect INT32 instead of INT64
        val inputType = textEnc.inputTypes[textEncInputName] ?: OnnxJavaType.INT64
        val inputBuffer: Any = if (inputType == OnnxJavaType.INT32) {
            longBufferToIntBuffer(tokenIds)
        } else {
            tokenIds
        }
        Log.i(TAG, "Text Encoder input type: $inputType")

        val textEncResult = textEnc.runMixed(
            mapOf(textEncInputName to Pair(inputBuffer, tokenShape))
        ) ?: run { Log.e(TAG, "generate: textEncoder.run returned null"); return null }
        val textEmbeddings = textEncResult.outputs.values.first() as FloatArray

        // Generate unconditional embeddings for CFG
        val uncondEmbeddings: FloatArray? = if (useCfg) {
            val (uncondTokenIds, uncondTokenShape) = tok.tokenize("")
            val uncondInput: Any = if (inputType == OnnxJavaType.INT32) {
                longBufferToIntBuffer(uncondTokenIds)
            } else {
                uncondTokenIds
            }
            val uncondResult = textEnc.runMixed(
                mapOf(textEncInputName to Pair(uncondInput, uncondTokenShape))
            ) ?: run { Log.e(TAG, "generate: uncond textEncoder.run returned null"); return null }
            uncondResult.outputs.values.first() as FloatArray
        } else null

        val textEncMs = nsToMs(System.nanoTime() - textEncStart)

        // === Stage: Generate Initial Noise ===
        listener?.onStageStart("noise_gen")
        val noiseGenStart = System.nanoTime()
        sched.setTimesteps(config.steps)
        val actualSteps = sched.getActualSteps()
        val latentElements = LATENT_CHANNELS * latentSize * latentSize
        var currentLatent = sched.generateInitialNoise(latentElements, SEED)
        val noiseGenMs = nsToMs(System.nanoTime() - noiseGenStart)

        // === Stage: UNet Denoising Loop (4ch) ===
        listener?.onStageStart("unet_loop")
        val unetLoopStart = System.nanoTime()

        // Prepare reusable buffers
        val textEmbShape = longArrayOf(1, 77, 768)
        val textEmbBuffer = FloatBuffer.wrap(textEmbeddings)
        val uncondEmbBuffer = if (useCfg && uncondEmbeddings != null) FloatBuffer.wrap(uncondEmbeddings) else null
        val latentShape = longArrayOf(1, LATENT_CHANNELS.toLong(), latentSize.toLong(), latentSize.toLong())
        val timestepShape = longArrayOf(1)

        // Detect timestep input type
        val unetNames = unetRunner.inputNames
        val sampleName = unetNames.getOrElse(0) { "sample" }
        val timestepName = unetNames.getOrElse(1) { "timestep" }
        val condName = unetNames.getOrElse(2) { "encoder_hidden_states" }
        val timestepType = unetRunner.inputTypes[timestepName] ?: OnnxJavaType.FLOAT
        Log.i(TAG, "UNet timestep input type: $timestepType")

        val timestepFloatBuf = if (timestepType == OnnxJavaType.FLOAT) FloatBuffer.allocate(1) else null
        val timestepIntBuf = if (timestepType == OnnxJavaType.INT32) IntBuffer.allocate(1) else null

        for (step in 0 until actualSteps) {
            listener?.onUnetStep(step, actualSteps)
            val stepStart = System.nanoTime()

            // Scale input
            val scaledLatent = sched.scaleModelInput(currentLatent)
            val latentBuffer = FloatBuffer.wrap(scaledLatent)

            // Set timestep
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

            val modelOutput: FloatArray

            if (useCfg && uncondEmbBuffer != null) {
                // CFG: run UNet twice (unconditional + conditional)
                // Unconditional pass
                uncondEmbBuffer.rewind()
                val uncondInputs = mapOf(
                    sampleName to Pair(latentBuffer as Any, latentShape),
                    timestepName to timestepPair,
                    condName to Pair(uncondEmbBuffer as Any, textEmbShape)
                )
                val uncondResult = unetRunner.runMixed(uncondInputs)
                    ?: run { Log.e(TAG, "generate: uncond unet.run returned null at step $step"); return null }
                val uncondOutput = uncondResult.outputs.values.first() as FloatArray

                // Conditional pass
                val condLatentBuffer = FloatBuffer.wrap(scaledLatent)
                textEmbBuffer.rewind()
                // Re-set timestep (buffer was consumed)
                when {
                    timestepIntBuf != null -> timestepIntBuf.put(0, timestepValue.toInt())
                    else -> timestepFloatBuf!!.put(0, timestepValue.toFloat())
                }
                val condInputs = mapOf(
                    sampleName to Pair(condLatentBuffer as Any, latentShape),
                    timestepName to timestepPair,
                    condName to Pair(textEmbBuffer as Any, textEmbShape)
                )
                val condResult = unetRunner.runMixed(condInputs)
                    ?: run { Log.e(TAG, "generate: cond unet.run returned null at step $step"); return null }
                val condOutput = condResult.outputs.values.first() as FloatArray

                // CFG combine: output = uncond + guidance_scale * (cond - uncond)
                modelOutput = FloatArray(condOutput.size)
                val gs = config.guidanceScale
                for (i in modelOutput.indices) {
                    modelOutput[i] = uncondOutput[i] + gs * (condOutput[i] - uncondOutput[i])
                }
            } else {
                // No CFG: single UNet pass
                textEmbBuffer.rewind()
                val unetInputs = mapOf(
                    sampleName to Pair(latentBuffer as Any, latentShape),
                    timestepName to timestepPair,
                    condName to Pair(textEmbBuffer as Any, textEmbShape)
                )
                val unetResult = unetRunner.runMixed(unetInputs)
                    ?: run { Log.e(TAG, "generate: unet.run returned null at step $step"); return null }
                modelOutput = unetResult.outputs.values.first() as FloatArray
            }

            // Scheduler step
            val schedStart = System.nanoTime()
            currentLatent = sched.step(currentLatent, modelOutput)
            val schedulerStepMs = nsToMs(System.nanoTime() - schedStart)

            val stepTotalMs = nsToMs(System.nanoTime() - stepStart)

            stepDetails.add(StepDetail(
                stepIndex = step,
                inputCreateMs = 0f, // simplified — overhead included in stepTotal
                sessionRunMs = stepTotalMs - schedulerStepMs,
                outputCopyMs = 0f,
                schedulerStepMs = schedulerStepMs,
                stepTotalMs = stepTotalMs
            ))
        }
        val unetTotalMs = nsToMs(System.nanoTime() - unetLoopStart)

        // === Stage: VAE Decode ===
        listener?.onStageStart("vae_decode")
        val vaeDecStart = System.nanoTime()
        val decResult = vaeDec.run(
            FloatBuffer.wrap(currentLatent),
            longArrayOf(1, LATENT_CHANNELS.toLong(), latentSize.toLong(), latentSize.toLong())
        ) ?: run { Log.e(TAG, "generate: vaeDecoder.run returned null"); return null }
        val outputImage = ImagePreprocessor.postprocess(
            decResult.outputs.values.first() as FloatArray,
            config.resolution, config.resolution
        )
        val vaeDecMs = nsToMs(System.nanoTime() - vaeDecStart)

        val schedulerOverheadMs = stepDetails.sumOf { it.schedulerStepMs.toDouble() }.toFloat()
        val pipelineWallClockMs = nsToMs(System.nanoTime() - wallClockStart)

        val stageTiming = StageTiming(
            tokenizeMs = tokenizeMs,
            textEncMs = textEncMs,
            noiseGenMs = noiseGenMs,
            unetTotalMs = unetTotalMs,
            vaeDecMs = vaeDecMs,
            schedulerOverheadMs = schedulerOverheadMs,
            pipelineWallClockMs = pipelineWallClockMs
        )

        val overhead = pipelineWallClockMs - stageTiming.generateE2eMs
        Log.i(TAG, "Generation complete: WallClock=${pipelineWallClockMs}ms, " +
                "ComponentSum=${stageTiming.generateE2eMs}ms, Overhead=${overhead}ms, " +
                "UNet=${unetTotalMs}ms (${actualSteps} steps, CFG=$useCfg)")

        return GenerateResult(
            outputImage = outputImage,
            stageTiming = stageTiming,
            stepDetails = stepDetails,
            actualSteps = actualSteps
        )
    }

    /**
     * Run warmup generations (results discarded).
     * @return Total warmup duration in milliseconds
     */
    fun warmup(count: Int = 2): Long {
        val warmupStart = System.nanoTime()
        Log.i(TAG, "Running $count warmup generations...")
        for (i in 0 until count) {
            generate()?.outputImage?.recycle()
            Log.i(TAG, "Warmup ${i + 1}/$count complete")
        }
        val warmupMs = (System.nanoTime() - warmupStart) / 1_000_000
        Log.i(TAG, "Warmup total: ${warmupMs}ms")
        return warmupMs
    }

    fun getUnetProfilingSummary(): OrtRunner.ProfilingSummary? {
        return unet?.analyzeProfilingData()
    }

    fun release() {
        textEncoder?.release()
        unet?.release()
        vaeDecoder?.release()
        textEncoder = null
        unet = null
        vaeDecoder = null
        ortEnv?.close()
        ortEnv = null
        tokenizer = null
        scheduler = null
        Log.i(TAG, "Pipeline released")
    }

    /** Convert LongBuffer (INT64) to IntBuffer (INT32) for precompiled models. */
    private fun longBufferToIntBuffer(longBuf: java.nio.LongBuffer): IntBuffer {
        val intBuf = IntBuffer.allocate(longBuf.capacity())
        longBuf.rewind()
        for (i in 0 until longBuf.capacity()) {
            intBuf.put(i, longBuf.get(i).toInt())
        }
        return intBuf
    }

    private fun nsToMs(ns: Long): Float = (ns / 1_000_000.0).toFloat()
}
