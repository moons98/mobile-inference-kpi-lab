package com.example.kpilab

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import android.content.Context
import android.graphics.Bitmap
import android.util.Half
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.ShortBuffer

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
        val vaeDecLoadMs: Long = 0,
        val initWallClockMs: Long = 0,
        val parallelInit: Boolean = false
    ) {
        /** Sum of individual component load times (always same regardless of sequential/parallel) */
        val totalLoadMs: Long get() = textEncLoadMs + unetLoadMs + vaeDecLoadMs
    }

    /** Stage timing for one generation run */
    data class StageTiming(
        val tokenizeMs: Float = 0f,
        val textEncMs: Float = 0f,
        val noiseGenMs: Float = 0f,
        val unetTotalMs: Float = 0f,
        val vaeDecMs: Float = 0f,
        val postprocessMs: Float = 0f,
        val schedulerOverheadMs: Float = 0f,
        val pipelineWallClockMs: Float = 0f
    ) {
        val generateE2eMs: Float
            get() = tokenizeMs + textEncMs + noiseGenMs + unetTotalMs + vaeDecMs + postprocessMs
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
     * @param parallelInit If true, create all 3 sessions concurrently.
     */
    fun initialize(parallelInit: Boolean = false): Boolean {
        val wallClockStart = System.nanoTime()
        Log.i(TAG, "=== Initializing Txt2Img Pipeline (parallel=$parallelInit) ===")
        Log.i(TAG, "Config: $config")

        try {
            tokenizer = Tokenizer(context)
        } catch (e: Exception) {
            lastError = "Tokenizer init failed: ${e.message}"
            Log.e(TAG, "$lastError. Run scripts/deploy/extract_tokenizer_assets.py to generate vocab.json + merges.txt")
            return false
        }
        scheduler = Scheduler(isLcm = config.modelVariant == ModelVariant.LCM_LORA)

        val ep = config.sdBackend
        val fp16 = config.useNpuFp16
        val perf = config.htpPerformanceMode
        val modelDir = config.modelDir

        val enableProfiling = config.phase == BenchmarkPhase.SINGLE_GENERATE

        val env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
        ortEnv = env

        val ok = if (parallelInit) {
            initializeParallel(ep, fp16, perf, modelDir, enableProfiling, env)
        } else {
            initializeSequential(ep, fp16, perf, modelDir, enableProfiling, env)
        }
        if (!ok) {
            // sequential 경로에서 부분 실패 시 this.textEncoder/unet에 이미 할당된
            // OrtRunner가 있을 수 있음. 해제하지 않으면 OrtSession 네이티브 메모리 누수.
            textEncoder?.release(); textEncoder = null
            unet?.release(); unet = null
            vaeDecoder?.release(); vaeDecoder = null
            return false
        }

        val wallClockMs = (System.nanoTime() - wallClockStart) / 1_000_000

        coldStartTiming = ColdStartTiming(
            textEncLoadMs = textEncoder!!.modelLoadMs + textEncoder!!.sessionCreateMs,
            unetLoadMs = unet!!.modelLoadMs + unet!!.sessionCreateMs,
            vaeDecLoadMs = vaeDecoder!!.modelLoadMs + vaeDecoder!!.sessionCreateMs,
            initWallClockMs = wallClockMs,
            parallelInit = parallelInit
        )
        Log.i(TAG, "Cold start: sum=${coldStartTiming!!.totalLoadMs}ms, " +
                "wall-clock=${wallClockMs}ms, parallel=$parallelInit")
        Log.i(TAG, "=== Txt2Img Pipeline Ready ===")
        return true
    }

    private fun initializeSequential(
        ep: ExecutionProvider, fp16: Boolean, perf: String,
        modelDir: String, enableProfiling: Boolean, env: OrtEnvironment
    ): Boolean {
        textEncoder = createTextEncoder(ep, fp16, perf, env) ?: return false

        unet = OrtRunner(context).also {
            val prec = config.sdPrecisionFor(SdComponent.UNET)
            val unetBase = config.modelVariant.unetPrefix
            val path = resolveModelPath(modelDir, SdComponent.UNET, prec, unetBase)
            if (!it.initialize(path, ep, fp16, true, perf, enableProfiling, env)) {
                lastError = "UNet init failed: ${it.lastError}"
                Log.e(TAG, lastError!!)
                return false
            }
        }

        vaeDecoder = OrtRunner(context).also {
            val prec = config.sdPrecisionFor(SdComponent.VAE_DECODER)
            val path = resolveModelPath(modelDir, SdComponent.VAE_DECODER, prec)
            if (!it.initialize(path, ep, fp16, true, perf, false, env)) {
                lastError = "VAE Decoder init failed: ${it.lastError}"
                Log.e(TAG, lastError!!)
                return false
            }
        }
        return true
    }

    private fun initializeParallel(
        ep: ExecutionProvider, fp16: Boolean, perf: String,
        modelDir: String, enableProfiling: Boolean, env: OrtEnvironment
    ): Boolean {
        // QNN HTP EP is NOT thread-safe for concurrent session creation.
        // Serialize the OrtSession.create() call via mutex while still allowing
        // coroutine scheduling to overlap other work between awaits.
        val sessionCreateMutex = Mutex()

        return runBlocking(Dispatchers.IO) {
            val dTextEnc = async {
                val runner = OrtRunner(context)
                val prec = config.sdPrecisionFor(SdComponent.TEXT_ENCODER)
                val path = resolveModelPath(modelDir, SdComponent.TEXT_ENCODER, prec)
                sessionCreateMutex.withLock {
                    if (runner.initialize(path, ep, fp16, true, perf, false, env)) runner else null
                }
            }
            val dUnet = async {
                val runner = OrtRunner(context)
                val prec = config.sdPrecisionFor(SdComponent.UNET)
                val unetBase = config.modelVariant.unetPrefix
                val path = resolveModelPath(modelDir, SdComponent.UNET, prec, unetBase)
                sessionCreateMutex.withLock {
                    if (runner.initialize(path, ep, fp16, true, perf, enableProfiling, env)) runner else null
                }
            }
            val dVae = async {
                val runner = OrtRunner(context)
                val prec = config.sdPrecisionFor(SdComponent.VAE_DECODER)
                val path = resolveModelPath(modelDir, SdComponent.VAE_DECODER, prec)
                sessionCreateMutex.withLock {
                    if (runner.initialize(path, ep, fp16, true, perf, false, env)) runner else null
                }
            }

            val te = dTextEnc.await()
            val un = dUnet.await()
            val vd = dVae.await()

            if (te == null || un == null || vd == null) {
                lastError = listOfNotNull(
                    if (te == null) "Text Encoder" else null,
                    if (un == null) "UNet" else null,
                    if (vd == null) "VAE Decoder" else null
                ).joinToString(", ") + " init failed"
                Log.e(TAG, lastError!!)
                // 성공한 runner들을 해제하지 않으면 OrtSession 네이티브 메모리 누수 발생
                te?.release()
                un?.release()
                vd?.release()
                false
            } else {
                textEncoder = te
                unet = un
                vaeDecoder = vd
                true
            }
        }
    }

    private fun createTextEncoder(
        ep: ExecutionProvider, fp16: Boolean, perf: String, env: OrtEnvironment
    ): OrtRunner? {
        return OrtRunner(context).also {
            val prec = config.sdPrecisionFor(SdComponent.TEXT_ENCODER)
            val path = resolveModelPath(config.modelDir, SdComponent.TEXT_ENCODER, prec)
            if (!it.initialize(path, ep, fp16, true, perf, false, env)) {
                lastError = "Text Encoder init failed: ${it.lastError}"
                Log.e(TAG, lastError!!)
                return null
            }
        }
    }

    /**
     * Resolve on-device model path: try primary filename first, then alt filenames.
     * Handles W8A8 naming divergence (_int8_qdq.onnx primary vs _w8a8.onnx deploy convention).
     */
    private fun resolveModelPath(
        dir: String,
        component: SdComponent,
        precision: SdPrecision,
        customBase: String = component.baseName
    ): String {
        val primary = "$dir/${component.filename(precision, customBase)}"
        if (File(primary).exists()) return primary
        for (alt in component.altFilenames(precision, customBase)) {
            val altPath = "$dir/$alt"
            if (File(altPath).exists()) {
                Log.i(TAG, "resolveModelPath: primary not found, using alt: $altPath")
                return altPath
            }
        }
        return primary  // let OrtRunner report the missing-file error
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
        val latentShape = longArrayOf(1, LATENT_CHANNELS.toLong(), latentSize.toLong(), latentSize.toLong())

        // Detect UNet input names by content (order varies between standard vs QAI Hub compiled)
        val unetNames = unetRunner.inputNames
        // Name detection covers multiple export conventions:
        //   standard diffusers export:   sample / timestep / encoder_hidden_states / output_0
        //   qai-hub-models W8A16 export: latent / timestep / text_emb / output_latent
        val sampleName = unetNames.firstOrNull { it.contains("sample") || it == "latent" } ?: "sample"
        val timestepName = unetNames.firstOrNull { it.contains("timestep") } ?: "timestep"
        val condName = unetNames.firstOrNull { it.contains("encoder") || it.contains("hidden") || it.contains("text_emb") }
            ?: "encoder_hidden_states"
        val timestepType = unetRunner.inputTypes[timestepName] ?: OnnxJavaType.FLOAT
        // [2] latent input type이 FLOAT16이면 W8A16 binary (activation FP16) → 모든 I/O가 FLOAT16
        val unetUseFp16 = unetRunner.inputTypes[sampleName] == OnnxJavaType.FLOAT16
        // [1] timestep shape를 session 메타데이터에서 직접 읽음 (규칙 추론 제거)
        val timestepShape = unetRunner.inputShapes[timestepName] ?: longArrayOf(1)
        Log.i(TAG, "UNet inputs: sample=$sampleName, timestep=$timestepName($timestepType${timestepShape.contentToString()}), cond=$condName, fp16IO=$unetUseFp16")

        val timestepFloatBuf = if (!unetUseFp16 && timestepType == OnnxJavaType.FLOAT) FloatBuffer.allocate(1) else null
        val timestepIntBuf = if (timestepType == OnnxJavaType.INT32) IntBuffer.allocate(1) else null

        // Pre-convert text embeddings to FP16 once if UNet expects FLOAT16 I/O
        val textEmbFp16: ShortArray? = if (unetUseFp16) ShortArray(textEmbeddings.size) { Half.toHalf(textEmbeddings[it]) } else null
        val uncondEmbFp16: ShortArray? = if (unetUseFp16 && uncondEmbeddings != null) ShortArray(uncondEmbeddings.size) { Half.toHalf(uncondEmbeddings[it]) } else null

        // FP32 buffers for non-FP16 path
        val textEmbBuffer = if (!unetUseFp16) FloatBuffer.wrap(textEmbeddings) else null
        val uncondEmbBuffer = if (!unetUseFp16 && useCfg && uncondEmbeddings != null) FloatBuffer.wrap(uncondEmbeddings) else null

        for (step in 0 until actualSteps) {
            listener?.onUnetStep(step, actualSteps)
            val stepStart = System.nanoTime()

            // Scale input
            val scaledLatent = sched.scaleModelInput(currentLatent)
            val latentFp16: ShortArray? = if (unetUseFp16) ShortArray(scaledLatent.size) { Half.toHalf(scaledLatent[it]) } else null
            val latentBuffer: Any = if (unetUseFp16) ShortBuffer.wrap(latentFp16!!) else FloatBuffer.wrap(scaledLatent)

            // Set timestep
            val timestepValue = sched.getCurrentTimestep()
            var timestepPair: Pair<Any, LongArray> = when {
                unetUseFp16 -> Pair(ShortBuffer.wrap(shortArrayOf(Half.toHalf(timestepValue.toFloat()))), timestepShape)
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

            val uncondEmbAvailable = (unetUseFp16 && uncondEmbFp16 != null) || (!unetUseFp16 && uncondEmbBuffer != null)
            if (useCfg && uncondEmbAvailable) {
                // CFG: run UNet twice (unconditional + conditional)
                // Unconditional pass
                val uncondEmbBuf: Any = if (unetUseFp16) ShortBuffer.wrap(uncondEmbFp16!!) else uncondEmbBuffer!!.also { it.rewind() }
                val uncondInputs = mapOf(
                    sampleName to Pair(latentBuffer, latentShape),
                    timestepName to timestepPair,
                    condName to Pair(uncondEmbBuf, textEmbShape)
                )
                val uncondResult = unetRunner.runMixed(uncondInputs)
                    ?: run { Log.e(TAG, "generate: uncond unet.run returned null at step $step"); return null }
                val uncondOutput = uncondResult.outputs.values.first() as FloatArray

                // Conditional pass
                val condLatentBuf: Any = if (unetUseFp16) ShortBuffer.wrap(latentFp16!!) else FloatBuffer.wrap(scaledLatent)
                val condTextEmbBuf: Any = if (unetUseFp16) ShortBuffer.wrap(textEmbFp16!!) else textEmbBuffer!!.also { it.rewind() }
                // Re-set timestep
                when {
                    unetUseFp16 -> timestepPair = Pair(ShortBuffer.wrap(shortArrayOf(Half.toHalf(timestepValue.toFloat()))), timestepShape)
                    timestepIntBuf != null -> timestepIntBuf.put(0, timestepValue.toInt())
                    else -> timestepFloatBuf!!.put(0, timestepValue.toFloat())
                }
                val condInputs = mapOf(
                    sampleName to Pair(condLatentBuf, latentShape),
                    timestepName to timestepPair,
                    condName to Pair(condTextEmbBuf, textEmbShape)
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
                val singleTextEmbBuf: Any = if (unetUseFp16) ShortBuffer.wrap(textEmbFp16!!) else textEmbBuffer!!.also { it.rewind() }
                val unetInputs = mapOf(
                    sampleName to Pair(latentBuffer, latentShape),
                    timestepName to timestepPair,
                    condName to Pair(singleTextEmbBuf, textEmbShape)
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
        // [3] VAE input precision을 session 메타데이터에서 감지
        val vaeInputName = vaeDec.inputNames.firstOrNull() ?: "latent_sample"
        val vaeUseFp16 = vaeDec.inputTypes[vaeInputName] == OnnxJavaType.FLOAT16
        val vaeLatentShape = longArrayOf(1, LATENT_CHANNELS.toLong(), latentSize.toLong(), latentSize.toLong())
        val vaeInputBuf: Any = if (vaeUseFp16)
            ShortBuffer.wrap(ShortArray(currentLatent.size) { Half.toHalf(currentLatent[it]) })
        else
            FloatBuffer.wrap(currentLatent)
        // output shape으로 HWC vs CHW 감지: qai-hub-models W8A16 = [1,H,W,3] → dim[1] != 3
        val vaeOutShape = vaeDec.outputShapes.values.firstOrNull() ?: longArrayOf(1, 3, 512, 512)
        val vaeIsHwc = vaeOutShape.size >= 4 && vaeOutShape[1] != 3L
        Log.i(TAG, "VAE Decoder: input=$vaeInputName(${vaeDec.inputTypes[vaeInputName]}), fp16IO=$vaeUseFp16, hwc=$vaeIsHwc")
        val decResult = vaeDec.runMixed(
            mapOf(vaeInputName to Pair(vaeInputBuf, vaeLatentShape))
        ) ?: run { Log.e(TAG, "generate: vaeDecoder.run returned null"); return null }
        val vaeDecMs = nsToMs(System.nanoTime() - vaeDecStart)

        // === Stage: Postprocess (float[] → Bitmap) ===
        val postprocessStart = System.nanoTime()
        // HWC (qai-hub-models W8A16) → /Div+/Clip baked in → [0,1]. CHW (standard) → [-1,1].
        val outputImage = ImagePreprocessor.postprocess(
            decResult.outputs.values.first() as FloatArray,
            config.resolution, config.resolution,
            normalized = vaeIsHwc,
            hwc = vaeIsHwc
        )
        val postprocessMs = nsToMs(System.nanoTime() - postprocessStart)

        val schedulerOverheadMs = stepDetails.sumOf { it.schedulerStepMs.toDouble() }.toFloat()
        val pipelineWallClockMs = nsToMs(System.nanoTime() - wallClockStart)

        val stageTiming = StageTiming(
            tokenizeMs = tokenizeMs,
            textEncMs = textEncMs,
            noiseGenMs = noiseGenMs,
            unetTotalMs = unetTotalMs,
            vaeDecMs = vaeDecMs,
            postprocessMs = postprocessMs,
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
        // QNN HTP driver does not support concurrent session teardown — parallel close
        // causes a deadlock. Release sequentially to avoid the hang.
        textEncoder?.release()
        unet?.release()
        vaeDecoder?.release()
        textEncoder = null
        unet = null
        vaeDecoder = null
        // OrtEnvironment.getEnvironment() is a process-level singleton.
        // Closing and recreating it between experiments tears down the QNN HTP backend,
        // causing a SIGSEGV on the second session init. Leave the env alive for the process lifetime.
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
