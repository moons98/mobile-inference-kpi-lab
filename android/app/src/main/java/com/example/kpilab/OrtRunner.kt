package com.example.kpilab

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.util.Log
import android.util.JsonReader
import android.util.JsonToken
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileReader
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.util.concurrent.CopyOnWriteArrayList

/**
 * ONNX Runtime-based inference runner with QNN Execution Provider support.
 * Provides detailed logging for debugging NPU/GPU/CPU execution paths.
 */
class OrtRunner(private val context: Context) {

    companion object {
        private const val TAG = "OrtRunner"

        /**
         * Initialize QNN libraries. Should be called once at app startup.
         * This extracts QNN libraries from assets and sets up ADSP_LIBRARY_PATH.
         */
        fun initializeQnnLibraries(context: Context): Boolean {
            val path = QnnLibraryManager.initialize(context)
            return path != null
        }
    }

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var currentModel: OnnxModelType? = null
    private var activeExecutionProvider: String = "CPU"

    // Input/Output info
    private var inputName: String = ""
    private var inputShape: LongArray = longArrayOf()
    private var outputName: String = ""

    // Source image bitmap (kept alive for per-iteration preprocessing)
    private var sourceBitmap: Bitmap? = null

    // Letterbox parameters for coordinate transform in postprocessing
    private var letterboxPadLeft: Float = 0f
    private var letterboxPadTop: Float = 0f
    private var letterboxScale: Float = 1f
    private var originalImageWidth: Int = 0
    private var originalImageHeight: Int = 0

    // Benchmark config info for CSV export
    private var benchmarkFrequencyHz: Int = 5
    private var warmupIterations: Int = 0

    // Cold start timing (ms)
    private var modelLoadMs: Long = 0
    private var sessionCreateMs: Long = 0  // Includes QNN compilation
    private var firstInferenceMs: Float = -1f

    // Additional metadata for CSV export
    private var modelFileSizeKb: Int = 0
    private var inputShapeStr: String = ""
    private var outputShapeStr: String = ""
    private var qnnOptionsStr: String = ""

    // Graph partitioning info (set by BenchmarkRunner after logcat parsing)
    private var ortTotalNodes: Int = 0
    private var ortQnnNodes: Int = 0
    private var ortCpuNodes: Int = 0
    private var ortFallbackOps: List<String> = emptyList()

    // ORT profiling: measures actual NPU compute time vs ORT framework overhead
    private var profilingEnabled: Boolean = false
    private var profilingSummary: ProfilingSummary? = null

    // Pipeline benchmark results
    private var pipelineResults: Map<String, Float>? = null

    data class ProfilingSummary(
        val totalRunUs: Long,       // Total session.run() time from profiling (µs)
        val npuComputeUs: Long,     // QNN EP node execution time (µs)
        val cpuComputeUs: Long,     // CPU EP node execution time (µs)
        val frameworkUs: Long,      // ORT overhead = total - npu - cpu (µs)
        val fenceUs: Long,          // fence_before + fence_after synchronization (µs)
        val iterationCount: Int,    // Number of inference iterations profiled
        val cpuOpCounts: Map<String, Int> = emptyMap()  // CPU op name -> count per iteration
    )

    // Logging (thread-safe: accessed from inference loop + system metrics coroutine)
    private val kpiRecords: MutableList<KpiRecord> = CopyOnWriteArrayList()
    private var sessionId: String = ""
    private var isForeground: Boolean = true

    /**
     * Raw inference result with timing and output data.
     */
    data class InferenceResult(
        val inferenceMs: Float,         // session.run() wall time
        val inputCreateMs: Float,       // OnnxTensor.createTensor() time (JNI + alloc)
        val outputCopyMs: Float,        // Output tensor extraction time (native → JVM copy)
        val outputData: FloatArray,
        val scoresData: FloatArray? = null,   // E2E models: per-class scores [1,8400,80]
        val nmsIndices: LongArray? = null      // E2E+NMS models: selected indices [N,3]
    )

    data class KpiRecord(
        val timestamp: Long,
        val eventType: EventType,
        val latencyMs: Float,         // Total E2E = preprocess + inference + postprocess
        val preprocessMs: Float,      // Image preprocessing time
        val inferenceMs: Float,       // Model inference time only (session.run)
        val inputCreateMs: Float,     // Input tensor creation (JNI boundary)
        val outputCopyMs: Float,      // Output tensor copy (native → JVM)
        val postprocessMs: Float,     // NMS + coordinate transform time
        val detectionCount: Int,      // Number of detections after NMS
        val thermalC: Float,
        val powerMw: Float,
        val memoryMb: Int,
        val isForeground: Boolean
    )

    // Last initialization error detail (surfaced to UI via BenchmarkRunner)
    var lastError: String? = null
        private set

    // Store settings for configureExecutionProvider
    var useNpuFp16: Boolean = true
        private set
    private var useContextCache: Boolean = false
    private var htpPerformanceMode: String = "burst"

    /**
     * Initialize ONNX Runtime session with specified model and execution provider
     */
    fun initialize(
        modelType: OnnxModelType,
        executionProvider: ExecutionProvider,
        useNpuFp16: Boolean = true,
        useContextCache: Boolean = false,
        htpPerformanceMode: String = "burst"
    ): Boolean {
        lastError = null
        return try {
            currentModel = modelType
            this.useNpuFp16 = useNpuFp16
            this.useContextCache = useContextCache
            this.htpPerformanceMode = htpPerformanceMode
            Log.i(TAG, "=== OrtRunner Initialization ===")
            Log.i(TAG, "Model: ${modelType.displayName}")
            Log.i(TAG, "Requested EP: ${executionProvider.displayName}")
            Log.i(TAG, "NPU FP16 precision: $useNpuFp16")
            Log.i(TAG, "Context cache: $useContextCache")

            // Reset cold start timing
            modelLoadMs = 0
            sessionCreateMs = 0
            firstInferenceMs = -1f

            // Create ONNX Runtime environment
            // INFO level captures graph partitioning info without verbose QNN DSP logs
            // Use VERBOSE for debugging QNN issues
            ortEnv = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)
            Log.i(TAG, "OrtEnvironment created")

            // Create session options with execution provider
            val sessionOptions = OrtSession.SessionOptions()
            configureExecutionProvider(sessionOptions, executionProvider)

            // Enable ORT profiling to measure actual NPU compute vs framework overhead
            val profilePrefix = "${context.filesDir.absolutePath}/ort_profile"
            sessionOptions.enableProfiling(profilePrefix)
            profilingEnabled = true
            profilingSummary = null
            Log.i(TAG, "ORT profiling enabled: $profilePrefix")

            // Load model from assets (timed)
            val loadStart = System.currentTimeMillis()
            val modelBytes = loadModelFromAssets(modelType.filename)
            modelLoadMs = System.currentTimeMillis() - loadStart
            modelFileSizeKb = modelBytes.size / 1024
            Log.i(TAG, "Model loaded: ${modelType.filename} ($modelFileSizeKb KB) in ${modelLoadMs}ms")

            // Create session (timed - includes QNN compilation for NPU)
            val sessionStart = System.currentTimeMillis()
            ortSession = ortEnv!!.createSession(modelBytes, sessionOptions)
            sessionCreateMs = System.currentTimeMillis() - sessionStart
            Log.i(TAG, "OrtSession created successfully in ${sessionCreateMs}ms")

            // Get input/output info
            extractIOInfo()

            // Load source image into memory (preprocessing runs per-iteration)
            loadSourceImage()

            // Log session info
            logSessionInfo()

            Log.i(TAG, "=== Initialization Complete ===")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize: ${e.message}", e)
            lastError = e.message
            false
        }
    }

    private fun configureExecutionProvider(
        options: OrtSession.SessionOptions,
        provider: ExecutionProvider
    ) {
        when (provider) {
            ExecutionProvider.QNN_NPU -> {
                try {
                    Log.i(TAG, "=== Configuring QNN Execution Provider (NPU) ===")

                    // QNN EP options for HTP (NPU)
                    val qnnOptions = mutableMapOf<String, String>()

                    // Use custom QNN library paths from QnnLibraryManager
                    val qnnLibPath = QnnLibraryManager.getLibraryPath()
                    if (qnnLibPath != null) {
                        // Custom backend path (Stub library)
                        qnnOptions["backend_path"] = "$qnnLibPath/libQnnHtp.so"
                        // Custom skel library directory for DSP
                        qnnOptions["skel_library_dir"] = qnnLibPath
                        Log.i(TAG, "Using custom QNN libs from: $qnnLibPath")
                        Log.i(TAG, "QNN SDK version: ${QnnLibraryManager.QNN_SDK_VERSION}")
                    } else {
                        // Fallback to bundled/system libraries
                        qnnOptions["backend_path"] = "libQnnHtp.so"
                        Log.i(TAG, "Using bundled/system QNN libs")
                    }

                    qnnOptions["htp_performance_mode"] = htpPerformanceMode
                    qnnOptions["htp_graph_finalization_optimization_mode"] = "3"
                    qnnOptions["enable_htp_fp16_precision"] = if (useNpuFp16) "1" else "0"

                    // Context cache options
                    if (useContextCache) {
                        val model = currentModel
                        if (model != null) {
                            val cacheDir = context.cacheDir
                            val precStr = if (useNpuFp16) "fp16" else "fp32"
                            val cachePath = "${cacheDir.absolutePath}/qnn_${model.filename}_${precStr}.bin"
                            qnnOptions["qnn_context_cache_enable"] = "1"
                            qnnOptions["qnn_context_cache_path"] = cachePath
                            Log.i(TAG, "Context cache enabled: $cachePath")
                        }
                    }

                    // Log options
                    qnnOptions.forEach { (key, value) ->
                        Log.i(TAG, "  $key = $value")
                    }

                    // Store options for CSV export (exclude cache path for brevity)
                    val libSource = if (qnnLibPath != null) "custom" else "bundled"
                    qnnOptionsStr = "backend=HTP;perf=$htpPerformanceMode;fp16=${if (useNpuFp16) "1" else "0"};cache=${if (useContextCache) "1" else "0"};libs=$libSource"

                    options.addQnn(qnnOptions)
                    activeExecutionProvider = "QNN_NPU"
                    Log.i(TAG, "QNN EP (NPU) configured")
                } catch (e: Exception) {
                    Log.e(TAG, "QNN EP failed: ${e.message}")
                    Log.w(TAG, "Falling back to CPU")
                    activeExecutionProvider = "CPU"
                    qnnOptionsStr = "fallback_to_cpu"
                }
            }

            ExecutionProvider.QNN_GPU -> {
                try {
                    Log.i(TAG, "=== Configuring QNN Execution Provider (GPU) ===")

                    val qnnOptions = mutableMapOf<String, String>()

                    // Use custom QNN library path if available
                    val qnnLibPath = QnnLibraryManager.getLibraryPath()
                    if (qnnLibPath != null) {
                        qnnOptions["backend_path"] = "$qnnLibPath/libQnnGpu.so"
                        Log.i(TAG, "Using custom QNN GPU lib from: $qnnLibPath")
                    } else {
                        qnnOptions["backend_path"] = "libQnnGpu.so"
                        Log.i(TAG, "Using bundled/system QNN GPU lib")
                    }

                    qnnOptionsStr = "backend=GPU"

                    options.addQnn(qnnOptions)
                    activeExecutionProvider = "QNN_GPU"
                    Log.i(TAG, "QNN EP (GPU) configured")
                } catch (e: Exception) {
                    Log.e(TAG, "QNN GPU EP failed: ${e.message}")
                    Log.w(TAG, "Falling back to CPU")
                    activeExecutionProvider = "CPU"
                    qnnOptionsStr = "fallback_to_cpu"
                }
            }

            ExecutionProvider.CPU -> {
                Log.i(TAG, "Using CPU execution provider")
                activeExecutionProvider = "CPU"
                qnnOptionsStr = "n/a"
            }
        }

        // Common options
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        options.setIntraOpNumThreads(4)
        Log.i(TAG, "Optimization level: ALL_OPT, Threads: 4")
    }

    private fun loadModelFromAssets(filename: String): ByteArray {
        return context.assets.open(filename).use { it.readBytes() }
    }

    private fun extractIOInfo() {
        val session = ortSession ?: return

        // Input info
        val inputInfo = session.inputInfo
        if (inputInfo.isNotEmpty()) {
            val firstInput = inputInfo.entries.first()
            inputName = firstInput.key
            val tensorInfo = firstInput.value.info as? TensorInfo
            inputShape = tensorInfo?.shape ?: longArrayOf()
            inputShapeStr = inputShape.contentToString()
            Log.i(TAG, "Input: name=$inputName, shape=$inputShapeStr, type=${tensorInfo?.type}")
        }

        // Output info (capture all outputs for multi-output models)
        val outputInfo = session.outputInfo
        if (outputInfo.isNotEmpty()) {
            val firstOutput = outputInfo.entries.first()
            outputName = firstOutput.key
            val shapes = outputInfo.entries.map { (name, info) ->
                val tensorInfo = info.info as? TensorInfo
                val shape = tensorInfo?.shape ?: longArrayOf()
                Log.i(TAG, "Output: name=$name, shape=${shape.contentToString()}, type=${tensorInfo?.type}")
                "$name:${shape.contentToString()}"
            }
            outputShapeStr = shapes.joinToString(" | ")
        }
    }

    /**
     * Load source image from assets into memory.
     * The bitmap is kept alive for per-iteration preprocessing.
     */
    private fun loadSourceImage() {
        sourceBitmap = context.assets.open("sample_image.jpg").use { inputStream ->
            BitmapFactory.decodeStream(inputStream)
        }
        originalImageWidth = sourceBitmap!!.width
        originalImageHeight = sourceBitmap!!.height
        Log.i(TAG, "Source image loaded: ${originalImageWidth}x${originalImageHeight}")
    }

    /**
     * Letterbox resize source image and extract ARGB pixels.
     * Shared by both preprocessFrame() and preprocessFrameUint8().
     * Updates letterboxScale, letterboxPadLeft, letterboxPadTop as side effects.
     */
    private fun letterboxPixels(): IntArray {
        val model = currentModel!!
        val bitmap = sourceBitmap!!
        val targetW = model.inputWidth
        val targetH = model.inputHeight

        val scaleW = targetW.toFloat() / bitmap.width
        val scaleH = targetH.toFloat() / bitmap.height
        letterboxScale = minOf(scaleW, scaleH)

        val newW = (bitmap.width * letterboxScale).toInt()
        val newH = (bitmap.height * letterboxScale).toInt()

        letterboxPadLeft = (targetW - newW) / 2f
        letterboxPadTop = (targetH - newH) / 2f

        val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true)

        val letterboxed = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(letterboxed)
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.drawBitmap(resized, letterboxPadLeft, letterboxPadTop, null)

        val pixels = IntArray(targetW * targetH)
        letterboxed.getPixels(pixels, 0, targetW, 0, 0, targetW, targetH)

        if (resized !== bitmap) {
            resized.recycle()
        }
        letterboxed.recycle()

        return pixels
    }

    /**
     * Preprocess source image for inference (runs every iteration).
     * Letterbox resize to model input size, normalize to [0,1], HWC->CHW.
     * Returns the CHW float array ready for model input.
     */
    private fun preprocessFrame(): FloatArray {
        val pixels = letterboxPixels()
        val planeSize = pixels.size
        val chw = FloatArray(3 * planeSize)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            chw[i] = ((pixel shr 16) and 0xFF) / 255.0f              // R plane
            chw[planeSize + i] = ((pixel shr 8) and 0xFF) / 255.0f   // G plane
            chw[2 * planeSize + i] = (pixel and 0xFF) / 255.0f       // B plane
        }

        return chw
    }

    /**
     * Preprocess source image for E2E models (runs every iteration).
     * Letterbox resize to model input size, pack as uint8 HWC byte array.
     * Normalize and HWC->CHW are baked into the ONNX graph.
     */
    private fun preprocessFrameUint8(): ByteArray {
        val pixels = letterboxPixels()
        val hwc = ByteArray(pixels.size * 3)

        for (i in pixels.indices) {
            val p = pixels[i]
            hwc[i * 3]     = ((p shr 16) and 0xFF).toByte()  // R
            hwc[i * 3 + 1] = ((p shr 8) and 0xFF).toByte()   // G
            hwc[i * 3 + 2] = (p and 0xFF).toByte()            // B
        }

        return hwc
    }

    private fun logSessionInfo() {
        val session = ortSession ?: return

        Log.i(TAG, "=== Session Info ===")
        Log.i(TAG, "Active EP: $activeExecutionProvider")
        Log.i(TAG, "Input count: ${session.inputInfo.size}")
        Log.i(TAG, "Output count: ${session.outputInfo.size}")

        // Log all inputs
        session.inputInfo.forEach { (name, info) ->
            val tensorInfo = info.info as? TensorInfo
            Log.i(TAG, "  Input '$name': type=${tensorInfo?.type}, shape=${tensorInfo?.shape?.contentToString()}")
        }

        // Log all outputs
        session.outputInfo.forEach { (name, info) ->
            val tensorInfo = info.info as? TensorInfo
            Log.i(TAG, "  Output '$name': type=${tensorInfo?.type}, shape=${tensorInfo?.shape?.contentToString()}")
        }
        Log.i(TAG, "====================")
    }

    /**
     * Preprocess: dispatch to float or uint8 path based on model input format.
     */
    private fun preprocess(): Any = when (currentModel!!.inputFormat) {
        InputFormat.FLOAT_NCHW -> preprocessFrame()
        InputFormat.UINT8_NHWC -> preprocessFrameUint8()
    }

    /**
     * Postprocess: dispatch to appropriate postprocessor based on model output format.
     */
    private fun postprocess(result: InferenceResult): List<Detection> =
        when (currentModel!!.outputFormat) {
            OutputFormat.RAW_84x8400 -> YoloPostProcessor.process(
                output = result.outputData,
                originalWidth = originalImageWidth,
                originalHeight = originalImageHeight,
                padLeft = letterboxPadLeft,
                padTop = letterboxPadTop,
                scale = letterboxScale
            )
            OutputFormat.BOXES_SCORES -> YoloPostProcessor.processBoxesScores(
                boxes = result.outputData,
                scores = result.scoresData!!,
                originalWidth = originalImageWidth,
                originalHeight = originalImageHeight,
                padLeft = letterboxPadLeft,
                padTop = letterboxPadTop,
                scale = letterboxScale
            )
            OutputFormat.BOXES_SCORES_NMS -> YoloPostProcessor.processNmsIndices(
                boxes = result.outputData,
                scores = result.scoresData!!,
                nmsIndices = result.nmsIndices!!,
                originalWidth = originalImageWidth,
                originalHeight = originalImageHeight,
                padLeft = letterboxPadLeft,
                padTop = letterboxPadTop,
                scale = letterboxScale
            )
        }

    /**
     * Run warm-up iterations to stabilize performance.
     * Includes full E2E pipeline (preprocess + inference + postprocess).
     */
    fun runWarmUp(iterations: Int = 10) {
        Log.i(TAG, "Running $iterations warm-up iterations (full E2E)...")
        for (i in 0 until iterations) {
            val inputData = preprocess()
            val result = runInferenceInternal(inputData)
            if (result != null) {
                postprocess(result)
            }
        }
        Log.i(TAG, "Warm-up completed")
    }

    /**
     * Run a dedicated profiling phase: limited iterations with ORT profiling enabled,
     * then immediately call endProfiling and parse the (small) trace file.
     *
     * This separates profiling from the main benchmark loop so that:
     * 1. The profiling file stays small (~8 MB for 50 iterations vs ~200 MB for 1500)
     * 2. The main loop runs without profiling overhead, improving P95/P99 accuracy
     *
     * Must be called AFTER warmup and BEFORE the main benchmark loop.
     */
    fun runProfilingPhase(iterations: Int = 50) {
        if (!profilingEnabled) {
            Log.w(TAG, "Profiling not enabled, skipping profiling phase")
            return
        }

        Log.i(TAG, "Running profiling phase ($iterations iterations)...")
        for (i in 0 until iterations) {
            val inputData = preprocess()
            val result = runInferenceInternal(inputData)
            if (result != null) {
                postprocess(result)
            }
        }

        // Collect profiling data now (small file, safe to parse)
        profilingSummary = analyzeProfilingData()
        profilingEnabled = false  // Prevent double-collection in endSession()
        Log.i(TAG, "Profiling phase completed ($iterations iterations)")
    }

    /**
     * Start a new logging session
     */
    fun startSession(sessionId: String) {
        this.sessionId = sessionId
        kpiRecords.clear()
        Log.i(TAG, "Session started: $sessionId")
    }

    /**
     * Run a single E2E inference: preprocess + inference + postprocess.
     * Preprocessing runs every iteration to measure true per-frame E2E cost.
     * Returns total E2E latency in ms.
     */
    fun runInference(): Float {
        currentModel ?: return -1f

        // Preprocess (every iteration)
        val preStart = System.nanoTime()
        val inputData = preprocess()
        val preEnd = System.nanoTime()
        val preprocessMs = ((preEnd - preStart) / 1_000_000.0).toFloat()

        // Inference
        val inferenceResult = runInferenceInternal(inputData) ?: return -1f

        // Postprocess
        val postStart = System.nanoTime()
        val detections = postprocess(inferenceResult)
        val postEnd = System.nanoTime()
        val postprocessMs = ((postEnd - postStart) / 1_000_000.0).toFloat()

        val totalMs = preprocessMs + inferenceResult.inputCreateMs +
                inferenceResult.inferenceMs + inferenceResult.outputCopyMs + postprocessMs

        // Capture first inference time (after any warmup)
        if (firstInferenceMs < 0) {
            firstInferenceMs = totalMs
            Log.i(TAG, "First E2E: total=${totalMs}ms (pre=${preprocessMs}ms, " +
                    "inputCreate=${inferenceResult.inputCreateMs}ms, " +
                    "inf=${inferenceResult.inferenceMs}ms, " +
                    "outputCopy=${inferenceResult.outputCopyMs}ms, " +
                    "post=${postprocessMs}ms), detections=${detections.size}")
            if (detections.isNotEmpty()) {
                detections.take(5).forEach { det ->
                    Log.i(TAG, "  ${det.className} (${det.confidence}): " +
                            "[${det.x1}, ${det.y1}, ${det.x2}, ${det.y2}]")
                }
            }
        }

        kpiRecords.add(
            KpiRecord(
                timestamp = System.currentTimeMillis(),
                eventType = EventType.INFERENCE,
                latencyMs = totalMs,
                preprocessMs = preprocessMs,
                inferenceMs = inferenceResult.inferenceMs,
                inputCreateMs = inferenceResult.inputCreateMs,
                outputCopyMs = inferenceResult.outputCopyMs,
                postprocessMs = postprocessMs,
                detectionCount = detections.size,
                thermalC = 0f,
                powerMw = 0f,
                memoryMb = 0,
                isForeground = isForeground
            )
        )

        return totalMs
    }

    /**
     * Run pipeline benchmark: overlap preprocess(N+1) with inference(N).
     * Uses static image stream to simulate continuous frames.
     * Measures actual pipelined throughput vs sequential baseline.
     *
     * @param iterations Number of frames to process
     * @return Map with "sequential_fps", "pipeline_fps", "speedup" keys
     */
    fun runPipelineBenchmark(iterations: Int = 100): Map<String, Float> {
        currentModel ?: return emptyMap()
        val executor = java.util.concurrent.Executors.newSingleThreadExecutor()

        // --- Phase 1: Sequential baseline ---
        val seqStart = System.nanoTime()
        for (i in 0 until iterations) {
            val input = preprocess()
            val result = runInferenceInternal(input) ?: continue
            postprocess(result)
        }
        val seqMs = (System.nanoTime() - seqStart) / 1_000_000.0

        // --- Phase 2: Pipelined (overlap preprocess with inference) ---
        val pipeStart = System.nanoTime()
        var pendingResult: java.util.concurrent.Future<InferenceResult?>? = null
        for (i in 0 until iterations) {
            // Start preprocessing for current frame
            val input = preprocess()

            // While preprocessing ran, collect previous inference result
            if (pendingResult != null) {
                val prevResult = pendingResult.get()
                if (prevResult != null) postprocess(prevResult)
            }

            // Submit current frame's inference to worker thread
            val capturedInput = input
            pendingResult = executor.submit(java.util.concurrent.Callable {
                runInferenceInternal(capturedInput)
            })
        }
        // Drain last pending result
        if (pendingResult != null) {
            val lastResult = pendingResult.get()
            if (lastResult != null) postprocess(lastResult)
        }
        val pipeMs = (System.nanoTime() - pipeStart) / 1_000_000.0

        executor.shutdown()

        val seqFps = (iterations * 1000.0 / seqMs).toFloat()
        val pipeFps = (iterations * 1000.0 / pipeMs).toFloat()
        val speedup = pipeFps / seqFps

        Log.i(TAG, "=== Pipeline Benchmark ($iterations iterations) ===")
        Log.i(TAG, "  Sequential: %.1f ms total, %.1f FPS".format(seqMs, seqFps))
        Log.i(TAG, "  Pipelined:  %.1f ms total, %.1f FPS".format(pipeMs, pipeFps))
        Log.i(TAG, "  Speedup:    %.2fx".format(speedup))

        val results = mapOf(
            "sequential_fps" to seqFps,
            "pipeline_fps" to pipeFps,
            "speedup" to speedup,
            "sequential_ms" to (seqMs / iterations).toFloat(),
            "pipeline_ms" to (pipeMs / iterations).toFloat()
        )
        pipelineResults = results
        return results
    }

    private fun runInferenceInternal(inputData: Any? = null): InferenceResult? {
        val env = ortEnv ?: return null
        val session = ortSession ?: return null
        val model = currentModel ?: return null
        val data = inputData ?: return null

        return try {
            // Create input tensor (timed: JNI boundary + buffer allocation)
            val inputCreateStart = System.nanoTime()
            val inputTensor = when (model.inputFormat) {
                InputFormat.FLOAT_NCHW -> OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(data as FloatArray), inputShape
                )
                InputFormat.UINT8_NHWC -> OnnxTensor.createTensor(
                    env, ByteBuffer.wrap(data as ByteArray), inputShape, OnnxJavaType.UINT8
                )
            }
            val inputCreateMs = ((System.nanoTime() - inputCreateStart) / 1_000_000.0).toFloat()

            // Run inference (timed: session.run only)
            val inputs = mapOf(inputName to inputTensor)
            val startTime = System.nanoTime()
            val results = session.run(inputs)
            val endTime = System.nanoTime()
            val inferenceMs = ((endTime - startTime) / 1_000_000.0).toFloat()

            // Extract outputs (timed: native → JVM buffer copy)
            val outputCopyStart = System.nanoTime()
            val result = when (model.outputFormat) {
                OutputFormat.RAW_84x8400 -> {
                    val buf = (results.get(0) as OnnxTensor).floatBuffer
                    val arr = FloatArray(buf.remaining())
                    buf.get(arr)
                    InferenceResult(inferenceMs, inputCreateMs, 0f, arr) // outputCopyMs set below
                }
                OutputFormat.BOXES_SCORES -> {
                    val boxesBuf = (results.get(0) as OnnxTensor).floatBuffer
                    val boxesArr = FloatArray(boxesBuf.remaining())
                    boxesBuf.get(boxesArr)

                    val scoresBuf = (results.get(1) as OnnxTensor).floatBuffer
                    val scoresArr = FloatArray(scoresBuf.remaining())
                    scoresBuf.get(scoresArr)

                    InferenceResult(inferenceMs, inputCreateMs, 0f, boxesArr, scoresArr)
                }
                OutputFormat.BOXES_SCORES_NMS -> {
                    val boxesBuf = (results.get(0) as OnnxTensor).floatBuffer
                    val boxesArr = FloatArray(boxesBuf.remaining())
                    boxesBuf.get(boxesArr)

                    val scoresBuf = (results.get(1) as OnnxTensor).floatBuffer
                    val scoresArr = FloatArray(scoresBuf.remaining())
                    scoresBuf.get(scoresArr)

                    val nmsBuf = (results.get(2) as OnnxTensor).longBuffer
                    val nmsArr = LongArray(nmsBuf.remaining())
                    nmsBuf.get(nmsArr)

                    InferenceResult(inferenceMs, inputCreateMs, 0f, boxesArr, scoresArr, nmsArr)
                }
            }
            val outputCopyMs = ((System.nanoTime() - outputCopyStart) / 1_000_000.0).toFloat()

            // Clean up
            inputTensor.close()
            results.close()

            // Return result with actual outputCopyMs
            result.copy(outputCopyMs = outputCopyMs)
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed: ${e.message}", e)
            null
        }
    }

    /**
     * Log system metrics
     */
    fun logSystemMetrics(thermalC: Float, powerMw: Float, memoryMb: Int) {
        kpiRecords.add(
            KpiRecord(
                timestamp = System.currentTimeMillis(),
                eventType = EventType.SYSTEM,
                latencyMs = 0f,
                preprocessMs = 0f,
                inferenceMs = 0f,
                inputCreateMs = 0f,
                outputCopyMs = 0f,
                postprocessMs = 0f,
                detectionCount = 0,
                thermalC = thermalC,
                powerMw = powerMw,
                memoryMb = memoryMb,
                isForeground = isForeground
            )
        )
    }

    /**
     * Update foreground/background state
     */
    fun setForeground(isForeground: Boolean) {
        this.isForeground = isForeground
    }

    /**
     * Set benchmark configuration for CSV export metadata
     */
    fun setBenchmarkConfig(frequencyHz: Int, warmupIters: Int) {
        this.benchmarkFrequencyHz = frequencyHz
        this.warmupIterations = warmupIters
    }

    /**
     * Set graph partitioning info from ORT logcat analysis.
     * Called by BenchmarkRunner after logcat parsing, before CSV export.
     */
    fun setPartitionInfo(totalNodes: Int, qnnNodes: Int, cpuNodes: Int, fallbackOps: List<String>) {
        this.ortTotalNodes = totalNodes
        this.ortQnnNodes = qnnNodes
        this.ortCpuNodes = cpuNodes
        this.ortFallbackOps = fallbackOps
        val coverage = if (totalNodes > 0) "%.1f".format(qnnNodes * 100.0 / totalNodes) else "N/A"
        Log.i(TAG, "Partition info: total=$totalNodes, qnn=$qnnNodes, cpu=$cpuNodes, coverage=$coverage%")
    }

    /**
     * Analyze ORT profiling data to separate NPU compute time from framework overhead.
     * Parses the Chrome Tracing JSON produced by ORT's enableProfiling.
     * Returns null if profiling is not enabled or parsing fails.
     */
    private fun analyzeProfilingData(): ProfilingSummary? {
        if (!profilingEnabled) return null
        val session = ortSession ?: return null

        return try {
            val profilePath = session.endProfiling()
            Log.i(TAG, "Profiling file: $profilePath")

            val file = File(profilePath)
            if (!file.exists()) {
                Log.w(TAG, "Profiling file not found: $profilePath")
                return null
            }

            val fileSizeKb = file.length() / 1024
            Log.i(TAG, "Profiling file size: ${fileSizeKb}KB")

            // Streaming parse — reads one event at a time, never loads full file
            var totalRunUs = 0L
            var npuComputeUs = 0L
            var cpuComputeUs = 0L
            var fenceUs = 0L
            var iterationCount = 0
            val cpuOpCounts = mutableMapOf<String, Int>()

            JsonReader(FileReader(file)).use { reader ->
                reader.beginArray()
                while (reader.hasNext()) {
                    var cat = ""
                    var name = ""
                    var dur = 0L
                    var provider = ""

                    var opName = ""
                    reader.beginObject()
                    while (reader.hasNext()) {
                        when (reader.nextName()) {
                            "cat" -> cat = reader.nextString()
                            "name" -> name = reader.nextString()
                            "dur" -> dur = reader.nextLong()
                            "args" -> {
                                reader.beginObject()
                                while (reader.hasNext()) {
                                    when (reader.nextName()) {
                                        "provider" -> provider = reader.nextString()
                                        "op_name" -> opName = reader.nextString()
                                        else -> reader.skipValue()
                                    }
                                }
                                reader.endObject()
                            }
                            else -> reader.skipValue()
                        }
                    }
                    reader.endObject()

                    when (cat) {
                        "Session" -> {
                            if (name == "model_run") {
                                totalRunUs += dur
                                iterationCount++
                            }
                        }
                        "Node" -> {
                            when {
                                provider.contains("QNN") -> npuComputeUs += dur
                                provider.contains("CPU") -> {
                                    cpuComputeUs += dur
                                    if (opName.isNotEmpty()) {
                                        cpuOpCounts[opName] = (cpuOpCounts[opName] ?: 0) + 1
                                    }
                                }
                            }
                        }
                        "fence_before", "fence_after" -> {
                            fenceUs += dur
                        }
                    }
                }
                reader.endArray()
            }

            val frameworkUs = totalRunUs - npuComputeUs - cpuComputeUs

            // Clean up profiling file
            file.delete()

            if (iterationCount == 0) {
                Log.w(TAG, "No profiling iterations found")
                return null
            }

            val perIterCpuOps = cpuOpCounts.mapValues { it.value / iterationCount }
            val summary = ProfilingSummary(
                totalRunUs = totalRunUs,
                npuComputeUs = npuComputeUs,
                cpuComputeUs = cpuComputeUs,
                frameworkUs = frameworkUs,
                fenceUs = fenceUs,
                iterationCount = iterationCount,
                cpuOpCounts = perIterCpuOps
            )

            val avgTotalMs = totalRunUs / 1000.0 / iterationCount
            val avgNpuMs = npuComputeUs / 1000.0 / iterationCount
            val avgCpuMs = cpuComputeUs / 1000.0 / iterationCount
            val avgFenceMs = fenceUs / 1000.0 / iterationCount
            val avgFrameworkMs = frameworkUs / 1000.0 / iterationCount

            Log.i(TAG, "=== ORT Profiling Summary ($iterationCount iterations) ===")
            Log.i(TAG, "  Avg session.run():    %.2f ms".format(avgTotalMs))
            Log.i(TAG, "  Avg NPU compute:      %.2f ms".format(avgNpuMs))
            Log.i(TAG, "  Avg CPU compute:      %.2f ms".format(avgCpuMs))
            Log.i(TAG, "  Avg fence (sync):     %.2f ms".format(avgFenceMs))
            Log.i(TAG, "  Avg ORT overhead:     %.2f ms".format(avgFrameworkMs))
            Log.i(TAG, "  NPU ratio:            %.1f%%".format(avgNpuMs / avgTotalMs * 100))
            if (cpuOpCounts.isNotEmpty()) {
                // Divide by iterationCount to get unique ops per iteration
                val uniqueOps = cpuOpCounts.mapValues { it.value / iterationCount }
                    .entries.sortedByDescending { it.value }
                Log.i(TAG, "  CPU ops (per iter):   ${uniqueOps.joinToString { "${it.key}(${it.value})" }}")
            }

            summary
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to analyze profiling: ${e.message}", e)
            null
        }
    }

    /**
     * End the current logging session
     */
    fun endSession() {
        // Only analyze profiling if not already collected by runProfilingPhase()
        if (profilingSummary == null) {
            profilingSummary = analyzeProfilingData()
        }
        Log.i(TAG, "Session ended: $sessionId, records: ${kpiRecords.size}")
    }

    /**
     * Export all logged data as CSV string
     */
    fun exportCsv(): String {
        val sb = StringBuilder()

        // Device info header
        val deviceInfo = KpiCollector.getDeviceInfoMap()
        sb.appendLine("# device_manufacturer,${deviceInfo["device_manufacturer"]}")
        sb.appendLine("# device_model,${deviceInfo["device_model"]}")
        sb.appendLine("# soc_manufacturer,${deviceInfo["soc_manufacturer"]}")
        sb.appendLine("# soc_model,${deviceInfo["soc_model"]}")
        sb.appendLine("# android_version,${deviceInfo["android_version"]}")
        sb.appendLine("# api_level,${deviceInfo["api_level"]}")

        // App info
        sb.appendLine("# app_version,${BuildConfig.VERSION_NAME}")
        sb.appendLine("# app_build,${BuildConfig.VERSION_CODE}")

        // Runtime info
        sb.appendLine("# runtime,ONNX Runtime")
        sb.appendLine("# ort_version,1.23.2")  // Matches build.gradle.kts dependency
        sb.appendLine("# execution_provider,$activeExecutionProvider")

        // Model info
        sb.appendLine("# model,${currentModel?.displayName ?: "Unknown"}")
        sb.appendLine("# model_file,${currentModel?.filename ?: "Unknown"}")
        sb.appendLine("# model_size_kb,$modelFileSizeKb")
        sb.appendLine("# precision,${currentModel?.precision ?: "FP32"}")
        sb.appendLine("# input_shape,$inputShapeStr")
        sb.appendLine("# output_shape,$outputShapeStr")

        // QNN options (if applicable)
        sb.appendLine("# qnn_options,$qnnOptionsStr")

        // Benchmark config
        sb.appendLine("# frequency_hz,$benchmarkFrequencyHz")
        sb.appendLine("# warmup_iters,$warmupIterations")

        // Cold start timing
        sb.appendLine("# model_load_ms,$modelLoadMs")
        sb.appendLine("# session_create_ms,$sessionCreateMs")
        val totalColdMs = modelLoadMs + sessionCreateMs + (if (firstInferenceMs >= 0) firstInferenceMs.toLong() else 0)
        sb.appendLine("# first_inference_ms,${if (firstInferenceMs >= 0) "%.2f".format(firstInferenceMs) else "N/A"}")
        sb.appendLine("# total_cold_ms,$totalColdMs")
        sb.appendLine("# preprocess_mode,per_iteration")

        // Graph partitioning info
        sb.appendLine("# ort_total_nodes,$ortTotalNodes")
        sb.appendLine("# ort_qnn_nodes,$ortQnnNodes")
        sb.appendLine("# ort_cpu_nodes,$ortCpuNodes")
        val coverageStr = if (ortTotalNodes > 0) "%.1f".format(ortQnnNodes * 100.0 / ortTotalNodes) else "N/A"
        sb.appendLine("# ort_coverage_percent,$coverageStr")
        sb.appendLine("# ort_fallback_ops,${ortFallbackOps.joinToString(";")}")

        // ORT profiling: NPU compute vs framework overhead
        val prof = profilingSummary
        if (prof != null && prof.iterationCount > 0) {
            sb.appendLine("# profiling_iterations,${prof.iterationCount}")
            sb.appendLine("# profiling_avg_total_ms,${"%.2f".format(prof.totalRunUs / 1000.0 / prof.iterationCount)}")
            sb.appendLine("# profiling_avg_npu_ms,${"%.2f".format(prof.npuComputeUs / 1000.0 / prof.iterationCount)}")
            sb.appendLine("# profiling_avg_cpu_ms,${"%.2f".format(prof.cpuComputeUs / 1000.0 / prof.iterationCount)}")
            sb.appendLine("# profiling_avg_fence_ms,${"%.2f".format(prof.fenceUs / 1000.0 / prof.iterationCount)}")
            sb.appendLine("# profiling_avg_overhead_ms,${"%.2f".format(prof.frameworkUs / 1000.0 / prof.iterationCount)}")
            if (prof.cpuOpCounts.isNotEmpty()) {
                val opsStr = prof.cpuOpCounts.entries.sortedByDescending { it.value }
                    .joinToString(";") { "${it.key}(${it.value})" }
                sb.appendLine("# profiling_cpu_ops,$opsStr")
            }
        }

        // Pipeline benchmark results
        val pipe = pipelineResults
        if (pipe != null) {
            sb.appendLine("# pipeline_sequential_fps,${"%.1f".format(pipe["sequential_fps"])}")
            sb.appendLine("# pipeline_pipelined_fps,${"%.1f".format(pipe["pipeline_fps"])}")
            sb.appendLine("# pipeline_speedup,${"%.2f".format(pipe["speedup"])}")
            sb.appendLine("# pipeline_sequential_ms,${"%.2f".format(pipe["sequential_ms"])}")
            sb.appendLine("# pipeline_pipelined_ms,${"%.2f".format(pipe["pipeline_ms"])}")
        }

        sb.appendLine("# session_id,$sessionId")

        // CSV header
        sb.appendLine("timestamp,event_type,latency_ms,preprocess_ms,inference_ms,input_create_ms,output_copy_ms,postprocess_ms,detection_count,thermal_c,power_mw,memory_mb,is_foreground")

        for (record in kpiRecords) {
            val isInference = record.eventType == EventType.INFERENCE
            val isSystem = record.eventType == EventType.SYSTEM
            val latency = if (isInference) "%.2f".format(record.latencyMs) else ""
            val preprocess = if (isInference) "%.2f".format(record.preprocessMs) else ""
            val inference = if (isInference) "%.2f".format(record.inferenceMs) else ""
            val inputCreate = if (isInference) "%.2f".format(record.inputCreateMs) else ""
            val outputCopy = if (isInference) "%.2f".format(record.outputCopyMs) else ""
            val postprocess = if (isInference) "%.2f".format(record.postprocessMs) else ""
            val detCount = if (isInference) record.detectionCount.toString() else ""
            val thermal = if (isSystem) "%.1f".format(record.thermalC) else ""
            val power = if (isSystem) "%.1f".format(record.powerMw) else ""
            val memory = if (isSystem && record.memoryMb >= 0) record.memoryMb.toString() else ""

            sb.appendLine("${record.timestamp},${record.eventType.name},$latency,$preprocess,$inference,$inputCreate,$outputCopy,$postprocess,$detCount,$thermal,$power,$memory,${record.isForeground}")
        }

        return sb.toString()
    }

    /**
     * Get number of records logged
     */
    fun getRecordCount(): Int = kpiRecords.size

    /**
     * Get device info summary
     */
    fun getDeviceInfo(): String {
        val modelName = currentModel?.displayName ?: "None"
        return "Model: $modelName | EP: $activeExecutionProvider | Runtime: ONNX"
    }

    /**
     * Check if NPU is active
     */
    fun isNpuActive(): Boolean = activeExecutionProvider == "QNN_NPU"

    /**
     * Get active execution provider name
     */
    fun getActiveExecutionProvider(): String = activeExecutionProvider

    /**
     * Release all resources
     */
    fun release() {
        ortSession?.close()
        ortSession = null

        ortEnv?.close()
        ortEnv = null

        sourceBitmap?.recycle()
        sourceBitmap = null
        currentModel = null
        profilingEnabled = false
        profilingSummary = null
        pipelineResults = null

        Log.i(TAG, "Resources released")
    }
}

/**
 * Input tensor format
 */
enum class InputFormat {
    FLOAT_NCHW,   // Original: float32 [1, 3, 640, 640] - CPU normalize + transpose
    UINT8_NHWC    // E2E: uint8 [1, 640, 640, 3] - normalize + transpose baked into graph
}

/**
 * Output tensor format
 */
enum class OutputFormat {
    RAW_84x8400,       // Original: [1, 84, 8400] - needs full CPU postprocessing
    BOXES_SCORES,      // E2E no-NMS: boxes [1,8400,4] + scores [1,8400,80]
    BOXES_SCORES_NMS   // E2E with NMS: boxes + scores + nms_indices [N,3]
}

/**
 * ONNX model types
 */
enum class OnnxModelType(
    val displayName: String,
    val filename: String,
    val inputWidth: Int,
    val inputHeight: Int,
    val inputChannels: Int,
    val isQuantized: Boolean,
    val precision: String,
    val inputFormat: InputFormat = InputFormat.FLOAT_NCHW,
    val outputFormat: OutputFormat = OutputFormat.RAW_84x8400
) {
    // YOLOv8n models (640x640, object detection)
    YOLOV8N(
        displayName = "YOLOv8n",
        filename = "yolov8n.onnx",
        inputWidth = 640,
        inputHeight = 640,
        inputChannels = 3,
        isQuantized = false,
        precision = "FP32"
    ),
    YOLOV8N_INT8_DYNAMIC(
        displayName = "YOLOv8n INT8 (Dynamic)",
        filename = "yolov8n_int8_dynamic.onnx",
        inputWidth = 640,
        inputHeight = 640,
        inputChannels = 3,
        isQuantized = true,
        precision = "INT8_DYNAMIC"
    ),
    YOLOV8N_INT8_QDQ(
        displayName = "YOLOv8n INT8 (QDQ)",
        filename = "yolov8n_int8_qdq.onnx",
        inputWidth = 640,
        inputHeight = 640,
        inputChannels = 3,
        isQuantized = true,
        precision = "INT8_QDQ"
    ),

    // YOLOv8m models (640x640, object detection - medium variant)
    YOLOV8M(
        displayName = "YOLOv8m",
        filename = "yolov8m.onnx",
        inputWidth = 640,
        inputHeight = 640,
        inputChannels = 3,
        isQuantized = false,
        precision = "FP32"
    ),
    YOLOV8M_INT8_QDQ(
        displayName = "YOLOv8m INT8 (QDQ)",
        filename = "yolov8m_int8_qdq.onnx",
        inputWidth = 640,
        inputHeight = 640,
        inputChannels = 3,
        isQuantized = true,
        precision = "INT8_QDQ"
    ),

    // E2E models: pre/post processing baked into ONNX graph
    YOLOV8N_PRE(
        displayName = "YOLOv8n +Pre",
        filename = "yolov8n_pre.onnx",
        inputWidth = 640,
        inputHeight = 640,
        inputChannels = 3,
        isQuantized = false,
        precision = "FP32",
        inputFormat = InputFormat.UINT8_NHWC,
        outputFormat = OutputFormat.RAW_84x8400
    ),
    YOLOV8N_E2E(
        displayName = "YOLOv8n +Pre+Post",
        filename = "yolov8n_e2e.onnx",
        inputWidth = 640,
        inputHeight = 640,
        inputChannels = 3,
        isQuantized = false,
        precision = "FP32",
        inputFormat = InputFormat.UINT8_NHWC,
        outputFormat = OutputFormat.BOXES_SCORES
    ),
    YOLOV8N_E2E_NMS(
        displayName = "YOLOv8n E2E",
        filename = "yolov8n_e2e_nms.onnx",
        inputWidth = 640,
        inputHeight = 640,
        inputChannels = 3,
        isQuantized = false,
        precision = "FP32",
        inputFormat = InputFormat.UINT8_NHWC,
        outputFormat = OutputFormat.BOXES_SCORES_NMS
    );

    val inputShape: LongArray
        get() = when (inputFormat) {
            InputFormat.FLOAT_NCHW -> longArrayOf(1, inputChannels.toLong(), inputHeight.toLong(), inputWidth.toLong())
            InputFormat.UINT8_NHWC -> longArrayOf(1, inputHeight.toLong(), inputWidth.toLong(), inputChannels.toLong())
        }
}

/**
 * Execution provider options
 */
enum class ExecutionProvider(val displayName: String) {
    QNN_NPU("QNN NPU (HTP)"),
    QNN_GPU("QNN GPU"),
    CPU("CPU")
}

