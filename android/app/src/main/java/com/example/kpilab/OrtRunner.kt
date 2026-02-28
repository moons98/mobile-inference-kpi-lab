package com.example.kpilab

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.util.Log
import java.nio.FloatBuffer
import java.util.*
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

    // Logging (thread-safe: accessed from inference loop + system metrics coroutine)
    private val kpiRecords: MutableList<KpiRecord> = CopyOnWriteArrayList()
    private var sessionId: String = ""
    private var isForeground: Boolean = true

    /**
     * Raw inference result with timing and output data.
     */
    data class InferenceResult(
        val inferenceMs: Float,
        val outputData: FloatArray
    )

    data class KpiRecord(
        val timestamp: Long,
        val eventType: EventType,
        val latencyMs: Float,         // Total E2E = preprocess + inference + postprocess
        val preprocessMs: Float,      // Image preprocessing time
        val inferenceMs: Float,       // Model inference time only
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

    /**
     * Initialize ONNX Runtime session with specified model and execution provider
     */
    fun initialize(
        modelType: OnnxModelType,
        executionProvider: ExecutionProvider,
        useNpuFp16: Boolean = true,
        useContextCache: Boolean = false
    ): Boolean {
        lastError = null
        return try {
            currentModel = modelType
            this.useNpuFp16 = useNpuFp16
            this.useContextCache = useContextCache
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

                    qnnOptions["htp_performance_mode"] = "burst"
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
                    qnnOptionsStr = "backend=HTP;perf=burst;fp16=${if (useNpuFp16) "1" else "0"};cache=${if (useContextCache) "1" else "0"};libs=$libSource"

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

        // Output info
        val outputInfo = session.outputInfo
        if (outputInfo.isNotEmpty()) {
            val firstOutput = outputInfo.entries.first()
            outputName = firstOutput.key
            val tensorInfo = firstOutput.value.info as? TensorInfo
            val outputShape = tensorInfo?.shape ?: longArrayOf()
            outputShapeStr = outputShape.contentToString()
            Log.i(TAG, "Output: name=$outputName, shape=$outputShapeStr, type=${tensorInfo?.type}")
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
     * Preprocess source image for inference (runs every iteration).
     * Letterbox resize to model input size, normalize to [0,1], HWC->CHW.
     * Returns the CHW float array ready for model input.
     */
    private fun preprocessFrame(): FloatArray {
        val model = currentModel!!
        val bitmap = sourceBitmap!!
        val targetW = model.inputWidth   // 640
        val targetH = model.inputHeight  // 640

        // Letterbox resize: scale to fit target while preserving aspect ratio
        val scaleW = targetW.toFloat() / bitmap.width
        val scaleH = targetH.toFloat() / bitmap.height
        letterboxScale = minOf(scaleW, scaleH)

        val newW = (bitmap.width * letterboxScale).toInt()
        val newH = (bitmap.height * letterboxScale).toInt()

        letterboxPadLeft = (targetW - newW) / 2f
        letterboxPadTop = (targetH - newH) / 2f

        // Resize bitmap
        val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true)

        // Create letterboxed bitmap (filled with YOLO letterbox gray = 114)
        val letterboxed = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(letterboxed)
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.drawBitmap(resized, letterboxPadLeft, letterboxPadTop, null)

        // Extract pixels
        val pixels = IntArray(targetW * targetH)
        letterboxed.getPixels(pixels, 0, targetW, 0, 0, targetW, targetH)

        // Convert to CHW float array, normalized to [0.0, 1.0]
        val planeSize = targetH * targetW
        val chw = FloatArray(3 * planeSize)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            chw[i] = ((pixel shr 16) and 0xFF) / 255.0f              // R plane
            chw[planeSize + i] = ((pixel shr 8) and 0xFF) / 255.0f   // G plane
            chw[2 * planeSize + i] = (pixel and 0xFF) / 255.0f       // B plane
        }

        // Cleanup intermediate bitmaps (source bitmap is kept alive)
        // Note: createScaledBitmap returns the SAME object when no scaling is needed,
        // so only recycle if it's a different bitmap than the source.
        if (resized !== bitmap) {
            resized.recycle()
        }
        letterboxed.recycle()

        return chw
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
     * Run warm-up iterations to stabilize performance.
     * Includes full E2E pipeline (preprocess + inference + postprocess).
     */
    fun runWarmUp(iterations: Int = 10) {
        Log.i(TAG, "Running $iterations warm-up iterations (full E2E)...")
        for (i in 0 until iterations) {
            val inputData = preprocessFrame()
            val result = runInferenceInternal(inputData)
            if (result != null) {
                YoloPostProcessor.process(
                    output = result.outputData,
                    originalWidth = originalImageWidth,
                    originalHeight = originalImageHeight,
                    padLeft = letterboxPadLeft,
                    padTop = letterboxPadTop,
                    scale = letterboxScale
                )
            }
        }
        Log.i(TAG, "Warm-up completed")
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
        val inputData = preprocessFrame()
        val preEnd = System.nanoTime()
        val preprocessMs = ((preEnd - preStart) / 1_000_000.0).toFloat()

        // Inference
        val inferenceResult = runInferenceInternal(inputData) ?: return -1f

        // Postprocess
        val postStart = System.nanoTime()
        val detections = YoloPostProcessor.process(
            output = inferenceResult.outputData,
            originalWidth = originalImageWidth,
            originalHeight = originalImageHeight,
            padLeft = letterboxPadLeft,
            padTop = letterboxPadTop,
            scale = letterboxScale
        )
        val postEnd = System.nanoTime()
        val postprocessMs = ((postEnd - postStart) / 1_000_000.0).toFloat()

        val totalMs = preprocessMs + inferenceResult.inferenceMs + postprocessMs

        // Capture first inference time (after any warmup)
        if (firstInferenceMs < 0) {
            firstInferenceMs = totalMs
            Log.i(TAG, "First E2E: total=${totalMs}ms (pre=${preprocessMs}ms, " +
                    "inf=${inferenceResult.inferenceMs}ms, post=${postprocessMs}ms), " +
                    "detections=${detections.size}")
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

    private fun runInferenceInternal(inputData: FloatArray? = null): InferenceResult? {
        val env = ortEnv ?: return null
        val session = ortSession ?: return null
        val data = inputData ?: return null

        return try {
            // Create input tensor from preprocessed image data (not timed)
            val inputTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(data),
                inputShape
            )

            // Run inference (timed separately from tensor creation)
            val inputs = Collections.singletonMap(inputName, inputTensor)
            val startTime = System.nanoTime()
            val results = session.run(inputs)
            val endTime = System.nanoTime()

            // Extract output tensor data before closing
            val outputTensor = results.get(0) as OnnxTensor
            val outputBuffer = outputTensor.floatBuffer
            val outputArray = FloatArray(outputBuffer.remaining())
            outputBuffer.get(outputArray)

            // Clean up
            inputTensor.close()
            results.close()

            InferenceResult(
                inferenceMs = ((endTime - startTime) / 1_000_000.0).toFloat(),
                outputData = outputArray
            )
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
     * End the current logging session
     */
    fun endSession() {
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

        sb.appendLine("# session_id,$sessionId")

        // CSV header
        sb.appendLine("timestamp,event_type,latency_ms,preprocess_ms,inference_ms,postprocess_ms,detection_count,thermal_c,power_mw,memory_mb,is_foreground")

        for (record in kpiRecords) {
            val isInference = record.eventType == EventType.INFERENCE
            val isSystem = record.eventType == EventType.SYSTEM
            val latency = if (isInference) "%.2f".format(record.latencyMs) else ""
            val preprocess = if (isInference) "%.2f".format(record.preprocessMs) else ""
            val inference = if (isInference) "%.2f".format(record.inferenceMs) else ""
            val postprocess = if (isInference) "%.2f".format(record.postprocessMs) else ""
            val detCount = if (isInference) record.detectionCount.toString() else ""
            val thermal = if (isSystem) "%.1f".format(record.thermalC) else ""
            val power = if (isSystem) "%.1f".format(record.powerMw) else ""
            val memory = if (isSystem && record.memoryMb >= 0) record.memoryMb.toString() else ""

            sb.appendLine("${record.timestamp},${record.eventType.name},$latency,$preprocess,$inference,$postprocess,$detCount,$thermal,$power,$memory,${record.isForeground}")
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

        Log.i(TAG, "Resources released")
    }
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
    val outputShape: IntArray,
    val isQuantized: Boolean,  // Whether model uses internal quantization (INT8)
    val precision: String      // FP32, INT8_DYNAMIC, INT8_QDQ
) {
    // YOLOv8n models (640x640, object detection)
    YOLOV8N(
        displayName = "YOLOv8n",
        filename = "yolov8n.onnx",
        inputWidth = 640,
        inputHeight = 640,
        inputChannels = 3,
        outputShape = intArrayOf(1, 84, 8400),
        isQuantized = false,
        precision = "FP32"
    ),
    YOLOV8N_INT8_DYNAMIC(
        displayName = "YOLOv8n INT8 (Dynamic)",
        filename = "yolov8n_int8_dynamic.onnx",
        inputWidth = 640,
        inputHeight = 640,
        inputChannels = 3,
        outputShape = intArrayOf(1, 84, 8400),
        isQuantized = true,  // Dynamic quant: input is FLOAT, quantized internally
        precision = "INT8_DYNAMIC"
    ),
    YOLOV8N_INT8_QDQ(
        displayName = "YOLOv8n INT8 (QDQ)",
        filename = "yolov8n_int8_qdq.onnx",
        inputWidth = 640,
        inputHeight = 640,
        inputChannels = 3,
        outputShape = intArrayOf(1, 84, 8400),
        isQuantized = true,  // QDQ: input is FLOAT, Q node quantizes it
        precision = "INT8_QDQ"
    );

    val inputShape: LongArray
        get() = longArrayOf(1, inputChannels.toLong(), inputHeight.toLong(), inputWidth.toLong())  // NCHW

    /** Filename-safe version of displayName for CSV export filenames */
    val filenameSafeName: String
        get() = displayName.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
}

/**
 * Execution provider options
 */
enum class ExecutionProvider(val displayName: String) {
    QNN_NPU("QNN NPU (HTP)"),
    QNN_GPU("QNN GPU"),
    CPU("CPU")
}

