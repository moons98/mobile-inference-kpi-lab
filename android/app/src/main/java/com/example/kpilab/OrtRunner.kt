package com.example.kpilab

import ai.onnxruntime.*
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer
import java.util.*

/**
 * ONNX Runtime-based inference runner with QNN Execution Provider support.
 * Provides detailed logging for debugging NPU/GPU/CPU execution paths.
 */
class OrtRunner(private val context: Context) {

    companion object {
        private const val TAG = "OrtRunner"
    }

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var currentModel: OnnxModelType? = null
    private var activeExecutionProvider: String = "CPU"

    // Input/Output info
    private var inputName: String = ""
    private var inputShape: LongArray = longArrayOf()
    private var outputName: String = ""

    // Pre-allocated input tensor data (all models use FLOAT input)
    private var inputFloatData: FloatArray? = null

    // Benchmark config info for CSV export
    private var benchmarkFrequencyHz: Int = 5
    private var warmupIterations: Int = 0

    // Logging
    private val kpiRecords = mutableListOf<KpiRecord>()
    private var sessionId: String = ""
    private var isForeground: Boolean = true

    data class KpiRecord(
        val timestamp: Long,
        val eventType: Int, // 0: INFERENCE, 1: SYSTEM
        val latencyMs: Float,
        val thermalC: Float,
        val powerMw: Float,
        val memoryMb: Int,
        val isForeground: Boolean
    )

    // Store settings for configureExecutionProvider
    private var useNpuFp16: Boolean = true
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
        return try {
            currentModel = modelType
            this.useNpuFp16 = useNpuFp16
            this.useContextCache = useContextCache
            Log.i(TAG, "=== OrtRunner Initialization ===")
            Log.i(TAG, "Model: ${modelType.displayName}")
            Log.i(TAG, "Requested EP: ${executionProvider.displayName}")
            Log.i(TAG, "NPU FP16 precision: $useNpuFp16")
            Log.i(TAG, "Context cache: $useContextCache")

            // Create ONNX Runtime environment
            ortEnv = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING)
            Log.i(TAG, "OrtEnvironment created")

            // Create session options with execution provider
            val sessionOptions = OrtSession.SessionOptions()
            configureExecutionProvider(sessionOptions, executionProvider)

            // Load model from assets
            val modelBytes = loadModelFromAssets(modelType.filename)
            Log.i(TAG, "Model loaded: ${modelType.filename} (${modelBytes.size / 1024} KB)")

            // Create session
            ortSession = ortEnv!!.createSession(modelBytes, sessionOptions)
            Log.i(TAG, "OrtSession created successfully")

            // Get input/output info
            extractIOInfo()

            // Allocate input data
            allocateInputData()

            // Log session info
            logSessionInfo()

            Log.i(TAG, "=== Initialization Complete ===")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize: ${e.message}", e)
            e.printStackTrace()
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
                    qnnOptions["backend_path"] = "libQnnHtp.so"
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

                    options.addQnn(qnnOptions)
                    activeExecutionProvider = "QNN_NPU"
                    Log.i(TAG, "QNN EP (NPU) configured")
                } catch (e: Exception) {
                    Log.e(TAG, "QNN EP failed: ${e.message}")
                    Log.w(TAG, "Falling back to CPU")
                    activeExecutionProvider = "CPU"
                }
            }

            ExecutionProvider.QNN_GPU -> {
                try {
                    Log.i(TAG, "=== Configuring QNN Execution Provider (GPU) ===")

                    val qnnOptions = mutableMapOf<String, String>()
                    qnnOptions["backend_path"] = "libQnnGpu.so"

                    options.addQnn(qnnOptions)
                    activeExecutionProvider = "QNN_GPU"
                    Log.i(TAG, "QNN EP (GPU) configured")
                } catch (e: Exception) {
                    Log.e(TAG, "QNN GPU EP failed: ${e.message}")
                    Log.w(TAG, "Falling back to CPU")
                    activeExecutionProvider = "CPU"
                }
            }

            ExecutionProvider.CPU -> {
                Log.i(TAG, "Using CPU execution provider")
                activeExecutionProvider = "CPU"
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
            Log.i(TAG, "Input: name=$inputName, shape=${inputShape.contentToString()}")
        }

        // Output info
        val outputInfo = session.outputInfo
        if (outputInfo.isNotEmpty()) {
            val firstOutput = outputInfo.entries.first()
            outputName = firstOutput.key
            val tensorInfo = firstOutput.value.info as? TensorInfo
            val outputShape = tensorInfo?.shape ?: longArrayOf()
            Log.i(TAG, "Output: name=$outputName, shape=${outputShape.contentToString()}")
        }
    }

    private fun allocateInputData() {
        val model = currentModel ?: return

        // ONNX uses NCHW format
        // All models expect FLOAT input (even quantized models - quantization is internal)
        val totalElements = model.inputChannels * model.inputHeight * model.inputWidth

        inputFloatData = FloatArray(totalElements) {
            (Math.random() * 255).toFloat() / 255.0f
        }

        val quantStr = if (model.isQuantized) " (internally quantized)" else ""
        Log.i(TAG, "Input data allocated: $totalElements FLOAT elements$quantStr")
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
     * Run warm-up iterations to stabilize performance
     */
    fun runWarmUp(iterations: Int = 10) {
        Log.i(TAG, "Running $iterations warm-up iterations...")
        for (i in 0 until iterations) {
            runInferenceInternal()
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
     * Run a single inference and return latency
     */
    fun runInference(): Float {
        val latencyMs = runInferenceInternal()

        if (latencyMs >= 0) {
            kpiRecords.add(
                KpiRecord(
                    timestamp = System.currentTimeMillis(),
                    eventType = 0,
                    latencyMs = latencyMs,
                    thermalC = 0f,
                    powerMw = 0f,
                    memoryMb = 0,
                    isForeground = isForeground
                )
            )
        }

        return latencyMs
    }

    private fun runInferenceInternal(): Float {
        val env = ortEnv ?: return -1f
        val session = ortSession ?: return -1f
        val model = currentModel ?: return -1f

        return try {
            val startTime = System.nanoTime()

            // Create input tensor (all models use FLOAT input)
            val inputTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(inputFloatData!!),
                inputShape
            )

            // Run inference
            val inputs = Collections.singletonMap(inputName, inputTensor)
            val results = session.run(inputs)

            val endTime = System.nanoTime()

            // Clean up
            inputTensor.close()
            results.close()

            ((endTime - startTime) / 1_000_000.0).toFloat()
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed: ${e.message}", e)
            -1f
        }
    }

    /**
     * Log system metrics
     */
    fun logSystemMetrics(thermalC: Float, powerMw: Float, memoryMb: Int) {
        kpiRecords.add(
            KpiRecord(
                timestamp = System.currentTimeMillis(),
                eventType = 1,
                latencyMs = 0f,
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
        sb.appendLine("# ort_version,${OrtEnvironment.getApiBase().getVersionString()}")
        sb.appendLine("# execution_provider,$activeExecutionProvider")

        // Model info
        sb.appendLine("# model,${currentModel?.displayName ?: "Unknown"}")
        sb.appendLine("# model_file,${currentModel?.filename ?: "Unknown"}")
        sb.appendLine("# precision,${currentModel?.precision ?: "FP32"}")

        // Benchmark config
        sb.appendLine("# frequency_hz,$benchmarkFrequencyHz")
        sb.appendLine("# warmup_iters,$warmupIterations")

        sb.appendLine("# session_id,$sessionId")
        sb.appendLine("#")

        // CSV header
        sb.appendLine("timestamp,event_type,latency_ms,thermal_c,power_mw,memory_mb,is_foreground")

        for (record in kpiRecords) {
            val eventType = if (record.eventType == 0) "INFERENCE" else "SYSTEM"
            val latency = if (record.eventType == 0) "%.2f".format(record.latencyMs) else ""
            val thermal = if (record.eventType == 1) "%.1f".format(record.thermalC) else ""
            val power = if (record.eventType == 1) "%.1f".format(record.powerMw) else ""
            // memory: -1 means "not sampled this interval", show as empty
            val memory = if (record.eventType == 1 && record.memoryMb >= 0) record.memoryMb.toString() else ""

            sb.appendLine("${record.timestamp},$eventType,$latency,$thermal,$power,$memory,${record.isForeground}")
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

        inputFloatData = null
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
    // MobileNetV2 models (224x224, ImageNet classification)
    MOBILENET_V2(
        displayName = "MobileNetV2",
        filename = "mobilenetv2.onnx",
        inputWidth = 224,
        inputHeight = 224,
        inputChannels = 3,
        outputShape = intArrayOf(1, 1000),
        isQuantized = false,
        precision = "FP32"
    ),
    MOBILENET_V2_INT8_DYNAMIC(
        displayName = "MobileNetV2 INT8 (Dynamic)",
        filename = "mobilenetv2_int8_dynamic.onnx",
        inputWidth = 224,
        inputHeight = 224,
        inputChannels = 3,
        outputShape = intArrayOf(1, 1000),
        isQuantized = true,  // Dynamic quant: input is FLOAT, quantized internally
        precision = "INT8_DYNAMIC"
    ),
    MOBILENET_V2_INT8_QDQ(
        displayName = "MobileNetV2 INT8 (QDQ)",
        filename = "mobilenetv2_int8_qdq.onnx",
        inputWidth = 224,
        inputHeight = 224,
        inputChannels = 3,
        outputShape = intArrayOf(1, 1000),
        isQuantized = true,  // QDQ: input is FLOAT, Q node quantizes it
        precision = "INT8_QDQ"
    ),
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
}

/**
 * Execution provider options
 */
enum class ExecutionProvider(val displayName: String) {
    QNN_NPU("QNN NPU (HTP)"),
    QNN_GPU("QNN GPU"),
    CPU("CPU")
}

