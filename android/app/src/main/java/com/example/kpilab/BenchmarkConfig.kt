package com.example.kpilab

/**
 * Benchmark configuration for ONNX Runtime
 */
data class BenchmarkConfig(
    // Model selection
    val modelType: OnnxModelType = OnnxModelType.YOLOV8N,

    // Execution provider (NPU/GPU/CPU)
    val executionProvider: ExecutionProvider = ExecutionProvider.QNN_NPU,

    // Benchmark settings
    val frequencyHz: Int = 5,
    val durationMinutes: Int = 5,

    // NPU precision: true = FP16, false = FP32 (only affects FP32 models on NPU)
    val useNpuFp16: Boolean = true,

    // QNN context cache: caches compiled HTP graph for faster subsequent loads
    val useContextCache: Boolean = false,

    // HTP performance mode: "burst", "sustained_high", "balanced", "power_saver"
    val htpPerformanceMode: String = "burst"
) {
    /**
     * Calculate interval between inferences in milliseconds
     */
    val intervalMs: Long
        get() = (1000.0 / frequencyHz).toLong()

    /**
     * Total duration in milliseconds
     */
    val durationMs: Long
        get() = durationMinutes * 60 * 1000L

    /**
     * Generate session ID based on config
     */
    fun generateSessionId(): String {
        val timestamp = System.currentTimeMillis()

        val modelStr = modelType.name.lowercase()

        val epStr = when (executionProvider) {
            ExecutionProvider.QNN_NPU -> "npu"
            ExecutionProvider.QNN_GPU -> "gpu"
            ExecutionProvider.CPU -> "cpu"
        }

        val precStr = if (useNpuFp16) "fp16" else "fp32"
        return "ort_${modelStr}_${epStr}_${precStr}_${frequencyHz}hz_${timestamp}"
    }

    override fun toString(): String {
        val precStr = if (useNpuFp16) "FP16" else "FP32"
        val cacheStr = if (useContextCache) "cached" else "no-cache"
        return "BenchmarkConfig(model=${modelType.displayName}, ep=${executionProvider.displayName}, " +
                "prec=$precStr, cache=$cacheStr, freq=${frequencyHz}Hz, duration=${durationMinutes}min)"
    }
}
