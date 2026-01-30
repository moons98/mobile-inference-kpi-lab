package com.example.kpilab

/**
 * Benchmark configuration for ONNX Runtime
 */
data class BenchmarkConfig(
    // Model selection
    val modelType: OnnxModelType = OnnxModelType.MOBILENET_V2,

    // Execution provider (NPU/GPU/CPU)
    val executionProvider: ExecutionProvider = ExecutionProvider.QNN_NPU,

    // Benchmark settings
    val frequencyHz: Int = 5,
    val warmUpEnabled: Boolean = false,
    val durationMinutes: Int = 5
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

        val modelStr = when (modelType) {
            OnnxModelType.MOBILENET_V2 -> "mnv2"
            OnnxModelType.MOBILENET_V2_QUANTIZED -> "mnv2_q"
        }

        val epStr = when (executionProvider) {
            ExecutionProvider.QNN_NPU -> "npu"
            ExecutionProvider.QNN_GPU -> "gpu"
            ExecutionProvider.CPU -> "cpu"
        }

        val warmStr = if (warmUpEnabled) "w" else "nw"
        return "ort_${modelStr}_${epStr}_${frequencyHz}hz_${warmStr}_${timestamp}"
    }

    override fun toString(): String {
        return "BenchmarkConfig(model=${modelType.displayName}, ep=${executionProvider.displayName}, " +
                "freq=${frequencyHz}Hz, warmUp=$warmUpEnabled, duration=${durationMinutes}min)"
    }
}
