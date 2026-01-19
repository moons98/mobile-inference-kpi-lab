package com.example.kpilab

/**
 * Benchmark configuration
 */
data class BenchmarkConfig(
    val modelType: ModelType = ModelType.MOBILENET_V2,
    val delegateMode: DelegateMode = DelegateMode.NPU_GPU_CPU,
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
            ModelType.MOBILENET_V2 -> "mnv2"
            ModelType.MOBILENET_V2_QUANTIZED -> "mnv2_q"
            ModelType.YOLOV8N -> "yolo"
            ModelType.YOLOV8N_QUANTIZED -> "yolo_q"
        }
        val modeStr = when (delegateMode) {
            DelegateMode.NPU_GPU_CPU -> "npu_gpu"
            DelegateMode.GPU_CPU -> "gpu"
            DelegateMode.CPU_ONLY -> "cpu"
        }
        val warmStr = if (warmUpEnabled) "w" else "nw"
        return "${modelStr}_${modeStr}_${frequencyHz}hz_${warmStr}_${timestamp}"
    }

    override fun toString(): String {
        return "BenchmarkConfig(model=${modelType.displayName}, mode=${delegateMode.displayName}, " +
                "freq=${frequencyHz}Hz, warmUp=$warmUpEnabled, duration=${durationMinutes}min)"
    }
}
