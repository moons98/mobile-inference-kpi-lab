package com.example.kpilab

/**
 * Execution path for inference
 */
enum class ExecutionPath(val value: Int, val displayName: String) {
    NPU_ONLY(0, "NPU-only"),
    NPU_FALLBACK(1, "NPU+FB"),
    GPU_ONLY(2, "GPU-only");

    companion object {
        fun fromValue(value: Int): ExecutionPath =
            entries.find { it.value == value } ?: NPU_FALLBACK
    }
}

/**
 * Benchmark configuration
 */
data class BenchmarkConfig(
    val executionPath: ExecutionPath = ExecutionPath.NPU_FALLBACK,
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
        val pathStr = when (executionPath) {
            ExecutionPath.NPU_ONLY -> "npu"
            ExecutionPath.NPU_FALLBACK -> "fb"
            ExecutionPath.GPU_ONLY -> "gpu"
        }
        val warmStr = if (warmUpEnabled) "w" else "nw"
        return "${pathStr}_${frequencyHz}hz_${warmStr}_${timestamp}"
    }

    override fun toString(): String {
        return "BenchmarkConfig(path=${executionPath.displayName}, freq=${frequencyHz}Hz, " +
                "warmUp=$warmUpEnabled, duration=${durationMinutes}min)"
    }
}
