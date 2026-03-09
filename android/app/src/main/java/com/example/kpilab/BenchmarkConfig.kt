package com.example.kpilab

/**
 * Benchmark configuration for ONNX Runtime
 */
data class BenchmarkConfig(
    // Model selection
    val modelType: OnnxModelType = OnnxModelType.YOLOV8N,

    // Execution provider (NPU/GPU/CPU)
    val executionProvider: ExecutionProvider = ExecutionProvider.QNN_NPU,

    // Benchmark phase (determines execution strategy)
    val phase: BenchmarkPhase = BenchmarkPhase.BURST,

    // Input source mode
    val inputMode: InputMode = InputMode.CAMERA_SINGLE,

    // Demo mode: show camera preview + detection overlay
    val demoMode: Boolean = false,

    // Benchmark settings (overridden by phase defaults when null)
    val frequencyHz: Int = 2,
    val durationMinutes: Int = 5,
    val iterations: Int = 100,

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
     * Total duration in milliseconds.
     * For BURST phase, derived from iterations × interval.
     * For SUSTAINED phase, uses durationMinutes.
     */
    val durationMs: Long
        get() = when (phase) {
            BenchmarkPhase.BURST -> iterations * intervalMs
            BenchmarkPhase.SUSTAINED -> durationMinutes * 60 * 1000L
        }

    /**
     * Whether this config uses camera input
     */
    val usesCamera: Boolean
        get() = inputMode != InputMode.STATIC_IMAGE

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

        val phaseStr = when (phase) {
            BenchmarkPhase.BURST -> "burst"
            BenchmarkPhase.SUSTAINED -> "sustained"
        }

        val precStr = if (useNpuFp16) "fp16" else "fp32"
        return "ort_${modelStr}_${epStr}_${precStr}_${phaseStr}_${timestamp}"
    }

    override fun toString(): String {
        val precStr = if (useNpuFp16) "FP16" else "FP32"
        val cacheStr = if (useContextCache) "cached" else "no-cache"
        return "BenchmarkConfig(model=${modelType.displayName}, ep=${executionProvider.displayName}, " +
                "phase=${phase.displayName}, input=${inputMode.displayName}, " +
                "prec=$precStr, cache=$cacheStr, freq=${frequencyHz}Hz)"
    }

    companion object {
        /** Create Phase 1 (Burst Latency) config */
        fun burst(
            modelType: OnnxModelType,
            executionProvider: ExecutionProvider,
            useNpuFp16: Boolean = true,
            useContextCache: Boolean = false,
            htpPerformanceMode: String = "burst"
        ) = BenchmarkConfig(
            modelType = modelType,
            executionProvider = executionProvider,
            phase = BenchmarkPhase.BURST,
            inputMode = InputMode.CAMERA_SINGLE,
            frequencyHz = 2,        // 500ms sleep
            iterations = 100,
            useNpuFp16 = useNpuFp16,
            useContextCache = useContextCache,
            htpPerformanceMode = htpPerformanceMode
        )

        /** Create Phase 2 (Sustained Throughput) config */
        fun sustained(
            modelType: OnnxModelType,
            executionProvider: ExecutionProvider,
            useNpuFp16: Boolean = true,
            useContextCache: Boolean = false,
            htpPerformanceMode: String = "sustained_high"
        ) = BenchmarkConfig(
            modelType = modelType,
            executionProvider = executionProvider,
            phase = BenchmarkPhase.SUSTAINED,
            inputMode = InputMode.CAMERA_LIVE,
            frequencyHz = 30,       // 33ms target
            durationMinutes = 5,
            useNpuFp16 = useNpuFp16,
            useContextCache = useContextCache,
            htpPerformanceMode = htpPerformanceMode
        )
    }
}
