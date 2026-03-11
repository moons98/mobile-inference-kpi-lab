package com.example.kpilab.batch

/**
 * Default values for AI Eraser experiment configuration.
 */
data class ExperimentDefaults(
    val phase: String = "SINGLE_ERASE",
    val steps: Int = 20,
    val strength: Float = 0.7f,
    val roiSize: String = "MEDIUM",
    val roiPaddingRatio: Float = 1.5f,
    val trials: Int = 5,
    val warmupTrials: Int = 2,
    val useNpuFp16: Boolean = true,
    val skipTextEncode: Boolean = true,
    val htpPerformanceMode: String = "burst",
    val modelDir: String = "/sdcard/sd_models",
    // YOLO defaults
    val yoloBackend: String = "QNN_NPU",
    val yoloPrecision: String = "FP32"
)
