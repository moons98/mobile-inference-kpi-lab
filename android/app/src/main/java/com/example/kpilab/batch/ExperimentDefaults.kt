package com.example.kpilab.batch

/**
 * Default values for text-to-image experiment configuration.
 */
data class ExperimentDefaults(
    val phase: String = "SINGLE_GENERATE",
    val modelVariant: String = "SD_V15",
    val steps: Int = 20,
    val guidanceScale: Float = 7.5f,
    val prompt: String = "a photo of a cat sitting on a windowsill",
    val trials: Int = 5,
    val warmupTrials: Int = 2,
    val useNpuFp16: Boolean = true,
    val htpPerformanceMode: String = "burst",
    val parallelInit: Boolean = false,
    val modelDir: String = "/sdcard/sd_models"
)
