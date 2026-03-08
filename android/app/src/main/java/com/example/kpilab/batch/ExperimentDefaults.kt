package com.example.kpilab.batch

/**
 * Default values for experiment configuration.
 * Used when individual experiments don't specify a value.
 */
data class ExperimentDefaults(
    val frequencyHz: Int = 2,
    val durationMinutes: Int = 5,
    val iterations: Int = 100,
    val phase: String = "BURST",
    val inputMode: String = "CAMERA_SINGLE",
    val useNpuFp16: Boolean = true,
    val useContextCache: Boolean = false,
    val htpPerformanceMode: String = "burst"
)
