package com.example.kpilab.batch

/**
 * Default values for experiment configuration.
 * Used when individual experiments don't specify a value.
 */
data class ExperimentDefaults(
    val frequencyHz: Int = 10,
    val durationMinutes: Int = 5,
    val warmUpEnabled: Boolean = true,
    val useNpuFp16: Boolean = true,
    val useContextCache: Boolean = false
)
