package com.example.kpilab.batch

/**
 * Progress state for batch experiment execution.
 */
data class BatchProgress(
    val isRunning: Boolean = false,
    val currentSetName: String = "",
    val currentExperimentIndex: Int = 0,
    val totalExperiments: Int = 0,
    val currentExperimentName: String = "",
    val completedExperiments: List<String> = emptyList(),  // List of completed CSV filenames
    val isCoolingDown: Boolean = false,
    val cooldownRemainingSeconds: Int = 0
) {
    /**
     * Format progress as "2/5" style string
     */
    fun formatProgress(): String {
        return if (totalExperiments > 0) {
            "$currentExperimentIndex / $totalExperiments"
        } else {
            "-- / --"
        }
    }

    /**
     * Get progress percentage (0-100)
     */
    val progressPercent: Int
        get() = if (totalExperiments > 0) {
            ((currentExperimentIndex - 1) * 100) / totalExperiments
        } else {
            0
        }
}
