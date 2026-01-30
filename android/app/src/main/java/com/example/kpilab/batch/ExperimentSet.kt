package com.example.kpilab.batch

/**
 * A named set of experiments to run sequentially.
 */
data class ExperimentSet(
    val id: String,
    val name: String,
    val experiments: List<ExperimentConfig>
) {
    override fun toString(): String = name
}
