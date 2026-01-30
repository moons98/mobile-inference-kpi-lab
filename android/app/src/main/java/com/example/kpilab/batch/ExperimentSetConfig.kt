package com.example.kpilab.batch

/**
 * Root structure for experiment_sets.json
 */
data class ExperimentSetConfig(
    val version: Int = 1,
    val defaults: ExperimentDefaults = ExperimentDefaults(),
    val experimentSets: List<ExperimentSet> = emptyList()
)
