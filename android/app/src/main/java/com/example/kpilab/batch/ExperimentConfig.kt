package com.example.kpilab.batch

import com.example.kpilab.BenchmarkConfig
import com.example.kpilab.ExecutionProvider
import com.example.kpilab.OnnxModelType

/**
 * Individual experiment configuration.
 * Null values inherit from ExperimentDefaults.
 */
data class ExperimentConfig(
    val model: String,                    // OnnxModelType enum name
    val executionProvider: String,        // ExecutionProvider enum name
    val frequencyHz: Int? = null,
    val durationMinutes: Int? = null,
    val useNpuFp16: Boolean? = null,
    val useContextCache: Boolean? = null
) {
    /**
     * Convert to BenchmarkConfig using defaults for missing values
     */
    fun toBenchmarkConfig(defaults: ExperimentDefaults): BenchmarkConfig {
        val modelType = try {
            OnnxModelType.valueOf(model)
        } catch (e: IllegalArgumentException) {
            OnnxModelType.MOBILENET_V2
        }

        val ep = try {
            ExecutionProvider.valueOf(executionProvider)
        } catch (e: IllegalArgumentException) {
            ExecutionProvider.QNN_NPU
        }

        return BenchmarkConfig(
            modelType = modelType,
            executionProvider = ep,
            frequencyHz = frequencyHz ?: defaults.frequencyHz,
            durationMinutes = durationMinutes ?: defaults.durationMinutes,
            useNpuFp16 = useNpuFp16 ?: defaults.useNpuFp16,
            useContextCache = useContextCache ?: defaults.useContextCache
        )
    }

    /**
     * Get display name for this experiment
     */
    fun getDisplayName(): String {
        val modelName = try {
            OnnxModelType.valueOf(model).displayName
        } catch (e: IllegalArgumentException) {
            model
        }
        val epName = try {
            ExecutionProvider.valueOf(executionProvider).displayName
        } catch (e: IllegalArgumentException) {
            executionProvider
        }
        return "$modelName / $epName"
    }
}
