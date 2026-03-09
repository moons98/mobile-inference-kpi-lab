package com.example.kpilab.batch

import com.example.kpilab.BenchmarkConfig
import com.example.kpilab.BenchmarkPhase
import com.example.kpilab.ExecutionProvider
import com.example.kpilab.InputMode
import com.example.kpilab.OnnxModelType

/**
 * Individual experiment configuration.
 * Null values inherit from ExperimentDefaults.
 */
data class ExperimentConfig(
    val model: String,                    // OnnxModelType enum name
    val executionProvider: String,        // ExecutionProvider enum name
    val phase: String? = null,            // BenchmarkPhase enum name
    val inputMode: String? = null,        // InputMode enum name
    val frequencyHz: Int? = null,
    val durationMinutes: Int? = null,
    val iterations: Int? = null,
    val useNpuFp16: Boolean? = null,
    val useContextCache: Boolean? = null,
    val htpPerformanceMode: String? = null
) {
    /**
     * Convert to BenchmarkConfig using defaults for missing values
     */
    fun toBenchmarkConfig(defaults: ExperimentDefaults): BenchmarkConfig {
        val modelType = try {
            OnnxModelType.valueOf(model)
        } catch (e: IllegalArgumentException) {
            OnnxModelType.YOLOV8N
        }

        val ep = try {
            ExecutionProvider.valueOf(executionProvider)
        } catch (e: IllegalArgumentException) {
            ExecutionProvider.QNN_NPU
        }

        val benchPhase = try {
            BenchmarkPhase.valueOf(phase ?: defaults.phase)
        } catch (e: IllegalArgumentException) {
            BenchmarkPhase.BURST
        }

        val benchInputMode = try {
            InputMode.valueOf(inputMode ?: defaults.inputMode)
        } catch (e: IllegalArgumentException) {
            InputMode.CAMERA_SINGLE
        }

        return BenchmarkConfig(
            modelType = modelType,
            executionProvider = ep,
            phase = benchPhase,
            inputMode = benchInputMode,
            frequencyHz = frequencyHz ?: defaults.frequencyHz,
            durationMinutes = durationMinutes ?: defaults.durationMinutes,
            iterations = iterations ?: defaults.iterations,
            useNpuFp16 = useNpuFp16 ?: defaults.useNpuFp16,
            useContextCache = useContextCache ?: defaults.useContextCache,
            htpPerformanceMode = htpPerformanceMode ?: defaults.htpPerformanceMode
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
        val perfSuffix = if (htpPerformanceMode != null && htpPerformanceMode != "burst") " ($htpPerformanceMode)" else ""
        return "$modelName / $epName$perfSuffix"
    }

    /**
     * Get a detail line showing this experiment's settings with overrides highlighted.
     * Shows: model / EP + any per-experiment overrides from defaults.
     */
    fun getDetailLine(defaults: ExperimentDefaults): String {
        val base = getDisplayName()
        val overrides = mutableListOf<String>()

        if (useNpuFp16 != null && useNpuFp16 != defaults.useNpuFp16) {
            overrides.add(if (useNpuFp16) "FP16" else "FP32")
        } else if (useNpuFp16 == false && defaults.useNpuFp16) {
            overrides.add("FP32")
        }
        if (phase != null && phase != defaults.phase) {
            overrides.add(phase)
        }
        if (inputMode != null && inputMode != defaults.inputMode) {
            overrides.add(inputMode.replace("_", " "))
        }
        if (frequencyHz != null && frequencyHz != defaults.frequencyHz) {
            overrides.add("${frequencyHz}Hz")
        }
        if (iterations != null && iterations != defaults.iterations) {
            overrides.add("${iterations}iter")
        }
        if (htpPerformanceMode != null && htpPerformanceMode != defaults.htpPerformanceMode) {
            overrides.add("htp:$htpPerformanceMode")
        }

        return if (overrides.isEmpty()) base else "$base [${overrides.joinToString(", ")}]"
    }
}
