package com.example.kpilab.batch

import com.example.kpilab.BenchmarkConfig
import com.example.kpilab.BenchmarkPhase
import com.example.kpilab.ExecutionProvider
import com.example.kpilab.RoiSize
import com.example.kpilab.SdComponent
import com.example.kpilab.SdPrecision
import com.example.kpilab.YoloPrecision

/**
 * Individual AI Eraser experiment configuration.
 * Null values inherit from ExperimentDefaults.
 */
data class ExperimentConfig(
    // SD Inpainting settings
    val sdBackend: String,              // ExecutionProvider enum name
    val sdPrecision: String,            // SdPrecision enum name
    val phase: String? = null,
    val steps: Int? = null,
    val strength: Float? = null,
    val roiSize: String? = null,
    val roiPaddingRatio: Float? = null,
    val trials: Int? = null,
    val useNpuFp16: Boolean? = null,
    val htpPerformanceMode: String? = null,
    val modelDir: String? = null,
    // YOLO-seg settings (독립)
    val yoloBackend: String? = null,
    val yoloPrecision: String? = null
) {
    fun toBenchmarkConfig(defaults: ExperimentDefaults): BenchmarkConfig {
        val sdEp = try { ExecutionProvider.valueOf(sdBackend) }
        catch (e: IllegalArgumentException) { ExecutionProvider.QNN_NPU }

        val sdPrec = try { SdPrecision.valueOf(sdPrecision) }
        catch (e: IllegalArgumentException) { SdPrecision.FP16 }

        val benchPhase = try { BenchmarkPhase.valueOf(phase ?: defaults.phase) }
        catch (e: IllegalArgumentException) { BenchmarkPhase.SINGLE_ERASE }

        val roi = try { RoiSize.valueOf(roiSize ?: defaults.roiSize) }
        catch (e: IllegalArgumentException) { RoiSize.MEDIUM }

        val yoloEp = try { ExecutionProvider.valueOf(yoloBackend ?: defaults.yoloBackend) }
        catch (e: IllegalArgumentException) { ExecutionProvider.QNN_NPU }

        val yoloPrec = try { YoloPrecision.valueOf(yoloPrecision ?: defaults.yoloPrecision) }
        catch (e: IllegalArgumentException) { YoloPrecision.FP32 }

        return BenchmarkConfig(
            sdBackend = sdEp,
            sdPrecisionMap = SdComponent.values().associateWith { sdPrec },
            yoloBackend = yoloEp,
            yoloPrecision = yoloPrec,
            phase = benchPhase,
            steps = steps ?: defaults.steps,
            strength = strength ?: defaults.strength,
            roiSize = roi,
            roiPaddingRatio = roiPaddingRatio ?: defaults.roiPaddingRatio,
            trials = trials ?: defaults.trials,
            warmupTrials = defaults.warmupTrials,
            useNpuFp16 = useNpuFp16 ?: defaults.useNpuFp16,
            htpPerformanceMode = htpPerformanceMode ?: defaults.htpPerformanceMode,
            modelDir = modelDir ?: defaults.modelDir
        )
    }

    fun getDisplayName(): String {
        val epName = try { ExecutionProvider.valueOf(sdBackend).displayName }
        catch (e: IllegalArgumentException) { sdBackend }
        val precName = try { SdPrecision.valueOf(sdPrecision).displayName }
        catch (e: IllegalArgumentException) { sdPrecision }
        val extras = mutableListOf<String>()
        if (steps != null) extras.add("${steps}steps")
        if (strength != null) extras.add("str=$strength")
        if (roiSize != null) extras.add("roi=$roiSize")
        if (yoloPrecision != null) extras.add("yolo=$yoloPrecision")
        val suffix = if (extras.isNotEmpty()) " [${extras.joinToString(", ")}]" else ""
        return "$precName / $epName$suffix"
    }
}
