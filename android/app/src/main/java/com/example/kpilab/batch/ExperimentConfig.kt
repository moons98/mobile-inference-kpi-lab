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
 *
 * Per-component precision: sdPrecision sets the baseline for all components.
 * Optional precVaeEnc/precTextEnc/precUnet/precVaeDec override individual components.
 */
data class ExperimentConfig(
    // SD Inpainting settings
    val sdBackend: String,              // ExecutionProvider enum name
    val sdPrecision: String,            // SdPrecision enum name (baseline for all components)
    val phase: String? = null,
    val steps: Int? = null,
    val strength: Float? = null,
    val roiSize: String? = null,
    val roiPaddingRatio: Float? = null,
    val trials: Int? = null,
    val useNpuFp16: Boolean? = null,
    val skipTextEncode: Boolean? = null,
    val htpPerformanceMode: String? = null,
    val modelDir: String? = null,
    // Per-component precision overrides (optional)
    val precVaeEnc: String? = null,
    val precTextEnc: String? = null,
    val precUnet: String? = null,
    val precVaeDec: String? = null,
    // YOLO-seg settings
    val yoloBackend: String? = null,
    val yoloPrecision: String? = null
) {
    fun toBenchmarkConfig(defaults: ExperimentDefaults): BenchmarkConfig {
        val sdEp = try { ExecutionProvider.valueOf(sdBackend) }
        catch (e: IllegalArgumentException) { ExecutionProvider.QNN_NPU }

        val basePrec = try { SdPrecision.valueOf(sdPrecision) }
        catch (e: IllegalArgumentException) { SdPrecision.FP16 }

        // Build per-component precision map
        val precMap = mutableMapOf<SdComponent, SdPrecision>()
        for (comp in SdComponent.values()) {
            val overrideStr = when (comp) {
                SdComponent.VAE_ENCODER -> precVaeEnc
                SdComponent.TEXT_ENCODER -> precTextEnc
                SdComponent.INPAINT_UNET -> precUnet
                SdComponent.VAE_DECODER -> precVaeDec
            }
            precMap[comp] = if (overrideStr != null) {
                try { SdPrecision.valueOf(overrideStr) }
                catch (e: IllegalArgumentException) { basePrec }
            } else {
                basePrec
            }
        }

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
            sdPrecisionMap = precMap,
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
            skipTextEncode = skipTextEncode ?: defaults.skipTextEncode,
            htpPerformanceMode = htpPerformanceMode ?: defaults.htpPerformanceMode,
            modelDir = modelDir ?: defaults.modelDir
        )
    }

    fun getDisplayName(): String {
        val epName = try { ExecutionProvider.valueOf(sdBackend).displayName }
        catch (e: IllegalArgumentException) { sdBackend }
        val precName = try { SdPrecision.valueOf(sdPrecision).displayName }
        catch (e: IllegalArgumentException) { sdPrecision }

        // Show per-component overrides if any
        val overrides = listOfNotNull(
            precVaeEnc?.let { "vae_enc=$it" },
            precTextEnc?.let { "txt_enc=$it" },
            precUnet?.let { "unet=$it" },
            precVaeDec?.let { "vae_dec=$it" }
        )
        val precDisplay = if (overrides.isNotEmpty()) {
            "$precName (${overrides.joinToString(", ")})"
        } else {
            precName
        }

        val extras = mutableListOf<String>()
        if (steps != null) extras.add("${steps}steps")
        if (strength != null) extras.add("str=$strength")
        if (roiSize != null) extras.add("roi=$roiSize")
        if (skipTextEncode == true) extras.add("skipTxtEnc")
        if (yoloPrecision != null) extras.add("yolo=$yoloPrecision")
        val suffix = if (extras.isNotEmpty()) " [${extras.joinToString(", ")}]" else ""
        return "$precDisplay / $epName$suffix"
    }
}
