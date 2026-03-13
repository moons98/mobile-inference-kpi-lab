package com.example.kpilab.batch

import com.example.kpilab.BenchmarkConfig
import com.example.kpilab.BenchmarkPhase
import com.example.kpilab.ExecutionProvider
import com.example.kpilab.ModelVariant
import com.example.kpilab.SdComponent
import com.example.kpilab.SdPrecision

/**
 * Individual text-to-image experiment configuration.
 * Null values inherit from ExperimentDefaults.
 *
 * Per-component precision: sdPrecision sets the baseline for all components.
 * Optional precTextEnc/precUnet/precVaeDec override individual components.
 */
data class ExperimentConfig(
    // SD settings
    val sdBackend: String,              // ExecutionProvider enum name
    val sdPrecision: String,            // SdPrecision enum name (baseline for all components)
    val modelVariant: String? = null,   // ModelVariant enum name
    val phase: String? = null,
    val steps: Int? = null,
    val guidanceScale: Float? = null,
    val prompt: String? = null,
    val trials: Int? = null,
    val useNpuFp16: Boolean? = null,
    val htpPerformanceMode: String? = null,
    val parallelInit: Boolean? = null,
    val modelDir: String? = null,
    // Per-component precision overrides (optional)
    val precTextEnc: String? = null,
    val precUnet: String? = null,
    val precVaeDec: String? = null
) {
    fun toBenchmarkConfig(defaults: ExperimentDefaults): BenchmarkConfig {
        val sdEp = try { ExecutionProvider.valueOf(sdBackend) }
        catch (e: IllegalArgumentException) { ExecutionProvider.QNN_NPU }

        val basePrec = try { SdPrecision.valueOf(sdPrecision) }
        catch (e: IllegalArgumentException) { SdPrecision.FP16 }

        val variant = try { ModelVariant.valueOf(modelVariant ?: defaults.modelVariant) }
        catch (e: IllegalArgumentException) { ModelVariant.SD_V15 }

        // Build per-component precision map
        val precMap = mutableMapOf<SdComponent, SdPrecision>()
        for (comp in SdComponent.values()) {
            val overrideStr = when (comp) {
                SdComponent.TEXT_ENCODER -> precTextEnc
                SdComponent.UNET -> precUnet
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
        catch (e: IllegalArgumentException) { BenchmarkPhase.SINGLE_GENERATE }

        return BenchmarkConfig(
            modelVariant = variant,
            sdBackend = sdEp,
            sdPrecisionMap = precMap,
            phase = benchPhase,
            steps = steps ?: defaults.steps,
            guidanceScale = guidanceScale ?: defaults.guidanceScale,
            prompt = prompt ?: defaults.prompt,
            trials = trials ?: defaults.trials,
            warmupTrials = defaults.warmupTrials,
            useNpuFp16 = useNpuFp16 ?: defaults.useNpuFp16,
            htpPerformanceMode = htpPerformanceMode ?: defaults.htpPerformanceMode,
            parallelInit = parallelInit ?: defaults.parallelInit,
            modelDir = modelDir ?: defaults.modelDir
        )
    }

    fun getDisplayName(): String {
        val epName = try { ExecutionProvider.valueOf(sdBackend).displayName }
        catch (e: IllegalArgumentException) { sdBackend }
        val precName = try { SdPrecision.valueOf(sdPrecision).displayName }
        catch (e: IllegalArgumentException) { sdPrecision }

        val overrides = listOfNotNull(
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
        if (modelVariant != null) extras.add(modelVariant)
        if (steps != null) extras.add("${steps}steps")
        if (guidanceScale != null) extras.add("cfg=$guidanceScale")
        val suffix = if (extras.isNotEmpty()) " [${extras.joinToString(", ")}]" else ""
        return "$precDisplay / $epName$suffix"
    }
}
