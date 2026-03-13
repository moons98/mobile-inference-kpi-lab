package com.example.kpilab

/**
 * Benchmark configuration for text-to-image generation pipeline.
 * SD v1.5 (baseline) vs LCM-LoRA (optimized few-step generation).
 * Sessions: Text Encoder + UNet + VAE Decoder.
 */
data class BenchmarkConfig(
    // Model variant
    val modelVariant: ModelVariant = ModelVariant.SD_V15,

    // Execution provider (applied to all SD sessions)
    val sdBackend: ExecutionProvider = ExecutionProvider.QNN_NPU,

    // Per-component model precision
    val sdPrecisionMap: Map<SdComponent, SdPrecision> = SdComponent.values().associateWith { SdPrecision.FP16 },

    // Benchmark phase
    val phase: BenchmarkPhase = BenchmarkPhase.SINGLE_GENERATE,

    // UNet denoising steps
    val steps: Int = 20,

    // CFG guidance scale (SD v1.5: 7.5, LCM-LoRA: 1.0)
    val guidanceScale: Float = 7.5f,

    // Text prompt for generation
    val prompt: String = "a photo of a cat sitting on a windowsill",

    // Phase 1: trials per config / Phase 2: total consecutive trials
    val trials: Int = 5,

    // Warmup generations before measurement
    val warmupTrials: Int = 2,

    // NPU FP16 precision override
    val useNpuFp16: Boolean = true,

    // HTP performance mode
    val htpPerformanceMode: String = "burst",

    // Parallel session initialization (load TextEnc + UNet + VAE concurrently)
    val parallelInit: Boolean = false,

    // Model directory on device
    val modelDir: String = "/sdcard/sd_models"
) {
    /** Get precision for a specific SD component */
    fun sdPrecisionFor(component: SdComponent): SdPrecision =
        sdPrecisionMap[component] ?: SdPrecision.FP16

    /** Primary SD precision (most common among components, for logging/session ID) */
    val sdPrecision: SdPrecision get() {
        val counts = sdPrecisionMap.values.groupingBy { it }.eachCount()
        return counts.maxByOrNull { it.value }?.key ?: SdPrecision.FP16
    }

    /** Whether all components use the same precision */
    val isMixedPrecision: Boolean get() = sdPrecisionMap.values.toSet().size > 1

    /** Generation resolution — 512 fixed */
    val resolution: Int get() = 512

    /** Actual UNet steps */
    val actualSteps: Int get() = steps

    /** Latent space size = resolution / 8 */
    val latentSize: Int get() = resolution / 8

    /** Total trials for this phase */
    val totalTrials: Int
        get() = when (phase) {
            BenchmarkPhase.SINGLE_GENERATE -> trials
            BenchmarkPhase.SUSTAINED_GENERATE -> trials
        }

    /** UNet model filename (variant-aware) */
    fun unetFilename(): String {
        val prec = sdPrecisionFor(SdComponent.UNET)
        return SdComponent.UNET.filename(prec, modelVariant.unetPrefix)
    }

    /** Generate session ID for CSV/logging */
    fun generateSessionId(): String {
        val timestamp = System.currentTimeMillis()
        val sdEp = when (sdBackend) {
            ExecutionProvider.QNN_NPU -> "npu"
            ExecutionProvider.QNN_GPU -> "gpu"
            ExecutionProvider.CPU -> "cpu"
        }
        val phaseStr = when (phase) {
            BenchmarkPhase.SINGLE_GENERATE -> "single"
            BenchmarkPhase.SUSTAINED_GENERATE -> "sustained"
        }
        val variantStr = when (modelVariant) {
            ModelVariant.SD_V15 -> "sd15"
            ModelVariant.LCM_LORA -> "lcm"
        }
        val precStr = if (isMixedPrecision) {
            "mixed_" + sdPrecisionMap.entries.joinToString("_") { "${it.key.baseName[0]}${it.value.dirSuffix}" }
        } else {
            sdPrecision.dirSuffix
        }
        return "txt2img_${variantStr}_${precStr}_${sdEp}_s${steps}_${phaseStr}_${timestamp}"
    }

    override fun toString(): String {
        val precStr = if (isMixedPrecision) {
            sdPrecisionMap.entries.joinToString(", ") { "${it.key.baseName}=${it.value.displayName}" }
        } else {
            sdPrecision.displayName
        }
        return "Txt2ImgConfig(variant=${modelVariant.displayName}, sdEp=${sdBackend.displayName}, " +
                "sdPrec=$precStr, steps=$steps, guidance=$guidanceScale, " +
                "phase=${phase.displayName})"
    }

    companion object {
        /** Create uniform precision map (all components same precision) */
        fun uniformPrecision(precision: SdPrecision): Map<SdComponent, SdPrecision> =
            SdComponent.values().associateWith { precision }

        /** Phase 1: Single Generate Profiling */
        fun singleGenerate(
            modelVariant: ModelVariant = ModelVariant.SD_V15,
            sdBackend: ExecutionProvider = ExecutionProvider.QNN_NPU,
            sdPrecisionMap: Map<SdComponent, SdPrecision> = uniformPrecision(SdPrecision.FP16),
            steps: Int = 20,
            guidanceScale: Float = 7.5f,
            prompt: String = "a photo of a cat sitting on a windowsill"
        ) = BenchmarkConfig(
            modelVariant = modelVariant,
            sdBackend = sdBackend,
            sdPrecisionMap = sdPrecisionMap,
            phase = BenchmarkPhase.SINGLE_GENERATE,
            steps = steps,
            guidanceScale = guidanceScale,
            prompt = prompt,
            trials = 5,
            warmupTrials = 2,
            htpPerformanceMode = "burst"
        )

        /** Phase 2: Sustained Generate Test */
        fun sustainedGenerate(
            modelVariant: ModelVariant = ModelVariant.SD_V15,
            sdBackend: ExecutionProvider = ExecutionProvider.QNN_NPU,
            sdPrecisionMap: Map<SdComponent, SdPrecision> = uniformPrecision(SdPrecision.FP16),
            steps: Int = 20,
            guidanceScale: Float = 7.5f
        ) = BenchmarkConfig(
            modelVariant = modelVariant,
            sdBackend = sdBackend,
            sdPrecisionMap = sdPrecisionMap,
            phase = BenchmarkPhase.SUSTAINED_GENERATE,
            steps = steps,
            guidanceScale = guidanceScale,
            trials = 10,
            warmupTrials = 2,
            htpPerformanceMode = "sustained_high"
        )
    }
}
