package com.example.kpilab

/**
 * Benchmark configuration for AI Eraser pipeline.
 * SD v1.5 Inpainting (5 sessions: YOLO-seg + VAE Enc + Text Enc + Inpaint UNet + VAE Dec)
 */
data class BenchmarkConfig(
    // SD Inpainting — execution provider (applied to all 4 SD sessions)
    val sdBackend: ExecutionProvider = ExecutionProvider.QNN_NPU,

    // SD Inpainting — per-component model precision
    val sdPrecisionMap: Map<SdComponent, SdPrecision> = SdComponent.values().associateWith { SdPrecision.FP16 },

    // YOLO-seg — execution provider (독립 설정)
    val yoloBackend: ExecutionProvider = ExecutionProvider.QNN_NPU,

    // YOLO-seg — model precision
    val yoloPrecision: YoloPrecision = YoloPrecision.FP32,

    // Benchmark phase
    val phase: BenchmarkPhase = BenchmarkPhase.SINGLE_ERASE,

    // SD pipeline parameters
    val steps: Int = 20,
    val strength: Float = 0.7f,

    // ROI size category (테스트 데이터 선택)
    val roiSize: RoiSize = RoiSize.MEDIUM,

    // ROI padding ratio (bbox 대비)
    val roiPaddingRatio: Float = 1.5f,

    // Phase 1: trials per config / Phase 2: total consecutive trials
    val trials: Int = 5,

    // Warmup generations before measurement
    val warmupTrials: Int = 2,

    // NPU FP16 precision override
    val useNpuFp16: Boolean = true,

    // Skip text encoder — use precomputed text_embeddings.npy instead
    val skipTextEncode: Boolean = true,

    // QNN context cache (always enabled for precompiled models)
    val useContextCache: Boolean = true,

    // HTP performance mode
    val htpPerformanceMode: String = "burst",

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

    /** Inpainting resolution — 512 fixed for NPU */
    val resolution: Int get() = 512

    /** Actual UNet steps = total_steps × strength */
    val actualSteps: Int get() = (steps * strength).toInt()

    /** Latent space size = resolution / 8 */
    val latentSize: Int get() = resolution / 8

    /** Fixed prompt for eraser */
    val prompt: String get() = "remove the object and fill the background naturally"

    /** Total trials for this phase */
    val totalTrials: Int
        get() = when (phase) {
            BenchmarkPhase.SINGLE_ERASE -> trials
            BenchmarkPhase.SUSTAINED_ERASE -> trials
            BenchmarkPhase.YOLO_SEG_ONLY -> 20  // YOLO-seg 별도 측정 고정 20회
        }

    /** YOLO-seg model filename */
    val yoloModelFilename: String
        get() = "yolov8n-seg${yoloPrecision.suffix}.onnx"

    /** Generate session ID for CSV/logging */
    fun generateSessionId(): String {
        val timestamp = System.currentTimeMillis()
        val sdEp = when (sdBackend) {
            ExecutionProvider.QNN_NPU -> "npu"
            ExecutionProvider.QNN_GPU -> "gpu"
            ExecutionProvider.CPU -> "cpu"
        }
        val phaseStr = when (phase) {
            BenchmarkPhase.SINGLE_ERASE -> "single"
            BenchmarkPhase.SUSTAINED_ERASE -> "sustained"
            BenchmarkPhase.YOLO_SEG_ONLY -> "yolo"
        }
        val precStr = if (isMixedPrecision) {
            "mixed_" + sdPrecisionMap.entries.joinToString("_") { "${it.key.baseName[0]}${it.value.dirSuffix}" }
        } else {
            sdPrecision.dirSuffix
        }
        return "eraser_${precStr}_${sdEp}_s${steps}_str${(strength * 10).toInt()}_${roiSize.name.lowercase()}_${phaseStr}_${timestamp}"
    }

    override fun toString(): String {
        val precStr = if (isMixedPrecision) {
            sdPrecisionMap.entries.joinToString(", ") { "${it.key.baseName}=${it.value.displayName}" }
        } else {
            sdPrecision.displayName
        }
        return "EraseConfig(sdEp=${sdBackend.displayName}, sdPrec=$precStr, " +
                "yoloEp=${yoloBackend.displayName}, yoloPrec=${yoloPrecision.displayName}, " +
                "steps=$steps, strength=$strength, roi=${roiSize.displayName}, " +
                "phase=${phase.displayName})"
    }

    companion object {
        /** Create uniform precision map (all components same precision) */
        fun uniformPrecision(precision: SdPrecision): Map<SdComponent, SdPrecision> =
            SdComponent.values().associateWith { precision }

        /** Phase 1: Single Erase Profiling */
        fun phase1(
            sdBackend: ExecutionProvider = ExecutionProvider.QNN_NPU,
            sdPrecisionMap: Map<SdComponent, SdPrecision> = uniformPrecision(SdPrecision.FP16),
            yoloBackend: ExecutionProvider = ExecutionProvider.QNN_NPU,
            yoloPrecision: YoloPrecision = YoloPrecision.FP32,
            steps: Int = 20,
            strength: Float = 0.7f,
            roiSize: RoiSize = RoiSize.MEDIUM
        ) = BenchmarkConfig(
            sdBackend = sdBackend,
            sdPrecisionMap = sdPrecisionMap,
            yoloBackend = yoloBackend,
            yoloPrecision = yoloPrecision,
            phase = BenchmarkPhase.SINGLE_ERASE,
            steps = steps,
            strength = strength,
            roiSize = roiSize,
            trials = 5,
            warmupTrials = 2,
            htpPerformanceMode = "burst"
        )

        /** Phase 2: Sustained Erase Test */
        fun phase2(
            sdBackend: ExecutionProvider = ExecutionProvider.QNN_NPU,
            sdPrecisionMap: Map<SdComponent, SdPrecision> = uniformPrecision(SdPrecision.FP16),
            steps: Int = 20,
            strength: Float = 0.7f,
            roiSize: RoiSize = RoiSize.MEDIUM
        ) = BenchmarkConfig(
            sdBackend = sdBackend,
            sdPrecisionMap = sdPrecisionMap,
            phase = BenchmarkPhase.SUSTAINED_ERASE,
            steps = steps,
            strength = strength,
            roiSize = roiSize,
            trials = 10,
            warmupTrials = 2,
            htpPerformanceMode = "sustained_high"
        )

        /** YOLO-seg only profiling */
        fun yoloOnly(
            yoloBackend: ExecutionProvider = ExecutionProvider.QNN_NPU,
            yoloPrecision: YoloPrecision = YoloPrecision.FP32
        ) = BenchmarkConfig(
            yoloBackend = yoloBackend,
            yoloPrecision = yoloPrecision,
            phase = BenchmarkPhase.YOLO_SEG_ONLY
        )
    }
}
