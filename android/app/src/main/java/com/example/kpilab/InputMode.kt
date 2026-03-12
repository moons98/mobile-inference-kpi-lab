package com.example.kpilab

/**
 * Benchmark phase defining the execution strategy.
 */
enum class BenchmarkPhase(val displayName: String) {
    /** Phase 1: Single generation profiling — 5 trials, cooldown between */
    SINGLE_GENERATE("Phase 1: Single Generate"),

    /** Phase 2: Sustained generation — 10 consecutive trials, no cooldown */
    SUSTAINED_GENERATE("Phase 2: Sustained Generate")
}

/**
 * Execution provider for ONNX Runtime.
 */
enum class ExecutionProvider(val displayName: String) {
    QNN_NPU("NPU (Hexagon HTP)"),
    QNN_GPU("GPU (Adreno)"),
    CPU("CPU")
}

/**
 * SD model precision.
 * FP32: baseline (CPU 전용 또는 참고용)
 * FP16: 품질 기준선, NPU/GPU에서 기본
 * W8A8: INT8 weights + INT8 activations, aggressive 양자화
 */
enum class SdPrecision(val displayName: String, val dirSuffix: String) {
    FP32("FP32", "fp32"),
    FP16("FP16", "fp16"),
    W8A8("W8A8", "w8a8")
}

/**
 * Model variant: SD v1.5 baseline vs LCM-LoRA optimized.
 */
enum class ModelVariant(val displayName: String, val unetPrefix: String) {
    SD_V15("SD v1.5", "unet_base"),
    LCM_LORA("LCM-LoRA", "unet_lcm")
}

/**
 * SD text-to-image pipeline component.
 * 각 precision별 별도 모델 파일명을 관리한다.
 */
enum class SdComponent(
    val displayName: String,
    val baseName: String
) {
    TEXT_ENCODER("Text Encoder", "text_encoder"),
    UNET("UNet", "unet"),
    VAE_DECODER("VAE Decoder", "vae_decoder");

    /**
     * On-device ONNX filename matching export script output.
     * FP16 uses FP32 model file + QNN EP useNpuFp16 runtime option.
     * For LCM-LoRA UNet, caller should use ModelVariant.unetPrefix instead of baseName.
     */
    fun filename(precision: SdPrecision): String = when (precision) {
        SdPrecision.FP32 -> "${baseName}_fp32.onnx"
        SdPrecision.FP16 -> "${baseName}_fp32.onnx"
        SdPrecision.W8A8 -> "${baseName}_int8_qdq.onnx"
    }

    /**
     * Filename with custom base name (for variant-specific UNet).
     */
    fun filename(precision: SdPrecision, customBaseName: String): String = when (precision) {
        SdPrecision.FP32 -> "${customBaseName}_fp32.onnx"
        SdPrecision.FP16 -> "${customBaseName}_fp32.onnx"
        SdPrecision.W8A8 -> "${customBaseName}_int8_qdq.onnx"
    }
}
