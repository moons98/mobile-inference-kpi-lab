package com.example.kpilab

/**
 * Benchmark phase defining the execution strategy.
 */
enum class BenchmarkPhase(val displayName: String) {
    /** Phase 1: Single erase profiling — 5 trials, cooldown between */
    SINGLE_ERASE("Phase 1: Single Erase"),

    /** Phase 2: Sustained erase — 10 consecutive trials, no cooldown */
    SUSTAINED_ERASE("Phase 2: Sustained Erase"),

    /** YOLO-seg only profiling — 20 trials, no cooldown */
    YOLO_SEG_ONLY("YOLO-seg Profiling")
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
 * YOLO-seg model precision.
 */
enum class YoloPrecision(val displayName: String, val suffix: String) {
    FP32("FP32", ""),
    INT8("INT8", "_int8_qdq")
}

/**
 * SD Inpainting pipeline component.
 * 각 precision별 별도 모델 파일명을 관리한다.
 */
enum class SdComponent(
    val displayName: String,
    val baseName: String
) {
    VAE_ENCODER("VAE Encoder", "vae_encoder"),
    TEXT_ENCODER("Text Encoder", "text_encoder"),
    INPAINT_UNET("Inpainting UNet", "unet"),
    VAE_DECODER("VAE Decoder", "vae_decoder");

    /**
     * On-device ONNX filename matching export_sd_to_onnx.py output.
     * FP16 uses FP32 model file + QNN EP useNpuFp16 runtime option.
     */
    fun filename(precision: SdPrecision): String = when (precision) {
        SdPrecision.FP32 -> "${baseName}_fp32.onnx"
        SdPrecision.FP16 -> "${baseName}_fp32.onnx"
        SdPrecision.W8A8 -> "${baseName}_int8_qdq.onnx"
    }
}

/**
 * ROI size category for test data selection.
 */
enum class RoiSize(val displayName: String, val testImage: String, val testMask: String) {
    SMALL("Small (~128²)", "scene_small.jpg", "mask_small.png"),
    MEDIUM("Medium (~256²)", "scene_medium.jpg", "mask_medium.png"),
    LARGE("Large (~400²)", "scene_large.jpg", "mask_large.png")
}
