package com.example.kpilab

/**
 * Data type for model tensors
 */
enum class DataType(val bytesPerElement: Int) {
    FLOAT32(4),
    INT8(1),
    UINT8(1)
}

/**
 * Supported model types for benchmarking
 */
enum class ModelType(
    val displayName: String,
    val filename: String,
    val inputWidth: Int,
    val inputHeight: Int,
    val inputChannels: Int,
    val outputShape: IntArray,
    val dataType: DataType
) {
    // Float32 models
    MOBILENET_V2(
        displayName = "MobileNetV2",
        filename = "mobilenetv2.tflite",
        inputWidth = 224,
        inputHeight = 224,
        inputChannels = 3,
        outputShape = intArrayOf(1, 1001),
        dataType = DataType.FLOAT32
    ),
    MOBILENET_V2_QUANTIZED(
        displayName = "MobileNetV2 (INT8)",
        filename = "mobilenetv2_quantized.tflite",
        inputWidth = 224,
        inputHeight = 224,
        inputChannels = 3,
        outputShape = intArrayOf(1, 1001),
        dataType = DataType.UINT8
    ),
    YOLOV8N(
        displayName = "YOLOv8n",
        filename = "yolov8n.tflite",
        inputWidth = 640,
        inputHeight = 640,
        inputChannels = 3,
        outputShape = intArrayOf(1, 84, 8400),
        dataType = DataType.FLOAT32
    ),
    YOLOV8N_QUANTIZED(
        displayName = "YOLOv8n (INT8)",
        filename = "yolov8n_quantized.tflite",
        inputWidth = 640,
        inputHeight = 640,
        inputChannels = 3,
        outputShape = intArrayOf(1, 84, 8400),
        dataType = DataType.UINT8
    );

    /**
     * Calculate input buffer size in bytes
     */
    val inputBufferSize: Int
        get() = 1 * inputHeight * inputWidth * inputChannels * dataType.bytesPerElement

    /**
     * Calculate output buffer size in bytes
     */
    val outputBufferSize: Int
        get() = outputShape.reduce { acc, i -> acc * i } * dataType.bytesPerElement

    /**
     * Check if this is a quantized model
     */
    val isQuantized: Boolean
        get() = dataType != DataType.FLOAT32
}
