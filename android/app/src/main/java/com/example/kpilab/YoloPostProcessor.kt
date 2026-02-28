package com.example.kpilab

/**
 * YOLO detection result.
 */
data class Detection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val classId: Int,
    val className: String
)

/**
 * YOLOv8 postprocessor.
 *
 * Processes raw model output [1, 84, 8400] into detection results:
 * 1. Parse per-detection features (cx, cy, w, h + 80 class scores)
 * 2. Confidence filter (max class score > threshold)
 * 3. Per-class NMS (Non-Maximum Suppression)
 * 4. Coordinate transform from letterboxed 640x640 back to original image
 */
object YoloPostProcessor {

    private const val NUM_CLASSES = 80
    private const val NUM_DETECTIONS = 8400
    private const val CONFIDENCE_THRESHOLD = 0.25f
    private const val NMS_IOU_THRESHOLD = 0.45f
    private const val MAX_DETECTIONS = 300

    val COCO_CLASSES = arrayOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    )

    /**
     * Process raw model output into detections.
     *
     * @param output Raw float array from model output [1, 84, 8400] row-major
     * @param originalWidth Original image width (before letterboxing)
     * @param originalHeight Original image height (before letterboxing)
     * @param padLeft Letterbox left padding in pixels
     * @param padTop Letterbox top padding in pixels
     * @param scale Letterbox scale factor
     * @return List of Detection objects
     */
    fun process(
        output: FloatArray,
        originalWidth: Int,
        originalHeight: Int,
        padLeft: Float,
        padTop: Float,
        scale: Float
    ): List<Detection> {
        // output layout: [1, 84, 8400] row-major
        // Row i (0..83): values for feature i across all 8400 detections
        // Rows 0-3: cx, cy, w, h
        // Rows 4-83: class scores (80 classes)

        val candidates = mutableListOf<Detection>()

        for (d in 0 until NUM_DETECTIONS) {
            var maxScore = 0f
            var maxClassId = 0
            for (c in 0 until NUM_CLASSES) {
                val score = output[(4 + c) * NUM_DETECTIONS + d]
                if (score > maxScore) {
                    maxScore = score
                    maxClassId = c
                }
            }

            if (maxScore < CONFIDENCE_THRESHOLD) continue

            val cx = output[0 * NUM_DETECTIONS + d]
            val cy = output[1 * NUM_DETECTIONS + d]
            val w = output[2 * NUM_DETECTIONS + d]
            val h = output[3 * NUM_DETECTIONS + d]

            // Convert to corner format in original image coordinates
            val x1 = ((cx - w / 2f) - padLeft) / scale
            val y1 = ((cy - h / 2f) - padTop) / scale
            val x2 = ((cx + w / 2f) - padLeft) / scale
            val y2 = ((cy + h / 2f) - padTop) / scale

            candidates.add(
                Detection(
                    x1 = x1.coerceIn(0f, originalWidth.toFloat()),
                    y1 = y1.coerceIn(0f, originalHeight.toFloat()),
                    x2 = x2.coerceIn(0f, originalWidth.toFloat()),
                    y2 = y2.coerceIn(0f, originalHeight.toFloat()),
                    confidence = maxScore,
                    classId = maxClassId,
                    className = COCO_CLASSES[maxClassId]
                )
            )
        }

        return nms(candidates)
    }

    /**
     * Per-class Non-Maximum Suppression.
     */
    private fun nms(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val byClass = detections.groupBy { it.classId }
        val result = mutableListOf<Detection>()

        for ((_, classDets) in byClass) {
            val sorted = classDets.sortedByDescending { it.confidence }
            val kept = BooleanArray(sorted.size) { true }

            for (i in sorted.indices) {
                if (!kept[i]) continue
                for (j in i + 1 until sorted.size) {
                    if (!kept[j]) continue
                    if (iou(sorted[i], sorted[j]) > NMS_IOU_THRESHOLD) {
                        kept[j] = false
                    }
                }
            }

            for (i in sorted.indices) {
                if (kept[i]) result.add(sorted[i])
            }
        }

        return result.sortedByDescending { it.confidence }.take(MAX_DETECTIONS)
    }

    /**
     * Compute Intersection over Union between two detections.
     */
    private fun iou(a: Detection, b: Detection): Float {
        val interX1 = maxOf(a.x1, b.x1)
        val interY1 = maxOf(a.y1, b.y1)
        val interX2 = minOf(a.x2, b.x2)
        val interY2 = minOf(a.y2, b.y2)

        val interArea = maxOf(0f, interX2 - interX1) * maxOf(0f, interY2 - interY1)
        val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
        val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
        val unionArea = areaA + areaB - interArea

        return if (unionArea > 0f) interArea / unionArea else 0f
    }
}
