package com.example.kpilab

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import java.nio.FloatBuffer
import java.util.Arrays
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Lightweight YOLOv8-seg runner for standalone profiling.
 * Notes:
 * - Decodes YOLOv8-seg outputs (det + proto) and computes real binary mask area.
 */
class YoloSegRunner(
    private val context: Context,
    private val config: BenchmarkConfig
) {
    companion object {
        private const val TAG = "YoloSegRunner"
        private const val INPUT_SIZE = 640
        private const val CONF_THRESHOLD = 0.25f
        private const val IOU_THRESHOLD = 0.45f
    }

    private var ortRunner: OrtRunner? = null

    data class ColdStartTiming(
        val loadMs: Long = 0
    )

    data class YoloResult(
        val e2eMs: Float,
        val preprocessMs: Float,
        val tensorCreateMs: Float,
        val inferenceMs: Float,
        val outputCopyMs: Float,
        val detParseMs: Float,
        val nmsMs: Float,
        val maskDecodeMs: Float,
        val maskCount: Int,
        val selectedMaskAreaPct: Float,
        val overlayBitmap: Bitmap? = null
    )

    var coldStartTiming: ColdStartTiming? = null
        private set

    fun initialize(): Boolean {
        val ep = config.yoloBackend
        val useFp16 = config.useNpuFp16 && config.yoloPrecision == YoloPrecision.FP32
        val perf = config.htpPerformanceMode
        val path = "${config.modelDir}/${config.yoloModelFilename}"

        val runner = OrtRunner(context)
        val ok = runner.initialize(
            modelPath = path,
            executionProvider = ep,
            useNpuFp16 = useFp16,
            useContextCache = true,
            htpPerformanceMode = perf,
            enableProfiling = false
        )
        if (!ok) {
            Log.e(TAG, "YOLO init failed: ${runner.lastError}")
            runner.release()
            return false
        }

        ortRunner = runner
        coldStartTiming = ColdStartTiming(
            loadMs = runner.modelLoadMs + runner.sessionCreateMs
        )
        Log.i(TAG, "YOLO ready: ${coldStartTiming?.loadMs}ms, model=${config.yoloModelFilename}")
        return true
    }

    fun warmup(input: Bitmap, count: Int) {
        for (i in 0 until count) {
            runOnce(input, generateOverlay = false)
        }
    }

    fun runOnce(inputBitmap: Bitmap, generateOverlay: Boolean = false): YoloResult? {
        val runner = ortRunner ?: return null

        val e2eStart = System.nanoTime()

        // Phase 1: Bitmap preprocessing (resize + normalize + CHW)
        val prepStart = System.nanoTime()
        val (inputBuffer, inputShape) = preprocess(inputBitmap)
        val preprocessMs = nsToMs(System.nanoTime() - prepStart)

        // Phase 2: ORT run (tensor create + session.run + output copy)
        val inputName = runner.inputNames.firstOrNull() ?: "images"
        val runResult = runner.run(mapOf(inputName to Pair(inputBuffer, inputShape))) ?: return null

        // Phase 3: Detection parsing (output → candidates)
        val detParseStart = System.nanoTime()
        val detOutput = findDetOutput(runner, runResult.outputs)
        val protoOutput = findProtoOutput(runner, runResult.outputs)
        if (detOutput == null) {
            val detParseMs = nsToMs(System.nanoTime() - detParseStart)
            return YoloResult(
                e2eMs = nsToMs(System.nanoTime() - e2eStart),
                preprocessMs = preprocessMs,
                tensorCreateMs = runResult.inputCreateMs,
                inferenceMs = runResult.sessionRunMs,
                outputCopyMs = runResult.outputCopyMs,
                detParseMs = detParseMs,
                nmsMs = 0f, maskDecodeMs = 0f,
                maskCount = 0, selectedMaskAreaPct = 0f
            )
        }
        val det = detOutput.first
        val detShape = detOutput.second
        val detLayout = parseDetLayout(detShape)
        if (detLayout == null) {
            val detParseMs = nsToMs(System.nanoTime() - detParseStart)
            return YoloResult(
                e2eMs = nsToMs(System.nanoTime() - e2eStart),
                preprocessMs = preprocessMs,
                tensorCreateMs = runResult.inputCreateMs,
                inferenceMs = runResult.sessionRunMs,
                outputCopyMs = runResult.outputCopyMs,
                detParseMs = detParseMs,
                nmsMs = 0f, maskDecodeMs = 0f,
                maskCount = 0, selectedMaskAreaPct = 0f
            )
        }
        val channels = detLayout.channels
        val boxes = detLayout.boxes
        if (channels <= 4 || boxes <= 0) {
            val detParseMs = nsToMs(System.nanoTime() - detParseStart)
            return YoloResult(
                e2eMs = nsToMs(System.nanoTime() - e2eStart),
                preprocessMs = preprocessMs,
                tensorCreateMs = runResult.inputCreateMs,
                inferenceMs = runResult.sessionRunMs,
                outputCopyMs = runResult.outputCopyMs,
                detParseMs = detParseMs,
                nmsMs = 0f, maskDecodeMs = 0f,
                maskCount = 0, selectedMaskAreaPct = 0f
            )
        }

        val numMasks = inferMaskChannels(protoOutput?.second, channels)
        val clsStart = 4
        val clsEnd = (channels - numMasks).coerceAtLeast(clsStart)
        val candidates = mutableListOf<Detection>()
        for (i in 0 until boxes) {
            var best = 0f
            for (c in clsStart until clsEnd) {
                best = max(best, detValue(det, detLayout, i, c))
            }
            if (best < CONF_THRESHOLD) continue
            val cx = detValue(det, detLayout, i, 0)
            val cy = detValue(det, detLayout, i, 1)
            val w = max(0f, detValue(det, detLayout, i, 2))
            val h = max(0f, detValue(det, detLayout, i, 3))
            val x1 = cx - w * 0.5f
            val y1 = cy - h * 0.5f
            val x2 = cx + w * 0.5f
            val y2 = cy + h * 0.5f
            val coeffs = if (numMasks > 0) FloatArray(numMasks) { k ->
                detValue(det, detLayout, i, clsEnd + k)
            } else floatArrayOf()
            candidates.add(Detection(x1, y1, x2, y2, best, coeffs))
        }
        val detParseMs = nsToMs(System.nanoTime() - detParseStart)

        // Phase 4: NMS
        val nmsStart = System.nanoTime()
        val selected = nms(candidates, IOU_THRESHOLD)
        val nmsMs = nsToMs(System.nanoTime() - nmsStart)

        // Phase 5: Mask decode
        val maskDecodeStart = System.nanoTime()
        val selectedAreaPct = decodeSelectedMaskAreaPct(selected, protoOutput)
        val maskDecodeMs = nsToMs(System.nanoTime() - maskDecodeStart)

        val e2eMs = nsToMs(System.nanoTime() - e2eStart)

        // Generate overlay AFTER all timing measurements
        val overlay = if (generateOverlay) {
            try {
                createOverlay(inputBitmap, selected, protoOutput)
            } catch (e: Exception) {
                Log.e(TAG, "createOverlay failed: ${e.message}", e)
                null
            }
        } else null

        return YoloResult(
            e2eMs = e2eMs,
            preprocessMs = preprocessMs,
            tensorCreateMs = runResult.inputCreateMs,
            inferenceMs = runResult.sessionRunMs,
            outputCopyMs = runResult.outputCopyMs,
            detParseMs = detParseMs,
            nmsMs = nmsMs,
            maskDecodeMs = maskDecodeMs,
            maskCount = selected.size,
            selectedMaskAreaPct = selectedAreaPct,
            overlayBitmap = overlay
        )
    }

    fun release() {
        ortRunner?.release()
        ortRunner = null
    }

    private data class Detection(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val score: Float,
        val maskCoeffs: FloatArray
    )

    private data class DetLayout(
        val channels: Int,
        val boxes: Int,
        val channelMajor: Boolean
    )

    private fun preprocess(bitmap: Bitmap): Pair<FloatBuffer, LongArray> {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        if (resized !== bitmap) resized.recycle()

        val plane = INPUT_SIZE * INPUT_SIZE
        val buffer = FloatBuffer.allocate(3 * plane)
        for (i in pixels.indices) {
            val p = pixels[i]
            buffer.put(i, ((p shr 16) and 0xFF) / 255f)
            buffer.put(plane + i, ((p shr 8) and 0xFF) / 255f)
            buffer.put(2 * plane + i, (p and 0xFF) / 255f)
        }
        return Pair(buffer, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))
    }

    private fun findDetOutput(
        runner: OrtRunner,
        outputs: Map<String, Any>
    ): Pair<FloatArray, LongArray>? {
        var bestName: String? = null
        var bestScore = -1
        for ((name, shape) in runner.outputShapes) {
            if (shape.size != 3) continue
            val s1 = shape[1].toInt()
            val s2 = shape[2].toInt()
            val score = max(s1, s2)
            if (score > bestScore) {
                bestScore = score
                bestName = name
            }
        }
        val name = bestName ?: return null
        val arr = outputs[name] as? FloatArray ?: return null
        val shape = runner.outputShapes[name] ?: return null
        return Pair(arr, shape)
    }

    private fun findProtoOutput(
        runner: OrtRunner,
        outputs: Map<String, Any>
    ): Pair<FloatArray, LongArray>? {
        var protoName: String? = null
        for ((name, shape) in runner.outputShapes) {
            if (shape.size == 4) {
                protoName = name
                break
            }
        }
        val name = protoName ?: return null
        val arr = outputs[name] as? FloatArray ?: return null
        val shape = runner.outputShapes[name] ?: return null
        return Pair(arr, shape)
    }

    private fun parseDetLayout(shape: LongArray): DetLayout? {
        if (shape.size != 3) return null
        val a = shape[1].toInt()
        val b = shape[2].toInt()
        if (a <= 0 || b <= 0) return null
        return if (a <= b) {
            // [1, channels, boxes]
            DetLayout(channels = a, boxes = b, channelMajor = true)
        } else {
            // [1, boxes, channels]
            DetLayout(channels = b, boxes = a, channelMajor = false)
        }
    }

    private fun detValue(det: FloatArray, layout: DetLayout, boxIdx: Int, chIdx: Int): Float {
        return if (layout.channelMajor) {
            det[chIdx * layout.boxes + boxIdx]
        } else {
            det[boxIdx * layout.channels + chIdx]
        }
    }

    private fun inferMaskChannels(proto: LongArray?, channels: Int): Int {
        // Common YOLOv8-seg: [1, 116, 8400] => 4 bbox + 80 cls + 32 mask coeff.
        val protoShape = proto
        if (protoShape != null && protoShape.size >= 2) {
            val protoC = protoShape[1].toInt()
            if (protoC in 1 until channels) return protoC
        }
        return if (channels > 84) channels - 84 else 0
    }

    private fun nms(src: List<Detection>, iouThreshold: Float): List<Detection> {
        if (src.isEmpty()) return emptyList()
        val sorted = src.sortedByDescending { it.score }.toMutableList()
        val kept = mutableListOf<Detection>()
        while (sorted.isNotEmpty()) {
            val cur = sorted.removeAt(0)
            kept.add(cur)
            sorted.removeAll { iou(cur, it) > iouThreshold }
        }
        return kept
    }

    private fun iou(a: Detection, b: Detection): Float {
        val ix1 = max(a.x1, b.x1)
        val iy1 = max(a.y1, b.y1)
        val ix2 = min(a.x2, b.x2)
        val iy2 = min(a.y2, b.y2)
        val iw = max(0f, ix2 - ix1)
        val ih = max(0f, iy2 - iy1)
        val inter = iw * ih
        if (inter <= 0f) return 0f
        val union = bboxArea(a.x1, a.y1, a.x2, a.y2) + bboxArea(b.x1, b.y1, b.x2, b.y2) - inter
        return if (union <= 0f) 0f else inter / union
    }

    private fun bboxArea(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        return max(0f, x2 - x1) * max(0f, y2 - y1)
    }

    private fun decodeSelectedMaskAreaPct(
        selected: List<Detection>,
        protoOutput: Pair<FloatArray, LongArray>?
    ): Float {
        if (selected.isEmpty()) return 0f
        val proto = protoOutput ?: return 0f
        val protoArr = proto.first
        val shape = proto.second
        if (shape.size != 4) return 0f
        val c = shape[1].toInt()
        val h = shape[2].toInt()
        val w = shape[3].toInt()
        if (c <= 0 || h <= 0 || w <= 0) return 0f
        val top = selected.first()
        if (top.maskCoeffs.isEmpty()) return 0f
        val coeffCount = min(c, top.maskCoeffs.size)
        val hw = h * w
        var positive = 0

        val sx1 = (top.x1 / INPUT_SIZE * w).toInt().coerceIn(0, w)
        val sy1 = (top.y1 / INPUT_SIZE * h).toInt().coerceIn(0, h)
        val sx2 = (top.x2 / INPUT_SIZE * w).toInt().coerceIn(0, w)
        val sy2 = (top.y2 / INPUT_SIZE * h).toInt().coerceIn(0, h)
        if (sx2 <= sx1 || sy2 <= sy1) return 0f

        for (yy in sy1 until sy2) {
            for (xx in sx1 until sx2) {
                val idx = yy * w + xx
                var acc = 0f
                for (k in 0 until coeffCount) {
                    acc += top.maskCoeffs[k] * protoArr[k * hw + idx]
                }
                val prob = sigmoid(acc)
                if (prob > 0.5f) positive++
            }
        }

        return (positive.toFloat() / hw.toFloat()) * 100f
    }

    private fun createOverlay(
        inputBitmap: Bitmap,
        selected: List<Detection>,
        protoOutput: Pair<FloatArray, LongArray>?
    ): Bitmap {
        val overlay = inputBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(overlay)
        val imgW = overlay.width
        val imgH = overlay.height
        Log.i(TAG, "createOverlay: ${imgW}x${imgH}, detections=${selected.size}, proto=${protoOutput?.second?.contentToString()}")

        val scaleX = imgW.toFloat() / INPUT_SIZE
        val scaleY = imgH.toFloat() / INPUT_SIZE

        // Scale stroke width relative to image size
        val baseStroke = max(2f, min(imgW, imgH) / 200f)

        val bboxPaint = Paint().apply {
            color = Color.argb(220, 0, 255, 0)
            style = Paint.Style.STROKE
            strokeWidth = baseStroke
            isAntiAlias = true
        }
        val labelPaint = Paint().apply {
            color = Color.WHITE
            textSize = max(24f, baseStroke * 5f)
            isAntiAlias = true
        }
        val labelBgPaint = Paint().apply {
            color = Color.argb(180, 0, 0, 0)
            style = Paint.Style.FILL
        }

        if (selected.isEmpty()) {
            // No detections: show message on overlay
            val msgPaint = Paint().apply {
                color = Color.RED
                textSize = max(32f, imgH / 20f)
                isAntiAlias = true
            }
            canvas.drawText("No detections", 20f, imgH / 2f, msgPaint)
            Log.w(TAG, "createOverlay: no detections to draw")
            return overlay
        }

        // --- Mask overlay ---
        val proto = protoOutput
        if (proto != null && proto.second.size == 4) {
            val protoArr = proto.first
            val c = proto.second[1].toInt()
            val ph = proto.second[2].toInt()
            val pw = proto.second[3].toInt()
            val hw = ph * pw
            Log.i(TAG, "Proto mask: c=$c, h=$ph, w=$pw, arr.size=${protoArr.size}")

            if (c > 0 && ph > 0 && pw > 0 && protoArr.size >= c * hw) {
                for ((di, det) in selected.withIndex()) {
                    if (det.maskCoeffs.isEmpty()) continue
                    val coeffCount = min(c, det.maskCoeffs.size)

                    // Bounding box in original image space
                    val bx1 = (det.x1 * scaleX).toInt().coerceIn(0, imgW - 1)
                    val by1 = (det.y1 * scaleY).toInt().coerceIn(0, imgH - 1)
                    val bx2 = (det.x2 * scaleX).toInt().coerceIn(0, imgW)
                    val by2 = (det.y2 * scaleY).toInt().coerceIn(0, imgH)
                    if (bx2 <= bx1 || by2 <= by1) continue

                    val bw = bx2 - bx1
                    val bh = by2 - by1
                    val regionPixels = IntArray(bw * bh)
                    overlay.getPixels(regionPixels, 0, bw, bx1, by1, bw, bh)

                    var maskPixelCount = 0
                    for (py in 0 until bh) {
                        val protoY = ((by1 + py).toFloat() / imgH * ph).toInt().coerceIn(0, ph - 1)
                        for (px in 0 until bw) {
                            val protoX = ((bx1 + px).toFloat() / imgW * pw).toInt().coerceIn(0, pw - 1)
                            val idx = protoY * pw + protoX
                            var acc = 0f
                            for (k in 0 until coeffCount) {
                                acc += det.maskCoeffs[k] * protoArr[k * hw + idx]
                            }
                            if (sigmoid(acc) > 0.5f) {
                                // Blend: tint mask region with semi-transparent color
                                val orig = regionPixels[py * bw + px]
                                val r = ((orig shr 16 and 0xFF) * 3 + 255) / 4   // lighten toward white
                                val g = ((orig shr 8 and 0xFF) + 128) / 2
                                val b = ((orig and 0xFF) + 128) / 2
                                regionPixels[py * bw + px] = Color.rgb(r, g, b)
                                maskPixelCount++
                            }
                        }
                    }
                    overlay.setPixels(regionPixels, 0, bw, bx1, by1, bw, bh)
                    Log.i(TAG, "Det[$di]: bbox=[${bx1},${by1},${bx2},${by2}], maskPx=$maskPixelCount/${bw*bh}, score=${"%.2f".format(det.score)}")
                }
            }
        } else {
            Log.w(TAG, "No proto output available for mask decoding")
        }

        // --- Bounding boxes + labels ---
        for (det in selected) {
            val x1 = det.x1 * scaleX
            val y1 = det.y1 * scaleY
            val x2 = det.x2 * scaleX
            val y2 = det.y2 * scaleY
            canvas.drawRect(x1, y1, x2, y2, bboxPaint)

            val label = String.format("%.0f%%", det.score * 100)
            val textW = labelPaint.measureText(label)
            val textH = labelPaint.textSize
            val labelY = max(y1, textH + 2f)
            canvas.drawRect(x1, labelY - textH, x1 + textW + 8f, labelY + 4f, labelBgPaint)
            canvas.drawText(label, x1 + 4f, labelY, labelPaint)
        }

        return overlay
    }

    private fun sigmoid(x: Float): Float {
        val z = x.coerceIn(-50f, 50f)
        return (1f / (1f + exp(-z)))
    }

    private fun nsToMs(ns: Long): Float = (ns / 1_000_000.0).toFloat()
}
