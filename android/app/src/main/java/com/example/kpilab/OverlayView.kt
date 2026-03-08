package com.example.kpilab

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * Overlay view for drawing detection bounding boxes on camera preview.
 * Used in demo mode only; does not affect benchmark timing.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<Detection> = emptyList()
    private var sourceWidth: Int = 640
    private var sourceHeight: Int = 640

    /** Last onDraw() duration in ms — read by BenchmarkRunner for E2E pipeline measurement */
    @Volatile
    var lastDrawMs: Float = 0f
        private set

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        isAntiAlias = true
    }

    private val textBackgroundPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        isAntiAlias = true
    }

    // Colors for different classes
    private val colors = intArrayOf(
        Color.rgb(255, 0, 0),     // Red
        Color.rgb(0, 255, 0),     // Green
        Color.rgb(0, 0, 255),     // Blue
        Color.rgb(255, 255, 0),   // Yellow
        Color.rgb(255, 0, 255),   // Magenta
        Color.rgb(0, 255, 255),   // Cyan
        Color.rgb(255, 128, 0),   // Orange
        Color.rgb(128, 0, 255),   // Purple
    )

    fun setDetections(detections: List<Detection>, sourceW: Int = 640, sourceH: Int = 640) {
        this.detections = detections
        this.sourceWidth = sourceW
        this.sourceHeight = sourceH
        postInvalidate()
    }

    fun clear() {
        this.detections = emptyList()
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty()) {
            lastDrawMs = 0f
            return
        }

        val t0 = System.nanoTime()

        // Match PreviewView's FILL_CENTER scaling: uniform scale + center offset.
        // This ensures bounding boxes align with the camera preview regardless of
        // aspect ratio differences between source image and view.
        val scale = maxOf(width.toFloat() / sourceWidth, height.toFloat() / sourceHeight)
        val offsetX = (width - sourceWidth * scale) / 2f
        val offsetY = (height - sourceHeight * scale) / 2f

        for (det in detections) {
            val color = colors[det.classId % colors.size]
            boxPaint.color = color
            textBackgroundPaint.color = color

            val rect = RectF(
                det.x1 * scale + offsetX,
                det.y1 * scale + offsetY,
                det.x2 * scale + offsetX,
                det.y2 * scale + offsetY
            )

            // Draw bounding box
            canvas.drawRect(rect, boxPaint)

            // Draw label
            val label = "${det.className} ${(det.confidence * 100).toInt()}%"
            val textWidth = textPaint.measureText(label)
            val textHeight = textPaint.textSize

            canvas.drawRect(
                rect.left, rect.top - textHeight - 4,
                rect.left + textWidth + 8, rect.top,
                textBackgroundPaint
            )
            canvas.drawText(label, rect.left + 4, rect.top - 4, textPaint)
        }

        lastDrawMs = ((System.nanoTime() - t0) / 1_000_000.0).toFloat()
    }
}
