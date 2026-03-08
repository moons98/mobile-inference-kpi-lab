package com.example.kpilab

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference

/**
 * Manages CameraX lifecycle and provides frames for inference pipeline.
 *
 * Supports two acquisition modes:
 * - Live: returns latest camera frame each call
 * - Single: captures one frame and caches it for repeated use
 */
class CameraManager(private val context: Context) {

    companion object {
        private const val TAG = "CameraManager"
        private const val TARGET_WIDTH = 640
        private const val TARGET_HEIGHT = 480
    }

    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    // Latest frame from camera (thread-safe) — Pair(bitmap, conversionMs)
    private val latestFrame = AtomicReference<Pair<Bitmap, Float>?>(null)

    // Cached single frame for CAMERA_SINGLE mode
    private var cachedSingleFrame: Bitmap? = null

    // Flag indicating camera is actively providing frames
    @Volatile
    var isRunning: Boolean = false
        private set

    /**
     * Start camera with optional preview.
     * @param lifecycleOwner Activity or Fragment lifecycle
     * @param previewView Optional PreviewView for demo mode overlay
     */
    fun start(lifecycleOwner: LifecycleOwner, previewView: PreviewView? = null) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()
            cameraProvider = provider

            // Build use cases
            val useCases = mutableListOf<androidx.camera.core.UseCase>()

            // Preview (only if previewView provided for demo mode)
            if (previewView != null) {
                val preview = Preview.Builder()
                    .build()
                    .also { it.setSurfaceProvider(previewView.surfaceProvider) }
                useCases.add(preview)
            }

            // ImageAnalysis for frame acquisition
            imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(TARGET_WIDTH, TARGET_HEIGHT))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }
            useCases.add(imageAnalysis!!)

            // Bind to lifecycle
            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    *useCases.toTypedArray()
                )
                isRunning = true
                Log.i(TAG, "Camera started with ${useCases.size} use cases")
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed: ${e.message}", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Stop camera and release resources.
     */
    fun stop() {
        cameraProvider?.unbindAll()
        isRunning = false
        cachedSingleFrame = null
        latestFrame.set(null)
        Log.i(TAG, "Camera stopped")
    }

    /**
     * Acquire a frame for inference.
     * Returns Pair(bitmap, acquireMs) where acquireMs is the YUV→Bitmap conversion time.
     *
     * @param mode Input mode determining frame source
     * @return Frame bitmap and acquisition time in ms, or null if no frame available
     */
    fun acquireFrame(mode: InputMode): Pair<Bitmap, Float>? {
        return when (mode) {
            InputMode.CAMERA_LIVE -> {
                // Return latest frame with its measured conversion time
                val frame = latestFrame.getAndSet(null)
                frame // Pair(bitmap, conversionMs) or null
            }
            InputMode.CAMERA_SINGLE -> {
                // Return cached frame, or capture first available
                val cached = cachedSingleFrame
                if (cached != null) {
                    Pair(cached, 0f) // No acquire cost for cached frame
                } else {
                    val frame = latestFrame.getAndSet(null)
                    if (frame != null) {
                        cachedSingleFrame = frame.first
                        frame
                    } else {
                        null
                    }
                }
            }
            InputMode.STATIC_IMAGE -> {
                // Not handled by CameraManager
                null
            }
        }
    }

    /**
     * Wait for a frame to become available (blocking, with timeout).
     * Useful for Camera Single mode during initialization.
     *
     * @param mode Input mode
     * @param timeoutMs Maximum wait time in ms
     * @return Frame bitmap and acquire time, or null on timeout
     */
    fun acquireFrameBlocking(mode: InputMode, timeoutMs: Long = 5000): Pair<Bitmap, Float>? {
        if (mode == InputMode.STATIC_IMAGE) return null

        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            val frame = acquireFrame(mode)
            if (frame != null) return frame
            Thread.sleep(10)
        }
        Log.w(TAG, "acquireFrameBlocking timed out after ${timeoutMs}ms")
        return null
    }

    /**
     * Process incoming camera frame: YUV420 → ARGB Bitmap.
     * Stores result in latestFrame for consumption by acquireFrame().
     */
    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val t0 = System.nanoTime()
            val bitmap = imageProxyToBitmap(imageProxy)
            val conversionMs = ((System.nanoTime() - t0) / 1_000_000.0).toFloat()
            if (bitmap != null) {
                latestFrame.set(Pair(bitmap, conversionMs))
            }
        } catch (e: Exception) {
            Log.w(TAG, "Frame conversion failed: ${e.message}")
        } finally {
            imageProxy.close()
        }
    }

    /**
     * Convert ImageProxy (YUV_420_888) to ARGB Bitmap.
     */
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 95, out)
        val jpegBytes = out.toByteArray()
        var bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size) ?: return null

        // Apply rotation if needed
        val rotation = imageProxy.imageInfo.rotationDegrees
        if (rotation != 0) {
            val matrix = Matrix()
            matrix.postRotate(rotation.toFloat())
            val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (rotated !== bitmap) bitmap.recycle()
            bitmap = rotated
        }

        return bitmap
    }

    fun release() {
        stop()
        analysisExecutor.shutdown()
    }
}
