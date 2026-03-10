package com.example.kpilab

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Image preprocessing/postprocessing for AI Eraser pipeline.
 * Handles: NCHW conversion, ROI crop/uncrop, mask processing, alpha feathering, blend.
 */
object ImagePreprocessor {

    // ── SD VAE input/output conversion ──

    /**
     * Preprocess a bitmap for VAE Encoder input.
     * Resize → normalize to [-1, 1] → NCHW float tensor.
     */
    fun preprocess(bitmap: Bitmap, targetSize: Int): Pair<FloatBuffer, LongArray> {
        val resized = Bitmap.createScaledBitmap(bitmap, targetSize, targetSize, true)
        val pixels = IntArray(targetSize * targetSize)
        resized.getPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize)
        if (resized !== bitmap) resized.recycle()

        val planeSize = targetSize * targetSize
        val buffer = FloatBuffer.allocate(3 * planeSize)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            buffer.put(i, ((pixel shr 16) and 0xFF) / 127.5f - 1.0f)              // R
            buffer.put(planeSize + i, ((pixel shr 8) and 0xFF) / 127.5f - 1.0f)   // G
            buffer.put(2 * planeSize + i, (pixel and 0xFF) / 127.5f - 1.0f)       // B
        }

        val shape = longArrayOf(1, 3, targetSize.toLong(), targetSize.toLong())
        return Pair(buffer, shape)
    }

    /**
     * Convert VAE Decoder output (NCHW [-1,1]) back to Bitmap.
     */
    fun postprocess(output: FloatArray, height: Int, width: Int): Bitmap {
        val planeSize = height * width
        val pixels = IntArray(planeSize)

        for (i in 0 until planeSize) {
            val r = ((output[i].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
            val g = ((output[planeSize + i].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
            val b = ((output[2 * planeSize + i].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    // ── ROI Crop / Uncrop ──

    /**
     * ROI crop 결과.
     * @param roiBitmap Cropped ROI image (정사각형, 아직 512 resize 전)
     * @param roiMask Cropped mask (동일 크기)
     * @param cropRect 원본 이미지 기준 crop 영역 (정사각형)
     */
    data class RoiCropResult(
        val roiBitmap: Bitmap,
        val roiMask: Bitmap,
        val cropRect: Rect
    )

    /**
     * Mask의 bounding box를 추출하고, padding 확장 → 정사각형 보정 → crop.
     *
     * @param originalImage 원본 갤러리 이미지
     * @param mask Binary mask (0=배경, 255=객체). 원본과 동일 크기.
     * @param paddingRatio bbox 대비 padding 비율 (e.g., 1.5 = 50% 확장)
     * @return 정사각형 ROI crop 결과 (null if mask empty)
     */
    fun cropRoiWithPadding(
        originalImage: Bitmap,
        mask: Bitmap,
        paddingRatio: Float = 1.5f
    ): RoiCropResult? {
        // 1. Mask에서 bbox 추출
        val bbox = extractMaskBbox(mask) ?: return null

        // 2. Padding 확장
        val bboxW = bbox.width()
        val bboxH = bbox.height()
        val padW = ((bboxW * paddingRatio - bboxW) / 2).roundToInt()
        val padH = ((bboxH * paddingRatio - bboxH) / 2).roundToInt()
        val padded = Rect(
            bbox.left - padW,
            bbox.top - padH,
            bbox.right + padW,
            bbox.bottom + padH
        )

        // 3. 정사각형 보정 (긴 쪽에 맞춤)
        val side = max(padded.width(), padded.height())
        val cx = padded.centerX()
        val cy = padded.centerY()
        val square = Rect(
            cx - side / 2,
            cy - side / 2,
            cx - side / 2 + side,
            cy - side / 2 + side
        )

        // 4. 이미지 경계 클리핑
        val clipped = Rect(
            max(0, square.left),
            max(0, square.top),
            min(originalImage.width, square.right),
            min(originalImage.height, square.bottom)
        )

        if (clipped.width() <= 0 || clipped.height() <= 0) return null

        // 5. Crop (이미지 경계 밖 부분은 검정으로 패딩)
        val cropBitmap = cropWithPadding(originalImage, square)
        val cropMask = cropWithPadding(mask, square)

        return RoiCropResult(cropBitmap, cropMask, square)
    }

    /**
     * Mask에서 non-zero pixel의 bounding box 추출.
     */
    private fun extractMaskBbox(mask: Bitmap): Rect? {
        val w = mask.width
        val h = mask.height
        val pixels = IntArray(w * h)
        mask.getPixels(pixels, 0, w, 0, 0, w, h)

        var minX = w; var minY = h; var maxX = 0; var maxY = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val pixel = pixels[y * w + x]
                // Check any channel > 0 (mask could be grayscale or binary)
                if ((pixel and 0x00FFFFFF) != 0) {
                    minX = min(minX, x)
                    minY = min(minY, y)
                    maxX = max(maxX, x)
                    maxY = max(maxY, y)
                }
            }
        }

        return if (maxX >= minX && maxY >= minY) Rect(minX, minY, maxX + 1, maxY + 1) else null
    }

    /**
     * 정사각형 영역으로 crop. 이미지 경계 밖은 검정(0)으로 채움.
     */
    private fun cropWithPadding(source: Bitmap, rect: Rect): Bitmap {
        val side = max(rect.width(), rect.height())
        val result = Bitmap.createBitmap(side, side, source.config ?: Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        canvas.drawColor(Color.BLACK)

        // source에서 실제로 복사할 영역
        val srcRect = Rect(
            max(0, rect.left),
            max(0, rect.top),
            min(source.width, rect.right),
            min(source.height, rect.bottom)
        )
        // result에서의 대응 위치
        val dstRect = Rect(
            srcRect.left - rect.left,
            srcRect.top - rect.top,
            srcRect.left - rect.left + srcRect.width(),
            srcRect.top - rect.top + srcRect.height()
        )

        canvas.drawBitmap(source, srcRect, dstRect, null)
        return result
    }

    // ── Mask Processing ──

    /**
     * Binary mask를 latent space 크기(64×64)로 resize.
     * 반환: FloatArray [1, 1, 64, 64], 값 0.0 or 1.0.
     */
    fun resizeMaskToLatent(mask: Bitmap, latentSize: Int = 64): Pair<FloatBuffer, LongArray> {
        val resized = Bitmap.createScaledBitmap(mask, latentSize, latentSize, true)
        val pixels = IntArray(latentSize * latentSize)
        resized.getPixels(pixels, 0, latentSize, 0, 0, latentSize, latentSize)
        if (resized !== mask) resized.recycle()

        val buffer = FloatBuffer.allocate(latentSize * latentSize)
        for (i in pixels.indices) {
            // Threshold: > 128 → 1.0 (mask area), else 0.0
            val gray = (pixels[i] and 0xFF)
            buffer.put(i, if (gray > 128) 1.0f else 0.0f)
        }

        val shape = longArrayOf(1, 1, latentSize.toLong(), latentSize.toLong())
        return Pair(buffer, shape)
    }

    /**
     * Masked image 생성: image × (1 − mask).
     * mask에서 1인 영역(객체)을 0으로 지운 이미지를 반환.
     *
     * @param roiBitmap ROI 이미지 (512×512)
     * @param roiMask ROI mask (512×512, 255=객체)
     * @return masked image bitmap (객체 영역이 검정인 이미지)
     */
    fun createMaskedImage(roiBitmap: Bitmap, roiMask: Bitmap): Bitmap {
        val w = roiBitmap.width
        val h = roiBitmap.height
        val imgPixels = IntArray(w * h)
        val maskPixels = IntArray(w * h)
        roiBitmap.getPixels(imgPixels, 0, w, 0, 0, w, h)
        roiMask.getPixels(maskPixels, 0, w, 0, 0, w, h)

        val result = IntArray(w * h)
        for (i in imgPixels.indices) {
            val maskVal = maskPixels[i] and 0xFF  // grayscale value
            if (maskVal > 128) {
                // Object area → zero out
                result[i] = 0xFF000000.toInt()
            } else {
                // Background → keep original
                result[i] = imgPixels[i]
            }
        }

        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(result, 0, w, 0, 0, w, h)
        return bitmap
    }

    // ── Composite / Blend ──

    /**
     * Inpainted ROI를 원본에 합성.
     * 1. inpainted ROI (512²) → 원본 ROI 크기로 resize
     * 2. mask 경계 alpha feathering
     * 3. 원본 이미지에 blending
     *
     * @param originalImage 원본 갤러리 이미지
     * @param inpaintedRoi Inpainted 결과 (512×512)
     * @param cropRect 원본 기준 crop 영역 (정사각형)
     * @param roiMask ROI mask (512×512 또는 crop 크기, 255=inpaint 영역)
     * @param featherRadius Feathering 반경 (pixels, crop 좌표계)
     * @return 최종 합성 이미지
     */
    fun composite(
        originalImage: Bitmap,
        inpaintedRoi: Bitmap,
        cropRect: Rect,
        roiMask: Bitmap,
        featherRadius: Int = 16
    ): Bitmap {
        val cropW = cropRect.width()
        val cropH = cropRect.height()

        // 1. Inpainted ROI → 원본 crop 크기로 resize
        val resizedRoi = Bitmap.createScaledBitmap(inpaintedRoi, cropW, cropH, true)
        val resizedMask = Bitmap.createScaledBitmap(roiMask, cropW, cropH, true)

        // 2. Feathered alpha map 생성
        val alphaMap = createFeatheredAlpha(resizedMask, featherRadius)

        // 3. 원본 복사 후 blending
        val result = originalImage.copy(Bitmap.Config.ARGB_8888, true)
        val roiPixels = IntArray(cropW * cropH)
        resizedRoi.getPixels(roiPixels, 0, cropW, 0, 0, cropW, cropH)

        val origW = result.width
        val origH = result.height

        for (y in 0 until cropH) {
            val origY = cropRect.top + y
            if (origY < 0 || origY >= origH) continue
            for (x in 0 until cropW) {
                val origX = cropRect.left + x
                if (origX < 0 || origX >= origW) continue

                val alpha = alphaMap[y * cropW + x]
                if (alpha <= 0f) continue

                val roiPixel = roiPixels[y * cropW + x]
                val origPixel = result.getPixel(origX, origY)

                // Alpha blend
                val r = lerp((origPixel shr 16) and 0xFF, (roiPixel shr 16) and 0xFF, alpha)
                val g = lerp((origPixel shr 8) and 0xFF, (roiPixel shr 8) and 0xFF, alpha)
                val b = lerp(origPixel and 0xFF, roiPixel and 0xFF, alpha)

                result.setPixel(origX, origY, (0xFF shl 24) or (r shl 16) or (g shl 8) or b)
            }
        }

        resizedRoi.recycle()
        resizedMask.recycle()
        return result
    }

    /**
     * Mask 경계에 alpha feathering 적용.
     * mask 내부 = 1.0, 경계 = gradient, 외부 = 0.0.
     */
    private fun createFeatheredAlpha(mask: Bitmap, radius: Int): FloatArray {
        val w = mask.width
        val h = mask.height
        val pixels = IntArray(w * h)
        mask.getPixels(pixels, 0, w, 0, 0, w, h)

        // Binary mask → distance map (approximate with erosion iterations)
        val binary = FloatArray(w * h) { if ((pixels[it] and 0xFF) > 128) 1f else 0f }

        if (radius <= 0) return binary

        // Simple box blur of binary mask for feathering effect
        val alpha = FloatArray(w * h)
        val r = radius
        for (y in 0 until h) {
            for (x in 0 until w) {
                if (binary[y * w + x] <= 0f) {
                    alpha[y * w + x] = 0f
                    continue
                }
                // Distance to nearest 0 pixel (approximate)
                var minDist = r + 1
                for (dy in -r..r) {
                    for (dx in -r..r) {
                        val ny = y + dy
                        val nx = x + dx
                        if (ny in 0 until h && nx in 0 until w) {
                            if (binary[ny * w + nx] <= 0f) {
                                val dist = kotlin.math.sqrt((dx * dx + dy * dy).toFloat()).toInt()
                                minDist = min(minDist, dist)
                            }
                        }
                    }
                }
                alpha[y * w + x] = if (minDist > r) 1f else (minDist.toFloat() / r).coerceIn(0f, 1f)
            }
        }

        return alpha
    }

    private fun lerp(a: Int, b: Int, t: Float): Int {
        return (a + (b - a) * t).roundToInt().coerceIn(0, 255)
    }
}
