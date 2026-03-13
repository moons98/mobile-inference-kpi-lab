package com.example.kpilab

import android.graphics.Bitmap

/**
 * Image postprocessing for text-to-image pipeline.
 * Converts VAE Decoder output (NCHW) to Bitmap.
 */
object ImagePreprocessor {

    /**
     * Convert VAE Decoder output (NCHW) to Bitmap.
     *
     * @param normalized true  → VAE output is [0,1]  (W8A16 compiled model: /Div+/Clip baked in)
     *                   false → VAE output is [-1,1] (FP32/FP16/W8A8 models)
     */
    fun postprocess(output: FloatArray, height: Int, width: Int, normalized: Boolean = false): Bitmap {
        val planeSize = height * width
        val pixels = IntArray(planeSize)

        for (i in 0 until planeSize) {
            val r: Int
            val g: Int
            val b: Int
            if (normalized) {
                r = (output[i].coerceIn(0f, 1f) * 255f).toInt()
                g = (output[planeSize + i].coerceIn(0f, 1f) * 255f).toInt()
                b = (output[2 * planeSize + i].coerceIn(0f, 1f) * 255f).toInt()
            } else {
                r = ((output[i].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
                g = ((output[planeSize + i].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
                b = ((output[2 * planeSize + i].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
            }
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }
}
