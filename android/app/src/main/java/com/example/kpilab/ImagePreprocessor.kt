package com.example.kpilab

import android.graphics.Bitmap

/**
 * Image postprocessing for text-to-image pipeline.
 * Converts VAE Decoder output (NCHW [-1,1]) to Bitmap.
 */
object ImagePreprocessor {

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
}
