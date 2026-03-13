package com.example.kpilab

import android.graphics.Bitmap

/**
 * Image postprocessing for text-to-image pipeline.
 * Converts VAE Decoder output (NCHW) to Bitmap.
 */
object ImagePreprocessor {

    /**
     * Convert VAE Decoder output to Bitmap.
     *
     * @param normalized true  → output is [0,1]  (W8A16: /Div+/Clip baked in)
     *                   false → output is [-1,1] (FP32/FP16/W8A8)
     * @param hwc        true  → layout is NHWC [1,H,W,3] (qai-hub-models W8A16 export)
     *                   false → layout is NCHW [1,3,H,W] (standard diffusers export)
     */
    fun postprocess(
        output: FloatArray, height: Int, width: Int,
        normalized: Boolean = false, hwc: Boolean = false
    ): Bitmap {
        val planeSize = height * width
        val pixels = IntArray(planeSize)

        for (i in 0 until planeSize) {
            val rv: Float
            val gv: Float
            val bv: Float
            if (hwc) {
                rv = output[i * 3]
                gv = output[i * 3 + 1]
                bv = output[i * 3 + 2]
            } else {
                rv = output[i]
                gv = output[planeSize + i]
                bv = output[2 * planeSize + i]
            }
            val r: Int
            val g: Int
            val b: Int
            if (normalized) {
                r = (rv.coerceIn(0f, 1f) * 255f).toInt()
                g = (gv.coerceIn(0f, 1f) * 255f).toInt()
                b = (bv.coerceIn(0f, 1f) * 255f).toInt()
            } else {
                r = ((rv.coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
                g = ((gv.coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
                b = ((bv.coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
            }
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }
}
