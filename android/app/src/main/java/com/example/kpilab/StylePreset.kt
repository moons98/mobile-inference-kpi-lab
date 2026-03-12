package com.example.kpilab

/**
 * Text-to-image style presets.
 * Selected style suffix is appended to the user prompt before text encoding.
 */
enum class StylePreset(
    val displayName: String,
    val suffix: String
) {
    NONE("None", ""),
    PHOTOREALISTIC("Photo", "photorealistic, highly detailed, sharp focus, 8k uhd, DSLR"),
    ANIME("Anime", "anime style, illustration, vibrant colors, detailed"),
    OIL_PAINTING("Oil Paint", "oil painting, masterpiece, rich colors, textured brushstrokes"),
    WATERCOLOR("Watercolor", "watercolor painting, soft edges, delicate, artistic"),
    CINEMATIC("Cinematic", "cinematic lighting, dramatic atmosphere, film still, moody"),
    PIXEL_ART("Pixel Art", "pixel art style, retro, 16-bit");

    /** Combine user prompt with style suffix */
    fun applyTo(prompt: String): String {
        if (suffix.isEmpty()) return prompt
        val trimmed = prompt.trimEnd().trimEnd(',')
        return "$trimmed, $suffix"
    }
}
