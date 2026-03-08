package com.example.kpilab

/**
 * Input source mode for inference pipeline.
 */
enum class InputMode(val displayName: String) {
    /** CameraX live frame every iteration (production simulation) */
    CAMERA_LIVE("Camera Live"),

    /** Capture one camera frame, reuse for all iterations (controlled variable) */
    CAMERA_SINGLE("Camera Single"),

    /** Use sample_image.jpg from assets (fallback, no camera needed) */
    STATIC_IMAGE("Static Image")
}

/**
 * Benchmark phase defining the execution strategy.
 */
enum class BenchmarkPhase(val displayName: String) {
    /** Burst latency: 100 iterations at 2 Hz, Camera Single input */
    BURST("Phase 1: Burst Latency"),

    /** Sustained throughput: 30 Hz target for 5 min, Camera Live input */
    SUSTAINED("Phase 2: Sustained Throughput")
}
