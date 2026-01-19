package com.example.kpilab

/**
 * JNI wrapper for native inference runner
 */
class NativeRunner {

    companion object {
        init {
            System.loadLibrary("inference_kpi_lab")
        }
    }

    /**
     * Initialize the inference engine
     * @param modelPath Path to the model file
     * @param executionPath 0=NPU_ONLY, 1=NPU_FALLBACK, 2=GPU_ONLY
     * @param warmUpEnabled Whether to run warm-up iterations
     * @return true if initialization successful
     */
    external fun initialize(
        modelPath: String,
        executionPath: Int,
        warmUpEnabled: Boolean
    ): Boolean

    /**
     * Start a new logging session
     * @param sessionId Unique identifier for this session
     */
    external fun startSession(sessionId: String)

    /**
     * Run a single inference and log the result
     * @return Latency in milliseconds, or -1 if failed
     */
    external fun runInference(): Float

    /**
     * Log system metrics (called from Kotlin side)
     */
    external fun logSystemMetrics(thermalC: Float, powerMw: Float, memoryMb: Int)

    /**
     * Update foreground/background state
     */
    external fun setForeground(isForeground: Boolean)

    /**
     * End the current logging session
     */
    external fun endSession()

    /**
     * Export all logged data as CSV string
     */
    external fun exportCsv(): String

    /**
     * Get number of records logged
     */
    external fun getRecordCount(): Int

    /**
     * Release all native resources
     */
    external fun release()

    /**
     * Get device info summary (SoC name, HTP version)
     */
    external fun getDeviceInfo(): String

    /**
     * Check if HTP (NPU) is supported on this device
     */
    external fun isHtpSupported(): Boolean

    /**
     * Get HTP architecture version (66, 68, 69, 73, 75, etc.)
     * Returns 0 if unknown
     */
    external fun getHtpVersion(): Int
}
