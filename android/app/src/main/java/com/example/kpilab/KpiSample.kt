package com.example.kpilab

/**
 * Event type for KPI samples
 */
enum class EventType {
    INFERENCE,
    SYSTEM
}

/**
 * Single KPI measurement sample
 */
data class KpiSample(
    val timestamp: Long,
    val eventType: EventType,
    val latencyMs: Float? = null,
    val thermalC: Float? = null,
    val powerMw: Float? = null,
    val memoryMb: Int? = null,
    val isForeground: Boolean = true
) {
    companion object {
        fun inference(latencyMs: Float, isForeground: Boolean = true): KpiSample {
            return KpiSample(
                timestamp = System.currentTimeMillis(),
                eventType = EventType.INFERENCE,
                latencyMs = latencyMs,
                isForeground = isForeground
            )
        }

        fun system(
            thermalC: Float,
            powerMw: Float,
            memoryMb: Int,
            isForeground: Boolean = true
        ): KpiSample {
            return KpiSample(
                timestamp = System.currentTimeMillis(),
                eventType = EventType.SYSTEM,
                thermalC = thermalC,
                powerMw = powerMw,
                memoryMb = memoryMb,
                isForeground = isForeground
            )
        }
    }

    fun toCsvRow(): String {
        val latency = latencyMs?.let { String.format("%.2f", it) } ?: ""
        val thermal = thermalC?.let { String.format("%.1f", it) } ?: ""
        val power = powerMw?.let { String.format("%.1f", it) } ?: ""
        val memory = memoryMb?.toString() ?: ""
        val fg = if (isForeground) "true" else "false"

        return "$timestamp,${eventType.name},$latency,$thermal,$power,$memory,$fg"
    }
}

/**
 * Aggregated KPI statistics for display
 */
data class KpiStats(
    val inferenceCount: Int = 0,
    val latencyP50: Float = 0f,
    val latencyP95: Float = 0f,
    val latencyLast: Float = 0f,
    val thermalCurrent: Float = 0f,
    val thermalSlope: Float = 0f,  // °C per minute
    val powerAverage: Float = 0f,
    val memoryPeak: Int = 0
)
