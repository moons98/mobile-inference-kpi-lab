package com.example.kpilab

import android.content.Context
import android.os.BatteryManager
import android.util.Log
import java.io.BufferedReader
import java.io.File
import java.io.FileReader

/**
 * Collects system KPI metrics: thermal, power, memory
 */
class KpiCollector(context: Context) {

    companion object {
        private const val TAG = "KpiCollector"

        // Common thermal zone paths on Qualcomm devices
        private val THERMAL_PATHS = listOf(
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/class/thermal/thermal_zone1/temp",
            "/sys/devices/virtual/thermal/thermal_zone0/temp"
        )

        // Memory info path
        private const val PROC_STATUS = "/proc/self/status"
    }

    private val batteryManager: BatteryManager =
        context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager

    private var thermalPath: String? = null

    init {
        // Find available thermal path
        thermalPath = THERMAL_PATHS.firstOrNull { File(it).exists() }
        if (thermalPath == null) {
            Log.w(TAG, "No thermal zone found, thermal readings will be unavailable")
        } else {
            Log.i(TAG, "Using thermal path: $thermalPath")
        }
    }

    /**
     * Read current CPU/SoC temperature in Celsius
     * @return Temperature in °C, or -1 if unavailable
     */
    fun readThermal(): Float {
        val path = thermalPath ?: return -1f

        return try {
            val reader = BufferedReader(FileReader(path))
            val tempStr = reader.readLine()
            reader.close()

            val tempRaw = tempStr.trim().toIntOrNull() ?: return -1f

            // Most devices report in millidegrees Celsius
            if (tempRaw > 1000) {
                tempRaw / 1000f
            } else {
                tempRaw.toFloat()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read thermal: ${e.message}")
            -1f
        }
    }

    /**
     * Read current power consumption in milliwatts
     * Uses BatteryManager API - accuracy varies by device
     * @return Power in mW, or -1 if unavailable
     */
    fun readPower(): Float {
        return try {
            // Get current in microamps (negative = discharging)
            val currentMicroAmps = batteryManager.getIntProperty(
                BatteryManager.BATTERY_PROPERTY_CURRENT_NOW
            )

            // Get voltage in microvolts
            val voltageMicroVolts = batteryManager.getIntProperty(
                BatteryManager.BATTERY_PROPERTY_VOLTAGE_NOW
            )

            if (currentMicroAmps == Int.MIN_VALUE || voltageMicroVolts == Int.MIN_VALUE) {
                Log.w(TAG, "Battery properties unavailable")
                return -1f
            }

            // Some devices report current as negative when discharging
            val absCurrent = kotlin.math.abs(currentMicroAmps)

            // Power (mW) = |Current (μA)| × Voltage (μV) / 10^9
            val powerMw = (absCurrent.toLong() * voltageMicroVolts.toLong()) / 1_000_000_000f

            powerMw
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read power: ${e.message}")
            -1f
        }
    }

    /**
     * Read current memory usage (VmRSS) in megabytes
     * @return Memory in MB, or -1 if unavailable
     */
    fun readMemory(): Int {
        return try {
            val reader = BufferedReader(FileReader(PROC_STATUS))
            var line: String?
            var vmRssKb = -1

            while (reader.readLine().also { line = it } != null) {
                if (line!!.startsWith("VmRSS:")) {
                    // Format: "VmRSS:     12345 kB"
                    val parts = line!!.trim().split(Regex("\\s+"))
                    if (parts.size >= 2) {
                        vmRssKb = parts[1].toIntOrNull() ?: -1
                    }
                    break
                }
            }
            reader.close()

            if (vmRssKb > 0) {
                vmRssKb / 1024  // Convert to MB
            } else {
                -1
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read memory: ${e.message}")
            -1
        }
    }

    /**
     * Collect all system metrics at once
     */
    fun collectAll(): SystemMetrics {
        return SystemMetrics(
            thermalC = readThermal(),
            powerMw = readPower(),
            memoryMb = readMemory()
        )
    }

    data class SystemMetrics(
        val thermalC: Float,
        val powerMw: Float,
        val memoryMb: Int
    )
}
