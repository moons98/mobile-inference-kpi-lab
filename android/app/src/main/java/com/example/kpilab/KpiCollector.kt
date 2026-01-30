package com.example.kpilab

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.HardwarePropertiesManager
import android.util.Log
import java.io.BufferedReader
import java.io.File
import java.io.FileReader

/**
 * Collects system KPI metrics: thermal, power, memory
 */
class KpiCollector(private val context: Context) {

    companion object {
        private const val TAG = "KpiCollector"

        // Extended thermal zone paths for various devices
        private val THERMAL_PATHS = listOf(
            // Standard thermal zones (try multiple indices)
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/class/thermal/thermal_zone1/temp",
            "/sys/class/thermal/thermal_zone2/temp",
            "/sys/class/thermal/thermal_zone3/temp",
            "/sys/class/thermal/thermal_zone4/temp",
            "/sys/class/thermal/thermal_zone5/temp",
            // Virtual thermal
            "/sys/devices/virtual/thermal/thermal_zone0/temp",
            "/sys/devices/virtual/thermal/thermal_zone1/temp",
            // CPU specific paths (Qualcomm)
            "/sys/class/hwmon/hwmon0/temp1_input",
            "/sys/class/hwmon/hwmon1/temp1_input",
            // Samsung specific
            "/sys/class/sec/temperature/value",
            // MTK specific
            "/sys/class/thermal/thermal_zone9/temp"
        )

        // Memory info path
        private const val PROC_STATUS = "/proc/self/status"

        /**
         * Get SoC model name (e.g., "SM8550" for Snapdragon 8 Gen 2)
         */
        fun getSocModel(): String = Build.SOC_MODEL

        /**
         * Get SoC manufacturer
         */
        fun getSocManufacturer(): String = Build.SOC_MANUFACTURER

        /**
         * Get device model
         */
        fun getDeviceModel(): String = "${Build.MANUFACTURER} ${Build.MODEL}"

        /**
         * Get full device info string
         */
        fun getFullDeviceInfo(): String = buildString {
            appendLine("Device: ${Build.MANUFACTURER} ${Build.MODEL}")
            appendLine("SoC: ${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}")
            appendLine("Android: ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
            appendLine("Board: ${Build.BOARD}")
            appendLine("Hardware: ${Build.HARDWARE}")
        }

        /**
         * Get device info as map for CSV header
         */
        fun getDeviceInfoMap(): Map<String, String> = mapOf(
            "device_manufacturer" to Build.MANUFACTURER,
            "device_model" to Build.MODEL,
            "soc_manufacturer" to Build.SOC_MANUFACTURER,
            "soc_model" to Build.SOC_MODEL,
            "android_version" to Build.VERSION.RELEASE,
            "api_level" to Build.VERSION.SDK_INT.toString(),
            "board" to Build.BOARD,
            "hardware" to Build.HARDWARE
        )
    }

    private val batteryManager: BatteryManager =
        context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager

    private var thermalPath: String? = null
    private var useBatteryTemperature: Boolean = false

    init {
        // Find available thermal path that is readable
        thermalPath = findReadableThermalPath()

        if (thermalPath == null) {
            Log.w(TAG, "No thermal zone found, will use battery temperature as fallback")
            useBatteryTemperature = true
        } else {
            Log.i(TAG, "Using thermal path: $thermalPath")
        }

        // Log all available thermal zones for debugging
        logAvailableThermalZones()
    }

    private fun findReadableThermalPath(): String? {
        Log.i(TAG, "=== Searching for readable thermal path ===")
        for (path in THERMAL_PATHS) {
            val file = File(path)
            val exists = file.exists()
            val canRead = file.canRead()

            if (exists && canRead) {
                // Try to actually read it
                try {
                    val reader = BufferedReader(FileReader(path))
                    val value = reader.readLine()
                    reader.close()
                    val parsed = value?.trim()?.toIntOrNull()
                    Log.i(TAG, "  $path: exists=$exists, canRead=$canRead, value='$value', parsed=$parsed")
                    if (parsed != null) {
                        Log.i(TAG, "  -> Selected this path!")
                        return path
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "  $path: exists=$exists, canRead=$canRead, error=${e.message}")
                }
            } else {
                // Only log first few for brevity
                if (THERMAL_PATHS.indexOf(path) < 3) {
                    Log.i(TAG, "  $path: exists=$exists, canRead=$canRead")
                }
            }
        }
        Log.w(TAG, "=== No readable thermal path found ===")
        return null
    }

    private fun logAvailableThermalZones() {
        val thermalDir = File("/sys/class/thermal/")
        Log.i(TAG, "=== Thermal zones summary ===")
        if (thermalDir.exists() && thermalDir.isDirectory) {
            val zones = thermalDir.listFiles()?.filter { it.name.startsWith("thermal_zone") }
            Log.i(TAG, "Found ${zones?.size ?: 0} thermal zones")

            // Only log key zones (CPU, GPU, NPU related)
            val keyTypes = listOf("cpu", "gpu", "nsp", "aoss", "ddr")
            var keyZoneCount = 0

            zones?.forEach { zone ->
                try {
                    val typeFile = File(zone, "type")
                    val tempFile = File(zone, "temp")

                    if (typeFile.exists() && tempFile.exists() && tempFile.canRead()) {
                        val type = BufferedReader(FileReader(typeFile)).use { it.readLine() }
                        // Only log if it matches key types
                        if (keyTypes.any { type.lowercase().contains(it) }) {
                            val temp = try {
                                BufferedReader(FileReader(tempFile)).use { it.readLine() }
                            } catch (e: Exception) { "error" }
                            Log.i(TAG, "  ${zone.name}: type=$type, temp=$temp")
                            keyZoneCount++
                        }
                    }
                } catch (e: Exception) {
                    // Skip errors silently
                }
            }
            Log.i(TAG, "Logged $keyZoneCount key thermal zones (cpu/gpu/npu/aoss/ddr)")
        } else {
            Log.w(TAG, "Thermal directory does not exist")
        }

        // Also try battery temperature
        val batteryTemp = readBatteryTemperature()
        Log.i(TAG, "Battery temperature fallback: $batteryTemp °C")
        Log.i(TAG, "=============================")
    }

    /**
     * Read current CPU/SoC temperature in Celsius
     * Falls back to battery temperature if thermal zone unavailable
     * @return Temperature in °C, or -1 if unavailable
     */
    fun readThermal(): Float {
        // Try thermal zone first
        val path = thermalPath
        if (path != null) {
            try {
                val reader = BufferedReader(FileReader(path))
                val tempStr = reader.readLine()
                reader.close()

                val tempRaw = tempStr?.trim()?.toIntOrNull()
                if (tempRaw != null) {
                    // Most devices report in millidegrees Celsius
                    val result = if (tempRaw > 1000) {
                        tempRaw / 1000f
                    } else {
                        tempRaw.toFloat()
                    }
                    // Log occasionally (every ~100 calls would spam, so just first few)
                    return result
                } else {
                    Log.w(TAG, "readThermal: tempStr='$tempStr' could not be parsed")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to read thermal zone: ${e.message}")
            }
        }

        // Fallback: use battery temperature
        val batteryTemp = readBatteryTemperature()
        if (batteryTemp < 0) {
            Log.w(TAG, "readThermal: Both thermal zone and battery temp failed")
        }
        return batteryTemp
    }

    /**
     * Read battery temperature as fallback
     * @return Temperature in °C, or -1 if unavailable
     */
    private fun readBatteryTemperature(): Float {
        return try {
            val batteryStatus: Intent? = IntentFilter(Intent.ACTION_BATTERY_CHANGED).let { filter ->
                context.registerReceiver(null, filter)
            }
            // Battery temperature is in tenths of degrees Celsius
            val tempTenths = batteryStatus?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
            if (tempTenths > 0) {
                tempTenths / 10f
            } else {
                -1f
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read battery temperature: ${e.message}")
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

            // Get voltage - use broadcast intent for compatibility (works on all API levels)
            val batteryStatus: Intent? = IntentFilter(Intent.ACTION_BATTERY_CHANGED).let { filter ->
                context.registerReceiver(null, filter)
            }
            val voltageMv = batteryStatus?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) ?: -1

            if (currentMicroAmps == Int.MIN_VALUE || voltageMv <= 0) {
                Log.w(TAG, "Battery properties unavailable")
                return -1f
            }

            // Some devices report current as negative when discharging
            val absCurrent = kotlin.math.abs(currentMicroAmps)

            // Power (mW) = |Current (μA)| × Voltage (mV) / 10^6
            val powerMw = (absCurrent.toLong() * voltageMv.toLong()) / 1_000_000f

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
