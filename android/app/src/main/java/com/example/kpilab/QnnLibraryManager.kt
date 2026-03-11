package com.example.kpilab

import android.content.Context
import android.os.Build
import android.util.Log

/**
 * QNN library path manager for HTP (NPU) execution.
 *
 * ORT 1.24.3+ bundles QNN SDK 2.42.0 in the AAR's native libs.
 * This manager provides fallback path resolution for non-bundled environments.
 */
object QnnLibraryManager {
    private const val TAG = "QnnLibraryManager"

    const val QNN_SDK_VERSION = "2.42.0"

    // Hexagon architecture for different SoCs
    private val SOC_TO_HEXAGON = mapOf(
        "SM8750" to "V79",  // Snapdragon 8 Elite
        "SM8650" to "V75",  // Snapdragon 8 Gen 3
        "SM8550" to "V73",  // Snapdragon 8 Gen 2
        "SM8475" to "V69",  // Snapdragon 8+ Gen 1
        "SM8450" to "V69",  // Snapdragon 8 Gen 1
        "SM8350" to "V68",  // Snapdragon 888
    )

    private var qnnLibPath: String? = null

    fun getHexagonVersion(): String? = SOC_TO_HEXAGON[Build.SOC_MODEL]

    /**
     * Initialize QNN library path. Returns the native lib directory
     * containing QNN libraries (bundled in ORT AAR).
     */
    fun initialize(context: Context): String? {
        if (qnnLibPath != null) return qnnLibPath

        val nativeLibDir = context.applicationInfo.nativeLibraryDir
        val bundledQnn = java.io.File("$nativeLibDir/libQnnHtp.so")
        if (bundledQnn.exists()) {
            qnnLibPath = nativeLibDir
            Log.i(TAG, "QNN libs (ORT-bundled): $nativeLibDir, Hexagon: ${getHexagonVersion() ?: "unknown"}")
            return qnnLibPath
        }

        Log.w(TAG, "QNN libraries not found in native lib dir: $nativeLibDir")
        return null
    }

    fun getLibraryPath(): String? = qnnLibPath
}
