package com.example.kpilab

import android.content.Context
import android.os.Build
import android.system.Os
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * Manages QNN library extraction and loading for HTP (NPU) execution.
 *
 * QNN libraries are bundled in assets/qnn_libs/ and extracted to app internal storage.
 * This allows the DSP to load Skel libraries from a path the app controls.
 */
object QnnLibraryManager {
    private const val TAG = "QnnLibraryManager"

    // QNN SDK version
    // ORT 1.24.3 bundles QNN SDK 2.42.0 in the AAR
    const val QNN_SDK_VERSION = "2.42.0"
    const val ORT_EXPECTED_QNN_VERSION = "2.42.0"

    // Hexagon architecture for different SoCs
    private val SOC_TO_HEXAGON = mapOf(
        "SM8750" to "V79",  // Snapdragon 8 Elite
        "SM8650" to "V75",  // Snapdragon 8 Gen 3
        "SM8550" to "V73",  // Snapdragon 8 Gen 2
        "SM8475" to "V69",  // Snapdragon 8+ Gen 1
        "SM8450" to "V69",  // Snapdragon 8 Gen 1
        "SM8350" to "V68",  // Snapdragon 888
    )

    // Required QNN libraries
    private val QNN_LIBRARIES = listOf(
        "libQnnHtp.so",
        "libQnnSystem.so",
        "libQnnHtpV73Stub.so",
        "libQnnHtpV73Skel.so"
    )

    private var initialized = false
    private var qnnLibPath: String? = null

    /**
     * Get the Hexagon version for the current device's SoC.
     */
    fun getHexagonVersion(): String? {
        val socModel = Build.SOC_MODEL
        return SOC_TO_HEXAGON[socModel]
    }

    /**
     * Check if the current device supports QNN HTP.
     */
    fun isDeviceSupported(): Boolean {
        return getHexagonVersion() != null
    }

    /**
     * Initialize QNN libraries by extracting from assets and setting up paths.
     * Must be called before creating ORT sessions with QNN EP.
     *
     * @return The path to QNN libraries, or null if initialization failed
     */
    fun initialize(context: Context): String? {
        if (initialized && qnnLibPath != null) {
            Log.i(TAG, "Already initialized, QNN lib path: $qnnLibPath")
            return qnnLibPath
        }

        val hexagonVersion = getHexagonVersion()
        if (hexagonVersion == null) {
            Log.w(TAG, "Device SoC ${Build.SOC_MODEL} not in supported list")
            // Continue anyway - might still work
        } else {
            Log.i(TAG, "Device: ${Build.SOC_MODEL}, Hexagon: $hexagonVersion")
        }

        // Create QNN library directory in app internal storage
        val libDir = File(context.filesDir, "qnn_libs")
        if (!libDir.exists()) {
            libDir.mkdirs()
        }

        // Extract libraries from assets
        val success = extractLibraries(context, libDir)
        if (!success) {
            Log.e(TAG, "Failed to extract QNN libraries")
            return null
        }

        // Set ADSP_LIBRARY_PATH environment variable
        // This tells FastRPC where to find Skel libraries for DSP
        try {
            val currentPath = System.getenv("ADSP_LIBRARY_PATH") ?: ""
            val newPath = if (currentPath.isEmpty()) {
                libDir.absolutePath
            } else {
                "${libDir.absolutePath}:$currentPath"
            }
            Os.setenv("ADSP_LIBRARY_PATH", newPath, true)
            Log.i(TAG, "ADSP_LIBRARY_PATH set to: $newPath")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set ADSP_LIBRARY_PATH: ${e.message}")
            // Continue anyway - skel_library_dir option might work
        }

        qnnLibPath = libDir.absolutePath
        initialized = true

        Log.i(TAG, "QNN libraries initialized at: $qnnLibPath")
        logLibraryInfo(libDir)

        return qnnLibPath
    }

    /**
     * Get the QNN library path. Returns null if not initialized.
     */
    fun getLibraryPath(): String? = qnnLibPath

    /**
     * Extract QNN libraries from assets to the target directory.
     */
    private fun extractLibraries(context: Context, targetDir: File): Boolean {
        val assetManager = context.assets

        try {
            val assetFiles = assetManager.list("qnn_libs") ?: emptyArray()
            Log.i(TAG, "Found ${assetFiles.size} files in assets/qnn_libs/")

            val selectedFiles = assetFiles.filter { it in QNN_LIBRARIES }
            for (filename in selectedFiles) {
                val targetFile = File(targetDir, filename)

                // Overwrite to avoid stale/partial files after app updates.
                if (targetFile.exists()) {
                    targetFile.delete()
                }

                // Extract from assets
                Log.i(TAG, "Extracting: $filename")
                assetManager.open("qnn_libs/$filename").use { input ->
                    FileOutputStream(targetFile).use { output ->
                        input.copyTo(output)
                    }
                }

                // Set executable permission
                targetFile.setExecutable(true, false)
                targetFile.setReadable(true, false)

                Log.i(TAG, "Extracted: $filename (${targetFile.length()} bytes)")
            }

            val missing = QNN_LIBRARIES.filterNot { File(targetDir, it).exists() }
            if (missing.isNotEmpty()) {
                Log.e(TAG, "Missing required QNN libs after extraction: $missing")
                return false
            }

            return true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to extract libraries: ${e.message}", e)
            return false
        }
    }

    /**
     * Log information about extracted libraries.
     */
    private fun logLibraryInfo(libDir: File) {
        Log.i(TAG, "=== QNN Library Info ===")
        Log.i(TAG, "QNN SDK Version: $QNN_SDK_VERSION")
        Log.w(TAG, "WARNING: ORT expects QNN $ORT_EXPECTED_QNN_VERSION - version mismatch may cause issues")
        Log.i(TAG, "Library path: ${libDir.absolutePath}")

        libDir.listFiles()?.forEach { file ->
            Log.i(TAG, "  ${file.name}: ${file.length()} bytes, executable=${file.canExecute()}")
        }
    }

    /**
     * Get QNN EP options map with custom library paths.
     */
    fun getQnnEpOptions(): Map<String, String> {
        val options = mutableMapOf<String, String>()

        qnnLibPath?.let { path ->
            // Backend path for Stub libraries
            options["backend_path"] = "$path/libQnnHtp.so"

            // Skel library directory for DSP
            options["skel_library_dir"] = path

            Log.i(TAG, "QNN EP options: backend_path=$path/libQnnHtp.so, skel_library_dir=$path")
        }

        return options
    }

    /**
     * Clean up extracted libraries (for testing/reset).
     */
    fun cleanup(context: Context) {
        val libDir = File(context.filesDir, "qnn_libs")
        if (libDir.exists()) {
            libDir.deleteRecursively()
            Log.i(TAG, "Cleaned up QNN libraries")
        }
        initialized = false
        qnnLibPath = null
    }
}
