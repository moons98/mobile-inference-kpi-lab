package com.example.kpilab

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.CompletableDeferred
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader

/**
 * Utility class to capture logcat output during benchmark runs.
 * Captures ORT verbose logs for graph partitioning and fallback ops analysis.
 */
class LogcatCapture {

    companion object {
        private const val TAG = "LogcatCapture"
    }

    private var captureJob: Job? = null
    private val capturedLogs = StringBuilder()
    private var isCapturing = false
    @Volatile private var captureProcess: Process? = null
    @Volatile private var captureReader: BufferedReader? = null

    /**
     * Start capturing logcat for specified tags.
     * Suspends until logcat process is ready to capture, preventing race conditions
     * where ORT initialization logs are missed.
     *
     * @param tags List of logcat tags to filter (e.g., "onnxruntime", "OrtRunner")
     * @param scope CoroutineScope for the capture job
     */
    suspend fun startCapture(
        tags: List<String> = listOf("onnxruntime", "OrtRunner"),
        scope: CoroutineScope
    ) {
        if (isCapturing) {
            Log.w(TAG, "Already capturing")
            return
        }

        capturedLogs.clear()
        isCapturing = true

        // Build logcat filter string: "onnxruntime:V OrtRunner:V *:S"
        val filterArgs = tags.map { "$it:V" } + "*:S"

        val ready = CompletableDeferred<Unit>()

        captureJob = scope.launch(Dispatchers.IO) {
            try {
                // Clear logcat buffer first
                Runtime.getRuntime().exec("logcat -c").waitFor()

                // Start logcat process with filters
                val process = Runtime.getRuntime().exec(
                    arrayOf("logcat", "-v", "time") + filterArgs.toTypedArray()
                )
                captureProcess = process

                val reader = BufferedReader(InputStreamReader(process.inputStream))
                captureReader = reader

                Log.i(TAG, "Started capturing logcat for tags: $tags")
                // Signal that logcat process is ready
                ready.complete(Unit)

                while (isActive && isCapturing) {
                    val line = reader.readLine() ?: break
                    synchronized(capturedLogs) {
                        capturedLogs.appendLine(line)
                    }
                }

                reader.close()
                process.destroy()
                Log.i(TAG, "Logcat capture stopped")

            } catch (e: Exception) {
                Log.e(TAG, "Logcat capture error: ${e.message}", e)
                ready.complete(Unit) // Don't block caller on failure
            } finally {
                try {
                    captureReader?.close()
                } catch (_: Throwable) {
                }
                try {
                    captureProcess?.destroy()
                } catch (_: Throwable) {
                }
                captureReader = null
                captureProcess = null
            }
        }

        // Wait until logcat process is ready before returning
        ready.await()
    }

    /**
     * Stop capturing logcat and return captured logs.
     *
     * @return Captured log content as string
     */
    fun stopCapture(): String {
        isCapturing = false
        try {
            captureReader?.close()
        } catch (_: Throwable) {
        }
        try {
            captureProcess?.destroy()
        } catch (_: Throwable) {
        }
        captureJob?.cancel()
        captureJob = null

        val logs = synchronized(capturedLogs) {
            capturedLogs.toString()
        }

        Log.i(TAG, "Captured ${logs.lines().size} log lines")
        return logs
    }

    /**
     * Get captured logs without stopping capture.
     */
    fun getCapturedLogs(): String {
        return synchronized(capturedLogs) {
            capturedLogs.toString()
        }
    }

    /**
     * Check if currently capturing.
     */
    fun isCapturing(): Boolean = isCapturing

    /**
     * Parse captured logs to extract ORT graph partitioning info.
     *
     * @return Parsed info as structured string
     */
    fun parseOrtInfo(): OrtLogInfo {
        val logs = getCapturedLogs()

        var totalNodes = 0
        var qnnNodes = 0
        var cpuNodes = 0
        val fallbackOps = mutableListOf<String>()
        val partitionInfo = mutableListOf<String>()
        var qnnSetupFailed = false
        var qnnErrorMessage: String? = null

        // Partition info appears only during model load (before first inference).
        // After finding a "nodes placed on" or "GetCapability" summary line,
        // we scan at most TAIL_LINES more lines to catch any trailing fallback ops,
        // then stop — avoiding scanning millions of verbose inference-step log lines.
        var partitionFound = false
        var tailLinesRemaining = -1
        val TAIL_LINES = 500

        for (line in logs.lines()) {
            if (partitionFound) {
                if (tailLinesRemaining-- <= 0) break
            }

            // Check for QNN setup failure
            if (line.contains("SetupBackend failed", ignoreCase = true) ||
                line.contains("QNN_DEVICE_ERROR", ignoreCase = true) ||
                line.contains("QNN_COMMON_ERROR", ignoreCase = true)) {
                qnnSetupFailed = true
                qnnErrorMessage = line.substringAfter("Error:").trim().ifEmpty { null }
                    ?: line.substringAfter("failed").trim().ifEmpty { null }
                partitionInfo.add(line.trim())
                if (!partitionFound) { partitionFound = true; tailLinesRemaining = TAIL_LINES }
            }

            // Parse node placement info from ORT
            // Format A: "All nodes placed on [QNNExecutionProvider]. Number of nodes: 77"
            if (line.contains("nodes placed on", ignoreCase = true)) {
                partitionInfo.add(line.trim())
                val nodeCountMatch = Regex("Number of nodes:\\s*(\\d+)").find(line)
                val nodeCount = nodeCountMatch?.groupValues?.get(1)?.toIntOrNull() ?: 0
                when {
                    line.contains("QNNExecutionProvider") -> qnnNodes += nodeCount
                    line.contains("CPUExecutionProvider") -> cpuNodes += nodeCount
                }
                totalNodes += nodeCount
                if (!partitionFound) { partitionFound = true; tailLinesRemaining = TAIL_LINES }
            }

            // Format B (ORT 1.x): "GetCapability] Number of partitions supported by QNN EP: 1,
            //   number of nodes in the graph: 233, number of nodes supported by QNN: 233"
            // ORT calls GetCapability multiple times (per optimization pass); use the last value.
            if (line.contains("GetCapability", ignoreCase = true) &&
                line.contains("number of nodes in the graph", ignoreCase = true)) {
                partitionInfo.add(line.trim())
                val graphNodesMatch = Regex("number of nodes in the graph:\\s*(\\d+)", RegexOption.IGNORE_CASE).find(line)
                val qnnNodesMatch = Regex("number of nodes supported by QNN:\\s*(\\d+)", RegexOption.IGNORE_CASE).find(line)
                val graphCount = graphNodesMatch?.groupValues?.get(1)?.toIntOrNull() ?: 0
                val qnnCount = qnnNodesMatch?.groupValues?.get(1)?.toIntOrNull() ?: 0
                totalNodes = graphCount
                qnnNodes = qnnCount
                cpuNodes = graphCount - qnnCount
                // Don't break yet — GetCapability fires multiple times; keep updating until tail ends
                if (!partitionFound) { partitionFound = true; tailLinesRemaining = TAIL_LINES }
                else tailLinesRemaining = TAIL_LINES  // reset window on each GetCapability hit
            }

            // Parse fallback ops
            // Example: "Op [Softmax] is not supported by QNN"
            if (line.contains("not supported", ignoreCase = true)) {
                val opMatch = Regex("\\[([^\\]]+)\\]").find(line)
                    ?: Regex("Op\\s+(\\w+)").find(line)
                opMatch?.groupValues?.get(1)?.let {
                    if (!fallbackOps.contains(it)) fallbackOps.add(it)
                }
                partitionInfo.add(line.trim())
            }

            // Example: "Validation FAILED for 1 nodes in NodeUnit (QuantizeLinear) :"
            if (line.contains("Validation FAILED", ignoreCase = true)) {
                val opMatch = Regex("NodeUnit\\s*\\(([^)]+)\\)").find(line)
                opMatch?.groupValues?.get(1)?.let {
                    if (!fallbackOps.contains(it)) fallbackOps.add(it)
                }
                partitionInfo.add(line.trim())
            }

            // Parse CPU fallback info
            if (line.contains("fallback", ignoreCase = true) && line.contains("CPU", ignoreCase = true)) {
                val nodeMatch = Regex("node[:\\s]+([\\w_]+)").find(line)
                nodeMatch?.groupValues?.get(1)?.let {
                    val opType = it.substringBefore("_")
                    if (!fallbackOps.contains(opType)) fallbackOps.add(opType)
                }
                partitionInfo.add(line.trim())
            }
        }

        // If QNN setup failed, all nodes run on CPU
        if (qnnSetupFailed && qnnNodes == 0 && cpuNodes == 0) {
            // No node placement info found, but we know it fell back to CPU
            qnnErrorMessage?.let { fallbackOps.add("QNN failed: $it") }
        }

        return OrtLogInfo(
            totalNodes = totalNodes,
            qnnNodes = qnnNodes,
            cpuNodes = cpuNodes,
            fallbackOps = fallbackOps.distinct(),
            partitionInfo = partitionInfo,
            rawLogs = logs
        )
    }
}

/**
 * Parsed ORT log information.
 */
data class OrtLogInfo(
    val totalNodes: Int,
    val qnnNodes: Int,
    val cpuNodes: Int,
    val fallbackOps: List<String>,
    val partitionInfo: List<String>,
    val rawLogs: String
) {
    /**
     * Format as summary string for CSV header or report.
     */
    fun toSummary(): String {
        return buildString {
            appendLine("=== ORT Graph Partitioning Info ===")
            appendLine("Total nodes: $totalNodes")
            appendLine("QNN nodes: $qnnNodes")
            appendLine("CPU fallback nodes: $cpuNodes")
            if (fallbackOps.isNotEmpty()) {
                appendLine("Fallback ops: ${fallbackOps.joinToString(", ")}")
            }
            appendLine("")
            appendLine("=== Partition Details ===")
            partitionInfo.forEach { appendLine(it) }
        }
    }

    /**
     * Format as CSV metadata lines.
     */
    fun toCsvMetadata(): String {
        return buildString {
            appendLine("# ort_total_nodes,$totalNodes")
            appendLine("# ort_qnn_nodes,$qnnNodes")
            appendLine("# ort_cpu_nodes,$cpuNodes")
            appendLine("# ort_fallback_ops,${fallbackOps.joinToString(";")}")
        }
    }

    /** True if any partition data was captured. */
    fun hasData(): Boolean = totalNodes > 0 || qnnNodes > 0 || cpuNodes > 0 || fallbackOps.isNotEmpty()

    /** Save to JSON file for later retrieval (survives cache-hit runs). */
    fun saveToFile(file: File) {
        val json = JSONObject().apply {
            put("totalNodes", totalNodes)
            put("qnnNodes", qnnNodes)
            put("cpuNodes", cpuNodes)
            put("fallbackOps", JSONArray(fallbackOps))
            put("partitionInfo", JSONArray(partitionInfo))
        }
        file.writeText(json.toString(2))
        Log.i("OrtLogInfo", "Saved partition info to ${file.name}: QNN=$qnnNodes CPU=$cpuNodes total=$totalNodes")
    }

    companion object {
        /** Cache filename for a given model. */
        fun cacheFileName(modelName: String): String = "ort_partition_${modelName}.json"

        /** Load from previously saved JSON file, or null if not found. */
        fun loadFromFile(file: File): OrtLogInfo? {
            if (!file.exists()) return null
            return try {
                val json = JSONObject(file.readText())
                val fallback = mutableListOf<String>()
                val fbArr = json.optJSONArray("fallbackOps")
                if (fbArr != null) for (i in 0 until fbArr.length()) fallback.add(fbArr.getString(i))
                val pInfo = mutableListOf<String>()
                val piArr = json.optJSONArray("partitionInfo")
                if (piArr != null) for (i in 0 until piArr.length()) pInfo.add(piArr.getString(i))
                OrtLogInfo(
                    totalNodes = json.optInt("totalNodes", 0),
                    qnnNodes = json.optInt("qnnNodes", 0),
                    cpuNodes = json.optInt("cpuNodes", 0),
                    fallbackOps = fallback,
                    partitionInfo = pInfo,
                    rawLogs = "(loaded from cache)"
                )
            } catch (e: Exception) {
                Log.w("OrtLogInfo", "Failed to load partition cache: ${e.message}")
                null
            }
        }

        /** Load partition info for a model, looking in the app's cache directory. */
        fun loadForModel(context: Context, modelName: String): OrtLogInfo? {
            val file = File(context.cacheDir, cacheFileName(modelName))
            return loadFromFile(file)
        }

        /** Save partition info for a model to the app's cache directory. */
        fun saveForModel(context: Context, modelName: String, info: OrtLogInfo) {
            val file = File(context.cacheDir, cacheFileName(modelName))
            info.saveToFile(file)
        }
    }
}
