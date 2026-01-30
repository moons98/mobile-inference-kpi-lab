package com.example.kpilab

import android.util.Log
import kotlinx.coroutines.*
import java.io.BufferedReader
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

    /**
     * Start capturing logcat for specified tags.
     *
     * @param tags List of logcat tags to filter (e.g., "onnxruntime", "OrtRunner")
     * @param scope CoroutineScope for the capture job
     */
    fun startCapture(
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

        captureJob = scope.launch(Dispatchers.IO) {
            try {
                // Clear logcat buffer first
                Runtime.getRuntime().exec("logcat -c").waitFor()

                // Start logcat process with filters
                val process = Runtime.getRuntime().exec(
                    arrayOf("logcat", "-v", "time") + filterArgs.toTypedArray()
                )

                val reader = BufferedReader(InputStreamReader(process.inputStream))

                Log.i(TAG, "Started capturing logcat for tags: $tags")

                while (isActive && isCapturing) {
                    val line = reader.readLine() ?: break
                    synchronized(capturedLogs) {
                        capturedLogs.appendLine(line)
                    }
                }

                process.destroy()
                Log.i(TAG, "Logcat capture stopped")

            } catch (e: Exception) {
                Log.e(TAG, "Logcat capture error: ${e.message}", e)
            }
        }
    }

    /**
     * Stop capturing logcat and return captured logs.
     *
     * @return Captured log content as string
     */
    fun stopCapture(): String {
        isCapturing = false
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

        for (line in logs.lines()) {
            // Parse node counts from graph partitioning logs
            // Example: "Number of nodes in the graph: 100"
            if (line.contains("nodes in the graph", ignoreCase = true)) {
                val match = Regex("(\\d+)\\s*nodes").find(line)
                match?.groupValues?.get(1)?.toIntOrNull()?.let { totalNodes = it }
            }

            // Parse QNN/CPU partition info
            // Example: "QNN EP assigned 80 nodes"
            if (line.contains("QNN", ignoreCase = true) && line.contains("node", ignoreCase = true)) {
                partitionInfo.add(line.trim())
                val match = Regex("(\\d+)\\s*node").find(line)
                match?.groupValues?.get(1)?.toIntOrNull()?.let { qnnNodes += it }
            }

            // Parse fallback ops
            // Example: "Op [Softmax] is not supported by QNN"
            if (line.contains("not supported", ignoreCase = true) ||
                line.contains("fallback", ignoreCase = true) ||
                line.contains("CPU EP", ignoreCase = true)) {
                val opMatch = Regex("\\[([^\\]]+)\\]").find(line)
                opMatch?.groupValues?.get(1)?.let { fallbackOps.add(it) }
                partitionInfo.add(line.trim())
            }
        }

        cpuNodes = totalNodes - qnnNodes

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
}
