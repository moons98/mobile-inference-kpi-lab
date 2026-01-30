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
        var qnnSetupFailed = false
        var qnnErrorMessage: String? = null

        for (line in logs.lines()) {
            // Check for QNN setup failure
            // Example: "QNN SetupBackend failed"
            if (line.contains("SetupBackend failed", ignoreCase = true) ||
                line.contains("QNN_DEVICE_ERROR", ignoreCase = true) ||
                line.contains("QNN_COMMON_ERROR", ignoreCase = true)) {
                qnnSetupFailed = true
                qnnErrorMessage = line.substringAfter("Error:").trim().ifEmpty { null }
                    ?: line.substringAfter("failed").trim().ifEmpty { null }
                partitionInfo.add(line.trim())
            }

            // Parse node placement info from ORT
            // Example: "All nodes placed on [QNNExecutionProvider]. Number of nodes: 77"
            // Example: "All nodes placed on [CPUExecutionProvider]. Number of nodes: 55"
            if (line.contains("nodes placed on", ignoreCase = true)) {
                partitionInfo.add(line.trim())
                val nodeCountMatch = Regex("Number of nodes:\\s*(\\d+)").find(line)
                val nodeCount = nodeCountMatch?.groupValues?.get(1)?.toIntOrNull() ?: 0

                when {
                    line.contains("QNNExecutionProvider") -> qnnNodes += nodeCount
                    line.contains("CPUExecutionProvider") -> cpuNodes += nodeCount
                }
                totalNodes += nodeCount
            }

            // Parse fallback ops
            // Example: "Op [Softmax] is not supported by QNN"
            // Example: "Falling back to CPU for node: Softmax_123"
            if (line.contains("not supported", ignoreCase = true)) {
                val opMatch = Regex("\\[([^\\]]+)\\]").find(line)
                    ?: Regex("Op\\s+(\\w+)").find(line)
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
}
