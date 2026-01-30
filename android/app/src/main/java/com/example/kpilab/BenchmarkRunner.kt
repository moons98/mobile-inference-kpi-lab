package com.example.kpilab

import android.content.Context
import android.os.Environment
import android.util.Log
import com.example.kpilab.batch.BatchProgress
import com.example.kpilab.batch.ExperimentDefaults
import com.example.kpilab.batch.ExperimentSet
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

/**
 * Benchmark state
 */
enum class BenchmarkState {
    IDLE,
    INITIALIZING,
    WARMING_UP,
    RUNNING,
    STOPPING
}

/**
 * Progress information
 */
data class BenchmarkProgress(
    val state: BenchmarkState = BenchmarkState.IDLE,
    val elapsedMs: Long = 0,
    val totalMs: Long = 0,
    val inferenceCount: Int = 0,
    val lastLatencyMs: Float = -1f,
    val lastThermalC: Float = -1f,
    val lastPowerMw: Float = -1f,
    val lastMemoryMb: Int = 0
) {
    val progressPercent: Int
        get() = if (totalMs > 0) ((elapsedMs * 100) / totalMs).toInt() else 0

    /** Throughput in inferences per second */
    val throughput: Float
        get() = if (elapsedMs > 0) (inferenceCount * 1000f) / elapsedMs else 0f

    fun formatElapsed(): String {
        val seconds = (elapsedMs / 1000) % 60
        val minutes = (elapsedMs / 1000) / 60
        return String.format("%02d:%02d", minutes, seconds)
    }

    fun formatTotal(): String {
        val seconds = (totalMs / 1000) % 60
        val minutes = (totalMs / 1000) / 60
        return String.format("%02d:%02d", minutes, seconds)
    }

    fun formatThroughput(): String {
        return String.format("%.2f inf/s", throughput)
    }
}

/**
 * Runs benchmark with ONNX Runtime
 */
class BenchmarkRunner(
    private val context: Context,
    private val kpiCollector: KpiCollector
) {
    companion object {
        private const val TAG = "BenchmarkRunner"
        private const val SYSTEM_METRICS_INTERVAL_MS = 1000L  // 1 second
        private const val MEMORY_METRICS_INTERVAL_MS = 5000L  // 5 seconds
        private const val COOLDOWN_SECONDS = 30  // Cooldown between batch experiments
    }

    private val _progress = MutableStateFlow(BenchmarkProgress())
    val progress: StateFlow<BenchmarkProgress> = _progress.asStateFlow()

    private val _batchProgress = MutableStateFlow(BatchProgress())
    val batchProgress: StateFlow<BatchProgress> = _batchProgress.asStateFlow()

    private var ortRunner: OrtRunner? = null
    private var benchmarkJob: Job? = null
    private var systemMetricsJob: Job? = null
    private var batchJob: Job? = null
    private var startTimeMs: Long = 0
    private var config: BenchmarkConfig? = null

    // Logcat capture for ORT verbose logs
    private val logcatCapture = LogcatCapture()
    private var lastOrtLogInfo: OrtLogInfo? = null

    /** Current model being benchmarked */
    var currentModel: OnnxModelType? = null
        private set

    val isRunning: Boolean
        get() = _progress.value.state == BenchmarkState.RUNNING ||
                _progress.value.state == BenchmarkState.WARMING_UP

    val isBatchRunning: Boolean
        get() = _batchProgress.value.isRunning

    /**
     * Start benchmark with given configuration
     */
    fun start(
        config: BenchmarkConfig,
        scope: CoroutineScope
    ) {
        if (isRunning || isBatchRunning) {
            Log.w(TAG, "Benchmark already running")
            return
        }

        this.config = config
        this.currentModel = config.modelType
        Log.i(TAG, "Starting benchmark: $config")

        benchmarkJob = scope.launch(Dispatchers.Default) {
            try {
                runBenchmark(config)
            } catch (e: CancellationException) {
                Log.i(TAG, "Benchmark cancelled")
            } catch (e: Exception) {
                Log.e(TAG, "Benchmark error: ${e.message}", e)
            } finally {
                cleanup()
            }
        }
    }

    /**
     * Stop running benchmark
     */
    fun stop() {
        Log.i(TAG, "Stopping benchmark")
        _progress.value = _progress.value.copy(state = BenchmarkState.STOPPING)
        benchmarkJob?.cancel()
        systemMetricsJob?.cancel()
    }

    /**
     * Stop running batch
     */
    fun stopBatch() {
        Log.i(TAG, "Stopping batch")
        batchJob?.cancel()
        stop()
        _batchProgress.value = BatchProgress()
    }

    private suspend fun runBenchmark(config: BenchmarkConfig) {
        // Initialize
        _progress.value = BenchmarkProgress(
            state = BenchmarkState.INITIALIZING,
            totalMs = config.durationMs
        )

        // Release previous runner if exists
        ortRunner?.let {
            Log.i(TAG, "Releasing previous runner")
            it.release()
            ortRunner = null
        }

        // Start logcat capture BEFORE ORT initialization to catch graph compilation logs
        val captureScope = CoroutineScope(currentCoroutineContext())
        logcatCapture.startCapture(
            tags = listOf("onnxruntime", "OrtRunner", "QNN"),
            scope = captureScope
        )
        Log.i(TAG, "Logcat capture started")

        // Create and initialize ONNX Runtime runner
        val initialized = withContext(Dispatchers.IO) {
            Log.i(TAG, "Initializing ONNX Runtime runner")
            val runner = OrtRunner(context)
            val success = runner.initialize(
                config.modelType,
                config.executionProvider,
                config.useNpuFp16,
                config.useContextCache
            )
            if (success) {
                ortRunner = runner
            }
            success
        }

        if (!initialized) {
            Log.e(TAG, "Failed to initialize runner")
            // Stop logcat capture even on failure - it may contain useful error info
            logcatCapture.stopCapture()
            lastOrtLogInfo = logcatCapture.parseOrtInfo()
            _progress.value = BenchmarkProgress(state = BenchmarkState.IDLE)
            return
        }

        val runner = ortRunner ?: return

        // Start session
        val sessionId = config.generateSessionId()
        runner.startSession(sessionId)
        runner.setBenchmarkConfig(config.frequencyHz, if (config.warmUpEnabled) 10 else 0)
        Log.i(TAG, "Session started: $sessionId")

        // Run warm-up if enabled (with proper state display)
        if (config.warmUpEnabled) {
            _progress.value = _progress.value.copy(state = BenchmarkState.WARMING_UP)
            withContext(Dispatchers.IO) {
                runner.runWarmUp()
            }
        }

        _progress.value = _progress.value.copy(state = BenchmarkState.RUNNING)
        startTimeMs = System.currentTimeMillis()

        // Start system metrics collection in parallel
        val metricsScope = CoroutineScope(currentCoroutineContext())
        systemMetricsJob = metricsScope.launch {
            collectSystemMetrics(config.durationMs, runner)
        }

        // Main inference loop
        var inferenceCount = 0
        val intervalMs = config.intervalMs

        while (currentCoroutineContext().isActive) {
            val elapsed = System.currentTimeMillis() - startTimeMs
            if (elapsed >= config.durationMs) {
                break
            }

            val loopStart = System.currentTimeMillis()

            // Run inference
            val latencyMs = withContext(Dispatchers.IO) {
                runner.runInference()
            }

            if (latencyMs >= 0) {
                inferenceCount++

                // Update progress
                val currentElapsed = System.currentTimeMillis() - startTimeMs
                _progress.value = _progress.value.copy(
                    elapsedMs = currentElapsed,
                    inferenceCount = inferenceCount,
                    lastLatencyMs = latencyMs
                )
            }

            // Wait for next interval
            val loopDuration = System.currentTimeMillis() - loopStart
            val sleepTime = intervalMs - loopDuration
            if (sleepTime > 0) {
                delay(sleepTime)
            }
        }

        // End session
        runner.endSession()
        Log.i(TAG, "Benchmark completed: $inferenceCount inferences")

        // Stop logcat capture and parse ORT info
        logcatCapture.stopCapture()
        lastOrtLogInfo = logcatCapture.parseOrtInfo()
        Log.i(TAG, "Logcat capture stopped, captured ${lastOrtLogInfo?.rawLogs?.lines()?.size ?: 0} lines")
    }

    private suspend fun collectSystemMetrics(durationMs: Long, runner: OrtRunner) {
        var lastMemoryTime = 0L
        var lastMemoryValue = 0

        while (currentCoroutineContext().isActive) {
            val elapsed = System.currentTimeMillis() - startTimeMs
            if (elapsed >= durationMs) {
                break
            }

            // Collect thermal and power every second
            val metrics = kpiCollector.collectAll()

            // Log to runner (memory every 5 seconds, use -1 for "not sampled")
            val memoryMb = if (elapsed - lastMemoryTime >= MEMORY_METRICS_INTERVAL_MS) {
                lastMemoryTime = elapsed
                lastMemoryValue = metrics.memoryMb
                metrics.memoryMb
            } else {
                -1  // Not sampled this interval
            }

            runner.logSystemMetrics(
                metrics.thermalC,
                metrics.powerMw,
                memoryMb
            )

            // Update progress (keep last known memory value for UI)
            _progress.value = _progress.value.copy(
                lastThermalC = metrics.thermalC,
                lastPowerMw = metrics.powerMw,
                lastMemoryMb = if (lastMemoryValue > 0) lastMemoryValue
                               else _progress.value.lastMemoryMb
            )

            delay(SYSTEM_METRICS_INTERVAL_MS)
        }
    }

    private fun cleanup() {
        systemMetricsJob?.cancel()
        if (logcatCapture.isCapturing()) {
            logcatCapture.stopCapture()
        }
        _progress.value = BenchmarkProgress(state = BenchmarkState.IDLE)
        config = null
        Log.i(TAG, "Cleanup completed")
    }

    /**
     * Export collected data as CSV
     */
    fun exportCsv(): String {
        return ortRunner?.exportCsv() ?: ""
    }

    /**
     * Get record count
     */
    fun getRecordCount(): Int {
        return ortRunner?.getRecordCount() ?: 0
    }

    /**
     * Get device info
     */
    fun getDeviceInfo(): String {
        return ortRunner?.getDeviceInfo() ?: "No runner active"
    }

    /**
     * Check if NPU is active
     */
    fun isNpuActive(): Boolean {
        return ortRunner?.isNpuActive() ?: false
    }

    /**
     * Set foreground state
     */
    fun setForeground(isForeground: Boolean) {
        ortRunner?.setForeground(isForeground)
    }

    /**
     * Start batch experiment execution
     */
    fun startBatch(
        experimentSet: ExperimentSet,
        defaults: ExperimentDefaults,
        scope: CoroutineScope,
        onExperimentComplete: (csvPath: String) -> Unit = {}
    ) {
        if (isRunning || isBatchRunning) {
            Log.w(TAG, "Benchmark or batch already running")
            return
        }

        Log.i(TAG, "Starting batch: ${experimentSet.name} with ${experimentSet.experiments.size} experiments")

        _batchProgress.value = BatchProgress(
            isRunning = true,
            currentSetName = experimentSet.name,
            currentExperimentIndex = 0,
            totalExperiments = experimentSet.experiments.size
        )

        batchJob = scope.launch(Dispatchers.Default) {
            val completedFiles = mutableListOf<String>()

            try {
                for ((index, experiment) in experimentSet.experiments.withIndex()) {
                    // Check if cancelled
                    if (!isActive) break

                    val experimentName = experiment.getDisplayName()
                    Log.i(TAG, "=== Batch Experiment ${index + 1}/${experimentSet.experiments.size}: $experimentName ===")

                    // Update batch progress
                    _batchProgress.value = _batchProgress.value.copy(
                        currentExperimentIndex = index + 1,
                        currentExperimentName = experimentName,
                        isCoolingDown = false
                    )

                    // Convert to BenchmarkConfig
                    val config = experiment.toBenchmarkConfig(defaults)
                    this@BenchmarkRunner.config = config
                    this@BenchmarkRunner.currentModel = config.modelType

                    // Run single benchmark
                    try {
                        runBenchmark(config)
                    } catch (e: CancellationException) {
                        Log.i(TAG, "Batch experiment cancelled")
                        throw e
                    } catch (e: Exception) {
                        Log.e(TAG, "Batch experiment failed: ${e.message}", e)
                        // Continue with next experiment
                        continue
                    }

                    // Export CSV
                    val csvPath = exportAndSaveCsv()
                    if (csvPath != null) {
                        completedFiles.add(csvPath)
                        _batchProgress.value = _batchProgress.value.copy(
                            completedExperiments = completedFiles.toList()
                        )
                        onExperimentComplete(csvPath)
                        Log.i(TAG, "Experiment CSV saved: $csvPath")
                    }

                    // Cooldown between experiments (except for last one)
                    if (index < experimentSet.experiments.size - 1 && isActive) {
                        Log.i(TAG, "Cooldown for $COOLDOWN_SECONDS seconds...")
                        _batchProgress.value = _batchProgress.value.copy(isCoolingDown = true)

                        for (remaining in COOLDOWN_SECONDS downTo 1) {
                            if (!isActive) break
                            _batchProgress.value = _batchProgress.value.copy(
                                cooldownRemainingSeconds = remaining
                            )
                            delay(1000)
                        }
                    }
                }

                Log.i(TAG, "=== Batch completed: ${completedFiles.size}/${experimentSet.experiments.size} experiments ===")

            } catch (e: CancellationException) {
                Log.i(TAG, "Batch cancelled")
            } finally {
                _batchProgress.value = BatchProgress(
                    completedExperiments = completedFiles.toList()
                )
                cleanup()
            }
        }
    }

    /**
     * Export current data and save to file, returning the file path
     */
    private fun exportAndSaveCsv(): String? {
        val csvData = exportCsv()
        if (csvData.isEmpty()) {
            return null
        }

        return try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val modelName = currentModel?.displayName
                ?.replace(" ", "")
                ?.replace("-", "")
                ?.replace("(", "")
                ?.replace(")", "")
                ?: "Unknown"
            val ep = getActiveExecutionProvider()
                .replace("_", "")
                .uppercase()
            val baseFilename = "kpi_${modelName}_${ep}_${timestamp}"

            val exportDir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            if (exportDir != null) {
                // Save CSV
                val csvFile = File(exportDir, "${baseFilename}.csv")
                FileWriter(csvFile).use { writer ->
                    writer.write(csvData)
                }
                Log.i(TAG, "CSV exported: ${csvFile.absolutePath}")

                // Save ORT logs alongside CSV
                lastOrtLogInfo?.let { ortInfo ->
                    if (ortInfo.rawLogs.isNotBlank()) {
                        val logFile = File(exportDir, "${baseFilename}_ort.log")
                        FileWriter(logFile).use { writer ->
                            writer.write(ortInfo.toSummary())
                            writer.write("\n\n=== Raw Logs ===\n")
                            writer.write(ortInfo.rawLogs)
                        }
                        Log.i(TAG, "ORT logs exported: ${logFile.absolutePath}")
                    }
                }

                csvFile.absolutePath
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "CSV export failed: ${e.message}", e)
            null
        }
    }

    /**
     * Get active execution provider name
     */
    fun getActiveExecutionProvider(): String {
        return ortRunner?.getActiveExecutionProvider() ?: "CPU"
    }

    /**
     * Get last ORT log info (graph partitioning, fallback ops, etc.)
     */
    fun getOrtLogInfo(): OrtLogInfo? = lastOrtLogInfo

    /**
     * Release resources
     */
    fun release() {
        stopBatch()
        stop()
        if (logcatCapture.isCapturing()) {
            logcatCapture.stopCapture()
        }
        ortRunner?.release()
        ortRunner = null
    }
}
