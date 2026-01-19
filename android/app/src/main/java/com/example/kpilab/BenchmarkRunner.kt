package com.example.kpilab

import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

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
    val lastLatencyMs: Float = 0f,
    val lastThermalC: Float = 0f,
    val lastPowerMw: Float = 0f,
    val lastMemoryMb: Int = 0
) {
    val progressPercent: Int
        get() = if (totalMs > 0) ((elapsedMs * 100) / totalMs).toInt() else 0

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
}

/**
 * Runs benchmark with configured settings
 */
class BenchmarkRunner(
    private val nativeRunner: NativeRunner,
    private val kpiCollector: KpiCollector
) {
    companion object {
        private const val TAG = "BenchmarkRunner"
        private const val SYSTEM_METRICS_INTERVAL_MS = 1000L  // 1 second
        private const val MEMORY_METRICS_INTERVAL_MS = 5000L  // 5 seconds
    }

    private val _progress = MutableStateFlow(BenchmarkProgress())
    val progress: StateFlow<BenchmarkProgress> = _progress.asStateFlow()

    private var benchmarkJob: Job? = null
    private var systemMetricsJob: Job? = null
    private var startTimeMs: Long = 0
    private var config: BenchmarkConfig? = null

    val isRunning: Boolean
        get() = _progress.value.state == BenchmarkState.RUNNING ||
                _progress.value.state == BenchmarkState.WARMING_UP

    /**
     * Start benchmark with given configuration
     */
    fun start(
        config: BenchmarkConfig,
        modelPath: String,
        scope: CoroutineScope
    ) {
        if (isRunning) {
            Log.w(TAG, "Benchmark already running")
            return
        }

        this.config = config
        Log.i(TAG, "Starting benchmark: $config")

        benchmarkJob = scope.launch(Dispatchers.Default) {
            try {
                runBenchmark(config, modelPath)
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

    private suspend fun runBenchmark(config: BenchmarkConfig, modelPath: String) {
        // Initialize
        _progress.value = BenchmarkProgress(
            state = BenchmarkState.INITIALIZING,
            totalMs = config.durationMs
        )

        val initialized = withContext(Dispatchers.IO) {
            nativeRunner.initialize(
                modelPath,
                config.executionPath.value,
                config.warmUpEnabled
            )
        }

        if (!initialized) {
            Log.e(TAG, "Failed to initialize native runner")
            _progress.value = BenchmarkProgress(state = BenchmarkState.IDLE)
            return
        }

        // Start session
        val sessionId = config.generateSessionId()
        nativeRunner.startSession(sessionId)
        Log.i(TAG, "Session started: $sessionId")

        // Update state
        if (config.warmUpEnabled) {
            _progress.value = _progress.value.copy(state = BenchmarkState.WARMING_UP)
            // Warm-up is handled in native code during initialize
        }

        _progress.value = _progress.value.copy(state = BenchmarkState.RUNNING)
        startTimeMs = System.currentTimeMillis()

        // Start system metrics collection in parallel
        val metricsScope = CoroutineScope(coroutineContext)
        systemMetricsJob = metricsScope.launch {
            collectSystemMetrics(config.durationMs)
        }

        // Main inference loop
        var inferenceCount = 0
        val intervalMs = config.intervalMs

        while (isActive) {
            val elapsed = System.currentTimeMillis() - startTimeMs
            if (elapsed >= config.durationMs) {
                break
            }

            val loopStart = System.currentTimeMillis()

            // Run inference
            val latencyMs = withContext(Dispatchers.IO) {
                nativeRunner.runInference()
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
        nativeRunner.endSession()
        Log.i(TAG, "Benchmark completed: $inferenceCount inferences")
    }

    private suspend fun collectSystemMetrics(durationMs: Long) {
        var lastMemoryTime = 0L

        while (isActive) {
            val elapsed = System.currentTimeMillis() - startTimeMs
            if (elapsed >= durationMs) {
                break
            }

            // Collect thermal and power every second
            val metrics = kpiCollector.collectAll()

            // Log to native
            nativeRunner.logSystemMetrics(
                metrics.thermalC,
                metrics.powerMw,
                if (elapsed - lastMemoryTime >= MEMORY_METRICS_INTERVAL_MS) {
                    lastMemoryTime = elapsed
                    metrics.memoryMb
                } else {
                    -1  // Skip memory this time
                }
            )

            // Update progress
            _progress.value = _progress.value.copy(
                lastThermalC = metrics.thermalC,
                lastPowerMw = metrics.powerMw,
                lastMemoryMb = if (metrics.memoryMb > 0) metrics.memoryMb
                               else _progress.value.lastMemoryMb
            )

            delay(SYSTEM_METRICS_INTERVAL_MS)
        }
    }

    private fun cleanup() {
        systemMetricsJob?.cancel()
        _progress.value = BenchmarkProgress(state = BenchmarkState.IDLE)
        config = null
        Log.i(TAG, "Cleanup completed")
    }

    /**
     * Export collected data as CSV
     */
    fun exportCsv(): String {
        return nativeRunner.exportCsv()
    }

    /**
     * Get record count
     */
    fun getRecordCount(): Int {
        return nativeRunner.getRecordCount()
    }

    /**
     * Release resources
     */
    fun release() {
        stop()
        nativeRunner.release()
    }
}
