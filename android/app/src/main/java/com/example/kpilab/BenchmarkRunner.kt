package com.example.kpilab

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.currentCoroutineContext
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
    }

    private val _progress = MutableStateFlow(BenchmarkProgress())
    val progress: StateFlow<BenchmarkProgress> = _progress.asStateFlow()

    private var ortRunner: OrtRunner? = null
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
     * Release resources
     */
    fun release() {
        stop()
        ortRunner?.release()
        ortRunner = null
    }
}
