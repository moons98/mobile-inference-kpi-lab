package com.example.kpilab

import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
import android.os.PowerManager
import android.util.Log
import com.example.kpilab.batch.BatchProgress
import com.example.kpilab.batch.ExperimentDefaults
import com.example.kpilab.batch.ExperimentSet
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

enum class BenchmarkState {
    IDLE, INITIALIZING, WARMING_UP, RUNNING, STOPPING, ERROR
}

data class BenchmarkProgress(
    val state: BenchmarkState = BenchmarkState.IDLE,
    val currentTrial: Int = 0,
    val totalTrials: Int = 0,
    val currentStep: Int = 0,
    val totalSteps: Int = 0,
    val currentStage: String = "",
    val lastE2eMs: Float = -1f,
    val lastThermalC: Float = -1f,
    val lastPowerMw: Float = -1f,
    val lastMemoryMb: Int = 0,
    val errorMessage: String? = null
) {
    val progressPercent: Int
        get() = if (totalTrials > 0) (currentTrial * 100 / totalTrials) else 0
}

/**
 * Text-to-image benchmark runner.
 * Phase 1 (SINGLE_GENERATE): 5 trials, cooldown between.
 * Phase 2 (SUSTAINED_GENERATE): 10 consecutive trials, no cooldown.
 * Outputs 3 CSV record types per experiment_design.md Appendix A.
 */
class BenchmarkRunner(
    private val context: Context,
    private val kpiCollector: KpiCollector
) {
    companion object {
        private const val TAG = "BenchmarkRunner"
        private const val COOLDOWN_MIN_SECONDS = 60
        private const val COOLDOWN_MAX_SECONDS = 180
        private const val COOLDOWN_TARGET_TEMP_C = 35f
        private const val SYSTEM_METRICS_INTERVAL_MS = 1000L
    }

    private val _progress = MutableStateFlow(BenchmarkProgress())
    val progress: StateFlow<BenchmarkProgress> = _progress.asStateFlow()

    private val _batchProgress = MutableStateFlow(BatchProgress())
    val batchProgress: StateFlow<BatchProgress> = _batchProgress.asStateFlow()

    private val _lastGeneratedImage = MutableStateFlow<Bitmap?>(null)
    val lastGeneratedImage: StateFlow<Bitmap?> = _lastGeneratedImage.asStateFlow()

    private var pipeline: Txt2ImgPipeline? = null
    private var benchmarkJob: Job? = null
    private var systemMetricsJob: Job? = null
    private var batchJob: Job? = null
    private var config: BenchmarkConfig? = null
    private var lastExportConfig: BenchmarkConfig? = null

    // Collected data for CSV export
    private val generateSummaries = mutableListOf<GenerateSummaryRecord>()
    private val unetStepDetails = mutableListOf<UnetStepRecord>()
    private var coldStartRecord: ColdStartRecord? = null
    @Volatile private var lastSystemThermal: Float = 0f
    @Volatile private var lastSystemPower: Float = 0f
    @Volatile private var lastSystemMemory: Int = 0
    @Volatile private var lastSystemNativeHeapMb: Float = 0f

    // Logcat capture
    private val logcatCapture = LogcatCapture()
    private var lastOrtLogInfo: OrtLogInfo? = null

    // System metrics coroutine scope (field so cleanup() can cancel the parent Job)
    private var metricsScope: CoroutineScope? = null

    // WakeLock — prevents CPU sleep during long benchmark runs
    private val wakeLock: PowerManager.WakeLock = (context.getSystemService(Context.POWER_SERVICE) as PowerManager)
        .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "kpilab:benchmark")

    val isRunning: Boolean
        get() = _progress.value.state in listOf(
            BenchmarkState.RUNNING, BenchmarkState.WARMING_UP, BenchmarkState.INITIALIZING,
            BenchmarkState.STOPPING)

    val isBatchRunning: Boolean
        get() = _batchProgress.value.isRunning

    // --- CSV Record Types ---

    /** Record Type 1: Generation Summary */
    data class GenerateSummaryRecord(
        val trialId: Int,
        val modelVariant: String,
        val prompt: String,
        val steps: Int,
        val actualSteps: Int,
        val guidanceScale: Float,
        val backendSd: String,
        val precisionSd: String,
        val generateE2eMs: Float,
        val tokenizeMs: Float,
        val textEncMs: Float,
        val unetTotalMs: Float,
        val vaeDecMs: Float,
        val unetPerStepMeanMs: Float,
        val unetPerStepP95Ms: Float,
        val schedulerOverheadMs: Float,
        val peakMemoryMb: Int,
        val startTempC: Float,
        val endTempC: Float,
        val avgPowerMw: Float,
        val pipelineWallClockMs: Float = 0f,
        val trialWallClockMs: Float = 0f,
        val nativeHeapMb: Float = 0f,
        val pssMb: Float = -1f
    )

    /** Record Type 2: UNet Step Detail */
    data class UnetStepRecord(
        val trialId: Int,
        val stepIndex: Int,
        val inputCreateMs: Float,
        val sessionRunMs: Float,
        val outputCopyMs: Float,
        val schedulerStepMs: Float,
        val stepTotalMs: Float,
        val thermalC: Float,
        val powerMw: Float,
        val memoryMb: Int = 0,
        val nativeHeapMb: Float = 0f
    )

    /** Record Type 3: Cold Start */
    data class ColdStartRecord(
        val startType: String,
        val textEncLoadMs: Long,
        val unetLoadMs: Long,
        val vaeDecLoadMs: Long,
        val totalLoadMs: Long,
        val initWallClockMs: Long = 0,
        val parallelInit: Boolean = false,
        val idleMemoryMb: Int,
        val peakMemoryAfterLoadMb: Int,
        val idleThermalC: Float,
        val idlePowerMw: Float,          // 5s 10-sample median (load 전 baseline)
        val firstInferenceWallClockMs: Float = 0f,
        val coldStartTotalMs: Float = 0f,
        val warmupTotalMs: Long = 0,
        val thermalZoneType: String = "unknown",
        val isCharging: Boolean = false
    )

    // --- Benchmark Control ---

    fun start(config: BenchmarkConfig, scope: CoroutineScope) {
        if (isRunning || isBatchRunning) return
        this.config = config
        wakeLock.acquire(30 * 60 * 1000L)  // 30 min max

        benchmarkJob = scope.launch(Dispatchers.Default) {
            try {
                runBenchmark(config)
            } catch (e: CancellationException) {
                Log.i(TAG, "Benchmark cancelled")
            } catch (e: Exception) {
                Log.e(TAG, "Benchmark error: ${e.message}", e)
                _progress.update { it.copy(state = BenchmarkState.ERROR, errorMessage = "Failed: ${e.message}") }
            } finally {
                withContext(NonCancellable) {
                    cleanup()
                }
            }
        }
    }

    fun stop() {
        _progress.update { it.copy(state = BenchmarkState.STOPPING) }
        benchmarkJob?.cancel()
        systemMetricsJob?.cancel()
    }

    fun stopBatch() {
        batchJob?.cancel()
        stop()
        _batchProgress.value = BatchProgress()
    }

    // --- Core Benchmark ---

    private suspend fun runBenchmark(config: BenchmarkConfig) {
        _progress.value = BenchmarkProgress(state = BenchmarkState.INITIALIZING)
        lastExportConfig = config

        generateSummaries.clear()
        unetStepDetails.clear()
        coldStartRecord = null

        val captureScopeJob = Job()
        val captureScope = CoroutineScope(Dispatchers.IO + captureScopeJob)
        logcatCapture.startCapture(
            listOf("onnxruntime", "OrtRunner", "Txt2ImgPipeline", "QNN"), captureScope)
        try {
            pipeline?.release()
            pipeline = null
            // Hint JVM GC and give QNN DSP memory time to settle before next session init.
            System.gc()
            delay(1000)

            runTxt2ImgBenchmark(config)

            // 활성 추론은 완료됐으므로 즉시 STOPPING으로 전환해 UI에 반영.
            // IDLE이 아닌 STOPPING을 사용하는 이유: isRunning에 STOPPING이 포함되어 있어
            // cleanup() 완료 전에 새 실행이 시작되는 race condition을 막을 수 있음.
            // IDLE 전환은 cleanup()에서 pipeline = null 이후에만 수행.
            if (_progress.value.state != BenchmarkState.ERROR) {
                _progress.update { it.copy(state = BenchmarkState.STOPPING) }
            }

            Log.i(TAG, "Benchmark complete: generate=${generateSummaries.size}")
        } finally {
            systemMetricsJob?.cancel()
            systemMetricsJob = null
            try { logcatCapture.stopCapture() } catch (e: Exception) { Log.w(TAG, "logcatCapture.stopCapture failed: ${e.message}") }
            captureScopeJob.cancel()
            try {
                val modelKey = "sd_${config.modelVariant.name.lowercase()}_${config.sdPrecision.dirSuffix}_${config.sdBackend.name.lowercase()}"
                // EpContext (.bin) 모델은 세션 생성 시 그래프 파티션 로그를 출력하지 않음.
                // 로그를 전부 스캔해도 항상 빈 결과 → 수만 줄 파싱을 skip하고 캐시만 조회.
                val isEpContext = java.io.File(config.modelDir).listFiles()
                    ?.any { it.extension == "bin" } == true
                if (!isEpContext) {
                    val parsed = withTimeoutOrNull(5000) { logcatCapture.parseOrtInfo() }
                    if (parsed != null) {
                        lastOrtLogInfo = parsed
                        if (parsed.hasData()) {
                            OrtLogInfo.saveForModel(context, modelKey, parsed)
                            Log.i(TAG, "Saved partition info for $modelKey")
                        }
                    }
                } else {
                    Log.i(TAG, "EpContext model: skipping ORT log parse, loading cached partition info")
                }
                if (lastOrtLogInfo == null || !lastOrtLogInfo!!.hasData()) {
                    val cached = OrtLogInfo.loadForModel(context, modelKey)
                    if (cached != null) {
                        lastOrtLogInfo = cached
                        Log.i(TAG, "Loaded cached partition info for $modelKey: QNN=${cached.qnnNodes} CPU=${cached.cpuNodes}")
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "ORT log parse/save failed: ${e.message}")
            }
        }
    }

    /**
     * Text-to-image benchmark: Phase 1 & 2.
     */
    private suspend fun runTxt2ImgBenchmark(config: BenchmarkConfig) {
        Log.i(TAG, "Initializing Txt2Img Pipeline: $config")

        // Idle baseline: 5s, 10 samples at 500ms interval, take median
        Log.i(TAG, "Collecting idle baseline (5s, 10 samples)...")
        val idlePowerSamples = mutableListOf<Float>()
        val idleThermalSamples = mutableListOf<Float>()
        var idleMetrics = kpiCollector.collectAll()
        repeat(10) {
            val m = kpiCollector.collectAll()
            if (m.powerMw > 0) idlePowerSamples.add(m.powerMw)
            if (m.thermalC > 0) idleThermalSamples.add(m.thermalC)
            idleMetrics = m  // keep last for memory baseline
            delay(500)
        }
        val idleBaselinePowerMw = if (idlePowerSamples.isNotEmpty()) {
            idlePowerSamples.sorted().let { it[it.size / 2] }
        } else -1f
        val idleBaselineThermalC = if (idleThermalSamples.isNotEmpty()) {
            idleThermalSamples.sorted().let { it[it.size / 2] }
        } else -1f
        val isCharging = kpiCollector.isCharging()
        val thermalZoneType = kpiCollector.getThermalZoneType()
        Log.i(TAG, "Idle baseline: mem=${idleMetrics.memoryMb}MB, thermal=${idleBaselineThermalC}C, " +
                "power=${idleBaselinePowerMw}mW (${idlePowerSamples.size} samples), " +
                "charging=$isCharging, thermalZone=$thermalZoneType")
        if (isCharging) {
            Log.w(TAG, "⚠ Device is charging — power measurements will be unreliable")
        }

        val pipe = Txt2ImgPipeline(context, config)
        val initialized = withContext(Dispatchers.IO) { pipe.initialize(config.parallelInit) }
        if (!initialized) {
            _progress.update { it.copy(state = BenchmarkState.ERROR, errorMessage = "Pipeline initialization failed: ${pipe.lastError ?: "unknown"}") }
            return
        }
        pipeline = pipe

        // First inference timing (post-load, pre-warmup — captures cold inference cost)
        val firstInfStart = System.nanoTime()
        withContext(Dispatchers.IO) { pipe.generate()?.outputImage?.recycle() }
        val firstInferenceWallClockMs = nsToMs(System.nanoTime() - firstInfStart)
        Log.i(TAG, "First inference wall-clock: ${firstInferenceWallClockMs}ms")

        // Cold start record (with idle baseline + first inference)
        pipe.coldStartTiming?.let { cs ->
            val memAfterLoad = kpiCollector.readMemory()
            coldStartRecord = ColdStartRecord(
                startType = "cold",
                textEncLoadMs = cs.textEncLoadMs,
                unetLoadMs = cs.unetLoadMs,
                vaeDecLoadMs = cs.vaeDecLoadMs,
                totalLoadMs = cs.totalLoadMs,
                initWallClockMs = cs.initWallClockMs,
                parallelInit = cs.parallelInit,
                idleMemoryMb = idleMetrics.memoryMb,
                peakMemoryAfterLoadMb = memAfterLoad,
                idleThermalC = idleBaselineThermalC,
                idlePowerMw = idleBaselinePowerMw,   // 10-sample median
                firstInferenceWallClockMs = firstInferenceWallClockMs,
                coldStartTotalMs = cs.initWallClockMs.toFloat() + firstInferenceWallClockMs,
                thermalZoneType = thermalZoneType,
                isCharging = isCharging
            )
        }

        // Warmup
        _progress.update { it.copy(state = BenchmarkState.WARMING_UP) }
        val warmupTotalMs = withContext(Dispatchers.IO) {
            pipe.warmup(config.warmupTrials)
        }
        coldStartRecord = coldStartRecord?.copy(warmupTotalMs = warmupTotalMs)
        Log.i(TAG, "Warmup total: ${warmupTotalMs}ms")

        // System metrics
        systemMetricsJob?.cancel()
        metricsScope?.cancel()
        val newMetricsScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
        metricsScope = newMetricsScope
        systemMetricsJob = newMetricsScope.launch { collectSystemMetrics() }

        _progress.update { it.copy(state = BenchmarkState.RUNNING) }

        for (i in 0 until config.trials) {
            if (!currentCoroutineContext().isActive) break
            val trialId = i + 1
            _progress.update { it.copy(currentTrial = trialId, totalTrials = config.trials) }

            val trial = runGenerateTrial(pipe)
            if (trial != null) {
                recordGenerateResult(trialId, config, trial)
                _lastGeneratedImage.value = trial.generateResult.outputImage
            } else {
                Log.e(TAG, "Trial $trialId failed (generate returned null)")
                _progress.update { it.copy(currentStage = "trial_failed") }
            }

            // Cooldown only in Phase 1
            if (config.phase == BenchmarkPhase.SINGLE_GENERATE && i < config.trials - 1) {
                cooldown()
            }
        }

        _progress.update { it.copy(currentStage = "done") }

        // Collect UNet profiling summary (SINGLE_GENERATE phase only).
        if (config.phase == BenchmarkPhase.SINGLE_GENERATE) {
            withTimeoutOrNull(5000) {
                pipe.getUnetProfilingSummary()?.let { summary ->
                    Log.i(TAG, "UNet profiling: total=${summary.totalRunUs}us, " +
                            "npu=${summary.npuComputeUs}us, cpu=${summary.cpuComputeUs}us, " +
                            "fence=${summary.fenceUs}us, iterations=${summary.iterationCount}")
                }
            } ?: Log.w(TAG, "UNet profiling timed out (5s)")
        }
    }

    // --- Single Trial ---

    private data class StepMetricSnapshot(
        val thermalC: Float,
        val powerMw: Float,
        val memoryMb: Int,
        val nativeHeapMb: Float
    )

    private data class GenerateTrialResult(
        val generateResult: Txt2ImgPipeline.GenerateResult,
        val startTempC: Float,
        val endTempC: Float,
        val avgPowerMw: Float,
        val peakMemoryMb: Int,
        val trialWallClockMs: Float = 0f,
        val peakNativeHeapMb: Float = 0f,
        val peakPssMb: Float = -1f,
        val stepSnapshots: List<StepMetricSnapshot> = emptyList()
    )

    private suspend fun runGenerateTrial(pipe: Txt2ImgPipeline): GenerateTrialResult? {
        val trialWallClockStart = System.nanoTime()
        val startTemp = kpiCollector.readThermal()
        val powerSamples = mutableListOf<Float>()
        var peakMemory = lastSystemMemory
        var peakNativeHeap = lastSystemNativeHeapMb
        val stepSnapshots = mutableListOf<StepMetricSnapshot>()

        val listener = object : Txt2ImgPipeline.ProgressListener {
            override fun onStageStart(stage: String) {
                _progress.update { it.copy(currentStage = stage) }
                val power = lastSystemPower
                if (power > 0) powerSamples.add(power)
                val mem = lastSystemMemory
                if (mem > peakMemory) peakMemory = mem
                val nativeHeap = lastSystemNativeHeapMb
                if (nativeHeap > peakNativeHeap) peakNativeHeap = nativeHeap
            }
            override fun onUnetStep(step: Int, totalSteps: Int) {
                _progress.update { it.copy(currentStep = step + 1, totalSteps = totalSteps) }
                val thermal = lastSystemThermal
                val power = lastSystemPower
                val mem = lastSystemMemory
                val nativeHeap = lastSystemNativeHeapMb
                stepSnapshots.add(StepMetricSnapshot(thermal, power, mem, nativeHeap))
                if (power > 0) powerSamples.add(power)
                if (mem > peakMemory) peakMemory = mem
                if (nativeHeap > peakNativeHeap) peakNativeHeap = nativeHeap
            }
        }

        val generateResult = withContext(Dispatchers.IO) {
            pipe.generate(listener)
        } ?: return null

        val endTemp = kpiCollector.readThermal()
        val avgPower = if (powerSamples.isNotEmpty()) powerSamples.average().toFloat() else lastSystemPower
        // PSS: measured once per trial (expensive call)
        val peakPss = kpiCollector.readPss()
        val trialWallClockMs = nsToMs(System.nanoTime() - trialWallClockStart)

        return GenerateTrialResult(
            generateResult = generateResult,
            startTempC = startTemp,
            endTempC = endTemp,
            avgPowerMw = avgPower,
            peakMemoryMb = peakMemory,
            trialWallClockMs = trialWallClockMs,
            peakNativeHeapMb = peakNativeHeap,
            peakPssMb = peakPss,
            stepSnapshots = stepSnapshots
        )
    }

    private fun recordGenerateResult(
        trialId: Int,
        config: BenchmarkConfig,
        trial: GenerateTrialResult
    ) {
        val result = trial.generateResult
        val staging = result.stageTiming
        val stepRunTimes = result.stepDetails.map { it.sessionRunMs }
        val sortedTimes = stepRunTimes.sorted()
        val p95Index = ((sortedTimes.size * 0.95).toInt()).coerceAtMost(sortedTimes.size - 1)

        generateSummaries.add(GenerateSummaryRecord(
            trialId = trialId,
            modelVariant = config.modelVariant.name,
            prompt = config.prompt,
            steps = config.steps,
            actualSteps = result.actualSteps,
            guidanceScale = config.guidanceScale,
            backendSd = config.sdBackend.name,
            precisionSd = config.sdPrecision.displayName,
            generateE2eMs = staging.generateE2eMs,
            tokenizeMs = staging.tokenizeMs,
            textEncMs = staging.textEncMs,
            unetTotalMs = staging.unetTotalMs,
            vaeDecMs = staging.vaeDecMs,
            unetPerStepMeanMs = if (stepRunTimes.isNotEmpty()) stepRunTimes.average().toFloat() else 0f,
            unetPerStepP95Ms = if (sortedTimes.isNotEmpty()) sortedTimes[p95Index] else 0f,
            schedulerOverheadMs = staging.schedulerOverheadMs,
            peakMemoryMb = trial.peakMemoryMb,
            startTempC = trial.startTempC,
            endTempC = trial.endTempC,
            avgPowerMw = trial.avgPowerMw,
            pipelineWallClockMs = staging.pipelineWallClockMs,
            trialWallClockMs = trial.trialWallClockMs,
            nativeHeapMb = trial.peakNativeHeapMb,
            pssMb = trial.peakPssMb
        ))

        for ((i, detail) in result.stepDetails.withIndex()) {
            val snap = trial.stepSnapshots.getOrNull(i)
            unetStepDetails.add(UnetStepRecord(
                trialId = trialId,
                stepIndex = detail.stepIndex,
                inputCreateMs = detail.inputCreateMs,
                sessionRunMs = detail.sessionRunMs,
                outputCopyMs = detail.outputCopyMs,
                schedulerStepMs = detail.schedulerStepMs,
                stepTotalMs = detail.stepTotalMs,
                thermalC = snap?.thermalC ?: lastSystemThermal,
                powerMw = snap?.powerMw ?: lastSystemPower,
                memoryMb = snap?.memoryMb ?: lastSystemMemory,
                nativeHeapMb = snap?.nativeHeapMb ?: lastSystemNativeHeapMb
            ))
        }

        _progress.update { it.copy(lastE2eMs = staging.generateE2eMs) }
    }

    // --- Cooldown ---

    private suspend fun cooldown() {
        Log.i(TAG, "Cooldown: min ${COOLDOWN_MIN_SECONDS}s...")
        for (remaining in COOLDOWN_MIN_SECONDS downTo 1) {
            if (!currentCoroutineContext().isActive) return
            _progress.update { it.copy(currentStage = "cooldown (${remaining}s)") }
            _batchProgress.value = _batchProgress.value.copy(
                isCoolingDown = true, cooldownRemainingSeconds = remaining)
            delay(1000)
        }
        val temp = kpiCollector.readThermal()
        if (temp > COOLDOWN_TARGET_TEMP_C) {
            val extraMax = COOLDOWN_MAX_SECONDS - COOLDOWN_MIN_SECONDS
            for (elapsed in 1..extraMax) {
                if (!currentCoroutineContext().isActive) return
                val remaining = extraMax - elapsed
                _progress.update {
                    it.copy(currentStage = "cooldown (${remaining}s, ${String.format("%.1f", kpiCollector.readThermal())}°C)")
                }
                _batchProgress.value = _batchProgress.value.copy(
                    isCoolingDown = true, cooldownRemainingSeconds = remaining)
                delay(1000)
                if (kpiCollector.readThermal() <= COOLDOWN_TARGET_TEMP_C) break
            }
        }
        _progress.update { it.copy(currentStage = "ready") }
        _batchProgress.value = _batchProgress.value.copy(
            isCoolingDown = false, cooldownRemainingSeconds = 0)
        Log.i(TAG, "Cooldown complete: ${kpiCollector.readThermal()}°C")
    }

    // --- System Metrics ---

    private suspend fun collectSystemMetrics() {
        while (currentCoroutineContext().isActive) {
            val metrics = kpiCollector.collectAll()
            lastSystemThermal = metrics.thermalC
            lastSystemPower = metrics.powerMw
            lastSystemMemory = metrics.memoryMb
            lastSystemNativeHeapMb = metrics.nativeHeapMb
            _progress.update { current ->
                if (current.state == BenchmarkState.RUNNING || current.state == BenchmarkState.WARMING_UP) {
                    current.copy(
                        lastThermalC = metrics.thermalC,
                        lastPowerMw = metrics.powerMw,
                        lastMemoryMb = metrics.memoryMb)
                } else {
                    current
                }
            }
            delay(SYSTEM_METRICS_INTERVAL_MS)
        }
    }

    // --- CSV Export ---

    fun exportCsv(): String {
        val sb = StringBuilder()
        val cfg = config ?: lastExportConfig ?: return ""

        val deviceInfo = KpiCollector.getDeviceInfoMap()
        sb.appendLine("# device_model,${deviceInfo["device_model"]}")
        sb.appendLine("# soc_model,${deviceInfo["soc_model"]}")
        sb.appendLine("# runtime,ONNX Runtime ${BuildConfig.ORT_VERSION}")
        sb.appendLine("# model_variant,${cfg.modelVariant.name}")
        sb.appendLine("# sd_backend,${cfg.sdBackend.name}")
        sb.appendLine("# sd_precision,${cfg.sdPrecision.displayName}")
        if (cfg.isMixedPrecision) {
            sb.appendLine("# sd_precision_per_component,${SdComponent.values().joinToString(";") { "${it.baseName}=${cfg.sdPrecisionFor(it).dirSuffix}" }}")
        }
        sb.appendLine("# prompt,\"${cfg.prompt.replace("\"", "\"\"")}\"")
        sb.appendLine("# steps,${cfg.steps}")
        sb.appendLine("# guidance_scale,${cfg.guidanceScale}")
        sb.appendLine("# phase,${cfg.phase.name}")

        // System measurement metadata
        coldStartRecord?.let { cs ->
            sb.appendLine("# thermal_zone_type,${cs.thermalZoneType}")
            sb.appendLine("# is_charging,${cs.isCharging}")
            sb.appendLine("# idle_baseline_power_mw,${"%.1f".format(cs.idlePowerMw)}")
        }

        // ORT graph partitioning metadata
        lastOrtLogInfo?.let { ort ->
            sb.appendLine("# ort_total_nodes,${ort.totalNodes}")
            sb.appendLine("# ort_qnn_nodes,${ort.qnnNodes}")
            sb.appendLine("# ort_cpu_nodes,${ort.cpuNodes}")
            if (ort.fallbackOps.isNotEmpty()) {
                sb.appendLine("# ort_fallback_ops,${ort.fallbackOps.joinToString(";")}")
            }
        }

        // Record Type 1: Generation Summary
        sb.appendLine()
        sb.appendLine("# GENERATE_SUMMARY")
        sb.appendLine("trial_id,model_variant,prompt,steps,actual_steps,guidance_scale," +
                "backend_sd,precision_sd," +
                "generate_e2e_ms,tokenize_ms,text_enc_ms," +
                "unet_total_ms,vae_dec_ms," +
                "unet_per_step_mean_ms,unet_per_step_p95_ms,scheduler_overhead_ms," +
                "peak_memory_mb,start_temp_c,end_temp_c,avg_power_mw," +
                "pipeline_wall_clock_ms,trial_wall_clock_ms,native_heap_mb,pss_mb")
        for (r in generateSummaries) {
            val escapedPrompt = "\"${r.prompt.replace("\"", "\"\"")}\""
            sb.appendLine("${r.trialId},${r.modelVariant},$escapedPrompt,${r.steps},${r.actualSteps},${r.guidanceScale}," +
                    "${r.backendSd},${r.precisionSd}," +
                    "${"%.2f".format(r.generateE2eMs)},${"%.2f".format(r.tokenizeMs)},${"%.2f".format(r.textEncMs)}," +
                    "${"%.2f".format(r.unetTotalMs)},${"%.2f".format(r.vaeDecMs)}," +
                    "${"%.2f".format(r.unetPerStepMeanMs)},${"%.2f".format(r.unetPerStepP95Ms)}," +
                    "${"%.2f".format(r.schedulerOverheadMs)}," +
                    "${r.peakMemoryMb},${"%.1f".format(r.startTempC)},${"%.1f".format(r.endTempC)},${"%.1f".format(r.avgPowerMw)}," +
                    "${"%.2f".format(r.pipelineWallClockMs)},${"%.2f".format(r.trialWallClockMs)}," +
                    "${"%.1f".format(r.nativeHeapMb)},${"%.1f".format(r.pssMb)}")
        }

        // Record Type 2: UNet Step Detail
        sb.appendLine()
        sb.appendLine("# UNET_STEP_DETAIL")
        sb.appendLine("trial_id,step_index,input_create_ms,session_run_ms,output_copy_ms," +
                "scheduler_step_ms,step_total_ms,thermal_c,power_mw," +
                "memory_mb,native_heap_mb")
        for (r in unetStepDetails) {
            sb.appendLine("${r.trialId},${r.stepIndex}," +
                    "${"%.2f".format(r.inputCreateMs)},${"%.2f".format(r.sessionRunMs)}," +
                    "${"%.2f".format(r.outputCopyMs)},${"%.2f".format(r.schedulerStepMs)}," +
                    "${"%.2f".format(r.stepTotalMs)}," +
                    "${"%.1f".format(r.thermalC)},${"%.1f".format(r.powerMw)}," +
                    "${r.memoryMb},${"%.1f".format(r.nativeHeapMb)}")
        }

        // Record Type 3: Cold Start
        sb.appendLine()
        sb.appendLine("# COLD_START")
        sb.appendLine("start_type,text_enc_load_ms," +
                "unet_load_ms,vae_dec_load_ms,total_load_ms,init_wall_clock_ms,parallel_init," +
                "idle_memory_mb,peak_memory_after_load_mb,memory_delta_mb," +
                "idle_thermal_c,idle_power_mw," +
                "first_inference_wall_clock_ms,cold_start_total_ms,warmup_total_ms," +
                "thermal_zone_type,is_charging")
        coldStartRecord?.let { r ->
            val memDelta = r.peakMemoryAfterLoadMb - r.idleMemoryMb
            sb.appendLine("${r.startType},${r.textEncLoadMs}," +
                    "${r.unetLoadMs},${r.vaeDecLoadMs}," +
                    "${r.totalLoadMs},${r.initWallClockMs},${r.parallelInit}," +
                    "${r.idleMemoryMb},${r.peakMemoryAfterLoadMb},${memDelta}," +
                    "${"%.1f".format(r.idleThermalC)},${"%.1f".format(r.idlePowerMw)}," +
                    "${"%.2f".format(r.firstInferenceWallClockMs)},${"%.2f".format(r.coldStartTotalMs)},${r.warmupTotalMs}," +
                    "${r.thermalZoneType},${r.isCharging}")
        }

        return sb.toString()
    }

    fun getRecordCount(): Int = generateSummaries.size

    fun exportAndSaveCsv(): String? {
        val csvData = exportCsv()
        if (csvData.isEmpty()) return null

        return try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val cfg = config ?: lastExportConfig ?: return null
            val phaseTag = when (cfg.phase) {
                BenchmarkPhase.SINGLE_GENERATE -> "single"
                BenchmarkPhase.SUSTAINED_GENERATE -> "sustained"
            }
            val variantTag = when (cfg.modelVariant) {
                ModelVariant.SD_V15 -> "sd15"
                ModelVariant.LCM_LORA -> "lcm"
            }
            val precTag = if (cfg.isMixedPrecision) {
                "mixed_" + SdComponent.values().joinToString("_") {
                    "${it.baseName[0]}${cfg.sdPrecisionFor(it).dirSuffix.replace('_', '-')}"
                }
            } else {
                cfg.sdPrecision.dirSuffix
            }
            val baseFilename = "txt2img_${variantTag}_${precTag}_${cfg.sdBackend.name.lowercase()}_" +
                    "s${cfg.steps}_${phaseTag}_${timestamp}"

            val exportDir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            if (exportDir != null) {
                val csvFile = File(exportDir, "${baseFilename}.csv")
                FileWriter(csvFile).use { it.write(csvData) }
                Log.i(TAG, "CSV: ${csvFile.absolutePath}")

                lastOrtLogInfo?.let { ortInfo ->
                    if (ortInfo.rawLogs.isNotBlank()) {
                        val logFile = File(exportDir, "${baseFilename}_ort.log")
                        FileWriter(logFile).use { writer ->
                            writer.write(ortInfo.toSummary())
                            writer.write("\n\n=== Raw Logs ===\n")
                            writer.write(ortInfo.rawLogs)
                        }
                    }
                }

                _lastGeneratedImage.value?.let { bitmap ->
                    val imgFile = File(exportDir, "${baseFilename}_last.png")
                    imgFile.outputStream().use { out ->
                        bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, out)
                    }
                    Log.i(TAG, "Image: ${imgFile.absolutePath}")
                }

                val promptFile = File(exportDir, "${baseFilename}_prompt.txt")
                promptFile.writeText(cfg.prompt)
                Log.i(TAG, "Prompt: ${promptFile.absolutePath}")

                csvFile.absolutePath
            } else null
        } catch (e: Exception) {
            Log.e(TAG, "Export failed: ${e.message}", e)
            null
        }
    }

    fun getOrtLogInfo(): OrtLogInfo? = lastOrtLogInfo

    // --- Batch ---

    fun startBatch(
        experimentSet: ExperimentSet,
        defaults: ExperimentDefaults,
        scope: CoroutineScope,
        onExperimentComplete: (csvPath: String) -> Unit = {}
    ) {
        if (isRunning || isBatchRunning) return
        wakeLock.acquire(3 * 60 * 60 * 1000L)  // 3 hours max for batch

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
                    if (!isActive) break
                    _batchProgress.value = _batchProgress.value.copy(
                        currentExperimentIndex = index + 1,
                        currentExperimentName = experiment.getDisplayName(),
                        isCoolingDown = false
                    )

                    val cfg = experiment.toBenchmarkConfig(defaults)
                    this@BenchmarkRunner.config = cfg

                    try {
                        runBenchmark(cfg)
                    } catch (e: CancellationException) { throw e }
                    catch (e: Throwable) {
                        Log.e(TAG, "Batch experiment failed: ${e.message}", e)
                        continue
                    }

                    val csvPath = exportAndSaveCsv()
                    if (csvPath != null) {
                        completedFiles.add(csvPath)
                        _batchProgress.value = _batchProgress.value.copy(
                            completedExperiments = completedFiles.toList()
                        )
                        onExperimentComplete(csvPath)
                    }

                    if (index < experimentSet.experiments.size - 1 && isActive) {
                        cooldown()
                    }
                }
            } catch (e: CancellationException) {
                Log.i(TAG, "Batch cancelled")
            } finally {
                _batchProgress.value = BatchProgress(completedExperiments = completedFiles.toList())
                withContext(NonCancellable) {
                    cleanup()
                }
            }
        }
    }

    // --- Cleanup ---

    private suspend fun cleanup() {
        systemMetricsJob?.let { job ->
            job.cancel()
            withTimeoutOrNull(3000) { job.join() }
        }
        systemMetricsJob = null
        metricsScope?.cancel()
        metricsScope = null
        try {
            if (logcatCapture.isCapturing()) logcatCapture.stopCapture()
        } catch (e: Exception) {
            Log.w(TAG, "logcatCapture.stopCapture failed: ${e.message}")
        }

        config = null
        Log.i(TAG, "Cleanup: releasing models...")

        pipeline?.release()
        pipeline = null
        if (wakeLock.isHeld) wakeLock.release()

        // IDLE 전환은 pipeline 해제 완료 후에 수행.
        // IDLE을 먼저 설정하면 start()의 isRunning 가드를 통과한 2차 실행이
        // cleanup()과 동시에 pipeline?.release()를 호출해 OrtSession 이중 close 발생.
        if (_progress.value.state != BenchmarkState.ERROR) {
            _progress.update { it.copy(state = BenchmarkState.IDLE) }
        }
        Log.i(TAG, "Cleanup complete")
    }

    fun release() {
        stopBatch()  // internally calls stop()
        if (logcatCapture.isCapturing()) logcatCapture.stopCapture()
        pipeline?.release()
        pipeline = null
        _lastGeneratedImage.value?.recycle()
        _lastGeneratedImage.value = null
    }

    private fun nsToMs(ns: Long): Float = (ns / 1_000_000.0).toFloat()
}
