package com.example.kpilab

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.os.Environment
import android.util.Log
import com.example.kpilab.batch.BatchProgress
import com.example.kpilab.batch.ExperimentDefaults
import com.example.kpilab.batch.ExperimentSet
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
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
 * AI Eraser benchmark runner.
 * Phase 1 (SINGLE_ERASE): 5 trials, cooldown between, pre-defined mask.
 * Phase 2 (SUSTAINED_ERASE): 10 consecutive trials, no cooldown.
 * YOLO_SEG_ONLY: 20 trials YOLO-seg standalone.
 * Outputs 4 CSV record types per experiment_design.md Appendix C.
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

    private var pipeline: InpaintPipeline? = null
    private var benchmarkJob: Job? = null
    private var systemMetricsJob: Job? = null
    private var batchJob: Job? = null
    private var config: BenchmarkConfig? = null

    // Source image + pre-defined mask
    private var sourceImage: Bitmap? = null
    private var sourceMask: Bitmap? = null

    // Collected data for CSV export
    private val eraseSummaries = mutableListOf<EraseSummaryRecord>()
    private val unetStepDetails = mutableListOf<UnetStepRecord>()
    private var coldStartRecord: ColdStartRecord? = null
    private val yoloSegDetails = mutableListOf<YoloSegRecord>()
    private var lastSystemThermal: Float = 0f
    private var lastSystemPower: Float = 0f
    private var lastSystemMemory: Int = 0

    // Logcat capture
    private val logcatCapture = LogcatCapture()
    private var lastOrtLogInfo: OrtLogInfo? = null

    val isRunning: Boolean
        get() = _progress.value.state in listOf(
            BenchmarkState.RUNNING, BenchmarkState.WARMING_UP, BenchmarkState.INITIALIZING)

    val isBatchRunning: Boolean
        get() = _batchProgress.value.isRunning

    // --- CSV Record Types (Appendix C) ---

    /** Record Type 1: Erase Summary */
    data class EraseSummaryRecord(
        val trialId: Int,
        val prompt: String,
        val steps: Int,
        val strength: Float,
        val actualSteps: Int,
        val roiSize: String,
        val backendSd: String,
        val precisionSd: String,
        val backendYolo: String,
        val precisionYolo: String,
        val fullE2eMs: Float,
        val inpaintE2eMs: Float,
        val yoloSegMs: Float,
        val roiCropMs: Float,
        val tokenizeMs: Float,
        val textEncMs: Float,
        val vaeEncMs: Float,
        val maskedImgPrepMs: Float,
        val unetTotalMs: Float,
        val vaeDecMs: Float,
        val compositeMs: Float,
        val unetPerStepMeanMs: Float,
        val unetPerStepP95Ms: Float,
        val schedulerOverheadMs: Float,
        val peakMemoryMb: Int,
        val startTempC: Float,
        val endTempC: Float,
        val avgPowerMw: Float
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
        val powerMw: Float
    )

    /** Record Type 3: Cold Start */
    data class ColdStartRecord(
        val startType: String,
        val yoloSegLoadMs: Long,
        val vaeEncLoadMs: Long,
        val textEncLoadMs: Long,
        val unetLoadMs: Long,
        val vaeDecLoadMs: Long,
        val totalLoadMs: Long,
        val peakMemoryAfterLoadMb: Int
    )

    /** Record Type 4: YOLO-seg Detail */
    data class YoloSegRecord(
        val trialId: Int,
        val testImage: String,
        val backend: String,
        val precision: String,
        val inferenceMs: Float,
        val inputCreateMs: Float,
        val nmsMs: Float,
        val maskDecodeMs: Float,
        val outputProcessMs: Float,
        val maskCount: Int,
        val selectedMaskAreaPct: Float
    )

    // --- Benchmark Control ---

    fun start(config: BenchmarkConfig, scope: CoroutineScope) {
        if (isRunning || isBatchRunning) return
        this.config = config

        benchmarkJob = scope.launch(Dispatchers.Default) {
            try {
                runBenchmark(config)
            } catch (e: CancellationException) {
                Log.i(TAG, "Benchmark cancelled")
            } catch (e: Exception) {
                Log.e(TAG, "Benchmark error: ${e.message}", e)
                _progress.value = BenchmarkProgress(
                    state = BenchmarkState.ERROR,
                    errorMessage = "Failed: ${e.message}"
                )
            } finally {
                cleanup()
            }
        }
    }

    fun stop() {
        _progress.value = _progress.value.copy(state = BenchmarkState.STOPPING)
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

        eraseSummaries.clear()
        unetStepDetails.clear()
        coldStartRecord = null
        yoloSegDetails.clear()

        val captureScope = CoroutineScope(currentCoroutineContext())
        logcatCapture.startCapture(
            listOf("onnxruntime", "OrtRunner", "InpaintPipeline", "QNN"), captureScope)

        // Load test data (image + pre-defined mask)
        loadTestData(config)

        pipeline?.release()
        pipeline = null

        when (config.phase) {
            BenchmarkPhase.YOLO_SEG_ONLY -> {
                // TODO: YoloSegRunner 구현 후 활성화
                Log.w(TAG, "YOLO-seg profiling: not yet implemented")
                _progress.value = BenchmarkProgress(
                    state = BenchmarkState.ERROR,
                    errorMessage = "YOLO-seg Only phase is not yet implemented"
                )
                return
            }
            BenchmarkPhase.SINGLE_ERASE,
            BenchmarkPhase.SUSTAINED_ERASE -> {
                runInpaintBenchmark(config)
            }
        }

        logcatCapture.stopCapture()
        lastOrtLogInfo = logcatCapture.parseOrtInfo()
        Log.i(TAG, "Benchmark complete: ${eraseSummaries.size} erase trials")
    }

    /**
     * Inpainting benchmark: Phase 1 & 2.
     * Pre-defined mask, no YOLO-seg (yolo_seg_ms = 0).
     */
    private suspend fun runInpaintBenchmark(config: BenchmarkConfig) {
        Log.i(TAG, "Initializing Inpaint Pipeline: $config")
        val pipe = InpaintPipeline(context, config)
        val initialized = withContext(Dispatchers.IO) { pipe.initialize() }
        if (!initialized) {
            _progress.value = BenchmarkProgress(
                state = BenchmarkState.ERROR,
                errorMessage = "Pipeline initialization failed"
            )
            return
        }
        pipeline = pipe

        // Cold start record
        pipe.coldStartTiming?.let { cs ->
            val memAfterLoad = kpiCollector.readMemory()
            coldStartRecord = ColdStartRecord(
                startType = "cold",
                yoloSegLoadMs = 0,
                vaeEncLoadMs = cs.vaeEncLoadMs,
                textEncLoadMs = cs.textEncLoadMs,
                unetLoadMs = cs.unetLoadMs,
                vaeDecLoadMs = cs.vaeDecLoadMs,
                totalLoadMs = cs.totalLoadMs,
                peakMemoryAfterLoadMb = memAfterLoad
            )
        }

        val img = sourceImage ?: return
        val mask = sourceMask ?: return

        // ROI crop (measure once — same crop reused across all trials with pre-defined mask)
        val roiCropStart = System.nanoTime()
        val roiResult = ImagePreprocessor.cropRoiWithPadding(img, mask, config.roiPaddingRatio)
        val roiCropMs = nsToMs(System.nanoTime() - roiCropStart)
        if (roiResult == null) {
            _progress.value = BenchmarkProgress(
                state = BenchmarkState.ERROR,
                errorMessage = "ROI crop failed: empty mask"
            )
            return
        }
        Log.i(TAG, "ROI crop: ${roiCropMs}ms (${roiResult.cropRect})")

        // Warmup
        _progress.value = _progress.value.copy(state = BenchmarkState.WARMING_UP)
        withContext(Dispatchers.IO) {
            pipe.warmup(roiResult.roiBitmap, roiResult.roiMask, config.warmupTrials)
        }

        // System metrics (cancel previous if batch mode)
        systemMetricsJob?.cancel()
        val metricsScope = CoroutineScope(currentCoroutineContext())
        systemMetricsJob = metricsScope.launch { collectSystemMetrics() }

        _progress.value = _progress.value.copy(state = BenchmarkState.RUNNING)

        for (i in 0 until config.trials) {
            if (!currentCoroutineContext().isActive) return
            val trialId = i + 1
            _progress.value = _progress.value.copy(
                currentTrial = trialId, totalTrials = config.trials)

            val trial = runEraseTrial(trialId, pipe, img, roiResult, roiCropMs)
            if (trial != null) {
                recordEraseResult(trialId, config, trial)
                // Recycle previous generated image before replacing
                _lastGeneratedImage.value?.recycle()
                _lastGeneratedImage.value = trial.compositedImage
            }

            // Cooldown only in Phase 1
            if (config.phase == BenchmarkPhase.SINGLE_ERASE && i < config.trials - 1) {
                cooldown()
            }
        }

        // Recycle ROI crop bitmaps
        roiResult.roiBitmap.recycle()
        roiResult.roiMask.recycle()

        // Collect UNet profiling summary (SINGLE_ERASE phase only)
        if (config.phase == BenchmarkPhase.SINGLE_ERASE) {
            pipe.getUnetProfilingSummary()?.let { summary ->
                Log.i(TAG, "UNet profiling: total=${summary.totalRunUs}us, " +
                        "npu=${summary.npuComputeUs}us, cpu=${summary.cpuComputeUs}us, " +
                        "fence=${summary.fenceUs}us, iterations=${summary.iterationCount}")
            }
        }
    }

    // --- Single Trial ---

    private data class EraseTrialResult(
        val inpaintResult: InpaintPipeline.InpaintResult,
        val compositedImage: Bitmap,
        val roiCropMs: Float,
        val compositeMs: Float,
        val startTempC: Float,
        val endTempC: Float,
        val avgPowerMw: Float,
        val peakMemoryMb: Int
    )

    private suspend fun runEraseTrial(
        trialId: Int,
        pipe: InpaintPipeline,
        originalImage: Bitmap,
        roiResult: ImagePreprocessor.RoiCropResult,
        roiCropMs: Float
    ): EraseTrialResult? {
        val startTemp = kpiCollector.readThermal()
        val powerSamples = mutableListOf<Float>()
        var peakMemory = lastSystemMemory

        val listener = object : InpaintPipeline.ProgressListener {
            override fun onStageStart(stage: String) {
                _progress.value = _progress.value.copy(currentStage = stage)
                val power = lastSystemPower
                if (power > 0) powerSamples.add(power)
                val mem = lastSystemMemory
                if (mem > peakMemory) peakMemory = mem
            }
            override fun onUnetStep(step: Int, totalSteps: Int) {
                _progress.value = _progress.value.copy(
                    currentStep = step + 1, totalSteps = totalSteps)
                val power = lastSystemPower
                if (power > 0) powerSamples.add(power)
                val mem = lastSystemMemory
                if (mem > peakMemory) peakMemory = mem
            }
        }

        // Inpainting
        val inpaintResult = withContext(Dispatchers.IO) {
            pipe.inpaint(roiResult.roiBitmap, roiResult.roiMask, listener)
        } ?: return null

        // Composite
        _progress.value = _progress.value.copy(currentStage = "composite")
        val compositeStart = System.nanoTime()
        val compositedImage = ImagePreprocessor.composite(
            originalImage, inpaintResult.outputImage,
            roiResult.cropRect, roiResult.roiMask
        )
        val compositeMs = nsToMs(System.nanoTime() - compositeStart)

        val endTemp = kpiCollector.readThermal()
        val avgPower = if (powerSamples.isNotEmpty()) powerSamples.average().toFloat() else lastSystemPower

        return EraseTrialResult(
            inpaintResult = inpaintResult,
            compositedImage = compositedImage,
            roiCropMs = roiCropMs,
            compositeMs = compositeMs,
            startTempC = startTemp,
            endTempC = endTemp,
            avgPowerMw = avgPower,
            peakMemoryMb = peakMemory
        )
    }

    private fun recordEraseResult(
        trialId: Int,
        config: BenchmarkConfig,
        trial: EraseTrialResult
    ) {
        val result = trial.inpaintResult
        val staging = result.stageTiming
        val stepRunTimes = result.stepDetails.map { it.sessionRunMs }
        val sortedTimes = stepRunTimes.sorted()
        val p95Index = ((sortedTimes.size * 0.95).toInt()).coerceAtMost(sortedTimes.size - 1)

        // Inpaint E2E = core pipeline only (tokenize ~ vae_dec)
        val inpaintE2eMs = staging.inpaintE2eMs
        // Full E2E = yolo_seg + roi_crop + inpaint_e2e + composite
        val fullE2eMs = 0f /*yoloSegMs*/ + trial.roiCropMs + inpaintE2eMs + trial.compositeMs

        eraseSummaries.add(EraseSummaryRecord(
            trialId = trialId,
            prompt = config.prompt,
            steps = config.steps,
            strength = config.strength,
            actualSteps = result.actualSteps,
            roiSize = config.roiSize.name,
            backendSd = config.sdBackend.name,
            precisionSd = config.sdPrecision.displayName,
            backendYolo = config.yoloBackend.name,
            precisionYolo = config.yoloPrecision.displayName,
            fullE2eMs = fullE2eMs,
            inpaintE2eMs = inpaintE2eMs,
            yoloSegMs = 0f,
            roiCropMs = trial.roiCropMs,
            tokenizeMs = staging.tokenizeMs,
            textEncMs = staging.textEncMs,
            vaeEncMs = staging.vaeEncMs,
            maskedImgPrepMs = staging.maskedImgPrepMs,
            unetTotalMs = staging.unetTotalMs,
            vaeDecMs = staging.vaeDecMs,
            compositeMs = trial.compositeMs,
            unetPerStepMeanMs = if (stepRunTimes.isNotEmpty()) stepRunTimes.average().toFloat() else 0f,
            unetPerStepP95Ms = if (sortedTimes.isNotEmpty()) sortedTimes[p95Index] else 0f,
            schedulerOverheadMs = staging.schedulerOverheadMs,
            peakMemoryMb = trial.peakMemoryMb,
            startTempC = trial.startTempC,
            endTempC = trial.endTempC,
            avgPowerMw = trial.avgPowerMw
        ))

        for (detail in result.stepDetails) {
            unetStepDetails.add(UnetStepRecord(
                trialId = trialId,
                stepIndex = detail.stepIndex,
                inputCreateMs = detail.inputCreateMs,
                sessionRunMs = detail.sessionRunMs,
                outputCopyMs = detail.outputCopyMs,
                schedulerStepMs = detail.schedulerStepMs,
                stepTotalMs = detail.stepTotalMs,
                thermalC = lastSystemThermal,
                powerMw = lastSystemPower
            ))
        }

        _progress.value = _progress.value.copy(lastE2eMs = inpaintE2eMs)
    }

    // --- Test Data ---

    private fun loadTestData(config: BenchmarkConfig) {
        val roiSize = config.roiSize
        sourceImage?.recycle()
        sourceMask?.recycle()

        sourceImage = try {
            context.assets.open("test_images/${roiSize.testImage}").use {
                BitmapFactory.decodeStream(it)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Test image not found, using sample_image.jpg")
            context.assets.open("sample_image.jpg").use { BitmapFactory.decodeStream(it) }
        }

        sourceMask = try {
            context.assets.open("test_masks/${roiSize.testMask}").use {
                BitmapFactory.decodeStream(it)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Test mask not found, creating dummy mask")
            createDummyMask(sourceImage!!)
        }
    }

    private fun createDummyMask(image: Bitmap): Bitmap {
        val w = image.width; val h = image.height
        val mask = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(w * h)
        val cx = w / 2; val cy = h / 2; val rw = w / 4; val rh = h / 4
        for (y in 0 until h) {
            for (x in 0 until w) {
                pixels[y * w + x] = if (x in (cx - rw)..(cx + rw) && y in (cy - rh)..(cy + rh))
                    0xFFFFFFFF.toInt() else 0xFF000000.toInt()
            }
        }
        mask.setPixels(pixels, 0, w, 0, 0, w, h)
        return mask
    }

    fun setSourceImage(bitmap: Bitmap) {
        sourceImage?.recycle()
        sourceImage = bitmap.copy(bitmap.config, false)
    }

    fun setSourceMask(bitmap: Bitmap) {
        sourceMask?.recycle()
        sourceMask = bitmap.copy(bitmap.config, false)
    }

    // --- Cooldown ---

    private suspend fun cooldown() {
        Log.i(TAG, "Cooldown: min ${COOLDOWN_MIN_SECONDS}s...")
        for (remaining in COOLDOWN_MIN_SECONDS downTo 1) {
            if (!currentCoroutineContext().isActive) return
            _batchProgress.value = _batchProgress.value.copy(
                isCoolingDown = true, cooldownRemainingSeconds = remaining)
            delay(1000)
        }
        val temp = kpiCollector.readThermal()
        if (temp > COOLDOWN_TARGET_TEMP_C) {
            val extraMax = COOLDOWN_MAX_SECONDS - COOLDOWN_MIN_SECONDS
            for (elapsed in 1..extraMax) {
                if (!currentCoroutineContext().isActive) return
                _batchProgress.value = _batchProgress.value.copy(
                    isCoolingDown = true, cooldownRemainingSeconds = extraMax - elapsed)
                delay(1000)
                if (kpiCollector.readThermal() <= COOLDOWN_TARGET_TEMP_C) break
            }
        }
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
            _progress.value = _progress.value.copy(
                lastThermalC = metrics.thermalC,
                lastPowerMw = metrics.powerMw,
                lastMemoryMb = metrics.memoryMb)
            delay(SYSTEM_METRICS_INTERVAL_MS)
        }
    }

    // --- CSV Export ---

    fun exportCsv(): String {
        val sb = StringBuilder()
        val cfg = config ?: return ""

        val deviceInfo = KpiCollector.getDeviceInfoMap()
        sb.appendLine("# device_model,${deviceInfo["device_model"]}")
        sb.appendLine("# soc_model,${deviceInfo["soc_model"]}")
        sb.appendLine("# runtime,ONNX Runtime ${BuildConfig.ORT_VERSION}")
        sb.appendLine("# sd_backend,${cfg.sdBackend.name}")
        sb.appendLine("# sd_precision,${cfg.sdPrecision.displayName}")
        sb.appendLine("# yolo_backend,${cfg.yoloBackend.name}")
        sb.appendLine("# yolo_precision,${cfg.yoloPrecision.displayName}")
        sb.appendLine("# steps,${cfg.steps}")
        sb.appendLine("# strength,${cfg.strength}")
        sb.appendLine("# roi_size,${cfg.roiSize.name}")
        sb.appendLine("# phase,${cfg.phase.name}")

        // Record Type 1: Erase Summary
        sb.appendLine()
        sb.appendLine("# ERASE_SUMMARY")
        sb.appendLine("trial_id,prompt,steps,strength,actual_steps,roi_size," +
                "backend_sd,precision_sd,backend_yolo,precision_yolo," +
                "full_e2e_ms,inpaint_e2e_ms," +
                "yolo_seg_ms,roi_crop_ms,tokenize_ms,text_enc_ms,vae_enc_ms,masked_img_prep_ms," +
                "unet_total_ms,vae_dec_ms,composite_ms," +
                "unet_per_step_mean_ms,unet_per_step_p95_ms,scheduler_overhead_ms," +
                "peak_memory_mb,start_temp_c,end_temp_c,avg_power_mw")
        for (r in eraseSummaries) {
            sb.appendLine("${r.trialId},${r.prompt},${r.steps},${r.strength},${r.actualSteps},${r.roiSize}," +
                    "${r.backendSd},${r.precisionSd},${r.backendYolo},${r.precisionYolo}," +
                    "${"%.2f".format(r.fullE2eMs)},${"%.2f".format(r.inpaintE2eMs)}," +
                    "${"%.2f".format(r.yoloSegMs)},${"%.2f".format(r.roiCropMs)}," +
                    "${"%.2f".format(r.tokenizeMs)},${"%.2f".format(r.textEncMs)}," +
                    "${"%.2f".format(r.vaeEncMs)},${"%.2f".format(r.maskedImgPrepMs)}," +
                    "${"%.2f".format(r.unetTotalMs)},${"%.2f".format(r.vaeDecMs)}," +
                    "${"%.2f".format(r.compositeMs)}," +
                    "${"%.2f".format(r.unetPerStepMeanMs)},${"%.2f".format(r.unetPerStepP95Ms)}," +
                    "${"%.2f".format(r.schedulerOverheadMs)}," +
                    "${r.peakMemoryMb},${"%.1f".format(r.startTempC)},${"%.1f".format(r.endTempC)},${"%.1f".format(r.avgPowerMw)}")
        }

        // Record Type 2: UNet Step Detail
        sb.appendLine()
        sb.appendLine("# UNET_STEP_DETAIL")
        sb.appendLine("trial_id,step_index,input_create_ms,session_run_ms,output_copy_ms," +
                "scheduler_step_ms,step_total_ms,thermal_c,power_mw")
        for (r in unetStepDetails) {
            sb.appendLine("${r.trialId},${r.stepIndex}," +
                    "${"%.2f".format(r.inputCreateMs)},${"%.2f".format(r.sessionRunMs)}," +
                    "${"%.2f".format(r.outputCopyMs)},${"%.2f".format(r.schedulerStepMs)}," +
                    "${"%.2f".format(r.stepTotalMs)}," +
                    "${"%.1f".format(r.thermalC)},${"%.1f".format(r.powerMw)}")
        }

        // Record Type 3: Cold Start
        sb.appendLine()
        sb.appendLine("# COLD_START")
        sb.appendLine("start_type,yolo_seg_load_ms,vae_enc_load_ms,text_enc_load_ms," +
                "unet_load_ms,vae_dec_load_ms,total_load_ms,peak_memory_after_load_mb")
        coldStartRecord?.let { r ->
            sb.appendLine("${r.startType},${r.yoloSegLoadMs},${r.vaeEncLoadMs}," +
                    "${r.textEncLoadMs},${r.unetLoadMs},${r.vaeDecLoadMs}," +
                    "${r.totalLoadMs},${r.peakMemoryAfterLoadMb}")
        }

        // Record Type 4: YOLO-seg Detail
        if (yoloSegDetails.isNotEmpty()) {
            sb.appendLine()
            sb.appendLine("# YOLO_SEG_DETAIL")
            sb.appendLine("trial_id,test_image,backend,precision," +
                    "inference_ms,input_create_ms,nms_ms,mask_decode_ms,output_process_ms," +
                    "mask_count,selected_mask_area_pct")
            for (r in yoloSegDetails) {
                sb.appendLine("${r.trialId},${r.testImage},${r.backend},${r.precision}," +
                        "${"%.2f".format(r.inferenceMs)},${"%.2f".format(r.inputCreateMs)}," +
                        "${"%.2f".format(r.nmsMs)},${"%.2f".format(r.maskDecodeMs)}," +
                        "${"%.2f".format(r.outputProcessMs)}," +
                        "${r.maskCount},${"%.2f".format(r.selectedMaskAreaPct)}")
            }
        }

        return sb.toString()
    }

    fun getRecordCount(): Int = eraseSummaries.size

    fun exportAndSaveCsv(): String? {
        val csvData = exportCsv()
        if (csvData.isEmpty()) return null

        return try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val cfg = config ?: return null
            val phaseTag = when (cfg.phase) {
                BenchmarkPhase.SINGLE_ERASE -> "single"
                BenchmarkPhase.SUSTAINED_ERASE -> "sustained"
                BenchmarkPhase.YOLO_SEG_ONLY -> "yolo"
            }
            val baseFilename = "eraser_${cfg.sdPrecision.dirSuffix}_${cfg.sdBackend.name.lowercase()}_" +
                    "s${cfg.steps}_str${(cfg.strength * 10).toInt()}_${cfg.roiSize.name.lowercase()}_${phaseTag}_${timestamp}"

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

                    if (csvPath != null && index < experimentSet.experiments.size - 1 && isActive) {
                        cooldown()
                    }
                }
            } catch (e: CancellationException) {
                Log.i(TAG, "Batch cancelled")
            } finally {
                _batchProgress.value = BatchProgress(completedExperiments = completedFiles.toList())
                cleanup()
            }
        }
    }

    // --- Cleanup ---

    private fun cleanup() {
        systemMetricsJob?.cancel()
        if (logcatCapture.isCapturing()) logcatCapture.stopCapture()
        pipeline?.release()
        pipeline = null
        if (_progress.value.state != BenchmarkState.ERROR) {
            _progress.value = BenchmarkProgress(state = BenchmarkState.IDLE)
        }
        config = null
    }

    fun release() {
        stopBatch()
        stop()
        if (logcatCapture.isCapturing()) logcatCapture.stopCapture()
        pipeline?.release()
        pipeline = null
        sourceImage?.recycle()
        sourceImage = null
        sourceMask?.recycle()
        sourceMask = null
    }

    private fun nsToMs(ns: Long): Float = (ns / 1_000_000.0).toFloat()
}
