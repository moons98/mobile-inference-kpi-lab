package com.example.kpilab

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.WindowManager
import android.widget.ImageView
import android.widget.ArrayAdapter
import android.widget.AdapterView
import android.widget.LinearLayout
import android.widget.TextView
import android.text.Editable
import android.text.TextWatcher
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.kpilab.batch.BatchProgress
import com.example.kpilab.batch.ExperimentSet
import com.example.kpilab.batch.ExperimentSetLoader
import com.example.kpilab.databinding.ActivityMainBinding
import com.google.android.material.chip.Chip
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var kpiCollector: KpiCollector
    private lateinit var benchmarkRunner: BenchmarkRunner
    private lateinit var experimentSetLoader: ExperimentSetLoader

    private val idlePollHandler = Handler(Looper.getMainLooper())
    private val idlePollRunnable = object : Runnable {
        override fun run() {
            if (!benchmarkRunner.isRunning && !benchmarkRunner.isBatchRunning) {
                lifecycleScope.launch(Dispatchers.IO) {
                    val t = kpiCollector.readThermal()
                    val p = kpiCollector.readPower()
                    launch(Dispatchers.Main) {
                        if (!benchmarkRunner.isRunning && !benchmarkRunner.isBatchRunning) {
                            binding.tvThermal.text = if (t >= 0) "%.1f °C".format(t) else "-- °C"
                            binding.tvPower.text = if (p >= 0) "%.0f mW".format(p) else "-- mW"
                        }
                    }
                }
            }
            idlePollHandler.postDelayed(this, 3000)
        }
    }

    private var isBatchMode = false
    private var experimentSets: List<ExperimentSet> = emptyList()
    private var selectedStylePreset: StylePreset = StylePreset.WATERCOLOR

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Keep screen on during benchmarks (even when charger disconnected)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        checkStoragePermission()
        initializeComponents()
        setupUI()
        setupBatchMode()
        observeProgress()
        observeBatchProgress()
        observeGeneratedImage()
    }

    private fun checkStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                    data = Uri.parse("package:$packageName")
                }
                startActivity(intent)
                Toast.makeText(this, "모델 파일 접근을 위해 파일 권한을 허용해주세요", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun initializeComponents() {
        val qnnInitialized = OrtRunner.initializeQnnLibraries(this)
        if (!qnnInitialized) {
            Toast.makeText(this, "QNN initialization failed", Toast.LENGTH_LONG).show()
        }

        kpiCollector = KpiCollector(this)
        benchmarkRunner = BenchmarkRunner(this, kpiCollector)
        experimentSetLoader = ExperimentSetLoader(this)
        experimentSets = experimentSetLoader.getExperimentSets()
    }

    private fun setupUI() {
        binding.btnStartStop.setOnClickListener {
            if (benchmarkRunner.isRunning || benchmarkRunner.isBatchRunning) {
                confirmStop()
            } else {
                startBenchmark()
            }
        }

        binding.btnExport.setOnClickListener { exportData() }
        binding.btnExport.isEnabled = false

        binding.btnManageModels.setOnClickListener {
            showDeviceModelManager()
        }

        // Precision filter: NPU disables FP32
        binding.radioGroupEp.setOnCheckedChangeListener { _, checkedId ->
            val isNpu = checkedId == R.id.radioEpNpu
            updatePrecisionOptions(allowFp32 = !isNpu)
        }

        // Model variant change → update step defaults + rescan models
        binding.radioGroupVariant.setOnCheckedChangeListener { _, checkedId ->
            when (checkedId) {
                R.id.radioVariantLcm -> binding.radioSteps4.isChecked = true
                R.id.radioVariantSd15 -> binding.radioSteps20.isChecked = true
            }
            refreshModelAvailability()
        }

        // Per-component precision spinners
        setupPrecisionSpinners()

        // Style presets
        setupStylePresets()

        // Image tap → full-screen viewer
        binding.imgGenerated.setOnClickListener { showImageFullscreen(binding.imgGenerated, "Generated") }
    }

    private fun setupStylePresets() {
        for (preset in StylePreset.values()) {
            val chip = Chip(this).apply {
                text = preset.displayName
                isCheckable = true
                isChecked = preset == StylePreset.WATERCOLOR
                tag = preset
                textSize = 12f
            }
            binding.chipGroupStyle.addView(chip)
        }

        binding.chipGroupStyle.setOnCheckedStateChangeListener { group, checkedIds ->
            val checkedChip = checkedIds.firstOrNull()?.let { group.findViewById<Chip>(it) }
            selectedStylePreset = (checkedChip?.tag as? StylePreset) ?: StylePreset.NONE
            updateCombinedPromptPreview()
        }

        binding.editPrompt.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) { updateCombinedPromptPreview() }
        })
    }

    private fun updateCombinedPromptPreview() {
        if (selectedStylePreset == StylePreset.NONE) {
            binding.tvCombinedPrompt.visibility = View.GONE
        } else {
            val combined = buildCombinedPrompt()
            binding.tvCombinedPrompt.text = combined
            binding.tvCombinedPrompt.visibility = View.VISIBLE
        }
    }

    private fun buildCombinedPrompt(): String {
        val base = binding.editPrompt.text?.toString()?.trim()
            ?: "a photo of a cat sitting on a windowsill"
        return selectedStylePreset.applyTo(base)
    }

    private val precisionSpinners by lazy {
        listOf(
            binding.spinnerPrecTextEnc,
            binding.spinnerPrecUnet,
            binding.spinnerPrecVaeDec
        )
    }

    /** Per-component available precisions (updated by model scanner). */
    private var textEncOptions: List<SdPrecision> = listOf(SdPrecision.FP16)
    private var unetOptions: List<SdPrecision> = listOf(SdPrecision.FP16)
    private var vaeDecOptions: List<SdPrecision> = listOf(SdPrecision.FP16)

    private fun setupPrecisionSpinners() {
        // allowFp32는 메인 스레드에서 미리 읽어야 함 (IO 스레드에서 UI 접근 금지)
        val allowFp32 = binding.radioGroupEp.checkedRadioButtonId != R.id.radioEpNpu
        lifecycleScope.launch(Dispatchers.IO) {
            val scanResult = ModelScanner.scan(BenchmarkConfig().modelDir, currentVariant())
            launch(Dispatchers.Main) {
                applyModelScanResult(scanResult, allowFp32)
            }
        }

        binding.btnPrecAllFp16.setOnClickListener { setAllPrecision(SdPrecision.FP16) }
        binding.btnPrecAllW8a8.setOnClickListener { setAllPrecision(SdPrecision.W8A8) }
        binding.btnPrecAllFp32.setOnClickListener { setAllPrecision(SdPrecision.FP32) }
    }

    /** Re-scan and update spinners when variant or EP changes. */
    private fun refreshModelAvailability() {
        val allowFp32 = binding.radioGroupEp.checkedRadioButtonId != R.id.radioEpNpu
        lifecycleScope.launch(Dispatchers.IO) {
            val result = ModelScanner.scan(BenchmarkConfig().modelDir, currentVariant())
            launch(Dispatchers.Main) { applyModelScanResult(result, allowFp32) }
        }
    }

    private fun applyModelScanResult(result: ModelScanner.Result, allowFp32: Boolean) {
        fun filter(options: List<SdPrecision>): List<SdPrecision> {
            val filtered = if (allowFp32) options else options.filter { it != SdPrecision.FP32 }
            return filtered.ifEmpty { listOf(SdPrecision.FP16) }
        }

        textEncOptions = filter(result.precisionsFor(SdComponent.TEXT_ENCODER))
        unetOptions    = filter(result.precisionsFor(SdComponent.UNET))
        vaeDecOptions  = filter(result.precisionsFor(SdComponent.VAE_DECODER))

        bindSpinner(binding.spinnerPrecTextEnc, textEncOptions)
        bindSpinner(binding.spinnerPrecUnet, unetOptions)
        bindSpinner(binding.spinnerPrecVaeDec, vaeDecOptions)

        binding.btnPrecAllFp32.visibility = if (allowFp32) View.VISIBLE else View.GONE

        if (result.isEmpty) {
            Toast.makeText(this, "모델 디렉터리에서 모델을 찾을 수 없습니다", Toast.LENGTH_SHORT).show()
        }
    }

    private fun bindSpinner(spinner: android.widget.Spinner, options: List<SdPrecision>) {
        val prev = options.getOrNull(spinner.selectedItemPosition)
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, options.map { it.displayName })
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinner.adapter = adapter
        val newIndex = options.indexOf(prev).takeIf { it >= 0 }
            ?: options.indexOf(SdPrecision.FP16).takeIf { it >= 0 }
            ?: 0
        spinner.setSelection(newIndex)
    }

    private fun setAllPrecision(precision: SdPrecision) {
        listOf(
            binding.spinnerPrecTextEnc to textEncOptions,
            binding.spinnerPrecUnet to unetOptions,
            binding.spinnerPrecVaeDec to vaeDecOptions
        ).forEach { (spinner, options) ->
            val idx = options.indexOf(precision)
            if (idx >= 0) spinner.setSelection(idx)
        }
    }

    private fun updatePrecisionOptions(allowFp32: Boolean) {
        lifecycleScope.launch(Dispatchers.IO) {
            val result = ModelScanner.scan(BenchmarkConfig().modelDir, currentVariant())
            launch(Dispatchers.Main) { applyModelScanResult(result, allowFp32) }
        }
    }

    private fun buildPrecisionMap(): Map<SdComponent, SdPrecision> = mapOf(
        SdComponent.TEXT_ENCODER to textEncOptions[binding.spinnerPrecTextEnc.selectedItemPosition],
        SdComponent.UNET         to unetOptions[binding.spinnerPrecUnet.selectedItemPosition],
        SdComponent.VAE_DECODER  to vaeDecOptions[binding.spinnerPrecVaeDec.selectedItemPosition]
    )

    private fun currentVariant(): ModelVariant =
        if (binding.radioGroupVariant.checkedRadioButtonId == R.id.radioVariantLcm)
            ModelVariant.LCM_LORA else ModelVariant.SD_V15

    private fun setupBatchMode() {
        val adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_item,
            experimentSets.map { it.name }
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerExperimentSet.adapter = adapter

        binding.checkBatchMode.setOnCheckedChangeListener { _, isChecked ->
            isBatchMode = isChecked
            updateBatchModeUI()
        }

        binding.spinnerExperimentSet.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                updateExperimentDetail(position)
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {
                binding.layoutExperimentDetail.visibility = View.GONE
            }
        }

        updateBatchModeUI()
    }

    private fun updateBatchModeUI() {
        binding.layoutBatchOptions.visibility = if (isBatchMode) View.VISIBLE else View.GONE
        if (isBatchMode) {
            updateExperimentDetail(binding.spinnerExperimentSet.selectedItemPosition)
        } else {
            binding.layoutExperimentDetail.visibility = View.GONE
        }

        binding.cardSdConfig.visibility = if (isBatchMode) View.GONE else View.VISIBLE
    }

    private fun updateExperimentDetail(position: Int) {
        if (position < 0 || position >= experimentSets.size) {
            binding.layoutExperimentDetail.visibility = View.GONE
            return
        }
        val set = experimentSets[position]
        val defaults = experimentSetLoader.getDefaults()

        binding.tvExperimentDefaults.text = "Defaults: ${defaults.phase} | variant=${defaults.modelVariant} | steps=${defaults.steps}"
        binding.tvExperimentList.text = set.experiments.mapIndexed { i, exp ->
            "${i + 1}. ${exp.getDisplayName()}"
        }.joinToString("\n")
        binding.layoutExperimentDetail.visibility = View.VISIBLE
    }

    private fun observeProgress() {
        lifecycleScope.launch {
            benchmarkRunner.progress.collectLatest { progress ->
                updateUI(progress)
            }
        }
    }

    private fun observeBatchProgress() {
        lifecycleScope.launch {
            benchmarkRunner.batchProgress.collectLatest { progress ->
                updateBatchUI(progress)
            }
        }
    }

    private fun observeGeneratedImage() {
        lifecycleScope.launch {
            benchmarkRunner.lastGeneratedImage.collectLatest { bitmap ->
                if (bitmap != null) {
                    binding.cardImagePreview.visibility = View.VISIBLE
                    binding.imgGenerated.setImageBitmap(bitmap)
                }
            }
        }
    }

    private fun updateUI(progress: BenchmarkProgress) {
        if (progress.state == BenchmarkState.ERROR && progress.errorMessage != null) {
            Toast.makeText(this, progress.errorMessage, Toast.LENGTH_LONG).show()
        }

        binding.tvStatus.text = when (progress.state) {
            BenchmarkState.IDLE -> "Idle"
            BenchmarkState.INITIALIZING -> "Initializing..."
            BenchmarkState.WARMING_UP -> "Warming up..."
            BenchmarkState.RUNNING -> "Running"
            BenchmarkState.STOPPING -> "Stopping..."
            BenchmarkState.ERROR -> "Error"
        }

        val isRunning = progress.state in listOf(
            BenchmarkState.RUNNING, BenchmarkState.WARMING_UP, BenchmarkState.INITIALIZING,
            BenchmarkState.STOPPING)

        binding.progressBar.visibility = if (isRunning) View.VISIBLE else View.GONE
        binding.progressBar.progress = progress.progressPercent

        binding.tvTrial.text = if (progress.totalTrials > 0) {
            "${progress.currentTrial} / ${progress.totalTrials}"
        } else "-- / --"

        binding.tvUnetStep.text = if (progress.totalSteps > 0)
            "${progress.currentStep} / ${progress.totalSteps}" else "-- / --"

        binding.tvStage.text = progress.currentStage.ifEmpty { "--" }

        binding.tvE2e.text = if (progress.lastE2eMs >= 0)
            String.format("%.0f ms", progress.lastE2eMs) else "-- ms"

        binding.tvThermal.text = if (progress.lastThermalC >= 0)
            String.format("%.1f °C", progress.lastThermalC) else "-- °C"

        binding.tvPower.text = if (progress.lastPowerMw >= 0)
            String.format("%.0f mW", progress.lastPowerMw) else "-- mW"

        binding.tvMemory.text = if (progress.lastMemoryMb > 0)
            "${progress.lastMemoryMb} MB" else "-- MB"

        binding.btnStartStop.text = if (isRunning)
            getString(R.string.stop_benchmark) else getString(R.string.start_benchmark)

        setControlsEnabled(!isRunning)
        // STOPPING: 추론 완료, cleanup 진행 중 — 데이터는 이미 확정됐으므로 export 허용
        val canExport = (progress.state == BenchmarkState.STOPPING || !isRunning) &&
                benchmarkRunner.getRecordCount() > 0
        binding.btnExport.isEnabled = canExport
    }

    private fun updateBatchUI(progress: BatchProgress) {
        binding.cardBatchProgress.visibility = if (progress.isRunning || progress.completedExperiments.isNotEmpty())
            View.VISIBLE else View.GONE

        binding.tvBatchExperiment.text = progress.formatProgress()
        binding.tvCurrentExperiment.text = progress.currentExperimentName.ifEmpty { "--" }

        if (progress.isCoolingDown) {
            binding.layoutCooldown.visibility = View.VISIBLE
            binding.tvCooldown.text = "${progress.cooldownRemainingSeconds} sec"
        } else {
            binding.layoutCooldown.visibility = View.GONE
        }

        if (progress.completedExperiments.isNotEmpty()) {
            binding.tvCompletedExperiments.text = "Completed:\n" + progress.completedExperiments
                .mapIndexed { i, path -> "${i + 1}. ${File(path).name}" }.joinToString("\n")
        } else {
            binding.tvCompletedExperiments.text = ""
        }

        if (progress.isRunning) {
            binding.btnStartStop.text = getString(R.string.stop_benchmark)
            setControlsEnabled(false)
        } else if (!benchmarkRunner.isRunning) {
            binding.btnStartStop.text = getString(R.string.start_benchmark)
            setControlsEnabled(true)
        }
    }

    private fun setControlsEnabled(enabled: Boolean) {
        binding.checkBatchMode.isEnabled = enabled
        binding.spinnerExperimentSet.isEnabled = enabled
        binding.radioPhaseSmoke.isEnabled = enabled
        binding.radioPhase1.isEnabled = enabled
        binding.radioPhase2.isEnabled = enabled
        binding.radioHtpBurst.isEnabled = enabled
        binding.radioHtpSustained.isEnabled = enabled
        binding.radioVariantSd15.isEnabled = enabled
        binding.radioVariantLcm.isEnabled = enabled
        for (s in precisionSpinners) { s.isEnabled = enabled }
        binding.btnPrecAllFp32.isEnabled = enabled
        binding.btnPrecAllFp16.isEnabled = enabled
        binding.btnPrecAllW8a8.isEnabled = enabled
        binding.radioEpNpu.isEnabled = enabled
        binding.radioEpGpu.isEnabled = enabled
        binding.radioEpCpu.isEnabled = enabled
        binding.radioSteps2.isEnabled = enabled
        binding.radioSteps4.isEnabled = enabled
        binding.radioSteps8.isEnabled = enabled
        binding.radioSteps20.isEnabled = enabled
        binding.radioSteps30.isEnabled = enabled
        binding.radioSteps50.isEnabled = enabled
        binding.btnManageModels.isEnabled = enabled
        binding.editPrompt.isEnabled = enabled
        for (i in 0 until binding.chipGroupStyle.childCount) {
            binding.chipGroupStyle.getChildAt(i).isEnabled = enabled
        }
    }

    private fun buildConfig(): BenchmarkConfig {
        val isSmoke = binding.radioGroupPhase.checkedRadioButtonId == R.id.radioPhaseSmoke
        val phase = when (binding.radioGroupPhase.checkedRadioButtonId) {
            R.id.radioPhase2 -> BenchmarkPhase.SUSTAINED_GENERATE
            else -> BenchmarkPhase.SINGLE_GENERATE
        }

        val variant = when (binding.radioGroupVariant.checkedRadioButtonId) {
            R.id.radioVariantLcm -> ModelVariant.LCM_LORA
            else -> ModelVariant.SD_V15
        }

        val guidanceScale = when (variant) {
            ModelVariant.LCM_LORA -> 1.0f
            ModelVariant.SD_V15 -> 7.5f
        }

        val htpPerf = when (binding.radioGroupHtpPerf.checkedRadioButtonId) {
            R.id.radioHtpSustained -> "sustained_high"
            else -> "burst"
        }

        return BenchmarkConfig(
            modelVariant = variant,
            sdBackend = when (binding.radioGroupEp.checkedRadioButtonId) {
                R.id.radioEpGpu -> ExecutionProvider.QNN_GPU
                R.id.radioEpCpu -> ExecutionProvider.CPU
                else -> ExecutionProvider.QNN_NPU
            },
            sdPrecisionMap = buildPrecisionMap(),
            phase = phase,
            steps = getSelectedSteps(),
            guidanceScale = guidanceScale,
            prompt = buildCombinedPrompt(),
            trials = when {
                isSmoke -> 1
                phase == BenchmarkPhase.SUSTAINED_GENERATE -> 10
                else -> 5
            },
            warmupTrials = if (isSmoke) 0 else 2,
            htpPerformanceMode = htpPerf
        )
    }

    private fun getSelectedSteps(): Int = when (binding.radioGroupSteps.checkedRadioButtonId) {
        R.id.radioSteps2 -> 2
        R.id.radioSteps4 -> 4
        R.id.radioSteps8 -> 8
        R.id.radioSteps30 -> 30
        R.id.radioSteps50 -> 50
        else -> 20
    }

    private fun startBenchmark() {
        if (isBatchMode) {
            startBatchBenchmark()
        } else {
            val config = buildConfig()
            Log.i(TAG, "Starting: $config")
            benchmarkRunner.start(config, lifecycleScope)
            Toast.makeText(this, "Benchmark started", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startBatchBenchmark() {
        if (experimentSets.isEmpty()) {
            Toast.makeText(this, "No experiment sets", Toast.LENGTH_SHORT).show()
            return
        }
        val idx = binding.spinnerExperimentSet.selectedItemPosition
        if (idx < 0 || idx >= experimentSets.size) return

        val set = experimentSets[idx]
        val defaults = experimentSetLoader.getDefaults()

        benchmarkRunner.startBatch(
            experimentSet = set,
            defaults = defaults,
            scope = lifecycleScope,
            onExperimentComplete = { csvPath ->
                runOnUiThread {
                    Toast.makeText(this, "Saved: ${File(csvPath).name}", Toast.LENGTH_SHORT).show()
                }
            }
        )
    }

    private fun confirmStop() {
        val msg = if (benchmarkRunner.isBatchRunning) "Stop batch?" else "Stop benchmark?"
        AlertDialog.Builder(this)
            .setTitle("Stop")
            .setMessage(msg)
            .setPositiveButton("Stop") { _, _ ->
                if (benchmarkRunner.isBatchRunning) benchmarkRunner.stopBatch()
                else benchmarkRunner.stop()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun exportData() {
        if (benchmarkRunner.getRecordCount() == 0) {
            Toast.makeText(this, "No data", Toast.LENGTH_SHORT).show()
            return
        }
        binding.btnExport.isEnabled = false
        lifecycleScope.launch(Dispatchers.IO) {
            val path = benchmarkRunner.exportAndSaveCsv()
            launch(Dispatchers.Main) {
                binding.btnExport.isEnabled = benchmarkRunner.getRecordCount() > 0
                if (path != null) {
                    Toast.makeText(this@MainActivity, "Exported to:\n${File(path).name}", Toast.LENGTH_LONG).show()
                } else {
                    Toast.makeText(this@MainActivity, "Export failed", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun showImageFullscreen(imageView: ImageView, title: String) {
        val drawable = imageView.drawable ?: return

        val dialogView = ImageView(this).apply {
            setImageDrawable(drawable)
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(0xFF000000.toInt())
        }

        val dialog = AlertDialog.Builder(this)
            .setTitle(title)
            .setView(dialogView)
            .setNegativeButton("닫기", null)
            .create()

        dialog.window?.apply {
            setLayout(WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT)
        }
        dialog.show()
    }

    private fun formatSize(bytes: Long): String = when {
        bytes >= 1024L * 1024 * 1024 -> "%.1f GB".format(bytes / (1024.0 * 1024 * 1024))
        bytes >= 1024L * 1024 -> "%.1f MB".format(bytes / (1024.0 * 1024))
        bytes >= 1024L -> "%.1f KB".format(bytes / 1024.0)
        else -> "$bytes B"
    }

    private data class DeviceModel(
        val name: String,
        val onnxFile: File,
        val binFile: File?,
        val onnxSize: Long,
        val binSize: Long
    ) {
        val totalSize: Long get() = onnxSize + binSize
        val isPrecompiled: Boolean get() = binFile != null
    }

    private fun showDeviceModelManager() {
        lifecycleScope.launch(Dispatchers.IO) {
            val modelDir = File(BenchmarkConfig().modelDir)

            val models = if (modelDir.exists()) {
                val onnxFiles = (modelDir.listFiles() ?: emptyArray())
                    .filter { it.isFile && it.name.endsWith(".onnx") }
                    .sortedBy { it.name }

                onnxFiles.map { onnx ->
                    val stem = onnx.nameWithoutExtension
                    val binFile = File(modelDir, "$stem.bin").takeIf { it.exists() }
                    DeviceModel(stem, onnx, binFile, onnx.length(), binFile?.length() ?: 0L)
                }
            } else emptyList()

            runOnUiThread {
                showDeviceModelList(models, modelDir)
            }
        }
    }

    /**
     * Classify model into a pipeline component group by name heuristics.
     */
    private fun classifyModelGroup(name: String): String {
        val lower = name.lowercase()
        return when {
            "unet" in lower -> "UNet"
            "text_encoder" in lower || "textencoder" in lower -> "Text Encoder"
            "vae" in lower -> "VAE"
            "safety" in lower -> "Safety Checker"
            "tokenizer" in lower -> "Tokenizer"
            "yolo" in lower || "seg" in lower -> "YOLO / Segmentation"
            else -> "Other"
        }
    }

    /**
     * Extract a short display name: strip common prefixes (sd15_, lcm_) and
     * the group keyword to keep the row concise.
     */
    private fun shortDisplayName(name: String): String {
        return name
            .removePrefix("sd15_").removePrefix("lcm_").removePrefix("sdxl_")
            .replace("_", " ")
    }

    private fun showDeviceModelList(models: List<DeviceModel>, modelDir: File) {
        if (models.isEmpty()) {
            AlertDialog.Builder(this)
                .setTitle("Models on Device")
                .setMessage("No models found in\n${modelDir.absolutePath}")
                .setNegativeButton("Close", null)
                .show()
            return
        }

        val totalSize = models.sumOf { it.totalSize }
        val inflater = LayoutInflater.from(this)
        val view = inflater.inflate(R.layout.dialog_model_manager, null)
        val llModelList = view.findViewById<LinearLayout>(R.id.llModelList)
        val tvSummary = view.findViewById<TextView>(R.id.tvModelSummary)

        tvSummary.text = "${models.size} models  ·  ${formatSize(totalSize)}"

        // Group models by pipeline component
        val grouped = models.groupBy { classifyModelGroup(it.name) }
        val groupOrder = listOf(
            "UNet", "Text Encoder", "VAE", "Safety Checker",
            "Tokenizer", "YOLO / Segmentation", "Other"
        )

        for (group in groupOrder) {
            val items = grouped[group] ?: continue

            // Section header
            val header = inflater.inflate(R.layout.item_model_group_header, llModelList, false)
            val groupSize = items.sumOf { it.totalSize }
            (header as TextView).text = "$group  (${items.size}  ·  ${formatSize(groupSize)})"
            llModelList.addView(header)

            // Model rows
            for (model in items) {
                val row = inflater.inflate(R.layout.item_model_row, llModelList, false)
                row.findViewById<TextView>(R.id.tvModelName).text = shortDisplayName(model.name)
                row.findViewById<TextView>(R.id.tvModelMeta).text =
                    if (model.isPrecompiled) "Precompiled · .onnx + .bin" else "ONNX"
                row.findViewById<TextView>(R.id.tvModelSize).text = formatSize(model.totalSize)
                row.setOnClickListener { showModelDetail(model) }
                llModelList.addView(row)
            }
        }

        val dialog = AlertDialog.Builder(this)
            .setTitle("Models on Device")
            .setView(view)
            .setNegativeButton("Close", null)
            .create()

        // Delete All button
        view.findViewById<View>(R.id.btnDeleteAll).setOnClickListener {
            dialog.dismiss()
            confirmDeleteAll(models)
        }

        dialog.show()
    }

    private fun showModelDetail(model: DeviceModel) {
        val info = buildString {
            if (model.isPrecompiled) {
                appendLine("Type:  Precompiled (EpContext)")
                appendLine("Stub:  ${model.onnxFile.name}  (${formatSize(model.onnxSize)})")
                appendLine("Binary: ${model.binFile!!.name}  (${formatSize(model.binSize)})")
            } else {
                appendLine("Type:  ONNX model")
                appendLine("File:  ${model.onnxFile.name}  (${formatSize(model.onnxSize)})")
            }
            appendLine()
            appendLine("Total: ${formatSize(model.totalSize)}")
            appendLine("Path:  ${model.onnxFile.parent}")
        }

        AlertDialog.Builder(this)
            .setTitle(model.name)
            .setMessage(info)
            .setNegativeButton("Back") { _, _ -> showDeviceModelManager() }
            .setPositiveButton("Delete") { _, _ ->
                confirmDeleteModel(model)
            }
            .show()
    }

    private fun confirmDeleteModel(model: DeviceModel) {
        val files = buildString {
            appendLine(model.onnxFile.name)
            if (model.binFile != null) appendLine(model.binFile.name)
        }
        AlertDialog.Builder(this)
            .setTitle("Delete ${model.name}?")
            .setMessage("Files to delete:\n$files\nTotal: ${formatSize(model.totalSize)}")
            .setPositiveButton("Delete") { _, _ ->
                model.onnxFile.delete()
                model.binFile?.delete()
                Toast.makeText(this, "${model.name} deleted", Toast.LENGTH_SHORT).show()
                showDeviceModelManager()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun confirmDeleteAll(models: List<DeviceModel>) {
        val totalSize = models.sumOf { it.totalSize }
        val fileCount = models.count { it.isPrecompiled } + models.size

        AlertDialog.Builder(this)
            .setTitle("Delete all ${models.size} models?")
            .setMessage("$fileCount files  ·  ${formatSize(totalSize)}\n\nThis cannot be undone.")
            .setPositiveButton("Delete All") { _, _ ->
                var deleted = 0
                for (m in models) {
                    if (m.onnxFile.delete()) deleted++
                    m.binFile?.delete()
                }
                Toast.makeText(this, "$deleted models deleted", Toast.LENGTH_SHORT).show()
                showDeviceModelManager()
            }
            .setNegativeButton("Cancel") { _, _ -> showDeviceModelManager() }
            .show()
    }

    override fun onResume() {
        super.onResume()
        idlePollHandler.post(idlePollRunnable)
    }

    override fun onPause() {
        super.onPause()
        idlePollHandler.removeCallbacks(idlePollRunnable)
    }

    override fun onDestroy() {
        super.onDestroy()
        idlePollHandler.removeCallbacks(idlePollRunnable)
        benchmarkRunner.release()
    }
}
