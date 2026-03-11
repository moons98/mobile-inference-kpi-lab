package com.example.kpilab

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.AdapterView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.kpilab.batch.BatchProgress
import com.example.kpilab.batch.ExperimentSet
import com.example.kpilab.batch.ExperimentSetLoader
import com.example.kpilab.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
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

    private var isBatchMode = false
    private var experimentSets: List<ExperimentSet> = emptyList()
    private var selectedImageBitmap: Bitmap? = null
    private var selectedMaskBitmap: Bitmap? = null

    /** Built-in test images available in assets */
    private data class BuiltinImage(
        val label: String,
        val imagePath: String,
        val maskPath: String
    )
    private val builtinImages = listOf(
        BuiltinImage("Wine Glass (Small)", "test_images/scene_small.jpg", "test_masks/mask_small.png"),
        BuiltinImage("Cup (Medium)", "test_images/scene_medium.jpg", "test_masks/mask_medium.png"),
        BuiltinImage("Person (Large)", "test_images/scene_large.jpg", "test_masks/mask_large.png"),
        BuiltinImage("Cat (sample_image)", "sample_image.jpg", "")
    )

    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { loadGalleryImage(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

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

        binding.btnSelectImage.setOnClickListener {
            galleryLauncher.launch("image/*")
        }

        binding.btnBuiltinImage.setOnClickListener {
            showBuiltinImagePicker()
        }

        binding.btnPrepareCache.setOnClickListener {
            prepareQnnCache()
        }
        binding.btnManageCache.setOnClickListener {
            showCacheManageDialog()
        }

        // Update cache status on launch
        updateCacheStatus()

        // Strength label update
        binding.radioGroupStrength.setOnCheckedChangeListener { _, _ -> updateStrengthLabel() }
        binding.radioGroupSteps.setOnCheckedChangeListener { _, _ -> updateStrengthLabel() }

        // QNN options visibility + precision filter
        binding.radioGroupEp.setOnCheckedChangeListener { _, checkedId ->
            val isNpu = checkedId == R.id.radioEpNpu
            val isCpu = checkedId == R.id.radioEpCpu
            binding.layoutQnnOptions.visibility = if (isCpu) View.GONE else View.VISIBLE
            updatePrecisionOptions(allowFp32 = !isNpu)
        }

        // Phase toggle: show/hide SD vs YOLO options
        binding.radioGroupPhase.setOnCheckedChangeListener { _, checkedId ->
            val isYoloOnly = checkedId == R.id.radioPhaseYolo
            binding.sectionSdOptions.visibility = if (isYoloOnly) View.GONE else View.VISIBLE
            binding.sectionYoloOptions.visibility = if (isYoloOnly) View.VISIBLE else View.GONE
        }

        // Per-component precision spinners
        setupPrecisionSpinners()
    }

    private val precisionSpinners by lazy {
        listOf(
            binding.spinnerPrecVaeEnc,
            binding.spinnerPrecTextEnc,
            binding.spinnerPrecUnet,
            binding.spinnerPrecVaeDec
        )
    }

    /** Precision options: NPU에서는 FP32가 실제로 FP16으로 실행되므로 제외 */
    private val npuPrecOptions = SdPrecision.values().filter { it != SdPrecision.FP32 }
    private val allPrecOptions = SdPrecision.values().toList()
    private var currentPrecOptions = allPrecOptions

    private fun setupPrecisionSpinners() {
        updatePrecisionOptions(allowFp32 = binding.radioGroupEp.checkedRadioButtonId != R.id.radioEpNpu)

        // Bulk preset buttons
        binding.btnPrecAllFp32.setOnClickListener { setAllPrecision(SdPrecision.FP32) }
        binding.btnPrecAllFp16.setOnClickListener { setAllPrecision(SdPrecision.FP16) }
        binding.btnPrecAllW8a8.setOnClickListener { setAllPrecision(SdPrecision.W8A8) }
    }

    /**
     * NPU에서는 FP32가 실제로 FP16으로 실행되므로 FP32 선택지를 숨긴다.
     * CPU/GPU에서는 전체 옵션을 표시한다.
     */
    private fun updatePrecisionOptions(allowFp32: Boolean) {
        currentPrecOptions = if (allowFp32) allPrecOptions else npuPrecOptions
        val displayNames = currentPrecOptions.map { it.displayName }
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, displayNames)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)

        for (spinner in precisionSpinners) {
            val prevPrecision = currentPrecOptions.getOrNull(spinner.selectedItemPosition)
            spinner.adapter = adapter
            // Restore previous selection if still available, otherwise default to FP16
            val newIndex = currentPrecOptions.indexOf(prevPrecision).takeIf { it >= 0 }
                ?: currentPrecOptions.indexOf(SdPrecision.FP16).takeIf { it >= 0 }
                ?: 0
            spinner.setSelection(newIndex)
        }

        // Hide "All FP32" button when NPU is selected
        binding.btnPrecAllFp32.visibility = if (allowFp32) View.VISIBLE else View.GONE
    }

    private fun setAllPrecision(precision: SdPrecision) {
        val index = currentPrecOptions.indexOf(precision)
        if (index < 0) return
        for (spinner in precisionSpinners) {
            spinner.setSelection(index)
        }
    }

    private fun buildPrecisionMap(): Map<SdComponent, SdPrecision> {
        val precValues = currentPrecOptions.toTypedArray()
        return mapOf(
            SdComponent.VAE_ENCODER to precValues[binding.spinnerPrecVaeEnc.selectedItemPosition],
            SdComponent.TEXT_ENCODER to precValues[binding.spinnerPrecTextEnc.selectedItemPosition],
            SdComponent.INPAINT_UNET to precValues[binding.spinnerPrecUnet.selectedItemPosition],
            SdComponent.VAE_DECODER to precValues[binding.spinnerPrecVaeDec.selectedItemPosition]
        )
    }

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

        binding.tvExperimentDefaults.text = "Defaults: ${defaults.phase} | steps=${defaults.steps} | strength=${defaults.strength}"
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
                    // Show original in imgBefore if not already set
                    if (selectedImageBitmap != null) {
                        binding.imgBefore.setImageBitmap(selectedImageBitmap)
                    }
                    binding.imgAfter.setImageBitmap(bitmap)
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
            BenchmarkState.RUNNING, BenchmarkState.WARMING_UP, BenchmarkState.INITIALIZING)

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
        binding.btnExport.isEnabled = !isRunning && benchmarkRunner.getRecordCount() > 0
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
        binding.btnSelectImage.isEnabled = enabled
        binding.btnBuiltinImage.isEnabled = enabled
        binding.checkBatchMode.isEnabled = enabled
        binding.spinnerExperimentSet.isEnabled = enabled
        binding.radioPhase1.isEnabled = enabled
        binding.radioPhase2.isEnabled = enabled
        binding.radioPhaseYolo.isEnabled = enabled
        for (s in precisionSpinners) { s.isEnabled = enabled }
        binding.btnPrecAllFp32.isEnabled = enabled
        binding.btnPrecAllFp16.isEnabled = enabled
        binding.btnPrecAllW8a8.isEnabled = enabled
        binding.radioEpNpu.isEnabled = enabled
        binding.radioEpGpu.isEnabled = enabled
        binding.radioEpCpu.isEnabled = enabled
        binding.radioSteps10.isEnabled = enabled
        binding.radioSteps20.isEnabled = enabled
        binding.radioSteps30.isEnabled = enabled
        binding.radioSteps50.isEnabled = enabled
        binding.radioStr05.isEnabled = enabled
        binding.radioStr07.isEnabled = enabled
        binding.radioStr08.isEnabled = enabled
        binding.radioStr10.isEnabled = enabled
        binding.radioRoiSmall.isEnabled = enabled
        binding.radioRoiMedium.isEnabled = enabled
        binding.radioRoiLarge.isEnabled = enabled
        binding.checkCache.isEnabled = enabled
        binding.btnPrepareCache.isEnabled = enabled
    }

    private fun buildConfig(): BenchmarkConfig {
        val phase = when (binding.radioGroupPhase.checkedRadioButtonId) {
            R.id.radioPhase2 -> BenchmarkPhase.SUSTAINED_ERASE
            R.id.radioPhaseYolo -> BenchmarkPhase.YOLO_SEG_ONLY
            else -> BenchmarkPhase.SINGLE_ERASE
        }

        val isYoloOnly = phase == BenchmarkPhase.YOLO_SEG_ONLY

        return BenchmarkConfig(
            sdBackend = when (binding.radioGroupEp.checkedRadioButtonId) {
                R.id.radioEpGpu -> ExecutionProvider.QNN_GPU
                R.id.radioEpCpu -> ExecutionProvider.CPU
                else -> ExecutionProvider.QNN_NPU
            },
            sdPrecisionMap = buildPrecisionMap(),
            yoloBackend = if (isYoloOnly) {
                when (binding.radioGroupYoloEp.checkedRadioButtonId) {
                    R.id.radioYoloEpGpu -> ExecutionProvider.QNN_GPU
                    R.id.radioYoloEpCpu -> ExecutionProvider.CPU
                    else -> ExecutionProvider.QNN_NPU
                }
            } else {
                when (binding.radioGroupEp.checkedRadioButtonId) {
                    R.id.radioEpGpu -> ExecutionProvider.QNN_GPU
                    R.id.radioEpCpu -> ExecutionProvider.CPU
                    else -> ExecutionProvider.QNN_NPU
                }
            },
            yoloPrecision = if (isYoloOnly) {
                when (binding.radioGroupYoloPrec.checkedRadioButtonId) {
                    R.id.radioYoloPrecInt8 -> YoloPrecision.INT8
                    else -> YoloPrecision.FP32
                }
            } else {
                if (buildPrecisionMap().values.any { it == SdPrecision.W8A8 })
                    YoloPrecision.INT8 else YoloPrecision.FP32
            },
            phase = phase,
            steps = getSelectedSteps(),
            strength = getSelectedStrength(),
            roiSize = when (binding.radioGroupRoiSize.checkedRadioButtonId) {
                R.id.radioRoiSmall -> RoiSize.SMALL
                R.id.radioRoiLarge -> RoiSize.LARGE
                else -> RoiSize.MEDIUM
            },
            trials = when (phase) {
                BenchmarkPhase.SUSTAINED_ERASE -> 10
                BenchmarkPhase.YOLO_SEG_ONLY -> 20
                BenchmarkPhase.SINGLE_ERASE -> 5
            },
            useContextCache = binding.checkCache.isChecked,
            htpPerformanceMode = when (phase) {
                BenchmarkPhase.SUSTAINED_ERASE -> "sustained_high"
                else -> "burst"
            }
        )
    }

    private fun updateStrengthLabel() {
        val strength = getSelectedStrength()
        val steps = getSelectedSteps()
        val actual = (steps * strength).toInt()
        binding.tvStrengthLabel.text = "Strength: $strength (actual steps: $actual)"
    }

    private fun getSelectedSteps(): Int = when (binding.radioGroupSteps.checkedRadioButtonId) {
        R.id.radioSteps10 -> 10
        R.id.radioSteps30 -> 30
        R.id.radioSteps50 -> 50
        else -> 20
    }

    private fun getSelectedStrength(): Float = when (binding.radioGroupStrength.checkedRadioButtonId) {
        R.id.radioStr05 -> 0.5f
        R.id.radioStr08 -> 0.8f
        R.id.radioStr10 -> 1.0f
        else -> 0.7f
    }

    private fun startBenchmark() {
        if (isBatchMode) {
            startBatchBenchmark()
        } else {
            val config = buildConfig()
            Log.i(TAG, "Starting: $config")
            if (config.phase == BenchmarkPhase.YOLO_SEG_ONLY) {
                val image = selectedImageBitmap
                if (image != null) {
                    benchmarkRunner.clearSourceInputs()
                    benchmarkRunner.setSourceImage(image)
                } else {
                    benchmarkRunner.clearSourceInputs()
                }
            } else {
                val image = selectedImageBitmap
                val mask = selectedMaskBitmap
                if (image != null && mask != null) {
                    benchmarkRunner.setSourceImage(image)
                    benchmarkRunner.setSourceMask(mask)
                } else {
                    if (image != null || mask != null) {
                        Toast.makeText(this, "Custom input requires both image and mask. Using fixed test data.", Toast.LENGTH_SHORT).show()
                    }
                    benchmarkRunner.clearSourceInputs()
                }
            }
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

        // Batch mode enforces fixed datasets from assets for reproducibility.
        benchmarkRunner.clearSourceInputs()
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
        val path = benchmarkRunner.exportAndSaveCsv()
        if (path != null) {
            Toast.makeText(this, "Exported to:\n${File(path).name}", Toast.LENGTH_LONG).show()
        } else {
            Toast.makeText(this, "Export failed", Toast.LENGTH_SHORT).show()
        }
    }

    private fun showBuiltinImagePicker() {
        val labels = builtinImages.map { it.label }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("Select Built-in Image")
            .setItems(labels) { _, which ->
                loadBuiltinImage(builtinImages[which])
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun loadBuiltinImage(builtin: BuiltinImage) {
        try {
            clearSelectedInputs()
            val bitmap = assets.open(builtin.imagePath).use { BitmapFactory.decodeStream(it) }
            if (bitmap != null) {
                selectedImageBitmap = bitmap

                // Load paired mask if available
                if (builtin.maskPath.isNotEmpty()) {
                    val mask = assets.open(builtin.maskPath).use { BitmapFactory.decodeStream(it) }
                    if (mask != null) {
                        selectedMaskBitmap = mask
                    }
                } else {
                    selectedMaskBitmap = null
                }

                binding.cardImagePreview.visibility = View.VISIBLE
                binding.imgBefore.setImageBitmap(bitmap)
                binding.imgAfter.setImageBitmap(null)

                val maskInfo = if (builtin.maskPath.isNotEmpty()) " + mask" else " (no mask)"
                Toast.makeText(this,
                    "${builtin.label} (${bitmap.width}x${bitmap.height})$maskInfo",
                    Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load built-in image: ${builtin.imagePath}", e)
            Toast.makeText(this, "Failed to load: ${builtin.label}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadGalleryImage(uri: Uri) {
        try {
            clearSelectedInputs()
            val bitmap = contentResolver.openInputStream(uri).use { inputStream ->
                BitmapFactory.decodeStream(inputStream)
            }
            if (bitmap != null) {
                selectedImageBitmap = bitmap
                selectedMaskBitmap = null
                binding.cardImagePreview.visibility = View.VISIBLE
                binding.imgBefore.setImageBitmap(bitmap)
                binding.imgAfter.setImageBitmap(null)
                Toast.makeText(this, "Image selected (${bitmap.width}×${bitmap.height})", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load gallery image", e)
            Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
        }
    }

    private fun clearSelectedInputs() {
        selectedImageBitmap?.recycle()
        selectedImageBitmap = null
        selectedMaskBitmap?.recycle()
        selectedMaskBitmap = null
    }

    private fun prepareQnnCache() {
        val ep = when (binding.radioGroupEp.checkedRadioButtonId) {
            R.id.radioEpGpu -> ExecutionProvider.QNN_GPU
            R.id.radioEpCpu -> {
                Toast.makeText(this, "CPU does not need QNN cache", Toast.LENGTH_SHORT).show()
                return
            }
            else -> ExecutionProvider.QNN_NPU
        }
        val precMap = buildPrecisionMap()

        val config = BenchmarkConfig(
            sdBackend = ep,
            sdPrecisionMap = precMap,
            useContextCache = true,
            useNpuFp16 = precMap.values.any { it != SdPrecision.FP32 }
        )

        binding.btnPrepareCache.isEnabled = false
        binding.btnStartStop.isEnabled = false
        binding.tvCacheStatus.text = "Preparing QNN cache..."

        lifecycleScope.launch(Dispatchers.IO) {
            val pipeline = InpaintPipeline(this@MainActivity, config)
            val results = try {
                pipeline.buildContextCache { name, index, total, status ->
                    runOnUiThread {
                        binding.tvCacheStatus.text = "[$index/$total] $name: $status"
                    }
                }
            } finally {
                pipeline.release()
            }

            runOnUiThread {
                val summary = results.entries.joinToString("\n") { "${it.key}: ${it.value}" }
                binding.tvCacheStatus.text = summary
                binding.btnPrepareCache.isEnabled = true
                binding.btnStartStop.isEnabled = true
                binding.checkCache.isChecked = true

                val allOk = results.values.all { it.startsWith("OK") }
                Toast.makeText(this@MainActivity,
                    if (allOk) "QNN cache ready!" else "Some models failed",
                    Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun updateCacheStatus() {
        lifecycleScope.launch(Dispatchers.IO) {
            // Check for both FP16 and W8A8 caches
            val fp16Config = BenchmarkConfig(sdPrecisionMap = BenchmarkConfig.uniformPrecision(SdPrecision.FP16), useNpuFp16 = true)
            val fp16Cache = InpaintPipeline(this@MainActivity, fp16Config).checkContextCache()

            runOnUiThread {
                val cached = fp16Cache.count { it.value }
                val total = fp16Cache.size
                if (cached > 0) {
                    val status = fp16Cache.entries.joinToString(" | ") { (name, ok) ->
                        if (ok) "$name:OK" else "$name:--"
                    }
                    binding.tvCacheStatus.text = "Cache (FP16): $cached/$total | $status"
                }
            }
        }
    }

    /** Represents one ONNX model and its associated cache/partition state. */
    private data class ModelEntry(
        val name: String,           // e.g., "yolov8n-seg", "vae_encoder_fp32"
        val modelFile: File,
        val dataFile: File?,        // .onnx.data companion (UNet external data)
        val cacheFile: File?,       // qnn_{name}_{prec}.bin
        val partitionFile: File?,   // ort_partition_{name}.json
        val partitionInfo: OrtLogInfo?,
        val modelSize: Long,
        val cacheSize: Long
    )

    private fun formatSize(bytes: Long): String = when {
        bytes >= 1024L * 1024 * 1024 -> "%.1f GB".format(bytes / (1024.0 * 1024 * 1024))
        bytes >= 1024L * 1024 -> "%.1f MB".format(bytes / (1024.0 * 1024))
        bytes >= 1024L -> "%.1f KB".format(bytes / 1024.0)
        else -> "$bytes B"
    }

    private fun showCacheManageDialog() {
        lifecycleScope.launch(Dispatchers.IO) {
            val appCacheDir = cacheDir
            val modelDir = File(BenchmarkConfig().modelDir)

            // Scan .onnx files (skip .onnx.data — they are companions)
            val onnxFiles = if (modelDir.exists()) {
                (modelDir.listFiles() ?: emptyArray())
                    .filter { it.isFile && it.name.endsWith(".onnx") && !it.name.endsWith(".onnx.data") }
                    .sortedBy { it.name }
            } else emptyList()

            // Index cache and partition files by model stem
            val cacheMap = (appCacheDir.listFiles() ?: emptyArray())
                .filter { it.name.startsWith("qnn_") && it.name.endsWith(".bin") }
                .associateBy { it.nameWithoutExtension }  // qnn_modelname_fp16

            val partMap = (appCacheDir.listFiles() ?: emptyArray())
                .filter { it.name.startsWith("ort_partition_") && it.name.endsWith(".json") }
                .associateBy { it.nameWithoutExtension.removePrefix("ort_partition_") }

            // Build model entries
            val entries = onnxFiles.map { onnx ->
                val stem = onnx.nameWithoutExtension  // e.g., "yolov8n-seg", "vae_encoder_fp32"
                val dataFile = File(onnx.absolutePath + ".data").takeIf { it.exists() }
                // Find cache: try fp16 first, then fp32
                val cache = cacheMap["qnn_${stem}_fp16"] ?: cacheMap["qnn_${stem}_fp32"]
                val part = partMap[stem]
                val partInfo = part?.let { OrtLogInfo.loadFromFile(it) }
                val totalModelSize = onnx.length() + (dataFile?.length() ?: 0L)
                ModelEntry(stem, onnx, dataFile, cache, part, partInfo, totalModelSize, cache?.length() ?: 0L)
            }

            // ORT profiling files (separate)
            val profileFiles = (filesDir.listFiles() ?: emptyArray())
                .filter { it.name.startsWith("ort_profile") && it.name.endsWith(".json") }

            runOnUiThread {
                showModelListDialog(entries, profileFiles)
            }
        }
    }

    private fun showModelListDialog(entries: List<ModelEntry>, profileFiles: List<File>) {
        val report = buildString {
            for (e in entries) {
                appendLine("${e.name}  (${formatSize(e.modelSize)})")
                val cacheTag = if (e.cacheFile != null) "OK ${formatSize(e.cacheSize)}" else "--"
                val partTag = if (e.partitionInfo != null)
                    "QNN:${e.partitionInfo.qnnNodes} CPU:${e.partitionInfo.cpuNodes}"
                else "--"
                appendLine("  Cache: $cacheTag  |  Partition: $partTag")
            }
            if (entries.isEmpty()) appendLine("(no models found)")

            if (profileFiles.isNotEmpty()) {
                appendLine()
                appendLine("ORT Profiles: ${profileFiles.size} file(s)")
            }
        }

        val builder = AlertDialog.Builder(this)
            .setTitle("Model & Cache Manager")
            .setMessage(report)
            .setNegativeButton("Close", null)

        val hasCache = entries.any { it.cacheFile != null }
        if (hasCache || profileFiles.isNotEmpty()) {
            builder.setPositiveButton("Clean Up...") { _, _ ->
                showCleanUpDialog(entries, profileFiles)
            }
        }

        if (entries.isNotEmpty()) {
            builder.setNeutralButton("Actions...") { _, _ ->
                showModelActionDialog(entries)
            }
        }

        builder.show()
    }

    private fun showModelActionDialog(entries: List<ModelEntry>) {
        val items = entries.map { e ->
            val status = buildString {
                append(e.name)
                append("  (${formatSize(e.modelSize)})")
                if (e.cacheFile != null) append("  [C]")
                if (e.partitionInfo != null) append("[P]")
            }
            status
        }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle("Select Model")
            .setItems(items) { _, which ->
                showSingleModelDialog(entries[which])
            }
            .setNegativeButton("Back", null)
            .show()
    }

    private fun showSingleModelDialog(entry: ModelEntry) {
        val info = buildString {
            appendLine("Model: ${entry.modelFile.name}")
            appendLine("Size: ${formatSize(entry.modelSize)}")
            if (entry.dataFile != null) {
                appendLine("External data: ${entry.dataFile.name} (${formatSize(entry.dataFile.length())})")
            }
            appendLine()
            if (entry.cacheFile != null) {
                appendLine("QNN Cache: ${entry.cacheFile.name}")
                appendLine("Cache size: ${formatSize(entry.cacheSize)}")
            } else {
                appendLine("QNN Cache: not built")
            }
            appendLine()
            if (entry.partitionInfo != null) {
                val p = entry.partitionInfo
                appendLine("Graph Partitioning:")
                appendLine("  Total nodes: ${p.totalNodes}")
                appendLine("  QNN nodes: ${p.qnnNodes}")
                appendLine("  CPU fallback: ${p.cpuNodes}")
                if (p.fallbackOps.isNotEmpty()) {
                    appendLine("  Fallback ops: ${p.fallbackOps.joinToString(", ")}")
                }
                val coverage = if (p.totalNodes > 0) "%.1f%%".format(p.qnnNodes * 100.0 / p.totalNodes) else "N/A"
                appendLine("  QNN coverage: $coverage")
            } else {
                appendLine("Partition info: not available")
            }
        }

        val builder = AlertDialog.Builder(this)
            .setTitle(entry.name)
            .setMessage(info)
            .setNegativeButton("Back", null)

        if (entry.cacheFile != null) {
            builder.setPositiveButton("Delete Cache") { _, _ ->
                entry.cacheFile.delete()
                entry.partitionFile?.delete()
                Toast.makeText(this, "Cache deleted. Will be rebuilt on next run.", Toast.LENGTH_SHORT).show()
                updateCacheStatus()
            }
        } else {
            builder.setPositiveButton("Build Cache") { _, _ ->
                buildCacheForModel(entry)
            }
        }

        builder.setNeutralButton("Delete Model") { _, _ ->
            AlertDialog.Builder(this)
                .setTitle("Delete ${entry.name}?")
                .setMessage("Model file + cache + partition will be deleted.")
                .setPositiveButton("Delete") { _, _ ->
                    entry.modelFile.delete()
                    entry.dataFile?.delete()
                    entry.cacheFile?.delete()
                    entry.partitionFile?.delete()
                    Toast.makeText(this, "${entry.name} deleted", Toast.LENGTH_SHORT).show()
                    updateCacheStatus()
                }
                .setNegativeButton("Cancel", null)
                .show()
        }

        builder.show()
    }

    private fun buildCacheForModel(entry: ModelEntry) {
        val ep = when (binding.radioGroupEp.checkedRadioButtonId) {
            R.id.radioEpGpu -> ExecutionProvider.QNN_GPU
            R.id.radioEpCpu -> {
                Toast.makeText(this, "CPU does not need QNN cache", Toast.LENGTH_SHORT).show()
                return
            }
            else -> ExecutionProvider.QNN_NPU
        }
        val useFp16 = buildPrecisionMap().values.any { it != SdPrecision.FP32 }

        binding.tvCacheStatus.text = "Building cache: ${entry.name}..."
        binding.btnManageCache.isEnabled = false
        binding.btnPrepareCache.isEnabled = false

        lifecycleScope.launch(Dispatchers.IO) {
            val logcatCapture = LogcatCapture()
            val captureScope = CoroutineScope(currentCoroutineContext())
            logcatCapture.startCapture(listOf("onnxruntime", "OrtRunner", "QNN"), captureScope)

            val runner = OrtRunner(this@MainActivity)
            val success = runner.initialize(
                modelPath = entry.modelFile.absolutePath,
                executionProvider = ep,
                useNpuFp16 = useFp16,
                useContextCache = true,
                htpPerformanceMode = "burst"
            )

            logcatCapture.stopCapture()
            val ortInfo = logcatCapture.parseOrtInfo()

            // Save partition info if captured
            if (ortInfo.hasData()) {
                OrtLogInfo.saveForModel(this@MainActivity, entry.name, ortInfo)
            }

            val resultMsg = if (success) {
                "Cache built: ${entry.name} (${runner.sessionCreateMs}ms)"
            } else {
                "Cache build failed: ${runner.lastError}"
            }

            runner.release()

            runOnUiThread {
                binding.tvCacheStatus.text = resultMsg
                binding.btnManageCache.isEnabled = true
                binding.btnPrepareCache.isEnabled = true
                Toast.makeText(this@MainActivity, resultMsg, Toast.LENGTH_LONG).show()
                updateCacheStatus()
            }
        }
    }

    private fun showCleanUpDialog(entries: List<ModelEntry>, profileFiles: List<File>) {
        data class CleanItem(val label: String, val files: List<File>)
        val items = mutableListOf<CleanItem>()

        val cacheFiles = entries.mapNotNull { it.cacheFile }
        if (cacheFiles.isNotEmpty()) {
            val size = formatSize(cacheFiles.sumOf { it.length() })
            items.add(CleanItem("QNN Cache (${cacheFiles.size}, $size)", cacheFiles))
        }
        val partFiles = entries.mapNotNull { it.partitionFile }
        if (partFiles.isNotEmpty()) {
            items.add(CleanItem("Partition Info (${partFiles.size})", partFiles))
        }
        if (profileFiles.isNotEmpty()) {
            val size = formatSize(profileFiles.sumOf { it.length() })
            items.add(CleanItem("ORT Profiles (${profileFiles.size}, $size)", profileFiles))
        }

        if (items.isEmpty()) {
            Toast.makeText(this, "Nothing to clean", Toast.LENGTH_SHORT).show()
            return
        }

        val labels = items.map { it.label }.toTypedArray()
        val checked = BooleanArray(items.size) { true }

        AlertDialog.Builder(this)
            .setTitle("Clean Up")
            .setMultiChoiceItems(labels, checked) { _, which, isChecked ->
                checked[which] = isChecked
            }
            .setPositiveButton("Delete") { _, _ ->
                var deleted = 0
                for ((i, item) in items.withIndex()) {
                    if (!checked[i]) continue
                    for (f in item.files) { if (f.delete()) deleted++ }
                }
                Toast.makeText(this, "Deleted $deleted file(s)", Toast.LENGTH_SHORT).show()
                updateCacheStatus()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        clearSelectedInputs()
        benchmarkRunner.release()
    }
}
