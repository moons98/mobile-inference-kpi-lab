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

        binding.btnManageModels.setOnClickListener {
            showDeviceModelManager()
        }

        // Strength label update
        binding.radioGroupStrength.setOnCheckedChangeListener { _, _ -> updateStrengthLabel() }
        binding.radioGroupSteps.setOnCheckedChangeListener { _, _ -> updateStrengthLabel() }

        // Precision filter: NPU disables FP32 (auto-converted to FP16)
        binding.radioGroupEp.setOnCheckedChangeListener { _, checkedId ->
            val isNpu = checkedId == R.id.radioEpNpu
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

        // Skip Text Encode toggle → hide/show Text Enc precision row
        binding.checkSkipTextEncode.setOnCheckedChangeListener { _, isChecked ->
            updateTextEncVisibility(!isChecked)
        }
        updateTextEncVisibility(!binding.checkSkipTextEncode.isChecked)
    }

    private fun updateTextEncVisibility(visible: Boolean) {
        val vis = if (visible) View.VISIBLE else View.GONE
        binding.tvLabelTextEnc.visibility = vis
        binding.spinnerPrecTextEnc.visibility = vis
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
        binding.checkSkipTextEncode.isEnabled = enabled
        binding.btnManageModels.isEnabled = enabled
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
            skipTextEncode = binding.checkSkipTextEncode.isChecked,
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

    private fun formatSize(bytes: Long): String = when {
        bytes >= 1024L * 1024 * 1024 -> "%.1f GB".format(bytes / (1024.0 * 1024 * 1024))
        bytes >= 1024L * 1024 -> "%.1f MB".format(bytes / (1024.0 * 1024))
        bytes >= 1024L -> "%.1f KB".format(bytes / 1024.0)
        else -> "$bytes B"
    }

    /** Represents a deployed model file pair (stub .onnx + .bin) on device. */
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
        val title = "${models.size} models (${formatSize(totalSize)})"

        // List items: "model_name\n  stub+bin  236.7 MB"
        val items = models.map { m ->
            if (m.isPrecompiled) {
                "${m.name}\n   stub+bin  ${formatSize(m.binSize)}"
            } else {
                "${m.name}\n   onnx  ${formatSize(m.onnxSize)}"
            }
        }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle(title)
            .setItems(items) { _, which ->
                showModelDetail(models[which])
            }
            .setNegativeButton("Close", null)
            .show()
    }

    private fun showModelDetail(model: DeviceModel) {
        val info = buildString {
            appendLine(model.name)
            appendLine()
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
                // Refresh list
                showDeviceModelManager()
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
