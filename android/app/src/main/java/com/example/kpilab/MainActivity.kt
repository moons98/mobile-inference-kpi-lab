package com.example.kpilab

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
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

        initializeComponents()
        setupUI()
        setupBatchMode()
        observeProgress()
        observeBatchProgress()
        observeGeneratedImage()
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

        // Strength label update
        binding.radioGroupStrength.setOnCheckedChangeListener { _, _ -> updateStrengthLabel() }
        binding.radioGroupSteps.setOnCheckedChangeListener { _, _ -> updateStrengthLabel() }

        // QNN options visibility
        binding.radioGroupEp.setOnCheckedChangeListener { _, checkedId ->
            binding.layoutQnnOptions.visibility = when (checkedId) {
                R.id.radioEpCpu -> View.GONE
                else -> View.VISIBLE
            }
        }
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
        binding.radioPrecFp32.isEnabled = enabled
        binding.radioPrecFp16.isEnabled = enabled
        binding.radioPrecW8a8.isEnabled = enabled
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
    }

    private fun buildConfig(): BenchmarkConfig {
        val phase = when (binding.radioGroupPhase.checkedRadioButtonId) {
            R.id.radioPhase2 -> BenchmarkPhase.SUSTAINED_ERASE
            R.id.radioPhaseYolo -> BenchmarkPhase.YOLO_SEG_ONLY
            else -> BenchmarkPhase.SINGLE_ERASE
        }

        return BenchmarkConfig(
            sdBackend = when (binding.radioGroupEp.checkedRadioButtonId) {
                R.id.radioEpGpu -> ExecutionProvider.QNN_GPU
                R.id.radioEpCpu -> ExecutionProvider.CPU
                else -> ExecutionProvider.QNN_NPU
            },
            sdPrecision = when (binding.radioGroupPrecision.checkedRadioButtonId) {
                R.id.radioPrecFp32 -> SdPrecision.FP32
                R.id.radioPrecW8a8 -> SdPrecision.W8A8
                else -> SdPrecision.FP16
            },
            phase = phase,
            steps = getSelectedSteps(),
            strength = getSelectedStrength(),
            roiSize = when (binding.radioGroupRoiSize.checkedRadioButtonId) {
                R.id.radioRoiSmall -> RoiSize.SMALL
                R.id.radioRoiLarge -> RoiSize.LARGE
                else -> RoiSize.MEDIUM
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
            selectedImageBitmap?.let { benchmarkRunner.setSourceImage(it) }
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

        selectedImageBitmap?.let { benchmarkRunner.setSourceImage(it) }
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
            val bitmap = assets.open(builtin.imagePath).use { BitmapFactory.decodeStream(it) }
            if (bitmap != null) {
                selectedImageBitmap = bitmap
                benchmarkRunner.setSourceImage(bitmap)

                // Load paired mask if available
                if (builtin.maskPath.isNotEmpty()) {
                    val mask = assets.open(builtin.maskPath).use { BitmapFactory.decodeStream(it) }
                    if (mask != null) {
                        selectedMaskBitmap = mask
                        benchmarkRunner.setSourceMask(mask)
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
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            if (bitmap != null) {
                selectedImageBitmap = bitmap
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

    override fun onDestroy() {
        super.onDestroy()
        benchmarkRunner.release()
    }
}
