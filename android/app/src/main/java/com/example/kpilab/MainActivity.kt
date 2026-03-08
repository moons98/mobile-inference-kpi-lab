package com.example.kpilab

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.kpilab.batch.BatchProgress
import com.example.kpilab.batch.ExperimentSet
import com.example.kpilab.batch.ExperimentSetLoader
import com.example.kpilab.databinding.ActivityMainBinding
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import android.widget.AdapterView
import java.io.File

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var binding: ActivityMainBinding

    private lateinit var kpiCollector: KpiCollector
    private lateinit var benchmarkRunner: BenchmarkRunner
    private lateinit var experimentSetLoader: ExperimentSetLoader
    private var cameraManager: CameraManager? = null

    private var isBatchMode = false
    private var experimentSets: List<ExperimentSet> = emptyList()
    private var cameraPermissionGranted = false

    // All available model types for the spinner
    private val modelTypes = OnnxModelType.entries

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        cameraPermissionGranted = granted
        if (granted) {
            initializeCamera()
        } else {
            Log.w(TAG, "Camera permission denied - camera modes unavailable")
            Toast.makeText(this, "Camera permission denied. Static image mode only.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initializeComponents()
        setupUI()
        setupModelSpinner()
        setupBatchMode()
        setupPhaseInputControls()
        setupEpOptionVisibility()
        observeProgress()
        observeBatchProgress()
        observeDetections()
        requestCameraPermission()
    }

    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            cameraPermissionGranted = true
            initializeCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun initializeCamera() {
        cameraManager = CameraManager(this)
        benchmarkRunner.cameraManager = cameraManager
        benchmarkRunner.overlayView = binding.overlayView
        Log.i(TAG, "CameraManager initialized")

        // Auto-start camera preview if overlay is enabled by default
        if (binding.checkDemoMode.isChecked) {
            updateCameraPreview(true)
        }
    }

    private fun initializeComponents() {
        // Initialize QNN libraries from assets
        val qnnInitialized = OrtRunner.initializeQnnLibraries(this)
        if (qnnInitialized) {
            Log.i(TAG, "QNN libraries initialized successfully")
        } else {
            Log.w(TAG, "QNN libraries initialization failed - NPU may not work")
            Toast.makeText(
                this,
                "QNN initialization failed - NPU/GPU may not work",
                Toast.LENGTH_LONG
            ).show()
        }

        kpiCollector = KpiCollector(this)
        benchmarkRunner = BenchmarkRunner(this, kpiCollector)
        experimentSetLoader = ExperimentSetLoader(this)

        // Load experiment sets
        experimentSets = experimentSetLoader.getExperimentSets()
        Log.i(TAG, "Loaded ${experimentSets.size} experiment sets")
    }

    private fun setupUI() {
        // Start/Stop button
        binding.btnStartStop.setOnClickListener {
            if (benchmarkRunner.isRunning || benchmarkRunner.isBatchRunning) {
                confirmStop()
            } else {
                startBenchmark()
            }
        }

        // Export button
        binding.btnExport.setOnClickListener {
            exportData()
        }

        // Initially disable export
        binding.btnExport.isEnabled = false
    }

    private fun setupPhaseInputControls() {
        // Phase selection auto-sets input mode defaults
        binding.radioGroupPhase.setOnCheckedChangeListener { _, checkedId ->
            when (checkedId) {
                R.id.radioPhaseBurst -> {
                    binding.radioInputCameraSingle.isChecked = true
                }
                R.id.radioPhaseSustained -> {
                    binding.radioInputCameraLive.isChecked = true
                }
            }
        }

        // Demo mode checkbox controls camera preview visibility
        binding.checkDemoMode.setOnCheckedChangeListener { _, isChecked ->
            updateCameraPreview(isChecked)
        }
    }

    private fun updateCameraPreview(demoMode: Boolean) {
        if (demoMode && cameraPermissionGranted) {
            binding.cardCameraPreview.visibility = View.VISIBLE
            // Start camera preview
            cameraManager?.start(this, binding.previewView)
        } else {
            binding.cardCameraPreview.visibility = View.GONE
            cameraManager?.stop()
        }
    }

    private fun setupBatchMode() {
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            experimentSets.map { it.name }
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerExperimentSet.adapter = adapter

        binding.checkBatchMode.setOnCheckedChangeListener { _, isChecked ->
            isBatchMode = isChecked
            updateBatchModeUI()
        }

        // Show experiment detail when set is selected
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

    private fun setupEpOptionVisibility() {
        binding.radioGroupPath.setOnCheckedChangeListener { _, checkedId ->
            updateQnnOptionsVisibility(checkedId)
        }
        updateQnnOptionsVisibility(binding.radioGroupPath.checkedRadioButtonId)
    }

    private fun updateQnnOptionsVisibility(checkedId: Int) {
        when (checkedId) {
            R.id.radioNpuGpuCpu -> {
                binding.layoutQnnOptions.visibility = View.VISIBLE
                binding.checkFp16.isEnabled = true
                binding.checkCache.isEnabled = true
            }
            R.id.radioGpuCpu -> {
                binding.layoutQnnOptions.visibility = View.VISIBLE
                binding.checkFp16.isEnabled = false
                binding.checkFp16.isChecked = false
                binding.checkCache.isEnabled = true
            }
            R.id.radioCpuOnly -> {
                binding.layoutQnnOptions.visibility = View.GONE
            }
        }
    }

    private fun updateExperimentDetail(position: Int) {
        if (position < 0 || position >= experimentSets.size) {
            binding.layoutExperimentDetail.visibility = View.GONE
            return
        }

        val set = experimentSets[position]
        val defaults = experimentSetLoader.getDefaults()

        // Defaults summary
        val phase = defaults.phase.replace("_", " ")
        val input = defaults.inputMode.replace("_", " ")
        val defaultsText = buildString {
            append("Defaults: $phase | $input")
            if (defaults.phase == "BURST") {
                append(" | ${defaults.iterations}iter @ ${defaults.frequencyHz}Hz")
            } else {
                append(" | ${defaults.durationMinutes}min @ ${defaults.frequencyHz}Hz")
            }
        }
        binding.tvExperimentDefaults.text = defaultsText

        // Experiment list with index and overrides
        val listText = set.experiments.mapIndexed { i, exp ->
            "${i + 1}. ${exp.getDetailLine(defaults)}"
        }.joinToString("\n")
        binding.tvExperimentList.text = listText

        binding.layoutExperimentDetail.visibility = View.VISIBLE
    }

    private fun updateBatchModeUI() {
        binding.layoutBatchOptions.visibility = if (isBatchMode) View.VISIBLE else View.GONE

        if (isBatchMode) {
            // Show detail for current selection
            updateExperimentDetail(binding.spinnerExperimentSet.selectedItemPosition)
        } else {
            binding.layoutExperimentDetail.visibility = View.GONE
        }

        val singleModeVisibility = if (isBatchMode) View.GONE else View.VISIBLE
        binding.cardPhaseInput.visibility = singleModeVisibility
        binding.cardSingleMode.visibility = singleModeVisibility
        binding.cardExecutionPath.visibility = singleModeVisibility
        binding.cardOptions.visibility = singleModeVisibility
    }

    private fun setupModelSpinner() {
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            modelTypes.map { it.displayName }
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerModel.adapter = adapter
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

    private fun observeDetections() {
        lifecycleScope.launch {
            benchmarkRunner.lastDetections.collectLatest { result ->
                if (binding.checkDemoMode.isChecked && result.detections.isNotEmpty()) {
                    binding.overlayView.setDetections(result.detections, result.sourceWidth, result.sourceHeight)
                }
            }
        }
    }

    private fun updateBatchUI(progress: BatchProgress) {
        binding.cardBatchProgress.visibility = if (progress.isRunning || progress.completedExperiments.isNotEmpty()) {
            View.VISIBLE
        } else {
            View.GONE
        }

        binding.tvBatchExperiment.text = progress.formatProgress()
        binding.tvCurrentExperiment.text = progress.currentExperimentName.ifEmpty { "--" }

        if (progress.isCoolingDown) {
            binding.layoutCooldown.visibility = View.VISIBLE
            binding.tvCooldown.text = "${progress.cooldownRemainingSeconds} sec"
        } else {
            binding.layoutCooldown.visibility = View.GONE
        }

        if (progress.completedExperiments.isNotEmpty()) {
            val completedText = progress.completedExperiments.mapIndexed { index, path ->
                val filename = File(path).name
                "${index + 1}. $filename"
            }.joinToString("\n")
            binding.tvCompletedExperiments.text = "Completed:\n$completedText"
        } else {
            binding.tvCompletedExperiments.text = ""
        }

        if (progress.isRunning) {
            binding.btnStartStop.text = getString(R.string.stop_benchmark)
            setControlsEnabled(false)
            binding.checkBatchMode.isEnabled = false
        } else if (!benchmarkRunner.isRunning) {
            binding.btnStartStop.text = getString(R.string.start_benchmark)
            setControlsEnabled(true)
            binding.checkBatchMode.isEnabled = true
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

        val isRunning = progress.state == BenchmarkState.RUNNING ||
                        progress.state == BenchmarkState.WARMING_UP ||
                        progress.state == BenchmarkState.INITIALIZING

        binding.progressBar.visibility = if (isRunning) View.VISIBLE else View.GONE
        binding.progressBar.progress = progress.progressPercent
        binding.tvProgress.text = "${progress.formatElapsed()} / ${progress.formatTotal()}"
        binding.tvInferences.text = progress.inferenceCount.toString()

        binding.tvThroughput.text = if (progress.throughput > 0) {
            progress.formatThroughput()
        } else {
            "-- inf/s"
        }

        binding.tvLatency.text = if (progress.lastLatencyMs >= 0) {
            String.format("%.1f ms", progress.lastLatencyMs)
        } else {
            "-- ms"
        }

        // Frame drop (show for sustained phase)
        if (progress.frameDropCount > 0 || progress.inferenceCount > 100) {
            binding.layoutFrameDrop.visibility = View.VISIBLE
            binding.tvFrameDrop.text = "${progress.frameDropCount} (${String.format("%.1f", progress.frameDropRate)}%)"
        }

        binding.tvThermal.text = if (progress.lastThermalC >= 0) {
            String.format("%.1f °C", progress.lastThermalC)
        } else {
            "-- °C"
        }

        binding.tvPower.text = if (progress.lastPowerMw >= 0) {
            String.format("%.0f mW", progress.lastPowerMw)
        } else {
            "-- mW"
        }

        binding.tvMemory.text = if (progress.lastMemoryMb > 0) {
            "${progress.lastMemoryMb} MB"
        } else {
            "-- MB"
        }

        binding.btnStartStop.text = if (isRunning) {
            getString(R.string.stop_benchmark)
        } else {
            getString(R.string.start_benchmark)
        }

        setControlsEnabled(!isRunning)
        binding.btnExport.isEnabled = !isRunning && benchmarkRunner.getRecordCount() > 0
    }

    private fun setControlsEnabled(enabled: Boolean) {
        binding.checkBatchMode.isEnabled = enabled
        binding.spinnerExperimentSet.isEnabled = enabled
        binding.spinnerModel.isEnabled = enabled

        // Phase & Input controls
        binding.radioPhaseBurst.isEnabled = enabled
        binding.radioPhaseSustained.isEnabled = enabled
        binding.radioInputCameraSingle.isEnabled = enabled
        binding.radioInputCameraLive.isEnabled = enabled
        binding.radioInputStatic.isEnabled = enabled
        binding.checkDemoMode.isEnabled = enabled

        // Execution provider
        binding.radioNpuGpuCpu.isEnabled = enabled
        binding.radioGpuCpu.isEnabled = enabled
        binding.radioCpuOnly.isEnabled = enabled

        // Options
        if (enabled) {
            updateQnnOptionsVisibility(binding.radioGroupPath.checkedRadioButtonId)
        } else {
            binding.checkFp16.isEnabled = false
            binding.checkCache.isEnabled = false
        }
        binding.radioDuration1.isEnabled = enabled
        binding.radioDuration2.isEnabled = enabled
        binding.radioDuration5.isEnabled = enabled
        binding.radioDuration10.isEnabled = enabled
    }

    private fun buildConfig(): BenchmarkConfig {
        val modelType = modelTypes[binding.spinnerModel.selectedItemPosition]

        val executionProvider = when (binding.radioGroupPath.checkedRadioButtonId) {
            R.id.radioNpuGpuCpu -> ExecutionProvider.QNN_NPU
            R.id.radioGpuCpu -> ExecutionProvider.QNN_GPU
            R.id.radioCpuOnly -> ExecutionProvider.CPU
            else -> ExecutionProvider.QNN_NPU
        }

        val phase = when (binding.radioGroupPhase.checkedRadioButtonId) {
            R.id.radioPhaseBurst -> BenchmarkPhase.BURST
            R.id.radioPhaseSustained -> BenchmarkPhase.SUSTAINED
            else -> BenchmarkPhase.BURST
        }

        val inputMode = when (binding.radioGroupInput.checkedRadioButtonId) {
            R.id.radioInputCameraSingle -> InputMode.CAMERA_SINGLE
            R.id.radioInputCameraLive -> InputMode.CAMERA_LIVE
            R.id.radioInputStatic -> InputMode.STATIC_IMAGE
            else -> InputMode.CAMERA_SINGLE
        }

        val durationMinutes = when (binding.radioGroupDuration.checkedRadioButtonId) {
            R.id.radioDuration1 -> 1
            R.id.radioDuration2 -> 2
            R.id.radioDuration5 -> 5
            R.id.radioDuration10 -> 10
            else -> 5
        }

        return BenchmarkConfig(
            modelType = modelType,
            executionProvider = executionProvider,
            phase = phase,
            inputMode = inputMode,
            demoMode = binding.checkDemoMode.isChecked,
            frequencyHz = if (phase == BenchmarkPhase.BURST) 2 else 30,
            durationMinutes = durationMinutes,
            iterations = 100,
            useNpuFp16 = binding.checkFp16.isChecked,
            useContextCache = binding.checkCache.isChecked
        )
    }

    private fun startBenchmark() {
        if (isBatchMode) {
            startBatchBenchmark()
        } else {
            startSingleBenchmark()
        }
    }

    private fun startSingleBenchmark() {
        val config = buildConfig()

        // Ensure camera is running for camera modes
        if (config.usesCamera && cameraPermissionGranted && cameraManager?.isRunning != true) {
            cameraManager?.start(this, if (config.demoMode) binding.previewView else null)
        }

        Log.i(TAG, "Starting benchmark with config: $config")
        benchmarkRunner.start(config, lifecycleScope)
        Toast.makeText(this, "Benchmark started", Toast.LENGTH_SHORT).show()
    }

    private fun startBatchBenchmark() {
        if (experimentSets.isEmpty()) {
            Toast.makeText(this, "No experiment sets available", Toast.LENGTH_SHORT).show()
            return
        }

        val selectedIndex = binding.spinnerExperimentSet.selectedItemPosition
        if (selectedIndex < 0 || selectedIndex >= experimentSets.size) {
            Toast.makeText(this, "Please select an experiment set", Toast.LENGTH_SHORT).show()
            return
        }

        val selectedSet = experimentSets[selectedIndex]
        val defaults = experimentSetLoader.getDefaults()

        // Start camera for batch (needed for camera input modes)
        if (cameraPermissionGranted && cameraManager?.isRunning != true) {
            cameraManager?.start(this)
        }

        Log.i(TAG, "Starting batch: ${selectedSet.name} with ${selectedSet.experiments.size} experiments")

        benchmarkRunner.startBatch(
            experimentSet = selectedSet,
            defaults = defaults,
            scope = lifecycleScope,
            onExperimentComplete = { csvPath ->
                runOnUiThread {
                    val filename = File(csvPath).name
                    Toast.makeText(this, "Saved: $filename", Toast.LENGTH_SHORT).show()
                }
            }
        )

        Toast.makeText(this, "Batch started: ${selectedSet.name}", Toast.LENGTH_SHORT).show()
    }

    private fun confirmStop() {
        val message = if (benchmarkRunner.isBatchRunning) {
            "Stop the batch experiment? Current experiment data will be lost."
        } else {
            "Stop the benchmark?"
        }

        AlertDialog.Builder(this)
            .setTitle("Stop Benchmark")
            .setMessage(message)
            .setPositiveButton("Stop") { _, _ -> doStop() }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun doStop() {
        Log.i(TAG, "Stopping benchmark")
        if (benchmarkRunner.isBatchRunning) {
            benchmarkRunner.stopBatch()
            Toast.makeText(this, "Batch stopped", Toast.LENGTH_SHORT).show()
        } else {
            benchmarkRunner.stop()
            Toast.makeText(this, "Benchmark stopped", Toast.LENGTH_SHORT).show()
        }
    }

    private fun exportData() {
        val recordCount = benchmarkRunner.getRecordCount()
        if (recordCount == 0) {
            Toast.makeText(this, "No data to export", Toast.LENGTH_SHORT).show()
            return
        }

        val csvPath = benchmarkRunner.exportAndSaveCsv()
        if (csvPath != null) {
            val filename = File(csvPath).name
            Toast.makeText(
                this,
                "Exported $recordCount records to:\n$filename",
                Toast.LENGTH_LONG
            ).show()
        } else {
            Toast.makeText(this, "Export failed", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onResume() {
        super.onResume()
        benchmarkRunner.setForeground(true)
        Log.d(TAG, "App resumed (foreground)")
    }

    override fun onPause() {
        super.onPause()
        benchmarkRunner.setForeground(false)
        Log.d(TAG, "App paused (background)")
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager?.release()
        benchmarkRunner.release()
    }
}
