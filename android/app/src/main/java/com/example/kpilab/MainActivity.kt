package com.example.kpilab

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Toast
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

    // All available model types for the spinner
    private val modelTypes = OnnxModelType.entries

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initializeComponents()
        setupUI()
        setupModelSpinner()
        setupBatchMode()
        setupEpOptionVisibility()
        observeProgress()
        observeBatchProgress()
    }

    private fun initializeComponents() {
        // Initialize QNN libraries from assets
        // This extracts QNN .so files and sets up ADSP_LIBRARY_PATH for DSP access
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

    private fun setupBatchMode() {
        // Setup experiment set spinner
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            experimentSets.map { it.name }
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerExperimentSet.adapter = adapter

        // Batch mode checkbox listener
        binding.checkBatchMode.setOnCheckedChangeListener { _, isChecked ->
            isBatchMode = isChecked
            updateBatchModeUI()
        }

        // Initial state
        updateBatchModeUI()
    }

    /**
     * Show/hide QNN-specific options (FP16, context cache) based on selected EP.
     * These options are only relevant for NPU; hidden for CPU, visible but FP16 disabled for GPU.
     */
    private fun setupEpOptionVisibility() {
        binding.radioGroupPath.setOnCheckedChangeListener { _, checkedId ->
            updateQnnOptionsVisibility(checkedId)
        }
        // Apply initial state
        updateQnnOptionsVisibility(binding.radioGroupPath.checkedRadioButtonId)
    }

    private fun updateQnnOptionsVisibility(checkedId: Int) {
        when (checkedId) {
            R.id.radioNpuGpuCpu -> {
                // NPU: show all QNN options
                binding.layoutQnnOptions.visibility = View.VISIBLE
                binding.checkFp16.isEnabled = true
                binding.checkCache.isEnabled = true
            }
            R.id.radioGpuCpu -> {
                // GPU: show cache option but FP16 is NPU-only
                binding.layoutQnnOptions.visibility = View.VISIBLE
                binding.checkFp16.isEnabled = false
                binding.checkFp16.isChecked = false
                binding.checkCache.isEnabled = true
            }
            R.id.radioCpuOnly -> {
                // CPU: hide all QNN options
                binding.layoutQnnOptions.visibility = View.GONE
            }
        }
    }

    private fun updateBatchModeUI() {
        // Show/hide batch options
        binding.layoutBatchOptions.visibility = if (isBatchMode) View.VISIBLE else View.GONE

        // Show/hide single mode cards
        val singleModeVisibility = if (isBatchMode) View.GONE else View.VISIBLE
        binding.cardSingleMode.visibility = singleModeVisibility
        binding.cardExecutionPath.visibility = singleModeVisibility
        binding.cardFrequency.visibility = singleModeVisibility
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

    private fun updateBatchUI(progress: BatchProgress) {
        // Show/hide batch progress card
        binding.cardBatchProgress.visibility = if (progress.isRunning || progress.completedExperiments.isNotEmpty()) {
            View.VISIBLE
        } else {
            View.GONE
        }

        // Update experiment progress
        binding.tvBatchExperiment.text = progress.formatProgress()
        binding.tvCurrentExperiment.text = progress.currentExperimentName.ifEmpty { "--" }

        // Update cooldown status
        if (progress.isCoolingDown) {
            binding.layoutCooldown.visibility = View.VISIBLE
            binding.tvCooldown.text = "${progress.cooldownRemainingSeconds} sec"
        } else {
            binding.layoutCooldown.visibility = View.GONE
        }

        // Update completed experiments list
        if (progress.completedExperiments.isNotEmpty()) {
            val completedText = progress.completedExperiments.mapIndexed { index, path ->
                val filename = File(path).name
                "${index + 1}. $filename"
            }.joinToString("\n")
            binding.tvCompletedExperiments.text = "Completed:\n$completedText"
        } else {
            binding.tvCompletedExperiments.text = ""
        }

        // Update button state for batch mode
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
        // Show error dialog on initialization failure
        if (progress.state == BenchmarkState.ERROR && progress.errorMessage != null) {
            Toast.makeText(this, progress.errorMessage, Toast.LENGTH_LONG).show()
        }

        // Update status text
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

        // Update progress bar
        binding.progressBar.visibility = if (isRunning) View.VISIBLE else View.GONE
        binding.progressBar.progress = progress.progressPercent

        // Update progress text
        binding.tvProgress.text = "${progress.formatElapsed()} / ${progress.formatTotal()}"

        // Update metrics
        binding.tvInferences.text = progress.inferenceCount.toString()

        // Throughput
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

        // Update button state
        binding.btnStartStop.text = if (isRunning) {
            getString(R.string.stop_benchmark)
        } else {
            getString(R.string.start_benchmark)
        }

        // Enable/disable controls
        setControlsEnabled(!isRunning)

        // Enable export if we have data
        binding.btnExport.isEnabled = !isRunning && benchmarkRunner.getRecordCount() > 0
    }

    private fun setControlsEnabled(enabled: Boolean) {
        // Batch mode toggle
        binding.checkBatchMode.isEnabled = enabled
        binding.spinnerExperimentSet.isEnabled = enabled

        // Model selection (single mode only)
        binding.spinnerModel.isEnabled = enabled

        // Execution provider selection
        binding.radioGroupPath.isEnabled = enabled
        binding.radioNpuGpuCpu.isEnabled = enabled
        binding.radioGpuCpu.isEnabled = enabled
        binding.radioCpuOnly.isEnabled = enabled

        // Frequency
        binding.radioGroupFrequency.isEnabled = enabled
        binding.radioFreq1.isEnabled = enabled
        binding.radioFreq5.isEnabled = enabled
        binding.radioFreq10.isEnabled = enabled

        // Options
        if (enabled) {
            // Restore QNN options visibility based on selected EP
            updateQnnOptionsVisibility(binding.radioGroupPath.checkedRadioButtonId)
        } else {
            binding.checkFp16.isEnabled = false
            binding.checkCache.isEnabled = false
        }
        binding.radioGroupDuration.isEnabled = enabled
        binding.radioDuration1.isEnabled = enabled
        binding.radioDuration2.isEnabled = enabled
        binding.radioDuration5.isEnabled = enabled
        binding.radioDuration10.isEnabled = enabled
    }

    private fun buildConfig(): BenchmarkConfig {
        // Model type from spinner selection
        val modelType = modelTypes[binding.spinnerModel.selectedItemPosition]

        // Execution provider
        val executionProvider = when (binding.radioGroupPath.checkedRadioButtonId) {
            R.id.radioNpuGpuCpu -> ExecutionProvider.QNN_NPU
            R.id.radioGpuCpu -> ExecutionProvider.QNN_GPU
            R.id.radioCpuOnly -> ExecutionProvider.CPU
            else -> ExecutionProvider.QNN_NPU
        }

        val frequencyHz = when (binding.radioGroupFrequency.checkedRadioButtonId) {
            R.id.radioFreq1 -> 1
            R.id.radioFreq5 -> 5
            R.id.radioFreq10 -> 10
            else -> 5
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
            frequencyHz = frequencyHz,
            durationMinutes = durationMinutes,
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

    /**
     * Show confirmation dialog before stopping a running benchmark.
     */
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
        benchmarkRunner.release()
    }
}
