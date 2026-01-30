package com.example.kpilab

import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.kpilab.batch.BatchProgress
import com.example.kpilab.batch.ExperimentSet
import com.example.kpilab.batch.ExperimentSetLoader
import com.example.kpilab.databinding.ActivityMainBinding
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var binding: ActivityMainBinding

    private lateinit var kpiCollector: KpiCollector
    private lateinit var benchmarkRunner: BenchmarkRunner
    private lateinit var experimentSetLoader: ExperimentSetLoader

    private var isForeground = true
    private var isBatchMode = false
    private var experimentSets: List<ExperimentSet> = emptyList()

    // All available model types for the spinner
    private val modelTypes = OnnxModelType.values()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initializeComponents()
        setupUI()
        setupModelSpinner()
        setupBatchMode()
        observeProgress()
        observeBatchProgress()
    }

    private fun initializeComponents() {
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
                stopBenchmark()
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
        // Update status text
        binding.tvStatus.text = when (progress.state) {
            BenchmarkState.IDLE -> "Idle"
            BenchmarkState.INITIALIZING -> "Initializing..."
            BenchmarkState.WARMING_UP -> "Warming up..."
            BenchmarkState.RUNNING -> "Running"
            BenchmarkState.STOPPING -> "Stopping..."
        }

        // Update progress
        binding.tvProgress.text = "${progress.formatElapsed()} / ${progress.formatTotal()}"

        // Update metrics
        binding.tvInferences.text = progress.inferenceCount.toString()

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

        // Update button state
        val isRunning = progress.state == BenchmarkState.RUNNING ||
                        progress.state == BenchmarkState.WARMING_UP ||
                        progress.state == BenchmarkState.INITIALIZING

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
        binding.checkFp16.isEnabled = enabled
        binding.checkCache.isEnabled = enabled
        binding.radioGroupDuration.isEnabled = enabled
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

    private fun stopBenchmark() {
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
        val csvData = benchmarkRunner.exportCsv()
        val recordCount = benchmarkRunner.getRecordCount()

        if (csvData.isEmpty() || recordCount == 0) {
            Toast.makeText(this, "No data to export", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            // Generate filename with model name, EP, and timestamp
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val modelName = benchmarkRunner.currentModel?.displayName
                ?.replace(" ", "")
                ?.replace("-", "")
                ?.replace("(", "")
                ?.replace(")", "")
                ?: "Unknown"
            val ep = benchmarkRunner.getActiveExecutionProvider()
                .replace("_", "")
                .uppercase()
            val baseFilename = "kpi_${modelName}_${ep}_${timestamp}"

            // Save to app's external files directory
            val exportDir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            if (exportDir != null) {
                // Save CSV
                val csvFile = File(exportDir, "${baseFilename}.csv")
                FileWriter(csvFile).use { writer ->
                    writer.write(csvData)
                }
                Log.i(TAG, "Exported $recordCount records to: ${csvFile.absolutePath}")

                // Save ORT logs alongside CSV
                var logExported = false
                benchmarkRunner.getOrtLogInfo()?.let { ortInfo ->
                    if (ortInfo.rawLogs.isNotBlank()) {
                        val logFile = File(exportDir, "${baseFilename}_ort.log")
                        FileWriter(logFile).use { writer ->
                            writer.write(ortInfo.toSummary())
                            writer.write("\n\n=== Raw Logs ===\n")
                            writer.write(ortInfo.rawLogs)
                        }
                        Log.i(TAG, "ORT logs exported: ${logFile.absolutePath}")
                        logExported = true
                    }
                }

                val logMsg = if (logExported) " + ORT log" else ""
                Toast.makeText(
                    this,
                    "Exported $recordCount records$logMsg to:\n${csvFile.name}",
                    Toast.LENGTH_LONG
                ).show()
            } else {
                Toast.makeText(this, "Export directory unavailable", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Export failed: ${e.message}", e)
            Toast.makeText(this, "Export failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onResume() {
        super.onResume()
        isForeground = true
        benchmarkRunner.setForeground(true)
        Log.d(TAG, "App resumed (foreground)")
    }

    override fun onPause() {
        super.onPause()
        isForeground = false
        benchmarkRunner.setForeground(false)
        Log.d(TAG, "App paused (background)")
    }

    override fun onDestroy() {
        super.onDestroy()
        benchmarkRunner.release()
    }
}
