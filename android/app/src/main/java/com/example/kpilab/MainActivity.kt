package com.example.kpilab

import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
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

    private var isForeground = true

    // All available model types for the spinner
    private val modelTypes = OnnxModelType.values()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initializeComponents()
        setupUI()
        setupModelSpinner()
        observeProgress()
    }

    private fun initializeComponents() {
        kpiCollector = KpiCollector(this)
        benchmarkRunner = BenchmarkRunner(this, kpiCollector)
    }

    private fun setupUI() {
        // Start/Stop button
        binding.btnStartStop.setOnClickListener {
            if (benchmarkRunner.isRunning) {
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
        // Model selection
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
        binding.checkWarmup.isEnabled = enabled
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
            warmUpEnabled = binding.checkWarmup.isChecked,
            durationMinutes = durationMinutes,
            useNpuFp16 = binding.checkFp16.isChecked,
            useContextCache = binding.checkCache.isChecked
        )
    }

    private fun startBenchmark() {
        val config = buildConfig()
        Log.i(TAG, "Starting benchmark with config: $config")

        benchmarkRunner.start(config, lifecycleScope)

        Toast.makeText(this, "Benchmark started", Toast.LENGTH_SHORT).show()
    }

    private fun stopBenchmark() {
        Log.i(TAG, "Stopping benchmark")
        benchmarkRunner.stop()
        Toast.makeText(this, "Benchmark stopped", Toast.LENGTH_SHORT).show()
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
                ?: "Unknown"
            val ep = benchmarkRunner.getActiveExecutionProvider()
                .replace("_", "")
                .uppercase()
            val filename = "kpi_${modelName}_${ep}_${timestamp}.csv"

            // Save to app's external files directory
            val exportDir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            if (exportDir != null) {
                val file = File(exportDir, filename)
                FileWriter(file).use { writer ->
                    writer.write(csvData)
                }

                Log.i(TAG, "Exported $recordCount records to: ${file.absolutePath}")
                Toast.makeText(
                    this,
                    "Exported $recordCount records to:\n${file.name}",
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
