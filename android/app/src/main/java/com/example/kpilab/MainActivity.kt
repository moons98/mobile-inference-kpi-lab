package com.example.kpilab

import android.os.Bundle
import android.os.Environment
import android.util.Log
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
        private const val MODEL_NAME = "mobilenetv3_small.dlc"
    }

    private lateinit var binding: ActivityMainBinding

    private lateinit var nativeRunner: NativeRunner
    private lateinit var kpiCollector: KpiCollector
    private lateinit var benchmarkRunner: BenchmarkRunner

    private var isForeground = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initializeComponents()
        setupUI()
        observeProgress()
    }

    private fun initializeComponents() {
        nativeRunner = NativeRunner()
        kpiCollector = KpiCollector(this)
        benchmarkRunner = BenchmarkRunner(nativeRunner, kpiCollector)
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

        binding.tvLatency.text = if (progress.lastLatencyMs > 0) {
            String.format("%.1f ms", progress.lastLatencyMs)
        } else {
            "-- ms"
        }

        binding.tvThermal.text = if (progress.lastThermalC > 0) {
            String.format("%.1f °C", progress.lastThermalC)
        } else {
            "-- °C"
        }

        binding.tvPower.text = if (progress.lastPowerMw > 0) {
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
        binding.radioGroupPath.isEnabled = enabled
        binding.radioNpuOnly.isEnabled = enabled
        binding.radioNpuFallback.isEnabled = enabled
        binding.radioGpuOnly.isEnabled = enabled
        binding.radioGroupFrequency.isEnabled = enabled
        binding.radioFreq1.isEnabled = enabled
        binding.radioFreq5.isEnabled = enabled
        binding.radioFreq10.isEnabled = enabled
        binding.checkWarmup.isEnabled = enabled
        binding.radioGroupDuration.isEnabled = enabled
        binding.radioDuration5.isEnabled = enabled
        binding.radioDuration10.isEnabled = enabled
    }

    private fun buildConfig(): BenchmarkConfig {
        val executionPath = when (binding.radioGroupPath.checkedRadioButtonId) {
            R.id.radioNpuOnly -> ExecutionPath.NPU_ONLY
            R.id.radioNpuFallback -> ExecutionPath.NPU_FALLBACK
            R.id.radioGpuOnly -> ExecutionPath.GPU_ONLY
            else -> ExecutionPath.NPU_FALLBACK
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
            executionPath = executionPath,
            frequencyHz = frequencyHz,
            warmUpEnabled = binding.checkWarmup.isChecked,
            durationMinutes = durationMinutes
        )
    }

    private fun startBenchmark() {
        val config = buildConfig()
        Log.i(TAG, "Starting benchmark with config: $config")

        // Model path - in production this would be extracted from assets
        // For now, use a placeholder path
        val modelPath = File(filesDir, MODEL_NAME).absolutePath

        benchmarkRunner.start(config, modelPath, lifecycleScope)

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
            // Generate filename with timestamp
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val filename = "kpi_log_${timestamp}.csv"

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
        nativeRunner.setForeground(true)
        Log.d(TAG, "App resumed (foreground)")
    }

    override fun onPause() {
        super.onPause()
        isForeground = false
        nativeRunner.setForeground(false)
        Log.d(TAG, "App paused (background)")
    }

    override fun onDestroy() {
        super.onDestroy()
        benchmarkRunner.release()
    }
}
