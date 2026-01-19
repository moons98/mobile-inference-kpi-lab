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
    }

    private lateinit var binding: ActivityMainBinding

    private lateinit var tfliteRunner: TFLiteRunner
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
        tfliteRunner = TFLiteRunner(this)
        kpiCollector = KpiCollector(this)
        benchmarkRunner = BenchmarkRunner(tfliteRunner, kpiCollector)
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
        binding.radioGroupModel.isEnabled = enabled
        binding.radioMobilenetV2.isEnabled = enabled
        binding.radioMobilenetV2Quant.isEnabled = enabled
        binding.radioYolov8n.isEnabled = enabled
        binding.radioYolov8nQuant.isEnabled = enabled
        binding.radioGroupPath.isEnabled = enabled
        binding.radioNpuGpuCpu.isEnabled = enabled
        binding.radioGpuCpu.isEnabled = enabled
        binding.radioCpuOnly.isEnabled = enabled
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
        val modelType = when (binding.radioGroupModel.checkedRadioButtonId) {
            R.id.radioMobilenetV2 -> ModelType.MOBILENET_V2
            R.id.radioMobilenetV2Quant -> ModelType.MOBILENET_V2_QUANTIZED
            R.id.radioYolov8n -> ModelType.YOLOV8N
            R.id.radioYolov8nQuant -> ModelType.YOLOV8N_QUANTIZED
            else -> ModelType.MOBILENET_V2
        }

        val delegateMode = when (binding.radioGroupPath.checkedRadioButtonId) {
            R.id.radioNpuGpuCpu -> DelegateMode.NPU_GPU_CPU
            R.id.radioGpuCpu -> DelegateMode.GPU_CPU
            R.id.radioCpuOnly -> DelegateMode.CPU_ONLY
            else -> DelegateMode.NPU_GPU_CPU
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
            delegateMode = delegateMode,
            frequencyHz = frequencyHz,
            warmUpEnabled = binding.checkWarmup.isChecked,
            durationMinutes = durationMinutes
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
        tfliteRunner.setForeground(true)
        Log.d(TAG, "App resumed (foreground)")
    }

    override fun onPause() {
        super.onPause()
        isForeground = false
        tfliteRunner.setForeground(false)
        Log.d(TAG, "App paused (background)")
    }

    override fun onDestroy() {
        super.onDestroy()
        benchmarkRunner.release()
    }
}
