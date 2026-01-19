package com.example.kpilab

import android.content.Context
import android.util.Log
import android.util.Pair
import com.example.kpilab.tflite.AIHubDefaults
import com.example.kpilab.tflite.TFLiteHelpers
import com.example.kpilab.tflite.TFLiteHelpers.DelegateType
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.util.HashSet

/**
 * TFLite-based inference runner with QNN/GPU delegate support
 */
class TFLiteRunner(private val context: Context) {

    companion object {
        private const val TAG = "TFLiteRunner"
    }

    private var interpreter: Interpreter? = null
    private var delegates: Map<DelegateType, Delegate>? = null
    private var modelBuffer: MappedByteBuffer? = null
    private var modelHash: String? = null
    private var currentModel: ModelType? = null

    // Input/Output buffers
    private var inputBuffer: ByteBuffer? = null
    private var outputBuffer: ByteBuffer? = null

    // Logging
    private val kpiRecords = mutableListOf<KpiRecord>()
    private var sessionId: String = ""
    private var isForeground: Boolean = true

    data class KpiRecord(
        val timestamp: Long,
        val eventType: Int, // 0: INFERENCE, 1: SYSTEM
        val latencyMs: Float,
        val thermalC: Float,
        val powerMw: Float,
        val memoryMb: Int,
        val isForeground: Boolean
    )

    /**
     * Initialize the TFLite interpreter with specified model and delegate mode
     * @param modelType Which model to use
     * @param delegateMode Which delegates to use
     * @param warmUpEnabled Whether to run warm-up iterations
     * @return true if initialization successful
     */
    fun initialize(modelType: ModelType, delegateMode: DelegateMode, warmUpEnabled: Boolean): Boolean {
        return try {
            currentModel = modelType

            // Load model file
            val modelAndHash = TFLiteHelpers.loadModelFile(context.assets, modelType.filename)
            modelBuffer = modelAndHash.first
            modelHash = modelAndHash.second
            Log.i(TAG, "Model loaded: ${modelType.filename} (hash: $modelHash)")

            // Determine delegate priority order based on mode
            val delegatePriorityOrder = getDelegatePriorityOrder(delegateMode)

            // Create interpreter with delegates
            val result = TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
                modelBuffer,
                delegatePriorityOrder,
                AIHubDefaults.numCPUThreads,
                context.applicationInfo.nativeLibraryDir,
                context.cacheDir.absolutePath,
                modelHash
            )

            interpreter = result.first
            delegates = result.second

            Log.i(TAG, "Interpreter created with delegates: ${delegates?.keys?.map { it.name }}")

            // Allocate input/output buffers
            allocateBuffers()

            // Run warm-up if enabled
            if (warmUpEnabled) {
                runWarmUp()
            }

            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize: ${e.message}", e)
            false
        }
    }

    private fun getDelegatePriorityOrder(delegateMode: DelegateMode): Array<Array<DelegateType>> {
        return when (delegateMode) {
            DelegateMode.NPU_GPU_CPU -> AIHubDefaults.delegatePriorityOrder
            DelegateMode.GPU_CPU -> AIHubDefaults.delegatePriorityOrderForDelegates(
                HashSet(listOf(DelegateType.GPUv2))
            )
            DelegateMode.CPU_ONLY -> AIHubDefaults.delegatePriorityOrderForDelegates(
                HashSet()
            )
        }
    }

    private fun allocateBuffers() {
        val model = currentModel ?: return

        // Input buffer based on model specs
        inputBuffer = ByteBuffer.allocateDirect(model.inputBufferSize).apply {
            order(ByteOrder.nativeOrder())
        }

        // Output buffer based on model specs
        outputBuffer = ByteBuffer.allocateDirect(model.outputBufferSize).apply {
            order(ByteOrder.nativeOrder())
        }

        // Fill input with dummy data (random values for benchmarking)
        val inputElements = model.inputWidth * model.inputHeight * model.inputChannels
        inputBuffer?.let { buffer ->
            buffer.rewind()
            when (model.dataType) {
                DataType.FLOAT32 -> {
                    // Float: normalized 0.0 ~ 1.0
                    for (i in 0 until inputElements) {
                        buffer.putFloat((Math.random() * 255).toFloat() / 255.0f)
                    }
                }
                DataType.INT8, DataType.UINT8 -> {
                    // Quantized: 0 ~ 255 as bytes
                    for (i in 0 until inputElements) {
                        buffer.put((Math.random() * 255).toInt().toByte())
                    }
                }
            }
        }

        val typeStr = if (model.isQuantized) "quantized (${model.dataType})" else "float32"
        Log.i(TAG, "Buffers allocated - Input: ${model.inputWidth}x${model.inputHeight}x${model.inputChannels} ($typeStr), Output: ${model.outputShape.contentToString()}")
    }

    private fun runWarmUp(iterations: Int = 10) {
        Log.i(TAG, "Running $iterations warm-up iterations...")
        for (i in 0 until iterations) {
            runInferenceInternal()
        }
        Log.i(TAG, "Warm-up completed")
    }

    /**
     * Start a new logging session
     */
    fun startSession(sessionId: String) {
        this.sessionId = sessionId
        kpiRecords.clear()
        Log.i(TAG, "Session started: $sessionId")
    }

    /**
     * Run a single inference and return latency
     * @return Latency in milliseconds, or -1 if failed
     */
    fun runInference(): Float {
        val latencyMs = runInferenceInternal()

        if (latencyMs >= 0) {
            // Log inference event
            kpiRecords.add(
                KpiRecord(
                    timestamp = System.currentTimeMillis(),
                    eventType = 0, // INFERENCE
                    latencyMs = latencyMs,
                    thermalC = 0f,
                    powerMw = 0f,
                    memoryMb = 0,
                    isForeground = isForeground
                )
            )
        }

        return latencyMs
    }

    private fun runInferenceInternal(): Float {
        val interp = interpreter ?: return -1f

        return try {
            // Reset buffers
            inputBuffer?.rewind()
            outputBuffer?.rewind()

            // Run inference
            interp.run(inputBuffer, outputBuffer)

            // Get latency from interpreter (nanoseconds -> milliseconds)
            val latencyNs = interp.lastNativeInferenceDurationNanoseconds
            (latencyNs / 1_000_000.0).toFloat()
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed: ${e.message}")
            -1f
        }
    }

    /**
     * Log system metrics
     */
    fun logSystemMetrics(thermalC: Float, powerMw: Float, memoryMb: Int) {
        kpiRecords.add(
            KpiRecord(
                timestamp = System.currentTimeMillis(),
                eventType = 1, // SYSTEM
                latencyMs = 0f,
                thermalC = thermalC,
                powerMw = powerMw,
                memoryMb = memoryMb,
                isForeground = isForeground
            )
        )
    }

    /**
     * Update foreground/background state
     */
    fun setForeground(isForeground: Boolean) {
        this.isForeground = isForeground
    }

    /**
     * End the current logging session
     */
    fun endSession() {
        Log.i(TAG, "Session ended: $sessionId, records: ${kpiRecords.size}")
    }

    /**
     * Export all logged data as CSV string with device info header
     */
    fun exportCsv(): String {
        val sb = StringBuilder()

        // Device info header (commented lines)
        val deviceInfo = KpiCollector.getDeviceInfoMap()
        sb.appendLine("# device_manufacturer,${deviceInfo["device_manufacturer"]}")
        sb.appendLine("# device_model,${deviceInfo["device_model"]}")
        sb.appendLine("# soc_manufacturer,${deviceInfo["soc_manufacturer"]}")
        sb.appendLine("# soc_model,${deviceInfo["soc_model"]}")
        sb.appendLine("# android_version,${deviceInfo["android_version"]}")
        sb.appendLine("# api_level,${deviceInfo["api_level"]}")
        sb.appendLine("# model,${currentModel?.displayName ?: "Unknown"}")
        sb.appendLine("# delegates,${delegates?.keys?.joinToString(";") { it.name } ?: "None"}")
        sb.appendLine("# session_id,$sessionId")
        sb.appendLine("#")

        // CSV header
        sb.appendLine("timestamp,event_type,latency_ms,thermal_c,power_mw,memory_mb,is_foreground")

        for (record in kpiRecords) {
            val eventType = if (record.eventType == 0) "INFERENCE" else "SYSTEM"
            val latency = if (record.eventType == 0) "%.2f".format(record.latencyMs) else ""
            val thermal = if (record.eventType == 1) "%.1f".format(record.thermalC) else ""
            val power = if (record.eventType == 1) "%.1f".format(record.powerMw) else ""
            val memory = if (record.eventType == 1) record.memoryMb.toString() else ""

            sb.appendLine("${record.timestamp},$eventType,$latency,$thermal,$power,$memory,${record.isForeground}")
        }

        return sb.toString()
    }

    /**
     * Get number of records logged
     */
    fun getRecordCount(): Int = kpiRecords.size

    /**
     * Get device info summary
     */
    fun getDeviceInfo(): String {
        val modelName = currentModel?.displayName ?: "None"
        val delegateNames = delegates?.keys?.joinToString(", ") { it.name } ?: "None"
        return "Model: $modelName | Delegates: $delegateNames | CPU Threads: ${AIHubDefaults.numCPUThreads}"
    }

    /**
     * Check if NPU (QNN) delegate is active
     */
    fun isNpuActive(): Boolean {
        return delegates?.containsKey(DelegateType.QNN_NPU) == true
    }

    /**
     * Release all resources
     */
    fun release() {
        interpreter?.close()
        interpreter = null

        delegates?.values?.forEach { it.close() }
        delegates = null

        modelBuffer = null
        inputBuffer = null
        outputBuffer = null
        currentModel = null

        Log.i(TAG, "Resources released")
    }
}

/**
 * Delegate mode for benchmark configuration
 */
enum class DelegateMode(val displayName: String) {
    NPU_GPU_CPU("NPU + GPU + CPU"),
    GPU_CPU("GPU + CPU"),
    CPU_ONLY("CPU Only")
}
