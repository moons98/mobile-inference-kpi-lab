package com.example.kpilab

import ai.onnxruntime.*
import android.content.Context
import android.util.JsonReader
import android.util.Log
import java.io.File
import java.io.FileReader
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Single ORT session wrapper for one SD component.
 * InpaintPipeline creates 4 instances (VAE Encoder, Text Encoder, Inpaint UNet, VAE Decoder).
 */
class OrtRunner(private val context: Context) {

    companion object {
        private const val TAG = "OrtRunner"

        /** Initialize QNN libraries. Call once at app startup. */
        fun initializeQnnLibraries(context: Context): Boolean {
            val path = QnnLibraryManager.initialize(context)
            return path != null
        }
    }

    private var ortEnv: OrtEnvironment? = null
    private var ownsOrtEnv: Boolean = false  // Only close if we created it
    private var ortSession: OrtSession? = null
    private var activeExecutionProvider: String = "CPU"

    // I/O metadata
    var inputNames: List<String> = emptyList()
        private set
    private var outputNames: List<String> = emptyList()
    var inputShapes: Map<String, LongArray> = emptyMap()
        private set
    var outputShapes: Map<String, LongArray> = emptyMap()
        private set

    // Cold start timing
    var modelLoadMs: Long = 0
        private set
    var sessionCreateMs: Long = 0
        private set

    // Model file info
    var modelFileSizeKb: Int = 0
        private set

    // ORT profiling
    private var profilingEnabled: Boolean = false
    var profilingSummary: ProfilingSummary? = null
        private set

    // Graph partitioning
    var totalNodes: Int = 0
        private set
    var qnnNodes: Int = 0
        private set
    var cpuNodes: Int = 0
        private set
    var fallbackOps: List<String> = emptyList()
        private set

    // QNN options string for logging
    var qnnOptionsStr: String = ""
        private set

    var lastError: String? = null
        private set

    data class ProfilingSummary(
        val totalRunUs: Long,
        val npuComputeUs: Long,
        val cpuComputeUs: Long,
        val frameworkUs: Long,
        val fenceUs: Long,
        val iterationCount: Int,
        val cpuOpCounts: Map<String, Int> = emptyMap()
    )

    data class RunResult(
        val sessionRunMs: Float,
        val inputCreateMs: Float,
        val outputCopyMs: Float,
        val outputs: Map<String, Any>  // name -> FloatArray
    )

    /**
     * Initialize session from a model file path on device storage.
     */
    fun initialize(
        modelPath: String,
        executionProvider: ExecutionProvider,
        useNpuFp16: Boolean = true,
        useContextCache: Boolean = false,
        htpPerformanceMode: String = "burst",
        enableProfiling: Boolean = false,
        sharedEnv: OrtEnvironment? = null
    ): Boolean {
        lastError = null
        return try {
            Log.i(TAG, "=== Initializing: $modelPath ===")
            Log.i(TAG, "EP: ${executionProvider.displayName}, FP16: $useNpuFp16")

            if (sharedEnv != null) {
                ortEnv = sharedEnv
                ownsOrtEnv = false
            } else {
                ortEnv = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
                ownsOrtEnv = true
            }

            val sessionOptions = OrtSession.SessionOptions()
            configureExecutionProvider(sessionOptions, executionProvider, useNpuFp16,
                useContextCache, htpPerformanceMode, modelPath)

            if (enableProfiling) {
                val profilePrefix = "${context.filesDir.absolutePath}/ort_profile"
                sessionOptions.enableProfiling(profilePrefix)
                profilingEnabled = true
            }

            // Model file info
            val file = File(modelPath)
            modelFileSizeKb = (file.length() / 1024).toInt()
            modelLoadMs = 0  // No separate load step with path-based API
            Log.i(TAG, "Model file: ${modelFileSizeKb}KB")

            // Create session from file path (avoids OOM from readBytes on large models)
            val sessionStart = System.currentTimeMillis()
            ortSession = ortEnv!!.createSession(modelPath, sessionOptions)
            sessionCreateMs = System.currentTimeMillis() - sessionStart
            Log.i(TAG, "Session created in ${sessionCreateMs}ms")

            extractIOInfo()
            true
        } catch (e: Exception) {
            Log.e(TAG, "Init failed: ${e.message}", e)
            lastError = e.message
            false
        }
    }

    /**
     * Run inference with named inputs (FloatBuffer only).
     */
    fun run(inputs: Map<String, Pair<FloatBuffer, LongArray>>): RunResult? {
        // Delegate to mixed-type run
        val mixedInputs = inputs.mapValues { (_, pair) -> Pair(pair.first as Any, pair.second) }
        return runMixed(mixedInputs)
    }

    /**
     * Run inference with mixed-type named inputs.
     * Supports FloatBuffer and LongBuffer (for INT64 tensors like token IDs).
     * @param inputs Map of input name -> (FloatBuffer|LongBuffer, shape)
     */
    fun runMixed(inputs: Map<String, Pair<Any, LongArray>>): RunResult? {
        val env = ortEnv ?: return null
        val session = ortSession ?: return null

        var tensorInputs: Map<String, OnnxTensor> = emptyMap()
        var results: OrtSession.Result? = null
        return try {
            // Create input tensors
            val inputCreateStart = System.nanoTime()
            tensorInputs = inputs.map { (name, pair) ->
                val (buffer, shape) = pair
                val tensor = when (buffer) {
                    is FloatBuffer -> OnnxTensor.createTensor(env, buffer, shape)
                    is LongBuffer -> OnnxTensor.createTensor(env, buffer, shape)
                    else -> throw IllegalArgumentException("Unsupported buffer type: ${buffer::class}")
                }
                name to tensor
            }.toMap()
            val inputCreateMs = ((System.nanoTime() - inputCreateStart) / 1_000_000.0).toFloat()

            // Run session
            val runStart = System.nanoTime()
            results = session.run(tensorInputs)
            val sessionRunMs = ((System.nanoTime() - runStart) / 1_000_000.0).toFloat()

            // Extract outputs
            val outputCopyStart = System.nanoTime()
            val outputMap = mutableMapOf<String, Any>()
            for (i in outputNames.indices) {
                val name = outputNames[i]
                val tensor = results.get(i) as OnnxTensor
                val buf = tensor.floatBuffer
                val arr = FloatArray(buf.remaining())
                buf.get(arr)
                outputMap[name] = arr
            }
            val outputCopyMs = ((System.nanoTime() - outputCopyStart) / 1_000_000.0).toFloat()

            RunResult(sessionRunMs, inputCreateMs, outputCopyMs, outputMap)
        } catch (e: Exception) {
            Log.e(TAG, "Run failed: ${e.message}", e)
            null
        } finally {
            results?.close()
            tensorInputs.values.forEach { it.close() }
        }
    }

    /**
     * Convenience: run with single input.
     */
    fun run(inputData: FloatBuffer, inputShape: LongArray): RunResult? {
        if (inputNames.isEmpty()) return null
        return run(mapOf(inputNames[0] to Pair(inputData, inputShape)))
    }

    fun getActiveExecutionProvider(): String = activeExecutionProvider

    /**
     * Analyze ORT profiling data.
     */
    fun analyzeProfilingData(): ProfilingSummary? {
        if (!profilingEnabled) return null
        val session = ortSession ?: return null

        return try {
            val profilePath = session.endProfiling()
            val file = File(profilePath)
            if (!file.exists()) return null

            var totalRunUs = 0L
            var npuComputeUs = 0L
            var cpuComputeUs = 0L
            var fenceUs = 0L
            var iterationCount = 0
            val cpuOpCounts = mutableMapOf<String, Int>()

            JsonReader(FileReader(file)).use { reader ->
                reader.beginArray()
                while (reader.hasNext()) {
                    var cat = ""; var name = ""; var dur = 0L; var provider = ""; var opName = ""
                    reader.beginObject()
                    while (reader.hasNext()) {
                        when (reader.nextName()) {
                            "cat" -> cat = reader.nextString()
                            "name" -> name = reader.nextString()
                            "dur" -> dur = reader.nextLong()
                            "args" -> {
                                reader.beginObject()
                                while (reader.hasNext()) {
                                    when (reader.nextName()) {
                                        "provider" -> provider = reader.nextString()
                                        "op_name" -> opName = reader.nextString()
                                        else -> reader.skipValue()
                                    }
                                }
                                reader.endObject()
                            }
                            else -> reader.skipValue()
                        }
                    }
                    reader.endObject()

                    when (cat) {
                        "Session" -> if (name == "model_run") { totalRunUs += dur; iterationCount++ }
                        "Node" -> {
                            if (name.endsWith("_fence_before") || name.endsWith("_fence_after")) {
                                fenceUs += dur
                            } else when {
                                provider.contains("QNN") -> npuComputeUs += dur
                                provider.contains("CPU") -> {
                                    cpuComputeUs += dur
                                    if (opName.isNotEmpty()) cpuOpCounts[opName] = (cpuOpCounts[opName] ?: 0) + 1
                                }
                            }
                        }
                    }
                }
                reader.endArray()
            }

            file.delete()
            if (iterationCount == 0) return null

            val summary = ProfilingSummary(
                totalRunUs, npuComputeUs, cpuComputeUs,
                totalRunUs - npuComputeUs - cpuComputeUs - fenceUs,
                fenceUs, iterationCount,
                cpuOpCounts.mapValues { it.value / iterationCount }
            )
            profilingSummary = summary
            summary
        } catch (e: Throwable) {
            Log.e(TAG, "Profiling analysis failed: ${e.message}", e)
            null
        }
    }

    fun release() {
        ortSession?.close()
        ortSession = null
        if (ownsOrtEnv) {
            ortEnv?.close()
        }
        ortEnv = null
        profilingEnabled = false
        profilingSummary = null
        Log.i(TAG, "Released")
    }

    private fun configureExecutionProvider(
        options: OrtSession.SessionOptions,
        provider: ExecutionProvider,
        useNpuFp16: Boolean,
        useContextCache: Boolean,
        htpPerformanceMode: String,
        modelPath: String
    ) {
        when (provider) {
            ExecutionProvider.QNN_NPU -> {
                try {
                    val qnnOptions = mutableMapOf<String, String>()
                    val qnnLibPath = QnnLibraryManager.getLibraryPath()
                    if (qnnLibPath != null) {
                        qnnOptions["backend_path"] = "$qnnLibPath/libQnnHtp.so"
                        qnnOptions["skel_library_dir"] = qnnLibPath
                    } else {
                        qnnOptions["backend_path"] = "libQnnHtp.so"
                    }
                    qnnOptions["htp_performance_mode"] = htpPerformanceMode
                    qnnOptions["htp_graph_finalization_optimization_mode"] = "3"
                    qnnOptions["enable_htp_fp16_precision"] = if (useNpuFp16) "1" else "0"

                    if (useContextCache) {
                        val cacheDir = context.cacheDir
                        val modelName = File(modelPath).nameWithoutExtension
                        val precStr = if (useNpuFp16) "fp16" else "fp32"
                        qnnOptions["qnn_context_cache_enable"] = "1"
                        qnnOptions["qnn_context_cache_path"] = "${cacheDir.absolutePath}/qnn_${modelName}_${precStr}.bin"
                    }

                    val libSource = if (qnnLibPath != null) "custom" else "bundled"
                    qnnOptionsStr = "backend=HTP;perf=$htpPerformanceMode;fp16=${if (useNpuFp16) "1" else "0"};cache=${if (useContextCache) "1" else "0"};libs=$libSource"

                    options.addQnn(qnnOptions)
                    activeExecutionProvider = "QNN_NPU"
                } catch (e: Exception) {
                    Log.e(TAG, "QNN NPU failed: ${e.message}, falling back to CPU")
                    activeExecutionProvider = "CPU"
                    qnnOptionsStr = "fallback_to_cpu"
                }
            }
            ExecutionProvider.QNN_GPU -> {
                try {
                    val qnnOptions = mutableMapOf<String, String>()
                    val qnnLibPath = QnnLibraryManager.getLibraryPath()
                    if (qnnLibPath != null) {
                        qnnOptions["backend_path"] = "$qnnLibPath/libQnnGpu.so"
                    } else {
                        qnnOptions["backend_path"] = "libQnnGpu.so"
                    }
                    qnnOptionsStr = "backend=GPU"
                    options.addQnn(qnnOptions)
                    activeExecutionProvider = "QNN_GPU"
                } catch (e: Exception) {
                    Log.e(TAG, "QNN GPU failed: ${e.message}, falling back to CPU")
                    activeExecutionProvider = "CPU"
                    qnnOptionsStr = "fallback_to_cpu"
                }
            }
            ExecutionProvider.CPU -> {
                activeExecutionProvider = "CPU"
                qnnOptionsStr = "n/a"
            }
        }

        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        options.setIntraOpNumThreads(4)
    }

    private fun extractIOInfo() {
        val session = ortSession ?: return
        inputNames = session.inputInfo.keys.toList()
        outputNames = session.outputInfo.keys.toList()
        inputShapes = session.inputInfo.mapValues { (_, info) ->
            (info.info as? TensorInfo)?.shape ?: longArrayOf()
        }
        outputShapes = session.outputInfo.mapValues { (_, info) ->
            (info.info as? TensorInfo)?.shape ?: longArrayOf()
        }

        Log.i(TAG, "Inputs: ${inputNames.zip(inputShapes.values.map { it.contentToString() })}")
        Log.i(TAG, "Outputs: ${outputNames.zip(outputShapes.values.map { it.contentToString() })}")
    }
}
