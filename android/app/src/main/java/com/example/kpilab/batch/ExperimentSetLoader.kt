package com.example.kpilab.batch

import android.content.Context
import android.os.Environment
import android.util.Log
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import java.io.File
import java.io.InputStreamReader

/**
 * Loads experiment set configuration from multiple JSON files.
 * Supports pattern: experiment_sets_*.json (e.g., experiment_sets_mobilenet.json)
 * Priority: external storage > assets
 */
class ExperimentSetLoader(private val context: Context) {

    companion object {
        private const val TAG = "ExperimentSetLoader"
        private const val FILE_PREFIX = "experiment_sets_"
        private const val FILE_SUFFIX = ".json"
        // Legacy single file support
        private const val LEGACY_FILENAME = "experiment_sets.json"
    }

    private val gson: Gson = GsonBuilder()
        .setLenient()
        .create()

    private var cachedSets: List<ExperimentSet>? = null
    private var cachedDefaults: ExperimentDefaults? = null

    /**
     * Load all experiment sets from multiple JSON files.
     * Returns cached result if already loaded.
     */
    fun load(): List<ExperimentSet> {
        if (cachedSets != null) {
            return cachedSets!!
        }

        val allSets = mutableListOf<ExperimentSet>()
        var defaults: ExperimentDefaults? = null

        // Try external storage first
        val externalSets = loadAllFromExternal()
        if (externalSets.isNotEmpty()) {
            Log.i(TAG, "Loaded ${externalSets.size} sets from external storage")
            allSets.addAll(externalSets.second)
            defaults = externalSets.first
        }

        // Load from assets (merge with external)
        val assetsSets = loadAllFromAssets()
        if (assetsSets.isNotEmpty()) {
            Log.i(TAG, "Loaded ${assetsSets.size} sets from assets")
            // Only add sets that don't already exist (external has priority)
            val existingIds = allSets.map { it.id }.toSet()
            assetsSets.second.filter { it.id !in existingIds }.forEach { allSets.add(it) }
            if (defaults == null) {
                defaults = assetsSets.first
            }
        }

        cachedSets = allSets
        cachedDefaults = defaults ?: ExperimentDefaults()

        Log.i(TAG, "Total experiment sets loaded: ${allSets.size}")
        return allSets
    }

    /**
     * Force reload configuration from disk.
     */
    fun reload(): List<ExperimentSet> {
        cachedSets = null
        cachedDefaults = null
        return load()
    }

    /**
     * Get list of available experiment sets.
     */
    fun getExperimentSets(): List<ExperimentSet> {
        return load()
    }

    /**
     * Get defaults from configuration.
     */
    fun getDefaults(): ExperimentDefaults {
        load() // Ensure loaded
        return cachedDefaults ?: ExperimentDefaults()
    }

    /**
     * Get experiment set by ID.
     */
    fun getExperimentSetById(id: String): ExperimentSet? {
        return load().find { it.id == id }
    }

    /**
     * Load all experiment sets from external storage.
     * Returns Pair<defaults, sets>
     */
    private fun loadAllFromExternal(): Pair<ExperimentDefaults?, List<ExperimentSet>> {
        val allSets = mutableListOf<ExperimentSet>()
        var defaults: ExperimentDefaults? = null

        try {
            val documentsDir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            if (documentsDir == null || !documentsDir.exists()) {
                return Pair(null, emptyList())
            }

            // Find all matching files
            val files = documentsDir.listFiles { file ->
                file.name.startsWith(FILE_PREFIX) && file.name.endsWith(FILE_SUFFIX)
            } ?: emptyArray()

            // Also check for legacy single file
            val legacyFile = File(documentsDir, LEGACY_FILENAME)
            val allFiles = if (legacyFile.exists() && files.isEmpty()) {
                arrayOf(legacyFile)
            } else {
                files
            }

            for (file in allFiles.sortedBy { it.name }) {
                Log.i(TAG, "Loading from external: ${file.name}")
                try {
                    val config = parseConfig(file.readText())
                    if (config != null) {
                        if (defaults == null) {
                            defaults = config.defaults
                        }
                        allSets.addAll(config.experimentSets)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load ${file.name}: ${e.message}")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load from external: ${e.message}", e)
        }

        return Pair(defaults, allSets)
    }

    /**
     * Load all experiment sets from assets.
     * Returns Pair<defaults, sets>
     */
    private fun loadAllFromAssets(): Pair<ExperimentDefaults?, List<ExperimentSet>> {
        val allSets = mutableListOf<ExperimentSet>()
        var defaults: ExperimentDefaults? = null

        try {
            val assetsList = context.assets.list("") ?: emptyArray()

            // Find all matching files
            val matchingFiles = assetsList.filter {
                it.startsWith(FILE_PREFIX) && it.endsWith(FILE_SUFFIX)
            }.sorted()

            // Also check for legacy single file if no split files found
            val filesToLoad = if (matchingFiles.isEmpty() && LEGACY_FILENAME in assetsList) {
                listOf(LEGACY_FILENAME)
            } else {
                matchingFiles
            }

            for (filename in filesToLoad) {
                Log.i(TAG, "Loading from assets: $filename")
                try {
                    context.assets.open(filename).use { inputStream ->
                        InputStreamReader(inputStream).use { reader ->
                            val config = parseConfig(reader.readText())
                            if (config != null) {
                                if (defaults == null) {
                                    defaults = config.defaults
                                }
                                allSets.addAll(config.experimentSets)
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load $filename: ${e.message}")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load from assets: ${e.message}", e)
        }

        return Pair(defaults, allSets)
    }

    private fun parseConfig(json: String): ExperimentSetConfig? {
        return try {
            val config = gson.fromJson(json, ExperimentSetConfig::class.java)
            logConfig(config)
            config
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse config: ${e.message}", e)
            null
        }
    }

    private fun logConfig(config: ExperimentSetConfig) {
        config.experimentSets.forEach { set ->
            Log.d(TAG, "  Set '${set.id}' (${set.name}): ${set.experiments.size} experiments")
        }
    }
}
