package com.example.kpilab

import java.io.File

/**
 * Scans the on-device model directory and reports which (component, precision) combinations
 * have their required model files present.
 *
 * Supports two W8A8 naming conventions:
 *  - _int8_qdq.onnx  (push_models_to_device.sh --int8 convention)
 *  - _w8a8.onnx      (QAI Hub deploy pipeline convention)
 */
object ModelScanner {

    data class Result(
        /** Available precisions per component, for the scanned variant. */
        val componentPrecisions: Map<SdComponent, List<SdPrecision>>,
        /** Model variants whose UNet is present in any precision. */
        val availableVariants: Set<ModelVariant>
    ) {
        fun precisionsFor(comp: SdComponent): List<SdPrecision> =
            componentPrecisions[comp] ?: emptyList()

        val isEmpty: Boolean get() = componentPrecisions.values.all { it.isEmpty() }

        companion object {
            val EMPTY = Result(
                componentPrecisions = SdComponent.values().associateWith { emptyList() },
                availableVariants = emptySet()
            )
        }
    }

    /**
     * Scans [modelDir] for model files and returns availability for [variant].
     * Runs on the calling thread — invoke from a background coroutine.
     */
    fun scan(modelDir: String, variant: ModelVariant): Result {
        val dir = File(modelDir)
        if (!dir.exists() || !dir.isDirectory) return Result.EMPTY

        val files = dir.list()?.toHashSet() ?: return Result.EMPTY

        val componentPrecisions = SdComponent.values().associateWith { comp ->
            val baseName = if (comp == SdComponent.UNET) variant.unetPrefix else comp.baseName
            SdPrecision.values().filter { prec ->
                isAvailable(comp, prec, baseName, files)
            }
        }

        val availableVariants = ModelVariant.values().filter { v ->
            val unetBase = v.unetPrefix
            SdPrecision.values().any { prec ->
                isAvailable(SdComponent.UNET, prec, unetBase, files)
            }
        }.toSet()

        return Result(componentPrecisions, availableVariants)
    }

    private fun isAvailable(
        comp: SdComponent,
        prec: SdPrecision,
        baseName: String,
        files: HashSet<String>
    ): Boolean {
        if (comp.filename(prec, baseName) in files) return true
        return comp.altFilenames(prec, baseName).any { it in files }
    }
}
