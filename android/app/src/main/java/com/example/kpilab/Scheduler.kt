package com.example.kpilab

import java.nio.FloatBuffer
import kotlin.math.pow

/**
 * EulerDiscrete scheduler for SD v1.5 Inpainting.
 * Handles noise schedule, add_noise for inpainting, and per-step denoising.
 */
class Scheduler(
    private val numTrainTimesteps: Int = 1000,
    betaStart: Float = 0.00085f,
    betaEnd: Float = 0.012f
) {
    // Precomputed noise schedule
    private val betas: FloatArray
    private val alphasCumprod: FloatArray

    // Runtime state
    private var timesteps: IntArray = intArrayOf()
    private var sigmas: FloatArray = floatArrayOf()
    private var currentStep: Int = 0

    init {
        // Scaled linear beta schedule (SD default)
        betas = FloatArray(numTrainTimesteps) { i ->
            val t = i.toFloat() / (numTrainTimesteps - 1)
            val beta = betaStart.toDouble().pow(0.5) * (1 - t) + betaEnd.toDouble().pow(0.5) * t
            (beta * beta).toFloat().coerceIn(0.0001f, 0.02f)
        }

        alphasCumprod = FloatArray(numTrainTimesteps)
        var cumprod = 1.0f
        for (i in betas.indices) {
            cumprod *= (1.0f - betas[i])
            alphasCumprod[i] = cumprod
        }
    }

    /**
     * Set up the schedule for a given number of total steps and strength.
     * For img2img, only the last (steps × strength) timesteps are used.
     */
    fun setTimesteps(numSteps: Int, strength: Float): IntArray {
        require(numSteps >= 1) { "numSteps must be >= 1, got $numSteps" }

        // Linearly spaced timesteps from numTrainTimesteps-1 to 0
        val allTimesteps = if (numSteps == 1) {
            intArrayOf(numTrainTimesteps - 1)
        } else {
            IntArray(numSteps) { i ->
                ((numTrainTimesteps - 1).toFloat() * (numSteps - 1 - i) / (numSteps - 1)).toInt()
            }
        }

        // For img2img: skip first steps based on strength
        val startStep = (numSteps - (numSteps * strength).toInt()).coerceIn(0, numSteps - 1)
        timesteps = allTimesteps.sliceArray(startStep until numSteps)

        // Compute sigmas for Euler method
        sigmas = FloatArray(timesteps.size + 1)
        for (i in timesteps.indices) {
            val alphaCP = alphasCumprod[timesteps[i]]
            sigmas[i] = kotlin.math.sqrt((1.0f - alphaCP) / alphaCP)
        }
        sigmas[timesteps.size] = 0.0f  // Terminal sigma

        currentStep = 0
        return timesteps
    }

    /**
     * Get the actual number of denoising steps.
     */
    fun getActualSteps(): Int = timesteps.size

    /**
     * Get current timestep value for UNet input.
     */
    fun getCurrentTimestep(): Int {
        return if (currentStep < timesteps.size) timesteps[currentStep] else 0
    }

    /**
     * Add noise to image latent for img2img initialization.
     * @param imageLatent Clean latent from VAE Encoder [1, 4, H, W]
     * @param seed Random seed for reproducibility
     * @return Noisy latent as starting point for denoising
     */
    fun addNoise(imageLatent: FloatArray, seed: Long): FloatArray {
        if (timesteps.isEmpty()) error("Call setTimesteps() first")

        val t = timesteps[0]
        val alphaCP = alphasCumprod[t]
        val sqrtAlpha = kotlin.math.sqrt(alphaCP)
        val sqrtOneMinusAlpha = kotlin.math.sqrt(1.0f - alphaCP)

        val noise = generateNoise(imageLatent.size, seed)
        val result = FloatArray(imageLatent.size)

        for (i in result.indices) {
            result[i] = sqrtAlpha * imageLatent[i] + sqrtOneMinusAlpha * noise[i]
        }

        return result
    }

    /**
     * Euler step: compute next latent from UNet noise prediction.
     * @param sample Current latent [1, 4, H, W]
     * @param modelOutput UNet predicted noise [1, 4, H, W]
     * @return Next latent (denoised one step)
     */
    fun step(sample: FloatArray, modelOutput: FloatArray): FloatArray {
        val sigma = sigmas[currentStep]
        val sigmaNext = sigmas[currentStep + 1]

        // Euler method: x_next = x + (sigma_next - sigma) * model_output / sigma
        // Simplified: predicted original = sample - sigma * model_output
        // Then: x_next = predicted + sigma_next * (sample - predicted) / sigma
        val dt = sigmaNext - sigma
        val result = FloatArray(sample.size)

        for (i in result.indices) {
            // Euler discrete step
            result[i] = sample[i] + dt * modelOutput[i]
        }

        currentStep++
        return result
    }

    /**
     * Scale model input (Euler scheduler requires scaling by sigma).
     */
    fun scaleModelInput(sample: FloatArray): FloatArray {
        val sigma = sigmas[currentStep]
        val scale = 1.0f / kotlin.math.sqrt(sigma * sigma + 1.0f)
        return FloatArray(sample.size) { sample[it] * scale }
    }

    /**
     * Get timestep as FloatBuffer for UNet input.
     */
    fun getTimestepTensor(): Pair<FloatBuffer, LongArray> {
        val buffer = FloatBuffer.allocate(1)
        buffer.put(0, getCurrentTimestep().toFloat())
        return Pair(buffer, longArrayOf(1))
    }

    /**
     * Generate gaussian noise with a fixed seed.
     * Uses Box-Muller transform for reproducible normal distribution.
     */
    private fun generateNoise(size: Int, seed: Long): FloatArray {
        val random = java.util.Random(seed)
        val noise = FloatArray(size)

        var i = 0
        while (i < size - 1) {
            val u1 = random.nextFloat().coerceIn(1e-7f, 1f)
            val u2 = random.nextFloat()
            val radius = kotlin.math.sqrt(-2.0f * kotlin.math.ln(u1))
            val theta = 2.0f * Math.PI.toFloat() * u2
            noise[i] = radius * kotlin.math.cos(theta)
            noise[i + 1] = radius * kotlin.math.sin(theta)
            i += 2
        }
        if (i < size) {
            val u1 = random.nextFloat().coerceIn(1e-7f, 1f)
            val u2 = random.nextFloat()
            noise[i] = kotlin.math.sqrt(-2.0f * kotlin.math.ln(u1)) * kotlin.math.cos(2.0f * Math.PI.toFloat() * u2)
        }

        return noise
    }
}
