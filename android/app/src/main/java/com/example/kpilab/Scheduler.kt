package com.example.kpilab

import java.nio.FloatBuffer
import kotlin.math.pow

/**
 * Noise scheduler for SD v1.5 text-to-image pipeline.
 *
 * isLcm=false (default): EulerDiscrete — SD v1.5 standard. Input scaled by 1/sqrt(sigma^2+1),
 *   step: x_next = x + (sigma_next - sigma) * noise_pred.
 *
 * isLcm=true: DDIM-style — matches diffusers LCMScheduler. No input scaling.
 *   step: predict x0 from noise, then blend toward next alpha level.
 *   Initial noise is unit Gaussian (no sigma scaling).
 */
class Scheduler(
    private val numTrainTimesteps: Int = 1000,
    betaStart: Float = 0.00085f,
    betaEnd: Float = 0.012f,
    val isLcm: Boolean = false
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
     * Set up the schedule for a given number of denoising steps.
     */
    fun setTimesteps(numSteps: Int): IntArray {
        require(numSteps >= 1) { "numSteps must be >= 1, got $numSteps" }

        // Linearly spaced timesteps from numTrainTimesteps-1 to 0
        timesteps = if (numSteps == 1) {
            intArrayOf(numTrainTimesteps - 1)
        } else {
            IntArray(numSteps) { i ->
                ((numTrainTimesteps - 1).toFloat() * (numSteps - 1 - i) / (numSteps - 1)).toInt()
            }
        }

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
     * Generate pure random noise as starting latent for text-to-image generation.
     *
     * EulerDiscrete: scales by sigma_0 (≈14.6) — noise lives in sigma-scaled space.
     * LCM: unit Gaussian — noise lives in DDIM/alpha-scaled space (x_T ~ N(0,I)).
     */
    fun generateInitialNoise(size: Int, seed: Long): FloatArray {
        if (timesteps.isEmpty()) error("Call setTimesteps() first")
        val noise = generateNoise(size, seed)
        if (!isLcm) {
            val sigma0 = sigmas[0]
            for (i in noise.indices) noise[i] *= sigma0
        }
        return noise
    }

    /**
     * Compute next latent from UNet noise prediction (epsilon).
     *
     * EulerDiscrete: x_next = x + (sigma_next - sigma) * noise_pred
     *
     * LCM (DDIM-style, matches diffusers LCMScheduler):
     *   alpha_t = alphasCumprod[t], sqrt_alpha = sqrt(alpha_t), sqrt_1ma = sqrt(1-alpha_t)
     *   predicted_x0 = (sample - sqrt_1ma * noise_pred) / sqrt_alpha
     *   If not last step: x_next = sqrt(alpha_prev) * predicted_x0 + sqrt(1-alpha_prev) * noise_pred
     *   If last step:     x_next = predicted_x0
     */
    fun step(sample: FloatArray, modelOutput: FloatArray): FloatArray {
        val result = FloatArray(sample.size)

        if (isLcm) {
            val alphaT = alphasCumprod[timesteps[currentStep]]
            val sqrtAlphaT = kotlin.math.sqrt(alphaT)
            val sqrtOneMinusAlphaT = kotlin.math.sqrt(1f - alphaT)
            val isLastStep = (currentStep + 1 >= timesteps.size)

            if (isLastStep) {
                for (i in result.indices) {
                    result[i] = (sample[i] - sqrtOneMinusAlphaT * modelOutput[i]) / sqrtAlphaT
                }
            } else {
                val alphaPrev = alphasCumprod[timesteps[currentStep + 1]]
                val sqrtAlphaPrev = kotlin.math.sqrt(alphaPrev)
                val sqrtOneMinusAlphaPrev = kotlin.math.sqrt(1f - alphaPrev)
                for (i in result.indices) {
                    val predictedX0 = (sample[i] - sqrtOneMinusAlphaT * modelOutput[i]) / sqrtAlphaT
                    result[i] = sqrtAlphaPrev * predictedX0 + sqrtOneMinusAlphaPrev * modelOutput[i]
                }
            }
        } else {
            val sigma = sigmas[currentStep]
            val sigmaNext = sigmas[currentStep + 1]
            val dt = sigmaNext - sigma
            for (i in result.indices) {
                result[i] = sample[i] + dt * modelOutput[i]
            }
        }

        currentStep++
        return result
    }

    /**
     * Scale model input before UNet inference.
     * EulerDiscrete: x / sqrt(sigma^2 + 1)  (sigma-scaling)
     * LCM: no scaling (diffusers LCMScheduler.scale_model_input returns input unchanged)
     */
    fun scaleModelInput(sample: FloatArray): FloatArray {
        if (isLcm) return sample.copyOf()
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
    fun generateNoise(size: Int, seed: Long): FloatArray {
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
