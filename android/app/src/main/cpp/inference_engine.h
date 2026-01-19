#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace kpilab {

enum class ExecutionPath {
    NPU_ONLY = 0,      // HTP backend, strict mode (no fallback)
    NPU_FALLBACK = 1,  // HTP backend, fallback allowed
    GPU_ONLY = 2       // GPU backend
};

struct InferenceConfig {
    ExecutionPath path = ExecutionPath::NPU_FALLBACK;
    bool warmUpEnabled = false;
    int warmUpIterations = 10;
};

struct InferenceResult {
    float latencyMs = 0.0f;
    bool success = false;
    bool fallbackOccurred = false;
    std::string errorMessage;
};

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    // Disable copy
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    /**
     * Initialize the inference engine with model and configuration
     * @param modelPath Path to the model file (.dlc for QNN)
     * @param config Execution configuration
     * @return true if initialization successful
     */
    bool initialize(const std::string& modelPath, const InferenceConfig& config);

    /**
     * Run a single inference with dummy input
     * @return InferenceResult containing latency and status
     */
    InferenceResult runInference();

    /**
     * Run warm-up iterations
     * @return Number of successful warm-up runs
     */
    int runWarmUp();

    /**
     * Release all resources
     */
    void release();

    /**
     * Check if engine is initialized
     */
    bool isInitialized() const { return initialized_; }

    /**
     * Get current configuration
     */
    const InferenceConfig& getConfig() const { return config_; }

private:
    bool initialized_ = false;
    InferenceConfig config_;
    std::string modelPath_;

    // Input/output dimensions (MobileNetV3-Small default)
    static constexpr int INPUT_HEIGHT = 224;
    static constexpr int INPUT_WIDTH = 224;
    static constexpr int INPUT_CHANNELS = 3;
    static constexpr int OUTPUT_CLASSES = 1000;

    // Dummy input buffer
    std::vector<float> inputBuffer_;
    std::vector<float> outputBuffer_;

#if QNN_ENABLED
    // QNN handles - forward declared
    void* qnnContext_ = nullptr;
    void* qnnGraph_ = nullptr;
    void* qnnBackend_ = nullptr;

    bool initializeQnn();
    void releaseQnn();
    InferenceResult runQnnInference();
#endif

    // Mock implementation for development without QNN SDK
    InferenceResult runMockInference();
    void prepareDummyInput();
};

} // namespace kpilab
