#include <jni.h>
#include <string>
#include <memory>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include "inference_engine.h"
#include "kpi_logger.h"
#include "device_info.h"

#define LOG_TAG "NativeRunner"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
    std::unique_ptr<kpilab::InferenceEngine> g_engine;
    std::unique_ptr<kpilab::KpiLogger> g_logger;
    bool g_isForeground = true;
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_example_kpilab_NativeRunner_initialize(
        JNIEnv* env,
        jobject /* this */,
        jstring modelPath,
        jint executionPath,
        jboolean warmUpEnabled) {

    const char* modelPathCStr = env->GetStringUTFChars(modelPath, nullptr);

    LOGI("Initializing native runner");
    LOGI("  Model path: %s", modelPathCStr);
    LOGI("  Execution path: %d", executionPath);
    LOGI("  Warm-up: %s", warmUpEnabled ? "true" : "false");

    // Create engine if not exists
    if (!g_engine) {
        g_engine = std::make_unique<kpilab::InferenceEngine>();
    }

    // Create logger if not exists
    if (!g_logger) {
        g_logger = std::make_unique<kpilab::KpiLogger>();
    }

    // Configure
    kpilab::InferenceConfig config;
    config.path = static_cast<kpilab::ExecutionPath>(executionPath);
    config.warmUpEnabled = warmUpEnabled;
    config.warmUpIterations = 10;

    bool success = g_engine->initialize(modelPathCStr, config);

    env->ReleaseStringUTFChars(modelPath, modelPathCStr);

    return success ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_example_kpilab_NativeRunner_startSession(
        JNIEnv* env,
        jobject /* this */,
        jstring sessionId) {

    if (!g_logger) {
        g_logger = std::make_unique<kpilab::KpiLogger>();
    }

    const char* sessionIdCStr = env->GetStringUTFChars(sessionId, nullptr);
    g_logger->startSession(sessionIdCStr);
    env->ReleaseStringUTFChars(sessionId, sessionIdCStr);
}

JNIEXPORT jfloat JNICALL
Java_com_example_kpilab_NativeRunner_runInference(
        JNIEnv* /* env */,
        jobject /* this */) {

    if (!g_engine || !g_engine->isInitialized()) {
        LOGE("Engine not initialized");
        return -1.0f;
    }

    auto result = g_engine->runInference();

    if (result.success && g_logger) {
        g_logger->logInference(result.latencyMs, g_isForeground);
    }

    return result.success ? result.latencyMs : -1.0f;
}

JNIEXPORT void JNICALL
Java_com_example_kpilab_NativeRunner_logSystemMetrics(
        JNIEnv* /* env */,
        jobject /* this */,
        jfloat thermalC,
        jfloat powerMw,
        jint memoryMb) {

    if (g_logger) {
        g_logger->logSystem(thermalC, powerMw, memoryMb, g_isForeground);
    }
}

JNIEXPORT void JNICALL
Java_com_example_kpilab_NativeRunner_setForeground(
        JNIEnv* /* env */,
        jobject /* this */,
        jboolean isForeground) {

    g_isForeground = isForeground;
    LOGI("Foreground state changed: %s", isForeground ? "true" : "false");
}

JNIEXPORT void JNICALL
Java_com_example_kpilab_NativeRunner_endSession(
        JNIEnv* /* env */,
        jobject /* this */) {

    if (g_logger) {
        g_logger->endSession();
    }
}

JNIEXPORT jstring JNICALL
Java_com_example_kpilab_NativeRunner_exportCsv(
        JNIEnv* env,
        jobject /* this */) {

    if (!g_logger) {
        return env->NewStringUTF("");
    }

    std::string csv = g_logger->exportToCsv();
    return env->NewStringUTF(csv.c_str());
}

JNIEXPORT jint JNICALL
Java_com_example_kpilab_NativeRunner_getRecordCount(
        JNIEnv* /* env */,
        jobject /* this */) {

    if (!g_logger) {
        return 0;
    }

    return static_cast<jint>(g_logger->getRecordCount());
}

JNIEXPORT void JNICALL
Java_com_example_kpilab_NativeRunner_release(
        JNIEnv* /* env */,
        jobject /* this */) {

    LOGI("Releasing native runner");

    if (g_engine) {
        g_engine->release();
        g_engine.reset();
    }

    if (g_logger) {
        g_logger->clear();
        g_logger.reset();
    }
}

JNIEXPORT jstring JNICALL
Java_com_example_kpilab_NativeRunner_getDeviceInfo(
        JNIEnv* env,
        jobject /* this */) {

    std::string summary = kpilab::DeviceDetector::getDeviceSummary();
    return env->NewStringUTF(summary.c_str());
}

JNIEXPORT jboolean JNICALL
Java_com_example_kpilab_NativeRunner_isHtpSupported(
        JNIEnv* /* env */,
        jobject /* this */) {

    return kpilab::DeviceDetector::isHtpSupported() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jint JNICALL
Java_com_example_kpilab_NativeRunner_getHtpVersion(
        JNIEnv* /* env */,
        jobject /* this */) {

    return static_cast<jint>(kpilab::DeviceDetector::getHtpArch());
}

} // extern "C"
