#include "kpi_logger.h"
#include <android/log.h>
#include <chrono>
#include <sstream>
#include <iomanip>

#define LOG_TAG "KpiLogger"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace kpilab {

KpiLogger::KpiLogger() {
    records_.reserve(10000);  // Pre-allocate for ~10 min at 10Hz + system events
}

KpiLogger::~KpiLogger() {
    if (sessionActive_) {
        endSession();
    }
}

void KpiLogger::startSession(const std::string& sessionId) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (sessionActive_) {
        LOGI("Previous session still active, ending it first");
    }

    records_.clear();
    sessionId_ = sessionId;
    sessionActive_ = true;

    LOGI("Started logging session: %s", sessionId.c_str());
}

void KpiLogger::logInference(float latencyMs, bool isForeground) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!sessionActive_) {
        return;
    }

    KpiRecord record;
    record.timestamp = getCurrentTimestamp();
    record.eventType = EVENT_INFERENCE;
    record.latencyMs = latencyMs;
    record.isForeground = isForeground;

    records_.push_back(record);
}

void KpiLogger::logSystem(float thermalC, float powerMw, int memoryMb, bool isForeground) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!sessionActive_) {
        return;
    }

    KpiRecord record;
    record.timestamp = getCurrentTimestamp();
    record.eventType = EVENT_SYSTEM;
    record.thermalC = thermalC;
    record.powerMw = powerMw;
    record.memoryMb = memoryMb;
    record.isForeground = isForeground;

    records_.push_back(record);
}

void KpiLogger::endSession() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!sessionActive_) {
        return;
    }

    sessionActive_ = false;
    LOGI("Ended logging session: %s with %zu records",
         sessionId_.c_str(), records_.size());
}

std::vector<KpiRecord> KpiLogger::getRecords() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_;
}

std::string KpiLogger::exportToCsv() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream oss;

    // Header
    oss << "timestamp,event_type,latency_ms,thermal_c,power_mw,memory_mb,is_foreground\n";

    // Data rows
    for (const auto& record : records_) {
        oss << record.timestamp << ",";
        oss << (record.eventType == EVENT_INFERENCE ? "INFERENCE" : "SYSTEM") << ",";

        // Latency (only for INFERENCE)
        if (record.eventType == EVENT_INFERENCE) {
            oss << std::fixed << std::setprecision(2) << record.latencyMs;
        }
        oss << ",";

        // Thermal (only for SYSTEM)
        if (record.eventType == EVENT_SYSTEM && record.thermalC >= 0) {
            oss << std::fixed << std::setprecision(1) << record.thermalC;
        }
        oss << ",";

        // Power (only for SYSTEM)
        if (record.eventType == EVENT_SYSTEM && record.powerMw >= 0) {
            oss << std::fixed << std::setprecision(1) << record.powerMw;
        }
        oss << ",";

        // Memory (only for SYSTEM)
        if (record.eventType == EVENT_SYSTEM && record.memoryMb >= 0) {
            oss << record.memoryMb;
        }
        oss << ",";

        // Foreground
        oss << (record.isForeground ? "true" : "false");
        oss << "\n";
    }

    return oss.str();
}

size_t KpiLogger::getRecordCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_.size();
}

void KpiLogger::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.clear();
    sessionId_.clear();
    sessionActive_ = false;
}

int64_t KpiLogger::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

} // namespace kpilab
