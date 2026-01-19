#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <mutex>

namespace kpilab {

struct KpiRecord {
    int64_t timestamp;      // epoch milliseconds
    int eventType;          // 0: INFERENCE, 1: SYSTEM
    float latencyMs;        // only for INFERENCE events
    float thermalC;         // only for SYSTEM events
    float powerMw;          // only for SYSTEM events
    int memoryMb;           // only for SYSTEM events
    bool isForeground;

    KpiRecord()
        : timestamp(0), eventType(0), latencyMs(-1), thermalC(-1),
          powerMw(-1), memoryMb(-1), isForeground(true) {}
};

class KpiLogger {
public:
    static constexpr int EVENT_INFERENCE = 0;
    static constexpr int EVENT_SYSTEM = 1;

    KpiLogger();
    ~KpiLogger();

    /**
     * Start a new logging session
     * @param sessionId Unique identifier for this session
     */
    void startSession(const std::string& sessionId);

    /**
     * Log an inference event
     */
    void logInference(float latencyMs, bool isForeground);

    /**
     * Log system metrics
     */
    void logSystem(float thermalC, float powerMw, int memoryMb, bool isForeground);

    /**
     * End the current session
     */
    void endSession();

    /**
     * Get all records from current session
     */
    std::vector<KpiRecord> getRecords() const;

    /**
     * Export records to CSV format string
     */
    std::string exportToCsv() const;

    /**
     * Get number of records
     */
    size_t getRecordCount() const;

    /**
     * Clear all records
     */
    void clear();

private:
    std::vector<KpiRecord> records_;
    std::string sessionId_;
    mutable std::mutex mutex_;
    bool sessionActive_ = false;

    int64_t getCurrentTimestamp() const;
};

} // namespace kpilab
