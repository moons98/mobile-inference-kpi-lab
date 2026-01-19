#pragma once

#include <string>
#include <map>

namespace kpilab {

/**
 * HTP (Hexagon Tensor Processor) architecture versions by Snapdragon generation
 */
enum class HtpArch {
    UNKNOWN = 0,
    V66 = 66,   // Snapdragon 865
    V68 = 68,   // Snapdragon 888
    V69 = 69,   // Snapdragon 8 Gen 1
    V73 = 73,   // Snapdragon 8 Gen 2
    V75 = 75,   // Snapdragon 8 Gen 3
    V79 = 79,   // Snapdragon 8s Gen 3 / future
    V81 = 81,   // Future chipsets
};

/**
 * Device/SoC information for runtime QNN configuration
 */
struct DeviceInfo {
    std::string socModel;       // e.g., "SM8550"
    std::string socName;        // e.g., "Snapdragon 8 Gen 2"
    HtpArch htpArch;            // HTP architecture version
    bool htpSupported;          // Whether HTP/NPU is available
    std::string androidVersion;
    int sdkVersion;             // Android SDK version

    std::string getHtpLibName() const;
    std::string getSkeletonLibName() const;
};

/**
 * Detect device information at runtime
 */
class DeviceDetector {
public:
    /**
     * Get device information (cached after first call)
     */
    static const DeviceInfo& getDeviceInfo();

    /**
     * Check if current device supports HTP
     */
    static bool isHtpSupported();

    /**
     * Get recommended HTP architecture for current device
     */
    static HtpArch getHtpArch();

    /**
     * Get human-readable device summary
     */
    static std::string getDeviceSummary();

private:
    static DeviceInfo detectDevice();
    static std::string readSysFile(const std::string& path);
    static std::string getSystemProperty(const std::string& prop);
};

} // namespace kpilab
