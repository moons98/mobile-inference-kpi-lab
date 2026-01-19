#include "device_info.h"
#include <android/log.h>
#include <fstream>
#include <sstream>
#include <sys/system_properties.h>
#include <mutex>

#define LOG_TAG "DeviceInfo"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace kpilab {

namespace {
    // SoC model to HTP architecture mapping
    // Reference: Qualcomm documentation
    const std::map<std::string, std::pair<HtpArch, std::string>> SOC_MAP = {
        // Snapdragon 8 Gen 3
        {"SM8650",  {HtpArch::V75, "Snapdragon 8 Gen 3"}},
        {"SM8675",  {HtpArch::V75, "Snapdragon 8s Gen 3"}},

        // Snapdragon 8 Gen 2
        {"SM8550",  {HtpArch::V73, "Snapdragon 8 Gen 2"}},
        {"SM8475",  {HtpArch::V73, "Snapdragon 8+ Gen 1"}},  // Some variants

        // Snapdragon 8 Gen 1
        {"SM8450",  {HtpArch::V69, "Snapdragon 8 Gen 1"}},
        {"SM8475",  {HtpArch::V69, "Snapdragon 8+ Gen 1"}},

        // Snapdragon 888
        {"SM8350",  {HtpArch::V68, "Snapdragon 888"}},
        {"SM8325",  {HtpArch::V68, "Snapdragon 888+"}},

        // Snapdragon 865
        {"SM8250",  {HtpArch::V66, "Snapdragon 865"}},
        {"SM8250-AB", {HtpArch::V66, "Snapdragon 865+"}},

        // Snapdragon 7 series (limited HTP)
        {"SM7550",  {HtpArch::V73, "Snapdragon 7+ Gen 2"}},
        {"SM7475",  {HtpArch::V69, "Snapdragon 7+ Gen 1"}},
        {"SM7450",  {HtpArch::V69, "Snapdragon 7 Gen 1"}},
    };

    std::once_flag g_detectOnce;
    DeviceInfo g_deviceInfo;
}

std::string DeviceInfo::getHtpLibName() const {
    switch (htpArch) {
        case HtpArch::V81: return "libQnnHtpV81Stub.so";
        case HtpArch::V79: return "libQnnHtpV79Stub.so";
        case HtpArch::V75: return "libQnnHtpV75Stub.so";
        case HtpArch::V73: return "libQnnHtpV73Stub.so";
        case HtpArch::V69: return "libQnnHtpV69Stub.so";
        case HtpArch::V68: return "libQnnHtpV68Stub.so";
        case HtpArch::V66: return "libQnnHtpV66Stub.so";
        default: return "";
    }
}

std::string DeviceInfo::getSkeletonLibName() const {
    switch (htpArch) {
        case HtpArch::V81: return "libQnnHtpV81Skel.so";
        case HtpArch::V79: return "libQnnHtpV79Skel.so";
        case HtpArch::V75: return "libQnnHtpV75Skel.so";
        case HtpArch::V73: return "libQnnHtpV73Skel.so";
        case HtpArch::V69: return "libQnnHtpV69Skel.so";
        case HtpArch::V68: return "libQnnHtpV68Skel.so";
        case HtpArch::V66: return "libQnnHtpV66Skel.so";
        default: return "";
    }
}

std::string DeviceDetector::readSysFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "";
    }

    std::string content;
    std::getline(file, content);
    return content;
}

std::string DeviceDetector::getSystemProperty(const std::string& prop) {
    char value[PROP_VALUE_MAX] = {0};
    __system_property_get(prop.c_str(), value);
    return std::string(value);
}

DeviceInfo DeviceDetector::detectDevice() {
    DeviceInfo info;

    // Get Android version info
    info.androidVersion = getSystemProperty("ro.build.version.release");
    std::string sdkStr = getSystemProperty("ro.build.version.sdk");
    info.sdkVersion = sdkStr.empty() ? 0 : std::stoi(sdkStr);

    // Try to get SoC model from multiple sources
    // Method 1: System property (most reliable)
    info.socModel = getSystemProperty("ro.soc.model");

    // Method 2: Hardware property
    if (info.socModel.empty()) {
        info.socModel = getSystemProperty("ro.hardware");
    }

    // Method 3: Read from sysfs
    if (info.socModel.empty()) {
        info.socModel = readSysFile("/sys/devices/soc0/soc_id");
    }

    // Method 4: Chipset property
    if (info.socModel.empty()) {
        std::string chipset = getSystemProperty("ro.hardware.chipname");
        if (!chipset.empty()) {
            info.socModel = chipset;
        }
    }

    // Look up HTP architecture
    auto it = SOC_MAP.find(info.socModel);
    if (it != SOC_MAP.end()) {
        info.htpArch = it->second.first;
        info.socName = it->second.second;
        info.htpSupported = true;
    } else {
        // Try partial match (e.g., "SM8550" in "SM8550-AB")
        for (const auto& entry : SOC_MAP) {
            if (info.socModel.find(entry.first) != std::string::npos ||
                entry.first.find(info.socModel) != std::string::npos) {
                info.htpArch = entry.second.first;
                info.socName = entry.second.second;
                info.htpSupported = true;
                break;
            }
        }

        if (!info.htpSupported) {
            info.htpArch = HtpArch::UNKNOWN;
            info.socName = "Unknown";
            info.htpSupported = false;
            LOGW("Unknown SoC model: %s - HTP may not be available", info.socModel.c_str());
        }
    }

    LOGI("Device detected:");
    LOGI("  SoC Model: %s", info.socModel.c_str());
    LOGI("  SoC Name: %s", info.socName.c_str());
    LOGI("  HTP Arch: v%d", static_cast<int>(info.htpArch));
    LOGI("  HTP Supported: %s", info.htpSupported ? "yes" : "no");
    LOGI("  Android: %s (SDK %d)", info.androidVersion.c_str(), info.sdkVersion);

    return info;
}

const DeviceInfo& DeviceDetector::getDeviceInfo() {
    std::call_once(g_detectOnce, []() {
        g_deviceInfo = detectDevice();
    });
    return g_deviceInfo;
}

bool DeviceDetector::isHtpSupported() {
    return getDeviceInfo().htpSupported;
}

HtpArch DeviceDetector::getHtpArch() {
    return getDeviceInfo().htpArch;
}

std::string DeviceDetector::getDeviceSummary() {
    const auto& info = getDeviceInfo();
    std::ostringstream oss;
    oss << info.socName << " (" << info.socModel << ")";
    if (info.htpSupported) {
        oss << " - HTP v" << static_cast<int>(info.htpArch);
    } else {
        oss << " - HTP not available";
    }
    return oss.str();
}

} // namespace kpilab
