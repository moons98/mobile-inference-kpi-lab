cmake_minimum_required(VERSION 3.22.1)
project("inference_kpi_lab")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QNN SDK path - tries multiple sources:
# 1. CMake cache variable
# 2. Environment variable QNN_SDK_ROOT
# 3. Common installation paths
if(NOT QNN_SDK_ROOT)
    set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}")
endif()

# Try common paths if not set
if(NOT QNN_SDK_ROOT OR NOT EXISTS "${QNN_SDK_ROOT}")
    # Windows common paths
    foreach(QNN_VERSION "2.42.0.251225" "2.41.0" "2.40.0")
        foreach(QNN_BASE "C:/Qualcomm" "D:/Qualcomm" "$ENV{USERPROFILE}/Qualcomm")
            set(QNN_TRY_PATH "${QNN_BASE}/v${QNN_VERSION}/qairt/${QNN_VERSION}")
            if(EXISTS "${QNN_TRY_PATH}/include/QNN")
                set(QNN_SDK_ROOT "${QNN_TRY_PATH}")
                message(STATUS "Found QNN SDK at: ${QNN_SDK_ROOT}")
                break()
            endif()
        endforeach()
        if(QNN_SDK_ROOT AND EXISTS "${QNN_SDK_ROOT}")
            break()
        endif()
    endforeach()
endif()

set(QNN_SDK_ROOT "${QNN_SDK_ROOT}" CACHE PATH "Path to QNN SDK")

# Source files
add_library(${CMAKE_PROJECT_NAME} SHARED
    native_runner.cpp
    inference_engine.cpp
    kpi_logger.cpp
    device_info.cpp
)

# Include directories
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# QNN SDK includes and libraries (if available)
if(EXISTS "${QNN_SDK_ROOT}")
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${QNN_SDK_ROOT}/include
        ${QNN_SDK_ROOT}/include/QNN
    )

    # Link QNN libraries
    target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${QNN_SDK_ROOT}/lib/aarch64-android
    )

    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
        QNN_ENABLED=1
    )
else()
    message(WARNING "QNN SDK not found at ${QNN_SDK_ROOT}. Building with mock implementation.")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
        QNN_ENABLED=0
    )
endif()

# Link Android libraries
find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${log-lib}
    ${android-lib}
)
