plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.kpilab"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.kpilab"
        minSdk = 31
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        buildConfigField("String", "ORT_VERSION", "\"1.24.3\"")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
        buildConfig = true
    }

    // 모델 파일 압축 방지
    androidResources {
        noCompress += listOf("onnx", "ort", "jpg")
    }

    // Extract native libs so QNN EP can find them by file path
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // JSON parsing for batch mode
    implementation("com.google.code.gson:gson:2.10.1")

    // ONNX Runtime with QNN Execution Provider (NPU)
    // Version 1.24.3 - upgraded for QAI Hub compiled binary compatibility
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.24.3")
}
