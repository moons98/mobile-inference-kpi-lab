// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.example.kpilab.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;
import android.util.Pair;

import com.qualcomm.qti.QnnDelegate;

import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

public class TFLiteHelpers {
    private static final String TAG = "TFLiteHelpers";

    public enum DelegateType {
        // GPUv2 Delegate: https://www.tensorflow.org/lite/performance/gpu
        GPUv2,

        // QNN Delegate (NPU): Qualcomm Neural Processing Unit
        // Applicable only on Qualcomm chipsets that support the QNN SDK.
        // Floating point compute is supported only on Snapdragon 8 Gen 1 and newer.
        QNN_NPU
    }

    /**
     * Create a TFLite interpreter from the given model.
     *
     * @param tfLiteModel           The model to load.
     * @param delegatePriorityOrder Delegates, in order they should be registered to the interpreter.
     * @param numCPUThreads         Number of CPU threads to use for layers on CPU.
     * @param nativeLibraryDir      Android.Context.nativeLibraryDir (native library directory location)
     * @param cacheDir              Android app cache directory.
     * @param modelIdentifier       Unique identifier string for the model being loaded.
     *
     * @return A pair of the created interpreter and associated delegates.
     */
    public static Pair<Interpreter, Map<DelegateType, Delegate>> CreateInterpreterAndDelegatesFromOptions(
            MappedByteBuffer tfLiteModel,
            DelegateType[][] delegatePriorityOrder,
            int numCPUThreads,
            String nativeLibraryDir,
            String cacheDir,
            String modelIdentifier) {

        Map<DelegateType, Delegate> delegates = new HashMap<>();
        Set<DelegateType> attemptedDelegates = new HashSet<>();

        for (DelegateType[] delegatesToRegister : delegatePriorityOrder) {
            Arrays.stream(delegatesToRegister)
                    .filter(delegateType -> !attemptedDelegates.contains(delegateType))
                    .forEach(delegateType -> {
                        Delegate delegate = CreateDelegate(delegateType, nativeLibraryDir, cacheDir, modelIdentifier);
                        if (delegate != null) {
                            delegates.put(delegateType, delegate);
                        }
                        attemptedDelegates.add(delegateType);
                    });

            if (Arrays.stream(delegatesToRegister).anyMatch(x -> !delegates.containsKey(x))) {
                continue;
            }

            Interpreter interpreter = CreateInterpreterFromDelegates(
                Arrays.stream(delegatesToRegister).map(
                        delegateType -> new Pair<>(delegateType, delegates.get(delegateType))
                ).toArray(Pair[]::new),
                numCPUThreads,
                tfLiteModel
            );

            if (interpreter == null) {
                continue;
            }

            delegates.keySet().stream()
                    .filter(delegateType -> Arrays.stream(delegatesToRegister).noneMatch(d -> d == delegateType))
                    .collect(Collectors.toSet())
                    .forEach(unusedDelegateType -> {
                        Objects.requireNonNull(delegates.remove(unusedDelegateType)).close();
                    });

            return new Pair<>(interpreter, delegates);
        }

        throw new RuntimeException("Unable to create an interpreter of any kind for the provided model. See log for details.");
    }

    /**
     * Create an interpreter from the given delegates.
     */
    public static Interpreter CreateInterpreterFromDelegates(
            final Pair<DelegateType, Delegate>[] delegates,
            int numCPUThreads,
            MappedByteBuffer tfLiteModel) {
        Interpreter.Options tfLiteOptions = new Interpreter.Options();
        tfLiteOptions.setRuntime(Interpreter.Options.TfLiteRuntime.FROM_APPLICATION_ONLY);
        tfLiteOptions.setAllowBufferHandleOutput(true);
        tfLiteOptions.setUseNNAPI(false);
        tfLiteOptions.setNumThreads(numCPUThreads);
        tfLiteOptions.setUseXNNPACK(true);

        Arrays.stream(delegates).forEach(x -> tfLiteOptions.addDelegate(x.second));

        try {
            Interpreter i = new Interpreter(tfLiteModel, tfLiteOptions);
            i.allocateTensors();
            return i;
        } catch (Exception e) {
            List<String> enabledDelegates = Arrays.stream(delegates).map(x -> x.first.name()).collect(Collectors.toCollection(ArrayList<String>::new));
            enabledDelegates.add("XNNPack");
            Log.e(TAG, "Failed to Load Interpreter with delegates {" + String.join(", ", enabledDelegates) + "} | " + e.getMessage());
            return null;
        }
    }

    /**
     * Load a TF Lite model from disk.
     */
    public static Pair<MappedByteBuffer, String> loadModelFile(AssetManager assets, String modelFilename)
            throws IOException, NoSuchAlgorithmException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        MappedByteBuffer buffer;
        String hash;

        try (FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();

            buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

            MessageDigest hashDigest = MessageDigest.getInstance("MD5");
            inputStream.skip(startOffset);
            try (DigestInputStream dis = new DigestInputStream(inputStream, hashDigest)) {
                byte[] data = new byte[8192];
                int numRead = 0;
                while (numRead < declaredLength) {
                    numRead += dis.read(data, 0, Math.min(8192, (int)declaredLength - numRead));
                }
            }

            StringBuilder hex = new StringBuilder();
            for (byte b : hashDigest.digest()) {
                hex.append(String.format("%02x", b));
            }
            hash = hex.toString();
        }

        return new Pair<>(buffer, hash);
    }

    /**
     * Create a delegate of the specified type.
     */
    static Delegate CreateDelegate(DelegateType delegateType, String nativeLibraryDir, String cacheDir, String modelIdentifier) {
        if (delegateType == DelegateType.GPUv2) {
            return CreateGPUv2Delegate(cacheDir, modelIdentifier);
        }
        if (delegateType == DelegateType.QNN_NPU) {
            return CreateQNN_NPUDelegate(nativeLibraryDir, cacheDir, modelIdentifier);
        }

        throw new RuntimeException("Delegate creation not implemented for type: " + delegateType.name());
    }

    /**
     * Create and configure the QNN NPU delegate.
     */
    static Delegate CreateQNN_NPUDelegate(String nativeLibraryDir, String cacheDir, String modelIdentifier) {
        QnnDelegate.Options qnnOptions = new QnnDelegate.Options();
        qnnOptions.setSkelLibraryDir(nativeLibraryDir);
        qnnOptions.setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_WARN);
        qnnOptions.setCacheDir(cacheDir);
        qnnOptions.setModelToken(modelIdentifier);

        if (QnnDelegate.checkCapability(QnnDelegate.Capability.DSP_RUNTIME)) {
            qnnOptions.setBackendType(QnnDelegate.Options.BackendType.DSP_BACKEND);
            qnnOptions.setDspOptions(QnnDelegate.Options.DspPerformanceMode.DSP_PERFORMANCE_BURST, QnnDelegate.Options.DspPdSession.DSP_PD_SESSION_ADAPTIVE);
        } else {
            boolean hasHTP_FP16 = QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16);
            boolean hasHTP_QUANT = QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_QUANTIZED);

            if (!hasHTP_FP16 && !hasHTP_QUANT) {
                Log.e(TAG, "QNN with NPU backend is not supported on this device.");
                return null;
            }

            qnnOptions.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND);
            qnnOptions.setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON);
            qnnOptions.setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST);

            if (hasHTP_FP16) {
                qnnOptions.setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16);
            }
        }

        try {
            return new QnnDelegate(qnnOptions);
        } catch (Exception e) {
            Log.e(TAG, "QNN with NPU backend failed to initialize: " + e.getMessage());
            return null;
        }
    }

    /**
     * Create and configure the GPUv2 delegate.
     */
    static Delegate CreateGPUv2Delegate(String cacheDir, String modelIdentifier) {
        GpuDelegateFactory.Options gpuOptions = new GpuDelegateFactory.Options();
        gpuOptions.setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
        gpuOptions.setPrecisionLossAllowed(true);
        gpuOptions.setSerializationParams(cacheDir, modelIdentifier);

        try {
            return new GpuDelegate(gpuOptions);
        } catch (Exception e) {
            Log.e(TAG, "GPUv2 delegate failed to initialize: " + e.getMessage());
            return null;
        }
    }
}
