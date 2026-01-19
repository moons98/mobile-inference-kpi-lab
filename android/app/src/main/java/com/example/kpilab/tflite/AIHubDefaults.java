// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.example.kpilab.tflite;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class AIHubDefaults {
    // Delegates enabled to replicate AI Hub's defaults on Qualcomm devices.
    public static final Set<TFLiteHelpers.DelegateType> enabledDelegates = new HashSet<>(Arrays.asList(
            TFLiteHelpers.DelegateType.QNN_NPU,
            TFLiteHelpers.DelegateType.GPUv2
    ));

    // Number of threads AI Hub uses by default for layers running on CPU.
    public static final int numCPUThreads = Runtime.getRuntime().availableProcessors() / 2;

    // The default delegate registry order for AI Hub.
    public static final TFLiteHelpers.DelegateType[][] delegatePriorityOrder = new TFLiteHelpers.DelegateType[][] {
            // 1. QNN_NPU + GPUv2 + XNNPack (Best performance on Qualcomm devices)
            { TFLiteHelpers.DelegateType.QNN_NPU, TFLiteHelpers.DelegateType.GPUv2 },

            // 2. GPUv2 + XNNPack (Fallback if NPU unavailable)
            { TFLiteHelpers.DelegateType.GPUv2 },

            // 3. XNNPack only (CPU-only fallback)
            { }
    };

    // Create a version of the above delegate priority order that can only use the provided delegates.
    public static TFLiteHelpers.DelegateType[][] delegatePriorityOrderForDelegates(Set<TFLiteHelpers.DelegateType> enabledDelegates) {
        return Arrays.stream(delegatePriorityOrder).filter(x -> enabledDelegates.containsAll(Arrays.asList(x))).toArray(TFLiteHelpers.DelegateType[][]::new);
    }
}
