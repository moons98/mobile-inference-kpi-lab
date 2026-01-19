#!/usr/bin/env python3
"""
Parse and process KPI log files from the Android app.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class KpiMetrics:
    """Aggregated KPI metrics from a benchmark run."""
    # Latency metrics
    latency_p50: float
    latency_p95: float
    latency_mean: float
    latency_std: float
    latency_min: float
    latency_max: float
    inference_count: int

    # Thermal metrics
    thermal_start: float
    thermal_end: float
    thermal_max: float
    thermal_slope: float  # °C per minute

    # Power metrics
    power_mean: float
    power_std: float

    # Memory metrics
    memory_peak: int
    memory_mean: float

    # Duration
    duration_seconds: float


def load_log(file_path: str) -> pd.DataFrame:
    """
    Load a KPI log CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame with parsed log data
    """
    df = pd.read_csv(file_path)

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Calculate relative time in seconds
    start_time = df['timestamp'].min()
    df['elapsed_seconds'] = (df['timestamp'] - start_time) / 1000.0

    return df


def split_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split log into inference and system events.

    Args:
        df: Full log DataFrame

    Returns:
        Tuple of (inference_df, system_df)
    """
    inference_df = df[df['event_type'] == 'INFERENCE'].copy()
    system_df = df[df['event_type'] == 'SYSTEM'].copy()

    return inference_df, system_df


def calculate_metrics(df: pd.DataFrame) -> KpiMetrics:
    """
    Calculate aggregated KPI metrics from a log file.

    Args:
        df: Full log DataFrame

    Returns:
        KpiMetrics with calculated values
    """
    inference_df, system_df = split_events(df)

    # Latency metrics
    latencies = inference_df['latency_ms'].dropna()

    # Thermal metrics
    thermals = system_df['thermal_c'].dropna()
    thermal_times = system_df.loc[thermals.index, 'elapsed_seconds']

    # Calculate thermal slope (°C per minute)
    if len(thermals) >= 2:
        # Linear regression
        coeffs = np.polyfit(thermal_times / 60.0, thermals, 1)
        thermal_slope = coeffs[0]
    else:
        thermal_slope = 0.0

    # Power metrics
    powers = system_df['power_mw'].dropna()

    # Memory metrics
    memories = system_df['memory_mb'].dropna()

    # Duration
    duration = df['elapsed_seconds'].max()

    return KpiMetrics(
        # Latency
        latency_p50=latencies.quantile(0.50) if len(latencies) > 0 else 0,
        latency_p95=latencies.quantile(0.95) if len(latencies) > 0 else 0,
        latency_mean=latencies.mean() if len(latencies) > 0 else 0,
        latency_std=latencies.std() if len(latencies) > 0 else 0,
        latency_min=latencies.min() if len(latencies) > 0 else 0,
        latency_max=latencies.max() if len(latencies) > 0 else 0,
        inference_count=len(latencies),

        # Thermal
        thermal_start=thermals.iloc[0] if len(thermals) > 0 else 0,
        thermal_end=thermals.iloc[-1] if len(thermals) > 0 else 0,
        thermal_max=thermals.max() if len(thermals) > 0 else 0,
        thermal_slope=thermal_slope,

        # Power
        power_mean=powers.mean() if len(powers) > 0 else 0,
        power_std=powers.std() if len(powers) > 0 else 0,

        # Memory
        memory_peak=int(memories.max()) if len(memories) > 0 else 0,
        memory_mean=memories.mean() if len(memories) > 0 else 0,

        # Duration
        duration_seconds=duration
    )


def find_steady_state(
    df: pd.DataFrame,
    warm_up_seconds: float = 30.0
) -> pd.DataFrame:
    """
    Extract steady-state portion of the log (after warm-up period).

    Args:
        df: Full log DataFrame
        warm_up_seconds: Seconds to skip at the beginning

    Returns:
        DataFrame with only steady-state data
    """
    return df[df['elapsed_seconds'] >= warm_up_seconds].copy()


def parse_session_id(filename: str) -> dict:
    """
    Parse experiment parameters from session ID in filename.

    Expected format: kpi_log_YYYYMMDD_HHMMSS.csv
    Session ID in data: {path}_{freq}hz_{warmup}_{timestamp}

    Args:
        filename: Log filename

    Returns:
        Dict with parsed parameters
    """
    # This would need the actual session ID from within the file
    # For now, return empty dict
    return {}


def load_all_logs(directory: str) -> pd.DataFrame:
    """
    Load all CSV logs from a directory and combine them.

    Args:
        directory: Path to directory containing CSV files

    Returns:
        Combined DataFrame with 'source_file' column
    """
    path = Path(directory)
    all_dfs = []

    for csv_file in path.glob("*.csv"):
        df = load_log(str(csv_file))
        df['source_file'] = csv_file.name
        all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_logs.py <log_file.csv>")
        sys.exit(1)

    file_path = sys.argv[1]

    print(f"Loading: {file_path}")
    df = load_log(file_path)

    print(f"\nTotal records: {len(df)}")
    print(f"Duration: {df['elapsed_seconds'].max():.1f} seconds")

    metrics = calculate_metrics(df)

    print("\n=== KPI Summary ===")
    print(f"\nLatency:")
    print(f"  P50: {metrics.latency_p50:.2f} ms")
    print(f"  P95: {metrics.latency_p95:.2f} ms")
    print(f"  Mean: {metrics.latency_mean:.2f} ms (±{metrics.latency_std:.2f})")
    print(f"  Count: {metrics.inference_count}")

    print(f"\nThermal:")
    print(f"  Start: {metrics.thermal_start:.1f} °C")
    print(f"  End: {metrics.thermal_end:.1f} °C")
    print(f"  Max: {metrics.thermal_max:.1f} °C")
    print(f"  Slope: {metrics.thermal_slope:.2f} °C/min")

    print(f"\nPower:")
    print(f"  Mean: {metrics.power_mean:.1f} mW (±{metrics.power_std:.1f})")

    print(f"\nMemory:")
    print(f"  Peak: {metrics.memory_peak} MB")
    print(f"  Mean: {metrics.memory_mean:.1f} MB")
