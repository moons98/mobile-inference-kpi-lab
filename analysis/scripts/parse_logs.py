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
class ColdStartMetrics:
    """Cold start timing metrics."""
    model_load_ms: float
    session_create_ms: float  # QNN compilation time
    first_inference_ms: float
    total_cold_ms: float


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

    # Throughput
    fps: float  # Inferences per second

    # Thermal drift metrics (for sustained performance analysis)
    first_30s_p50: float  # Latency p50 in first 30 seconds
    last_30s_p50: float   # Latency p50 in last 30 seconds

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

    # Cold start (optional, from metadata)
    cold_start: Optional[ColdStartMetrics] = None


def load_metadata(file_path: str) -> dict:
    """
    Extract metadata from comment lines in a KPI log CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        Dict with metadata key-value pairs
    """
    metadata = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('# '):
                parts = line[2:].strip().split(',', 1)
                if len(parts) == 2:
                    metadata[parts[0]] = parts[1]
            elif not line.startswith('#'):
                break  # Stop at data rows
    return metadata


def load_log(file_path: str) -> pd.DataFrame:
    """
    Load a KPI log CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame with parsed log data
    """
    # Skip comment lines starting with '#' (metadata)
    df = pd.read_csv(file_path, comment='#')

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


def calculate_metrics(df: pd.DataFrame, metadata: dict = None) -> KpiMetrics:
    """
    Calculate aggregated KPI metrics from a log file.

    Args:
        df: Full log DataFrame
        metadata: Optional metadata dict from CSV header

    Returns:
        KpiMetrics with calculated values
    """
    inference_df, system_df = split_events(df)

    # Latency metrics
    latencies = inference_df['latency_ms'].dropna()

    # Duration
    duration = df['elapsed_seconds'].max()

    # Throughput (FPS)
    fps = len(latencies) / duration if duration > 0 else 0

    # First 30s and Last 30s latency p50 (for thermal drift analysis)
    first_30s_latencies = inference_df[inference_df['elapsed_seconds'] <= 30]['latency_ms'].dropna()
    last_30s_latencies = inference_df[inference_df['elapsed_seconds'] >= duration - 30]['latency_ms'].dropna()

    first_30s_p50 = first_30s_latencies.quantile(0.50) if len(first_30s_latencies) > 0 else 0
    last_30s_p50 = last_30s_latencies.quantile(0.50) if len(last_30s_latencies) > 0 else 0

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

    # Cold start metrics from metadata
    cold_start = None
    if metadata:
        try:
            model_load = float(metadata.get('model_load_ms', 0))
            session_create = float(metadata.get('session_create_ms', 0))
            first_inf_str = metadata.get('first_inference_ms', 'N/A')
            first_inf = float(first_inf_str) if first_inf_str != 'N/A' else 0
            total_cold = float(metadata.get('total_cold_ms', 0))

            if model_load > 0 or session_create > 0:
                cold_start = ColdStartMetrics(
                    model_load_ms=model_load,
                    session_create_ms=session_create,
                    first_inference_ms=first_inf,
                    total_cold_ms=total_cold
                )
        except (ValueError, TypeError):
            pass

    return KpiMetrics(
        # Latency
        latency_p50=latencies.quantile(0.50) if len(latencies) > 0 else 0,
        latency_p95=latencies.quantile(0.95) if len(latencies) > 0 else 0,
        latency_mean=latencies.mean() if len(latencies) > 0 else 0,
        latency_std=latencies.std() if len(latencies) > 0 else 0,
        latency_min=latencies.min() if len(latencies) > 0 else 0,
        latency_max=latencies.max() if len(latencies) > 0 else 0,
        inference_count=len(latencies),

        # Throughput
        fps=fps,

        # Thermal drift
        first_30s_p50=first_30s_p50,
        last_30s_p50=last_30s_p50,

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
        duration_seconds=duration,

        # Cold start
        cold_start=cold_start
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


def generate_single_report(file_path: str) -> Tuple[str, dict, KpiMetrics]:
    """
    Generate detailed report string for a single log file.

    Args:
        file_path: Path to CSV file

    Returns:
        Tuple of (report_string, metadata, metrics)
    """
    lines = []
    lines.append(f"Loading: {file_path}")

    # Load metadata
    metadata = load_metadata(file_path)
    if metadata:
        lines.append("\n=== Metadata ===")
        lines.append(f"  Device: {metadata.get('device_manufacturer', 'N/A')} {metadata.get('device_model', 'N/A')}")
        lines.append(f"  SoC: {metadata.get('soc_model', 'N/A')}")
        lines.append(f"  EP: {metadata.get('execution_provider', 'N/A')}")
        lines.append(f"  Model: {metadata.get('model', 'N/A')}")

    df = load_log(file_path)

    lines.append(f"\nTotal records: {len(df)}")
    lines.append(f"Duration: {df['elapsed_seconds'].max():.1f} seconds")

    metrics = calculate_metrics(df, metadata)

    # Cold Start metrics
    if metrics.cold_start:
        lines.append("\n=== Cold Start ===")
        lines.append(f"  Model Load: {metrics.cold_start.model_load_ms:.0f} ms")
        lines.append(f"  Session Create (QNN compile): {metrics.cold_start.session_create_ms:.0f} ms")
        lines.append(f"  First Inference: {metrics.cold_start.first_inference_ms:.2f} ms")
        lines.append(f"  Total Cold: {metrics.cold_start.total_cold_ms:.0f} ms")

    lines.append("\n=== Sustained Performance ===")
    lines.append(f"\nLatency:")
    lines.append(f"  P50: {metrics.latency_p50:.2f} ms")
    lines.append(f"  P95: {metrics.latency_p95:.2f} ms")
    lines.append(f"  Mean: {metrics.latency_mean:.2f} ms (±{metrics.latency_std:.2f})")
    lines.append(f"  Min: {metrics.latency_min:.2f} ms")
    lines.append(f"  Max: {metrics.latency_max:.2f} ms")
    lines.append(f"  Count: {metrics.inference_count}")

    lines.append(f"\nThroughput:")
    lines.append(f"  FPS: {metrics.fps:.2f} inf/s")

    lines.append(f"\nThermal Drift:")
    lines.append(f"  First 30s P50: {metrics.first_30s_p50:.2f} ms")
    lines.append(f"  Last 30s P50: {metrics.last_30s_p50:.2f} ms")
    drift_pct = ((metrics.last_30s_p50 - metrics.first_30s_p50) / metrics.first_30s_p50 * 100) if metrics.first_30s_p50 > 0 else 0
    lines.append(f"  Drift: {drift_pct:+.1f}%")

    lines.append(f"\nThermal:")
    lines.append(f"  Start: {metrics.thermal_start:.1f} °C")
    lines.append(f"  End: {metrics.thermal_end:.1f} °C")
    lines.append(f"  Max: {metrics.thermal_max:.1f} °C")
    lines.append(f"  Slope: {metrics.thermal_slope:.2f} °C/min")

    lines.append(f"\nPower:")
    lines.append(f"  Mean: {metrics.power_mean:.1f} mW (±{metrics.power_std:.1f})")

    lines.append(f"\nMemory:")
    lines.append(f"  Peak: {metrics.memory_peak} MB")
    lines.append(f"  Mean: {metrics.memory_mean:.1f} MB")

    return "\n".join(lines), metadata, metrics


def print_single_report(file_path: str):
    """Print detailed report for a single log file."""
    report, metadata, metrics = generate_single_report(file_path)
    print(report)
    return metadata, metrics


def generate_comparison_table(file_paths: list) -> str:
    """
    Generate comparison table string for multiple log files.

    Args:
        file_paths: List of paths to CSV files

    Returns:
        Comparison table as string
    """
    lines = []
    lines.append("=" * 130)
    lines.append("Comparison Report")
    lines.append("=" * 130)

    # Header
    lines.append(f"\n{'File':<25} {'EP':<10} {'Model':<15} {'P50':<8} {'P95':<8} {'FPS':<8} {'Cold':<8} {'Drift%':<8} {'Power':<8}")
    lines.append("-" * 130)

    for file_path in file_paths:
        metadata = load_metadata(file_path)
        df = load_log(file_path)
        metrics = calculate_metrics(df, metadata)

        name = Path(file_path).stem[-23:]
        ep = metadata.get('execution_provider', 'N/A')[:8]
        model = metadata.get('model', 'N/A')[:13]

        # Cold start total
        cold_ms = metrics.cold_start.total_cold_ms if metrics.cold_start else 0

        # Drift percentage
        drift_pct = ((metrics.last_30s_p50 - metrics.first_30s_p50) / metrics.first_30s_p50 * 100) if metrics.first_30s_p50 > 0 else 0

        lines.append(f"{name:<25} {ep:<10} {model:<15} "
              f"{metrics.latency_p50:<8.2f} "
              f"{metrics.latency_p95:<8.2f} "
              f"{metrics.fps:<8.1f} "
              f"{cold_ms:<8.0f} "
              f"{drift_pct:<+8.1f} "
              f"{metrics.power_mean:<8.1f}")

    lines.append("")
    return "\n".join(lines)


def print_comparison_table(file_paths: list):
    """Print comparison table for multiple log files."""
    report = generate_comparison_table(file_paths)
    print(report)


if __name__ == "__main__":
    import sys
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Parse and analyze KPI log files")
    parser.add_argument("paths", nargs="+", help="Log files or directories to analyze")
    parser.add_argument("--compare", "-c", action="store_true", help="Show comparison table")
    parser.add_argument("--output", "-o", help="Output file path (txt). If not specified, auto-generates in input location")
    parser.add_argument("--print", "-p", action="store_true", help="Print to console instead of saving to file")

    args = parser.parse_args()

    # Collect all log files
    log_files = []
    input_dir = None
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            input_dir = path
            # Support both old (kpi_log_*) and new (kpi_Model_EP_*) filename formats
            log_files.extend(sorted(path.glob("kpi_*.csv")))
        elif path.exists():
            if input_dir is None:
                input_dir = path.parent
            log_files.append(path)
        else:
            print(f"Warning: {p} not found")

    if not log_files:
        print("No log files found")
        sys.exit(1)

    header = f"Found {len(log_files)} log file(s)\n"

    # Generate report
    is_comparison = args.compare or len(log_files) > 1
    if is_comparison:
        report = generate_comparison_table([str(f) for f in log_files])
    else:
        report, _, _ = generate_single_report(str(log_files[0]))

    full_report = header + report

    # Determine output path
    if args.print:
        # Print to console
        print(full_report)
    else:
        # Save to file
        if args.output:
            output_path = Path(args.output)
        else:
            # Auto-generate output path in input location
            if is_comparison:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = input_dir / f"comparison_report_{timestamp}.txt"
            else:
                # Single file: {input_stem}_report.txt
                output_path = input_dir / f"{log_files[0].stem}_report.txt"

        output_path.write_text(full_report, encoding='utf-8')
        print(f"Report saved to: {output_path}")
