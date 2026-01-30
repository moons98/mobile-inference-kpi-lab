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


def load_ort_log(csv_path: str) -> Optional[dict]:
    """
    Load ORT log file if it exists alongside CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Dict with ORT log info or None if not found
    """
    # ORT log file has same name as CSV but with _ort.log suffix
    # e.g., kpi_MobileNetV2_QNNNPU_20260130_173916.csv -> kpi_MobileNetV2_QNNNPU_20260130_173916_ort.log
    log_path = Path(csv_path).with_suffix('').with_suffix('_ort.log')

    # Also try the pattern where .csv is simply replaced
    if not log_path.exists():
        log_path = Path(str(csv_path).replace('.csv', '_ort.log'))

    if not log_path.exists():
        return None

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        ort_info = {
            'raw_log': content,
            'total_nodes': 0,
            'qnn_nodes': 0,
            'cpu_nodes': 0,
            'fallback_ops': []
        }

        # Parse summary section
        for line in content.split('\n'):
            if 'Total nodes:' in line:
                try:
                    ort_info['total_nodes'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'QNN nodes:' in line:
                try:
                    ort_info['qnn_nodes'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'CPU fallback nodes:' in line:
                try:
                    ort_info['cpu_nodes'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'Fallback ops:' in line:
                ops_str = line.split(':')[1].strip()
                if ops_str:
                    ort_info['fallback_ops'] = [op.strip() for op in ops_str.split(',')]

        return ort_info
    except Exception as e:
        print(f"Warning: Failed to load ORT log {log_path}: {e}")
        return None


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
    Generate sectioned comparison report for multiple log files.

    Args:
        file_paths: List of paths to CSV files

    Returns:
        Comparison report as string with multiple sections
    """
    import re

    # Load all data first
    experiments = []
    for file_path in file_paths:
        metadata = load_metadata(file_path)
        df = load_log(file_path)
        metrics = calculate_metrics(df, metadata)
        ort_log = load_ort_log(file_path)  # Load ORT log if available

        # Extract short label (remove kpi_ prefix and timestamp)
        name = Path(file_path).stem
        label = re.sub(r'^kpi_', '', name)
        label = re.sub(r'_\d{8}_\d{6}$', '', label)

        experiments.append({
            'label': label,
            'name': name,
            'ep': metadata.get('execution_provider', 'N/A'),
            'model': metadata.get('model', 'N/A'),
            'metrics': metrics,
            'metadata': metadata,
            'ort_log': ort_log
        })

    lines = []
    sep = "=" * 120

    # Title
    lines.append(sep)
    lines.append("KPI Comparison Report")
    lines.append(sep)

    # Section 1: Experiment Overview
    lines.append("\n[1] Experiment Overview")
    lines.append("-" * 120)
    lines.append(f"{'#':<4} {'Label':<40} {'EP':<12} {'Model':<30}")
    lines.append("-" * 120)
    for i, exp in enumerate(experiments, 1):
        lines.append(f"{i:<4} {exp['label']:<40} {exp['ep']:<12} {exp['model']:<30}")
    lines.append("")
    lines.append("    EP: Execution Provider (QNN_NPU=Hexagon HTP, QNN_GPU=Adreno GPU, CPU=ARM CPU)")

    # Section 2: Latency Performance
    lines.append(f"\n\n[2] Latency Performance")
    lines.append("-" * 120)
    lines.append(f"{'#':<4} {'Label':<40} {'P50(ms)':<10} {'P95(ms)':<10} {'Mean(ms)':<10} {'Std(ms)':<10} {'MaxFPS':<10}")
    lines.append("-" * 120)
    for i, exp in enumerate(experiments, 1):
        m = exp['metrics']
        # Theoretical max FPS = 1000ms / P50 latency
        max_fps = 1000.0 / m.latency_p50 if m.latency_p50 > 0 else 0
        lines.append(f"{i:<4} {exp['label']:<40} {m.latency_p50:<10.2f} {m.latency_p95:<10.2f} "
                     f"{m.latency_mean:<10.2f} {m.latency_std:<10.2f} {max_fps:<10.1f}")
    lines.append("")
    lines.append("    P50/P95: 50th/95th percentile latency (lower=better)")
    lines.append("    MaxFPS: Theoretical max throughput = 1000/P50 (higher=better)")

    # Section 3: Cold Start Breakdown
    lines.append(f"\n\n[3] Cold Start Breakdown")
    lines.append("-" * 120)
    lines.append(f"{'#':<4} {'Label':<40} {'Total(ms)':<12} {'Load(ms)':<12} {'Session(ms)':<12} {'1stInf(ms)':<12}")
    lines.append("-" * 120)
    for i, exp in enumerate(experiments, 1):
        m = exp['metrics']
        if m.cold_start:
            total = m.cold_start.total_cold_ms
            load = m.cold_start.model_load_ms
            session = m.cold_start.session_create_ms
            first = m.cold_start.first_inference_ms
            lines.append(f"{i:<4} {exp['label']:<40} {total:<12.0f} {load:<12.0f} {session:<12.0f} {first:<12.2f}")
        else:
            lines.append(f"{i:<4} {exp['label']:<40} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    lines.append("")
    lines.append("    Total: 앱 시작부터 첫 추론 완료까지 = Load + Session + 1stInf")
    lines.append("    Load: ONNX 모델 파일 로드 시간")
    lines.append("    Session: ORT 세션 생성 시간 (QNN EP는 HTP 그래프 컴파일 포함)")
    lines.append("    1stInf: 첫 번째 추론 지연시간")

    # Section 4: Thermal Drift Analysis
    lines.append(f"\n\n[4] Thermal Drift Analysis (Latency P50 변화)")
    lines.append("-" * 120)
    lines.append(f"{'#':<4} {'Label':<40} {'First30s(ms)':<14} {'Last30s(ms)':<14} {'Drift%':<10} {'Verdict':<16}")
    lines.append("-" * 120)
    for i, exp in enumerate(experiments, 1):
        m = exp['metrics']
        first_30s = m.first_30s_p50
        last_30s = m.last_30s_p50
        drift_pct = ((last_30s - first_30s) / first_30s * 100) if first_30s > 0 else 0

        # Verdict based on drift
        if abs(drift_pct) < 3:
            verdict = "Stable"
        elif drift_pct > 10:
            verdict = "Throttling"
        elif drift_pct > 3:
            verdict = "Slight throttle"
        elif drift_pct < -3:
            verdict = "Warmup effect"
        else:
            verdict = "Normal"

        lines.append(f"{i:<4} {exp['label']:<40} {first_30s:<14.2f} {last_30s:<14.2f} {drift_pct:<+10.1f} {verdict:<16}")
    lines.append("")
    lines.append("    First30s/Last30s: 벤치마크 처음/마지막 30초 동안의 Latency P50")
    lines.append("    Drift%: (Last30s - First30s) / First30s × 100")
    lines.append("    Verdict: Stable(<±3%), Slight throttle(3~10%), Throttling(>10%), Warmup effect(<-3%)")

    # Section 5: System Resources
    lines.append(f"\n\n[5] System Resources")
    lines.append("-" * 120)
    lines.append(f"{'#':<4} {'Label':<40} {'Power(mW)':<12} {'Slope(°C/min)':<14} {'MemPeak(MB)':<12}")
    lines.append("-" * 120)
    for i, exp in enumerate(experiments, 1):
        m = exp['metrics']
        lines.append(f"{i:<4} {exp['label']:<40} {m.power_mean:<12.1f} {m.thermal_slope:<14.2f} {m.memory_peak:<12}")
    lines.append("")
    lines.append("    Power: 평균 전력 소비량 (mW)")
    lines.append("    Slope: 온도 상승률 (°C/min, lower=better)")
    lines.append("    MemPeak: 최대 메모리 사용량 (MB)")

    # Section 6: Model Details
    lines.append(f"\n\n[6] Model Details")
    lines.append("-" * 120)
    lines.append(f"{'#':<4} {'Label':<40} {'Size(KB)':<12} {'Input Shape':<24} {'QNN Options':<30}")
    lines.append("-" * 120)
    for i, exp in enumerate(experiments, 1):
        meta = exp['metadata']
        size_kb = meta.get('model_size_kb', 'N/A')
        input_shape = meta.get('input_shape', 'N/A')
        qnn_opts = meta.get('qnn_options', 'N/A')
        lines.append(f"{i:<4} {exp['label']:<40} {size_kb:<12} {input_shape:<24} {qnn_opts:<30}")
    lines.append("")
    lines.append("    Size: 모델 파일 크기 (KB)")
    lines.append("    Input Shape: 입력 텐서 shape [N,C,H,W]")
    lines.append("    QNN Options: QNN Execution Provider 옵션 (NPU/GPU 사용시)")
    # Section 7: Graph Partitioning (if ORT logs available)
    has_ort_logs = any(exp.get('ort_log') for exp in experiments)
    if has_ort_logs:
        lines.append(f"\n\n[7] Graph Partitioning (from ORT logs)")
        lines.append("-" * 120)
        lines.append(f"{'#':<4} {'Label':<40} {'Total':<10} {'QNN':<10} {'CPU':<10} {'Fallback Ops':<40}")
        lines.append("-" * 120)
        for i, exp in enumerate(experiments, 1):
            ort = exp.get('ort_log')
            if ort:
                total = ort.get('total_nodes', 0)
                qnn = ort.get('qnn_nodes', 0)
                cpu = ort.get('cpu_nodes', 0)
                fallback = ', '.join(ort.get('fallback_ops', []))[:40] or '-'
                lines.append(f"{i:<4} {exp['label']:<40} {total:<10} {qnn:<10} {cpu:<10} {fallback:<40}")
            else:
                lines.append(f"{i:<4} {exp['label']:<40} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'(no ORT log)':<40}")
        lines.append("")
        lines.append("    Total: 그래프 총 노드 수")
        lines.append("    QNN: QNN EP에서 실행되는 노드 수 (NPU/GPU)")
        lines.append("    CPU: CPU fallback 노드 수")
        lines.append("    Fallback Ops: QNN에서 지원하지 않아 CPU로 실행되는 연산자")
    else:
        lines.append("")
        lines.append("    NOTE: Graph partitioning 정보를 보려면 앱에서 벤치마크 실행 후")
        lines.append("          생성되는 *_ort.log 파일을 CSV와 같은 폴더에 두세요.")

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
            # Search for CSV files in the specified directory only (not subdirectories)
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
