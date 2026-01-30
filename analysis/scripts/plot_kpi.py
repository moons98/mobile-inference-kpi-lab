#!/usr/bin/env python3
"""
Visualization functions for KPI analysis.
"""

import sys

# Set non-interactive backend when running as script (before importing pyplot)
if __name__ == "__main__" and "--show" not in sys.argv and "-s" not in sys.argv:
    import matplotlib
    matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple

from parse_logs import load_log, split_events, calculate_metrics


def setup_style():
    """Set up matplotlib style for consistent plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


def plot_latency_timeline(
    df: pd.DataFrame,
    title: str = "Latency Over Time",
    save_path: Optional[str] = None
):
    """
    Plot latency over time.

    Args:
        df: Log DataFrame
        title: Plot title
        save_path: Optional path to save figure
    """
    inference_df, _ = split_events(df)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        inference_df['elapsed_seconds'] / 60,
        inference_df['latency_ms'],
        'b-', alpha=0.7, linewidth=0.5
    )

    # Add moving average
    window = min(50, len(inference_df) // 10)
    if window > 1:
        rolling_mean = inference_df['latency_ms'].rolling(window=window).mean()
        ax.plot(
            inference_df['elapsed_seconds'] / 60,
            rolling_mean,
            'r-', linewidth=2, label=f'Moving Avg (n={window})'
        )

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig, ax


def plot_latency_histogram(
    df: pd.DataFrame,
    title: str = "Latency Distribution",
    save_path: Optional[str] = None
):
    """
    Plot latency histogram with percentiles.

    Args:
        df: Log DataFrame
        title: Plot title
        save_path: Optional path to save figure
    """
    inference_df, _ = split_events(df)
    latencies = inference_df['latency_ms'].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(latencies, bins=50, alpha=0.7, edgecolor='black')

    # Add percentile lines
    p50 = latencies.quantile(0.50)
    p95 = latencies.quantile(0.95)

    ax.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'P50: {p50:.2f} ms')
    ax.axvline(p95, color='red', linestyle='--', linewidth=2, label=f'P95: {p95:.2f} ms')

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig, ax


def plot_thermal_timeline(
    df: pd.DataFrame,
    title: str = "Thermal Over Time",
    save_path: Optional[str] = None
):
    """
    Plot temperature over time with trend line.

    Args:
        df: Log DataFrame
        title: Plot title
        save_path: Optional path to save figure
    """
    _, system_df = split_events(df)
    thermal_data = system_df[system_df['thermal_c'] > 0]

    fig, ax = plt.subplots(figsize=(12, 5))

    times = thermal_data['elapsed_seconds'] / 60
    temps = thermal_data['thermal_c']

    ax.plot(times, temps, 'b-', linewidth=1.5, label='Temperature')

    # Add linear trend
    if len(times) >= 2:
        coeffs = np.polyfit(times, temps, 1)
        trend_line = np.poly1d(coeffs)
        ax.plot(times, trend_line(times), 'r--', linewidth=2,
                label=f'Trend: {coeffs[0]:.2f} °C/min')

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig, ax


def plot_power_timeline(
    df: pd.DataFrame,
    title: str = "Power Consumption Over Time",
    save_path: Optional[str] = None
):
    """
    Plot power consumption over time.

    Args:
        df: Log DataFrame
        title: Plot title
        save_path: Optional path to save figure
    """
    _, system_df = split_events(df)
    power_data = system_df[system_df['power_mw'] > 0]

    fig, ax = plt.subplots(figsize=(12, 5))

    times = power_data['elapsed_seconds'] / 60
    power = power_data['power_mw']

    ax.plot(times, power, 'g-', alpha=0.7, linewidth=0.5)

    # Add moving average
    window = min(30, len(power) // 5)
    if window > 1:
        rolling_mean = power.rolling(window=window).mean()
        ax.plot(times, rolling_mean, 'darkgreen', linewidth=2,
                label=f'Moving Avg (n={window})')

    # Add mean line
    mean_power = power.mean()
    ax.axhline(mean_power, color='red', linestyle='--',
               label=f'Mean: {mean_power:.0f} mW')

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Power (mW)')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig, ax


def plot_kpi_dashboard(
    df: pd.DataFrame,
    title: str = "KPI Dashboard",
    save_path: Optional[str] = None
):
    """
    Create a 2x2 dashboard with all KPI plots.

    Args:
        df: Log DataFrame
        title: Overall title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    inference_df, system_df = split_events(df)

    # 1. Latency timeline (top-left)
    ax = axes[0, 0]
    times = inference_df['elapsed_seconds'] / 60
    latencies = inference_df['latency_ms']
    ax.plot(times, latencies, 'b-', alpha=0.5, linewidth=0.5)
    window = min(50, len(inference_df) // 10)
    if window > 1:
        rolling = latencies.rolling(window=window).mean()
        ax.plot(times, rolling, 'r-', linewidth=2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Over Time')

    # 2. Latency histogram (top-right)
    ax = axes[0, 1]
    ax.hist(latencies.dropna(), bins=40, alpha=0.7, edgecolor='black')
    p50 = latencies.quantile(0.50)
    p95 = latencies.quantile(0.95)
    ax.axvline(p50, color='green', linestyle='--', label=f'P50: {p50:.1f}')
    ax.axvline(p95, color='red', linestyle='--', label=f'P95: {p95:.1f}')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Latency Distribution')
    ax.legend()

    # 3. Thermal (bottom-left)
    ax = axes[1, 0]
    thermal_data = system_df[system_df['thermal_c'] > 0]
    if len(thermal_data) > 0:
        times = thermal_data['elapsed_seconds'] / 60
        temps = thermal_data['thermal_c']
        ax.plot(times, temps, 'orange', linewidth=1.5)
        if len(times) >= 2:
            coeffs = np.polyfit(times, temps, 1)
            trend = np.poly1d(coeffs)
            ax.plot(times, trend(times), 'r--', linewidth=2,
                    label=f'Slope: {coeffs[0]:.2f} °C/min')
            ax.legend()
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Thermal')

    # 4. Power (bottom-right)
    ax = axes[1, 1]
    power_data = system_df[system_df['power_mw'] > 0]
    if len(power_data) > 0:
        times = power_data['elapsed_seconds'] / 60
        power = power_data['power_mw']
        ax.plot(times, power, 'g-', alpha=0.5, linewidth=0.5)
        mean_power = power.mean()
        ax.axhline(mean_power, color='darkgreen', linestyle='--',
                   label=f'Mean: {mean_power:.0f} mW')
        ax.legend()
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Power (mW)')
    ax.set_title('Power Consumption')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig, axes


def compare_experiments(
    log_files: List[str],
    labels: List[str],
    save_path: Optional[str] = None
):
    """
    Compare KPI metrics across multiple experiments with overlaid time series.

    Args:
        log_files: List of log file paths
        labels: Labels for each experiment
        save_path: Optional path to save figure
    """
    # Load all data
    data_list = []
    metrics_list = []

    for file_path in log_files:
        df = load_log(file_path)
        metrics = calculate_metrics(df)
        inference_df, system_df = split_events(df)
        data_list.append({
            'df': df,
            'inference_df': inference_df,
            'system_df': system_df
        })
        metrics_list.append(metrics)

    # Color scheme:
    # - NPU FP32: red shades
    # - NPU Quantized (INT8): orange/brown shades
    # - GPU: blue shades
    # - CPU: green shades
    # Line style by quantization: FP32=solid, Dynamic=dashed, QDQ/Static=dash-dot
    npu_fp32_shades = ['#d62728', '#c42020', '#ff4444']  # reds
    npu_quant_shades = ['#ff7f0e', '#d2691e', '#8b4513', '#cd853f']  # orange, chocolate, saddle brown, peru
    gpu_shades = ['#1f77b4', '#6baed6', '#08519c', '#3498db']  # blues
    cpu_shades = ['#2ca02c', '#74c476', '#006d2c', '#27ae60']  # greens

    ep_counters = {'npu_fp32': 0, 'npu_quant': 0, 'gpu': 0, 'cpu': 0}

    def is_quantized(label: str) -> bool:
        label_upper = label.upper()
        return 'INT8' in label_upper or 'QDQ' in label_upper or 'DYNAMIC' in label_upper or 'QUANT' in label_upper

    def get_color_category(label: str) -> str:
        label_upper = label.upper()
        if 'NPU' in label_upper or 'HTP' in label_upper:
            if is_quantized(label):
                return 'npu_quant'
            return 'npu_fp32'
        elif 'GPU' in label_upper:
            return 'gpu'
        return 'cpu'

    def get_color_for_label(label: str) -> str:
        cat = get_color_category(label)
        idx = ep_counters[cat]
        ep_counters[cat] += 1
        if cat == 'npu_fp32':
            return npu_fp32_shades[idx % len(npu_fp32_shades)]
        elif cat == 'npu_quant':
            return npu_quant_shades[idx % len(npu_quant_shades)]
        elif cat == 'gpu':
            return gpu_shades[idx % len(gpu_shades)]
        return cpu_shades[idx % len(cpu_shades)]

    def get_line_style(label: str) -> str:
        label_upper = label.upper()
        if 'QDQ' in label_upper or 'STATIC' in label_upper:
            return '-.'  # dash-dot for static/QDQ quantization
        elif 'INT8' in label_upper or 'DYNAMIC' in label_upper or 'QUANT' in label_upper:
            return '--'  # dashed for dynamic quantization
        return '-'  # solid for FP32

    colors = [get_color_for_label(label) for label in labels]
    styles = [get_line_style(label) for label in labels]

    # Create numbered labels: "1: Label", "2: Label", ...
    numbered_labels = [f"{i+1}: {label}" for i, label in enumerate(labels)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment Comparison (Overlaid)', fontsize=14, fontweight='bold')

    # 1. Latency Over Time (top-left) - overlaid with moving average
    ax = axes[0, 0]
    for i, (data, label, color, style) in enumerate(zip(data_list, numbered_labels, colors, styles)):
        inference_df = data['inference_df']
        times = inference_df['elapsed_seconds'] / 60
        latencies = inference_df['latency_ms']

        # Moving average only (cleaner for comparison)
        window = min(50, max(1, len(inference_df) // 10))
        if window > 1:
            rolling = latencies.rolling(window=window).mean()
            ax.plot(times, rolling, color=color, linestyle=style, linewidth=2.5, label=f"{i+1}")
        else:
            ax.plot(times, latencies, color=color, linestyle=style, linewidth=1.5, alpha=0.8, label=f"{i+1}")

    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Over Time (Moving Avg)')
    ax.legend(loc='upper right', fontsize=9)

    # 2. Thermal Over Time (top-right) - overlaid
    ax = axes[0, 1]
    for i, (data, label, color, style) in enumerate(zip(data_list, numbered_labels, colors, styles)):
        system_df = data['system_df']
        thermal_data = system_df[system_df['thermal_c'] > 0]
        if len(thermal_data) > 0:
            times = thermal_data['elapsed_seconds'] / 60
            temps = thermal_data['thermal_c']
            ax.plot(times, temps, color=color, linestyle=style, linewidth=2, label=f"{i+1}")

    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Thermal Over Time')
    ax.legend(loc='upper right', fontsize=9)

    # 3. Power Over Time (bottom-left) - overlaid with moving average
    ax = axes[1, 0]
    for i, (data, label, color, style) in enumerate(zip(data_list, numbered_labels, colors, styles)):
        system_df = data['system_df']
        power_data = system_df[system_df['power_mw'] > 0]
        if len(power_data) > 0:
            times = power_data['elapsed_seconds'] / 60
            power = power_data['power_mw']

            # Moving average for cleaner comparison
            window = min(30, max(1, len(power) // 5))
            if window > 1:
                rolling = power.rolling(window=window).mean()
                ax.plot(times, rolling, color=color, linestyle=style, linewidth=2, label=f"{i+1}")
            else:
                ax.plot(times, power, color=color, linestyle=style, linewidth=1.5, alpha=0.8, label=f"{i+1}")

    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Power (mW)')
    ax.set_title('Power Consumption (Moving Avg)')
    ax.legend(loc='upper right', fontsize=9)

    # 4. Summary table (bottom-right)
    ax = axes[1, 1]
    ax.axis('off')

    table_data = []
    headers = ['#', 'Label', 'P50', 'P95', 'Slope', 'Power']

    for i, (label, m) in enumerate(zip(labels, metrics_list)):
        table_data.append([
            f'{i+1}',
            label[:35] + '...' if len(label) > 35 else label,
            f'{m.latency_p50:.1f}',
            f'{m.latency_p95:.1f}',
            f'{m.thermal_slope:.2f}',
            f'{m.power_mean:.1f}'
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.06, 0.4, 0.12, 0.12, 0.12, 0.12]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Style header row
    for j, key in enumerate(headers):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig, axes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate KPI dashboard plots")
    parser.add_argument("path", help="Log CSV file or directory containing kpi_*.csv files")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory for plots (default: outputs/ in project root)")
    parser.add_argument("--show", "-s", action="store_true", help="Show plot window (default: save only)")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare mode: treat path as directory and generate comparison plot")

    args = parser.parse_args()

    setup_style()

    input_path = Path(args.path)

    if args.compare or input_path.is_dir():
        # Directory mode: compare all kpi_*.csv files
        if not input_path.is_dir():
            print(f"Error: {input_path} is not a directory")
            sys.exit(1)

        log_files = sorted(input_path.glob("kpi_*.csv"))
        if not log_files:
            print(f"No kpi_*.csv files found in {input_path}")
            sys.exit(1)

        print(f"Found {len(log_files)} log file(s) in {input_path}")

        # Extract labels from filenames (remove kpi_ prefix and timestamp suffix)
        # Format: kpi_ModelName_EP_YYYYMMDD_HHMMSS -> ModelName_EP
        import re
        def extract_label(filename: str) -> str:
            name = filename.replace("kpi_", "")
            # Remove timestamp suffix (YYYYMMDD_HHMMSS)
            name = re.sub(r'_\d{8}_\d{6}$', '', name)
            return name
        labels = [extract_label(f.stem) for f in log_files]

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Default: outputs/ folder in project root
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            output_dir = project_root / "outputs"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Use input folder name in output filename
        folder_name = input_path.name
        output_path = output_dir / f"{folder_name}_comparison.png"

        print("Generating comparison plot...")
        compare_experiments(
            log_files=[str(f) for f in log_files],
            labels=labels,
            save_path=str(output_path)
        )
        print(f"Saved: {output_path}")

    else:
        # Single file mode: generate dashboard
        if not input_path.exists():
            print(f"Error: {input_path} does not exist")
            sys.exit(1)

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Default: outputs/ folder in project root
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            output_dir = project_root / "outputs"

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading: {input_path}")
        df = load_log(str(input_path))

        base_name = input_path.stem

        print("Generating dashboard...")
        plot_kpi_dashboard(df, save_path=str(output_dir / f"{base_name}_dashboard.png"))

    print("Done!")

    if args.show:
        plt.show()
