#!/usr/bin/env python3
"""
Visualization functions for KPI analysis.
"""

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
    Compare KPI metrics across multiple experiments.

    Args:
        log_files: List of log file paths
        labels: Labels for each experiment
        save_path: Optional path to save figure
    """
    metrics_list = []

    for file_path in log_files:
        df = load_log(file_path)
        metrics = calculate_metrics(df)
        metrics_list.append(metrics)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Experiment Comparison', fontsize=14, fontweight='bold')

    x = np.arange(len(labels))
    width = 0.35

    # Latency comparison
    ax = axes[0, 0]
    p50_vals = [m.latency_p50 for m in metrics_list]
    p95_vals = [m.latency_p95 for m in metrics_list]
    ax.bar(x - width/2, p50_vals, width, label='P50')
    ax.bar(x + width/2, p95_vals, width, label='P95')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    # Thermal slope comparison
    ax = axes[0, 1]
    slopes = [m.thermal_slope for m in metrics_list]
    colors = ['green' if s < 0.5 else 'orange' if s < 1.0 else 'red' for s in slopes]
    ax.bar(x, slopes, color=colors)
    ax.set_ylabel('Thermal Slope (°C/min)')
    ax.set_title('Thermal Increase Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Power comparison
    ax = axes[1, 0]
    power_vals = [m.power_mean for m in metrics_list]
    ax.bar(x, power_vals, color='green')
    ax.set_ylabel('Average Power (mW)')
    ax.set_title('Power Consumption')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Summary table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = []
    headers = ['Experiment', 'P50', 'P95', 'Thermal', 'Power']

    for i, (label, m) in enumerate(zip(labels, metrics_list)):
        table_data.append([
            label,
            f'{m.latency_p50:.1f}',
            f'{m.latency_p95:.1f}',
            f'{m.thermal_slope:.2f}',
            f'{m.power_mean:.0f}'
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig, axes


if __name__ == "__main__":
    import sys

    setup_style()

    if len(sys.argv) < 2:
        print("Usage: python plot_kpi.py <log_file.csv> [output_dir]")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    print(f"Loading: {file_path}")
    df = load_log(file_path)

    base_name = Path(file_path).stem

    # Generate all plots
    print("Generating dashboard...")
    plot_kpi_dashboard(df, save_path=f"{output_dir}/{base_name}_dashboard.png")

    print("Done!")
    plt.show()
