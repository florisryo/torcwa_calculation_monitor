from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil


def _load_smatrix_windows(events_path: Path) -> Dict[int, Tuple[float, float]]:
    with events_path.open("r", encoding="utf-8") as f:
        events = json.load(f)

    windows: Dict[int, Dict[str, float]] = {}
    for event in events:
        label = event.get("label", "")
        if not label.startswith("example2_order_smatrix_"):
            continue
        order = event.get("metadata", {}).get("order")
        if order is None:
            continue
        suffix = label.rsplit("_", 1)[-1]
        windows.setdefault(int(order), {})[suffix] = float(event.get("time_s", 0.0))

    result: Dict[int, Tuple[float, float]] = {}
    for order, times in windows.items():
        start = times.get("start")
        end = times.get("end")
        if start is None or end is None or end <= start:
            continue
        result[order] = (start, end)
    return result


def _estimate_interval(df: pd.DataFrame) -> float:
    diffs = df["elapsed_s"].diff().dropna()
    if diffs.empty:
        return 0.5
    positive = diffs[diffs > 0]
    if positive.empty:
        return float(abs(diffs.iloc[0]))
    return float(positive.median())


def summarize(log_path: Path, events_path: Path, orders: Iterable[int]) -> pd.DataFrame:
    df = pd.read_csv(log_path)
    windows = _load_smatrix_windows(events_path)
    logical_cpus = psutil.cpu_count(logical=True) or 1
    fallback_interval = _estimate_interval(df)

    order_list = [int(o) for o in orders if int(o) in windows] if orders else sorted(windows)

    records: List[Dict[str, float]] = []
    for order in order_list:
        start, end = windows[order]
        mask = (df["elapsed_s"] >= start) & (df["elapsed_s"] <= end)
        window_df = df.loc[mask]

        if window_df.empty:
            lower = max(0.0, start - fallback_interval)
            upper = end + fallback_interval
            window_df = df.loc[(df["elapsed_s"] >= lower) & (df["elapsed_s"] <= upper)]

        if window_df.empty:
            idx = (df["elapsed_s"] - start).abs().idxmin()
            window_df = df.loc[[idx]]

        if window_df.empty:
            continue

        duration = end - start
        gpu_total = window_df["gpu_memory_total_mb"].dropna().iloc[0] if window_df["gpu_memory_total_mb"].notna().any() else float("nan")
        gpu_mean_mb = float(window_df["gpu_memory_mb"].mean())
        if window_df["gpu_memory_percent"].notna().any():
            gpu_percent = float(window_df["gpu_memory_percent"].mean())
        else:
            gpu_percent = (gpu_mean_mb / gpu_total * 100.0) if gpu_total else float("nan")
        cpu_mean = float(window_df["cpu_percent"].mean())
        ram_mean = float(window_df["system_memory_percent"].mean())
        records.append(
            {
                "order": order,
                "start_s": start,
                "end_s": end,
                "duration_s": duration,
                "gpu_mean_mb": gpu_mean_mb,
                "gpu_total_mb": gpu_total,
                "gpu_percent": gpu_percent,
                "gpu_effective_percent": gpu_percent,
                "cpu_mean_percent": cpu_mean,
                "cpu_effective_percent": cpu_mean / logical_cpus,
                "ram_mean_percent": ram_mean,
                "ram_effective_percent": ram_mean,
            }
        )

    summary_df = pd.DataFrame(records)
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values("start_s").reset_index(drop=True)


def make_table(df: pd.DataFrame) -> str:
    rows = [
        "|  N  | Time [s] | GPU VRAM [MB] (used / total) -> effective[%] | CPU Util (avg %) (effective %) | RAM (avg %) (effective %) |",
        "| :-: | :----: | :---------------------------------: | :-------------------------: | :---------------------: |",
    ]
    for _, row in df.iterrows():
        order = int(row["order"])
        duration = row["duration_s"]
        gpu_used = row["gpu_mean_mb"]
        gpu_total = row["gpu_total_mb"]
        gpu_pct = row["gpu_percent"]
        cpu_avg = row["cpu_mean_percent"]
        cpu_eff = row["cpu_effective_percent"]
        ram_avg = row["ram_mean_percent"]
        ram_eff = row["ram_effective_percent"]
        gpu_cell = "N/A" if pd.isna(gpu_total) else f"{gpu_used:.0f} / {gpu_total:.0f} -> {gpu_pct:.1f} %"
        cpu_cell = f"{cpu_avg:.1f} (effective ~= {cpu_eff:.1f} %)"
        ram_cell = f"{ram_avg:.1f} -> {ram_eff:.1f} %"
        rows.append(f"| {order:3d} | {duration:.2f} | {gpu_cell:^33} | {cpu_cell:^25} | {ram_cell:^21} |")
    return "\n".join(rows)


def _assign_text_levels(starts: List[float], log_threshold: float = 0.15) -> List[int]:
    placed: List[Tuple[float, int]] = []
    levels: List[int] = []
    for start in starts:
        safe_start = max(start, 1e-6)
        log_start = np.log10(safe_start)
        level = 0
        while any(abs(log_start - prev_log) < log_threshold and prev_level == level for prev_log, prev_level in placed):
            level += 1
        placed.append((log_start, level))
        levels.append(level)
    return levels


def plot_metrics(df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("GPU VRAM", "gpu_percent", "gpu_effective_percent", "#d62728"),
        ("CPU Util", "cpu_mean_percent", "cpu_effective_percent", "#1f77b4"),
        ("System RAM", "ram_mean_percent", "ram_effective_percent", "#2ca02c"),
    ]

    starts_raw = df["start_s"].to_numpy()
    orders = df["order"].astype(int).to_numpy()

    starts = starts_raw.copy()
    positive = starts > 0
    if positive.any():
        min_positive = starts[positive].min()
        starts[~positive] = min_positive * 0.5
    else:
        starts[~positive] = 1e-3

    fig, axes = plt.subplots(len(metrics), 1, sharex=True, figsize=(12, 9))

    for ax, (title, avg_col, eff_col, color) in zip(axes, metrics):
        avg_values = df[avg_col].to_numpy()
        eff_values = df[eff_col].to_numpy()

        ax.plot(starts, avg_values, marker="o", color=color, label="Average %")
        #ax.plot(starts, eff_values, marker="s", linestyle="--", color=color, alpha=0.7, label="Effective %")

        for xi, yi, order in zip(starts, avg_values, orders):
            ax.annotate(
                f"N={order}",
                xy=(xi, yi),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
            )

        ax.set_ylabel("Percent [%]")
        ax.set_title(title)
        y_max = max(100.0, float(np.nanmax([avg_values.max(), eff_values.max()])) + 5.0)
        ax.set_ylim(0, y_max)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")
        ax.set_xscale("log")

    line_levels = _assign_text_levels(starts.tolist())
    for ax in axes:
        for start in starts:
            ax.axvline(start, color="#888888", linestyle=":", linewidth=0.8, alpha=0.35, zorder=0)

    bottom_ax = axes[-1]
    y_min, y_max = bottom_ax.get_ylim()
    offset_step = max((y_max - y_min) * 0.04, 6.0)

    axes[-1].set_xlabel("Elapsed Time [s]")
    axes[-1].set_xlim(left=starts.min() * 0.8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and visualize resource usage per diffraction order.")
    parser.add_argument("--log", type=Path, default=Path("benchmarks/output/order/resource_log.csv"), help="Path to resource_log.csv")
    parser.add_argument("--events", type=Path, default=Path("benchmarks/output/order/events.json"), help="Path to events.json")
    parser.add_argument("--orders", type=int, nargs="*", help="Optional subset of orders to include")
    parser.add_argument("--plot-path", type=Path, help="Output path for the generated plot")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    df = summarize(args.log, args.events, args.orders or [])
    if df.empty:
        print("No matching S-matrix windows found for the provided inputs.")
        return

    print(make_table(df))

    if not args.no_plot:
        default_plot = args.log.parent / "resource_summary.png"
        plot_path = args.plot_path or default_plot
        plot_metrics(df, plot_path)
        print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    main()
