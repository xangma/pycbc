#!/usr/bin/env python3
# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Generate publication-quality benchmark plots from sealed JAX performance receipts.

Outputs:
- jax_throughput_scaling.png: Calculation rate vs batch size across workloads
- jax_speedup_matrix.png: Relative speedup vs baseline CPU
- jax_latency_breakdown.png: Per-template latency composition (waveform, transfer, filter)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


ARM_CONFIG = {
    "branch_cpu": {
        "label": "Current CPU (LAL)",
        "color": "#777777",
        "linestyle": ":",
        "marker": "s",
    },
    "jax_cpu_lal": {
        "label": "JAX CPU (LAL)",
        "color": "#1f77b4",
        "linestyle": "--",
        "marker": "^",
    },
    "jax_cpu_diffgw": {
        "label": "JAX CPU (diffgw)",
        "color": "#17becf",
        "linestyle": "-",
        "marker": "D",
    },
    "jax_cuda_lal": {
        "label": "JAX CUDA (LAL + H2D)",
        "color": "#ff7f0e",
        "linestyle": "--",
        "marker": "v",
    },
    "jax_cuda_diffgw": {
        "label": "JAX CUDA (diffgw on-device)",
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": "*",
    },
}


def plot_throughput_scaling(data: Dict[str, Any], output_path: Path):
    """Dual-panel scaling figure for short and long synthetic transforms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    tracks = [
        ("streaming_n131072", "Short-transform microbenchmark (N = 131,072, 2048 Hz)", axes[0]),
        ("inspiral_n2097152", "Long-transform microbenchmark (N = 2,097,152, 4096 Hz)", axes[1]),
    ]

    for track_key, track_title, ax in tracks:
        exp = data.get("experiments", {}).get(track_key, {})
        arms = exp.get("arms", {})

        if not arms:
            # Fallback if only legacy keys exist
            cpu_base = exp.get("cpu_baseline", {})
            jax_scale = exp.get("jax_scaling", {})
            if cpu_base and jax_scale:
                base_tps = cpu_base.get("templates_per_sec", 1.0)
                ax.axhline(base_tps, color="#333333", linestyle="--", label="CPU Baseline (1 thread)")
                batches = sorted([int(k) for k in jax_scale.get("batches", {}).keys()])
                tps = [jax_scale["batches"][str(b)]["templates_per_sec"] for b in batches]
                ax.plot(batches, tps, marker="*", color="#2ca02c", linewidth=2, label="JAX GPU (diffgw)")
        else:
            for arm_key, arm_style in ARM_CONFIG.items():
                if arm_key not in arms:
                    continue
                arm_data = arms[arm_key]
                batches = sorted([int(k) for k in arm_data.get("batches", {}).keys()])
                if not batches:
                    continue
                tps = [arm_data["batches"][str(b)]["templates_per_sec"] for b in batches]
                ax.plot(
                    batches,
                    tps,
                    label=arm_style["label"],
                    color=arm_style["color"],
                    linestyle=arm_style["linestyle"],
                    marker=arm_style["marker"],
                    linewidth=2.2 if "cuda" in arm_key else 1.8,
                    markersize=8,
                )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Template Batch Size (B)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Waveform + transfer + filter rate (templates / sec)", fontsize=12, fontweight="bold")
        ax.set_title(track_title, fontsize=13, fontweight="bold", pad=12)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_speedup_matrix(data: Dict[str, Any], output_path: Path):
    """Grouped bar chart displaying relative speedup vs CPU baseline."""
    exp = data.get("experiments", {}).get("streaming_n131072", {})
    arms = exp.get("arms", {})
    if not arms:
        print("Speedup matrix skipped: no arms in artifact")
        return

    baseline = arms.get("branch_cpu", {}).get("batches", {}).get("1", {})
    baseline_tps = baseline.get("templates_per_sec", 0)
    if baseline_tps <= 0:
        print("Speedup matrix skipped: no current CPU batch-1 reference")
        return

    eval_batches = [1, 16, 64]
    active_arms = [a for a in ARM_CONFIG.keys() if a in arms]

    x = np.arange(len(active_arms))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, b in enumerate(eval_batches):
        speedups = []
        for arm in active_arms:
            b_str = str(b)
            if b_str in arms[arm].get("batches", {}):
                speedups.append(arms[arm]["batches"][b_str]["templates_per_sec"] / baseline_tps)
            else:
                speedups.append(0.0)

        bars = ax.bar(x + (idx - 1) * width, speedups, width, label=f"Batch Size B={b}")
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}×" if height >= 2 else f"{height:.2f}×",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45 if height >= 100 else 0,
                )

    ax.axhline(1.0, color="black", linestyle=":", alpha=0.7, label="Baseline (1.0×)")
    ax.set_yscale("log")
    ax.set_ylabel("Speedup vs current CPU (B=1)", fontsize=12, fontweight="bold")
    ax.set_title("Relative Acceleration (Short-transform Microbenchmark)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_CONFIG[a]["label"] for a in active_arms], rotation=25, ha="right", fontsize=10)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, which="major", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_latency_breakdown(data: Dict[str, Any], output_path: Path):
    """Stacked bar chart showing per-template time spent in waveform, transfer, and filter."""
    exp = data.get("experiments", {}).get("streaming_n131072", {})
    arms = exp.get("arms", {})
    if not arms:
        print("Latency breakdown skipped: no arms in artifact")
        return

    # Select representative batch size
    target_b = "16"
    active_arms = [a for a in ARM_CONFIG.keys() if a in arms and target_b in arms[a].get("batches", {})]
    if not active_arms:
        active_arms = [a for a in ARM_CONFIG.keys() if a in arms and "1" in arms[a].get("batches", {})]
        target_b = "1"

    b_val = int(target_b)
    wf_times = []
    tr_times = []
    filt_times = []

    for arm in active_arms:
        b_data = arms[arm]["batches"][target_b]
        wf_times.append((b_data.get("median_waveform_sec", 0.0) / b_val) * 1000.0)
        tr_times.append((b_data.get("median_transfer_sec", 0.0) / b_val) * 1000.0)
        filt_times.append((b_data.get("median_filter_sec", 0.0) / b_val) * 1000.0)

    y_pos = np.arange(len(active_arms))

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.barh(y_pos, wf_times, color="#4c72b0", label="Waveform Synthesis")
    ax.barh(y_pos, tr_times, left=wf_times, color="#c44e52", label="Array Transfer")
    left_sum = [w + t for w, t in zip(wf_times, tr_times)]
    ax.barh(y_pos, filt_times, left=left_sum, color="#55a868", label="Matched Filtering Kernel")

    ax.set_xlim(left=0)
    ax.set_xlabel("Per-Template Latency (ms / template)", fontsize=12, fontweight="bold")
    ax.set_title(f"Per-Template Latency (B={target_b}, Short-transform Microbenchmark)", fontsize=14, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([ARM_CONFIG[a]["label"] for a in active_arms], fontsize=10)
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, which="major", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot PyCBC JAX Performance Scaling Figures")
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/jax_benchmark_results.json",
        help="Path to sealed benchmark JSON artifact",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/_static",
        help="Directory to save generated PNG figures",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Benchmark artifact not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_throughput_scaling(data, out_dir / "jax_throughput_scaling.png")
    plot_speedup_matrix(data, out_dir / "jax_speedup_matrix.png")
    plot_latency_breakdown(data, out_dir / "jax_latency_breakdown.png")


if __name__ == "__main__":
    main()
