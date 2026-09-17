#!/usr/bin/env python3
"""Generate publication-grade comparison plots for PyCBC Torch benchmark suites."""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

COLORS = {
    "original_standard": "#4a5568",  # Slate gray
    "original_cpu": "#4a5568",       # Slate gray
    "branch_standard": "#3182ce",    # Standard Blue
    "branch_cpu": "#3182ce",         # Standard Blue
    "torch_cpu": "#dd6b20",          # Rust / Orange
    "torch_cuda_lal": "#805ad5",     # Purple / Violet (LAL on host)
    "torch_cuda": "#319795",         # Teal / Blue-green (RTX 4090 GPU)
    "torch_cuda_diffgw": "#38a169",  # Emerald Green (diffgw GPU)
}

LABELS = {
    "original_standard": "Original Standard CPU (40e94792b3)",
    "original_cpu": "Original Standard CPU (40e94792b3)",
    "branch_standard": "Branch Standard CPU (qualified)",
    "branch_cpu": "Branch Standard CPU (qualified)",
    "torch_cpu": "Torch CPU (MKL 1 thread)",
    "torch_cuda_lal": "Torch CUDA (LAL on host)",
    "torch_cuda": "Torch CUDA (RTX 4090)",
    "torch_cuda_diffgw": "Torch CUDA (DiffGW compiled)",
}


def plot_live_benchmarks(live_json: Path, output_png: Path):
    with open(live_json) as f:
        data = json.load(f)

    summary = data["summary_by_batch"]
    batches = sorted([int(k.split("_")[1]) for k in summary.keys()])
    routes = ["original_standard", "branch_standard", "torch_cpu", "torch_cuda"]

    fig, (ax_tp, ax_lat) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Throughput Plot (Log Scale)
    for route in routes:
        tps = [summary[f"batch_{b}"][route]["throughput_wps"]["p50"] for b in batches]
        lows = [summary[f"batch_{b}"][route]["throughput_wps"]["median_ci95"]["low"] for b in batches]
        highs = [summary[f"batch_{b}"][route]["throughput_wps"]["median_ci95"]["high"] for b in batches]
        ax_tp.plot(batches, tps, marker="o", linewidth=2, label=LABELS[route], color=COLORS[route])
        ax_tp.fill_between(batches, lows, highs, color=COLORS[route], alpha=0.15)

    ax_tp.set_xscale("log", base=2)
    ax_tp.set_yscale("log")
    ax_tp.set_xticks(batches)
    ax_tp.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_tp.set_xlabel("Batch Size $B$ (waveforms)")
    ax_tp.set_ylabel("Throughput (waveforms / second)")
    ax_tp.set_title("Live Batch Throughput vs Batch Size (N=131072)")
    ax_tp.legend(loc="upper left")

    # Annotate max speedup
    max_cuda_tp = summary["batch_32"]["torch_cuda"]["throughput_wps"]["p50"]
    base_tp = summary["batch_32"]["original_standard"]["throughput_wps"]["p50"]
    speedup = max_cuda_tp / base_tp
    ax_tp.annotate(
        f"{speedup:.1f}x Speedup\n({int(max_cuda_tp):,} wf/s)",
        xy=(32, max_cuda_tp),
        xytext=(14, 30000),
        arrowprops=dict(facecolor="#276749", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color="#276749",
    )

    # 2. Latency Plot (ms/block)
    for route in routes:
        lats = [summary[f"batch_{b}"][route]["latency_block_ms"]["p50"] for b in batches]
        ax_lat.plot(batches, lats, marker="s", linewidth=2, label=LABELS[route], color=COLORS[route])

    ax_lat.set_xscale("log", base=2)
    ax_lat.set_xticks(batches)
    ax_lat.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_lat.set_xlabel("Batch Size $B$ (waveforms)")
    ax_lat.set_ylabel("Block Latency (ms / 131072 samples)")
    ax_lat.set_title("Block Latency vs Batch Size")
    ax_lat.legend(loc="upper left")

    # Annotate flat CUDA latency
    cuda_lat_32 = summary["batch_32"]["torch_cuda"]["latency_block_ms"]["p50"]
    ax_lat.annotate(
        "0.79 ms/block (flat across batch sizes)",
        xy=(32, cuda_lat_32),
        xytext=(4, 18),
        arrowprops=dict(facecolor="#276749", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color="#276749",
    )

    fig.suptitle("PyCBC Live Batch Matched Filtering: 4-Route Performance Evidence", y=1.02)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_png}")


def plot_inspiral_workload_breakdown(inspiral_json: Path, output_png: Path):
    with open(inspiral_json) as f:
        data = json.load(f)

    summary = data["summary"]
    arms = ["original_cpu", "branch_cpu", "torch_cpu", "torch_cuda"]
    arm_labels = [LABELS[a] for a in arms]

    # Phases to plot
    phase_keys = [
        "startup_import_sec",
        "conditioning_sec",
        "waveform_prep_sec",
        "matched_filter_sec",
        "vetoes_clustering_sec",
        "serialization_io_sec",
    ]
    phase_names = [
        "Startup & Imports",
        "Data Conditioning (Frame + PSD)",
        "Waveform Prep",
        "Core Matched Filtering (calc_time)",
        "Vetoes & Clustering",
        "Serialization & I/O",
    ]
    phase_colors = ["#cbd5e0", "#4299e1", "#ed8936", "#48bb78", "#9f7aea", "#f56565"]

    matrix = np.zeros((len(phase_keys), len(arms)))
    for j, arm in enumerate(arms):
        for i, pk in enumerate(phase_keys):
            matrix[i, j] = summary[arm]["phases"][pk]["median"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bottom = np.zeros(len(arms))
    width = 0.55

    bars = []
    for i in range(len(phase_keys)):
        b = ax.bar(
            arm_labels,
            matrix[i],
            width,
            bottom=bottom,
            label=phase_names[i],
            color=phase_colors[i],
            edgecolor="white",
            linewidth=1,
        )
        bars.append(b)
        bottom += matrix[i]

    # Total wall time annotations on top of bars
    for j, arm in enumerate(arms):
        wall = summary[arm]["wall_sec"]["median"]
        spdup = summary[arm].get("speedup_wall_vs_original", 1.0)
        spdup_str = f" ({spdup:.2f}x)" if spdup != 1.0 else ""
        ax.text(j, wall + 1.5, f"{wall:.1f}s{spdup_str}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("pycbc_inspiral 6-Phase Workload Breakdown (384 Templates, 5 Segments H1 Data)")
    ax.set_ylim(0, 175)
    ax.legend(loc="upper left", ncol=2)
    plt.xticks(rotation=10, ha="right")

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_png}")


def plot_inspiral_calc_speedup(inspiral_json: Path, output_png: Path):
    with open(inspiral_json) as f:
        data = json.load(f)

    summary = data["summary"]
    arms = ["original_cpu", "branch_cpu", "torch_cpu", "torch_cuda"]
    arm_labels = ["Original\nStandard CPU", "Branch\nStandard CPU", "Torch CPU\n(MKL 1-thread)", "Torch CUDA\n(RTX 4090)"]

    calcs = [summary[a]["calc_time_sec"]["median"] for a in arms]
    walls = [summary[a]["wall_sec"]["median"] for a in arms]

    x = np.arange(len(arms))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # 1. Absolute Times (s)
    rects1 = ax1.bar(x - width/2, calcs, width, label="Filtering calc_time", color="#3182ce")
    rects2 = ax1.bar(x + width/2, walls, width, label="Process Wall Time", color="#4a5568")

    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Filtering calc_time vs Total Wall Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(arm_labels)
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 150)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}s", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}s", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Speedup Factor vs Original CPU
    base_calc = calcs[0]
    base_wall = walls[0]
    calc_spdups = [base_calc / c for c in calcs]
    wall_spdups = [base_wall / w for w in walls]

    r_sp1 = ax2.bar(x - width/2, calc_spdups, width, label="calc_time Speedup", color="#38a169")
    r_sp2 = ax2.bar(x + width/2, wall_spdups, width, label="Wall Time Speedup", color="#2b6cb0")
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1.5)

    ax2.set_ylabel("Speedup Factor vs Baseline CPU (x)")
    ax2.set_title("Acceleration Factors Relative to Original CPU")
    ax2.set_xticks(x)
    ax2.set_xticklabels(arm_labels)
    ax2.legend(loc="upper left")
    ax2.set_ylim(0, 8.5)

    for rect in r_sp1:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}x", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for rect in r_sp2:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}x", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=10)

    fig.suptitle("pycbc_inspiral: Matched Filtering & Process Acceleration", y=1.02)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_png}")


def plot_inspiral_batch_scaling(sweep_json: Path, output_png: Path):
    with open(sweep_json) as f:
        data = json.load(f)

    results = data["results"]
    batches = [r["batch_size"] for r in results]
    success_batches = [r["batch_size"] for r in results if r["status"] == "success"]

    calc_times = [r["calc_time_sec"] for r in results if r["status"] == "success"]
    vram_mib = [r["peak_vram_mib"] for r in results if r["status"] == "success"]

    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)

    color_calc = "#2b6cb0"
    ax1.set_xlabel("Batch Size $B$ (waveforms)")
    ax1.set_ylabel("Matched Filter calc_time (s)", color=color_calc)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(batches)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    line1 = ax1.plot(success_batches, calc_times, marker="o", color=color_calc, linewidth=2.5, label="calc_time (s)")
    ax1.tick_params(axis="y", labelcolor=color_calc)
    ax1.set_ylim(1.5, 2.5)

    # Secondary axis for Peak VRAM
    ax2 = ax1.twinx()
    color_vram = "#dd6b20"
    ax2.set_ylabel("Peak VRAM Allocated (GiB)", color=color_vram)
    vram_gib = [v / 1024.0 for v in vram_mib]
    line2 = ax2.plot(success_batches, vram_gib, marker="s", color=color_vram, linewidth=2.5, linestyle="--", label="Peak VRAM (GiB)")
    ax2.tick_params(axis="y", labelcolor=color_vram)
    ax2.set_ylim(0, 26)

    # 24GB VRAM ceiling line
    ax2.axhline(24.0, color="#e53e3e", linestyle=":", linewidth=2, label="RTX 4090 24 GiB VRAM Limit")

    # Mark OOM zone
    ax1.axvspan(384, 1024, color="#fed7d7", alpha=0.35, label="OOM Region (B >= 512)")
    ax1.text(600, 2.35, "OOM Region\n(Exceeds 24 GiB)", color="#9b2c2c", fontweight="bold", ha="center")

    for x_val, y_val in zip(success_batches, vram_gib):
        ax2.annotate(f"{y_val:.1f} GiB", xy=(x_val, y_val), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)

    for x_val, y_val in zip(success_batches, calc_times):
        ax1.annotate(f"{y_val:.2f}s", xy=(x_val, y_val), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=9)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("pycbc_inspiral RTX 4090 Batch Scaling & Memory Footprint (1024 Templates, N=524288)")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_png}")


def plot_inspiral_diffgw_speedup(diffgw_json: Path, output_png: Path):
    with open(diffgw_json) as f:
        data = json.load(f)

    summary = data["summary"]
    preferred_order = [
        "original_cpu",
        "branch_cpu",
        "torch_cpu",
        "torch_cpu_diffgw",
        "torch_cuda_lal",
        "torch_cuda_diffgw",
    ]
    arms = [a for a in preferred_order if a in summary]
    arm_label_map = {
        "original_cpu": "Original CPU\n(40e94792b3)",
        "branch_cpu": "Branch CPU\n(qualified)",
        "torch_cpu": "Torch CPU\n(LAL)",
        "torch_cpu_diffgw": "Torch CPU\n(DiffGW compiled)",
        "torch_cuda_lal": "Torch CUDA\n(LAL on host)",
        "torch_cuda_diffgw": "Torch CUDA\n(DiffGW compiled)",
    }
    arm_labels = [arm_label_map.get(a, a) for a in arms]

    wall_times = [summary[a]["wall_sec"]["median"] for a in arms]
    calc_times = [summary[a]["calc_time_sec"]["median"] for a in arms]
    wall_speedup = [summary[a].get("speedup_wall_vs_original", 1.0) for a in arms]
    calc_speedup = [summary[a].get("speedup_calc_vs_original", 1.0) for a in arms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Execution Times (seconds)
    x = np.arange(len(arms))
    width = 0.35
    b1 = ax1.bar(x - width / 2, calc_times, width, label="Calc Time (s)", color="#2b6cb0")
    b2 = ax1.bar(x + width / 2, wall_times, width, label="Process Wall Time (s)", color="#4a5568")

    ax1.set_ylabel("Execution Time (seconds)")
    ax1.set_title("pycbc_inspiral Execution Time (512 TaylorF2 Templates)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(arm_labels, fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(loc="upper right")

    for rect in b1:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.1f}s",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for rect in b2:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.1f}s",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2. Speedup Factor vs Original CPU Baseline
    r1 = ax2.bar(x - width / 2, calc_speedup, width, label="Calc Time Speedup", color="#2b6cb0")
    r2 = ax2.bar(x + width / 2, wall_speedup, width, label="Process Wall Speedup", color="#38a169")

    ax2.set_ylabel("Speedup Factor vs Original CPU")
    ax2.set_title("Track 2 Speedup Factor vs Baseline (RTX 4090 + diffgw)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(arm_labels, fontsize=9)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left")

    for rect in r1:
        h = rect.get_height()
        ax2.annotate(
            f"{h:.2f}x",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for rect in r2:
        h = rect.get_height()
        ax2.annotate(
            f"{h:.2f}x",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("Track 2: Dynamic Waveform Synthesis Acceleration with diffgw & PyTorch CUDA", y=1.02)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison plots.")
    parser.add_argument("--live-json", type=Path, default=None)
    parser.add_argument("--inspiral-json", type=Path, default=None)
    parser.add_argument("--sweep-json", type=Path, default=None)
    parser.add_argument("--diffgw-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.live_json and args.live_json.is_file():
        plot_live_benchmarks(args.live_json, args.output_dir / "pycbc_live_throughput_latency.png")
    if args.inspiral_json and args.inspiral_json.is_file():
        plot_inspiral_workload_breakdown(args.inspiral_json, args.output_dir / "pycbc_inspiral_workload_breakdown.png")
        plot_inspiral_calc_speedup(args.inspiral_json, args.output_dir / "pycbc_inspiral_calc_speedup.png")
    if args.sweep_json and args.sweep_json.is_file():
        plot_inspiral_batch_scaling(args.sweep_json, args.output_dir / "pycbc_inspiral_batch_scaling.png")
    if args.diffgw_json and args.diffgw_json.is_file():
        plot_inspiral_diffgw_speedup(args.diffgw_json, args.output_dir / "pycbc_inspiral_diffgw_speedup.png")


if __name__ == "__main__":
    main()
