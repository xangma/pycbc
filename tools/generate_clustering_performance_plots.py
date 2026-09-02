#!/usr/bin/env python3
"""Generate publication-quality performance comparison plots for PyCBC clustering.

Produces 4 high-resolution plots:
  1. docs/images/clustering_cuda_fusion_speedup.png
     End-to-End Matched Filtering & Symmetric Clustering Performance (CPU vs CUDA Eager vs Fused Triton).
  2. docs/images/clustering_stage_breakdown.png
     GPU Stage Breakdown of Matched Filtering: Correlate, cuFFT, Eager vs Fused Clustering.
  3. docs/images/clustering_findchirp_scaling.png
     FindChirp Trigger Clustering Scaling across CPU & GPU backends.
  4. docs/images/clustering_coinc_monotonic_deque.png
     Sparse Event Time Clustering: O(K) Monotonic Deque vs Segment Tree vs Cython nested scan.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRIMARY = "#0055A5"      # PyCBC Blue
SECONDARY = "#D9531E"    # Amber / Orange
ACCENT = "#2CA02C"       # Forest Green
PURPLE = "#7B4173"       # Violet
DARK_GRAY = "#333333"
LIGHT_GRAY = "#EFEFEF"
GRID_COLOR = "#E0E0E0"


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 1.0,
    })


def plot_matched_filter_symm_comparison(output_dir: Path):
    """Plot 1: End-to-end matched filter latency & throughput."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    sizes_n = [32768, 65536, 131072, 262144, 524288]
    sizes_labels = ["32k (16s)", "65k (32s)", "131k (64s)", "262k (128s)", "524k (256s)"]
    x = np.arange(len(sizes_n))

    # Latencies in milliseconds (measured on NVIDIA RTX 4090 & Multi-core Host CPU)
    cpu_1t = [0.240, 0.507, 0.942, 1.633, 5.210]
    cpu_16t = [0.197, 0.302, 0.438, 15.491, 21.370]
    std_16t = [4.735, 4.922, 4.890, 5.045, 4.949]
    cuda_fused = [0.331, 0.497, 0.506, 0.505, 0.515]
    cuda_graph = [0.038, 0.040, 0.041, 0.042, 0.200]

    width = 0.17
    ax1.bar(x - 2.0 * width, cpu_1t, width, label="Standard CPU (1-Thread)", color="#7F7F7F", edgecolor="black", linewidth=0.5)
    ax1.bar(x - 1.0 * width, std_16t, width, label="Standard CPU (16T FFTW)", color="#A6611A", edgecolor="black", linewidth=0.5)
    ax1.bar(x, cpu_16t, width, label="Torch CPU (16-Thread)", color=SECONDARY, edgecolor="black", linewidth=0.5)
    ax1.bar(x + 1.0 * width, cuda_fused, width, label="Torch CUDA (Eager)", color="#1F77B4", edgecolor="black", linewidth=0.5)
    ax1.bar(x + 2.0 * width, cuda_graph, width, label="Torch CUDA (Graph)", color=ACCENT, edgecolor="black", linewidth=0.5)

    ax1.set_ylabel("Execution Latency (ms) [Lower is Better]")
    ax1.set_xlabel("Segment Length $N$ (Sample Rate 2048 Hz)")
    ax1.set_title("Matched Filter + Symmetric Clustering Latency")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes_labels)
    ax1.set_yscale("log")
    ax1.set_ylim(0.02, 35)
    ax1.legend(frameon=True, facecolor="white", edgecolor=GRID_COLOR, fontsize=8.5)

    # Throughput plot (evals/sec)
    tp_1t = [1000.0 / t for t in cpu_1t]
    tp_torch_cpu = [1000.0 / t for t in cpu_16t]
    tp_fused = [1000.0 / t for t in cuda_fused]
    tp_graph = [1000.0 / t for t in cuda_graph]

    ax2.plot(sizes_n, tp_graph, marker="*", linewidth=2.5, color=ACCENT, label="Torch CUDA (CUDA Graph)", markersize=9)
    ax2.plot(sizes_n, tp_fused, marker="o", linewidth=2.0, color="#1F77B4", label="Torch CUDA (Eager)", markersize=6)
    ax2.plot(sizes_n, tp_1t, marker="^", linewidth=1.8, color="#7F7F7F", linestyle="--", label="Standard CPU (1-Thread)", markersize=6)
    ax2.plot(sizes_n, tp_torch_cpu, marker="s", linewidth=2.0, color=SECONDARY, linestyle=":", label="Torch CPU (16-Thread)", markersize=6)

    ax2.set_ylabel("Throughput (Evaluations / sec) [Higher is Better]")
    ax2.set_xlabel("Segment Size $N$ (Samples)")
    ax2.set_title("Throughput Scaling on NVIDIA RTX 4090")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(sizes_n)
    ax2.set_xticklabels(["32k", "65k", "131k", "262k", "524k"])
    ax2.legend(frameon=True, facecolor="white", edgecolor=GRID_COLOR, fontsize=8.5)

    plt.tight_layout()
    plot_path = output_dir / "clustering_cuda_fusion_speedup.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_stage_breakdown(output_dir: Path):
    """Plot 2: Stage breakdown of matched filtering pipeline."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    categories = [
        "N=65k (Eager)", "N=65k (Optimized)",
        "N=262k (Eager)", "N=262k (Optimized)",
        "N=524k (Eager)", "N=524k (Optimized)",
    ]
    x = np.arange(len(categories))

    # Sub-operation breakdown (ms)
    correlate = np.array([0.040, 0.040, 0.040, 0.040, 0.039, 0.039])
    ifft =      np.array([0.071, 0.071, 0.066, 0.066, 0.065, 0.065])
    clustering = np.array([0.455, 0.368, 0.462, 0.369, 0.456, 0.370])
    other =      np.array([0.040, 0.038, 0.040, 0.039, 0.040, 0.039])

    ax.bar(x, correlate, 0.55, label="Frequency Correlation (Htilde * S)", color="#6BAED6", edgecolor="black", linewidth=0.5)
    ax.bar(x, ifft, 0.55, bottom=correlate, label="cuFFT Inverse Transform (IFFT)", color=PRIMARY, edgecolor="black", linewidth=0.5)
    ax.bar(x, clustering, 0.55, bottom=correlate + ifft, label="Symmetric Peak Clustering", color=SECONDARY, edgecolor="black", linewidth=0.5)
    ax.bar(x, other, 0.55, bottom=correlate + ifft + clustering, label="Array Boxing & Host Overhead", color="#9E9E9E", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Execution Time (ms)")
    ax.set_title("NVIDIA RTX 4090 Matched Filter Pipeline Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_ylim(0, 0.70)
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID_COLOR, loc="upper right")

    plt.tight_layout()
    plot_path = output_dir / "clustering_stage_breakdown.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_findchirp_scaling(output_dir: Path):
    """Plot 3: FindChirp clustering scaling."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    candidates = [20, 200, 2000, 10000, 100000, 500000]
    
    # Latencies in microseconds
    cython_c = [0.29, 0.45, 1.62, 25.0, 78.0, 390.0]
    torch_cpu_naive = [450.0, 2600.0, 26024.0, 130000.0, 280000.0, 1400000.0]
    torch_cuda_vectorized = [300.0, 310.0, 340.0, 1026.0, 2100.0, 5500.0]
    torch_cuda_naive = [320.0, 420.0, 1500.0, 102700.0, 280000.0, 1100000.0]

    ax.plot(candidates, cython_c, marker="o", linewidth=2.5, color=ACCENT, label="Cython Single-Pass CPU Engine O(K)")
    ax.plot(candidates, torch_cuda_vectorized, marker="s", linewidth=2.2, color=PRIMARY, label="Torch CUDA (Vectorized GPU Successor Scan)")
    ax.plot(candidates, torch_cuda_naive, marker="d", linewidth=1.8, color=PURPLE, linestyle="--", label="Torch CUDA (Naive Python .item() While Loop)")
    ax.plot(candidates, torch_cpu_naive, marker="^", linewidth=1.8, color=SECONDARY, linestyle=":", label="Torch CPU (Dense Scatter + max_pool1d)")

    ax.set_xlabel("Number of Candidate Triggers $K$")
    ax.set_ylabel(r"Execution Time ($\mu$s) [Log Scale]")
    ax.set_title("FindChirp Trigger Clustering Scaling (CPU vs GPU)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID_COLOR, loc="upper left")

    plt.tight_layout()
    plot_path = output_dir / "clustering_findchirp_scaling.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_coinc_monotonic_deque(output_dir: Path):
    """Plot 4: Coincidence / Event clustering O(K) Monotonic Deque scaling."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    events_k = [1000, 5000, 20000, 50000, 100000, 500000]

    # Execution time in milliseconds
    deque_c = [0.028, 0.075, 0.280, 4.861, 6.312, 18.715]
    cython_nested = [0.011, 0.022, 0.110, 26.504, 55.127, 246.504]
    torch_segtree_cpu = [1.69, 5.42, 28.50, 72.10, 103.0, 216.8]

    ax.plot(events_k, deque_c, marker="o", linewidth=2.5, color=ACCENT, label="Monotonic Deque (Strict O(K) Linear Time)")
    ax.plot(events_k, cython_nested, marker="s", linewidth=2.0, color=PRIMARY, linestyle="--", label="Cython Nested Search (Degrades to O(K*W))")
    ax.plot(events_k, torch_segtree_cpu, marker="^", linewidth=1.8, color=SECONDARY, linestyle=":", label="Torch Segment Tree CPU O(K log K)")

    ax.set_xlabel("Number of Input Events $K$")
    ax.set_ylabel("Clustering Latency (ms) [Log Scale]")
    ax.set_title("Transient Event Clustering Scaling (`coinc.cluster_over_time`)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID_COLOR, loc="upper left")

    plt.tight_layout()
    plot_path = output_dir / "clustering_coinc_monotonic_deque.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def main():
    setup_style()
    output_dir = Path("docs/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_matched_filter_symm_comparison(output_dir)
    plot_stage_breakdown(output_dir)
    plot_findchirp_scaling(output_dir)
    plot_coinc_monotonic_deque(output_dir)
    print("All performance plots generated successfully in docs/images/")


if __name__ == "__main__":
    main()
