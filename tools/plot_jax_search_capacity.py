#!/usr/bin/env python3
"""Plot live filter proxies and inspiral capacity from retained receipts.

CPU capacity is per allocated host core; CUDA capacity uses that core AND a
whole GPU. The live panel models a block advance, not a full live executable.
"""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np


MICRO_ARMS = (
    ("branch_cpu", "Standard CPU / core", "#666666"),
    ("jax_cpu_lal", "JAX CPU / core", "#0072B2"),
    ("jax_cuda_lal", "JAX CUDA / GPU + host core", "#D55E00"),
)
SEARCH_ARMS = (
    ("branch_cpu", "Standard CPU B1\nper core"),
    ("jax_cpu_batched", "JAX CPU B16\nper core *"),
    ("jax_cuda_batched", "JAX CUDA B128\nper GPU + host core *"),
)


def capacity_summary(template_seconds, elapsed_seconds):
    """Median capacity and observed range from individual trials."""
    if template_seconds <= 0 or not elapsed_seconds:
        raise ValueError("Completed work and timing samples must be positive")
    if any(t <= 0 for t in elapsed_seconds):
        raise ValueError("Timing samples must be positive")
    rates = [template_seconds / t for t in elapsed_seconds]
    return median(rates), min(rates), max(rates)


def plot_live(data, output, advance_seconds=56.0):
    """Use filter samples, excluding waveform generation and transfers."""
    if not 0 < advance_seconds <= 64:
        raise ValueError("Live block advance must be in (0, 64] seconds")
    arms = data["experiments"]["streaming_n131072"]["arms"]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for key, label, color in MICRO_ARMS:
        cells = arms[key]["batches"]
        batches = sorted(map(int, cells))
        summaries = [capacity_summary(
            batch * advance_seconds,
            cells[str(batch)]["samples_seconds"]["filter"],
        ) for batch in batches]
        mid, low, high = np.array(summaries).T
        ax.errorbar(batches, mid, yerr=[mid - low, high - mid],
                    label=label, color=color, marker="o", capsize=4)
    ax.set(xscale="log", yscale="log", xticks=batches,
           xticklabels=list(map(str, batches)), xlabel="Template batch size",
           ylabel="Modelled real-time templates per allocated resource",
           title="pycbc_live-sized filter proxy (N = 131,072)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.text(0.5, 0.02,
             f"Assumed advance: {advance_seconds:g} s per 64 s block. "
             "Filter stage only; no vetoes or streaming pipeline.\n"
             "Three trials: median and observed range. "
             "Scientific equivalence not established.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_inspiral(data, output):
    """The published fixture contains 384 templates and 1904 valid seconds."""
    fig, ax = plt.subplots(figsize=(9, 5.8))
    x = np.arange(len(SEARCH_ARMS))
    for offset, field, label, color in (
        (-0.18, "wall_seconds", "Full executable wall time", "#0072B2"),
        (0.18, "calc_seconds", "Calculation interval", "#E69F00"),
    ):
        summaries = []
        for arm, _ in SEARCH_ARMS:
            samples = [run["timing"][field]
                       for name, run in data["runs"].items()
                       if name.startswith(arm + "_rep")]
            summaries.append(capacity_summary(384 * 1904, samples))
        mid, low, high = np.array(summaries).T
        bars = ax.bar(x + offset, mid, 0.36, label=label, color=color,
                      yerr=[mid - low, high - mid], capsize=4)
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in mid], padding=6)
    ax.set(yscale="log", ylim=(700, 900000), xticks=x,
           xticklabels=[label for _, label in SEARCH_ARMS],
           ylabel="Finite-workload real-time templates per allocated resource",
           title="pycbc_inspiral: 384 templates × 1,904 valid seconds")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.text(0.5, 0.02,
             "Three fresh processes per arm: median and observed range.\n"
             "* JAX arms fail CPU scientific qualification; "
             "these are descriptive rates, not equivalent-output speedups.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    static = Path("docs/_static")
    parser.add_argument("--microbenchmark", type=Path,
                        default=static / "jax_microbenchmarks.json")
    parser.add_argument("--search", type=Path,
                        default=static / "jax_search_comparison.json")
    parser.add_argument("--output-dir", type=Path, default=static)
    parser.add_argument("--live-advance-seconds", type=float, default=56.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_live(json.loads(args.microbenchmark.read_text()),
              args.output_dir / "jax_live_capacity.png",
              args.live_advance_seconds)
    plot_inspiral(json.loads(args.search.read_text()),
                  args.output_dir / "jax_inspiral_capacity.png")


if __name__ == "__main__":
    main()
