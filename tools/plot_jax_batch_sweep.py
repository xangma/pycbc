#!/usr/bin/env python3
"""Plot memory and uninstrumented timings from a batch-sweep receipt."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_sweep(data, output):
    variants = data["variants"]
    x = np.arange(len(variants))
    labels = [v["label"].replace(" / ", "\n") for v in variants]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.suptitle("JAX batch sweep · RTX 4090", fontsize=17, weight="bold")
    for offset, field, label, color in (
        (-0.18, "steady_live_gib", "Distinct live buffers at B6", "#26828e"),
        (0.18, "peak_gib", "Peak allocator use", "#425a9b"),
    ):
        values = [v["memory"][field] for v in variants]
        bars = axes[0].bar(x + offset, values, 0.34, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[0].set_ylabel("GPU memory (GiB)")
    axes[0].set_title("Memory from separate synchronized runs")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_ylim(0, max(v["memory"]["peak_gib"] for v in variants) * 1.3)
    for i, variant in enumerate(variants):
        rates = data["template_count"] / np.asarray(variant["calc_seconds"])
        median = float(np.median(rates))
        axes[1].bar(i, median, 0.6, color="#26828e", alpha=0.7)
        axes[1].scatter(i + np.linspace(-0.12, 0.12, len(rates)), rates,
                        color="#24324a", s=23, zorder=3)
        axes[1].text(i, max(rates) * 1.025, f"{median:.1f}", ha="center")
    axes[1].set_title("Throughput · median and all 3 runs")
    axes[1].set_ylabel("Bank templates / calculation second")
    axes[1].set_ylim(bottom=0)
    axes[1].margins(y=0.18)
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.18)
    fig.text(0.5, 0.025, data["figure_note"], ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_sweep(json.loads(args.input.read_text()), args.output)
