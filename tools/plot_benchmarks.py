#!/usr/bin/env python3
"""Runner to generate and update benchmark plots for PyCBC Torch suite."""

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure repository root is in sys.path before local package imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402

from tools.plot_benchmark_results import (  # noqa: E402
    plot_inspiral_batch_scaling,
    plot_inspiral_calc_speedup,
    plot_inspiral_diffgw_speedup,
    plot_inspiral_workload_breakdown,
    plot_live_benchmarks,
)

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("plot_benchmarks")


def plot_approximant_sweep_comparison(
    taylor_json: Path,
    imr_json: Path,
    output_png: Path,
):
    """Plot multi-approximant batch scaling comparison."""
    datasets = {}
    if taylor_json.is_file():
        with open(taylor_json) as f:
            datasets["TaylorF2"] = json.load(f)
    if imr_json.is_file():
        with open(imr_json) as f:
            datasets["IMRPhenomD"] = json.load(f)

    if not datasets:
        logger.warning("No sweep datasets found for multi-approximant plot.")
        return

    fig, (ax_calc, ax_vram, ax_tp) = plt.subplots(
        1, 3, figsize=(18, 5.5), dpi=300
    )

    approx_styles = {
        "TaylorF2": {
            "color": "#3182ce",
            "marker": "o",
            "label": "TaylorF2 (512 tmpl)",
        },
        "IMRPhenomD": {
            "color": "#dd6b20",
            "marker": "s",
            "label": "IMRPhenomD (512 tmpl)",
        },
    }

    all_batches = sorted({
        r["batch_size"]
        for d in datasets.values()
        for r in d.get("results", [])
    })

    for name, data in datasets.items():
        results = data.get("results", [])
        style = approx_styles.get(
            name, {"color": "#718096", "marker": "^", "label": name}
        )

        succ = [r for r in results if r.get("status") == "success"]
        fail = [r for r in results if r.get("status") != "success"]

        b_succ = [r["batch_size"] for r in succ]
        calc = [r["calc_time_sec"] for r in succ]
        vram_gib = [r["peak_vram_mib"] / 1024.0 for r in succ]
        tp_calc = [r["templates_per_sec_calc"] for r in succ]

        # 1. Calc time
        ax_calc.plot(
            b_succ,
            calc,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            label=style["label"],
        )
        for x_val, y_val in zip(b_succ, calc):
            ax_calc.annotate(
                f"{y_val:.2f}s",
                xy=(x_val, y_val),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )

        # 2. Peak VRAM
        ax_vram.plot(
            b_succ,
            vram_gib,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            label=style["label"],
        )
        for x_val, y_val in zip(b_succ, vram_gib):
            ax_vram.annotate(
                f"{y_val:.1f} GiB",
                xy=(x_val, y_val),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )

        # Mark failures on VRAM plot
        for r in fail:
            b_fail = r["batch_size"]
            ax_vram.scatter(
                [b_fail], [24.0], color="red", marker="x", s=80, zorder=5
            )
            ax_vram.annotate(
                f"OOM (B={b_fail})",
                xy=(b_fail, 23.5),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                color="#c53030",
                fontweight="bold",
                fontsize=8.5,
            )

        # 3. Throughput (templates/s)
        ax_tp.plot(
            b_succ,
            tp_calc,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            label=style["label"],
        )
        for x_val, y_val in zip(b_succ, tp_calc):
            ax_tp.annotate(
                f"{y_val:.0f} t/s",
                xy=(x_val, y_val),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )

    # Format Axes
    for ax in (ax_calc, ax_vram, ax_tp):
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_batches)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel("Batch Size $B$ (waveforms)")
        ax.legend(loc="best")

    ax_calc.set_ylabel("Matched Filter calc_time (s)")
    ax_calc.set_title("Calculation Time vs Batch Size")
    ax_calc.set_ylim(bottom=0)

    ax_vram.set_ylabel("Peak VRAM Allocated (GiB)")
    ax_vram.set_title("VRAM Footprint vs 24 GiB Ceiling")
    ax_vram.axhline(
        24.0,
        color="#e53e3e",
        linestyle=":",
        linewidth=2,
        label="RTX 4090 24 GiB Limit",
    )
    ax_vram.set_ylim(0, 26)
    ax_vram.legend(loc="upper left")

    ax_tp.set_ylabel("Throughput (templates/sec)")
    ax_tp.set_title("Matched Filtering Throughput")
    ax_tp.set_ylim(bottom=0)

    fig.suptitle(
        "GPU Inspiral Batch Scaling: TaylorF2 vs IMRPhenomD "
        "(512 Templates, RTX 4090)",
        y=1.02,
    )
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and update PyCBC benchmark plots."
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=Path("artifacts/benchmarks-20260917"),
        help="Path to benchmark JSON directory "
             "(default: artifacts/benchmarks-20260917)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: <benchmarks-dir>/plots)",
    )
    args = parser.parse_args()

    benchmarks_dir = args.benchmarks_dir
    output_dir = args.output_dir or (benchmarks_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    live_json = benchmarks_dir / "live_benchmark_results.json"
    inspiral_json = benchmarks_dir / "inspiral_campaign_results.json"
    sweep_json = benchmarks_dir / "inspiral_batch_sweep.json"
    diffgw_json = benchmarks_dir / "inspiral_diffgw_campaign_results.json"
    taylor_json = benchmarks_dir / "inspiral_taylorf2_sweep_512.json"
    imr_json = benchmarks_dir / "inspiral_imrphenomd_sweep_512.json"

    # 1. Standard campaign plots if available
    if live_json.is_file():
        logger.info(f"Plotting live benchmarks from {live_json}...")
        plot_live_benchmarks(
            live_json, output_dir / "pycbc_live_throughput_latency.png"
        )

    if inspiral_json.is_file():
        logger.info(
            f"Plotting inspiral workload breakdown from {inspiral_json}..."
        )
        plot_inspiral_workload_breakdown(
            inspiral_json,
            output_dir / "pycbc_inspiral_workload_breakdown.png",
        )
        logger.info(f"Plotting inspiral calc speedup from {inspiral_json}...")
        plot_inspiral_calc_speedup(
            inspiral_json, output_dir / "pycbc_inspiral_calc_speedup.png"
        )

    if sweep_json.is_file():
        logger.info(f"Plotting inspiral batch scaling from {sweep_json}...")
        plot_inspiral_batch_scaling(
            sweep_json, output_dir / "pycbc_inspiral_batch_scaling.png"
        )

    if diffgw_json.is_file():
        logger.info(f"Plotting inspiral diffgw speedup from {diffgw_json}...")
        plot_inspiral_diffgw_speedup(
            diffgw_json, output_dir / "pycbc_inspiral_diffgw_speedup.png"
        )

    # 2. Multi-approximant batch sweep comparison (512 templates)
    if taylor_json.is_file() or imr_json.is_file():
        logger.info("Plotting multi-approximant batch sweep comparison...")
        plot_approximant_sweep_comparison(
            taylor_json,
            imr_json,
            output_dir / "pycbc_inspiral_approximant_batch_scaling.png",
        )

    logger.info("All benchmark plots updated successfully.")


if __name__ == "__main__":
    main()
