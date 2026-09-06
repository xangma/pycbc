#!/usr/bin/env python3
"""Render concise docs figures from the pinned September 2026 evidence archive.

This is a presentation-only renderer. It reads verified archive summaries,
checks their SHA-256 hashes, and never imports PyCBC or runs a benchmark.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator, StrMethodFormatter  # noqa: E402

BLUE = "#2878B5"
ORANGE = "#D9822B"
GRAY = "#697583"
INK = "#283440"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": "#B1B8BF",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "svg.hashsalt": "pycbc-torch-docs-20260906",
}
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/images/torch-benchmarks-20260906/manifest.json"
BATCHES = (1, 8, 32, 128, 512, 1024)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def setup(nrows, ncols, title, subtitle, height, legend=()):
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, height), squeeze=False)
    fig.text(
        0.26 if ncols == 1 else 0.25, 0.975, title, fontsize=17, weight="bold", va="top"
    )
    fig.text(
        0.26 if ncols == 1 else 0.25, 0.925, subtitle, fontsize=10, va="top", color=GRAY
    )
    if legend:
        fig.legend(
            handles=legend,
            loc="upper left",
            bbox_to_anchor=(0.24, 0.885),
            frameon=False,
            ncol=len(legend),
            fontsize=10,
        )
    fig.subplots_adjust(
        left=0.26 if ncols == 1 else 0.25,
        right=0.98,
        top=(0.78 if ncols > 1 else 0.83) if not legend else 0.77,
        bottom=0.13,
        wspace=0.21,
        hspace=0.52,
    )
    for ax in axes.flat:
        ax.grid(axis="x", color="#E6E8EA", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=9)
        ax.xaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    return fig, axes


def bars(ax, labels, values, ranges, colors, unit, hatches=None, limit=None):
    for i, (value, bounds, color) in enumerate(
        zip(values, ranges, colors, strict=True)
    ):
        ax.barh(
            i,
            value,
            height=0.62,
            color=color,
            edgecolor=INK,
            linewidth=0.5,
            hatch=hatches[i] if hatches else None,
            xerr=[[value - bounds[0]], [bounds[1] - value]],
            error_kw={"ecolor": INK, "elinewidth": 0.9, "capsize": 2},
        )
        ax.annotate(
            number(value),
            (bounds[1], i),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, limit or max(high for _, high in ranges) * 1.26)
    ax.set_xlabel(unit, labelpad=9)


def pairs(ax, groups, series, unit, limit):
    """Each series is (color, hatch, values, ranges)."""
    n = len(series)
    height = 0.72 / n
    for j, (color, hatch, values, ranges) in enumerate(series):
        for i, (value, bounds) in enumerate(zip(values, ranges, strict=True)):
            y = i + (j - (n - 1) / 2) * height
            ax.barh(
                y,
                value,
                height=height * 0.88,
                color=color,
                edgecolor=INK,
                linewidth=0.5,
                hatch=hatch,
                xerr=[[value - bounds[0]], [bounds[1] - value]],
                error_kw={"ecolor": INK, "elinewidth": 0.9, "capsize": 2},
            )
            ax.annotate(
                number(value),
                (bounds[1], y),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
    ax.set_yticks(range(len(groups)), groups)
    ax.invert_yaxis()
    ax.set_xlim(0, limit)
    ax.set_xlabel(unit, labelpad=9)


def route_label(route, threads):
    name = {
        "branch_standard": "Standard CPU",
        "standard_cpu": "Standard CPU",
        "torch_cpu": "Torch CPU",
        "cuda": "Torch CUDA",
        "torch_cuda": "Torch CUDA",
    }[route]
    return f"{name} · {threads} thread{'s' if threads != 1 else ''}"


def route_color(route):
    return ORANGE if "cuda" in route else GRAY if "standard" in route else BLUE


def save(fig, name, out, manifest):
    fig.savefig(out / name, dpi=160, facecolor="white")
    plt.close(fig)
    record = next(r for r in manifest["figures"] if r["file"] == name)
    record.update(sha256=digest(out / name), bytes=(out / name).stat().st_size)


def exact_rows(rows, dimensions, expected, name):
    """Reject incomplete or duplicate matrices before writing any figures."""
    keyed = {tuple(row.get(field) for field in dimensions): row for row in rows}
    if len(rows) != len(keyed) or set(keyed) != expected:
        missing, extra = expected - set(keyed), set(keyed) - expected
        raise ValueError(
            f"{name}: incomplete or duplicate matrix; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return keyed


def check_range(center, bounds, name):
    values = (center, *bounds)
    if (
        len(bounds) != 2
        or any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or x <= 0
            for x in values
        )
        or not bounds[0] <= center <= bounds[1]
    ):
        raise ValueError(f"{name}: invalid center or observed worker range")


def validate_batch_data(data):
    """Consume only complete qualified six-batch summaries, including controls."""
    live = data["live"]
    cpu_routes = ("branch_standard", "torch_cpu", "torch_cpu_native")
    live_expected = {
        (head, threads, route, batch)
        for head in ("main", "cpu")
        for threads in (1, 4)
        for route in (
            cpu_routes + ("torch_cuda", "torch_cuda_native")
            if head == "main" and threads == 1
            else cpu_routes
        )
        for batch in BATCHES
    }
    rows = exact_rows(
        live["rows"],
        ("head", "threads", "route", "batch"),
        live_expected,
        "Live matched filter",
    )
    if any(
        live.get(key) != len(live_expected)
        for key in ("qualified_rows", "total_rows", "expected_rows")
    ):
        raise ValueError("Live matched filter: full campaign is not qualified")
    for identity, row in rows.items():
        if (
            row.get("qualified") is not True
            or row.get("qualification_issues")
            or row.get("revision") != live[f"{row['head']}_revision"]
        ):
            raise ValueError(f"Live matched filter: unqualified row {identity}")
        check_range(
            row["median_throughput_wps"], row["worker_range_wps"], str(identity)
        )

    waveform = data["waveform"]
    wave_cpu = ("standard-cpu", "torch-cpu-scalar", "torch-cpu-batch")
    wave_expected = {
        (route, threads, batch, precision)
        for threads in (1, 4)
        for route in (
            wave_cpu + ("torch-cuda-scalar", "torch-cuda-batch")
            if threads == 1
            else wave_cpu
        )
        for batch in BATCHES
        for precision in ("double", "single")
    }
    rows = exact_rows(
        list(waveform["cells"].values()),
        ("route", "threads", "batch", "precision"),
        wave_expected,
        "TaylorF2 baseline",
    )
    if (
        waveform.get("qualification_passed") is not True
        or waveform.get("verification_passed") is not True
        or waveform.get("raw_worker_files") != 288
        or waveform.get("qualified_timed_cells") != 48
        or waveform.get("verified_unsupported_cells") != 48
    ):
        raise ValueError("TaylorF2 baseline: full campaign is not qualified")
    for identity, row in rows.items():
        if row["precision"] == "single":
            if (
                row.get("status") != "unsupported"
                or row.get("eligible")
                or row.get("timing")
            ):
                raise ValueError(
                    f"TaylorF2 baseline: invalid single precision row {identity}"
                )
            continue
        if row.get("eligible") is not True or row.get("verification_errors"):
            raise ValueError(f"TaylorF2 baseline: unqualified row {identity}")
        timing = row["timing"]
        check_range(
            timing["waveforms_per_second"],
            (timing["min_replicate_throughput"], timing["max_replicate_throughput"]),
            str(identity),
        )

    triton = data["triton"]
    triton_expected = {
        (bins, route, batch)
        for bins in (4097, 32769)
        for route in ("cuda-off", "cuda-on")
        for batch in BATCHES
    }
    rows = exact_rows(
        triton["rows"], ("bins", "route", "batch"), triton_expected, "TaylorF2 Triton"
    )
    if (
        triton.get("synthetic")
        or triton.get("global_problems")
        or triton.get("qualified_group_count") != 24
        or triton.get("raw_worker_files") != 72
    ):
        raise ValueError("TaylorF2 Triton: full campaign is not qualified")
    for identity, row in rows.items():
        if row.get("qualified") is not True or row.get("problems"):
            raise ValueError(f"TaylorF2 Triton: unqualified row {identity}")
        for middle, low, high in (
            ("throughput_median", "throughput_min", "throughput_max"),
            ("cold_median_seconds", "cold_min_seconds", "cold_max_seconds"),
        ):
            check_range(row[middle], (row[low], row[high]), str(identity))


def batch_rows(rows, **identity):
    selected = sorted(
        (row for row in rows if all(row.get(k) == v for k, v in identity.items())),
        key=lambda row: row["batch"],
    )
    if tuple(row["batch"] for row in selected) != BATCHES:
        raise ValueError(f"Expected every batch exactly once: {identity}")
    return selected


def batch_setup(ncols, title, subtitle, legend, unit):
    fig, axes = plt.subplots(1, ncols, figsize=(9.5, 4.9), squeeze=False)
    fig.text(0.10, 0.975, title, fontsize=17, weight="bold", va="top")
    fig.text(0.10, 0.925, subtitle, fontsize=10, va="top", color=GRAY)
    fig.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0.09, 0.885),
        frameon=False,
        ncol=len(legend),
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.72,
        bottom=0.21 if ncols == 3 else 0.15,
        wspace=0.20 if ncols == 3 else 0.30,
    )
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCHES, [str(batch) for batch in BATCHES])
        ax.set_xlim(0.8, 1350)
        ax.minorticks_off()
        ax.tick_params(
            axis="x",
            labelsize=8 if ncols == 3 else 9,
            labelrotation=60 if ncols == 3 else 0,
        )
        ax.grid(axis="y", color="#E6E8EA", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("Batch size", labelpad=9)
    axes[0, 0].set_ylabel(unit, labelpad=9)
    return fig, axes


def batch_line(ax, values, ranges, color, marker):
    low, high = zip(*ranges, strict=True)
    ax.fill_between(BATCHES, low, high, color=color, alpha=0.10)
    ax.errorbar(
        BATCHES,
        values,
        yerr=[
            [v - lo for v, lo in zip(values, low, strict=True)],
            [hi - v for hi, v in zip(high, values, strict=True)],
        ],
        color=color,
        marker=marker,
        markersize=5,
        linewidth=1.4,
        elinewidth=0.8,
        capsize=2,
    )


def render(data, out, manifest):
    validate_batch_data(data)
    live = data["live"]["rows"]
    routes = (
        ("branch_standard", "Standard CPU", GRAY, "o"),
        ("torch_cpu", "Torch CPU", BLUE, "s"),
        ("torch_cuda", "Torch CUDA", ORANGE, "^"),
    )
    legend = [
        Line2D([], [], color=color, marker=marker, label=label)
        for _, label, color, marker in routes
    ]
    fig, axes = batch_setup(
        3,
        "Matched-filter throughput",
        "3 blocks · FFT 131,072 · panel scales differ · higher is faster",
        legend,
        "Template-blocks / second",
    )
    selected = [
        r
        for r in live
        if r["head"] == "main"
        and r["route"] in ("branch_standard", "torch_cpu", "torch_cuda")
    ]
    for ax, device, threads in zip(
        axes.flat, ("CPU", "CPU", "CUDA"), (1, 4, 1), strict=True
    ):
        limit = 0
        for route, _, color, marker in routes[2:] if device == "CUDA" else routes[:2]:
            rows = batch_rows(selected, route=route, threads=threads)
            batch_line(
                ax,
                [r["median_throughput_wps"] for r in rows],
                [r["worker_range_wps"] for r in rows],
                color,
                marker,
            )
            limit = max(limit, max(r["worker_range_wps"][1] for r in rows))
        ax.set_ylim(0, limit * 1.10)
        thread_label = "host thread" if device == "CUDA" else "thread"
        ax.set_title(
            f"{device} · {threads} {thread_label}{'s' if threads != 1 else ''}",
            loc="left",
            pad=14,
        )
    save(fig, "main-live.png", out, manifest)

    rows = [r for r in data["waveform"]["cells"].values() if r["precision"] == "double"]
    legend = [
        Line2D([], [], color=color, marker=marker, label=label)
        for label, color, marker in (
            ("CPU/LAL loop", GRAY, "o"),
            ("Torch scalar loop", BLUE, "s"),
            ("Torch batch", ORANGE, "^"),
        )
    ]
    fig, axes = batch_setup(
        3,
        "TaylorF2 waveform throughput",
        "4,097 bins · complex128 · panel scales differ · higher is faster",
        legend,
        "Waveforms / second",
    )
    for ax, device, threads in zip(
        axes.flat, ("cpu", "cpu", "cuda"), (1, 4, 1), strict=True
    ):
        routes = [
            (f"torch-{device}-scalar", BLUE, "s"),
            (f"torch-{device}-batch", ORANGE, "^"),
        ]
        if device == "cpu":
            routes.insert(0, ("standard-cpu", GRAY, "o"))
        limit = 0
        for route, color, marker in routes:
            selected = batch_rows(rows, route=route, threads=threads)
            timing = [r["timing"] for r in selected]
            ranges = [
                (t["min_replicate_throughput"], t["max_replicate_throughput"])
                for t in timing
            ]
            batch_line(
                ax, [t["waveforms_per_second"] for t in timing], ranges, color, marker
            )
            limit = max(limit, max(high for _, high in ranges))
        ax.set_ylim(0, limit * 1.10)
        thread_label = "host thread" if device == "cuda" else "thread"
        ax.set_title(
            f"{device.upper()} · {threads} {thread_label}{'s' if threads != 1 else ''}",
            loc="left",
            pad=14,
        )
    save(fig, "waveform.png", out, manifest)

    triton = data["triton"]["rows"]
    legend = [
        Patch(facecolor=GRAY, edgecolor=INK, label="Triton off"),
        Patch(facecolor=BLUE, edgecolor=INK, hatch="//", label="Triton on"),
    ]
    for cold in (False, True):
        key = "cold" if cold else "throughput"
        mid = "cold_median_seconds" if cold else "throughput_median"
        low = "cold_min_seconds" if cold else "throughput_min"
        high = "cold_max_seconds" if cold else "throughput_max"
        title = (
            "TaylorF2 first-call latency" if cold else "TaylorF2 with optional Triton"
        )
        sub = (
            "Fresh Triton cache · lower is faster"
            if cold
            else "Warm batch calls · complex128 · higher is faster"
        )
        fig, axes = setup(1, 2, title, sub, 7.1, legend)
        limit = max(r[high] for r in triton) * 1.32
        for ax, bins in zip(axes.flat, (4097, 32769), strict=True):
            series = []
            for route, color, hatch in (
                ("cuda-off", GRAY, None),
                ("cuda-on", BLUE, "//"),
            ):
                rows = batch_rows(triton, bins=bins, route=route)
                series.append(
                    (
                        color,
                        hatch,
                        [r[mid] for r in rows],
                        [(r[low], r[high]) for r in rows],
                    )
                )
            pairs(
                ax,
                [f"Batch {r['batch']}" for r in rows],
                series,
                "Seconds" if cold else "Waveforms / second",
                limit,
            )
            ax.set_title(f"{bins:,} frequency bins", loc="left", pad=14)
            if cold:
                ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        axes[0, 1].set_yticklabels([])
        save(fig, f"taylorf2-{key}.png", out, manifest)

    inference = data["inference"]["cells"]
    fig, axes = setup(
        1,
        2,
        "Likelihood throughput",
        "H1/L1 · 32 s at 2,048 Hz · double precision · higher is faster",
        4.9,
    )
    limit = max(r["evaluations_per_second_range"][1] for r in inference) * 1.25
    for ax, model, title in zip(
        axes.flat, ("gaussian", "relative"), ("GaussianNoise", "Relative"), strict=True
    ):
        rows = [r for r in inference if r["model"] == model]
        bars(
            ax,
            [route_label(r["route"], r["threads"]) for r in rows],
            [r["evaluations_per_second"] for r in rows],
            [r["evaluations_per_second_range"] for r in rows],
            [route_color(r["route"]) for r in rows],
            "Evaluations / second",
            limit=limit,
        )
        ax.set_title(title, loc="left", pad=14)
    axes[0, 1].set_yticklabels([])
    save(fig, "inference.png", out, manifest)

    legend = [
        Line2D([], [], color=GRAY, marker="o", linestyle="", label="GaussianNoise"),
        Line2D([], [], color=BLUE, marker="s", linestyle="", label="Relative"),
    ]
    fig, axes = setup(
        2,
        2,
        "Inference startup intervals",
        "Separate phase scales · seconds · lower is faster",
        8.2,
        legend,
    )
    phases = (
        ("imports_ns", "Imports"),
        ("cold_scheme_ns", "Scheme entry"),
        ("cold_model_setup_ns", "Model setup"),
        ("cold_first_update_likelihood_ns", "First likelihood"),
    )
    for ax, (phase, title) in zip(axes.flat, phases, strict=True):
        limit = max(r["cold"][phase]["range_ns"][1] for r in inference) / 1e9 * 1.18
        for model, color, marker, offset in (
            ("gaussian", GRAY, "o", -0.14),
            ("relative", BLUE, "s", 0.14),
        ):
            rows = [r for r in inference if r["model"] == model]
            for i, r in enumerate(rows):
                v = r["cold"][phase]
                center = v["median_ns"] / 1e9
                lo, hi = (x / 1e9 for x in v["range_ns"])
                ax.errorbar(
                    center,
                    i + offset,
                    xerr=[[center - lo], [hi - center]],
                    color=color,
                    marker=marker,
                    markersize=5,
                    capsize=2,
                    linewidth=0.9,
                )
        ax.set_yticks(
            range(len(rows)), [route_label(r["route"], r["threads"]) for r in rows]
        )
        ax.invert_yaxis()
        ax.set_xlim(0, limit)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.set_xlabel("Seconds")
        ax.set_title(title, loc="left", pad=14)
    for ax in axes[:, 1]:
        ax.set_yticklabels([])
    save(fig, "inference-cold.png", out, manifest)

    fft = data["fft"]["rows"]
    labels = [
        "Main FFTW",
        "Optional FFTW",
        "MEASURE",
        "Cached wisdom",
        "Main Torch fallback",
        "Optional Torch fallback",
    ]
    fig, axes = setup(
        1,
        2,
        "Inverse FFT costs",
        "FFTW: 1 thread · Torch: 4 threads · lower is faster",
        5.4,
    )
    for ax, metric, title in zip(
        axes.flat,
        ("plan_ms", "execute_ms"),
        ("Plan construction", "Warm execution"),
        strict=True,
    ):
        bars(
            ax,
            labels,
            [r[metric]["median"] for r in fft],
            [(r[metric]["minimum"], r[metric]["maximum"]) for r in fft],
            [GRAY if r["head"] == "main" else BLUE for r in fft],
            "Milliseconds",
        )
        ax.set_title(title, loc="left", pad=14)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    axes[0, 1].set_yticklabels([])
    save(fig, "fft.png", out, manifest)

    routes = (
        ("main", "branch_standard", "Standard CPU", GRAY, "o"),
        ("main", "torch_cpu_native", "Main native Torch CPU", BLUE, "s"),
        ("cpu", "torch_cpu_native", "Optional native Torch CPU", ORANGE, "^"),
    )
    legend = [
        Line2D([], [], color=color, marker=marker, label=label)
        for _, _, label, color, marker in routes
    ]
    fig, axes = batch_setup(
        2,
        "Optional CPU matched-filter tuning",
        "3 blocks · FFT 131,072 · complex64 · higher is faster",
        legend,
        "Template-blocks / second",
    )
    selected = [
        r
        for r in live
        if r["route"] == "torch_cpu_native"
        or (r["head"] == "main" and r["route"] == "branch_standard")
    ]
    limit = max(r["worker_range_wps"][1] for r in selected) * 1.10
    for ax, threads in zip(axes.flat, (1, 4), strict=True):
        for head, route, _, color, marker in routes:
            rows = batch_rows(selected, head=head, route=route, threads=threads)
            batch_line(
                ax,
                [r["median_throughput_wps"] for r in rows],
                [r["worker_range_wps"] for r in rows],
                color,
                marker,
            )
        ax.set_ylim(0, limit)
        ax.set_title(
            f"{threads} CPU thread{'s' if threads > 1 else ''}", loc="left", pad=14
        )
    save(fig, "optional-cpu.png", out, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    data = {}
    for source in manifest["summary_inputs"]:
        path = args.archive / source["path"]
        if digest(path) != source["sha256"]:
            raise ValueError(f"Archive summary checksum mismatch: {source['path']}")
        data[source["key"]] = json.loads(path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        render(data, args.output, manifest)
    manifest["documentation_renderer_sha256"] = digest(Path(__file__))
    manifest["matplotlib_version"] = matplotlib.__version__
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Rendered {len(manifest['figures'])} figures from {len(data)} verified summaries."
    )


if __name__ == "__main__":
    main()
