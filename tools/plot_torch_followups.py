#!/usr/bin/env python3
"""Render the 8 September executable and separate prototype comparisons.

Uses three byte-pinned summaries; no PyCBC imports or benchmark execution.
The default data directory is committed with the documentation. Rendering
requires Matplotlib and a new output directory. --verify-only checks an
existing manifest, summaries, renderer and images without Matplotlib.
This checks presentation, not scientific HDF comparisons or raw receipts.
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
REVISION = "ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f"
PINS = {
    "loader-v1-summary.json":
        "270a3a47d6cb98160e8730d09cb7acb8a053f2ff7f39c51461e404919e662a2a",
    "descriptor-reuse-timing-v1-summary.json":
        "71dc531bc1e08c8c93969e118276abbd6458f9cb9de76c38d9f8afd92b6463e0",
    "graph-v2-timing-summary.json":
        "bd2d5101dea6b72dda2a154d4ff3894bc6dfff03a0773927978dc9e10fcbd225",
}
IMAGES = tuple(f"{name}.{ext}" for name in
               ("executable-wall", "prototype-followups")
               for ext in ("png", "svg"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def stats(values):
    require(values and all(math.isfinite(x) and x > 0 for x in values),
            "Expected finite positive wall seconds")
    return {"values": values, "median": median(values),
            "minimum": min(values), "maximum": max(values)}


def close(actual, expected):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"Summary mismatch: {actual} != {expected}")


def pair(baseline, candidate):
    require(len(baseline) == len(candidate), "Unpaired sample counts")
    return {"baseline": stats(baseline), "candidate": stats(candidate),
            "paired_candidate_over_baseline":
                [c / b for b, c in zip(baseline, candidate)],
            "median_wall_reduction_percent":
                100 * (1 - median(candidate) / median(baseline))}


def load_results(data_dir):
    sources, consumed = {}, {}
    for name, expected in PINS.items():
        path = data_dir / name
        require(path.is_file() and not path.is_symlink(),
                f"Expected regular input: {name}")
        raw = path.read_bytes()
        consumed[name] = fingerprint(raw)
        require(consumed[name]["sha256"] == expected,
                f"Pinned summary changed: {name}")
        sources[name] = json.loads(raw)

    loader, reuse, graph = (sources[name] for name in PINS)
    require(loader["source_pins"]["revisions"]["candidate"] == REVISION,
            "Loader source mismatch")
    require(loader["qualified_routes"] == 6 and
            loader["verified_trigger_comparisons"] ==
            {"qualification-v1": 8, "timing-v1": 36},
            "Loader qualification summary mismatch")
    loader_results = {}
    for role, record in loader["roles"].items():
        values = [row["full_wall_seconds"] for row in record["samples"]]
        require(len(values) == 3, "Expected three loader workers per role")
        result = stats(values)
        close(result["median"], record["medians"]["full_wall_seconds"])
        require([min(values), max(values)] ==
                record["full_wall_range_seconds"],
                "Loader range mismatch")
        for row in record["samples"]:
            close(384 * 1904 / row["full_wall_seconds"],
                  row["template_seconds_per_wall_second"])
        result["median_template_seconds_per_wall_second"] = median(
            [384 * 1904 / value for value in values])
        loader_results[role] = result

    reuse_results = {}
    for role, records in reuse["samples"].items():
        require([row["repeat"] for row in records] == [1, 2, 3, 4],
                "Descriptor repeat order mismatch")
        values = [row["metrics"]["full_wall_seconds"] for row in records]
        require(stats(values) == reuse["summaries"][role]["full_wall_seconds"],
                "Descriptor sample/summary mismatch")
        reuse_results[role] = values
    comparisons = {}
    for name in ("standard-reuse", "cuda-reuse"):
        record = reuse["paired_comparisons"][name]
        baseline, candidate = (reuse_results[record[key]]
                               for key in ("baseline", "candidate"))
        result = pair(baseline, candidate)
        require(result["paired_candidate_over_baseline"] ==
                record["candidate_over_baseline"]["values"],
                "Descriptor pairing mismatch")
        comparisons[name] = result

    require(graph["accepted"] is True and len(graph["E"]) == 4,
            "Expected accepted four-pair graph campaign")
    graph_result = pair(graph["E"], graph["G"])
    for role in ("E", "G"):
        close(median(graph[role]), graph[f"median_{role}"])
        require([min(graph[role]), max(graph[role])] ==
                graph[f"range_{role}"],
                "Graph range mismatch")
    require([e - g for e, g in zip(graph["E"], graph["G"])] ==
            graph["E_minus_G"] and all(x > 0 for x in graph["E_minus_G"]),
            "Graph paired differences mismatch")
    require(graph_result["paired_candidate_over_baseline"] ==
            graph["G_over_E"],
            "Graph pairing mismatch")
    close(graph_result["median_wall_reduction_percent"],
          graph["percent_reduction"])
    comparisons["cuda-graph"] = graph_result
    return {"schema_version": 1, "source_revision": REVISION,
            "consumed_files": consumed,
            "scope": "Fresh full-process wall time; shared len, CPU 8, one "
                     "numerical thread; CUDA additionally uses one RTX 4090. "
                     "R and G are separate prototype campaigns, not timings "
                     "of an integrated production implementation. G eager "
                     "already includes descriptor reuse. Ranges are observed, "
                     "not confidence intervals or sustained capacity.",
            "result": {"loader": loader_results, "prototypes": comparisons},
            "renderer": fingerprint(Path(__file__).read_bytes())}


def save_figure(fig, stem):
    images = {}
    for extension in ("png", "svg"):
        stream = io.BytesIO()
        creator = "PyCBC executable and follow-up plot renderer"
        metadata = ({"Date": None, "Creator": creator} if extension == "svg"
                    else {"Software": creator})
        fig.savefig(stream, format=extension, dpi=180, metadata=metadata)
        raw = stream.getvalue()
        if extension == "svg":
            lines = (line.rstrip() for line in raw.splitlines())
            raw = b"\n".join(lines) + b"\n"
        images[f"{stem}.{extension}"] = raw
    return images


def render(result):
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def axes_style(ax):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e3e8ed", linewidth=.7)
        ax.tick_params(length=0, pad=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#b6c1cc")

    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        matplotlib.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 11,
            "svg.fonttype": "none", "svg.hashsalt": "pycbc-followups-20260908",
            "text.color": "#253444", "axes.labelcolor": "#253444",
            "xtick.color": "#253444", "ytick.color": "#52606d",
        })
        fig = Figure(figsize=(11.2, 7.6), facecolor="white")
        FigureCanvasAgg(fig)
        fig.text(.09, .94, "pycbc_inspiral · latest matched executable",
                 fontsize=21, weight="bold")
        fig.text(.09, .887, f"Source {REVISION[:8]} · bounded frame loader "
                 "on all three backends", fontsize=12)
        fig.text(.09, .842, "384 compressed templates × 1,904 H1 seconds "
                 "· 5 segments per template")
        fig.text(.09, .800, "1 host core / 1 thread · CUDA: 1 RTX 4090 "
                 "· shared len (unreserved)", color="#52606d")
        ax = fig.add_axes([.09, .37, .85, .37])
        axes_style(ax)
        ax.set(ylim=(0, 130), xlim=(-.65, 2.65),
               ylabel="Full-process wall time (seconds) · lower is faster")
        ax.set_xticks([0, 1, 2],
                      ["Standard CPU / MKL", "Torch CPU", "Torch CUDA"])
        colors = ("#526f91", "#a56932", "#007f73")
        for x, (role, color) in enumerate(zip(
                ("standard-candidate", "cpu-candidate", "cuda-candidate"),
                colors)):
            row = result["loader"][role]
            low, high, middle = row["minimum"], row["maximum"], row["median"]
            ax.bar(x, middle, width=.64, color=color, alpha=.13)
            ax.hlines(middle, x - .32, x + .32, color=color, linewidth=2)
            ax.scatter([x - .21, x, x + .21], row["values"], s=60,
                       color=color, edgecolor="white", zorder=4)
            ax.vlines(x + .40, low, high, color=color)
            ax.hlines([low, high], x + .36, x + .44, color=color)
            ax.text(x, high + 6, f"{middle:.3f} s", ha="center",
                    fontsize=15, weight="bold", color=color)
            ax.text(x, -.24, f"Range {low:.3f}–{high:.3f} s", ha="center",
                    fontsize=10, transform=ax.get_xaxis_transform())
            rate = row["median_template_seconds_per_wall_second"]
            ax.text(x, -.35, f"{rate:,.2f} template-s / wall-s", ha="center",
                    fontsize=10, transform=ax.get_xaxis_transform())
        fig.text(.09, .178, "Dots: three fresh workers · line: median "
                 "· whisker: observed min–max", fontsize=10)
        fig.text(.09, .134, "Full clock includes runtime verification, "
                 "startup, setup, filtering and HDF output.", fontsize=10)
        fig.text(.09, .091, "Finite-workload rate per assigned host core; "
                 "CUDA also consumes a GPU. "
                 "No sustained-capacity claim.",
                 fontsize=10, color="#52606d")
        fig.text(.09, .049, "Timing campaign: 18 workers across baseline "
                 "and candidate; this figure shows "
                 "the nine candidate workers.",
                 fontsize=10, color="#52606d")
        images = save_figure(fig, "executable-wall")

        fig = Figure(figsize=(11.8, 6.6), facecolor="white")
        FigureCanvasAgg(fig)
        fig.text(.07, .94, "Executable follow-ups · separate prototypes",
                 fontsize=21, weight="bold")
        fig.text(.07, .886, f"Source {REVISION[:8]} + pinned Python helpers "
                 "· 4 fresh worker pairs per comparison", fontsize=11)
        panels = (("standard-reuse", "R · Standard CPU", "No reuse", "Reuse"),
                  ("cuda-reuse", "R · CUDA", "No reuse", "Reuse"),
                  ("cuda-graph", "G · CUDA", "Eager + reuse", "Graph + reuse"))
        for index, (key, title, left, right) in enumerate(panels):
            ax = fig.add_axes([.07 + .32 * index, .34, .25, .39])
            axes_style(ax)
            row = result["prototypes"][key]
            before, after = row["baseline"], row["candidate"]
            for b, c in zip(before["values"], after["values"]):
                ax.plot([0, 1], [b, c], "o-", color="#007f73", alpha=.55,
                        linewidth=1, markersize=5)
            for x, data in enumerate((before, after)):
                ax.hlines(data["median"], x - .15, x + .15, color="#253444",
                          linewidth=2.5, zorder=5)
            ax.set_xlim(-.35, 1.35)
            ax.margins(y=.17)
            ax.set_xticks([0, 1], [left, right], fontsize=10)
            ax.set_title(title, pad=17, weight="bold", fontsize=13)
            ax.set_ylabel("Full wall seconds · expanded scale", fontsize=10)
            label = (f"Medians {before['median']:.3f} → "
                     f"{after['median']:.3f} s")
            ax.text(.5, -.24, label,
                    transform=ax.transAxes, ha="center", fontsize=10)
            reduction = row['median_wall_reduction_percent']
            ax.text(.5, -.37, f"{reduction:.2f}% lower wall time",
                    transform=ax.transAxes, ha="center", fontsize=12,
                    weight="bold", color="#007f73")
        fig.text(.07, .125, "Lines join same-repeat workers; dark marks are "
                 "medians. Each panel has its own expanded time axis.",
                 fontsize=10)
        fig.text(.07, .078, "G eager already includes descriptor reuse. "
                 "R and G are separate campaigns; do not add their gains.",
                 fontsize=10, color="#52606d")
        fig.text(.07, .034, "Full process includes setup and cleanup, "
                 "and graph capture where enabled. "
                 "Shared host; no integrated-revision timing.",
                 fontsize=10, color="#52606d")
        images.update(save_figure(fig, "prototype-followups"))
        return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "docs/data/torch-20260908")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    expected = load_results(args.data)
    if args.verify_only:
        require(args.output.is_dir() and not args.output.is_symlink(),
                "Expected regular output directory")
        require({p.name for p in args.output.iterdir()} ==
                {*IMAGES, "manifest.json"}, "Output inventory mismatch")
        for name in (*IMAGES, "manifest.json"):
            require((args.output / name).is_file() and
                    not (args.output / name).is_symlink(),
                    f"Expected regular output: {name}")
        images = {name: fingerprint((args.output / name).read_bytes())
                  for name in IMAGES}
        require(json.loads((args.output / "manifest.json").read_bytes()) ==
                {**expected, "images": images}, "Output manifest mismatch")
    else:
        require(not args.output.is_symlink() and
                (not args.output.exists() or (args.output.is_dir() and
                 not any(args.output.iterdir()))),
                "Rendering requires a new or empty output directory")
        images = render(expected["result"])
        args.output.mkdir(parents=True, exist_ok=True)
        for name, raw in images.items():
            (args.output / name).write_bytes(raw)
        manifest = {**expected, "images":
                    {name: fingerprint(raw) for name, raw in images.items()}}
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("Verified summaries and presentation" if args.verify_only
          else "Rendered executable and prototype figures")


if __name__ == "__main__":
    main()
