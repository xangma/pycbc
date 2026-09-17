#!/usr/bin/env python3
"""Render the finite pycbc_inspiral backend wall-time evidence offline.

Usage: python tools/plot_torch_executable.py --evidence DIR --output DIR
Add --verify-only to check existing artifacts without rendering or writing.
Only two hash-pinned summaries are consumed. This validates their presentation,
not the archived receipts, source/native binaries, or scientific HDF outputs.
Rendering requires matplotlib; the output directory must be new or empty.
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median
import sys


ARCHIVE = {
    "repository": "xangma/pycbc",
    "commit": "a742e59004779b35e3caea1a088ad5854042b704",
    "directory": "device-profile-20260907-r3",
    "url": ("https://github.com/xangma/pycbc/tree/"
            "a742e59004779b35e3caea1a088ad5854042b704/"
            "device-profile-20260907-r3"),
}
# Fixed summary bytes; no CLI or supplied SHA256SUMS can change pins.
SUMMARY_SHA256 = {
    "handoff.json":
        "f48a0ba19a3a9cc85a662a1a85580025852822cbd50ccee0df558c296949e1f3",
    "evidence-verification.json":
        "8b02ff2ef4df7546226d5c63a6c2e47b93568eb39138675a8a0e2abff678c59d",
}
REVISION = "d2647addb884ead3249914ebc980f3c132076d93"
BACKENDS = {"cpu": "cpu:1", "torchcpu": "torch:cpu:1",
            "cuda": "torch:cuda:0"}
INTERVALS = [[1187007160.0, 1187009064.0]]
QUALIFICATION_CHECKS = {
    "all_decompressions_succeeded", "all_interpolations_observed",
    "all_segment_psds_saved", "all_segment_psds_valid_for_filter",
    "bank_0_all_templates_once", "bank_0_compression_enabled",
    "conditioned_strain_recorded", "no_generation_fallback",
    "nonempty_segments", "one_compressed_bank", "one_scalar_controller",
    "recorded_geometry_matches_controller",
    "scalar_ifft_executed_for_every_pair",
}
SCIENCE_FIELDS = {
    "bank_chisq", "bank_chisq_dof", "chisq", "chisq_dof", "coa_phase",
    "cont_chisq", "cont_chisq_dof", "sg_chisq", "sigmasq", "snr",
    "template_duration",
}
SCIENCE_CASES = {
    f"{kind}-{backend}" for backend in BACKENDS
    for kind in ("cprofile", "filter-cprofile", "timing-1", "timing-2",
                 "timing-3")
} | {"cuda-trace", "filter-perf-cpu", "filter-perf-torchcpu",
     "qualify-torchcpu", "qualify-cuda"}
NUMERICAL_BUDGETS = {"rtol": 1e-4, "atol": 1e-5, "sigmasq_rtol": 1e-5,
                     "phase_atol": 1e-4, "max_examples": 12}
IMAGES = ("executable-wall.png", "executable-wall.svg")
SCOPE = {
    "executable": "pycbc_inspiral",
    "compressed_templates": 384,
    "unique_valid_seconds": 1904,
    "detector": "H1",
    "segments_per_template": 5,
    "resources": ("One host core / one thread; CUDA uses one RTX 4090; "
                  "shared len, no exclusive reservation."),
    "timing": ("Fresh unprofiled full-process wall time: immediately before "
               "checked-worker launch through child exit, including runtime "
               "wrapper, startup, setup, filtering and HDF output."),
    "interpretation": ("Three samples per backend; median and observed range. "
                       "Not confidence intervals or sustained capacity. "
                       "No old-versus-new speedup comparison."),
    "validation": ("Offline presentation validation of two hash-pinned "
                   "summaries. No scientific replay or re-audit of archived "
                   "receipts, source/native binaries or HDF outputs; "
                   "no benchmark execution."),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def finite_tree(value):
    """Reject exponent overflow as well as explicit nonfinite constants."""
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite numeric value")


def read_json(data):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    result = json.loads(data, parse_constant=invalid, object_pairs_hook=unique)
    finite_tree(result)
    return result


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value),
            f"Invalid numeric value: {label}")
    return value


def close(actual, expected, label):
    require(math.isclose(number(actual, label), expected,
                         rel_tol=1e-12, abs_tol=1e-12),
            f"Inconsistent {label}")


def canonical(value):
    # Structural equality alone would accept a boolean in place of 0 or 1.
    return json.dumps(value, sort_keys=True, allow_nan=False)


def validate_science(comparisons):
    require(set(comparisons) == SCIENCE_CASES,
            "Incomplete scientific comparison cohort")
    results = {}
    for case, row in comparisons.items():
        require(row["status"] == "pass" and row["failures"] == []
                and row["review_reasons"] == [],
                f"Scientific comparison failed: {case}")
        close(row["sample_rate"], 4096, f"sample rate: {case}")
        require(set(row["detectors"]) == {"H1"}, "Unexpected detectors")
        h1 = row["detectors"]["H1"]
        require(h1["valid_intervals"] == INTERVALS,
                f"Science interval mismatch: {case}")
        for field in ("baseline_count", "candidate_count", "matched_count"):
            close(h1[field], 1991, f"{case} {field}")
        require(set(h1["unmatched"]) == {"baseline_only", "candidate_only"},
                f"Incomplete unmatched identities: {case}")
        for unmatched in h1["unmatched"].values():
            close(unmatched["count"], 0, f"unmatched identities: {case}")
            require(unmatched["classifications"] == {}
                    and unmatched["examples"] == [],
                    f"Unmatched scientific identities: {case}")
        require(set(h1["metrics"]) == SCIENCE_FIELDS,
                f"Incomplete scientific fields: {case}")
        for field, metric in h1["metrics"].items():
            close(metric["violations"], 0, f"{case} {field} violations")
            for key in ("max_absolute_error", "max_relative_error"):
                require(number(metric[key], f"{case} {field} {key}") >= 0,
                        f"Negative scientific error: {case} {field}")
        results[case] = {"status": "pass", "matched_triggers": 1991,
                         "compared_fields": len(SCIENCE_FIELDS)}
    return results


def validate_summaries(sources):
    """Cross-check pinned summaries; never open their receipt/HDF paths."""
    finite_tree(sources)
    handoff = sources["handoff.json"]
    verified = sources["evidence-verification.json"]
    require(handoff["state"] == "complete" and verified["state"] == "pass",
            "Evidence is incomplete or failed")
    for summary in (handoff, verified):
        require(summary["revision"] == REVISION, "Unexpected source revision")
        require(set(summary["numerical_budgets"]) == set(NUMERICAL_BUDGETS),
                "Incomplete numerical budgets")
        for key, value in NUMERICAL_BUDGETS.items():
            close(summary["numerical_budgets"][key], value, key)
        require(set(summary["timing"]) == set(BACKENDS),
                "Unexpected backend cohort")
    release = handoff["release"]
    require(release["state"] == "validated-release"
            and release["host"] == "len"
            and release["source_unchanged"] is True
            and release["all_source_native_input_helper_pins_unchanged"]
            is True
            and verified["native_input_pins_verified"] is True,
            "Source/input verification failed")
    require(set(verified["qualifications"]) == set(BACKENDS),
            "Incomplete backend qualifications")
    science = validate_science(verified["scientific_comparisons"])
    stats, samples, qualifications = {}, {}, {}
    workload = SCOPE["compressed_templates"] * SCOPE["unique_valid_seconds"]
    for backend, scheme in BACKENDS.items():
        qualification = verified["qualifications"][backend]
        checks = qualification["checks"]
        require(set(checks) == QUALIFICATION_CHECKS
                and all(value is True for value in checks.values()),
                f"Incomplete or failed qualification: {backend}")
        for key, expected in (("templates", 384), ("segments", 5),
                              ("valid_seconds", 1904)):
            close(qualification[key], expected, f"{backend} {key}")
        qualifications[backend] = {
            key: qualification[key] for key in
            ("sha256", "checks", "templates", "segments", "valid_seconds")}
        timing = verified["timing"][backend]
        rows = timing["samples"]
        require([row["case"] for row in rows]
                == [f"timing-{i}-{backend}" for i in (1, 2, 3)],
                f"Expected three ordered unprofiled {backend} repeats")
        values = []
        samples[backend] = []
        for row in rows:
            case = row["case"]
            require(row["scheme"] == scheme
                    and row["valid_intervals"] == INTERVALS,
                    f"Invalid workload: {case}")
            for key, expected in (("templates", 384),
                                  ("inferred_templates", 384),
                                  ("inferred_segments", 5)):
                close(row[key], expected, f"{case} {key}")
            require(row["qualification"] == qualification["path"],
                    f"Qualification association mismatch: {case}")
            comparison = verified["scientific_comparisons"][case]
            require(row["trigger_sha256"] == comparison["candidate_sha256"]
                    and row["receipt_sha256"]
                    == comparison["candidate_receipt_sha256"],
                    f"Science/timing association mismatch: {case}")
            value = number(row["full_wall_seconds"], case)
            require(value > 0, f"Invalid full wall time: {case}")
            rate = workload / value
            close(row["template_seconds_per_wall_second"], rate,
                  f"{case} throughput")
            values.append(value)
            samples[backend].append({
                "case": case, "full_wall_seconds": value,
                "template_seconds_per_wall_second": rate,
                "trigger_sha256": row["trigger_sha256"],
                "receipt_sha256": row["receipt_sha256"],
            })
        middle = median(values)
        stats[backend] = {
            "median_wall_seconds": middle, "min_wall_seconds": min(values),
            "max_wall_seconds": max(values),
            "median_template_seconds_per_wall_second": workload / middle,
            "relative_range": (max(values) - min(values)) / middle,
        }
        for key, value in stats[backend].items():
            close(timing[key], value, f"verification {backend} {key}")
            close(handoff["timing"][backend][key], value,
                  f"handoff {backend} {key}")
        median_worker = canonical(timing["median_wall_sample"])
        require(any(row["full_wall_seconds"] == middle
                    and median_worker == canonical(row) for row in rows),
                f"Wrong median worker: {backend}")
    return {"revision": REVISION, "backends": BACKENDS, "samples": samples,
            "full_wall": stats, "template_seconds": workload,
            "qualifications": qualifications, "science": science,
            "numerical_budgets": NUMERICAL_BUDGETS}


def load_evidence(root):
    sources, consumed = {}, {}
    for name, expected in SUMMARY_SHA256.items():
        data = (root / name).read_bytes()
        consumed[name] = fingerprint(data)
        require(consumed[name]["sha256"] == expected,
                f"Pinned SHA256 mismatch: {name}")
        sources[name] = read_json(data)
    return {"schema_version": 1, "archive": ARCHIVE, "scope": SCOPE,
            "consumed_files": consumed, "result": validate_summaries(sources),
            "renderer": {"path": "tools/plot_torch_executable.py",
                         **fingerprint(Path(__file__).read_bytes())}}


def render(result):
    """Fixed canvas, fonts, SVG IDs and metadata; no timestamps/local paths."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        matplotlib.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 11,
            "svg.fonttype": "none", "svg.hashsalt": "pycbc-executable-r3",
            "text.color": "#253444", "axes.labelcolor": "#253444",
            "xtick.color": "#253444", "ytick.color": "#52606d",
        })
        fig = Figure(figsize=(11.2, 7.6), facecolor="white")
        FigureCanvasAgg(fig)
        fig.text(.09, .94, "pycbc_inspiral · backend wall time",
                 fontsize=22, weight="bold")
        fig.text(.09, .895,
                 f"Fresh unprofiled workers · frozen source {REVISION[:8]}",
                 fontsize=12)
        fig.text(.09, .857,
                 "384 compressed templates × 1,904 H1 seconds "
                 "· 5 segments per template", fontsize=11)
        fig.text(.09, .820,
                 "1 host core / 1 thread · CUDA: 1 RTX 4090 "
                 "· shared len (unreserved)", fontsize=11, color="#52606d")

        ax = fig.add_axes([.09, .36, .85, .40])
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e3e8ed", linewidth=.7)
        ax.set(xlim=(-.65, 2.65), ylim=(0, 140),
               ylabel="Full-process wall time (seconds)")
        ax.set_yticks(range(0, 141, 20))
        ax.set_xticks([0, 1, 2], ["CPU / MKL", "Torch CPU", "Torch CUDA"])
        ax.tick_params(axis="both", length=0, pad=10)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#b6c1cc")
        colors = ("#526f91", "#a56932", "#007f73")
        for x, (backend, color) in enumerate(zip(BACKENDS, colors)):
            stats = result["full_wall"][backend]
            middle = stats["median_wall_seconds"]
            low, high = stats["min_wall_seconds"], stats["max_wall_seconds"]
            ax.bar(x, middle, width=.64, color=color, alpha=.13)
            ax.hlines(middle, x - .32, x + .32, color=color, linewidth=2)
            # Horizontal offsets make all three dots visible at the zero scale.
            ax.scatter([x - .21, x, x + .21],
                       [row["full_wall_seconds"]
                        for row in result["samples"][backend]],
                       s=62, color=color, edgecolor="white", linewidth=1,
                       zorder=4)
            ax.vlines(x + .40, low, high, color=color)
            ax.hlines([low, high], x + .36, x + .44, color=color, linewidth=1)
            ax.text(x, high + 7, f"{middle:.3f} s", ha="center",
                    weight="bold", fontsize=15, color=color)
            ax.text(x, -.21, f"Range  {low:.3f}–{high:.3f} s", ha="center",
                    fontsize=9.5, transform=ax.get_xaxis_transform())
            ax.text(x, -.30,
                    f"{stats['median_template_seconds_per_wall_second']:,.2f}"
                    " template-s / wall-s", ha="center", fontsize=10,
                    transform=ax.get_xaxis_transform())
        fig.legend(handles=[
            Line2D([], [], color="#52606d", marker="o", linestyle="none",
                   label="Samples 1, 2, 3 (left to right)"),
            Line2D([], [], color="#52606d", linewidth=2, label="Median"),
            Line2D([], [], color="#52606d", marker="|", markersize=9,
                   linewidth=1, label="Observed min–max"),
        ], loc="center", bbox_to_anchor=(.515, .185), ncol=3, frameon=False,
            fontsize=10)
        fig.text(.09, .133,
                 "Includes runtime wrapper, startup, setup, filtering "
                 "and HDF output.", fontsize=10)
        fig.text(.09, .094,
                 "Three samples per backend; descriptive ranges, "
                 "not confidence intervals or sustained capacity.",
                 fontsize=10, color="#52606d")
        fig.text(.09, .055,
                 "All 21 science outputs passed fixed trigger and numerical "
                 "gates. Acquired 7 Sep 2026.",
                 fontsize=10, color="#52606d")

        images = {}
        for name in IMAGES:
            extension = name.rsplit(".", 1)[1]
            creator = "PyCBC executable backend plot renderer"
            metadata = ({"Date": None, "Creator": creator}
                        if extension == "svg" else {"Software": creator})
            stream = io.BytesIO()
            fig.savefig(stream, format=extension, dpi=180, metadata=metadata)
            data = stream.getvalue()
            if extension == "svg":
                data = b"\n".join(line.rstrip() for line in data.splitlines())
                data += b"\n"
            images[name] = data
        return images


def verify_output(output, expected):
    require(output.is_dir() and not output.is_symlink(),
            "Output must be a regular directory")
    require({path.name for path in output.iterdir()}
            == {*IMAGES, "manifest.json"},
            "Unexpected or missing output files")
    for name in (*IMAGES, "manifest.json"):
        require((output / name).is_file() and not (output / name).is_symlink(),
                f"Output must be a regular file: {name}")
    images = {name: fingerprint((output / name).read_bytes())
              for name in IMAGES}
    require(canonical(read_json((output / "manifest.json").read_bytes()))
            == canonical({**expected, "images": images}),
            "Manifest mismatch: evidence, renderer, results or images changed")


def write_output(output, expected):
    require(not output.is_symlink() and (not output.exists()
            or (output.is_dir() and not any(output.iterdir()))),
            "Rendering requires a new or empty output directory")
    images = render(expected["result"])
    manifest = {**expected, "images": {
        name: fingerprint(data) for name, data in images.items()}}
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creates also prevent overwrites if another writer races us.
    for name, data in images.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True,
                                allow_nan=False) + "\n")
    verify_output(output, expected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        expected = load_evidence(args.evidence)
        if args.verify_only:
            verify_output(args.output, expected)
        else:
            write_output(args.output, expected)
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    action = "Verified" if args.verify_only else "Rendered and verified"
    print(f"{action} {args.output}: two pinned summaries, nine wall samples; "
          "offline presentation validation only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
