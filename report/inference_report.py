#!/usr/bin/env python3
"""Render the pinned-head public inference benchmark; imports no PyCBC code.

python inference_report.py --input-dir RAW_INFERENCE --output-dir REPORT
Only plotting requires matplotlib. Input files are never changed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


REVISION = "607bce53ead14f12af32552a5b2441d3bc667267"
TOLERANCE = {"rtol": 2e-7, "atol": 2e-5}
MODELS = {"gaussian": "GaussianNoise", "relative": "Relative"}
CONFIGURATIONS = (
    ("standard_cpu", 1, "Standard CPU · 1 thread", "#536171"),
    ("standard_cpu", 4, "Standard CPU · 4 threads", "#8E9AA8"),
    ("torch_cpu", 1, "Torch CPU · 1 thread", "#0072B2"),
    ("torch_cpu", 4, "Torch CPU · 4 threads", "#56B4E9"),
    ("cuda", 1, "Torch CUDA · 1 CPU thread", "#D55E00"),
)
COLD_PHASES = {
    "imports_ns": ("PyCBC / numerical imports", "#536171"),
    "cold_scheme_ns": ("Scheme entry + synchronization", "#009E73"),
    "cold_model_setup_ns": ("Model / data setup", "#0072B2"),
    "cold_first_update_likelihood_ns": ("First update + likelihood", "#D55E00"),
}
BOUNDARY = (
    "End-to-end public model.update(**varied_parameters) + "
    "float(model.loglikelihood), including waveform generation, detector "
    "projection, likelihood reduction and host scalar return. CUDA is "
    "synchronized before and after every timed group. Setup, parity and "
    "warmup are outside steady-state timing."
)
LIMITATIONS = [
    "Synthetic H1/L1 injection plus seeded colored noise, 32 s at 2048 Hz; "
    "float64/complex128 and TaylorF2 near masses 10/8 solar masses.",
    "Real GaussianNoise and Relative public models; Relative uses epsilon=0.1 "
    "without phase or distance marginalization or Earth rotation.",
    "Parameters are host Python scalars. Timed calls include the host scalar "
    "result; input frequency data and PSD construction are model setup costs.",
    "Each process has 12 parity points, 8 warmup points and 5 timed groups of "
    "16 varied parameter points; all cold/parity/warm/timed mass pairs differ.",
    "Three fresh processes per configuration provide observed ranges. "
    "No confidence intervals, tail latency or sampler throughput are inferred.",
    "Parity compares each model with its own standard CPU reference. "
    "It does not establish Relative's approximation accuracy against GaussianNoise.",
    "Cold phases are separate measured intervals; interpreter startup, input "
    "array loading and runtime configuration are not a complete measured total.",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not positive or value > 0))


def read_json(path):
    try:
        def reject_constant(value):
            raise ValueError(f"non-finite JSON constant: {value}")
        value = json.loads(path.read_text(), parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value, None
    except (OSError, ValueError) as exc:
        return {}, f"{path.name}: {exc}"


def case_problems(case, directory):
    issues = []
    preparation = case.get("preparation", {})
    if (preparation.get("sha") != REVISION or preparation.get("clean") is not True
            or preparation.get("status") != ""):
        issues.append("case preparation is not the exact clean required revision")
    expected = {"duration": 32, "sample_rate": 2048, "detectors": ["H1", "L1"],
                "delta_f": 1 / 32, "frequency_bins": 32769, "f_low": 20.0,
                "f_high": 1023.0, "precision": "float64/complex128", "seed": 20260906,
                "groups": 5, "evaluations_per_group": 16}
    for key, value in expected.items():
        if case.get(key) != value:
            issues.append(f"case {key} differs from full benchmark requirement {value!r}")
    if case.get("static") != {"approximant": "TaylorF2", "f_lower": 20.0,
                              "spin1z": 0.0, "spin2z": 0.0}:
        issues.append("unexpected waveform settings")
    points = []
    for key, count in (("parity_points", 12), ("warmup_points", 8), ("timing_points", 80)):
        values = case.get(key, [])
        if not isinstance(values, list) or len(values) != count:
            issues.append(f"expected {count} {key}")
        else:
            points.extend(values)
    points.append(case.get("cold_point", {}))
    try:
        if len({(p["mass1"], p["mass2"]) for p in points}) != len(points):
            issues.append("reused intrinsic mass pair across benchmark phases")
    except (TypeError, KeyError):
        issues.append("missing mass pair in benchmark points")
    data_path = directory / "case.npz"
    if not data_path.is_file() or sha256(data_path) != case.get("data_sha256"):
        issues.append("case.npz missing or its SHA256 differs from case.json")
    return issues


def assess_worker(record, case, case_hash, model, route, threads, replicate):
    issues = []
    for key, expected in (("sha", REVISION), ("final_sha", REVISION),
                          ("status", "ok"), ("clean", True), ("final_status", ""),
                          ("model", model), ("route", route), ("threads", threads),
                          ("replicate", replicate), ("case_sha256", case_hash),
                          ("data_sha256", case.get("data_sha256")),
                          ("harness_sha256", case.get("preparation", {}).get("harness_sha256"))):
        if record.get(key) != expected or (key == "clean" and record.get(key) is not True):
            issues.append(f"{key}: expected {expected!r}, got {record.get(key)!r}")
    # status='ok' replaces the starting git status; final_status is the end check.
    for key in ("host", "python", "python_version", "platform", "numpy_version", "harness_sha256"):
        if not isinstance(record.get(key), str) or not record[key]:
            issues.append(f"missing runtime identity: {key}")
    if not isinstance(record.get("pid"), int) or record["pid"] <= 0:
        issues.append("missing worker process ID")
    environment = record.get("thread_environment", {})
    if any(environment.get(key) != str(threads) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")):
        issues.append("recorded thread environment differs from configuration")
    if route != "standard_cpu" and (record.get("torch_threads") != threads
                                     or record.get("torch_interop_threads") != 1):
        issues.append("Torch thread counts differ from configuration")
    if record.get("parity_tolerance") != TOLERANCE:
        issues.append("parity tolerance differs from the declared fixed tolerance")
    reference = case.get("reference", {}).get(model, {})
    expected_values = reference.get("values", [])
    parity = record.get("parity", [])
    if (reference.get("status") != "ok" or len(expected_values) != 12
            or not isinstance(parity, list) or len(parity) != 12):
        issues.append("missing complete 12-point CPU reference or parity comparison")
    else:
        for index, (row, expected) in enumerate(zip(parity, expected_values)):
            passed = row.get("passed") is True
            passed = passed and row.get("params") == case["parity_points"][index]
            passed = passed and row.get("expected") == expected
            for key in ("loglikelihood", "loglr"):
                actual, target = row.get("actual", {}).get(key), expected.get(key)
                passed = passed and number(actual) and number(target)
                if number(actual) and number(target):
                    error = abs(actual - target)
                    passed = passed and error <= TOLERANCE["atol"] + TOLERANCE["rtol"] * abs(target)
                    stored = row.get("absolute_errors", {}).get(key)
                    passed = passed and number(stored) and math.isclose(
                        error, stored, rel_tol=1e-12, abs_tol=1e-14)
            if not passed:
                issues.append(f"parity point {index}: failed recomputed finite-value check")
    samples = record.get("group_samples", [])
    means = []
    if not isinstance(samples, list) or len(samples) != 5:
        issues.append("expected 5 completed timing groups")
    else:
        for index, sample in enumerate(samples):
            values = sample.get("loglikelihoods", [])
            total, stored = sample.get("total_ns"), sample.get("ns_per_evaluation")
            valid = (sample.get("group") == index and sample.get("evaluations") == 16
                     and isinstance(values, list) and len(values) == 16
                     and all(number(v) for v in values)
                     and number(total, positive=True) and number(stored, positive=True))
            if valid:
                valid = math.isclose(total / 16, stored, rel_tol=1e-12, abs_tol=1e-6)
            if valid:
                means.append(total / 16)
            else:
                issues.append(f"timing group {index}: invalid/inconsistent samples")
    cold = {key: record.get(key) for key in COLD_PHASES}
    cold_valid = all(number(value, positive=True) for value in cold.values())
    return {
        "replicate": replicate, "qualified": not issues, "issues": issues,
        "group_mean_latencies_ns": means,
        "median_group_mean_latency_ns": statistics.median(means) if len(means) == 5 else None,
        "cold_phases_ns": cold, "cold_complete": cold_valid,
        "raw_record": record,
    }


def aggregate(directory):
    case_path = directory / "case.json"
    case, error = read_json(case_path)
    issues = [error] if error else []
    issues.extend(case_problems(case, directory))
    case_hash = sha256(case_path) if case_path.is_file() else None
    expected_files = {f"{m}-{r}-t{t}-r{i}.json" for m in MODELS
                      for r, t, _, _ in CONFIGURATIONS for i in range(3)}
    extra = sorted(p.name for m in MODELS for p in directory.glob(f"{m}-*.json")
                   if p.name not in expected_files)
    if extra:
        issues.append(f"unexpected worker files: {extra}")
    cells = []
    for model in MODELS:
        for route, threads, label, color in CONFIGURATIONS:
            workers = []
            for replicate in range(3):
                path = directory / f"{model}-{route}-t{threads}-r{replicate}.json"
                record, error = read_json(path)
                worker = assess_worker(record, case, case_hash, model, route, threads, replicate)
                worker.update(source_file=path.name,
                              source_sha256=sha256(path) if path.is_file() else None)
                if error:
                    worker["issues"].insert(0, error)
                    worker["qualified"] = False
                workers.append(worker)
            cell_issues = list(issues)
            cell_issues.extend(f"replicate {w['replicate']}: {problem}"
                               for w in workers for problem in w["issues"])
            # All processes in a comparison must refer to one runtime/harness.
            identities = {tuple(w["raw_record"].get(k) for k in
                                ("host", "python", "python_version", "platform", "numpy_version"))
                          for w in workers}
            if len(identities) != 1:
                cell_issues.append("replicate runtime identities differ")
            if len({w["raw_record"].get("pid") for w in workers}) != 3:
                cell_issues.append("three distinct worker process IDs were not recorded")
            qualified = not cell_issues
            medians = [w["median_group_mean_latency_ns"] for w in workers]
            center = statistics.median(medians) if qualified else None
            cold = {}
            if qualified and all(w["cold_complete"] for w in workers):
                for key in COLD_PHASES:
                    values = [w["cold_phases_ns"][key] for w in workers]
                    cold[key] = {"replicate_values_ns": values, "median_ns": statistics.median(values),
                                 "range_ns": [min(values), max(values)]}
            cells.append({"model": model, "route": route, "threads": threads,
                          "label": label, "color": color, "qualified": qualified,
                          "qualification_issues": cell_issues, "workers": workers,
                          "median_of_process_median_group_mean_latency_ns": center,
                          "replicate_median_latency_range_ns": [min(medians), max(medians)] if qualified else None,
                          "evaluations_per_second": 1e9 / center if qualified else None,
                          "evaluations_per_second_range": [1e9 / max(medians), 1e9 / min(medians)] if qualified else None,
                          "cold": cold})
    for cell in cells:
        for threads, name in ((1, "standard_cpu_1_thread"),
                              (cell["threads"], "standard_cpu_same_threads")):
            baseline = next(c for c in cells if c["model"] == cell["model"]
                            and c["route"] == "standard_cpu" and c["threads"] == threads)
            # Ratios need the entire compared cell plus its entire CPU baseline.
            comparable = cell["qualified"] and baseline["qualified"]
            if comparable:
                keys = ("host", "python", "python_version", "platform", "numpy_version", "harness_sha256")
                comparable = all(cell["workers"][0]["raw_record"].get(k)
                                 == baseline["workers"][0]["raw_record"].get(k) for k in keys)
            cell["speedup_vs_" + name] = (
                baseline["median_of_process_median_group_mean_latency_ns"]
                / cell["median_of_process_median_group_mean_latency_ns"] if comparable else None)
    return {"artifact_type": "public_inference_report", "required_revision": REVISION,
            "input_directory": str(directory), "case_sha256": case_hash, "case": case,
            "case_issues": issues, "parity_tolerance": TOLERANCE,
            "aggregation": "Median of three per-process medians of five group mean latencies; "
                           "evaluations/s = 1e9 / that latency. Whiskers invert the observed "
                           "minimum/maximum process median latencies. No confidence intervals.",
            "speedup_definition": "Baseline aggregate latency divided by route aggregate latency; "
                                  "requires both complete three-process cells to qualify.",
            "timing_boundary": BOUNDARY, "limitations": LIMITATIONS, "cells": cells,
            "qualified_cells": sum(c["qualified"] for c in cells),
            "expected_cells": len(cells), "renderer_sha256": sha256(Path(__file__).resolve())}


def render_plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    figures = []

    def decorate(ax, model, xlabel):
        ax.set_title(MODELS[model], loc="left", fontweight="bold")
        ax.set_yticks(range(5), [c[2] for c in CONFIGURATIONS])
        ax.set_ylim(4.5, -0.5)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.2, which="both")
        ax.set_axisbelow(True)

    def save(fig, name):
        for suffix in ("png", "svg"):
            filename = f"{name}.{suffix}"
            fig.savefig(output / filename, dpi=180, bbox_inches="tight")
            figures.append(filename)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), layout="constrained")
    for ax, model in zip(axes, MODELS):
        subset = [c for c in summary["cells"] if c["model"] == model]
        for y, cell in enumerate(subset):
            if cell["qualified"]:
                center = cell["evaluations_per_second"]
                low, high = cell["evaluations_per_second_range"]
                ax.errorbar(center, y, xerr=[[center - low], [high - center]],
                            color=cell["color"], marker="o", capsize=4, linewidth=1.6)
            else:
                ax.text(0.03, y, "Unqualified / incomplete", transform=ax.get_yaxis_transform(),
                        va="center", color="#8B3A3A", fontsize=9)
        decorate(ax, model, "Public likelihood evaluations / second · log scale")
        if not any(c["qualified"] for c in subset):
            ax.set_xlim(0.5, 2)
    fig.suptitle("Synthetic two-detector inference · 32 s at 2048 Hz\n"
                 "End-to-end public update + host scalar likelihood · median and observed range of 3 process medians",
                 fontsize=13)
    save(fig, "inference")
    if any(c["cold"] for c in summary["cells"]):
        from matplotlib.lines import Line2D
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained")
        for ax, model in zip(axes, MODELS):
            subset = [c for c in summary["cells"] if c["model"] == model]
            for y, cell in enumerate(subset):
                if not cell["cold"]:
                    ax.text(0.03, y, "No qualified cold data", transform=ax.get_yaxis_transform(),
                            va="center", color="#8B3A3A", fontsize=9)
                for index, (key, (_, color)) in enumerate(COLD_PHASES.items()):
                    if key not in cell["cold"]:
                        continue
                    phase = cell["cold"][key]
                    center = phase["median_ns"] / 1e6
                    low, high = (v / 1e6 for v in phase["range_ns"])
                    ax.errorbar(center, y + (index - 1.5) * 0.12,
                                xerr=[[center - low], [high - center]], marker="o", markersize=4,
                                color=color, capsize=3, linewidth=1.3)
            decorate(ax, model, "Milliseconds · log scale")
            if not any(c["cold"] for c in subset):
                ax.set_xlim(0.5, 2)
        handles = [Line2D([], [], marker="o", color=color, label=label)
                   for label, color in COLD_PHASES.values()]
        fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
        fig.suptitle("Separate measured cold phases · public inference\n"
                     "Median and observed range across 3 fresh processes; these phases do not cover total startup",
                     fontsize=13)
        save(fig, "inference-cold")
    return figures


def markdown(summary):
    lines = ["# Public inference benchmark", "", BOUNDARY, "",
             f"Required checkout: `{REVISION}`, clean before and after every worker.", "",
             "Synthetic H1/L1, 32 s at 2048 Hz, TaylorF2, float64/complex128. "
             "These measurements concern public likelihood calls; no sampler evidence is provided.", "",
             summary["aggregation"], "", "Parity requires all 12 points in all three processes "
             "for each model/configuration; both loglikelihood and loglr use "
             "`abs(actual - CPU_reference) <= 2e-5 + 2e-7 * abs(CPU_reference)`.", "",
             f"Qualified configurations: {summary['qualified_cells']}/{summary['expected_cells']}.", "",
             "| Model | Route | Qualified workers | Evaluations/s | Process range, evals/s | Median latency, ms | Ratio vs CPU 1 | Ratio vs CPU same threads |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]

    def fmt(value):
        return f"{value:.4g}" if value is not None else "—"

    for cell in summary["cells"]:
        interval = cell["evaluations_per_second_range"]
        extent = "–".join(fmt(v) for v in interval) if interval else "—"
        latency = cell["median_of_process_median_group_mean_latency_ns"]
        lines.append(f"| {MODELS[cell['model']]} | {cell['label']} | "
                     f"{sum(w['qualified'] for w in cell['workers'])}/3 | "
                     f"{fmt(cell['evaluations_per_second'])} | {extent} | "
                     f"{fmt(latency / 1e6 if latency is not None else None)} | "
                     f"{fmt(cell['speedup_vs_standard_cpu_1_thread'])} | "
                     f"{fmt(cell['speedup_vs_standard_cpu_same_threads'])} |")
    lines += ["", "Ratios require a qualified route and CPU baseline with matching runtime identity. "
              "CUDA uses the one-thread CPU baseline. A qualified worker count can still fail "
              "the shared case or runtime checks; unqualified cells have no published rates.", "",
              "![Public inference throughput](inference.png)", "", "## Cold phases", "",
              "| Model | Route | Phase | Median ms | Process range ms |",
              "|---|---|---|---:|---:|"]
    for cell in summary["cells"]:
        for key, phase in cell["cold"].items():
            extent = "–".join(fmt(v / 1e6) for v in phase["range_ns"])
            lines.append(f"| {MODELS[cell['model']]} | {cell['label']} | {COLD_PHASES[key][0]} | "
                         f"{fmt(phase['median_ns'] / 1e6)} | {extent} |")
    if any(c["cold"] for c in summary["cells"]):
        lines += ["", "![Separate cold timing phases](inference-cold.png)"]
    else:
        lines += ["", "No complete, qualified cold-phase measurements."]
    lines += ["", "## Qualification failures", ""]
    failed = [c for c in summary["cells"] if not c["qualified"]]
    if not failed:
        lines.append("All ten configurations qualify.")
    for cell in failed:
        lines += [f"### {MODELS[cell['model']]} · {cell['label']}", ""]
        lines.extend("- " + issue.replace("\n", " ") for issue in cell["qualification_issues"])
        for worker in cell["workers"]:
            raw = worker["raw_record"]
            reason = raw.get("error") or raw.get("reason") or raw.get("reference", {}).get("error")
            if reason:
                lines.append(f"- Replicate {worker['replicate']} recorded failure: {reason}")
        lines.append("")
    lines += ["## Scope and interpretation", ""]
    lines.extend("- " + item for item in LIMITATIONS)
    lines += ["", "## Provenance", "", f"- Case SHA256: `{summary['case_sha256']}`",
              f"- Renderer SHA256: `{summary['renderer_sha256']}`",
              f"- Input directory: `{summary['input_directory']}`", "",
              "`inference-summary.json` preserves every expected configuration, source-file SHA256, "
              "qualification reason, raw worker record, parity value/tolerance, per-process group "
              "latency, cold interval, device/storage description and runtime provenance.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory, output = args.input_dir.resolve(), args.output_dir.resolve()
    if directory == output:
        parser.error("input and report directories must differ")
    output.mkdir(parents=True, exist_ok=True)
    summary = aggregate(directory)
    # Preserve the complete evidence even if the plotting dependency is missing.
    (output / "inference-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "inference-summary.md").write_text(markdown(summary))
    summary["figures"] = render_plots(summary, output)
    (output / "inference-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"qualified_cells": summary["qualified_cells"],
                      "expected_cells": summary["expected_cells"], "figures": summary["figures"]}))


if __name__ == "__main__":
    main()
