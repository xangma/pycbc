#!/usr/bin/env python3
"""Independently verify and render a complete public TaylorF2 benchmark run.

Usage: waveform_report.py --input-dir FULLRUN --output-dir REPORT
Only the standard library is needed for validation; rendering needs matplotlib.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

SHA = "607bce53ead14f12af32552a5b2441d3bc667267"
BATCHES = (1, 8, 32)
CPU_ROUTES = ("standard-cpu", "torch-cpu-scalar", "torch-cpu-batch")
CUDA_ROUTES = ("torch-cuda-scalar", "torch-cuda-batch", "torch-cuda-triton-batch")
LABELS = {"standard-cpu": "Standard CPU / LAL loop",
          "torch-cpu-scalar": "Torch CPU scalar loop",
          "torch-cpu-batch": "Torch CPU batch",
          "torch-cuda-scalar": "Torch CUDA scalar loop",
          "torch-cuda-batch": "Torch CUDA batch",
          "torch-cuda-triton-batch": "Requested CUDA Triton batch"}


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_number(actual, expected):
    return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0))


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def cell_key(route, batch, threads, precision):
    return f"{route}-b{batch}-t{threads}-{precision}"


def expected_cells():
    for threads in (1, 4):
        routes = CPU_ROUTES + CUDA_ROUTES if threads == 1 else CPU_ROUTES
        for route in routes:
            for batch in BATCHES:
                for precision in ("double", "single"):
                    yield route, batch, threads, precision


def verify_parity(value, batch, gate, tolerance, name):
    errors = []
    if not isinstance(value, dict):
        return [f"{name}: missing parity record"]
    if value.get("passed") is not True or value.get("metadata_equal") is not True:
        errors.append(f"{name}: reported parity/metadata failure")
    if value.get("gate") != gate or value.get("tolerance") != tolerance:
        errors.append(f"{name}: unexpected gate or tolerance")
    metrics = value.get("metrics", [])
    identities = {(m.get("polarization"), m.get("row")) for m in metrics}
    if len(metrics) != 2 * batch or identities != {(p, r) for p in (0, 1) for r in range(batch)}:
        errors.append(f"{name}: incomplete polarization/row coverage")
    for metric in metrics:
        if any(metric.get(field) is not True for field in ("finite", "exact_zero_support", "passed")):
            errors.append(f"{name}: nonfinite values or zero-support/parity failure")
            break
        for field in ("relative_l2", "max_pointwise_relative", "max_relative_amplitude", "max_wrapped_phase_radians"):
            value = metric.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                errors.append(f"{name}: missing/nonfinite {field}")
        metric_key = "relative_l2" if gate == "relative_l2" else "max_pointwise_relative"
        if gate is not None:
            value = metric.get(metric_key)
            if not isinstance(value, (int, float)) or not (
                    value < tolerance if gate == "relative_l2" else value <= tolerance):
                errors.append(f"{name}: recorded numerical error exceeds tolerance")
                break
    return errors


def verify_worker(raw, job, cell, manifest):
    route, batch, threads, precision = cell
    errors = []
    expected_identity = dict(route=route, batch=batch, threads=threads, precision=precision,
                             replicate=job.get("replicate"))
    for key, value in expected_identity.items():
        if raw.get(key) != value:
            errors.append(f"worker identity mismatch: {key}")
    source = raw.get("source", {})
    if source.get("sha") != SHA or source.get("clean") is not True or source.get("status_porcelain") != "":
        errors.append("source is not the exact clean published SHA")
    if not source.get("tree") or not source.get("waveform_source_sha256"):
        errors.append("source tree/hash provenance missing")
    if raw.get("harness_sha256") != manifest.get("worker_sha256"):
        errors.append("worker hash differs from campaign manifest")
    status = raw.get("status")
    if status not in ("ok", "failed", "unsupported") or job.get("status") != status:
        errors.append("worker/manifest statuses disagree or are invalid")
    if status in ("ok", "unsupported") and (job.get("returncode") != 0 or job.get("timeout")):
        errors.append("successful/unsupported record has failed or timed-out process")
    if raw.get("runtime", {}).get("pid") != job.get("pid"):
        errors.append("worker PID differs from manifest")
    if status != "ok":
        if not raw.get("reason"):
            errors.append("non-success record has no reason")
        if status == "unsupported" and (precision == "single" or "triton" in route):
            if raw.get("dispatch", {}).get("actual") is not None or "timing" in raw:
                errors.append("unsupported source capability must not have an actual route or timings")
        return errors, None
    if source.get("status_after") != "":
        errors.append("clean source after execution was not confirmed")
    if precision != "double" or "triton" in route:
        errors.append("unsupported precision/Triton route reported successful timing")
    runtime = raw.get("runtime", {})
    if runtime.get("torch_threads") != threads or runtime.get("torch_interop_threads") != 1:
        errors.append("effective Torch thread count differs from requested policy")
    for module in ("pycbc", "taylorf2_torch"):
        origin = runtime.get("module_origins", {}).get(module)
        try:
            Path(origin).relative_to(Path(source["root"]))
        except (TypeError, ValueError, KeyError):
            errors.append(f"{module} origin is outside recorded source checkout")
    is_batch, is_cuda = route.endswith("batch"), "cuda" in route
    expected_dispatch = ({"native_scalar": 0, "native_batch": 0, "lal_fd": batch}
                         if route == "standard-cpu" else
                         {"native_scalar": 0 if is_batch else batch,
                          "native_batch": 1 if is_batch else 0, "lal_fd": 0})
    dispatch = raw.get("dispatch", {})
    expected_actual = ("standard LAL CPU" if route == "standard-cpu" else
                       f"native Torch {'cuda:0' if is_cuda else 'cpu'} {'batch' if is_batch else 'scalar loop'}")
    if dispatch.get("probe_counts") != expected_dispatch or dispatch.get("passed") is not True:
        errors.append("actual native/LAL dispatch counts do not match the requested route")
    if dispatch.get("actual") != expected_actual or dispatch.get("triton_actual") is not False:
        errors.append("actual dispatch label/Triton declaration mismatch")
    output = raw.get("output", {})
    if output.get("dtypes") != ["complex128", "complex128"] or output.get("shapes") != [[batch, 4097]] * 2:
        errors.append("unexpected dtype or waveform shape")
    devices = output.get("devices", [])
    if not devices or not all(str(d).startswith("cuda") if is_cuda else d == "cpu" for d in devices):
        errors.append("output device differs from actual route")
    for name, gate, tolerance in (
            ("parity_scalar_vs_cpu_reference", "relative_l2", 1e-11),
            ("parity_direct_vs_cpu_reference", None if is_batch else "relative_l2", None if is_batch else 1e-11)):
        errors.extend(verify_parity(raw.get(name), batch, gate, tolerance, name))
    if is_batch:
        errors.extend(verify_parity(raw.get("parity_batch_vs_native_scalar"), batch,
                                    "pointwise_relative", 2e-10, "parity_batch_vs_native_scalar"))
    timing = raw.get("timing", {})
    totals, samples = timing.get("sample_total_seconds", []), timing.get("sample_seconds_per_call", [])
    inner = timing.get("inner_calls_per_sample")
    if (len(totals) != 5 or len(samples) != 5 or not all(positive(x) for x in totals + samples)
            or not isinstance(inner, int) or isinstance(inner, bool) or inner < 1):
        errors.append("need five finite positive samples and a positive integer inner count")
        return errors, None
    per_call = [t / inner for t in totals]
    if not all(same_number(a, b) for a, b in zip(samples, per_call)):
        errors.append("sample durations do not equal total time / inner count")
    median = statistics.median(per_call)
    for key, value in (("median_seconds_per_call", median),
                       ("median_seconds_per_waveform", median / batch),
                       ("waveforms_per_second", batch / median)):
        if not same_number(timing.get(key), value):
            errors.append(f"worker aggregation mismatch: {key}")
    if not positive(timing.get("cold_call_seconds")):
        errors.append("cold timing is missing/nonpositive")
    if is_cuda and timing.get("synchronization") != "CUDA synchronize before/after each timed block":
        errors.append("CUDA synchronization policy was not confirmed")
    return errors, median


def verify_campaign(input_dir):
    manifest_path, summary_path = input_dir / "manifest.json", input_dir / "summary.json"
    manifest, original = read_json(manifest_path), read_json(summary_path)
    global_errors = []
    if manifest.get("expected_sha") != SHA:
        global_errors.append("manifest expected SHA does not match published main")
    if not manifest.get("finished_utc"):
        global_errors.append("campaign has not recorded completion")
    if not manifest.get("worker_sha256"):
        global_errors.append("manifest has no worker hash")
    jobs_by_key = defaultdict(list)
    for job in manifest.get("jobs", []):
        jobs_by_key[job.get("key")].append(job)
    expected_keys = {cell_key(*cell) for cell in expected_cells()}
    if set(jobs_by_key) - expected_keys:
        global_errors.append("manifest contains cells outside the fixed full-run matrix")
    if set(original.get("groups", {})) != expected_keys:
        global_errors.append("existing summary does not cover the complete expected matrix")
    cells, raw_hashes, runtime_inventory = {}, {}, {}
    parameters_by_batch, source_fingerprints = {}, set()
    for cell in expected_cells():
        route, batch, threads, precision = cell
        key = cell_key(*cell)
        jobs = sorted(jobs_by_key.get(key, []), key=lambda j: j.get("replicate", 0))
        errors, records, medians, cold = [], [], [], []
        if [j.get("replicate") for j in jobs] != [1, 2, 3]:
            errors.append("exactly three independent replicates numbered 1,2,3 are required")
        identities = []
        for job in jobs:
            # Campaign paths may come from another host; only read basenames in input-dir.
            raw_path = input_dir / Path(job.get("output", "")).name
            try:
                raw = read_json(raw_path)
            except (OSError, ValueError) as exc:
                errors.append(f"cannot read {raw_path.name}: {exc}")
                continue
            raw_hashes[raw_path.name] = digest(raw_path)
            record_errors, median = verify_worker(raw, job, cell, manifest)
            errors.extend(f"replicate {job.get('replicate')}: {e}" for e in record_errors)
            runtime = raw.get("runtime", {})
            identities.append((runtime.get("hostname"), runtime.get("pid"), runtime.get("started_utc")))
            runtime_key = runtime.get("hostname", "unknown")
            runtime_inventory.setdefault(runtime_key, {k: v for k, v in runtime.items()
                                                       if k not in ("pid", "started_utc", "import_seconds", "worker_total_seconds")})
            if runtime.get("cuda_device"):
                runtime_inventory[runtime_key]["cuda_device"] = runtime["cuda_device"]
            source = raw.get("source", {})
            source_fingerprints.add(json.dumps({k: source.get(k) for k in ("sha", "tree", "waveform_source_sha256")}, sort_keys=True))
            params = raw.get("parameters")
            if params is None or len(params) != batch:
                errors.append(f"replicate {job.get('replicate')}: parameter rows missing")
            elif batch in parameters_by_batch and params != parameters_by_batch[batch]:
                errors.append(f"replicate {job.get('replicate')}: physical inputs differ across routes")
            else:
                parameters_by_batch[batch] = params
            records.append({"file": raw_path.name, "status": raw.get("status"),
                            "reason": raw.get("reason"), "actual_route": raw.get("dispatch", {}).get("actual"),
                            "verification_errors": record_errors})
            if median is not None and raw.get("status") == "ok":
                medians.append(median)
                cold.append(raw.get("timing", {}).get("cold_call_seconds"))
        if len(identities) != len(set(identities)):
            errors.append("replicates do not identify distinct subprocess executions")
        statuses = [r["status"] for r in records]
        original_cell = original.get("groups", {}).get(key, {})
        raw_ok = len(records) == 3 and statuses == ["ok"] * 3
        for name, expected in (("route", route), ("batch", batch), ("threads", threads),
                               ("precision", precision), ("replicate_count", len(records)),
                               ("valid_replicates", statuses.count("ok")), ("statuses", statuses), ("eligible", raw_ok)):
            if original_cell.get(name) != expected:
                errors.append(f"existing summary disagrees with raw records: {name}")
        aggregate = None
        if raw_ok and len(medians) == 3:
            center = statistics.median(medians)
            aggregate = {"replicate_medians_seconds_per_call": medians,
                         "median_seconds_per_call": center,
                         "min_replicate_median_seconds": min(medians),
                         "max_replicate_median_seconds": max(medians),
                         "waveforms_per_second": batch / center,
                         "min_replicate_throughput": batch / max(medians),
                         "max_replicate_throughput": batch / min(medians),
                         "cold_call_seconds_by_replicate": cold}
            for field in ("median_seconds_per_call", "min_replicate_median_seconds", "max_replicate_median_seconds", "waveforms_per_second"):
                if not same_number(original_cell.get(field), aggregate[field]):
                    errors.append(f"existing summary aggregation mismatch: {field}")
            for field, expected in (("replicate_medians_seconds", medians), ("cold_call_seconds", cold)):
                recorded = original_cell.get(field, [])
                if len(recorded) != 3 or not all(same_number(x, y) for x, y in zip(recorded, expected)):
                    errors.append(f"existing summary aggregation mismatch: {field}")
        eligible = raw_ok and aggregate is not None and not errors
        state = ("verified" if eligible else "unsupported" if statuses == ["unsupported"] * 3 and not errors
                 else "failed" if "failed" in statuses else "unverified")
        cells[key] = {"route": route, "route_label": LABELS[route], "batch": batch,
                      "threads": threads, "precision": precision, "status": state,
                      "eligible": eligible, "records": records, "verification_errors": errors,
                      "timing": aggregate if eligible else None}
    if len(source_fingerprints) != 1:
        global_errors.append("raw records do not share one exact source tree and source-file hash set")
    for batch, rows in parameters_by_batch.items():
        if 32 in parameters_by_batch and rows != parameters_by_batch[32][:batch]:
            global_errors.append(f"batch {batch} is not a prefix of the common physical inputs")
    if global_errors:
        for cell in cells.values():
            if cell["eligible"]:
                cell["eligible"], cell["status"], cell["timing"] = False, "unverified", None
    for key, cell in cells.items():
        if not cell["eligible"]:
            continue
        baseline = cells[cell_key("standard-cpu", cell["batch"], cell["threads"], cell["precision"])]
        if baseline["eligible"]:
            ratio = baseline["timing"]["median_seconds_per_call"] / cell["timing"]["median_seconds_per_call"]
            cell["timing"]["speedup_vs_standard_cpu_same_threads"] = ratio
            prior = original["groups"][key].get("speedup_vs_standard_cpu_same_threads")
            if not same_number(prior, ratio):
                cell["verification_errors"].append("existing summary speedup disagrees with raw medians")
                cell["eligible"], cell["status"], cell["timing"] = False, "unverified", None
    return {"schema": 1, "source_sha": SHA, "input_dir": str(input_dir),
            "verification_passed": not global_errors and not any(c["verification_errors"] for c in cells.values()),
            "global_verification_errors": global_errors,
            "expected_cells": len(expected_keys), "expected_replicates_per_cell": 3,
            "samples_per_replicate": 5, "runtime_inventory": runtime_inventory,
            "manifest_sha256": digest(manifest_path), "input_summary_sha256": digest(summary_path),
            "renderer_sha256": digest(Path(__file__)), "raw_file_sha256": raw_hashes,
            "method": "median of three independently recomputed worker medians; min/max replicate medians, not confidence intervals",
            "cold_definition": "first waveform call after imports, capability discovery and scheme entry; excluded from steady-state plots",
            "parity_limit": "Recorded parity metrics and gates are checked; waveform arrays are not stored, so this renderer does not rerun waveform generation.",
            "cells": cells}


def escaped(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown(report):
    lines = ["# TaylorF2 public API benchmark", "", f"Source: `{SHA}`.", "",
             f"Independent record verification: **{'passed' if report['verification_passed'] else 'failed'}**.", "",
             "4097 frequency bins, both polarizations, the same deterministic BNS parameter rows across routes. "
             "CPU/LAL and native scalar routes loop through rows; native batch routes use one public batch call. "
             "Timing includes host parameter conversion and allocation. CUDA timed blocks are synchronized.", "",
             "Throughput uses the median of three process medians, each calculated from five samples. "
             "Ranges are the observed min/max of the three replicate medians, **not confidence intervals**. "
             "A speedup below 1 means slower than the standard CPU/LAL baseline at the same batch size and host-thread count.", "",
             "![Verified TaylorF2 throughput](waveform.png)", "",
             "## Steady-state double precision", "",
             "| Actual route | Host threads | Batch | Waveforms/s | Replicate range (waveforms/s) | Speedup vs CPU/LAL | Status |",
             "| --- | ---: | ---: | ---: | --- | ---: | --- |"]
    for cell in report["cells"].values():
        if cell["precision"] != "double" or "triton" in cell["route"]:
            continue
        timing = cell["timing"]
        if timing:
            rate = f"{timing['waveforms_per_second']:,.2f}"
            interval = f"{timing['min_replicate_throughput']:,.2f}–{timing['max_replicate_throughput']:,.2f}"
            ratio = timing.get("speedup_vs_standard_cpu_same_threads")
            speedup = f"{ratio:.3f}×" if ratio is not None else "unavailable"
        else:
            rate, interval, speedup = "unavailable", "unavailable", "unavailable"
        lines.append(f"| {cell['route_label']} | {cell['threads']} | {cell['batch']} | {rate} | {interval} | {speedup} | {cell['status']} |")
    lines += ["", "## Cold calls, separately", "",
              "Milliseconds for the first waveform call in each fresh process, after imports, capability discovery and scheme entry. "
              "These are not cold driver/OS-cache measurements and are excluded from throughput and speedup calculations.", "",
              "| Actual route | Host threads | Batch | Replicate 1 (ms) | Replicate 2 (ms) | Replicate 3 (ms) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for cell in report["cells"].values():
        if cell["timing"]:
            values = " | ".join(f"{1000 * x:.4f}" for x in cell["timing"]["cold_call_seconds_by_replicate"])
            lines.append(f"| {cell['route_label']} | {cell['threads']} | {cell['batch']} | {values} |")
    lines += ["", "## Unsupported, failed or unverified cells", "",
              "No numeric zeros substitute for unsupported capabilities or missing results. "
              "The public CPU/CUDA API at this source SHA selects complex128 internally; it exposes no single precision selector or TaylorF2 Triton route. "
              "CUDA was requested with one host thread only.", "",
              "| Request | Precision | Host threads | Batch | Status | Reason / verification errors |",
              "| --- | --- | ---: | ---: | --- | --- |"]
    for cell in report["cells"].values():
        if cell["eligible"]:
            continue
        reasons = sorted({r["reason"] for r in cell["records"] if r.get("reason")})
        reasons.extend(cell["verification_errors"])
        if not reasons:
            reasons = report["global_verification_errors"] or ["No complete verified result"]
        lines.append(f"| {cell['route']} | {cell['precision']} | {cell['threads']} | {cell['batch']} | {cell['status']} | {escaped('; '.join(reasons))} |")
    lines += ["", "## Verification scope", "",
              "The renderer independently recomputes sample normalization, each worker median, the median/range across exactly three distinct subprocesses, "
              "throughput and baseline ratios. It checks exact clean source SHA before/after successful workers, source/harness hashes, module origins, "
              "dtype/device, full row/polarization coverage, dispatch counts, recorded numerical parity metrics, statuses and equal physical inputs. "
              "All three replicates must pass before a cell is plotted. The original summary is compared against these computations.", "",
              report["parity_limit"], "",
              "Native scalar versus CPU/LAL requires complex relative L2 <1e-11, exact zero support and matching metadata. "
              "Batch versus same-device native scalar requires pointwise complex relative error ≤2e-10 with zero absolute tolerance; "
              "those scalar outputs are independently checked against CPU/LAL. Direct batch-to-LAL complex, amplitude and phase metrics are retained in raw records.", "",
              "Runtime inventory and SHA256 hashes of every consumed raw record are in `waveform-summary.json`. "
              "Cold timing values remain separate. This bounded waveform workload does not establish end-to-end inference/search performance."]
    if report["global_verification_errors"]:
        lines += ["", "Global verification errors: " + escaped("; ".join(report["global_verification_errors"]))]
    return "\n".join(lines) + "\n"


def plot(report, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"standard-cpu": "#52616B", "torch-cpu-scalar": "#C86A2B",
              "torch-cpu-batch": "#27835D", "torch-cuda-scalar": "#865DA6", "torch-cuda-batch": "#2374AB"}
    markers = dict(zip(colors, ("o", "s", "D", "^", "P")))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#24323C", "text.color": "#24323C"}):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.1), sharey=True)
        any_values = False
        for ax, threads in zip(axes, (1, 4)):
            routes = CPU_ROUTES + CUDA_ROUTES[:2] if threads == 1 else CPU_ROUTES
            count, missing = 0, []
            for index, route in enumerate(routes):
                xs, ys, lows, highs = [], [], [], []
                offset = (index - (len(routes) - 1) / 2) * 0.035
                for position, batch in enumerate(BATCHES):
                    cell = report["cells"][cell_key(route, batch, threads, "double")]
                    xs.append(position + offset)
                    if not cell["eligible"]:
                        ys.append(math.nan)
                        lows.append(math.nan)
                        highs.append(math.nan)
                        missing.append(f"{LABELS[route]} B{batch}: {cell['status']}")
                        continue
                    value = cell["timing"]
                    center = value["waveforms_per_second"]
                    ys.append(center)
                    lows.append(max(0.0, center - value["min_replicate_throughput"]))
                    highs.append(max(0.0, value["max_replicate_throughput"] - center))
                    count += 1
                ax.errorbar(xs, ys, yerr=[lows, highs], color=colors[route], marker=markers[route],
                            markersize=5.5, linewidth=1.6, elinewidth=1.3, capsize=3,
                            linestyle="--" if route.endswith("scalar") else "-")
            any_values = any_values or count > 0
            ax.set_title(f"{threads} host thread{'s' if threads != 1 else ''}", loc="left", fontweight="bold", pad=13)
            ax.set_xticks(range(3), [str(x) for x in BATCHES])
            ax.set_xlim(-0.16, 2.16)
            ax.set_xlabel("Waveforms per public API workload (batch size)", labelpad=10)
            ax.grid(axis="y", which="major", alpha=0.18)
            ax.set_axisbelow(True)
            if not count:
                ax.text(0.5, 0.5, "No verified timing cells", ha="center", va="center", transform=ax.transAxes)
            panel_note = "CUDA uses one host thread; see left panel." if threads == 4 else "CPU/LAL and scalar routes loop over rows."
            if missing:
                panel_note += f"  {len(missing)} cells unavailable; see report."
            ax.text(0, -0.23, panel_note, transform=ax.transAxes, fontsize=8.5, color="#52616B", va="top", wrap=True)
        if any_values:
            axes[0].set_yscale("log")
            axes[0].set_ylabel("Waveforms / second (log scale)", labelpad=10)
        else:
            axes[0].set_yticks([])
        handles = [Line2D([0], [0], color=colors[r], marker=markers[r],
                          linestyle="--" if r.endswith("scalar") else "-", label=LABELS[r]) for r in colors]
        fig.suptitle("TaylorF2 public API throughput", x=0.065, ha="left", y=0.98, fontsize=18, fontweight="bold")
        fig.text(0.065, 0.925, f"PyCBC {SHA[:12]}  •  complex128  •  4097 frequency bins  •  both polarizations", fontsize=10.5)
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.058, 0.885), ncol=3, frameon=False, fontsize=9)
        fig.text(0.065, 0.035, "Median of 3 independent process medians × 5 samples. Whiskers: observed replicate min/max, not confidence intervals.\n"
                 "Cold calls are reported separately. Unsupported single precision and Triton requests have no plotted numeric values.", fontsize=9, color="#52616B")
        fig.subplots_adjust(left=0.08, right=0.98, top=0.72, bottom=0.24, wspace=0.13)
        fig.savefig(output_dir / "waveform.png", dpi=180, facecolor="white")
        fig.savefig(output_dir / "waveform.svg", facecolor="white", metadata={"Date": None})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    report = verify_campaign(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "waveform-summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output_dir / "waveform-summary.md").write_text(markdown(report))
    plot(report, output_dir)
    print(json.dumps({"verification_passed": report["verification_passed"],
                      "eligible_cells": sum(c["eligible"] for c in report["cells"].values()),
                      "total_cells": len(report["cells"]), "output_dir": str(output_dir)}))
    return 0 if report["verification_passed"] else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
