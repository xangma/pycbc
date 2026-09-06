#!/usr/bin/env python3
"""Summarize 20 untimed dispatch probes without importing benchmark code.

Usage: python probe_report.py --input-dir ../supplement/probes --output-dir .
Instrumented child timings are intentionally absent from all outputs.
"""

import argparse
import hashlib
import json
from pathlib import Path


REVISIONS = {
    "main": "607bce53ead14f12af32552a5b2441d3bc667267",
    "cpu": "1a2ebea088d9e0a31cbb22c19ad24f96ffea2b7c",
}
PROBE_SHA256 = "792ea7cb4b90bca5fb6e9dac65c5b762d905f7f3c240a95ff18698232fa66b32"
HELPERS = {
    "CPU correlation": "_try_cpu_native_batch_correlate",
    "CUDA correlation": "_try_cuda_native_batch_correlate",
    "CPU peaks": "_try_torch_cpu_native_batch_peak_values",
    "CUDA peaks": "_try_torch_cuda_native_batch_peak_values",
    "FFTW single": "_execute_fftw_cpu_plan",
    "FFTW batch": "_execute_fftw_cpu_batch_plan",
    "MKL IFFT": "_execute_mkl_cpu_ifft_plan",
}
UNSAFE_HELPER = "_torch_batch_peak_and_threshold_gpu"
FLAGS = {
    "CPU correlation": "PYCBC_TORCH_CPU_NATIVE_BATCH_CORRELATE",
    "CUDA correlation": "PYCBC_TORCH_CUDA_NATIVE_BATCH_CORRELATE",
    "CPU peaks": "PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK",
    "CUDA peaks": "PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK",
    "FFTW batch": "PYCBC_TORCH_CPU_FFTW_BATCH",
    "MKL IFFT": "PYCBC_TORCH_CPU_MKL_IFFT",
    "On-device peaks": "PYCBC_TORCH_ONDEVICE_PEAKS",
}
LIMITATIONS = [
    "These are instrumented dispatch observations, not timing or speed measurements. Child-driver timings are excluded.",
    "Success means the observed helper returned its admission/success signal; attempts include rejected admissions. Only the named helpers are observed.",
    "0/0 means an available helper was not called. It does not by itself establish that a route was disabled or that a downstream fallback ran.",
    "Unavailable means the symbol is absent on that source head; its attempt count is unknown, not zero.",
    "The probe records fallback counts but does not instrument internal rejection reasons. Missing reasons remain explicitly unrecorded; requested flags are not substituted for observed reasons.",
    "Main native CPU requests correlation and FFTW batching. Optional CPU additionally requests native batch peaks. Requested features can still be bypassed or rejected.",
    "The on-device peak helper exclusion requires the probe assertion, a disabled effective flag, and an available helper with zero observed attempts.",
    "All cells use FFT length 131072 and batches 8 or 32. Batch 1 is not included in this admission matrix.",
]


def expected_cells():
    cells = [
        (head, route, threads, batch)
        for head in ("main", "cpu")
        for threads in (1, 4)
        for route in ("torch_cpu", "torch_cpu_native")
        for batch in (8, 32)
    ]
    cells += [
        ("main", route, 1, batch)
        for route in ("torch_cuda", "torch_cuda_native")
        for batch in (8, 32)
    ]
    return cells


def flag_label(flag):
    if flag is None:
        return "not recorded on head"
    if flag.get("applies_to_route") is False:
        return "not applicable"
    enabled = flag.get("enabled")
    if enabled is None:
        return "unknown"
    return (
        ("on" if enabled else "off")
        + " ("
        + str(flag.get("selection", "unknown"))
        + ")"
    )


def summarize_counter(counter):
    result = {"recorded_counter": counter}
    if counter is None:
        result.update(observation="missing counter", consistent=False)
        return result
    if counter.get("available") is False:
        result.update(observation="helper unavailable", consistent=True)
        return result
    names = ("attempts", "successes", "fallbacks", "exceptions")
    counts = {name: counter.get(name) for name in names}
    valid = counter.get("available") is True and all(
        type(v) is int and v >= 0 for v in counts.values()
    )
    valid = valid and counts["attempts"] == sum(counts[name] for name in names[1:])
    result["consistent"] = valid
    if not valid:
        result["observation"] = "invalid counter"
        return result
    result.update(counts)
    result["observation"] = (
        "not called"
        if counts["attempts"] == 0
        else "exception observed"
        if counts["exceptions"]
        else "all attempts succeeded"
        if counts["successes"] == counts["attempts"]
        else "some attempts succeeded"
        if counts["successes"]
        else "all attempts fell back"
    )
    result["recorded_reason_fields"] = {
        key: value
        for key, value in counter.items()
        if "reason" in key.lower() or "error" in key.lower()
    }
    if counts["fallbacks"]:
        result["fallback_reason_status"] = (
            "recorded" if result["recorded_reason_fields"] else "not recorded by probe"
        )
    return result


def load_row(path, cell):
    head, route, threads, batch = cell
    if not path.exists():
        return {
            "head": head,
            "route": route,
            "threads": threads,
            "batch": batch,
            "file": path.name,
            "qualified": False,
            "status": "missing",
            "issues": ["expected probe artifact is missing"],
            "helpers": {},
            "unsafe_helper_excluded_verified": False,
            "configuration": {},
        }
    raw = path.read_bytes()
    data = json.loads(raw)
    issues = []
    checks = {
        "probe did not pass": data.get("status") == "passed",
        "wrong measured revision": data.get("revision") == REVISIONS[head],
        "wrong expected revision": data.get("expected_revision") == REVISIONS[head],
        "source is dirty or cleanliness missing": data.get("tracked_clean") is True,
        "wrong probe source hash": data.get("harness_sha256") == PROBE_SHA256,
        "wrong route, threads, batch, or FFT size": (
            data.get("route"),
            data.get("threads"),
            data.get("batch"),
            data.get("size"),
        )
        == (route, threads, batch, 131072),
    }
    issues.extend(message for message, passed in checks.items() if not passed)
    configuration = data.get("configuration", {})
    flags = configuration.get("feature_flags", {})
    helpers = {
        label: summarize_counter(data.get("counts", {}).get(name))
        for label, name in HELPERS.items()
    }
    for label, result in helpers.items():
        if not result["consistent"]:
            issues.append(f"{label}: missing or inconsistent counts")
    unsafe = summarize_counter(data.get("counts", {}).get(UNSAFE_HELPER))
    excluded = (
        data.get("unsafe_helper_excluded") is True
        and flags.get(FLAGS["On-device peaks"], {}).get("enabled") is False
        and unsafe["consistent"]
        and unsafe.get("attempts") == 0
    )
    if not excluded:
        issues.append("on-device peak helper exclusion is not verified")
    return {
        "head": head,
        "revision": data.get("revision"),
        "route": route,
        "threads": threads,
        "batch": batch,
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "status": data.get("status"),
        "qualified": not issues,
        "issues": issues,
        "helpers": helpers,
        "unsafe_helper_excluded_verified": excluded,
        "unsafe_helper": unsafe,
        "configuration": configuration,
        "backend_import_environment": data.get("backend_import_environment"),
        "source_root": data.get("source_root"),
        "pycbc_module": data.get("pycbc_module"),
        "harness_sha256": data.get("harness_sha256"),
        "error": data.get("error"),
        "recorded_reason_fields": {
            key: value for key, value in data.items() if "reason" in key.lower()
        },
    }


def count_label(counter):
    if not counter:
        return "missing"
    if counter["observation"] == "helper unavailable":
        return "unavailable"
    if not counter["consistent"]:
        return "invalid"
    return f"{counter['successes']}/{counter['attempts']}"


def markdown(summary):
    lines = [
        "Untimed dispatch observations from the fresh live workload probes.",
        "",
        f"{summary['qualified_rows']}/{summary['expected_rows']} cells qualified. Each counter below is **successes / attempts**, not a performance ratio. Raw counters and errors are retained in the JSON summary.",
        "",
        "| Head | Route | Threads | Batch | "
        + " | ".join(HELPERS)
        + " | On-device helper excluded | Status |",
        "| --- | --- | ---: | ---: | "
        + " | ".join(["---:"] * len(HELPERS))
        + " | --- | --- |",
    ]
    for row in summary["rows"]:
        counts = " | ".join(count_label(row["helpers"].get(label)) for label in HELPERS)
        status = "qualified" if row["qualified"] else "; ".join(row["issues"])
        excluded = (
            "verified" if row["unsafe_helper_excluded_verified"] else "not verified"
        )
        lines.append(
            f"| {row['head']} | {row['route']} | {row['threads']} | {row['batch']} | {counts} | {excluded} | {status} |"
        )
    lines += [
        "",
        "Requested configuration (effective flags and their selection, distinct from actual admissions):",
        "",
        "| Head | Route | " + " | ".join(FLAGS) + " |",
        "| --- | --- | " + " | ".join(["---"] * len(FLAGS)) + " |",
    ]
    for row in summary["configurations"]:
        values = " | ".join(
            flag_label(row["feature_flags"].get(flag)) for flag in FLAGS.values()
        )
        lines.append(f"| {row['head']} | {row['route']} | {values} |")
    lines += [
        "",
        "Fallback and exception observations (unrecorded causes remain unknown):",
        "",
        "| Probe | Helper | Fallbacks | Exceptions | Recorded reason |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    observations = 0
    for row in summary["rows"]:
        for label, counter in row["helpers"].items():
            if counter.get("fallbacks", 0) or counter.get("exceptions", 0):
                reasons = counter.get("recorded_reason_fields")
                text = (
                    json.dumps(reasons, sort_keys=True)
                    if reasons
                    else "not recorded by probe"
                )
                lines.append(
                    f"| {row['file']} | {label} | {counter.get('fallbacks', 'unknown')} | {counter.get('exceptions', 'unknown')} | {text.replace('|', '&#124;')} |"
                )
                observations += 1
        if row.get("error"):
            lines.extend(
                [
                    "",
                    f"Probe error `{row['file']}`: `{json.dumps(row['error'], sort_keys=True)}`",
                    "",
                ]
            )
    if not observations:
        lines.append("| — | No recorded fallbacks or exceptions | — | — | — |")
    lines += ["", "Interpretation limits:", ""]
    lines += [f"- {limit}" for limit in LIMITATIONS]
    lines += ["", "Source revisions:", ""]
    lines += [f"- {head}: `{revision}`." for head, revision in REVISIONS.items()]
    lines += [f"- Probe source SHA-256: `{PROBE_SHA256}`.", "", "Input hashes:", ""]
    lines += [
        f"- `{row['file']}`: `{row.get('sha256', 'missing')}`."
        for row in summary["rows"]
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    configurations = {}
    for cell in expected_cells():
        head, route, threads, batch = cell
        path = args.input_dir / f"probe-{head}-{route}-t{threads}-b{batch}.json"
        row = load_row(path, cell)
        rows.append(row)
        key = (head, route)
        flags = row["configuration"].get("feature_flags", {})
        if flags:
            if key in configurations and configurations[key] != flags:
                row["qualified"] = False
                row["issues"].append(
                    "effective flags differ across cells with the same head and route"
                )
            else:
                configurations[key] = flags
    summary = {
        "schema_version": 1,
        "scope": "untimed dispatch instrumentation; no timing claims",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "revisions": REVISIONS,
        "expected_probe_sha256": PROBE_SHA256,
        "expected_rows": 20,
        "qualified_rows": sum(r["qualified"] for r in rows),
        "all_unsafe_helper_exclusions_verified": all(
            r["unsafe_helper_excluded_verified"] for r in rows
        ),
        "rows": rows,
        "configurations": [
            {"head": head, "route": route, "feature_flags": flags}
            for (head, route), flags in configurations.items()
        ],
        "limitations": LIMITATIONS,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dispatch-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "dispatch-summary.md").write_text(markdown(summary))
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "qualified_rows",
                    "expected_rows",
                    "all_unsafe_helper_exclusions_verified",
                )
            }
        )
    )
    return 0 if summary["qualified_rows"] == 20 else 1


if __name__ == "__main__":
    raise SystemExit(main())
