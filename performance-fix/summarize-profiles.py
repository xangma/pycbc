"""Summarize twelve complete, paired diagnostic profiles; never benchmark them.

Run with standard-library Python after fetching postchecks/profiles, commands,
and scripts-sha256.json. All twelve baseline/candidate pairs must be present.
Only profile-summary.json and profile-summary.md are written, after validation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import pstats
import sys


FILTER = "pycbc/filter/matchedfilter.py"
TORCH_FILTER = "pycbc/filter/matchedfilter_torch.py"
WAVEFORM = "pycbc/waveform/taylorf2_torch.py"
TARGETS = {
    FILTER: {
        "process_data",
        "_process_batch",
        "_torch_batch_peak_values",
        "_torch_batch_peak_magnitudes",
        "_try_torch_cuda_native_batch_peak_values",
    },
    TORCH_FILTER: {
        "_batch_outputs_are_disjoint",
        "_spans_overlap",
        "_batch_tensor_contract",
        "_cuda_batch_tensor_contract",
        "_logical_storage_span",
        "_has_autograd_state",
        "_same_array_tensors",
        "standard_peak_tensor",
        "batch_correlate_execute",
    },
    WAVEFORM: {
        "taylorf2_fd_batch",
        "taylorf2_aligned_phasing",
        "_evaluate_phase_polynomial",
        "_batch_validate",
    },
}
FOCUS_OPS = {
    "aten::log",
    "aten::addcmul",
    "aten::mul",
    "aten::add",
    "aten::pow",
    "aten::square",
    "aten::sum",
    "aten::argmax",
    "aten::cat",
    "aten::split",
    "aten::slice",
    "aten::narrow",
    "aten::view_as_real",
}
METHOD = [
    "Diagnostic attribution only: instrumented durations are not throughput "
    "evidence, speedup estimates, or replicated timing measurements.",
    "Each pstats file covers one warmed public call. Each Torch trace covers "
    "a separate warmed public call; do not add measurements across these calls.",
    "cProfile self time excludes profiled callees; cumulative time includes "
    "callees. Nested cumulative times must not be added. CUDA CPU times can "
    "include launch or synchronization costs and are not GPU kernel durations.",
    "Operator counts come from every complete cpu_op event in trace.json, "
    "not the truncated operators.txt table. Their inclusive CPU durations "
    "can overlap through nesting and must not be summed as wall time.",
    "Chrome trace event durations are microseconds; displayTimeUnit is a "
    "viewer preference. Tables below use milliseconds for cProfile costs.",
    "Absent functions have zero observed calls in that captured call; this "
    "does not establish that the implementation or capability is absent.",
    "Waveform timing.json supplies workload metadata only; its timing samples "
    "and derived throughput are deliberately excluded from this report.",
    "Baseline provenance is the supplied source-environment snapshot and "
    "embedded profile paths; no contemporaneous baseline command receipt or "
    "baseline script-hash receipt is available. Candidate command receipts "
    "and output/script hashes are validated against supplied source identity.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def receipt(path):
    return {
        "path": str(path.resolve()),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
    }


def read_json(path):
    return json.loads(path.read_text())


def matrix(root):
    cases = []
    for route in ("torch_cpu", "torch_cpu_native", "torch_cuda", "torch_cuda_native"):
        for batch in (32, 1024):
            label = f"profile-live-{route}-b{batch}-t1"
            cases.append(
                {
                    "kind": "live",
                    "route": route,
                    "device": "cuda" if "cuda" in route else "cpu",
                    "batch": batch,
                    "threads": 1,
                    "label": label,
                    "baseline": root / "profiles-baseline" / f"{route}-b{batch}",
                    "candidate": root / "postchecks/profiles" / label,
                }
            )
    for device, batch in (("cpu", 32), ("cpu", 1024), ("cuda", 1), ("cuda", 1024)):
        label = f"profile-waveform-{device}-b{batch}-t1"
        cases.append(
            {
                "kind": "waveform",
                "device": device,
                "batch": batch,
                "threads": 1,
                "label": label,
                "baseline": root / "wave-profiles-baseline" / f"{device}-b{batch}",
                "candidate": root / "postchecks/profiles" / label,
            }
        )
    return cases


def profile_names(case, side):
    names = ["python.pstats", "python.txt", "operators.txt", "trace.json"]
    names.append("status.json" if case["kind"] == "live" else "timing.json")
    if side == "candidate":
        names.append("files-sha256.json")
        if case["kind"] == "live":
            names.append("worker.log")
    return names


def preflight(root, cases):
    paths = [
        root / "source-environment.json",
        root / "profile-live.py",
        root / "profile-waveform.py",
        root / "postchecks/scripts-sha256.json",
    ]
    for case in cases:
        paths.append(root / "postchecks/commands" / (case["label"] + ".json"))
        for side in ("baseline", "candidate"):
            paths.extend(case[side] / name for name in profile_names(case, side))
    missing = [
        str(path) for path in paths if not path.is_file() or not path.stat().st_size
    ]
    require(
        not missing,
        f"Incomplete profile inputs ({len(missing)} missing/empty files); "
        "no summary written:\n"
        + "\n".join(missing[:12])
        + (f"\n... and {len(missing) - 12} more" if len(missing) > 12 else ""),
    )


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def python_profile(path, source_side):
    stats = pstats.Stats(str(path), stream=io.StringIO())
    require(stats.stats, f"Empty profile: {path}")
    rows, roots = [], set()
    for (filename, line, name), (
        primitive,
        calls,
        self_s,
        cumulative_s,
        _,
    ) in stats.stats.items():
        require(
            all(finite(v) for v in (primitive, calls, self_s, cumulative_s)),
            f"Invalid profile count or duration: {path}: {filename}:{line}",
        )
        if "/pycbc/" in filename:
            source_root, suffix = filename.rsplit("/pycbc/", 1)
            relative = "pycbc/" + suffix
        else:
            source_root, relative = None, filename
        if relative not in TARGETS:
            continue
        require(
            Path(source_root).name == source_side,
            f"Wrong embedded {source_side} source path: {filename}",
        )
        roots.add(source_root)
        rows.append(
            {
                "source_path": filename,
                "relative_path": relative,
                "line": line,
                "function": name,
                "primitive_calls": primitive,
                "calls": calls,
                "self_ms": self_s * 1000,
                "cumulative_ms": cumulative_s * 1000,
            }
        )
    require(len(roots) == 1, f"Missing or mixed source roots: {path}: {roots}")
    focused = [row for row in rows if row["function"] in TARGETS[row["relative_path"]]]
    identifiers = [(row["relative_path"], row["function"]) for row in focused]
    require(
        len(identifiers) == len(set(identifiers)),
        f"Ambiguous focused functions: {path}",
    )
    return {
        "embedded_source_root": roots.pop(),
        "total_calls": stats.total_calls,
        "primitive_calls": stats.prim_calls,
        "total_self_ms": stats.total_tt * 1000,
        "focused_functions": sorted(
            focused, key=lambda row: (row["relative_path"], row["function"])
        ),
        "top_source_functions_by_self": sorted(rows, key=lambda row: -row["self_ms"])[
            :15
        ],
    }


def operator_profile(path, host, device):
    trace = read_json(path)
    require(
        trace.get("host_name") == host,
        f"Trace host differs from source snapshot: {path}",
    )
    require(
        trace.get("record_shapes") == 1, f"Trace is missing recorded shapes: {path}"
    )
    events = trace.get("traceEvents")
    require(isinstance(events, list) and events, f"Missing trace events: {path}")
    categories, ops = Counter(), {}
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        categories[category] += 1
        duration = event.get("dur")
        require(finite(duration), f"Invalid complete-event duration: {path}")
        if category != "cpu_op":
            continue
        name = event["name"]
        op = ops.setdefault(name, {"calls": 0, "inclusive_cpu_us": 0.0, "shapes": {}})
        op["calls"] += 1
        op["inclusive_cpu_us"] += duration
        args = event.get("args", {})
        dims, types = args.get("Input Dims"), args.get("Input type")
        require(
            isinstance(dims, list), f"Missing operator input dimensions: {path}: {name}"
        )
        shape_key = json.dumps([dims, types], sort_keys=True)
        shape = op["shapes"].setdefault(
            shape_key,
            {
                "input_dims": dims,
                "input_types": types,
                "calls": 0,
                "inclusive_cpu_us": 0.0,
            },
        )
        shape["calls"] += 1
        shape["inclusive_cpu_us"] += duration
    require(ops, f"No complete CPU operator events: {path}")
    require(
        device != "cuda" or categories["kernel"] > 0,
        f"CUDA trace lacks kernel events: {path}",
    )
    for op in ops.values():
        op["shapes"] = sorted(
            op["shapes"].values(),
            key=lambda row: (-row["calls"], str(row["input_dims"])),
        )
    metadata = {key: value for key, value in trace.items() if key != "traceEvents"}
    return {
        "metadata": metadata,
        "complete_events_by_category": dict(categories),
        "operators": dict(sorted(ops.items())),
    }


def read_profile(case, side, host):
    folder = case[side]
    files = {name: receipt(folder / name) for name in profile_names(case, side)}
    if side == "candidate":
        expected = read_json(folder / "files-sha256.json")
        required = set(files) - {"files-sha256.json"}
        require(set(expected) == required, f"Profile file manifest differs: {folder}")
        for name, sha in expected.items():
            require(
                files[name]["sha256"] == sha, f"Profile hash mismatch: {folder / name}"
            )
    if case["kind"] == "live":
        metadata = read_json(folder / "status.json")
        require(
            metadata.get("completed") is True and metadata.get("calls", 0) >= 8,
            f"Instrumented worker incomplete: {folder}",
        )
    else:
        metadata = read_json(folder / "timing.json")
        require(
            all(
                metadata.get(key) == case[key] for key in ("device", "batch", "threads")
            ),
            f"Waveform workload mismatch: {folder}",
        )
        metadata = {key: metadata[key] for key in ("device", "batch", "threads")}
    result = {
        "directory": str(folder.resolve()),
        "files": files,
        "workload_metadata": metadata,
        "python": python_profile(folder / "python.pstats", side),
        "torch": operator_profile(folder / "trace.json", host, case["device"]),
    }
    campaign = Path(result["python"]["embedded_source_root"]).parent
    if side == "baseline":
        expected_trace = campaign / folder.parent.name / folder.name
    else:
        expected_trace = campaign / "postchecks/profiles" / folder.name
    require(
        result["torch"]["metadata"].get("traceName")
        == str(expected_trace / "trace.json"),
        f"Trace output path differs from expected campaign/workload: {folder}",
    )
    required = "process_data" if case["kind"] == "live" else "taylorf2_fd_batch"
    observed = [
        row
        for row in result["python"]["focused_functions"]
        if row["function"] == required
    ]
    require(
        len(observed) == 1 and observed[0]["calls"] == 1,
        f"Expected one warmed public-call implementation in profile: {folder}",
    )
    return result


def validate_command(root, case, profile, environment):
    path = root / "postchecks/commands" / (case["label"] + ".json")
    command = read_json(path)
    keys = (
        ("batch", "threads", "route")
        if case["kind"] == "live"
        else ("batch", "threads", "device")
    )
    require(
        all(command.get(key) == case[key] for key in keys),
        f"Candidate command workload mismatch: {path}",
    )
    require(
        command.get("label") == case["label"]
        and command.get("stage") == "profile-" + case["kind"]
        and command.get("returncode") == 0
        and command.get("finished_utc")
        and command.get("source") == "candidate"
        and command.get("host") == environment["host"],
        f"Candidate command incomplete or mismatched: {path}",
    )
    identity = command["source_identity"]
    expected = environment["sources"]["candidate"]
    require(
        identity["root"] == profile["python"]["embedded_source_root"]
        and identity["sha"] == expected["head"]
        and identity["tree"] == expected["tree"]
        and not identity["status"],
        f"Candidate source identity differs: {path}",
    )
    for name, sha in expected["files"].items():
        require(
            identity["tracked_sha256"].get(name) == sha,
            f"Candidate source hash differs: {path}: {name}",
        )
    require(
        identity["native_binary_sha256"] == expected["native_binaries"],
        f"Candidate native binaries differ: {path}",
    )
    argv = command["command"]
    options = {
        "--root": identity["root"],
        "--out": str(Path(profile["torch"]["metadata"]["traceName"]).parent),
        "--batch": str(case["batch"]),
        "--threads": str(case["threads"]),
    }
    options["--route" if case["kind"] == "live" else "--device"] = case.get(
        "route", case["device"]
    )
    for option, expected_value in options.items():
        require(
            argv.count(option) == 1
            and argv.index(option) + 1 < len(argv)
            and argv[argv.index(option) + 1] == expected_value,
            f"Candidate command argument differs: {path}: {option}",
        )
    require(
        str(Path(identity["root"]).parent / ("profile-" + case["kind"] + ".py"))
        in argv,
        f"Candidate command does not invoke expected profile script: {path}",
    )
    return {
        "file": receipt(path),
        "label": command["label"],
        "command": command["command"],
        "cwd": command["cwd"],
        "host": command["host"],
        "environment": command["environment"],
        "started_utc": command["started_utc"],
        "finished_utc": command["finished_utc"],
        "source_root": identity["root"],
        "source_head": identity["sha"],
        "source_tree": identity["tree"],
    }


def pair_functions(baseline, candidate):
    maps = [
        {
            (row["relative_path"], row["function"]): row
            for row in data["python"]["focused_functions"]
        }
        for data in (baseline, candidate)
    ]
    paired = []
    for key in sorted(maps[0].keys() | maps[1].keys()):
        sides = []
        for mapping in maps:
            row = mapping.get(key)
            sides.append(
                dict(row, observed=True)
                if row
                else {
                    "observed": False,
                    "calls": 0,
                    "primitive_calls": 0,
                    "self_ms": 0.0,
                    "cumulative_ms": 0.0,
                    "source_path": None,
                    "line": None,
                }
            )
        paired.append(
            {
                "relative_path": key[0],
                "function": key[1],
                "baseline": sides[0],
                "candidate": sides[1],
                "candidate_minus_baseline": {
                    field: sides[1][field] - sides[0][field]
                    for field in ("calls", "self_ms", "cumulative_ms")
                },
            }
        )
    return paired


def pair_operators(baseline, candidate):
    maps = [data["torch"]["operators"] for data in (baseline, candidate)]
    result = []
    for name in sorted((maps[0].keys() | maps[1].keys()) & FOCUS_OPS):
        sides = [
            dict(mapping[name], observed=True)
            if name in mapping
            else {"observed": False, "calls": 0, "inclusive_cpu_us": 0.0, "shapes": []}
            for mapping in maps
        ]
        result.append(
            {
                "operator": name,
                "baseline": sides[0],
                "candidate": sides[1],
                "candidate_minus_baseline_calls": sides[1]["calls"] - sides[0]["calls"],
            }
        )
    return result


def markdown(report):
    def link(item):
        return Path(item['path']).relative_to(Path(report['artifact_root'])).as_posix()

    lines = [
        "# Paired diagnostic profiles",
        "",
        *[paragraph + "\n" for paragraph in METHOD],
    ]
    lines += [
        "All values below are baseline → candidate. All 8 live and 4 waveform pairs passed validation.",
        "",
    ]
    sources = report["provenance"]["source_environment"]["data"]
    lines += [
        f"Host: `{sources['host']}`. Source identities:",
        "",
        "| Source | Commit | Tree |",
        "|---|---|---|",
    ]
    for side in ("baseline", "candidate"):
        identity = sources["sources"][side]
        lines.append(f"| {side} | `{identity['head']}` | `{identity['tree']}` |")
    lines += [
        "",
        "## Live overlap validation and peak extraction",
        "",
        "| Route | Batch | Disjoint validation cumulative ms | Span-comparison calls | Peak extraction cumulative ms |",
        "|---|---:|---:|---:|---:|",
    ]

    def comparison(case, name, field, operator=False):
        rows = case["operators" if operator else "functions"]
        selected = [
            row for row in rows if row["operator" if operator else "function"] == name
        ]
        if not selected:
            return "0 → 0 (unobserved)"
        row = selected[0]
        values = [row[side][field] for side in ("baseline", "candidate")]
        return " → ".join(f"{v:,}" if field == "calls" else f"{v:.3f}" for v in values)

    for case in report["pairs"]:
        if case["kind"] != "live":
            continue
        cells = [
            case["route"],
            str(case["batch"]),
            comparison(case, "_batch_outputs_are_disjoint", "cumulative_ms"),
            comparison(case, "_spans_overlap", "calls"),
            comparison(case, "_torch_batch_peak_values", "cumulative_ms"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## TaylorF2 phase evaluation and tensor arithmetic",
        "",
        "Counts cover the whole public waveform call, including parameter and reference-frequency operations. "
        "The exact input shapes below distinguish these from frequency-grid operations.",
        "",
        "| Device | Batch | Phase cumulative ms | Phase calls | log calls | addcmul calls | mul calls | add calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report["pairs"]:
        if case["kind"] != "waveform":
            continue
        cells = [
            case["device"],
            str(case["batch"]),
            comparison(case, "_evaluate_phase_polynomial", "cumulative_ms"),
            comparison(case, "_evaluate_phase_polynomial", "calls"),
        ]
        cells += [
            comparison(case, "aten::" + name, "calls", True)
            for name in ("log", "addcmul", "mul", "add")
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "A logarithm or Horner-call reduction is supported only if its observed count falls. "
        "Fewer multiply/add events with unchanged log/addcmul counts instead indicate reduced surrounding arithmetic. "
        "Operation counts alone do not establish numerical equivalence or causal throughput improvement.",
        "",
    ]
    for case in report["pairs"]:
        lines += [
            f"## {case['label']}",
            "",
            "| Function | Calls | Self ms | Cumulative ms |",
            "|---|---:|---:|---:|",
        ]
        for row in case["functions"]:
            name = row["function"]
            lines.append(
                "| `"
                + name
                + "` | "
                + " | ".join(
                    comparison(case, name, field)
                    for field in ("calls", "self_ms", "cumulative_ms")
                )
                + " |"
            )
        if case["kind"] == "waveform":
            lines += ["", "| Operator | Input dimensions | Calls |", "|---|---|---:|"]
            for op in case["operators"]:
                if op["operator"] not in {
                    "aten::log",
                    "aten::addcmul",
                    "aten::mul",
                    "aten::add",
                }:
                    continue
                shapes = defaultdict(lambda: [0, 0])
                for i, side in enumerate(("baseline", "candidate")):
                    for shape in op[side]["shapes"]:
                        shapes[json.dumps(shape["input_dims"], separators=(",", ":"))][
                            i
                        ] += shape["calls"]
                for shape, counts in sorted(shapes.items()):
                    lines.append(
                        f"| `{op['operator']}` | `{shape}` | {counts[0]} → {counts[1]} |"
                    )
        lines += [
            "",
            "Profile input paths and SHA256 hashes:",
            "",
            "| Side | File | SHA256 |",
            "|---|---|---|",
        ]
        for side in ("baseline", "candidate"):
            for name, item in case[side]["files"].items():
                lines.append(
                    f"| {side} | [{name}](<{link(item)}>) | `{item['sha256']}` |"
                )
        lines.append("")
    lines += [
        "## Source and profiling-script hashes",
        "",
        "Exact source paths and line numbers for each observed focused function, candidate command receipts, "
        "all operator shape/type groups, and input hashes are retained in profile-summary.json.",
        "",
        "| Source | Path | SHA256 |",
        "|---|---|---|",
    ]
    for side in ("baseline", "candidate"):
        for path, sha in sources["sources"][side]["files"].items():
            lines.append(f"| {side} | `{path}` | `{sha}` |")
    for name, item in report["provenance"]["profiling_scripts"].items():
        lines.append(
            f"| supplied script | [{name}](<{link(item)}>) | `{item['sha256']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def build_report(root):
    cases = matrix(root)
    preflight(root, cases)
    source_path = root / "source-environment.json"
    environment = read_json(source_path)
    for side in ("baseline", "candidate"):
        require(
            not environment["sources"][side]["status"],
            f"Supplied {side} source snapshot is dirty",
        )
    scripts_path = root / "postchecks/scripts-sha256.json"
    script_hashes = read_json(scripts_path)
    scripts = {
        name: receipt(root / name)
        for name in ("profile-live.py", "profile-waveform.py")
    }
    for name, item in scripts.items():
        require(
            script_hashes.get(name) == item["sha256"],
            f"Candidate profiling script hash mismatch: {name}",
        )
    pairs = []
    for case in cases:
        baseline = read_profile(case, "baseline", environment["host"])
        candidate = read_profile(case, "candidate", environment["host"])
        command = validate_command(root, case, candidate, environment)
        require(
            Path(baseline["python"]["embedded_source_root"]).parent
            == Path(candidate["python"]["embedded_source_root"]).parent,
            f"Pair source roots are from different campaigns: {case['label']}",
        )
        pairs.append(
            {
                key: value
                for key, value in case.items()
                if key not in ("baseline", "candidate")
            }
            | {
                "baseline": baseline,
                "candidate": candidate,
                "candidate_command": command,
                "functions": pair_functions(baseline, candidate),
                "operators": pair_operators(baseline, candidate),
            }
        )
    return {
        "schema_version": 1,
        "artifact_root": str(root),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "instrumented diagnostic attribution; not throughput evidence",
        "methodology": METHOD,
        "complete": True,
        "pair_count": len(pairs),
        "provenance": {
            "source_environment": {"file": receipt(source_path), "data": environment},
            "profiling_scripts": scripts,
            "candidate_script_manifest": receipt(scripts_path),
            "summarizer": receipt(Path(__file__)),
        },
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, help="Default: --root")
    args = parser.parse_args()
    root = args.root.resolve()
    report = build_report(root)
    # Serialize both before writing either: invalid inputs never produce a partial report.
    json_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    markdown_text = markdown(report)
    output = (args.output_dir or root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("profile-summary.json", json_text),
        ("profile-summary.md", markdown_text),
    ):
        path = output / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value)
        temporary.replace(path)
        print(path)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, EOFError) as exc:
        print(f"Profile summary refused: {exc}", file=sys.stderr)
        raise SystemExit(1)
