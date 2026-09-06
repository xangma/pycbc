#!/usr/bin/env python3
"""Validate the completed optional CPU peak follow-up and export JSON/Markdown.

Usage: python build-cpu-followup-report.py --input postchecks/cpu-benchmark
The sibling build-report.py supplies reviewed numerical validators, read-only.
No checkout, benchmark, remote command, or plotting code is executed.
"""

import argparse
import copy
import importlib.util
import json
import re
import shutil
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPER = HERE / "build-report.py"
spec = importlib.util.spec_from_file_location("paired_campaign_report", HELPER)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
require = shared.require

BATCHES = (32, 1024)
THREADS = (1, 4)
REPLICATES = (1, 2, 3)
EXPECTED = {
    "cpu-baseline": "bd53914be6d2e4324cc867d52b3842b77cc6729a",
    "cpu-candidate": "1514327669fc7be125523b991c847868c3a2a17e",
}
CASES = (
    ("branch_standard", "cpu-baseline"),
    ("torch_cpu_native", "cpu-baseline"),
    ("torch_cpu_native", "cpu-candidate"),
)
FLAG = "PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK"
SCRIPTS = (
    "profile-live.py",
    "profile-waveform.py",
    "setup-dependent.py",
    "run-postchecks.py",
)


def tag(key):
    return "cpu-" + shared.label(*key)


def hash_mapping(value, description):
    require(isinstance(value, dict) and value, f"{description}: missing hashes")
    for name, digest in value.items():
        require(
            isinstance(name, str)
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest),
            f"{description}: malformed hash",
        )


def checked_parity(document, before, after, control, batch):
    """Adapt only route identifiers to the shared numerical validator."""
    require(
        set(document) == {f"batch_{batch}", "all_passed_globally"},
        "unexpected parity document fields",
    )
    cell = document[f"batch_{batch}"]
    routes = cell["routes_evaluated"]
    require(
        len(routes) == 3 and set(routes) == {"branch_standard", *EXPECTED},
        "incorrect follow-up parity route coverage",
    )
    require(
        set(cell["comparisons"])
        == {f"{source}_vs_branch_standard" for source in EXPECTED},
        "incorrect follow-up parity comparison coverage",
    )
    view = copy.deepcopy(document)
    translated = view[f"batch_{batch}"]
    translated["routes_evaluated"] = [
        "branch_standard",
        "torch_cpu_native",
        "torch_cpu_native_fixed",
    ]
    translated["comparisons"] = {
        f"{route}_vs_branch_standard": cell["comparisons"][
            f"{source}_vs_branch_standard"
        ]
        for source, route in (
            ("cpu-baseline", "torch_cpu_native"),
            ("cpu-candidate", "torch_cpu_native_fixed"),
        )
    }
    shared.checked_live_parity(view, before, after, control, "torch_cpu_native", batch)


def validate(input_dir):
    manifest = {}

    def read(path):
        manifest[str(path)] = shared.digest(path)
        return shared.load(path)

    parent = input_dir.parent
    status = read(parent / "postcheck-status.json")
    plan = read(parent / "plan.json")
    require(
        status["state"] == "complete" and not status.get("error"),
        "postchecks did not complete successfully",
    )
    require(
        "cpu-benchmark" not in status["skipped_stages"]
        and status["skipped_stages"] == plan["skipped_stages"],
        "CPU benchmark was skipped or skip metadata differs",
    )
    planned_labels = [step["label"] for step in plan["steps"]]
    require(
        len(set(planned_labels)) == len(planned_labels)
        and status["completed"] == planned_labels,
        "postcheck completion ledger differs from its ordered plan",
    )
    require(plan["affinity"] == "8-11", "unexpected CPU affinity")
    sources = read(parent / "sources-before.json")
    final_sources = read(parent / "sources-after.json")
    require(sources == final_sources, "source or native binaries changed")
    for name, expected in EXPECTED.items():
        source = sources[name]
        shared.identity(source, name)
        require(
            source["sha"] == plan["expected_sources"][name] == expected
            and source["status"] == ""
            and source["root"] == str(Path(plan["root"]) / name),
            f"{name}: unexpected source revision, root or dirty status",
        )
        hash_mapping(source["tracked_sha256"], f"{name}: tracked files")
        hash_mapping(source["native_binary_sha256"], f"{name}: native binaries")
    require(
        sources["cpu-baseline"]["tree"] != sources["cpu-candidate"]["tree"],
        "before and after trees are identical",
    )
    for name in ("tools/benchmark_artifact.py", "tools/bench_production_live_batch.py"):
        require(
            sources["cpu-baseline"]["tracked_sha256"][name]
            == sources["cpu-candidate"]["tracked_sha256"][name],
            f"benchmark harness differs between revisions: {name}",
        )
    scripts = read(parent / "scripts-sha256.json")
    require(
        scripts == read(parent / "scripts-after-sha256.json")
        and set(scripts) == set(SCRIPTS),
        "postcheck scripts changed or script inventory is incomplete",
    )
    hash_mapping(scripts, "postcheck scripts")
    for name, expected in scripts.items():
        path = parent.parent / name
        manifest[str(path)] = shared.digest(path)
        require(manifest[str(path)] == expected, f"local script differs: {name}")

    keys = [
        (route, batch, threads, replicate, source)
        for batch in BATCHES
        for threads in THREADS
        for replicate in REPLICATES
        for route, source in CASES
    ]
    parity_keys = [
        (batch, threads, replicate)
        for batch in BATCHES
        for threads in THREADS
        for replicate in REPLICATES
    ]
    labels = {tag(key) for key in keys}
    steps = {
        step["label"]: step
        for step in plan["steps"]
        if step["stage"] == "cpu-benchmark"
    }
    require(set(steps) == labels, "expected exactly 36 CPU benchmark plan steps")
    parity_labels = [f"parity-b{b}-t{t}-r{r}" for b, t, r in parity_keys]
    shared.exact_files(input_dir, labels | set(parity_labels))
    records, values, pids = {}, {}, set()
    for key in keys:
        route, batch, threads, replicate, name = key
        label = tag(key)
        step = steps[label]
        source = sources[name]
        record = read(input_dir / f"{label}.json")
        receipt = read(parent / "commands" / f"{label}.json")
        require(
            record["comparison_label"] == label
            and all(receipt[field] == value for field, value in step.items())
            and receipt["source_identity"] == source
            and receipt["returncode"] == 0
            and receipt["cwd"] == source["root"]
            and receipt["host"] == status["host"],
            f"{label}: command receipt mismatch",
        )
        require(
            (
                step["route"],
                step["batch"],
                step["threads"],
                step["replicate"],
                step["source"],
            )
            == (route, batch, threads, replicate, name),
            f"{label}: planned workload mismatch",
        )
        require(
            receipt["command"] == record["command"]
            and record["command"][3] == plan["python"],
            f"{label}: benchmark command mismatch",
        )
        pid = record["pid"]
        require(
            type(pid) is int and pid > 0 and pid == receipt["pid"] and pid not in pids,
            f"{label}: repeated or incorrect worker PID",
        )
        pids.add(pid)
        shared.number(receipt["wall_seconds"], label, positive=True)
        require(
            datetime.fromisoformat(receipt["finished_utc"])
            > datetime.fromisoformat(receipt["started_utc"]),
            f"{label}: command timestamps",
        )
        environment = receipt["environment"]
        for variable, expected in {
            "PYTHONPATH": source["root"],
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }.items():
            require(environment[variable] == expected, f"{label}: {variable}")
        if route == "torch_cpu_native":
            flag = record["routing"]["feature_flags"][FLAG]
            require(
                flag["enabled"] is True
                and flag["environment_value"] == "1"
                and flag["applies_to_route"] is True
                and flag["selection"] == "explicit"
                and environment[FLAG] == "1",
                f"{label}: optional CPU peak flag is inactive",
            )
        log = parent / "logs" / f"{label}.log"
        require(
            receipt["log_path"] == str(Path(plan["output"]) / "logs" / log.name),
            f"{label}: log path mismatch",
        )
        manifest[str(log)] = shared.digest(log)
        lines = [
            line.removeprefix("RESULT_JSON=")
            for line in log.read_text().splitlines()
            if line.startswith("RESULT_JSON=")
        ]
        require(len(lines) == 1, f"{label}: expected one worker result in log")
        logged = json.loads(lines[0], object_pairs_hook=shared.no_duplicates)
        logged.update(
            comparison_source=source,
            comparison_label=label,
            comparison_replicate=replicate,
            command=step["command"],
        )
        require(logged == record, f"{label}: exported result differs from worker log")
        # The runner adds a cpu- label prefix; only this identifier is adapted.
        view = {**record, "comparison_label": shared.label(*key)}
        values[key] = shared.checked_live(view, key, source)
        records[key] = record

    all_records = list(records.values())
    shared.consistent(
        all_records,
        ("python", "numpy_version", "measurement", "dtypes"),
        "CPU worker runtime/workload",
    )
    shared.consistent(
        [r for r in all_records if r["route"] == "torch_cpu_native"],
        ("torch_version",),
        "CPU Torch runtime",
    )
    for route, source in CASES:
        for threads in THREADS:
            shared.consistent(
                [
                    r
                    for k, r in records.items()
                    if k[0] == route and k[2] == threads and k[-1] == source
                ],
                ("routing",),
                f"{source}/{route}/{threads}: routing",
            )
    parity_maxima = {}
    for batch, threads, replicate in parity_keys:
        document = read(input_dir / f"parity-b{batch}-t{threads}-r{replicate}.json")
        before, after = (
            records[("torch_cpu_native", batch, threads, replicate, name)]
            for name in EXPECTED
        )
        control = records[
            ("branch_standard", batch, threads, replicate, "cpu-baseline")
        ]
        checked_parity(document, before, after, control, batch)
        shared.consistent(
            [before, after, control], ("injection_metadata",), "paired inputs"
        )
        for comparison in document[f"batch_{batch}"]["comparisons"].values():
            for metric in (
                "max_snr_diff",
                "max_phase_diff",
                "max_sigmasq_relative_diff",
                "relative_output_l2_diff",
            ):
                parity_maxima[metric] = max(
                    parity_maxima.get(metric, 0), comparison[metric]
                )

    rows, controls = [], []
    for threads in THREADS:
        for batch in BATCHES:
            before, after = (
                [
                    values[("torch_cpu_native", batch, threads, rep, name)]
                    for rep in REPLICATES
                ]
                for name in EXPECTED
            )
            ratios = [a / b for a, b in zip(after, before, strict=True)]
            rows.append(
                dict(
                    threads=threads,
                    batch=batch,
                    before=shared.summary(before),
                    after=shared.summary(after),
                    paired_ratio=shared.summary(ratios),
                    ratio_of_medians=statistics.median(after)
                    / statistics.median(before),
                )
            )
            controls.append(
                dict(
                    threads=threads,
                    batch=batch,
                    throughput=shared.summary(
                        [
                            values[
                                ("branch_standard", batch, threads, rep, "cpu-baseline")
                            ]
                            for rep in REPLICATES
                        ]
                    ),
                )
            )
    for row in rows:
        row["search_capacity"] = shared.search_capacity(
            "torch_cpu_native",
            row["threads"],
            {state: row[state] for state in ("before", "after")},
        )
    for row in controls:
        row["search_capacity"] = shared.search_capacity(
            "branch_standard", row["threads"], {"throughput": row["throughput"]}
        )
    manifest[str(HELPER)] = shared.digest(HELPER)
    manifest[str(Path(__file__).resolve())] = shared.digest(Path(__file__).resolve())
    return dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        title="Optional CPU native peak follow-up",
        unit="templates/second",
        search_capacity_units=dict(shared.CAPACITY_UNITS),
        search_capacity_basis=shared.capacity_basis(
            {name: sources[name] for name in EXPECTED}
        ),
        source_identities={
            name: shared.identity(value, name)
            for name, value in sources.items()
            if name in EXPECTED
        },
        host=status["host"],
        python_executable=plan["python"],
        runtime={name: all_records[0][name] for name in ("python", "numpy_version")},
        torch_version=next(
            r["torch_version"] for r in all_records if r["route"] == "torch_cpu_native"
        ),
        counts=dict(
            worker_processes=36,
            standard_controls=12,
            native_workers=24,
            parity_documents=12,
            parity_comparisons=24,
        ),
        configuration=dict(
            route="torch_cpu_native",
            required_environment={FLAG: "1"},
            batches=list(BATCHES),
            threads=list(THREADS),
            replicates=list(REPLICATES),
            cpu_affinity="8-11",
        ),
        notes={
            "aggregation": shared.NOTES["aggregation"],
            "workload": shared.NOTES["live"],
            "search_capacity": shared.NOTES["search_capacity"],
            "parity": shared.NOTES["live_parity"],
            "timing": "Sequential fresh workers, with three timed public API iterations "
            "after one cold iteration and one warmup. Each iteration filters all "
            "templates against three strain blocks; throughput is 3*batch/seconds. "
            "No profiling instrumentation is included in these worker commands.",
            "scope": "Both revisions explicitly enable PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK=1 "
            "alongside native correlation and FFTW batch routing. Enabled routing permits "
            "fallback when admission requires it; these records do not count native "
            "dispatches. The branch_standard control belongs to cpu-baseline. "
            "This report summarizes only the optional CPU follow-up, not the other "
            "postcheck tests or profiles.",
        },
        numerical_parity=dict(all_passed=True, recomputed=True, maxima=parity_maxima),
        comparisons=rows,
        standard_controls=controls,
    ), manifest


def markdown(report):
    lines = [
        "# Optional CPU native peak follow-up",
        "",
        f"Host: `{report['host']}`. Both revisions explicitly set `{FLAG}=1`.",
        "",
    ]
    for name, source in report["source_identities"].items():
        lines.append(f"- {name}: `{source['sha']}` (tree `{source['tree']}`).")
    lines.extend(
        [
            "",
            "Each capacity cell is median [minimum, maximum] in templates/core at real time over "
            "three separate worker medians. The paired ratio uses after/before within "
            "each replicate, then reports its median and full range.",
            "",
            "| Configured cores | Batch | Before templates/core at real time | After templates/core at real time | Paired ratio |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["comparisons"]:
        lines.append(
            f"| {row['threads']} | {row['batch']} | "
            f"{shared.format_stat(row['search_capacity']['before'])} | {shared.format_stat(row['search_capacity']['after'])} | "
            f"{shared.format_stat(row['paired_ratio'], ratio=True)} |"
        )
    lines.extend(
        [
            "",
            "Standard CPU controls (cpu-baseline):",
            "",
            "| Configured cores | Batch | Templates/core at real time |",
            "| ---: | ---: | ---: |",
        ]
    )
    for row in report["standard_controls"]:
        lines.append(
            f"| {row['threads']} | {row['batch']} | "
            f"{shared.format_stat(row['search_capacity']['throughput'])} |"
        )
    lines.extend(
        [
            "",
            "Validated 36 distinct successful worker processes and 12 parity documents "
            "covering 24 native/control comparisons. All stored parity flags passed, "
            "and trigger fields and aggregate norm differences were independently "
            "recomputed from the exported worker results.",
            "",
            "| Parity metric | Maximum across all comparisons |",
            "| --- | ---: |",
        ]
    )
    for metric, value in report["numerical_parity"]["maxima"].items():
        lines.append(f"| `{metric}` | {value:.9g} |")
    for value in report["notes"].values():
        lines.extend(["", value])
    lines.extend(
        [
            "",
            "The input manifest records SHA-256 hashes for the source snapshots, "
            "completion ledger, plan, worker JSON, command receipts, raw logs, parity "
            "documents and local reporting/runner scripts. Before/after source snapshots "
            "and native binary hashes are unchanged within each revision. JSON retains "
            "every worker median, paired ratio and the separate ratio of aggregate medians.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=HERE / "postchecks" / "cpu-benchmark"
    )
    parser.add_argument("--out", type=Path, default=HERE / "cpu-followup-report")
    args = parser.parse_args()
    input_dir, output = args.input.resolve(), args.out.resolve()
    temporary = None
    try:
        require(
            not output.exists(), f"output already exists; choose a new --out: {output}"
        )
        report, manifest = validate(input_dir)
        require(
            output.parent.is_dir(), f"output parent does not exist: {output.parent}"
        )
        temporary = Path(
            tempfile.mkdtemp(prefix=".cpu-followup-report-", dir=output.parent)
        )
        manifest_path = temporary / "input-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        report["input_manifest"] = dict(
            path="input-manifest.json",
            files=len(manifest),
            sha256=shared.digest(manifest_path),
        )
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        (temporary / "report.md").write_text(markdown(report))
        for path, expected in manifest.items():
            require(
                shared.digest(Path(path)) == expected,
                f"input changed during report: {path}",
            )
        temporary.rename(output)
        temporary = None
        print(json.dumps(dict(status="ok", out=str(output), counts=report["counts"])))
    except (shared.InvalidCampaign, OSError, KeyError, TypeError, ValueError) as error:
        print(f"CPU follow-up report refused: {error}", file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
