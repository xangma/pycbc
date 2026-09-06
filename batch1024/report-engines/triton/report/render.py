#!/usr/bin/env python3
"""Qualify and render the complete 72-worker TaylorF2 Triton experiment."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

BATCHES = (1, 8, 32, 128, 512, 1024)
GRIDS = ((4097, 0.25), (32769, 0.03125))
ROUTES = ("cuda-off", "cuda-on")
LABELS = {"cuda-off": "Torch CUDA · gate off", "cuda-on": "Triton CUDA · gate on"}
COLORS = {"cuda-off": "#0072B2", "cuda-on": "#D55E00"}
SOURCE_FILES = (
    "pycbc/waveform/taylorf2_torch.py",
    "pycbc/waveform/waveform.py",
    "pycbc/waveform/torch_waveform_registry.py",
    "pycbc/waveform/taylorf2_triton.py",
)
PARITY = {
    "parity_actual_vs_gate_off": (2e-10, 2e-10, False),
    "parity_gate_off_vs_native_scalar": (2e-10, 2e-10, False),
    "parity_native_scalar_vs_lal": (None, 1e-11, True),
    "parity_actual_vs_lal": (None, None, False),
}
LIMITATIONS = [
    "Complete public get_fd_waveform_batch call, including validation, host-list conversion, coefficient work, allocation and output wrapping; host output copies are excluded.",
    "Complex128 hplus and hcross, one host thread, no autograd; eight distinct deterministic BNS mass pairs repeat in larger batches.",
    "All bins and both polarizations of every row are checked. Exact repeated rows reuse their complete scalar reference; no rows or frequency bins are sampled.",
    "Warm timing uses five synchronized groups of at least 50 ms per worker. Three process replicates support an observed range, not a confidence interval or tail-latency claim.",
    "Cold time is the first complete public call with a fresh per-worker Triton cache. It includes any compilation plus lazy initialization and is not isolated compiler time; the CUDA driver cache is not cleared.",
    "The measured candidate's gate-off route is the same-revision control. These results do not compare different commits or establish a whole-search or inference speedup.",
    "Performance cells qualify only after exact-source, dispatch, timing and waveform checks. Failed and incomplete cells remain listed, without performance comparisons.",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    return (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-15)
    )


def positive(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def expected_parameters(batch, delta_f):
    return [
        dict(
            mass1=1.4 + 0.05 * (i % 8),
            mass2=1.3 - 0.02 * (i % 8),
            spin1z=0.02,
            spin2z=-0.01,
            distance=100.0,
            inclination=0.4,
            coa_phase=0.2,
            delta_f=delta_f,
            f_lower=20.0,
            f_final=1024.0,
            f_ref=30.0,
        )
        for i in range(batch)
    ]


def key_for(route, batch, bins):
    return f"{route}-b{batch}-n{bins}"


def check_worker(value, batch, bins, delta_f, route, replicate, args, worker_hash):
    problems = []

    def require(condition, message):
        if not condition:
            problems.append(message)

    require(value.get("schema") == 1, "unexpected worker schema")
    require(
        value.get("status") == "ok",
        "worker status: "
        + str(value.get("status"))
        + "; "
        + str(value.get("reason", "")),
    )
    if value.get("status") != "ok":
        return problems
    for name, expected in dict(
        batch=batch,
        delta_f=delta_f,
        route=route,
        replicate=replicate,
        threads=1,
        expected_sha=args.expected_sha,
        harness_sha256=worker_hash,
    ).items():
        require(value.get(name) == expected, f"wrong {name}")
    require(
        bool(value.get("synthetic", False)) == args.synthetic_test,
        "synthetic/real data designation mismatch",
    )
    source = value.get("source", {})
    require(
        source.get("sha") == args.expected_sha
        and source.get("clean") is True
        and source.get("status_porcelain") == source.get("status_after") == "",
        "wrong revision or unclean source",
    )
    require(
        set(source.get("sha256", {})) == set(SOURCE_FILES)
        and all(
            isinstance(v, str) and len(v) == 64
            for v in source.get("sha256", {}).values()
        ),
        "missing source hashes",
    )
    runtime = value.get("runtime", {})
    require(
        str(runtime.get("started_utc", ""))[:10] == args.run_date,
        "worker date differs from requested date",
    )
    require(
        runtime.get("torch_threads") == 1 and runtime.get("torch_interop_threads") == 1,
        "wrong Torch thread configuration",
    )
    require(isinstance(runtime.get("pid"), int), "worker PID missing")
    origins = runtime.get("module_origins", {})
    for module in ("pycbc", "taylorf2_torch", "triton_kernel_module"):
        try:
            Path(origins[module]).relative_to(Path(source["root"]))
        except (KeyError, ValueError, TypeError):
            require(False, f"{module} not proven to originate in the measured checkout")
    expected_rows = expected_parameters(batch, delta_f)
    parameters_hash = hashlib.sha256(
        json.dumps(expected_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        value.get("parameters") == expected_rows
        and value.get("parameters_sha256") == parameters_hash,
        "frozen input rows or their hash differ",
    )
    workload = value.get("workload", {})
    require(
        workload.get("bins") == bins
        and workload.get("approximant") == "TaylorF2"
        and workload.get("precision") == "complex128"
        and workload.get("polarizations") == ["hplus", "hcross"]
        and workload.get("autograd") is False,
        "unexpected workload or precision",
    )
    environment = value.get("environment", {})
    require(
        environment.get("PYCBC_TAYLORF2_TRITON") == ("1" if route == "cuda-on" else "0")
        and environment.get("PYCBC_TAYLORF2_NATIVE") == "1",
        "wrong route flags",
    )
    cache = value.get("cache", {})
    require(
        cache.get("initial_files") == []
        and cache.get("triton_cache_dir") == environment.get("TRITON_CACHE_DIR"),
        "cold cache identity or empty initial state missing",
    )
    if route == "cuda-on":
        require(
            bool(cache.get("files_after_cold")),
            "no compiled cache artifacts after Triton cold call",
        )
    timing = value.get("timing", {})
    totals = timing.get("sample_total_seconds", [])
    counts = timing.get("inner_calls_per_sample", [])
    per_call = timing.get("sample_seconds_per_call", [])
    require(
        timing.get("warmup_calls") == 2
        and timing.get("minimum_sample_seconds") == 0.05
        and timing.get("minimum_duration_passed") is True,
        "wrong warmup or duration contract",
    )
    if (
        len(totals) == len(counts) == len(per_call) == 5
        and all(positive(x) and x >= 0.05 for x in totals)
        and all(
            isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in counts
        )
    ):
        computed = [total / count for total, count in zip(totals, counts)]
        median = statistics.median(computed)
        require(
            all(close(x, y) for x, y in zip(computed, per_call)),
            "raw timing division mismatch",
        )
        require(
            close(timing.get("median_seconds_per_call"), median)
            and close(timing.get("waveforms_per_second"), batch / median)
            and close(timing.get("median_seconds_per_waveform"), median / batch),
            "stored timing estimates disagree with raw samples",
        )
    else:
        require(False, "five valid synchronized groups of at least 50 ms are required")
    require(positive(timing.get("cold_call_seconds")), "invalid cold first-call time")
    require(
        timing.get("timed_surface")
        == "complete public call; no monkey patches or profiler active"
        and timing.get("synchronization") == "CUDA synchronize before/after each block",
        "unexpected timed surface or synchronization",
    )
    dispatch = value.get("dispatch", {})
    enabled = int(route == "cuda-on")
    counts_expected = dict(
        native_batch=1,
        native_scalar=0,
        lal_fd=0,
        launcher=enabled,
        kernel_run_attempts=enabled,
        kernel_run_successes=enabled,
    )
    require(
        dispatch.get("counts") == counts_expected
        and dispatch.get("expected_counts") == counts_expected
        and dispatch.get("passed") is True
        and dispatch.get("triton_actual") is bool(enabled),
        "actual kernel dispatch does not match requested route",
    )
    distinct = min(batch, 8)
    for name, expected in (
        ("native_scalar", dict(native_scalar=distinct, lal_fd=0)),
        ("lal", dict(native_scalar=0, lal_fd=distinct)),
    ):
        reference_dispatch = value.get("reference_dispatch", {}).get(name, {})
        require(
            reference_dispatch.get("counts") == expected
            and reference_dispatch.get("expected_counts") == expected
            and reference_dispatch.get("passed") is True,
            f"{name} reference dispatch is unqualified",
        )
    metadata = dict(
        delta_f=delta_f,
        epoch=-1 / delta_f,
        first_bins=[math.ceil(20 / delta_f)] * batch,
        end_bins=[bins] * batch,
    )
    output = value.get("output", {})
    require(
        output.get("metadata") == metadata
        and output.get("expected_metadata") == metadata
        and output.get("dtypes") == ["complex128", "complex128"]
        and output.get("shapes") == [[batch, bins], [batch, bins]]
        and output.get("devices") == ["cuda:0", "cuda:0"],
        "output metadata, dtype, shape or device differs",
    )
    reference_policy = value.get("reference_policy", {})
    require(
        reference_policy.get("distinct_rows") == distinct
        and reference_policy.get("row_map") == [i % distinct for i in range(batch)],
        "scalar reference row mapping differs",
    )
    for name, (point_limit, l2_limit, scalar) in PARITY.items():
        parity = value.get(name, {})
        require(
            parity.get("passed") is True
            and parity.get("metadata_equal") is True
            and parity.get("pointwise_tolerance") == point_limit
            and parity.get("relative_l2_tolerance") == l2_limit,
            f"{name}: missing or failed parity contract",
        )
        metrics = parity.get("metrics", [])
        count = distinct if scalar else batch
        require(
            len(metrics) == count * 2
            and {(m.get("polarization"), m.get("row")) for m in metrics}
            == {(p, i) for p in (0, 1) for i in range(count)},
            f"{name}: complete row/polarization metrics missing",
        )
        for metric in metrics:
            l2_value, point_value = (
                metric.get("relative_l2"),
                metric.get("max_pointwise_relative"),
            )
            numerical = all(
                isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
                for v in (l2_value, point_value)
            )
            passed = (
                numerical
                and (l2_limit is None or l2_value <= l2_limit)
                and (point_limit is None or point_value <= point_limit)
            )
            if not (
                passed
                and metric.get("passed") is True
                and metric.get("finite") is True
                and metric.get("exact_zero_support") is True
            ):
                require(
                    False,
                    f"{name}: row {metric.get('row')} polarization {metric.get('polarization')} fails numerical/support gate",
                )
                break
    return problems


def load_campaign(args):
    manifest_path, summary_path = (
        args.input_dir / "manifest.json",
        args.input_dir / "summary.json",
    )
    manifest, stored_summary = (
        json.loads(manifest_path.read_text()),
        json.loads(summary_path.read_text()),
    )
    hashes = {
        str(manifest_path): digest(manifest_path),
        str(summary_path): digest(summary_path),
    }
    worker_hash = digest(args.harness_dir / "worker.py")
    run_hash = digest(args.harness_dir / "run.py")
    global_problems = []
    conditions = dict(
        batches=list(BATCHES),
        delta_f=[g[1] for g in GRIDS],
        routes=list(ROUTES),
        threads=1,
        replicates=3,
        samples=5,
        minimum_sample_ms=50,
    )
    for condition, message in (
        (
            manifest.get("schema") == stored_summary.get("schema") == 1,
            "unknown campaign schema",
        ),
        (
            manifest.get("expected_sha")
            == stored_summary.get("expected_sha")
            == args.expected_sha,
            "campaign revision differs",
        ),
        (
            manifest.get("conditions") == conditions,
            "campaign is not the requested full 72-worker matrix",
        ),
        (
            manifest.get("worker_sha256") == worker_hash
            and manifest.get("orchestrator_sha256") == run_hash,
            "campaign harness hashes differ from supplied harness",
        ),
        (
            str(manifest.get("created_utc", ""))[:10]
            == str(manifest.get("finished_utc", ""))[:10]
            == args.run_date,
            "campaign dates differ or campaign unfinished",
        ),
        (
            bool(manifest.get("synthetic", False)) == args.synthetic_test,
            "synthetic/real campaign designation mismatch",
        ),
    ):
        if not condition:
            global_problems.append(message)
    jobs = manifest.get("jobs", [])
    expected_jobs = {
        f"{key_for(route, batch, bins)}-r{replicate}.json"
        for bins, _ in GRIDS
        for batch in BATCHES
        for route in ROUTES
        for replicate in (1, 2, 3)
    }
    job_map = {}
    for job in jobs:
        job_map.setdefault(Path(job.get("output", "")).name, []).append(job)
    if (
        len(jobs) != 72
        or set(job_map) != expected_jobs
        or any(len(v) != 1 for v in job_map.values())
    ):
        global_problems.append(
            "manifest must contain each of the 72 worker jobs exactly once"
        )
    groups_expected = {
        key_for(route, batch, bins)
        for bins, _ in GRIDS
        for batch in BATCHES
        for route in ROUTES
    }
    if set(stored_summary.get("groups", {})) != groups_expected:
        global_problems.append(
            "summary does not contain the expected 24 route/workload groups"
        )
    rows, all_workers, missing = [], [], []
    for bins, delta_f in GRIDS:
        for batch in BATCHES:
            for route in ROUTES:
                key = key_for(route, batch, bins)
                row = dict(
                    key=key,
                    bins=bins,
                    delta_f=delta_f,
                    batch=batch,
                    route=route,
                    label=LABELS[route],
                    qualified=False,
                    worker_records=[],
                    problems=[],
                )
                values, worker_results = [], []
                for replicate in (1, 2, 3):
                    filename = f"{key}-r{replicate}.json"
                    path = args.input_dir / filename
                    problems = []
                    if not path.is_file():
                        missing.append(filename)
                        problems.append("raw worker artifact missing")
                        value = {}
                    else:
                        hashes[str(path)] = digest(path)
                        try:
                            value = json.loads(path.read_text())
                            problems.extend(
                                check_worker(
                                    value,
                                    batch,
                                    bins,
                                    delta_f,
                                    route,
                                    replicate,
                                    args,
                                    worker_hash,
                                )
                            )
                        except (ValueError, TypeError, KeyError) as exc:
                            value = {}
                            problems.append("invalid worker record: " + str(exc))
                    jobs_for_file = job_map.get(filename, [])
                    if (
                        len(jobs_for_file) != 1
                        or jobs_for_file[0].get("returncode") != 0
                        or jobs_for_file[0].get("status") != "ok"
                    ):
                        problems.append(
                            "manifest job missing, duplicated or unsuccessful"
                        )
                    worker_result = dict(
                        replicate=replicate,
                        path=str(path),
                        status=value.get("status", "missing"),
                        reason=value.get("reason"),
                        qualified=not problems,
                        problems=problems,
                    )
                    row["worker_records"].append(worker_result)
                    worker_results.append(worker_result)
                    values.append(value)
                    if not problems:
                        all_workers.append(value)
                    row["problems"].extend(f"worker {replicate}: {p}" for p in problems)
                row["qualified_workers"] = sum(r["qualified"] for r in worker_results)
                row["qualified"] = row["qualified_workers"] == 3
                if row["qualified"]:
                    medians = [
                        statistics.median(
                            total / n
                            for total, n in zip(
                                v["timing"]["sample_total_seconds"],
                                v["timing"]["inner_calls_per_sample"],
                            )
                        )
                        for v in values
                    ]
                    throughputs = [batch / latency for latency in medians]
                    cold = [v["timing"]["cold_call_seconds"] for v in values]
                    stored = stored_summary.get("groups", {}).get(key, {})
                    consistency = (
                        stored.get("eligible") is True
                        and stored.get("valid_replicates") == 3
                        and stored.get("completed_replicates") == 3
                        and stored.get("statuses") == ["ok"] * 3
                        and stored.get("parameter_sha256")
                        == values[0]["parameters_sha256"]
                        and len(stored.get("replicate_medians_seconds", [])) == 3
                        and all(
                            close(x, y)
                            for x, y in zip(
                                medians, stored.get("replicate_medians_seconds", [])
                            )
                        )
                        and close(
                            stored.get("median_seconds_per_call"),
                            statistics.median(medians),
                        )
                        and close(
                            stored.get("waveforms_per_second"),
                            statistics.median(throughputs),
                        )
                    )
                    if not consistency:
                        row.update(qualified=False)
                        row["problems"].append(
                            "stored campaign aggregate disagrees with independently derived worker estimates"
                        )
                    else:
                        row.update(
                            worker_medians_seconds=medians,
                            worker_throughputs=throughputs,
                            throughput_median=statistics.median(throughputs),
                            throughput_min=min(throughputs),
                            throughput_max=max(throughputs),
                            warm_call_seconds_median=statistics.median(medians),
                            cold_call_seconds=cold,
                            cold_median_seconds=statistics.median(cold),
                            cold_min_seconds=min(cold),
                            cold_max_seconds=max(cold),
                        )
                rows.append(row)
    if missing:
        global_problems.append(
            f"{len(missing)} of the 72 raw worker artifacts are missing; plots suppressed"
        )
    if all_workers:
        signatures = {
            json.dumps(
                dict(
                    source=v["source"]["sha256"],
                    tree=v["source"]["tree"],
                    hostname=v["runtime"]["hostname"],
                    gpu=v["runtime"]["cuda_device"],
                    packages={
                        key: v["runtime"].get(key)
                        for key in (
                            "torch",
                            "numpy",
                            "lalsuite",
                            "scipy",
                            "triton",
                            "torch_cuda_build",
                        )
                    },
                ),
                sort_keys=True,
            )
            for v in all_workers
        }
        if len(signatures) != 1:
            global_problems.append(
                "qualified workers do not share source contents, runtime versions and GPU/host identity"
            )
        cache_dirs = [v["cache"]["triton_cache_dir"] for v in all_workers]
        if len(set(cache_dirs)) != len(cache_dirs):
            global_problems.append(
                "Triton cache directories were reused between worker processes"
            )
    if global_problems:
        for row in rows:
            row["qualified"] = False
            row["problems"].extend(global_problems)
            for field in tuple(row):
                if field.startswith(
                    (
                        "throughput_",
                        "cold_",
                        "warm_",
                        "worker_medians",
                        "worker_throughputs",
                    )
                ):
                    row.pop(field)
    row_map = {row["key"]: row for row in rows}
    for row in rows:
        baseline = row_map[key_for("cuda-off", row["batch"], row["bins"])]
        if row["qualified"] and baseline["qualified"]:
            row["speedup_vs_gate_off"] = (
                row["throughput_median"] / baseline["throughput_median"]
            )
            stored_ratio = stored_summary["groups"][row["key"]].get(
                "speedup_vs_cuda_gate_off"
            )
            if not close(row["speedup_vs_gate_off"], stored_ratio):
                row["stored_ratio_warning"] = (
                    "stored speedup differs; report ratio is independently recomputed"
                )
    first_runtime = all_workers[0]["runtime"] if all_workers else {}
    return dict(
        schema=1,
        synthetic=args.synthetic_test,
        status="qualified"
        if all(r["qualified"] for r in rows)
        else "incomplete qualification",
        expected_sha=args.expected_sha,
        run_date=args.run_date,
        renderer_sha256=digest(Path(__file__)),
        input_hashes=hashes,
        supplied_harness_hashes=dict(worker=worker_hash, run=run_hash),
        global_problems=global_problems,
        raw_worker_files=len(hashes) - 2,
        expected_worker_count=72,
        expected_group_count=24,
        qualified_group_count=sum(r["qualified"] for r in rows),
        runtime=dict(
            hostname=first_runtime.get("hostname"),
            cuda_device=first_runtime.get("cuda_device"),
            torch=first_runtime.get("torch"),
            triton=first_runtime.get("triton"),
        ),
        estimand="median of three worker medians; throughput derived separately for each worker before aggregation",
        ranges="min/max of three worker estimates; observed ranges, not confidence intervals",
        limitations=LIMITATIONS,
        rows=rows,
    )


def format_number(value):
    if value >= 1000:
        return f"{value:,.0f}"
    return f"{value:.3g}"


def markdown(report):
    prefix = "SYNTHETIC TEST — " if report["synthetic"] else ""
    lines = [
        f"# {prefix}TaylorF2 Triton: public batch calls",
        "",
        f"Revision `{report['expected_sha']}` · {report['run_date']} · "
        f"{report['qualified_group_count']}/24 route/workload groups qualified from {report['raw_worker_files']}/72 worker files.",
        "",
        "CUDA gate off and gate on use the same candidate revision. Warm throughput is the median of three worker estimates. "
        "Parentheses show their observed minimum–maximum, not a confidence interval.",
        "",
    ]
    if report["synthetic"]:
        lines.extend(
            [
                "**Synthetic validation data only. These are not measured benchmark results.**",
                "",
            ]
        )
    device = report["runtime"].get("cuda_device") or {}
    if device:
        lines.extend(
            [
                f"Hardware: {device.get('name')} on `{report['runtime'].get('hostname')}`; "
                f"Torch {report['runtime'].get('torch')}, Triton {report['runtime'].get('triton')}.",
                "",
            ]
        )
    for problem in report["global_problems"]:
        lines.append(f"- {problem}")
    if report["global_problems"]:
        lines.append("")
    for bins, delta_f in GRIDS:
        lines.extend(
            [
                f"## {bins:,} bins · Δf = {delta_f:g} Hz",
                "",
                "| Batch | Route | Qualified workers | Waveforms/s (range) | On/off ratio | Cold first call, ms (range) |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["rows"]:
            if row["bins"] != bins:
                continue
            qualified = f"{row['qualified_workers']}/3"
            throughput = ratio = cold = "—"
            if row["qualified"]:
                throughput = f"{format_number(row['throughput_median'])} ({format_number(row['throughput_min'])}–{format_number(row['throughput_max'])})"
                cold = f"{format_number(row['cold_median_seconds'] * 1000)} ({format_number(row['cold_min_seconds'] * 1000)}–{format_number(row['cold_max_seconds'] * 1000)})"
                ratio = (
                    f"{row['speedup_vs_gate_off']:.3f}×"
                    if "speedup_vs_gate_off" in row
                    else "—"
                )
            else:
                qualified += " · unqualified"
            label = (
                "Torch CUDA, off" if row["route"] == "cuda-off" else "Triton CUDA, on"
            )
            lines.append(
                f"| {row['batch']} | {label} | {qualified} | {throughput} | {ratio} | {cold} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Qualification",
            "",
            "The renderer verifies all 72 raw worker records and independently recomputes the five-group worker medians and campaign estimates. "
            "It checks candidate and harness hashes, clean source, module origins, fixed inputs, runtime identity, output dtype/support metadata, "
            "sample duration, actual launcher and kernel launches, and both links of the scalar/LAL reference chain.",
            "",
            "Actual-versus-gate-off and gate-off-versus-native-scalar require pointwise relative and relative-L2 errors ≤2e-10. "
            "Native scalar versus LAL requires relative L2 ≤1e-11. Direct actual-versus-LAL errors are additionally retained in the raw records. "
            "All comparisons require finite values, exact zero support, and exact metadata.",
            "",
        ]
    )
    failed = [row for row in report["rows"] if not row["qualified"]]
    if failed:
        lines.extend(["Unqualified cells:", ""])
        for row in failed:
            reasons = "; ".join(dict.fromkeys(row["problems"]))
            lines.append(f"- `{row['key']}`: {reasons}")
        lines.append("")
    lines.extend(["## Measurement scope", ""])
    lines.extend(f"- {text}" for text in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def plots(report, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FAFBFC",
            "axes.facecolor": "#FAFBFC",
            "text.color": "#182533",
            "axes.labelcolor": "#34495B",
            "xtick.color": "#4A5968",
            "ytick.color": "#4A5968",
            "svg.fonttype": "none",
        }
    )
    prefix = "SYNTHETIC TEST — " if report["synthetic"] else ""
    for kind in ("throughput", "cold"):
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
        title = (
            "TaylorF2 · complete public batch call"
            if kind == "throughput"
            else "TaylorF2 · cold first public call"
        )
        fig.suptitle(
            prefix + title, x=0.065, y=0.965, ha="left", fontsize=17, fontweight="bold"
        )
        subtitle = (
            "Warm throughput · complex128 · both polarizations · one host thread"
            if kind == "throughput"
            else "Fresh per-worker Triton cache · includes compilation and lazy initialization"
        )
        fig.text(0.065, 0.898, subtitle, fontsize=10.3, color="#4A5968")
        for ax, (bins, delta_f) in zip(axes, GRIDS):
            ax.set_title(f"{bins:,} bins  |  Δf = {delta_f:g} Hz", loc="left", pad=12)
            for route in ROUTES:
                rows = [
                    next(
                        r
                        for r in report["rows"]
                        if r["bins"] == bins
                        and r["batch"] == batch
                        and r["route"] == route
                    )
                    for batch in BATCHES
                ]
                field = "throughput" if kind == "throughput" else "cold"
                scale = 1 if kind == "throughput" else 1000
                keys = (
                    ("throughput_median", "throughput_min", "throughput_max")
                    if kind == "throughput"
                    else ("cold_median_seconds", "cold_min_seconds", "cold_max_seconds")
                )
                x, middle, lower, upper = [], [], [], []
                for index, row in enumerate(rows):
                    if row["qualified"]:
                        values = [row[key] * scale for key in keys]
                        x.append(index)
                        middle.append(values[0])
                        lower.append(max(0, values[0] - values[1]))
                        upper.append(max(0, values[2] - values[0]))
                    else:
                        ax.annotate(
                            "unqualified",
                            (index, 0.025 if route == "cuda-off" else 0.085),
                            xycoords=("data", "axes fraction"),
                            ha="center",
                            fontsize=7.2,
                            color=COLORS[route],
                            rotation=25,
                        )
                if x:
                    # Only adjacent qualified cells are connected; a failed cell leaves a gap.
                    full = [math.nan] * len(BATCHES)
                    for index, value in zip(x, middle):
                        full[index] = value
                    ax.plot(
                        range(len(BATCHES)),
                        full,
                        color=COLORS[route],
                        linewidth=1.8,
                        alpha=0.8,
                    )
                    ax.errorbar(
                        x,
                        middle,
                        yerr=[lower, upper],
                        fmt="o" if route == "cuda-off" else "s",
                        markersize=6.5,
                        capsize=3.5,
                        elinewidth=1.2,
                        color=COLORS[route],
                        label=LABELS[route],
                    )
                else:
                    ax.plot(
                        [],
                        [],
                        "o-" if route == "cuda-off" else "s-",
                        color=COLORS[route],
                        label=LABELS[route],
                    )
                del field
            ax.set_xticks(range(len(BATCHES)), [str(b) for b in BATCHES])
            ax.set_xlabel("Waveforms per batch")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _position: format_number(value))
            )
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(axis="y", which="major", color="#D9E0E7", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.28, 4.28)
        axes[0].set_ylabel(
            "Waveforms / second  (log scale)"
            if kind == "throughput"
            else "First call, milliseconds  (log scale)"
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.06, 0.075),
            ncol=2,
            frameon=False,
        )
        footnote = "Points: median of 3 workers. Whiskers: observed worker range, not a confidence interval."
        if kind == "cold":
            footnote += " CUDA driver cache retained."
        fig.text(0.065, 0.037, footnote, fontsize=8.5, color="#596979")
        fig.text(
            0.945,
            0.965,
            report["expected_sha"][:10],
            ha="right",
            fontsize=9,
            color="#596979",
        )
        fig.subplots_adjust(left=0.08, right=0.975, top=0.78, bottom=0.25, wspace=0.17)
        for suffix in ("png", "svg"):
            fig.savefig(
                output_dir / f"taylorf2-{kind}.{suffix}", dpi=180, bbox_inches="tight"
            )
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument(
        "--harness-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "harness",
    )
    p.add_argument("--expected-sha", required=True)
    p.add_argument("--run-date", default="2026-09-06")
    p.add_argument("--tables-only", action="store_true")
    p.add_argument(
        "--synthetic-test",
        action="store_true",
        help="test-only; every input must be marked synthetic and output must be under the system temp directory",
    )
    args = p.parse_args()
    args.input_dir, args.output_dir, args.harness_dir = (
        args.input_dir.resolve(),
        args.output_dir.resolve(),
        args.harness_dir.resolve(),
    )
    if args.synthetic_test:
        try:
            args.output_dir.relative_to(Path(tempfile.gettempdir()).resolve())
        except ValueError:
            p.error("synthetic output must stay under the system temporary directory")
    report = load_campaign(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report["plots_generated"] = (
        not args.tables_only
        and not report["global_problems"]
        and report["raw_worker_files"] == 72
    )
    if report["plots_generated"]:
        plots(report, args.output_dir)
    else:
        # A failed rerender must not leave earlier figures looking current.
        for stem in ("taylorf2-throughput", "taylorf2-cold"):
            for suffix in ("png", "svg"):
                (args.output_dir / f"{stem}.{suffix}").unlink(missing_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "report.md").write_text(markdown(report))
    print(
        json.dumps(
            dict(
                status=report["status"],
                groups=report["qualified_group_count"],
                worker_files=report["raw_worker_files"],
                plots_generated=report["plots_generated"],
                output_dir=str(args.output_dir),
            )
        )
    )
    return int(report["qualified_group_count"] != 24)


if __name__ == "__main__":
    raise SystemExit(main())
