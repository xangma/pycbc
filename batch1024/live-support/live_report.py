#!/usr/bin/env python3
"""Render only the fresh, pinned-head live benchmark campaign.

Example: python live_report.py --input-dir ../live --output-dir .
No benchmark or repository code is imported. Requires matplotlib for plotting.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


MAIN_REVISION = "4885b64560e9f39b740e85b6a976898869dd360e"
CPU_REVISION = "d544420232428225c214a4be84fbe1262a6d307b"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
BATCHES = (1, 8, 32, 128, 512, 1024)
CPU_ROUTES = ("branch_standard", "torch_cpu", "torch_cpu_native")
ALL_ROUTES = (*CPU_ROUTES, "torch_cuda", "torch_cuda_native")
LABELS = {
    "branch_standard": "Standard CPU (same-head control)",
    "torch_cpu": "Torch CPU (defaults)",
    "torch_cpu_native": "Torch CPU (native requested)*",
    "torch_cuda": "Torch CUDA (defaults)",
    "torch_cuda_native": "Torch CUDA (native requested)",
}
COLORS = dict(zip(ALL_ROUTES, ("#536171", "#0072B2", "#009E73", "#D55E00", "#8A4DA8")))
CONDITIONS = {
    "batches": list(BATCHES),
    "replicates": 3,
    "samples": 5,
    "warmups": 2,
    "num_blocks": 3,
    "size": 131072,
    "seed": 7101,
    "call_surface": "public",
}
LIMITATIONS = [
    "Synthetic LiveBatchMatchedFilter.process_data library workload; not full CLI or application throughput.",
    "Waveform generation, PSD estimation, bank loading, frame I/O, startup, and workflow scheduling are excluded.",
    "Fixed complex64 strain/templates/output and float32 PSD; bank size equals batch size (1, 8, 32, 128, 512, or 1024); FFT length 131072.",
    "Chi-square is disabled and sine-Gaussian post-processing is stubbed in this harness.",
    "Parity is the harness's final trigger comparison and aggregate output-L2 check, not pointwise output equivalence.",
    "Three workers with five warm iterations each do not support population confidence intervals or tail-latency claims.",
    "Native requested labels describe configuration; admission and fallback require the separate untimed dispatch probes.",
    "Main native CPU requests correlation and FFTW batching. Optional CPU additionally requests native batch peaks; this comparison includes that configuration change.",
    "CUDA on-device peak helper is disabled in these routes; timings do not validate the separately reported asynchronous host-copy race.",
    "Cross-head ratios compare independently run campaign cells, not paired samples; each route is qualified against its own head's standard CPU control.",
]


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-9)


def finite_output(record):
    values = [record.get("output_l2", float("nan"))]
    blocks = record.get("block_triggers", [])
    if len(blocks) != CONDITIONS["num_blocks"]:
        return False
    for block in blocks:
        for name in ("coa_phases", "end_times", "sigmasqs", "snrs"):
            values.extend(block.get(name, []))
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def effective_flags(record):
    return {
        name: {
            key: flag.get(key)
            for key in ("applies_to_route", "enabled", "environment_value", "selection")
        }
        for name, flag in record["routing"]["feature_flags"].items()
    }


def unavailable_campaign(path, head, threads, revision, issue):
    routes = ALL_ROUTES if head == "main" and threads == 1 else CPU_ROUTES
    rows = [
        dict(
            head=head,
            revision=revision,
            threads=threads,
            batch=batch,
            route=route,
            label=LABELS[route],
            source_file=path.name,
            qualified=False,
            qualification_issues=[issue],
            parity_status="unqualified",
            median_throughput_wps=None,
            worker_range_wps=None,
            workers=[],
            feature_flags={},
            speedup_vs_same_head_standard_cpu=None,
        )
        for batch in BATCHES
        for route in routes
    ]
    return rows, dict(
        file=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
        head=head,
        revision=revision,
        threads=threads,
        started_utc="unavailable",
        finished_utc="unavailable",
        host="unavailable",
        runtime={},
        args={},
        source_identities={},
        all_parity_passed=False,
        qualification_issue=issue,
    )


def load_campaign(path, head, threads, revision, run_date):
    raw = path.read_bytes()
    data = json.loads(raw)
    assert data["artifact_type"] == "external_production_live_batch", path
    assert not data.get("synthetic_test", False), (
        "Synthetic validation input is not evidence"
    )
    assert data["source_unchanged_after"] is True, "Source changed during measurement"
    assert (
        data["support_sha256"]
        == hashlib.sha256(
            Path(__file__).with_name("run-live.py").read_bytes()
        ).hexdigest()
    ), "Wrong acquisition harness"
    material = dict(data)
    seal = material.pop("content_sha256", None)
    assert (
        seal
        == hashlib.sha256(
            json.dumps(
                material, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode()
        ).hexdigest()
    ), "Broken artifact content seal"
    assert data["started_utc"][:10] == run_date, f"Historical/wrong-date input: {path}"
    assert data["finished_utc"][:10] == run_date, f"Wrong completion date: {path}"
    assert data["args"]["threads"] == threads, path
    assert data["reference_route"] == "branch_standard", path
    for key, value in CONDITIONS.items():
        assert data["args"][key] == value, (path, key, data["args"][key], value)
    assert data["measurement"]["scope"] == "public_library_api", path
    required_routes = ALL_ROUTES if head == "main" and threads == 1 else CPU_ROUTES
    assert tuple(data["routes"]) == required_routes, (path, data["routes"])
    for route in required_routes:
        identity = data["source_identities"][route]
        assert identity["revision"] == revision, (path, route, "wrong revision")
        assert identity["dirty"] is False, (path, route, "dirty source")
        assert identity["tracked_diff_sha256"] == EMPTY_SHA256, (
            path,
            route,
            "tracked edits",
        )
        assert (
            identity["files_sha256"]["tools/bench_production_live_batch.py"]
            == data["launcher_sha256"]
        ), "Wrong committed worker source hash"
    assert len(data["records"]) <= len(required_routes) * len(BATCHES) * 3, (
        path,
        "excess worker count",
    )

    rows = []
    for batch in BATCHES:
        for route in required_routes:
            records = sorted(
                (
                    r
                    for r in data["records"]
                    if r["batch"] == batch and r["route"] == route
                ),
                key=lambda r: r["replicate"],
            )
            problems = []
            workers = []
            if [r["replicate"] for r in records] != [0, 1, 2]:
                problems.append("missing or duplicate worker replicates")
            relevant_failures = [
                failure
                for failure in data["failures"]
                if failure["batch"] == batch and failure["route"] == route
            ]
            for failure in relevant_failures:
                problems.append(
                    f"worker {failure['replicate']}: {failure.get('error_type', 'failure')}: {failure.get('error', 'see raw receipt')}"
                )
            flags = effective_flags(records[0]) if records else {}
            for record in records:
                replicate = record["replicate"]
                if effective_flags(record) != flags:
                    problems.append(f"worker {replicate}: inconsistent route flags")
                if (
                    record["threads"] != threads
                    or record["size"] != 131072
                    or record["num_blocks"] != 3
                    or record["seed"]
                    != 7101 + 10000 * replicate + 100 * record["cell_index"]
                    or record["dtypes"]
                    != {
                        "output": "complex64",
                        "psd": "float32",
                        "strain": "complex64",
                        "template": "complex64",
                    }
                ):
                    problems.append(f"worker {replicate}: mismatched workload")
                expected_cell_index = replicate * len(BATCHES) + BATCHES.index(batch)
                if (
                    record["cell_index"] != expected_cell_index
                    or record["source_root"] != data["root"]
                ):
                    problems.append(
                        f"worker {replicate}: incorrect seed cell or source root"
                    )
                if (
                    record["routing"]["feature_flags"]
                    .get("PYCBC_BATCH_MAXELEMENTS", {})
                    .get("selection")
                    != "constructor_argument"
                ):
                    problems.append(
                        f"worker {replicate}: logical batch size overridden"
                    )
                receipt = [
                    r
                    for r in data["worker_receipts"]
                    if r["batch"] == batch
                    and r["route"] == route
                    and r["replicate"] == replicate
                ]
                if len(receipt) != 1 or receipt[0]["status"] != "completed":
                    problems.append(f"worker {replicate}: missing acquisition receipt")
                else:
                    worker_path = path.parent / receipt[0]["raw_file"]
                    if (
                        not worker_path.is_file()
                        or hashlib.sha256(worker_path.read_bytes()).hexdigest()
                        != receipt[0]["raw_sha256"]
                        or json.loads(worker_path.read_bytes()) != record
                    ):
                        problems.append(
                            f"worker {replicate}: raw acquisition file missing or changed"
                        )
                latencies = record["warm_iteration_latencies_ms"]
                if len(latencies) != 5 or not all(
                    math.isfinite(v) and v > 0 for v in latencies
                ):
                    problems.append(f"worker {replicate}: invalid warm samples")
                    continue
                samples = [batch * 3 * 1000.0 / value for value in latencies]
                median = statistics.median(samples)
                stored = record["throughput_wps_summary"]
                if (
                    not close(median, stored["median"])
                    or len(stored["samples"]) != 5
                    or not all(close(a, b) for a, b in zip(samples, stored["samples"]))
                ):
                    problems.append(
                        f"worker {replicate}: inconsistent throughput summary"
                    )
                if not finite_output(record):
                    problems.append(f"worker {replicate}: missing/nonfinite output")
                comparison = None
                if route != "branch_standard":
                    comparison = (
                        data["parity_analysis"]
                        .get(f"batch_{batch}", {})
                        .get("replicates", {})
                        .get(f"replicate_{replicate}", {})
                        .get("comparisons", {})
                        .get(f"{route}_vs_branch_standard")
                    )
                    if comparison is None or comparison.get("passed") is not True:
                        problems.append(f"worker {replicate}: parity failed or missing")
                if (
                    route.startswith("torch")
                    and flags.get("PYCBC_TORCH_ONDEVICE_PEAKS", {}).get("enabled")
                    is not False
                ):
                    problems.append(
                        f"worker {replicate}: on-device peak helper not disabled"
                    )
                workers.append(
                    {
                        "replicate": replicate,
                        "pid": record["pid"],
                        "median_throughput_wps": median,
                        "throughput_samples_wps": samples,
                        "parity_comparison": comparison,
                    }
                )
            values = [w["median_throughput_wps"] for w in workers]
            center = statistics.median(values) if values else None
            summary = data["summary_by_batch"].get(f"batch_{batch}", {}).get(route, {})
            stored_estimate = summary.get("replicate_estimands", {}).get(
                "throughput_wps", {}
            )
            if center is None or not close(
                center, stored_estimate.get("median", float("nan"))
            ):
                problems.append(
                    "worker-derived estimate differs from artifact hierarchical estimate"
                )
            rows.append(
                {
                    "head": head,
                    "revision": revision,
                    "threads": threads,
                    "batch": batch,
                    "route": route,
                    "label": LABELS[route],
                    "source_file": path.name,
                    "qualified": not problems,
                    "qualification_issues": problems,
                    "parity_status": "reference finite output"
                    if route == "branch_standard"
                    else "passed"
                    if not problems
                    else "unqualified",
                    "median_throughput_wps": center,
                    "worker_range_wps": [min(values), max(values)] if values else None,
                    "workers": workers,
                    "feature_flags": flags,
                    "failures": relevant_failures,
                }
            )
    for row in rows:
        control = next(
            r
            for r in rows
            if r["batch"] == row["batch"] and r["route"] == "branch_standard"
        )
        row["speedup_vs_same_head_standard_cpu"] = (
            row["median_throughput_wps"] / control["median_throughput_wps"]
            if row["qualified"] and control["qualified"]
            else None
        )
    metadata = {
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "head": head,
        "revision": revision,
        "threads": threads,
        "started_utc": data["started_utc"],
        "finished_utc": data["finished_utc"],
        "host": data["host"],
        "runtime": data["runtime"],
        "args": data["args"],
        "source_identities": data["source_identities"],
        "all_parity_passed": data["parity_analysis"].get("all_passed_globally"),
        "completed_workers": len(data["records"]),
        "failed_workers": len(data["failures"]),
        "support_sha256": data["support_sha256"],
    }
    return rows, metadata


def cross_head_rows(rows):
    result = []
    for threads in (1, 4):
        for batch in BATCHES:
            for route in CPU_ROUTES:
                pair = [
                    next(
                        r
                        for r in rows
                        if r["head"] == head
                        and r["threads"] == threads
                        and r["batch"] == batch
                        and r["route"] == route
                    )
                    for head in ("main", "cpu")
                ]
                before, after = pair
                qualified = all(r["qualified"] for r in pair)
                result.append(
                    {
                        "threads": threads,
                        "batch": batch,
                        "route": route,
                        "qualified": qualified,
                        "main_median_wps": before["median_throughput_wps"],
                        "optional_cpu_median_wps": after["median_throughput_wps"],
                        "optional_over_main": after["median_throughput_wps"]
                        / before["median_throughput_wps"]
                        if qualified
                        else None,
                        "configuration_change": "optional CPU additionally requests native batch peaks"
                        if route == "torch_cpu_native"
                        else None,
                    }
                )
    return result


def render_plots(rows, output, run_date):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )

    def chart(ax, subset, label, color, marker="o", ratio=False):
        valid = [
            r
            for r in subset
            if r["qualified"]
            and (not ratio or r["speedup_vs_same_head_standard_cpu"] is not None)
        ]
        if not valid:
            return
        xs = [BATCHES.index(r["batch"]) for r in valid]
        if ratio:
            ax.plot(
                xs,
                [r["speedup_vs_same_head_standard_cpu"] for r in valid],
                label=label,
                color=color,
                marker=marker,
                linewidth=1.6,
            )
        else:
            ys = [r["median_throughput_wps"] for r in valid]
            errors = [
                [y - r["worker_range_wps"][0] for y, r in zip(ys, valid)],
                [r["worker_range_wps"][1] - y for y, r in zip(ys, valid)],
            ]
            ax.errorbar(
                xs,
                ys,
                yerr=errors,
                label=label,
                color=color,
                marker=marker,
                capsize=4,
                linewidth=1.6,
            )

    def decorate(ax, ratio=False):
        ax.set_xticks(range(len(BATCHES)), BATCHES)
        ax.set_xlabel("Batch size / templates in bank")
        ax.set_yscale("log", base=10 if not ratio else 2)
        ax.grid(axis="y", alpha=0.22, which="both")
        ax.set_axisbelow(True)
        if ratio:
            ax.axhline(1, color="#536171", linestyle="--", linewidth=1)
            ax.set_ylabel("Throughput ratio (×; higher is faster)")
        else:
            ax.set_ylabel("Template-block evaluations / second")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), layout="constrained")
    for i, threads in enumerate((1, 4)):
        routes = ALL_ROUTES if threads == 1 else CPU_ROUTES
        for route in routes:
            subset = [
                r
                for r in rows
                if r["head"] == "main"
                and r["threads"] == threads
                and r["route"] == route
            ]
            chart(axes[i, 0], subset, LABELS[route], COLORS[route])
            if route != "branch_standard":
                chart(axes[i, 1], subset, LABELS[route], COLORS[route], ratio=True)
        for j in (0, 1):
            decorate(axes[i, j], ratio=bool(j))
        axes[i, 0].set_title(
            f"Warm throughput · {threads} CPU thread{'s' if threads > 1 else ''}",
            loc="left",
        )
        axes[i, 1].set_title(
            f"Ratio to same-head standard CPU · {threads} CPU thread{'s' if threads > 1 else ''}",
            loc="left",
        )
    handles = [
        Line2D([], [], color=COLORS[r], marker="o", label=LABELS[r]) for r in ALL_ROUTES
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(
        f"Main stack: synthetic live matched filtering · {run_date}\nMedian of 3 worker medians; throughput whiskers = observed worker range, not a confidence interval\n*Main native CPU requests correlation and FFTW batching; ratios are ratios of displayed medians",
        fontsize=13,
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"main-live.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        2, 3, figsize=(14, 8.2), layout="constrained", sharey="row"
    )
    for i, threads in enumerate((1, 4)):
        for j, route in enumerate(CPU_ROUTES):
            ax = axes[i, j]
            for head, color, marker in (
                ("main", "#536171", "o"),
                ("cpu", "#0072B2", "s"),
            ):
                subset = [
                    r
                    for r in rows
                    if r["head"] == head
                    and r["threads"] == threads
                    and r["route"] == route
                ]
                chart(
                    ax,
                    subset,
                    "Main stack" if head == "main" else "Optional CPU follow-up",
                    color,
                    marker,
                )
            decorate(ax)
            ax.set_title(
                f"{LABELS[route]}\n{threads} CPU thread{'s' if threads > 1 else ''}",
                loc="left",
            )
            if j:
                ax.set_ylabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(
        f"Optional CPU follow-up: warm live throughput · {run_date}\nMedian of 3 worker medians; whiskers = observed worker range, not a confidence interval\n*Native CPU configuration adds requested batch peaks in the optional follow-up",
        fontsize=13,
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"optional-cpu.{ext}", dpi=180)
    plt.close(fig)


def markdown(summary):
    lines = [
        f"Fresh live benchmark results ({summary['run_date']}).",
        "",
        "Throughput is the median of three independent worker medians, each from five warm iterations. Ranges show the minimum and maximum worker median, not confidence intervals. One unit is one template evaluated against one data block; waveform generation is excluded. Ratios divide the displayed median throughputs and have no uncertainty bars.",
        "",
        f"Main revision: `{summary['main_revision']}`. Optional CPU revision: `{summary['cpu_revision']}`.",
        "",
        "Each route is compared with standard CPU from the same source head, thread count, and batch size. Rows failing the recorded parity or provenance requirements have no performance claim. All measured routes, including ratios below one, remain in the tables.",
        "",
        "| Head | CPU threads | Batch | Route | Throughput / s | Worker range / s | Ratio to same-head CPU | Qualification |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["rows"]:
        ok = row["qualified"]
        rate = f"{row['median_throughput_wps']:,.1f}" if ok else "—"
        interval = "–".join(f"{v:,.1f}" for v in row["worker_range_wps"]) if ok else "—"
        ratio = row["speedup_vs_same_head_standard_cpu"]
        relative = f"{ratio:.3f}×" if ratio is not None else "—"
        status = row["parity_status"] if ok else "; ".join(row["qualification_issues"])
        lines.append(
            f"| {row['head']} | {row['threads']} | {row['batch']} | {row['label']} | {rate} | {interval} | {relative} | {status} |"
        )
    lines += [
        "",
        "Optional CPU / main ratios use independently measured cells. Native requested CPU additionally enables native batch peaks on the optional head; this measures both the follow-up and its supplied route configuration.",
        "",
        "| CPU threads | Batch | Route | Optional CPU / main | Configuration difference |",
        "| ---: | ---: | --- | ---: | --- |",
    ]
    for row in summary["cross_head_cpu"]:
        ratio = row["optional_over_main"]
        relative = f"{ratio:.3f}×" if ratio is not None else "unqualified"
        lines.append(
            f"| {row['threads']} | {row['batch']} | {LABELS[row['route']]} | {relative} | {row['configuration_change'] or 'None requested by the route'} |"
        )
    lines += ["", "Measurement limits:", ""]
    lines += [f"- {limit}" for limit in LIMITATIONS]
    lines += ["", "Source files (SHA-256):", ""]
    lines += [
        f"- `{s['file']}`: `{s['sha256']}`; {s['started_utc']} to {s['finished_utc']}; host `{s['host']}`."
        for s in summary["sources"]
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-date", default="2026-09-06")
    parser.add_argument("--main-revision", required=True)
    parser.add_argument("--cpu-revision", required=True)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    rows, sources = [], []
    for head, revision in (("main", args.main_revision), ("cpu", args.cpu_revision)):
        for threads in (1, 4):
            path = args.input_dir / f"{head}-t{threads}.json"
            try:
                new_rows, metadata = load_campaign(
                    path, head, threads, revision, args.run_date
                )
            except (AssertionError, KeyError, ValueError, TypeError, OSError) as exc:
                new_rows, metadata = unavailable_campaign(
                    path,
                    head,
                    threads,
                    revision,
                    f"Campaign unavailable or invalid: {type(exc).__name__}: {exc}",
                )
            rows.extend(new_rows)
            sources.append(metadata)
    valid_sources = [s for s in sources if "qualification_issue" not in s]
    assert len({s["host"] for s in valid_sources}) <= 1, (
        "Cross-head inputs used different hosts"
    )
    for key in ("affinity", "snr_threshold"):
        assert len({s["args"][key] for s in valid_sources}) <= 1, f"Mismatched {key}"
    runtime_keys = ("python", "torch", "torch_cuda", "hardware")
    for key in runtime_keys:
        assert (
            len(
                {
                    json.dumps(s["runtime"].get(key), sort_keys=True)
                    for s in valid_sources
                }
            )
            <= 1
        ), f"Mismatched runtime {key}"
    summary = {
        "schema_version": 1,
        "run_date": args.run_date,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "main_revision": args.main_revision,
        "cpu_revision": args.cpu_revision,
        "estimand": "median of three worker median throughputs",
        "dispersion": "observed min/max of three worker medians; not a confidence interval",
        "ratio_estimand": "ratio of displayed median-of-worker-median throughputs",
        "workload": CONDITIONS,
        "limitations": LIMITATIONS,
        "rows": rows,
        "cross_head_cpu": cross_head_rows(rows),
        "sources": sources,
        "qualified_rows": sum(r["qualified"] for r in rows),
        "total_rows": len(rows),
        "expected_rows": len(BATCHES) * (len(ALL_ROUTES) + 3 * len(CPU_ROUTES)),
        "expected_workers": len(BATCHES) * (len(ALL_ROUTES) + 3 * len(CPU_ROUTES)) * 3,
    }
    assert len(rows) == summary["expected_rows"], "Incomplete report matrix"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "live-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "live-summary.md").write_text(markdown(summary))
    if not args.tables_only and any(r["qualified"] for r in rows):
        render_plots(rows, args.output_dir, args.run_date)
    else:
        for filename in (
            "main-live.png",
            "main-live.svg",
            "optional-cpu.png",
            "optional-cpu.svg",
        ):
            (args.output_dir / filename).unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "qualified_rows": summary["qualified_rows"],
                "total_rows": len(rows),
                "expected_rows": len(BATCHES) * (len(ALL_ROUTES) + 3 * len(CPU_ROUTES)),
                "expected_workers": len(BATCHES)
                * (len(ALL_ROUTES) + 3 * len(CPU_ROUTES))
                * 3,
                "output": str(args.output_dir.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
