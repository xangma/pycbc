#!/usr/bin/env python3
"""Export a separate before/v1/v2 CUDA supplement from completed measurements.

The original campaign and build-report.py remain unchanged. V2 was measured in
a later run: replicate-matched ratios are descriptive, not paired timing or
confidence intervals. The output directory must not exist.
"""

import argparse
import copy
import importlib.util
import json
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

SOURCES = ("baseline", "candidate", "candidate-v2")
EXPECTED = {
    "baseline": (
        "dfd42bf76766cadca0eecf609a1eaeac73534676",
        "b0d5ef6e3b39bfe246b4c416528839256edadf1f",
    ),
    "candidate": (
        "97bf1614f3afe53a6e2edb4d8c7dff79e9661782",
        "d9819f6f6ebf2998ac967178778cc1911757e455",
    ),
    "candidate-v2": (
        "0d00581251e642a5d6b56b2497a9adad93069e6b",
        "fa7a7c09df6d93133324f762d756842de172433d",
    ),
}
ROUTES = ("torch_cuda", "torch_cuda_native")
DISPLAY = {"baseline": "Before", "candidate": "V1", "candidate-v2": "V2 (follow-on)"}
RUNNER_SHA256 = "a69ef8762c585c2040e5d6a3e91e16b371c606a2a63353b8ee57fdc83844d795"


def checked_parity(document, records, control, route, batch):
    """Check all three measured CUDA revisions against the original control."""
    require(
        set(document) == {f"batch_{batch}", "all_passed_globally"}
        and document["all_passed_globally"] is True,
        f"{route}/{batch}: supplement global parity",
    )
    cell = document[f"batch_{batch}"]
    require(
        cell["batch"] == batch
        and cell["all_passed"] is True
        and cell["routes_evaluated"] == ["branch_standard", *SOURCES]
        and set(cell["comparisons"])
        == {f"{name}_vs_branch_standard" for name in SOURCES},
        f"{route}/{batch}: supplement parity coverage",
    )
    for name in ("candidate", "candidate-v2"):
        view = copy.deepcopy(document)
        adapted = view[f"batch_{batch}"]
        adapted["routes_evaluated"] = ["branch_standard", route, route + "_fixed"]
        adapted["comparisons"] = {
            f"{route}_vs_branch_standard": cell["comparisons"][
                "baseline_vs_branch_standard"
            ],
            f"{route}_fixed_vs_branch_standard": cell["comparisons"][
                f"{name}_vs_branch_standard"
            ],
        }
        shared.checked_live_parity(
            view, records["baseline"], records[name], control, route, batch
        )


def validate(original, original_status_path, supplement):
    # This validates the whole frozen v1 campaign, including original CUDA
    # parity and its completion ledger, without exporting or editing its report.
    main, manifest = shared.validate(original, original_status_path)

    def read(path):
        manifest[str(path)] = shared.digest(path)
        return shared.load(path)

    status_path = supplement / "status.json"
    status = read(status_path)
    require(
        status["state"] == "complete" and not status.get("error"),
        "CUDA v2 supplement is incomplete or failed",
    )
    source = status["source"]
    sources = {**main["sources"], "candidate-v2": source}
    root = Path(sources["baseline"]["root"]).parent
    for name, value in sources.items():
        shared.identity(value, name)
        require(
            (value["sha"], value["tree"]) == EXPECTED[name]
            and value["root"] == str(root / name)
            and value["status"] == "",
            f"{name}: unexpected source revision, tree, root or dirty status",
        )
    require(
        status["source_before"] == status["source_after"] == source,
        "v2 source changed during measurement",
    )
    require(status["host"] == main["runtime"]["hostname"], "supplement host differs")
    require(
        status["original_sources"] == main["sources"]
        and status["original_status_sha256"] == manifest[str(original_status_path)],
        "supplement references a different original campaign",
    )
    started = datetime.fromisoformat(status["started_utc"])
    finished = datetime.fromisoformat(status["finished_utc"])
    main_finished = datetime.fromisoformat(main["campaign_finished_utc"])
    require(finished > started and finished >= main_finished, "supplement timestamps")
    measured_start = datetime.fromisoformat(status["timing_started_utc"])
    require(main_finished <= measured_start < finished, "v2 timing did not follow v1")
    runner = supplement.parent / "run-accelerator-refinement.py"
    manifest[str(runner)] = shared.digest(runner)
    require(
        manifest[str(runner)] == status["runner_sha256"] == RUNNER_SHA256,
        "v2 runner hash differs from the reviewed setup repair",
    )
    snapshots = read(supplement / "sources-before.json")
    require(
        snapshots == read(supplement / "sources-after.json"),
        "source bytes, modes or native binaries changed during v2 run",
    )

    def compact(value):
        return {key: value[key] for key in ("root", "sha", "tree", "status")}

    require(
        {name: compact(value) for name, value in snapshots.items()}
        == status["sources"],
        "full snapshots differ from status identities",
    )
    for name in SOURCES:
        require(
            compact(snapshots[name]) == sources[name]
            and snapshots[name]["tracked_files"]
            and snapshots[name]["native_binaries"],
            f"{name}: incomplete source snapshot",
        )
    require(
        snapshots["candidate"]["native_binaries"]
        == snapshots["candidate-v2"]["native_binaries"],
        "v1/v2 native binaries differ",
    )
    harness_files = (
        "tools/bench_production_live_batch.py",
        "tools/benchmark_artifact.py",
    )
    require(set(status["harness_sha256"]) == set(harness_files), "harness inventory")
    for path in harness_files:
        reference = snapshots["baseline"]["tracked_files"][path]
        require(
            reference["sha256"] == status["harness_sha256"][path]
            and all(
                snapshots[name]["tracked_files"][path] == reference for name in SOURCES
            ),
            f"harness bytes or modes differ: {path}",
        )
    frozen_inputs = read(supplement / "inputs-sha256.json")
    require(
        frozen_inputs[str(root / "comparison-status.json")]
        == status["original_status_sha256"],
        "frozen original status hash differs",
    )

    keys = [
        (route, batch, 1, rep)
        for route in ROUTES
        for batch in shared.BATCHES
        for rep in shared.REPLICATES
    ]
    labels = [shared.label(*key, "candidate-v2") for key in keys]
    require(
        len(status["completed"]) == len(set(status["completed"])) == 36
        and set(status["completed"]) == set(labels),
        "v2 completion ledger must contain exactly the expected 36 workers",
    )
    shared.exact_files(supplement / "live", labels)
    shared.exact_files(supplement / "parity", [shared.label(*key) for key in keys])
    records, values, controls, pids = {}, {}, {}, set()
    parity_maxima = {}
    for key in keys:
        route, batch, _, rep = key
        group = {}
        for name in SOURCES:
            directory = supplement if name == "candidate-v2" else original
            record = read(directory / "live" / f"{shared.label(*key, name)}.json")
            values[(*key, name)] = shared.checked_live(
                record, (*key, name), sources[name]
            )
            records[(*key, name)] = group[name] = record
            stem = shared.label(*key, name)
            if name != "candidate-v2":
                require(
                    frozen_inputs[str(root / "comparison" / "live" / f"{stem}.json")]
                    == manifest[str(directory / "live" / f"{stem}.json")],
                    f"{stem}: reused worker changed",
                )
            else:
                receipt = read(supplement / "commands" / f"{stem}.json")
                require(
                    receipt["label"] == stem
                    and receipt["returncode"] == 0
                    and receipt["host"] == status["host"]
                    and receipt["cwd"] == source["root"]
                    and receipt["worker_pid"] == record["pid"]
                    and receipt["worker_command"] == record["command"]
                    and receipt["source_before"] == receipt["source_after"] == source
                    and record["comparison_variant"] == "candidate-v2",
                    f"{stem}: worker receipt mismatch",
                )
                require(
                    receipt["command"]
                    == [
                        record["command"][3],
                        str(root / "run-accelerator-refinement.py"),
                        "--worker",
                        "--root",
                        str(root),
                        "--python",
                        record["command"][3],
                        "--route",
                        route,
                        "--batch",
                        str(batch),
                    ],
                    f"{stem}: wrapper command mismatch",
                )
                shared.number(receipt["wall_seconds"], stem, positive=True)
                shared.number(record["worker_wall_seconds"], stem, positive=True)
                require(
                    measured_start
                    <= datetime.fromisoformat(receipt["started_utc"])
                    < datetime.fromisoformat(receipt["finished_utc"])
                    <= finished,
                    f"{stem}: receipt timestamps outside timing window",
                )
                log = supplement / "logs" / f"{stem}.log"
                require(
                    receipt["log_path"]
                    == str(Path(status["log_path"]).parent / "logs" / log.name),
                    f"{stem}: log path mismatch",
                )
                manifest[str(log)] = shared.digest(log)
                lines = [
                    line.removeprefix("RESULT_JSON=")
                    for line in log.read_text().splitlines()
                    if line.startswith("RESULT_JSON=")
                ]
                require(len(lines) == 1, f"{stem}: expected one raw worker result")
                logged = json.loads(lines[0], object_pairs_hook=shared.no_duplicates)
                logged.update(
                    comparison_source=source,
                    comparison_label=stem,
                    comparison_variant="candidate-v2",
                    comparison_replicate=rep,
                )
                require(
                    logged == record, f"{stem}: exported worker differs from raw log"
                )
            pid = record["pid"]
            require(
                type(pid) is int and pid > 0 and pid not in pids,
                f"{shared.label(*key, name)}: repeated or invalid worker PID",
            )
            pids.add(pid)
        control_key = (batch, rep)
        if control_key not in controls:
            control_path = (
                original
                / "live"
                / f"{shared.label('branch_standard', batch, 1, rep, 'baseline')}.json"
            )
            control = read(control_path)
            require(
                frozen_inputs[str(root / "comparison" / "live" / control_path.name)]
                == manifest[str(control_path)],
                "reused standard control changed",
            )
            shared.checked_live(
                control,
                ("branch_standard", batch, 1, rep, "baseline"),
                sources["baseline"],
            )
            require(
                type(control["pid"]) is int
                and control["pid"] > 0
                and control["pid"] not in pids,
                "repeated standard control PID",
            )
            pids.add(control["pid"])
            controls[control_key] = control
        control = controls[control_key]
        shared.consistent(
            [*group.values(), control],
            ("injection_metadata",),
            "CUDA/standard seeded inputs",
        )
        # Commands differ only in the source checkout path and label; the shared
        # validator has already checked every argument and timing denominator.
        shared.consistent(
            list(group.values()), ("routing",), f"{route}: routing across revisions"
        )
        require(
            len({r["command"][3] for r in group.values()}) == 1,
            f"{route}/{batch}: Python executable changed",
        )
        document = read(supplement / "parity" / f"{shared.label(*key)}.json")
        checked_parity(document, group, control, route, batch)
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

    all_records = list(records.values())
    shared.consistent(
        all_records,
        (
            "python",
            "numpy_version",
            "torch_version",
            "cuda_device_name",
            "measurement",
            "dtypes",
        ),
        "CUDA runtime and workload across all revisions",
    )
    require(
        all_records[0]["cuda_device_name"] == main["cuda_device"]["name"],
        "CUDA supplement device differs from original campaign",
    )
    rows = []
    for route in ROUTES:
        for batch in shared.BATCHES:
            workers = {
                name: [
                    values[(route, batch, 1, rep, name)] for rep in shared.REPLICATES
                ]
                for name in SOURCES
            }
            comparisons = {}
            for numerator, denominator in (
                ("candidate", "baseline"),
                ("candidate-v2", "baseline"),
                ("candidate-v2", "candidate"),
            ):
                ratios = [
                    a / b
                    for a, b in zip(
                        workers[numerator], workers[denominator], strict=True
                    )
                ]
                comparisons[f"{numerator}_over_{denominator}"] = dict(
                    replicate_matched_ratio=shared.summary(ratios),
                    ratio_of_medians=statistics.median(workers[numerator])
                    / statistics.median(workers[denominator]),
                    measurement_order="paired original campaign"
                    if numerator == "candidate"
                    else "follow-on v2 versus earlier original campaign; not paired timing",
                )
            rows.append(
                dict(
                    route=route,
                    batch=batch,
                    threads=1,
                    throughput={
                        name: shared.summary(workers[name]) for name in SOURCES
                    },
                    ratios=comparisons,
                )
            )
    for row in rows:
        normalized = shared.search_capacity(row["route"], 1, row["throughput"])
        row["search_capacity"] = {
            key: value for key, value in normalized.items() if key not in SOURCES
        }
        row["search_capacity"]["throughput"] = {
            name: normalized[name] for name in SOURCES
        }
    manifest[str(HELPER)] = shared.digest(HELPER)
    manifest[str(Path(__file__).resolve())] = shared.digest(Path(__file__).resolve())
    return dict(
        schema=1,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        title="CUDA refinement: before, v1 and later v2 measurements",
        sources=sources,
        runtime=main["runtime"],
        cuda_device=main["cuda_device"],
        unit="templates/second",
        search_capacity_units=dict(shared.CAPACITY_UNITS),
        search_capacity_basis=shared.capacity_basis(
            {name: snapshots[name] for name in SOURCES}
        ),
        batches=list(shared.BATCHES),
        replicates=list(shared.REPLICATES),
        original_campaign=dict(
            started_utc=main["campaign_started_utc"],
            finished_utc=main["campaign_finished_utc"],
            status_path=str(original_status_path),
            fully_validated=True,
        ),
        v2_campaign=dict(
            started_utc=status["started_utc"],
            finished_utc=status["finished_utc"],
            timing_started_utc=status.get("timing_started_utc"),
            status_path=str(status_path),
            source_snapshot_pair_checked="source_before" in status,
            recorded_host=status.get("host"),
            runner_sha256=status.get("runner_sha256"),
        ),
        counts=dict(
            original_cuda_workers=72,
            fresh_v2_cuda_workers=36,
            reused_original_standard_controls=18,
            v2_parity_documents=36,
            v2_parity_comparisons=108,
        ),
        validation=dict(
            complete=True,
            all_original_campaign_gates_passed=True,
            all_v2_parity_passed=True,
            parity_recomputed=True,
            parity_maxima=parity_maxima,
        ),
        comparisons=rows,
        notes={
            "aggregation": "Every point is the median of three distinct worker throughput medians; "
            "ranges are the full minimum–maximum across workers, not confidence intervals. "
            "The JSON also retains each worker value and every replicate-matched ratio.",
            "measurement_order": "Before and v1 came from the original paired campaign. V2 was "
            "measured afterward in a separate follow-on run. Ratios involving v2 match replicate "
            "labels for description; they are not paired timing estimates and do not remove "
            "possible run-period effects. The original before/v1 results remain unchanged.",
            "timing": "All workers use one thread, CPU affinity 8–11, the same interpreter and "
            "device, one cold iteration, one warmup and three timed iterations. Each timed "
            "iteration filters the batch against three strain blocks. Profiling is excluded.",
            "workload": shared.NOTES["live"],
            "search_capacity": shared.NOTES["search_capacity"],
            "parity": shared.NOTES["live_parity"]
            + " The supplement reuses the original "
            "standard controls for the same batch, seed and replicate and independently "
            "recomputes before/control, v1/control and v2/control comparisons. It does not "
            "contain newly measured contemporaneous standard controls.",
            "scope": "This supplement covers the two CUDA matched-filter routes only. It "
            "does not replace the original full campaign report or claim new CPU or waveform "
            "timings. Native route labels indicate enabled admission gates and can include "
            "fallbacks; these records do not count native dispatches.",
        },
    ), manifest


def figures(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    styles = {
        "baseline": dict(
            color="#616B75", linestyle="--", marker="o", markerfacecolor="white"
        ),
        "candidate": dict(
            color="#B2601C", linestyle=":", marker="s", markerfacecolor="white"
        ),
        "candidate-v2": dict(
            color="#0072B2", linestyle="-", marker="o", markerfacecolor="#0072B2"
        ),
    }
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
        peak = 0
        for ax, route in zip(axes, ROUTES, strict=True):
            rows = [row for row in report["comparisons"] if row["route"] == route]
            for source in SOURCES:
                series = [row["search_capacity"]["throughput"][source] for row in rows]
                medians = [value["median"] for value in series]
                lower = [value["minimum"] for value in series]
                upper = [value["maximum"] for value in series]
                peak = max(peak, *upper)
                ax.fill_between(
                    shared.BATCHES,
                    lower,
                    upper,
                    color=styles[source]["color"],
                    alpha=0.09,
                )
                ax.errorbar(
                    shared.BATCHES,
                    medians,
                    yerr=[
                        [
                            median - low
                            for median, low in zip(medians, lower, strict=True)
                        ],
                        [
                            high - median
                            for median, high in zip(medians, upper, strict=True)
                        ],
                    ],
                    capsize=3,
                    linewidth=1.8,
                    markersize=5,
                    label=DISPLAY[source],
                    **styles[source],
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks(shared.BATCHES, [str(batch) for batch in shared.BATCHES])
            ax.set_xlabel("Batch size (templates)")
            ax.set_title(
                "CUDA default"
                if route == "torch_cuda"
                else "CUDA native routing enabled"
            )
            ax.grid(axis="y", color="#E4E7EA", linewidth=0.7)
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axes[0].set_ylabel("Templates / GPU at real time")
        axes[0].set_ylim(0, peak * 1.1)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=3,
            frameon=False,
        )
        fig.suptitle(
            "CUDA filtering capacity: before, v1 and follow-on v2", y=0.98, fontsize=15
        )
        fig.text(
            0.5,
            0.825,
            "Synthetic matched-filter component · 56 s valid strain/block · Vetoes and I/O excluded",
            ha="center",
            fontsize=9,
        )
        fig.text(
            0.5,
            0.105,
            "Points: median of 3 workers · Bars/bands: full worker range · V2 measured later; ratios are descriptive",
            ha="center",
            fontsize=9,
        )
        revisions = " · ".join(
            f"{DISPLAY[name]} {report['sources'][name]['sha'][:12]}" for name in SOURCES
        )
        fig.text(0.5, 0.055, revisions, ha="center", fontsize=8, color="#555555")
        fig.subplots_adjust(top=0.78, bottom=0.23, left=0.09, right=0.98, wspace=0.16)
        names = []
        for extension in ("png", "svg"):
            name = f"cuda-before-v1-v2.{extension}"
            fig.savefig(output / name, dpi=220, facecolor="white")
            names.append(name)
        plt.close(fig)
    return names


def markdown(report):
    lines = [
        "# CUDA refinement: before, v1 and later v2 measurements",
        "",
        report["notes"]["measurement_order"],
        "",
        "| Version | Commit | Tree |",
        "| --- | --- | --- |",
    ]
    for name in SOURCES:
        source = report["sources"][name]
        lines.append(f"| {DISPLAY[name]} | `{source['sha']}` | `{source['tree']}` |")
    lines.extend(
        [
            "",
            f"Original campaign: {report['original_campaign']['started_utc']} to "
            f"{report['original_campaign']['finished_utc']}.",
            "",
            f"V2 run: {report['v2_campaign']['started_utc']} to {report['v2_campaign']['finished_utc']}.",
            "",
        ]
    )
    if report["v2_campaign"]["timing_started_utc"]:
        lines.extend(
            [f"V2 timing began: {report['v2_campaign']['timing_started_utc']}.", ""]
        )
    for name in report.get("figures", []):
        if name.endswith(".png"):
            lines.extend(
                [f"![Before, v1 and later v2 CUDA filtering capacity]({name})", ""]
            )
    lines.extend(
        [
            "Capacity cells are median [minimum, maximum] in templates/GPU at real time across three workers.",
            "",
            "| Route | Batch | Before | V1 | V2 (follow-on) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["comparisons"]:
        cells = " | ".join(
            shared.format_stat(row["search_capacity"]["throughput"][name])
            for name in SOURCES
        )
        lines.append(f"| {row['route']} | {row['batch']} | {cells} |")
    lines.extend(
        [
            "",
            "Ratios are median [minimum, maximum] of the three replicate-matched "
            "worker ratios. V2 comparisons span separate run periods.",
            "",
            "| Route | Batch | V1 / before | V2 / before | V2 / V1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["comparisons"]:
        cells = " | ".join(
            shared.format_stat(
                row["ratios"][key]["replicate_matched_ratio"], ratio=True
            )
            for key in (
                "candidate_over_baseline",
                "candidate-v2_over_baseline",
                "candidate-v2_over_candidate",
            )
        )
        lines.append(f"| {row['route']} | {row['batch']} | {cells} |")
    lines.extend(
        [
            "",
            "The complete original campaign passed its existing validation. This supplement "
            "adds 36 distinct v2 workers and 36 parity documents, with all 108 before/v1/v2 "
            "comparisons to the 18 reused standard controls passing independently recomputed "
            "trigger and aggregate norm checks.",
            "",
            "| Parity metric | Maximum |",
            "| --- | ---: |",
        ]
    )
    for metric, value in report["validation"]["parity_maxima"].items():
        lines.append(f"| `{metric}` | {value:.9g} |")
    for key, value in report["notes"].items():
        if key != "measurement_order":
            lines.extend(["", value])
    lines.extend(
        [
            "",
            "The input manifest preserves the hashes of the frozen original evidence, "
            "supplement status, all v2 worker/parity records and reporting scripts. "
            "JSON also retains full source identities, raw worker medians, every ratio "
            "and the separate ratio of aggregate medians.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=HERE / "comparison")
    parser.add_argument("--original-status", type=Path)
    parser.add_argument("--input", type=Path, default=HERE / "accelerator-refinement")
    parser.add_argument("--out", type=Path, default=HERE / "cuda-v2-followup-report")
    parser.add_argument(
        "--no-plots", action="store_true", help="Export JSON/Markdown only"
    )
    args = parser.parse_args()
    original, supplement, output = (
        args.original.resolve(),
        args.input.resolve(),
        args.out.resolve(),
    )
    original_status = (
        args.original_status or original.parent / "comparison-status.json"
    ).resolve()
    temporary = None
    try:
        require(
            not output.exists(), f"output already exists; choose a new --out: {output}"
        )
        report, manifest = validate(original, original_status, supplement)
        require(
            output.parent.is_dir(), f"output parent does not exist: {output.parent}"
        )
        temporary = Path(tempfile.mkdtemp(prefix=".cuda-v2-report-", dir=output.parent))
        report["figures"] = [] if args.no_plots else figures(report, temporary)
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
        print(f"CUDA v2 report refused: {error}", file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
