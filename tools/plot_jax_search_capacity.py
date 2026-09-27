#!/usr/bin/env python3
"""Render capacity summaries for complete JAX search campaign receipts.

The primary input is a complete-executable receipt.  The older synthetic
filter receipt remains available through :func:`plot_live` as an explicit
diagnostic, but is never selected by the command-line defaults.
"""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np


MICRO_ARMS = (
    ("branch_cpu", "Standard CPU / core", "#666666"),
    ("jax_cpu_lal", "JAX CPU / core", "#0072B2"),
    ("jax_cuda_lal", "JAX CUDA / GPU + host core", "#D55E00"),
)


def capacity_summary(work, elapsed):
    """Return median, minimum and maximum capacity for measured samples."""
    if work <= 0 or not elapsed:
        raise ValueError("Completed work and timing samples must be positive")
    if any(value <= 0 for value in elapsed):
        raise ValueError("Timing samples must be positive")
    rates = [work / value for value in elapsed]
    return median(rates), min(rates), max(rates)


def plot_live(data, output, advance_seconds=56.0):
    """Render the retained synthetic filter diagnostic (opt-in only)."""
    if not 0 < advance_seconds <= 64:
        raise ValueError("Live block advance must be in (0, 64]")
    arms = data["experiments"]["streaming_n131072"]["arms"]
    trial_counts = {
        len(cell["samples_seconds"]["filter"])
        for key, _, _ in MICRO_ARMS
        for cell in arms[key]["batches"].values()
    }
    if len(trial_counts) != 1:
        raise ValueError(
            "Live filter cells must have the same number of trials")
    trials = trial_counts.pop()
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for key, label, color in MICRO_ARMS:
        cells = arms[key]["batches"]
        batches = sorted(map(int, cells))
        summaries = [capacity_summary(
            batch * advance_seconds,
            cells[str(batch)]["samples_seconds"]["filter"],
        ) for batch in batches]
        mid, low, high = np.array(summaries).T
        ax.errorbar(batches, mid, yerr=[mid - low, high - mid],
                    label=label, color=color, marker="o", capsize=4)
    ax.set(xscale="log", yscale="log", xticks=batches,
           xticklabels=list(map(str, batches)), xlabel="Template batch size",
           ylabel="Diagnostic real-time templates per allocated resource",
           title="Synthetic filter diagnostic (N = 131,072)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.text(0.5, 0.02,
             f"Assumed advance: {advance_seconds:g} s per 64 s block. "
             "Filter stage only; no vetoes or streaming pipeline.\n"
             f"{trials} timed trials: median and observed range. "
             "Not pycbc_live capacity.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _timing(run, name):
    """Read current and legacy timing keys while receipts migrate."""
    if name in run:
        return run[name]
    return run.get("timing", {}).get({
        "elapsed_wall_sec": "wall_seconds",
        "wall_seconds": "wall_seconds",
        "steady_time_sec": "calc_seconds",
        "calc_time_sec": "calc_seconds",
    }.get(name, name))


def _work(run, receipt):
    if run.get("completed_template_seconds") is not None:
        return float(run["completed_template_seconds"])
    raise ValueError("Each run must record completed_template_seconds")


def _resources(run):
    resources = run.get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError("Each run must record explicit resources")
    if "physical_cpu_cores" in resources:
        cores = resources["physical_cpu_cores"]
    elif "cpu_cores" in resources:
        cores = resources["cpu_cores"]
    else:
        raise ValueError("Each run must record physical CPU cores")
    if "gpus" in resources:
        gpus = resources["gpus"]
    elif "gpu_count" in resources:
        gpus = resources["gpu_count"]
    else:
        raise ValueError("Each run must record GPU allocation")
    host_cores = resources.get("allocated_host_cores", cores)
    try:
        cores, gpus, host_cores = (
            float(cores), float(gpus), float(host_cores))
    except (TypeError, ValueError):
        raise ValueError("Resource counts must be numeric") from None
    if (not np.isfinite(cores) or not np.isfinite(gpus) or
            not np.isfinite(host_cores) or cores <= 0 or gpus < 0 or
            host_cores <= 0 or cores != int(cores) or
            gpus != int(gpus) or host_cores != int(host_cores)):
        raise ValueError("Resource counts must be finite integers")
    return int(cores), int(gpus), int(host_cores)


def _finite_positive(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite and positive") from None
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _metric_summary(summaries, key):
    """Aggregate raw metric samples while omitting large sample arrays."""
    present = [summary.get("samples") is not None for summary in summaries]
    if any(present) and not all(present):
        raise ValueError(f"{key} samples are incomplete")
    samples = []
    for summary in summaries:
        values = summary.get("samples")
        if values is None:
            continue
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{key} samples must be a sequence")
        for value in values:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} samples must be finite") from None
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"{key} samples must be finite and non-negative")
            samples.append(value)
    if not samples:
        return None
    ordered = np.asarray(samples, dtype=float)
    return {
        "unit": summaries[0].get("unit", "seconds"),
        "count": int(ordered.size),
        "p50": float(np.percentile(ordered, 50)),
        "p95": float(np.percentile(ordered, 95)),
        "p99": float(np.percentile(ordered, 99)),
        "minimum": float(np.min(ordered)),
        "maximum": float(np.max(ordered)),
    }


def _aggregate_latency(runs, receipt_latency, arm):
    present = [run.get("latency") for run in runs]
    if any(value is not None for value in present) and not all(
            isinstance(value, dict) for value in present):
        raise ValueError("latency samples are incomplete")
    summaries = [value for value in present if isinstance(value, dict)]
    if not summaries:
        value = (receipt_latency.get(arm)
                 if isinstance(receipt_latency, dict) else None)
        if (value is None and isinstance(receipt_latency, dict)
                and receipt_latency):
            value = receipt_latency
        summaries = [value] if isinstance(value, dict) else []
    summaries = [summary for summary in summaries if isinstance(summary, dict)]
    if not summaries:
        return None
    latency = _metric_summary(summaries, "latency")
    if latency is None:
        # Legacy receipts may contain only already-aggregated percentiles. Keep
        # those explicit values, but never pretend multiple runs were pooled.
        if len(summaries) > 1:
            return {"per_run": [
                {key: value for key, value in summary.items()
                 if key != "samples"}
                for summary in summaries
            ]}
        return {key: value for key, value in summaries[0].items()
                if key != "samples"}
    backlog_summaries = [run["backlog"] for run in runs
                         if isinstance(run.get("backlog"), dict)]
    backlog = _metric_summary(backlog_summaries, "backlog")
    if backlog is not None:
        latency["backlog"] = backlog
    misses = [run.get("deadline_miss_count") for run in runs
              if run.get("deadline_miss_count") is not None]
    counts = [run.get("deadline_count") for run in runs
              if run.get("deadline_count") is not None]
    if misses:
        latency["deadline_miss_count"] = sum(int(value) for value in misses)
    if counts:
        latency["deadline_count"] = sum(int(value) for value in counts)
    return latency


def _science(runs, receipt, arm):
    """Aggregate per-run science status without hiding a failed run."""
    receipt_science = receipt.get("science", {})
    if not isinstance(receipt_science, dict):
        receipt_science = {}
    if arm not in receipt_science and "passed" in receipt_science:
        receipt_science = {arm: receipt_science}
    statuses = [run["science"] for run in runs
                if isinstance(run.get("science"), dict)]
    if (any("science" not in run for run in runs) and
            arm not in receipt_science):
        statuses.append({"passed": False, "missing_gates":
                         ["science status missing"]})
    if arm in receipt_science and isinstance(receipt_science[arm], dict):
        statuses.append(receipt_science[arm])
    qualification, qualification_field_present = _qualification_entries(
        receipt, arm)
    if qualification_field_present and not qualification:
        statuses.append({"passed": False, "missing_gates":
                         ["qualification result missing"]})
    if isinstance(qualification, dict):
        qualification = [qualification]
    if isinstance(qualification, (list, tuple)):
        for result in qualification:
            if not isinstance(result, dict):
                statuses.append({"passed": False, "missing_gates":
                                 ["invalid qualification result"]})
            elif isinstance(result.get("science"), dict):
                statuses.append(result["science"])
                if result.get("status") not in (None, "passed"):
                    statuses.append({"passed": False, "missing_gates":
                                     ["qualification failed"]})
            else:
                if result.get("status") not in (None, "passed"):
                    statuses.append({"passed": False, "missing_gates":
                                     ["qualification failed"]})
                elif arm not in receipt_science:
                    statuses.append({"passed": False, "missing_gates":
                                     ["qualification science status missing"]})
    if not statuses:
        return {}
    missing = sorted({gate for status in statuses
                      for gate in status.get("missing_gates", [])})
    scopes = sorted({status["scope"] for status in statuses
                     if status.get("scope")})
    return {
        "passed": all(status.get("passed") is True for status in statuses),
        "scope": ", ".join(scopes),
        "missing_gates": missing,
    }


def _qualification_entries(receipt, arm):
    """Return qualification records for an arm and whether the field exists."""
    field_present = ("qualification_results" in receipt or
                     "qualification" in receipt)
    if not field_present:
        return [], False
    qualification = receipt.get(
        "qualification_results", receipt.get("qualification"))
    if not isinstance(qualification, dict):
        return ([qualification] if qualification is not None else []), True
    if ("passed" in qualification or "science" in qualification or
            "status" in qualification):
        return [qualification], True
    if arm not in qualification:
        return [], True
    value = qualification[arm]
    if isinstance(value, (list, tuple)):
        return list(value), True
    return ([value] if value is not None else []), True


def _science_label(status):
    if not status:
        return "science unreported"
    if status.get("passed"):
        return "science passed"
    missing = status.get("missing_gates", [])
    return "science failed" + (f": {', '.join(missing)}" if missing else "")


def _resource_label(row):
    host_cores = row.get("allocated_host_cores", row["physical_cpu_cores"])
    if row["gpus"]:
        return (f"{row['gpus']} GPU + {host_cores} host core" +
                ("s" if host_cores != 1 else ""))
    return (f"{row['physical_cpu_cores']} physical CPU core" +
            ("s" if row["physical_cpu_cores"] != 1 else ""))


def _excluded_run(run):
    """Identify qualification or explicitly profiled runs."""
    utilization = run.get("utilization")
    profiled_samples = (isinstance(utilization, dict) and
                        bool(utilization.get("samples")))
    return bool(
        run.get("qualification_run") or run.get("profiled") or
        run.get("profile_run") or run.get("profiling_run") or
        run.get("profile_utilization") or profiled_samples)


def _qualification_arms(receipt):
    """Return arm names represented by a per-arm qualification mapping."""
    qualification = receipt.get(
        "qualification_results", receipt.get("qualification"))
    if not isinstance(qualification, dict):
        return set()
    if ("passed" in qualification or "science" in qualification or
            "status" in qualification):
        return set()
    return set(qualification)


def _not_timed_reason(receipt, arm, science):
    """Describe why an arm has no capacity measurement."""
    entries, _ = _qualification_entries(receipt, arm)
    reasons = []
    policy = receipt.get("timing_policy", {})
    skipped = (policy.get("skipped_arms", {})
               if isinstance(policy, dict) else {})
    skipped = skipped.get(arm, {}) if isinstance(skipped, dict) else {}
    if isinstance(skipped, dict) and skipped.get("reason"):
        reasons.append(str(skipped["reason"]))
    for entry in entries:
        if isinstance(entry, dict):
            reason = entry.get("skip_reason") or entry.get("reason")
            if reason:
                reasons.append(str(reason))
    label = _science_label(science)
    if label != "science unreported" and label != "science passed":
        reasons.insert(0, label)
    suffix = f" ({'; '.join(dict.fromkeys(reasons))})" if reasons else ""
    return "not timed" + suffix


def _skipped_row(receipt, arm, science):
    workload = receipt.get("workload", {})
    if not isinstance(workload, dict):
        workload = {}
    return {
        "arm": arm,
        "measured": False,
        "work": None,
        "templates": workload.get("templates"),
        "valid_detector_seconds": workload.get("valid_detector_seconds"),
        "wall_seconds": [],
        "steady_seconds": [],
        "wall_capacity": None,
        "steady_capacity": None,
        "physical_cpu_cores": None,
        "allocated_host_cores": None,
        "gpus": None,
        "resource_count": None,
        "resource_name": None,
        "resource_label": "n/a",
        "science": science,
        "latency": None,
        "skip_reason": _not_timed_reason(receipt, arm, science),
    }


def campaign_rows(receipt):
    """Return measured rows from a complete campaign receipt."""
    executable = receipt.get("executable")
    if executable not in ("pycbc_live", "pycbc_inspiral"):
        raise ValueError(
            "Receipt executable must be pycbc_live or pycbc_inspiral")
    raw = receipt.get("raw_results", receipt.get("runs"))
    if not isinstance(raw, dict):
        raise ValueError("Receipt must contain raw_results by arm")
    rows = []
    arms = list(raw)
    for arm in _qualification_arms(receipt):
        if arm not in raw:
            arms.append(arm)
    for arm in arms:
        runs = raw.get(arm, [])
        if isinstance(runs, dict):
            runs = list(runs.values())
        all_runs = list(runs)
        all_science = _science(all_runs, receipt, arm)
        runs = [run for run in runs if not _excluded_run(run)]
        if not runs:
            rows.append(_skipped_row(receipt, arm, all_science))
            continue
        receipt_latency = receipt.get("latency", {})
        if not isinstance(receipt_latency, dict):
            receipt_latency = {}
        if arm not in receipt_latency and receipt_latency:
            receipt_latency = {arm: receipt_latency}
        science = all_science
        latency = _aggregate_latency(runs, receipt_latency, arm)
        wall, steady, work, cores, gpus, host_cores = [], [], [], [], [], []
        for run in runs:
            if run.get("returncode") not in (None, 0):
                raise ValueError(f"{arm} run failed with returncode "
                                 f"{run['returncode']}")
            if run.get("status") in ("failed", "error"):
                raise ValueError(f"{arm} run has failed status")
            elapsed = _timing(run, "elapsed_wall_sec")
            if elapsed is None:
                raise ValueError(f"{arm} run has no elapsed wall time")
            steady_time = _timing(run, "steady_time_sec")
            if steady_time is None:
                steady_time = _timing(run, "calc_time_sec")
            wall.append(_finite_positive(elapsed, "elapsed wall time"))
            if steady_time is not None:
                steady.append(_finite_positive(steady_time, "steady time"))
            work.append(_finite_positive(
                _work(run, receipt), "completed work"))
            cpu, gpu, host = _resources(run)
            cores.append(cpu)
            gpus.append(gpu)
            host_cores.append(host)
        if steady and len(steady) != len(runs):
            raise ValueError(f"{arm} runs have incomplete steady samples")
        if len(set(work)) != 1:
            raise ValueError(f"{arm} runs do not have equal completed work")
        if (len(set(cores)) != 1 or len(set(gpus)) != 1 or
                len(set(host_cores)) != 1):
            raise ValueError(f"{arm} runs do not have stable resource counts")
        total_work = work[0]
        cpu_count, gpu_count, host_count = (cores[0], gpus[0], host_cores[0])
        resource_count = gpu_count if gpu_count else cpu_count
        resource_name = "GPU" if gpu_count else "physical CPU core"
        work_per_resource = total_work / resource_count
        rows.append({
            "arm": arm,
            "measured": True,
            "work": total_work,
            "templates": runs[0].get(
                "templates", receipt.get("workload", {}).get("templates")),
            "valid_detector_seconds": runs[0].get(
                "valid_detector_seconds",
                receipt.get("workload", {}).get("valid_detector_seconds")),
            "wall_seconds": wall,
            "steady_seconds": steady,
            "wall_capacity": capacity_summary(work_per_resource, wall),
            "steady_capacity": (
                capacity_summary(work_per_resource, steady)
                if steady else None),
            "physical_cpu_cores": cpu_count,
            "allocated_host_cores": host_count,
            "gpus": gpu_count,
            "resource_count": resource_count,
            "resource_name": resource_name,
            "resource_label": _resource_label({
                "gpus": gpu_count, "physical_cpu_cores": cpu_count,
                "allocated_host_cores": host_count}),
            "science": science,
            "latency": latency,
        })
    if not rows:
        raise ValueError("Receipt contains no runs")
    return rows


def render_campaign_report(receipt):
    """Render a concise text report from a complete campaign receipt."""
    rows = campaign_rows(receipt)
    lines = [f"{receipt['executable']} campaign",
             "arm | wall capacity | steady capacity | resources | science"]
    for row in rows:
        if not row["measured"]:
            lines.append(f"{row['arm']} | not timed | n/a | n/a | "
                         f"{row['skip_reason']}")
            continue
        wall = _summary_text(row["wall_capacity"])
        steady = (
            _summary_text(row["steady_capacity"])
            if row["steady_capacity"] else "n/a")
        status = _science_label(row["science"])
        lines.append(f"{row['arm']} | {wall} | {steady} | "
                     f"{row['resource_label']} | {status}")
        if row.get("latency"):
            latency = row["latency"]
            if "per_run" in latency:
                lines.append(
                    f"  latency: {len(latency['per_run'])} per-run summaries")
            else:
                values = " ".join(
                    f"{key}={value:.3g}"
                    for key in ("p50", "p95", "p99")
                    if (value := latency.get(
                        key, latency.get(f"{key}_seconds"))) is not None
                )
                misses = latency.get("deadline_miss_count")
                total = latency.get("deadline_count")
                if misses is not None:
                    values += f" misses={misses}"
                    if total is not None:
                        values += f"/{total}"
                backlog = latency.get("backlog")
                if backlog and backlog.get("p95") is not None:
                    values += f" backlog_p95={backlog['p95']:.3g}s"
                lines.append(f"  latency: {values}")
    return "\n".join(lines)


def _summary_text(summary):
    """Format median capacity and its measured minimum/maximum range."""
    median_value, minimum, maximum = summary
    return f"{median_value:.3g} [{minimum:.3g}, {maximum:.3g}]"


def plot_campaign(receipt, output):
    """Plot full-wall and steady capacity from a complete campaign receipt."""
    rows = campaign_rows(receipt)
    if not any(row["measured"] for row in rows):
        fig, ax = plt.subplots(figsize=(10, max(3.5, 1.0 + 0.6 * len(rows))))
        ax.axis("off")
        ax.text(0.02, 0.98, "No timed capacity measurements",
                va="top", ha="left", fontsize=12, weight="bold")
        ax.text(0.02, 0.88, "\n".join(
            f"{row['arm']}: {row['skip_reason']}" for row in rows),
                va="top", ha="left", fontsize=10)
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    positions = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.8))
    wall = np.array([row["wall_capacity"][0]
                     if row["measured"] else np.nan for row in rows])
    steady = np.array([
        row["steady_capacity"][0]
        if row["measured"] and row["steady_capacity"] else np.nan
        for row in rows
    ])
    wall_err_low = []
    wall_err_high = []
    for row in rows:
        if row["measured"]:
            mid, low, high = row["wall_capacity"]
            wall_err_low.append(mid - low)
            wall_err_high.append(high - mid)
        else:
            wall_err_low.append(np.nan)
            wall_err_high.append(np.nan)
    wall_err = np.array([wall_err_low, wall_err_high])
    steady_err = np.full((2, len(rows)), np.nan)
    for index, row in enumerate(rows):
        if row["measured"] and row["steady_capacity"]:
            mid, low, high = row["steady_capacity"]
            steady_err[:, index] = (mid - low, high - mid)
    failed = [_science_label(row["science"]) != "science passed"
              for row in rows]
    measured = [row for row in rows if row["measured"]]
    has_steady = any(row["steady_capacity"] for row in measured)
    values = [row["wall_capacity"][0] for row in measured]
    values.extend(row["steady_capacity"][0] for row in measured
                  if row["steady_capacity"])
    use_log = (len(measured) > 1 and max(values) / min(values) > 20)
    wall_positions = positions - width / 2 if has_steady else positions
    if use_log:
        ax.errorbar(wall_positions, wall, yerr=wall_err, fmt="o",
                    color="#0072B2", capsize=4, label="Full process wall")
        if has_steady:
            ax.errorbar(positions + width / 2, steady, yerr=steady_err,
                        fmt="o", color="#E69F00", capsize=4,
                        label="Steady interval")
    else:
        ax.bar(wall_positions, wall, width, label="Full process wall",
               color="#0072B2", yerr=wall_err, capsize=4,
               hatch=["//" if value else "" for value in failed])
        if has_steady:
            ax.bar(positions + width / 2, steady, width,
                   label="Steady interval", color="#E69F00",
                   yerr=steady_err, capsize=4,
                   hatch=["//" if value else "" for value in failed])
    labels = [(
        f"{row['arm']}\n{row['resource_label']}\n"
        f"{_science_label(row['science'])}"
        if row["measured"] else
        f"{row['arm']}\nnot timed\n{row['skip_reason']}"
    ) for row in rows]
    units = ({bool(row["gpus"]) for row in measured})
    ylabel = ("Templates/GPU at real time" if units == {True} else
              "Templates/core at real time" if units == {False} else
              "Real-time templates per allocated resource")
    ax.set(xticks=positions, xticklabels=labels, xlim=(-0.6, len(rows) - 0.4),
           ylabel=ylabel,
           title=f"{receipt['executable']}: complete campaign capacity")
    if use_log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    caption = ("Points show medians; whiskers span measured min/max. "
               "Science status is shown below each arm." if use_log else
               "Bars show medians; whiskers span measured min/max. Hatched bars "
               "have failed or missing science gates.")
    fig.text(0.5, 0.02, caption,
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_inspiral(data, output):
    """Backward-compatible renderer for the historical comparison receipt."""
    if "raw_results" in data or data.get("executable"):
        return plot_campaign(data, output)
    work = 384 * 1904
    arms = ("branch_cpu", "jax_cpu_batched", "jax_cuda_batched")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    x = np.arange(len(arms))
    for offset, field, label, color in (
        (-0.18, "wall_seconds", "Full executable wall time", "#0072B2"),
        (0.18, "calc_seconds", "Calculation interval", "#E69F00"),
    ):
        summaries = []
        for arm in arms:
            samples = [
                run["timing"][field] for name, run in data["runs"].items()
                if name.startswith(arm + "_rep")
            ]
            summaries.append(capacity_summary(work, samples))
        mid, low, high = np.array(summaries).T
        ax.bar(x + offset, mid, 0.36, label=label, color=color,
               yerr=[mid - low, high - mid], capsize=4)
    ax.set(yscale="log", xticks=x, xticklabels=arms,
           ylabel="Finite-workload real-time templates per resource",
           title="Historical pycbc_inspiral comparison")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    static = Path("docs/_static")
    parser.add_argument("--campaign", type=Path,
                        help="complete pycbc_live or pycbc_inspiral receipt")
    parser.add_argument("--output", type=Path,
                        help="capacity plot path (required with --campaign)")
    parser.add_argument("--report", action="store_true",
                        help="print the campaign capacity report")
    parser.add_argument("--search", type=Path, default=None,
                        help="historical inspiral receipt")
    parser.add_argument("--microbenchmark", type=Path,
                        help="explicitly render retained filter diagnostic")
    parser.add_argument("--output-dir", type=Path, default=static)
    parser.add_argument("--live-advance-seconds", type=float, default=56.0)
    args = parser.parse_args()
    if args.campaign:
        receipt = json.loads(args.campaign.read_text())
        if args.report:
            print(render_campaign_report(receipt))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            plot_campaign(receipt, args.output)
        elif not args.report:
            parser.error("--output is required when rendering --campaign")
        return
    if args.microbenchmark:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_live(json.loads(args.microbenchmark.read_text()),
                  args.output_dir / "jax_live_capacity_diagnostic.png",
                  args.live_advance_seconds)
    if args.search:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_inspiral(json.loads(args.search.read_text()),
                      args.output_dir / "jax_inspiral_capacity.png")
    if not args.microbenchmark and not args.search:
        parser.error("specify --campaign, --search, or --microbenchmark")


if __name__ == "__main__":
    main()
