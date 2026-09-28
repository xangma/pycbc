#!/usr/bin/env python3
"""Run reproducible, full-executable ``pycbc_live`` campaigns.

The runner deliberately takes a complete live configuration from JSON (or an
argument file).  This keeps the campaign tied to the real bank, frame cache,
channels, and live thresholds instead of silently benchmarking a tiny fixture.
It supports an unpaced cached replay for throughput and a paced historical
replay using ``pycbc_live --replay-clock --replay-rate``.  ``cpu`` is the
pristine upstream reference (run through the observer); ``branch_cpu`` is the
candidate checkout's ordinary CPU path.  The latter is required to detect
changes to the CPU implementation while evaluating JAX arms.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import glob
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    import h5py
except ImportError:  # pragma: no cover - the executable itself requires h5py
    h5py = None

try:
    import psutil
except ImportError:  # pragma: no cover - optional host telemetry
    psutil = None

try:
    from tools.benchmark_artifact import (
        compilation_audit,
        file_sha256,
        percentile as _artifact_percentile,
        source_identity,
    )
    from tools.benchmark_reference import validate_reference
    from tools.observe_pycbc_live import validate_source
except ImportError:  # script execution from the tools directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_artifact import (
        compilation_audit,
        file_sha256,
        percentile as _artifact_percentile,
        source_identity,
    )
    from benchmark_reference import validate_reference
    from observe_pycbc_live import validate_source


SCHEMA_VERSION = 1
BENCHMARK_SAMPLE_RATE = 2048.0
BENCHMARK_PRECISION = "complex64"
ARM_SCHEMES = {
    "cpu": "cpu:1",
    "branch_cpu": "cpu:1",
    "jax_cpu": "jax:cpu",
    "jax_cuda": "jax:cuda:0",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _input_contract(config: Mapping[str, Any]) -> dict:
    """Describe inputs and physics which define a comparable workload.

    Launch resources and replay controls deliberately do not enter this
    contract; they are recorded separately in the receipt.
    """
    args = config_args(config)
    bank = config.get("bank_file") or option_value(args, "--bank-file")
    frames = _frame_input_paths(config)
    hashes = {str(path): file_sha256(path) for path in frames if path.exists()}
    physics = normalized_science_args(config)
    return {
        "bank_file": str(Path(bank).resolve()) if bank else None,
        "bank_sha256": (file_sha256(Path(bank)) if bank and Path(bank).is_file()
                        else None),
        "frame_files": hashes,
        "science_args": physics,
        "workload": workload_from_config(config),
    }


def workload_digest(config: Mapping[str, Any]) -> dict:
    contract = _input_contract(config)
    return {"sha256": hashlib.sha256(_canonical_json(contract).encode()).hexdigest(),
            "contract": contract}


def validate_input_contract(config: Mapping[str, Any]) -> None:
    """Reject a real launch without explicit bank, frame, and interval inputs."""
    args = config_args(config)
    bank = config.get("bank_file") or option_value(args, "--bank-file")
    cli_bank = option_value(args, "--bank-file")
    if not cli_bank:
        raise ValueError("live benchmark requires an explicit CLI bank-file input contract")
    if not Path(cli_bank).is_absolute():
        raise ValueError("bank-file must be absolute because each process uses its own output directory")
    if config.get("bank_file") and cli_bank:
        if Path(config["bank_file"]).resolve() != Path(cli_bank).resolve():
            raise ValueError("declared bank-file disagrees with CLI bank-file")
    declared_frames = _declared_frame_paths(config)
    cli_frames = _cli_frame_paths(args)
    if not cli_frames:
        raise ValueError("live benchmark requires explicit CLI frame-src inputs")
    if declared_frames and cli_frames and declared_frames != cli_frames:
        raise ValueError("declared frame inputs disagree with CLI frame-src")
    declared_inputs = declared_frames | cli_frames
    if declared_inputs and not all(path.is_file() for path in declared_inputs):
        raise ValueError("frame input contract names a missing file")
    frames = _frame_input_paths(config)
    workload = config.get("workload", {})
    if not bank:
        raise ValueError("live benchmark requires an explicit bank-file input contract")
    bank_path = Path(bank)
    if not bank_path.is_file():
        raise ValueError(f"bank-file input does not exist: {bank_path}")
    if not frames or not all(path.is_file() for path in frames):
        raise ValueError("live benchmark requires an explicit frame-file input contract")
    if not isinstance(workload, Mapping) or not all(workload.get(key) is not None for key in
                                                    ("templates", "analysis_chunk_sec", "start_time", "end_time")):
        raise ValueError("live benchmark workload must declare templates and valid interval")
    cli_chunk = option_value(args, "--analysis-chunk")
    cli_start = option_value(args, "--start-time")
    cli_end = option_value(args, "--end-time")
    if cli_chunk is None or cli_start is None or cli_end is None:
        raise ValueError("live benchmark requires analysis-chunk, start-time, and end-time CLI inputs")
    try:
        if float(workload["analysis_chunk_sec"]) != float(cli_chunk):
            raise ValueError("workload analysis chunk disagrees with CLI")
        if str(workload["start_time"]) != str(cli_start) or str(workload["end_time"]) != str(cli_end):
            raise ValueError("workload interval disagrees with CLI")
        templates = int(workload["templates"])
        start, end, chunk = float(cli_start), float(cli_end), float(cli_chunk)
        if (not all(math.isfinite(v) for v in (start, end, chunk)) or
                end <= start or chunk <= 0 or templates <= 0 or
                float(workload['templates']) != templates):
            raise ValueError('invalid interval or template count')
    except (TypeError, ValueError) as exc:
        raise ValueError("workload metadata must match numeric CLI inputs") from exc
    inventory = _bank_templates(bank_path)
    if inventory is None or templates != inventory:
        raise ValueError("workload template count must match the bank")
    if not _bank_has_distinct_physical_rows(bank_path):
        raise ValueError("bank template rows must have distinct physical mass/spin parameters")


def _pristine_source(identity: Mapping[str, Any] | None) -> bool:
    """Whether a source identity names a reproducible clean git checkout."""
    return bool(isinstance(identity, Mapping) and
                identity.get("revision") and identity.get("dirty") is False)


def _invalidate_science(results: Mapping[str, list], reason: str) -> None:
    """Prevent diagnostic outputs from being presented as qualified science."""
    for values in results.values():
        for result in values:
            if not isinstance(result, Mapping):
                continue
            original = result.get("science")
            if not isinstance(original, Mapping):
                continue
            diagnostic = original.get("diagnostic_science", original)
            missing = set(original.get("missing_gates", []))
            if isinstance(diagnostic, Mapping):
                missing.update(diagnostic.get("missing_gates", []))
            result["science"] = {
                "passed": False,
                "missing_gates": sorted(missing | {reason}),
                "diagnostic_science": dict(diagnostic),
            }


def _science_summary(arms: Sequence[str], qualification: Mapping[str, list],
                     selected: Mapping[str, list]) -> dict[str, dict]:
    """Summarize qualification and the receipt's selected science runs per arm."""
    summary = {}
    for arm in arms:
        qualification_runs = qualification.get(arm, [])
        selected_runs = selected.get(arm, [])
        details = {
            "qualification": [dict(run.get("science", {})) for run in qualification_runs
                              if isinstance(run, Mapping)],
            "selected": [dict(run.get("science", {})) for run in selected_runs
                          if isinstance(run, Mapping)],
        }
        missing = set()
        all_passed = True
        for label, runs in details.items():
            if not runs:
                missing.add(f"{label} result unavailable")
                all_passed = False
                continue
            for science in runs:
                if science.get("passed") is not True:
                    all_passed = False
                    missing.update(science.get("missing_gates", []))
        summary[arm] = {
            "passed": bool(all_passed),
            "missing_gates": sorted(missing),
            **details,
        }
    return summary


def arm_order_for_replicate(arms: Sequence[str], replicate: int) -> list[str]:
    """Return counterbalanced arm order for zero-based replicate index."""
    return list(arms) if replicate % 2 == 0 else list(reversed(arms))


def percentile(values: Iterable[float], fraction: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return _artifact_percentile(ordered, fraction)


def latency_summary(values: Iterable[float], unit: str = "seconds") -> dict:
    samples = [float(value) for value in values]
    return {
        "unit": unit,
        "count": len(samples),
        "samples": samples,
        "p50": percentile(samples, 0.50),
        "p95": percentile(samples, 0.95),
        "p99": percentile(samples, 0.99),
        "minimum": min(samples) if samples else None,
        "maximum": max(samples) if samples else None,
    }


def parse_stage_events(text: str) -> List[dict]:
    """Parse ``PYCBC_STAGE`` records from interleaved MPI logs."""
    events = []
    for line in text.splitlines():
        marker = "PYCBC_STAGE_EVENT "
        if marker not in line:
            marker = "PYCBC_STAGE "
        if marker not in line:
            continue
        try:
            payload = json.loads(line.split(marker, 1)[1].strip())
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        if "stage" in payload and "name" not in payload:
            payload["name"] = payload["stage"]
        if "monotonic_ns" in payload and "time_ns" not in payload:
            payload["time_ns"] = payload["monotonic_ns"]
        if {"name", "event", "time_ns"} <= payload.keys():
            events.append(payload)
    return events


def block_metrics(events: Sequence[Mapping[str, Any]], analysis_chunk: float,
                  deadline_budget: float | None = None) -> dict:
    """Pair block stage events and derive latency/deadline/backlog metrics."""
    service_budget = (analysis_chunk if deadline_budget is None
                      else float(deadline_budget))
    # Every MPI rank emits the outer block marker.  The root owns completion,
    # output, and detector validity, so use that stream for process-level
    # latency and avoid multiplying work by the number of filtering ranks.
    root_events = [event for event in events if event.get("rank") == 0]
    # Worker markers cannot establish process-level completion.  Do not fall
    # back to them if the root stream is absent or truncated.
    events = root_events
    starts: Dict[tuple, Mapping[str, Any]] = {}
    latencies = []
    wall_latencies = []
    deadlines = []
    data_intervals = []
    block_starts = []
    replay_lags = []
    invalid_block_count = 0
    unpaired_end_count = 0
    for event in events:
        if event.get("name") != "block":
            continue
        key = (event.get("pid"), event.get("rank"), event.get("data_start"))
        if event.get("event") == "start":
            if key in starts:
                unpaired_end_count += 1
            starts[key] = event
            continue
        if event.get("event") != "end":
            continue
        if key not in starts:
            unpaired_end_count += 1
            continue
        start = starts.pop(key)
        try:
            elapsed = event.get("processing_elapsed_sec", event.get("elapsed_sec"))
            if elapsed is None:
                elapsed = (float(event["time_ns"]) - float(start["time_ns"])) / 1e9
            wall_elapsed = event.get("elapsed_sec", elapsed)
            start_data = float(start.get("data_start"))
            end_data = float(event.get("data_end"))
            elapsed = float(elapsed)
            wall_elapsed = float(wall_elapsed)
        except (KeyError, TypeError, ValueError):
            invalid_block_count += 1
            continue
        if not (math.isfinite(start_data) and math.isfinite(end_data) and
                math.isfinite(elapsed) and math.isfinite(wall_elapsed) and
                end_data >= start_data and elapsed >= 0 and wall_elapsed >= 0):
            invalid_block_count += 1
            continue
        block_starts.append(start_data)
        latencies.append(float(elapsed))
        wall_latencies.append(float(wall_elapsed))
        deadlines.append(bool(event.get("deadline_met", float(elapsed) <= service_budget)))
        if event.get("lag_sec") is not None:
            replay_lags.append(float(event["lag_sec"]))
        if event.get("data_end") is not None:
            data_intervals.append({
                "start": start_data,
                "end": end_data,
                "duration": end_data - start_data,
                "live_detectors": event.get("live_detectors", []),
            })
    # A queue backlog carries over between blocks.  Keep local processing
    # overrun separate so a fast block can repay an earlier miss.
    backlog = []
    overrun = []
    queued = 0.0
    for value in latencies:
        local_overrun = max(0.0, value - service_budget)
        overrun.append(local_overrun)
        queued = max(0.0, queued + value - service_budget)
        backlog.append(queued)
    return {
        "latency": latency_summary(latencies),
        "wall_latency": latency_summary(wall_latencies),
        "backlog": latency_summary(backlog),
        "backlog_method": "derived_from_processing_latency",
        "replay_lag": latency_summary(replay_lags),
        "processing_overrun": latency_summary(overrun),
        "deadline_count": len(deadlines),
        "deadlines_met": sum(deadlines),
        "deadline_miss_count": len(deadlines) - sum(deadlines),
        "valid_detector_intervals": data_intervals,
        "block_starts": block_starts,
        "block_count": len(block_starts),
        "unpaired_block_count": len(starts) + unpaired_end_count,
        "invalid_block_count": invalid_block_count,
        "root_block_events": bool(root_events),
    }


def stage_metrics(events: Sequence[Mapping[str, Any]]) -> dict:
    """Summarize instrumented stages, naming frame reads truthfully."""
    starts = {}
    values = {}
    for event in events:
        name = event.get("name")
        key = (event.get("pid"), event.get("rank"), name,
               event.get("ifo"), event.get("data_start"))
        if event.get("event") == "start":
            starts[key] = event
        elif event.get("event") == "end" and key in starts:
            start = starts.pop(key)
            try:
                elapsed = (float(event.get("time_ns", 0)) -
                           float(start.get("time_ns", 0))) / 1e9
            except (TypeError, ValueError):
                continue
            label = "read_and_condition" if name == "frame_read" else str(name)
            values.setdefault(label, []).append(max(0.0, elapsed))
    return {name: latency_summary(samples) for name, samples in values.items()}


def observed_filter_contract(events: Sequence[Mapping[str, Any]]) -> dict:
    """Validate measured filter dtype/rate markers, rather than declarations."""
    ends = [event for event in events
            if event.get("name") == "filter" and event.get("event") == "end"]
    rates = sorted({event.get("sample_rate") for event in ends}, key=repr)
    dtypes = set()
    for event in ends:
        value = event.get("signal_dtypes", [])
        dtypes.add(tuple([value] if isinstance(value, str) else value))
    dtypes = sorted(dtypes)
    valid = bool(ends) and all(rate == BENCHMARK_SAMPLE_RATE for rate in rates) and \
        dtypes == [("complex64",)]
    return {"passed": valid, "count": len(ends), "sample_rates": rates,
            "signal_dtypes": [list(dtype) for dtype in dtypes]}


def _valid_filter_contract(contract: Any) -> bool:
    return bool(isinstance(contract, Mapping) and
                contract.get("passed") is True and
                isinstance(contract.get("count"), int) and contract["count"] > 0 and
                contract.get("sample_rates") == [BENCHMARK_SAMPLE_RATE] and
                contract.get("signal_dtypes") == [[BENCHMARK_PRECISION]])


def _matches_workload_digest(digest: Any, expected: str) -> bool:
    return bool(isinstance(digest, Mapping) and
                isinstance(digest.get("contract"), Mapping) and
                digest.get("sha256") == expected and
                hashlib.sha256(_canonical_json(digest["contract"]).encode()).hexdigest()
                == expected)


def _affinity_cpus(affinity: str | None) -> list[int]:
    if affinity:
        cpus = set()
        for item in affinity.split(','):
            bounds = item.strip().split('-')
            if len(bounds) == 1:
                cpus.add(int(bounds[0]))
            else:
                cpus.update(range(int(bounds[0]), int(bounds[-1]) + 1))
        if cpus:
            return sorted(cpus)
    if psutil is not None:
        try:
            return list(psutil.Process().cpu_affinity())
        except (AttributeError, psutil.Error, OSError):
            pass
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def physical_core_count(affinity: str | None = None) -> Optional[int]:
    """Count physical cores in the launch affinity, counting SMT once."""
    cpus = _affinity_cpus(affinity)
    identities = set()
    for cpu in cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = topology / "physical_package_id"
        core = topology / "core_id"
        if package.exists() and core.exists():
            identities.add((package.read_text().strip(), core.read_text().strip()))
    if identities:
        return len(identities)
    if psutil is not None:
        return psutil.cpu_count(logical=False)
    return len(cpus) or os.cpu_count()


def host_snapshot() -> dict:
    snapshot = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": physical_core_count(),
    }
    if psutil is not None:
        snapshot["load_average"] = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
        snapshot["memory_bytes"] = psutil.virtual_memory().total
        try:
            snapshot["controller_affinity"] = psutil.Process().cpu_affinity()
        except (AttributeError, psutil.Error, OSError):
            snapshot["controller_affinity"] = None
    return snapshot


def _remove_option(args: List[str], names: Sequence[str],
                   boolean_names: Sequence[str] = ()) -> List[str]:
    result = []
    skip = False
    names = set(names)
    boolean_names = set(boolean_names)
    for token in args:
        if skip:
            skip = False
            continue
        if token in names:
            skip = token not in boolean_names
            continue
        if any(token.startswith(name + "=") for name in names):
            continue
        result.append(token)
    return result


def config_args(config: Mapping[str, Any]) -> List[str]:
    value = config.get("args", config.get("live_args", config.get("command")))
    if value is None:
        raise ValueError("configuration must contain args, live_args, or command")
    if isinstance(value, str):
        return shlex.split(value)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("live args must be a shell string or list of strings")
    return list(value)


def normalized_science_args(config: Mapping[str, Any], scheme_override: str | None = None) -> List[str]:
    """Return physics/configuration arguments without runner-owned controls."""
    args = _remove_option(config_args(config), (
        "--processing-scheme", "--output-path", "--replay-clock",
        "--replay-rate", "--benchmark-evidence-path", "--output-status",
        "--jax-chisq-mode",
    ), boolean_names=("--replay-clock",))
    scheme = scheme_override or option_value(config_args(config), "--processing-scheme")
    mode = option_value(config_args(config), "--jax-chisq-mode")
    if mode is not None and mode not in ("cpu-compatible", "direct-phase"):
        raise ValueError("--jax-chisq-mode must be cpu-compatible or direct-phase")
    if scheme and scheme.split(':', 1)[0] == "jax":
        args.extend(["--jax-chisq-mode", mode or "cpu-compatible"])
    elif mode is not None and scheme_override is None:
        raise ValueError("--jax-chisq-mode requires a JAX processing scheme")
    return args


def science_config_for_arm(config: Mapping[str, Any], arm: str,
                           workload_digest: Mapping[str, Any]) -> dict:
    """Build per-arm evidence config with mode declared separately."""
    args = normalized_science_args(config, scheme_override=ARM_SCHEMES[arm])
    mode_free_args = _remove_option(args, ("--jax-chisq-mode",))
    result = {
        "args": mode_free_args,
        "config": config.get("science", {}),
        "workload_digest": workload_digest,
    }
    if ARM_SCHEMES[arm].startswith("jax"):
        result["--jax-chisq-mode"] = option_value(
            args, "--jax-chisq-mode") or "cpu-compatible"
    return result


def option_value(args: Sequence[str], *names: str) -> Optional[str]:
    """Return a scalar option from either ``--name value`` spelling."""
    names = set(names)
    for index, token in enumerate(args):
        if token in names and index + 1 < len(args):
            return args[index + 1]
        for name in names:
            if token.startswith(name + "="):
                return token[len(name) + 1:]
    return None


def validate_benchmark_config(config: Mapping[str, Any]) -> None:
    """Reject live configurations that cannot represent this benchmark.

    The campaign is deliberately fixed at 2048 Hz and complex64.  ``pycbc_live``
    has no precision command-line switch, so precision is an explicit science
    declaration and is checked against the executable's complex64 path by the
    campaign outputs.  Silently substituting a default would make an old 4096
    Hz receipt look comparable to the fixed workload.
    """
    args = config_args(config)
    raw_rate = option_value(args, "--sample-rate")
    science = config.get("science", {})
    if not isinstance(science, Mapping):
        raise ValueError("science must declare sample_rate and precision")
    declared_rate = science.get("sample_rate")
    try:
        rates = (float(raw_rate), float(declared_rate))
    except (TypeError, ValueError):
        raise ValueError("live benchmark requires --sample-rate and science.sample_rate")
    if any(rate != BENCHMARK_SAMPLE_RATE for rate in rates):
        raise ValueError("live benchmark requires a 2048 Hz sample rate")
    precision = str(science.get("precision", ""))
    if precision != BENCHMARK_PRECISION:
        raise ValueError("live benchmark requires science.precision=complex64")
    if science.get('reference_arm', 'cpu') != 'cpu':
        raise ValueError('the scientific reference arm must be cpu')
    if option_value(args, "--precision", "--dtype") is not None:
        raise ValueError("pycbc_live has no supported precision command-line option")


def build_live_command(
    config: Mapping[str, Any],
    arm: str,
    output_dir: Path,
    source_root: Path,
    python_bin: str,
    ranks: int,
    replay_mode: str,
    replay_rate: float,
    affinity: str | None = None,
    observe_pristine_cpu: bool = False,
) -> List[str]:
    """Build the actual MPI command while preserving the supplied campaign."""
    original_args = config_args(config)
    args = _remove_option(original_args, (
        "--processing-scheme", "--output-path", "--replay-clock", "--replay-rate",
        "--benchmark-evidence-path", "--jax-chisq-mode",
    ), boolean_names=("--replay-clock",))
    args.extend(["--processing-scheme", ARM_SCHEMES[arm], "--output-path", str(output_dir)])
    if ARM_SCHEMES[arm].startswith("jax"):
        mode = option_value(original_args, "--jax-chisq-mode") or "cpu-compatible"
        if mode not in ("cpu-compatible", "direct-phase"):
            raise ValueError("--jax-chisq-mode must be cpu-compatible or direct-phase")
        args.extend(["--jax-chisq-mode", mode])
    science = config.get("science", {})
    if not observe_pristine_cpu:
        if isinstance(science, Mapping) and science.get("capture_evidence"):
            args.extend(["--benchmark-evidence-path", str(output_dir / "evidence")])
        if replay_mode in {"unpaced", "paced"}:
            args.append("--replay-clock")
        if replay_mode == "paced":
            if replay_rate <= 0:
                raise ValueError("paced replay requires a positive replay rate")
            args.extend(["--replay-rate", str(replay_rate)])
    executable = source_root / "bin" / "pycbc_live"
    launcher = os.environ.get("MPIEXEC", "mpirun")
    # Keep MPI workers on distinct physical cores when the launcher supports
    # Open MPI's standard binding option.  The receipt still records the host
    # count separately from GPU capacity.
    command = [launcher, "-n", str(ranks), "--bind-to", "core"]
    if affinity:
        command.extend(["--cpu-set", affinity])
    if observe_pristine_cpu:
        observer = Path(__file__).resolve().with_name("observe_pycbc_live.py")
        command.extend([python_bin, "-m", "mpi4py", str(observer), str(executable)])
    else:
        command.extend([python_bin, "-m", "mpi4py", str(executable)])
    return command + args


def _bank_templates(bank_file: Path) -> int | None:
    if h5py is None or not bank_file.exists():
        return None
    with h5py.File(bank_file, "r") as handle:
        for name in (" mass1", "mass1", "mass1_spin1z", "template_duration"):
            if name in handle:
                return len(handle[name])
        for group_name in ("/tmpltbank", "/templates", "/template_bank"):
            group = handle.get(group_name)
            if group is None:
                continue
            if hasattr(group, "shape"):
                return int(group.shape[0])
            for name in ("mass1", "mass2", "template_duration", "f_lower"):
                if name in group and hasattr(group[name], "shape"):
                    return int(group[name].shape[0])
    return None


def _bank_has_distinct_physical_rows(bank_file: Path) -> bool:
    """Require finite, distinct physical mass/spin rows in the native bank."""
    if h5py is None or not bank_file.is_file():
        return False
    with h5py.File(bank_file, "r") as handle:
        if not all(name in handle for name in ('mass1', 'mass2')):
            return False
        datasets = {}
        for name in ('mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z',
                     'spin2x', 'spin2y', 'spin2z'):
            if name not in handle:
                continue
            column = handle[name]
            if (not isinstance(column, h5py.Dataset) or len(column.shape) != 1
                    or column.dtype.kind not in 'fiu'):
                return False
            values = column[...].tolist()
            if not all(math.isfinite(value) and (value > 0 if name.startswith('mass') else True)
                       for value in values):
                return False
            datasets[name] = values
        count = len(next(iter(datasets.values())))
        if not count or any(len(values) != count for values in datasets.values()):
            return False
        rows = set(zip(*(tuple(values) for values in datasets.values())))
        return len(rows) == count


def _expand_frame_tokens(values: Iterable[str]) -> set[Path]:
    paths = set()
    for value in values:
        for name in glob.glob(str(value)) or [str(value)]:
            paths.add(Path(name).resolve())
    return paths


def _declared_frame_paths(config: Mapping[str, Any]) -> set[Path]:
    values = []
    for key in ("frame_files", "frame_src", "input_files"):
        value = config.get(key, [])
        values.extend(value if isinstance(value, list) else [value])
    return _expand_frame_tokens(str(value) for value in values if value)


def _cli_frame_paths(args: Sequence[str]) -> set[Path]:
    values = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--frame-src":
            index += 1
            while index < len(args) and not args[index].startswith("--"):
                values.append(args[index].split(":", 1)[-1])
                index += 1
            continue
        if token.startswith("--frame-src="):
            values.append(token.split("=", 1)[1].split(":", 1)[-1])
        index += 1
    return _expand_frame_tokens(values)


def workload_from_config(config: Mapping[str, Any]) -> dict:
    workload = dict(config.get("workload", {}))
    args = config_args(config)
    templates = workload.get("templates")
    bank_file = config.get("bank_file") or option_value(args, "--bank-file")
    inventory = _bank_templates(Path(bank_file)) if bank_file else None
    if templates is None:
        templates = inventory
    workload["templates"] = int(templates) if templates is not None else None
    workload["bank_template_count"] = inventory
    workload["distinct_templates"] = (
        inventory if bank_file and _bank_has_distinct_physical_rows(Path(bank_file))
        else None)
    workload["template_count_matches_bank"] = (
        inventory is None or templates is None or int(templates) == int(inventory)
    )
    chunk = config.get("analysis_chunk", option_value(args, "--analysis-chunk"))
    workload.setdefault("analysis_chunk_sec", float(chunk or 0.0))
    workload.setdefault("start_time", config.get(
        "start_time", option_value(args, "--start-time")))
    workload.setdefault("end_time", config.get(
        "end_time", option_value(args, "--end-time")))
    if bank_file:
        workload.setdefault("bank_file", str(bank_file))
    return workload


def _collect_utilization(process: subprocess.Popen, samples: list, observed_pids: set,
                         stop: list) -> None:
    """Sample one process tree without recreating handles or zeroing CPU data."""
    if psutil is None:
        return
    try:
        root = psutil.Process(process.pid)
        handles = {root.pid: root}
        for handle in handles.values():
            handle.cpu_percent(None)  # establish the interval baseline
    except (psutil.Error, OSError):
        return
    while not stop[0]:
        stamp = time.monotonic()
        item = {"time_sec": stamp}
        try:
            current = [root] + root.children(recursive=True)
            for handle in current:
                handles.setdefault(handle.pid, handle)
            observed_pids.update(handles)
            running = [handle for handle in handles.values()
                       if handle.is_running()]
            item["cpu_percent"] = sum(handle.cpu_percent(None) for handle in running)
            item["rss_bytes"] = sum(handle.memory_info().rss for handle in running)
        except (AttributeError, psutil.Error, OSError):
            pass
        samples.append(item)
        time.sleep(0.1)


def _discover_hdf_outputs(output_dir: Path) -> tuple[list[str], list[str]]:
    """Find block and evidence HDFs, including date-prefixed output trees."""
    trigger_outputs = sorted(
        str(path) for path in output_dir.rglob("*.hdf")
        if "evidence" not in path.relative_to(output_dir).parts and
        not path.name.startswith("candidate_") and
        "LIVE_BACKGROUND" not in path.name)
    evidence_outputs = sorted(str(path) for path in
                             (output_dir / "evidence").rglob("*.hdf"))
    return trigger_outputs, evidence_outputs


def run_one(
    config: Mapping[str, Any],
    arm: str,
    output_dir: Path,
    source_root: Path,
    python_bin: str,
    ranks: int,
    replay_mode: str,
    replay_rate: float,
    physical_cores: int,
    gpus: int,
    profile_utilization: bool = False,
    affinity: str | None = None,
    observe_pristine_cpu: bool = False,
    expected_digest: str | None = None,
    jax_cache_dir: Path | None = None,
    require_cache_hits: bool = False,
) -> dict:
    validate_benchmark_config(config)
    validate_input_contract(config)
    output_dir = output_dir.resolve()
    source_root = source_root.resolve()
    frozen_workload = workload_digest(config)
    if expected_digest is not None and frozen_workload['sha256'] != expected_digest:
        raise ValueError('workload input contract changed before launch')
    if not (source_root / "bin" / "pycbc_live").is_file():
        raise ValueError(f"live executable does not exist: {source_root / 'bin' / 'pycbc_live'}")
    if observe_pristine_cpu:
        validate_source(source_root / "bin" / "pycbc_live")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_live_command(config, arm, output_dir, source_root, python_bin,
                                 ranks, replay_mode, replay_rate, affinity,
                                 observe_pristine_cpu)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    timeline_path = output_dir / "timeline.json"
    is_jax = ARM_SCHEMES[arm].startswith('jax')
    if is_jax:
        if jax_cache_dir is None:
            raise ValueError('JAX benchmark arm requires a persistent cache directory')
        jax_cache_dir = Path(jax_cache_dir).resolve()
        jax_cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ('PYCBC_OBSERVER_EVIDENCE', 'PYCBC_OBSERVER_REPLAY_MODE',
                'PYCBC_OBSERVER_REPLAY_RATE'):
        env.pop(key, None)
    science_config = science_config_for_arm(config, arm, frozen_workload)
    env.update({
        "PYTHONPATH": str(source_root),
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MKL_DYNAMIC": "FALSE",
        "MKL_THREADING_LAYER": "GNU",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
        "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
        "PYCBC_REPLAY_CLOCK": "1" if replay_mode in {"unpaced", "paced"} else "0",
        "PYCBC_BENCHMARK_STAGES": "1",
        "PYCBC_SCIENCE_CONFIG": json.dumps(science_config, sort_keys=True),
    })
    if is_jax:
        env.update({
            "JAX_ENABLE_COMPILATION_CACHE": "true",
            "JAX_COMPILATION_CACHE_DIR": str(jax_cache_dir),
            "PYCBC_JAX_COMPILATION_AUDIT_DIR": str(
                output_dir / "jax-compilation-audit"),
        })
    else:
        for key in ("JAX_ENABLE_COMPILATION_CACHE", "JAX_COMPILATION_CACHE_DIR",
                    "JAX_RAISE_PERSISTENT_CACHE_ERRORS",
                    "PYCBC_JAX_COMPILATION_AUDIT_DIR"):
            env.pop(key, None)
    if observe_pristine_cpu:
        env.update({
            "PYCBC_OBSERVER_REPLAY_MODE": replay_mode,
            "PYCBC_OBSERVER_REPLAY_RATE": str(replay_rate),
        })
        if config.get("science", {}).get("capture_evidence"):
            env["PYCBC_OBSERVER_EVIDENCE"] = str(output_dir / "evidence")
    start = time.perf_counter()
    samples: list = []
    observed_pids = set()
    if profile_utilization:
        env["PYCBC_BENCHMARK_SYNCHRONIZE_STAGES"] = "1"
        try:
            from tools.profile_jax_gpu_timeline import run_profiling_campaign
        except ModuleNotFoundError:  # direct invocation from tools/
            from profile_jax_gpu_timeline import run_profiling_campaign
        profile_environment = {key: env[key] for key in (
            "PYTHONPATH", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "MKL_DYNAMIC", "MKL_THREADING_LAYER", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
            "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_ENABLE_COMPILATION_CACHE",
            "JAX_COMPILATION_CACHE_DIR",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
            "JAX_RAISE_PERSISTENT_CACHE_ERRORS",
            "PYCBC_JAX_COMPILATION_AUDIT_DIR", "PYCBC_REPLAY_CLOCK",
            "PYCBC_BENCHMARK_STAGES", "PYCBC_BENCHMARK_SYNCHRONIZE_STAGES",
            "PYCBC_SCIENCE_CONFIG",
            "PYCBC_OBSERVER_EVIDENCE", "PYCBC_OBSERVER_REPLAY_MODE",
            "PYCBC_OBSERVER_REPLAY_RATE") if key in env}
        timeline = run_profiling_campaign(
            command=command, output_json=timeline_path, cwd=output_dir,
            environment=profile_environment,
            workload={"arm": arm, "scheme": ARM_SCHEMES[arm],
                      "workload_digest": frozen_workload,
                      "resources": {"physical_cpu_cores": physical_cores,
                                    "gpus": gpus if arm == "jax_cuda" else 0}},
            affinity_core=str(affinity or ""), completion_grace_sec=30.0)
        returncode = int(timeline.get("returncode", 1))
        stderr_path.write_text("".join(item[1] for item in timeline.get("stderr_log", [])))
        stdout_path.write_text("")
        samples = list(timeline.get("telemetry", []))
        events = list(timeline.get("stage_events", []))
    else:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            process = subprocess.Popen(command, cwd=str(output_dir), env=env,
                                       stdout=stdout, stderr=stderr)
            returncode = process.wait()
        events = None
    elapsed = time.perf_counter() - start
    cache_audit = (
        compilation_audit(
            output_dir,
            f"arm={arm}",
            jax_cache_dir,
            require_cache_hits,
        )
        if is_jax else None
    )
    inputs_stable = workload_digest(config) == frozen_workload
    for sample in samples:
        if "time_sec" in sample:
            sample["elapsed_sec"] = max(0.0, sample["time_sec"] - start)
        if physical_cores:
            sample["physical_core_fraction"] = (
                sample.get("cpu_percent", 0.0) / (100.0 * physical_cores))
    if events is None:
        text = "\n".join(path.read_text(errors="replace") for path in (stdout_path, stderr_path))
        events = parse_stage_events(text)
    else:
        # The generic profiler uses elapsed seconds and stage names; adapt its
        # complete timeline to the live runner's event contract.
        normalized_events = []
        for event in events:
            item = dict(event)
            item.setdefault("name", item.get("stage"))
            item.setdefault("rank", 0)
            item.setdefault("pid", 0)
            if "time_ns" not in item and item.get("elapsed_sec") is not None:
                item["time_ns"] = int(float(item["elapsed_sec"]) * 1_000_000_000)
            normalized_events.append(item)
        events = normalized_events
    filter_contract = observed_filter_contract(events)
    workload = workload_from_config(config)
    analysis_chunk = float(workload.get("analysis_chunk_sec", 0.0))
    deadline_budget = (analysis_chunk / replay_rate
                       if replay_mode == "paced" and replay_rate > 0
                       else analysis_chunk)
    metrics = block_metrics(events, analysis_chunk, deadline_budget)
    stages = stage_metrics(events)
    intervals = metrics["valid_detector_intervals"]
    detector_intervals = {}
    for interval in intervals:
        for ifo in interval.get("live_detectors", []):
            detector_intervals.setdefault(ifo, []).append(
                (interval["start"], interval["end"]))
    valid_seconds_by_ifo = {}
    for ifo, spans in detector_intervals.items():
        merged = []
        for start_point, end_point in sorted(spans):
            if not merged or start_point > merged[-1][1]:
                merged.append([start_point, end_point])
            else:
                merged[-1][1] = max(merged[-1][1], end_point)
        valid_seconds_by_ifo[ifo] = sum(end - start for start, end in merged)
    valid_seconds = sum(valid_seconds_by_ifo.values())
    templates = workload.get("templates")
    count_valid = workload.get("template_count_matches_bank", True)
    completed = (float(templates * valid_seconds)
                 if templates is not None and count_valid else None)
    process_complete = bool(
        returncode == 0 and metrics["block_count"] > 0 and
        metrics["unpaired_block_count"] == 0 and
        metrics["invalid_block_count"] == 0 and
        completed is not None and math.isfinite(float(completed)) and
        float(completed) > 0.0 and filter_contract["passed"] and inputs_stable)
    capacity = None
    if completed is not None and elapsed > 0:
        denominator = gpus if arm == "jax_cuda" and gpus else physical_cores
        capacity = completed / elapsed / denominator if denominator else None
    followup_enabled = "--run-snr-optimization" in config_args(config)
    active_followups = []
    if followup_enabled and psutil is not None:
        for pid in sorted(observed_pids):
            if pid == process.pid:
                continue
            try:
                child = psutil.Process(pid)
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    active_followups.append(pid)
            except (psutil.Error, OSError):
                continue
    followups = {
        "enabled": followup_enabled,
        "scope": "descendant PIDs observed by the process sampler",
        "completion_observed": bool(
            returncode == 0 and (not followup_enabled or
                                 (psutil is not None and not active_followups))),
        "active_pids_at_main_exit": active_followups,
        "main_process_returncode": returncode,
        "note": "pycbc_live starts followups asynchronously; completion covers only PIDs observed before MPI exit",
    }
    trigger_outputs, evidence_outputs = _discover_hdf_outputs(output_dir)
    science_config = science_config_for_arm(config, arm, frozen_workload)
    science_config.update({
        "workload": workload,
        "sample_rate": BENCHMARK_SAMPLE_RATE,
        "precision": BENCHMARK_PRECISION,
        "replay_mode": replay_mode,
        "replay_rate": replay_rate,
    })
    return {
        "arm": arm,
        "scheme": ARM_SCHEMES[arm],
        "source_root": str(source_root),
        "source_role": ("pristine_upstream_reference" if observe_pristine_cpu
                         else "candidate_checkout"),
        "execution_description": (
            "observer-wrapped pristine upstream CPU reference"
            if observe_pristine_cpu else
            f"candidate checkout {arm} execution using {ARM_SCHEMES[arm]}"),
        "command": command,
        "environment": {key: env[key] for key in (
            "PYTHONPATH", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "XLA_PYTHON_CLIENT_PREALLOCATE",
            "JAX_ENABLE_COMPILATION_CACHE", "JAX_COMPILATION_CACHE_DIR",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
            "PYCBC_JAX_COMPILATION_AUDIT_DIR",
            "PYCBC_REPLAY_CLOCK", "PYCBC_BENCHMARK_STAGES",
            "PYCBC_BENCHMARK_SYNCHRONIZE_STAGES",
            "PYCBC_SCIENCE_CONFIG", "PYCBC_OBSERVER_EVIDENCE",
            "PYCBC_OBSERVER_REPLAY_MODE", "PYCBC_OBSERVER_REPLAY_RATE") if key in env},
        "returncode": returncode,
        "status": "passed" if process_complete else "failed",
        "process_complete": process_complete,
        "profile_utilization": bool(profile_utilization),
        "timeline": str(timeline_path) if profile_utilization else None,
        "timeline_path": str(timeline_path) if profile_utilization else None,
        "profile_scope": ("complete process tree and structured stage telemetry"
                           if profile_utilization else None),
        "filter_contract": filter_contract,
        "elapsed_wall_sec": elapsed,
        "resources": {
            "physical_cpu_cores": physical_cores,
            "gpus": gpus if arm == "jax_cuda" else 0,
            "allocated_host_cores": physical_cores,
        },
        "workload": workload,
        "sample_rate": BENCHMARK_SAMPLE_RATE,
        "precision": BENCHMARK_PRECISION,
        "valid_detector_seconds": valid_seconds,
        "valid_detector_seconds_by_ifo": valid_seconds_by_ifo,
        "completed_template_seconds": completed,
        "templates_per_core": (
            completed / elapsed / physical_cores
            if completed and elapsed and physical_cores else None
        ),
        "templates_per_gpu": (
            completed / elapsed / gpus
            if completed and elapsed and gpus and arm == "jax_cuda" else None
        ),
        "capacity": capacity,
        "latency": metrics["latency"],
        "wall_latency": metrics["wall_latency"],
        "backlog": metrics["backlog"],
        "backlog_method": metrics["backlog_method"],
        "stage_timings": stages,
        "replay_lag": metrics["replay_lag"],
        "deadline_basis": "processing_elapsed_sec",
        "deadline_budget_sec": (
            float(workload.get("analysis_chunk_sec", 0.0)) / replay_rate
            if replay_rate > 0 and replay_mode == "paced"
            else float(workload.get("analysis_chunk_sec", 0.0))
        ),
        "replay_rate": replay_rate,
        "processing_overrun": metrics["processing_overrun"],
        "deadline_count": metrics["deadline_count"],
        "deadlines_met": metrics["deadlines_met"],
        "deadline_miss_count": metrics["deadline_miss_count"],
        "valid_detector_intervals": intervals,
        "block_starts": metrics["block_starts"],
        "block_count": metrics["block_count"],
        "unpaired_block_count": metrics["unpaired_block_count"],
        "invalid_block_count": metrics["invalid_block_count"],
        "utilization": {"samples": samples, "unit": "percent-of-processes"},
        "followups": followups,
        "trigger_outputs": trigger_outputs,
        "evidence_outputs": evidence_outputs,
        "science_config": science_config,
        "workload_digest": frozen_workload,
        "inputs_stable": inputs_stable,
        "compilation_cache": cache_audit,
        "science": {
            "passed": False,
            "scope": "live output pending common qualifier",
            "missing_gates": [
                "conditioned strain and PSD evidence",
                "detector geometry evidence",
                "scientific output comparison",
            ],
        },
        "logs": {"stdout": str(stdout_path), "stderr": str(stderr_path)},
    }


def _evidence_block_key(path: str) -> float | None:
    """Read the recorded block start used to map evidence to trigger HDFs."""
    if h5py is None:
        return None
    try:
        with h5py.File(path, "r") as handle:
            value = handle.attrs.get("analyze_start")
            return float(value) if value is not None else None
    except (OSError, TypeError, ValueError):
        return None


def _science_sample_rate(config: Mapping[str, Any]) -> float:
    validate_benchmark_config(config)
    return BENCHMARK_SAMPLE_RATE


def _sample_grid_key(value: float, sample_rate: float) -> int | None:
    try:
        value = float(value)
        if not math.isfinite(value) or not math.isfinite(sample_rate) or sample_rate <= 0:
            return None
        return int(round(value * sample_rate))
    except (TypeError, ValueError, OverflowError):
        return None


def _trigger_block_start(path: str) -> float | None:
    """Read the recorded block start from ``<start>-<valid_pad>.hdf``."""
    stem = Path(path).stem
    match = re.search(r"-(\d+(?:\.\d+)?)-\d+(?:\.\d+)?$", stem)
    return float(match.group(1)) if match else None


def _live_science_comparator():
    """Load the live-aware strict comparator.

    Falling back to the generic inspiral comparator would misclassify live
    block-level ``gates`` and ``psd`` arrays as trigger rows, so absence of
    the live helper is an explicit setup failure.
    """
    try:
        from tools.benchmark_science import compare_live_scientific_hdf
        return compare_live_scientific_hdf
    except ImportError:  # script execution from the tools directory
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from benchmark_science import compare_live_scientific_hdf
        return compare_live_scientific_hdf


def _map_block_evidence(trigger_paths: Sequence[str], evidence_paths: Sequence[str],
                        config: Mapping[str, Any],
                        block_starts: Sequence[float] | None = None
                        ) -> tuple[dict[str, str], list[str]]:
    """Map evidence by recorded block start and fail closed on coverage gaps."""
    sample_rate = _science_sample_rate(config)
    # Actual rank-0 block markers are authoritative.  The nominal command
    # start/end are only a compatibility fallback for callers without a run
    # receipt; they are not used by the campaign path.
    expected = [float(value) for value in (block_starts or [])]
    if not expected:
        workload = workload_from_config(config)
        chunk = float(workload.get("analysis_chunk_sec", 0.0))
        start = workload.get("start_time")
        end = workload.get("end_time")
        try:
            expected = [float(start) + index * chunk
                        for index in range(int(round((float(end) - float(start)) / chunk)))]
        except (TypeError, ValueError, ZeroDivisionError):
            expected = []
    expected_grid = {_sample_grid_key(value, sample_rate): value for value in expected}
    evidence_by_key = {}
    errors = []
    for path in evidence_paths:
        key = _evidence_block_key(path)
        grid = _sample_grid_key(key, sample_rate) if key is not None else None
        if grid is None or grid in evidence_by_key:
            errors.append(f"invalid or duplicate evidence block: {Path(path).name}")
        else:
            evidence_by_key[grid] = path
    for grid in expected_grid:
        if grid not in evidence_by_key:
            errors.append("missing evidence block for recorded analysis start")
    trigger_grids = set()
    mapping = {}
    used = set()
    for path in trigger_paths:
        start = _trigger_block_start(path)
        key = _sample_grid_key(start, sample_rate) if start is not None else None
        if key is not None:
            trigger_grids.add(key)
        if key is None or key not in evidence_by_key or key in used:
            errors.append(f"missing evidence mapping: {Path(path).name}")
            continue
        mapping[path] = evidence_by_key[key]
        used.add(key)
    # Evidence is emitted for every processed block, including initial
    # buffer-fill blocks. Every evidence block and every recorded block must
    # have a corresponding trigger HDF; the live writer emits a zero-detector
    # HDF for buffer blocks. If that output is absent, fail closed rather than
    # allowing evidence-only content to escape scientific comparison.
    if set(evidence_by_key) != set(expected_grid):
        errors.append("evidence coverage does not match trigger output blocks")
    if trigger_grids != set(expected_grid):
        errors.append("trigger output coverage does not match recorded blocks")
    return mapping, errors


def _scientific_comparisons(runs: Mapping[str, list], config: Mapping[str, Any]) -> None:
    """Attach strict common-qualifier results when comparable HDF evidence exists."""
    science = config.get("science", {})
    if not isinstance(science, Mapping):
        science = {}
    reference_arm = str(science.get("reference_arm", "cpu"))
    reference_runs = runs.get(reference_arm, [])
    evidence = science.get("evidence", {})
    if not isinstance(evidence, Mapping):
        evidence = {}
    compare_scientific_hdf = _live_science_comparator()
    rate = _science_sample_rate(config)
    for arm, arm_runs in runs.items():
        for index, candidate in enumerate(arm_runs):
            if index >= len(reference_runs):
                continue
            reference = reference_runs[index]
            ref_by_name = {Path(path).name: path for path in reference.get("trigger_outputs", [])}
            cand_by_name = {Path(path).name: path for path in candidate.get("trigger_outputs", [])}
            exact_output_set = set(ref_by_name) == set(cand_by_name)
            comparisons = []
            ref_evidence, ref_errors = _map_block_evidence(
                list(ref_by_name.values()), reference.get("evidence_outputs", []), config,
                reference.get("block_starts"))
            cand_evidence, cand_errors = _map_block_evidence(
                list(cand_by_name.values()), candidate.get("evidence_outputs", []), config,
                candidate.get("block_starts"))
            # Explicit evidence paths are accepted only when they cover every
            # output; positional assignment is deliberately unsupported.
            if evidence.get(reference_arm) is not None or evidence.get(arm) is not None:
                ref_errors.append("explicit evidence mapping requires per-block keys")
                cand_errors.append("explicit evidence mapping requires per-block keys")
            for name in sorted(ref_by_name.keys() & cand_by_name.keys()):
                comparisons.append(compare_scientific_hdf(
                    ref_by_name[name], cand_by_name[name], rate,
                    evidence_reference=ref_evidence.get(ref_by_name[name]),
                    evidence_candidate=cand_evidence.get(cand_by_name[name])))
            missing = set(ref_errors + cand_errors)
            if not comparisons:
                missing.add("matching trigger output HDF")
            if not exact_output_set:
                missing.add("trigger output set mismatch")
                missing.update(f"missing candidate output: {name}"
                              for name in sorted(set(ref_by_name) - set(cand_by_name)))
                missing.update(f"extra candidate output: {name}"
                              for name in sorted(set(cand_by_name) - set(ref_by_name)))
            for result in comparisons:
                missing.update(result.get("missing_gates", []))
                if not result.get("passed", False) and not result.get("missing_gates"):
                    missing.add("scientific fields or evidence mismatch")
            candidate["science"] = {
                "passed": exact_output_set and not missing and bool(comparisons) and
                          all(r.get("passed", False) for r in comparisons),
                "scope": "per-block trigger outputs with validated evidence coverage",
                "reference_arm": reference_arm,
                "comparisons": comparisons,
                "missing_gates": sorted(missing),
            }


def _timed_science_comparisons(runs: Mapping[str, list], qualification: Mapping[str, list],
                               config: Mapping[str, Any]) -> None:
    """Compare timed trigger outputs to the qualified CPU reference.

    Qualification owns the full PSD/strain/configuration gate.  Timed runs
    only need the trigger/metadata gate, since writing evidence in a timed
    process would change the throughput being measured.
    """
    science = config.get("science", {})
    if not isinstance(science, Mapping):
        science = {}
    reference_arm = str(science.get("reference_arm", "cpu"))
    qualified_reference = qualification.get(reference_arm, [])
    rate = _science_sample_rate(config)
    reference = qualified_reference[0] if qualified_reference else None
    compare_scientific_hdf = _live_science_comparator()
    ref_outputs = ({Path(path).name: path for path in reference.get("trigger_outputs", [])}
                   if reference else {})
    for arm, arm_runs in runs.items():
        qualified_arm = qualification.get(arm, [])
        qualification_passed = bool(
            qualified_arm and qualified_arm[0].get("returncode") == 0 and
            qualified_arm[0].get("science", {}).get("passed", False))
        for candidate in arm_runs:
            cand_outputs = {Path(path).name: path
                            for path in candidate.get("trigger_outputs", [])}
            exact_output_set = set(ref_outputs) == set(cand_outputs)
            comparisons = []
            missing = set()
            if reference is None:
                missing.add("qualified CPU reference")
            if not exact_output_set:
                missing.add("trigger output set mismatch")
                missing.update(f"missing candidate output: {name}"
                              for name in sorted(set(ref_outputs) - set(cand_outputs)))
                missing.update(f"extra candidate output: {name}"
                              for name in sorted(set(cand_outputs) - set(ref_outputs)))
            for name in sorted(set(ref_outputs) & set(cand_outputs)):
                result = compare_scientific_hdf(ref_outputs[name], cand_outputs[name], rate)
                comparisons.append(result)
                if not result.get("observed_trigger_and_metadata_pass", False):
                    missing.add("trigger identities and metadata")
            trigger_passed = exact_output_set and bool(comparisons) and not missing
            candidate["science"] = {
                "passed": qualification_passed and trigger_passed,
                "scope": "timed trigger outputs against qualified CPU; full qualification separate",
                "reference_arm": reference_arm,
                "qualification_passed": qualification_passed,
                "trigger_passed": trigger_passed,
                "comparisons": comparisons,
                "missing_gates": sorted(missing | ({"full qualification"}
                                                     if not qualification_passed else set())),
            }


def _frame_input_paths(config: Mapping[str, Any]) -> list[Path]:
    args = config_args(config)
    values = []
    for key in ("frame_files", "frame_src", "input_files"):
        value = config.get(key, [])
        values.extend(value if isinstance(value, list) else [value])
    # ``--bank-file`` and frame sources are independent inputs.  Always
    # inspect the command line for frame sources when the config does not
    # spell them out; the presence of a bank must not hide an unlisted frame.
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--frame-src":
            index += 1
            while index < len(args) and not args[index].startswith("--"):
                values.append(args[index].split(":", 1)[-1])
                index += 1
            continue
        if token.startswith("--frame-src="):
            values.append(token.split("=", 1)[1].split(":", 1)[-1])
        index += 1
    paths = []
    for value in values:
        for name in glob.glob(str(value)) or [str(value)]:
            path = Path(name)
            if path.is_file():
                paths.append(path.resolve())
    return sorted(set(paths))


def _input_paths(config: Mapping[str, Any]) -> list[Path]:
    args = config_args(config)
    paths = []
    bank = config.get("bank_file") or option_value(args, "--bank-file")
    if bank:
        paths.append(Path(bank))
    paths.extend(_frame_input_paths(config))
    return sorted({path.resolve() for path in paths})


def _source_manifest_files(source_root: Path) -> list[Path]:
    """Return the complete runtime source manifest used by receipt/resume."""
    paths = [source_root / "bin" / "pycbc_live",
             source_root / "tools" / "bench_jax_live_campaign.py",
             source_root / "tools" / "benchmark_science.py"]
    observer = source_root / "tools" / "observe_pycbc_live.py"
    if observer.is_file():
        paths.append(observer)
    paths.extend(source_root.glob("pycbc/**/*.py"))
    paths.extend(source_root.glob("pycbc/**/*.so"))
    return sorted({path.resolve() for path in paths if path.is_file()})


def _observer_identity() -> dict | None:
    observer = Path(__file__).resolve().with_name("observe_pycbc_live.py")
    if not observer.is_file():
        return None
    return {"path": str(observer), "sha256": file_sha256(observer)}


def campaign_identity(config: Mapping[str, Any], source_root: Path,
                      config_path: Path, python_bin: str,
                      execution: Mapping[str, Any] | None = None,
                      reference_source_root: Path | None = None) -> dict:
    validate_benchmark_config(config)
    execution = dict(execution or {})
    identity = {
        "config_sha256": file_sha256(config_path),
        "source": source_identity(source_root, _source_manifest_files(source_root)),
        "inputs": {str(path): file_sha256(path) for path in _input_paths(config)},
        "runtime": {
            "hostname": platform.node(),
            "python": str(Path(python_bin).resolve()),
            "platform": platform.platform(),
        },
            "execution": execution,
        "benchmark_physics": {
            "sample_rate": BENCHMARK_SAMPLE_RATE,
            "precision": BENCHMARK_PRECISION,
        },
    }
    observer = _observer_identity()
    if observer is not None:
        identity["observer"] = observer
    if reference_source_root is not None:
        reference_source_root = reference_source_root.resolve()
        identity["reference_source"] = source_identity(
            reference_source_root,
            _source_manifest_files(reference_source_root),
        )
    return identity


def make_receipt(config: Mapping[str, Any], runs: Mapping[str, list], source_root: Path,
                 config_path: Path, replay_mode: str, replay_rate: float,
                 qualification: Mapping[str, list] | None = None,
                 failed_history: Sequence[dict] = (),
                 timing_policy: Mapping[str, Any] | None = None,
                 reference_source_root: Path | None = None) -> dict:
    validate_benchmark_config(config)
    inputs = {"config": str(config_path), "config_sha256": file_sha256(config_path)}
    args = config_args(config)
    bank_file = config.get("bank_file") or option_value(args, "--bank-file")
    if bank_file:
        path = Path(bank_file)
        inputs["bank_file"] = str(path)
        if path.exists():
            inputs["bank_file_sha256"] = file_sha256(path)
    # Keep frame/cache identities alongside the configuration.  Globs are
    # expanded at receipt creation so a later rerun cannot accidentally use a
    # different cache with the same command line.
    frame_hashes = []
    bank_path = Path(bank_file).resolve() if bank_file else None
    for path in _input_paths(config):
        if bank_path is None or path != bank_path:
            frame_hashes.append({"path": str(path), "sha256": file_sha256(path)})
    if frame_hashes:
        inputs["frame_files"] = frame_hashes
    workload = workload_from_config(config)
    source_files = _source_manifest_files(source_root)
    digest = workload_digest(config)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executable": "pycbc_live",
        "sample_rate": BENCHMARK_SAMPLE_RATE,
        "precision": BENCHMARK_PRECISION,
        "source": source_identity(source_root, source_files),
        "inputs": inputs,
        "input_contract": digest["contract"],
        "workload": workload,
        "workload_digest": digest,
        "science_config": {
            "args": args,
            "config": config.get("science", {}),
        },
        "replay": {
            "mode": replay_mode, "rate": replay_rate,
            "clock": replay_mode in {"unpaced", "paced"},
        },
        "host": host_snapshot(),
        "raw_results": dict(runs),
        "qualification_results": dict(qualification or {}),
        "failed_history": list(failed_history),
        "timing_policy": dict(timing_policy or {}),
    }
    observer = _observer_identity()
    if observer is not None:
        receipt["observer"] = observer
    if reference_source_root is not None:
        reference_source_root = reference_source_root.resolve()
        receipt["reference_source"] = source_identity(
            reference_source_root,
            _source_manifest_files(reference_source_root),
        )
    return receipt


def _completed_run(result: Any, arm: str, require_evidence: bool = False,
                   expected_digest: str | None = None) -> bool:
    if not isinstance(result, Mapping):
        return False
    if result.get("arm") != arm or result.get("status") != "passed":
        return False
    if result.get("process_complete") is False:
        return False
    if result.get("returncode") != 0 or not result.get("command"):
        return False
    if not _valid_filter_contract(result.get("filter_contract")):
        return False
    if result.get("inputs_stable") is False:
        return False
    if expected_digest is not None:
        digest = result.get("workload_digest", {})
        if not _matches_workload_digest(digest, expected_digest):
            return False
    try:
        elapsed = float(result.get("elapsed_wall_sec"))
        completed = float(result.get("completed_template_seconds"))
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(elapsed) and elapsed > 0 and
            math.isfinite(completed) and completed > 0):
        return False
    if "block_count" in result and int(result.get("block_count", 0)) <= 0:
        return False
    logs = result.get("logs", {})
    if not all(name in logs and Path(logs[name]).is_file()
               for name in ("stdout", "stderr")):
        return False
    if not result.get("trigger_outputs") or not all(
            Path(path).is_file() for path in result["trigger_outputs"]):
        return False
    if require_evidence and (not result.get("evidence_outputs") or not all(
            Path(path).is_file() for path in result["evidence_outputs"])):
        return False
    if ARM_SCHEMES[arm].startswith('jax'):
        audit = result.get('compilation_cache')
        if (not isinstance(audit, Mapping) or audit.get('enabled') is not True
                or int(audit.get('compile_requests', 0)) <= 0):
            return False
        if not require_evidence and audit.get('all_requests_hit') is not True:
            return False
    return True


def _enforce_workload_digest(results: Mapping[str, list], expected: str | None) -> None:
    if expected is None:
        return
    for values in results.values():
        for result in values:
            if not isinstance(result, Mapping):
                continue
            digest = result.get("workload_digest", {})
            if (_matches_workload_digest(digest, expected) and
                    result.get("inputs_stable") is not False):
                continue
            result["science"] = {
                "passed": False,
                "missing_gates": ["workload contract digest"],
                "scope": "run did not carry the frozen workload contract",
            }


def _save_progress(path: Path, identity: Mapping[str, Any], status: str,
                   qualification: Mapping[str, list], runs: Mapping[str, list],
                   failed_history: Sequence[dict], **extra: Any) -> None:
    payload = {
        "status": status,
        "identity": identity,
        "qualification_results": qualification,
        "raw_results": runs,
        "failed_history": list(failed_history),
    }
    payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _retry_output_dir(base: Path, previous: Mapping[str, Any] | None) -> Path:
    """Choose a fresh directory when a prior attempt would be overwritten."""
    if not previous:
        return base
    index = 1
    while True:
        candidate = base.with_name(f"{base.name}.retry{index}")
        if not candidate.exists():
            return candidate
        index += 1


def qualification_and_timed_configs(config: Mapping[str, Any]) -> tuple[dict, dict]:
    """Build isolated evidence and timed configurations for one campaign."""
    qual_config = dict(config)
    science = config.get("science", {})
    qual_science = dict(science) if isinstance(science, Mapping) else {}
    qual_science["capture_evidence"] = True
    qual_config["science"] = qual_science
    timed_config = dict(config)
    timed_science = dict(qual_science)
    timed_science["capture_evidence"] = False
    timed_config["science"] = timed_science
    return qual_config, timed_config


def qualification_gate(result: Any) -> bool:
    """Return whether an arm may enter throughput timing."""
    science = result.get("science", {}) if isinstance(result, Mapping) else {}
    return bool(isinstance(result, Mapping) and
                result.get("status") == "passed" and
                result.get("returncode") == 0 and
                result.get("process_complete") is True and
                isinstance(science, Mapping) and science.get("passed", False))


def qualification_skip_reason(arm: str, result: Any) -> dict:
    """Describe why an arm was excluded from equivalent-output timing."""
    science = result.get("science", {}) if isinstance(result, Mapping) else {}
    return {
        "arm": arm,
        "reason": "qualification_gate_failed",
        "status": result.get("status") if isinstance(result, Mapping) else None,
        "returncode": result.get("returncode") if isinstance(result, Mapping) else None,
        "process_complete": bool(result.get("process_complete", False))
        if isinstance(result, Mapping) else False,
        "science_passed": bool(science.get("passed", False))
        if isinstance(science, Mapping) else False,
        "missing_gates": list(science.get("missing_gates", []))
        if isinstance(science, Mapping) else ["qualification result unavailable"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON config containing complete realistic pycbc_live args")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reference-source-root", type=Path, default=None,
                        help="immutable clean checkout used for the native CPU reference")
    parser.add_argument("--reference-revision", default=None,
                        help="full 40-character commit SHA pinned for the CPU reference")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARM_SCHEMES),
                        default=["cpu", "branch_cpu", "jax_cpu", "jax_cuda"],
                        help="arms to run: cpu is pristine upstream reference; branch_cpu is candidate CPU; jax_cpu and jax_cuda are candidate JAX paths")
    parser.add_argument("--replicates", type=int, default=None,
                        help="timing repetitions (default: 1 for quick, otherwise 3)")
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--affinity", required=True,
                        help="explicit comma/range CPU affinity passed to MPI")
    parser.add_argument("--physical-cpu-cores", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--profile-utilization", action="store_true",
                        help="collect process-tree utilization in a separate profiled run")
    parser.add_argument("--resume", action="store_true",
                        help="resume only validated completed runs from the progress checkpoint")
    parser.add_argument("--replay-mode", choices=("unpaced", "paced", "none"), default="unpaced")
    parser.add_argument("--replay-rate", type=float, default=1.0)
    parser.add_argument("--allow-unqualified-timings", action="store_true",
                        help="run diagnostic timings even when qualification fails")
    parser.add_argument(
        "--quick", action="store_true",
        help=("run exactly one fresh timing repetition per arm and mark all "
              "performance output diagnostic-only"),
    )
    parser.add_argument("--qualification-only", action="store_true",
                        help="run full scientific qualification and zero timed repetitions; receipt makes no performance claim")
    args = parser.parse_args(argv)
    if args.replicates is None:
        args.replicates = 1 if args.quick else 3
    if args.replicates < 1 or args.ranks < 1:
        parser.error("--replicates and --ranks must be positive")
    if args.quick and (args.qualification_only or args.profile_utilization):
        parser.error("--quick cannot be combined with qualification-only or profiling")
    if args.quick and args.replicates != 1:
        parser.error("--quick requires exactly one timing repetition")
    if "jax_cuda" in args.arms and args.gpus != 1:
        parser.error("the jax_cuda arm uses exactly one GPU")
    if not math.isfinite(args.replay_rate) or args.replay_rate <= 0:
        parser.error("--replay-rate must be finite and positive")
    derived_cores = physical_core_count(args.affinity)
    if derived_cores is None or derived_cores < args.ranks:
        parser.error("--affinity must resolve to at least one physical core per MPI rank")
    if args.physical_cpu_cores is not None and args.physical_cpu_cores != derived_cores:
        parser.error("--physical-cpu-cores disagrees with physical cores in --affinity")
    config_text = args.config.read_text()
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError:
        # An argument file is also useful for long, hand-reviewed live
        # commands.  Shell quoting is honored, and all workload metadata can
        # still be supplied through the JSON form when needed.
        config = {"args": shlex.split(config_text)}
    if not isinstance(config, dict):
        parser.error("--config must contain a JSON object")
    try:
        validate_benchmark_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    progress_path = args.output.with_suffix(args.output.suffix + ".progress.json")
    source_root = args.source_root.resolve()
    reference_source_root = (args.reference_source_root.resolve()
                             if args.reference_source_root is not None else None)
    if reference_source_root is not None:
        if not args.allow_unqualified_timings and not args.reference_revision:
            parser.error("--reference-source-root requires --reference-revision")
        try:
            pinned_reference_revision = (
                validate_reference(reference_source_root, args.reference_revision)
                if args.reference_revision else None)
            if pinned_reference_revision:
                validate_source(reference_source_root / "bin" / "pycbc_live")
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    else:
        pinned_reference_revision = None
    config_path = args.config.resolve()
    try:
        validate_input_contract(config)
        expected_workload = workload_digest(config)
        expected_workload_digest = expected_workload["sha256"]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    execution = {
        "arms": list(args.arms),
        "ranks": args.ranks,
        "affinity": args.affinity,
        "physical_cpu_cores": derived_cores,
        "gpus": args.gpus,
        "profile_utilization": bool(args.profile_utilization),
        "replay_mode": args.replay_mode,
        "replay_rate": args.replay_rate,
        "allow_unqualified_timings": bool(args.allow_unqualified_timings),
        "qualification_only": bool(args.qualification_only),
        "quick": bool(args.quick),
        "qualification_gate": "process_complete_and_science_passed",
        "reference_source_root": (str(reference_source_root)
                                   if reference_source_root is not None else None),
        "reference_revision": pinned_reference_revision,
    }
    identity = campaign_identity(config, source_root, config_path, args.python,
                                 execution, reference_source_root)
    identity["workload_digest"] = expected_workload_digest
    reference_required = True
    reference_identity = identity.get("reference_source")
    if pinned_reference_revision:
        identity["reference_revision"] = pinned_reference_revision
    reference_ready = ("cpu" in args.arms and
                        reference_source_root is not None and
                        pinned_reference_revision is not None and
                        _pristine_source(reference_identity))
    if reference_required and (args.quick or not args.allow_unqualified_timings):
        if "cpu" not in args.arms:
            parser.error("qualification requires the cpu reference arm")
        if not reference_ready:
            parser.error("qualification requires --reference-source-root "
                         "to name an immutable clean git checkout")
    qualification: Dict[str, list] = {arm: [] for arm in args.arms}
    runs: Dict[str, list] = {arm: [] for arm in args.arms}
    profile_results: Dict[str, list] = {arm: [] for arm in args.arms}
    failed_history: list[dict] = []
    if args.resume:
        if not progress_path.is_file():
            parser.error(f"--resume requires checkpoint {progress_path}")
        checkpoint = json.loads(progress_path.read_text())
        if checkpoint.get("identity") != identity:
            parser.error("checkpoint identity differs in source, configuration, input, or runtime")
        qualification.update(checkpoint.get("qualification_results", {}))
        runs.update(checkpoint.get("raw_results", {}))
        failed_history.extend(checkpoint.get("failed_history", []))

    qual_config, timed_config = qualification_and_timed_configs(config)

    def before_launch() -> None:
        validate_input_contract(config)
        if workload_digest(config)["sha256"] != expected_workload_digest:
            raise ValueError("workload input contract changed before launch")
        if pinned_reference_revision:
            validate_reference(reference_source_root, pinned_reference_revision)

    for arm in args.arms:
        old = qualification.get(arm, [])
        if old and _completed_run(old[0], arm, require_evidence=True,
                                 expected_digest=expected_workload_digest):
            continue
        if old:
            failed_history.extend(item for item in old if isinstance(item, Mapping))
        base_dir = args.output_dir / "qualification" / arm
        run_dir = _retry_output_dir(base_dir, old[0] if old else None)
        arm_source_root = (reference_source_root if arm == "cpu" and
                           reference_source_root is not None else source_root)
        before_launch()
        result = run_one(qual_config, arm, run_dir,
                         arm_source_root, args.python, args.ranks, args.replay_mode,
                         args.replay_rate, derived_cores, args.gpus, False,
                         args.affinity,
                         arm == "cpu" and pinned_reference_revision is not None,
                         expected_workload_digest,
                         args.output_dir / 'jax-compilation-cache' / arm,
                         False)
        qualification[arm] = [result]
        if result.get("returncode") != 0 or not result.get("process_complete", False):
            failed_history.append(result)
        _save_progress(progress_path, identity, "qualification", qualification, runs,
                       failed_history, arm=arm,
                       input_contract=expected_workload["contract"],
                       workload=workload_from_config(config))
        if not result.get("process_complete", False) and not args.allow_unqualified_timings:
            break
    _scientific_comparisons(qualification, qual_config)
    _enforce_workload_digest(qualification, expected_workload_digest)
    timing_policy = {
        "qualification_required": True,
        "allow_unqualified_timings": bool(args.allow_unqualified_timings),
        "qualification_only": bool(args.qualification_only),
        "quick": bool(args.quick),
        "diagnostic_only": bool(args.quick or args.allow_unqualified_timings),
        "effective_replicates": args.replicates,
        "minimum_publishable_replicates": 3,
        "classification": (
            "quick_known_divergence_diagnostic"
            if args.quick and args.allow_unqualified_timings else
            "known_divergence_diagnostic"
            if args.allow_unqualified_timings else
            "quick_diagnostic" if args.quick else "equivalent_output"),
        "skipped_arms": {},
    }
    qualification_failed = any(
        not qualification_gate(values[0] if values else {})
        for values in qualification.values())
    for arm in args.arms:
        result = qualification.get(arm, [{}])
        result = result[0] if result else {}
        if not args.allow_unqualified_timings and qualification_failed:
            reason = qualification_skip_reason(arm, result)
            reason["reason"] = "qualification_gate_failed_for_campaign"
            timing_policy["skipped_arms"][arm] = reason
    _save_progress(progress_path, identity, "timed", qualification, runs,
                   failed_history, timing_policy=timing_policy,
                   input_contract=expected_workload["contract"],
                   workload=workload_from_config(config))

    stop_timing = bool(args.qualification_only or args.profile_utilization)
    for replicate in range(args.replicates):
        if stop_timing:
            break
        # Alternate arm order so warm-up, host pressure, and filesystem state
        # are counterbalanced across replicate pairs.
        arm_order = arm_order_for_replicate(args.arms, replicate)
        for arm in arm_order:
            if arm in timing_policy["skipped_arms"]:
                continue
            existing = runs.get(arm, [])
            if len(existing) > replicate and _completed_run(
                    existing[replicate], arm,
                    expected_digest=expected_workload_digest):
                continue
            if len(existing) > replicate and existing[replicate]:
                failed_history.append(existing[replicate])
            base_dir = args.output_dir / arm / f"run-{replicate + 1:03d}"
            previous = existing[replicate] if len(existing) > replicate else None
            run_dir = _retry_output_dir(base_dir, previous)
            arm_source_root = (reference_source_root if arm == "cpu" and
                               reference_source_root is not None else source_root)
            before_launch()
            result = run_one(timed_config, arm, run_dir, arm_source_root,
                                     args.python, args.ranks, args.replay_mode,
                                     args.replay_rate, derived_cores, args.gpus,
                                     False, args.affinity,
                                     arm == "cpu" and pinned_reference_revision is not None,
                                     expected_workload_digest,
                                     args.output_dir / 'jax-compilation-cache' / arm,
                                     True)
            while len(runs[arm]) <= replicate:
                runs[arm].append({})
            runs[arm][replicate] = result
            _timed_science_comparisons({arm: [result]}, qualification, timed_config)
            _enforce_workload_digest({arm: [result]}, expected_workload_digest)
            if not qualification_gate(result):
                failed_history.append(result)
                if not args.allow_unqualified_timings:
                    timing_policy["stopped_after"] = {
                        "arm": arm, "replicate": replicate + 1,
                        "reason": "timed process or scientific comparison failed",
                    }
                    stop_timing = True
            _save_progress(progress_path, identity, "running", qualification, runs,
                           failed_history, replicate=replicate + 1, arm=arm,
                           timing_policy=timing_policy,
                           input_contract=expected_workload["contract"],
                           workload=workload_from_config(config))
            if stop_timing:
                break
    _timed_science_comparisons(runs, qualification, timed_config)
    _enforce_workload_digest(runs, expected_workload_digest)
    if args.profile_utilization:
        # Profiling is a separate campaign: exactly one complete process-tree
        # timeline per arm, never included in throughput repetitions.
        for arm in args.arms:
            profile_dir = args.output_dir / "profile" / arm
            arm_source_root = (reference_source_root if arm == "cpu" and
                               reference_source_root is not None else source_root)
            before_launch()
            result = run_one(timed_config, arm, profile_dir, arm_source_root,
                             args.python, args.ranks, args.replay_mode,
                             args.replay_rate, derived_cores, args.gpus,
                             True, args.affinity,
                             arm == "cpu" and pinned_reference_revision is not None,
                             expected_workload_digest,
                             args.output_dir / 'jax-compilation-cache' / arm,
                             True)
            profile_results[arm] = [result]
            _timed_science_comparisons({arm: [result]}, qualification, timed_config)
            _enforce_workload_digest({arm: [result]}, expected_workload_digest)
    inputs_stable = workload_digest(config)["sha256"] == expected_workload_digest
    candidate_stable = source_identity(source_root, _source_manifest_files(source_root)) == identity["source"]
    reference_stable = (reference_source_root is None or
                        source_identity(reference_source_root,
                                       _source_manifest_files(reference_source_root)) ==
                        identity.get("reference_source"))
    if reference_source_root is not None and pinned_reference_revision:
        try:
            validate_reference(reference_source_root, pinned_reference_revision)
        except ValueError:
            reference_stable = False
    if not reference_ready or not reference_stable:
        reason = ("immutable CPU reference unavailable" if not reference_ready
                  else "immutable CPU reference changed during campaign")
        _invalidate_science(qualification, reason)
        _invalidate_science(runs, reason)
        _invalidate_science(profile_results, reason)
    if not inputs_stable or not candidate_stable:
        reason = ("workload inputs changed during campaign" if not inputs_stable
                  else "candidate source changed during campaign")
        _invalidate_science(qualification, reason)
        _invalidate_science(runs, reason)
        _invalidate_science(profile_results, reason)
    receipt = make_receipt(config, runs, source_root, config_path,
                           args.replay_mode, args.replay_rate, qualification,
                           failed_history, timing_policy, reference_source_root)
    receipt["profile_results"] = profile_results
    selected_science = (profile_results if args.profile_utilization else
                        qualification if args.qualification_only else runs)
    receipt["science"] = _science_summary(args.arms, qualification, selected_science)
    receipt["profiling"] = {
        "enabled": bool(args.profile_utilization),
        "profiles_per_arm": 1 if args.profile_utilization else 0,
        "excluded_from_timed_repetitions": True,
        "performance_claim": False if args.profile_utilization else None,
        "timeline": "timeline.json per arm" if args.profile_utilization else None,
    }
    all_results = [run for values in runs.values() for run in values]
    all_qualifications = [run for values in qualification.values() for run in values]
    process_complete = (args.qualification_only or args.profile_utilization or
                        (all(len(runs[arm]) == args.replicates for arm in args.arms) and all(
        bool(run.get("process_complete", run.get("status") == "passed" and
                         run.get("returncode") == 0)) for run in all_results)))
    profile_complete = (not args.profile_utilization or
                        all(len(profile_results[arm]) == 1 and
                            bool(profile_results[arm][0].get("process_complete"))
                            for arm in args.arms))
    science_results = ([run for values in profile_results.values() for run in values]
                       if args.profile_utilization else
                       (all_qualifications if args.qualification_only else all_results))
    science_qualified = bool(all_qualifications and science_results) and all(
        bool(run.get("science", {}).get("passed", False))
        for run in (all_qualifications + science_results))
    receipt["baseline"] = {
        "required": reference_required,
        "provided": reference_source_root is not None,
        "unqualified": bool(args.allow_unqualified_timings),
        "pristine": _pristine_source(reference_identity),
        "stable": reference_stable,
        "publishable": (reference_ready and reference_stable and
                        not args.allow_unqualified_timings and not args.quick),
        "qualification_only": bool(args.qualification_only),
    }
    receipt["campaign"] = {
        "process_complete": process_complete,
        "inputs_stable": inputs_stable,
        "candidate_stable": candidate_stable,
        "science_qualified": (science_qualified and reference_ready and
                               reference_stable and
                               not args.allow_unqualified_timings),
        "passed": (process_complete and profile_complete and science_qualified and reference_ready and
                    reference_stable and not args.allow_unqualified_timings
                    and not args.quick),
    }
    if args.allow_unqualified_timings:
        receipt["campaign"].update({
            "status": "complete_known_divergence",
            "performance_claim": False,
            "known_divergence": True,
        })
        receipt["status"] = "complete_known_divergence"
    if args.profile_utilization:
        receipt["campaign"]["profile_runs"] = 1
    if args.qualification_only:
        receipt["campaign"].update({
            "performance_claim": False,
            "status": "science_qualification_only",
        })
    if args.quick:
        receipt.update({
            "status": "complete_quick_diagnostic",
            "performance_claim": False,
            "publishable": False,
            "diagnostic_only": True,
            "benchmark_tier": "quick",
        })
        receipt["campaign"].update({
            "status": "complete_quick_diagnostic",
            "performance_claim": False,
            "publishable": False,
            "quick": True,
            "diagnostic_only": True,
            "known_divergence": bool(args.allow_unqualified_timings),
        })
    else:
        receipt.setdefault("performance_claim", bool(receipt["campaign"]["passed"]))
        receipt.setdefault("publishable", bool(receipt["campaign"]["passed"]))
        receipt.setdefault("diagnostic_only", bool(args.allow_unqualified_timings))
        receipt.setdefault("benchmark_tier", "full")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _save_progress(progress_path, identity, "complete", qualification, runs,
                   failed_history, timing_policy=timing_policy,
                   input_contract=expected_workload["contract"],
                   workload=workload_from_config(config))
    return 0 if (receipt["campaign"]["passed"]
                 or (args.quick and process_complete and profile_complete)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
