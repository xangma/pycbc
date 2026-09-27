#!/usr/bin/env python3
"""Run counterbalanced fresh-process pycbc_inspiral benchmark arms.

Compare a baseline/current CPU checkout and selected JAX CPU/CUDA paths, with
optional JAX batching. Record process wall time and executable HDF timers.
Explicit diffgw arms are rejected because the current executable does not
expose that provider as a campaign CLI mode. Log-derived phase estimates are approximate
and do not constitute an independently instrumented six-phase decomposition.

Every repeat is checked against exact CPU trigger identities. Separate
qualification runs capture complete strain/PSD/geometry evidence; missing gates
remain explicit and cannot qualify a speedup.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

try:
    from tools.benchmark_science import compare_scientific_hdf
    from tools.benchmark_artifact import runtime_metadata, source_identity
    from tools.benchmark_reference import validate_reference
    from tools.observe_pycbc_inspiral import validate_source as validate_observer_source
except ModuleNotFoundError:
    from benchmark_science import compare_scientific_hdf
    from benchmark_artifact import runtime_metadata, source_identity
    from benchmark_reference import validate_reference
    from observe_pycbc_inspiral import validate_source as validate_observer_source


ARM_NAMES = (
    "original_cpu",
    "branch_cpu",
    "jax_cpu",
    "jax_cpu_batched",
    "jax_cpu_diffgw",
    "jax_cuda_lal",
    "jax_cuda",
    "jax_cuda_batched",
    "jax_cuda_diffgw",
)

DEFAULT_ARMS = ("original_cpu", "branch_cpu", "jax_cpu", "jax_cuda")
BATCHED_ARMS = (
    "original_cpu",
    "branch_cpu",
    "jax_cpu_batched",
    "jax_cuda_batched",
)
BENCHMARK_SAMPLE_RATE = 2048.0
BENCHMARK_PRECISION = "complex64"
WAVEFORM_MODES = ("compressed", "generated")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
        )
    except Exception:
        return "unknown"


def sample_summary(samples: List[float], unit: str = "seconds") -> Dict[str, Any]:
    arr = np.asarray(samples, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"count": 0, "samples": [], "unit": unit}
    sorted_arr = np.sort(arr)
    median = float(np.median(sorted_arr))
    mean = float(np.mean(sorted_arr))
    stddev = float(np.std(sorted_arr, ddof=1)) if n > 1 else 0.0
    p25 = float(np.percentile(sorted_arr, 25))
    p75 = float(np.percentile(sorted_arr, 75))

    # 95% bootstrap CI for median
    rng = np.random.default_rng(7101)
    if n >= 3:
        boot_medians = [
            float(np.median(rng.choice(arr, size=n, replace=True)))
            for _ in range(2000)
        ]
        ci_low = float(np.percentile(boot_medians, 2.5))
        ci_high = float(np.percentile(boot_medians, 97.5))
    else:
        ci_low = float(sorted_arr[0])
        ci_high = float(sorted_arr[-1])

    return {
        "count": n,
        "samples": [float(x) for x in samples],
        "unit": unit,
        "median": median,
        "mean": mean,
        "stddev": stddev,
        "min": float(sorted_arr[0]),
        "max": float(sorted_arr[-1]),
        "p25": p25,
        "p75": p75,
        "median_ci95": {"low": ci_low, "high": ci_high},
    }


def _parse_stderr_phases(
    stderr_lines: List[str],
    process_wall_sec: float,
    calc_time_sec: float,
    tsetup_sec: float,
) -> Dict[str, float]:
    """Extract the 6 mutually exclusive phases from verbose timestamps and timers."""
    dt_re = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)")
    events: List[Tuple[float, str]] = []
    t0_dt = None

    for line in stderr_lines:
        m = dt_re.match(line)
        if m:
            try:
                ts_str = m.group(1)
                parts = ts_str.split(".")
                frac = (parts[1] + "000000")[:6]
                clean_ts = f"{parts[0]}.{frac}"
                dt = datetime.fromisoformat(clean_ts)
                if t0_dt is None:
                    t0_dt = dt
                offset = (dt - t0_dt).total_seconds()
                events.append((offset, line))
            except Exception:
                pass

    if not events:
        filter_sec = max(0.0, calc_time_sec)
        cond_sec = max(0.0, tsetup_sec)
        startup_sec = max(0.0, process_wall_sec - filter_sec - cond_sec)
        return {
            "startup_import_sec": startup_sec,
            "conditioning_sec": cond_sec,
            "waveform_prep_sec": 0.0,
            "matched_filter_sec": filter_sec,
            "vetoes_clustering_sec": 0.0,
            "serialization_io_sec": 0.0,
        }

    t_start_cond = 0.0
    t_end_cond = 0.0
    t_first_filter = None
    t_last_filter = None
    t_start_write = None
    t_finished = None

    for offset, line in events:
        if "Reading Frames" in line and t_start_cond == 0.0:
            t_start_cond = offset
        elif ("Read in template bank" in line or "generating" in line) and t_end_cond == 0.0:
            t_end_cond = offset
        elif "Filtering template" in line and t_first_filter is None:
            t_first_filter = offset
        elif ("We currently have" in line or "Outputting" in line) and t_last_filter is None:
            t_last_filter = offset
        elif "Writing out triggers" in line and t_start_write is None:
            t_start_write = offset
        elif "Finished" in line:
            t_finished = offset

    first_log_offset = events[0][0]
    last_log_offset = events[-1][0] if events else 0.0

    log_span = last_log_offset - first_log_offset
    startup_sec = max(0.1, process_wall_sec - log_span)

    if t_end_cond > t_start_cond > 0:
        cond_sec = t_end_cond - t_start_cond
    else:
        cond_sec = tsetup_sec

    matched_filter_sec = max(0.0, calc_time_sec)
    filter_span = (t_last_filter - t_first_filter) if (t_first_filter and t_last_filter) else calc_time_sec
    waveform_prep_sec = max(0.0, filter_span - matched_filter_sec)

    if t_last_filter and t_start_write and t_start_write >= t_last_filter:
        veto_sec = t_start_write - t_last_filter
    else:
        veto_sec = 0.05

    if t_start_write and t_finished and t_finished >= t_start_write:
        io_sec = t_finished - t_start_write
    else:
        io_sec = 0.05

    other_sum = cond_sec + waveform_prep_sec + matched_filter_sec + veto_sec + io_sec
    startup_sec = max(0.0, process_wall_sec - other_sum)

    return {
        "startup_import_sec": startup_sec,
        "conditioning_sec": cond_sec,
        "waveform_prep_sec": waveform_prep_sec,
        "matched_filter_sec": matched_filter_sec,
        "vetoes_clustering_sec": veto_sec,
        "serialization_io_sec": io_sec,
    }


def compare_trigger_parity(reference_hdf, candidate_hdf,
                           sample_rate=BENCHMARK_SAMPLE_RATE):
    """Strict trigger-only gate; full qualification additionally needs evidence."""
    result = compare_scientific_hdf(reference_hdf, candidate_hdf, sample_rate)
    result['passed'] = result['observed_trigger_and_metadata_pass']
    result['scope'] = 'trigger identities, all fields and metadata only'
    return result


def bank_inventory(path):
    """Count distinct physical templates; repeated rows cannot inflate capacity."""
    with h5py.File(path, 'r') as bank:
        fields = [name for name in ('mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z',
                                     'spin2x', 'spin2y', 'spin2z') if name in bank]
        if not {'mass1', 'mass2'} <= set(fields):
            raise ValueError('Bank must contain mass1 and mass2')
        columns = [np.asarray(bank[name]) for name in fields]
        if any(column.ndim != 1 or column.shape != columns[0].shape
               or column.dtype.kind not in 'fiu' for column in columns):
            raise ValueError('Bank physical parameters must be aligned numeric vectors')
        rows = np.column_stack(columns)
        if not np.isfinite(rows).all() or not len(rows):
            raise ValueError('Bank must contain finite nonempty physical templates')
        if np.any(rows[:, :2] <= 0):
            raise ValueError('Bank masses must be positive')
        distinct = len(np.unique(rows, axis=0))
        if distinct != len(rows):
            raise ValueError('Repeated template rows cannot establish search capacity')
        return dict(templates=len(rows), distinct_templates=distinct,
                    identity_fields=fields)


def resolve_waveform_mode(mode=None, uncompressed=False, track=None):
    """Resolve explicit aliases without changing waveform model or bank data."""
    selections = [value for value in (
        mode, 'generated' if uncompressed else None,
        {'track1': 'compressed', 'track2': 'generated'}.get(track),
    ) if value is not None]
    if any(value not in WAVEFORM_MODES for value in selections):
        raise ValueError('Unknown waveform mode')
    if len(set(selections)) > 1:
        raise ValueError('Conflicting waveform mode, --uncompressed or --track')
    return selections[0] if selections else 'compressed'


def validate_bank_mode(path, mode, approximant, low_frequency_cutoff=30):
    """Reject missing compressed rows instead of allowing generation fallback."""
    inventory = bank_inventory(path)
    with h5py.File(path, 'r') as bank:
        compressed = bank.get('compressed_waveforms')
        stored_approximants = None
        if 'approximant' in bank:
            values = np.asarray(bank['approximant'])
            if values.shape != (inventory['templates'],):
                raise ValueError('Bank approximants must have one value per row')
            stored_approximants = sorted({
                value.decode() if isinstance(value, bytes) else str(value)
                for value in values
            })
        if mode == 'compressed':
            if not isinstance(compressed, h5py.Group):
                raise ValueError('Compressed mode requires a compressed_waveforms group; '
                                 'select --waveform-mode generated explicitly to generate templates')
            if stored_approximants is not None and stored_approximants != [approximant]:
                raise ValueError('Compressed bank approximants do not match --approximant')
            if 'template_hash' not in bank:
                raise ValueError('Compressed bank requires template_hash for every row')
            hashes = np.asarray(bank['template_hash'])
            if (hashes.shape != (inventory['templates'],) or hashes.dtype.kind not in 'iu'
                    or len(np.unique(hashes)) != len(hashes)):
                raise ValueError('Compressed bank template_hash must be a unique integer vector')
            for template_hash in hashes:
                group = compressed.get(str(template_hash))
                required = ('sample_points', 'amplitude', 'phase')
                if not isinstance(group, h5py.Group) or any(
                        not isinstance(group.get(name), h5py.Dataset) for name in required):
                    raise ValueError(f'Missing compressed waveform data for {template_hash}')
                arrays = [np.asarray(group[name]) for name in required]
                if any(array.ndim != 1 or array.shape != arrays[0].shape
                       or array.dtype.kind not in 'fiu' or not np.isfinite(array).all()
                       for array in arrays) or len(arrays[0]) < 2:
                    raise ValueError(f'Invalid compressed waveform arrays for {template_hash}')
                if (np.any(np.diff(arrays[0]) <= 0) or arrays[0][0] < 0
                        or arrays[0][0] > low_frequency_cutoff
                        or arrays[0][-1] <= low_frequency_cutoff):
                    raise ValueError(f'Invalid compressed frequency support for {template_hash}')
                required_attrs = {'interpolation', 'tolerance', 'mismatch', 'precision',
                                  'compression_factor'}
                if not required_attrs <= set(group.attrs):
                    raise ValueError(f'Missing compressed waveform metadata for {template_hash}')
        elif mode != 'generated':
            raise ValueError('Unknown waveform mode')
    return dict(inventory, bank_contains_compressed_waveforms=compressed is not None,
                stored_approximants=stored_approximants)


def _contract_fingerprint(contract):
    payload = {key: value for key, value in contract.items() if key != 'fingerprint'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def input_contract(bank_file, frame_file, mode, approximant, order,
                   decompression_method, search_config=None):
    """Bind waveform handling and scientific inputs independently of backend."""
    search_config = _validate_search_overrides(search_config or {})
    inventory = validate_bank_mode(
        bank_file, mode, approximant, float(search_config.get('low_frequency_cutoff', 30)))
    contract = dict(
        version=1, waveform_mode=mode,
        track='track1_compressed' if mode == 'compressed' else 'track2_generated',
        bank_sha256=file_sha256(Path(bank_file)), frame_sha256=file_sha256(Path(frame_file)),
        bank=inventory, approximant=approximant, order=order,
        decompression_method=decompression_method if mode == 'compressed' else None,
        sample_rate=BENCHMARK_SAMPLE_RATE, precision=BENCHMARK_PRECISION,
        search_config=search_config,
    )
    contract['fingerprint'] = _contract_fingerprint(contract)
    return contract


def _require_matching_contract(reference, candidate):
    for contract in (reference, candidate):
        if not isinstance(contract, dict) or contract.get('fingerprint') != _contract_fingerprint(contract):
            raise ValueError('Missing or invalid input contract fingerprint')
    if reference != candidate:
        raise ValueError('Input contract mismatch; cannot compare different waveform workloads')


def _validate_search_overrides(search_config):
    if not isinstance(search_config, dict):
        raise ValueError('Search configuration must be a JSON object')
    reserved = {'processing-scheme', 'output', 'bank-file', 'frame-files', 'batch-size',
                'fft-backends', 'sample-rate', 'precision', 'dtype', 'approximant', 'order',
                'use-compressed-waveforms', 'waveform-decompression-method',
                'enable-diffgw', 'disable-diffgw', 'waveform-mode', 'verbose'}
    normalized = {}
    for key, value in search_config.items():
        if not isinstance(key, str):
            raise ValueError('Search override names must be strings')
        key = key.replace('-', '_')
        if key.replace('_', '-') in reserved:
            raise ValueError('Use campaign options for --' + key.replace('_', '-'))
        if key in normalized:
            raise ValueError('Duplicate search override: ' + key)
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ValueError('Search overrides must be scalar option values')
        if key == 'jax_chisq_mode' and value not in ('cpu-compatible', 'direct-phase'):
            raise ValueError(
                'jax_chisq_mode must be cpu-compatible or direct-phase'
            )
        normalized[key] = value
    return normalized


def physical_cores(affinity):
    """Resolve Linux CPU affinity to distinct physical cores (SMT counts once)."""
    cpus = set()
    for item in affinity.split(','):
        bounds = item.split('-')
        cpus.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    if not cpus:
        raise ValueError('Empty CPU affinity')
    identities = set()
    for cpu in cpus:
        root = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
        if not root.is_dir():
            raise ValueError('Physical core accounting requires Linux topology')
        identities.add(((root / 'physical_package_id').read_text().strip(),
                        (root / 'core_id').read_text().strip()))
    return len(identities)


def atomic_receipt(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def _source_files(root: Path) -> List[Path]:
    """Files whose contents identify the executable and its providers."""
    files = [root / "bin" / "pycbc_inspiral"]
    files.extend((root / "pycbc").rglob("*.py"))
    files.extend((root / "pycbc").rglob("*.so"))
    files.extend((root / "pycbc").rglob("*.so.*"))
    return sorted({path for path in files if path.is_file()})


def _source_records(original: Path, branch: Path) -> Dict[str, Any]:
    return {
        name: source_identity(root, _source_files(root))
        for name, root in (("original", original), ("branch", branch))
    }


def _require_pristine_original(source_record: Dict[str, Any]) -> None:
    """Require the scientific reference checkout to be a pinned clean revision."""
    if source_record.get("revision") in (None, ""):
        raise RuntimeError("original_cpu source must be a git checkout with a pinned revision")
    if source_record.get("dirty") is not False:
        raise RuntimeError("original_cpu source must be clean; use branch_cpu for modified candidates")


def _require_source_unchanged(root: Path, expected: Dict[str, Any]) -> None:
    """Reject a campaign if its pinned source changed while it was running."""
    observed = source_identity(root, _source_files(root))
    if observed != expected:
        raise RuntimeError("source changed during benchmark")


def _campaign_identity(
    args, source_records: Dict[str, Any], bank_sha: str, frame_sha: str,
    search_config: Dict[str, Any], contract: Dict[str, Any],
) -> Dict[str, Any]:
    """Return the immutable identity used to validate ``--resume``."""
    return {
        "source": source_records,
        "reference_revision": args.reference_revision,
        "observer_sha256": file_sha256(Path(__file__).with_name('observe_pycbc_inspiral.py')),
        "input_contract": contract,
        "inputs": {
            "bank_sha256": bank_sha,
            "frame_sha256": frame_sha,
        },
        "command": {
            "python": str(Path(args.python).resolve()),
            "arms": list(args.arms),
            "replicates": args.replicates,
            "affinity": args.affinity,
            "approximant": args.approximant,
            "order": args.order,
            "uncompressed": args.uncompressed,
            "decompression_method": args.decompression_method,
            "batch_size": args.batch_size,
            "track": args.track,
            "waveform_mode": args.waveform_mode,
            "qualification_only": args.qualification_only,
            "profile_utilization": args.profile_utilization,
            "allow_unqualified_timings": args.allow_unqualified_timings,
            "search_config": search_config,
            "sample_rate": BENCHMARK_SAMPLE_RATE,
            "precision": BENCHMARK_PRECISION,
        },
    }


def _completed_case(run: Dict[str, Any], qualification: bool = False,
                    expected_contract=None) -> bool:
    """Only reuse a complete run with matching process/work receipts."""
    trigger_path = run.get("triggers_path")
    if not trigger_path or not Path(trigger_path).is_file():
        return False
    run_dir = Path(trigger_path).parent
    process_path = run_dir / "process.json"
    command_path = run_dir / "command.json"
    work_path = run_dir / "work.json"
    if not process_path.is_file() or not command_path.is_file() or not work_path.is_file():
        return False
    try:
        process = json.loads(process_path.read_text())
        command = json.loads(command_path.read_text())
        work = json.loads(work_path.read_text())
    except (OSError, ValueError):
        return False
    if process.get("returncode") != 0 or not isinstance(work, dict):
        return False
    if not {"completed_template_seconds", "valid_detector_seconds"} <= set(work):
        return False
    if run.get("command") and command.get("command") != run["command"]:
        return False
    if expected_contract is not None:
        try:
            _require_matching_contract(expected_contract, run.get('input_contract'))
            _require_matching_contract(expected_contract, command.get('input_contract'))
            _validate_benchmark_work(work, expected_contract['bank']['templates'])
        except (ValueError, RuntimeError):
            return False
    if qualification:
        evidence = run.get("evidence_path")
        if not evidence or not Path(evidence).is_file():
            return False
    return True


def _retry_output_dir(path: Path) -> Path:
    """Keep failed logs and place a resumed attempt in a fresh directory."""
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}.retry{index}")
        if not candidate.exists():
            return candidate
        index += 1


def _profile_command_spec(run: Dict[str, Any], profile_dir: Path):
    """Make a fresh profiler command from a completed qualification manifest."""
    trigger_path = run.get("triggers_path")
    if not trigger_path:
        raise RuntimeError("qualified run has no command manifest")
    raw_qualification_dir = Path(trigger_path).parent
    qualification_dir = Path(trigger_path).resolve().parent
    manifest_path = qualification_dir / "command.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"qualified run has no command manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    command = manifest.get("command") or run.get("command")
    environment = manifest.get("environment", {})
    if not isinstance(command, list) or not command:
        raise RuntimeError("qualified run command manifest is missing argv")
    profile_dir = profile_dir.resolve()
    # macOS commonly exposes temporary paths through both /var and /private/var.
    # Rewrite either spelling so the fresh profile cannot target qualification
    # artifacts through a symlink alias.
    old_paths = {str(raw_qualification_dir), str(qualification_dir)}
    new = str(profile_dir)
    command = [str(item) for item in command]
    for old in sorted(old_paths, key=len, reverse=True):
        command = [item.replace(old, new) for item in command]
    environment = {
        str(key): str(value)
        for key, value in environment.items()
    }
    for old in sorted(old_paths, key=len, reverse=True):
        environment = {
            key: value.replace(old, new) for key, value in environment.items()
        }
    # This is injected by profile_jax_gpu_timeline and is part of the actual
    # profiled launch contract, so expose it in the saved command environment.
    environment.setdefault("PYCBC_BENCHMARK_STAGES", "1")
    return command, environment


def _require_profile_inputs_unchanged(frame_file, bank_file, contract):
    """Check the frozen campaign inputs immediately around a profile run."""
    if frame_file is None or bank_file is None or contract is None:
        return
    if (file_sha256(Path(frame_file)) != contract.get("frame_sha256")
            or file_sha256(Path(bank_file)) != contract.get("bank_sha256")):
        raise RuntimeError("Benchmark inputs changed during profile execution")


def _profile_source_for_arm(arm, sources, source_records):
    """Return the source checkout and identity expected for a profile arm."""
    if not sources or not source_records:
        return None, None
    source_key = "original" if arm == "original_cpu" else "branch"
    return sources.get(arm), source_records.get(source_key)


def _run_profile_campaign(
    arms, qualification, output_dir, reference=None, contract=None,
    sources=None, source_records=None, frame_file=None, bank_file=None,
    allow_unqualified=False,
):
    """Run one fresh profile per qualified arm and requalify its outputs.

    Profile output is deliberately treated as a new scientific execution.  A
    timeline alone cannot qualify a run: its trigger and full evidence files
    must pass the same frozen gates as the original qualification.
    """
    try:
        from tools.profile_jax_gpu_timeline import run_command_profiling
    except ModuleNotFoundError:  # script execution from tools/
        from profile_jax_gpu_timeline import run_command_profiling

    results = {}
    for arm in arms:
        source_root, expected_source = _profile_source_for_arm(
            arm, sources, source_records)
        if source_root is not None and expected_source is not None:
            _require_source_unchanged(Path(source_root), expected_source)
        run_contract = qualification[arm].get("input_contract") or contract
        _require_profile_inputs_unchanged(frame_file, bank_file, run_contract)
        profile_dir = Path(output_dir).resolve() / "profiles" / arm
        profile_dir.mkdir(parents=True, exist_ok=True)
        command, environment = _profile_command_spec(qualification[arm], profile_dir)
        timeline_path = profile_dir / "timeline.json"
        output_hdf = profile_dir / "triggers.hdf"
        workload = {
            "arm": arm,
            "sample_rate_hz": BENCHMARK_SAMPLE_RATE,
            "precision": BENCHMARK_PRECISION,
            "input_contract": run_contract,
            "resources": dict(qualification[arm].get("resources", {})),
            "excluded_from_unprofiled_timing": True,
        }
        profile = run_command_profiling(
            command=command,
            output_json=timeline_path,
            output_hdf=output_hdf,
            workload=workload,
            environment=environment,
            cwd=profile_dir,
            # The recorded command already contains its affinity wrapper.
            affinity_core="",
        )
        if profile.get("returncode") != 0:
            raise RuntimeError(
                f"Profile run failed ({profile.get('returncode')}) for arm={arm}"
            )
        completion = profile.get("process_tree", {}).get("completion", {})
        if (not completion.get("root_exited")
                or not completion.get("all_observed_processes_exited")
                or completion.get("remaining_pids") != []):
            raise RuntimeError(
                "Profile process tree did not record all observed process exits "
                f"for arm={arm}: {completion!r}"
            )
        work_path = profile_dir / "work.json"
        profile_triggers = profile_dir / "triggers.hdf"
        profile_evidence = profile_dir / "science.hdf"
        if not profile_triggers.is_file():
            raise RuntimeError(f"Profile run has no trigger output for arm={arm}")
        if not profile_evidence.is_file():
            raise RuntimeError(f"Profile run has no science evidence for arm={arm}")
        if not work_path.is_file():
            raise RuntimeError(f"Profile run has no work receipt for arm={arm}")
        try:
            profile_work = json.loads(work_path.read_text())
            expected_templates = (run_contract or {}).get("bank", {}).get("templates")
            _validate_benchmark_work(profile_work, expected_templates)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Profile run failed the observed 2048 Hz complex64 workload "
                f"gate for arm={arm}: {exc}"
            ) from exc
        profile_science = None
        if reference is not None:
            reference_evidence = reference.get("evidence_path")
            if not reference_evidence or not Path(reference_evidence).is_file():
                raise RuntimeError("Qualified reference science evidence is missing")
            profile_science = compare_scientific_hdf(
                reference["triggers_path"], str(profile_triggers),
                BENCHMARK_SAMPLE_RATE, reference_evidence, str(profile_evidence))
            if (not profile_science.get("passed", False)
                    and not allow_unqualified):
                raise RuntimeError(
                    "Profile scientific qualification failed for "
                    f"arm={arm}: {profile_science.get('missing_gates', [])}"
                )
        _require_profile_inputs_unchanged(frame_file, bank_file, run_contract)
        if source_root is not None and expected_source is not None:
            _require_source_unchanged(Path(source_root), expected_source)
        # Add the complete launch contract to the timeline itself.  The
        # profiler records hashes and command metadata, while this preserves
        # the exact environment and cwd needed to reproduce the run.
        timeline = json.loads(timeline_path.read_text())
        timeline.update({
            "campaign_arm": arm,
            "command": command,
            "environment": environment,
            "cwd": str(profile_dir),
            "excluded_from_unprofiled_timing": True,
            "resources": dict(qualification[arm].get("resources", {})),
            "science": profile_science,
        })
        atomic_receipt(timeline_path, timeline)
        results[arm] = {
            "arm": arm,
            "profile_dir": str(profile_dir),
            "timeline_path": str(timeline_path),
            "command": command,
            "environment": environment,
            "cwd": str(profile_dir),
            "returncode": profile.get("returncode"),
            "summary": profile.get("summary", {}),
            "process_completion": completion,
            "resources": dict(qualification[arm].get("resources", {})),
            "profile_triggers_path": str(profile_triggers),
            "profile_evidence_path": str(profile_evidence),
            "science": profile_science,
            "excluded_from_unprofiled_timing": True,
            "fresh_process": True,
        }
    return results


def _legacy_options(command: List[str]) -> Dict[str, List[str | None]]:
    """Parse long options while retaining repeated values and flag options."""
    options: Dict[str, List[str | None]] = {}
    index = 0
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            index += 1
            continue
        value = None
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            value = command[index + 1]
            index += 1
        options.setdefault(token, []).append(value)
        index += 1
    return options


def _legacy_arm(manifest_path: Path) -> str:
    """Infer the arm from a qualification or timing manifest location."""
    parts = manifest_path.parts
    for marker in ("qualification", "runs"):
        if marker in parts:
            index = parts.index(marker) + 1
            if index < len(parts):
                name = parts[index]
                if marker == "runs":
                    name = re.sub(r"\.retry\d+$", "", name)
                    name = re.sub(r"_rep\d+$", "", name)
                if name in ARM_NAMES:
                    return name
    raise ValueError(f"Cannot determine campaign arm from {manifest_path}")


def _same_executable(left: str, right: Path) -> bool:
    """Compare executable paths through symlinks (venv python names vary)."""
    try:
        return Path(left).resolve() == right.resolve()
    except (OSError, RuntimeError):
        return os.path.realpath(left) == os.path.realpath(str(right))


def _legacy_expected_options(
    args, arm: str, frame_file: Path, bank_file: Path, output_path: str,
    search_config: Dict[str, Any],
) -> Dict[str, List[str | None]]:
    """Build the option set emitted by the current campaign harness.

    This intentionally mirrors the fixed scientific CLI in ``_run_single_case``.
    Keeping the complete set here makes migration reject stale search settings,
    while output paths remain relocatable within a checkpoint directory.
    """
    scheme = "cpu:1" if arm in ("original_cpu", "branch_cpu") else (
        "jax:cpu" if "cpu" in arm else "jax:cuda:0")
    values = [
        ("--verbose", None), ("--frame-files", str(frame_file)),
        ("--channel-name", "H1:LOSC-STRAIN"),
        ("--gps-start-time", "1187007048"), ("--gps-end-time", "1187009080"),
        ("--trig-start-time", "1187007160"), ("--trig-end-time", "1187009064"),
        ("--sample-rate", str(int(BENCHMARK_SAMPLE_RATE))), ("--low-frequency-cutoff", "30"),
        ("--strain-high-pass", "25"), ("--pad-data", "8"),
        ("--autogating-threshold", "100"), ("--autogating-cluster", "5"),
        ("--autogating-width", "0.25"), ("--autogating-taper", "0.25"),
        ("--autogating-pad", "16"), ("--autogating-max-iterations", "1"),
        ("--psd-estimation", "median"), ("--psd-segment-length", "32"),
        ("--psd-segment-stride", "16"), ("--psd-num-segments", "126"),
        ("--psd-inverse-length", "16"), ("--invpsd-trunc-method", "hann"),
        ("--invpsd-trunc-which-spectrum", "invasd"),
        ("--approximant", args.approximant), ("--order", str(args.order)),
        ("--snr-threshold", "5.5"), ("--newsnr-threshold", "5"),
        ("--chisq-bins", "16"), ("--cluster-window", "1"),
        ("--cluster-function", "symmetric"),
        ("--fft-backends", "jax" if "jax" in scheme else "mkl"),
        ("--bank-file", str(bank_file)), ("--segment-length", "512"),
        ("--segment-start-pad", "112"), ("--segment-end-pad", "16"),
        ("--processing-scheme", scheme), ("--output", output_path),
    ]
    if scheme.startswith("jax"):
        values.append(("--jax-chisq-mode", search_config.get(
            "jax_chisq_mode", "cpu-compatible")))
    if "_batched" in arm:
        batch = 16 if arm == "jax_cpu_batched" else args.batch_size
    else:
        batch = 1
    if batch > 1:
        values.append(("--batch-size", str(batch)))
    if not args.uncompressed:
        values.extend([
            ("--use-compressed-waveforms", None),
            ("--waveform-decompression-method", args.decompression_method),
        ])
    expected = {key: [value] for key, value in values}
    for key, value in search_config.items():
        flag = "--" + key.replace("_", "-")
        if flag in expected and expected[flag] != [None]:
            expected[flag] = [str(value)]
    return expected


def _verify_legacy_manifests(
    output_dir: Path, args, frame_file: Path, bank_file: Path,
    search_config: Dict[str, Any] | None = None,
) -> None:
    """Validate old checkpoints before explicitly migrating their identity.

    Legacy receipts contain no source hashes, so this check is deliberately
    strict about every scientific CLI option.  Migration remains opt-in; the
    caller is responsible for having verified the frozen source checkout.
    """
    search_config = search_config or {}
    manifests = sorted(output_dir.rglob("command.json"))
    if not manifests:
        raise ValueError("Legacy checkpoint contains no command manifests")
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unreadable resume manifest: {manifest_path}") from exc
        command = [str(item) for item in manifest.get("command", [])]
        if not command:
            raise ValueError(f"Resume manifest has no command: {manifest_path}")
        arm = _legacy_arm(manifest_path)
        expected_source = (args.original_source if arm == "original_cpu"
                           else args.branch_source).resolve() / "bin" / "pycbc_inspiral"
        executable_tokens = [token for token in command if token.endswith("pycbc_inspiral")]
        if len(executable_tokens) != 1 or not _same_executable(executable_tokens[0], expected_source):
            raise ValueError(f"Resume source mismatch in {manifest_path}")
        executable_index = command.index(executable_tokens[0])
        expected_prefix = (["taskset", "-c", args.affinity]
                           if sys.platform == "linux" and shutil.which("taskset")
                           else [])
        if command[:executable_index - 1] != expected_prefix:
            raise ValueError(f"Resume affinity launcher mismatch in {manifest_path}")
        python_tokens = [
            token for token in command[:executable_index]
            if Path(token).name.startswith("python")
        ]
        python_candidates = [Path(args.python), Path(sys.executable)]
        if not python_tokens or not any(
            any(_same_executable(token, candidate) for candidate in python_candidates)
            for token in python_tokens
        ):
            raise ValueError(f"Resume Python mismatch in {manifest_path}")
        options = _legacy_options(command)
        if "--enable-diffgw" in options:
            raise ValueError(f"Unsupported diffgw option in {manifest_path}")
        # The obsolete flag was removed from the current executable.  It is
        # the only tolerated command-line difference during migration.
        options.pop("--disable-diffgw", None)
        output_values = options.pop("--output", [])
        if len(output_values) != 1 or output_values[0] is None:
            raise ValueError(f"Resume output mismatch in {manifest_path}")
        output_path = Path(output_values[0])
        try:
            output_path.resolve().relative_to(output_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Resume output escapes checkpoint: {manifest_path}") from exc
        expected = _legacy_expected_options(
            args, arm, frame_file, bank_file, str(output_values[0]), search_config
        )
        expected.pop("--output", None)
        if options != expected:
            raise ValueError(f"Resume command/config mismatch in {manifest_path}")


def _run_single_case(
    arm: str,
    source_root: Path,
    python_bin: str,
    output_dir: Path,
    frame_file: Path,
    bank_file: Path,
    affinity: str = "8",
    approximant: str = "IMRPhenomD",
    order: int = -1,
    use_compressed_waveforms: bool = True,
    waveform_decompression_method: str = "inline_linear",
    batch_size: int = 64,
    search_config=None,
    qualification=False,
    expected_contract=None,
) -> Dict[str, Any]:
    """Execute a single unprofiled run of pycbc_inspiral."""
    if arm.endswith('_diffgw'):
        raise ValueError("Explicit diffgw arms are unsupported by this campaign")
    contract = input_contract(
        bank_file, frame_file, 'compressed' if use_compressed_waveforms else 'generated',
        approximant, order, waveform_decompression_method, search_config)
    if expected_contract is not None:
        _require_matching_contract(expected_contract, contract)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    triggers_hdf = output_dir / "triggers.hdf"
    time_txt = output_dir / "time.txt"
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    source_root = source_root.resolve()
    frame_file = frame_file.resolve()
    bank_file = bank_file.resolve()

    if arm in ("original_cpu", "branch_cpu"):
        scheme = "cpu:1"
    elif arm in ("jax_cpu", "jax_cpu_batched", "jax_cpu_diffgw"):
        scheme = "jax:cpu"
    elif arm in ("jax_cuda_lal", "jax_cuda", "jax_cuda_batched", "jax_cuda_diffgw"):
        scheme = "jax:cuda:0"
    else:
        raise ValueError(f"Unknown arm: {arm}")

    executable = source_root / "bin" / "pycbc_inspiral"
    if arm == 'original_cpu':
        validate_observer_source(executable)

    cli_args = [
        "--verbose",
        "--frame-files",
        str(frame_file),
        "--channel-name",
        "H1:LOSC-STRAIN",
        "--gps-start-time",
        "1187007048",
        "--gps-end-time",
        "1187009080",
        "--trig-start-time",
        "1187007160",
        "--trig-end-time",
        "1187009064",
        "--sample-rate",
        str(int(BENCHMARK_SAMPLE_RATE)),
        "--low-frequency-cutoff",
        "30",
        "--strain-high-pass",
        "25",
        "--pad-data",
        "8",
        "--autogating-threshold",
        "100",
        "--autogating-cluster",
        "5",
        "--autogating-width",
        "0.25",
        "--autogating-taper",
        "0.25",
        "--autogating-pad",
        "16",
        "--autogating-max-iterations",
        "1",
        "--psd-estimation",
        "median",
        "--psd-segment-length",
        "32",
        "--psd-segment-stride",
        "16",
        "--psd-num-segments",
        "126",
        "--psd-inverse-length",
        "16",
        "--invpsd-trunc-method",
        "hann",
        "--invpsd-trunc-which-spectrum",
        "invasd",
        "--approximant",
        approximant,
        "--order",
        str(order),
        "--snr-threshold",
        "5.5",
        "--newsnr-threshold",
        "5",
        "--chisq-bins",
        "16",
        "--cluster-window",
        "1",
        "--cluster-function",
        "symmetric",
        "--fft-backends",
        "jax" if "jax" in scheme else "mkl",
        "--bank-file",
        str(bank_file),
        "--segment-length",
        "512",
        "--segment-start-pad",
        "112",
        "--segment-end-pad",
        "16",
        "--processing-scheme",
        scheme,
        "--output",
        str(triggers_hdf),
    ]
    if scheme.startswith("jax"):
        cli_args.extend([
            "--jax-chisq-mode",
            str((search_config or {}).get("jax_chisq_mode", "cpu-compatible")),
        ])

    if "_batched" in arm:
        eff_batch_size = 16 if arm == "jax_cpu_batched" else batch_size
    elif arm in ("jax_cuda_diffgw", "jax_cpu_diffgw"):
        eff_batch_size = batch_size
    else:
        eff_batch_size = 1

    if eff_batch_size > 1:
        cli_args.extend(["--batch-size", str(eff_batch_size)])

    if use_compressed_waveforms:
        cli_args.extend([
            "--use-compressed-waveforms",
            "--waveform-decompression-method",
            waveform_decompression_method,
        ])
    else:
        if arm in ("jax_cpu", "jax_cpu_batched", "jax_cuda_lal", "jax_cuda", "jax_cuda_batched"):
            # The current executable defaults this provider off for reference
            # arms; its removed --disable-diffgw flag must not be passed.
            pass
        elif arm in ("jax_cuda_diffgw", "jax_cpu_diffgw"):
            raise ValueError("Explicit diffgw arms are unsupported by this campaign")

    search_config = _validate_search_overrides(search_config or {})
    for key, value in search_config.items():
        if key == 'jax_chisq_mode' and not scheme.startswith('jax'):
            # Shared search configuration is also used for the CPU reference;
            # its CLI intentionally omits the JAX-only mode option.
            continue
        flag = '--' + key.replace('_', '-')
        if flag not in cli_args:
            raise ValueError('Unsupported search setting ' + flag)
        cli_args[cli_args.index(flag) + 1] = str(value)
    science_config = {cli_args[i]: cli_args[i + 1] for i in range(len(cli_args) - 1)
                      if cli_args[i].startswith('--') and not cli_args[i + 1].startswith('--')}
    for key in ('--processing-scheme', '--output', '--batch-size', '--fft-backends'):
        science_config.pop(key, None)
    sample_rate = float(science_config['--sample-rate'])
    if sample_rate != BENCHMARK_SAMPLE_RATE:
        raise ValueError('The inspiral campaign is fixed at 2048 Hz')
    science_config['benchmark_precision'] = BENCHMARK_PRECISION
    science_config['input_contract'] = contract
    command = [
        python_bin,
        *([str(Path(__file__).with_name('observe_pycbc_inspiral.py').resolve())]
          if arm == 'original_cpu' else []),
        str(executable),
        *cli_args,
    ]

    # Prepend taskset on Linux systems if available
    if sys.platform == "linux" and shutil.which("taskset"):
        command = ["taskset", "-c", affinity] + command

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MKL_DYNAMIC": "FALSE",
            "MKL_THREADING_LAYER": "GNU",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(source_root),
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
            "JAX_COMPILATION_CACHE_DIR": "off",
        }
    )

    env.update(PYCBC_BENCHMARK_WORK=str(output_dir / 'work.json'),
               PYCBC_BENCHMARK_SCIENCE_CONFIG=json.dumps(science_config, sort_keys=True),
               XLA_PYTHON_CLIENT_PREALLOCATE='false',
               JAX_ENABLE_COMPILATION_CACHE='false',
               JAX_COMPILATION_CACHE_DIR='off')
    env.pop('PYCBC_BENCHMARK_EVIDENCE', None)
    env.pop('PYCBC_BENCHMARK_STAGES', None)
    if qualification:
        env['PYCBC_BENCHMARK_EVIDENCE'] = str(output_dir / 'science.hdf')
    manifest = dict(command=command, cwd=str(output_dir), input_contract=contract,
                    environment={k: v for k, v in env.items() if k in (
                        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'PYTHONPATH', 'XLA_PYTHON_CLIENT_PREALLOCATE',
                        'JAX_ENABLE_COMPILATION_CACHE', 'JAX_COMPILATION_CACHE_DIR')
                        or k.startswith('PYCBC_BENCHMARK_')})
    atomic_receipt(output_dir / 'command.json', manifest)
    t0 = time.perf_counter()
    with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
        p = subprocess.run(
            command,
            cwd=str(output_dir),
            env=env,
            stdout=out_f,
            stderr=err_f,
            check=False,
        )
    elapsed_wall = time.perf_counter() - t0

    atomic_receipt(output_dir / 'process.json', dict(
        **manifest, returncode=p.returncode, elapsed_wall_sec=elapsed_wall))
    if p.returncode != 0:
        stderr_sample = stderr_path.read_text()[-1000:] if stderr_path.exists() else ""
        raise RuntimeError(
            f"Run failed ({p.returncode}) for arm={arm}:\n{stderr_sample}"
        )
    if (file_sha256(bank_file) != contract['bank_sha256']
            or file_sha256(frame_file) != contract['frame_sha256']):
        raise RuntimeError('Benchmark inputs changed during execution')

    with h5py.File(triggers_hdf, "r") as hf:
        search_grp = hf["H1/search"]
        run_time_sec = float(search_grp["run_time"][0])
        setup_frac = float(search_grp["setup_time_fraction"][0])
        num_triggers = len(hf["H1/snr"]) if "H1/snr" in hf else 0
        snr_dtype = str(hf["H1/snr"].dtype) if "H1/snr" in hf else None
    if snr_dtype is not None and np.dtype(snr_dtype) != np.dtype(np.float32):
        raise RuntimeError(
            f"inspiral output SNR magnitude dtype {snr_dtype} is not the required float32"
        )

    tsetup_sec = run_time_sec * setup_frac
    calc_time_sec = run_time_sec - tsetup_sec

    time_info = {}
    if time_txt.exists():
        for line in time_txt.read_text().splitlines():
            if "User time (seconds):" in line:
                time_info["user_time_sec"] = float(line.split(":")[-1].strip())
            elif "System time (seconds):" in line:
                time_info["system_time_sec"] = float(line.split(":")[-1].strip())
            elif "Maximum resident set size (kbytes):" in line:
                time_info["max_rss_kib"] = int(line.split(":")[-1].strip())

    stderr_lines = stderr_path.read_text().splitlines() if stderr_path.exists() else []
    phases = _parse_stderr_phases(
        stderr_lines, elapsed_wall, calc_time_sec, tsetup_sec
    )

    work_path = output_dir / 'work.json'
    work = json.loads(work_path.read_text()) if work_path.exists() else {}
    signal_dtypes = _validate_benchmark_work(work, contract['bank']['templates'])
    return {
        **work,
        "command": command,
        "input_contract": contract,
        "sample_rate": sample_rate,
        "precision": BENCHMARK_PRECISION,
        "signal_dtypes": signal_dtypes,
        "observed_snr_magnitude_dtype": snr_dtype,
        "resources": {"physical_cpu_cores": physical_cores(affinity),
                      "gpus": 1 if 'cuda' in scheme else 0},
        "evidence_path": str(output_dir / 'science.hdf') if qualification else None,
        "qualification_run": qualification,
        "phase_method": "legacy log estimates; diagnostic only",
        "arm": arm,
        "scheme": scheme,
        "elapsed_wall_sec": elapsed_wall,
        "run_time_sec": run_time_sec,
        "calc_time_sec": calc_time_sec,
        "tsetup_sec": tsetup_sec,
        "num_triggers": num_triggers,
        "time_info": time_info,
        "phases": phases,
        "triggers_path": str(triggers_hdf),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _validate_benchmark_work(work, expected_templates=None):
    """Require the executable's observed 2048 Hz complex64 work receipt."""
    # ``snr`` is a stored magnitude and is therefore real float32 even when
    # the matched-filter signal path is complex64.  The executable writes the
    # latter to the benchmark work receipt; require that receipt so a run
    # cannot be labelled complex64 from configuration alone.
    signal_dtypes = work.get('signal_dtypes')
    if signal_dtypes != [BENCHMARK_PRECISION]:
        raise RuntimeError(
            'inspiral benchmark signal_dtypes must be ["complex64"], '
            f'got {signal_dtypes!r}'
        )
    observed_sample_rate = work.get('sample_rate')
    if observed_sample_rate != BENCHMARK_SAMPLE_RATE:
        raise RuntimeError(
            'inspiral benchmark work sample_rate must be 2048 Hz, '
            f'got {observed_sample_rate!r}'
        )
    if expected_templates is not None:
        seconds = work.get('valid_detector_seconds')
        completed = work.get('completed_template_seconds')
        if (work.get('completed_templates') != expected_templates
                or not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds <= 0
                or not isinstance(completed, (int, float)) or not np.isfinite(completed)
                or completed != expected_templates * seconds):
            raise RuntimeError('Incomplete or inconsistent template work receipt')
    return signal_dtypes


def compare_campaign_parity(raw_results):
    """Check every repeat, including CPU repeatability, without nearest matches."""
    # Scientific parity is always anchored to the pristine pinned checkout.
    # branch_cpu is a candidate arm and must never redefine the reference.
    reference_arm = 'original_cpu' if raw_results.get('original_cpu') else None
    if reference_arm is None:
        return None, {}
    reference = Path(raw_results[reference_arm][0]['triggers_path'])
    contract = raw_results[reference_arm][0].get('input_contract')
    for runs in raw_results.values():
        for run in runs:
            _require_matching_contract(contract, run.get('input_contract'))
    results = {arm: [compare_trigger_parity(reference, Path(run['triggers_path']),
                                          run.get('sample_rate', BENCHMARK_SAMPLE_RATE))
                     for run in runs]
               for arm, runs in raw_results.items()}
    return reference_arm, results


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-arm pycbc_inspiral reference campaign benchmark for JAX."
    )
    parser.add_argument(
        "--original-source",
        type=Path,
        required=True,
        help="Pinned baseline PyCBC repo root for original_cpu",
    )
    parser.add_argument(
        "--reference-revision",
        required=True,
        help="Expected full 40-character commit SHA of the unchanged CPU reference",
    )
    parser.add_argument(
        "--branch-source",
        type=Path,
        required=True,
        help="Candidate PyCBC repo root",
    )
    parser.add_argument(
        "--python",
        type=str,
        required=True,
        help="Path to python executable with virtualenv",
    )
    parser.add_argument(
        "--frame-file",
        type=Path,
        required=True,
        help="H1 GWF frame file",
    )
    parser.add_argument(
        "--bank-file",
        type=Path,
        required=True,
        help="HDF bank of distinct physical templates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for benchmark receipt JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store runs and intermediate artifacts",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help="Number of counterbalanced replicates (default: 3)",
    )
    parser.add_argument(
        "--affinity",
        type=str,
        default="8",
        help="CPU affinity core (default: 8)",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARM_NAMES,
        default=list(DEFAULT_ARMS),
        help="Benchmark arms to execute; standard CPU references are scalar (no branch_cpu_batched arm)",
    )
    parser.add_argument(
        "--track",
        type=str,
        choices=["track1", "track2"],
        default=None,
        help="Waveform-mode alias: track1 (compressed), track2 (generated); does not change approximant/order",
    )
    parser.add_argument(
        "--approximant",
        type=str,
        default="IMRPhenomD",
        help="Waveform approximant (default: IMRPhenomD)",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=-1,
        help="Waveform phase PN order (default: -1)",
    )
    parser.add_argument(
        "--waveform-mode",
        choices=WAVEFORM_MODES,
        default=None,
        help="compressed (default, requires complete stored waveforms) or explicitly generated",
    )
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        default=False,
        help="Explicit legacy alias for --waveform-mode generated",
    )
    parser.add_argument(
        "--decompression-method",
        type=str,
        default="inline_linear",
        help="Decompression method when running compressed waveforms (default: inline_linear)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="CUDA template batch size for batched arms (default: 64); JAX CPU batched uses 16",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume successful qualification/runs from the existing output receipt",
    )
    parser.add_argument(
        "--migrate-legacy-checkpoint",
        action="store_true",
        help="Explicitly migrate a pre-identity checkpoint after strict manifest validation",
    )
    parser.add_argument(
        "--batched",
        action="store_true",
        default=False,
        help="Run JAX batched arms with scalar CPU references (original_cpu, branch_cpu, jax_cpu_batched, jax_cuda_batched)",
    )
    parser.add_argument(
        "--qualification-only",
        action="store_true",
        help="Run full science qualification for every arm with zero timed repetitions",
    )
    parser.add_argument(
        "--allow-unqualified-timings",
        action="store_true",
        help=("Run explicitly diagnostic timings after failed qualification; "
              "never emit an equivalence-qualified performance claim"),
    )
    parser.add_argument(
        "--profile-utilization",
        action="store_true",
        help="Run one fresh per-arm GPU/CPU timeline campaign after qualification; exclude it from unprofiled timing",
    )
    parser.add_argument('--search-config', type=Path,
                        help='JSON overrides for the fixed scientific CLI settings')
    args = parser.parse_args()
    unsupported_diffgw = [arm for arm in args.arms if arm.endswith("_diffgw")]
    if unsupported_diffgw:
        parser.error(
            "diffgw arms are unsupported by this executable: "
            + ", ".join(unsupported_diffgw)
        )
    if args.replicates < 3 and not (args.qualification_only or args.profile_utilization):
        parser.error('At least three fresh timing repetitions are required')
    if 'original_cpu' not in args.arms:
        parser.error('original_cpu is required as the pristine scientific reference')
    search_config = json.loads(args.search_config.read_text()) if args.search_config else {}
    try:
        args.waveform_mode = resolve_waveform_mode(args.waveform_mode, args.uncompressed, args.track)
        search_config = _validate_search_overrides(search_config)
    except ValueError as exc:
        parser.error(str(exc))
    args.uncompressed = args.waveform_mode == 'generated'
    args.track = 'track2_generated' if args.uncompressed else 'track1_compressed'
    if args.batched and args.arms == list(DEFAULT_ARMS):
        args.arms = list(BATCHED_ARMS)

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output = args.output.resolve()
    args.frame_file = args.frame_file.resolve()
    args.bank_file = args.bank_file.resolve()

    sources = {
        "original_cpu": args.original_source.resolve(),
        "branch_cpu": args.branch_source.resolve(),
        "jax_cpu": args.branch_source.resolve(),
        "jax_cpu_batched": args.branch_source.resolve(),
        "jax_cpu_diffgw": args.branch_source.resolve(),
        "jax_cuda_lal": args.branch_source.resolve(),
        "jax_cuda": args.branch_source.resolve(),
        "jax_cuda_batched": args.branch_source.resolve(),
        "jax_cuda_diffgw": args.branch_source.resolve(),
    }

    if not args.frame_file.is_file():
        raise FileNotFoundError(f"Frame file not found: {args.frame_file}")
    if not args.bank_file.is_file():
        raise FileNotFoundError(f"Bank file not found: {args.bank_file}")

    bank_sha = file_sha256(args.bank_file)
    frame_sha = file_sha256(args.frame_file)
    contract = input_contract(
        args.bank_file, args.frame_file, args.waveform_mode, args.approximant,
        args.order, args.decompression_method, search_config)
    inventory = contract['bank']
    validate_reference(args.original_source, args.reference_revision)
    validate_observer_source(args.original_source / 'bin' / 'pycbc_inspiral')

    source_commits = {
        "original": _git_commit(args.original_source),
        "branch": _git_commit(args.branch_source),
    }
    source_records = _source_records(args.original_source, args.branch_source)
    _require_pristine_original(source_records["original"])
    identity = _campaign_identity(
        args, source_records, bank_sha, frame_sha, search_config, contract
    )

    print("=" * 80)
    print("STARTING MULTI-ARM JAX PYCBC_INSPIRAL REFERENCE CAMPAIGN BENCHMARK")
    print(f"Arms:            {args.arms}")
    print(f"Replicates:      {args.replicates}")
    print(f"Baseline Commit: {source_commits['original']}")
    print(f"Branch Commit:   {source_commits['branch']}")
    print(f"Bank SHA256:     {bank_sha}")
    print(f"Frame SHA256:    {frame_sha}")
    print(f"Affinity Core:   {args.affinity}")
    print("=" * 80, flush=True)

    if args.resume:
        if not args.output.is_file():
            parser.error(f"--resume requires an existing checkpoint: {args.output}")
        checkpoint = json.loads(args.output.read_text())
        if checkpoint.get("status") == "complete":
            parser.error("Checkpoint is already complete; choose a new output")
        old_identity = checkpoint.get("identity")
        if old_identity is not None:
            if old_identity != identity:
                parser.error("Checkpoint identity mismatch; refusing unsafe resume")
        else:
            if not args.migrate_legacy_checkpoint:
                parser.error(
                    "Checkpoint has no campaign identity; refusing legacy resume. "
                    "Verify the frozen source/input/config and retry with "
                    "--migrate-legacy-checkpoint"
                )
            try:
                _verify_legacy_manifests(
                    args.output_dir, args, args.frame_file, args.bank_file,
                    search_config,
                )
            except ValueError as exc:
                parser.error(str(exc))
            old_workload = checkpoint.get("workload", {})
            if old_workload.get("bank_sha256") not in (None, bank_sha):
                parser.error("Checkpoint bank hash mismatch; refusing resume")
            if old_workload.get("frame_sha256") not in (None, frame_sha):
                parser.error("Checkpoint frame hash mismatch; refusing resume")
            checkpoint["identity"] = identity
        raw_results = checkpoint.setdefault("raw_results", {})
        for arm in args.arms:
            raw_results.setdefault(arm, [])
            for index, run in enumerate(raw_results[arm], start=1):
                run.setdefault("case_name", f"{arm}_rep{index}")
                run.setdefault("replicate", index)
        qualification = checkpoint.setdefault("qualification", {})
        checkpoint.setdefault("science", {})
        checkpoint.setdefault("profile_results", {})
        checkpoint["status"] = "running"
        atomic_receipt(args.output, checkpoint)
    else:
        raw_results = {arm: [] for arm in args.arms}
        checkpoint = dict(
            schema_version=4,
            executable="pycbc_inspiral",
            status="running",
            workload=inventory,
            input_contract=contract,
            raw_results=raw_results,
            qualification={},
            science={},
            profile_results={},
            identity=identity,
        )
        atomic_receipt(args.output, checkpoint)
        qualification = checkpoint["qualification"]
    for arm in args.arms:
        validate_reference(args.original_source, args.reference_revision)
        if arm in qualification and _completed_case(
                qualification[arm], qualification=True, expected_contract=contract):
            print("Reusing completed qualification " + arm, flush=True)
            continue
        print('Qualifying ' + arm, flush=True)
        qual_dir = args.output_dir / "qualification" / arm
        if args.resume:
            qual_dir = _retry_output_dir(qual_dir)
        try:
            qualification[arm] = _run_single_case(
                arm, sources[arm], args.python, qual_dir,
                args.frame_file, args.bank_file, affinity=args.affinity,
                approximant=args.approximant, order=args.order,
                use_compressed_waveforms=not args.uncompressed,
                waveform_decompression_method=args.decompression_method,
                batch_size=args.batch_size, search_config=search_config, qualification=True,
                expected_contract=contract)
        except Exception as exc:
            checkpoint.update(status="failed", failure={
                "phase": "qualification", "arm": arm,
                "output_dir": str(qual_dir), "error": str(exc),
            })
            atomic_receipt(args.output, checkpoint)
            raise
        checkpoint['qualification'] = qualification
        atomic_receipt(args.output, checkpoint)
    _require_source_unchanged(args.original_source, source_records["original"])
    validate_reference(args.original_source, args.reference_revision)
    reference_arm = 'original_cpu'
    ref = qualification[reference_arm]
    try:
        for arm, run in qualification.items():
            _require_matching_contract(contract, run.get('input_contract'))
            evidence = [ref.get('evidence_path'), run.get('evidence_path')]
            evidence = [x if x and Path(x).exists() else None for x in evidence]
            checkpoint['science'][arm] = compare_scientific_hdf(
                ref['triggers_path'], run['triggers_path'], run['sample_rate'], *evidence)
    except Exception as exc:
        checkpoint.update(status="failed", failure={
            "phase": "qualification_science", "error": str(exc),
        })
        atomic_receipt(args.output, checkpoint)
        raise
    atomic_receipt(args.output, checkpoint)
    failed_arms = [arm for arm in args.arms if not checkpoint['science'][arm]['passed']]
    if failed_arms and not args.allow_unqualified_timings:
        checkpoint.update(status='failed', failure={
            'phase': 'qualification_science', 'arms': failed_arms,
            'error': 'Scientific qualification failed; no timing runs launched',
        })
        atomic_receipt(args.output, checkpoint)
        raise RuntimeError(checkpoint['failure']['error'])
    timing_policy = {
        "qualification_required": True,
        "allow_unqualified_timings": bool(args.allow_unqualified_timings),
        "qualification_only": bool(args.qualification_only),
        "failed_qualification_arms": failed_arms,
        "classification": ("known_divergence_diagnostic"
                           if args.allow_unqualified_timings else
                           "equivalent_output"),
    }
    checkpoint["timing_policy"] = timing_policy
    atomic_receipt(args.output, checkpoint)

    # Profiling is a distinct fresh-process campaign.  It is intentionally
    # completed before any optional unprofiled timing, and its wall times are
    # never admitted to benchmark summaries.
    profile_results = checkpoint.setdefault("profile_results", {})
    if args.profile_utilization:
        try:
            profile_results = _run_profile_campaign(
                args.arms, qualification, args.output_dir,
                reference=ref, contract=contract, sources=sources,
                source_records=source_records, frame_file=args.frame_file,
                bank_file=args.bank_file,
                allow_unqualified=args.allow_unqualified_timings,
            )
            checkpoint["profile_results"] = profile_results
            atomic_receipt(args.output, checkpoint)
        except Exception as exc:
            checkpoint.update(status="failed", failure={
                "phase": "profile_utilization", "error": str(exc),
            })
            atomic_receipt(args.output, checkpoint)
            raise
        _require_source_unchanged(args.original_source, source_records["original"])
        validate_reference(args.original_source, args.reference_revision)

    # Qualification-only and profiling modes make no unprofiled performance
    # claim.  In profile mode the effective profile repetition count is one
    # per arm even if --replicates retained its normal value on the command.
    if args.qualification_only or args.profile_utilization:
        science = {
            arm: dict(checkpoint["science"][arm]) for arm in args.arms
        }
        status = ("science_qualification_only" if args.qualification_only
                  else "profiled_science_qualification")
        science_qualified = all(item.get("passed", False) for item in science.values())
        profile_science = {
            arm: result.get("science")
            for arm, result in profile_results.items()
            if result.get("science") is not None
        }
        if args.profile_utilization:
            science_qualified = science_qualified and (
                set(profile_science) == set(args.arms)
                and all(item.get("passed", False) for item in profile_science.values())
            )
        receipt = {
            "schema_version": 4,
            "executable": "pycbc_inspiral",
            "sample_rate": BENCHMARK_SAMPLE_RATE,
            "precision": BENCHMARK_PRECISION,
            "status": status,
            "performance_claim": False,
            "timing_policy": timing_policy,
            "campaign": {
                "status": status,
                "process_complete": True,
                "science_qualified": science_qualified,
                "passed": (science_qualified and
                           not args.allow_unqualified_timings),
                "performance_claim": False,
                "known_divergence": bool(args.allow_unqualified_timings),
                "profile_science": profile_science,
            },
            "timing": {
                "replicates": 0,
                "timing_runs_launched": 0,
                "excluded_profile_runs": bool(args.profile_utilization),
            },
            "profile_utilization": {
                "requested": bool(args.profile_utilization),
                "effective_replicates": 1 if args.profile_utilization else 0,
                "science": profile_science,
                "results": profile_results,
            },
            "profile_results": profile_results,
            "qualification": qualification,
            "science": science,
            "profile_science": profile_science,
            "source": source_records,
            "identity": identity,
            "input_contract": contract,
            "runtime": runtime_metadata(),
            "environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONHASHSEED": "0",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "JAX_ENABLE_COMPILATION_CACHE": "false",
                "JAX_COMPILATION_CACHE_DIR": "off",
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "workload": {
                **inventory,
                "search_config": search_config,
                "track": args.track,
                "waveform_mode": args.waveform_mode,
                "frame_file": str(args.frame_file),
                "frame_sha256": frame_sha,
                "bank_file": str(args.bank_file),
                "bank_sha256": bank_sha,
                "approximant": args.approximant,
                "order": args.order,
                "uncompressed": args.uncompressed,
                "decompression_method": contract['decompression_method'],
                "batch_size": args.batch_size,
                "sample_rate": BENCHMARK_SAMPLE_RATE,
                "precision": BENCHMARK_PRECISION,
            },
            "provenance": {
                "original_source": str(args.original_source),
                "branch_source": str(args.branch_source),
                "original_commit": source_commits["original"],
                "branch_commit": source_commits["branch"],
                "python_executable": args.python,
                "affinity_core": args.affinity,
            },
            "replicates_requested": args.replicates,
            "arms": args.arms,
            "summaries": {},
            "parity_reference_arm": None,
            "parity_validation": {},
            "raw_results": {arm: [] for arm in args.arms},
        }
        atomic_receipt(args.output, receipt)
        print("=" * 80)
        print(f"Receipt successfully written to: {args.output}")
        print("No unprofiled timing runs or performance claims were recorded.")
        print("=" * 80)
        return 0

    orderings = [
        ["original_cpu", "branch_cpu", "jax_cpu", "jax_cpu_batched", "jax_cpu_diffgw", "jax_cuda_lal", "jax_cuda", "jax_cuda_batched", "jax_cuda_diffgw"],
        ["jax_cuda_diffgw", "jax_cuda_batched", "jax_cuda", "jax_cuda_lal", "jax_cpu_diffgw", "jax_cpu_batched", "jax_cpu", "branch_cpu", "original_cpu"],
        ["branch_cpu", "jax_cuda_lal", "jax_cuda_diffgw", "jax_cuda_batched", "original_cpu", "jax_cpu_diffgw", "jax_cuda", "jax_cpu_batched", "jax_cpu"],
    ]

    run_count = 0
    total_runs = args.replicates * len(args.arms)
    for rep in range(args.replicates):
        order = [a for a in orderings[rep % len(orderings)] if a in args.arms]
        for a in args.arms:
            if a not in order:
                order.append(a)
        for arm in order:
            validate_reference(args.original_source, args.reference_revision)
            run_count += 1
            case_name = f"{arm}_rep{rep + 1}"
            case_dir = args.output_dir / "runs" / case_name
            prior = next(
                (run for run in raw_results.get(arm, [])
                 if run.get("case_name") == case_name),
                None,
            )
            if prior is not None and _completed_case(prior, expected_contract=contract):
                print(f"[{run_count:2d}/{total_runs}] Reusing {case_name}", flush=True)
                continue
            if args.resume:
                case_dir = _retry_output_dir(case_dir)
            print(
                f"[{run_count:2d}/{total_runs}] Running {arm:16s} (Rep {rep + 1}/{args.replicates}) ... ",
                end="",
                flush=True,
            )
            try:
                res = _run_single_case(
                    arm=arm,
                    source_root=sources[arm],
                    python_bin=args.python,
                    output_dir=case_dir,
                    frame_file=args.frame_file,
                    bank_file=args.bank_file,
                    affinity=args.affinity,
                    approximant=args.approximant,
                    order=args.order,
                    use_compressed_waveforms=not args.uncompressed,
                    waveform_decompression_method=args.decompression_method,
                    batch_size=args.batch_size,
                    search_config=search_config,
                    expected_contract=contract,
                )
            except Exception as exc:
                checkpoint.update(status="failed", failure={
                    "phase": "timed_run", "arm": arm,
                    "replicate": rep + 1, "case_name": case_name,
                    "output_dir": str(case_dir), "error": str(exc),
                })
                atomic_receipt(args.output, checkpoint)
                raise
            res["case_name"] = case_name
            res["replicate"] = rep + 1
            try:
                _require_matching_contract(contract, res.get('input_contract'))
                trigger_gate = compare_trigger_parity(
                    Path(ref['triggers_path']),
                    Path(res['triggers_path']),
                    res['sample_rate'],
                )
                res['science'] = dict(
                    passed=checkpoint['science'][arm]['passed'] and trigger_gate['passed'],
                    scope=checkpoint['science'][arm]['scope'],
                    missing_gates=checkpoint['science'][arm]['missing_gates'],
                    qualification_arm=arm, trigger_gate=trigger_gate)
            except Exception as exc:
                checkpoint.update(status="failed", failure={
                    "phase": "run_science", "arm": arm,
                    "replicate": rep + 1, "case_name": case_name,
                    "output_dir": str(case_dir), "error": str(exc),
                })
                atomic_receipt(args.output, checkpoint)
                raise
            raw_results[arm].append(res)
            atomic_receipt(args.output, checkpoint)
            if (not res['science']['passed']
                    and not args.allow_unqualified_timings):
                checkpoint.update(status='failed', failure={
                    'phase': 'run_science', 'arm': arm, 'case_name': case_name,
                    'error': 'Scientific timing gate failed; remaining runs stopped',
                })
                atomic_receipt(args.output, checkpoint)
                raise RuntimeError(checkpoint['failure']['error'])
            print(
                f"DONE in {res['elapsed_wall_sec']:5.1f}s | Calc: {res['calc_time_sec']:5.2f}s | Trigs: {res['num_triggers']}",
                flush=True,
            )

    _require_source_unchanged(args.original_source, source_records["original"])
    validate_reference(args.original_source, args.reference_revision)
    parity_reference_arm, parity_results = compare_campaign_parity(raw_results)

    summaries: Dict[str, Dict[str, Any]] = {}
    for arm in args.arms:
        runs = raw_results[arm]
        walls = [r["elapsed_wall_sec"] for r in runs]
        calcs = [r["calc_time_sec"] for r in runs]
        tsetups = [r["tsetup_sec"] for r in runs]
        rss = [r["time_info"].get("max_rss_kib", 0) for r in runs]

        phase_sums = {}
        for phase_name in (
            "startup_import_sec",
            "conditioning_sec",
            "waveform_prep_sec",
            "matched_filter_sec",
            "vetoes_clustering_sec",
            "serialization_io_sec",
        ):
            p_vals = [r["phases"][phase_name] for r in runs]
            phase_sums[phase_name] = sample_summary(p_vals, "seconds")

        summaries[arm] = {
            "wall_sec": sample_summary(walls, "seconds"),
            "calc_time_sec": sample_summary(calcs, "seconds"),
            "tsetup_sec": sample_summary(tsetups, "seconds"),
            "max_rss_kib": sample_summary(rss, "kib"),
            "phases": phase_sums,
            "triggers_count": runs[0]["num_triggers"],
        }

    if (not args.allow_unqualified_timings and "original_cpu" in summaries
            and all(r["science"]["passed"]
                    for runs in raw_results.values() for r in runs)):
        base_wall = summaries["original_cpu"]["wall_sec"]["median"]
        base_calc = summaries["original_cpu"]["calc_time_sec"]["median"]
        for arm in args.arms:
            arm_wall = summaries[arm]["wall_sec"]["median"]
            arm_calc = summaries[arm]["calc_time_sec"]["median"]
            summaries[arm]["speedup_wall_vs_original"] = (
                base_wall / arm_wall if arm_wall > 0 else 0.0
            )
            summaries[arm]["speedup_calc_vs_original"] = (
                base_calc / arm_calc if arm_calc > 0 else 0.0
            )

    valid_seconds = {r.get('valid_detector_seconds') for runs in raw_results.values() for r in runs}
    science_qualified = all(
        run['science']['passed']
        for runs in raw_results.values() for run in runs
    )
    campaign_passed = science_qualified and not args.allow_unqualified_timings
    campaign_status = ("complete" if campaign_passed
                       else "complete_known_divergence")
    receipt = {
        "schema_version": 4,
        "executable": "pycbc_inspiral",
        "sample_rate": BENCHMARK_SAMPLE_RATE,
        "precision": BENCHMARK_PRECISION,
        "status": campaign_status,
        "performance_claim": campaign_passed,
        "timing_policy": timing_policy,
        "campaign": {
            "status": campaign_status,
            "process_complete": True,
            "science_qualified": science_qualified,
            "passed": campaign_passed,
            "performance_claim": campaign_passed,
            "known_divergence": bool(args.allow_unqualified_timings),
        },
        "qualification": qualification,
        "science": {arm: dict(checkpoint['science'][arm], passed=all(r['science']['passed'] for r in runs))
                    for arm, runs in raw_results.items()},
        "source": source_records,
        "identity": identity,
        "input_contract": contract,
        "runtime": runtime_metadata(),
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
            "JAX_COMPILATION_CACHE_DIR": "off",
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workload": {
            **inventory,
            "valid_detector_seconds": next(iter(valid_seconds)) if len(valid_seconds) == 1 else None,
            "search_config": search_config,
            "track": args.track,
            "waveform_mode": args.waveform_mode,
            "description": "Distinct physical templates on real detector frames; finite-workload result",
            "frame_file": str(args.frame_file),
            "frame_sha256": frame_sha,
            "bank_file": str(args.bank_file),
            "bank_sha256": bank_sha,
            "approximant": args.approximant,
            "order": args.order,
            "uncompressed": args.uncompressed,
            "decompression_method": contract['decompression_method'],
            "batch_size": args.batch_size,
            "sample_rate": BENCHMARK_SAMPLE_RATE,
            "precision": BENCHMARK_PRECISION,
            "jax_chisq_mode": (
                search_config.get("jax_chisq_mode", "cpu-compatible")
                if any("jax" in arm for arm in args.arms) else None
            ),
        },
        "provenance": {
            "original_source": str(args.original_source),
            "branch_source": str(args.branch_source),
            "original_commit": source_commits["original"],
            "branch_commit": source_commits["branch"],
            "python_executable": args.python,
            "affinity_core": args.affinity,
        },
        "replicates": args.replicates,
        "arms": args.arms,
        "summaries": summaries,
        "parity_reference_arm": parity_reference_arm,
        "parity_validation": parity_results,
        "raw_results": raw_results,
    }

    atomic_receipt(args.output, receipt)
    print("=" * 80)
    print(f"Receipt successfully written to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
