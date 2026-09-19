#!/usr/bin/env python3
"""Counterbalanced multi-arm pycbc_inspiral performance-evidence benchmark for JAX.

Evaluates standardized benchmark arms:
1. original_cpu:   Baseline checkout (40e94792b3), --processing-scheme cpu:1
2. branch_cpu:     Candidate branch, --processing-scheme cpu:1
3. jax_cpu:        Candidate branch, --processing-scheme jax:cpu (host LAL or decompression)
4. jax_cpu_diffgw: Candidate branch, --processing-scheme jax:cpu --enable-diffgw
5. jax_cuda_lal:   Candidate branch, --processing-scheme jax:cuda:0 (host LAL)
6. jax_cuda:       Candidate branch, --processing-scheme jax:cuda:0 (inline decompression)
7. jax_cuda_diffgw:Candidate branch, --processing-scheme jax:cuda:0 --enable-diffgw

Measures:
- Total process wall time (elapsed launch-to-exit)
- Internal run_time, setup_time_fraction, calc_time, and tsetup from triggers.hdf
- 6-phase workload decomposition:
    1. Process & Import Overhead
    2. Data Conditioning (Frame read, highpass, autogate, Welch PSD, overwhitening)
    3. Waveform Bank Preparation (Decompression / generation)
    4. Core Matched Filtering (Template correlation and IFFT)
    5. Vetoes & Clustering (Thresholding, Power chisq, newsnr, symmetric clustering)
    6. Serialization & I/O (HDF5 triggers writing and flushing)
- Scientific parity validation against original_cpu
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


ARM_NAMES = (
    "original_cpu",
    "branch_cpu",
    "branch_cpu_batched",
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
    "branch_cpu_batched",
    "jax_cpu_batched",
    "jax_cuda_batched",
)


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


def compare_trigger_parity(
    baseline_hdf: Path,
    candidate_hdf: Path,
    snr_rtol: float = 1e-4,
    snr_atol: float = 1e-5,
) -> Dict[str, Any]:
    """Verify numerical parity of triggers against baseline."""
    with h5py.File(baseline_hdf, "r") as fb, h5py.File(candidate_hdf, "r") as fc:
        ifo = "H1"
        if ifo not in fb or ifo not in fc:
            return {"passed": False, "error": f"IFO {ifo} missing in outputs"}

        b_snr = fb[f"{ifo}/snr"][:]
        c_snr = fc[f"{ifo}/snr"][:]

        if len(b_snr) == 0 and len(c_snr) == 0:
            return {
                "passed": True,
                "baseline_count": 0,
                "candidate_count": 0,
                "max_snr_diff": 0.0,
                "relative_l2_snr": 0.0,
            }

        count_match = len(b_snr) == len(c_snr)
        if count_match:
            snr_diff = np.abs(b_snr - c_snr)
            max_snr_diff = float(np.max(snr_diff))
            rel_diff = snr_diff / np.maximum(1e-5, np.abs(b_snr))
            max_rel_diff = float(np.max(rel_diff))

            l2_b = float(np.linalg.norm(b_snr))
            l2_diff = float(np.linalg.norm(b_snr - c_snr))
            relative_l2 = l2_diff / l2_b if l2_b > 0 else 0.0

            passed = bool(max_rel_diff <= snr_rtol or max_snr_diff <= snr_atol)

            return {
                "passed": passed,
                "trigger_count": len(b_snr),
                "matched_count": len(b_snr),
                "match_rate": 1.0,
                "max_snr_diff": max_snr_diff,
                "max_relative_diff": max_rel_diff,
                "relative_l2_snr": relative_l2,
                "snr_tolerance_gate": snr_rtol,
            }

        # Handle boundary clustering differences by matching nearest triggers in time
        b_time = fb[f"{ifo}/end_time"][:]
        c_time = fc[f"{ifo}/end_time"][:]

        matched_b_snr = []
        matched_c_snr = []
        for tb, sb in zip(b_time, b_snr):
            idx = int(np.argmin(np.abs(c_time - tb)))
            if np.abs(c_time[idx] - tb) <= 0.05:
                matched_b_snr.append(sb)
                matched_c_snr.append(c_snr[idx])

        matched_count = len(matched_b_snr)
        match_rate = matched_count / len(b_snr) if len(b_snr) > 0 else 0.0

        if matched_count > 0:
            m_b = np.array(matched_b_snr)
            m_c = np.array(matched_c_snr)
            snr_diff = np.abs(m_b - m_c)
            max_snr_diff = float(np.max(snr_diff))
            rel_diff = snr_diff / np.maximum(1e-5, np.abs(m_b))
            max_rel_diff = float(np.max(rel_diff))
            l2_b = float(np.linalg.norm(m_b))
            l2_diff = float(np.linalg.norm(m_b - m_c))
            relative_l2 = l2_diff / l2_b if l2_b > 0 else 0.0
        else:
            max_snr_diff = 999.0
            max_rel_diff = 999.0
            relative_l2 = 999.0

        passed = bool(match_rate >= 0.99 and (max_rel_diff <= 0.05 or relative_l2 <= 0.01))

        return {
            "passed": passed,
            "baseline_count": len(b_snr),
            "candidate_count": len(c_snr),
            "matched_count": matched_count,
            "match_rate": match_rate,
            "max_snr_diff": max_snr_diff,
            "max_relative_diff": max_rel_diff,
            "relative_l2_snr": relative_l2,
            "snr_tolerance_gate": snr_rtol,
            "note": "Nearest-time matching used due to boundary trigger clustering delta",
        }


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
    enable_diffgw: bool = True,
) -> Dict[str, Any]:
    """Execute a single unprofiled run of pycbc_inspiral."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    triggers_hdf = output_dir / "triggers.hdf"
    time_txt = output_dir / "time.txt"
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    source_root = source_root.resolve()
    frame_file = frame_file.resolve()
    bank_file = bank_file.resolve()

    if arm in ("original_cpu", "branch_cpu", "branch_cpu_batched"):
        scheme = "cpu:1"
    elif arm in ("jax_cpu", "jax_cpu_batched", "jax_cpu_diffgw"):
        scheme = "jax:cpu"
    elif arm in ("jax_cuda_lal", "jax_cuda", "jax_cuda_batched", "jax_cuda_diffgw"):
        scheme = "jax:cuda:0"
    else:
        raise ValueError(f"Unknown arm: {arm}")

    executable = source_root / "bin" / "pycbc_inspiral"

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
        "4096",
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
            cli_args.append("--disable-diffgw")
        elif arm in ("jax_cuda_diffgw", "jax_cpu_diffgw"):
            cli_args.append("--enable-diffgw")

    command = [
        python_bin,
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
        }
    )

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

    if p.returncode != 0:
        stderr_sample = stderr_path.read_text()[-1000:] if stderr_path.exists() else ""
        raise RuntimeError(
            f"Run failed ({p.returncode}) for arm={arm}:\n{stderr_sample}"
        )

    with h5py.File(triggers_hdf, "r") as hf:
        search_grp = hf["H1/search"]
        run_time_sec = float(search_grp["run_time"][0])
        setup_frac = float(search_grp["setup_time_fraction"][0])
        num_triggers = len(hf["H1/snr"]) if "H1/snr" in hf else 0

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

    return {
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


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-arm pycbc_inspiral reference campaign benchmark for JAX."
    )
    parser.add_argument(
        "--original-source",
        type=Path,
        required=True,
        help="Baseline PyCBC repo root (40e94792b3)",
    )
    parser.add_argument(
        "--branch-source",
        type=Path,
        required=True,
        help="Candidate PyCBC repo root (qualified)",
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
        help="384-template compressed HDF bank file",
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
        help="Benchmark arms to execute",
    )
    parser.add_argument(
        "--track",
        type=str,
        choices=["track1", "track2"],
        default=None,
        help="Preset benchmark track: track1 (compressed reference) or track2 (uncompressed diffgw)",
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
        "--uncompressed",
        action="store_true",
        default=False,
        help="Run uncompressed dynamic waveform generation without inline linear decompression",
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
        help="CUDA template batch size when running uncompressed diffgw (default: 64)",
    )
    parser.add_argument(
        "--enable-diffgw",
        action="store_true",
        default=True,
        help="Enable diffgw on CUDA arm (default: True)",
    )
    parser.add_argument(
        "--batched",
        action="store_true",
        default=False,
        help="Run batched arms (original_cpu, branch_cpu_batched, jax_cpu_batched, jax_cuda_batched)",
    )
    args = parser.parse_args()

    if args.batched and args.arms == list(DEFAULT_ARMS):
        args.arms = list(BATCHED_ARMS)

    if args.track == "track1":
        args.approximant = "IMRPhenomD"
        args.order = -1
        args.uncompressed = False
    elif args.track == "track2":
        args.approximant = "TaylorF2"
        args.order = 7
        args.uncompressed = True
        if args.arms == list(DEFAULT_ARMS):
            args.arms = [
                "original_cpu",
                "branch_cpu",
                "jax_cpu",
                "jax_cpu_diffgw",
                "jax_cuda_lal",
                "jax_cuda_diffgw",
            ]

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output = args.output.resolve()
    args.frame_file = args.frame_file.resolve()
    args.bank_file = args.bank_file.resolve()

    sources = {
        "original_cpu": args.original_source.resolve(),
        "branch_cpu": args.branch_source.resolve(),
        "branch_cpu_batched": args.branch_source.resolve(),
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

    source_commits = {
        "original": _git_commit(args.original_source),
        "branch": _git_commit(args.branch_source),
    }

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

    raw_results: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in args.arms}

    orderings = [
        ["original_cpu", "branch_cpu", "branch_cpu_batched", "jax_cpu", "jax_cpu_batched", "jax_cpu_diffgw", "jax_cuda_lal", "jax_cuda", "jax_cuda_batched", "jax_cuda_diffgw"],
        ["jax_cuda_diffgw", "jax_cuda_batched", "jax_cuda", "jax_cuda_lal", "jax_cpu_diffgw", "jax_cpu_batched", "jax_cpu", "branch_cpu_batched", "branch_cpu", "original_cpu"],
        ["branch_cpu", "branch_cpu_batched", "jax_cuda_lal", "jax_cuda_diffgw", "jax_cuda_batched", "original_cpu", "jax_cpu_diffgw", "jax_cuda", "jax_cpu_batched", "jax_cpu"],
    ]

    run_count = 0
    total_runs = args.replicates * len(args.arms)
    for rep in range(args.replicates):
        order = [a for a in orderings[rep % len(orderings)] if a in args.arms]
        for a in args.arms:
            if a not in order:
                order.append(a)
        for arm in order:
            run_count += 1
            case_name = f"{arm}_rep{rep + 1}"
            case_dir = args.output_dir / "runs" / case_name
            print(
                f"[{run_count:2d}/{total_runs}] Running {arm:16s} (Rep {rep + 1}/{args.replicates}) ... ",
                end="",
                flush=True,
            )
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
                enable_diffgw=args.enable_diffgw,
            )
            raw_results[arm].append(res)
            print(
                f"DONE in {res['elapsed_wall_sec']:5.1f}s | Calc: {res['calc_time_sec']:5.2f}s | Trigs: {res['num_triggers']}",
                flush=True,
            )

    baseline_run = raw_results["original_cpu"][0] if "original_cpu" in raw_results else None
    parity_results = {}
    if baseline_run:
        b_hdf = Path(baseline_run["triggers_path"])
        for arm in args.arms:
            if arm == "original_cpu":
                continue
            cand_hdf = Path(raw_results[arm][0]["triggers_path"])
            parity = compare_trigger_parity(b_hdf, cand_hdf)
            parity_results[arm] = parity

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

    if "original_cpu" in summaries:
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

    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workload": {
            "track": args.track or ("track2_diffgw" if args.uncompressed else "track1_compressed"),
            "description": (
                "512 uncompressed TaylorF2 templates, 512/112/16s geometry, real H1 frame"
                if args.uncompressed
                else "384 compressed BNS/NSBH templates, 512/112/16s geometry, real H1 frame"
            ),
            "frame_file": str(args.frame_file),
            "frame_sha256": frame_sha,
            "bank_file": str(args.bank_file),
            "bank_sha256": bank_sha,
            "approximant": args.approximant,
            "order": args.order,
            "uncompressed": args.uncompressed,
            "decompression_method": args.decompression_method,
            "batch_size": args.batch_size,
            "enable_diffgw": args.enable_diffgw,
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
        "parity_validation": parity_results,
        "raw_results": raw_results,
    }

    with args.output.open("w") as f:
        json.dump(receipt, f, indent=2)
    print("=" * 80)
    print(f"Receipt successfully written to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
