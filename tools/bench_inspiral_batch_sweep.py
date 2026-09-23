#!/usr/bin/env python3
# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Standalone benchmark tool for sweeping pycbc_inspiral batch sizes.

Benchmarks pycbc_inspiral across specified batch sizes measuring:
- Total process wall time
- Internal calc_time (matched filter calculation time)
- Internal tsetup (data conditioning / PSD setup time)
- Peak VRAM allocated on CUDA (torch.cuda.max_memory_allocated())
- Throughput: templates/second and template-seconds/wall-second

Outputs a structured benchmark JSON receipt and prints an execution
summary table.

Example:
  python tools/bench_inspiral_batch_sweep.py \\
    --batch-sizes 16,64,128,256,512,1024 \\
    --device cuda:0 \\
    --approximant TaylorF2 \\
    --num-templates 1024 \\
    --output artifacts/bench_inspiral_batch_sweep.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

# Optional integration with PyCBC benchmark artifact conventions
try:
    from tools.benchmark_artifact import (
        atomic_write_json,
        runtime_metadata as artifact_runtime_metadata,
        seal_artifact,
    )
except ModuleNotFoundError:
    try:
        from benchmark_artifact import (
            atomic_write_json,
            runtime_metadata as artifact_runtime_metadata,
            seal_artifact,
        )
    except ModuleNotFoundError:
        def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            tmp.replace(path)

        def artifact_runtime_metadata() -> Dict[str, Any]:
            return {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
            }

        def seal_artifact(payload: Dict[str, Any]) -> Dict[str, Any]:
            return payload


logger = logging.getLogger("bench_inspiral_batch_sweep")

DEFAULT_BATCH_SIZES = [16, 64, 128, 256, 512, 1024]
SUPPORTED_APPROXIMANTS = ("TaylorF2", "IMRPhenomD", "IMRPhenomXAS")


def parse_batch_sizes(value: Any) -> List[int]:
    """Parse comma-separated or space-separated batch size integers."""
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            if isinstance(item, int):
                if item <= 0:
                    raise ValueError(
                        f"Batch size must be positive, got {item}"
                    )
                result.append(item)
            elif isinstance(item, str):
                for part in item.replace(",", " ").split():
                    val = int(part)
                    if val <= 0:
                        raise ValueError(
                            f"Batch size must be positive, got {val}"
                        )
                    result.append(val)
        return result
    elif isinstance(value, str):
        result = []
        for part in value.replace(",", " ").split():
            val = int(part)
            if val <= 0:
                raise ValueError(f"Batch size must be positive, got {val}")
            result.append(val)
        return result
    raise ValueError(f"Unsupported batch size input format: {value!r}")


def normalize_device_and_scheme(
    device: str,
) -> Tuple[str, str, Optional[int]]:
    """Normalize input device to (dev_str, scheme, cuda_device_index).

    Examples:
      'cuda:0' -> ('cuda:0', 'torch:cuda:0', 0)
      'cuda'   -> ('cuda:0', 'torch:cuda:0', 0)
      'cpu'    -> ('cpu', 'torch:cpu:1', None)
      'torch:cuda:1' -> ('cuda:1', 'torch:cuda:1', 1)
      'torch:cpu:1'  -> ('cpu', 'torch:cpu:1', None)
    """
    dev = device.strip()
    cuda_idx = None

    if dev.startswith("torch:"):
        scheme = dev
        sub = dev.split(":", 1)[1]
        if "cuda" in sub:
            parts = sub.split(":")
            if len(parts) > 1 and parts[1].isdigit():
                cuda_idx = int(parts[1])
            else:
                cuda_idx = 0
            dev_str = f"cuda:{cuda_idx}"
        else:
            dev_str = "cpu"
    elif "cuda" in dev.lower():
        if ":" in dev:
            try:
                cuda_idx = int(dev.split(":")[-1])
            except ValueError:
                cuda_idx = 0
        else:
            cuda_idx = 0
        dev_str = f"cuda:{cuda_idx}"
        scheme = f"torch:cuda:{cuda_idx}"
    elif dev.lower() == "cpu":
        dev_str = "cpu"
        scheme = "torch:cpu:1"
    else:
        dev_str = dev
        scheme = dev

    return dev_str, scheme, cuda_idx


def generate_template_bank(
    path: Path | str,
    num_templates: int,
    approximant: str,
    f_lower: float = 30.0,
) -> None:
    """Generate a temporary template bank HDF5 file for pycbc_inspiral."""
    if num_templates <= 0:
        raise ValueError(
            f"num_templates must be positive, got {num_templates}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    app_upper = approximant.upper()
    if "TAYLOR" in app_upper:
        # BNS / low-mass regime
        m1_min, m1_max = 1.4, 2.5
        m2_min, m2_max = 1.2, 1.4
    else:
        # BBH regime (IMRPhenomD, IMRPhenomXAS, etc.)
        m1_min, m1_max = 10.0, 50.0
        m2_min, m2_max = 5.0, 25.0

    if num_templates > 1:
        mass1 = np.linspace(m1_min, m1_max, num_templates, dtype=np.float64)
        mass2 = np.linspace(m2_min, m2_max, num_templates, dtype=np.float64)
    else:
        mass1 = np.array([m1_min], dtype=np.float64)
        mass2 = np.array([m2_min], dtype=np.float64)

    spin1z = np.zeros(num_templates, dtype=np.float64)
    spin2z = np.zeros(num_templates, dtype=np.float64)
    flow = np.full(num_templates, float(f_lower), dtype=np.float64)

    with h5py.File(str(path), "w") as f:
        f["mass1"] = mass1
        f["mass2"] = mass2
        f["spin1z"] = spin1z
        f["spin2z"] = spin2z
        f["f_lower"] = flow
        f.attrs["parameters"] = [
            "mass1", "mass2", "spin1z", "spin2z", "f_lower"
        ]
        f.attrs["approximant"] = str(approximant)


def _worker_main(spec_file: str) -> None:
    """Entrypoint for child worker process executing pycbc_inspiral."""
    with open(spec_file, "r", encoding="utf-8") as f:
        spec = json.load(f)

    device = spec["device"]
    _, _, cuda_idx = normalize_device_and_scheme(device)

    # Initialize PyTorch and CUDA memory tracking if applicable
    torch = None
    cuda_active = False
    if cuda_idx is not None:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA requested on device '{device}', "
                    "but torch.cuda.is_available() is False"
                )
            torch.cuda.set_device(cuda_idx)
            torch.cuda.synchronize(cuda_idx)
            torch.cuda.reset_peak_memory_stats(cuda_idx)
            cuda_active = True
        except ImportError as exc:
            raise RuntimeError(
                f"PyTorch is required for CUDA execution: {exc}"
            ) from exc

    inspiral_entrypoint = spec["inspiral_entrypoint"]
    argv = spec["argv"]
    output_file = spec["output_file"]
    num_templates = spec["num_templates"]
    channel_name = spec.get("channel_name", "H1:FAKE")

    # Set up argv and measure wall time
    orig_argv = sys.argv
    sys.argv = argv
    t_start = time.perf_counter()
    exit_error: Optional[str] = None

    try:
        runpy.run_path(inspiral_entrypoint, run_name="__main__")
    except SystemExit as se:
        code = se.code if se.code is not None else 0
        if code != 0:
            exit_error = f"pycbc_inspiral exited with status code {code}"
    except Exception as exc:
        exit_error = f"pycbc_inspiral raised {type(exc).__name__}: {exc}"
    finally:
        sys.argv = orig_argv

    # Capture CUDA peak memory
    peak_vram_bytes = 0
    if cuda_active and torch is not None:
        try:
            torch.cuda.synchronize(cuda_idx)
            peak_vram_bytes = int(torch.cuda.max_memory_allocated(cuda_idx))
        except Exception:
            pass

    wall_time_sec = time.perf_counter() - t_start

    if exit_error is not None:
        result = {
            "status": "failed",
            "error": exit_error,
            "wall_time_sec": wall_time_sec,
            "peak_vram_bytes": peak_vram_bytes,
            "peak_vram_mib": float(peak_vram_bytes) / (1024.0 * 1024.0),
        }
    else:
        if not os.path.isfile(output_file):
            result = {
                "status": "failed",
                "error": f"Output file not found after run: {output_file}",
                "wall_time_sec": wall_time_sec,
                "peak_vram_bytes": peak_vram_bytes,
                "peak_vram_mib": float(peak_vram_bytes) / (1024.0 * 1024.0),
            }
        else:
            try:
                with h5py.File(output_file, "r") as hf:
                    if ":" in channel_name:
                        ifo = channel_name.split(":")[0]
                    else:
                        ifo = "H1"
                    search_grp = hf[f"{ifo}/search"]
                    internal_runtime_sec = float(search_grp["run_time"][0])
                    setup_time_frac = float(
                        search_grp["setup_time_fraction"][0]
                    )
                    start_time = float(search_grp["start_time"][0])
                    end_time = float(search_grp["end_time"][0])

                    triggers_count = 0
                    if f"{ifo}/snr" in hf:
                        triggers_count = len(hf[f"{ifo}/snr"])

                setup_time_sec = internal_runtime_sec * setup_time_frac
                calc_time_sec = internal_runtime_sec - setup_time_sec
                valid_data_seconds = max(0.0, end_time - start_time)
                total_template_seconds = num_templates * valid_data_seconds

                if wall_time_sec > 0:
                    tmplt_sec_wall = float(num_templates) / wall_time_sec
                    tmplt_s_per_wall_s = (
                        total_template_seconds / wall_time_sec
                    )
                else:
                    tmplt_sec_wall = 0.0
                    tmplt_s_per_wall_s = 0.0

                if calc_time_sec > 0:
                    tmplt_sec_calc = float(num_templates) / calc_time_sec
                    tmplt_s_per_calc_s = (
                        total_template_seconds / calc_time_sec
                    )
                else:
                    tmplt_sec_calc = 0.0
                    tmplt_s_per_calc_s = 0.0

                result = {
                    "status": "success",
                    "batch_size": spec["batch_size"],
                    "wall_time_sec": wall_time_sec,
                    "calc_time_sec": calc_time_sec,
                    "setup_time_sec": setup_time_sec,
                    "internal_runtime_sec": internal_runtime_sec,
                    "setup_time_fraction": setup_time_frac,
                    "peak_vram_bytes": peak_vram_bytes,
                    "peak_vram_mib": (
                        float(peak_vram_bytes) / (1024.0 * 1024.0)
                    ),
                    "templates_per_sec_wall": tmplt_sec_wall,
                    "templates_per_sec_calc": tmplt_sec_calc,
                    "template_seconds_per_wall_sec": tmplt_s_per_wall_s,
                    "template_seconds_per_calc_sec": tmplt_s_per_calc_s,
                    "total_template_seconds": total_template_seconds,
                    "valid_data_seconds": valid_data_seconds,
                    "triggers_count": triggers_count,
                }
            except Exception as read_exc:
                result = {
                    "status": "failed",
                    "error": (
                        f"Failed reading metrics from {output_file}: "
                        f"{read_exc}"
                    ),
                    "wall_time_sec": wall_time_sec,
                    "peak_vram_bytes": peak_vram_bytes,
                    "peak_vram_mib": (
                        float(peak_vram_bytes) / (1024.0 * 1024.0)
                    ),
                }

    result_file = spec.get("result_file")
    if result_file:
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print("RESULT_JSON=" + json.dumps(result))


def _run_batch_subprocess(
    spec: Dict[str, Any],
    script_path: Path,
) -> Dict[str, Any]:
    """Execute worker in a clean, isolated subprocess."""
    result_file = Path(spec["result_file"])
    spec_file = result_file.with_suffix(".spec.json")

    with open(spec_file, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--worker-spec",
        str(spec_file),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True, check=False
    )
    proc_wall_time = time.perf_counter() - t0

    if result_file.is_file():
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                res = json.load(f)
            if "wall_time_sec" not in res or res["wall_time_sec"] is None:
                res["wall_time_sec"] = proc_wall_time
            return res
        except Exception:
            pass

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            try:
                res = json.loads(line[len("RESULT_JSON="):])
                return res
            except Exception:
                pass

    err_out = proc.stderr.strip() or proc.stdout.strip()
    error_msg = err_out or f"Process exited with code {proc.returncode}"
    return {
        "status": "failed",
        "batch_size": spec["batch_size"],
        "error": error_msg,
        "wall_time_sec": proc_wall_time,
        "peak_vram_bytes": 0,
        "peak_vram_mib": 0.0,
    }


def run_batch_sweep(
    batch_sizes: List[int] | Tuple[int, ...] | str = DEFAULT_BATCH_SIZES,
    device: str = "cuda:0",
    approximant: str = "TaylorF2",
    num_templates: int = 1024,
    output: Optional[Path | str] = None,
    enable_diffgw: bool = False,
    native_gpu_conditioning: bool = False,
    sample_rate: int = 2048,
    segment_length: int = 256,
    duration: int = 256,
    psd_model: str = "aLIGOZeroDetHighPower",
    fake_strain_seed: int = 42,
    snr_threshold: float = 5.5,
    chisq_bins: int = 8,
    low_frequency_cutoff: float = 30.0,
    keep_temp: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Perform a batch size sweep benchmarking pycbc_inspiral."""
    batch_list = parse_batch_sizes(batch_sizes)
    dev_str, scheme, _ = normalize_device_and_scheme(device)

    repo_root = Path(__file__).resolve().parent.parent
    inspiral_entrypoint = repo_root / "bin" / "pycbc_inspiral"
    if not inspiral_entrypoint.is_file():
        raise FileNotFoundError(
            f"pycbc_inspiral not found at {inspiral_entrypoint}"
        )

    script_path = Path(__file__).resolve()

    gps_start = 1000000000
    gps_end = gps_start + duration
    segment_start_pad = 64
    segment_end_pad = 16
    channel_name = "H1:FAKE"

    metadata = artifact_runtime_metadata()
    results: List[Dict[str, Any]] = []

    print("=== Inspiral Batch Size Sweep Benchmark ===")
    print(f"Device:               {dev_str} (scheme: {scheme})")
    print(f"Approximant:          {approximant}")
    print(f"Templates:            {num_templates}")
    print(f"Batch sizes:          {batch_list}")
    print(f"Enable diffgw:        {enable_diffgw}")
    print(f"Native GPU condition: {native_gpu_conditioning}")
    print(f"Sample rate:          {sample_rate} Hz")
    print(
        f"Duration:             {duration} s "
        f"(segment length: {segment_length} s)"
    )
    print("-" * 78)

    temp_root = Path(tempfile.mkdtemp(prefix="bench_inspiral_sweep_"))
    try:
        for b in batch_list:
            b_dir = temp_root / f"batch_{b}"
            b_dir.mkdir(parents=True, exist_ok=True)

            bank_path = b_dir / f"bank_{approximant}_{num_templates}.hdf"
            output_hdf = b_dir / f"triggers_b{b}.hdf"
            result_json = b_dir / f"result_b{b}.json"

            # 1. Generate bank file for this batch run
            generate_template_bank(
                bank_path,
                num_templates=num_templates,
                approximant=approximant,
                f_lower=low_frequency_cutoff,
            )

            # 2. Build argv for pycbc_inspiral
            argv = [
                "pycbc_inspiral",
                "--bank-file", str(bank_path),
                "--approximant", approximant,
                "--channel-name", channel_name,
                "--gps-start-time", str(gps_start),
                "--gps-end-time", str(gps_end),
                "--sample-rate", str(sample_rate),
                "--fake-strain", psd_model,
                "--fake-strain-seed", str(fake_strain_seed),
                "--low-frequency-cutoff", str(low_frequency_cutoff),
                "--pad-data", "8",
                "--strain-high-pass", "20",
                "--segment-length", str(segment_length),
                "--segment-start-pad", str(segment_start_pad),
                "--segment-end-pad", str(segment_end_pad),
                "--psd-model", psd_model,
                "--psd-inverse-length", "4",
                "--snr-threshold", str(snr_threshold),
                "--chisq-bins", str(chisq_bins),
                "--cluster-window", "1",
                "--cluster-function", "symmetric",
                "--processing-scheme", scheme,
                "--batch-size", str(b),
                "--output", str(output_hdf),
            ]
            if enable_diffgw:
                argv.append("--enable-diffgw")
            if native_gpu_conditioning:
                argv.append("--native-gpu-conditioning")

            spec = {
                "batch_size": b,
                "argv": argv,
                "output_file": str(output_hdf),
                "result_file": str(result_json),
                "device": dev_str,
                "num_templates": num_templates,
                "channel_name": channel_name,
                "inspiral_entrypoint": str(inspiral_entrypoint),
            }

            print(f"Running batch size {b:4d} ...", end="", flush=True)
            res = _run_batch_subprocess(spec, script_path)
            res["batch_size"] = b
            results.append(res)

            if res["status"] == "success":
                wall_s = res["wall_time_sec"]
                calc_s = res["calc_time_sec"]
                setup_s = res["setup_time_sec"]
                vram_mib = res["peak_vram_mib"]
                tmplt_s = res["templates_per_sec_wall"]
                tmplt_sec_per_s = res["template_seconds_per_wall_sec"]
                print(
                    f" [PASS] wall: {wall_s:6.2f}s | calc: {calc_s:6.2f}s | "
                    f"setup: {setup_s:6.2f}s | "
                    f"peak VRAM: {vram_mib:6.1f} MiB | "
                    f"{tmplt_s:7.1f} tmplts/s | "
                    f"{tmplt_sec_per_s:9.1f} tmplt-s/s"
                )
            else:
                err = res.get("error", "unknown error")
                first_line = err.splitlines()[-1] if err.splitlines() else err
                print(f" [FAIL] {first_line[:60]}")

    finally:
        if not keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)

    passing = [r for r in results if r["status"] == "success"]
    summary: Dict[str, Any] = {
        "total_runs": len(results),
        "passing_runs": len(passing),
        "failed_runs": len(results) - len(passing),
    }

    if passing:
        best_wall = max(passing, key=lambda r: r["templates_per_sec_wall"])
        best_calc = max(passing, key=lambda r: r["templates_per_sec_calc"])
        summary["best_batch_by_wall_throughput"] = {
            "batch_size": best_wall["batch_size"],
            "templates_per_sec_wall": best_wall["templates_per_sec_wall"],
            "template_seconds_per_wall_sec": (
                best_wall["template_seconds_per_wall_sec"]
            ),
            "wall_time_sec": best_wall["wall_time_sec"],
        }
        summary["best_batch_by_calc_throughput"] = {
            "batch_size": best_calc["batch_size"],
            "templates_per_sec_calc": best_calc["templates_per_sec_calc"],
            "calc_time_sec": best_calc["calc_time_sec"],
        }
        summary["peak_vram_mib_overall"] = max(
            r["peak_vram_mib"] for r in passing
        )

    receipt: Dict[str, Any] = {
        "schema_version": 2,
        "benchmark_type": "inspiral_batch_sweep",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "batch_sizes": batch_list,
            "device": dev_str,
            "processing_scheme": scheme,
            "approximant": approximant,
            "num_templates": num_templates,
            "enable_diffgw": enable_diffgw,
            "native_gpu_conditioning": native_gpu_conditioning,
            "sample_rate": sample_rate,
            "segment_length": segment_length,
            "duration": duration,
            "psd_model": psd_model,
            "low_frequency_cutoff": low_frequency_cutoff,
        },
        "metadata": metadata,
        "results": results,
        "summary": summary,
    }

    receipt = seal_artifact(receipt)

    if output is not None:
        out_path = Path(output)
        atomic_write_json(out_path, receipt)
        print(
            f"\nStructured benchmark receipt written to: {out_path.resolve()}"
        )

    # Print summary table
    print("\n" + "=" * 90)
    header = (
        f"{'Batch':>7} | {'Status':>7} | {'Wall (s)':>8} | {'Calc (s)':>8} | "
        f"{'Setup (s)':>9} | {'VRAM (MiB)':>10} | {'Tmplts/s':>10} | "
        f"{'Tmplt-s/Wall-s':>14}"
    )
    print(header)
    print("-" * 90)
    for r in results:
        b = r["batch_size"]
        st = r["status"]
        if st == "success":
            wall_str = f"{r['wall_time_sec']:8.2f}"
            calc_str = f"{r['calc_time_sec']:8.2f}"
            setup_str = f"{r['setup_time_sec']:9.2f}"
            vram_str = f"{r['peak_vram_mib']:10.1f}"
            tmplt_str = f"{r['templates_per_sec_wall']:10.1f}"
            ratio_str = f"{r['template_seconds_per_wall_sec']:14.1f}"
            print(
                f"{b:7d} | {st:>7} | {wall_str} | {calc_str} | "
                f"{setup_str} | {vram_str} | {tmplt_str} | {ratio_str}"
            )
        else:
            print(
                f"{b:7d} | {st:>7} | {'N/A':>8} | {'N/A':>8} | "
                f"{'N/A':>9} | {'N/A':>10} | {'N/A':>10} | {'N/A':>14}"
            )
    print("=" * 90 + "\n")

    return receipt


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Worker-mode dispatch
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-spec",
        type=str,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--approximant",
        default="TaylorF2",
        choices=SUPPORTED_APPROXIMANTS,
        help="Gravitational-wave approximant model (default: %(default)s)",
    )
    parser.add_argument(
        "--num-templates",
        type=int,
        default=1024,
        help="Total templates in the generated bank (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/bench_inspiral_batch_sweep.json",
        help=(
            "Destination path for benchmark receipt JSON "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--enable-diffgw",
        action="store_true",
        default=False,
        help="Enable diffgw on-device batched waveform generation",
    )
    parser.add_argument(
        "--native-gpu-conditioning",
        action="store_true",
        default=False,
        help="Enable native GPU strain conditioning",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=2048,
        help="Strain sample rate in Hz (default: %(default)s)",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=256,
        help="Length of analysis segments in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=256,
        help=(
            "Duration of synthetic strain data in seconds "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--psd-model",
        type=str,
        default="aLIGOZeroDetHighPower",
        help="Analytical PSD model (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        default=False,
        help="Retain temporary bank and trigger files for debugging",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose output",
    )

    args = parser.parse_args(argv)

    if args.worker:
        if not args.worker_spec:
            parser.error("--worker requires --worker-spec")
        _worker_main(args.worker_spec)
        return 0

    receipt = run_batch_sweep(
        batch_sizes=args.batch_sizes,
        device=args.device,
        approximant=args.approximant,
        num_templates=args.num_templates,
        output=args.output,
        enable_diffgw=args.enable_diffgw,
        native_gpu_conditioning=args.native_gpu_conditioning,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        duration=args.duration,
        psd_model=args.psd_model,
        keep_temp=args.keep_temp,
        verbose=args.verbose,
    )

    summary = receipt["summary"]
    if summary["passing_runs"] == 0 and summary["total_runs"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
