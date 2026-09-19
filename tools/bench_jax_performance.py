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

"""Benchmark harness for JAX matched filtering and waveform calculation throughput.

Evaluates the 6-arm comparison matrix:
1. original_cpu: Standard PyCBC CPU matched filter with LAL waveforms
2. branch_cpu: Current branch PyCBC CPU matched filter with LAL waveforms
3. jax_cpu_lal: JAX CPU matched filter with host LAL waveforms
4. jax_cpu_diffgw: JAX CPU matched filter with pure JAX diffgw waveforms
5. jax_cuda_lal: JAX CUDA matched filter with host LAL waveforms (PCIe transfer bottleneck)
6. jax_cuda_diffgw: JAX CUDA matched filter with on-device diffgw waveforms (pure GPU)
"""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import scipy.linalg

# Compatibility shim for older JAX releases on SciPy >= 1.13
if not hasattr(scipy.linalg, "tril"):
    scipy.linalg.tril = np.tril
if not hasattr(scipy.linalg, "triu"):
    scipy.linalg.triu = np.triu

# Ensure PyCBC is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.benchmark_artifact import sample_summary, seal_artifact


ARM_LABELS = {
    "original_cpu": "Original CPU (LAL + NumPy)",
    "branch_cpu": "Branch CPU (LAL + NumPy)",
    "jax_cpu_lal": "JAX CPU (LAL)",
    "jax_cpu_diffgw": "JAX CPU (diffgw)",
    "jax_cuda_lal": "JAX CUDA (LAL)",
    "jax_cuda_diffgw": "JAX CUDA (diffgw)",
}


def get_hardware_info() -> Dict[str, Any]:
    """Gather detailed host CPU and GPU hardware metadata."""
    info = {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # CPU model
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        elif platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            info["cpu_model"] = out
    except Exception:
        info["cpu_model"] = platform.processor()

    # GPU model via nvidia-smi
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        info["gpu_model"] = parts[0]
        info["gpu_memory"] = parts[1]
        info["gpu_driver"] = parts[2]
    except Exception:
        info["gpu_model"] = "None"

    return info


def generate_lal_templates(
    m1_vals: np.ndarray,
    m2_vals: np.ndarray,
    flow: float,
    df: float,
    n_freq: int,
    precision: str = "single",
) -> np.ndarray:
    """Generate a batch of TaylorF2 frequency-domain templates via LAL."""
    from pycbc.waveform import get_fd_waveform

    c_dtype = np.complex64 if precision == "single" else np.complex128
    b = len(m1_vals)
    batch_arr = np.zeros((b, n_freq), dtype=c_dtype)
    for i in range(b):
        hp, _ = get_fd_waveform(
            approximant="TaylorF2",
            mass1=float(m1_vals[i]),
            mass2=float(m2_vals[i]),
            f_lower=flow,
            delta_f=df,
        )
        valid_len = min(len(hp), n_freq)
        batch_arr[i, :valid_len] = hp[:valid_len].astype(c_dtype)
    return batch_arr


def generate_diffgw_templates(
    m1_vals: np.ndarray,
    m2_vals: np.ndarray,
    flow: float,
    df: float,
    n_freq: int,
    device: Optional[Any] = None,
    precision: str = "single",
) -> Any:
    """Generate a batch of TaylorF2 frequency-domain templates via diffgw."""
    import diffgw
    import jax
    import jax.numpy as jnp

    use_x64 = (precision == "double")
    jax.config.update("jax_enable_x64", use_x64)
    f_dtype = jnp.float64 if use_x64 else jnp.float32

    b = len(m1_vals)
    freqs = jnp.arange(n_freq, dtype=f_dtype) * df
    s1z = jnp.zeros(b, dtype=f_dtype)
    s2z = jnp.zeros(b, dtype=f_dtype)
    dist = jnp.ones(b, dtype=f_dtype)
    inc = jnp.zeros(b, dtype=f_dtype)
    phic = jnp.zeros(b, dtype=f_dtype)

    ctx = jax.default_device(device) if device is not None else jax.named_scope("diffgw")
    with ctx:
        hp, _ = diffgw.get_fd_waveform(
            "TaylorF2",
            mass1=jnp.asarray(m1_vals, dtype=f_dtype),
            mass2=jnp.asarray(m2_vals, dtype=f_dtype),
            spin1z=s1z,
            spin2z=s2z,
            distance=dist,
            inclination=inc,
            phic=phic,
            sample_frequencies=freqs,
            backend="jax",
            dtype=f_dtype,
            f_isco_cutoff=False,
        )
        if b == 1 and hp.ndim == 1:
            hp = hp[None, :]
        return hp


def run_benchmark_arm(
    arm: str,
    n_time: int,
    batch_size: int,
    trials: int = 10,
    flow: float = 30.0,
    fhigh: float = 1000.0,
    gpu_dev: Optional[Any] = None,
    cpu_dev: Optional[Any] = None,
    precision: str = "single",
) -> Dict[str, Any]:
    """Execute timed trials for a specific benchmark arm and batch size."""
    import jax
    import jax.numpy as jnp
    from pycbc.filter.matchedfilter import matched_filter
    from pycbc.filter.matchedfilter_jax import batch_matched_filter_bank
    from pycbc.types import FrequencySeries

    try:
        import diffgw
    except Exception:
        diffgw = None

    use_x64 = (precision == "double")
    jax.config.update("jax_enable_x64", use_x64)
    f_dtype = jnp.float64 if use_x64 else jnp.float32
    c_dtype = jnp.complex128 if use_x64 else jnp.complex64
    np_f_dtype = np.float64 if use_x64 else np.float32
    np_c_dtype = np.complex128 if use_x64 else np.complex64

    sample_rate = 2048.0 if n_time <= 131072 else 4096.0
    df = sample_rate / n_time
    n_freq = n_time // 2 + 1

    m1_vals = np.linspace(1.4, 2.0, batch_size)
    m2_vals = np.full(batch_size, 1.4)

    rng = np.random.RandomState(42)
    strain_noise = (rng.randn(n_freq) + 1j * rng.randn(n_freq)).astype(np_c_dtype)
    psd_noise = np.ones(n_freq, dtype=np_f_dtype)

    strain_fs = FrequencySeries(strain_noise, delta_f=df)
    psd_fs = FrequencySeries(psd_noise, delta_f=df)

    strain_jnp_cpu = jax.device_put(jnp.asarray(strain_noise, dtype=c_dtype), cpu_dev)
    psd_jnp_cpu = jax.device_put(jnp.asarray(psd_noise, dtype=f_dtype), cpu_dev)

    if gpu_dev is not None:
        strain_jnp_gpu = jax.device_put(jnp.asarray(strain_noise, dtype=c_dtype), gpu_dev)
        psd_jnp_gpu = jax.device_put(jnp.asarray(psd_noise, dtype=f_dtype), gpu_dev)
    else:
        strain_jnp_gpu = None
        psd_jnp_gpu = None

    freqs_jnp = jnp.arange(n_freq, dtype=f_dtype) * df
    s1z_jnp = jnp.zeros(batch_size, dtype=f_dtype)
    s2z_jnp = jnp.zeros(batch_size, dtype=f_dtype)
    dist_jnp = jnp.ones(batch_size, dtype=f_dtype)
    inc_jnp = jnp.zeros(batch_size, dtype=f_dtype)
    phic_jnp = jnp.zeros(batch_size, dtype=f_dtype)
    m1_jnp = jnp.asarray(m1_vals, dtype=f_dtype)
    m2_jnp = jnp.asarray(m2_vals, dtype=f_dtype)

    # JIT filter kernels & diffgw generators
    if cpu_dev is not None:
        with jax.default_device(cpu_dev):
            jitted_filter_cpu = jax.jit(
                lambda tmpl, st, p: batch_matched_filter_bank(
                    tmpl, st, p, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh, delta_f=df
                )
            )
            if diffgw is not None:
                jitted_diffgw_cpu = jax.jit(
                    lambda m1, m2: diffgw.get_fd_waveform(
                        "TaylorF2",
                        mass1=m1,
                        mass2=m2,
                        spin1z=s1z_jnp,
                        spin2z=s2z_jnp,
                        distance=dist_jnp,
                        inclination=inc_jnp,
                        phic=phic_jnp,
                        sample_frequencies=freqs_jnp,
                        backend="jax",
                        dtype=jnp.float64,
                        f_isco_cutoff=False,
                    )[0]
                )
            else:
                jitted_diffgw_cpu = None
    else:
        jitted_filter_cpu = None
        jitted_diffgw_cpu = None

    if gpu_dev is not None:
        with jax.default_device(gpu_dev):
            jitted_filter_gpu = jax.jit(
                lambda tmpl, st, p: batch_matched_filter_bank(
                    tmpl, st, p, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh, delta_f=df
                )
            )
            if diffgw is not None:
                jitted_diffgw_gpu = jax.jit(
                    lambda m1, m2: diffgw.get_fd_waveform(
                        "TaylorF2",
                        mass1=m1,
                        mass2=m2,
                        spin1z=s1z_jnp,
                        spin2z=s2z_jnp,
                        distance=dist_jnp,
                        inclination=inc_jnp,
                        phic=phic_jnp,
                        sample_frequencies=freqs_jnp,
                        backend="jax",
                        dtype=jnp.float64,
                        f_isco_cutoff=False,
                    )[0]
                )
            else:
                jitted_diffgw_gpu = None
    else:
        jitted_filter_gpu = None
        jitted_diffgw_gpu = None

    def execute_single_pass():
        t_w0 = time.perf_counter()
        t_transfer = 0.0

        if arm in ("original_cpu", "branch_cpu"):
            # LAL waveform generation
            lal_tmpls = generate_lal_templates(m1_vals, m2_vals, flow, df, n_freq, precision=precision)
            t_w1 = time.perf_counter()
            t_waveform = t_w1 - t_w0

            # PyCBC standard CPU matched filtering
            t_f0 = time.perf_counter()
            for b_idx in range(batch_size):
                tmpl_i = FrequencySeries(lal_tmpls[b_idx], delta_f=df)
                _ = matched_filter(
                    tmpl_i, strain_fs, psd=psd_fs, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh
                )
            t_f1 = time.perf_counter()
            t_filter = t_f1 - t_f0

        elif arm == "jax_cpu_lal":
            # LAL waveform generation
            lal_tmpls = generate_lal_templates(m1_vals, m2_vals, flow, df, n_freq, precision=precision)
            t_w1 = time.perf_counter()
            t_waveform = t_w1 - t_w0

            # Transfer to JAX CPU array
            t_tr0 = time.perf_counter()
            tmpls_jnp = jax.device_put(jnp.asarray(lal_tmpls, dtype=c_dtype), cpu_dev)
            tmpls_jnp.block_until_ready()
            t_transfer = time.perf_counter() - t_tr0

            # JAX CPU filter
            t_f0 = time.perf_counter()
            snr, sq = jitted_filter_cpu(tmpls_jnp, strain_jnp_cpu, psd_jnp_cpu)
            snr.block_until_ready()
            sq.block_until_ready()
            t_filter = time.perf_counter() - t_f0

        elif arm == "jax_cpu_diffgw":
            # diffgw generation directly on CPU
            if jitted_diffgw_cpu is not None:
                diffgw_tmpls = jitted_diffgw_cpu(m1_jnp, m2_jnp)
            else:
                diffgw_tmpls = generate_diffgw_templates(m1_vals, m2_vals, flow, df, n_freq, device=cpu_dev, precision=precision)
            diffgw_tmpls.block_until_ready()
            t_w1 = time.perf_counter()
            t_waveform = t_w1 - t_w0
            t_transfer = 0.0

            if batch_size == 1 and diffgw_tmpls.ndim == 1:
                diffgw_tmpls = diffgw_tmpls[None, :]

            # JAX CPU filter
            t_f0 = time.perf_counter()
            snr, sq = jitted_filter_cpu(diffgw_tmpls, strain_jnp_cpu, psd_jnp_cpu)
            snr.block_until_ready()
            sq.block_until_ready()
            t_filter = time.perf_counter() - t_f0

        elif arm == "jax_cuda_lal":
            if gpu_dev is None:
                raise RuntimeError("CUDA device not available for jax_cuda_lal")
            # LAL generation on host CPU
            lal_tmpls = generate_lal_templates(m1_vals, m2_vals, flow, df, n_freq, precision=precision)
            t_w1 = time.perf_counter()
            t_waveform = t_w1 - t_w0

            # Host-to-Device transfer across PCIe
            t_tr0 = time.perf_counter()
            tmpls_gpu = jax.device_put(jnp.asarray(lal_tmpls, dtype=c_dtype), gpu_dev)
            tmpls_gpu.block_until_ready()
            t_transfer = time.perf_counter() - t_tr0

            # JAX CUDA filter
            t_f0 = time.perf_counter()
            snr, sq = jitted_filter_gpu(tmpls_gpu, strain_jnp_gpu, psd_jnp_gpu)
            snr.block_until_ready()
            sq.block_until_ready()
            t_filter = time.perf_counter() - t_f0

        elif arm == "jax_cuda_diffgw":
            if gpu_dev is None:
                raise RuntimeError("CUDA device not available for jax_cuda_diffgw")
            # diffgw generation directly in GPU VRAM
            if jitted_diffgw_gpu is not None:
                diffgw_tmpls = jitted_diffgw_gpu(m1_jnp, m2_jnp)
            else:
                diffgw_tmpls = generate_diffgw_templates(m1_vals, m2_vals, flow, df, n_freq, device=gpu_dev, precision=precision)
            diffgw_tmpls.block_until_ready()
            t_w1 = time.perf_counter()
            t_waveform = t_w1 - t_w0
            t_transfer = 0.0

            if batch_size == 1 and diffgw_tmpls.ndim == 1:
                diffgw_tmpls = diffgw_tmpls[None, :]

            # JAX CUDA filter
            t_f0 = time.perf_counter()
            snr, sq = jitted_filter_gpu(diffgw_tmpls, strain_jnp_gpu, psd_jnp_gpu)
            snr.block_until_ready()
            sq.block_until_ready()
            t_filter = time.perf_counter() - t_f0
        else:
            raise ValueError(f"Unknown arm: {arm}")

        t_total = t_waveform + t_transfer + t_filter
        return {
            "t_waveform": t_waveform,
            "t_transfer": t_transfer,
            "t_filter": t_filter,
            "t_total": t_total,
        }

    # Warmup pass
    _ = execute_single_pass()

    # Timed trials
    total_samples = []
    wf_samples = []
    tr_samples = []
    filt_samples = []

    for _ in range(trials):
        res = execute_single_pass()
        total_samples.append(res["t_total"])
        wf_samples.append(res["t_waveform"])
        tr_samples.append(res["t_transfer"])
        filt_samples.append(res["t_filter"])

    tot_summary = sample_summary(total_samples, unit="seconds")
    wf_summary = sample_summary(wf_samples, unit="seconds")
    tr_summary = sample_summary(tr_samples, unit="seconds")
    filt_summary = sample_summary(filt_samples, unit="seconds")

    median_total = tot_summary["median"]
    tps_end_to_end = batch_size / median_total if median_total > 0 else 0.0
    median_filt = filt_summary["median"]
    tps_filter_only = batch_size / median_filt if median_filt > 0 else 0.0

    return {
        "arm": arm,
        "label": ARM_LABELS.get(arm, arm),
        "batch_size": batch_size,
        "trials": trials,
        "median_total_sec": median_total,
        "median_waveform_sec": wf_summary["median"],
        "median_transfer_sec": tr_summary["median"],
        "median_filter_sec": filt_summary["median"],
        "templates_per_sec": tps_end_to_end,
        "filter_templates_per_sec": tps_filter_only,
        "per_template_ms": (median_total / batch_size) * 1000.0,
        "summary": tot_summary,
        "waveform_summary": wf_summary,
        "transfer_summary": tr_summary,
        "filter_summary": filt_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="PyCBC JAX 6-Arm Performance Benchmark Harness")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="JAX CUDA device (auto, cuda:0, cpu)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 16, 32, 64],
        help="Batch sizes to evaluate",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[131072, 2097152],
        help="Transform sample lengths (131072=64s@2048Hz, 2097152=512s@4096Hz)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of timed benchmark trials per batch size",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        type=str,
        default=None,
        help="Arms to run: original_cpu branch_cpu jax_cpu_lal jax_cpu_diffgw jax_cuda_lal jax_cuda_diffgw",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip GPU arms (useful on machines without CUDA)",
    )
    parser.add_argument(
        "--merge-with",
        type=str,
        default=None,
        help="Path to existing JSON artifact to merge with",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["single", "double"],
        default="single",
        help="Arithmetic precision: 'single' (complex64/float32, production search default) or 'double' (complex128/float64)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/jax_benchmark_results.json",
        help="Output JSON artifact path",
    )
    args = parser.parse_args()

    import jax

    hw_info = get_hardware_info()
    cpu_dev = jax.devices("cpu")[0]

    gpu_devs = [d for d in jax.devices() if d.platform in ("gpu", "cuda")]
    if gpu_devs and not args.skip_gpu and args.device != "cpu":
        gpu_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
        gpu_dev = gpu_devs[min(gpu_idx, len(gpu_devs) - 1)]
    else:
        gpu_dev = None

    if args.arms is not None:
        selected_arms = args.arms
    else:
        selected_arms = ["original_cpu", "branch_cpu", "jax_cpu_lal", "jax_cpu_diffgw"]
        if gpu_dev is not None:
            selected_arms.extend(["jax_cuda_lal", "jax_cuda_diffgw"])

    print("=== PyCBC JAX 6-Arm Performance Benchmark ===")
    print(f"Host:      {hw_info['hostname']}")
    print(f"CPU:       {hw_info.get('cpu_model', 'Unknown')}")
    print(f"GPU:       {hw_info.get('gpu_model', 'None')}")
    print(f"JAX:       {jax.__version__} (devices: {jax.devices()})")
    print(f"GPU Dev:   {gpu_dev}")
    print(f"Lengths:   {args.lengths}")
    print(f"Batches:   {args.batch_sizes}")
    print(f"Arms:      {selected_arms}")
    print(f"Trials:    {args.trials}")
    print("=" * 48)

    # Load existing artifact to merge if requested
    existing_data = {}
    if args.merge_with and Path(args.merge_with).is_file():
        try:
            with open(args.merge_with, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"Loaded existing artifact for merging from: {args.merge_with}")
        except Exception as e:
            print(f"Warning: could not load {args.merge_with}: {e}")

    results = {
        "schema_version": 3,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": existing_data.get("hardware", hw_info),
        "jax_version": jax.__version__,
        "experiments": existing_data.get("experiments", {}),
    }

    for n in args.lengths:
        track_name = "streaming_n131072" if n == 131072 else "inspiral_n2097152"
        print("\n==========================================")
        print(f"--- Benchmark Track: {track_name} (N={n}) ---")
        print("==========================================")

        if track_name not in results["experiments"]:
            results["experiments"][track_name] = {
                "n_time": n,
                "arms": {},
            }
        elif "arms" not in results["experiments"][track_name]:
            results["experiments"][track_name]["arms"] = {}

        track_arms = results["experiments"][track_name]["arms"]

        for arm in selected_arms:
            print(f"\n>> Running Arm: {arm} ({ARM_LABELS.get(arm, arm)}) ...")
            if arm not in track_arms:
                track_arms[arm] = {
                    "label": ARM_LABELS.get(arm, arm),
                    "batches": {},
                }
            elif "batches" not in track_arms[arm]:
                track_arms[arm]["batches"] = {}

            for b in args.batch_sizes:
                t0_arm = time.perf_counter()
                b_res = run_benchmark_arm(
                    arm=arm,
                    n_time=n,
                    batch_size=b,
                    trials=args.trials,
                    gpu_dev=gpu_dev,
                    cpu_dev=cpu_dev,
                    precision=args.precision,
                )
                track_arms[arm]["batches"][str(b)] = b_res
                elapsed_arm = time.perf_counter() - t0_arm
                print(
                    f"  [B={b:2d}] Total: {b_res['median_total_sec']*1000:7.2f} ms "
                    f"({b_res['per_template_ms']:6.3f} ms/tmpl) -> {b_res['templates_per_sec']:8.1f} tmpl/s "
                    f"[Wf: {b_res['median_waveform_sec']*1000:6.2f}ms, "
                    f"Tr: {b_res['median_transfer_sec']*1000:5.2f}ms, "
                    f"Filt: {b_res['median_filter_sec']*1000:6.2f}ms] "
                    f"(elapsed: {elapsed_arm:5.2f}s)"
                )

        # Calculate speedups vs baseline (Arm 1: original_cpu, batch 1)
        base_arm = "original_cpu" if "original_cpu" in track_arms else "branch_cpu"
        if base_arm in track_arms and "1" in track_arms[base_arm]["batches"]:
            baseline_tps = track_arms[base_arm]["batches"]["1"]["templates_per_sec"]
        else:
            baseline_tps = 1.0

        for a_key, a_data in track_arms.items():
            for b_str, b_data in a_data["batches"].items():
                tps = b_data["templates_per_sec"]
                b_data["speedup_vs_baseline"] = tps / baseline_tps if baseline_tps > 0 else 1.0

        # Populate backward-compatible keys
        if "original_cpu" in track_arms and "1" in track_arms["original_cpu"]["batches"]:
            orig_b1 = track_arms["original_cpu"]["batches"]["1"]
            results["experiments"][track_name]["cpu_baseline"] = {
                "n_time": n,
                "sample_rate": 2048.0 if n <= 131072 else 4096.0,
                "trials": orig_b1["trials"],
                "median_sec": orig_b1["median_total_sec"],
                "templates_per_sec": orig_b1["templates_per_sec"],
                "summary": orig_b1["summary"],
            }

        # jax_scaling points to jax_cuda_diffgw if available, else jax_cpu_diffgw
        scaling_src = "jax_cuda_diffgw" if "jax_cuda_diffgw" in track_arms else "jax_cpu_diffgw"
        if scaling_src in track_arms:
            results["experiments"][track_name]["jax_scaling"] = {
                "device": str(gpu_dev if "cuda" in scaling_src else cpu_dev),
                "n_time": n,
                "sample_rate": 2048.0 if n <= 131072 else 4096.0,
                "trials": args.trials,
                "batches": track_arms[scaling_src]["batches"],
            }

    # Save sealed artifact
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sealed = seal_artifact(results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sealed, f, indent=2)
    print(f"\n[DONE] Sealed benchmark receipt written to: {out_path}")


if __name__ == "__main__":
    main()
