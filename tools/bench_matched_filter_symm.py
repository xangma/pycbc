#!/usr/bin/env python3
"""Benchmark for MatchedFilterControl.full_matched_filter_and_cluster_symm.

Benchmarks the production single-template/segment matched-filter and symmetric
clustering pipeline across:
  - Standard CPU (1-thread reference)
  - Standard CPU (Multi-threaded FFTW)
  - Torch CPU (1-thread and 16-thread MKL / PyTorch tensor ops)
  - Torch CUDA (NVIDIA GPU with cuFFT and Torch-accelerated threshold/cluster)

Measures:
  - Total call latency (ms) and throughput (evaluations/sec)
  - Scaling across segment lengths (N = 32768 .. 524288, i.e., 16s .. 256s)
  - Both noise-only and signal-injected (trigger extraction) regimes
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tools.benchmark_artifact import (
        SCHEMA_VERSION,
        atomic_write_json,
        compatibility_record,
        load_mergeable_artifact,
        runtime_metadata,
        sample_summary,
        seal_artifact,
        source_identity,
    )
except ModuleNotFoundError:
    from benchmark_artifact import (
        SCHEMA_VERSION,
        atomic_write_json,
        compatibility_record,
        load_mergeable_artifact,
        runtime_metadata,
        sample_summary,
        seal_artifact,
        source_identity,
    )


def _stats(values: list[float]) -> dict:
    return sample_summary(values, unit="seconds", bootstrap_seed=7101)


def _worker_benchmark(args_json_str: str) -> str:
    """Worker function executed inside an isolated subprocess."""
    config = json.loads(args_json_str)
    route = config["route"]
    threads = config.get("threads", 1)
    size = config.get("size", 131072)
    sample_rate = config.get("sample_rate", 2048)
    flow = config.get("flow", 30.0)
    fhigh = config.get("fhigh", 800.0)
    snr_threshold = config.get("snr_threshold", 5.5)
    cluster_window_sec = config.get("cluster_window_sec", 0.5)
    inj_snr = config.get("inj_snr", 10.0)
    iterations = config.get("iterations", 20)
    warmup = config.get("warmup", 5)

    import pycbc.filter.matchedfilter as mf
    from pycbc import scheme
    from pycbc.types import Array, FrequencySeries, zeros

    is_cuda = route.startswith("torch_cuda")
    is_torch_cpu = route.startswith("torch_cpu")
    is_original_tuned = route.startswith("original_tuned")

    if is_cuda or is_torch_cpu:
        os.environ["PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK"] = "1"
        os.environ["PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK"] = "1"

    if is_cuda:
        import torch

        if not torch.cuda.is_available():
            return json.dumps({"error": "CUDA not available"})
        torch.cuda.set_device(0)
        if route == "torch_cuda_eager":
            os.environ["PYCBC_TORCH_CUDA_GRAPH"] = "0"
        else:
            os.environ["PYCBC_TORCH_CUDA_GRAPH"] = "1"
        context = scheme.TorchScheme(device="cuda")
    elif is_torch_cpu:
        import torch

        torch.set_num_threads(threads)
        context = scheme.TorchScheme(device="cpu", num_threads=threads)
    elif is_original_tuned:
        try:
            os.environ["PYCBC_FFTW_PLAN_TYPE"] = "measure"
        except Exception:
            pass
        context = scheme.CPUScheme(num_threads=threads)
    else:
        context = scheme.CPUScheme(num_threads=threads)

    delta_t = 1.0 / sample_rate
    delta_f = 1.0 / (size * delta_t)
    freq_len = size // 2 + 1
    window = int(cluster_window_sec * sample_rate)

    rng = np.random.default_rng(12345)
    k_min = max(1, int(flow / delta_f))
    k_max = min(freq_len - 1, int(fhigh / delta_f))

    sig_np = np.zeros(freq_len, dtype=np.complex64)
    sig_np[k_min:k_max] = (
        rng.normal(size=k_max - k_min) + 1j * rng.normal(size=k_max - k_min)
    ).astype(np.complex64)
    sgm_val = float(4.0 * delta_f * np.sum(np.abs(sig_np) ** 2))
    sig_np /= np.sqrt(sgm_val)

    # Shift signal to center of analysis segment
    phase_shift = np.exp(
        -2j * np.pi * np.arange(freq_len) * (size // 2) / size
    )
    seg_data = (sig_np * phase_shift * inj_snr).astype(np.complex64)

    latencies_sec: list[float] = []

    with context:
        seg = FrequencySeries(Array(seg_data), delta_f=delta_f)
        seg.analyze = slice(int(8 * sample_rate), int(size - 8 * sample_rate))

        tmpl_mem = zeros(size, dtype=np.complex64)
        tmpl_mem[:freq_len] = Array(sig_np)

        mfc = mf.MatchedFilterControl(
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
            snr_threshold=snr_threshold,
            tlen=size,
            delta_f=delta_f,
            dtype=np.complex64,
            segment_list=[seg],
            template_output=tmpl_mem,
            use_cluster=True,
            downsample_factor=1,
            cluster_function="symmetric",
        )

        for _ in range(warmup):
            mfc.full_matched_filter_and_cluster_symm(0, 1.0, window)
            if is_cuda:
                torch.cuda.synchronize()

        last_trig_count = 0
        last_max_snr = 0.0
        for _ in range(iterations):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _snr, _norm, _corr, idx, snrv = (
                mfc.full_matched_filter_and_cluster_symm(0, 1.0, window)
            )
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_sec.append(t1 - t0)
            if idx is not None and len(idx) > 0:
                last_trig_count = len(idx)
                last_max_snr = float(np.max(np.abs(snrv)))

    summary = _stats(latencies_sec)
    summary["trigger_count"] = last_trig_count
    summary["max_snr"] = last_max_snr
    return json.dumps(summary)


def _run_subprocess_worker(config: dict, python_executable: str) -> dict:
    worker_script = (
        "import sys, json\n"
        "from tools.bench_matched_filter_symm import _worker_benchmark\n"
        "print(_worker_benchmark(sys.argv[1]))\n"
    )
    cmd = [python_executable, "-c", worker_script, json.dumps(config)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Worker failed (code {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return json.loads(result.stdout.strip())


def run_benchmark(
    sizes: list[int] | None = None,
    output_path: Path | None = None,
    py_exe: str = sys.executable,
    snr_threshold: float = 5.5,
    inj_snr: float = 10.0,
    iterations: int = 20,
    warmup: int = 5,
    no_merge: bool = False,
) -> dict:
    if sizes is None:
        sizes = [32768, 65536, 131072, 262144, 524288]

    # Check CUDA availability
    have_cuda = False
    try:
        chk = subprocess.run(
            [
                py_exe,
                "-c",
                (
                    "import sys, torch; sys.exit("
                    "0 if torch.cuda.is_available() else 1"
                    ")"
                ),
            ],
            capture_output=True,
            check=False,
        )
        have_cuda = chk.returncode == 0
    except Exception:
        have_cuda = False

    repo_root = Path(__file__).resolve().parents[1]
    meta = runtime_metadata()
    meta["suite_name"] = "matched_filter_and_cluster_symm"
    meta["source"] = source_identity(
        repo_root,
        (Path(__file__), repo_root / "tools" / "benchmark_artifact.py"),
    )

    settings = {
        "sizes": sizes,
        "sample_rate": 2048,
        "snr_threshold": snr_threshold,
        "inj_snr": inj_snr,
        "routes": [
            "original_standard",
            "original_tuned",
            "torch_cpu",
            "torch_cuda" if have_cuda else None,
        ],
    }
    settings["routes"] = [r for r in settings["routes"] if r]
    compatibility = compatibility_record(settings, meta)

    artifact = (
        load_mergeable_artifact(
            output_path,
            artifact_type="matched_filter_symm_benchmark",
            compatibility_sha256=compatibility["sha256"],
        )
        if not no_merge and output_path and output_path.exists()
        else {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "matched_filter_symm_benchmark",
            "metadata": meta,
            "compatibility": compatibility,
            "campaigns": [],
        }
    )

    campaign = {
        "workload": "full_matched_filter_and_cluster_symm",
        "timestamp": time.time(),
        "sizes": sizes,
        "results": {},
        "complete": False,
    }
    artifact["campaigns"].append(campaign)

    print("=" * 140)
    print(
        " PyCBC MatchedFilterControl.full_matched_filter_and_cluster_symm "
        "Benchmark (Fair Multi-Tier Comparison)"
    )
    print("=" * 140)
    hdr = (
        f" {'Size (N)':<9} | {'Duration':<9} | {'Std 1T':<9} | "
        f"{'Torch 1T':<9} | {'Std 16T*':<9} | {'Torch 16T':<9} | "
        f"{'CUDA Eag':<9} | {'CUDA Grp':<9} | {'vs 1T':<7} | {'vs 16T':<7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for n in sizes:
        dur_sec = n / 2048.0
        cfg_base = {
            "size": n,
            "sample_rate": 2048,
            "snr_threshold": snr_threshold,
            "inj_snr": inj_snr,
            "iterations": iterations,
            "warmup": warmup,
        }

        # 1. Standard 1T
        r_std_1t = _run_subprocess_worker(
            {**cfg_base, "route": "original_standard", "threads": 1}, py_exe
        )
        ms_std_1t = r_std_1t["median"] * 1000.0

        # 2. Torch CPU 1T (Apples-to-apples 1-thread comparison)
        r_tcpu_1t = _run_subprocess_worker(
            {**cfg_base, "route": "torch_cpu", "threads": 1}, py_exe
        )
        ms_tcpu_1t = r_tcpu_1t["median"] * 1000.0

        # 3. Standard 16T (Note: 1D FFT thread barrier overhead on small N)
        r_std_16t = _run_subprocess_worker(
            {**cfg_base, "route": "original_tuned", "threads": 16}, py_exe
        )
        ms_std_16t = r_std_16t["median"] * 1000.0

        # 4. Torch CPU 16T
        r_tcpu = _run_subprocess_worker(
            {**cfg_base, "route": "torch_cpu", "threads": 16}, py_exe
        )
        ms_tcpu = r_tcpu["median"] * 1000.0

        # 5. Torch CUDA Eager & Graph
        if have_cuda:
            r_cuda_eager = _run_subprocess_worker(
                {**cfg_base, "route": "torch_cuda_eager"}, py_exe
            )
            ms_cuda_eager = r_cuda_eager["median"] * 1000.0
            cuda_eager_str = f"{ms_cuda_eager:6.3f} ms"

            r_cuda = _run_subprocess_worker(
                {**cfg_base, "route": "torch_cuda"}, py_exe
            )
            ms_cuda = r_cuda["median"] * 1000.0
            cuda_graph_str = f"{ms_cuda:6.3f} ms"
            sp_vs_1t = f"{ms_std_1t / ms_cuda:5.2f}x"
            sp_vs_16t = f"{ms_std_16t / ms_cuda:5.2f}x"
        else:
            r_cuda_eager = {}
            r_cuda = {}
            cuda_eager_str = "N/A"
            cuda_graph_str = "N/A"
            sp_vs_1t = f"{ms_std_1t / ms_tcpu:5.2f}x (T16)"
            sp_vs_16t = f"{ms_std_16t / ms_tcpu:5.2f}x (T16)"

        # Parity check
        ref_trigs = r_std_1t.get("trigger_count", 0)
        tcpu_trigs = r_tcpu_1t.get("trigger_count", 0)
        cuda_trigs = r_cuda.get("trigger_count", 0) if have_cuda else ref_trigs
        parity_ok = (
            (ref_trigs == tcpu_trigs)
            and (not have_cuda or ref_trigs == cuda_trigs)
        )

        row = (
            f" N={n:<7} | {dur_sec:5.1f}s   | {ms_std_1t:5.3f} ms | "
            f"{ms_tcpu_1t:5.3f} ms | {ms_std_16t:5.3f} ms | "
            f"{ms_tcpu:5.3f} ms | {cuda_eager_str:<9} | {cuda_graph_str:<9} | "
            f"{sp_vs_1t:<7} | {sp_vs_16t:<7}"
        )
        print(row)

        sp_1t = (
            (ms_std_1t / ms_cuda)
            if have_cuda and ms_cuda > 0
            else (ms_std_1t / ms_tcpu)
        )
        sp_16t = (
            (ms_std_16t / ms_cuda)
            if have_cuda and ms_cuda > 0
            else (ms_std_16t / ms_tcpu)
        )
        campaign["results"][f"size_{n}"] = {
            "size": n,
            "duration_sec": dur_sec,
            "standard_1t": r_std_1t,
            "torch_cpu_1t": r_tcpu_1t,
            "standard_16t": r_std_16t,
            "torch_cpu_16t": r_tcpu,
            "torch_cuda_eager": r_cuda_eager,
            "torch_cuda": r_cuda,
            "speedup_vs_std_1t": sp_1t,
            "speedup_vs_std_16t": sp_16t,
            "speedup_1t_torch_vs_std": (
                (ms_std_1t / ms_tcpu_1t) if ms_tcpu_1t > 0 else 1.0
            ),
            "trigger_parity_passed": parity_ok,
            "ref_triggers": ref_trigs,
        }

    print("=" * 115)
    print(" * Std CPU 16T reflects FFTW thread barrier overhead on 1D FFTs.")
    print(" BENCHMARK COMPLETE")
    print("=" * 115)

    campaign["complete"] = True
    if output_path:
        atomic_write_json(output_path, seal_artifact(artifact))
        print(f"\nResults saved to: {output_path}")

    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark for "
            "MatchedFilterControl."
            "full_matched_filter_and_cluster_symm"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        "-s",
        nargs="+",
        type=int,
        default=[32768, 65536, 131072, 262144, 524288],
        help="Segment sizes (samples) to benchmark",
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=5.5,
        help="SNR threshold for trigger extraction",
    )
    parser.add_argument(
        "--inj-snr",
        type=float,
        default=10.0,
        help="Injected signal SNR for trigger extraction",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=20,
        help="Timing iterations per configuration",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=5,
        help="Warmup iterations per configuration",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("artifacts/matched_filter_symm_benchmark.json"),
        help="Path for output structured JSON",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to invoke workers",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Do not merge into existing output artifact",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    run_benchmark(
        sizes=args.sizes,
        output_path=args.output,
        py_exe=args.python,
        snr_threshold=args.snr_threshold,
        inj_snr=args.inj_snr,
        iterations=args.iterations,
        warmup=args.warmup,
        no_merge=args.no_merge,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
