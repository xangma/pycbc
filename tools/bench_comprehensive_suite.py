#!/usr/bin/env python3
"""Component microbenchmarks for standard PyCBC and Torch CPU/CUDA routes.

Pushes the original PyCBC code to its performance limits:
- FFTW wisdom pre-calibration (FFTW_MEASURE / FFTW_PATIENT)
- Multi-threaded CPU scaling (1, 4, 8, 16 threads)
- Cython SIMD thresholding & Cython matched filtering

Compares against PyCBC Torch:
- Single-template & Batched (B = 1, 2, 4, 8, 16, 32, 64)
- Torch CPU (Native Batch Correlate, FFTW/MKL Batching, Vector Peak Scan)
- Torch CUDA (2D tensor correlate and GPU-side peak scan)

Measures:
1. Waveform Generation throughput (templates/sec)
2. Pure Correlation + IFFT throughput (transforms/sec & GFlops)
3. Peak Finding / Clustering throughput (MSamples/sec)
4. Synthetic LiveBatchMatchedFilter microbenchmark latency and throughput

The fourth workload uses generated PSDs, templates, and strain.  It is useful
for component attribution but is not the counterbalanced production benchmark;
use ``bench_production_live_batch.py`` for production evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
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
except ModuleNotFoundError:  # Direct execution from the tools directory.
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
    batch_size = config.get("batch_size", 1)
    size = config.get("size", 131072)
    workload = config["workload"]
    iterations = config.get("iterations", 10)
    warmup = config.get("warmup", 3)
    cuda_device = config.get("cuda_device", 0)

    import pycbc
    import pycbc.filter.matchedfilter as mf
    import pycbc.waveform
    from pycbc import scheme
    from pycbc.types import Array, FrequencySeries, zeros

    is_cuda = route == "torch_cuda"
    is_torch_cpu = route.startswith("torch_cpu")
    is_original_tuned = route.startswith("original_tuned")

    # Set up environment flags
    if is_cuda or is_torch_cpu:
        os.environ["PYCBC_IMRPHENOMXAS_EXACT_SCALAR_DERIVATIVES"] = "1"
        os.environ["PYCBC_IMRPHENOMXAS_EXACT_SCALAR_AMP_DERIVATIVES"] = "1"
        os.environ["PYCBC_IMRPHENOMXAS_REGION_PRUNING"] = "1"
        os.environ["PYCBC_IMRPHENOMXAS_SCALAR_DERIVATIVE_PLAN_CSE"] = "1"
        os.environ["PYCBC_IMRPHENOMXAS_PHASE_PLAN_BULK_COLLOCATION"] = "1"
        os.environ["PYCBC_IMRPHENOMXPHM_TWIST_REUSE"] = "1"
        os.environ["PYCBC_IMRPHENOMXPHM_BULK_TWIST_EXPONENTIALS"] = "1"
        os.environ["PYCBC_IMRPHENOMXPHM_TWIST_EXPONENTIAL_RECURRENCE"] = "1"
        os.environ["PYCBC_IMRPHENOMXPHM_BULK_TWIST_HARMONICS"] = "1"
        os.environ["PYCBC_IMRPHENOMXPHM_STACKED_TWIST"] = "1"

    if is_cuda:
        import torch

        torch.cuda.set_device(cuda_device)
        torch.cuda.empty_cache()
        torch.set_grad_enabled(False)
        os.environ["PYCBC_TORCH_CUDA_NATIVE_BATCH_CORRELATE"] = "1"
        os.environ["PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK"] = "1"
        os.environ["PYCBC_TORCH_CUDA_PROMOTED_ROWS"] = "1"
        context = scheme.TorchScheme(f"cuda:{cuda_device}")
    elif is_torch_cpu:
        import torch

        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        os.environ["PYCBC_TORCH_CPU_NATIVE_BATCH_CORRELATE"] = "1"
        os.environ["PYCBC_TORCH_CPU_FFTW_BATCH"] = "1"
        os.environ["PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK"] = "1"
        context = scheme.TorchScheme("cpu")
    elif is_original_tuned:
        # Pushing original FFTW to its limits
        try:
            from pycbc.fft import fftw

            # Measure / patient planning for original FFTW
            os.environ["PYCBC_FFTW_PLAN_TYPE"] = "measure"
        except Exception:
            pass
        context = scheme.CPUScheme(num_threads=threads)
    else:
        context = scheme.CPUScheme(num_threads=threads)

    freq_len = size // 2 + 1
    sample_rate = 2048.0
    delta_f = sample_rate / size
    rng = np.random.default_rng(config.get("seed", 42))

    latencies_sec = []

    # -------------------------------------------------------------
    # Workload 1: Waveform Generation
    # -------------------------------------------------------------
    if workload == "waveform":
        approx = config.get("approximant", "IMRPhenomD")
        params = {
            "mass1": 30.0,
            "mass2": 25.0,
            "spin1z": 0.4,
            "spin2z": -0.3,
            "f_lower": 20.0,
            "delta_f": delta_f,
            "distance": 500.0,
        }
        if "XP" in approx:
            params.update(
                {
                    "spin1x": 0.2,
                    "spin1y": 0.1,
                    "spin2x": -0.1,
                    "spin2y": 0.05,
                    "inclination": 0.7,
                }
            )

        with context:
            for _ in range(warmup):
                hp, hc = pycbc.waveform.get_fd_waveform(approximant=approx, **params)
                if is_cuda:
                    torch.cuda.synchronize()

            for _ in range(iterations):
                t0 = time.perf_counter()
                for _ in range(batch_size):
                    hp, hc = pycbc.waveform.get_fd_waveform(
                        approximant=approx, **params
                    )
                if is_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies_sec.append((t1 - t0) / batch_size)

    # -------------------------------------------------------------
    # Workload 2: Pure Matched Filter Correlate + IFFT Core
    # -------------------------------------------------------------
    elif workload == "correlate_ifft":
        with context:
            x_raw = (
                rng.normal(size=(batch_size, freq_len))
                + 1j * rng.normal(size=(batch_size, freq_len))
            ).astype(np.complex64)
            y_raw = (rng.normal(size=freq_len) + 1j * rng.normal(size=freq_len)).astype(
                np.complex64
            )

            xs = [Array(x_raw[b]) for b in range(batch_size)]
            y = Array(y_raw)
            correlation_memory = zeros(batch_size * size, dtype=np.complex64)
            output_memory = zeros(batch_size * size, dtype=np.complex64)
            zs = [
                correlation_memory[index * size : (index + 1) * size]
                for index in range(batch_size)
            ]
            correlator = mf.BatchCorrelator(xs, zs, freq_len)

            if is_cuda or is_torch_cpu:
                from pycbc.fft import torchfft

                engine_type = torchfft.IFFT
            else:
                from pycbc.fft import fftw

                engine_type = fftw.IFFT

            ifft_engine = engine_type(
                correlation_memory, output_memory, nbatch=batch_size, size=size
            )

            for _ in range(warmup):
                correlator.execute(y)
                ifft_engine.execute()
                if is_cuda:
                    torch.cuda.synchronize()

            for _ in range(iterations):
                t0 = time.perf_counter()
                correlator.execute(y)
                ifft_engine.execute()
                if is_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies_sec.append(t1 - t0)

    # -------------------------------------------------------------
    # Workload 3: Pure Peak Finding / Thresholding
    # -------------------------------------------------------------
    elif workload == "peak_finding":
        with context:
            output_memory = zeros(batch_size * size, dtype=np.complex64)
            output_rows = [
                output_memory[index * size : (index + 1) * size]
                for index in range(batch_size)
            ]

            if is_cuda:
                os.environ["PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK"] = "1"
            elif is_torch_cpu:
                os.environ["PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK"] = "1"

            if is_cuda or is_torch_cpu:
                from pycbc.filter.matchedfilter import _torch_batch_peak_values

                def _do_peaks():
                    return _torch_batch_peak_values(
                        output_memory, batch_size, size, slice(0, size)
                    )
            else:

                def _do_peaks():
                    indices = np.empty(batch_size, dtype=np.int64)
                    values = np.empty(batch_size, dtype=np.complex64)
                    for index, output in enumerate(output_rows):
                        loc = output.abs_arg_max()
                        indices[index] = loc
                        values[index] = output[loc]
                    return indices, values

            for _ in range(warmup):
                _do_peaks()
                if is_cuda:
                    torch.cuda.synchronize()

            for _ in range(iterations):
                t0 = time.perf_counter()
                _do_peaks()
                if is_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies_sec.append(t1 - t0)

    # -------------------------------------------------------------
    # Workload 4: End-to-End LiveBatchMatchedFilter Pipeline
    # -------------------------------------------------------------
    elif workload in ("synthetic_live_batch", "live_pipeline"):
        import types

        num_blocks = config.get("num_blocks", 10)
        with context:
            # 1. Analytic PSD
            psd_vals = np.ones(freq_len, dtype=np.float32)
            freqs = np.linspace(0, sample_rate / 2.0, freq_len)
            with np.errstate(divide="ignore"):
                psd_vals = (np.maximum(freqs, 20.0) / 100.0) ** (-4.0) + 1.0
            psd = FrequencySeries(psd_vals, delta_f=delta_f)

            # 2. Bandlimited templates normalized by sigmasq
            k_min = max(1, int(30.0 / delta_f))
            k_max = min(freq_len - 1, int(800.0 / delta_f))
            templates_np = []
            for _i in range(batch_size):
                t_raw = np.zeros(freq_len, dtype=np.complex64)
                t_raw[k_min:k_max] = (
                    rng.normal(size=k_max - k_min) + 1j * rng.normal(size=k_max - k_min)
                ).astype(np.complex64)
                sgm_val = float(4.0 * delta_f * np.sum(np.abs(t_raw) ** 2 / psd_vals))
                t_raw /= np.sqrt(sgm_val)
                templates_np.append(t_raw)

            templates = []
            for i in range(batch_size):
                fs = FrequencySeries(templates_np[i], delta_f=delta_f)
                fs.id = 1000 + i
                fs.params = np.array(
                    [(10.0 + i * 0.5, 10.0 + i * 0.5)],
                    dtype=[("mass1", np.float32), ("mass2", np.float32)],
                )[0]
                fs.sigmasq = lambda _p, raw=templates_np[i]: float(
                    4.0 * delta_f * np.sum(np.abs(raw) ** 2 / psd_vals)
                )
                templates.append(fs)

            # 3. Stream data blocks with injections
            data_readers = []
            for b in range(num_blocks):
                noise = (
                    rng.normal(size=freq_len) + 1j * rng.normal(size=freq_len)
                ).astype(np.complex64) * 0.001
                inj_tmpl_idx = b % batch_size
                inj_snr = 8.0 + float(b % 4) * 1.5
                t0 = 20000 + (b * 1337) % 50000
                phase_shift = np.exp(-2j * np.pi * np.arange(freq_len) * t0 / size)
                noise += (templates_np[inj_tmpl_idx] * inj_snr * phase_shift).astype(
                    np.complex64
                )

                stilde = FrequencySeries(noise, delta_f=delta_f)
                stilde.psd = psd
                reader = types.SimpleNamespace(
                    overwhitened_data=lambda _df, s=stilde: s,
                    trim_padding=4096,
                    blocksize=56.0,
                    sample_rate=sample_rate,
                    start_time=1000000000.0 + b * 56.0,
                )
                data_readers.append(reader)

            filter_engine = mf.LiveBatchMatchedFilter(
                templates,
                snr_threshold=5.5,
                chisq_bins=0,
                sg_chisq=types.SimpleNamespace(values=lambda *args: None),
            )

            # Warmup
            for reader in data_readers[:2]:
                filter_engine.process_data(reader)
                if is_cuda:
                    torch.cuda.synchronize()

            # Timed runs
            for reader in data_readers:
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                filter_engine.process_data(reader)
                if is_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies_sec.append(t1 - t0)

    # -------------------------------------------------------------
    # Workload 5: Multi-Detector Network Response & Delays
    # -------------------------------------------------------------
    elif workload == "detector_network":
        from pycbc.detector import Detector, NetworkGeometry

        num_sky = config.get("num_sky", 1000)
        net_ifos = config.get("ifos", ["H1", "L1", "V1"])
        ras = rng.uniform(0, 2 * np.pi, num_sky)
        decs = rng.uniform(-np.pi / 2, np.pi / 2, num_sky)
        psis = rng.uniform(0, np.pi, num_sky)
        times = np.full(num_sky, 1000000000.0)

        if route == "original_standard":
            detectors = [Detector(ifo) for ifo in net_ifos]
            for _ in range(warmup):
                for i in range(min(num_sky, 10)):
                    for d in detectors:
                        _ = d.antenna_pattern(ras[i], decs[i], psis[i], times[i])
                        _ = d.time_delay_from_earth_center(
                            ras[i], decs[i], times[i]
                        )
            for _ in range(iterations):
                t0 = time.perf_counter()
                for i in range(num_sky):
                    for d in detectors:
                        _ = d.antenna_pattern(ras[i], decs[i], psis[i], times[i])
                        _ = d.time_delay_from_earth_center(
                            ras[i], decs[i], times[i]
                        )
                t1 = time.perf_counter()
                latencies_sec.append(t1 - t0)
        else:
            net = NetworkGeometry(net_ifos)
            for _ in range(warmup):
                _ = net.antenna_pattern_and_time_delay(ras, decs, psis, times)
            for _ in range(iterations):
                t0 = time.perf_counter()
                _ = net.antenna_pattern_and_time_delay(ras, decs, psis, times)
                t1 = time.perf_counter()
                latencies_sec.append(t1 - t0)

    # -------------------------------------------------------------
    # Workload 6: Batched Relative Binning Summary Likelihood
    # -------------------------------------------------------------
    elif workload == "relbin_likelihood":
        from pycbc.inference.models import relbin_torch
        import torch

        n_bins = 64
        num_samples = config.get("num_samples", 1000)
        fedges = torch.linspace(20.0, 1024.0, n_bins + 1, dtype=torch.float64)
        hp_edge = torch.randn(n_bins + 1, dtype=torch.complex128)
        hc_edge = torch.randn(n_bins + 1, dtype=torch.complex128)
        h00_edge = torch.randn(n_bins + 1, dtype=torch.float64)
        a0 = torch.randn(n_bins, dtype=torch.complex128)
        a1 = torch.randn(n_bins, dtype=torch.complex128)
        b0 = torch.randn(n_bins, dtype=torch.float64)
        b1 = torch.randn(n_bins, dtype=torch.float64)
        fp = torch.linspace(0.1, 1.0, num_samples, dtype=torch.float64)
        fc = torch.linspace(-0.5, 0.5, num_samples, dtype=torch.float64)
        dtc = torch.linspace(-0.01, 0.01, num_samples, dtype=torch.float64)

        if route == "original_standard":
            for _ in range(warmup):
                for i in range(min(num_samples, 10)):
                    _ = relbin_torch.likelihood_parts(
                        fedges, fp[i], fc[i], dtc[i],
                        hp_edge, hc_edge, h00_edge, a0, a1, b0, b1
                    )
            for _ in range(iterations):
                t0 = time.perf_counter()
                for i in range(num_samples):
                    _ = relbin_torch.likelihood_parts(
                        fedges, fp[i], fc[i], dtc[i],
                        hp_edge, hc_edge, h00_edge, a0, a1, b0, b1
                    )
                t1 = time.perf_counter()
                latencies_sec.append(t1 - t0)
        else:
            for _ in range(warmup):
                _ = relbin_torch.likelihood_parts(
                    fedges, fp, fc, dtc,
                    hp_edge, hc_edge, h00_edge, a0, a1, b0, b1
                )
            for _ in range(iterations):
                t0 = time.perf_counter()
                _ = relbin_torch.likelihood_parts(
                    fedges, fp, fc, dtc,
                    hp_edge, hc_edge, h00_edge, a0, a1, b0, b1
                )
                t1 = time.perf_counter()
                latencies_sec.append(t1 - t0)

    # -------------------------------------------------------------
    # Workload 7: MatchedFilterControl.full_matched_filter_and_cluster_symm
    # -------------------------------------------------------------
    elif workload in ("matched_filter_symm", "full_matched_filter_and_cluster_symm"):
        with context:
            sample_rate = config.get("sample_rate", 2048)
            flow = config.get("flow", 30.0)
            fhigh = config.get("fhigh", 800.0)
            snr_threshold = config.get("snr_threshold", 5.5)
            inj_snr = config.get("inj_snr", 10.0)
            cluster_window_sec = config.get("cluster_window_sec", 0.5)

            delta_t = 1.0 / sample_rate
            delta_f = 1.0 / (size * delta_t)
            freq_len = size // 2 + 1
            window = int(cluster_window_sec * sample_rate)

            k_min = max(1, int(flow / delta_f))
            k_max = min(freq_len - 1, int(fhigh / delta_f))

            sig_np = np.zeros(freq_len, dtype=np.complex64)
            sig_np[k_min:k_max] = (
                rng.normal(size=k_max - k_min) + 1j * rng.normal(size=k_max - k_min)
            ).astype(np.complex64)
            sgm_val = float(4.0 * delta_f * np.sum(np.abs(sig_np) ** 2))
            sig_np /= np.sqrt(sgm_val)

            phase_shift = np.exp(-2j * np.pi * np.arange(freq_len) * (size // 2) / size)
            seg_data = (sig_np * phase_shift * inj_snr).astype(np.complex64)

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

            for _ in range(iterations):
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                mfc.full_matched_filter_and_cluster_symm(0, 1.0, window)
                if is_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies_sec.append(t1 - t0)


    stats = _stats(latencies_sec)
    return json.dumps(stats)


def _run_subprocess_worker(config: dict, py_exe: str) -> dict:
    cmd = [
        py_exe,
        "-c",
        "import sys, json; from tools.bench_comprehensive_suite import _worker_benchmark; print(_worker_benchmark(sys.argv[1]))",
        json.dumps(config),
    ]
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    current_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{current_pp}" if current_pp else repo_root
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess worker failed for {config}:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )
    stdout = result.stdout.strip()
    # Find JSON payload in stdout
    for line in reversed(stdout.splitlines()):
        line_s = line.strip()
        if line_s.startswith("{") and line_s.endswith("}"):
            return json.loads(line_s)
    return json.loads(stdout)


def _probe_worker_runtime(py_exe: str) -> dict:
    source = """
import json
import platform
import sys

result = {
    "executable": sys.executable,
    "python": platform.python_version(),
    "platform": platform.platform(),
}
try:
    import torch
except ImportError:
    result.update({"torch": None, "torch_cuda": None, "cuda_available": False})
else:
    available = bool(torch.cuda.is_available())
    result.update({
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": available,
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ] if available else [],
    })
print(json.dumps(result, sort_keys=True))
"""
    process = subprocess.run(
        [py_exe, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(
            f"failed to probe benchmark interpreter {py_exe}: {process.stderr.strip()}"
        )
    return json.loads(process.stdout)


FAMILIES = {
    "all": [
        "TaylorF2",
        "IMRPhenomD",
        "IMRPhenomXAS",
        "IMRPhenomXHM",
        "IMRPhenomXP",
        "IMRPhenomXPHM",
    ],
    "phenomx": [
        "IMRPhenomD",
        "IMRPhenomXAS",
        "IMRPhenomXHM",
        "IMRPhenomXP",
        "IMRPhenomXPHM",
    ],
    "phenom": [
        "IMRPhenomD",
        "IMRPhenomXAS",
        "IMRPhenomXHM",
        "IMRPhenomXP",
        "IMRPhenomXPHM",
    ],
    "pn": [
        "TaylorF2",
    ],
    "taylor": [
        "TaylorF2",
    ],
    "aligned": [
        "TaylorF2",
        "IMRPhenomD",
        "IMRPhenomXAS",
    ],
    "precessing": [
        "IMRPhenomXP",
        "IMRPhenomXPHM",
    ],
    "higher_modes": [
        "IMRPhenomXHM",
        "IMRPhenomXPHM",
    ],
    "hm": [
        "IMRPhenomXHM",
        "IMRPhenomXPHM",
    ],
}


def run_full_suite(
    py_exe: str = sys.executable,
    output_path: Path | None = None,
    workloads: list[str] | None = None,
    approximants: list[str] | None = None,
    batch_sizes: list[int] | None = None,
    no_merge: bool = False,
) -> dict:
    if workloads is None or "all" in workloads:
        workloads = [
            "waveform",
            "correlate_ifft",
            "peak_finding",
            "synthetic_live_batch",
            "matched_filter_symm",
            "detector_network",
            "relbin_likelihood",
        ]
    workload_aliases = {
        "live_pipeline": "synthetic_live_batch",
        "pipeline": "synthetic_live_batch",
        "full_matched_filter_and_cluster_symm": "matched_filter_symm",
    }
    workloads = list(
        dict.fromkeys(
            workload_aliases.get(workload, workload) for workload in workloads
        )
    )
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    if approximants is None:
        approximants = FAMILIES["all"]

    worker_runtime = _probe_worker_runtime(py_exe)
    have_cuda = worker_runtime["cuda_available"]
    print("=" * 90)
    print(" PYCBC COMPONENT MICROBENCHMARKS: STANDARD vs TORCH CPU & CUDA")
    print(
        f" Host: {platform.node()} ({platform.machine()}), "
        f"Worker Python: {worker_runtime['python']}"
    )
    print(f" PyTorch: {worker_runtime['torch'] or 'N/A'}")
    if have_cuda:
        print(
            f" CUDA Devices: {', '.join(worker_runtime['cuda_devices'])} "
            f"(CUDA {worker_runtime['torch_cuda']})"
        )
    print(f" Active Workloads: {', '.join(workloads)}")
    print("=" * 90)

    repo_root = Path(__file__).resolve().parents[1]
    metadata = runtime_metadata()
    metadata["worker_runtime"] = worker_runtime
    metadata["source"] = source_identity(
        repo_root,
        (Path(__file__), repo_root / "tools" / "benchmark_artifact.py"),
    )
    settings = {
        "batch_sizes": batch_sizes,
        "sample_size": 131072,
        "waveform_batch_size": 16,
        "routes": [
            "original_standard",
            "original_tuned",
            "torch_cpu",
            "torch_cuda" if have_cuda else None,
        ],
        "worker_runtime": worker_runtime,
    }
    settings["routes"] = [route for route in settings["routes"] if route]
    compatibility = compatibility_record(settings, metadata)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "component_microbenchmarks",
        "metadata": metadata,
        "compatibility": compatibility,
        "campaigns": [],
        "results": {},
    }
    if not no_merge and output_path and output_path.exists():
        artifact = load_mergeable_artifact(
            output_path,
            artifact_type="component_microbenchmarks",
            compatibility_sha256=compatibility["sha256"],
        )
        artifact["metadata"] = metadata
    artifact.setdefault("campaigns", []).append(
        {
            "timestamp_utc": metadata["timestamp_utc"],
            "workloads": workloads,
            "approximants": approximants,
            "complete": False,
        }
    )
    results = artifact.setdefault("results", {})
    results.setdefault("waveform_generation", {})
    results.setdefault("correlate_ifft", {})
    results.setdefault("peak_finding", {})
    results.setdefault("synthetic_live_batch", {})
    results.setdefault("detector_network", {})
    results.setdefault("relbin_likelihood", {})
    results.setdefault("matched_filter_symm", {})

    # -------------------------------------------------------------
    # 1. Waveform Generation Benchmark
    # -------------------------------------------------------------
    if "waveform" in workloads:
        print("\n" + "-" * 90)
        print(" 1. WAVEFORM GENERATION THROUGHPUT (Templates / Second)")
        print("-" * 90)
        print(
            f" {'Approximant':<16} | {'Orig CPU (1T)':<14} | {'Orig Max (16T)':<14} | {'Torch CPU (16T)':<15} | {'Torch CUDA':<14} | {'CUDA Speedup':<12}"
        )
        print("-" * 90)

        for approx in approximants:
            cfg_base = {
                "workload": "waveform",
                "approximant": approx,
                "batch_size": 16,
                "iterations": 5,
                "warmup": 2,
            }

            # Orig 1T
            r_orig_1t = _run_subprocess_worker(
                {**cfg_base, "route": "original_standard", "threads": 1}, py_exe
            )
            rate_orig_1t = 1.0 / r_orig_1t["median"] if r_orig_1t["median"] > 0 else 0

            # Orig Max (16T)
            r_orig_16t = _run_subprocess_worker(
                {**cfg_base, "route": "original_tuned", "threads": 16}, py_exe
            )
            rate_orig_16t = (
                1.0 / r_orig_16t["median"] if r_orig_16t["median"] > 0 else 0
            )

            # Torch CPU (16T)
            r_tcpu = _run_subprocess_worker(
                {**cfg_base, "route": "torch_cpu", "threads": 16}, py_exe
            )
            rate_tcpu = 1.0 / r_tcpu["median"] if r_tcpu["median"] > 0 else 0

            # Torch CUDA
            if have_cuda:
                r_cuda = _run_subprocess_worker(
                    {**cfg_base, "route": "torch_cuda"}, py_exe
                )
                rate_cuda = 1.0 / r_cuda["median"] if r_cuda["median"] > 0 else 0
                cuda_str = f"{rate_cuda:8.1f} wf/s"
                speedup = f"{rate_cuda / rate_orig_1t:5.2f}x"
            else:
                r_cuda = {}
                cuda_str = "N/A"
                speedup = f"{rate_tcpu / rate_orig_1t:5.2f}x (CPU)"

            print(
                f" {approx:<16} | {rate_orig_1t:8.1f} wf/s   | {rate_orig_16t:8.1f} wf/s   | {rate_tcpu:8.1f} wf/s    | {cuda_str:<14} | {speedup:<12}"
            )

            results["waveform_generation"][approx] = {
                "original_1t": r_orig_1t,
                "original_16t": r_orig_16t,
                "torch_cpu_16t": r_tcpu,
                "torch_cuda": r_cuda,
            }

    # -------------------------------------------------------------
    # 2. Pure Matched Filter Correlate + IFFT Throughput
    # -------------------------------------------------------------
    if "correlate_ifft" in workloads:
        print("\n" + "-" * 90)
        print(
            " 2. PURE MATCHED FILTER CORRELATION + IFFT THROUGHPUT (N=131072, 64s at 2048Hz)"
        )
        print("-" * 90)
        print(
            f" {'Batch (B)':<10} | {'Orig CPU (1T)':<14} | {'Orig Max (16T)':<14} | {'Torch CPU (16T)':<15} | {'Torch CUDA':<14} | {'CUDA Speedup':<12}"
        )
        print("-" * 90)

        for b in batch_sizes:
            cfg_base = {
                "workload": "correlate_ifft",
                "size": 131072,
                "batch_size": b,
                "iterations": 15 if b < 128 else 6,
                "warmup": 5 if b < 128 else 2,
            }

            # Orig 1T
            r_orig_1t = _run_subprocess_worker(
                {**cfg_base, "route": "original_standard", "threads": 1}, py_exe
            )
            rate_orig_1t = b / r_orig_1t["median"] if r_orig_1t["median"] > 0 else 0

            # Orig Max (16T)
            r_orig_16t = _run_subprocess_worker(
                {**cfg_base, "route": "original_tuned", "threads": 16}, py_exe
            )
            rate_orig_16t = b / r_orig_16t["median"] if r_orig_16t["median"] > 0 else 0

            # Torch CPU (16T)
            r_tcpu = _run_subprocess_worker(
                {**cfg_base, "route": "torch_cpu", "threads": 16}, py_exe
            )
            rate_tcpu = b / r_tcpu["median"] if r_tcpu["median"] > 0 else 0

            # Torch CUDA
            if have_cuda:
                r_cuda = _run_subprocess_worker(
                    {**cfg_base, "route": "torch_cuda"}, py_exe
                )
                rate_cuda = b / r_cuda["median"] if r_cuda["median"] > 0 else 0
                cuda_str = f"{rate_cuda:8.1f} xf/s"
                speedup = f"{rate_cuda / rate_orig_1t:5.2f}x"
            else:
                r_cuda = {}
                cuda_str = "N/A"
                speedup = f"{rate_tcpu / rate_orig_1t:5.2f}x (CPU)"

            print(
                f" B = {b:<6} | {rate_orig_1t:8.1f} xf/s   | {rate_orig_16t:8.1f} xf/s   | {rate_tcpu:8.1f} xf/s    | {cuda_str:<14} | {speedup:<12}"
            )

            results["correlate_ifft"][f"batch_{b}"] = {
                "original_1t": r_orig_1t,
                "original_16t": r_orig_16t,
                "torch_cpu_16t": r_tcpu,
                "torch_cuda": r_cuda,
            }

    # -------------------------------------------------------------
    # 3. Peak Finding & Thresholding Throughput
    # -------------------------------------------------------------
    if "peak_finding" in workloads:
        print("\n" + "-" * 90)
        print(" 3. PEAK FINDING & THRESHOLDING THROUGHPUT (MSamples / Second)")
        print("-" * 90)
        print(
            f" {'Batch (B)':<10} | {'Orig CPU':<14} | {'Orig SIMD':<14} | {'Torch CPU Vector':<16} | {'Torch CUDA Peak':<16} | {'Speedup':<10}"
        )
        print("-" * 90)

        for b in batch_sizes:
            cfg_base = {
                "workload": "peak_finding",
                "size": 131072,
                "batch_size": b,
                "iterations": 15 if b < 128 else 6,
                "warmup": 5 if b < 128 else 2,
            }
            total_samples = b * 131072

            r_orig = _run_subprocess_worker(
                {**cfg_base, "route": "original_standard", "threads": 1}, py_exe
            )
            mrate_orig = (total_samples / r_orig["median"]) / 1e6

            r_simd = _run_subprocess_worker(
                {**cfg_base, "route": "original_tuned", "threads": 1}, py_exe
            )
            mrate_simd = (total_samples / r_simd["median"]) / 1e6

            r_tcpu = _run_subprocess_worker(
                {**cfg_base, "route": "torch_cpu", "threads": 1}, py_exe
            )
            mrate_tcpu = (total_samples / r_tcpu["median"]) / 1e6

            if have_cuda:
                r_cuda = _run_subprocess_worker(
                    {**cfg_base, "route": "torch_cuda"}, py_exe
                )
                mrate_cuda = (total_samples / r_cuda["median"]) / 1e6
                cuda_str = f"{mrate_cuda:8.1f} MS/s"
                speedup = f"{mrate_cuda / mrate_orig:5.2f}x"
            else:
                r_cuda = {}
                cuda_str = "N/A"
                speedup = f"{mrate_tcpu / mrate_orig:5.2f}x"

            print(
                f" B = {b:<6} | {mrate_orig:8.1f} MS/s   | {mrate_simd:8.1f} MS/s   | {mrate_tcpu:8.1f} MS/s     | {cuda_str:<16} | {speedup:<10}"
            )

            results["peak_finding"][f"batch_{b}"] = {
                "original": r_orig,
                "original_simd": r_simd,
                "torch_cpu": r_tcpu,
                "torch_cuda": r_cuda,
            }

    # -------------------------------------------------------------
    # 4. Synthetic LiveBatchMatchedFilter microbenchmark
    # -------------------------------------------------------------
    if "synthetic_live_batch" in workloads:
        print("\n" + "-" * 90)
        print(
            " 4. SYNTHETIC LIVEBATCH MICROBENCHMARK "
            "(generated PSD, templates, and strain)"
        )
        print("-" * 90)
        print(
            f" {'Batch (B)':<10} | {'Orig CPU (1T)':<14} | {'Orig Max (16T)':<14} | {'Torch CPU (16T)':<15} | {'Torch CUDA':<14} | {'CUDA Speedup':<12}"
        )
        print("-" * 90)

        for b in batch_sizes:
            cfg_base = {
                "workload": "synthetic_live_batch",
                "size": 131072,
                "batch_size": b,
                "num_blocks": 10 if b < 128 else 5,
                "warmup": 2,
                "iterations": 10 if b < 128 else 5,
            }

            # Orig 1T
            r_orig_1t = _run_subprocess_worker(
                {**cfg_base, "route": "original_standard", "threads": 1}, py_exe
            )
            ms_orig_1t = r_orig_1t["median"] * 1000.0

            # Orig Max (16T)
            r_orig_16t = _run_subprocess_worker(
                {**cfg_base, "route": "original_tuned", "threads": 16}, py_exe
            )
            ms_orig_16t = r_orig_16t["median"] * 1000.0

            # Torch CPU (16T)
            r_tcpu = _run_subprocess_worker(
                {**cfg_base, "route": "torch_cpu", "threads": 16}, py_exe
            )
            ms_tcpu = r_tcpu["median"] * 1000.0

            # Torch CUDA
            if have_cuda:
                r_cuda = _run_subprocess_worker(
                    {**cfg_base, "route": "torch_cuda"}, py_exe
                )
                ms_cuda = r_cuda["median"] * 1000.0
                cuda_str = f"{ms_cuda:7.2f} ms"
                speedup = f"{ms_orig_1t / ms_cuda:5.2f}x"
            else:
                r_cuda = {}
                cuda_str = "N/A"
                speedup = f"{ms_orig_1t / ms_tcpu:5.2f}x (CPU)"

            print(
                f" B = {b:<6} | {ms_orig_1t:7.2f} ms     | {ms_orig_16t:7.2f} ms     | {ms_tcpu:7.2f} ms      | {cuda_str:<14} | {speedup:<12}"
            )

            results["synthetic_live_batch"][f"batch_{b}"] = {
                "original_1t": r_orig_1t,
                "original_16t": r_orig_16t,
                "torch_cpu_16t": r_tcpu,
                "torch_cuda": r_cuda,
            }

    # -------------------------------------------------------------
    # 5. Multi-Detector Network Response Benchmark
    # -------------------------------------------------------------
    if "detector_network" in workloads:
        print("\n" + "-" * 90)
        print(" 5. MULTI-DETECTOR NETWORK ANTENNA PATTERNS & DELAYS (H1, L1, V1)")
        print("-" * 90)
        print(
            f" {'Sky Points (N)':<16} | {'Sequential (1T)':<16} | {'Vectorized Network':<20} | {'Speedup':<12}"
        )
        print("-" * 90)

        for n_sky in [10, 100, 1000, 10000]:
            cfg_base = {
                "workload": "detector_network",
                "num_sky": n_sky,
                "ifos": ["H1", "L1", "V1"],
                "iterations": 5,
                "warmup": 2,
            }
            r_seq = _run_subprocess_worker(
                {**cfg_base, "route": "original_standard", "threads": 1}, py_exe
            )
            r_vec = _run_subprocess_worker(
                {**cfg_base, "route": "torch_cpu", "threads": 1}, py_exe
            )
            ms_seq = r_seq["median"] * 1000.0
            ms_vec = r_vec["median"] * 1000.0
            speedup = f"{ms_seq / ms_vec:6.1f}x"
            print(
                f" N = {n_sky:<12} | {ms_seq:9.2f} ms      | {ms_vec:9.2f} ms          | {speedup:<12}"
            )
            results["detector_network"][f"points_{n_sky}"] = {
                "sequential": r_seq,
                "vectorized": r_vec,
            }

    # -------------------------------------------------------------
    # 6. Batched Relative Binning Likelihood Benchmark
    # -------------------------------------------------------------
    if "relbin_likelihood" in workloads:
        print("\n" + "-" * 90)
        print(" 6. BATCHED RELATIVE BINNING SUMMARY EVALUATION (relbin_torch)")
        print("-" * 90)
        print(
            f" {'Samples (B)':<16} | {'Sequential (1T)':<16} | {'Batched Summary':<20} | {'Speedup':<12}"
        )
        print("-" * 90)

        for b_samples in [10, 100, 1000, 10000]:
            cfg_base = {
                "workload": "relbin_likelihood",
                "num_samples": b_samples,
                "iterations": 5,
                "warmup": 2,
            }
            r_seq = _run_subprocess_worker(
                {**cfg_base, "route": "original_standard", "threads": 1}, py_exe
            )
            r_bat = _run_subprocess_worker(
                {**cfg_base, "route": "torch_cpu", "threads": 1}, py_exe
            )
            ms_seq = r_seq["median"] * 1000.0
            ms_bat = r_bat["median"] * 1000.0
            speedup = f"{ms_seq / ms_bat:6.1f}x"
            print(
                f" B = {b_samples:<12} | {ms_seq:9.2f} ms      | {ms_bat:9.2f} ms          | {speedup:<12}"
            )
            results["relbin_likelihood"][f"samples_{b_samples}"] = {
                "sequential": r_seq,
                "batched": r_bat,
            }

    # -------------------------------------------------------------
    # 7. MatchedFilterControl.full_matched_filter_and_cluster_symm
    # -------------------------------------------------------------
    if (
        "matched_filter_symm" in workloads
        or "full_matched_filter_and_cluster_symm" in workloads
    ):
        print("\n" + "-" * 105)
        print(
            " 7. MATCHED_FILTER_AND_CLUSTER_SYMM "
            "(Single-template/segment production matched-filter & cluster)"
        )
        print("-" * 105)
        print(
            f" {'Size (N)':<10} | {'Duration':<10} | {'Std CPU (1T)':<14} | {'Std CPU (16T)':<14} | {'Torch CPU (16T)':<15} | {'Torch CUDA':<14} | {'CUDA Speedup':<12}"
        )
        print("-" * 105)

        mfc_sizes = [32768, 65536, 131072, 262144, 524288]
        for n in mfc_sizes:
            dur_sec = n / 2048.0
            cfg_base = {
                "workload": "matched_filter_symm",
                "size": n,
                "sample_rate": 2048,
                "snr_threshold": 5.5,
                "inj_snr": 10.0,
                "iterations": 15 if n <= 131072 else 8,
                "warmup": 3 if n <= 131072 else 2,
            }

            r_std_1t = _run_subprocess_worker(
                {**cfg_base, "route": "original_standard", "threads": 1}, py_exe
            )
            ms_std_1t = r_std_1t["median"] * 1000.0

            r_std_16t = _run_subprocess_worker(
                {**cfg_base, "route": "original_tuned", "threads": 16}, py_exe
            )
            ms_std_16t = r_std_16t["median"] * 1000.0

            r_tcpu = _run_subprocess_worker(
                {**cfg_base, "route": "torch_cpu", "threads": 16}, py_exe
            )
            ms_tcpu = r_tcpu["median"] * 1000.0

            if have_cuda:
                r_cuda = _run_subprocess_worker(
                    {**cfg_base, "route": "torch_cuda"}, py_exe
                )
                ms_cuda = r_cuda["median"] * 1000.0
                cuda_str = f"{ms_cuda:7.3f} ms"
                speedup = f"{ms_std_1t / ms_cuda:5.2f}x"
            else:
                r_cuda = {}
                cuda_str = "N/A"
                speedup = f"{ms_std_1t / ms_tcpu:5.2f}x (CPU)"

            print(
                f" N={n:<7} | {dur_sec:6.1f}s   | {ms_std_1t:7.3f} ms     | {ms_std_16t:7.3f} ms     | {ms_tcpu:7.3f} ms      | {cuda_str:<14} | {speedup:<12}"
            )

            results["matched_filter_symm"][f"size_{n}"] = {
                "size": n,
                "duration_sec": dur_sec,
                "standard_1t": r_std_1t,
                "standard_16t": r_std_16t,
                "torch_cpu_16t": r_tcpu,
                "torch_cuda": r_cuda,
            }

    print("\n" + "=" * 90)
    print(" BENCHMARK COMPLETE")
    print("=" * 90)

    artifact["campaigns"][-1]["complete"] = True
    if output_path:
        atomic_write_json(output_path, seal_artifact(artifact))
        print(f"\nStructured benchmark results saved to: {output_path}")

    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Comprehensive PyCBC Torch Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run quick benchmark suite with reduced workloads, batch sizes, "
            "and approximants"
        ),
    )
    parser.add_argument(
        "--workloads",
        "-w",
        nargs="+",
        default=None,
        choices=[
            "all",
            "waveform",
            "correlate_ifft",
            "peak_finding",
            "synthetic_live_batch",
            "matched_filter_symm",
            "full_matched_filter_and_cluster_symm",
            "detector_network",
            "relbin_likelihood",
            "live_pipeline",
            "pipeline",
        ],
        help=(
            "Workloads to execute; live_pipeline/pipeline and "
            "full_matched_filter_and_cluster_symm are aliases"
        ),
    )
    parser.add_argument(
        "--family",
        "-f",
        choices=list(FAMILIES.keys()),
        default=None,
        help="Filter waveform approximants by family",
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        default=None,
        help="Specific waveform approximants to benchmark",
    )
    parser.add_argument(
        "--batches",
        "-b",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Batch sizes for correlate, peak finding, and synthetic live-batch "
            "microbenchmarks"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("artifacts/comprehensive_benchmark_results.json"),
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
        help="Overwrite existing JSON file instead of merging results for selected components",
    )
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        args.workloads = (
            args.workloads
            if args.workloads is not None
            else ["correlate_ifft", "peak_finding", "synthetic_live_batch"]
        )
        args.batches = (
            args.batches
            if args.batches is not None
            else [1, 8, 32]
        )
        if args.models is not None:
            args.selected_models = args.models
        elif args.family is not None:
            args.selected_models = FAMILIES[args.family]
        else:
            args.selected_models = ["TaylorF2", "IMRPhenomD", "IMRPhenomXAS"]
    else:
        args.workloads = (
            args.workloads
            if args.workloads is not None
            else ["all"]
        )
        args.batches = (
            args.batches
            if args.batches is not None
            else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        )
        if args.models is not None:
            args.selected_models = args.models
        elif args.family is not None:
            args.selected_models = FAMILIES[args.family]
        else:
            args.selected_models = FAMILIES["all"]
    return args


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    parsed = parser.parse_args(args)
    return _resolve_args(parsed)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    run_full_suite(
        py_exe=args.python,
        output_path=args.output,
        workloads=args.workloads,
        approximants=args.selected_models,
        batch_sizes=args.batches,
        no_merge=args.no_merge,
    )


if __name__ == "__main__":
    main()
