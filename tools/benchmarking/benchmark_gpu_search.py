#!/usr/bin/env python
# Copyright (C) 2026
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

"""
End-to-end performance benchmarking and qualification script for the
PyCBC persistent PyTorch GPU search engine.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pycbc.types import FrequencySeries
from pycbc.filter.gpu_search import (
    prepare_bank,
    bind_psd,
    SelectionPolicy,
    SearchEngine,
)


def generate_synthetic_bank(num_templates: int, flen: int, delta_f: float, seed: int = 42) -> List[FrequencySeries]:
    rng = np.random.default_rng(seed)
    templates = []
    for i in range(num_templates):
        h_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(np.complex64)
        h_vals[0] = 0.0
        # Normalize template power
        pwr = np.sum(np.abs(h_vals) ** 2)
        if pwr > 0:
            h_vals /= np.sqrt(pwr)
        t = FrequencySeries(h_vals, delta_f=delta_f)
        t.id = i
        templates.append(t)
    return templates


def generate_synthetic_data(flen: int, delta_f: float, seed: int = 123) -> Tuple[FrequencySeries, FrequencySeries]:
    rng = np.random.default_rng(seed)
    data_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(np.complex64)
    data_vals[0] = 0.0
    stilde = FrequencySeries(data_vals, delta_f=delta_f)
    psd_vals = np.ones(flen, dtype=np.float32) * 2.0
    psd = FrequencySeries(psd_vals, delta_f=delta_f)
    return stilde, psd


def benchmark_tile_scaling(
    device: str,
    num_templates: int = 512,
    size: int = 1024,
    tile_sizes: List[int] = [1, 16, 64, 128, 256],
    warmup: int = 5,
    iterations: int = 20,
) -> Dict[str, Any]:
    flen = size // 2 + 1
    delta_f = 1.0 / size
    templates = generate_synthetic_bank(num_templates, flen, delta_f)
    stilde, psd = generate_synthetic_data(flen, delta_f)
    valid_interval = (size // 8, size - size // 8)
    policy = SelectionPolicy(snr_threshold=5.5, cluster_policy="live_peak")

    results = {}
    for ts in tile_sizes:
        if ts > num_templates:
            continue
        bank_plan = prepare_bank(templates, tile_size=ts, device=device)
        psd_plan = bind_psd(bank_plan, psd, device=device)
        engine = SearchEngine(bank_plan, policy, device=device)

        # Warmup
        for _ in range(warmup):
            engine.submit(stilde, psd_plan, valid_interval)
            engine.drain()

        if str(device).startswith("cuda") and torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            engine.submit(stilde, psd_plan, valid_interval)
            engine.drain()
            if str(device).startswith("cuda") and torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        engine.close()
        avg_time = float(np.mean(times))
        std_time = float(np.std(times))
        tmplt_per_sec = float(num_templates / avg_time)

        results[f"tile_size_{ts}"] = {
            "tile_size": ts,
            "avg_time_sec": avg_time,
            "std_time_sec": std_time,
            "templates_per_sec": tmplt_per_sec,
        }
    return results


def benchmark_cuda_graphs(
    num_templates: int = 512,
    size: int = 1024,
    tile_size: int = 64,
    warmup: int = 10,
    iterations: int = 50,
) -> Dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    flen = size // 2 + 1
    delta_f = 1.0 / size
    templates = generate_synthetic_bank(num_templates, flen, delta_f)
    stilde, psd = generate_synthetic_data(flen, delta_f)
    valid_interval = (size // 8, size - size // 8)
    policy = SelectionPolicy(snr_threshold=5.5, cluster_policy="live_peak")

    bank_plan = prepare_bank(templates, tile_size=tile_size, device="cuda")
    psd_plan = bind_psd(bank_plan, psd, device="cuda")

    # 1. Eager engine
    eager_engine = SearchEngine(bank_plan, policy, device="cuda", use_cuda_graphs=False)
    for _ in range(warmup):
        eager_engine.submit(stilde, psd_plan, valid_interval)
        eager_engine.drain()
    torch.cuda.synchronize()

    eager_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        eager_engine.submit(stilde, psd_plan, valid_interval)
        eager_engine.drain()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        eager_times.append((t1 - t0) * 1000.0)  # ms
    eager_engine.close()

    # 2. CUDA Graph engine
    graph_engine = SearchEngine(bank_plan, policy, device="cuda", use_cuda_graphs=True)
    for _ in range(warmup):
        graph_engine.submit(stilde, psd_plan, valid_interval)
        graph_engine.drain()
    torch.cuda.synchronize()

    graph_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        graph_engine.submit(stilde, psd_plan, valid_interval)
        graph_engine.drain()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        graph_times.append((t1 - t0) * 1000.0)  # ms

    stats = graph_engine.graph_stats
    graph_engine.close()

    eager_p50 = float(np.percentile(eager_times, 50))
    eager_p95 = float(np.percentile(eager_times, 95))
    eager_p99 = float(np.percentile(eager_times, 99))
    eager_mean = float(np.mean(eager_times))

    graph_p50 = float(np.percentile(graph_times, 50))
    graph_p95 = float(np.percentile(graph_times, 95))
    graph_p99 = float(np.percentile(graph_times, 99))
    graph_mean = float(np.mean(graph_times))

    speedup = eager_mean / graph_mean if graph_mean > 0 else 1.0

    return {
        "num_templates": num_templates,
        "tile_size": tile_size,
        "iterations": iterations,
        "eager": {
            "mean_ms": eager_mean,
            "p50_ms": eager_p50,
            "p95_ms": eager_p95,
            "p99_ms": eager_p99,
        },
        "cuda_graph": {
            "mean_ms": graph_mean,
            "p50_ms": graph_p50,
            "p95_ms": graph_p95,
            "p99_ms": graph_p99,
            "capture_count": stats["capture_count"],
            "replay_count": stats["replay_count"],
        },
        "speedup": speedup,
    }


def benchmark_live_streaming_latency(
    num_templates: int = 512,
    size: int = 1024,
    tile_size: int = 64,
    num_blocks: int = 50,
    double_buffering: bool = True,
) -> Dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    flen = size // 2 + 1
    delta_f = 1.0 / size
    templates = generate_synthetic_bank(num_templates, flen, delta_f)
    stilde, psd = generate_synthetic_data(flen, delta_f)
    valid_interval = (size // 8, size - size // 8)
    policy = SelectionPolicy(snr_threshold=5.5, cluster_policy="live_peak")

    bank_plan = prepare_bank(templates, tile_size=tile_size, device="cuda")
    psd_plan = bind_psd(bank_plan, psd, device="cuda")

    engine = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device="cuda",
        use_cuda_graphs=True,
        num_workspaces=2 if double_buffering else 1,
        enable_async_transfers=double_buffering,
    )

    # Warmup
    for _ in range(5):
        engine.submit(stilde, psd_plan, valid_interval)
        engine.drain()
    torch.cuda.synchronize()

    initial_vram = torch.cuda.memory_allocated()

    latencies = []
    for i in range(num_blocks):
        t0 = time.perf_counter()
        engine.submit(stilde, psd_plan, valid_interval, block_id=i)
        ready = engine.drain()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    final_vram = torch.cuda.memory_allocated()
    peak_vram = torch.cuda.max_memory_allocated()

    engine.close()

    vram_growth = final_vram - initial_vram

    return {
        "num_blocks": num_blocks,
        "double_buffering": double_buffering,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_max_ms": float(np.max(latencies)),
        "latency_mean_ms": float(np.mean(latencies)),
        "vram_initial_mb": float(initial_vram / (1024 * 1024)),
        "vram_final_mb": float(final_vram / (1024 * 1024)),
        "vram_peak_mb": float(peak_vram / (1024 * 1024)),
        "vram_leak_detected": bool(vram_growth > 1024 * 1024),  # > 1 MB growth
    }


def main():
    parser = argparse.ArgumentParser(description="PyCBC GPU Search Engine Qualification Benchmark")
    parser.add_argument("--output", type=str, default="artifacts/gpu_search_qualification_receipt.json", help="Output JSON path")
    args = parser.parse_args()

    device_name = "CPU"
    gpu_info = {}
    if torch is not None and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_info = {
            "name": device_name,
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "cuda_version": torch.version.cuda,
        }

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": getattr(torch, "__version__", None),
        "gpu": gpu_info,
        "benchmarks": {},
    }

    print(f"=== Running PyCBC GPU Search Engine Qualification on {device_name} ===")

    # 1. Tile scaling benchmark
    if torch is not None and torch.cuda.is_available():
        print("--> Running GPU tile scaling benchmark...")
        report["benchmarks"]["gpu_tile_scaling"] = benchmark_tile_scaling(
            device="cuda", num_templates=512, tile_sizes=[1, 16, 64, 128, 256]
        )
    print("--> Running CPU tile scaling benchmark baseline...")
    report["benchmarks"]["cpu_tile_scaling"] = benchmark_tile_scaling(
        device="cpu", num_templates=64, tile_sizes=[1, 16, 64]
    )

    # 2. CUDA Graph comparison
    if torch is not None and torch.cuda.is_available():
        print("--> Running CUDA Graph vs Eager benchmark...")
        report["benchmarks"]["cuda_graph_comparison"] = benchmark_cuda_graphs(
            num_templates=512, tile_size=64, iterations=50
        )

        # 3. Live streaming latency & VRAM boundedness
        print("--> Running Live streaming latency and VRAM boundedness benchmark...")
        report["benchmarks"]["live_streaming_latency"] = benchmark_live_streaming_latency(
            num_templates=512, tile_size=64, num_blocks=50, double_buffering=True
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Benchmark Complete. Receipt written to {out_path} ===")
    print(json.dumps(report["benchmarks"], indent=2))


if __name__ == "__main__":
    main()
