#!/usr/bin/env python3
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

"""Comprehensive benchmark suite for torchwave integration in PyCBC.

Benchmarks:
1. LiveFilterBank cold-start template initialization across bank sizes.
2. FilterBank get_batch_tensor batch synthesis across batch sizes.
3. End-to-end batched matched filtering with TiledMatchedFilterControl.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
import h5py
import numpy as np
import torch

try:
    import torchwave  # noqa: F401
    HAS_TORCHWAVE = True
except ImportError:
    HAS_TORCHWAVE = False

from pycbc import DYN_RANGE_FAC
from pycbc.filter.gpu_search.adapter import TiledMatchedFilterControl
from pycbc.filter.matchedfilter import match, sigmasq
from pycbc.types import FrequencySeries, zeros
from pycbc.waveform.bank import FilterBank, LiveFilterBank


def create_mock_bank_file(n_templates: int, approximant: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".hdf")
    os.close(fd)
    np.random.seed(42)

    if approximant == "TaylorF2":
        m1 = np.random.uniform(1.2, 2.5, n_templates).astype(np.float32)
        m2 = np.random.uniform(1.1, m1, n_templates).astype(np.float32)
        s1 = np.random.uniform(-0.05, 0.05, n_templates).astype(np.float32)
        s2 = np.random.uniform(-0.05, 0.05, n_templates).astype(np.float32)
        flow = np.full(n_templates, 30.0, dtype=np.float32)
    else:  # IMRPhenomD
        m1 = np.random.uniform(15.0, 50.0, n_templates).astype(np.float32)
        m2 = np.random.uniform(10.0, m1, n_templates).astype(np.float32)
        s1 = np.random.uniform(-0.5, 0.5, n_templates).astype(np.float32)
        s2 = np.random.uniform(-0.5, 0.5, n_templates).astype(np.float32)
        flow = np.full(n_templates, 20.0, dtype=np.float32)

    with h5py.File(path, "w") as f:
        f["mass1"] = m1
        f["mass2"] = m2
        f["spin1z"] = s1
        f["spin2z"] = s2
        f["f_lower"] = flow
        f.attrs["parameters"] = [
            "mass1",
            "mass2",
            "spin1z",
            "spin2z",
            "f_lower",
        ]

    return path


def bench_live_filter_bank(approximant: str, n_sizes=(128, 256, 512, 1024), reps=3):
    results = []
    print(f"\n--- [1] LiveFilterBank Cold-Start Benchmark ({approximant}) ---")
    for n in n_sizes:
        path = create_mock_bank_file(n, approximant)
        try:
            # Baseline
            lfb_base = LiveFilterBank(
                path,
                sample_rate=2048,
                minimum_buffer=4,
                approximant=approximant,
                enable_torchwave=False,
            )
            times_base = []
            for _ in range(reps):
                t0 = time.perf_counter()
                _ = list(lfb_base)
                times_base.append(time.perf_counter() - t0)
            t_base = float(np.median(times_base))

            # Torchwave
            lfb_tw = LiveFilterBank(
                path,
                sample_rate=2048,
                minimum_buffer=4,
                approximant=approximant,
                enable_torchwave=True,
            )
            times_tw = []
            for _ in range(reps):
                t0 = time.perf_counter()
                _ = list(lfb_tw)
                times_tw.append(time.perf_counter() - t0)
            t_tw = float(np.median(times_tw))

            speedup = t_base / t_tw
            results.append({
                "n_templates": n,
                "approximant": approximant,
                "baseline_ms": t_base * 1000,
                "torchwave_ms": t_tw * 1000,
                "baseline_rate": n / t_base,
                "torchwave_rate": n / t_tw,
                "speedup": speedup,
            })
            print(
                f"  N={n:4d}: Baseline={t_base*1000:6.1f} ms ({n/t_base:6.1f} tmpl/s) | "
                f"Torchwave={t_tw*1000:6.1f} ms ({n/t_tw:6.1f} tmpl/s) | "
                f"Speedup: {speedup:4.2f}x"
            )
        finally:
            if os.path.exists(path):
                os.remove(path)
    return results


def bench_filter_bank_batch_tensor(approximant: str, batches=(16, 32, 64, 128, 256), reps=5):
    results = []
    print(f"\n--- [2] FilterBank Batch Synthesis Benchmark ({approximant}) ---")
    max_b = max(batches)
    path = create_mock_bank_file(max_b, approximant)
    try:
        flen = 2049
        delta_f = 0.5
        fb_base = FilterBank(
            path, flen, delta_f, dtype=np.complex64, approximant=approximant, enable_torchwave=False
        )
        fb_tw = FilterBank(
            path, flen, delta_f, dtype=np.complex64, approximant=approximant, enable_torchwave=True
        )

        for b in batches:
            tnums = list(range(b))

            # Warmup
            _ = [fb_base[i] for i in tnums[:min(4, b)]]
            _ = fb_tw.get_batch_tensor(tnums[:min(4, b)], device="cpu")

            # Baseline (loop + stack)
            times_base = []
            for _ in range(reps):
                t0 = time.perf_counter()
                tmpls = [fb_base[i] for i in tnums]
                _ = np.stack([np.asarray(t)[:flen] for t in tmpls], axis=0)
                times_base.append(time.perf_counter() - t0)
            t_base = float(np.median(times_base))

            # Torchwave direct 2D batch tensor
            times_tw = []
            for _ in range(reps):
                t0 = time.perf_counter()
                _, _ = fb_tw.get_batch_tensor(tnums, device="cpu")
                times_tw.append(time.perf_counter() - t0)
            t_tw = float(np.median(times_tw))

            speedup = t_base / t_tw
            results.append({
                "batch_size": b,
                "approximant": approximant,
                "baseline_ms": t_base * 1000,
                "torchwave_ms": t_tw * 1000,
                "baseline_rate": b / t_base,
                "torchwave_rate": b / t_tw,
                "speedup": speedup,
            })
            print(
                f"  B={b:3d}: Baseline={t_base*1000:6.2f} ms ({b/t_base:7.1f} tmpl/s) | "
                f"Torchwave={t_tw*1000:6.2f} ms ({b/t_tw:7.1f} tmpl/s) | "
                f"Speedup: {speedup:4.2f}x"
            )
    finally:
        if os.path.exists(path):
            os.remove(path)
    return results


def bench_matched_filter_e2e(approximant: str, batches=(16, 32, 64, 128), reps=5):
    results = []
    print(f"\n--- [3] Tiled Matched Filter End-to-End Ingestion ({approximant}) ---")
    max_b = max(batches)
    path = create_mock_bank_file(max_b, approximant)
    try:
        flen = 2049
        tlen = (flen - 1) * 2
        delta_f = 0.5
        dt = 1.0 / (delta_f * tlen)
        flow = 30.0 if approximant == "TaylorF2" else 20.0

        # Synthetic strain segment and flat PSD
        np.random.seed(1234)
        white_noise = np.random.normal(0, 1.0, flen) + 1j * np.random.normal(0, 1.0, flen)
        stilde = FrequencySeries(white_noise.astype(np.complex64), delta_f=delta_f)
        psd = FrequencySeries(np.ones(flen, dtype=np.float32), delta_f=delta_f)
        stilde.psd = psd

        class MockSegment:
            def __init__(self, data, psd):
                self.data = data
                self.psd = psd
                self.analyze = slice(100, tlen - 100)

            def __getitem__(self, item):
                return self.data[item]

            def __len__(self):
                return len(self.data)

        seg = MockSegment(stilde, psd)

        fb_base = FilterBank(path, flen, delta_f, dtype=np.complex64, approximant=approximant)
        fb_tw = FilterBank(path, flen, delta_f, dtype=np.complex64, approximant=approximant, enable_torchwave=True)

        for b in batches:
            tnums = list(range(b))
            tiled = TiledMatchedFilterControl(
                low_frequency_cutoff=flow,
                high_frequency_cutoff=min(flow + 500.0, (flen - 1) * delta_f),
                snr_threshold=4.0,
                tlen=tlen,
                delta_f=delta_f,
                dtype=np.complex64,
                segment_list=[seg],
                template_output=zeros(tlen, dtype=np.complex64),
                use_cluster=True,
                tile_size=b,
                device="cpu",
            )

            # Baseline: templates list -> prepare_template_batch -> filter
            tmpls_base = [fb_base[i] for i in tnums]
            sigmasqs_base = [float(sigmasq(t, psd=psd, low_frequency_cutoff=flow)) for t in tmpls_base]

            # Torchwave: 2D tensor -> filter
            tensor_tw, tmpls_tw = fb_tw.get_batch_tensor(tnums, device="cpu")
            sigmasqs_tw = [float(sigmasq(t, psd=psd, low_frequency_cutoff=flow)) for t in tmpls_tw]

            # Warmup
            _ = tiled.batched_matched_filter_and_cluster(0, tmpls_base, sigmasqs_base, window=100)
            _ = tiled.batched_matched_filter_and_cluster(0, tensor_tw, sigmasqs_tw, window=100)

            # Measure baseline filtering wall time
            times_base = []
            for _ in range(reps):
                t0 = time.perf_counter()
                _ = tiled.batched_matched_filter_and_cluster(0, tmpls_base, sigmasqs_base, window=100)
                times_base.append(time.perf_counter() - t0)
            t_base = float(np.median(times_base))

            # Measure torchwave filtering wall time
            times_tw = []
            for _ in range(reps):
                t0 = time.perf_counter()
                _ = tiled.batched_matched_filter_and_cluster(0, tensor_tw, sigmasqs_tw, window=100)
                times_tw.append(time.perf_counter() - t0)
            t_tw = float(np.median(times_tw))

            speedup = t_base / t_tw
            results.append({
                "batch_size": b,
                "approximant": approximant,
                "filtering_baseline_ms": t_base * 1000,
                "filtering_torchwave_ms": t_tw * 1000,
                "speedup": speedup,
            })
            print(
                f"  B={b:3d}: Filter (PyCBC list)={t_base*1000:6.2f} ms | "
                f"Filter (Torchwave 2D)={t_tw*1000:6.2f} ms | "
                f"Speedup: {speedup:4.2f}x"
            )
    finally:
        if os.path.exists(path):
            os.remove(path)
    return results


def main():
    print("==============================================================================")
    print("TORCHWAVE <-> PYCBC BENCHMARK CAMPAIGN")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__} | Device: CPU")
    print("==============================================================================")

    all_results = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
        "live_benchmarks": {},
        "batch_benchmarks": {},
        "matched_filter_benchmarks": {},
    }

    for approx in ["TaylorF2", "IMRPhenomD"]:
        all_results["live_benchmarks"][approx] = bench_live_filter_bank(approx)
        all_results["batch_benchmarks"][approx] = bench_filter_bank_batch_tensor(approx)
        all_results["matched_filter_benchmarks"][approx] = bench_matched_filter_e2e(approx)

    out_file = Path("artifacts/torchwave_pipeline_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_results, indent=2))
    print("\n==============================================================================")
    print(f"Benchmark artifact written to: {out_file.resolve()}")
    print("==============================================================================")


if __name__ == "__main__":
    main()
