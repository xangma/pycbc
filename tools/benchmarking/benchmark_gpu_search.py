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
import concurrent.futures
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    prepare_power_chisq_plan,
    VetoManager,
)
from pycbc.filter.gpu_search.candidates import _select_numpy_symmetric
from pycbc.filter.matchedfilter import matched_filter_core, sigmasq
from pycbc.vetoes.chisq import (
    power_chisq_bins,
    power_chisq_at_points_from_precomputed,
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


def generate_taylorf2_bank(
    num_templates: int,
    flen: int,
    delta_f: float,
    f_lower: float = 20.0,
    f_upper: float = 1000.0,
) -> List[FrequencySeries]:
    """Generate physical TaylorF2 templates across astrophysical binary parameter space."""
    templates = []
    m1_vals = np.linspace(10.0, 50.0, num_templates, dtype=np.float64)
    m2_vals = np.linspace(1.4, 25.0, num_templates, dtype=np.float64)

    has_lalsim = False
    try:
        from pycbc.waveform import get_fd_waveform

        _hp, _ = get_fd_waveform(
            approximant="TaylorF2",
            mass1=20.0,
            mass2=10.0,
            delta_f=delta_f,
            f_lower=f_lower,
        )
        has_lalsim = True
    except Exception:
        has_lalsim = False

    for i in range(num_templates):
        m1 = float(m1_vals[i])
        m2 = float(m2_vals[i])
        if has_lalsim:
            hp, _ = get_fd_waveform(
                approximant="TaylorF2",
                mass1=m1,
                mass2=m2,
                delta_f=delta_f,
                f_lower=f_lower,
                f_final=f_upper,
            )
            hp_data = np.zeros(flen, dtype=np.complex64)
            copy_len = min(len(hp), flen)
            hp_data[:copy_len] = hp.numpy()[:copy_len]
        else:
            M = m1 + m2
            eta = (m1 * m2) / (M**2)
            M_sec = M * 4.925491025543576e-6
            freqs = np.linspace(0, (flen - 1) * delta_f, flen, dtype=np.float64)
            kmin = max(1, int(f_lower / delta_f))
            kmax = min(flen - 1, int(f_upper / delta_f))
            v = (np.pi * M_sec * np.maximum(freqs[kmin:kmax], 1e-4)) ** (1.0 / 3.0)
            psi = (3.0 / (128.0 * eta * (v**5))) * (
                1.0
                + (3715.0 / 756.0 + 55.0 / 9.0 * eta) * (v**2)
                - 16.0 * np.pi * (v**3)
                + (
                    15293365.0 / 508032.0
                    + 27145.0 / 504.0 * eta
                    + 3085.0 / 72.0 * (eta**2)
                )
                * (v**4)
            )
            amp = (freqs[kmin:kmax] ** (-7.0 / 6.0)).astype(np.float32)
            phase = psi - np.pi / 4.0
            hp_data = np.zeros(flen, dtype=np.complex64)
            hp_data[kmin:kmax] = amp * (np.cos(phase) - 1j * np.sin(phase))

        pwr = np.sum(np.abs(hp_data) ** 2)
        if pwr > 0:
            hp_data /= np.sqrt(pwr)

        t = FrequencySeries(hp_data, delta_f=delta_f)
        t.id = i
        t.params = type(
            "Params", (), {"template_hash": i, "mass1": m1, "mass2": m2}
        )()
        templates.append(t)

    return templates


def generate_aligo_psd(
    flen: int, delta_f: float, f_lower: float = 20.0
) -> FrequencySeries:
    """Generate colored aLIGOZeroDetHighPower PSD with consistent dynamic-range scaling."""
    from pycbc import DYN_RANGE_FAC

    try:
        from pycbc.psd import aLIGOZeroDetHighPower

        psd = aLIGOZeroDetHighPower(flen, delta_f, low_freq_cutoff=f_lower)
    except Exception:
        freqs = np.linspace(0, (flen - 1) * delta_f, flen, dtype=np.float64)
        x = np.maximum(freqs / 215.0, 1e-4)
        s0 = 1e-49
        psd_vals = s0 * (
            x ** (-4.14)
            - 5.0 * (x ** (-2))
            + 111.0 * (1.0 - (x**2) + 0.5 * (x**4)) / (1.0 + 0.5 * (x**2))
        )
        psd_vals = np.maximum(psd_vals, 1e-50)
        kmin = int(f_lower / delta_f)
        psd_vals[:kmin] = 0.0
        psd = FrequencySeries(psd_vals, delta_f=delta_f)

    # Scale PSD by DYN_RANGE_FAC**2 to match PyCBC single-precision convention
    psd_vals = (psd.numpy() if hasattr(psd, "numpy") else np.asarray(psd)).astype(np.float64)
    psd_vals = (psd_vals * (float(DYN_RANGE_FAC) ** 2)).astype(np.float32)
    psd = FrequencySeries(psd_vals, delta_f=delta_f)
    psd.dyn_range_factor = float(DYN_RANGE_FAC)
    return psd


def generate_colored_strain(
    psd: FrequencySeries, seed: int = 123
) -> FrequencySeries:
    """Generate frequency-domain strain data colored by a given PSD."""
    flen = len(psd)
    delta_f = psd.delta_f
    rng = np.random.default_rng(seed)
    psd_np = psd.numpy() if hasattr(psd, "numpy") else np.asarray(psd)
    white_noise = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(
        np.complex64
    )
    sigma = np.sqrt(np.maximum(psd_np, 0.0) / (4.0 * delta_f)).astype(
        np.float32
    )
    stilde_vals = white_noise * sigma
    stilde_vals[0] = 0.0
    stilde = FrequencySeries(stilde_vals, delta_f=delta_f)
    stilde.psd = psd
    if hasattr(psd, "dyn_range_factor"):
        stilde.dyn_range_factor = psd.dyn_range_factor
    return stilde


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
        t_setup_0 = time.perf_counter()
        bank_plan = prepare_bank(templates, tile_size=ts, device=device)
        psd_plan = bind_psd(bank_plan, psd, device=device)
        engine = SearchEngine(bank_plan, policy, device=device)
        if str(device).startswith("cuda") and torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_setup_1 = time.perf_counter()
        setup_time = float(t_setup_1 - t_setup_0)
        # Measure true cold-start first iteration before warmup
        t_cold_0 = time.perf_counter()
        engine.submit(stilde, psd_plan, valid_interval)
        engine.drain()
        if str(device).startswith("cuda") and torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_cold_1 = time.perf_counter()
        first_iter_calc_time = float(t_cold_1 - t_cold_0)

        # Warmup
        for _ in range(max(0, warmup - 1)):
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
        avg_calc_time = float(np.mean(times))
        std_calc_time = float(np.std(times))
        first_iter_wall_time = setup_time + first_iter_calc_time
        amortized_e2e_time = setup_time + avg_calc_time
        total_time_single = amortized_e2e_time
        calc_tmplt_per_sec = float(num_templates / avg_calc_time) if avg_calc_time > 0 else 0.0
        e2e_tmplt_per_sec = float(num_templates / amortized_e2e_time) if amortized_e2e_time > 0 else 0.0
        first_iter_tmplt_per_sec = float(num_templates / first_iter_wall_time) if first_iter_wall_time > 0 else 0.0

        results[f"tile_size_{ts}"] = {
            "tile_size": ts,
            "setup_time_sec": setup_time,
            "calc_time_sec": avg_calc_time,
            "calc_time_std_sec": std_calc_time,
            "first_iter_wall_time_sec": float(first_iter_wall_time),
            "amortized_e2e_time_sec": float(amortized_e2e_time),
            "total_wall_time_sec": float(total_time_single),
            "calc_templates_per_sec": calc_tmplt_per_sec,
            "e2e_templates_per_sec": e2e_tmplt_per_sec,
            "first_iter_templates_per_sec": first_iter_tmplt_per_sec,
            "avg_time_sec": avg_calc_time,
            "templates_per_sec": calc_tmplt_per_sec,
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
        _ = engine.drain()
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


def _validate_integer_field(
    value: Any,
    field_name: str,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    context: str = "",
) -> int:
    """
    Validate that an integer field (e.g., sample_idx, chisq_dof, template_id)
    is a finite integer and not a fractional value (e.g., 100.9 or 14.9),
    and conforms to optional range bounds.
    """
    if value is None:
        raise ValueError(f"Field '{field_name}' cannot be None{context}")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"Boolean value not allowed for {field_name}: {value}{context}"
        )
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"Non-finite {field_name} detected: {value}{context}")
        if not float(value).is_integer():
            raise ValueError(f"Fractional {field_name} detected: {value}{context}")
    elif isinstance(value, (int, np.integer)):
        pass
    else:
        try:
            f_val = float(value)
            if not np.isfinite(f_val):
                raise ValueError(f"Non-finite {field_name} detected: {value}{context}")
            if not f_val.is_integer():
                raise ValueError(f"Fractional {field_name} detected: {value}{context}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid {field_name} detected: {value}{context}") from e

    int_val = int(value)
    if min_val is not None and int_val < min_val:
        if field_name == "chisq_dof":
            raise ValueError(
                f"Non-positive GPU chisq_dof detected: dof={int_val}{context}"
            )
        raise ValueError(
            f"{field_name} {int_val} is less than minimum {min_val}{context}"
        )
    if max_val is not None and int_val > max_val:
        if field_name == "template_id":
            raise ValueError(
                f"Unexpected template_id {int_val} outside valid range [{min_val}, {max_val}]{context}"
            )
        raise ValueError(
            f"{field_name} {int_val} is greater than maximum {max_val}{context}"
        )
    return int_val


def extract_and_validate_gpu_candidates(
    seg_batches: Any, num_templates: int, segment_idx: int = 0
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Extract, validate, and index GPU candidates for a segment by template ID.

    Strictly validates that required candidate fields (template_id, sample_idx,
    snr, chisq, chisq_dof) are present, have matching array lengths, contain
    valid identities (0 <= tid < num_templates), and have finite integer/float values
    (strictly rejecting python and numpy booleans).
    """
    gpu_cands_by_tmpl: Dict[int, List[Dict[str, Any]]] = {
        t: [] for t in range(num_templates)
    }

    if not isinstance(seg_batches, (list, tuple)):
        batches = [seg_batches]
    else:
        batches = seg_batches

    for batch in batches:
        if hasattr(batch, "overflow") and batch.overflow:
            raise RuntimeError(f"GPU batch overflow in segment {segment_idx}")
        if hasattr(batch, "aborted") and batch.aborted:
            raise RuntimeError(f"GPU batch aborted in segment {segment_idx}")

        if hasattr(batch, "results"):
            results = batch.results
        elif isinstance(batch, dict) and "results" in batch:
            results = batch["results"]
        elif isinstance(batch, dict):
            results = [batch]
        elif isinstance(batch, list):
            results = batch
        else:
            results = []

        for r in results:
            if not isinstance(r, dict):
                continue
            required_fields = ["template_id", "sample_idx", "snr", "chisq", "chisq_dof"]
            for field in required_fields:
                if field not in r or r[field] is None:
                    raise ValueError(
                        f"Missing required field '{field}' in GPU candidate results (segment={segment_idx})"
                    )

            t_ids = r["template_id"]
            s_idxs = r["sample_idx"]
            snrs = r["snr"]
            chisqs = r["chisq"]
            dofs = r["chisq_dof"]
            n_cands = len(t_ids)

            for field_name, arr in [
                ("sample_idx", s_idxs),
                ("snr", snrs),
                ("chisq", chisqs),
                ("chisq_dof", dofs),
            ]:
                if len(arr) != n_cands:
                    raise ValueError(
                        f"Inconsistent candidate field length for '{field_name}' in segment {segment_idx}: "
                        f"expected {n_cands} from template_id, got {len(arr)}"
                    )

            for int_field_name, int_arr in [
                ("template_id", t_ids),
                ("sample_idx", s_idxs),
                ("chisq_dof", dofs),
            ]:
                if hasattr(int_arr, "dtype") and (
                    int_arr.dtype == bool
                    or np.issubdtype(int_arr.dtype, np.bool_)
                ):
                    raise ValueError(
                        f"Boolean array not allowed for '{int_field_name}' in segment {segment_idx}"
                    )

            if n_cands == 0:
                continue

            for i in range(n_cands):
                tid = _validate_integer_field(
                    t_ids[i],
                    "template_id",
                    min_val=0,
                    max_val=num_templates - 1,
                    context=f" (segment={segment_idx}, candidate={i})",
                )
                s_idx_val = _validate_integer_field(
                    s_idxs[i],
                    "sample_idx",
                    min_val=0,
                    context=f" (segment={segment_idx}, template={tid}, candidate={i})",
                )

                snr_val = complex(snrs[i])
                if not np.isfinite(snr_val.real) or not np.isfinite(snr_val.imag):
                    raise ValueError(
                        f"Non-finite GPU SNR detected: segment={segment_idx}, template={tid}, snr={snrs[i]}"
                    )

                c_dof = _validate_integer_field(
                    dofs[i],
                    "chisq_dof",
                    min_val=1,
                    context=f" (segment={segment_idx}, template={tid}, candidate={i})",
                )

                raw_chisq = chisqs[i]
                if not np.isfinite(raw_chisq):
                    raise ValueError(
                        f"Non-finite GPU chisq detected: segment={segment_idx}, template={tid}, chisq={raw_chisq}"
                    )
                c_chisq = float(raw_chisq)
                c_red = float(c_chisq / c_dof)
                if not np.isfinite(c_red):
                    raise ValueError(
                        f"Non-finite GPU reduced chisq detected: segment={segment_idx}, template={tid}, red_chisq={c_red}"
                    )

                gpu_cands_by_tmpl[tid].append(
                    {
                        "sample_idx": s_idx_val,
                        "snr": snr_val,
                        "chisq": c_chisq,
                        "chisq_dof": c_dof,
                        "red_chisq": c_red,
                    }
                )

    for tid, cands in gpu_cands_by_tmpl.items():
        sample_indices = [c["sample_idx"] for c in cands]
        if len(sample_indices) != len(set(sample_indices)):
            raise ValueError(
                f"Segment {segment_idx} template {tid}: duplicate sample indices detected in GPU candidates: {sample_indices}"
            )

    return gpu_cands_by_tmpl


def compute_canonical_cpu_reference_template(
    template: Any,
    data_segment: Any,
    psd: Any,
    sigmasq: float,
    f_lower: float,
    f_upper: float,
    chisq_bins: Any,
    valid_interval: Tuple[int, int],
    policy: SelectionPolicy,
    segment_idx: int = 0,
    template_idx: int = 0,
) -> List[Dict[str, Any]]:
    """
    Compute canonical CPU reference matched filter and power chisq for a single template.

    Passes unnormalized raw SNR to power_chisq_at_points_from_precomputed to ensure
    accurate physical chisq computation, returning candidates with normalized SNR.
    """
    ref_snr_series, ref_corr, ref_norm = matched_filter_core(
        template,
        data_segment,
        psd=psd,
        low_frequency_cutoff=f_lower,
        high_frequency_cutoff=f_upper,
        h_norm=sigmasq,
    )
    if not np.isfinite(ref_norm) or ref_norm <= 0:
        raise RuntimeError(
            f"Non-finite CPU reference norm: segment={segment_idx}, template={template_idx}, norm={ref_norm}"
        )

    v_start, v_stop = valid_interval
    vals = np.asarray(ref_snr_series)[np.newaxis, v_start:v_stop]
    norms_arr = np.array([ref_norm], dtype=np.float32)
    sigmasq_arr = np.array([sigmasq], dtype=np.float32)

    sel_res = _select_numpy_symmetric(
        vals, norms_arr, sigmasq_arr, v_start, policy
    )
    cands_c = sel_res.get("candidates", {})
    c_s_idxs = cands_c.get("sample_idx", np.empty(0, dtype=np.int64))
    c_snrs = cands_c.get("snr", np.empty(0, dtype=np.complex64))

    for snrv in c_snrs:
        if not np.isfinite(complex(snrv).real) or not np.isfinite(
            complex(snrv).imag
        ):
            raise RuntimeError(
                f"Non-finite CPU SNR detected: segment={segment_idx}, template={template_idx}, snr={snrv}"
            )

    dof = 2 * (len(chisq_bins) - 1) - 2
    if len(c_s_idxs) > 0:
        raw_snr_series_np = np.asarray(ref_snr_series)
        raw_c_snrs = np.asarray([raw_snr_series_np[j] for j in c_s_idxs], dtype=np.complex64)
        ref_c_arr = power_chisq_at_points_from_precomputed(
            ref_corr,
            raw_c_snrs,
            ref_norm,
            chisq_bins,
            c_s_idxs,
        )
        for c_val in ref_c_arr:
            if not np.isfinite(c_val):
                raise RuntimeError(
                    f"Non-finite CPU chisq detected: segment={segment_idx}, template={template_idx}, chisq={c_val}"
                )
    else:
        ref_c_arr = np.empty(0, dtype=np.float32)

    c_list = [
        {
            "sample_idx": int(c_s_idxs[j]),
            "snr": complex(c_snrs[j]),
            "chisq": float(ref_c_arr[j]),
            "chisq_dof": dof,
            "red_chisq": float(ref_c_arr[j] / dof),
        }
        for j in range(len(c_s_idxs))
    ]
    return c_list


def compare_template_candidates(
    gpu_list: List[Dict[str, Any]],
    cpu_list: List[Dict[str, Any]],
    segment_idx: int = 0,
    template_idx: int = 0,
    tolerance_snr: float = 1e-3,
    tolerance_chisq: float = 0.05,
) -> Dict[str, Any]:
    """
    Compare GPU and CPU candidates for parity on a specific segment and template.

    Enforces duplicate sample index rejection, candidate count equality, sample index equality,
    SNR tolerance, DOF equality, and reduced chisq tolerance.
    """
    for c in gpu_list:
        _validate_integer_field(
            c.get("sample_idx"),
            "sample_idx",
            min_val=0,
            context=f" in GPU candidate (segment={segment_idx}, template={template_idx})",
        )
        if "chisq_dof" in c and c["chisq_dof"] is not None:
            _validate_integer_field(
                c["chisq_dof"],
                "chisq_dof",
                min_val=1,
                context=f" in GPU candidate (segment={segment_idx}, template={template_idx})",
            )
    for c in cpu_list:
        _validate_integer_field(
            c.get("sample_idx"),
            "sample_idx",
            min_val=0,
            context=f" in CPU candidate (segment={segment_idx}, template={template_idx})",
        )
        if "chisq_dof" in c and c["chisq_dof"] is not None:
            _validate_integer_field(
                c["chisq_dof"],
                "chisq_dof",
                min_val=1,
                context=f" in CPU candidate (segment={segment_idx}, template={template_idx})",
            )

    gpu_samples = [c["sample_idx"] for c in gpu_list]
    if len(gpu_samples) != len(set(gpu_samples)):
        raise ValueError(
            f"Segment {segment_idx} template {template_idx}: duplicate sample indices in GPU candidates: {gpu_samples}"
        )
    cpu_samples = [c["sample_idx"] for c in cpu_list]
    if len(cpu_samples) != len(set(cpu_samples)):
        raise ValueError(
            f"Segment {segment_idx} template {template_idx}: duplicate sample indices in CPU candidates: {cpu_samples}"
        )

    gpu_sorted = sorted(gpu_list, key=lambda c: c["sample_idx"])
    cpu_sorted = sorted(cpu_list, key=lambda c: c["sample_idx"])

    if len(gpu_sorted) != len(cpu_sorted):
        if len(gpu_sorted) == 0:
            raise RuntimeError(
                f"Segment {segment_idx} template {template_idx}: missing GPU trigger(s) (CPU found {len(cpu_sorted)}, GPU found 0)"
            )
        elif len(cpu_sorted) == 0:
            raise RuntimeError(
                f"Segment {segment_idx} template {template_idx}: spurious GPU trigger(s) (GPU found {len(gpu_sorted)}, CPU found 0)"
            )
        else:
            raise RuntimeError(
                f"Segment {segment_idx} template {template_idx}: trigger count mismatch (GPU found {len(gpu_sorted)}, CPU found {len(cpu_sorted)})"
            )

    if len(gpu_sorted) == 0:
        return {
            "matched": True,
            "max_snr_diff": 0.0,
            "max_chisq_diff": 0.0,
            "max_time_diff": 0,
        }

    max_snr_diff = 0.0
    max_chisq_diff = 0.0
    max_time_diff = 0

    for g_c, c_c in zip(gpu_sorted, cpu_sorted):
        diff_time = int(abs(g_c["sample_idx"] - c_c["sample_idx"]))
        if diff_time != 0:
            raise RuntimeError(
                f"Segment {segment_idx} template {template_idx}: sample position mismatch: GPU={g_c['sample_idx']}, CPU={c_c['sample_idx']}"
            )
        max_time_diff = max(max_time_diff, diff_time)

        diff_snr = float(abs(abs(g_c["snr"]) - abs(c_c["snr"])))
        if not np.isfinite(diff_snr) or diff_snr >= tolerance_snr:
            raise RuntimeError(
                f"Segment {segment_idx} template {template_idx}: SNR mismatch: diff={diff_snr:.6e} (GPU={abs(g_c['snr']):.4f}, CPU={abs(c_c['snr']):.4f})"
            )
        max_snr_diff = max(max_snr_diff, diff_snr)

        if g_c.get("chisq_dof") != c_c.get("chisq_dof"):
            raise RuntimeError(
                f"Segment {segment_idx} template {template_idx}: chisq DOF mismatch: GPU={g_c.get('chisq_dof')}, CPU={c_c.get('chisq_dof')}"
            )

        if g_c.get("red_chisq") is None or c_c.get("red_chisq") is None:
            raise RuntimeError(
                f"Segment {segment_idx} template {template_idx}: missing reduced chisq in candidate comparison"
            )

        diff_chisq = float(abs(g_c["red_chisq"] - c_c["red_chisq"]))
        if not np.isfinite(diff_chisq) or diff_chisq >= tolerance_chisq:
            raise RuntimeError(
                f"Segment {segment_idx} template {template_idx}: chisq mismatch: diff={diff_chisq:.6e} (GPU={g_c['red_chisq']:.6f}, CPU={c_c['red_chisq']:.6f})"
            )
        max_chisq_diff = max(max_chisq_diff, diff_chisq)

    return {
        "matched": True,
        "max_snr_diff": max_snr_diff,
        "max_chisq_diff": max_chisq_diff,
        "max_time_diff": max_time_diff,
    }


def benchmark_production_inspiral(
    num_templates: int = 384,
    size: int = 2097152,  # 2^21
    tile_size: int = 64,
    num_segments: int = 5,
    device: str = "cuda",
    cpu_ref_workers: int = 1,
) -> Dict[str, Any]:
    """
    Benchmark the full production inspiral search workload:
    512 s @ 4096 Hz (N=2^21, delta_f=1/512 Hz), 384 physical TaylorF2 templates,
    colored aLIGOZeroDetHighPower PSD, 5 distinct data segments, Power chisq vetoes,
    and CPU reference comparison.
    """
    flen = size // 2 + 1
    delta_f = 1.0 / 512.0  # 512 seconds of data @ 4096 Hz
    f_lower = 20.0
    f_upper = 1000.0
    valid_interval = (int(0.05 * size), int(0.95 * size))

    t_gen_0 = time.perf_counter()
    print(
        f"   [1/5] Generating physical TaylorF2 bank ({num_templates} templates, flen={flen}, delta_f={delta_f})..."
    )
    templates = generate_taylorf2_bank(
        num_templates, flen, delta_f, f_lower=f_lower, f_upper=f_upper
    )

    print("   [2/5] Generating colored aLIGOZeroDetHighPower PSD...")
    psd = generate_aligo_psd(flen, delta_f, f_lower=f_lower)

    print(f"   [3/5] Generating {num_segments} distinct data segments...")
    data_segments = [
        generate_colored_strain(psd, seed=1000 + s_idx)
        for s_idx in range(num_segments)
    ]

    psd_np = psd.numpy() if hasattr(psd, "numpy") else np.asarray(psd)
    psd_hash = hashlib.sha256(psd_np.tobytes()).hexdigest()

    tmpl_hasher = hashlib.sha256()
    for t in templates:
        t_arr = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
        tmpl_hasher.update(t_arr.tobytes())
    templates_hash = tmpl_hasher.hexdigest()

    # Retain explicit hashes of raw pre-injection noise segments
    raw_noise_segment_hashes = []
    for s_seg in data_segments:
        s_arr = s_seg.numpy() if hasattr(s_seg, "numpy") else np.asarray(s_seg)
        raw_noise_segment_hashes.append(hashlib.sha256(s_arr.tobytes()).hexdigest())

    # Inject a realistic signal into segment 0 to test detection and Power chisq veto parity
    tmpl0_np = (
        templates[0].numpy()
        if hasattr(templates[0], "numpy")
        else np.asarray(templates[0])
    )
    kmin = max(1, int(f_lower / delta_f))
    kmax = min(flen - 1, int(f_upper / delta_f))
    sigmasq_ref = float(
        np.sum(
            (np.abs(tmpl0_np[kmin:kmax]) ** 2)
            * (4.0 * delta_f / np.maximum(psd_np[kmin:kmax], 1e-10))
        )
    )
    target_snr = 25.0
    amp = target_snr / np.sqrt(max(sigmasq_ref, 1e-30))
    k_vec = np.arange(flen)
    shift = np.exp(-2j * np.pi * k_vec * (size // 2) / size).astype(
        np.complex64
    )
    inj = (tmpl0_np * shift * amp).astype(np.complex64)
    data_segments[0][:flen] += inj

    # Hash the final strain arrays after signal injection, immediately before filtering
    final_segment_hashes = []
    for s_seg in data_segments:
        s_arr = s_seg.numpy() if hasattr(s_seg, "numpy") else np.asarray(s_seg)
        final_segment_hashes.append(hashlib.sha256(s_arr.tobytes()).hexdigest())
    t_gen_1 = time.perf_counter()
    generation_time = float(t_gen_1 - t_gen_0)

    if (
        torch is not None
        and torch.cuda.is_available()
        and str(device).startswith("cuda")
    ):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t_setup_0 = time.perf_counter()
    bank_plan = prepare_bank(
        templates,
        tile_size=tile_size,
        f_lower=f_lower,
        f_upper=f_upper,
        device=device,
    )
    psd_plan = bind_psd(bank_plan, psd, device=device)
    chisq_plan = prepare_power_chisq_plan(
        bank_plan, psd_plan, num_bins=16, device=device
    )
    veto_mgr = VetoManager(power_chisq_plan=chisq_plan)

    sample_rate = float(size * delta_f)
    cluster_window = int(1.0 * sample_rate)  # 1-second symmetric clustering
    policy = SelectionPolicy(
        snr_threshold=5.5,
        cluster_policy="symmetric",
        cluster_window=cluster_window,
    )
    engine = SearchEngine(
        bank_plan, policy, veto_manager=veto_mgr, device=device
    )
    if (
        str(device).startswith("cuda")
        and torch is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize()
    t_setup_1 = time.perf_counter()
    setup_time = float(t_setup_1 - t_setup_0)

    # Measure cold-start first segment (unwarmed)
    t_cold_0 = time.perf_counter()
    engine.submit(data_segments[0], psd_plan, valid_interval, block_id=0)
    ready0 = engine.drain()
    if (
        str(device).startswith("cuda")
        and torch is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize()
    t_cold_1 = time.perf_counter()
    first_seg_calc_time = float(t_cold_1 - t_cold_0)

    # Filter remaining distinct segments
    segment_times = [first_seg_calc_time]
    all_ready = [ready0]
    for seg_idx in range(1, num_segments):
        t0 = time.perf_counter()
        engine.submit(
            data_segments[seg_idx],
            psd_plan,
            valid_interval,
            block_id=seg_idx,
        )
        ready_seg = engine.drain()
        if (
            str(device).startswith("cuda")
            and torch is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        segment_times.append(float(t1 - t0))
        all_ready.append(ready_seg)

    if (
        str(device).startswith("cuda")
        and torch is not None
        and torch.cuda.is_available()
    ):
        peak_vram_bytes = torch.cuda.max_memory_allocated()
    else:
        peak_vram_bytes = 0

    engine.close()

    # [4/5] Independent CPU reference comparison across workload
    print(
        f"   [4/5] Running independent CPU reference comparison across all {num_segments} segments and {num_templates} templates..."
    )
    cpu_ref_match = False
    injection_recovered = False
    cpu_max_snr_diff = 0.0
    cpu_max_chisq_diff = 0.0
    cpu_peak_time_diff = 0
    cpu_comparison_count = 0
    gpu_cand_count = sum(
        sum(len(r.get("sample_idx", [])) for r in batch.results)
        for seg_ready in all_ready
        for batch in seg_ready
    )
    gpu_inj_snr = 0.0
    cpu_inj_snr = 0.0
    gpu_inj_chisq = 0.0
    cpu_inj_chisq = 0.0

    t_cpu_ref_0 = time.perf_counter()
    try:
        # Precompute reference norms and chisq bins once across all templates
        sigmasqs_ref = [
            sigmasq(tmpl, psd, f_lower, f_upper) for tmpl in templates
        ]
        all_ref_bins = [
            power_chisq_bins(tmpl, 16, psd, f_lower, f_upper)
            for tmpl in templates
        ]

        for s_idx in range(num_segments):
            seg_batches = all_ready[s_idx]
            gpu_cands_by_tmpl = extract_and_validate_gpu_candidates(
                seg_batches, num_templates, segment_idx=s_idx
            )

            # Evaluate CPU reference for each template in segment
            def _eval_cpu_template(t_idx: int) -> Tuple[int, List[Dict[str, Any]]]:
                c_list = compute_canonical_cpu_reference_template(
                    templates[t_idx],
                    data_segments[s_idx],
                    psd=psd,
                    sigmasq=sigmasqs_ref[t_idx],
                    f_lower=f_lower,
                    f_upper=f_upper,
                    chisq_bins=all_ref_bins[t_idx],
                    valid_interval=valid_interval,
                    policy=policy,
                    segment_idx=s_idx,
                    template_idx=t_idx,
                )
                return t_idx, c_list

            if cpu_ref_workers <= 1:
                # Safe serial reference execution avoids concurrent FFTW planner initialization
                cpu_results = [
                    _eval_cpu_template(t_idx) for t_idx in range(num_templates)
                ]
            else:
                # If concurrent workers are requested, serialize planning/execution
                # through a worker lock to prevent un-thread-safe FFTW planning races
                _ref_lock = threading.Lock()

                def _safe_eval_cpu_template(
                    t_idx: int,
                ) -> Tuple[int, List[Dict[str, Any]]]:
                    with _ref_lock:
                        return _eval_cpu_template(t_idx)

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=cpu_ref_workers
                ) as executor:
                    cpu_results = list(
                        executor.map(
                            _safe_eval_cpu_template, range(num_templates)
                        )
                    )

            for t_idx, cpu_list in cpu_results:
                gpu_list = gpu_cands_by_tmpl[t_idx]
                cmp_res = compare_template_candidates(
                    gpu_list,
                    cpu_list,
                    segment_idx=s_idx,
                    template_idx=t_idx,
                    tolerance_snr=1e-3,
                    tolerance_chisq=0.05,
                )
                cpu_peak_time_diff = max(cpu_peak_time_diff, cmp_res["max_time_diff"])
                cpu_max_snr_diff = max(cpu_max_snr_diff, cmp_res["max_snr_diff"])
                cpu_max_chisq_diff = max(cpu_max_chisq_diff, cmp_res["max_chisq_diff"])

                if s_idx == 0 and t_idx == 0:
                    expected_sample = size // 2
                    inj_g = gpu_list[0]
                    inj_c = cpu_list[0]
                    gpu_inj_snr = float(abs(inj_g["snr"]))
                    cpu_inj_snr = float(abs(inj_c["snr"]))
                    gpu_inj_chisq = float(inj_g["red_chisq"])
                    cpu_inj_chisq = float(inj_c["red_chisq"])
                    if (
                        abs(inj_g["sample_idx"] - expected_sample) > 5
                        or gpu_inj_snr < 20.0
                        or abs(gpu_inj_snr - cpu_inj_snr) > 1e-3
                    ):
                        raise RuntimeError(
                            f"Injected signal parameters incorrect: sample={inj_g['sample_idx']} (expected ~{expected_sample}), SNR={gpu_inj_snr:.2f} (expected ~25.0, CPU SNR={cpu_inj_snr:.2f})"
                        )
                    injection_recovered = True

                cpu_comparison_count += 1

        total_matched_filters = num_templates * num_segments  # 384 * 5 = 1,920
        if cpu_comparison_count != total_matched_filters:
            raise RuntimeError(
                f"Incomplete reference comparison coverage: {cpu_comparison_count} != {total_matched_filters}"
            )
        cpu_ref_match = True
    except Exception as e:
        print(f"   Error: Independent CPU reference comparison failed: {e}")
        raise
    finally:
        t_cpu_ref_1 = time.perf_counter()
        cpu_validation_time_sec = float(t_cpu_ref_1 - t_cpu_ref_0)

    total_matched_filters = num_templates * num_segments  # 384 * 5 = 1,920
    total_calc_time = float(sum(segment_times))
    avg_calc_time_per_seg = float(np.mean(segment_times))
    filtering_wall_time = setup_time + total_calc_time
    total_wall_time = generation_time + setup_time + total_calc_time
    first_iter_wall_time = setup_time + first_seg_calc_time

    return {
        "transform_length": size,
        "duration_sec": 512.0,
        "sample_rate_hz": 4096.0,
        "delta_f": delta_f,
        "waveform_approximant": "TaylorF2",
        "psd_model": "aLIGOZeroDetHighPower",
        "distinct_segments": True,
        "veto_type": "PowerChisq",
        "num_templates": num_templates,
        "tile_size": tile_size,
        "num_segments": num_segments,
        "total_matched_filters": total_matched_filters,
        "cluster_policy": "symmetric",
        "cluster_window_samples": cluster_window,
        "generation_time_sec": generation_time,
        "setup_time_sec": setup_time,
        "first_seg_calc_time_sec": first_seg_calc_time,
        "first_seg_wall_time_sec": first_iter_wall_time,
        "avg_calc_time_per_seg_sec": avg_calc_time_per_seg,
        "total_calc_time_sec": total_calc_time,
        "calc_timer_boundary": "per_segment_submit_filter_veto_drain_sync",
        "filtering_wall_time_sec": filtering_wall_time,
        "total_wall_time_sec": total_wall_time,
        "calc_templates_per_sec": (
            float(total_matched_filters / total_calc_time)
            if total_calc_time > 0
            else 0.0
        ),
        "filtering_templates_per_sec": (
            float(total_matched_filters / filtering_wall_time)
            if filtering_wall_time > 0
            else 0.0
        ),
        "e2e_templates_per_sec": (
            float(total_matched_filters / total_wall_time)
            if total_wall_time > 0
            else 0.0
        ),
        "peak_vram_mb": float(peak_vram_bytes / (1024 * 1024)),
        "vram_budget_bounded": bool(
            peak_vram_bytes <= 16 * 1024 * 1024 * 1024
        ),
        "injection_recovered": injection_recovered,
        "injection_snr": gpu_inj_snr,
        "injection_chisq": gpu_inj_chisq,
        "cpu_inj_chisq": cpu_inj_chisq,
        "gpu_cand_count": gpu_cand_count,
        "cpu_ref_type": "independent_canonical_matched_filter",
        "cpu_reference_parity": cpu_ref_match,
        "cpu_comparison_count": cpu_comparison_count,
        "cpu_validation_time_sec": cpu_validation_time_sec,
        "cpu_max_snr_diff": cpu_max_snr_diff,
        "cpu_max_chisq_diff": cpu_max_chisq_diff,
        "cpu_peak_time_diff": cpu_peak_time_diff,
        "input_provenance": {
            "mass_grid_config": {
                "mass1_range": [10.0, 50.0],
                "mass2_range": [1.4, 25.0],
                "num_templates": num_templates,
                "grid_type": "linear",
            },
            "segment_seed_base": 1000,
            "segment_seeds": [1000 + s_idx for s_idx in range(num_segments)],
            "waveform_approximant": "TaylorF2",
            "psd_model": "aLIGOZeroDetHighPower",
            "psd_sha256": psd_hash,
            "templates_sha256": templates_hash,
            "segment_sha256_list": final_segment_hashes,
            "submitted_segment_sha256_list": final_segment_hashes,
            "raw_noise_segment_sha256_list": raw_noise_segment_hashes,
        },
    }


def validate_qualification_receipt(
    receipt: Dict[str, Any], require_full_workload: bool = True
) -> bool:
    """
    Validate that a qualification receipt conforms to scientific constraints:
    - Expected benchmarks are present.
    - All scientific metrics (SNR, chisq, differences) are strictly finite numbers.
    - Workload dimensions are valid and integers:
      cpu_comparison_count == total_matched_filters == num_templates * num_segments.
    - In production qualification mode (require_full_workload=True), strictly requires
      presence of production_inspiral_workload and 384 templates * 5 segments = 1,920 comparisons.
    """
    if not isinstance(receipt, dict):
        raise ValueError("Receipt must be a dictionary")
    benchmarks = receipt.get("benchmarks", {})
    if not benchmarks:
        raise ValueError("Receipt contains no benchmarks")

    if require_full_workload:
        if "production_inspiral_workload" not in benchmarks:
            raise ValueError(
                "Missing required 'production_inspiral_workload' benchmark for production qualification"
            )

    if "production_inspiral_workload" in benchmarks:
        insp = benchmarks["production_inspiral_workload"]
        for dim_name in [
            "num_templates",
            "num_segments",
            "total_matched_filters",
            "cpu_comparison_count",
        ]:
            if dim_name not in insp:
                raise ValueError(
                    f"Missing required workload dimension '{dim_name}' in production inspiral benchmark"
                )
            val = insp[dim_name]
            if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, np.integer)):
                raise ValueError(
                    f"Workload dimension '{dim_name}' must be an integer, got {type(val).__name__}: {val}"
                )

        num_tmpls = int(insp["num_templates"])
        num_segs = int(insp["num_segments"])
        total = int(insp["total_matched_filters"])
        count = int(insp["cpu_comparison_count"])

        if num_tmpls <= 0 or num_segs <= 0:
            raise ValueError(
                f"Invalid dimensions: num_templates={num_tmpls}, num_segments={num_segs}"
            )

        expected_total = num_tmpls * num_segs
        if total != expected_total:
            raise ValueError(
                f"total_matched_filters ({total}) != num_templates ({num_tmpls}) * num_segments ({num_segs}) = {expected_total}"
            )

        if require_full_workload:
            if num_tmpls != 384 or num_segs != 5 or total != 1920:
                raise ValueError(
                    f"Production qualification requires 384 templates x 5 segments = 1,920 matched filters, "
                    f"got {num_tmpls} templates, {num_segs} segments, {total} total"
                )
            if count != 1920:
                raise ValueError(
                    f"cpu_comparison_count ({count}) does not match expected total matched filters ({expected_total})"
                )
        else:
            if count <= 1:
                raise ValueError(f"cpu_comparison_count must be an integer > 1, got {count}")
            if count > total:
                raise ValueError(
                    f"cpu_comparison_count ({count}) cannot exceed total_matched_filters ({total})"
                )

        numeric_fields = [
            "cpu_max_snr_diff",
            "cpu_max_chisq_diff",
            "injection_snr",
            "injection_chisq",
            "calc_templates_per_sec",
            "filtering_templates_per_sec",
            "e2e_templates_per_sec",
            "total_calc_time_sec",
            "total_filtering_time_sec",
            "total_wall_time_sec",
        ]
        for field in numeric_fields:
            if field in insp:
                val = insp[field]
                if val is None or not np.isfinite(val):
                    raise ValueError(f"Metric '{field}' must be finite, got {val}")

        if insp.get("cpu_reference_parity") is not True:
            raise ValueError("cpu_reference_parity must be True")
        if insp.get("injection_recovered") is not True:
            raise ValueError("injection_recovered must be True")
        if insp.get("cpu_peak_time_diff", 0) != 0:
            raise ValueError(
                f"cpu_peak_time_diff mismatch: {insp.get('cpu_peak_time_diff')} != 0"
            )
        if insp.get("cpu_max_snr_diff", 1.0) >= 1e-3:
            raise ValueError(
                f"cpu_max_snr_diff exceeds threshold: {insp.get('cpu_max_snr_diff')}"
            )
        if insp.get("cpu_max_chisq_diff", 1.0) >= 0.05:
            raise ValueError(
                f"cpu_max_chisq_diff exceeds threshold: {insp.get('cpu_max_chisq_diff')}"
            )

    if "live_streaming_latency" in benchmarks:
        live = benchmarks["live_streaming_latency"]
        for field in [
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "latency_max_ms",
            "latency_mean_ms",
        ]:
            if field in live:
                val = live[field]
                if val is None or not np.isfinite(val):
                    raise ValueError(
                        f"Metric '{field}' in live_streaming_latency must be finite, got {val}"
                    )

    return True


def inspect_historical_receipt(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inspect and validate a historical receipt without production qualification acceptance.

    Validates basic schema and numerical sanity under non-production constraints,
    returning an inspection summary with qualification status.
    """
    valid = validate_qualification_receipt(receipt, require_full_workload=False)
    benchmarks = receipt.get("benchmarks", {})
    insp = benchmarks.get("production_inspiral_workload", {})
    count = insp.get("cpu_comparison_count")
    total = insp.get("total_matched_filters")
    is_production_qualified = (
        count == 1920
        and total == 1920
        and insp.get("num_templates") == 384
        and insp.get("num_segments") == 5
    )
    return {
        "valid_historical_receipt": valid,
        "is_production_qualified": is_production_qualified,
        "cpu_comparison_count": count,
        "total_matched_filters": total,
        "provenance_recorded": "provenance" in receipt,
    }


def capture_execution_provenance(
    repo_root: Optional[Union[str, Path]] = None,
    benchmark_args: Optional[Dict[str, Any]] = None,
    extra_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Capture authentic execution provenance dynamically at runtime:
    - Git commit SHA, branch name, dirty state.
    - SHA256 hashes of core GPU-search source files and benchmark script.
    - Benchmark execution configuration and arguments.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    else:
        repo_root = Path(repo_root).resolve()

    git_commit = "unknown"
    git_branch = "unknown"
    git_dirty = None
    try:
        res_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if res_commit.returncode == 0:
            git_commit = res_commit.stdout.strip()

        res_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if res_branch.returncode == 0:
            git_branch = res_branch.stdout.strip()

        res_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if res_status.returncode == 0:
            status_lines = res_status.stdout.strip().splitlines()
            git_dirty = len(status_lines) > 0
    except Exception as e:
        git_commit = f"unavailable ({e})"

    source_files = [
        "pycbc/filter/gpu_search/engine.py",
        "pycbc/filter/gpu_search/vetoes.py",
        "pycbc/filter/gpu_search/candidates.py",
        "pycbc/filter/gpu_search/plans.py",
        "pycbc/filter/gpu_search/adapter.py",
        "tools/benchmarking/benchmark_gpu_search.py",
    ]
    source_hashes = {}
    for rel_path in source_files:
        p = repo_root / rel_path
        if p.exists() and p.is_file():
            hasher = hashlib.sha256()
            hasher.update(p.read_bytes())
            source_hashes[rel_path] = hasher.hexdigest()
        else:
            source_hashes[rel_path] = "missing"

    provenance = {
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "source_hashes_sha256": source_hashes,
        "benchmark_args": benchmark_args or {},
    }
    if extra_details:
        provenance.update(extra_details)
    return provenance


def save_qualification_receipt(
    report: Dict[str, Any],
    output_path: Union[str, Path],
    require_full_workload: bool = True,
) -> Path:
    """
    Strictly validate and atomically write a qualification receipt.

    Validates report against validate_qualification_receipt first. If validation fails,
    an exception is raised immediately and the destination file is never modified or overwritten.
    Writes to a temporary file in the same directory before atomically renaming to target path.
    """
    validate_qualification_receipt(report, require_full_workload=require_full_workload)

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w") as f:
            json.dump(report, f, indent=2)
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return out_path


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="PyCBC GPU Search Engine Qualification Benchmark"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/gpu_search_qualification_receipt.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Transform length N (default: 1024 for microbenchmark)",
    )
    parser.add_argument(
        "--include-production-inspiral",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include N=2^21 production inspiral benchmark (default: false)",
    )
    parser.add_argument(
        "--cpu-ref-workers",
        type=int,
        default=1,
        help="Number of CPU reference comparison workers (default: 1 for safe serial execution)",
    )
    args = parser.parse_args(argv)

    if args.include_production_inspiral and (torch is None or not torch.cuda.is_available()):
        raise RuntimeError(
            "Requested workload '--include-production-inspiral' cannot execute: "
            "CUDA is unavailable on this device."
        )

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
        "transform_length": args.size,
        "benchmarks": {},
    }

    repo_root = Path(__file__).resolve().parent.parent.parent
    report["provenance"] = capture_execution_provenance(
        repo_root=repo_root,
        benchmark_args=vars(args),
    )

    print(f"=== Running PyCBC GPU Search Engine Qualification on {device_name} (N={args.size}) ===")

    # 1. Tile scaling benchmark
    if torch is not None and torch.cuda.is_available():
        print(f"--> Running GPU tile scaling benchmark (N={args.size})...")
        report["benchmarks"]["gpu_tile_scaling"] = benchmark_tile_scaling(
            device="cuda", num_templates=512, size=args.size, tile_sizes=[1, 16, 64, 128, 256]
        )
    print(f"--> Running CPU tile scaling benchmark baseline (N={args.size})...")
    report["benchmarks"]["cpu_tile_scaling"] = benchmark_tile_scaling(
        device="cpu", num_templates=64, size=args.size, tile_sizes=[1, 16, 64]
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

        # 4. Production Inspiral Workload (N=2^21, 384 templates, 5 segments = 1,920 matched filters)
        if args.include_production_inspiral:
            print("--> Running Production Inspiral workload (N=2^21, 384 templates, 5 segments = 1,920 filters)...")
            report["benchmarks"]["production_inspiral_workload"] = benchmark_production_inspiral(
                num_templates=384,
                size=2097152,
                tile_size=64,
                num_segments=5,
                device="cuda",
                cpu_ref_workers=args.cpu_ref_workers,
            )

    require_full = args.include_production_inspiral
    out_path = save_qualification_receipt(
        report, args.output, require_full_workload=require_full
    )

    print(f"\n=== Benchmark Complete. Receipt written to {out_path} ===")
    print(json.dumps(report["benchmarks"], indent=2))


if __name__ == "__main__":
    main()
