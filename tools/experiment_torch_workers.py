"""Experiment: Persistent workers across shards and CPU/GPU preparation overlap."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import resource
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

sys_path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path_root not in sys.path:
    sys.path.insert(0, sys_path_root)


# --- Global worker cache state (per spawned process) ---
_WORKER_CACHE: Dict[str, Any] = {
    "bank_key": None,
    "bank_plan": None,
    "psd_key": None,
    "psd_plan": None,
    "engine": None,
}


def safe_float(val: float) -> Optional[float]:
    if math.isnan(val) or math.isinf(val):
        return None
    return float(val)


def get_process_rss_mb() -> float:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss_kb / 1024.0 if sys.platform != "darwin" else rss_kb / (1024.0 * 1024.0))


def _worker_init(device: str):
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in worker")
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def generate_template_specs(B: int, flen: int, bank_variant: int, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed + bank_variant * 1000)
    kmin, kmax = 20, flen - 1
    num_act = kmax - kmin
    templates = []
    for _ in range(B):
        spec = np.zeros(flen, dtype=np.complex64)
        raw = (rng.standard_normal(num_act) + 1j * rng.standard_normal(num_act)).astype(np.complex64)
        norm = float(np.linalg.norm(raw))
        spec[kmin:kmax] = raw / (norm if norm > 0 else 1.0)
        templates.append(spec)
    return templates


def compute_psd_data(flen: int, psd_var: float) -> np.ndarray:
    freqs = np.linspace(0, flen - 1, flen, dtype=np.float32)
    return ((2.0 + (freqs / 1000.0) ** 2) * np.float32(psd_var)).astype(np.float32)


def prepare_strain_numpy(spec: Dict[str, Any]) -> np.ndarray:
    """Pure CPU NumPy preparation work: phase ramp injection, noise, and explicit CPU overwhitening."""
    flen = spec["flen"]
    N = spec["N"]
    kmin, kmax = 20, flen - 1
    freqs = np.arange(kmin, kmax, dtype=np.float64)
    t_inj = spec["t_inj"]
    amp = spec["amplitude"]
    tmpl_inj = spec.get("template_inj")
    psd_data = spec["psd_data"]

    strain = np.zeros(flen, dtype=np.complex64)
    if amp > 0.0 and tmpl_inj is not None:
        phase = -2.0 * np.pi * freqs * (float(t_inj) / float(N))
        pulse = tmpl_inj[kmin:kmax] * np.exp(1j * phase).astype(np.complex64) * np.float32(amp)
        strain[kmin:kmax] = pulse

    if spec.get("noise_seed") is not None:
        rng = np.random.default_rng(spec["noise_seed"])
        noise = (rng.standard_normal(kmax - kmin) + 1j * rng.standard_normal(kmax - kmin)) * (0.001 / np.sqrt(N))
        strain[kmin:kmax] += noise.astype(np.complex64)

    # Explicit CPU overwhitening: strain = strain / PSD
    safe_psd = np.maximum(psd_data[kmin:kmax], 1e-12)
    strain[kmin:kmax] /= safe_psd
    return strain


def execute_shard_in_worker(shard_spec: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from pycbc.filter.gpu_search import prepare_bank, bind_psd, SelectionPolicy, SearchEngine
    from pycbc.types import FrequencySeries

    pid = os.getpid()
    t_start = time.perf_counter()
    device = shard_spec["device"]
    N = shard_spec["N"]
    B = shard_spec["B"]
    flen = N // 2 + 1
    tile_size = shard_spec["tile_size"]
    bank_var = shard_spec["bank_variant"]
    bank_seed = shard_spec["bank_seed"]
    psd_var = shard_spec["psd_variant"]
    block_id = shard_spec["block_id"]
    inj_idx = shard_spec["injection_idx"]
    target_snr = shard_spec["target_snr"]

    bank_key = (N, B, tile_size, bank_var, bank_seed, device)
    psd_key = (N, psd_var, bank_key, device)

    t_bank_prep = 0.0
    t_psd_prep = 0.0
    cache_hit_bank = False
    cache_hit_psd = False

    # 1. Bank Plan and Engine cache management
    if _WORKER_CACHE["bank_key"] != bank_key or _WORKER_CACHE["engine"] is None:
        if _WORKER_CACHE["engine"] is not None:
            _WORKER_CACHE["engine"].close()
            _WORKER_CACHE["engine"] = None
        _WORKER_CACHE["bank_plan"] = None
        _WORKER_CACHE["psd_plan"] = None
        _WORKER_CACHE["psd_key"] = None

        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        tmplt_arrays = generate_template_specs(B, flen, bank_var, seed=bank_seed)
        fseries_list = []
        for idx, arr in enumerate(tmplt_arrays):
            fs = FrequencySeries(arr, delta_f=1.0)
            fs.id = 100 + idx
            fseries_list.append(fs)

        bank_plan = prepare_bank(
            fseries_list,
            tile_size=tile_size,
            f_lower=20,
            f_upper=flen - 2,
            device=device,
        )
        policy = SelectionPolicy(snr_threshold=5.5, cluster_policy="live_peak")
        engine = SearchEngine(
            bank_plan,
            policy,
            device=device,
            use_cuda_graphs=False,
            num_workspaces=1,
            enable_async_transfers=False,
            num_threads=1,
        )
        if device == "cuda": torch.cuda.synchronize()
        t_bank_prep = time.perf_counter() - t0

        _WORKER_CACHE["bank_key"] = bank_key
        _WORKER_CACHE["bank_plan"] = bank_plan
        _WORKER_CACHE["engine"] = engine
    else:
        cache_hit_bank = True

    bank_plan = _WORKER_CACHE["bank_plan"]
    engine = _WORKER_CACHE["engine"]

    # 2. PSD Plan cache management
    psd_data = compute_psd_data(flen, psd_var)
    if _WORKER_CACHE["psd_key"] != psd_key or _WORKER_CACHE["psd_plan"] is None:
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        psd_fs = FrequencySeries(psd_data, delta_f=1.0)
        psd_plan = bind_psd(bank_plan, psd_fs, psd_version=f"v_{psd_var}", device=device)
        if device == "cuda": torch.cuda.synchronize()
        t_psd_prep = time.perf_counter() - t0
        _WORKER_CACHE["psd_key"] = psd_key
        _WORKER_CACHE["psd_plan"] = psd_plan
    else:
        cache_hit_psd = True

    psd_plan = _WORKER_CACHE["psd_plan"]

    # Host amplitude calculation using correct tile and row for injection_idx
    amp = 0.0
    tmpl_inj = None
    if target_snr > 0.0:
        tile_idx = inj_idx // tile_size
        row_in_tile = inj_idx % tile_size
        sig_tensor = psd_plan.tile_sigmasqs[tile_idx][row_in_tile]
        sig_val = float(sig_tensor.item() if hasattr(sig_tensor, "item") else sig_tensor)
        amp = target_snr / np.sqrt(max(1e-12, sig_val))
        # Generate exact template for injection
        tmpl_inj = generate_template_specs(B, flen, bank_var, seed=bank_seed)[inj_idx]

    # 3. CPU Preparation work (NumPy with explicit overwhitening)
    t_prep0 = time.perf_counter()
    prep_spec = {
        "flen": flen,
        "N": N,
        "t_inj": shard_spec["t_inj"],
        "amplitude": amp,
        "template_inj": tmpl_inj,
        "noise_seed": shard_spec.get("noise_seed"),
        "psd_data": psd_data,
    }
    strain_np = prepare_strain_numpy(prep_spec)
    t_prep = time.perf_counter() - t_prep0

    # 4. SearchEngine execution with already_overwhitened=True
    strain_fs = FrequencySeries(strain_np, delta_f=1.0)
    if device == "cuda": torch.cuda.synchronize()
    t_search0 = time.perf_counter()
    engine.submit(
        strain_fs,
        psd_plan,
        valid_interval=(N // 4, 3 * N // 4),
        block_id=block_id,
        already_overwhitened=True,
    )
    completed_tickets = engine.drain()
    if device == "cuda": torch.cuda.synchronize()
    t_search = time.perf_counter() - t_search0

    assert len(completed_tickets) == 1, "Expected exactly 1 drained ticket"
    tkt = completed_tickets[0]
    assert tkt.completed and not tkt.aborted and not tkt.overflow, f"Ticket failed: completed={tkt.completed}, abort={tkt.aborted}, overflow={tkt.overflow}"
    assert tkt.block_id == block_id, f"Block ID mismatch: {tkt.block_id} != {block_id}"

    # Deep copy candidate arrays from Ticket.results
    serialized_results = []
    for cand_dict in tkt.results:
        serialized_results.append({
            "template_id": np.array(cand_dict["template_id"], copy=True),
            "template_idx": np.array(cand_dict["template_idx"], copy=True),
            "sample_idx": np.array(cand_dict["sample_idx"], copy=True),
            "snr": np.array(cand_dict["snr"], copy=True),
            "sigmasq": np.array(cand_dict["sigmasq"], copy=True),
        })

    validate_injection(serialized_results, shard_spec)
    worker_rss = get_process_rss_mb()
    t_total = time.perf_counter() - t_start
    affinity = list(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [0]

    return {
        "pid": pid,
        "worker_rss_mb": worker_rss,
        "actual_torch_threads": torch.get_num_threads(),
        "affinity": affinity,
        "block_id": block_id,
        "cache_hit_bank": cache_hit_bank,
        "cache_hit_psd": cache_hit_psd,
        "t_total": t_total,
        "t_bank_prep": t_bank_prep,
        "t_psd_prep": t_psd_prep,
        "t_prep": t_prep,
        "t_search": t_search,
        "results": serialized_results,
    }


def compare_shard_results(res1: List[Dict[str, Any]], res2: List[Dict[str, Any]]) -> bool:
    if len(res1) != len(res2):
        return False
    for r1, r2 in zip(res1, res2):
        for k in ["template_id", "template_idx", "sample_idx", "snr", "sigmasq"]:
            a1, a2 = r1[k], r2[k]
            if a1.dtype != a2.dtype or a1.shape != a2.shape or not np.array_equal(a1, a2):
                return False
    return True


def compute_results_digest(results: List[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for cand_dict in results:
        for k in ["template_id", "template_idx", "sample_idx", "snr", "sigmasq"]:
            arr = cand_dict[k]
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def validate_injection(results, spec):
    count = sum(len(d["sample_idx"]) for d in results)
    if spec["target_snr"] == 0:
        assert count == 0, (spec["block_id"], count)
        return
    expected_id = 100 + spec["injection_idx"]
    for d in results:
        mask = (d["template_id"] == expected_id) & (d["sample_idx"] == spec["t_inj"])
        if np.any(mask):
            peak = float(np.abs(d["snr"][mask][0]))
            assert abs(peak - spec["target_snr"]) < 0.01, peak
            return
    raise AssertionError(f"Missing injection block {spec['block_id']}, template {expected_id}")


def run_part_a_campaigns(device: str, iterations: int) -> Dict[str, Any]:
    N = 65536
    B = 17  # 17 templates with tile_size=16 exercises partial tail (tile 0: 16, tile 1: 1)
    tile_size = 16

    # 4 distinct shards exercising template 0 and tail template B-1, plus quiet block
    shards_spec = [
        # Block 0: Bank 1, PSD 1.0, inject template 0 at N//2 (t=32768), target SNR 9.0
        {"N": N, "B": B, "tile_size": tile_size, "device": device, "bank_variant": 1, "bank_seed": 10, "psd_variant": 1.0, "injection_idx": 0, "t_inj": N // 2, "target_snr": 9.0, "noise_seed": 101, "block_id": 0},
        # Block 1: Bank 1, PSD 1.0, inject tail template B-1 at N//2+7 (t=32775), target SNR 9.0
        {"N": N, "B": B, "tile_size": tile_size, "device": device, "bank_variant": 1, "bank_seed": 10, "psd_variant": 1.0, "injection_idx": B - 1, "t_inj": N // 2 + 7, "target_snr": 9.0, "noise_seed": 102, "block_id": 1},
        # Block 2: Bank 1, PSD 1.5, Quiet block (target SNR 0.0, no noise)
        {"N": N, "B": B, "tile_size": tile_size, "device": device, "bank_variant": 1, "bank_seed": 10, "psd_variant": 1.5, "injection_idx": 0, "t_inj": N // 2 + 14, "target_snr": 0.0, "noise_seed": None, "block_id": 2},
        # Block 3: Bank 2 (bank change), PSD 1.0, inject template 0 at N//2+21 (t=32789), target SNR 9.0
        {"N": N, "B": B, "tile_size": tile_size, "device": device, "bank_variant": 2, "bank_seed": 20, "psd_variant": 1.0, "injection_idx": 0, "t_inj": N // 2 + 21, "target_snr": 9.0, "noise_seed": 104, "block_id": 3},
    ]

    ctx = mp.get_context("spawn")
    order_rng = random.Random(42)
    fresh_times, pers_times = [], []
    parity_checks = []
    cache_audit_per_iter = []
    raw_shard_records = []
    per_shard_timings = []

    expected_bank_hits = [False, True, True, False]
    expected_psd_hits = [False, True, False, False]

    last_fresh_outs = None
    last_pers_outs = None

    for iter_idx in range(iterations):
        campaign_order = ["fresh", "persistent"]
        order_rng.shuffle(campaign_order)
        results_by_route = {}

        for route in campaign_order:
            t0 = time.perf_counter()
            shard_outputs = []
            if route == "fresh":
                for s_spec in shards_spec:
                    with ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_worker_init, initargs=(device,)) as ex:
                        fut = ex.submit(execute_shard_in_worker, s_spec)
                        shard_outputs.append(fut.result(timeout=120))
            else:
                with ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_worker_init, initargs=(device,)) as ex:
                    for s_spec in shards_spec:
                        fut = ex.submit(execute_shard_in_worker, s_spec)
                        shard_outputs.append(fut.result(timeout=120))
            t_wall = time.perf_counter() - t0
            results_by_route[route] = (t_wall, shard_outputs)

        fresh_t, fresh_outs = results_by_route["fresh"]
        pers_t, pers_outs = results_by_route["persistent"]
        fresh_times.append(fresh_t)
        pers_times.append(pers_t)
        last_fresh_outs = fresh_outs
        last_pers_outs = pers_outs

        # Assert and verify cache hits for EVERY iteration
        for route, outputs in [("fresh", fresh_outs), ("persistent", pers_outs)]:
            for record in outputs:
                raw_shard_records.append({"iteration": iter_idx, "route": route,
                    **{k: v for k, v in record.items() if k != "results"},
                    "candidate_digest": compute_results_digest(record["results"])})
        assert not any(s["cache_hit_psd"] for s in fresh_outs)
        actual_fresh_bank = [s["cache_hit_bank"] for s in fresh_outs]
        actual_pers_bank = [s["cache_hit_bank"] for s in pers_outs]
        actual_pers_psd = [s["cache_hit_psd"] for s in pers_outs]

        assert not any(actual_fresh_bank), f"Fresh route unexpectedly hit bank cache in iter {iter_idx}"
        assert actual_pers_bank == expected_bank_hits, f"Persistent bank hits mismatch in iter {iter_idx}: {actual_pers_bank} != {expected_bank_hits}"
        assert actual_pers_psd == expected_psd_hits, f"Persistent PSD hits mismatch in iter {iter_idx}: {actual_pers_psd} != {expected_psd_hits}"

        cache_audit_per_iter.append({
            "iteration": iter_idx,
            "persistent_bank_hits": actual_pers_bank,
            "persistent_psd_hits": actual_pers_psd,
        })

        # Exact 5-field array equality across every block
        iter_exact = True
        for s_idx in range(len(shards_spec)):
            if not compare_shard_results(fresh_outs[s_idx]["results"], pers_outs[s_idx]["results"]):
                iter_exact = False
        parity_checks.append(iter_exact)

    # Candidate counts and injection verification across every block
    cands_summary = []
    for s_idx, s_out in enumerate(last_pers_outs):
        spec = shards_spec[s_idx]
        res = s_out["results"]
        total_cands = sum(len(d["sample_idx"]) for d in res)
        expected_tmpl_id = 100 + spec["injection_idx"]
        expected_t = spec["t_inj"]
        target_snr = spec["target_snr"]

        if target_snr == 0.0:
            # Quiet block verification
            assert total_cands == 0, f"Quiet block {s_idx} expected 0 candidates, got {total_cands}"
        else:
            assert total_cands > 0, f"Triggered block {s_idx} had 0 candidates"
            # Verify expected template injected is detected with exact sample and peak SNR ~ 9.0
            found_inj = False
            for d in res:
                mask = (d["template_id"] == expected_tmpl_id) & (d["sample_idx"] == expected_t)
                if np.any(mask):
                    detected_snr_mag = float(np.abs(d["snr"][mask][0]))
                    assert abs(detected_snr_mag - target_snr) <= 0.05, f"Detected SNR {detected_snr_mag} outside tolerance of target {target_snr}"
                    found_inj = True
                    break
            assert found_inj, f"Shard {s_idx} failed to detect injected template {expected_tmpl_id} at sample {expected_t}"

        cands_summary.append({
            "shard": s_idx,
            "block_id": spec["block_id"],
            "total_candidates": total_cands,
            "injection_detected": (total_cands > 0),
            "digest": compute_results_digest(res),
        })

    # Per-shard timing summary
    for s_idx in range(len(shards_spec)):
        per_shard_timings.append({
            "shard": s_idx,
            "block_id": shards_spec[s_idx]["block_id"],
            "fresh_t_total_s": float(np.mean([last_fresh_outs[s_idx]["t_total"]])),
            "pers_t_total_s": float(np.mean([last_pers_outs[s_idx]["t_total"]])),
            "pers_t_prep_s": float(np.mean([last_pers_outs[s_idx]["t_prep"]])),
            "pers_t_search_s": float(np.mean([last_pers_outs[s_idx]["t_search"]])),
            "pers_bank_hit": last_pers_outs[s_idx]["cache_hit_bank"],
            "pers_psd_hit": last_pers_outs[s_idx]["cache_hit_psd"],
            "worker_rss_mb": last_pers_outs[s_idx]["worker_rss_mb"],
            "worker_pid": last_pers_outs[s_idx]["pid"],
            "worker_threads": last_pers_outs[s_idx]["actual_torch_threads"],
            "worker_affinity": last_pers_outs[s_idx]["affinity"],
        })

    return {
        "all_exact": all(parity_checks),
        "cache_audit_per_iter": cache_audit_per_iter,
        "raw_shard_records": raw_shard_records,
        "fresh_p50_s": float(np.percentile(fresh_times, 50)),
        "fresh_p95_s": float(np.percentile(fresh_times, 95)),
        "persistent_p50_s": float(np.percentile(pers_times, 50)),
        "persistent_p95_s": float(np.percentile(pers_times, 95)),
        "speedup_p50": float(np.percentile(fresh_times, 50) / max(1e-9, np.percentile(pers_times, 50))),
        "raw_fresh_s": fresh_times,
        "raw_persistent_s": pers_times,
        "cands_summary": cands_summary,
        "per_shard_timings": per_shard_timings,
    }


def _worker_overlap_trial(trial_spec: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from pycbc.filter.gpu_search import prepare_bank, bind_psd, SelectionPolicy, SearchEngine
    from pycbc.types import FrequencySeries

    device = trial_spec["device"]
    N = trial_spec["N"]
    B = trial_spec["B"]
    tile_size = trial_spec["tile_size"]
    flen = N // 2 + 1
    iterations = trial_spec["iterations"]

    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    tmplt_arrays = generate_template_specs(B, flen, bank_variant=1, seed=123)
    fseries_list = [FrequencySeries(arr, delta_f=1.0) for arr in tmplt_arrays]
    for idx, fs in enumerate(fseries_list):
        fs.id = 100 + idx

    bank_plan = prepare_bank(fseries_list, tile_size=tile_size, f_lower=20, f_upper=flen - 2, device=device)
    policy = SelectionPolicy(snr_threshold=5.5, cluster_policy="live_peak")
    engine = SearchEngine(bank_plan, policy, device=device, use_cuda_graphs=False, num_workspaces=1, enable_async_transfers=False, num_threads=1)
    psd_data = compute_psd_data(flen, 1.0)
    psd_fs = FrequencySeries(psd_data, delta_f=1.0)
    psd_plan = bind_psd(bank_plan, psd_fs, psd_version="v1", device=device)
    if device == "cuda": torch.cuda.synchronize()
    setup_time_s = time.perf_counter() - t0

    # 4 blocks: alternating template 0 and tail template B-1, plus quiet block
    shards_spec = []
    for k in range(4):
        inj_idx = (B - 1) if (k == 1) else 0
        tgt_snr = 0.0 if (k == 2) else 9.0
        amp = 0.0
        tmpl_inj = None
        if tgt_snr > 0.0:
            tile_idx = inj_idx // tile_size
            row_in_tile = inj_idx % tile_size
            sig_val = float(psd_plan.tile_sigmasqs[tile_idx][row_in_tile].item() if hasattr(psd_plan.tile_sigmasqs[tile_idx][row_in_tile], "item") else psd_plan.tile_sigmasqs[tile_idx][row_in_tile])
            amp = tgt_snr / np.sqrt(max(1e-12, sig_val))
            tmpl_inj = tmplt_arrays[inj_idx]
        shards_spec.append({
            "flen": flen,
            "N": N,
            "t_inj": N // 2 + k * 7,
            "amplitude": amp,
            "template_inj": tmpl_inj,
            "noise_seed": (200 + k) if (tgt_snr > 0.0) else None,
            "psd_data": psd_data,
            "block_id": k,
            "injection_idx": inj_idx,
            "target_snr": tgt_snr,
        })

    # Warmup with already_overwhitened=True
    s_np = prepare_strain_numpy(shards_spec[0])
    engine.submit(FrequencySeries(s_np, delta_f=1.0), psd_plan, valid_interval=(N // 4, 3 * N // 4), block_id=0, already_overwhitened=True)
    _ = engine.drain()
    if device == "cuda": torch.cuda.synchronize()

    serial_times, prefetch_times = [], []
    order_rng = random.Random(99)
    last_serial_blocks = []
    parity_per_iter = []

    for iter_idx in range(iterations):
        order = ["serial", "prefetch"]
        order_rng.shuffle(order)
        iter_results = {}

        for route in order:
            collected_blocks = []
            if route == "serial":
                if device == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                for spec in shards_spec:
                    s_np = prepare_strain_numpy(spec)
                    engine.submit(FrequencySeries(s_np, delta_f=1.0), psd_plan, valid_interval=(N // 4, 3 * N // 4), block_id=spec["block_id"], already_overwhitened=True)
                    tkts = engine.drain()
                    assert len(tkts) == 1 and tkts[0].completed and not tkts[0].aborted and not tkts[0].overflow
                    collected_blocks.append({
                        "block_id": tkts[0].block_id,
                        "results": [{
                            "template_id": np.array(d["template_id"], copy=True),
                            "template_idx": np.array(d["template_idx"], copy=True),
                            "sample_idx": np.array(d["sample_idx"], copy=True),
                            "snr": np.array(d["snr"], copy=True),
                            "sigmasq": np.array(d["sigmasq"], copy=True),
                        } for d in tkts[0].results],
                    })
                if device == "cuda": torch.cuda.synchronize()
                serial_times.append(time.perf_counter() - t0)
                last_serial_blocks = collected_blocks
                iter_results["serial"] = collected_blocks
            else:
                if device == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(prepare_strain_numpy, shards_spec[0])
                    for k in range(len(shards_spec)):
                        cur_s_np = fut.result()
                        if k + 1 < len(shards_spec):
                            fut = pool.submit(prepare_strain_numpy, shards_spec[k + 1])
                        engine.submit(FrequencySeries(cur_s_np, delta_f=1.0), psd_plan, valid_interval=(N // 4, 3 * N // 4), block_id=shards_spec[k]["block_id"], already_overwhitened=True)
                        tkts = engine.drain()
                        assert len(tkts) == 1 and tkts[0].completed and not tkts[0].aborted and not tkts[0].overflow
                        collected_blocks.append({
                            "block_id": tkts[0].block_id,
                            "results": [{
                                "template_id": np.array(d["template_id"], copy=True),
                                "template_idx": np.array(d["template_idx"], copy=True),
                                "sample_idx": np.array(d["sample_idx"], copy=True),
                                "snr": np.array(d["snr"], copy=True),
                                "sigmasq": np.array(d["sigmasq"], copy=True),
                            } for d in tkts[0].results],
                        })
                if device == "cuda": torch.cuda.synchronize()
                prefetch_times.append(time.perf_counter() - t0)
                iter_results["prefetch"] = collected_blocks

        # Strict 5-field comparison across every block for this iteration
        ser_blks = iter_results["serial"]
        pref_blks = iter_results["prefetch"]
        iter_match = True
        for b_idx in range(len(shards_spec)):
            sb = ser_blks[b_idx]
            pb = pref_blks[b_idx]
            validate_injection(sb["results"], shards_spec[b_idx])
            validate_injection(pb["results"], shards_spec[b_idx])
            assert sb["block_id"] == pb["block_id"] == shards_spec[b_idx]["block_id"]
            if not compare_shard_results(sb["results"], pb["results"]):
                iter_match = False
        parity_per_iter.append(iter_match)

    engine.close()

    # Verification of candidate counts and injection fidelity
    block_audits = []
    for b_idx, blk in enumerate(last_serial_blocks):
        spec = shards_spec[b_idx]
        res = blk["results"]
        total_cands = sum(len(d["sample_idx"]) for d in res)
        expected_tmpl_id = 100 + spec["injection_idx"]
        expected_t = spec["t_inj"]
        target_snr = spec["target_snr"]

        if target_snr == 0.0:
            assert total_cands == 0, f"Overlap quiet block {b_idx} expected 0 cands, got {total_cands}"
        else:
            assert total_cands > 0, f"Overlap triggered block {b_idx} had 0 cands"
            found_inj = False
            for d in res:
                mask = (d["template_id"] == expected_tmpl_id) & (d["sample_idx"] == expected_t)
                if np.any(mask):
                    detected_snr_mag = float(np.abs(d["snr"][mask][0]))
                    assert abs(detected_snr_mag - target_snr) <= 0.05, f"Detected SNR {detected_snr_mag} outside tolerance of target {target_snr}"
                    found_inj = True
                    break
            assert found_inj, f"Overlap block {b_idx} failed to detect injected template {expected_tmpl_id} at sample {expected_t}"

        block_audits.append({
            "block_id": blk["block_id"],
            "total_candidates": total_cands,
            "digest": compute_results_digest(res),
        })

    affinity = list(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [0]
    return {
        "setup_time_s": setup_time_s,
        "all_exact": all(parity_per_iter),
        "block_audits": block_audits,
        "serial_p50_ms": float(np.percentile(serial_times, 50) * 1000),
        "serial_p95_ms": float(np.percentile(serial_times, 95) * 1000),
        "prefetch_p50_ms": float(np.percentile(prefetch_times, 50) * 1000),
        "prefetch_p95_ms": float(np.percentile(prefetch_times, 95) * 1000),
        "speedup_p50": float(np.percentile(serial_times, 50) / max(1e-9, np.percentile(prefetch_times, 50))),
        "raw_serial_s": serial_times,
        "raw_prefetch_s": prefetch_times,
        "worker_rss_mb": get_process_rss_mb(),
        "worker_pid": os.getpid(),
        "worker_threads": torch.get_num_threads(),
        "worker_affinity": affinity,
    }


def run_experiment(device: str = "cpu", iterations: int = 3, part: str = "all", shapes_filter: Optional[List[str]] = None, output_path: str = "torch_workers_benchmark.json") -> Dict[str, Any]:
    if iterations <= 0:
        raise ValueError(f"Iterations must be positive, got {iterations}")

    stage_errors: Dict[str, str] = {}
    part_a_res = None
    if part in ["all", "workers"]:
        try:
            part_a_res = run_part_a_campaigns(device=device, iterations=iterations)
        except Exception as e:
            stage_errors["part_a_workers"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

    part_b_results = {}
    if part in ["all", "overlap"]:
        ctx = mp.get_context("spawn")
        available_overlap_shapes = {
            "small": {"name": "small_N65536_B17_tile16", "N": 65536, "B": 17, "tile_size": 16},
            "large": {"name": "large_N2097152_B5_tile4", "N": 2097152, "B": 5, "tile_size": 4},
        }
        if shapes_filter:
            selected_shapes = [available_overlap_shapes[k] for k in shapes_filter if k in available_overlap_shapes]
            if not selected_shapes:
                raise ValueError(f"Invalid shapes specified: {shapes_filter}. Valid: {list(available_overlap_shapes.keys())}")
        else:
            selected_shapes = list(available_overlap_shapes.values())

        for sh in selected_shapes:
            trial_spec = {
                "device": device,
                "N": sh["N"],
                "B": sh["B"],
                "tile_size": sh["tile_size"],
                "iterations": iterations,
            }
            try:
                with ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_worker_init, initargs=(device,)) as ex:
                    fut = ex.submit(_worker_overlap_trial, trial_spec)
                    part_b_results[sh["name"]] = fut.result(timeout=180)
            except Exception as e:
                stage_errors[f"part_b_{sh['name']}"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

    source_sha256 = {}
    for p in ["tools/experiment_torch_workers.py", "pycbc/filter/gpu_search/core.py", "pycbc/filter/gpu_search/candidates.py", "pycbc/filter/gpu_search/engine.py", "pycbc/filter/gpu_search/plans.py"]:
        fp = os.path.join(sys_path_root, p)
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                source_sha256[p] = hashlib.sha256(f.read()).hexdigest()

    receipt = {
        "metadata": {
            "experiment": "persistent_workers_and_prep_overlap",
            "status": "prototype_only",
            "device": device,
            "iterations": iterations,
            "part": part,
            "parent_pid": os.getpid(),
            "parent_rss_mb": get_process_rss_mb(),
            "source_sha256": source_sha256,
            "stage_errors": stage_errors,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "part_a_process_pool": part_a_res,
        "part_b_overlap": part_b_results,
    }
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(receipt, f, indent=2)
    return receipt


def main():
    parser = argparse.ArgumentParser(description="Experiment: Persistent workers and CPU prep overlap")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--part", choices=["all", "workers", "overlap"], default="all")
    parser.add_argument("--shapes", nargs="*", choices=["small", "large"], default=None)
    parser.add_argument("--output", type=str, default="torch_workers_benchmark.json")
    args = parser.parse_args()

    receipt = run_experiment(
        device=args.device,
        iterations=args.iterations,
        part=args.part,
        shapes_filter=args.shapes,
        output_path=args.output,
    )
    print(f"Experiment complete. Output written to {args.output}")
    errors = receipt["metadata"].get("stage_errors", {})
    if errors:
        for k, err in errors.items():
            print(f"[ERROR in {k}]:\n{err}", file=sys.stderr)
        sys.exit(1)

    pa = receipt["part_a_process_pool"]
    if pa is not None:
        print(f"[Part A] Fresh p50: {pa['fresh_p50_s']:.3f}s | Persistent p50: {pa['persistent_p50_s']:.3f}s | Speedup: {pa['speedup_p50']:.2f}x | Exact Parity: {pa['all_exact']}")
        if not pa["all_exact"]:
            sys.exit(1)
    for k, v in receipt["part_b_overlap"].items():
        print(f"[Part B Overlap: {k}] Serial p50: {v['serial_p50_ms']:.2f}ms | Prefetch p50: {v['prefetch_p50_ms']:.2f}ms | Speedup: {v['speedup_p50']:.2f}x | Exact Parity: {v['all_exact']}")
        if not v["all_exact"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
