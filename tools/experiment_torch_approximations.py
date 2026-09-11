"""Trial multirate and reduced-basis approximation methods on physical TaylorF2 bank."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys_path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path_root not in sys.path:
    sys.path.insert(0, sys_path_root)

import pycbc
from pycbc.types import FrequencySeries
from pycbc.waveform import get_fd_waveform
from pycbc.filter.gpu_search.plans import prepare_bank, bind_psd
from pycbc.filter.gpu_search.candidates import SelectionPolicy
from pycbc.filter.gpu_search.engine import SearchEngine
from pycbc.filter.gpu_search.multirate import prepare_multirate_plan, MultirateSearchEngine
from pycbc.filter.gpu_search.reduced_basis import (
    compute_reduced_basis,
    bind_reduced_basis_psd,
    ReducedBasisSearchEngine,
)


def safe_float(val: float) -> Optional[float]:
    if math.isnan(val) or math.isinf(val):
        return None
    return float(val)


def synchronize(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_taylorf2_waveform(m: float, delta_f: float = 0.125, flen: int = 8193) -> np.ndarray:
    hp, _ = get_fd_waveform(
        approximant="TaylorF2",
        mass1=float(m),
        mass2=float(0.8 * m),
        delta_f=float(delta_f),
        f_lower=30.0,
        f_final=1024.0,
        distance=100.0,
    )
    hp.resize(flen)
    # Normalization in complex128 before downcasting to complex64
    arr_c128 = np.asarray(hp.data, dtype=np.complex128)
    max_val = np.max(np.abs(arr_c128))
    if max_val > 0:
        arr_c128 = arr_c128 / max_val
    arr = arr_c128.astype(np.complex64)
    # Zero frequency bins below 30 Hz
    k_30 = int(round(30.0 / delta_f))
    arr[:k_30] = 0.0
    return arr


def compute_synthetic_psd(flen: int, delta_f: float = 0.125) -> np.ndarray:
    freqs = np.linspace(0, (flen - 1) * delta_f, flen, dtype=np.float32)
    f_safe = np.maximum(freqs, 10.0)
    psd = (70.0 / f_safe) ** 4 + 2.0 + 2.0 * (f_safe / 200.0) ** 2
    return psd.astype(np.float32)


def flatten_ticket_results(results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    out_keys = ["template_id", "template_idx", "sample_idx", "snr", "sigmasq"]
    if not results:
        return {
            "template_id": np.empty(0, dtype=np.int64),
            "template_idx": np.empty(0, dtype=np.int64),
            "sample_idx": np.empty(0, dtype=np.int64),
            "snr": np.empty(0, dtype=np.complex64),
            "sigmasq": np.empty(0, dtype=np.float32),
        }
    for r in results:
        lengths = []
        for k in out_keys:
            arr = np.asarray(r[k])
            assert arr.ndim == 1, f"Candidate array for key '{k}' must be 1D, got shape {arr.shape}"
            lengths.append(len(arr))
        assert len(set(lengths)) == 1, f"Candidate array length mismatch across keys: {dict(zip(out_keys, lengths))}"

    merged = {}
    for k in out_keys:
        arrays = [np.asarray(r[k]) for r in results]
        merged[k] = np.concatenate(arrays)
    return merged


def compare_candidates(ref: Dict[str, np.ndarray], test: Dict[str, np.ndarray]) -> Dict[str, Any]:
    per_field_exact = {}
    for k in ["template_id", "template_idx", "sample_idx", "snr", "sigmasq"]:
        r_arr = ref[k]
        t_arr = test[k]
        per_field_exact[k] = bool(r_arr.dtype == t_arr.dtype and r_arr.shape == t_arr.shape and np.array_equal(r_arr, t_arr))
    exact_fields = all(per_field_exact.values())

    ref_keys = list(zip(ref["template_id"], ref["sample_idx"]))
    test_keys = list(zip(test["template_id"], test["sample_idx"]))
    ref_set = set(ref_keys)
    test_set = set(test_keys)
    missed_keys = len(ref_set - test_set)
    extra_keys = len(test_set - ref_set)

    matched_within_4 = 0
    max_snr_mag_err = 0.0
    max_snr_cplx_err = 0.0

    for r_tid, r_sidx, r_snr in zip(ref["template_id"], ref["sample_idx"], ref["snr"]):
        best_match = None
        best_dist = 5
        for t_tid, t_sidx, t_snr in zip(test["template_id"], test["sample_idx"], test["snr"]):
            if r_tid == t_tid and abs(r_sidx - t_sidx) <= 4:
                if abs(r_sidx - t_sidx) < best_dist:
                    best_dist = abs(r_sidx - t_sidx)
                    best_match = t_snr
        if best_match is not None:
            matched_within_4 += 1
            c_err = float(np.abs(r_snr - best_match))
            m_err = float(abs(np.abs(r_snr) - np.abs(best_match)))
            max_snr_cplx_err = max(max_snr_cplx_err, c_err)
            max_snr_mag_err = max(max_snr_mag_err, m_err)

    recall = float(matched_within_4 / len(ref_keys)) if ref_keys else 1.0
    false_positive_count = len(test_keys) - matched_within_4 if not ref_keys else extra_keys
    detection_miss = bool(len(ref_keys) > 0 and len(test_keys) == 0)

    return {
        "exact_5_fields": exact_fields,
        "per_field_exact": per_field_exact,
        "detection_miss": detection_miss,
        "ref_count": len(ref_keys),
        "test_count": len(test_keys),
        "missed_keys": missed_keys,
        "extra_keys": extra_keys,
        "recall_within_4": recall,
        "false_positives": false_positive_count,
        "max_snr_cplx_err": safe_float(max_snr_cplx_err),
        "max_snr_mag_err": safe_float(max_snr_mag_err),
    }


def run_experiment(device: str = "cpu", iterations: int = 3, output_path: str = "torch_approximations_benchmark.json") -> Dict[str, Any]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    stage_errors: Dict[str, str] = {}
    delta_f = 0.125
    N = 16384
    flen = N // 2 + 1  # 8193
    v_start, v_end = N // 4, 3 * N // 4  # 4096 to 12288
    injection_sample = N // 2 + 3       # 8195 (strictly inside valid interval)
    policy = SelectionPolicy(snr_threshold=5.5, cluster_policy="live_peak")
    calib_policy = SelectionPolicy(snr_threshold=0.0, cluster_policy="live_peak")

    owned_engines = []
    full_engine: Optional[SearchEngine] = None
    calib_engine: Optional[SearchEngine] = None
    multirate_routes: Dict[str, Dict[str, Any]] = {}
    reduced_basis_routes: Dict[str, Dict[str, Any]] = {}
    active_routes: List[str] = ["full_engine"]
    science_validation: Dict[str, Any] = {}
    benchmark_summary: Dict[str, Any] = {}
    routes_metadata: Dict[str, Any] = {}
    calibration_records: Dict[str, Any] = {}

    masses = np.linspace(8.0, 35.0, 16)
    t_full_plan = 0.0
    t_full_eng = 0.0
    t_full_first_sub = 0.0
    route_setup_timings: Dict[str, Dict[str, float]] = {}
    first_submit_timings: Dict[str, float] = {}

    try:
        # 1. Build Physical Bank: 16 TaylorF2 waveforms
        bank_templates = []
        for idx, m in enumerate(masses):
            data = generate_taylorf2_waveform(m, delta_f=delta_f, flen=flen)
            fs = FrequencySeries(data, delta_f=delta_f)
            fs.id = 100 + idx
            fs.f_lower = 30.0
            fs.f_upper = 1024.0
            bank_templates.append(fs)

        # 2. Synthetic colored PSD
        psd_data = compute_synthetic_psd(flen, delta_f=delta_f)
        psd_fs = FrequencySeries(psd_data, delta_f=delta_f)
        coarse_psd_data_d4 = psd_data[: (N // 4) // 2 + 1]
        coarse_psd_fs_d4 = FrequencySeries(coarse_psd_data_d4, delta_f=delta_f)
        coarse_psd_data_d8 = psd_data[: (N // 8) // 2 + 1]
        coarse_psd_fs_d8 = FrequencySeries(coarse_psd_data_d8, delta_f=delta_f)

        # 3. Setup Routes and Measure Timings
        # Full Engine setup timing (strictly excludes calibration engine)
        synchronize(device)
        t0 = time.perf_counter()
        full_bank_plan = prepare_bank(bank_templates, tile_size=16, f_lower=30.0, f_upper=1024.0, device=device)
        full_psd_plan = bind_psd(full_bank_plan, psd_fs, psd_version="full_v1", device=device)
        synchronize(device)
        t_full_plan = time.perf_counter() - t0

        synchronize(device)
        t0 = time.perf_counter()
        full_engine = SearchEngine(full_bank_plan, policy, device=device, use_cuda_graphs=False, num_threads=1)
        owned_engines.append(full_engine)
        synchronize(device)
        t_full_eng = time.perf_counter() - t0
        route_setup_timings["full_engine"] = {"plan_s": t_full_plan, "engine_s": t_full_eng}

        # Create Calibration engine separately
        calib_engine = SearchEngine(full_bank_plan, calib_policy, device=device, use_cuda_graphs=False, num_threads=1)
        owned_engines.append(calib_engine)

        dummy_fs = FrequencySeries(np.zeros(flen, dtype=np.complex64), delta_f=delta_f)
        synchronize(device)
        t0 = time.perf_counter()
        full_first_ticket = full_engine.submit(dummy_fs, full_psd_plan, valid_interval=(v_start, v_end), block_id=0, already_overwhitened=False)
        dr = full_engine.drain()
        synchronize(device)
        t_full_first_sub = time.perf_counter() - t0
        first_submit_timings["full_engine"] = t_full_first_sub
        assert len(dr) == 1 and dr[0] is full_first_ticket, "full_engine first submit must return matching ticket"
        assert dr[0].completed and not dr[0].aborted and not dr[0].overflow and dr[0].block_id == 0

        # Multirate Routes
        for d in [4, 8]:
            try:
                synchronize(device)
                t0 = time.perf_counter()
                mr_plan = prepare_multirate_plan(bank_templates, decimation_factor=d, tile_size=16, device=device)
                c_psd = coarse_psd_fs_d4 if d == 4 else coarse_psd_fs_d8
                mr_coarse_psd_plan = bind_psd(mr_plan.coarse_bank_plan, c_psd, psd_version=f"mr_c_d{d}", device=device)
                mr_full_psd_plan = bind_psd(mr_plan.full_bank_plan, psd_fs, psd_version=f"mr_f_d{d}", device=device)
                synchronize(device)
                t_plan_s = time.perf_counter() - t0

                for m_val in [0.0, 1.0, 5.5]:
                    r_name = f"multirate_d{d}_m{str(m_val).replace('.', '_')}"
                    synchronize(device)
                    t0 = time.perf_counter()
                    mr_engine = MultirateSearchEngine(mr_plan, policy, proposal_margin=m_val, device=device, use_cuda_graphs=False)
                    owned_engines.append(mr_engine)
                    synchronize(device)
                    t_eng_s = time.perf_counter() - t0
                    route_setup_timings[r_name] = {"plan_s": t_plan_s, "engine_s": t_eng_s}

                    synchronize(device)
                    t0 = time.perf_counter()
                    tkt = mr_engine.submit(dummy_fs, mr_full_psd_plan, mr_coarse_psd_plan, valid_interval=(v_start, v_end), block_id=0)
                    dr = mr_engine.drain()
                    synchronize(device)
                    first_submit_timings[r_name] = time.perf_counter() - t0
                    assert len(dr) == 1 and dr[0] is tkt, "multirate first submit drain must return matching ticket"
                    assert tkt.completed and not tkt.aborted and not tkt.overflow and tkt.block_id == 0

                    multirate_routes[r_name] = {
                        "engine": mr_engine,
                        "plan": mr_plan,
                        "full_psd": mr_full_psd_plan,
                        "coarse_psd": mr_coarse_psd_plan,
                        "d": d,
                        "margin": m_val,
                    }
                    active_routes.append(r_name)
            except Exception as e:
                stage_errors[f"setup_multirate_d{d}"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        # Reduced-Basis Routes
        rb_configs = [
            ("reduced_basis_r4", 4, 0.0),
            ("reduced_basis_r8", 8, 0.0),
            ("reduced_basis_r16", 16, 0.0),
            ("reduced_basis_auto_tol1e3", None, 1e-3),
        ]
        for r_name, r_rank, r_tol in rb_configs:
            try:
                synchronize(device)
                t0 = time.perf_counter()
                rb_plan = compute_reduced_basis(bank_templates, max_rank=r_rank, tolerance=r_tol, f_lower=30.0, f_upper=1024.0, device=device)
                rb_psd_plan = bind_reduced_basis_psd(rb_plan, psd_fs, psd_version=f"rb_{r_name}", device=device)
                synchronize(device)
                t_plan_s = time.perf_counter() - t0

                synchronize(device)
                t0 = time.perf_counter()
                rb_engine = ReducedBasisSearchEngine(rb_plan, policy, device=device)
                owned_engines.append(rb_engine)
                synchronize(device)
                t_eng_s = time.perf_counter() - t0
                route_setup_timings[r_name] = {"plan_s": t_plan_s, "engine_s": t_eng_s}

                orig_sig0 = full_psd_plan.tile_sigmasqs[0]
                if hasattr(orig_sig0, "cpu"): orig_sig0 = orig_sig0.cpu().numpy()
                rb_sig0 = rb_psd_plan.tile_sigmasqs
                if hasattr(rb_sig0, "cpu"): rb_sig0 = rb_sig0.cpu().numpy()
                sig_rel_err = float(np.max(np.abs(orig_sig0 - rb_sig0) / np.maximum(orig_sig0, 1e-12)))

                synchronize(device)
                t0 = time.perf_counter()
                tkt = rb_engine.submit(dummy_fs, rb_psd_plan, valid_interval=(v_start, v_end), block_id=0)
                dr = rb_engine.drain()
                synchronize(device)
                first_submit_timings[r_name] = time.perf_counter() - t0
                assert len(dr) == 0, "reduced_basis first submit drain must be empty []"
                assert tkt.completed and not tkt.aborted and not tkt.overflow and tkt.block_id == 0

                reduced_basis_routes[r_name] = {
                    "engine": rb_engine,
                    "plan": rb_plan,
                    "psd_plan": rb_psd_plan,
                    "actual_rank": rb_plan.rank,
                    "sigmasq_rel_err": safe_float(sig_rel_err),
                }
                active_routes.append(r_name)
            except Exception as e:
                stage_errors[f"setup_{r_name}"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        # 4. Calibration: Exact frequency bin phase ramp exp(-2j * pi * k * injection_sample / N)
        test_sources = {
            "bank_idx0": {"mass": masses[0], "data": generate_taylorf2_waveform(masses[0], delta_f, flen), "expected_id": 100},
            "bank_idx8": {"mass": masses[8], "data": generate_taylorf2_waveform(masses[8], delta_f, flen), "expected_id": 108},
            "bank_idx15": {"mass": masses[15], "data": generate_taylorf2_waveform(masses[15], delta_f, flen), "expected_id": 115},
            "heldout_9_1": {"mass": 9.1, "data": generate_taylorf2_waveform(9.1, delta_f, flen), "expected_id": None},
            "heldout_12_3": {"mass": 12.3, "data": generate_taylorf2_waveform(12.3, delta_f, flen), "expected_id": None},
            "heldout_24_7": {"mass": 24.7, "data": generate_taylorf2_waveform(24.7, delta_f, flen), "expected_id": None},
        }

        k_bins = np.arange(flen, dtype=np.float64)
        phase_inj = np.exp(-2.0 * np.pi * 1j * k_bins * (float(injection_sample) / float(N))).astype(np.complex64)
        calib_peaks = {}

        for src_name, sinfo in test_sources.items():
            raw_shifted = sinfo["data"] * phase_inj
            fs_in = FrequencySeries(raw_shifted, delta_f=delta_f)
            calib_engine.submit(fs_in, full_psd_plan, valid_interval=(v_start, v_end), block_id=0, already_overwhitened=False)
            tkts = calib_engine.drain()
            assert len(tkts) == 1, "Calib engine drain must return 1 ticket"
            tkt = tkts[0]
            assert tkt.completed and not tkt.aborted and not tkt.overflow, "Calib ticket failed"
            res = flatten_ticket_results(tkt.results)
            assert len(res["snr"]) > 0, f"Calibration failed: zero candidates for {src_name}"

            peak_snr = float(np.max(np.abs(res["snr"])))
            assert peak_snr > 0.0 and math.isfinite(peak_snr), f"Invalid unit peak {peak_snr} for {src_name}"
            calib_peaks[src_name] = peak_snr

            argmax_idx = np.argmax(np.abs(res["snr"]))
            det_tid = res["template_id"][argmax_idx]
            det_sidx = res["sample_idx"][argmax_idx]
            if sinfo["expected_id"] is not None:
                assert det_tid == sinfo["expected_id"], f"Calib template ID mismatch for {src_name}: {det_tid} != {sinfo['expected_id']}"
                assert det_sidx == injection_sample, f"Calib sample index mismatch for {src_name}: {det_sidx} != {injection_sample}"
            calibration_records[src_name] = {
                "unit_peak_snr": safe_float(peak_snr),
                "calib_template_id": int(det_tid),
                "calib_sample_idx": int(det_sidx),
                "sample_shift_from_inj": int(det_sidx - injection_sample),
            }

        # 5. Build 25 Science Verification Test Cases
        science_cases = []
        targets = [5.49, 5.51, 8.0, 12.0]
        for src_name in test_sources:
            unit_peak = calib_peaks[src_name]
            for tgt in targets:
                scale = tgt / unit_peak
                s_data = test_sources[src_name]["data"] * phase_inj * np.float32(scale)
                science_cases.append({
                    "case_name": f"{src_name}_target{str(tgt).replace('.', '_')}",
                    "src_name": src_name,
                    "target_snr": tgt,
                    "scale": float(scale),
                    "data": s_data,
                    "expected_id": test_sources[src_name]["expected_id"],
                })
        science_cases.append({
            "case_name": "quiet_block",
            "src_name": "quiet",
            "target_snr": 0.0,
            "scale": 0.0,
            "data": np.zeros(flen, dtype=np.complex64),
            "expected_id": None,
        })
        assert len(science_cases) == 25, f"Expected 25 science cases, got {len(science_cases)}"

        # Validate calibration scaling across every scaled target using calib_engine (threshold=0.0)
        for sc in science_cases:
            if sc["target_snr"] > 0.0:
                fs_check = FrequencySeries(sc["data"], delta_f=delta_f)
                calib_engine.submit(fs_check, full_psd_plan, valid_interval=(v_start, v_end), block_id=0, already_overwhitened=False)
                tkts = calib_engine.drain()
                assert len(tkts) == 1 and tkts[0].completed and not tkts[0].aborted and not tkts[0].overflow
                res = flatten_ticket_results(tkts[0].results)
                assert len(res["snr"]) > 0
                measured_peak = float(np.max(np.abs(res["snr"])))
                peak_index = int(np.argmax(np.abs(res["snr"])))
                sc["threshold0_peak"] = {
                    "snr_magnitude": measured_peak,
                    "template_id": int(res["template_id"][peak_index]),
                    "sample_idx": int(res["sample_idx"][peak_index]),
                }
                assert abs(measured_peak - sc["target_snr"]) <= 0.01, f"Scaled injection peak {measured_peak} deviates from target {sc['target_snr']} by >0.01 for {sc['case_name']}"

        # Route execution helper
        def run_route(r_key: str, s_data: np.ndarray, b_id: int = 0) -> Dict[str, np.ndarray]:
            fs = FrequencySeries(s_data, delta_f=delta_f)
            if r_key == "full_engine":
                assert full_engine is not None
                submitted = full_engine.submit(fs, full_psd_plan, valid_interval=(v_start, v_end), block_id=b_id, already_overwhitened=False)
                tkts = full_engine.drain()
                assert len(tkts) == 1 and tkts[0] is submitted, "full_engine drain must return matching ticket"
                tkt = tkts[0]
                assert tkt.completed and not tkt.aborted and not tkt.overflow
                assert tkt.block_id == b_id
                return flatten_ticket_results(tkt.results)
            elif r_key.startswith("multirate"):
                mr_info = multirate_routes[r_key]
                tkt = mr_info["engine"].submit(fs, mr_info["full_psd"], mr_info["coarse_psd"], valid_interval=(v_start, v_end), block_id=b_id)
                dr = mr_info["engine"].drain()
                assert len(dr) == 1 and dr[0] is tkt, "multirate drain must return matching ticket"
                assert tkt.completed and not tkt.aborted and not tkt.overflow
                assert tkt.block_id == b_id
                return flatten_ticket_results(tkt.results)
            elif r_key.startswith("reduced_basis"):
                rb_info = reduced_basis_routes[r_key]
                tkt = rb_info["engine"].submit(fs, rb_info["psd_plan"], valid_interval=(v_start, v_end), block_id=b_id)
                dr = rb_info["engine"].drain()
                assert len(dr) == 0, "reduced_basis drain must return empty list []"
                assert tkt.completed and not tkt.aborted and not tkt.overflow
                assert tkt.block_id == b_id
                return flatten_ticket_results(tkt.results)
            raise ValueError(f"Unknown route {r_key}")

        # 6. Validate All 25 Science Cases Outside Timed Loops
        for case_idx, sc in enumerate(science_cases):
            c_name = sc["case_name"]
            sc_res: Dict[str, Any] = {
                "case_name": c_name,
                "src_name": sc["src_name"],
                "target_snr": sc["target_snr"],
                "scale": sc["scale"],
                "threshold0_peak": sc.get("threshold0_peak"),
            }
            try:
                ref_cands = run_route("full_engine", sc["data"], b_id=case_idx)
                ref_count = len(ref_cands["sample_idx"])
                ref_peak = float(np.max(np.abs(ref_cands["snr"]))) if ref_count > 0 else 0.0
                sc_res["ref_cands_count"] = ref_count
                sc_res["ref_peak_snr"] = safe_float(ref_peak)

                if sc["target_snr"] < 5.5:
                    assert ref_count == 0, f"Expected 0 cands for subthreshold {c_name}, got {ref_count}"
                else:
                    assert ref_count > 0, f"Expected detections for superthreshold {c_name}"
                    assert abs(ref_peak - sc["target_snr"]) <= 0.05, f"Ref peak {ref_peak} deviates from target {sc['target_snr']}"
                    argmax_idx = np.argmax(np.abs(ref_cands["snr"]))
                    det_tid = ref_cands["template_id"][argmax_idx]
                    det_sidx = ref_cands["sample_idx"][argmax_idx]
                    sc_res["ref_argmax_template_id"] = int(det_tid)
                    sc_res["ref_argmax_sample_idx"] = int(det_sidx)
                    sc_res["ref_sample_shift_from_inj"] = int(det_sidx - injection_sample)
                    if sc["expected_id"] is not None:
                        assert det_tid == sc["expected_id"], f"Ref template ID mismatch: {det_tid} != {sc['expected_id']}"
                        assert det_sidx == injection_sample, f"Ref sample mismatch: {det_sidx} != {injection_sample}"
            except Exception as e:
                stage_errors[f"science_ref_{c_name}"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                sc_res["ref_error"] = str(e)
                science_validation[c_name] = sc_res
                continue

            # Evaluate approximation routes against reference
            for r_name in active_routes:
                if r_name == "full_engine":
                    continue
                try:
                    test_cands = run_route(r_name, sc["data"], b_id=case_idx)
                    sc_res[r_name] = compare_candidates(ref_cands, test_cands)
                except Exception as e:
                    stage_errors[f"science_{c_name}_{r_name}"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    sc_res[r_name] = {"error": str(e)}
            science_validation[c_name] = sc_res

        # 7. Timed Benchmarking on Representative Scenarios
        bench_scenarios = [
            ("quiet", np.zeros(flen, dtype=np.complex64)),
            ("bank_idx0_target8", [sc for sc in science_cases if sc["case_name"] == "bank_idx0_target8_0"][0]["data"]),
            ("heldout_9_1_target8", [sc for sc in science_cases if sc["case_name"] == "heldout_9_1_target8_0"][0]["data"]),
        ]

        # Warmup 1 call per route on bank target 8
        for r_name in active_routes:
            try:
                _ = run_route(r_name, bench_scenarios[1][1], b_id=999)
                synchronize(device)
            except Exception:
                stage_errors[f"warmup_{r_name}"] = traceback.format_exc()

        bench_timings: Dict[str, Dict[str, List[float]]] = {s_name: {r_name: [] for r_name in active_routes} for s_name, _ in bench_scenarios}
        order_rng = random.Random(42)

        for it in range(iterations):
            for s_name, s_data in bench_scenarios:
                shuffled_routes = list(active_routes)
                order_rng.shuffle(shuffled_routes)
                for r_name in shuffled_routes:
                    try:
                        synchronize(device)
                        t0 = time.perf_counter()
                        _ = run_route(r_name, s_data, b_id=it * 100)
                        synchronize(device)
                        bench_timings[s_name][r_name].append(time.perf_counter() - t0)
                    except Exception:
                        stage_errors[f"benchmark_{s_name}_{r_name}_iter{it}"] = traceback.format_exc()

        for s_name in bench_timings:
            benchmark_summary[s_name] = {}
            for r_name in active_routes:
                raw = bench_timings[s_name][r_name]
                if raw:
                    benchmark_summary[s_name][r_name] = {
                        "p50_ms": float(np.percentile(raw, 50) * 1000),
                        "p95_ms": float(np.percentile(raw, 95) * 1000),
                        "raw_s": raw,
                    }

        routes_metadata = {
            "full_engine": {"setup": route_setup_timings.get("full_engine"), "first_submit_s": first_submit_timings.get("full_engine")},
            "multirate": {
                r_name: {
                    "decimation_factor": minfo["d"],
                    "margin": minfo["margin"],
                    "setup": route_setup_timings.get(r_name),
                    "first_submit_s": first_submit_timings.get(r_name),
                    "full_geometry": {
                        "f_lower": minfo["plan"].full_bank_plan.geometry.f_lower,
                        "f_upper": minfo["plan"].full_bank_plan.geometry.f_upper,
                        "filter_length": minfo["plan"].full_bank_plan.geometry.filter_length,
                        "transform_length": minfo["plan"].full_bank_plan.geometry.transform_length,
                    },
                    "coarse_geometry": {
                        "f_lower": minfo["plan"].band.f_lower,
                        "f_upper": minfo["plan"].band.f_upper,
                        "filter_length": minfo["plan"].band.filter_length,
                        "transform_length": minfo["plan"].band.transform_length,
                    },
                    "refinement_limitation": "Prototype performs full bank refinement whenever any coarse proposal exceeds threshold; not template or window selective.",
                } for r_name, minfo in multirate_routes.items()
            },
            "reduced_basis": {
                r_name: {
                    "actual_rank": rbinfo["actual_rank"],
                    "sigmasq_rel_err": rbinfo["sigmasq_rel_err"],
                    "setup": route_setup_timings.get(r_name),
                    "first_submit_s": first_submit_timings.get(r_name),
                    "normalization_basis": "Template sigmasqs and norms are precomputed on reconstructed templates (A @ Vh) under PSD.",
                } for r_name, rbinfo in reduced_basis_routes.items()
            },
        }
    except Exception as e:
        stage_errors["experiment_fatal"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    finally:
        for engine_index, engine in enumerate(reversed(owned_engines)):
            try:
                engine.close()
            except Exception:
                stage_errors[f"cleanup_engine_{engine_index}"] = traceback.format_exc()

    source_sha256 = {}
    for p in [
        "tools/experiment_torch_approximations.py",
        "pycbc/filter/gpu_search/multirate.py",
        "pycbc/filter/gpu_search/reduced_basis.py",
        "pycbc/filter/gpu_search/plans.py",
        "pycbc/filter/gpu_search/engine.py",
    ]:
        fp = os.path.join(sys_path_root, p)
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                source_sha256[p] = hashlib.sha256(f.read()).hexdigest()

    affinity = list(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [0]
    peak_vram = torch.cuda.max_memory_allocated() if (device == "cuda" and torch.cuda.is_available()) else 0

    receipt = {
        "metadata": {
            "experiment": "multirate_and_reduced_basis_approximations",
            "status": "prototype_research_only",
            "promotable": False,
            "promotion_reason": "Research evaluation only: multirate and reduced basis prototypes are not qualified for production pipelines",
            "device": device,
            "iterations": iterations,
            "host": os.uname().nodename,
            "pid": os.getpid(),
            "approximant": "TaylorF2",
            "masses_bank": masses.tolist(),
            "masses_heldout": [9.1, 12.3, 24.7],
            "injection_sample": injection_sample,
            "delta_t": 1.0 / (N * delta_f),
            "delta_f": delta_f,
            "N": N,
            "threshold": 5.5,
            "psd_formula": "(70/f)^4 + 2 + 2*(f/200)^2 (noiseless limitation)",
            "torch_threads": torch.get_num_threads(),
            "affinity": affinity,
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "pycbc_version": getattr(pycbc, "__version__", "unknown"),
            "peak_vram_bytes": peak_vram,
            "source_sha256": source_sha256,
            "calibration_records": calibration_records,
            "routes_metadata": routes_metadata,
            "stage_errors": stage_errors,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "science_validation_25_cases": science_validation,
        "benchmark_timings": benchmark_summary,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(receipt, f, indent=2, allow_nan=False)
    return receipt


def main():
    parser = argparse.ArgumentParser(description="Multirate & Reduced Basis Approximations Benchmark")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=str, default="torch_approximations_benchmark.json")
    args = parser.parse_args()

    receipt = run_experiment(device=args.device, iterations=args.iterations, output_path=args.output)
    print(f"Approximations trial complete. Output written to {args.output}")
    errors = receipt["metadata"].get("stage_errors", {})
    if errors:
        for k, err in errors.items():
            print(f"[ERROR in {k}]:\n{err}", file=sys.stderr)
        sys.exit(1)

    print(f"{'Route':<28} | {'Plan (s)':<10} | {'Eng (s)':<10} | {'1st Sub (s)':<12} | {'Quiet (ms)':<10} | {'Bank8 (ms)':<10}")
    print("-" * 86)
    bm = receipt["benchmark_timings"]
    meta = receipt["metadata"]["routes_metadata"]
    for rk in bm.get("quiet", {}):
        q_p50 = bm["quiet"][rk]["p50_ms"]
        b_p50 = bm["bank_idx0_target8"][rk]["p50_ms"]
        p_time = 0.0
        e_time = 0.0
        f_sub = 0.0
        if rk == "full_engine":
            p_time = meta["full_engine"]["setup"]["plan_s"]
            e_time = meta["full_engine"]["setup"]["engine_s"]
            f_sub = meta["full_engine"]["first_submit_s"]
        elif rk in meta.get("multirate", {}):
            p_time = meta["multirate"][rk]["setup"]["plan_s"]
            e_time = meta["multirate"][rk]["setup"]["engine_s"]
            f_sub = meta["multirate"][rk]["first_submit_s"]
        elif rk in meta.get("reduced_basis", {}):
            p_time = meta["reduced_basis"][rk]["setup"]["plan_s"]
            e_time = meta["reduced_basis"][rk]["setup"]["engine_s"]
            f_sub = meta["reduced_basis"][rk]["first_submit_s"]
        print(f"{rk:<28} | {p_time:<10.3f} | {e_time:<10.3f} | {f_sub:<12.3f} | {q_p50:<10.2f} | {b_p50:<10.2f}")


if __name__ == "__main__":
    main()
