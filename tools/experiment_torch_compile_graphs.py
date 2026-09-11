"""Experiment: Compiled symmetric clustering and offline CUDA graph replay."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

sys_path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path_root not in sys.path:
    sys.path.insert(0, sys_path_root)

from pycbc.filter.gpu_search.candidates import SelectionPolicy, select_tile_candidates
from pycbc.filter.gpu_search.core import correlate_and_ifft


def _symmetric_cluster_core(
    sq_mag: torch.Tensor,
    norms: torch.Tensor,
    window: int,
    snr_thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, total_len = sq_mag.shape
    num_blocks = (total_len + window - 1) // window
    if num_blocks == 0:
        empty_f = torch.empty((batch_size, 0), device=sq_mag.device, dtype=sq_mag.dtype)
        empty_i = torch.empty((batch_size, 0), device=sq_mag.device, dtype=torch.int64)
        empty_b = torch.empty((batch_size, 0), device=sq_mag.device, dtype=torch.bool)
        return empty_f, empty_i, empty_b

    pad = num_blocks * window - total_len
    if pad > 0:
        pad_val = torch.full((batch_size, pad), float("-inf"), device=sq_mag.device, dtype=sq_mag.dtype)
        sq_padded = torch.cat([sq_mag, pad_val], dim=-1)
    else:
        sq_padded = sq_mag

    blocks = sq_padded.view(batch_size, num_blocks, window)
    block_max, block_idx = torch.max(blocks, dim=-1)
    block_idx = block_idx + torch.arange(num_blocks, device=sq_mag.device).unsqueeze(0) * window

    safe_norms = torch.clamp(norms, min=1e-12)
    thresh_sq = (snr_thresh / safe_norms).square().unsqueeze(-1)
    keep = block_max > thresh_sq

    if num_blocks > 1:
        keep[:, 1:] &= block_max[:, 1:] > block_max[:, :-1]
        keep[:, :-1] &= block_max[:, :-1] >= block_max[:, 1:]
        keep[:, 0] &= block_max[:, 0] > block_max[:, 1]

    return block_max, block_idx, keep


def cluster_complex_fn(
    vals: torch.Tensor,
    norms: torch.Tensor,
    window: int,
    snr_thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sq_mag = vals.real.square() + vals.imag.square()
    sq_mag = torch.nan_to_num(sq_mag, nan=0.0)
    return _symmetric_cluster_core(sq_mag, norms, window, snr_thresh)


def cluster_real_imag_fn(
    vals_real: torch.Tensor,
    vals_imag: torch.Tensor,
    norms: torch.Tensor,
    window: int,
    snr_thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sq_mag = vals_real.square() + vals_imag.square()
    sq_mag = torch.nan_to_num(sq_mag, nan=0.0)
    return _symmetric_cluster_core(sq_mag, norms, window, snr_thresh)


def materialize_candidates_host(
    vals: torch.Tensor,
    norms: torch.Tensor,
    sigmasqs: torch.Tensor,
    valid_start: int,
    block_idx: torch.Tensor,
    keep: torch.Tensor,
) -> Dict[str, np.ndarray]:
    if keep.numel() == 0:
        return {
            "template_idx": np.empty(0, dtype=np.int64),
            "sample_idx": np.empty(0, dtype=np.int64),
            "snr": np.empty(0, dtype=np.complex64),
            "sigmasq": np.empty(0, dtype=np.float32),
        }
    surv_indices = torch.nonzero(keep, as_tuple=True)
    sel_tmplt = surv_indices[0]
    block_pos = surv_indices[1]
    if sel_tmplt.numel() == 0:
        return {
            "template_idx": np.empty(0, dtype=np.int64),
            "sample_idx": np.empty(0, dtype=np.int64),
            "snr": np.empty(0, dtype=np.complex64),
            "sigmasq": np.empty(0, dtype=np.float32),
        }
    sel_sample_rel = block_idx[sel_tmplt, block_pos]
    sel_sample = sel_sample_rel + valid_start
    peak_samples = vals[sel_tmplt, sel_sample_rel]
    sel_snr = peak_samples * norms[sel_tmplt]
    sel_sigmasq = sigmasqs[sel_tmplt]
    return {
        "template_idx": sel_tmplt.detach().cpu().numpy(),
        "sample_idx": sel_sample.detach().cpu().numpy(),
        "snr": sel_snr.detach().cpu().numpy(),
        "sigmasq": sel_sigmasq.detach().cpu().numpy(),
    }


def compare_candidate_dicts(ref: Dict[str, np.ndarray], test: Dict[str, np.ndarray]) -> Tuple[bool, Any]:
    keys = ["template_idx", "sample_idx", "snr", "sigmasq"]
    if not all(k in test for k in keys) or not all(k in ref for k in keys):
        return False, None
    for k in keys:
        if ref[k].dtype != test[k].dtype or ref[k].shape != test[k].shape:
            return False, None
    exact_all = (
        np.array_equal(ref["template_idx"], test["template_idx"]) and
        np.array_equal(ref["sample_idx"], test["sample_idx"]) and
        np.array_equal(ref["snr"], test["snr"]) and
        np.array_equal(ref["sigmasq"], test["sigmasq"])
    )
    if len(ref["snr"]) == 0:
        return exact_all, 0.0
    snr_diff = float(np.max(np.abs(ref["snr"] - test["snr"])))
    diff_out = None if not math.isfinite(snr_diff) else snr_diff
    return bool(exact_all), diff_out


def compare_helper_tuples(ref_t: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], test_t: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
    bm_r, bi_r, kp_r = ref_t
    bm_t, bi_t, kp_t = test_t
    eq_mask = bool(torch.equal(kp_r, kp_t))
    eq_idx = bool(torch.equal(bi_r, bi_t))
    eq_max = bool(torch.equal(bm_r, bm_t))
    max_diff = None
    if bm_r.numel() > 0:
        raw_diff = float(torch.max(torch.abs(bm_r - bm_t)).item() if bm_r.is_cuda else torch.max(torch.abs(bm_r - bm_t)))
        max_diff = None if not math.isfinite(raw_diff) else raw_diff
    return {
        "exact_mask": eq_mask,
        "exact_idx": eq_idx,
        "exact_max": eq_max,
        "max_abs_diff": max_diff,
        "all_exact": eq_mask and eq_idx and eq_max,
    }


def make_fixtures(
    batch_size: int,
    tlen: int,
    device: str,
    seed: int = 123,
    quiet: bool = False,
    data_scale: complex = 1.0 + 0.0j,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    kmin, kmax = tlen // 16, tlen // 2
    num_active = kmax - kmin
    data_np = np.zeros(tlen, dtype=np.complex64)
    data_np[kmin:kmax] = data_scale
    tmplt_np = np.zeros((batch_size, tlen), dtype=np.complex64)
    norms_np = rng.uniform(0.8, 1.2, size=batch_size).astype(np.float32)
    sigmasqs_np = rng.uniform(10.0, 50.0, size=batch_size).astype(np.float32)

    # Quiet background in frequency domain
    scale = 0.01 / np.sqrt(tlen)
    rand_corr = (rng.standard_normal((batch_size, num_active)) + 1j * rng.standard_normal((batch_size, num_active))) * scale
    tmplt_np[:, kmin:kmax] = (np.conj(rand_corr) / np.conj(data_scale)).astype(np.complex64)

    if not quiet:
        pulse_times = [3 * tlen // 8, tlen // 2, 5 * tlen // 8]
        freqs = np.arange(kmin, kmax, dtype=np.float64)
        for b in range(batch_size):
            norm_b = float(norms_np[b])
            for idx, t_peak in enumerate(pulse_times):
                target_snr = 8.0 + (idx % 2) * 3.0
                amp = target_snr / (norm_b * float(num_active))
                phase = -2.0 * np.pi * freqs * (float(t_peak) / float(tlen))
                corr_pulse = amp * np.exp(1j * phase)
                # Multiply by conj(data_scale) in denominator so conj(template) * data == corr_pulse
                tmplt_pulse = np.conj(corr_pulse) / np.conj(data_scale)
                tmplt_np[b, kmin:kmax] += tmplt_pulse.astype(np.complex64)

    dev = torch.device(device)
    return (
        torch.as_tensor(tmplt_np, device=dev),
        torch.as_tensor(data_np, device=dev),
        torch.as_tensor(norms_np, device=dev),
        torch.as_tensor(sigmasqs_np, device=dev),
    )


def run_edge_fixtures(device: str, comp_c: Optional[Callable], comp_ri: Optional[Callable]) -> Dict[str, Any]:
    dev = torch.device(device)
    edge_results = {}
    policy = SelectionPolicy(cluster_policy="symmetric", cluster_window=16, snr_threshold=5.0)

    for total_len, v_start in [(0, 0), (1, 7), (50, 7), (1, 11)]:
        B = 4096 if v_start == 11 else 6
        vals = torch.zeros((B, total_len), device=dev, dtype=torch.complex64)
        norms = torch.tensor([1., 1., 1., 1.25, 1.25, 1.25], device=dev)
        sigmasqs = torch.arange(B, device=dev, dtype=torch.float32) + 20
        if v_start == 11:
            # Complex values near the circular threshold expose fused rounding.
            angles = np.random.default_rng(209).uniform(-np.pi, np.pi, B)
            circular = (5.0 * np.exp(1j * angles)).astype(np.complex64)
            vals[:, 0] = torch.as_tensor(circular, device=dev)
            norms = torch.ones(B, device=dev, dtype=torch.float32)
        elif total_len == 1:
            # Independent rows prevent neighboring peaks hiding a boundary case.
            base = np.array([5., 5., 5., 4., 4., 4.], dtype=np.float32)
            base[[0, 3]] = np.nextafter(base[[0, 3]], np.float32(-np.inf))
            base[[2, 5]] = np.nextafter(base[[2, 5]], np.float32(np.inf))
            vals[:, 0] = torch.as_tensor(base, device=dev)
        elif total_len == 50:
            vals[0, 2] = complex(float('nan'), float('nan'))
            vals[0, 20:22] = 8.  # Within-block tie: first index wins.
            vals[1, 3] = vals[1, 20] = 12.  # Equal adjacent block maxima.
            vals[2, 49] = 10.  # Partial final block.
            vals[3, 25] = 20.
            norms[3] = 0.  # Nonzero signal at zero normalization.
            vals[4, 36] = 11.
            vals[5, 5] = 9.

        # Full out_mem with valid_start offset
        out_dummy = torch.zeros((B, v_start + total_len), device=dev, dtype=torch.complex64)
        out_dummy[:, v_start:v_start + total_len] = vals

        prod_cands = select_tile_candidates(out_dummy, norms, sigmasqs, v_start, v_start + total_len, policy)["candidates"]
        ref_tuple = cluster_complex_fn(vals, norms, policy.cluster_window, policy.snr_threshold)
        eager_cands = materialize_candidates_host(vals, norms, sigmasqs, v_start, ref_tuple[1], ref_tuple[2])
        p_eager, _ = compare_candidate_dicts(prod_cands, eager_cands)

        res_c = {"parity_candidates": False, "helper_parity": None, "error": None}
        if comp_c is not None:
            try:
                t_c = comp_c(vals, norms, policy.cluster_window, policy.snr_threshold)
                cands_c = materialize_candidates_host(vals, norms, sigmasqs, v_start, t_c[1], t_c[2])
                p_c, _ = compare_candidate_dicts(prod_cands, cands_c)
                res_c["parity_candidates"] = p_c
                res_c["helper_parity"] = compare_helper_tuples(ref_tuple, t_c)
            except Exception as e:
                res_c["error"] = str(e)

        res_ri = {"parity_candidates": False, "helper_parity": None, "error": None}
        if comp_ri is not None:
            try:
                t_ri = comp_ri(vals.real, vals.imag, norms, policy.cluster_window, policy.snr_threshold)
                cands_ri = materialize_candidates_host(vals, norms, sigmasqs, v_start, t_ri[1], t_ri[2])
                p_ri, _ = compare_candidate_dicts(prod_cands, cands_ri)
                res_ri["parity_candidates"] = p_ri
                res_ri["helper_parity"] = compare_helper_tuples(ref_tuple, t_ri)
            except Exception as e:
                res_ri["error"] = str(e)

        edge_results[f"shape_len{total_len}_vstart{v_start}"] = {
            "parity_eager": p_eager,
            "compiled_complex": res_c,
            "compiled_real_imag": res_ri,
        }
    return edge_results


def run_experiment(device: str = "cpu", iterations: int = 15, quick: bool = False, output_path: str = "compile_graph_benchmark.json") -> Dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but torch.cuda.is_available() is False")
    dev = torch.device(device)
    order_rng = random.Random(42)
    policy = SelectionPolicy(cluster_policy="symmetric", cluster_window=64, snr_threshold=5.5)

    # This experiment intentionally covers more shapes than the default cache.
    # A per-run configuration prevents the ninth shape being mistaken for an
    # unsupported backend. No production compiler configuration is changed.
    from torch._dynamo import config as dynamo_config
    dynamo_config.recompile_limit = 32

    # Attempt wrapper creation
    c_opts = {"triton.cudagraphs": False} if device == "cuda" else {}
    comp_complex = None
    comp_real_imag = None
    creation_errors = {}
    try:
        comp_complex = torch.compile(cluster_complex_fn, fullgraph=True, dynamic=False, options=c_opts)
    except Exception as error:
        creation_errors["complex"] = str(error)
        comp_complex = None
    try:
        comp_real_imag = torch.compile(cluster_real_imag_fn, fullgraph=True, dynamic=False, options=c_opts)
    except Exception as error:
        creation_errors["real_imag"] = str(error)
        comp_real_imag = None

    shapes = [
        {"name": "N65536_B1", "B": 1, "N": 65536, "quiet": False},
        {"name": "N65536_B16", "B": 16, "N": 65536, "quiet": False},
        {"name": "N65536_B16_quiet", "B": 16, "N": 65536, "quiet": True},
        {"name": "N65536_B16_tailB3", "B": 3, "N": 65536, "quiet": False},
    ]
    if not quick:
        shapes.extend([
            {"name": "N2097152_B1", "B": 1, "N": 2097152, "quiet": False},
            {"name": "N2097152_B4", "B": 4, "N": 2097152, "quiet": False},
            {"name": "N2097152_B4_tailB3", "B": 3, "N": 2097152, "quiet": False},
        ])

    # Edge fixtures execution
    t_edge0 = time.perf_counter()
    edge_results = run_edge_fixtures(device, comp_complex, comp_real_imag)
    edge_fixture_time_s = time.perf_counter() - t_edge0

    source_sha256 = {}
    for p in ["tools/experiment_torch_compile_graphs.py", "pycbc/filter/gpu_search/candidates.py", "pycbc/filter/gpu_search/core.py"]:
        fp = os.path.join(sys_path_root, p)
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                source_sha256[p] = hashlib.sha256(f.read()).hexdigest()

    results = []
    for sh in shapes:
        B, N, quiet = sh["B"], sh["N"], sh["quiet"]
        v_start, v_end = N // 4 + 3, 3 * N // 4 - 5
        kmin, kmax = N // 16, N // 2

        # Prepare fixture A and fixture B with guaranteed distinct data contents (asserted)
        c_scale_B = 0.6 + 0.8j
        tmpltA, dataA, norms, sigmasqs = make_fixtures(B, N, device, seed=10, quiet=quiet, data_scale=1.0 + 0.0j)
        tmpltB, dataB, _, _ = make_fixtures(B, N, device, seed=20, quiet=quiet, data_scale=c_scale_B)
        assert not torch.equal(dataA, dataB), "Data fixture B must differ from data fixture A"
        assert not torch.equal(tmpltA, tmpltB), "Template fixture B must differ from template fixture A"

        cout_ws = torch.zeros((B, N), device=dev, dtype=torch.complex64)
        out_ws = torch.zeros((B, N), device=dev, dtype=torch.complex64)

        correlate_and_ifft(tmpltA, dataA, cout_ws, out_ws, N, kmin=kmin, kmax=kmax, batch_size=B)
        # Clone valsA frozen so subsequent timed core executions do not mutate selector inputs
        frozen_out = out_ws.clone()
        valsA_frozen = frozen_out[:, v_start:v_end]

        # Measure compilation execution time and verify parity on first calls
        t_comp_c_s = None
        c_err = None
        t_c = None
        if comp_complex is not None:
            try:
                if dev.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                t_c = comp_complex(valsA_frozen, norms, policy.cluster_window, policy.snr_threshold)
                if dev.type == "cuda": torch.cuda.synchronize()
                t_comp_c_s = time.perf_counter() - t0
                for _ in range(2): _ = comp_complex(valsA_frozen, norms, policy.cluster_window, policy.snr_threshold)
            except Exception as e:
                c_err = str(e)
                comp_complex = None

        t_comp_ri_s = None
        ri_err = None
        t_ri = None
        if comp_real_imag is not None:
            try:
                if dev.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                t_ri = comp_real_imag(valsA_frozen.real, valsA_frozen.imag, norms, policy.cluster_window, policy.snr_threshold)
                if dev.type == "cuda": torch.cuda.synchronize()
                t_comp_ri_s = time.perf_counter() - t0
                for _ in range(2): _ = comp_real_imag(valsA_frozen.real, valsA_frozen.imag, norms, policy.cluster_window, policy.snr_threshold)
            except Exception as e:
                ri_err = str(e)
                comp_real_imag = None

        # Production and Eager Parity
        prod_cands = select_tile_candidates(out_ws, norms, sigmasqs, v_start, v_end, policy)["candidates"]
        ref_tuple = cluster_complex_fn(valsA_frozen, norms, policy.cluster_window, policy.snr_threshold)
        eager_cands = materialize_candidates_host(valsA_frozen, norms, sigmasqs, v_start, ref_tuple[1], ref_tuple[2])
        p_eager, _ = compare_candidate_dicts(prod_cands, eager_cands)

        p_comp, diff_comp = False, None
        tuple_parity_c = None
        if t_c is not None:
            c_cands = materialize_candidates_host(valsA_frozen, norms, sigmasqs, v_start, t_c[1], t_c[2])
            p_comp, diff_comp = compare_candidate_dicts(prod_cands, c_cands)
            tuple_parity_c = compare_helper_tuples(ref_tuple, t_c)

        p_ri, diff_ri = False, None
        tuple_parity_ri = None
        if t_ri is not None:
            ri_cands = materialize_candidates_host(valsA_frozen, norms, sigmasqs, v_start, t_ri[1], t_ri[2])
            p_ri, diff_ri = compare_candidate_dicts(prod_cands, ri_cands)
            tuple_parity_ri = compare_helper_tuples(ref_tuple, t_ri)

        if quiet:
            assert len(prod_cands["sample_idx"]) == 0, f"Quiet fixture produced {len(prod_cands['sample_idx'])} triggers"
        else:
            assert 0 < len(prod_cands["sample_idx"]) <= B * 8, f"Trigger count out of bounds: {len(prod_cands['sample_idx'])}"

        # CUDA Graph Capture for core
        g_core = None
        graph_status = {"scope": "core_only", "captured": False, "error": None, "exact_match": False, "cands_exact": False}
        static_tmplt = tmpltA.clone()
        static_data = dataA.clone()
        static_cout = cout_ws.clone()
        static_out = out_ws.clone()

        if device == "cuda":
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        correlate_and_ifft(static_tmplt, static_data, static_cout, static_out, N, kmin=kmin, kmax=kmax, batch_size=B)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    correlate_and_ifft(static_tmplt, static_data, static_cout, static_out, N, kmin=kmin, kmax=kmax, batch_size=B)

                # Validate replay against fresh eager core on altered dataB/tmpltB
                static_tmplt.copy_(tmpltB)
                static_data.copy_(dataB)
                g.replay()
                ref_cout = torch.zeros_like(static_cout)
                ref_out = torch.zeros_like(static_out)
                correlate_and_ifft(tmpltB, dataB, ref_cout, ref_out, N, kmin=kmin, kmax=kmax, batch_size=B)
                torch.cuda.synchronize()
                eq_cout = torch.equal(static_cout, ref_cout)
                eq_out = torch.equal(static_out, ref_out)
                # Validate candidate fields from graph output
                ref_cands_B = select_tile_candidates(ref_out, norms, sigmasqs, v_start, v_end, policy)["candidates"]
                graph_cands_B = select_tile_candidates(static_out, norms, sigmasqs, v_start, v_end, policy)["candidates"]
                p_cands_graph, _ = compare_candidate_dicts(ref_cands_B, graph_cands_B)

                if eq_cout and eq_out:
                    g_core = g
                    graph_status = {"scope": "core_only", "captured": True, "exact_match": True, "cands_exact": p_cands_graph, "error": None}
                else:
                    graph_status = {"scope": "core_only", "captured": True, "exact_match": False, "cands_exact": False, "error": "Replay mismatch on altered data/templates"}
            except Exception as e:
                graph_status = {"scope": "core_only", "captured": False, "exact_match": False, "cands_exact": False, "error": str(e)}

        # Timed routes execution
        route_times: Dict[str, List[float]] = {
            "sel_eager": [],
            "sel_compiled_complex": [],
            "sel_compiled_real_imag": [],
            "core_eager": [],
            "core_graph": [],
            "core_plus_sel_eager": [],
            "core_plus_sel_graph": [],
        }
        peak_vram = {}
        inputs_pool = [(tmpltA, dataA), (tmpltB, dataB)]

        for it in range(iterations):
            cur_t, cur_d = inputs_pool[it % 2]
            routelist = ["sel_eager", "core_eager", "core_plus_sel_eager"]
            if comp_complex is not None:
                routelist.append("sel_compiled_complex")
            if comp_real_imag is not None:
                routelist.append("sel_compiled_real_imag")
            if g_core is not None:
                routelist.extend(["core_graph", "core_plus_sel_graph"])
            order_rng.shuffle(routelist)

            for rname in routelist:
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                    torch.cuda.synchronize()
                t_start = time.perf_counter()

                if rname == "sel_eager":
                    bm, bi, kp = cluster_complex_fn(valsA_frozen, norms, policy.cluster_window, policy.snr_threshold)
                    _ = materialize_candidates_host(valsA_frozen, norms, sigmasqs, v_start, bi, kp)
                elif rname == "sel_compiled_complex":
                    bm, bi, kp = comp_complex(valsA_frozen, norms, policy.cluster_window, policy.snr_threshold)
                    _ = materialize_candidates_host(valsA_frozen, norms, sigmasqs, v_start, bi, kp)
                elif rname == "sel_compiled_real_imag":
                    bm, bi, kp = comp_real_imag(valsA_frozen.real, valsA_frozen.imag, norms, policy.cluster_window, policy.snr_threshold)
                    _ = materialize_candidates_host(valsA_frozen, norms, sigmasqs, v_start, bi, kp)
                elif rname == "core_eager":
                    static_tmplt.copy_(cur_t)
                    static_data.copy_(cur_d)
                    correlate_and_ifft(static_tmplt, static_data, cout_ws, out_ws, N, kmin=kmin, kmax=kmax, batch_size=B)
                elif rname == "core_graph":
                    static_tmplt.copy_(cur_t)
                    static_data.copy_(cur_d)
                    g_core.replay()
                elif rname == "core_plus_sel_eager":
                    static_tmplt.copy_(cur_t)
                    static_data.copy_(cur_d)
                    correlate_and_ifft(static_tmplt, static_data, cout_ws, out_ws, N, kmin=kmin, kmax=kmax, batch_size=B)
                    v = out_ws[:, v_start:v_end]
                    bm, bi, kp = cluster_complex_fn(v, norms, policy.cluster_window, policy.snr_threshold)
                    _ = materialize_candidates_host(v, norms, sigmasqs, v_start, bi, kp)
                elif rname == "core_plus_sel_graph":
                    static_tmplt.copy_(cur_t)
                    static_data.copy_(cur_d)
                    g_core.replay()
                    v = static_out[:, v_start:v_end]
                    bm, bi, kp = cluster_complex_fn(v, norms, policy.cluster_window, policy.snr_threshold)
                    _ = materialize_candidates_host(v, norms, sigmasqs, v_start, bi, kp)

                if dev.type == "cuda": torch.cuda.synchronize()
                route_times[rname].append(time.perf_counter() - t_start)
                if dev.type == "cuda":
                    peak_vram[rname] = max(peak_vram.get(rname, 0), torch.cuda.max_memory_allocated(dev))

        stats = {}
        for rk, times in route_times.items():
            if times:
                stats[rk] = {
                    "p50_ms": float(np.percentile(times, 50) * 1000),
                    "p95_ms": float(np.percentile(times, 95) * 1000),
                    "raw_s": times,
                }

        results.append({
            "shape": sh,
            "parity_candidates": {"eager": p_eager, "compiled_complex": p_comp, "compiled_real_imag": p_ri},
            "helper_tuple_parity": {"compiled_complex": tuple_parity_c, "compiled_real_imag": tuple_parity_ri},
            "numerical_diffs": {"complex": diff_comp, "real_imag": diff_ri},
            "first_compile_s": {"complex": t_comp_c_s, "real_imag": t_comp_ri_s},
            "compile_errors": {"complex": c_err, "real_imag": ri_err},
            "graph_status": graph_status,
            "stats": stats,
            "peak_vram_bytes": peak_vram,
            "valid_for_promotion": False,
            "promotion_reason": "Prototype only: measurement and parity audit; no production integration claim",
        })

    receipt = {
        "metadata": {
            "experiment": "compiled_clustering_and_cuda_graphs",
            "status": "prototype_only",
            "device": device,
            "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
            "torch_version": str(torch.__version__),
            "edge_fixture_time_s": edge_fixture_time_s,
            "compile_recompile_limit": 32,
            "creation_errors": creation_errors,
            "source_sha256": source_sha256,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": results,
        "edge_results": edge_results,
    }
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(receipt, f, indent=2)
    return receipt


def main():
    parser = argparse.ArgumentParser(description="Experiment: Compiled clustering and CUDA graphs")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=str, default="compile_graph_benchmark.json")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    receipt = run_experiment(device=args.device, iterations=args.iterations, quick=args.quick, output_path=args.output)
    print(f"Experiment complete. Output written to {args.output}")
    for r in receipt["results"]:
        sh = r["shape"]
        stats = r["stats"]
        e_p50 = stats.get("sel_eager", {}).get("p50_ms", 0.0)
        c_p50 = stats.get("sel_compiled_complex", {}).get("p50_ms", 0.0)
        cg_p50 = stats.get("core_graph", {}).get("p50_ms", None)
        print(f"[{sh['name']}] Sel Eager: {e_p50:.2f}ms | Sel Comp: {c_p50:.2f}ms | Graph Core: {cg_p50} | Parity: {r['parity_candidates']}")


if __name__ == "__main__":
    main()
