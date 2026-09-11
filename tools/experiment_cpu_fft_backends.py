"""Benchmark CPU FFT backends and correlation batching in PyCBC."""

import argparse
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import threadpoolctl
except ImportError:
    threadpoolctl = None

try:
    import scipy.fft as sp_fft
except ImportError:
    sp_fft = None

sys_path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path_root not in sys.path:
    sys.path.insert(0, sys_path_root)

from pycbc import scheme
from pycbc.fft import fftw, torchfft
from pycbc.filter.gpu_search.candidates import SelectionPolicy, select_tile_candidates
from pycbc.types import zeros


def safe_float(val: float) -> Optional[float]:
    if math.isnan(val) or math.isinf(val):
        return None
    return float(val)


def make_cpu_fixtures(B: int, N: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    kmin, kmax = 20, N // 2
    num_act = kmax - kmin
    strain = np.zeros(N, dtype=np.complex64)
    strain[kmin:kmax] = 1.0 + 0.0j

    tmplt = np.zeros((B, N), dtype=np.complex64)
    norms = rng.uniform(0.8, 1.2, size=B).astype(np.float32)
    sigmasqs = rng.uniform(10.0, 50.0, size=B).astype(np.float32)

    pulse_times = [3 * N // 8, N // 2, 5 * N // 8]
    freqs = np.arange(kmin, kmax, dtype=np.float64)
    for b in range(B):
        for idx, t_p in enumerate(pulse_times):
            target_snr = 9.0 + (idx % 2) * 3.0
            amp = target_snr / (float(norms[b]) * float(num_act))
            phase = -2.0 * np.pi * freqs * (float(t_p) / float(N))
            corr_pulse = amp * np.exp(1j * phase)
            tmplt[b, kmin:kmax] += np.conj(corr_pulse).astype(np.complex64)
    return tmplt, strain, norms, sigmasqs


def extract_plan_metadata(plan: Any) -> Dict[str, Optional[str]]:
    def _type_name(attr: str) -> Optional[str]:
        obj = getattr(plan, attr, None)
        return type(obj).__name__ if obj is not None else None
    return {
        "_fftw_plan": _type_name("_fftw_plan"),
        "_fftw_batch_plan": _type_name("_fftw_batch_plan"),
        "_mkl_plan": _type_name("_mkl_plan"),
        "_promoted_batch_plan": _type_name("_promoted_batch_plan"),
    }


def compare_candidate_dicts(ref: Dict[str, np.ndarray], test: Dict[str, np.ndarray]) -> Dict[str, Any]:
    keys = ["template_idx", "sample_idx", "snr", "sigmasq"]
    if not all(k in test for k in keys) or not all(k in ref for k in keys):
        return {"all_exact": False, "ids_match": False, "snr_max_diff": None}

    # Guard against length/shape mismatch before array operations
    for k in keys:
        if ref[k].dtype != test[k].dtype or ref[k].shape != test[k].shape:
            return {"all_exact": False, "ids_match": False, "snr_max_diff": None}

    ids_match = (
        np.array_equal(ref["template_idx"], test["template_idx"]) and
        np.array_equal(ref["sample_idx"], test["sample_idx"])
    )
    exact_all = (
        ids_match and
        np.array_equal(ref["sigmasq"], test["sigmasq"]) and
        np.array_equal(ref["snr"], test["snr"])
    )
    snr_diff = None
    if len(ref["snr"]) > 0:
        diff_val = float(np.max(np.abs(ref["snr"] - test["snr"])))
        snr_diff = safe_float(diff_val)
    return {"all_exact": bool(exact_all), "ids_match": bool(ids_match), "snr_max_diff": snr_diff}


def compare_arrays(ref_arr: np.ndarray, test_arr: np.ndarray) -> bool:
    if ref_arr.dtype != test_arr.dtype or ref_arr.shape != test_arr.shape:
        return False
    return bool(np.array_equal(ref_arr, test_arr))


def build_routes(
    B: int,
    N: int,
    kmin: int,
    kmax: int,
    threads: int,
    tmplt_np: np.ndarray,
    strain_np: np.ndarray,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float], Dict[str, str]]:
    routes: Dict[str, Dict[str, Any]] = {}
    setup_costs: Dict[str, float] = {}
    route_errors: Dict[str, str] = {}

    # 1. Raw PyTorch CPU (uses from_numpy views of shared tmplt_np and strain_np)
    try:
        t_s0 = time.perf_counter()
        t_tmplt = torch.from_numpy(tmplt_np)
        t_strain = torch.from_numpy(strain_np)
        raw_cout = torch.zeros((B, N), dtype=torch.complex64)
        raw_out = torch.zeros((B, N), dtype=torch.complex64)

        def run_raw_torch():
            raw_cout.zero_()
            torch.mul(torch.conj(t_tmplt[:, kmin:kmax]), t_strain[kmin:kmax], out=raw_cout[:, kmin:kmax])
            torch.fft.ifft(raw_cout, n=N, dim=-1, norm="forward", out=raw_out)

        def run_raw_torch_fft():
            torch.fft.ifft(raw_cout, n=N, dim=-1, norm="forward", out=raw_out)

        setup_costs["raw_torch"] = time.perf_counter() - t_s0
        routes["raw_torch"] = {
            "fn": run_raw_torch,
            "fft_fn": run_raw_torch_fft,
            "out": lambda: raw_out.numpy().copy(),
            "plan_metadata": {"type": "torch.fft.ifft"},
        }
    except Exception as e:
        route_errors["raw_torch"] = str(e)

    # 2. PyCBC legacy FFTW scalar loop (Reference Route)
    try:
        t_s0 = time.perf_counter()
        with scheme.CPUScheme(num_threads=threads):
            f_scal_srcs = [zeros(N, dtype=np.complex64) for _ in range(B)]
            f_scal_dsts = [zeros(N, dtype=np.complex64) for _ in range(B)]
            f_scal_plans = [fftw.IFFT(f_scal_srcs[i], f_scal_dsts[i], nbatch=1, size=N) for i in range(B)]
            f_scal_src_views = [src.data for src in f_scal_srcs]
            f_scal_dst_views = [dst.data for dst in f_scal_dsts]

        def run_fftw_scalar():
            with scheme.CPUScheme(num_threads=threads):
                for i in range(B):
                    f_scal_src_views[i].fill(0)
                    f_scal_src_views[i][kmin:kmax] = np.conj(tmplt_np[i, kmin:kmax]) * strain_np[kmin:kmax]
                    f_scal_plans[i].execute()

        def run_fftw_scalar_fft():
            with scheme.CPUScheme(num_threads=threads):
                for i in range(B):
                    f_scal_plans[i].execute()

        setup_costs["fftw_legacy_scalar"] = time.perf_counter() - t_s0
        routes["fftw_legacy_scalar"] = {
            "fn": run_fftw_scalar,
            "fft_fn": run_fftw_scalar_fft,
            "out": lambda: np.stack(f_scal_dst_views, axis=0).copy(),
            "plan_metadata": {"type": "fftw.IFFT_scalar_row_loop"},
        }
    except Exception as e:
        route_errors["fftw_legacy_scalar"] = str(e)

    # 3. PyCBC legacy FFTW batched
    try:
        t_s0 = time.perf_counter()
        with scheme.CPUScheme(num_threads=threads):
            f_bat_src = zeros(B * N, dtype=np.complex64)
            f_bat_dst = zeros(B * N, dtype=np.complex64)
            f_bat_src_view = f_bat_src.data.reshape(B, N)
            f_bat_dst_view = f_bat_dst.data.reshape(B, N)
            plan_fftw_bat = fftw.IFFT(f_bat_src, f_bat_dst, nbatch=B, size=N)

        def run_fftw_batch():
            with scheme.CPUScheme(num_threads=threads):
                f_bat_src_view.fill(0)
                f_bat_src_view[:, kmin:kmax] = np.conj(tmplt_np[:, kmin:kmax]) * strain_np[kmin:kmax]
                plan_fftw_bat.execute()

        def run_fftw_batch_fft():
            with scheme.CPUScheme(num_threads=threads):
                plan_fftw_bat.execute()

        setup_costs["fftw_legacy_batch"] = time.perf_counter() - t_s0
        routes["fftw_legacy_batch"] = {
            "fn": run_fftw_batch,
            "fft_fn": run_fftw_batch_fft,
            "out": lambda: f_bat_dst_view.copy(),
            "plan_metadata": {"type": "fftw.IFFT_native_batched"},
        }
    except Exception as e:
        route_errors["fftw_legacy_batch"] = str(e)

    # 4. PyCBC torchfft scalar loop
    try:
        t_s0 = time.perf_counter()
        with scheme.TorchScheme("cpu"):
            t_scal_srcs = [zeros(N, dtype=np.complex64) for _ in range(B)]
            t_scal_dsts = [zeros(N, dtype=np.complex64) for _ in range(B)]
            t_scal_plans = [torchfft.IFFT(t_scal_srcs[i], t_scal_dsts[i], nbatch=1, size=N) for i in range(B)]
            t_scal_src_tensors = [s._data.tensor for s in t_scal_srcs]
            t_scal_dst_views = [d._data.tensor.numpy() for d in t_scal_dsts]
            meta_torch_scal = extract_plan_metadata(t_scal_plans[0])

        def run_torchfft_scalar():
            with scheme.TorchScheme("cpu"):
                for i in range(B):
                    t_scal_src_tensors[i].zero_()
                    torch.mul(torch.conj(t_tmplt[i, kmin:kmax]), t_strain[kmin:kmax], out=t_scal_src_tensors[i][kmin:kmax])
                    t_scal_plans[i].execute()

        def run_torchfft_scalar_fft():
            with scheme.TorchScheme("cpu"):
                for i in range(B):
                    t_scal_plans[i].execute()

        setup_costs["torchfft_scalar"] = time.perf_counter() - t_s0
        routes["torchfft_scalar"] = {
            "fn": run_torchfft_scalar,
            "fft_fn": run_torchfft_scalar_fft,
            "out": lambda: np.stack(t_scal_dst_views, axis=0).copy(),
            "plan_metadata": meta_torch_scal,
        }
    except Exception as e:
        route_errors["torchfft_scalar"] = str(e)

    # 5. PyCBC torchfft batched (default)
    try:
        t_s0 = time.perf_counter()
        with scheme.TorchScheme("cpu"):
            t_bat_src = zeros(B * N, dtype=np.complex64)
            t_bat_dst = zeros(B * N, dtype=np.complex64)
            t_bat_src_tensor = t_bat_src._data.tensor.view(B, N)
            t_bat_dst_view = t_bat_dst._data.tensor.view(B, N).numpy()
            plan_torch_bat = torchfft.IFFT(t_bat_src, t_bat_dst, nbatch=B, size=N)
            meta_torch_bat = extract_plan_metadata(plan_torch_bat)

        def run_torchfft_batch():
            with scheme.TorchScheme("cpu"):
                t_bat_src_tensor.zero_()
                torch.mul(torch.conj(t_tmplt[:, kmin:kmax]), t_strain[kmin:kmax], out=t_bat_src_tensor[:, kmin:kmax])
                plan_torch_bat.execute()

        def run_torchfft_batch_fft():
            with scheme.TorchScheme("cpu"):
                plan_torch_bat.execute()

        setup_costs["torchfft_batch"] = time.perf_counter() - t_s0
        routes["torchfft_batch"] = {
            "fn": run_torchfft_batch,
            "fft_fn": run_torchfft_batch_fft,
            "out": lambda: t_bat_dst_view.copy(),
            "plan_metadata": meta_torch_bat,
        }
    except Exception as e:
        route_errors["torchfft_batch"] = str(e)

    # 6. PyCBC torchfft gated FFTW batch (Only when B=8, N=131072, threads=1)
    if B == 8 and N == 131072 and threads == 1:
        try:
            t_s0 = time.perf_counter()
            orig_gate = os.environ.get("PYCBC_TORCH_CPU_FFTW_BATCH", None)
            try:
                os.environ["PYCBC_TORCH_CPU_FFTW_BATCH"] = "1"
                with scheme.TorchScheme("cpu"):
                    t_gate_src = zeros(B * N, dtype=np.complex64)
                    t_gate_dst = zeros(B * N, dtype=np.complex64)
                    t_gate_src_tensor = t_gate_src._data.tensor.view(B, N)
                    t_gate_dst_view = t_gate_dst._data.tensor.view(B, N).numpy()
                    plan_gated = torchfft.IFFT(t_gate_src, t_gate_dst, nbatch=B, size=N)
                    meta_gated = extract_plan_metadata(plan_gated)

                def run_torchfft_gated():
                    old_val = os.environ.get("PYCBC_TORCH_CPU_FFTW_BATCH", None)
                    os.environ["PYCBC_TORCH_CPU_FFTW_BATCH"] = "1"
                    try:
                        with scheme.TorchScheme("cpu"):
                            t_gate_src_tensor.zero_()
                            torch.mul(torch.conj(t_tmplt[:, kmin:kmax]), t_strain[kmin:kmax], out=t_gate_src_tensor[:, kmin:kmax])
                            plan_gated.execute()
                    finally:
                        if old_val is None:
                            os.environ.pop("PYCBC_TORCH_CPU_FFTW_BATCH", None)
                        else:
                            os.environ["PYCBC_TORCH_CPU_FFTW_BATCH"] = old_val

                def run_torchfft_gated_fft():
                    old_val = os.environ.get("PYCBC_TORCH_CPU_FFTW_BATCH", None)
                    os.environ["PYCBC_TORCH_CPU_FFTW_BATCH"] = "1"
                    try:
                        with scheme.TorchScheme("cpu"):
                            plan_gated.execute()
                    finally:
                        if old_val is None:
                            os.environ.pop("PYCBC_TORCH_CPU_FFTW_BATCH", None)
                        else:
                            os.environ["PYCBC_TORCH_CPU_FFTW_BATCH"] = old_val

                setup_costs["torchfft_gated_fftw_batch"] = time.perf_counter() - t_s0
                routes["torchfft_gated_fftw_batch"] = {
                    "fn": run_torchfft_gated,
                    "fft_fn": run_torchfft_gated_fft,
                    "out": lambda: t_gate_dst_view.copy(),
                    "plan_metadata": meta_gated,
                }
            finally:
                if orig_gate is None:
                    os.environ.pop("PYCBC_TORCH_CPU_FFTW_BATCH", None)
                else:
                    os.environ["PYCBC_TORCH_CPU_FFTW_BATCH"] = orig_gate
        except Exception as e:
            route_errors["torchfft_gated_fftw_batch"] = str(e)

    # 7. Scipy FFT (direct unnormalized norm='forward')
    if sp_fft is not None:
        try:
            t_s0 = time.perf_counter()
            sp_cout = np.zeros((B, N), dtype=np.complex64)
            sp_out = np.zeros((B, N), dtype=np.complex64)

            def run_scipy():
                sp_cout.fill(0)
                sp_cout[:, kmin:kmax] = np.conj(tmplt_np[:, kmin:kmax]) * strain_np[kmin:kmax]
                np.copyto(sp_out, sp_fft.ifft(sp_cout, axis=-1, workers=threads, norm="forward"))

            def run_scipy_fft():
                np.copyto(sp_out, sp_fft.ifft(sp_cout, axis=-1, workers=threads, norm="forward"))

            setup_costs["scipy_fft"] = time.perf_counter() - t_s0
            routes["scipy_fft"] = {
                "fn": run_scipy,
                "fft_fn": run_scipy_fft,
                "out": lambda: sp_out.copy(),
                "plan_metadata": {"workers": threads, "norm": "forward"},
            }
        except Exception as e:
            route_errors["scipy_fft"] = str(e)

    return routes, setup_costs, route_errors


def run_benchmark(threads: int, iterations: int, output_path: str, shape_filter: Optional[List[str]] = None) -> Dict[str, Any]:
    if threads <= 0:
        raise ValueError(f"Threads must be positive, got {threads}")
    if iterations <= 0:
        raise ValueError(f"Iterations must be positive, got {iterations}")

    affinity_list = list(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    if threads > len(affinity_list):
        raise ValueError(f"Requested threads ({threads}) exceeds available CPU affinity count ({len(affinity_list)})")

    torch.set_num_threads(threads)
    actual_torch_threads = torch.get_num_threads()
    threadpool_info = threadpoolctl.threadpool_info() if threadpoolctl is not None else None
    order_rng = random.Random(42)

    all_shapes = [
        {"name": "N32768_B1", "N": 32768, "B": 1},
        {"name": "N32768_B16", "N": 32768, "B": 16},
        {"name": "N131072_B1", "N": 131072, "B": 1},
        {"name": "N131072_B8", "N": 131072, "B": 8},
        {"name": "N2097152_B1", "N": 2097152, "B": 1},
        {"name": "N2097152_B4", "N": 2097152, "B": 4},
    ]
    if shape_filter:
        shapes = [s for s in all_shapes if s["name"] in shape_filter]
        if not shapes:
            raise ValueError(f"No matching shapes found for filter: {shape_filter}")
    else:
        shapes = all_shapes

    source_sha256 = {}
    for p in ["tools/experiment_cpu_fft_backends.py", "pycbc/fft/torchfft.py", "pycbc/fft/fftw.py", "pycbc/filter/gpu_search/core.py"]:
        fp = os.path.join(sys_path_root, p)
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                source_sha256[p] = hashlib.sha256(f.read()).hexdigest()

    policy = SelectionPolicy(cluster_policy="symmetric", cluster_window=64, snr_threshold=5.5)
    results = []

    for sh in shapes:
        B, N = sh["B"], sh["N"]
        tmplt_np, strain_np, norms_np, sig_np = make_cpu_fixtures(B, N, seed=123)
        kmin, kmax = 20, N // 2
        v_start, v_end = N // 4, 3 * N // 4

        # Unrounded analytical complex128 NumPy IFFT reference for primary pulse fixture
        c128_tmplt = tmplt_np.astype(np.complex128)
        c128_strain = strain_np.astype(np.complex128)
        ideal_corr_c128 = np.zeros((B, N), dtype=np.complex128)
        ideal_corr_c128[:, kmin:kmax] = np.conj(c128_tmplt[:, kmin:kmax]) * c128_strain[kmin:kmax]
        ref_unrounded_c128 = np.fft.ifft(ideal_corr_c128, axis=-1, norm="forward")

        routes, setup_costs, route_errors = build_routes(B, N, kmin, kmax, threads, tmplt_np, strain_np)

        # Warmup and active route triage
        active_routes = {}
        for rname, rinfo in routes.items():
            try:
                for _ in range(3):
                    rinfo["fn"]()
                active_routes[rname] = rinfo
            except Exception as e:
                route_errors[rname] = str(e)

        # Reference route: fftw_legacy_scalar
        ref_scalar_cands = None
        ref_scalar_out = None
        if "fftw_legacy_scalar" in active_routes:
            ref_scalar_out = active_routes["fftw_legacy_scalar"]["out"]()
            ref_t_out = torch.from_numpy(ref_scalar_out)
            ref_t_norms = torch.from_numpy(norms_np)
            ref_t_sig = torch.from_numpy(sig_np)
            ref_scalar_cands = select_tile_candidates(ref_t_out, ref_t_norms, ref_t_sig, v_start, v_end, policy)["candidates"]
            assert 0 < len(ref_scalar_cands["sample_idx"]) <= B * 8, "Reference candidates not bounded and non-empty"

        # Primary parity validation across active routes
        primary_parity = {}
        for rname, rinfo in active_routes.items():
            out_arr = rinfo["out"]()
            max_abs_c128 = float(np.max(np.abs(out_arr.astype(np.complex128) - ref_unrounded_c128)))
            rms_c128 = float(np.sqrt(np.mean(np.abs(out_arr.astype(np.complex128) - ref_unrounded_c128) ** 2)))

            exact_arr_eq = False
            cands_parity = {"all_exact": False, "ids_match": False, "snr_max_diff": None}
            if ref_scalar_out is not None:
                exact_arr_eq = compare_arrays(ref_scalar_out, out_arr)
                t_cands = select_tile_candidates(torch.from_numpy(out_arr), torch.from_numpy(norms_np), torch.from_numpy(sig_np), v_start, v_end, policy)["candidates"]
                cands_parity = compare_candidate_dicts(ref_scalar_cands, t_cands)

            primary_parity[rname] = {
                "exact_array_equal_to_scalar_fftw": exact_arr_eq,
                "max_abs_err_vs_c128": safe_float(max_abs_c128),
                "rms_err_vs_c128": safe_float(rms_c128),
                "candidates_parity": cands_parity,
                "plan_metadata": rinfo["plan_metadata"],
                "setup_cost_s": setup_costs.get(rname, 0.0),
            }

        # Timed loops with randomized paired (route, scope)
        timings_full: Dict[str, List[float]] = {k: [] for k in active_routes}
        timings_fft: Dict[str, List[float]] = {k: [] for k in active_routes}

        for _ in range(iterations):
            for rinfo in active_routes.values():
                rinfo["fn"]()

            tasks = []
            for rk in active_routes:
                tasks.append((rk, "full"))
                tasks.append((rk, "fft"))
            order_rng.shuffle(tasks)

            for rk, scope in tasks:
                rinfo = active_routes[rk]
                if scope == "full":
                    t0 = time.perf_counter()
                    rinfo["fn"]()
                    timings_full[rk].append(time.perf_counter() - t0)
                else:
                    t0 = time.perf_counter()
                    rinfo["fft_fn"]()
                    timings_fft[rk].append(time.perf_counter() - t0)

        # Independent Random Wideband Parity Probe AFTER primary benchmark
        probe_rng = np.random.default_rng(999)
        num_act = kmax - kmin
        rand_scale = 0.05 / np.sqrt(N)
        # Mutate tmplt_np and strain_np IN PLACE so all views reflect the probe inputs
        strain_np.fill(0)
        strain_np[kmin:kmax] = (probe_rng.standard_normal(num_act) + 1j * probe_rng.standard_normal(num_act)) * rand_scale
        tmplt_np.fill(0)
        for b in range(B):
            tmplt_np[b, kmin:kmax] = (probe_rng.standard_normal(num_act) + 1j * probe_rng.standard_normal(num_act)) * rand_scale

        # Recompute unrounded complex128 analytical reference for the wideband probe
        c128_tmplt_probe = tmplt_np.astype(np.complex128)
        c128_strain_probe = strain_np.astype(np.complex128)
        ideal_probe_c128 = np.zeros((B, N), dtype=np.complex128)
        ideal_probe_c128[:, kmin:kmax] = np.conj(c128_tmplt_probe[:, kmin:kmax]) * c128_strain_probe[kmin:kmax]
        ref_probe_c128 = np.fft.ifft(ideal_probe_c128, axis=-1, norm="forward")

        # Run all active routes on the probe data
        probe_outputs = {}
        for rname, rinfo in active_routes.items():
            try:
                rinfo["fn"]()
                probe_outputs[rname] = rinfo["out"]()
            except Exception as e:
                probe_outputs[rname] = None
                route_errors[f"{rname}_probe"] = str(e)

        ref_probe_scalar_out = probe_outputs.get("fftw_legacy_scalar", None)
        wideband_probe_parity = {}
        for rname, out_arr in probe_outputs.items():
            if out_arr is None:
                continue
            max_abs_probe = float(np.max(np.abs(out_arr.astype(np.complex128) - ref_probe_c128)))
            rms_probe = float(np.sqrt(np.mean(np.abs(out_arr.astype(np.complex128) - ref_probe_c128) ** 2)))
            exact_probe_eq = False
            if ref_probe_scalar_out is not None:
                exact_probe_eq = compare_arrays(ref_probe_scalar_out, out_arr)
            wideband_probe_parity[rname] = {
                "exact_array_equal_to_scalar_fftw": exact_probe_eq,
                "max_abs_err_vs_c128": safe_float(max_abs_probe),
                "rms_err_vs_c128": safe_float(rms_probe),
            }

        stats = {}
        for rk in active_routes:
            stats[rk] = {
                "total_p50_ms": float(np.percentile(timings_full[rk], 50) * 1000),
                "total_p95_ms": float(np.percentile(timings_full[rk], 95) * 1000),
                "fft_p50_ms": float(np.percentile(timings_fft[rk], 50) * 1000),
                "fft_p95_ms": float(np.percentile(timings_fft[rk], 95) * 1000),
                "raw_total_s": timings_full[rk],
                "raw_fft_s": timings_fft[rk],
            }

        results.append({
            "shape": sh,
            "primary_parity": primary_parity,
            "wideband_probe_parity": wideband_probe_parity,
            "stats": stats,
            "route_errors": route_errors,
            "valid_for_promotion": False,
            "promotion_reason": "Prototype CPU FFT measurement only; no production CLI changes",
        })

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = float(rss_kb / 1024.0 if sys.platform != "darwin" else rss_kb / (1024.0 * 1024.0))

    receipt = {
        "metadata": {
            "experiment": "cpu_fft_backends",
            "threads_requested": threads,
            "actual_torch_threads": actual_torch_threads,
            "affinity_cpus": affinity_list,
            "threadpool_info": threadpool_info,
            "python_version": sys.version,
            "torch_version": str(torch.__version__),
            "source_sha256": source_sha256,
            "peak_rss_mb": rss_mb,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": results,
    }
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(receipt, f, indent=2)
    return receipt


def main():
    parser = argparse.ArgumentParser(description="CPU FFT backends benchmark")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--output", type=str, default="cpu_fft_benchmark.json")
    parser.add_argument("--shapes", nargs="*", default=None, help="Optional shape name subset")
    args = parser.parse_args()

    receipt = run_benchmark(threads=args.threads, iterations=args.iterations, output_path=args.output, shape_filter=args.shapes)
    print(f"Experiment complete (threads={args.threads}). Output: {args.output}")
    print(f"{'Shape':<16} | {'Backend':<26} | {'Total p50 (ms)':<14} | {'FFT p50 (ms)':<14} | {'Err vs c128':<12} | {'Exact FFTW'}")
    print("-" * 102)
    for r in receipt["results"]:
        sh = r["shape"]["name"]
        for bname, bstats in r["stats"].items():
            par = r["primary_parity"][bname]
            err_str = f"{par['max_abs_err_vs_c128']:.2e}" if par['max_abs_err_vs_c128'] is not None else "None"
            print(f"{sh:<16} | {bname:<26} | {bstats['total_p50_ms']:<14.2f} | {bstats['fft_p50_ms']:<14.2f} | {err_str:<12} | {par['exact_array_equal_to_scalar_fftw']}")


if __name__ == "__main__":
    main()
