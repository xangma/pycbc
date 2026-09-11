"""Experiment: nvmath-python cuFFT load callback (prolog) fusion."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import site
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

bootstrap_error = None
opt_path = os.environ.get("PYCBC_EXPERIMENT_OPTIONAL")
if opt_path:
    try:
        site.addsitedir(opt_path)
        import cuda
        cuda_bindings_loc = str(importlib.metadata.distribution("cuda-bindings").locate_file("cuda"))
        cuda.__path__.insert(0, os.path.join(opt_path, "cuda"))
        cuda.__path__.insert(1, cuda_bindings_loc)
    except Exception as e:
        bootstrap_error = f"{type(e).__name__}: {str(e)}"

import numpy as np
import torch

nvmath_err = None
try:
    import nvmath
    from nvmath.fft import FFT, FFTDirection, DeviceCallable, compile_prolog
except Exception as e:
    nvmath = None
    FFT = None
    FFTDirection = None
    DeviceCallable = None
    compile_prolog = None
    nvmath_err = f"{type(e).__name__}: {str(e)}"

numba_err = None
try:
    import numba
except Exception as e:
    numba = None
    numba_err = f"{type(e).__name__}: {str(e)}"

sys_path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path_root not in sys.path:
    sys.path.insert(0, sys_path_root)

from pycbc.filter.gpu_search.candidates import SelectionPolicy, select_tile_candidates


def safe_float(val: float) -> Optional[float]:
    if math.isnan(val) or math.isinf(val):
        return None
    return float(val)


def make_fixtures(
    B: int,
    N: int,
    device: str,
    seed: int = 42,
    variant: str = "A",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    kmin, kmax = 20, N // 2
    num_act = kmax - kmin
    dev = torch.device(device)

    strain_np = np.zeros(N, dtype=np.complex64)
    if variant == "A":
        strain_np[kmin:kmax] = 1.0 + 0.0j
        shift = 0
        phasor = 1.0 + 0.0j
        target_boost = 1.0
    else:
        # Distinct complex strain and amplitude boost for variant B
        strain_np[kmin:kmax] = (0.6 + 0.8j) * 1.1
        shift = 7
        phasor = 0.70710678 + 0.70710678j
        target_boost = 1.1

    tmplt_np = np.zeros((B, N), dtype=np.complex64)
    norms_np = rng.uniform(0.8, 1.2, size=B).astype(np.float32)
    sigmasqs_np = rng.uniform(10.0, 50.0, size=B).astype(np.float32)

    pulse_times = [3 * N // 8 + shift, N // 2 + shift, 5 * N // 8 + shift]
    freqs = np.arange(kmin, kmax, dtype=np.float64)
    for b in range(B):
        norm_b = float(norms_np[b])
        for idx, t_p in enumerate(pulse_times):
            target_snr = (9.0 + (idx % 2) * 3.0) * target_boost
            amp = target_snr / (norm_b * float(num_act))
            phase = -2.0 * np.pi * freqs * (float(t_p) / float(N))
            # Form desired correlation pulse and divide by conj(strain) so conj(template) * strain == corr_pulse
            corr_pulse = amp * np.exp(1j * phase) * phasor
            active_strain = strain_np[kmin:kmax]
            tmplt_np[b, kmin:kmax] += (np.conj(corr_pulse) / np.conj(active_strain)).astype(np.complex64)

    return (
        torch.as_tensor(tmplt_np, device=dev),
        torch.as_tensor(strain_np, device=dev),
        torch.as_tensor(norms_np, device=dev),
        torch.as_tensor(sigmasqs_np, device=dev),
    )


def compare_candidate_dicts(ref: Dict[str, np.ndarray], test: Dict[str, np.ndarray]) -> Dict[str, Any]:
    keys = ["template_idx", "sample_idx", "snr", "sigmasq"]
    if not all(k in test for k in keys) or not all(k in ref for k in keys):
        return {"all_exact": False, "ids_match": False, "snr_max_diff": None}
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


def run_experiment(iterations: int = 15, shape_filter: Optional[List[str]] = None, output_path: str = "nvmath_callback_benchmark.json") -> Dict[str, Any]:
    if iterations <= 0:
        raise ValueError(f"Iterations must be positive, got {iterations}")

    if not torch.cuda.is_available():
        receipt_blocked = {
            "metadata": {
                "experiment": "nvmath_cufft_callback_trial",
                "status": "blocked",
                "reason": "CUDA is unavailable. nvmath callback trial requires GPU.",
                "bootstrap_error": bootstrap_error,
                "nvmath_error": nvmath_err,
                "numba_error": numba_err,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "results": [],
        }
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(receipt_blocked, f, indent=2)
        return receipt_blocked

    dev = torch.device("cuda")
    curr_stream = torch.cuda.current_stream()
    order_rng = random.Random(42)
    policy = SelectionPolicy(cluster_policy="symmetric", cluster_window=64, snr_threshold=5.5)

    all_shapes = [
        {"name": "N65536_B1", "N": 65536, "B": 1},
        {"name": "N65536_B16", "N": 65536, "B": 16},
        {"name": "N65536_B3_tail", "N": 65536, "B": 3},
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
    for p in ["tools/experiment_nvmath_callback.py", "pycbc/filter/gpu_search/core.py", "pycbc/filter/gpu_search/candidates.py"]:
        fp = os.path.join(sys_path_root, p)
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                source_sha256[p] = hashlib.sha256(f.read()).hexdigest()

    results = []
    for sh in shapes:
        B, N = sh["B"], sh["N"]
        kmin, kmax = 20, N // 2
        v_start, v_end = N // 4, 3 * N // 4

        tmpltA, strainA, norms, sigmasqs = make_fixtures(B, N, "cuda", seed=101, variant="A")
        tmpltB, strainB, _, _ = make_fixtures(B, N, "cuda", seed=101, variant="B")
        assert not torch.equal(strainA, strainB), "Fixture B strain must differ from Fixture A"
        assert not torch.equal(tmpltA, tmpltB), "Fixture B templates must differ from Fixture A"

        # Allocate persistent working tensors
        t_alloc0 = time.perf_counter()
        static_tmplt = tmpltA.clone()
        static_strain = strainA.clone()
        raw_cout = torch.zeros((B, N), device=dev, dtype=torch.complex64)
        raw_out = torch.zeros((B, N), device=dev, dtype=torch.complex64)
        torch.cuda.synchronize()
        ref_alloc_s = time.perf_counter() - t_alloc0

        routes = {}
        setup_timings = {}
        first_exec_timings = {}
        route_errors = {}
        plan_layouts = {}
        cleanup_errors = {}

        # Route 1: Exact standalone PyTorch reference (zero-conjmul-ifft)
        def run_ref_core():
            raw_cout.zero_()
            torch.mul(torch.conj(static_tmplt[:, kmin:kmax]), static_strain[kmin:kmax], out=raw_cout[:, kmin:kmax])
            torch.fft.ifft(raw_cout, n=N, dim=-1, norm="forward", out=raw_out)
            return raw_out

        def run_ref_full():
            out_t = run_ref_core()
            return select_tile_candidates(out_t, norms, sigmasqs, v_start, v_end, policy)["candidates"]

        setup_timings["torch_reference"] = {"alloc_s": ref_alloc_s}
        torch.cuda.synchronize()
        t_fe0 = time.perf_counter()
        _ = run_ref_core()
        torch.cuda.synchronize()
        first_exec_timings["torch_reference"] = time.perf_counter() - t_fe0
        routes["torch_reference"] = {"core": run_ref_core, "full": run_ref_full}

        # Route 2: nvmath plain plan (no callback) on precomputed correlation
        nvmath_plain_plan = None
        if FFT is not None:
            try:
                t_p0 = time.perf_counter()
                nv_plain_in = torch.zeros((B, N), device=dev, dtype=torch.complex64)
                nvmath_plain_plan = FFT(nv_plain_in, axes=(-1,), stream=curr_stream)
                nvmath_plain_plan.plan()
                torch.cuda.synchronize()
                setup_timings["nvmath_plain"] = {"plan_s": time.perf_counter() - t_p0}
                if hasattr(nvmath_plain_plan, "get_input_layout"):
                    try:
                        plan_layouts["nvmath_plain"] = repr(nvmath_plain_plan.get_input_layout())
                    except Exception as e:
                        plan_layouts["nvmath_plain"] = f"error: {e}"

                def run_nvmath_plain_core():
                    nv_plain_in.zero_()
                    torch.mul(torch.conj(static_tmplt[:, kmin:kmax]), static_strain[kmin:kmax], out=nv_plain_in[:, kmin:kmax])
                    return nvmath_plain_plan.execute(direction=FFTDirection.INVERSE, stream=curr_stream)

                def run_nvmath_plain_full():
                    out_t = run_nvmath_plain_core()
                    return select_tile_candidates(out_t, norms, sigmasqs, v_start, v_end, policy)["candidates"]

                torch.cuda.synchronize()
                t_fe0 = time.perf_counter()
                _ = run_nvmath_plain_core()
                torch.cuda.synchronize()
                first_exec_timings["nvmath_plain"] = time.perf_counter() - t_fe0
                routes["nvmath_plain"] = {"core": run_nvmath_plain_core, "full": run_nvmath_plain_full}
            except Exception as e:
                route_errors["nvmath_plain"] = f"{type(e).__name__}: {str(e)}"
        else:
            route_errors["nvmath_plain"] = f"nvmath unavailable: {nvmath_err}"

        # Route 3: nvmath with load callback (prolog) fusion
        nvmath_cb_plan = None
        if FFT is not None and compile_prolog is not None and numba is not None:
            try:
                t_cp0 = time.perf_counter()
                def make_prolog_fn(N_val, kmin_val, kmax_val):
                    def prolog_cb(data_in, offset, user_info, reserved):
                        col = offset % N_val
                        if col >= kmin_val and col < kmax_val:
                            t_val = data_in[offset]
                            s_val = user_info[col]
                            r = t_val.real * s_val.real + t_val.imag * s_val.imag
                            i = t_val.real * s_val.imag - t_val.imag * s_val.real
                            return numba.complex64(complex(r, i))
                        return numba.complex64(0.0 + 0.0j)
                    return prolog_cb

                cb_fn = make_prolog_fn(N, kmin, kmax)
                ltoir = compile_prolog(cb_fn, element_dtype="complex64", user_info_dtype="complex64")
                t_comp_prolog_s = time.perf_counter() - t_cp0

                t_plan0 = time.perf_counter()
                d_callable = DeviceCallable(ltoir=ltoir, data=static_strain.data_ptr())
                nvmath_cb_plan = FFT(static_tmplt, axes=(-1,), stream=curr_stream)
                nvmath_cb_plan.plan(prolog=d_callable)
                torch.cuda.synchronize()
                t_plan_cb_s = time.perf_counter() - t_plan0
                setup_timings["nvmath_callback"] = {"compile_prolog_s": t_comp_prolog_s, "plan_s": t_plan_cb_s}
                if hasattr(nvmath_cb_plan, "get_input_layout"):
                    try:
                        plan_layouts["nvmath_callback"] = repr(nvmath_cb_plan.get_input_layout())
                    except Exception as e:
                        plan_layouts["nvmath_callback"] = f"error: {e}"

                def run_nvmath_cb_core():
                    return nvmath_cb_plan.execute(direction=FFTDirection.INVERSE, stream=curr_stream)

                def run_nvmath_cb_full():
                    out_t = run_nvmath_cb_core()
                    return select_tile_candidates(out_t, norms, sigmasqs, v_start, v_end, policy)["candidates"]

                torch.cuda.synchronize()
                t_fe0 = time.perf_counter()
                _ = run_nvmath_cb_core()
                torch.cuda.synchronize()
                first_exec_timings["nvmath_callback"] = time.perf_counter() - t_fe0
                routes["nvmath_callback"] = {"core": run_nvmath_cb_core, "full": run_nvmath_cb_full}
            except Exception as e:
                route_errors["nvmath_callback"] = f"{type(e).__name__}: {str(e)}"
        else:
            route_errors["nvmath_callback"] = f"nvmath/numba compile_prolog unavailable: nvmath_err={nvmath_err}, numba_err={numba_err}"

        # Warmup (3 iterations)
        static_tmplt.copy_(tmpltA)
        static_strain.copy_(strainA)
        torch.cuda.synchronize()
        active_routes = {}
        for rk, rinfo in routes.items():
            try:
                for _ in range(3):
                    _ = rinfo["core"]()
                torch.cuda.synchronize()
                active_routes[rk] = rinfo
            except Exception as e:
                route_errors[f"{rk}_warmup"] = f"{type(e).__name__}: {str(e)}"

        # Validation on Fixture A (Reference output cloned)
        static_tmplt.copy_(tmpltA)
        static_strain.copy_(strainA)
        torch.cuda.synchronize()
        ref_out_A = active_routes["torch_reference"]["core"]().clone()
        ref_cands_A = select_tile_candidates(ref_out_A, norms, sigmasqs, v_start, v_end, policy)["candidates"]
        assert 0 < len(ref_cands_A["sample_idx"]) <= B * 8, f"Fixture A trigger count out of bounds: {len(ref_cands_A['sample_idx'])}"

        parity_report_A = {}
        for rk, rinfo in active_routes.items():
            out_t = rinfo["core"]()
            torch.cuda.synchronize()
            exact_out = bool(ref_out_A.shape == out_t.shape and ref_out_A.dtype == out_t.dtype and torch.equal(ref_out_A, out_t))
            max_abs_out = float(torch.max(torch.abs(ref_out_A - out_t)).item()) if ref_out_A.shape == out_t.shape else None
            test_cands = select_tile_candidates(out_t, norms, sigmasqs, v_start, v_end, policy)["candidates"]
            c_parity = compare_candidate_dicts(ref_cands_A, test_cands)
            parity_report_A[rk] = {
                "exact_out_tensor": exact_out,
                "max_abs_diff_out": safe_float(max_abs_out) if max_abs_out is not None else None,
                "candidates_parity": c_parity,
            }

        # Validation on Changed Fixture B (Catch stale retained operand buffers)
        static_tmplt.copy_(tmpltB)
        static_strain.copy_(strainB)
        torch.cuda.synchronize()
        ref_out_B = active_routes["torch_reference"]["core"]().clone()
        assert not torch.equal(ref_out_A, ref_out_B), "Reference output B must differ from reference output A"
        ref_cands_B = select_tile_candidates(ref_out_B, norms, sigmasqs, v_start, v_end, policy)["candidates"]
        assert 0 < len(ref_cands_B["sample_idx"]) <= B * 8, f"Fixture B trigger count out of bounds: {len(ref_cands_B['sample_idx'])}"

        stale_operand_checks_B = {}
        for rk, rinfo in active_routes.items():
            out_B = rinfo["core"]()
            torch.cuda.synchronize()
            exact_out_B = bool(ref_out_B.shape == out_B.shape and ref_out_B.dtype == out_B.dtype and torch.equal(ref_out_B, out_B))
            max_abs_B = float(torch.max(torch.abs(ref_out_B - out_B)).item()) if ref_out_B.shape == out_B.shape else None
            cands_B = select_tile_candidates(out_B, norms, sigmasqs, v_start, v_end, policy)["candidates"]
            c_parity_B = compare_candidate_dicts(ref_cands_B, cands_B)
            stale_operand_checks_B[rk] = {
                "exact_out_tensor_B": exact_out_B,
                "max_abs_diff_B": safe_float(max_abs_B) if max_abs_B is not None else None,
                "candidates_parity_B": c_parity_B,
            }

        # Timed benchmark loops: alternating fixtures pool [(tmpltA, strainA), (tmpltB, strainB)]
        timings_core: Dict[str, List[float]] = {k: [] for k in active_routes}
        timings_full: Dict[str, List[float]] = {k: [] for k in active_routes}
        fixtures_pool = [(tmpltA, strainA), (tmpltB, strainB)]

        for it in range(iterations):
            cur_t, cur_s = fixtures_pool[it % 2]
            tasks = []
            for rk in active_routes:
                tasks.append((rk, "core"))
                tasks.append((rk, "full"))
            order_rng.shuffle(tasks)

            for rk, scope in tasks:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                static_tmplt.copy_(cur_t)
                static_strain.copy_(cur_s)
                if scope == "core":
                    _ = active_routes[rk]["core"]()
                else:
                    _ = active_routes[rk]["full"]()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                if scope == "core":
                    timings_core[rk].append(elapsed)
                else:
                    timings_full[rk].append(elapsed)

        # Independent Random Wideband Complex Input Parity Probe AFTER primary timings
        probe_rng = np.random.default_rng(777)
        num_act = kmax - kmin
        rand_scale = 0.05 / np.sqrt(N)
        probe_tmplt_np = np.zeros((B, N), dtype=np.complex64)
        probe_strain_np = np.zeros(N, dtype=np.complex64)
        probe_strain_np[kmin:kmax] = (probe_rng.standard_normal(num_act) + 1j * probe_rng.standard_normal(num_act)) * rand_scale
        for b in range(B):
            probe_tmplt_np[b, kmin:kmax] = (probe_rng.standard_normal(num_act) + 1j * probe_rng.standard_normal(num_act)) * rand_scale

        # Exact unrounded complex128 NumPy IFFT analytical reference (inputs cast before conjmul)
        c128_t = probe_tmplt_np.astype(np.complex128)
        c128_s = probe_strain_np.astype(np.complex128)
        ideal_corr_c128 = np.zeros((B, N), dtype=np.complex128)
        ideal_corr_c128[:, kmin:kmax] = np.conj(c128_t[:, kmin:kmax]) * c128_s[kmin:kmax]
        ref_probe_c128 = np.fft.ifft(ideal_corr_c128, axis=-1, norm="forward")

        # Copy probe data to device static tensors
        static_tmplt.copy_(torch.from_numpy(probe_tmplt_np))
        static_strain.copy_(torch.from_numpy(probe_strain_np))
        torch.cuda.synchronize()

        ref_probe_torch = active_routes["torch_reference"]["core"]().clone()
        wideband_probe_parity = {}
        for rk, rinfo in active_routes.items():
            out_probe = rinfo["core"]()
            torch.cuda.synchronize()
            exact_vs_torch = bool(
                ref_probe_torch.shape == out_probe.shape and
                ref_probe_torch.dtype == out_probe.dtype and
                torch.equal(ref_probe_torch, out_probe)
            )
            out_probe_np = out_probe.detach().cpu().numpy().astype(np.complex128)
            max_abs_c128 = float(np.max(np.abs(out_probe_np - ref_probe_c128)))
            rms_c128 = float(np.sqrt(np.mean(np.abs(out_probe_np - ref_probe_c128) ** 2)))
            wideband_probe_parity[rk] = {
                "exact_array_equal_to_torch_ref": exact_vs_torch,
                "max_abs_err_vs_c128": safe_float(max_abs_c128),
                "rms_err_vs_c128": safe_float(rms_c128),
            }

        # Plan resource cleanup
        if nvmath_plain_plan is not None and hasattr(nvmath_plain_plan, "free"):
            try:
                nvmath_plain_plan.free()
            except Exception as e:
                cleanup_errors["nvmath_plain"] = str(e)
        if nvmath_cb_plan is not None and hasattr(nvmath_cb_plan, "free"):
            try:
                nvmath_cb_plan.free()
            except Exception as e:
                cleanup_errors["nvmath_callback"] = str(e)

        stats = {}
        for rk in active_routes:
            stats[rk] = {
                "core_p50_ms": float(np.percentile(timings_core[rk], 50) * 1000),
                "core_p95_ms": float(np.percentile(timings_core[rk], 95) * 1000),
                "full_p50_ms": float(np.percentile(timings_full[rk], 50) * 1000),
                "full_p95_ms": float(np.percentile(timings_full[rk], 95) * 1000),
                "raw_core_s": timings_core[rk],
                "raw_full_s": timings_full[rk],
            }

        results.append({
            "shape": sh,
            "setup_timings": setup_timings,
            "first_exec_timings": first_exec_timings,
            "plan_layouts": plan_layouts,
            "parity_fixture_A": parity_report_A,
            "stale_operand_checks_B": stale_operand_checks_B,
            "wideband_probe_parity": wideband_probe_parity,
            "stats": stats,
            "route_errors": route_errors,
            "cleanup_errors": cleanup_errors,
            "valid_for_promotion": False,
            "promotion_reason": "Experimental trial for cuFFT callback load fusion; measurement only",
        })

    peak_vram_alloc = torch.cuda.max_memory_allocated(dev)
    peak_vram_res = torch.cuda.max_memory_reserved(dev)

    receipt = {
        "metadata": {
            "experiment": "nvmath_cufft_callback_trial",
            "status": "prototype_only",
            "device": str(dev),
            "gpu_name": torch.cuda.get_device_name(dev),
            "torch_version": str(torch.__version__),
            "nvmath_version": getattr(nvmath, "__version__", None),
            "numba_version": getattr(numba, "__version__", None),
            "bootstrap_error": bootstrap_error,
            "peak_vram_allocated_bytes": peak_vram_alloc,
            "peak_vram_reserved_bytes": peak_vram_res,
            "source_sha256": source_sha256,
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
    parser = argparse.ArgumentParser(description="nvmath cuFFT load callback fusion trial")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--shapes", nargs="*", default=None)
    parser.add_argument("--output", type=str, default="nvmath_callback_benchmark.json")
    args = parser.parse_args()

    receipt = run_experiment(iterations=args.iterations, shape_filter=args.shapes, output_path=args.output)
    print(f"Trial complete. Output: {args.output}")
    print(f"{'Shape':<18} | {'Route':<18} | {'Core p50 (ms)':<14} | {'Full p50 (ms)':<14} | {'Exact Tensor'}")
    print("-" * 80)
    for r in receipt["results"]:
        sh = r["shape"]["name"]
        for rk, rstats in r["stats"].items():
            par = r["parity_fixture_A"].get(rk, {})
            ex = par.get("exact_out_tensor", False)
            print(f"{sh:<18} | {rk:<18} | {rstats['core_p50_ms']:<14.2f} | {rstats['full_p50_ms']:<14.2f} | {ex}")


if __name__ == "__main__":
    main()
