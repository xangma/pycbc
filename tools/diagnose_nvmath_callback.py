"""Numerical diagnosis for nvmath FFT load callback (prolog).

Isolates numerical discrepancies between PyTorch IFFT, plain nvmath FFT,
passthrough callback, fused complex-multiply callback, and optional
IEEE-rounded multiply callback across small (N=65536) and large (N=2097152)
FFT transform lengths.

Usage:
  PYTHONPATH=. python tools/diagnose_nvmath_callback.py \
    --output /path/to/diagnosis_receipt.json
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from tools.experiment_nvmath_callback import (
    FFT,
    DeviceCallable,
    FFTDirection,
    bootstrap_error,
    compile_prolog,
    compare_candidate_dicts,
    make_fixtures,
    numba,
    numba_err,
    nvmath,
    nvmath_err,
)
from pycbc.filter.gpu_search.candidates import (
    SelectionPolicy,
    select_tile_candidates,
)


def safe_float(val: Optional[float]) -> Optional[float]:
    if val is None or math.isnan(val) or math.isinf(val):
        return None
    return float(val)


def make_wideband_fixtures(
    B: int, N: int, kmin: int, kmax: int, device: str, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    scale = 1e-4

    tmplt_np = np.zeros((B, N), dtype=np.complex64)
    strain_np = np.zeros(N, dtype=np.complex64)
    tmplt_np[:, kmin:kmax] = (
        rng.standard_normal((B, kmax - kmin))
        + 1j * rng.standard_normal((B, kmax - kmin))
    ).astype(np.complex64) * scale
    strain_np[kmin:kmax] = (
        rng.standard_normal(kmax - kmin)
        + 1j * rng.standard_normal(kmax - kmin)
    ).astype(np.complex64) * scale

    norms_np = np.ones(B, dtype=np.float32)
    sigmasqs_np = np.ones(B, dtype=np.float32)

    return (
        torch.as_tensor(tmplt_np, device=dev),
        torch.as_tensor(strain_np, device=dev),
        torch.as_tensor(norms_np, device=dev),
        torch.as_tensor(sigmasqs_np, device=dev),
    )


def build_prolog_fused(N_val: int, kmin_val: int, kmax_val: int):
    def prolog_fused(data_in, offset, user_info, reserved):
        col = offset % N_val
        if col >= kmin_val and col < kmax_val:
            t_val = data_in[offset]
            s_val = user_info[col]
            r = t_val.real * s_val.real + t_val.imag * s_val.imag
            i = t_val.real * s_val.imag - t_val.imag * s_val.real
            return numba.complex64(complex(r, i))
        return numba.complex64(0.0 + 0.0j)

    return prolog_fused


def build_prolog_passthrough():
    def prolog_passthrough(data_in, offset, user_info, reserved):
        return data_in[offset]

    return prolog_passthrough


def build_prolog_rounded(N_val: int, kmin_val: int, kmax_val: int):
    try:
        from numba.cuda import libdevice

        fmul = libdevice.fmul_rn
        fadd = libdevice.fadd_rn
        fsub = libdevice.fsub_rn

        def prolog_rounded(data_in, offset, user_info, reserved):
            col = offset % N_val
            if col >= kmin_val and col < kmax_val:
                t_val = data_in[offset]
                s_val = user_info[col]
                rr = fmul(t_val.real, s_val.real)
                ii = fmul(t_val.imag, s_val.imag)
                r = fadd(rr, ii)
                ri = fmul(t_val.real, s_val.imag)
                ir = fmul(t_val.imag, s_val.real)
                i = fsub(ri, ir)
                return numba.complex64(complex(r, i))
            return numba.complex64(0.0 + 0.0j)

        return prolog_rounded
    except Exception:
        return None


def time_execution_median(
    fn, stream: torch.cuda.Stream, iterations: int = 5
) -> float:
    for _ in range(3):
        fn()
    stream.synchronize()

    times = []
    for _ in range(iterations):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record(stream)
        fn()
        end_ev.record(stream)
        end_ev.synchronize()
        times.append(start_ev.elapsed_time(end_ev) * 1e-3)
    return float(np.median(times))


def diagnose_shape(
    B: int,
    N: int,
    kmin: int,
    kmax: int,
    fixture_path: Optional[str] = None,
    iterations: int = 5,
    policy: Optional[SelectionPolicy] = None,
) -> Dict[str, Any]:
    dev = torch.device("cuda")
    curr_stream = torch.cuda.current_stream()
    v_start, v_end = N // 4, 3 * N // 4
    if policy is None:
        policy = SelectionPolicy(
            cluster_policy="symmetric", cluster_window=64, snr_threshold=5.5
        )

    t_setup0 = time.perf_counter()
    static_tmplt = torch.zeros((B, N), device=dev, dtype=torch.complex64)
    static_strain = torch.zeros(N, device=dev, dtype=torch.complex64)
    raw_cout = torch.zeros((B, N), device=dev, dtype=torch.complex64)
    raw_out = torch.zeros((B, N), device=dev, dtype=torch.complex64)
    nv_plain_in = torch.zeros((B, N), device=dev, dtype=torch.complex64)

    plan_plain = None
    plan_pass = None
    plan_fused = None
    plan_rounded = None
    setup_errors: Dict[str, str] = {}
    setup_timings: Dict[str, float] = {}
    cleanup_errors: Dict[str, str] = {}

    try:
        # 1. Plain Plan Setup
        try:
            t0 = time.perf_counter()
            plan_plain = FFT(nv_plain_in, axes=(-1,), stream=curr_stream)
            plan_plain.plan()
            curr_stream.synchronize()
            setup_timings["plain_plan_s"] = time.perf_counter() - t0
        except Exception as e:
            setup_errors["nvmath_plain"] = f"{type(e).__name__}: {str(e)}"

        # 2. Passthrough Plan Setup
        try:
            t0 = time.perf_counter()
            fn_pass = build_prolog_passthrough()
            lto_pass = compile_prolog(
                fn_pass, element_dtype="complex64", user_info_dtype="complex64"
            )
            d_pass = DeviceCallable(
                ltoir=lto_pass, data=static_strain.data_ptr()
            )
            plan_pass = FFT(nv_plain_in, axes=(-1,), stream=curr_stream)
            plan_pass.plan(prolog=d_pass)
            curr_stream.synchronize()
            setup_timings["pass_plan_s"] = time.perf_counter() - t0
        except Exception as e:
            setup_errors["nvmath_passthrough"] = (
                f"{type(e).__name__}: {str(e)}"
            )

        # 3. Fused Plan Setup (using verified kmin/kmax)
        try:
            t0 = time.perf_counter()
            fn_fused = build_prolog_fused(N, kmin, kmax)
            lto_fused = compile_prolog(
                fn_fused,
                element_dtype="complex64",
                user_info_dtype="complex64",
            )
            d_fused = DeviceCallable(
                ltoir=lto_fused, data=static_strain.data_ptr()
            )
            plan_fused = FFT(static_tmplt, axes=(-1,), stream=curr_stream)
            plan_fused.plan(prolog=d_fused)
            curr_stream.synchronize()
            setup_timings["fused_plan_s"] = time.perf_counter() - t0
        except Exception as e:
            setup_errors["nvmath_fused"] = f"{type(e).__name__}: {str(e)}"

        # 4. Rounded Plan Setup (using verified kmin/kmax)
        fn_rnd = build_prolog_rounded(N, kmin, kmax)
        if fn_rnd is not None:
            try:
                t0 = time.perf_counter()
                lto_rnd = compile_prolog(
                    fn_rnd,
                    element_dtype="complex64",
                    user_info_dtype="complex64",
                )
                d_rnd = DeviceCallable(
                    ltoir=lto_rnd, data=static_strain.data_ptr()
                )
                plan_rounded = FFT(
                    static_tmplt, axes=(-1,), stream=curr_stream
                )
                plan_rounded.plan(prolog=d_rnd)
                curr_stream.synchronize()
                setup_timings["rounded_plan_s"] = time.perf_counter() - t0
            except Exception as e:
                setup_errors["nvmath_rounded"] = (
                    f"{type(e).__name__}: {str(e)}"
                )
        else:
            setup_errors["nvmath_rounded"] = (
                "libdevice rounded intrinsics unavailable"
            )

        setup_s = time.perf_counter() - t_setup0

        # Execution closures
        def run_torch_core():
            raw_cout.zero_()
            torch.mul(
                torch.conj(static_tmplt[:, kmin:kmax]),
                static_strain[kmin:kmax],
                out=raw_cout[:, kmin:kmax],
            )
            torch.fft.ifft(raw_cout, n=N, dim=-1, norm="forward", out=raw_out)
            return raw_out

        def run_plain_core():
            nv_plain_in.zero_()
            torch.mul(
                torch.conj(static_tmplt[:, kmin:kmax]),
                static_strain[kmin:kmax],
                out=nv_plain_in[:, kmin:kmax],
            )
            return plan_plain.execute(
                direction=FFTDirection.INVERSE, stream=curr_stream
            )

        def run_pass_core():
            nv_plain_in.zero_()
            torch.mul(
                torch.conj(static_tmplt[:, kmin:kmax]),
                static_strain[kmin:kmax],
                out=nv_plain_in[:, kmin:kmax],
            )
            return plan_pass.execute(
                direction=FFTDirection.INVERSE, stream=curr_stream
            )

        def run_fused_core():
            return plan_fused.execute(
                direction=FFTDirection.INVERSE, stream=curr_stream
            )

        def run_rounded_core():
            return plan_rounded.execute(
                direction=FFTDirection.INVERSE, stream=curr_stream
            )

        # Prepare Fixtures
        fixture_variants: List[
            Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        if fixture_path is not None:
            with np.load(fixture_path) as npz:
                tmps_raw = npz["templates"]
                strain_raw = npz["strain"]
                norms_raw = npz["norms"]
                sigmasqs_raw = npz["sigmasqs"]
            flen_npz = tmps_raw.shape[1]
            tmps_pad = np.zeros((B, N), dtype=np.complex64)
            strain_pad = np.zeros(N, dtype=np.complex64)
            tmps_pad[:, : min(N, flen_npz)] = tmps_raw[:B, : min(N, flen_npz)]
            strain_pad[: min(N, flen_npz)] = strain_raw[: min(N, flen_npz)]
            fixture_variants.append(
                (
                    "real_npz",
                    torch.as_tensor(tmps_pad, device=dev),
                    torch.as_tensor(strain_pad, device=dev),
                    torch.as_tensor(norms_raw[:B], device=dev),
                    torch.as_tensor(sigmasqs_raw[:B], device=dev),
                )
            )
        else:
            tA, sA, nA, sigA = make_fixtures(
                B, N, "cuda", seed=101, variant="A"
            )
            tB, sB, nB, sigB = make_fixtures(
                B, N, "cuda", seed=101, variant="B"
            )
            tW, sW, nW, sigW = make_wideband_fixtures(
                B, N, kmin, kmax, "cuda", seed=42
            )
            fixture_variants = [
                ("pulse_A", tA, sA, nA, sigA),
                ("pulse_B", tB, sB, nB, sigB),
                ("wideband", tW, sW, nW, sigW),
            ]

        fixture_diagnostics = []
        prev_ref_out = None

        for var_name, tmplt_in, strain_in, norms, sigmasqs in fixture_variants:
            static_tmplt.copy_(tmplt_in)
            static_strain.copy_(strain_in)
            curr_stream.synchronize()

            out_torch = run_torch_core()
            curr_stream.synchronize()
            assert (
                torch.linalg.norm(out_torch).item() > 0.0
            ), f"{var_name} torch IFFT output is zero"
            if prev_ref_out is not None and var_name == "pulse_B":
                assert not torch.equal(
                    prev_ref_out, out_torch
                ), "Pulse B must differ from Pulse A"
            prev_ref_out = out_torch.clone()

            cands_torch = select_tile_candidates(
                out_torch, norms, sigmasqs, v_start, v_end, policy
            )["candidates"]
            if var_name in ("pulse_A", "pulse_B", "real_npz"):
                assert (
                    len(cands_torch["sample_idx"]) > 0
                ), f"Expected populated candidates for {var_name}"

            var_diag: Dict[str, Any] = {
                "variant": var_name,
                "candidate_count_ref": int(len(cands_torch["sample_idx"])),
                "routes": {},
            }

            route_callables = [("torch", run_torch_core)]
            if plan_plain is not None and "nvmath_plain" not in setup_errors:
                route_callables.append(("plain", run_plain_core))
            if (
                plan_pass is not None
                and "nvmath_passthrough" not in setup_errors
            ):
                route_callables.append(("passthrough", run_pass_core))
            if plan_fused is not None and "nvmath_fused" not in setup_errors:
                route_callables.append(("fused", run_fused_core))
            if (
                plan_rounded is not None
                and "nvmath_rounded" not in setup_errors
            ):
                route_callables.append(("rounded", run_rounded_core))

            route_outputs: Dict[str, torch.Tensor] = {
                "torch": out_torch.clone()
            }
            route_cands: Dict[str, Dict[str, np.ndarray]] = {
                "torch": cands_torch
            }

            for r_name, r_fn in route_callables:
                try:
                    out_r = r_fn()
                    curr_stream.synchronize()
                    assert (
                        torch.linalg.norm(out_r).item() > 0.0
                    ), f"{r_name} output is zero"
                    out_r_clone = out_r.clone()
                    route_outputs[r_name] = out_r_clone
                    cands_r = select_tile_candidates(
                        out_r_clone, norms, sigmasqs, v_start, v_end, policy
                    )["candidates"]
                    route_cands[r_name] = cands_r

                    diff = torch.abs(out_torch - out_r_clone)
                    max_abs = float(torch.max(diff).item())
                    ref_l2 = float(torch.linalg.norm(out_torch).item())
                    rel_l2 = float(
                        torch.linalg.norm(diff).item() / max(ref_l2, 1e-12)
                    )
                    exact_eq = bool(
                        out_torch.shape == out_r_clone.shape
                        and out_torch.dtype == out_r_clone.dtype
                        and torch.equal(out_torch, out_r_clone)
                    )
                    c_comp = compare_candidate_dicts(cands_torch, cands_r)
                    dur_s = time_execution_median(
                        r_fn, curr_stream, iterations=iterations
                    )

                    var_diag["routes"][r_name] = {
                        "status": "ok",
                        "exact_equal_ref": exact_eq,
                        "max_abs_diff_ref": safe_float(max_abs),
                        "rel_l2_diff_ref": safe_float(rel_l2),
                        "candidates": c_comp,
                        "candidate_count": int(len(cands_r["sample_idx"])),
                        "median_core_s": dur_s,
                    }
                except Exception as e:
                    var_diag["routes"][r_name] = {
                        "status": "error",
                        "error": f"{type(e).__name__}: {str(e)}",
                    }

            # Compare plain vs passthrough
            if "plain" in route_outputs and "passthrough" in route_outputs:
                out_pl = route_outputs["plain"]
                out_ps = route_outputs["passthrough"]
                diff_pp = torch.abs(out_pl - out_ps)
                max_pp = float(torch.max(diff_pp).item())
                rel_pp = float(
                    torch.linalg.norm(diff_pp).item()
                    / max(float(torch.linalg.norm(out_pl).item()), 1e-12)
                )
                var_diag["plain_vs_passthrough"] = {
                    "exact_equal": bool(torch.equal(out_pl, out_ps)),
                    "max_abs_diff": safe_float(max_pp),
                    "rel_l2_diff": safe_float(rel_pp),
                    "candidates": compare_candidate_dicts(
                        route_cands["plain"], route_cands["passthrough"]
                    ),
                }

            # Compare fused vs rounded
            if "fused" in route_outputs and "rounded" in route_outputs:
                out_fu = route_outputs["fused"]
                out_ro = route_outputs["rounded"]
                diff_fr = torch.abs(out_fu - out_ro)
                max_fr = float(torch.max(diff_fr).item())
                rel_fr = float(
                    torch.linalg.norm(diff_fr).item()
                    / max(float(torch.linalg.norm(out_fu).item()), 1e-12)
                )
                var_diag["fused_vs_rounded"] = {
                    "exact_equal": bool(torch.equal(out_fu, out_ro)),
                    "max_abs_diff": safe_float(max_fr),
                    "rel_l2_diff": safe_float(rel_fr),
                    "candidates": compare_candidate_dicts(
                        route_cands["fused"], route_cands["rounded"]
                    ),
                }

            fixture_diagnostics.append(var_diag)

        return {
            "B": B,
            "N": N,
            "kmin": kmin,
            "kmax": kmax,
            "setup_s": safe_float(setup_s),
            "setup_timings": setup_timings,
            "setup_errors": setup_errors,
            "cleanup_errors": cleanup_errors,
            "fixture_diagnostics": fixture_diagnostics,
        }

    finally:
        for p_name, plan_obj in [
            ("plain", plan_plain),
            ("passthrough", plan_pass),
            ("fused", plan_fused),
            ("rounded", plan_rounded),
        ]:
            if plan_obj is not None:
                try:
                    plan_obj.free()
                except Exception as e:
                    cleanup_errors[p_name] = f"{type(e).__name__}: {str(e)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=None, help="Path to output diagnosis receipt JSON"
    )
    parser.add_argument(
        "--fixture",
        default=None,
        help="Optional path to real-waveform NPZ fixture",
    )
    parser.add_argument(
        "--shapes", default=None, help="Comma-separated shape names filter"
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Median timing iterations"
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        print(
            f"--iterations must be positive, got {args.iterations}",
            file=sys.stderr,
        )
        return 1

    if not torch.cuda.is_available() or FFT is None:
        receipt_blocked = {
            "status": "blocked",
            "reason": "CUDA or nvmath unavailable",
            "bootstrap_error": bootstrap_error,
            "nvmath_error": nvmath_err,
            "numba_error": numba_err,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if args.output:
            os.makedirs(
                os.path.dirname(os.path.abspath(args.output)), exist_ok=True
            )
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(receipt_blocked, f, indent=2)
        print(json.dumps(receipt_blocked, indent=2))
        return 1

    source_hashes = {}
    sys_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for p in [
        "tools/experiment_nvmath_callback.py",
        "tools/diagnose_nvmath_callback.py",
        "pycbc/filter/gpu_search/candidates.py",
    ]:
        fp = os.path.join(sys_root, p)
        if os.path.exists(fp):
            source_hashes[p] = hashlib.sha256(
                Path(fp).read_bytes()
            ).hexdigest()

    all_shapes = [
        {"name": "N65536_B1", "N": 65536, "B": 1},
        {"name": "N65536_B3", "N": 65536, "B": 3},
        {"name": "N65536_B16", "N": 65536, "B": 16},
        {"name": "N2097152_B1", "N": 2097152, "B": 1},
        {"name": "N2097152_B4", "N": 2097152, "B": 4},
    ]

    if args.fixture:
        if not os.path.exists(args.fixture):
            print(f"Fixture path not found: {args.fixture}", file=sys.stderr)
            return 1
        with np.load(args.fixture) as npz:
            req_keys = [
                "templates",
                "strain",
                "norms",
                "sigmasqs",
                "delta_f",
                "kmin",
                "kmax",
            ]
            for rk in req_keys:
                if rk not in npz:
                    print(
                        f"Missing required NPZ key '{rk}' in {args.fixture}",
                        file=sys.stderr,
                    )
                    return 1
            tmps = npz["templates"]
            strain = npz["strain"]
            norms = npz["norms"]
            sigmasqs = npz["sigmasqs"]
            kmin = int(npz["kmin"])
            kmax = int(npz["kmax"])

            if tmps.dtype != np.complex64 or tmps.ndim != 2:
                print(
                    f"Bad templates: {tmps.dtype}, shape={tmps.shape}",
                    file=sys.stderr,
                )
                return 1
            if strain.dtype != np.complex64 or strain.ndim != 1:
                print(
                    f"Bad strain: {strain.dtype}, shape={strain.shape}",
                    file=sys.stderr,
                )
                return 1
            if (
                norms.dtype != np.float32
                or norms.ndim != 1
                or len(norms) != tmps.shape[0]
            ):
                print("norms must be 1D float32 of length B", file=sys.stderr)
                return 1
            if (
                sigmasqs.dtype != np.float32
                or sigmasqs.ndim != 1
                or len(sigmasqs) != tmps.shape[0]
            ):
                print(
                    "sigmasqs must be 1D float32 of length B", file=sys.stderr
                )
                return 1

            df = npz["delta_f"]
            if (
                df.ndim != 0
                or not np.isfinite(df)
                or float(df) <= 0
                or any(
                    not np.all(np.isfinite(a))
                    for a in (tmps, strain, norms, sigmasqs)
                )
            ):
                raise ValueError("Fixture arrays and delta_f must be finite")
            if tmps.shape[0] == 0 or tmps.shape[1] < 2:
                raise ValueError("Fixture bank must be nonempty")
            B_npz = tmps.shape[0]
            flen_npz = tmps.shape[1]
            if len(strain) != flen_npz:
                print(
                    "strain length does not match templates filter length",
                    file=sys.stderr,
                )
                return 1
            N_npz = 2 * (flen_npz - 1)
            if not (0 <= kmin < kmax <= flen_npz):
                print(
                    f"Invalid cutoffs {kmin}:{kmax}; filter length {flen_npz}",
                    file=sys.stderr,
                )
                return 1

        shapes_to_run = [
            {
                "name": f"N{N_npz}_B{B_npz}_npz",
                "N": N_npz,
                "B": B_npz,
                "kmin": kmin,
                "kmax": kmax,
            }
        ]
    elif args.shapes:
        names = {x.strip() for x in args.shapes.split(",")}
        shapes_to_run = [s for s in all_shapes if s["name"] in names]
        if names - {s["name"] for s in all_shapes} or not shapes_to_run:
            print(
                f"Unknown or empty shape selection: {args.shapes}",
                file=sys.stderr,
            )
            return 1
        for s in shapes_to_run:
            s["kmin"] = 20
            s["kmax"] = s["N"] // 2
    else:
        shapes_to_run = []
        for s in all_shapes:
            s_copy = dict(s)
            s_copy["kmin"] = 20
            s_copy["kmax"] = s["N"] // 2
            shapes_to_run.append(s_copy)

    t_all0 = time.perf_counter()
    shape_results = []
    has_failures = False

    for sh in shapes_to_run:
        print(
            f"Diagnosing shape {sh['name']} (B={sh['B']}, N={sh['N']})...",
            file=sys.stderr,
        )
        res = diagnose_shape(
            B=sh["B"],
            N=sh["N"],
            kmin=sh["kmin"],
            kmax=sh["kmax"],
            fixture_path=args.fixture,
            iterations=args.iterations,
        )
        shape_results.append(res)
        if res["cleanup_errors"]:
            has_failures = True
        for fix in res["fixture_diagnostics"]:
            for req_route in ["torch", "plain", "passthrough", "fused"]:
                r_st = fix["routes"].get(req_route, {}).get("status")
                if r_st != "ok":
                    has_failures = True

    total_time_s = time.perf_counter() - t_all0
    receipt = {
        "status": "completed" if not has_failures else "failed",
        "metadata": {
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "nvmath_version": getattr(nvmath, "__version__", "unknown"),
            "cuda_version": torch.version.cuda,
            "source_sha256": source_hashes,
            "total_time_s": safe_float(total_time_s),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "shapes": shape_results,
    }

    if args.output:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
        print(f"Diagnostic receipt written to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(receipt, indent=2))

    return 0 if not has_failures else 1


if __name__ == "__main__":
    sys.exit(main())
