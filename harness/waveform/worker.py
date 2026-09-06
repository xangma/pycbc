#!/usr/bin/env python3
"""Isolated, public-API TaylorF2 timing and parity worker; no source edits."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

SHA = "607bce53ead14f12af32552a5b2441d3bc667267"
ROUTES = ("standard-cpu", "torch-cpu-scalar", "torch-cpu-batch",
          "torch-cuda-scalar", "torch-cuda-batch", "torch-cuda-triton-batch")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "PYCBC_NUM_THREADS")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    """Keep a failed numerical run serializable without inventing finite data."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--route", choices=ROUTES, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--replicate", type=int, default=1)
    p.add_argument("--precision", choices=("double", "single"), default="double")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--sample-ms", type=float, default=50)
    p.add_argument("--max-inner", type=int, default=64)
    p.add_argument("--expected-sha", default=SHA)
    a = p.parse_args()
    if min(a.batch, a.threads, a.replicate, a.samples, a.max_inner) < 1 or a.sample_ms <= 0:
        p.error("counts and sample-ms must be positive")
    return a


def parameters(batch):
    # A deterministic common grid: each route gets exactly the same row list.
    return [dict(mass1=1.4 + 0.05 * (i % 8), mass2=1.3 - 0.02 * (i % 8),
                 spin1z=0.02, spin2z=-0.01, distance=100.0, inclination=0.4,
                 coa_phase=0.2, delta_f=0.25, f_lower=20.0, f_final=1024.0,
                 f_ref=30.0) for i in range(batch)]


def run(a, r):
    root = a.root.resolve()
    r["source"] = {"root": str(root), "sha": git(root, "rev-parse", "HEAD"),
                   "status_porcelain": git(root, "status", "--porcelain"),
                   "tree": git(root, "rev-parse", "HEAD^{tree}")}
    r["source"]["clean"] = not r["source"]["status_porcelain"]
    if r["source"]["sha"] != a.expected_sha or not r["source"]["clean"]:
        raise RuntimeError("source must be clean and match --expected-sha")
    r["source"]["waveform_source_sha256"] = {
        name: digest(root / name) for name in (
            "pycbc/waveform/taylorf2_torch.py", "pycbc/waveform/waveform.py",
            "pycbc/waveform/torch_waveform_registry.py")}
    r["capabilities"] = {
        "public_cpu_cuda_precision": "double only, chosen internally",
        "public_taylorf2_triton": False,
        "basis": "Exact SHA source: CPU/CUDA complex128; no Triton TaylorF2 entry point",
    }
    r["dispatch"] = {"requested": a.route, "actual": None,
                     "triton_requested": "triton" in a.route,
                     "triton_actual": False, "probe_counts": None}
    r["parameters"] = parameters(a.batch)
    r["workload"] = {"approximant": "TaylorF2", "nominal_bins": 4097,
                     "input_policy": "Python host scalars/lists; conversion included in calls",
                     "output_policy": "device output retained; host copy excluded from timing",
                     "autograd": False}
    for package in ("numpy", "torch", "lalsuite", "scipy", "triton"):
        try:
            r["runtime"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            r["runtime"][package] = None
    # Capability rows are cheap and do not launch CUDA or import scientific packages.
    if "triton" in a.route or a.precision == "single":
        r["status"] = "unsupported"
        r["reason"] = ("No public TaylorF2 Triton route exists at this SHA"
                       if "triton" in a.route else
                       "Public TaylorF2 CPU/CUDA outputs are complex128; no single precision selector")
        return

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    for key in THREAD_VARS:
        os.environ[key] = str(a.threads)
    for key in ("PYCBC_TORCH_NATIVE", "PYCBC_TORCH_NATIVE_PORTS"):
        os.environ.pop(key, None)
    os.environ["PYCBC_TAYLORF2_NATIVE"] = "0" if a.route == "standard-cpu" else "1"
    r["environment"] = {k: v for k, v in os.environ.items()
                        if k in THREAD_VARS or k.startswith("PYCBC_") or
                        k in ("CUDA_VISIBLE_DEVICES", "MKL_THREADING_LAYER")}
    started = time.perf_counter()
    import numpy as np
    import pycbc
    from pycbc import scheme
    from pycbc.waveform import get_fd_waveform, get_fd_waveform_batch
    import pycbc.waveform.taylorf2_torch as native
    import lalsimulation
    import torch
    r["runtime"]["import_seconds"] = time.perf_counter() - started
    r["runtime"]["module_origins"] = {
        "pycbc": str(Path(pycbc.__file__).resolve()),
        "taylorf2_torch": str(Path(native.__file__).resolve()),
        "torch": str(Path(torch.__file__).resolve()),
        "numpy": str(Path(np.__file__).resolve()),
        "lalsimulation": str(Path(lalsimulation.__file__).resolve()),
    }
    for name in ("pycbc", "taylorf2_torch"):
        Path(r["runtime"]["module_origins"][name]).relative_to(root)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    r["runtime"].update(torch_threads=torch.get_num_threads(),
                         torch_interop_threads=torch.get_num_interop_threads(),
                         torch_cuda_build=torch.version.cuda)
    cuda = torch.cuda.is_available()
    r["capabilities"]["cuda_available"] = cuda
    is_cuda = "cuda" in a.route
    if is_cuda and not cuda:
        r["status"], r["reason"] = "unsupported", "torch.cuda.is_available() is false"
        return
    device = "cuda:0" if is_cuda else "cpu"
    if is_cuda:
        props = torch.cuda.get_device_properties(0)
        r["runtime"]["cuda_device"] = {
            "index": 0, "name": props.name, "memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor]}
    context = (scheme.CPUScheme(a.threads) if a.route == "standard-cpu" else
               scheme.TorchScheme(device, num_threads=a.threads))
    rows = r["parameters"]
    batch_params = {key: (values[0] if key == "delta_f" else values)
                    for key in rows[0] for values in [[row[key] for row in rows]]}
    is_batch = a.route.endswith("batch")

    def scalar_call():
        return [get_fd_waveform(approximant="TaylorF2", **row) for row in rows]

    def operation():
        return (get_fd_waveform_batch("TaylorF2", **batch_params)
                if is_batch else scalar_call())

    def sync():
        if is_cuda:
            torch.cuda.synchronize(0)

    def measure(count):
        sync()
        begin = time.perf_counter_ns()
        output = None
        for _ in range(count):
            output = operation()
        sync()
        return (time.perf_counter_ns() - begin) / 1e9, output

    def arrays(output, batch):
        if batch:
            data = [output.hplus.detach().cpu().numpy().copy(),
                    output.hcross.detach().cpu().numpy().copy()]
            meta = [{"delta_f": output.delta_f, "epoch": float(output.epoch)}] * a.batch
            devices = [str(output.hplus.device), str(output.hcross.device)]
        else:
            data = [np.stack([pair[p].numpy().copy() for pair in output]) for p in (0, 1)]
            meta = [{"delta_f": pair[0].delta_f, "epoch": float(pair[0].epoch)} for pair in output]
            for pair in output:
                assert pair[0].delta_f == pair[1].delta_f
                assert float(pair[0].epoch) == float(pair[1].epoch)
            devices = sorted({str(getattr(series._data, "tensor", np.empty(0)).device)
                              if hasattr(series._data, "tensor") else "cpu"
                              for pair in output for series in pair})
        return data, meta, devices

    def compare(actual, reference, actual_meta, reference_meta, gate, tolerance):
        metrics, ok = [], actual_meta == reference_meta
        for pol, (av, rv) in enumerate(zip(actual, reference)):
            same_shape = av.shape == rv.shape
            ok = ok and same_shape
            if not same_shape:
                metrics.append({"polarization": pol, "shape_matches": False,
                                "actual_shape": list(av.shape), "reference_shape": list(rv.shape)})
                continue
            for i in range(a.batch):
                x, ref = av[i], rv[i]
                support = np.abs(ref) > 0
                finite = bool(np.isfinite(x).all() and np.isfinite(ref).all())
                exact_support = bool(np.array_equal(x == 0, ref == 0))
                rel_l2 = float(np.linalg.norm(x - ref) / np.linalg.norm(ref))
                point = float(np.max(np.abs(x[support] - ref[support]) / np.abs(ref[support])))
                amp = float(np.max(np.abs(np.abs(x[support]) / np.abs(ref[support]) - 1)))
                phase = float(np.max(np.abs(np.angle(x[support] * np.conj(ref[support])))))
                passed = finite and exact_support and (gate is None or
                         (rel_l2 < tolerance if gate == "relative_l2" else point <= tolerance))
                ok = ok and passed
                metrics.append(dict(polarization=pol, row=i, finite=finite,
                                    exact_zero_support=exact_support, relative_l2=rel_l2,
                                    max_pointwise_relative=point, max_relative_amplitude=amp,
                                    max_wrapped_phase_radians=phase, passed=passed))
        return dict(passed=bool(ok), metadata_equal=actual_meta == reference_meta,
                    gate=gate, tolerance=tolerance, metrics=metrics)

    with torch.no_grad(), context:
        cold, _ = measure(1)
        for _ in range(2):
            operation()
        calibration, _ = measure(1)
        inner = max(1, min(a.max_inner, math.ceil(a.sample_ms / 1000 / calibration)))
        elapsed = []
        for _ in range(a.samples):
            sample, _ = measure(inner)
            elapsed.append(sample)
        per_call = [value / inner for value in elapsed]
        median = statistics.median(per_call)
        r["timing"] = {
            "cold_call_seconds": cold, "warmup_calls": 2,
            "calibration_seconds": calibration, "inner_calls_per_sample": inner,
            "sample_total_seconds": elapsed, "sample_seconds_per_call": per_call,
            "median_seconds_per_call": median,
            "median_seconds_per_waveform": median / a.batch,
            "waveforms_per_second": a.batch / median,
            "synchronization": "CUDA synchronize before/after each timed block" if is_cuda else "synchronous CPU",
            "cold_definition": "first waveform call after imports, capability discovery and scheme entry",
        }
        # Probe after timing so counters and host copies cannot inflate the measurements.
        counts = {"native_scalar": 0, "native_batch": 0, "lal_fd": 0}
        patches = [(native, "taylorf2_fd_torch", "native_scalar"),
                   (native, "taylorf2_fd_batch", "native_batch"),
                   (lalsimulation, "SimInspiralChooseFDWaveform", "lal_fd")]
        originals = []
        for module, name, key in patches:
            original = getattr(module, name)
            originals.append((module, name, original))

            def recording(*args, _original=original, _key=key, **kwargs):
                counts[_key] += 1
                return _original(*args, **kwargs)

            setattr(module, name, recording)
        try:
            actual, actual_meta, devices = arrays(operation(), is_batch)
            sync()
        finally:
            for module, name, original in originals:
                setattr(module, name, original)
        r["dispatch"]["probe_counts"] = counts
        expected = ({"native_scalar": 0, "native_batch": 0, "lal_fd": a.batch}
                    if a.route == "standard-cpu" else
                    {"native_scalar": 0 if is_batch else a.batch,
                     "native_batch": 1 if is_batch else 0, "lal_fd": 0})
        r["dispatch"]["expected_counts"] = expected
        r["dispatch"]["passed"] = counts == expected
        r["dispatch"]["actual"] = ("standard LAL CPU" if counts["lal_fd"] == a.batch
                                    else f"native Torch {device} {'batch' if is_batch else 'scalar loop'}"
                                    if counts == expected else "unexpected dispatch")
        r["output"] = {"devices": devices, "dtypes": [str(x.dtype) for x in actual],
                       "shapes": [list(x.shape) for x in actual], "metadata": actual_meta}
        assert all(x.dtype == np.complex128 for x in actual), "unexpected precision"
        assert all(d.startswith("cuda") if is_cuda else d == "cpu" for d in devices)
        if is_batch:
            scalar, scalar_meta, _ = arrays(scalar_call(), False)
            r["parity_batch_vs_native_scalar"] = compare(
                actual, scalar, actual_meta, scalar_meta, "pointwise_relative", 2e-10)
        else:
            scalar, scalar_meta = actual, actual_meta
    os.environ["PYCBC_TAYLORF2_NATIVE"] = "0"
    with scheme.CPUScheme(a.threads):
        reference, reference_meta, _ = arrays(scalar_call(), False)
    r["parity_scalar_vs_cpu_reference"] = compare(
        scalar, reference, scalar_meta, reference_meta, "relative_l2", 1e-11)
    r["parity_direct_vs_cpu_reference"] = compare(
        actual, reference, actual_meta, reference_meta,
        None if is_batch else "relative_l2", None if is_batch else 1e-11)
    r["parity_method"] = (
        "Full complex hplus/hcross, exact zero support, frequency/epoch metadata. "
        "Scalar-vs-LAL relative L2 <1e-11 from test_taylorf2_public_torch_parity_and_dispatch; "
        "batch-vs-native-scalar pointwise rtol=2e-10, atol=0 from test_each_batch_row_matches_scalar_taylorf2. "
        "Direct batch-vs-LAL complex/amplitude/phase metrics are additionally reported without a new tolerance.")
    passed = r["dispatch"]["passed"] and all(
        value["passed"] for key, value in r.items() if key.startswith("parity_") and isinstance(value, dict))
    r["status"] = "ok" if passed else "failed"
    if not passed:
        r["reason"] = "dispatch or full waveform parity gate failed; exclude timings from speedups"
    r["source"]["status_after"] = git(root, "status", "--porcelain")
    if r["source"]["status_after"] != r["source"]["status_porcelain"]:
        raise RuntimeError("source checkout changed during worker")


def main():
    a = parse_args()
    r = {"schema": 1, "status": "failed", "route": a.route, "batch": a.batch,
         "threads": a.threads, "replicate": a.replicate, "precision": a.precision,
         "harness_sha256": digest(__file__),
         "runtime": {"hostname": platform.node(), "platform": platform.platform(),
                     "machine": platform.machine(), "processor": platform.processor(),
                     "python": sys.version, "python_executable": sys.executable,
                     "pid": os.getpid(), "cwd": os.getcwd(),
                     "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "cpu_count": os.cpu_count()}}
    start = time.perf_counter()
    try:
        run(a, r)
    except Exception as exc:
        r["status"], r["reason"] = "failed", f"{type(exc).__name__}: {exc}"
        r["traceback"] = traceback.format_exc()
    r["runtime"]["worker_total_seconds"] = time.perf_counter() - start
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.out.with_suffix(a.out.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(r), indent=2, allow_nan=False) + "\n")
    temporary.replace(a.out)
    print(json.dumps({"out": str(a.out), "status": r["status"], "reason": r.get("reason")}))
    return 0 if r["status"] in ("ok", "unsupported") else 1


if __name__ == "__main__":
    raise SystemExit(main())
