#!/usr/bin/env python3
"""Measure one exact-revision public TaylorF2 route in a fresh process."""

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

ROUTES = ("cuda-off", "cuda-on", "standard-cpu")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "PYCBC_NUM_THREADS")
GATE = "PYCBC_TAYLORF2_TRITON"
DEFAULT_KERNEL = "pycbc.waveform.taylorf2_triton"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def safe(value):
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parameters(batch, delta_f):
    """Frozen host rows, repeated exactly; no device input preparation cache."""
    return [dict(mass1=1.4 + 0.05 * (i % 8), mass2=1.3 - 0.02 * (i % 8),
                 spin1z=0.02, spin2z=-0.01, distance=100.0, inclination=0.4,
                 coa_phase=0.2, delta_f=delta_f, f_lower=20.0, f_final=1024.0,
                 f_ref=30.0) for i in range(batch)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-sha", required=True)
    p.add_argument("--route", choices=ROUTES, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--delta-f", type=float, choices=(0.25, 0.03125), default=0.25)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--replicate", type=int, default=1)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--sample-ms", type=float, default=50)
    p.add_argument("--max-inner", type=int, default=4096)
    p.add_argument("--kernel-module", default=DEFAULT_KERNEL)
    p.add_argument("--launcher", default="evaluate_taylorf2")
    p.add_argument("--kernel", default="_taylorf2_kernel")
    a = p.parse_args()
    if min(a.batch, a.threads, a.replicate, a.samples, a.max_inner) < 1:
        p.error("counts must be positive")
    if a.sample_ms < 50:
        p.error("sample-ms must be at least 50")
    if a.out.exists():
        p.error("output already exists; choose a fresh path")
    return a


def run(a, r):
    root = a.root.resolve()
    r["source"] = dict(root=str(root), sha=git(root, "rev-parse", "HEAD"),
                       tree=git(root, "rev-parse", "HEAD^{tree}"),
                       status_porcelain=git(root, "status", "--porcelain"))
    r["source"]["clean"] = not r["source"]["status_porcelain"]
    if r["source"]["sha"] != a.expected_sha or not r["source"]["clean"]:
        raise RuntimeError("source must be clean and match --expected-sha")
    sources = ("pycbc/waveform/taylorf2_torch.py", "pycbc/waveform/waveform.py",
               "pycbc/waveform/torch_waveform_registry.py",
               a.kernel_module.replace(".", "/") + ".py")
    r["source"]["sha256"] = {name: digest(root / name) for name in sources}
    r["parameters"] = parameters(a.batch, a.delta_f)
    r["parameters_sha256"] = hashlib.sha256(json.dumps(
        r["parameters"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    r["workload"] = dict(approximant="TaylorF2", bins=int(1024 / a.delta_f) + 1,
                         precision="complex128", polarizations=["hplus", "hcross"],
                         input_policy="Python host scalars/lists; public validation, conversion and allocation included",
                         output_policy="device output retained; host copy excluded from timing",
                         autograd=False)
    for package in ("numpy", "torch", "lalsuite", "scipy", "triton"):
        try:
            r["runtime"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            r["runtime"][package] = None
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    for key in THREAD_VARS:
        os.environ[key] = str(a.threads)
    for key in ("PYCBC_TORCH_NATIVE", "PYCBC_TORCH_NATIVE_PORTS"):
        os.environ.pop(key, None)
    os.environ[GATE] = "1" if a.route == "cuda-on" else "0"
    os.environ["PYCBC_TAYLORF2_NATIVE"] = "0" if a.route == "standard-cpu" else "1"
    cache_dir = a.out.resolve().with_suffix(".triton-cache")
    if cache_dir.exists():
        raise RuntimeError("fresh per-worker Triton cache required")
    cache_dir.mkdir(parents=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    r["cache"] = {"triton_cache_dir": str(cache_dir), "initial_files": [],
                  "scope": "fresh for each worker; CUDA driver cache is not cleared"}
    r["environment"] = {k: v for k, v in os.environ.items() if k in THREAD_VARS or
                        k.startswith("PYCBC_") or k.startswith("TRITON_") or
                        k in ("CUDA_VISIBLE_DEVICES", "CUDA_CACHE_PATH", "MKL_THREADING_LAYER")}
    import_start = time.perf_counter()
    import numpy as np
    import pycbc
    from pycbc import scheme
    from pycbc.waveform import get_fd_waveform, get_fd_waveform_batch
    import pycbc.waveform.taylorf2_torch as native
    import lalsimulation
    import torch
    r["runtime"]["import_seconds"] = time.perf_counter() - import_start
    r["runtime"]["module_origins"] = {name: str(Path(module.__file__).resolve())
        for name, module in (("pycbc", pycbc), ("taylorf2_torch", native),
                             ("torch", torch), ("numpy", np), ("lalsimulation", lalsimulation))}
    for name in ("pycbc", "taylorf2_torch"):
        Path(r["runtime"]["module_origins"][name]).relative_to(root)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    r["runtime"].update(torch_threads=torch.get_num_threads(),
                         torch_interop_threads=torch.get_num_interop_threads(),
                         torch_cuda_build=torch.version.cuda)
    is_cuda = a.route.startswith("cuda")
    if is_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested route")
    if is_cuda:
        props = torch.cuda.get_device_properties(0)
        r["runtime"]["cuda_device"] = dict(index=0, name=props.name,
            memory_bytes=props.total_memory, compute_capability=[props.major, props.minor])
        torch.cuda.reset_peak_memory_stats(0)
    rows = r["parameters"]
    batch_params = {key: rows[0][key] if key == "delta_f" else [row[key] for row in rows]
                    for key in rows[0]}

    def operation():
        if is_cuda:
            return get_fd_waveform_batch("TaylorF2", **batch_params)
        return [get_fd_waveform(approximant="TaylorF2", **row) for row in rows]

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
        elapsed = (time.perf_counter_ns() - begin) / 1e9
        del output
        return elapsed

    def arrays(output, batch):
        if batch:
            data = [output.hplus.detach().cpu().numpy().copy(),
                    output.hcross.detach().cpu().numpy().copy()]
            meta = dict(delta_f=output.delta_f, epoch=float(output.epoch),
                        first_bins=output.first_bins.detach().cpu().tolist(),
                        end_bins=output.end_bins.detach().cpu().tolist())
            devices = [str(output.hplus.device), str(output.hcross.device)]
        else:
            data = [np.stack([pair[p].numpy().copy() for pair in output]) for p in (0, 1)]
            assert all(pair[0].delta_f == pair[1].delta_f == a.delta_f and
                       float(pair[0].epoch) == float(pair[1].epoch) == -1 / a.delta_f
                       for pair in output)
            meta = dict(delta_f=a.delta_f, epoch=-1 / a.delta_f,
                        first_bins=[math.ceil(row["f_lower"] / a.delta_f) for row in rows[:len(output)]],
                        end_bins=[math.floor(row["f_final"] / a.delta_f) + 1 for row in rows[:len(output)]])
            devices = sorted({str(series._data.tensor.device) if hasattr(series._data, "tensor")
                              else "cpu" for pair in output for series in pair})
        return data, meta, devices

    def scalar_reference(reference_rows, expected_lal):
        """Confirm each link in the scalar reference chain without timing it."""
        counts = dict(native_scalar=0, lal_fd=0)
        originals = []
        for owner, name, key in ((native, "taylorf2_fd_torch", "native_scalar"),
                                 (lalsimulation, "SimInspiralChooseFDWaveform", "lal_fd")):
            original = getattr(owner, name)
            originals.append((owner, name, original))
            def recording(*args, _original=original, _key=key, **kwargs):
                counts[_key] += 1
                return _original(*args, **kwargs)
            setattr(owner, name, recording)
        try:
            values = [get_fd_waveform(approximant="TaylorF2", **row) for row in reference_rows]
        finally:
            for owner, name, original in reversed(originals):
                setattr(owner, name, original)
        expected = dict(native_scalar=0 if expected_lal else len(reference_rows),
                        lal_fd=len(reference_rows) if expected_lal else 0)
        key = "lal" if expected_lal else "native_scalar"
        r.setdefault("reference_dispatch", {})[key] = dict(counts=counts,
            expected_counts=expected, passed=counts == expected)
        if counts != expected:
            raise RuntimeError(f"unexpected {key} reference dispatch")
        return values

    def compare(actual, reference, actual_meta, reference_meta, *, point=2e-10, l2=2e-10,
                row_map=None):
        metrics = []
        metadata_equal = actual_meta == reference_meta if row_map is None else (
            actual_meta["delta_f"] == reference_meta["delta_f"] and
            actual_meta["epoch"] == reference_meta["epoch"] and all(
                actual_meta[key][i] == reference_meta[key][j]
                for key in ("first_bins", "end_bins") for i, j in enumerate(row_map)))
        ok = metadata_equal
        for pol, (av, rv) in enumerate(zip(actual, reference)):
            mapping = list(range(len(av))) if row_map is None else row_map
            if len(av) != len(mapping) or av.shape[1:] != rv.shape[1:]:
                metrics.append(dict(polarization=pol, passed=False, shape_matches=False,
                                    actual_shape=list(av.shape), reference_shape=list(rv.shape)))
                ok = False
                continue
            for i, j in enumerate(mapping):
                x, ref = av[i], rv[j]
                support = np.abs(ref) > 0
                finite = bool(np.isfinite(x).all() and np.isfinite(ref).all())
                exact_zeros = bool(np.array_equal(x == 0, ref == 0))
                denominator = float(np.linalg.norm(ref))
                rel_l2 = float(np.linalg.norm(x - ref) / denominator) if denominator else math.inf
                pointwise = float(np.max(np.abs(x[support] - ref[support]) / np.abs(ref[support]))) if support.any() else math.inf
                passed = finite and exact_zeros and (l2 is None or rel_l2 <= l2) and (point is None or pointwise <= point)
                metrics.append(dict(polarization=pol, row=i, reference_row=j,
                    finite=finite, exact_zero_support=exact_zeros, relative_l2=rel_l2,
                    max_pointwise_relative=pointwise, passed=passed))
                ok = ok and passed
        return dict(passed=bool(ok), metadata_equal=metadata_equal,
                    pointwise_tolerance=point, relative_l2_tolerance=l2, metrics=metrics)

    context = scheme.TorchScheme("cuda:0", num_threads=a.threads) if is_cuda else scheme.CPUScheme(a.threads)
    with torch.no_grad(), context:
        cold = measure(1)
        r["cache"]["files_after_cold"] = {str(p.relative_to(cache_dir)): digest(p)
            for p in sorted(cache_dir.rglob("*")) if p.is_file()}
        r["timing"] = dict(cold_call_seconds=cold, warmup_calls=2,
            cold_definition="first complete public call after imports and scheme entry; includes lazy initialization and any compilation, not pure compiler time")
        for _ in range(2):
            operation()
        calibration = measure(1)
        target = a.sample_ms / 1000
        inner = max(1, min(a.max_inner, math.ceil(target / calibration * 1.15)))
        calibration_trials, sample_totals, sample_counts = [], [], []
        for _ in range(a.samples):
            while True:
                elapsed = measure(inner)
                if elapsed >= target:
                    break
                calibration_trials.append(dict(inner_calls=inner, seconds=elapsed))
                if inner == a.max_inner:
                    raise RuntimeError("max-inner prevents the minimum sample duration")
                inner = min(a.max_inner, max(inner + 1, math.ceil(inner * target / elapsed * 1.15)))
            sample_totals.append(elapsed)
            sample_counts.append(inner)
        per_call = [elapsed / count for elapsed, count in zip(sample_totals, sample_counts)]
        median = statistics.median(per_call)
        r["timing"].update(calibration_seconds=calibration,
            discarded_duration_calibrations=calibration_trials,
            inner_calls_per_sample=sample_counts, sample_total_seconds=sample_totals,
            sample_seconds_per_call=per_call, median_seconds_per_call=median,
            minimum_sample_seconds=target, minimum_duration_passed=all(x >= target for x in sample_totals),
            median_seconds_per_waveform=median / a.batch, waveforms_per_second=a.batch / median,
            synchronization="CUDA synchronize before/after each block" if is_cuda else "synchronous CPU",
            timed_surface="complete public call; no monkey patches or profiler active")
        # Instrumentation is installed only after every timed call is finished.
        kernel_module = importlib.import_module(a.kernel_module) if is_cuda else None
        if kernel_module is not None:
            origin = Path(kernel_module.__file__).resolve()
            origin.relative_to(root)
            r["runtime"]["module_origins"]["triton_kernel_module"] = str(origin)
        counts = dict(native_batch=0, native_scalar=0, lal_fd=0, launcher=0,
                      kernel_run_attempts=0, kernel_run_successes=0)
        patches = []

        def wrap(owner, name, key):
            original = getattr(owner, name)
            patches.append((owner, name, original))
            def recording(*args, **kwargs):
                counts[key] += 1
                return original(*args, **kwargs)
            setattr(owner, name, recording)

        wrap(native, "taylorf2_fd_batch", "native_batch")
        wrap(native, "taylorf2_fd_torch", "native_scalar")
        wrap(lalsimulation, "SimInspiralChooseFDWaveform", "lal_fd")
        if kernel_module is not None:
            wrap(kernel_module, a.launcher, "launcher")
            kernel = getattr(kernel_module, a.kernel)
            from triton.runtime.jit import JITFunction
            if not isinstance(kernel, JITFunction):
                raise RuntimeError("kernel probe requires the actual Triton JITFunction")
            original_run = kernel.run
            patches.append((kernel, "run", original_run))
            def kernel_run(*args, **kwargs):
                counts["kernel_run_attempts"] += 1
                result = original_run(*args, **kwargs)
                if not kwargs.get("warmup", False):
                    counts["kernel_run_successes"] += 1
                return result
            kernel.run = kernel_run
        try:
            actual, actual_meta, devices = arrays(operation(), is_cuda)
            sync()
        finally:
            for owner, name, original in reversed(patches):
                setattr(owner, name, original)
        expected = dict(native_batch=int(is_cuda), native_scalar=0,
                        lal_fd=0 if is_cuda else a.batch,
                        launcher=int(a.route == "cuda-on"),
                        kernel_run_attempts=int(a.route == "cuda-on"),
                        kernel_run_successes=int(a.route == "cuda-on"))
        r["dispatch"] = dict(requested=a.route, counts=counts, expected_counts=expected,
            passed=counts == expected, triton_actual=counts["kernel_run_successes"] > 0,
            method="separate post-timing public call; launcher and actual JITFunction.run counted; synchronized after return")
        expected_meta = dict(delta_f=a.delta_f, epoch=-1 / a.delta_f,
            first_bins=[math.ceil(row["f_lower"] / a.delta_f) for row in rows],
            end_bins=[math.floor(row["f_final"] / a.delta_f) + 1 for row in rows])
        r["output"] = dict(devices=devices, dtypes=[str(x.dtype) for x in actual],
                           shapes=[list(x.shape) for x in actual], metadata=actual_meta,
                           expected_metadata=expected_meta)
        assert actual_meta == expected_meta, "unexpected output metadata"
        assert all(x.dtype == np.complex128 and x.shape == (a.batch, r["workload"]["bins"]) for x in actual), "unexpected precision or shape"
        assert all(d.startswith("cuda") if is_cuda else d == "cpu" for d in devices)
        unique_count = min(8, a.batch)
        unique_rows = rows[:unique_count]
        row_map = [i % unique_count for i in range(a.batch)]
        r["reference_policy"] = dict(distinct_rows=unique_count, row_map=row_map,
            method="all bins and both polarizations in every row checked; identical frozen rows reuse an exact scalar reference")
        if is_cuda:
            os.environ[GATE] = "0"
            off, off_meta, _ = arrays(operation(), True)
            r["parity_actual_vs_gate_off"] = compare(actual, off, actual_meta, off_meta)
            scalar_output = scalar_reference(unique_rows, expected_lal=False)
            scalar, scalar_meta, scalar_devices = arrays(scalar_output, False)
            assert all(device.startswith("cuda") for device in scalar_devices)
            del scalar_output
            r["parity_gate_off_vs_native_scalar"] = compare(off, scalar, off_meta, scalar_meta, row_map=row_map)
            del off
        else:
            scalar, scalar_meta = [x[:unique_count] for x in actual], {
                **actual_meta, "first_bins": actual_meta["first_bins"][:unique_count],
                "end_bins": actual_meta["end_bins"][:unique_count]}
    os.environ["PYCBC_TAYLORF2_NATIVE"] = "0"
    os.environ[GATE] = "0"
    with scheme.CPUScheme(a.threads):
        reference_output = scalar_reference(unique_rows, expected_lal=True)
        reference, reference_meta, reference_devices = arrays(reference_output, False)
        assert reference_devices == ["cpu"]
    r["parity_native_scalar_vs_lal"] = compare(scalar, reference, scalar_meta, reference_meta, point=None, l2=1e-11)
    r["parity_actual_vs_lal"] = compare(actual, reference, actual_meta, reference_meta,
                                       point=None, l2=None, row_map=row_map)
    r["parity_method"] = ("Every output bin, row and both polarizations checked. "
        "Actual vs gate off and gate off vs native scalar require pointwise and relative L2 <=2e-10. "
        "Native scalar vs LAL requires relative L2 <=1e-11. Direct actual-vs-LAL complex error "
        "is additionally reported without a separate error tolerance. All comparisons require "
        "finite values, exact zero support and exact frequency/epoch/support metadata.")
    r["source"]["status_after"] = git(root, "status", "--porcelain")
    if r["source"]["status_after"] != r["source"]["status_porcelain"]:
        raise RuntimeError("source checkout changed during worker")
    if is_cuda:
        r["runtime"]["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        r["runtime"]["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
    parity_ok = all(value["passed"] for key, value in r.items() if key.startswith("parity_") and isinstance(value, dict))
    r["status"] = "ok" if parity_ok and r["dispatch"]["passed"] else "failed"
    if r["status"] != "ok":
        r["reason"] = "complete waveform parity or actual dispatch failed; timings excluded from comparisons"


def main():
    a = parse_args()
    r = dict(schema=1, status="failed", route=a.route, batch=a.batch, delta_f=a.delta_f,
        threads=a.threads, replicate=a.replicate, expected_sha=a.expected_sha,
        harness_sha256=digest(__file__), command=sys.argv,
        runtime=dict(hostname=platform.node(), platform=platform.platform(), machine=platform.machine(),
            python=sys.version, python_executable=sys.executable, pid=os.getpid(), cwd=os.getcwd(),
            cpu_count=os.cpu_count(), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    begin = time.perf_counter()
    try:
        run(a, r)
    except BaseException as exc:
        r.update(status="failed", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    r["runtime"]["worker_total_seconds"] = time.perf_counter() - begin
    write_json(a.out, r)
    print(json.dumps(dict(out=str(a.out), status=r["status"], reason=r.get("reason"))))
    return int(r["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
