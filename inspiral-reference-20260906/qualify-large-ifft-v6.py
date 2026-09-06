#!/usr/bin/env python3
"""Measure explicit large-IFFT candidates without changing dispatch allowlists.

Run each --threads value in a separate, appropriately pinned process. Supply
the usual controlled math-library environment before Python starts. Timings
are serial: cold construction, first execution, then warmed median/spread.
The legacy FFTW accuracy limit is never relaxed. No source or remote writes.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument("--source", required=True, type=Path)
bootstrap_args, _ = bootstrap.parse_known_args()
source_root = bootstrap_args.source.resolve(strict=True)
if not (source_root / "pycbc" / "__init__.py").is_file():
    raise SystemExit("--source must be an existing PyCBC source checkout")
sys.path.insert(0, str(source_root))

import numpy as np
import torch

import pycbc
from pycbc import scheme
from pycbc.fft import fftw, mkl, torchfft
from pycbc.types import Array, zeros
from pycbc.types.array_torch import TorchArrayData


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def source_state():
    root = Path(pycbc.__file__).resolve().parent.parent
    return {
        "root": str(root),
        "head": command(root, "rev-parse", "HEAD"),
        "status": command(root, "status", "--porcelain=v1", "--untracked-files=all"),
        "torchfft_path": str(Path(torchfft.__file__).resolve()),
        "torchfft_sha256": sha256(torchfft.__file__),
        "harness_sha256": sha256(__file__),
    }


def save(path, result):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def values_for(size, seed, pattern, scale):
    rng = np.random.default_rng(seed)
    values = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(np.complex64)
    if pattern == "banded":
        # Analytic matched-filter spectrum: finite positive-frequency band.
        values[: size // 128] = 0
        values[size // 3 :] = 0
    elif pattern == "impulse":
        values[:] = 0
        values[0] = 2 - 3j
        values[size // 8] = 0.5 + 0.25j
    values *= np.float32(scale)
    return values


def error_metrics(actual, truth):
    difference = actual.astype(np.complex128) - truth
    l2 = float(np.linalg.norm(difference))
    peak = float(np.max(np.abs(difference)))
    norm = float(np.linalg.norm(truth))
    peak_truth = float(np.max(np.abs(truth)))
    return {
        "l2": l2,
        "max_abs": peak,
        "relative_l2": l2 / norm if norm else l2,
        "normalized_max_abs": peak / peak_truth if peak_truth else peak,
    }


def ratio(value, baseline):
    # Retain exact zero failures without JSON Infinity or an artificial floor.
    return value / baseline if baseline else (0.0 if value == 0 else None)


def compare_error(metrics, legacy):
    return {
        "l2_ratio": ratio(metrics["l2"], legacy["l2"]),
        "max_abs_ratio": ratio(metrics["max_abs"], legacy["max_abs"]),
        "passed": metrics["l2"] <= legacy["l2"]
        and metrics["max_abs"] <= legacy["max_abs"],
    }


class Route:
    def __init__(self, name, size, nthreads):
        self.name = name
        self.size = size
        self.native_library_threads = nthreads
        self.owner = None
        self.close = lambda: None
        if name.startswith("legacy_"):
            precision = np.complex128 if name == "legacy_mkl_double" else np.complex64
            with scheme.CPUScheme(num_threads=nthreads):
                self.source = zeros(size, dtype=precision)
                self.target = zeros(size, dtype=precision)
                backend = fftw if name == "legacy_fftw_single" else mkl
                self.owner = backend.IFFT(self.source, self.target)
            self.assign = lambda values: self.source.data.__setitem__(
                slice(None), values
            )
            self.output = lambda: self.target.numpy().copy()
            self.input = lambda: self.source.numpy().copy()
            self.execute = self.owner.execute
            if backend is mkl:

                def close_mkl():
                    import ctypes

                    descriptor = self.owner.desc
                    mkl.check_status(
                        mkl.lib.DftiFreeDescriptor(ctypes.byref(descriptor))
                    )

                self.close = close_mkl
        else:
            self.source = torch.empty(size, dtype=torch.complex64, device="cpu")
            self.target = torch.empty_like(self.source)
            self.assign = lambda values: self.source.copy_(torch.from_numpy(values))
            self.output = lambda: self.target.numpy().copy()
            self.input = lambda: self.source.numpy().copy()
            if name == "current_torch_dispatch":
                with scheme.TorchScheme("cpu"):
                    self.owner = torchfft.IFFT(
                        Array(TorchArrayData(self.source), copy=False),
                        Array(TorchArrayData(self.target), copy=False),
                    )
                self.execute = self.owner.execute
            elif name == "fftw_double_workspace_one_native_thread":
                # This is precisely the old one-thread CPU search work plan;
                # with --threads > 1, Torch copy_ has the requested thread count.
                self.native_library_threads = 1
                self.owner = torchfft._FFTWCPUWorkPlan(fftw, size, False, 0)
                self.execute = lambda: self.owner.execute(self.source, self.target)
            else:
                self.owner = torchfft._MKLCPUDirectIFFTPlan(
                    mkl,
                    size,
                    self.source,
                    self.target,
                    nthreads=nthreads,
                    promote=name == "mkl_double_workspace",
                )
                self.execute = lambda: self.owner.execute(self.source, self.target)


def timed_route(name, size, threads, repeats, values):
    torch.set_num_threads(threads)
    started = time.perf_counter()
    route = Route(name, size, threads)
    construction = time.perf_counter() - started
    # CPUScheme exits by restoring its OpenMP runtime to one thread. Restore
    # Torch's requested worker count before timing copies or Torch dispatch.
    torch.set_num_threads(threads)
    route.assign(values)
    started = time.perf_counter()
    route.execute()
    first = time.perf_counter() - started
    for _ in range(2):
        route.execute()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        route.execute()
        samples.append(time.perf_counter() - started)
    result = {
        "construction_seconds": construction,
        "first_execute_seconds": first,
        "warmup_executions": 2,
        "steady_seconds": samples,
        "steady_median_seconds": statistics.median(samples),
        "steady_min_seconds": min(samples),
        "steady_max_seconds": max(samples),
        "native_library_threads": route.native_library_threads,
        "plan_type": type(route.owner).__name__,
    }
    if name == "mkl_double_workspace":
        result["retained_workspace_bytes"] = 2 * size * 16
    elif name == "fftw_double_workspace_one_native_thread":
        result["retained_workspace_bytes"] = size * 16
    return route, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2**20, 2**21, 2**22])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 91, 812, 20260906])
    parser.add_argument("--scales", type=float, nargs="+", default=[1e-12, 1.0, 1e12])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if Path(pycbc.__file__).resolve().parent.parent != args.source.resolve():
        parser.error("imported PyCBC source differs from --source")
    if (
        args.output.exists()
        or args.output.with_name(args.output.name + ".tmp").exists()
    ):
        parser.error("preserve existing output; choose a new path")
    if args.threads < 1 or args.repeats < 3 or min(args.sizes) < 8:
        parser.error("positive threads, at least three repeats and sizes >= 8 required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    fftw.set_measure_level(0)
    result = {
        "schema": "torch-large-ifft-qualification-v6",
        "state": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "command": sys.argv,
        "source_before": source_state(),
        "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "libraries": {
            "mkl": mkl.lib._name,
            "fftw_float": fftw.float_lib._name,
            "fftw_double": fftw.double_lib._name,
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "MKL_DYNAMIC",
                "MKL_THREADING_LAYER",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "policy": {
            "truth": "NumPy complex128 unnormalized inverse",
            "error_gate": "candidate L2 and max absolute errors each <= legacy FFTW errors; no tolerance floor",
            "parity_gate": "bitwise matching-precision standard PyCBC MKL result cast to complex64",
            "release": "manual only after review and complete scientific campaign",
        },
        "sizes": [],
    }
    save(args.output, result)
    names = (
        "legacy_fftw_single",
        "legacy_mkl_single",
        "legacy_mkl_double",
        "current_torch_dispatch",
        "fftw_double_workspace_one_native_thread",
        "mkl_single_direct",
        "mkl_double_workspace",
    )
    try:
        for size in args.sizes:
            row = {"size": size, "timings": {}, "cases": []}
            result["sizes"].append(row)
            routes = {}
            try:
                initial = values_for(size, args.seeds[0], "dense", 1.0)
                for name in names:
                    route, timing = timed_route(
                        name, size, args.threads, args.repeats, initial
                    )
                    routes[name], row["timings"][name] = route, timing
                    print(
                        size,
                        name,
                        "cold",
                        timing["construction_seconds"],
                        "median",
                        timing["steady_median_seconds"],
                        flush=True,
                    )
                for seed in args.seeds:
                    for pattern in ("dense", "banded", "impulse"):
                        for scale in args.scales:
                            values = values_for(size, seed, pattern, scale)
                            truth = np.fft.ifft(values.astype(np.complex128)) * size
                            outputs, metrics = {}, {}
                            for name, route in routes.items():
                                route.assign(values)
                                route.execute()
                                outputs[name] = route.output()
                                if not np.array_equal(route.input(), values):
                                    raise RuntimeError(f"{name} mutated its source")
                                if not np.all(np.isfinite(outputs[name])):
                                    raise RuntimeError(
                                        f"{name} produced nonfinite output"
                                    )
                                metrics[name] = error_metrics(outputs[name], truth)
                            gates = {}
                            for candidate, reference in (
                                ("mkl_single_direct", "legacy_mkl_single"),
                                ("mkl_double_workspace", "legacy_mkl_double"),
                            ):
                                gates[candidate] = compare_error(
                                    metrics[candidate], metrics["legacy_fftw_single"]
                                )
                                gates[candidate]["bitwise_mkl_parity"] = np.array_equal(
                                    outputs[candidate].view(np.uint32),
                                    outputs[reference]
                                    .astype(np.complex64)
                                    .view(np.uint32),
                                )
                                gates[candidate]["passed"] &= gates[candidate][
                                    "bitwise_mkl_parity"
                                ]
                            row["cases"].append(
                                {
                                    "seed": seed,
                                    "pattern": pattern,
                                    "scale": scale,
                                    "errors": metrics,
                                    "gates": gates,
                                }
                            )
                        print(size, "completed", seed, pattern, flush=True)
                        save(args.output, result)
                row["summary"] = {}
                baseline = row["timings"]["current_torch_dispatch"][
                    "steady_median_seconds"
                ]
                for candidate in ("mkl_single_direct", "mkl_double_workspace"):
                    steady = row["timings"][candidate]["steady_median_seconds"]
                    row["summary"][candidate] = {
                        "all_precision_cases_passed": all(
                            case["gates"][candidate]["passed"] for case in row["cases"]
                        ),
                        "speedup_vs_current_dispatch": baseline / steady,
                        "faster_than_current_dispatch": steady < baseline,
                    }
                save(args.output, result)
            finally:
                for route in routes.values():
                    route.close()
                routes.clear()
                gc.collect()
        result["source_after"] = source_state()
        if result["source_after"] != result["source_before"]:
            raise RuntimeError("source changed during qualification")
        result["state"] = "complete"
    except BaseException as error:
        result["state"] = "failed"
        result["error"] = repr(error)
        raise
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.output, result)


if __name__ == "__main__":
    main()
