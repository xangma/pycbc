#!/usr/bin/env python3
"""One fresh-process qualified Torch CPU IFFT / wisdom-cache measurement.

This external harness does not modify the measured checkout. Invoke separately
for main/off, optional/off, optional/cold, optional/warm. Each cold/warm pair
must use its own cache directory, with cold completed before warm starts.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def cache_files(directory):
    if directory is None or not directory.exists():
        return []
    return [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(directory.glob("*.wisdom"))
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-mode", choices=("off", "cold", "warm"), required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--threads", type=int, choices=(1, 4), default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7101)
    return parser.parse_args()


def run(args):
    root = args.source_root.resolve()
    assert git(root, "rev-parse", "HEAD") == args.expected_revision
    assert not git(root, "status", "--porcelain", "--untracked-files=no")
    assert args.samples >= 3 and args.inner >= 1 and args.warmups >= 0
    assert not args.output.exists(), args.output
    if args.cache_mode != "off":
        assert args.cache_dir is not None, (
            "Enabled caching requires an isolated directory"
        )
        assert args.threads == 1, (
            "Automatic wisdom is qualified only at one Torch thread"
        )
    cache_dir = args.cache_dir.resolve() if args.cache_dir else None
    before_files = cache_files(cache_dir)
    if args.cache_mode == "cold":
        assert not before_files, "Cold worker refuses an already populated cache"
    if args.cache_mode == "warm":
        assert before_files, "Warm worker requires wisdom from a completed cold worker"

    # Set before NumPy/Torch/PyCBC import, and never inherit experimental routing.
    for name in list(os.environ):
        if name.startswith("PYCBC_TORCH_") or name in (
            "PYCBC_ENABLE_CUDA_GRAPHS",
            "PYCBC_FFTW_PLAN_TYPE",
        ):
            os.environ.pop(name)
    os.environ.update(
        {
            "OMP_NUM_THREADS": str(args.threads),
            "MKL_NUM_THREADS": str(args.threads),
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "PYCBC_SCHEME": "torch:cpu",
        }
    )
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    import pycbc
    from pycbc import scheme
    from pycbc.fft import IFFT, fftw, torchfft
    from pycbc.types import Array, zeros

    assert Path(pycbc.__file__).resolve().parent.parent == root
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    fftw.set_measure_level(0)
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append({"level": record.levelname, "message": record.getMessage()})

    logger = logging.getLogger("pycbc.fft.wisdom_cache")
    logger.setLevel(logging.INFO)
    logger.addHandler(Capture())
    try:
        from pycbc.fft import wisdom_cache
    except ImportError:
        wisdom_cache = None
    if args.cache_mode != "off":
        assert wisdom_cache is not None, "This head has no automatic wisdom cache"
    if wisdom_cache is not None:
        wisdom_cache.configure_from_cli(
            SimpleNamespace(
                fftw_wisdom_cache=args.cache_mode != "off",
                fftw_wisdom_cache_dir=str(cache_dir) if cache_dir else None,
            )
        )

    size = 131072
    rng = np.random.default_rng(args.seed)
    raw = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(np.complex64)
    expected = np.fft.ifft(raw.astype(np.complex128)) * size
    with scheme.TorchScheme("cpu"):
        source = Array(raw)
        target = zeros(size, dtype=np.complex64)
        started = time.perf_counter()
        engine = IFFT(source, target)
        plan_seconds = time.perf_counter() - started
        plan = getattr(engine, "_fftw_plan", None)
        plan_info = {
            "engine_class": f"{type(engine).__module__}.{type(engine).__name__}",
            "plan_class": type(plan).__name__ if plan is not None else None,
            "measure_level": getattr(plan, "_measure_level", None),
            "retained_workspace": getattr(plan, "_use_retained_workspace", None),
            "aligned": getattr(plan, "_aligned", None),
            "threads": getattr(plan, "_nthreads", None),
        }

        # An untimed observation proves execution admission, then restores the
        # exact production function before warmups and timing.
        original_execute = torchfft._execute_fftw_cpu_plan
        original_fallback = torchfft.ifft
        dispatch = {"attempts": 0, "native_successes": 0, "torch_fallback_calls": 0}

        def observe(fftobj):
            result = original_execute(fftobj)
            dispatch["attempts"] += 1
            dispatch["native_successes"] += int(result)
            return result

        def observe_fallback(*a, **kw):
            dispatch["torch_fallback_calls"] += 1
            return original_fallback(*a, **kw)

        torchfft._execute_fftw_cpu_plan = observe
        torchfft.ifft = observe_fallback
        try:
            engine.execute()
        finally:
            torchfft._execute_fftw_cpu_plan = original_execute
            torchfft.ifft = original_fallback
        actual = target.numpy().copy()
        reference_scale = float(np.max(np.abs(expected)))
        max_relative_to_peak = float(
            np.max(np.abs(actual - expected)) / reference_scale
        )
        relative_l2 = float(
            np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        )
        passed = bool(
            np.all(np.isfinite(actual))
            and max_relative_to_peak <= 2e-6
            and relative_l2 <= 2e-6
            and np.array_equal(source.numpy(), raw)
        )
        samples = []
        for _ in range(args.warmups):
            engine.execute()
        for _ in range(args.samples):
            started = time.perf_counter()
            for _ in range(args.inner):
                engine.execute()
            samples.append((time.perf_counter() - started) / args.inner)
        final_output = target.numpy().copy()
        passed = bool(passed and np.array_equal(final_output, actual))

    after_files = cache_files(cache_dir)
    imports = sum("Imported cached FFTW wisdom" in row["message"] for row in messages)
    exports = sum("Updated automatic FFTW wisdom" in row["message"] for row in messages)
    direct = plan_info["plan_class"] == "_FFTWCPUDirectPlan"
    qualified = bool(
        direct and plan_info["retained_workspace"] and dispatch["native_successes"] == 1
    )
    expected_route = (
        qualified
        if args.threads == 1
        else bool(
            plan is None
            and dispatch["native_successes"] == 0
            and dispatch["torch_fallback_calls"] == 1
        )
    )
    if args.cache_mode == "off":
        cache_confirmed = (
            not imports and not exports and plan_info["measure_level"] in (0, None)
        )
    elif args.cache_mode == "cold":
        cache_confirmed = bool(
            not imports
            and exports == 1
            and after_files
            and plan_info["measure_level"] == 1
        )
    else:
        cache_confirmed = bool(
            imports == 1
            and not exports
            and before_files == after_files
            and plan_info["measure_level"] == 0
        )
    result = {
        "schema_version": 1,
        "label": args.label,
        "status": "passed"
        if passed and expected_route and cache_confirmed
        else "failed",
        "scope": "Public Torch CPU IFFT; plan construction and warm execution separately",
        "source": {
            "root": str(root),
            "revision": args.expected_revision,
            "tree": git(root, "rev-parse", "HEAD^{tree}"),
            "tracked_clean": True,
            "pycbc_module": pycbc.__file__,
            "torchfft_sha256": digest(Path(torchfft.__file__)),
            "harness_sha256": digest(Path(__file__)),
        },
        "environment": {
            "host": platform.node(),
            "pid": os.getpid(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "torch_threads": torch.get_num_threads(),
            "fftw_float_library": str(fftw.float_lib._name),
        },
        "workload": {
            "size": size,
            "batch": 1,
            "dtype": "complex64",
            "seed": args.seed,
            "inner": args.inner,
            "warmups": args.warmups,
        },
        "plan": plan_info
        | {
            "construction_seconds": plan_seconds,
            "untimed_dispatch": dispatch,
            "cache_route_qualified": qualified,
            "expected_route_confirmed": expected_route,
        },
        "cache": {
            "mode": args.cache_mode,
            "directory": str(cache_dir) if cache_dir else None,
            "enabled": bool(wisdom_cache is not None and wisdom_cache._config.enabled),
            "before": before_files,
            "after": after_files,
            "successful_imports": imports,
            "successful_exports": exports,
            "confirmed": cache_confirmed,
            "messages": messages,
        },
        "parity": {
            "passed": passed,
            "reference": "NumPy complex128 unnormalized IFFT",
            "max_error_relative_to_reference_peak": max_relative_to_peak,
            "relative_l2": relative_l2,
            "tolerance": 2e-6,
            "input_preserved": bool(np.array_equal(source.numpy(), raw)),
        },
        "timing": {
            "unit": "seconds_per_ifft",
            "raw_samples": samples,
            "median": statistics.median(samples),
            "minimum": min(samples),
            "maximum": max(samples),
            "sample_count": len(samples),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(
        json.dumps(
            {
                "label": args.label,
                "status": result["status"],
                "output": str(args.output),
                "plan_seconds": plan_seconds,
                "median_seconds": statistics.median(samples),
                "plan": plan_info,
                "cache_imports": imports,
                "cache_exports": exports,
            }
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
