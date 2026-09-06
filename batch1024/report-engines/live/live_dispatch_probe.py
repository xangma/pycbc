#!/usr/bin/env python3
"""Untimed native-dispatch observation using a current head's live workload.

Never include the child driver's instrumented timing output as benchmark data.
This probe preserves return values and observes actual successful admissions.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=131072)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def run_probe(args, result):
    root = args.source_root.resolve()
    actual_sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    result["revision"] = actual_sha
    assert actual_sha == args.expected_revision, (
        "Measured revision differs from expected"
    )
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    result["tracked_clean"] = not dirty
    assert not dirty, "Measured source has tracked modifications"
    for name in list(os.environ):
        if name.startswith("PYCBC_TORCH_") or name in (
            "PYCBC_ENABLE_CUDA_GRAPHS",
            "PYCBC_BATCH_MAXELEMENTS",
        ):
            os.environ.pop(name)
    os.environ.update(
        {
            "OMP_NUM_THREADS": str(args.threads),
            "MKL_NUM_THREADS": str(args.threads),
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
        }
    )
    sys.path.insert(0, str(root))
    from tools import bench_production_live_batch as bench

    assert Path(bench.__file__).resolve().parents[1] == root
    # Mirror _child: configure the chosen route before backend imports, including
    # native gates that modules may consult during their initial import.
    configured_environment = bench.route_environment(args.route, dict(os.environ))
    for name in bench.FEATURE_FLAG_DEFAULTS:
        os.environ.pop(name, None)
    os.environ.update(
        {
            name: value
            for name, value in configured_environment.items()
            if name in bench.FEATURE_FLAG_DEFAULTS
        }
    )
    result["configuration"] = bench.route_configuration(args.route)
    result["backend_import_environment"] = {
        name: os.environ.get(name) for name in bench.FEATURE_FLAG_DEFAULTS
    }
    from pycbc.filter import matchedfilter, matchedfilter_torch
    from pycbc.fft import torchfft
    import pycbc

    assert Path(pycbc.__file__).resolve().parent.parent == root
    result["pycbc_module"] = pycbc.__file__
    counts = result["counts"]
    originals = []
    result["batch_structure"] = []
    original_init = matchedfilter.LiveBatchMatchedFilter.__init__
    originals.append((matchedfilter.LiveBatchMatchedFilter, "__init__", original_init))

    def observe_init(instance, *positional, **keywords):
        original_init(instance, *positional, **keywords)
        result["batch_structure"].append(
            {
                "chunks": [int(value) for value in instance.chunks],
                "template_group_sizes": [len(group) for group in instance.tgroups],
                "fft_output_elements": [
                    len(value) for value in instance.out_mem.values()
                ],
                "fft_input_elements": [
                    len(value) for value in instance.cout_mem.values()
                ],
            }
        )

    matchedfilter.LiveBatchMatchedFilter.__init__ = observe_init

    def wrap(module, name, success):
        if not hasattr(module, name):
            counts[name] = {"available": False}
            return
        original = getattr(module, name)
        originals.append((module, name, original))
        counts[name] = {
            "available": True,
            "attempts": 0,
            "successes": 0,
            "fallbacks": 0,
            "exceptions": 0,
        }

        def observe(*a, **kw):
            row = counts[name]
            row["attempts"] += 1
            try:
                result = original(*a, **kw)
            except Exception:
                row["exceptions"] += 1
                raise
            admitted = success(result)
            row["successes"] += int(admitted)
            row["fallbacks"] += int(not admitted)
            return result

        setattr(module, name, observe)

    for name in ("_try_cpu_native_batch_correlate", "_try_cuda_native_batch_correlate"):
        wrap(matchedfilter_torch, name, bool)
    for name in (
        "_try_torch_cpu_native_batch_peak_values",
        "_try_torch_cuda_native_batch_peak_values",
    ):
        wrap(matchedfilter, name, lambda value: value is not None)
    for name in (
        "_execute_fftw_cpu_batch_plan",
        "_execute_fftw_cpu_plan",
        "_execute_mkl_cpu_ifft_plan",
    ):
        wrap(torchfft, name, bool)
    # The unsafe optional helper must remain uncalled under these route definitions.
    wrap(
        matchedfilter_torch,
        "_torch_batch_peak_and_threshold_gpu",
        lambda value: value is not None,
    )
    try:
        bench._child(
            argparse.Namespace(
                source_root=str(root),
                route=args.route,
                threads=args.threads,
                cuda_device=args.cuda_device,
                batch=args.batch,
                size=args.size,
                num_blocks=3,
                samples=3,
                warmups=0,
                snr_threshold=5.5,
                seed=7101,
                call_surface="public",
            )
        )
    finally:
        for module, name, original in originals:
            setattr(module, name, original)
    assert not matchedfilter._torch_ondevice_peaks_enabled()
    assert counts["_torch_batch_peak_and_threshold_gpu"]["attempts"] == 0
    result["unsafe_helper_excluded"] = True
    assert len(result["batch_structure"]) == 1
    structure = result["batch_structure"][0]
    assert structure["chunks"] == [args.batch], structure
    assert structure["template_group_sizes"] == [args.batch], structure
    assert structure["fft_output_elements"] == [args.batch * args.size], structure
    assert structure["fft_input_elements"] == [args.batch * args.size], structure
    final_sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    final_dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    assert final_sha == args.expected_revision and not final_dirty
    result["source_unchanged_after"] = True


def main():
    args = parse_args()
    assert not args.output.exists(), "Refusing to replace an existing artifact"
    result = {
        "status": "running",
        "scope": "Untimed instrumentation only; child timings are diagnostic",
        "expected_revision": args.expected_revision,
        "source_root": str(args.source_root.resolve()),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "route": args.route,
        "threads": args.threads,
        "batch": args.batch,
        "size": args.size,
        "cuda_device": args.cuda_device,
        "counts": {},
    }
    error = None
    try:
        run_probe(args, result)
        result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        error = exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print("DISPATCH_PROBE=" + json.dumps(result, sort_keys=True))
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
