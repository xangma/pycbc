#!/usr/bin/env python3
"""Compare the two veto functions on identical inputs in one final environment.

Run on len only. This artifact does not modify production source. The baseline
file is an unmodified extraction of vetoes.py at 9ff3a7ec. This is a function
microbenchmark, not a pipeline benchmark or reference-FFT decision certificate.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


BASELINE_COMMIT = "9ff3a7ec5b5643fe7b0a3b94d082799c05d31775"
FINAL_COMMIT = "d0aa34d64cc848b00ebff5d3406a65449817f66e"
BASELINE_SHA256 = "bc3ea1e64b4ad797855bccd23d363489fff32c82f7905554afdce4227cd20a6a"
FINAL_SHA256 = "b0a5d8ee162a0e6816ea3f53a694c14b04e4522b343377f9dabe3e74e77ac3de"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-budget-bytes", type=int, default=64 * 1024**2)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if git(repo, "rev-parse", "HEAD") != FINAL_COMMIT:
        raise RuntimeError("Final checkout HEAD differs from the pinned commit")
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Final checkout contains tracked changes")
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    import pycbc.filter.gpu_search.vetoes as final_module

    final_path = Path(final_module.__file__).resolve()
    expected_path = repo / "pycbc/filter/gpu_search/vetoes.py"
    if final_path != expected_path or file_hash(final_path) != FINAL_SHA256:
        raise RuntimeError("Loaded final veto module has unexpected identity")
    baseline_path = Path(__file__).with_name("vetoes_baseline_9ff3a7ec.py")
    if file_hash(baseline_path) != BASELINE_SHA256:
        raise RuntimeError("Baseline veto module hash differs from git extraction")
    spec = importlib.util.spec_from_file_location(
        "_pycbc_baseline_vetoes_9ff3a7ec", baseline_path)
    baseline_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline_module
    spec.loader.exec_module(baseline_module)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This measurement requires an available CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)

    n, m, batch, bins, seed = 131072, 8, 4, 16, 20260912
    flen = n // 2 + 1
    rng = np.random.default_rng(seed)
    clean_corr = ((rng.standard_normal((batch, flen)) +
                   1j * rng.standard_normal((batch, flen))) /
                  np.sqrt(2)).astype(np.complex64)
    # Each row has the same bin count and a different lower cutoff. This tests
    # independent half-open ranges within one uniform bin-edge matrix.
    lower = np.array([31, 127, 511, 1023], dtype=np.int64)
    upper = np.full(batch, flen - 17, dtype=np.int64)
    edges = np.stack([np.linspace(lo, hi, bins + 1, dtype=np.int64)
                      for lo, hi in zip(lower, upper)])
    norms = (1 / np.sqrt(upper - lower)).astype(np.float32)
    ti = np.arange(m, dtype=np.int64) % batch
    si = np.array([0, 1, 37, 997, 4093, n // 2 + 123, n - 2, n - 1],
                  dtype=np.int64)

    def array_hash(value):
        value = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
        return digest.hexdigest()

    def oracle(corr):
        """Independent real/imaginary FP64 sums, separately within each bin."""
        bin_sums = np.zeros((m, bins), dtype=np.complex128)
        for row, (template, sample) in enumerate(zip(ti, si)):
            for bi, (lo, hi) in enumerate(zip(edges[template, :-1],
                                             edges[template, 1:])):
                k = np.arange(lo, hi, dtype=np.int64)
                angle = ((sample * k) % n).astype(np.float64) * (2 * np.pi / n)
                cosine, sine = np.cos(angle), np.sin(angle)
                re = corr[template, lo:hi].real.astype(np.float64)
                im = corr[template, lo:hi].imag.astype(np.float64)
                bin_sums[row, bi] = (
                    np.sum(re * cosine - im * sine, dtype=np.float64) +
                    1j * np.sum(re * sine + im * cosine, dtype=np.float64))
        snr = (bin_sums.sum(axis=1) * norms[ti]).astype(np.complex64)
        # Use the stored complex64 SNR in the oracle as both APIs receive it.
        expected = np.maximum(
            bins * np.sum(bin_sums.real**2 + bin_sums.imag**2, axis=1) *
            norms[ti].astype(np.float64)**2 -
            np.abs(snr.astype(np.complex128))**2, 0)
        return snr, expected, bin_sums

    report = {
        "schema": "power-chisq-identical-workload-v1",
        "scope": "Two batched_power_chisq functions in the final checkout environment",
        "not_measured": ["full search throughput", "candidate selection",
                         "physical waveform equivalence", "FFT-boundary decisions"],
        "source": {
            "baseline_commit": BASELINE_COMMIT, "final_commit": FINAL_COMMIT,
            "baseline_module": str(baseline_path.resolve()),
            "final_module": str(final_path),
            "baseline_module_sha256": BASELINE_SHA256,
            "final_module_sha256": FINAL_SHA256,
            "benchmark_sha256": file_hash(__file__),
        },
        "environment": {
            "host": platform.node(), "platform": platform.platform(),
            "cwd": os.getcwd(), "pid": os.getpid(), "argv": sys.argv,
            "python": sys.version, "executable": sys.executable,
            "numpy": np.__version__, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "torch_threads": torch.get_num_threads(),
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                    "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
        },
        "workload": {
            "seed": seed, "transform_length": n, "frequency_length": flen,
            "template_count": batch, "candidate_count": m, "num_bins": bins,
            "corr_dtype": "complex64", "norm_dtype": "float32",
            "snr_dtype": "complex64", "chunk_size": 2048,
            "snr_threshold": None, "output_boundary": "default NumPy outputs",
            "bins": "uniform count; different per-template lower cutoffs",
            "edges_sha256": array_hash(edges), "norms_sha256": array_hash(norms),
            "template_idx_sha256": array_hash(ti), "sample_idx_sha256": array_hash(si),
            "repetitions": 5, "warmup_calls_per_implementation_per_case": 2,
            "gate": {"rtol": 2e-5, "atol": 1e-5,
                     "dof": "exact", "shape": "exact", "finite": True},
        },
        "memory_measurement": {
            "quantity": "max_memory_allocated minus memory_allocated after caller input allocation",
            "includes": "all new live Torch allocations in the call, including output/index arrays",
            "excludes": "caller inputs, CUDA allocator caches, non-Torch allocations, host allocations",
            "not_a_total_device_memory_measurement": True,
            "new_requested_scratch_budget_bytes": args.scratch_budget_bytes,
            "new_budget_contract": "Explicit blocked intermediates only; excludes O(M) outputs/indices/SNR copies, allocator caches and backend workspace",
            "old_budget_contract": "No budget argument; internal 64 MiB heuristic counts 8 bytes per full-frequency candidate element, not all concurrent arrays",
            "old_effective_candidate_rows": min(m, 2048, max(1, 64 * 1024**2 // (8 * flen))),
            "old_frequency_width": flen,
            "new_effective_rows_and_frequency_width": list(
                final_module.power_chisq_scratch_shape(bins, min(2048, m),
                                                       args.scratch_budget_bytes)),
        },
        "timing_measurement": "Wall time with CUDA synchronization immediately before and after each call; includes the common NumPy output boundary; warmup and input/oracle creation excluded",
        "cases": [],
    }
    for package in ("pycbc", "lalsuite"):
        try:
            report["environment"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["environment"][package] = None

    clean_snr, clean_expected, _ = oracle(clean_corr)
    for case_name in ("seeded_clean", "excluded_prefix_contamination"):
        corr = clean_corr.copy()
        if case_name == "excluded_prefix_contamination":
            corr[:, 0] = np.complex64(1e15 + 5e14j)
        snr, expected, bin_sums = oracle(corr)
        if not (np.array_equal(snr, clean_snr) and
                np.array_equal(expected, clean_expected)):
            raise RuntimeError("Excluded-frequency contamination changed the oracle")
        caller = {
            "corr_tile": torch.as_tensor(corr, device=device),
            "candidates": {
                "template_idx": torch.as_tensor(ti, device=device),
                "sample_idx": torch.as_tensor(si, device=device),
                "snr": torch.as_tensor(snr, device=device),
            },
            "tile_bin_edges": torch.as_tensor(edges, device=device),
            "tile_norms": torch.as_tensor(norms, device=device),
            "num_bins": bins, "snr_threshold": None,
            "transform_length": n, "chunk_size": 2048,
        }
        torch.cuda.synchronize(device)
        case = {
            "name": case_name, "corr_sha256": array_hash(corr),
            "snr_sha256": array_hash(snr), "oracle_sha256": array_hash(expected),
            "oracle_bin_sums_sha256": array_hash(bin_sums),
            "oracle_chisq": expected.tolist(), "expected_dof": [2 * bins - 2] * m,
            "oracle_unchanged_from_clean": True, "implementations": {},
        }
        for label, module in (("baseline", baseline_module), ("final", final_module)):
            extra = ({"scratch_budget_bytes": args.scratch_budget_bytes}
                     if label == "final" else {})

            def invoke():
                with torch.inference_mode():
                    return module.batched_power_chisq(**caller, **extra)

            for _ in range(2):
                invoke()
            torch.cuda.synchronize(device)
            times, peaks, hashes = [], [], []
            for repeat in range(5):
                gc.collect()
                torch.cuda.synchronize(device)
                allocated_before = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                actual, dof = invoke()
                torch.cuda.synchronize(device)
                times.append(time.perf_counter() - start)
                peaks.append(torch.cuda.max_memory_allocated(device) - allocated_before)
                if not (isinstance(actual, np.ndarray) and isinstance(dof, np.ndarray)):
                    raise RuntimeError("An implementation did not honor NumPy output boundary")
                hashes.append({"chisq": array_hash(actual), "dof": array_hash(dof)})
                print(f"{case_name} {label} repeat={repeat + 1} seconds={times[-1]:.6f} peak={peaks[-1]}",
                      file=sys.stderr, flush=True)
            error = np.abs(actual.astype(np.float64) - expected)
            valid = bool(actual.shape == expected.shape and dof.shape == expected.shape
                         and np.all(np.isfinite(actual))
                         and np.all(dof == 2 * bins - 2)
                         and np.allclose(actual, expected, rtol=2e-5, atol=1e-5))
            unchanged = all(array_hash(value.detach().cpu().numpy()) == array_hash(source)
                            for value, source in (
                                (caller["corr_tile"], corr),
                                (caller["tile_bin_edges"], edges),
                                (caller["tile_norms"], norms),
                                (caller["candidates"]["template_idx"], ti),
                                (caller["candidates"]["sample_idx"], si),
                                (caller["candidates"]["snr"], snr)))
            if not unchanged:
                raise RuntimeError(f"{label} mutated shared caller inputs")
            case["implementations"][label] = {
                "scientific_gate_passed": valid, "caller_inputs_unchanged": unchanged,
                "chisq": actual.tolist(), "dof": dof.tolist(),
                "max_absolute_error": float(error.max()),
                "max_relative_error": float((error / np.maximum(np.abs(expected), 1e-30)).max()),
                "timings_seconds": times, "median_seconds": statistics.median(times),
                "peak_new_torch_allocation_bytes": peaks,
                "maximum_peak_new_torch_allocation_bytes": max(peaks),
                "repeat_output_hashes": hashes,
                "repeat_outputs_identical": all(value == hashes[0] for value in hashes),
            }
        old, new = (case["implementations"][name] for name in ("baseline", "final"))
        qualified = all(item["scientific_gate_passed"] and item["repeat_outputs_identical"]
                        for item in (old, new))
        case["performance_comparison_qualified"] = qualified
        case["baseline_over_final_median_time"] = (
            old["median_seconds"] / new["median_seconds"] if qualified else None)
        case["qualification_note"] = (
            "Both functions pass this workload's independent FP64 gate; ratio applies only to this microbenchmark"
            if qualified else
            "At least one scientific or repeatability gate failed; timings are diagnostic only and no speedup is claimed")
        report["cases"].append(case)
        del caller
        gc.collect()

    native = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith(("pycbc.", "lal.", "lalsimulation.")) and path and path.endswith(".so"):
            native[name] = {"path": str(Path(path).resolve()), "sha256": file_hash(path)}
    report["environment"]["loaded_pycbc_lal_native_modules"] = native
    report["all_final_scientific_gates_passed"] = all(
        case["implementations"]["final"]["scientific_gate_passed"] for case in report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(str(args.output.resolve()), flush=True)
    if not report["all_final_scientific_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
