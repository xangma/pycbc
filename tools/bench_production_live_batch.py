#!/usr/bin/env python3
"""Counterbalanced production LiveBatchMatchedFilter benchmark.

Benchmarks the production ``LiveBatchMatchedFilter._process_batch`` pipeline across 4 routes:
- Route A (original_standard): Original PyCBC (standard CPU)
- Route B (branch_standard): Branch Standard CPU
- Route C (torch_cpu): Torch CPU (with native batching enabled)
- Route D (torch_cuda): Torch CUDA (with native batching enabled)

Workload: N=131072, batch sizes B in {1, 2, 4, 8, 16, 32}, with multiple blocks of
strain data, PSD/sigma caching, threshold decisions, and SNR trigger lists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

try:
    from tools.benchmark_artifact import (
        SCHEMA_VERSION,
        atomic_write_json,
        file_sha256,
        runtime_metadata,
        sample_summary,
        seal_artifact,
        source_identity,
    )
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from benchmark_artifact import (
        SCHEMA_VERSION,
        atomic_write_json,
        file_sha256,
        runtime_metadata,
        sample_summary,
        seal_artifact,
        source_identity,
    )


ROUTE_NAMES = (
    "original_standard",
    "branch_standard",
    "torch_cpu",
    "torch_cuda",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _at_least_three_int(value: str) -> int:
    parsed = int(value)
    if parsed < 3:
        raise argparse.ArgumentTypeError("value must be at least three")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _summary(values: list[float], unit: str) -> dict:
    return sample_summary(values, unit=unit, bootstrap_seed=7101)


def _route_source(root: Path, route: str) -> Path:
    if route == "original_standard":
        candidate = root / "original"
    elif route == "branch_standard":
        candidate = root / "branch"
        if not candidate.exists():
            candidate = root / "branch_cpu"
    elif route == "torch_cpu":
        candidate = root / "branch_cpu"
        if not candidate.exists():
            candidate = root / "branch"
    elif route == "torch_cuda":
        candidate = root / "branch_cuda"
        if not candidate.exists():
            candidate = root / "branch"
    else:
        candidate = root

    return candidate if candidate.exists() else root


def _child(args: argparse.Namespace) -> None:
    source_root = str(Path(args.source_root).resolve())
    sys.path.insert(0, source_root)

    import numpy as np

    import pycbc
    from pycbc import scheme
    from pycbc.filter import matchedfilter
    from pycbc.types import FrequencySeries

    is_cuda = args.route == "torch_cuda"
    is_torch_cpu = args.route == "torch_cpu"
    is_torch = is_cuda or is_torch_cpu

    torch_module = None
    if is_torch:
        import torch

        torch_module = torch
        torch.set_grad_enabled(False)
        if is_cuda:
            torch.cuda.set_device(args.cuda_device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            context = scheme.TorchScheme(f"cuda:{args.cuda_device}")
        else:
            torch.set_num_threads(args.threads)
            os.environ["PYCBC_TORCH_CPU_NATIVE_BATCH_CORRELATE"] = "1"
            os.environ["PYCBC_TORCH_CPU_FFTW_BATCH"] = "1"
            os.environ["PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK"] = "1"
            context = scheme.TorchScheme("cpu")
    else:
        context = scheme.CPUScheme(num_threads=args.threads)

    size = args.size
    freq_len = size // 2 + 1
    sample_rate = 2048.0
    delta_f = sample_rate / size
    batch_size = args.batch
    num_blocks = args.num_blocks
    trim_padding = 4096
    blocksize = 56.0
    snr_threshold = args.snr_threshold

    rng = np.random.default_rng(args.seed)

    # 1. Construct deterministic PSD
    psd_data = np.ones(freq_len, dtype=np.float32)
    freqs = np.linspace(0, sample_rate / 2.0, freq_len, endpoint=True)
    with np.errstate(divide="ignore"):
        psd_shape = np.clip(
            (np.maximum(freqs, delta_f) / 100.0) ** (-1.0) + (freqs / 500.0) ** 2.0,
            0.1,
            100.0,
        )
    psd_data *= psd_shape.astype(np.float32)

    # 2. Construct templates
    k_min = max(1, int(30.0 / delta_f))
    k_max = min(freq_len - 1, int(800.0 / delta_f))
    templates_np = []
    for _i in range(batch_size):
        t_raw = np.zeros(freq_len, dtype=np.complex64)
        t_raw[k_min:k_max] = (
            rng.normal(size=k_max - k_min) + 1j * rng.normal(size=k_max - k_min)
        ).astype(np.complex64)
        sgm_val = float(4.0 * delta_f * np.sum(np.abs(t_raw) ** 2 / psd_data))
        t_raw /= np.sqrt(sgm_val)
        templates_np.append(t_raw)

    # 3. Construct multiple strain data blocks with controlled signal injections
    strain_blocks_np = []
    injection_metadata = []
    for b in range(num_blocks):
        noise = (rng.normal(size=freq_len) + 1j * rng.normal(size=freq_len)).astype(
            np.complex64
        ) * 0.001
        inj_tmpl_idx = b % batch_size
        inj_snr = 8.0 + float(b % 4) * 1.5
        t0 = 20000 + (b * 1337) % 50000
        phase_shift = np.exp(-2j * np.pi * np.arange(freq_len) * t0 / size)
        noise += (templates_np[inj_tmpl_idx] * inj_snr * phase_shift).astype(
            np.complex64
        )
        strain_blocks_np.append(noise)
        injection_metadata.append(
            {
                "block": b,
                "injected_template_idx": inj_tmpl_idx,
                "injected_template_id": 1000 + inj_tmpl_idx,
                "expected_snr": inj_snr,
                "offset_sample": t0,
            }
        )

    def _sync():
        if is_cuda and torch_module is not None:
            torch_module.cuda.synchronize()

    with context:
        psd = FrequencySeries(psd_data, delta_f=delta_f)

        # Build templates with sigmasq caching
        templates = []
        sigmasq_eval_counts = [0] * batch_size
        for i in range(batch_size):
            t = FrequencySeries(templates_np[i], delta_f=delta_f)
            t.id = int(1000 + i)
            t.params = np.array(
                [(10.0 + i * 0.5, 10.0 + i * 0.5)],
                dtype=[("mass1", np.float32), ("mass2", np.float32)],
            )[0]

            cache_dict = {}

            def _make_sigmasq(idx, raw_array, cache):
                def _sigmasq(p):
                    sigmasq_eval_counts[idx] += 1
                    key = id(p)
                    if key not in cache:
                        p_arr = np.asarray(p)
                        cache[key] = float(
                            4.0 * delta_f * np.sum(np.abs(raw_array) ** 2 / p_arr)
                        )
                    return cache[key]

                return _sigmasq

            t.sigmasq = _make_sigmasq(i, templates_np[i], cache_dict)
            templates.append(t)

        _sync()
        alloc_start = time.perf_counter_ns()
        batch_filter = matchedfilter.LiveBatchMatchedFilter(
            templates,
            snr_threshold=snr_threshold,
            chisq_bins=0,
            sg_chisq=types.SimpleNamespace(),
            maxelements=batch_size * size,
        )
        _sync()
        allocation_setup_ns = time.perf_counter_ns() - alloc_start

        # Prepare data readers for all blocks
        data_readers = []
        for b in range(num_blocks):
            stilde = FrequencySeries(strain_blocks_np[b], delta_f=delta_f)
            stilde.psd = psd
            reader = types.SimpleNamespace(
                overwhitened_data=lambda _df, s=stilde: s,
                trim_padding=trim_padding,
                blocksize=blocksize,
                sample_rate=sample_rate,
                start_time=1000000000.0 + b * blocksize,
            )
            data_readers.append(reader)

        def _run_block(block_idx: int):
            batch_filter.set_data(data_readers[block_idx])
            _sync()
            t0 = time.perf_counter_ns()
            res, veto = batch_filter._process_batch()
            _sync()
            t1 = time.perf_counter_ns()
            return t1 - t0, res, veto

        def _run_all_blocks():
            latencies_ns = []
            block_outputs = []
            for b in range(num_blocks):
                elapsed_ns, res, veto = _run_block(b)
                latencies_ns.append(elapsed_ns)
                block_outputs.append((res, veto))
            return latencies_ns, block_outputs

        # 1. Cold pass
        cold_block_latencies_ns, cold_outputs = _run_all_blocks()
        cold_total_ns = sum(cold_block_latencies_ns)
        cold_sigmasq_evals = list(sigmasq_eval_counts)

        # 2. Warmups
        for _ in range(args.warmups):
            _run_all_blocks()

        # 3. Warm measurement samples
        warm_iteration_latencies_ns = []
        warm_block_latencies_ns = []
        last_block_outputs = None
        for _ in range(args.samples):
            iter_block_latencies, last_block_outputs = _run_all_blocks()
            warm_block_latencies_ns.extend(iter_block_latencies)
            warm_iteration_latencies_ns.append(sum(iter_block_latencies))

        # Memory & output checksums
        mid = batch_filter.mids[0]
        out_mem = batch_filter.out_mem[mid]
        if hasattr(out_mem, "_data") and hasattr(out_mem._data, "tensor"):
            out_np = (
                out_mem._data.tensor.detach()
                .cpu()
                .numpy()
                .reshape(batch_size, size)
                .copy()
            )
        else:
            out_np = out_mem.numpy().reshape(batch_size, size).copy()

        output_sha256 = hashlib.sha256(out_np.tobytes()).hexdigest()
        output_l2 = float(np.linalg.norm(out_np.astype(np.complex128)))

        peak_memory_bytes = (
            torch_module.cuda.max_memory_allocated()
            if is_cuda and torch_module
            else None
        )

        # Extract structured trigger lists from last execution
        block_triggers = []
        for b, (res, veto) in enumerate(last_block_outputs):
            block_triggers.append(
                {
                    "block": b,
                    "num_triggers": int(len(res["snr"])),
                    "template_ids": [int(x) for x in res["template_id"]],
                    "snrs": [float(x) for x in res["snr"]],
                    "end_times": [float(x) for x in res["end_time"]],
                    "coa_phases": [float(x) for x in res["coa_phase"]],
                    "sigmasqs": [float(x) for x in res["sigmasq"]],
                    "veto_count": int(len(veto)),
                }
            )

        waveforms_per_iteration = batch_size * num_blocks
        throughputs_wps = [
            waveforms_per_iteration / (ns / 1.0e9) for ns in warm_iteration_latencies_ns
        ]
        warm_block_latencies_ms = [ns / 1.0e6 for ns in warm_block_latencies_ns]
        warm_iter_latencies_ms = [ns / 1.0e6 for ns in warm_iteration_latencies_ns]
        cold_block_latencies_ms = [ns / 1.0e6 for ns in cold_block_latencies_ns]

        record = {
            "route": args.route,
            "source_root": source_root,
            "batch": batch_size,
            "size": size,
            "num_blocks": num_blocks,
            "threads": args.threads,
            "seed": args.seed,
            "snr_threshold": snr_threshold,
            "allocation_setup_ms": allocation_setup_ns / 1.0e6,
            "cold_block_latencies_ms": cold_block_latencies_ms,
            "cold_total_ms": cold_total_ns / 1.0e6,
            "cold_sigmasq_evals": cold_sigmasq_evals,
            "total_sigmasq_evals": list(sigmasq_eval_counts),
            "warm_block_latencies_ms": warm_block_latencies_ms,
            "warm_iteration_latencies_ms": warm_iter_latencies_ms,
            "latency_block_ms_summary": _summary(warm_block_latencies_ms, "ms/block"),
            "latency_iteration_ms_summary": _summary(
                warm_iter_latencies_ms, "ms/iteration"
            ),
            "latency_per_waveform_ms_summary": _summary(
                [ms / float(waveforms_per_iteration) for ms in warm_iter_latencies_ms],
                "ms/waveform",
            ),
            "throughput_wps_summary": _summary(throughputs_wps, "waveforms/second"),
            "output_sha256": output_sha256,
            "output_l2": output_l2,
            "peak_memory_bytes": peak_memory_bytes,
            "block_triggers": block_triggers,
            "injection_metadata": injection_metadata,
            "python": sys.version,
            "pycbc_version": getattr(pycbc, "__version__", None),
            "numpy_version": np.__version__,
            "torch_version": getattr(torch_module, "__version__", None)
            if torch_module
            else None,
            "cuda_device": args.cuda_device if is_cuda else None,
            "cuda_device_name": torch_module.cuda.get_device_name(args.cuda_device)
            if is_cuda and torch_module
            else None,
            "pid": os.getpid(),
        }

        print("RESULT_JSON=" + json.dumps(record, sort_keys=True))


def _run_child(
    python_bin: str,
    script_path: Path,
    route: str,
    source_root: Path,
    batch: int,
    size: int,
    num_blocks: int,
    threads: int,
    samples: int,
    warmups: int,
    snr_threshold: float,
    cuda_device: int,
    seed: int,
    affinity: str,
) -> dict:
    command = [
        python_bin,
        str(script_path),
        "child",
        "--route",
        route,
        "--source-root",
        str(source_root),
        "--batch",
        str(batch),
        "--size",
        str(size),
        "--num-blocks",
        str(num_blocks),
        "--threads",
        str(threads),
        "--samples",
        str(samples),
        "--warmups",
        str(warmups),
        "--snr-threshold",
        str(snr_threshold),
        "--cuda-device",
        str(cuda_device),
        "--seed",
        str(seed),
    ]
    if affinity:
        command = ["taskset", "-c", affinity, *command]

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(source_root),
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    if route == "torch_cpu":
        environment.update(
            {
                "PYCBC_TORCH_CPU_NATIVE_BATCH_CORRELATE": "1",
                "PYCBC_TORCH_CPU_FFTW_BATCH": "1",
                "PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK": "1",
            }
        )

    completed = subprocess.run(
        command,
        cwd=str(source_root),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Child process failed ({completed.returncode}) for route={route}, batch={batch}:\n"
            f"command: {command!r}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    lines = [
        line.removeprefix("RESULT_JSON=")
        for line in completed.stdout.splitlines()
        if line.startswith("RESULT_JSON=")
    ]
    if len(lines) != 1:
        raise RuntimeError(
            f"Child emitted {len(lines)} result records for route={route}, batch={batch}:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    result = json.loads(lines[0])
    result["stderr"] = completed.stderr
    result["command"] = command
    return result


def _counterbalanced(routes: list[str], replicate: int, cell_index: int) -> list[str]:
    shift = (replicate + cell_index) % len(routes)
    order = list(routes[shift:] + routes[:shift])
    if replicate % 2 == 1:
        order.reverse()
    return order


def _verify_parity(records_by_batch: dict[int, dict[str, dict]]) -> dict:
    """Verify trigger parity and waveform parity across all 4 routes."""
    import numpy as np

    parity_results = {}
    all_passed_globally = True

    for batch, route_records in records_by_batch.items():
        batch_parity = {
            "batch": batch,
            "routes_evaluated": list(route_records.keys()),
            "comparisons": {},
            "all_passed": True,
        }

        ref_route = (
            "original_standard"
            if "original_standard" in route_records
            else "branch_standard"
        )
        ref_record = route_records[ref_route]
        ref_triggers = ref_record["block_triggers"]
        ref_output_l2 = ref_record["output_l2"]

        for route, record in route_records.items():
            if route == ref_route:
                continue

            triggers = record["block_triggers"]
            output_l2 = record["output_l2"]

            block_count_match = len(ref_triggers) == len(triggers)
            trigger_count_match = True
            template_id_match = True
            end_time_match = True
            max_snr_diff = 0.0
            max_phase_diff = 0.0
            max_sigmasq_rel_diff = 0.0
            veto_count_match = True
            sequence_length_match = True

            for b in range(min(len(ref_triggers), len(triggers))):
                ref_b = ref_triggers[b]
                test_b = triggers[b]

                if ref_b["num_triggers"] != test_b["num_triggers"]:
                    trigger_count_match = False
                if ref_b["template_ids"] != test_b["template_ids"]:
                    template_id_match = False
                if ref_b["veto_count"] != test_b["veto_count"]:
                    veto_count_match = False

                sequence_fields = ("end_times", "snrs", "coa_phases", "sigmasqs")
                if any(
                    len(ref_b[field]) != len(test_b[field]) for field in sequence_fields
                ):
                    sequence_length_match = False
                    end_time_match = False

                for ref_et, test_et in zip(
                    ref_b["end_times"], test_b["end_times"], strict=False
                ):
                    if abs(ref_et - test_et) > 1e-4:
                        end_time_match = False

                for ref_s, test_s in zip(ref_b["snrs"], test_b["snrs"], strict=False):
                    diff = abs(ref_s - test_s)
                    if diff > max_snr_diff:
                        max_snr_diff = diff

                for ref_p, test_p in zip(
                    ref_b["coa_phases"], test_b["coa_phases"], strict=False
                ):
                    p_diff = abs((ref_p - test_p + np.pi) % (2.0 * np.pi) - np.pi)
                    if p_diff > max_phase_diff:
                        max_phase_diff = p_diff

                for ref_sigma, test_sigma in zip(
                    ref_b["sigmasqs"], test_b["sigmasqs"], strict=False
                ):
                    sigma_diff = abs(ref_sigma - test_sigma) / max(
                        abs(ref_sigma), 1e-12
                    )
                    if sigma_diff > max_sigmasq_rel_diff:
                        max_sigmasq_rel_diff = sigma_diff

            rel_l2_diff = abs(output_l2 - ref_output_l2) / max(ref_output_l2, 1e-12)

            passed = (
                block_count_match
                and trigger_count_match
                and template_id_match
                and end_time_match
                and sequence_length_match
                and veto_count_match
                and max_snr_diff < 1e-3
                and max_phase_diff < 1e-3
                and max_sigmasq_rel_diff < 1e-3
                and rel_l2_diff < 1e-3
            )

            if not passed:
                batch_parity["all_passed"] = False
                all_passed_globally = False

            batch_parity["comparisons"][f"{route}_vs_{ref_route}"] = {
                "block_count_match": block_count_match,
                "trigger_count_match": trigger_count_match,
                "template_id_match": template_id_match,
                "end_time_match": end_time_match,
                "sequence_length_match": sequence_length_match,
                "veto_count_match": veto_count_match,
                "max_snr_diff": float(max_snr_diff),
                "max_phase_diff": float(max_phase_diff),
                "max_sigmasq_relative_diff": float(max_sigmasq_rel_diff),
                "relative_output_l2_diff": float(rel_l2_diff),
                "passed": passed,
            }

        parity_results[f"batch_{batch}"] = batch_parity

    parity_results["all_passed_globally"] = all_passed_globally
    return parity_results


def _verify_replicate_parity(records: list[dict], batches: list[int]) -> dict:
    """Run the full parity comparison for every independent replicate."""

    replicate_ids = sorted({int(record["replicate"]) for record in records})
    grouped = {
        replicate: {batch: {} for batch in batches} for replicate in replicate_ids
    }
    for record in records:
        grouped[int(record["replicate"])][int(record["batch"])][record["route"]] = (
            record
        )

    replicate_results = {
        f"replicate_{replicate}": _verify_parity(grouped[replicate])
        for replicate in replicate_ids
    }
    result = {
        "replicates": replicate_results,
        "all_passed_globally": all(
            parity["all_passed_globally"] for parity in replicate_results.values()
        ),
    }
    for batch in batches:
        batch_replicates = {
            name: parity[f"batch_{batch}"] for name, parity in replicate_results.items()
        }
        result[f"batch_{batch}"] = {
            "batch": batch,
            "replicates": batch_replicates,
            "all_passed": all(
                parity["all_passed"] for parity in batch_replicates.values()
            ),
        }
    return result


def _orchestrate(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    script_path = Path(__file__).resolve()
    output_path = Path(args.output).resolve()
    batches = list(args.batches)
    routes = list(args.routes)
    if len(routes) < 2:
        raise ValueError("at least two routes are required for parity evidence")
    if not {"original_standard", "branch_standard"}.intersection(routes):
        raise ValueError(
            "routes must include original_standard or branch_standard as a "
            "parity reference"
        )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output_path}; pass --overwrite explicitly"
        )

    route_sources = {route: _route_source(root, route).resolve() for route in routes}
    if "original_standard" in route_sources:
        original_source = route_sources["original_standard"]
        colliding = [
            route
            for route, source in route_sources.items()
            if route != "original_standard" and source == original_source
        ]
        if colliding:
            raise ValueError(
                "original_standard must use a distinct source checkout; shared "
                f"with {', '.join(colliding)}"
            )

    identities_by_path = {}
    source_identities = {}
    for route, source in route_sources.items():
        if not (source / "pycbc" / "__init__.py").is_file():
            raise FileNotFoundError(
                f"route {route} does not resolve to a PyCBC checkout: {source}"
            )
        source_key = str(source)
        if source_key not in identities_by_path:
            identities_by_path[source_key] = source_identity(source)
        identity = identities_by_path[source_key]
        if identity.get("revision") is None:
            raise RuntimeError(f"cannot identify source revision for {route}: {source}")
        if identity.get("dirty") and not args.allow_dirty_source:
            raise RuntimeError(
                f"route {route} has a dirty source tree at {source}; commit or "
                "stash it, or pass --allow-dirty-source for a diagnostic run"
            )
        source_identities[route] = identity

    started_utc = datetime.now(timezone.utc)
    records = []

    print(
        "================================================================================"
    )
    print(f"Starting Production Live Batch Benchmark across {len(routes)} routes")
    print(f"Batch sizes: {batches}, Size: {args.size}, Blocks: {args.num_blocks}")
    print(
        f"Replicates: {args.replicates}, Samples: {args.samples}, Warmups: {args.warmups}"
    )
    print(f"Root: {root}")
    print(
        "================================================================================",
        flush=True,
    )

    cell_index = 0
    for replicate in range(args.replicates):
        for batch in batches:
            order = _counterbalanced(routes, replicate, cell_index)
            seed_base = args.seed + 10000 * replicate + 100 * cell_index

            for route in order:
                source_root = route_sources[route]
                seed = seed_base
                print(
                    f"[Rep {replicate + 1}/{args.replicates}] Running batch={batch:2d} route={route:18s} ...",
                    end="",
                    flush=True,
                )
                t0 = time.perf_counter()
                record = _run_child(
                    python_bin=args.python,
                    script_path=script_path,
                    route=route,
                    source_root=source_root,
                    batch=batch,
                    size=args.size,
                    num_blocks=args.num_blocks,
                    threads=args.threads,
                    samples=args.samples,
                    warmups=args.warmups,
                    snr_threshold=args.snr_threshold,
                    cuda_device=args.cuda_device,
                    seed=seed,
                    affinity=args.affinity,
                )
                elapsed = time.perf_counter() - t0
                record.update(
                    {
                        "replicate": replicate,
                        "cell_index": cell_index,
                        "execution_order": order,
                    }
                )
                records.append(record)

                tp = record["throughput_wps_summary"]["median"]
                lat = record["latency_block_ms_summary"]["median"]
                print(
                    f" done in {elapsed:5.1f}s | Lat: {lat:7.3f} ms/blk | Tput: {tp:9.2f} wf/s",
                    flush=True,
                )

            cell_index += 1

    records_by_route_batch: dict[str, dict[int, list[dict]]] = {
        route: {batch: [] for batch in batches} for route in routes
    }
    for rec in records:
        records_by_route_batch[rec["route"]][rec["batch"]].append(rec)

    parity_analysis = _verify_replicate_parity(records, batches)

    reference_route = (
        "original_standard" if "original_standard" in routes else "branch_standard"
    )
    summary_table = {}
    for batch in batches:
        batch_summary = {}
        reference_by_replicate = {
            record["replicate"]: record
            for record in records_by_route_batch[reference_route][batch]
        }
        for route in routes:
            recs = records_by_route_batch[route][batch]
            all_throughputs = [r["throughput_wps_summary"]["median"] for r in recs]
            all_latencies_ms = [r["latency_block_ms_summary"]["median"] for r in recs]
            paired_speedups = [
                record["throughput_wps_summary"]["median"]
                / reference_by_replicate[record["replicate"]]["throughput_wps_summary"][
                    "median"
                ]
                for record in recs
            ]
            route_seed = args.seed + batch * 100 + routes.index(route)
            throughput_summary = sample_summary(
                all_throughputs,
                unit="waveforms/second",
                bootstrap_seed=route_seed,
            )
            latency_summary = sample_summary(
                all_latencies_ms,
                unit="ms/block",
                bootstrap_seed=route_seed + 1,
            )
            speedup_summary = sample_summary(
                paired_speedups,
                unit=f"ratio-vs-{reference_route}",
                bootstrap_seed=route_seed + 2,
            )
            batch_summary[route] = {
                "throughput_wps": throughput_summary,
                "latency_block_ms": latency_summary,
                "speedup_vs_reference": speedup_summary,
                "throughput_wps_median": throughput_summary["median"],
                "throughput_wps_mean": throughput_summary["mean"],
                "throughput_wps_p25": throughput_summary["p25"],
                "throughput_wps_p75": throughput_summary["p75"],
                "latency_block_median_ms": latency_summary["median"],
                "latency_block_p25_ms": latency_summary["p25"],
                "latency_block_p75_ms": latency_summary["p75"],
                "latency_per_waveform_median_ms": (latency_summary["median"] / batch),
            }

        summary_table[f"batch_{batch}"] = batch_summary

    source_hashes = {}
    for route in routes:
        source = route_sources[route]
        for rel_file in (
            "pycbc/filter/matchedfilter.py",
            "pycbc/filter/matchedfilter_cpu.pyx",
            "pycbc/filter/matchedfilter_torch.py",
            "pycbc/fft/fftw.py",
            "pycbc/fft/torchfft.py",
        ):
            file_path = source / rel_file
            if file_path.exists():
                source_hashes[f"{route}:{rel_file}"] = file_sha256(file_path)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "schema": SCHEMA_VERSION,
        "artifact_type": "production_live_batch",
        "benchmark": "production_live_batch",
        "started_utc": started_utc.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "root": str(root),
        "launcher_sha256": file_sha256(script_path),
        "runtime": runtime_metadata(),
        "args": vars(args),
        "routes": routes,
        "batches": batches,
        "reference_route": reference_route,
        "source_sha256": source_hashes,
        "source_identities": source_identities,
        "summary_by_batch": summary_table,
        "parity_analysis": parity_analysis,
        "records": records,
    }

    atomic_write_json(output_path, seal_artifact(payload))
    print(f"\nSuccessfully wrote benchmark results to {output_path}")

    print("\n" + "=" * 90)
    print("PRODUCTION LIVE BATCH BENCHMARK SUMMARY (N=131072)")
    print("=" * 90)
    print(
        f"{'Batch':>5} | {'Route':<18} | {'Latency (ms/blk)':>16} | "
        f"{'Per-WF (ms)':>12} | {'Tput (wf/s)':>14} | "
        f"{'Paired speedup':>14}"
    )
    print("-" * 90)
    for batch in batches:
        b_sum = summary_table[f"batch_{batch}"]
        for route in routes:
            lat = b_sum[route]["latency_block_median_ms"]
            pwf = b_sum[route]["latency_per_waveform_median_ms"]
            tput = b_sum[route]["throughput_wps_median"]
            sp = b_sum[route]["speedup_vs_reference"]["median"]
            print(
                f"{batch:5d} | {route:<18} | {lat:16.3f} | {pwf:12.3f} | {tput:14.2f} | {sp:13.2f}x"
            )
        print("-" * 90)

    print("\nPARITY VERIFICATION:")
    for batch in batches:
        p = parity_analysis[f"batch_{batch}"]
        status = "PASSED" if p["all_passed"] else "FAILED"
        print(f"  Batch {batch:2d}: {status}")
        for replicate_name, replicate in p["replicates"].items():
            for comp_name, comp in replicate["comparisons"].items():
                print(
                    f"    - {replicate_name}/{comp_name}: "
                    f"SNR max delta={comp['max_snr_diff']:.2e}, "
                    f"Rel L2 diff={comp['relative_output_l2_diff']:.2e}, "
                    f"Triggers match={comp['trigger_count_match']}"
                )
    print("=" * 90)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LiveBatchMatchedFilter production benchmark."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    child = subparsers.add_parser("child", help="Run single isolated worker cell.")
    child.add_argument("--route", choices=ROUTE_NAMES, required=True)
    child.add_argument("--source-root", required=True)
    child.add_argument("--batch", type=_positive_int, required=True)
    child.add_argument("--size", type=_positive_int, default=131072)
    child.add_argument("--num-blocks", type=_positive_int, default=5)
    child.add_argument("--threads", type=_positive_int, default=1)
    child.add_argument("--samples", type=_at_least_three_int, default=5)
    child.add_argument("--warmups", type=_nonnegative_int, default=2)
    child.add_argument("--snr-threshold", type=float, default=5.5)
    child.add_argument("--cuda-device", type=_nonnegative_int, default=0)
    child.add_argument("--seed", type=int, default=7101)

    orchestrate = subparsers.add_parser(
        "orchestrate", help="Run full counterbalanced orchestrator."
    )
    orchestrate.add_argument(
        "--root", required=True, help="Root folder containing original, branch, etc."
    )
    orchestrate.add_argument(
        "--python", required=True, help="Python executable to invoke workers."
    )
    orchestrate.add_argument(
        "--output", required=True, help="Path for output structured JSON."
    )
    orchestrate.add_argument(
        "--routes", nargs="+", choices=ROUTE_NAMES, default=list(ROUTE_NAMES)
    )
    orchestrate.add_argument(
        "--batches",
        nargs="+",
        type=_positive_int,
        default=[1, 2, 4, 8, 16, 32],
    )
    orchestrate.add_argument("--size", type=_positive_int, default=131072)
    orchestrate.add_argument("--num-blocks", type=_positive_int, default=5)
    orchestrate.add_argument("--threads", type=_positive_int, default=1)
    orchestrate.add_argument("--replicates", type=_at_least_three_int, default=3)
    orchestrate.add_argument("--samples", type=_at_least_three_int, default=5)
    orchestrate.add_argument("--warmups", type=_nonnegative_int, default=2)
    orchestrate.add_argument("--snr-threshold", type=float, default=5.5)
    orchestrate.add_argument("--cuda-device", type=_nonnegative_int, default=0)
    orchestrate.add_argument("--affinity", default="")
    orchestrate.add_argument("--seed", type=int, default=7101)
    orchestrate.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Permit diagnostic evidence from dirty source trees",
    )
    orchestrate.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output artifact",
    )

    return parser


if __name__ == "__main__":
    parsed = _parser().parse_args()
    if parsed.mode == "child":
        _child(parsed)
    else:
        _orchestrate(parsed)
