#!/usr/bin/env python3
"""Validate the complete paired campaign and export offline JSON/Markdown/figures.

Usage: python build-report.py --input comparison --out report

The output directory must not exist. No source checkout is imported or modified.
Chart contract: before/after capacity or generation rate against six batch sizes, using
semilog-x small multiples. Each point is the median of three worker medians;
bands are the full worker range, not confidence intervals or pooled samples.
Before is neutral/dashed/open; after is blue/solid/filled. Live columns share
y limits between default and native-enabled routes. Waveform panels have their
own y limits. PNG and SVG are the requested standalone publication surfaces.
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BATCHES = (1, 8, 32, 128, 512, 1024)
REPLICATES = (1, 2, 3)
SOURCES = ("baseline", "candidate")
BLOCK_SECONDS = 56.0
SAMPLE_RATE_HZ = 2048.0
CAPACITY_UNITS = {
    "cpu": "templates/core at real time",
    "cuda": "templates/GPU at real time",
}
# Independently read from the frozen Git trees. The optional CPU harness adds
# routing flags but retains the same block duration and public data construction.
CAPACITY_HARNESSES = {
    "079ab813e3ed76902b007882611cf495eb2c63b8f75560a0666d956c350fdffe": {
        "git_blob": "673df95cba04f49c5704c150b0f3c9f061ce82bb",
        "sample_rate_line": 323,
        "blocksize_line": 328,
        "public_data_start_line": 438,
    },
    "8503f204a42e8a6e1af0efd29c081e4c07237771168ed4bf3d8b24eb1bfa16de": {
        "git_blob": "acc96b8cd0508e01373c4c35ab521d20b3f3d84e",
        "sample_rate_line": 326,
        "blocksize_line": 331,
        "public_data_start_line": 441,
    },
}
# Each tuple pins tree, harness SHA-256, matchedfilter SHA-256, its Git blob,
# and the first line of the valid-sample interval calculation.
CAPACITY_SOURCES = {
    "dfd42bf76766cadca0eecf609a1eaeac73534676": (
        "b0d5ef6e3b39bfe246b4c416528839256edadf1f",
        "079ab813e3ed76902b007882611cf495eb2c63b8f75560a0666d956c350fdffe",
        "8d40acb0d70547bac98464a165b775d4943d822de21e03e17bbde7fff518cc2e",
        "140397eabdb5e4ea4dfe5272d2bc0483665adbc1",
        2908,
    ),
    "97bf1614f3afe53a6e2edb4d8c7dff79e9661782": (
        "d9819f6f6ebf2998ac967178778cc1911757e455",
        "079ab813e3ed76902b007882611cf495eb2c63b8f75560a0666d956c350fdffe",
        "b6d98c6cd9d3e1502b3a80f3512a8437b46691d8d31aa26aab193d754e103720",
        "84f6620cb9a74a59f4b1b8a7433ed2603cfad2b2",
        2927,
    ),
    "0d00581251e642a5d6b56b2497a9adad93069e6b": (
        "fa7a7c09df6d93133324f762d756842de172433d",
        "079ab813e3ed76902b007882611cf495eb2c63b8f75560a0666d956c350fdffe",
        "40774107fdb566a626987dcded36dbd20f1a74c900d8d2801503209f3d7406c0",
        "c6a7bd138782fbeee320dbb4b7a03c2335124481",
        2930,
    ),
    "bd53914be6d2e4324cc867d52b3842b77cc6729a": (
        "ffddeb39a1b90d97736b79164aa620133eec3a1b",
        "8503f204a42e8a6e1af0efd29c081e4c07237771168ed4bf3d8b24eb1bfa16de",
        "3640bc607ee916684ae26cf7f28f069d1a7d5a2cab5509f01fae96d12ad8c434",
        "bf2829c638d683bbb52d574a2e395a96941de6c5",
        3079,
    ),
    "1514327669fc7be125523b991c847868c3a2a17e": (
        "0bbf15de82c467649cb88fb6378aaf74d829cfca",
        "8503f204a42e8a6e1af0efd29c081e4c07237771168ed4bf3d8b24eb1bfa16de",
        "277d812bea4b8b30f8922dca6b2ccaf57a04a2114ae5c886c428711abeb99b65",
        "61d9b055940f369334bf94950f3ce3265f4855d1",
        3098,
    ),
}
LIVE_ROUTES = (
    ("torch_cpu", 1),
    ("torch_cpu", 4),
    ("torch_cuda", 1),
    ("torch_cpu_native", 1),
    ("torch_cpu_native", 4),
    ("torch_cuda_native", 1),
)
WAVE_ROUTES = (("torch-cpu-batch", 1), ("torch-cpu-batch", 4), ("torch-cuda-batch", 1))
WAVE_FILES = (
    "pycbc/waveform/taylorf2_torch.py",
    "pycbc/waveform/waveform.py",
    "pycbc/waveform/torch_waveform_registry.py",
    "pycbc/waveform/taylorf2_triton.py",
)
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "PYCBC_NUM_THREADS",
)
METRICS = (
    "relative_l2",
    "max_pointwise_relative",
    "max_relative_amplitude",
    "max_wrapped_phase_radians",
)
NOTES = {
    "aggregation": (
        "Three separate worker processes per source, route, thread count and batch. "
        "Throughput is the median of their three warm throughput medians. Ranges "
        "are min–max across workers, not confidence intervals. The paired ratio "
        "is the median of candidate/baseline worker ratios matched by replicate; "
        "it need not equal the ratio of the two aggregate medians."
    ),
    "timing": (
        "Sequential workers pinned to CPU cores 8–11; before/after order reverses "
        "for replicate 2. Timing excludes profiling and dispatch instrumentation. "
        "Live workers use three warm iterations after one warmup; waveform workers "
        "use five calibrated timed blocks after two warmups. CUDA calls synchronize "
        "at timing boundaries. These three replicates characterize run variation, "
        "not independent scientific workloads."
    ),
    "live": (
        "Synthetic seeded templates and injected strain; N=131072, three strain "
        "blocks per iteration, complex64 inputs/outputs, SNR threshold 5.5. "
        "The public LiveBatchMatchedFilter.process_data API is timed with chi-square "
        "and sine-Gaussian vetoes disabled. Frame I/O, PSD estimation, bank loading, "
        "waveform generation and CLI startup are excluded. The raw unit templates/s "
        "counts one template filtered against one strain block. Native labels mean "
        "opt-in routing is enabled; admission can select a fallback for a given batch."
    ),
    "live_parity": (
        "Each baseline and candidate is checked against the corresponding standard "
        "CPU control: trigger counts and template IDs, end times within 1e-4 s, "
        "SNR and wrapped phase differences below 1e-3, relative sigma-squared and "
        "aggregate output-norm differences below 1e-3. These checks do not establish "
        "pointwise equivalence of every matched-filter output sample."
    ),
    "search_capacity": (
        "Real-time filtering capacity is derived from the measured template–block "
        "rate: multiply by 56 seconds of valid strain per block, then divide CPU "
        "routes by their configured 1- or 4-core budget. CUDA capacity uses one GPU "
        "and includes the timed host work. The CPU divisor is the configured thread "
        "count, not measured CPU utilization; affinity CPUs 8–11 are distinct "
        "physical cores. This is synthetic matched-filter component capacity with "
        "vetoes and I/O excluded, not complete production search capacity. Raw "
        "template–block rates and all unchanged ratios remain in JSON. The live "
        "harness's legacy throughput_wps_summary unit says waveforms/second, but "
        "its numerator is batch size times block count: template–block evaluations, "
        "not generated waveforms. That raw label is preserved; standalone waveform "
        "generation has a separate waveforms/second metric. The 56 s "
        "basis, pinned source trees, full file hashes and source excerpts are "
        "recorded in search_capacity_basis."
    ),
    "waveform": (
        "Standalone waveform generation, measured in waveforms/second; this is "
        "not search capacity. Complete public get_fd_waveform_batch TaylorF2 calls, both polarizations, "
        "complex128 and 4097 frequency bins (delta_f=0.25 Hz, 20–1024 Hz). Host "
        "parameter conversion is included; output stays on the requested device "
        "and host copies are excluded. Triton and autograd are disabled. Every row "
        "and both polarizations pass native-scalar pointwise relative error <=2e-10 "
        "and native-scalar versus LAL CPU relative L2 <1e-11, with finite values, "
        "exact zero support, frequency spacing and epoch checks. Direct batch/LAL "
        "errors are reported without an additional numerical tolerance."
    ),
}


class InvalidCampaign(ValueError):
    """Evidence is incomplete, inconsistent, or failed a required gate."""


def require(condition, message):
    if not condition:
        raise InvalidCampaign(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    def bad_constant(value):
        raise InvalidCampaign(f"nonfinite JSON constant in {path}: {value}")

    return json.loads(
        path.read_text(), object_pairs_hook=no_duplicates, parse_constant=bad_constant
    )


def number(value, label, positive=False):
    require(
        type(value) in (int, float) and math.isfinite(value), f"{label}: not finite"
    )
    require(value > 0 if positive else value >= 0, f"{label}: invalid sign")
    return value


def close(actual, expected, label):
    number(actual, label)
    require(
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15),
        f"{label}: {actual!r} != recomputed {expected!r}",
    )


def identity(source, label):
    for key in ("sha", "tree"):
        require(
            isinstance(source[key], str) and re.fullmatch(r"[0-9a-f]{40}", source[key]),
            f"{label}: invalid {key}",
        )
    require(
        isinstance(source["root"], str) and source["root"].startswith("/"),
        f"{label}: invalid source root",
    )
    return {key: source[key] for key in ("root", "sha", "tree")}


def label(route, batch, threads, replicate, source=None):
    result = f"{route}-b{batch}-t{threads}-r{replicate}"
    return result if source is None else f"{result}-{source}"


def expected_keys(routes, sources=SOURCES):
    return [
        (route, batch, threads, rep, source)
        for route, threads in routes
        for batch in BATCHES
        for rep in REPLICATES
        for source in sources
    ]


def exact_files(directory, names):
    require(directory.is_dir(), f"missing input directory: {directory}")
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    expected = {f"{name}.json" for name in names}
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    require(
        not missing and not extra,
        f"{directory.name}: expected {len(expected)} files; "
        f"missing {len(missing)} {missing[:4]}, unexpected {len(extra)} {extra[:4]}",
    )


def checked_live(record, key, source):
    route, batch, threads, rep, name = key
    tag = label(*key)
    require(record["comparison_label"] == tag, f"{tag}: comparison label")
    require(record["comparison_source"] == source, f"{tag}: source mismatch")
    require(record["source_root"] == source["root"], f"{tag}: import source root")
    for field, value in (
        ("route", route),
        ("batch", batch),
        ("threads", threads),
        ("comparison_replicate", rep),
        ("size", 131072),
        ("num_blocks", 3),
        ("seed", 7101),
        ("snr_threshold", 5.5),
    ):
        require(record[field] == value, f"{tag}: incorrect {field}")
    require(
        record["measurement"]["scope"] == "public_library_api"
        and record["measurement"]["call_surface"]
        == "LiveBatchMatchedFilter.process_data",
        f"{tag}: wrong measurement surface",
    )
    require(
        record["state"]
        == {
            "cold_iterations": 1,
            "warmup_iterations": 1,
            "measured_warm_iterations": 3,
            "blocks_per_iteration": 3,
        },
        f"{tag}: timing state",
    )
    require(
        record["dimensions"]
        == {
            "batch_templates": batch,
            "bank_templates": batch,
            "fft_samples": 131072,
            "frequency_bins": 65537,
            "blocks_per_iteration": 3,
            "waveforms_per_iteration": 3 * batch,
        },
        f"{tag}: dimensions",
    )
    require(
        record["dtypes"]
        == {
            "template": "complex64",
            "strain": "complex64",
            "psd": "float32",
            "output": "complex64",
        },
        f"{tag}: precision",
    )
    device = "cuda:0" if "cuda" in route else "cpu"
    require(record["device"] == device, f"{tag}: device")
    routing = record["routing"]
    require(
        routing["experimental"] is route.endswith("_native"), f"{tag}: routing mode"
    )
    flags = routing["feature_flags"]
    prefix = "PYCBC_TORCH_CPU_" if "cpu" in route else "PYCBC_TORCH_CUDA_"
    if route != "branch_standard":
        suffixes = (
            ("NATIVE_BATCH_CORRELATE", "FFTW_BATCH")
            if "cpu" in route
            else ("NATIVE_BATCH_CORRELATE", "NATIVE_BATCH_PEAK")
        )
        for suffix in suffixes:
            require(
                flags[prefix + suffix]["enabled"] is route.endswith("_native"),
                f"{tag}: {suffix} selection",
            )
    command = record["command"]
    require(command[:3] == ["taskset", "-c", "8-11"], f"{tag}: CPU affinity command")
    require(
        command[4:]
        == [
            source["root"] + "/tools/bench_production_live_batch.py",
            "child",
            "--route",
            route,
            "--source-root",
            source["root"],
            "--batch",
            str(batch),
            "--size",
            "131072",
            "--num-blocks",
            "3",
            "--threads",
            str(threads),
            "--samples",
            "3",
            "--warmups",
            "1",
            "--snr-threshold",
            "5.5",
            "--cuda-device",
            "0",
            "--seed",
            "7101",
            "--call-surface",
            "public",
        ],
        f"{tag}: unexpected worker command",
    )
    summary = record["throughput_wps_summary"]
    samples = summary["samples"]
    require(
        summary["count"] == len(samples) == 3 and summary["unit"] == "waveforms/second",
        f"{tag}: warm throughput samples",
    )
    latencies = record["warm_iteration_latencies_ms"]
    require(len(latencies) == 3, f"{tag}: iteration count")
    for sample, latency in zip(samples, latencies, strict=True):
        number(latency, tag, positive=True)
        close(sample, 3 * batch * 1000 / latency, f"{tag}: throughput denominator")
    close(summary["median"], statistics.median(samples), f"{tag}: worker median")
    number(summary["median"], tag, positive=True)
    require(len(record["block_triggers"]) == 3, f"{tag}: output block count")
    number(record["output_l2"], f"{tag}: output norm")
    return summary["median"]


def checked_live_parity(document, before, after, control, route, batch):
    tag = f"{route}, batch {batch}"
    require(
        set(document) == {f"batch_{batch}", "all_passed_globally"}
        and document["all_passed_globally"] is True,
        f"{tag}: live global parity",
    )
    cell = document[f"batch_{batch}"]
    require(
        cell["batch"] == batch and cell["all_passed"] is True, f"{tag}: batch parity"
    )
    require(
        cell["routes_evaluated"] == ["branch_standard", route, route + "_fixed"],
        f"{tag}: parity route coverage",
    )
    expected = {f"{name}_vs_branch_standard" for name in (route, route + "_fixed")}
    require(set(cell["comparisons"]) == expected, f"{tag}: comparison coverage")
    for name, record in ((route, before), (route + "_fixed", after)):
        comparison = cell["comparisons"][f"{name}_vs_branch_standard"]
        for flag in (
            "passed",
            "finite_values",
            "block_count_match",
            "trigger_count_match",
            "template_id_match",
            "end_time_match",
            "sequence_length_match",
            "veto_count_match",
        ):
            require(comparison[flag] is True, f"{tag}: failed {flag}")
        maxima = {
            "max_snr_diff": 0.0,
            "max_phase_diff": 0.0,
            "max_sigmasq_relative_diff": 0.0,
        }
        for actual, reference in zip(
            record["block_triggers"], control["block_triggers"], strict=True
        ):
            count = reference["num_triggers"]
            require(
                type(count) is int and count >= 0 and actual["num_triggers"] == count,
                f"{tag}: trigger count",
            )
            require(
                actual["template_ids"] == reference["template_ids"],
                f"{tag}: template IDs",
            )
            require(
                actual["veto_count"] is reference["veto_count"] is None,
                f"{tag}: public veto count",
            )
            for field in (
                "template_ids",
                "end_times",
                "snrs",
                "coa_phases",
                "sigmasqs",
            ):
                av, rv = actual[field], reference[field]
                require(len(av) == len(rv) == count, f"{tag}: trigger field length")
                require(
                    all(type(x) in (int, float) and math.isfinite(x) for x in av + rv),
                    f"{tag}: nonfinite trigger field",
                )
                for a, r in zip(av, rv, strict=True):
                    if field == "end_times":
                        require(abs(a - r) <= 1e-4, f"{tag}: end time parity")
                    elif field == "snrs":
                        maxima["max_snr_diff"] = max(maxima["max_snr_diff"], abs(a - r))
                    elif field == "coa_phases":
                        delta = abs((r - a + math.pi) % (2 * math.pi) - math.pi)
                        maxima["max_phase_diff"] = max(maxima["max_phase_diff"], delta)
                    elif field == "sigmasqs":
                        delta = abs(a - r) / max(abs(r), 1e-12)
                        maxima["max_sigmasq_relative_diff"] = max(
                            maxima["max_sigmasq_relative_diff"], delta
                        )
        maxima["relative_output_l2_diff"] = abs(
            record["output_l2"] - control["output_l2"]
        ) / max(control["output_l2"], 1e-12)
        for metric, value in maxima.items():
            require(value < 1e-3, f"{tag}: recomputed {metric} failed")
            close(comparison[metric], value, f"{tag}: stored {metric}")


def checked_wave(record, key, source):
    route, batch, threads, rep, _ = key
    tag = label(*key)
    require(
        record["status"] == "ok" and record["schema"] == 1, f"{tag}: waveform status"
    )
    for field, value in (
        ("route", route),
        ("batch", batch),
        ("threads", threads),
        ("replicate", rep),
        ("precision", "double"),
    ):
        require(record[field] == value, f"{tag}: incorrect {field}")
    require(
        identity(record["source"], tag) == identity(source, tag),
        f"{tag}: source identity",
    )
    src = record["source"]
    require(
        src["clean"] is True and src["status_porcelain"] == src["status_after"] == "",
        f"{tag}: dirty waveform source",
    )
    hashes = src["waveform_source_sha256"]
    require(
        set(hashes) == set(WAVE_FILES)
        and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes.values()),
        f"{tag}: waveform source file hashes",
    )
    device = "cuda:0" if "cuda" in route else "cpu"
    dispatch = record["dispatch"]
    expected = {"native_scalar": 0, "native_batch": 1, "lal_fd": 0}
    require(
        dispatch["passed"] is True
        and dispatch["requested"] == route
        and dispatch["actual"] == f"native Torch {device} batch"
        and dispatch["probe_counts"] == dispatch["expected_counts"] == expected
        and dispatch["triton_requested"] is False
        and dispatch["triton_actual"] is False,
        f"{tag}: waveform dispatch",
    )
    require(
        record["workload"]
        == {
            "approximant": "TaylorF2",
            "nominal_bins": 4097,
            "input_policy": "Python host scalars/lists; conversion included in calls",
            "output_policy": "device output retained; host copy excluded from timing",
            "autograd": False,
        },
        f"{tag}: waveform workload",
    )
    require(len(record["parameters"]) == batch, f"{tag}: parameter rows")
    for i, row in enumerate(record["parameters"]):
        require(
            row
            == dict(
                mass1=1.4 + 0.05 * (i % 8),
                mass2=1.3 - 0.02 * (i % 8),
                spin1z=0.02,
                spin2z=-0.01,
                distance=100.0,
                inclination=0.4,
                coa_phase=0.2,
                delta_f=0.25,
                f_lower=20.0,
                f_final=1024.0,
                f_ref=30.0,
            ),
            f"{tag}: parameter grid",
        )
    output = record["output"]
    require(
        output["devices"] == [device, device]
        and output["dtypes"] == ["complex128"] * 2
        and output["shapes"] == [[batch, 4097]] * 2
        and output["metadata"] == [{"delta_f": 0.25, "epoch": -4.0}] * batch,
        f"{tag}: output shape/device/precision/metadata",
    )
    runtime, environment = record["runtime"], record["environment"]
    require(
        runtime["torch_threads"] == threads and runtime["torch_interop_threads"] == 1,
        f"{tag}: Torch thread count",
    )
    require(
        all(environment[k] == str(threads) for k in THREAD_VARS)
        and environment["PYCBC_TAYLORF2_NATIVE"] == "1"
        and environment["PYCBC_TAYLORF2_TRITON"] == "0",
        f"{tag}: waveform environment",
    )
    for name, gate, tolerance in (
        ("parity_batch_vs_native_scalar", "pointwise_relative", 2e-10),
        ("parity_scalar_vs_cpu_reference", "relative_l2", 1e-11),
        ("parity_direct_vs_cpu_reference", None, None),
    ):
        parity = record[name]
        require(
            parity["passed"] is True
            and parity["metadata_equal"] is True
            and parity["gate"] == gate
            and parity["tolerance"] == tolerance,
            f"{tag}: {name} gate",
        )
        metrics = parity["metrics"]
        require(
            len(metrics) == 2 * batch
            and {(m["polarization"], m["row"]) for m in metrics}
            == {(pol, i) for pol in (0, 1) for i in range(batch)},
            f"{tag}: full parity coverage",
        )
        for metric in metrics:
            require(
                all(
                    metric[flag] is True
                    for flag in ("passed", "finite", "exact_zero_support")
                ),
                f"{tag}: failed waveform row",
            )
            for field in METRICS:
                number(metric[field], f"{tag}: {field}")
            if gate == "relative_l2":
                require(
                    metric["relative_l2"] < tolerance, f"{tag}: scalar/LAL tolerance"
                )
            elif gate == "pointwise_relative":
                require(
                    metric["max_pointwise_relative"] <= tolerance,
                    f"{tag}: batch/scalar tolerance",
                )
    timing = record["timing"]
    samples, elapsed, inner = (
        timing[k]
        for k in (
            "sample_seconds_per_call",
            "sample_total_seconds",
            "inner_calls_per_sample",
        )
    )
    require(
        len(samples) == len(elapsed) == 5
        and type(inner) is int
        and 1 <= inner <= 64
        and timing["warmup_calls"] == 2,
        f"{tag}: waveform timing samples",
    )
    for sample, total in zip(samples, elapsed, strict=True):
        number(sample, tag, positive=True)
        close(sample, total / inner, f"{tag}: waveform inner denominator")
    median = statistics.median(samples)
    close(timing["median_seconds_per_call"], median, f"{tag}: waveform median")
    close(
        timing["median_seconds_per_waveform"],
        median / batch,
        f"{tag}: waveform latency",
    )
    close(timing["waveforms_per_second"], batch / median, f"{tag}: waveform throughput")
    return timing["waveforms_per_second"]


def summary(values):
    require(len(values) == 3, "aggregate must have exactly three workers")
    return dict(
        median=statistics.median(values),
        minimum=min(values),
        maximum=max(values),
        worker_values=values,
    )


def capacity_basis(sources):
    """Pin the duration derivation to inspected source trees and run snapshots."""
    receipts = {}
    harness_path = "tools/bench_production_live_batch.py"
    filter_path = "pycbc/filter/matchedfilter.py"
    for name, source in sources.items():
        require(source["sha"] in CAPACITY_SOURCES, f"{name}: unknown duration basis")
        tree, harness_hash, filter_hash, filter_blob, line = CAPACITY_SOURCES[
            source["sha"]
        ]
        require(source["tree"] == tree, f"{name}: duration source tree differs")
        checked_snapshot = False
        for path, expected in (
            (harness_path, harness_hash),
            (filter_path, filter_hash),
        ):
            if "tracked_sha256" in source:
                require(
                    source["tracked_sha256"][path] == expected,
                    f"{name}: duration source hash differs: {path}",
                )
                checked_snapshot = True
            if "tracked_files" in source:
                require(
                    source["tracked_files"][path]["sha256"] == expected,
                    f"{name}: duration source hash differs: {path}",
                )
                checked_snapshot = True
        receipts[name] = {
            "commit": source["sha"],
            "tree": tree,
            "run_snapshot_file_hashes_checked": checked_snapshot,
            "harness": {
                "path": harness_path,
                "sha256": harness_hash,
                **CAPACITY_HARNESSES[harness_hash],
                "sample_rate_source": "sample_rate = 2048.0",
                "blocksize_source": "blocksize = 56.0",
                "public_data_source": [
                    "blocksize=blocksize,",
                    "sample_rate=sample_rate,",
                    "start_time=1000000000.0 + b * blocksize,",
                ],
            },
            "valid_interval": {
                "path": filter_path,
                "sha256": filter_hash,
                "git_blob": filter_blob,
                "start_line": line,
                "source": [
                    "valid_end = int(psize - self.data.trim_padding)",
                    "valid_start = int(",
                    "    valid_end - self.data.blocksize * self.data.sample_rate",
                    ")",
                ],
            },
        }
    return {
        "schema": 1,
        "valid_seconds_per_block": BLOCK_SECONDS,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "valid_samples_per_block": int(BLOCK_SECONDS * SAMPLE_RATE_HZ),
        "cpu_formula": "raw_templates_per_second * 56 / configured_threads",
        "cuda_formula": "raw_templates_per_second * 56 / 1 GPU",
        "units": dict(CAPACITY_UNITS),
        "cpu_divisor": "configured core budget (thread count), not measured utilization",
        "cpu_affinity": "8-11; four distinct physical cores",
        "verification": "File bytes, Git blobs and excerpts were independently inspected "
        "from the pinned frozen source trees when preparing this reporter. Each report "
        "checks worker source commit/tree identities; available full run snapshots also "
        "check both file hashes. The supplied Git source bundles permit source audit.",
        "scope": NOTES["search_capacity"],
        "sources": receipts,
    }


def search_capacity(route, threads, statistics_by_state):
    """Add derived capacity without replacing raw rates or recomputing ratios."""
    device = "cuda" if "cuda" in route else "cpu"
    require(threads in (1, 4), "unsupported configured CPU core budget")
    resource_count = 1 if device == "cuda" else threads
    factor = BLOCK_SECONDS / resource_count
    scaled = {
        state: {
            "median": value["median"] * factor,
            "minimum": value["minimum"] * factor,
            "maximum": value["maximum"] * factor,
            "worker_values": [worker * factor for worker in value["worker_values"]],
        }
        for state, value in statistics_by_state.items()
    }
    return {
        "unit": CAPACITY_UNITS[device],
        "valid_seconds_per_block": BLOCK_SECONDS,
        "resource_count": resource_count,
        "resource": "GPU" if device == "cuda" else "configured CPU core",
        "scale_from_raw_rate": factor,
        **scaled,
    }


def aggregate(values, routes):
    rows = []
    for route, threads in routes:
        for batch in BATCHES:
            before, after = (
                [values[(route, batch, threads, r, name)] for r in REPLICATES]
                for name in SOURCES
            )
            ratios = [a / b for a, b in zip(after, before, strict=True)]
            rows.append(
                dict(
                    route=route,
                    threads=threads,
                    batch=batch,
                    before=summary(before),
                    after=summary(after),
                    paired_ratio=summary(ratios),
                    ratio_of_medians=statistics.median(after)
                    / statistics.median(before),
                )
            )
    return rows


def consistent(records, fields, description):
    for field in fields:
        values = {json.dumps(record[field], sort_keys=True) for record in records}
        require(len(values) == 1, f"inconsistent {description}: {field}")


def validate(input_dir, status_path):
    status = load(status_path)
    require(
        status["state"] == "complete" and "error" not in status,
        f"campaign is not complete: {status.get('state')}",
    )
    require(
        datetime.fromisoformat(status["finished_utc"])
        >= datetime.fromisoformat(status["started_utc"]),
        "campaign timestamps are reversed",
    )
    sources = status["sources"]
    require(set(sources) == set(SOURCES), "missing or unexpected source identity")
    for name, source in sources.items():
        identity(source, name)
        require(source["status"] == "", f"{name}: dirty campaign source")
    require(
        sources["baseline"]["sha"] != sources["candidate"]["sha"],
        "identical source revisions",
    )
    live_keys = expected_keys(LIVE_ROUTES) + expected_keys(
        (("branch_standard", 1), ("branch_standard", 4)), ("baseline",)
    )
    wave_keys = expected_keys(WAVE_ROUTES)
    parity_keys = [
        (r, b, t, p) for r, t in LIVE_ROUTES for b in BATCHES for p in REPLICATES
    ]
    names = {
        "live": [label(*k) for k in live_keys],
        "waveform": [label(*k) for k in wave_keys],
        "parity": [label(*k) for k in parity_keys],
    }
    completed = status["completed"]
    require(
        len(completed) == len(set(completed)) == 360
        and set(completed) == set(names["live"] + names["waveform"]),
        "completed worker ledger mismatch",
    )
    manifest = {str(status_path): digest(status_path)}
    for kind, expected in names.items():
        exact_files(input_dir / kind, expected)
        for stem in expected:
            path = input_dir / kind / f"{stem}.json"
            manifest[str(path)] = digest(path)
    live_records, live_values, wave_records, wave_values = {}, {}, {}, {}
    for key in live_keys:
        record = load(input_dir / "live" / f"{label(*key)}.json")
        live_values[key] = checked_live(record, key, sources[key[-1]])
        live_records[key] = record
    for route, batch, threads, rep in parity_keys:
        key = (route, batch, threads, rep)
        doc = load(input_dir / "parity" / f"{label(*key)}.json")
        checked_live_parity(
            doc,
            live_records[(*key, "baseline")],
            live_records[(*key, "candidate")],
            live_records[("branch_standard", batch, threads, rep, "baseline")],
            route,
            batch,
        )
    for key in wave_keys:
        record = load(input_dir / "waveform" / f"{label(*key)}.json")
        wave_values[key] = checked_wave(record, key, sources[key[-1]])
        wave_records[key] = record
    all_live, all_wave = list(live_records.values()), list(wave_records.values())
    consistent(
        all_live, ("python", "numpy_version", "measurement", "dtypes"), "live workers"
    )
    consistent(
        [r for r in all_live if r["route"] != "branch_standard"],
        ("torch_version",),
        "live Torch runtime",
    )
    consistent(
        [r for r in all_live if "cuda" in r["route"]],
        ("cuda_device_name",),
        "live CUDA device",
    )
    require(
        len({r["command"][3] for r in all_live}) == 1,
        "inconsistent live Python executable",
    )
    consistent(all_wave, ("harness_sha256", "workload"), "waveform worker")
    consistent(
        [r["runtime"] for r in all_wave],
        (
            "hostname",
            "platform",
            "machine",
            "python",
            "python_executable",
            "numpy",
            "torch",
            "lalsuite",
            "scipy",
            "triton",
            "torch_cuda_build",
        ),
        "waveform runtime",
    )
    consistent(
        [r["runtime"] for r in all_wave if "cuda" in r["route"]],
        ("cuda_device",),
        "waveform CUDA device",
    )
    for name in SOURCES:
        consistent(
            [r["source"] for k, r in wave_records.items() if k[-1] == name],
            ("waveform_source_sha256",),
            f"{name} waveform source",
        )
    for route, threads in LIVE_ROUTES:
        consistent(
            [r for k, r in live_records.items() if k[0] == route and k[2] == threads],
            ("routing",),
            f"{route} configuration",
        )
    for batch in BATCHES:
        consistent(
            [r for k, r in live_records.items() if k[1] == batch],
            ("injection_metadata",),
            f"batch {batch} injections",
        )
    runtime = all_wave[0]["runtime"]
    require(
        all_live[0]["python"] == runtime["python"]
        and all_live[0]["numpy_version"] == runtime["numpy"]
        and all_live[0]["command"][3] == runtime["python_executable"],
        "live/waveform runtime mismatch",
    )
    require(
        all(
            r["torch_version"] == runtime["torch"]
            for r in all_live
            if r["route"] != "branch_standard"
        ),
        "live/waveform Torch version mismatch",
    )
    cuda = next(r["runtime"]["cuda_device"] for r in all_wave if "cuda" in r["route"])
    require(
        all(
            r["cuda_device_name"] == cuda["name"]
            for r in all_live
            if "cuda" in r["route"]
        ),
        "live/waveform CUDA device mismatch",
    )
    worker_path = Path(__file__).with_name("waveform-worker.py")
    require(
        all_wave[0]["harness_sha256"] == digest(worker_path),
        "local waveform worker differs from measured worker",
    )
    manifest[str(worker_path)] = digest(worker_path)
    orchestration = Path(__file__).with_name("run-comparison.py")
    manifest[str(orchestration)] = digest(orchestration)
    controls = [
        dict(
            route="branch_standard",
            threads=t,
            batch=b,
            before=summary(
                [
                    live_values[("branch_standard", b, t, p, "baseline")]
                    for p in REPLICATES
                ]
            ),
        )
        for t in (1, 4)
        for b in BATCHES
    ]
    live_rows = aggregate(live_values, LIVE_ROUTES)
    for row in live_rows:
        row["search_capacity"] = search_capacity(
            row["route"],
            row["threads"],
            {state: row[state] for state in ("before", "after")},
        )
    for row in controls:
        row["search_capacity"] = search_capacity(
            row["route"], row["threads"], {"before": row["before"]}
        )
    return {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_started_utc": status["started_utc"],
        "campaign_finished_utc": status["finished_utc"],
        "sources": sources,
        "batches": list(BATCHES),
        "replicates": list(REPLICATES),
        "counts": {
            "live_workers": len(live_keys),
            "standard_controls": 36,
            "waveform_workers": len(wave_keys),
            "live_parity_records": len(parity_keys),
        },
        "validation": {
            "complete": True,
            "all_live_parity_passed": True,
            "all_waveform_dispatch_and_parity_passed": True,
        },
        "runtime": {
            k: runtime[k]
            for k in ("hostname", "platform", "python", "numpy", "torch", "lalsuite")
        },
        "cuda_device": cuda,
        "notes": NOTES,
        "live_unit": "templates/second",
        "waveform_unit": "waveforms/second",
        "search_capacity_basis": capacity_basis(sources),
        "search_capacity_units": dict(CAPACITY_UNITS),
        "live": live_rows,
        "waveform": aggregate(wave_values, WAVE_ROUTES),
        "standard_controls": controls,
        "waveform_source_sha256": {
            name: next(
                r["source"]["waveform_source_sha256"]
                for k, r in wave_records.items()
                if k[-1] == name
            )
            for name in SOURCES
        },
    }, manifest


def title(route, threads):
    device = "CUDA" if "cuda" in route else "CPU"
    native = " · native enabled" if route.endswith("_native") else ""
    return (
        f"Torch {device} · {threads} {'thread' if threads == 1 else 'threads'}{native}"
    )


def figures(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullLocator

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#899098",
            "axes.linewidth": 0.7,
            "text.color": "#20262D",
            "axes.labelcolor": "#20262D",
            "xtick.color": "#46515C",
            "ytick.color": "#46515C",
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    styles = {
        "before": dict(
            color="#747C85", marker="o", markerfacecolor="white", linestyle="--"
        ),
        "after": dict(
            color="#2166AC", marker="o", markerfacecolor="#2166AC", linestyle="-"
        ),
    }
    paths = []
    for kind, routes, shape, size, heading, subtitle, unit in (
        (
            "live",
            LIVE_ROUTES,
            (2, 3),
            (15, 8.5),
            "Live filtering capacity at real time",
            "Synthetic matched-filter component · 56 s valid strain/block · Vetoes and I/O excluded",
            "Templates / second",
        ),
        (
            "waveform",
            WAVE_ROUTES,
            (1, 3),
            (15, 5.3),
            "TaylorF2 batch waveform throughput",
            "Complete public API calls · 4,097 complex128 bins · 3 paired worker replicates",
            "Waveforms / second",
        ),
    ):
        fig, axes = plt.subplots(*shape, figsize=size, sharex=True, squeeze=False)
        fig.subplots_adjust(
            left=0.072,
            right=0.985,
            top=0.77 if kind == "waveform" else 0.84,
            bottom=0.24 if kind == "waveform" else 0.16,
            hspace=0.48,
            wspace=0.29,
        )
        fig.text(0.072, 0.964, heading, fontsize=18, weight="bold", va="top")
        fig.text(0.072, 0.913, subtitle, fontsize=11, va="top", color="#46515C")
        groups = {
            (route, threads): [
                r["search_capacity"] if kind == "live" else r
                for r in report[kind]
                if r["route"] == route and r["threads"] == threads
            ]
            for route, threads in routes
        }
        limits = {}
        for i, key in enumerate(routes):
            column = i % 3
            high = max(r[s]["maximum"] for r in groups[key] for s in styles)
            limits[column] = max(limits.get(column, 0), high)
        for i, (ax, key) in enumerate(zip(axes.flat, routes, strict=True)):
            rows = groups[key]
            for state, style in styles.items():
                ax.plot(
                    BATCHES,
                    [r[state]["median"] for r in rows],
                    label=state.title(),
                    linewidth=1.8,
                    markersize=5,
                    markeredgewidth=1.2,
                    **style,
                )
                ax.fill_between(
                    BATCHES,
                    [r[state]["minimum"] for r in rows],
                    [r[state]["maximum"] for r in rows],
                    color=style["color"],
                    alpha=0.12,
                    linewidth=0,
                )
                ax.errorbar(
                    BATCHES,
                    [r[state]["median"] for r in rows],
                    yerr=[
                        [r[state]["median"] - r[state]["minimum"] for r in rows],
                        [r[state]["maximum"] - r[state]["median"] for r in rows],
                    ],
                    fmt="none",
                    ecolor=style["color"],
                    elinewidth=0.8,
                    capsize=2,
                    alpha=0.7,
                )
            ax.set_title(title(*key), loc="left", pad=11)
            ax.set_xscale("log", base=2)
            ax.set_xticks(
                BATCHES,
                [str(b) for b in BATCHES],
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
            ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis="x", labelbottom=True)
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:,.0f}")
            )
            ax.set_ylim(0, limits[i % 3] * 1.12)
            ax.set_xlim(0.75, 1365)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E5E8EB", linewidth=0.7)
            ax.set_xlabel("Batch size (log scale)", labelpad=8)
            axis_unit = unit
            if kind == "live":
                resource = "GPU" if "cuda" in key[0] else "core"
                axis_unit = f"Templates / {resource} at real time"
            ax.set_ylabel(axis_unit, labelpad=7)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.984, 0.977),
            ncol=2,
            frameon=False,
            handlelength=2.8,
        )
        source = f"Before {report['sources']['baseline']['sha'][:12]}  ·  After {report['sources']['candidate']['sha'][:12]}"
        foot = (
            "Points: median of 3 worker medians. Bands/whiskers: full worker range. "
            + source
        )
        fig.text(
            0.072,
            0.043 if kind == "waveform" else 0.032,
            foot,
            fontsize=9,
            color="#46515C",
        )
        for extension in ("png", "svg"):
            path = output / f"{kind}-before-after.{extension}"
            fig.savefig(
                path, dpi=220, metadata={"Creator": "PyCBC paired campaign report"}
            )
            paths.append(path.name)
        plt.close(fig)
    return paths


def format_stat(value, ratio=False):
    digits = 3 if ratio else 1
    return f"{value['median']:,.{digits}f} ({value['minimum']:,.{digits}f}–{value['maximum']:,.{digits}f})"


def markdown(report):
    parts = [
        "# Paired Torch performance comparison",
        "",
        f"Before: `{report['sources']['baseline']['sha']}`. After: `{report['sources']['candidate']['sha']}`.",
        "",
        f"Campaign completed {report['campaign_finished_utc']} on {report['runtime']['hostname']}; "
        f"CUDA device: {report['cuda_device']['name']}. All 252 live workers (including 36 standard CPU "
        "controls), 108 waveform workers and 108 live parity records are present and validated.",
        "",
        NOTES["aggregation"],
        "",
        NOTES["timing"],
        "",
    ]
    for kind, heading, unit in (
        ("live", "Live filtering capacity at real time", None),
        ("waveform", "Standalone TaylorF2 waveform generation", "waveforms/s"),
    ):
        parts.extend(
            [
                f"## {heading}",
                "",
                f"![{heading} before and after]({kind}-before-after.png)",
                "",
                "Parentheses show the range across three workers. "
                "Paired after/before values above 1 indicate higher capacity or generation rate.",
                "",
                "| Route | Batch | Unit | Before | After | Paired after/before |",
                "|:--|--:|:--|--:|--:|--:|",
            ]
        )
        for row in report[kind]:
            displayed = row["search_capacity"] if kind == "live" else row
            row_unit = displayed["unit"] if kind == "live" else unit
            parts.append(
                f"| {title(row['route'], row['threads'])} | {row['batch']} | {row_unit} | "
                f"{format_stat(displayed['before'])} | {format_stat(displayed['after'])} | "
                f"{format_stat(row['paired_ratio'], ratio=True)} |"
            )
        parts.extend(["", NOTES[kind], ""])
        if kind == "live":
            parts.extend([NOTES["search_capacity"], "", NOTES["live_parity"], ""])
    parts.extend(
        [
            "## Standard CPU controls",
            "",
            "These baseline controls anchor live parity; "
            "the campaign did not time candidate standard CPU controls.",
            "",
            "| Configured cores | Batch | Templates/core at real time, median (range) |",
            "|--:|--:|--:|",
        ]
    )
    for row in report["standard_controls"]:
        parts.append(
            f"| {row['threads']} | {row['batch']} | {format_stat(row['search_capacity']['before'])} |"
        )
    parts.extend(
        [
            "",
            "All source identities, worker aggregates and paired ratios are in "
            "[report.json](report.json). [input-manifest.json](input-manifest.json) records "
            "SHA-256 hashes of every input. The waveform worker hash is checked against "
            "the local worker script; the orchestration script hash is a local provenance "
            "record. Live parity was also recomputed from the recorded triggers and norms.",
            "",
        ]
    )
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=base / "comparison")
    parser.add_argument(
        "--status", type=Path, help="Defaults to INPUT/../comparison-status.json"
    )
    parser.add_argument("--out", type=Path, default=base / "report")
    args = parser.parse_args()
    input_dir, output = args.input.resolve(), args.out.resolve()
    status_path = (args.status or input_dir.parent / "comparison-status.json").resolve()
    temporary = None
    try:
        require(
            not output.exists(), f"output already exists; choose a new --out: {output}"
        )
        report, manifest = validate(input_dir, status_path)
        require(
            output.parent.is_dir(), f"output parent does not exist: {output.parent}"
        )
        temporary = Path(tempfile.mkdtemp(prefix=".paired-report-", dir=output.parent))
        report["figures"] = figures(report, temporary)
        manifest[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
        manifest_path = temporary / "input-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        report["input_manifest"] = {
            "path": "input-manifest.json",
            "sha256": digest(manifest_path),
            "files": len(manifest),
        }
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        (temporary / "report.md").write_text(markdown(report))
        for path, expected in manifest.items():
            require(
                digest(Path(path)) == expected,
                f"input changed during report generation: {path}",
            )
        temporary.rename(output)
        temporary = None
        print(
            json.dumps({"status": "ok", "out": str(output), "counts": report["counts"]})
        )
    except (InvalidCampaign, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Report refused: {exc}", file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
