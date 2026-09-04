"""Pure-Python checks for Torch performance evidence tooling."""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from tools import bench_production_live_batch as live_batch
from tools.benchmark_artifact import (
    SCHEMA_VERSION,
    artifact_content_sha256,
    load_mergeable_artifact,
    sample_summary,
    seal_artifact,
    summary_median_ci95,
    summary_statistic,
)
from tools.generate_torch_performance_plots import (
    _parity_cells,
    parse_live_batch_artifact,
)


@pytest.mark.parametrize("name", (
    "live-batch-cpu-cuda-t1.json", "live-batch-cpu-t4.json",
))
def test_published_historical_samples_keep_original_hash_and_seal(name):
    directory = (Path(__file__).resolve().parents[1] / "docs" / "_static"
                 / "torch-evidence" / "2026-09-04")
    hashes = dict(line.split()[::-1] for line in
                  (directory / "SHA256SUMS").read_text().splitlines())
    raw = (directory / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == hashes[name]
    payload = json.loads(raw)
    assert payload["content_sha256"] == artifact_content_sha256(payload)


def _legacy_summary(samples, *, unit="seconds"):
    ordered = sorted(samples)
    median = ordered[len(ordered) // 2]
    return {
        "unit": unit,
        "count": len(samples),
        "samples": list(samples),
        "median": median,
        "median_ci95": {"low": min(samples), "high": max(samples)},
    }


def test_sample_summary_preserves_samples_tails_and_ci():
    samples = [4.0, 1.0, 100.0, 3.0, 2.0]

    summary = sample_summary(
        samples,
        unit="ms/block",
        bootstrap_seed=9,
        bootstrap_resamples=100,
    )

    assert summary["samples"] == samples
    assert summary["median"] == summary["p50"] == 3.0
    assert summary["percentiles"] == {
        "p50": 3.0,
        "p95": pytest.approx(80.8),
        "p99": pytest.approx(96.16),
    }
    assert summary["median_ci95"]["low"] <= summary["p50"]
    assert summary["median_ci95"]["high"] >= summary["p50"]
    assert summary["bootstrap"] == {
        "method": "percentile",
        "confidence": 0.95,
        "resamples": 100,
        "seed": 9,
    }


def test_legacy_summary_accessors_recover_tail_and_existing_interval():
    legacy = _legacy_summary([1.0, 2.0, 3.0])

    assert summary_statistic(legacy, "p50") == 2.0
    assert summary_statistic(legacy, "p95") == pytest.approx(2.9)
    assert summary_statistic(legacy, "p99") == pytest.approx(2.98)
    assert summary_median_ci95(legacy) == (1.0, 3.0)


def test_schema_v2_artifact_remains_mergeable(tmp_path):
    path = tmp_path / "legacy.json"
    payload = seal_artifact(
        {
            "schema_version": 2,
            "schema": 2,
            "artifact_type": "component_microbenchmarks",
            "compatibility": {"sha256": "compatible"},
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_mergeable_artifact(
        path,
        artifact_type="component_microbenchmarks",
        compatibility_sha256="compatible",
    )

    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["schema"] == SCHEMA_VERSION


def test_default_and_native_routes_have_isolated_feature_flags():
    contaminated = {
        name: "1" for name in live_batch.FEATURE_FLAG_DEFAULTS
    }
    contaminated["UNRELATED_SETTING"] = "preserved"

    production_env = live_batch.route_environment("torch_cpu", contaminated)
    native_env = live_batch.route_environment("torch_cpu_native", contaminated)

    assert "torch_cpu_native" not in live_batch.DEFAULT_ROUTES
    assert "torch_cuda_native" not in live_batch.DEFAULT_ROUTES
    assert production_env["UNRELATED_SETTING"] == "preserved"
    assert not (
        live_batch.FEATURE_FLAG_DEFAULTS.keys() & production_env.keys()
    )
    assert native_env["PYCBC_TORCH_CPU_NATIVE_BATCH_CORRELATE"] == "1"
    assert native_env["PYCBC_TORCH_CPU_FFTW_BATCH"] == "1"
    assert native_env["PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK"] == "1"
    assert "PYCBC_TORCH_CUDA_NATIVE_BATCH_CORRELATE" not in native_env

    production = live_batch.route_configuration("torch_cpu")
    native = live_batch.route_configuration("torch_cpu_native")
    cuda = live_batch.route_configuration("torch_cuda")
    assert production["routing_mode"] == "production_default"
    assert native["routing_mode"] == "experimental_native"
    assert production["feature_flags"][
        "PYCBC_TORCH_CPU_NATIVE_BATCH_CORRELATE"
    ]["enabled"] is False
    assert native["feature_flags"][
        "PYCBC_TORCH_CPU_NATIVE_BATCH_CORRELATE"
    ]["enabled"] is True
    assert cuda["feature_flags"]["PYCBC_TORCH_DIRECT_BATCH_IFFT"][
        "enabled"
    ] is True
    assert production["feature_flags"]["PYCBC_BATCH_MAXELEMENTS"][
        "selection"
    ] == "constructor_argument"


def test_orchestrator_defaults_to_public_production_routes():
    parsed = live_batch._parser().parse_args(
        [
            "orchestrate",
            "--root",
            "/benchmark",
            "--python",
            "/python",
            "--output",
            "/artifact.json",
        ]
    )

    assert parsed.routes == list(live_batch.DEFAULT_ROUTES)
    assert parsed.include_experimental_routes is False
    assert parsed.call_surface == "public"


def _trigger_parity_record():
    return {
        "output_l2": 10.0,
        "block_triggers": [{
            "num_triggers": 1,
            "template_ids": [1000],
            "veto_count": None,
            "end_times": [1000000001.0],
            "snrs": [8.0],
            "coa_phases": [0.2],
            "sigmasqs": [1.0],
        }],
    }


@pytest.mark.parametrize(
    "field",
    ("end_times", "snrs", "coa_phases", "sigmasqs", "output_l2",
     "num_triggers", "template_ids", "veto_count"),
)
@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
@pytest.mark.parametrize("side", ("reference", "candidate", "both"))
def test_live_parity_rejects_nonfinite_values(field, value, side):
    reference = _trigger_parity_record()
    candidate = deepcopy(reference)
    selected = {
        "reference": [reference], "candidate": [candidate],
        "both": [reference, candidate],
    }[side]
    for record in selected:
        if field == "output_l2":
            record[field] = value
        elif field in ("num_triggers", "veto_count"):
            record["block_triggers"][0][field] = value
        else:
            record["block_triggers"][0][field][0] = value

    result = live_batch._verify_parity({
        1: {"branch_standard": reference, "torch_cpu": candidate},
    })

    comparison = result["batch_1"]["comparisons"][
        "torch_cpu_vs_branch_standard"
    ]
    assert comparison["finite_values"] is False
    assert comparison["passed"] is False
    assert result["all_passed_globally"] is False


@pytest.mark.parametrize("overflow", (False, True))
def test_live_parity_phase_wrap_rejects_overflow(overflow):
    reference = _trigger_parity_record()
    candidate = deepcopy(reference)
    if overflow:
        reference["block_triggers"][0]["coa_phases"] = [1.5e308]
        candidate["block_triggers"][0]["coa_phases"] = [-1.5e308]
    else:
        candidate["block_triggers"][0]["coa_phases"][0] += 2 * math.pi

    result = live_batch._verify_parity({
        1: {"branch_standard": reference, "torch_cpu": candidate},
    })

    assert result["all_passed_globally"] is (not overflow)


@pytest.mark.parametrize("parity_passes", (False, True))
def test_live_orchestrator_exit_preserves_diagnostic_artifact(
    tmp_path, monkeypatch, parity_passes,
):
    source = tmp_path / "source"
    (source / "pycbc").mkdir(parents=True)
    (source / "pycbc" / "__init__.py").write_text("")
    artifact = tmp_path / "result.json"
    args = live_batch._parser().parse_args([
        "orchestrate", "--root", str(source), "--python", "/unused/python",
        "--output", str(artifact), "--routes", "branch_standard", "torch_cpu",
        "--batches", "1", "--replicates", "3", "--samples", "3",
    ])

    def summary(values, *, unit, **kwargs):
        return _legacy_summary(list(values), unit=unit) | {
            name: 1.0 for name in ("p50", "p95", "p99", "mean", "p25", "p75")
        }

    def child(**kwargs):
        record = _trigger_parity_record()
        record.update(route=kwargs["route"], batch=kwargs["batch"])
        for name in (
            "python", "pycbc_version", "numpy_version", "torch_version",
        ):
            record[name] = "test"
        for name in (
            "throughput_wps_summary", "latency_block_ms_summary",
            "latency_iteration_ms_summary", "latency_per_waveform_ms_summary",
            "cold_latency_block_ms_summary",
        ):
            record[name] = summary([1.0] * 3, unit="test")
        if not parity_passes and kwargs["route"] == "torch_cpu":
            record["block_triggers"][0]["snrs"] = [9.0]
        return record

    monkeypatch.setattr(live_batch, "_run_child", child)
    monkeypatch.setattr(live_batch, "sample_summary", summary)
    monkeypatch.setattr(
        live_batch, "source_identity",
        lambda path: {"revision": "test", "dirty": False},
    )
    monkeypatch.setattr(
        live_batch, "runtime_metadata",
        lambda: {"hardware": {}, "software": {}},
    )
    if parity_passes:
        live_batch._orchestrate(args)
    else:
        with pytest.raises(SystemExit) as exc:
            live_batch._orchestrate(args)
        assert exc.value.code == 1

    payload = json.loads(artifact.read_text())
    assert payload["parity_analysis"]["all_passed_globally"] is parity_passes
    assert len(payload["records"]) == 6
    assert payload["content_sha256"]


def test_production_artifact_parser_preserves_scope_samples_and_uncertainty():
    artifact = {
        "batches": [2],
        "routes": ["original_standard", "torch_cpu_native"],
        "reference_route": "original_standard",
        "route_definitions": {
            "original_standard": {
                "label": "Original standard CPU",
                "routing_mode": "production_default",
            },
            "torch_cpu_native": {
                "label": "Torch CPU native",
                "routing_mode": "experimental_native",
            },
        },
        "measurement": {
            "scope": "public_library_api",
            "call_surface": "LiveBatchMatchedFilter.process_data",
            "full_cli_end_to_end": {"measured": False},
        },
        "summary_by_batch": {
            "batch_2": {
                "original_standard": {
                    "latency_block_ms": _legacy_summary(
                        [2.0, 2.5, 3.0], unit="ms/block"
                    ),
                    "cold_latency_block_ms": _legacy_summary(
                        [3.0, 3.5, 4.0], unit="ms/block"
                    ),
                    "throughput_wps": _legacy_summary(
                        [8.0, 10.0, 12.0], unit="waveforms/second"
                    ),
                },
                "torch_cpu_native": {
                    "latency_block_ms": _legacy_summary(
                        [1.0, 1.25, 1.5], unit="ms/block"
                    ),
                    "cold_latency_block_ms": _legacy_summary(
                        [2.0, 2.25, 2.5], unit="ms/block"
                    ),
                    "throughput_wps": _legacy_summary(
                        [16.0, 20.0, 24.0], unit="waveforms/second"
                    ),
                    "peak_cuda_memory": {
                        "allocated_bytes": _legacy_summary(
                            [100.0, 110.0], unit="bytes"
                        ),
                        "reserved_bytes": _legacy_summary(
                            [200.0, 220.0], unit="bytes"
                        ),
                        "api": [
                            "torch.cuda.max_memory_allocated",
                            "torch.cuda.max_memory_reserved",
                        ],
                        "boundary": "whole measured child",
                    },
                },
            }
        },
        "parity_analysis": {
            "batch_2": {
                "batch": 2,
                "all_passed": True,
                "replicates": {
                    "replicate_0": {
                        "comparisons": {
                            "torch_cpu_native_vs_original_standard": {
                                "max_snr_diff": 1.0e-5,
                                "max_phase_diff": 2.0e-5,
                                "max_sigmasq_relative_diff": 3.0e-5,
                                "relative_output_l2_diff": 4.0e-5,
                                "passed": True,
                            }
                        }
                    },
                    "replicate_1": {
                        "comparisons": {
                            "torch_cpu_native_vs_original_standard": {
                                "max_snr_diff": 5.0e-5,
                                "max_phase_diff": 6.0e-5,
                                "max_sigmasq_relative_diff": 7.0e-5,
                                "relative_output_l2_diff": 8.0e-5,
                                "passed": True,
                            }
                        }
                    },
                },
            },
            "all_passed_globally": True,
        },
    }

    evidence = parse_live_batch_artifact(artifact)
    native = evidence["routes"]["torch_cpu_native"]

    assert evidence["measurement"]["scope"] == "public_library_api"
    assert native["definition"]["routing_mode"] == "experimental_native"
    assert native["label"].endswith("[experimental]")
    assert native["latency_block_ms"][0]["samples"] == [1.0, 1.25, 1.5]
    assert native["latency_block_ms"][0]["p95"] == pytest.approx(1.475)
    assert native["cold_latency_block_ms"][0]["p50"] == 2.25
    assert native["throughput_wps"][0]["median_ci95"] == {
        "low": 16.0,
        "high": 24.0,
    }
    assert native["speedup_vs_reference"][0]["p50"] == 2.0
    assert native["speedup_vs_reference"][0]["median_ci95"] == {
        "low": pytest.approx(16.0 / 12.0),
        "high": 3.0,
    }
    assert native["peak_cuda_allocated_bytes"][0]["p50"] == 110.0
    assert native["peak_cuda_reserved_bytes"][0]["p50"] == 220.0
    assert native["peak_cuda_memory_measurement"] == {
        "api": [
            "torch.cuda.max_memory_allocated",
            "torch.cuda.max_memory_reserved",
        ],
        "boundary": "whole measured child",
    }

    parity = _parity_cells(evidence)
    cell = parity["torch_cpu_native_vs_original_standard"][2]
    assert cell["passed"] is True
    assert cell["replicate_count"] == 2
    assert cell["max_snr_diff"] == 5.0e-5
    assert cell["relative_output_l2_diff"] == 8.0e-5


def test_legacy_component_artifact_is_not_promoted_to_end_to_end():
    artifact = {
        "results": {
            "synthetic_live_batch": {
                "batch_2": {
                    "original_1t": _legacy_summary([0.15, 0.20, 0.25]),
                    "torch_cpu_16t": _legacy_summary([0.075, 0.10, 0.125]),
                }
            }
        }
    }

    evidence = parse_live_batch_artifact(artifact)
    torch_cpu = evidence["routes"]["torch_cpu"]

    assert evidence["artifact_kind"] == "component_microbenchmarks"
    assert evidence["measurement"]["scope"] == "component_internal"
    assert "_process_batch" in evidence["measurement"]["call_surface"]
    assert evidence["measurement"]["full_cli_end_to_end"]["measured"] is False
    assert torch_cpu["definition"]["routing_mode"] == "unknown_legacy"
    assert torch_cpu["latency_block_ms"][0]["samples"] == [75.0, 100.0, 125.0]
    assert torch_cpu["throughput_wps"][0]["p50"] == 20.0
    assert torch_cpu["throughput_wps"][0]["median_ci95"] == {
        "low": 16.0,
        "high": pytest.approx(2.0 / 0.075),
    }
