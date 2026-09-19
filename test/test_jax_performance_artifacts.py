"""Pure-Python checks for JAX performance evidence tooling."""

import json

import pytest

from tools.benchmark_artifact import (
    SCHEMA_VERSION,
    load_mergeable_artifact,
    sample_summary,
    seal_artifact,
    summary_median_ci95,
    summary_statistic,
)


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
    path.write_text(json.dumps(payload))
    loaded = load_mergeable_artifact(
        path,
        artifact_type="component_microbenchmarks",
        compatibility_sha256="compatible",
    )
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["schema"] == SCHEMA_VERSION
