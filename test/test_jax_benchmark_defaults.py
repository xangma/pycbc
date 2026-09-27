"""Default benchmark and plotting modes avoid publishing short proxies."""

import json
import sys

import numpy as np
import pytest

from tools import bench_jax_performance as benchmark
from tools import plot_jax_benchmarks as plotting


def test_benchmark_lengths_are_explicit_and_lossless():
    assert benchmark.default_lengths() == [1048576]
    assert benchmark.default_lengths(True) == [131072, 1048576]
    assert benchmark.track_name_for_length(131072) == "streaming_n131072"
    assert benchmark.track_name_for_length(1048576) == "inspiral_n1048576"
    assert benchmark.track_name_for_length(123456) == "transform_n123456"
    assert benchmark.BENCHMARK_SAMPLE_RATE == 2048.0
    assert benchmark.BENCHMARK_PRECISION == "single"
    assert benchmark.BENCHMARK_COMPLEX_DTYPE == np.complex64


def test_double_precision_is_rejected_for_published_benchmark():
    with pytest.raises(ValueError, match="single precision only"):
        benchmark._require_benchmark_precision("double")


def test_plot_defaults_skip_historical_diagnostics(monkeypatch, tmp_path):
    artifact = tmp_path / "receipt.json"
    artifact.write_text(json.dumps({"experiments": {}}), encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        plotting,
        "plot_throughput_scaling",
        lambda data, output, include_historical=False: calls.append(
            ("throughput", include_historical)
        ),
    )
    monkeypatch.setattr(
        plotting,
        "plot_speedup_matrix",
        lambda *args: calls.append(("speedup",)),
    )
    monkeypatch.setattr(
        plotting,
        "plot_latency_breakdown",
        lambda *args: calls.append(("latency",)),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_jax_benchmarks.py",
            "--input", str(artifact),
            "--output-dir", str(tmp_path),
        ],
    )
    plotting.main()

    assert calls == [("throughput", False)]


def test_plot_historical_diagnostics_are_opt_in(monkeypatch, tmp_path):
    artifact = tmp_path / "receipt.json"
    artifact.write_text(json.dumps({"experiments": {}}), encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        plotting,
        "plot_throughput_scaling",
        lambda data, output, include_historical=False: calls.append(
            ("throughput", include_historical)
        ),
    )
    monkeypatch.setattr(
        plotting,
        "plot_speedup_matrix",
        lambda *args: calls.append(("speedup",)),
    )
    monkeypatch.setattr(
        plotting,
        "plot_latency_breakdown",
        lambda *args: calls.append(("latency",)),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_jax_benchmarks.py",
            "--input",
            str(artifact),
            "--output-dir",
            str(tmp_path),
            "--historical-diagnostics",
        ],
    )
    plotting.main()

    assert calls == [("throughput", True), ("speedup",), ("latency",)]
