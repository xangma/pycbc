"""Regression checks for capacity units in the published benchmark plots."""

import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from tools.plot_jax_search_capacity import (  # noqa: E402
    capacity_summary,
    plot_inspiral,
    plot_live,
)


STATIC = Path(__file__).resolve().parents[1] / "docs" / "_static"


def test_live_uses_valid_advance_and_filter_stage(tmp_path):
    data = json.loads((STATIC / "jax_microbenchmarks.json").read_text())
    cells = data["experiments"]["streaming_n131072"]["arms"]
    # Values use 56 valid seconds, not the 64-second FFT duration or total
    # waveform-plus-transfer-plus-filter time. CUDA means a GPU + host core.
    for arm, expected in (("branch_cpu", 10854),
                          ("jax_cuda_lal", 212857)):
        timings = cells[arm]["batches"]["1"]["samples_seconds"]["filter"]
        assert round(capacity_summary(56, timings)[0]) == expected
    output = tmp_path / "live.png"
    plot_live(data, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_inspiral_distinguishes_wall_and_calculation_capacity(tmp_path):
    data = json.loads((STATIC / "jax_search_comparison.json").read_text())
    expected = {"branch_cpu": (10675, 14848),
                "jax_cpu_batched": (1409, 1568),
                "jax_cuda_batched": (42184, 384650)}
    for arm, capacities in expected.items():
        for field, target in zip(("wall_seconds", "calc_seconds"), capacities):
            samples = [run["timing"][field]
                       for name, run in data["runs"].items()
                       if name.startswith(arm + "_rep")]
            mid, low, high = capacity_summary(384 * 1904, samples)
            assert round(mid) == target
            assert low <= mid <= high
    output = tmp_path / "inspiral.png"
    plot_inspiral(data, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_live_rejects_advance_exceeding_fft_duration(tmp_path):
    with pytest.raises(ValueError, match="block advance"):
        plot_live({}, tmp_path / "invalid.png", advance_seconds=65)
