"""Focused tests for de-duplicated JAX memory plotting."""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plot_jax_memory", ROOT / "tools" / "plot_jax_memory.py"
)
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


def test_distinct_pointer_group_uses_max_and_marks_unknown():
    snapshot = {
        "live_arrays": {
            "arrays": [
                {"device": "cuda:0", "unsafe_buffer_pointer": 7, "nbytes": 10},
                {"device": "cuda:0", "unsafe_buffer_pointer": 7, "nbytes": 40},
                {"device": "cuda:1", "unsafe_buffer_pointer": 7, "nbytes": 3},
                {
                    "device": "cuda:0", "unsafe_buffer_pointer": None,
                    "nbytes": 9,
                },
            ]
        }
    }
    result = plotter.distinct_live_bytes(snapshot)
    assert result["distinct_bytes"] == 43
    assert result["unknown_count"] == 1
    assert result["unknown_logical_bytes"] == 9


def test_plot_accepts_one_snapshot_and_keeps_unknown_unavailable(tmp_path):
    data = {
        "snapshots": [{
            "label": "batch 1",
            "process_rss_bytes": 2**20,
            "live_arrays": {"arrays": [
                {"device": "cuda:0", "unsafe_buffer_pointer": None,
                 "nbytes": 2**20},
            ]},
            "device_memory_stats": [{"memory_stats": {
                "bytes_in_use": 3 * 2**20,
                "pool_bytes": 4 * 2**20,
                "peak_bytes_in_use": 5 * 2**20,
            }}],
        }]
    }
    output = tmp_path / "memory.png"
    plotter.plot_memory_profile(data, output)
    assert output.stat().st_size > 0
