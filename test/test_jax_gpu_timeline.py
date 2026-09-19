"""Telemetry regressions without requiring a GPU or the PyCBC runtime."""

import ctypes
import importlib.util
import os
from pathlib import Path
import sys
from unittest import mock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profiler = load_tool("profile_jax_gpu_timeline")
plotter = load_tool("plot_jax_gpu_timeline")


def test_pcie_direction_units_zero_and_failed_queries():
    collector = profiler.NVMLTelemetryCollector.__new__(
        profiler.NVMLTelemetryCollector
    )
    collector.handle = None
    collector.nvml = mock.Mock()

    def query(handle, counter, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_uint))[0] = {
            0: 0,
            1: 123456,
        }[counter]
        return 0

    collector.nvml.nvmlDeviceGetPcieThroughput.side_effect = query
    assert collector._pcie_throughput(1) == 123456
    assert collector._pcie_throughput(0) == 0
    collector.nvml.nvmlDeviceGetPcieThroughput.side_effect = None
    collector.nvml.nvmlDeviceGetPcieThroughput.return_value = 3
    assert collector._pcie_throughput(1) is None
    collector.nvml = object()
    assert collector._pcie_throughput(0) is None


def test_campaign_emits_distinct_device_memory_fields(tmp_path):
    executable = tmp_path / "child.py"
    executable.write_text("import time; time.sleep(0.15)\n")

    class FakeCollector:
        def __init__(self, device_index=0):
            pass

        def sample(self, target_pid=None):
            return {
                "gpu_util_percent": 10.0,
                "gpu_mem_util_percent": 20.0,
                "gpu_device_used_bytes": 2 * 1024**2,
                "gpu_device_total_bytes": 8 * 1024**2,
                "gpu_proc_used_bytes": 1 * 1024**2,
                "gpu_pcie_rx_kb_per_sec": None,
                "gpu_pcie_tx_kb_per_sec": None,
            }

        def close(self):
            pass

    with mock.patch.object(profiler, "NVMLTelemetryCollector", FakeCollector):
        result = profiler.run_profiling_campaign(
            executable=executable,
            frame_file=tmp_path / "frame.gwf",
            bank_file=tmp_path / "bank.hdf",
            output_hdf=tmp_path / "triggers.hdf",
            output_json=tmp_path / "receipt.json",
            python_bin=sys.executable,
            sampling_interval_ms=10,
            affinity_core="",
        )

    assert result["returncode"] == 0
    sample = result["telemetry"][0]
    assert sample["gpu_device_used_mib"] == 2.0
    assert sample["gpu_device_total_mib"] == 8.0
    assert "gpu_total_vram_mib" not in sample
    assert result["target_pid"] > 0
    assert result["trace_sync"] is None


def test_nvtx_sync_brackets_origin_and_records_target_pid(tmp_path):
    executable = tmp_path / "child.py"
    executable.write_text("import time; time.sleep(0.05)\n")
    calls = []

    class FakeCollector:
        def __init__(self, device_index=0):
            pass

        def sample(self, target_pid=None):
            return {
                "gpu_util_percent": 0.0,
                "gpu_mem_util_percent": 0.0,
                "gpu_device_used_bytes": 0,
                "gpu_device_total_bytes": 0,
                "gpu_proc_used_bytes": 0,
                "gpu_pcie_rx_kb_per_sec": None,
                "gpu_pcie_tx_kb_per_sec": None,
            }

        def close(self):
            pass

    class FakeNvtx:
        def __init__(self, library):
            assert library == "libnvToolsExt.so"
            self.nvtxMarkA = self
            self.argtypes = None
            self.restype = None

        def __call__(self, value):
            calls.append(value.decode("ascii"))

    with (
        mock.patch.object(profiler, "NVMLTelemetryCollector", FakeCollector),
        mock.patch.object(
            profiler.ctypes.util,
            "find_library",
            return_value="libnvToolsExt.so",
        ),
        mock.patch.object(profiler.ctypes, "CDLL", FakeNvtx),
    ):
        result = profiler.run_profiling_campaign(
            executable=executable,
            frame_file=tmp_path / "frame.gwf",
            bank_file=tmp_path / "bank.hdf",
            output_hdf=tmp_path / "triggers.hdf",
            output_json=tmp_path / "receipt.json",
            python_bin=sys.executable,
            sampling_interval_ms=10,
            affinity_core="",
            nvtx_sync=True,
        )

    assert calls == ["pycbc.timeline.init", "pycbc.timeline.origin"]
    assert result["target_pid"] > 0
    assert result["target_pid"] != os.getpid()
    assert result["trace_sync"]["marker"] == "pycbc.timeline.origin"
    assert result["trace_sync"]["collector_pid"] == os.getpid()
    assert result["trace_sync"]["clock_uncertainty_sec"] >= 0.0
    assert all(sample["elapsed_sec"] >= 0 for sample in result["telemetry"])


def test_nvtx_sync_requires_nvtools_library(tmp_path):
    class FakeCollector:
        def __init__(self, device_index=0):
            pass

    with (
        mock.patch.object(profiler, "NVMLTelemetryCollector", FakeCollector),
        mock.patch.object(
            profiler.ctypes.util, "find_library", return_value=None
        ),
    ):
        with pytest.raises(RuntimeError, match="requires libnvToolsExt"):
            profiler.run_profiling_campaign(
                executable=tmp_path / "child.py",
                frame_file=tmp_path / "frame.gwf",
                bank_file=tmp_path / "bank.hdf",
                output_hdf=tmp_path / "triggers.hdf",
                output_json=tmp_path / "receipt.json",
                affinity_core="",
                nvtx_sync=True,
            )


def test_missing_nvml_does_not_report_zero_transfers():
    with mock.patch.object(profiler.ctypes, "CDLL", side_effect=OSError):
        collector = profiler.NVMLTelemetryCollector()
    sample = collector.sample()
    assert sample["gpu_pcie_rx_kb_per_sec"] is None
    assert sample["gpu_pcie_tx_kb_per_sec"] is None


def receipt():
    return {
        "telemetry": [
            {
                "elapsed_sec": t,
                "gpu_util_percent": 20,
                "gpu_proc_vram_mib": 100,
                "proc_cpu_percent": 200,
                "proc_rss_mib": 300,
            }
            for t in range(3)
        ],
        "phases": [],
        "summary": {"total_wall_sec": 3},
    }


def test_legacy_receipt_shows_unavailable_not_fabricated_rates(tmp_path):
    with mock.patch.object(plt, "close"):
        plotter.plot_profiling_timeline(receipt(), tmp_path / "old.png")
        fig = plt.gcf()
    try:
        assert len(fig.axes) == 6
        assert "unknown processing scheme" in fig._suptitle.get_text()
        assert any("not recorded" in t.get_text() for t in fig.axes[2].texts)
        assert all(len(line.get_ydata()) == 0 for line in fig.axes[2].lines)
        assert fig.axes[4].get_ylim()[1] > 200
        assert (tmp_path / "old.png").stat().st_size > 0
    finally:
        plt.close(fig)


def test_plot_title_uses_receipt_processing_scheme(tmp_path):
    data = receipt()
    data["workload"] = {"processing_scheme": "jax:cuda:0"}
    with mock.patch.object(plt, "close"):
        plotter.plot_profiling_timeline(data, tmp_path / "scheme.png")
        fig = plt.gcf()
    try:
        assert "jax:cuda:0" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_transfer_gaps_and_direction_are_preserved(tmp_path):
    data = receipt()
    for sample, rx, tx in zip(
        data["telemetry"], [1000, None, 0], [0, 2000, None]
    ):
        sample.update(gpu_pcie_rx_kb_per_sec=rx, gpu_pcie_tx_kb_per_sec=tx)
    with mock.patch.object(plt, "close"):
        plotter.plot_profiling_timeline(data, tmp_path / "new.png")
        fig = plt.gcf()
    try:
        rx_line, tx_line = fig.axes[2].lines
        np.testing.assert_equal(rx_line.get_ydata(), [1000, np.nan, 0])
        np.testing.assert_equal(tx_line.get_ydata(), [0, 2000, np.nan])
        assert "Into GPU" in rx_line.get_label()
        assert "Out of GPU" in tx_line.get_label()
    finally:
        plt.close(fig)
