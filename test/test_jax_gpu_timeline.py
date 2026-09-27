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


def test_nvml_metric_failures_are_unknown_not_zero():
    collector = profiler.NVMLTelemetryCollector.__new__(
        profiler.NVMLTelemetryCollector
    )
    collector.available = True
    collector.handle = None
    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
    class Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]
    class Process(ctypes.Structure):
        _fields_ = [("pid", ctypes.c_uint),
                    ("usedGpuMemory", ctypes.c_ulonglong)]
    collector.nvmlUtilization_t = Util
    collector.nvmlMemory_t = Memory
    collector.nvmlProcessInfo_t = Process
    collector.nvml = mock.Mock()
    collector.nvml.nvmlDeviceGetUtilizationRates.side_effect = RuntimeError
    collector.nvml.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError
    collector.nvml.nvmlDeviceGetComputeRunningProcesses.side_effect = (
        RuntimeError
    )
    metrics = collector.sample(target_pids={123})
    assert metrics["gpu_util_percent"] is None
    assert metrics["gpu_mem_util_percent"] is None
    assert metrics["gpu_device_used_bytes"] is None
    assert metrics["gpu_device_total_bytes"] is None
    assert metrics["gpu_proc_used_bytes"] is None


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
    assert result["command"][result["command"].index("--sample-rate") + 1] == "2048"
    assert result["workload"]["sample_rate_hz"] == 2048
    assert result["workload"]["signal_dtype"] == "complex64"
    sample = result["telemetry"][0]
    assert sample["gpu_device_used_mib"] == 2.0
    assert sample["gpu_device_total_mib"] == 8.0
    assert "gpu_total_vram_mib" not in sample
    assert "disk_read_mib_per_sec" in sample
    assert "disk_write_mib_per_sec" in sample
    assert sample["disk_used_gib"] is not None
    assert result["telemetry_metadata"]["disk_io_scope"].startswith("whole host")
    assert result["target_pid"] > 0
    assert result["trace_sync"] is None


def test_generic_command_tracks_process_tree_and_structured_stages(tmp_path):
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

    marker = (
        "import sys,time; "
        "print('PYCBC_STAGE_EVENT {\"event\":\"start\","
        "\"stage\":\"filter\",\"templates\":64,\"core\":3}', "
        "file=sys.stderr,flush=True); time.sleep(.03); "
        "print('PYCBC_STAGE_EVENT {\"event\":\"end\","
        "\"stage\":\"filter\"}', file=sys.stderr,flush=True)"
    )
    with mock.patch.object(profiler, "NVMLTelemetryCollector", FakeCollector):
        result = profiler.run_profiling_campaign(
            command=[sys.executable, "-c", marker],
            output_json=tmp_path / "generic.json",
            sampling_interval_ms=5,
            affinity_core="",
            workload={"campaign": "pycbc_live", "templates": 64, "core": 3},
            environment={"PYCBC_TEST_PROFILE": "generic"},
        )

    assert result["returncode"] == 0
    assert result["workload"] == {
        "campaign": "pycbc_live", "templates": 64, "core": 3
    }
    assert result["cwd"] == str(tmp_path.resolve())
    metadata = result["telemetry_metadata"]
    assert metadata["command_cwd"] == str(tmp_path.resolve())
    assert metadata["explicit_environment_keys"] == ["PYCBC_TEST_PROFILE"]
    assert len(metadata["explicit_environment_hash"]) == 64
    assert len(metadata["source_config_hash"]) == 64
    assert result["summary"]["observed_process_count"] >= 1
    assert result["stage_events"][0]["stage"] == "filter"
    assert result["phases"][0]["boundary_source"] == "structured_event"


def test_process_tree_cpu_samples_cached_busy_child(tmp_path):
    child_script = "import time\nend=time.time()+0.25\nwhile time.time()<end: pass"
    script = (
        "import subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable,'-c',{child_script!r}]); "
        "time.sleep(0.3); p.wait()"
    )

    class FakeCollector:
        def __init__(self, device_index=0):
            pass

        def sample(self, target_pid=None):
            return {
                "gpu_util_percent": None,
                "gpu_mem_util_percent": None,
                "gpu_device_used_bytes": None,
                "gpu_device_total_bytes": None,
                "gpu_proc_used_bytes": None,
                "gpu_pcie_rx_kb_per_sec": None,
                "gpu_pcie_tx_kb_per_sec": None,
            }

        def close(self):
            pass

    with mock.patch.object(profiler, "NVMLTelemetryCollector", FakeCollector):
        result = profiler.run_profiling_campaign(
            command=[sys.executable, "-c", script],
            output_json=tmp_path / "busy.json",
            sampling_interval_ms=5,
            affinity_core="",
        )

    assert result["returncode"] == 0
    assert result["summary"]["observed_process_count"] >= 2
    assert any(sample["proc_cpu_percent"] > 0 for sample in result["telemetry"])
    outputs = plotter.plot_campaign_figures(
        result, tmp_path / "busy-overview.png", tmp_path / "busy-zooms"
    )
    assert outputs[0] == tmp_path / "busy-overview.png"
    assert len(outputs) == 2
    assert all(output.stat().st_size > 0 for output in outputs)


def test_process_tree_drops_dead_pids_but_keeps_historical_inventory():
    class FakeProcess:
        def __init__(self, pid, children=(), running=True, zombie=False):
            self.pid = pid
            self._children = list(children)
            self._running = running
            self._zombie = zombie

        def children(self, recursive=True):
            return self._children

        def create_time(self):
            return float(self.pid) + 0.5

        def cpu_percent(self, interval=None):
            return 10.0

        def is_running(self):
            return self._running

        def status(self):
            return profiler.psutil.STATUS_ZOMBIE if self._zombie else "running"

        def memory_info(self):
            return type("Memory", (), {"rss": 1024})()

    root = FakeProcess(100)
    child = FakeProcess(101)
    root._children = [child]
    active = set()
    known = set()
    cache = {}
    identity = {}
    with mock.patch.object(
        profiler.psutil, "Process", side_effect=lambda pid: {
            100: root, 101: child
        }[pid]
    ):
        profiler._sample_process_tree(
            100, known, cache, active, identity
        )
        assert known == {100, 101}
        root._children = []
        _, _, live, observed = profiler._sample_process_tree(
            100, known, cache, active, identity
        )
        assert live == [100, 101]
        child._running = False
        _, _, live, observed = profiler._sample_process_tree(
            100, known, cache, active, identity
        )
    assert live == [100]
    assert observed == [100]
    assert known == {100, 101}


def test_membership_receipt_uses_compact_changes(tmp_path):
    script = (
        "import subprocess,sys,time; "
        "[subprocess.Popen([sys.executable,'-c','import time; time.sleep(.03)']) "
        "for _ in range(8)]; time.sleep(.18)"
    )
    result = profiler.run_profiling_campaign(
        command=[sys.executable, "-c", script],
        output_json=tmp_path / "compact.json",
        sampling_interval_ms=5,
        affinity_core="",
    )
    membership = result["process_membership"]
    assert membership
    assert all("observed_pids" not in item for item in membership)
    assert all(set(item) == {
        "elapsed_sec", "added_pids", "removed_pids", "live_pids"
    } for item in membership)
    assert len(membership) <= len(result["telemetry"])
    assert result["telemetry_metadata"]["process_membership_encoding"] == (
        "changes; historical inventory is in process_tree.observed_pids"
    )


def test_structured_stage_pairing_uses_pid_and_monotonic_clock():
    events = profiler._parse_stage_events(
        [
            (0.4, 'PYCBC_STAGE_EVENT {"event":"start","stage":"filter",'
             '"pid":11,"monotonic_ns":1010000000}'),
            (0.5, 'PYCBC_STAGE_EVENT {"event":"start","stage":"filter",'
             '"pid":12,"monotonic_ns":1020000000,"elapsed_sec":99}'),
            (0.6, 'PYCBC_STAGE_EVENT {"event":"end","stage":"filter",'
             '"pid":11,"monotonic_ns":1030000000}'),
            (0.7, 'PYCBC_STAGE_EVENT {"event":"end","stage":"filter",'
             '"pid":12,"monotonic_ns":1040000000}'),
        ],
        total_wall=2,
        origin_monotonic_ns=1000000000,
    )
    phases = profiler._structured_stage_phases(events, 2)
    assert [(p["pid"], p["start_sec"], p["end_sec"]) for p in phases] == [
        (11, 0.01, 0.03),
        (12, 0.02, 0.04),
    ]


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


def test_legacy_plot_uses_executable_and_manual_zoom(tmp_path):
    data = receipt()
    data["workload"] = {"executable": "pycbc_live"}
    with mock.patch.object(plt, "close"):
        plotter.plot_profiling_timeline(
            data, tmp_path / "legacy-zoom.png", zoom_start=1.0, zoom_end=2.0
        )
        fig = plt.gcf()
    try:
        assert fig._suptitle.get_text().startswith(
            "pycbc_live · CPU, GPU and device transfers over time"
        )
        assert fig.axes[-1].get_xlim() == (1.0, 2.0)
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


def test_stage_zoom_windows_skip_open_events_and_keep_padding():
    data = {
        "phases": [
            {"name": "conditioning", "label": "Conditioning",
             "start_sec": 1.0, "end_sec": 3.0},
            {"name": "open", "start_sec": 3.0, "end_sec": None},
        ]
    }
    windows = plotter.stage_zoom_windows(data, padding=0.1)
    assert windows == [{
        "index": 1,
        "name": "conditioning",
        "label": "Conditioning",
        "start_sec": 0.8,
        "end_sec": 3.2,
    }]


def test_stage_zoom_windows_are_bounded_representatives():
    phases = [
        {"name": "search", "start_sec": 0, "end_sec": 100},
        *[
            {"name": f"filter_batch_{i}", "label": f"Batch {i}",
             "start_sec": 100 + i, "end_sec": 101 + i}
            for i in range(3000)
        ],
    ]
    windows = plotter.stage_zoom_windows(
        {"phases": phases}, max_windows=12, max_window_sec=20
    )
    assert len(windows) == 1
    assert windows[0]["name"] == "filter_batch_0"
    assert windows[0]["end_sec"] - windows[0]["start_sec"] == pytest.approx(1.04)


def test_overview_collapsing_preserves_stage_gaps():
    phases = [
        {
            "name": f"filter_batch_{index}",
            "label": "filter",
            "start_sec": float(index) * 0.51,
            "end_sec": float(index) * 0.51 + 0.5,
        }
        for index in range(300)
    ]
    overview = plotter._overview_phases(phases, max_spans=16)
    aggregate = next(item for item in overview if item.get("count", 0) > 1)
    assert len(aggregate["segments"]) > 1
    assert aggregate["end_sec"] - aggregate["start_sec"] > sum(
        end - start for start, end in aggregate["segments"]
    )


def test_default_overview_is_bounded_for_dense_live_markers():
    phases = [
        {
            "name": name,
            "label": name,
            "start_sec": float(index),
            "end_sec": float(index + 1),
        }
        for index in range(100)
        for name in ("block", "frame_read", "postprocess", "filter")
    ]
    assert len(plotter._overview_phases(phases)) <= 24


def test_manual_zoom_keeps_middle_stage_markers():
    phases = [
        {"name": f"stage_{index}", "start_sec": index,
         "end_sec": index + 0.5}
        for index in range(100)
    ]
    selected = plotter._phases_for_view(phases, zoom_start=49, zoom_end=51)
    assert [phase["name"] for phase in selected] == [
        "stage_49", "stage_50"
    ]


def test_cpu_only_plot_has_all_stage_lanes_and_no_gpu_panels(tmp_path):
    data = receipt()
    data["workload"] = {
        "executable": "pycbc_live",
        "processing_scheme": "cpu:1",
        "resources": {"gpus": 0},
    }
    data["phases"] = [
        {"name": "filter", "rank": 0, "start_sec": 0.1, "end_sec": 0.3},
        {"name": "filter", "rank": 0, "start_sec": 2.2, "end_sec": 2.8},
        {"name": "output", "rank": 1, "start_sec": 1.2, "end_sec": 1.4},
    ]
    with mock.patch.object(plt, "close"):
        plotter.plot_profiling_timeline(
            data, tmp_path / "cpu.png", zoom_start=1.0, zoom_end=2.0
        )
        fig = plt.gcf()
    try:
        assert len(fig.axes) == 3
        assert fig.axes[0].get_xlim() == (1.0, 2.0)
        assert len(fig.axes[0].collections) == 2
        assert fig.axes[0].get_yticklabels()[0].get_text() == "filter · rank 0"
        assert "CPU process utilization" in fig._suptitle.get_text()
        footer = " ".join(text.get_text() for text in fig.texts)
        assert "No GPU telemetry is plotted" in footer
        assert "GPU activity" not in fig._suptitle.get_text()
    finally:
        plt.close(fig)
