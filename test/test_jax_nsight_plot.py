"""Focused tests for the process-scoped Nsight timeline rendering."""

import importlib.util
from pathlib import Path
from unittest import mock

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "plot_jax_gpu_timeline_nsight",
        ROOT / "tools" / "plot_jax_gpu_timeline.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plotter = load_tool()


def receipt_with_cuda_trace():
    return {
        "telemetry": [
            {
                "elapsed_sec": time,
                "proc_cpu_percent": 80.0,
                "gpu_proc_vram_mib": 100.0 + time,
                "proc_rss_mib": 300.0 + time,
                # These must not be used for CUDA panels below.
                "gpu_util_percent": 99.0,
                "gpu_pcie_rx_kb_per_sec": 999.0,
                "gpu_pcie_tx_kb_per_sec": 888.0,
            }
            for time in (0.0, 1.0, 2.0, 3.0)
        ],
        "phases": [
            {
                "name": "startup_imports",
                "label": "Startup & Imports",
                "start_sec": 0.0,
                "end_sec": 1.0,
            },
            {
                "name": "filter_batch_1",
                "label": "Filter Batch 1 (0–15)",
                "start_sec": 1.0,
                "end_sec": 2.0,
            },
            {
                "name": "filter_batch_12",
                "label": "Filter Batch 12 (176–191)",
                "start_sec": 2.0,
                "end_sec": 3.0,
            },
        ],
        "summary": {"total_wall_sec": 3.0},
        "workload": {"processing_scheme": "jax:cuda:0", "batch_size": 16},
        "cuda_trace": {
            "source": "Nsight Systems / CUPTI",
            "target_pid": 4242,
            "device_id": 0,
            "origin_ns": 123,
            "bin_width_ms": 100,
            "bins": [
                {
                    "start_sec": 1.0,
                    "end_sec": 1.1,
                    "kernel_active_percent": 37.5,
                    "h2d_bytes": 2 * 1024**2,
                    "d2h_bytes": 1 * 1024**2,
                    "d2d_bytes": 3 * 1024**2,
                },
                {
                    "start_sec": 1.1,
                    "end_sec": 1.2,
                    "kernel_active_percent": 12.5,
                    "h2d_bytes": 0,
                    "d2h_bytes": 4 * 1024**2,
                    "d2d_bytes": 0,
                },
            ],
            "transfers": [
                {
                    "start_sec": 1.01, "end_sec": 1.03,
                    "bytes": 2**20, "direction": "H2D",
                },
                {
                    "start_sec": 1.04, "end_sec": 1.06,
                    "bytes": 2**20, "direction": "D2H",
                },
                {
                    "start_sec": 1.07, "end_sec": 1.09,
                    "bytes": 2**20, "direction": "D2D",
                },
            ],
            "kernels": [{"start_sec": 1.0, "end_sec": 1.08}],
            "totals": {"kernel_count": 1, "transfer_count": 3},
        },
    }


def test_nsight_trace_drives_process_panels_and_zoom(tmp_path):
    data = receipt_with_cuda_trace()
    with mock.patch.object(plt, "close"):
        plotter.plot_profiling_timeline(
            data, tmp_path / "nsight.png", zoom_start=1.0, zoom_end=1.2
        )
        fig = plt.gcf()
    try:
        assert len(fig.axes) == 7
        assert fig.axes[-1].get_xlim() == (1.0, 1.2)
        assert fig.axes[1].get_ylabel() == "Kernel active (%)"
        assert fig.axes[2].get_ylabel() == "CUDA copies (MiB/bin)"
        assert fig.axes[3].get_ylabel() == "CUDA events"

        kernel_bars = [
            patch for patch in fig.axes[1].patches
            if patch.get_facecolor()[:3] == (33 / 255, 140 / 255, 76 / 255)
        ]
        assert [bar.get_height() for bar in kernel_bars] == [37.5, 12.5]

        transfer_heights = [bar.get_height() for bar in fig.axes[2].patches]
        assert 2.0 in transfer_heights
        assert 4.0 in transfer_heights
        assert len(fig.axes[3].collections) >= 4

        ribbon_text = " ".join(text.get_text() for text in fig.axes[0].texts)
        assert "B1" in ribbon_text
        assert "B12" not in ribbon_text  # Outside the requested zoom.
        assert "Startup & Imports" not in ribbon_text  # Also outside the zoom.
        assert not any(
            text.get_rotation() == 90 and "Filter Batch" in text.get_text()
            for text in fig.axes[0].texts
        )
        footer = " ".join(text.get_text() for text in fig.texts)
        assert "target PID 4242" in footer
        assert "CPU/memory sampled" in footer
        assert "host-log stage boundaries approximate" in footer
        assert "not bus bandwidth" in footer
        assert (tmp_path / "nsight.png").stat().st_size > 0
    finally:
        plt.close(fig)


def test_empty_nsight_trace_does_not_fallback_to_nvml(tmp_path):
    data = receipt_with_cuda_trace()
    data["cuda_trace"] = {
        "source": "Nsight Systems / CUPTI",
        "target_pid": 4242,
        "device_id": 0,
        "bin_width_ms": 100,
        "bins": [],
        "transfers": [],
        "kernels": [],
        "totals": {},
    }
    with mock.patch.object(plt, "close"):
        plotter.plot_profiling_timeline(data, tmp_path / "empty-nsight.png")
        fig = plt.gcf()
    try:
        assert len(fig.axes) == 7
        assert fig.axes[1].get_ylabel() == "Kernel active (%)"
        assert fig.axes[2].get_ylabel() == "CUDA copies (MiB/bin)"
        assert not any(
            "Whole device" in text.get_text() for text in fig.axes[2].texts
        )
        assert any(
            "trace present" in text.get_text() for text in fig.axes[1].texts
        )
        assert any(
            "No exact CUDA events" in text.get_text()
            for text in fig.axes[3].texts
        )
        assert (tmp_path / "empty-nsight.png").stat().st_size > 0
    finally:
        plt.close(fig)
