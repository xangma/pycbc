#!/usr/bin/env python3
# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Generate GPU, CPU, memory and transfer timeline profiling figures.

Reads the structured JSON receipt produced by profile_jax_gpu_timeline.py and
generates a multi-panel timeline showing GPU activity, PCIe traffic, VRAM,
CPU utilization, and host memory RSS across logged pipeline stages.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

PHASE_PALETTE = [
    {
        "bg": "#edf2f7",
        "edge": "#4a5568",
        "bar": "#cbd5e0",
        "label": "#2d3748",
    },  # Startup & Imports
    {
        "bg": "#fefcbf",
        "edge": "#d69e2e",
        "bar": "#faf089",
        "label": "#744210",
    },  # Frame Reading
    {
        "bg": "#e6fffa",
        "edge": "#319795",
        "bar": "#b2f5ea",
        "label": "#234e52",
    },  # Conditioning & PSD
    {
        "bg": "#feebc8",
        "edge": "#dd6b20",
        "bar": "#fbd38d",
        "label": "#7b341e",
    },  # Bank & JIT Warmup
    {
        "bg": "#c6f6d5",
        "edge": "#38a169",
        "bar": "#9ae6b4",
        "label": "#22543d",
    },  # Batch 1
    {
        "bg": "#9ae6b4",
        "edge": "#2f855a",
        "bar": "#68d391",
        "label": "#1c4532",
    },  # Batch 2
    {
        "bg": "#68d391",
        "edge": "#276749",
        "bar": "#48bb78",
        "label": "#1c4532",
    },  # Batch 3
    {
        "bg": "#e9d8fd",
        "edge": "#805ad5",
        "bar": "#d6bcfa",
        "label": "#44337a",
    },  # Clustering & HDF5
]

CUDA_COLORS = {
    "kernels": "#218c4c",
    "H2D": "#0072b2",
    "D2H": "#d55e00",
    "D2D": "#7b2cbf",
}


def _load_receipt(path):
    """Load a JSON receipt, including the committed ``.json.gz`` form."""
    opener = gzip.open if Path(path).suffix.lower() == ".gz" else Path.open
    with opener(path, "rt") as stream:
        return json.load(stream)


def _trace_number(value, default=np.nan):
    """Return a finite float for a trace field, or *default*."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _trace_phase_label(phase):
    """Make batch identity and host stage explicit in the ribbon."""
    name = str(phase.get("name", ""))
    label = str(phase.get("label", name or "Stage"))
    if name.startswith("filter_batch_"):
        suffix = name.removeprefix("filter_batch_")
        return f"B{suffix}\n{label}"
    details = []
    if phase.get("templates") is not None:
        details.append(f"templates={phase['templates']}")
    if phase.get("core") is not None:
        details.append(f"core={phase['core']}")
    return label + (f" [{', '.join(details)}]" if details else "")


def _workload_detail(workload):
    fields = []
    for key in ("templates", "template_count", "core", "cores", "batch_size"):
        if workload.get(key) is not None and key not in {
            "template_count" if "templates" in workload else "",
            "cores" if "core" in workload else "",
        }:
            fields.append(f"{key}={workload[key]}")
    return ", ".join(fields)


def _stage_family(phase):
    """Group repeated batch boundaries for bounded automatic plots."""
    name = str(phase.get("name", "stage"))
    if name.startswith("filter_batch_"):
        return "filter_batches"
    return name


def _phase_segments(phase):
    """Return drawable stage intervals, preserving gaps in aggregates."""
    raw_segments = phase.get("segments")
    if raw_segments:
        segments = raw_segments
    else:
        segments = [(phase.get("start_sec"), phase.get("end_sec"))]
    result = []
    for segment in segments:
        try:
            start, end = float(segment[0]), float(segment[1])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(start) and np.isfinite(end) and end > start:
            result.append((start, end))
    return result


def _overview_phases(phases, max_spans=24):
    """Collapse dense adjacent stage events before drawing the overview."""
    complete = []
    for phase in phases or []:
        try:
            start = float(phase["start_sec"])
            end = float(phase["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(start) and np.isfinite(end) and end > start:
            complete.append((start, end, phase))
    if len(complete) <= max_spans:
        return [phase for _, _, phase in complete]

    collapsed = []
    for start, end, phase in complete:
        family = _stage_family(phase)
        if collapsed:
            previous = collapsed[-1]
            gap = max(0.0, start - previous["end_sec"])
            if previous["family"] == family and gap <= 0.02:
                previous["end_sec"] = max(previous["end_sec"], end)
                previous["segments"].append((start, end))
                previous["count"] += 1
                previous["label"] = f"{family} ({previous['count']} intervals)"
                continue
        collapsed.append({
            "family": family,
            "start_sec": start,
            "end_sec": end,
            "label": str(phase.get("label", family)),
            "name": str(phase.get("name", family)),
            "count": 1,
            "segments": [(start, end)],
        })
    if len(collapsed) <= max_spans:
        # A long run can contain thousands of one-second batch markers. Keep
        # the aggregate visibly discontinuous without creating thousands of
        # matplotlib artists; the process telemetry remains fully sampled.
        for item in collapsed:
            segments = item.get("segments", [])
            if len(segments) > 64:
                keep = 32
                item["segments"] = segments[:keep] + segments[-keep:]
        return collapsed

    # Preserve the launch and completion context while bounding rendering
    # work for pathological marker streams.
    keep = max_spans // 2
    selected = collapsed[:keep] + collapsed[-(max_spans - keep):]
    for item in selected:
        segments = item.get("segments", [])
        if len(segments) > 64:
            item["segments"] = segments[:32] + segments[-32:]
    return selected


def _phases_for_view(phases, zoom_start=None, zoom_end=None):
    """Use bounded phases for overviews and exact markers for manual zooms."""
    if zoom_start is None and zoom_end is None:
        return _overview_phases(phases)
    lower = -np.inf if zoom_start is None else float(zoom_start)
    upper = np.inf if zoom_end is None else float(zoom_end)
    visible = []
    for phase in phases or []:
        try:
            start = float(phase["start_sec"])
            end = float(phase["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(start) and np.isfinite(end) and end > start:
            if end > lower and start < upper:
                visible.append(phase)
    return _overview_phases(visible, max_spans=64)


def _plot_cuda_trace_timeline(
    data, output_path, zoom_start=None, zoom_end=None
):
    """Render process-scoped Nsight/CUPTI events and sampled host telemetry.

    A present ``cuda_trace`` is authoritative for the CUDA panels, including
    when it contains no bins or events.  This prevents an empty Nsight trace
    from being silently replaced with unrelated device-wide NVML telemetry.
    """
    # Keep this path authoritative whenever the key is present.  An extractor
    # may legitimately produce an empty trace when no matching CUPTI rows are
    # available, and falling back to device-wide NVML would mislabel it.
    trace = data.get("cuda_trace") or {}
    telemetry = data.get("telemetry") or []
    phases = _phases_for_view(
        data.get("phases") or [], zoom_start=zoom_start, zoom_end=zoom_end
    )
    summary = data.get("summary") or {}

    def sample_values(key):
        return np.array(
            [_trace_number(sample.get(key)) for sample in telemetry],
            dtype=float,
        )

    bins = []
    for item in trace.get("bins", []) or []:
        start = _trace_number(item.get("start_sec"))
        end = _trace_number(item.get("end_sec"))
        if np.isfinite(start) and np.isfinite(end) and end > start:
            bins.append((start, end, item))

    transfers = []
    for item in trace.get("transfers", []) or []:
        start = _trace_number(item.get("start_sec"))
        end = _trace_number(item.get("end_sec"))
        direction = str(item.get("direction", "")).upper()
        if (
            np.isfinite(start)
            and np.isfinite(end)
            and end > start
            and direction in ("H2D", "D2H", "D2D")
        ):
            transfers.append((start, end, item, direction))

    kernels = []
    for item in trace.get("kernels", []) or []:
        start = _trace_number(item.get("start_sec"))
        end = _trace_number(item.get("end_sec"))
        if np.isfinite(start) and np.isfinite(end) and end > start:
            kernels.append((start, end, item))

    extents = []
    extents.extend((start, end) for start, end, _ in bins)
    extents.extend((start, end) for start, end, _, _ in transfers)
    extents.extend((start, end) for start, end, _ in kernels)
    for phase in phases:
        start = _trace_number(phase.get("start_sec"))
        end = _trace_number(phase.get("end_sec"))
        if np.isfinite(start) and np.isfinite(end) and end > start:
            extents.append((start, end))
    sample_times = np.array(
        [_trace_number(sample.get("elapsed_sec")) for sample in telemetry],
        dtype=float,
    )
    finite_sample_times = sample_times[np.isfinite(sample_times)]
    if len(finite_sample_times):
        extents.append((finite_sample_times[0], finite_sample_times[-1]))
    wall_extent = _trace_number(summary.get("total_wall_sec"), np.nan)
    if np.isfinite(wall_extent) and wall_extent > 0:
        extents.append((0.0, wall_extent))
    if extents:
        data_start = min(start for start, _ in extents)
        data_end = max(end for _, end in extents)
    else:
        data_start, data_end = 0.0, 1.0
    if data_end <= data_start:
        data_end = data_start + 1.0

    if zoom_start is None:
        zoom_start = data_start
    if zoom_end is None:
        zoom_end = data_end
    if zoom_start >= zoom_end:
        raise ValueError("zoom-start must be less than zoom-end")

    # Clip exact CUDA events before constructing any matplotlib artists.  A
    # whole-process overview may contain many thousands of events; retaining
    # the full lists for extent discovery is cheap, while drawing only the
    # events intersecting the requested viewport keeps zooms responsive.
    visible_bins = [
        item for item in bins
        if item[1] > zoom_start and item[0] < zoom_end
    ]
    visible_transfers = [
        item for item in transfers
        if item[1] > zoom_start and item[0] < zoom_end
    ]
    visible_kernels = [
        item for item in kernels
        if item[1] > zoom_start and item[0] < zoom_end
    ]

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={
            "height_ratios": [0.5, 1.0, 1.15, 0.8, 1.0, 1.0, 1.0],
            "hspace": 0.12,
        },
    )
    ribbon, gpu_ax, transfer_ax, event_ax, vram_ax, cpu_ax, rss_ax = axes
    ribbon.set(ylim=(0, 1), yticks=[], ylabel="Stage")
    timeline_span = zoom_end - zoom_start

    # Stage shading is deliberately neutral: the semantic colors belong to
    # kernels and the three CUDA memcpy directions below.
    label_lanes = []
    for index, phase in enumerate(phases):
        start = _trace_number(phase.get("start_sec"))
        end = _trace_number(phase.get("end_sec"))
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        segments = _phase_segments(phase)
        visible_segments = [
            (max(segment_start, zoom_start), min(segment_end, zoom_end))
            for segment_start, segment_end in segments
            if min(segment_end, zoom_end) > max(segment_start, zoom_start)
        ]
        if not visible_segments:
            continue
        label_start = min(segment[0] for segment in visible_segments)
        label_end = max(segment[1] for segment in visible_segments)
        for lane, ranges in enumerate(label_lanes):
            if all(
                label_end <= start or label_start >= end
                for start, end in ranges
            ):
                ranges.append((label_start, label_end))
                break
        else:
            label_lanes.append([(label_start, label_end)])
            lane = len(label_lanes) - 1
        shade = "#e9eef2" if index % 2 == 0 else "#f5f7f9"
        for segment_start, segment_end in visible_segments:
            ribbon.add_patch(
                patches.Rectangle(
                    (segment_start, 0.08), segment_end - segment_start, 0.84,
                    facecolor=shade, edgecolor="#8a98a6", linewidth=0.7,
                )
            )
        visible_start, visible_end = max(
            visible_segments, key=lambda segment: segment[1] - segment[0]
        )
        center = (visible_start + visible_end) / 2
        phase_label = _trace_phase_label(phase)
        is_batch = str(phase.get("name", "")).startswith("filter_batch_")
        short_phase = (end - start) < 0.07 * timeline_span
        if visible_end - visible_start < 0.008 * timeline_span:
            # Tiny visible slivers cannot fit even a rotated label.
            pass
        elif is_batch:
            batch_number = str(phase["name"]).removeprefix("filter_batch_")
            ribbon.text(
                center, 0.72, f"B{batch_number}",
                ha="center", va="center", fontsize=8,
                color="#263238", clip_on=True,
            )
        elif short_phase:
            stage_label = phase_label.split("\n", 1)[-1]
            stage_label = stage_label.split("(", 1)[0].rstrip()
            ribbon.text(
                center, max(0.12, 0.24 - 0.12 * (lane % 3)), stage_label,
                ha="center", va="center", rotation=90,
                rotation_mode="anchor", fontsize=6.5,
                color="#263238", clip_on=True,
            )
        else:
            label_y = (0.73, 0.27, 0.5)[lane % 3]
            ribbon.text(
                center, label_y, phase_label,
                ha="center", va="center", fontsize=8,
                color="#263238", clip_on=True,
            )
        for ax in axes[1:]:
            for segment_start, segment_end in visible_segments:
                ax.axvspan(
                    segment_start, segment_end, color=shade, alpha=0.7,
                    zorder=0,
                )
                ax.axvline(
                    segment_start, color="#8a98a6", ls="--", lw=0.55,
                    alpha=0.7,
                )

    if visible_bins:
        starts = np.array([item[0] for item in visible_bins])
        widths = np.array([item[1] - item[0] for item in visible_bins])
        active = np.array([
            _trace_number(item[2].get("kernel_active_percent"))
            for item in visible_bins
        ])
        gpu_ax.bar(
            starts,
            active,
            width=widths,
            align="edge",
            color=CUDA_COLORS["kernels"],
            alpha=0.88,
            label="Process CUDA kernels (union active per bin)",
        )
    else:
        gpu_ax.plot(
            [], [], color=CUDA_COLORS["kernels"],
            label="Process CUDA kernels (union active per bin)",
        )
        gpu_ax.text(
            0.5, 0.45, "Nsight/CUPTI trace present; no kernel bins",
            transform=gpu_ax.transAxes,
            ha="center", va="center", fontsize=10,
        )
    gpu_ax.set(ylim=(0, 105), ylabel="Kernel active (%)")

    transfer_fields = (
        ("H2D", "h2d_bytes", "Host → device"),
        ("D2H", "d2h_bytes", "Device → host"),
        ("D2D", "d2d_bytes", "Device → device"),
    )
    if visible_bins:
        width = widths / 3.3
        for offset, (direction, field, description) in enumerate(
            transfer_fields
        ):
            mib = np.array([
                _trace_number(item[2].get(field), 0.0) / (1024 ** 2)
                for item in visible_bins
            ])
            transfer_ax.bar(
                starts + offset * width,
                mib,
                width=width,
                align="edge",
                color=CUDA_COLORS[direction],
                label=f"{direction} ({description})",
            )
    else:
        for direction, _, description in transfer_fields:
            transfer_ax.plot(
                [], [], color=CUDA_COLORS[direction],
                label=f"{direction} ({description})",
            )
        transfer_ax.text(
            0.5, 0.45, "No per-bin CUDA transfer totals",
            transform=transfer_ax.transAxes,
            ha="center", va="center", fontsize=10,
        )
    visible_copy_values = [
        _trace_number(item[2].get(field), 0.0) / (1024 ** 2)
        for item in visible_bins
        for _, field, _ in transfer_fields
    ]
    finite_copy_values = np.asarray(visible_copy_values)[
        np.isfinite(visible_copy_values)
    ]
    copy_max = max(finite_copy_values, default=0.0)
    transfer_ax.set(
        ylabel="CUDA copies (MiB/bin)",
        ylim=(0, max(1.0, copy_max * 1.18)),
    )

    lane_positions = {
        "kernels": 3.5, "H2D": 2.5, "D2H": 1.5, "D2D": 0.5
    }
    event_intervals = {lane: [] for lane in lane_positions}
    event_intervals["kernels"] = [
        (start, end - start) for start, end, _ in visible_kernels
    ]
    for start, end, item, direction in visible_transfers:
        event_intervals[direction].append((start, end - start))
    for lane, intervals in event_intervals.items():
        if not intervals:
            continue
        event_ax.broken_barh(
            intervals, (lane_positions[lane], 0.75),
            facecolors=CUDA_COLORS[lane],
        )
    event_ax.set(
        ylim=(0, 4.5),
        yticks=[3.875, 2.875, 1.875, 0.875],
        yticklabels=["Kernels", "H2D", "D2H", "D2D"],
        ylabel="CUDA events",
    )
    if not visible_kernels and not visible_transfers:
        event_ax.text(
            0.5, 0.45, "No exact CUDA events in trace",
            transform=event_ax.transAxes,
            ha="center", va="center", fontsize=10,
        )

    cpu = sample_values("proc_cpu_percent")
    cpu_smooth = np.array([
        np.nanmean(cpu[max(0, i - 4):i + 5])
        if np.any(np.isfinite(cpu[max(0, i - 4):i + 5])) else np.nan
        for i in range(len(cpu))
    ])
    vram = sample_values("gpu_proc_vram_mib")
    rss = sample_values("proc_rss_mib")
    sampled_metrics = (
        (vram_ax, vram, "#2980b9", "Process VRAM (sampled)", "VRAM (MiB)"),
        (cpu_ax, cpu, "#666666", "Process CPU (sampled)", "CPU (%)"),
        (rss_ax, rss, "#8e44ad", "Host RSS (sampled)", "Host RSS (MiB)"),
    )
    for ax, metric, color, label, ylabel in sampled_metrics:
        ax.plot(
            sample_times,
            metric,
            color=color,
            lw=1.4,
            label=label,
        )
        ax.set(ylabel=ylabel, ylim=(0, None))
    if len(cpu):
        cpu_ax.plot(
            sample_times,
            cpu_smooth,
            color="#4d4d4d",
            lw=0.8,
            alpha=0.45,
            label="CPU centred mean",
        )

    for ax in axes[1:]:
        ax.grid(axis="y", linestyle=":", alpha=0.32)
        if ax is not event_ax:
            ax.legend(loc="upper left", fontsize=7.8, framealpha=0.92, ncol=2)
        for spine in ax.spines.values():
            spine.set_color("#b0b7bf")
    for spine in ribbon.spines.values():
        spine.set_color("#b0b7bf")

    rss_ax.set(
        xlim=(zoom_start, zoom_end),
        xlabel="Elapsed time from profiler launch (seconds)",
    )
    workload = data.get("workload", {})
    campaign = (
        data.get("campaign")
        or workload.get("campaign")
        or workload.get("executable")
        or "pycbc"
    )
    wall = summary.get("total_wall_sec", data_end)
    fig.suptitle(
        f"{campaign} · whole-process CUDA timeline\n"
        f"{workload.get('processing_scheme', 'JAX CUDA')} · "
        f"{_workload_detail(workload) or 'workload unknown'} · "
        f"wall {wall:.2f} s",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    source = trace.get("source", "Nsight/CUPTI")
    target_pid = trace.get("target_pid", "unknown")
    device_id = trace.get("device_id", "unknown")
    bin_width_ms = trace.get("bin_width_ms", "unknown")
    fig.text(
        0.12,
        0.018,
        f"{source} · target PID {target_pid} · device {device_id} · "
        f"{bin_width_ms} ms bins · CPU/memory sampled (process tree) · "
        "host-log stage boundaries approximate. "
        "CUDA bars are process-scoped; transfer panels show completed bytes, "
        "not bus bandwidth.",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.075)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(
        f"[INFO] Successfully generated timeline profiling plot: {output_path}"
    )


def plot_profiling_timeline(
    data: Dict[str, Any],
    output_path: Path,
    zoom_start=None,
    zoom_end=None,
):
    """Plot observed telemetry; unavailable transfer readings remain gaps."""
    if _cpu_only_workload(data):
        return _plot_cpu_only_timeline(
            data, output_path, zoom_start=zoom_start, zoom_end=zoom_end
        )
    if "cuda_trace" in data:
        return _plot_cuda_trace_timeline(
            data, output_path, zoom_start=zoom_start, zoom_end=zoom_end
        )
    telemetry = data.get("telemetry", [])
    phases = _phases_for_view(
        data.get("phases", []), zoom_start=zoom_start, zoom_end=zoom_end
    )
    summary = data.get("summary", {})
    if not telemetry:
        raise ValueError("Telemetry data array is empty!")

    times = np.array([s["elapsed_sec"] for s in telemetry])

    def values(key):
        return np.array(
            [
                s.get(key, np.nan) if s.get(key) is not None else np.nan
                for s in telemetry
            ],
            dtype=float,
        )

    gpu = values("gpu_util_percent")
    cpu = values("proc_cpu_percent")
    cpu_smooth = np.array(
        [np.mean(cpu[max(0, i - 4):i + 5]) for i in range(len(cpu))]
    )
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(14, 13),
        sharex=True,
        gridspec_kw={
            "height_ratios": [0.42, 1, 1.15, 1, 1, 1],
            "hspace": 0.13,
        },
    )
    ribbon, gpu_ax, transfer_ax, vram_ax, cpu_ax, rss_ax = axes
    ribbon.set(ylim=(0, 1), yticks=[], ylabel="Stage")
    phase_key = []
    for i, phase in enumerate(phases):
        start, end = phase["start_sec"], phase["end_sec"]
        if end <= start:
            continue
        palette = PHASE_PALETTE[i % len(PHASE_PALETTE)]
        segments = _phase_segments(phase)
        for segment_start, segment_end in segments:
            ribbon.add_patch(
                patches.Rectangle(
                    (segment_start, 0.05),
                    segment_end - segment_start,
                    0.9,
                    facecolor=palette["bar"],
                    edgecolor=palette["edge"],
                    linewidth=1,
                )
            )
        # Narrow phases remain in the key, without overlapping ribbon text.
        if end - start >= summary.get("total_wall_sec", times[-1]) * 0.025:
            ribbon.text(
                (start + end) / 2,
                0.5,
                f"{i + 1}\n{end - start:.2f} s",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=palette["label"],
                clip_on=True,
            )
        phase_key.append(f"{i + 1}. {phase['label']}")
        for ax in axes[1:]:
            for segment_start, segment_end in segments:
                ax.axvspan(
                    segment_start, segment_end, color=palette["bg"], alpha=0.65
                )
                ax.axvline(
                    segment_start, color=palette["edge"], ls="--", lw=0.7,
                    alpha=0.6,
                )

    gpu_ax.step(
        times,
        gpu,
        where="post",
        color="#218c4c",
        lw=1.7,
        label="Device GPU activity (NVML kernel-active time)",
    )
    gpu_ax.set(
        ylim=(-3, 105), yticks=[0, 25, 50, 75, 100], ylabel="GPU activity (%)"
    )

    # Keep NVML's documented KB/s units without assuming a binary conversion.
    # PCIe RX/TX are device-wide bus activity, not exact CUDA memcpy events.
    any_transfer = False
    for key, label, color in (
        (
            "gpu_pcie_rx_kb_per_sec",
            "Into GPU (PCIe RX; host → device direction)",
            "#0072b2",
        ),
        (
            "gpu_pcie_tx_kb_per_sec",
            "Out of GPU (PCIe TX; device → host direction)",
            "#d55e00",
        ),
    ):
        rates = values(key)
        if np.any(np.isfinite(rates)):
            any_transfer = True
            transfer_ax.plot(times, rates, label=label, color=color, lw=1.4)
        else:
            transfer_ax.plot(
                [], [], label=label + " — unavailable", color=color
            )
    transfer_ax.set(ylabel="PCIe traffic (KB/s)", ylim=(0, None))
    if not any_transfer:
        transfer_ax.text(
            0.5,
            0.4,
            "Transfers not recorded / unavailable\n"
            "Re-run the profiler to collect PCIe RX/TX",
            transform=transfer_ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
    transfer_ax.text(
        0.99,
        0.91,
        "Whole device • 20 ms NVML window",
        transform=transfer_ax.transAxes,
        ha="right",
        fontsize=8.5,
    )
    transfer_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 4))

    for ax, key, color, label, ylabel in (
        (
            vram_ax,
            "gpu_proc_vram_mib",
            "#2980b9",
            "Process VRAM",
            "VRAM (MiB)",
        ),
        (
            rss_ax,
            "proc_rss_mib",
            "#8e44ad",
            "Process resident host RAM",
            "Host RSS (MiB)",
        ),
    ):
        metric = values(key)
        ax.plot(times, metric, color=color, lw=1.6, label=label)
        ax.fill_between(times, 0, metric, color=color, alpha=0.12)
        ax.set(ylabel=ylabel, ylim=(0, None))
        finite = metric[np.isfinite(metric)]
        if len(finite):
            ax.text(
                0.99,
                0.9,
                f"Peak: {max(finite):,.0f} MiB",
                transform=ax.transAxes,
                ha="right",
                fontsize=9,
            )

    cpu_ax.plot(
        times,
        cpu,
        color="#e74c3c",
        alpha=0.2,
        lw=0.7,
        label="Process CPU (raw)",
    )
    cpu_ax.plot(
        times,
        cpu_smooth,
        color="#c0392b",
        lw=1.6,
        label="Process CPU (9-sample centred mean)",
    )
    cpu_ax.axhline(100, color="#777777", ls=":", lw=0.8)
    cpu_ax.set(ylabel="Process tree CPU (%)", ylim=(0, None))
    cpu_ax.text(
        0.99,
        0.9,
        "100% = one logical CPU",
        transform=cpu_ax.transAxes,
        ha="right",
        fontsize=9,
    )

    for ax in axes[1:]:
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.93)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color("#b0b7bf")
    wall = summary.get("total_wall_sec", times[-1])
    rss_ax.set(
        xlim=(
            0 if zoom_start is None else zoom_start,
            max(wall, times[-1]) if zoom_end is None else zoom_end,
        ),
        xlabel="Elapsed time from profiler launch (seconds)",
    )
    workload = data.get("workload", {})
    campaign = (
        data.get("campaign")
        or workload.get("campaign")
        or workload.get("executable")
        or "pycbc"
    )
    batch_size = workload.get("batch_size", "unknown")
    processing_scheme = workload.get(
        "processing_scheme", "unknown processing scheme"
    )
    fig.suptitle(
        f"{campaign} · CPU, GPU and device transfers over time\n"
        f"{processing_scheme} · "
        f"{_workload_detail(workload) or f'batch {batch_size}'} · "
        f"profiled wall {wall:.2f} s",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    # A numbered key keeps short output/teardown phases legible.
    key_lines = []
    for item in phase_key:
        if not key_lines or len(key_lines[-1]) + len(item) + 5 > 125:
            key_lines.append(item)
        else:
            key_lines[-1] += "   |   " + item
    phase_legend = fig.text(
        0.12,
        0.925,
        "\n".join(key_lines),
        fontsize=9,
        va="top",
        linespacing=1.5,
    )
    actual_interval = (
        f"{np.median(np.diff(times)) * 1000:.1f} ms"
        if len(times) > 1
        else "unknown"
    )
    fig.text(
        0.12,
        0.018,
        "Stage boundaries are host log timestamps. GPU work is asynchronous; "
        "device metrics include other processes.\n"
        f"Median polling interval: {actual_interval}. "
        "GPU activity has a separate "
        "hardware averaging window. PCIe excludes on-device copies.",
        fontsize=9,
    )
    # Longer searches have more batch labels; reserve their rendered height.
    fig.canvas.draw()
    legend_bottom = phase_legend.get_window_extent(
        fig.canvas.get_renderer()
    ).transformed(fig.transFigure.inverted()).y0
    fig.subplots_adjust(
        left=0.12, right=0.98,
        top=min(0.86, legend_bottom - 0.015), bottom=0.07,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(
        f"[INFO] Successfully generated timeline profiling plot: {output_path}"
    )


def stage_zoom_windows(
    data: Dict[str, Any], padding: float = 0.02,
    max_windows: int = 12, max_window_sec: float = 20.0,
):
    """Return bounded, padded representative windows for observed stages.

    Repeated filter batches are one family and only their longest complete
    occurrence is selected.  Broad search/filter spans are omitted when they
    exceed ``max_window_sec``; the whole-process overview retains them.
    """
    if padding < 0:
        raise ValueError("padding must be non-negative")
    if max_windows <= 0 or max_window_sec <= 0:
        raise ValueError("max_windows and max_window_sec must be positive")
    candidates = {}
    for index, phase in enumerate(data.get("phases") or [], 1):
        try:
            start = float(phase["start_sec"])
            end = float(phase["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        duration = end - start
        family = _stage_family(phase)
        current = candidates.get(family)
        if current is None or (duration, -start) > (current[0], -current[1]):
            candidates[family] = (duration, start, end, phase, index)

    selected = []
    for duration, start, end, phase, index in candidates.values():
        family = _stage_family(phase)
        if duration > max_window_sec and family in {"search", "filter"}:
            continue
        if duration > max_window_sec:
            center = (start + end) / 2
            start = center - max_window_sec / 2
            end = center + max_window_sec / 2
            duration = max_window_sec
        selected.append((index, duration, start, end, phase))
    selected.sort(key=lambda item: (-item[1], item[0]))
    selected = sorted(selected[:max_windows], key=lambda item: item[0])
    windows = []
    for index, duration, start, end, phase in selected:
        windows.append({
            "index": index,
            "name": str(phase.get("name", f"stage_{index}")),
            "label": str(phase.get("label", phase.get("name", index))),
            "start_sec": max(0.0, start - padding * duration),
            "end_sec": end + padding * duration,
        })
    return windows


def plot_campaign_figures(
    data: Dict[str, Any],
    output_path: Path,
    stage_zoom_dir: Path = None,
    zoom_start=None,
    zoom_end=None,
):
    """Write the full-run overview and up to 12 representative stage zooms."""
    plot_profiling_timeline(
        data, output_path, zoom_start=zoom_start, zoom_end=zoom_end
    )
    outputs = [Path(output_path)]
    if stage_zoom_dir is None:
        return outputs
    stage_zoom_dir = Path(stage_zoom_dir)
    stage_zoom_dir.mkdir(parents=True, exist_ok=True)
    for window in stage_zoom_windows(data):
        safe_name = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in window["name"]
        ).strip("_") or f"stage_{window['index']}"
        stage_output = stage_zoom_dir / (
            f"{window['index']:02d}_{safe_name}.png"
        )
        plot_profiling_timeline(
            data,
            stage_output,
            zoom_start=window["start_sec"],
            zoom_end=window["end_sec"],
        )
        outputs.append(stage_output)
    return outputs


def _cpu_only_workload(data):
    """Return whether the receipt explicitly describes a zero-GPU run."""
    resources = (data.get("workload") or {}).get("resources") or {}
    return resources.get("gpus") == 0


def _plot_cpu_only_timeline(
    data: Dict[str, Any],
    output_path: Path,
    zoom_start=None,
    zoom_end=None,
):
    """Render all stage intervals with process CPU and RSS for CPU runs."""
    telemetry = data.get("telemetry") or []
    if not telemetry:
        raise ValueError("Telemetry data array is empty!")
    workload = data.get("workload") or {}
    summary = data.get("summary") or {}
    times = np.array([
        _trace_number(sample.get("elapsed_sec")) for sample in telemetry
    ])
    finite_times = times[np.isfinite(times)]
    if not len(finite_times):
        raise ValueError("Telemetry elapsed times are unavailable")
    wall = _trace_number(summary.get("total_wall_sec"), np.nan)
    if not np.isfinite(wall) or wall <= 0:
        wall = float(finite_times[-1])
    if wall <= 0:
        wall = 1.0
    if zoom_start is None:
        zoom_start = 0.0
    if zoom_end is None:
        zoom_end = wall
    if zoom_start >= zoom_end:
        raise ValueError("zoom-start must be less than zoom-end")

    def values(key):
        return np.array([
            _trace_number(sample.get(key)) for sample in telemetry
        ])

    cpu = values("proc_cpu_percent")
    rss = values("proc_rss_mib")
    families = {}
    for phase in data.get("phases") or []:
        try:
            start = float(phase["start_sec"])
            end = float(phase["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        name = str(phase.get("name", phase.get("label", "stage")))
        rank = phase.get("rank")
        family = (name, str(rank) if rank is not None else "")
        families.setdefault(family, []).append((start, end - start))
    family_items = sorted(families.items(), key=lambda item: item[0])
    labels = [
        name if not rank else f"{name} · rank {rank}"
        for (name, rank), _ in family_items
    ]
    lane_height = 0.75
    lane_gap = 0.25
    stage_height = max(1.8, len(family_items) * (lane_height + lane_gap))
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, max(8.0, stage_height + 5.0)),
        sharex=True,
        gridspec_kw={
            "height_ratios": [max(1.8, stage_height / 2.0), 1.0, 1.0],
            "hspace": 0.16,
        },
    )
    stage_ax, cpu_ax, rss_ax = axes
    stage_ax.set_ylim(-0.15, max(0.85, len(family_items) - 0.15))
    stage_ax.set_yticks(np.arange(len(labels)) + 0.125)
    stage_ax.set_yticklabels(labels)
    stage_ax.set_ylabel("Stages")
    for row, ((name, _rank), intervals) in enumerate(family_items):
        color = PHASE_PALETTE[row % len(PHASE_PALETTE)]["bar"]
        stage_ax.broken_barh(
            intervals,
            (row - 0.25, lane_height),
            facecolors=color,
            edgecolors=PHASE_PALETTE[row % len(PHASE_PALETTE)]["edge"],
            linewidth=0.5,
        )
    stage_ax.grid(axis="x", linestyle=":", alpha=0.35)
    stage_ax.text(
        0.01,
        1.01,
        "Every observed interval is shown; lanes are stage name and rank",
        transform=stage_ax.transAxes,
        va="bottom",
        fontsize=9,
    )

    cpu_ax.plot(times, cpu, color="#c0392b", lw=1.2, label="Process tree CPU")
    cpu_ax.axhline(100, color="#777777", ls=":", lw=0.8)
    cpu_ax.set_ylabel("CPU (%)")
    cpu_ax.set_ylim(bottom=0)
    cpu_ax.grid(axis="y", linestyle=":", alpha=0.35)
    cpu_ax.legend(loc="upper left", fontsize=8.5, framealpha=0.93)
    cpu_ax.text(
        0.99,
        0.9,
        "100% = one logical CPU",
        transform=cpu_ax.transAxes,
        ha="right",
        fontsize=9,
    )

    rss_ax.plot(times, rss, color="#8e44ad", lw=1.3, label="Process RSS")
    rss_ax.fill_between(times, 0, rss, color="#8e44ad", alpha=0.12)
    rss_ax.set_ylabel("RSS (MiB)")
    rss_ax.set_ylim(bottom=0)
    rss_ax.grid(axis="y", linestyle=":", alpha=0.35)
    rss_ax.legend(loc="upper left", fontsize=8.5, framealpha=0.93)
    finite_rss = rss[np.isfinite(rss)]
    if len(finite_rss):
        rss_ax.text(
            0.99,
            0.9,
            f"Peak: {max(finite_rss):,.0f} MiB",
            transform=rss_ax.transAxes,
            ha="right",
            fontsize=9,
        )

    for ax in axes:
        ax.set_xlim(zoom_start, zoom_end)
        for spine in ax.spines.values():
            spine.set_color("#b0b7bf")
    rss_ax.set_xlabel("Elapsed time from profiler launch (seconds)")
    campaign = (
        data.get("campaign")
        or workload.get("campaign")
        or workload.get("executable")
        or "pycbc"
    )
    fig.suptitle(
        f"{campaign} · CPU process utilization and stages\n"
        f"{workload.get('processing_scheme', 'CPU')} · "
        f"{_workload_detail(workload) or 'workload unknown'} · "
        f"profiled wall {wall:.2f} s",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.12,
        0.015,
        "CPU-only workload (resources.gpus=0). No GPU telemetry is plotted. "
        "Stage boundaries are host log timestamps.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.19, right=0.98, top=0.91, bottom=0.07)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Successfully generated timeline profiling plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GPU/CPU timeline profiling plot"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/jax_gpu_timeline.json"),
        help="Path to JSON telemetry receipt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/jax_gpu_timeline.png"),
        help="Path to output PNG plot",
    )
    parser.add_argument(
        "--zoom-start",
        type=float,
        default=None,
        help="Optional left edge of the plotted time window (seconds)",
    )
    parser.add_argument(
        "--zoom-end",
        type=float,
        default=None,
        help="Optional right edge of the plotted time window (seconds)",
    )
    parser.add_argument(
        "--stage-zoom-dir",
        type=Path,
        default=None,
        help="Also write one consistently padded plot per observed stage",
    )
    args = parser.parse_args()

    data = _load_receipt(args.input)

    plot_campaign_figures(
        data,
        args.output,
        stage_zoom_dir=args.stage_zoom_dir,
        zoom_start=args.zoom_start,
        zoom_end=args.zoom_end,
    )


if __name__ == "__main__":
    main()
