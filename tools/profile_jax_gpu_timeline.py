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

"""GPU, CPU, memory and transfer timeline profiler for pycbc_inspiral.

Executes pycbc_inspiral with JAX CUDA while capturing synchronized high-rate
telemetry (GPU activity, PCIe RX/TX, GPU VRAM allocation, process/system CPU
utilization, and host memory RSS) alongside pipeline phase transitions.
Polling frequency is a request: NVML queries may take longer than the interval.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil


class NVMLTelemetryCollector:
    """Lightweight NVML wrapper for querying GPU compute and memory metrics."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.available = False
        try:
            self.nvml = ctypes.CDLL("libnvidia-ml.so.1")
            ret = self.nvml.nvmlInit()
            if ret == 0:
                self.handle = ctypes.c_void_p()
                ret = self.nvml.nvmlDeviceGetHandleByIndex(
                    device_index, ctypes.byref(self.handle)
                )
                if ret == 0:
                    self.available = True
        except Exception:
            self.available = False

        if self.available:

            class nvmlUtilization_t(ctypes.Structure):
                _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

            class nvmlMemory_t(ctypes.Structure):
                _fields_ = [
                    ("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong),
                ]

            class nvmlProcessInfo_t(ctypes.Structure):
                _fields_ = [
                    ("pid", ctypes.c_uint),
                    ("usedGpuMemory", ctypes.c_ulonglong),
                ]

            self.nvmlUtilization_t = nvmlUtilization_t
            self.nvmlMemory_t = nvmlMemory_t
            self.nvmlProcessInfo_t = nvmlProcessInfo_t

    def sample(self, target_pid: Optional[int] = None) -> Dict[str, Any]:
        """Query GPU activity windows, memory allocations and PCIe rates."""
        if not self.available:
            return {
                "gpu_util_percent": 0.0,
                "gpu_mem_util_percent": 0.0,
                "gpu_device_used_bytes": 0,
                "gpu_device_total_bytes": 0,
                "gpu_proc_used_bytes": 0,
                "gpu_pcie_rx_kb_per_sec": None,
                "gpu_pcie_tx_kb_per_sec": None,
            }

        util = self.nvmlUtilization_t()
        mem = self.nvmlMemory_t()
        self.nvml.nvmlDeviceGetUtilizationRates(
            self.handle, ctypes.byref(util)
        )
        self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem))

        proc_vram = 0
        if target_pid:
            procs = (self.nvmlProcessInfo_t * 64)()
            count = ctypes.c_uint(64)
            ret = self.nvml.nvmlDeviceGetComputeRunningProcesses(
                self.handle, ctypes.byref(count), procs
            )
            if ret == 0:
                for i in range(count.value):
                    if procs[i].pid == target_pid:
                        proc_vram = procs[i].usedGpuMemory
                        break

        return {
            "gpu_util_percent": float(util.gpu),
            "gpu_mem_util_percent": float(util.memory),
            "gpu_device_used_bytes": int(mem.used),
            "gpu_device_total_bytes": int(mem.total),
            "gpu_proc_used_bytes": int(proc_vram),
            "gpu_pcie_rx_kb_per_sec": self._pcie_throughput(1),
            "gpu_pcie_tx_kb_per_sec": self._pcie_throughput(0),
        }

    def _pcie_throughput(self, counter: int) -> Optional[int]:
        """Device-wide PCIe rate in NVML KB/s, over its own 20 ms window.

        RX (1) is traffic into the GPU; TX (0) is traffic out. These are
        bus counters, not per-process CUDA memcpy events. Missing support or
        a failed read is unknown, never evidence of zero transfers.
        """
        try:
            value = ctypes.c_uint()
            ret = self.nvml.nvmlDeviceGetPcieThroughput(
                self.handle, counter, ctypes.byref(value)
            )
        except AttributeError:
            return None
        return int(value.value) if ret == 0 else None

    def close(self):
        if self.available:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass


def run_profiling_campaign(
    executable: Path,
    frame_file: Path,
    bank_file: Path,
    output_hdf: Path,
    output_json: Path,
    python_bin: str = sys.executable,
    batch_size: int = 128,
    sampling_interval_ms: int = 25,
    affinity_core: str = "8",
    approximant: str = "IMRPhenomD",
    nvtx_sync: bool = False,
) -> Dict[str, Any]:
    """Run pycbc_inspiral while collecting hardware and phase telemetry."""
    if sampling_interval_ms <= 0:
        raise ValueError("sampling_interval_ms must be positive")
    executable = executable.resolve()
    frame_file = frame_file.resolve()
    bank_file = bank_file.resolve()
    output_hdf = output_hdf.resolve()
    output_json = output_json.resolve()
    output_hdf.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cli_args = [
        "--verbose",
        "--frame-files",
        str(frame_file),
        "--channel-name",
        "H1:LOSC-STRAIN",
        "--gps-start-time",
        "1187007048",
        "--gps-end-time",
        "1187009080",
        "--trig-start-time",
        "1187007160",
        "--trig-end-time",
        "1187009064",
        "--sample-rate",
        "4096",
        "--low-frequency-cutoff",
        "30",
        "--strain-high-pass",
        "25",
        "--pad-data",
        "8",
        "--autogating-threshold",
        "100",
        "--autogating-cluster",
        "5",
        "--autogating-width",
        "0.25",
        "--autogating-taper",
        "0.25",
        "--autogating-pad",
        "16",
        "--autogating-max-iterations",
        "1",
        "--psd-estimation",
        "median",
        "--psd-segment-length",
        "32",
        "--psd-segment-stride",
        "16",
        "--psd-num-segments",
        "126",
        "--psd-inverse-length",
        "16",
        "--invpsd-trunc-method",
        "hann",
        "--invpsd-trunc-which-spectrum",
        "invasd",
        "--approximant",
        approximant,
        "--order",
        "-1",
        "--snr-threshold",
        "5.5",
        "--newsnr-threshold",
        "5",
        "--chisq-bins",
        "16",
        "--cluster-window",
        "1",
        "--cluster-function",
        "symmetric",
        "--fft-backends",
        "jax",
        "--bank-file",
        str(bank_file),
        "--segment-length",
        "512",
        "--segment-start-pad",
        "112",
        "--segment-end-pad",
        "16",
        "--processing-scheme",
        "jax:cuda:0",
        "--batch-size",
        str(batch_size),
        "--use-compressed-waveforms",
        "--waveform-decompression-method",
        "inline_linear",
        "--output",
        str(output_hdf),
    ]

    command = [python_bin, str(executable), *cli_args]
    if sys.platform == "linux" and shutil.which("taskset") and affinity_core:
        command = ["taskset", "-c", affinity_core] + command

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MKL_DYNAMIC": "FALSE",
            "MKL_THREADING_LAYER": "GNU",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        }
    )

    nvml = NVMLTelemetryCollector(device_index=0)

    telemetry: List[Dict[str, Any]] = []
    stop_event = threading.Event()
    process_holder: List[Optional[subprocess.Popen]] = [None]
    stderr_lines: List[Tuple[float, str]] = []

    trace_sync = None
    if nvtx_sync:
        library = ctypes.util.find_library("nvToolsExt")
        if not library:
            raise RuntimeError("--nvtx-sync requires libnvToolsExt")
        nvtx = ctypes.CDLL(library)
        nvtx.nvtxMarkA.argtypes = [ctypes.c_char_p]
        nvtx.nvtxMarkA.restype = None
        nvtx.nvtxMarkA(b"pycbc.timeline.init")
        before = time.perf_counter()
        nvtx.nvtxMarkA(b"pycbc.timeline.origin")
        after = time.perf_counter()
        t_start = (before + after) / 2
        trace_sync = {
            "marker": "pycbc.timeline.origin",
            "collector_pid": os.getpid(),
            "clock_uncertainty_sec": (after - before) / 2,
        }
    else:
        t_start = time.perf_counter()

    def telemetry_worker():
        interval = sampling_interval_ms / 1000.0
        proc_obj: Optional[psutil.Process] = None
        target_pid = None

        while not stop_event.is_set():
            now = time.perf_counter()
            elapsed = now - t_start

            if process_holder[0] is not None and target_pid is None:
                target_pid = process_holder[0].pid
                try:
                    proc_obj = psutil.Process(target_pid)
                    proc_obj.cpu_percent(None)
                except Exception:
                    proc_obj = None

            proc_cpu = 0.0
            proc_rss = 0
            if proc_obj is not None:
                try:
                    if proc_obj.is_running():
                        proc_cpu = proc_obj.cpu_percent(None)
                        proc_rss = proc_obj.memory_info().rss
                except Exception:
                    pass

            gpu_metrics = nvml.sample(target_pid)
            sys_cpu = psutil.cpu_percent(None)
            sys_mem = psutil.virtual_memory()

            sample = {
                "elapsed_sec": round(elapsed, 4),
                "gpu_util_percent": gpu_metrics["gpu_util_percent"],
                "gpu_mem_util_percent": gpu_metrics["gpu_mem_util_percent"],
                "gpu_pcie_rx_kb_per_sec": gpu_metrics[
                    "gpu_pcie_rx_kb_per_sec"
                ],
                "gpu_pcie_tx_kb_per_sec": gpu_metrics[
                    "gpu_pcie_tx_kb_per_sec"
                ],
                "gpu_proc_vram_mib": round(
                    gpu_metrics["gpu_proc_used_bytes"] / (1024**2), 2
                ),
                "gpu_device_used_mib": round(
                    gpu_metrics["gpu_device_used_bytes"] / (1024**2), 2
                ),
                "gpu_device_total_mib": round(
                    gpu_metrics["gpu_device_total_bytes"] / (1024**2), 2
                ),
                "proc_cpu_percent": round(proc_cpu, 2),
                "sys_cpu_percent": round(sys_cpu, 2),
                "proc_rss_mib": round(proc_rss / (1024**2), 2),
                "sys_mem_used_gib": round(sys_mem.used / (1024**3), 2),
            }
            telemetry.append(sample)

            sleep_remaining = interval - (time.perf_counter() - now)
            if sleep_remaining > 0:
                time.sleep(sleep_remaining)

    worker_thread = threading.Thread(target=telemetry_worker, daemon=True)
    worker_thread.start()

    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(output_hdf.parent),
    )
    process_holder[0] = proc

    # Read stderr in real time to stamp log line events
    if proc.stderr is not None:
        for line in proc.stderr:
            t_line = time.perf_counter() - t_start
            clean_line = line.rstrip()
            stderr_lines.append((round(t_line, 4), clean_line))
            print(f"[{t_line:6.2f}s] {clean_line}", flush=True)

    proc.wait()
    t_end = time.perf_counter()
    stop_event.set()
    worker_thread.join(timeout=2.0)
    nvml.close()

    total_wall_sec = t_end - t_start

    # Parse phase boundaries from timestamped stderr lines
    phases = _identify_pipeline_phases(stderr_lines, total_wall_sec)

    # Calculate summary metrics
    gpu_utils = [s["gpu_util_percent"] for s in telemetry]
    proc_vrams = [s["gpu_proc_vram_mib"] for s in telemetry]
    proc_cpus = [s["proc_cpu_percent"] for s in telemetry]
    proc_rsss = [s["proc_rss_mib"] for s in telemetry]
    intervals_ms = [
        (right["elapsed_sec"] - left["elapsed_sec"]) * 1000
        for left, right in zip(telemetry, telemetry[1:])
    ]

    summary = {
        "total_wall_sec": round(total_wall_sec, 4),
        "sampling_interval_ms": sampling_interval_ms,
        "observed_median_interval_ms": (
            round(statistics.median(intervals_ms), 3) if intervals_ms else None
        ),
        "sample_count": len(telemetry),
        "peak_gpu_util_percent": max(gpu_utils) if gpu_utils else 0.0,
        "mean_gpu_util_percent": (
            round(float(sum(gpu_utils) / len(gpu_utils)), 2)
            if gpu_utils
            else 0.0
        ),
        "peak_proc_vram_mib": max(proc_vrams) if proc_vrams else 0.0,
        "peak_proc_cpu_percent": max(proc_cpus) if proc_cpus else 0.0,
        "mean_proc_cpu_percent": (
            round(float(sum(proc_cpus) / len(proc_cpus)), 2)
            if proc_cpus
            else 0.0
        ),
        "peak_proc_rss_mib": max(proc_rsss) if proc_rsss else 0.0,
    }

    result = {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "command": command,
        "returncode": proc.returncode,
        "target_pid": proc.pid,
        "trace_sync": trace_sync,
        "telemetry_metadata": {
            "gpu_util_scope": (
                "whole device; kernel-active time, not occupancy"
            ),
            "pcie_scope": (
                "whole device; includes other processes and PCIe traffic"
            ),
            "pcie_source": "nvmlDeviceGetPcieThroughput",
            "pcie_units": "KB/s as reported by NVML",
            "pcie_window_ms": 20,
            "sample_timestamp": "start of sequential metric queries",
            "pcie_rx_direction": "into GPU (host-to-device direction)",
            "pcie_tx_direction": "out of GPU (device-to-host direction)",
            "missing_values": "null means unavailable, not zero",
            "cpu_percent": "100% is one logical CPU; process only",
            "affinity_core": affinity_core,
            "xla_preallocate": False,
        },
        "workload": {
            "track": "track1_compressed",
            "batch_size": batch_size,
            "sample_rate_hz": 4096,
            "segment_length_sec": 512,
            "frame_file": str(frame_file),
            "bank_file": str(bank_file),
            "approximant": approximant,
            "processing_scheme": "jax:cuda:0",
        },
        "summary": summary,
        "phases": phases,
        "telemetry": telemetry,
        "stderr_log": stderr_lines,
    }

    with output_json.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[INFO] Saved profiling telemetry receipt to: {output_json}")
    print(
        f"[INFO] Total Wall: {total_wall_sec:.2f}s | "
        f"Samples: {len(telemetry)} | "
        f"Peak VRAM: {summary['peak_proc_vram_mib']} MiB | "
        f"Peak RSS: {summary['peak_proc_rss_mib']} MiB"
    )
    return result


def _identify_pipeline_phases(
    stderr_lines: List[Tuple[float, str]], total_wall: float
) -> List[Dict[str, Any]]:
    """Derive host-side phase intervals from observed log milestones only.

    These boundaries timestamp log receipt, not GPU completion. Missing
    milestones are omitted; no timing or utilization values are inferred.
    """
    import re

    events = []
    seen = set()
    decompressions = []
    filtering = []
    warmup_time = None
    milestones = (
        ("Reading Frames", "frame_io", "Frame Reading"),
        (
            "Highpass Filtering",
            "strain_conditioning",
            "Strain Conditioning & PSD",
        ),
        (
            "Resampling data",
            "strain_conditioning",
            "Strain Conditioning & PSD",
        ),
        (
            "Making frequency-domain data segments",
            "strain_conditioning",
            "Strain Conditioning & PSD",
        ),
        ("Read in template bank", "bank_load", "Bank Load"),
        ("Warming up JAX JIT compilation", "jit_warmup", "JIT Warmup"),
        ("We currently have", "clustering", "Trigger Finalization"),
        ("Removing triggers", "clustering", "Trigger Finalization"),
        ("Outputting", "clustering", "Trigger Finalization"),
        ("Writing out triggers", "writing_triggers", "HDF5 Output"),
        ("Finished", "teardown", "Process Teardown"),
    )

    for timestamp, line in sorted(stderr_lines, key=lambda item: item[0]):
        if not 0 <= timestamp <= total_wall:
            continue
        for marker, name, label in milestones:
            if marker in line and name not in seen:
                events.append((timestamp, name, label))
                seen.add(name)
                if name == "jit_warmup":
                    warmup_time = timestamp
                break
        match = re.search(r"Decompressing template batch (\d+)-(\d+)", line)
        if match:
            decompressions.append((timestamp, *map(int, match.groups())))
        match = re.search(r"Filtering template batch (\d+)-(\d+)", line)
        if match:
            filtering.append((timestamp, *map(int, match.groups())))

    warmup_decompression = next(
        (
            item
            for item in decompressions
            if warmup_time is not None and item[0] >= warmup_time
        ),
        None,
    )
    batch_events = []
    if filtering:
        # The warmup calls the filtering kernel directly without this log.
        # Pair each actual search batch with its most recent decompression.
        seen_batches = set()
        for timestamp, first, last in filtering:
            if (first, last) in seen_batches:
                continue
            seen_batches.add((first, last))
            preceding = [
                t
                for t, lo, hi in decompressions
                if lo == first
                and hi == last
                and t <= timestamp
                and (t, lo, hi) != warmup_decompression
            ]
            start = preceding[-1] if preceding else timestamp
            batch_events.append((start, first, last))
    else:
        # Older logs may omit Filtering lines. After a warmup, only a second
        # occurrence of its first batch provides evidence of the search start.
        candidates = [
            item
            for item in decompressions
            if warmup_time is None or item[0] >= warmup_time
        ]
        if warmup_time is not None and candidates:
            first_range = candidates[0][1:]
            restart = next(
                (
                    i
                    for i, item in enumerate(candidates[1:], 1)
                    if item[1:] == first_range
                ),
                None,
            )
            candidates = candidates[restart:] if restart is not None else []
        batch_events = candidates

    for index, (timestamp, first, last) in enumerate(batch_events, 1):
        events.append(
            (
                timestamp,
                f"filter_batch_{index}",
                f"Filter Batch {index} ({first}–{last})",
            )
        )

    events.sort(key=lambda item: item[0])
    if not events:
        events = [(0.0, "execution", "Process Execution")]
    elif events[0][0] > 0:
        events.insert(0, (0.0, "startup_imports", "Startup & Imports"))

    phases = []
    for index, (start, name, label) in enumerate(events):
        end = events[index + 1][0] if index + 1 < len(events) else total_wall
        if end <= start:
            continue
        phases.append(
            {
                "name": name,
                "label": label,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": round(end - start, 4),
                "boundary_source": (
                    "process_launch" if start == 0 else "stderr_log"
                ),
            }
        )
    return phases


def main():
    parser = argparse.ArgumentParser(
        description="GPU/CPU and PCIe timeline profiler for pycbc_inspiral"
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("bin/pycbc_inspiral"),
        help="Path to pycbc_inspiral executable",
    )
    parser.add_argument(
        "--frame-file",
        type=Path,
        required=True,
        help="Path to GWF strain frame file",
    )
    parser.add_argument(
        "--bank-file",
        type=Path,
        required=True,
        help="Path to compressed template bank HDF5 file",
    )
    parser.add_argument(
        "--output-hdf",
        type=Path,
        default=Path("/tmp/profile_triggers.hdf"),
        help="Path to output triggers HDF5",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/benchmarks-20260917/jax_gpu_timeline.json"),
        help="Path to output JSON telemetry receipt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for pycbc_inspiral (default: 128)",
    )
    parser.add_argument(
        "--sampling-interval-ms",
        type=int,
        default=25,
        help="Requested polling interval in ms; queries may take longer (25)",
    )
    parser.add_argument(
        "--affinity-core",
        type=str,
        default="8",
        help="CPU affinity pin (taskset -c affinity)",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable to invoke",
    )

    parser.add_argument(
        "--nvtx-sync", action="store_true",
        help="Emit an NVTX clock marker when running the collector under nsys",
    )
    args = parser.parse_args()

    result = run_profiling_campaign(
        executable=args.executable,
        frame_file=args.frame_file,
        bank_file=args.bank_file,
        output_hdf=args.output_hdf,
        output_json=args.output_json,
        python_bin=args.python_bin,
        batch_size=args.batch_size,
        sampling_interval_ms=args.sampling_interval_ms,
        affinity_core=args.affinity_core,
        nvtx_sync=args.nvtx_sync,
    )
    if result["returncode"]:
        raise SystemExit(result["returncode"])


if __name__ == "__main__":
    main()
