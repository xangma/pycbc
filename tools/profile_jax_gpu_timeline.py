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

"""GPU, CPU, memory and transfer timeline profiler for PyCBC campaigns.

Executes a PyCBC command while capturing high-rate
telemetry (GPU activity, PCIe RX/TX, GPU VRAM allocation, process/system CPU
utilization, and host memory RSS) alongside pipeline phase transitions.  The
process metrics cover the complete observed root process tree.
Polling frequency is a request: NVML queries may take longer than the interval.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
from datetime import datetime, timezone
import hashlib
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import psutil


def _stable_hash(value: Any) -> str:
    """Hash JSON-compatible profile configuration without exposing values."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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

    def sample(
        self,
        target_pid: Optional[int] = None,
        target_pids: Optional[Iterable[int]] = None,
    ) -> Dict[str, Any]:
        """Query GPU activity windows, memory allocations and PCIe rates."""
        if not self.available:
            return {
                "gpu_util_percent": None,
                "gpu_mem_util_percent": None,
                "gpu_device_used_bytes": None,
                "gpu_device_total_bytes": None,
                "gpu_proc_used_bytes": None,
                "gpu_pcie_rx_kb_per_sec": None,
                "gpu_pcie_tx_kb_per_sec": None,
            }

        util = self.nvmlUtilization_t()
        mem = self.nvmlMemory_t()
        gpu_util = gpu_mem_util = None
        try:
            ret = self.nvml.nvmlDeviceGetUtilizationRates(
                self.handle, ctypes.byref(util)
            )
            if ret == 0:
                gpu_util = float(util.gpu)
                gpu_mem_util = float(util.memory)
        except Exception:
            pass
        device_used = device_total = None
        try:
            ret = self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem))
            if ret == 0:
                device_used = int(mem.used)
                device_total = int(mem.total)
        except Exception:
            pass

        pids = set(int(pid) for pid in (target_pids or ()) if pid)
        if target_pid:
            pids.add(int(target_pid))
        proc_vram = None
        if pids:
            # The process-query ABI has changed across NVML versions.  A
            # missing versioned struct, allocation error, exception, or
            # nonzero return leaves process-scoped memory unknown; zero would
            # incorrectly claim that the observed process used no VRAM.
            try:
                procs = (self.nvmlProcessInfo_t * 64)()
                count = ctypes.c_uint(64)
                ret = self.nvml.nvmlDeviceGetComputeRunningProcesses(
                    self.handle, ctypes.byref(count), procs
                )
                if ret == 0:
                    proc_vram = 0
                    for i in range(min(int(count.value), len(procs))):
                        if procs[i].pid in pids:
                            proc_vram += procs[i].usedGpuMemory
            except Exception:
                proc_vram = None

        return {
            "gpu_util_percent": gpu_util,
            "gpu_mem_util_percent": gpu_mem_util,
            "gpu_device_used_bytes": device_used,
            "gpu_device_total_bytes": device_total,
            "gpu_proc_used_bytes": proc_vram,
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
        except Exception:
            return None
        return int(value.value) if ret == 0 else None

    def close(self):
        if self.available:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass


def _process_tree(
    root_pid: Optional[int], known_pids: Optional[set[int]] = None,
    active_pids: Optional[set[int]] = None,
):
    """Discover descendants, keeping historical and active sets separate.

    ``known_pids`` is an inventory for the receipt and is never used as a
    polling set.  ``active_pids`` also retains children that were observed
    before they were reparented, which is important for short-lived orphaned
    workers, until the next liveness check removes them.
    """
    known = known_pids if known_pids is not None else set()
    active = active_pids if active_pids is not None else known
    if root_pid is None:
        return active
    try:
        root = psutil.Process(int(root_pid))
        current = [root, *root.children(recursive=True)]
    except (psutil.Error, ValueError, TypeError):
        current = []
    for process in current:
        try:
            known.add(process.pid)
            active.add(process.pid)
        except psutil.Error:
            pass
    return active


def _sample_process_tree(
    root_pid: Optional[int], known_pids: set[int],
    process_cache: Optional[Dict[int, psutil.Process]] = None,
    active_pids: Optional[set[int]] = None,
    process_identity: Optional[Dict[int, float]] = None,
):
    """Aggregate CPU and RSS over live members, never historical PIDs."""
    active = active_pids if active_pids is not None else set(known_pids)
    _process_tree(root_pid, known_pids, active)
    process_cache = process_cache if process_cache is not None else {}
    process_identity = process_identity if process_identity is not None else {}
    cpu = 0.0
    rss = 0
    live_pids = []
    for pid in sorted(active):
        try:
            process = process_cache.get(pid)
            if process is None:
                process = psutil.Process(pid)
                process_cache[pid] = process
                process_identity[pid] = float(process.create_time())
                # psutil needs one baseline before cpu_percent can report a
                # delta.  Reuse this object on all later samples.
                process.cpu_percent(None)
                continue
            if (
                not process.is_running()
                or process.status() == psutil.STATUS_ZOMBIE
                or float(process.create_time()) != process_identity.get(pid)
            ):
                active.discard(pid)
                process_cache.pop(pid, None)
                process_identity.pop(pid, None)
                continue
            live_pids.append(pid)
            cpu_value = process.cpu_percent(None)
            memory = process.memory_info().rss
            if cpu_value is not None:
                cpu += float(cpu_value)
            rss += int(memory)
        except (psutil.Error, OSError):
            active.discard(pid)
            process_cache.pop(pid, None)
            process_identity.pop(pid, None)
            continue
    return cpu, rss, live_pids, sorted(active)


def run_profiling_campaign(
    executable: Optional[Path] = None,
    frame_file: Optional[Path] = None,
    bank_file: Optional[Path] = None,
    output_hdf: Optional[Path] = None,
    output_json: Optional[Path] = None,
    python_bin: str = sys.executable,
    batch_size: int = 128,
    sampling_interval_ms: int = 25,
    affinity_core: str = "8",
    approximant: str = "IMRPhenomD",
    nvtx_sync: bool = False,
    command: Optional[Sequence[str]] = None,
    workload: Optional[Mapping[str, Any]] = None,
    environment: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
    completion_grace_sec: float = 30.0,
) -> Dict[str, Any]:
    """Run a PyCBC command while collecting complete process-tree telemetry.

    The historical positional arguments still construct the standard
    ``pycbc_inspiral`` invocation.  Supplying ``command`` makes this helper
    usable for ``pycbc_live`` and other campaign commands without shell
    interpolation.  CPU, RSS and process VRAM fields are scoped to the root
    process and descendants observed during the run.
    """
    if sampling_interval_ms <= 0:
        raise ValueError("sampling_interval_ms must be positive")
    if completion_grace_sec < 0:
        raise ValueError("completion_grace_sec must be non-negative")
    if output_json is None:
        raise ValueError("output_json is required")
    output_json = Path(output_json).resolve()
    output_hdf = Path(output_hdf).resolve() if output_hdf else None
    executable = Path(executable).resolve() if executable else None
    frame_file = Path(frame_file).resolve() if frame_file else None
    bank_file = Path(bank_file).resolve() if bank_file else None
    if output_hdf is not None:
        output_hdf.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    standard_command = command is None
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
        "2048",
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
    if standard_command:
        if executable is None or frame_file is None or bank_file is None:
            raise ValueError(
                "executable, frame_file and bank_file are required "
                "when command is not supplied"
            )
        if output_hdf is None:
            raise ValueError("output_hdf is required for the standard command")
        command = [python_bin, str(executable), *cli_args]
    else:
        command = [str(item) for item in command]
        if not command:
            raise ValueError("command must not be empty")
    if sys.platform == "linux" and shutil.which("taskset") and affinity_core:
        command = ["taskset", "-c", affinity_core] + command

    explicit_environment = {
        str(key): str(value) for key, value in (environment or {}).items()
    }
    forced_environment = {
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
            # Structured stage markers are cheap stderr records; the profiler
            # enables them for both inspiral and live generic commands.
            "PYCBC_BENCHMARK_STAGES": "1",
    }
    env = os.environ.copy()
    env.update(explicit_environment)
    env.update(forced_environment)

    run_cwd = Path(cwd).resolve() if cwd else (
        output_hdf.parent if output_hdf else output_json.parent
    )
    # Keep provenance useful without copying the complete inherited process
    # environment (which may contain credentials or unrelated host settings).
    profile_source = {
        "command": command,
        "cwd": str(run_cwd),
        "workload": dict(workload or {}),
        "explicit_environment": explicit_environment,
    }

    nvml = NVMLTelemetryCollector(device_index=0)

    telemetry: List[Dict[str, Any]] = []
    stop_event = threading.Event()
    process_holder: List[Optional[subprocess.Popen]] = [None]
    stderr_lines: List[Tuple[float, str]] = []
    known_pids: set[int] = set()
    active_pids: set[int] = set()
    process_membership: List[Dict[str, Any]] = []
    process_cache: Dict[int, psutil.Process] = {}
    process_identity: Dict[int, float] = {}
    previous_live_pids: set[int] = set()

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
    origin_monotonic_ns = time.monotonic_ns()

    def telemetry_worker():
        nonlocal previous_live_pids
        interval = sampling_interval_ms / 1000.0
        target_pid = None
        previous_disk = None

        while not stop_event.is_set():
            now = time.perf_counter()
            elapsed = now - t_start

            if process_holder[0] is not None and target_pid is None:
                target_pid = process_holder[0].pid
                _process_tree(target_pid, known_pids, active_pids)
            proc_cpu, proc_rss, live_pids, observed_pids = (
                _sample_process_tree(
                    target_pid, known_pids, process_cache, active_pids,
                    process_identity,
                )
            )
            live_set = set(live_pids)
            added_pids = sorted(live_set - previous_live_pids)
            removed_pids = sorted(previous_live_pids - live_set)
            if added_pids or removed_pids or not process_membership:
                # Membership is an event stream.  The historical inventory is
                # emitted once in process_tree below; repeating it here made
                # long campaigns grow quadratically in both JSON size and
                # serialization time.
                process_membership.append({
                    "elapsed_sec": round(elapsed, 4),
                    "added_pids": added_pids,
                    "removed_pids": removed_pids,
                    "live_pids": live_pids,
                })
            previous_live_pids = live_set
            try:
                gpu_metrics = nvml.sample(
                    target_pid=target_pid, target_pids=active_pids
                )
            except TypeError:
                # Keep compatibility with lightweight test collectors and
                # third-party wrappers implementing the original API.
                gpu_metrics = nvml.sample(target_pid=target_pid)
            sys_cpu = psutil.cpu_percent(None)
            sys_mem = psutil.virtual_memory()
            try:
                disk_counters = psutil.disk_io_counters()
            except (OSError, psutil.Error):
                disk_counters = None
            disk_read_rate = disk_write_rate = None
            if disk_counters is not None:
                if previous_disk is not None:
                    previous_time, previous_read, previous_write = previous_disk
                    seconds = now - previous_time
                    read_delta = disk_counters.read_bytes - previous_read
                    write_delta = disk_counters.write_bytes - previous_write
                    if seconds > 0 and read_delta >= 0 and write_delta >= 0:
                        disk_read_rate = round(read_delta / seconds / (1024**2), 3)
                        disk_write_rate = round(write_delta / seconds / (1024**2), 3)
                previous_disk = (
                    now, disk_counters.read_bytes, disk_counters.write_bytes
                )
            try:
                disk_used = round(
                    shutil.disk_usage(run_cwd).used / (1024**3), 3
                )
            except OSError:
                disk_used = None

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
                "gpu_proc_vram_mib": (
                    round(gpu_metrics["gpu_proc_used_bytes"] / (1024**2), 2)
                    if gpu_metrics.get("gpu_proc_used_bytes") is not None
                    else None
                ),
                "gpu_device_used_mib": (
                    round(gpu_metrics["gpu_device_used_bytes"] / (1024**2), 2)
                    if gpu_metrics.get("gpu_device_used_bytes") is not None
                    else None
                ),
                "gpu_device_total_mib": (
                    round(gpu_metrics["gpu_device_total_bytes"] / (1024**2), 2)
                    if gpu_metrics.get("gpu_device_total_bytes") is not None
                    else None
                ),
                "proc_cpu_percent": round(proc_cpu, 2),
                "sys_cpu_percent": round(sys_cpu, 2),
                "proc_rss_mib": round(proc_rss / (1024**2), 2),
                "process_count": len(live_pids),
                "process_pids": live_pids,
                "sys_mem_used_gib": round(sys_mem.used / (1024**3), 2),
                "disk_read_mib_per_sec": disk_read_rate,
                "disk_write_mib_per_sec": disk_write_rate,
                "disk_used_gib": disk_used,
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
        cwd=str(run_cwd),
    )
    process_holder[0] = proc
    # Record the root immediately so very short commands still have a
    # complete process identity even if the sampler has not run yet.
    known_pids.add(proc.pid)
    active_pids.add(proc.pid)

    # Read stderr in real time to stamp log line events
    if proc.stderr is not None:
        for line in proc.stderr:
            t_line = time.perf_counter() - t_start
            clean_line = line.rstrip()
            stderr_lines.append((round(t_line, 4), clean_line))
            print(f"[{t_line:6.2f}s] {clean_line}", flush=True)

    proc.wait()
    completion = _wait_for_process_tree(
        proc.pid, known_pids, completion_grace_sec, active_pids,
        process_cache, process_identity,
    )
    t_end = time.perf_counter()
    stop_event.set()
    worker_thread.join(timeout=2.0)
    nvml.close()

    total_wall_sec = t_end - t_start

    # Parse phase boundaries from timestamped stderr lines
    stage_events = _parse_stage_events(
        stderr_lines, total_wall_sec, origin_monotonic_ns
    )
    structured_phases = _structured_stage_phases(stage_events, total_wall_sec)
    phases = structured_phases or _identify_pipeline_phases(
        stderr_lines, total_wall_sec
    )

    # Calculate summary metrics
    gpu_utils = [s["gpu_util_percent"] for s in telemetry
                 if s.get("gpu_util_percent") is not None]
    proc_vrams = [s["gpu_proc_vram_mib"] for s in telemetry
                  if s.get("gpu_proc_vram_mib") is not None]
    proc_cpus = [s["proc_cpu_percent"] for s in telemetry
                 if s.get("proc_cpu_percent") is not None]
    proc_rsss = [s["proc_rss_mib"] for s in telemetry
                 if s.get("proc_rss_mib") is not None]
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
        "peak_gpu_util_percent": max(gpu_utils) if gpu_utils else None,
        "mean_gpu_util_percent": (
            round(float(sum(gpu_utils) / len(gpu_utils)), 2)
            if gpu_utils
            else None
        ),
        "peak_proc_vram_mib": max(proc_vrams) if proc_vrams else None,
        "peak_proc_cpu_percent": max(proc_cpus) if proc_cpus else None,
        "mean_proc_cpu_percent": (
            round(float(sum(proc_cpus) / len(proc_cpus)), 2)
            if proc_cpus
            else None
        ),
        "peak_proc_rss_mib": max(proc_rsss) if proc_rsss else None,
        "observed_process_count": len(known_pids),
    }

    if standard_command:
        result_workload = {
            "track": "track1_compressed",
            "batch_size": batch_size,
            "sample_rate_hz": 2048,
            "precision": "single",
            "signal_dtype": "complex64",
            "segment_length_sec": 512,
            "frame_file": str(frame_file) if frame_file else None,
            "bank_file": str(bank_file) if bank_file else None,
            "approximant": approximant,
            "processing_scheme": "jax:cuda:0",
        }
    else:
        # Generic campaigns must describe only the workload supplied by their
        # runner.  Standard inspiral defaults would be fabricated metadata for
        # pycbc_live, CPU commands, or other arbitrary commands.
        result_workload = dict(workload or {})

    result = {
        "schema_version": 3,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "command": command,
        "returncode": proc.returncode,
        "target_pid": proc.pid,
        "cwd": str(run_cwd),
        "target_pids": sorted(known_pids),
        "process_tree": {
            "root_pid": proc.pid,
            "observed_pids": sorted(known_pids),
            "completion": completion,
        },
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
            "process_membership_encoding": "changes; historical inventory is in process_tree.observed_pids",
            "cpu_percent": (
                "100% is one logical CPU; root process and all observed "
                "descendants summed"
            ),
            "gpu_proc_scope": "sum of NVML process allocations for observed PIDs",
            "disk_io_scope": "whole host; includes other processes",
            "disk_io_source": "psutil.disk_io_counters read_bytes/write_bytes",
            "disk_usage_scope": "filesystem containing command cwd",
            "disk_usage_path": str(run_cwd),
            "affinity_core": affinity_core,
            "xla_preallocate": False,
            "command_cwd": str(run_cwd),
            "command_argv_hash": _stable_hash(command),
            "explicit_environment_keys": sorted(explicit_environment),
            "explicit_environment_hash": _stable_hash(explicit_environment),
            "forced_environment_keys": sorted(forced_environment),
            "forced_environment_hash": _stable_hash(forced_environment),
            "source_config_hash": _stable_hash(profile_source),
        },
        "workload": result_workload,
        "summary": summary,
        "phases": phases,
        "stage_events": stage_events,
        "telemetry": telemetry,
        "stderr_log": stderr_lines,
        "process_membership": process_membership,
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


def run_command_profiling(
    command: Sequence[str], output_json: Path, **kwargs
) -> Dict[str, Any]:
    """Explicit generic-command alias used by live campaign runners."""
    return run_profiling_campaign(
        command=command, output_json=output_json, **kwargs
    )


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


STAGE_EVENT_PREFIX = "PYCBC_STAGE_EVENT "


def _parse_stage_events(
    stderr_lines: Sequence[Tuple[float, str]],
    total_wall: float,
    origin_monotonic_ns: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Parse structured stage markers without requiring stage synchronization."""
    events = []
    for receipt_time, line in stderr_lines:
        if not line.startswith(STAGE_EVENT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(STAGE_EVENT_PREFIX):])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        event = str(payload.get("event", ""))
        stage = str(payload.get("stage", payload.get("name", "")))
        if event not in {"start", "end"} or not stage:
            continue
        timestamp = None
        if payload.get("monotonic_ns") is not None and origin_monotonic_ns is not None:
            try:
                timestamp = (
                    int(payload["monotonic_ns"]) - origin_monotonic_ns
                ) / 1_000_000_000.0
            except (TypeError, ValueError):
                timestamp = None
        if timestamp is None:
            timestamp = payload.get("elapsed_sec")
        if timestamp is None:
            timestamp = receipt_time
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError):
            timestamp = receipt_time
        if not 0 <= timestamp <= total_wall:
            continue
        item = dict(payload)
        item.update({
            "event": event,
            "stage": stage,
            "elapsed_sec": timestamp,
            "receipt_elapsed_sec": receipt_time,
        })
        events.append(item)
    return events


def _structured_stage_phases(
    events: Sequence[Mapping[str, Any]], total_wall: float
) -> List[Dict[str, Any]]:
    """Pair structured stage events, retaining unmatched intervals explicitly."""
    active: Dict[tuple, List[Mapping[str, Any]]] = {}
    phases = []
    for event in sorted(events, key=lambda item: item["elapsed_sec"]):
        stage = str(event["stage"])
        key = (event.get("pid"), event.get("rank"), stage)
        timestamp = float(event["elapsed_sec"])
        if event["event"] == "start":
            active.setdefault(key, []).append(event)
            continue
        starts = active.get(key)
        start_event = starts.pop() if starts else None
        if starts == []:
            active.pop(key, None)
        if start_event is None:
            continue
        start = float(start_event["elapsed_sec"])
        if timestamp <= start:
            continue
        item = {
            "name": stage,
            "label": str(start_event.get("label", stage)),
            "start_sec": start,
            "end_sec": timestamp,
            "duration_sec": round(timestamp - start, 4),
            "boundary_source": "structured_event",
        }
        for key_name in ("pid", "rank"):
            if key_name in start_event:
                item[key_name] = start_event[key_name]
        for key in ("templates", "core", "metadata"):
            if key in start_event:
                item[key] = start_event[key]
        phases.append(item)
    # A live stage can remain open if a process is killed.  Keep it visible,
    # but avoid fabricating its duration for summary comparisons.
    for (pid, rank, stage), starts in active.items():
        for start_event in starts:
            start = float(start_event["elapsed_sec"])
            if total_wall <= start:
                continue
            item = {
                "name": stage,
                "label": str(start_event.get("label", stage)),
                "start_sec": start,
                "end_sec": None,
                "duration_sec": None,
                "boundary_source": "structured_event_open",
            }
            for key_name, value in (("pid", pid), ("rank", rank)):
                if value is not None:
                    item[key_name] = value
            for key in ("templates", "core", "metadata"):
                if key in start_event:
                    item[key] = start_event[key]
            phases.append(item)
    return sorted(phases, key=lambda item: item["start_sec"])


def _wait_for_process_tree(
    root_pid: int,
    known_pids: set[int],
    grace_sec: float,
    active_pids: Optional[set[int]] = None,
    process_cache: Optional[Dict[int, psutil.Process]] = None,
    process_identity: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Wait briefly for descendants and record completion membership."""
    active = active_pids if active_pids is not None else set(known_pids)
    process_cache = process_cache if process_cache is not None else {}
    process_identity = process_identity if process_identity is not None else {}

    def live_members():
        _process_tree(root_pid, known_pids, active)
        result = []
        for pid in list(active):
            try:
                process = psutil.Process(pid)
                if (
                    process.is_running()
                    and process.status() != psutil.STATUS_ZOMBIE
                    and (
                        pid not in process_identity
                        or float(process.create_time()) == process_identity[pid]
                    )
                ):
                    result.append(pid)
                else:
                    active.discard(pid)
            except psutil.Error:
                active.discard(pid)
        return result

    live = live_members()
    if not live:
        return {
            "root_exited": True,
            "all_observed_processes_exited": True,
            "remaining_pids": [],
        }
    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        live = live_members()
        if not live:
            return {
                "root_exited": True,
                "all_observed_processes_exited": True,
                "remaining_pids": [],
            }
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    return {
        "root_exited": True,
        "all_observed_processes_exited": False,
        "remaining_pids": sorted(live),
    }


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
        "--frame-file", type=Path,
        help="Path to GWF strain frame file (standard inspiral command)",
    )
    parser.add_argument(
        "--bank-file",
        type=Path,
        help="Path to compressed template bank HDF5 file (standard command)",
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
        default=Path("artifacts/jax_gpu_timeline.json"),
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
        "--command-json",
        type=Path,
        help=(
            "JSON file containing argv, or an object with command, workload, "
            "environment and cwd for a generic campaign"
        ),
    )

    parser.add_argument(
        "--nvtx-sync", action="store_true",
        help="Emit an NVTX clock marker when running the collector under nsys",
    )
    args = parser.parse_args()

    generic = {}
    if args.command_json:
        with args.command_json.open("r") as command_file:
            generic = json.load(command_file)
        if isinstance(generic, list):
            generic = {"command": generic}
        if not isinstance(generic, dict) or not generic.get("command"):
            parser.error("--command-json must contain a command argv list")

    result = run_profiling_campaign(
        executable=None if generic else args.executable,
        frame_file=None if generic else args.frame_file,
        bank_file=None if generic else args.bank_file,
        output_hdf=args.output_hdf,
        output_json=args.output_json,
        command=generic.get("command"),
        workload=generic.get("workload"),
        environment=generic.get("environment"),
        cwd=generic.get("cwd"),
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
