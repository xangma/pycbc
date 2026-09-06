#!/usr/bin/env python3
"""Whole-executable Torch CPU/CUDA tracing, not benchmark timing.

Usage: python profile-inspiral-torch.py --output-dir NEW_DIR -- /source/bin/pycbc_inspiral CLI
"""
import argparse
import hashlib
import json
import os
import runpy
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="failed", argv=command, executable_exit_code=None, host=os.uname().nodename, cwd=str(Path.cwd()), pid=os.getpid(),
                   wrapper=identity(Path(__file__).resolve()), started_utc=datetime.now(timezone.utc).isoformat(),
                   purpose="Instrumented Torch operator/device trace; not benchmark timing. Torch import and profiler initialization precede executable entry.")
    old_argv, old_path, profiler, code = sys.argv, sys.path[:], None, 1
    try:
        executable = Path(command[0]).resolve(strict=True)
        if executable.name != "pycbc_inspiral" or executable.parent.name != "bin":
            raise ValueError("Require the exact source/bin/pycbc_inspiral executable")
        receipt["executable"] = identity(executable)
        receipt["source_root"] = str(executable.parent.parent)
        sys.path.insert(0, str(executable.parent.parent))
        sys.argv = [str(executable), *command[1:]]
        receipt["executed_argv"] = sys.argv[:]
        import torch

        receipt["torch_version"] = str(torch.__version__)
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        if not torch.cuda.is_available() or not set(activities).issubset(torch.profiler.supported_activities()):
            raise RuntimeError("CUDA device and CPU/CUDA profiler activities are required")
        receipt["affinity"] = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        profiler = torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False, with_stack=False)
        with profiler:
            try:
                runpy.run_path(str(executable), run_name="__main__")
            except BaseException as error:
                receipt["executable_exit_code"] = (error.code or 0) if isinstance(error, SystemExit) and isinstance(error.code, (int, type(None))) else 1
                if not isinstance(error, SystemExit) or receipt["executable_exit_code"]:
                    raise
            else:
                receipt["executable_exit_code"] = 0
            finally:
                torch.cuda.synchronize()
        pycbc = sys.modules.get("pycbc")
        receipt["imported_pycbc"] = str(Path(pycbc.__file__).resolve()) if pycbc else None
        if pycbc and Path(pycbc.__file__).resolve().parent.parent != executable.parent.parent:
            raise RuntimeError("Imported PyCBC differs from the executable source root")
        code, receipt["status"] = 0, "success"
    except BaseException as error:
        receipt["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        sys.argv, sys.path[:] = old_argv, old_path
        try:
            if profiler is not None:
                trace = out / "trace.json"
                profiler.export_chrome_trace(str(trace))
                rows = [dict(key=e.key, count=e.count, device_type=str(e.device_type), self_cpu_time_us=e.self_cpu_time_total,
                             self_device_time_us=e.self_device_time_total) for e in profiler.key_averages()]
                summary = dict(units="microseconds", rows=rows,
                    cpu_event_self_time_us=sum(r["self_cpu_time_us"] for r in rows if r["device_type"].endswith(".CPU")),
                    cuda_device_event_self_time_us=sum(r["self_device_time_us"] for r in rows if r["device_type"].endswith(".CUDA")),
                    note="CUDA total includes only CUDA device-event rows, excluding CPU operator associations to prevent double counting. Device-event durations may overlap across streams; their sum is neither wall time nor utilization. CPU data covers profiler-visible operators/runtime events, not all Python/native execution.")
                averages = out / "key-averages.json"
                averages.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
                receipt.update(trace=identity(trace), key_averages=identity(averages))
                if summary["cuda_device_event_self_time_us"] <= 0:
                    raise RuntimeError("No positive CUDA device-event time captured; trace is incomplete")
        except BaseException as error:
            receipt["status"], code = "failed", 1
            receipt["export_error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        receipt.update(wrapper_exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat())
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
