#!/usr/bin/env python3
"""Launch the probe in the existing checked CPU environment; run on len only."""

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("/home/xangma/pycbc-torch-baseline-final-20260908")
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=Path(
            "/home/xangma/pycbc-torch-precision-validation-20260908/attempt-2/capture-proposed-cpu/capture"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument(
        "--max-seconds", type=int, default=300, choices=range(30, 301), metavar="30..300"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must be new")
    base = args.base.resolve()
    spec = importlib.util.spec_from_file_location(
        "checked_probe_launcher", base / "checked-inspiral.py"
    )
    checked = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checked)
    config = json.loads((base / "config.json").read_text())
    environment = checked.clean_environment(os.environ, config, base / "proposed")
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).with_name("injection-probe.py")),
        "--capture",
        str(args.capture),
        "--source",
        str(base / "proposed"),
        "--original-source",
        str(base / "original"),
        "--source-pins",
        str(args.capture.parent / "source-pins.json"),
        "--output",
        str(args.output),
        "--cpu",
        str(args.cpu),
        "--max-seconds",
        str(args.max_seconds),
    ]
    # The inherited descriptor holds the existing benchmark coordination lock.
    # A busy lock fails immediately; no other workload is interrupted.
    with checked.LOCK.open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        child = subprocess.Popen(command, env=environment, start_new_session=True,
                                 pass_fds=(lock.fileno(),))
        receipt = dict(status="running", launcher_pid=os.getpid(), child_pid=child.pid,
                       command=command, host=os.uname().nodename, cwd=str(Path.cwd()),
                       log=str(args.output.resolve()/"probe.log"),
                       expected_next_check_seconds=30, deadline_seconds=args.max_seconds,
                       termination_grace_seconds=2, stop_command=f"kill -TERM {os.getpid()}")
        print(json.dumps(receipt), flush=True)

        def stop(*_):
            raise InterruptedError("Probe launcher interrupted")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            code = child.wait(timeout=max(0, args.max_seconds-(time.monotonic()-started)))
            receipt.update(status="complete" if code == 0 else "failed", returncode=code)
        except (subprocess.TimeoutExpired, InterruptedError) as error:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            receipt.update(status="failed", error=type(error).__name__, returncode=child.returncode)
            raise
        finally:
            receipt["elapsed_seconds"] = time.monotonic()-started
            if args.output.is_dir():
                (args.output/"launcher-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
            print(json.dumps(receipt), flush=True)
        raise SystemExit(code)


if __name__ == "__main__":
    main()
