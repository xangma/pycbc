#!/usr/bin/env python3
"""Run all 60 untimed live dispatch cells serially, preserving failed cells."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import traceback

SPEC = importlib.util.spec_from_file_location(
    "live_support", Path(__file__).with_name("run-live.py")
)
SUPPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPPORT)


def cells():
    values = [
        (head, route, threads, batch)
        for head in ("main", "cpu")
        for threads in (1, 4)
        for route in ("torch_cpu", "torch_cpu_native")
        for batch in SUPPORT.BATCHES
    ]
    values += [
        ("main", route, 1, batch)
        for route in ("torch_cuda", "torch_cuda_native")
        for batch in SUPPORT.BATCHES
    ]
    return values


def main():
    args = SUPPORT.parse_args()
    assert not args.session
    args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, SUPPORT.terminate)
    signal.signal(signal.SIGINT, SUPPORT.terminate)
    probe = Path(__file__).with_name("live_dispatch_probe.py").resolve()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    results = []
    thread = threading.Thread(
        target=SUPPORT.monitor, args=(args, args.output), daemon=True
    )
    thread.start()
    SUPPORT.write_json(
        args.output / "manifest.json",
        dict(
            started_utc=SUPPORT.utc(),
            expected_cells=len(cells()),
            cells=cells(),
            generator_sha256=SUPPORT.sha256(Path(__file__)),
            probe_sha256=SUPPORT.sha256(probe),
            controller_pid=os.getpid(),
            heads=dict(main=args.expected_main, cpu=args.expected_cpu),
        ),
    )
    try:
        for head, route, threads, batch in cells():
            label = f"probe-{head}-{route}-t{threads}-b{batch}"
            root = getattr(args, head + "_root")
            command = [
                args.python,
                str(probe),
                "--source-root",
                str(root),
                "--expected-revision",
                getattr(args, "expected_" + head),
                "--route",
                route,
                "--threads",
                str(threads),
                "--batch",
                str(batch),
                "--size",
                "131072",
                "--cuda-device",
                str(args.cuda_device),
                "--output",
                str(args.output / f"{label}.json"),
            ]
            if args.affinity:
                command = ["taskset", "-c", args.affinity, *command]
            result = dict(
                label=label, started_utc=SUPPORT.utc(), command=command, cwd=str(root)
            )
            try:
                result["capacity_before"] = SUPPORT.capacity(args, "cuda" in route)
                with (args.output / f"{label}.log").open("x") as handle:
                    SUPPORT.ACTIVE = subprocess.Popen(
                        command,
                        cwd=root,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    result["pid"] = SUPPORT.ACTIVE.pid
                    SUPPORT.write_json(args.output / "active.json", result)
                    print(json.dumps(result), flush=True)
                    try:
                        result["returncode"] = SUPPORT.ACTIVE.wait(
                            timeout=args.worker_timeout
                        )
                    except subprocess.TimeoutExpired:
                        os.killpg(SUPPORT.ACTIVE.pid, signal.SIGTERM)
                        try:
                            SUPPORT.ACTIVE.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(SUPPORT.ACTIVE.pid, signal.SIGKILL)
                            SUPPORT.ACTIVE.wait()
                        result.update(
                            returncode=SUPPORT.ACTIVE.returncode, timeout=True
                        )
            except Exception as exc:
                result.update(
                    returncode=1, error=str(exc), traceback=traceback.format_exc()
                )
            result["finished_utc"] = SUPPORT.utc()
            results.append(result)
            SUPPORT.write_json(args.output / f"{label}.receipt.json", result)
            SUPPORT.write_json(args.output / "runs.json", results)
            print(
                json.dumps(
                    {k: result[k] for k in ("label", "returncode", "finished_utc")}
                ),
                flush=True,
            )
    finally:
        SUPPORT.STOP.set()
        thread.join(timeout=10)
    passed = len(results) == len(cells()) and all(r["returncode"] == 0 for r in results)
    SUPPORT.write_json(
        args.output / "status.json",
        dict(
            finished_utc=SUPPORT.utc(),
            finished=True,
            passed=passed,
            expected_cells=len(cells()),
            completed_cells=len(results),
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
