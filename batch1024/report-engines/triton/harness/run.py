#!/usr/bin/env python3
"""Counterbalanced, sequential fresh-process TaylorF2 on/off experiment."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from worker import ROUTES, THREAD_VARS, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-sha", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--routes", nargs="+", choices=ROUTES, default=["cuda-off", "cuda-on"])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128, 512])
    p.add_argument("--delta-f", nargs="+", type=float, choices=(0.25, 0.03125), default=[0.25, 0.03125])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--sample-ms", type=float, default=50)
    p.add_argument("--max-inner", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=240)
    a = p.parse_args()
    if min(a.batches + [a.threads, a.replicates, a.samples, a.max_inner]) < 1:
        p.error("counts must be positive")
    if a.sample_ms < 50 or a.timeout <= 0:
        p.error("sample-ms must be at least 50 and timeout must be positive")
    if len(set(a.routes)) != len(a.routes) or len(set(a.batches)) != len(a.batches) or len(set(a.delta_f)) != len(a.delta_f):
        p.error("route, batch and frequency lists must contain no duplicates")
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "manifest.json").exists():
        p.error("output already contains a manifest; choose a fresh directory")
    worker = Path(__file__).with_name("worker.py").resolve()
    manifest = dict(schema=1, command=sys.argv, cwd=os.getcwd(), pid=os.getpid(),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        expected_sha=a.expected_sha,
        worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
        orchestrator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        method="sequential fresh processes; route order rotated by replicate within each workload; median of worker medians, no pooled inner calls",
        conditions=dict(batches=a.batches, delta_f=a.delta_f, routes=a.routes,
                        threads=a.threads, replicates=a.replicates, samples=a.samples,
                        minimum_sample_ms=a.sample_ms), jobs=[])
    write_json(a.out / "manifest.json", manifest)
    active = None

    def stop(signum, _frame):
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()
        manifest["interrupted_signal"] = signum
        manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(a.out / "manifest.json", manifest)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    results = {}

    def summarize():
        groups = {}
        for delta_f, batch, route in itertools.product(a.delta_f, a.batches, a.routes):
            key = f"{route}-b{batch}-n{int(1024 / delta_f) + 1}"
            values = results.get(key, [])
            valid = [r for r in values if r["status"] == "ok" and
                     r.get("source", {}).get("sha") == a.expected_sha and
                     r.get("source", {}).get("clean") and
                     r.get("harness_sha256") == manifest["worker_sha256"] and
                     r.get("dispatch", {}).get("passed") and
                     r.get("timing", {}).get("minimum_duration_passed")]
            group = dict(route=route, batch=batch, bins=int(1024 / delta_f) + 1,
                delta_f=delta_f, threads=a.threads, expected_replicates=a.replicates,
                completed_replicates=len(values), valid_replicates=len(valid),
                statuses=[r["status"] for r in values], reasons=[r.get("reason") for r in values],
                eligible=len(valid) == a.replicates,
                parameter_sha256=values[0].get("parameters_sha256") if values else None)
            if group["eligible"] and len({r["parameters_sha256"] for r in valid}) != 1:
                group.update(eligible=False, qualification_error="replicate inputs differ")
            if group["eligible"]:
                medians = [r["timing"]["median_seconds_per_call"] for r in valid]
                center = statistics.median(medians)
                group.update(replicate_medians_seconds=medians, median_seconds_per_call=center,
                    min_replicate_median_seconds=min(medians), max_replicate_median_seconds=max(medians),
                    waveforms_per_second=batch / center,
                    cold_call_seconds=[r["timing"]["cold_call_seconds"] for r in valid],
                    triton_actual=[r["dispatch"]["triton_actual"] for r in valid])
            groups[key] = group
        for group in groups.values():
            if not group["eligible"]:
                continue
            for base_route, ratio_key in (("cuda-off", "speedup_vs_cuda_gate_off"),
                                           ("standard-cpu", "speedup_vs_standard_cpu")):
                base = groups.get(f"{base_route}-b{group['batch']}-n{group['bins']}")
                if base and base["eligible"] and base["parameter_sha256"] == group["parameter_sha256"]:
                    group[ratio_key] = base["median_seconds_per_call"] / group["median_seconds_per_call"]
        output = dict(schema=1, method=manifest["method"], expected_sha=a.expected_sha,
            uncertainty="observed range of worker medians; not a confidence interval",
            groups=groups)
        write_json(a.out / "summary.json", output)
        return groups

    for delta_f, batch, replicate in itertools.product(a.delta_f, a.batches, range(1, a.replicates + 1)):
        offset = (replicate - 1) % len(a.routes)
        ordered_routes = a.routes[offset:] + a.routes[:offset]
        for route in ordered_routes:
            key = f"{route}-b{batch}-n{int(1024 / delta_f) + 1}"
            stem = f"{key}-r{replicate}"
            output, log = a.out / f"{stem}.json", a.out / f"{stem}.log"
            command = [a.python, "-B", str(worker), "--root", str(a.root.resolve()),
                "--out", str(output), "--expected-sha", a.expected_sha, "--route", route,
                "--batch", str(batch), "--delta-f", str(delta_f), "--threads", str(a.threads),
                "--replicate", str(replicate), "--samples", str(a.samples),
                "--sample-ms", str(a.sample_ms), "--max-inner", str(a.max_inner)]
            env = os.environ.copy()
            env.update({name: str(a.threads) for name in THREAD_VARS})
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env.pop("PYTHONPATH", None)
            job = dict(key=key, replicate=replicate, command=command, log=str(log), output=str(output), status="running")
            manifest["jobs"].append(job)
            write_json(a.out / "manifest.json", manifest)
            print(f"START {stem} log={log}", flush=True)
            begin = time.perf_counter()
            with log.open("w") as stream:
                active = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                          env=env, start_new_session=True)
                job["pid"] = active.pid
                write_json(a.out / "manifest.json", manifest)
                try:
                    job["returncode"] = active.wait(timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(active.pid, signal.SIGKILL)
                    active.wait()
                    job.update(returncode=active.returncode, timeout=True)
                active = None
            job["seconds"] = time.perf_counter() - begin
            if output.exists():
                result = json.loads(output.read_text())
            else:
                result = dict(status="failed", reason="worker timed out or exited without JSON", job=job)
                write_json(output, result)
            if job["returncode"] != 0 and result["status"] == "ok":
                result.update(status="failed", reason="worker returned nonzero exit status")
                write_json(output, result)
            job["status"] = result["status"]
            results.setdefault(key, []).append(result)
            write_json(a.out / "manifest.json", manifest)
            summarize()
            print(f"DONE {stem} {result['status']} {job['seconds']:.2f}s", flush=True)
    groups = summarize()
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["status"] = "ok" if all(g["eligible"] for g in groups.values()) else "failed"
    write_json(a.out / "manifest.json", manifest)
    return int(manifest["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
