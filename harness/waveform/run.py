#!/usr/bin/env python3
"""Run independent TaylorF2 workers sequentially and aggregate only valid rows."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from worker import ROUTES, SHA, THREAD_VARS


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    p.add_argument("--precisions", nargs="+", choices=["double", "single"], default=["double", "single"])
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--sample-ms", type=float, default=50)
    p.add_argument("--max-inner", type=int, default=64)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--expected-sha", default=SHA)
    a = p.parse_args()
    if min(a.batches + a.threads + [a.replicates, a.samples, a.max_inner]) < 1:
        p.error("counts must be positive")
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "manifest.json").exists():
        p.error("output directory already contains a manifest; choose a fresh output directory")
    worker = Path(__file__).with_name("worker.py").resolve()
    manifest = {"schema": 1, "command": sys.argv, "cwd": os.getcwd(), "pid": os.getpid(),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "expected_sha": a.expected_sha,
                "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
                "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "method": "sequential fresh processes; median of replicate medians; no pooling of inner calls",
                "jobs": []}
    write_json(a.out / "manifest.json", manifest)
    groups = {}
    # Fixed threads=1 phase then threads=4; CUDA uses host threads=1 only.
    for threads, route, batch, precision in itertools.product(a.threads, a.routes, a.batches, a.precisions):
        if "cuda" in route and threads != a.threads[0]:
            continue
        key = f"{route}-b{batch}-t{threads}-{precision}"
        results = []
        for rep in range(1, a.replicates + 1):
            stem = f"{key}-r{rep}"
            output, log = a.out / f"{stem}.json", a.out / f"{stem}.log"
            command = [a.python, "-B", str(worker), "--root", str(a.root.resolve()), "--out", str(output),
                       "--route", route, "--batch", str(batch), "--threads", str(threads),
                       "--precision", precision, "--replicate", str(rep), "--samples", str(a.samples),
                       "--sample-ms", str(a.sample_ms), "--max-inner", str(a.max_inner),
                       "--expected-sha", a.expected_sha]
            env = os.environ.copy()
            env.update({name: str(threads) for name in THREAD_VARS})
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            job = {"key": key, "replicate": rep, "command": command, "log": str(log), "output": str(output)}
            manifest["jobs"].append(job)
            write_json(a.out / "manifest.json", manifest)
            print(f"START {stem} log={log}", flush=True)
            start = time.perf_counter()
            with log.open("w") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
                job["pid"] = process.pid
                write_json(a.out / "manifest.json", manifest)
                try:
                    job["returncode"] = process.wait(timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    job["returncode"], job["timeout"] = process.returncode, True
            job["seconds"] = time.perf_counter() - start
            if output.exists():
                result = json.loads(output.read_text())
            else:
                result = {"status": "failed", "reason": "worker timed out or exited without JSON", "job": job}
                write_json(output, result)
            if job["returncode"] != 0 and result["status"] == "ok":
                result["status"], result["reason"] = "failed", "nonzero worker exit"
            job["status"] = result["status"]
            results.append(result)
            print(f"DONE {stem} {result['status']} {job['seconds']:.2f}s", flush=True)
            write_json(a.out / "manifest.json", manifest)
        valid = [x for x in results if x["status"] == "ok"]
        group = {"route": route, "batch": batch, "threads": threads, "precision": precision,
                 "replicate_count": len(results), "valid_replicates": len(valid),
                 "statuses": [x["status"] for x in results],
                 "reasons": [x.get("reason") for x in results], "eligible": len(valid) == a.replicates}
        if group["eligible"]:
            medians = [x["timing"]["median_seconds_per_call"] for x in valid]
            group.update(replicate_medians_seconds=medians,
                         median_seconds_per_call=statistics.median(medians),
                         min_replicate_median_seconds=min(medians), max_replicate_median_seconds=max(medians),
                         cold_call_seconds=[x["timing"]["cold_call_seconds"] for x in valid],
                         waveforms_per_second=batch / statistics.median(medians))
        groups[key] = group
        write_json(a.out / "summary.json", {"method": manifest["method"], "groups": groups})
    for group in groups.values():
        baseline = groups.get(f"standard-cpu-b{group['batch']}-t{group['threads']}-{group['precision']}")
        if group["eligible"] and baseline and baseline["eligible"]:
            group["speedup_vs_standard_cpu_same_threads"] = (
                baseline["median_seconds_per_call"] / group["median_seconds_per_call"])
    write_json(a.out / "summary.json", {"method": manifest["method"], "groups": groups})
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(a.out / "manifest.json", manifest)
    return int(any("failed" in group["statuses"] for group in groups.values()))


if __name__ == "__main__":
    raise SystemExit(main())
