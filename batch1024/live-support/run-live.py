#!/usr/bin/env python3
"""Run unchanged committed live workers, preserving every success and failure.

Only this external driver writes files. Source checkouts must remain clean.
The committed route configuration, counterbalancing and parity checks are used
without modification. Each measurement gets its own fresh Python process.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback

BATCHES = (1, 8, 32, 128, 512, 1024)
CPU_ROUTES = ("branch_standard", "torch_cpu", "torch_cpu_native")
CONDITIONS = dict(
    batches=list(BATCHES),
    replicates=3,
    samples=5,
    warmups=2,
    num_blocks=3,
    size=131072,
    seed=7101,
    call_surface="public",
    snr_threshold=5.5,
)
ACTIVE = None
STOP = threading.Event()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory():
    values = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    return int(values["MemAvailable"].split()[0]) / (1024 * 1024)


def gpu(device):
    value = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(device),
            "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    fields = [part.strip() for part in value.split(",")]
    return dict(
        uuid=fields[0],
        name=fields[1],
        utilization_percent=float(fields[2]),
        used_mib=float(fields[3]),
        free_mib=float(fields[4]),
    )


def capacity(args, cuda):
    deadline = time.monotonic() + args.capacity_wait
    while True:
        observed = dict(utc=utc(), available_ram_gib=memory())
        enough = observed["available_ram_gib"] >= args.min_ram_gib
        if cuda:
            observed["gpu"] = gpu(args.cuda_device)
            enough = enough and observed["gpu"]["free_mib"] >= args.min_vram_gib * 1024
            enough = enough and observed["gpu"]["utilization_percent"] <= 10
        if enough:
            return observed
        if time.monotonic() >= deadline:
            raise RuntimeError("Capacity gate did not clear: " + json.dumps(observed))
        time.sleep(2)


def monitor(args, output):
    with (output / "telemetry.jsonl").open("x") as handle:
        while not STOP.is_set():
            record = dict(utc=utc())
            try:
                record.update(
                    available_ram_gib=memory(),
                    gpu=gpu(args.cuda_device),
                    load=Path("/proc/loadavg").read_text().strip(),
                )
                record["compute_apps"] = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,gpu_uuid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            except Exception as exc:
                record["error"] = str(exc)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            STOP.wait(5)


def terminate(signum, _frame):
    STOP.set()
    if ACTIVE is not None and ACTIVE.poll() is None:
        os.killpg(ACTIVE.pid, signal.SIGTERM)
    raise SystemExit(128 + signum)


def timeout(_signum, _frame):
    raise TimeoutError("Committed worker exceeded its time limit")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--cpu-root", type=Path, required=True)
    parser.add_argument("--expected-main", required=True)
    parser.add_argument("--expected-cpu", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--affinity", default="8-11")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--min-ram-gib", type=float, default=16)
    parser.add_argument("--min-vram-gib", type=float, default=10)
    parser.add_argument("--capacity-wait", type=int, default=30)
    parser.add_argument("--worker-timeout", type=int, default=900)
    parser.add_argument("--session-timeout", type=int, default=14400)
    parser.add_argument(
        "--session",
        choices=("main-t1", "cpu-t1", "main-t4", "cpu-t4"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    args.main_root = args.main_root.resolve()
    args.cpu_root = args.cpu_root.resolve()
    args.output = args.output.resolve()
    return args


def session(args):
    head, thread_label = args.session.split("-")
    threads = int(thread_label[1:])
    root = getattr(args, head + "_root")
    expected = getattr(args, "expected_" + head)
    sys.path.insert(0, str(root))
    from tools import bench_production_live_batch as bench

    assert (
        Path(bench.__file__).resolve() == root / "tools/bench_production_live_batch.py"
    )
    identity = bench.source_identity(
        root,
        [
            root / "tools/bench_production_live_batch.py",
            root / "tools/benchmark_artifact.py",
        ],
    )
    assert identity["revision"] == expected, "Unexpected source revision"
    assert identity["dirty"] is False, "Source checkout must be clean"
    routes = list(CPU_ROUTES)
    if args.session == "main-t1":
        routes += ["torch_cuda", "torch_cuda_native"]
    destination = args.output / args.session
    destination.mkdir(exist_ok=False)
    records, failures, receipts = [], [], []
    started = utc()
    signal.signal(signal.SIGALRM, timeout)
    for replicate in range(CONDITIONS["replicates"]):
        for batch_index, batch in enumerate(BATCHES):
            cell_index = replicate * len(BATCHES) + batch_index
            order = bench._counterbalanced(routes, replicate, cell_index)
            for route in order:
                label = f"r{replicate}-b{batch}-{route}"
                receipt = dict(
                    label=label,
                    route=route,
                    batch=batch,
                    replicate=replicate,
                    cell_index=cell_index,
                    started_utc=utc(),
                    status="running",
                )
                arguments = dict(
                    python_bin=args.python,
                    script_path=Path(bench.__file__),
                    route=route,
                    source_root=root,
                    batch=batch,
                    size=131072,
                    num_blocks=3,
                    threads=threads,
                    samples=5,
                    warmups=2,
                    snr_threshold=5.5,
                    cuda_device=args.cuda_device,
                    seed=7101 + 10000 * replicate + 100 * cell_index,
                    affinity=args.affinity,
                    call_surface="public",
                )
                write_json(
                    args.output / "active-worker.json",
                    dict(
                        receipt,
                        session=args.session,
                        pid=os.getpid(),
                        cwd=str(root),
                        worker_arguments={
                            k: str(v) if isinstance(v, Path) else v
                            for k, v in arguments.items()
                        },
                    ),
                )
                print(json.dumps(dict(receipt, session=args.session)), flush=True)
                try:
                    receipt["capacity_before"] = capacity(args, "cuda" in route)
                    signal.alarm(args.worker_timeout)
                    # This unchanged function launches the committed worker CLI.
                    # No measured process or production function is instrumented.
                    record = bench._run_child(**arguments)
                    signal.alarm(0)
                    record.update(
                        replicate=replicate,
                        cell_index=cell_index,
                        execution_order=order,
                    )
                    record_path = destination / f"{label}.json"
                    write_json(record_path, record)
                    receipt.update(
                        status="completed",
                        raw_file=str(record_path.relative_to(args.output)),
                        raw_sha256=sha256(record_path),
                        worker_pid=record["pid"],
                    )
                    records.append(record)
                except Exception as exc:
                    signal.alarm(0)
                    receipt.update(
                        status="failed",
                        error_type=type(exc).__name__,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                    failures.append(receipt)
                receipt["finished_utc"] = utc()
                receipts.append(receipt)
                write_json(destination / f"{label}.receipt.json", receipt)
                write_json(destination / "workers.json", receipts)
                print(
                    json.dumps(
                        {k: receipt[k] for k in ("label", "status", "finished_utc")}
                    ),
                    flush=True,
                )

    parity = {"all_passed_globally": not failures}
    for batch in BATCHES:
        replicates = {}
        for replicate in range(3):
            subset = {
                r["route"]: r
                for r in records
                if r["batch"] == batch and r["replicate"] == replicate
            }
            if "branch_standard" in subset:
                checked = bench._verify_parity({batch: subset})[f"batch_{batch}"]
                if set(subset) != set(routes):
                    checked["all_passed"] = False
                    checked["missing_routes"] = sorted(set(routes) - set(subset))
            else:
                checked = dict(
                    batch=batch,
                    comparisons={},
                    all_passed=False,
                    error="Standard CPU control is missing",
                )
            replicates[f"replicate_{replicate}"] = checked
        passed = all(r["all_passed"] for r in replicates.values())
        parity[f"batch_{batch}"] = dict(
            batch=batch, replicates=replicates, all_passed=passed
        )
        parity["all_passed_globally"] = parity["all_passed_globally"] and passed

    final_identity = bench.source_identity(root)
    assert final_identity["revision"] == expected and final_identity["dirty"] is False
    summaries = {}
    for batch in BATCHES:
        summaries[f"batch_{batch}"] = {}
        for route in routes:
            workers = [
                r for r in records if r["batch"] == batch and r["route"] == route
            ]
            if len(workers) == 3:
                medians = [
                    statistics.median(
                        batch * 3 * 1000 / latency
                        for latency in r["warm_iteration_latencies_ms"]
                    )
                    for r in workers
                ]
                summaries[f"batch_{batch}"][route] = dict(
                    replicate_estimands=dict(
                        throughput_wps=dict(median=statistics.median(medians))
                    )
                )
    payload = dict(
        schema_version=1,
        artifact_type="external_production_live_batch",
        started_utc=started,
        finished_utc=utc(),
        host=platform.node(),
        runtime=bench.runtime_metadata(),
        root=str(root),
        source_identities={r: identity for r in routes},
        source_unchanged_after=True,
        support_sha256=sha256(Path(__file__)),
        launcher_sha256=sha256(Path(bench.__file__)),
        args=dict(
            CONDITIONS,
            threads=threads,
            affinity=args.affinity,
            cuda_device=args.cuda_device,
        ),
        measurement=dict(
            scope="public_library_api",
            call_surface="public",
            acquisition="unchanged committed child CLI in fresh processes",
        ),
        reference_route="branch_standard",
        routes=routes,
        records=records,
        failures=failures,
        worker_receipts=receipts,
        expected_workers=len(routes) * len(BATCHES) * 3,
        parity_analysis=parity,
        summary_by_batch=summaries,
    )
    bench.atomic_write_json(
        args.output / f"{args.session}.json", bench.seal_artifact(payload)
    )
    return 0 if parity["all_passed_globally"] else 1


def main():
    global ACTIVE
    args = parse_args()
    os.environ.update(
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        OMP_DYNAMIC="FALSE",
    )
    os.environ.pop("PYTHONPATH", None)
    # Preserve route defaults and the original benchmark's explicit gates.
    for name in list(os.environ):
        if name.startswith("PYCBC_TORCH_") or name in (
            "PYCBC_ENABLE_CUDA_GRAPHS",
            "PYCBC_BATCH_MAXELEMENTS",
        ):
            os.environ.pop(name)
    if args.session:
        return session(args)
    args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    write_json(
        args.output / "manifest.json",
        dict(
            started_utc=utc(),
            host=platform.node(),
            controller_pid=os.getpid(),
            cwd=os.getcwd(),
            command=sys.argv,
            support_sha256=sha256(Path(__file__)),
            expected_groups=84,
            expected_workers=252,
            batches=list(BATCHES),
            heads=dict(main=args.expected_main, cpu=args.expected_cpu),
            roots=dict(main=str(args.main_root), cpu=str(args.cpu_root)),
            stop_command=f"kill -TERM {os.getpid()}",
        ),
    )
    thread = threading.Thread(target=monitor, args=(args, args.output), daemon=True)
    thread.start()
    results = []
    try:
        for label in ("main-t1", "cpu-t1", "main-t4", "cpu-t4"):
            command = [
                args.python,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--session",
                label,
            ]
            result = dict(label=label, command=command, started_utc=utc())
            with (args.output / f"{label}.log").open("x") as handle:
                ACTIVE = subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                result["pid"] = ACTIVE.pid
                write_json(args.output / "active.json", result)
                print(json.dumps(result), flush=True)
                try:
                    result["returncode"] = ACTIVE.wait(timeout=args.session_timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(ACTIVE.pid, signal.SIGTERM)
                    try:
                        ACTIVE.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(ACTIVE.pid, signal.SIGKILL)
                        ACTIVE.wait()
                    result.update(returncode=ACTIVE.returncode, timeout=True)
            result["finished_utc"] = utc()
            results.append(result)
            write_json(args.output / "runs.json", results)
            print(json.dumps(result), flush=True)
    finally:
        STOP.set()
        thread.join(timeout=10)
    passed = len(results) == 4 and all(r["returncode"] == 0 for r in results)
    write_json(
        args.output / "status.json",
        dict(finished_utc=utc(), finished=True, passed=passed, sessions=len(results)),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
