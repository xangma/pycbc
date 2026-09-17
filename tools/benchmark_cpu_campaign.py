#!/usr/bin/env python3
r"""Controlled, observational CPU campaigns (Linux, Python standard library).

The command file is either an argv array or {"argv": [...], "metadata": {...}}.
Metadata is retained verbatim for source/build provenance; it is user supplied,
not independently verified. Use absolute input paths: each worker runs in a new
directory. Literal {output_dir} substrings are replaced with that directory,
without a shell or any other formatting. Work is declared by --templates and
--valid-seconds, per worker. Example:

  python3 tools/benchmark_cpu_campaign.py --command-json command.json \
      --output campaign --workers physical --repeats 3 \
      --valid-seconds 240 --templates 1000

Full wall time runs from a common barrier release through executable exit,
including executable startup and output. Preflight observations cannot reserve
a host or prove exclusivity. No services or unrelated processes are stopped.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import select
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

THREAD_LIMITS = dict.fromkeys(
    (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OPENBLAS_DEFAULT_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "TBB_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "POLARS_MAX_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ),
    "1",
)
THREAD_LIMITS.update(MKL_DYNAMIC="FALSE", OMP_DYNAMIC="FALSE")
IDLE_LIMIT = 5.0


def cpu_list(value):
    """Parse Linux CPU lists such as '0-3,8,10-11'."""
    result = set()
    for part in value.strip().split(","):
        if not part:
            continue
        bounds = [int(n) for n in part.split("-")]
        if len(bounds) == 1:
            bounds *= 2
        if len(bounds) != 2 or bounds[0] < 0 or bounds[1] < bounds[0]:
            raise ValueError("Invalid CPU list: " + value)
        result.update(range(bounds[0], bounds[1] + 1))
    return sorted(result)


def discover_topology(root=Path("/sys/devices/system/cpu"), allowed=None):
    """Read online CPUs, retaining disallowed SMT siblings for monitoring."""
    root = Path(root)
    allowed = set(os.sched_getaffinity(0) if allowed is None else allowed)
    online = cpu_list((root / "online").read_text())
    topology = []
    for cpu in online:
        directory = root / ("cpu%d/topology" % cpu)
        die = directory / "die_id"
        topology.append(
            {
                "cpu": cpu,
                "allowed": cpu in allowed,
                "package_id": int(
                    (directory / "physical_package_id").read_text()
                ),
                "die_id": int(die.read_text()) if die.exists() else -1,
                "core_id": int((directory / "core_id").read_text()),
                "smt_siblings": cpu_list(
                    (directory / "thread_siblings_list").read_text()
                ),
            }
        )
    if not allowed or not allowed.issubset(online):
        raise ValueError(
            "Affinity is empty or contains CPUs absent from sysfs"
        )
    return topology


def select_cpus(topology, workers):
    """Choose exactly one allowed logical CPU per distinct physical core."""
    cores = {}
    for entry in sorted(topology, key=lambda row: row["cpu"]):
        if entry["allowed"]:
            key = tuple(entry[k] for k in ("package_id", "die_id", "core_id"))
            if key[0] < 0 or key[2] < 0:
                raise ValueError("Physical core identity unavailable")
            cores.setdefault(key, entry["cpu"])
    count = len(cores) if workers == "physical" else int(workers)
    if not 1 <= count <= len(cores):
        raise ValueError(
            "Workers must be between 1 and %d physical cores" % len(cores)
        )
    chosen = list(cores.values())[:count]
    # Fail closed if sysfs core identities disagree with sibling masks.
    for entry in topology:
        if entry["cpu"] in chosen:
            if (
                entry["cpu"] not in entry["smt_siblings"]
                or len(set(chosen) & set(entry["smt_siblings"])) != 1
            ):
                raise ValueError("Inconsistent physical core/SMT topology")
    return chosen


def snapshot(proc=Path("/proc")):
    """Capture counters and process summaries, excluding command arguments."""
    proc = Path(proc)
    ticks = {}
    for line in (proc / "stat").read_text().splitlines():
        fields = line.split()
        if fields and fields[0].startswith("cpu") and fields[0][3:].isdigit():
            values = [int(n) for n in fields[1:9]]
            if len(values) < 5:
                raise ValueError("Incomplete /proc/stat CPU counters")
            # guest/guest_nice are already included in user/nice.
            ticks[fields[0][3:]] = {
                "total": sum(values),
                "idle": values[3] + values[4],
            }
    if not ticks:
        raise ValueError("No per-CPU counters in /proc/stat")
    memory = {}
    for line in (proc / "meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        memory[key] = value.strip()
    processes = []
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat = (directory / "stat").read_text()
            left, right = stat.index("("), stat.rindex(")")
            fields = stat[right + 2:].split()
            processes.append(
                {
                    "pid": int(directory.name),
                    "name": stat[left + 1:right],
                    "state": fields[0],
                    "ppid": int(fields[1]),
                    "process_group": int(fields[2]),
                    "cpu_ticks": int(fields[11]) + int(fields[12]),
                    "threads": int(fields[17]),
                    "cpu": int(fields[36]),
                }
            )
        except (OSError, ValueError, IndexError):
            # A process can exit or become unreadable during traversal.
            continue
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_seconds": time.monotonic(),
        "cpu_ticks": ticks,
        "memory": memory,
        "loadavg": (proc / "loadavg").read_text().strip(),
        "processes": sorted(processes, key=lambda row: row["pid"]),
    }


def utilization(before, after):
    """Host percentage is weighted by ticks across *all* online host CPUs."""
    previous, current = before["cpu_ticks"], after["cpu_ticks"]
    complete = previous.keys() == current.keys()
    total = busy = 0
    per_cpu = {}
    for cpu in current.keys() & previous.keys():
        elapsed = current[cpu]["total"] - previous[cpu]["total"]
        idle = current[cpu]["idle"] - previous[cpu]["idle"]
        if elapsed <= 0 or idle < 0 or idle > elapsed:
            complete = False
            continue
        total += elapsed
        busy += elapsed - idle
        per_cpu[cpu] = 100.0 * (elapsed - idle) / elapsed
    return {
        "complete": complete and bool(total),
        "host_percent": 100.0 * busy / total if total else None,
        "per_cpu_percent": per_cpu,
        "interval_seconds": (
            after["monotonic_seconds"] - before["monotonic_seconds"]
        ),
    }


class Sampler:
    """Persist observations immediately, even if the campaign later fails."""

    def __init__(self, path):
        self.path = Path(path)
        self.previous = None

    def take(self, phase):
        current = snapshot()
        delta = utilization(self.previous, current) if self.previous else None
        current.update(phase=phase, utilization=delta)
        with self.path.open("a") as stream:
            stream.write(json.dumps(current, sort_keys=True) + "\n")
        self.previous = current
        return current


def idle_assessment(samples, topology, selected):
    online = {str(entry["cpu"]) for entry in topology}
    monitored = {
        str(cpu)
        for entry in topology
        if entry["cpu"] in selected
        for cpu in entry["smt_siblings"]
    } & online
    reasons = []
    elapsed = sum(sample["interval_seconds"] for sample in samples)
    if elapsed < 5.0:
        reasons.append("Fewer than five seconds of preflight observations")
    for sample in samples:
        if not sample["complete"] or set(sample["per_cpu_percent"]) != online:
            reasons.append(
                "Missing/reset CPU counters or changing CPU topology"
            )
            continue
        if sample["host_percent"] > IDLE_LIMIT:
            reasons.append("Whole-host utilization exceeded 5%")
        if any(
            sample["per_cpu_percent"][cpu] > IDLE_LIMIT for cpu in monitored
        ):
            reasons.append(
                "Selected CPU or SMT sibling utilization exceeded 5%"
            )
    return {
        "passed": not reasons,
        "observed_seconds": elapsed,
        "threshold_percent": IDLE_LIMIT,
        "monitored_cpus_including_siblings": sorted(map(int, monitored)),
        "reasons": sorted(set(reasons)),
        "intervals": samples,
    }


def host_label(shared_host, reservation_note, passed):
    if shared_host:
        return "shared_host"
    if not passed:
        return "rejected_host_checks"
    return (
        "user_asserted_reservation_with_observed_idle"
        if reservation_note
        else "observational_idle"
    )


def throughput_metrics(work, makespan, physical_cores):
    """Divide concurrent work by wall time times allocated physical cores."""
    if makespan <= 0 or physical_cores < 1:
        raise ValueError("Positive wall time and physical core count required")
    return {
        "total_template_seconds": sum(work),
        "makespan_seconds": makespan,
        "allocated_physical_cores": physical_cores,
        "template_seconds_per_wall_second_per_physical_core": sum(work)
        / (makespan * physical_cores),
    }


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def worker_argv(cpu, ready_fd, gate_fd):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        str(cpu),
        str(ready_fd),
        str(gate_fd),
    ]


def worker_main(cpu, ready_fd, gate_fd):
    """Prepare affinity before joining the barrier, then replace ourselves."""
    command = json.loads(Path("command.json").read_text())
    os.sched_setaffinity(0, {cpu})
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != [cpu]:
        raise RuntimeError("Worker affinity was not applied")
    os.write(
        ready_fd,
        (
            json.dumps({"pid": os.getpid(), "affinity": affinity}) + "\n"
        ).encode(),
    )
    os.close(ready_fd)
    if os.read(gate_fd, 1) != b"G":
        raise RuntimeError("Barrier closed before release")
    os.close(gate_fd)
    os.execvpe(command[0], command, os.environ)


def stop_groups(processes):
    """Terminate our own process groups, including their descendants."""

    def send(process, sig):
        process.poll()
        for attempt in range(2):
            try:
                os.killpg(process.pid, sig)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                # Darwin returns EPERM for a group containing only an unreaped
                # zombie. Reap the leader and retry; do not hide live failures.
                if attempt or process.poll() is None:
                    raise

    groups = []
    for process in processes:
        if send(process, signal.SIGTERM):
            groups.append(process)
    if groups:
        time.sleep(0.2)
    for process in groups:
        send(process, signal.SIGKILL)
    for process in processes:
        process.wait()
    return [process.pid for process in groups]


def run_repeat(command, directory, cpus, template_seconds, timeout, sample):
    """Run identical commands concurrently; always return a failure receipt."""
    directory = Path(directory).resolve()
    directory.mkdir()
    ready_read, ready_write = os.pipe()
    gate_read, gate_write = os.pipe()
    descriptors = {ready_read, ready_write, gate_read, gate_write}
    processes, futures = [], {}
    pool = None
    started = None
    result = {"status": "failed", "workers": [], "metrics": None}
    deadline = time.monotonic() + timeout

    def wait_process(process):
        return process.wait(), time.monotonic()

    try:
        for index, cpu in enumerate(cpus):
            cwd = directory / ("worker-%03d" % index)
            cwd.mkdir()
            argv = [arg.replace("{output_dir}", str(cwd)) for arg in command]
            save_json(cwd / "command.json", argv)
            row = {
                "cpu": cpu,
                "cwd": str(cwd),
                "argv": argv,
                "stdout": str(cwd / "stdout.log"),
                "stderr": str(cwd / "stderr.log"),
                "template_seconds": template_seconds,
            }
            result["workers"].append(row)
            with open(row["stdout"], "wb") as stdout, open(
                row["stderr"], "wb"
            ) as stderr:
                process = subprocess.Popen(
                    worker_argv(cpu, ready_write, gate_read),
                    cwd=cwd,
                    env=dict(os.environ, **THREAD_LIMITS),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    pass_fds=(ready_write, gate_read),
                )
            processes.append(process)
            row["pid"] = process.pid
        for fd in (ready_write, gate_read):
            os.close(fd)
            descriptors.remove(fd)
        ready, buffer = {}, b""
        while len(ready) != len(cpus):
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out preparing workers at barrier")
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("Worker exited before barrier release")
            if select.select([ready_read], [], [], 0.05)[0]:
                chunk = os.read(ready_read, 65536)
                if not chunk:
                    raise RuntimeError("Worker readiness pipe closed")
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    message = json.loads(line)
                    ready[message["pid"]] = message["affinity"]
        for row in result["workers"]:
            row["observed_affinity"] = ready[row["pid"]]
            if row["observed_affinity"] != [row["cpu"]]:
                raise RuntimeError("Worker reported incorrect affinity")
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(cpus))
        futures = {
            pool.submit(wait_process, process): row
            for process, row in zip(processes, result["workers"])
        }
        started = time.monotonic()
        result["barrier_release_monotonic_seconds"] = started
        os.write(gate_write, b"G" * len(cpus))
        pending = set(futures)
        next_sample = started
        while pending:
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError("Worker execution exceeded timeout")
            if now >= next_sample:
                sample("during")
                next_sample = time.monotonic() + 1.0
            done, pending = concurrent.futures.wait(
                pending,
                timeout=max(0, min(next_sample, deadline) - time.monotonic()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if any(future.result()[0] != 0 for future in done):
                raise RuntimeError("Worker command failed; see worker logs")
        result["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        result["error"] = "%s: %s" % (type(error).__name__, error)
    finally:
        for fd in descriptors:
            os.close(fd)
        result["cleaned_process_groups"] = stop_groups(processes)
        if pool is not None:
            pool.shutdown(wait=True)
        if (
            result["status"] == "completed"
            and result["cleaned_process_groups"]
        ):
            result.update(
                status="failed", error="Workers left descendants running"
            )
        by_pid = {row["pid"]: row for row in result["workers"] if "pid" in row}
        for process in processes:
            by_pid[process.pid]["returncode"] = process.returncode
        for future, row in futures.items():
            code, ended = future.result()
            row.update(
                returncode=code,
                exit_monotonic_seconds=ended,
                wall_seconds=ended - started if started is not None else None,
            )
        if result["status"] == "completed":
            makespan = (
                max(row["exit_monotonic_seconds"] for row in result["workers"])
                - started
            )
            result["metrics"] = throughput_metrics(
                [row["template_seconds"] for row in result["workers"]],
                makespan,
                len(cpus),
            )
            for row in result["workers"]:
                row["template_seconds_per_wall_second_per_physical_core"] = (
                    template_seconds / row["wall_seconds"]
                )
        save_json(directory / "receipt.json", result)
    return result


def command_spec(path):
    raw = Path(path).read_bytes()
    spec = json.loads(raw)
    if isinstance(spec, list):
        spec = {"argv": spec, "metadata": {}}
    if not isinstance(spec, dict):
        raise ValueError("Command JSON must be an argv array or object")
    argv = spec.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(arg, str) and "\0" not in arg for arg in argv)
        or not argv[0]
    ):
        raise ValueError("Command argv must be a nonempty array of strings")
    executable = shutil.which(argv[0])
    if executable is None:
        raise ValueError("Executable not found: " + argv[0])
    # Preserve executable symlinks (notably virtualenv Python entry points).
    executable = os.path.abspath(executable)
    return [executable, *argv[1:]], {
        "command_file": str(Path(path).resolve()),
        "command_file_sha256": hashlib.sha256(raw).hexdigest(),
        "supplied_command_spec": spec,
        "runner_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "resolved_executable": executable,
        "provenance_verification": "Supplied metadata retained, not verified",
    }


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--command-json", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New directory for campaign and worker receipts",
    )
    parser.add_argument(
        "--workers", default="1", help="1, physical, or a positive N"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--valid-seconds", type=positive, required=True)
    parser.add_argument("--templates", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=positive, default=3600)
    parser.add_argument("--preflight-seconds", type=positive, default=5.0)
    parser.add_argument(
        "--shared-host",
        action="store_true",
        help="Explicitly permit failed idle checks; results are shared-host",
    )
    parser.add_argument(
        "--reservation-note",
        default="",
        help="User assertion of a reservation; not independently verified",
    )
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.templates < 1 or args.preflight_seconds < 5:
        parser.error(
            "Positive repeats/templates and >=5 preflight seconds required"
        )
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(str(error))
    report = {
        "schema_version": 1,
        "status": "failed",
        "repeats": [],
        "preflights": [],
        "shared_host": args.shared_host,
        "reservation_note": args.reservation_note,
        "host_label": "not_assessed",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "controller_pid": os.getpid(),
        "invocation_cwd": str(Path.cwd()),
        "thread_limits": THREAD_LIMITS,
        "samples_file": str(output / "samples.jsonl"),
        "templates_per_worker": args.templates,
        "valid_seconds_per_template": args.valid_seconds,
        "timeout_seconds": args.timeout_seconds,
        "measurement": "Barrier release to executable exit observed by waiter;"
        " includes executable startup and output",
        "host_claim": "Preflight measures observational idle only; "
        "it does not reserve CPUs or establish exclusive host use. "
        "Reservation notes are user assertions. Thread limits are set via "
        "environment; process summaries record observed threads.",
    }
    sampler = None
    old_handlers = {}

    def interrupt(signum, frame):
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt("Received signal %d" % signum)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupt)
        if sys.platform != "linux":
            raise RuntimeError(
                "This runner requires Linux sysfs, /proc, and CPU affinity"
            )
        command, report["provenance"] = command_spec(args.command_json)
        topology = discover_topology()
        cpus = select_cpus(topology, args.workers)
        report.update(
            topology=topology,
            selected_cpus=cpus,
            allocated_physical_cores=len(cpus),
        )
        sampler = Sampler(output / "samples.jsonl")
        for index in range(args.repeats):
            sampler.take("before_preflight")
            end = time.monotonic() + args.preflight_seconds
            samples = []
            while time.monotonic() < end:
                time.sleep(min(1.0, max(0, end - time.monotonic())))
                samples.append(sampler.take("preflight")["utilization"])
            assessment = idle_assessment(samples, topology, cpus)
            report["preflights"].append(assessment)
            report["host_label"] = host_label(
                args.shared_host, args.reservation_note, assessment["passed"]
            )
            save_json(output / "receipt.json", report)
            if not assessment["passed"] and not args.shared_host:
                raise RuntimeError(
                    "Strict preflight rejected host: "
                    + "; ".join(assessment["reasons"])
                )
            result = run_repeat(
                command,
                output / ("repeat-%03d" % index),
                cpus,
                args.templates * args.valid_seconds,
                args.timeout_seconds,
                sampler.take,
            )
            report["repeats"].append(result)
            sampler.take("after")
            save_json(output / "receipt.json", report)
            if result["status"] != "completed":
                raise RuntimeError(result["error"])
        report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = "%s: %s" % (type(error).__name__, error)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        save_json(output / "receipt.json", report)
    print(str(output / "receipt.json"))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_worker":
        worker_main(*map(int, sys.argv[2:]))
    else:
        sys.exit(main())
