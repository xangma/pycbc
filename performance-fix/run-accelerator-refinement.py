"""Queue isolated v2 qualification and 36 fresh GPU workers after postchecks.

--plan-only is read-only and imports no checkout. Actual execution creates only
three fresh v2 clones and accelerator-refinement outputs; existing campaign
sources and evidence are guarded by hashes and never written by this runner.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET


ROOT = Path("/home/xangma/pycbc-torch-performance-fix-20260906")
PYTHON = "/home/xangma/pycbc-torch-split-20260905/venv/bin/python"
BASELINE = "dfd42bf76766cadca0eecf609a1eaeac73534676"
MANIFEST_SHA256 = "1a2f55ae5075402c987d45c85498816ba15331eeb5ba76e9ce3555ecd797fc51"
BUNDLE_SHA256 = "6132363ae474aa3e291cad7f76775efbc0b4cf9b3a9ffd5f44a796da981413e0"
SOURCES = {
    "candidate-v2": (
        "candidate",
        "97bf1614f3afe53a6e2edb4d8c7dff79e9661782",
        "0d00581251e642a5d6b56b2497a9adad93069e6b",
        "fa7a7c09df6d93133324f762d756842de172433d",
    ),
    "fft-candidate-v2": (
        "fft-candidate",
        "d6407d32742a57e4f026c461a26ef7b3929c5849",
        "c3202b4b0b681a1ace527d1e6231c47470856ba6",
        "e5f773b444186f267d4825f59b3d2c50cee31010",
    ),
    "cpu-candidate-v2": (
        "cpu-candidate",
        "1514327669fc7be125523b991c847868c3a2a17e",
        "665fa5a0f41946c0c2873d8a79a16c8b6975a090",
        "225b9b0f29832057511312db81b73ec6bae52195",
    ),
}
MAIN = [
    "test/test_torch_batch_overlap_scaling.py",
    "test/test_live_batch_torch_peaks.py",
    "test/test_torch_cuda_native_batch.py",
    "test/test_torch_cpu_native_batch.py",
    "test/test_torch_cuda_native_peaks.py",
    "test/test_torch_matchedfilter_cpu_optimization.py",
    "test/test_live_batch_torch_fft_integration.py",
    "test/test_torch_large_batches.py",
    "test/test_torch_filter_pipeline.py",
    "test/test_torch_peak_contracts.py",
    "test/test_torch_batched_fft.py",
    "test/test_torch_fft_cpu_native.py",
    "test/waveform/test_taylorf2_phase_evaluation.py",
    "test/waveform/test_taylorf2_torch.py",
    "test/waveform/test_taylorf2_batch.py",
    "test/waveform/test_spa_tmplt_sequence_torch.py",
    "test/test_spatmplt_torch.py",
    "test/test_torch_waveform_generator.py",
]
FFT = MAIN + [
    "test/test_fft_batched_backends.py",
    "test/test_fft_cli_wisdom.py",
    "test/test_fftw_wisdom_cache.py",
    "test/test_torch_fft_writes.py",
    "test/test_torch_fft_cuda_workspace.py",
]
CPU = MAIN + [
    "test/test_cpu_batch_peaks.py",
    "test/test_torch_cpu_native_peaks.py",
    "test/test_torch_cpu_fft_tuning.py",
    "test/test_torch_chisq_cpu_optimization.py",
    "test/test_torch_chisq_sparse_dispatch.py",
    "test/test_chisq_torch.py",
    "test/test_torch_performance_artifacts.py",
]
NATIVE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".pyx",
    ".pxd",
    ".cu",
    ".f",
    ".f90",
}
BUILD_FILES = {"setup.py", "setup.cfg", "pyproject.toml", "MANIFEST.in"}
HARNESS_FILES = ("tools/bench_production_live_batch.py", "tools/benchmark_artifact.py")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value, *, replace=False):
    require(replace or not path.exists(), f"Refusing existing artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def compact(snapshot):
    return {key: snapshot[key] for key in ("root", "sha", "tree", "status")}


def snapshot(root, expected):
    sha = git(root, "rev-parse", "HEAD")
    dirty = git(root, "status", "--porcelain", "--untracked-files=all")
    require(
        sha == expected and not dirty,
        f"Source identity/purity failure: {root}: {sha}: {dirty}",
    )
    entries = subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "-rz", "HEAD"]
    )
    tracked = {}
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        info, name = entry.split(b"\t", 1)
        mode, kind, blob = info.decode().split()
        relative = os.fsdecode(name)
        path = root / relative
        require(
            kind == "blob" and mode in {"100644", "100755", "120000"},
            f"Unsupported tracked entry: {root}: {relative}: {mode} {kind}",
        )
        if mode == "120000":
            require(path.is_symlink(), f"Expected tracked symlink: {path}")
            data = os.fsencode(os.readlink(path))
        else:
            require(
                path.is_file() and not path.is_symlink(),
                f"Expected tracked file: {path}",
            )
            require(
                bool(path.stat().st_mode & 0o111) == (mode == "100755"),
                f"Tracked executable mode differs: {path}",
            )
            data = path.read_bytes()
        actual_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        require(
            actual_blob == blob, f"Tracked bytes differ from committed source: {path}"
        )
        tracked[relative] = {
            "mode": mode,
            "kind": kind,
            "git_blob": blob,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    binaries = {}
    for path in sorted((root / "pycbc").rglob("*.so")):
        require(
            path.is_file() and not path.is_symlink(), f"Unsafe binary entry: {path}"
        )
        binaries[str(path.relative_to(root))] = {
            "sha256": digest(path),
            "mode": stat.S_IMODE(path.stat().st_mode),
            "bytes": path.stat().st_size,
        }
    return {
        "root": str(root),
        "sha": sha,
        "tree": git(root, "rev-parse", "HEAD^{tree}"),
        "status": dirty,
        "tracked_files": tracked,
        "native_binaries": binaries,
    }


def cells():
    return [
        (route, batch, rep)
        for batch in (1, 8, 32, 128, 512, 1024)
        for rep in (1, 2, 3)
        for route in ("torch_cuda", "torch_cuda_native")
    ]


def label(route, batch, rep, source=None):
    value = f"{route}-b{batch}-t1-r{rep}"
    return value if source is None else value + "-" + source


def worker_command(args, route, batch):
    root = args.root / "candidate-v2"
    return [
        "taskset",
        "-c",
        "8-11",
        args.python,
        str(root / HARNESS_FILES[0]),
        "child",
        "--route",
        route,
        "--source-root",
        str(root),
        "--batch",
        str(batch),
        "--size",
        "131072",
        "--num-blocks",
        "3",
        "--threads",
        "1",
        "--samples",
        "3",
        "--warmups",
        "1",
        "--snr-threshold",
        "5.5",
        "--cuda-device",
        "0",
        "--seed",
        "7101",
        "--call-surface",
        "public",
    ]


def test_steps(args):
    steps = []
    for source, files in (
        ("candidate-v2", MAIN),
        ("fft-candidate-v2", FFT),
        ("cpu-candidate-v2", CPU),
    ):
        name = "test-" + source
        command = [
            "taskset",
            "-c",
            "8-11",
            args.python,
            "-m",
            "pytest",
            "-q",
            "-ra",
            "-p",
            "no:cacheprovider",
            "--junitxml",
            str(args.output / "tests" / (name + ".xml")),
            "--basetemp",
            str(args.output / "tests" / (name + "-temp")),
            *files,
        ]
        steps.append(
            {"label": name, "source": source, "files": files, "command": command}
        )
    return steps


def plan(args):
    setup = []
    for name, (old, old_sha, sha, tree) in SOURCES.items():
        root = args.root / name
        setup.append(
            {
                "source": name,
                "sha": sha,
                "tree": tree,
                "binary_source": old,
                "binary_source_sha": old_sha,
                "commands": [
                    [
                        "git",
                        "clone",
                        "--no-hardlinks",
                        "--no-checkout",
                        str(args.root / "baseline"),
                        str(root),
                    ],
                    [
                        "git",
                        "-C",
                        str(root),
                        "fetch",
                        str(args.root / "dependent-sources-v2.bundle"),
                        "+refs/heads/*:refs/remotes/bundle/*",
                    ],
                    ["git", "-C", str(root), "checkout", "--detach", sha],
                ],
                "copy_policy": "Verify every tracked byte/mode against its commit; require only matchedfilter.py "
                "to differ from v1 and all native input bytes/modes identical before copying 11 verified .so files.",
            }
        )
    return {
        "wait_for": str(args.root / "postchecks/postcheck-status.json"),
        "required_state": "complete",
        "output": str(args.output),
        "setup": setup,
        "tests": test_steps(args),
        "workers": [
            {
                "label": label(route, batch, rep, "candidate-v2"),
                "route": route,
                "batch": batch,
                "replicate": rep,
                "command": worker_command(args, route, batch),
            }
            for route, batch, rep in cells()
        ],
        "new_worker_count": 36,
        "test_file_counts": [18, 23, 25],
        "controls": "Reuse immutable baseline, candidate (v1), and branch_standard records matched by route/batch/replicate; "
        "separate later cohort, no new CPU or waveform measurements.",
    }


def load_harness(root):
    sys.path.insert(0, str(root / "tools"))
    spec = importlib.util.spec_from_file_location(
        "refinement_live_harness", root / HARNESS_FILES[0]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(args):
    """Thin process wrapper; the existing benchmark helper remains unmodified."""
    harness = load_harness(args.root / "baseline")
    root = args.root / "candidate-v2"
    started = time.monotonic()
    result = harness._run_child(
        args.python,
        root / HARNESS_FILES[0],
        args.route,
        root,
        args.batch,
        131072,
        3,
        1,
        3,
        1,
        5.5,
        0,
        7101,
        "8-11",
        "public",
    )
    result["worker_wall_seconds"] = time.monotonic() - started
    print("RESULT_JSON=" + json.dumps(result, allow_nan=False), flush=True)


def recorded_environment(environment):
    keys = {
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "OMP_DYNAMIC",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
    }
    return {k: v for k, v in environment.items() if k in keys or k.startswith("PYCBC_")}


class Runner:
    def __init__(self, args):
        self.args, self.root, self.out = args, args.root, args.output
        self.out.mkdir(parents=True, exist_ok=False)
        for name in ("commands", "logs", "tests", "live", "parity"):
            (self.out / name).mkdir()
        self.guards, self.inputs = {}, {}
        self.status = {
            "schema": "torch_accelerator_refinement_v2",
            "state": "queued",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "cwd": str(Path.cwd()),
            "command": [sys.executable, *sys.argv],
            "queue_started_utc": utc(),
            "started_utc": utc(),
            "timing_started_utc": None,
            "finished_utc": None,
            "completed": [],
            "completed_tests": [],
            "source": None,
            "source_before": None,
            "source_after": None,
            "runner_sha256": digest(Path(__file__)),
            "stop_command": f"kill -TERM {os.getpid()}",
            "expected_next_check_seconds": 30,
            "log_path": str(self.out / "runner.log"),
        }
        self.update()
        save(self.out / "plan.json", plan(args))

    def update(self):
        self.status["updated_utc"] = utc()
        save(self.out / "status.json", self.status, replace=True)

    def log(self, message):
        line = f"{utc()} {message}"
        print(line, flush=True)
        with (self.out / "runner.log").open("a") as stream:
            stream.write(line + "\n")

    def remember(self, path):
        key = str(path)
        value = digest(path)
        require(
            key not in self.inputs or self.inputs[key] == value,
            f"Frozen input changed: {path}",
        )
        self.inputs[key] = value

    def check_inputs(self):
        for path, expected in self.inputs.items():
            require(digest(Path(path)) == expected, f"Frozen input changed: {path}")

    def guard(self, name, expected=None):
        expected = expected or self.guards[name]["sha"]
        current = snapshot(self.root / name, expected)
        if name in self.guards:
            require(
                current == self.guards[name],
                f"Source bytes/modes/binaries changed: {name}",
            )
        else:
            self.guards[name] = current
        return current

    def wait(self):
        path = self.root / "postchecks/postcheck-status.json"
        self.log(
            f"Waiting for {path}; host {self.status['host']}; PID {os.getpid()}; "
            f"stop: {self.status['stop_command']}; next check in 30 seconds"
        )
        started = time.monotonic()
        while True:
            require(
                time.monotonic() - started < self.args.wait_timeout,
                "Postcheck wait timed out",
            )
            if path.is_file():
                record = json.loads(path.read_text())
                state = record.get("state")
                if state == "complete" and record.get("finished_utc"):
                    self.remember(path)
                    self.status["postcheck_status_sha256"] = self.inputs[str(path)]
                    self.status["postcheck_finished_utc"] = record["finished_utc"]
                    save(self.out / "postcheck-status-snapshot.json", record)
                    break
                require(
                    state in {"queued", "waiting", "running", "complete"},
                    f"Postchecks did not complete: {state}",
                )
            self.update()
            time.sleep(30)

    def execute(self, name, command, cwd, *, test=False):
        path = self.out / "commands" / (name + ".json")
        log = self.out / "logs" / (name + ".log")
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if test:
            environment = {
                k: v for k, v in environment.items() if not k.startswith("PYCBC_")
            }
            for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONSTARTUP"):
                environment.pop(key, None)
            environment.update(
                PYTHONPATH=str(cwd),
                OMP_DYNAMIC="FALSE",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
            )
        record = {
            "label": name,
            "command": command,
            "cwd": str(cwd),
            "host": self.status["host"],
            "environment": recorded_environment(environment),
            "log_path": str(log),
            "started_utc": utc(),
        }
        self.status.update(
            state="running",
            current=name,
            current_command=command,
            current_cwd=str(cwd),
            current_log=str(log),
        )
        self.log(f"Starting {name}: {shlex.join(command)}; cwd {cwd}; log {log}")
        started = time.monotonic()
        with log.open("x") as output:
            process = None
            try:
                # Delay interruption until the child's process-group PID is known.
                previous_mask = signal.pthread_sigmask(
                    signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT}
                )
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=cwd,
                        env=environment,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        preexec_fn=lambda: signal.pthread_sigmask(
                            signal.SIG_SETMASK, previous_mask
                        ),
                    )
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                record.update(
                    pid=process.pid, stop_command=f"kill -TERM -- -{process.pid}"
                )
                self.status.update(
                    current_pid=process.pid, current_stop_command=record["stop_command"]
                )
                save(path, record)
                self.update()
                while True:
                    try:
                        process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        require(
                            time.monotonic() - started < self.args.command_timeout,
                            f"Command timed out: {name}",
                        )
                        self.update()
            except BaseException:
                # The wrapper and its _run_child worker share this process group.
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
            finally:
                record.update(
                    pid=process.pid if process is not None else None,
                    returncode=process.poll() if process is not None else None,
                    finished_utc=utc(),
                    wall_seconds=time.monotonic() - started,
                )
                save(path, record, replace=True)
        self.status.pop("current_pid", None)
        self.status.pop("current_stop_command", None)
        self.update()
        require(
            record["returncode"] == 0,
            f"Command failed: {name}: {record['returncode']}; see {log}",
        )
        return record, path, log

    def prepare(self):
        manifest_path = self.root / "dependent-sources-v2.json"
        bundle = self.root / "dependent-sources-v2.bundle"
        require(
            digest(manifest_path) == MANIFEST_SHA256
            and digest(bundle) == BUNDLE_SHA256,
            "V2 source manifest or bundle hash differs from reviewed inputs",
        )
        manifest = json.loads(manifest_path.read_text())
        require(
            manifest["state"] == "complete"
            and manifest["bundle"]["sha256"] == BUNDLE_SHA256,
            "Incomplete or mismatched v2 source manifest",
        )
        for path in (
            Path(__file__),
            manifest_path,
            bundle,
            self.root / "run-comparison.py",
            self.root / "run-postchecks.py",
        ):
            self.remember(path)
        original_path = self.root / "comparison-status.json"
        original = json.loads(original_path.read_text())
        require(
            original["state"] == "complete" and len(original["completed"]) == 360,
            "Original 360-worker campaign is incomplete",
        )
        self.remember(original_path)
        self.status.update(
            original_status_sha256=self.inputs[str(original_path)],
            original_sources=original["sources"],
        )
        save(self.out / "original-status-snapshot.json", original)
        require(
            all(not (self.root / name).exists() for name in SOURCES),
            "A v2 clone already exists; refusing overwrite",
        )
        baseline = self.guard("baseline", BASELINE)
        require(
            compact(baseline) == original["sources"]["baseline"],
            "Original baseline identity changed",
        )
        for _, (old, old_sha, _, _) in SOURCES.items():
            self.guard(old, old_sha)
        require(
            compact(self.guards["candidate"]) == original["sources"]["candidate"],
            "Original candidate identity changed",
        )
        # Freeze every reused worker before any setup, qualification, or GPU work.
        self.controls = {}
        for route, batch, rep in cells():
            control = {}
            for key, control_route, source in (
                ("branch_standard", "branch_standard", "baseline"),
                ("baseline", route, "baseline"),
                ("candidate", route, "candidate"),
            ):
                path = (
                    self.root
                    / "comparison/live"
                    / (label(control_route, batch, rep, source) + ".json")
                )
                self.remember(path)
                record = json.loads(path.read_text())
                require(
                    record["comparison_source"] == original["sources"][source]
                    and record["comparison_label"] == path.stem
                    and record["comparison_replicate"] == rep
                    and record["route"] == control_route
                    and record["batch"] == batch
                    and record["threads"] == 1,
                    f"Reused worker identity mismatch: {path}",
                )
                control[key] = record
            self.controls[(route, batch, rep)] = control
        self.execute(
            "verify-bundle",
            ["git", "-C", str(self.root / "baseline"), "bundle", "verify", str(bundle)],
            self.root,
        )
        prepared = {}
        for step in plan(self.args)["setup"]:
            name = step["source"]
            for number, command in enumerate(step["commands"], 1):
                self.execute(f"setup-{name}-{number}", command, self.root)
            root = self.root / name
            current = snapshot(root, step["sha"])
            require(
                current["tree"] == step["tree"] and not current["native_binaries"],
                f"Unexpected fresh source: {name}",
            )
            old = self.guard(step["binary_source"])
            before, after = old["tracked_files"], current["tracked_files"]
            require(set(before) == set(after), f"V2 changed tracked path set: {name}")
            differences = {path for path in before if before[path] != after[path]}
            require(
                differences == {"pycbc/filter/matchedfilter.py"},
                f"V2 changes exceed peak fast path: {name}: {differences}",
            )
            require(
                before["pycbc/filter/matchedfilter.py"]["mode"]
                == after["pycbc/filter/matchedfilter.py"]["mode"],
                f"V2 changed runtime source mode: {name}",
            )
            declaration = (
                manifest["main_fix"]
                if name == "candidate-v2"
                else next(
                    item
                    for item in manifest["sources"]
                    if item["key"] == ("fft" if name.startswith("fft") else "cpu")
                )
            )
            for path, metadata in declaration.get(
                "files", declaration.get("fix_files", {})
            ).items():
                require(
                    after[path] == metadata,
                    f"V2 source differs from manifest: {name}: {path}",
                )
            native = {
                path: metadata
                for path, metadata in after.items()
                if Path(path).suffix.lower() in NATIVE_SUFFIXES or path in BUILD_FILES
            }
            require(
                native
                and all(before[path] == metadata for path, metadata in native.items()),
                f"Native build inputs differ: {name}",
            )
            if "native_build_inputs" in declaration:
                require(
                    native == declaration["native_build_inputs"],
                    f"Native inputs differ from manifest: {name}",
                )
            require(
                len(old["native_binaries"]) == 11,
                f"Expected 11 built native extensions: {name}",
            )
            for relative, metadata in old["native_binaries"].items():
                source, destination = (
                    self.root / step["binary_source"] / relative,
                    root / relative,
                )
                require(
                    destination.parent.is_dir() and not destination.exists(),
                    f"Unsafe binary destination: {destination}",
                )
                require(
                    digest(source) == metadata["sha256"]
                    and stat.S_IMODE(source.stat().st_mode) == metadata["mode"],
                    f"Native binary changed before copy: {source}",
                )
                shutil.copy2(source, destination)
                require(
                    digest(destination) == metadata["sha256"]
                    and stat.S_IMODE(destination.stat().st_mode) == metadata["mode"],
                    f"Native binary copy did not preserve bytes/mode: {destination}",
                )
            ready = self.guard(name, step["sha"])
            require(
                ready["native_binaries"] == old["native_binaries"],
                f"Native binary set differs after copy: {name}",
            )
            self.guard(step["binary_source"])
            prepared[name] = {
                "source": compact(ready),
                "binary_source": compact(old),
                "native_build_inputs": native,
                "native_binaries": ready["native_binaries"],
            }
            save(self.out / "setup-environment.partial.json", prepared, replace=True)
        save(self.out / "setup-environment.json", prepared)
        for name in self.guards:
            self.guard(name)
        # GPU timing sources share baseline harness bytes. The optional CPU
        # branch has its own route flag, which must stay unchanged from v1.
        for path in HARNESS_FILES:
            for name, source in self.guards.items():
                reference = (
                    self.guards["cpu-candidate"] if name.startswith("cpu-candidate")
                    else baseline
                )["tracked_files"][path]
                require(
                    source["tracked_files"][path] == reference,
                    f"Harness differs: {name}: {path}",
                )
            self.remember(self.root / "baseline" / path)
        self.status["harness_sha256"] = {
            path: digest(self.root / "baseline" / path) for path in HARNESS_FILES
        }
        self.status["source"] = compact(self.guards["candidate-v2"])
        self.status["source_before"] = self.status["source"]
        self.status["sources"] = {
            name: compact(value) for name, value in self.guards.items()
        }
        save(self.out / "sources-before.json", self.guards)
        save(self.out / "inputs-sha256.json", self.inputs)
        self.update()

    def tests(self):
        for step in test_steps(self.args):
            name, source = step["label"], step["source"]
            before = self.guard(source)
            record, path, _ = self.execute(
                name, step["command"], self.root / source, test=True
            )
            self.guard(source)
            cases = list(
                ET.parse(self.out / "tests" / (name + ".xml"))
                .getroot()
                .iter("testcase")
            )
            counts = {
                "total": len(cases),
                "skipped": sum(c.find("skipped") is not None for c in cases),
                "failures": sum(c.find("failure") is not None for c in cases),
                "errors": sum(c.find("error") is not None for c in cases),
                "selected_files": step["files"],
                "source_before": compact(before),
                "source_after": compact(self.guards[source]),
            }
            save(self.out / "tests" / (name + "-summary.json"), counts)
            require(
                counts["total"] > counts["skipped"]
                and not counts["failures"]
                and not counts["errors"],
                f"Qualification tests failed: {name}: {counts}",
            )
            record.update(
                source_before=compact(before), source_after=compact(self.guards[source])
            )
            save(path, record, replace=True)
            self.status["completed_tests"].append(name)
            self.update()

    def benchmarks(self):
        harness = load_harness(self.root / "baseline")
        self.status["timing_started_utc"] = utc()
        self.update()
        for route, batch, rep in cells():
            name = label(route, batch, rep, "candidate-v2")
            self.guard("candidate-v2")
            command = [
                self.args.python,
                str(Path(__file__).resolve()),
                "--worker",
                "--root",
                str(self.root),
                "--python",
                self.args.python,
                "--route",
                route,
                "--batch",
                str(batch),
            ]
            record, path, log = self.execute(name, command, self.root / "candidate-v2")
            self.guard("candidate-v2")
            lines = [
                line.removeprefix("RESULT_JSON=")
                for line in log.read_text().splitlines()
                if line.startswith("RESULT_JSON=")
            ]
            require(len(lines) == 1, f"Expected one unchanged worker result: {name}")
            result = json.loads(lines[0])
            expected = self.controls[(route, batch, rep)]["baseline"]
            for field in (
                "route",
                "batch",
                "threads",
                "size",
                "num_blocks",
                "seed",
                "snr_threshold",
                "cuda_device",
                "state",
                "measurement",
                "routing",
                "dimensions",
                "dtypes",
                "injection_metadata",
                "python",
                "torch_version",
                "numpy_version",
                "cuda_device_name",
            ):
                require(
                    result[field] == expected[field],
                    f"Worker workload/environment changed: {name}: {field}",
                )
            require(
                result["source_root"] == str(self.root / "candidate-v2")
                and result["command"] == worker_command(self.args, route, batch),
                f"Worker source/command differs: {name}",
            )
            samples = result["throughput_wps_summary"]["samples"]
            require(
                len(samples) == 3
                and all(math.isfinite(value) and value > 0 for value in samples),
                f"Invalid timing samples: {name}",
            )
            result.update(
                comparison_source=self.status["source"],
                comparison_label=name,
                comparison_variant="candidate-v2",
                comparison_replicate=rep,
            )
            save(self.out / "live" / (name + ".json"), result)
            record.update(
                worker_pid=result["pid"],
                worker_command=result["command"],
                source_before=self.status["source"],
                source_after=compact(self.guards["candidate-v2"]),
            )
            save(path, record, replace=True)
            pair = self.controls[(route, batch, rep)] | {"candidate-v2": result}
            parity = harness._verify_parity({batch: pair})
            save(self.out / "parity" / (label(route, batch, rep) + ".json"), parity)
            require(parity["all_passed_globally"], f"V2 parity failed: {name}")
            self.status["completed"].append(name)
            self.update()
            self.log(
                f"Completed {name}; worker PID {result['pid']}; median {result['throughput_wps_summary']['median']:.6g} waveforms/s"
            )

    def run(self):
        try:
            self.wait()
            self.prepare()
            self.tests()
            self.benchmarks()
            require(
                len(self.status["completed"]) == 36
                and len(self.status["completed_tests"]) == 3,
                "Incomplete v2 qualification matrix",
            )
            self.check_inputs()
            for name in self.guards:
                self.guard(name)
            save(self.out / "sources-after.json", self.guards)
            self.status.update(
                state="complete", source_after=compact(self.guards["candidate-v2"])
            )
        except BaseException as exc:
            self.status.update(state="failed", error=repr(exc))
            (self.out / "failure.log").write_text(traceback.format_exc())
            actual = {}
            for name, expected in self.guards.items():
                try:
                    actual[name] = snapshot(self.root / name, expected["sha"])
                except BaseException as problem:
                    actual[name] = {"error": repr(problem)}
            if not (self.out / "sources-after.json").exists():
                save(self.out / "sources-after.json", actual)
            raise
        finally:
            self.status["finished_utc"] = utc()
            self.update()
            self.log(
                f"Refinement {self.status['state']}; status: {self.out / 'status.json'}"
            )


def interrupted(number, _frame):
    raise InterruptedError(f"Received signal {number}; stopping active process group")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--wait-timeout", type=int, default=86400)
    parser.add_argument("--command-timeout", type=int, default=7200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--route", choices=("torch_cuda", "torch_cuda_native"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--batch", type=int, choices=(1, 8, 32, 128, 512, 1024), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.output = (args.output or args.root / "accelerator-refinement").resolve()
    if args.plan_only:
        print(json.dumps(plan(args), indent=2))
        return
    if args.worker:
        require(
            args.route is not None and args.batch is not None,
            "Internal worker requires route and batch",
        )
        worker(args)
        return
    require(
        args.output == args.root / "accelerator-refinement",
        "Use the separate accelerator-refinement output directory",
    )
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Runner(args).run()


if __name__ == "__main__":
    main()
