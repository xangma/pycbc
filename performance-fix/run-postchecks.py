"""Serial post-campaign profiles, regression tests, and paired CPU checks.

Run with the campaign interpreter after reviewing ``--plan-only``. The default
run waits for comparison-status.json, invokes setup-dependent.py exactly once,
and refuses existing output. --skip-setup requires already prepared clones.
All output is outside the source checkouts; no build or installation is done.
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
import signal
import socket
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET


R = Path("/home/xangma/pycbc-torch-performance-fix-20260906")
P = "/home/xangma/pycbc-torch-split-20260905/venv/bin/python"
EXPECTED = {
    "candidate": "97bf1614f3afe53a6e2edb4d8c7dff79e9661782",
    "fft-candidate": "d6407d32742a57e4f026c461a26ef7b3929c5849",
    "cpu-candidate": "1514327669fc7be125523b991c847868c3a2a17e",
    "cpu-baseline": "bd53914be6d2e4324cc867d52b3842b77cc6729a",
}
MAIN_EXTRA = [
    "test/waveform/test_taylorf2ecc_torch.py",
    "test/waveform/test_taylorf2redspin_torch.py",
    "test/waveform/test_taylorf2nltides_torch.py",
    "test/test_nltides_torch.py",
    "test/test_torch_gaussian_noise.py",
    "test/test_torch_relative_binning.py",
    "test/test_torch_marginalized_gaussian.py",
    "test/test_torch_inference_core.py",
    "test/test_torch_inference_tools.py",
    "test/test_torch_inference_cli.py",
]
MAIN_AFFECTED = [
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
FFT_EXTRA = [
    "test/test_fft_batched_backends.py",
    "test/test_fft_cli_wisdom.py",
    "test/test_fftw_wisdom_cache.py",
    "test/test_torch_fft_writes.py",
    "test/test_torch_fft_cuda_workspace.py",
]
CPU_EXTRA = [
    "test/test_cpu_batch_peaks.py",
    "test/test_torch_cpu_native_peaks.py",
    "test/test_torch_cpu_fft_tuning.py",
    "test/test_torch_chisq_cpu_optimization.py",
    "test/test_torch_chisq_sparse_dispatch.py",
    "test/test_chisq_torch.py",
    "test/test_torch_performance_artifacts.py",
]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def snapshot(root, expected):
    sha = git(root, "rev-parse", "HEAD")
    status = git(root, "status", "--porcelain", "--untracked-files=all")
    if sha != expected or status:
        raise RuntimeError(f"Source identity/purity failure: {root}: {sha} {status}")
    paths = git(root, "ls-files", "--", "pycbc", "test", "tools").splitlines()
    hashes = {name: digest(root / name) for name in paths if (root / name).is_file()}
    binaries = {
        str(path.relative_to(root)): digest(path)
        for path in sorted((root / "pycbc").rglob("*.so"))
    }
    return dict(
        root=str(root),
        sha=sha,
        tree=git(root, "rev-parse", "HEAD^{tree}"),
        status=status,
        tracked_sha256=hashes,
        native_binary_sha256=binaries,
    )


def base_environment(root, threads):
    # Keep login/runtime settings but remove inherited PyCBC overrides. Each
    # benchmark's own route_environment supplies its exact feature flags.
    environment = {k: v for k, v in os.environ.items() if not k.startswith("PYCBC_")}
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONSTARTUP"):
        environment.pop(key, None)
    environment.update(
        PYTHONPATH=str(root),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_DYNAMIC="FALSE",
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    return environment


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


def load_harness(root, name):
    # The harness and benchmark_artifact helper use only the standard library
    # at module scope. They are loaded only after campaign completion.
    sys.path.insert(0, str(root / "tools"))
    path = root / "tools/bench_production_live_batch.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_plan(args):
    root, out = args.root, args.output or args.root / "postchecks"
    prefix = ["taskset", "-c", "8-11", args.python]
    steps = []
    needs_dependents = not (args.skip_tests and args.skip_cpu_benchmark)
    if needs_dependents and not args.skip_setup:
        steps.append(
            dict(
                label="setup-dependent",
                stage="setup",
                source=None,
                command=[args.python, str(root / "setup-dependent.py")],
            )
        )
    if not args.skip_profiles:
        for route in [
            "torch_cpu",
            "torch_cpu_native",
            "torch_cuda",
            "torch_cuda_native",
        ]:
            for batch in [32, 1024]:
                label = f"profile-live-{route}-b{batch}-t1"
                steps.append(
                    dict(
                        label=label,
                        stage="profile-live",
                        source="candidate",
                        route=route,
                        batch=batch,
                        threads=1,
                        command=prefix
                        + [
                            str(root / "profile-live.py"),
                            "--root",
                            str(root / "candidate"),
                            "--out",
                            str(out / "profiles" / label),
                            "--route",
                            route,
                            "--batch",
                            str(batch),
                            "--threads",
                            "1",
                        ],
                    )
                )
        for device, batch in [("cpu", 32), ("cpu", 1024), ("cuda", 1), ("cuda", 1024)]:
            label = f"profile-waveform-{device}-b{batch}-t1"
            steps.append(
                dict(
                    label=label,
                    stage="profile-waveform",
                    source="candidate",
                    device=device,
                    batch=batch,
                    threads=1,
                    command=prefix
                    + [
                        str(root / "profile-waveform.py"),
                        "--root",
                        str(root / "candidate"),
                        "--out",
                        str(out / "profiles" / label),
                        "--device",
                        device,
                        "--batch",
                        str(batch),
                        "--threads",
                        "1",
                    ],
                )
            )
    if not args.skip_tests:
        for source, files in [
            ("candidate", MAIN_EXTRA),
            ("fft-candidate", MAIN_AFFECTED + FFT_EXTRA),
            ("cpu-candidate", MAIN_AFFECTED + CPU_EXTRA),
        ]:
            label = f"test-{source}"
            steps.append(
                dict(
                    label=label,
                    stage="tests",
                    source=source,
                    threads=1,
                    files=files,
                    command=prefix
                    + [
                        "-m",
                        "pytest",
                        "-q",
                        "-ra",
                        "-p",
                        "no:cacheprovider",
                        "--junitxml",
                        str(out / "tests" / f"{label}.xml"),
                        "--basetemp",
                        str(out / "tests" / f"{label}-temp"),
                        *files,
                    ],
                )
            )
    if not args.skip_cpu_benchmark:
        for batch in [32, 1024]:
            for threads in [1, 4]:
                for rep in [1, 2, 3]:
                    sources = ["cpu-baseline", "cpu-candidate"]
                    if rep % 2 == 0:
                        sources.reverse()
                    for source, route in [("cpu-baseline", "branch_standard")] + [
                        (name, "torch_cpu_native") for name in sources
                    ]:
                        label = f"cpu-{route}-b{batch}-t{threads}-r{rep}-{source}"
                        steps.append(
                            dict(
                                label=label,
                                stage="cpu-benchmark",
                                source=source,
                                route=route,
                                batch=batch,
                                threads=threads,
                                replicate=rep,
                                command=prefix
                                + [
                                    str(
                                        root
                                        / source
                                        / "tools/bench_production_live_batch.py"
                                    ),
                                    "child",
                                    "--route",
                                    route,
                                    "--source-root",
                                    str(root / source),
                                    "--batch",
                                    str(batch),
                                    "--size",
                                    "131072",
                                    "--num-blocks",
                                    "3",
                                    "--threads",
                                    str(threads),
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
                                ],
                            )
                        )
    return dict(
        root=str(root),
        output=str(out),
        python=args.python,
        affinity="8-11",
        wait_for=str(root / "comparison-status.json"),
        expected_sources=EXPECTED,
        skipped_stages=[
            name
            for name, skip in [
                ("profiles", args.skip_profiles),
                ("tests", args.skip_tests),
                ("cpu-benchmark", args.skip_cpu_benchmark),
                ("setup", args.skip_setup),
            ]
            if skip
        ],
        test_rationale="Main adds TaylorF2 descendants and public inference/gradient coverage; "
        "dependent branches repeat the 18 affected tests and add their feature regressions.",
        benchmark_scope="Public process_data, 3 blocks, N=131072, 1 cold + 1 warmup + "
        "3 timed iterations; 3 fresh replicates, standard control for every pair. "
        "Public trigger/norm parity is distinct from pointwise waveform parity in tests.",
        steps=steps,
    )


class Runner:
    def __init__(self, args, plan):
        self.args, self.plan = args, plan
        self.root, self.out = args.root, Path(plan["output"])
        self.out.mkdir(parents=True, exist_ok=False)
        self.sources, self.harnesses, self.pairs, self.worker_pids = {}, {}, {}, set()
        self.status = dict(
            state="waiting",
            pid=os.getpid(),
            host=socket.gethostname(),
            cwd=str(Path.cwd()),
            command=[sys.executable, *sys.argv],
            started_utc=utc(),
            runner_log=str(self.out / "runner.log"),
            stop_command=f"kill -TERM {os.getpid()}",
            completed=[],
            skipped_stages=plan["skipped_stages"],
        )
        save(self.out / "plan.json", plan)
        self.update()

    def update(self):
        self.status["updated_utc"] = utc()
        self.status["expected_next_check_utc"] = (
            dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30)
        ).isoformat()
        save(self.out / "postcheck-status.json", self.status)

    def log(self, message):
        line = f"{utc()} {message}"
        print(line, flush=True)
        with (self.out / "runner.log").open("a") as output:
            output.write(line + "\n")

    def wait_for_campaign(self):
        path = self.root / "comparison-status.json"
        self.log(
            f"Waiting for {path}; PID {os.getpid()}; stop: kill -TERM {os.getpid()}"
        )
        while True:
            if path.exists():
                campaign = json.loads(path.read_text())
                state = campaign["state"]
                if state == "complete":
                    if campaign["sources"]["candidate"]["sha"] != EXPECTED["candidate"]:
                        raise RuntimeError(
                            "Completed campaign has unexpected candidate SHA"
                        )
                    save(self.out / "comparison-complete.json", campaign)
                    return
                if state != "running":
                    raise RuntimeError(f"Campaign did not complete: {state}")
            self.update()
            time.sleep(30)

    def check_source(self, source):
        current = snapshot(self.root / source, EXPECTED[source])
        if source not in self.sources:
            self.sources[source] = current
            save(self.out / "sources-before.json", self.sources)
        elif current != self.sources[source]:
            save(
                self.out / "source-purity-failure.json",
                dict(source=source, actual=current),
            )
            raise RuntimeError(f"Source or native binaries changed: {source}")
        return current

    def harness(self, source):
        if source not in self.harnesses:
            if self.harnesses:
                first = next(iter(self.harnesses))
                helper = "tools/benchmark_artifact.py"
                if digest(self.root / source / helper) != digest(
                    self.root / first / helper
                ):
                    raise RuntimeError("Harness helper differs across source checkouts")
            self.harnesses[source] = load_harness(
                self.root / source, "postcheck_" + source.replace("-", "_")
            )
        return self.harnesses[source]

    def execute(self, step):
        source, stage, label = step["source"], step["stage"], step["label"]
        cwd = self.root / source if source else self.root
        if source:
            self.check_source(source)
        environment = base_environment(cwd, step.get("threads", 1))
        if stage in {"profile-live", "cpu-benchmark"}:
            environment = self.harness(source).route_environment(
                step["route"], environment
            )
        elif stage == "profile-waveform":
            environment["PYCBC_TAYLORF2_TRITON"] = "0"
        if stage == "tests":
            for name in step["files"]:
                if not (cwd / name).is_file():
                    raise RuntimeError(f"Missing selected test: {cwd / name}")
            (self.out / "tests").mkdir(exist_ok=True)
        log_path = self.out / "logs" / f"{label}.log"
        log_path.parent.mkdir(exist_ok=True)
        record = dict(
            step,
            cwd=str(cwd),
            environment=recorded_environment(environment),
            source_identity=self.sources.get(source),
            started_utc=utc(),
            log_path=str(log_path),
            host=socket.gethostname(),
        )
        self.status.update(
            state="running",
            current=label,
            current_command=step["command"],
            current_log=str(log_path),
            current_cwd=str(cwd),
        )
        self.log(f"Starting {label}: {shlex.join(step['command'])}; log: {log_path}")
        start = time.monotonic()
        with log_path.open("x") as output:
            process = subprocess.Popen(
                step["command"],
                cwd=cwd,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["pid"] = process.pid
            self.status.update(
                current_pid=process.pid,
                current_stop_command=f"kill -TERM -- -{process.pid}",
            )
            self.update()
            save(self.out / "commands" / f"{label}.json", record)
            try:
                while process.poll() is None:
                    if time.monotonic() - start > self.args.command_timeout:
                        raise TimeoutError(
                            f"Command exceeded {self.args.command_timeout}s: {label}"
                        )
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        self.update()
                record["returncode"] = process.returncode
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                record.update(
                    returncode=process.returncode,
                    finished_utc=utc(),
                    wall_seconds=time.monotonic() - start,
                )
                save(self.out / "commands" / f"{label}.json", record)
        if source:
            self.check_source(source)
        if record["returncode"]:
            raise RuntimeError(
                f"{label} failed with exit {record['returncode']}; see {log_path}"
            )
        self.validate(step, log_path)
        self.status["completed"].append(label)
        self.status.pop("current_pid", None)
        self.status.pop("current_stop_command", None)
        self.update()
        self.log(f"Completed {label} in {record['wall_seconds']:.1f}s")

    def validate(self, step, log_path):
        label, stage = step["label"], step["stage"]
        if stage == "setup":
            receipt = json.loads((self.root / "dependent-environment.json").read_text())
            for source in ("fft-candidate", "cpu-candidate", "cpu-baseline"):
                current = self.check_source(source)
                if (
                    receipt[source]["sha"] != current["sha"]
                    or receipt[source]["tree"] != current["tree"]
                    or receipt[source]["root"] != current["root"]
                    or receipt[source]["native_binaries"]
                    != current["native_binary_sha256"]
                ):
                    raise RuntimeError(
                        f"Prepared source does not match setup receipt: {source}"
                    )
            save(self.out / "dependent-environment.json", receipt)
        elif stage.startswith("profile-"):
            folder = self.out / "profiles" / label
            expected = ["python.pstats", "python.txt", "operators.txt", "trace.json"]
            expected += (
                ["status.json", "worker.log"]
                if stage == "profile-live"
                else ["timing.json"]
            )
            if any(
                not (folder / name).is_file() or (folder / name).stat().st_size == 0
                for name in expected
            ):
                raise RuntimeError(f"Profile outputs incomplete: {label}")
            if stage == "profile-live":
                state = json.loads((folder / "status.json").read_text())
                if not state["completed"] or state["calls"] < 8:
                    raise RuntimeError(f"Instrumented calls not reached: {label}")
            else:
                timing = json.loads((folder / "timing.json").read_text())
                if (
                    any(
                        timing[key] != step[key]
                        for key in ("device", "batch", "threads")
                    )
                    or len(timing["samples_seconds"]) != 7
                    or not all(
                        math.isfinite(value) and value > 0
                        for value in timing["samples_seconds"]
                    )
                ):
                    raise RuntimeError(
                        f"Waveform profile metadata/timings invalid: {label}"
                    )
            save(
                folder / "files-sha256.json",
                {name: digest(folder / name) for name in expected},
            )
        elif stage == "tests":
            suites = ET.parse(self.out / "tests" / f"{label}.xml").getroot()
            cases = list(suites.iter("testcase"))
            counts = dict(
                total=len(cases),
                skipped=sum(c.find("skipped") is not None for c in cases),
                failures=sum(c.find("failure") is not None for c in cases),
                errors=sum(c.find("error") is not None for c in cases),
            )
            save(self.out / "tests" / f"{label}-summary.json", counts)
            if (
                not cases
                or counts["total"] == counts["skipped"]
                or counts["failures"]
                or counts["errors"]
            ):
                raise RuntimeError(f"Test report failed: {label}: {counts}")
        elif stage == "cpu-benchmark":
            lines = [
                line.removeprefix("RESULT_JSON=")
                for line in log_path.read_text().splitlines()
                if line.startswith("RESULT_JSON=")
            ]
            if len(lines) != 1:
                raise RuntimeError(
                    f"Expected one benchmark result: {label}, got {len(lines)}"
                )
            result = json.loads(lines[0])
            for key in ("route", "batch", "threads"):
                if result[key] != step[key]:
                    raise RuntimeError(f"Benchmark metadata mismatch: {label}: {key}")
            if result["source_root"] != str(self.root / step["source"]):
                raise RuntimeError(f"Wrong benchmark source: {label}")
            if step["route"] == "torch_cpu_native":
                flag = result["routing"]["feature_flags"][
                    "PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK"
                ]
                if flag["environment_value"] != "1" or flag["enabled"] is not True:
                    raise RuntimeError(
                        f"Optional CPU native peak gate inactive: {label}"
                    )
            if result["pid"] in self.worker_pids:
                raise RuntimeError(f"Repeated benchmark worker PID: {result['pid']}")
            self.worker_pids.add(result["pid"])
            result.update(
                comparison_source=self.sources[step["source"]],
                comparison_label=label,
                comparison_replicate=step["replicate"],
                command=step["command"],
            )
            save(self.out / "cpu-benchmark" / f"{label}.json", result)
            key = (step["batch"], step["threads"], step["replicate"])
            pair = self.pairs.setdefault(key, {})
            route_key = (
                "branch_standard"
                if step["route"] == "branch_standard"
                else step["source"]
            )
            pair[route_key] = result
            if len(pair) == 3:
                parity = self.harness("cpu-baseline")._verify_parity(
                    {step["batch"]: pair}
                )
                save(
                    self.out
                    / "cpu-benchmark"
                    / f"parity-b{key[0]}-t{key[1]}-r{key[2]}.json",
                    parity,
                )
                if not parity["all_passed_globally"]:
                    raise RuntimeError(f"Public benchmark parity failed: {key}")

    def run(self):
        try:
            self.wait_for_campaign()
            scripts = [
                "profile-live.py",
                "profile-waveform.py",
                "setup-dependent.py",
                "run-postchecks.py",
            ]
            script_hashes = {
                name: digest(self.root / name)
                for name in scripts
                if (self.root / name).is_file()
            }
            save(self.out / "scripts-sha256.json", script_hashes)
            if self.args.skip_setup and not (
                self.args.skip_tests and self.args.skip_cpu_benchmark
            ):
                self.validate(dict(label="verify-existing-setup", stage="setup"), None)
            for step in self.plan["steps"]:
                self.execute(step)
            final = {name: self.check_source(name) for name in self.sources}
            save(self.out / "sources-after.json", final)
            if not self.args.skip_cpu_benchmark and (
                len(self.pairs) != 12 or any(len(p) != 3 for p in self.pairs.values())
            ):
                raise RuntimeError("Incomplete paired CPU benchmark matrix")
            final_scripts = {name: digest(self.root / name) for name in script_hashes}
            save(self.out / "scripts-after-sha256.json", final_scripts)
            if final_scripts != script_hashes:
                raise RuntimeError(
                    "Postcheck/profile/setup scripts changed during execution"
                )
            self.status["state"] = "complete"
        except BaseException as error:
            self.status.update(
                state="failed", error=repr(error), traceback=traceback.format_exc()
            )
            # Preserve a final source snapshot even when a command fails.
            actual = {}
            for source in self.sources:
                try:
                    actual[source] = snapshot(self.root / source, EXPECTED[source])
                except BaseException as problem:
                    actual[source] = {"error": repr(problem)}
            save(self.out / "sources-after.json", actual)
            raise
        finally:
            self.status["finished_utc"] = utc()
            self.update()
            self.log(
                f"Postchecks {self.status['state']}; status: {self.out / 'postcheck-status.json'}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=R)
    parser.add_argument("--python", default=P)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--command-timeout", type=int, default=7200)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print exact commands without filesystem writes or imports from a checkout",
    )
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-cpu-benchmark", action="store_true")
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Use previously prepared, clean dependent clones",
    )
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.output:
        args.output = args.output.resolve()
    if args.command_timeout <= 0:
        parser.error("--command-timeout must be positive")
    plan = build_plan(args)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    output = Path(plan["output"])
    if output == args.root or any(
        output.is_relative_to(args.root / source) for source in [*EXPECTED, "baseline"]
    ):
        parser.error("Output must be outside the source checkouts")

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    Runner(args, plan).run()


if __name__ == "__main__":
    main()
