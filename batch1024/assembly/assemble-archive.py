#!/usr/bin/env python3
"""Assemble a sealed batch-1024 supplement; never run acquisition or mutate Git."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET

BASE = "5c6c12d895fa7229fd0da2d10ade5ecfbae5153b"
REVISIONS = {
    "main": "4885b64560e9f39b740e85b6a976898869dd360e",
    "fft": "d840198592a0ca128f4f89d90987a7469ec37c8f",
    "cpu": "d544420232428225c214a4be84fbe1262a6d307b",
}
HISTORICAL_MAIN = "607bce53ead14f12af32552a5b2441d3bc667267"
HISTORICAL_FFT = "e6073eaf1a89cfed69af53707f52321eadf129f1"
FILE_LIMIT = 100_000_000  # Strictly below GitHub's 100 MB threshold.
TOTAL_LIMIT = 20 * 1024**3
MEMBER_LIMIT = 50_000
RESERVED = {"assembly", "report-engines", "documentation", "report",
            "validation", "REPRODUCE.md", "README.md", "SHA256SUMS"}
LOCAL_COPIES = {
    "live-support/live_report.py": "report-engines/live/live_report.py",
    "live-support/probe_report.py": "report-engines/live/probe_report.py",
    "live-support/run-live.py": "report-engines/live/run-live.py",
    "live-support/run-probes.py": "report-engines/live/run-probes.py",
    "live-support/live_dispatch_probe.py": "report-engines/live/live_dispatch_probe.py",
    "wave-support/waveform/report/waveform_report.py":
        "report-engines/waveform/report/waveform_report.py",
    "wave-support/waveform/harness/run.py": "report-engines/waveform/harness/run.py",
    "wave-support/waveform/harness/worker.py": "report-engines/waveform/harness/worker.py",
    "wave-support/triton/report/render.py": "report-engines/triton/report/render.py",
    "triton-original/run.py": "report-engines/triton/harness/run.py",
    "triton-original/worker.py": "report-engines/triton/harness/worker.py",
    "local-validation.json": "validation/local-validation.json",
    "ci-f401-validation.json": "validation/ci-f401-validation.json",
    "assemble-archive.py": "assembly/assemble-archive.py",
    "assemble-archive-README.md": "assembly/assemble-archive-README.md",
}
SUMMARY_PATHS = {
    "live": "batch1024/report/live-summary.json",
    "waveform": "batch1024/report/waveform-summary.json",
    "triton": "batch1024/report/triton/report.json",
    "inference": "report/inference-summary.json",
    "fft": "report/fft-summary.json",
}
FIGURES = {
    "main-live.png": ("live", "Main default routes at all six batches; separate CPU/1 thread, CPU/4 threads and CUDA/1 thread panels with different zero-based scales."),
    "waveform.png": ("waveform", "Eligible public and scalar routes at all six batches; separate CPU/1 thread, CPU/4 threads and CUDA/1 thread panels with different zero-based scales."),
    "taylorf2-throughput.png": ("triton", "All 24 route/grid/batch groups; warm public-call throughput."),
    "taylorf2-cold.png": ("triton", "All 24 route/grid/batch groups; separate cold first public calls."),
    "inference.png": ("inference", "Retained historical single-evaluation inference; all ten qualified cells."),
    "inference-cold.png": ("inference", "Retained historical inference; four separate cold intervals in each qualified cell."),
    "fft.png": ("fft", "Retained historical single-vector FFT; six plan/execution configurations."),
    "optional-cpu.png": ("live", "All six batches and both CPU thread settings, including standard control; the optional native configuration enables an additional peak kernel."),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(), object_pairs_hook=unique_object)


def write_json(path, value):
    write_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def write_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f"Refusing symlink output: {path}")
    temporary = path.with_name(path.name + ".assembly-tmp")
    with temporary.open("xb") as stream:
        stream.write(value)
    os.replace(temporary, path)


def safe_name(name):
    require(isinstance(name, str) and bool(name), "Empty/non-string member path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name, f"Noncanonical path: {name}")
    require(not any(p in (".", "..", ".git") for p in path.parts), f"Unsafe path: {name}")
    require(not any(ord(c) < 32 or c == "\\" for c in name), f"Unsafe path characters: {name!r}")
    return path


def file_record(path):
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode), f"Non-regular file: {path}")
    size = path.stat().st_size
    require(size < FILE_LIMIT, f"File must be below 100 MB: {path} ({size} bytes)")
    return {"size": size, "sha256": digest(path), "mode": stat.S_IMODE(mode)}


def inventory(root, excluded=()):
    result = {}
    for parent, directories, names in os.walk(root, followlinks=False):
        parent = Path(parent)
        for name in list(directories):
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if relative == ".git" or relative in excluded:
                directories.remove(name)
            else:
                require(not path.is_symlink(), f"Symlink directory: {path}")
        for name in names:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if relative == ".git" or relative in excluded:
                continue
            safe_name(relative)
            result[relative] = file_record(path)
    return dict(sorted(result.items()))


def git(archive, *args):
    return subprocess.check_output(["git", "-C", str(archive), *args], text=True).strip()


def check_archive(archive, baseline=None):
    require(git(archive, "rev-parse", "HEAD") == BASE, "Archive HEAD differs from the reviewed baseline")
    require(not git(archive, "diff", "--cached", "--name-only"), "Archive index must remain unchanged")
    if baseline is None:
        require(not git(archive, "status", "--porcelain", "--untracked-files=all"), "Prepare requires a clean archive checkout")
        return inventory(archive)
    require(inventory(archive, {"batch1024"}) == baseline, "An existing archive file changed")
    return baseline


def verify_tar(source, destination):
    """Manually extract only regular files after validating the complete inventory."""
    with tarfile.open(source, "r:gz") as archive:
        members = {}
        total = 0
        for member in archive:
            safe_name(member.name)
            require(member.name not in members, f"Duplicate tar member: {member.name}")
            require(member.isfile() and not member.issparse(), f"Non-regular/sparse tar member: {member.name}")
            require(0 <= member.size < FILE_LIMIT, f"Oversized tar member: {member.name}")
            require(member.mode & 0o7000 == 0, f"Special permission bits: {member.name}")
            require(PurePosixPath(member.name).parts[0] not in RESERVED, f"Reserved generated path: {member.name}")
            members[member.name] = member
            total += member.size
            require(len(members) <= MEMBER_LIMIT and total <= TOTAL_LIMIT, "Tar exceeds bounded inventory limits")
        require("sealed-manifest.json" in members, "Missing sealed-manifest.json")
        manifest = json.loads(archive.extractfile(members["sealed-manifest.json"]).read(),
                              object_pairs_hook=unique_object)
        require(isinstance(manifest, dict), "Invalid sealed inventory")
        require(set(manifest) == set(members) - {"sealed-manifest.json"}, "Tar and sealed manifest file sets differ")
        for name, record in manifest.items():
            safe_name(name)
            require(isinstance(record, dict) and set(record) == {"size", "sha256"}, f"Invalid inventory record: {name}")
            require(type(record["size"]) is int and record["size"] == members[name].size, f"Size mismatch: {name}")
            require(isinstance(record["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), f"Invalid SHA256: {name}")
        for name, member in members.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            target.chmod(member.mode & 0o777)
            require(target.stat().st_size == member.size, f"Truncated tar member: {name}")
            if name in manifest:
                require(digest(target) == manifest[name]["sha256"], f"Sealed hash mismatch: {name}")
    return manifest


def verify_seal(root):
    manifest = read_json(root / "sealed-manifest.json")
    for name, row in manifest.items():
        safe_name(name)
        actual = file_record(root / name)
        require(all(actual[key] == row[key] for key in ("size", "sha256")), f"Sealed file changed: {name}")


def completion(root):
    queue = read_json(root / "queue-status.json")
    correctness = read_json(root / "correctness/status.json")
    final = read_json(root / "finalization.json")
    require(queue.get("finished") is True and correctness.get("finished") is True,
            "Benchmark queue and correctness must both have finished")
    for name in ("triton", "waveform"):
        require(read_json(root / name / "manifest.json").get("finished_utc"), f"Incomplete {name} acquisition")
    require(final.get("finished_utc"), "Missing finalization timestamp")
    require(type(queue.get("passed")) is bool and type(correctness.get("passed")) is bool,
            "Missing final pass/fail records")
    require(final.get("queue_passed") == queue["passed"] and final.get("correctness_passed") == correctness["passed"],
            "Finalization status disagrees with queue/correctness")
    sources = final.get("sources", [])
    require(len(sources) == 3 and {s["name"] for s in sources} == set(REVISIONS), "Incomplete finalized source set")
    prepared = read_json(root / "preparation.json")
    require(len(prepared) == 3 and {s["name"] for s in prepared} == set(REVISIONS), "Incomplete preparation source set")
    prepared = {s["name"]: s for s in prepared}
    for source in sources:
        name = source["name"]
        require(source["revision"] == prepared[name]["sha"] == REVISIONS[name]
                and not source["status"] and source.get("native_unchanged") is True
                and source["native_sha256"] == prepared[name]["binaries"], f"Unqualified source seal: {name}")
    require(correctness.get("test_sha256") == digest(root / "test_torch_large_batches.py"), "Correctness test hash mismatch")
    runs = correctness.get("runs", [])
    require(len(runs) == 3 and {r["name"] for r in runs} == set(REVISIONS), "Incomplete correctness runs")
    counts = []
    fixed = read_json(root / "correctness-preparation.json")
    require(len(fixed) == 3 and {s["name"] for s in fixed} == set(REVISIONS), "Incomplete corrected source set")
    fixed = {s["name"]: s for s in fixed}
    sealed_fixed = {s["name"]: s for s in final["correctness_sources"]}
    require(len(final["correctness_sources"]) == 3 and set(sealed_fixed) == set(REVISIONS), "Incomplete corrected source seal")
    for run in runs:
        name = run["name"]
        preparation = fixed[name]
        seal = sealed_fixed[name]
        require(run["base_revision"] == preparation["base_revision"] == REVISIONS[name], "Corrected source has wrong benchmark base")
        require(run["revision"] == run["head_after"] == preparation["revision"]
                and not run["tracked_changes_after"] and run.get("finished_utc")
                and type(run.get("returncode")) is int, f"Incomplete correctness receipt: {name}")
        require(not seal["status"] and seal["verified_unchanged"] is True
                and all(seal[key] == value for key, value in preparation.items()), "Corrected source changed after validation")
        for key in ("file_sha256", "native_sha256", "runtime_patch_sha256", "generated_metadata_sha256"):
            require(run[key] == preparation[key], f"Correctness identity changed: {name}/{key}")
        require(preparation["native_sha256"] == prepared[name]["binaries"], "Corrected native binaries differ")
        require(set(run["file_sha256"]) == {"pycbc/filter/matchedfilter.py"}
                and run["file_sha256"]["pycbc/filter/matchedfilter.py"] == digest(root / "correctness" / f"{name}-matchedfilter.py"), "Corrected runtime snapshot differs")
        require(run["runtime_patch_sha256"] == digest(root / "correctness" / f"{name}-runtime.patch"), "Corrected runtime patch differs")
        require(run["generated_metadata_sha256"]["pycbc/version.py"] == digest(root / "correctness" / f"{name}-version.py"), "Corrected version metadata differs")
        require(run["log_sha256"] == digest(root / "correctness" / f"{name}.log"), f"Correctness log hash mismatch: {name}")
        xml = ET.parse(root / "correctness" / f"{name}.xml").getroot()
        suites = [xml] if xml.tag == "testsuite" else list(xml.findall("testsuite"))
        require(bool(suites), f"Missing JUnit test suites: {name}")
        tally = {key: sum(int(s.get(key, "0")) for s in suites)
                 for key in ("tests", "failures", "errors", "skipped")}
        require(all(value >= 0 for value in tally.values()) and
                tally["tests"] >= sum(tally[k] for k in ("failures", "errors", "skipped")), f"Invalid JUnit counts: {name}")
        tally["passed"] = tally["tests"] - sum(tally[k] for k in ("failures", "errors", "skipped"))
        counts.append({"name": name, "revision": run["revision"], "base_revision": run["base_revision"],
                       "file_sha256": run["file_sha256"], "runtime_patch_sha256": run["runtime_patch_sha256"], "returncode": run["returncode"],
                       "junit_counts": tally, "junit_sha256": digest(root / "correctness" / f"{name}.xml")})
    require(correctness["passed"] == all(r["returncode"] == 0 for r in runs), "Correctness success flag disagrees with commands")
    return {"queue_passed": queue["passed"], "correctness_passed": correctness["passed"], "correctness": counts}


def copy_file(source, target):
    record = file_record(source)
    require(not target.exists(), f"Copy would overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    target.chmod(record["mode"])
    require(file_record(target) == record, f"Copy changed bytes/mode: {source}")
    return record


def reproduction(source):
    text = source.read_text()
    replacements = {
        "python wave-support/triton/report/render.py": "python report-engines/triton/report/render.py",
        "--harness-dir triton-original": "--harness-dir report-engines/triton/harness",
        "python wave-support/waveform/report/waveform_report.py": "python report-engines/waveform/report/waveform_report.py",
        "--harness-dir wave-support/waveform/harness": "--harness-dir report-engines/waveform/harness",
        "python live-support/live_report.py": "python report-engines/live/live_report.py",
        "python live-support/probe_report.py": "python report-engines/live/probe_report.py",
    }
    for old, new in replacements.items():
        require(old in text, f"Reproduction draft changed; review substitution: {old}")
        text = text.replace(old, new)
    start = text.index("The final documentation manifest pins")
    end = text.index("`sealed-manifest.json`", start)
    text = text[:start] + """The standalone `documentation/render-manifest.json` binds all five summary
inputs, including unchanged historical inference/FFT summaries. It intentionally
has no commit pin to its own archive. From this supplement directory, run:

```sh
python documentation/plot_torch_benchmark_docs.py --archive .. \\
  --manifest documentation/render-manifest.json --output NEW_FIGURE_DIRECTORY
```

The renderer checks summary hashes and complete six-batch matrices before
plotting. `documentation/figures/` contains all eight rendered PNGs and the
renderer output manifest. The source documentation manifest can pin this
archive only after the archive commit exists. Do not modify the sealed remote
report scripts; `report-engines/` retains the later strict engines and their
required acquisition files separately. Whiskers and bands show observed worker
ranges, not confidence intervals. Results do not establish whole-search,
sampler, MPS or other-hardware performance.

""" + text[end:]
    return text.encode()


def supplement_inventory(root):
    return inventory(root, {"assembly/state.json", "assembly/.phase-lock", "SHA256SUMS"})


def prepare(args):
    archive = args.archive
    require(args.sealed is not None and args.docs_root is not None, "prepare requires --sealed and --docs-root")
    require(not (archive / "batch1024").exists(), "batch1024 already exists; refusing to replace it")
    baseline = check_archive(archive)
    sealed_hash = digest(args.sealed)
    with tempfile.TemporaryDirectory(prefix="batch1024-assembly-", dir=archive.parent) as temporary:
        root = Path(temporary) / "batch1024"
        root.mkdir()
        sealed = verify_tar(args.sealed, root)
        outcomes = completion(root)
        copies = {}
        for source, target in LOCAL_COPIES.items():
            copies[target] = copy_file(args.local_root / source, root / target)
            # The acquisition files must match the actual sealed campaign.
            if "/harness/" in target or target.rsplit("/", 1)[-1] in (
                    "run-live.py", "run-probes.py", "live_dispatch_probe.py"):
                require(source in sealed and copies[target]["sha256"] == sealed[source]["sha256"],
                        f"Local acquisition source differs from sealed source: {source}")
        renderer = "tools/plot_torch_benchmark_docs.py"
        copies["documentation/plot_torch_benchmark_docs.py"] = copy_file(
            args.docs_root / renderer, root / "documentation/plot_torch_benchmark_docs.py")
        template = read_json(args.docs_root / "docs/images/torch-benchmarks-20260906/manifest.json")
        require({r["file"] for r in template["figures"]} == set(FIGURES)
                and len(template["figures"]) == 8, "Unexpected documentation figure set")
        require({r["key"] for r in template["summary_inputs"]} == set(SUMMARY_PATHS), "Unexpected documentation summary set")
        local_validation = read_json(root / "validation/local-validation.json")
        require(local_validation["ci_f401"]["sha256"] ==
                digest(root / "validation/ci-f401-validation.json"), "CI F401 receipt hash mismatch")
        matches = {}
        for source, expected in local_validation.get("file_sha256", {}).items():
            safe_name(source)
            target = root / "validation/source-snapshot" / source
            copies[str(target.relative_to(root))] = copy_file(args.docs_root / source, target)
            matches[source] = {"recorded_sha256": expected, "copied_sha256": digest(target),
                               "matches_recorded_validation": expected == digest(target)}
        require(digest(args.docs_root / "test/test_torch_large_batches.py") ==
                digest(root / "test_torch_large_batches.py"), "Added regression file differs from completed external test")
        write_json(root / "validation/source-matches.json", matches)
        write_json(root / "validation/completed-correctness.json", outcomes)
        write_json(root / "assembly/local-copies.json", copies)
        write_json(root / "assembly/baseline-inventory.json", baseline)
        copy_file(archive / "README.md", root / "assembly/original-root-README.md")
        write_bytes(root / "REPRODUCE.md", reproduction(args.local_root / "REPRODUCE-draft.md"))
        verify_seal(root)
        require(digest(args.sealed) == sealed_hash, "Tar changed during preparation")
        state = {"schema": 1, "phase": "prepared", "archive_baseline": BASE,
                 "archive_path": str(archive), "sealed_tar_sha256": sealed_hash,
                 "sealed_file_count": len(sealed), "helper_sha256": digest(Path(__file__)),
                 "prepared_utc": datetime.now(timezone.utc).isoformat(),
                 "inventory": supplement_inventory(root)}
        write_json(root / "assembly/state.json", state)
        check_archive(archive, baseline)
        require(not (archive / "batch1024").exists(), "Concurrent preparation detected")
        root.rename(archive / "batch1024")
    print("Prepared batch1024; sealed files preserved byte-for-byte. Next phase: render.")


@contextmanager
def phase_lock(root):
    lock = root / "assembly/.phase-lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    try:
        yield
    finally:
        lock.unlink()


def load_phase(args, expected):
    root = args.archive / "batch1024"
    state = read_json(root / "assembly/state.json")
    require(state["phase"] == expected, f"Expected phase {expected}; found {state['phase']}. Refusing replay/overwrite.")
    require(state["helper_sha256"] == digest(Path(__file__)), "Helper changed since preparation")
    require(state["archive_path"] == str(args.archive), "Prepared checkout was moved")
    check_archive(args.archive, read_json(root / "assembly/baseline-inventory.json"))
    require(supplement_inventory(root) == state["inventory"], "Prepared supplement changed")
    verify_seal(root)
    completion(root)
    return root, state


def run_logged(command, cwd, log, timeout, environment):
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with log.open("x") as output:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"host": socket.gethostname(), "cwd": str(cwd), "command": command,
                          "pid": process.pid, "log": str(log), "next_check": f"completion or {timeout}s timeout",
                          "stop_command": f"kill -TERM -- -{process.pid}"}), flush=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    return {"command": command, "cwd": str(cwd), "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(), "returncode": code,
            "log": str(log.relative_to(cwd)), "log_sha256": digest(log)}


def report_commands(python):
    main, cpu = REVISIONS["main"], REVISIONS["cpu"]
    return [
        [python, "-B", "report-engines/triton/report/render.py", "--input-dir", "triton",
         "--output-dir", "report/triton", "--expected-sha", main,
         "--harness-dir", "report-engines/triton/harness", "--run-date", "2026-09-06", "--tables-only"],
        [python, "-B", "report-engines/waveform/report/waveform_report.py", "--input-dir", "waveform",
         "--output-dir", "report", "--expected-sha", main,
         "--harness-dir", "report-engines/waveform/harness", "--tables-only"],
        [python, "-B", "report-engines/live/live_report.py", "--input-dir", "live", "--output-dir", "report",
         "--main-revision", main, "--cpu-revision", cpu, "--tables-only"],
        [python, "-B", "report-engines/live/probe_report.py", "--input-dir", "probes", "--output-dir", "report",
         "--main-revision", main, "--cpu-revision", cpu],
    ]


def verify_reports(root):
    live = read_json(root / "report/live-summary.json")
    probe = read_json(root / "report/dispatch-summary.json")
    waveform = read_json(root / "report/waveform-summary.json")
    triton = read_json(root / "report/triton/report.json")
    require(live["expected_rows"] == live["total_rows"] == live["qualified_rows"] == 84,
            "Live strict report did not qualify all expected rows")
    require(probe["expected_rows"] == probe["qualified_rows"] == 60
            and probe["all_unsafe_helper_exclusions_verified"] is True, "Dispatch strict report failed")
    require(waveform["qualification_passed"] is True and waveform["qualified_timed_cells"] == 48
            and waveform["verified_unsupported_cells"] == 48, "Waveform strict report failed")
    require(triton["qualified_group_count"] == triton["expected_group_count"] == 24
            and triton["raw_worker_files"] == 72 and not triton["global_problems"], "Triton strict report failed")
    return {"live": live, "probe": probe, "waveform": waveform, "triton": triton}


def verify_figures(root):
    manifest = read_json(root / "documentation/render-manifest.json")
    figures = manifest.get("figures", [])
    require(len(figures) == 8 and {r["file"] for r in figures} == set(FIGURES),
            "Incomplete documentation figure inventory")
    require(manifest.get("documentation_renderer_sha256") ==
            digest(root / "documentation/plot_torch_benchmark_docs.py"), "Renderer hash mismatch")
    for row in figures:
        actual = file_record(root / "documentation/figures" / row["file"])
        require(row["sha256"] == actual["sha256"] and row["bytes"] == actual["size"],
                f"Figure hash/size mismatch: {row['file']}")
    require(manifest == read_json(root / "documentation/figures/manifest.json"),
            "Standalone and renderer output manifests disagree")
    return manifest


def render_manifest(archive):
    command = ("python batch1024/documentation/plot_torch_benchmark_docs.py --archive . "
               "--manifest batch1024/documentation/render-manifest.json --output NEW_FIGURE_DIRECTORY")
    raw = {"live": "batch1024/live/", "waveform": "batch1024/waveform/", "triton": "batch1024/triton/",
           "inference": "supplement/inference/", "fft": "supplement/fft/"}
    revisions = {"live": [REVISIONS["main"]], "waveform": [REVISIONS["main"]],
                 "triton": [REVISIONS["main"]], "inference": [HISTORICAL_MAIN],
                 "fft": [HISTORICAL_MAIN, HISTORICAL_FFT]}
    figures = []
    for filename, (key, selection) in FIGURES.items():
        figures.append({"file": filename, "summary_input": key, "selection": selection,
                        "source_revisions": [REVISIONS["main"], REVISIONS["cpu"]]
                        if filename == "optional-cpu.png" else revisions[key],
                        "raw_results": raw[key], "renderer_command": command,
                        "archive_figure_path": "batch1024/documentation/figures/" + filename})
    return {"schema_version": 2, "campaign_date": "2026-09-06", "standalone": True,
            "artifact_repository": "https://github.com/xangma/pycbc", "baseline_archive_commit": BASE,
            "commit_pin_policy": "No pin to this archive's future commit; summary and figure hashes are self-contained.",
            "checksums_path": "batch1024/SHA256SUMS", "reproduction_paths": ["batch1024/REPRODUCE.md"],
            "renderer_cwd": "Evidence archive checkout", "runner": "Manual Linux runner; no GitHub Actions CI run",
            "documentation_renderer": "batch1024/documentation/plot_torch_benchmark_docs.py",
            "summary_inputs": [{"key": key, "path": path, "sha256": digest(archive / path)}
                               for key, path in SUMMARY_PATHS.items()], "figures": figures}


def prepared_readmes(root, reports):
    correctness = read_json(root / "validation/completed-correctness.json")
    text = ("# Batch sizes through 1024\n\n"
            "Batched campaigns cover 1, 8, 32, 128, 512 and 1024. Historical single-vector FFT "
            "and single-evaluation inference retain their original sources.\n\n"
            f"Strict reports qualified {reports['live']['qualified_rows']} live groups, "
            f"{reports['probe']['qualified_rows']} untimed dispatch probes, "
            f"{reports['waveform']['qualified_timed_cells']} waveform timing cells plus "
            f"{reports['waveform']['verified_unsupported_cells']} unsupported cells, and "
            f"{reports['triton']['qualified_group_count']} Triton groups.\n\n"
            "| Correctness source | Passed | Skipped | Failures | Errors | Exit code |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: |\n")
    for row in correctness["correctness"]:
        count = row["junit_counts"]
        text += (f"| {row['name']} `{row['revision']}` | {count['passed']} | {count['skipped']} | "
                 f"{count['failures']} | {count['errors']} | {row['returncode']} |\n")
    text += ("\nCounts above come from completed JUnit records; exact commands and outcomes are in "
             "[correctness/status.json](correctness/status.json). The external regression-file hash "
             "is retained there. [Local validation](validation/local-validation.json) keeps its original "
             "scope; [source matches](validation/source-matches.json) distinguish validated and later files.\n\n"
             "The expanded tests exposed a crash when real chi-square calculators are disabled. "
             "Correctness sources are the measured revisions plus the archived one-condition fallback fix; "
             "their exact parents, patches and runtime file hashes are recorded separately. Earlier failed "
             "fixture and runtime attempts remain under `correctness/initial-failed/` and "
             "`correctness/disabled-veto-failed/`. Benchmark measurements retain their original sources.\n\n"
             "[Reproduce acquisition and reports](REPRODUCE.md). Full strict tables: "
             "[live](report/live-summary.md), [dispatch](report/dispatch-summary.md), "
             "[waveforms](report/waveform-summary.md), [Triton](report/triton/report.md).\n\n"
             "Live timing is a synthetic driver with chi-square disabled, stubbed sine-Gaussian processing "
             "and unsafe asynchronous peak copies disabled. It excludes waveform generation, PSD estimation, "
             "I/O and startup. Worker ranges are observed ranges, not confidence intervals. These results "
             "do not establish full production-search or sampler performance.\n\n")
    text += "\n".join(f"![{name.removesuffix('.png')}](documentation/figures/{name})\n" for name in FIGURES)
    write_bytes(root / "README.md", text.encode())
    prefix = ("**Batch sizes through 1024:** the [new supplement](batch1024/README.md) contains "
              "completed live, waveform and Triton measurements, separate dispatch probes, correctness "
              "receipts for all three clean sources, and eight simplified documentation figures. "
              "[Reproduction instructions](batch1024/REPRODUCE.md) and "
              "[standalone figure manifest](batch1024/documentation/render-manifest.json) accompany it. "
              "The historical campaign below retains its original sources and files.\n\n")
    write_bytes(root / "assembly/root-README.prepared.md",
                prefix.encode() + (root / "assembly/original-root-README.md").read_bytes())


def render(args):
    root = args.archive / "batch1024"
    with phase_lock(root):
        root, state = load_phase(args, "prepared")
        require(not (root / "report").exists() and not (root / "documentation/figures").exists(), "Output directories already exist")
        state["phase"] = "render-started"
        write_json(root / "assembly/state.json", state)
        receipts = []
        try:
            with tempfile.TemporaryDirectory(prefix="batch1024-matplotlib-") as cache:
                environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=cache)
                environment.pop("PYTHONPATH", None)
                for index, command in enumerate(report_commands(args.python)):
                    receipts.append(run_logged(command, root, root / f"assembly/logs/report-{index}.log", args.timeout, environment))
                    write_json(root / "assembly/report-receipts.json", receipts)
                require(all(r["returncode"] == 0 for r in receipts), "At least one strict report failed; outputs retained")
                reports = verify_reports(root)
                manifest = render_manifest(args.archive)
                write_json(root / "documentation/render-manifest.json", manifest)
                command = [args.python, "-B", "documentation/plot_torch_benchmark_docs.py", "--archive", "..",
                           "--manifest", "documentation/render-manifest.json", "--output", "documentation/figures"]
                receipt = run_logged(command, root, root / "assembly/logs/documentation-render.log", args.timeout, environment)
                write_json(root / "assembly/documentation-render-receipt.json", receipt)
                require(receipt["returncode"] == 0, "Documentation renderer failed; outputs retained")
            generated = read_json(root / "documentation/figures/manifest.json")
            require({r["file"] for r in generated["figures"]} == set(FIGURES), "Renderer did not produce the eight expected figures")
            for row in generated["figures"]:
                actual = file_record(root / "documentation/figures" / row["file"])
                require(row["sha256"] == actual["sha256"] and row["bytes"] == actual["size"], "Figure hash/size mismatch")
            write_json(root / "documentation/render-manifest.json", generated)
            prepared_readmes(root, reports)
            verify_seal(root)
            check_archive(args.archive, read_json(root / "assembly/baseline-inventory.json"))
            state.update(phase="rendered", rendered_utc=datetime.now(timezone.utc).isoformat(),
                         inventory=supplement_inventory(root))
            write_json(root / "assembly/state.json", state)
        except BaseException as error:
            state.update(phase="render-failed", error=f"{type(error).__name__}: {error}")
            write_json(root / "assembly/state.json", state)
            raise
    print("Rendered eight figures. Inspect batch1024/documentation/figures and assembly/root-README.prepared.md before finish.")


def checksums(root, excluded):
    return "".join(f"{row['sha256']}  {name}\n" for name, row in inventory(root, excluded).items()).encode()


def finish(args):
    root = args.archive / "batch1024"
    with phase_lock(root):
        root, state = load_phase(args, "rendered")
        verify_reports(root)
        manifest = verify_figures(root)
        require("artifact_commit" not in manifest and "artifact_url" not in manifest, "Standalone manifest contains a circular archive pin")
        require(manifest["summary_inputs"] == render_manifest(args.archive)["summary_inputs"], "Summary inputs changed")
        baseline = read_json(root / "assembly/baseline-inventory.json")
        state["phase"] = "finish-started"
        write_json(root / "assembly/state.json", state)
        try:
            write_json(root / "assembly/finalization.json", {
                "finished_utc": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
                "python": sys.version, "archive_baseline": BASE, "sealed_tar_sha256": state["sealed_tar_sha256"],
                "standalone_manifest_sha256": digest(root / "documentation/render-manifest.json"),
                "sealed_files_unchanged": True, "all_four_strict_reports_passed": True,
                "checksum_policy": "Supplement checksum excludes itself; root checksum covers it and excludes only root SHA256SUMS. State inventory excludes itself, the phase lock and supplement SHA256SUMS."})
            state.update(phase="finished", inventory=supplement_inventory(root))
            write_json(root / "assembly/state.json", state)
            write_bytes(root / "SHA256SUMS", checksums(root, {"SHA256SUMS", "assembly/.phase-lock"}))
            check_archive(args.archive, baseline)
            write_bytes(args.archive / "README.md", (root / "assembly/root-README.prepared.md").read_bytes())
            # These are the only two pre-existing paths that this helper writes.
            write_bytes(args.archive / "SHA256SUMS", checksums(args.archive, {"SHA256SUMS", "batch1024/assembly/.phase-lock"}))
            expected = dict(baseline)
            expected.update({name: file_record(args.archive / name) for name in ("README.md", "SHA256SUMS")})
            check_archive(args.archive, expected)
            verify_seal(root)
        except BaseException as error:
            # Keep partial writes for review; never roll back another process's work.
            state.update(phase="finish-failed", error=f"{type(error).__name__}: {error}")
            write_json(root / "assembly/state.json", state)
            raise
    print("Finished archive files and checksums. No files were staged, committed or pushed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "render", "finish"))
    parser.add_argument("--archive", required=True, type=Path, help="Clean evidence checkout at the reviewed baseline")
    parser.add_argument("--sealed", type=Path, help="Completed sealed-evidence.tar.gz; prepare only")
    parser.add_argument("--local-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--docs-root", type=Path, help="Current documentation checkout; prepare only")
    parser.add_argument("--python", default=sys.executable, help="Interpreter with matplotlib; report/render phase only")
    parser.add_argument("--timeout", type=int, default=300, help="Per-child report/render timeout in seconds")
    args = parser.parse_args()
    require(1 <= args.timeout <= 3600, "Timeout must be 1..3600 seconds")
    for field in ("archive", "sealed", "local_root", "docs_root"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.resolve(strict=True))
    resolved_python = shutil.which(args.python)
    require(resolved_python is not None, f"Python executable not found: {args.python}")
    args.python = str(Path(resolved_python).resolve())
    print(f"Phase {args.phase}; archive {args.archive}; helper SHA256 {digest(Path(__file__))}", flush=True)
    {"prepare": prepare, "render": render, "finish": finish}[args.phase](args)


if __name__ == "__main__":
    main()
