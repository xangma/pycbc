#!/usr/bin/env python3
"""Reproduce the four-arm, 2048 Hz/complex64 complete-search benchmark suite.

The plan command is read-only. prepare builds inputs; qualify performs a small
science-only preflight; run executes either a quick diagnostic or the full
timing, scaling, paced-replay, and profiling suite. Every invocation writes to
a new phase directory and stops on the first failure. The scope selects
inspiral, live, or both without changing their pinned arms.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

try:
    from tools.benchmark_artifact import file_sha256 as sha256, source_identity
    from tools.benchmark_reference import validate_reference
except ModuleNotFoundError:
    from benchmark_artifact import file_sha256 as sha256, source_identity
    from benchmark_reference import validate_reference


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "40e94792b3edf59f39b18b65102b28a4f74433a7"
INSPIRAL_ARMS = ["original_cpu", "branch_cpu", "jax_cpu_batched", "jax_cuda_batched"]
LIVE_ARMS = ["cpu", "branch_cpu", "jax_cpu", "jax_cuda"]
QUICK_LIVE_DURATION_SEC = 64
PATH_FIELDS = ("reference_source", "candidate_source", "python", "inputs_dir",
               "frame_file", "output_dir", "bank_xml")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def load_config(path):
    config = json.loads(Path(path).read_text())
    allowed = set(PATH_FIELDS) | {"live_l1_frame_file", "reference_revision", "frame_sha256",
        "live_l1_frame_sha256",
        "bank_xml_sha256", "sizes", "qualification_size", "replicates",
        "inspiral_affinity", "live_affinity", "live_ranks"}
    if set(config) - allowed:
        raise ValueError(f"Unknown configuration fields: {sorted(set(config) - allowed)}")
    for key in PATH_FIELDS:
        if key == "bank_xml" and not config.get(key):
            continue
        value = Path(config[key]).expanduser()
        if not value.is_absolute():
            value = Path(path).resolve().parent / value
        config[key] = str(value.resolve())
    if config.get("live_l1_frame_file"):
        if not config.get("live_l1_frame_sha256"):
            raise ValueError("live_l1_frame_sha256 is required with live_l1_frame_file")
        value = Path(config["live_l1_frame_file"]).expanduser()
        if not value.is_absolute():
            value = Path(path).resolve().parent / value
        config["live_l1_frame_file"] = str(value.resolve())
    elif config.get("live_l1_frame_sha256"):
        raise ValueError("live_l1_frame_file is required with live_l1_frame_sha256")
    for key in ("frame_sha256", "live_l1_frame_sha256", "bank_xml_sha256"):
        if key == "live_l1_frame_sha256" and not config.get("live_l1_frame_file"):
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", config[key]):
            raise ValueError(f"{key} must be a pinned SHA256")
    if config.get("reference_revision") != REFERENCE:
        raise ValueError("This workload pins the original CPU revision to " + REFERENCE)
    sizes = config.setdefault("sizes", [1536, 3072, 6144])
    if (len(sizes) < 3 or any(type(n) is not int or n < 1536 for n in sizes)
            or any(b != 2 * a for a, b in zip(sizes, sizes[1:]))):
        raise ValueError("sizes must contain at least three successive doublings starting at >=1536")
    if config.setdefault("qualification_size", 32) != 32:
        raise ValueError("The science-only preflight uses 32 distinct templates")
    if type(config.setdefault("replicates", 3)) is not int or config["replicates"] < 3:
        raise ValueError("At least three fresh timing repetitions are required")
    if not re.fullmatch(r"\d+", str(config["inspiral_affinity"])):
        raise ValueError("inspiral_affinity must select one logical CPU / physical core")
    if not config.get("live_affinity") or config.get("live_ranks", 0) < 1:
        raise ValueError("Explicit live affinity and positive rank count are required")
    for key in ("inputs_dir", "output_dir"):
        for source in ("reference_source", "candidate_source"):
            if Path(config[key]).is_relative_to(config[source]):
                raise ValueError(f"{key} must be outside both source checkouts")
    return config


def live_config(config, size, quick=False):
    result = json.loads((ROOT / "tools/live_campaign_h1.example.json").read_text())
    bank = str(Path(config["inputs_dir"]) / f"o2-subset-{size}.hdf")
    result["bank_file"] = bank
    result["frame_files"] = ([config["frame_file"], config["live_l1_frame_file"]]
                              if config.get("live_l1_frame_file") else
                              [config["frame_file"]])
    result["workload"]["templates"] = size
    args = result["args"]
    args[args.index("--bank-file") + 1] = bank
    if quick:
        start = int(result["workload"]["start_time"])
        end = start + QUICK_LIVE_DURATION_SEC
        result["workload"]["end_time"] = end
        args[args.index("--end-time") + 1] = str(end)
    frame_pos = args.index("--frame-src")
    args[frame_pos + 1] = "H1:" + config["frame_file"]
    channel_pos = args.index("--channel-name")
    args[channel_pos + 1] = "H1:LOSC-STRAIN"
    if config.get("live_l1_frame_file"):
        args.insert(frame_pos + 2, "L1:" + config["live_l1_frame_file"])
        args.insert(channel_pos + 2, "L1:LOSC-STRAIN")
        if "--enable-background-estimation" not in args:
            args.append("--enable-background-estimation")
        for option in ("--ifar-double-followup-threshold", "--ifar-upload-threshold"):
            args[args.index(option) + 1] = "1e9"
    return result


def campaign(config, directory, kind, size, mode="timing",
             allow_unqualified_timings=False, replicates=None, quick=False,
             short_live=False):
    """Return an explicit argv and any configuration file it consumes."""
    output = Path(directory) / f"{kind}-{size}-{mode}"
    repetitions = config["replicates"] if replicates is None else replicates
    common = ["--python", config["python"], "--reference-revision", REFERENCE,
              "--replicates", str(repetitions),
              "--output", str(output / "campaign.json"),
              "--output-dir", str(output / "runs")]
    generated = {}
    if kind == "inspiral":
        command = [config["python"], str(ROOT / "tools/bench_jax_inspiral_campaign.py"),
            "--original-source", config["reference_source"],
            "--branch-source", config["candidate_source"],
            "--frame-file", config["frame_file"],
            "--bank-file", str(Path(config["inputs_dir"]) / f"o2-compressed-{size}.hdf"),
            "--waveform-mode", "compressed", "--approximant", "IMRPhenomD",
            "--order", "-1", "--batch-size", "64", "--arms", *INSPIRAL_ARMS,
            "--affinity", str(config["inspiral_affinity"]), *common]
    else:
        configuration = output / "live-config.json"
        generated[str(configuration)] = live_config(
            config, size, quick=(quick or short_live))
        command = [config["python"], str(ROOT / "tools/bench_jax_live_campaign.py"),
            "--config", str(configuration), "--source-root", config["candidate_source"],
            "--reference-source-root", config["reference_source"],
            "--arms", *LIVE_ARMS, "--ranks", str(config["live_ranks"]),
            "--gpus", "1", "--affinity", str(config["live_affinity"]),
            "--replay-mode", "paced" if kind == "live-paced" else "unpaced",
            "--replay-rate", "1", *common]
    if mode == "qualification":
        command.append("--qualification-only")
    elif mode == "profile":
        command.append("--profile-utilization")
    if allow_unqualified_timings:
        command.append("--allow-unqualified-timings")
    if quick:
        command.append("--quick")
    return {"kind": kind, "templates": size, "mode": mode, "command": command,
            "receipt": str(output / "campaign.json"), "generated_files": generated}


def make_plan(config, phase, scope="both", allow_unqualified_timings=False,
              preset="full", quick_profiles=False):
    if scope not in ("both", "inspiral", "live"):
        raise ValueError("scope must be both, inspiral, or live")
    if preset not in ("quick", "full"):
        raise ValueError("preset must be quick or full")
    if quick_profiles and (phase != "run" or preset != "quick"):
        raise ValueError("quick_profiles requires the quick run preset")
    directory_name = "quick" if phase == "run" and preset == "quick" else phase
    directory = Path(config["output_dir"]) / directory_name
    if phase == "run" and preset == "quick":
        size = config["qualification_size"]
        steps = [campaign(config, directory, kind, size, "timing",
                          allow_unqualified_timings, replicates=1, quick=True)
                 for kind in ("inspiral", "live-unpaced")]
        if quick_profiles:
            steps += [campaign(config, directory, kind, size, "profile",
                               allow_unqualified_timings, replicates=1,
                               short_live=True)
                      for kind in ("inspiral", "live-unpaced")]
    else:
        steps = [campaign(config, directory, kind, 32, "qualification",
                          allow_unqualified_timings)
                 for kind in ("inspiral", "live-unpaced")]
    if phase == "run" and preset == "full":
        steps += [campaign(config, directory, kind, size, "timing",
                           allow_unqualified_timings)
                  for size in config["sizes"] for kind in ("inspiral", "live-unpaced")]
        size = config["sizes"][-1]
        steps.append(campaign(config, directory, "live-paced", size, "timing",
                              allow_unqualified_timings))
        steps += [campaign(config, directory, kind, size, "profile",
                           allow_unqualified_timings)
                  for kind in ("inspiral", "live-unpaced", "live-paced")]
    if scope != "both":
        selected_kinds = ({"inspiral"} if scope == "inspiral"
                          else {"live-unpaced", "live-paced"})
        steps = [step for step in steps if step["kind"] in selected_kinds]
    return {"config": config, "phase": phase, "scope": scope,
            "preset": preset, "quick_profiles": bool(quick_profiles),
            "steps": steps,
            "allow_unqualified_timings": bool(allow_unqualified_timings),
            "sample_rate": 2048, "precision": "complex64",
            "performance_claim": (preset == "full"
                                  and not allow_unqualified_timings),
            "convergence": "<5% change in median full-process capacity at each of two successive template doublings; bank-size convergence only, always finite detector duration"}


def preparation_command(config):
    command = [config["python"], str(ROOT / "tools/prepare_jax_benchmark_inputs.py"),
        "--output-dir", config["inputs_dir"], "--python", config["python"],
        "--reference-source", config["reference_source"],
        "--reference-revision", REFERENCE, "--frame-file", config["frame_file"],
        "--frame-sha256", config["frame_sha256"],
        "--bank-xml-sha256", config["bank_xml_sha256"],
        "--sizes", "32", *map(str, config["sizes"])]
    if config.get("bank_xml"):
        command += ["--bank-xml", config["bank_xml"]]
    return command


def capture_provenance(config, directory):
    """Preserve source patches/untracked files and selected-runtime package pins."""
    result = {}
    for role in ("reference", "candidate"):
        source = Path(config[role + "_source"])
        result[role] = source_identity(source)
        patch = subprocess.check_output(["git", "-C", str(source), "diff", "--binary", "HEAD"])
        (directory / f"{role}.patch").write_bytes(patch)
        names = subprocess.check_output(["git", "-C", str(source), "ls-files",
                                         "--others", "--exclude-standard", "-z"])
        with tarfile.open(directory / f"{role}-untracked.tar.gz", "w:gz") as archive:
            for name in names.decode().split("\0"):
                if name and (source / name).is_file():
                    archive.add(source / name, arcname=name, recursive=False)
    code = ("import importlib.metadata as m,json,platform,sys; "
            "print(json.dumps({'python':sys.version,'prefix':sys.prefix,"
            "'platform':platform.platform(),'packages':"
            "{d.metadata['Name']:d.version for d in m.distributions()}}))")
    result["runtime"] = json.loads(subprocess.check_output(
        [config["python"], "-c", code], text=True))
    pins = result["runtime"]["packages"]
    wheel_options = ""
    if any(name.lower() == "jaxlib" and "+cuda" in version
           for name, version in pins.items()):
        result["jax_wheel_repository"] = "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        wheel_options = "--find-links " + result["jax_wheel_repository"] + "\n"
    (directory / "requirements-resolved.txt").write_text(wheel_options + "".join(
        f"{name}=={version}\n" for name, version in sorted(pins.items())
        if name.lower() != "pycbc"))
    prefix = Path(result["runtime"]["prefix"])
    conda = shutil.which("conda") or str(prefix.parent.parent / "bin/conda")
    if (prefix / "conda-meta").is_dir() and Path(conda).is_file():
        explicit = subprocess.check_output(
            [conda, "list", "--prefix", str(prefix), "--explicit"], text=True)
        (directory / "conda-explicit.txt").write_text(explicit)
        packages = json.loads(subprocess.check_output(
            [conda, "list", "--prefix", str(prefix), "--json"], text=True))
        (directory / "pip-only.txt").write_text(wheel_options + "".join(
            f"{p['name']}=={p['version']}\n" for p in packages
            if p.get("channel") == "pypi" and p["name"].lower() != "pycbc"))
        result["environment_restore"] = "conda-explicit.txt plus pip-only.txt, then editable PyCBC sources"
    else:
        result["environment_restore"] = "requirements-resolved.txt plus matching native libraries and Python"
    # Record tools/drivers and topology, including logical-to-physical mappings.
    for label, command in (("cpu", ["lscpu", "--json"]),
                           ("topology", ["lscpu", "-p=CPU,CORE,SOCKET,ONLINE"]),
                           ("gpu", ["nvidia-smi", "-q"]),
                           ("mpi", ["mpirun", "--version"])):
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            result[label] = {"returncode": completed.returncode, "stdout": completed.stdout,
                             "stderr": completed.stderr}
        except OSError as exc:
            result[label] = {"unavailable": str(exc)}
    result["environment"] = {key: os.environ[key] for key in (
        "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "JAX_ENABLE_X64",
        "XLA_FLAGS", "LD_LIBRARY_PATH") if key in os.environ}
    write_json(directory / "provenance.json", result)
    return result


def science_passed(receipt):
    science = receipt.get("science", {})
    arms = LIVE_ARMS if receipt.get("executable") == "pycbc_live" else INSPIRAL_ARMS
    if set(science) != set(arms) or not all(
            value.get("passed") is True for value in science.values()):
        return False
    if receipt.get("campaign", {}).get("passed") is False:
        return False
    if receipt.get("executable") == "pycbc_live":
        return receipt.get("campaign", {}).get("passed") is True
    return receipt.get("status") in (
        "complete", "science_qualification_only", "profiled_science_qualification")


def known_divergence_complete(receipt, step):
    """Accept complete process evidence without converting failed science to pass."""
    policy = receipt.get("timing_policy", {})
    campaign_result = receipt.get("campaign", {})
    arms = INSPIRAL_ARMS if step["kind"] == "inspiral" else LIVE_ARMS
    science = receipt.get("science", {})
    if (policy.get("quick") is True
            or policy.get("allow_unqualified_timings") is not True
            or campaign_result.get("process_complete") is not True
            or set(science) != set(arms)
            or any(type(science[arm].get("passed")) is not bool for arm in arms)
            or campaign_result.get("passed") is not False
            or campaign_result.get("performance_claim") is not False):
        return False
    if step["mode"] == "timing":
        runs = receipt.get("raw_results", {})
        return (set(runs) == set(arms)
                and all(len(runs[arm]) > 0 for arm in arms)
                and all(run.get("process_complete",
                                run.get("elapsed_wall_sec", 0) > 0) is True
                        for arm in arms for run in runs[arm]))
    if step["mode"] == "profile":
        profiles = receipt.get("profile_results", {})
        return set(profiles) == set(arms)
    qualification = receipt.get(
        "qualification" if step["kind"] == "inspiral"
        else "qualification_results", {})
    return set(qualification) == set(arms)


def quick_diagnostic_complete(receipt, step):
    """Require complete single-sample evidence with no performance claim."""
    if step["mode"] != "timing":
        return False
    arms = INSPIRAL_ARMS if step["kind"] == "inspiral" else LIVE_ARMS
    policy = receipt.get("timing_policy", {})
    campaign_result = receipt.get("campaign", {})
    science = receipt.get("science", {})
    runs = receipt.get("raw_results", {})
    qualification = receipt.get(
        "qualification" if step["kind"] == "inspiral"
        else "qualification_results", {})
    profiles = receipt.get("profile_results", {})
    profile_free = (
        isinstance(profiles, dict)
        and all(not results for results in profiles.values())
    )
    qualification_complete = (
        all(isinstance(qualification[arm], dict)
            and qualification[arm].get("qualification_run") is True
            and qualification[arm].get("elapsed_wall_sec", 0) > 0
            for arm in arms)
        if step["kind"] == "inspiral" else
        all(isinstance(qualification[arm], list)
            and len(qualification[arm]) == 1
            and qualification[arm][0].get("process_complete") is True
            and qualification[arm][0].get("returncode") == 0
            for arm in arms)
    ) if set(qualification) == set(arms) else False
    timed_complete = (
        set(runs) == set(arms)
        and all(len(runs[arm]) == 1 for arm in arms)
        and all(run.get("process_complete",
                        run.get("elapsed_wall_sec", 0) > 0) is True
                and run.get("returncode", 0) == 0
                for arm in arms for run in runs[arm])
    )
    science_failed = any(
        science.get(arm, {}).get("passed") is False for arm in arms)
    return (
        receipt.get("executable") == (
            "pycbc_inspiral" if step["kind"] == "inspiral" else "pycbc_live")
        and receipt.get("status") == "complete_quick_diagnostic"
        and receipt.get("benchmark_tier") == "quick"
        and receipt.get("diagnostic_only") is True
        and policy.get("quick") is True
        and policy.get("diagnostic_only") is True
        and policy.get("effective_replicates") == 1
        and policy.get("minimum_publishable_replicates") == 3
        and campaign_result.get("quick") is True
        and campaign_result.get("status") == "complete_quick_diagnostic"
        and campaign_result.get("diagnostic_only") is True
        and campaign_result.get("process_complete") is True
        and campaign_result.get("passed") is False
        and campaign_result.get("performance_claim") is False
        and campaign_result.get("publishable") is False
        and receipt.get("performance_claim") is False
        and receipt.get("publishable") is False
        and set(science) == set(arms)
        and all(type(science[arm].get("passed")) is bool for arm in arms)
        and (not science_failed
             or policy.get("allow_unqualified_timings") is True)
        and qualification_complete
        and timed_complete
        and profile_free
        and (step["kind"] == "inspiral"
             or (receipt.get("baseline", {}).get("pristine") is True
                 and receipt.get("baseline", {}).get("provided") is True
                 and receipt.get("baseline", {}).get("stable") is True))
    )


def completed_science_failure(receipt, step):
    """Recognize a finished qualification with failed science, not a crashed run."""
    if step["mode"] != "qualification":
        return False
    inspiral = step["kind"] == "inspiral"
    arms = INSPIRAL_ARMS if inspiral else LIVE_ARMS
    executable = "pycbc_inspiral" if inspiral else "pycbc_live"
    science = receipt.get("science", {})
    qualification = receipt.get(
        "qualification" if inspiral else "qualification_results", {})
    if (receipt.get("executable") != executable or set(science) != set(arms)
            or set(qualification) != set(arms)
            or any(type(science[arm].get("passed")) is not bool for arm in arms)
            or not any(science[arm]["passed"] is False for arm in arms)):
        return False
    if inspiral:
        failed_arms = {arm for arm in arms if science[arm]["passed"] is False}
        failure = receipt.get("failure", {})
        return (receipt.get("status") == "failed"
                and failure.get("phase") == "qualification_science"
                and set(failure.get("arms", [])) == failed_arms
                and all(isinstance(qualification[arm], dict)
                        and qualification[arm].get("qualification_run") is True
                        and qualification[arm].get("triggers_path")
                        and qualification[arm].get("evidence_path")
                        for arm in arms))
    campaign = receipt.get("campaign", {})
    return (campaign.get("process_complete") is True
            and all(isinstance(qualification[arm], list)
                    and len(qualification[arm]) == 1
                    and qualification[arm][0].get("returncode") == 0
                    and qualification[arm][0].get("process_complete") is True
                    for arm in arms))


def reconcile_science_failure(state):
    """Use a finished qualification receipt to update an earlier process error.

    A campaign may be resumed independently after the suite driver has stopped.
    Only a failed, fully populated science receipt can replace the suite's old
    subprocess error; this never changes a step to passed or starts later steps.
    """
    if state.get("status") != "failed":
        return
    for step in state.get("steps", []):
        if step.get("status") != "failed" or step.get("mode") != "qualification":
            continue
        path = Path(step["receipt"])
        if not path.is_file():
            continue
        receipt = json.loads(path.read_text())
        arms = INSPIRAL_ARMS if step.get("kind") == "inspiral" else LIVE_ARMS
        executable = "pycbc_inspiral" if step.get("kind") == "inspiral" else "pycbc_live"
        failure = receipt.get("failure", {})
        science = receipt.get("science", {})
        qualification = receipt.get("qualification", receipt.get("qualification_results", {}))
        failed_arms = {arm for arm, result in science.items() if result.get("passed") is False}
        if (receipt.get("executable") != executable or receipt.get("status") != "failed"
                or failure.get("phase") != "qualification_science"
                or set(qualification) != set(arms) or set(science) != set(arms)
                or any(type(result.get("passed")) is not bool
                       for result in science.values())
                or not failed_arms or set(failure.get("arms", [])) != failed_arms):
            continue
        error = (f"Campaign phase qualification_science failed for {step['kind']} "
                 f"{step['templates']} templates: {', '.join(sorted(failed_arms))}; "
                 f"{failure.get('error', 'science gates failed')}. Receipt: {path}")
        if state.get("error") != error:
            step.setdefault("prior_execution_error", state.get("error"))
            state["error"] = error
        step["failure"] = failure


def reconcile_quick_diagnostics(state):
    """Recover a quick suite after completed receipts pass current validation."""
    if state.get("preset") != "quick" or not state.get("steps"):
        return False
    receipts = []
    for step in state["steps"]:
        path = Path(step["receipt"])
        if not path.is_file():
            return False
        receipt = json.loads(path.read_text())
        complete = (quick_diagnostic_complete(receipt, step)
                    if step["mode"] == "timing" else
                    (science_passed(receipt)
                     or known_divergence_complete(receipt, step)))
        if not complete:
            return False
        receipts.append(receipt)
    for step, receipt in zip(state["steps"], receipts):
        step["status"] = (
            "complete_quick_diagnostic" if step["mode"] == "timing" else
            "passed" if science_passed(receipt) else
            "complete_known_divergence")
        step["figures"] = []
        step["failed_science_arms"] = [
            arm for arm, result in receipt["science"].items()
            if result.get("passed") is False
        ]
    state.update(
        status="complete_quick_diagnostic",
        performance_claim=False,
        publishable=False,
        diagnostic_only=True,
    )
    state.pop("error", None)
    return True


def validate_inputs(config, manifest):
    """Require the frozen prepared bank set, not merely files with familiar names."""
    if manifest.get("reference_revision") != REFERENCE:
        raise ValueError("Inputs were not prepared with the pinned reference")
    for key in ("frame_sha256", "bank_xml_sha256"):
        if manifest.get("inputs", {}).get(key) != config[key]:
            raise ValueError("Input manifest mismatch: " + key)
    if sha256(config["frame_file"]) != config["frame_sha256"]:
        raise ValueError("Frame checksum mismatch")
    if config.get("live_l1_frame_file") and sha256(config["live_l1_frame_file"]) != config["live_l1_frame_sha256"]:
        raise ValueError("Live L1 frame checksum mismatch")
    for size in [32, *config["sizes"]]:
        record = manifest.get("banks", {}).get(str(size), {})
        indices = record.get("source_rows", [])
        if len(indices) != size or len(set(indices)) != size:
            raise ValueError(f"Missing distinct source-row manifest for {size}")
        for kind, stem in (("subset", "o2-subset"), ("compressed", "o2-compressed")):
            path = Path(config["inputs_dir"]) / f"{stem}-{size}.hdf"
            if sha256(path) != record.get(kind + "_sha256"):
                raise ValueError(f"Prepared {kind} bank checksum mismatch: {size}")
    for small, large in zip([32, *config["sizes"]], config["sizes"]):
        if not set(manifest["banks"][str(small)]["source_rows"]) <= set(
                manifest["banks"][str(large)]["source_rows"]):
            raise ValueError("Prepared banks must use nested distinct subsets")


def render_figures(step):
    """Build capacity and whole-process/stage figures from qualified receipts."""
    try:
        from tools.plot_jax_search_capacity import plot_campaign
        from tools.plot_jax_gpu_timeline import plot_campaign_figures, stage_zoom_windows
    except ModuleNotFoundError:
        from plot_jax_search_capacity import plot_campaign
        from plot_jax_gpu_timeline import plot_campaign_figures, stage_zoom_windows
    path = Path(step["receipt"])
    receipt = json.loads(path.read_text())
    outputs = []
    if (not science_passed(receipt)
            and not known_divergence_complete(receipt, step)
            and not quick_diagnostic_complete(receipt, step)):
        return outputs
    if step["mode"] == "timing":
        output = path.parent / "capacity.png"
        plot_campaign(receipt, output)
        outputs.append(str(output))
    elif step["mode"] == "profile":
        profiles = receipt.get("profile_results", {})
        expected = INSPIRAL_ARMS if step["kind"] == "inspiral" else LIVE_ARMS
        if set(profiles) != set(expected):
            raise ValueError("Missing profile arms")
        for arm, entries in profiles.items():
            run = entries[0] if isinstance(entries, list) else entries
            timeline_path = Path(run["timeline_path"])
            data = json.loads(timeline_path.read_text())
            # Pristine reference executables cannot emit the branch's
            # structured markers.  Their observer still reconstructs phases
            # from stable log boundaries, which is sufficient for stage
            # figures.  Require usable windows rather than one marker source.
            if (data.get("returncode") != 0 or not data.get("telemetry")
                    or not stage_zoom_windows(data)):
                raise ValueError(f"Incomplete process/stage profile for {arm}")
            if not data.get("process_tree", {}).get("completion", {}).get(
                    "all_observed_processes_exited"):
                raise ValueError(f"Profile process tree did not complete for {arm}")
            outputs.extend(map(str, plot_campaign_figures(data,
                timeline_path.parent / "process-timeline.png",
                stage_zoom_dir=timeline_path.parent / "stages")))
    return outputs


def render_report(directory):
    """Describe the actual commands/results, including incomplete or failed runs."""
    try:
        from tools.plot_jax_search_capacity import campaign_rows, render_campaign_report
    except ModuleNotFoundError:
        from plot_jax_search_capacity import campaign_rows, render_campaign_report
    state = json.loads((directory / "suite.json").read_text())
    plan_path = directory / "plan.json"
    plan_config = (json.loads(plan_path.read_text()).get("config", {})
                   if plan_path.exists() else {})
    scope = state.get("scope", "both")
    preset = state.get("preset", "full")
    live_duration = QUICK_LIVE_DURATION_SEC if preset == "quick" else 640
    live_start = 1187007080
    live_end = live_start + live_duration
    live_detectors = "H1+L1" if plan_config.get("live_l1_frame_file") else "H1"
    selected = ("Both executables" if scope == "both" else
                "pycbc_inspiral" if scope == "inspiral" else "pycbc_live")
    lines = ["JAX complete-search benchmark run", "=================================", "",
             f"Suite status: {state['status']}.  Analysis: 2048 Hz, complex64.", "",
             f"Selected scope: {selected}.  Preset: {preset}.",
             "Selected executable(s) compare pristine original CPU, branch ordinary CPU, JAX CPU and JAX CUDA.",
             "Full-process timing includes startup, conditioning, compilation, output and shutdown.",
             "Qualification and profiling processes do not contribute to timing medians.", "",
             "The exact input hashes, bank row selection, source patches, dependency versions and commands",
             "are retained alongside this report in inputs-manifest.json, provenance.json and plan.json.", ""]
    if preset == "quick":
        lines += ["Quick diagnostic", "----------------", "",
                  "This preset records one fresh timing sample per arm at the 32-template diagnostic bank size.",
                  "It is diagnostic-only: replication and convergence are not established, and no result",
                  "in this report is a publishable performance claim.", ""]
        if state.get("quick_profiles"):
            lines += ["Separate 32-template process and stage profiles were also captured for each arm.",
                      "Profile processes are excluded from the timing samples above.", ""]
    if state.get("allow_unqualified_timings"):
        lines += ["Known-divergence timing policy", "------------------------------", "",
                  "Scientific gates remain strict and their failures are reported below. Timings are descriptive",
                  "diagnostics only; they do not establish equivalent-output speedups or scientific qualification.", ""]
    if state.get("error"):
        lines += ["Failure::", "", *["   " + line for line in state["error"].splitlines()], ""]
    provenance_path = directory / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        for role in ("reference", "candidate"):
            identity = provenance[role]
            lines.append(f"* {role}: revision ``{identity.get('revision')}``, "
                         f"dirty={identity.get('dirty')}; diff SHA256 ``{identity.get('tracked_diff_sha256')}``.")
        lines += [""]
    samples = {}
    for step in state["steps"]:
        lines += [f"{step['kind']}: {step['templates']} templates, {step['mode']}",
                  "-" * 72, "", f"Execution status: {step.get('status', 'not started')}.", ""]
        if step["kind"] == "inspiral":
            description = ("IMRPhenomD stored single-precision compressed waveforms, inline_linear decompression; "
                "GPS input [1187007048,1187009080), requested trigger interval [1187007160,1187009064). "
                "512 s segments, 112/16 s start/end pads; CPU batch 1, JAX CPU batch 16, JAX CUDA batch 64.")
        else:
            description = (f"TaylorF2 generated templates; {live_detectors} replay "
                f"[{live_start},{live_end}) in 8 s chunks, {live_duration} s requested input; "
                + ("paced at real time." if step["kind"] == "live-paced"
                   else "unpaced throughput replay."))
        channel_description = ("H1:LOSC-STRAIN; L1:LOSC-STRAIN;" if step["kind"] != "inspiral" and live_detectors == "H1+L1"
                               else "H1:LOSC-STRAIN;")
        lines += [description, channel_description + " low-frequency cutoff 30 Hz; SNR threshold 5.5; chi-square 16 bins.",
                  "Valid detector-seconds below are measured; requested duration does not define completed work.", "",
                  "Command (argv)::", "", "   " + json.dumps(step["command"]), ""]
        receipt_path = Path(step["receipt"])
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        lines += [f"Receipt: ``{receipt_path}``.",
                  f"Scientific qualification passed: {science_passed(receipt)}.", ""]
        for arm, result in receipt.get("science", {}).items():
            lines.append(f"* {arm}: passed={result.get('passed')}; "
                         f"missing gates={result.get('missing_gates', [])}.")
        lines += [""]
        for figure in step.get("figures", []):
            relative = os.path.relpath(figure, directory)
            lines += [f".. image:: {relative}", "   :width: 100%", ""]
        if (step["mode"] == "timing" and
                (science_passed(receipt) or
                 known_divergence_complete(receipt, step) or
                 quick_diagnostic_complete(receipt, step))):
            rows = campaign_rows(receipt)
            heading = ("Single-sample diagnostic throughput"
                       if preset == "quick" else
                       "Capacity (templates/core for CPU, templates/GPU for CUDA)")
            lines += [heading + "::", ""]
            lines += ["   " + line for line in render_campaign_report(receipt).splitlines()]
            for row in rows:
                lines.append(f"   {row['arm']}: valid detector-seconds={row['valid_detector_seconds']}; "
                             f"completed template-seconds={row['work']}; wall samples={row['wall_seconds']}")
            lines += [""]
            if preset != "quick":
                samples.setdefault(step["kind"], []).append((step["templates"], rows))
    lines += ["Workload scaling", "----------------", ""]
    for kind in ("inspiral", "live-unpaced"):
        if scope == "inspiral" and kind != "inspiral":
            continue
        if scope == "live" and kind == "inspiral":
            continue
        entries = sorted(samples.get(kind, []))
        for arm in (INSPIRAL_ARMS if kind == "inspiral" else LIVE_ARMS):
            values = [(size, row) for size, rows in entries for row in rows
                      if row["arm"] == arm and row["measured"] and row["science"].get("passed")]
            converged = False
            changes = []
            if len(values) >= 3:
                values = values[-3:]
                changes = [abs(b["wall_capacity"][0] / a["wall_capacity"][0] - 1)
                           for (_, a), (_, b) in zip(values, values[1:])]
                converged = (all(b[0] == 2 * a[0] for a, b in zip(values, values[1:]))
                    and len({row["valid_detector_seconds"] for _, row in values}) == 1
                    and all(change < .05 for change in changes))
            label = ("bank-size criterion met at fixed duration; finite-duration result"
                     if converged else "finite-workload; convergence not established")
            lines += [f"* {kind}, {arm}: {label}; relative changes={changes}."]
    lines += ["", "Paced replay reports latency/deadlines separately and does not establish throughput convergence.", ""]
    (directory / "benchmark-report.rst").write_text("\n".join(lines))


def execute_suite(config, phase, scope="both", allow_unqualified_timings=False,
                  preset="full", quick_profiles=False):
    if Path(config["candidate_source"]) != ROOT:
        raise ValueError("Run the driver from the configured candidate checkout")
    validate_reference(Path(config["reference_source"]), REFERENCE)
    manifest_path = Path(config["inputs_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_inputs(config, manifest)
    directory_name = "quick" if phase == "run" and preset == "quick" else phase
    directory = Path(config["output_dir"]) / directory_name
    directory.mkdir(parents=True, exist_ok=False)
    plan = make_plan(config, phase, scope, allow_unqualified_timings, preset,
                     quick_profiles)
    write_json(directory / "plan.json", plan)
    write_json(directory / "inputs-manifest.json", manifest)
    frozen = capture_provenance(config, directory)
    state = {"status": "running", "scope": scope, "preset": preset,
             "quick_profiles": bool(quick_profiles),
             "performance_claim": False if preset == "quick" else None,
             "publishable": False if preset == "quick" else None,
             "diagnostic_only": preset == "quick",
             "allow_unqualified_timings": bool(allow_unqualified_timings),
             "steps": copy.deepcopy(plan["steps"])}
    write_json(directory / "suite.json", state)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    science_failures = []
    try:
        for index, step in enumerate(state["steps"]):
            if (science_failures and not allow_unqualified_timings
                    and step["mode"] != "qualification"):
                raise RuntimeError("Scientific qualification failed; no timing or profiling launched: "
                                   + ", ".join(science_failures))
            validate_inputs(config, manifest)
            for role in ("reference", "candidate"):
                if source_identity(Path(config[role + "_source"])) != frozen[role]:
                    raise ValueError(f"{role} source changed during suite")
            for name, content in step["generated_files"].items():
                write_json(name, content)
            step["status"] = "running"
            write_json(directory / "suite.json", state)
            print(f"[{index+1}/{len(state['steps'])}] {step['kind']} {step['templates']} {step['mode']}", flush=True)
            with (directory / f"step-{index+1:02d}.log").open("w") as log:
                process = subprocess.run(step["command"], env=env, stdout=log,
                                         stderr=subprocess.STDOUT, check=False)
            if not Path(step["receipt"]).is_file():
                raise RuntimeError("Campaign receipt missing: " + step["receipt"])
            receipt = json.loads(Path(step["receipt"]).read_text())
            validate_inputs(config, manifest)
            for role in ("reference", "candidate"):
                if source_identity(Path(config[role + "_source"])) != frozen[role]:
                    raise ValueError(f"{role} source changed during suite")
            diagnostic_complete = (
                allow_unqualified_timings
                and known_divergence_complete(receipt, step)
            )
            quick_complete = (
                preset == "quick" and quick_diagnostic_complete(receipt, step)
            )
            if (process.returncode != 0
                    and not completed_science_failure(receipt, step)
                    and not diagnostic_complete
                    and not quick_complete):
                raise subprocess.CalledProcessError(process.returncode, step["command"])
            if preset == "quick" and step["mode"] == "timing":
                if not quick_complete:
                    raise ValueError("Quick campaign did not produce a complete diagnostic receipt: "
                                     + step["receipt"])
                step["status"] = "complete_quick_diagnostic"
                step["figures"] = []
                step["failed_science_arms"] = [
                    arm for arm, result in receipt["science"].items()
                    if result.get("passed") is False]
                write_json(directory / "suite.json", state)
                render_report(directory)
                continue
            if not science_passed(receipt):
                if diagnostic_complete:
                    step["status"] = "complete_known_divergence"
                    step["figures"] = render_figures(step)
                    failed = [arm for arm, result in receipt["science"].items()
                              if result.get("passed") is False]
                    step["failed_science_arms"] = failed
                    write_json(directory / "suite.json", state)
                    render_report(directory)
                    continue
                if not completed_science_failure(receipt, step):
                    raise ValueError("Campaign did not pass scientific qualification: " + step["receipt"])
                step["status"] = "failed"
                step["failure"] = receipt.get("failure", {
                    "phase": "qualification_science", "error": "scientific qualification failed"})
                science_failures.append(f"{step['kind']} {step['templates']} templates")
                write_json(directory / "suite.json", state)
                render_report(directory)
                continue
            step["figures"] = render_figures(step)
            step["status"] = "passed"
            write_json(directory / "suite.json", state)
            render_report(directory)
        if science_failures and not allow_unqualified_timings:
            raise RuntimeError("Scientific qualification failed; no timing or profiling launched: "
                               + ", ".join(science_failures))
        state["status"] = (
            "complete_quick_diagnostic" if preset == "quick" else
            "complete_known_divergence" if allow_unqualified_timings else
            "complete")
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        for step in state["steps"]:
            if step.get("status") == "running":
                step["status"] = "failed"
        raise
    finally:
        write_json(directory / "suite.json", state)
        render_report(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "prepare", "qualify", "run", "report"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scope", choices=("both", "inspiral", "live"),
                        default="both")
    parser.add_argument(
        "--preset", choices=("quick", "full"), default="full",
        help=("quick runs one diagnostic sample per arm at the 32-template "
              "diagnostic size; full runs replicated scaling, paced replay "
              "and profiles"),
    )
    parser.add_argument(
        "--allow-unqualified-timings", action="store_true",
        help=("continue with clearly labelled known-divergence diagnostics "
              "when strict scientific qualification fails"),
    )
    parser.add_argument(
        "--quick-profiles", action="store_true",
        help=("with the quick run preset, add separate 32-template process "
              "and stage profiles that are excluded from timing samples"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.phase == "plan":
        plan = make_plan(config, "run", args.scope,
                         args.allow_unqualified_timings, args.preset,
                         args.quick_profiles)
        plan["prepare_command"] = preparation_command(config)
        print(json.dumps(plan, indent=2))
    elif args.phase == "prepare":
        subprocess.run(preparation_command(config), check=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    elif args.phase == "report":
        directory = Path(config["output_dir"]) / (
            "quick" if args.preset == "quick" else "run")
        state = json.loads((directory / "suite.json").read_text())
        reconcile_science_failure(state)
        if args.preset == "quick":
            reconcile_quick_diagnostics(state)
        for step in state["steps"]:
            if step.get("status") in ("passed", "complete_known_divergence",
                                       "complete_quick_diagnostic"):
                step["figures"] = (
                    [] if args.preset == "quick" and step["mode"] == "timing"
                    else render_figures(step)
                )
        write_json(directory / "suite.json", state)
        render_report(directory)
    else:
        execute_suite(config, args.phase, args.scope,
                      args.allow_unqualified_timings, args.preset,
                      args.quick_profiles)


if __name__ == "__main__":
    main()
