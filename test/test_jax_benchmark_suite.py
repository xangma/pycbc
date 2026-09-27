"""Tests for the reproducibility suite driver using synthetic inputs."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


_PATH = Path(__file__).parents[1] / "tools" / "run_jax_benchmarks.py"
sys.path.insert(0, str(_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("run_jax_benchmarks", _PATH)
suite = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(suite)


def _config(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    frame = tmp_path / "frame.gwf"
    frame.write_bytes(b"synthetic frame")
    l1_frame = tmp_path / "l1-frame.gwf"
    l1_frame.write_bytes(b"synthetic l1 frame")
    return {
        "reference_source": str(tmp_path / "reference"),
        "candidate_source": str(suite.ROOT),
        "python": sys.executable,
        "inputs_dir": str(inputs),
        "frame_file": str(frame),
        "live_l1_frame_file": str(l1_frame),
        "output_dir": str(tmp_path / "outputs"),
        "frame_sha256": "a" * 64,
        "live_l1_frame_sha256": "c" * 64,
        "bank_xml_sha256": "b" * 64,
        "reference_revision": suite.REFERENCE,
        "sizes": [1536, 3072, 6144],
        "qualification_size": 32,
        "replicates": 3,
        "inspiral_affinity": "0",
        "live_affinity": "1",
        "live_ranks": 1,
    }


def _write_config(tmp_path):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(_config(tmp_path)))
    return path


def test_load_config_rejects_unknown_precision_or_path_fields(tmp_path):
    path = _write_config(tmp_path)
    payload = json.loads(path.read_text())
    payload["precision"] = "float64"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        suite.load_config(path)

    payload.pop("precision")
    payload["unknown_path"] = "/tmp/input"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        suite.load_config(path)


def test_load_config_requires_pinned_hashes_and_explicit_benchmark_constants(tmp_path):
    path = _write_config(tmp_path)
    payload = json.loads(path.read_text())
    payload["frame_sha256"] = "bad"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frame_sha256"):
        suite.load_config(path)

    payload["frame_sha256"] = "a" * 64
    payload["qualification_size"] = 31
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="32 distinct templates"):
        suite.load_config(path)


def test_live_config_contains_both_frames_and_background_estimation(tmp_path):
    config = suite.load_config(_write_config(tmp_path))
    generated = suite.live_config(config, 32)
    assert generated["frame_files"] == [config["frame_file"], config["live_l1_frame_file"]]
    args = generated["args"]
    assert args.count("--frame-src") == 1
    assert "L1:" + config["live_l1_frame_file"] in args
    assert args.count("--channel-name") == 1
    assert "L1:LOSC-STRAIN" in args
    assert "--enable-background-estimation" in args
    for option in ("--ifar-double-followup-threshold", "--ifar-upload-threshold"):
        assert args[args.index(option) + 1] == "1e9"


def test_validate_inputs_checks_live_l1_frame_hash(tmp_path, monkeypatch):
    config = suite.load_config(_write_config(tmp_path))
    expected_l1 = config["live_l1_frame_sha256"]
    monkeypatch.setattr(suite, "sha256", lambda path: {
        Path(config["frame_file"]): config["frame_sha256"],
        Path(config["live_l1_frame_file"]): expected_l1,
    }.get(Path(path), "x"))
    manifest = _manifest(config)
    config["live_l1_frame_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="Live L1 frame checksum"):
        suite.validate_inputs(config, manifest)


def test_load_config_requires_l1_hash_when_l1_frame_is_configured(tmp_path):
    path = _write_config(tmp_path)
    payload = json.loads(path.read_text())
    payload.pop("live_l1_frame_sha256")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="live_l1_frame_sha256"):
        suite.load_config(path)


def test_plan_has_four_arms_for_both_executables_and_all_modes(tmp_path):
    config = suite.load_config(_write_config(tmp_path))
    plan = suite.make_plan(config, "run")
    assert plan["sample_rate"] == 2048
    assert plan["precision"] == "complex64"
    assert plan["convergence"]
    assert {step["kind"] for step in plan["steps"]} == {
        "inspiral", "live-unpaced", "live-paced"
    }
    assert {step["mode"] for step in plan["steps"]} == {
        "qualification", "timing", "profile"
    }
    for step in plan["steps"]:
        command = step["command"]
        if step["mode"] == "qualification":
            assert "--qualification-only" in command
        if step["mode"] == "profile":
            assert "--profile-utilization" in command
        if step["kind"] == "inspiral":
            assert command.count("--arms") == 1
            assert command[command.index("--arms") + 1:command.index("--affinity")] == suite.INSPIRAL_ARMS
        else:
            assert command[command.index("--arms") + 1:command.index("--ranks")] == suite.LIVE_ARMS
    assert all(Path(step["command"][1]).name.startswith("bench_jax_")
               for step in plan["steps"])


def test_known_divergence_plan_marks_every_campaign(tmp_path):
    config = suite.load_config(_write_config(tmp_path))
    plan = suite.make_plan(config, "run", allow_unqualified_timings=True)
    assert plan["allow_unqualified_timings"] is True
    assert all("--allow-unqualified-timings" in step["command"]
               for step in plan["steps"])


def test_known_divergence_completion_does_not_pass_science():
    step = {"kind": "inspiral", "mode": "timing"}
    receipt = {
        "timing_policy": {"allow_unqualified_timings": True},
        "campaign": {"process_complete": True, "passed": False,
                     "performance_claim": False},
        "science": {arm: {"passed": arm == "original_cpu"}
                    for arm in suite.INSPIRAL_ARMS},
        "raw_results": {arm: [{"elapsed_wall_sec": 1.0}]
                        for arm in suite.INSPIRAL_ARMS},
    }
    assert suite.known_divergence_complete(receipt, step)
    assert not suite.science_passed(receipt)


@pytest.mark.parametrize("scope", ["inspiral", "live"])
@pytest.mark.parametrize("phase", ["qualify", "run"])
def test_selected_scope_filters_plan_without_changing_pinned_arms(
        tmp_path, scope, phase):
    config = suite.load_config(_write_config(tmp_path))
    whole = suite.make_plan(config, phase)
    selected = suite.make_plan(config, phase, scope)
    assert selected["scope"] == scope
    expected = [step for step in whole["steps"] if
                (step["kind"] == "inspiral") == (scope == "inspiral")]
    assert selected["steps"] == expected
    assert selected["steps"]
    assert selected["steps"][0]["mode"] == "qualification"
    if phase == "run":
        assert {step["mode"] for step in selected["steps"]} == {
            "qualification", "timing", "profile"}


def test_selected_scope_rejects_unknown_name(tmp_path):
    config = suite.load_config(_write_config(tmp_path))
    with pytest.raises(ValueError, match="scope must be"):
        suite.make_plan(config, "qualify", "unknown")


@pytest.mark.parametrize("scope, expected_kind", [
    ("inspiral", "inspiral"),
    ("live", "live-unpaced"),
])
def test_selected_qualification_records_only_requested_code(
        tmp_path, monkeypatch, scope, expected_kind):
    config = suite.load_config(_write_config(tmp_path))
    (Path(config["inputs_dir"]) / "manifest.json").write_text("{}")
    identity = {"revision": "fixed", "dirty": False}
    monkeypatch.setattr(suite, "validate_reference", lambda *a: None)
    monkeypatch.setattr(suite, "validate_inputs", lambda *a: None)
    monkeypatch.setattr(suite, "source_identity", lambda *a: identity)
    monkeypatch.setattr(suite, "capture_provenance", lambda *a: {
        "reference": identity, "candidate": identity})
    monkeypatch.setattr(suite, "render_report", lambda *a: None)
    monkeypatch.setattr(suite, "render_figures", lambda *a: [])
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        receipt_path = Path(command[command.index("--output") + 1])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        arms = (suite.INSPIRAL_ARMS if scope == "inspiral"
                else suite.LIVE_ARMS)
        receipt_path.write_text(json.dumps({
            "executable": ("pycbc_inspiral" if scope == "inspiral"
                           else "pycbc_live"),
            "status": "science_qualification_only",
            "science": {arm: {"passed": True} for arm in arms},
            "campaign": {"passed": True},
        }))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    suite.execute_suite(config, "qualify", scope)
    directory = Path(config["output_dir"]) / "qualify"
    state = json.loads((directory / "suite.json").read_text())
    plan = json.loads((directory / "plan.json").read_text())
    assert state["scope"] == plan["scope"] == scope
    assert state["status"] == "complete"
    assert len(calls) == 1
    assert len(state["steps"]) == 1
    assert state["steps"][0]["kind"] == expected_kind


def test_plan_is_read_only_and_does_not_launch_benchmarks(tmp_path, monkeypatch, capsys):
    path = _write_config(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                        (_ for _ in ()).throw(AssertionError("benchmark launched")))
    monkeypatch.setattr(sys, "argv", ["run_jax_benchmarks.py", "plan", "--config", str(path)])
    suite.main()
    assert json.loads(capsys.readouterr().out)["phase"] == "run"


def test_plan_cli_accepts_selected_scope(tmp_path, monkeypatch, capsys):
    path = _write_config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_jax_benchmarks.py", "plan",
                                      "--config", str(path), "--scope", "live"])
    suite.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["scope"] == "live"
    assert {step["kind"] for step in plan["steps"]} == {
        "live-unpaced", "live-paced"}


def test_scientific_failure_stops_later_steps_and_saves_failed_status(tmp_path, monkeypatch):
    config = suite.load_config(_write_config(tmp_path))
    manifest_path = Path(config["inputs_dir"]) / "manifest.json"
    manifest_path.write_text("{}")
    reference_identity = {"repository": config["reference_source"],
                         "revision": suite.REFERENCE, "dirty": False}
    monkeypatch.setattr(suite, "validate_reference", lambda *a: suite.REFERENCE)
    monkeypatch.setattr(suite, "validate_inputs", lambda *a: None)
    monkeypatch.setattr(suite, "sha256", lambda path: config["frame_sha256"])
    monkeypatch.setattr(suite, "source_identity",
                        lambda root, *args: reference_identity if str(root) == config["reference_source"]
                        else {"repository": str(root), "revision": "candidate", "dirty": False})
    monkeypatch.setattr(suite, "capture_provenance",
                        lambda cfg, directory: {"reference": reference_identity,
                                                "candidate": suite.source_identity(suite.ROOT)})
    monkeypatch.setattr(suite, "render_report", lambda directory: None)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        receipt = Path(command[command.index("--output") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"status": "science_qualification_only",
                                       "science": {"original_cpu": {"passed": False}}}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="Campaign did not pass scientific qualification"):
        suite.execute_suite(config, "qualify")
    state = json.loads((Path(config["output_dir"]) / "qualify" / "suite.json").read_text())
    assert state["status"] == "failed"
    assert len(calls) == 1


def test_complete_science_failures_collect_both_executables_before_timing(
        tmp_path, monkeypatch):
    config = suite.load_config(_write_config(tmp_path))
    (Path(config["inputs_dir"]) / "manifest.json").write_text("{}")
    identities = {
        "reference": {"repository": config["reference_source"],
                      "revision": suite.REFERENCE, "dirty": False},
        "candidate": {"repository": config["candidate_source"],
                      "revision": "candidate", "dirty": True},
    }
    monkeypatch.setattr(suite, "validate_reference", lambda *a: suite.REFERENCE)
    monkeypatch.setattr(suite, "validate_inputs", lambda *a: None)
    monkeypatch.setattr(suite, "source_identity", lambda root: identities[
        "reference" if str(root) == config["reference_source"] else "candidate"])
    monkeypatch.setattr(suite, "capture_provenance", lambda *a: identities)
    monkeypatch.setattr(suite, "render_report", lambda *a: None)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        receipt_path = Path(command[command.index("--output") + 1])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        if "inspiral" in Path(command[1]).name:
            arms = suite.INSPIRAL_ARMS
            receipt = {
                "executable": "pycbc_inspiral", "status": "failed",
                "qualification": {
                    arm: {"qualification_run": True,
                          "triggers_path": f"{arm}/triggers.hdf",
                          "evidence_path": f"{arm}/science.hdf"}
                    for arm in arms},
                "science": {arm: {"passed": arm != "jax_cpu_batched"}
                            for arm in arms},
                "failure": {"phase": "qualification_science",
                            "arms": ["jax_cpu_batched"]},
            }
        else:
            arms = suite.LIVE_ARMS
            receipt = {
                "executable": "pycbc_live",
                "qualification_results": {
                    arm: [{"returncode": 0, "process_complete": True}]
                    for arm in arms},
                "science": {arm: {"passed": arm != "jax_cuda"}
                            for arm in arms},
                "campaign": {"process_complete": True, "passed": False},
            }
        receipt_path.write_text(json.dumps(receipt))
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="no timing or profiling launched"):
        suite.execute_suite(config, "run")
    state = json.loads((Path(config["output_dir"]) / "run" / "suite.json").read_text())
    assert state["status"] == "failed"
    assert [step.get("status") for step in state["steps"][:2]] == ["failed", "failed"]
    assert all("status" not in step for step in state["steps"][2:])
    assert len(calls) == 2
    assert {Path(command[1]).name for command in calls} == {
        "bench_jax_inspiral_campaign.py", "bench_jax_live_campaign.py"}


def test_incomplete_live_qualification_is_not_a_science_only_failure():
    receipt = {
        "executable": "pycbc_live",
        "qualification_results": {
            arm: [{"returncode": 0, "process_complete": True}]
            for arm in suite.LIVE_ARMS},
        "science": {arm: {"passed": arm != "jax_cuda"}
                    for arm in suite.LIVE_ARMS},
        "campaign": {"process_complete": True, "passed": False},
    }
    step = {"kind": "live-unpaced", "mode": "qualification"}
    assert suite.completed_science_failure(receipt, step)
    receipt["qualification_results"]["jax_cuda"][0]["process_complete"] = False
    assert not suite.completed_science_failure(receipt, step)


def test_report_reconciles_resumed_science_failure_without_running_timing(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    directory = tmp_path / "outputs" / "run"
    directory.mkdir(parents=True)
    receipt_path = directory / "inspiral-32-qualification" / "campaign.json"
    receipt_path.parent.mkdir()
    receipt_path.write_text(json.dumps({
        "executable": "pycbc_inspiral", "status": "failed",
        "qualification": {arm: {} for arm in suite.INSPIRAL_ARMS},
        "science": {arm: {"passed": arm not in {"jax_cpu_batched", "jax_cuda_batched"}}
                    for arm in suite.INSPIRAL_ARMS},
        "failure": {"phase": "qualification_science",
                    "arms": ["jax_cpu_batched", "jax_cuda_batched"],
                    "error": "Scientific qualification failed; no timing runs launched"},
    }))
    state = {"status": "failed", "error": "old CUDA cuFFT subprocess error", "steps": [
        {"kind": "inspiral", "templates": 32, "mode": "qualification",
         "status": "failed", "command": ["synthetic-test-only"],
         "receipt": str(receipt_path)},
        {"kind": "inspiral", "templates": 1536, "mode": "timing",
         "command": ["synthetic-test-only"],
         "receipt": str(directory / "inspiral-1536-timing" / "campaign.json")},
    ]}
    suite.write_json(directory / "suite.json", state)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                        (_ for _ in ()).throw(AssertionError("benchmark launched")))
    monkeypatch.setattr(sys, "argv", ["run_jax_benchmarks.py", "report",
                                      "--config", str(config_path)])
    suite.main()
    reconciled = json.loads((directory / "suite.json").read_text())
    report = (directory / "benchmark-report.rst").read_text()
    assert reconciled["status"] == "failed"
    assert reconciled["steps"][0]["status"] == "failed"
    assert "status" not in reconciled["steps"][1]
    assert reconciled["steps"][0]["prior_execution_error"] == "old CUDA cuFFT subprocess error"
    assert "qualification_science" in str(reconciled["steps"][0]["failure"])
    assert "Scientific qualification failed" in reconciled["error"]
    assert "qualification_science" in reconciled["error"]
    assert "old CUDA cuFFT" not in reconciled["error"]
    assert "old CUDA cuFFT" not in report
    assert "qualification_science" in report
    assert "Scientific qualification failed" in report
    assert "Execution status: not started" in report


def test_report_keeps_original_error_for_incomplete_science_receipt(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    directory = tmp_path / "outputs" / "run"
    directory.mkdir(parents=True)
    receipt_path = directory / "campaign.json"
    receipt_path.write_text(json.dumps({
        "executable": "pycbc_inspiral", "status": "failed",
        "qualification": {"original_cpu": {}},
        "science": {"jax_cuda_batched": {"passed": False}},
        "failure": {"phase": "qualification_science", "arms": ["jax_cuda_batched"]},
    }))
    original = "CUDA cuFFT subprocess error"
    suite.write_json(directory / "suite.json", {"status": "failed", "error": original,
        "steps": [{"kind": "inspiral", "templates": 32, "mode": "qualification",
                   "status": "failed", "command": ["synthetic-test-only"],
                   "receipt": str(receipt_path)}]})
    monkeypatch.setattr(sys, "argv", ["run_jax_benchmarks.py", "report",
                                      "--config", str(config_path)])
    suite.main()
    assert json.loads((directory / "suite.json").read_text())["error"] == original


def _manifest(config):
    banks = {}
    for size in [32, *config["sizes"]]:
        banks[str(size)] = {
            "source_rows": list(range(size)),
            "subset_sha256": "s" * 64,
            "compressed_sha256": "c" * 64,
        }
    return {
        "reference_revision": suite.REFERENCE,
        "inputs": {"frame_sha256": config["frame_sha256"],
                    "bank_xml_sha256": config["bank_xml_sha256"]},
        "banks": banks,
    }


def test_validate_inputs_rejects_hash_mismatch_and_row_nesting(tmp_path, monkeypatch):
    config = suite.load_config(_write_config(tmp_path))
    for size in [32, *config["sizes"]]:
        for stem in ("o2-subset", "o2-compressed"):
            (Path(config["inputs_dir"]) / f"{stem}-{size}.hdf").write_bytes(b"bank")
    monkeypatch.setattr(suite, "sha256", lambda path: (
        config["frame_sha256"] if Path(path) == Path(config["frame_file"])
        else "s" * 64 if "subset" in str(path) else "c" * 64))
    manifest = _manifest(config)
    bad_hash = json.loads(json.dumps(manifest))
    bad_hash["inputs"]["frame_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="Input manifest mismatch"):
        suite.validate_inputs(config, bad_hash)

    bad_rows = json.loads(json.dumps(manifest))
    bad_rows["banks"]["3072"]["source_rows"] = list(range(1, 3073))
    with pytest.raises(ValueError, match="nested distinct subsets"):
        suite.validate_inputs(config, bad_rows)


def test_render_figures_requires_all_profiles_and_stage_events(tmp_path, monkeypatch):
    receipt_path = tmp_path / "campaign.json"
    receipt_path.write_text(json.dumps({
        "executable": "pycbc_live", "campaign": {"passed": True},
        "science": {arm: {"passed": True} for arm in suite.LIVE_ARMS},
        "profile_results": {"cpu": {"timeline_path": "unused"}},
    }))
    step = {"kind": "live-unpaced", "mode": "profile", "receipt": str(receipt_path)}
    with pytest.raises(ValueError, match="Missing profile arms"):
        suite.render_figures(step)

    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    profiles = {}
    for arm in suite.LIVE_ARMS:
        timeline = profile_dir / f"{arm}.json"
        timeline.write_text(json.dumps({
            "returncode": 0, "telemetry": [{"timestamp": 1}],
            "stage_events": [], "process_tree": {
                "completion": {"all_observed_processes_exited": True}},
        }))
        profiles[arm] = {"timeline_path": str(timeline)}
    receipt_path.write_text(json.dumps({
        "executable": "pycbc_live", "campaign": {"passed": True},
        "science": {arm: {"passed": True} for arm in suite.LIVE_ARMS},
        "profile_results": profiles,
    }))
    with pytest.raises(ValueError, match="Incomplete process/stage profile"):
        suite.render_figures(step)


def test_report_preserves_measured_resources_and_finite_workload_label(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    directory.mkdir()
    receipt = directory / "campaign.json"
    receipt.write_text(json.dumps({"status": "complete", "science": {
        arm: {"passed": True} for arm in suite.INSPIRAL_ARMS}}))
    state = {"status": "complete", "steps": [{
        "kind": "inspiral", "templates": 1536, "mode": "timing",
        "status": "passed", "command": ["python", "bench_jax_inspiral_campaign.py"],
        "receipt": str(receipt), "figures": [],
    }]}
    (directory / "suite.json").write_text(json.dumps(state))
    try:
        from tools import plot_jax_search_capacity as plots
    except ModuleNotFoundError:
        import plot_jax_search_capacity as plots
    monkeypatch.setattr(plots, "campaign_rows", lambda _: [{
        "arm": "original_cpu", "valid_detector_seconds": 8.0,
        "work": 123.0, "wall_seconds": [2.0], "measured": True,
        "science": {"passed": True}, "wall_capacity": (61.5,),
    }])
    monkeypatch.setattr(plots, "render_campaign_report",
                        lambda _: "measured work=123 resources=4")
    suite.render_report(directory)
    report = (directory / "benchmark-report.rst").read_text()
    assert "measured work=123 resources=4" in report
    assert "finite-workload; convergence not established" in report


def test_inspiral_profile_failure_and_missing_cpu_cannot_pass_suite():
    receipt = {
        "executable": "pycbc_inspiral", "status": "profiled_science_qualification",
        "science": {arm: {"passed": True} for arm in suite.INSPIRAL_ARMS},
        "campaign": {"passed": False},
    }
    assert not suite.science_passed(receipt)
    receipt["campaign"]["passed"] = True
    assert suite.science_passed(receipt)
    del receipt["science"]["branch_cpu"]
    assert not suite.science_passed(receipt)


def test_real_report_and_capacity_plot_keep_four_arms_and_finite_duration(tmp_path):
    steps = []
    for size in (1536, 3072, 6144):
        path = tmp_path / f"inspiral-{size}" / "campaign.json"
        path.parent.mkdir()
        receipt = {
            "executable": "pycbc_inspiral", "status": "complete",
            "sample_rate": 2048, "precision": "complex64",
            "workload": {"templates": size, "valid_detector_seconds": 8},
            "science": {arm: {"passed": True} for arm in suite.INSPIRAL_ARMS},
            "raw_results": {arm: [{
                "elapsed_wall_sec": size / 32, "completed_template_seconds": size * 8,
                "resources": {"physical_cpu_cores": 2, "gpus": int("cuda" in arm)},
            } for _ in range(3)] for arm in suite.INSPIRAL_ARMS},
        }
        path.write_text(json.dumps(receipt))
        steps.append({"kind": "inspiral", "templates": size, "mode": "timing",
                      "status": "passed", "command": ["synthetic-test-only"],
                      "receipt": str(path)})
    steps[0]["figures"] = suite.render_figures(steps[0])
    assert Path(steps[0]["figures"][0]).read_bytes().startswith(b"\x89PNG")
    (tmp_path / "suite.json").write_text(json.dumps({"status": "complete", "steps": steps}))
    suite.render_report(tmp_path)
    report = (tmp_path / "benchmark-report.rst").read_text()
    assert all(arm in report for arm in suite.INSPIRAL_ARMS)
    assert "1 GPU + 2 host cores" in report
    assert "2 physical CPU cores" in report
    assert "bank-size criterion met at fixed duration; finite-duration result" in report
    docutils = pytest.importorskip("docutils.core")
    docutils.publish_doctree(report, settings_overrides={"halt_level": 2, "report_level": 2})
