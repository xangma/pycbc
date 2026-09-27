"""Unit tests for the full-executable pycbc_live campaign helpers."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).parents[1] / "tools" / "bench_jax_live_campaign.py"
_SPEC = importlib.util.spec_from_file_location("bench_jax_live_campaign", _MODULE_PATH)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)
_SCIENCE_SPEC = importlib.util.spec_from_file_location(
    "benchmark_science", Path(__file__).parents[1] / "tools" / "benchmark_science.py")
science = importlib.util.module_from_spec(_SCIENCE_SPEC)
_SCIENCE_SPEC.loader.exec_module(science)


def _contract_fixture(tmp_path, rows=((1.0, 2.0, 0.1, -0.2),
                                      (3.0, 4.0, 0.3, -0.4))):
    h5py = pytest.importorskip("h5py")
    bank = tmp_path / "bank.hdf"
    frame = tmp_path / "frame.gwf"
    with h5py.File(bank, "w") as handle:
        handle.create_dataset("mass1", data=[row[0] for row in rows])
        handle.create_dataset("mass2", data=[row[1] for row in rows])
        handle.create_dataset("spin1z", data=[row[2] for row in rows])
        handle.create_dataset("spin2z", data=[row[3] for row in rows])
    frame.write_bytes(b"frame")
    return {
        "workload": {"templates": len(rows), "analysis_chunk_sec": 8,
                     "start_time": 1, "end_time": 9},
        "science": {"sample_rate": 2048, "precision": "complex64"},
        "args": ["--sample-rate", "2048", "--bank-file", str(bank),
                 "--frame-src", f"H1:{frame}", "--analysis-chunk", "8",
                 "--start-time", "1", "--end-time", "9"],
    }


def _write_main_config(tmp_path):
    config = _contract_fixture(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return config, path


def _mock_reference(monkeypatch, tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir(exist_ok=True)
    revision = "a" * 40
    monkeypatch.setattr(bench, "validate_reference", lambda root, expected: revision)
    monkeypatch.setattr(bench, "validate_source", lambda path: None)
    monkeypatch.setattr(bench, "source_identity", lambda root, files: {
        "repository": str(Path(root).resolve()), "revision": revision,
        "dirty": False, "files_sha256": {},
    })
    return reference, ["--reference-source-root", str(reference),
                       "--reference-revision", revision]


def _decorate_result(config, result):
    result.setdefault("workload_digest", bench.workload_digest(config))
    result.setdefault("inputs_stable", True)
    result.setdefault("filter_contract", {
        "passed": True, "count": 1, "sample_rates": [2048.0],
        "signal_dtypes": [["complex64"]],
    })
    return result


def test_launch_resolves_output_before_changing_process_directory(tmp_path, monkeypatch):
    config = _contract_fixture(tmp_path)
    executable = tmp_path / 'source/bin/pycbc_live'
    executable.parent.mkdir(parents=True)
    executable.write_text('pass\n')
    monkeypatch.chdir(tmp_path)

    class LaunchObserved(Exception):
        pass

    def inspect_launch(command, *, cwd, env, **kwargs):
        output = str(tmp_path / 'relative/runs/cpu')
        assert cwd == output
        assert command[command.index('--output-path') + 1] == output
        assert str(executable) in command
        assert env['PYTHONPATH'] == str(tmp_path / 'source')
        raise LaunchObserved

    monkeypatch.setattr(bench.subprocess, 'Popen', inspect_launch)
    with pytest.raises(LaunchObserved):
        bench.run_one(config, 'cpu', Path('relative/runs/cpu'), Path('source'),
                      'python', 2, 'unpaced', 1.0, 2, 0,
                      expected_digest=bench.workload_digest(config)['sha256'])


def test_stage_events_pair_root_and_report_tail_percentiles():
    lines = """
PYCBC_STAGE_EVENT {"event":"start","stage":"block","rank":1,"pid":9,"monotonic_ns":10,"data_start":100}
PYCBC_STAGE_EVENT {"event":"start","stage":"block","rank":0,"pid":8,"monotonic_ns":20,"data_start":100}
PYCBC_STAGE_EVENT {"event":"end","stage":"block","rank":1,"pid":9,"monotonic_ns":30,"data_start":100,"data_end":108}
PYCBC_STAGE_EVENT {"event":"end","stage":"block","rank":0,"pid":8,"monotonic_ns":1020,"data_start":100,"data_end":108,"elapsed_sec":1.0,"deadline_met":true,"live_detectors":["H1"]}
"""
    metrics = bench.block_metrics(bench.parse_stage_events(lines), 8)
    assert metrics["latency"]["p50"] == 1.0
    assert metrics["deadline_miss_count"] == 0
    assert metrics["valid_detector_intervals"] == [{
        "start": 100.0, "end": 108.0, "duration": 8.0,
        "live_detectors": ["H1"],
    }]


def test_block_metrics_requires_root_pairs_and_rejects_invalid_intervals():
    worker_only = [
        {"name": "block", "event": "start", "rank": 1, "pid": 4,
         "time_ns": 0, "data_start": 100},
        {"name": "block", "event": "end", "rank": 1, "pid": 4,
         "time_ns": 1, "data_start": 100, "data_end": 108,
         "processing_elapsed_sec": 1.0},
    ]
    metrics = bench.block_metrics(worker_only, 8)
    assert metrics["block_count"] == 0
    assert metrics["root_block_events"] is False

    invalid = [
        {"name": "block", "event": "start", "rank": 0, "pid": 4,
         "time_ns": 0, "data_start": 100},
        {"name": "block", "event": "end", "rank": 0, "pid": 4,
         "time_ns": 1, "data_start": 100, "data_end": 99,
         "processing_elapsed_sec": 1.0},
    ]
    assert bench.block_metrics(invalid, 8)["invalid_block_count"] == 1


def test_block_metrics_rejects_unpaired_end_markers():
    events = [{"name": "block", "event": "end", "rank": 0, "pid": 4,
               "time_ns": 1, "data_start": 100, "data_end": 108,
               "processing_elapsed_sec": 1.0}]
    assert bench.block_metrics(events, 8)["unpaired_block_count"] == 1


def test_block_metrics_retains_malformed_pair_as_invalid():
    events = [
        {"name": "block", "event": "start", "rank": 0, "pid": 4,
         "data_start": 100},
        {"name": "block", "event": "end", "rank": 0, "pid": 4,
         "data_start": 100, "data_end": 108},
    ]
    assert bench.block_metrics(events, 8)["invalid_block_count"] == 1


def test_build_command_replaces_scheme_and_output(tmp_path):
    config = {"args": ["--bank-file", "real-bank.hdf", "--processing-scheme", "cpu:4"]}
    command = bench.build_live_command(config, "jax_cpu", tmp_path,
                                       Path("/source"), "python", 3,
                                       "paced", 2.0)
    assert command[:8] == ["mpirun", "-n", "3", "--bind-to", "core",
                           "python", "-m", "mpi4py"]
    assert "jax:cpu" in command
    assert "cpu:4" not in command
    assert "--replay-clock" in command
    assert command[command.index("--replay-rate") + 1] == "2.0"
    assert command[command.index("--jax-chisq-mode") + 1] == "cpu-compatible"


def test_live_mode_is_preserved_and_cpu_omits_it(tmp_path):
    jax_config = {"args": ["--processing-scheme", "jax:cpu",
                            "--jax-chisq-mode", "direct-phase"]}
    command = bench.build_live_command(jax_config, "jax_cpu", tmp_path,
                                       Path("/source"), "python", 1,
                                       "none", 0.0)
    assert command[command.index("--jax-chisq-mode") + 1] == "direct-phase"
    cpu_command = bench.build_live_command(jax_config, "branch_cpu", tmp_path,
                                            Path("/source"), "python", 1,
                                            "none", 0.0)
    assert "--jax-chisq-mode" not in cpu_command
    assert bench.normalized_science_args(jax_config)[-2:] == [
        "--jax-chisq-mode", "direct-phase"
    ]
    cpu_args = bench.normalized_science_args(jax_config, scheme_override="cpu:1")
    assert "--jax-chisq-mode" not in cpu_args
    jax_args = bench.normalized_science_args(jax_config, scheme_override="jax:cpu")
    assert jax_args.count("--jax-chisq-mode") == 1


def test_actual_arm_science_config_compares_cpu_and_jax_default(tmp_path):
    config = {"args": ["--processing-scheme", "jax:cpu",
                        "--jax-chisq-mode", "cpu-compatible",
                        "--sample-rate", "2048"]}
    digest = {"sha256": "stable"}
    cpu = bench.science_config_for_arm(config, "branch_cpu", digest)
    jax = bench.science_config_for_arm(config, "jax_cpu", digest)
    assert cpu["args"] == jax["args"]
    assert "--jax-chisq-mode" not in cpu
    assert jax["--jax-chisq-mode"] == "cpu-compatible"
    attrs = lambda value: {
        "/@science_config": np.asarray(json.dumps(value)),
        **({"/@jax_chisq_mode": np.asarray(value["--jax-chisq-mode"])}
           if "--jax-chisq-mode" in value else {}),
    }
    compared = science._configuration_attributes(attrs(cpu), attrs(jax))
    assert compared["/@science_config"]["passed"]
    assert compared["/@jax_chisq_mode"]["passed"]
    direct = dict(jax)
    direct["--jax-chisq-mode"] = "direct-phase"
    assert not science._configuration_attributes(attrs(cpu), attrs(direct))[
        "/@jax_chisq_mode"]["passed"]


def test_branch_cpu_uses_candidate_cpu_scheme_without_reference_observer(tmp_path):
    command = bench.build_live_command(
        {"args": ["--sample-rate", "2048"]}, "branch_cpu", tmp_path,
        Path("/candidate"), "python", 1, "none", 0.0)
    assert "cpu:1" in command
    assert not any("observe_pycbc_live.py" in item for item in command)


def test_followup_processing_scheme_keeps_mkl_single_threaded():
    import ast

    source = (Path(__file__).parents[1] / "bin" / "pycbc_live").read_text()
    tree = ast.parse(source)
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "followup_processing_scheme")
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "live", "exec"),
         namespace)
    followup_scheme = namespace["followup_processing_scheme"]

    assert followup_scheme("mkl:8") == "mkl:1"
    assert followup_scheme("cpu:8") == "cpu:1"


def test_boolean_replay_clock_does_not_consume_next_option(tmp_path):
    config = {"args": ["--replay-clock", "--bank-file", "real-bank.hdf"]}
    command = bench.build_live_command(config, "cpu", tmp_path,
                                       Path("/source"), "python", 2,
                                       "unpaced", 0.0)
    assert "--bank-file" in command
    assert command[command.index("--bank-file") + 1] == "real-bank.hdf"


def test_single_detector_dump_initialization_is_jax_only():
    import ast
    from types import SimpleNamespace

    source = (Path(__file__).parents[1] / "bin" / "pycbc_live").read_text()
    tree = ast.parse(source)
    branch = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.If)
                  and any(isinstance(child, ast.Assign)
                          and any(isinstance(target, ast.Name)
                                  and target.id == "best_coinc"
                                  for target in child.targets)
                          for child in node.body))
    code = compile(ast.Module(body=[branch], type_ignores=[]), "live", "exec")

    class JAXScheme:
        pass

    for ctx in (object(), JAXScheme()):
        scope = {"ctx": ctx, "scheme": SimpleNamespace(JAXScheme=JAXScheme)}
        exec(code, scope)
        if isinstance(ctx, JAXScheme):
            assert scope["best_coinc"] == {}
        else:
            assert "best_coinc" not in scope


def test_capture_evidence_is_opt_in(tmp_path):
    config = {"args": ["--bank-file", "real-bank.hdf"],
              "science": {"capture_evidence": True}}
    command = bench.build_live_command(config, "cpu", tmp_path,
                                       Path("/source"), "python", 2,
                                       "none", 0.0)
    assert command[command.index("--benchmark-evidence-path") + 1] == str(tmp_path / "evidence")


def test_science_configuration_retains_full_physics_args():
    config = {"args": ["--low-frequency-cutoff", "30", "--processing-scheme",
                        "cpu:1", "--output-path", "/tmp/out", "--sample-rate", "2048"]}
    assert bench.normalized_science_args(config) == [
        "--low-frequency-cutoff", "30", "--sample-rate", "2048"
    ]


def test_benchmark_physics_is_explicit_and_fixed():
    valid = {"science": {"sample_rate": 2048, "precision": "complex64"},
             "args": ["--sample-rate", "2048"]}
    bench.validate_benchmark_config(valid)
    for invalid in (
        {"science": {"sample_rate": 4096, "precision": "complex64"},
         "args": ["--sample-rate", "4096"]},
        {"science": {"sample_rate": 2048, "precision": "float64"},
         "args": ["--sample-rate", "2048"]},
    ):
        try:
            bench.validate_benchmark_config(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("non-conforming benchmark configuration accepted")


def test_qualification_and_timed_configs_separate_evidence_io():
    config = {"science": {"sample_rate": 2048}, "args": ["--sample-rate", "2048"]}
    qualification, timed = bench.qualification_and_timed_configs(config)
    assert qualification["science"]["capture_evidence"] is True
    assert timed["science"]["capture_evidence"] is False


def test_pristine_cpu_wrapper_receives_no_runner_control_flags(tmp_path):
    command = bench.build_live_command(
        {"args": ["--sample-rate", "2048", "--replay-clock", "--replay-rate", "2",
                   "--benchmark-evidence-path", "stale"],
         "science": {"capture_evidence": True}},
        "cpu", tmp_path, tmp_path, "python", 1, "paced", 2.0, "0",
        observe_pristine_cpu=True)
    assert any("observe_pycbc_live.py" in item for item in command)
    assert "--replay-clock" not in command
    assert "--replay-rate" not in command
    assert "--benchmark-evidence-path" not in command


def test_main_runs_one_qualification_before_timed_pass(tmp_path, monkeypatch):
    calls = []

    def fake_run_one(config, arm, output_dir, *args):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stdout.log").write_text("")
        (output_dir / "stderr.log").write_text("")
        calls.append((arm, config["science"]["capture_evidence"]))
        return _decorate_result(config, {"arm": arm, "status": "passed", "returncode": 0,
                "process_complete": True,
                "command": [arm], "logs": {"stdout": str(output_dir / "stdout.log"),
                                            "stderr": str(output_dir / "stderr.log")},
                "trigger_outputs": [], "evidence_outputs": [],
                "completed_template_seconds": 1.0, "block_count": 1})

    monkeypatch.setattr(bench, "run_one", fake_run_one)
    def fake_science(runs, config):
        for values in runs.values():
            for result in values:
                result["science"] = {"passed": True}
    monkeypatch.setattr(bench, "_scientific_comparisons", fake_science)
    monkeypatch.setattr(bench, "_timed_science_comparisons",
                        lambda runs, qualification, config:
                        fake_science(runs, config))
    config, config_path = _write_main_config(tmp_path)
    output = tmp_path / "receipt.json"
    assert bench.main(["--config", str(config_path), *_mock_reference(monkeypatch, tmp_path)[1], "--output", str(output),
                       "--output-dir", str(tmp_path / "runs"), "--arms", "cpu",
                       "--replicates", "1", "--ranks", "1", "--affinity", "0"]) == 0
    assert calls == [("cpu", True), ("cpu", False)]
    receipt = __import__("json").loads(output.read_text())
    assert receipt["qualification_results"]["cpu"][0]["status"] == "passed"


def test_main_exit_fails_when_science_qualification_is_missing(tmp_path, monkeypatch):
    def fake_run_one(config, arm, output_dir, *args):
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout = output_dir / "stdout.log"
        stderr = output_dir / "stderr.log"
        stdout.write_text("")
        stderr.write_text("")
        return _decorate_result(config, {"arm": arm, "status": "passed", "process_complete": True,
                "returncode": 0, "command": [arm],
                "logs": {"stdout": str(stdout), "stderr": str(stderr)},
                "completed_template_seconds": 1.0, "block_count": 1})

    monkeypatch.setattr(bench, "run_one", fake_run_one)
    monkeypatch.setattr(bench, "_scientific_comparisons", lambda runs, config: None)
    monkeypatch.setattr(bench, "_timed_science_comparisons",
                        lambda runs, qualification, config: None)
    config, config_path = _write_main_config(tmp_path)
    output = tmp_path / "receipt.json"
    assert bench.main(["--config", str(config_path), *_mock_reference(monkeypatch, tmp_path)[1], "--output", str(output),
                       "--output-dir", str(tmp_path / "runs"), "--arms", "cpu",
                       "--replicates", "1", "--ranks", "1", "--affinity", "0"]) == 1
    assert __import__("json").loads(output.read_text())["campaign"] == {
        "process_complete": False, "science_qualified": False, "passed": False,
        "inputs_stable": True, "candidate_stable": True}


def test_qualification_only_runs_no_timed_repetitions(tmp_path, monkeypatch):
    calls = []

    def fake_run_one(config, arm, output_dir, *args):
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout = output_dir / "stdout.log"
        stderr = output_dir / "stderr.log"
        stdout.write_text("")
        stderr.write_text("")
        calls.append((arm, config["science"]["capture_evidence"]))
        return _decorate_result(config, {"arm": arm, "status": "passed",
                "process_complete": True, "returncode": 0, "command": [arm],
                "logs": {"stdout": str(stdout), "stderr": str(stderr)},
                "completed_template_seconds": 1.0, "block_count": 1})

    monkeypatch.setattr(bench, "run_one", fake_run_one)
    monkeypatch.setattr(bench, "_scientific_comparisons",
                        lambda runs, config: [item.update({"science": {"passed": True}})
                                              for values in runs.values() for item in values])
    monkeypatch.setattr(bench, "_timed_science_comparisons", lambda *args: None)
    config, config_path = _write_main_config(tmp_path)
    output = tmp_path / "receipt.json"
    assert bench.main(["--config", str(config_path), *_mock_reference(monkeypatch, tmp_path)[1],
                       "--output", str(output), "--output-dir", str(tmp_path / "runs"),
                       "--arms", "cpu", "branch_cpu", "--qualification-only",
                       "--replicates", "3", "--ranks", "1", "--affinity", "0"]) == 0
    assert calls == [("cpu", True), ("branch_cpu", True)]
    receipt = json.loads(output.read_text())
    assert receipt["campaign"]["status"] == "science_qualification_only"
    assert receipt["campaign"]["performance_claim"] is False
    assert receipt["raw_results"] == {"cpu": [], "branch_cpu": []}
    assert set(receipt["science"]) == {"cpu", "branch_cpu"}
    assert all(item["passed"] for item in receipt["science"].values())


def test_profile_utilization_is_one_separate_profile_per_arm(tmp_path, monkeypatch):
    calls = []

    def fake_run_one(config, arm, output_dir, *args):
        output_dir.mkdir(parents=True, exist_ok=True)
        calls.append((arm, bool(args[7]), output_dir.name))
        return _decorate_result(config, {"arm": arm, "status": "passed",
                "process_complete": True, "returncode": 0, "command": [arm],
                "logs": {}, "trigger_outputs": [], "evidence_outputs": [],
                "completed_template_seconds": 1.0, "block_count": 1})

    monkeypatch.setattr(bench, "run_one", fake_run_one)
    monkeypatch.setattr(bench, "_scientific_comparisons",
                        lambda runs, config: [item.update({"science": {"passed": True}})
                                              for values in runs.values() for item in values])
    monkeypatch.setattr(bench, "_timed_science_comparisons",
                        lambda runs, qualification, config: [item.update({"science": {"passed": True}})
                                                              for values in runs.values() for item in values])
    config, config_path = _write_main_config(tmp_path)
    output = tmp_path / "receipt.json"
    assert bench.main(["--config", str(config_path), *_mock_reference(monkeypatch, tmp_path)[1],
                       "--output", str(output), "--output-dir", str(tmp_path / "runs"),
                       "--arms", "cpu", "branch_cpu", "--profile-utilization",
                       "--replicates", "3", "--ranks", "1", "--affinity", "0"]) == 0
    # qualification x2, profile x2; profile is exactly one per arm.
    assert calls == [("cpu", False, "cpu"), ("branch_cpu", False, "branch_cpu"),
                     ("cpu", True, "cpu"), ("branch_cpu", True, "branch_cpu")]
    receipt = json.loads(output.read_text())
    assert receipt["profiling"]["profiles_per_arm"] == 1
    assert receipt["profiling"]["performance_claim"] is False
    assert set(receipt["science"]) == {"cpu", "branch_cpu"}
    assert all(item["passed"] for item in receipt["science"].values())


def test_main_receipt_retains_process_completion_failures(tmp_path, monkeypatch):
    def fake_run_one(config, arm, output_dir, *args):
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout = output_dir / "stdout.log"
        stderr = output_dir / "stderr.log"
        stdout.write_text("")
        stderr.write_text("")
        return _decorate_result(config, {"arm": arm, "status": "failed", "process_complete": False,
                "returncode": 0, "command": [arm],
                "logs": {"stdout": str(stdout), "stderr": str(stderr)},
                "completed_template_seconds": None, "block_count": 0})

    monkeypatch.setattr(bench, "run_one", fake_run_one)
    monkeypatch.setattr(bench, "_scientific_comparisons", lambda runs, config: None)
    monkeypatch.setattr(bench, "_timed_science_comparisons",
                        lambda runs, qualification, config: None)
    config, config_path = _write_main_config(tmp_path)
    output = tmp_path / "receipt.json"
    assert bench.main(["--config", str(config_path), *_mock_reference(monkeypatch, tmp_path)[1], "--output", str(output),
                       "--output-dir", str(tmp_path / "runs"), "--arms", "cpu",
                       "--replicates", "1", "--ranks", "1", "--affinity", "0"]) == 1
    receipt = json.loads(output.read_text())
    assert len(receipt["failed_history"]) == 1
    assert all(not item["process_complete"] for item in receipt["failed_history"])


def test_unqualified_arms_are_skipped_with_explicit_reasons(tmp_path, monkeypatch):
    calls = []

    def fake_run_one(config, arm, output_dir, *args):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stdout.log").write_text("")
        (output_dir / "stderr.log").write_text("")
        calls.append((arm, config["science"]["capture_evidence"]))
        return _decorate_result(config, {"arm": arm, "status": "passed", "process_complete": True,
                "returncode": 0, "command": [arm], "completed_template_seconds": 1.0,
                "block_count": 1})

    monkeypatch.setattr(bench, "run_one", fake_run_one)
    monkeypatch.setattr(bench, "_scientific_comparisons",
                        lambda runs, config: [
                            result.update({"science": {"passed": False,
                                                        "missing_gates": ["trigger"]}})
                            for values in runs.values() for result in values])
    monkeypatch.setattr(
        bench, "_timed_science_comparisons",
        lambda runs, qualification, config: [
            item.update({"science": {"passed": True}})
            for values in runs.values() for item in values],
    )
    config, config_path = _write_main_config(tmp_path)
    output = tmp_path / "receipt.json"
    with pytest.raises(SystemExit):
        bench.main(["--config", str(config_path), "--output", str(output),
                    "--output-dir", str(tmp_path / "runs"), "--arms", "cpu", "jax_cpu",
                    "--replicates", "1", "--ranks", "1", "--affinity", "0"])
    assert calls == []


def test_any_failed_arm_blocks_all_timing(tmp_path, monkeypatch):
    calls = []

    def fake_run_one(config, arm, output_dir, *args):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stdout.log").write_text("")
        (output_dir / "stderr.log").write_text("")
        calls.append((arm, config["science"]["capture_evidence"]))
        return _decorate_result(config, {"arm": arm, "status": "passed", "process_complete": True,
                "returncode": 0, "command": [arm], "completed_template_seconds": 1.0,
                "block_count": 1})

    monkeypatch.setattr(bench, "run_one", fake_run_one)

    def fake_qualification(runs, config):
        for arm, values in runs.items():
            for result in values:
                result["science"] = {"passed": arm == "cpu",
                                      "missing_gates": [] if arm == "cpu" else ["trigger"]}

    monkeypatch.setattr(bench, "_scientific_comparisons", fake_qualification)
    monkeypatch.setattr(bench, "_timed_science_comparisons",
                        lambda runs, qualification, config:
                        [result.update({"science": {"passed": True}})
                         for values in runs.values() for result in values])
    config, config_path = _write_main_config(tmp_path)
    output = tmp_path / "receipt.json"
    assert bench.main(["--config", str(config_path), *_mock_reference(monkeypatch, tmp_path)[1], "--output", str(output),
                    "--output-dir", str(tmp_path / "runs"), "--arms", "cpu", "jax_cpu",
                    "--replicates", "1", "--ranks", "1", "--affinity", "0"]) == 1
    assert calls == [("cpu", True), ("jax_cpu", True)]
    receipt = json.loads(output.read_text())
    assert set(receipt["timing_policy"]["skipped_arms"]) == {"cpu", "jax_cpu"}


def test_allow_unqualified_timings_runs_diagnostic_arm_and_binds_identity(tmp_path, monkeypatch):
    calls = []

    def fake_run_one(config, arm, output_dir, *args):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stdout.log").write_text("")
        (output_dir / "stderr.log").write_text("")
        calls.append((arm, config["science"]["capture_evidence"]))
        return _decorate_result(config, {"arm": arm, "status": "passed", "process_complete": True,
                "returncode": 0, "command": [arm], "completed_template_seconds": 1.0,
                "block_count": 1})

    monkeypatch.setattr(bench, "run_one", fake_run_one)
    monkeypatch.setattr(bench, "_scientific_comparisons",
                        lambda runs, config: [
                            result.update({"science": {"passed": False,
                                                        "missing_gates": ["trigger"]}})
                            for values in runs.values() for result in values])
    monkeypatch.setattr(bench, "_timed_science_comparisons",
                        lambda runs, qualification, config:
                        [result.update({"science": {"passed": False}})
                         for values in runs.values() for result in values])
    config, config_path = _write_main_config(tmp_path)
    output = tmp_path / "receipt.json"
    assert bench.main(["--config", str(config_path), *_mock_reference(monkeypatch, tmp_path)[1], "--output", str(output),
                       "--output-dir", str(tmp_path / "runs"), "--arms", "cpu", "jax_cpu",
                       "--allow-unqualified-timings", "--replicates", "1", "--ranks", "1",
                       "--affinity", "0"]) == 1
    receipt = json.loads(output.read_text())
    assert calls == [("cpu", True), ("jax_cpu", True), ("cpu", False),
                     ("jax_cpu", False)]
    assert receipt["timing_policy"]["allow_unqualified_timings"] is True
    assert receipt["timing_policy"]["skipped_arms"] == {}
    checkpoint = json.loads((tmp_path / "receipt.json.progress.json").read_text())
    assert checkpoint["identity"]["execution"]["allow_unqualified_timings"] is True
    assert checkpoint["input_contract"] == receipt["input_contract"]
    assert checkpoint["workload"] == receipt["workload"]


def test_timed_failure_stops_remaining_repetitions(tmp_path, monkeypatch):
    calls = []

    def fake_run_one(config, arm, output_dir, *args):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stdout.log").write_text("")
        (output_dir / "stderr.log").write_text("")
        timed = not config["science"]["capture_evidence"]
        failed = timed and not any(item[1] for item in calls if item[0] == arm)
        calls.append((arm, timed))
        return _decorate_result(config, {"arm": arm,
                "status": "failed" if failed else "passed",
                "process_complete": not failed, "returncode": 0 if not failed else 1,
                "command": [arm], "logs": {"stdout": str(output_dir / "stdout.log"),
                                             "stderr": str(output_dir / "stderr.log")},
                "completed_template_seconds": 1.0 if not failed else None,
                "block_count": 1})

    monkeypatch.setattr(bench, "run_one", fake_run_one)
    monkeypatch.setattr(bench, "_scientific_comparisons",
                        lambda runs, config: [item.update({"science": {"passed": True}})
                                              for values in runs.values() for item in values])
    monkeypatch.setattr(bench, "_timed_science_comparisons",
                        lambda runs, qualification, config: [
                            item.update({"science": {"passed": item.get("process_complete", False)}})
                            for values in runs.values() for item in values])
    config, config_path = _write_main_config(tmp_path)
    output = tmp_path / "receipt.json"
    assert bench.main(["--config", str(config_path),
                       *_mock_reference(monkeypatch, tmp_path)[1], "--output", str(output),
                       "--output-dir", str(tmp_path / "runs"), "--arms", "cpu", "jax_cpu",
                       "--replicates", "3", "--ranks", "1", "--affinity", "0"]) == 1
    assert calls == [("cpu", False), ("jax_cpu", False), ("cpu", True)]
    assert json.loads(output.read_text())["timing_policy"]["stopped_after"]["arm"] == "cpu"


def test_changed_input_prevents_next_launch(tmp_path, monkeypatch):
    calls = []
    config, config_path = _write_main_config(tmp_path)
    frame = Path(config["args"][config["args"].index("--frame-src") + 1].split(":", 1)[1])

    def fake_run_one(config, arm, output_dir, *args):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stdout.log").write_text("")
        (output_dir / "stderr.log").write_text("")
        calls.append(arm)
        if len(calls) == 1:
            frame.write_bytes(b"changed")
        return _decorate_result(config, {"arm": arm, "status": "passed",
                "process_complete": True, "returncode": 0, "command": [arm],
                "logs": {"stdout": str(output_dir / "stdout.log"),
                         "stderr": str(output_dir / "stderr.log")},
                "completed_template_seconds": 1.0, "block_count": 1})

    monkeypatch.setattr(bench, "run_one", fake_run_one)
    monkeypatch.setattr(bench, "_scientific_comparisons",
                        lambda runs, config: [item.update({"science": {"passed": True}})
                                              for values in runs.values() for item in values])
    monkeypatch.setattr(bench, "_timed_science_comparisons",
                        lambda runs, qualification, config: [item.update({"science": {"passed": True}})
                                                              for values in runs.values() for item in values])
    with pytest.raises(ValueError, match="workload input contract changed"):
        bench.main(["--config", str(config_path),
                    *_mock_reference(monkeypatch, tmp_path)[1], "--output", str(tmp_path / "receipt.json"),
                    "--output-dir", str(tmp_path / "runs"), "--arms", "cpu", "jax_cpu",
                    "--replicates", "1", "--ranks", "1", "--affinity", "0"])
    assert calls == ["cpu"]


def test_campaign_dispatches_cpu_reference_and_jax_candidate_sources(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    config_path = tmp_path / "config.json"
    config, config_path = _write_main_config(tmp_path)
    output = tmp_path / "receipt.json"
    seen_sources = []

    def clean_identity(root, files):
        return {"repository": str(root.resolve()), "revision": "clean",
                "dirty": False, "files_sha256": {}}

    def fake_run_one(config, arm, output_dir, *args):
        seen_sources.append((arm, Path(args[0]).resolve()))
        output_dir.mkdir(parents=True, exist_ok=True)
        return _decorate_result(config, {"arm": arm, "status": "passed", "returncode": 0,
                "process_complete": True, "command": [arm],
                "logs": {}, "trigger_outputs": [], "evidence_outputs": [],
                "completed_template_seconds": 1.0, "block_count": 1})

    monkeypatch.setattr(bench, "source_identity", clean_identity)
    monkeypatch.setattr(bench, "validate_reference", lambda root, revision: "a" * 40)
    monkeypatch.setattr(bench, "validate_source", lambda path: None)
    monkeypatch.setattr(bench, "run_one", fake_run_one)
    monkeypatch.setattr(bench, "_scientific_comparisons",
                        lambda runs, config: [
                            item.update({"science": {"passed": True}})
                            for values in runs.values() for item in values])
    monkeypatch.setattr(
        bench, "_timed_science_comparisons",
        lambda runs, qualification, config: [
            item.update({"science": {"passed": True}})
            for values in runs.values() for item in values],
    )
    assert bench.main([
        "--config", str(config_path), "--source-root", str(candidate),
        "--reference-source-root", str(reference), "--output", str(output),
        "--output-dir", str(tmp_path / "runs"), "--arms", "cpu", "jax_cpu",
        "--reference-revision", "a" * 40, "--replicates", "1", "--ranks", "1",
        "--affinity", "0",
    ]) == 0
    assert seen_sources == [
        ("cpu", reference.resolve()), ("jax_cpu", candidate.resolve()),
        ("cpu", reference.resolve()), ("jax_cpu", candidate.resolve()),
    ]


def test_invalid_reference_cannot_leave_passing_science_in_receipt(tmp_path, monkeypatch):
    config, config_path = _write_main_config(tmp_path)
    output = tmp_path / "receipt.json"

    def fake_run_one(config, arm, output_dir, *args):
        output_dir.mkdir(parents=True, exist_ok=True)
        return _decorate_result(config, {"arm": arm, "status": "passed", "returncode": 0,
                "process_complete": True, "command": [arm], "logs": {},
                "trigger_outputs": [], "evidence_outputs": [],
                "completed_template_seconds": 1.0, "block_count": 1})

    def mark_passed(groups, *args):
        for values in groups.values():
            for result in values:
                result["science"] = {"passed": True}

    monkeypatch.setattr(bench, "run_one", fake_run_one)
    monkeypatch.setattr(bench, "_scientific_comparisons", mark_passed)
    monkeypatch.setattr(bench, "_timed_science_comparisons", mark_passed)
    assert bench.main([
        "--config", str(config_path), "--output", str(output),
        "--output-dir", str(tmp_path / "runs"), "--arms", "cpu", "jax_cpu",
        "--allow-unqualified-timings", "--replicates", "1", "--ranks", "1",
        "--affinity", "0",
    ]) == 1
    receipt = json.loads(output.read_text())
    assert receipt["baseline"]["publishable"] is False
    for groups in (receipt["qualification_results"], receipt["raw_results"]):
        for values in groups.values():
            for result in values:
                assert result["science"]["passed"] is False
                assert result["science"]["diagnostic_science"]["passed"] is True


def test_science_invalidation_is_idempotent_and_preserves_diagnostics():
    groups = {"cpu": [{"science": {"passed": True,
                                     "missing_gates": ["comparison"]}}]}
    bench._invalidate_science(groups, "reference changed")
    bench._invalidate_science(groups, "inputs changed")
    science = groups["cpu"][0]["science"]
    assert science["passed"] is False
    assert science["missing_gates"] == ["comparison", "inputs changed", "reference changed"]
    assert science["diagnostic_science"] == {
        "passed": True, "missing_gates": ["comparison"]}


def test_mutated_reference_invalidates_recorded_science(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    config, config_path = _write_main_config(tmp_path)
    output = tmp_path / "receipt.json"
    calls = {candidate.resolve(): 0, reference.resolve(): 0}

    def changing_identity(root, files):
        root = root.resolve()
        calls[root] += 1
        changed = root == reference.resolve() and calls[root] > 1
        return {"repository": str(root), "revision": "clean", "dirty": False,
                "files_sha256": {"changed": str(changed)}}

    def fake_run_one(config, arm, output_dir, *args):
        output_dir.mkdir(parents=True, exist_ok=True)
        return _decorate_result(config, {"arm": arm, "status": "passed", "returncode": 0,
                "process_complete": True, "command": [arm], "logs": {},
                "trigger_outputs": [], "evidence_outputs": [],
                "completed_template_seconds": 1.0, "block_count": 1})

    def mark_passed(groups, *args):
        for values in groups.values():
            for result in values:
                result["science"] = {"passed": True}

    monkeypatch.setattr(bench, "source_identity", changing_identity)
    monkeypatch.setattr(bench, "validate_reference", lambda root, revision: "a" * 40)
    monkeypatch.setattr(bench, "validate_source", lambda path: None)
    monkeypatch.setattr(bench, "run_one", fake_run_one)
    monkeypatch.setattr(bench, "_scientific_comparisons", mark_passed)
    monkeypatch.setattr(bench, "_timed_science_comparisons", mark_passed)
    assert bench.main([
        "--config", str(config_path), "--source-root", str(candidate),
        "--reference-source-root", str(reference), "--output", str(output),
        "--output-dir", str(tmp_path / "runs"), "--arms", "cpu", "jax_cpu",
        "--reference-revision", "a" * 40, "--replicates", "1", "--ranks", "1",
        "--affinity", "0",
    ]) == 1
    receipt = json.loads(output.read_text())
    assert receipt["baseline"]["stable"] is False
    for values in receipt["raw_results"].values():
        for result in values:
            assert result["science"]["passed"] is False


def test_replicate_order_is_counterbalanced():
    arms = ["cpu", "jax_cpu", "jax_cuda"]
    assert bench.arm_order_for_replicate(arms, 0) == arms
    assert bench.arm_order_for_replicate(arms, 1) == list(reversed(arms))


def test_latency_summary_empty_is_serializable():
    summary = bench.latency_summary([])
    assert summary["count"] == 0
    assert summary["p95"] is None


def test_workload_reads_metadata_from_real_command_args():
    config = {"args": [
        "--bank-file", "/inputs/bank.hdf",
        "--analysis-chunk", "8",
        "--start-time=1187007080",
        "--end-time", "1187009000",
    ]}
    workload = bench.workload_from_config(config)
    assert workload["bank_file"] == "/inputs/bank.hdf"
    assert workload["analysis_chunk_sec"] == 8.0
    assert workload["start_time"] == "1187007080"
    assert workload["end_time"] == "1187009000"


def test_stage_metrics_labels_frame_read_as_read_and_condition():
    events = [
        {"name": "frame_read", "event": "start", "pid": 1, "rank": 0,
         "ifo": "H1", "data_start": 10, "time_ns": 0},
        {"name": "frame_read", "event": "end", "pid": 1, "rank": 0,
         "ifo": "H1", "data_start": 10, "time_ns": 2_000_000},
    ]
    stages = bench.stage_metrics(events)
    assert "frame_read" not in stages
    assert stages["read_and_condition"]["samples"] == [0.002]


def test_input_contract_digest_changes_with_frame_content(tmp_path):
    bank = tmp_path / "bank.hdf"
    frame = tmp_path / "frame.gwf"
    import h5py
    with h5py.File(bank, "w") as handle:
        handle.create_dataset("mass1", data=[1.0, 2.0])
    frame.write_bytes(b"frame-a")
    config = {
        "workload": {"templates": 2, "analysis_chunk_sec": 8,
                     "start_time": 1, "end_time": 9},
        "science": {"sample_rate": 2048, "precision": "complex64"},
        "args": ["--sample-rate", "2048", "--bank-file", str(bank),
                 "--frame-src", f"H1:{frame}"],
    }
    first = bench.workload_digest(config)
    frame.write_bytes(b"frame-b")
    second = bench.workload_digest(config)
    assert first["sha256"] != second["sha256"]
    assert first["contract"]["bank_sha256"] == second["contract"]["bank_sha256"]


def test_input_contract_rejects_bare_launch():
    config = {"science": {"sample_rate": 2048, "precision": "complex64"},
             "args": ["--sample-rate", "2048"]}
    try:
        bench.validate_input_contract(config)
    except ValueError as exc:
        assert "bank-file" in str(exc)
    else:
        raise AssertionError("bare live launch accepted")


@pytest.mark.parametrize("rows", [
    ((1.0, 2.0, 0.1, -0.2), (1.0, 2.0, 0.1, -0.2)),
    ((1.0, 2.0, 0.1, -0.2), (float("nan"), 4.0, 0.3, -0.4)),
])
def test_input_contract_rejects_duplicate_or_nonfinite_physical_rows(tmp_path, rows):
    config = _contract_fixture(tmp_path, rows)
    with pytest.raises(ValueError, match="physical mass/spin"):
        bench.validate_input_contract(config)


def test_input_contract_rejects_bank_and_frame_cli_disagreement(tmp_path):
    config = _contract_fixture(tmp_path)
    config["bank_file"] = config["args"][config["args"].index("--bank-file") + 1]
    other_bank = tmp_path / "other.hdf"
    other_bank.write_bytes(b"different")
    config["args"][config["args"].index("--bank-file") + 1] = str(other_bank)
    with pytest.raises(ValueError, match="bank-file"):
        bench.validate_input_contract(config)

    config = _contract_fixture(tmp_path)
    frame_arg = config["args"][config["args"].index("--frame-src") + 1]
    config["frame_files"] = [frame_arg.split(":", 1)[1]]
    other_frame = tmp_path / "other.gwf"
    other_frame.write_bytes(b"other")
    config["args"][config["args"].index("--frame-src") + 1] = f"H1:{other_frame}"
    with pytest.raises(ValueError, match="frame"):
        bench.validate_input_contract(config)


def test_qualification_and_timing_configs_share_workload_digest(tmp_path):
    config = _contract_fixture(tmp_path)
    qualification, timed = bench.qualification_and_timed_configs(config)
    assert bench.workload_digest(qualification)["sha256"] == \
        bench.workload_digest(timed)["sha256"]


@pytest.mark.parametrize("contract", [
    {},
    {"passed": True, "count": 1, "sample_rates": [4096],
     "signal_dtypes": [["complex64"]]},
    {"passed": True, "count": 1, "sample_rates": [2048],
     "signal_dtypes": [["float64"]]},
])
def test_completed_run_rejects_missing_or_wrong_filter_contract(tmp_path, contract):
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    trigger = tmp_path / "trigger.hdf"
    for path in (stdout, stderr, trigger):
        path.write_text("")
    result = {"arm": "cpu", "status": "passed", "returncode": 0,
              "command": ["pycbc_live"], "elapsed_wall_sec": 2.0,
              "completed_template_seconds": 8.0,
              "logs": {"stdout": str(stdout), "stderr": str(stderr)},
              "trigger_outputs": [str(trigger)], "filter_contract": contract}
    assert not bench._completed_run(result, "cpu")


def test_observed_filter_contract_requires_measured_2048_complex64():
    valid = [{"name": "filter", "event": "end", "sample_rate": 2048,
              "signal_dtypes": ["complex64"]}]
    assert bench.observed_filter_contract(valid)["passed"]
    assert not bench.observed_filter_contract([
        {"name": "filter", "event": "end", "sample_rate": 4096,
         "signal_dtypes": ["complex64"]}])["passed"]
    assert not bench.observed_filter_contract([
        {"name": "filter", "event": "end", "sample_rate": 2048,
         "signal_dtypes": ["float64"]}])["passed"]


def test_changed_input_invalidates_completed_run(tmp_path):
    config = _contract_fixture(tmp_path)
    digest = bench.workload_digest(config)["sha256"]
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    trigger = tmp_path / "trigger.hdf"
    for path in (stdout, stderr, trigger):
        path.write_text("")
    result = {"arm": "cpu", "status": "passed", "returncode": 0,
              "command": ["pycbc_live"], "elapsed_wall_sec": 2.0,
              "completed_template_seconds": 8.0,
              "logs": {"stdout": str(stdout), "stderr": str(stderr)},
              "trigger_outputs": [str(trigger)],
              "filter_contract": {"passed": True, "count": 1,
                                  "sample_rates": [2048.0],
                                  "signal_dtypes": [["complex64"]],
                                  "inputs_stable": True},
              "workload_digest": bench.workload_digest(config)}
    assert bench._completed_run(result, "cpu", expected_digest=digest)
    frame = Path(config["args"][config["args"].index("--frame-src") + 1].split(":", 1)[1])
    frame.write_bytes(b"changed")
    changed = bench.workload_digest(config)["sha256"]
    assert changed != digest
    assert not bench._completed_run(result, "cpu", expected_digest=changed)


def test_backlog_carries_between_blocks_and_wall_latency_is_separate():
    events = []
    for index, (compute, wall) in enumerate(((10.0, 12.0), (2.0, 4.0))):
        start = 100 + index * 8
        events.extend([
            {"name": "block", "event": "start", "rank": 0, "pid": 1,
             "time_ns": start * 1_000_000_000, "data_start": start},
            {"name": "block", "event": "end", "rank": 0, "pid": 1,
             "time_ns": (start + wall) * 1_000_000_000,
             "data_start": start, "data_end": start + 8,
             "processing_elapsed_sec": compute, "elapsed_sec": wall,
             "deadline_met": compute <= 8, "lag_sec": float(index),
             "live_detectors": ["H1"]},
        ])
    metrics = bench.block_metrics(events, 8)
    assert metrics["latency"]["samples"] == [10.0, 2.0]
    assert metrics["wall_latency"]["samples"] == [12.0, 4.0]
    assert metrics["backlog"]["samples"] == [2.0, 0.0]
    assert metrics["replay_lag"]["samples"] == [0.0, 1.0]
    assert metrics["backlog_method"] == "derived_from_processing_latency"


def test_replay_rate_scales_deadline_and_backlog_budget():
    events = [
        {"name": "block", "event": "start", "rank": 0, "pid": 1,
         "time_ns": 0, "data_start": 0},
        {"name": "block", "event": "end", "rank": 0, "pid": 1,
         "time_ns": 5_000_000_000, "data_start": 0, "data_end": 8,
         "processing_elapsed_sec": 5.0},
    ]
    metrics = bench.block_metrics(events, 8.0, deadline_budget=4.0)
    assert metrics["backlog"]["samples"] == [1.0]
    assert metrics["deadline_miss_count"] == 1


def test_science_comparison_fails_closed_on_missing_trigger_output(monkeypatch):
    runs = {
        "cpu": [{"trigger_outputs": ["/tmp/reference-100.hdf"],
                 "evidence_outputs": []}],
        "jax_cpu": [{"trigger_outputs": [], "evidence_outputs": []}],
    }
    config = {"science": {"reference_arm": "cpu", "sample_rate": 2048,
                           "precision": "complex64"},
              "args": ["--analysis-chunk", "8", "--start-time", "100",
                        "--end-time", "108", "--sample-rate", "2048"]}
    monkeypatch.setattr(bench, "_live_science_comparator",
                        lambda: (lambda *args, **kwargs: {"passed": False,
                                                           "missing_gates": []}))
    bench._scientific_comparisons(runs, config)
    science = runs["cpu"][0]["science"]
    assert science["passed"] is False
    assert "scientific fields or evidence mismatch" in science["missing_gates"]


def test_evidence_mapping_requires_complete_block_coverage(tmp_path):
    if bench.h5py is None:
        return
    evidence = tmp_path / "evidence.hdf"
    with bench.h5py.File(evidence, "w") as handle:
        handle.attrs["analyze_start"] = 100.0
        handle.attrs["analyze_end"] = 108.0
    mapping, errors = bench._map_block_evidence(
        ["H1-live-100-0.5.hdf", "H1-live-108-0.5.hdf"],
        [str(evidence)],
        {"args": ["--analysis-chunk", "8", "--start-time", "100",
                      "--end-time", "116", "--sample-rate", "2048"],
         "science": {"sample_rate": 2048, "precision": "complex64"}},
    )
    assert mapping == {"H1-live-100-0.5.hdf": str(evidence)}
    assert any("missing evidence mapping" in error for error in errors)


def test_evidence_mapping_uses_actual_fractional_block_start(tmp_path):
    if bench.h5py is None:
        return
    evidence = tmp_path / "block.hdf"
    actual_start = 1187007076.069336
    with bench.h5py.File(evidence, "w") as handle:
        handle.attrs["analyze_start"] = actual_start
        handle.attrs["analyze_end"] = actual_start + 8
    trigger = f"{tmp_path}/H1-live-{actual_start:.6f}-8.hdf"
    mapping, errors = bench._map_block_evidence(
        [trigger], [str(evidence)],
        {"args": ["--sample-rate", "2048"],
         "science": {"sample_rate": 2048, "precision": "complex64"}},
        block_starts=[actual_start],
    )
    assert mapping == {trigger: str(evidence)}
    assert errors == []


def test_evidence_mapping_requires_buffer_trigger_output(tmp_path):
    if bench.h5py is None:
        return
    evidence_paths = []
    starts = [1187007068.069336, 1187007076.069336]
    for index, start in enumerate(starts):
        path = tmp_path / f"block-{index}.hdf"
        with bench.h5py.File(path, "w") as handle:
            handle.attrs["analyze_start"] = start
            handle.attrs["analyze_end"] = start + 8
        evidence_paths.append(str(path))
    trigger = f"{tmp_path}/H1-live-{starts[1]:.6f}-8.hdf"
    mapping, errors = bench._map_block_evidence(
        [trigger], evidence_paths, {"args": ["--sample-rate", "2048"],
                                    "science": {"sample_rate": 2048, "precision": "complex64"}},
        block_starts=starts,
    )
    assert mapping == {trigger: evidence_paths[1]}
    assert any("trigger output coverage" in error for error in errors)


def test_evidence_mapping_pairs_buffer_trigger_output(tmp_path):
    if bench.h5py is None:
        return
    evidence_paths = []
    starts = [1187007068.069336, 1187007076.069336]
    for index, start in enumerate(starts):
        path = tmp_path / f"block-{index}.hdf"
        with bench.h5py.File(path, "w") as handle:
            handle.attrs["analyze_start"] = start
            handle.attrs["analyze_end"] = start + 8
        evidence_paths.append(str(path))
    triggers = [f"{tmp_path}/H1-live-{start:.6f}-8.hdf" for start in starts]
    mapping, errors = bench._map_block_evidence(
        triggers, evidence_paths, {"args": ["--sample-rate", "2048"],
                                    "science": {"sample_rate": 2048, "precision": "complex64"}},
        block_starts=starts,
    )
    assert set(mapping) == set(triggers)
    assert errors == []


def test_nested_hdf_discovery_excludes_evidence_and_background(tmp_path):
    nested = tmp_path / "2026_09_20"
    nested.mkdir()
    (nested / "H1-live-100.000000-8.hdf").write_text("")
    (nested / "H1-LIVE_BACKGROUND-100.hdf").write_text("")
    (nested / "candidate_100.0_abcd.hdf").write_text("")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "block.hdf").write_text("")
    triggers, evidence_paths = bench._discover_hdf_outputs(tmp_path)
    assert triggers == [str(nested / "H1-live-100.000000-8.hdf")]
    assert evidence_paths == [str(evidence / "block.hdf")]


def test_input_paths_include_frame_src_even_when_bank_is_present(tmp_path):
    bank = tmp_path / "bank.hdf"
    frame = tmp_path / "frame.gwf"
    bank.write_bytes(b"bank")
    frame.write_bytes(b"frame")
    paths = bench._input_paths({"args": ["--bank-file", str(bank),
                                           "--frame-src", f"H1:{frame}"]})
    assert paths == [bank.resolve(), frame.resolve()]


def test_campaign_identity_binds_complete_source_and_execution_manifest(tmp_path):
    config_path = tmp_path / "config.json"
    valid_config = {"args": ["--sample-rate", "2048"],
                    "science": {"sample_rate": 2048, "precision": "complex64"}}
    config_path.write_text(json.dumps(valid_config))
    root = Path(__file__).parents[1]
    first = bench.campaign_identity(valid_config, root, config_path, "python",
                                    {"arms": ["cpu"], "ranks": 1})
    second = bench.campaign_identity(valid_config, root, config_path, "python",
                                     {"arms": ["cpu"], "ranks": 2})
    files = first["source"]["files_sha256"]
    assert "bin/pycbc_live" in files
    assert "tools/bench_jax_live_campaign.py" in files
    assert "tools/benchmark_science.py" in files
    assert first != second


def test_campaign_identity_records_distinct_reference_source(tmp_path):
    config_path = tmp_path / "config.json"
    valid_config = {"args": ["--sample-rate", "2048"],
                    "science": {"sample_rate": 2048, "precision": "complex64"}}
    config_path.write_text(json.dumps(valid_config))
    candidate = Path(__file__).parents[1]
    reference = tmp_path / "pristine-reference"
    reference.mkdir()
    identity = bench.campaign_identity(
        valid_config, candidate, config_path, "python", {"arms": ["jax_cpu"]},
        reference,
    )
    assert identity["reference_source"]["repository"] == str(reference.resolve())
    assert bench._pristine_source({"revision": "abc", "dirty": False})
    assert not bench._pristine_source({"revision": "abc", "dirty": True})
    assert not bench._pristine_source(identity["reference_source"])


def test_completed_run_requires_outputs_work_and_evidence_when_requested(tmp_path):
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    trigger = tmp_path / "trigger.hdf"
    evidence = tmp_path / "evidence.hdf"
    for path in (stdout, stderr, trigger, evidence):
        path.write_text("")
    result = {"arm": "cpu", "status": "passed", "returncode": 0,
              "command": ["pycbc_live"], "elapsed_wall_sec": 2.0,
              "completed_template_seconds": 8.0,
              "logs": {"stdout": str(stdout), "stderr": str(stderr)},
              "trigger_outputs": [str(trigger)],
              "evidence_outputs": [str(evidence)],
              "filter_contract": {"passed": True, "count": 1,
                                  "sample_rates": [2048.0],
                                  "signal_dtypes": [["complex64"]]},
              "workload_digest": {"sha256": "unused", "contract": {}}}
    assert bench._completed_run(result, "cpu", require_evidence=True)
    result["completed_template_seconds"] = 0
    assert not bench._completed_run(result, "cpu", require_evidence=True)


def test_retry_directory_preserves_prior_attempt(tmp_path):
    base = tmp_path / "run-001"
    base.mkdir()
    retry = bench._retry_output_dir(base, {"status": "failed"})
    assert retry == tmp_path / "run-001.retry1"
    retry.mkdir()
    assert bench._retry_output_dir(base, {"status": "failed"}) == \
        tmp_path / "run-001.retry2"


def test_progress_checkpoint_replacement_is_atomic(tmp_path):
    path = tmp_path / "campaign.progress.json"
    bench._save_progress(path, {"identity": 1}, "running", {}, {}, [])
    assert json.loads(path.read_text())["status"] == "running"
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_live_receipt_exposes_stable_input_contract(tmp_path):
    config, config_path = _write_main_config(tmp_path)
    first = bench.make_receipt(
        config, {}, tmp_path, config_path, "unpaced", 1.0
    )
    second_config_path = tmp_path / "separate-run.json"
    second_config_path.write_text(json.dumps(config))
    second = bench.make_receipt(
        config, {}, tmp_path, second_config_path, "unpaced", 1.0
    )
    assert first["inputs"]["config"] != second["inputs"]["config"]
    assert first["input_contract"] == second["input_contract"]
    assert first["input_contract"] == first["workload_digest"]["contract"]
