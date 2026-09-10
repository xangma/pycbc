"""Small synthetic fixtures only; never acquire or claim campaign results.

Set PYCBC_BATCH_SWEEP_SCHEMA to the frozen R4 helper directory when it is not
present under this checkout's artifacts/. The real controller checks smoke
evidence; orchestration tests stub only its full-size result validation.
"""

from contextlib import contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "plot_batch", ROOT / "tools/plot_torch_batch_sweep.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    schema = Path(os.environ.get("PYCBC_BATCH_SWEEP_SCHEMA", str(
        ROOT / "artifacts/torch-current-batch-sweep-20260907-r4")))
    if not (schema / "batch-campaign.py").exists():
        pytest.skip("Set PYCBC_BATCH_SWEEP_SCHEMA to frozen R4 helpers")
    root = tmp_path / "synthetic-campaign"
    root.mkdir()
    for name in (*plot.HELPERS, "stage-manifest.json"):
        shutil.copyfile(schema / name, root / name)
    original_loader = plot.frozen_controller
    with original_loader(plot.Archive(root)) as c:
        policy = runpy.run_path(str(root / "batch-worker.py"))["POLICY"]
    remote = "/remote/synthetic-campaign"
    frozen = {name: plot.digest(root / name) for name in plot.HELPERS}
    save(root / "queued-inputs.json", frozen)
    save(root / "batch-plan.json", c.PLAN)
    save(root / "provenance.json", {
        "helper_sha256": {f"{remote}/{name}": sha for name, sha in frozen.items()},
        "plan": c.PLAN, "source": {"commit": c.REVISION, "status": ""},
        "native": json.loads((root / "native-provenance.json").read_text())})
    labels = []

    def metric(samples, error=0.0001):
        return dict(samples=samples, failed_samples=0, nonfinite_actual=0,
                    nonfinite_reference=0, max_abs_error=error,
                    max_tolerance_ratio=error / 0.001)

    def norm(source):
        fields = dict(norm=1.0, sigmasq=1.0, source=source)
        return dict(fields, sha256=c.canonical_hash(fields))

    def result(label, route, batch, seed, smoke=False, repeat=None):
        # Full fixtures deliberately contain one row, rather than fabricating
        # 1024-template science. The real validator must reject those fixtures.
        bank, size = (8, 32768) if smoke else (1, 131072)
        phase = "smoke" if smoke else "qual"
        ref_label = f"{phase}-s{seed}-b1-branch_standard"
        is_reference = label == ref_label
        directory = f"{remote}/runs/{ref_label}"
        error = 7e-6 if seed == 7103 else 1e-6
        error *= {"branch_standard": 1.0, "torch_cpu": 0.5, "torch_cuda": 0.75}[route]
        compatibility = 0.0 if route == "branch_standard" else error / 2
        rows = [{**metric(size, error), "template_id": 1000 + i,
                 "sha256": "a" * 64, "reference_sha256": "a" * 64,
                 "oracle": {"sha256": "b" * 64,
                            "normalization": norm("independent_float64_power_sum")},
                 "normalization": norm("observed_scalar_sigmasq_callback"),
                 "normalized_oracle": metric(size, error),
                 "normalized_mkl_compatibility": metric(size, compatibility),
                 "sigmasq_oracle": {"passed": True, "relative_error": 0.0},
                 "raw_complex_v1_audit": {**metric(size), "classification": "audit_only_not_v2_acceptance"},
                 "normwise_diagnostics": {"classification": "diagnostic_only_no_threshold"}}
                for i in range(bank)]
        pointwise = [{**metric(bank * size, error), "block": b,
                      "processed_once": True, "processing_counts": [1] * bank,
                      "rows": copy.deepcopy(rows), "normalization_failed_rows": 0,
                      "normalized_oracle": metric(bank * size, error),
                      "normalized_mkl_compatibility": metric(bank * size, compatibility)}
                     for b in range(3)]
        inputs = {"seed": seed, "geometry": {
            "bank_templates": bank, "fft_samples": size, "blocks": 3,
            "valid_start": 12288, "valid_end": 126976, "sample_rate": 2048}}
        groups = [list(range(1000 + i, 1000 + min(i + batch, bank)))
                  for i in range(0, bank, batch)]
        value = dict(schema_version=1, route=route, batch=batch,
                     mode="qualify" if repeat is None else "timing", expected_head=c.REVISION,
                     arguments={"seed": seed, "output_dir": f"{remote}/runs/{label}",
                                "reference_dir": None if is_reference else directory, "cuda_device": 0},
                     tolerance_policy=policy, policy_sha256=c.canonical_hash(policy),
                     source={"revision": c.REVISION, "tracked_dirty": False},
                     worker_sha256=frozen["batch-worker.py"], inputs=inputs,
                     input_sha256=c.canonical_hash(inputs),
                     thread_environment={"OMP_NUM_THREADS": "1"}, affinity=[8],
                     runtime={"torch_num_threads": 1, "torch_num_interop_threads": 1, "device": "cuda:0"},
                     group_layout=groups,
                     fft_plans=[{"nbatch": len(g), "class": "pycbc.fft.mkl.IFFT"} for g in groups],
                     status="pass", failures=[], exit_code=0, pointwise_blocks=pointwise,
                     trigger_blocks=[{"count": 4}] * 3,
                     trigger_comparisons=[] if is_reference else [{"passed": True}] * (24 if repeat else 3),
                     host="synthetic-host", pid=len(labels) + 100,
                     oracle=dict(created=is_reference, directory=directory, filename="oracle.npy",
                                 dtype="complex128", shape=[3, bank, size]),
                     reference=dict(created=is_reference, directory=directory, filename="outputs.npy",
                                    dtype="complex64", shape=[3, bank, size]))
        if not is_reference:
            value.update(reference_directory=directory, reference_result_sha256=plot.digest(
                root / "runs" / ref_label / "result.json"))
        if repeat:
            elapsed = 1000. * repeat / {"branch_standard": 1, "torch_cpu": 2, "torch_cuda": 4}[route]
            value["timing"] = {"instrumented": False, "call_surface": "LiveBatchMatchedFilter.process_data",
                               "warm_iteration_ms": [elapsed] * 5,
                               "warm_block_ms": [[elapsed / 3] * 3] * 5,
                               "templates_per_iteration": 3072}
        labels.append(label)
        save(root / "runs" / label / "result.json", value)
        save(root / "runs" / label / "acquisition.json", {
            "result_sha256": plot.digest(root / "runs" / label / "result.json"),
            "stage_receipt": f"{remote}/stages/{label}.json"})
        save(root / "stages" / f"{label}.json", {
            "state": "complete", "returncode": 0, "name": label,
            "started": 2 * len(labels), "finished": 2 * len(labels) + 1,
            "pid": value["pid"], "host": value["host"]})
        return value

    for smoke, phase, batches in ((True, "smoke", (1, 8)), (False, "qual", c.BATCHES)):
        records = {}
        for seed in c.PLAN["qualification_seeds"]:
            for batch in batches:
                for route in c.ROUTES:
                    label = f"{phase}-s{seed}-b{batch}-{route}"
                    records[label] = result(label, route, batch, seed, smoke)
        save(root / f"{phase}-qualifications.json", records)
    timings = [{"repeat": r, "batch": b, "route": route,
                "result": result(f"time-r{r}-b{b}-{route}", route, b, 7102, repeat=r)}
               for r, b, route in c.timing_order()]
    save(root / "timings.json", timings)
    save(root / "summary.json", c.summarize(timings))
    save(root / "batch-status.json", {
        "state": "complete", "phase": "complete", "source_unchanged": True,
        "finished": 2 * len(labels) + 2, "child_pid": None, "current": None, "qualifications_passed": 36,
        "timing_workers": 54, "completed": labels, "cwd": remote, "host": "synthetic-host"})
    calls = []

    @contextmanager
    def small_loader(archive):
        with original_loader(archive) as controller:
            validate = controller.validate_result

            def small_validate(*args):
                calls.append(args[1:])
                if args[5]:  # Real gates on all smoke rows; full fixtures are stubs.
                    validate(*args)

            controller.validate_result = small_validate
            yield controller

    monkeypatch.setattr(plot, "frozen_controller", small_loader)
    return root, calls, original_loader


def change(root, name, mutate):
    path = root / name
    value = json.loads(path.read_text())
    mutate(value)
    save(path, value)


def damage_smoke(root, mutate):
    label = "smoke-s7102-b1-torch_cpu"
    path = f"runs/{label}/result.json"
    change(root, path, mutate)
    value = json.loads((root / path).read_text())
    change(root, "smoke-qualifications.json", lambda data: data.update({label: value}))
    change(root, f"runs/{label}/acquisition.json", lambda data: data.update(result_sha256=plot.digest(root / path)))


def test_matrix_delegation_summary_and_both_seed_maxima(campaign):
    root, calls, _ = campaign
    archive, summary, accuracy = plot.validate_campaign(root)
    assert len(calls) == 102
    assert sum(call[2] == "qualify" and call[4] for call in calls) == 12
    assert sum(call[2] == "qualify" and not call[4] for call in calls) == 36
    assert sum(call[2] == "timing" for call in calls) == 54
    assert {call[5] for call in calls if call[2] == "qualify"} == {7102, 7103}
    assert summary["cells"][0]["median_templates_per_second"] == 1536
    assert summary["cells"][0]["templates_per_second_range"] == [1024, 3072]
    assert all(row["normalized_oracle"] == 7e-6 * {
        "branch_standard": 1.0, "torch_cpu": 0.5, "torch_cuda": 0.75}[row["route"]]
               for row in accuracy)
    assert all(row["normalized_mkl_compatibility"] == (
        0 if row["route"] == "branch_standard" else row["normalized_oracle"] / 2)
               for row in accuracy)
    assert "stage-manifest.json" in archive.hashes
    assert not list(root.rglob("*.pyc"))


@pytest.mark.parametrize("state", ["running", "failed", "cancelled", "waiting", "starting"])
def test_unfinished_rejected_before_loading_code(tmp_path, state):
    root = tmp_path / "unfinished"
    save(root / "batch-status.json", {"state": state, "phase": state})
    with pytest.raises(ValueError, match="unfinished"):
        plot.run(root, tmp_path / "plots")
    assert not (tmp_path / "plots").exists()


@pytest.mark.parametrize("name", ["smoke-qualifications.json", "qual-qualifications.json",
                                  "timings.json"])
def test_missing_matrix_cell_rejected(campaign, name):
    root, _, _ = campaign
    change(root, name, lambda data: data.pop() if isinstance(data, list) else data.pop(next(iter(data))))
    with pytest.raises(ValueError, match="Incomplete"):
        plot.validate_campaign(root)


@pytest.mark.parametrize("name", ["batch-worker.py", "batch-campaign.py", "NUMERICAL-POLICY.md",
                                  "campaign_controls.py"])
def test_changed_helpers_rejected_before_import(campaign, name):
    root, _, _ = campaign
    with (root / name).open("a") as stream:
        stream.write("\nchanged\n")
    with pytest.raises(ValueError, match="helper hash changed"):
        plot.validate_campaign(root)


@pytest.mark.parametrize("damage", ["fake_pass", "hidden_row", "oracle", "normalization", "trigger", "coverage"])
def test_real_controller_rejects_smoke_evidence_despite_pass(campaign, damage):
    root, _, _ = campaign

    def mutate(value):
        row = value["pointwise_blocks"][1]["rows"][2]
        if damage == "fake_pass":
            value["failures"] = ["synthetic failure"]
        elif damage == "hidden_row":
            row["normalized_oracle"]["failed_samples"] = 1
        elif damage == "oracle":
            row["oracle"]["sha256"] = "c" * 64
        elif damage == "normalization":
            row["normalization"]["norm"] = 2
        elif damage == "trigger":
            value["trigger_comparisons"][0]["passed"] = False
        else:
            value["pointwise_blocks"][0]["processing_counts"][0] = 0

    damage_smoke(root, mutate)
    with pytest.raises(ValueError):
        plot.validate_campaign(root)


def test_changed_summary_rejected(campaign):
    root, _, _ = campaign
    change(root, "summary.json", lambda data: data["cells"][0].update(median_templates_per_second=1e9))
    with pytest.raises(ValueError, match="Stored summary"):
        plot.validate_campaign(root)


@pytest.mark.parametrize("label,started,finished,message", [
    ("time-r1-b1-branch_standard", 0, 1, "before all qualifications"),
    ("smoke-s7102-b1-branch_standard", 2, 5, "Overlapping worker intervals"),
    ("smoke-s7102-b1-branch_standard", 3, 2, "Invalid worker interval"),
    ("smoke-s7102-b1-branch_standard", 2, "1e999", "Invalid worker interval"),
])
def test_stage_intervals_rejected(campaign, label, started, finished, message):
    root, _, _ = campaign
    path = root / "stages" / f"{label}.json"
    change(root, str(path.relative_to(root)),
           lambda data: data.update(started=started, finished=finished))
    if finished == "1e999":
        # A JSON number can overflow to infinity without a NaN/Infinity token.
        path.write_text(path.read_text().replace('"1e999"', '1e999'))
    with pytest.raises(ValueError, match=message):
        plot.validate_campaign(root)


def test_small_fixture_cannot_pass_real_full_validation(campaign, monkeypatch):
    root, _, original_loader = campaign
    monkeypatch.setattr(plot, "frozen_controller", original_loader)
    with pytest.raises(ValueError, match="dimensions"):
        plot.validate_campaign(root)


def test_real_controller_rejects_instrumented_timing(campaign):
    root, _, original_loader = campaign
    result = json.loads((root / "runs/smoke-s7102-b1-torch_cpu/result.json").read_text())
    result.update(mode="timing", trigger_comparisons=[{"passed": True}] * 24,
                  timing={"instrumented": False, "templates_per_iteration": 24,
                          "warm_iteration_ms": [3.0] * 5, "warm_block_ms": [[1.0] * 3] * 5})
    reference = plot.OfflineReference(
        result["reference_directory"], root / "runs/smoke-s7102-b1-branch_standard")
    with original_loader(plot.Archive(root)) as controller:
        controller.validate_result(result, "torch_cpu", 1, "timing", reference, smoke=True, seed=7102)
        result["timing"]["instrumented"] = True
        with pytest.raises(ValueError, match="timing"):
            controller.validate_result(result, "torch_cpu", 1, "timing", reference, smoke=True, seed=7102)


def test_offline_reference_preserves_remote_string(tmp_path):
    reference = plot.OfflineReference("/remote/declared", tmp_path)
    assert str(reference) == "/remote/declared"
    assert reference / "result.json" == tmp_path / "result.json"


def test_render_verify_and_preserve_inputs(campaign, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    suptitle = Figure.suptitle

    def synthetic_title(figure, title, **kwargs):
        return suptitle(figure, "SYNTHETIC FIXTURE · " + title, **kwargs)

    monkeypatch.setattr(Figure, "suptitle", synthetic_title)
    root, _, _ = campaign
    output = tmp_path / "synthetic-plots"
    original = {str(p.relative_to(root)): plot.digest(p) for p in root.rglob("*") if p.is_file()}
    manifest = plot.run(root, output)
    assert set(manifest["image_sha256"]) == set(plot.IMAGES)
    assert manifest["input_sha256"]["stage-manifest.json"] == original["stage-manifest.json"]
    assert "does not replay .npy" in manifest["scope"]["validation"]
    assert (output / "batch-throughput.png").read_bytes().startswith(b"\x89PNG")
    svg = (output / "batch-accuracy.svg").read_text()
    assert "0.001" in svg and "7102/7103" in svg and "9578a710479b" in svg
    assert "symlog" in svg and "linear near zero below 1e-8" in svg
    assert "Template-block evaluations / second" in (output / "batch-throughput.svg").read_text()
    assert original == {str(p.relative_to(root)): plot.digest(p) for p in root.rglob("*") if p.is_file()}
    # Verify does not need matplotlib and must not draw anything.
    monkeypatch.setattr(plot, "render", lambda *args: pytest.fail("verify must not render"))
    assert plot.run(root, output, verify_only=True) == manifest
    with pytest.raises(ValueError, match="new or empty"):
        plot.run(root, output)
    with (output / "batch-throughput.svg").open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="Image hash mismatch"):
        plot.run(root, output, verify_only=True)


def test_verify_rejects_stage_manifest_and_renderer_changes(campaign, tmp_path, monkeypatch):
    root, _, _ = campaign
    output = tmp_path / "plots"

    def fake_render(summary, accuracy, directory):
        for name in plot.IMAGES:
            (directory / name).write_bytes(b"synthetic image")
        return "test"

    monkeypatch.setattr(plot, "render", fake_render)
    plot.run(root, output)
    stage = root / "stage-manifest.json"
    before = stage.read_bytes()
    stage.write_bytes(before + b"\n")
    with pytest.raises(ValueError, match="Manifest input"):
        plot.run(root, output, verify_only=True)
    stage.write_bytes(before)
    change(output, "manifest.json", lambda data: data.update(renderer_sha256="0" * 64))
    with pytest.raises(ValueError, match="Manifest input"):
        plot.run(root, output, verify_only=True)


def test_output_cannot_overlap_inputs(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        plot.run(tmp_path, tmp_path / "plots")


@pytest.mark.parametrize("text", ['{"x": 1, "x": 2}', '{"x": NaN}'])
def test_ambiguous_json_rejected(text):
    with pytest.raises(ValueError):
        plot.read_json(text)
