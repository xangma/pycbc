"""Offline checks with synthetic summaries, not claimed benchmark results."""

import copy
import importlib.util
import json
from pathlib import Path
from statistics import median

import pytest


SPEC = importlib.util.spec_from_file_location(
    "plot_executable", Path(__file__).resolve().parents[1]
    / "tools/plot_torch_executable.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)
HANDOFF = "handoff.json"
VERIFIED = "evidence-verification.json"


@pytest.fixture
def summaries():
    # Unordered times expose selection of a sample instead of the median.
    # Receipt/HDF/qualification paths intentionally do not exist.
    timing, qualifications, science = {}, {}, {}
    for backend, values in (("cpu", [70., 60., 80.]),
                            ("torchcpu", [120., 100., 110.]),
                            ("cuda", [30., 20., 25.])):
        path = f"/nonexistent/qualify-{backend}/qualification.json"
        qualifications[backend] = dict(
            path=path, sha256="a" * 64, templates=384, segments=5,
            valid_seconds=1904,
            checks={key: True for key in plot.QUALIFICATION_CHECKS})
        rows = [dict(
            case=f"timing-{i}-{backend}", full_wall_seconds=value,
            templates=384, scheme=plot.BACKENDS[backend],
            valid_intervals=copy.deepcopy(plot.INTERVALS),
            inferred_templates=384., inferred_segments=5., qualification=path,
            template_seconds_per_wall_second=731136/value,
            receipt=f"/nonexistent/timing-{i}-{backend}/receipt.json",
            trigger_sha256="b" * 64, receipt_sha256="c" * 64)
            for i, value in enumerate(values, 1)]
        timing[backend] = dict(
            samples=rows, median_wall_seconds=median(values),
            min_wall_seconds=min(values), max_wall_seconds=max(values),
            median_template_seconds_per_wall_second=731136/median(values),
            relative_range=(max(values)-min(values))/median(values),
            median_wall_sample=copy.deepcopy(next(
                row for row in rows
                if row["full_wall_seconds"] == median(values))))
    for case in plot.SCIENCE_CASES:
        science[case] = dict(
            status="pass", failures=[], review_reasons=[], sample_rate=4096,
            candidate_sha256="b" * 64, candidate_receipt_sha256="c" * 64,
            detectors={"H1": dict(
                baseline_count=1991, candidate_count=1991, matched_count=1991,
                valid_intervals=copy.deepcopy(plot.INTERVALS),
                metrics={key: dict(violations=0, max_absolute_error=1e-6,
                                   max_relative_error=1e-7)
                         for key in plot.SCIENCE_FIELDS},
                unmatched={side: dict(count=0, classifications={}, examples=[])
                           for side in ("baseline_only", "candidate_only")})})
    return {
        HANDOFF: dict(
            state="complete", revision=plot.REVISION,
            numerical_budgets=plot.NUMERICAL_BUDGETS.copy(),
            release=dict(state="validated-release", host="len",
                         source_unchanged=True,
                         all_source_native_input_helper_pins_unchanged=True),
            timing={backend: {key: value for key, value in entry.items()
                              if key not in ("samples", "median_wall_sample")}
                    for backend, entry in timing.items()}),
        VERIFIED: dict(
            state="pass", revision=plot.REVISION,
            numerical_budgets=plot.NUMERICAL_BUDGETS.copy(),
            native_input_pins_verified=True, qualifications=qualifications,
            scientific_comparisons=science, timing=timing),
    }


@pytest.fixture
def evidence(tmp_path, monkeypatch, summaries):
    root = tmp_path / "synthetic-evidence"
    root.mkdir()
    hashes = {}
    for name, value in summaries.items():
        data = json.dumps(value).encode()
        (root / name).write_bytes(data)
        hashes[name] = plot.fingerprint(data)["sha256"]
    # Only unit fixtures substitute trust anchors; the CLI has no override.
    monkeypatch.setattr(plot, "SUMMARY_SHA256", hashes)
    return root


def test_recomputes_medians_ranges_and_rates(summaries):
    result = plot.validate_summaries(summaries)
    assert result["template_seconds"] == 731136
    assert result["full_wall"]["cpu"]["median_wall_seconds"] == 70
    assert result["full_wall"]["torchcpu"]["median_wall_seconds"] == 110
    cuda = result["full_wall"]["cuda"]
    assert cuda["min_wall_seconds"] == 20
    assert cuda["max_wall_seconds"] == 30
    assert cuda["median_wall_seconds"] == 25
    assert cuda["relative_range"] == .4
    assert cuda["median_template_seconds_per_wall_second"] == 731136/25
    assert result["samples"]["cuda"][0][
        "template_seconds_per_wall_second"] == 731136/30
    assert len(result["science"]) == 20
    assert all(item["status"] == "pass" for item in result["science"].values())


@pytest.mark.parametrize("name", plot.SUMMARY_SHA256)
def test_tampered_source_cannot_replace_fixed_hashes(evidence, name):
    path = evidence / name
    path.write_bytes(path.read_bytes() + b"\n")
    # A supplied inventory must not launder modified source bytes.
    (evidence / "SHA256SUMS").write_text(
        f"{plot.fingerprint(path.read_bytes())['sha256']}  {name}\n")
    with pytest.raises(ValueError, match="Pinned SHA256 mismatch"):
        plot.load_evidence(evidence)


@pytest.mark.parametrize("change", [
    "missing", "duplicate", "extra", "reorder", "profile", "scheme",
    "templates", "interval", "segments", "qualification", "trigger", "receipt",
    "rate", "nonfinite", "infinite", "negative", "zero", "boolean", "string"])
def test_rejects_bad_timing_cohort(summaries, change):
    rows = summaries[VERIFIED]["timing"]["cuda"]["samples"]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[2] = copy.deepcopy(rows[0])
    elif change == "extra":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "reorder":
        rows.reverse()
    else:
        key, value = {
            "profile": ("case", "cuda-trace"),
            "scheme": ("scheme", "torch:cpu:1"),
            "templates": ("templates", 383),
            "interval": ("valid_intervals", [[1187007160., 1187009065.]]),
            "segments": ("inferred_segments", 4),
            "qualification": ("qualification", "/wrong/qualification.json"),
            "trigger": ("trigger_sha256", "0" * 64),
            "receipt": ("receipt_sha256", "0" * 64),
            "rate": ("template_seconds_per_wall_second", 1),
            "nonfinite": ("full_wall_seconds", float("nan")),
            "infinite": ("full_wall_seconds", float("inf")),
            "negative": ("full_wall_seconds", -1),
            "zero": ("full_wall_seconds", 0),
            "boolean": ("full_wall_seconds", True),
            "string": ("full_wall_seconds", "30"),
        }[change]
        rows[0][key] = value
    with pytest.raises(ValueError):
        plot.validate_summaries(summaries)


@pytest.mark.parametrize("file", [HANDOFF, VERIFIED])
@pytest.mark.parametrize("key", [
    "median_wall_seconds", "min_wall_seconds", "max_wall_seconds",
    "median_template_seconds_per_wall_second", "relative_range"])
def test_rejects_inconsistent_timing_summaries(summaries, file, key):
    summaries[file]["timing"]["cpu"][key] += 1
    with pytest.raises(ValueError):
        plot.validate_summaries(summaries)


@pytest.mark.parametrize("file", [HANDOFF, VERIFIED])
@pytest.mark.parametrize("key,value", [
    ("revision", "0" * 40), ("state", "failed"),
    ("numerical_budgets", {**plot.NUMERICAL_BUDGETS, "rtol": .001})])
def test_rejects_changed_source_or_status(summaries, file, key, value):
    summaries[file][key] = value
    with pytest.raises(ValueError):
        plot.validate_summaries(summaries)


@pytest.mark.parametrize("change", [
    "missing-backend", "extra-backend", "missing-qualification", "coverage",
    "empty-checks", "missing-check", "failed-check", "truthy-check",
    "native-pins", "source-pins", "host", "median-worker"])
def test_rejects_incomplete_qualification_or_provenance(summaries, change):
    verified = summaries[VERIFIED]
    qualification = verified["qualifications"]["cpu"]
    if change == "missing-backend":
        verified["timing"].pop("cuda")
    elif change == "extra-backend":
        verified["timing"]["profiled-cuda"] = verified["timing"]["cuda"]
    elif change == "missing-qualification":
        verified["qualifications"].pop("cpu")
    elif change == "coverage":
        qualification["valid_seconds"] = 1903
    elif change == "empty-checks":
        qualification["checks"] = {}
    elif change == "missing-check":
        qualification["checks"].pop("bank_0_all_templates_once")
    elif change in ("failed-check", "truthy-check"):
        qualification["checks"]["bank_0_all_templates_once"] = (
            False if change == "failed-check" else 1)
    elif change == "native-pins":
        verified["native_input_pins_verified"] = False
    elif change == "source-pins":
        summaries[HANDOFF]["release"]["source_unchanged"] = False
    elif change == "host":
        summaries[HANDOFF]["release"]["host"] = "other-host"
    else:
        verified["timing"]["cpu"]["median_wall_sample"] = copy.deepcopy(
            verified["timing"]["cpu"]["samples"][1])
    with pytest.raises(ValueError):
        plot.validate_summaries(summaries)


@pytest.mark.parametrize("change", [
    "missing-case", "extra-case", "failure", "review", "status", "identities",
    "missing-field", "violation", "boolean-violation", "nonfinite-error",
    "unmatched", "missing-detector", "interval"])
def test_rejects_failed_or_incomplete_science(summaries, change):
    science = summaries[VERIFIED]["scientific_comparisons"]
    row = science["timing-2-cuda"]
    h1 = row["detectors"]["H1"]
    if change == "missing-case":
        science.pop("qualify-cuda")
    elif change == "extra-case":
        science["old-run"] = copy.deepcopy(row)
    elif change == "failure":
        row["failures"] = ["failed comparison"]
    elif change == "review":
        row["review_reasons"] = ["unresolved"]
    elif change == "status":
        row["status"] = "fail"
    elif change == "identities":
        h1["matched_count"] = 1990
    elif change == "missing-field":
        h1["metrics"].pop("snr")
    elif change in ("violation", "boolean-violation"):
        h1["metrics"]["snr"]["violations"] = (
            1 if change == "violation" else False)
    elif change == "nonfinite-error":
        h1["metrics"]["snr"]["max_relative_error"] = float("nan")
    elif change == "unmatched":
        h1["unmatched"]["candidate_only"]["count"] = 1
    elif change == "missing-detector":
        row["detectors"] = {}
    else:
        h1["valid_intervals"] = []
    with pytest.raises(ValueError):
        plot.validate_summaries(summaries)


def test_render_is_deterministic_and_uses_zero_axis(summaries, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    figures = []
    original = Figure.savefig

    def capture(fig, *args, **kwargs):
        figures.append(fig)
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    result = plot.validate_summaries(summaries)
    first, second = plot.render(result), plot.render(result)
    assert first == second
    assert figures[0].axes[0].get_ylim()[0] == 0
    assert first["executable-wall.png"].startswith(b"\x89PNG\r\n\x1a\n")
    svg = first["executable-wall.svg"].decode()
    assert "<dc:date>" not in svg
    assert all(line == line.rstrip() for line in svg.splitlines())
    for label in ("CPU / MKL", "Torch CPU", "Torch CUDA", "70.000 s",
                  "110.000 s", "25.000 s", "All 21 science outputs passed"):
        assert label in svg
    assert "speedup" not in svg and "less median" not in svg


@pytest.fixture
def rendered(evidence, tmp_path, monkeypatch):
    output = tmp_path / "plots"
    expected = plot.load_evidence(evidence)
    # Manifest checks do not need matplotlib; actual rendering is tested above.
    monkeypatch.setattr(plot, "render", lambda _: {
        name: name.encode() for name in plot.IMAGES})
    plot.write_output(output, expected)
    return output, expected


def test_verify_only_is_read_only(evidence, rendered, monkeypatch):
    output, _ = rendered
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
              for p in output.iterdir()}

    def forbidden(_):
        pytest.fail("verify-only attempted rendering")

    monkeypatch.setattr(plot, "render", forbidden)
    assert plot.main(["--evidence", str(evidence), "--output", str(output),
                      "--verify-only"]) == 0
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in output.iterdir()}


@pytest.mark.parametrize("change", [
    *plot.IMAGES, "manifest.json", "boolean-manifest", "renderer", "missing",
    "extra-file", "symlink-file", "symlink-directory"])
def test_verify_rejects_changed_artifacts(rendered, change, tmp_path):
    output, expected = rendered
    if change == "renderer":
        expected["renderer"]["sha256"] = "0" * 64
    elif change == "missing":
        (output / plot.IMAGES[0]).unlink()
    elif change in ("manifest.json", "boolean-manifest"):
        path = output / "manifest.json"
        manifest = json.loads(path.read_bytes())
        if change == "boolean-manifest":
            manifest["schema_version"] = True
        else:
            wall = manifest["result"]["full_wall"]["cpu"]
            wall["median_wall_seconds"] = 99
        path.write_text(json.dumps(manifest))
    elif change == "extra-file":
        (output / "extra.png").write_bytes(b"extra")
    elif change == "symlink-file":
        path = output / plot.IMAGES[0]
        target = tmp_path / "target.png"
        path.rename(target)
        path.symlink_to(target)
    elif change == "symlink-directory":
        link = tmp_path / "linked-output"
        link.symlink_to(output, target_is_directory=True)
        output = link
    else:
        with (output / change).open("ab") as stream:
            stream.write(b"tampered")
    with pytest.raises(ValueError):
        plot.verify_output(output, expected)


def test_nonempty_output_is_never_overwritten(rendered):
    output, expected = rendered
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="new or empty"):
        plot.write_output(output, expected)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_render_rejects_symlink_output(rendered, tmp_path):
    _, expected = rendered
    target = tmp_path / "empty"
    target.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="new or empty"):
        plot.write_output(link, expected)
    assert list(target.iterdir()) == []


def test_invalid_evidence_leaves_no_output(evidence, tmp_path):
    (evidence / HANDOFF).write_text("{}")
    output = tmp_path / "must-not-exist"
    assert plot.main(["--evidence", str(evidence),
                      "--output", str(output)]) == 1
    assert not output.exists()


@pytest.mark.parametrize("data", [
    '{"wall": 1, "wall": 2}', '{"wall": NaN}', '{"wall": Infinity}',
    '{"wall": -Infinity}', '{"wall": 1e999}',
    '{"nested": [{"a": 1, "a": 2}]}'])
def test_ambiguous_or_nonfinite_json_is_rejected(data):
    with pytest.raises(ValueError):
        plot.read_json(data)
