"""Small, complete-grid boundary tests for early inspiral decisions."""

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from tools.benchmark_early_decisions import (
    compare_early_decisions, replay_summary, summarize_series,
)
from tools.observe_inspiral_early import EarlyObserver, _observed_tree


ROOT = Path(__file__).resolve().parents[1]


def _trace(path, first=None, *, complete=True, corrupt=False):
    default = np.array([0, 4, 0, 0], dtype=np.complex64)
    with h5py.File(path, "w") as out:
        out.attrs["schema"] = "pycbc-early-selection-v1"
        out.attrs["complete"] = complete
        out.attrs["expected_templates"] = 32
        out.attrs["expected_segments"] = 1
        rows = out.create_group("rows")
        for template_index in range(32):
            series = np.asarray(first if template_index == 0 and first is not None
                                else default, dtype=np.complex64)
            row = rows.create_group(f"{template_index:03d}-000")
            row.attrs["template_index"] = template_index
            row.attrs["segment_index"] = 0
            row.attrs["template_hash"] = template_index + 100
            row.attrs["window"] = 2
            row.attrs["threshold_sq"] = 9.0
            row.attrs["series_length"] = len(series)
            row.attrs["analyze_start"] = 0
            top, top_score, runner, runner_score = summarize_series(series, 2)
            row.create_dataset("top_index", data=top)
            row.create_dataset("top_score_sq", data=top_score)
            row.create_dataset("runner_index", data=runner)
            row.create_dataset("runner_score_sq", data=runner_score)
            selected = replay_summary(top, top_score.copy(), 9.0)
            if corrupt and template_index == 0:
                selected = np.array([], dtype=np.int64)
            row.create_dataset("selected_index", data=selected)


def _compare(tmp_path, reference=None, candidate=None, **kwargs):
    a, b = tmp_path / "reference.hdf", tmp_path / "candidate.hdf"
    _trace(a, reference)
    _trace(b, candidate, **kwargs)
    return compare_early_decisions(a, b)


def test_complete_grid_and_margin_pass(tmp_path):
    result = _compare(tmp_path, [0, 4, 0, 0], [0, 3.9, 0, 0])
    assert result["passed"]
    assert result["rows_checked"] == 32
    assert result["samples_checked"] == 128
    assert not result["complete_search_qualification"]
    assert result["missing_scope"]


def test_threshold_margin_is_strict_even_with_same_retained_peak(tmp_path):
    result = _compare(tmp_path, [0, 4, 0, 0], [0, 5, 0, 0])
    assert not result["passed"]
    assert "threshold drift" in result["failures"][0]


def test_ranking_gap_and_tie_are_strict(tmp_path):
    assert _compare(tmp_path, [5, 4, 0, 0], [4.7, 4.3, 0, 0])["passed"]
    at_gap = _compare(tmp_path, [5, 4, 0, 0], [4, 4, 0, 0])
    assert not at_gap["passed"]
    assert "ranking gap or tie" in at_gap["failures"][0]
    changed_tie = _compare(
        tmp_path, [5, 5, 0, 0], [5, np.nextafter(np.float32(5), np.float32(0)), 0, 0]
    )
    assert not changed_tie["passed"]
    assert "ranking gap or tie" in changed_tie["failures"][0]


def test_adjacent_tie_and_partial_last_window(tmp_path):
    tied = _compare(tmp_path, [4, 0, 4, 0], [3.9, 0, 3.9, 0])
    assert not tied["passed"]
    assert "ranking gap or tie" in tied["failures"][0]
    top, score, runner, runner_score = summarize_series(
        np.asarray([0, 4, 0, 0, 5], dtype=np.complex64), 2
    )
    assert top.tolist() == [1, 2, 4]
    assert runner.tolist() == [0, 3, -1]
    assert runner_score[-1] == -1
    assert replay_summary(top, score.copy(), 9.0).tolist() == [1, 4]


def test_missing_or_corrupt_observation_fails_closed(tmp_path):
    bad = _compare(tmp_path, corrupt=True)
    assert not bad["passed"]
    assert "native cluster output" in bad["failures"][0]
    a, b = tmp_path / "reference.hdf", tmp_path / "candidate.hdf"
    _trace(a)
    _trace(b, complete=False)
    assert not compare_early_decisions(a, b)["passed"]
    with h5py.File(b, "r+") as out:
        out.attrs["complete"] = True
        del out["rows/031-000"]
    assert not compare_early_decisions(a, b)["passed"]


def test_analysis_offset_must_match(tmp_path):
    a, b = tmp_path / "reference.hdf", tmp_path / "candidate.hdf"
    _trace(a)
    _trace(b)
    with h5py.File(b, "r+") as out:
        out["rows/000-000"].attrs["analyze_start"] = 1
    result = compare_early_decisions(a, b)
    assert not result["passed"]
    assert "analysis offset differs" in result["failures"][0]


def test_observer_compiles_both_executable_shapes_without_source_edits():
    pristine = subprocess.check_output(
        ["git", "show", "40e94792b3:bin/pycbc_inspiral"],
        cwd=ROOT, text=True,
    )
    candidate = (ROOT / "bin/pycbc_inspiral").read_text()
    for name, source in (("pristine", pristine), ("candidate", candidate)):
        tree = _observed_tree(source, name)
        compile(tree, name, "exec")
        methods = [node.func.attr for node in ast.walk(tree)
                   if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute)
                   and isinstance(node.func.value, ast.Name)
                   and node.func.value.id == "__pycbc_early"]
        assert methods.count("scalar") == methods.count("finish") == 1
        assert methods.count("begin_batch") == methods.count("batch")
        assert methods.count("batch") == (name == "candidate")


def test_observer_writes_compact_no_hit_cells_and_requires_full_grid(tmp_path):
    path = tmp_path / "observed.hdf"
    observer = EarlyObserver(path, 1)
    stilde = SimpleNamespace(analyze=slice(0, 4))
    control = SimpleNamespace(snr_threshold=3.0)
    for template_index in range(31):
        template = SimpleNamespace(
            params=SimpleNamespace(template_hash=template_index + 100)
        )
        series = np.asarray([0, 4, 0, 0] if template_index else [0, 0, 0, 0],
                            dtype=np.complex64)
        indices = np.asarray([1] if template_index else [], dtype=np.int64)
        observer._save(
            template_index, 0, template, stilde, control, 2, 1.0,
            indices, series[indices], series, "cpu",
        )
    with pytest.raises(RuntimeError, match="incomplete"):
        observer.finish()
    with h5py.File(path) as out:
        assert not out.attrs["complete"]
        assert "valid_snr" not in out["rows/001-000"]
        assert len(out["rows/000-000/selected_index"]) == 0
    template = SimpleNamespace(params=SimpleNamespace(template_hash=131))
    series = np.asarray([0, 4, 0, 0], dtype=np.complex64)
    observer._save(31, 0, template, stilde, control, 2, 1.0,
                   np.asarray([1]), series[[1]], series, "cpu")
    observer.finish()
    with h5py.File(path) as out:
        assert out.attrs["complete"]
        assert len(out["rows"]) == 32


def test_one_template_one_segment_smoke_trace_is_not_qualification(tmp_path):
    path = tmp_path / "smoke.hdf"
    observer = EarlyObserver(path, expected_segments=1, expected_templates=1)
    series = np.asarray([0, 4, 0], dtype=np.complex64)
    observer._save(
        0, 0, SimpleNamespace(params=SimpleNamespace(template_hash=100)),
        SimpleNamespace(analyze=slice(0, 3)),
        SimpleNamespace(snr_threshold=3.0),
        2, 1.0, np.asarray([1]), series[[1]], series, "cpu",
    )
    observer.finish()
    with h5py.File(path) as out:
        assert out.attrs["complete"]
        assert int(out.attrs["expected_templates"]) == 1
    result = compare_early_decisions(path, path)
    assert not result["passed"]
    assert "32-template scope" in result["failures"][0]
