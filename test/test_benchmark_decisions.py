"""Boundary cases for the later event-manager decision gate."""

from copy import deepcopy
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "benchmark_decisions",
    Path(__file__).resolve().parents[1] / "tools" / "benchmark_decisions.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compare_decision_traces = _MODULE.compare_decision_traces


def _row(time_index, snr, template_hash=7):
    return dict(
        template_hash=template_hash,
        time_index=time_index,
        snr_real=snr,
        snr_imag=0.0,
        chisq=1.0,
        chisq_dof=1,
    )


def _trace(*records):
    body = [
        dict(
            sequence=1,
            kind="observer_start",
            before=None,
            after=None,
            detail={},
        )
    ]
    body.extend(deepcopy(records))
    body.append(dict(kind="observer_end", before=None, after=None, detail={}))
    for sequence, record in enumerate(body, 1):
        record["sequence"] = sequence
    return body


def _threshold_trace(scores, retained):
    rows = [_row(i, score) for i, score in enumerate(scores)]
    after = [rows[i] for i in retained]
    return _trace(
        dict(
            kind="newsnr_threshold",
            before=rows,
            after=after,
            detail={"threshold": 5.0},
        ),
        dict(kind="write_events", before=after, after=after, detail={}),
    )


def _compare_threshold(reference, candidate):
    return compare_decision_traces(
        reference,
        candidate,
        reference_output=reference[-2]["before"],
        candidate_output=candidate[-2]["before"],
        required_stages=("newsnr_threshold", "write_events"),
    )


def test_threshold_margin_and_zero_margin_exactness():
    ref = _threshold_trace([4.0, 5.0, 6.0], [1, 2])
    close = _threshold_trace([4.5, 5.0, 6.5], [1, 2])
    assert _compare_threshold(ref, close)["passed"]

    at_margin = _threshold_trace([5.0, 5.0, 6.0], [0, 1, 2])
    assert not _compare_threshold(ref, at_margin)["passed"]

    zero_margin_changed = _threshold_trace(
        [4.0, 5.000000476837158, 6.0], [1, 2]
    )
    failure = _compare_threshold(ref, zero_margin_changed)
    assert not failure["passed"]
    assert "threshold margin" in failure["failures"][0]


def test_threshold_replay_rejects_wrong_retained_rows_and_ambiguous_identity():
    ref = _threshold_trace([4.0, 6.0], [1])
    wrong = _threshold_trace([4.0, 6.0], [0])
    assert not _compare_threshold(ref, wrong)["passed"]

    duplicate = deepcopy(ref)
    duplicate[1]["before"][1]["time_index"] = 0
    assert not _compare_threshold(ref, duplicate)["passed"]


def _cluster_trace(scores, survivors):
    rows = [_row(100 + 2 * i, score) for i, score in enumerate(scores)]
    after = [rows[i] for i in survivors]
    return _trace(
        dict(
            kind="cluster_template_events",
            before=rows,
            after=after,
            detail={"window_size": 10},
        ),
        dict(kind="write_events", before=after, after=after, detail={}),
    )


def _compare_cluster(reference, candidate, **kwargs):
    return compare_decision_traces(
        reference,
        candidate,
        cluster_score="abs_snr",
        reference_output=reference[-2]["before"],
        candidate_output=candidate[-2]["before"],
        required_stages=("cluster_template_events", "write_events"),
        **kwargs
    )


def test_cluster_combined_drift_must_be_strictly_below_gap():
    ref = _cluster_trace([6.0, 5.0], [0])
    close = _cluster_trace([5.75, 5.25], [0])
    assert _compare_cluster(ref, close)["passed"]

    at_gap = _cluster_trace([5.5, 5.5], [0])
    failure = _compare_cluster(ref, at_gap)
    assert not failure["passed"]
    assert "ranking gap" in failure["failures"][0]

    loser_selected = _cluster_trace([5.75, 5.25], [1])
    assert not _compare_cluster(ref, loser_selected)["passed"]


def test_cluster_tie_requires_exact_scores_and_order():
    ref = _cluster_trace([5.0, 5.0], [0])
    assert _compare_cluster(ref, deepcopy(ref))["passed"]
    changed = _cluster_trace([5.0, 5.000000476837158], [0])
    assert not _compare_cluster(ref, changed)["passed"]


def test_missing_output_and_unsupported_stage_fail_closed():
    ref = _cluster_trace([6.0, 5.0], [0])
    incomplete = compare_decision_traces(
        ref,
        deepcopy(ref),
        cluster_score="abs_snr",
        required_stages=("cluster_template_events", "write_events"),
    )
    assert incomplete["observed_decisions_pass"]
    assert not incomplete["passed"]
    assert incomplete["missing_scope"] == ["saved HDF trigger order"]
    assert "chisq_threshold" in incomplete["unexercised_optional_stages"]
    assert not incomplete["complete_search_qualification"]
    assert any(
        "chisq_threshold" in item for item in incomplete["scope_limitations"]
    )

    undeclared = compare_decision_traces(
        ref,
        deepcopy(ref),
        reference_output=ref[-2]["before"],
        candidate_output=ref[-2]["before"],
        required_stages=("cluster_template_events", "write_events"),
    )
    assert not undeclared["passed"]

    broken = deepcopy(ref)
    broken[-1]["kind"] = "chisq_threshold"
    assert not _compare_cluster(ref, broken)["passed"]


def test_cluster_requires_sorted_inputs_and_real_subsequence():
    ref = _cluster_trace([6.0, 5.0], [0])
    unsorted = deepcopy(ref)
    unsorted[1]["before"] = unsorted[1]["before"][::-1]
    assert not _compare_cluster(ref, unsorted)["passed"]

    invented = deepcopy(ref)
    invented[1]["after"][0] = deepcopy(invented[1]["after"][0])
    invented[1]["after"][0]["time_index"] = 200
    assert not _compare_cluster(ref, invented)["passed"]


def test_saved_output_order_is_required_and_checked():
    ref = _threshold_trace([6.0, 7.0], [0, 1])
    result = compare_decision_traces(
        ref,
        deepcopy(ref),
        reference_output=ref[-2]["before"],
        candidate_output=ref[-2]["before"][::-1],
        required_stages=("newsnr_threshold", "write_events"),
    )
    assert not result["passed"]
    assert "saved HDF trigger order differs" in result["failures"]


def test_missing_stage_and_bad_observer_sequence_fail():
    ref = _threshold_trace([6.0], [0])
    result = compare_decision_traces(
        ref, deepcopy(ref), required_stages=("cluster_template_events",)
    )
    assert not result["passed"]
    assert "required observer stages absent" in result["failures"][0]

    broken = deepcopy(ref)
    broken[1]["sequence"] = 99
    assert not _compare_threshold(ref, broken)["passed"]
