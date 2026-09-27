"""Fail-closed comparison of captured inspiral event-manager decisions.

This gate is independent of the strict numerical comparator. It evaluates
observed decisions only; callers must separately qualify array/trigger fields
and provide the actual saved output order. The observer trace schema is the
one archived by the JAX event-manager decision audit.
"""

import math

import numpy as np

REQUIRED_STAGES = (
    "cluster_template_events",
    "finalize_template_events",
    "cut_events_via_mask",
    "newsnr_threshold",
    "consolidate_events",
    "write_events",
)
OPTIONAL_STAGES = (
    "chisq_threshold",
    "cut_events_via_indices",
    "keep_near_injection",
    "keep_loudest_in_interval",
)
_SENTINELS = {"observer_start", "observer_end"}


def _identity(row):
    if not isinstance(row, dict):
        raise ValueError("event row is not an object")
    values = (row.get("template_hash"), row.get("time_index"))
    if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ValueError(
            "event identity requires integer template_hash and time_index"
        )
    return values


def _identities(rows):
    if not isinstance(rows, list):
        raise ValueError("event snapshot is missing")
    keys = [_identity(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate event identity")
    return keys


def _is_subsequence(selected, source):
    iterator = iter(source)
    return all(
        any(candidate == key for candidate in iterator) for key in selected
    )


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is missing or nonnumeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} is nonfinite")
    return value


def _abs_snr(row):
    real = _number(row.get("snr_real"), "snr_real")
    imag = _number(row.get("snr_imag"), "snr_imag")
    # The captured event column is complex64. Match PyCBC's first abs() cast
    # before its float64 NewSNR calculation, including threshold boundaries.
    magnitude = float(np.abs(np.complex64(complex(real, imag))))
    if not math.isfinite(magnitude):
        raise ValueError("SNR magnitude is nonfinite")
    return magnitude


def _newsnr(row):
    snr = _abs_snr(row)
    chisq = _number(row.get("chisq"), "chisq")
    dof = row.get("chisq_dof")
    if isinstance(dof, bool) or not isinstance(dof, int) or dof <= 0:
        raise ValueError("chisq_dof must be a positive integer")
    reduced = chisq / dof
    if reduced < 0:
        raise ValueError("negative reduced chi-square")
    value = (
        snr * (0.5 * (1.0 + reduced**3.0)) ** (-1.0 / 6.0)
        if reduced > 1.0
        else snr
    )
    if not math.isfinite(value):
        raise ValueError("NewSNR score is nonfinite")
    return value


def _trace_stages(trace):
    if not isinstance(trace, list) or len(trace) < 2:
        raise ValueError("observer trace is missing")
    if any(not isinstance(record, dict) for record in trace):
        raise ValueError("observer record is not an object")
    if (
        trace[0].get("kind") != "observer_start"
        or trace[-1].get("kind") != "observer_end"
    ):
        raise ValueError("observer trace lacks start/end sentinels")
    if [record.get("sequence") for record in trace] != list(
        range(1, len(trace) + 1)
    ):
        raise ValueError("observer sequence is incomplete or out of order")
    stages = []
    for record in trace[1:-1]:
        kind = record.get("kind")
        if kind in _SENTINELS or kind not in REQUIRED_STAGES + OPTIONAL_STAGES:
            raise ValueError(f"unsupported observer stage {kind!r}")
        if not isinstance(record.get("detail"), dict):
            raise ValueError(f"{kind} detail is missing")
        _identities(record.get("before"))
        _identities(record.get("after"))
        stages.append(record)
    return stages


def _check_threshold(reference, candidate):
    threshold = _number(reference["detail"].get("threshold"), "threshold")
    if threshold != _number(candidate["detail"].get("threshold"), "threshold"):
        raise ValueError("threshold changed")
    reference_before = _identities(reference["before"])
    candidate_before = _identities(candidate["before"])
    if reference_before != candidate_before:
        raise ValueError("pre-cut identity/order mismatch")
    ref_scores = [_newsnr(row) for row in reference["before"]]
    cand_scores = [_newsnr(row) for row in candidate["before"]]
    ref_keep = [score >= threshold for score in ref_scores]
    cand_keep = [score >= threshold for score in cand_scores]
    if ref_keep != cand_keep:
        raise ValueError("threshold decisions differ")
    for ref_score, cand_score in zip(ref_scores, cand_scores):
        margin = abs(ref_score - threshold)
        drift = abs(cand_score - ref_score)
        if (margin == 0.0 and cand_score != ref_score) or (
            margin > 0.0 and not drift < margin
        ):
            raise ValueError("score drift reaches pristine threshold margin")
    expected_ref = [
        key for key, keep in zip(reference_before, ref_keep) if keep
    ]
    expected_cand = [
        key for key, keep in zip(candidate_before, cand_keep) if keep
    ]
    if (
        _identities(reference["after"]) != expected_ref
        or _identities(candidate["after"]) != expected_cand
    ):
        raise ValueError("captured retained events disagree with score replay")
    return dict(
        candidates=len(ref_scores),
        retained=len(expected_ref),
        minimum_margin_minus_drift=min(
            (
                abs(a - threshold) - abs(b - a)
                for a, b in zip(ref_scores, cand_scores)
            ),
            default=None,
        ),
    )


def _check_mask(reference, candidate):
    before = _identities(reference["before"])
    if before != _identities(candidate["before"]):
        raise ValueError("pre-cut identity/order mismatch")
    masks = []
    for record in (reference, candidate):
        mask = record["detail"].get("selected")
        if (
            not isinstance(mask, list)
            or len(mask) != len(before)
            or any(type(v) is not bool for v in mask)
        ):
            raise ValueError("selection mask is missing or invalid")
        masks.append(mask)
        expected = [key for key, keep in zip(before, mask) if keep]
        if _identities(record["after"]) != expected:
            raise ValueError("retained events disagree with native mask")
    if masks[0] != masks[1]:
        raise ValueError("selection decisions differ")
    return dict(candidates=len(before), retained=sum(masks[0]))


def _check_cluster(reference, candidate, cluster_score):
    if cluster_score != "abs_snr":
        raise ValueError("cluster ranking column must be declared as abs_snr")
    x = reference["detail"].get("window_size")
    y = candidate["detail"].get("window_size")
    if isinstance(x, bool) or not isinstance(x, int) or x <= 0 or x != y:
        raise ValueError("cluster window is missing or changed")
    for record in (reference, candidate):
        column = record["detail"].get("column", "snr")
        if column != "snr":
            raise ValueError("captured cluster ranking column is unsupported")
    before = _identities(reference["before"])
    if before != _identities(candidate["before"]):
        raise ValueError("pre-cluster identity/order mismatch")
    if len({template_hash for template_hash, _ in before}) > 1:
        raise ValueError("cluster call mixes template identities")
    times = [time_index for _, time_index in before]
    if times != sorted(times):
        raise ValueError("cluster input is not time sorted")
    scores_ref = [_abs_snr(row) for row in reference["before"]]
    scores_cand = [_abs_snr(row) for row in candidate["before"]]
    checked = 0
    min_remaining = None
    # Check every contestable pair, a conservative superset of the adjacent
    # comparisons in FindChirp's greedy clustering.
    for i, (hash_i, time_i) in enumerate(before):
        for j in range(i + 1, len(before)):
            hash_j, time_j = before[j]
            if hash_i != hash_j or abs(time_j - time_i) > x:
                continue
            gap = abs(scores_ref[i] - scores_ref[j])
            drift = abs(scores_cand[i] - scores_ref[i]) + abs(
                scores_cand[j] - scores_ref[j]
            )
            if (
                gap == 0.0
                and (
                    scores_cand[i] != scores_ref[i]
                    or scores_cand[j] != scores_ref[j]
                )
            ) or (gap > 0.0 and not drift < gap):
                raise ValueError(
                    "combined score drift reaches pristine ranking gap"
                )
            checked += 1
            remaining = gap - drift
            min_remaining = (
                remaining
                if min_remaining is None
                else min(min_remaining, remaining)
            )
    ref_after = _identities(reference["after"])
    cand_after = _identities(candidate["after"])
    if not _is_subsequence(ref_after, before) or not _is_subsequence(
        cand_after, before
    ):
        raise ValueError("cluster survivor is not an input subsequence")
    if ref_after != cand_after:
        raise ValueError("cluster survivor identity/order mismatch")
    return dict(
        candidates=len(before),
        survivors=len(reference["after"]),
        contestable_pairs=checked,
        minimum_gap_minus_drift=min_remaining,
    )


def compare_decision_traces(
    reference_trace,
    candidate_trace,
    *,
    cluster_score=None,
    reference_output=None,
    candidate_output=None,
    required_stages=REQUIRED_STAGES,
):
    """Compare observed decisions; saved output rows are required for a pass.

    ``cluster_score`` must name the actual clustered column. The current
    ``pycbc_inspiral`` observer omits it, while its call site uses ``snr``;
    callers must explicitly declare ``abs_snr`` after checking that call site.
    ``reference_output`` and ``candidate_output`` are rows read from the
    respective saved HDF files, in file order. Unknown stages and incomplete
    traces fail. Uninvoked optional cuts are reported as uncovered scope.
    """
    result = dict(
        passed=False,
        observed_decisions_pass=False,
        failures=[],
        stages=[],
        missing_scope=[],
        scope_limitations=[
            "early matched-filter candidate selection is outside this trace",
            "strict numerical array and trigger-field gates are separate",
        ],
        unexercised_optional_stages=[],
        scope="captured later event-manager stages only",
        complete_search_qualification=False,
    )
    try:
        reference = _trace_stages(reference_trace)
        candidate = _trace_stages(candidate_trace)
        kinds_ref = [record["kind"] for record in reference]
        kinds_cand = [record["kind"] for record in candidate]
        if kinds_ref != kinds_cand:
            raise ValueError("observer stage sequence differs")
        missing = [kind for kind in required_stages if kind not in kinds_ref]
        if missing:
            raise ValueError(f"required observer stages absent: {missing!r}")
        result["unexercised_optional_stages"] = [
            kind for kind in OPTIONAL_STAGES if kind not in kinds_ref
        ]
        result["scope_limitations"].extend(
            [
                f"unexercised optional stage: {kind}"
                for kind in result["unexercised_optional_stages"]
            ]
        )
        for number, (left, right) in enumerate(zip(reference, candidate)):
            kind = left["kind"]
            if kind in OPTIONAL_STAGES:
                raise ValueError(
                    f"no decision adapter for optional stage {kind!r}"
                )
            if kind == "newsnr_threshold":
                detail = _check_threshold(left, right)
            elif kind == "cut_events_via_mask":
                detail = _check_mask(left, right)
            elif kind == "cluster_template_events":
                detail = _check_cluster(left, right, cluster_score)
            else:
                if _identities(left["before"]) != _identities(right["before"]):
                    raise ValueError(
                        f"{kind} pre-stage identity/order mismatch"
                    )
                if kind != "write_events" and _identities(
                    left["after"]
                ) != _identities(right["after"]):
                    raise ValueError(
                        f"{kind} retained identity/order mismatch"
                    )
                if kind == "write_events":
                    for record in (left, right):
                        prewrite = set(_identities(record["before"]))
                        postwrite = set(_identities(record["after"]))
                        if prewrite != postwrite:
                            raise ValueError(
                                "write changed in-memory identities"
                            )
                detail = dict(candidates=len(left["before"]))
            result["stages"].append(
                dict(number=number, kind=kind, passed=True, **detail)
            )
        result["observed_decisions_pass"] = True
        if reference_output is None or candidate_output is None:
            result["missing_scope"].append("saved HDF trigger order")
        else:
            output_left = _identities(reference_output)
            output_right = _identities(candidate_output)
            writers = [
                row for row in reference if row["kind"] == "write_events"
            ]
            if len(writers) != 1:
                raise ValueError("exactly one output write is required")
            if output_left != output_right:
                raise ValueError("saved HDF trigger order differs")
            if set(output_left) != set(_identities(writers[0]["before"])):
                raise ValueError(
                    "saved output identities disagree with captured write"
                )
        result["passed"] = not result["missing_scope"]
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        result["failures"].append(str(exc))
    return result
