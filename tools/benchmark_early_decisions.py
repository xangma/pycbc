"""Fail-closed, independent gate for the first inspiral selection decision.

The opt-in observer reduces the *actual* valid complex SNR series to the top
two squared scores and indices in every fixed window, then stores the native
cluster output. This gate replays the symmetric decision and bounds every
decision-relevant comparison by the pristine gap.
It deliberately does not qualify chi-square, later event-manager decisions,
or the separate strict numerical science gate.
"""

import argparse
import json
import math

import h5py
import numpy as np


def _scores(series):
    values = np.asarray(series, dtype=np.complex64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("valid SNR series is missing, nonfinite, or not one-dimensional")
    return values.real * values.real + values.imag * values.imag


def _replay(scores, threshold_sq, window):
    if not isinstance(window, int) or window <= 0 or not len(scores):
        raise ValueError("empty series or invalid cluster window")
    if not math.isfinite(threshold_sq) or threshold_sq <= 0:
        raise ValueError("invalid squared SNR threshold")
    starts = np.arange(0, len(scores), window, dtype=np.int64)
    peaks = np.empty(len(starts), dtype=np.int64)
    maxima = np.empty(len(starts), dtype=np.float32)
    for block, start in enumerate(starts):
        end = min(int(start) + window, len(scores))
        peaks[block] = start + int(np.argmax(scores[start:end]))
        maxima[block] = scores[peaks[block]]
    keep = maxima > threshold_sq
    if len(maxima) > 1:
        keep[0] &= maxima[0] > maxima[1]
        keep[-1] &= maxima[-1] > maxima[-2]
        if len(maxima) > 2:
            keep[1:-1] &= (maxima[1:-1] > maxima[:-2]) & (
                maxima[1:-1] >= maxima[2:]
            )
    return peaks, maxima, peaks[keep]


def summarize_series(series, window):
    """Sufficient window statistics for the symmetric clustering decision."""
    scores = _scores(series)
    if not isinstance(window, int) or window <= 0 or not len(scores):
        raise ValueError("empty series or invalid cluster window")
    top_index, runner_index = [], []
    top_score, runner_score = [], []
    for start in range(0, len(scores), window):
        block = scores[start:start + window]
        first = int(np.argmax(block))
        top_index.append(start + first)
        top_score.append(block[first])
        if len(block) == 1:
            runner_index.append(-1)
            runner_score.append(-1)
        else:
            rivals = block.copy()
            rivals[first] = -1
            second = int(np.argmax(rivals))
            runner_index.append(start + second)
            runner_score.append(block[second])
    return (
        np.asarray(top_index, dtype=np.int64),
        np.asarray(top_score, dtype=np.float32),
        np.asarray(runner_index, dtype=np.int64),
        np.asarray(runner_score, dtype=np.float32),
    )


def replay_summary(top_index, top_score, threshold_sq):
    """Replay threshold and adjacent-window clustering from top scores."""
    if len(top_index) != len(top_score) or not len(top_index):
        raise ValueError("window summary is missing")
    keep = top_score > threshold_sq
    if len(top_score) > 1:
        keep[0] &= top_score[0] > top_score[1]
        keep[-1] &= top_score[-1] > top_score[-2]
        if len(top_score) > 2:
            keep[1:-1] &= (top_score[1:-1] > top_score[:-2]) & (
                top_score[1:-1] >= top_score[2:]
            )
    return top_index[keep]


def _compare_contest(ref, cand, winner):
    """Return the minimum remaining gap for all challengers, or fail."""
    if len(ref) != len(cand) or winner < 0 or winner >= len(ref):
        raise ValueError("ranking contest has mismatched scores")
    if len(ref) < 2:
        return None
    rivals = np.arange(len(ref)) != winner
    gap = ref[winner].astype(np.float64) - ref[rivals].astype(np.float64)
    drift = abs(float(cand[winner]) - float(ref[winner])) + np.abs(
        cand[rivals].astype(np.float64) - ref[rivals].astype(np.float64)
    )
    if np.any(gap < 0):
        raise ValueError("pristine ranking winner is inconsistent")
    if np.any((gap == 0) & (drift != 0)) or np.any((gap > 0) & (drift >= gap)):
        raise ValueError("combined score drift reaches pristine ranking gap or tie")
    return float(np.min(gap - drift))


def _rows(hdf):
    if hdf.attrs.get("schema") != "pycbc-early-selection-v1" or not bool(
        hdf.attrs.get("complete", False)
    ):
        raise ValueError("early observer trace is missing or incomplete")
    expected_templates = int(hdf.attrs["expected_templates"])
    expected_segments = int(hdf.attrs["expected_segments"])
    if expected_templates != 32 or expected_segments <= 0:
        raise ValueError("trace is outside the predeclared 32-template scope")
    rows = hdf.get("rows")
    if rows is None or len(rows) != expected_templates * expected_segments:
        raise ValueError("template/segment coverage is incomplete")
    keys = {}
    for name, row in rows.items():
        key = (int(row.attrs["template_index"]), int(row.attrs["segment_index"]))
        if key in keys or name != f"{key[0]:03d}-{key[1]:03d}":
            raise ValueError("duplicate or malformed template/segment identity")
        keys[key] = row
    if {t for t, _ in keys} != set(range(32)) or {
        s for _, s in keys
    } != set(range(expected_segments)):
        raise ValueError("template/segment grid is incomplete")
    return keys


def _checked(row):
    length = int(row.attrs["series_length"])
    threshold_sq = float(row.attrs["threshold_sq"])
    window = int(row.attrs["window"])
    if length <= 0 or window <= 0 or not math.isfinite(threshold_sq) or threshold_sq <= 0:
        raise ValueError("invalid series length, window, or threshold")
    top_index = np.asarray(row["top_index"], dtype=np.int64)
    top_score = np.asarray(row["top_score_sq"], dtype=np.float32)
    runner_index = np.asarray(row["runner_index"], dtype=np.int64)
    runner_score = np.asarray(row["runner_score_sq"], dtype=np.float32)
    nblocks = (length + window - 1) // window
    if any(array.ndim != 1 or len(array) != nblocks for array in (
        top_index, top_score, runner_index, runner_score
    )):
        raise ValueError("window summary coverage is incomplete")
    if not np.isfinite(top_score).all() or not np.isfinite(runner_score).all():
        raise ValueError("window summary contains nonfinite scores")
    for block in range(nblocks):
        start, stop = block * window, min((block + 1) * window, length)
        if not start <= top_index[block] < stop or top_score[block] < 0:
            raise ValueError("invalid top window entry")
        if stop - start == 1:
            if runner_index[block] != -1 or runner_score[block] != -1:
                raise ValueError("invalid singleton runner entry")
        elif (not start <= runner_index[block] < stop
              or runner_index[block] == top_index[block]
              or runner_score[block] < 0
              or runner_score[block] > top_score[block]):
            raise ValueError("invalid runner window entry")
    selected = replay_summary(top_index, top_score.copy(), threshold_sq)
    native = np.asarray(row["selected_index"], dtype=np.int64)
    if native.ndim != 1 or not np.array_equal(native, selected):
        raise ValueError("native cluster output disagrees with window-summary replay")
    return length, threshold_sq, window, top_index, top_score, runner_index, runner_score, selected


def compare_early_decisions(reference_path, candidate_path):
    """Compare two observer HDFs; never claim full-search qualification."""
    result = dict(
        passed=False,
        early_stages_passed=False,
        complete_search_qualification=False,
        rows_checked=0,
        samples_checked=0,
        minimum_threshold_margin_minus_drift=None,
        minimum_ranking_gap_minus_drift=None,
        missing_scope=[
            "chi-square and later event-manager decisions",
            "strict numerical science gate",
        ],
        failures=[],
    )
    try:
        with h5py.File(reference_path) as ref_hdf, h5py.File(candidate_path) as cand_hdf:
            ref_rows, cand_rows = _rows(ref_hdf), _rows(cand_hdf)
            if ref_rows.keys() != cand_rows.keys():
                raise ValueError("template/segment identity grids differ")
            if int(ref_hdf.attrs["expected_segments"]) != int(
                cand_hdf.attrs["expected_segments"]
            ):
                raise ValueError("segment counts differ")
            for key in sorted(ref_rows):
                left, right = ref_rows[key], cand_rows[key]
                if int(left.attrs["template_hash"]) != int(right.attrs["template_hash"]):
                    raise ValueError(f"template hash differs at {key}")
                if int(left.attrs["analyze_start"]) != int(right.attrs["analyze_start"]):
                    raise ValueError(f"analysis offset differs at {key}")
                la, ta, wa, pa, ma, ra, qa, sa = _checked(left)
                lb, tb, wb, pb, mb, rb, qb, sb = _checked(right)
                if la != lb or wa != wb or len(pa) != len(pb):
                    raise ValueError(f"valid range or cluster window differs at {key}")
                if not np.array_equal(sa, sb):
                    raise ValueError(f"cluster survivor identities differ at {key}")
                threshold_margin = np.abs(ma.astype(np.float64) - ta)
                threshold_drift = np.abs(
                    (mb.astype(np.float64) - tb) - (ma.astype(np.float64) - ta)
                )
                if np.any((threshold_margin == 0) & (threshold_drift != 0)) or np.any(
                    (threshold_margin > 0) & (threshold_drift >= threshold_margin)
                ):
                    raise ValueError(f"threshold drift reaches pristine margin at {key}")
                residual = float(np.min(threshold_margin - threshold_drift))
                previous = result["minimum_threshold_margin_minus_drift"]
                result["minimum_threshold_margin_minus_drift"] = (
                    residual if previous is None else min(previous, residual)
                )
                for block, (p_ref, p_cand) in enumerate(zip(pa, pb)):
                    if p_ref != p_cand:
                        raise ValueError(f"within-window peak identity differs at {key}, block {block}")
                    # An unselected window cannot affect clustering. For a
                    # threshold-eligible window, the captured runner is the
                    # strongest possible rival in each arm. Require its
                    # identity to be stable so paired score drift is defined.
                    if ma[block] > ta:
                        if ra[block] != rb[block]:
                            raise ValueError(f"within-window runner identity differs at {key}, block {block}")
                        if ra[block] < 0:
                            continue
                        gap = _compare_contest(
                            np.asarray([ma[block], qa[block]]),
                            np.asarray([mb[block], qb[block]]), 0,
                        )
                        previous = result["minimum_ranking_gap_minus_drift"]
                        result["minimum_ranking_gap_minus_drift"] = (
                            gap if previous is None else min(previous, gap)
                        )
                for block in range(len(ma) - 1):
                    other = block + 1
                    if ma[block] <= ta and ma[other] <= ta:
                        continue
                    winner = 0 if ma[block] >= ma[other] else 1
                    gap = _compare_contest(
                        ma[np.array([block, other])],
                        mb[np.array([block, other])], winner,
                    )
                    previous = result["minimum_ranking_gap_minus_drift"]
                    result["minimum_ranking_gap_minus_drift"] = (
                        gap if previous is None else min(previous, gap)
                    )
                result["rows_checked"] += 1
                result["samples_checked"] += la
        result["early_stages_passed"] = True
        result["passed"] = True
    except (KeyError, OSError, TypeError, ValueError, OverflowError) as exc:
        result["failures"].append(str(exc))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_hdf")
    parser.add_argument("candidate_hdf")
    args = parser.parse_args(argv)
    result = compare_early_decisions(args.reference_hdf, args.candidate_hdf)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
