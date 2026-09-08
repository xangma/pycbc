#!/usr/bin/env python3
"""Analyze only the two frozen CPU qualification outputs; never rerun selection.

Requires NumPy, SciPy, h5py and the sibling acquisition/compare-triggers.py.
Run with -B to avoid bytecode writes. Inputs and frozen git objects are read-only.
The report and JSON are written next to this script unless --output-dir is given.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import h5py
import numpy as np
import scipy
from scipy.stats import kendalltau, spearmanr


FROZEN = {
    "original": ("40e94792b3edf59f39b18b65102b28a4f74433a7", 1988,
                 "5aefb746a8efb64b8d68ed552cd37deae5741913bee37b6edc7690c43511bbc8"),
    "proposed": ("123e1fb3ef1b338cada636e71c3e9c7987002402", 1991,
                 "7201381d82a11704a6cfedf0851355ddf516997eab80d505c67a2b36e74da942"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = np.asarray(values, dtype=float)
    return dict(zip(("min", "p05", "median", "p95", "max"),
                    map(float, np.quantile(values, [0, .05, .5, .95, 1]))))


def newsnr(data):
    dof = 2 * data["chisq_dof"].astype(np.float64) - 2
    require(np.all(dof > 0), "Nonpositive physical chi-squared DOF")
    reduced = data["chisq"].astype(np.float64) / dof
    score = data["snr"].astype(np.float64)
    penalized = reduced > 1
    score[penalized] *= (.5 * (1 + reduced[penalized] ** 3)) ** (-1 / 6)
    require(np.all(np.isfinite(score)), "Nonfinite reconstructed newSNR")
    return score, reduced


def ordered(keys, scores, index):
    return sorted(keys, key=lambda key: (-scores[index[key]], key))


def ranks(order):
    return {key: position + 1 for position, key in enumerate(order)}


def identity(key):
    # Decimal text preserves the signed 64-bit hash in JavaScript consumers.
    return {"detector": "H1", "template_hash": str(key[0]),
            "gps_sample_tick": key[1]}


def verify_source(repo, datasets, scores):
    sources = {}
    funcs = {}
    for label, (commit, _, _) in FROZEN.items():
        source = subprocess.check_output(
            ["git", "show", f"{commit}:pycbc/events/ranking.py"],
            cwd=repo, text=True)
        nodes = {n.name: n for n in ast.parse(source).body
                 if isinstance(n, ast.FunctionDef)}
        funcs[label] = nodes["newsnr"]
        sources[label] = {
            "commit": commit, "path": "pycbc/events/ranking.py",
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "newsnr_lines": [nodes["newsnr"].lineno, nodes["newsnr"].end_lineno],
            "get_newsnr_lines": [nodes["get_newsnr"].lineno,
                                 nodes["get_newsnr"].end_lineno],
        }
        if label == "original":
            namespace = {"numpy": np}
            module = ast.Module(body=[nodes["newsnr"], nodes["get_newsnr"]],
                                type_ignores=[])
            exec(compile(module, "frozen-original-ranking", "exec"), namespace)
    # Proposed adds the Torch dispatch before the same NumPy CPU body.
    original_body = [ast.dump(n) for n in funcs["original"].body[1:]]
    proposed_body = [ast.dump(n) for n in funcs["proposed"].body[3:]]
    require(original_body == proposed_body, "Frozen CPU newSNR bodies differ")
    for label, data in datasets.items():
        reduced = data["chisq"] / (2 * data["chisq_dof"] - 2)
        reference = namespace["newsnr"](data["snr"], reduced)
        require(np.array_equal(scores[label], reference),
                f"{label}: formula differs from frozen CPU newsnr")
        require(np.array_equal(scores[label].astype(np.float32),
                               namespace["get_newsnr"](data)),
                f"{label}: float32 output differs from frozen get_newsnr")
        scalar = np.array([float(s) if r <= 1 else
                           float(s) * math.pow(.5 * (1 + float(r) ** 3), -1 / 6)
                           for s, r in zip(data["snr"], reduced)])
        require(np.allclose(scores[label], scalar, rtol=2e-15, atol=0),
                f"{label}: independent scalar formula disagreement")
    return sources


def analyze(acquisition, repo):
    sys.dont_write_bytecode = True
    comparator_path = acquisition / "compare-triggers.py"
    spec = importlib.util.spec_from_file_location("frozen_compare", comparator_path)
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    loaded = {}
    for label, (commit, count, sha) in FROZEN.items():
        item = comparator.load(acquisition / f"runs/qual-{label}-cpu/triggers.hdf")
        require(not item["metadata"]["issues"], str(item["metadata"]["issues"]))
        require(item["sha256"] == sha, f"{label}: unexpected frozen HDF hash")
        require(item["metadata"]["source_snapshot"]["commit"] == commit,
                f"{label}: unexpected frozen source")
        require(set(item["detectors"]) == {"H1"}, "Expected H1 only")
        require(item["detectors"]["H1"]["count"] == count, "Unexpected count")
        loaded[label] = item
    left, right = [loaded[x]["metadata"] for x in FROZEN]
    for field in ("sample_rate", "analysis_options", "thresholds"):
        require(left[field] == right[field], f"Mismatched {field}")
    for option in ("--processing-scheme", "--fft-backends"):
        require(left["options"][option] == right["options"][option],
                f"Mismatched {option}")
    input_files = set()
    for option in comparator.FILE_OPTIONS:
        input_files.update(left["options"].get(option, []))
    for path in input_files:
        require(left["consumed_input_sha256"][path] ==
                right["consumed_input_sha256"][path], "Consumed input mismatch")
    groups = {label: value["detectors"]["H1"] for label, value in loaded.items()}
    for field in ("intervals", "gating"):
        require(groups["original"][field] == groups["proposed"][field],
                f"Mismatched {field}")
    data = {label: value["data"] for label, value in groups.items()}
    indices = {label: value["index"] for label, value in groups.items()}
    common = sorted(indices["original"].keys() & indices["proposed"].keys())
    union = sorted(indices["original"].keys() | indices["proposed"].keys())
    scores, reduced, full, shared, round32 = {}, {}, {}, {}, {}
    for label in FROZEN:
        scores[label], reduced[label] = newsnr(data[label])
        full[label] = ranks(ordered(indices[label], scores[label], indices[label]))
        shared[label] = ranks(ordered(common, scores[label], indices[label]))
        round32[label] = ranks(ordered(indices[label], scores[label].astype(np.float32),
                                      indices[label]))
    source_checks = verify_source(repo, data, scores)
    records = {}
    for key in union:
        record = identity(key)
        for label in FROZEN:
            if key not in indices[label]:
                record[label] = None
                continue
            i = indices[label][key]
            record[label] = {
                "row_index": i, "end_time": float(data[label]["end_time"][i]),
                "snr": float(data[label]["snr"][i]),
                "chisq": float(data[label]["chisq"][i]),
                "stored_bin_count": float(data[label]["chisq_dof"][i]),
                "reduced_chisq": float(reduced[label][i]),
                "newsnr": float(scores[label][i]),
                "full_output_rank": full[label][key],
                "matched_only_rank": shared[label].get(key),
            }
        if key in shared["original"]:
            a, b = record["original"], record["proposed"]
            record["newsnr_delta"] = b["newsnr"] - a["newsnr"]
            record["newsnr_relative_delta_percent"] = 100 * record["newsnr_delta"] / a["newsnr"]
            record["full_rank_improvement"] = a["full_output_rank"] - b["full_output_rank"]
            record["matched_rank_improvement"] = a["matched_only_rank"] - b["matched_only_rank"]
        records[key] = record
    x, y = [np.array([records[k][label]["newsnr"] for k in common]) for label in FROZEN]
    delta = y - x
    matched_rank_delta = np.array([records[k]["matched_rank_improvement"] for k in common])
    full_rank_delta = np.array([records[k]["full_rank_improvement"] for k in common])
    top = {label: {k for k, r in full[label].items() if r <= 100} for label in FROZEN}
    thresholds = []
    for threshold in sorted({left["thresholds"]["newsnr"], 5.1, 5.25, 5.5, 5.75, 6., 6.25, 6.5}):
        a, b = x >= threshold, y >= threshold
        entry = {"threshold": threshold,
                 "is_actual_newsnr_cut": threshold == left["thresholds"]["newsnr"],
                 "matched_both_retained": int(np.sum(a & b)),
                 "matched_both_below": int(np.sum(~a & ~b)),
                 "matched_upcrossings": int(np.sum(~a & b)),
                 "matched_downcrossings": int(np.sum(a & ~b)),
                 "original_matched_retained": int(a.sum()),
                 "proposed_matched_retained": int(b.sum()),
                 "upcrossing_identities": [identity(k) for k, keep in zip(common, ~a & b) if keep],
                 "downcrossing_identities": [identity(k) for k, keep in zip(common, a & ~b) if keep]}
        for label in FROZEN:
            entry[label + "_saved_retained"] = int(np.sum(scores[label] >= threshold))
            entry[label + "_only_retained"] = sum(
                records[k][label]["newsnr"] >= threshold
                for k in indices[label] if k not in shared[label])
        require(sum(entry[k] for k in ("matched_both_retained", "matched_both_below",
                                      "matched_upcrossings", "matched_downcrossings")) == len(common),
                "Retention accounting failed")
        thresholds.append(entry)
    input_provenance = {}
    precision = {}
    for label, item in loaded.items():
        input_provenance[label] = {k: item[k] for k in ("path", "sha256", "receipt_sha256", "metadata")}
        unique, counts = np.unique(scores[label], return_counts=True)
        precision[label] = {
            "exact_tied_score_groups": int(np.sum(counts > 1)),
            "exact_tied_events": int(np.sum(counts[counts > 1])),
            "float32_rank_changes": sum(full[label][k] != round32[label][k] for k in full[label]),
            "float32_top100_membership_changes": len(top[label] ^ {k for k, r in round32[label].items() if r <= 100}),
            "float32_threshold_membership_changes": {
                str(t["threshold"]): int(np.sum((scores[label] >= t["threshold"]) !=
                                               (scores[label].astype(np.float32).astype(float) >= t["threshold"])))
                for t in thresholds},
            "max_float32_score_rounding": float(np.max(np.abs(scores[label] - scores[label].astype(np.float32))))}
    result = {
        "scope": "Two preselected, clustered H1 CPU qualification outputs only; no injection population, background population, FAR or selection-efficiency validation.",
        "method": {"identity": "detector, exact signed template hash, round(end_time * sample_rate); no fuzzy matching",
                   "template_hash_json_encoding": "decimal string to preserve int64",
                   "formula": "r = chisq/(2*stored_chisq_dof-2); newSNR = snr for r<=1, else snr*[0.5*(1+r^3)]^(-1/6)",
                   "arithmetic": "float64 from stored values; float32 getter sensitivity also checked",
                   "rank": "1 is highest; ties resolved by signed template hash then GPS tick",
                   "rank_improvement": "original rank minus proposed rank; positive means moved up",
                   "retention": "post-hoc newSNR >= threshold within saved outputs; unavailable rows have no inferred score"},
        "provenance": input_provenance,
        "comparator": {"path": str(comparator_path), "sha256": digest(comparator_path)},
        "source_verification": source_checks,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "h5py": h5py.__version__},
        "sample_rate": left["sample_rate"], "valid_intervals": groups["original"]["intervals"],
        "actual_thresholds": left["thresholds"],
        "membership": {"original": len(indices["original"]), "proposed": len(indices["proposed"]),
                       "matched": len(common), "union": len(union),
                       "original_only": len(indices["original"]) - len(common),
                       "proposed_only": len(indices["proposed"]) - len(common),
                       "original_retained_identity_fraction": len(common) / len(indices["original"]),
                       "union_jaccard": len(common) / len(union)},
        "matched_scores": {"original": summary(x), "proposed": summary(y),
                           "delta": summary(delta), "absolute_delta": summary(abs(delta)),
                           "relative_delta_percent": summary(100 * delta / x),
                           "absolute_relative_delta_percent": summary(abs(100 * delta / x)),
                           "increased": int(np.sum(delta > 0)), "decreased": int(np.sum(delta < 0)),
                           "unchanged": int(np.sum(delta == 0))},
        "matched_ranking": {"spearman": float(spearmanr(x, y).statistic),
                            "kendall_tau_b": float(kendalltau(x, y).statistic),
                            "matched_only_absolute_shift": summary(abs(matched_rank_delta)),
                            "full_output_absolute_shift": summary(abs(full_rank_delta)),
                            "matched_only_changed_count": int(np.sum(matched_rank_delta != 0)),
                            "full_output_changed_count": int(np.sum(full_rank_delta != 0))},
        "top100": {"intersection": len(top["original"] & top["proposed"]),
                   "original_cutoff": min(records[k]["original"]["newsnr"] for k in top["original"]),
                   "proposed_cutoff": min(records[k]["proposed"]["newsnr"] for k in top["proposed"]),
                   "exits": [records[k] for k in sorted(top["original"] - top["proposed"], key=lambda k: full["original"][k])],
                   "entrants": [records[k] for k in sorted(top["proposed"] - top["original"], key=lambda k: full["proposed"][k])]},
        "threshold_transitions": thresholds, "precision_sensitivity": precision,
        "largest_absolute_score_changes": sorted([records[k] for k in common], key=lambda r: -abs(r["newsnr_delta"]))[:10],
        "largest_matched_rank_changes": sorted([records[k] for k in common], key=lambda r: -abs(r["matched_rank_improvement"]))[:10],
        "original_only": [records[k] for k in union if records[k]["proposed"] is None],
        "proposed_only": [records[k] for k in union if records[k]["original"] is None],
        "events": list(records.values()),
    }
    # Detect concurrent input changes before writing results.
    for item in loaded.values():
        require(digest(item["path"]) == item["sha256"], "HDF changed during analysis")
        require(digest(Path(item["path"]).with_name("receipt.json")) == item["receipt_sha256"],
                "Receipt changed during analysis")
    return result


def render(r):
    m, s, ranking, top = [r[k] for k in ("membership", "matched_scores", "matched_ranking", "top100")]
    lines = ["# Frozen CPU trigger ranking consequences", "",
             f"The frozen original/proposed outputs contain {m['original']:,}/{m['proposed']:,} triggers: "
             f"{m['matched']:,} exact matches, {m['original_only']} original-only and {m['proposed_only']} proposed-only. "
             f"Their top 100 overlap in {top['intersection']} identities.", "",
             "## Scope and method", "",
             "Read-only analysis of `qual-original-cpu/triggers.hdf` and `qual-proposed-cpu/triggers.hdf` under "
             "`artifacts/torch-baseline-final-20260908/acquisition`. These are the existing thresholded and clustered "
             "H1 qualification outputs. They are not the newly split CPU branch's output. "
             "The original/proposed source commits and HDF/receipt hashes are recorded in the JSON.", "",
             f"Both receipts specify CPU:1, MKL, {r['sample_rate']} Hz, SNR >= {r['actual_thresholds']['snr']:g}, "
             f"newSNR >= {r['actual_thresholds']['newsnr']:g}, and valid GPS interval [1187007160, 1187009064). "
             "Analysis options, consumed frame/bank hashes, gating and valid intervals agree. "
             "Each HDF matches its completed receipt and remains unchanged after analysis.", "",
             "Match identity is `(H1, signed template_hash, GPS sample tick)` with unique keys and validated sample-grid times; "
             "there is no nearest-time substitution. JSON stores template hashes as decimal strings to preserve all 64 bits.", "",
             "Stored `chisq_dof` is the bin count p=16, so physical DOF is 2p-2=30. "
             "Reconstruct r=chisq/30 and newSNR=snr when r<=1, otherwise snr*[0.5*(1+r^3)]^(-1/6). "
             "Arithmetic is float64 from the stored values. The NumPy CPU formula is unchanged between frozen sources; "
             "results match extracted original `newsnr` exactly, the float32 `get_newsnr` exactly after casting, "
             "and an independent scalar calculation within 2e-15 relative error. "
             "The HDF DOF conversion and >= selection are confirmed in frozen `pycbc/events/eventmgr.py` "
             "(original lines 296–305 and 549–552).", "",
             "Ranks descend by reconstructed newSNR; ties use signed template hash then GPS tick. "
             "Both full-output ranks and ranks restricted to the 1,959 common identities are reported. "
             "A positive rank improvement is original rank minus proposed rank.", "",
             "## Matched-event changes", "",
             "| Metric | Result |", "| --- | --- |",
             f"| Score increases / decreases / exactly unchanged | {s['increased']} / {s['decreased']} / {s['unchanged']} |",
             f"| Signed newSNR delta: minimum / median / maximum | {s['delta']['min']:.8g} / {s['delta']['median']:.8g} / {s['delta']['max']:.8g} |",
             f"| Absolute newSNR delta: median / 95th percentile / maximum | {s['absolute_delta']['median']:.8g} / {s['absolute_delta']['p95']:.8g} / {s['absolute_delta']['max']:.8g} |",
             f"| Relative newSNR delta: minimum / maximum | {s['relative_delta_percent']['min']:.6g}% / {s['relative_delta_percent']['max']:.6g}% |",
             f"| Absolute relative delta: median / 95th percentile | {s['absolute_relative_delta_percent']['median']:.6g}% / {s['absolute_relative_delta_percent']['p95']:.6g}% |",
             f"| Spearman / Kendall tau-b on common-event scores | {ranking['spearman']:.8f} / {ranking['kendall_tau_b']:.8f} |",
             f"| Common-only absolute rank shift: median / 95th percentile / maximum | {ranking['matched_only_absolute_shift']['median']:g} / {ranking['matched_only_absolute_shift']['p95']:g} / {ranking['matched_only_absolute_shift']['max']:g} |",
             f"| Full-output absolute rank shift: median / 95th percentile / maximum | {ranking['full_output_absolute_shift']['median']:g} / {ranking['full_output_absolute_shift']['p95']:g} / {ranking['full_output_absolute_shift']['max']:g} |",
             f"| Common-event ranks changed: common-only / full-output | {ranking['matched_only_changed_count']} / {ranking['full_output_changed_count']} |", "",
             "## Top 100", "",
             f"{top['intersection']} identities remain in both top-100 sets, with {len(top['exits'])} exits and "
             f"{len(top['entrants'])} entrants. Cutoff newSNR changes from {top['original_cutoff']:.8f} "
             f"to {top['proposed_cutoff']:.8f}. All these entrants/exits are present in both saved outputs; "
             "they reflect score ordering within the common events.", "",
             "| Change | Template hash | GPS time | Original rank → proposed rank | Original newSNR → proposed newSNR |",
             "| --- | --- | --- | --- | --- |"]
    for change in ("exits", "entrants"):
        for event in top[change]:
            a, b = event["original"], event["proposed"]
            require(a is not None and b is not None, "Unexpected unmatched top-100 change")
            lines.append(f"| {change[:-1]} | {event['template_hash']} | {a['end_time']:.9f} | "
                         f"{a['full_output_rank']} → {b['full_output_rank']} | {a['newsnr']:.6f} → {b['newsnr']:.6f} |")
    lines += ["", "## Retention and membership", "",
              f"The observed saved-output union has {m['union']:,} identities: {m['matched']:,} retained in both, "
              f"{m['original_only']} retained only by original, and {m['proposed_only']} only by proposed. "
              f"That is 61 changed memberships and a net +3, with {100*m['original_retained_identity_fraction']:.4f}% "
              "of original identities also present in proposed. These are output-set transitions, not inferred "
              "crossings of a particular cut. The absent counterpart's SNR/newSNR is unavailable.", "",
              "The table below applies post-hoc newSNR cuts to already saved triggers. "
              "Only 5.0 is the actual recorded newSNR cut; the remaining levels are descriptive probes. "
              "Up/down counts use only exact common identities. Saved counts additionally include that output's unmatched events.", "",
              "| newSNR cut | Common retained: original / proposed | Common up / down | Saved retained: original / proposed |",
              "| --- | --- | --- | --- |"]
    for t in r["threshold_transitions"]:
        lines.append(f"| {t['threshold']:g}{' (actual)' if t['is_actual_newsnr_cut'] else ''} | "
                     f"{t['original_matched_retained']} / {t['proposed_matched_retained']} | "
                     f"{t['matched_upcrossings']} / {t['matched_downcrossings']} | "
                     f"{t['original_saved_retained']} / {t['proposed_saved_retained']} |")
    for label in FROZEN:
        values = [e[label]["newsnr"] for e in r[label + "_only"]]
        lines += ["", f"{label.capitalize()}-only reconstructed newSNR spans {min(values):.8f}–{max(values):.8f}. "
                  "All unmatched identities and their available scores are in the JSON; missing counterparts remain null."]
    lines += ["", "## Numerical checks and limits", ""]
    for label, p in r["precision_sensitivity"].items():
        lines.append(f"- {label.capitalize()}: {p['exact_tied_score_groups']} exact score-tie groups; "
                     f"casting scores to float32 changes {p['float32_rank_changes']} full ranks, "
                     f"{p['float32_top100_membership_changes']} top-100 memberships, and "
                     f"{sum(p['float32_threshold_membership_changes'].values())} decisions across the listed cuts. "
                     f"Maximum score-rounding difference is {p['max_float32_score_rounding']:.3g}.")
    lines += ["", "The HDF contains rounded output magnitudes, not every internal complex SNR or rejected candidate. "
              "Reconstructed scores are sufficient for this saved-output comparison but do not replay the complete "
              "threshold/clustering process. The common-event analysis conditions on surviving both pipelines. "
              "It cannot measure selection efficiency or attribute unmatched events to SNR, newSNR, or clustering. "
              "A high overall rank correlation does not establish unchanged tail membership.", "",
              "No injection population, background population, coincidence statistic, FAR, or astrophysical sensitivity "
              "has been validated. This analysis does not establish ranking equivalence of the proposed CPU corrections "
              "or isolate the effects of individual corrections.", "",
              "Reproduce from the repository with the existing offline environment:", "",
              "```sh",
              "/private/tmp/pycbc-cpu-precision-env-20260908/bin/python -B artifacts/torch-precision-validation-20260908/trigger-ranking-consequences.py",
              "```", "",
              "[Analysis script](trigger-ranking-consequences.py) · [Full JSON, event records and provenance](trigger-ranking-consequences.json)", ""]
    return "\n".join(lines)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path,
                        default=here.parent / "torch-baseline-final-20260908/acquisition")
    parser.add_argument("--repo", type=Path, default=here.parents[1])
    parser.add_argument("--output-dir", type=Path, default=here)
    args = parser.parse_args()
    result = analyze(args.acquisition.resolve(), args.repo.resolve())
    result["analysis_script_sha256"] = digest(__file__)
    report = render(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "trigger-ranking-consequences"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    stem.with_suffix(".md").write_text(report)
    print(json.dumps({k: result[k] for k in ("membership", "matched_scores", "matched_ranking", "precision_sensitivity")}, indent=2))


if __name__ == "__main__":
    main()
