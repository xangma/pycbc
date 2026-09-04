#!/usr/bin/env python3
"""Compare two artifacts produced by generate.py against a named policy."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import sys

import numpy as np


STRUCTURE_KEYS = (
    "shape",
    "dtype",
    "delta_t",
    "delta_f",
    "epoch",
    "kind",
    "corrupted_samples",
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).with_name("policy.json"),
    )
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _paths(stem):
    if stem.suffix in (".json", ".npz"):
        stem = stem.with_suffix("")
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def _settings_for(name, profile):
    settings = dict(profile["defaults"])
    for pattern, overrides in profile.get("records", {}).items():
        if fnmatch.fnmatchcase(name, pattern):
            settings.update(overrides)
    return settings


def _relative_l2(reference, candidate, finite):
    ref = reference[finite].astype(np.complex128, copy=False)
    cand = candidate[finite].astype(np.complex128, copy=False)
    # Scale before subtraction and squaring so very small strain amplitudes
    # cannot underflow into a false zero error (nor large values overflow).
    magnitude = max(float(np.max(np.abs(ref))), float(np.max(np.abs(cand))))
    if magnitude == 0.0:
        return 0.0
    error = float(np.linalg.norm(cand / magnitude - ref / magnitude))
    scale = float(np.linalg.norm(ref / magnitude))
    if scale == 0.0:
        return 0.0 if error == 0.0 else float("inf")
    return error / scale


def _compare_record(name, reference, candidate, ref_meta, cand_meta, settings):
    failures = []
    for key in STRUCTURE_KEYS:
        if ref_meta.get(key) != cand_meta.get(key):
            failures.append(
                f"metadata {key}: {cand_meta.get(key)!r} != "
                f"{ref_meta.get(key)!r}"
            )

    if reference.shape != candidate.shape:
        return {
            "passed": False,
            "failures": failures + [
                f"array shape: {candidate.shape!r} != {reference.shape!r}"
            ],
        }

    ref_finite = np.isfinite(reference)
    cand_finite = np.isfinite(candidate)
    if not reference.size:
        failures.append("empty record cannot establish parity")
    if not ref_finite.all() or not cand_finite.all():
        failures.append(
            "parity corpus requires finite reference and candidate values"
        )
    finite = ref_finite & cand_finite

    if settings.get("zero_pattern", False):
        if not np.array_equal(reference == 0, candidate == 0):
            failures.append("exact zero pattern differs")

    if finite.any():
        difference = np.abs(candidate[finite] - reference[finite])
        max_abs = float(np.max(difference))
        nonzero = np.abs(reference[finite]) > 0
        if nonzero.any():
            max_rel = float(
                np.max(
                    difference[nonzero]
                    / np.abs(reference[finite][nonzero])
                )
            )
        else:
            max_rel = 0.0 if max_abs == 0.0 else float("inf")
        relative_l2 = _relative_l2(reference, candidate, finite)
    else:
        max_abs = max_rel = relative_l2 = 0.0

    l2_limit = float(settings["relative_l2"])
    if not np.isfinite(relative_l2) or relative_l2 > l2_limit:
        failures.append(
            f"relative L2 {relative_l2:.6e} exceeds {l2_limit:.6e}"
        )

    if settings.get("allclose", False):
        rtol = float(settings["rtol"])
        atol = float(settings["atol"])
        if not np.allclose(
            candidate,
            reference,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        ):
            failures.append(f"allclose failed (rtol={rtol:g}, atol={atol:g})")

    return {
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "relative_l2": relative_l2,
            "max_absolute_error": max_abs,
            "max_relative_error": max_rel,
        },
        "policy": settings,
    }


def main():
    args = _parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    try:
        profile = policy["profiles"][args.profile]
    except KeyError as exc:
        raise ValueError(
            f"unknown comparison profile {args.profile!r}"
        ) from exc

    ref_json_path, ref_npz_path = _paths(args.reference)
    cand_json_path, cand_npz_path = _paths(args.candidate)
    ref_manifest = json.loads(ref_json_path.read_text(encoding="utf-8"))
    cand_manifest = json.loads(cand_json_path.read_text(encoding="utf-8"))

    ref_records = ref_manifest["records"]
    cand_records = cand_manifest["records"]
    ref_names = set(ref_records)
    cand_names = set(cand_records)
    global_failures = []
    if not ref_names or not cand_names:
        global_failures.append(
            "parity corpus must contain at least one record"
        )
    if ref_names != cand_names:
        missing = sorted(ref_names - cand_names)
        extra = sorted(cand_names - ref_names)
        global_failures.append(
            f"record sets differ; missing={missing}, extra={extra}"
        )

    results = {}
    with np.load(ref_npz_path, allow_pickle=False) as ref_arrays, np.load(
        cand_npz_path, allow_pickle=False
    ) as cand_arrays:
        for name in sorted(ref_names & cand_names):
            settings = _settings_for(name, profile)
            results[name] = _compare_record(
                name,
                ref_arrays[name],
                cand_arrays[name],
                ref_records[name],
                cand_records[name],
                settings,
            )

    passed = not global_failures and all(
        result["passed"] for result in results.values()
    )
    report = {
        "schema_version": 1,
        "passed": passed,
        "profile": args.profile,
        "reference": {
            "label": ref_manifest["label"],
            "revision": ref_manifest["runtime"]["source_revision"],
        },
        "candidate": {
            "label": cand_manifest["label"],
            "revision": cand_manifest["runtime"]["source_revision"],
        },
        "global_failures": global_failures,
        "records": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"{ref_manifest['label']} -> {cand_manifest['label']} "
        f"[{args.profile}]"
    )
    for failure in global_failures:
        print(f"FAIL matrix: {failure}")
    for name, result in results.items():
        metrics = result.get("metrics", {})
        state = "PASS" if result["passed"] else "FAIL"
        print(
            f"{state} {name}: rel_l2="
            f"{metrics.get('relative_l2', float('nan')):.6e}, max_abs="
            f"{metrics.get('max_absolute_error', float('nan')):.6e}"
        )
        for failure in result["failures"]:
            print(f"  {failure}")
    print(f"overall: {'PASS' if passed else 'FAIL'}; report={args.report}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
