#!/usr/bin/env python3
"""Compare same-configuration pycbc_inspiral trigger HDF files across schemes.

Usage: python compare-triggers.py BASELINE.hdf CANDIDATE.hdf [CANDIDATE.hdf ...]
Each input requires its sibling receipt.json from run-case.py. Analysis options,
input content hashes, sample rate and valid intervals must agree. Only processing
scheme, FFT backend, verbosity and output destination may differ. Segment-length
changes are deliberately ineligible: they can change PSDs and trigger selection.

Identity is (detector, signed integer template_hash, GPS sample tick), independent
of row order. No one-sample timing shift is accepted. Default float32 numerical
budgets are |a-b| <= 1e-5 + 1e-4*max(|a|,|b|), except sigmasq (relative 1e-5,
zero absolute floor), exact DOF and circular phase (1e-4 radians). These are
explicit comparison budgets, not a physical waveform-accuracy guarantee.

JSON goes to stdout. Exit 0=pass, 1=fail, 2=review (only threshold-adjacent missing
triggers or no matched triggers). Near-threshold classification never passes an
unmatched trigger; clustering can also change trigger membership. --self-test
runs small in-memory HDF fixtures without creating files. Requires numpy/h5py.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import h5py
import numpy as np


REQUIRED = {"template_hash", "end_time", "snr", "chisq", "chisq_dof", "sigmasq"}
ROUTING = {"--processing-scheme", "--fft-backends", "--output", "--verbose"}
FILE_OPTIONS = {"--bank-file", "--frame-files", "--frame-cache", "--psd-file",
                "--asd-file", "--injection-file", "--gating-file"}
DEFAULTS = {"rtol": 1e-4, "atol": 1e-5, "sigmasq_rtol": 1e-5,
            "phase_atol": 1e-4, "max_examples": 12}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def options(argv):
    result = {}
    key = None
    if not isinstance(argv, list) or not argv:
        raise ValueError("Receipt lacks executable_cli")
    for arg in argv[1:]:
        if arg.startswith("--"):
            key, separator, value = arg.partition("=")
            if key in result:
                raise ValueError(f"Repeated CLI option {key}")
            result[key] = [value] if separator else []
        elif key is None:
            raise ValueError("Unexpected positional argument in executable_cli")
        else:
            result[key].append(arg)
    return result


def number(opts, name, default=None):
    values = opts.get(name)
    if values is None:
        return default
    if len(values) != 1 or not math.isfinite(float(values[0])):
        raise ValueError(f"Expected one finite value for {name}")
    return float(values[0])


def receipt_metadata(receipt):
    opts = options(receipt.get("executable_cli"))
    fs = number(opts, "--sample-rate")
    if fs is None or fs <= 0 or fs != int(fs):
        raise ValueError("Missing or invalid receipt sample rate")
    issues = []
    if receipt.get("state") != "complete" or receipt.get("returncode") != 0:
        issues.append("run receipt is not successfully complete")
    source = receipt.get("source_info", {})
    if not source.get("commit"):
        issues.append("receipt lacks source revision")
    if receipt.get("source_status_after") != source.get("status"):
        issues.append("source status changed or lacks post-run check")
    before = receipt.get("input_sha256", {})
    after = receipt.get("input_sha256_after", {})
    consumed = {receipt["executable_cli"][0]}
    for name in FILE_OPTIONS:
        consumed.update(opts.get(name, []))
    hashes = {}
    for name in sorted(consumed):
        value = before.get(name)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            issues.append(f"missing input hash: {name}")
        elif after.get(name) != value:
            issues.append(f"input changed or lacks post-run hash: {name}")
        hashes[name] = value
    return {
        "sample_rate": int(fs), "options": opts,
        "source_snapshot": source,
        "analysis_options": {k: v for k, v in opts.items() if k not in ROUTING},
        "consumed_input_sha256": hashes, "issues": issues,
        "thresholds": {"snr": number(opts, "--snr-threshold"),
                       "newsnr": number(opts, "--newsnr-threshold"),
                       "chisq": number(opts, "--chisq-threshold", 0),
                       "chisq_delta": number(opts, "--chisq-delta", 0)},
    }


def read_hdf(stream, sample_rate):
    detectors = {}
    for detector, group in stream.items():
        if not isinstance(group, h5py.Group) or not re.fullmatch(r"[A-Z][0-9]", detector):
            continue
        start = np.asarray(group["search/start_time"][:], dtype=np.float64)
        end = np.asarray(group["search/end_time"][:], dtype=np.float64)
        if (start.ndim != 1 or not len(start) or start.shape != end.shape
                or not np.all(np.isfinite(start)) or not np.all(np.isfinite(end))
                or np.any(end <= start) or np.any(start[1:] < end[:-1])):
            raise ValueError(f"{detector}: invalid valid-analysis intervals")
        data = {name: obj[:] for name, obj in group.items()
                if isinstance(obj, h5py.Dataset)}
        # EventManager omits every trigger dataset when zero events survive.
        if data and not REQUIRED.issubset(data):
            raise ValueError(f"{detector}: missing columns {sorted(REQUIRED - data.keys())}")
        if not data:
            data = {key: np.array([], dtype=np.int64 if key == "template_hash" else float)
                    for key in REQUIRED}
        count = len(data["snr"])
        for name, array in data.items():
            if array.shape != (count,) or array.dtype.kind not in "iuf":
                raise ValueError(f"{detector}/{name}: invalid event array shape or dtype")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{detector}/{name}: nonfinite value")
            if name.endswith("_dof") and (np.any(array != np.rint(array)) or np.any(array < 0)):
                raise ValueError(f"{detector}/{name}: invalid DOF")
        if data["template_hash"].dtype.kind not in "iu":
            raise ValueError(f"{detector}: template_hash must be an integer array")
        if any(np.any(data[k] < 0) for k in ("snr", "chisq")) or np.any(data["sigmasq"] <= 0):
            raise ValueError(f"{detector}: invalid SNR/chisq/sigmasq")
        times = np.asarray(data["end_time"], dtype=np.float64)
        scaled = times * sample_rate
        if np.any(np.abs(scaled) >= 2 ** 53):
            raise ValueError("GPS tick conversion exceeds exact float64 integer range")
        ticks = np.rint(scaled).astype(np.int64)
        if np.any(np.abs(scaled - ticks) > 0.01):
            raise ValueError(f"{detector}: GPS time is off the sample grid (>0.01 sample)")
        in_interval = np.zeros(count, dtype=bool)
        for lo, hi in zip(start, end):
            in_interval |= (times >= lo) & (times < hi)
        if not np.all(in_interval):
            raise ValueError(f"{detector}: event outside valid analysis interval")
        keys = [(int(h), int(t)) for h, t in zip(data["template_hash"], ticks)]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{detector}: duplicate template_hash/GPS-sample identity")
        gating = {}
        if "gating" in group:
            def collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    values = obj[()]
                    if not np.all(np.isfinite(values)):
                        raise ValueError("Nonfinite gating metadata")
                    gating[name] = values.tolist()
            group["gating"].visititems(collect)
        detectors[detector] = {"data": data, "index": dict(zip(keys, range(count))),
                               "intervals": list(map(list, zip(start.tolist(), end.tolist()))),
                               "gating": gating, "count": count}
    if not detectors:
        raise ValueError("No detector groups found")
    return detectors


def load(path):
    path = Path(path).resolve()
    receipt_path = path.with_name("receipt.json")
    receipt = json.loads(receipt_path.read_text())
    metadata = receipt_metadata(receipt)
    sha = digest(path)
    if receipt.get("trigger_sha256") != sha:
        metadata["issues"].append("trigger HDF hash does not match receipt")
    with h5py.File(path, "r") as stream:
        detectors = read_hdf(stream, metadata["sample_rate"])
    channel = metadata["options"].get("--channel-name", [])
    if len(channel) != 1 or set(detectors) != {channel[0][:2]}:
        metadata["issues"].append("detector groups disagree with receipt channel")
    # Check that HDF intervals agree with CLI, not merely with the other HDF.
    opts = metadata["options"]
    lo = number(opts, "--trig-start-time")
    hi = number(opts, "--trig-end-time")
    if lo is None:
        lo = number(opts, "--gps-start-time") + number(opts, "--segment-start-pad", 0)
    if hi is None:
        hi = number(opts, "--gps-end-time") - number(opts, "--segment-end-pad", 0)
    for detector in detectors.values():
        if detector["intervals"] != [[lo, hi]]:
            metadata["issues"].append("HDF valid interval disagrees with receipt")
    return {"path": str(path), "sha256": sha, "receipt_sha256": digest(receipt_path),
            "metadata": metadata, "detectors": detectors}


def threshold_proximity(data, index, thresholds, tol):
    snr = float(data["snr"][index])
    chisq = float(data["chisq"][index])
    physical_dof = 2 * float(data["chisq_dof"][index]) - 2
    stats = {"snr": snr}
    if physical_dof > 0:
        reduced = chisq / physical_dof
        stats["newsnr"] = snr if reduced <= 1 else snr * (0.5 * (1 + reduced ** 3)) ** (-1 / 6)
        denominator = physical_dof + thresholds["chisq_delta"] * snr ** 2
        if denominator > 0:
            stats["chisq"] = chisq / denominator
    near = []
    for name, value in stats.items():
        boundary = thresholds.get(name)
        if boundary is not None and boundary > 0:
            band = tol["atol"] + tol["rtol"] * abs(boundary)
            if abs(value - boundary) <= band:
                near.append({"statistic": name, "value": value, "threshold": boundary,
                             "absolute_distance": abs(value - boundary), "band": band})
    return near


def compare(base, candidate, tol):
    failures = []
    reviews = []
    for label, item in (("baseline", base), ("candidate", candidate)):
        failures.extend(f"{label}: {issue}" for issue in item["metadata"]["issues"])
    for key in ("sample_rate", "analysis_options", "consumed_input_sha256", "source_snapshot"):
        if base["metadata"][key] != candidate["metadata"][key]:
            failures.append(f"configuration mismatch: {key}")
    if set(base["detectors"]) != set(candidate["detectors"]):
        failures.append("detector identities differ")
    results = {}
    for name in sorted(base["detectors"].keys() & candidate["detectors"].keys()):
        left, right = base["detectors"][name], candidate["detectors"][name]
        for key in ("intervals", "gating"):
            if left[key] != right[key]:
                failures.append(f"{name}: {key} differ")
        a, b = left["data"], right["data"]
        if left["count"] and right["count"] and set(a) != set(b):
            failures.append(f"{name}: event columns differ")
        common = sorted(left["index"].keys() & right["index"].keys())
        ia = [left["index"][key] for key in common]
        ib = [right["index"][key] for key in common]
        metrics = {}
        for field in sorted(a.keys() & b.keys() - {"template_hash", "end_time"}):
            if not common:
                continue
            x, y = a[field][ia].astype(float), b[field][ib].astype(float)
            difference = np.abs(x - y)
            if field == "coa_phase":
                difference = np.abs(np.angle(np.exp(1j * (x - y))))
                budget = np.full(len(x), tol["phase_atol"])
            elif field.endswith("_dof") or a[field].dtype.kind in "iu":
                budget = np.zeros(len(x))
            elif field == "sigmasq":
                budget = tol["sigmasq_rtol"] * np.maximum(np.abs(x), np.abs(y))
            else:
                budget = tol["atol"] + tol["rtol"] * np.maximum(np.abs(x), np.abs(y))
            bad = difference > budget
            worst = int(np.argmax(difference))
            metrics[field] = {"violations": int(bad.sum()), "max_absolute_error": float(difference[worst]),
                              "max_relative_error": float(np.max(np.divide(
                                  difference, np.maximum(np.abs(x), np.abs(y)),
                                  out=np.zeros_like(difference),
                                  where=np.maximum(np.abs(x), np.abs(y)) > 0))),
                              "worst_identity": list(common[worst]),
                              "baseline_value": float(x[worst]), "candidate_value": float(y[worst]),
                              "budget_at_worst": float(budget[worst])}
            if np.any(bad):
                failures.append(f"{name}/{field}: {int(bad.sum())} numerical violations")
        unmatched = {}
        for label, source, other in (("baseline_only", left, right), ("candidate_only", right, left)):
            keys = sorted(source["index"].keys() - other["index"].keys())
            counts = Counter()
            examples = []
            for identity in keys:
                index = source["index"][identity]
                near = threshold_proximity(source["data"], index, base["metadata"]["thresholds"], tol)
                classification = "near_threshold_requires_review" if near else "unexplained_mismatch"
                counts[classification] += 1
                if len(examples) < tol["max_examples"]:
                    neighbors = [key[1] - identity[1] for key in other["index"] if key[0] == identity[0]]
                    examples.append({"template_hash": identity[0], "gps_sample_tick": identity[1],
                                     "end_time": float(source["data"]["end_time"][index]),
                                     "snr": float(source["data"]["snr"][index]),
                                     "classification": classification, "near_thresholds": near,
                                     "nearest_same_template_delta_samples": min(neighbors, key=abs) if neighbors else None})
            if counts["unexplained_mismatch"]:
                failures.append(f"{name}/{label}: {counts['unexplained_mismatch']} unexplained triggers")
            if counts["near_threshold_requires_review"]:
                reviews.append(f"{name}/{label}: {counts['near_threshold_requires_review']} threshold-adjacent triggers")
            unmatched[label] = {"count": len(keys), "classifications": dict(counts), "examples": examples}
        if not common:
            reviews.append(f"{name}: no matched triggers; numerical parity is untested")
        results[name] = {"baseline_count": left["count"], "candidate_count": right["count"],
                         "matched_count": len(common), "valid_intervals": left["intervals"],
                         "metrics": metrics, "unmatched": unmatched}
    status = "fail" if failures else "review" if reviews else "pass"
    return {"status": status, "baseline": base["path"], "candidate": candidate["path"],
            "baseline_sha256": base["sha256"], "candidate_sha256": candidate["sha256"],
            "baseline_receipt_sha256": base.get("receipt_sha256"),
            "candidate_receipt_sha256": candidate.get("receipt_sha256"),
            "sample_rate": base["metadata"]["sample_rate"], "failures": failures,
            "review_reasons": reviews, "detectors": results}


def self_test():
    """Small fixtures exercise membership, numerics, timestamps and HDF reading."""
    import copy

    checks = []
    fs, epoch = 4096, 1187007160.0
    arrays = {"template_hash": np.array([7, 8], dtype=np.int64),
              "end_time": np.array([epoch + 1, epoch + 2]), "snr": np.array([8., 5.50001], dtype=np.float32),
              "chisq": np.array([30., 30.], dtype=np.float32), "chisq_dof": np.array([16., 16.]),
              "sigmasq": np.array([1e6, 2e6], dtype=np.float32)}
    metadata = {"sample_rate": fs, "analysis_options": {"--segment-length": ["256"]},
                "consumed_input_sha256": {}, "source_snapshot": {"commit": "fixture"}, "issues": [],
                "thresholds": {"snr": 5.5, "newsnr": 5., "chisq": 0., "chisq_delta": 0.}}

    def fixture(values):
        with h5py.File("fixture.hdf", "w", driver="core", backing_store=False) as stream:
            for key, value in values.items():
                stream[f"H1/{key}"] = value
            stream["H1/search/start_time"] = [epoch]
            stream["H1/search/end_time"] = [epoch + 10]
            detectors = read_hdf(stream, fs)
        return {"path": "fixture", "sha256": "fixture", "metadata": copy.deepcopy(metadata),
                "detectors": detectors}

    base = fixture(arrays)

    def check(label, values, expected):
        actual = compare(base, fixture(values), DEFAULTS)["status"]
        if actual != expected:
            raise AssertionError(f"{label}: expected {expected}, got {actual}")
        checks.append(label)

    values = {k: v[::-1].copy() for k, v in arrays.items()}
    values["snr"] *= 1 + 1e-6
    check("row permutation and small float32 error pass", values, "pass")
    values = copy.deepcopy(arrays)
    values["sigmasq"][0] *= 1.01
    check("normalization drift fails", values, "fail")
    values = copy.deepcopy(arrays)
    values["end_time"][0] += 1 / fs
    check("one-sample shift fails", values, "fail")
    values = copy.deepcopy(arrays)
    values["chisq_dof"][0] += 1
    check("DOF change fails exactly", values, "fail")
    values = copy.deepcopy(arrays)
    values["template_hash"][0] += 100
    check("changed template identity fails", values, "fail")
    check("threshold-adjacent omission needs review", {k: v[:1] for k, v in arrays.items()}, "review")
    check("loud omission fails", {k: v[1:] for k, v in arrays.items()}, "fail")
    other = fixture(arrays)
    other["metadata"]["analysis_options"]["--segment-length"] = ["512"]
    if compare(base, other, DEFAULTS)["status"] != "fail":
        raise AssertionError("Segment mismatch passed")
    checks.append("changed FFT duration fails configuration guard")
    for label, key, value in (
            ("valid interval mismatch fails", "intervals", [[epoch, epoch + 9]]),
            ("gating mismatch fails", "gating", {"auto/time": [epoch + 4]})):
        other = fixture(arrays)
        other["detectors"]["H1"][key] = value
        if compare(base, other, DEFAULTS)["status"] != "fail":
            raise AssertionError(label)
        checks.append(label)
    other = fixture(arrays)
    other["detectors"]["L1"] = other["detectors"].pop("H1")
    if compare(base, other, DEFAULTS)["status"] != "fail":
        raise AssertionError("Detector mismatch passed")
    checks.append("detector mismatch fails")
    boundary = {"snr": [8.0], "chisq_dof": [16.0],
                "chisq": [30 * (2 * (8 / 5) ** 6 - 1) ** (1 / 3)]}
    near = threshold_proximity(boundary, 0, metadata["thresholds"], DEFAULTS)
    if not any(item["statistic"] == "newsnr" for item in near):
        raise AssertionError("Stored bin count was not converted to physical DOF")
    checks.append("newSNR threshold uses 2p-2 physical DOF")
    for label, change in (
            ("nonfinite statistic rejected", lambda a: a["snr"].__setitem__(0, np.nan)),
            ("off-grid timestamp rejected", lambda a: a["end_time"].__setitem__(0, epoch + 0.3 / fs)),
            ("duplicate trigger identity rejected", lambda a: (a["end_time"].__setitem__(1, a["end_time"][0]),
                                                               a["template_hash"].__setitem__(1, a["template_hash"][0])))):
        values = copy.deepcopy(arrays)
        change(values)
        try:
            fixture(values)
        except ValueError:
            checks.append(label)
        else:
            raise AssertionError(label)
    return {"status": "pass", "fixture_checks": checks, "count": len(checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path, nargs="?")
    parser.add_argument("candidates", type=Path, nargs="*")
    parser.add_argument("--self-test", action="store_true")
    for key, value in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=int if key == "max_examples" else float, default=value)
    args = parser.parse_args()
    tol = {key: getattr(args, key) for key in DEFAULTS}
    if any(not math.isfinite(v) or v < 0 for v in tol.values()):
        parser.error("Tolerances and example limit must be finite and nonnegative")
    if args.self_test:
        report = self_test()
    else:
        if not args.baseline or not args.candidates:
            parser.error("Provide baseline and at least one candidate, or --self-test")
        comparisons = []
        base = load(args.baseline)
        for path in args.candidates:
            try:
                comparisons.append(compare(base, load(path), tol))
            except (OSError, ValueError, KeyError, TypeError) as error:
                comparisons.append({"candidate": str(path), "status": "fail", "error": str(error)})
        status = "fail" if any(r["status"] == "fail" for r in comparisons) else (
            "review" if any(r["status"] == "review" for r in comparisons) else "pass")
        report = {"schema_version": 1, "status": status, "tolerances": tol, "comparisons": comparisons,
                  "scope": "Same analysis configuration only; unmatched triggers never pass."}
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return {"pass": 0, "fail": 1, "review": 2}[report["status"]]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}, sort_keys=True))
        sys.exit(1)
