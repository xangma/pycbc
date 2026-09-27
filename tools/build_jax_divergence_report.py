#!/usr/bin/env python3
"""Build a browsable diagnostic report from a JAX benchmark suite.

This is a view of existing comparator verdicts, never a second science gate.
An archived suite can show aggregate receipts; a full run directory can also
show bounded HDF value examples. No benchmark is executed by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name("jax_divergence_viewer")
STAGES = (
    (
        "conditioning",
        "Conditioned strain",
        "bin/pycbc_inspiral",
        "bin/pycbc_live",
        "pycbc/strain/strain_jax.py",
    ),
    (
        "psd",
        "PSD",
        "pycbc/psd/estimate.py",
        "pycbc/psd/estimate.py",
        "pycbc/psd/estimate_jax.py",
    ),
    (
        "segment",
        "Segment spectrum",
        "bin/pycbc_inspiral",
        "bin/pycbc_live",
        "bin/pycbc_inspiral",
    ),
    (
        "filter",
        "Matched filter",
        "pycbc/filter/matchedfilter.py",
        "pycbc/filter/matchedfilter.py",
        "pycbc/filter/matchedfilter_jax.py",
    ),
    (
        "chisq",
        "Chi-square",
        "pycbc/vetoes/chisq.py",
        "pycbc/vetoes/chisq.py",
        "pycbc/vetoes/chisq_jax.py",
    ),
    (
        "selection",
        "Selection and output",
        "bin/pycbc_inspiral",
        "bin/pycbc_live",
        "bin/pycbc_live",
    ),
    ("gates", "Qualification gates", "", "", ""),
)

# These are places to begin inspecting each stage, not inferred causes of a
# discrepancy. Lines are resolved only against identity-verified source copies.
STAGE_ENTRY_ANCHORS = {
    "conditioning": {
        "inspiral": {
            "reference": 'stage_event("conditioning", "start")',
            "candidate": "def _fused_autogate_pipeline_core(",
        },
        "live": {
            "reference": "_stage_event('frame_read', 'start'",
            "candidate": "def _fused_autogate_pipeline_core(",
        },
    },
    "psd": {
        "inspiral": {
            "reference": "def welch(",
            "candidate": "def welch_jax(",
        },
        "live": {
            "reference": "def welch(",
            "candidate": "def welch_jax(",
        },
    },
    "segment": {
        "inspiral": {
            "reference": 'stage_event("segments_psd", "start")',
            "candidate": 'stage_event("segments_psd", "start")',
        },
    },
    "filter": {
        "inspiral": {
            "reference": "def full_matched_filter_and_cluster_symm(",
            "candidate": "def batched_matched_filter_and_cluster_jax(",
        },
        "live": {
            "reference": "def _process_batch(",
            "candidate": "def live_process_batch_jax(",
        },
    },
    "chisq": {
        "inspiral": {
            "reference": "def power_chisq_at_points_from_precomputed(",
            "candidate": "def batch_power_chisq_jax(",
        },
        "live": {
            "reference": "def power_chisq_at_points_from_precomputed(",
            "candidate": "def power_chisq_at_points_from_precomputed(",
        },
    },
    "selection": {
        "inspiral": {
            "reference": "def template_triggers(",
            "candidate": "def batch_template_triggers(",
        },
        "live": {
            "reference": "def dump(",
            "candidate": "def dump(",
        },
    },
}


class SuiteReader:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.read_hashes = {}
        if self.path.is_dir():
            self.archive = None
        else:
            self.archive = tarfile.open(self.path, "r:gz")

    def read(self, name):
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("suite member must be relative")
        if self.archive:
            try:
                member = self.archive.getmember(name)
            except KeyError:
                return None
            if not member.isfile() or member.size > 100_000_000:
                raise ValueError("invalid or oversized suite member: " + name)
            return json.load(self.archive.extractfile(member))
        path = self.path / name
        if not path.is_file():
            return None
        data = path.read_bytes()
        self.read_hashes[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    def source_sha256(self):
        if self.archive:
            return sha256_file(self.path)
        manifest = json.dumps(
            self.read_hashes, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(manifest.encode()).hexdigest()

    def close(self):
        if self.archive:
            self.archive.close()

    def bytes(self, name):
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("suite member must be relative")
        if self.archive:
            try:
                member = self.archive.getmember(name)
            except KeyError:
                return None
            if not member.isfile() or member.size > 50_000_000:
                raise ValueError("invalid or oversized suite member: " + name)
            return self.archive.extractfile(member).read()
        path = self.path / name
        return path.read_bytes() if path.is_file() else None


def stage_for(path):
    lower = path.lower()
    if "conditioned_strain" in lower:
        return "conditioning"
    if "/psd" in lower or lower.endswith("psd"):
        return "psd"
    if "segments/" in lower or "segment" in lower:
        return "segment"
    if "chisq" in lower:
        return "chisq"
    if any(token in lower for token in ("snr", "sigmasq", "coa_phase")):
        return "filter"
    return "selection"


def json_value(value):
    value = np.asarray(value)
    if value.ndim:
        return value.tolist()
    item = value.item()
    if isinstance(item, complex):
        return {"real": float(item.real), "imag": float(item.imag)}
    if isinstance(item, bytes):
        return item.decode(errors="replace")
    if isinstance(item, (float, np.floating)) and not np.isfinite(item):
        return str(item)
    return item


def point_examples(
    reference,
    candidate,
    path,
    matched=False,
    rate=2048,
    max_elements=5_000_000,
):
    """Return bounded *exact-value* examples, separate from the strict gate.

    Matching follows the comparator's template-hash/sample key. Ambiguous
    identities are omitted. Large datasets are deliberately left in raw HDF.
    """
    if (
        not reference
        or not candidate
        or not Path(reference).is_file()
        or not Path(candidate).is_file()
    ):
        return {"status": "raw HDF unavailable"}
    try:
        with h5py.File(reference) as left, h5py.File(candidate) as right:
            if path not in left or path not in right:
                return {"status": "dataset unavailable"}
            if (
                left[path].size > max_elements
                or right[path].size > max_elements
            ):
                return {
                    "status": "dataset exceeds detail limit",
                    "limit": max_elements,
                }
            a, b = np.asarray(left[path][()]), np.asarray(right[path][()])
            keys = None
            if matched:
                detector = path.split("/", 1)[0]
                hashes = f"{detector}/template_hash"
                times = f"{detector}/end_time"
                if (
                    hashes not in left
                    or hashes not in right
                    or times not in left
                    or times not in right
                ):
                    return {"status": "identity arrays unavailable"}
                ta, tb = np.asarray(left[times]), np.asarray(right[times])
                origin = int(
                    min(
                        float(ta.min()) if ta.size else 0,
                        float(tb.min()) if tb.size else 0,
                    )
                )

                def keyed(file, time):
                    groups = {}
                    for i, (hash_value, gps) in enumerate(
                        zip(np.asarray(file[hashes]), time)
                    ):
                        key = (
                            json_value(hash_value),
                            int(np.rint((float(gps) - origin) * rate)),
                        )
                        groups.setdefault(key, []).append(i)
                    return groups

                ag, bg = keyed(left, ta), keyed(right, tb)
                keys = sorted(
                    k
                    for k in ag.keys() & bg.keys()
                    if len(ag[k]) == len(bg[k]) == 1
                )
                a = a[[ag[k][0] for k in keys]]
                b = b[[bg[k][0] for k in keys]]
            if a.shape != b.shape:
                return {
                    "status": "shape mismatch",
                    "reference_shape": list(a.shape),
                    "candidate_shape": list(b.shape),
                }
            if a.size == 0:
                return {"status": "no matched values"}
            unequal = np.asarray(a != b).reshape(-1)
            indices = np.flatnonzero(unequal)
            if not len(indices):
                return {"status": "exact values agree"}
            first = int(indices[0])
            if a.dtype.kind in "biu" and b.dtype.kind in "biu":
                delta = np.asarray(
                    np.abs(b.astype(object) - a.astype(object)), dtype=float
                )
            elif a.dtype.kind in "fc" and b.dtype.kind in "fc":
                delta = np.abs(
                    b.astype(
                        np.complex128
                        if "c" in (a.dtype.kind, b.dtype.kind)
                        else np.float64
                    )
                    - a
                )
            else:
                delta = np.zeros(a.shape, dtype=float)
            magnitude = np.nan_to_num(
                delta.reshape(-1), nan=-1, posinf=-1, neginf=-1
            )
            worst = int(indices[np.argmax(magnitude[indices])])

            def point(flat):
                coord = [int(index) for index in np.unravel_index(flat, a.shape)]
                result = {
                    "index": coord,
                    "reference": json_value(a[tuple(coord)]),
                    "candidate": json_value(b[tuple(coord)]),
                    "absolute_difference": json_value(delta[tuple(coord)]),
                }
                if keys is not None:
                    hash_value, sample = keys[coord[0]]
                    result["identity"] = {
                        "template_hash": hash_value,
                        "sample": sample,
                        "gps": origin + sample / rate,
                    }
                elif (
                    path == "conditioned_strain"
                    and "strain_epoch" in left.attrs
                    and "strain_delta_t" in left.attrs
                ):
                    result["gps"] = float(left.attrs["strain_epoch"]) + coord[
                        0
                    ] * float(left.attrs["strain_delta_t"])
                elif path == "conditioned_strain" and "segments" in left:
                    for group in left["segments"].values():
                        if "strain" in group and all(
                            key in group.attrs for key in ("epoch", "delta_t")
                        ):
                            result["gps"] = float(
                                group.attrs["epoch"]
                            ) + coord[0] * float(group.attrs["delta_t"])
                            break
                elif "/" in path:
                    group = left[path.rsplit("/", 1)[0]]
                    if path.endswith("/strain") and "delta_t" in group.attrs:
                        result["gps"] = float(group.attrs["epoch"]) + coord[
                            0
                        ] * float(group.attrs["delta_t"])
                    elif "delta_f" in group.attrs:
                        result["frequency_hz"] = coord[0] * float(
                            group.attrs["delta_f"]
                        )
                return result

            return {
                "status": "available",
                "exact_different_elements": len(indices),
                "first": point(first),
                "worst_absolute": point(worst),
            }
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        return {"status": "detail extraction failed", "reason": str(exc)}


def field_rows(comparison, step_id, arm, scope, evidence_pair=None):
    rows = []
    for detector, result in comparison.get("detectors", {}).items():
        identity = {
            key: result.get(key)
            for key in (
                "reference_triggers",
                "candidate_triggers",
                "matched_triggers",
                "reference_unmatched",
                "candidate_unmatched",
                "identity_pass",
                "inactive_buffer_block",
                "reference_unmatched_identities",
                "candidate_unmatched_identities",
            )
            if key in result
        }
        rows.append(
            dict(
                step=step_id,
                arm=arm,
                scope=scope,
                detector=detector,
                stage="selection",
                path="trigger identities",
                kind="identity",
                verdict=(
                    "pass"
                    if result.get(
                        "identity_pass", result.get("inactive_buffer_block")
                    )
                    else "fail"
                ),
                metric=identity,
                reference_file=result.get("reference"),
                candidate_file=result.get("candidate"),
            )
        )
        for kind in ("fields", "attributes"):
            for path, metric in result.get(kind, {}).items():
                row = dict(
                    step=step_id,
                    arm=arm,
                    scope=scope,
                    detector=detector,
                    stage=stage_for(path),
                    path=path,
                    kind=kind[:-1],
                    verdict="pass" if metric.get("passed") is True else "fail",
                    metric=metric,
                    reference_file=result.get("reference"),
                    candidate_file=result.get("candidate"),
                )
                if kind == "fields" and row["verdict"] == "fail":
                    row["examples"] = point_examples(
                        row["reference_file"],
                        row["candidate_file"],
                        path,
                        matched=metric.get("matched_trigger_field", False),
                        rate=result.get("sample_rate", 2048),
                    )
                rows.append(row)
    for path, metric in comparison.get("evidence", {}).items():
        row = dict(
            step=step_id,
            arm=arm,
            scope=scope,
            detector="",
            path=path,
            stage=stage_for(path),
            kind="evidence",
            verdict="pass" if metric.get("passed") is True else "fail",
            metric=metric,
            reference_file=evidence_pair[0] if evidence_pair else None,
            candidate_file=evidence_pair[1] if evidence_pair else None,
        )
        if row["verdict"] == "fail" and evidence_pair:
            row["examples"] = point_examples(*evidence_pair, path)
        rows.append(row)
    for gate in comparison.get("missing_gates", []):
        rows.append(
            dict(
                step=step_id,
                arm=arm,
                scope=scope,
                detector="",
                path=gate,
                stage="gates",
                kind="missing gate",
                verdict="missing",
                metric={"reason": gate},
            )
        )
    return rows


def science_comparisons(science):
    """Yield comparison objects in both inspiral and live receipt shapes."""
    if "detectors" in science or "evidence" in science:
        yield "qualification", science
    for phase in ("qualification", "selected"):
        for index, run in enumerate(science.get(phase, [])):
            for block, comparison in enumerate(run.get("comparisons", [])):
                yield f"{phase} run {index + 1}, block {block + 1}", comparison


def timing_rows(receipt, step, step_id):
    rows = []
    qualification = receipt.get(
        "qualification", receipt.get("qualification_results", {})
    )
    for arm, value in qualification.items():
        runs = value if isinstance(value, list) else [value]
        for index, run in enumerate(runs):
            if not isinstance(run, dict):
                continue
            rows.append(
                dict(
                    step=step_id,
                    arm=arm,
                    scope="qualification diagnostic",
                    replicate=index + 1,
                    wall_seconds=run.get("elapsed_wall_sec"),
                    calc_seconds=run.get("calc_time_sec"),
                    stages=run.get("stage_timings", run.get("phases", {})),
                    publishable=False,
                )
            )
    qualified = (
        step.get("status") == "passed"
        and step.get("mode") == "timing"
        and bool(receipt.get("science"))
        and all(
            s.get("passed") is True
            for s in receipt.get("science", {}).values()
        )
    )
    for arm, runs in receipt.get("raw_results", {}).items():
        for index, run in enumerate(runs):
            rows.append(
                dict(
                    step=step_id,
                    arm=arm,
                    scope=(
                        "qualified full process"
                        if qualified
                        else "unqualified diagnostic"
                    ),
                    replicate=index + 1,
                    wall_seconds=run.get("elapsed_wall_sec"),
                    calc_seconds=run.get("calc_time_sec"),
                    stages=run.get("stage_timings", run.get("phases", {})),
                    publishable=qualified,
                )
            )
    return rows


def provisional_live_receipt(checkpoint, step):
    """Expose completed checkpoint evidence without treating it as a final run."""
    if not isinstance(checkpoint, dict):
        return None
    qualification = checkpoint.get("qualification_results")
    runs = checkpoint.get("raw_results")
    if not isinstance(qualification, dict) or not isinstance(runs, dict):
        return None
    for collection in (qualification, runs):
        if any(
            not isinstance(arm, str) or not isinstance(values, list)
            or any(not isinstance(value, dict) for value in values)
            for arm, values in collection.items()
        ):
            return None
    arms = sorted(set(qualification) | set(runs))
    if not any(qualification.get(arm) or runs.get(arm) for arm in arms):
        return None
    science = {}
    for arm in arms:
        qualifying = [
            run.get("science") if isinstance(run.get("science"), dict) else {}
            for run in qualification.get(arm, [])
        ]
        selected = (
            qualifying if step.get("mode") == "qualification" else
            [run.get("science") if isinstance(run.get("science"), dict) else {}
             for run in runs.get(arm, [])]
        )
        missing_gates = {"final campaign receipt unavailable"}
        for result in qualifying + selected:
            missing_gates.update(result.get("missing_gates", []))
        science[arm] = {
            "passed": False,
            "missing_gates": sorted(missing_gates),
            "qualification": qualifying,
            "selected": selected,
        }
    return {
        "executable": "pycbc_live",
        "qualification_results": qualification,
        "raw_results": runs,
        "science": science,
        "input_contract": checkpoint.get("input_contract", {}),
        "workload": checkpoint.get("workload", {}),
        "failure": {
            "phase": "incomplete_campaign",
            "error": "final campaign receipt unavailable; checkpoint evidence is provisional",
        },
    }


def compact_sources(provenance):
    result = {}
    for role in ("reference", "candidate"):
        source = provenance.get(role, {})
        result[role] = {
            key: source.get(key)
            for key in (
                "repository",
                "revision",
                "dirty",
                "tracked_diff_sha256",
                "status_sha256",
                "untracked_files_sha256",
                "files_sha256",
            )
        }
    return result


def stage_entry_points(stage_id, inspiral, live, jax):
    """Describe semantic source entry points without claiming causality."""
    result = {}
    for code, program_path in (("inspiral", inspiral), ("live", live)):
        if not program_path:
            continue
        candidate_path = (
            program_path if stage_id in {"segment", "selection"} else jax
        )
        result[code] = {}
        for role, path in (
            ("reference", program_path),
            ("candidate", candidate_path),
        ):
            anchor = (
                STAGE_ENTRY_ANCHORS.get(stage_id, {})
                .get(code, {})
                .get(role)
            )
            target = {"path": path, "kind": "stage entry point"}
            if anchor:
                target["symbol"] = (
                    anchor[4:].split("(", 1)[0] + "()"
                    if anchor.startswith("def ")
                    else anchor
                )
                target["anchor_text"] = anchor
            result[code][role] = target
    return result


def locate_captured_entry_points(report, output):
    """Add exact line anchors only for identity-verified captured files."""
    output = Path(output)
    for stage in report["stages"]:
        for code in stage["entry_points"].values():
            for role, target in code.items():
                capture = report.get("source_snapshots", {}).get(role, {})
                record = (
                    capture.get("files", {}).get(target["path"])
                    if capture.get("status") == "verified snapshot"
                    else None
                )
                if not record:
                    target.pop("anchor_text", None)
                    continue
                snapshot = (output / record["path"]).resolve()
                if (
                    not snapshot.is_relative_to(output.resolve())
                    or not snapshot.is_file()
                    or sha256_file(snapshot) != record["sha256"]
                ):
                    target.pop("anchor_text", None)
                    continue
                target["snapshot_path"] = record["path"]
                anchor = target.pop("anchor_text", None)
                if anchor:
                    for line, text in enumerate(
                        snapshot.read_text(errors="replace").splitlines(), 1
                    ):
                        if text.lstrip().startswith(anchor):
                            target["line"] = line
                            break


def build_report(suite_path):
    reader = SuiteReader(suite_path)
    try:
        suite = reader.read("suite.json")
        if not suite:
            raise ValueError("suite.json is missing")
        plan = reader.read("plan.json") or {}
        provenance = reader.read("provenance.json") or {}
        steps, rows, timings = [], [], []
        for index, step in enumerate(suite.get("steps", []), 1):
            step_id = f"step-{index:02d}"
            member = (
                f"{step['kind']}-{step['templates']}-{step['mode']}"
                "/campaign.json"
            )
            campaign = reader.read(member)
            provisional = False
            checkpoint_member = member + ".progress.json"
            if campaign is None and step["kind"].startswith("live-"):
                campaign = provisional_live_receipt(
                    reader.read(checkpoint_member), step
                )
                provisional = campaign is not None
            status = step.get("status", "not run")
            if (campaign is None or provisional) and status == "passed":
                status = "missing receipt"
            record = dict(
                id=step_id,
                kind=step["kind"],
                templates=step["templates"],
                mode=step["mode"],
                status=status,
                receipt=(checkpoint_member if provisional else member)
                if campaign else None,
                provisional=provisional,
                command=step.get("command", []),
                executable=(
                    campaign.get("executable")
                    if campaign
                    else (
                        "pycbc_inspiral"
                        if step["kind"] == "inspiral"
                        else "pycbc_live"
                    )
                ),
                workload=campaign.get("workload", {}) if campaign else {},
                input_contract=(
                    campaign.get("input_contract")
                    or campaign.get("workload_digest", {}).get("contract")
                    or campaign.get("inputs", {})
                    if campaign else {}
                ),
                science=(
                    {
                        arm: {
                            "passed": value.get("passed"),
                            "missing_gates": value.get("missing_gates", []),
                        }
                        for arm, value in campaign.get("science", {}).items()
                    }
                    if campaign
                    else {}
                ),
                failure=(step.get("failure") if provisional else None)
                or (campaign.get("failure") if campaign else None)
                or step.get("failure"),
            )
            steps.append(record)
            if campaign is None:
                continue
            for arm, science in campaign.get("science", {}).items():
                for scope, comparison in science_comparisons(science):
                    pair = (
                        comparison.get("evidence_reference"),
                        comparison.get("evidence_candidate"),
                    )
                    if (
                        not all(pair)
                        and record["executable"] == "pycbc_inspiral"
                    ):
                        qualification = campaign.get("qualification", {})
                        reference = qualification.get("original_cpu", {})
                        candidate = qualification.get(arm, {})
                        pair = (
                            reference.get("evidence_path"),
                            candidate.get("evidence_path"),
                        )
                    rows.extend(
                        field_rows(comparison, step_id, arm, scope, pair)
                    )
                if "detectors" not in science and "evidence" not in science:
                    rows.extend(
                        field_rows(
                            {
                                "missing_gates": science.get(
                                    "missing_gates", []
                                )
                            },
                            step_id,
                            arm,
                            "campaign qualification",
                        )
                    )
            timings.extend(timing_rows(campaign, step, step_id))
        return dict(
            schema_version=1,
            scope=suite.get("scope", plan.get("scope", "both")),
            suite_status=suite.get("status"),
            suite_error=suite.get("error"),
            source_artifact=Path(suite_path).name,
            source_sha256=reader.source_sha256(),
            sources=compact_sources(provenance),
            runtime={
                "platform": provenance.get("runtime", {}).get("platform"),
                "python": provenance.get("runtime", {}).get("python"),
                "packages": {
                    name: version
                    for name, version in provenance.get("runtime", {})
                    .get("packages", {})
                    .items()
                    if name
                    in {
                        "jax",
                        "jaxlib",
                        "jax-cuda13-pjrt",
                        "numpy",
                        "scipy",
                    }
                },
                "sample_rate": plan.get("sample_rate"),
                "precision": plan.get("precision"),
            },
            input_hashes={
                key: value
                for key, value in (plan.get("config") or {}).items()
                if key.endswith("sha256")
            },
            stages=[
                dict(
                    id=key,
                    label=label,
                    inspiral_source=inspiral,
                    live_source=live,
                    jax_source=jax,
                    entry_points=stage_entry_points(
                        key, inspiral, live, jax
                    ),
                )
                for key, label, inspiral, live, jax in STAGES
            ],
            steps=steps,
            comparisons=rows,
            timings=timings,
            coverage_note=(
                "Earliest observed difference only; uninstrumented stages "
                "remain unknown."
            ),
        )
    finally:
        reader.close()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_sources(reader, output):
    """Copy stage source only when the checkout still matches the suite.

    A dirty candidate's commit URL is insufficient; snapshots require the
    complete recorded tracked diff and untracked-file identity to match.
    """
    try:
        from tools.benchmark_artifact import source_identity
    except ModuleNotFoundError:
        from benchmark_artifact import source_identity
    provenance = reader.read("provenance.json") or {}
    captures = {}
    paths = {path for stage in STAGES for path in stage[2:] if path}
    for role in ("reference", "candidate"):
        expected = provenance.get(role, {})
        root = Path(expected.get("repository") or "/nonexistent")
        record = {"status": "source checkout unavailable", "files": {}}
        if root.is_dir():
            actual = source_identity(root)
            keys = (
                "revision",
                "dirty",
                "status_sha256",
                "tracked_diff_sha256",
                "untracked_files_sha256",
            )
            if all(actual.get(key) == expected.get(key) for key in keys):
                record["status"] = "verified snapshot"
                for path in sorted(paths):
                    source = root / path
                    if source.is_file():
                        destination = output / "source" / role / path
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, destination)
                        record["files"][path] = dict(
                            path=f"source/{role}/{path}",
                            sha256=sha256_file(destination),
                        )
            else:
                record["status"] = "checkout changed since suite"
        captures[role] = record
    return captures


def attach_annotations(report, path):
    """Attach only evidence-backed, run-specific explanatory notes."""
    if path is None:
        return
    annotations = json.loads(Path(path).read_text())
    if annotations.get("schema_version") != 1:
        raise ValueError("annotation schema_version must be 1")
    if annotations.get("source_sha256") != report.get("source_sha256"):
        raise ValueError(
            "annotation source_sha256 does not match suite archive"
        )
    allowed = {"observed", "supported by replay", "hypothesis", "unresolved"}
    entries = annotations.get("entries")
    if not isinstance(entries, list):
        raise ValueError("annotation entries must be a list")
    for entry in entries:
        if entry.get("status") not in allowed or not isinstance(
            entry.get("claim"), str
        ):
            raise ValueError("invalid annotation status or claim")
        evidence = entry.get("evidence", [])
        if (
            entry["status"] in {"observed", "supported by replay"}
            and not evidence
        ):
            raise ValueError("an observed or supported claim needs evidence")
        if not isinstance(evidence, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("label"), str)
            and isinstance(item.get("href"), str)
            for item in evidence
        ):
            raise ValueError("invalid annotation evidence")
        matched = [
            row
            for row in report["comparisons"]
            if row["step"] == entry.get("step")
            and row["arm"] == entry.get("arm")
            and row["stage"] == entry.get("stage")
            and row["path"] == entry.get("path")
        ]
        if not matched:
            raise ValueError("annotation does not match a comparison row")
        for row in matched:
            row["explanation"] = {
                key: entry[key]
                for key in ("status", "claim", "evidence")
                if key in entry
            }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        required=True,
        type=Path,
        help="suite phase directory or archived .tar.gz receipt",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new directory for a portable report",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="optional run-specific, evidence-linked explanation JSON",
    )
    parser.add_argument(
        "--reference-web-url",
        default="https://github.com/gwastro/pycbc",
        help="repository URL for reference commit links",
    )
    parser.add_argument(
        "--candidate-web-url",
        default="https://github.com/xangma/pycbc",
        help="repository URL for candidate commit links",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output directory already exists")
    report = build_report(args.suite)
    report["source_web_urls"] = {
        "reference": args.reference_web_url.rstrip("/"),
        "candidate": args.candidate_web_url.rstrip("/"),
    }
    attach_annotations(report, args.annotations)
    args.output.mkdir(parents=True)
    for name in ("index.html", "app.js", "style.css"):
        shutil.copy2(ASSETS / name, args.output / name)
    reader = SuiteReader(args.suite)
    try:
        patches = {}
        for role in ("reference", "candidate"):
            data = reader.bytes(role + ".patch")
            if data is not None:
                name = role + ".patch"
                (args.output / name).write_bytes(data)
                patches[role] = dict(
                    path=name, sha256=hashlib.sha256(data).hexdigest()
                )
        report["source_patches"] = patches
        report["source_snapshots"] = capture_sources(reader, args.output)
        locate_captured_entry_points(report, args.output)
    finally:
        reader.close()
    payload = json.dumps(report, indent=2, allow_nan=False)
    (args.output / "data.json").write_text(payload + "\n")
    (args.output / "data.js").write_text(
        "window.PYCBC_DIVERGENCE_REPORT = JSON.parse("
        + json.dumps(payload)
        + ");\n"
    )
    print(f"Built {args.output / 'index.html'}")
    print("Open index.html directly, or serve the directory over HTTP.")


if __name__ == "__main__":
    main()
