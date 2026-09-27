"""Behavioral checks for the receipt-only divergence viewer builder."""

import json
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_jax_divergence_report",
    ROOT / "tools" / "build_jax_divergence_report.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
attach_annotations = MODULE.attach_annotations
build_report = MODULE.build_report
locate_captured_entry_points = MODULE.locate_captured_entry_points
point_examples = MODULE.point_examples
timing_rows = MODULE.timing_rows


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_failed_qualification_does_not_create_live_or_publishable_timing(
    tmp_path,
):
    _write(
        tmp_path / "suite.json",
        {
            "status": "failed",
            "steps": [
                {
                    "kind": "inspiral",
                    "templates": 32,
                    "mode": "qualification",
                    "status": "failed",
                },
                {
                    "kind": "live-unpaced",
                    "templates": 32,
                    "mode": "qualification",
                },
            ],
        },
    )
    _write(
        tmp_path / "inspiral-32-qualification" / "campaign.json",
        {
            "executable": "pycbc_inspiral",
            "status": "failed",
            "qualification": {
                "original_cpu": {"elapsed_wall_sec": 10},
                "jax_cpu_batched": {"elapsed_wall_sec": 9},
            },
            "raw_results": {"original_cpu": [], "jax_cpu_batched": []},
            "science": {
                "jax_cpu_batched": {
                    "passed": False,
                    "detectors": {
                        "H1": {
                            "reference_triggers": 3,
                            "candidate_triggers": 3,
                            "matched_triggers": 3,
                            "identity_pass": True,
                            "fields": {
                                "H1/chisq": {
                                    "passed": False,
                                    "failed_elements": 1,
                                    "max_absolute_difference": 0.02,
                                }
                            },
                        }
                    },
                    "evidence": {
                        "conditioned_strain": {
                            "passed": False,
                            "failed_elements": 20,
                        }
                    },
                }
            },
        },
    )
    report = build_report(tmp_path)
    assert report["steps"][1]["status"] == "not run"
    assert {
        r["stage"] for r in report["comparisons"] if r["verdict"] == "fail"
    } == {"conditioning", "chisq"}
    assert all(not item["publishable"] for item in report["timings"])
    assert not any(item["step"] == "step-02" for item in report["comparisons"])


def test_scoped_suite_contains_only_selected_code(tmp_path):
    _write(
        tmp_path / "suite.json",
        {
            "scope": "live",
            "steps": [
                {
                    "kind": "live-unpaced",
                    "templates": 32,
                    "mode": "qualification",
                    "status": "failed",
                }
            ],
        },
    )
    report = build_report(tmp_path)
    assert report["scope"] == "live"
    assert [step["executable"] for step in report["steps"]] == [
        "pycbc_live"
    ]


def test_stage_line_anchor_requires_verified_captured_source(tmp_path):
    source = tmp_path / "source" / "reference" / "pycbc" / "psd"
    source.mkdir(parents=True)
    captured = source / "estimate.py"
    captured.write_text("# heading\ndef welch(values):\n    pass\n")
    path = "pycbc/psd/estimate.py"
    report = {
        "stages": [
            {
                "entry_points": {
                    "inspiral": {
                        "reference": {
                            "path": path,
                            "anchor_text": "def welch(",
                        },
                        "candidate": {
                            "path": path,
                            "anchor_text": "def welch(",
                        },
                    }
                }
            }
        ],
        "source_snapshots": {
            "reference": {
                "status": "verified snapshot",
                "files": {
                    path: {
                        "path": "source/reference/" + path,
                        "sha256": MODULE.sha256_file(captured),
                    }
                },
            },
            "candidate": {
                "status": "checkout changed since suite",
                "files": {},
            },
        },
    }
    locate_captured_entry_points(report, tmp_path)
    reference = report["stages"][0]["entry_points"]["inspiral"][
        "reference"
    ]
    candidate = report["stages"][0]["entry_points"]["inspiral"][
        "candidate"
    ]
    assert reference["line"] == 2
    assert reference["snapshot_path"] == "source/reference/" + path
    assert "line" not in candidate
    assert "snapshot_path" not in candidate
    assert "anchor_text" not in candidate


def test_portable_report_keeps_source_links_without_exact_snapshot(tmp_path):
    suite = tmp_path / "suite"
    _write(
        suite / "suite.json",
        {
            "scope": "inspiral",
            "steps": [
                {
                    "kind": "inspiral",
                    "templates": 32,
                    "mode": "qualification",
                    "status": "not run",
                }
            ],
        },
    )
    output = tmp_path / "report"
    MODULE.main(["--suite", str(suite), "--output", str(output)])
    report = json.loads((output / "data.json").read_text())
    target = report["stages"][0]["entry_points"]["inspiral"][
        "candidate"
    ]
    assert report["scope"] == "inspiral"
    assert target["path"] == "pycbc/strain/strain_jax.py"
    assert target["kind"] == "stage entry point"
    assert "line" not in target
    assert "snapshot_path" not in target
    assert "anchor_text" not in target


def test_raw_trigger_examples_match_identity_not_hdf_row(tmp_path):
    reference = tmp_path / "reference.hdf"
    candidate = tmp_path / "candidate.hdf"
    with h5py.File(reference, "w") as out:
        group = out.create_group("H1")
        group.create_dataset(
            "template_hash", data=np.array([10, 20], dtype="uint64")
        )
        group.create_dataset("end_time", data=np.array([100.0, 100.5]))
        group.create_dataset(
            "chisq", data=np.array([3.0, 4.0], dtype="float32")
        )
    with h5py.File(candidate, "w") as out:
        group = out.create_group("H1")
        group.create_dataset(
            "template_hash", data=np.array([20, 10], dtype="uint64")
        )
        group.create_dataset("end_time", data=np.array([100.5, 100.0]))
        group.create_dataset(
            "chisq", data=np.array([4.25, 3.0], dtype="float32")
        )
    detail = point_examples(reference, candidate, "H1/chisq", matched=True)
    assert detail["status"] == "available"
    assert detail["exact_different_elements"] == 1
    assert detail["first"]["identity"] == {
        "template_hash": 20,
        "sample": 1024,
        "gps": 100.5,
    }
    assert detail["first"]["absolute_difference"] == 0.25
    json.dumps(detail, allow_nan=False)


def test_live_evidence_points_have_time_or_frequency(tmp_path):
    reference = tmp_path / "reference-evidence.hdf"
    candidate = tmp_path / "candidate-evidence.hdf"
    for path, middle in ((reference, 1.0), (candidate, 1.5)):
        with h5py.File(path, "w") as out:
            out.create_dataset("conditioned_strain", data=[0.0, middle, 0.0])
            group = out.create_group("segments/H1")
            group.attrs["epoch"] = 100.0
            group.attrs["delta_t"] = 0.5
            group.attrs["delta_f"] = 2.0
            group.create_dataset("strain", data=[0.0, middle, 0.0])
            group.create_dataset("psd", data=[0.0, middle, 0.0])
    strain = point_examples(reference, candidate, "conditioned_strain")
    psd = point_examples(reference, candidate, "segments/H1/psd")
    assert strain["first"]["gps"] == 100.5
    assert psd["first"]["frequency_hz"] == 2.0
    json.dumps(psd, allow_nan=False)


def test_live_campaign_missing_gate_is_separate_from_selection(tmp_path):
    _write(
        tmp_path / "suite.json",
        {
            "status": "failed",
            "steps": [
                {
                    "kind": "live-unpaced",
                    "templates": 32,
                    "mode": "qualification",
                    "status": "failed",
                }
            ],
        },
    )
    _write(
        tmp_path / "live-unpaced-32-qualification" / "campaign.json",
        {
            "executable": "pycbc_live",
            "science": {
                "jax_cpu": {
                    "passed": False,
                    "missing_gates": ["worker decision trace"],
                    "qualification": [],
                }
            },
        },
    )
    report = build_report(tmp_path)
    assert [
        (r["stage"], r["path"], r["verdict"]) for r in report["comparisons"]
    ] == [("gates", "worker decision trace", "missing")]


def test_live_report_uses_semantic_contract_and_suite_failure(tmp_path):
    _write(tmp_path / "suite.json", {
        "status": "failed", "steps": [{
            "kind": "live-unpaced", "templates": 32,
            "mode": "qualification", "status": "failed",
            "failure": {"error": "science gate failed"},
        }],
    })
    contract = {"bank_sha256": "same-across-runs"}
    _write(tmp_path / "live-unpaced-32-qualification" / "campaign.json", {
        "executable": "pycbc_live",
        "inputs": {"config": "/unique/job/path/config.json"},
        "workload_digest": {"sha256": "digest", "contract": contract},
        "science": {},
    })
    report = build_report(tmp_path)
    assert report["steps"][0]["input_contract"] == contract
    assert report["steps"][0]["failure"] == {"error": "science gate failed"}


def test_interrupted_live_checkpoint_is_provisional(tmp_path):
    _write(tmp_path / "suite.json", {
        "status": "failed", "steps": [{
            "kind": "live-unpaced", "templates": 32,
            "mode": "timing", "status": "failed",
        }],
    })
    _write(
        tmp_path / "live-unpaced-32-timing" / "campaign.json.progress.json",
        {
            "status": "running",
            "input_contract": {"bank_sha256": "fixed"},
            "workload": {"templates": 32},
            "qualification_results": {"jax_cpu": [{
                "elapsed_wall_sec": 12,
                "science": {"passed": False,
                            "missing_gates": ["conditioned strain"]},
            }]},
            "raw_results": {"jax_cpu": [{"elapsed_wall_sec": 10}]},
        },
    )
    report = build_report(tmp_path)
    step = report["steps"][0]
    assert step["provisional"] is True
    assert step["receipt"].endswith("campaign.json.progress.json")
    assert step["input_contract"] == {"bank_sha256": "fixed"}
    assert step["science"]["jax_cpu"]["passed"] is False
    assert step["failure"]["phase"] == "incomplete_campaign"
    assert any(row["path"] == "conditioned strain"
               for row in report["comparisons"])
    assert len(report["timings"]) == 2
    assert all(not row["publishable"] for row in report["timings"])


def test_timing_without_science_cannot_be_publishable():
    receipt = {
        "raw_results": {"jax_cpu": [{"elapsed_wall_sec": 2.0}]},
        "science": {},
    }
    step = {"status": "passed", "mode": "timing"}
    assert not timing_rows(receipt, step, "step-01")[0]["publishable"]


def test_live_per_block_comparison_is_preserved(tmp_path):
    reference = tmp_path / "reference-evidence.hdf"
    candidate = tmp_path / "candidate-evidence.hdf"
    for path, middle in ((reference, 1.0), (candidate, 1.5)):
        with h5py.File(path, "w") as out:
            out.attrs["strain_epoch"] = 100.0
            out.attrs["strain_delta_t"] = 0.5
            out.create_dataset("conditioned_strain", data=[0.0, middle, 0.0])
    _write(
        tmp_path / "suite.json",
        {
            "status": "failed",
            "steps": [
                {
                    "kind": "live-unpaced",
                    "templates": 32,
                    "mode": "qualification",
                    "status": "failed",
                }
            ],
        },
    )
    comparison = {
        "evidence_reference": str(reference),
        "evidence_candidate": str(candidate),
        "evidence": {
            "conditioned_strain": {"passed": False, "failed_elements": 1}
        },
        "detectors": {
            "H1": {
                "identity_pass": True,
                "matched_triggers": 2,
                "fields": {
                    "H1/chisq": {"passed": False, "failed_elements": 1}
                },
            }
        },
    }
    _write(
        tmp_path / "live-unpaced-32-qualification" / "campaign.json",
        {
            "executable": "pycbc_live",
            "science": {
                "jax_cpu": {
                    "passed": False,
                    "qualification": [
                        {"comparisons": [comparison], "passed": False}
                    ],
                    "selected": [],
                }
            },
        },
    )
    report = build_report(tmp_path)
    failures = [r for r in report["comparisons"] if r["verdict"] == "fail"]
    assert [(r["scope"], r["stage"], r["path"]) for r in failures] == [
        ("qualification run 1, block 1", "chisq", "H1/chisq"),
        (
            "qualification run 1, block 1",
            "conditioning",
            "conditioned_strain",
        ),
    ]
    assert failures[1]["examples"]["first"]["gps"] == 100.5


def test_directory_annotation_key_changes_with_campaign_receipt(tmp_path):
    _write(
        tmp_path / "suite.json",
        {
            "steps": [
                {
                    "kind": "inspiral",
                    "templates": 1,
                    "mode": "qualification",
                    "status": "failed",
                }
            ]
        },
    )
    campaign = tmp_path / "inspiral-1-qualification" / "campaign.json"
    _write(campaign, {"science": {"jax_cpu": {"passed": False}}})
    before = build_report(tmp_path)["source_sha256"]
    _write(campaign, {"science": {"jax_cpu": {"passed": True}}})
    after = build_report(tmp_path)["source_sha256"]
    assert before != after


def test_annotation_requires_matching_run_and_evidence(tmp_path):
    report = {
        "source_sha256": "a" * 64,
        "comparisons": [
            {
                "step": "step-01",
                "arm": "jax_cpu_batched",
                "stage": "chisq",
                "path": "H1/chisq",
            }
        ],
    }
    annotation = {
        "schema_version": 1,
        "source_sha256": "b" * 64,
        "entries": [],
    }
    path = tmp_path / "annotations.json"
    _write(path, annotation)
    with pytest.raises(ValueError, match="does not match"):
        attach_annotations(report, path)
    annotation["source_sha256"] = "a" * 64
    annotation["entries"] = [
        {
            "step": "step-01",
            "arm": "jax_cpu_batched",
            "stage": "chisq",
            "path": "H1/chisq",
            "status": "supported by replay",
            "claim": "Controlled result",
            "evidence": [],
        }
    ]
    _write(path, annotation)
    with pytest.raises(ValueError, match="needs evidence"):
        attach_annotations(report, path)
