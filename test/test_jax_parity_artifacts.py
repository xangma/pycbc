"""Focused tests for JAX parity artifacts and policy comparison."""

import json
from pathlib import Path

import numpy as np

from tools.jax_parity import compare, manifest


def test_compare_detects_mismatched_shapes(tmp_path):
    ref_json = tmp_path / "ref.json"
    ref_npz = tmp_path / "ref.npz"
    cand_json = tmp_path / "cand.json"
    cand_npz = tmp_path / "cand.npz"
    report = tmp_path / "report.json"

    meta = {
        "schema_version": 1,
        "label": "ref",
        "runtime": {"source_revision": "abc"},
        "timings_seconds": {},
        "records": {"arr": {"shape": [4], "dtype": "float64", "storage": "numpy"}},
    }
    ref_json.write_text(json.dumps(meta))
    np.savez(ref_npz, arr=np.arange(4, dtype=np.float64))

    cand_meta = dict(meta, label="cand", records={"arr": {"shape": [5], "dtype": "float64", "storage": "numpy"}})
    cand_json.write_text(json.dumps(cand_meta))
    np.savez(cand_npz, arr=np.arange(5, dtype=np.float64))

    policy = Path(__file__).resolve().parents[1] / "tools/jax_parity/policy.json"
    ret = compare.main([str(ref_json), str(cand_json), "--profile", "jax", "--policy", str(policy), "--report", str(report)])
    assert ret == 1
    report_data = json.loads(report.read_text())
    assert not report_data["passed"]


def test_manifest_sealing_and_verification(tmp_path):
    output = tmp_path / "manifest.json"
    data = manifest.build_manifest("test-label", output)
    assert output.exists()
    assert data["content_sha256"] is not None
    loaded = json.loads(output.read_text())
    assert loaded["label"] == "test-label"
