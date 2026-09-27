"""Fail-closed trigger and evidence comparisons for benchmark campaigns."""

from pathlib import Path
import importlib.util
import json

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "benchmark_science", ROOT / "tools" / "benchmark_science.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compare_scientific_hdf = _MODULE.compare_scientific_hdf
compare_live_scientific_hdf = _MODULE.compare_live_scientific_hdf


def _write_triggers(path, detectors=("H1",), omit=(), hash_delta=0,
                    include_extra=True):
    times = np.array([1000.0, 1000.25], dtype=np.float64)
    hashes = np.array([2**63 + 17, 2**63 + 29], dtype=np.uint64)
    values = {
        "template_hash": hashes,
        "end_time": times,
        "snr": np.array([8.0, 9.0]),
        "chisq": np.array([1.0, 1.2]),
        "chisq_dof": np.array([30, 30], dtype=np.int32),
        "sigmasq": np.array([2.0, 2.1]),
        "coa_phase": np.array([np.pi - 1e-5, -np.pi + 0.2]),
        "template_duration": np.array([1.0, 1.5]),
    }
    with h5py.File(path, "w") as out:
        for detector in detectors:
            for name, value in values.items():
                if (detector, name) not in omit:
                    data = value.copy()
                    if name == "template_hash":
                        data = data + np.uint64(hash_delta)
                    if name == "coa_phase" and detector == "L1":
                        data = data + 0.01
                    out.create_dataset(f"{detector}/{name}", data=data)
            if include_extra and (detector, "extra_metric") not in omit:
                out.create_dataset(f"{detector}/extra_metric", data=[3.0, 4.0])


def _write_evidence(path, config='{"sample_rate": 4096}'):
    with h5py.File(path, "w") as out:
        out.attrs["science_config"] = config
        out.create_dataset("conditioned_strain", data=np.array([1.0, 2.0]))
        segment = out.create_group("segments/0")
        segment.create_dataset("strain", data=np.array([3.0, 4.0]))
        segment.create_dataset("psd", data=np.array([np.inf, 2.0]))
        segment.attrs["analyze_start"] = 1
        segment.attrs["analyze_stop"] = 2
        segment.attrs["segment_start"] = 0
        segment.attrs["segment_stop"] = 2


def _set_chisq_mode(path, mode):
    with h5py.File(path, "a") as out:
        config = json.loads(out.attrs["science_config"])
        config["--jax-chisq-mode"] = mode
        out.attrs["science_config"] = json.dumps(config, sort_keys=True)
        out.attrs["jax_chisq_mode"] = mode


def _set_live_command(path, command):
    with h5py.File(path, "a") as out:
        out.attrs["command_line"] = np.asarray(command, dtype=h5py.string_dtype())


def _write_empty_live_block(path, partial=False):
    """Write a zero-detector block, or an intentionally partial one."""
    with h5py.File(path, "w") as out:
        out.attrs["num_live_detectors"] = 1 if partial else 0
        out.attrs["command_line"] = np.asarray(
            ["/source/bin/pycbc_live"], dtype=h5py.string_dtype())
        if partial:
            # An active block without the complete trigger schema must fail.
            out.create_dataset("H1/snr", data=np.array([], dtype=np.float64))
            return
        out.create_dataset("H1/template_hash", data=np.array([], dtype=np.uint64))
        out.create_dataset("H1/end_time", data=np.array([], dtype=np.float64))
        out.create_dataset("H1/snr", data=np.array([], dtype=np.float64))
        out.create_dataset("H1/chisq", data=np.array([], dtype=np.float64))
        out.create_dataset("H1/chisq_dof", data=np.array([], dtype=np.int32))
        out.create_dataset("H1/sigmasq", data=np.array([], dtype=np.float64))
        out.create_dataset("H1/coa_phase", data=np.array([], dtype=np.float64))
        out.create_dataset("H1/template_duration", data=np.array([], dtype=np.float64))


def _write_gates_only_live_block(path, num_live_detectors=0):
    """Write the buffer-fill shape emitted when no detector is live."""
    with h5py.File(path, "w") as out:
        out.attrs["num_live_detectors"] = num_live_detectors
        out.attrs["command_line"] = np.asarray(
            ["/source/bin/pycbc_live"], dtype=h5py.string_dtype())
        out.create_dataset("H1/gates", data=np.array(
            [(100.0, 0.5, 0.25)],
            dtype=[("center_time", "f8"), ("zero_half_width", "f8"),
                   ("taper_width", "f8")]))


def _write_metadata_only_evidence(path):
    with h5py.File(path, "w") as out:
        out.attrs["science_config"] = '{"sample_rate": 4096}'
        segment = out.create_group("segments/0")
        segment.attrs["analyze_start"] = 1
        segment.attrs["analyze_stop"] = 2
        segment.attrs["segment_start"] = 0
        segment.attrs["segment_stop"] = 2


def test_all_fields_identity_and_phase_wrap_are_strict_but_reordered(tmp_path):
    reference = tmp_path / "reference.hdf"
    candidate = tmp_path / "candidate.hdf"
    _write_triggers(reference)
    _write_triggers(candidate)
    with h5py.File(candidate, "a") as out:
        for name in ("template_hash", "end_time", "snr", "chisq",
                     "chisq_dof", "sigmasq", "coa_phase",
                     "template_duration", "extra_metric"):
            data = out[f"H1/{name}"][()]
            out[f"H1/{name}"][...] = data[::-1]
    result = compare_scientific_hdf(reference, candidate)
    assert result["observed_trigger_and_metadata_pass"]
    assert result["detectors"]["H1"]["fields"]["H1/coa_phase"]["passed"]
    assert result["detectors"]["H1"]["fields"]["H1/extra_metric"]["passed"]


def test_missing_field_duplicate_identity_and_hash_precision_fail_closed(tmp_path):
    reference = tmp_path / "reference.hdf"
    missing = tmp_path / "missing.hdf"
    duplicate = tmp_path / "duplicate.hdf"
    hash_changed = tmp_path / "hash_changed.hdf"
    _write_triggers(reference)
    _write_triggers(missing, omit={("H1", "chisq")})
    _write_triggers(duplicate)
    with h5py.File(duplicate, "a") as out:
        out["H1/end_time"][1] = out["H1/end_time"][0]
    _write_triggers(hash_changed, hash_delta=1)

    missing_result = compare_scientific_hdf(reference, missing)
    assert not missing_result["observed_trigger_and_metadata_pass"]
    assert "H1/chisq" in missing_result["detectors"]["H1"]["missing_fields"]
    assert not compare_scientific_hdf(reference, duplicate)[
        "observed_trigger_and_metadata_pass"
    ]
    hash_result = compare_scientific_hdf(reference, hash_changed)
    assert not hash_result["observed_trigger_and_metadata_pass"]
    assert hash_result["detectors"]["H1"]["reference_unmatched"]


def test_extra_detector_is_checked(tmp_path):
    reference = tmp_path / "reference.hdf"
    candidate = tmp_path / "candidate.hdf"
    _write_triggers(reference, detectors=("H1", "L1"))
    _write_triggers(candidate, detectors=("H1",))
    result = compare_scientific_hdf(reference, candidate)
    assert not result["observed_trigger_and_metadata_pass"]
    assert not result["detectors"]["L1"]["observed_trigger_and_metadata_pass"]


def test_complete_evidence_masks_matching_infinities_and_checks_config(tmp_path):
    reference = tmp_path / "reference.hdf"
    candidate = tmp_path / "candidate.hdf"
    evidence_reference = tmp_path / "reference-science.hdf"
    evidence_candidate = tmp_path / "candidate-science.hdf"
    _write_triggers(reference)
    _write_triggers(candidate)
    _write_evidence(evidence_reference)
    _write_evidence(evidence_candidate)
    result = compare_scientific_hdf(
        reference, candidate, evidence_reference=evidence_reference,
        evidence_candidate=evidence_candidate,
    )
    assert result["passed"]
    assert not result["missing_gates"]
    assert result["evidence_reference"] == str(evidence_reference)
    assert result["evidence_candidate"] == str(evidence_candidate)
    assert result["evidence"]["segments/0/psd"]["passed"]

    with h5py.File(evidence_candidate, "a") as out:
        out.attrs["science_config"] = '{"sample_rate": 2048}'
    changed = compare_scientific_hdf(
        reference, candidate, evidence_reference=evidence_reference,
        evidence_candidate=evidence_candidate,
    )
    assert not changed["passed"]
    assert not changed["evidence"]["/@science_config"]["passed"]


def test_cpu_mode_omission_is_equivalent_to_explicit_compatible(tmp_path):
    reference = tmp_path / "reference.hdf"
    candidate = tmp_path / "candidate.hdf"
    evidence_reference = tmp_path / "reference-science.hdf"
    evidence_candidate = tmp_path / "candidate-science.hdf"
    _write_triggers(reference)
    _write_triggers(candidate)
    _write_evidence(evidence_reference)
    _write_evidence(evidence_candidate)
    _set_chisq_mode(evidence_candidate, "cpu-compatible")
    result = compare_scientific_hdf(
        reference, candidate, evidence_reference=evidence_reference,
        evidence_candidate=evidence_candidate)
    assert result["passed"]
    mode = result["evidence"]["/@jax_chisq_mode"]
    assert mode["reference_mode"] == mode["candidate_mode"] == "cpu-compatible"
    assert mode["reference_declared"] is None
    assert mode["candidate_declared"] == "cpu-compatible"


def test_direct_phase_mode_must_match_reference(tmp_path):
    reference = tmp_path / "reference.hdf"
    candidate = tmp_path / "candidate.hdf"
    evidence_reference = tmp_path / "reference-science.hdf"
    evidence_candidate = tmp_path / "candidate-science.hdf"
    _write_triggers(reference)
    _write_triggers(candidate)
    _write_evidence(evidence_reference)
    _write_evidence(evidence_candidate)
    _set_chisq_mode(evidence_candidate, "direct-phase")
    result = compare_scientific_hdf(
        reference, candidate, evidence_reference=evidence_reference,
        evidence_candidate=evidence_candidate)
    assert not result["passed"]
    assert not result["evidence"]["/@jax_chisq_mode"]["passed"]

    _set_chisq_mode(evidence_reference, "direct-phase")
    same = compare_scientific_hdf(
        reference, candidate, evidence_reference=evidence_reference,
        evidence_candidate=evidence_candidate)
    assert same["passed"]


def test_other_science_configuration_mismatch_still_fails(tmp_path):
    reference = tmp_path / "reference.hdf"
    candidate = tmp_path / "candidate.hdf"
    evidence_reference = tmp_path / "reference-science.hdf"
    evidence_candidate = tmp_path / "candidate-science.hdf"
    _write_triggers(reference)
    _write_triggers(candidate)
    _write_evidence(evidence_reference, '{"sample_rate": 4096}')
    _write_evidence(evidence_candidate, '{"sample_rate": 2048}')
    result = compare_scientific_hdf(
        reference, candidate, evidence_reference=evidence_reference,
        evidence_candidate=evidence_candidate)
    assert not result["passed"]
    assert not result["evidence"]["/@science_config"]["passed"]


def test_live_auxiliary_psd_and_gates_are_not_reordered_with_triggers(tmp_path):
    reference = tmp_path / "reference-live.hdf"
    candidate = tmp_path / "candidate-live.hdf"
    _write_triggers(reference)
    _write_triggers(candidate)
    with h5py.File(reference, "a") as out:
        out.attrs["num_live_detectors"] = 1
        out.create_dataset("H1/psd", data=np.array([np.inf, 2.0, 3.0]))
        out.create_dataset("H1/gates", data=np.array(
            [(100.0, 0.5, 0.25), (200.0, 0.25, 0.1)],
            dtype=[("center_time", "f8"), ("zero_half_width", "f8"),
                   ("taper_width", "f8")]))
    with h5py.File(candidate, "a") as out:
        out.attrs["num_live_detectors"] = 1
        for name in ("template_hash", "end_time", "snr", "chisq",
                     "chisq_dof", "sigmasq", "coa_phase",
                     "template_duration", "extra_metric"):
            data = out[f"H1/{name}"][()]
            out[f"H1/{name}"][...] = data[::-1]
        # These arrays are block-level live metadata, not trigger rows.
        out.create_dataset("H1/psd", data=np.array([np.inf, 2.0, 3.0]))
        out.create_dataset("H1/gates", data=np.array(
            [(100.0, 0.5, 0.25), (200.0, 0.25, 0.1)],
            dtype=[("center_time", "f8"), ("zero_half_width", "f8"),
                   ("taper_width", "f8")]))

    result = compare_live_scientific_hdf(reference, candidate)
    detector = result["detectors"]["H1"]
    assert result["observed_trigger_and_metadata_pass"]
    assert detector["fields"]["H1/psd"]["passed"]
    assert detector["fields"]["H1/gates"]["passed"]
    assert not detector["fields"]["H1/psd"]["matched_trigger_field"]
    assert not detector["fields"]["H1/gates"]["matched_trigger_field"]


def test_live_command_normalizes_only_runner_controls(tmp_path):
    reference = tmp_path / "reference-live.hdf"
    candidate = tmp_path / "candidate-live.hdf"
    _write_triggers(reference)
    _write_triggers(candidate)
    _set_live_command(reference, [
        "/reference/bin/pycbc_live", "--processing-scheme", "cpu:1",
        "--output-path", "/reference/out", "--benchmark-evidence-path",
        "/reference/evidence", "--sample-rate", "4096",
    ])
    _set_live_command(candidate, [
        "/candidate/bin/pycbc_live", "--processing-scheme", "jax:cuda:0",
        "--output-path", "/candidate/out", "--benchmark-evidence-path",
        "/candidate/evidence", "--sample-rate", "4096",
    ])
    result = compare_live_scientific_hdf(reference, candidate)
    assert result["observed_trigger_and_metadata_pass"]
    assert result["detectors"]["H1"]["attributes"]["/@command_line"]["passed"]

    _set_live_command(candidate, [
        "/candidate/bin/pycbc_live", "--processing-scheme", "jax:cuda:0",
        "--output-path", "/candidate/out", "--sample-rate", "2048",
    ])
    changed = compare_live_scientific_hdf(reference, candidate)
    assert not changed["observed_trigger_and_metadata_pass"]
    assert not changed["detectors"]["H1"]["attributes"]["/@command_line"]["passed"]


def test_live_command_normalizes_explicit_cpu_chisq_mode(tmp_path):
    reference = tmp_path / "reference-live.hdf"
    candidate = tmp_path / "candidate-live.hdf"
    _write_triggers(reference)
    _write_triggers(candidate)
    _set_live_command(reference, ["/reference/bin/pycbc_live", "--sample-rate", "2048"])
    _set_live_command(candidate, ["/candidate/bin/pycbc_live", "--jax-chisq-mode",
                                 "cpu-compatible", "--sample-rate", "2048"])
    result = compare_live_scientific_hdf(reference, candidate)
    assert result["detectors"]["H1"]["attributes"]["/@command_line"]["passed"]
    _set_live_command(candidate, ["/candidate/bin/pycbc_live",
                                 "--jax-chisq-mode=cpu-compatible", "--sample-rate", "2048"])
    result = compare_live_scientific_hdf(reference, candidate)
    assert result["detectors"]["H1"]["attributes"]["/@command_line"]["passed"]
    _set_live_command(candidate, ["/candidate/bin/pycbc_live", "--jax-chisq-mode",
                                 "direct-phase", "--sample-rate", "2048"])
    result = compare_live_scientific_hdf(reference, candidate)
    assert not result["detectors"]["H1"]["attributes"]["/@command_line"]["passed"]


def test_live_observer_controls_and_source_provenance_are_recorded(tmp_path):
    reference, candidate = tmp_path / 'reference.hdf', tmp_path / 'candidate.hdf'
    for path, version in ((reference, 'upstream revision'), (candidate, 'jax revision')):
        _write_triggers(path)
        with h5py.File(path, 'a') as out:
            out.attrs['pycbc_version'] = version
    _set_live_command(reference, ['/upstream/bin/pycbc_live', '--sample-rate', '2048'])
    _set_live_command(candidate, [
        '/candidate/bin/pycbc_live', '--replay-clock', '--replay-rate=1',
        '--sample-rate', '2048'])
    result = compare_live_scientific_hdf(reference, candidate)
    assert result['observed_trigger_and_metadata_pass']
    version = result['detectors']['H1']['attributes']['/@pycbc_version']
    assert not version['exact']
    assert version['reference'] == 'upstream revision'
    assert version['candidate'] == 'jax revision'
    # An unknown metadata change and a missing version still fail closed.
    with h5py.File(candidate, 'a') as out:
        out.attrs['scientific_option'] = 'changed'
    assert not compare_live_scientific_hdf(reference, candidate)[
        'observed_trigger_and_metadata_pass']
    with h5py.File(candidate, 'a') as out:
        del out.attrs['scientific_option']
        del out.attrs['pycbc_version']
    assert not compare_live_scientific_hdf(reference, candidate)[
        'observed_trigger_and_metadata_pass']


def test_live_zero_detector_block_compares_complete_schema_and_evidence(tmp_path):
    reference = tmp_path / "reference-empty-live.hdf"
    candidate = tmp_path / "candidate-empty-live.hdf"
    evidence_reference = tmp_path / "reference-science.hdf"
    evidence_candidate = tmp_path / "candidate-science.hdf"
    _write_empty_live_block(reference)
    _write_empty_live_block(candidate)
    _write_evidence(evidence_reference)
    _write_evidence(evidence_candidate)

    result = compare_live_scientific_hdf(
        reference, candidate, evidence_reference=evidence_reference,
        evidence_candidate=evidence_candidate,
    )
    assert result["observed_trigger_and_metadata_pass"]
    assert result["passed"]
    assert not result["missing_gates"]


def test_live_gates_only_buffer_fill_uses_metadata_evidence_and_inactive_gate(tmp_path):
    reference = tmp_path / "reference-buffer-live.hdf"
    candidate = tmp_path / "candidate-buffer-live.hdf"
    evidence_reference = tmp_path / "reference-metadata.hdf"
    evidence_candidate = tmp_path / "candidate-metadata.hdf"
    _write_gates_only_live_block(reference)
    _write_gates_only_live_block(candidate)
    _write_metadata_only_evidence(evidence_reference)
    _write_metadata_only_evidence(evidence_candidate)

    result = compare_live_scientific_hdf(
        reference, candidate, evidence_reference=evidence_reference,
        evidence_candidate=evidence_candidate,
    )
    assert result["passed"]
    assert result["detectors"]["H1"]["inactive_buffer_block"]
    assert not result["missing_gates"]

    _write_gates_only_live_block(candidate, num_live_detectors=1)
    active = compare_live_scientific_hdf(
        reference, candidate, evidence_reference=evidence_reference,
        evidence_candidate=evidence_candidate,
    )
    assert not active["passed"]
    assert active["missing_gates"] == ["conditioned_strain", "full_psd"]
    assert not active["detectors"]["H1"].get("inactive_buffer_block", False)


def test_live_active_partial_schema_fails_closed(tmp_path):
    reference = tmp_path / "reference-partial-live.hdf"
    candidate = tmp_path / "candidate-partial-live.hdf"
    _write_empty_live_block(reference, partial=True)
    _write_empty_live_block(candidate, partial=True)

    result = compare_live_scientific_hdf(reference, candidate)
    assert not result["observed_trigger_and_metadata_pass"]
    assert not result["detectors"]["H1"].get("inactive_buffer_block", False)
