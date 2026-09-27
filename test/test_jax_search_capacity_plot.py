"""Regression checks for capacity units in the published benchmark plots."""

import matplotlib
import pytest

matplotlib.use("Agg")

from tools.plot_jax_search_capacity import (  # noqa: E402
    capacity_summary,
    campaign_rows,
    plot_inspiral,
    plot_campaign,
    plot_live,
    render_campaign_report,
)

def test_live_uses_valid_advance_and_filter_stage(tmp_path):
    data = {
        "sample_rate_hz": 2048,
        "matched_filter_dtype": "complex64",
        "experiments": {"streaming_n131072": {"arms": {
            arm: {"batches": {"1": {"samples_seconds": {
                "filter": samples}}}}
            for arm, samples in {
                "branch_cpu": [0.005, 0.005],
                "jax_cpu_lal": [0.002, 0.002],
                "jax_cuda_lal": [0.001, 0.001],
            }.items()
        }}}
    }
    cells = data["experiments"]["streaming_n131072"]["arms"]
    # Values use 56 valid seconds, not the 64-second FFT duration or total
    # waveform-plus-transfer-plus-filter time. CUDA means a GPU + host core.
    for arm, expected in (("branch_cpu", 11200),
                          ("jax_cuda_lal", 56000)):
        timings = cells[arm]["batches"]["1"]["samples_seconds"]["filter"]
        assert round(capacity_summary(56, timings)[0]) == expected
    output = tmp_path / "live.png"
    plot_live(data, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_inspiral_distinguishes_wall_and_calculation_capacity(tmp_path):
    data = {
        "sample_rate_hz": 2048,
        "matched_filter_dtype": "complex64",
        "runs": {
            "branch_cpu_rep1": {"timing": {"wall_seconds": 10,
                                              "calc_seconds": 5}},
            "branch_cpu_rep2": {"timing": {"wall_seconds": 12,
                                              "calc_seconds": 6}},
            "jax_cpu_batched_rep1": {"timing": {"wall_seconds": 20,
                                                   "calc_seconds": 10}},
            "jax_cpu_batched_rep2": {"timing": {"wall_seconds": 22,
                                                   "calc_seconds": 11}},
            "jax_cuda_batched_rep1": {"timing": {"wall_seconds": 2,
                                                   "calc_seconds": 1}},
            "jax_cuda_batched_rep2": {"timing": {"wall_seconds": 2.5,
                                                   "calc_seconds": 1.25}},
        },
    }
    work = 384 * 1904
    expected = {"branch_cpu": (round((work / 10 + work / 12) / 2),
                                round((work / 5 + work / 6) / 2)),
                "jax_cpu_batched": (round((work / 20 + work / 22) / 2),
                                     round((work / 10 + work / 11) / 2)),
                "jax_cuda_batched": (round((work / 2 + work / 2.5) / 2),
                                     round((work / 1 + work / 1.25) / 2))}
    for arm, capacities in expected.items():
        for field, target in zip(("wall_seconds", "calc_seconds"), capacities):
            samples = [run["timing"][field]
                       for name, run in data["runs"].items()
                       if name.startswith(arm + "_rep")]
            mid, low, high = capacity_summary(384 * 1904, samples)
            assert round(mid) == target
            assert low <= mid <= high
    output = tmp_path / "inspiral.png"
    plot_inspiral(data, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_live_rejects_advance_exceeding_fft_duration(tmp_path):
    with pytest.raises(ValueError, match="block advance"):
        plot_live({}, tmp_path / "invalid.png", advance_seconds=65)


def test_complete_campaign_reports_wall_steady_resources_and_science(tmp_path):
    receipt = {
        "executable": "pycbc_live",
        "workload": {"templates": 256, "valid_detector_seconds": 56},
        "raw_results": {
            "cpu": [
                {"elapsed_wall_sec": 10.0, "steady_time_sec": 8.0,
                 "completed_template_seconds": 256 * 56,
                 "resources": {"physical_cpu_cores": 2, "gpus": 0}},
                {"elapsed_wall_sec": 12.0, "steady_time_sec": 9.0,
                 "completed_template_seconds": 256 * 56,
                 "resources": {"physical_cpu_cores": 2, "gpus": 0}},
            ],
            "cuda": [
                {"elapsed_wall_sec": 2.0, "steady_time_sec": 1.5,
                 "completed_template_seconds": 256 * 56,
                 "resources": {"physical_cpu_cores": 1, "gpus": 1}},
                {"elapsed_wall_sec": 2.5, "steady_time_sec": 1.8,
                 "completed_template_seconds": 256 * 56,
                 "resources": {"physical_cpu_cores": 1, "gpus": 1}},
            ],
        },
        "science": {
            "cpu": {"passed": True, "scope": "trigger parity"},
            "cuda": {"passed": False, "scope": "pending",
                     "missing_gates": ["PSD"]},
        },
        "latency": {"cpu": {"p95_seconds": 11.0}},
    }
    rows = campaign_rows(receipt)
    assert rows[0]["templates"] == 256
    assert rows[0]["physical_cpu_cores"] == 2
    assert rows[1]["gpus"] == 1
    assert round(rows[1]["wall_capacity"][0]) == round(
        (256 * 56 / 2 + 256 * 56 / 2.5) / 2)
    report = render_campaign_report(receipt)
    assert "cpu" in report and "science passed" in report
    assert "science failed: PSD" in report
    assert "latency" in report
    output = tmp_path / "campaign.png"
    plot_campaign(receipt, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_campaign_rejects_mismatched_completed_work():
    receipt = {
        "executable": "pycbc_inspiral",
        "raw_results": {"cpu": [
            {"elapsed_wall_sec": 1, "completed_template_seconds": 10,
             "resources": {"physical_cpu_cores": 1, "gpus": 0}},
            {"elapsed_wall_sec": 1, "completed_template_seconds": 11,
             "resources": {"physical_cpu_cores": 1, "gpus": 0}},
        ]},
    }
    with pytest.raises(ValueError, match="equal completed work"):
        campaign_rows(receipt)


def test_campaign_requires_completed_work_even_with_workload_metadata():
    receipt = {
        "executable": "pycbc_live",
        "workload": {"templates": 8, "valid_detector_seconds": 4},
        "raw_results": {"cpu": [{
            "elapsed_wall_sec": 1,
            "resources": {"physical_cpu_cores": 1},
        }]},
    }
    with pytest.raises(ValueError, match="completed_template_seconds"):
        campaign_rows(receipt)


def _valid_campaign_run(**overrides):
    run = {
        "elapsed_wall_sec": 2.0,
        "completed_template_seconds": 100.0,
        "resources": {"physical_cpu_cores": 2, "gpus": 0},
    }
    run.update(overrides)
    return run


def test_campaign_rejects_missing_allocation_and_failed_or_nonfinite_runs():
    base = {"executable": "pycbc_inspiral",
            "raw_results": {"cpu": [_valid_campaign_run()]}}
    missing_cores = {"executable": "pycbc_inspiral", "raw_results": {
        "cpu": [_valid_campaign_run(resources={"gpus": 0})]}}
    with pytest.raises(ValueError, match="physical CPU cores"):
        campaign_rows(missing_cores)
    failed = {"executable": "pycbc_inspiral", "raw_results": {
        "cpu": [_valid_campaign_run(returncode=1)]}}
    with pytest.raises(ValueError, match="returncode"):
        campaign_rows(failed)
    nonfinite = {"executable": "pycbc_inspiral", "raw_results": {
        "cpu": [_valid_campaign_run(elapsed_wall_sec=float("nan"))]}}
    with pytest.raises(ValueError, match="finite"):
        campaign_rows(nonfinite)
    assert campaign_rows(base)[0]["physical_cpu_cores"] == 2


def test_campaign_rejects_incomplete_steady_samples():
    receipt = {"executable": "pycbc_live", "raw_results": {"cpu": [
        _valid_campaign_run(steady_time_sec=1.0),
        _valid_campaign_run(),
    ]}}
    with pytest.raises(ValueError, match="incomplete steady"):
        campaign_rows(receipt)


def test_campaign_aggregates_latency_and_backlog_without_raw_samples():
    runs = [
        _valid_campaign_run(
            latency={"unit": "seconds", "samples": [1, 3]},
            backlog={"unit": "seconds", "samples": [0, 2]},
            deadline_count=2, deadline_miss_count=1),
        _valid_campaign_run(
            latency={"unit": "seconds", "samples": [5, 7]},
            backlog={"unit": "seconds", "samples": [1, 3]},
            deadline_count=2, deadline_miss_count=0),
    ]
    receipt = {"executable": "pycbc_live", "raw_results": {"cpu": runs}}
    row = campaign_rows(receipt)[0]
    assert row["latency"]["count"] == 4
    assert row["latency"]["p50"] == 4.0
    assert row["latency"]["deadline_miss_count"] == 1
    assert row["latency"]["deadline_count"] == 4
    assert row["latency"]["backlog"]["p95"] == pytest.approx(2.85)
    assert "samples" not in row["latency"]
    assert "samples" not in row["latency"]["backlog"]
    report = render_campaign_report(receipt)
    assert "p50=4" in report and "p95=" in report and "p99=" in report
    assert "misses=1/4" in report and "backlog_p95=" in report


def test_campaign_uses_timed_results_and_excludes_qualification_and_profiles():
    timed = _valid_campaign_run(
        elapsed_wall_sec=4.0,
        completed_template_seconds=400.0,
        science={"passed": True, "scope": "timed"},
    )
    receipt = {
        "executable": "pycbc_live",
        "raw_results": {"cpu": [
            timed,
            _valid_campaign_run(elapsed_wall_sec=0.01,
                                completed_template_seconds=999999,
                                utilization={"samples": [
                                    {"cpu_percent": 100}]},
                                science={"passed": True}),
        ]},
        "qualification_results": {"cpu": [{
            "status": "passed",
            "elapsed_wall_sec": 0.001,
            "completed_template_seconds": 1e12,
            "science": {"passed": True, "scope": "qualification"},
        }]},
    }
    rows = campaign_rows(receipt)
    assert rows[0]["wall_seconds"] == [4.0]
    assert rows[0]["work"] == 400.0
    # Qualification is a gate, but its timing/work is never a capacity sample.
    assert rows[0]["science"]["passed"] is True


def test_campaign_shows_missing_or_failed_qualification_science():
    receipt = {
        "executable": "pycbc_inspiral",
        "raw_results": {"cpu": [_valid_campaign_run()]},
        "qualification_results": {"cpu": [{
            "status": "failed",
            "science": {"passed": False, "missing_gates": ["full PSD"]},
        }]},
    }
    rows = campaign_rows(receipt)
    assert rows[0]["science"]["passed"] is False
    assert "full PSD" in rows[0]["science"]["missing_gates"]
    assert "science failed" in render_campaign_report(receipt)


def test_campaign_uses_allocated_host_cores_in_gpu_label():
    receipt = {"executable": "pycbc_live", "raw_results": {"cuda": [
        _valid_campaign_run(
            resources={"physical_cpu_cores": 8, "allocated_host_cores": 1,
                       "gpus": 1}),
    ]}}
    row = campaign_rows(receipt)[0]
    assert row["physical_cpu_cores"] == 8
    assert row["allocated_host_cores"] == 1
    assert row["resource_label"] == "1 GPU + 1 host core"


def test_campaign_reports_unqualified_arm_that_was_not_timed(tmp_path):
    receipt = {
        "executable": "pycbc_live",
        "raw_results": {"cpu": [], "cuda": [
            _valid_campaign_run(
                science={"passed": True, "scope": "timed"}),
        ]},
        "qualification_results": {"cpu": [{
            "status": "passed",
            "science": {"passed": False,
                        "missing_gates": ["full PSD"]},
        }]},
        "timing_policy": {"skipped_arms": {
            "cpu": {"reason": "qualification_gate_failed"}}},
    }
    rows = campaign_rows(receipt)
    cpu = next(row for row in rows if row["arm"] == "cpu")
    assert cpu["measured"] is False
    assert cpu["wall_capacity"] is None
    assert "not timed" in cpu["skip_reason"]
    assert "full PSD" in cpu["skip_reason"]
    assert "qualification_gate_failed" in cpu["skip_reason"]
    report = render_campaign_report(receipt)
    assert "cpu | not timed | n/a" in report
    output = tmp_path / "skipped.png"
    plot_campaign(receipt, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_campaign_plot_handles_all_arms_skipped(tmp_path):
    receipt = {
        "executable": "pycbc_inspiral",
        "raw_results": {"jax_cpu": []},
        "qualification_results": {"jax_cpu": [{
            "status": "failed",
            "science": {"passed": False,
                        "missing_gates": ["trigger parity"]},
        }]},
    }
    rows = campaign_rows(receipt)
    assert rows[0]["measured"] is False
    report = render_campaign_report(receipt)
    assert "not timed" in report and "trigger parity" in report
    output = tmp_path / "all-skipped.png"
    plot_campaign(receipt, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
