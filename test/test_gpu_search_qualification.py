# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
End-to-end qualification tests for the PyCBC persistent GPU search engine.
Verifies scientific output parity, streaming stability, latency bounds,
and bounded device memory under sustained load.
"""

import os
import subprocess
import sys
import time
import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

from pycbc.types import FrequencySeries
from pycbc.filter.gpu_search import (
    prepare_bank,
    bind_psd,
    SelectionPolicy,
    SearchEngine,
    prepare_power_chisq_plan,
    VetoManager,
    TiledLiveBatchMatchedFilter,
)

CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA_AVAILABLE else [])


def _make_qualification_fixtures(num_templates=8, size=512, tile_size=4, device="cpu"):
    flen = size // 2 + 1
    delta_f = 1.0 / size
    rng = np.random.default_rng(20260910)

    templates = []
    for i in range(num_templates):
        h_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(np.complex64)
        h_vals[0] = 0.0
        pwr = np.sum(np.abs(h_vals) ** 2)
        if pwr > 0:
            h_vals /= np.sqrt(pwr)
        t = FrequencySeries(h_vals, delta_f=delta_f)
        t.id = 100 + i
        templates.append(t)

    bank_plan = prepare_bank(templates, tile_size=tile_size, device=device)

    psd_vals = np.ones(flen, dtype=np.float32) * 2.0
    psd = FrequencySeries(psd_vals, delta_f=delta_f)
    psd_plan = bind_psd(bank_plan, psd, psd_version="psd-v1", device=device)

    # Signal + Gaussian noise data
    data_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(np.complex64)
    data_vals[0] = 0.0
    # Inject template 0 centered at sample index size // 2 with targeted SNR ~ 20.0
    sigmasq0 = float(psd_plan.tile_sigmasqs[0][0])
    target_snr = 20.0
    amp = target_snr / np.sqrt(sigmasq0)
    k = np.arange(flen)
    shift_phases = np.exp(-2j * np.pi * k * (size // 2) / size).astype(np.complex64)
    data_vals += (templates[0].numpy() * shift_phases * amp).astype(np.complex64)
    data = FrequencySeries(data_vals, delta_f=delta_f)
    data.psd = psd

    return bank_plan, psd_plan, data, templates


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_scientific_output_and_veto_parity_cuda():
    """Verify that CUDA graph engine with Power Chisq vetoes achieves exact scientific parity with CPU reference."""
    # 1. CPU Reference Run
    bank_cpu, psd_cpu, data_cpu, _ = _make_qualification_fixtures(
        num_templates=8, size=512, tile_size=4, device="cpu"
    )
    chisq_plan_cpu = prepare_power_chisq_plan(
        bank_cpu, psd_plan=psd_cpu, num_bins=8, device="cpu"
    )
    veto_mgr_cpu = VetoManager(power_chisq_plan=chisq_plan_cpu)
    policy = SelectionPolicy(snr_threshold=4.0, cluster_policy="live_peak")
    valid_interval = (64, 448)

    engine_cpu = SearchEngine(bank_cpu, policy, veto_manager=veto_mgr_cpu, device="cpu")
    engine_cpu.submit(data_cpu, psd_cpu, valid_interval)
    res_cpu = engine_cpu.drain()

    # 2. CUDA Graph Engine Run
    bank_cuda, psd_cuda, data_cuda, _ = _make_qualification_fixtures(
        num_templates=8, size=512, tile_size=4, device="cuda"
    )
    chisq_plan_cuda = prepare_power_chisq_plan(
        bank_cuda, psd_plan=psd_cuda, num_bins=8, device="cuda"
    )
    veto_mgr_cuda = VetoManager(power_chisq_plan=chisq_plan_cuda)

    engine_cuda = SearchEngine(
        bank_cuda,
        policy,
        veto_manager=veto_mgr_cuda,
        device="cuda",
        use_cuda_graphs=True,
        num_workspaces=2,
    )
    engine_cuda.submit(data_cuda, psd_cuda, valid_interval)
    res_cuda = engine_cuda.drain()

    # Verify triggers match
    assert len(res_cpu) == 1 and len(res_cuda) == 1
    cpu_triggers = res_cpu[0].results
    cuda_triggers = res_cuda[0].results
    assert len(cpu_triggers) == len(cuda_triggers)
    assert len(cpu_triggers) > 0
    assert any(len(t.get("sample_idx", [])) > 0 for t in cpu_triggers)

    for r_cpu, r_cuda in zip(cpu_triggers, cuda_triggers):
        np.testing.assert_array_equal(r_cuda["template_id"], r_cpu["template_id"])
        np.testing.assert_array_equal(r_cuda["sample_idx"], r_cpu["sample_idx"])
        np.testing.assert_allclose(r_cuda["snr"], r_cpu["snr"], atol=1e-4)
        if "chisq" in r_cuda and "chisq" in r_cpu:
            np.testing.assert_allclose(r_cuda["chisq"], r_cpu["chisq"], atol=1e-3)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_sustained_streaming_vram_boundedness():
    """Verify zero memory growth and strict latency bounds over sustained streaming blocks."""
    bank_plan, psd_plan, data, _ = _make_qualification_fixtures(
        num_templates=16, size=512, tile_size=8, device="cuda"
    )
    policy = SelectionPolicy(snr_threshold=5.0, cluster_policy="live_peak")
    valid_interval = (64, 448)

    engine = SearchEngine(
        bank_plan,
        policy,
        device="cuda",
        use_cuda_graphs=True,
        num_workspaces=2,
        enable_async_transfers=True,
    )

    # Warmup
    for _ in range(5):
        engine.submit(data, psd_plan, valid_interval)
        engine.drain()
    torch.cuda.synchronize()

    initial_allocated = torch.cuda.memory_allocated()

    # Run 40 consecutive blocks
    times = []
    for i in range(40):
        t0 = time.perf_counter()
        engine.submit(data, psd_plan, valid_interval, block_id=i)
        ready = engine.drain()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        assert len(ready) == 1

    final_allocated = torch.cuda.memory_allocated()
    engine.close()

    # Verify zero memory leak
    assert final_allocated <= initial_allocated + 4096  # within small page rounding
    # Verify latency meets < 50ms requirement for 512-point batches on RTX 4090
    p99_latency = np.percentile(times, 99)
    assert p99_latency < 50.0  # ms


@pytest.mark.parametrize("device", DEVICES)
def test_live_adapter_sustained_workflow(device):
    """Verify TiledLiveBatchMatchedFilter over sequential streaming chunks."""
    bank_plan, psd_plan, data, templates = _make_qualification_fixtures(
        num_templates=6, size=256, tile_size=2, device=device
    )

    import types
    data_reader = types.SimpleNamespace(
        overwhitened_data=lambda _df: data,
        trim_padding=16,
        blocksize=128,
        sample_rate=1,
        start_time=10000.0,
    )

    live_filter = TiledLiveBatchMatchedFilter(
        templates=templates,
        snr_threshold=0.0,
        chisq_bins=8,
        sg_chisq=types.SimpleNamespace(do=False),
        tile_size=2,
        device=device,
        use_cuda_graphs=(device == "cuda"),
    )

    res = live_filter.process_data(data_reader)
    assert res is not False
    assert len(res["template_id"]) == 6
    assert len(res["snr"]) == 6


def test_qualification_receipt_metrics():
    """Verify sealed qualification receipt conforms to production constraints."""
    import json
    from pathlib import Path
    from tools.benchmarking.benchmark_gpu_search import validate_qualification_receipt

    receipt_path = (
        Path(__file__).resolve().parent.parent
        / "artifacts"
        / "gpu_search_qualification_receipt.json"
    )
    if not receipt_path.exists():
        pytest.skip("Receipt file not found")

    with open(receipt_path, "r") as f:
        receipt = json.load(f)

    benchmarks = receipt.get("benchmarks", {})
    insp = benchmarks.get("production_inspiral_workload", {})

    # Production qualification strictly requires full workload validation (no fallback)
    assert validate_qualification_receipt(receipt, require_full_workload=True) is True

    assert "gpu_tile_scaling" in benchmarks
    assert "cpu_tile_scaling" in benchmarks
    assert "cuda_graph_comparison" in benchmarks
    assert "live_streaming_latency" in benchmarks

    # Live streaming latency
    live = benchmarks["live_streaming_latency"]
    assert live["latency_p99_ms"] < 50.0
    assert live["vram_leak_detected"] is False
    assert np.isfinite(live["latency_p99_ms"])

    # CUDA graphs
    graph = benchmarks["cuda_graph_comparison"]
    assert graph["speedup"] > 1.0
    assert np.isfinite(graph["speedup"])

    # Production inspiral workload: 384 templates * 5 segments = 1,920 comparisons
    assert "production_inspiral_workload" in benchmarks
    assert insp["num_templates"] == 384
    assert insp["num_segments"] == 5
    assert insp["total_matched_filters"] == 1920
    assert insp["cpu_comparison_count"] == 1920
    assert (
        insp["cpu_comparison_count"]
        == insp["total_matched_filters"]
        == insp["num_templates"] * insp["num_segments"]
    )
    assert insp["transform_length"] == 2097152
    assert insp["calc_templates_per_sec"] > 1000.0
    assert insp["vram_budget_bounded"] is True
    assert insp["injection_recovered"] is True
    assert insp["cpu_reference_parity"] is True
    assert insp["cpu_peak_time_diff"] == 0
    assert insp["cpu_max_snr_diff"] < 1e-3
    assert insp["cpu_max_chisq_diff"] < 0.05
    assert insp["gpu_cand_count"] > 0

    # Ensure cpu_comparison_count is explicitly present and strictly == 1920
    assert "cpu_comparison_count" in insp
    assert isinstance(insp["cpu_comparison_count"], (int, np.integer))
    assert insp["cpu_comparison_count"] == 1920

    # Require all numeric fields to be finite
    for k, v in insp.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            assert np.isfinite(v), f"Metric {k} is not finite: {v}"

    if "waveform_approximant" in insp:
        assert insp["waveform_approximant"] == "TaylorF2"
        assert insp["psd_model"] == "aLIGOZeroDetHighPower"
        assert insp["distinct_segments"] is True
        assert insp["veto_type"] == "PowerChisq"
        assert insp["duration_sec"] == 512.0
        assert insp["sample_rate_hz"] == 4096.0
        assert insp["delta_f"] == 1.0 / 512.0


def test_qualification_receipt_rejections():
    """Verify validator rejects receipts with partial evidence, missing/invalid dimensions, and non-finite values."""
    import copy
    from tools.benchmarking.benchmark_gpu_search import validate_qualification_receipt

    base_receipt = {
        "benchmarks": {
            "production_inspiral_workload": {
                "num_templates": 384,
                "num_segments": 5,
                "total_matched_filters": 1920,
                "cpu_comparison_count": 1920,
                "cpu_reference_parity": True,
                "injection_recovered": True,
                "cpu_peak_time_diff": 0,
                "cpu_max_snr_diff": 1e-5,
                "cpu_max_chisq_diff": 1e-4,
                "calc_templates_per_sec": 2000.0,
            }
        }
    }

    # Baseline full workload passes
    assert validate_qualification_receipt(base_receipt) is True

    # Partial count = 5 rejected by default in production qualification
    rec = copy.deepcopy(base_receipt)
    rec["benchmarks"]["production_inspiral_workload"]["cpu_comparison_count"] = 5
    with pytest.raises(ValueError, match="does not match expected total matched filters"):
        validate_qualification_receipt(rec)

    # Partial count = 2 rejected by default in production qualification
    rec = copy.deepcopy(base_receipt)
    rec["benchmarks"]["production_inspiral_workload"]["cpu_comparison_count"] = 2
    with pytest.raises(ValueError, match="does not match expected total matched filters"):
        validate_qualification_receipt(rec)

    # Historical review mode allows partial counts > 1
    assert validate_qualification_receipt(rec, require_full_workload=False) is True

    # Completely missing production_inspiral_workload benchmark rejected in production mode
    rec_no_prod = {"benchmarks": {"live_streaming_latency": {"latency_p50_ms": 3.0}}}
    with pytest.raises(ValueError, match="Missing required 'production_inspiral_workload'"):
        validate_qualification_receipt(rec_no_prod, require_full_workload=True)

    # Missing workload dimensions
    for dim in ["num_templates", "num_segments", "total_matched_filters", "cpu_comparison_count"]:
        rec = copy.deepcopy(base_receipt)
        del rec["benchmarks"]["production_inspiral_workload"][dim]
        with pytest.raises(ValueError, match=f"Missing required workload dimension '{dim}'"):
            validate_qualification_receipt(rec)

    # Dimension type validation (reject floats, booleans, and np.bool_)
    for bad_dim, bad_val in [
        ("num_templates", 384.5),
        ("num_segments", True),
        ("num_templates", np.bool_(True)),
        ("num_segments", np.bool_(False)),
        ("cpu_comparison_count", "1920"),
    ]:
        rec = copy.deepcopy(base_receipt)
        rec["benchmarks"]["production_inspiral_workload"][bad_dim] = bad_val
        with pytest.raises(ValueError, match=f"Workload dimension '{bad_dim}' must be an integer"):
            validate_qualification_receipt(rec)

    # Dimension product mismatch
    rec = copy.deepcopy(base_receipt)
    rec["benchmarks"]["production_inspiral_workload"]["total_matched_filters"] = 1900
    with pytest.raises(ValueError, match="total_matched_filters .* != num_templates"):
        validate_qualification_receipt(rec)

    # Non-384x5 dimensions in production qualification
    rec = copy.deepcopy(base_receipt)
    rec["benchmarks"]["production_inspiral_workload"]["num_templates"] = 100
    rec["benchmarks"]["production_inspiral_workload"]["num_segments"] = 5
    rec["benchmarks"]["production_inspiral_workload"]["total_matched_filters"] = 500
    rec["benchmarks"]["production_inspiral_workload"]["cpu_comparison_count"] = 500
    with pytest.raises(ValueError, match="Production qualification requires 384 templates x 5 segments = 1,920"):
        validate_qualification_receipt(rec, require_full_workload=True)

    # But passes under historical review mode
    assert validate_qualification_receipt(rec, require_full_workload=False) is True

    # count = 1 rejected even with require_full_workload=False
    rec = copy.deepcopy(base_receipt)
    rec["benchmarks"]["production_inspiral_workload"]["cpu_comparison_count"] = 1
    with pytest.raises(ValueError, match="cpu_comparison_count must be an integer > 1"):
        validate_qualification_receipt(rec, require_full_workload=False)

    # NaN in cpu_max_chisq_diff rejected
    rec = copy.deepcopy(base_receipt)
    rec["benchmarks"]["production_inspiral_workload"]["cpu_max_chisq_diff"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        validate_qualification_receipt(rec)

    # Inf in cpu_max_snr_diff rejected
    rec = copy.deepcopy(base_receipt)
    rec["benchmarks"]["production_inspiral_workload"]["cpu_max_snr_diff"] = float("inf")
    with pytest.raises(ValueError, match="must be finite"):
        validate_qualification_receipt(rec)

    # Peak time diff != 0 rejected
    rec = copy.deepcopy(base_receipt)
    rec["benchmarks"]["production_inspiral_workload"]["cpu_peak_time_diff"] = 1
    with pytest.raises(ValueError, match="cpu_peak_time_diff mismatch"):
        validate_qualification_receipt(rec)


def test_canonical_cpu_reference_template_chisq_positive():
    """Verify compute_canonical_cpu_reference_template computes accurate, positive physical chisq."""
    from pycbc.filter.matchedfilter import sigmasq
    from pycbc.vetoes.chisq import power_chisq_bins
    from tools.benchmarking.benchmark_gpu_search import (
        compute_canonical_cpu_reference_template,
    )

    size = 512
    flen = size // 2 + 1
    delta_f = 1.0
    rng = np.random.default_rng(20260910)

    # White template
    h_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(
        np.complex64
    )
    h_vals[0] = 0.0
    pwr = np.sum(np.abs(h_vals) ** 2)
    if pwr > 0:
        h_vals /= np.sqrt(pwr)
    tmpl0 = FrequencySeries(h_vals, delta_f=delta_f)
    tmpl0.id = 0

    psd_vals = np.ones(flen, dtype=np.float32) * 2.0
    psd = FrequencySeries(psd_vals, delta_f=delta_f)

    # Signal + Gaussian noise data
    data_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(
        np.complex64
    )
    data_vals[0] = 0.0
    f_lower = 20.0
    f_upper = 240.0
    s_ref = float(sigmasq(tmpl0, psd, f_lower, f_upper))

    target_snr = 20.0
    amp = target_snr / np.sqrt(s_ref)
    k = np.arange(flen)
    shift_phases = np.exp(-2j * np.pi * k * (size // 2) / size).astype(
        np.complex64
    )
    data_vals += (tmpl0.numpy() * shift_phases * amp).astype(np.complex64)
    data = FrequencySeries(data_vals, delta_f=delta_f)
    data.psd = psd

    chisq_bins = power_chisq_bins(tmpl0, 8, psd, f_lower, f_upper)
    valid_interval = (64, 448)
    policy = SelectionPolicy(snr_threshold=4.0, cluster_policy="live_peak")

    cands = compute_canonical_cpu_reference_template(
        tmpl0,
        data,
        psd,
        sigmasq=s_ref,
        f_lower=f_lower,
        f_upper=f_upper,
        chisq_bins=chisq_bins,
        valid_interval=valid_interval,
        policy=policy,
        segment_idx=0,
        template_idx=0,
    )
    assert len(cands) > 0
    top = max(cands, key=lambda c: abs(c["snr"]))
    # SNR should be near injected 20.0 at sample index 256
    assert abs(top["sample_idx"] - 256) <= 2
    assert abs(top["snr"]) > 18.0
    # Chisq must be positive and physical (reduced chisq ~ 0.5 to 2.5, NOT negative like -4373)
    assert top["chisq"] > 0.0
    assert top["red_chisq"] > 0.0
    assert top["red_chisq"] < 5.0
    assert np.isfinite(top["chisq"])
    assert np.isfinite(top["red_chisq"])


def test_extract_and_validate_gpu_candidates_rejections():
    """Verify extract_and_validate_gpu_candidates strictly rejects malformed or unexpected outputs."""
    from tools.benchmarking.benchmark_gpu_search import (
        extract_and_validate_gpu_candidates,
    )

    class DummyBatch:
        def __init__(self, r):
            self.results = [r]
            self.overflow = False
            self.aborted = False

    valid_res = {
        "template_id": np.array([0, 1], dtype=np.int64),
        "sample_idx": np.array([100, 200], dtype=np.int64),
        "snr": np.array([6.0 + 1j, 7.0 + 2j], dtype=np.complex64),
        "chisq": np.array([14.0, 15.0], dtype=np.float32),
        "chisq_dof": np.array([14, 14], dtype=np.int32),
    }

    # Baseline valid output passes
    extracted = extract_and_validate_gpu_candidates(
        DummyBatch(valid_res), num_templates=2
    )
    assert len(extracted[0]) == 1
    assert len(extracted[1]) == 1

    # 1. Missing chisq field
    bad = dict(valid_res)
    del bad["chisq"]
    with pytest.raises(ValueError, match="Missing required field 'chisq'"):
        extract_and_validate_gpu_candidates(DummyBatch(bad), num_templates=2)

    # 2. Missing chisq_dof field (must not synthesize)
    bad = dict(valid_res)
    del bad["chisq_dof"]
    with pytest.raises(ValueError, match="Missing required field 'chisq_dof'"):
        extract_and_validate_gpu_candidates(DummyBatch(bad), num_templates=2)

    # 3. Missing template_id, sample_idx, or snr
    for missing_key in ["template_id", "sample_idx", "snr"]:
        bad = dict(valid_res)
        del bad[missing_key]
        with pytest.raises(ValueError, match=f"Missing required field '{missing_key}'"):
            extract_and_validate_gpu_candidates(
                DummyBatch(bad), num_templates=2
            )

    # 4. Inconsistent array lengths
    bad = dict(valid_res)
    bad["snr"] = np.array([6.0 + 1j], dtype=np.complex64)
    with pytest.raises(ValueError, match="Inconsistent candidate field length"):
        extract_and_validate_gpu_candidates(DummyBatch(bad), num_templates=2)

    # 5. Unexpected template_id (out of bounds)
    bad = dict(valid_res)
    bad["template_id"] = np.array([0, 999], dtype=np.int64)
    with pytest.raises(ValueError, match="Unexpected template_id"):
        extract_and_validate_gpu_candidates(DummyBatch(bad), num_templates=2)

    # 6. NaN template_id
    bad = dict(valid_res)
    bad["template_id"] = np.array([0.0, float("nan")])
    with pytest.raises(ValueError, match="Non-finite template_id"):
        extract_and_validate_gpu_candidates(DummyBatch(bad), num_templates=2)

    # 7. Non-finite SNR
    bad = dict(valid_res)
    bad["snr"] = np.array([float("nan") + 1j, 7.0 + 2j], dtype=np.complex64)
    with pytest.raises(ValueError, match="Non-finite GPU SNR"):
        extract_and_validate_gpu_candidates(DummyBatch(bad), num_templates=2)

    # 8. Non-finite or non-positive DOF
    bad = dict(valid_res)
    bad["chisq_dof"] = np.array([0, 14], dtype=np.int32)
    with pytest.raises(ValueError, match="Non-positive GPU chisq_dof"):
        extract_and_validate_gpu_candidates(DummyBatch(bad), num_templates=2)

    # 9. Non-finite chisq
    bad = dict(valid_res)
    bad["chisq"] = np.array([float("nan"), 15.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Non-finite GPU chisq"):
        extract_and_validate_gpu_candidates(DummyBatch(bad), num_templates=2)

    # 10. Duplicate sample indices for the same template
    bad = {
        "template_id": np.array([0, 0], dtype=np.int64),
        "sample_idx": np.array([100, 100], dtype=np.int64),
        "snr": np.array([6.0 + 1j, 6.1 + 1j], dtype=np.complex64),
        "chisq": np.array([14.0, 14.1], dtype=np.float32),
        "chisq_dof": np.array([14, 14], dtype=np.int32),
    }
    with pytest.raises(ValueError, match="duplicate sample indices"):
        extract_and_validate_gpu_candidates(DummyBatch(bad), num_templates=2)


def test_compare_template_candidates_rejections():
    """Verify compare_template_candidates strictly enforces parity and catches mismatches."""
    from tools.benchmarking.benchmark_gpu_search import (
        compare_template_candidates,
    )

    base_cand = {
        "sample_idx": 100,
        "snr": 10.0 + 0j,
        "chisq": 14.0,
        "chisq_dof": 14,
        "red_chisq": 1.0,
    }

    # Baseline match passes
    res = compare_template_candidates([base_cand], [base_cand])
    assert res["matched"] is True
    assert res["max_snr_diff"] == 0.0
    assert res["max_chisq_diff"] == 0.0

    # Both empty passes
    res = compare_template_candidates([], [])
    assert res["matched"] is True

    # Missing GPU trigger
    with pytest.raises(RuntimeError, match="missing GPU trigger"):
        compare_template_candidates([], [base_cand])

    # Spurious GPU trigger
    with pytest.raises(RuntimeError, match="spurious GPU trigger"):
        compare_template_candidates([base_cand], [])

    # Candidate count mismatch
    with pytest.raises(RuntimeError, match="trigger count mismatch"):
        compare_template_candidates(
            [base_cand, dict(base_cand, sample_idx=200)], [base_cand]
        )

    # Sample position mismatch
    with pytest.raises(RuntimeError, match="sample position mismatch"):
        compare_template_candidates(
            [dict(base_cand, sample_idx=105)], [base_cand]
        )

    # SNR mismatch
    with pytest.raises(RuntimeError, match="SNR mismatch"):
        compare_template_candidates(
            [dict(base_cand, snr=10.1 + 0j)], [base_cand]
        )

    # Chisq DOF mismatch
    with pytest.raises(RuntimeError, match="chisq DOF mismatch"):
        compare_template_candidates(
            [dict(base_cand, chisq_dof=28)], [base_cand]
        )

    # Chisq mismatch
    with pytest.raises(RuntimeError, match="chisq mismatch"):
        compare_template_candidates(
            [dict(base_cand, red_chisq=1.2)], [base_cand]
        )

    # Duplicate sample index
    with pytest.raises(ValueError, match="duplicate sample indices"):
        compare_template_candidates(
            [base_cand, base_cand], [base_cand, base_cand]
        )



@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_production_inspiral_scientific_parity():
    """Verify scientific parity between CPU and CUDA GPU search on physical templates and colored PSD across full bank."""
    size = 2048
    flen = size // 2 + 1
    delta_f = 1.0
    num_templates = 8
    tile_size = 4
    f_lower = 20.0
    f_upper = 500.0

    # Generate templates
    m1_vals = np.linspace(10.0, 30.0, num_templates)
    m2_vals = np.linspace(1.4, 15.0, num_templates)
    templates = []
    for i in range(num_templates):
        M = m1_vals[i] + m2_vals[i]
        eta = (m1_vals[i] * m2_vals[i]) / (M**2)
        M_sec = M * 4.925491025543576e-6
        freqs = np.linspace(0, (flen - 1) * delta_f, flen, dtype=np.float64)
        kmin = max(1, int(f_lower / delta_f))
        kmax = min(flen - 1, int(f_upper / delta_f))
        v = (np.pi * M_sec * np.maximum(freqs[kmin:kmax], 1e-4)) ** (1.0 / 3.0)
        psi = (3.0 / (128.0 * eta * (v**5))) * (
            1.0 + (3715.0 / 756.0 + 55.0 / 9.0 * eta) * (v**2)
        )
        amp = (freqs[kmin:kmax] ** (-7.0 / 6.0)).astype(np.float32)
        phase = psi - np.pi / 4.0
        hp = np.zeros(flen, dtype=np.complex64)
        hp[kmin:kmax] = amp * (np.cos(phase) - 1j * np.sin(phase))
        pwr = float(np.sum(np.abs(hp) ** 2))
        if pwr > 0:
            hp /= np.sqrt(pwr)
        t = FrequencySeries(hp, delta_f=delta_f)
        t.id = i
        t.params = type(
            "Params",
            (),
            {"template_hash": i, "mass1": m1_vals[i], "mass2": m2_vals[i]},
        )()
        templates.append(t)

    # Colored PSD with physical scale ~ 1e-47 rescaled by DYN_RANGE_FAC**2
    from pycbc import DYN_RANGE_FAC
    from pycbc.filter.matchedfilter import matched_filter_core
    from pycbc.vetoes.chisq import power_chisq_bins, power_chisq_at_points_from_precomputed

    freqs = np.linspace(0, (flen - 1) * delta_f, flen, dtype=np.float64)
    x = np.maximum(freqs / 100.0, 0.1)
    psd_vals = (1.52e-47 * (2.0 + 5.0 * (x ** (-2)))).astype(np.float64)
    kmin = int(f_lower / delta_f)
    psd_vals[:kmin] = 0.0
    # Scaled PSD for single precision processing
    psd = FrequencySeries(
        (psd_vals * (float(DYN_RANGE_FAC) ** 2)).astype(np.float32),
        delta_f=delta_f,
    )
    psd.dyn_range_factor = float(DYN_RANGE_FAC)

    # Distinct colored strain segment with injected signal
    rng = np.random.default_rng(20260910)
    white = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(
        np.complex64
    )
    stilde_vals = (white * np.sqrt(psd.numpy() / (4.0 * delta_f))).astype(
        np.complex64
    )
    stilde_vals[0] = 0.0
    k_vec = np.arange(flen)
    shift = np.exp(-2j * np.pi * k_vec * (size // 2) / size).astype(
        np.complex64
    )
    sigmasq_ref = float(
        np.sum(
            (np.abs(templates[0].numpy()) ** 2)
            * (4.0 * delta_f / np.maximum(psd.numpy(), 1e-10))
        )
    )
    amp = 20.0 / np.sqrt(max(sigmasq_ref, 1e-10))
    stilde_vals += (templates[0].numpy() * shift * amp).astype(np.complex64)
    stilde = FrequencySeries(stilde_vals.astype(np.complex64), delta_f=delta_f)
    stilde.psd = psd

    valid_interval = (size // 4, size * 3 // 4)
    policy = SelectionPolicy(
        snr_threshold=5.0, cluster_policy="symmetric", cluster_window=10
    )

    # Run SearchEngine (on cuda if available, else cpu)
    test_device = "cuda" if CUDA_AVAILABLE else "cpu"
    bank_plan = prepare_bank(
        templates,
        tile_size=tile_size,
        f_lower=f_lower,
        f_upper=f_upper,
        device=test_device,
    )
    # Test that bind_psd correctly handles both physical unscaled and pre-scaled PSD
    psd_plan = bind_psd(bank_plan, psd, device=test_device)
    chisq_plan = prepare_power_chisq_plan(
        bank_plan, psd_plan, num_bins=8, device=test_device
    )
    veto_mgr = VetoManager(power_chisq_plan=chisq_plan)
    engine = SearchEngine(
        bank_plan, policy, veto_manager=veto_mgr, device=test_device
    )
    engine.submit(stilde, psd_plan, valid_interval, block_id=0)
    ready = engine.drain()
    engine.close()

    assert len(ready) == 1
    tile_res = ready[0].results
    assert len(tile_res) > 0

    # Verify ALL templates in the bank against independent canonical CPU references
    v_start, v_stop = valid_interval
    for t_idx in range(num_templates):
        ref_snr_series, ref_corr, ref_norm = matched_filter_core(
            templates[t_idx],
            stilde,
            psd=psd,
            low_frequency_cutoff=f_lower,
            high_frequency_cutoff=f_upper,
        )
        ref_abs = np.abs(np.asarray(ref_snr_series[v_start:v_stop])) * float(ref_norm)
        ref_peak_offset = int(np.argmax(ref_abs))
        ref_peak_idx = v_start + ref_peak_offset
        ref_peak_snr = float(ref_abs[ref_peak_offset])
        assert np.isfinite(ref_peak_snr)

        # Collect any engine triggers for t_idx across all tiles
        cand_indices = []
        cand_snrs = []
        cand_chisqs = []
        cand_dofs = []
        for t_res in tile_res:
            mask = (t_res["template_id"] == t_idx)
            if np.any(mask):
                cand_indices.extend(t_res["sample_idx"][mask].tolist())
                cand_snrs.extend(np.abs(t_res["snr"][mask]).tolist())
                cand_chisqs.extend(t_res["chisq"][mask].tolist())
                cand_dofs.extend(t_res["chisq_dof"][mask].tolist())

        if ref_peak_snr >= 5.0:
            # Template crosses threshold (template 0 injection or template overlap)
            assert len(cand_indices) > 0, f"Template {t_idx} above threshold ({ref_peak_snr:.2f}) not recovered by engine"
            best_cand_idx = int(np.argmax(cand_snrs))
            engine_peak_idx = int(cand_indices[best_cand_idx])
            engine_peak_snr = float(cand_snrs[best_cand_idx])
            engine_chisq = float(cand_chisqs[best_cand_idx])
            engine_dof = int(cand_dofs[best_cand_idx])

            ref_bins = power_chisq_bins(templates[t_idx], 8, psd, f_lower, f_upper)
            ref_chisq_raw = power_chisq_at_points_from_precomputed(
                ref_corr,
                np.asarray([ref_snr_series[ref_peak_idx]]),
                ref_norm,
                ref_bins,
                np.array([ref_peak_idx], dtype=np.int64),
            )
            ref_dof = 2 * 8 - 2
            ref_red_chisq = float(ref_chisq_raw[0] / ref_dof)

            assert np.isfinite(engine_peak_snr)
            assert np.isfinite(engine_chisq)
            assert np.isfinite(ref_red_chisq)
            assert engine_peak_idx == ref_peak_idx
            assert engine_dof == ref_dof
            np.testing.assert_allclose(engine_peak_snr, ref_peak_snr, atol=1e-3)
            engine_red_chisq = engine_chisq / engine_dof
            np.testing.assert_allclose(engine_red_chisq, ref_red_chisq, atol=0.02)
            np.testing.assert_allclose(engine_chisq, float(ref_chisq_raw[0]), atol=0.05)
        else:
            # Quiet template: verify absence of spurious triggers
            assert len(cand_indices) == 0, (
                f"Spurious triggers found for quiet template {t_idx}: {cand_indices}"
            )


def test_save_qualification_receipt_atomic_and_prevalidation(tmp_path):
    """Verify save_qualification_receipt strictly validates before writing and writes atomically."""
    import copy
    import json
    from tools.benchmarking.benchmark_gpu_search import save_qualification_receipt

    valid_report = {
        "benchmarks": {
            "production_inspiral_workload": {
                "num_templates": 384,
                "num_segments": 5,
                "total_matched_filters": 1920,
                "cpu_comparison_count": 1920,
                "cpu_reference_parity": True,
                "injection_recovered": True,
                "cpu_peak_time_diff": 0,
                "cpu_max_snr_diff": 1e-5,
                "cpu_max_chisq_diff": 1e-4,
                "calc_templates_per_sec": 2000.0,
            }
        }
    }

    out_file = tmp_path / "receipt.json"

    # 1. Invalid report must fail before file creation
    invalid_report = copy.deepcopy(valid_report)
    invalid_report["benchmarks"]["production_inspiral_workload"]["cpu_comparison_count"] = 5
    with pytest.raises(ValueError, match="does not match expected total matched filters"):
        save_qualification_receipt(invalid_report, out_file, require_full_workload=True)
    assert not out_file.exists(), "Target file must not be created on validation failure"

    # 2. Valid report writes successfully
    save_qualification_receipt(valid_report, out_file, require_full_workload=True)
    assert out_file.exists()
    with open(out_file, "r") as f:
        loaded = json.load(f)
    assert loaded["benchmarks"]["production_inspiral_workload"]["cpu_comparison_count"] == 1920

    # 3. Invalid report must fail without overwriting existing valid receipt
    invalid_report2 = copy.deepcopy(valid_report)
    invalid_report2["benchmarks"]["production_inspiral_workload"]["cpu_max_snr_diff"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        save_qualification_receipt(invalid_report2, out_file, require_full_workload=True)

    # 4. Missing production section when require_full_workload=True must fail without overwriting
    invalid_report3 = copy.deepcopy(valid_report)
    invalid_report3["benchmarks"] = {"cpu_tile_scaling": {"duration_sec": 1.0}}
    with pytest.raises(ValueError, match="Missing required 'production_inspiral_workload'"):
        save_qualification_receipt(invalid_report3, out_file, require_full_workload=True)

    # 5. Non-integer or boolean dimension values must fail without overwriting
    invalid_report4 = copy.deepcopy(valid_report)
    invalid_report4["benchmarks"]["production_inspiral_workload"]["num_templates"] = np.bool_(True)
    with pytest.raises(ValueError, match="must be an integer"):
        save_qualification_receipt(invalid_report4, out_file, require_full_workload=True)

    # Verify original file content is intact and was not corrupted/overwritten
    with open(out_file, "r") as f:
        loaded_after = json.load(f)
    assert loaded_after["benchmarks"]["production_inspiral_workload"]["cpu_comparison_count"] == 1920
    assert loaded_after["benchmarks"]["production_inspiral_workload"]["cpu_max_snr_diff"] == 1e-5


def test_main_include_production_inspiral_cuda_unavailable_preserves_receipt(tmp_path, monkeypatch):
    """Verify main() fails immediately and preserves existing receipt when --include-production-inspiral is requested on CPU."""
    from tools.benchmarking import benchmark_gpu_search
    import json

    out_file = tmp_path / "existing_receipt.json"
    initial_content = {"retained_marker": "valid_receipt_unmodified", "benchmarks": {}}
    with open(out_file, "w") as f:
        json.dump(initial_content, f)

    if benchmark_gpu_search.torch is not None:
        monkeypatch.setattr(benchmark_gpu_search.torch.cuda, "is_available", lambda: False)
    else:
        monkeypatch.setattr(benchmark_gpu_search, "torch", None)

    with pytest.raises(RuntimeError, match="Requested workload '--include-production-inspiral' cannot execute"):
        benchmark_gpu_search.main(["--include-production-inspiral", "--output", str(out_file)])

    with open(out_file, "r") as f:
        loaded = json.load(f)
    assert loaded == initial_content, "Existing receipt must be preserved without modification"


def test_candidate_integer_validation_fractional_fields():
    """Verify that candidate sample_idx, chisq_dof, and template_id reject fractional values like 100.9 and 14.9."""
    from tools.benchmarking.benchmark_gpu_search import (
        extract_and_validate_gpu_candidates,
        compare_template_candidates,
        _validate_integer_field,
    )

    class DummyBatch:
        def __init__(self, r):
            self.results = [r]
            self.overflow = False
            self.aborted = False

    base_res = {
        "template_id": np.array([0], dtype=np.int64),
        "sample_idx": np.array([100], dtype=np.int64),
        "snr": np.array([6.0 + 1j], dtype=np.complex64),
        "chisq": np.array([14.0], dtype=np.float32),
        "chisq_dof": np.array([14], dtype=np.int32),
    }

    # 1. extract_and_validate_gpu_candidates rejects fractional sample_idx (e.g. 100.9)
    bad_sample = dict(base_res, sample_idx=np.array([100.9]))
    with pytest.raises(ValueError, match="Fractional sample_idx detected"):
        extract_and_validate_gpu_candidates(DummyBatch(bad_sample), num_templates=1)

    # 2. extract_and_validate_gpu_candidates rejects fractional chisq_dof (e.g. 14.9)
    bad_dof = dict(base_res, chisq_dof=np.array([14.9]))
    with pytest.raises(ValueError, match="Fractional chisq_dof detected"):
        extract_and_validate_gpu_candidates(DummyBatch(bad_dof), num_templates=1)

    # 3. extract_and_validate_gpu_candidates rejects fractional template_id (e.g. 0.5)
    bad_tid = dict(base_res, template_id=np.array([0.5]))
    with pytest.raises(ValueError, match="Fractional template_id detected"):
        extract_and_validate_gpu_candidates(DummyBatch(bad_tid), num_templates=1)

    # 4. Integer-valued floats (100.0, 14.0) are accepted and converted to int
    float_int_res = dict(
        base_res,
        sample_idx=np.array([100.0]),
        chisq_dof=np.array([14.0]),
    )
    cands = extract_and_validate_gpu_candidates(DummyBatch(float_int_res), num_templates=1)
    cand = cands[0][0]
    assert cand["sample_idx"] == 100 and isinstance(cand["sample_idx"], int)
    assert cand["chisq_dof"] == 14 and isinstance(cand["chisq_dof"], int)

    # 5. Reject scalar bool and np.bool_ for template_id, sample_idx, chisq_dof
    for bad_bool in [True, False, np.bool_(True), np.bool_(False)]:
        with pytest.raises(ValueError, match="Boolean value not allowed for sample_idx"):
            _validate_integer_field(bad_bool, "sample_idx")
        with pytest.raises(ValueError, match="Boolean value not allowed for chisq_dof"):
            _validate_integer_field(bad_bool, "chisq_dof")
        with pytest.raises(ValueError, match="Boolean value not allowed for template_id"):
            _validate_integer_field(bad_bool, "template_id")

    # 6. Reject numpy boolean arrays in extract_and_validate_gpu_candidates
    for bool_arr in [np.array([True]), np.array([False], dtype=np.bool_), np.array([True], dtype=bool)]:
        bad_sample_arr = dict(base_res, sample_idx=bool_arr)
        with pytest.raises(ValueError, match="Boolean array not allowed for 'sample_idx'"):
            extract_and_validate_gpu_candidates(DummyBatch(bad_sample_arr), num_templates=1)

        bad_dof_arr = dict(base_res, chisq_dof=bool_arr)
        with pytest.raises(ValueError, match="Boolean array not allowed for 'chisq_dof'"):
            extract_and_validate_gpu_candidates(DummyBatch(bad_dof_arr), num_templates=1)

        bad_tid_arr = dict(base_res, template_id=bool_arr)
        with pytest.raises(ValueError, match="Boolean array not allowed for 'template_id'"):
            extract_and_validate_gpu_candidates(DummyBatch(bad_tid_arr), num_templates=1)

    # 7. compare_template_candidates rejects fractional and boolean candidate fields
    cand_valid = {
        "sample_idx": 100,
        "snr": 10.0 + 0j,
        "chisq": 14.0,
        "chisq_dof": 14,
        "red_chisq": 1.0,
    }
    cand_bad_sample = dict(cand_valid, sample_idx=100.9)
    with pytest.raises(ValueError, match="Fractional sample_idx detected"):
        compare_template_candidates([cand_bad_sample], [cand_valid])
    with pytest.raises(ValueError, match="Fractional sample_idx detected"):
        compare_template_candidates([cand_valid], [cand_bad_sample])

    cand_bad_dof = dict(cand_valid, chisq_dof=14.9)
    with pytest.raises(ValueError, match="Fractional chisq_dof detected"):
        compare_template_candidates([cand_bad_dof], [cand_valid])
    with pytest.raises(ValueError, match="Fractional chisq_dof detected"):
        compare_template_candidates([cand_valid], [cand_bad_dof])

    for bad_bool in [True, False, np.bool_(True), np.bool_(False)]:
        cand_bool_sample = dict(cand_valid, sample_idx=bad_bool)
        with pytest.raises(ValueError, match="Boolean value not allowed for sample_idx"):
            compare_template_candidates([cand_bool_sample], [cand_valid])
        cand_bool_dof = dict(cand_valid, chisq_dof=bad_bool)
        with pytest.raises(ValueError, match="Boolean value not allowed for chisq_dof"):
            compare_template_candidates([cand_bool_dof], [cand_valid])


def test_historical_receipt_inspection_separation():
    """Verify inspect_historical_receipt inspects historical receipts without accepting them as production qualified."""
    from tools.benchmarking.benchmark_gpu_search import inspect_historical_receipt

    partial_receipt = {
        "provenance": {"git_commit": "abc1234"},
        "benchmarks": {
            "production_inspiral_workload": {
                "num_templates": 384,
                "num_segments": 5,
                "total_matched_filters": 1920,
                "cpu_comparison_count": 5,
                "cpu_reference_parity": True,
                "injection_recovered": True,
                "cpu_peak_time_diff": 0,
                "cpu_max_snr_diff": 1e-5,
                "cpu_max_chisq_diff": 1e-4,
                "calc_templates_per_sec": 2000.0,
            }
        },
    }

    # Historical inspection passes but clearly reports non-qualification
    res = inspect_historical_receipt(partial_receipt)
    assert res["valid_historical_receipt"] is True
    assert res["is_production_qualified"] is False
    assert res["cpu_comparison_count"] == 5
    assert res["total_matched_filters"] == 1920
    assert res["provenance_recorded"] is True

    # Full workload receipt is marked production qualified
    full_receipt = {
        "benchmarks": {
            "production_inspiral_workload": {
                "num_templates": 384,
                "num_segments": 5,
                "total_matched_filters": 1920,
                "cpu_comparison_count": 1920,
                "cpu_reference_parity": True,
                "injection_recovered": True,
                "cpu_peak_time_diff": 0,
                "cpu_max_snr_diff": 1e-5,
                "cpu_max_chisq_diff": 1e-4,
                "calc_templates_per_sec": 2000.0,
            }
        }
    }
    res_full = inspect_historical_receipt(full_receipt)
    assert res_full["valid_historical_receipt"] is True
    assert res_full["is_production_qualified"] is True
    assert res_full["cpu_comparison_count"] == 1920


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_production_inspiral_strain_hashing_after_injection():
    """Verify that strain hashing reflects final post-injection data and captures mass grid provenance."""
    from tools.benchmarking.benchmark_gpu_search import benchmark_production_inspiral

    res = benchmark_production_inspiral(
        num_templates=2,
        size=65536,
        tile_size=2,
        num_segments=3,
        device="cpu",
    )
    prov = res["input_provenance"]

    # 1. Post-injection strain hashes must match submitted segment hashes
    assert prov["submitted_segment_sha256_list"] == prov["segment_sha256_list"]
    assert len(prov["segment_sha256_list"]) == 3
    assert len(prov["raw_noise_segment_sha256_list"]) == 3

    # 2. Segment 0 has injected signal, so post-injection hash differs from pre-injection noise
    assert prov["segment_sha256_list"][0] != prov["raw_noise_segment_sha256_list"][0]

    # 3. Segments 1 and 2 have no injection, so post-injection hash equals pre-injection noise
    assert prov["segment_sha256_list"][1] == prov["raw_noise_segment_sha256_list"][1]
    assert prov["segment_sha256_list"][2] == prov["raw_noise_segment_sha256_list"][2]

    # 4. Input provenance records deterministic mass grid config and segment seeds, no unused bank_seed
    assert "mass_grid_config" in prov
    assert prov["mass_grid_config"]["mass1_range"] == [10.0, 50.0]
    assert prov["mass_grid_config"]["mass2_range"] == [1.4, 25.0]
    assert prov["mass_grid_config"]["grid_type"] == "linear"
    assert prov["segment_seeds"] == [1000, 1001, 1002]
    assert "bank_seed" not in prov


def test_fresh_process_cpu_reference_concurrency_regression():
    """Verify that isolated CPU reference execution in a fresh process does not segfault on uninitialized FFTW."""
    cmd = [
        sys.executable,
        "-X",
        "faulthandler",
        "-m",
        "pytest",
        "-q",
        "test/test_gpu_search_qualification.py::test_production_inspiral_strain_hashing_after_injection",
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert res.returncode == 0, (
        f"Fresh-process test failed (returncode={res.returncode}):\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )




