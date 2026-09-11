"""Test suite for fixed-capacity GPU-resident candidate experiment.

Verifies exact correctness against the reference route on CPU and CUDA,
asserts no host transfers in resident_device_stage, tests zero norms, aborts,
empty intervals, boundary thresholds, ties/NaNs, and overflow fallback.
"""

import os
import sys
import pytest
import numpy as np
import torch

sys_path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path_root not in sys.path:
    sys.path.insert(0, sys_path_root)

from pycbc.filter.gpu_search import SelectionPolicy
try:
    from tools.experiment_torch_resident_candidates import (
        generate_self_consistent_inputs,
        reference_candidate_pipeline,
        resident_candidate_pipeline,
        resident_device_stage,
    )
except ImportError:
    sys.path.insert(0, os.path.join(sys_path_root, "tools"))
    from experiment_torch_resident_candidates import (
        generate_self_consistent_inputs,
        reference_candidate_pipeline,
        resident_candidate_pipeline,
        resident_device_stage,
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("triggered", [False, True])
def test_resident_equality_to_reference(device, triggered):
    policy = SelectionPolicy(
        cluster_policy="symmetric",
        cluster_window=16,
        snr_threshold=5.5,
    )
    out_mem, norms_np, sigmasqs_np, corrs, bin_edges = generate_self_consistent_inputs(
        batch_size=2,
        transform_len=4096,
        num_bins=16,
        device=device,
        triggered=triggered,
        num_triggers_per_template=3,
        seed=42,
    )
    v_start = 512
    v_end = 3584

    ref = reference_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, v_start, v_end, policy, newsnr_threshold=5.5)
    res = resident_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, v_start, v_end, policy, capacity_per_template=32, newsnr_threshold=5.5)

    assert ref["aborted"] == res["aborted"]
    assert res["overflow"] is False
    assert res["fallback"] is False

    ref_s = ref["survivors"]
    res_s = res["survivors"]

    for k in ["template_idx", "sample_idx", "snr", "sigmasq", "chisq", "newsnr"]:
        assert k in ref_s and k in res_s
        assert ref_s[k].dtype == res_s[k].dtype, f"dtype mismatch for key {k}"

    np.testing.assert_array_equal(ref_s["template_idx"], res_s["template_idx"])
    np.testing.assert_array_equal(ref_s["sample_idx"], res_s["sample_idx"])
    np.testing.assert_array_equal(ref_s["sigmasq"], res_s["sigmasq"])
    np.testing.assert_allclose(ref_s["snr"], res_s["snr"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ref_s["chisq"], res_s["chisq"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ref_s["newsnr"], res_s["newsnr"], rtol=1e-4, atol=1e-4)

    if not triggered:
        assert len(res_s["sample_idx"]) == 0
        assert res["cands_before_cut"] == 0
    else:
        assert len(res_s["sample_idx"]) > 0, "Triggered test had zero survivors"
        assert res["cands_before_cut"] >= len(res_s["sample_idx"])
        assert np.all(res_s["sample_idx"] >= v_start)
        assert np.all(res_s["sample_idx"] < v_end)
        dof = 2.0 * (len(bin_edges) - 1) - 2.0
        assert np.any(res_s["chisq"] > dof), "Expected at least one survivor with chisq > DOF (reweighted)"


def test_zero_norm_and_empty_interval():
    device = "cpu"
    policy = SelectionPolicy(
        cluster_policy="symmetric",
        cluster_window=16,
        snr_threshold=5.5,
    )
    out_mem, norms_np, sigmasqs_np, corrs, bin_edges = generate_self_consistent_inputs(
        batch_size=2,
        transform_len=2048,
        num_bins=8,
        device=device,
        triggered=True,
        seed=99,
    )
    # Template 0 has zero norm
    norms_np[0] = 0.0
    v_start = 256
    v_end = 1024

    ref = reference_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, v_start, v_end, policy, newsnr_threshold=5.0)
    res = resident_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, v_start, v_end, policy, capacity_per_template=16, newsnr_threshold=5.0)

    # Template 0 should produce 0 survivors because of zero norm
    assert 0 not in res["survivors"]["template_idx"]
    np.testing.assert_array_equal(ref["survivors"]["template_idx"], res["survivors"]["template_idx"])
    np.testing.assert_array_equal(ref["survivors"]["sample_idx"], res["survivors"]["sample_idx"])

    # Empty valid interval
    ref_empty = reference_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, 100, 100, policy)
    res_empty = resident_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, 100, 100, policy, capacity_per_template=16)
    assert len(ref_empty["survivors"]["sample_idx"]) == 0
    assert len(res_empty["survivors"]["sample_idx"]) == 0


def test_synthetic_selector_stress_ties_nans_partial_window():
    """Explicit synthetic selector-only stress fixture testing ties, NaNs, and partial window."""
    policy = SelectionPolicy(
        cluster_policy="symmetric",
        cluster_window=16,
        snr_threshold=5.0,
    )
    batch_size = 2
    transform_len = 50  # Partial window: 50 = 3*16 + 2
    N = transform_len
    norms_np = np.ones(batch_size, dtype=np.float32)
    sigmasqs_np = np.ones(batch_size, dtype=np.float32) * 20.0
    bin_edges = [1, N // 4, N // 2]
    corrs = torch.zeros((batch_size, N), dtype=torch.complex64)

    # Construct non-FFT synthetic selector stress output
    out_mem = torch.zeros((batch_size, N), dtype=torch.complex64)
    out_mem[0, 5] = complex(float("nan"), float("nan"))
    out_mem[1, 10] = complex(float("nan"), 0.0)
    # Consecutive tie
    out_mem[0, 20] = complex(10.0, 0.0)
    out_mem[0, 21] = complex(10.0, 0.0)

    ref = reference_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, 0, N, policy, newsnr_threshold=0.0)
    res = resident_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, 0, N, policy, capacity_per_template=16, newsnr_threshold=0.0)

    np.testing.assert_array_equal(ref["survivors"]["template_idx"], res["survivors"]["template_idx"])
    np.testing.assert_array_equal(ref["survivors"]["sample_idx"], res["survivors"]["sample_idx"])


def test_abort_threshold():
    device = "cpu"
    policy = SelectionPolicy(
        cluster_policy="symmetric",
        cluster_window=16,
        snr_threshold=5.0,
        snr_abort_threshold=15.0,
    )
    out_mem, norms_np, sigmasqs_np, corrs, bin_edges = generate_self_consistent_inputs(
        batch_size=1,
        transform_len=2048,
        num_bins=8,
        device=device,
        triggered=False,
        seed=77,
    )
    # Set massive pulse in positive frequency support to produce real abort peak via FFT
    k_pos_end = 1024
    corrs[:, 1:k_pos_end] += 1e5
    out_abort = torch.fft.ifft(corrs, n=2048, dim=-1, norm="forward")
    ref = reference_candidate_pipeline(out_abort, norms_np, sigmasqs_np, corrs, bin_edges, 200, 800, policy)
    res = resident_candidate_pipeline(out_abort, norms_np, sigmasqs_np, corrs, bin_edges, 200, 800, policy, capacity_per_template=16)

    assert ref["aborted"] is True
    assert res["aborted"] is True


def test_capacity_rejection_and_overflow_fallback():
    device = "cpu"
    policy = SelectionPolicy(
        cluster_policy="symmetric",
        cluster_window=16,
        snr_threshold=5.0,
    )
    out_mem, norms_np, sigmasqs_np, corrs, bin_edges = generate_self_consistent_inputs(
        batch_size=1,
        transform_len=2048,
        num_bins=8,
        device=device,
        triggered=True,
        num_triggers_per_template=5,
        seed=123,
    )
    # Reject capacity <= 0 cleanly
    with pytest.raises(ValueError):
        resident_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, 200, 1800, policy, capacity_per_template=0)

    # Deliberately tiny capacity = 1 to trigger overflow fallback
    ref = reference_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, 200, 1800, policy, newsnr_threshold=5.0)
    res = resident_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, 200, 1800, policy, capacity_per_template=1, newsnr_threshold=5.0)

    assert res["overflow"] is True
    assert res["fallback"] is True
    assert res["overflow_count"] > 0
    assert len(res["survivors"]["sample_idx"]) > 1
    np.testing.assert_array_equal(ref["survivors"]["sample_idx"], res["survivors"]["sample_idx"])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_threshold_below_exact_above_nextafter(device):
    """Probe decisions at below, exact, and above np.nextafter boundaries on both CPU and CUDA."""
    policy = SelectionPolicy(
        cluster_policy="symmetric",
        cluster_window=16,
        snr_threshold=5.0,
    )
    out_mem, norms_np, sigmasqs_np, corrs, bin_edges = generate_self_consistent_inputs(
        batch_size=1,
        transform_len=2048,
        num_bins=8,
        device=device,
        triggered=True,
        num_triggers_per_template=3,
        seed=234,
    )
    ref_initial = reference_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, 200, 1800, policy, newsnr_threshold=5.0)
    actual_newsnr = ref_initial["survivors"]["newsnr"]
    assert len(actual_newsnr) > 0
    boundary_val = float(actual_newsnr[0])

    # 1. Exactly at boundary
    ref_exact = reference_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, 200, 1800, policy, newsnr_threshold=boundary_val)
    res_exact = resident_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, 200, 1800, policy, capacity_per_template=16, newsnr_threshold=boundary_val)
    np.testing.assert_array_equal(ref_exact["survivors"]["sample_idx"], res_exact["survivors"]["sample_idx"])

    # 2. Nextafter below (must retain survivor)
    val_below = np.nextafter(boundary_val, -np.inf)
    ref_below = reference_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, 200, 1800, policy, newsnr_threshold=val_below)
    res_below = resident_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, 200, 1800, policy, capacity_per_template=16, newsnr_threshold=val_below)
    np.testing.assert_array_equal(ref_below["survivors"]["sample_idx"], res_below["survivors"]["sample_idx"])

    # 3. Nextafter above (cuts survivor; exact agreement guaranteed via conservative boundary guard)
    val_above = np.nextafter(boundary_val, np.inf)
    ref_above = reference_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, 200, 1800, policy, newsnr_threshold=val_above)
    res_above = resident_candidate_pipeline(out_mem, norms_np, sigmasqs_np, corrs, bin_edges, 200, 1800, policy, capacity_per_template=16, newsnr_threshold=val_above)
    np.testing.assert_array_equal(ref_above["survivors"]["sample_idx"], res_above["survivors"]["sample_idx"])
    assert len(res_above["survivors"]["sample_idx"]) < len(res_exact["survivors"]["sample_idx"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for device stage isolation test")
def test_cuda_resident_device_stage_no_host_transfer():
    """Assert resident_device_stage does zero host exports/conversions on CUDA."""
    device = "cuda"
    policy = SelectionPolicy(
        cluster_policy="symmetric",
        cluster_window=16,
        snr_threshold=5.5,
    )
    out_mem, norms_np, sigmasqs_np, corrs, bin_edges = generate_self_consistent_inputs(
        batch_size=2,
        transform_len=2048,
        num_bins=8,
        device=device,
        triggered=True,
        num_triggers_per_template=2,
        seed=345,
    )
    v_start = 256
    v_end = 1792
    norms_tensor = torch.as_tensor(norms_np, device=device, dtype=torch.float32)
    sigmasqs_tensor = torch.as_tensor(sigmasqs_np, device=device, dtype=torch.float32)

    # Monkeypatch forbidden calls during device stage execution
    orig_item = torch.Tensor.item
    orig_cpu = torch.Tensor.cpu
    orig_numpy = torch.Tensor.numpy
    orig_bool = torch.Tensor.__bool__

    def forbidden_call(name):
        raise AssertionError(f"Forbidden host conversion '{name}' called inside resident_device_stage!")

    try:
        torch.Tensor.item = lambda self: forbidden_call("item")
        torch.Tensor.cpu = lambda self: forbidden_call("cpu")
        torch.Tensor.numpy = lambda self: forbidden_call("numpy")
        torch.Tensor.__bool__ = lambda self: forbidden_call("__bool__")

        # Must execute without calling any of the forbidden host-transfer methods
        dev_res = resident_device_stage(
            out_mem=out_mem,
            norms_tensor=norms_tensor,
            sigmasqs_tensor=sigmasqs_tensor,
            corrs=corrs,
            bin_edges=bin_edges,
            valid_start=v_start,
            valid_end=v_end,
            policy=policy,
            capacity_per_template=16,
            newsnr_threshold=5.5,
            norms_np=norms_np,
        )
        assert isinstance(dev_res["surv_mask"], torch.Tensor)
    finally:
        torch.Tensor.item = orig_item
        torch.Tensor.cpu = orig_cpu
        torch.Tensor.numpy = orig_numpy
        torch.Tensor.__bool__ = orig_bool
