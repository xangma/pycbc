"""Candidate ownership and independently calculated blocked veto regressions."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc.filter.gpu_search.candidates import (  # noqa: E402
    CandidateBuffer, SelectionPolicy, select_tile_candidates,
)
from pycbc.filter.gpu_search.vetoes import (  # noqa: E402
    batched_power_chisq, power_chisq_scratch_shape,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("policy", ["live_peak", "symmetric", "threshold_only"])
def test_device_selection_owns_samples_and_has_no_host_payload(device, policy, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    values = torch.tensor([[0, 8j, 1, 0, 9, 0]], device=device,
                          dtype=torch.complex64)
    norms = torch.ones(1, device=device)
    cfg = SelectionPolicy(5, cluster_policy=policy, cluster_window=2)
    expected = select_tile_candidates(values, norms, norms, 0, 6, cfg)
    with monkeypatch.context() as patch:
        def no_cpu(*args, **kwargs):
            raise AssertionError("candidate payload copied to host")
        patch.setattr(torch.Tensor, "cpu", no_cpu)
        result = select_tile_candidates(values, norms, norms, 0, 6, cfg,
                                        return_device=True)
    values.zero_()
    for key, value in result["candidates"].items():
        assert value.device.type == device
        np.testing.assert_array_equal(value.cpu(), expected["candidates"][key])
    empty = select_tile_candidates(values, norms, norms, 0, 0, cfg,
                                   return_device=True)
    assert all(v.device.type == device and v.numel() == 0
               for v in empty["candidates"].values())


def test_dense_buffer_growth_preserves_every_candidate():
    values = torch.full((3, 11), 8j)
    norms = torch.ones(3)
    buffer = CandidateBuffer(capacity=1, device="cpu", allow_growth=True)
    result = select_tile_candidates(
        values, norms, norms, 0, 11,
        SelectionPolicy(5, cluster_policy="threshold_only"),
        buffer=buffer, return_device=True,
    )
    assert not result["overflow"]
    assert buffer.count == 33
    np.testing.assert_array_equal(result["candidates"]["sample_idx"],
                                  np.tile(np.arange(11), 3))


@pytest.mark.parametrize("device", ["numpy", "cpu", "cuda"])
@pytest.mark.parametrize("ragged", [False, True])
def test_blocked_veto_against_independent_direct_sum(device, ragged, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    n = 8192
    rng = np.random.default_rng(67)
    corr = (rng.normal(size=(2, n//2+1)) +
            1j*rng.normal(size=(2, n//2+1))).astype(np.complex128)
    # Contributions before the first bin must not contaminate narrow bins.
    corr[:, :31] *= 1e12
    edges = [np.array([31, 1024, 1024, 1025, 3300]),
             np.array([31, 32, 1025, 3300]) if ragged else
             np.array([31, 32, 32, 1025, 3300])]
    if not ragged:
        edges = np.stack(edges)
    ti = np.array([1, 0, 1, 0])
    si = np.array([n-1, n//2+1, 0, 1997])
    snr = np.array([1j, 2, 3j, 4], dtype=np.complex128)
    norms = np.array([0.8, 1.2])
    expected = []
    for t, sample, rho in zip(ti, si, snr):
        bins = edges[t]
        contributions = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            # Independent bin-local dot product; no cumulative endpoints.
            k = np.arange(lo, hi, dtype=np.int64)
            phase = np.exp(2j*np.pi*((k*sample) % n)/n)
            contributions.append(np.dot(corr[t, lo:hi], phase))
        expected.append(max(0, (len(bins)-1)*norms[t]**2 *
                            np.sum(np.abs(contributions)**2)-abs(rho)**2))
    candidates = dict(template_idx=ti, sample_idx=si, snr=snr)
    if device != "numpy":
        corr = torch.as_tensor(corr, device=device)
        candidates = {k: torch.as_tensor(v, device=device)
                      for k, v in candidates.items()}
    with monkeypatch.context() as patch:
        if device != "numpy":
            def no_cpu(*args, **kwargs):
                raise AssertionError("veto payload copied to host")
            patch.setattr(torch.Tensor, "cpu", no_cpu)
        actual, dof = batched_power_chisq(
            corr, candidates, edges, norms, transform_length=n,
            chunk_size=3, scratch_budget_bytes=8192, return_device=True,
        )
    if device != "numpy":
        assert actual.device.type == device
        actual, dof = actual.cpu().numpy(), dof.cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-9)
    np.testing.assert_array_equal(dof, [2*(len(edges[t])-1)-2 for t in ti])


def test_scratch_plan_bounds_simultaneously_live_storage():
    for budget in (8192, 1 << 20, 64 << 20):
        for bins in (1, 16, 31):
            rows, width = power_chisq_scratch_shape(bins, 2048, budget)
            assert 0 < rows <= 2048 and 0 < width <= 4096
            assert rows*(128*width + 128*bins + 256) <= budget
    with pytest.raises(ValueError, match="too small"):
        power_chisq_scratch_shape(100, 2048, 1024)


@pytest.mark.parametrize("device", ["numpy", "cpu", "cuda"])
def test_different_row_cutoffs_exclude_large_unrelated_bins(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    corr = np.array([[1, 1, 0], [1e20, 1, 1]], dtype=np.complex128)
    if device != "numpy":
        corr = torch.as_tensor(corr, device=device)
    result, _ = batched_power_chisq(
        corr, dict(template_idx=[0, 1], sample_idx=[0, 0], snr=[0j, 0j]),
        np.array([[0, 1, 2], [1, 2, 3]]), np.ones(2),
        transform_length=4,
    )
    np.testing.assert_array_equal(result, [4, 4])
