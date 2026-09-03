"""Tests for rate posterior and efficiency utilities."""

from types import SimpleNamespace

import numpy as np
import pytest

import pycbc
from pycbc import rate


def _posterior(dtype=np.float64):
    mu = np.linspace(0.0, 10.0, 101, dtype=dtype)
    post = np.exp(-0.5 * ((mu - 4.0) / 1.5) ** 2)
    return mu, post


def test_rate_posterior_numpy_regression():
    mu, post = _posterior()

    normalized_mu, normalized = rate.normalize_pdf(mu, post)
    assert normalized_mu is mu
    np.testing.assert_allclose(
        rate.integral_element(normalized_mu, normalized).sum(), 1.0
    )
    assert rate.compute_upper_limit(mu, post, alpha=0.9) == 5.9
    assert rate.compute_lower_limit(mu, post, alpha=0.9) == 2.1
    np.testing.assert_allclose(
        rate.confidence_interval_min_width(mu, post, alpha=0.9),
        (1.4, 6.3),
    )

    threshold = rate.hpd_threshold(mu, post, alpha=0.9, tol=1e-3)
    scaled_threshold = rate.hpd_threshold(mu, 7.0 * post, alpha=0.9, tol=1e-3)
    np.testing.assert_allclose(scaled_threshold, 7.0 * threshold)
    np.testing.assert_allclose(
        rate.hpd_coverage(mu, post, threshold) / rate.integral_element(mu, post).sum(),
        0.9007723596866316,
    )
    assert rate.hpd_credible_interval(mu, post, alpha=0.9) == (1.6, 6.4)
    assert rate.hpd_credible_interval(mu, 7.0 * post, alpha=0.9) == (
        1.6,
        6.4,
    )


@pytest.mark.parametrize(
    "function",
    [
        rate.compute_upper_limit,
        rate.compute_lower_limit,
        rate.confidence_interval_min_width,
        rate.hpd_credible_interval,
    ],
)
def test_rate_intervals_reject_invalid_confidence(function):
    mu, post = _posterior()
    with pytest.raises(ValueError, match="Confidence level"):
        function(mu, post, alpha=0.0)
    with pytest.raises(ValueError, match="Confidence level"):
        function(mu, post, alpha=1.1)


def test_hpd_threshold_rejects_invalid_tolerance():
    mu, post = _posterior()
    with pytest.raises(ValueError, match="tolerance"):
        rate.hpd_threshold(mu, post, alpha=0.9, tol=0.0)


def test_efficiency_numpy_regression():
    found = np.array([0.1, 0.2, 0.4])
    missed = np.array([0.15, 0.35, 0.45])
    bins = np.array([0.0, 0.25, 0.5, 0.75])

    efficiency, error = rate.compute_efficiency(found, missed, bins)
    np.testing.assert_allclose(efficiency, [2.0 / 3.0, 1.0 / 3.0, 0.0])
    np.testing.assert_allclose(error, [0.2721655269759087, 0.2721655269759087, 0.0])
    volume, volume_error = rate.integrate_efficiency(bins, efficiency, error)
    np.testing.assert_allclose(volume, 0.17998707911191522)
    np.testing.assert_allclose(volume_error, 0.12097898615592095)

    found_rows = [SimpleNamespace(distance=value) for value in found]
    missed_rows = [SimpleNamespace(distance=value) for value in missed]
    mean_efficiency = rate.mean_efficiency_volume(found_rows, missed_rows, bins)
    np.testing.assert_allclose(mean_efficiency[0], efficiency)
    np.testing.assert_allclose(mean_efficiency[1], error)
    np.testing.assert_allclose(mean_efficiency[2], volume)
    np.testing.assert_allclose(mean_efficiency[3], volume_error)

    log_bins = np.array([1.0, 2.0, 4.0, 8.0])
    log_volume, log_error = rate.integrate_efficiency(
        log_bins, efficiency, error, logbins=True
    )
    assert np.isfinite(log_volume)
    assert np.isfinite(log_error)


@pytest.fixture(params=("cpu", "cuda", "mps"))
def torch_device(request):
    if not pycbc.HAVE_TORCH:
        pytest.skip("PyCBC built without Torch support")

    import torch

    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")
    return device


def test_rate_tensor_paths_stay_on_device(torch_device, monkeypatch):
    import torch

    from pycbc.types.array_torch import TorchArrayData

    dtype = torch.float32 if torch_device == "mps" else torch.float64
    mu = torch.linspace(0.0, 10.0, 101, dtype=dtype, device=torch_device)
    post = torch.exp(-0.5 * ((mu - 4.0) / 1.5) ** 2)
    found = torch.tensor([0.1, 0.2, 0.4], dtype=dtype, device=torch_device)
    missed = torch.tensor([0.15, 0.35, 0.45], dtype=dtype, device=torch_device)
    bins = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=dtype, device=torch_device)
    found_rows = [SimpleNamespace(distance=value) for value in (0.1, 0.2, 0.4)]
    missed_rows = [SimpleNamespace(distance=value) for value in (0.15, 0.35, 0.45)]

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("rate calculation copied through host memory")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", reject_host_transfer)
        patch.setattr(torch.Tensor, "numpy", reject_host_transfer)
        patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
        patch.setattr(rate.numpy, "array", reject_host_transfer)
        patch.setattr(rate.numpy, "zeros", reject_host_transfer)
        normalized_mu, normalized = rate.normalize_pdf(mu, post)
        upper = rate.compute_upper_limit(mu, post, alpha=0.9)
        lower = rate.compute_lower_limit(mu, post, alpha=0.9)
        minimum = rate.confidence_interval_min_width(mu, post, alpha=0.9)
        threshold = rate.hpd_threshold(mu, post, alpha=0.9, tol=1e-3)
        hpd = rate.hpd_credible_interval(mu, post, alpha=0.9)
        efficiency, error = rate.compute_efficiency(found, missed, bins)
        volume, volume_error = rate.integrate_efficiency(bins, efficiency, error)
        mean_efficiency = rate.mean_efficiency_volume(found_rows, missed_rows, bins)
        empty_efficiency = rate.mean_efficiency_volume([], missed_rows, bins)

    tensors = (
        normalized_mu,
        normalized,
        upper,
        lower,
        *minimum,
        threshold,
        *hpd,
        efficiency,
        error,
        volume,
        volume_error,
        *mean_efficiency,
        *empty_efficiency,
    )
    assert all(isinstance(value, torch.Tensor) for value in tensors)
    assert all(value.device.type == torch_device for value in tensors)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    np.testing.assert_allclose(upper.item(), 5.9, rtol=tolerance)
    np.testing.assert_allclose(lower.item(), 2.1, rtol=tolerance)
    expected_minimum = [1.7, 6.6] if dtype == torch.float32 else [1.4, 6.3]
    np.testing.assert_allclose(
        [value.item() for value in minimum], expected_minimum, rtol=tolerance
    )
    np.testing.assert_allclose(
        [value.item() for value in hpd], [1.6, 6.4], rtol=tolerance
    )
    np.testing.assert_allclose(
        efficiency.detach().tolist(),
        [2.0 / 3.0, 1.0 / 3.0, 0.0],
        rtol=tolerance,
    )
    np.testing.assert_allclose(volume.item(), 0.17998707911191522, rtol=tolerance)
    assert volume_error.item() > 0.0
    np.testing.assert_allclose(
        mean_efficiency[0].detach().tolist(),
        [2.0 / 3.0, 1.0 / 3.0, 0.0],
        rtol=tolerance,
    )
    assert all(torch.count_nonzero(value) == 0 for value in empty_efficiency)


def test_rate_tensor_paths_preserve_gradients(torch_device):
    import torch

    dtype = torch.float32 if torch_device == "mps" else torch.float64
    mu = torch.linspace(0.0, 10.0, 101, dtype=dtype, device=torch_device)
    post = torch.exp(-0.5 * ((mu - 4.0) / 1.5) ** 2).requires_grad_()
    _, normalized = rate.normalize_pdf(mu, post)
    weights = torch.linspace(
        1.0, 2.0, len(normalized), dtype=dtype, device=torch_device
    )
    torch.sum(normalized * weights).backward()
    assert post.grad is not None
    assert torch.isfinite(post.grad).all()

    bins = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=dtype, device=torch_device)
    efficiency = torch.tensor(
        [2.0 / 3.0, 1.0 / 3.0, 0.0],
        dtype=dtype,
        device=torch_device,
        requires_grad=True,
    )
    volume, _ = rate.integrate_efficiency(bins, efficiency)
    volume.backward()
    assert efficiency.grad is not None
    assert torch.isfinite(efficiency.grad).all()


def test_rate_pycbc_arrays_remain_torch_backed(torch_device, monkeypatch):
    import torch

    from pycbc import scheme
    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData

    dtype = np.float32 if torch_device == "mps" else np.float64
    mu_values, post_values = _posterior(dtype=dtype)
    found_values = np.array([0.1, 0.2, 0.4], dtype=dtype)
    missed_values = np.array([0.15, 0.35, 0.45], dtype=dtype)
    bin_values = np.array([0.0, 0.25, 0.5, 0.75], dtype=dtype)
    found_rows = [SimpleNamespace(distance=value) for value in found_values]
    missed_rows = [SimpleNamespace(distance=value) for value in missed_values]

    context = scheme.TorchScheme(torch_device)
    try:
        with context:
            mu = Array(mu_values)
            post = Array(post_values)
            found = Array(found_values)
            missed = Array(missed_values)
            bins = Array(bin_values)

            def reject_host_transfer(*_args, **_kwargs):
                raise AssertionError("PyCBC rate array left the Torch device")

            monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            elements = rate.integral_element(mu, post)
            normalized_mu, normalized = rate.normalize_pdf(mu, post)
            upper = rate.compute_upper_limit(mu, post, alpha=0.9)
            efficiency, error = rate.compute_efficiency(found, missed, bins)
            mean_efficiency, mean_error, mean_volume, mean_volume_error = (
                rate.mean_efficiency_volume(found_rows, missed_rows, bins)
            )
    finally:
        del context
        scheme.Scheme._single = None

    for value in (
        elements,
        normalized_mu,
        normalized,
        efficiency,
        error,
        mean_efficiency,
        mean_error,
    ):
        assert isinstance(value, Array)
        assert isinstance(value._data, TorchArrayData)
        assert value._data.tensor.device.type == torch_device
    assert isinstance(upper, torch.Tensor)
    assert upper.device.type == torch_device
    assert isinstance(mean_volume, torch.Tensor)
    assert isinstance(mean_volume_error, torch.Tensor)
    assert mean_volume.device.type == torch_device
    assert mean_volume_error.device.type == torch_device
