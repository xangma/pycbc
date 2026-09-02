"""Tests for search-sensitivity volume utilities."""

import numpy as np
import pytest

import pycbc
from pycbc import scheme, sensitivity


def _injections(dtype=np.float64):
    return (
        np.array([10.0, 20.0, 35.0], dtype=dtype),
        np.array([25.0, 40.0, 50.0, 60.0], dtype=dtype),
        np.array([1.1, 1.4, 2.0], dtype=dtype),
        np.array([1.2, 1.6, 2.3, 3.0], dtype=dtype),
    )


_MONTE_CARLO_CASES = (
    (
        "direct",
        ("distance", "uniform", "distance", None, None),
        (155297.83386103573, 100474.16165297432),
    ),
    (
        "direct",
        ("chirp_distance", "log", "chirp_distance", 8.0, 48.0),
        (206620.92065328962, 158201.93921739288),
    ),
    (
        "direct",
        ("chirp_distance", "volume", "chirp_distance", None, None),
        (76656.47799623031, 50312.95932686046),
    ),
    (
        "chirp",
        ("chirp_distance", "log", "chirp_distance", 8.0, 48.0),
        (77183.53645226148, 46193.92269486361),
    ),
)


def _monte_carlo(kind, injections, arguments):
    function = (
        sensitivity.volume_montecarlo
        if kind == "direct"
        else sensitivity.chirp_volume_montecarlo
    )
    return function(*injections, *arguments)


def test_sensitivity_numpy_regression():
    volume = np.array([100.0, 1.0])
    error = np.array([20.0, 2.0])
    distance, upper, lower = sensitivity.volume_to_distance_with_errors(volume, error)
    np.testing.assert_allclose(distance, [2.87941191, 0.62035049])
    np.testing.assert_allclose(upper, [0.18041983, 0.27434974])
    np.testing.assert_allclose(lower, [0.20640268, 0.62035049])

    injections = _injections()
    for kind, arguments, expected in _MONTE_CARLO_CASES:
        np.testing.assert_allclose(
            _monte_carlo(kind, injections, arguments), expected, rtol=2e-15
        )

    found = np.array([30.0, 10.0, 20.0])
    missed = np.array([40.0, 15.0, 25.0, 35.0])
    np.testing.assert_allclose(
        sensitivity.volume_shell(found, missed),
        (97389.37226128358, 63844.70704490627),
    )
    np.testing.assert_array_equal(found, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(missed, [15.0, 25.0, 35.0, 40.0])


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


def test_volume_to_distance_stays_on_device(torch_device, monkeypatch):
    import torch

    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData

    dtype = np.float32 if torch_device == "mps" else np.float64
    volume = np.array([100.0, 1.0], dtype=dtype)
    error = np.array([20.0, 2.0], dtype=dtype)
    expected = sensitivity.volume_to_distance_with_errors(volume, error)

    context = scheme.TorchScheme(torch_device)
    try:
        with context:
            volume_array = Array(volume)
            error_array = Array(error)

            def reject_host_transfer(*_args, **_kwargs):
                raise AssertionError("sensitivity calculation left Torch")

            with monkeypatch.context() as patch:
                patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
                patch.setattr(sensitivity.numpy, "where", reject_host_transfer)
                actual = sensitivity.volume_to_distance_with_errors(
                    volume_array, error_array
                )
    finally:
        del context
        scheme.Scheme._single = None

    tolerance = 3e-5 if dtype == np.float32 else 2e-12
    for result, reference in zip(actual, expected):
        assert isinstance(result, Array)
        assert isinstance(result._data, TorchArrayData)
        tensor = result._data.tensor
        assert tensor.device.type == torch_device
        assert tensor.dtype == (torch.float32 if dtype == np.float32 else torch.float64)
        np.testing.assert_allclose(
            tensor.detach().cpu().numpy(), reference, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize(
    "kind,arguments,_expected",
    _MONTE_CARLO_CASES,
    ids=("distance", "chirp-log", "chirp-volume", "chirp-wrapper"),
)
def test_volume_montecarlo_stays_on_device(
    torch_device, monkeypatch, kind, arguments, _expected
):
    import torch

    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData

    dtype = np.float32 if torch_device == "mps" else np.float64
    injections = _injections(dtype=dtype)
    expected = _monte_carlo(kind, injections, arguments)

    context = scheme.TorchScheme(torch_device)
    try:
        with context:
            device_injections = tuple(Array(values) for values in injections)

            def reject_host_transfer(*_args, **_kwargs):
                raise AssertionError("sensitivity calculation left Torch")

            with monkeypatch.context() as patch:
                patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
                patch.setattr(sensitivity.numpy, "concatenate", reject_host_transfer)
                actual = _monte_carlo(kind, device_injections, arguments)
    finally:
        del context
        scheme.Scheme._single = None

    tolerance = 8e-5 if dtype == np.float32 else 3e-12
    for result, reference in zip(actual, expected):
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
        assert result.device.type == torch_device
        np.testing.assert_allclose(
            result.detach().cpu().numpy(), reference, rtol=tolerance, atol=tolerance
        )


def test_volume_shell_stays_on_device(torch_device, monkeypatch):
    import torch

    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData

    dtype = np.float32 if torch_device == "mps" else np.float64
    found = np.array([30.0, 10.0, 20.0], dtype=dtype)
    missed = np.array([40.0, 15.0, 25.0, 35.0], dtype=dtype)
    expected = sensitivity.volume_shell(found.copy(), missed.copy())

    context = scheme.TorchScheme(torch_device)
    try:
        with context:
            found_array = Array(found)
            missed_array = Array(missed)

            def reject_host_transfer(*_args, **_kwargs):
                raise AssertionError("shell-volume calculation left Torch")

            with monkeypatch.context() as patch:
                patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
                patch.setattr(sensitivity.numpy, "concatenate", reject_host_transfer)
                actual = sensitivity.volume_shell(found_array, missed_array)
    finally:
        del context
        scheme.Scheme._single = None

    tolerance = 8e-5 if dtype == np.float32 else 3e-12
    for result, reference in zip(actual, expected):
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
        assert result.device.type == torch_device
        np.testing.assert_allclose(
            result.detach().cpu().numpy(), reference, rtol=tolerance, atol=tolerance
        )


def test_sensitivity_tensor_path_preserves_gradients(torch_device):
    import torch

    dtype = torch.float32 if torch_device == "mps" else torch.float64
    found_d, missed_d, found_mchirp, missed_mchirp = (
        torch.as_tensor(values, device=torch_device, dtype=dtype)
        for values in _injections()
    )
    found_d.requires_grad_()
    volume, error = sensitivity.volume_montecarlo(
        found_d,
        missed_d,
        found_mchirp,
        missed_mchirp,
        "distance",
        "uniform",
        "distance",
        max_param=60.0,
    )
    (volume + error).backward()

    assert found_d.grad is not None
    assert torch.isfinite(found_d.grad).all()

    shell_found = found_d.detach().clone().requires_grad_()
    shell_missed = missed_d.detach().clone()
    shell_volume, shell_error = sensitivity.volume_shell(shell_found, shell_missed)
    (shell_volume + shell_error).backward()
    assert shell_found.grad is not None
    assert torch.isfinite(shell_found.grad).all()
