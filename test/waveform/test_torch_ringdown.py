# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Parity and device-residency tests for Torch ringdown waveforms."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pycbc  # noqa: E402
from pycbc import scheme  # noqa: E402
from pycbc.types import FrequencySeries, TimeSeries  # noqa: E402
from pycbc.types.array_torch import TorchArrayData  # noqa: E402
from pycbc.waveform import ringdown  # noqa: E402
from pycbc.waveform import ringdown_torch  # noqa: E402

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


@pytest.fixture(params=("cpu", "cuda", "mps"))
def torch_device_ctx(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")

    ctx = scheme.TorchScheme(device)
    try:
        yield ctx, device
    finally:
        del ctx
        scheme.Scheme._single = None


def _parameters():
    return {
        "lmns": ["221", "331", "201"],
        "amp220": 1.2,
        "phi220": 0.4,
        "f_220": 250.0,
        "tau_220": 0.02,
        "amp330": 0.35,
        "phi330": -0.3,
        "f_330": 410.0,
        "tau_330": 0.012,
        "amp200": 0.15,
        "phi200": 0.8,
        "f_200": 180.0,
        "tau_200": 0.018,
        "inclination": 0.7,
        "azimuthal": 0.2,
    }


def _assert_close(actual, expected, device):
    dtype = torch.float32 if device == "mps" else torch.float64
    if np.iscomplexobj(expected):
        dtype = torch.complex64 if device == "mps" else torch.complex128
    tolerances = (
        {"rtol": 3e-5, "atol": 1e-6}
        if device == "mps"
        else {"rtol": 1e-12, "atol": 1e-14}
    )
    torch.testing.assert_close(
        actual.detach().cpu(),
        torch.as_tensor(expected, dtype=dtype),
        **tolerances,
    )


def _reject_host_transfer(_self):
    raise AssertionError("ringdown copied Torch data to the host")


def _reject_numpy_exp(*_args, **_kwargs):
    raise AssertionError("ringdown evaluated exponentials with NumPy")


def _reject_lal_harmonic(*_args, **_kwargs):
    raise AssertionError("ringdown evaluated harmonics with LAL")


def test_public_primitive_dispatches_to_torch_backend(
    torch_device_ctx, monkeypatch
):
    ctx, _ = torch_device_ctx
    expected = object(), object()

    def fake_damped_sinusoid(*args, **kwargs):
        assert args == (100.0, 0.1, 1.0, 0.0, [0.0])
        assert kwargs["m"] == 2
        return expected

    monkeypatch.setattr(
        ringdown_torch, "td_damped_sinusoid", fake_damped_sinusoid
    )
    with ctx:
        actual = ringdown.td_damped_sinusoid(
            100.0, 0.1, 1.0, 0.0, [0.0]
        )
    assert actual == expected


def test_td_ringdown_torch_matches_cpu_without_host_transfer(
    torch_device_ctx, monkeypatch
):
    ctx, device = torch_device_ctx
    parameters = _parameters() | {
        "delta_t": 1 / 4096,
        "t_final": 0.05,
        "taper": True,
        "dbeta": 0.1,
        "dphi": -0.2,
    }
    reference_plus, reference_cross = ringdown.get_td_from_freqtau(
        **parameters
    )
    expected_plus = reference_plus.numpy().copy()
    expected_cross = reference_cross.numpy().copy()

    with ctx:
        monkeypatch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
        monkeypatch.setattr(ringdown.numpy, "exp", _reject_numpy_exp)
        monkeypatch.setattr(
            ringdown.lal,
            "SpinWeightedSphericalHarmonic",
            _reject_lal_harmonic,
        )
        plus, cross = ringdown.get_td_from_freqtau(**parameters)

    expected_dtype = torch.float32 if device == "mps" else torch.float64
    for result in (plus, cross):
        assert isinstance(result, TimeSeries)
        assert result._data.tensor.device.type == device
        assert result._data.tensor.dtype == expected_dtype
    assert plus.delta_t == reference_plus.delta_t
    assert float(plus.start_time) == float(reference_plus.start_time)
    _assert_close(plus._data.tensor, expected_plus, device)
    _assert_close(cross._data.tensor, expected_cross, device)


def test_fd_ringdown_torch_matches_cpu_without_host_transfer(
    torch_device_ctx, monkeypatch
):
    ctx, device = torch_device_ctx
    parameters = _parameters() | {
        "f_lower": 20.0,
        "f_final": 1024.0,
        "t_0": 0.003,
    }
    reference_plus, reference_cross = ringdown.get_fd_from_freqtau(
        **parameters
    )
    expected_plus = reference_plus.numpy().copy()
    expected_cross = reference_cross.numpy().copy()

    with ctx:
        monkeypatch.setattr(TorchArrayData, "numpy", _reject_host_transfer)
        monkeypatch.setattr(ringdown.numpy, "exp", _reject_numpy_exp)
        monkeypatch.setattr(
            ringdown.lal,
            "SpinWeightedSphericalHarmonic",
            _reject_lal_harmonic,
        )
        plus, cross = ringdown.get_fd_from_freqtau(**parameters)

    expected_dtype = (
        torch.complex64 if device == "mps" else torch.complex128
    )
    for result in (plus, cross):
        assert isinstance(result, FrequencySeries)
        assert result._data.tensor.device.type == device
        assert result._data.tensor.dtype == expected_dtype
    assert plus.delta_f == reference_plus.delta_f
    kmin = int(parameters["f_lower"] / plus.delta_f)
    assert torch.count_nonzero(plus._data.tensor[:kmin]) == 0
    _assert_close(plus._data.tensor, expected_plus, device)
    _assert_close(cross._data.tensor, expected_cross, device)
