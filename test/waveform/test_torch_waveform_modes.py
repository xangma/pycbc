"""Torch integration tests for the public waveform-mode helpers."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")

from pycbc import scheme  # noqa: E402
from pycbc.types import FrequencySeries  # noqa: E402
from pycbc.types.backend import (  # noqa: E402
    backend_array,
    wrap_backend_array,
)
from pycbc.waveform import (  # noqa: E402
    filter_approximants,
    get_fd_waveform,
    get_fd_waveform_sequence,
    get_sgburst_waveform,
)
from pycbc.waveform import waveform as waveform_module  # noqa: E402
from pycbc.waveform import waveform_modes  # noqa: E402
from pycbc.waveform.waveform_modes import get_glm, sum_modes  # noqa: E402


@pytest.fixture
def torch_cpu_ctx():
    ctx = scheme.TorchScheme("cpu")
    try:
        yield ctx
    finally:
        del ctx
        scheme.Scheme._single = None


def _modes():
    x = np.arange(32, dtype=np.float64)
    return {
        (2, 2): FrequencySeries(x + 1j * x[::-1], delta_f=0.25),
        (3, -2): FrequencySeries(0.5 * x - 0.25j * x, delta_f=0.25),
        (4, 1): FrequencySeries(np.cos(x) + 1j * np.sin(x), delta_f=0.25),
    }


def test_sum_modes_matches_lal_path(torch_cpu_ctx):
    reference_modes = _modes()
    reference = sum_modes(reference_modes, inclination=0.7, phi=-0.2)

    with torch_cpu_ctx:
        torch_modes = {
            mode: FrequencySeries(series.numpy(), delta_f=series.delta_f)
            for mode, series in reference_modes.items()
        }
        actual = sum_modes(torch_modes, inclination=0.7, phi=-0.2)

    assert actual._data.tensor.device.type == "cpu"
    np.testing.assert_allclose(
        actual.numpy(), reference.numpy(), rtol=2e-13, atol=2e-13
    )


def test_sum_modes_preserves_metadata_and_sample_gradients(torch_cpu_ctx):
    inclination, phi = 0.7, -0.2
    modes = ((2, 2), (3, -2), (4, 1))
    samples = [torch.tensor([1.0 + 2.0j, 3.0 - 1.0j],
                            dtype=torch.complex128, requires_grad=True)
               for _ in modes]
    with torch_cpu_ctx:
        supplied = {
            mode: FrequencySeries(wrap_backend_array(value), delta_f=0.25,
                                  epoch=123, copy=False)
            for mode, value in zip(modes, samples)
        }
        result = sum_modes(supplied, inclination, phi)
        data = backend_array(result, "torch")
        assert data.dtype == torch.complex128
        assert data.device.type == "cpu"
        assert result.delta_f == 0.25
        assert result.epoch == 123
        gradients = torch.autograd.grad(data.real.sum(), samples)

    for mode, gradient in zip(modes, gradients):
        harmonic = lal.SpinWeightedSphericalHarmonic(
            inclination, phi, -2, *mode
        )
        torch.testing.assert_close(
            gradient, torch.full_like(gradient, harmonic.conjugate()),
            rtol=2e-13, atol=2e-13,
        )


def test_get_glm_remains_lal_compatible():
    expected = lal.SpinWeightedSphericalHarmonic(0.9, 0.0, -2, 3, -1).real
    assert get_glm(3, -1, 0.9) == pytest.approx(expected, abs=0.0)


def test_interpolated_lal_waveform_is_torch_backed(torch_cpu_ctx):
    with torch_cpu_ctx:
        hp, hc = get_fd_waveform(
            approximant="TaylorF2_INTERP",
            mass1=20,
            mass2=10,
            delta_f=0.25,
            f_lower=30,
        )

    assert hp._data.tensor.device.type == "cpu"
    assert hc._data.tensor.device.type == "cpu"
    assert torch.isfinite(hp._data.tensor).all()
    assert torch.isfinite(hc._data.tensor).all()


def test_unimplemented_filter_is_not_advertised(torch_cpu_ctx):
    with torch_cpu_ctx:
        assert "SPAtmplt" not in filter_approximants()


def test_lal_sgburst_fallback_is_torch_backed(torch_cpu_ctx):
    with torch_cpu_ctx:
        hp, hc = get_sgburst_waveform(
            q=8,
            frequency=150,
            hrss=1e-21,
            delta_t=1 / 4096,
        )

    assert hp._data.tensor.device.type == "cpu"
    assert hc._data.tensor.device.type == "cpu"
    assert torch.isfinite(hp._data.tensor).all()
    assert torch.isfinite(hc._data.tensor).all()


def test_native_mode_availability_is_torch_only(monkeypatch, torch_cpu_ctx):
    monkeypatch.setattr(
        waveform_modes,
        "native_approximants",
        lambda interface: (f"TorchOnly-{interface}",),
    )
    cpu = scheme.CPUScheme()

    assert "TorchOnly-fd_modes" not in (
        waveform_modes.fd_waveform_mode_approximants(cpu)
    )
    assert "TorchOnly-td_modes" not in (
        waveform_modes.td_waveform_mode_approximants(cpu)
    )
    assert "TorchOnly-fd_modes" in (
        waveform_modes.fd_waveform_mode_approximants(torch_cpu_ctx)
    )
    assert "TorchOnly-td_modes" in (
        waveform_modes.td_waveform_mode_approximants(torch_cpu_ctx)
    )


def test_native_sequence_availability_is_torch_only(
    monkeypatch, torch_cpu_ctx
):
    approximant = "TorchOnlySequence"
    monkeypatch.setattr(
        waveform_module,
        "native_approximants",
        lambda interface: (approximant,) if interface == "sequence" else (),
    )

    def generate(**params):
        return params["approximant"], params["sample_points"]

    generate.required = ()
    monkeypatch.setattr(waveform_module, "_lalsim_fd_sequence", generate)

    with pytest.raises(ValueError, match="not available"):
        get_fd_waveform_sequence(
            approximant=approximant,
            sample_points=np.arange(4),
        )
    with torch_cpu_ctx:
        actual_approximant, sample_points = get_fd_waveform_sequence(
            approximant=approximant,
            sample_points=np.arange(4),
        )

    assert actual_approximant == approximant
    assert sample_points._data.tensor.device.type == "cpu"
