"""Focused Torch tests for waveform generation and projection helpers."""

import importlib
import sys
import types

import numpy
import pytest


torch = pytest.importorskip("torch")

import pycbc  # noqa: E402
from pycbc import scheme  # noqa: E402
from pycbc.types import FrequencySeries  # noqa: E402
from pycbc.types.array_torch import TorchArrayData  # noqa: E402


if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


@pytest.fixture
def torch_context():
    context = scheme.TorchScheme("cpu")
    try:
        yield context
    finally:
        del context
        scheme.Scheme._single = None


def _load_generator(monkeypatch):
    fake_strain = types.ModuleType("pycbc.strain")
    fake_strain.apply_gates_to_fd = lambda series, gates: series
    monkeypatch.setitem(sys.modules, "pycbc.strain", fake_strain)
    monkeypatch.delitem(sys.modules, "pycbc.waveform.generator", raising=False)
    return importlib.import_module("pycbc.waveform.generator")


class _StaticCPUGenerator:
    def __init__(self, variable_args=(), **frozen_params):
        self.variable_args = tuple(variable_args)
        self.frozen_params = frozen_params

    def generate(self, **kwargs):
        old_state = scheme.mgr.state
        old_single = scheme.Scheme._single
        scheme.Scheme._single = None
        scheme.mgr.state = scheme.CPUScheme()
        try:
            hp = FrequencySeries(
                numpy.ones(8, dtype=numpy.complex128), delta_f=0.25
            )
            hc = FrequencySeries(
                1j * numpy.ones(8, dtype=numpy.complex128), delta_f=0.25
            )
        finally:
            scheme.mgr.state = old_state
            scheme.Scheme._single = old_single
        return hp, hc


class _StaticTorchGenerator:
    def __init__(self, variable_args=(), **frozen_params):
        self.variable_args = tuple(variable_args)
        self.frozen_params = frozen_params

    def generate(self, **kwargs):
        hp = FrequencySeries(
            TorchArrayData(torch.ones(8, dtype=torch.complex128)),
            delta_f=0.25,
            copy=False,
        )
        hc = FrequencySeries(
            TorchArrayData(1j * torch.ones(8, dtype=torch.complex128)),
            delta_f=0.25,
            copy=False,
        )
        return hp, hc


def test_detector_frame_generator_moves_output_to_torch(
        monkeypatch, torch_context):
    generator = _load_generator(monkeypatch)

    with torch_context:
        detgen = generator.FDomainDetFrameGenerator(
            _StaticCPUGenerator,
            epoch=0.0,
            detectors=None,
            delta_f=0.25,
        )
        result = detgen.generate()["RF"]

    assert isinstance(result._data.tensor, torch.Tensor)
    assert result._data.tensor.device.type == "cpu"


def test_time_shift_uses_an_explicit_sample_axis(torch_context):
    from pycbc.waveform.utils_torch import apply_fseries_time_shift

    with torch_context:
        batch_size = frequency_count = 4
        raw = torch.ones(
            (batch_size, frequency_count), dtype=torch.complex128
        )
        series = FrequencySeries(
            TorchArrayData(raw.clone()),
            delta_f=0.5,
            epoch=0.0,
            copy=False,
        )
        shifts = torch.linspace(0.01, 0.04, batch_size)
        actual = apply_fseries_time_shift(
            series, shifts, copy=False
        )._data.tensor

    frequencies = torch.arange(frequency_count) * 0.5
    expected = raw * torch.exp(
        -2j * torch.pi * shifts[:, None] * frequencies
    )
    torch.testing.assert_close(actual, expected)


def test_fused_projection_matches_individual_detector_results():
    from pycbc.waveform.utils_torch import fused_detector_strain_fd_torch

    hp = torch.randn(32, dtype=torch.complex128)
    hc = torch.randn(32, dtype=torch.complex128)
    fplus = [0.8, -0.4]
    fcross = [0.2, 0.7]
    delays = [0.002, -0.005]
    delta_f = 0.25

    actual = fused_detector_strain_fd_torch(
        hp, hc, fplus, fcross, delays, delta_f, kmin=2
    )
    assert actual.shape == (2, 32)
    frequencies = torch.arange(2, 32, dtype=torch.float64) * delta_f
    for index in range(2):
        expected = fplus[index] * hp + fcross[index] * hc
        expected = expected.clone()
        expected[2:] *= torch.exp(
            -2j * torch.pi * delays[index] * frequencies
        )
        torch.testing.assert_close(actual[index], expected)


def test_arrival_time_is_centered_before_float32_offsets_are_applied():
    from pycbc.waveform.generator import _arrival_time_and_shift

    epoch = 1126259460.0
    reference_time = 1126259462.125
    offset = torch.tensor(0.003, dtype=torch.float32)
    arrival, relative = _arrival_time_and_shift(
        reference_time, offset, epoch
    )

    assert arrival.dtype == torch.float64
    assert relative.dtype == torch.float64
    torch.testing.assert_close(
        relative,
        torch.tensor(2.128, dtype=torch.float64),
        rtol=0.0,
        atol=1e-8,
    )


@pytest.mark.parametrize("tensor_subclass", [False, True])
def test_arrival_time_accepts_public_backend_values(tensor_subclass):
    from pycbc.waveform.generator import _arrival_time_and_shift

    class UserTensor(torch.Tensor):
        pass

    class BackendValue:
        backend = "torch"

        def __init__(self, tensor):
            self.backend_array = tensor

    def wrap(tensor):
        if tensor_subclass:
            return tensor.as_subclass(UserTensor)
        return BackendValue(tensor)

    epoch = 1126259460.0
    reference = torch.tensor(
        epoch + 2.125, dtype=torch.float64, requires_grad=True
    )
    offset = torch.tensor(0.003, dtype=torch.float32, requires_grad=True)
    arrival, relative = _arrival_time_and_shift(
        wrap(reference), wrap(offset), epoch
    )
    torch.testing.assert_close(arrival, reference + offset.double())
    torch.testing.assert_close(
        relative, reference - epoch + offset.double()
    )
    relative.backward()
    torch.testing.assert_close(reference.grad, torch.ones_like(reference))
    torch.testing.assert_close(offset.grad, torch.ones_like(offset))


def test_detector_projection_preserves_parameter_gradients(
        monkeypatch, torch_context):
    generator = _load_generator(monkeypatch)

    class DifferentiableDetector:
        def arrival_time(self, ref_tc, ra, dec, _reference_frame):
            return ref_tc + 0.02 * ra - 0.03 * dec

        @staticmethod
        def antenna_pattern(ra, dec, polarization, arrival_time):
            return (
                torch.cos(2.0 * polarization)
                + 0.01 * ra
                + 0.001 * arrival_time,
                torch.sin(2.0 * polarization)
                + 0.01 * dec
                - 0.001 * arrival_time,
            )

    with torch_context:
        detgen = generator.FDomainDetFrameGenerator(
            _StaticTorchGenerator,
            epoch=0.0,
            detectors=["H1"],
            variable_args=["tc", "ra", "dec", "polarization"],
            delta_f=0.25,
        )
        detgen.detectors = {"H1": DifferentiableDetector()}
        params = {
            "tc": torch.tensor(2.0, requires_grad=True),
            "ra": torch.tensor(1.1, requires_grad=True),
            "dec": torch.tensor(-0.4, requires_grad=True),
            "polarization": torch.tensor(0.3, requires_grad=True),
        }
        result = detgen.generate(**params)["H1"]._data.tensor
        (result.real.sum() + result.imag.sum()).backward()

    for value in params.values():
        assert value.grad is not None
        assert torch.isfinite(value.grad)
