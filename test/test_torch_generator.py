import importlib
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pycbc
from pycbc import scheme
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


@pytest.fixture
def torch_ctx():
    ctx = scheme.TorchScheme("cpu")
    try:
        yield ctx
    finally:
        del ctx
        scheme.Scheme._single = None


def _load_generator_module(monkeypatch):
    fake_strain = types.ModuleType("pycbc.strain")
    fake_strain.apply_gates_to_fd = lambda series, gates: series
    monkeypatch.setitem(sys.modules, "pycbc.strain", fake_strain)
    monkeypatch.delitem(sys.modules, "pycbc.waveform.generator", raising=False)
    return importlib.import_module("pycbc.waveform.generator")


class StaticRFrameGenerator:
    def __init__(self, variable_args=(), **frozen_params):
        self.variable_args = tuple(variable_args)
        self.frozen_params = frozen_params

    def generate(self, **kwargs):
        old_scheme = scheme.mgr.state
        old_single = scheme.Scheme._single
        scheme.Scheme._single = None
        scheme.mgr.state = scheme.CPUScheme()
        try:
            hp = FrequencySeries(np.ones(8, dtype=np.complex128), delta_f=0.25)
            hc = FrequencySeries(1j * np.ones(8, dtype=np.complex128), delta_f=0.25)
        finally:
            scheme.mgr.state = old_scheme
            scheme.Scheme._single = old_single
        return hp, hc


class StaticTorchRFrameGenerator:
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


def test_fdomain_det_frame_generator_moves_outputs_to_torch(monkeypatch, torch_ctx):
    generator = _load_generator_module(monkeypatch)

    with torch_ctx:
        detgen = generator.FDomainDetFrameGenerator(
            StaticRFrameGenerator,
            epoch=0.0,
            detectors=None,
            delta_f=0.25,
        )
        out = detgen.generate()

    series = out["RF"]
    assert isinstance(series._data.tensor, torch.Tensor)
    assert series._data.tensor.device.type == "cpu"


def test_freq_grid_cache_and_apply_fseries_time_shift(torch_ctx):
    from pycbc.waveform.utils_torch import (
        _FREQ_GRID_CACHE,
        _get_freq_grid,
        apply_fseries_time_shift,
    )
    from pycbc.types.array_torch import TorchArrayData

    _FREQ_GRID_CACHE.clear()
    grid1 = _get_freq_grid(0, 32, torch.device("cpu"), torch.float64)
    assert (0, 32, torch.device("cpu"), torch.float64) in _FREQ_GRID_CACHE
    grid2 = _get_freq_grid(0, 32, torch.device("cpu"), torch.float64)
    assert grid1 is grid2

    with torch_ctx:
        raw = torch.randn(32, dtype=torch.complex128)
        fs = FrequencySeries(
            TorchArrayData(raw.clone()), delta_f=0.5, epoch=0.0, copy=False
        )
        dt = 0.123
        shifted = apply_fseries_time_shift(fs, dt, kmin=4, copy=True)
        assert isinstance(shifted._data.tensor, torch.Tensor)

        # Verify against reference
        f = torch.arange(4, 32, dtype=torch.float64) * 0.5
        ref = raw.clone()
        ref[4:] = ref[4:] * torch.exp(-2j * torch.pi * dt * f)
        assert torch.allclose(shifted._data.tensor, ref, atol=1e-14)


def test_apply_fseries_time_shift_uses_explicit_batch_axis(torch_ctx):
    from pycbc.waveform.utils_torch import apply_fseries_time_shift

    with torch_ctx:
        # Deliberately make batch size equal the number of frequency bins.
        batch_size = n_freq = 4
        raw = torch.ones(
            (batch_size, n_freq), dtype=torch.complex128
        )
        series = FrequencySeries(
            TorchArrayData(raw.clone()), delta_f=0.5, epoch=0.0, copy=False
        )
        dt = torch.linspace(0.01, 0.04, batch_size, dtype=torch.float64)

        shifted = apply_fseries_time_shift(
            series, dt, copy=False
        )._data.tensor
        frequencies = torch.arange(n_freq, dtype=torch.float64) * 0.5
        expected = raw * torch.exp(
            -2j * torch.pi * dt[:, None] * frequencies
        )
        torch.testing.assert_close(shifted, expected)


def test_apply_fseries_time_shift_rejects_new_batch_axis(torch_ctx):
    from pycbc.waveform.utils_torch import apply_fseries_time_shift

    with torch_ctx:
        series = FrequencySeries(
            TorchArrayData(torch.ones(4, dtype=torch.complex128)),
            delta_f=0.5,
            epoch=0.0,
            copy=False,
        )
        with pytest.raises(ValueError, match="cannot introduce sample axes"):
            apply_fseries_time_shift(
                series, torch.linspace(0.01, 0.04, 4), copy=False
            )


def test_arrival_time_is_centered_before_float32_cast():
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


def test_fdomain_det_frame_generator_centers_numpy_large_gps_shift(
        monkeypatch):
    generator = _load_generator_module(monkeypatch)

    class OffsetDetector:
        name = 'H1'

        @staticmethod
        def time_delay_from_earth_center(ra, dec, tc):
            return 0.003

        @staticmethod
        def antenna_pattern(ra, dec, polarization, tc):
            return 1.0, 0.0

    epoch = 1126259460.0
    reference_time = 1126259462.125
    detgen = generator.FDomainDetFrameGenerator(
        StaticRFrameGenerator,
        epoch=epoch,
        detectors=['H1'],
        variable_args=['tc', 'ra', 'dec', 'polarization'],
        delta_f=0.25,
    )
    detgen.detectors = {'H1': OffsetDetector()}

    actual = detgen.generate(
        tc=reference_time, ra=1.0, dec=-0.5, polarization=0.2
    )['H1']
    dt = (reference_time - epoch) + 0.003
    frequencies = np.arange(len(actual)) * actual.delta_f
    expected = np.exp(-2j * np.pi * dt * frequencies)

    np.testing.assert_allclose(actual.numpy(), expected, rtol=0.0, atol=1e-12)


def test_large_float32_time_is_rejected_with_relative_epoch():
    from pycbc.waveform.generator import _arrival_time_and_shift

    with pytest.raises(ValueError, match="must use torch.float64"):
        _arrival_time_and_shift(
            torch.tensor(1126259462.0, dtype=torch.float32),
            0.003,
            epoch=0.0,
        )


def test_fused_detector_strain_fd_torch(torch_ctx):
    from pycbc.waveform.utils_torch import fused_detector_strain_fd_torch

    with torch_ctx:
        N = 64
        hp = torch.randn(N, dtype=torch.complex128)
        hc = torch.randn(N, dtype=torch.complex128)
        fp_list = [0.8, -0.4, 0.5]
        fc_list = [0.2, 0.7, -0.1]
        dt_list = [0.002, -0.005, 0.001]
        delta_f = 0.25

        strains = fused_detector_strain_fd_torch(
            hp, hc, fp_list, fc_list, dt_list, delta_f, kmin=2
        )
        assert strains.shape == (3, N)

        for i in range(len(fp_list)):
            ref = fp_list[i] * hp + fc_list[i] * hc
            f = torch.arange(2, N, dtype=torch.float64) * delta_f
            phase = torch.exp(-2j * torch.pi * f * dt_list[i])
            ref_shifted = ref.clone()
            ref_shifted[2:] = ref_shifted[2:] * phase
            assert torch.allclose(strains[i], ref_shifted, atol=1e-14)


def test_fused_detector_strain_has_explicit_sample_axis(torch_ctx):
    from pycbc.waveform.utils_torch import fused_detector_strain_fd_torch

    with torch_ctx:
        n_freq = 8
        hp = torch.ones(n_freq, dtype=torch.complex128)
        hc = 1j * torch.ones(n_freq, dtype=torch.complex128)
        # This deliberately makes the sample and frequency dimensions equal.
        fp = torch.linspace(0.2, 0.9, n_freq, dtype=torch.float64)
        fc = torch.linspace(-0.4, 0.3, n_freq, dtype=torch.float64)
        dt = torch.linspace(-0.01, 0.01, n_freq, dtype=torch.float64)

        actual = fused_detector_strain_fd_torch(
            hp, hc, [fp], [fc], [dt], delta_f=0.5
        )
        frequencies = torch.arange(n_freq, dtype=torch.float64) * 0.5
        expected = (fp[:, None] * hp + fc[:, None] * hc) * torch.exp(
            -2j * torch.pi * dt[:, None] * frequencies
        )
        assert actual.shape == (1, n_freq, n_freq)
        torch.testing.assert_close(actual[0], expected)


def test_fdomain_det_frame_generator_preserves_projection_gradients(
        monkeypatch, torch_ctx):
    generator = _load_generator_module(monkeypatch)

    class DifferentiableDetector:
        def __init__(self):
            self.arrival_time_value = None
            self.antenna_time_value = None

        def arrival_time(self, ref_tc, ra, dec, _refframe):
            self.arrival_time_value = ref_tc + 0.02 * ra - 0.03 * dec
            return self.arrival_time_value

        def antenna_pattern(self, ra, dec, polarization, tc):
            self.antenna_time_value = tc
            return (
                torch.cos(2.0 * polarization) + 0.01 * ra + 0.001 * tc,
                torch.sin(2.0 * polarization) + 0.01 * dec - 0.001 * tc,
            )

    with torch_ctx:
        detector = DifferentiableDetector()
        detgen = generator.FDomainDetFrameGenerator(
            StaticTorchRFrameGenerator,
            epoch=0.0,
            detectors=['H1'],
            variable_args=['tc', 'ra', 'dec', 'polarization'],
            delta_f=0.25,
        )
        detgen.detectors = {'H1': detector}
        tc = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        ra = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)
        dec = torch.tensor(-0.4, dtype=torch.float64, requires_grad=True)
        polarization = torch.tensor(
            0.3, dtype=torch.float64, requires_grad=True
        )

        result = detgen.generate(
            tc=tc, ra=ra, dec=dec, polarization=polarization
        )['H1']._data.tensor
        (result.real.sum() + result.imag.sum()).backward()

        assert result.device == tc.device
        torch.testing.assert_close(
            detector.antenna_time_value,
            detector.arrival_time_value,
        )
        for value in (tc, ra, dec, polarization):
            assert value.grad is not None
            assert torch.isfinite(value.grad)
            assert value.grad != 0.0


def test_fdomain_det_frame_generator_rejects_batched_location_args(
        monkeypatch, torch_ctx):
    generator = _load_generator_module(monkeypatch)

    with torch_ctx:
        detgen = generator.FDomainDetFrameGenerator(
            StaticTorchRFrameGenerator,
            epoch=0.0,
            detectors=None,
            variable_args=['tc'],
            delta_f=0.25,
        )
        with pytest.raises(ValueError, match="does not return a batch container"):
            detgen.generate(tc=torch.tensor([1.0, 2.0]))
        with pytest.raises(ValueError, match="tc_ref_frame"):
            detgen.generate(
                tc=1.0,
                tc_ref_frame=np.asarray(['H1', 'L1']),
            )


def test_fdomain_det_frame_generator_multi_detector_torch(monkeypatch, torch_ctx):
    generator = _load_generator_module(monkeypatch)

    with torch_ctx:
        detgen = generator.FDomainDetFrameGenerator(
            StaticRFrameGenerator,
            epoch=0.0,
            detectors=['H1', 'L1'],
            variable_args=['tc', 'ra', 'dec', 'polarization'],
            delta_f=0.25,
        )
        out = detgen.generate(
            tc=1126259462.0, ra=1.37, dec=-1.26, polarization=2.76
        )

        assert 'H1' in out and 'L1' in out
        for ifo in ['H1', 'L1']:
            series = out[ifo]
            assert isinstance(series, FrequencySeries)
            assert isinstance(series._data.tensor, torch.Tensor)
            assert series._data.tensor.device.type == "cpu"
            assert len(series) == 8


def test_sum_modes_torch(torch_ctx):
    from pycbc.waveform.waveform_modes import sum_modes
    from pycbc.types.array_torch import TorchArrayData

    with torch_ctx:
        modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]
        inclination = 0.7
        phi = 1.2
        N = 64
        hlms = {}
        for m in modes:
            t = torch.randn(N, dtype=torch.complex128)
            hlms[m] = FrequencySeries(
                TorchArrayData(t), delta_f=0.5, epoch=0.0, copy=False
            )

        res = sum_modes(hlms, inclination, phi)
        assert isinstance(res, FrequencySeries)
        assert isinstance(res._data.tensor, torch.Tensor)
        assert len(res) == N
