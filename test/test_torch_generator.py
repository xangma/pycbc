import importlib
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pycbc
from pycbc import scheme
from pycbc.types import FrequencySeries

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

