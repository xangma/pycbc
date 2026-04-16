import importlib
import sys
import types

import numpy as np
import pytest
import torch

import pycbc
from pycbc import scheme
from pycbc.types import FrequencySeries

pytest.importorskip("torch")
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
