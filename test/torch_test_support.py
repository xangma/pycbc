"""Shared fixtures and builders for opt-in Torch test modules.

This module is imported only by tests that already require Torch. Keeping the
fixtures here avoids enabling or importing Torch from the repository-wide
``conftest.py``.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.detector import Detector
from pycbc.inference.models import single_template
from pycbc.strain.strain import gate_data
from pycbc.types import TimeSeries


@pytest.fixture(autouse=True)
def stub_execute_cached_fft_import(monkeypatch):
    """Avoid importing the full strain stack in Welch tests."""
    fake_pkg = types.ModuleType("pycbc.strain")
    fake_mod = types.ModuleType("pycbc.strain.strain")

    def _execute_cached_fft(*args, **kwargs):
        raise AssertionError("Welch cache path should not be used in this test")

    def _execute_cached_ifft(*args, **kwargs):
        raise AssertionError("PSD cache path should not be used in this test")

    def _create_memory_and_engine_for_class_based_fft(*args, **kwargs):
        raise AssertionError("Welch cache path should not be used in this test")

    fake_mod.execute_cached_fft = _execute_cached_fft
    fake_mod.execute_cached_ifft = _execute_cached_ifft
    fake_mod.create_memory_and_engine_for_class_based_fft = (
        _create_memory_and_engine_for_class_based_fft
    )
    fake_pkg.gate_data = gate_data
    monkeypatch.setitem(sys.modules, "pycbc.strain", fake_pkg)
    monkeypatch.setitem(sys.modules, "pycbc.strain.strain", fake_mod)


@pytest.fixture
def torch_ctx():
    ctx = scheme.TorchScheme("cpu")
    try:
        yield ctx
    finally:
        del ctx
        scheme.Scheme._single = None


@pytest.fixture(params=("cpu", "cuda", "mps"))
def torch_device_ctx(request):
    import torch

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


def _relative_l2(actual, expected):
    diff = actual - expected
    return np.linalg.norm(diff) / np.linalg.norm(expected)


def _load_inference_model_module(name):
    """Load a model module without inference's optional dependencies."""
    module_path = (
        Path(pycbc.__file__).parent / "inference" / "models" / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"_pycbc_inference_{name}_torch_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native_scalar_single_template_model(
        dtype=np.complex128, detectors=("H1",)):
    """Build the minimum real-detector model needed by the scalar path."""
    epoch = 1126259462.0
    delta_t = 1.0 / 128.0
    samples = 1025
    indices = np.arange(samples, dtype=np.float64)
    model = object.__new__(single_template.SingleTemplate)
    model._current_params = {
        "inclination": np.float64(0.71),
        "polarization": np.float64(0.39),
        "coa_phase": np.float64(0.23),
        "distance": np.float64(1.8),
        "ra": np.float64(1.0),
        "dec": np.float64(-0.4),
        "tc": np.float64(epoch + 4.003),
    }
    scales = {"H1": 1.0, "L1": 1.3}
    model.sh = {
        ifo: TimeSeries(
            (
                (1.0 + scale * 0.002 * indices)
                * np.exp(scale * 0.017j * indices)
            ).astype(dtype),
            delta_t=delta_t,
            epoch=epoch,
        )
        for ifo, scale in ((ifo, scales[ifo]) for ifo in detectors)
    }
    norms = {"H1": 1.7, "L1": 2.1}
    model.hh = {ifo: np.float64(norms[ifo]) for ifo in detectors}
    model.snr = {}
    model.det = {ifo: Detector(ifo) for ifo in model.sh}
    model.dts = {}
    model.htfs = {}
    model._sh_storage_is_host = False
    model.marginalize_phase = False
    model.marginalize_distance = False
    model.distance_marginalization = False
    model.distance_interpolator = None
    model.marginalize_vector_params = {}
    model.vsamples = 1
    model.marginalize_vector_weights = 0.0
    model.reconstruct_phase = False
    model.reconstruct_distance = False
    model.reconstruct_vector = False
    model.snr_draw = lambda **_kwargs: None
    return model
