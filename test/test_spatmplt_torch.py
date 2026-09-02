import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme
from pycbc.types import FrequencySeries, zeros
from pycbc.types.array_torch import TorchArrayData
import pycbc.waveform.spa_tmplt as spa_tmplt_module
from pycbc.waveform.spa_tmplt import (
    spa_distance,
    spa_tmplt_engine,
    spa_tmplt_norm,
    spa_tmplt_precondition,
)


def _torch_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture
def params():
    return dict(
        kmin=1,
        phase_order=7,
        delta_f=1.0 / 64.0,
        piM=np.pi * 20.0,
        pfaN=1.0,
        pfa2=0.1,
        pfa3=-0.05,
        pfa4=0.01,
        pfa5=0.02,
        pfl5=-0.03,
        pfa6=0.04,
        pfl6=0.01,
        pfa7=-0.02,
        amp_factor=1.0,
    )


def _run_engine(scheme_ctx, dtype, params):
    scheme.Scheme._single = None
    with scheme_ctx:
        htilde = zeros(256, dtype=dtype)
        spa_tmplt_engine(htilde, **params)
    # reset singleton to allow another scheme
    scheme.Scheme._single = None
    return htilde.numpy()


def test_spatmplt_torch_matches_cpu(params, monkeypatch):
    monkeypatch.setenv("PYCBC_SPATPLT_NATIVE", "1")
    scheme.Scheme._single = None
    cpu = _run_engine(scheme.CPUScheme(), np.complex64, params)
    torch_out = _run_engine(scheme.TorchScheme("cpu"), np.complex64, params)

    rel_l2 = np.linalg.norm(torch_out - cpu) / np.linalg.norm(cpu)
    assert rel_l2 < 1e-5


def test_spatmplt_torch_dtype_and_device(params, monkeypatch):
    monkeypatch.setenv("PYCBC_SPATPLT_NATIVE", "1")
    scheme.Scheme._single = None
    ctx = scheme.TorchScheme("cpu")
    with ctx:
        htilde = zeros(64, dtype=np.complex64)
        spa_tmplt_engine(htilde, **params)
        # Should remain complex64 on CPU device
        assert isinstance(htilde._data.tensor, torch.Tensor)
        assert htilde._data.tensor.dtype == torch.complex64
        assert htilde._data.tensor.device.type == "cpu"
    scheme.Scheme._single = None


def test_spatmplt_torch_is_native_by_default(params, monkeypatch):
    spa_torch = importlib.import_module("pycbc.waveform.spa_tmplt_torch")
    for name in (
        "PYCBC_TORCH_NATIVE_PORTS",
        "PYCBC_TORCH_NATIVE",
        "PYCBC_SPATPLT_NATIVE",
    ):
        monkeypatch.delenv(name, raising=False)

    def reject_cpu_fallback(*_args, **_kwargs):
        raise AssertionError("default SPAtmplt path copied through the CPU kernel")

    monkeypatch.setattr(spa_torch, "_cpu_reference", reject_cpu_fallback)
    scheme.Scheme._single = None
    with scheme.TorchScheme("cpu"):
        htilde = zeros(64, dtype=np.complex64)
        spa_tmplt_engine(htilde, **params)
        assert isinstance(htilde._data.tensor, torch.Tensor)
    scheme.Scheme._single = None


@pytest.mark.parametrize("device", _torch_devices())
def test_spatmplt_native_frequency_grid_stays_on_device(
    params, monkeypatch, device
):
    spa_torch = importlib.import_module("pycbc.waveform.spa_tmplt_torch")

    def reject_numpy_grid(*_args, **_kwargs):
        raise AssertionError("native SPA built a bulk frequency grid with NumPy")

    for operation in ("arange", "cbrt", "log", "power"):
        monkeypatch.setattr(spa_torch._np, operation, reject_numpy_grid)

    native_params = {
        key: value for key, value in params.items() if key != "phase_order"
    }
    result = spa_torch._torch_native_spa(
        64,
        device=device,
        dtype_out=torch.complex64,
        **native_params,
    )

    for key in ("v", "logv", "kfac", "out"):
        assert result[key].device.type == device
        assert torch.isfinite(result[key]).all()


@pytest.mark.parametrize("device", _torch_devices())
def test_spatmplt_norm_stays_on_torch_device(monkeypatch, device):
    length = 513
    delta_f = 0.25
    f_lower = 20.0
    psd_values = np.linspace(1.0, 4.0, length, dtype=np.float32)

    monkeypatch.setattr(spa_tmplt_module, "_prec", None)
    monkeypatch.setattr(spa_tmplt_module, "_torch_prec", {})

    scheme.Scheme._single = None
    with scheme.CPUScheme():
        cpu_psd = FrequencySeries(psd_values, delta_f=delta_f)
        cpu_prec = spa_tmplt_precondition(length, delta_f).numpy()
        cpu_norm = spa_tmplt_norm(
            cpu_psd, length, delta_f, f_lower
        )
        cpu_distance = spa_distance(
            cpu_psd, 1.4, 1.4, f_lower
        )
    scheme.Scheme._single = None

    def fail_numpy(self):
        raise AssertionError("SPA normalization copied Torch data to NumPy")

    with scheme.TorchScheme(device):
        torch_psd = FrequencySeries(psd_values, delta_f=delta_f)
        monkeypatch.setattr(TorchArrayData, "numpy", fail_numpy)
        torch_prec = spa_tmplt_precondition(length, delta_f)
        torch_norm = spa_tmplt_norm(
            torch_psd, length, delta_f, f_lower
        )
        torch_distance = spa_distance(
            torch_psd, 1.4, 1.4, f_lower
        )

        assert torch_prec._data.tensor.device.type == device
        assert torch_norm._data.tensor.device.type == device
        prec_tensor = torch_prec._data.tensor.detach().cpu()
        norm_tensor = torch_norm._data.tensor.detach().cpu()
    scheme.Scheme._single = None

    torch.testing.assert_close(
        prec_tensor,
        torch.as_tensor(cpu_prec, dtype=prec_tensor.dtype),
        rtol=2e-5,
        atol=1e-7,
    )
    torch.testing.assert_close(
        norm_tensor,
        torch.as_tensor(cpu_norm, dtype=norm_tensor.dtype),
        rtol=2e-5,
        atol=1e-7,
    )
    assert torch_distance == pytest.approx(cpu_distance, rel=2e-5)
