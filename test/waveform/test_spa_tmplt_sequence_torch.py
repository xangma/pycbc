import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.types import Array  # noqa: E402
from pycbc.types.array_torch import TorchArrayData  # noqa: E402
from pycbc.waveform import spa_tmplt_cpu  # noqa: E402

spa_module = importlib.import_module("pycbc.waveform.spa_tmplt")


@pytest.fixture(autouse=True)
def preserve_scheme():
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        yield
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


@pytest.fixture(params=("cpu", "cuda", "mps"))
def torch_device(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")
    return device


def _activate_scheme(state):
    _scheme.Scheme._single = None
    _scheme.mgr.state = state


SAMPLE_POINTS = np.array(
    [100.0, 20.0, 23.5, 50.0, 400.0, 1000.0], dtype=np.float32
)

CASES = (
    {
        "mass1": 30.0,
        "mass2": 20.0,
        "spin1z": 0.3,
        "spin2z": -0.1,
        "distance": 500.0,
        "phase_order": -1,
        "spin_order": -1,
    },
    {
        "mass1": 10.0,
        "mass2": 8.0,
        "spin1z": 0.0,
        "spin2z": 0.4,
        "distance": 400.0,
        "phase_order": 7,
        "spin_order": 5,
    },
    {
        "mass1": 1.4,
        "mass2": 1.3,
        "spin1z": 0.02,
        "spin2z": -0.01,
        "distance": 100.0,
        "phase_order": -1,
        "spin_order": -1,
    },
)


def _cpu_reference(monkeypatch, params):
    monkeypatch.setenv("PYCBC_SPATPLT_NATIVE", "0")
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    return spa_module.spa_tmplt(
        sample_points=SAMPLE_POINTS, **params
    ).copy()


@pytest.mark.parametrize("params", CASES)
def test_spa_sequence_matches_cpu_and_stays_on_device(
    monkeypatch, torch_device, params
):
    expected = _cpu_reference(monkeypatch, params)

    monkeypatch.setenv("PYCBC_SPATPLT_NATIVE", "1")
    # Isolate synthesis parity here; native phasing is covered separately.
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "0")
    _activate_scheme(_scheme.TorchScheme(torch_device))
    sample_points = Array(SAMPLE_POINTS)

    def reject_host_transfer(_self):
        raise AssertionError("native SPAtmplt copied samples to the host")

    def reject_cpu_sequence(*_args, **_kwargs):
        raise AssertionError("native SPAtmplt called its CPU sequence kernel")

    with monkeypatch.context() as patch:
        patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
        patch.setattr(
            spa_tmplt_cpu,
            "spa_tmplt_inline_sequence",
            reject_cpu_sequence,
        )
        actual = spa_module.spa_tmplt(
            sample_points=sample_points, **params
        )

    assert isinstance(actual, Array)
    assert actual._data.tensor.device.type == torch_device
    assert actual._data.tensor.dtype == torch.complex64
    actual_values = actual._data.tensor.detach().cpu().numpy()
    relative_error = np.linalg.norm(
        actual_values.astype(np.complex128)
        - expected.astype(np.complex128)
    ) / np.linalg.norm(expected.astype(np.complex128))
    tolerance = 6.0e-3 if torch_device == "mps" else 6.0e-4
    assert relative_error < tolerance


def test_spa_sequence_full_native_path_avoids_lalsimulation(
    monkeypatch, torch_device
):
    params = dict(CASES[1], phase_order=4)
    expected = _cpu_reference(monkeypatch, params)

    monkeypatch.delenv("PYCBC_SPATPLT_NATIVE")
    monkeypatch.delenv("PYCBC_TAYLORF2_NATIVE")
    monkeypatch.delenv("PYCBC_TORCH_NATIVE_PORTS", raising=False)
    monkeypatch.delenv("PYCBC_TORCH_NATIVE", raising=False)
    _activate_scheme(_scheme.TorchScheme(torch_device))
    real_dtype = torch.float32 if torch_device == "mps" else torch.float64
    sample_points = torch.tensor(
        SAMPLE_POINTS, dtype=real_dtype, device=torch_device
    )

    def reject_cpu_path(*_args, **_kwargs):
        raise AssertionError("native SPAtmplt reached LAL or its CPU kernel")

    with monkeypatch.context() as patch:
        patch.setattr(
            spa_tmplt_cpu,
            "spa_tmplt_inline_sequence",
            reject_cpu_path,
        )
        patch.setattr(spa_module.lal, "CreateDict", reject_cpu_path)
        patch.setattr(
            spa_module.lalsimulation,
            "SimInspiralTaylorF2AlignedPhasing",
            reject_cpu_path,
        )
        patch.setattr(
            spa_module.lalsimulation,
            "SimInspiralWaveformParamsInsertPNPhaseOrder",
            reject_cpu_path,
        )
        patch.setattr(
            spa_module.lalsimulation,
            "SimInspiralWaveformParamsInsertPNSpinOrder",
            reject_cpu_path,
        )
        actual = spa_module.spa_tmplt(
            sample_points=sample_points, **params
        )

    assert actual._data.tensor.device.type == torch_device
    assert bool(torch.all(torch.isfinite(actual._data.tensor)))
    actual_values = actual._data.tensor.detach().cpu().numpy()
    relative_error = np.linalg.norm(
        actual_values.astype(np.complex128)
        - expected.astype(np.complex128)
    ) / np.linalg.norm(expected.astype(np.complex128))
    tolerance = 6.0e-3 if torch_device == "mps" else 6.0e-4
    assert relative_error < tolerance


def test_spa_sequence_legacy_fallback_accepts_torch_array(monkeypatch):
    expected = _cpu_reference(monkeypatch, CASES[0])

    _activate_scheme(_scheme.TorchScheme("cpu"))
    sample_points = Array(SAMPLE_POINTS.astype(np.float64))
    actual = spa_module.spa_tmplt(
        sample_points=sample_points, **CASES[0]
    )

    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.complex64
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("sample_points", "message"),
    (
        ([], "non-empty vector"),
        ([[20.0, 30.0]], "non-empty vector"),
        ([20.0, np.inf], "finite"),
        ([20.0, 0.0], "positive"),
    ),
)
def test_spa_sequence_rejects_invalid_frequencies(
    monkeypatch, sample_points, message
):
    monkeypatch.setenv("PYCBC_SPATPLT_NATIVE", "1")
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))

    with pytest.raises(ValueError, match=message):
        spa_module.spa_tmplt(
            sample_points=sample_points, **CASES[0]
        )
