import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme
from pycbc.waveform import get_fd_waveform
from pycbc.waveform.taylorf2ecc_torch import taylorf2ecc_native_supported


@pytest.fixture
def preserve_scheme():
    """Restore the process-wide PyCBC scheme singleton after a test."""
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        yield
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def _activate_scheme(scheme_type, *args):
    _scheme.Scheme._single = None
    _scheme.mgr.state = scheme_type(*args)


def _reference_and_native(params, monkeypatch):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORF2ECC_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme)
    reference = get_fd_waveform(approximant="TaylorF2Ecc", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_TAYLORF2ECC_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme)
    native = get_fd_waveform(approximant="TaylorF2Ecc", **params)
    return reference, reference_arrays, native


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=30.0,
            mass2=20.0,
            spin1z=0.3,
            spin2z=-0.1,
            delta_f=0.5,
            f_lower=20.0,
            distance=500.0,
            eccentricity=0.1,
        ),
        dict(
            mass1=10.0,
            mass2=8.0,
            spin2z=0.4,
            delta_f=0.25,
            f_lower=15.0,
            f_final=400.0,
            f_ref=25.0,
            distance=400.0,
            eccentricity=0.07,
            eccentricity_order=4,
            phase_order=5,
            spin_order=3,
            inclination=1.2,
            coa_phase=0.3,
            long_asc_nodes=0.37,
        ),
        dict(
            mass1=1.4,
            mass2=1.3,
            spin1z=0.02,
            spin2z=-0.01,
            delta_f=1.0,
            f_lower=20.0,
            f_final=500.0,
            f_ref=30.0,
            distance=100.0,
            eccentricity=0.03,
            lambda1=800.0,
            lambda2=700.0,
            dquad_mon1=0.0,
            dquad_mon2=0.0,
            tidal_order=15,
            dchi3=0.02,
            dchi6l=-0.01,
        ),
        dict(
            mass1=2.0,
            mass2=1.6,
            delta_f=1.0,
            f_lower=20.0,
            f_final=300.0,
            distance=150.0,
            eccentricity=0.0,
            inclination=0.8,
            long_asc_nodes=0.2,
        ),
    ],
)
def test_taylorf2ecc_public_torch_parity(
    params, monkeypatch, preserve_scheme
):
    reference, reference_arrays, native = _reference_and_native(
        params, monkeypatch
    )

    for expected, expected_array, actual in zip(
        reference, reference_arrays, native
    ):
        assert len(actual) == len(expected)
        assert actual.delta_f == expected.delta_f
        assert float(actual.epoch) == float(expected.epoch)
        assert actual._data.tensor.device.type == "cpu"
        np.testing.assert_array_equal(
            actual.numpy() == 0.0,
            expected_array == 0.0,
        )
        nonzero = np.abs(expected_array) > 0.0
        relative_error = np.linalg.norm(
            actual.numpy()[nonzero] - expected_array[nonzero]
        ) / np.linalg.norm(expected_array[nonzero])
        assert relative_error < 1.0e-11


@pytest.mark.parametrize("eccentricity_order", [-1, 0, 1, 2, 3, 4, 5, 6])
def test_taylorf2ecc_all_eccentricity_orders_match_lal(
    eccentricity_order, monkeypatch, preserve_scheme
):
    params = dict(
        mass1=12.0,
        mass2=7.0,
        spin1z=0.15,
        spin2z=-0.08,
        delta_f=1.0,
        f_lower=20.0,
        f_final=180.0,
        f_ref=31.0,
        distance=300.0,
        eccentricity=0.08,
        eccentricity_order=eccentricity_order,
    )
    _, reference_arrays, native = _reference_and_native(params, monkeypatch)

    for expected, actual in zip(reference_arrays, native):
        nonzero = np.abs(expected) > 0.0
        relative_error = np.linalg.norm(
            actual.numpy()[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < 1.0e-11


def test_taylorf2ecc_public_dispatches_native(monkeypatch, preserve_scheme):
    params = dict(
        mass1=20.0,
        mass2=15.0,
        delta_f=1.0,
        f_lower=20.0,
        f_final=150.0,
        eccentricity=0.05,
    )
    import pycbc.waveform.taylorf2ecc_torch as taylorf2ecc_mod

    native = taylorf2ecc_mod.taylorf2ecc_fd_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    monkeypatch.setattr(
        taylorf2ecc_mod, "taylorf2ecc_fd_torch", recording_native
    )
    monkeypatch.setenv("PYCBC_TAYLORF2ECC_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme)
    waveform = get_fd_waveform(approximant="TaylorF2Ecc", **params)

    assert calls == 1
    assert waveform[0]._data.tensor.device.type == "cpu"


def test_taylorf2ecc_unsupported_options_use_lal(
    monkeypatch, preserve_scheme
):
    params = dict(
        mass1=20.0,
        mass2=15.0,
        delta_f=1.0,
        f_lower=20.0,
        f_final=150.0,
        eccentricity=0.05,
        amplitude_order=2,
    )
    import pycbc.waveform.taylorf2ecc_torch as taylorf2ecc_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported TaylorF2Ecc parameters reached Torch")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        taylorf2ecc_mod, "taylorf2ecc_fd_torch", unexpected_native
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_TAYLORF2ECC_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme)
    waveform = get_fd_waveform(approximant="TaylorF2Ecc", **params)

    assert lal_calls == 1
    assert waveform[0]._data.tensor.device.type == "cpu"


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, True),
        ({"eccentricity": 0.2}, True),
        ({"eccentricity_order": 6}, True),
        ({"eccentricity": -0.1}, False),
        ({"eccentricity": 1.0}, False),
        ({"eccentricity_order": 7}, False),
        ({"amplitude_order": 2}, False),
        ({"spin1x": 0.1}, False),
        ({"lambda_octu1": 10.0}, False),
        ({"dalpha1": 0.1}, False),
    ],
)
def test_taylorf2ecc_native_support_boundary(params, expected):
    assert taylorf2ecc_native_supported(params) is expected


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_taylorf2ecc_public_torch_uses_mps(monkeypatch, preserve_scheme):
    params = dict(
        mass1=20.0,
        mass2=15.0,
        spin1z=0.2,
        spin2z=-0.1,
        delta_f=1.0,
        f_lower=20.0,
        f_final=180.0,
        f_ref=30.0,
        distance=400.0,
        eccentricity=0.05,
    )
    monkeypatch.setenv("PYCBC_TAYLORF2ECC_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme)
    reference, _ = get_fd_waveform(approximant="TaylorF2Ecc", **params)
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_TAYLORF2ECC_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme, "mps")
    actual, _ = get_fd_waveform(approximant="TaylorF2Ecc", **params)

    assert actual._data.tensor.device.type == "mps"
    assert actual._data.tensor.dtype == torch.complex64
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual.numpy()[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    assert relative_error < 1.0e-4
