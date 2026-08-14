import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_fd_waveform  # noqa: E402
from pycbc.waveform.spintaylorf2_torch import (  # noqa: E402
    spintaylorf2_native_supported,
)


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


def _activate_scheme(state):
    _scheme.Scheme._single = None
    _scheme.mgr.state = state


def _assert_waveform_matches(reference, actual, tolerance, reference_array=None):
    assert len(actual) == len(reference)
    assert actual.delta_f == reference.delta_f
    assert float(actual.epoch) == float(reference.epoch)

    if reference_array is None:
        reference_array = reference.numpy()
    actual_array = actual.numpy()
    np.testing.assert_array_equal(
        actual_array == 0.0,
        reference_array == 0.0,
    )
    nonzero = np.abs(reference_array) > 0.0
    if not nonzero.any():
        assert np.count_nonzero(actual_array) == 0
        return
    relative_error = np.linalg.norm(
        actual_array[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    assert relative_error < tolerance


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=10.0,
            mass2=8.0,
            spin1x=0.1,
            spin1z=0.4,
            inclination=0.3,
            coa_phase=2.0,
            delta_f=0.2,
            f_lower=15.0,
            distance=400.0,
            amplitude_order=7,
            phase_order=7,
            spin_order=5,
            f_ref=0.0,
            side_bands=0,
            dchi0=0.02,
            dchi1=-0.01,
            dchi2=0.015,
            dchi3=-0.02,
            dchi4=0.012,
            dchi5=-0.008,
            dchi5l=0.006,
            dchi6=-0.01,
            dchi6l=0.007,
            dchi7=-0.005,
        ),
        dict(
            mass1=12.0,
            mass2=7.0,
            spin1x=-0.18,
            spin1y=0.12,
            spin1z=0.35,
            inclination=1.1,
            coa_phase=0.4,
            long_asc_nodes=0.37,
            delta_f=0.5,
            f_lower=18.0,
            f_final=220.0,
            f_ref=31.0,
            distance=300.0,
            phase_order=6,
            spin_order=4,
            side_bands=2,
            dquad_mon1=1.3,
            dchi3=0.02,
            dchi6l=-0.01,
        ),
        dict(
            mass1=30.0,
            mass2=20.0,
            spin1x=0.2,
            spin1y=0.1,
            spin1z=0.3,
            inclination=0.7,
            coa_phase=1.0,
            long_asc_nodes=-0.2,
            delta_f=0.25,
            f_lower=20.0,
            f_final=180.0,
            f_ref=30.0,
            distance=500.0,
            phase_order=2,
            spin_order=0,
            side_bands=-1,
            dchi1=0.01,
            dchi5l=-0.03,
        ),
        dict(
            mass1=15.0,
            mass2=9.0,
            inclination=0.8,
            coa_phase=0.3,
            delta_f=0.5,
            f_lower=20.0,
            f_final=180.0,
            f_ref=30.0,
            distance=400.0,
        ),
        dict(
            mass1=15.0,
            mass2=9.0,
            spin1z=-0.4,
            inclination=0.8,
            coa_phase=0.3,
            delta_f=0.5,
            f_lower=20.0,
            f_final=180.0,
            f_ref=30.0,
            distance=400.0,
            side_bands=1,
        ),
    ],
)
def test_spintaylorf2_public_torch_parity_and_dispatch(
    params, monkeypatch, preserve_scheme
):
    """The public API selects native Torch and retains LAL parity."""
    monkeypatch.setenv("PYCBC_SPINTAYLORF2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="SpinTaylorF2", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.spintaylorf2_torch as spintaylorf2_mod
    import pycbc.waveform.waveform as waveform_mod

    native = spintaylorf2_mod.spintaylorf2_torch
    native_calls = 0

    def recording_native(**native_params):
        nonlocal native_calls
        native_calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("supported SpinTaylorF2 parameters reached LAL")

    monkeypatch.setattr(
        spintaylorf2_mod,
        "spintaylorf2_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_SPINTAYLORF2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="SpinTaylorF2", **params)

    assert native_calls == 1
    for reference_series, reference_array, actual_series in zip(
        reference, reference_arrays, actual
    ):
        assert actual_series._data.tensor.device.type == "cpu"
        assert actual_series._data.tensor.dtype == torch.complex128
        _assert_waveform_matches(
            reference_series,
            actual_series,
            1.0e-11,
            reference_array,
        )


def test_spintaylorf2_disabled_native_uses_lal(monkeypatch, preserve_scheme):
    params = dict(
        mass1=10.0,
        mass2=8.0,
        spin1x=0.1,
        spin1z=0.4,
        inclination=0.3,
        delta_f=0.5,
        f_lower=15.0,
        distance=400.0,
    )
    import pycbc.waveform.spintaylorf2_torch as spintaylorf2_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("disabled SpinTaylorF2 native path was called")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        spintaylorf2_mod,
        "spintaylorf2_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "1")
    monkeypatch.setenv("PYCBC_SPINTAYLORF2_NATIVE", "0")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    waveform = get_fd_waveform(approximant="SpinTaylorF2", **params)

    assert lal_calls == 1
    assert waveform[0]._data.tensor.device.type == "cpu"


def test_spintaylorf2_unsupported_options_use_lal(monkeypatch, preserve_scheme):
    params = dict(
        mass1=10.0,
        mass2=8.0,
        spin1x=0.1,
        spin1z=0.4,
        inclination=0.3,
        delta_f=0.5,
        f_lower=15.0,
        distance=400.0,
        lambda1=100.0,
    )
    monkeypatch.setenv("PYCBC_SPINTAYLORF2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="SpinTaylorF2", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.spintaylorf2_torch as spintaylorf2_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported SpinTaylorF2 parameters reached Torch")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        spintaylorf2_mod,
        "spintaylorf2_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_SPINTAYLORF2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_fd_waveform(approximant="SpinTaylorF2", **params)

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert actual._data.tensor.device.type == "cpu"
        np.testing.assert_allclose(actual.numpy(), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, True),
        ({"amplitude_order": 7}, True),
        ({"spin_order": 1, "phase_order": 4}, True),
        ({"side_bands": 3, "dquad_mon1": 1.2, "dchi3": 0.1}, True),
        ({"spin2z": 0.1}, False),
        ({"lambda1": 100.0}, False),
        ({"tidal_order": 0}, False),
        ({"eccentricity": 0.1}, False),
        ({"frame_axis": 1}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"dalpha1": 0.1}, False),
    ],
)
def test_spintaylorf2_native_support_boundary(params, expected):
    assert spintaylorf2_native_supported(params) is expected


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_spintaylorf2_public_torch_uses_mps(monkeypatch, preserve_scheme):
    params = dict(
        mass1=10.0,
        mass2=8.0,
        spin1x=0.1,
        spin1y=0.05,
        spin1z=0.4,
        inclination=0.3,
        coa_phase=0.2,
        long_asc_nodes=0.2,
        delta_f=0.5,
        f_lower=15.0,
        f_ref=0.0,
        distance=400.0,
        side_bands=1,
    )
    monkeypatch.setenv("PYCBC_SPINTAYLORF2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="SpinTaylorF2", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_SPINTAYLORF2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("mps"))
    actual = get_fd_waveform(approximant="SpinTaylorF2", **params)

    for reference_series, reference_array, actual_series in zip(
        reference, reference_arrays, actual
    ):
        assert actual_series._data.tensor.device.type == "mps"
        assert actual_series._data.tensor.dtype == torch.complex64
        _assert_waveform_matches(
            reference_series,
            actual_series,
            3.0e-4,
            reference_array,
        )
