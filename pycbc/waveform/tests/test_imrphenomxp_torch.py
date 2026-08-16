import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenomxp_torch import (  # noqa: E402
    imrphenomxp_native_supported,
    imrphenomxp_sequence_native_supported,
)


_MODEL_FLAGS = dict(
    phenom_x_prec_version=102,
    phenom_xp_convention=0,
    phenom_xp_final_spin_mod=0,
)


@pytest.fixture
def preserve_scheme():
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        yield
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def _activate_scheme(scheme):
    _scheme.Scheme._single = None
    _scheme.mgr.state = scheme


def _relative_error(actual, expected):
    nonzero = np.abs(expected) > 0.0
    assert nonzero.any()
    return np.linalg.norm(actual[nonzero] - expected[nonzero]) / np.linalg.norm(
        expected[nonzero]
    )


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=40.0,
            mass2=20.0,
            spin1x=0.2,
            spin1y=0.1,
            spin1z=0.3,
            spin2x=-0.1,
            spin2y=0.05,
            spin2z=-0.2,
            distance=500.0,
            inclination=0.7,
            coa_phase=1.2,
            long_asc_nodes=0.3,
            delta_f=0.5,
            f_lower=20.0,
            f_final=512.0,
            f_ref=30.0,
        ),
        # Exercises component reordering, f_ref=0, and non-bin frequencies.
        dict(
            mass1=12.0,
            mass2=35.0,
            spin1x=0.15,
            spin1y=-0.25,
            spin1z=0.4,
            spin2x=0.05,
            spin2y=0.2,
            spin2z=-0.3,
            distance=320.0,
            inclination=1.1,
            coa_phase=0.0,
            long_asc_nodes=-0.4,
            delta_f=0.25,
            f_lower=17.3,
            f_final=900.3,
            f_ref=0.0,
        ),
        # The aligned-spin limit takes a special source-frame branch.
        dict(
            mass1=30.0,
            mass2=30.0,
            spin1z=0.2,
            spin2z=-0.1,
            distance=800.0,
            inclination=0.2,
            coa_phase=2.1,
            delta_f=0.5,
            f_lower=20.0,
            f_ref=20.0,
        ),
    ],
)
def test_imrphenomxp_torch_matches_lalsimulation(
    params,
    monkeypatch,
    preserve_scheme,
):
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(
        approximant="IMRPhenomXP",
        **_MODEL_FLAGS,
        **params,
    )
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(
        approximant="IMRPhenomXP",
        **_MODEL_FLAGS,
        **params,
    )

    for expected, expected_array, result in zip(
        reference,
        reference_arrays,
        actual,
    ):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected_array == 0.0)
        assert _relative_error(result_array, expected_array) < 2.0e-12


def test_imrphenomxp_sequence_matches_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=12.0,
        mass2=35.0,
        spin1x=0.15,
        spin1y=-0.25,
        spin1z=0.4,
        spin2x=0.05,
        spin2y=0.2,
        spin2z=-0.3,
        distance=320.0,
        inclination=1.1,
        coa_phase=0.0,
        long_asc_nodes=-0.4,
        f_ref=0.0,
    )
    sample_points = [17.3, 22.0, 150.0, 400.0, 850.0, 1000.0]
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXP",
        sample_points=sample_points,
        **_MODEL_FLAGS,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.waveform as waveform_mod

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXP sequence called lalsimulation")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomXP",
        sample_points=sample_points,
        **_MODEL_FLAGS,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < 2.0e-12


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, False),
        (_MODEL_FLAGS, True),
        (dict(_MODEL_FLAGS, phenom_x_prec_version=223), False),
        (dict(_MODEL_FLAGS, phenom_xp_convention=1), False),
        (dict(_MODEL_FLAGS, phenom_xp_final_spin_mod=3), False),
        (dict(_MODEL_FLAGS, lambda1=100.0), False),
        (dict(_MODEL_FLAGS, dchi3=0.1), False),
        (dict(_MODEL_FLAGS, eccentricity=0.1), False),
        (dict(_MODEL_FLAGS, spin_order=4), False),
        (dict(_MODEL_FLAGS, mode_array=[(2, 2)]), False),
        (dict(_MODEL_FLAGS, frame_axis=1), False),
        (dict(_MODEL_FLAGS, numrel_data="waveform.h5"), False),
        (dict(_MODEL_FLAGS, approximant="IMRPhenomXAS"), False),
    ],
)
def test_imrphenomxp_native_support_boundary(params, expected):
    assert imrphenomxp_native_supported(params) is expected
    assert imrphenomxp_sequence_native_supported(params) is expected


def test_imrphenomxp_public_native_dispatch_avoids_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=35.0,
        mass2=22.0,
        spin1x=0.2,
        spin1y=-0.15,
        spin1z=0.3,
        spin2x=0.1,
        spin2y=0.05,
        spin2z=-0.2,
        distance=500.0,
        inclination=0.8,
        coa_phase=1.1,
        long_asc_nodes=0.37,
        delta_f=0.5,
        f_lower=20.0,
        f_ref=30.0,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(
        approximant="IMRPhenomXP",
        **_MODEL_FLAGS,
        **params,
    )
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.waveform as waveform_mod

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXP called lalsimulation")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(
        approximant="IMRPhenomXP",
        **_MODEL_FLAGS,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert _relative_error(result.numpy(), expected) < 2.0e-12


def test_imrphenomxp_default_configuration_uses_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=35.0,
        mass2=20.0,
        spin1x=0.1,
        spin1z=0.2,
        spin2y=0.1,
        spin2z=-0.1,
        distance=500.0,
        delta_f=1.0,
        f_lower=20.0,
    )
    import pycbc.waveform.imrphenomxp_torch as xp_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("default IMRPhenomXP reached the bounded native path")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(xp_mod, "imrphenomxp_fd_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    result = get_fd_waveform(approximant="IMRPhenomXP", **params)

    assert lal_calls == 1
    assert all(series._data.tensor.device.type == "cpu" for series in result)


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomxp_native_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = dict(
        mass1=35.0,
        mass2=20.0,
        spin1x=0.2,
        spin1y=-0.15,
        spin1z=0.3,
        spin2x=0.1,
        spin2y=0.05,
        spin2z=-0.2,
        distance=500.0,
        inclination=0.8,
        coa_phase=1.1,
        delta_f=1.0,
        f_lower=20.0,
        f_ref=30.0,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(
        approximant="IMRPhenomXP",
        **_MODEL_FLAGS,
        **params,
    )
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual = get_fd_waveform(
        approximant="IMRPhenomXP",
        **_MODEL_FLAGS,
        **params,
    )

    expected_dtype = torch.complex64 if device_name == "mps" else torch.complex128
    tolerance = 4.0e-3 if device_name == "mps" else 2.0e-12
    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == device_name
        assert result._data.tensor.dtype == expected_dtype
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < tolerance
