import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import pnutils, scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenompv3_torch import (  # noqa: E402
    imrphenompv3hm_native_supported,
    imrphenompv3hm_sequence_native_supported,
)


_NATIVE_FLAGS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_IMRPHENOMPV3HM_NATIVE",
)

REGULAR_CASES = (
    dict(
        mass1=40.0,
        mass2=20.0,
        spin1x=0.3,
        spin1y=-0.2,
        spin1z=0.45,
        spin2x=-0.1,
        spin2y=0.25,
        spin2z=-0.3,
        delta_f=1.0,
        f_lower=19.0,
        f_final=300.0,
        f_ref=30.0,
        distance=400.0,
        inclination=1.1,
        coa_phase=0.4,
    ),
    dict(
        mass1=18.0,
        mass2=47.0,
        spin1x=-0.12,
        spin1y=0.31,
        spin1z=-0.22,
        spin2x=0.4,
        spin2y=0.05,
        spin2z=0.37,
        delta_f=2.0,
        f_lower=19.75,
        f_final=522.0,
        f_ref=0.0,
        distance=720.0,
        inclination=2.0,
        coa_phase=-0.8,
        long_asc_nodes=0.63,
        mode_array=[(2, 2), (2, 1), (4, 3)],
    ),
    dict(
        mass1=80.0,
        mass2=8.0,
        spin1z=-0.99,
        spin2z=0.0,
        delta_f=1.0,
        f_lower=19.0,
        f_final=300.0,
        f_ref=20.0,
        distance=540.0,
        inclination=0.7,
        coa_phase=0.4,
        long_asc_nodes=-0.37,
        mode_array=[(2, 2), (2, 1), (3, 3), (4, 3)],
    ),
    dict(
        mass1=32.0,
        mass2=32.0,
        spin1z=0.25,
        spin2z=-0.38,
        delta_f=1.25,
        f_lower=18.75,
        f_final=301.3,
        f_ref=0.0,
        distance=610.0,
        inclination=1.4,
        coa_phase=-0.35,
        mode_array=[(2, 2), (2, 1), (3, 3), (4, 4)],
    ),
)

SUPPORT_BASE = {
    "approximant": "IMRPhenomPv3HM",
    **REGULAR_CASES[0],
    "sample_points": [20.0, 31.5, 80.0],
}


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


def _clear_native_flags(monkeypatch):
    for name in _NATIVE_FLAGS:
        monkeypatch.delenv(name, raising=False)


def _assert_close(
    reference,
    actual,
    *,
    tolerance=5.0e-11,
    expected_prefix=None,
):
    for expected, result in zip(reference, actual):
        expected_array = (
            expected.numpy()
            if hasattr(expected, "numpy")
            else np.asarray(expected.data.data)
        )
        if expected_prefix is not None:
            expected_array = expected_array[:expected_prefix]
        result_array = result.numpy()
        assert result._data.tensor.device.type == "cpu"
        assert result_array.shape == expected_array.shape
        scale = np.max(np.abs(expected_array))
        np.testing.assert_allclose(
            result_array,
            expected_array,
            rtol=tolerance,
            atol=scale * tolerance * 0.02,
        )


def _lal_frequencies(values):
    frequencies = lal.CreateREAL8Vector(len(values))
    frequencies.data[:] = values
    return frequencies


def _lal_mode_dictionary(modes):
    dictionary = lal.CreateDict()
    if modes is None:
        return dictionary
    mode_array = lalsimulation.SimInspiralCreateModeArray()
    for ell, emm in modes:
        lalsimulation.SimInspiralModeArrayActivateMode(mode_array, ell, emm)
    lalsimulation.SimInspiralWaveformParamsInsertModeArray(dictionary, mode_array)
    return dictionary


def _lal_pv3hm_sequence(params, sample_points):
    reference_frequency = params.get("f_ref", 0.0) or sample_points[0]
    return lalsimulation.SimIMRPhenomPv3HMGetHplusHcross(
        _lal_frequencies(sample_points),
        params["mass1"] * lal.MSUN_SI,
        params["mass2"] * lal.MSUN_SI,
        params.get("spin1x", 0.0),
        params.get("spin1y", 0.0),
        params.get("spin1z", 0.0),
        params.get("spin2x", 0.0),
        params.get("spin2y", 0.0),
        params.get("spin2z", 0.0),
        pnutils.megaparsecs_to_meters(params["distance"]),
        params["inclination"],
        params["coa_phase"],
        0.0,
        reference_frequency,
        _lal_mode_dictionary(params.get("mode_array")),
    )


@pytest.mark.parametrize("params", REGULAR_CASES)
def test_imrphenompv3hm_regular_matches_lal_without_lalsimulation(
    params,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.waveform as waveform

    monkeypatch.setenv("PYCBC_IMRPHENOMPV3HM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomPv3HM", **params)

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomPv3HM called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        reject_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomPv3HM", **params)

    assert actual[0].delta_f == reference[0].delta_f
    assert float(actual[0].epoch) == pytest.approx(float(reference[0].epoch))
    _assert_close(reference, actual)


@pytest.mark.parametrize(
    "params, expected",
    (
        ({}, True),
        ({"mode_array": [(2, 2)]}, True),
        ({"mode_array": [(4, 4), (2, 2), (3, 2)]}, True),
        ({"mode_array": []}, False),
        ({"mode_array": [(2, 1)]}, False),
        ({"mode_array": [(2, -2)]}, False),
        ({"mode_array": [(2, 2), (5, 5)]}, False),
        (
            {
                "spin1x": 0.0,
                "spin1y": 0.0,
                "spin1z": -1.0,
                "spin2x": 0.0,
                "spin2y": 0.0,
                "spin2z": 1.0,
            },
            True,
        ),
        ({"spin1x": 0.1}, True),
        ({"spin1z": 1.01}, False),
        ({"spin1x": 0.8, "spin1y": 0.8}, False),
        ({"spin1z": np.nan}, False),
        ({"spin1z": "invalid"}, False),
        ({"mass2": 0.0}, False),
        ({"distance": 0.0}, False),
        ({"f_ref": -1.0}, False),
        ({"lambda1": 10.0}, False),
        ({"nl_tides_f1": 10.0}, False),
        ({"dchi4": 0.1}, False),
        ({"phenom_xp_convention": 0}, False),
        ({"approximant": "IMRPhenomHM"}, False),
    ),
)
def test_imrphenompv3hm_native_support_boundary(params, expected):
    full_params = {**SUPPORT_BASE, **params}
    assert imrphenompv3hm_native_supported(full_params) is expected
    assert imrphenompv3hm_sequence_native_supported(full_params) is expected


def test_imrphenompv3hm_support_defaults_to_own_approximant():
    params = dict(SUPPORT_BASE)
    params.pop("approximant")
    assert imrphenompv3hm_native_supported(params)
    assert imrphenompv3hm_sequence_native_supported(params)


@pytest.mark.parametrize(
    "override",
    (
        {"delta_f": 0.0},
        {"f_lower": 0.0},
        {"f_final": 18.0},
        {"long_asc_nodes": np.nan},
    ),
)
def test_imrphenompv3hm_regular_sampling_boundary(override):
    assert not imrphenompv3hm_native_supported({**SUPPORT_BASE, **override})


@pytest.mark.parametrize(
    "sample_points",
    ([], [20.0, 20.0], [21.0, 20.0], [0.0, 20.0], [20.0, np.nan]),
)
def test_imrphenompv3hm_sequence_sampling_boundary(sample_points):
    params = {**SUPPORT_BASE, "sample_points": sample_points}
    assert not imrphenompv3hm_sequence_native_supported(params)


@pytest.mark.parametrize(
    "params, sample_points",
    (
        (REGULAR_CASES[0], [20.0, 31.5, 80.0, 180.0, 500.0]),
        (REGULAR_CASES[1], [21.0, 37.25, 95.0, 240.0, 610.0]),
    ),
)
def test_imrphenompv3hm_sequence_matches_direct_lal_without_generic_dispatch(
    params,
    sample_points,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.waveform as waveform

    _activate_scheme(_scheme.CPUScheme())
    reference = _lal_pv3hm_sequence(params, sample_points)

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomPv3HM used generic LAL dispatch")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        reject_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomPv3HM",
        sample_points=sample_points,
        **params,
    )

    _assert_close(reference, actual)


@pytest.mark.parametrize("sample_points", ([23.0], [23.0, 47.5]))
def test_imrphenompv3hm_short_sequence_fref_zero_matches_lal_prefix(
    sample_points,
    monkeypatch,
    preserve_scheme,
):
    """One- and two-point native sequences extend LAL's vector API."""

    import pycbc.waveform.waveform as waveform

    params = {**REGULAR_CASES[1], "f_ref": 0.0}
    reference_points = [23.0, 47.5, 91.0]
    reference = _lal_pv3hm_sequence(params, reference_points)

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("short native IMRPhenomPv3HM used generic LAL dispatch")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        reject_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomPv3HM",
        sample_points=sample_points,
        **params,
    )

    _assert_close(reference, actual, expected_prefix=len(sample_points))


@pytest.mark.parametrize("device_name", ("cpu", "mps", "cuda"))
def test_imrphenompv3hm_native_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.waveform as waveform

    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    monkeypatch.setenv("PYCBC_IMRPHENOMPV3HM_NATIVE", "1")

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("supported IMRPhenomPv3HM called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        reject_lal,
    )
    _activate_scheme(_scheme.TorchScheme(device_name))
    params = REGULAR_CASES[2] if device_name == "mps" else REGULAR_CASES[0]
    actual = get_fd_waveform(
        approximant="IMRPhenomPv3HM",
        **params,
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    for polarization in actual:
        assert polarization._data.tensor.device.type == device_name
        assert polarization._data.tensor.dtype == expected_dtype


def test_imrphenompv3hm_precessing_mps_uses_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    if not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")

    import pycbc.waveform.imrphenompv3_torch as pv3_torch
    import pycbc.waveform.waveform as waveform

    original = waveform.lalsimulation.SimInspiralChooseFDWaveform
    calls = 0

    def record_lal(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    def reject_native(**_params):
        raise AssertionError("precessing MPS request reached native Pv3HM")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        record_lal,
    )
    monkeypatch.setattr(pv3_torch, "imrphenompv3hm_fd_torch", reject_native)
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("mps"))
    actual = get_fd_waveform(
        approximant="IMRPhenomPv3HM",
        **REGULAR_CASES[0],
    )

    assert calls == 1
    assert all(item._data.tensor.device.type == "mps" for item in actual)


@pytest.mark.parametrize(
    "override",
    (
        {"lambda1": 10.0},
        {"mode_array": [(2, 1)]},
    ),
)
def test_imrphenompv3hm_unsupported_requests_use_lal_fallback(
    override,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenompv3_torch as pv3_torch
    import pycbc.waveform.waveform as waveform

    original = waveform.lalsimulation.SimInspiralChooseFDWaveform
    calls = 0

    def record_lal(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    def reject_native(**_params):
        raise AssertionError("unsupported IMRPhenomPv3HM reached Torch")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        record_lal,
    )
    monkeypatch.setattr(
        pv3_torch,
        "imrphenompv3hm_fd_torch",
        reject_native,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = {**REGULAR_CASES[0], **override}

    with pytest.raises(RuntimeError):
        get_fd_waveform(approximant="IMRPhenomPv3HM", **params)
    assert calls == 1


def test_imrphenompv3hm_component_opt_out_uses_lal(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.waveform as waveform

    original = waveform.lalsimulation.SimInspiralChooseFDWaveform
    calls = 0

    def record_lal(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        record_lal,
    )
    _clear_native_flags(monkeypatch)
    monkeypatch.setenv("PYCBC_IMRPHENOMPV3HM_NATIVE", "0")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    get_fd_waveform(approximant="IMRPhenomPv3HM", **REGULAR_CASES[0])

    assert calls == 1
