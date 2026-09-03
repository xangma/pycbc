import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme
from pycbc.waveform import get_fd_waveform, get_fd_waveform_sequence
from pycbc.waveform.taylorf2nltides_torch import (
    taylorf2nltides_fd_sequence_torch,
    taylorf2nltides_native_supported,
    taylorf2nltides_sequence_native_supported,
)


_NL_TIDES = dict(
    nl_tides_a1=1.0e-8,
    nl_tides_n1=2.5,
    nl_tides_f1=60.0,
    nl_tides_a2=2.0e-8,
    nl_tides_n2=3.0,
    nl_tides_f2=90.0,
)

_SEQUENCE_PARAMS = dict(
    mass1=1.6,
    mass2=1.3,
    spin1z=0.03,
    spin2z=-0.02,
    distance=120.0,
    inclination=0.7,
    coa_phase=0.4,
    f_ref=30.0,
    lambda1=800.0,
    lambda2=650.0,
    **_NL_TIDES,
)

_NATIVE_FLAG_ENVS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_TAYLORF2NLTIDES_NATIVE",
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


def _activate_scheme(scheme_type, *args):
    _scheme.Scheme._single = None
    _scheme.mgr.state = scheme_type(*args)


def _clear_native_flags(monkeypatch):
    """Remove every native flag so the registry default applies."""
    for name in _NATIVE_FLAG_ENVS:
        monkeypatch.delenv(name, raising=False)


def _reference_and_native(params, monkeypatch):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORF2NLTIDES_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme)
    reference = get_fd_waveform(
        approximant="TaylorF2NLTides",
        **params,
    )
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_TAYLORF2NLTIDES_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme)
    native = get_fd_waveform(
        approximant="TaylorF2NLTides",
        **params,
    )
    return reference, reference_arrays, native


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=1.6,
            mass2=1.3,
            spin1z=0.03,
            spin2z=-0.02,
            distance=120.0,
            inclination=0.7,
            coa_phase=0.4,
            long_asc_nodes=0.2,
            f_lower=20.0,
            f_final=512.0,
            delta_f=0.25,
            f_ref=30.0,
            lambda1=800.0,
            lambda2=650.0,
            **_NL_TIDES,
        ),
        dict(
            mass1=1.8,
            mass2=1.2,
            spin1z=0.04,
            spin2z=-0.01,
            distance=90.0,
            inclination=1.1,
            coa_phase=0.2,
            long_asc_nodes=0.31,
            f_lower=18.0,
            f_final=450.0,
            delta_f=0.5,
            lambda1=600.0,
            lambda2=300.0,
            dquad_mon1=0.0,
            dquad_mon2=0.0,
            tidal_order=12,
            nl_tides_a1=1.2e-8,
            nl_tides_n1=3.0,
            nl_tides_f1=110.0,
            nl_tides_a2=2.2e-8,
            nl_tides_n2=3.5,
            nl_tides_f2=50.0,
        ),
        dict(
            mass1=3.0,
            mass2=2.0,
            spin1z=0.1,
            spin2z=0.05,
            distance=200.0,
            inclination=0.4,
            f_lower=15.0,
            delta_f=1.0,
            f_ref=27.0,
            phase_order=5,
            spin_order=3,
            tidal_order=0,
            dchi3=0.02,
            dchi6l=-0.01,
            nl_tides_a1=0.8e-8,
            nl_tides_n1=2.0,
            nl_tides_f1=40.0,
            nl_tides_a2=0.0,
            nl_tides_n2=3.0,
            nl_tides_f2=70.0,
        ),
    ],
)
def test_taylorf2nltides_public_torch_parity(
    params,
    monkeypatch,
    preserve_scheme,
):
    reference, reference_arrays, native = _reference_and_native(
        params,
        monkeypatch,
    )

    for expected, expected_array, actual in zip(
        reference,
        reference_arrays,
        native,
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


def test_taylorf2nltides_parameters_reach_lal():
    lalsimulation = pytest.importorskip("lalsimulation")

    import pycbc.waveform.waveform as waveform_mod

    params = waveform_mod.props(None, **_NL_TIDES)
    lal_params = waveform_mod._check_lal_pars(params)

    assert lalsimulation.SimInspiralWaveformParamsLookupNLTidesA1(
        lal_params
    ) == pytest.approx(_NL_TIDES["nl_tides_a1"])
    assert lalsimulation.SimInspiralWaveformParamsLookupNLTidesN1(
        lal_params
    ) == pytest.approx(_NL_TIDES["nl_tides_n1"])
    assert lalsimulation.SimInspiralWaveformParamsLookupNLTidesF1(
        lal_params
    ) == pytest.approx(_NL_TIDES["nl_tides_f1"])
    assert lalsimulation.SimInspiralWaveformParamsLookupNLTidesA2(
        lal_params
    ) == pytest.approx(_NL_TIDES["nl_tides_a2"])
    assert lalsimulation.SimInspiralWaveformParamsLookupNLTidesN2(
        lal_params
    ) == pytest.approx(_NL_TIDES["nl_tides_n2"])
    assert lalsimulation.SimInspiralWaveformParamsLookupNLTidesF2(
        lal_params
    ) == pytest.approx(_NL_TIDES["nl_tides_f2"])


def test_taylorf2nltides_public_dispatches_native(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.taylorf2nltides_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    native = native_mod.taylorf2nltides_fd_torch
    calls = 0

    def recording_native(**params):
        nonlocal calls
        calls += 1
        return native(**params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("default TaylorF2NLTides dispatch called LAL")

    monkeypatch.setattr(
        native_mod,
        "taylorf2nltides_fd_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme)
    waveform = get_fd_waveform(
        approximant="TaylorF2NLTides",
        mass1=1.5,
        mass2=1.3,
        delta_f=1.0,
        f_lower=20.0,
        f_final=200.0,
        **_NL_TIDES,
    )

    assert calls == 1
    assert waveform[0]._data.tensor.device.type == "cpu"


def test_taylorf2nltides_unsupported_options_use_lal(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.taylorf2nltides_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError(
            "unsupported TaylorF2NLTides parameters reached Torch"
        )

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        native_mod,
        "taylorf2nltides_fd_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme)
    waveform = get_fd_waveform(
        approximant="TaylorF2NLTides",
        mass1=1.5,
        mass2=1.3,
        delta_f=1.0,
        f_lower=20.0,
        f_final=200.0,
        amplitude_order=2,
        **_NL_TIDES,
    )

    assert lal_calls == 1
    assert np.isfinite(waveform[0].numpy()).all()


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, True),
        ({"nl_tides_a1": None}, False),
        ({"nl_tides_a1": np.nan}, False),
        ({"nl_tides_n2": np.inf}, False),
        ({"nl_tides_f1": 0.0}, False),
        ({"nl_tides_f2": -1.0}, False),
        ({"tidal_order": 10}, True),
        ({"tidal_order": 13}, False),
        ({"amplitude_order": 2}, False),
        ({"spin1x": 0.1}, False),
    ],
)
def test_taylorf2nltides_native_support_boundary(changes, expected):
    params = dict(_NL_TIDES)
    params.update(changes)
    assert taylorf2nltides_native_supported(params) is expected
    assert taylorf2nltides_sequence_native_supported(params) is expected


def _regular_samples(params, sample_points):
    """Return regular-grid values at the requested bin-aligned points."""
    regular = get_fd_waveform(approximant="TaylorF2NLTides", **params)
    indices = [
        int(frequency / params["delta_f"]) for frequency in sample_points
    ]
    return tuple(series.numpy()[indices] for series in regular)


def test_taylorf2nltides_sequence_public_parity_and_dispatch(
    monkeypatch,
    preserve_scheme,
):
    sample_points = [140.0, 20.0, 60.0, 90.0, 20.0, 110.0]
    regular_params = dict(
        _SEQUENCE_PARAMS,
        delta_f=0.25,
        f_lower=20.0,
        f_final=160.0,
        long_asc_nodes=0.0,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORF2NLTIDES_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme)
    expected = _regular_samples(regular_params, sample_points)

    import pycbc.waveform.taylorf2nltides_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    native = native_mod.taylorf2nltides_fd_sequence_torch
    calls = 0

    def recording_native(**params):
        nonlocal calls
        calls += 1
        return native(**params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native TaylorF2NLTides sequence called LAL")

    monkeypatch.setattr(
        native_mod,
        "taylorf2nltides_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme, "cpu")
    actual = get_fd_waveform_sequence(
        approximant="TaylorF2NLTides",
        sample_points=sample_points,
        long_asc_nodes=0.91,
        **_SEQUENCE_PARAMS,
    )

    assert calls == 1
    for expected_array, result in zip(expected, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        np.testing.assert_allclose(
            result.numpy(),
            expected_array,
            rtol=2.0e-10,
            atol=0.0,
        )
        assert result[1] == result[4]


@pytest.mark.parametrize(
    ("sample_points", "message"),
    [
        ([], "non-empty vector"),
        ([[20.0, 30.0]], "non-empty vector"),
        ([20.0, float("nan")], "finite"),
        ([20.0, float("inf")], "finite"),
        ([20.0, 0.0], "positive"),
        ([20.0, -30.0], "positive"),
    ],
)
def test_taylorf2nltides_sequence_rejects_invalid_frequencies(
    sample_points,
    message,
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme, "cpu")
    with pytest.raises(ValueError, match=message):
        taylorf2nltides_fd_sequence_torch(
            sample_points=sample_points,
            **_SEQUENCE_PARAMS,
        )


def test_taylorf2nltides_sequence_unsupported_options_reach_lal(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.taylorf2nltides_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    class LALFallbackReached(Exception):
        pass

    calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported TaylorF2NLTides reached Torch")

    def recording_lal(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise LALFallbackReached

    monkeypatch.setattr(
        native_mod,
        "taylorf2nltides_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme, "cpu")
    with pytest.raises(LALFallbackReached):
        get_fd_waveform_sequence(
            approximant="TaylorF2NLTides",
            sample_points=[20.0, 60.0, 90.0],
            amplitude_order=2,
            **_SEQUENCE_PARAMS,
        )
    assert calls == 1


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_taylorf2nltides_sequence_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    from pycbc.types.array_torch import TorchArrayData

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme, device_name)

    def reject_host_transfer(_self):
        raise AssertionError(
            "native TaylorF2NLTides sequence copied through NumPy"
        )

    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    actual = get_fd_waveform_sequence(
        approximant="TaylorF2NLTides",
        sample_points=[140.0, 20.0, 60.0, 90.0, 20.0],
        **_SEQUENCE_PARAMS,
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    for result in actual:
        assert result._data.tensor.device.type == device_name
        assert result._data.tensor.dtype == expected_dtype


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_taylorf2nltides_public_torch_uses_mps(
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=1.6,
        mass2=1.3,
        spin1z=0.03,
        spin2z=-0.02,
        distance=120.0,
        inclination=0.7,
        coa_phase=0.4,
        f_lower=20.0,
        f_final=300.0,
        delta_f=0.5,
        f_ref=30.0,
        **_NL_TIDES,
    )
    monkeypatch.setenv("PYCBC_TAYLORF2NLTIDES_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme)
    reference, _ = get_fd_waveform(
        approximant="TaylorF2NLTides",
        **params,
    )
    reference_array = reference.numpy().copy()

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme, "mps")
    actual, _ = get_fd_waveform(
        approximant="TaylorF2NLTides",
        **params,
    )

    assert actual._data.tensor.device.type == "mps"
    assert actual._data.tensor.dtype == torch.complex64
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual.numpy()[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    # BNS TaylorF2 phases accumulate many more cycles than the higher-mass
    # MPS cases used by the base model's tests. MPS is limited to float32.
    assert relative_error < 2.0e-3


@pytest.mark.parametrize(
    ("interface", "lal_name"),
    (
        ("regular", "SimInspiralChooseFDWaveform"),
        ("sequence", "SimInspiralChooseFDWaveformSequence"),
    ),
)
@pytest.mark.parametrize(
    ("disabled_flag", "global_enabled"),
    (
        ("PYCBC_TORCH_NATIVE_PORTS", False),
        ("PYCBC_TAYLORF2NLTIDES_NATIVE", True),
    ),
)
def test_taylorf2nltides_default_native_opt_out_reaches_lal(
    interface,
    lal_name,
    disabled_flag,
    global_enabled,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.taylorf2nltides_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    class LALFallbackReached(Exception):
        pass

    calls = 0

    def unexpected_native(**_params):
        raise AssertionError("disabled TaylorF2NLTides reached Torch")

    def recording_lal(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise LALFallbackReached

    native_name = (
        "taylorf2nltides_fd_torch"
        if interface == "regular"
        else "taylorf2nltides_fd_sequence_torch"
    )
    monkeypatch.setattr(native_mod, native_name, unexpected_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        lal_name,
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    if global_enabled:
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "1")
    monkeypatch.setenv(disabled_flag, "0")
    _activate_scheme(_scheme.TorchScheme, "cpu")

    with pytest.raises(LALFallbackReached):
        if interface == "regular":
            get_fd_waveform(
                approximant="TaylorF2NLTides",
                delta_f=1.0,
                f_lower=20.0,
                f_final=160.0,
                **_SEQUENCE_PARAMS,
            )
        else:
            get_fd_waveform_sequence(
                approximant="TaylorF2NLTides",
                sample_points=[20.0, 60.0, 90.0],
                **_SEQUENCE_PARAMS,
            )

    assert calls == 1
