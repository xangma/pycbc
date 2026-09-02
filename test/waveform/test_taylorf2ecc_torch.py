import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme
from pycbc.waveform import get_fd_waveform, get_fd_waveform_sequence
from pycbc.waveform.taylorf2ecc_torch import (
    _eccentric_phase_polynomial,
    _eccentric_phase_scalar,
    taylorf2ecc_fd_sequence_torch,
    taylorf2ecc_native_supported,
    taylorf2ecc_sequence_native_supported,
)


_SEQUENCE_PARAMS = dict(
    mass1=20.0,
    mass2=15.0,
    spin1z=0.2,
    spin2z=-0.1,
    distance=400.0,
    inclination=0.8,
    coa_phase=0.3,
    eccentricity=0.05,
)

_NATIVE_FLAG_ENVS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_TAYLORF2ECC_NATIVE",
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
    monkeypatch.setenv("PYCBC_TAYLORF2ECC_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme)
    reference = get_fd_waveform(approximant="TaylorF2Ecc", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _clear_native_flags(monkeypatch)
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


@pytest.mark.parametrize("eccentricity_order", [-1, 0, 1, 2, 3, 4, 5, 6])
def test_taylorf2ecc_scalar_phase_matches_tensor(eccentricity_order):
    total_mass = 2.7
    eta = 1.4 * 1.3 / total_mass**2
    frequency = 20.0
    f_ecc = 100.0
    eccentricity = 0.1
    pi_mass = np.pi * total_mass * lal.MTSUN_SI
    velocity = torch.tensor(
        (pi_mass * frequency) ** (1.0 / 3.0),
        dtype=torch.float64,
    )
    velocity0 = torch.tensor(
        (pi_mass * f_ecc) ** (1.0 / 3.0),
        dtype=torch.float64,
    )
    tensor_phase = (
        _eccentric_phase_polynomial(
            velocity,
            velocity0,
            eccentricity,
            eta,
            eccentricity_order,
        )
        / velocity**5
    )

    scalar_phase = _eccentric_phase_scalar(
        frequency,
        f_ecc,
        total_mass,
        eta,
        eccentricity,
        eccentricity_order,
    )
    assert scalar_phase == pytest.approx(tensor_phase.item(), rel=2.0e-15)


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
    import pycbc.waveform.waveform as waveform_mod

    native = taylorf2ecc_mod.taylorf2ecc_fd_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("default TaylorF2Ecc dispatch called LAL")

    monkeypatch.setattr(
        taylorf2ecc_mod, "taylorf2ecc_fd_torch", recording_native
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    _clear_native_flags(monkeypatch)
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
    _clear_native_flags(monkeypatch)
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
    assert taylorf2ecc_sequence_native_supported(params) is expected


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

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme, "mps")
    actual, _ = get_fd_waveform(approximant="TaylorF2Ecc", **params)

    assert actual._data.tensor.device.type == "mps"
    assert actual._data.tensor.dtype == torch.complex64
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual.numpy()[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    assert relative_error < 1.0e-4


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_taylorf2ecc_mps_phase_accuracy_boundary(preserve_scheme):
    safe = dict(
        mass1=1.4,
        mass2=1.3,
        f_lower=20.0,
        f_ref=100.0,
        eccentricity=0.1,
    )
    unsafe = dict(safe, eccentricity=0.4)

    _activate_scheme(_scheme.TorchScheme, "mps")
    assert taylorf2ecc_native_supported(safe)
    assert not taylorf2ecc_native_supported(unsafe)
    assert taylorf2ecc_native_supported(dict(unsafe, eccentricity=0.0))
    assert taylorf2ecc_sequence_native_supported(
        dict(safe, sample_points=[100.0, 20.0, 50.0])
    )
    assert not taylorf2ecc_sequence_native_supported(
        dict(unsafe, sample_points=[100.0, 20.0, 50.0])
    )

    _activate_scheme(_scheme.CPUScheme)
    assert taylorf2ecc_native_supported(unsafe)
    assert taylorf2ecc_sequence_native_supported(
        dict(unsafe, sample_points=[100.0, 20.0, 50.0])
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_taylorf2ecc_mps_phase_boundary_uses_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.taylorf2ecc_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    class LALSequenceFallbackReached(Exception):
        pass

    params = dict(
        mass1=1.4,
        mass2=1.3,
        delta_f=1.0,
        f_lower=20.0,
        f_final=160.0,
        f_ref=100.0,
        distance=100.0,
        eccentricity=0.4,
    )
    regular_lal = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    regular_calls = 0
    sequence_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsafe MPS TaylorF2Ecc request reached Torch")

    def recording_regular(*args, **kwargs):
        nonlocal regular_calls
        regular_calls += 1
        return regular_lal(*args, **kwargs)

    def recording_sequence(*_args, **_kwargs):
        nonlocal sequence_calls
        sequence_calls += 1
        raise LALSequenceFallbackReached

    monkeypatch.setattr(native_mod, "taylorf2ecc_fd_torch", unexpected_native)
    monkeypatch.setattr(
        native_mod,
        "taylorf2ecc_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_regular,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_sequence,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme, "mps")

    regular = get_fd_waveform(approximant="TaylorF2Ecc", **params)
    with pytest.raises(LALSequenceFallbackReached):
        get_fd_waveform_sequence(
            approximant="TaylorF2Ecc",
            sample_points=[100.0, 20.0, 50.0],
            **params,
        )

    assert regular_calls == 1
    assert sequence_calls == 1
    for waveform in regular:
        assert waveform._data.tensor.device.type == "mps"
        assert waveform._data.tensor.dtype == torch.complex64


def _regular_samples(params, sample_points):
    """Return regular-grid values at the requested bin-aligned points."""
    regular = get_fd_waveform(approximant="TaylorF2Ecc", **params)
    indices = [
        int(frequency / params["delta_f"]) for frequency in sample_points
    ]
    return tuple(series.numpy()[indices] for series in regular)


def test_taylorf2ecc_sequence_public_parity_and_dispatch(
    monkeypatch,
    preserve_scheme,
):
    sample_points = [20.0, 30.0, 50.0, 80.0, 120.0]
    regular_params = dict(
        _SEQUENCE_PARAMS,
        delta_f=0.25,
        f_lower=20.0,
        f_final=130.0,
        f_ref=30.0,
        long_asc_nodes=0.0,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORF2ECC_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme)
    expected = _regular_samples(regular_params, sample_points)

    import pycbc.waveform.taylorf2ecc_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    native = native_mod.taylorf2ecc_fd_sequence_torch
    calls = 0

    def recording_native(**params):
        nonlocal calls
        calls += 1
        return native(**params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native TaylorF2Ecc sequence called LAL")

    monkeypatch.setattr(
        native_mod,
        "taylorf2ecc_fd_sequence_torch",
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
        approximant="TaylorF2Ecc",
        sample_points=sample_points,
        long_asc_nodes=0.91,
        **_SEQUENCE_PARAMS,
        f_ref=30.0,
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


def test_taylorf2ecc_sequence_f_ref_zero_uses_lowest_frequency(
    monkeypatch,
    preserve_scheme,
):
    sample_points = [80.0, 20.0, 50.0, 20.0, 30.0]
    regular_params = dict(
        _SEQUENCE_PARAMS,
        delta_f=0.25,
        f_lower=min(sample_points),
        f_final=100.0,
        f_ref=0.0,
        long_asc_nodes=0.0,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORF2ECC_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme)
    expected = _regular_samples(regular_params, sample_points)

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme, "cpu")
    actual = get_fd_waveform_sequence(
        approximant="TaylorF2Ecc",
        sample_points=sample_points,
        f_ref=0.0,
        **_SEQUENCE_PARAMS,
    )
    reordered_points = list(reversed(sample_points))
    reordered = get_fd_waveform_sequence(
        approximant="TaylorF2Ecc",
        sample_points=reordered_points,
        f_ref=0.0,
        **_SEQUENCE_PARAMS,
    )

    order = [sample_points.index(frequency) for frequency in reordered_points]
    for expected_array, result, reordered_result in zip(
        expected,
        actual,
        reordered,
    ):
        np.testing.assert_allclose(
            result.numpy(), expected_array, rtol=2.0e-10, atol=0.0
        )
        torch.testing.assert_close(
            reordered_result._data.tensor,
            result._data.tensor[order],
            rtol=1.0e-13,
            atol=1.0e-30,
        )
        assert result[1] == result[3]


def test_taylorf2ecc_sequence_ignores_long_asc_nodes(
    monkeypatch,
    preserve_scheme,
):
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme, "cpu")
    params = dict(
        approximant="TaylorF2Ecc",
        sample_points=[20.0, 30.0, 50.0, 80.0],
        f_ref=30.0,
        **_SEQUENCE_PARAMS,
    )
    baseline = get_fd_waveform_sequence(long_asc_nodes=0.0, **params)
    changed = get_fd_waveform_sequence(long_asc_nodes=0.91, **params)

    for expected, result in zip(baseline, changed):
        torch.testing.assert_close(
            result._data.tensor,
            expected._data.tensor,
            rtol=0.0,
            atol=0.0,
        )


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
def test_taylorf2ecc_sequence_rejects_invalid_frequencies(
    sample_points,
    message,
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme, "cpu")
    with pytest.raises(ValueError, match=message):
        taylorf2ecc_fd_sequence_torch(
            sample_points=sample_points,
            f_ref=30.0,
            **_SEQUENCE_PARAMS,
        )


def test_taylorf2ecc_sequence_unsupported_options_reach_lal(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.taylorf2ecc_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    class LALFallbackReached(Exception):
        pass

    calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported TaylorF2Ecc reached Torch")

    def recording_lal(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise LALFallbackReached

    monkeypatch.setattr(
        native_mod,
        "taylorf2ecc_fd_sequence_torch",
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
            approximant="TaylorF2Ecc",
            sample_points=[20.0, 30.0, 50.0],
            amplitude_order=2,
            f_ref=30.0,
            **_SEQUENCE_PARAMS,
        )
    assert calls == 1


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_taylorf2ecc_sequence_stays_on_requested_device(
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
            "native TaylorF2Ecc sequence copied through NumPy"
        )

    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    actual = get_fd_waveform_sequence(
        approximant="TaylorF2Ecc",
        sample_points=[80.0, 20.0, 50.0, 20.0],
        f_ref=0.0,
        **_SEQUENCE_PARAMS,
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    for result in actual:
        assert result._data.tensor.device.type == device_name
        assert result._data.tensor.dtype == expected_dtype


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
        ("PYCBC_TAYLORF2ECC_NATIVE", True),
    ),
)
def test_taylorf2ecc_default_native_opt_out_reaches_lal(
    interface,
    lal_name,
    disabled_flag,
    global_enabled,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.taylorf2ecc_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    class LALFallbackReached(Exception):
        pass

    calls = 0

    def unexpected_native(**_params):
        raise AssertionError("disabled TaylorF2Ecc reached Torch")

    def recording_lal(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise LALFallbackReached

    native_name = (
        "taylorf2ecc_fd_torch"
        if interface == "regular"
        else "taylorf2ecc_fd_sequence_torch"
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
                approximant="TaylorF2Ecc",
                delta_f=1.0,
                f_lower=20.0,
                f_final=160.0,
                f_ref=30.0,
                **_SEQUENCE_PARAMS,
            )
        else:
            get_fd_waveform_sequence(
                approximant="TaylorF2Ecc",
                sample_points=[20.0, 60.0, 90.0],
                f_ref=30.0,
                **_SEQUENCE_PARAMS,
            )

    assert calls == 1
