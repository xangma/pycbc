import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.taylorf2redspin_torch import (  # noqa: E402
    taylorf2redspin_fd_sequence_torch,
    taylorf2redspin_fd_torch,
    taylorf2redspin_native_supported,
    taylorf2redspin_sequence_native_supported,
)


_ENV_KEYS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_TAYLORF2REDSPIN_NATIVE",
    "PYCBC_TAYLORF2REDSPINTIDAL_NATIVE",
)


def _native_flag(approximant):
    return f"PYCBC_{approximant.upper()}_NATIVE"


def _clear_native_flags(monkeypatch):
    """Remove every native flag so the registry default applies."""
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def preserve_scheme():
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


def _assert_waveform_close(reference, actual, tolerance=2.0e-11):
    for expected, result in zip(reference, actual):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        expected_array = expected.numpy()
        result_array = result.numpy()
        np.testing.assert_array_equal(
            result_array == 0.0,
            expected_array == 0.0,
        )
        nonzero = np.abs(expected_array) > 0.0
        if nonzero.any():
            relative_error = np.linalg.norm(
                result_array[nonzero] - expected_array[nonzero]
            ) / np.linalg.norm(expected_array[nonzero])
            assert relative_error < tolerance


@pytest.mark.parametrize(
    ("approximant", "extra"),
    [
        ("TaylorF2RedSpin", {}),
        (
            "TaylorF2RedSpin",
            {"phase_order": 4, "amplitude_order": 3, "f_final": 180.3},
        ),
        (
            "TaylorF2RedSpinTidal",
            {"lambda1": 333.0, "lambda2": 777.0},
        ),
        (
            "TaylorF2RedSpinTidal",
            {
                "lambda1": 1200.0,
                "lambda2": 40.0,
                "phase_order": 0,
                "amplitude_order": 6,
                "f_final": 180.0,
            },
        ),
    ],
)
def test_taylorf2redspin_matches_lalsimulation(
    approximant,
    extra,
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=12.3,
        mass2=7.8,
        spin1z=0.45,
        spin2z=-0.27,
        delta_f=0.25,
        f_lower=23.1,
        f_ref=41.7,
        distance=321.0,
        inclination=0.8,
        coa_phase=0.37,
        long_asc_nodes=0.23,
    )
    params.update(extra)
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv(_native_flag(approximant), "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant=approximant, **params)

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant=approximant, **params)

    for series in actual:
        assert series._data.tensor.device.type == "cpu"
        assert series._data.tensor.dtype == torch.complex128
    _assert_waveform_close(reference, actual)


@pytest.mark.parametrize("order", [-1, 0, 1, 2, 3, 4, 5, 6, 7])
def test_taylorf2redspin_all_pn_orders_match_lal(
    order,
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=22.0,
        mass2=9.0,
        spin1z=-0.6,
        spin2z=0.3,
        delta_f=1.0,
        f_lower=31.3,
        f_final=96.2,
        distance=270.0,
        phase_order=order,
        amplitude_order=order,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORF2REDSPIN_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(
        approximant="TaylorF2RedSpin",
        **params,
    )

    monkeypatch.setenv("PYCBC_TAYLORF2REDSPIN_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(
        approximant="TaylorF2RedSpin",
        **params,
    )
    _assert_waveform_close(reference, actual)


@pytest.mark.parametrize("f_final", [20.0, 23.0, 23.1])
def test_taylorf2redspin_preserves_lal_all_zero_ranges(
    f_final,
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        approximant="TaylorF2RedSpin",
        mass1=12.3,
        mass2=7.8,
        delta_f=0.25,
        f_lower=23.1,
        f_final=f_final,
        distance=321.0,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORF2REDSPIN_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(**params)

    monkeypatch.setenv("PYCBC_TAYLORF2REDSPIN_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(**params)
    _assert_waveform_close(reference, actual)
    assert not np.any(actual[0].numpy())
    assert len(actual[0]) == int(f_final / params["delta_f"] + 1.0)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"approximant": "TaylorF2RedSpin"}, True),
        ({"approximant": "TaylorF2RedSpinTidal", "lambda1": -20.0}, True),
        ({"approximant": "TaylorF2"}, False),
        ({"approximant": "TaylorF2RedSpin", "lambda1": 10.0}, False),
        ({"approximant": "TaylorF2RedSpin", "spin1x": 0.1}, False),
        ({"approximant": "TaylorF2RedSpin", "phase_order": 8}, False),
        ({"approximant": "TaylorF2RedSpin", "amplitude_order": 1.5}, False),
        ({"approximant": "TaylorF2RedSpin", "spin_order": 0}, False),
        ({"approximant": "TaylorF2RedSpin", "dchi3": 0.1}, False),
        ({"approximant": "TaylorF2RedSpin", "eccentricity": 0.1}, False),
        ({"approximant": "TaylorF2RedSpin", "mode_array": [(2, 2)]}, False),
        (
            {
                "approximant": "TaylorF2RedSpinTidal",
                "lambda1": float("nan"),
            },
            False,
        ),
    ],
)
def test_taylorf2redspin_native_support_boundary(params, expected):
    assert taylorf2redspin_native_supported(params) is expected
    assert taylorf2redspin_sequence_native_supported(params) is expected


@pytest.mark.parametrize(
    ("approximant", "extra"),
    [
        ("TaylorF2RedSpin", {}),
        (
            "TaylorF2RedSpinTidal",
            {"lambda1": 333.0, "lambda2": 777.0},
        ),
    ],
)
def test_taylorf2redspin_public_dispatch_avoids_lalsimulation(
    approximant,
    extra,
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=12.3,
        mass2=7.8,
        spin1z=0.45,
        spin2z=-0.27,
        delta_f=0.5,
        f_lower=23.1,
        distance=321.0,
        **extra,
    )
    monkeypatch.setenv(_native_flag(approximant), "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant=approximant, **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.taylorf2redspin_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    native = native_mod.taylorf2redspin_fd_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native TaylorF2RedSpin called lalsimulation")

    monkeypatch.setattr(native_mod, "taylorf2redspin_fd_torch", recording_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant=approximant, **params)

    assert calls == 1
    for expected, result in zip(reference_arrays, actual):
        np.testing.assert_allclose(
            result.numpy(),
            expected,
            rtol=2.0e-11,
            atol=0.0,
        )


@pytest.mark.parametrize(
    ("approximant", "unsupported"),
    (
        ("TaylorF2RedSpin", {"lambda1": 10.0}),
        (
            "TaylorF2RedSpinTidal",
            {"lambda1": 333.0, "lambda2": 777.0, "spin1x": 0.1},
        ),
    ),
)
def test_taylorf2redspin_unsupported_options_reach_lal_by_default(
    approximant,
    unsupported,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.taylorf2redspin_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported reduced-spin parameters reached Torch")

    lal_calls = 0

    def expected_lal(*_args, **_kwargs):
        nonlocal lal_calls
        lal_calls += 1
        raise RuntimeError("TaylorF2RedSpin LAL fallback reached")

    monkeypatch.setattr(native_mod, "taylorf2redspin_fd_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        expected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(RuntimeError, match="LAL fallback reached"):
        get_fd_waveform(
            approximant=approximant,
            mass1=12.3,
            mass2=7.8,
            delta_f=0.5,
            f_lower=23.1,
            distance=321.0,
            **unsupported,
        )
    assert lal_calls == 1


_SEQUENCE_PARAMS = dict(
    mass1=12.3,
    mass2=7.8,
    spin1z=0.45,
    spin2z=-0.27,
    f_ref=41.7,
    distance=321.0,
    inclination=0.8,
    coa_phase=0.37,
)


@pytest.mark.parametrize(
    ("approximant", "extra"),
    [
        ("TaylorF2RedSpin", {}),
        (
            "TaylorF2RedSpinTidal",
            {"lambda1": 333.0, "lambda2": 777.0},
        ),
    ],
)
def test_taylorf2redspin_sequence_matches_regular_lal_grid_and_dispatches(
    approximant,
    extra,
    monkeypatch,
    preserve_scheme,
):
    sample_points = [23.25, 37.5, 80.0, 150.0]
    delta_f = 0.25
    monkeypatch.setenv(_native_flag(approximant), "0")
    _activate_scheme(_scheme.CPUScheme())
    regular = get_fd_waveform(
        approximant=approximant,
        delta_f=delta_f,
        f_lower=23.1,
        long_asc_nodes=0.0,
        **extra,
        **_SEQUENCE_PARAMS,
    )
    indices = [int(frequency / delta_f) for frequency in sample_points]
    expected = tuple(series.numpy()[indices] for series in regular)

    import pycbc.waveform.taylorf2redspin_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    native = native_mod.taylorf2redspin_fd_sequence_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native reduced-spin sequence called LAL")

    monkeypatch.setattr(
        native_mod,
        "taylorf2redspin_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        long_asc_nodes=0.91,
        **extra,
        **_SEQUENCE_PARAMS,
    )

    assert calls == 1
    for expected_array, result in zip(expected, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        np.testing.assert_allclose(
            result.numpy(),
            expected_array,
            rtol=2.0e-11,
            atol=0.0,
        )


def test_taylorf2redspin_sequence_conventions(monkeypatch, preserve_scheme):
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    sample_points = [150.0, 23.25, 80.0, 10000.0]
    unordered = get_fd_waveform_sequence(
        approximant="TaylorF2RedSpinTidal",
        sample_points=sample_points,
        long_asc_nodes=0.91,
        lambda1=333.0,
        lambda2=777.0,
        **_SEQUENCE_PARAMS,
    )
    sorted_points = sorted(sample_points)
    ordered = get_fd_waveform_sequence(
        approximant="TaylorF2RedSpinTidal",
        sample_points=sorted_points,
        long_asc_nodes=0.0,
        lambda1=333.0,
        lambda2=777.0,
        **_SEQUENCE_PARAMS,
    )
    order = [sorted_points.index(frequency) for frequency in sample_points]
    for unordered_array, ordered_array in zip(unordered, ordered):
        torch.testing.assert_close(
            unordered_array._data.tensor,
            ordered_array._data.tensor[order],
            rtol=1.0e-13,
            atol=0.0,
        )
        assert unordered_array[-1] == 0.0


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_taylorf2redspin_native_stays_on_requested_device_without_host_transfer(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    from pycbc.types.array_torch import TorchArrayData

    params = dict(
        approximant="TaylorF2RedSpinTidal",
        lambda1=333.0,
        lambda2=777.0,
        **_SEQUENCE_PARAMS,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    reference_regular = get_fd_waveform(
        delta_f=0.25,
        f_lower=23.1,
        **params,
    )
    reference_sequence = get_fd_waveform_sequence(
        sample_points=[23.25, 37.5, 120.0, 10000.0],
        **params,
    )
    references = tuple(
        array._data.tensor.clone()
        for pair in (reference_regular, reference_sequence)
        for array in pair
    )

    _activate_scheme(_scheme.TorchScheme(device_name))

    def reject_host_transfer(_self):
        raise AssertionError("native reduced-spin waveform copied through NumPy")

    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    actual_regular = get_fd_waveform(
        delta_f=0.25,
        f_lower=23.1,
        **params,
    )
    actual_sequence = get_fd_waveform_sequence(
        sample_points=[23.25, 37.5, 120.0, 10000.0],
        **params,
    )
    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    tolerance = 1.0e-3 if device_name == "mps" else 2.0e-11
    actuals = tuple(
        array
        for pair in (actual_regular, actual_sequence)
        for array in pair
    )
    for reference, actual in zip(references, actuals):
        tensor = actual._data.tensor
        assert tensor.device.type == device_name
        assert tensor.dtype == expected_dtype
        torch.testing.assert_close(
            tensor,
            reference.to(device=tensor.device, dtype=expected_dtype),
            rtol=tolerance,
            atol=0.0,
        )


@pytest.mark.parametrize(
    ("sample_points", "message"),
    [
        ([], "non-empty vector"),
        ([[20.0, 30.0]], "non-empty vector"),
        ([20.0, float("nan")], "finite"),
        ([20.0, 0.0], "positive"),
        ([20.0, -30.0], "positive"),
    ],
)
def test_taylorf2redspin_sequence_rejects_invalid_frequencies(
    sample_points,
    message,
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(ValueError, match=message):
        taylorf2redspin_fd_sequence_torch(
            approximant="TaylorF2RedSpin",
            sample_points=sample_points,
            **_SEQUENCE_PARAMS,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mass1": 0.0}, "masses must be positive"),
        ({"distance": 0.0}, "distance must be positive"),
        ({"f_ref": -1.0}, "f_ref must be non-negative"),
        ({"delta_f": 0.0}, "delta_f and f_lower must be positive"),
        ({"f_final": -1.0}, "f_final must be non-negative"),
        ({"spin1z": 1.1}, "component spins"),
    ],
)
def test_taylorf2redspin_rejects_invalid_inputs(
    changes,
    message,
    preserve_scheme,
):
    params = dict(
        approximant="TaylorF2RedSpin",
        mass1=12.3,
        mass2=7.8,
        spin1z=0.45,
        spin2z=-0.27,
        delta_f=0.25,
        f_lower=23.1,
        distance=321.0,
    )
    params.update(changes)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(ValueError, match=message):
        taylorf2redspin_fd_torch(**params)


def test_tidal_model_validates_effective_not_individual_spin(preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    waveform = taylorf2redspin_fd_torch(
        approximant="TaylorF2RedSpinTidal",
        mass1=10.0,
        mass2=10.0,
        spin1z=1.2,
        spin2z=-1.2,
        delta_f=1.0,
        f_lower=30.0,
        f_final=60.0,
        distance=100.0,
    )
    assert any(waveform[0]._data.tensor != 0.0)


@pytest.mark.parametrize(
    ("approximant", "unsupported"),
    (
        ("TaylorF2RedSpin", {"spin1x": 0.1}),
        (
            "TaylorF2RedSpinTidal",
            {"lambda1": 333.0, "lambda2": 777.0, "spin1x": 0.1},
        ),
    ),
)
def test_sequence_unsupported_options_reach_lal_by_default(
    approximant,
    unsupported,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.taylorf2redspin_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported reduced-spin sequence reached Torch")

    lal_calls = 0

    def expected_lal(*_args, **_kwargs):
        nonlocal lal_calls
        lal_calls += 1
        raise RuntimeError("TaylorF2RedSpin sequence LAL fallback reached")

    monkeypatch.setattr(
        native_mod,
        "taylorf2redspin_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        expected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(RuntimeError, match="sequence LAL fallback reached"):
        get_fd_waveform_sequence(
            approximant=approximant,
            sample_points=[20.0, 30.0],
            **unsupported,
            **_SEQUENCE_PARAMS,
        )
    assert lal_calls == 1


def test_default_cutoff_is_schwarzschild_isco(monkeypatch, preserve_scheme):
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    total_mass = _SEQUENCE_PARAMS["mass1"] + _SEQUENCE_PARAMS["mass2"]
    f_isco = 1.0 / (6.0**1.5 * np.pi * total_mass * lal.MTSUN_SI)
    below, exact, above = get_fd_waveform_sequence(
        approximant="TaylorF2RedSpin",
        sample_points=[np.nextafter(f_isco, 0.0), f_isco, np.nextafter(f_isco, np.inf)],
        **_SEQUENCE_PARAMS,
    )[0]
    assert below != 0.0
    assert exact != 0.0
    assert above == 0.0


@pytest.mark.parametrize(
    ("approximant", "extra"),
    (
        ("TaylorF2RedSpin", {}),
        (
            "TaylorF2RedSpinTidal",
            {"lambda1": 333.0, "lambda2": 777.0},
        ),
    ),
)
@pytest.mark.parametrize(
    ("interface", "lal_name"),
    (
        ("regular", "SimInspiralChooseFDWaveform"),
        ("sequence", "SimInspiralChooseFDWaveformSequence"),
    ),
)
@pytest.mark.parametrize(
    ("disabled_kind", "global_enabled"),
    (("global", False), ("component", True)),
)
def test_taylorf2redspin_default_native_opt_out_reaches_lal(
    approximant,
    extra,
    interface,
    lal_name,
    disabled_kind,
    global_enabled,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.taylorf2redspin_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    class LALFallbackReached(Exception):
        pass

    calls = 0

    def unexpected_native(**_params):
        raise AssertionError("disabled reduced-spin model reached Torch")

    def recording_lal(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise LALFallbackReached

    native_name = (
        "taylorf2redspin_fd_torch"
        if interface == "regular"
        else "taylorf2redspin_fd_sequence_torch"
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
    disabled_flag = (
        "PYCBC_TORCH_NATIVE_PORTS"
        if disabled_kind == "global"
        else _native_flag(approximant)
    )
    monkeypatch.setenv(disabled_flag, "0")
    _activate_scheme(_scheme.TorchScheme("cpu"))

    with pytest.raises(LALFallbackReached):
        if interface == "regular":
            get_fd_waveform(
                approximant=approximant,
                delta_f=0.5,
                f_lower=23.1,
                f_final=180.0,
                **extra,
                **_SEQUENCE_PARAMS,
            )
        else:
            get_fd_waveform_sequence(
                approximant=approximant,
                sample_points=[23.25, 80.0, 150.0],
                **extra,
                **_SEQUENCE_PARAMS,
            )

    assert calls == 1


def test_environment_key_list_is_complete():
    assert set(_ENV_KEYS) == {
        "PYCBC_TORCH_NATIVE_PORTS",
        "PYCBC_TORCH_NATIVE",
        "PYCBC_TAYLORF2REDSPIN_NATIVE",
        "PYCBC_TAYLORF2REDSPINTIDAL_NATIVE",
    }
    assert all(key.startswith("PYCBC_") for key in _ENV_KEYS)
