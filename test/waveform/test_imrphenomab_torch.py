import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenomab_torch import (  # noqa: E402
    imrphenomab_default_native_supported,
    imrphenomab_fd_sequence_torch,
    imrphenomab_fd_torch,
    imrphenomab_native_supported,
    imrphenomab_sequence_native_supported,
)


_ENV_KEYS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_IMRPHENOMA_NATIVE",
    "PYCBC_IMRPHENOMB_NATIVE",
)


def _native_flag(approximant):
    return f"PYCBC_{approximant.upper()}_NATIVE"


def _clear_native_flags(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


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


def _run_case(approximant, params):
    env_backup = {key: os.environ.get(key) for key in _ENV_KEYS}
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ[_native_flag(approximant)] = "0"
        _activate_scheme(_scheme.CPUScheme())
        reference = get_fd_waveform(approximant=approximant, **params)
        reference_arrays = tuple(series.numpy().copy() for series in reference)
        reference_metadata = tuple(
            (len(series), series.delta_f, float(series.epoch))
            for series in reference
        )

        os.environ[_native_flag(approximant)] = "1"
        _activate_scheme(_scheme.TorchScheme("cpu"))
        actual = get_fd_waveform(approximant=approximant, **params)
        actual_arrays = tuple(series.numpy().copy() for series in actual)
        actual_metadata = tuple(
            (len(series), series.delta_f, float(series.epoch))
            for series in actual
        )
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single
    return reference_arrays, actual_arrays, reference_metadata, actual_metadata


@pytest.mark.parametrize(
    ("approximant", "params"),
    [
        (
            "IMRPhenomA",
            dict(
                mass1=35.0,
                mass2=20.0,
                delta_f=0.5,
                f_lower=20.3,
                distance=400.0,
                inclination=0.7,
                coa_phase=0.4,
                long_asc_nodes=0.3,
            ),
        ),
        (
            "IMRPhenomA",
            dict(
                mass1=13.0,
                mass2=9.0,
                delta_f=0.25,
                f_lower=17.3,
                f_final=1777.3,
                f_ref=80.0,
                distance=275.0,
                inclination=1.2,
                coa_phase=1.1,
            ),
        ),
        (
            "IMRPhenomB",
            dict(
                mass1=35.0,
                mass2=20.0,
                spin1z=0.4,
                spin2z=-0.2,
                delta_f=0.5,
                f_lower=20.3,
                distance=400.0,
                inclination=0.7,
                coa_phase=0.4,
                long_asc_nodes=0.3,
            ),
        ),
        (
            "IMRPhenomB",
            dict(
                mass1=12.0,
                mass2=31.0,
                spin1z=-0.7,
                spin2z=0.6,
                delta_f=0.25,
                f_lower=17.3,
                f_final=933.3,
                f_ref=110.0,
                distance=600.0,
                inclination=1.4,
                coa_phase=2.1,
                long_asc_nodes=-0.2,
            ),
        ),
    ],
)
def test_imrphenomab_torch_matches_lalsimulation(approximant, params):
    reference, actual, reference_metadata, actual_metadata = _run_case(
        approximant, params
    )
    assert actual_metadata == reference_metadata
    for expected, result in zip(reference, actual):
        np.testing.assert_array_equal(result == 0.0, expected == 0.0)
        nonzero = np.abs(expected) > 0.0
        assert nonzero.any()
        relative_error = np.linalg.norm(
            result[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < 3.0e-11


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"approximant": "IMRPhenomA"}, True),
        ({"approximant": "IMRPhenomB", "spin1z": 0.3}, True),
        ({"approximant": "IMRPhenomC"}, False),
        ({"approximant": "IMRPhenomA", "spin1z": 0.1}, False),
        ({"approximant": "IMRPhenomB", "spin1x": 0.1}, False),
        ({"approximant": "IMRPhenomB", "lambda1": 100.0}, False),
        ({"approximant": "IMRPhenomB", "dchi3": 0.1}, False),
        ({"approximant": "IMRPhenomB", "phase_order": 4}, False),
        ({"approximant": "IMRPhenomA", "mode_array": [(2, 2)]}, False),
    ],
)
def test_imrphenomab_native_support_boundary(params, expected):
    assert imrphenomab_native_supported(params) is expected
    assert imrphenomab_sequence_native_supported(params) is expected


def test_imrphenomab_default_supports_double_precision_devices(
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    assert imrphenomab_default_native_supported({})


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_imrphenomab_default_rejects_mps(preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("mps"))
    assert not imrphenomab_default_native_supported({})


@pytest.mark.parametrize(
    ("approximant", "spin_params"),
    [
        ("IMRPhenomA", {}),
        ("IMRPhenomB", {"spin1z": 0.4, "spin2z": -0.2}),
    ],
)
def test_imrphenomab_public_dispatch_avoids_lalsimulation(
    approximant,
    spin_params,
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=35.0,
        mass2=20.0,
        delta_f=0.5,
        f_lower=20.3,
        distance=400.0,
        inclination=0.7,
        coa_phase=0.4,
        long_asc_nodes=0.3,
        **spin_params,
    )
    monkeypatch.setenv(_native_flag(approximant), "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant=approximant, **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomab_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    native = native_mod.imrphenomab_fd_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomA/B called lalsimulation")

    monkeypatch.setattr(native_mod, "imrphenomab_fd_torch", recording_native)
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
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        np.testing.assert_array_equal(result.numpy() == 0.0, expected == 0.0)
        nonzero = np.abs(expected) > 0.0
        relative_error = np.linalg.norm(
            result.numpy()[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < 3.0e-11


def test_imrphenomab_unsupported_options_reach_lal(
    monkeypatch, preserve_scheme
):
    import pycbc.waveform.imrphenomab_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported IMRPhenomB parameters reached Torch")

    lal_calls = 0

    def expected_lal(*_args, **_kwargs):
        nonlocal lal_calls
        lal_calls += 1
        raise RuntimeError("IMRPhenomB LAL fallback reached")

    monkeypatch.setattr(native_mod, "imrphenomab_fd_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        expected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMB_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(RuntimeError, match="LAL fallback reached"):
        get_fd_waveform(
            approximant="IMRPhenomB",
            mass1=35.0,
            mass2=20.0,
            spin1x=0.1,
            delta_f=0.5,
            f_lower=20.0,
            distance=400.0,
        )
    assert lal_calls == 1


@pytest.mark.parametrize(
    ("global_value", "component_value", "expected_native"),
    [
        (None, None, True),
        ("0", None, False),
        ("1", None, True),
        ("1", "0", False),
        ("0", "1", True),
    ],
)
@pytest.mark.parametrize(
    ("approximant", "spin_params"),
    [
        ("IMRPhenomA", {}),
        ("IMRPhenomB", {"spin1z": 0.4, "spin2z": -0.2}),
    ],
)
def test_imrphenomab_native_flag_precedence(
    approximant,
    spin_params,
    global_value,
    component_value,
    expected_native,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomab_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    native_generator = native_mod.imrphenomab_fd_torch
    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    calls = {"native": 0, "lal": 0}

    def recording_native(**params):
        calls["native"] += 1
        return native_generator(**params)

    def recording_lal(*args, **kwargs):
        calls["lal"] += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(native_mod, "imrphenomab_fd_torch", recording_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.delenv("PYCBC_TORCH_NATIVE", raising=False)
    if global_value is None:
        monkeypatch.delenv("PYCBC_TORCH_NATIVE_PORTS", raising=False)
    else:
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", global_value)
    if component_value is None:
        monkeypatch.delenv(_native_flag(approximant), raising=False)
    else:
        monkeypatch.setenv(_native_flag(approximant), component_value)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    get_fd_waveform(
        approximant=approximant,
        mass1=35.0,
        mass2=20.0,
        delta_f=0.5,
        f_lower=20.0,
        distance=400.0,
        **spin_params,
    )

    assert calls == {
        "native": int(expected_native),
        "lal": int(not expected_native),
    }


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_imrphenomab_default_mps_uses_lalsimulation_fallbacks(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomab_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    class LALSequenceFallbackReached(Exception):
        pass

    params = dict(
        mass1=35.0,
        mass2=20.0,
        spin1z=0.4,
        spin2z=-0.2,
        delta_f=0.5,
        f_lower=20.0,
        distance=400.0,
    )
    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    calls = {"lal": 0, "sequence": 0}

    def unexpected_native(**_params):
        raise AssertionError("default MPS IMRPhenomA/B request reached Torch")

    def recording_lal(*args, **kwargs):
        calls["lal"] += 1
        return lal_generator(*args, **kwargs)

    def recording_sequence(*_args, **_kwargs):
        calls["sequence"] += 1
        raise LALSequenceFallbackReached

    monkeypatch.setattr(native_mod, "imrphenomab_fd_torch", unexpected_native)
    monkeypatch.setattr(
        native_mod,
        "imrphenomab_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_sequence,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("mps"))
    result = get_fd_waveform(approximant="IMRPhenomB", **params)
    with pytest.raises(LALSequenceFallbackReached):
        get_fd_waveform_sequence(
            approximant="IMRPhenomB",
            sample_points=[20.0, 30.0, 50.0],
            mass1=35.0,
            mass2=20.0,
            spin1z=0.4,
            spin2z=-0.2,
            distance=400.0,
        )

    assert calls == {"lal": 1, "sequence": 1}
    for series in result:
        assert series._data.tensor.device.type == "mps"
        assert series._data.tensor.dtype == torch.complex64


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomab_public_native_stays_on_requested_device(
    device_name, monkeypatch, preserve_scheme
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = dict(
        mass1=35.0,
        mass2=20.0,
        spin1z=0.4,
        spin2z=-0.2,
        delta_f=0.5,
        f_lower=20.3,
        distance=400.0,
        inclination=0.7,
        coa_phase=0.4,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMB_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform(approximant="IMRPhenomB", **params)
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMB_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, cross = get_fd_waveform(approximant="IMRPhenomB", **params)
    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    for series in (actual, cross):
        assert series._data.tensor.device.type == device_name
        assert series._data.tensor.dtype == expected_dtype

    actual_array = actual.numpy()
    np.testing.assert_array_equal(actual_array == 0.0, reference_array == 0.0)
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual_array[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    tolerance = 3.0e-3 if device_name == "mps" else 3.0e-11
    assert relative_error < tolerance


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mass1": 0.0}, "masses must be positive"),
        ({"distance": 0.0}, "distance must be positive"),
        ({"f_ref": -1.0}, "f_ref must be non-negative"),
        ({"f_final": 10.0}, "f_final is <= f_lower"),
        ({"delta_f": 0.0}, "delta_f and f_lower must be positive"),
        ({"spin1z": 2.0, "spin2z": 2.0}, "effective spin"),
    ],
)
def test_imrphenomab_native_rejects_invalid_inputs(
    changes, message, preserve_scheme
):
    params = dict(
        approximant="IMRPhenomB",
        mass1=35.0,
        mass2=20.0,
        spin1z=0.4,
        spin2z=-0.2,
        delta_f=0.5,
        f_lower=20.0,
        distance=400.0,
    )
    params.update(changes)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(ValueError, match=message):
        imrphenomab_fd_torch(**params)


SEQUENCE_PARAMS = dict(
    mass1=35.0,
    mass2=20.0,
    f_ref=30.0,
    distance=400.0,
    inclination=0.7,
    coa_phase=0.4,
)


@pytest.mark.parametrize(
    ("approximant", "spin_params"),
    [
        ("IMRPhenomA", {}),
        ("IMRPhenomB", {"spin1z": 0.4, "spin2z": -0.2}),
    ],
)
def test_imrphenomab_sequence_matches_lal_grid_and_dispatches_native(
    approximant,
    spin_params,
    monkeypatch,
    preserve_scheme,
):
    sample_points = [20.0, 31.25, 80.0, 150.0]
    delta_f = 0.25
    monkeypatch.setenv(_native_flag(approximant), "0")
    _activate_scheme(_scheme.CPUScheme())
    regular = get_fd_waveform(
        approximant=approximant,
        delta_f=delta_f,
        f_lower=20.0,
        long_asc_nodes=0.0,
        **spin_params,
        **SEQUENCE_PARAMS,
    )
    indices = [int(frequency / delta_f) for frequency in sample_points]
    expected = tuple(series.numpy()[indices] for series in regular)

    import pycbc.waveform.imrphenomab_torch as native_mod
    import pycbc.waveform.waveform as waveform_mod

    native = native_mod.imrphenomab_fd_sequence_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomA/B sequence called LAL")

    monkeypatch.setattr(
        native_mod, "imrphenomab_fd_sequence_torch", recording_native
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
        long_asc_nodes=0.73,
        **spin_params,
        **SEQUENCE_PARAMS,
    )

    assert calls == 1
    for expected_array, result in zip(expected, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        np.testing.assert_allclose(
            result.numpy(), expected_array, rtol=3.0e-11, atol=0.0
        )


def test_imrphenomab_sequence_conventions(monkeypatch, preserve_scheme):
    monkeypatch.setenv("PYCBC_IMRPHENOMB_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    sample_points = [150.0, 20.0, 80.0, 10000.0]
    unordered = get_fd_waveform_sequence(
        approximant="IMRPhenomB",
        sample_points=sample_points,
        long_asc_nodes=0.73,
        spin1z=0.4,
        spin2z=-0.2,
        **SEQUENCE_PARAMS,
    )
    sorted_points = sorted(sample_points)
    ordered = get_fd_waveform_sequence(
        approximant="IMRPhenomB",
        sample_points=sorted_points,
        long_asc_nodes=0.0,
        spin1z=0.4,
        spin2z=-0.2,
        **SEQUENCE_PARAMS,
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
def test_imrphenomab_sequence_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    from pycbc.types.array_torch import TorchArrayData

    sample_points = [20.0, 31.25, 80.0, 150.0, 10000.0]
    monkeypatch.setenv("PYCBC_IMRPHENOMB_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomB",
        sample_points=sample_points,
        spin1z=0.4,
        spin2z=-0.2,
        **SEQUENCE_PARAMS,
    )
    reference_tensors = tuple(array._data.tensor.clone() for array in reference)

    _activate_scheme(_scheme.TorchScheme(device_name))

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomA/B sequence copied through NumPy")

    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomB",
        sample_points=sample_points,
        spin1z=0.4,
        spin2z=-0.2,
        **SEQUENCE_PARAMS,
    )
    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    tolerance = 3.0e-3 if device_name == "mps" else 3.0e-11
    for reference_tensor, result in zip(reference_tensors, actual):
        tensor = result._data.tensor
        assert tensor.device.type == device_name
        assert tensor.dtype == expected_dtype
        torch.testing.assert_close(
            tensor,
            reference_tensor.to(device=tensor.device, dtype=expected_dtype),
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
def test_imrphenomab_sequence_rejects_invalid_frequencies(
    sample_points,
    message,
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(ValueError, match=message):
        imrphenomab_fd_sequence_torch(
            approximant="IMRPhenomB",
            sample_points=sample_points,
            spin1z=0.4,
            spin2z=-0.2,
            **SEQUENCE_PARAMS,
        )
