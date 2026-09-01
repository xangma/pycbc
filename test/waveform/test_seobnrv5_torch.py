import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.types import Array  # noqa: E402
from pycbc.types.array_torch import TorchArrayData  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.seobnrv5_torch import (  # noqa: E402
    _dominant_mode_selected,
    seobnrv5_native_supported,
    seobnrv5_sequence_native_supported,
)


_ROM_FILENAME = "SEOBNRv5ROM_v1.0.hdf5"
_WAVEFORM_DIR = Path(__file__).resolve().parent.parent
_NATIVE_FLAGS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_SEOBNRV5_NATIVE",
)
_BASE_PARAMS = {
    "mass1": 35.0,
    "mass2": 25.0,
    "spin1z": 0.2,
    "spin2z": -0.1,
    "distance": 500.0,
    "inclination": 0.7,
    "coa_phase": 0.3,
    "f_ref": 20.0,
}
_TIDAL_PARAMS = {
    "approximant": "SEOBNRv5_ROM_NRTidalv3",
    "mass1": 1.2,
    "mass2": 1.6,
    "spin1z": -0.04,
    "spin2z": 0.08,
    "lambda1": 900.0,
    "lambda2": 400.0,
    "distance": 100.0,
    "inclination": 0.7,
    "coa_phase": 0.3,
    "f_ref": 50.0,
}


@pytest.fixture(autouse=True)
def preserve_process_state():
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    old_environment = {name: os.environ.get(name) for name in _NATIVE_FLAGS}
    try:
        yield
    finally:
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def _require_rom_data():
    search_paths = [_WAVEFORM_DIR]
    search_paths.extend(
        Path(path)
        for path in os.environ.get("LAL_DATA_PATH", "").split(os.pathsep)
        if path
    )
    if not any((path / _ROM_FILENAME).is_file() for path in search_paths):
        pytest.skip("SEOBNRv5 ROM data is not available on LAL_DATA_PATH")


def _activate_scheme(state):
    _scheme.Scheme._single = None
    _scheme.mgr.state = state


def _clear_native_flags(monkeypatch):
    for name in _NATIVE_FLAGS:
        monkeypatch.delenv(name, raising=False)


def _generate(params, *, native):
    params = dict(params)
    approximant = params.pop("approximant", "SEOBNRv5_ROM")
    os.environ["PYCBC_SEOBNRV5_NATIVE"] = "1" if native else "0"
    state = _scheme.TorchScheme("cpu") if native else _scheme.CPUScheme()
    _activate_scheme(state)
    return get_fd_waveform(approximant=approximant, **params)


def _generate_sequence(params, sample_points, *, native):
    params = dict(params)
    approximant = params.pop("approximant", "SEOBNRv5_ROM")
    os.environ["PYCBC_SEOBNRV5_NATIVE"] = "1" if native else "0"
    state = _scheme.TorchScheme("cpu") if native else _scheme.CPUScheme()
    _activate_scheme(state)
    return get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )


def _snapshot(series_pair):
    return tuple(series.numpy().copy() for series in series_pair)


def _assert_parity(reference, actual, relative_tolerance=1.0e-10):
    for expected, result in zip(reference, actual):
        values = result.numpy()
        np.testing.assert_array_equal(values == 0.0, expected == 0.0)
        nonzero = expected != 0.0
        assert nonzero.any(), "waveform contains no non-zero samples"
        relative_error = np.linalg.norm(
            values[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < relative_tolerance


@pytest.mark.parametrize(
    "params",
    (
        {
            **_BASE_PARAMS,
            "delta_f": 0.5,
            "f_lower": 20.0,
            "f_final": 0.0,
            "long_asc_nodes": 0.37,
            "phase_order": 2.5,
            "amplitude_order": "3",
            "eccentricity_order": 4,
        },
        {
            **_BASE_PARAMS,
            "mass1": 18.0,
            "mass2": 42.0,
            "spin1z": -0.3,
            "spin2z": 0.45,
            "delta_f": 0.5,
            "f_lower": 20.0,
            "f_final": 1024.0,
            "mode_array": [(2, -2)],
        },
    ),
)
def test_seobnrv5_regular_matches_lal(params):
    _require_rom_data()
    cpu = _generate(params, native=False)
    reference = _snapshot(cpu)
    metadata = (len(cpu[0]), float(cpu[0].epoch), float(cpu[0].delta_f))

    actual = _generate(params, native=True)

    assert (
        len(actual[0]),
        float(actual[0].epoch),
        float(actual[0].delta_f),
    ) == metadata
    assert all(series._data.tensor.dtype == torch.complex128 for series in actual)
    _assert_parity(reference, actual)


def test_seobnrv5_sequence_matches_lal_and_dispatches_native(monkeypatch):
    _require_rom_data()
    params = {
        **_BASE_PARAMS,
        "mass1": 18.0,
        "mass2": 42.0,
        "spin1z": -0.3,
        "spin2z": 0.45,
        "mode_array": [(2, -2)],
        "phase_order": 2,
        "amplitude_order": 3,
        "eccentricity_order": 4,
    }
    sample_points = [20.0, 400.0, 22.5, 150.0, 1000.0, 4000.0]
    reference = _snapshot(
        _generate_sequence(params, sample_points, native=False)
    )

    import pycbc.waveform.seobnrv5_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    native = native_module.seobnrv5_fd_sequence_torch
    native_calls = 0

    def recording_native(**native_params):
        nonlocal native_calls
        native_calls += 1
        return native(**native_params)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native SEOBNRv5_ROM sequence called lalsimulation")

    monkeypatch.setattr(
        native_module,
        "seobnrv5_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lalsimulation,
    )
    actual = _generate_sequence(params, sample_points, native=True)

    assert native_calls == 1
    assert all(series._data.tensor.device.type == "cpu" for series in actual)
    assert all(series._data.tensor.dtype == torch.complex128 for series in actual)
    _assert_parity(reference, actual)


def test_seobnrv5_regular_dispatch_does_not_call_lalsimulation(monkeypatch):
    _require_rom_data()
    params = {
        **_BASE_PARAMS,
        "delta_f": 1.0,
        "f_lower": 20.0,
        "f_final": 256.0,
        "phase_order": 2,
        "amplitude_order": 3,
        "eccentricity_order": 4,
    }

    import pycbc.waveform.waveform as waveform_module

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native SEOBNRv5_ROM called lalsimulation")

    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lalsimulation,
    )
    hp, hc = _generate(params, native=True)

    assert len(hp) == len(hc) == 257
    assert isinstance(hp._data.tensor, torch.Tensor)


@pytest.mark.parametrize(
    "params",
    (
        {
            **_TIDAL_PARAMS,
            "delta_f": 4.0,
            "f_lower": 30.0,
            "f_final": 0.0,
            "long_asc_nodes": 0.37,
        },
        {
            **_TIDAL_PARAMS,
            "mass1": 1.7,
            "mass2": 1.1,
            "spin1z": 0.12,
            "spin2z": -0.06,
            "lambda1": 300.0,
            "lambda2": 1100.0,
            "dquad_mon1": 1.7,
            "dquad_mon2": 3.2,
            "delta_f": 4.0,
            "f_lower": 30.0,
            "f_final": 4097.0,
        },
    ),
)
def test_seobnrv5_nrtidalv3_regular_matches_lal(params):
    _require_rom_data()
    cpu = _generate(params, native=False)
    reference = _snapshot(cpu)
    metadata = (len(cpu[0]), float(cpu[0].epoch), float(cpu[0].delta_f))

    actual = _generate(params, native=True)

    assert (
        len(actual[0]),
        float(actual[0].epoch),
        float(actual[0].delta_f),
    ) == metadata
    assert all(series._data.tensor.dtype == torch.complex128 for series in actual)
    _assert_parity(reference, actual, relative_tolerance=2.0e-10)


def test_seobnrv5_nrtidalv3_sequence_matches_lal_and_dispatches_native(
    monkeypatch,
):
    _require_rom_data()
    sample_points = [30.0, 45.0, 100.0, 300.0, 800.0, 1600.0, 3000.0]
    reference = _snapshot(
        _generate_sequence(_TIDAL_PARAMS, sample_points, native=False)
    )

    import pycbc.waveform.waveform as waveform_module

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError(
            "native SEOBNRv5_ROM_NRTidalv3 sequence called lalsimulation"
        )

    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lalsimulation,
    )
    actual = _generate_sequence(_TIDAL_PARAMS, sample_points, native=True)

    assert all(series._data.tensor.device.type == "cpu" for series in actual)
    assert all(series._data.tensor.dtype == torch.complex128 for series in actual)
    _assert_parity(reference, actual, relative_tolerance=2.0e-10)


@pytest.mark.parametrize(
    ("device_name", "approximant", "base_params", "f_lower", "sample_points"),
    (
        pytest.param(
            "cpu",
            "SEOBNRv5_ROM",
            _BASE_PARAMS,
            20.0,
            [20.0, 30.0, 50.0, 100.0, 400.0],
            id="cpu-bbh",
        ),
        pytest.param(
            "cpu",
            "SEOBNRv5_ROM_NRTidalv3",
            _TIDAL_PARAMS,
            30.0,
            [30.0, 100.0, 500.0, 1000.0],
            id="cpu-nrtidalv3",
        ),
        pytest.param(
            "mps",
            "SEOBNRv5_ROM",
            _BASE_PARAMS,
            20.0,
            [20.0, 30.0, 50.0, 100.0, 400.0],
            id="mps-bbh",
        ),
        pytest.param(
            "cuda",
            "SEOBNRv5_ROM",
            _BASE_PARAMS,
            20.0,
            [20.0, 30.0, 50.0, 100.0, 400.0],
            id="cuda-bbh",
        ),
        pytest.param(
            "cuda",
            "SEOBNRv5_ROM_NRTidalv3",
            _TIDAL_PARAMS,
            30.0,
            [30.0, 100.0, 500.0, 1000.0],
            id="cuda-nrtidalv3",
        ),
    ),
)
def test_seobnrv5_default_supported_requests_stay_on_device(
    device_name,
    approximant,
    base_params,
    f_lower,
    sample_points,
    monkeypatch,
):
    _require_rom_data()
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = dict(base_params)
    params.pop("approximant", None)
    regular_params = {
        **params,
        "approximant": approximant,
        "delta_f": 2.0,
        "f_lower": f_lower,
        "f_final": 256.0,
    }
    regular_reference = _snapshot(_generate(regular_params, native=False))
    sequence_reference = _snapshot(
        _generate_sequence(
            {**params, "approximant": approximant},
            sample_points,
            native=False,
        )
    )

    import pycbc.waveform.waveform as waveform_module

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("default-supported request called lalsimulation")

    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lalsimulation,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lalsimulation,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme(device_name))
    regular_params.pop("approximant")
    regular = get_fd_waveform(approximant=approximant, **regular_params)
    sequence = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    tolerance = {"cpu": 2.0e-10, "cuda": 1.0e-8, "mps": 1.0e-4}[
        device_name
    ]
    for reference, actual in (
        (regular_reference, regular),
        (sequence_reference, sequence),
    ):
        assert all(
            series._data.tensor.device.type == device_name for series in actual
        )
        assert all(series._data.tensor.dtype == expected_dtype for series in actual)
        _assert_parity(reference, actual, relative_tolerance=tolerance)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
@pytest.mark.parametrize(
    ("approximant", "base_params", "f_lower"),
    (
        ("SEOBNRv5_ROM", _BASE_PARAMS, 2.0),
        ("SEOBNRv5_ROM_NRTidalv3", _TIDAL_PARAMS, 30.0),
    ),
)
def test_seobnrv5_mps_unsafe_requests_use_lalsimulation(
    approximant,
    base_params,
    f_lower,
    monkeypatch,
):
    _require_rom_data()

    import pycbc.waveform.waveform as waveform_module

    original_regular = waveform_module.lalsimulation.SimInspiralChooseFDWaveform
    original_sequence = (
        waveform_module.lalsimulation.SimInspiralChooseFDWaveformSequence
    )
    regular_calls = 0
    sequence_calls = 0

    def recording_regular(*args, **kwargs):
        nonlocal regular_calls
        regular_calls += 1
        return original_regular(*args, **kwargs)

    def recording_sequence(*args, **kwargs):
        nonlocal sequence_calls
        sequence_calls += 1
        return original_sequence(*args, **kwargs)

    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_regular,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_sequence,
    )
    _clear_native_flags(monkeypatch)
    monkeypatch.setenv("PYCBC_SEOBNRV5_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("mps"))
    params = dict(base_params)
    params.pop("approximant", None)
    regular = get_fd_waveform(
        approximant=approximant,
        delta_f=4.0,
        f_lower=f_lower,
        f_final=256.0 if approximant == "SEOBNRv5_ROM" else 0.0,
        **params,
    )
    sequence = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=(
            [50.0, f_lower, 100.0, 200.0]
            if approximant == "SEOBNRv5_ROM"
            else [f_lower, 50.0, 100.0, 200.0]
        ),
        **params,
    )

    assert regular_calls == 1
    assert sequence_calls == 1
    for result in (*regular, *sequence):
        assert result._data.tensor.device.type == "mps"
        assert result._data.tensor.dtype == torch.complex64
        assert bool(torch.isfinite(result._data.tensor).all())


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_seobnrv5_mps_support_boundary():
    _activate_scheme(_scheme.TorchScheme("mps"))
    mass1 = 45.0
    mass2 = 15.0
    total_mass = mass1 + mass2
    symmetric_mass_ratio = mass1 * mass2 / total_mass**2
    minimum_start_mf = 2.0e-3 * (
        0.25 / symmetric_mass_ratio
    ) ** (3.0 / 5.0)
    minimum_frequency = minimum_start_mf / (total_mass * lal.MTSUN_SI)
    common = {
        "approximant": "SEOBNRv5_ROM",
        "mass1": mass1,
        "mass2": mass2,
    }

    assert not seobnrv5_native_supported(
        {**common, "f_lower": minimum_frequency * 0.99}
    )
    assert seobnrv5_native_supported(
        {**common, "f_lower": minimum_frequency * 1.01}
    )
    assert not seobnrv5_sequence_native_supported(
        {**common, "sample_points": [minimum_frequency * 0.99, 100.0]}
    )
    assert not seobnrv5_sequence_native_supported(
        {
            **common,
            "sample_points": [
                minimum_frequency * 1.01,
                minimum_frequency * 0.99,
            ],
        }
    )
    assert seobnrv5_sequence_native_supported(
        {**common, "sample_points": [minimum_frequency * 1.01, 100.0]}
    )
    assert not seobnrv5_native_supported(
        {
            **common,
            "approximant": "SEOBNRv5_ROM_NRTidalv3",
            "f_lower": 1000.0,
        }
    )


@pytest.mark.parametrize(
    ("approximant", "base_params", "sample_points"),
    (
        (
            "SEOBNRv5_ROM",
            _BASE_PARAMS,
            [20.0, 30.0, 100.0, 400.0],
        ),
        (
            "SEOBNRv5_ROM_NRTidalv3",
            _TIDAL_PARAMS,
            [30.0, 100.0, 500.0, 1000.0],
        ),
    ),
)
def test_seobnrv5_default_sequence_avoids_host_transfer(
    approximant,
    base_params,
    sample_points,
    monkeypatch,
):
    _require_rom_data()

    import pycbc.waveform.waveform as waveform_module

    def reject_host_transfer(_self):
        raise AssertionError("native SEOBNRv5 sequence transferred to NumPy")

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("default SEOBNRv5 sequence called lalsimulation")

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    frequencies = Array(sample_points)
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lalsimulation,
    )
    params = dict(base_params)
    params.pop("approximant", None)
    with torch.no_grad():
        hp, hc = get_fd_waveform_sequence(
            approximant=approximant,
            sample_points=frequencies,
            **params,
        )

    assert isinstance(hp._data.tensor, torch.Tensor)
    assert isinstance(hc._data.tensor, torch.Tensor)
    assert hp._data.tensor.device.type == "cpu"
    assert hc._data.tensor.device.type == "cpu"


@pytest.mark.parametrize(
    ("interface", "lal_name", "native_name"),
    (
        (
            "regular",
            "SimInspiralChooseFDWaveform",
            "seobnrv5_fd_torch",
        ),
        (
            "sequence",
            "SimInspiralChooseFDWaveformSequence",
            "seobnrv5_fd_sequence_torch",
        ),
    ),
)
@pytest.mark.parametrize(
    ("approximant", "base_params"),
    (
        ("SEOBNRv5_ROM", _BASE_PARAMS),
        ("SEOBNRv5_ROM_NRTidalv3", _TIDAL_PARAMS),
    ),
)
def test_seobnrv5_default_unsupported_options_use_lal_fallback(
    interface,
    lal_name,
    native_name,
    approximant,
    base_params,
    monkeypatch,
):
    _require_rom_data()

    import pycbc.waveform.seobnrv5_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    lal_generator = getattr(waveform_module.lalsimulation, lal_name)
    lal_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported SEOBNRv5 request reached Torch")

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(native_module, native_name, unexpected_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        lal_name,
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = dict(base_params)
    params.pop("approximant", None)
    params["dchi0"] = 0.01
    if interface == "regular":
        result = get_fd_waveform(
            approximant=approximant,
            delta_f=4.0,
            f_lower=30.0,
            f_final=256.0,
            **params,
        )
    else:
        result = get_fd_waveform_sequence(
            approximant=approximant,
            sample_points=[30.0, 50.0, 100.0, 200.0],
            **params,
        )

    assert lal_calls == 1
    assert all(series._data.tensor.device.type == "cpu" for series in result)


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
        ("PYCBC_SEOBNRV5_NATIVE", True),
    ),
)
def test_seobnrv5_default_native_opt_out(
    interface,
    lal_name,
    disabled_flag,
    global_enabled,
    monkeypatch,
):
    _require_rom_data()

    import pycbc.waveform.waveform as waveform_module

    _clear_native_flags(monkeypatch)
    if global_enabled:
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "1")
    monkeypatch.setenv(disabled_flag, "0")
    lal_generator = getattr(waveform_module.lalsimulation, lal_name)
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        waveform_module.lalsimulation,
        lal_name,
        recording_lal,
    )
    _activate_scheme(_scheme.TorchScheme("cpu"))
    if interface == "regular":
        result = get_fd_waveform(
            approximant="SEOBNRv5_ROM",
            delta_f=2.0,
            f_lower=20.0,
            f_final=128.0,
            **_BASE_PARAMS,
        )
    else:
        result = get_fd_waveform_sequence(
            approximant="SEOBNRv5_ROM",
            sample_points=[20.0, 40.0, 80.0, 120.0],
            **_BASE_PARAMS,
        )

    assert lal_calls == 1
    assert all(series._data.tensor.device.type == "cpu" for series in result)


@pytest.mark.parametrize(
    ("params", "expected"),
    (
        ({}, True),
        ({"mode_array": [(2, -2)]}, True),
        ({"spin1x": 0.1}, False),
        ({"phase_order": 2}, True),
        ({"amplitude_order": 3}, True),
        ({"eccentricity_order": 4}, True),
        ({"phase_order": 2.5}, True),
        ({"amplitude_order": "3"}, True),
        ({"eccentricity_order": 4.0}, False),
        ({"phase_order": 1 << 31}, False),
        ({"spin_order": 2}, False),
        ({"tidal_order": 12}, False),
        ({"lambda1": 100.0}, False),
        ({"dchi3": 0.1}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"mode_array": [(3, -3)]}, False),
        ({"mode_array": []}, False),
        ({"numrel_data": "waveform.h5"}, False),
        ({"approximant": "SEOBNRv5_ROM_NRTidalv3"}, True),
        (
            {
                "approximant": "SEOBNRv5_ROM_NRTidalv3",
                "lambda1": 100.0,
                "lambda2": 200.0,
                "phase_order": 2,
                "amplitude_order": 3,
                "eccentricity_order": 4,
            },
            True,
        ),
        (
            {
                "approximant": "SEOBNRv5_ROM_NRTidalv3",
                "lambda1": -1.0,
            },
            False,
        ),
        (
            {
                "approximant": "SEOBNRv5_ROM_NRTidalv3",
                "dquad_mon1": -2.0,
            },
            False,
        ),
        (
            {
                "approximant": "SEOBNRv5_ROM_NRTidalv3",
                "lambda_octu1": 1.0,
            },
            False,
        ),
    ),
)
def test_seobnrv5_native_support(params, expected):
    assert seobnrv5_native_supported(params) is expected
    assert seobnrv5_sequence_native_supported(params) is expected


def test_seobnrv5_mode_array_validation():
    assert _dominant_mode_selected(None)
    assert _dominant_mode_selected([(2, -2)])
    assert not _dominant_mode_selected([])
    with pytest.raises(ValueError, match="directly modeled"):
        _dominant_mode_selected([(2, 2)])
    with pytest.raises(ValueError, match="not available"):
        _dominant_mode_selected([(3, -3)])
    with pytest.raises(ValueError, match="pairs"):
        _dominant_mode_selected([2])
