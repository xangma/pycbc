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
from pycbc.waveform.seobnrv5hm_torch import (  # noqa: E402
    _LM_MODES,
    _active_mode_indices,
    seobnrv5hm_native_supported,
    seobnrv5hm_sequence_native_supported,
)


_ROM_FILENAME = "SEOBNRv5HMROM_v1.0.hdf5"
_WAVEFORM_DIR = Path(__file__).resolve().parent.parent
_NATIVE_FLAGS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_SEOBNRV5HM_NATIVE",
)
_BASE_PARAMS = {
    "mass1": 18.0,
    "mass2": 42.0,
    "spin1z": -0.3,
    "spin2z": 0.45,
    "distance": 500.0,
    "inclination": 0.7,
    "coa_phase": 0.3,
    "f_ref": 20.0,
    "mode_array": [(4, -3)],
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
        pytest.skip("SEOBNRv5HM ROM data is not available on LAL_DATA_PATH")


def _activate_scheme(state):
    _scheme.Scheme._single = None
    _scheme.mgr.state = state


def _clear_native_flags(monkeypatch):
    for name in _NATIVE_FLAGS:
        monkeypatch.delenv(name, raising=False)


def _generate(params, *, native):
    os.environ["PYCBC_SEOBNRV5HM_NATIVE"] = "1" if native else "0"
    state = _scheme.TorchScheme("cpu") if native else _scheme.CPUScheme()
    _activate_scheme(state)
    return get_fd_waveform(approximant="SEOBNRv5HM_ROM", **params)


def _generate_sequence(params, sample_points, *, native):
    os.environ["PYCBC_SEOBNRV5HM_NATIVE"] = "1" if native else "0"
    state = _scheme.TorchScheme("cpu") if native else _scheme.CPUScheme()
    _activate_scheme(state)
    return get_fd_waveform_sequence(
        approximant="SEOBNRv5HM_ROM",
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


def test_seobnrv5hm_regular_matches_lal():
    _require_rom_data()
    params = {
        **_BASE_PARAMS,
        "delta_f": 1.0,
        "f_lower": 20.0,
        "f_final": 1024.0,
        "long_asc_nodes": 0.37,
        "phase_order": 2.5,
        "amplitude_order": "3",
        "eccentricity_order": 4,
    }
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


def test_seobnrv5hm_sequence_matches_lal_and_dispatches_native(monkeypatch):
    _require_rom_data()
    params = {
        **_BASE_PARAMS,
        "phase_order": 2,
        "amplitude_order": 3,
        "eccentricity_order": 4,
    }
    sample_points = [20.0, 500.0, 31.5, 180.0, 1200.0]
    reference = _snapshot(_generate_sequence(params, sample_points, native=False))

    import pycbc.waveform.seobnrv5hm_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    native = native_module.seobnrv5hm_fd_sequence_torch
    native_calls = 0

    def recording_native(**native_params):
        nonlocal native_calls
        native_calls += 1
        return native(**native_params)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native SEOBNRv5HM_ROM sequence called lalsimulation")

    monkeypatch.setattr(
        native_module,
        "seobnrv5hm_fd_sequence_torch",
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


@pytest.mark.parametrize("device_name", ("cpu", "mps", "cuda"))
def test_seobnrv5hm_default_supported_requests_stay_on_device(
    device_name,
    monkeypatch,
):
    _require_rom_data()
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = {
        "mass1": 50.0,
        "mass2": 5.0,
        "spin1z": 0.998,
        "spin2z": -0.998,
        "distance": 500.0,
        "inclination": 0.7,
        "coa_phase": 0.3,
        "f_ref": 80.0,
        "mode_array": [(5, -5)],
    }
    regular_params = {
        **params,
        "delta_f": 4.0,
        "f_lower": 80.0,
        "f_final": 512.0,
    }
    sample_points = [80.0, 104.0, 160.0, 320.0, 500.0]
    regular_reference = _snapshot(_generate(regular_params, native=False))
    sequence_reference = _snapshot(
        _generate_sequence(params, sample_points, native=False)
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
    regular = get_fd_waveform(
        approximant="SEOBNRv5HM_ROM",
        **regular_params,
    )
    sequence = get_fd_waveform_sequence(
        approximant="SEOBNRv5HM_ROM",
        sample_points=sample_points,
        **params,
    )

    expected_dtype = torch.complex64 if device_name == "mps" else torch.complex128
    tolerance = {"cpu": 2.0e-10, "cuda": 1.0e-8, "mps": 1.0e-4}[device_name]
    for reference, actual in (
        (regular_reference, regular),
        (sequence_reference, sequence),
    ):
        assert all(series._data.tensor.device.type == device_name for series in actual)
        assert all(series._data.tensor.dtype == expected_dtype for series in actual)
        _assert_parity(reference, actual, relative_tolerance=tolerance)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_seobnrv5hm_mps_support_boundary():
    _activate_scheme(_scheme.TorchScheme("mps"))
    mass1 = 50.0
    mass2 = 5.0
    total_mass = mass1 + mass2
    symmetric_mass_ratio = mass1 * mass2 / total_mass**2
    minimum_start_mf = 1.0e-2 * (0.25 / symmetric_mass_ratio) ** (3.0 / 5.0)
    minimum_frequency = minimum_start_mf / (total_mass * lal.MTSUN_SI)
    common = {
        "approximant": "SEOBNRv5HM_ROM",
        "mass1": mass1,
        "mass2": mass2,
        "mode_array": [(5, -5)],
    }

    assert not seobnrv5hm_native_supported(
        {**common, "f_lower": minimum_frequency * 0.99}
    )
    assert seobnrv5hm_native_supported({**common, "f_lower": minimum_frequency * 1.01})
    assert not seobnrv5hm_sequence_native_supported(
        {
            **common,
            "sample_points": [
                minimum_frequency * 1.01,
                minimum_frequency * 0.99,
            ],
        }
    )
    assert seobnrv5hm_sequence_native_supported(
        {
            **common,
            "sample_points": [
                minimum_frequency * 1.01,
                minimum_frequency * 1.5,
            ],
        }
    )
    assert not seobnrv5hm_native_supported(
        {
            **common,
            "mass1": 55.0,
            "f_lower": 1000.0,
        }
    )

    _activate_scheme(_scheme.CPUScheme())
    assert seobnrv5hm_native_supported(
        {
            **common,
            "mass1": 500.0,
            "f_lower": 20.0,
        }
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_seobnrv5hm_mps_unsafe_requests_use_lalsimulation(monkeypatch):
    _require_rom_data()

    import pycbc.waveform.seobnrv5hm_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    original_regular = waveform_module.lalsimulation.SimInspiralChooseFDWaveform
    original_sequence = (
        waveform_module.lalsimulation.SimInspiralChooseFDWaveformSequence
    )
    regular_calls = 0
    sequence_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsafe MPS request reached native SEOBNRv5HM")

    def recording_regular(*args, **kwargs):
        nonlocal regular_calls
        regular_calls += 1
        return original_regular(*args, **kwargs)

    def recording_sequence(*args, **kwargs):
        nonlocal sequence_calls
        sequence_calls += 1
        return original_sequence(*args, **kwargs)

    monkeypatch.setattr(native_module, "seobnrv5hm_fd_torch", unexpected_native)
    monkeypatch.setattr(
        native_module,
        "seobnrv5hm_fd_sequence_torch",
        unexpected_native,
    )
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
    _activate_scheme(_scheme.TorchScheme("mps"))
    regular = get_fd_waveform(
        approximant="SEOBNRv5HM_ROM",
        delta_f=4.0,
        f_lower=20.0,
        f_final=256.0,
        **_BASE_PARAMS,
    )
    sequence = get_fd_waveform_sequence(
        approximant="SEOBNRv5HM_ROM",
        sample_points=[100.0, 20.0, 60.0, 200.0],
        **_BASE_PARAMS,
    )

    assert regular_calls == 1
    assert sequence_calls == 1
    for result in (*regular, *sequence):
        assert result._data.tensor.device.type == "mps"
        assert result._data.tensor.dtype == torch.complex64
        assert bool(torch.isfinite(result._data.tensor).all())


def test_seobnrv5hm_default_sequence_avoids_host_transfer(monkeypatch):
    _require_rom_data()

    import pycbc.waveform.waveform as waveform_module

    def reject_host_transfer(_self):
        raise AssertionError("native SEOBNRv5HM sequence transferred to NumPy")

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("default SEOBNRv5HM sequence called lalsimulation")

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    frequencies = Array([20.0, 40.0, 80.0, 120.0])
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lalsimulation,
    )
    with torch.no_grad():
        hp, hc = get_fd_waveform_sequence(
            approximant="SEOBNRv5HM_ROM",
            sample_points=frequencies,
            **_BASE_PARAMS,
        )

    assert hp._data.tensor.device.type == "cpu"
    assert hc._data.tensor.device.type == "cpu"


@pytest.mark.parametrize(
    ("interface", "lal_name", "native_name"),
    (
        (
            "regular",
            "SimInspiralChooseFDWaveform",
            "seobnrv5hm_fd_torch",
        ),
        (
            "sequence",
            "SimInspiralChooseFDWaveformSequence",
            "seobnrv5hm_fd_sequence_torch",
        ),
    ),
)
def test_seobnrv5hm_default_unsupported_options_use_lal_fallback(
    interface,
    lal_name,
    native_name,
    monkeypatch,
):
    _require_rom_data()

    import pycbc.waveform.seobnrv5hm_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    lal_generator = getattr(waveform_module.lalsimulation, lal_name)
    lal_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported SEOBNRv5HM request reached Torch")

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
    params = {**_BASE_PARAMS, "dchi0": 0.01}
    if interface == "regular":
        result = get_fd_waveform(
            approximant="SEOBNRv5HM_ROM",
            delta_f=4.0,
            f_lower=20.0,
            f_final=256.0,
            **params,
        )
    else:
        result = get_fd_waveform_sequence(
            approximant="SEOBNRv5HM_ROM",
            sample_points=[20.0, 50.0, 100.0, 200.0],
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
        ("PYCBC_SEOBNRV5HM_NATIVE", True),
    ),
)
def test_seobnrv5hm_default_native_opt_out(
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
            approximant="SEOBNRv5HM_ROM",
            delta_f=4.0,
            f_lower=20.0,
            f_final=256.0,
            **_BASE_PARAMS,
        )
    else:
        result = get_fd_waveform_sequence(
            approximant="SEOBNRv5HM_ROM",
            sample_points=[20.0, 50.0, 100.0, 200.0],
            **_BASE_PARAMS,
        )

    assert lal_calls == 1
    assert all(series._data.tensor.device.type == "cpu" for series in result)


def test_seobnrv5hm_empty_mode_array_avoids_rom_loading(monkeypatch):
    import pycbc.waveform.seobnrv5hm_torch as native_module

    def unexpected_rom_load(*_args, **_kwargs):
        raise AssertionError("empty mode_array loaded SEOBNRv5HM ROM data")

    monkeypatch.setattr(native_module, "_load_rom", unexpected_rom_load)
    params = {
        **_BASE_PARAMS,
        "mode_array": [],
        "delta_f": 1.0,
        "f_lower": 20.0,
        "f_final": 128.0,
    }
    hp, hc = _generate(params, native=True)
    sequence = _generate_sequence(
        {
            key: value
            for key, value in params.items()
            if key not in {"delta_f", "f_lower", "f_final"}
        },
        [20.0, 40.0, 100.0],
        native=True,
    )

    assert len(hp) == len(hc) == 129
    assert np.count_nonzero(hp.numpy()) == 0
    assert np.count_nonzero(hc.numpy()) == 0
    assert all(np.count_nonzero(series.numpy()) == 0 for series in sequence)


def test_seobnrv5hm_mode_array_semantics():
    assert _active_mode_indices(None) == tuple(range(len(_LM_MODES)))
    assert _active_mode_indices([(4, -3), (2, -2), (4, -3)]) == (0, 6)
    assert _active_mode_indices([(3, -2), (2, -1)]) == (2, 5)
    assert _active_mode_indices([]) == ()
    with pytest.raises(ValueError, match="positive-m"):
        _active_mode_indices([(4, 3)])
    with pytest.raises(ValueError, match="not available"):
        _active_mode_indices([(5, -4)])
    with pytest.raises(ValueError, match="pairs"):
        _active_mode_indices([2])


@pytest.mark.parametrize(
    ("params", "expected"),
    (
        ({}, True),
        ({"mode_array": []}, True),
        ({"mode_array": [(3, -2), (4, -3)]}, True),
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
        ({"mode_array": [(6, -6)]}, False),
        ({"numrel_data": "waveform.h5"}, False),
        ({"approximant": "SEOBNRv5_ROM"}, False),
    ),
)
def test_seobnrv5hm_native_support(params, expected):
    assert seobnrv5hm_native_supported(params) is expected
    assert seobnrv5hm_sequence_native_supported(params) is expected
