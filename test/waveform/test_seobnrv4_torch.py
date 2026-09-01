import os
import warnings
from pathlib import Path

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
from pycbc.waveform.seobnrv4_torch import (  # noqa: E402
    _clear_rom_cache,
    seobnrv4_rom_native_supported,
    seobnrv4_rom_sequence_native_supported,
)
from pycbc.waveform.nsbh_torch import (  # noqa: E402
    bbh_final_mass_non_precessing_uib2016,
    bbh_final_spin_non_precessing_uib2016,
    bhns_mass_aligned,
    bhns_spin_aligned,
    nsbh_compactness_from_lambda,
    nsbh_r_kerr_isco,
    nsbh_torus_mass_fit,
    nsbh_xi_tide,
    seobnrv4_nsbh_amplitude,
)


_ROM_FILENAME = "SEOBNRv4ROM_v3.0.hdf5"
_WAVEFORM_DIR = Path(__file__).resolve().parent.parent
_NATIVE_FLAGS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_SEOBNRV4_NATIVE",
)
_BASE_PARAMS = dict(
    distance=400.0,
    inclination=0.7,
    coa_phase=0.5,
)


@pytest.fixture(scope="module", autouse=True)
def require_rom_data():
    search_paths = [_WAVEFORM_DIR]
    search_paths.extend(
        Path(path)
        for path in os.environ.get("LAL_DATA_PATH", "").split(os.pathsep)
        if path
    )
    if not any((path / _ROM_FILENAME).is_file() for path in search_paths):
        pytest.skip(f"{_ROM_FILENAME} is not available on LAL_DATA_PATH")
    yield
    _clear_rom_cache()


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


def _activate_scheme(state):
    _scheme.Scheme._single = None
    _scheme.mgr.state = state


def _clear_native_flags(monkeypatch):
    for flag in _NATIVE_FLAGS:
        monkeypatch.delenv(flag, raising=False)


def _generate(params, *, native, device=None, approximant="SEOBNRv4_ROM"):
    os.environ["PYCBC_SEOBNRV4_NATIVE"] = "1" if native else "0"
    if device is None:
        _activate_scheme(_scheme.CPUScheme())
    else:
        _activate_scheme(_scheme.TorchScheme(device))
    return get_fd_waveform(approximant=approximant, **params)


def _generate_sequence(
    params,
    sample_points,
    *,
    native,
    device=None,
    approximant="SEOBNRv4_ROM",
):
    os.environ["PYCBC_SEOBNRV4_NATIVE"] = "1" if native else "0"
    if device is None:
        _activate_scheme(_scheme.CPUScheme())
    else:
        _activate_scheme(_scheme.TorchScheme(device))
    return get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )


def _snapshot(series_pair):
    return tuple(
        (len(series), series.delta_f, float(series.epoch), series.numpy().copy())
        for series in series_pair
    )


def _assert_parity(
    reference, actual, relative_tolerance, *, exact_zero_mask=True
):
    for expected, result in zip(reference, actual):
        expected_length, expected_delta_f, expected_epoch, expected_array = expected
        assert len(result) == expected_length
        assert result.delta_f == expected_delta_f
        assert float(result.epoch) == expected_epoch

        result_array = result.numpy()
        if exact_zero_mask:
            np.testing.assert_array_equal(
                result_array == 0.0, expected_array == 0.0
            )
        nonzero = expected_array != 0.0
        assert nonzero.any(), "waveform contains no non-zero bins"
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected_array[nonzero]
        ) / np.linalg.norm(expected_array[nonzero])
        assert relative_error < relative_tolerance


def _assert_sequence_parity(
    reference, actual, relative_tolerance, *, exact_zero_mask=True
):
    for expected_array, result in zip(reference, actual):
        result_array = result.numpy()
        if exact_zero_mask:
            np.testing.assert_array_equal(
                result_array == 0.0, expected_array == 0.0
            )
        nonzero = expected_array != 0.0
        assert nonzero.any(), "waveform contains no non-zero samples"
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected_array[nonzero]
        ) / np.linalg.norm(expected_array[nonzero])
        assert relative_error < relative_tolerance


@pytest.mark.parametrize(
    ("native", "reference_name", "arguments"),
    [
        (
            nsbh_compactness_from_lambda,
            "SimNSBH_compactness_from_lambda",
            (800.0,),
        ),
        (nsbh_r_kerr_isco, "SimNSBH_rKerrISCO", (0.8,)),
        (nsbh_xi_tide, "SimNSBH_xi_tide", (5.0, 0.8, 0.8)),
        (nsbh_torus_mass_fit, "SimNSBH_torus_mass_fit", (5.0, 0.8, 0.16)),
        (
            bbh_final_mass_non_precessing_uib2016,
            "bbh_final_mass_non_precessing_UIB2016",
            (8.0, 1.4, 0.8, 0.0),
        ),
        (
            bbh_final_spin_non_precessing_uib2016,
            "bbh_final_spin_non_precessing_UIB2016",
            (8.0, 1.4, 0.8, 0.0),
        ),
        (bhns_mass_aligned, "BHNS_mass_aligned", (8.0, 1.4, 0.8, 800.0)),
        (bhns_spin_aligned, "BHNS_spin_aligned", (8.0, 1.4, 0.8, 800.0)),
    ],
)
def test_nsbh_scalar_fits_match_lal(native, reference_name, arguments):
    expected = getattr(lalsimulation, reference_name)(*arguments)
    assert native(*arguments) == pytest.approx(
        expected, rel=2.0e-12, abs=2.0e-14
    )


@pytest.mark.parametrize(
    ("mass1", "mass2", "spin1z", "lambda2"),
    [
        (8.0, 1.4, 0.8, 800.0),
        (5.0, 1.4, -0.2, 1000.0),
        (20.0, 2.5, 0.5, 0.0),
    ],
)
def test_seobnrv4_nsbh_amplitude_matches_lal(
    mass1, mass2, spin1z, lambda2
):
    frequencies = np.array(
        [20.0, 100.0, 500.0, 1000.0, 2000.0, 4096.0]
    )
    lal_frequencies = lal.CreateREAL8Sequence(len(frequencies))
    lal_frequencies.data[:] = frequencies
    expected = lal.CreateREAL8Sequence(len(frequencies))
    lalsimulation.SEOBNRv4ROMNSBHAmplitudeCorrectionFrequencySeries(
        expected,
        lal_frequencies,
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        spin1z,
        lambda2,
    )

    frequency_tensor = torch.as_tensor(frequencies, dtype=torch.float64)
    actual = seobnrv4_nsbh_amplitude(
        frequency_tensor, mass1, mass2, spin1z, lambda2
    )

    assert actual.device == frequency_tensor.device
    assert actual.dtype == frequency_tensor.dtype
    np.testing.assert_allclose(
        actual.numpy(), expected.data, rtol=2.0e-11, atol=2.0e-13
    )


@pytest.mark.parametrize(
    "params",
    [
        {
            **_BASE_PARAMS,
            "mass1": 30.0,
            "mass2": 25.0,
            "spin1z": 0.2,
            "spin2z": -0.1,
            "delta_f": 0.25,
            "f_lower": 20.0,
            "f_final": 0.0,
            "f_ref": 20.0,
        },
        {
            **_BASE_PARAMS,
            "mass1": 12.0,
            "mass2": 40.0,
            "spin1z": -0.4,
            "spin2z": 0.7,
            "delta_f": 0.25,
            "f_lower": 17.3,
            "f_final": 133.3,
            "f_ref": 37.0,
            "long_asc_nodes": 0.23,
        },
        {
            **_BASE_PARAMS,
            "mass1": 40.0,
            "mass2": 1.0,
            "spin1z": 0.997,
            "spin2z": -0.3,
            "delta_f": 0.5,
            "f_lower": 10.0,
            "f_final": 0.0,
            "f_ref": 31.0,
        },
        {
            **_BASE_PARAMS,
            "mass1": 25.0,
            "mass2": 25.0,
            "spin1z": 1.0,
            "spin2z": -1.0,
            "delta_f": 0.5,
            "f_lower": 20.0,
            "f_final": 4096.0,
            "f_ref": 0.0,
        },
        {
            **_BASE_PARAMS,
            "mass1": 30.0,
            "mass2": 25.0,
            "spin1z": 0.2,
            "spin2z": -0.1,
            "delta_f": 128.0 / 8192.0,
            "f_lower": 20.0,
            "f_final": 128.001,
            "f_ref": 20.0,
        },
    ],
)
def test_seobnrv4_rom_cpu_torch_parity(params):
    reference = _snapshot(_generate(params, native=False))
    actual = _generate(params, native=True, device="cpu")
    _assert_parity(reference, actual, relative_tolerance=1.0e-8)


@pytest.mark.parametrize(
    ("approximant", "params"),
    [
        (
            "SEOBNRv4_ROM_NRTidal",
            {
                **_BASE_PARAMS,
                "mass1": 1.4,
                "mass2": 1.2,
                "spin1z": 0.05,
                "spin2z": -0.02,
                "lambda1": 400.0,
                "lambda2": 800.0,
                "delta_f": 0.5,
                "f_lower": 20.0,
                "f_ref": 30.0,
            },
        ),
        (
            "SEOBNRv4_ROM_NRTidalv2",
            {
                **_BASE_PARAMS,
                "mass1": 1.15,
                "mass2": 1.55,
                "spin1z": 0.1,
                "spin2z": -0.15,
                "lambda1": 900.0,
                "lambda2": 300.0,
                "delta_f": 0.5,
                "f_lower": 18.0,
                "f_final": 4096.1,
                "f_ref": 0.0,
                "long_asc_nodes": 0.31,
            },
        ),
        (
            "SEOBNRv4_ROM_NRTidalv2_NSBH",
            {
                **_BASE_PARAMS,
                "mass1": 8.0,
                "mass2": 1.4,
                "spin1z": 0.8,
                "spin2z": 0.0,
                "lambda1": 0.0,
                "lambda2": 800.0,
                "delta_f": 0.5,
                "f_lower": 20.0,
                "f_final": 4096.1,
                "f_ref": 30.0,
                "long_asc_nodes": 0.23,
            },
        ),
    ],
)
def test_seobnrv4_rom_nrtidal_parity(approximant, params):
    reference = _snapshot(
        _generate(params, native=False, approximant=approximant)
    )
    actual = _generate(
        params, native=True, device="cpu", approximant=approximant
    )
    tolerance = (
        1.0e-7
        if approximant == "SEOBNRv4_ROM_NRTidalv2_NSBH"
        else 1.0e-8
    )
    _assert_parity(reference, actual, relative_tolerance=tolerance)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, True),
        ({"long_asc_nodes": 0.4}, True),
        ({"lambda1": 0.0, "dchi3": 0.0}, True),
        ({"approximant": "SEOBNRv4"}, False),
        (
            {
                "approximant": "SEOBNRv4_ROM_NRTidalv2",
                "lambda1": 400.0,
                "lambda2": 800.0,
            },
            True,
        ),
        (
            {"approximant": "SEOBNRv4_ROM_NRTidal", "lambda1": -1.0},
            False,
        ),
        (
            {
                "approximant": "SEOBNRv4_ROM_NRTidalv2_NSBH",
                "mass1": 8.0,
                "mass2": 1.4,
                "lambda1": 0.0,
                "lambda2": 800.0,
            },
            True,
        ),
        (
            {
                "approximant": "SEOBNRv4_ROM_NRTidalv2_NSBH",
                "mass1": 1.4,
                "mass2": 8.0,
                "lambda1": 0.0,
                "lambda2": 800.0,
            },
            False,
        ),
        (
            {
                "approximant": "SEOBNRv4_ROM_NRTidalv2_NSBH",
                "mass1": 8.0,
                "mass2": 1.4,
                "lambda1": 1.0,
                "lambda2": 800.0,
            },
            False,
        ),
        (
            {
                "approximant": "SEOBNRv4_ROM_NRTidalv2_NSBH",
                "mass1": 8.0,
                "mass2": 3.1,
                "lambda1": 0.0,
                "lambda2": 800.0,
            },
            False,
        ),
        (
            {
                "approximant": "SEOBNRv4_ROM_NRTidalv2_NSBH",
                "mass1": 8.0,
                "mass2": 1.4,
                "lambda1": 0.0,
                "lambda2": 5000.1,
            },
            False,
        ),
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
        ({"eccentricity": 0.1}, False),
        ({"dchi3": 0.1}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"frame_axis": 1}, False),
        ({"numrel_data": "waveform.h5"}, False),
    ],
)
def test_seobnrv4_rom_native_support_boundary(params, expected):
    assert seobnrv4_rom_native_supported(params) is expected


@pytest.mark.parametrize(
    ("approximant", "params", "sample_points"),
    [
        (
            "SEOBNRv4_ROM",
            {
                **_BASE_PARAMS,
                "mass1": 35.0,
                "mass2": 28.0,
                "spin1z": 0.2,
                "spin2z": -0.1,
                "f_ref": 30.0,
                "long_asc_nodes": 0.37,
                "phase_order": 2,
                "amplitude_order": 3,
                "eccentricity_order": 4,
            },
            [20.0, 23.5, 30.0, 45.0, 100.0, 400.0, 700.0, 10000.0],
        ),
        (
            "SEOBNRv4_ROM",
            {
                **_BASE_PARAMS,
                "mass1": 18.0,
                "mass2": 42.0,
                "spin1z": -0.4,
                "spin2z": 0.7,
                "f_ref": 0.0,
                "long_asc_nodes": 0.21,
            },
            [17.3, 400.0, 22.0, 150.0],
        ),
        (
            "SEOBNRv4_ROM_NRTidal",
            {
                **_BASE_PARAMS,
                "mass1": 1.4,
                "mass2": 1.2,
                "spin1z": 0.05,
                "spin2z": -0.02,
                "lambda1": 400.0,
                "lambda2": 800.0,
                "f_ref": 30.0,
            },
            [20.0, 30.0, 100.0, 500.0, 1000.0, 2048.0, 10000.0],
        ),
        (
            "SEOBNRv4_ROM_NRTidalv2",
            {
                **_BASE_PARAMS,
                "mass1": 1.15,
                "mass2": 1.55,
                "spin1z": 0.1,
                "spin2z": -0.15,
                "lambda1": 900.0,
                "lambda2": 300.0,
                "f_ref": 0.0,
                "long_asc_nodes": 0.31,
            },
            [18.0, 30.0, 150.0, 1024.0, 1500.0, 3000.0],
        ),
        (
            "SEOBNRv4_ROM_NRTidalv2_NSBH",
            {
                **_BASE_PARAMS,
                "mass1": 8.0,
                "mass2": 1.4,
                "spin1z": 0.8,
                "spin2z": 0.0,
                "lambda1": 0.0,
                "lambda2": 800.0,
                "f_ref": 30.0,
            },
            [20.0, 30.0, 100.0, 500.0, 1000.0, 2048.0, 4096.0, 10000.0],
        ),
    ],
)
def test_seobnrv4_rom_sequence_defaults_to_native_without_calling_lal(
    approximant, params, sample_points, monkeypatch
):
    reference = tuple(
        series.numpy().copy()
        for series in _generate_sequence(
            params,
            sample_points,
            native=False,
            approximant=approximant,
        )
    )

    import pycbc.waveform.seobnrv4_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    native = native_module.seobnrv4_fd_sequence_torch
    native_calls = 0

    def recording_native(**native_params):
        nonlocal native_calls
        native_calls += 1
        return native(**native_params)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native SEOBNRv4_ROM sequence called lalsimulation")

    monkeypatch.setattr(
        native_module,
        "seobnrv4_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lalsimulation,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )

    assert native_calls == 1
    assert all(series._data.tensor.device.type == "cpu" for series in actual)
    assert all(series._data.tensor.dtype == torch.complex128 for series in actual)
    _assert_sequence_parity(reference, actual, relative_tolerance=1.0e-8)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, True),
        ({"approximant": "SEOBNRv4_ROM_NRTidal"}, True),
        ({"approximant": "SEOBNRv4_ROM_NRTidalv2"}, True),
        (
            {
                "approximant": "SEOBNRv4_ROM_NRTidalv2_NSBH",
                "mass1": 8.0,
                "mass2": 1.4,
                "lambda1": 0.0,
                "lambda2": 800.0,
            },
            True,
        ),
        (
            {
                "phase_order": 2,
                "amplitude_order": 3,
                "eccentricity_order": 4,
            },
            True,
        ),
        ({"approximant": "SEOBNRv4"}, False),
        ({"spin1x": 0.1}, False),
        ({"spin_order": 2}, False),
        ({"tidal_order": 12}, False),
        ({"dchi3": 0.1}, False),
    ],
)
def test_seobnrv4_rom_sequence_native_support_boundary(changes, expected):
    assert seobnrv4_rom_sequence_native_supported(changes) is expected


def test_seobnrv4_rom_sequence_unsupported_options_use_lal_fallback(
    monkeypatch,
):
    params = {
        **_BASE_PARAMS,
        "mass1": 30.0,
        "mass2": 20.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "f_ref": 20.0,
        "dchi3": 0.1,
    }
    sample_points = [20.0, 30.0, 100.0, 400.0]
    reference = tuple(
        series.numpy().copy()
        for series in _generate_sequence(
            params, sample_points, native=False
        )
    )

    import pycbc.waveform.seobnrv4_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_params):
        raise AssertionError("unsupported SEOBNRv4_ROM sequence reached Torch")

    lal_generator = waveform_module.lalsimulation.SimInspiralChooseFDWaveformSequence
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        native_module,
        "seobnrv4_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_fd_waveform_sequence(
        approximant="SEOBNRv4_ROM",
        sample_points=sample_points,
        **params,
    )

    assert lal_calls == 1
    assert all(isinstance(series._data.tensor, torch.Tensor) for series in fallback)
    _assert_sequence_parity(reference, fallback, relative_tolerance=1.0e-14)


def test_seobnrv4_rom_nrtidal_sequence_requires_increasing_frequencies():
    params = {
        **_BASE_PARAMS,
        "mass1": 1.4,
        "mass2": 1.2,
        "spin1z": 0.05,
        "spin2z": -0.02,
        "lambda1": 400.0,
        "lambda2": 800.0,
        "f_ref": 20.0,
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        _generate_sequence(
            params,
            [20.0, 100.0, 50.0, 400.0],
            native=True,
            device="cpu",
            approximant="SEOBNRv4_ROM_NRTidalv2",
        )


@pytest.mark.parametrize(
    ("approximant", "params"),
    [
        (
            "SEOBNRv4_ROM",
            {
                **_BASE_PARAMS,
                "mass1": 35.0,
                "mass2": 28.0,
                "spin1z": 0.2,
                "spin2z": -0.1,
                "delta_f": 0.25,
                "f_lower": 20.0,
                "f_ref": 30.0,
                "long_asc_nodes": 0.37,
                "phase_order": 2.5,
                "amplitude_order": "3",
                "eccentricity_order": 4,
            },
        ),
        (
            "SEOBNRv4_ROM_NRTidal",
            {
                **_BASE_PARAMS,
                "mass1": 1.4,
                "mass2": 1.2,
                "spin1z": 0.05,
                "spin2z": -0.02,
                "lambda1": 400.0,
                "lambda2": 800.0,
                "delta_f": 0.5,
                "f_lower": 20.0,
                "f_ref": 30.0,
            },
        ),
        (
            "SEOBNRv4_ROM_NRTidalv2",
            {
                **_BASE_PARAMS,
                "mass1": 1.4,
                "mass2": 1.2,
                "spin1z": 0.05,
                "spin2z": -0.02,
                "lambda1": 400.0,
                "lambda2": 800.0,
                "delta_f": 0.5,
                "f_lower": 20.0,
                "f_ref": 30.0,
            },
        ),
        (
            "SEOBNRv4_ROM_NRTidalv2_NSBH",
            {
                **_BASE_PARAMS,
                "mass1": 8.0,
                "mass2": 1.4,
                "spin1z": 0.8,
                "spin2z": 0.0,
                "lambda1": 0.0,
                "lambda2": 800.0,
                "delta_f": 1.0,
                "f_lower": 20.0,
                "f_ref": 30.0,
            },
        ),
    ],
)
def test_seobnrv4_rom_defaults_to_native_without_lalsimulation(
    approximant, params, monkeypatch
):
    reference = _snapshot(
        _generate(params, native=False, approximant=approximant)
    )

    import pycbc.waveform.seobnrv4_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    native = native_module.seobnrv4_fd_torch
    native_calls = 0

    def recording_native(**native_params):
        nonlocal native_calls
        native_calls += 1
        return native(**native_params)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native SEOBNRv4_ROM called lalsimulation")

    monkeypatch.setattr(native_module, "seobnrv4_fd_torch", recording_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lalsimulation,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant=approximant, **params)

    assert native_calls == 1
    assert all(series._data.tensor.device.type == "cpu" for series in actual)
    assert all(series._data.tensor.dtype == torch.complex128 for series in actual)
    _assert_parity(reference, actual, relative_tolerance=1.0e-8)


@pytest.mark.parametrize(
    ("approximant", "unsupported"),
    [
        ("SEOBNRv4_ROM", {"dchi3": 0.1}),
        (
            "SEOBNRv4_ROM_NRTidalv2",
            {
                "lambda1": 400.0,
                "lambda2": 800.0,
                "dquad_mon1": 0.1,
            },
        ),
    ],
)
def test_seobnrv4_rom_unsupported_options_use_lal_fallback(
    approximant, unsupported, monkeypatch
):
    params = {
        **_BASE_PARAMS,
        "mass1": 30.0,
        "mass2": 20.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "delta_f": 0.5,
        "f_lower": 20.0,
        **unsupported,
    }
    reference = _snapshot(
        _generate(params, native=False, approximant=approximant)
    )

    import pycbc.waveform.seobnrv4_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_params):
        raise AssertionError("unsupported parameters reached the native ROM")

    lal_generator = waveform_module.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(native_module, "seobnrv4_fd_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant=approximant, **params)

    assert lal_calls == 1
    assert all(isinstance(series._data.tensor, torch.Tensor) for series in actual)
    _assert_parity(reference, actual, relative_tolerance=1.0e-14)


@pytest.mark.parametrize(
    "flag",
    ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_SEOBNRV4_NATIVE"),
)
def test_seobnrv4_rom_switch_fallback(flag, monkeypatch):
    params = {
        **_BASE_PARAMS,
        "mass1": 20.0,
        "mass2": 15.0,
        "spin1z": 0.1,
        "spin2z": -0.05,
        "delta_f": 0.5,
        "f_lower": 20.0,
    }
    reference = _snapshot(_generate(params, native=False))
    _clear_native_flags(monkeypatch)
    monkeypatch.setenv(flag, "0")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="SEOBNRv4_ROM", **params)
    _assert_parity(reference, actual, relative_tolerance=1.0e-14)


def test_seobnrv4_public_fd_defaults_to_lal_fallback(monkeypatch):
    params = dict(
        mass1=50.0,
        mass2=40.0,
        spin1z=0.1,
        spin2z=-0.05,
        delta_f=1.0,
        f_lower=30.0,
        distance=400.0,
        inclination=0.4,
        coa_phase=0.2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reference = _snapshot(
            _generate(params, native=False, approximant="SEOBNRv4")
        )

    import pycbc.waveform.seobnrv4phm_torch as native_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_params):
        raise AssertionError("default SEOBNRv4 request reached the native port")

    lal_generator = waveform_module.lalsimulation.SimInspiralChooseTDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(native_module, "seobnrv4_fd_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        recording_lal,
    )
    for flag in _NATIVE_FLAGS:
        os.environ.pop(flag, None)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = get_fd_waveform(approximant="SEOBNRv4", **params)
    assert lal_calls >= 1
    _assert_parity(reference, actual, relative_tolerance=1.0e-14)


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_seobnrv4_rom_stays_on_requested_device(device_name):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = {
        **_BASE_PARAMS,
        "mass1": 30.0,
        "mass2": 25.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "delta_f": 0.5,
        "f_lower": 20.0,
        "f_ref": 37.0,
        "long_asc_nodes": 0.23,
    }
    reference = _snapshot(_generate(params, native=False))
    actual = _generate(params, native=True, device=device_name)

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert all(series._data.tensor.device.type == device_name for series in actual)
    assert all(series._data.tensor.dtype == expected_dtype for series in actual)
    tolerance = 2.0e-2 if device_name == "mps" else 1.0e-8
    _assert_parity(reference, actual, relative_tolerance=tolerance)


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize(
    ("approximant", "params"),
    [
        (
            "SEOBNRv4_ROM_NRTidalv2",
            {
                **_BASE_PARAMS,
                "mass1": 1.4,
                "mass2": 1.2,
                "spin1z": 0.05,
                "spin2z": -0.02,
                "lambda1": 400.0,
                "lambda2": 800.0,
                "delta_f": 2.0,
                "f_lower": 20.0,
                "f_ref": 30.0,
            },
        ),
        (
            "SEOBNRv4_ROM_NRTidalv2_NSBH",
            {
                **_BASE_PARAMS,
                "mass1": 8.0,
                "mass2": 1.4,
                "spin1z": 0.8,
                "spin2z": 0.0,
                "lambda1": 0.0,
                "lambda2": 800.0,
                "delta_f": 2.0,
                "f_lower": 20.0,
                "f_ref": 30.0,
            },
        ),
    ],
)
def test_seobnrv4_rom_nrtidal_stays_on_requested_device(
    device_name, approximant, params
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    reference = _snapshot(
        _generate(params, native=False, approximant=approximant)
    )
    actual = _generate(
        params,
        native=True,
        device=device_name,
        approximant=approximant,
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert all(series._data.tensor.device.type == device_name for series in actual)
    assert all(series._data.tensor.dtype == expected_dtype for series in actual)
    tolerance = 2.0e-2 if device_name == "mps" else 1.0e-8
    _assert_parity(
        reference,
        actual,
        relative_tolerance=tolerance,
        exact_zero_mask=device_name != "mps",
    )


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_seobnrv4_rom_sequence_stays_on_requested_device(device_name):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    approximant = "SEOBNRv4_ROM_NRTidalv2"
    params = {
        **_BASE_PARAMS,
        "mass1": 1.4,
        "mass2": 1.2,
        "spin1z": 0.05,
        "spin2z": -0.02,
        "lambda1": 400.0,
        "lambda2": 800.0,
        "f_ref": 0.0,
    }
    sample_points = [20.0, 30.0, 100.0, 500.0, 1000.0, 2048.0]
    reference = tuple(
        series.numpy().copy()
        for series in _generate_sequence(
            params,
            sample_points,
            native=False,
            approximant=approximant,
        )
    )
    actual = _generate_sequence(
        params,
        sample_points,
        native=True,
        device=device_name,
        approximant=approximant,
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert all(
        series._data.tensor.device.type == device_name for series in actual
    )
    assert all(series._data.tensor.dtype == expected_dtype for series in actual)
    tolerance = 2.0e-2 if device_name == "mps" else 1.0e-8
    _assert_sequence_parity(
        reference,
        actual,
        relative_tolerance=tolerance,
        exact_zero_mask=device_name != "mps",
    )


def test_seobnrv4_rom_sequence_avoids_host_transfer(monkeypatch):
    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData

    def reject_host_transfer(_self):
        raise AssertionError("native SEOBNRv4_ROM sequence transferred to NumPy")

    monkeypatch.setenv("PYCBC_SEOBNRV4_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    sample_points = Array([20.0, 30.0, 100.0, 400.0])
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    with torch.no_grad():
        hp, hc = get_fd_waveform_sequence(
            approximant="SEOBNRv4_ROM_NRTidalv2",
            sample_points=sample_points,
            mass1=1.4,
            mass2=1.2,
            spin1z=0.05,
            spin2z=-0.02,
            lambda1=400.0,
            lambda2=800.0,
            f_ref=0.0,
            distance=100.0,
            inclination=0.4,
        )

    assert isinstance(hp._data.tensor, torch.Tensor)
    assert isinstance(hc._data.tensor, torch.Tensor)
    assert hp._data.tensor.device.type == "cpu"
    assert hc._data.tensor.device.type == "cpu"


def test_seobnrv4_rom_reconstruction_uses_torch_tensors(monkeypatch):
    import pycbc.waveform.seobnrv4_torch as native_module

    evaluated_submodels = []
    spline_points = []
    evaluate_submodel = native_module._evaluate_submodel
    spline_eval = native_module._spline_eval

    def recording_submodel(*args, **kwargs):
        result = evaluate_submodel(*args, **kwargs)
        evaluated_submodels.extend(result)
        return result

    def recording_spline(points, *args, **kwargs):
        spline_points.append(points)
        return spline_eval(points, *args, **kwargs)

    monkeypatch.setattr(native_module, "_evaluate_submodel", recording_submodel)
    monkeypatch.setattr(native_module, "_spline_eval", recording_spline)
    actual = _generate(
        {
            **_BASE_PARAMS,
            "mass1": 30.0,
            "mass2": 25.0,
            "spin1z": 0.2,
            "spin2z": -0.1,
            "delta_f": 0.5,
            "f_lower": 20.0,
        },
        native=True,
        device="cpu",
    )

    assert evaluated_submodels
    assert spline_points
    assert all(isinstance(value, torch.Tensor) for value in evaluated_submodels)
    assert all(isinstance(value, torch.Tensor) for value in spline_points)
    assert all(series._data.tensor.device.type == "cpu" for series in actual)
