import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_td_waveform, get_td_waveform_modes  # noqa: E402
from pycbc.waveform.imrphenomthm_torch import (  # noqa: E402
    _DEFAULT_MODES,
    imrphenomthm_td_torch,
)
from pycbc.waveform.imrphenomtphm_torch import (  # noqa: E402
    imrphenomtphm_modes_native_supported,
    imrphenomtphm_modes_torch,
    imrphenomtphm_native_supported,
    imrphenomtphm_td_torch,
)
from pycbc.waveform.imrphenomtp_waveform_torch import (  # noqa: E402
    _build_imrphenomtp_modes,
)


_BASE_CASE = {
    "mass1": 80.0,
    "mass2": 40.0,
    "spin1x": 0.2,
    "spin1y": -0.1,
    "spin1z": 0.3,
    "spin2x": -0.1,
    "spin2y": 0.2,
    "spin2z": -0.2,
    "distance": 100.0,
    "inclination": 0.7,
    "coa_phase": 0.2,
    "delta_t": 1.0 / 2048.0,
    "f_lower": 30.0,
    "f_ref": 30.0,
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


@pytest.fixture(params=("cpu", "cuda", "mps"))
def torch_device(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")
    return device


def _activate_scheme(state):
    _scheme.Scheme._single = None
    _scheme.mgr.state = state


def _normalized_correlation(expected, actual):
    return abs(np.vdot(expected, actual)) / (
        np.linalg.norm(expected) * np.linalg.norm(actual)
    )


def _lal_arguments(parameters, only22=0):
    return (
        parameters["mass1"] * lal.MSUN_SI,
        parameters["mass2"] * lal.MSUN_SI,
        *(parameters[f"spin1{axis}"] for axis in "xyz"),
        *(parameters[f"spin2{axis}"] for axis in "xyz"),
        parameters["distance"] * 1.0e6 * lal.PC_SI,
        parameters["inclination"],
        parameters["delta_t"],
        parameters["f_lower"],
        parameters["f_ref"],
        parameters["coa_phase"],
        lal.CreateDict(),
        only22,
    )


def _lal_mode(mode_series, mode):
    return np.asarray(
        lalsimulation.SphHarmTimeSeriesGetMode(
            mode_series,
            *mode,
        ).data.data
    )


@pytest.mark.parametrize(
    ("parameters", "expected"),
    (
        ({}, True),
        ({"approximant": "IMRPhenomTPHM"}, True),
        ({"spin1x": 0.2, "spin2y": -0.1}, True),
        ({"mode_array": [(2, 2), (3, -3)]}, True),
        ({"mode_array": [(2, 1), (2, 1)]}, True),
        ({"phenom_x_prec_version": 300}, True),
        ({"phenom_xp_convention": 1}, True),
        ({"phenom_xp_final_spin_mod": 4}, True),
        ({"mode_array": []}, False),
        ({"mode_array": [(3, 2)]}, False),
        ({"mode_array": [(2.0, 2)]}, False),
        ({"phenom_x_prec_version": 223}, False),
        ({"phenom_xp_convention": 0}, False),
        ({"phenom_xp_final_spin_mod": 2}, False),
        ({"lambda1": 100.0}, False),
        ({"dchi3": 0.1}, False),
        ({"eccentricity": 0.1}, False),
        ({"phase_order": 2}, False),
        ({"approximant": "IMRPhenomTP"}, False),
    ),
)
def test_imrphenomtphm_native_support_boundary(parameters, expected):
    assert imrphenomtphm_native_supported(parameters) is expected


@pytest.mark.parametrize(
    ("parameters", "expected"),
    (
        ({}, True),
        ({"ell_max": 5}, True),
        ({"ell_max": np.int64(5)}, True),
        ({"ell_max": -1}, True),
        ({"ell_max": 5.0}, False),
        ({"ell_max": None}, False),
        ({"ell_max": "5"}, False),
        ({"mode_array": [(3, 2)]}, False),
    ),
)
def test_imrphenomtphm_modes_native_support_boundary(parameters, expected):
    assert imrphenomtphm_modes_native_supported(parameters) is expected


def test_imrphenomtphm_mode_frames_match_lalsuite(preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    _, coprecessing, j_frame, l0_frame = _build_imrphenomtp_modes(
        _BASE_CASE,
        _DEFAULT_MODES,
    )
    arguments = _lal_arguments(_BASE_CASE)
    references = (
        lalsimulation.SimIMRPhenomTPHM_CoprecModes(*arguments)[0],
        lalsimulation.SimIMRPhenomTPHM_JModes(*arguments)[0],
        lalsimulation.SimIMRPhenomTPHM_L0Modes(*arguments),
    )

    for mode, actual in coprecessing.items():
        expected = _lal_mode(references[0], mode)
        if np.linalg.norm(expected) == 0.0:
            assert torch.count_nonzero(actual) == 0
        else:
            assert _normalized_correlation(expected, actual.numpy()) > 0.999999

    for reference, actual_modes in zip(references[1:], (j_frame, l0_frame)):
        for mode, actual in actual_modes.items():
            expected = _lal_mode(reference, mode)
            if np.linalg.norm(expected) == 0.0:
                assert torch.count_nonzero(actual) == 0
            else:
                assert _normalized_correlation(expected, actual.numpy()) > 0.999


@pytest.mark.parametrize(
    "mode_array",
    (
        None,
        [(3, 3)],
        [(3, -3)],
        [(2, 2), (2, -1), (4, 4)],
    ),
)
def test_imrphenomtphm_modes_match_lalsuite(mode_array, preserve_scheme):
    parameters = dict(_BASE_CASE, mode_array=mode_array)
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform_modes(
        approximant="IMRPhenomTPHM",
        **parameters,
    )
    reference_arrays = {
        mode: real.numpy().copy() + 1j * imag.numpy().copy()
        for mode, (real, imag) in reference.items()
    }

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = imrphenomtphm_modes_torch(
        approximant="IMRPhenomTPHM",
        **parameters,
    )

    assert set(actual) == set(reference)
    expected_modes = []
    actual_modes = []
    for mode in sorted(reference):
        expected_real, _ = reference[mode]
        actual_real, actual_imag = actual[mode]
        expected = reference_arrays[mode]
        result = actual_real.numpy() + 1j * actual_imag.numpy()
        assert len(actual_real) == len(actual_imag) == len(expected_real)
        assert actual_real.delta_t == actual_imag.delta_t == expected_real.delta_t
        assert abs(
            float(actual_real.start_time - expected_real.start_time)
        ) < actual_real.delta_t
        assert actual_real._data.tensor.device.type == "cpu"
        assert actual_imag._data.tensor.device.type == "cpu"
        expected_norm = np.linalg.norm(expected)
        if expected_norm == 0.0:
            assert np.count_nonzero(result) == 0
        else:
            assert _normalized_correlation(expected, result) > 0.9995
        expected_modes.append(expected)
        actual_modes.append(result)

    expected_all = np.concatenate(expected_modes)
    actual_all = np.concatenate(actual_modes)
    assert np.linalg.norm(actual_all - expected_all) / np.linalg.norm(
        expected_all
    ) < 1.0e-3


def test_imrphenomtphm_modes_ignore_phase_and_inclination(preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    reference = imrphenomtphm_modes_torch(
        **_BASE_CASE,
        mode_array=[(2, 2), (3, -3)],
    )
    actual = imrphenomtphm_modes_torch(
        **dict(_BASE_CASE, coa_phase=-1.7, inclination=2.1),
        mode_array=[(2, 2), (3, -3)],
    )
    assert set(actual) == set(reference)
    for mode in reference:
        for expected, result in zip(reference[mode], actual[mode]):
            torch.testing.assert_close(
                result._data.tensor,
                expected._data.tensor,
                rtol=0.0,
                atol=0.0,
            )


@pytest.mark.parametrize(
    "mode_array",
    (
        None,
        [(3, 3)],
        [(3, -3)],
        [(2, 2), (2, -1), (4, 4)],
        [(2, 1), (2, -1), (5, 5), (5, -5)],
    ),
)
def test_imrphenomtphm_waveform_matches_lalsuite(
    mode_array,
    preserve_scheme,
):
    parameters = dict(
        _BASE_CASE,
        mode_array=mode_array,
        long_asc_nodes=0.37,
    )
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="IMRPhenomTPHM", **parameters)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = imrphenomtphm_td_torch(**parameters)

    for expected, expected_array, result in zip(
        reference,
        reference_arrays,
        actual,
    ):
        result_array = result.numpy()
        assert len(result) == len(expected)
        assert result.delta_t == expected.delta_t
        assert abs(float(result.start_time - expected.start_time)) < result.delta_t
        assert result._data.tensor.device.type == "cpu"
        relative_norm_error = abs(
            np.linalg.norm(result_array) / np.linalg.norm(expected_array) - 1.0
        )
        # A single negative co-precessing mode produces weak inertial modes
        # whose rotations amplify the carrier's sub-sample reference offset.
        assert relative_norm_error < 1.0e-3
        assert _normalized_correlation(expected_array, result_array) > 0.99999


def test_imrphenomtphm_aligned_limit_matches_thm(preserve_scheme):
    parameters = dict(
        _BASE_CASE,
        spin1x=0.0,
        spin1y=0.0,
        spin2x=0.0,
        spin2y=0.0,
        mode_array=[(2, 2), (3, -3), (5, 5)],
    )
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = imrphenomtphm_td_torch(**parameters)
    reference = imrphenomthm_td_torch(**parameters)
    for expected, result in zip(reference, actual):
        torch.testing.assert_close(
            result._data.tensor,
            expected._data.tensor,
            rtol=0.0,
            atol=0.0,
        )


def test_imrphenomtphm_default_and_duplicate_modes(preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    default = imrphenomtphm_td_torch(**_BASE_CASE)
    explicit = imrphenomtphm_td_torch(
        **_BASE_CASE,
        mode_array=list(_DEFAULT_MODES),
    )
    duplicate = imrphenomtphm_td_torch(
        **_BASE_CASE,
        mode_array=[(3, 3), (2, -2), (3, 3), (2, -2)],
    )
    unique = imrphenomtphm_td_torch(
        **_BASE_CASE,
        mode_array=[(3, 3), (2, -2)],
    )
    for expected, actual in zip(default, explicit):
        torch.testing.assert_close(actual._data.tensor, expected._data.tensor)
    for expected, actual in zip(unique, duplicate):
        torch.testing.assert_close(actual._data.tensor, expected._data.tensor)


def test_imrphenomtphm_public_native_dispatch_avoids_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
    parameters = dict(
        _BASE_CASE,
        f_lower=40.0,
        f_ref=40.0,
        mode_array=[(2, 2), (3, -3)],
    )
    import pycbc.waveform.imrphenomtphm_torch as tphm_module
    import pycbc.waveform.waveform as waveform_module

    native_generator = tphm_module.imrphenomtphm_td_torch
    native_calls = 0

    def recording_native(**native_parameters):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomTPHM called lalsimulation")

    monkeypatch.setattr(tphm_module, "imrphenomtphm_td_torch", recording_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        unexpected_lalsimulation,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMTPHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_td_waveform(approximant="IMRPhenomTPHM", **parameters)

    assert native_calls == 1
    for series in actual:
        assert series._data.tensor.device.type == "cpu"
        assert torch.isfinite(series._data.tensor).all()


def test_imrphenomtphm_modes_public_native_dispatch_avoids_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomtphm_torch as tphm_module
    import pycbc.waveform.waveform_modes as waveform_modes_module

    native_generator = tphm_module.imrphenomtphm_modes_torch
    native_calls = 0

    def recording_native(**native_parameters):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomTPHM modes called lalsimulation")

    monkeypatch.setattr(
        tphm_module,
        "imrphenomtphm_modes_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_modes_module.lalsimulation,
        "SimInspiralChooseTDModes",
        unexpected_lalsimulation,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMTPHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_td_waveform_modes(
        approximant="IMRPhenomTPHM",
        **dict(
            _BASE_CASE,
            f_lower=40.0,
            f_ref=40.0,
            mode_array=[(2, 2), (3, -3)],
        ),
    )

    assert native_calls == 1
    assert actual
    for real, imag in actual.values():
        assert real._data.tensor.device.type == "cpu"
        assert imag._data.tensor.device.type == "cpu"
        assert torch.isfinite(real._data.tensor).all()
        assert torch.isfinite(imag._data.tensor).all()


@pytest.mark.parametrize(
    ("component_enabled", "modifications"),
    (("0", {}), ("1", {"phenom_x_prec_version": 223})),
)
def test_imrphenomtphm_disabled_or_unsupported_uses_lal_fallback(
    component_enabled,
    modifications,
    monkeypatch,
    preserve_scheme,
):
    parameters = dict(_BASE_CASE, f_lower=40.0, f_ref=40.0)
    parameters.update(modifications)
    import pycbc.waveform.imrphenomtphm_torch as tphm_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_parameters):
        raise AssertionError("unsupported IMRPhenomTPHM parameters reached Torch")

    lal_generator = waveform_module.lalsimulation.SimInspiralChooseTDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(tphm_module, "imrphenomtphm_td_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMTPHM_NATIVE", component_enabled)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_td_waveform(approximant="IMRPhenomTPHM", **parameters)

    assert lal_calls == 1
    for series in fallback:
        assert isinstance(series._data.tensor, torch.Tensor)
        assert series._data.tensor.device.type == "cpu"


@pytest.mark.parametrize(
    ("component_enabled", "modifications"),
    (("0", {}), ("1", {"phenom_x_prec_version": 223})),
)
def test_imrphenomtphm_modes_disabled_or_unsupported_uses_lal_fallback(
    component_enabled,
    modifications,
    monkeypatch,
    preserve_scheme,
):
    parameters = dict(_BASE_CASE, f_lower=40.0, f_ref=40.0)
    parameters.update(modifications)
    import pycbc.waveform.imrphenomtphm_torch as tphm_module
    import pycbc.waveform.waveform_modes as waveform_modes_module

    def unexpected_native(**_parameters):
        raise AssertionError("unsupported IMRPhenomTPHM modes reached Torch")

    lal_generator = (
        waveform_modes_module.lalsimulation.SimInspiralChooseTDModes
    )
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        tphm_module,
        "imrphenomtphm_modes_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_modes_module.lalsimulation,
        "SimInspiralChooseTDModes",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMTPHM_NATIVE", component_enabled)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_td_waveform_modes(
        approximant="IMRPhenomTPHM",
        **parameters,
    )

    assert lal_calls == 1
    assert fallback
    for real, imag in fallback.values():
        assert isinstance(real._data.tensor, torch.Tensor)
        assert isinstance(imag._data.tensor, torch.Tensor)
        assert real._data.tensor.device.type == "cpu"
        assert imag._data.tensor.device.type == "cpu"


@pytest.mark.parametrize(
    "mode_array",
    ([], [(3, 2)], [(2.0, 2)], [22], ["22"], [(2, 2, 1)]),
)
def test_imrphenomtphm_mode_array_validation(mode_array):
    with pytest.raises(ValueError, match="IMRPhenomTPHM"):
        imrphenomtphm_td_torch(**_BASE_CASE, mode_array=mode_array)


def test_imrphenomtphm_active_torch_device(torch_device, preserve_scheme):
    _activate_scheme(_scheme.TorchScheme(torch_device))
    plus, cross = imrphenomtphm_td_torch(
        **dict(
            _BASE_CASE,
            f_lower=40.0,
            f_ref=40.0,
            mode_array=[(2, 2), (3, -3)],
        )
    )
    modes = imrphenomtphm_modes_torch(
        **dict(
            _BASE_CASE,
            f_lower=40.0,
            f_ref=40.0,
            mode_array=[(2, 2), (3, -3)],
        )
    )
    expected_dtype = torch.float32 if torch_device == "mps" else torch.float64
    mode_series = tuple(
        series for pair in modes.values() for series in pair
    )
    for series in (plus, cross, *mode_series):
        assert series._data.tensor.device.type == torch_device
        assert series._data.tensor.dtype == expected_dtype
        assert torch.isfinite(series._data.tensor).all()
