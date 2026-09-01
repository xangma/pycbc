import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_td_waveform  # noqa: E402
from pycbc.waveform.imrphenomthm_torch import (  # noqa: E402
    _DEFAULT_MODES,
    _assemble_modes,
    imrphenomthm_default_native_supported,
    imrphenomthm_native_supported,
    imrphenomthm_td_torch,
)
from pycbc.waveform.imrphenomt_torch import (  # noqa: E402
    _build_imrphenomt_core,
)


_BASE_CASE = {
    "mass1": 30.0,
    "mass2": 20.0,
    "spin1z": 0.2,
    "spin2z": -0.1,
    "distance": 400.0,
    "inclination": 0.7,
    "coa_phase": 0.3,
    "delta_t": 1.0 / 4096.0,
    "f_lower": 30.0,
    "f_ref": 30.0,
}
_HIGH_Q_CASE = {
    "mass1": 50.0,
    "mass2": 10.0,
    "spin1z": 0.7,
    "spin2z": -0.4,
    "distance": 300.0,
    "inclination": 1.1,
    "coa_phase": -0.2,
    "delta_t": 1.0 / 2048.0,
    "f_lower": 25.0,
    "f_ref": 40.0,
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
    return np.dot(expected, actual) / (
        np.linalg.norm(expected) * np.linalg.norm(actual)
    )


@pytest.mark.parametrize(
    ("parameters", "expected"),
    (
        ({}, True),
        ({"approximant": "IMRPhenomTHM", "f_ref": 100.0}, True),
        ({"long_asc_nodes": 0.4}, True),
        ({"mode_array": [(2, 2), (3, -3)]}, True),
        ({"mode_array": [(2, 1), (2, 1)]}, True),
        ({"lambda1": 0.0}, True),
        ({"mode_array": []}, False),
        ({"mode_array": [(3, 2)]}, False),
        ({"mode_array": [(2.0, 2)]}, False),
        ({"spin1x": 0.1}, False),
        ({"lambda1": 100.0}, False),
        ({"dquad_mon1": 0.1}, False),
        ({"dchi3": 0.1}, False),
        ({"eccentricity": 0.1}, False),
        ({"phase_order": 2}, False),
        ({"spin_order": 2}, False),
        ({"frame_axis": 1}, False),
        ({"numrel_data": "waveform.h5"}, False),
        ({"approximant": "IMRPhenomT"}, False),
    ),
)
def test_imrphenomthm_native_support_boundary(parameters, expected):
    assert imrphenomthm_native_supported(parameters) is expected


def test_imrphenomthm_default_support_is_conservative():
    assert imrphenomthm_default_native_supported(_BASE_CASE)
    assert not imrphenomthm_default_native_supported(
        dict(_BASE_CASE, mode_array=[(2, 2)])
    )

    total_mass_seconds = 50.0 * 4.9254909476412675e-6
    low_frequency = 0.006 / total_mass_seconds
    root_sensitive = dict(
        _BASE_CASE,
        mass1=100.0 / 3.0,
        mass2=50.0 / 3.0,
        spin1z=0.99,
        spin2z=0.99,
        delta_t=1.0 / 2048.0,
        f_lower=low_frequency,
        f_ref=low_frequency,
    )
    assert imrphenomthm_native_supported(root_sensitive)
    assert not imrphenomthm_default_native_supported(root_sensitive)


@pytest.mark.parametrize(
    ("parameters", "mode_array"),
    (
        (_BASE_CASE, None),
        (_HIGH_Q_CASE, None),
        (_BASE_CASE, [(3, 3)]),
        (_BASE_CASE, [(3, -3)]),
        (_BASE_CASE, [(2, 2), (2, -1), (4, 4)]),
        (_HIGH_Q_CASE, [(2, 1), (2, -1), (5, 5), (5, -5)]),
    ),
)
def test_imrphenomthm_waveform_matches_lalsuite(
    parameters, mode_array, preserve_scheme
):
    parameters = dict(parameters, mode_array=mode_array, long_asc_nodes=0.37)
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="IMRPhenomTHM", **parameters)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = imrphenomthm_td_torch(**parameters)

    for expected, expected_array, result in zip(reference, reference_arrays, actual):
        result_array = result.numpy()
        assert len(result) == len(expected)
        assert result.delta_t == expected.delta_t
        assert abs(float(result.start_time - expected.start_time)) < result.delta_t
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.float64
        assert np.isfinite(result_array).all()

        # LAL's loose root tolerance can shift the reference time by a small
        # fraction of a sample. Higher-m phases amplify that harmless offset.
        relative_norm_error = abs(
            np.linalg.norm(result_array) / np.linalg.norm(expected_array) - 1.0
        )
        assert relative_norm_error < 5.0e-5
        assert _normalized_correlation(expected_array, result_array) > 0.9995


def test_imrphenomthm_public_native_dispatch_avoids_lalsimulation(
    monkeypatch, preserve_scheme
):
    parameters = dict(
        _BASE_CASE,
        long_asc_nodes=0.37,
        mode_array=[(2, 2), (2, -1), (4, 4)],
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMTHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="IMRPhenomTHM", **parameters)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomthm_torch as imrphenomthm_module
    import pycbc.waveform.waveform as waveform_module

    native_generator = imrphenomthm_module.imrphenomthm_td_torch
    native_calls = 0

    def recording_native(**native_parameters):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomTHM called lalsimulation")

    monkeypatch.setattr(
        imrphenomthm_module,
        "imrphenomthm_td_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        unexpected_lalsimulation,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMTHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_td_waveform(approximant="IMRPhenomTHM", **parameters)

    assert native_calls == 1
    for expected, expected_array, result in zip(reference, reference_arrays, actual):
        result_array = result.numpy()
        assert len(result) == len(expected)
        assert result.delta_t == expected.delta_t
        assert abs(float(result.start_time - expected.start_time)) < result.delta_t
        assert result._data.tensor.device.type == "cpu"
        relative_norm_error = abs(
            np.linalg.norm(result_array) / np.linalg.norm(expected_array) - 1.0
        )
        assert relative_norm_error < 5.0e-5
        assert _normalized_correlation(expected_array, result_array) > 0.9995


def test_imrphenomthm_default_dispatch_avoids_lalsimulation(
    monkeypatch, preserve_scheme
):
    parameters = dict(_BASE_CASE, long_asc_nodes=0.37)
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="IMRPhenomTHM", **parameters)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomthm_torch as imrphenomthm_module
    import pycbc.waveform.waveform as waveform_module

    native_generator = imrphenomthm_module.imrphenomthm_td_torch
    native_calls = 0

    def recording_native(**native_parameters):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("default IMRPhenomTHM called lalsimulation")

    for name in (
        "PYCBC_TORCH_NATIVE_PORTS",
        "PYCBC_TORCH_NATIVE",
        "PYCBC_IMRPHENOMTHM_NATIVE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        imrphenomthm_module,
        "imrphenomthm_td_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        unexpected_lalsimulation,
    )
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_td_waveform(approximant="IMRPhenomTHM", **parameters)

    assert native_calls == 1
    for expected, expected_array, result in zip(reference, reference_arrays, actual):
        result_array = result.numpy()
        assert len(result) == len(expected)
        assert abs(float(result.start_time - expected.start_time)) < result.delta_t
        assert result._data.tensor.device.type == "cpu"
        relative_norm_error = abs(
            np.linalg.norm(result_array) / np.linalg.norm(expected_array) - 1.0
        )
        assert relative_norm_error < 5.0e-5
        assert _normalized_correlation(expected_array, result_array) > 0.9995


def test_imrphenomthm_default_root_sensitive_request_uses_lal(
    monkeypatch, preserve_scheme
):
    total_mass_seconds = 50.0 * 4.9254909476412675e-6
    low_frequency = 0.006 / total_mass_seconds
    parameters = dict(
        _BASE_CASE,
        mass1=100.0 / 3.0,
        mass2=50.0 / 3.0,
        spin1z=0.99,
        spin2z=0.99,
        delta_t=1.0 / 2048.0,
        f_lower=low_frequency,
        f_ref=low_frequency,
    )

    import pycbc.waveform.imrphenomthm_torch as imrphenomthm_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_parameters):
        raise AssertionError("root-sensitive default request reached Torch")

    lal_generator = waveform_module.lalsimulation.SimInspiralChooseTDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    for name in (
        "PYCBC_TORCH_NATIVE_PORTS",
        "PYCBC_TORCH_NATIVE",
        "PYCBC_IMRPHENOMTHM_NATIVE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        imrphenomthm_module,
        "imrphenomthm_td_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        recording_lal,
    )
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_td_waveform(approximant="IMRPhenomTHM", **parameters)

    assert lal_calls == 1
    assert all(
        isinstance(series._data.tensor, torch.Tensor) for series in fallback
    )


@pytest.mark.parametrize(
    ("component_enabled", "modifications"),
    (("0", {}), ("1", {"phase_order": 2})),
)
def test_imrphenomthm_disabled_or_unsupported_uses_lal_fallback(
    component_enabled,
    modifications,
    monkeypatch,
    preserve_scheme,
):
    parameters = dict(_HIGH_Q_CASE, f_lower=40.0, f_ref=40.0)
    parameters.update(modifications)
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="IMRPhenomTHM", **parameters)

    import pycbc.waveform.imrphenomthm_torch as imrphenomthm_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_parameters):
        raise AssertionError("unsupported IMRPhenomTHM parameters reached Torch")

    lal_generator = waveform_module.lalsimulation.SimInspiralChooseTDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        imrphenomthm_module,
        "imrphenomthm_td_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMTHM_NATIVE", component_enabled)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_td_waveform(approximant="IMRPhenomTHM", **parameters)

    assert lal_calls == 1
    for expected, actual in zip(reference, fallback):
        assert len(actual) == len(expected)
        assert isinstance(actual._data.tensor, torch.Tensor)
        assert actual._data.tensor.device.type == "cpu"


def test_imrphenomthm_default_and_duplicate_mode_semantics(preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    default = imrphenomthm_td_torch(**_BASE_CASE)
    explicit = imrphenomthm_td_torch(
        **_BASE_CASE,
        mode_array=list(_DEFAULT_MODES),
    )
    duplicate = imrphenomthm_td_torch(
        **_BASE_CASE,
        mode_array=[(3, 3), (2, -2), (3, 3), (2, -2)],
    )
    unique = imrphenomthm_td_torch(
        **_BASE_CASE,
        mode_array=[(3, 3), (2, -2)],
    )

    for expected, actual in zip(default, explicit):
        torch.testing.assert_close(actual._data.tensor, expected._data.tensor)
    for expected, actual in zip(unique, duplicate):
        torch.testing.assert_close(actual._data.tensor, expected._data.tensor)


def test_imrphenomthm_equal_binary_odd_modes_are_zero(preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    parameters = dict(
        _BASE_CASE,
        mass1=35.0,
        mass2=35.0,
        spin1z=0.2,
        spin2z=0.2,
        mode_array=[(2, 1), (2, -1), (3, 3), (3, -3), (5, 5), (5, -5)],
    )
    plus, cross = imrphenomthm_td_torch(**parameters)
    assert torch.count_nonzero(plus._data.tensor) == 0
    assert torch.count_nonzero(cross._data.tensor) == 0


@pytest.mark.parametrize("ell", (2, 3, 4, 5))
def test_imrphenomthm_negative_mode_symmetry(ell, preserve_scheme):
    mode = (2, 1) if ell == 2 else (ell, ell)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    core = _build_imrphenomt_core(_BASE_CASE)
    modes = (mode, (mode[0], -mode[1]))
    mode_data = _assemble_modes(core, modes)
    torch.testing.assert_close(
        mode_data[modes[1]],
        (-1) ** ell * mode_data[mode].conj(),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "mode_array",
    (
        [],
        [(3, 2)],
        [(2.0, 2)],
        [22],
        ["22"],
        [(2, 2, 1)],
    ),
)
def test_imrphenomthm_mode_array_validation(mode_array):
    with pytest.raises(ValueError, match="IMRPhenomTHM"):
        imrphenomthm_td_torch(**_BASE_CASE, mode_array=mode_array)


def test_imrphenomthm_active_torch_device(torch_device, preserve_scheme):
    _activate_scheme(_scheme.TorchScheme(torch_device))
    plus, cross = imrphenomthm_td_torch(
        **_HIGH_Q_CASE,
        mode_array=[(2, 2), (3, -3)],
    )
    expected_dtype = torch.float32 if torch_device == "mps" else torch.float64
    for series in (plus, cross):
        assert series._data.tensor.device.type == torch_device
        assert series._data.tensor.dtype == expected_dtype
        assert torch.isfinite(series._data.tensor).all()
