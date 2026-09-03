import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_td_waveform  # noqa: E402
from pycbc.waveform.imrphenomt_torch import (  # noqa: E402
    imrphenomt_native_supported,
    imrphenomt_td_torch,
)


_CASES = (
    {
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
    },
    {
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
    },
    {
        "mass1": 35.0,
        "mass2": 35.0,
        "spin1z": 0.0,
        "spin2z": 0.0,
        "distance": 500.0,
        "inclination": 0.0,
        "coa_phase": 0.0,
        "delta_t": 1.0 / 4096.0,
        "f_lower": 25.0,
        "f_ref": 0.0,
    },
)

_ROOT_ROUNDOFF_CASE = {
    "mass1": 16.0,
    "mass2": 4.0,
    "spin1z": 0.95,
    "spin2z": 0.9,
    "distance": 400.0,
    "inclination": 0.8,
    "coa_phase": 0.3,
    "delta_t": 1.0 / 2048.0,
    "f_lower": 21.89255446964404,
    "f_ref": 28.46032081053725,
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
    return np.real(np.vdot(expected, actual)) / (
        np.linalg.norm(expected) * np.linalg.norm(actual)
    )


@pytest.mark.parametrize(
    ("parameters", "expected"),
    (
        ({}, True),
        ({"approximant": "IMRPhenomT", "f_ref": 100.0}, True),
        ({"long_asc_nodes": 0.4}, True),
        ({"lambda1": 0.0}, True),
        ({"spin1x": 0.1}, False),
        ({"lambda1": 100.0}, False),
        ({"dquad_mon1": 0.1}, False),
        ({"dchi3": 0.1}, False),
        ({"dalpha1": 0.1}, False),
        ({"eccentricity": 0.1}, False),
        ({"mean_per_ano": 0.1}, False),
        ({"phase_order": 2}, False),
        ({"spin_order": 2}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"frame_axis": 1}, False),
        ({"numrel_data": "waveform.h5"}, False),
        ({"approximant": "IMRPhenomTHM"}, False),
    ),
)
def test_imrphenomt_native_support_boundary(parameters, expected):
    assert imrphenomt_native_supported(parameters) is expected


@pytest.mark.parametrize("parameters", _CASES)
def test_imrphenomt_waveform_matches_lalsuite(parameters, preserve_scheme):
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="IMRPhenomT", **parameters)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = imrphenomt_td_torch(**parameters)

    for expected, expected_array, result in zip(
        reference, reference_arrays, actual
    ):
        result_array = result.numpy()
        assert len(result) == len(expected)
        assert result.delta_t == expected.delta_t
        assert abs(float(result.start_time - expected.start_time)) < result.delta_t
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.float64
        assert np.isfinite(result_array).all()

        # LAL's frequency-to-time inversion stops at a loose 1e-4 relative
        # Brent bracket. Roundoff-level coefficient differences can therefore
        # choose reference times a small fraction of one sample apart. Compare
        # the resulting waveforms through phase-insensitive norm parity and a
        # normalized correlation while keeping strict coefficient tests in the
        # phase and amplitude modules.
        relative_norm_error = abs(
            np.linalg.norm(result_array) / np.linalg.norm(expected_array) - 1.0
        )
        assert relative_norm_error < 5.0e-5
        assert _normalized_correlation(expected_array, result_array) > 0.99994


def test_imrphenomt_ill_conditioned_root_matches_lalsuite(preserve_scheme):
    """Cover the GSL coefficient rounding that selects the Brent branch."""

    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(
        approximant="IMRPhenomT",
        **_ROOT_ROUNDOFF_CASE,
    )
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = imrphenomt_td_torch(**_ROOT_ROUNDOFF_CASE)

    assert len(actual[0]) == len(reference[0])
    assert actual[0].start_time == reference[0].start_time
    expected_strain = reference_arrays[0] - 1j * reference_arrays[1]
    actual_strain = actual[0].numpy() - 1j * actual[1].numpy()
    assert _normalized_correlation(expected_strain, actual_strain) > 0.99999


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device unavailable",
)
def test_imrphenomt_mps_uses_float64_root_metadata(preserve_scheme):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    cpu = imrphenomt_td_torch(**_ROOT_ROUNDOFF_CASE)

    _activate_scheme(_scheme.TorchScheme("mps"))
    mps = imrphenomt_td_torch(**_ROOT_ROUNDOFF_CASE)

    assert len(mps[0]) == len(cpu[0])
    assert mps[0].start_time == cpu[0].start_time
    assert mps[0]._data.tensor.device.type == "mps"


def test_imrphenomt_public_native_dispatch_avoids_lalsimulation(
    monkeypatch, preserve_scheme
):
    parameters = dict(_CASES[0], long_asc_nodes=0.37)
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMT_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="IMRPhenomT", **parameters)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomt_torch as imrphenomt_module
    import pycbc.waveform.waveform as waveform_module

    native_generator = imrphenomt_module.imrphenomt_td_torch
    native_calls = 0

    def recording_native(**native_parameters):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomT called lalsimulation")

    monkeypatch.setattr(
        imrphenomt_module, "imrphenomt_td_torch", recording_native
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        unexpected_lalsimulation,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMT_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_td_waveform(approximant="IMRPhenomT", **parameters)

    assert native_calls == 1
    for expected, expected_array, result in zip(
        reference, reference_arrays, actual
    ):
        result_array = result.numpy()
        assert len(result) == len(expected)
        assert result.delta_t == expected.delta_t
        assert abs(float(result.start_time - expected.start_time)) < result.delta_t
        assert result._data.tensor.device.type == "cpu"
        relative_norm_error = abs(
            np.linalg.norm(result_array) / np.linalg.norm(expected_array) - 1.0
        )
        assert relative_norm_error < 5.0e-5
        assert _normalized_correlation(expected_array, result_array) > 0.99994


@pytest.mark.parametrize(
    ("component_enabled", "modifications"),
    (("0", {}), ("1", {"phase_order": 2})),
)
def test_imrphenomt_disabled_or_unsupported_uses_lal_fallback(
    component_enabled,
    modifications,
    monkeypatch,
    preserve_scheme,
):
    parameters = dict(_CASES[1], f_lower=40.0, f_ref=40.0)
    parameters.update(modifications)
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="IMRPhenomT", **parameters)

    import pycbc.waveform.imrphenomt_torch as imrphenomt_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_parameters):
        raise AssertionError("unsupported IMRPhenomT parameters reached Torch")

    lal_generator = waveform_module.lalsimulation.SimInspiralChooseTDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        imrphenomt_module, "imrphenomt_td_torch", unexpected_native
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMT_NATIVE", component_enabled)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_td_waveform(approximant="IMRPhenomT", **parameters)

    assert lal_calls == 1
    for expected, actual in zip(reference, fallback):
        assert len(actual) == len(expected)
        assert isinstance(actual._data.tensor, torch.Tensor)
        assert actual._data.tensor.device.type == "cpu"


def test_imrphenomt_swaps_bodies_consistently(preserve_scheme):
    parameters = dict(_CASES[1])
    _activate_scheme(_scheme.TorchScheme("cpu"))
    original = imrphenomt_td_torch(**parameters)
    parameters.update(
        mass1=_CASES[1]["mass2"],
        mass2=_CASES[1]["mass1"],
        spin1z=_CASES[1]["spin2z"],
        spin2z=_CASES[1]["spin1z"],
    )
    swapped = imrphenomt_td_torch(**parameters)

    for expected, actual in zip(original, swapped):
        assert actual.start_time == expected.start_time
        torch.testing.assert_close(
            actual._data.tensor,
            expected._data.tensor,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("mass1", 0.0),
        ("mass2", -1.0),
        ("spin1z", 1.01),
        ("spin2z", -1.01),
        ("distance", 0.0),
        ("delta_t", 0.0),
        ("f_lower", 0.0),
        ("f_ref", -1.0),
        ("inclination", float("nan")),
        ("mass1", "not-a-number"),
    ),
)
def test_imrphenomt_rejects_invalid_inputs(
    name, value, preserve_scheme
):
    parameters = dict(_CASES[0])
    parameters[name] = value
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(ValueError):
        imrphenomt_td_torch(**parameters)


def test_imrphenomt_requires_torch_scheme(preserve_scheme):
    _activate_scheme(_scheme.CPUScheme())
    with pytest.raises(TypeError, match="active TorchScheme"):
        imrphenomt_td_torch(**_CASES[0])


def test_imrphenomt_runs_on_active_torch_device(
    torch_device, preserve_scheme
):
    parameters = dict(_CASES[1], f_lower=40.0, f_ref=40.0)
    _activate_scheme(_scheme.TorchScheme(torch_device))
    waveform = imrphenomt_td_torch(**parameters)
    expected_dtype = torch.float32 if torch_device == "mps" else torch.float64

    for series in waveform:
        tensor = series._data.tensor
        assert tensor.device.type == torch_device
        assert tensor.dtype == expected_dtype
        assert torch.isfinite(tensor).all()
