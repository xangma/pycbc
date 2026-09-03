from dataclasses import replace

import numpy as np
import pytest
from scipy import special

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenomxp_torch import (  # noqa: E402
    imrphenomxp_native_supported,
    imrphenomxp_sequence_native_supported,
)
from pycbc.waveform.imrphenomxp_msa_torch import (  # noqa: E402
    _jacobi_sn_squared,
)


_NNLO_MODEL_FLAGS = dict(
    phenom_x_prec_version=102,
    phenom_xp_convention=0,
    phenom_xp_final_spin_mod=0,
)
_MSA_MODEL_FLAGS = dict(
    phenom_x_prec_version=223,
    phenom_xp_convention=1,
    phenom_xp_final_spin_mod=0,
)
_MSA_ALIAS_FLAGS = dict(_MSA_MODEL_FLAGS, phenom_x_prec_version=300)
_DEFAULT_MODEL_FLAGS = {}
_MSA_FINAL_SPIN_FLAGS = dict(_MSA_MODEL_FLAGS, phenom_xp_final_spin_mod=4)
_NATIVE_MODELS = (
    _NNLO_MODEL_FLAGS,
    _MSA_MODEL_FLAGS,
    _DEFAULT_MODEL_FLAGS,
)
_TIDAL_APPROXIMANTS = (
    "IMRPhenomXP_NRTidalv2",
    "IMRPhenomXP_NRTidalv3",
)
_TIDAL_PARAMS = dict(
    # The reversed mass order also exercises body-labelled matter reordering.
    mass1=1.2,
    mass2=1.6,
    spin1x=0.015,
    spin1y=-0.02,
    spin1z=-0.04,
    spin2x=0.01,
    spin2y=0.012,
    spin2z=0.05,
    lambda1=800.0,
    lambda2=300.0,
    dquad_mon1=3.0,
    dquad_mon2=4.0,
    distance=130.0,
    inclination=0.8,
    coa_phase=0.6,
    long_asc_nodes=0.2,
    delta_f=2.0,
    f_lower=19.3,
    f_final=2048.0,
    f_ref=0.0,
)
_TIDAL_SAMPLE_POINTS = [19.3, 30.0, 100.0, 500.0, 1024.0, 2048.0, 5000.0]
_NATIVE_FLAG_ENVS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_IMRPHENOMXP_NATIVE",
)


def _lalsimulation_version(module):
    try:
        return tuple(
            int(part)
            for part in module.__version__.split("+", 1)[0].split(".")[:3]
        )
    except (AttributeError, TypeError, ValueError):
        return ()


def _old_lalsimulation_reference(module):
    version = _lalsimulation_version(module)
    return bool(version) and version <= (5, 3, 1)


def _skip_old_xp_reference(module, model_flags):
    if (
        _old_lalsimulation_reference(module)
        and model_flags.get("phenom_x_prec_version") in (102, 223, 300)
    ):
        pytest.skip(
            "installed lalsimulation predates the IMRPhenomXP reference "
            "used by the native port"
        )


def _lal_fd_approximant_available(approximant):
    import pycbc.waveform.waveform as waveform_module

    return approximant in waveform_module._lalsim_fd_approximants


def _assert_native_waveform(actual, expected_size=None):
    assert len(actual) == 2
    for result in actual:
        tensor = result._data.tensor
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.complex128
        if expected_size is None:
            assert tensor.numel() > 0
        else:
            assert tensor.numel() == expected_size
        assert bool(torch.isfinite(tensor).all())


def _clear_native_flags(monkeypatch):
    """Remove every native flag so the registry default applies."""
    for name in _NATIVE_FLAG_ENVS:
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


def _activate_scheme(scheme):
    _scheme.Scheme._single = None
    _scheme.mgr.state = scheme


def _relative_error(actual, expected):
    nonzero = np.abs(expected) > 0.0
    assert nonzero.any()
    return np.linalg.norm(actual[nonzero] - expected[nonzero]) / np.linalg.norm(
        expected[nonzero]
    )


def _full_relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def _raw_series_bytes(series):
    tensor = series._data.tensor.detach().contiguous().cpu()
    return tensor.numpy().tobytes()


def _tidal_sequence_params():
    return {
        key: value
        for key, value in _TIDAL_PARAMS.items()
        if key not in {"delta_f", "f_lower", "f_final", "long_asc_nodes"}
    }


def test_imrphenomxp_msa_jacobi_matches_scipy():
    arguments, parameters = np.meshgrid(
        [-100.0, -20.0, -2.0, 0.0, 0.3, 4.0, 20.0, 100.0],
        [0.0, 0.1, 0.5, 0.9, 0.999999],
    )
    argument = torch.tensor(arguments.ravel(), dtype=torch.float64)
    parameter = torch.tensor(parameters.ravel(), dtype=torch.float64)
    expected = special.ellipj(argument.numpy(), parameter.numpy())[0] ** 2

    actual = _jacobi_sn_squared(argument, parameter)

    assert actual.device.type == "cpu"
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2.0e-13, atol=2.0e-14)


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=40.0,
            mass2=20.0,
            spin1x=0.2,
            spin1y=0.1,
            spin1z=0.3,
            spin2x=-0.1,
            spin2y=0.05,
            spin2z=-0.2,
            inclination=0.7,
            coa_phase=1.2,
            long_asc_nodes=0.3,
            f_ref=30.0,
        ),
        dict(
            mass1=12.0,
            mass2=35.0,
            spin1x=0.15,
            spin1y=-0.25,
            spin1z=0.4,
            spin2x=0.05,
            spin2y=0.2,
            spin2z=-0.3,
            inclination=1.1,
            long_asc_nodes=-0.4,
            f_ref=0.0,
        ),
        dict(
            mass1=30.0,
            mass2=30.0,
            spin1z=0.2,
            spin2z=-0.1,
            inclination=0.2,
            coa_phase=2.1,
            f_ref=20.0,
        ),
    ],
)
def test_imrphenomxp_reference_angle_reuse_is_raw_byte_exact_and_skips_call(
    params,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxp_msa_torch as msa_mod
    import pycbc.waveform.imrphenomxp_torch as xp_mod

    _activate_scheme(_scheme.TorchScheme("cpu"))
    waveform_params = {
        **params,
        **_MSA_MODEL_FLAGS,
        "approximant": "IMRPhenomXP",
        "distance": 500.0,
        "delta_f": 1.0,
        "f_lower": 20.0,
        "f_final": 512.0,
    }
    original_msa_angles = msa_mod.msa_angles
    call_count = 0

    def recording_msa_angles(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_msa_angles(*args, **kwargs)

    monkeypatch.setattr(msa_mod, "msa_angles", recording_msa_angles)
    monkeypatch.setattr(xp_mod, "msa_angles", recording_msa_angles)

    monkeypatch.setenv("PYCBC_IMRPHENOMXP_REFERENCE_ANGLE_REUSE", "0")
    reference = xp_mod.imrphenomxp_fd_torch(**waveform_params)
    reference_calls = call_count

    call_count = 0
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_REFERENCE_ANGLE_REUSE", "1")
    actual = xp_mod.imrphenomxp_fd_torch(**waveform_params)

    assert reference_calls == 3
    assert call_count == 2
    assert tuple(map(_raw_series_bytes, actual)) == tuple(
        map(_raw_series_bytes, reference)
    )


def test_imrphenomxp_reference_angle_reuse_gate_and_support_boundary(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxp_torch as xp_mod

    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = dict(
        mass1=35.0,
        mass2=22.0,
        spin1x=0.2,
        spin1y=-0.15,
        spin1z=0.3,
        spin2x=0.1,
        spin2y=0.05,
        spin2z=-0.2,
        distance=500.0,
        inclination=0.8,
        coa_phase=1.1,
        f_lower=20.0,
        f_ref=30.0,
        **_MSA_MODEL_FLAGS,
    )
    inputs = xp_mod._validated_inputs(params)

    monkeypatch.delenv("PYCBC_IMRPHENOMXP_REFERENCE_ANGLE_REUSE", raising=False)
    assert xp_mod._reference_angle_reuse_enabled() is False
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_REFERENCE_ANGLE_REUSE", "invalid")
    with pytest.raises(ValueError, match="PYCBC_IMRPHENOMXP_REFERENCE_ANGLE_REUSE"):
        xp_mod._reference_angle_reuse_enabled()

    assert xp_mod._reference_angle_reuse_supported(inputs) is True
    assert (
        xp_mod._reference_angle_reuse_supported(
            replace(
                inputs,
                real_dtype=torch.float32,
                complex_dtype=torch.complex64,
            )
        )
        is False
    )
    assert (
        xp_mod._reference_angle_reuse_supported(
            replace(inputs, device=torch.device("cuda"))
        )
        is False
    )
    assert (
        xp_mod._reference_angle_reuse_supported(
            replace(
                inputs,
                alpha0=torch.tensor(inputs.alpha0, requires_grad=True),
            )
        )
        is False
    )


def test_imrphenomxp_reference_angle_state_stores_only_python_floats(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxp_msa_torch as msa_mod
    import pycbc.waveform.imrphenomxp_torch as xp_mod

    _activate_scheme(_scheme.TorchScheme("cpu"))
    inputs = xp_mod._validated_inputs(
        dict(
            mass1=35.0,
            mass2=22.0,
            spin1x=0.2,
            spin1y=-0.15,
            spin1z=0.3,
            spin2x=0.1,
            spin2y=0.05,
            spin2z=-0.2,
            distance=500.0,
            inclination=0.8,
            coa_phase=1.1,
            f_lower=20.0,
            f_ref=30.0,
            **_MSA_MODEL_FLAGS,
        )
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_REFERENCE_ANGLE_REUSE", "1")

    model = xp_mod._build_model(inputs)
    residuals = msa_mod._reference_angle_residuals(model.msa_state)

    assert residuals is not None
    assert all(type(value) is float for value in residuals)
    assert all(
        not isinstance(value, torch.Tensor)
        for key, value in model.msa_state.items()
        if key.startswith("_reference_")
    )


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=40.0,
            mass2=20.0,
            spin1x=0.2,
            spin1y=0.1,
            spin1z=0.3,
            spin2x=-0.1,
            spin2y=0.05,
            spin2z=-0.2,
            distance=500.0,
            inclination=0.7,
            coa_phase=1.2,
            long_asc_nodes=0.3,
            delta_f=0.5,
            f_lower=20.0,
            f_final=512.0,
            f_ref=30.0,
        ),
        # Exercises component reordering, f_ref=0, and non-bin frequencies.
        dict(
            mass1=12.0,
            mass2=35.0,
            spin1x=0.15,
            spin1y=-0.25,
            spin1z=0.4,
            spin2x=0.05,
            spin2y=0.2,
            spin2z=-0.3,
            distance=320.0,
            inclination=1.1,
            coa_phase=0.0,
            long_asc_nodes=-0.4,
            delta_f=0.25,
            f_lower=17.3,
            f_final=900.3,
            f_ref=0.0,
        ),
        # The aligned-spin limit takes a special source-frame branch.
        dict(
            mass1=30.0,
            mass2=30.0,
            spin1z=0.2,
            spin2z=-0.1,
            distance=800.0,
            inclination=0.2,
            coa_phase=2.1,
            delta_f=0.5,
            f_lower=20.0,
            f_ref=20.0,
        ),
        # The zero-spin MSA limit bypasses the Jacobi evolution.
        dict(
            mass1=30.0,
            mass2=20.0,
            distance=500.0,
            inclination=0.7,
            coa_phase=0.3,
            long_asc_nodes=0.2,
            delta_f=0.5,
            f_lower=20.0,
            f_ref=20.0,
        ),
    ],
)
@pytest.mark.parametrize(
    "model_flags",
    _NATIVE_MODELS,
    ids=("nnlo-v102", "msa-v223", "default-msa-final-spin"),
)
def test_imrphenomxp_torch_matches_lalsimulation(
    model_flags,
    params,
    monkeypatch,
    preserve_scheme,
):
    pytest.importorskip("lal")
    lalsimulation = pytest.importorskip("lalsimulation")
    import pycbc.waveform.waveform as waveform_mod

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXP called lalsimulation")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        reject_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(
        approximant="IMRPhenomXP",
        **model_flags,
        **params,
    )
    _assert_native_waveform(actual)
    _skip_old_xp_reference(lalsimulation, model_flags)

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        lal_generator,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(
        approximant="IMRPhenomXP",
        **model_flags,
        **params,
    )
    reference_arrays = tuple(series.numpy().copy() for series in reference)
    tolerance = (
        2.0e-12
        if model_flags.get("phenom_x_prec_version") == 102
        else 2.0e-11
    )

    for expected, expected_array, result in zip(
        reference,
        reference_arrays,
        actual,
    ):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected_array == 0.0)
        assert _relative_error(result_array, expected_array) < tolerance


@pytest.mark.parametrize(
    "model_flags",
    (*_NATIVE_MODELS, _MSA_ALIAS_FLAGS),
    ids=(
        "nnlo-v102",
        "msa-v223",
        "default-msa-final-spin",
        "msa-v300-alias",
    ),
)
def test_imrphenomxp_sequence_matches_lalsimulation(
    model_flags,
    monkeypatch,
    preserve_scheme,
):
    pytest.importorskip("lal")
    lalsimulation = pytest.importorskip("lalsimulation")
    params = dict(
        mass1=12.0,
        mass2=35.0,
        spin1x=0.15,
        spin1y=-0.25,
        spin1z=0.4,
        spin2x=0.05,
        spin2y=0.2,
        spin2z=-0.3,
        distance=320.0,
        inclination=1.1,
        coa_phase=0.0,
        long_asc_nodes=-0.4,
        f_ref=0.0,
        phase_order=2.5,
        amplitude_order="3",
        spin_order=4.5,
        tidal_order=0,
        eccentricity_order=4,
    )
    sample_points = [17.3, 22.0, 150.0, 400.0, 850.0, 1000.0]
    import pycbc.waveform.waveform as waveform_mod

    lal_sequence = waveform_mod.lalsimulation.SimInspiralChooseFDWaveformSequence

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXP sequence called lalsimulation")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        reject_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomXP",
        sample_points=sample_points,
        **model_flags,
        **params,
    )
    _assert_native_waveform(actual, len(sample_points))
    _skip_old_xp_reference(lalsimulation, model_flags)

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        lal_sequence,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXP",
        sample_points=sample_points,
        **model_flags,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < 1.0e-5


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, True),
        (_NNLO_MODEL_FLAGS, True),
        (_MSA_MODEL_FLAGS, True),
        (_MSA_ALIAS_FLAGS, True),
        (_MSA_FINAL_SPIN_FLAGS, True),
        (dict(_MSA_MODEL_FLAGS, phenom_xp_final_spin_mod=3), True),
        ({"phenom_x_prec_version": 223}, True),
        (dict(_NNLO_MODEL_FLAGS, phenom_x_prec_version=223), False),
        (dict(_MSA_MODEL_FLAGS, phenom_xp_convention=0), False),
        (dict(_NNLO_MODEL_FLAGS, phenom_xp_convention=1), False),
        (dict(_NNLO_MODEL_FLAGS, phenom_xp_final_spin_mod=3), False),
        (dict(_NNLO_MODEL_FLAGS, lambda1=100.0), False),
        (dict(_NNLO_MODEL_FLAGS, dchi3=0.1), False),
        (dict(_NNLO_MODEL_FLAGS, eccentricity=0.1), False),
        (dict(_NNLO_MODEL_FLAGS, phase_order=2.5), True),
        (dict(_NNLO_MODEL_FLAGS, amplitude_order="3"), True),
        (dict(_NNLO_MODEL_FLAGS, spin_order=4.5), True),
        (dict(_NNLO_MODEL_FLAGS, tidal_order=0), True),
        (dict(_NNLO_MODEL_FLAGS, eccentricity_order=4), True),
        (dict(_NNLO_MODEL_FLAGS, eccentricity_order=4.0), False),
        (dict(_NNLO_MODEL_FLAGS, tidal_order=12.0), False),
        (dict(_NNLO_MODEL_FLAGS, phase_order=1 << 31), False),
        (dict(_NNLO_MODEL_FLAGS, mode_array=[(2, 2)]), False),
        (dict(_NNLO_MODEL_FLAGS, frame_axis=1), False),
        (dict(_NNLO_MODEL_FLAGS, numrel_data="waveform.h5"), False),
        (dict(_NNLO_MODEL_FLAGS, approximant="IMRPhenomXAS"), False),
    ],
)
def test_imrphenomxp_native_support_boundary(params, expected):
    assert imrphenomxp_native_supported(params) is expected
    assert imrphenomxp_sequence_native_supported(params) is expected


@pytest.mark.parametrize(
    "model_flags",
    _NATIVE_MODELS,
    ids=("nnlo-v102", "msa-v223", "default-msa-final-spin"),
)
def test_imrphenomxp_public_native_dispatch_avoids_lalsimulation(
    model_flags,
    monkeypatch,
    preserve_scheme,
):
    pytest.importorskip("lal")
    lalsimulation = pytest.importorskip("lalsimulation")
    params = dict(
        mass1=35.0,
        mass2=22.0,
        spin1x=0.2,
        spin1y=-0.15,
        spin1z=0.3,
        spin2x=0.1,
        spin2y=0.05,
        spin2z=-0.2,
        distance=500.0,
        inclination=0.8,
        coa_phase=1.1,
        long_asc_nodes=0.37,
        delta_f=0.5,
        f_lower=20.0,
        f_ref=30.0,
        phase_order=0,
        amplitude_order=3,
        spin_order=0,
        tidal_order=0,
        eccentricity_order=4,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(
        approximant="IMRPhenomXP",
        **model_flags,
        **params,
    )
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.waveform as waveform_mod

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXP called lalsimulation")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(
        approximant="IMRPhenomXP",
        **model_flags,
        **params,
    )

    for result in actual:
        assert result._data.tensor.device.type == "cpu"
    _skip_old_xp_reference(lalsimulation, model_flags)

    for expected, result in zip(reference_arrays, actual):
        assert _relative_error(result.numpy(), expected) < 5.0e-12


def test_imrphenomxp_default_configuration_avoids_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    params = dict(
        mass1=35.0,
        mass2=20.0,
        spin1x=0.1,
        spin1z=0.2,
        spin2y=0.1,
        spin2z=-0.1,
        distance=500.0,
        delta_f=1.0,
        f_lower=20.0,
    )
    import pycbc.waveform.waveform as waveform_mod

    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXP", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("default native IMRPhenomXP called lalsimulation")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    result = get_fd_waveform(approximant="IMRPhenomXP", **params)

    for expected, series in zip(reference_arrays, result):
        assert series._data.tensor.device.type == "cpu"
        assert _relative_error(series.numpy(), expected) < 5.0e-12


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize(
    "model_flags",
    _NATIVE_MODELS,
    ids=("nnlo-v102", "msa-v223", "default-msa-final-spin"),
)
def test_imrphenomxp_native_stays_on_requested_device(
    model_flags,
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    pytest.importorskip("lal")
    lalsimulation = pytest.importorskip("lalsimulation")
    params = dict(
        mass1=35.0,
        mass2=20.0,
        spin1x=0.2,
        spin1y=-0.15,
        spin1z=0.3,
        spin2x=0.1,
        spin2y=0.05,
        spin2z=-0.2,
        distance=500.0,
        inclination=0.8,
        coa_phase=1.1,
        delta_f=1.0,
        f_lower=20.0,
        f_ref=30.0,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(
        approximant="IMRPhenomXP",
        **model_flags,
        **params,
    )
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual = get_fd_waveform(
        approximant="IMRPhenomXP",
        **model_flags,
        **params,
    )

    expected_dtype = torch.complex64 if device_name == "mps" else torch.complex128
    tolerance = 4.0e-3 if device_name == "mps" else 2.0e-12
    for result in actual:
        assert result._data.tensor.device.type == device_name
        assert result._data.tensor.dtype == expected_dtype
    _skip_old_xp_reference(lalsimulation, model_flags)

    for expected, result in zip(reference_arrays, actual):
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < tolerance


@pytest.mark.parametrize("approximant", _TIDAL_APPROXIMANTS)
@pytest.mark.parametrize(
    "model_flags",
    (_NNLO_MODEL_FLAGS, _DEFAULT_MODEL_FLAGS),
    ids=("nnlo-v102", "default-msa-final-spin"),
)
def test_imrphenomxp_nrtidal_matches_lalsimulation(
    approximant,
    model_flags,
    monkeypatch,
    preserve_scheme,
):
    pytest.importorskip("lal")
    lalsimulation = pytest.importorskip("lalsimulation")
    params = {
        **_TIDAL_PARAMS,
        "phase_order": 0,
        "amplitude_order": 3,
        "spin_order": 0,
        "tidal_order": 0,
        "eccentricity_order": 4,
    }
    import pycbc.waveform.waveform as waveform_mod

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError(f"native {approximant} called lalsimulation")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(
        approximant=approximant,
        **model_flags,
        **params,
    )
    _assert_native_waveform(actual)
    if not _lal_fd_approximant_available(approximant):
        pytest.skip(f"installed lalsimulation does not provide {approximant}")
    _skip_old_xp_reference(lalsimulation, model_flags)

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        lal_generator,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(
        approximant=approximant,
        **model_flags,
        **params,
    )
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    for expected, expected_array, result in zip(
        reference,
        reference_arrays,
        actual,
    ):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        assert _full_relative_error(result.numpy(), expected_array) < 1.0e-6


@pytest.mark.parametrize("approximant", _TIDAL_APPROXIMANTS)
@pytest.mark.parametrize(
    "model_flags",
    (_NNLO_MODEL_FLAGS, _DEFAULT_MODEL_FLAGS),
    ids=("nnlo-v102", "default-msa-final-spin"),
)
def test_imrphenomxp_nrtidal_sequence_matches_lalsimulation(
    approximant,
    model_flags,
    monkeypatch,
    preserve_scheme,
):
    pytest.importorskip("lal")
    lalsimulation = pytest.importorskip("lalsimulation")
    params = {
        **_tidal_sequence_params(),
        "phase_order": 0,
        "amplitude_order": 3,
        "spin_order": 0,
        "tidal_order": 0,
        "eccentricity_order": 4,
    }
    import pycbc.waveform.waveform as waveform_mod

    lal_sequence = waveform_mod.lalsimulation.SimInspiralChooseFDWaveformSequence

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError(f"native {approximant} sequence called lalsimulation")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=_TIDAL_SAMPLE_POINTS,
        **model_flags,
        **params,
    )
    _assert_native_waveform(actual, len(_TIDAL_SAMPLE_POINTS))
    if not _lal_fd_approximant_available(approximant):
        pytest.skip(f"installed lalsimulation does not provide {approximant}")
    _skip_old_xp_reference(lalsimulation, model_flags)

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        lal_sequence,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=_TIDAL_SAMPLE_POINTS,
        **model_flags,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        assert _full_relative_error(result.numpy(), expected) < 1.0e-9


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"approximant": "IMRPhenomXP_NRTidalv2"}, True),
        ({"approximant": "IMRPhenomXP_NRTidalv3"}, True),
        (
            {
                "approximant": "IMRPhenomXP_NRTidalv2",
                "lambda1": 400.0,
                "lambda2": 700.0,
            },
            True,
        ),
        (
            {
                "approximant": "IMRPhenomXP_NRTidalv3",
                "lambda1": 400.0,
                "dquad_mon1": 3.0,
            },
            True,
        ),
        ({"approximant": "IMRPhenomXP", "lambda1": 400.0}, False),
        ({"approximant": "IMRPhenomXP_NRTidalv2", "lambda1": -1.0}, False),
        (
            {"approximant": "IMRPhenomXP_NRTidalv3", "lambda1": float("nan")},
            False,
        ),
        (
            {"approximant": "IMRPhenomXP_NRTidalv2", "dquad_mon1": -1.0},
            False,
        ),
        (
            {"approximant": "IMRPhenomXP_NRTidalv3", "lambda_octu1": 10.0},
            False,
        ),
        (
            {"approximant": "IMRPhenomXP_NRTidalv2", "mode_array": [(2, 2)]},
            False,
        ),
    ],
)
def test_imrphenomxp_nrtidal_native_support_boundary(changes, expected):
    assert imrphenomxp_native_supported(changes) is expected
    assert imrphenomxp_sequence_native_supported(changes) is expected


def test_imrphenomxp_nrtidal_unsupported_options_use_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    params = {**_TIDAL_PARAMS, "dchi3": 0.1}
    import pycbc.waveform.imrphenomxp_torch as xp_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported XP NRTidal parameters reached Torch")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(xp_mod, "imrphenomxp_fd_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    result = get_fd_waveform(
        approximant="IMRPhenomXP_NRTidalv2",
        **params,
    )

    assert lal_calls == 1
    assert all(isinstance(series._data.tensor, torch.Tensor) for series in result)


@pytest.mark.parametrize(
    "opt_out_flag",
    ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_IMRPHENOMXP_NATIVE"),
)
def test_imrphenomxp_native_opt_out_uses_lal(
    opt_out_flag,
    monkeypatch,
    preserve_scheme,
):
    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    params = dict(
        mass1=35.0,
        mass2=20.0,
        spin1z=0.2,
        spin2z=-0.1,
        distance=500.0,
        delta_f=1.0,
        f_lower=20.0,
    )
    import pycbc.waveform.imrphenomxp_torch as xp_mod

    def unexpected_native(**_params):
        raise AssertionError("opted-out IMRPhenomXP reached the Torch generator")

    # Opt out through one flag while every other native flag stays absent.
    monkeypatch.setenv(opt_out_flag, "0")
    for name in _NATIVE_FLAG_ENVS:
        if name != opt_out_flag:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(xp_mod, "imrphenomxp_fd_torch", unexpected_native)

    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXP", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    result = get_fd_waveform(approximant="IMRPhenomXP", **params)

    assert all(isinstance(series._data.tensor, torch.Tensor) for series in result)
    for expected, series in zip(reference_arrays, result):
        np.testing.assert_array_equal(series.numpy(), expected)


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("approximant", _TIDAL_APPROXIMANTS)
def test_imrphenomxp_nrtidal_stays_on_requested_device(
    approximant,
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    params = {**_TIDAL_PARAMS, "f_final": 1024.0}
    reference_array = None
    if _lal_fd_approximant_available(approximant):
        monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "0")
        _activate_scheme(_scheme.CPUScheme())
        reference, _ = get_fd_waveform(approximant=approximant, **params)
        reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, _ = get_fd_waveform(approximant=approximant, **params)

    expected_dtype = torch.complex64 if device_name == "mps" else torch.complex128
    assert actual._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    if reference_array is None:
        pytest.skip(f"installed lalsimulation does not provide {approximant}")
    # LAL's XPHM multibanding can leave the final requested bins zero. The
    # native path deliberately evaluates the full grid, as does native XPHM.
    # Compare their common support; MPS performs the long BNS phase in float32.
    tolerance = 1.0e-2 if device_name == "mps" else 1.0e-6
    assert _relative_error(actual.numpy(), reference_array) < tolerance


@pytest.mark.parametrize("approximant", _TIDAL_APPROXIMANTS)
def test_imrphenomxp_nrtidal_native_avoids_host_transfer(
    approximant,
    monkeypatch,
    preserve_scheme,
):
    from pycbc.types.array_torch import TorchArrayData

    def reject_host_transfer(_self):
        raise AssertionError(f"native {approximant} transferred data to NumPy")

    monkeypatch.setenv("PYCBC_IMRPHENOMXP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    with torch.no_grad():
        regular = get_fd_waveform(
            approximant=approximant,
            **{**_TIDAL_PARAMS, "f_final": 1024.0},
        )
        sequence = get_fd_waveform_sequence(
            approximant=approximant,
            sample_points=_TIDAL_SAMPLE_POINTS,
            **_tidal_sequence_params(),
        )

    assert all(isinstance(series._data.tensor, torch.Tensor) for series in regular)
    assert all(isinstance(array._data.tensor, torch.Tensor) for array in sequence)
