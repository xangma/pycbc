import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.types import Array  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenomxphm_torch import (  # noqa: E402
    _wigner_columns,
    imrphenomxphm_native_supported,
    imrphenomxphm_sequence_native_supported,
)


_MSA_FLAGS = dict(
    phenom_x_prec_version=223,
    phenom_xp_convention=1,
    phenom_xp_final_spin_mod=0,
)
_MSA_FINAL_SPIN_FLAGS = dict(_MSA_FLAGS, phenom_xp_final_spin_mod=3)
_MSA_ALIAS_FLAGS = dict(
    _MSA_FLAGS,
    phenom_x_prec_version=300,
    phenom_xp_final_spin_mod=4,
)
_NATIVE_MODELS = (
    {},
    _MSA_FLAGS,
    _MSA_FINAL_SPIN_FLAGS,
    _MSA_ALIAS_FLAGS,
)

_SEQUENCE_PARAMS = dict(
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
)
_SAMPLE_POINTS = [17.3, 22.0, 50.0, 150.0, 400.0, 850.0, 1000.0, 1500.0]


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


@pytest.mark.parametrize("mode", [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)])
def test_imrphenomxphm_wigner_columns_are_orthonormal(mode):
    beta = torch.tensor([0.1, 0.7, 1.4], dtype=torch.float64)
    positive, negative = _wigner_columns(
        *mode,
        torch.cos(beta / 2.0),
        torch.sin(beta / 2.0),
    )
    ell, _ = mode

    assert len(positive) == 2 * ell + 1
    assert len(negative) == 2 * ell + 1
    positive = torch.stack(positive)
    negative = torch.stack(negative)
    torch.testing.assert_close(
        torch.sum(positive * positive, dim=0),
        torch.ones_like(beta),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    torch.testing.assert_close(
        torch.sum(negative * negative, dim=0),
        torch.ones_like(beta),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    torch.testing.assert_close(
        torch.sum(positive * negative, dim=0),
        torch.zeros_like(beta),
        rtol=0.0,
        atol=2.0e-14,
    )


@pytest.mark.parametrize(
    "model_flags",
    _NATIVE_MODELS,
    ids=("default", "msa-final-spin-0", "msa-final-spin-3", "msa-v300-alias"),
)
def test_imrphenomxphm_sequence_matches_lalsimulation(
    model_flags,
    monkeypatch,
    preserve_scheme,
):
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **model_flags,
        **_SEQUENCE_PARAMS,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.waveform as waveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXPHM sequence called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        reject_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **model_flags,
        **_SEQUENCE_PARAMS,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < 5.0e-5


def test_imrphenomxphm_regular_grid_matches_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
    params = dict(
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
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXPHM", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.waveform as waveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXPHM called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        reject_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomXPHM", **params)

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
        assert _relative_error(result_array, expected_array) < 7.0e-4


@pytest.mark.parametrize(
    ("mode_array", "tolerance"),
    [
        ([(2, 2)], 5.0e-5),
        ([(2, 1)], 5.0e-5),
        ([(3, 3)], 5.0e-5),
        ([(3, 2)], 5.0e-4),
        ([(4, 4)], 5.0e-5),
        ([(3, 3), (4, 4)], 5.0e-5),
        ([(4, 4), (2, 1), (2, 1)], 5.0e-5),
    ],
    ids=(
        "22",
        "21",
        "33",
        "32",
        "44",
        "multi",
        "duplicate-reordered",
    ),
)
def test_imrphenomxphm_sequence_mode_subsets_match_lalsimulation(
    mode_array,
    tolerance,
    monkeypatch,
    preserve_scheme,
):
    params = dict(_SEQUENCE_PARAMS, mode_array=mode_array)
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.waveform as waveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXPHM sequence called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        reject_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < tolerance


def test_imrphenomxphm_empty_mode_array_is_zero(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.waveform as waveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("zero-mode IMRPhenomXPHM called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        reject_lal,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        reject_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "1")

    _activate_scheme(_scheme.TorchScheme("cpu"))
    sequence = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        mode_array=[],
        **_SEQUENCE_PARAMS,
    )
    for polarization in sequence:
        assert polarization._data.tensor.device.type == "cpu"
        assert polarization._data.tensor.dtype == torch.complex128
        np.testing.assert_array_equal(polarization.numpy(), 0.0)

    grid = get_fd_waveform(
        approximant="IMRPhenomXPHM",
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
        mode_array=[],
        **_SEQUENCE_PARAMS,
    )
    assert len(grid[0]) == 1025
    for series in grid:
        assert series._data.tensor.device.type == "cpu"
        assert series._data.tensor.dtype == torch.complex128
        np.testing.assert_array_equal(series.numpy(), 0.0)


def test_imrphenomxphm_regular_grid_mode_subset_matches_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
    params = dict(
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
        mode_array=[(3, 3), (4, 4)],
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXPHM", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.waveform as waveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXPHM called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        reject_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomXPHM", **params)

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
        # LAL's regular-grid multibanding may leave the requested upper
        # endpoint zero for a sparse mode set. The native path deliberately
        # performs full mode evaluation, so include that endpoint in the norm.
        relative_error = np.linalg.norm(result_array - expected_array)
        relative_error /= np.linalg.norm(expected_array)
        assert relative_error < 5.0e-3


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, True),
        (_MSA_FLAGS, True),
        (_MSA_FINAL_SPIN_FLAGS, True),
        (_MSA_ALIAS_FLAGS, True),
        ({"phenom_x_prec_version": 223}, True),
        (dict(_MSA_FLAGS, mode_array=[]), True),
        (dict(_MSA_FLAGS, mode_array=[(2, 2)]), True),
        (dict(_MSA_FLAGS, mode_array=[(4, 4), (2, 1), (2, 1)]), True),
        (dict(_MSA_FLAGS, mode_array=(3, 3)), False),
        ({"phenom_x_prec_version": 102}, False),
        (dict(_MSA_FLAGS, phenom_xp_convention=0), False),
        (dict(_MSA_FLAGS, phenom_xp_final_spin_mod=2), False),
        (dict(_MSA_FLAGS, phenom_xp_final_spin_mod=3.5), False),
        (dict(_MSA_FLAGS, mode_array=[(2, -1)]), False),
        (dict(_MSA_FLAGS, mode_array=[(3, 1)]), False),
        (dict(_MSA_FLAGS, mode_array=[(2.0, 2.0)]), False),
        (dict(_MSA_FLAGS, mode_array=["22"]), False),
        (dict(_MSA_FLAGS, mode_array=[(2, 2, 1)]), False),
        (dict(_MSA_FLAGS, lambda1=100.0), False),
        (dict(_MSA_FLAGS, dchi3=0.1), False),
        (dict(_MSA_FLAGS, eccentricity=0.1), False),
        (dict(_MSA_FLAGS, spin_order=4), False),
        (dict(_MSA_FLAGS, frame_axis=1), False),
        (dict(_MSA_FLAGS, numrel_data="waveform.h5"), False),
        ({"approximant": "IMRPhenomXP"}, False),
    ],
)
def test_imrphenomxphm_native_support_boundary(params, expected):
    full_params = {"approximant": "IMRPhenomXPHM", **params}
    assert imrphenomxphm_native_supported(full_params) is expected
    assert imrphenomxphm_sequence_native_supported(full_params) is expected


def test_imrphenomxphm_sequence_avoids_host_transfer(
    monkeypatch,
    preserve_scheme,
):
    from pycbc.types.array_torch import TorchArrayData

    import pycbc.waveform.waveform as waveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXPHM sequence called lalsimulation")

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomXPHM sequence transferred to NumPy")

    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    sample_points = Array(_SAMPLE_POINTS)
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        reject_lal,
    )
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    with torch.no_grad():
        polarizations = get_fd_waveform_sequence(
            approximant="IMRPhenomXPHM",
            sample_points=sample_points,
            **_SEQUENCE_PARAMS,
        )

    for polarization in polarizations:
        assert isinstance(polarization._data.tensor, torch.Tensor)


def test_imrphenomxphm_unsupported_options_use_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch
    import pycbc.waveform.waveform as waveform

    params = {**_SEQUENCE_PARAMS, "mode_array": [(2, 2), (2, -1)]}
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    lal_generator = waveform.lalsimulation.SimInspiralChooseFDWaveformSequence
    lal_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported XPHM sequence reached Torch")

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        xphm_torch,
        "imrphenomxphm_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **params,
    )

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert isinstance(actual._data.tensor, torch.Tensor)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1.0e-14, atol=0.0)


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomxphm_sequence_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **_SEQUENCE_PARAMS,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **_SEQUENCE_PARAMS,
    )

    expected_dtype = torch.complex64 if device_name == "mps" else torch.complex128
    tolerance = 1.0e-2 if device_name == "mps" else 5.0e-5
    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == device_name
        assert result._data.tensor.dtype == expected_dtype
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < tolerance
