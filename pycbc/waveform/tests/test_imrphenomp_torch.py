import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.types import Array  # noqa: E402
from pycbc.types.array_torch import TorchArrayData  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenomp_torch import (  # noqa: E402
    imrphenomp_fd_sequence_torch,
    imrphenomp_native_supported,
    imrphenomp_sequence_native_supported,
)


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
        # Component reordering, the f_ref fallback, and non-bin bounds.
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
        # The aligned-spin limit takes a special orientation branch.
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
    ],
)
def test_imrphenomp_torch_matches_lalsimulation(
    params,
    monkeypatch,
    preserve_scheme,
):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomP", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomP", **params)

    for expected_series, expected, result in zip(
        reference,
        reference_arrays,
        actual,
    ):
        assert len(result) == len(expected_series)
        assert result.delta_f == expected_series.delta_f
        assert float(result.epoch) == float(expected_series.epoch)
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < 1.0e-9


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, True),
        ({"spin1x": 0.2, "spin2y": -0.1}, True),
        ({"long_asc_nodes": 0.4}, True),
        ({"f_ref": 100.0}, True),
        ({"frame_axis": 0}, True),
        ({"phase_order": 2}, True),
        ({"lambda1": 100.0}, False),
        ({"dchi3": 0.1}, False),
        ({"eccentricity": 0.1}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"frame_axis": 1}, False),
        ({"modes_choice": 1}, False),
        ({"side_bands": 1}, False),
        ({"numrel_data": "waveform.h5"}, False),
        ({"approximant": "IMRPhenomPv2"}, False),
    ],
)
def test_imrphenomp_native_support_boundary(params, expected):
    assert imrphenomp_native_supported(params) is expected
    assert imrphenomp_sequence_native_supported(params) is expected


def test_imrphenomp_public_dispatch_avoids_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
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
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomP", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomp_torch as imrphenomp_mod
    import pycbc.waveform.waveform as waveform_mod

    native = imrphenomp_mod.imrphenomp_fd_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomP called lalsimulation")

    monkeypatch.setattr(imrphenomp_mod, "imrphenomp_fd_torch", recording_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomP", **params)

    assert calls == 1
    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        assert _relative_error(result.numpy(), expected) < 1.0e-9


def test_imrphenomp_unsupported_options_use_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=35.0,
        mass2=20.0,
        spin1x=0.1,
        spin1z=0.2,
        spin2y=0.1,
        spin2z=-0.1,
        distance=500.0,
        delta_f=0.5,
        f_lower=20.0,
        dchi3=0.01,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomP", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomp_torch as imrphenomp_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported IMRPhenomP parameters reached Torch")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        imrphenomp_mod,
        "imrphenomp_fd_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_fd_waveform(approximant="IMRPhenomP", **params)

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert isinstance(actual._data.tensor, torch.Tensor)
        np.testing.assert_allclose(
            actual.numpy(),
            expected,
            rtol=1.0e-14,
            atol=0.0,
        )


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomp_public_native_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")
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
        delta_f=2.0,
        f_lower=20.0,
        f_ref=30.0,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomP", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual = get_fd_waveform(approximant="IMRPhenomP", **params)

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    tolerance = 3.0e-3 if device_name == "mps" else 1.0e-9
    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == device_name
        assert result._data.tensor.dtype == expected_dtype
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < tolerance


def test_imrphenomp_sequence_matches_lal_without_host_transfer(
    monkeypatch,
    preserve_scheme,
):
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
    )
    sample_points = Array([17.3, 22.0, 150.0, 400.0, 850.0, 1000.0])
    monkeypatch.setenv("PYCBC_IMRPHENOMP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomP",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.waveform as waveform_mod

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomP sequence called lalsimulation")

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomP sequence transferred to NumPy")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    monkeypatch.setenv("PYCBC_IMRPHENOMP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomP",
        sample_points=sample_points,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result._data.tensor.detach().numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < 1.0e-9


@pytest.mark.parametrize(
    ("sample_points", "match"),
    [
        ([], "non-empty vector"),
        ([20.0, 20.0], "strictly increasing"),
        ([30.0, 20.0], "strictly increasing"),
        ([0.0, 20.0], "positive"),
        ([20.0, float("nan")], "finite"),
        ([1000.0, 1200.0], "fCut must exceed"),
    ],
)
def test_imrphenomp_sequence_validates_frequencies(
    sample_points,
    match,
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with pytest.raises(ValueError, match=match):
        imrphenomp_fd_sequence_torch(
            approximant="IMRPhenomP",
            sample_points=sample_points,
            mass1=40.0,
            mass2=20.0,
            distance=500.0,
            f_ref=0.0,
        )


def test_imrphenomp_rejects_v1_domain_violations(monkeypatch, preserve_scheme):
    monkeypatch.setenv("PYCBC_IMRPHENOMP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    common = dict(
        approximant="IMRPhenomP",
        delta_f=1.0,
        f_lower=20.0,
        distance=500.0,
    )
    with pytest.raises(ValueError, match="mass ratio"):
        get_fd_waveform(mass1=21.0, mass2=1.0, **common)
    with pytest.raises(ValueError, match="effective spin"):
        get_fd_waveform(
            mass1=30.0,
            mass2=20.0,
            spin1z=0.95,
            spin2z=0.95,
            **common,
        )
