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


_NATIVE_FLAG_ENVS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_IMRPHENOMP_NATIVE",
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


def _clear_native_flags(monkeypatch):
    """Remove every native flag so the registry default applies."""

    for name in _NATIVE_FLAG_ENVS:
        monkeypatch.delenv(name, raising=False)


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


def test_imrphenomp_regular_grid_power_of_two_boundary(
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=5.0,
        mass2=2.0,
        spin1x=0.1,
        spin1z=0.2,
        spin2y=-0.05,
        spin2z=-0.1,
        distance=200.0,
        inclination=0.6,
        coa_phase=0.4,
        delta_f=2.0,
        f_lower=16.0,
        f_final=256.1,
        f_ref=30.0,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomP", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMP_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomP", **params)

    for expected, expected_array, result in zip(
        reference,
        reference_arrays,
        actual,
    ):
        assert len(result) == len(expected) == 129
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        np.testing.assert_array_equal(
            result.numpy() == 0.0,
            expected_array == 0.0,
        )
        assert _relative_error(result.numpy(), expected_array) < 1.0e-9


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
def test_imrphenomp_native_support_boundary(
    params,
    expected,
    preserve_scheme,
):
    _activate_scheme(_scheme.TorchScheme("cpu"))
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
    _clear_native_flags(monkeypatch)
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
    _clear_native_flags(monkeypatch)
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


@pytest.mark.parametrize(
    ("interface", "native_name"),
    (
        ("regular", "imrphenomp_fd_torch"),
        ("sequence", "imrphenomp_fd_sequence_torch"),
    ),
)
@pytest.mark.parametrize(
    "opt_out_flag",
    ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_IMRPHENOMP_NATIVE"),
)
def test_imrphenomp_default_native_opt_out_uses_lal(
    interface,
    native_name,
    opt_out_flag,
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
        f_ref=20.0,
    )
    import pycbc.waveform.imrphenomp_torch as imrphenomp_mod

    def unexpected_native(**_params):
        raise AssertionError("opted-out IMRPhenomP reached Torch")

    monkeypatch.setattr(imrphenomp_mod, native_name, unexpected_native)
    _clear_native_flags(monkeypatch)
    monkeypatch.setenv(opt_out_flag, "0")

    _activate_scheme(_scheme.CPUScheme())
    if interface == "regular":
        reference = get_fd_waveform(
            approximant="IMRPhenomP",
            delta_f=1.0,
            f_lower=20.0,
            f_final=128.0,
            **params,
        )
    else:
        reference = get_fd_waveform_sequence(
            approximant="IMRPhenomP",
            sample_points=[20.0, 30.0, 100.0],
            **params,
        )
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    if interface == "regular":
        actual = get_fd_waveform(
            approximant="IMRPhenomP",
            delta_f=1.0,
            f_lower=20.0,
            f_final=128.0,
            **params,
        )
    else:
        actual = get_fd_waveform_sequence(
            approximant="IMRPhenomP",
            sample_points=[20.0, 30.0, 100.0],
            **params,
        )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
@pytest.mark.parametrize(
    ("interface", "changes", "expected"),
    (
        ("regular", {"delta_f": 2.0, "f_lower": 19.0}, False),
        ("regular", {"delta_f": 2.0, "f_lower": 20.0}, True),
        ("sequence", {"sample_points": [18.0, 30.0, 100.0]}, False),
        ("sequence", {"sample_points": [19.0, 30.0, 100.0]}, True),
    ),
)
def test_imrphenomp_mps_support_boundary(
    interface,
    changes,
    expected,
    preserve_scheme,
):
    params = {
        "approximant": "IMRPhenomP",
        "mass1": 10.0,
        "mass2": 1.0,
        **changes,
    }
    _activate_scheme(_scheme.TorchScheme("mps"))
    supported = (
        imrphenomp_native_supported(params)
        if interface == "regular"
        else imrphenomp_sequence_native_supported(params)
    )
    assert supported is expected


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
@pytest.mark.parametrize(
    ("interface", "lal_name", "native_name"),
    (
        ("regular", "SimInspiralChooseFDWaveform", "imrphenomp_fd_torch"),
        (
            "sequence",
            "SimInspiralChooseFDWaveformSequence",
            "imrphenomp_fd_sequence_torch",
        ),
    ),
)
def test_imrphenomp_mps_low_frequency_uses_lal_fallback(
    interface,
    lal_name,
    native_name,
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=10.0,
        mass2=1.0,
        spin1x=0.1,
        spin1z=0.2,
        spin2y=0.05,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.6,
        coa_phase=0.4,
        f_ref=20.0,
    )
    import pycbc.waveform.imrphenomp_torch as imrphenomp_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsafe MPS IMRPhenomP reached Torch")

    lal_generator = getattr(waveform_mod.lalsimulation, lal_name)
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(imrphenomp_mod, native_name, unexpected_native)
    monkeypatch.setattr(waveform_mod.lalsimulation, lal_name, recording_lal)
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("mps"))
    if interface == "regular":
        result = get_fd_waveform(
            approximant="IMRPhenomP",
            delta_f=2.0,
            f_lower=19.0,
            f_final=128.0,
            **params,
        )
    else:
        result = get_fd_waveform_sequence(
            approximant="IMRPhenomP",
            sample_points=[18.0, 30.0, 100.0],
            **params,
        )

    assert lal_calls == 1
    assert all(series._data.tensor.device.type == "mps" for series in result)


@pytest.mark.parametrize("interface", ["regular", "sequence"])
@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomp_public_native_stays_on_requested_device(
    device_name,
    interface,
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
        f_ref=30.0,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMP_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    if interface == "regular":
        reference = get_fd_waveform(
            approximant="IMRPhenomP",
            delta_f=2.0,
            f_lower=20.0,
            **params,
        )
    else:
        reference = get_fd_waveform_sequence(
            approximant="IMRPhenomP",
            sample_points=[20.0, 30.0, 80.0, 120.0],
            **params,
        )
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomp_torch as imrphenomp_mod

    spline_devices = []
    natural_cubic_coeff = imrphenomp_mod._natural_cubic_coeff

    def recording_spline(knots, values):
        spline_devices.append((knots.device.type, values.device.type))
        return natural_cubic_coeff(knots, values)

    monkeypatch.setattr(
        imrphenomp_mod, "_natural_cubic_coeff", recording_spline
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme(device_name))
    if interface == "regular":
        actual = get_fd_waveform(
            approximant="IMRPhenomP",
            delta_f=2.0,
            f_lower=20.0,
            **params,
        )
    else:
        actual = get_fd_waveform_sequence(
            approximant="IMRPhenomP",
            sample_points=[20.0, 30.0, 80.0, 120.0],
            **params,
        )

    assert spline_devices == [(device_name, device_name)]
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
    _clear_native_flags(monkeypatch)
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
