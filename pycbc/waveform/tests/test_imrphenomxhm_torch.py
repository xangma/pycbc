import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenomxhm_torch import (  # noqa: E402
    imrphenomxhm_fd_native_supported,
    imrphenomxhm_modes_native_supported,
    imrphenomxhm_sequence_native_supported,
)
from pycbc.waveform.waveform_modes import get_fd_waveform_modes  # noqa: E402


CASES = [
    dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=300.0,
        f_ref=25.0,
        distance=350.0,
        coa_phase=0.4,
        mode_array=[(2, 2), (2, -2)],
    ),
    dict(
        mass1=18.0,
        mass2=42.0,
        spin1z=-0.4,
        spin2z=0.7,
        delta_f=0.25,
        f_lower=17.3,
        f_final=133.3,
        f_ref=0.0,
        distance=700.0,
        coa_phase=0.6,
        mode_array=[(2, 2)],
    ),
    dict(
        mass1=35.0,
        mass2=28.0,
        spin1z=0.2,
        spin2z=-0.1,
        delta_f=0.5,
        f_lower=20.0,
        f_ref=30.0,
        distance=500.0,
        coa_phase=1.1,
        mode_array=[(2, -2)],
    ),
    dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=300.0,
        f_ref=25.0,
        distance=350.0,
        coa_phase=0.4,
        mode_array=[(2, -2), (2, -1), (2, 1)],
    ),
    dict(
        mass1=17.0,
        mass2=43.0,
        spin1z=-0.45,
        spin2z=0.65,
        delta_f=0.5,
        f_lower=18.0,
        f_final=220.0,
        f_ref=0.0,
        distance=800.0,
        coa_phase=0.2,
        mode_array=[(2, -1)],
    ),
    dict(
        mass1=54.0,
        mass2=9.0,
        spin1z=0.92,
        spin2z=0.7,
        delta_f=0.5,
        f_lower=15.0,
        f_final=250.0,
        f_ref=30.0,
        distance=600.0,
        coa_phase=0.8,
        mode_array=[(2, -1)],
    ),
    dict(
        mass1=80.0,
        mass2=3.0,
        spin1z=0.6,
        spin2z=-0.4,
        delta_f=0.25,
        f_lower=10.0,
        f_final=150.0,
        f_ref=20.0,
        distance=900.0,
        coa_phase=0.3,
        mode_array=[(2, 1)],
    ),
    dict(
        mass1=30.0,
        mass2=30.0,
        spin1z=0.3,
        spin2z=0.3,
        delta_f=1.0,
        f_lower=20.0,
        f_final=250.0,
        f_ref=20.0,
        distance=500.0,
        coa_phase=0.0,
        mode_array=[(2, -1), (2, 1)],
    ),
    dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=500.0,
        f_ref=25.0,
        distance=350.0,
        coa_phase=0.4,
        mode_array=[(2, -2), (2, -1), (3, -3), (3, 3)],
    ),
    dict(
        mass1=17.0,
        mass2=43.0,
        spin1z=-0.45,
        spin2z=0.65,
        delta_f=0.5,
        f_lower=18.0,
        f_final=620.0,
        f_ref=0.0,
        distance=800.0,
        coa_phase=0.2,
        mode_array=[(3, -3)],
    ),
    dict(
        mass1=600.0 / 11.0,
        mass2=60.0 / 11.0,
        spin1z=0.98,
        spin2z=0.8,
        delta_f=1.0,
        f_lower=15.0,
        f_final=800.0,
        f_ref=25.0,
        distance=500.0,
        coa_phase=0.37,
        mode_array=[(3, 3)],
    ),
    dict(
        mass1=30.0,
        mass2=30.0,
        spin1z=0.3,
        spin2z=0.3,
        delta_f=1.0,
        f_lower=20.0,
        f_final=600.0,
        f_ref=20.0,
        distance=500.0,
        coa_phase=0.0,
        mode_array=[(3, -3), (3, 3)],
    ),
    dict(
        mass1=30.0,
        mass2=30.0,
        spin1z=0.6,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=600.0,
        f_ref=20.0,
        distance=500.0,
        coa_phase=0.0,
        mode_array=[(3, -3)],
    ),
    dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=700.0,
        f_ref=25.0,
        distance=350.0,
        coa_phase=0.4,
        mode_array=[(2, -2), (3, -3), (4, -4), (4, 4)],
    ),
    dict(
        mass1=17.0,
        mass2=43.0,
        spin1z=-0.45,
        spin2z=0.65,
        delta_f=0.5,
        f_lower=18.0,
        f_ref=0.0,
        distance=800.0,
        coa_phase=0.2,
        mode_array=[(4, -4)],
    ),
    dict(
        mass1=600.0 / 11.0,
        mass2=60.0 / 11.0,
        spin1z=0.98,
        spin2z=0.8,
        delta_f=1.0,
        f_lower=15.0,
        f_final=900.0,
        f_ref=25.0,
        distance=500.0,
        coa_phase=0.37,
        mode_array=[(4, 4)],
    ),
    dict(
        mass1=30.0,
        mass2=30.0,
        spin1z=0.3,
        spin2z=0.3,
        delta_f=1.0,
        f_lower=20.0,
        f_final=800.0,
        f_ref=20.0,
        distance=500.0,
        coa_phase=0.0,
        mode_array=[(4, -4), (4, 4)],
    ),
    dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=700.0,
        f_ref=25.0,
        distance=350.0,
        coa_phase=0.4,
        mode_array=[(3, -2), (3, 2)],
    ),
    dict(
        mass1=17.0,
        mass2=43.0,
        spin1z=-0.45,
        spin2z=0.65,
        delta_f=0.5,
        f_lower=18.0,
        f_final=620.0,
        f_ref=0.0,
        distance=800.0,
        coa_phase=0.2,
        mode_array=[(3, -2)],
    ),
    dict(
        mass1=600.0 / 11.0,
        mass2=60.0 / 11.0,
        spin1z=0.98,
        spin2z=0.8,
        delta_f=1.0,
        f_lower=15.0,
        f_final=900.0,
        f_ref=25.0,
        distance=500.0,
        coa_phase=0.37,
        mode_array=[(3, 2)],
    ),
    dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=700.0,
        f_ref=25.0,
        distance=350.0,
        coa_phase=0.4,
        mode_array=None,
    ),
]

POLARIZATION_CASES = [
    dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=700.0,
        f_ref=25.0,
        distance=350.0,
        inclination=0.9,
        coa_phase=0.4,
        long_asc_nodes=0.37,
    ),
    dict(
        mass1=17.0,
        mass2=43.0,
        spin1z=-0.45,
        spin2z=0.65,
        delta_f=0.5,
        f_lower=18.0,
        f_final=620.0,
        f_ref=0.0,
        distance=800.0,
        inclination=1.2,
        coa_phase=0.2,
        long_asc_nodes=-0.21,
        mode_array=[(2, -2), (2, 1), (3, -3), (3, 2), (4, 4)],
    ),
    dict(
        mass1=600.0 / 11.0,
        mass2=60.0 / 11.0,
        spin1z=0.98,
        spin2z=0.8,
        delta_f=1.0,
        f_lower=15.0,
        f_final=900.0,
        f_ref=25.0,
        distance=500.0,
        inclination=0.6,
        coa_phase=0.37,
        mode_array=[(3, -2), (3, 2)],
    ),
]


SEQUENCE_CASES = [
    (
        dict(
            mass1=46.0,
            mass2=19.0,
            spin1z=0.35,
            spin2z=-0.2,
            distance=350.0,
            inclination=0.7,
            coa_phase=0.4,
            f_ref=25.0,
        ),
        [20.0, 31.5, 80.0, 180.0, 500.0, 900.0, 1000.0, 2000.0],
        5.0e-4,
    ),
    (
        dict(
            mass1=43.0,
            mass2=17.0,
            spin1z=0.65,
            spin2z=-0.45,
            distance=800.0,
            inclination=1.2,
            coa_phase=0.2,
            f_ref=0.0,
            long_asc_nodes=0.91,
            mode_array=[(2, -2), (2, 1), (3, -3), (3, 2), (4, 4)],
        ),
        [17.3, 500.0, 22.0, 150.0],
        5.0e-3,
    ),
    (
        dict(
            mass1=600.0 / 11.0,
            mass2=60.0 / 11.0,
            spin1z=0.98,
            spin2z=0.8,
            distance=500.0,
            inclination=0.6,
            coa_phase=0.37,
            f_ref=25.0,
            mode_array=[(3, -2), (3, 2)],
        ),
        [15.0, 25.0, 80.0, 250.0, 900.0],
        5.0e-2,
    ),
]


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


@pytest.mark.parametrize("params", CASES)
def test_imrphenomxhm_native_modes_match_lal(params, monkeypatch, preserve_scheme):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_modes(approximant="IMRPhenomXHM", **params)
    reference_arrays = {
        mode: tuple(series.numpy().copy() for series in polarizations)
        for mode, polarizations in reference.items()
    }

    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_modes(approximant="IMRPhenomXHM", **params)

    assert actual.keys() == reference.keys()
    for mode, polarizations in actual.items():
        for expected, expected_array, result in zip(
            reference[mode], reference_arrays[mode], polarizations
        ):
            assert len(result) == len(expected)
            assert result.delta_f == expected.delta_f
            assert float(result.epoch) == float(expected.epoch)
            assert result._data.tensor.device.type == "cpu"
            assert result._data.tensor.dtype == torch.complex128
            np.testing.assert_array_equal(
                result.numpy() == 0.0,
                expected_array == 0.0,
            )
            nonzero = np.abs(expected_array) > 0.0
            if not np.any(nonzero):
                continue
            relative_error = np.linalg.norm(
                result.numpy()[nonzero] - expected_array[nonzero]
            ) / np.linalg.norm(expected_array[nonzero])
            if mode[0] == 3 and abs(mode[1]) == 2:
                mass_ratio = max(params["mass1"], params["mass2"]) / min(
                    params["mass1"], params["mass2"]
                )
                # LAL obtains the mixed-mode phase curvature from a 1e-7
                # finite difference. The native path uses its stable analytic
                # derivative, so LAL's amplified roundoff is visible near the
                # edge of the calibrated parameter space.
                tolerance = 5.0e-2 if mass_ratio >= 8.0 else 5.0e-4
            else:
                # The higher-mode eight-condition intermediate-amplitude
                # systems are ill-conditioned; different LU implementations
                # retain slightly different double-precision roundoff.
                tolerance = 1.0e-8 if mode[0] >= 3 else 1.0e-10
            assert relative_error < tolerance


@pytest.mark.parametrize("params", POLARIZATION_CASES)
def test_imrphenomxhm_native_polarizations_match_lal(
    params, monkeypatch, preserve_scheme
):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXHM", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomXHM", **params)

    mass_ratio = max(params["mass1"], params["mass2"]) / min(
        params["mass1"], params["mass2"]
    )
    # LAL's ordinary polarization path enables higher-mode multibanding,
    # whereas its one-mode interface and the native kernels perform the full
    # evaluation.  Sparse, sign-asymmetric mode selections can expose a few
    # parts in 1e3 of multibanding error through polarization cancellation.
    tolerance = 5.0e-2 if mass_ratio >= 8.0 else 5.0e-3
    for expected, expected_array, result in zip(
        reference, reference_arrays, actual
    ):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        nonzero = np.abs(expected_array) > 0.0
        relative_error = np.linalg.norm(
            result.numpy()[nonzero] - expected_array[nonzero]
        ) / np.linalg.norm(expected_array[nonzero])
        assert relative_error < tolerance


def test_imrphenomxhm_native_support_is_deliberately_narrow():
    params = {"approximant": "IMRPhenomXHM", **CASES[0]}
    assert imrphenomxhm_modes_native_supported(params)
    assert imrphenomxhm_modes_native_supported(
        {**params, "mode_array": [(2, -2), (2, -1), (2, 1)]}
    )
    assert imrphenomxhm_modes_native_supported(
        {**params, "mode_array": [(3, -3), (3, 3)]}
    )
    assert imrphenomxhm_modes_native_supported(
        {**params, "mode_array": [(4, -4), (4, 4)]}
    )
    assert imrphenomxhm_modes_native_supported(
        {**params, "mode_array": [(3, -2), (3, 2)]}
    )
    assert imrphenomxhm_modes_native_supported({**params, "mode_array": None})
    assert not imrphenomxhm_modes_native_supported({**params, "spin1x": 0.1})
    assert not imrphenomxhm_modes_native_supported({**params, "lambda1": 100.0})
    assert imrphenomxhm_fd_native_supported({**params, "mode_array": None})
    assert not imrphenomxhm_fd_native_supported({**params, "mode_array": []})


def test_imrphenomxhm_unsupported_options_use_lal_fallback(
    monkeypatch, preserve_scheme
):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch
    import pycbc.waveform.waveform_modes as waveform_modes

    params = {**CASES[0], "mode_array": [(3, 2)], "lambda1": 100.0}
    lal_generator = waveform_modes.lalsimulation.SimIMRPhenomXHMGenerateFDOneMode
    lal_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported XHM mode reached the Torch generator")

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        xhm_torch,
        "imrphenomxhm_modes_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_modes.lalsimulation,
        "SimIMRPhenomXHMGenerateFDOneMode",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    result = get_fd_waveform_modes(approximant="IMRPhenomXHM", **params)

    assert lal_calls == 1
    assert result.keys() == {(3, 2)}


def test_imrphenomxhm_polarization_unsupported_options_use_lal_fallback(
    monkeypatch, preserve_scheme
):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch
    import pycbc.waveform.waveform as waveform

    params = {**POLARIZATION_CASES[0], "dchi0": 0.01}
    lal_generator = waveform.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported XHM parameters reached the Torch generator")

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(xhm_torch, "imrphenomxhm_fd_torch", unexpected_native)
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    hp, hc = get_fd_waveform(approximant="IMRPhenomXHM", **params)

    assert lal_calls == 1
    assert len(hp) == len(hc)


def test_imrphenomxhm_native_avoids_lal_and_host_transfer(monkeypatch, preserve_scheme):
    import pycbc.waveform.waveform as waveform
    import pycbc.waveform.waveform_modes as waveform_modes
    from pycbc.types.array_torch import TorchArrayData

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXHM mode called LAL")

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomXHM transferred data to NumPy")

    monkeypatch.setattr(
        waveform_modes.lalsimulation,
        "SimIMRPhenomXHMGenerateFDOneMode",
        reject_lal,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        reject_lal,
    )
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = {**CASES[0], "mode_array": None}
    with torch.no_grad():
        modes = get_fd_waveform_modes(approximant="IMRPhenomXHM", **params)
        polarizations = get_fd_waveform(
            approximant="IMRPhenomXHM",
            inclination=0.9,
            long_asc_nodes=0.2,
            **params,
        )

    assert list(modes) == [
        (2, 2),
        (2, 1),
        (3, 3),
        (3, 2),
        (4, 4),
        (2, -2),
        (2, -1),
        (3, -3),
        (3, -2),
        (4, -4),
    ]
    for mode_polarizations in modes.values():
        for series in mode_polarizations:
            assert isinstance(series._data.tensor, torch.Tensor)
    for series in polarizations:
        assert isinstance(series._data.tensor, torch.Tensor)


@pytest.mark.parametrize(
    "mode",
    [
        (2, -2),
        (2, -1),
        (3, -3),
        (3, -2),
        (3, 2),
        (3, 3),
        (4, -4),
        (4, 4),
    ],
)
@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomxhm_modes_stay_on_requested_device(
    device_name, mode, monkeypatch, preserve_scheme
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = {**CASES[0], "mode_array": [mode]}
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_modes(approximant="IMRPhenomXHM", **params)
    reference_array = reference[mode][0].numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual = get_fd_waveform_modes(approximant="IMRPhenomXHM", **params)
    series = actual[mode][0]

    expected_dtype = torch.complex64 if device_name == "mps" else torch.complex128
    assert series._data.tensor.device.type == device_name
    assert series._data.tensor.dtype == expected_dtype
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        series.numpy()[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    if device_name == "mps":
        tolerance = 5.0e-3
    elif mode[0] == 3 and abs(mode[1]) == 2:
        tolerance = 5.0e-4
    else:
        tolerance = 1.0e-8 if mode[0] >= 3 else 1.0e-10
    assert relative_error < tolerance


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomxhm_polarizations_stay_on_requested_device(
    device_name, monkeypatch, preserve_scheme
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = POLARIZATION_CASES[0]
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform(approximant="IMRPhenomXHM", **params)
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, _ = get_fd_waveform(approximant="IMRPhenomXHM", **params)

    expected_dtype = torch.complex64 if device_name == "mps" else torch.complex128
    assert actual._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual.numpy()[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    tolerance = 5.0e-3 if device_name == "mps" else 5.0e-4
    assert relative_error < tolerance


@pytest.mark.parametrize(
    ("params", "sample_points", "tolerance"),
    SEQUENCE_CASES,
)
def test_imrphenomxhm_sequence_matches_lal(
    params,
    sample_points,
    tolerance,
    monkeypatch,
    preserve_scheme,
):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXHM",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomXHM",
        sample_points=sample_points,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        relative_error = np.linalg.norm(result.numpy() - expected) / np.linalg.norm(
            expected
        )
        assert relative_error < tolerance


def test_imrphenomxhm_sequence_support_is_deliberately_narrow():
    params = {"approximant": "IMRPhenomXHM"}
    assert imrphenomxhm_sequence_native_supported(params)
    assert imrphenomxhm_sequence_native_supported(
        {**params, "mode_array": [(2, -1), (3, 2), (4, -4)]}
    )
    assert imrphenomxhm_sequence_native_supported(
        {**params, "mode_array": []}
    )
    assert not imrphenomxhm_sequence_native_supported(
        {**params, "mode_array": [(5, -5)]}
    )
    assert not imrphenomxhm_sequence_native_supported(
        {**params, "spin1x": 0.1}
    )
    assert not imrphenomxhm_sequence_native_supported(
        {**params, "lambda1": 100.0}
    )
    assert not imrphenomxhm_sequence_native_supported(
        {**params, "dchi0": 0.01}
    )


def test_imrphenomxhm_sequence_empty_mode_array_is_zero(
    monkeypatch,
    preserve_scheme,
):
    params, sample_points, _ = SEQUENCE_CASES[0]
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    polarizations = get_fd_waveform_sequence(
        approximant="IMRPhenomXHM",
        sample_points=sample_points,
        mode_array=[],
        **params,
    )

    for polarization in polarizations:
        assert torch.count_nonzero(polarization._data.tensor) == 0


def test_imrphenomxhm_sequence_public_dispatch_avoids_lal_and_host_transfer(
    monkeypatch,
    preserve_scheme,
):
    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch
    import pycbc.waveform.waveform as waveform

    params, sample_values, _ = SEQUENCE_CASES[0]
    native = xhm_torch.imrphenomxhm_fd_sequence_torch
    native_calls = 0

    def recording_native(**native_params):
        nonlocal native_calls
        native_calls += 1
        return native(**native_params)

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXHM sequence called LAL")

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomXHM sequence transferred to NumPy")

    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    sample_points = Array(sample_values)
    monkeypatch.setattr(
        xhm_torch,
        "imrphenomxhm_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        reject_lal,
    )
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    with torch.no_grad():
        polarizations = get_fd_waveform_sequence(
            approximant="IMRPhenomXHM",
            sample_points=sample_points,
            **params,
        )

    assert native_calls == 1
    for polarization in polarizations:
        assert isinstance(polarization._data.tensor, torch.Tensor)


def test_imrphenomxhm_sequence_unsupported_options_use_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch
    import pycbc.waveform.waveform as waveform

    base, sample_points, _ = SEQUENCE_CASES[0]
    params = {**base, "dchi0": 0.01}
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXHM",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    lal_generator = waveform.lalsimulation.SimInspiralChooseFDWaveformSequence
    lal_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported XHM sequence reached Torch")

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        xhm_torch,
        "imrphenomxhm_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_fd_waveform_sequence(
        approximant="IMRPhenomXHM",
        sample_points=sample_points,
        **params,
    )

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
def test_imrphenomxhm_sequence_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params, sample_points, _ = SEQUENCE_CASES[0]
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform_sequence(
        approximant="IMRPhenomXHM",
        sample_points=sample_points,
        **params,
    )
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, _ = get_fd_waveform_sequence(
        approximant="IMRPhenomXHM",
        sample_points=sample_points,
        **params,
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert actual._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    relative_error = np.linalg.norm(actual.numpy() - reference_array) / np.linalg.norm(
        reference_array
    )
    tolerance = 5.0e-3 if device_name == "mps" else 5.0e-4
    assert relative_error < tolerance
