import os
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenomd_torch import (  # noqa: E402
    imrphenomd_native_supported,
    imrphenomd_sequence_native_supported,
)


@pytest.fixture
def preserve_scheme():
    """Restore the process-wide PyCBC scheme singleton after a test."""
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


def _tol(dtype):
    if dtype == np.complex64:
        return dict(rel=5e-5, mag=2e-4, phase_mean=5e-3, phase_std=5e-2)
    return dict(rel=1e-10, mag=1e-11, phase_mean=1e-10, phase_std=1e-10)


def _run_case(params, use_native=True, approximant="IMRPhenomD"):
    env_backup = {
        key: os.environ.get(key)
        for key in (
            "PYCBC_TORCH_NATIVE_PORTS",
            "PYCBC_IMRPHENOMD_NATIVE",
        )
    }
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        # CPU reference (LAL)
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        _scheme.mgr.state.prefix = "cpu"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ["PYCBC_IMRPHENOMD_NATIVE"] = "0"
        h_cpu, _ = get_fd_waveform(approximant=approximant, **params)

        # Torch path
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1" if use_native else "0"
        os.environ["PYCBC_IMRPHENOMD_NATIVE"] = "1" if use_native else "0"
        h_torch, _ = get_fd_waveform(approximant=approximant, **params)
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    return h_cpu.numpy(), h_torch.numpy()


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=35.0,
            mass2=28.0,
            spin1z=0.2,
            spin2z=-0.1,
            delta_f=1.0 / 64,
            f_lower=20.0,
            f_final=0.0,
            f_ref=20.0,
            distance=500.0,
            inclination=0.4,
            coa_phase=1.1,
            long_asc_nodes=0.37,
        ),
        dict(
            mass1=10.0,
            mass2=8.0,
            spin1z=0.6,
            spin2z=0.3,
            delta_f=0.5,
            f_lower=15.0,
            f_final=0.0,
            f_ref=30.0,
            distance=300.0,
            inclination=1.2,
            coa_phase=0.3,
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
            inclination=0.8,
            coa_phase=0.6,
        ),
        dict(
            mass1=67.0,
            mass2=43.5,
            spin1z=0.9,
            spin2z=-0.17,
            delta_f=0.125,
            f_lower=19.0,
            f_final=0.0,
            f_ref=245.0,
            distance=407.0,
            inclination=0.68,
            coa_phase=2.17,
            long_asc_nodes=0.21,
        ),
    ],
)
def test_imrphenomd_torch_parity(params):
    cpu, tor = _run_case(params, use_native=True)
    np.testing.assert_array_equal(tor == 0.0, cpu == 0.0)
    mask = np.abs(cpu) > 1e-26
    assert mask.any(), "waveform contains no non-zero bins"
    rel = np.linalg.norm(tor[mask] - cpu[mask]) / np.linalg.norm(cpu[mask])
    mag_ratio = np.mean(np.abs(tor[mask]) / np.abs(cpu[mask]))
    phase_diff = np.angle(tor[mask] * np.conj(cpu[mask]))
    tol = _tol(tor.dtype)
    assert rel < tol["rel"]
    assert abs(mag_ratio - 1.0) < tol["mag"]
    assert abs(phase_diff.mean()) < tol["phase_mean"]
    assert phase_diff.std() < tol["phase_std"]


@pytest.mark.parametrize(
    ("approximant", "params"),
    [
        (
            "IMRPhenomD_NRTidal",
            dict(
                mass1=1.4,
                mass2=1.2,
                spin1z=0.05,
                spin2z=-0.02,
                lambda1=400.0,
                lambda2=800.0,
                delta_f=0.5,
                f_lower=20.0,
                f_ref=30.0,
                distance=100.0,
                inclination=0.4,
                coa_phase=0.7,
            ),
        ),
        (
            "IMRPhenomD_NRTidalv2",
            dict(
                mass1=1.15,
                mass2=1.55,
                spin1z=0.1,
                spin2z=-0.15,
                lambda1=900.0,
                lambda2=300.0,
                delta_f=0.5,
                f_lower=18.0,
                f_final=4096.1,
                f_ref=0.0,
                distance=120.0,
                inclination=0.8,
                coa_phase=0.2,
                long_asc_nodes=0.31,
            ),
        ),
    ],
)
def test_imrphenomd_nrtidal_torch_parity(approximant, params):
    cpu, tor = _run_case(params, use_native=True, approximant=approximant)
    np.testing.assert_array_equal(tor == 0.0, cpu == 0.0)
    nonzero = np.abs(cpu) > 0.0
    relative_error = np.linalg.norm(
        tor[nonzero] - cpu[nonzero]
    ) / np.linalg.norm(cpu[nonzero])
    assert relative_error < 1.0e-10


def test_imrphenomd_torch_global_switch_fallback():
    params = dict(
        mass1=20.0,
        mass2=18.0,
        spin1z=0.1,
        spin2z=-0.05,
        delta_f=1.0 / 32,
        f_lower=20.0,
        f_final=0.0,
        f_ref=20.0,
        distance=400.0,
        inclination=0.3,
        coa_phase=0.2,
    )
    cpu, tor = _run_case(params, use_native=False)
    np.testing.assert_allclose(tor, cpu, rtol=1e-12, atol=1e-18)


def test_imrphenomd_torch_native_emits_no_runtime_warnings():
    params = dict(
        mass1=30.0,
        mass2=20.0,
        spin1z=0.1,
        spin2z=0.05,
        delta_f=0.5,
        f_lower=20.0,
        f_final=0.0,
        f_ref=20.0,
        distance=100.0,
        inclination=0.3,
        coa_phase=0.2,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run_case(params, use_native=True)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime == []


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, True),
        ({"long_asc_nodes": 0.4}, True),
        ({"phase_order": 2, "amplitude_order": 0}, True),
        ({"eccentricity": 0.1}, True),
        ({"lambda1": 0.0}, True),
        (
            {
                "approximant": "IMRPhenomD_NRTidal",
                "lambda1": 400.0,
                "lambda2": 800.0,
            },
            True,
        ),
        (
            {
                "approximant": "IMRPhenomD_NRTidalv2",
                "lambda1": -1.0,
            },
            False,
        ),
        ({"spin1x": 0.1}, False),
        ({"spin_order": 2}, False),
        ({"tidal_order": 0}, False),
        ({"lambda1": 100.0}, False),
        ({"approximant": "IMRPhenomXAS_NRTidalv2"}, False),
        ({"dchi3": 0.1}, False),
        ({"dalpha1": 0.1}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"frame_axis": 1}, False),
    ],
)
def test_imrphenomd_native_support_boundary(params, expected):
    assert imrphenomd_native_supported(params) is expected


@pytest.mark.parametrize(
    ("approximant", "params"),
    [
        (
            "IMRPhenomD",
            dict(
                mass1=35.0,
                mass2=28.0,
                spin1z=0.2,
                spin2z=-0.1,
                delta_f=0.25,
                f_lower=20.0,
                f_ref=30.0,
                distance=500.0,
                inclination=0.4,
                coa_phase=1.1,
                long_asc_nodes=0.37,
            ),
        ),
        (
            "IMRPhenomD_NRTidalv2",
            dict(
                mass1=1.4,
                mass2=1.2,
                spin1z=0.05,
                spin2z=-0.02,
                lambda1=400.0,
                lambda2=800.0,
                delta_f=0.5,
                f_lower=20.0,
                f_ref=30.0,
                distance=100.0,
                inclination=0.4,
                coa_phase=0.7,
            ),
        ),
    ],
)
def test_imrphenomd_public_native_dispatch_and_metadata(
    approximant, params, monkeypatch, preserve_scheme
):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant=approximant, **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomd_torch as imrphenomd_mod
    import pycbc.waveform.waveform as waveform_mod

    native = imrphenomd_mod.imrphenomd_fd_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    monkeypatch.setattr(
        imrphenomd_mod, "imrphenomd_fd_torch", recording_native
    )

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomD called lalsimulation")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform(approximant=approximant, **params)

    assert calls == 1
    for expected, expected_array, result in zip(
        reference, reference_arrays, actual
    ):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128

        result_array = result.numpy()
        np.testing.assert_array_equal(
            result_array == 0.0,
            expected_array == 0.0,
        )
        nonzero = np.abs(expected_array) > 0.0
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected_array[nonzero]
        ) / np.linalg.norm(expected_array[nonzero])
        assert relative_error < 1.0e-10


@pytest.mark.parametrize(
    ("approximant", "unsupported"),
    [
        ("IMRPhenomD", {"dchi3": 0.1}),
        (
            "IMRPhenomD_NRTidalv2",
            {
                "lambda1": 400.0,
                "lambda2": 800.0,
                "dquad_mon1": 0.1,
            },
        ),
    ],
)
def test_imrphenomd_unsupported_options_use_lal_fallback(
    approximant, unsupported, monkeypatch, preserve_scheme
):
    params = dict(
        mass1=30.0,
        mass2=20.0,
        spin1z=0.1,
        spin2z=-0.05,
        delta_f=0.5,
        f_lower=20.0,
        distance=400.0,
        **unsupported,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant=approximant, **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomd_torch as imrphenomd_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported IMRPhenomD parameters reached Torch")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        imrphenomd_mod, "imrphenomd_fd_torch", unexpected_native
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    fallback = get_fd_waveform(approximant=approximant, **params)

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert isinstance(actual._data.tensor, torch.Tensor)
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=1.0e-14, atol=0.0
        )


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomd_public_native_stays_on_requested_device(
    device_name, monkeypatch, preserve_scheme
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = dict(
        mass1=40.0,
        mass2=15.0,
        spin1z=0.6,
        spin2z=-0.3,
        delta_f=0.5,
        f_lower=18.0,
        f_ref=25.0,
        distance=350.0,
        inclination=0.9,
        coa_phase=0.3,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform(approximant="IMRPhenomD", **params)
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, _ = get_fd_waveform(approximant="IMRPhenomD", **params)

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert actual._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    actual_array = actual.numpy()
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual_array[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    assert relative_error < 5.0e-5


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomd_nrtidal_stays_on_requested_device(
    device_name, monkeypatch, preserve_scheme
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = dict(
        mass1=1.4,
        mass2=1.2,
        spin1z=0.05,
        spin2z=-0.02,
        lambda1=400.0,
        lambda2=800.0,
        delta_f=2.0,
        f_lower=20.0,
        f_ref=30.0,
        distance=100.0,
        inclination=0.4,
    )
    approximant = "IMRPhenomD_NRTidalv2"
    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform(approximant=approximant, **params)
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, _ = get_fd_waveform(approximant=approximant, **params)

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert actual._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    actual_array = actual.numpy()
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual_array[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    tolerance = 5.0e-3 if device_name == "mps" else 5.0e-5
    assert relative_error < tolerance


def test_imrphenomd_frequency_hot_path_uses_torch(
    monkeypatch, preserve_scheme
):
    import pycbc.waveform.imrphenomd_torch as imrphenomd_mod

    amplitude_inputs = []
    phase_inputs = []
    original_amplitude = imrphenomd_mod._IMRPhenDAmplitude
    original_phase = imrphenomd_mod._IMRPhenDPhase

    def recording_amplitude(frequencies, *args, **kwargs):
        amplitude_inputs.append(frequencies)
        return original_amplitude(frequencies, *args, **kwargs)

    def recording_phase(frequencies, *args, **kwargs):
        phase_inputs.append(frequencies)
        return original_phase(frequencies, *args, **kwargs)

    def unexpected_numpy_arange(*_args, **_kwargs):
        raise AssertionError("waveform grid was allocated with NumPy")

    monkeypatch.setattr(
        imrphenomd_mod, "_IMRPhenDAmplitude", recording_amplitude
    )
    monkeypatch.setattr(imrphenomd_mod, "_IMRPhenDPhase", recording_phase)
    monkeypatch.setattr(imrphenomd_mod._np, "arange", unexpected_numpy_arange)
    _activate_scheme(_scheme.TorchScheme())
    hp, hc = imrphenomd_mod.imrphenomd_fd_torch(
        mass1=30.0,
        mass2=20.0,
        spin1z=0.2,
        spin2z=-0.1,
        delta_f=0.25,
        f_lower=20.0,
        f_ref=25.0,
        distance=400.0,
        inclination=0.7,
    )

    assert len(amplitude_inputs) == 1
    assert isinstance(amplitude_inputs[0], torch.Tensor)
    assert amplitude_inputs[0].device.type == "cpu"
    full_phase_inputs = [
        value
        for value in phase_inputs
        if isinstance(value, torch.Tensor) and value.numel() > 1
    ]
    scalar_phase_inputs = [
        value for value in phase_inputs if isinstance(value, np.ndarray)
    ]
    assert len(full_phase_inputs) == 1
    assert [value.size for value in scalar_phase_inputs] == [1]
    assert hp._data.tensor.device.type == "cpu"
    assert hc._data.tensor.device.type == "cpu"


@pytest.mark.parametrize(
    ("approximant", "params", "sample_points", "tolerance"),
    [
        (
            "IMRPhenomD",
            dict(
                mass1=35.0,
                mass2=28.0,
                spin1z=0.2,
                spin2z=-0.1,
                f_ref=30.0,
                distance=500.0,
                inclination=0.4,
                coa_phase=1.1,
                long_asc_nodes=0.37,
            ),
            [20.0, 23.5, 30.0, 45.0, 100.0, 400.0, 700.0, 10000.0],
            1.0e-10,
        ),
        (
            "IMRPhenomD",
            dict(
                mass1=18.0,
                mass2=42.0,
                spin1z=-0.4,
                spin2z=0.7,
                f_ref=0.0,
                distance=700.0,
                inclination=0.8,
                coa_phase=0.6,
                long_asc_nodes=0.21,
            ),
            [17.3, 400.0, 22.0, 150.0],
            1.0e-10,
        ),
        (
            "IMRPhenomD_NRTidal",
            dict(
                mass1=1.4,
                mass2=1.2,
                spin1z=0.05,
                spin2z=-0.02,
                lambda1=400.0,
                lambda2=800.0,
                f_ref=30.0,
                distance=100.0,
                inclination=0.4,
                coa_phase=0.7,
            ),
            [20.0, 30.0, 100.0, 500.0, 1000.0, 2048.0, 10000.0],
            1.0e-10,
        ),
        (
            "IMRPhenomD_NRTidalv2",
            dict(
                mass1=1.15,
                mass2=1.55,
                spin1z=0.1,
                spin2z=-0.15,
                lambda1=900.0,
                lambda2=300.0,
                f_ref=0.0,
                distance=120.0,
                inclination=0.8,
                coa_phase=0.2,
                long_asc_nodes=0.31,
            ),
            [18.0, 1500.0, 30.0, 1024.0],
            1.0e-9,
        ),
    ],
)
def test_imrphenomd_sequence_matches_lal_without_calling_lal(
    approximant,
    params,
    sample_points,
    tolerance,
    monkeypatch,
    preserve_scheme,
):
    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.imrphenomd_torch as imrphenomd_mod
    import pycbc.waveform.waveform as waveform_mod

    native = imrphenomd_mod.imrphenomd_fd_sequence_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError(f"native {approximant} sequence called LAL")

    monkeypatch.setattr(
        imrphenomd_mod,
        "imrphenomd_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )

    assert calls == 1
    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        nonzero = np.abs(expected) > 0.0
        np.testing.assert_allclose(
            result_array[nonzero],
            expected[nonzero],
            rtol=tolerance,
            atol=0.0,
        )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, True),
        ({"approximant": "IMRPhenomD"}, True),
        ({"approximant": "IMRPhenomD_NRTidal"}, True),
        ({"approximant": "IMRPhenomD_NRTidalv2"}, True),
        ({"approximant": "IMRPhenomD_NRTidalv2", "lambda1": -1.0}, False),
        ({"approximant": "IMRPhenomXAS"}, False),
        ({"spin1x": 0.1}, False),
        ({"dchi3": 0.1}, False),
        ({"lambda1": 100.0}, False),
    ],
)
def test_imrphenomd_sequence_native_support_boundary(changes, expected):
    assert imrphenomd_sequence_native_supported(changes) is expected


def test_imrphenomd_sequence_unsupported_options_use_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=30.0,
        mass2=20.0,
        spin1z=0.1,
        spin2z=-0.05,
        f_ref=20.0,
        distance=400.0,
        inclination=0.3,
        dchi3=0.1,
    )
    sample_points = [20.0, 30.0, 100.0, 400.0]
    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomD",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.imrphenomd_torch as imrphenomd_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported IMRPhenomD sequence reached Torch")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveformSequence
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        imrphenomd_mod,
        "imrphenomd_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    fallback = get_fd_waveform_sequence(
        approximant="IMRPhenomD",
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
def test_imrphenomd_sequence_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = dict(
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
    sample_points = [20.0, 30.0, 100.0, 500.0, 1000.0, 2048.0]
    approximant = "IMRPhenomD_NRTidalv2"
    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, _ = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert actual._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    actual_array = actual.numpy()
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual_array[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    tolerance = 5.0e-3 if device_name == "mps" else 1.0e-10
    assert relative_error < tolerance


@pytest.mark.parametrize(
    ("approximant", "params"),
    [
        (
            "IMRPhenomD",
            dict(
                mass1=35.0,
                mass2=28.0,
                spin1z=0.2,
                spin2z=-0.1,
                f_ref=0.0,
                distance=500.0,
                inclination=0.4,
            ),
        ),
        (
            "IMRPhenomD_NRTidalv2",
            dict(
                mass1=1.4,
                mass2=1.2,
                spin1z=0.05,
                spin2z=-0.02,
                lambda1=400.0,
                lambda2=800.0,
                f_ref=0.0,
                distance=100.0,
                inclination=0.4,
            ),
        ),
    ],
)
def test_imrphenomd_sequence_native_avoids_host_transfer(
    approximant,
    params,
    monkeypatch,
    preserve_scheme,
):
    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomD sequence transferred to NumPy")

    monkeypatch.setenv("PYCBC_IMRPHENOMD_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    sample_points = Array([20.0, 30.0, 100.0, 400.0])
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    with torch.no_grad():
        hp, hc = get_fd_waveform_sequence(
            approximant=approximant,
            sample_points=sample_points,
            **params,
        )

    assert isinstance(hp._data.tensor, torch.Tensor)
    assert isinstance(hc._data.tensor, torch.Tensor)
