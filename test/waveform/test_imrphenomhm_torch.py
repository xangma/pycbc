import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_modes,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenomhm_torch import (  # noqa: E402
    _active_modes,
    _requested_modes,
    imrphenomhm_modes_native_supported,
    imrphenomhm_native_supported,
    imrphenomhm_sequence_native_supported,
)


_NATIVE_FLAGS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_IMRPHENOMHM_NATIVE",
)


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
        [20.0, 31.5, 80.0, 180.0, 500.0, 900.0, 2000.0],
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
            mode_array=[(2, 2), (2, 1), (3, 3), (4, 3)],
        ),
        [17.3, 22.0, 150.0, 500.0],
    ),
    (
        dict(
            mass1=67.0,
            mass2=43.5,
            spin1z=0.9,
            spin2z=-0.17,
            distance=407.0,
            inclination=1.4,
            coa_phase=2.1,
            f_ref=245.0,
            mode_array=[(3, 2), (4, 4)],
        ),
        [19.0, 28.5, 76.0, 245.0, 900.0],
    ),
]

MODE_PARAMS = dict(
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
)

SIGNED_MODES = [
    (2, 2),
    (2, 1),
    (3, 3),
    (3, 2),
    (4, 4),
    (4, 3),
    (2, -2),
    (2, -1),
    (3, -3),
    (3, -2),
    (4, -4),
    (4, -3),
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


def _clear_native_flags(monkeypatch):
    for name in _NATIVE_FLAGS:
        monkeypatch.delenv(name, raising=False)


def _run_case(params, *, use_native=True):
    env_backup = {key: os.environ.get(key) for key in _NATIVE_FLAGS}
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single

    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ["PYCBC_IMRPHENOMHM_NATIVE"] = "0"
        hp_cpu, hc_cpu = get_fd_waveform(
            approximant="IMRPhenomHM", **params
        )

        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        enabled = "1" if use_native else "0"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = enabled
        os.environ["PYCBC_IMRPHENOMHM_NATIVE"] = enabled
        hp_torch, hc_torch = get_fd_waveform(
            approximant="IMRPhenomHM", **params
        )
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    return (hp_cpu, hc_cpu), (hp_torch, hc_torch)


def _assert_parity(cpu_polarizations, torch_polarizations):
    for cpu_series, torch_series in zip(
        cpu_polarizations, torch_polarizations
    ):
        assert len(torch_series) == len(cpu_series)
        assert float(torch_series.epoch) == pytest.approx(
            float(cpu_series.epoch)
        )
        assert isinstance(torch_series._data.tensor, torch.Tensor)
        assert torch_series._data.tensor.device.type == "cpu"

        cpu = cpu_series.numpy()
        actual = torch_series.numpy()
        scale = np.max(np.abs(cpu))
        if scale == 0.0:
            np.testing.assert_array_equal(actual, cpu)
        else:
            np.testing.assert_allclose(
                actual,
                cpu,
                rtol=5e-11,
                atol=scale * 1e-12,
            )


def _assert_mode_parity(reference, actual, device_name="cpu"):
    assert list(actual) == list(reference)
    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    for mode, polarizations in actual.items():
        for expected, result in zip(reference[mode], polarizations):
            assert len(result) == len(expected)
            assert result.delta_f == expected.delta_f
            assert float(result.epoch) == float(expected.epoch)
            assert result._data.tensor.device.type == device_name
            assert result._data.tensor.dtype == expected_dtype
            expected_array = np.asarray(expected._data)
            actual_array = result.numpy()
            nonzero = np.abs(expected_array) > 0.0
            np.testing.assert_array_equal(
                actual_array == 0.0,
                expected_array == 0.0,
            )
            if np.any(nonzero):
                tolerance = 2.0e-4 if device_name == "mps" else 5.0e-11
                error = np.linalg.norm(
                    actual_array[nonzero] - expected_array[nonzero]
                ) / np.linalg.norm(expected_array[nonzero])
                assert error < tolerance


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=50.0,
            mass2=35.0,
            spin1z=0.2,
            spin2z=0.1,
            delta_f=0.5,
            f_lower=15.0,
            f_final=0.0,
            f_ref=20.0,
            distance=500.0,
            inclination=0.7,
            coa_phase=1.0,
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
            long_asc_nodes=0.37,
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
            inclination=1.4,
            coa_phase=2.1,
            mode_array=[(2, 2), (3, 3), (4, 4)],
        ),
    ],
)
def test_imrphenomhm_torch_parity(params):
    _assert_parity(*_run_case(params))


@pytest.mark.parametrize(
    "mode",
    [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)],
)
def test_imrphenomhm_individual_mode_parity(mode):
    params = dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=300.0,
        f_ref=25.0,
        distance=350.0,
        inclination=1.1,
        coa_phase=0.4,
        mode_array=[mode],
    )
    _assert_parity(*_run_case(params))


def test_imrphenomhm_empty_mode_array_matches_lal():
    params = dict(
        mass1=40.0,
        mass2=30.0,
        delta_f=1.0,
        f_lower=20.0,
        distance=400.0,
        mode_array=[],
    )
    _assert_parity(*_run_case(params))


def test_imrphenomhm_empty_frequency_grid_matches_lal():
    params = dict(
        mass1=40.0,
        mass2=30.0,
        delta_f=1000.0,
        f_lower=20.0,
        f_final=100.0,
        distance=400.0,
    )
    _assert_parity(*_run_case(params))


@pytest.mark.parametrize("mode", SIGNED_MODES)
def test_imrphenomhm_fd_mode_matches_lal(mode, monkeypatch, preserve_scheme):
    params = {**MODE_PARAMS, "mode_array": [mode]}
    monkeypatch.setenv("PYCBC_IMRPHENOMHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_modes(
        approximant="IMRPhenomHM",
        **params,
    )

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_modes(
        approximant="IMRPhenomHM",
        **params,
    )

    _assert_mode_parity(reference, actual)


def test_imrphenomhm_default_fd_modes_match_lal(
    monkeypatch,
    preserve_scheme,
):
    params = {**MODE_PARAMS, "f_final": 0.0}
    monkeypatch.setenv("PYCBC_IMRPHENOMHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_modes(
        approximant="IMRPhenomHM",
        **params,
    )

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_modes(
        approximant="IMRPhenomHM",
        **params,
    )

    assert list(reference) == SIGNED_MODES
    _assert_mode_parity(reference, actual)


def test_imrphenomhm_fd_modes_recompose_lal_polarizations(
    monkeypatch,
    preserve_scheme,
):
    inclination = 1.1
    long_asc_nodes = 0.37
    params = {
        **MODE_PARAMS,
        "inclination": inclination,
        "long_asc_nodes": long_asc_nodes,
    }
    monkeypatch.setenv("PYCBC_IMRPHENOMHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(
        approximant="IMRPhenomHM",
        **params,
    )
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    modes = get_fd_waveform_modes(
        approximant="IMRPhenomHM",
        **params,
    )

    hp = np.zeros(len(next(iter(modes.values()))[0]), dtype=np.complex128)
    hc = np.zeros_like(hp)
    for (ell, emm), (ulm, vlm) in modes.items():
        harmonic = lal.SpinWeightedSphericalHarmonic(
            inclination,
            0.0,
            -2,
            ell,
            emm,
        )
        hp += harmonic * ulm.numpy()
        hc -= harmonic * vlm.numpy()
    cos_nodes = np.cos(2.0 * long_asc_nodes)
    sin_nodes = np.sin(2.0 * long_asc_nodes)
    recomposed = (
        cos_nodes * hp + sin_nodes * hc,
        cos_nodes * hc - sin_nodes * hp,
    )

    for expected, actual in zip(reference_arrays, recomposed):
        scale = np.max(np.abs(expected))
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=5.0e-11,
            atol=scale * 1.0e-12,
        )


def test_imrphenomhm_fd_modes_avoid_lal_and_host_transfer(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.waveform_modes as waveform_modes
    from pycbc.types.array_torch import TorchArrayData

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomHM modes called LAL")

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomHM modes transferred to NumPy")

    monkeypatch.setattr(
        waveform_modes.lalsimulation,
        "SimIMRPhenomHMGethlmModes",
        reject_lal,
    )
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    with torch.no_grad():
        modes = get_fd_waveform_modes(
            approximant="IMRPhenomHM",
            **MODE_PARAMS,
        )

    assert list(modes) == SIGNED_MODES
    for polarizations in modes.values():
        for series in polarizations:
            assert isinstance(series._data.tensor, torch.Tensor)


def test_imrphenomhm_unsupported_fd_modes_use_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomhm_torch as hm_torch
    import pycbc.waveform.waveform_modes as waveform_modes

    lal_generator = waveform_modes.lalsimulation.SimIMRPhenomHMGethlmModes
    lal_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported IMRPhenomHM modes reached Torch")

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        hm_torch,
        "imrphenomhm_modes_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_modes.lalsimulation,
        "SimIMRPhenomHMGethlmModes",
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    modes = get_fd_waveform_modes(
        approximant="IMRPhenomHM",
        mode_array=[(2, 2), (2, -2)],
        dchi0=0.01,
        **MODE_PARAMS,
    )

    assert lal_calls == 1
    assert list(modes) == [(2, 2), (2, -2)]
    for polarizations in modes.values():
        for series in polarizations:
            assert isinstance(series._data.tensor, torch.Tensor)


def test_imrphenomhm_fd_modes_component_opt_out_uses_lal(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomhm_torch as hm_torch
    import pycbc.waveform.waveform_modes as waveform_modes

    lal_generator = waveform_modes.lalsimulation.SimIMRPhenomHMGethlmModes
    lal_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("disabled IMRPhenomHM modes reached Torch")

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        hm_torch,
        "imrphenomhm_modes_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_modes.lalsimulation,
        "SimIMRPhenomHMGethlmModes",
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "1")
    monkeypatch.setenv("PYCBC_IMRPHENOMHM_NATIVE", "0")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    modes = get_fd_waveform_modes(
        approximant="IMRPhenomHM",
        mode_array=[(2, 2), (3, -2)],
        **MODE_PARAMS,
    )

    assert lal_calls == 1
    assert list(modes) == [(2, 2), (3, -2)]
    for polarizations in modes.values():
        for series in polarizations:
            assert isinstance(series._data.tensor, torch.Tensor)


@pytest.mark.parametrize(
    "params, expected",
    [
        ({}, True),
        ({"mode_array": [(2, 2), (3, 3)]}, True),
        ({"mode_array": []}, True),
        ({"mode_array": [(2, -2)]}, False),
        ({"mode_array": [(5, 5)]}, False),
        ({"mode_array": [(2.5, 2)]}, False),
        ({"spin1x": 0.1}, False),
        ({"lambda1": 10.0}, False),
        ({"dquad_mon1": 0.1}, False),
        ({"dchi4": 0.1}, False),
        ({"nl_tides_a1": 0.1}, False),
        ({"phenom_x_prec_version": 300}, False),
        ({"phase_order": 7}, False),
        ({"numrel_data": "waveform.h5"}, False),
    ],
)
def test_imrphenomhm_native_support_boundary(params, expected):
    full_params = {"approximant": "IMRPhenomHM", **params}
    assert imrphenomhm_native_supported(full_params) is expected


@pytest.mark.parametrize(
    "mode_array, expected",
    [
        (None, True),
        (SIGNED_MODES, True),
        ([], True),
        ([(4, -3), (2, 2), (4, -3)], True),
        ([(2, 0)], False),
        ([(5, 5)], False),
        ([(2.0, 2)], False),
        ([None], False),
    ],
)
def test_imrphenomhm_fd_modes_support_boundary(mode_array, expected):
    params = {
        "approximant": "IMRPhenomHM",
        "mode_array": mode_array,
        "inclination": np.nan,
        "long_asc_nodes": np.inf,
    }
    assert imrphenomhm_modes_native_supported(params) is expected


def test_imrphenomhm_requested_modes_preserve_order_and_remove_duplicates():
    requested = [(4, -3), (2, 2), (4, -3), (3, -2)]
    assert _requested_modes(requested) == (
        (4, -3),
        (2, 2),
        (3, -2),
    )


def test_imrphenomhm_active_modes_preserve_model_order():
    requested = [(4, 4), (2, 1), (4, 4)]
    assert _active_modes(requested) == ((2, 1), (4, 4))


def test_imrphenomhm_global_switch_disabled_uses_lalsim():
    params = dict(
        mass1=25.0,
        mass2=20.0,
        spin1z=0.1,
        spin2z=-0.05,
        delta_f=0.5,
        f_lower=20.0,
        distance=300.0,
        inclination=0.4,
        coa_phase=0.1,
    )
    _assert_parity(*_run_case(params, use_native=False))


def test_imrphenomhm_default_regular_dispatch_avoids_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.waveform as waveform

    params = dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=300.0,
        f_ref=25.0,
        distance=350.0,
        inclination=0.7,
        coa_phase=0.4,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomHM", **params)

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomHM called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        reject_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomHM", **params)

    _assert_parity(reference, actual)


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomhm_native_stays_on_requested_device(
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
    monkeypatch.setenv("PYCBC_IMRPHENOMHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform(
        approximant="IMRPhenomHM", **params
    )
    reference_array = reference.numpy().copy()

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, cross = get_fd_waveform(
        approximant="IMRPhenomHM", **params
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert actual._data.tensor.device.type == device_name
    assert cross._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual.numpy()[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    tolerance = 1.0e-4 if device_name == "mps" else 1.0e-10
    assert relative_error < tolerance


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomhm_fd_modes_stay_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = {
        **MODE_PARAMS,
        "mode_array": [(4, 3), (4, -3)],
    }
    monkeypatch.setenv("PYCBC_IMRPHENOMHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_modes(
        approximant="IMRPhenomHM",
        **params,
    )

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual = get_fd_waveform_modes(
        approximant="IMRPhenomHM",
        **params,
    )

    _assert_mode_parity(reference, actual, device_name)


@pytest.mark.parametrize(("params", "sample_points"), SEQUENCE_CASES)
def test_imrphenomhm_sequence_matches_lal(
    params,
    sample_points,
    monkeypatch,
    preserve_scheme,
):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomHM",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomHM",
        sample_points=sample_points,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        relative_error = np.linalg.norm(
            result.numpy() - expected
        ) / np.linalg.norm(expected)
        assert relative_error < 1.0e-10


def test_imrphenomhm_sequence_support_is_deliberately_narrow():
    params = {"approximant": "IMRPhenomHM"}
    assert imrphenomhm_sequence_native_supported(params)
    assert imrphenomhm_sequence_native_supported(
        {**params, "mode_array": [(2, 1), (3, 2), (4, 3)]}
    )
    assert imrphenomhm_sequence_native_supported(
        {**params, "mode_array": []}
    )
    assert not imrphenomhm_sequence_native_supported(
        {**params, "mode_array": [(2, -2)]}
    )
    assert not imrphenomhm_sequence_native_supported(
        {**params, "spin1x": 0.1}
    )
    assert not imrphenomhm_sequence_native_supported(
        {**params, "lambda1": 100.0}
    )
    assert not imrphenomhm_sequence_native_supported(
        {**params, "dchi0": 0.01}
    )


def test_imrphenomhm_sequence_empty_mode_array_is_zero(
    monkeypatch,
    preserve_scheme,
):
    params, sample_points = SEQUENCE_CASES[0]
    monkeypatch.setenv("PYCBC_IMRPHENOMHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    polarizations = get_fd_waveform_sequence(
        approximant="IMRPhenomHM",
        sample_points=sample_points,
        mode_array=[],
        **params,
    )

    for polarization in polarizations:
        assert torch.count_nonzero(polarization._data.tensor) == 0


@pytest.mark.parametrize(
    ("sample_points", "message"),
    [
        ([], "non-empty vector"),
        ([20.0, 20.0, 30.0], "strictly increasing"),
        ([30.0, 20.0], "strictly increasing"),
        ([0.0, 20.0], "positive"),
        ([20.0, np.nan], "finite"),
    ],
)
def test_imrphenomhm_sequence_validates_frequencies(
    sample_points,
    message,
    monkeypatch,
    preserve_scheme,
):
    params, _ = SEQUENCE_CASES[0]
    monkeypatch.setenv("PYCBC_IMRPHENOMHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))

    with pytest.raises(ValueError, match=message):
        get_fd_waveform_sequence(
            approximant="IMRPhenomHM",
            sample_points=sample_points,
            **params,
        )


def test_imrphenomhm_default_sequence_dispatch_avoids_lal_and_host_transfer(
    monkeypatch,
    preserve_scheme,
):
    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData
    import pycbc.waveform.imrphenomhm_torch as hm_torch
    import pycbc.waveform.waveform as waveform

    params, sample_values = SEQUENCE_CASES[0]
    native = hm_torch.imrphenomhm_fd_sequence_torch
    native_calls = 0

    def recording_native(**native_params):
        nonlocal native_calls
        native_calls += 1
        return native(**native_params)

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomHM sequence called LAL")

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomHM sequence transferred to NumPy")

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    sample_points = Array(sample_values)
    monkeypatch.setattr(
        hm_torch,
        "imrphenomhm_fd_sequence_torch",
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
            approximant="IMRPhenomHM",
            sample_points=sample_points,
            **params,
        )

    assert native_calls == 1
    for polarization in polarizations:
        assert isinstance(polarization._data.tensor, torch.Tensor)


@pytest.mark.parametrize(
    ("interface", "lal_name", "native_name"),
    (
        (
            "regular",
            "SimInspiralChooseFDWaveform",
            "imrphenomhm_fd_torch",
        ),
        (
            "sequence",
            "SimInspiralChooseFDWaveformSequence",
            "imrphenomhm_fd_sequence_torch",
        ),
    ),
)
def test_imrphenomhm_unsupported_options_use_lal_fallback(
    interface,
    lal_name,
    native_name,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomhm_torch as hm_torch
    import pycbc.waveform.waveform as waveform

    params, sample_points = SEQUENCE_CASES[0]
    lal_generator = getattr(waveform.lalsimulation, lal_name)
    lal_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported HM sequence reached Torch")

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        hm_torch,
        native_name,
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        lal_name,
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    if interface == "regular":
        polarizations = get_fd_waveform(
            approximant="IMRPhenomHM",
            delta_f=2.0,
            f_lower=20.0,
            f_final=128.0,
            dchi0=0.01,
            **params,
        )
    else:
        polarizations = get_fd_waveform_sequence(
            approximant="IMRPhenomHM",
            sample_points=sample_points,
            dchi0=0.01,
            **params,
        )

    assert lal_calls == 1
    for polarization in polarizations:
        assert isinstance(polarization._data.tensor, torch.Tensor)


@pytest.mark.parametrize(
    ("interface", "lal_name"),
    (
        ("regular", "SimInspiralChooseFDWaveform"),
        ("sequence", "SimInspiralChooseFDWaveformSequence"),
    ),
)
@pytest.mark.parametrize(
    ("disabled_flag", "global_enabled"),
    (
        ("PYCBC_TORCH_NATIVE_PORTS", False),
        ("PYCBC_IMRPHENOMHM_NATIVE", True),
    ),
)
def test_imrphenomhm_default_native_opt_out(
    interface,
    lal_name,
    disabled_flag,
    global_enabled,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.waveform as waveform

    _clear_native_flags(monkeypatch)
    if global_enabled:
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "1")
    monkeypatch.setenv(disabled_flag, "0")
    original = getattr(waveform.lalsimulation, lal_name)
    calls = 0

    def record_lal(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(waveform.lalsimulation, lal_name, record_lal)
    params, sample_points = SEQUENCE_CASES[0]
    _activate_scheme(_scheme.TorchScheme("cpu"))
    if interface == "regular":
        result = get_fd_waveform(
            approximant="IMRPhenomHM",
            delta_f=2.0,
            f_lower=20.0,
            f_final=128.0,
            **params,
        )
    else:
        result = get_fd_waveform_sequence(
            approximant="IMRPhenomHM",
            sample_points=sample_points,
            **params,
        )

    assert calls == 1
    assert all(series._data.tensor.device.type == "cpu" for series in result)


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomhm_sequence_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    base, sample_points = SEQUENCE_CASES[0]
    params = {**base, "f_ref": 0.0}
    monkeypatch.setenv("PYCBC_IMRPHENOMHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform_sequence(
        approximant="IMRPhenomHM",
        sample_points=sample_points,
        **params,
    )
    reference_array = reference.numpy().copy()

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, cross = get_fd_waveform_sequence(
        approximant="IMRPhenomHM",
        sample_points=sample_points,
        **params,
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert actual._data.tensor.device.type == device_name
    assert cross._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    relative_error = np.linalg.norm(
        actual.numpy() - reference_array
    ) / np.linalg.norm(reference_array)
    tolerance = 1.0e-4 if device_name == "mps" else 1.0e-10
    assert relative_error < tolerance
