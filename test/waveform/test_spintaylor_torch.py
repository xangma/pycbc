import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import get_td_waveform, get_td_waveform_modes  # noqa: E402
from pycbc.waveform.spintaylor_modes_torch import (  # noqa: E402
    spintaylor_modes_from_orbit,
)
from pycbc.waveform.spintaylor_torch import (  # noqa: E402
    _orbital_l_spin_to_radiation_frame,
    spintaylor_polarizations_from_orbit,
    spintaylor_t1_modes_native_supported,
    spintaylor_t1_modes_torch,
    spintaylor_t1_native_supported,
    spintaylor_t1_orbit,
    spintaylor_t1_td_torch,
    spintaylor_t4_modes_native_supported,
    spintaylor_t4_modes_torch,
    spintaylor_t4_native_supported,
    spintaylor_t4_orbit,
    spintaylor_t4_td_torch,
    spintaylor_t5_modes_native_supported,
    spintaylor_t5_modes_torch,
    spintaylor_t5_native_supported,
    spintaylor_t5_orbit,
    spintaylor_t5_td_torch,
)


_PUBLIC_PARAMETERS = {
    "mass1": 45.0,
    "mass2": 25.0,
    "spin1x": 0.25,
    "spin1y": -0.1,
    "spin1z": 0.4,
    "spin2x": -0.15,
    "spin2y": 0.05,
    "spin2z": -0.2,
    "distance": 300.0,
    "inclination": 0.8,
    "coa_phase": 0.37,
    "long_asc_nodes": 0.23,
    "delta_t": 1.0 / 2048.0,
    "f_lower": 30.0,
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


def _activate_scheme(state):
    _scheme.Scheme._single = None
    _scheme.mgr.state = state


def _lal_series(name, samples):
    series = lal.CreateREAL8TimeSeries(
        name,
        lal.LIGOTimeGPS(0),
        0.0,
        1.0,
        lal.DimensionlessUnit,
        len(samples),
    )
    series.data.data[:] = samples
    return series


def _orbit(dtype=torch.float64, device="cpu"):
    velocity = torch.linspace(0.14, 0.29, 17, dtype=dtype, device=device)
    phase = torch.linspace(-0.7, 3.1, 17, dtype=dtype, device=device)
    beta = torch.linspace(0.2, 0.8, 17, dtype=dtype, device=device)
    lnhat = torch.stack(
        (torch.sin(beta), torch.zeros_like(beta), torch.cos(beta)),
        dim=-1,
    )
    e1 = torch.stack(
        (torch.cos(beta), torch.zeros_like(beta), -torch.sin(beta)),
        dim=-1,
    )
    spin1 = torch.stack(
        (
            0.31 + 0.02 * torch.sin(phase),
            -0.22 + 0.01 * torch.cos(phase),
            0.43 + 0.03 * torch.sin(beta),
        ),
        dim=-1,
    )
    spin2 = torch.stack(
        (
            -0.17 + 0.01 * torch.cos(phase),
            0.25 + 0.02 * torch.sin(phase),
            -0.34 + 0.01 * torch.cos(beta),
        ),
        dim=-1,
    )
    return velocity, phase, spin1, spin2, lnhat, e1


def _lal_polarizations(orbit, amplitude_order):
    velocity, phase, spin1, spin2, lnhat, e1 = (
        value.detach().cpu().numpy() for value in orbit
    )
    series = (
        _lal_series("V", velocity),
        _lal_series("Phi", phase),
        *(
            _lal_series(f"S1{axis}", spin1[:, index])
            for index, axis in enumerate("xyz")
        ),
        *(
            _lal_series(f"S2{axis}", spin2[:, index])
            for index, axis in enumerate("xyz")
        ),
        *(
            _lal_series(f"LNhat{axis}", lnhat[:, index])
            for index, axis in enumerate("xyz")
        ),
        *(_lal_series(f"E1{axis}", e1[:, index]) for index, axis in enumerate("xyz")),
    )
    plus, cross = lalsimulation.SimInspiralPrecessingPolarizationWaveforms(
        *series,
        37.0 * lal.MSUN_SI,
        19.0 * lal.MSUN_SI,
        240.0e6 * lal.PC_SI,
        amplitude_order,
    )
    return plus.data.data, cross.data.data


def _lal_modes(orbit, amplitude_order, ells=(2, 3, 4)):
    velocity, phase, spin1, spin2, lnhat, e1 = (
        value.detach().cpu().numpy() for value in orbit
    )
    series = (
        _lal_series("V", velocity),
        _lal_series("Phi", phase),
        *(
            _lal_series(f"LNhat{axis}", lnhat[:, index])
            for index, axis in enumerate("xyz")
        ),
        *(_lal_series(f"E1{axis}", e1[:, index]) for index, axis in enumerate("xyz")),
        *(
            _lal_series(f"S1{axis}", spin1[:, index])
            for index, axis in enumerate("xyz")
        ),
        *(
            _lal_series(f"S2{axis}", spin2[:, index])
            for index, axis in enumerate("xyz")
        ),
    )
    modes = {}
    for ell in ells:
        status, linked_modes = getattr(lalsimulation, f"SimInspiralSpinPNMode{ell}m")(
            *series,
            37.0 * lal.MSUN_SI,
            19.0 * lal.MSUN_SI,
            240.0e6 * lal.PC_SI,
            amplitude_order,
        )
        assert status == 0
        while linked_modes:
            modes[(linked_modes.l, linked_modes.m)] = np.asarray(
                linked_modes.mode.data.data
            ).copy()
            linked_modes = linked_modes.next
    return modes


def _lal_spintaylor_orbit(
    mass1,
    mass2,
    delta_t,
    f_lower,
    spin1,
    spin2,
    *,
    coa_phase,
    f_ref,
    f_final,
    lnhat,
    e1,
    lambda1=0.0,
    lambda2=0.0,
    tidal_order=-1,
    approximant=lalsimulation.SpinTaylorT4,
):
    parameters = lal.CreateDict()
    if f_final:
        lalsimulation.SimInspiralWaveformParamsInsertFinalFreq(
            parameters,
            f_final,
        )
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(
        parameters,
        lambda1,
    )
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(
        parameters,
        lambda2,
    )
    if tidal_order != -1:
        lalsimulation.SimInspiralWaveformParamsInsertPNTidalOrder(
            parameters,
            tidal_order,
        )
    series = lalsimulation.SimInspiralSpinTaylorOrbitalDriver(
        coa_phase,
        delta_t,
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        f_lower,
        f_ref,
        *spin1,
        *spin2,
        *lnhat,
        *e1,
        parameters,
        approximant,
    )
    values = [np.asarray(item.data.data) for item in series]
    return {
        "velocity": values[0],
        "phase": values[1],
        "spin1": np.stack(values[2:5], axis=-1),
        "spin2": np.stack(values[5:8], axis=-1),
        "lnhat": np.stack(values[8:11], axis=-1),
        "e1": np.stack(values[11:14], axis=-1),
        "epoch": float(series[0].epoch),
    }


@pytest.mark.parametrize("amplitude_order", (-1, 0, 1, 2, 3))
def test_spintaylor_polarizations_match_lalsimulation(amplitude_order):
    orbit = _orbit()
    expected_plus, expected_cross = _lal_polarizations(orbit, amplitude_order)
    actual_plus, actual_cross = spintaylor_polarizations_from_orbit(
        *orbit,
        37.0,
        19.0,
        240.0,
        amplitude_order=amplitude_order,
    )

    np.testing.assert_allclose(
        actual_plus.numpy(),
        expected_plus,
        rtol=2.0e-13,
        atol=1.0e-32,
    )
    np.testing.assert_allclose(
        actual_cross.numpy(),
        expected_cross,
        rtol=2.0e-13,
        atol=1.0e-32,
    )


@pytest.mark.parametrize("device_name", ("cpu", "cuda", "mps"))
def test_spintaylor_polarizations_preserve_device_and_dtype(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")
    dtype = torch.float32 if device_name == "mps" else torch.float64
    orbit = _orbit(dtype=dtype, device=device_name)
    plus, cross = spintaylor_polarizations_from_orbit(
        *orbit,
        37.0,
        19.0,
        240.0,
    )

    assert plus.device.type == device_name
    assert cross.device.type == device_name
    assert plus.dtype == dtype
    assert cross.dtype == dtype
    assert torch.isfinite(plus).all()
    assert torch.isfinite(cross).all()


def test_spintaylor_polarizations_validate_amplitude_order():
    with pytest.raises(ValueError, match="unsupported amplitude_order"):
        spintaylor_polarizations_from_orbit(
            *_orbit(),
            37.0,
            19.0,
            240.0,
            amplitude_order=4,
        )


@pytest.mark.parametrize("amplitude_order", (-1, 0, 1, 2, 3))
def test_spintaylor_modes_match_lalsimulation(amplitude_order):
    orbit = _orbit()
    expected = _lal_modes(orbit, amplitude_order)
    actual = spintaylor_modes_from_orbit(
        *orbit,
        37.0,
        19.0,
        240.0,
        amplitude_order=amplitude_order,
    )

    assert set(actual) == set(expected)
    for mode, (real, imaginary) in actual.items():
        np.testing.assert_allclose(
            real.numpy() + 1j * imaginary.numpy(),
            expected[mode],
            rtol=3.0e-13,
            atol=1.0e-32,
        )


@pytest.mark.parametrize("device_name", ("cpu", "cuda", "mps"))
def test_spintaylor_modes_preserve_device_and_dtype(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")
    dtype = torch.float32 if device_name == "mps" else torch.float64
    modes = spintaylor_modes_from_orbit(
        *_orbit(dtype=dtype, device=device_name),
        37.0,
        19.0,
        240.0,
        ells=(2,),
    )

    assert set(modes) == {(2, emm) for emm in range(-2, 3)}
    for real, imaginary in modes.values():
        assert real.device.type == device_name
        assert imaginary.device.type == device_name
        assert real.dtype == dtype
        assert imaginary.dtype == dtype
        assert torch.isfinite(real).all()
        assert torch.isfinite(imaginary).all()


@pytest.mark.parametrize("f_ref", (0.0, 25.0, 34.0))
def test_spintaylor_t4_orbit_matches_lalsimulation(f_ref):
    mass1 = 45.0
    mass2 = 25.0
    delta_t = 1.0 / 2048.0
    f_lower = 25.0
    f_final = 52.0
    coa_phase = 0.37
    spin1 = (0.25, -0.1, 0.4)
    spin2 = (-0.15, 0.05, -0.2)
    lnhat = (0.2, -0.3, np.sqrt(0.87))
    norm = np.sqrt(0.91)
    e1 = (np.sqrt(0.87) / norm, 0.0, -0.2 / norm)
    expected = _lal_spintaylor_orbit(
        mass1,
        mass2,
        delta_t,
        f_lower,
        spin1,
        spin2,
        coa_phase=coa_phase,
        f_ref=f_ref,
        f_final=f_final,
        lnhat=lnhat,
        e1=e1,
    )
    actual = spintaylor_t4_orbit(
        mass1,
        mass2,
        delta_t,
        f_lower,
        spin1,
        spin2,
        coa_phase=coa_phase,
        f_ref=f_ref,
        f_final=f_final,
        lnhat=lnhat,
        e1=e1,
    )

    assert len(actual) == len(expected["velocity"])
    assert actual.delta_t == delta_t
    assert actual.epoch == pytest.approx(expected["epoch"], abs=1.0e-9)
    np.testing.assert_allclose(
        actual.velocity.numpy(),
        expected["velocity"],
        rtol=2.0e-8,
        atol=3.0e-9,
    )
    np.testing.assert_allclose(
        actual.phase.numpy(),
        expected["phase"],
        rtol=2.0e-8,
        atol=5.0e-8,
    )
    for name in ("spin1", "spin2", "lnhat", "e1"):
        np.testing.assert_allclose(
            getattr(actual, name).numpy(),
            expected[name],
            rtol=2.0e-8,
            atol=1.0e-8,
        )


def test_spintaylor_t4_orbit_validates_tidal_order():
    with pytest.raises(ValueError, match="tidal_order must be one of"):
        spintaylor_t4_orbit(
            45.0,
            25.0,
            1.0 / 2048.0,
            30.0,
            (0.25, -0.1, 0.4),
            (-0.15, 0.05, -0.2),
            tidal_order=8,
        )


@pytest.mark.parametrize(
    ("orbit_generator", "lal_approximant"),
    (
        (spintaylor_t1_orbit, lalsimulation.SpinTaylorT1),
        (spintaylor_t4_orbit, lalsimulation.SpinTaylorT4),
        (spintaylor_t5_orbit, lalsimulation.SpinTaylorT5),
    ),
)
def test_spintaylor_orbit_reaches_physical_boundary(orbit_generator, lal_approximant):
    parameters = {
        "mass1": 45.0,
        "mass2": 25.0,
        "delta_t": 1.0 / 1024.0,
        "f_lower": 30.0,
        "spin1": (0.25, -0.1, 0.4),
        "spin2": (-0.15, 0.05, -0.2),
        "coa_phase": 0.2,
        "f_ref": 30.0,
    }
    expected = _lal_spintaylor_orbit(
        **parameters,
        f_final=0.0,
        lnhat=(0.0, 0.0, 1.0),
        e1=(1.0, 0.0, 0.0),
        approximant=lal_approximant,
    )
    actual = orbit_generator(**parameters)

    assert len(actual) == len(expected["velocity"])
    np.testing.assert_allclose(
        actual.velocity.numpy(),
        expected["velocity"],
        rtol=2.0e-8,
        atol=3.0e-9,
    )
    np.testing.assert_allclose(
        actual.phase.numpy(),
        expected["phase"],
        rtol=2.0e-8,
        atol=5.0e-8,
    )


@pytest.mark.parametrize("tidal_order", (-1, 0, 10, 12))
@pytest.mark.parametrize(
    ("orbit_generator", "lal_approximant"),
    (
        (spintaylor_t1_orbit, lalsimulation.SpinTaylorT1),
        (spintaylor_t4_orbit, lalsimulation.SpinTaylorT4),
        (spintaylor_t5_orbit, lalsimulation.SpinTaylorT5),
    ),
)
def test_spintaylor_orbit_matches_lalsimulation_with_tides(
    tidal_order, orbit_generator, lal_approximant
):
    parameters = {
        "mass1": 45.0,
        "mass2": 25.0,
        "delta_t": 1.0 / 2048.0,
        "f_lower": 30.0,
        "spin1": (0.25, -0.1, 0.4),
        "spin2": (-0.15, 0.05, -0.2),
        "coa_phase": 0.37,
        "f_ref": 30.0,
        "lambda1": 300.0,
        "lambda2": 500.0,
        "tidal_order": tidal_order,
    }
    expected = _lal_spintaylor_orbit(
        **parameters,
        f_final=0.0,
        lnhat=(0.0, 0.0, 1.0),
        e1=(1.0, 0.0, 0.0),
        approximant=lal_approximant,
    )
    actual = orbit_generator(**parameters)

    assert len(actual) == len(expected["velocity"])
    assert actual.epoch == pytest.approx(expected["epoch"], abs=1.0e-9)
    np.testing.assert_allclose(
        actual.velocity.numpy(),
        expected["velocity"],
        rtol=2.0e-8,
        atol=3.0e-9,
    )
    np.testing.assert_allclose(
        actual.phase.numpy(),
        expected["phase"],
        rtol=2.0e-8,
        atol=5.0e-8,
    )
    for name in ("spin1", "spin2", "lnhat", "e1"):
        np.testing.assert_allclose(
            getattr(actual, name).numpy(),
            expected[name],
            rtol=2.0e-8,
            atol=1.0e-8,
        )


@pytest.mark.parametrize("device_name", ("cpu", "cuda", "mps"))
def test_spintaylor_t4_orbit_preserves_device_and_dtype(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")
    dtype = torch.float32 if device_name == "mps" else torch.float64
    actual = spintaylor_t4_orbit(
        60.0,
        30.0,
        1.0 / 1024.0,
        40.0,
        torch.tensor((0.2, -0.1, 0.3), device=device_name, dtype=dtype),
        torch.tensor((-0.1, 0.05, -0.2), device=device_name, dtype=dtype),
        f_ref=40.0,
        f_final=48.0,
        device=device_name,
        dtype=dtype,
    )

    assert actual.state.device.type == device_name
    assert actual.state.dtype == dtype
    assert torch.isfinite(actual.state).all()


def test_spintaylor_t4_orbit_validates_reference_frequency():
    with pytest.raises(ValueError, match="f_ref must be zero or at least f_lower"):
        spintaylor_t4_orbit(
            45.0,
            25.0,
            1.0 / 2048.0,
            25.0,
            (0.25, -0.1, 0.4),
            (-0.15, 0.05, -0.2),
            f_ref=20.0,
        )


def test_orbital_l_spin_rotation_matches_lalsimulation():
    inclination = 0.8
    coa_phase = 0.37
    spin1 = (0.25, -0.1, 0.4)
    spin2 = (-0.15, 0.05, -0.2)
    expected = lalsimulation.SimInspiralInitialConditionsPrecessingApproxs(
        inclination,
        *spin1,
        *spin2,
        45.0 * lal.MSUN_SI,
        25.0 * lal.MSUN_SI,
        30.0,
        coa_phase,
        lalsimulation.SIM_INSPIRAL_FRAME_AXIS_ORBITAL_L,
    )

    assert expected[0] == pytest.approx(inclination)
    np.testing.assert_allclose(
        _orbital_l_spin_to_radiation_frame(spin1, inclination, coa_phase),
        expected[1:4],
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        _orbital_l_spin_to_radiation_frame(spin2, inclination, coa_phase),
        expected[4:7],
        rtol=0.0,
        atol=2.0e-16,
    )


@pytest.mark.parametrize(
    ("parameters", "expected"),
    (
        ({}, True),
        ({"phase_order": 7, "spin_order": 6, "tidal_order": 12}, True),
        ({"phase_order": np.int64(8), "amplitude_order": 3}, True),
        ({"phase_order": 6}, False),
        ({"spin_order": 5}, False),
        ({"spin_order": 7}, False),
        ({"tidal_order": 8}, False),
        ({"amplitude_order": 4}, False),
        ({"eccentricity_order": 0}, False),
        ({"lambda1": 300.0}, True),
        ({"dquad_mon1": 0.1}, False),
        ({"eccentricity": 0.01}, False),
        ({"frame_axis": 2}, True),
        ({"frame_axis": 1}, False),
        ({"modes_choice": 1}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"numrel_data": "waveform.h5"}, False),
    ),
)
@pytest.mark.parametrize(
    "support",
    (
        spintaylor_t1_native_supported,
        spintaylor_t4_native_supported,
        spintaylor_t5_native_supported,
    ),
)
def test_spintaylor_native_support_boundary(parameters, expected, support):
    assert support(parameters) is expected


def test_spintaylor_native_support_rejects_other_approximant():
    assert not spintaylor_t1_native_supported({"approximant": "SpinTaylorT4"})
    assert not spintaylor_t4_native_supported({"approximant": "SpinTaylorT1"})
    assert not spintaylor_t5_native_supported({"approximant": "SpinTaylorT1"})


@pytest.mark.parametrize(
    ("parameters", "expected"),
    (
        ({}, True),
        ({"ell_max": 2}, True),
        ({"ell_max": 5}, True),
        ({"ell_max": 1}, False),
        ({"ell_max": 6}, False),
        ({"ell_max": 3.0}, False),
        ({"mode_array": [(2, 2)]}, True),
        ({"mode_array": [(3, 1), (4, -4)]}, True),
        ({"mode_array": [(2, 2)], "ell_max": 1}, True),
        ({"mode_array": []}, False),
        ({"mode_array": [(2, 3)]}, False),
        ({"mode_array": [(5, 5)]}, False),
        ({"mode_array": [(2.0, 2)]}, False),
        ({"phase_order": 6}, False),
        ({"frame_axis": 1}, False),
    ),
)
@pytest.mark.parametrize(
    "support",
    (
        spintaylor_t1_modes_native_supported,
        spintaylor_t4_modes_native_supported,
        spintaylor_t5_modes_native_supported,
    ),
)
def test_spintaylor_modes_native_support_boundary(parameters, expected, support):
    assert support(parameters) is expected


@pytest.mark.parametrize("f_ref", (0.0, 38.0))
@pytest.mark.parametrize("amplitude_order", (-1, 0, 3))
@pytest.mark.parametrize(
    ("approximant", "generator"),
    (
        ("SpinTaylorT1", spintaylor_t1_td_torch),
        ("SpinTaylorT4", spintaylor_t4_td_torch),
        ("SpinTaylorT5", spintaylor_t5_td_torch),
    ),
)
def test_spintaylor_waveform_matches_lalsuite(
    f_ref, amplitude_order, approximant, generator, preserve_scheme
):
    parameters = dict(
        _PUBLIC_PARAMETERS,
        f_ref=f_ref,
        amplitude_order=amplitude_order,
    )
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant=approximant, **parameters)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = generator(**parameters)

    for expected, expected_array, result in zip(reference, reference_arrays, actual):
        result_array = result.numpy()
        assert len(result) == len(expected)
        assert result.delta_t == expected.delta_t
        assert abs(float(result.start_time - expected.start_time)) <= 1.0e-9
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.float64
        relative_error = np.linalg.norm(result_array - expected_array) / np.linalg.norm(
            expected_array
        )
        assert relative_error < 2.0e-7


@pytest.mark.parametrize("tidal_order", (-1, 0, 10, 12))
@pytest.mark.parametrize(
    ("approximant", "generator"),
    (
        ("SpinTaylorT1", spintaylor_t1_td_torch),
        ("SpinTaylorT4", spintaylor_t4_td_torch),
        ("SpinTaylorT5", spintaylor_t5_td_torch),
    ),
)
def test_spintaylor_tidal_waveform_matches_lalsuite(
    tidal_order, approximant, generator, preserve_scheme
):
    parameters = dict(
        _PUBLIC_PARAMETERS,
        lambda1=300.0,
        lambda2=500.0,
        tidal_order=tidal_order,
    )
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant=approximant, **parameters)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = generator(**parameters)

    for expected, expected_array, result in zip(reference, reference_arrays, actual):
        assert len(result) == len(expected)
        assert result.delta_t == expected.delta_t
        assert abs(float(result.start_time - expected.start_time)) <= 1.0e-9
        relative_error = np.linalg.norm(
            result.numpy() - expected_array
        ) / np.linalg.norm(expected_array)
        assert relative_error < 2.0e-7


@pytest.mark.parametrize(
    "updates",
    (
        {"f_ref": 0.0, "amplitude_order": -1, "ell_max": 4},
        {
            "f_ref": 38.0,
            "amplitude_order": 3,
            "mode_array": [(3, 1)],
            "lambda1": 300.0,
            "lambda2": 500.0,
            "tidal_order": 12,
        },
    ),
)
@pytest.mark.parametrize(
    ("approximant", "generator"),
    (
        ("SpinTaylorT1", spintaylor_t1_modes_torch),
        ("SpinTaylorT4", spintaylor_t4_modes_torch),
        ("SpinTaylorT5", spintaylor_t5_modes_torch),
    ),
)
def test_spintaylor_modes_match_lalsuite(
    updates, approximant, generator, preserve_scheme
):
    parameters = dict(_PUBLIC_PARAMETERS, **updates)
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform_modes(approximant=approximant, **parameters)
    reference_arrays = {
        mode: real.numpy().copy() + 1j * imaginary.numpy().copy()
        for mode, (real, imaginary) in reference.items()
    }

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = generator(**parameters)

    assert set(actual) == set(reference)
    for mode, (actual_real, actual_imaginary) in actual.items():
        expected_real, _ = reference[mode]
        result = actual_real.numpy() + 1j * actual_imaginary.numpy()
        expected = reference_arrays[mode]
        assert len(actual_real) == len(expected_real)
        assert actual_real.delta_t == expected_real.delta_t
        assert abs(float(actual_real.start_time - expected_real.start_time)) <= 1.0e-9
        assert actual_real._data.tensor.device.type == "cpu"
        assert actual_imaginary._data.tensor.device.type == "cpu"
        relative_error = np.linalg.norm(result - expected) / np.linalg.norm(expected)
        assert relative_error < 5.0e-7


@pytest.mark.parametrize(
    "generator",
    (
        spintaylor_t1_modes_torch,
        spintaylor_t4_modes_torch,
        spintaylor_t5_modes_torch,
    ),
)
def test_spintaylor_modes_ignore_coalescence_phase(generator, preserve_scheme):
    parameters = dict(_PUBLIC_PARAMETERS, f_ref=38.0, ell_max=2)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    first = generator(**parameters)
    second = generator(**dict(parameters, coa_phase=-2.1))

    for mode in first:
        for first_series, second_series in zip(first[mode], second[mode]):
            torch.testing.assert_close(
                first_series._data.tensor,
                second_series._data.tensor,
                rtol=0.0,
                atol=0.0,
            )


@pytest.mark.parametrize(
    ("approximant", "component_flag", "generator_name"),
    (
        ("SpinTaylorT1", "PYCBC_SPINTAYLORT1_NATIVE", "spintaylor_t1_td_torch"),
        ("SpinTaylorT4", "PYCBC_SPINTAYLORT4_NATIVE", "spintaylor_t4_td_torch"),
        ("SpinTaylorT5", "PYCBC_SPINTAYLORT5_NATIVE", "spintaylor_t5_td_torch"),
    ),
)
def test_spintaylor_public_native_dispatch_avoids_lalsimulation(
    approximant, component_flag, generator_name, monkeypatch, preserve_scheme
):
    parameters = dict(
        _PUBLIC_PARAMETERS,
        f_ref=38.0,
        lambda1=300.0,
        lambda2=500.0,
        tidal_order=12,
    )
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv(component_flag, "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant=approximant, **parameters)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.spintaylor_torch as spintaylor_module
    import pycbc.waveform.waveform as waveform_module

    native_generator = getattr(spintaylor_module, generator_name)
    native_calls = 0

    def recording_native(**native_parameters):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError(f"native {approximant} called lalsimulation")

    monkeypatch.setattr(
        spintaylor_module,
        generator_name,
        recording_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        unexpected_lalsimulation,
    )
    monkeypatch.setenv(component_flag, "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_td_waveform(approximant=approximant, **parameters)

    assert native_calls == 1
    for expected, expected_array, result in zip(reference, reference_arrays, actual):
        assert len(result) == len(expected)
        assert result._data.tensor.device.type == "cpu"
        relative_error = np.linalg.norm(
            result.numpy() - expected_array
        ) / np.linalg.norm(expected_array)
        assert relative_error < 2.0e-7


@pytest.mark.parametrize(
    ("approximant", "component_flag", "generator_name"),
    (
        (
            "SpinTaylorT1",
            "PYCBC_SPINTAYLORT1_NATIVE",
            "spintaylor_t1_modes_torch",
        ),
        (
            "SpinTaylorT4",
            "PYCBC_SPINTAYLORT4_NATIVE",
            "spintaylor_t4_modes_torch",
        ),
        (
            "SpinTaylorT5",
            "PYCBC_SPINTAYLORT5_NATIVE",
            "spintaylor_t5_modes_torch",
        ),
    ),
)
def test_spintaylor_public_native_mode_dispatch_avoids_lalsimulation(
    approximant, component_flag, generator_name, monkeypatch, preserve_scheme
):
    parameters = dict(_PUBLIC_PARAMETERS, f_ref=38.0, ell_max=3)
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv(component_flag, "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform_modes(approximant=approximant, **parameters)

    import pycbc.waveform.spintaylor_torch as spintaylor_module
    import pycbc.waveform.waveform_modes as waveform_modes_module

    native_generator = getattr(spintaylor_module, generator_name)
    native_calls = 0

    def recording_native(**native_parameters):
        nonlocal native_calls
        native_calls += 1
        return native_generator(**native_parameters)

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError(f"native {approximant} modes called lalsimulation")

    monkeypatch.setattr(
        spintaylor_module,
        generator_name,
        recording_native,
    )
    monkeypatch.setattr(
        waveform_modes_module.lalsimulation,
        "SimInspiralChooseTDModes",
        unexpected_lalsimulation,
    )
    monkeypatch.setenv(component_flag, "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_td_waveform_modes(approximant=approximant, **parameters)

    assert native_calls == 1
    assert set(actual) == set(reference)
    assert all(
        isinstance(series._data.tensor, torch.Tensor)
        for pair in actual.values()
        for series in pair
    )


def test_spintaylor_t4_unsupported_frame_uses_lal_fallback(
    monkeypatch, preserve_scheme
):
    parameters = dict(_PUBLIC_PARAMETERS, frame_axis=1)
    _activate_scheme(_scheme.CPUScheme())
    reference = get_td_waveform(approximant="SpinTaylorT4", **parameters)

    import pycbc.waveform.spintaylor_torch as spintaylor_module
    import pycbc.waveform.waveform as waveform_module

    def unexpected_native(**_parameters):
        raise AssertionError("unsupported SpinTaylorT4 parameters reached Torch")

    lal_generator = waveform_module.lalsimulation.SimInspiralChooseTDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        spintaylor_module,
        "spintaylor_t4_td_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_module.lalsimulation,
        "SimInspiralChooseTDWaveform",
        recording_lal,
    )
    monkeypatch.setenv("PYCBC_SPINTAYLORT4_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_td_waveform(approximant="SpinTaylorT4", **parameters)

    assert lal_calls == 1
    for expected, actual in zip(reference, fallback):
        assert len(actual) == len(expected)
        assert actual._data.tensor.device.type == "cpu"


@pytest.mark.parametrize(
    ("waveform_generator", "mode_generator"),
    (
        (spintaylor_t1_td_torch, spintaylor_t1_modes_torch),
        (spintaylor_t4_td_torch, spintaylor_t4_modes_torch),
        (spintaylor_t5_td_torch, spintaylor_t5_modes_torch),
    ),
)
def test_spintaylor_requires_torch_scheme(
    waveform_generator, mode_generator, preserve_scheme
):
    _activate_scheme(_scheme.CPUScheme())
    with pytest.raises(TypeError, match="active TorchScheme"):
        waveform_generator(**_PUBLIC_PARAMETERS)
    with pytest.raises(TypeError, match="active TorchScheme"):
        mode_generator(**_PUBLIC_PARAMETERS)
