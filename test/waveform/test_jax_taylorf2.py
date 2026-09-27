"""Parity tests for the exact JAX TaylorF2 port."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("lalsimulation")

from pycbc.scheme import JAXScheme
from pycbc.waveform import get_fd_waveform, get_fd_waveform_sequence
from pycbc.waveform.taylorf2_jax import generate_fd


def _parameters():
    return dict(
        approximant="TaylorF2", mass1=30.0, mass2=20.0,
        spin1z=0.2, spin2z=-0.1, distance=100.0,
        inclination=0.7, coa_phase=0.35, f_ref=31.0,
        f_lower=20.0, f_final=220.0, delta_f=1.0,
        phase_order=-1, spin_order=-1, amplitude_order=0,
        tidal_order=-1,
    )


@pytest.mark.parametrize("mass1,mass2,spin1z,spin2z,phase_order,spin_order,f_ref,delta_f,f_final", [
    (1.4, 1.4, 0.0, 0.0, -1, 0, 0.0, 1.0, 0.0),
    (30.0, 20.0, 0.2, -0.1, 3, 3, 31.0, 1.0, 220.0),
    (70.0, 10.0, 0.7, -0.5, 7, 7, 31.0, 0.25, 220.0),
])
def test_taylorf2_jax_varied_cpu_reference(
        mass1, mass2, spin1z, spin2z, phase_order, spin_order, f_ref,
        delta_f, f_final):
    params = _parameters()
    params.update(mass1=mass1, mass2=mass2, spin1z=spin1z, spin2z=spin2z,
                  phase_order=phase_order, spin_order=spin_order,
                  f_ref=f_ref, delta_f=delta_f, f_final=f_final)
    expected = get_fd_waveform(**params)
    with JAXScheme():
        actual = get_fd_waveform(**params)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                                   rtol=3e-11, atol=2e-30)


def test_taylorf2_jax_public_sequence_matches_cpu():
    params = _parameters()
    params.pop("f_final")
    params["sample_points"] = np.array([20.0, 31.0, 64.0, 128.0, 200.0])
    expected = get_fd_waveform_sequence(**params)
    with JAXScheme():
        actual = get_fd_waveform_sequence(**params)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                                   rtol=3e-11, atol=2e-30)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_taylorf2_jax_public_cuda_when_available(device):
    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip(f"JAX {device} backend unavailable")
    params = _parameters()
    expected = get_fd_waveform(**params)
    with JAXScheme(device):
        actual = get_fd_waveform(**params)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                                   rtol=3e-11, atol=2e-30)


def test_taylorf2_jax_matches_lal_complex128():
    params = _parameters()
    expected = get_fd_waveform(**params)
    with JAXScheme():
        actual = generate_fd(**params, dtype="complex128")
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                                   rtol=3e-11, atol=2e-30)
        assert got.epoch == want.epoch


def test_taylorf2_jax_complex64_is_device_result_and_matches_lal():
    params = _parameters()
    expected = get_fd_waveform(**params)
    with JAXScheme():
        actual = generate_fd(**params, dtype="complex64")
    for got, want in zip(actual, expected):
        assert type(got._data).__name__ == "JAXArrayData"
        got, want = np.asarray(got), np.asarray(want)
        mask = np.abs(want) > np.max(np.abs(want)) * 1e-4
        np.testing.assert_allclose(np.abs(got[mask]) / np.abs(want[mask]),
                                   1.0, rtol=0.0, atol=2e-6)
        np.testing.assert_allclose(np.angle(got[mask] / want[mask]),
                                   0.0, rtol=0.0, atol=1e-7)


def test_taylorf2_jax_rejects_unimplemented_tides():
    params = _parameters()
    params["lambda1"] = 100.0
    with JAXScheme():
        with pytest.raises(ValueError, match="unsupported"):
            generate_fd(**params)
