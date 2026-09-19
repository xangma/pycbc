# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""JAX integration tests for the public waveform-mode helpers."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
lal = pytest.importorskip("lal")

from pycbc import scheme  # noqa: E402
from pycbc.types import FrequencySeries  # noqa: E402
from pycbc.types.backend import (  # noqa: E402
    backend_array,
    wrap_backend_array,
)
from pycbc.waveform import (  # noqa: E402
    filter_approximants,
    get_fd_waveform,
    get_fd_waveform_sequence,
    get_sgburst_waveform,
)
from pycbc.waveform import waveform as waveform_module  # noqa: E402
from pycbc.waveform import waveform_modes  # noqa: E402
from pycbc.waveform.waveform_modes import get_glm, sum_modes  # noqa: E402


@pytest.fixture
def jax_ctx():
    ctx = scheme.JAXScheme()
    try:
        yield ctx
    finally:
        del ctx
        scheme.Scheme._single = None


def _modes():
    x = np.arange(32, dtype=np.float64)
    return {
        (2, 2): FrequencySeries(x + 1j * x[::-1], delta_f=0.25),
        (3, -2): FrequencySeries(0.5 * x - 0.25j * x, delta_f=0.25),
        (4, 1): FrequencySeries(np.cos(x) + 1j * np.sin(x), delta_f=0.25),
    }


def test_sum_modes_matches_lal_path(jax_ctx):
    reference_modes = _modes()
    reference = sum_modes(reference_modes, inclination=0.7, phi=-0.2)

    with jax_ctx:
        jax_modes = {
            mode: FrequencySeries(series.numpy(), delta_f=series.delta_f)
            for mode, series in reference_modes.items()
        }
        actual = sum_modes(jax_modes, inclination=0.7, phi=-0.2)

    assert backend_array(actual, "jax") is not None
    np.testing.assert_allclose(
        actual.numpy(), reference.numpy(), rtol=2e-13, atol=2e-13
    )


def test_sum_modes_preserves_metadata_and_sample_gradients(jax_ctx):
    inclination, phi = 0.7, -0.2
    modes = ((2, 2), (3, -2), (4, 1))

    # Test gradients using JAX autodiff directly on mode inputs
    def loss_fn(mode_real, mode_imag):
        supplied = {}
        for m, (ell, emm) in enumerate(modes):
            data = mode_real[m] + 1j * mode_imag[m]
            fs = FrequencySeries(wrap_backend_array(data), delta_f=0.25, epoch=123, copy=False)
            supplied[(ell, emm)] = fs
        result = sum_modes(supplied, inclination, phi)
        arr = backend_array(result, "jax")
        return jnp.sum(jnp.real(arr))

    init_real = jnp.array([[1.0, 3.0], [0.5, -0.5], [2.0, 1.0]], dtype=jnp.float64)
    init_imag = jnp.array([[2.0, -1.0], [1.0, 2.0], [-1.0, 3.0]], dtype=jnp.float64)

    with jax_ctx:
        grad_r, grad_i = jax.grad(loss_fn, argnums=(0, 1))(init_real, init_imag)

    for m, mode in enumerate(modes):
        harmonic = lal.SpinWeightedSphericalHarmonic(inclination, phi, -2, *mode)
        # derivative of Re(Y * h) with respect to Re(h) is Re(Y)
        # derivative of Re(Y * h) with respect to Im(h) is -Im(Y)
        np.testing.assert_allclose(grad_r[m], harmonic.real, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(grad_i[m], -harmonic.imag, rtol=2e-12, atol=2e-12)


def test_get_glm_remains_lal_compatible():
    expected = lal.SpinWeightedSphericalHarmonic(0.9, 0.0, -2, 3, -1).real
    assert get_glm(3, -1, 0.9) == pytest.approx(expected, abs=0.0)


def test_interpolated_lal_waveform_is_jax_backed(jax_ctx):
    with jax_ctx:
        hp, hc = get_fd_waveform(
            approximant="TaylorF2_INTERP",
            mass1=20,
            mass2=10,
            delta_f=0.25,
            f_lower=30,
        )

    assert backend_array(hp, "jax") is not None
    assert backend_array(hc, "jax") is not None
    assert np.isfinite(hp.numpy()).all()
    assert np.isfinite(hc.numpy()).all()


def test_spatmplt_filter_is_advertised(jax_ctx):
    with jax_ctx:
        assert "SPAtmplt" in filter_approximants()


def test_lal_sgburst_fallback_is_jax_backed(jax_ctx):
    with jax_ctx:
        hp, hc = get_sgburst_waveform(
            q=8,
            frequency=150,
            hrss=1e-21,
            delta_t=1 / 4096,
        )

    assert backend_array(hp, "jax") is not None
    assert backend_array(hc, "jax") is not None
    assert np.isfinite(hp.numpy()).all()
    assert np.isfinite(hc.numpy()).all()


@pytest.mark.parametrize("domain", ("fd", "td"))
def test_mode_availability_preserves_lal_fallbacks(jax_ctx, domain):
    available = getattr(waveform_modes, f"{domain}_waveform_mode_approximants")
    assert available(jax_ctx) == available(scheme.CPUScheme())
    assert available(jax_ctx)


def test_native_sequence_availability_is_jax_only(monkeypatch, jax_ctx):
    approximant = "JAXOnlySequence"
    monkeypatch.setattr(
        waveform_module,
        "native_approximants",
        lambda interface: (approximant,) if interface == "sequence" else (),
    )

    def generate(**params):
        return params["approximant"], params["sample_points"]

    generate.required = ()
    monkeypatch.setattr(waveform_module, "_lalsim_fd_sequence", generate)

    with pytest.raises(ValueError, match="not available"):
        get_fd_waveform_sequence(
            approximant=approximant,
            sample_points=np.arange(4),
        )
    with jax_ctx:
        actual_approximant, sample_points = get_fd_waveform_sequence(
            approximant=approximant,
            sample_points=np.arange(4),
        )

    assert actual_approximant == approximant
    assert backend_array(sample_points, "jax") is not None


def test_sum_modes_rejects_different_frequency_grids(jax_ctx):
    with jax_ctx:
        modes = {
            (2, 2): FrequencySeries(np.ones(4, dtype=complex), delta_f=0.25),
            (2, 1): FrequencySeries(np.ones(4, dtype=complex), delta_f=0.5),
        }
        with pytest.raises(ValueError, match="different delta_f"):
            sum_modes(modes, inclination=0.7, phi=0.2)
