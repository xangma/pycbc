# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""JAX PSD generation, estimation, and interpolation test suite."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.psd import (
    from_string,
    interpolate,
    inverse_spectrum_truncation,
    welch,
)
from pycbc.psd.analytical_jax import (
    ALIGO_JAX_ANALYTICAL_MODELS,
    GROUND_FIT_JAX_ANALYTICAL_MODELS,
    ILIGO_JAX_ANALYTICAL_MODELS,
    analytical_psd,
    analytical_psd_jax,
    get_jax_psd_list,
)
from pycbc.psd.estimate_jax import (
    interpolate_jax,
    inverse_spectrum_truncation_jax,
    welch_jax,
)
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.types.array_jax import JAXArrayData, to_jax
from pycbc.types.backend import is_backend

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy


@pytest.fixture(autouse=True)
def clean_scheme_state():
    """Ensure clean scheme manager state before and after each test."""
    yield
    scheme.Scheme._single = None


def _relative_diff(a, b, mask=None):
    """Compute maximum relative error between two arrays on positive bins."""
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    if mask is None:
        mask = (arr_b > 0.0) & np.isfinite(arr_b) & np.isfinite(arr_a)
    if not np.any(mask):
        return 0.0
    return np.max(np.abs(arr_a[mask] - arr_b[mask]) / arr_b[mask])


@pytest.mark.parametrize(
    "model",
    sorted(
        list(
            ALIGO_JAX_ANALYTICAL_MODELS
            | GROUND_FIT_JAX_ANALYTICAL_MODELS
            | ILIGO_JAX_ANALYTICAL_MODELS
            | {"flat_unity"}
        )
    ),
)
def test_analytical_psd_matches_lal(model):
    """Verify JAX analytical PSD matches LALSimulation / CPU PyCBC to < 2e-9.
    """
    length = 2048
    delta_f = 0.5
    flow = 15.0

    cpu_psd = from_string(model, length, delta_f, flow)
    with scheme.JAXScheme():
        jax_psd = from_string(model, length, delta_f, flow)
        assert isinstance(jax_psd, FrequencySeries)
        assert is_backend(jax_psd, "jax")
        assert isinstance(jax_psd._data, JAXArrayData)

    rel_err = _relative_diff(jax_psd.numpy(), cpu_psd.numpy())
    assert rel_err < 2e-9, f"Model {model} failed with rel_err {rel_err}"


@pytest.mark.parametrize(
    "model",
    [
        "aLIGOEarlyLowSensitivityP1200087",
        "aLIGODesignSensitivityP1200087",
        "CosmicExplorerP1600143",
        "EinsteinTelescopeP1600143",
    ],
)
def test_data_file_psd_matches_lal(model):
    """Verify JAX data-file log-linear interpolation matches LAL to < 1e-10."""
    length = 2048
    delta_f = 0.5
    flow = 10.0

    cpu_psd = from_string(model, length, delta_f, flow)
    with scheme.JAXScheme():
        jax_psd = from_string(model, length, delta_f, flow)
        assert is_backend(jax_psd, "jax")

    rel_err = _relative_diff(jax_psd.numpy(), cpu_psd.numpy())
    assert rel_err < 1e-10, f"Data file {model} failed with rel_err {rel_err}"


def test_get_jax_psd_list():
    """Verify list of supported JAX PSD models."""
    models = get_jax_psd_list()
    assert len(models) >= 80
    assert "aLIGOZeroDetHighPower" in models
    assert "Virgo" in models
    assert "CosmicExplorerP1600143" in models
    assert "flat_unity" in models


def test_analytical_psd_jax_jit():
    """Verify analytical_psd_jax is JIT-compilable with @jax.jit."""
    frequencies = jnp.linspace(10.0, 1000.0, 500)

    @jax.jit
    def jitted_psd(f):
        return analytical_psd_jax("aLIGOZeroDetHighPower", f, 10.0)

    val = jitted_psd(frequencies)
    assert isinstance(val, jax.Array)
    assert val.shape == (500,)
    assert jnp.all(val > 0.0)

    # Test differentiability with jax.grad
    grad_fn = jax.jit(
        jax.grad(
            lambda f: jnp.sum(
                analytical_psd_jax("aLIGOZeroDetHighPower", f, 10.0)
            )
        )
    )
    grads = grad_fn(frequencies)
    assert isinstance(grads, jax.Array)
    assert grads.shape == (500,)
    assert jnp.all(jnp.isfinite(grads))


@pytest.mark.parametrize("avg_method", ["median", "mean", "median-mean"])
def test_welch_jax_matches_cpu(avg_method):
    """Verify Welch estimation in JAX matches CPU PyCBC to < 1e-11."""
    np.random.seed(1234)
    data = np.random.normal(0, 1, 16384)
    dt = 1.0 / 4096
    ts = TimeSeries(data, delta_t=dt)

    seg_len = 2048
    seg_stride = 1024

    cpu_psd = welch(
        ts,
        seg_len=seg_len,
        seg_stride=seg_stride,
        avg_method=avg_method,
    )
    with scheme.JAXScheme():
        jax_psd = welch(
            ts,
            seg_len=seg_len,
            seg_stride=seg_stride,
            avg_method=avg_method,
        )
        assert isinstance(jax_psd, FrequencySeries)
        assert is_backend(jax_psd, "jax")

    rel_err = _relative_diff(jax_psd.numpy(), cpu_psd.numpy())
    assert (
        rel_err < 1e-11
    ), f"Welch avg_method={avg_method} rel_err={rel_err} exceeds 1e-11"


@pytest.mark.parametrize("trunc_method", ["hann", None])
def test_inverse_spectrum_truncation_jax_matches_cpu(trunc_method):
    """Verify inverse spectrum truncation matches CPU PyCBC to < 1e-11."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 16384)
    dt = 1.0 / 4096
    ts = TimeSeries(data, delta_t=dt)
    psd = welch(ts, seg_len=2048, seg_stride=1024)

    cpu_trunc = inverse_spectrum_truncation(
        psd,
        max_filter_len=512,
        low_frequency_cutoff=20.0,
        trunc_method=trunc_method,
    )
    with scheme.JAXScheme():
        jax_trunc = inverse_spectrum_truncation(
            psd,
            max_filter_len=512,
            low_frequency_cutoff=20.0,
            trunc_method=trunc_method,
        )
        assert isinstance(jax_trunc, FrequencySeries)
        assert is_backend(jax_trunc, "jax")

    rel_err = _relative_diff(jax_trunc.numpy(), cpu_trunc.numpy())
    assert (
        rel_err < 1e-11
    ), f"Trunc method={trunc_method} rel_err={rel_err} exceeds 1e-11"


def test_interpolate_jax_matches_cpu():
    """Verify JAX interpolation matches CPU PyCBC."""
    vals = np.exp(-np.linspace(0, 5, 1025))
    fs = FrequencySeries(vals, delta_f=0.5)

    cpu_interp = interpolate(fs, delta_f=0.25, length=2049)
    with scheme.JAXScheme():
        jax_interp = interpolate(fs, delta_f=0.25, length=2049)
        assert isinstance(jax_interp, FrequencySeries)
        assert is_backend(jax_interp, "jax")

    assert np.allclose(jax_interp.numpy(), cpu_interp.numpy(), atol=1e-14)


def test_jax_scheme_pipeline():
    """Verify full PSD generation, estimation, and interpolation under
    JAXScheme.
    """
    with scheme.JAXScheme():
        # Analytical generation
        psd = from_string("aLIGOZeroDetHighPower", 2048, 0.5, 15.0)
        assert is_backend(psd, "jax")
        assert isinstance(psd._data, JAXArrayData)

        # Zero-copy extraction
        jarr = to_jax(psd)
        assert isinstance(jarr, jax.Array)
        assert jarr is psd._data.array

        # Flat unity
        flat = from_string("flat_unity", 2048, 0.5, 20.0)
        assert is_backend(flat, "jax")
        assert flat[10] == 0.0
        assert flat[100] == 1.0

        # Welch under JAXScheme
        np.random.seed(999)
        ts = TimeSeries(np.random.normal(0, 1, 8192), delta_t=1.0 / 2048)
        psd_est = welch(ts, seg_len=1024, seg_stride=512)
        assert is_backend(psd_est, "jax")

        # Inverse spectrum truncation under JAXScheme
        trunc = inverse_spectrum_truncation(
            psd_est, max_filter_len=256, low_frequency_cutoff=30.0
        )
        assert is_backend(trunc, "jax")

        # Interpolate under JAXScheme
        interp = interpolate(trunc, delta_f=0.25)
        assert is_backend(interp, "jax")
        assert interp.delta_f == 0.25


def test_direct_jax_estimation_api():
    """Verify direct calls to analytical_psd, welch_jax, inv_trunc_jax,
    and interpolate_jax.
    """
    with scheme.JAXScheme():
        psd1 = analytical_psd("aLIGOZeroDetHighPower", 1024, 1.0, 20.0)
        assert is_backend(psd1, "jax")

        np.random.seed(42)
        ts = TimeSeries(np.random.normal(0, 1, 4096), delta_t=1.0 / 2048)
        psd2 = welch_jax(ts, seg_len=512, seg_stride=256)
        assert is_backend(psd2, "jax")

        trunc = inverse_spectrum_truncation_jax(psd2, max_filter_len=128)
        assert is_backend(trunc, "jax")

        interp = interpolate_jax(trunc, delta_f=0.5)
        assert is_backend(interp, "jax")
