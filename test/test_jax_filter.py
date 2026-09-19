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

"""Unit test suite for JAX matched filtering, clustering, and vetoes."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.events import threshold, threshold_only
from pycbc.events.eventmgr import ThresholdCluster
from pycbc.events.threshold_cpu import CPUThresholdCluster
from pycbc.events.threshold_jax import (
    JAXThresholdCluster,
    threshold_and_cluster as jax_threshold_cluster,
)
from pycbc.filter import (
    correlate,
    make_frequency_series,
    match,
    matched_filter,
    matched_filter_core,
    sigmasq,
    sigmasq_series,
)
from pycbc.filter.matchedfilter import BatchCorrelator
from pycbc.filter.autocorrelation import calculate_acf, calculate_acl
from pycbc.filter.matchedfilter_jax import (
    correlate_jax,
    match_jax,
    matched_filter_core_jax,
    matched_filter_jax,
    sigmasq_jax,
    sigmasq_series_jax,
)
from pycbc.types import Array, FrequencySeries, TimeSeries, zeros
from pycbc.types.array_jax import JAXArrayData
from pycbc.types.backend import is_backend
from pycbc.vetoes import (
    chisq_accum_bin,
    power_chisq_at_points_from_precomputed,
)

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy


@pytest.fixture(autouse=True)
def clean_scheme_state():
    """Ensure clean scheme manager state before and after each test."""
    yield
    scheme.Scheme._single = None


def test_correlate_parity():
    """Verify elementwise conjugate correlation matches CPU."""
    n = 1024
    np.random.seed(10)
    x_data = np.random.randn(n) + 1j * np.random.randn(n)
    y_data = np.random.randn(n) + 1j * np.random.randn(n)

    x_cpu = Array(x_data, dtype=np.complex64)
    y_cpu = Array(y_data, dtype=np.complex64)
    z_cpu = zeros(n, dtype=np.complex64)
    correlate(x_cpu, y_cpu, z_cpu)

    with scheme.JAXScheme():
        x_jax = Array(x_data, dtype=np.complex64)
        y_jax = Array(y_data, dtype=np.complex64)
        z_jax = zeros(n, dtype=np.complex64)
        correlate(x_jax, y_jax, z_jax)

        assert is_backend(z_jax, "jax")
        assert isinstance(z_jax._data, JAXArrayData)

    diff = np.max(np.abs(z_cpu.numpy() - z_jax.numpy()))
    assert diff < 1e-6

    direct_c = correlate_jax(x_data, y_data)
    np.testing.assert_allclose(z_cpu.numpy(), np.asarray(direct_c), rtol=1e-6)


def test_batch_correlator():
    """Verify BatchCorrelator under JAXScheme."""
    n = 512
    n_batches = 4
    np.random.seed(20)
    xs_data = [
        np.random.randn(n) + 1j * np.random.randn(n) for _ in range(n_batches)
    ]
    y_data = np.random.randn(n) + 1j * np.random.randn(n)

    # CPU run
    xs_cpu = [Array(x, dtype=np.complex64) for x in xs_data]
    zs_cpu = [zeros(n, dtype=np.complex64) for _ in range(n_batches)]
    y_cpu = Array(y_data, dtype=np.complex64)
    batch_cpu = BatchCorrelator(xs_cpu, zs_cpu, n)
    batch_cpu.batch_correlate_execute(y_cpu)

    with scheme.JAXScheme():
        xs_jax = [Array(x, dtype=np.complex64) for x in xs_data]
        zs_jax = [zeros(n, dtype=np.complex64) for _ in range(n_batches)]
        y_jax = Array(y_data, dtype=np.complex64)
        batch_jax = BatchCorrelator(xs_jax, zs_jax, n)
        batch_jax.batch_correlate_execute(y_jax)

    for z_c, z_j in zip(zs_cpu, zs_jax):
        diff = np.max(np.abs(z_c.numpy() - z_j.numpy()))
        assert diff < 1e-6


def test_sigmasq_and_series_parity():
    """Verify sigmasq and cumulative power series match CPU to < 1e-12."""
    np.random.seed(30)
    dt = 1.0 / 2048
    data = np.sin(np.linspace(0, 50, 4096)) + 0.1 * np.random.randn(4096)
    ts = TimeSeries(data, delta_t=dt, dtype=np.float64)
    fs = make_frequency_series(ts)
    psd = FrequencySeries(
        np.full(len(fs), 2.0, dtype=np.float64), delta_f=fs.delta_f
    )

    flow = 20.0
    fhigh = 500.0

    # Test CPU
    cpu_s = sigmasq(
        fs,
        psd=psd,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
    )
    cpu_vec = sigmasq_series(
        fs,
        psd=psd,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
    )

    # Test JAXScheme
    with scheme.JAXScheme():
        jax_s = sigmasq(
            fs,
            psd=psd,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
        )
        jax_vec = sigmasq_series(
            fs,
            psd=psd,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
        )

    assert abs(cpu_s - jax_s) / cpu_s < 1e-12
    vec_diff = np.max(np.abs(cpu_vec.numpy() - jax_vec.numpy()))
    assert vec_diff < 1e-12

    # Test pure JAX function
    direct_s = sigmasq_jax(
        fs,
        psd=psd,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
        delta_f=fs.delta_f,
    )
    assert abs(cpu_s - direct_s) / cpu_s < 1e-12

    direct_vec = sigmasq_series_jax(
        fs,
        psd=psd,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
        delta_f=fs.delta_f,
    )
    assert np.max(np.abs(cpu_vec.numpy() - np.asarray(direct_vec))) < 1e-12


def test_matched_filter_core_and_snr_parity():
    """Verify matched_filter_core and matched_filter match CPU to < 1e-10."""
    np.random.seed(40)
    dt = 1.0 / 4096
    n = 8192
    time = np.linspace(0, 2.0, n)
    sig = np.sin(2.0 * np.pi * 60.0 * time) * np.exp(-time * 2.0)
    template = TimeSeries(sig, delta_t=dt, dtype=np.float64)
    strain = TimeSeries(
        sig + 0.05 * np.random.randn(n), delta_t=dt, dtype=np.float64
    )

    fs_temp = make_frequency_series(template)
    psd = FrequencySeries(
        np.full(len(fs_temp), 1.5, dtype=np.float64),
        delta_f=fs_temp.delta_f,
    )

    flow = 30.0
    fhigh = 800.0

    # CPU matched filter
    cpu_snr_unnorm, cpu_corr, cpu_norm = matched_filter_core(
        template,
        strain,
        psd=psd,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
    )
    cpu_snr = matched_filter(
        template,
        strain,
        psd=psd,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
    )

    # JAXScheme matched filter
    with scheme.JAXScheme():
        jax_snr_unnorm, jax_corr, jax_norm = matched_filter_core(
            template,
            strain,
            psd=psd,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
        )
        jax_snr = matched_filter(
            template,
            strain,
            psd=psd,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
        )

        assert is_backend(jax_snr, "jax")

    assert abs(cpu_norm - jax_norm) / cpu_norm < 1e-12
    snr_diff = np.max(np.abs(cpu_snr.numpy() - jax_snr.numpy()))
    assert snr_diff < 1e-10

    # Direct functional API
    j_snr, j_corr, j_norm = matched_filter_core_jax(
        template,
        strain,
        psd=psd,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
    )
    assert abs(cpu_norm - j_norm) / cpu_norm < 1e-12
    assert np.max(np.abs(cpu_snr_unnorm.numpy() - np.asarray(j_snr))) < 1e-10

    direct_snr = matched_filter_jax(
        template,
        strain,
        psd=psd,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
    )
    assert np.max(np.abs(cpu_snr.numpy() - np.asarray(direct_snr))) < 1e-10


def test_match_parity():
    """Verify waveform match under JAXScheme matches CPU to < 1e-12."""
    dt = 1.0 / 4096
    n = 4096
    t = np.linspace(0, 1.0, n)
    v1 = TimeSeries(
        np.sin(2.0 * np.pi * 100.0 * t), delta_t=dt, dtype=np.float64
    )
    v2 = TimeSeries(
        np.roll(v1.numpy(), 20), delta_t=dt, dtype=np.float64
    )

    cpu_m, cpu_idx = match(v1, v2)
    with scheme.JAXScheme():
        jax_m, jax_idx = match(v1, v2)

    assert abs(cpu_m - jax_m) < 1e-12
    assert cpu_idx == jax_idx

    direct_m, direct_idx = match_jax(v1, v2)
    assert abs(cpu_m - direct_m) < 1e-12
    assert cpu_idx == direct_idx


def test_threshold_parity():
    """Verify threshold and threshold_only match CPU exactly."""
    np.random.seed(50)
    arr = np.random.randn(2000).astype(np.complex64)
    arr[100] = 8.0 + 2.0j
    arr[500] = -7.0 - 5.0j
    arr[1200] = 10.0 + 0.0j
    ts = TimeSeries(arr, delta_t=1.0)

    # CPU reference
    locs_cpu, vals_cpu = threshold(ts, 4.0)

    with scheme.JAXScheme():
        locs_jax, vals_jax = threshold(ts, 4.0)
        locs_only, vals_only = threshold_only(ts, 4.0)

    np.testing.assert_array_equal(locs_cpu, locs_jax)
    np.testing.assert_allclose(vals_cpu, vals_jax)
    np.testing.assert_array_equal(locs_jax, locs_only)


def test_threshold_and_cluster_parity():
    """Verify threshold_and_cluster matches CPU across multiple windows."""
    np.random.seed(60)
    arr = np.random.randn(3000).astype(np.complex64)
    # Inject prominent peaks
    arr[50] = 12.0
    arr[320] = 15.0
    arr[325] = 14.0  # Should be clustered into peak at 320
    arr[1500] = 20.0
    ts = TimeSeries(arr, delta_t=1.0)

    for window in [10, 25, 50, 100]:
        thresh = 5.0
        # CPU clustering
        cluster_cpu = CPUThresholdCluster(ts)
        cvals_cpu, clocs_cpu = cluster_cpu.threshold_and_cluster(
            thresh, window
        )

        # JAX clustering via class
        cluster_jax = JAXThresholdCluster(ts)
        cvals_jax, clocs_jax = cluster_jax.threshold_and_cluster(
            thresh, window
        )

        np.testing.assert_array_equal(clocs_cpu, clocs_jax)
        np.testing.assert_allclose(cvals_cpu, cvals_jax)

        # Via ThresholdCluster factory under JAXScheme
        with scheme.JAXScheme():
            factory_cluster = ThresholdCluster(ts)
            cvals_f, clocs_f = factory_cluster.threshold_and_cluster(
                thresh, window
            )
            np.testing.assert_array_equal(clocs_cpu, clocs_f)
            np.testing.assert_allclose(cvals_cpu, cvals_f)

        # Via functional API
        cvals_fn, clocs_fn = jax_threshold_cluster(ts, thresh, window)
        np.testing.assert_array_equal(clocs_cpu, clocs_fn)
        np.testing.assert_allclose(cvals_cpu, cvals_fn)


def test_chisq_accum_bin_parity():
    """Verify chisq_accum_bin under JAXScheme matches CPU."""
    np.random.seed(70)
    n = 2048
    q_data = np.random.randn(n).astype(np.complex64)
    q = Array(q_data, dtype=np.complex64)

    z_cpu = zeros(n, dtype=np.float32)
    chisq_accum_bin(z_cpu, q)

    with scheme.JAXScheme():
        z_jax = zeros(n, dtype=np.float32)
        chisq_accum_bin(z_jax, q)

    np.testing.assert_allclose(z_cpu.numpy(), z_jax.numpy(), rtol=1e-6)


def test_power_chisq_at_points_parity():
    """Verify power_chisq_at_points_from_precomputed matches CPU to < 1e-7."""
    n = 2048
    np.random.seed(80)
    corr_data = np.random.randn(n) + 1j * np.random.randn(n)
    corr = FrequencySeries(corr_data, delta_f=1.0)
    bins = np.array([10, 50, 150, 400, 800, 1000], dtype=np.uint32)
    points = np.array([2, 25, 100, 350, 750, 1500], dtype=np.uint32)
    snr = np.random.randn(len(points)) + 1j * np.random.randn(len(points))
    snr_norm = 0.75

    cpu_chisq = power_chisq_at_points_from_precomputed(
        corr, snr, snr_norm, bins, points
    )

    with scheme.JAXScheme():
        jax_chisq = power_chisq_at_points_from_precomputed(
            corr, snr, snr_norm, bins, points
        )

    rel_diff = np.max(
        np.abs(cpu_chisq - jax_chisq) / np.abs(cpu_chisq)
    )
    assert rel_diff < 1e-7


def test_autocorrelation_acf_and_acl():
    """Verify ACF and ACL calculations in JAX match CPU to < 1e-12."""
    np.random.seed(90)
    data = np.random.randn(2048)
    ts = TimeSeries(data, delta_t=1.0 / 2048)

    cpu_acf = calculate_acf(ts)
    cpu_acl = calculate_acl(ts)

    with scheme.JAXScheme():
        jax_acf = calculate_acf(ts)
        jax_acl = calculate_acl(ts)

    diff = np.max(np.abs(cpu_acf.numpy() - jax_acf.numpy()))
    assert diff < 1e-12
    assert cpu_acl == jax_acl


def test_jax_jit_autodiff_matchedfilter():
    """Verify JIT compilation and autodiff through matched filtering."""
    n = 1024
    freqs = jnp.linspace(10.0, 500.0, n)
    htilde = jnp.exp(1j * freqs * 0.01)
    stilde = jnp.exp(1j * freqs * 0.01) + 0.1

    @jax.jit
    def jitted_matched_filter(h, s):
        snr_series, _, _ = matched_filter_core_jax(
            h, s, delta_f=1.0, delta_t=1.0 / 2048
        )
        return jnp.sum(jnp.abs(snr_series) ** 2)

    val = jitted_matched_filter(htilde, stilde)
    assert isinstance(val, jax.Array)
    assert float(val) > 0.0

    # Gradient with respect to template
    grad_fn = jax.jit(jax.grad(lambda h: jitted_matched_filter(h, stilde)))
    grads = grad_fn(htilde)
    assert isinstance(grads, jax.Array)
    assert grads.shape == htilde.shape
    assert jnp.all(jnp.isfinite(grads))
