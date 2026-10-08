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

import os
import subprocess
import sys

import numpy as np
import pytest

try:
    import jax
except ImportError:
    pytest.skip("JAX is unavailable", allow_module_level=True)

from pycbc import scheme
from pycbc.events import threshold, threshold_only
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
from pycbc.filter.matchedfilter import Correlator
from pycbc.filter import matchedfilter_jax
from pycbc.filter.autocorrelation import calculate_acf, calculate_acl
from pycbc.filter.matchedfilter_jax import (
    correlate_jax,
    match_jax,
    matched_filter_core_jax,
    matched_filter_jax,
    overlap_jax,
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

jax.config.update("jax_enable_x64", True)
jnp = jax.numpy


_SCHEME_SPEC = os.environ.get("PYCBC_TEST_SCHEME", "jax:cpu")
if _SCHEME_SPEC != "jax" and not _SCHEME_SPEC.startswith("jax:"):
    pytest.skip("filter tests require a JAX PYCBC_TEST_SCHEME",
                allow_module_level=True)
_DEVICE = _SCHEME_SPEC.split(":", 1)[1] if ":" in _SCHEME_SPEC else "cpu"


def _jax_context(**kwargs):
    return scheme.JAXScheme(_DEVICE, **kwargs)


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

    with _jax_context():
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

    with _jax_context():
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
    with _jax_context():
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
    with _jax_context():
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
    with _jax_context():
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

    with _jax_context():
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

    with _jax_context():
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

    with _jax_context():
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

    with _jax_context():
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


def test_functional_helpers_place_mixed_inputs_on_one_device():
    """Exercise retained device storage after exiting a scheme on any host."""
    env = os.environ.copy()
    env.pop("PYCBC_SCHEME", None)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    result = subprocess.run([sys.executable, "-c", """
import jax
import numpy as np
from pycbc import scheme
from pycbc.filter import matchedfilter_jax as mf
from pycbc.types import FrequencySeries

jax.config.update('jax_enable_x64', True)
devices = jax.devices('cpu')
rng = np.random.default_rng(987)
h = rng.normal(size=33) + 1j * rng.normal(size=33)
s = rng.normal(size=33) + 1j * rng.normal(size=33)
p = rng.uniform(.5, 2., size=33)
n1, n2 = mf.sigmasq_jax(h, p), mf.sigmasq_jax(s, p)

def evaluate(h, s, p, supplied=False):
    first = n1 if supplied else None
    second = n2 if supplied else None
    return (
        mf.sigmasq_jax(h, p), mf.sigmasq_series_jax(h, p),
        *mf.matched_filter_core_jax(h, s, p, h_norm=first),
        mf.matched_filter_jax(h, s, p, sigmasq=first),
        mf.overlap_jax(h, s, p),
        *mf.match_jax(h, s, p, v1_norm=first, v2_norm=second),
    )

def check(actual, expected, device):
    for value, reference in zip(actual, expected):
        if isinstance(value, jax.Array):
            assert value.devices() == {device}
            np.testing.assert_allclose(value, reference, rtol=2e-14,
                                       atol=2e-14)
        else:
            assert value == reference

with scheme.JAXScheme('1'):
    retained = FrequencySeries(p, delta_f=1.)
assert retained._data.device == devices[1]
for supplied in (False, True):
    expected = evaluate(h, s, p, supplied)
    # Host primary inputs inherit the retained PSD's device outside a scheme.
    check(evaluate(h, s, retained, supplied), expected, devices[1])
    # Primary input storage takes precedence over other input devices.
    first = jax.device_put(h, devices[0])
    second = jax.device_put(s, devices[1])
    check(evaluate(first, second, retained, supplied), expected, devices[0])
    # An active scheme explicitly overrides input devices.
    with scheme.JAXScheme('1'):
        check(evaluate(first, second, retained, supplied), expected,
              devices[1])

with jax.default_device(devices[1]):
    check(evaluate(h, s, p), evaluate(h, s, retained), devices[1])
with scheme.JAXScheme('1'):
    compiled = jax.jit(lambda h, s, p: mf.matched_filter_core_jax(h, s, p))
    values = compiled(jax.device_put(h, devices[1]),
                      jax.device_put(s, devices[1]),
                      jax.device_put(p, devices[1]))
    assert all(value.devices() == {devices[1]} for value in values)
print('device-ok')
"""], env=env, capture_output=True, check=True, text=True, timeout=45)
    assert result.stdout.strip() == "device-ok"


@pytest.fixture
def deterministic_original_fft(monkeypatch):
    # FFTW aligned/unaligned plans can round differently after independent
    # CPU allocations. Fix the original provider for this byte comparison.
    from pycbc.fft import backend_cpu

    monkeypatch.setattr(backend_cpu, "cpu_backend", "numpy")


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("engine", [False, True])
def test_native_correlate_retains_original_bytes_and_views(dtype, engine):
    rng = np.random.default_rng(4321)
    x = (rng.normal(size=17) + 1j * rng.normal(size=17)).astype(dtype)
    y = (rng.normal(size=17) + 1j * rng.normal(size=17)).astype(dtype)
    with scheme.CPUScheme():
        expected = zeros(23, dtype=dtype)
        expected[0] = 7
        expected[-1] = 8
        correlate(Array(x), Array(y), expected[3:20])
        expected_values = expected.numpy().copy()
    with _jax_context(reference_operations=("correlate",)) as ctx:
        actual = zeros(23, dtype=dtype)
        actual[0] = 7
        actual[-1] = 8
        view = actual[3:20]
        storage = view._data
        if engine:
            Correlator(Array(x), Array(y), view).correlate()
        else:
            correlate(Array(x), Array(y), view)
        assert scheme.mgr.state is ctx
        assert view._data is storage
        assert actual._data.device == ctx.jax_device
        assert actual.numpy().tobytes() == expected_values.tobytes()


def test_native_batch_correlate_preserves_sibling_output_tails():
    rng = np.random.default_rng(234)
    rows, size, stride = 3, 17, 24
    xs = [(rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        np.complex64) for _ in range(rows)]
    y = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        np.complex64)
    with scheme.CPUScheme():
        expected = Array(np.full(rows * stride, 3j, dtype=np.complex64))
        destinations = [expected[r * stride:(r + 1) * stride]
                        for r in range(rows)]
        BatchCorrelator([Array(x) for x in xs], destinations, size).execute(
            Array(y))
        expected_values = expected.numpy().copy()
    with _jax_context(reference_operations=("correlate",)):
        actual = Array(np.full(rows * stride, 3j, dtype=np.complex64))
        destinations = [actual[r * stride:(r + 1) * stride]
                        for r in range(rows)]
        BatchCorrelator([Array(x) for x in xs], destinations, size).execute(
            Array(y))
        assert actual.numpy().tobytes() == expected_values.tobytes()


def test_correlate_default_remains_on_device(monkeypatch):
    def reject_cpu(*args, **kwargs):
        raise AssertionError("default correlation used native CPU validation")
    monkeypatch.setattr(matchedfilter_jax, "_cpu_correlate", reject_cpu)
    with _jax_context() as ctx:
        output = zeros(4, dtype=np.complex64)
        correlate(Array(np.ones(4, dtype=np.complex64)),
                  Array(np.ones(4, dtype=np.complex64)), output)
        assert output._data.device == ctx.jax_device
        np.testing.assert_array_equal(output.numpy(), np.ones(4))


def test_correlation_output_write_keeps_numpy_storage_and_errors():
    output = np.zeros(3, dtype=np.complex64)
    matchedfilter_jax._set_output_array(output, jnp.ones(3, jnp.complex64))
    np.testing.assert_array_equal(output, np.ones(3))

    class Unwritable:
        def __setitem__(self, key, value):
            raise RuntimeError("unwritable output")
    with pytest.raises(RuntimeError, match="unwritable output"):
        matchedfilter_jax._set_output_array(Unwritable(), jnp.ones(3))


@pytest.mark.parametrize("calculate", [
    sigmasq_jax, sigmasq_series_jax, matched_filter_core_jax, overlap_jax,
    match_jax])
@pytest.mark.parametrize("cutoffs", [(-1., None), (6., 4.)])
def test_direct_helpers_keep_original_cutoff_validation(calculate, cutoffs):
    data = jnp.ones(17, dtype=jnp.complex128)
    args = (data,) if calculate in (sigmasq_jax, sigmasq_series_jax) else (
        data, data)
    with pytest.raises(ValueError):
        calculate(*args, low_frequency_cutoff=cutoffs[0],
                  high_frequency_cutoff=cutoffs[1])


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("weighted", [False, True])
def test_direct_sigmasq_honors_native_inner_selection(dtype, weighted):
    data = (np.random.default_rng(765).normal(size=65) + 2j).astype(dtype)
    with scheme.CPUScheme():
        series = FrequencySeries(data, delta_f=.5)
        real_dtype = np.float32 if dtype == np.complex64 else np.float64
        psd = (FrequencySeries(np.full(65, 2., dtype=real_dtype), delta_f=.5)
               if weighted else None)
        expected = sigmasq(series, psd=psd)
    operation = "weighted_inner" if weighted else "inner"
    with _jax_context(reference_operations=(operation,)):
        actual = sigmasq_jax(series, psd=psd, delta_f=.5)
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_direct_sigmasq_series_honors_native_array_selections(dtype):
    data = (np.random.default_rng(765).normal(size=65) + 2j).astype(dtype)
    with scheme.CPUScheme():
        series = FrequencySeries(data, delta_f=.5)
        expected = sigmasq_series(series).numpy().copy()
    with _jax_context(reference_operations=("squared_norm", "cumsum")):
        actual = sigmasq_series_jax(series, delta_f=.5)
    assert np.asarray(actual).tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_matched_filter_granular_native_routes_restore_original_bytes(dtype):
    rng = np.random.default_rng(334)
    h = (rng.normal(size=65) + 1j * rng.normal(size=65)).astype(dtype)
    s = (rng.normal(size=65) + 1j * rng.normal(size=65)).astype(dtype)
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    p = rng.uniform(.5, 2., size=65).astype(real_dtype)
    with scheme.CPUScheme():
        template = FrequencySeries(h, delta_f=.5, epoch=123.)
        data = FrequencySeries(s, delta_f=.5, epoch=123.)
        psd = FrequencySeries(p, delta_f=.5)
        expected_snr, expected_corr, expected_norm = matched_filter_core(
            template, data, psd=psd)
        expected_snr_values = expected_snr.numpy().copy()
        expected_corr_values = expected_corr.numpy().copy()
    operations = ("correlate", "divide", "ifft", "weighted_inner")
    with _jax_context(reference_operations=operations) as ctx:
        snr, corr, norm = matched_filter_core(template, data, psd=psd)
        assert scheme.mgr.state is ctx
        assert snr._data.device == corr._data.device == ctx.jax_device
        assert snr.numpy().tobytes() == expected_snr_values.tobytes()
        assert corr.numpy().tobytes() == expected_corr_values.tobytes()
        assert (np.asarray(norm).tobytes()
                == np.asarray(expected_norm).tobytes())
        assert snr.delta_t == expected_snr.delta_t
        assert corr.delta_f == expected_corr.delta_f
        assert snr._epoch == expected_snr._epoch
        assert corr._epoch == expected_corr._epoch

        raw_snr, raw_corr, raw_norm = matched_filter_core_jax(
            template, data, psd=psd)
        assert np.asarray(raw_snr).tobytes() == expected_snr_values.tobytes()
        assert np.asarray(raw_corr).tobytes() == expected_corr_values.tobytes()
        assert (np.asarray(raw_norm).tobytes()
                == np.asarray(expected_norm).tobytes())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("unbiased", [False, True])
@pytest.mark.parametrize("as_series", [False, True])
def test_acf_preserves_original_precision_scaling_and_metadata(
        monkeypatch, dtype, unbiased, as_series):
    from pycbc import reference_jax
    values = np.random.default_rng(123).normal(size=127).astype(dtype)
    data = (TimeSeries(values, delta_t=.125, epoch=123.)
            if as_series else values)
    with scheme.CPUScheme():
        expected = calculate_acf(data, delta_t=.125, unbiased=unbiased)

    def reject_cpu(*args, **kwargs):
        raise AssertionError("default autocorrelation used the CPU worker")
    monkeypatch.setattr(reference_jax, "cpu_reference", reject_cpu)
    with _jax_context() as ctx:
        actual = calculate_acf(data, delta_t=.125, unbiased=unbiased)
        assert actual.dtype == expected.dtype == np.float64
        assert actual.delta_t == expected.delta_t
        assert actual._epoch == expected._epoch
        assert actual._data.device == ctx.jax_device
        tolerance = 2e-7 if dtype == np.float32 and unbiased else 1e-12
        np.testing.assert_allclose(actual.numpy(), expected.numpy(),
                                   rtol=tolerance, atol=1e-14)


def test_direct_acf_helper_keeps_cpu_output_outside_jax_scheme():
    from pycbc.filter.autocorrelation_jax import calculate_acf_jax
    values = np.random.default_rng(123).normal(size=127)
    with scheme.CPUScheme():
        expected = calculate_acf(values, delta_t=.125, unbiased=True)
        actual = calculate_acf_jax(values, .125, True)
        assert actual.dtype == expected.dtype
        assert actual.delta_t == expected.delta_t
        assert is_backend(actual, "numpy")
        np.testing.assert_allclose(actual.numpy(), expected.numpy(),
                                   rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("operation", ["autocorrelation_mean",
                                       "autocorrelation_variance"])
def test_acf_scalar_reductions_select_original_numpy_independently(
        monkeypatch, operation):
    from pycbc.filter import autocorrelation_jax
    from pycbc import reference_jax
    values = np.random.default_rng(123).normal(size=127).astype(np.float32)
    with scheme.CPUScheme():
        expected = calculate_acf(TimeSeries(values, delta_t=.125),
                                 unbiased=True)

    def reject(*args, **kwargs):
        raise AssertionError("selected native reduction used a JAX reduction")
    selected = "mean" if operation == "autocorrelation_mean" else "var"
    monkeypatch.setattr(autocorrelation_jax.jnp, selected, reject)
    monkeypatch.setattr(reference_jax, "cpu_reference", reject)
    with _jax_context(reference_operations=(operation,)) as ctx:
        result = calculate_acf(TimeSeries(values, delta_t=.125), unbiased=True)
        assert result._data.device == ctx.jax_device
        tolerance = 2e-7 if selected == "mean" else 1e-12
        np.testing.assert_allclose(result.numpy(), expected.numpy(),
                                   rtol=tolerance, atol=1e-14)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("unbiased", [False, True])
def test_acf_granular_controls_restore_original_bytes(
        monkeypatch, dtype, unbiased, deterministic_original_fft):
    from pycbc import reference_jax
    values = np.random.default_rng(123).normal(size=127).astype(dtype)
    with scheme.CPUScheme():
        expected = calculate_acf(TimeSeries(values, delta_t=.125),
                                 unbiased=unbiased)

    def reject_worker(*args, **kwargs):
        raise AssertionError("granular ACF controls used the whole CPU worker")
    monkeypatch.setattr(reference_jax, "cpu_reference", reject_worker)
    operations = ("autocorrelation_mean", "autocorrelation_variance", "fft",
                  "ifft", "correlate", "divide")
    with _jax_context(reference_operations=operations) as ctx:
        actual = calculate_acf(TimeSeries(values, delta_t=.125),
                               unbiased=unbiased)
        assert actual._data.device == ctx.jax_device
        assert actual.numpy().tobytes() == expected.numpy().tobytes()
        assert actual.delta_t == expected.delta_t
        assert actual._epoch == expected._epoch


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("unbiased", [False, True])
@pytest.mark.parametrize("as_series", [False, True])
def test_native_acf_acl_preserve_original_values(
        dtype, unbiased, as_series, deterministic_original_fft):
    values = np.random.default_rng(123).normal(size=127).astype(dtype)
    data = (TimeSeries(values, delta_t=.125, epoch=123.)
            if as_series else values)
    with scheme.CPUScheme():
        expected = calculate_acf(data, delta_t=.125, unbiased=unbiased)
        expected_acl = calculate_acl(data, dtype=float)
    with _jax_context(reference_operations=("autocorrelation",)) as ctx:
        actual = calculate_acf(data, delta_t=.125, unbiased=unbiased)
        actual_acl = calculate_acl(data, dtype=float)
        assert scheme.mgr.state is ctx
        assert actual._data.device == ctx.jax_device
        assert actual.numpy().tobytes() == expected.numpy().tobytes()
        assert actual.delta_t == expected.delta_t
        assert actual._epoch == expected._epoch
        assert (np.asarray(actual_acl).tobytes()
                == np.asarray(expected_acl).tobytes())


def test_cluster_rounds_threshold_before_squaring():
    threshold_value = 1.00000007
    data = TimeSeries(np.array([np.float32(threshold_value) + 0j],
                               dtype=np.complex64), delta_t=1.)
    expected = CPUThresholdCluster(data).threshold_and_cluster(
        threshold_value, 1)
    with _jax_context():
        actual = jax_threshold_cluster(data, threshold_value, 1)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_native_cluster_handles_fused_magnitude_threshold_boundary():
    data = TimeSeries(np.array([-.7931225 - .13891791j], np.complex64),
                      delta_t=1.)
    threshold_value = .8051965236663818
    expected = CPUThresholdCluster(data).threshold_and_cluster(
        threshold_value, 1)
    np.testing.assert_array_equal(expected[1], [0])
    with _jax_context(reference_operations=("threshold_cluster",)) as ctx:
        values, indices = jax_threshold_cluster(data, threshold_value, 1)
        assert values.devices() == indices.devices() == {ctx.jax_device}
        assert np.asarray(values).tobytes() == expected[0].tobytes()
        assert np.asarray(indices).tobytes() == expected[1].tobytes()
