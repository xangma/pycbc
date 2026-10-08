"""Independent science regressions for the accelerator chi-square kernels."""

import numpy as np
import pytest
from types import SimpleNamespace

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pycbc import scheme
from pycbc.types import FrequencySeries
from pycbc.vetoes import chisq_jax
from pycbc.vetoes.chisq import power_chisq_bins


@pytest.fixture(autouse=True)
def _restore_x64_config(monkeypatch):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    # These tests assert the historical Fourier/direct-phase implementation.
    # Compatibility-mode coverage lives in test_jax_chisq_modes.py.
    monkeypatch.setattr(chisq_jax, "_chisq_mode", lambda: "direct-phase")
    yield
    jax.config.update("jax_enable_x64", previous)


def _point_power_oracle(row, points, bins, kmin, ntime):
    """Direct double-precision per-bin Fourier sums (no prefix subtraction)."""
    k = np.arange(len(row), dtype=np.int64) + kmin
    angle = ((k[:, None] * points[None, :]) % ntime) * (2*np.pi/ntime)
    weighted = row[:, None].astype(np.complex128) * np.exp(1j*angle)
    sums = np.array([weighted[a:b].sum(axis=0)
                     for a, b in zip(bins[:-1], bins[1:])])
    return np.sum(abs(sums)**2, axis=0)


@pytest.mark.parametrize("core", ["single", "row", "batch"])
@pytest.mark.parametrize("coherent", [False, True])
def test_long_segment_point_power(core, coherent):
    # Products k*t exceed float32's exact integer range by four orders of
    # magnitude. Short arrays and early trigger samples hide this defect.
    ntime, kmin, nfreq = 2097152, 15360, 131072
    points = np.array([1, 1048577, 1500001, 2097151], dtype=np.int32)
    bins = np.linspace(0, nfreq, 17, dtype=np.int32)
    rng = np.random.default_rng(88412)
    row = (rng.normal(size=nfreq) + 1j*rng.normal(size=nfreq)).astype(np.complex64)
    if coherent:
        k = np.arange(nfreq, dtype=np.int64) + kmin
        row = np.exp(-2j*np.pi*((k*points[2]) % ntime)/ntime).astype(np.complex64)
        points = points[2:3]
    expected = _point_power_oracle(row, points.astype(np.int64), bins, kmin, ntime)
    arr, pts, edges = jnp.array(row), jnp.array(points), jnp.array(bins)
    if core == "single":
        got = chisq_jax._shift_sum_gpu_core(arr, float(kmin), pts, edges, float(ntime))
    elif core == "row":
        got = chisq_jax._shift_sum_gpu_core_row(
            arr[None, :], 0, float(kmin), pts, edges, float(ntime)
        )
    else:
        got = chisq_jax._batched_points_chisq_core(
            arr[None, :], jnp.zeros(len(points), dtype=jnp.int32), pts,
            edges[None, :], float(kmin), float(ntime)
        )
    np.testing.assert_allclose(got, expected, rtol=2e-6)
    if coherent:
        # A perfectly matched equal-power signal must not acquire a large
        # negative chi-square when subtracting the coherent SNR power.
        np.testing.assert_allclose((16*np.asarray(got)-nfreq**2)/nfreq, 0, atol=0.05)


def test_full_frequency_series_uses_complex_fft_length(monkeypatch):
    rng = np.random.default_rng(43)
    ntime = 4096
    row = (rng.normal(size=ntime) + 1j*rng.normal(size=ntime)).astype(np.complex64)
    bins = np.array([19, 233, 1127, 2037], dtype=np.int32)
    points = np.array([103, 1025, 3097], dtype=np.int32)
    expected = _point_power_oracle(row, points.astype(np.int64), bins, 0, ntime)
    # Exercise GPU dispatch on CPU CI as well; the kernels are device agnostic.
    corr = FrequencySeries(row, delta_f=0.25)
    got = chisq_jax.shift_sum(corr, points, bins)
    np.testing.assert_allclose(got, expected, rtol=2e-6)


@pytest.mark.parametrize("core", ["single", "row", "batch", "dispatch"])
def test_gpu_bins_outside_crop_match_zero_padded_fft(core, monkeypatch):
    rng = np.random.default_rng(563)
    ntime, kmin, nfreq = 1024, 113, 271
    row = (rng.normal(size=nfreq)+1j*rng.normal(size=nfreq)).astype(np.complex64)
    full = np.zeros(ntime, dtype=np.complex128)
    full[kmin:kmin+nfreq] = row
    bins = np.array([0, 51, 179, 305, 431, 512], dtype=np.int32)
    points = np.array([0, 359, 1023], dtype=np.int32)
    expected = np.zeros(len(points))
    for start, stop in zip(bins[:-1], bins[1:]):
        band = np.zeros(ntime, dtype=np.complex128)
        band[start:stop] = full[start:stop]
        expected += abs((np.fft.ifft(band)*ntime)[points])**2
    arr, pts, edges = jnp.array(row), jnp.array(points), jnp.array(bins-kmin)
    if core == "single":
        got = chisq_jax._shift_sum_gpu_core(arr, float(kmin), pts, edges, float(ntime))
    elif core == "row":
        got = chisq_jax._shift_sum_gpu_core_row(
            arr[None, :], 0, float(kmin), pts, edges, float(ntime)
        )
    elif core == "batch":
        got = chisq_jax._batched_points_chisq_core(
            arr[None, :], jnp.zeros(len(points), dtype=jnp.int32), pts,
            edges[None, :], float(kmin), float(ntime)
        )
    else:
        psd = object()
        tmpl = SimpleNamespace(_bin_cache={id(psd): bins})
        corr = SimpleNamespace(_kmin=kmin, _tlen=ntime)
        snrv = np.ones(len(points), dtype=np.complex64)
        results = [(None, 1., corr, points, snrv)]
        output = chisq_jax.batch_power_chisq_jax(
            arr[None, :], results, [tmpl], psd, 0
        )
        got, dof = output[0]
        # Empty bins remain part of the canonical statistic and its dof.
        expected = expected*(len(bins)-1)-1
        np.testing.assert_array_equal(dof, np.repeat(2*(len(bins)-1)-2, len(points)))
    np.testing.assert_allclose(got, expected, rtol=2e-6)


def test_long_template_bin_edges_match_standard_backend():
    nfreq, kmin = 131072, 15360
    # Long, decreasing spectra magnify serial-vs-parallel float32 scan error
    # enough to move high-frequency bin boundaries by several samples.
    power = (np.arange(kmin, kmin+nfreq, dtype=np.float64)/kmin)**(-7/3)
    h = np.zeros(kmin+nfreq+1, dtype=np.complex64)
    h[kmin:-1] = np.sqrt(power).astype(np.complex64)
    psd = FrequencySeries(np.ones(len(h), dtype=np.float32), delta_f=0.25)
    templates = [FrequencySeries(h, delta_f=0.25),
                 FrequencySeries(h*1.7, delta_f=0.25)]
    with scheme.CPUScheme():
        expected = np.array([power_chisq_bins(t, 16, psd, kmin*0.25)
                             for t in templates])
    with scheme.JAXScheme():
        scalar = chisq_jax.power_chisq_bins_jax(templates[0], 16, psd, kmin*0.25)
        batch = chisq_jax.batch_power_chisq_bins_jax(templates, 16, psd, kmin*0.25)
    np.testing.assert_array_equal(scalar, expected[0])
    np.testing.assert_array_equal(batch, expected)


def test_batch_bin_edges_match_with_per_template_lower_cutoffs():
    nfreq = 16385
    psd = FrequencySeries(
        np.linspace(0.75, 2.0, nfreq, dtype=np.float32), delta_f=0.25
    )
    base = np.linspace(0.05, 1.0, nfreq, dtype=np.float32).astype(
        np.complex64
    )
    templates = [
        FrequencySeries(base * scale, delta_f=0.25)
        for scale in (1.0, 1.7, 0.6)
    ]
    flows = [20.0, 31.0, 24.0]
    with scheme.CPUScheme():
        expected = np.asarray([
            power_chisq_bins(template, 8, psd, flow)
            for template, flow in zip(templates, flows)
        ])
    with scheme.JAXScheme():
        got = chisq_jax.batch_power_chisq_bins_jax(
            templates, 8, psd, flows
        )
    np.testing.assert_array_equal(got, expected)


def test_gpu_ordered_bin_scan_matches_numpy_exactly():
    try:
        device = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("JAX GPU backend unavailable")
    rng = np.random.default_rng(42)
    values = rng.random((3, 16385), dtype=np.float32)
    rows = jax.device_put(values, device)
    if not chisq_jax._use_gpu_ordered_scan(rows):
        pytest.skip("Pallas Triton ordered scan unavailable")
    result = chisq_jax._ordered_cumsum_rows(rows, True)
    np.testing.assert_array_equal(np.asarray(result),
                                  np.cumsum(values, axis=-1))

    psd = FrequencySeries(
        np.linspace(0.75, 2.0, 16385, dtype=np.float32), delta_f=0.25
    )
    templates = [FrequencySeries(row.astype(np.complex64), delta_f=0.25)
                 for row in values]
    flows = [20.0, 31.0, 24.0]
    with scheme.CPUScheme():
        expected = np.asarray([power_chisq_bins(t, 8, psd, flow)
                               for t, flow in zip(templates, flows)])
    with scheme.JAXScheme("cuda"):
        got = chisq_jax.batch_power_chisq_bins_jax(
            templates, 8, psd, flows
        )
    np.testing.assert_array_equal(np.asarray(got), expected)




@pytest.mark.parametrize("entry", ["single", "row", "batch", "public"])
def test_point_chisq_rejects_disabled_x64(monkeypatch, entry):
    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        from jax.experimental import enable_x64
    monkeypatch.setenv("PYCBC_JAX_ENABLE_X64", "0")
    with enable_x64(False):
        arr = jnp.ones(32, dtype=jnp.complex64)
        pts = jnp.array([1500001], dtype=jnp.int32)
        bins = jnp.array([0, 16, 32], dtype=jnp.int32)
        with pytest.raises(RuntimeError, match="PYCBC_JAX_ENABLE_X64=1"):
            if entry == "single":
                chisq_jax._shift_sum_gpu_core(arr, 15360., pts, bins, 2097152.)
            elif entry == "row":
                chisq_jax._shift_sum_gpu_core_row(
                    arr[None, :], 0, 15360., pts, bins, 2097152.
                )
            elif entry == "batch":
                chisq_jax._batched_points_chisq_core(
                    arr[None, :], jnp.array([0]), pts, bins[None, :],
                    15360., 2097152.
                )
            else:
                chisq_jax.shift_sum(np.ones(32, dtype=np.complex64), pts, bins)
    assert jax.config.jax_enable_x64


