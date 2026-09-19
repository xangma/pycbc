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

"""Tests for JAX batched search engine, 2D overwhitening, vetoes, and ranking."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import pycbc
from pycbc import scheme
from pycbc.events import cuts, ranking, veto
from pycbc.filter.matchedfilter import matched_filter, sigmasq
from pycbc.filter.matchedfilter_jax import (
    _batched_filter_and_screen,
    _batched_filter_and_screen_lean,
    batch_matched_filter_bank,
    batch_peak_magnitudes,
    batch_peak_values,
)
from pycbc.strain.strain_jax import detect_loud_glitches_jax
from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.types.array_jax import is_jax_array


if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


def test_batch_peak_values_and_magnitudes_parity():
    """Verify batched peak reduction matches per-template search."""
    np.random.seed(42)
    template_count = 8
    template_size = 512
    data = (
        np.random.randn(template_count, template_size)
        + 1j * np.random.randn(template_count, template_size)
    ).astype(np.complex128)

    # Spike a known max into each row
    valid_start = 64
    valid_end = 448
    expected_locs = []
    expected_peaks = []
    for i in range(template_count):
        spike_loc = np.random.randint(valid_start, valid_end)
        data[i, spike_loc] = 100.0 + 50.0j * (i + 1)
        expected_locs.append(spike_loc - valid_start)
        expected_peaks.append(data[i, spike_loc])

    flat_data = data.reshape(-1)
    seg = slice(valid_start, valid_end)

    with scheme.JAXScheme():
        j_output = Array(flat_data)
        indices, peaks = batch_peak_values(
            j_output, template_count, template_size, seg
        )
        mags = batch_peak_magnitudes(peaks)

    np.testing.assert_array_equal(indices, expected_locs)
    np.testing.assert_allclose(peaks, expected_peaks, rtol=1e-12)
    np.testing.assert_allclose(mags, np.abs(expected_peaks), rtol=1e-12)


def test_batch_matched_filter_bank_parity():
    """Verify 2D overwhitened batch filtering matches 1D matched_filter."""
    np.random.seed(50)
    n_time = 2048
    dt = 1.0 / 2048
    n_freq = n_time // 2 + 1
    df = 1.0 / (n_time * dt)

    flow = 30.0
    fhigh = 800.0

    # Create mock strain and psd
    strain_data = np.random.randn(n_time)
    strain_ts = TimeSeries(strain_data, delta_t=dt, dtype=np.float64)
    strain_fs = strain_ts.to_frequencyseries()

    psd_vals = np.full(n_freq, 2.0, dtype=np.float64)
    psd_fs = FrequencySeries(psd_vals, delta_f=df)

    # Create 4 mock templates
    num_templates = 4
    templates = []
    templates_matrix = np.zeros((num_templates, n_freq), dtype=np.complex128)
    expected_snrs = []
    expected_sigmasqs = []

    for k in range(num_templates):
        sig = np.sin(2.0 * np.pi * (50.0 + k * 20.0) * np.linspace(0, 1.0, n_time))
        t_ts = TimeSeries(sig, delta_t=dt, dtype=np.float64)
        t_fs = t_ts.to_frequencyseries()
        templates.append(t_fs)
        templates_matrix[k, :] = t_fs.numpy()

        # Single template CPU reference
        s_sq = sigmasq(
            t_fs, psd=psd_fs, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh
        )
        snr_ref = matched_filter(
            t_fs,
            strain_fs,
            psd=psd_fs,
            low_frequency_cutoff=flow,
            high_frequency_cutoff=fhigh,
        )
        expected_sigmasqs.append(s_sq)
        expected_snrs.append(snr_ref.numpy())

    # Execute JAX batched bank filter
    batch_snr, batch_sigmasq = batch_matched_filter_bank(
        templates_matrix,
        strain_fs.numpy(),
        psd=psd_vals,
        low_frequency_cutoff=flow,
        high_frequency_cutoff=fhigh,
        delta_f=df,
    )

    assert is_jax_array(batch_snr)
    assert is_jax_array(batch_sigmasq)
    assert batch_snr.shape == (num_templates, n_time)
    assert batch_sigmasq.shape == (num_templates,)

    # Verify numerical match across all templates
    for k in range(num_templates):
        np.testing.assert_allclose(
            float(batch_sigmasq[k]), expected_sigmasqs[k], rtol=1e-9
        )
        np.testing.assert_allclose(
            np.asarray(batch_snr[k]), expected_snrs[k], rtol=1e-8, atol=1e-8
        )


def test_jax_veto_within_and_outside_times():
    """Verify segment vetoing on device matches CPU reference."""
    times = np.array([1.0, 2.5, 3.2, 5.0, 7.8, 9.0, 11.5], dtype=np.float64)
    start = np.array([2.0, 7.0], dtype=np.float64)
    end = np.array([4.0, 8.5], dtype=np.float64)

    expected_within = veto.indices_within_times(times, start, end)
    expected_outside = veto.indices_outside_times(times, start, end)

    with scheme.JAXScheme():
        j_times = Array(times)
        j_start = Array(start)
        j_end = Array(end)

        actual_within = veto.indices_within_times(j_times, j_start, j_end)
        actual_outside = veto.indices_outside_times(j_times, j_start, j_end)

    assert isinstance(actual_within, Array)
    assert isinstance(actual_outside, Array)
    np.testing.assert_array_equal(actual_within.numpy(), expected_within)
    np.testing.assert_array_equal(actual_outside.numpy(), expected_outside)


def test_jax_ranking_effsnr_and_newsnr():
    """Verify effsnr and newsnr on device."""
    snr_vals = np.array([6.0, 8.5, 12.0, 15.0], dtype=np.float64)
    rchisq_vals = np.array([0.9, 1.2, 2.5, 5.0], dtype=np.float64)

    expected_eff = ranking.effsnr(snr_vals, rchisq_vals)
    expected_new = ranking.newsnr(snr_vals, rchisq_vals)

    with scheme.JAXScheme():
        j_snr = Array(snr_vals)
        j_rchisq = Array(rchisq_vals)

        actual_eff = ranking.effsnr(j_snr, j_rchisq)
        actual_new = ranking.newsnr(j_snr, j_rchisq)

    assert isinstance(actual_eff, Array)
    assert isinstance(actual_new, Array)
    np.testing.assert_allclose(actual_eff.numpy(), expected_eff, rtol=1e-12)
    np.testing.assert_allclose(actual_new.numpy(), expected_new, rtol=1e-12)


def test_jax_trigger_cuts():
    """Verify trigger cuts evaluation on device."""
    snr_vals = np.array([5.0, 6.5, 8.0, 10.0, 12.0], dtype=np.float64)
    rchisq_vals = np.array([1.0, 1.5, 2.0, 0.8, 1.1], dtype=np.float64)

    triggers_host = {"snr": snr_vals, "chisq": rchisq_vals}
    # Cut: snr > 6.0 and chisq < 1.8
    cut_dict = {
        ("snr", lambda x, t: x > t): 6.0,
        ("chisq", lambda x, t: x < t): 1.8,
    }
    expected_indices = cuts.apply_trigger_cuts(triggers_host, cut_dict)

    with scheme.JAXScheme():
        triggers_jax = {"snr": Array(snr_vals), "chisq": Array(rchisq_vals)}
        actual_indices = cuts.apply_trigger_cuts(triggers_jax, cut_dict)

    assert isinstance(actual_indices, Array)
    np.testing.assert_array_equal(actual_indices.numpy(), expected_indices)


def test_jax_autogating_detection():
    """Verify loud glitch detection in JAX."""
    dt = 1.0 / 2048
    duration = 32.0
    n = int(duration / dt)
    np.random.seed(99)
    noise = 0.01 * np.random.randn(n)

    # Add loud glitch at 16.0s
    spike_idx = int(16.0 / dt)
    noise[spike_idx - 5:spike_idx + 5] += 10.0

    strain = TimeSeries(noise, delta_t=dt, dtype=np.float64, epoch=0.0)

    with scheme.JAXScheme():
        times, snrs = detect_loud_glitches_jax(
            strain,
            psd_duration=4.0,
            psd_stride=2.0,
            threshold=10.0,
            cluster_window=2.0,
            corrupt_time=2.0,
        )

    assert len(times) >= 1
    assert any(abs(t - 16.0) < 1.0 for t in times)


def test_batched_matched_filter_and_cluster_parity():
    """Verify batched_matched_filter_and_cluster parity against sequential execution."""
    from pycbc.filter.matchedfilter import MatchedFilterControl, sigmasq
    from pycbc.types import zeros

    np.random.seed(123)
    n_time = 4096
    dt = 1.0 / 2048
    n_freq = n_time // 2 + 1
    df = 1.0 / (n_time * dt)
    flow = 30.0
    fhigh = 800.0

    # Overwhitened segment
    strain_data = np.random.randn(n_time).astype(np.float32)
    strain_ts = TimeSeries(strain_data, delta_t=dt)
    seg = strain_ts.to_frequencyseries()
    seg.analyze = slice(512, n_time - 512)

    # PSD for normalization
    psd_vals = np.full(n_freq, 1.0, dtype=np.float32)
    psd_fs = FrequencySeries(psd_vals, delta_f=df)

    num_templates = 4
    templates = []
    sigmasqs = []
    for k in range(num_templates):
        sig = np.sin(2.0 * np.pi * (50.0 + k * 25.0) * np.linspace(0, 1.0, n_time)).astype(np.float32)
        # Inject loud signal in template 2
        if k == 2:
            sig *= 50.0
        t_ts = TimeSeries(sig, delta_t=dt)
        t_fs = t_ts.to_frequencyseries()
        templates.append(t_fs)
        sigmasqs.append(sigmasq(t_fs, psd=psd_fs, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh))

    with scheme.JAXScheme():
        template_mem = zeros(n_time, dtype=np.complex64)
        mf_control = MatchedFilterControl(
            flow, fhigh, 5.5, n_time, df, np.complex64,
            [seg], template_mem, use_cluster=True, cluster_function="symmetric"
        )

        # Sequential results
        seq_results = []
        for tmpl, ssq in zip(templates, sigmasqs):
            mf_control.htilde[:] = tmpl[:]
            res = mf_control.matched_filter_and_cluster(0, ssq, window=64)
            seq_results.append(res)

        # Batched results
        batch_results = mf_control.batched_matched_filter_and_cluster(
            0, templates, sigmasqs, window=64
        )

    assert len(batch_results) == num_templates
    for k in range(num_templates):
        snr_s, norm_s, corr_s, idx_s, snrv_s = seq_results[k]
        snr_b, norm_b, corr_b, idx_b, snrv_b = batch_results[k]

        np.testing.assert_allclose(norm_b, norm_s, rtol=1e-5)
        np.testing.assert_array_equal(idx_b, idx_s)
        if len(idx_s) > 0:
            np.testing.assert_allclose(snrv_b, snrv_s, rtol=1e-4, atol=1e-4)

    # Exercise the production TemplateBatchList path, which supplies the
    # full tensor to the JAX backend, in both output modes.
    from pycbc.waveform.bank import TemplateBatchList
    batch_templates = TemplateBatchList(templates)
    batch_templates._batch_tensor = np.stack([tmpl.numpy() for tmpl in templates])
    for need_snr in (False, True):
        mf_control.need_snr_series = need_snr
        mf_control._jax_segments_tensor = None
        mf_control.clear_batch_cache()
        with scheme.JAXScheme():
            tensor_results = mf_control.batched_matched_filter_and_cluster(
                0, batch_templates, sigmasqs, window=64
            )

        mf_control._jax_segments_tensor = None
        mf_control.clear_batch_cache()
        with scheme.JAXScheme():
            list_results = mf_control.batched_matched_filter_and_cluster(
                0, templates, sigmasqs, window=64
            )

        for tensor_result, list_result in zip(tensor_results, list_results):
            snr_t, norm_t, _, idx_t, snrv_t = tensor_result
            snr_l, norm_l, _, idx_l, snrv_l = list_result
            np.testing.assert_allclose(norm_t, norm_l, rtol=1e-5)
            np.testing.assert_array_equal(idx_t, idx_l)
            np.testing.assert_allclose(snrv_t, snrv_l, rtol=1e-4, atol=1e-4)
            if need_snr:
                if isinstance(snr_t, list) or isinstance(snr_l, list):
                    assert snr_t == [] and snr_l == []
                else:
                    assert snr_t is not None and snr_l is not None
                    np.testing.assert_allclose(
                        snr_t.numpy(), snr_l.numpy(), rtol=1e-5, atol=1e-5
                    )
            else:
                assert all(
                    snr is None or (isinstance(snr, list) and not snr)
                    for snr in (snr_t, snr_l)
                )

    # Quiet batches preserve normalization and the empty-result contract in
    # both modes when normalization values are transferred as one array.
    for need_snr in (False, True):
        mf_control.snr_threshold = 1e9
        mf_control.need_snr_series = need_snr
        mf_control._jax_segments_tensor = None
        with scheme.JAXScheme():
            quiet_results = mf_control.batched_matched_filter_and_cluster(
                0, templates, sigmasqs, window=64
            )
        assert len(quiet_results) == num_templates
        for quiet, reference in zip(quiet_results, batch_results):
            snr, norm, corr, idx, snrv = quiet
            assert snr == [] and corr == []
            np.testing.assert_allclose(norm, reference[1], rtol=1e-7)
            assert idx.size == 0 and idx.dtype == np.uint32
            assert snrv.size == 0 and snrv.dtype == np.complex64


def test_batched_filter_full_tensor_matches_cropped_tensor():
    """Slicing full templates inside JIT preserves both screening paths."""
    rng = np.random.default_rng(1234)
    template_count = 3
    n_freq = 17
    kmin, kmax = 3, 12
    tlen = 2 * (n_freq - 1)
    valid_start, valid_stop = 4, 28
    templates = (
        rng.normal(size=(template_count, n_freq))
        + 1j * rng.normal(size=(template_count, n_freq))
    ).astype(np.complex64)
    seg_slice = (
        rng.normal(size=kmax - kmin) + 1j * rng.normal(size=kmax - kmin)
    ).astype(np.complex64)
    cropped = templates[:, kmin:kmax]

    with scheme.JAXScheme():
        full = _batched_filter_and_screen(
            templates,
            seg_slice,
            kmin,
            kmax,
            tlen,
            valid_start,
            valid_stop,
            full_templates=True,
        )
        reference = _batched_filter_and_screen(
            cropped,
            seg_slice,
            kmin,
            kmax,
            tlen,
            valid_start,
            valid_stop,
        )
        full_lean = _batched_filter_and_screen_lean(
            templates,
            seg_slice,
            kmin,
            kmax,
            tlen,
            valid_start,
            valid_stop,
            full_templates=True,
        )
        reference_lean = _batched_filter_and_screen_lean(
            cropped,
            seg_slice,
            kmin,
            kmax,
            tlen,
            valid_start,
            valid_stop,
        )

    for got, expected in zip(full, reference):
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    for got, expected in zip(full_lean, reference_lean):
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_jax_power_chisq_parity():
    """Verify JAX power chisq shift_sum numerical parity against CPU reference."""
    from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed

    np.random.seed(42)
    n_time = 32768
    kmin = 200
    kmax = 8000
    corr_data = (np.random.randn(n_time) + 1j * np.random.randn(n_time)).astype(
        np.complex64
    )
    corr_fs = FrequencySeries(corr_data, delta_f=1.0)
    bins = np.linspace(kmin, kmax, 17, dtype=np.uint32)
    pts = np.array([500, 1200, 5000, 15000], dtype=np.uint32)
    snr_norm = 0.05
    snrv = np.array(
        [10.0 + 5.0j, 8.0 - 3.0j, 12.0 + 1.0j, 7.0 + 2.0j], dtype=np.complex64
    )

    # CPU reference under default scheme
    cpu_chisq = power_chisq_at_points_from_precomputed(
        corr_fs, snrv, snr_norm, bins, pts
    )

    # JAX scheme
    with scheme.JAXScheme():
        jax_chisq = power_chisq_at_points_from_precomputed(
            corr_fs, snrv, snr_norm, bins, pts
        )

    np.testing.assert_allclose(jax_chisq, cpu_chisq, rtol=1e-3, atol=1e-3)


def test_batch_power_chisq_jax_parity():
    """Verify batch_power_chisq_jax matches SingleDetPowerChisq.values parity."""
    import jax.numpy as jnp
    from pycbc.vetoes.chisq import SingleDetPowerChisq
    from pycbc.vetoes.chisq_jax import batch_power_chisq_jax
    from pycbc.waveform.bank import LazyFrequencySeries

    np.random.seed(42)
    B = 4
    n_slice = 2048
    kmin = 100
    tlen = 8192
    delta_f = 0.5
    num_bins = 16

    corr_slice = jax.random.normal(jax.random.PRNGKey(1), (B, n_slice), dtype=jnp.complex64)
    psd_dummy = FrequencySeries(np.ones(tlen // 2 + 1, dtype=np.float32), delta_f=delta_f)
    psd_id = id(psd_dummy)
    psd_dummy._chisq_cached_key = {}

    templates = []
    for i in range(B):
        tmpl = FrequencySeries(np.ones(tlen // 2 + 1, dtype=np.complex64), delta_f=delta_f)
        tmpl.params = object()
        tmpl._bin_cache = {psd_id: np.linspace(kmin, kmin + n_slice, num_bins + 1, dtype=np.uint32)}
        psd_dummy._chisq_cached_key[id(tmpl.params)] = True
        templates.append(tmpl)

    batch_results = []
    expected_chisqs = {}
    expected_dofs = {}
    analyze_start = 50

    power_chisq = SingleDetPowerChisq("16", snr_threshold=5.0)

    for i in range(B):
        n_trig = 4
        idx = np.array([100 * (j + 1) for j in range(n_trig)], dtype=np.uint32)
        # 2 points above threshold (snr >= 5), 2 below (snr < 5)
        snrv = np.array([20.0, 80.0, 10.0, 100.0], dtype=np.complex64)
        norm = 0.1
        corr = LazyFrequencySeries(corr_slice, i, delta_f)
        corr._kmin = kmin
        corr._tlen = tlen
        batch_results.append((None, norm, corr, idx, snrv))

        with scheme.JAXScheme():
            exp_c, exp_dof = power_chisq.values(corr, snrv, norm, psd_dummy, idx + analyze_start, templates[i])
        expected_chisqs[i] = exp_c
        expected_dofs[i] = exp_dof

    with scheme.JAXScheme():
        chisq_map = batch_power_chisq_jax(
            corr_slice, batch_results, templates, psd_dummy, analyze_start, snr_threshold=5.0, power_chisq=power_chisq
        )

    assert chisq_map is not None
    for i in range(B):
        actual_c, actual_dof = chisq_map[i]
        np.testing.assert_allclose(actual_c, expected_chisqs[i], rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(actual_dof, expected_dofs[i])
