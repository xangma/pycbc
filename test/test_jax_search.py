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
