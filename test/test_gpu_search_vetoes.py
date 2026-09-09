# Copyright (C) 2026
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

"""
Unit tests for GPU search engine batched full-statistic vetoes (Power chisq & SG chisq).
"""

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

DEVICES = ["cpu"] + (
    ["cuda"] if (torch is not None and torch.cuda.is_available()) else []
)

from pycbc.types import FrequencySeries
from pycbc.filter.matchedfilter import (
    sigmasq_series,
    get_cutoff_indices,
    matched_filter_core,
)
from pycbc.vetoes.chisq import (
    power_chisq_bins_from_sigmasq_series,
    power_chisq_at_points_from_precomputed,
)
from pycbc.filter.gpu_search.plans import prepare_bank, bind_psd
from pycbc.filter.gpu_search.candidates import SelectionPolicy
from pycbc.filter.gpu_search.vetoes import (
    prepare_power_chisq_plan,
    batched_power_chisq,
)
from pycbc.filter.gpu_search.engine import SearchEngine


def _make_test_fixtures(filter_length=513, delta_f=0.5, num_templates=2):
    np.random.seed(42)
    N = (filter_length - 1) * 2

    templates = []
    for i in range(num_templates):
        h_data = (
            np.random.randn(filter_length) + 1j * np.random.randn(filter_length)
        ).astype(np.complex64)
        h_data[0] = 0.0
        htilde = FrequencySeries(h_data, delta_f=delta_f)
        htilde.id = i
        templates.append(htilde)

    s_data = (
        np.random.randn(filter_length) + 1j * np.random.randn(filter_length)
    ).astype(np.complex64)
    s_data[0] = 0.0
    stilde = FrequencySeries(s_data, delta_f=delta_f)

    psd_data = np.ones(filter_length, dtype=np.float32) * 2.0
    psd = FrequencySeries(psd_data, delta_f=delta_f)

    return templates, stilde, psd, N


@pytest.mark.parametrize("device", DEVICES)
def test_power_chisq_plan_creation_and_bin_edges(device):
    templates, _, psd, N = _make_test_fixtures(num_templates=3)
    bank_plan = prepare_bank(
        templates, tile_size=3, f_lower=20.0, f_upper=200.0, device=device
    )
    psd_plan = bind_psd(bank_plan, psd, device=device)

    num_bins = 16
    plan = prepare_power_chisq_plan(
        bank_plan, psd_plan, num_bins=num_bins, snr_threshold=5.5, device=device
    )

    assert plan.num_bins == num_bins
    assert plan.snr_threshold == 5.5
    assert plan.dof == 2 * num_bins - 2
    assert 0 in plan.tile_bin_edges

    tile_bins = plan.tile_bin_edges[0]
    if hasattr(tile_bins, "cpu"):
        tile_bins = tile_bins.cpu().numpy()

    assert tile_bins.shape == (3, num_bins + 1)

    kmin, kmax = get_cutoff_indices(20.0, 200.0, bank_plan.geometry.delta_f, N)

    # Check each template bin edges match reference power_chisq_bins_from_sigmasq_series
    for i in range(3):
        h = templates[i]
        s_series = sigmasq_series(
            h, psd=psd, low_frequency_cutoff=20.0, high_frequency_cutoff=200.0
        )
        ref_bins = power_chisq_bins_from_sigmasq_series(s_series, num_bins, kmin, kmax)
        np.testing.assert_array_equal(tile_bins[i], ref_bins)


@pytest.mark.parametrize("device", DEVICES)
def test_batched_power_chisq_numerical_parity(device):
    templates, stilde, psd, N = _make_test_fixtures(num_templates=1)
    htilde = templates[0]

    bank_plan = prepare_bank(
        [htilde], tile_size=1, f_lower=20.0, f_upper=200.0, device=device
    )
    psd_plan = bind_psd(bank_plan, psd, device=device)

    num_bins = 16
    plan = prepare_power_chisq_plan(
        bank_plan, psd_plan, num_bins=num_bins, device=device
    )

    ref_snr_series, ref_corr, ref_norm = matched_filter_core(
        htilde, stilde, psd=psd, low_frequency_cutoff=20.0, high_frequency_cutoff=200.0
    )

    points = np.array([120, 250, 480], dtype=np.int64)
    ref_snrv = np.asarray(ref_snr_series[points])
    tile_bins = plan.tile_bin_edges[0]
    if hasattr(tile_bins, "cpu"):
        tile_bins_np = tile_bins.cpu().numpy()[0]
    else:
        tile_bins_np = tile_bins[0]

    ref_chisq = power_chisq_at_points_from_precomputed(
        ref_corr, ref_snrv, ref_norm, tile_bins_np, points
    )
    ref_chisq_vals = np.asarray(ref_chisq)

    # Setup candidate dict for batched_power_chisq
    cand_snr_normed = ref_snrv * ref_norm
    candidates = {
        "template_idx": np.zeros(len(points), dtype=np.int64),
        "sample_idx": points,
        "snr": cand_snr_normed,
    }

    flen = bank_plan.geometry.filter_length
    corr_np = np.zeros((1, N), dtype=np.complex64)
    corr_np[0, :flen] = np.asarray(ref_corr.numpy()[:flen])

    if torch is not None and device != "numpy":
        corr_mem = torch.as_tensor(corr_np, device=torch.device(device))
        norms_mem = psd_plan.tile_norms[0]
        bins_mem = plan.tile_bin_edges[0]
    else:
        corr_mem = corr_np
        norms_mem = psd_plan.tile_norms[0]
        bins_mem = plan.tile_bin_edges[0]

    chisq, chisq_dof = batched_power_chisq(
        corr_tile=corr_mem,
        candidates=candidates,
        tile_bin_edges=bins_mem,
        tile_norms=norms_mem,
        num_bins=num_bins,
        snr_threshold=None,
        transform_length=N,
    )

    assert chisq.shape == (len(points),)
    assert (chisq_dof == 2 * num_bins - 2).all()
    np.testing.assert_allclose(chisq, ref_chisq_vals, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_power_chisq_activation_threshold(device):
    templates, stilde, psd, N = _make_test_fixtures(num_templates=1)
    bank_plan = prepare_bank(templates, tile_size=1, device=device)
    psd_plan = bind_psd(bank_plan, psd, device=device)
    plan = prepare_power_chisq_plan(bank_plan, psd_plan, num_bins=16, device=device)

    snr_series, ref_corr, ref_norm = matched_filter_core(
        templates[0], stilde * 10.0, psd=psd
    )
    snr_vals = np.asarray(snr_series.numpy() * ref_norm)

    # Trigger 0 has low SNR (e.g. 2.0), Trigger 1 has high SNR from the filtered series
    candidates = {
        "template_idx": np.array([0, 0], dtype=np.int64),
        "sample_idx": np.array([150, 250], dtype=np.int64),
        "snr": np.array([2.0 + 0j, snr_vals[250]], dtype=np.complex64),
    }

    flen = bank_plan.geometry.filter_length
    corr_np = np.zeros((1, N), dtype=np.complex64)
    corr_np[0, :flen] = ref_corr.numpy()[:flen]

    corr_mem = (
        torch.as_tensor(corr_np, device=torch.device(device))
        if (torch is not None and device != "numpy")
        else corr_np
    )

    chisq, chisq_dof = batched_power_chisq(
        corr_tile=corr_mem,
        candidates=candidates,
        tile_bin_edges=plan.tile_bin_edges[0],
        tile_norms=psd_plan.tile_norms[0],
        num_bins=16,
        snr_threshold=5.0,
        transform_length=N,
    )

    # Below threshold: 0.0; Above threshold: > 0.0
    assert chisq[0] == 0.0
    assert chisq[1] > 0.0
    assert chisq_dof[0] == 30
    assert chisq_dof[1] == 30


@pytest.mark.parametrize("device", DEVICES)
def test_power_chisq_large_sample_stability(device):
    # Test large time offset (e.g. 500,000) for phase stability
    large_N = 1048576  # 2^20
    flen = large_N // 2 + 1
    kmin, kmax = 40, 400
    num_bins = 16

    corr_np = np.zeros((1, flen), dtype=np.complex64)
    corr_np[0, kmin:kmax] = 1.0 + 0.5j

    bins_np = np.linspace(kmin, kmax, num_bins + 1, dtype=np.int64)[np.newaxis, :]
    norms_np = np.array([0.05], dtype=np.float32)

    # Large time indices
    large_points = np.array([500000, 999999], dtype=np.int64)
    candidates = {
        "template_idx": np.array([0, 0], dtype=np.int64),
        "sample_idx": large_points,
        "snr": np.array([8.0 + 2j, 9.0 - 1j], dtype=np.complex64),
    }

    if torch is not None and device != "numpy":
        corr_mem = torch.as_tensor(corr_np, device=torch.device(device))
        bins_mem = torch.as_tensor(bins_np, device=torch.device(device))
        norms_mem = torch.as_tensor(norms_np, device=torch.device(device))
    else:
        corr_mem = corr_np
        bins_mem = bins_np
        norms_mem = norms_np

    chisq, _ = batched_power_chisq(
        corr_tile=corr_mem,
        candidates=candidates,
        tile_bin_edges=bins_mem,
        tile_norms=norms_mem,
        num_bins=num_bins,
        snr_threshold=None,
        transform_length=large_N,
    )

    # Check finite and non-negative
    assert np.isfinite(chisq).all()
    assert (chisq >= 0.0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_engine_integration_with_power_chisq(device):
    templates, stilde, psd, N = _make_test_fixtures(num_templates=2)
    bank_plan = prepare_bank(templates, tile_size=2, device=device)
    psd_plan = bind_psd(bank_plan, psd, device=device)

    chisq_plan = prepare_power_chisq_plan(
        bank_plan, psd_plan, num_bins=16, snr_threshold=0.0, device=device
    )

    policy = SelectionPolicy(
        snr_threshold=0.0,
        cluster_policy="live_peak",
    )

    engine = SearchEngine(
        bank_plan, policy, veto_manager=chisq_plan, device=device
    )

    ticket = engine.submit(stilde, psd_plan, valid_interval=(100, 900))
    assert ticket.completed
    assert not ticket.aborted

    ready = engine.drain()
    assert len(ready) == 1
    cands = ready[0].results[0]

    assert "chisq" in cands
    assert "chisq_dof" in cands
    assert len(cands["chisq"]) == len(cands["sample_idx"])
    assert (cands["chisq_dof"] == 30).all()

    engine.close()
