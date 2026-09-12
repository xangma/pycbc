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
Unit and integration tests for early consistency screening in the
persistent GPU search engine.
"""

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

from pycbc.types import FrequencySeries
from pycbc.filter.gpu_search import (
    prepare_bank,
    bind_psd,
    prepare_power_chisq_plan,
    VetoManager,
    ConsistencyScreen,
)

CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA_AVAILABLE else [])


def _make_screening_fixtures(num_templates=2, filter_length=513, delta_f=0.5, device="cpu"):
    rng = np.random.default_rng(42)
    N = (filter_length - 1) * 2

    templates = []
    for i in range(num_templates):
        h_data = (rng.normal(size=filter_length) + 1j * rng.normal(size=filter_length)).astype(np.complex64)
        h_data[0] = 0.0
        norm = np.linalg.norm(h_data)
        if norm > 0:
            h_data /= norm
        t = FrequencySeries(h_data, delta_f=delta_f)
        t.id = i
        t.f_lower = 20.0
        t.f_upper = 200.0
        templates.append(t)

    bank_plan = prepare_bank(templates, tile_size=num_templates, f_lower=20.0, f_upper=200.0, device=device)
    psd_data = np.ones(filter_length, dtype=np.float32) * 2.0
    psd = FrequencySeries(psd_data, delta_f=delta_f)
    psd_plan = bind_psd(bank_plan, psd, psd_version="psd-v1", device=device)

    # Injected signal matching template 0 at t0 = 150 with target SNR 16.0
    t0 = 150
    target_snr = 16.0
    sigmasq = float(psd_plan.tile_sigmasqs[0][0])
    A = target_snr / np.sqrt(sigmasq)
    phase_shift = np.exp(-2j * np.pi * np.arange(filter_length) * delta_f * (t0 / (N * delta_f)))
    signal_fd = templates[0].numpy() * phase_shift * A

    # Synthesize overwhitened correlation tile
    inv_psd = (4.0 * delta_f) / psd_data
    inv_psd[:40] = 0.0
    inv_psd[400:] = 0.0

    stilde = signal_fd / psd_data
    stilde[:40] = 0.0
    stilde[400:] = 0.0

    corr_tile_np = np.zeros((num_templates, N), dtype=np.complex64)
    for i in range(num_templates):
        corr_tile_np[i, :filter_length] = np.conj(templates[i].numpy()) * stilde

    if torch is not None and str(device) != "numpy":
        corr_tile = torch.from_numpy(corr_tile_np).to(device=device)
        tile_norms = psd_plan.tile_norms[0]
    else:
        corr_tile = corr_tile_np
        tile_norms = psd_plan.tile_norms[0]

    return bank_plan, psd_plan, corr_tile, tile_norms, N, t0


@pytest.mark.parametrize("device", DEVICES)
def test_consistency_screen_signal_and_glitch(device):
    """Verify diagnostic screening scores a signal and a discrepant candidate."""
    bank_plan, psd_plan, corr_tile, tile_norms, N, t0 = _make_screening_fixtures(device=device)

    power_plan = prepare_power_chisq_plan(bank_plan, psd_plan, num_bins=8, device=device)
    bin_edges = power_plan.tile_bin_edges[0]

    # Candidate 0: True signal (SNR ~ 16.0 at sample t0)
    # Candidate 1: Simulated delta-glitch candidate (high SNR 20.0 with concentrated high-freq energy)
    candidates = {
        "sample_idx": np.array([t0, t0 + 20], dtype=np.int64),
        "template_idx": np.array([0, 0], dtype=np.int64),
        "snr": np.array([16.0 + 0.0j, 20.0 + 0.0j], dtype=np.complex64),
        "sigmasq": np.array([float(psd_plan.tile_sigmasqs[0][0]), float(psd_plan.tile_sigmasqs[0][0])], dtype=np.float32),
    }

    # For the second candidate, corrupt corr_tile at sample t0 + 20 to simulate a high-frequency burst
    # (zero out lower half of frequency range so all energy is in the upper half)

    # In a glitch candidate whose energy is entirely in the upper half:
    # z1 = 0, so 2 * z1 * norm - snr = -snr, yielding chi^2 = |snr|^2 = 400.0 >> 50.0!
    # For candidate 0 (true signal), z1 ~ z/2, yielding chi^2 ~ 0!
    screen = ConsistencyScreen(screen_threshold=50.0, device=device, diagnostics=True)
    filtered = screen.filter(
        corr_tile=corr_tile,
        candidates=candidates,
        tile_id=0,
        tile_norms=tile_norms,
        transform_length=N,
        bin_edges=bin_edges,
    )

    # True signal must survive
    assert len(filtered["sample_idx"]) >= 1
    assert filtered["sample_idx"][0] == t0
    assert filtered["screen_chisq"][0] < 5.0
    assert screen.accepted_count >= 1


@pytest.mark.parametrize("device", DEVICES)
def test_veto_manager_with_consistency_screen(device):
    """Verify diagnostic screening preserves candidates for full power chisq."""
    bank_plan, psd_plan, corr_tile, tile_norms, N, t0 = _make_screening_fixtures(device=device)

    power_plan = prepare_power_chisq_plan(bank_plan, psd_plan, num_bins=8, device=device)
    screen = ConsistencyScreen(screen_threshold=50.0, device=device, diagnostics=True)
    veto_mgr = VetoManager(
        power_chisq_plan=power_plan,
        consistency_screen=screen,
    )

    # Candidate with huge discrepancy (snr=20.0, but zero lower-frequency correlation)
    # The true signal candidate at t0 will have small screening score
    candidates = {
        "sample_idx": np.array([t0], dtype=np.int64),
        "template_idx": np.array([0], dtype=np.int64),
        "snr": np.array([16.0 + 0.0j], dtype=np.complex64),
        "sigmasq": np.array([float(psd_plan.tile_sigmasqs[0][0])], dtype=np.float32),
    }

    result = veto_mgr.evaluate(
        corr_tile=corr_tile,
        candidates=candidates,
        tile_id=0,
        tile_norms=tile_norms,
        transform_length=N,
    )

    assert len(result["sample_idx"]) == 1
    assert "chisq" in result
    assert "screen_chisq" in result
    assert result["screen_chisq"][0] < 5.0
    assert result["chisq"][0] < 5.0


@pytest.mark.parametrize("device", DEVICES)
def test_consistency_screen_empty(device):
    """Verify screener handles empty candidate buffers gracefully."""
    _, _, corr_tile, tile_norms, N, _ = _make_screening_fixtures(device=device)
    screen = ConsistencyScreen(screen_threshold=50.0, device=device, diagnostics=True)

    empty_cands = {}
    res = screen.filter(corr_tile, empty_cands, 0, tile_norms, N)
    assert res == {}

    empty_cands2 = {"sample_idx": np.empty(0, dtype=np.int64)}
    res2 = screen.filter(corr_tile, empty_cands2, 0, tile_norms, N)
    assert len(res2["sample_idx"]) == 0


@pytest.mark.parametrize("device", ["numpy"] + DEVICES)
def test_screen_cutoff_counterexample_survives_real_veto_and_ranking(device):
    from pycbc.events.ranking import newsnr
    from pycbc.filter.gpu_search.candidates import candidates_to_host
    from pycbc.filter.gpu_search.vetoes import PowerChisqPlan, batched_power_chisq

    n = 64
    corr = np.zeros((1, n), dtype=np.complex64)
    corr[0, 1:17] = np.r_[np.full(8, (10 + np.sqrt(60)) / 16),
                          np.full(8, (10 - np.sqrt(60)) / 16)]
    edges = np.arange(1, 18, dtype=np.int64)[None, :]
    norms = np.ones(1, dtype=np.float32)
    if torch is not None and device != "numpy":
        corr = torch.tensor(corr, device=device)
        edges = torch.tensor(edges, device=device)
        norms = torch.tensor(norms, device=device)
    candidates = {
        "template_idx": np.array([0]), "sample_idx": np.array([0]),
        "snr": np.array([10 + 0j], dtype=np.complex64),
    }
    screen = ConsistencyScreen(device=device)
    score = screen.evaluate_screening_chisq(corr, candidates, norms, n, edges)
    chi, dof = batched_power_chisq(corr, candidates, edges, norms, transform_length=n)
    np.testing.assert_allclose(score, [60], atol=1e-4)
    np.testing.assert_allclose(chi, [60], atol=1e-4)
    np.testing.assert_array_equal(dof, [30])
    assert newsnr(10, chi / dof) > 7.7
    manager = VetoManager(
        power_chisq_plan=PowerChisqPlan(tile_bin_edges={0: edges}),
        consistency_screen=screen,
    )
    result = manager.evaluate(corr, candidates, 0, norms, n)
    assert len(result["sample_idx"]) == 1
    assert screen.rejected_count == 0
    # Ranking consumes public host results after the device veto boundary.
    result = candidates_to_host(result)
    assert newsnr(abs(result["snr"]), result["chisq"] / result["chisq_dof"])[0] > 7.7


@pytest.mark.parametrize("device", DEVICES)
def test_compatible_screen_preserves_tensor_fields_without_scoring(device, monkeypatch):
    if torch is None:
        pytest.skip("Torch unavailable")
    screen = ConsistencyScreen(device=device)
    candidates = {
        "sample_idx": torch.tensor([0, 1], device=device),
        "snr": torch.tensor([10 + 0j, 20 + 0j], device=device),
    }
    def forbidden(*args, **kwargs):
        raise AssertionError("Compatible screening must not compute diagnostic scores")
    monkeypatch.setattr(screen, "evaluate_screening_chisq", forbidden)
    result = screen.filter(None, candidates, 0, None, 32)
    assert result is candidates
    assert result["snr"] is candidates["snr"]
    assert screen.accepted_count == 2


@pytest.mark.parametrize("device", ["numpy"] + DEVICES)
def test_odd_full_bin_count_uses_unequal_group_fractions(device):
    corr = np.zeros((1, 32), dtype=np.complex64)
    corr[0, 1:6] = 2
    edges = np.arange(1, 7, dtype=np.int64)[None, :]
    norms = np.ones(1, dtype=np.float32)
    if torch is not None and device != "numpy":
        corr = torch.tensor(corr, device=device)
    candidates = {"sample_idx": np.array([0]), "template_idx": np.array([0]),
                  "snr": np.array([10 + 0j], dtype=np.complex64)}
    score = ConsistencyScreen(device=device).evaluate_screening_chisq(
        corr, candidates, norms, 32, edges,
    )
    np.testing.assert_allclose(score, [0], atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_experimental_rejection_masks_tensor_fields(device):
    if torch is None:
        pytest.skip("Torch unavailable")
    screen = ConsistencyScreen(device=device, experimental_rejection=True)
    candidates = {
        "sample_idx": torch.tensor([0, 0], device=device),
        "template_idx": torch.tensor([0, 0], device=device),
        "snr": torch.tensor([0 + 0j, 20 + 0j], device=device),
    }
    corr = torch.zeros((1, 32), dtype=torch.complex64, device=device)
    result = screen.filter(corr, candidates, 0, torch.ones(1, device=device), 32)
    assert result["sample_idx"].device == candidates["sample_idx"].device
    assert len(result["sample_idx"]) == len(result["snr"]) == 1
    assert screen.rejected_count == 1
