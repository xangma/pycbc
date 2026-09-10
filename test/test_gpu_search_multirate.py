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
Unit and integration tests for multirate matched filtering in the
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
    prepare_multirate_plan,
    MultirateSearchEngine,
    bind_psd,
    SelectionPolicy,
    SearchEngine,
    prepare_power_chisq_plan,
    VetoManager,
)

CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA_AVAILABLE else [])


def _make_multirate_fixtures(num_templates=4, size=512, decimation_factor=4, device="cpu"):
    flen = size // 2 + 1
    delta_f = 1.0 / size
    rng = np.random.default_rng(777)

    flen_coarse = (size // decimation_factor) // 2 + 1

    templates = []
    for i in range(num_templates):
        h_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(np.complex64)
        h_vals[0] = 0.0
        # Concentrate power in low frequency range to represent realistic inspiral
        h_vals[flen_coarse:] *= 0.1
        pwr = np.sum(np.abs(h_vals) ** 2)
        if pwr > 0:
            h_vals /= np.sqrt(pwr)
        t = FrequencySeries(h_vals, delta_f=delta_f)
        t.id = 100 + i
        templates.append(t)

    plan = prepare_multirate_plan(
        templates,
        decimation_factor=decimation_factor,
        tile_size=2,
        device=device,
    )

    psd = FrequencySeries(np.ones(flen, dtype=np.float32) * 2.0, delta_f=delta_f)
    psd_coarse = FrequencySeries(np.ones(flen_coarse, dtype=np.float32) * 2.0, delta_f=delta_f)

    full_psd = bind_psd(plan.full_bank_plan, psd, psd_version="psd-v1", device=device)
    coarse_psd = bind_psd(plan.coarse_bank_plan, psd_coarse, psd_version="psd-v1", device=device)

    # Injected signal at sample 200 for template 0 with target SNR 15.0
    t0 = 200
    target_snr = 15.0
    sigmasq = float(full_psd.tile_sigmasqs[0][0])
    A = target_snr / np.sqrt(sigmasq)
    phase_shift = np.exp(-2j * np.pi * np.arange(flen) * delta_f * (t0 / (size * delta_f)))
    data_vals = templates[0].numpy() * phase_shift * A
    data = FrequencySeries(data_vals, delta_f=delta_f)
    data.psd = psd

    return plan, full_psd, coarse_psd, data, templates, t0


@pytest.mark.parametrize("device", DEVICES)
def test_multirate_plan_creation(device):
    """Verify multirate plan creation and geometry constraints."""
    plan, _, _, _, templates, _ = _make_multirate_fixtures(
        num_templates=4, size=512, decimation_factor=4, device=device
    )
    assert plan.decimation_factor == 4
    assert plan.band.transform_length == 128
    assert plan.band.filter_length == 65
    assert len(plan.coarse_bank_plan.tiles) == 2
    assert len(plan.full_bank_plan.tiles) == 2


@pytest.mark.parametrize("device", DEVICES)
def test_multirate_search_recall_and_parity(device):
    """Verify that multirate search recovers injected signals with 100% recall and peak parity."""
    plan, full_psd, coarse_psd, data, templates, t0 = _make_multirate_fixtures(
        num_templates=4, size=512, decimation_factor=4, device=device
    )
    valid_interval = (64, 448)

    # 1. Standard full-rate search engine
    policy = SelectionPolicy(snr_threshold=5.0, cluster_policy="live_peak")
    full_engine = SearchEngine(plan.full_bank_plan, policy, device=device)
    full_engine.submit(data, full_psd, valid_interval)
    ref_ready = full_engine.drain()
    ref_cands = ref_ready[0].results[0]
    assert len(ref_cands["template_id"]) > 0

    # 2. Multirate search engine
    mr_engine = MultirateSearchEngine(
        multirate_plan=plan,
        selection_policy=policy,
        device=device,
        use_cuda_graphs=(device == "cuda"),
    )
    mr_engine.submit(data, full_psd, coarse_psd, valid_interval)
    mr_ready = mr_engine.drain()
    mr_cands = mr_ready[0].results[0]

    # Verify candidate recovery
    assert len(mr_cands["template_id"]) == len(ref_cands["template_id"])
    np.testing.assert_array_equal(mr_cands["template_id"], ref_cands["template_id"])
    np.testing.assert_array_equal(mr_cands["sample_idx"], ref_cands["sample_idx"])
    np.testing.assert_allclose(mr_cands["snr"], ref_cands["snr"], atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_multirate_noise_rejection(device):
    """Verify early exit at coarse stage for sub-threshold noise chunks."""
    plan, full_psd, coarse_psd, _, _, _ = _make_multirate_fixtures(
        num_templates=4, size=512, decimation_factor=4, device=device
    )
    # Pure low-amplitude noise
    flen = plan.full_bank_plan.geometry.filter_length
    df = plan.full_bank_plan.geometry.delta_f
    noise = FrequencySeries(np.zeros(flen, dtype=np.complex64), delta_f=df)

    policy = SelectionPolicy(snr_threshold=8.0, cluster_policy="live_peak")
    mr_engine = MultirateSearchEngine(plan, policy, device=device)
    mr_engine.submit(noise, full_psd, coarse_psd, (64, 448))
    ready = mr_engine.drain()

    # No candidates should be proposed or refined
    assert len(ready[0].results) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_multirate_veto_evaluation(device):
    """Verify Power Chisq evaluation on refined multirate triggers."""
    plan, full_psd, coarse_psd, data, _, _ = _make_multirate_fixtures(
        num_templates=4, size=512, decimation_factor=4, device=device
    )
    chisq_plan = prepare_power_chisq_plan(plan.full_bank_plan, full_psd, num_bins=8, device=device)
    veto_mgr = VetoManager(power_chisq_plan=chisq_plan)

    policy = SelectionPolicy(snr_threshold=4.0, cluster_policy="live_peak")
    mr_engine = MultirateSearchEngine(
        plan, policy, veto_manager=veto_mgr, device=device
    )
    mr_engine.submit(data, full_psd, coarse_psd, (64, 448))
    ready = mr_engine.drain()

    assert len(ready[0].results) > 0
    res = ready[0].results[0]
    assert "chisq" in res
    assert len(res["chisq"]) > 0
    assert np.all(res["chisq"] >= 0.0)
