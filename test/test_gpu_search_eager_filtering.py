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
Unit tests for GPU search engine eager dense filtering and candidate selection.
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
from pycbc.filter.matchedfilter import matched_filter_core
from pycbc.filter.gpu_search.plans import prepare_bank, bind_psd
from pycbc.filter.gpu_search.candidates import CandidateBuffer, SelectionPolicy
from pycbc.filter.gpu_search.engine import SearchEngine


def _make_test_fixtures(filter_length=513, delta_f=0.5):
    """Create deterministic template and strain data for filtering test."""
    np.random.seed(12345)
    N = (filter_length - 1) * 2

    # Template
    h_data = (
        np.random.randn(filter_length) + 1j * np.random.randn(filter_length)
    ).astype(np.complex64)
    h_data[0] = 0.0  # Zero DC
    htilde = FrequencySeries(h_data, delta_f=delta_f)
    htilde.id = 0

    # Strain data in frequency domain
    s_data = (
        np.random.randn(filter_length) + 1j * np.random.randn(filter_length)
    ).astype(np.complex64)
    s_data[0] = 0.0  # Zero DC
    stilde = FrequencySeries(s_data, delta_f=delta_f)

    # Flat PSD
    psd_data = np.ones(filter_length, dtype=np.float32) * 2.0
    psd = FrequencySeries(psd_data, delta_f=delta_f)

    return htilde, stilde, psd, N


@pytest.mark.parametrize("device", DEVICES)
def test_candidate_buffer(device):
    buf = CandidateBuffer(capacity=100, device=device)
    assert buf.capacity == 100
    assert buf.count == 0
    assert not buf.has_overflow

    buf.reset()
    assert buf.count == 0


@pytest.mark.parametrize("device", DEVICES)
def test_engine_lifecycle_and_core_parity(device):
    htilde, stilde, psd, N = _make_test_fixtures()
    templates = [htilde]

    bank_plan = prepare_bank(templates, tile_size=1, device=device)
    psd_plan = bind_psd(bank_plan, psd, device=device)

    policy = SelectionPolicy(
        snr_threshold=0.0,  # Record peak
        snr_abort_threshold=1000.0,
        cluster_policy="live_peak",
    )

    engine = SearchEngine(bank_plan, policy, device=device)

    # Reference PyCBC matched_filter_core (which returns unnormalized time series _q and norm)
    ref_snr_series, ref_corr, ref_norm = matched_filter_core(htilde, stilde, psd=psd)
    ref_snr_vals = ref_snr_series.numpy() * ref_norm

    # Valid window: e.g. [100, 900] out of 1024
    valid_start = 100
    valid_end = 900
    ticket = engine.submit(stilde, psd_plan, valid_interval=(valid_start, valid_end))

    assert ticket.completed
    assert not ticket.aborted
    assert not ticket.overflow

    ready_batches = engine.drain()
    assert len(ready_batches) == 1
    committed = ready_batches[0]
    assert len(committed.results) == 1

    cands = committed.results[0]
    assert len(cands["template_id"]) == 1
    assert cands["template_id"][0] == 0

    # Check peak sample index and value against reference
    cand_sample = cands["sample_idx"][0]
    cand_snr = cands["snr"][0]

    # Verify that engine's internal out_workspace matches ref_snr_vals
    engine_snr_full = engine.out_workspace[0]
    if hasattr(engine_snr_full, "cpu"):
        engine_snr_np = engine_snr_full.detach().cpu().numpy()
    else:
        engine_snr_np = np.asarray(engine_snr_full)

    norm_val = psd_plan.tile_norms[0][0]
    if hasattr(norm_val, "item"):
        norm_val = norm_val.item()

    engine_snr_np = engine_snr_np * norm_val
    np.testing.assert_allclose(engine_snr_np, ref_snr_vals, atol=1e-4)

    # In valid window, find argmax of reference
    ref_window = ref_snr_vals[valid_start:valid_end]
    ref_peak_idx = np.argmax(np.abs(ref_window) ** 2) + valid_start
    ref_peak_val = ref_snr_vals[ref_peak_idx]

    assert cand_sample == ref_peak_idx
    np.testing.assert_allclose(cand_snr, ref_peak_val, atol=1e-4)

    engine.close()


@pytest.mark.parametrize("device", DEVICES)
def test_engine_abort_threshold(device):
    htilde, stilde, psd, N = _make_test_fixtures()
    templates = [htilde]

    bank_plan = prepare_bank(templates, tile_size=1, device=device)
    psd_plan = bind_psd(bank_plan, psd, device=device)

    # Set threshold to 0.0 and abort threshold very low (e.g. 1.0) so it triggers abort
    policy = SelectionPolicy(
        snr_threshold=0.0,
        snr_abort_threshold=1.0,
        cluster_policy="live_peak",
    )

    engine = SearchEngine(bank_plan, policy, device=device)
    ticket = engine.submit(stilde, psd_plan, valid_interval=(100, 900))

    assert ticket.completed
    assert ticket.aborted
    assert len(ticket.results) == 0

    engine.close()
