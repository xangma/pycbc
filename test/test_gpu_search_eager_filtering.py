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

from pycbc.types import FrequencySeries, TimeSeries
from pycbc.filter.matchedfilter import matched_filter_core
from pycbc.filter.gpu_search.plans import prepare_bank, bind_psd
from pycbc.filter.gpu_search.candidates import (
    CandidateBuffer,
    SelectionPolicy,
    select_tile_candidates,
    _store_in_buffer,
)
from pycbc.filter.gpu_search.engine import SearchEngine

try:
    from pycbc.events.threshold_torch import threshold_and_cluster
except ImportError:
    threshold_and_cluster = None


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


@pytest.mark.parametrize("device", DEVICES)
def test_candidate_buffer_resize(device):
    buf = CandidateBuffer(capacity=4, device=device)
    assert buf.capacity == 4

    # Populate 3 entries
    if torch is not None and device != "numpy":
        dev = torch.device(device)
        sel_tmplt = torch.tensor([0, 1, 2], dtype=torch.int64, device=dev)
        sel_sample = torch.tensor([10, 20, 30], dtype=torch.int64, device=dev)
        sel_snr = torch.tensor([5.0 + 1j, 6.0 + 2j, 7.0 + 3j], dtype=torch.complex64, device=dev)
        sel_sigmasq = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float32, device=dev)
    else:
        sel_tmplt = np.array([0, 1, 2], dtype=np.int64)
        sel_sample = np.array([10, 20, 30], dtype=np.int64)
        sel_snr = np.array([5.0 + 1j, 6.0 + 2j, 7.0 + 3j], dtype=np.complex64)
        sel_sigmasq = np.array([100.0, 200.0, 300.0], dtype=np.float32)

    overflow = _store_in_buffer(buf, sel_tmplt, sel_sample, sel_snr, sel_sigmasq, 3)
    assert not overflow
    assert buf.count == 3

    # Resize to 16
    buf.resize(16)
    assert buf.capacity >= 16
    assert buf.count == 3

    d = buf.to_dict()
    np.testing.assert_array_equal(d["template_indices"], [0, 1, 2])
    np.testing.assert_array_equal(d["sample_indices"], [10, 20, 30])
    np.testing.assert_allclose(d["snr"], [5.0 + 1j, 6.0 + 2j, 7.0 + 3j])
    np.testing.assert_allclose(d["sigmasq"], [100.0, 200.0, 300.0])


@pytest.mark.parametrize("device", DEVICES)
def test_candidate_buffer_overflow(device):
    buf = CandidateBuffer(capacity=2, device=device)
    assert buf.capacity == 2

    # Attempt to store 3 records into capacity 2
    if torch is not None and device != "numpy":
        dev = torch.device(device)
        sel_tmplt = torch.tensor([0, 1, 2], dtype=torch.int64, device=dev)
        sel_sample = torch.tensor([10, 20, 30], dtype=torch.int64, device=dev)
        sel_snr = torch.tensor([5.0, 6.0, 7.0], dtype=torch.complex64, device=dev)
        sel_sigmasq = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float32, device=dev)
    else:
        sel_tmplt = np.array([0, 1, 2], dtype=np.int64)
        sel_sample = np.array([10, 20, 30], dtype=np.int64)
        sel_snr = np.array([5.0, 6.0, 7.0], dtype=np.complex64)
        sel_sigmasq = np.array([100.0, 200.0, 300.0], dtype=np.float32)

    overflow = _store_in_buffer(buf, sel_tmplt, sel_sample, sel_snr, sel_sigmasq, 3)
    assert overflow
    assert buf.has_overflow


@pytest.mark.parametrize("device", DEVICES)
def test_symmetric_clustering_parity(device):
    window = 16
    thresh = 1.5
    valid_start = 50
    valid_end = 450
    L = 500

    np.random.seed(42)
    raw_vals = (
        np.random.randn(3, L) + 1j * np.random.randn(3, L)
    ).astype(np.complex64)
    raw_norms = np.array([1.2, 0.8, 1.5], dtype=np.float32)
    raw_sigmasqs = np.array([10.0, 20.0, 15.0], dtype=np.float32)

    if torch is not None and device != "numpy":
        dev = torch.device(device)
        vals = torch.as_tensor(raw_vals, device=dev)
        norms = torch.as_tensor(raw_norms, device=dev)
        sigmasqs = torch.as_tensor(raw_sigmasqs, device=dev)
    else:
        vals = raw_vals
        norms = raw_norms
        sigmasqs = raw_sigmasqs

    policy = SelectionPolicy(
        snr_threshold=thresh,
        cluster_policy="symmetric",
        cluster_window=window,
    )
    res = select_tile_candidates(vals, norms, sigmasqs, valid_start, valid_end, policy)
    assert not res["aborted"]
    assert not res["overflow"]

    if threshold_and_cluster is not None and torch is not None:
        from pycbc import scheme

        valid_raw = raw_vals[:, valid_start:valid_end]
        with scheme.TorchScheme(device):
            for i in range(3):
                series = TimeSeries(valid_raw[i] * raw_norms[i], delta_t=1.0)
                ref_vals, ref_locs = threshold_and_cluster(series, thresh, window)
                ref_locs = np.asarray(ref_locs) + valid_start
                ref_vals = np.asarray(ref_vals)

                cand_mask = res["candidates"]["template_idx"] == i
                cand_locs = res["candidates"]["sample_idx"][cand_mask]
                cand_snrs = res["candidates"]["snr"][cand_mask]

                np.testing.assert_array_equal(cand_locs, ref_locs)
                np.testing.assert_allclose(cand_snrs, ref_vals, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_engine_symmetric_clustering(device):
    htilde, stilde, psd, N = _make_test_fixtures()
    templates = [htilde]

    bank_plan = prepare_bank(templates, tile_size=1, device=device)
    psd_plan = bind_psd(bank_plan, psd, device=device)

    policy = SelectionPolicy(
        snr_threshold=1.5,
        cluster_policy="symmetric",
        cluster_window=32,
    )

    engine = SearchEngine(bank_plan, policy, device=device)
    ticket = engine.submit(stilde, psd_plan, valid_interval=(100, 900))

    assert ticket.completed
    assert not ticket.aborted
    assert not ticket.overflow

    ready = engine.drain()
    assert len(ready) == 1
    cands = ready[0].results
    assert len(cands) == 1
    # Check that sample indices are in valid range
    assert (cands[0]["sample_idx"] >= 100).all()
    assert (cands[0]["sample_idx"] < 900).all()

    engine.close()


@pytest.mark.parametrize("device", DEVICES)
def test_engine_buffer_overflow(device):
    htilde, stilde, psd, N = _make_test_fixtures()
    templates = [htilde]

    bank_plan = prepare_bank(templates, tile_size=1, device=device)
    psd_plan = bind_psd(bank_plan, psd, device=device)

    # Use a low threshold to generate multiple triggers and a buffer with capacity 1
    policy = SelectionPolicy(
        snr_threshold=0.5,
        cluster_policy="symmetric",
        cluster_window=16,
    )

    engine = SearchEngine(bank_plan, policy, device=device)
    engine.candidate_buffer = CandidateBuffer(capacity=1, device=device)

    ticket = engine.submit(stilde, psd_plan, valid_interval=(100, 900))

    assert ticket.completed
    assert ticket.overflow

    engine.close()
