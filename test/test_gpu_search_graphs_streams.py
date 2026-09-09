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
Unit tests for CUDA graphs, asynchronous streams, and double-buffering
in the persistent GPU search engine.
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
    SelectionPolicy,
    SearchEngine,
    CUDAGraphManager,
)

CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA_AVAILABLE else [])


def _make_bank_and_data(
    num_templates=5,
    size=512,
    tile_size=2,
    device="cpu",
):
    flen = size // 2 + 1
    delta_f = 1.0 / size
    rng = np.random.default_rng(12345)

    templates = []
    for i in range(num_templates):
        h_vals = (
            rng.normal(size=flen) + 1j * rng.normal(size=flen)
        ).astype(np.complex64)
        h_vals[0] = 0.0
        t = FrequencySeries(h_vals, delta_f=delta_f)
        t.id = 100 + i
        templates.append(t)

    bank_plan = prepare_bank(
        templates,
        tile_size=tile_size,
        device=device,
    )

    psd_vals = np.ones(flen, dtype=np.float32) * 2.0
    psd = FrequencySeries(psd_vals, delta_f=delta_f)
    psd_plan = bind_psd(bank_plan, psd, psd_version="psd-v1", device=device)

    data_vals = (
        rng.normal(size=flen) + 1j * rng.normal(size=flen)
    ).astype(np.complex64)
    data_vals[0] = 0.0
    data = FrequencySeries(data_vals, delta_f=delta_f)

    return bank_plan, psd_plan, data, templates


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_graph_lifecycle_and_parity():
    """Verify CUDA graph capture, replay counts, and numerical parity with eager filtering."""
    bank_plan, psd_plan, data, _ = _make_bank_and_data(
        num_templates=4, size=512, tile_size=2, device="cuda"
    )
    policy = SelectionPolicy(snr_threshold=0.0)
    valid_interval = (64, 448)

    # 1. Eager engine
    eager_engine = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device="cuda",
        use_cuda_graphs=False,
    )
    _ = eager_engine.submit(
        data, psd_plan, valid_interval=valid_interval
    )
    res_eager = eager_engine.drain()

    # 2. Graph engine
    graph_engine = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device="cuda",
        use_cuda_graphs=True,
    )
    assert graph_engine.graph_stats["capture_count"] == 0
    assert graph_engine.graph_stats["replay_count"] == 0

    # First submission: warms up and captures graphs
    _ = graph_engine.submit(
        data, psd_plan, valid_interval=valid_interval
    )
    res_g1 = graph_engine.drain()

    num_tiles = len(bank_plan.tiles)
    assert graph_engine.graph_stats["capture_count"] == num_tiles
    assert graph_engine.graph_stats["replay_count"] == num_tiles

    # Compare 1st run results with eager
    assert len(res_g1[0].results) == len(res_eager[0].results)
    for r_g, r_e in zip(res_g1[0].results, res_eager[0].results):
        np.testing.assert_array_equal(r_g["template_id"], r_e["template_id"])
        np.testing.assert_array_equal(r_g["sample_idx"], r_e["sample_idx"])
        np.testing.assert_allclose(r_g["snr"], r_e["snr"], atol=1e-5)

    # Second submission with new data: replays existing graphs without new captures
    rng = np.random.default_rng(999)
    flen = bank_plan.geometry.filter_length
    data2_vals = (
        rng.normal(size=flen) + 1j * rng.normal(size=flen)
    ).astype(np.complex64)
    data2_vals[0] = 0.0
    data2 = FrequencySeries(data2_vals, delta_f=bank_plan.geometry.delta_f)

    _ = eager_engine.submit(
        data2, psd_plan, valid_interval=valid_interval
    )
    res_eager2 = eager_engine.drain()

    _ = graph_engine.submit(
        data2, psd_plan, valid_interval=valid_interval
    )
    res_g2 = graph_engine.drain()

    # Capture count should NOT increase; replay count should double
    assert graph_engine.graph_stats["capture_count"] == num_tiles
    assert graph_engine.graph_stats["replay_count"] == 2 * num_tiles

    # Verify parity on 2nd run
    for r_g, r_e in zip(res_g2[0].results, res_eager2[0].results):
        np.testing.assert_array_equal(r_g["template_id"], r_e["template_id"])
        np.testing.assert_array_equal(r_g["sample_idx"], r_e["sample_idx"])
        np.testing.assert_allclose(r_g["snr"], r_e["snr"], atol=1e-5)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_graph_tail_tiles_variable_batch():
    """Verify CUDA graph captures correctly handle uneven tail tiles."""
    # 5 templates with tile_size=2 results in 3 tiles: sizes 2, 2, 1
    bank_plan, psd_plan, data, _ = _make_bank_and_data(
        num_templates=5, size=512, tile_size=2, device="cuda"
    )
    assert len(bank_plan.tiles) == 3
    assert [t.batch_size for t in bank_plan.tiles] == [2, 2, 1]

    policy = SelectionPolicy(snr_threshold=0.0)
    engine = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device="cuda",
        use_cuda_graphs=True,
    )
    _ = engine.submit(data, psd_plan, valid_interval=(64, 448))
    res = engine.drain()

    assert engine.graph_stats["capture_count"] == 3
    assert engine.graph_stats["replay_count"] == 3
    assert len(res[0].results) == 3


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_graph_fork_safety():
    """Verify that process fork invalidates cached CUDA graphs."""
    manager = CUDAGraphManager(enabled=True, device="cuda")
    assert manager._captured_pid > 0

    # Simulate process fork by altering captured PID
    manager._cache[("dummy_key",)] = "dummy_entry"
    manager._captured_pid = 99999999

    manager.check_fork_safety()
    assert len(manager._cache) == 0
    assert manager.invalidation_count >= 1


def test_cuda_graph_fallback_on_cpu():
    """Verify that enabling graphs on CPU falls back cleanly to eager execution."""
    bank_plan, psd_plan, data, _ = _make_bank_and_data(
        num_templates=3, size=256, tile_size=2, device="cpu"
    )
    policy = SelectionPolicy(snr_threshold=0.0)
    engine = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device="cpu",
        use_cuda_graphs=True,
    )
    assert not engine.graph_manager.is_available()

    _ = engine.submit(data, psd_plan, valid_interval=(32, 224))
    res = engine.drain()

    assert len(res) == 1
    assert engine.graph_stats["capture_count"] == 0
    assert engine.graph_stats["replay_count"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_double_buffering_and_async_streams(device):
    """Verify double-buffering with multi-workspace concurrency."""
    bank_plan, psd_plan, data, _ = _make_bank_and_data(
        num_templates=4, size=512, tile_size=2, device=device
    )
    policy = SelectionPolicy(snr_threshold=0.0)
    valid_interval = (64, 448)

    # Search engine with 2 workspace slots and async transfers enabled
    engine = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device=device,
        num_workspaces=2,
        enable_async_transfers=True,
        use_cuda_graphs=(device == "cuda"),
    )
    assert engine.num_workspaces == 2

    # Submit 3 blocks consecutively before draining
    t1 = engine.submit(data, psd_plan, valid_interval, block_id=1)
    assert t1.slot_idx == 0

    t2 = engine.submit(data, psd_plan, valid_interval, block_id=2)
    assert t2.slot_idx == 1

    t3 = engine.submit(data, psd_plan, valid_interval, block_id=3)
    assert t3.slot_idx == 0  # Rolled over to slot 0

    ready = engine.drain()
    assert len(ready) == 3
    assert [t.block_id for t in ready] == [1, 2, 3]

    # Verify each ticket has valid candidate results
    for t in ready:
        assert len(t.results) == len(bank_plan.tiles)
        total_cands = sum(len(r["template_id"]) for r in t.results)
        assert total_cands == 4


@pytest.mark.parametrize("device", DEVICES)
def test_delayed_consumer_and_flush(device):
    """Verify delayed drain and flush lifecycle across multiple queued blocks."""
    bank_plan, psd_plan, data, _ = _make_bank_and_data(
        num_templates=3, size=256, tile_size=2, device=device
    )
    policy = SelectionPolicy(snr_threshold=0.0)
    valid_interval = (32, 224)

    engine = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device=device,
        num_workspaces=2,
    )

    for i in range(5):
        engine.submit(data, psd_plan, valid_interval, block_id=i)

    # Flush all queued blocks
    flushed = engine.flush()
    assert len(flushed) == 5
    assert [t.block_id for t in flushed] == [0, 1, 2, 3, 4]
    for t in flushed:
        assert t.completed
