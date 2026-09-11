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

from pycbc import scheme
from pycbc.types import FrequencySeries
from pycbc.types.backend import backend_array
from pycbc.filter.matchedfilter import get_cutoff_indices
from pycbc.filter.gpu_search import (
    prepare_bank,
    bind_psd,
    SelectionPolicy,
    SearchEngine,
    CUDAGraphManager,
    WorkspaceBudget,
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
    """Verify CUDA graph capture, replay counts, and numerical parity with
    eager filtering.
    """
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


@pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA required for persistent cache"
)
def test_reciprocal_psd_caching_and_invalidation(monkeypatch):
    """Verify reciprocal PSD is cached across submissions on CUDA and safely
    invalidated upon mutation, reassignment, new plan, grad, or inference.
    """
    bank_plan, psd_plan, data, _ = _make_bank_and_data(
        num_templates=2, size=256, tile_size=2, device="cuda"
    )
    policy = SelectionPolicy(snr_threshold=100.0)  # Empty candidates
    valid_interval = (32, 224)

    engine = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device="cuda",
        num_workspaces=2,
    )

    # Instrument torch.where to count reciprocal calculations
    orig_where = torch.where
    where_count = [0]

    def counted_where(*args, **kwargs):
        where_count[0] += 1
        return orig_where(*args, **kwargs)

    monkeypatch.setattr(torch, "where", counted_where)

    # 1. First submission: computes and caches reciprocal
    where_count[0] = 0
    engine.submit(data, psd_plan, valid_interval, block_id=1)
    assert where_count[0] == 1
    cached_inv = engine._cached_inv_psd
    assert cached_inv is not None

    # 2. Second submission with unchanged PSDPlan: reuses cached reciprocal
    engine.submit(data, psd_plan, valid_interval, block_id=2)
    assert where_count[0] == 1  # Reused without recomputing!
    assert engine._cached_inv_psd is cached_inv

    # 3. In-place PSD mutation: increments _version, triggers recomputation
    psd_plan.psd_data[10] = 999.0
    engine.submit(data, psd_plan, valid_interval, block_id=3)
    assert where_count[0] == 2  # Recomputed!
    assert engine._cached_inv_psd is not cached_inv
    np.testing.assert_allclose(
        float(engine._cached_inv_psd[10].cpu()), 1.0 / 999.0, rtol=1e-5
    )

    # 4. Reassignment of psd_data: source identity mismatch triggers recomputation
    psd_plan.psd_data = torch.ones_like(psd_plan.psd_data) * 4.0
    engine.submit(data, psd_plan, valid_interval, block_id=4)
    assert where_count[0] == 3  # Recomputed!
    np.testing.assert_allclose(
        float(engine._cached_inv_psd[10].cpu()), 0.25, rtol=1e-5
    )

    # 5. New PSDPlan: plan identity mismatch triggers recomputation
    flen = engine.flen
    delta_f = bank_plan.geometry.delta_f
    fresh_psd = FrequencySeries(
        np.full(flen, 8.0, dtype=np.float32), delta_f=delta_f
    )
    fresh_plan = bind_psd(
        bank_plan, fresh_psd, psd_version="v-fresh", device="cuda"
    )
    engine.submit(data, fresh_plan, valid_interval, block_id=5)
    assert where_count[0] == 4  # Recomputed!
    np.testing.assert_allclose(
        float(engine._cached_inv_psd[10].cpu()), 0.125, rtol=1e-5
    )

    # 6. Untrackable tensor (requires_grad): recomputes and clears cache
    grad_psd = torch.ones(
        flen, dtype=torch.float32, device="cuda", requires_grad=True
    )
    grad_plan = bind_psd(
        bank_plan, fresh_psd, psd_version="v-grad", device="cuda"
    )
    grad_plan.psd_data = grad_psd
    engine.submit(data, grad_plan, valid_interval, block_id=6)
    assert engine._cached_inv_psd is None  # Cleared!

    # 7. Inference tensor: does not read _version, recomputes, clears cache
    with torch.inference_mode():
        inf_psd = torch.full((flen,), 16.0, dtype=torch.float32, device="cuda")
    inf_plan = bind_psd(
        bank_plan, fresh_psd, psd_version="v-inf", device="cuda"
    )
    inf_plan.psd_data = inf_psd
    where_count[0] = 0
    engine.submit(data, inf_plan, valid_interval, block_id=7)
    assert where_count[0] == 1
    assert engine._cached_inv_psd is None  # Cleared!

    # Mutate inference tensor values and submit again
    with torch.inference_mode():
        inf_psd2 = torch.full(
            (flen,), 32.0, dtype=torch.float32, device="cuda"
        )
    inf_plan.psd_data = inf_psd2
    engine.submit(data, inf_plan, valid_interval, block_id=8)
    assert where_count[0] == 2
    assert engine._cached_inv_psd is None

    engine.close()


@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.skipif(torch is None, reason="PyTorch required")
def test_cpu_and_numpy_psd_external_mutation_fallback(as_tensor):
    """Verify CPU and NumPy PSDs do not persistently cache and external
    mutations to aliased memory are immediately observed in output.
    """
    bank_plan, psd_plan, data, _ = _make_bank_and_data(
        num_templates=2, size=256, tile_size=2, device="cpu"
    )
    valid_interval = (32, 224)
    engine = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=SelectionPolicy(snr_threshold=0.0),
        device="cpu",
        num_workspaces=1,
    )

    # Verify no persistent cache is retained on CPU
    engine.submit(data, psd_plan, valid_interval, block_id=1)
    assert engine._cached_inv_psd is None
    engine.drain()

    # External mutation via from_numpy aliased memory
    flen = engine.flen
    psd_np = np.full(flen, 2.0, dtype=np.float32)
    psd_plan.psd_data = torch.from_numpy(psd_np) if as_tensor else psd_np

    engine.submit(data, psd_plan, valid_interval, block_id=2)
    res_before = engine.drain()[0].results[0]["snr"]

    # External in-place mutation to underlying NumPy buffer
    psd_np[10:50] = 50.0  # Does NOT increment Torch tensor._version!

    engine.submit(data, psd_plan, valid_interval, block_id=3)
    res_after = engine.drain()[0].results[0]["snr"]

    # Must produce different output due to CPU recomputation fallback
    assert not np.allclose(res_before, res_after)

    # Verify output matches a fresh engine given the same mutated plan
    fresh_engine = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=SelectionPolicy(snr_threshold=0.0),
        device="cpu",
        num_workspaces=1,
    )
    fresh_engine.submit(data, psd_plan, valid_interval, block_id=1)
    fresh_res = fresh_engine.drain()[0].results[0]["snr"]
    np.testing.assert_allclose(res_after, fresh_res, atol=1e-5)

    engine.close()
    fresh_engine.close()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.skipif(torch is None, reason="PyTorch required")
def test_inference_psd_mutation_recomputes(device):
    bank, psd, data, _ = _make_bank_and_data(device=device)
    engine = SearchEngine(
        bank, SelectionPolicy(snr_threshold=0), device=device
    )
    with torch.inference_mode():
        psd.psd_data = torch.full((engine.flen,), 2.0, device=device)
    engine.submit(data, psd, (64, 448))
    before = engine.drain()[0].results[0]["snr"].copy()
    with torch.inference_mode():
        psd.psd_data.fill_(4.0)
    engine.submit(data, psd, (64, 448))
    after = engine.drain()[0].results[0]["snr"]
    assert engine._cached_inv_psd is None
    assert np.any(np.abs(before) > 0)
    np.testing.assert_array_equal(after * 2, before)
    engine.close()


@pytest.mark.parametrize("device", DEVICES)
def test_budgeted_engine_disables_cache_and_matches_unbudgeted(device):
    """Verify that specifying max_workspace_bytes disables persistent cache
    while producing numerically identical results.
    """
    bank_plan, psd_plan, data, _ = _make_bank_and_data(
        num_templates=2, size=256, tile_size=2, device=device
    )
    valid_interval = (32, 224)
    policy = SelectionPolicy(snr_threshold=0.0)

    # Unbudgeted engine
    engine_unbudgeted = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device=device,
        num_workspaces=2,
    )
    engine_unbudgeted.submit(data, psd_plan, valid_interval, block_id=1)
    res_unbudgeted = engine_unbudgeted.drain()[0].results[0]["snr"]

    # Budgeted engine with explicit max_workspace_bytes
    budget = WorkspaceBudget(max_workspace_bytes=100 * 1024 * 1024)
    engine_budgeted = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device=device,
        num_workspaces=2,
        workspace_budget=budget,
    )
    engine_budgeted.submit(data, psd_plan, valid_interval, block_id=1)
    assert engine_budgeted._cached_inv_psd is None  # Cache not retained!
    res_budgeted = engine_budgeted.drain()[0].results[0]["snr"]

    # Numerical parity
    np.testing.assert_allclose(res_budgeted, res_unbudgeted, atol=1e-5)

    engine_unbudgeted.close()
    engine_budgeted.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_cuda_device_normalization_wrapper_and_bare_tensors(monkeypatch):
    """Verify CUDA FrequencySeries wrapper and bare tensor inputs with
    device='cuda' vs concrete 'cuda:0', 2 slots, and separate caller streams.
    """
    bank_plan, psd_plan, data, _ = _make_bank_and_data(
        num_templates=2, size=256, tile_size=2, device="cuda"
    )
    valid_interval = (32, 224)
    policy = SelectionPolicy(snr_threshold=0.0)

    # Engine initialized with generic 'cuda'
    engine = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device="cuda",
        num_workspaces=2,
        enable_async_transfers=True,
    )
    with scheme.TorchScheme("cuda"):
        data = FrequencySeries(data.numpy(), delta_f=data.delta_f)
        bare_cuda = backend_array(data)
        assert isinstance(bare_cuda, torch.Tensor) and bare_cuda.is_cuda

        def forbid_host_copy():
            pytest.fail("CUDA input must remain on its device")

        monkeypatch.setattr(data, "numpy", forbid_host_copy)
        torch.cuda.synchronize()

        stream_0 = torch.cuda.Stream()
        stream_1 = torch.cuda.Stream()

        # 1. FrequencySeries wrapper with native CUDA storage on cuda:0
        with torch.cuda.stream(stream_0):
            t1 = engine.submit(data, psd_plan, valid_interval, block_id=1)
        assert t1.slot_idx == 0

        # 2. Bare CUDA tensor input on stream_1 (slot 1)
        with torch.cuda.stream(stream_1):
            t2 = engine.submit(bare_cuda, psd_plan, valid_interval, block_id=2)
        assert t2.slot_idx == 1

        ready = engine.drain()
        assert len(ready) == 2
        np.testing.assert_array_equal(
            ready[0].results[0]["snr"], ready[1].results[0]["snr"]
        )

        # In-place PSD mutation on CUDA tensor: verify output matches fresh engine
        with torch.cuda.stream(stream_0):
            psd_plan.psd_data[15] = 200.0
            engine.submit(data, psd_plan, valid_interval, block_id=3)
        mutated_res = engine.drain()[0].results[0]["snr"]

        fresh_engine = SearchEngine(
            bank_plan=bank_plan,
            selection_policy=policy,
            device="cuda",
            num_workspaces=1,
        )
        fresh_engine.submit(data, psd_plan, valid_interval, block_id=1)
        fresh_res = fresh_engine.drain()[0].results[0]["snr"]
        np.testing.assert_allclose(mutated_res, fresh_res, atol=1e-5)

        engine.close()
        fresh_engine.close()


@pytest.mark.parametrize("device", DEVICES)
def test_cross_stream_and_multi_workspace_readiness(device):
    """Verify reciprocal reuse across workspace slots and CUDA caller streams."""
    bank_plan, psd_plan, data, _ = _make_bank_and_data(
        num_templates=2, size=256, tile_size=2, device=device
    )
    policy = SelectionPolicy(snr_threshold=0.0)
    valid_interval = (32, 224)

    engine = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device=device,
        num_workspaces=2,
    )

    is_cuda = (
        device == "cuda" and torch is not None and torch.cuda.is_available()
    )
    stream_a = torch.cuda.Stream() if is_cuda else None
    stream_b = torch.cuda.Stream() if is_cuda else None

    # Submit on stream A (slot 0)
    if is_cuda:
        with torch.cuda.stream(stream_a):
            t1 = engine.submit(data, psd_plan, valid_interval, block_id=1)
    else:
        t1 = engine.submit(data, psd_plan, valid_interval, block_id=1)
    assert t1.slot_idx == 0

    # Submit on stream B (slot 1) - tests cross-stream ready-event dependency
    if is_cuda:
        with torch.cuda.stream(stream_b):
            t2 = engine.submit(data, psd_plan, valid_interval, block_id=2)
    else:
        t2 = engine.submit(data, psd_plan, valid_interval, block_id=2)
    assert t2.slot_idx == 1

    ready = engine.drain()
    assert len(ready) == 2
    assert [t.block_id for t in ready] == [1, 2]
    np.testing.assert_allclose(
        ready[0].results[0]["snr"], ready[1].results[0]["snr"], rtol=1e-5
    )

    engine.close()


@pytest.mark.parametrize("device", DEVICES)
def test_caller_input_unchanged_and_scaling(device):
    """Verify caller input data is unmodified and strain_scale=1 avoids copy."""
    if torch is None:
        pytest.skip("PyTorch not available")

    bank_plan, psd_plan, data, _ = _make_bank_and_data(
        num_templates=2, size=256, tile_size=2, device=device
    )
    policy = SelectionPolicy(snr_threshold=100.0)
    valid_interval = (32, 224)

    engine = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device=device,
        num_workspaces=2,
    )

    flen = engine.flen
    dev = torch.device(device)
    raw_tensor = torch.ones(flen, dtype=torch.complex64, device=dev)
    expected_bytes = raw_tensor.detach().cpu().numpy().tobytes()

    # Submit raw device tensor with strain_scale == 1.0
    engine.submit(raw_tensor, psd_plan, valid_interval, block_id=1)
    engine.drain()

    # Verify caller's input tensor is strictly unchanged (no cutoff zeros applied)
    assert raw_tensor.detach().cpu().numpy().tobytes() == expected_bytes

    # Also verify already_overwhitened=True leaves caller input untouched
    engine.submit(
        raw_tensor,
        psd_plan,
        valid_interval,
        block_id=2,
        already_overwhitened=True,
    )
    engine.drain()
    assert raw_tensor.detach().cpu().numpy().tobytes() == expected_bytes

    engine.close()


@pytest.mark.parametrize("device", ["numpy"] + DEVICES)
@pytest.mark.parametrize("overwhitened", [False, True])
def test_fullband_veto_buffer_conditional_skip(device, overwhitened):
    """Verify full-band clone is skipped when veto_manager is None,
    retained when present.
    """
    bank_plan, psd_plan, data, _ = _make_bank_and_data(
        num_templates=2, size=256, tile_size=2, device=device
    )
    data[0] = 1.0 + 2.0j
    valid_interval = (32, 224)
    policy = SelectionPolicy(snr_threshold=0.0)

    # 1. Without veto manager: full-band clone skipped
    engine_no_veto = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device=device,
        num_workspaces=1,
    )
    assert engine_no_veto.veto_manager is None
    engine_no_veto.submit(
        data,
        psd_plan,
        valid_interval,
        block_id=1,
        already_overwhitened=overwhitened,
    )
    res1 = engine_no_veto.drain()
    assert len(res1) == 1
    engine_no_veto.close()

    # 2. With mock veto manager: full-band uncut data retained and passed
    class MockVetoManager:
        def __init__(self):
            self.eval_calls = []

        def evaluate(self, **kwargs):
            self.eval_calls.append(kwargs)
            return kwargs["candidates"]

    engine_veto = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=policy,
        device=device,
        num_workspaces=1,
    )
    mock_vm = MockVetoManager()
    engine_veto.veto_manager = mock_vm

    engine_veto.submit(
        data,
        psd_plan,
        valid_interval,
        block_id=2,
        already_overwhitened=overwhitened,
    )
    res2 = engine_veto.drain()
    assert len(res2) == 1
    assert len(mock_vm.eval_calls) > 0

    call_kwargs = mock_vm.eval_calls[0]
    stilde_cut = call_kwargs["stilde"]
    full_stilde = call_kwargs["full_stilde"]
    full_np = (
        full_stilde.cpu().numpy()
        if torch is not None and isinstance(full_stilde, torch.Tensor)
        else full_stilde
    )
    np.testing.assert_array_equal(
        full_np, data.numpy() if overwhitened else data.numpy() / 2
    )

    kmin, kmax = get_cutoff_indices(
        bank_plan.geometry.f_lower,
        bank_plan.geometry.f_upper,
        bank_plan.geometry.delta_f,
        engine_veto.tlen,
    )

    # stilde must have cutoff zeros applied
    if torch is not None and isinstance(stilde_cut, torch.Tensor):
        assert torch.all(stilde_cut[:kmin] == 0)
        # full_stilde must retain uncut values
        assert not torch.all(full_stilde[:kmin] == 0)
    else:
        assert np.all(stilde_cut[:kmin] == 0)
        assert not np.all(full_stilde[:kmin] == 0)

    engine_veto.close()


@pytest.mark.parametrize("device", DEVICES)
def test_ticket_identity_and_lifetime(device):
    """Verify distinct ticket identities across empty and triggered submissions."""
    bank_plan, psd_plan, data, _ = _make_bank_and_data(
        num_templates=2, size=256, tile_size=2, device=device
    )
    valid_interval = (32, 224)

    engine = SearchEngine(
        bank_plan=bank_plan,
        selection_policy=SelectionPolicy(snr_threshold=0.0),
        device=device,
        num_workspaces=2,
    )

    policy_empty = SelectionPolicy(snr_threshold=1000.0)
    policy_triggered = SelectionPolicy(snr_threshold=0.0)

    engine.selection_policy = policy_empty
    t_empty = engine.submit(data, psd_plan, valid_interval, block_id=10)

    engine.selection_policy = policy_triggered
    t_trig = engine.submit(data, psd_plan, valid_interval, block_id=20)

    assert t_empty is not t_trig
    assert t_empty.ticket_id != t_trig.ticket_id
    assert t_empty.block_id == 10
    assert t_trig.block_id == 20

    ready = engine.drain()
    assert len(ready) == 2
    assert ready[0] is t_empty
    assert ready[1] is t_trig

    # Empty ticket has 0 survivors
    assert all(len(r["template_id"]) == 0 for r in ready[0].results)
    # Triggered ticket has > 0 survivors
    assert any(len(r["template_id"]) > 0 for r in ready[1].results)

    engine.close()
