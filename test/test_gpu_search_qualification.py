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
End-to-end qualification tests for the PyCBC persistent GPU search engine.
Verifies scientific output parity, streaming stability, latency bounds,
and bounded device memory under sustained load.
"""

import time
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
    prepare_power_chisq_plan,
    VetoManager,
    TiledLiveBatchMatchedFilter,
)

CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA_AVAILABLE else [])


def _make_qualification_fixtures(num_templates=8, size=512, tile_size=4, device="cpu"):
    flen = size // 2 + 1
    delta_f = 1.0 / size
    rng = np.random.default_rng(20260910)

    templates = []
    for i in range(num_templates):
        h_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(np.complex64)
        h_vals[0] = 0.0
        pwr = np.sum(np.abs(h_vals) ** 2)
        if pwr > 0:
            h_vals /= np.sqrt(pwr)
        t = FrequencySeries(h_vals, delta_f=delta_f)
        t.id = 100 + i
        templates.append(t)

    bank_plan = prepare_bank(templates, tile_size=tile_size, device=device)

    psd_vals = np.ones(flen, dtype=np.float32) * 2.0
    psd = FrequencySeries(psd_vals, delta_f=delta_f)
    psd_plan = bind_psd(bank_plan, psd, psd_version="psd-v1", device=device)

    # Signal + Gaussian noise data
    data_vals = (rng.normal(size=flen) + 1j * rng.normal(size=flen)).astype(np.complex64)
    data_vals[0] = 0.0
    # Inject template 0 with SNR ~ 10
    data_vals += templates[0].numpy() * 10.0
    data = FrequencySeries(data_vals, delta_f=delta_f)
    data.psd = psd

    return bank_plan, psd_plan, data, templates


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_scientific_output_and_veto_parity_cuda():
    """Verify that CUDA graph engine with Power Chisq vetoes achieves exact scientific parity with CPU reference."""
    # 1. CPU Reference Run
    bank_cpu, psd_cpu, data_cpu, _ = _make_qualification_fixtures(
        num_templates=8, size=512, tile_size=4, device="cpu"
    )
    chisq_plan_cpu = prepare_power_chisq_plan(
        bank_cpu, psd_plan=psd_cpu, num_bins=8, device="cpu"
    )
    veto_mgr_cpu = VetoManager(power_chisq_plan=chisq_plan_cpu)
    policy = SelectionPolicy(snr_threshold=4.0, cluster_policy="live_peak")
    valid_interval = (64, 448)

    engine_cpu = SearchEngine(bank_cpu, policy, veto_manager=veto_mgr_cpu, device="cpu")
    engine_cpu.submit(data_cpu, psd_cpu, valid_interval)
    res_cpu = engine_cpu.drain()

    # 2. CUDA Graph Engine Run
    bank_cuda, psd_cuda, data_cuda, _ = _make_qualification_fixtures(
        num_templates=8, size=512, tile_size=4, device="cuda"
    )
    chisq_plan_cuda = prepare_power_chisq_plan(
        bank_cuda, psd_plan=psd_cuda, num_bins=8, device="cuda"
    )
    veto_mgr_cuda = VetoManager(power_chisq_plan=chisq_plan_cuda)

    engine_cuda = SearchEngine(
        bank_cuda,
        policy,
        veto_manager=veto_mgr_cuda,
        device="cuda",
        use_cuda_graphs=True,
        num_workspaces=2,
    )
    engine_cuda.submit(data_cuda, psd_cuda, valid_interval)
    res_cuda = engine_cuda.drain()

    # Verify triggers match
    assert len(res_cpu) == 1 and len(res_cuda) == 1
    cpu_triggers = res_cpu[0].results
    cuda_triggers = res_cuda[0].results
    assert len(cpu_triggers) == len(cuda_triggers)

    for r_cpu, r_cuda in zip(cpu_triggers, cuda_triggers):
        np.testing.assert_array_equal(r_cuda["template_id"], r_cpu["template_id"])
        np.testing.assert_array_equal(r_cuda["sample_idx"], r_cpu["sample_idx"])
        np.testing.assert_allclose(r_cuda["snr"], r_cpu["snr"], atol=1e-4)
        if "chisq" in r_cuda and "chisq" in r_cpu:
            np.testing.assert_allclose(r_cuda["chisq"], r_cpu["chisq"], atol=1e-3)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_sustained_streaming_vram_boundedness():
    """Verify zero memory growth and strict latency bounds over sustained streaming blocks."""
    bank_plan, psd_plan, data, _ = _make_qualification_fixtures(
        num_templates=16, size=512, tile_size=8, device="cuda"
    )
    policy = SelectionPolicy(snr_threshold=5.0, cluster_policy="live_peak")
    valid_interval = (64, 448)

    engine = SearchEngine(
        bank_plan,
        policy,
        device="cuda",
        use_cuda_graphs=True,
        num_workspaces=2,
        enable_async_transfers=True,
    )

    # Warmup
    for _ in range(5):
        engine.submit(data, psd_plan, valid_interval)
        engine.drain()
    torch.cuda.synchronize()

    initial_allocated = torch.cuda.memory_allocated()

    # Run 40 consecutive blocks
    times = []
    for i in range(40):
        t0 = time.perf_counter()
        engine.submit(data, psd_plan, valid_interval, block_id=i)
        ready = engine.drain()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        assert len(ready) == 1

    final_allocated = torch.cuda.memory_allocated()
    engine.close()

    # Verify zero memory leak
    assert final_allocated <= initial_allocated + 4096  # within small page rounding
    # Verify latency meets < 50ms requirement for 512-point batches on RTX 4090
    p99_latency = np.percentile(times, 99)
    assert p99_latency < 50.0  # ms


@pytest.mark.parametrize("device", DEVICES)
def test_live_adapter_sustained_workflow(device):
    """Verify TiledLiveBatchMatchedFilter over sequential streaming chunks."""
    bank_plan, psd_plan, data, templates = _make_qualification_fixtures(
        num_templates=6, size=256, tile_size=2, device=device
    )

    import types
    data_reader = types.SimpleNamespace(
        overwhitened_data=lambda _df: data,
        trim_padding=16,
        blocksize=128,
        sample_rate=1,
        start_time=10000.0,
    )

    live_filter = TiledLiveBatchMatchedFilter(
        templates=templates,
        snr_threshold=0.0,
        chisq_bins=8,
        sg_chisq=types.SimpleNamespace(do=False),
        tile_size=2,
        device=device,
        use_cuda_graphs=(device == "cuda"),
    )

    res = live_filter.process_data(data_reader)
    assert res is not False
    assert len(res["template_id"]) == 6
    assert len(res["snr"]) == 6
