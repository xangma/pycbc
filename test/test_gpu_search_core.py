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
Unit tests for the shared tiled correlation/IFFT/workspace core
(FilteringWorkspaceSlot, FilteringWorkspace, correlate_and_ifft).
"""

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

from pycbc.filter.gpu_search.core import (
    FilteringWorkspaceSlot,
    FilteringWorkspace,
    correlate_and_ifft,
)

CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA_AVAILABLE else [])


def _generate_fixtures(batch_size=4, tlen=512, seed=42):
    rng = np.random.default_rng(seed)
    flen = tlen // 2 + 1
    # Complex templates
    h_real = rng.normal(size=(batch_size, flen)).astype(np.float32)
    h_imag = rng.normal(size=(batch_size, flen)).astype(np.float32)
    templates_np = (h_real + 1j * h_imag).astype(np.complex64)
    templates_np[:, 0] = 0.0

    # Complex strain/overwhitened data
    s_real = rng.normal(size=flen).astype(np.float32)
    s_imag = rng.normal(size=flen).astype(np.float32)
    data_np = (s_real + 1j * s_imag).astype(np.complex64)
    data_np[0] = 0.0

    return templates_np, data_np, tlen, flen


# =========================================================================
# 1. Core Arithmetic Parity Against Former Formula
# =========================================================================

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4])
def test_correlate_and_ifft_torch_parity_against_former_formula(
    device, batch_size
):
    """Verify correlate_and_ifft against the exact former PyTorch formula."""
    if torch is None:
        pytest.skip("PyTorch not installed")

    templates_np, data_np, tlen, flen = _generate_fixtures(
        batch_size=batch_size, tlen=512, seed=101
    )
    dev = torch.device(device)
    templates_t = torch.as_tensor(
        templates_np, device=dev, dtype=torch.complex64
    )
    data_t = torch.as_tensor(data_np, device=dev, dtype=torch.complex64)

    kmin, kmax = 20, 180
    corr_slice = slice(kmin, kmax)

    # --- Former PyTorch formula (reference) ---
    ref_cout = torch.zeros(
        (batch_size, tlen), dtype=torch.complex64, device=dev
    )
    ref_out = torch.zeros(
        (batch_size, tlen), dtype=torch.complex64, device=dev
    )
    ref_cout.zero_()
    torch.mul(
        torch.conj(templates_t[:batch_size, corr_slice]),
        data_t[corr_slice].unsqueeze(0),
        out=ref_cout[:batch_size, corr_slice],
    )
    torch.fft.ifft(
        ref_cout[:batch_size],
        n=tlen,
        dim=-1,
        norm="forward",
        out=ref_out[:batch_size],
    )

    # --- Shared core implementation ---
    ws = FilteringWorkspace(
        max_batch_size=batch_size,
        tlen=tlen,
        flen=flen,
        device=device,
    )
    correlate_and_ifft(
        templates=templates_t,
        data=data_t,
        cout_workspace=ws.cout_workspace,
        out_workspace=ws.out_workspace,
        tlen=tlen,
        flen=flen,
        kmin=kmin,
        kmax=kmax,
        batch_size=batch_size,
    )

    # Parity check
    np.testing.assert_allclose(
        ws.cout_workspace[:batch_size].cpu().numpy(),
        ref_cout[:batch_size].cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        ws.out_workspace[:batch_size].cpu().numpy(),
        ref_out[:batch_size].cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_correlate_and_ifft_numpy_parity_against_former_formula(batch_size):
    """Verify correlate_and_ifft against the exact former NumPy formula."""
    templates_np, data_np, tlen, flen = _generate_fixtures(
        batch_size=batch_size, tlen=512, seed=202
    )
    kmin, kmax = 15, 200
    corr_slice = slice(kmin, kmax)

    # --- Former NumPy formula (reference) ---
    ref_cout = np.zeros((batch_size, tlen), dtype=np.complex64)
    ref_cout.fill(0)
    ref_cout[:batch_size, corr_slice] = (
        np.conj(templates_np[:batch_size, corr_slice])
        * data_np[corr_slice][np.newaxis, :]
    )
    ref_out = (
        np.fft.ifft(ref_cout[:batch_size], n=tlen, axis=-1)
        * tlen
    )

    # --- Shared core implementation ---
    ws = FilteringWorkspace(
        max_batch_size=batch_size,
        tlen=tlen,
        flen=flen,
        device="numpy",
    )
    correlate_and_ifft(
        templates=templates_np,
        data=data_np,
        cout_workspace=ws.cout_workspace,
        out_workspace=ws.out_workspace,
        tlen=tlen,
        flen=flen,
        kmin=kmin,
        kmax=kmax,
        batch_size=batch_size,
    )

    np.testing.assert_allclose(
        ws.cout_workspace[:batch_size],
        ref_cout[:batch_size],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        ws.out_workspace[:batch_size],
        ref_out[:batch_size],
        rtol=1e-5,
        atol=1e-5,
    )


def test_torch_numpy_cross_backend_numerical_equivalence():
    """Verify PyTorch (norm='forward') and NumPy (* tlen) match."""
    if torch is None:
        pytest.skip("PyTorch not installed")

    templates_np, data_np, tlen, flen = _generate_fixtures(
        batch_size=3, tlen=256, seed=303
    )
    kmin, kmax = 10, 100

    # NumPy run
    ws_np = FilteringWorkspace(
        max_batch_size=3, tlen=tlen, flen=flen, device="numpy"
    )
    correlate_and_ifft(
        templates=templates_np,
        data=data_np,
        cout_workspace=ws_np.cout_workspace,
        out_workspace=ws_np.out_workspace,
        tlen=tlen,
        flen=flen,
        kmin=kmin,
        kmax=kmax,
        batch_size=3,
    )

    # Torch run (CPU)
    ws_torch = FilteringWorkspace(
        max_batch_size=3, tlen=tlen, flen=flen, device="cpu"
    )
    correlate_and_ifft(
        templates=torch.as_tensor(templates_np),
        data=torch.as_tensor(data_np),
        cout_workspace=ws_torch.cout_workspace,
        out_workspace=ws_torch.out_workspace,
        tlen=tlen,
        flen=flen,
        kmin=kmin,
        kmax=kmax,
        batch_size=3,
    )

    np.testing.assert_allclose(
        ws_np.cout_workspace,
        ws_torch.cout_workspace.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        ws_np.out_workspace,
        ws_torch.out_workspace.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


# =========================================================================
# 2. Batch Sizes: 1D Input, Batch=1, and Batch > 1
# =========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_batch_size_one_1d_and_2d(device):
    """Verify batch=1 works correctly for both 1D and 2D template inputs."""
    templates_np, data_np, tlen, flen = _generate_fixtures(
        batch_size=1, tlen=256, seed=404
    )
    ws = FilteringWorkspace(
        max_batch_size=1, tlen=tlen, flen=flen, device=device
    )

    # 1D template input
    t_1d = templates_np[0]
    if torch is not None and device != "numpy":
        t_in = torch.as_tensor(t_1d, device=torch.device(device))
        d_in = torch.as_tensor(data_np, device=torch.device(device))
    else:
        t_in = t_1d
        d_in = data_np

    correlate_and_ifft(
        templates=t_in,
        data=d_in,
        cout_workspace=ws.cout_workspace,
        out_workspace=ws.out_workspace,
        tlen=tlen,
        flen=flen,
        kmin=10,
        kmax=90,
        batch_size=1,
    )

    res_out1 = (
        ws.out_workspace[0].cpu().numpy().copy()
        if hasattr(ws.out_workspace, "cpu")
        else ws.out_workspace[0].copy()
    )

    # 2D template input
    t_2d = templates_np[:1]
    if torch is not None and device != "numpy":
        t_in2 = torch.as_tensor(t_2d, device=torch.device(device))
    else:
        t_in2 = t_2d

    correlate_and_ifft(
        templates=t_in2,
        data=d_in,
        cout_workspace=ws.cout_workspace,
        out_workspace=ws.out_workspace,
        tlen=tlen,
        flen=flen,
        kmin=10,
        kmax=90,
        batch_size=1,
    )

    res_out2 = (
        ws.out_workspace[0].cpu().numpy()
        if hasattr(ws.out_workspace, "cpu")
        else ws.out_workspace[0].copy()
    )

    np.testing.assert_array_equal(res_out1, res_out2)


# =========================================================================
# 3. Tail Batches and Buffer Reuse
# =========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_tail_batch_buffer_reuse_isolation(device):
    """Verify tail batches execute cleanly without contamination."""
    max_b = 6
    tail_b = 2
    templates_np, data_np, tlen, flen = _generate_fixtures(
        batch_size=max_b, tlen=256, seed=505
    )

    ws = FilteringWorkspace(
        max_batch_size=max_b, tlen=tlen, flen=flen, device=device
    )

    if torch is not None and device != "numpy":
        dev = torch.device(device)
        t_tensor = torch.as_tensor(templates_np, device=dev)
        d_tensor = torch.as_tensor(data_np, device=dev)
    else:
        t_tensor = templates_np
        d_tensor = data_np

    # Pass 1: run full batch of 6
    correlate_and_ifft(
        templates=t_tensor,
        data=d_tensor,
        cout_workspace=ws.cout_workspace,
        out_workspace=ws.out_workspace,
        tlen=tlen,
        flen=flen,
        kmin=10,
        kmax=100,
        batch_size=max_b,
    )

    # Record expected output for the first 2 templates from full pass
    expected_tail = (
        ws.out_workspace[:tail_b].cpu().numpy().copy()
        if hasattr(ws.out_workspace, "cpu")
        else ws.out_workspace[:tail_b].copy()
    )

    # Fill workspace with garbage values to simulate dirty reused memory
    if torch is not None and device != "numpy":
        ws.cout_workspace.fill_(999.0 + 999.0j)
        ws.out_workspace.fill_(888.0 + 888.0j)
    else:
        ws.cout_workspace.fill(999.0 + 999.0j)
        ws.out_workspace.fill(888.0 + 888.0j)

    # Pass 2: run tail batch of 2
    correlate_and_ifft(
        templates=t_tensor[:tail_b],
        data=d_tensor,
        cout_workspace=ws.cout_workspace,
        out_workspace=ws.out_workspace,
        tlen=tlen,
        flen=flen,
        kmin=10,
        kmax=100,
        batch_size=tail_b,
    )

    actual_tail = (
        ws.out_workspace[:tail_b].cpu().numpy()
        if hasattr(ws.out_workspace, "cpu")
        else ws.out_workspace[:tail_b]
    )

    np.testing.assert_allclose(
        actual_tail, expected_tail, rtol=1e-5, atol=1e-5
    )


# =========================================================================
# 4. Changing Cutoff Geometry and Zero Invariant
# =========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_reused_buffer_cutoff_zeros_and_geometry_change(device):
    """Verify changing cutoff geometry does not leave stale frequency bins."""
    templates_np, data_np, tlen, flen = _generate_fixtures(
        batch_size=3, tlen=512, seed=606
    )
    ws = FilteringWorkspace(
        max_batch_size=3, tlen=tlen, flen=flen, device=device
    )

    if torch is not None and device != "numpy":
        dev = torch.device(device)
        t_tensor = torch.as_tensor(templates_np, device=dev)
        d_tensor = torch.as_tensor(data_np, device=dev)
    else:
        t_tensor = templates_np
        d_tensor = data_np

    # 1. Wide band pass: kmin=10, kmax=200
    correlate_and_ifft(
        templates=t_tensor,
        data=d_tensor,
        cout_workspace=ws.cout_workspace,
        out_workspace=ws.out_workspace,
        tlen=tlen,
        flen=flen,
        kmin=10,
        kmax=200,
        batch_size=3,
    )

    # 2. Narrow band pass in same workspace: kmin=50, kmax=120
    correlate_and_ifft(
        templates=t_tensor,
        data=d_tensor,
        cout_workspace=ws.cout_workspace,
        out_workspace=ws.out_workspace,
        tlen=tlen,
        flen=flen,
        kmin=50,
        kmax=120,
        batch_size=3,
    )

    cout_arr = (
        ws.cout_workspace[:3].cpu().numpy()
        if hasattr(ws.cout_workspace, "cpu")
        else ws.cout_workspace[:3]
    )

    # Verify zero invariant: all bins < 50 and >= 120 must be STRICTLY ZERO
    assert np.all(cout_arr[:, :50] == 0.0)
    assert np.all(cout_arr[:, 120:] == 0.0)
    # Active band must NOT be all zero
    assert not np.all(cout_arr[:, 50:120] == 0.0)

    # 3. Full band pass: kmin=None, kmax=None
    # Pre-zero data outside 20:220 to simulate engine behavior
    if torch is not None and device != "numpy":
        d_tensor_win = d_tensor.clone()
        d_tensor_win[:20].zero_()
        d_tensor_win[220:].zero_()
    else:
        d_tensor_win = d_tensor.copy()
        d_tensor_win[:20] = 0.0
        d_tensor_win[220:] = 0.0

    correlate_and_ifft(
        templates=t_tensor,
        data=d_tensor_win,
        cout_workspace=ws.cout_workspace,
        out_workspace=ws.out_workspace,
        tlen=tlen,
        flen=flen,
        kmin=None,
        kmax=None,
        batch_size=3,
    )

    cout_arr2 = (
        ws.cout_workspace[:3].cpu().numpy()
        if hasattr(ws.cout_workspace, "cpu")
        else ws.cout_workspace[:3]
    )
    # Verify padding flen:tlen is zeroed
    assert np.all(cout_arr2[:, flen:] == 0.0)


# =========================================================================
# 5. FilteringWorkspace Slots and Dynamic Growth
# =========================================================================

def test_filtering_workspace_multi_slots_and_growth():
    """Verify multi-slot allocation and ensure_capacity resizing."""
    tlen = 256
    flen = 129

    ws = FilteringWorkspace(
        max_batch_size=2,
        tlen=tlen,
        flen=flen,
        device="numpy",
        num_slots=2,
        allocate_stilde=True,
    )

    assert ws.num_slots == 2
    assert len(ws.slots) == 2
    assert isinstance(ws.slots[0], FilteringWorkspaceSlot)
    assert isinstance(ws.slots[1], FilteringWorkspaceSlot)
    assert len(ws.cout_workspaces) == 2
    assert len(ws.out_workspaces) == 2
    assert len(ws.stilde_bufs) == 2

    assert ws.cout_workspaces[0].shape == (2, tlen)
    assert ws.cout_workspaces[1].shape == (2, tlen)
    assert ws.stilde_bufs[0].shape == (flen,)

    # Slot isolation check
    ws.cout_workspaces[0].fill(1.0)
    ws.cout_workspaces[1].fill(2.0)
    assert np.all(ws.cout_workspaces[0] == 1.0)
    assert np.all(ws.cout_workspaces[1] == 2.0)

    # Dynamic capacity growth on slot 0
    ws.ensure_capacity(batch_size=8, slot_idx=0)
    assert ws.cout_workspaces[0].shape == (8, tlen)
    assert ws.out_workspaces[0].shape == (8, tlen)
    assert ws.cout_workspaces[1].shape == (2, tlen)  # slot 1 unchanged

    # Dynamic capacity growth on all slots
    ws.ensure_capacity(batch_size=10, slot_idx=None)
    assert ws.cout_workspaces[0].shape == (10, tlen)
    assert ws.cout_workspaces[1].shape == (10, tlen)

    # Clear
    ws.clear()
    assert len(ws.slots) == 0
    assert len(ws.cout_workspaces) == 0


# =========================================================================
# 6. CUDA Graph Compatibility
# =========================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_graph_compatibility_with_shared_core():
    """Verify correlate_and_ifft CUDA graph capture and replay."""
    batch_size = 4
    tlen = 512
    flen = tlen // 2 + 1
    templates_np, data_np, _, _ = _generate_fixtures(
        batch_size=batch_size, tlen=tlen, seed=707
    )

    dev = torch.device("cuda")
    tmplt_data = torch.as_tensor(templates_np, device=dev)
    stilde_dev = torch.as_tensor(data_np, device=dev)

    ws = FilteringWorkspace(
        max_batch_size=batch_size, tlen=tlen, flen=flen, device="cuda"
    )

    stream = torch.cuda.Stream(device=dev)

    # Warmup
    with torch.cuda.stream(stream):
        correlate_and_ifft(
            templates=tmplt_data,
            data=stilde_dev,
            cout_workspace=ws.cout_workspace,
            out_workspace=ws.out_workspace,
            tlen=tlen,
            flen=flen,
            batch_size=batch_size,
            stream=stream,
        )
    stream.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        correlate_and_ifft(
            templates=tmplt_data,
            data=stilde_dev,
            cout_workspace=ws.cout_workspace,
            out_workspace=ws.out_workspace,
            tlen=tlen,
            flen=flen,
            batch_size=batch_size,
            stream=stream,
        )

    # Run eager reference
    ws_eager = FilteringWorkspace(
        max_batch_size=batch_size, tlen=tlen, flen=flen, device="cuda"
    )
    correlate_and_ifft(
        templates=tmplt_data,
        data=stilde_dev,
        cout_workspace=ws_eager.cout_workspace,
        out_workspace=ws_eager.out_workspace,
        tlen=tlen,
        flen=flen,
        batch_size=batch_size,
    )

    # Replay graph
    with torch.cuda.stream(stream):
        g.replay()
    stream.synchronize()

    # Numerical parity check
    np.testing.assert_allclose(
        ws.out_workspace.cpu().numpy(),
        ws_eager.out_workspace.cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


# =========================================================================
# 7. Edge Validation, Empty Batches, Dtypes, and Allocation
# =========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_correlate_and_ifft_empty_batch(device):
    """Verify that batch_size=0 returns empty slices without error."""
    templates_np, data_np, tlen, flen = _generate_fixtures(
        batch_size=2, tlen=256, seed=801
    )
    ws = FilteringWorkspace(
        max_batch_size=2, tlen=tlen, flen=flen, device=device
    )

    if torch is not None and device != "numpy":
        t_in = torch.as_tensor(templates_np, device=torch.device(device))
        d_in = torch.as_tensor(data_np, device=torch.device(device))
    else:
        t_in = templates_np
        d_in = data_np

    cout, out = correlate_and_ifft(
        templates=t_in,
        data=d_in,
        cout_workspace=ws.cout_workspace,
        out_workspace=ws.out_workspace,
        tlen=tlen,
        flen=flen,
        kmin=10,
        kmax=90,
        batch_size=0,
    )
    assert cout.shape[0] == 0
    assert out.shape[0] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_correlate_and_ifft_geometry_and_capacity_rejections(device):
    """Verify invalid geometry, sizes, and small buffers are rejected."""
    templates_np, data_np, tlen, flen = _generate_fixtures(
        batch_size=2, tlen=256, seed=802
    )
    ws = FilteringWorkspace(
        max_batch_size=2, tlen=tlen, flen=flen, device=device
    )

    if torch is not None and device != "numpy":
        t_in = torch.as_tensor(templates_np, device=torch.device(device))
        d_in = torch.as_tensor(data_np, device=torch.device(device))
    else:
        t_in = templates_np
        d_in = data_np

    # Negative batch size
    with pytest.raises(ValueError, match="batch_size"):
        correlate_and_ifft(
            templates=t_in,
            data=d_in,
            cout_workspace=ws.cout_workspace,
            out_workspace=ws.out_workspace,
            tlen=tlen,
            batch_size=-1,
        )

    # Invalid tlen
    with pytest.raises(ValueError, match="tlen"):
        correlate_and_ifft(
            templates=t_in,
            data=d_in,
            cout_workspace=ws.cout_workspace,
            out_workspace=ws.out_workspace,
            tlen=0,
        )

    # Invalid flen
    with pytest.raises(ValueError, match="flen"):
        correlate_and_ifft(
            templates=t_in,
            data=d_in,
            cout_workspace=ws.cout_workspace,
            out_workspace=ws.out_workspace,
            tlen=tlen,
            flen=tlen + 10,
        )

    # kmin < 0
    with pytest.raises(ValueError, match="kmin"):
        correlate_and_ifft(
            templates=t_in,
            data=d_in,
            cout_workspace=ws.cout_workspace,
            out_workspace=ws.out_workspace,
            tlen=tlen,
            kmin=-5,
        )

    # kmax < kmin
    with pytest.raises(ValueError, match="kmax"):
        correlate_and_ifft(
            templates=t_in,
            data=d_in,
            cout_workspace=ws.cout_workspace,
            out_workspace=ws.out_workspace,
            tlen=tlen,
            kmin=50,
            kmax=30,
        )

    # kmax > flen
    with pytest.raises(ValueError, match="kmax"):
        correlate_and_ifft(
            templates=t_in,
            data=d_in,
            cout_workspace=ws.cout_workspace,
            out_workspace=ws.out_workspace,
            tlen=tlen,
            flen=flen,
            kmax=flen + 5,
        )

    # cout_workspace too small for batch
    small_cout = ws.cout_workspace[:1]
    with pytest.raises(ValueError, match="cout_workspace capacity"):
        correlate_and_ifft(
            templates=t_in,
            data=d_in,
            cout_workspace=small_cout,
            out_workspace=ws.out_workspace,
            tlen=tlen,
            batch_size=2,
        )

    # templates length too short for kmax
    short_t = t_in[:, :50]
    with pytest.raises(ValueError, match="templates length"):
        correlate_and_ifft(
            templates=short_t,
            data=d_in,
            cout_workspace=ws.cout_workspace,
            out_workspace=ws.out_workspace,
            tlen=tlen,
            kmin=10,
            kmax=80,
            batch_size=2,
        )

    # data length too short for kmax
    short_d = d_in[:50]
    with pytest.raises(ValueError, match="data length"):
        correlate_and_ifft(
            templates=t_in,
            data=short_d,
            cout_workspace=ws.cout_workspace,
            out_workspace=ws.out_workspace,
            tlen=tlen,
            kmin=10,
            kmax=80,
            batch_size=2,
        )


def test_workspace_slot_index_and_stilde_allocation():
    """Verify invalid slot index and allocate_stilde on existing slot."""
    ws = FilteringWorkspace(
        max_batch_size=2,
        tlen=256,
        flen=129,
        device="numpy",
        num_slots=2,
        allocate_stilde=False,
    )
    assert ws.stilde_bufs[0] is None
    assert ws.stilde_bufs[1] is None

    # Invalid slot index rejected
    with pytest.raises(IndexError, match="Slot index"):
        ws.ensure_capacity(batch_size=4, slot_idx=5)

    with pytest.raises(IndexError, match="Slot index"):
        ws.ensure_capacity(batch_size=4, slot_idx=-1)

    # Request allocate_stilde on an already allocated slot
    ws.ensure_capacity(batch_size=2, slot_idx=0, allocate_stilde=True)
    assert ws.stilde_bufs[0] is not None
    assert ws.stilde_bufs[0].shape == (129,)
    assert ws.stilde_bufs[1] is None  # slot 1 still None

    # Request allocate_stilde across all slots
    ws.ensure_capacity(batch_size=2, slot_idx=None, allocate_stilde=True)
    assert ws.stilde_bufs[1] is not None
    assert ws.stilde_bufs[1].shape == (129,)


def test_dtype_resolution_and_support():
    """Verify dtype validation: complex64/128 allowed, others rejected."""
    # complex64
    ws64 = FilteringWorkspace(
        max_batch_size=1, tlen=64, device="numpy", dtype=np.complex64
    )
    assert ws64.dtype == np.dtype(np.complex64)
    assert ws64.cout_workspace.dtype == np.complex64

    # complex128
    ws128 = FilteringWorkspace(
        max_batch_size=1, tlen=64, device="numpy", dtype=np.complex128
    )
    assert ws128.dtype == np.dtype(np.complex128)
    assert ws128.cout_workspace.dtype == np.complex128

    if torch is not None:
        ws_t64 = FilteringWorkspace(
            max_batch_size=1, tlen=64, device="cpu", dtype=torch.complex64
        )
        assert ws_t64.cout_workspace.dtype == torch.complex64

        ws_t128 = FilteringWorkspace(
            max_batch_size=1, tlen=64, device="cpu", dtype=torch.complex128
        )
        assert ws_t128.cout_workspace.dtype == torch.complex128

    # Non-complex rejected
    with pytest.raises(ValueError, match="Unsupported complex dtype"):
        FilteringWorkspace(
            max_batch_size=1, tlen=64, device="numpy", dtype=np.float32
        )

    with pytest.raises(ValueError, match="Unsupported complex dtype"):
        FilteringWorkspace(
            max_batch_size=1, tlen=64, device="numpy", dtype="float64"
        )


@pytest.mark.parametrize("device", ["numpy"] + DEVICES)
def test_correlate_and_ifft_oversized_workspace_rejection(device):
    """Verify that workspace widths not exactly equal to tlen are rejected."""
    tlen = 256
    flen = tlen // 2 + 1
    t_in, d_in, _, _ = _generate_fixtures(batch_size=2, tlen=tlen, seed=99)
    if torch is not None and device != "numpy":
        dev = torch.device(device)
        t_in = torch.as_tensor(t_in, device=dev)
        d_in = torch.as_tensor(d_in, device=dev)
        ws_cout = torch.zeros(
            (2, tlen + 16), dtype=torch.complex64, device=dev
        )
        ws_out = torch.zeros((2, tlen), dtype=torch.complex64, device=dev)
        ws_out_over = torch.zeros(
            (2, tlen + 16), dtype=torch.complex64, device=dev
        )
        ws_cout_normal = torch.zeros(
            (2, tlen), dtype=torch.complex64, device=dev
        )
    else:
        ws_cout = np.zeros((2, tlen + 16), dtype=np.complex64)
        ws_out = np.zeros((2, tlen), dtype=np.complex64)
        ws_out_over = np.zeros((2, tlen + 16), dtype=np.complex64)
        ws_cout_normal = np.zeros((2, tlen), dtype=np.complex64)

    with pytest.raises(ValueError, match="cout_workspace length"):
        correlate_and_ifft(
            templates=t_in,
            data=d_in,
            cout_workspace=ws_cout,
            out_workspace=ws_out,
            tlen=tlen,
            flen=flen,
        )

    with pytest.raises(ValueError, match="out_workspace length"):
        correlate_and_ifft(
            templates=t_in,
            data=d_in,
            cout_workspace=ws_cout_normal,
            out_workspace=ws_out_over,
            tlen=tlen,
            flen=flen,
        )


@pytest.mark.parametrize("device", ["numpy"] + DEVICES)
def test_correlate_and_ifft_invalid_template_ndim_rejection(device):
    """Verify that templates with ndim not in (1, 2) raise ValueError."""
    tlen = 256
    flen = tlen // 2 + 1
    _, d_in, _, _ = _generate_fixtures(batch_size=2, tlen=tlen, seed=100)
    if torch is not None and device != "numpy":
        dev = torch.device(device)
        d_in = torch.as_tensor(d_in, device=dev)
        t_3d = torch.zeros((2, 3, flen), dtype=torch.complex64, device=dev)
        t_0d = torch.tensor(1.0 + 0j, dtype=torch.complex64, device=dev)
    else:
        t_3d = np.zeros((2, 3, flen), dtype=np.complex64)
        t_0d = np.array(1.0 + 0j, dtype=np.complex64)

    ws = FilteringWorkspace(
        max_batch_size=2, tlen=tlen, flen=flen, device=device
    )

    with pytest.raises(ValueError, match="templates must be 1D or 2D"):
        correlate_and_ifft(
            templates=t_3d,
            data=d_in,
            cout_workspace=ws.cout_workspace,
            out_workspace=ws.out_workspace,
            tlen=tlen,
        )

    with pytest.raises(ValueError, match="templates must be 1D or 2D"):
        correlate_and_ifft(
            templates=t_0d,
            data=d_in,
            cout_workspace=ws.cout_workspace,
            out_workspace=ws.out_workspace,
            tlen=tlen,
        )
