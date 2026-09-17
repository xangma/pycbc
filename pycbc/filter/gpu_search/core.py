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
Shared core execution and workspace management for tiled matched filtering.

Provides:
- FilteringWorkspaceSlot: Container for correlation and IFFT buffers.
- FilteringWorkspace: Multi-slot buffer manager with capacity growth.
- correlate_and_ifft: Unified batched matched-filter correlation and inverse
  FFT kernel supporting PyTorch and NumPy backends.
"""

from typing import Any, List, Optional, Tuple
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def _resolve_complex_dtype(dtype: Any) -> Tuple[np.dtype, Any]:
    """
    Resolve and validate requested complex dtype.

    Returns (np_dtype, torch_dtype).
    Raises ValueError if requested dtype is not complex64 or complex128.
    """
    if dtype is None:
        dtype = np.complex64

    if torch is not None and isinstance(dtype, torch.dtype):
        if dtype == torch.complex64:
            return np.dtype(np.complex64), torch.complex64
        elif dtype == torch.complex128:
            return np.dtype(np.complex128), torch.complex128
        else:
            raise ValueError(
                f"Unsupported torch dtype {dtype}; must be torch.complex64 "
                f"or torch.complex128"
            )

    try:
        np_dt = np.dtype(dtype)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid dtype {dtype}: {e}") from e

    if np_dt == np.dtype(np.complex64):
        torch_dt = torch.complex64 if torch is not None else None
        return np_dt, torch_dt
    elif np_dt == np.dtype(np.complex128):
        torch_dt = torch.complex128 if torch is not None else None
        return np_dt, torch_dt
    else:
        raise ValueError(
            f"Unsupported complex dtype {dtype} (resolved as {np_dt}); "
            f"must be complex64 or complex128"
        )


class FilteringWorkspaceSlot:
    """Single workspace slot for correlation, IFFT, and auxiliary buffers."""

    def __init__(
        self,
        batch_size: int,
        tlen: int,
        flen: Optional[int] = None,
        device: str = "cpu",
        dtype: Any = np.complex64,
        allocate_stilde: bool = False,
    ):
        b = int(batch_size)
        if b < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        tlen_val = int(tlen)
        if tlen_val <= 0:
            raise ValueError(f"tlen must be positive, got {tlen}")

        self.tlen = tlen_val
        self.flen = int(flen) if flen is not None else (self.tlen // 2 + 1)
        if self.flen <= 0 or self.flen > self.tlen:
            raise ValueError(
                f"flen must satisfy 1 <= flen <= tlen, got flen={self.flen}, "
                f"tlen={self.tlen}"
            )

        self.device = str(device)
        self.np_dtype, self.torch_dtype = _resolve_complex_dtype(dtype)
        self.dtype = self.np_dtype
        self.batch_size = 0
        self.cout = None
        self.out = None
        self.stilde = None
        if b > 0:
            self.allocate(b, allocate_stilde=allocate_stilde)

    def allocate(self, batch_size: int, allocate_stilde: bool = False):
        """Allocate or reallocate workspace buffers for given batch size."""
        b = int(batch_size)
        if b < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        self.batch_size = b
        if torch is not None and self.device != "numpy":
            dev = torch.device(self.device)
            self.cout = torch.zeros(
                (b, self.tlen), dtype=self.torch_dtype, device=dev
            )
            self.out = torch.zeros(
                (b, self.tlen), dtype=self.torch_dtype, device=dev
            )
            if allocate_stilde:
                self.stilde = torch.zeros(
                    (self.flen,), dtype=self.torch_dtype, device=dev
                )
        else:
            self.cout = np.zeros((b, self.tlen), dtype=self.np_dtype)
            self.out = np.zeros((b, self.tlen), dtype=self.np_dtype)
            if allocate_stilde:
                self.stilde = np.zeros((self.flen,), dtype=self.np_dtype)

    def ensure_capacity(self, batch_size: int, allocate_stilde: bool = False):
        """Ensure slot buffers can hold at least batch_size templates."""
        b = int(batch_size)
        if b < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")
        need_stilde = allocate_stilde or (self.stilde is not None)
        if self.cout is None or self.cout.shape[0] < b:
            self.allocate(b, allocate_stilde=need_stilde)
        elif allocate_stilde and self.stilde is None:
            if torch is not None and self.device != "numpy":
                dev = torch.device(self.device)
                self.stilde = torch.zeros(
                    (self.flen,), dtype=self.torch_dtype, device=dev
                )
            else:
                self.stilde = np.zeros((self.flen,), dtype=self.np_dtype)


class FilteringWorkspace:
    """
    Manages allocation, resizing, and ownership of workspace buffers.
    """

    def __init__(
        self,
        max_batch_size: int,
        tlen: int,
        flen: Optional[int] = None,
        device: str = "cpu",
        dtype: Any = np.complex64,
        num_slots: int = 1,
        allocate_stilde: bool = False,
    ):
        max_b = int(max_batch_size)
        if max_b < 0:
            raise ValueError(
                f"max_batch_size must be >= 0, got {max_batch_size}"
            )
        tlen_val = int(tlen)
        if tlen_val <= 0:
            raise ValueError(f"tlen must be positive, got {tlen}")
        slots_count = int(num_slots)
        if slots_count <= 0:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")

        self.max_batch_size = max_b
        self.tlen = tlen_val
        self.flen = int(flen) if flen is not None else (self.tlen // 2 + 1)
        if self.flen <= 0 or self.flen > self.tlen:
            raise ValueError(
                f"flen must satisfy 1 <= flen <= tlen, got flen={self.flen}, "
                f"tlen={self.tlen}"
            )

        self.device = str(device)
        self.np_dtype, self.torch_dtype = _resolve_complex_dtype(dtype)
        self.dtype = self.np_dtype
        self.num_slots = slots_count
        self.allocate_stilde = bool(allocate_stilde)

        self.slots: List[FilteringWorkspaceSlot] = [
            FilteringWorkspaceSlot(
                batch_size=self.max_batch_size,
                tlen=self.tlen,
                flen=self.flen,
                device=self.device,
                dtype=self.dtype,
                allocate_stilde=self.allocate_stilde,
            )
            for _ in range(self.num_slots)
        ]

        self.cout_workspaces: List[Any] = [s.cout for s in self.slots]
        self.out_workspaces: List[Any] = [s.out for s in self.slots]
        self.stilde_bufs: List[Any] = [s.stilde for s in self.slots]

    def _sync_lists(self):
        """Sync workspace list references with underlying slot instances."""
        self.cout_workspaces[:] = [s.cout for s in self.slots]
        self.out_workspaces[:] = [s.out for s in self.slots]
        self.stilde_bufs[:] = [s.stilde for s in self.slots]

    def ensure_capacity(
        self,
        batch_size: int,
        slot_idx: Optional[int] = 0,
        allocate_stilde: Optional[bool] = None,
    ):
        """
        Ensure specified slot (or all slots if slot_idx is None) has capacity.

        Raises IndexError if slot_idx is invalid.
        Raises ValueError if batch_size < 0.
        """
        b = int(batch_size)
        if b < 0:
            raise ValueError(f"batch_size must be >= 0, got {batch_size}")

        need_stilde = (
            self.allocate_stilde
            if allocate_stilde is None
            else bool(allocate_stilde)
        )
        if need_stilde:
            self.allocate_stilde = True

        if slot_idx is None:
            for s in self.slots:
                s.ensure_capacity(b, allocate_stilde=need_stilde)
            self.max_batch_size = max(self.max_batch_size, b)
            self._sync_lists()
        elif 0 <= slot_idx < len(self.slots):
            self.slots[slot_idx].ensure_capacity(
                b, allocate_stilde=need_stilde
            )
            self.max_batch_size = max(self.max_batch_size, b)
            self._sync_lists()
        else:
            raise IndexError(
                f"Slot index {slot_idx} is out of range for workspace with "
                f"{len(self.slots)} slots"
            )

    @property
    def cout_workspace(self) -> Any:
        return self.slots[0].cout if self.slots else None

    @cout_workspace.setter
    def cout_workspace(self, val: Any):
        if self.slots:
            self.slots[0].cout = val
            self._sync_lists()

    @property
    def out_workspace(self) -> Any:
        return self.slots[0].out if self.slots else None

    @out_workspace.setter
    def out_workspace(self, val: Any):
        if self.slots:
            self.slots[0].out = val
            self._sync_lists()

    @property
    def stilde_buf(self) -> Any:
        return self.slots[0].stilde if self.slots else None

    @stilde_buf.setter
    def stilde_buf(self, val: Any):
        if self.slots:
            self.slots[0].stilde = val
            self._sync_lists()

    def clear(self):
        """Release allocated buffers and clear slot references."""
        for s in self.slots:
            s.cout = None
            s.out = None
            s.stilde = None
        self.slots.clear()
        self.cout_workspaces.clear()
        self.out_workspaces.clear()
        self.stilde_bufs.clear()


def correlate_and_ifft(
    templates: Any,
    data: Any,
    cout_workspace: Any,
    out_workspace: Any,
    tlen: int,
    flen: Optional[int] = None,
    kmin: Optional[int] = None,
    kmax: Optional[int] = None,
    batch_size: Optional[int] = None,
    stream: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """
    Unified batched matched-filter correlation and inverse FFT kernel.

    Calculates in frequency domain:
        q_tilde_i(f) = h_tilde_i^*(f) * s_tilde(f)
    and transforms to time domain:
        q_i(t) = IFFT(q_tilde_i(f))

    Writes frequency-domain correlation into cout_workspace[:b] and
    time-domain SNR into out_workspace[:b].

    Parameters
    ----------
    templates : torch.Tensor or np.ndarray
        Batch of frequency-domain templates. Shape (B, L) where L >= kmax.
    data : torch.Tensor or np.ndarray
        Frequency-domain strain data. Shape (L,) or (B, L) where L >= kmax.
    cout_workspace : torch.Tensor or np.ndarray
        Preallocated output workspace for correlation. Shape (>= B, tlen).
    out_workspace : torch.Tensor or np.ndarray
        Preallocated output workspace for IFFT. Shape (>= B, tlen).
    tlen : int
        Transform length (FFT length).
    flen : int, optional
        Filter length (tlen // 2 + 1). Defaults to tlen // 2 + 1.
    kmin : int, optional
        Lower frequency cutoff index.
    kmax : int, optional
        Upper frequency cutoff index.
    batch_size : int, optional
        Number of templates in current batch. Inferred from templates if
        omitted.
    stream : optional
        PyTorch CUDA stream to execute operations within.

    Returns
    -------
    cout : slice
        cout_workspace[:batch_size]
    out : slice
        out_workspace[:batch_size]
    """
    if templates.ndim not in (1, 2):
        raise ValueError(
            f"templates must be 1D or 2D, got {templates.ndim}D"
        )
    if templates.ndim == 1:
        templates = (
            templates.unsqueeze(0)
            if hasattr(templates, "unsqueeze")
            else templates[np.newaxis, :]
        )

    b = int(batch_size) if batch_size is not None else int(templates.shape[0])
    if b < 0:
        raise ValueError(f"batch_size must be >= 0, got {b}")
    if b == 0:
        return cout_workspace[:0], out_workspace[:0]

    # Validate transform and filter lengths
    tlen_val = int(tlen)
    if tlen_val <= 0:
        raise ValueError(f"tlen must be positive, got {tlen_val}")

    flen_val = int(flen) if flen is not None else (tlen_val // 2 + 1)
    if flen_val <= 0 or flen_val > tlen_val:
        raise ValueError(
            f"flen must satisfy 1 <= flen <= tlen, got flen={flen_val}, "
            f"tlen={tlen_val}"
        )

    # Validate and compute frequency cutoff geometry once
    k_start = 0 if kmin is None else int(kmin)
    k_end = flen_val if kmax is None else int(kmax)

    if k_start < 0:
        raise ValueError(f"kmin must be >= 0, got {k_start}")
    if k_end < k_start:
        raise ValueError(
            f"kmax ({k_end}) cannot be less than kmin ({k_start})"
        )
    if k_end > flen_val:
        raise ValueError(
            f"kmax ({k_end}) cannot exceed filter length ({flen_val})"
        )

    corr_slice = slice(k_start, k_end)

    # Validate workspace capacities using metadata only (no tensor reads /
    # sync)
    if cout_workspace.ndim != 2:
        raise ValueError(
            f"cout_workspace must be 2D, got shape {cout_workspace.shape}"
        )
    if out_workspace.ndim != 2:
        raise ValueError(
            f"out_workspace must be 2D, got shape {out_workspace.shape}"
        )

    if cout_workspace.shape[0] < b:
        raise ValueError(
            f"cout_workspace capacity ({cout_workspace.shape[0]}) < "
            f"batch size ({b})"
        )
    if out_workspace.shape[0] < b:
        raise ValueError(
            f"out_workspace capacity ({out_workspace.shape[0]}) < "
            f"batch size ({b})"
        )
    if cout_workspace.shape[1] != tlen_val:
        raise ValueError(
            f"cout_workspace length ({cout_workspace.shape[1]}) must be "
            f"exactly tlen ({tlen_val})"
        )
    if out_workspace.shape[1] != tlen_val:
        raise ValueError(
            f"out_workspace length ({out_workspace.shape[1]}) must be "
            f"exactly tlen ({tlen_val})"
        )

    if templates.shape[0] < b:
        raise ValueError(
            f"templates batch size ({templates.shape[0]}) < "
            f"requested batch size ({b})"
        )
    if templates.shape[1] < k_end:
        raise ValueError(
            f"templates length ({templates.shape[1]}) < kmax ({k_end})"
        )

    if data.ndim == 1:
        if data.shape[0] < k_end:
            raise ValueError(
                f"data length ({data.shape[0]}) < kmax ({k_end})"
            )
    elif data.ndim == 2:
        if data.shape[0] < b:
            raise ValueError(
                f"data batch size ({data.shape[0]}) < "
                f"requested batch size ({b})"
            )
        if data.shape[1] < k_end:
            raise ValueError(
                f"data length ({data.shape[1]}) < kmax ({k_end})"
            )
    else:
        raise ValueError(f"data must be 1D or 2D, got {data.ndim}D")

    # Reuse active-row views throughout execution and in the return value.
    # Creating repeated Torch views adds host overhead for small tiles.
    active_cout = cout_workspace[:b]
    active_out = out_workspace[:b]

    # Backend-specific execution
    if torch is not None and isinstance(cout_workspace, torch.Tensor):
        is_cuda = cout_workspace.is_cuda
        need_stream_context = (
            is_cuda
            and stream is not None
            and torch.cuda.current_stream(cout_workspace.device) != stream
        )

        def _execute_torch():
            # Preserve the offline adapter's single clear launch when a
            # lower cutoff is present. Splitting this into two smaller clears
            # saves bandwidth on large tiles but slows short CUDA batches.
            if k_start > 0:
                active_cout.zero_()
            elif k_end < tlen_val:
                active_cout[:, k_end:].zero_()

            # Overwrite every active bin [k_start:k_end]
            if k_end > k_start:
                data_slice = (
                    data[corr_slice].unsqueeze(0)
                    if data.ndim == 1
                    else data[:b, corr_slice]
                )
                torch.mul(
                    torch.conj(templates[:b, corr_slice]),
                    data_slice,
                    out=active_cout[:, corr_slice],
                )

            # Batched inverse FFT
            torch.fft.ifft(
                active_cout,
                n=tlen_val,
                dim=-1,
                norm="forward",
                out=active_out,
            )

        if need_stream_context:
            with torch.cuda.stream(stream):
                _execute_torch()
        else:
            _execute_torch()

    else:
        # NumPy execution path
        # Clear inactive bins for the active b rows; leave tail untouched
        if k_start > 0:
            active_cout[:, :k_start] = 0.0
        if k_end < tlen_val:
            active_cout[:, k_end:] = 0.0

        # Overwrite every active bin [k_start:k_end]
        if k_end > k_start:
            data_slice = (
                data[corr_slice][np.newaxis, :]
                if data.ndim == 1
                else data[:b, corr_slice]
            )
            active_cout[:, corr_slice] = (
                np.conj(templates[:b, corr_slice]) * data_slice
            )

        # Batched inverse FFT
        active_out[:] = (
            np.fft.ifft(active_cout, n=tlen_val, axis=-1) * tlen_val
        )

    return active_cout, active_out
