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
Candidate selection policies and preallocated Structure-of-Arrays device queues.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

try:
    import torch
except ImportError:
    torch = None


@dataclass(frozen=True)
class SelectionPolicy:
    """Policy governing candidate detection and clustering."""

    snr_threshold: float
    snr_abort_threshold: Optional[float] = None
    cluster_policy: str = "live_peak"  # "live_peak", "symmetric", or "threshold_only"
    cluster_window: int = 0  # In samples
    max_triggers_in_batch: Optional[int] = None


class CandidateBuffer:
    """
    Preallocated Structure-of-Arrays (SoA) buffer for device-resident candidate collection.
    """

    def __init__(self, capacity: int = 65536, device: str = "cpu"):
        self.capacity = int(capacity)
        self.device = device
        self._allocate()

    def _allocate(self):
        if torch is not None and self.device != "numpy":
            dev = torch.device(self.device)
            self.sample_indices = torch.zeros(
                self.capacity, dtype=torch.int64, device=dev
            )
            self.template_indices = torch.zeros(
                self.capacity, dtype=torch.int64, device=dev
            )
            self.snr_values = torch.zeros(
                self.capacity, dtype=torch.complex64, device=dev
            )
            self.sigmasq_values = torch.zeros(
                self.capacity, dtype=torch.float32, device=dev
            )
            self.device_count = torch.zeros((), dtype=torch.int64, device=dev)
            self.overflow_flag = torch.zeros((), dtype=torch.bool, device=dev)
        else:
            self.sample_indices = np.zeros(self.capacity, dtype=np.int64)
            self.template_indices = np.zeros(self.capacity, dtype=np.int64)
            self.snr_values = np.zeros(self.capacity, dtype=np.complex64)
            self.sigmasq_values = np.zeros(self.capacity, dtype=np.float32)
            self.device_count = np.array(0, dtype=np.int64)
            self.overflow_flag = np.array(False, dtype=bool)

    def reset(self):
        """Reset the queue counters to zero."""
        if torch is not None and isinstance(self.device_count, torch.Tensor):
            self.device_count.zero_()
            self.overflow_flag.zero_()
        else:
            self.device_count.fill(0)
            self.overflow_flag.fill(False)

    def resize(self, new_capacity: int):
        """Grow candidate buffer capacity preserving existing contents."""
        new_capacity = max(self.capacity * 2, int(new_capacity))
        old_count = min(self.count, self.capacity)
        old_samples = self.sample_indices[:old_count]
        old_tmplts = self.template_indices[:old_count]
        old_snr = self.snr_values[:old_count]
        old_sigmasq = self.sigmasq_values[:old_count]

        self.capacity = new_capacity
        self._allocate()

        if old_count > 0:
            if torch is not None and isinstance(self.sample_indices, torch.Tensor):
                self.sample_indices[:old_count].copy_(old_samples)
                self.template_indices[:old_count].copy_(old_tmplts)
                self.snr_values[:old_count].copy_(old_snr)
                self.sigmasq_values[:old_count].copy_(old_sigmasq)
                self.device_count.fill_(old_count)
            else:
                self.sample_indices[:old_count] = old_samples
                self.template_indices[:old_count] = old_tmplts
                self.snr_values[:old_count] = old_snr
                self.sigmasq_values[:old_count] = old_sigmasq
                self.device_count = np.array(old_count, dtype=np.int64)

    @property
    def count(self) -> int:
        if torch is not None and isinstance(self.device_count, torch.Tensor):
            return int(self.device_count.item())
        return int(self.device_count)

    @property
    def has_overflow(self) -> bool:
        if torch is not None and isinstance(self.overflow_flag, torch.Tensor):
            return bool(self.overflow_flag.item())
        return bool(self.overflow_flag)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert committed records up to device count to numpy arrays."""
        n = min(self.count, self.capacity)
        if n == 0:
            return {
                "sample_indices": np.empty(0, dtype=np.int64),
                "template_indices": np.empty(0, dtype=np.int64),
                "snr": np.empty(0, dtype=np.complex64),
                "sigmasq": np.empty(0, dtype=np.float32),
            }

        if torch is not None and isinstance(self.sample_indices, torch.Tensor):
            return {
                "sample_indices": self.sample_indices[:n].detach().cpu().numpy(),
                "template_indices": self.template_indices[:n].detach().cpu().numpy(),
                "snr": self.snr_values[:n].detach().cpu().numpy(),
                "sigmasq": self.sigmasq_values[:n].detach().cpu().numpy(),
            }
        else:
            return {
                "sample_indices": self.sample_indices[:n].copy(),
                "template_indices": self.template_indices[:n].copy(),
                "snr": self.snr_values[:n].copy(),
                "sigmasq": self.sigmasq_values[:n].copy(),
            }


def _store_in_buffer(
    buffer: Optional[CandidateBuffer],
    sel_tmplt: Any,
    sel_sample: Any,
    sel_snr: Any,
    sel_sigmasq: Any,
    num_survivors: int,
) -> bool:
    """Store survivor batch in preallocated buffer, detecting overflow."""
    if buffer is None:
        return False
    start_c = buffer.count
    if start_c + num_survivors > buffer.capacity:
        if torch is not None and isinstance(buffer.overflow_flag, torch.Tensor):
            buffer.overflow_flag.fill_(True)
        else:
            buffer.overflow_flag.fill(True)
        return True

    if torch is not None and isinstance(buffer.sample_indices, torch.Tensor):
        buffer.template_indices[start_c : start_c + num_survivors].copy_(sel_tmplt)
        buffer.sample_indices[start_c : start_c + num_survivors].copy_(sel_sample)
        buffer.snr_values[start_c : start_c + num_survivors].copy_(sel_snr)
        buffer.sigmasq_values[start_c : start_c + num_survivors].copy_(sel_sigmasq)
        buffer.device_count.add_(num_survivors)
    else:
        buffer.template_indices[start_c : start_c + num_survivors] = sel_tmplt
        buffer.sample_indices[start_c : start_c + num_survivors] = sel_sample
        buffer.snr_values[start_c : start_c + num_survivors] = sel_snr
        buffer.sigmasq_values[start_c : start_c + num_survivors] = sel_sigmasq
        buffer.device_count += num_survivors
    return False


def _select_torch_symmetric(
    vals: torch.Tensor,
    norms: torch.Tensor,
    sigmasqs: torch.Tensor,
    valid_start: int,
    policy: SelectionPolicy,
    buffer: Optional[CandidateBuffer] = None,
) -> Dict[str, Any]:
    """Execute batched 2D symmetric clustering across templates on device."""
    batch_size = vals.shape[0]
    total_len = vals.shape[-1]
    window = max(1, int(policy.cluster_window))

    sq_mag = vals.real.square() + vals.imag.square()
    sq_mag = torch.nan_to_num(sq_mag, nan=0.0)

    num_blocks = (total_len + window - 1) // window
    if num_blocks == 0:
        return {
            "aborted": False,
            "overflow": False,
            "candidates": {
                "template_idx": np.empty(0, dtype=np.int64),
                "sample_idx": np.empty(0, dtype=np.int64),
                "snr": np.empty(0, dtype=np.complex64),
                "sigmasq": np.empty(0, dtype=np.float32),
            },
        }

    pad = num_blocks * window - total_len
    if pad > 0:
        pad_val = torch.full(
            (batch_size, pad), float("-inf"), device=vals.device, dtype=sq_mag.dtype
        )
        sq_padded = torch.cat([sq_mag, pad_val], dim=-1)
    else:
        sq_padded = sq_mag

    blocks = sq_padded.view(batch_size, num_blocks, window)
    block_max, block_idx = torch.max(blocks, dim=-1)
    block_idx.add_(
        torch.arange(num_blocks, device=vals.device).unsqueeze(0), alpha=window
    )

    # Threshold check
    safe_norms = torch.clamp(norms, min=1e-12)
    thresh_sq = (policy.snr_threshold / safe_norms).square().unsqueeze(-1)
    keep = block_max > thresh_sq

    # Abort check
    if policy.snr_abort_threshold is not None:
        abort_thresh_sq = (policy.snr_abort_threshold / safe_norms).square().unsqueeze(-1)
        if (block_max >= abort_thresh_sq).any():
            return {"aborted": True, "candidates": {}}

    if num_blocks > 1:
        # Match parallel_thresh_cluster exactly:
        # Candidate strictly > previous neighbor, and >= next neighbor.
        keep[:, 1:] &= block_max[:, 1:] > block_max[:, :-1]
        keep[:, :-1] &= block_max[:, :-1] >= block_max[:, 1:]
        keep[:, 0] &= block_max[:, 0] > block_max[:, 1]

    survivor_indices = torch.nonzero(keep, as_tuple=True)
    sel_tmplt = survivor_indices[0]
    block_pos = survivor_indices[1]
    num_survivors = int(sel_tmplt.numel())

    if num_survivors == 0:
        return {
            "aborted": False,
            "overflow": False,
            "candidates": {
                "template_idx": np.empty(0, dtype=np.int64),
                "sample_idx": np.empty(0, dtype=np.int64),
                "snr": np.empty(0, dtype=np.complex64),
                "sigmasq": np.empty(0, dtype=np.float32),
            },
        }

    sel_sample_rel = block_idx[sel_tmplt, block_pos]
    sel_sample = sel_sample_rel + valid_start
    peak_samples = vals[sel_tmplt, sel_sample_rel]
    sel_snr = peak_samples * norms[sel_tmplt]
    sel_sigmasq = sigmasqs[sel_tmplt]

    overflow = _store_in_buffer(
        buffer, sel_tmplt, sel_sample, sel_snr, sel_sigmasq, num_survivors
    )
    if overflow:
        return {"aborted": False, "overflow": True, "candidates": {}}

    return {
        "aborted": False,
        "overflow": False,
        "candidates": {
            "template_idx": sel_tmplt.detach().cpu().numpy(),
            "sample_idx": sel_sample.detach().cpu().numpy(),
            "snr": sel_snr.detach().cpu().numpy(),
            "sigmasq": sel_sigmasq.detach().cpu().numpy(),
        },
    }


def _select_numpy_symmetric(
    vals: np.ndarray,
    norms: np.ndarray,
    sigmasqs: np.ndarray,
    valid_start: int,
    policy: SelectionPolicy,
    buffer: Optional[CandidateBuffer] = None,
) -> Dict[str, Any]:
    """Execute batched symmetric clustering using numpy."""
    batch_size = vals.shape[0]
    total_len = vals.shape[-1]
    window = max(1, int(policy.cluster_window))

    sq_mag = vals.real**2 + vals.imag**2
    sq_mag = np.nan_to_num(sq_mag, nan=0.0)

    num_blocks = (total_len + window - 1) // window
    if num_blocks == 0:
        return {
            "aborted": False,
            "overflow": False,
            "candidates": {
                "template_idx": np.empty(0, dtype=np.int64),
                "sample_idx": np.empty(0, dtype=np.int64),
                "snr": np.empty(0, dtype=np.complex64),
                "sigmasq": np.empty(0, dtype=np.float32),
            },
        }

    pad = num_blocks * window - total_len
    if pad > 0:
        pad_val = np.full((batch_size, pad), -np.inf, dtype=sq_mag.dtype)
        sq_padded = np.concatenate([sq_mag, pad_val], axis=-1)
    else:
        sq_padded = sq_mag

    blocks = sq_padded.reshape(batch_size, num_blocks, window)
    block_max = np.max(blocks, axis=-1)
    block_idx = np.argmax(blocks, axis=-1)
    block_idx = block_idx + np.arange(num_blocks)[np.newaxis, :] * window

    safe_norms = np.maximum(norms, 1e-12)
    thresh_sq = (policy.snr_threshold / safe_norms) ** 2
    thresh_sq = thresh_sq[:, np.newaxis]
    keep = block_max > thresh_sq

    if policy.snr_abort_threshold is not None:
        abort_thresh_sq = (policy.snr_abort_threshold / safe_norms) ** 2
        abort_thresh_sq = abort_thresh_sq[:, np.newaxis]
        if (block_max >= abort_thresh_sq).any():
            return {"aborted": True, "candidates": {}}

    if num_blocks > 1:
        keep[:, 1:] &= block_max[:, 1:] > block_max[:, :-1]
        keep[:, :-1] &= block_max[:, :-1] >= block_max[:, 1:]
        keep[:, 0] &= block_max[:, 0] > block_max[:, 1]

    survivor_indices = np.nonzero(keep)
    sel_tmplt = survivor_indices[0]
    block_pos = survivor_indices[1]
    num_survivors = len(sel_tmplt)

    if num_survivors == 0:
        return {
            "aborted": False,
            "overflow": False,
            "candidates": {
                "template_idx": np.empty(0, dtype=np.int64),
                "sample_idx": np.empty(0, dtype=np.int64),
                "snr": np.empty(0, dtype=np.complex64),
                "sigmasq": np.empty(0, dtype=np.float32),
            },
        }

    sel_sample_rel = block_idx[sel_tmplt, block_pos]
    sel_sample = sel_sample_rel + valid_start
    peak_samples = vals[sel_tmplt, sel_sample_rel]
    sel_snr = peak_samples * norms[sel_tmplt]
    sel_sigmasq = sigmasqs[sel_tmplt]

    overflow = _store_in_buffer(
        buffer, sel_tmplt, sel_sample, sel_snr, sel_sigmasq, num_survivors
    )
    if overflow:
        return {"aborted": False, "overflow": True, "candidates": {}}

    return {
        "aborted": False,
        "overflow": False,
        "candidates": {
            "template_idx": sel_tmplt,
            "sample_idx": sel_sample,
            "snr": sel_snr,
            "sigmasq": sel_sigmasq,
        },
    }


def select_tile_candidates(
    out_mem: Any,  # (B, transform_length) complex64
    tile_norms: Any,  # (B,) float32/float64
    tile_sigmasqs: Any,  # (B,) float32
    valid_start: int,
    valid_end: int,
    policy: SelectionPolicy,
    buffer: Optional[CandidateBuffer] = None,
) -> Dict[str, Any]:
    """
    Execute selection policy on device-resident tile outputs without per-row host round trips.
    """
    valid_slice = slice(valid_start, valid_end)

    if torch is not None and isinstance(out_mem, torch.Tensor):
        vals = out_mem[:, valid_slice]
        batch_size = vals.shape[0]

        norms = (
            tile_norms
            if isinstance(tile_norms, torch.Tensor)
            else torch.as_tensor(tile_norms, device=vals.device)
        )
        sigmasqs = (
            tile_sigmasqs
            if isinstance(tile_sigmasqs, torch.Tensor)
            else torch.as_tensor(tile_sigmasqs, device=vals.device)
        )

        if policy.cluster_policy == "symmetric":
            return _select_torch_symmetric(
                vals, norms, sigmasqs, valid_start, policy, buffer
            )

        # 1-peak per template (live_peak policy)
        sq_mag = vals.real.square() + vals.imag.square()
        sq_mag = torch.nan_to_num(sq_mag, nan=0.0)

        max_sq_mag, argmax_idx = torch.max(sq_mag, dim=-1)
        peak_samples = vals[
            torch.arange(batch_size, device=vals.device), argmax_idx
        ]

        snr_mags = torch.sqrt(max_sq_mag) * norms

        if policy.snr_abort_threshold is not None:
            if (snr_mags >= policy.snr_abort_threshold).any():
                return {"aborted": True, "candidates": {}}

        survivors = snr_mags >= policy.snr_threshold
        survivor_indices = torch.nonzero(survivors, as_tuple=False).squeeze(-1)

        num_survivors = int(survivor_indices.numel())
        if num_survivors == 0:
            return {
                "aborted": False,
                "overflow": False,
                "candidates": {
                    "template_idx": np.empty(0, dtype=np.int64),
                    "sample_idx": np.empty(0, dtype=np.int64),
                    "snr": np.empty(0, dtype=np.complex64),
                    "sigmasq": np.empty(0, dtype=np.float32),
                },
            }

        sel_tmplt = survivor_indices
        sel_sample = argmax_idx[sel_tmplt] + valid_start
        sel_snr = peak_samples[sel_tmplt] * norms[sel_tmplt]
        sel_sigmasq = sigmasqs[sel_tmplt]

        overflow = _store_in_buffer(
            buffer, sel_tmplt, sel_sample, sel_snr, sel_sigmasq, num_survivors
        )
        if overflow:
            return {"aborted": False, "overflow": True, "candidates": {}}

        return {
            "aborted": False,
            "overflow": False,
            "candidates": {
                "template_idx": sel_tmplt.detach().cpu().numpy(),
                "sample_idx": sel_sample.detach().cpu().numpy(),
                "snr": sel_snr.detach().cpu().numpy(),
                "sigmasq": sel_sigmasq.detach().cpu().numpy(),
            },
        }

    # Numpy fallback
    out_np = np.asarray(out_mem)[:, valid_slice]
    batch_size = out_np.shape[0]
    norms = np.asarray(tile_norms)
    sigmasqs = np.asarray(tile_sigmasqs)

    if policy.cluster_policy == "symmetric":
        return _select_numpy_symmetric(
            out_np, norms, sigmasqs, valid_start, policy, buffer
        )

    sq_mag = out_np.real**2 + out_np.imag**2
    sq_mag = np.nan_to_num(sq_mag, nan=0.0)

    argmax_idx = np.argmax(sq_mag, axis=-1)
    max_sq_mag = sq_mag[np.arange(batch_size), argmax_idx]
    peak_samples = out_np[np.arange(batch_size), argmax_idx]

    snr_mags = np.sqrt(max_sq_mag) * norms

    if policy.snr_abort_threshold is not None:
        if (snr_mags >= policy.snr_abort_threshold).any():
            return {"aborted": True, "candidates": {}}

    survivors = np.where(snr_mags >= policy.snr_threshold)[0]
    num_survivors = len(survivors)
    if num_survivors == 0:
        return {
            "aborted": False,
            "overflow": False,
            "candidates": {
                "template_idx": np.empty(0, dtype=np.int64),
                "sample_idx": np.empty(0, dtype=np.int64),
                "snr": np.empty(0, dtype=np.complex64),
                "sigmasq": np.empty(0, dtype=np.float32),
            },
        }

    sel_tmplt = survivors
    sel_sample = argmax_idx[sel_tmplt] + valid_start
    sel_snr = peak_samples[sel_tmplt] * norms[sel_tmplt]
    sel_sigmasq = sigmasqs[sel_tmplt]

    overflow = _store_in_buffer(
        buffer, sel_tmplt, sel_sample, sel_snr, sel_sigmasq, num_survivors
    )
    if overflow:
        return {"aborted": False, "overflow": True, "candidates": {}}

    return {
        "aborted": False,
        "overflow": False,
        "candidates": {
            "template_idx": sel_tmplt,
            "sample_idx": sel_sample,
            "snr": sel_snr,
            "sigmasq": sel_sigmasq,
        },
    }
