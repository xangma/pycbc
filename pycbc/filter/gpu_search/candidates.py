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

        # Compute square magnitude on device
        sq_mag = vals.real.square() + vals.imag.square()
        sq_mag = torch.nan_to_num(sq_mag, nan=0.0)

        if policy.cluster_policy == "live_peak":
            max_sq_mag, argmax_idx = torch.max(sq_mag, dim=-1)
            # Complex peak sample
            peak_samples = vals[
                torch.arange(batch_size, device=vals.device), argmax_idx
            ]

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

            # SNR magnitude = sqrt(max_sq_mag) * norm
            snr_mags = torch.sqrt(max_sq_mag) * norms

            # Check abort threshold
            if policy.snr_abort_threshold is not None:
                if (snr_mags >= policy.snr_abort_threshold).any():
                    return {"aborted": True, "candidates": {}}

            survivors = snr_mags >= policy.snr_threshold
            survivor_indices = torch.nonzero(survivors, as_tuple=False).squeeze(-1)

            num_survivors = int(survivor_indices.numel())
            if num_survivors == 0:
                return {
                    "aborted": False,
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

            if buffer is not None:
                start_c = buffer.count
                if start_c + num_survivors > buffer.capacity:
                    buffer.overflow_flag.fill_(True)
                    return {"aborted": False, "overflow": True, "candidates": {}}

                buffer.template_indices[start_c : start_c + num_survivors].copy_(
                    sel_tmplt
                )
                buffer.sample_indices[start_c : start_c + num_survivors].copy_(
                    sel_sample
                )
                buffer.snr_values[start_c : start_c + num_survivors].copy_(sel_snr)
                buffer.sigmasq_values[start_c : start_c + num_survivors].copy_(
                    sel_sigmasq
                )
                buffer.device_count.add_(num_survivors)

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
    sq_mag = out_np.real**2 + out_np.imag**2
    sq_mag = np.nan_to_num(sq_mag, nan=0.0)

    argmax_idx = np.argmax(sq_mag, axis=-1)
    max_sq_mag = sq_mag[np.arange(batch_size), argmax_idx]
    peak_samples = out_np[np.arange(batch_size), argmax_idx]

    norms = np.asarray(tile_norms)
    sigmasqs = np.asarray(tile_sigmasqs)
    snr_mags = np.sqrt(max_sq_mag) * norms

    if policy.snr_abort_threshold is not None:
        if (snr_mags >= policy.snr_abort_threshold).any():
            return {"aborted": True, "candidates": {}}

    survivors = np.where(snr_mags >= policy.snr_threshold)[0]
    num_survivors = len(survivors)
    if num_survivors == 0:
        return {
            "aborted": False,
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
