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
[EXPERIMENTAL PROTOTYPE - Milestone M7]
Early consistency screening for candidate triggers in the persistent GPU search engine.
This module is an experimental research prototype not yet qualified for
production pipeline execution.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional
import numpy as np

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("pycbc.filter.gpu_search.screening")


@dataclass
class ConsistencyScreen:
    """
    Optional diagnostic two-group score, without rejection by default.

    A fixed score cutoff is not a conservative reweighted-SNR cut. In
    compatible mode return candidates unchanged; ``diagnostics=True`` attaches
    scores but keeps every candidate. ``experimental_rejection=True`` enables
    the historical, uncertified cutoff and can discard accepted signals.
    Diagnostic/experimental evaluation may synchronize and copy to the host.
    """

    screen_threshold: float = 50.0
    device: str = "cpu"
    experimental_rejection: bool = False
    diagnostics: bool = False

    def __post_init__(self):
        if not np.isfinite(self.screen_threshold) or self.screen_threshold < 0:
            raise ValueError("screen_threshold must be finite and nonnegative")
        self.rejected_count = 0
        self.accepted_count = 0

    def evaluate_screening_chisq(
        self,
        corr_tile: Any,
        candidates: Dict[str, Any],
        tile_norms: Any,
        transform_length: int,
        bin_edges: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Compute fast 2-bin energy consistency chi-squared for candidate triggers.
        """
        num_cands = len(candidates.get("sample_idx", []))
        if num_cands == 0:
            return np.empty(0, dtype=np.float32)

        def host_array(value):
            if torch is not None and isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        sample_idx = host_array(candidates["sample_idx"])
        tmplt_idx = host_array(candidates["template_idx"])
        snr_vals = host_array(candidates["snr"])

        # Convert norms to numpy / cpu if needed
        if hasattr(tile_norms, "cpu"):
            tile_norms_np = tile_norms.cpu().numpy()
        else:
            tile_norms_np = np.asarray(tile_norms)

        flen = corr_tile.shape[-1]
        # Determine split index k_mid
        if bin_edges is not None:
            if hasattr(bin_edges, "cpu"):
                edges_np = bin_edges.cpu().numpy()
            else:
                edges_np = np.asarray(bin_edges)
            num_bins = edges_np.shape[-1] - 1
            if num_bins < 2:
                raise ValueError("Screening requires at least two full-statistic bins")
            mid_bin = num_bins // 2
            fraction = mid_bin / num_bins
            k_mids = edges_np[tmplt_idx, mid_bin]
            k_mins = edges_np[tmplt_idx, 0]
        else:
            fraction = 0.5
            k_mins = np.zeros(num_cands, dtype=np.int64)
            k_mids = np.full(num_cands, flen // 2, dtype=np.int64)

        chisq_vals = np.zeros(num_cands, dtype=np.float32)

        if torch is not None and str(self.device) != "numpy" and isinstance(corr_tile, torch.Tensor):
            dev = corr_tile.device
            t_len = float(transform_length)
            
            for j in range(num_cands):
                t = int(sample_idx[j])
                m = int(tmplt_idx[j])
                k_min = int(k_mins[j])
                k_mid = int(k_mids[j])
                norm = float(tile_norms_np[m])
                snr = complex(snr_vals[j])

                if k_mid <= k_min:
                    chisq_vals[j] = 0.0
                    continue

                k = torch.arange(k_min, k_mid, device=dev, dtype=torch.float64)
                phase = torch.exp(2j * np.pi * torch.remainder(k * t, t_len) / t_len)
                corr_slice = corr_tile[m, k_min:k_mid].to(torch.complex128)
                z1 = torch.sum(corr_slice * phase).item()

                # Group fractions matter when the full bin count is odd.
                # This is an exact-arithmetic lower bound only when z is
                # the sum of the same full bins; no rounding bound is supplied.
                res = z1 * norm - fraction * snr
                chisq_vals[j] = float(abs(res)**2 / (fraction * (1 - fraction)))
        else:
            corr_tile_np = corr_tile.cpu().numpy() if hasattr(corr_tile, "cpu") else np.asarray(corr_tile)
            t_len = float(transform_length)

            for j in range(num_cands):
                t = int(sample_idx[j])
                m = int(tmplt_idx[j])
                k_min = int(k_mins[j])
                k_mid = int(k_mids[j])
                norm = float(tile_norms_np[m])
                snr = complex(snr_vals[j])

                if k_mid <= k_min:
                    chisq_vals[j] = 0.0
                    continue

                k = np.arange(k_min, k_mid, dtype=np.float64)
                phase = np.exp(2j * np.pi * np.remainder(k * t, t_len) / t_len)
                corr_slice = corr_tile_np[m, k_min:k_mid].astype(np.complex128)
                z1 = np.sum(corr_slice * phase)

                res = z1 * norm - fraction * snr
                chisq_vals[j] = float(abs(res)**2 / (fraction * (1 - fraction)))

        return chisq_vals

    def filter(
        self,
        corr_tile: Any,
        candidates: Dict[str, Any],
        tile_id: int,
        tile_norms: Any,
        transform_length: int,
        bin_edges: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Retain all candidates unless uncertified rejection is explicitly enabled.
        """
        num_cands = len(candidates.get("sample_idx", []))
        if num_cands == 0:
            return candidates
        if not self.experimental_rejection and not self.diagnostics:
            self.accepted_count += num_cands
            return candidates

        scores = self.evaluate_screening_chisq(
            corr_tile=corr_tile,
            candidates=candidates,
            tile_norms=tile_norms,
            transform_length=transform_length,
            bin_edges=bin_edges,
        )

        keep = (
            scores <= self.screen_threshold if self.experimental_rejection
            else np.ones(num_cands, dtype=bool)
        )
        num_survivors = int(np.sum(keep))
        num_rejected = num_cands - num_survivors

        self.accepted_count += num_survivors
        self.rejected_count += num_rejected

        if num_survivors == num_cands:
            candidates["screen_chisq"] = scores
            return candidates

        filtered = {}
        for key, arr in candidates.items():
            if torch is not None and isinstance(arr, torch.Tensor):
                filtered[key] = arr[torch.as_tensor(keep, device=arr.device)]
            elif isinstance(arr, np.ndarray):
                filtered[key] = arr[keep]
            elif isinstance(arr, list):
                filtered[key] = [arr[idx] for idx, b in enumerate(keep) if b]
            else:
                filtered[key] = arr

        filtered["screen_chisq"] = scores[keep]
        return filtered
