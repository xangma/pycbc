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
Early consistency screening for candidate triggers in the persistent GPU search engine.
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
    Early consistency screener to reject obvious non-Gaussian glitches
    before evaluating expensive full-statistic vetoes.
    """

    screen_threshold: float = 50.0
    device: str = "cpu"

    def __post_init__(self):
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

        sample_idx = candidates["sample_idx"]
        tmplt_idx = candidates["template_idx"]
        snr_vals = candidates["snr"]

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
            mid_bin = max(1, num_bins // 2)
            k_mids = edges_np[tmplt_idx, mid_bin]
            k_mins = edges_np[tmplt_idx, 0]
        else:
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

                k = torch.arange(k_min, k_mid, device=dev, dtype=torch.float32)
                phase = torch.exp(2j * np.pi * k * (t / t_len))
                corr_slice = corr_tile[m, k_min:k_mid]
                z1 = torch.sum(corr_slice * phase).item()

                # 2-bin test: chi2_2 = |2 * z1 * norm - snr|^2
                res = 2.0 * z1 * norm - snr
                chisq_vals[j] = float(res.real**2 + res.imag**2)
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

                k = np.arange(k_min, k_mid, dtype=np.float32)
                phase = np.exp(2j * np.pi * k * (t / t_len))
                corr_slice = corr_tile_np[m, k_min:k_mid]
                z1 = np.sum(corr_slice * phase)

                res = 2.0 * z1 * norm - snr
                chisq_vals[j] = float(res.real**2 + res.imag**2)

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
        Screen candidates, dropping those whose consistency score exceeds screen_threshold.
        """
        num_cands = len(candidates.get("sample_idx", []))
        if num_cands == 0:
            return candidates

        scores = self.evaluate_screening_chisq(
            corr_tile=corr_tile,
            candidates=candidates,
            tile_norms=tile_norms,
            transform_length=transform_length,
            bin_edges=bin_edges,
        )

        keep = scores <= self.screen_threshold
        num_survivors = int(np.sum(keep))
        num_rejected = num_cands - num_survivors

        self.accepted_count += num_survivors
        self.rejected_count += num_rejected

        if num_survivors == num_cands:
            candidates["screen_chisq"] = scores
            return candidates

        filtered = {}
        for key, arr in candidates.items():
            if isinstance(arr, np.ndarray):
                filtered[key] = arr[keep]
            elif isinstance(arr, list):
                filtered[key] = [arr[idx] for idx, b in enumerate(keep) if b]
            else:
                filtered[key] = arr

        filtered["screen_chisq"] = scores[keep]
        return filtered
