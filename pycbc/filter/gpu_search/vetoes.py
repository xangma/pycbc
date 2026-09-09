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
Batched full-statistic vetoes for persistent GPU search engine.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import math
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pycbc.filter.matchedfilter import get_cutoff_indices
from pycbc.filter.gpu_search.plans import BankPlan, PSDPlan


@dataclass
class PowerChisqPlan:
    """Precomputed power chi-square configuration and device-resident bin edges."""

    num_bins: int = 16
    snr_threshold: Optional[float] = None
    tile_bin_edges: Dict[int, Any] = field(default_factory=dict)
    device: str = "cpu"

    @property
    def dof(self) -> int:
        """Statistical degrees of freedom: 2 * num_bins - 2."""
        return 2 * self.num_bins - 2


def prepare_power_chisq_plan(
    bank_plan: BankPlan,
    psd_plan: PSDPlan,
    num_bins: int = 16,
    snr_threshold: Optional[float] = None,
    device: str = "cpu",
) -> PowerChisqPlan:
    """
    Precompute and cache equal-power bin edges on device for all bank tiles.
    """
    flen = bank_plan.geometry.filter_length
    tlen = bank_plan.geometry.transform_length
    delta_f = bank_plan.geometry.delta_f

    kmin, kmax = get_cutoff_indices(
        bank_plan.geometry.f_lower,
        bank_plan.geometry.f_upper,
        delta_f,
        tlen,
    )

    tile_bin_edges = {}

    if torch is not None and device != "numpy":
        dev = torch.device(device)
        psd_t = (
            psd_plan.psd_data
            if isinstance(psd_plan.psd_data, torch.Tensor)
            else torch.as_tensor(psd_plan.psd_data, device=dev, dtype=torch.float32)
        )
        inv_psd = torch.where(
            psd_t[:flen] > 0,
            (4.0 * delta_f) / psd_t[:flen],
            torch.zeros(flen, device=dev, dtype=torch.float32),
        )
        inv_psd[:kmin].zero_()
        if kmax < flen:
            inv_psd[kmax:].zero_()

        for tile in bank_plan.tiles:
            b = tile.batch_size
            pm = tile.power_matrix[:b]
            if not isinstance(pm, torch.Tensor):
                pm = torch.as_tensor(pm, device=dev, dtype=torch.float32)

            pd = pm * inv_psd.unsqueeze(0)
            s_t = torch.cumsum(pd, dim=-1)

            sigmasq_t = s_t[:, kmax - 1].to(torch.float64)
            edge_vec_t = (
                torch.arange(num_bins, dtype=torch.float64, device=dev).unsqueeze(0)
                * sigmasq_t.unsqueeze(1)
            ) / num_bins

            sub_s = s_t[:, kmin:kmax].to(torch.float64).contiguous()
            bins_2d = torch.searchsorted(sub_s, edge_vec_t, right=True) + kmin
            kmax_col = torch.full((b, 1), kmax, device=dev, dtype=bins_2d.dtype)
            all_bins = torch.cat([bins_2d, kmax_col], dim=-1)
            tile_bin_edges[tile.tile_id] = all_bins
    else:
        psd_np = (
            psd_plan.psd_data.numpy()
            if hasattr(psd_plan.psd_data, "numpy")
            else np.asarray(psd_plan.psd_data)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_psd_np = np.where(
                psd_np[:flen] > 0, (4.0 * delta_f) / psd_np[:flen], 0.0
            ).astype(np.float32)
        inv_psd_np[:kmin] = 0.0
        if kmax < flen:
            inv_psd_np[kmax:] = 0.0

        for tile in bank_plan.tiles:
            b = tile.batch_size
            pm_np = np.asarray(tile.power_matrix[:b])
            pd_np = pm_np * inv_psd_np[np.newaxis, :]
            s_np = np.cumsum(pd_np, axis=-1)

            all_bins_list = []
            for i in range(b):
                sigmasq_val = s_np[i, kmax - 1]
                edge_vec = np.arange(num_bins, dtype=np.float64) * sigmasq_val / num_bins
                bins_i = (
                    np.searchsorted(s_np[i, kmin:kmax], edge_vec, side="right") + kmin
                )
                all_bins_list.append(np.append(bins_i, kmax))

            tile_bin_edges[tile.tile_id] = np.stack(all_bins_list, axis=0)

    return PowerChisqPlan(
        num_bins=num_bins,
        snr_threshold=snr_threshold,
        tile_bin_edges=tile_bin_edges,
        device=device,
    )


def batched_power_chisq(
    corr_tile: Any,  # (B, transform_length) complex64
    candidates: Dict[str, Any],
    tile_bin_edges: Any,  # (B, num_bins + 1)
    tile_norms: Any,  # (B,) float32
    num_bins: int,
    snr_threshold: Optional[float],
    transform_length: int,
    chunk_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate power chi-square statistic for candidates with double-precision phase stability.

    Returns:
        (chisq, chisq_dof): numpy arrays of shape (num_candidates,)
    """
    sample_indices = candidates.get("sample_idx", [])
    if len(sample_indices) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

    template_indices = candidates["template_idx"]
    snrs = candidates["snr"]
    dof_val = 2 * num_bins - 2
    N = int(transform_length)
    M = len(sample_indices)

    if torch is not None and isinstance(corr_tile, torch.Tensor):
        dev = corr_tile.device
        flen = min(corr_tile.shape[-1], N // 2 + 1)

        t_indices = (
            template_indices
            if isinstance(template_indices, torch.Tensor)
            else torch.as_tensor(template_indices, device=dev, dtype=torch.int64)
        )
        s_indices = (
            sample_indices
            if isinstance(sample_indices, torch.Tensor)
            else torch.as_tensor(sample_indices, device=dev, dtype=torch.int64)
        )
        snr_vals = (
            snrs
            if isinstance(snrs, torch.Tensor)
            else torch.as_tensor(snrs, device=dev, dtype=torch.complex64)
        )
        norms = (
            tile_norms
            if isinstance(tile_norms, torch.Tensor)
            else torch.as_tensor(tile_norms, device=dev, dtype=torch.float32)
        )
        bin_edges = (
            tile_bin_edges
            if isinstance(tile_bin_edges, torch.Tensor)
            else torch.as_tensor(tile_bin_edges, device=dev, dtype=torch.int64)
        )

        chisq_out = torch.zeros(M, device=dev, dtype=torch.float32)
        dof_out = torch.full((M,), dof_val, device=dev, dtype=torch.int32)

        # Activation mask
        snr_mags = torch.abs(snr_vals)
        if snr_threshold is not None:
            active = snr_mags >= snr_threshold
        else:
            active = torch.ones(M, device=dev, dtype=torch.bool)

        active_indices = torch.nonzero(active, as_tuple=False).squeeze(-1)
        num_active = int(active_indices.numel())

        if num_active > 0:
            two_pi_over_N = (2.0 * math.pi) / N
            k_range = torch.arange(flen, device=dev, dtype=torch.float64).unsqueeze(0)

            for c_start in range(0, num_active, chunk_size):
                c_end = min(c_start + chunk_size, num_active)
                chunk_act = active_indices[c_start:c_end]
                cur_M = len(chunk_act)

                cur_tmplt = t_indices[chunk_act]
                cur_sample = s_indices[chunk_act]
                cur_snr = snr_vals[chunk_act]
                cur_norm = norms[cur_tmplt]

                # Exact integer modulo in double precision for phase stability
                pts_d = cur_sample.unsqueeze(1).to(torch.float64)
                phase_angle = two_pi_over_N * torch.fmod(k_range * pts_d, N)
                phases = torch.complex(
                    torch.cos(phase_angle).to(torch.float32),
                    torch.sin(phase_angle).to(torch.float32),
                )

                # Corr row per candidate: shape (cur_M, flen)
                cur_corr = corr_tile[cur_tmplt, :flen]
                C = cur_corr * phases

                # Bounded cumulative sum
                zero_col = torch.zeros((cur_M, 1), device=dev, dtype=C.dtype)
                S = torch.cat([zero_col, torch.cumsum(C, dim=-1)], dim=-1)

                cur_bin_starts = bin_edges[cur_tmplt, :-1]
                cur_bin_ends = bin_edges[cur_tmplt, 1:]

                S_starts = torch.gather(S, dim=1, index=cur_bin_starts)
                S_ends = torch.gather(S, dim=1, index=cur_bin_ends)
                zb = S_ends - S_starts

                zb_sq = zb.real.square() + zb.imag.square()
                chisq_raw = torch.sum(zb_sq, dim=-1)

                snr_sq = cur_snr.real.square() + cur_snr.imag.square()
                chisq_vals = (num_bins * chisq_raw) * cur_norm.square() - snr_sq
                chisq_out[chunk_act] = torch.clamp(chisq_vals, min=0.0)

        return chisq_out.detach().cpu().numpy(), dof_out.detach().cpu().numpy()

    # Numpy fallback
    corr_np = np.asarray(corr_tile)
    flen = min(corr_np.shape[-1], N // 2 + 1)
    t_indices = np.asarray(template_indices, dtype=np.int64)
    s_indices = np.asarray(sample_indices, dtype=np.int64)
    snr_vals = np.asarray(snrs, dtype=np.complex64)
    norms = np.asarray(tile_norms, dtype=np.float32)
    bin_edges = np.asarray(tile_bin_edges, dtype=np.int64)

    chisq_out = np.zeros(M, dtype=np.float32)
    dof_out = np.full(M, dof_val, dtype=np.int32)

    snr_mags = np.abs(snr_vals)
    if snr_threshold is not None:
        active = snr_mags >= snr_threshold
    else:
        active = np.ones(M, dtype=bool)

    active_indices = np.nonzero(active)[0]
    num_active = len(active_indices)

    if num_active > 0:
        two_pi_over_N = (2.0 * math.pi) / N
        k_range = np.arange(flen, dtype=np.float64)[np.newaxis, :]

        for c_start in range(0, num_active, chunk_size):
            c_end = min(c_start + chunk_size, num_active)
            chunk_act = active_indices[c_start:c_end]
            cur_M = len(chunk_act)

            cur_tmplt = t_indices[chunk_act]
            cur_sample = s_indices[chunk_act]
            cur_snr = snr_vals[chunk_act]
            cur_norm = norms[cur_tmplt]

            pts_d = cur_sample[:, np.newaxis].astype(np.float64)
            phase_angle = two_pi_over_N * ((k_range * pts_d) % N)
            phases = (np.cos(phase_angle) + 1j * np.sin(phase_angle)).astype(
                np.complex64
            )

            cur_corr = corr_np[cur_tmplt, :flen]
            C = cur_corr * phases

            zero_col = np.zeros((cur_M, 1), dtype=C.dtype)
            S = np.concatenate([zero_col, np.cumsum(C, axis=-1)], axis=-1)

            cur_bin_starts = bin_edges[cur_tmplt, :-1]
            cur_bin_ends = bin_edges[cur_tmplt, 1:]

            S_starts = np.take_along_axis(S, cur_bin_starts, axis=1)
            S_ends = np.take_along_axis(S, cur_bin_ends, axis=1)
            zb = S_ends - S_starts

            zb_sq = zb.real**2 + zb.imag**2
            chisq_raw = np.sum(zb_sq, axis=-1)

            snr_sq = cur_snr.real**2 + cur_snr.imag**2
            chisq_vals = (num_bins * chisq_raw) * (cur_norm**2) - snr_sq
            chisq_out[chunk_act] = np.maximum(chisq_vals, 0.0)

    return chisq_out, dof_out


@dataclass
class SineGaussianPlan:
    """Optional sine-Gaussian veto configuration."""

    snr_threshold: float = 6.0
    chisq_locations: Optional[Dict[str, str]] = None


@dataclass
class VetoManager:
    """Manager coordinating power chi-square and auxiliary veto evaluations."""

    power_chisq_plan: Optional[PowerChisqPlan] = None
    sg_plan: Optional[SineGaussianPlan] = None

    def evaluate(
        self,
        corr_tile: Any,
        candidates: Dict[str, Any],
        tile_id: int,
        tile_norms: Any,
        transform_length: int,
    ) -> Dict[str, Any]:
        """
        Evaluate configured vetoes for candidate triggers while correlation is live.
        """
        if not candidates or len(candidates.get("sample_idx", [])) == 0:
            return candidates

        if self.power_chisq_plan is not None:
            bin_edges = self.power_chisq_plan.tile_bin_edges.get(tile_id)
            if bin_edges is not None:
                chisq, chisq_dof = batched_power_chisq(
                    corr_tile=corr_tile,
                    candidates=candidates,
                    tile_bin_edges=bin_edges,
                    tile_norms=tile_norms,
                    num_bins=self.power_chisq_plan.num_bins,
                    snr_threshold=self.power_chisq_plan.snr_threshold,
                    transform_length=transform_length,
                )
                candidates["chisq"] = chisq
                candidates["chisq_dof"] = chisq_dof

        return candidates
