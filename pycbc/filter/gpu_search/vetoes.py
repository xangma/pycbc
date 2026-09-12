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
from typing import Any, Dict, Optional, Sequence, Tuple
import math
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pycbc.types import FrequencySeries
from pycbc.filter.matchedfilter import get_cutoff_indices
from pycbc.filter.gpu_search.plans import BankPlan, PSDPlan
from pycbc.filter.gpu_search.candidates import candidates_to_host


@dataclass
class PowerChisqPlan:
    """Precomputed power chi-square configuration and device-resident bin edges."""

    num_bins: Any = 16
    snr_threshold: Optional[float] = None
    tile_bin_edges: Dict[int, Any] = field(default_factory=dict)
    tile_num_bins: Dict[int, Any] = field(default_factory=dict)
    device: str = "cpu"

    @property
    def dof(self) -> Any:
        """Statistical degrees of freedom: 2 * num_bins - 2."""
        if isinstance(self.num_bins, int):
            return 2 * self.num_bins - 2
        return 2 * np.asarray(self.num_bins) - 2


def _wrap_tmpl_params(tmpl: Any) -> Any:
    if tmpl is None or not hasattr(tmpl, "params"):
        return tmpl
    p = getattr(tmpl, "params")
    if (
        isinstance(p, np.void)
        and hasattr(p, "dtype")
        and p.dtype.names is not None
    ):
        import types

        ns = types.SimpleNamespace(**{k: p[k] for k in p.dtype.names})

        class _WrappedTmpl:
            pass

        wt = _WrappedTmpl()
        for attr in ("id", "f_lower", "delta_f", "epoch"):
            if hasattr(tmpl, attr):
                setattr(wt, attr, getattr(tmpl, attr))
        wt.params = ns
        return wt
    return tmpl


def _resolve_num_bins(
    num_bins_spec: Any, t_id: int, tmpl: Optional[Any] = None
) -> int:
    if isinstance(num_bins_spec, dict):
        if t_id in num_bins_spec:
            return int(num_bins_spec[t_id])
        if tmpl is not None and getattr(tmpl, "id", None) in num_bins_spec:
            return int(num_bins_spec[tmpl.id])
    if isinstance(num_bins_spec, (list, tuple, np.ndarray)):
        if t_id < len(num_bins_spec):
            return int(num_bins_spec[t_id])
    if callable(num_bins_spec):
        try:
            return int(num_bins_spec(tmpl if tmpl is not None else t_id))
        except Exception:
            pass
    if isinstance(num_bins_spec, str):
        if tmpl is not None:
            eval_tmpl = _wrap_tmpl_params(tmpl)
            try:
                from pycbc.vetoes.chisq import SingleDetPowerChisq

                return int(
                    SingleDetPowerChisq.parse_option(eval_tmpl, num_bins_spec)
                )
            except Exception:
                pass
        try:
            return int(eval(num_bins_spec, {"__builtins__": None}, {}))
        except Exception:
            pass
        try:
            return int(num_bins_spec)
        except Exception:
            pass
    try:
        return int(num_bins_spec)
    except Exception:
        return 16


def prepare_power_chisq_plan(
    bank_plan: BankPlan,
    psd_plan: PSDPlan,
    num_bins: Any = 16,
    snr_threshold: Optional[float] = None,
    device: str = "cpu",
) -> PowerChisqPlan:
    """
    Precompute and cache equal-power bin edges on device for all bank tiles.
    Supports scalar, per-template mapped, and expression-derived bin counts.
    """
    tlen = bank_plan.geometry.transform_length
    delta_f = bank_plan.geometry.delta_f

    kmin, kmax = get_cutoff_indices(
        bank_plan.geometry.f_lower,
        bank_plan.geometry.f_upper,
        delta_f,
        tlen,
    )

    tile_bin_edges = {}
    tile_num_bins = {}

    psd_np = (
        psd_plan.psd_data.detach().cpu().numpy()
        if hasattr(psd_plan.psd_data, "detach")
        else (
            psd_plan.psd_data.numpy()
            if hasattr(psd_plan.psd_data, "numpy")
            else np.asarray(psd_plan.psd_data)
        )
    )
    psd_slice = psd_np[kmin:kmax]

    is_torch = torch is not None and device != "numpy"
    dev = torch.device(device) if is_torch else None

    for tile in bank_plan.tiles:
        b = tile.batch_size
        pm = tile.power_matrix[:b]
        if hasattr(pm, "detach"):
            pm_np = pm.detach().cpu().numpy()
        elif hasattr(pm, "numpy"):
            pm_np = pm.numpy()
        else:
            pm_np = np.asarray(pm)

        with np.errstate(divide="ignore", invalid="ignore"):
            # Direct division matching historical sigmasq_series: mag /= psd
            pd_slice_np = np.where(
                psd_slice[np.newaxis, :] > 0,
                pm_np[:, kmin:kmax] / psd_slice[np.newaxis, :],
                0.0,
            ).astype(np.float32)

        s_slice_np = np.cumsum(pd_slice_np, axis=-1)

        tile_bins_list = []
        tile_nbins_list = []
        for i in range(b):
            t_id = tile.template_ids[i]
            tmpl = None
            if (
                tile is not None
                and getattr(tile, "template_by_id", None) is not None
                and t_id in tile.template_by_id
            ):
                tmpl = tile.template_by_id[t_id]
            elif (
                bank_plan is not None
                and getattr(bank_plan, "template_by_id", None) is not None
                and t_id in bank_plan.template_by_id
            ):
                tmpl = bank_plan.template_by_id[t_id]
            elif (
                tile is not None
                and getattr(tile, "templates", None) is not None
                and i < len(tile.templates)
            ):
                tmpl = tile.templates[i]
            elif (
                bank_plan is not None
                and getattr(bank_plan, "templates", None) is not None
            ):
                for idx, t in enumerate(bank_plan.templates):
                    if getattr(t, "id", idx) == t_id:
                        tmpl = t
                        break
            nb = _resolve_num_bins(num_bins, t_id, tmpl)
            tile_nbins_list.append(nb)

            sigmasq_val = float(s_slice_np[i, -1]) * (4.0 * delta_f)
            s_row = s_slice_np[i].astype(np.float64) * (4.0 * delta_f)
            edge_vec = np.arange(nb, dtype=np.float64) * sigmasq_val / nb
            bins_i = (
                np.searchsorted(s_row, edge_vec, side="right") + kmin
            )
            tile_bins_list.append(np.append(bins_i, kmax))

        is_uniform = len(set(tile_nbins_list)) <= 1
        if is_uniform:
            all_bins_np = np.stack(tile_bins_list, axis=0)
            if is_torch:
                tile_bin_edges[tile.tile_id] = torch.as_tensor(
                    all_bins_np, device=dev, dtype=torch.int64
                )
            else:
                tile_bin_edges[tile.tile_id] = all_bins_np
            tile_num_bins[tile.tile_id] = tile_nbins_list[0]
        else:
            if is_torch:
                tile_bin_edges[tile.tile_id] = [
                    torch.as_tensor(edges_i, device=dev, dtype=torch.int64)
                    for edges_i in tile_bins_list
                ]
                tile_num_bins[tile.tile_id] = torch.as_tensor(
                    tile_nbins_list, device=dev, dtype=torch.int64
                )
            else:
                tile_bin_edges[tile.tile_id] = tile_bins_list
                tile_num_bins[tile.tile_id] = np.array(
                    tile_nbins_list, dtype=np.int64
                )

    return PowerChisqPlan(
        num_bins=num_bins,
        snr_threshold=snr_threshold,
        tile_bin_edges=tile_bin_edges,
        tile_num_bins=tile_num_bins,
        device=device,
    )


def power_chisq_scratch_shape(num_bins, chunk_size, scratch_budget_bytes):
    """Conservative explicit scratch-storage bound for blocked Fourier sums.

    Budget excludes caller inputs, O(candidate count) result/index arrays,
    normalized SNR copies, allocator caches and backend-internal workspace. Allow 128 bytes per candidate-frequency
    element and 128 bytes per candidate-bin plus 256 bytes per candidate for
    concurrently live intermediates. Frequency blocks never exceed 4096.
    """
    if chunk_size < 1 or num_bins < 1:
        raise ValueError("Positive chunk size and bin count required")
    per_row = 128 * num_bins + 256
    if scratch_budget_bytes < per_row + 128:
        raise ValueError("Power chi-square scratch budget is too small")
    rows = min(int(chunk_size), int(scratch_budget_bytes) // (per_row + 128))
    width = min(4096, (int(scratch_budget_bytes) // rows - per_row) // 128)
    return rows, width


def batched_power_chisq(
    corr_tile: Any,
    candidates: Dict[str, Any],
    tile_bin_edges: Any,
    tile_norms: Any,
    num_bins: Any = 16,
    snr_threshold: Optional[float] = None,
    transform_length: int = 0,
    chunk_size: int = 2048,
    return_device: bool = False,
    scratch_budget_bytes: int = 64 * 1024 * 1024,
) -> Tuple[Any, Any]:
    """Evaluate selected-time power chi-square with bounded FP64 reductions.

    Accumulate bin contributions in complex128 from short frequency blocks;
    no full-transform candidate-by-frequency array is materialized. Bin edges
    retain the half-open reference convention, including empty/ragged bins.
    Outputs default to NumPy; return_device retains Torch inputs' device.
    This is numerical compatibility, not a certificate of identical decisions
    at floating-point boundaries against a particular reference FFT.
    """
    N = int(transform_length)
    if N < 2 or N >= 2**31:
        raise ValueError("transform_length must be in [2, 2**31)")
    use_torch = torch is not None and isinstance(corr_tile, torch.Tensor)
    flen = min(corr_tile.shape[-1], N // 2 + 1)
    ragged = isinstance(tile_bin_edges, (list, tuple))
    M = len(candidates.get("sample_idx", []))

    if use_torch:
        dev = corr_tile.device
        def as_index(value):
            return torch.as_tensor(value, device=dev, dtype=torch.int64)
        ti = as_index(candidates.get("template_idx", []))
        si = as_index(candidates.get("sample_idx", []))
        snr = torch.as_tensor(candidates.get("snr", []), device=dev,
                              dtype=torch.complex128)
        norms = torch.as_tensor(tile_norms, device=dev, dtype=torch.float64)
        edges_list = [as_index(e) for e in tile_bin_edges] if ragged else None
        edges = None if ragged else as_index(tile_bin_edges)
        counts = (torch.tensor([len(e)-1 for e in edges_list], device=dev)
                  if ragged else torch.full((corr_tile.shape[0],),
                                            edges.shape[1]-1, device=dev))
        out = torch.zeros(M, device=dev, dtype=corr_tile.real.dtype)
        dof = (2 * counts[ti] - 2).to(torch.int32)
        active = (torch.arange(M, device=dev) if snr_threshold is None else
                  torch.nonzero(torch.abs(snr) >= snr_threshold).flatten())
        max_bins = max((len(e)-1 for e in edges_list), default=1) if ragged else edges.shape[1]-1
    else:
        ti = np.asarray(candidates.get("template_idx", []), dtype=np.int64)
        si = np.asarray(candidates.get("sample_idx", []), dtype=np.int64)
        snr = np.asarray(candidates.get("snr", []), dtype=np.complex128)
        norms = np.asarray(tile_norms, dtype=np.float64)
        edges_list = [np.asarray(e, dtype=np.int64) for e in tile_bin_edges] if ragged else None
        edges = None if ragged else np.asarray(tile_bin_edges, dtype=np.int64)
        counts = (np.array([len(e)-1 for e in edges_list]) if ragged else
                  np.full(corr_tile.shape[0], edges.shape[1]-1))
        out = np.zeros(M, dtype=np.asarray(corr_tile).real.dtype)
        dof = (2 * counts[ti] - 2).astype(np.int32)
        active = (np.arange(M) if snr_threshold is None else
                  np.flatnonzero(np.abs(snr) >= snr_threshold))
        max_bins = max((len(e)-1 for e in edges_list), default=1) if ragged else edges.shape[1]-1

    rows, width = power_chisq_scratch_shape(
        max_bins, min(chunk_size, max(1, M)), scratch_budget_bytes,
    )
    # Ragged plans group by template; uniform plans process all active rows.
    # Dynamic groups and edge extents may synchronize, but candidate payloads
    # remain on device. This stage is outside correlation/IFFT graph capture.
    if ragged:
        unique = torch.unique(ti[active]) if use_torch else np.unique(ti[active])
        groups = [(active[ti[active] == t], edges_list[int(t)]) for t in unique]
    else:
        groups = [(active, None)]
    for group, row_edges in groups:
        for first in range(0, len(group), rows):
            ix = group[first:first+rows]
            ct, cs = ti[ix], si[ix]
            if use_torch:
                ce = edges[ct] if row_edges is None else row_edges.expand(len(ix), -1)
                invalid = ((ce < 0) | (ce > flen)).any() | (ce[:, 1:] < ce[:, :-1]).any()
                if bool(invalid):
                    raise ValueError("Invalid power chi-square bin edges")
                lo, hi = int(ce.min()), int(ce.max())
                zb = torch.zeros((len(ix), ce.shape[1]-1), device=dev, dtype=torch.complex128)
            else:
                ce = edges[ct] if row_edges is None else np.broadcast_to(row_edges, (len(ix), len(row_edges)))
                if np.any((ce < 0) | (ce > flen)) or np.any(np.diff(ce, axis=1) < 0):
                    raise ValueError("Invalid power chi-square bin edges")
                lo, hi = int(ce.min()), int(ce.max())
                zb = np.zeros((len(ix), ce.shape[1]-1), dtype=np.complex128)
            for start in range(lo, hi, width):
                end = min(start + width, hi)
                if use_torch:
                    k = torch.arange(start, end, device=dev, dtype=torch.int64)
                    angle = ((cs[:, None] * k) % N).to(torch.float64)
                    angle.mul_(2.0 * math.pi / N)
                    phase = torch.complex(torch.cos(angle), torch.sin(angle))
                    product = corr_tile[ct, start:end] * phase
                    # Bin-local reductions avoid subtracting nearly equal
                    # cumulative endpoints after large unrelated bins.
                    for bi in range(ce.shape[1]-1):
                        mask = ((k >= ce[:, bi:bi+1]) &
                                (k < ce[:, bi+1:bi+2]))
                        contribution = torch.where(mask, product, 0)
                        zb[:, bi].add_(contribution.sum(dim=1))
                        del mask, contribution
                else:
                    k = np.arange(start, end, dtype=np.int64)
                    angle = ((cs[:, None] * k) % N).astype(np.float64)
                    angle *= 2.0 * math.pi / N
                    phase = np.cos(angle) + 1j * np.sin(angle)
                    product = corr_tile[ct, start:end] * phase
                    for bi in range(ce.shape[1]-1):
                        mask = ((k >= ce[:, bi:bi+1]) &
                                (k < ce[:, bi+1:bi+2]))
                        contribution = np.where(mask, product, 0)
                        zb[:, bi] += contribution.sum(axis=1)
                        del mask, contribution
                # Do not retain the previous block while allocating the next.
                del angle, phase, product
            if use_torch:
                raw = (zb.real.square() + zb.imag.square()).sum(dim=1)
                vals = (ce.shape[1]-1) * raw * norms[ct].square() - snr[ix].abs().square()
                out[ix] = vals.clamp(min=0).to(out.dtype)
            else:
                raw = np.sum(zb.real**2 + zb.imag**2, axis=1)
                vals = (ce.shape[1]-1) * raw * norms[ct]**2 - np.abs(snr[ix])**2
                out[ix] = np.maximum(vals, 0)
    if use_torch and not return_device:
        return out.detach().cpu().numpy(), dof.detach().cpu().numpy()
    return out, dof


@dataclass
class SineGaussianPlan:
    """Optional sine-Gaussian veto configuration."""

    snr_threshold: float = 6.0
    chisq_locations: Optional[Dict[str, str]] = None
    sg_chisq: Optional[Any] = None
    evaluator: Optional[Any] = None
    enabled: bool = True


@dataclass
class VetoManager:
    """Manager coordinating power chi-square and auxiliary veto evaluations."""

    power_chisq_plan: Optional[PowerChisqPlan] = None
    sg_plan: Optional[SineGaussianPlan] = None
    consistency_screen: Optional[Any] = None

    def evaluate(
        self,
        corr_tile: Any,
        candidates: Dict[str, Any],
        tile_id: int,
        tile_norms: Any,
        transform_length: int,
        stilde: Optional[Any] = None,
        stilde_buf: Optional[Any] = None,
        psd_plan: Optional[Any] = None,
        tile: Optional[Any] = None,
        bank_plan: Optional[Any] = None,
        templates: Optional[Sequence[Any]] = None,
        full_stilde: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate configured vetoes for candidate triggers while correlation is live.
        """
        if not candidates or len(candidates.get("sample_idx", [])) == 0:
            return candidates

        bin_edges = None
        if self.power_chisq_plan is not None:
            bin_edges = self.power_chisq_plan.tile_bin_edges.get(tile_id)

        if self.consistency_screen is not None:
            candidates = self.consistency_screen.filter(
                corr_tile=corr_tile,
                candidates=candidates,
                tile_id=tile_id,
                tile_norms=tile_norms,
                transform_length=transform_length,
                bin_edges=bin_edges,
            )
            if not candidates or len(candidates.get("sample_idx", [])) == 0:
                return candidates

        if self.power_chisq_plan is not None and bin_edges is not None:
            chisq, chisq_dof = batched_power_chisq(
                corr_tile=corr_tile,
                candidates=candidates,
                tile_bin_edges=bin_edges,
                tile_norms=tile_norms,
                num_bins=self.power_chisq_plan.num_bins,
                snr_threshold=self.power_chisq_plan.snr_threshold,
                transform_length=transform_length,
                return_device=True,
            )
            candidates["chisq"] = chisq
            candidates["chisq_dof"] = chisq_dof

        if self.sg_plan is not None:
            # Legacy/custom SG evaluators expose a host-array protocol.
            # This is an explicit CPU boundary, after device power chi-square.
            candidates = candidates_to_host(candidates)
            num_cands = len(candidates.get("sample_idx", []))
            evaluator = getattr(self.sg_plan, "evaluator", None) or getattr(
                self.sg_plan, "sg_chisq", None
            )
            is_enabled = getattr(self.sg_plan, "enabled", True) and (
                getattr(evaluator, "do", True) if evaluator is not None else True
            )
            if evaluator is not None and is_enabled:
                res = None
                if callable(evaluator):
                    try:
                        res = evaluator(
                            corr_tile=corr_tile,
                            candidates=candidates,
                            tile_id=tile_id,
                            tile_norms=tile_norms,
                            transform_length=transform_length,
                        )
                    except TypeError:
                        try:
                            res = evaluator(candidates)
                        except TypeError:
                            res = evaluator()
                elif hasattr(evaluator, "values") and callable(evaluator.values):
                    try:
                        res = evaluator.values(
                            corr_tile=corr_tile,
                            candidates=candidates,
                            tile_id=tile_id,
                            tile_norms=tile_norms,
                            transform_length=transform_length,
                        )
                    except TypeError:
                        try:
                            res = evaluator.values(candidates)
                        except TypeError:
                            # Evaluate via SingleDetSGChisq protocol:
                            # values(stilde, template, psd, snrv, snr_norm, bchisq, bchisq_dof, indices)
                            res_list = []
                            st = (
                                full_stilde
                                if full_stilde is not None
                                else (
                                    stilde_buf
                                    if stilde_buf is not None
                                    else (
                                        stilde
                                        if stilde is not None
                                        else getattr(self.sg_plan, "stilde", None)
                                    )
                                )
                            )
                            df = (
                                getattr(st, "delta_f", None)
                                or (
                                    psd_plan.delta_f
                                    if psd_plan is not None
                                    else None
                                )
                                or (
                                    bank_plan.geometry.delta_f
                                    if bank_plan is not None
                                    else None
                                )
                                or (1.0 / transform_length)
                            )
                            if st is not None and not hasattr(st, "delta_f"):
                                st_raw = (
                                    st.detach().cpu().numpy()
                                    if hasattr(st, "detach")
                                    else (
                                        st.numpy()
                                        if hasattr(st, "numpy")
                                        else np.asarray(st)
                                    )
                                )
                                st = FrequencySeries(st_raw, delta_f=df)

                            # Prefer bound PSD from psd_plan.psd_data to match engine-prepared strain
                            psd_val = None
                            if (
                                psd_plan is not None
                                and getattr(psd_plan, "psd_data", None) is not None
                            ):
                                psd_val = getattr(
                                    psd_plan, "_bound_psd_fseries", None
                                )
                                if psd_val is None:
                                    psd_raw = psd_plan.psd_data
                                    psd_raw = (
                                        psd_raw.detach().cpu().numpy()
                                        if hasattr(psd_raw, "detach")
                                        else (
                                            psd_raw.numpy()
                                            if hasattr(psd_raw, "numpy")
                                            else np.asarray(psd_raw)
                                        )
                                    )
                                    if psd_raw.dtype != np.float32:
                                        psd_raw = psd_raw.astype(np.float32)
                                    psd_val = FrequencySeries(
                                        psd_raw, delta_f=psd_plan.delta_f
                                    )
                                    try:
                                        psd_plan._bound_psd_fseries = psd_val
                                    except Exception:
                                        pass

                            if psd_val is None:
                                # When bound PSD is unavailable, select first non-None PSD
                                # from supported sources without suppressing valid inputs.
                                for candidate_psd in [
                                    getattr(st, "psd", None) if st is not None else None,
                                    getattr(psd_plan, "psd", None) if psd_plan is not None else None,
                                    getattr(self.sg_plan, "psd", None) if self.sg_plan is not None else None,
                                ]:
                                    if candidate_psd is not None:
                                        psd_val = candidate_psd
                                        break

                                if psd_val is not None:
                                    if getattr(psd_val, "dtype", None) != np.float32:
                                        psd_val = psd_val.astype(np.float32)
                                    if getattr(psd_val, "delta_f", None) != df:
                                        psd_val._delta_f = df

                            if st is not None and psd_val is not None:
                                st.psd = psd_val
                                if (
                                    psd_plan is not None
                                    and hasattr(psd_plan, "dyn_range_factor")
                                    and psd_plan.dyn_range_factor is not None
                                ):
                                    st.dyn_range_factor = psd_plan.dyn_range_factor
                                elif (
                                    hasattr(psd_val, "dyn_range_factor")
                                    and psd_val.dyn_range_factor is not None
                                ):
                                    st.dyn_range_factor = psd_val.dyn_range_factor

                            for i in range(num_cands):
                                tmpl_idx = int(candidates["template_idx"][i])
                                global_tmpl_id = (
                                    tile.template_ids[tmpl_idx]
                                    if tile is not None
                                    and hasattr(tile, "template_ids")
                                    else tmpl_idx
                                )
                                tmpl = None
                                if (
                                    tile is not None
                                    and getattr(tile, "templates", None) is not None
                                    and tmpl_idx < len(tile.templates)
                                ):
                                    tmpl = tile.templates[tmpl_idx]
                                elif (
                                    tile is not None
                                    and getattr(tile, "template_by_id", None) is not None
                                    and global_tmpl_id in tile.template_by_id
                                ):
                                    tmpl = tile.template_by_id[global_tmpl_id]
                                elif isinstance(templates, dict) and global_tmpl_id in templates:
                                    tmpl = templates[global_tmpl_id]
                                elif (
                                    bank_plan is not None
                                    and getattr(bank_plan, "template_by_id", None) is not None
                                    and global_tmpl_id in bank_plan.template_by_id
                                ):
                                    tmpl = bank_plan.template_by_id[global_tmpl_id]
                                elif templates is not None and not isinstance(templates, dict):
                                    for idx, t in enumerate(templates):
                                        if getattr(t, "id", idx) == global_tmpl_id:
                                            tmpl = t
                                            break
                                if (
                                    tmpl is None
                                    and bank_plan is not None
                                    and getattr(bank_plan, "templates", None) is not None
                                ):
                                    for idx, t in enumerate(bank_plan.templates):
                                        if getattr(t, "id", idx) == global_tmpl_id:
                                            tmpl = t
                                            break
                                if (
                                    tmpl is None
                                    and templates is not None
                                    and not isinstance(templates, dict)
                                    and tmpl_idx < len(templates)
                                    and getattr(templates[tmpl_idx], "id", tmpl_idx) == global_tmpl_id
                                ):
                                    tmpl = templates[tmpl_idx]
                                if tmpl is None:
                                    raise ValueError(
                                        f"SingleDetSGChisq requires valid template FrequencySeries for template {global_tmpl_id}, none found in BankPlan or templates"
                                    )
                                if psd_val is None:
                                    raise ValueError(
                                        "SingleDetSGChisq requires valid PSD FrequencySeries, none found in PSDPlan or stilde"
                                    )
                                if st is None:
                                    raise ValueError(
                                        "SingleDetSGChisq requires valid stilde FrequencySeries"
                                    )

                                if hasattr(tile_norms, "__getitem__"):
                                    raw_norm = tile_norms[tmpl_idx]
                                else:
                                    raw_norm = tile_norms
                                norm_val = (
                                    float(raw_norm.item())
                                    if hasattr(raw_norm, "item")
                                    else float(raw_norm)
                                )
                                snrv = np.array(
                                    [candidates["snr"][i] / norm_val],
                                    dtype=np.complex64,
                                )
                                bchisq = (
                                    candidates["chisq"][i : i + 1]
                                    if "chisq" in candidates
                                    else np.array([1.0], dtype=np.float32)
                                )
                                bchisq_dof = (
                                    candidates["chisq_dof"][i : i + 1]
                                    if "chisq_dof" in candidates
                                    else np.array([30], dtype=np.uint32)
                                )
                                idx = [int(candidates["sample_idx"][i])]

                                # Propagate any failure directly without masking
                                val = evaluator.values(
                                    st,
                                    tmpl,
                                    psd_val,
                                    snrv,
                                    norm_val,
                                    bchisq,
                                    bchisq_dof,
                                    idx,
                                )
                                res_list.append(
                                    val[0]
                                    if val is not None and len(val) > 0
                                    else 1.0
                                )
                            res = np.array(res_list, dtype=np.float32)
                elif hasattr(evaluator, "evaluate") and callable(
                    evaluator.evaluate
                ):
                    try:
                        res = evaluator.evaluate(
                            corr_tile=corr_tile,
                            candidates=candidates,
                            tile_id=tile_id,
                            tile_norms=tile_norms,
                            transform_length=transform_length,
                        )
                    except TypeError:
                        try:
                            res = evaluator.evaluate(candidates)
                        except TypeError:
                            res = evaluator.evaluate()

                if res is not None:
                    candidates["sg_chisq"] = np.asarray(res, dtype=np.float32)
                else:
                    candidates["sg_chisq"] = np.ones(num_cands, dtype=np.float32)
            else:
                candidates["sg_chisq"] = np.ones(num_cands, dtype=np.float32)

        return candidates
