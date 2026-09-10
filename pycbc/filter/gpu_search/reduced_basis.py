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
Reduced-basis matched filtering for the PyCBC persistent GPU search engine.
This module is an experimental research prototype not yet qualified for
production pipeline execution.

Approximates groups of physically similar templates using a low-rank orthonormal
basis via Singular Value Decomposition (SVD). Filters only R basis waveforms
via IFFT, then reconstructs full template responses using high-throughput
matrix multiplication (GEMM).
"""

from dataclasses import dataclass
import logging
from typing import Any, List, Optional, Tuple
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pycbc.filter.matchedfilter import get_cutoff_indices
from .plans import BankGeometry
from .candidates import SelectionPolicy, CandidateBuffer, select_tile_candidates
from .engine import Ticket
from .vetoes import VetoManager

logger = logging.getLogger("pycbc.filter.gpu_search.reduced_basis")


@dataclass
class ReducedBasisPlan:
    """Prepared low-rank basis and expansion coefficients for template subbanks."""

    rank: int
    num_templates: int
    basis_data: Any
    coefficients: Any
    singular_values: Any
    template_ids: np.ndarray
    geometry: BankGeometry
    tolerance: float
    device: str

    def to(self, device: str) -> "ReducedBasisPlan":
        """Move plan tensors to the specified device."""
        dev_str = str(device)
        if torch is not None and hasattr(self.basis_data, "to"):
            target_dev = torch.device(dev_str)
            return ReducedBasisPlan(
                rank=self.rank,
                num_templates=self.num_templates,
                basis_data=self.basis_data.to(device=target_dev),
                coefficients=self.coefficients.to(device=target_dev),
                singular_values=self.singular_values.to(device=target_dev),
                template_ids=self.template_ids,
                geometry=self.geometry,
                tolerance=self.tolerance,
                device=dev_str,
            )
        return self


class ReducedBasisPSDPlan:
    """Bound PSD properties for reduced-basis template reconstruction."""

    def __init__(
        self,
        psd_data: Any,
        tile_norms: Any,
        tile_sigmasqs: Any,
        psd_version: str = "psd-v1",
        device: str = "cpu",
    ):
        self.psd_data = psd_data
        self.tile_norms = tile_norms
        self.tile_sigmasqs = tile_sigmasqs
        self.psd_version = psd_version
        self.device = str(device)

    def to(self, device: str) -> "ReducedBasisPSDPlan":
        """Move PSD tensors to the specified device."""
        dev_str = str(device)
        if torch is not None:
            target_dev = torch.device(dev_str)
            norms = (
                self.tile_norms.to(device=target_dev)
                if hasattr(self.tile_norms, "to")
                else self.tile_norms
            )
            sigmasqs = (
                self.tile_sigmasqs.to(device=target_dev)
                if hasattr(self.tile_sigmasqs, "to")
                else self.tile_sigmasqs
            )
            psd_data = (
                self.psd_data.to(device=target_dev)
                if hasattr(self.psd_data, "to")
                else self.psd_data
            )
            return ReducedBasisPSDPlan(
                psd_data=psd_data,
                tile_norms=norms,
                tile_sigmasqs=sigmasqs,
                psd_version=self.psd_version,
                device=dev_str,
            )
        return self


def compute_reduced_basis(
    templates: List[Any],
    max_rank: Optional[int] = None,
    tolerance: float = 1e-3,
    f_lower: Optional[float] = None,
    f_upper: Optional[float] = None,
    device: str = "cpu",
) -> ReducedBasisPlan:
    """
    Compute an orthonormal reduced basis for a group of templates via SVD.

    Parameters
    ----------
    templates : List[FrequencySeries]
        Templates to compress into a reduced basis.
    max_rank : int, optional
        Maximum number of basis vectors to retain.
    tolerance : float, optional
        Target fractional energy truncation tolerance (default 1e-3).
    f_lower : float, optional
        Low-frequency cutoff (Hz).
    f_upper : float, optional
        High-frequency cutoff (Hz).
    device : str, optional
        Target device ('cpu', 'cuda', etc.).

    Returns
    -------
    ReducedBasisPlan
        Reduced basis with orthonormal filters and expansion coefficients.
    """
    if not templates:
        raise ValueError("templates list cannot be empty")

    num_templates = len(templates)
    first = templates[0]
    delta_f = float(first.delta_f)
    flen = len(first)
    tlen = (flen - 1) * 2
    fs = tlen * delta_f
    delta_t = 1.0 / fs

    # Collect template frequency series into matrix H (M, flen)
    h_rows = []
    t_ids = []
    for i, t in enumerate(templates):
        arr = getattr(t, "data", t)
        if hasattr(arr, "numpy"):
            data_np = arr.numpy()
        elif hasattr(arr, "cpu"):
            data_np = arr.cpu().numpy()
        else:
            data_np = np.asarray(arr)
        h_rows.append(data_np.astype(np.complex64))
        t_ids.append(getattr(t, "id", i))

    h_matrix_np = np.stack(h_rows, axis=0)
    template_ids = np.array(t_ids, dtype=np.int64)

    if torch is not None and str(device) != "numpy":
        dev = torch.device(device)
        h_tensor = torch.from_numpy(h_matrix_np).to(device=dev)
        u, s, vh = torch.linalg.svd(h_tensor, full_matrices=False)

        total_energy = torch.sum(s**2)
        cum_energy = torch.cumsum(s**2, dim=0)
        energy_frac = cum_energy / total_energy

        # Determine rank R satisfying tolerance
        cutoff_mask = (1.0 - energy_frac) <= (tolerance**2)
        valid_indices = torch.nonzero(cutoff_mask)
        if len(valid_indices) > 0:
            target_r = valid_indices[0].item() + 1
        else:
            target_r = num_templates

        if max_rank is not None:
            target_r = min(target_r, max_rank)
        target_r = max(1, min(target_r, num_templates))

        basis_data = vh[:target_r, :]
        coeffs = u[:, :target_r] * s[:target_r].unsqueeze(0)
        sing_vals = s
    else:
        u, s, vh = np.linalg.svd(h_matrix_np, full_matrices=False)
        total_energy = np.sum(s**2)
        cum_energy = np.cumsum(s**2)
        energy_frac = cum_energy / total_energy

        cutoff_mask = (1.0 - energy_frac) <= (tolerance**2)
        valid_indices = np.nonzero(cutoff_mask)[0]
        if len(valid_indices) > 0:
            target_r = valid_indices[0] + 1
        else:
            target_r = num_templates

        if max_rank is not None:
            target_r = min(target_r, max_rank)
        target_r = max(1, min(target_r, num_templates))

        basis_data = vh[:target_r, :]
        coeffs = u[:, :target_r] * s[:target_r][np.newaxis, :]
        sing_vals = s

    geom = BankGeometry(
        delta_f=delta_f,
        filter_length=flen,
        transform_length=tlen,
        sample_rate=fs,
        delta_t=delta_t,
        f_lower=f_lower if f_lower is not None else getattr(first, "f_lower", None),
        f_upper=f_upper if f_upper is not None else getattr(first, "f_upper", None),
    )

    return ReducedBasisPlan(
        rank=target_r,
        num_templates=num_templates,
        basis_data=basis_data,
        coefficients=coeffs,
        singular_values=sing_vals,
        template_ids=template_ids,
        geometry=geom,
        tolerance=float(tolerance),
        device=str(device),
    )


def bind_reduced_basis_psd(
    plan: ReducedBasisPlan,
    psd: Any,
    psd_version: str = "psd-v1",
    device: Optional[str] = None,
) -> ReducedBasisPSDPlan:
    """
    Bind a PSD to a ReducedBasisPlan, precomputing original template sigmasqs and norms.
    """
    if device is None:
        device = plan.device

    if hasattr(psd, "numpy"):
        psd_np = psd.numpy()
    elif hasattr(psd, "cpu"):
        psd_np = psd.cpu().numpy()
    else:
        psd_np = np.asarray(psd)

    psd_np = psd_np.astype(np.float32)
    flen = plan.geometry.filter_length
    delta_f = plan.geometry.delta_f

    kmin, kmax = get_cutoff_indices(
        plan.geometry.f_lower,
        plan.geometry.f_upper,
        delta_f,
        plan.geometry.transform_length,
    )

    # Reconstruct original templates in memory to get exact sigmasqs under PSD
    if torch is not None and str(device) != "numpy":
        dev = torch.device(device)
        psd_tensor = torch.as_tensor(psd_np[:flen], device=dev, dtype=torch.float32)
        inv_psd = torch.where(
            psd_tensor > 0, (4.0 * delta_f) / psd_tensor, torch.zeros_like(psd_tensor)
        )
        inv_psd[:kmin].zero_()
        if kmax < len(inv_psd):
            inv_psd[kmax:].zero_()

        # Reconstruct templates: H = A @ Vh
        h_all = torch.matmul(plan.coefficients, plan.basis_data)
        pwr = torch.abs(h_all) ** 2
        sigmasqs = torch.sum(pwr * inv_psd.unsqueeze(0), dim=-1)
        norms = torch.where(
            sigmasqs > 0,
            (4.0 * delta_f) / torch.sqrt(torch.clamp(sigmasqs, min=1e-20)),
            torch.zeros_like(sigmasqs),
        )
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_psd = np.where(
                psd_np[:flen] > 0, (4.0 * delta_f) / psd_np[:flen], 0.0
            ).astype(np.float32)
        inv_psd[:kmin] = 0.0
        if kmax < len(inv_psd):
            inv_psd[kmax:] = 0.0

        h_all = np.matmul(plan.coefficients, plan.basis_data)
        pwr = np.abs(h_all) ** 2
        sigmasqs = np.sum(pwr * inv_psd[np.newaxis, :], axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            norms = np.where(
                sigmasqs > 0,
                (4.0 * delta_f) / np.sqrt(np.maximum(sigmasqs, 1e-20)),
                0.0,
            )

    return ReducedBasisPSDPlan(
        psd_data=psd_np[:flen],
        tile_norms=norms,
        tile_sigmasqs=sigmasqs,
        psd_version=psd_version,
        device=str(device),
    )


class ReducedBasisSearchEngine:
    """
    Search engine that filters low-rank basis waveforms and reconstructs
    candidate template responses via GEMM matrix multiplication.
    """

    def __init__(
        self,
        basis_plan: ReducedBasisPlan,
        selection_policy: SelectionPolicy,
        veto_manager: Optional[Any] = None,
        candidate_capacity: int = 65536,
        device: str = "cpu",
    ):
        self.basis_plan = basis_plan
        self.selection_policy = selection_policy
        if veto_manager is not None and hasattr(veto_manager, "tile_bin_edges"):
            self.veto_manager = VetoManager(power_chisq_plan=veto_manager)
        else:
            self.veto_manager = veto_manager
        self.candidate_capacity = int(candidate_capacity)
        self.device = str(device)

        self.r = basis_plan.rank
        self.m = basis_plan.num_templates
        self.flen = basis_plan.geometry.filter_length
        self.tlen = basis_plan.geometry.transform_length

        self._allocate_workspaces()
        self._ticket_counter = 0

    @property
    def cout_workspace(self):
        return self.basis_cout

    @property
    def out_workspace(self):
        return self.basis_out

    def _allocate_workspaces(self):
        """Allocate bounded basis and reconstructed correlation workspaces."""
        if torch is not None and self.device != "numpy":
            dev = torch.device(self.device)
            # Basis workspaces: shape (R, tlen)
            self.basis_cout = torch.zeros((self.r, self.tlen), dtype=torch.complex64, device=dev)
            self.basis_out = torch.zeros((self.r, self.tlen), dtype=torch.complex64, device=dev)
            # Reconstructed workspace: shape (M, tlen)
            self.q_reconstructed = torch.zeros((self.m, self.tlen), dtype=torch.complex64, device=dev)
        else:
            self.basis_cout = np.zeros((self.r, self.tlen), dtype=np.complex64)
            self.basis_out = np.zeros((self.r, self.tlen), dtype=np.complex64)
            self.q_reconstructed = np.zeros((self.m, self.tlen), dtype=np.complex64)

        self.candidate_buffer = CandidateBuffer(
            capacity=self.candidate_capacity, device=self.device
        )

    def submit(
        self,
        data_block: Any,
        psd_plan: ReducedBasisPSDPlan,
        valid_interval: Tuple[int, int],
        block_id: int = 0,
    ) -> Ticket:
        """
        Submit a data block for reduced-basis filtering and reconstruction.
        """
        self._ticket_counter += 1
        ticket = Ticket(
            ticket_id=self._ticket_counter,
            block_id=block_id,
            psd_version=psd_plan.psd_version,
            valid_interval=valid_interval,
        )

        valid_start, valid_end = valid_interval
        self.candidate_buffer.reset()

        # Prepare overwhitened data
        if torch is not None and self.device != "numpy":
            dev = torch.device(self.device)
            if isinstance(data_block, torch.Tensor):
                data_tensor = data_block.to(device=dev, dtype=torch.complex64)
            elif hasattr(data_block, "data") and isinstance(data_block.data, torch.Tensor):
                data_tensor = data_block.data.to(device=dev, dtype=torch.complex64)
            else:
                data_np = data_block.numpy() if hasattr(data_block, "numpy") else np.asarray(data_block)
                data_tensor = torch.as_tensor(data_np[:self.flen], device=dev, dtype=torch.complex64)

            psd_tensor = torch.as_tensor(psd_plan.psd_data[:self.flen], device=dev, dtype=torch.float32)
            inv_psd = torch.where(psd_tensor > 0, 1.0 / psd_tensor, torch.zeros_like(psd_tensor))
            stilde = data_tensor[:self.flen] * inv_psd[:self.flen]

            kmin, kmax = get_cutoff_indices(
                self.basis_plan.geometry.f_lower,
                self.basis_plan.geometry.f_upper,
                self.basis_plan.geometry.delta_f,
                self.tlen,
            )
            stilde[:kmin].zero_()
            if kmax < len(stilde):
                stilde[kmax:].zero_()

            # 1. Filter the R basis waveforms
            torch.mul(
                torch.conj(self.basis_plan.basis_data),
                stilde.unsqueeze(0),
                out=self.basis_cout[:, :self.flen],
            )
            if self.flen < self.tlen:
                self.basis_cout[:, self.flen:].zero_()

            torch.fft.ifft(
                self.basis_cout,
                n=self.tlen,
                dim=-1,
                norm="forward",
                out=self.basis_out,
            )

            # 2. Reconstruct M template responses via GEMM: Q = A.conj() @ y
            torch.matmul(
                torch.conj(self.basis_plan.coefficients),
                self.basis_out,
                out=self.q_reconstructed,
            )

            # 3. Candidate selection on reconstructed responses
            sel = select_tile_candidates(
                self.q_reconstructed,
                psd_plan.tile_norms,
                psd_plan.tile_sigmasqs,
                valid_start,
                valid_end,
                self.selection_policy,
                buffer=self.candidate_buffer,
            )
        else:
            data_np = data_block.numpy() if hasattr(data_block, "numpy") else np.asarray(data_block)
            psd_np = psd_plan.psd_data
            with np.errstate(divide="ignore", invalid="ignore"):
                stilde_np = np.where(psd_np > 0, data_np[:self.flen] / psd_np[:self.flen], 0.0).astype(np.complex64)

            kmin, kmax = get_cutoff_indices(
                self.basis_plan.geometry.f_lower,
                self.basis_plan.geometry.f_upper,
                self.basis_plan.geometry.delta_f,
                self.tlen,
            )
            stilde_np[:kmin] = 0.0
            if kmax < len(stilde_np):
                stilde_np[kmax:] = 0.0

            self.basis_cout[:, :self.flen] = (
                np.conj(self.basis_plan.basis_data) * stilde_np[np.newaxis, :]
            )
            if self.flen < self.tlen:
                self.basis_cout[:, self.flen:] = 0.0

            self.basis_out = np.fft.ifft(self.basis_cout, n=self.tlen, axis=-1) * self.tlen

            # GEMM
            self.q_reconstructed = np.matmul(np.conj(self.basis_plan.coefficients), self.basis_out)

            sel = select_tile_candidates(
                self.q_reconstructed,
                psd_plan.tile_norms,
                psd_plan.tile_sigmasqs,
                valid_start,
                valid_end,
                self.selection_policy,
                buffer=self.candidate_buffer,
            )

        if sel.get("aborted", False):
            ticket.aborted = True
            ticket.completed = True
            return ticket

        if sel.get("overflow", False):
            ticket.overflow = True
            ticket.completed = True
            return ticket

        cands = sel.get("candidates", {})
        if len(cands.get("template_idx", [])) > 0:
            if self.veto_manager is not None:
                if torch is not None and self.device != "numpy":
                    corr_freq = torch.matmul(
                        torch.conj(self.basis_plan.coefficients),
                        self.basis_cout[:, : self.flen],
                    )
                    corr_tile = torch.zeros(
                        (self.m, self.tlen),
                        dtype=torch.complex64,
                        device=corr_freq.device,
                    )
                    corr_tile[:, : self.flen] = corr_freq
                else:
                    corr_freq = np.matmul(
                        np.conj(self.basis_plan.coefficients),
                        self.basis_cout[:, : self.flen],
                    )
                    corr_tile = np.zeros(
                        (self.m, self.tlen), dtype=np.complex64
                    )
                    corr_tile[:, : self.flen] = corr_freq

                cands = self.veto_manager.evaluate(
                    corr_tile=corr_tile,
                    candidates=cands,
                    tile_id=0,
                    tile_norms=psd_plan.tile_norms,
                    transform_length=self.tlen,
                )

            global_tmplt_ids = self.basis_plan.template_ids[cands["template_idx"]]
            cands["template_id"] = global_tmplt_ids
            ticket.results = [cands]

        ticket.completed = True
        return ticket

    def drain(self) -> List[Ticket]:
        """Drain completed ticket."""
        return []

    def close(self):
        """Clean up buffers."""
        if hasattr(self, "basis_cout"):
            del self.basis_cout
        if hasattr(self, "basis_out"):
            del self.basis_out
        if hasattr(self, "q_reconstructed"):
            del self.q_reconstructed
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
