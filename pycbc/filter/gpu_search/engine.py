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
Persistent GPU search engine with submit/drain/flush lifecycle and workspace ownership.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pycbc.filter.matchedfilter import get_cutoff_indices
from .candidates import CandidateBuffer, SelectionPolicy, select_tile_candidates
from .plans import BankPlan, PSDPlan, WorkspaceBudget

logger = logging.getLogger("pycbc.filter.gpu_search.engine")


@dataclass
class Ticket:
    """Submission receipt tracking status and committed outputs."""

    ticket_id: int
    block_id: int
    psd_version: str
    valid_interval: Tuple[int, int]
    completed: bool = False
    aborted: bool = False
    overflow: bool = False
    results: List[Dict[str, Any]] = field(default_factory=list)


class SearchEngine:
    """
    Persistent, tiled GPU search engine.
    """

    def __init__(
        self,
        bank_plan: BankPlan,
        selection_policy: SelectionPolicy,
        veto_manager: Optional[Any] = None,
        workspace_budget: Optional[WorkspaceBudget] = None,
        candidate_capacity: int = 65536,
        device: str = "cpu",
    ):
        self.bank_plan = bank_plan
        self.selection_policy = selection_policy
        self.candidate_capacity = int(candidate_capacity)
        self.device = device

        if veto_manager is not None and hasattr(veto_manager, "tile_bin_edges"):
            from .vetoes import VetoManager
            self.veto_manager = VetoManager(power_chisq_plan=veto_manager)
        else:
            self.veto_manager = veto_manager

        self.workspace_budget = workspace_budget or WorkspaceBudget()
        self._ticket_counter = 0
        self._provisional_batches: List[Ticket] = []
        self._committed_batches: List[Ticket] = []

        # Determine max batch size across all tiles
        self.max_batch_size = max(t.batch_size for t in bank_plan.tiles)
        self.flen = bank_plan.geometry.filter_length
        self.tlen = bank_plan.geometry.transform_length

        # Allocate device workspaces
        self._allocate_workspaces()

    def _allocate_workspaces(self):
        """Allocate bounded device buffers for correlation, IFFT, and candidate queues."""
        if torch is not None and self.device != "numpy":
            dev = torch.device(self.device)
            # cout_workspace: (max_B, N) complex64 initialized to zero
            self.cout_workspace = torch.zeros(
                (self.max_batch_size, self.tlen), dtype=torch.complex64, device=dev
            )
            # out_workspace: (max_B, N) complex64
            self.out_workspace = torch.zeros(
                (self.max_batch_size, self.tlen), dtype=torch.complex64, device=dev
            )
        else:
            self.cout_workspace = np.zeros(
                (self.max_batch_size, self.tlen), dtype=np.complex64
            )
            self.out_workspace = np.zeros(
                (self.max_batch_size, self.tlen), dtype=np.complex64
            )

        self.candidate_buffer = CandidateBuffer(
            capacity=self.candidate_capacity, device=self.device
        )

    def submit(
        self,
        data_block: Any,
        psd_plan: PSDPlan,
        valid_interval: Tuple[int, int],
        block_id: int = 0,
        tile_id: Optional[int] = None,
    ) -> Ticket:
        """
        Submit a data block for tiled filtering against the bank.
        """
        if psd_plan.bank_version_hash != self.bank_plan.version_hash:
            raise ValueError(
                f"PSDPlan bound to bank version {psd_plan.bank_version_hash}, "
                f"but engine has bank version {self.bank_plan.version_hash}"
            )

        self._ticket_counter += 1
        ticket = Ticket(
            ticket_id=self._ticket_counter,
            block_id=block_id,
            psd_version=psd_plan.psd_version,
            valid_interval=valid_interval,
        )

        valid_start, valid_end = valid_interval
        self.candidate_buffer.reset()

        # Prepare overwhitened data: s_tilde(f) / S_n(f)
        if torch is not None and self.device != "numpy":
            dev = torch.device(self.device)
            if isinstance(data_block, torch.Tensor):
                data_tensor = data_block.to(device=dev, dtype=torch.complex64)
            elif hasattr(data_block, "data") and isinstance(
                data_block.data, torch.Tensor
            ):
                data_tensor = data_block.data.to(
                    device=dev, dtype=torch.complex64
                )
            else:
                data_np = (
                    data_block.numpy()
                    if hasattr(data_block, "numpy")
                    else np.asarray(data_block)
                )
                data_tensor = torch.as_tensor(
                    data_np[: self.flen], device=dev, dtype=torch.complex64
                )

            if isinstance(psd_plan.psd_data, torch.Tensor):
                psd_tensor = psd_plan.psd_data.to(
                    device=dev, dtype=torch.float32
                )
            else:
                psd_tensor = torch.as_tensor(
                    psd_plan.psd_data[: self.flen],
                    device=dev,
                    dtype=torch.float32,
                )

            inv_psd = torch.where(
                psd_tensor > 0,
                1.0 / psd_tensor,
                torch.zeros_like(psd_tensor),
            )
            stilde = data_tensor[: self.flen] * inv_psd[: self.flen]

            kmin, kmax = get_cutoff_indices(
                self.bank_plan.geometry.f_lower,
                self.bank_plan.geometry.f_upper,
                self.bank_plan.geometry.delta_f,
                self.tlen,
            )
            stilde[:kmin].zero_()
            if kmax < len(stilde):
                stilde[kmax:].zero_()
        else:
            data_np = (
                data_block.numpy()
                if hasattr(data_block, "numpy")
                else np.asarray(data_block)
            )
            psd_np = (
                psd_plan.psd_data.numpy()
                if hasattr(psd_plan.psd_data, "numpy")
                else np.asarray(psd_plan.psd_data)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                overwhitened_np = np.where(
                    psd_np > 0, data_np[: self.flen] / psd_np[: self.flen], 0.0
                ).astype(np.complex64)

            kmin, kmax = get_cutoff_indices(
                self.bank_plan.geometry.f_lower,
                self.bank_plan.geometry.f_upper,
                self.bank_plan.geometry.delta_f,
                self.tlen,
            )
            overwhitened_np[:kmin] = 0.0
            if kmax < len(overwhitened_np):
                overwhitened_np[kmax:] = 0.0
            stilde = overwhitened_np

        tiles_to_process = (
            [self.bank_plan.tiles[tile_id]]
            if tile_id is not None
            else self.bank_plan.tiles
        )
        tile_results = []
        for tile in tiles_to_process:
            b = tile.batch_size
            norms = psd_plan.tile_norms[tile.tile_id]
            sigmasqs = psd_plan.tile_sigmasqs[tile.tile_id]

            if torch is not None and isinstance(self.cout_workspace, torch.Tensor):
                # 1. Batched correlation: conj(template) * overwhitened_data
                torch.mul(
                    torch.conj(tile.template_data),
                    stilde.unsqueeze(0),
                    out=self.cout_workspace[:b, : self.flen],
                )
                if self.flen < self.tlen:
                    self.cout_workspace[:b, self.flen :].zero_()

                # 2. Batched inverse FFT with unnormalized convention
                torch.fft.ifft(
                    self.cout_workspace[:b],
                    n=self.tlen,
                    dim=-1,
                    norm="forward",
                    out=self.out_workspace[:b],
                )

                # 3. Device candidate selection
                sel = select_tile_candidates(
                    self.out_workspace[:b],
                    norms,
                    sigmasqs,
                    valid_start,
                    valid_end,
                    self.selection_policy,
                    buffer=self.candidate_buffer,
                )
            else:
                # Numpy fallback
                self.cout_workspace[:b, : self.flen] = (
                    np.conj(tile.template_data) * overwhitened_np[np.newaxis, :]
                )
                if self.flen < self.tlen:
                    self.cout_workspace[:b, self.flen :] = 0.0

                self.out_workspace[:b] = (
                    np.fft.ifft(self.cout_workspace[:b], n=self.tlen, axis=-1)
                    * self.tlen
                )

                sel = select_tile_candidates(
                    self.out_workspace[:b],
                    norms,
                    sigmasqs,
                    valid_start,
                    valid_end,
                    self.selection_policy,
                    buffer=self.candidate_buffer,
                )

            if sel.get("aborted", False):
                ticket.aborted = True
                ticket.completed = True
                self._provisional_batches.append(ticket)
                return ticket

            if sel.get("overflow", False):
                ticket.overflow = True
                ticket.completed = True
                self._provisional_batches.append(ticket)
                return ticket

            cands = sel.get("candidates", {})
            if len(cands.get("template_idx", [])) > 0:
                if self.veto_manager is not None:
                    cands = self.veto_manager.evaluate(
                        corr_tile=self.cout_workspace[:b],
                        candidates=cands,
                        tile_id=tile.tile_id,
                        tile_norms=norms,
                        transform_length=self.tlen,
                    )
                # Map tile-local template index to global template ID
                global_tmplt_ids = np.array(
                    [tile.template_ids[idx] for idx in cands["template_idx"]],
                    dtype=np.int64,
                )
                cands["template_id"] = global_tmplt_ids
                tile_results.append(cands)

        ticket.results = tile_results
        ticket.completed = True
        self._provisional_batches.append(ticket)
        return ticket

    def drain(self) -> List[Ticket]:
        """
        Drain committed results whose boundary and veto dependencies are resolved.
        """
        ready = []
        remaining = []
        for ticket in self._provisional_batches:
            if ticket.completed:
                ready.append(ticket)
            else:
                remaining.append(ticket)

        self._provisional_batches = remaining
        self._committed_batches.extend(ready)
        return ready

    def flush(self) -> List[Ticket]:
        """
        Flush all outstanding provisional batches.
        """
        batches = list(self._provisional_batches)
        self._provisional_batches.clear()
        self._committed_batches.extend(batches)
        return batches

    def close(self):
        """Release workspaces and clean up resources."""
        self._provisional_batches.clear()
        self._committed_batches.clear()
        if hasattr(self, "cout_workspace"):
            del self.cout_workspace
        if hasattr(self, "out_workspace"):
            del self.out_workspace
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
