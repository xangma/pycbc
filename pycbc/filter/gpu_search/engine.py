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
from .graphs import CUDAGraphManager
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
    slot_idx: int = 0
    event: Optional[Any] = None


class _TorchThreadContext:
    """Scoped context manager to configure and restore PyTorch CPU threads."""

    def __init__(self, target_threads: Optional[int]):
        self.target_threads = (
            int(target_threads) if target_threads is not None else None
        )
        self.orig_threads = None

    def __enter__(self):
        if self.target_threads is not None and torch is not None:
            self.orig_threads = torch.get_num_threads()
            if self.orig_threads != self.target_threads:
                torch.set_num_threads(self.target_threads)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.orig_threads is not None and torch is not None:
            torch.set_num_threads(self.orig_threads)


class SearchEngine:
    """
    Persistent, tiled GPU search engine supporting eager and CUDA graph filtering,
    async double-buffering, and device candidate selection.
    """

    def __init__(
        self,
        bank_plan: BankPlan,
        selection_policy: SelectionPolicy,
        veto_manager: Optional[Any] = None,
        workspace_budget: Optional[WorkspaceBudget] = None,
        candidate_capacity: int = 65536,
        device: str = "cpu",
        use_cuda_graphs: bool = False,
        num_workspaces: int = 1,
        enable_async_transfers: bool = False,
        num_threads: Optional[int] = None,
    ):
        self.bank_plan = bank_plan
        self.selection_policy = selection_policy
        self.candidate_capacity = int(candidate_capacity)
        self.device = str(device)
        self.use_cuda_graphs = bool(use_cuda_graphs)
        self.num_workspaces = max(1, int(num_workspaces))
        self.enable_async_transfers = bool(enable_async_transfers)
        self.num_threads = (
            int(num_threads) if num_threads is not None else None
        )
        self._current_slot = 0

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

        self.graph_manager = CUDAGraphManager(
            enabled=self.use_cuda_graphs, device=self.device
        )

        # Allocate device workspaces
        self._allocate_workspaces()

    def _allocate_workspaces(self):
        """Allocate bounded device buffers for correlation, IFFT, and candidate queues."""
        is_cuda = (
            torch is not None
            and self.device != "numpy"
            and self.device.startswith("cuda")
            and torch.cuda.is_available()
        )
        dev = (
            torch.device(self.device)
            if torch is not None and self.device != "numpy"
            else None
        )

        self.cout_workspaces = []
        self.out_workspaces = []
        self.stilde_bufs = []
        self.candidate_buffers = []
        self._compute_streams = []
        self._transfer_streams = []
        self._compute_events = []
        self._transfer_events = []
        self._pinned_stagings = []
        self._slot_events = [None] * self.num_workspaces

        for _ in range(self.num_workspaces):
            if torch is not None and self.device != "numpy":
                cout = torch.zeros(
                    (self.max_batch_size, self.tlen),
                    dtype=torch.complex64,
                    device=dev,
                )
                out = torch.zeros(
                    (self.max_batch_size, self.tlen),
                    dtype=torch.complex64,
                    device=dev,
                )
                stilde_buf = torch.zeros(
                    (self.flen,), dtype=torch.complex64, device=dev
                )
            else:
                cout = np.zeros(
                    (self.max_batch_size, self.tlen), dtype=np.complex64
                )
                out = np.zeros(
                    (self.max_batch_size, self.tlen), dtype=np.complex64
                )
                stilde_buf = np.zeros((self.flen,), dtype=np.complex64)

            cb = CandidateBuffer(
                capacity=self.candidate_capacity, device=self.device
            )

            self.cout_workspaces.append(cout)
            self.out_workspaces.append(out)
            self.stilde_bufs.append(stilde_buf)
            self.candidate_buffers.append(cb)

            if is_cuda:
                comp_s = torch.cuda.Stream(device=dev)
                trans_s = (
                    torch.cuda.Stream(device=dev)
                    if self.enable_async_transfers
                    else None
                )
                comp_ev = torch.cuda.Event()
                trans_ev = (
                    torch.cuda.Event() if self.enable_async_transfers else None
                )
                pinned = (
                    torch.empty(
                        (self.flen,), dtype=torch.complex64, pin_memory=True
                    )
                    if self.enable_async_transfers
                    else None
                )
            else:
                comp_s = None
                trans_s = None
                comp_ev = None
                trans_ev = None
                pinned = None

            self._compute_streams.append(comp_s)
            self._transfer_streams.append(trans_s)
            self._compute_events.append(comp_ev)
            self._transfer_events.append(trans_ev)
            self._pinned_stagings.append(pinned)

    @property
    def cout_workspace(self) -> Any:
        return self.cout_workspaces[0] if self.cout_workspaces else None

    @cout_workspace.setter
    def cout_workspace(self, val: Any):
        if self.cout_workspaces:
            self.cout_workspaces[0] = val
        else:
            self.cout_workspaces = [val]

    @property
    def out_workspace(self) -> Any:
        return self.out_workspaces[0] if self.out_workspaces else None

    @out_workspace.setter
    def out_workspace(self, val: Any):
        if self.out_workspaces:
            self.out_workspaces[0] = val
        else:
            self.out_workspaces = [val]

    @property
    def candidate_buffer(self) -> CandidateBuffer:
        return self.candidate_buffers[0] if self.candidate_buffers else None

    @candidate_buffer.setter
    def candidate_buffer(self, val: CandidateBuffer):
        if self.candidate_buffers:
            self.candidate_buffers[0] = val
        else:
            self.candidate_buffers = [val]

    @property
    def graph_stats(self) -> Dict[str, int]:
        """Diagnostics for CUDA graph captures, replays, and invalidations."""
        return {
            "capture_count": self.graph_manager.capture_count,
            "replay_count": self.graph_manager.replay_count,
            "invalidation_count": self.graph_manager.invalidation_count,
        }

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
        with _TorchThreadContext(self.num_threads):
            return self._submit_impl(
                data_block=data_block,
                psd_plan=psd_plan,
                valid_interval=valid_interval,
                block_id=block_id,
                tile_id=tile_id,
            )

    def _submit_impl(
        self,
        data_block: Any,
        psd_plan: PSDPlan,
        valid_interval: Tuple[int, int],
        block_id: int = 0,
        tile_id: Optional[int] = None,
    ) -> Ticket:
        if psd_plan.bank_version_hash != self.bank_plan.version_hash:
            raise ValueError(
                f"PSDPlan bound to bank version {psd_plan.bank_version_hash}, "
                f"but engine has bank version {self.bank_plan.version_hash}"
            )

        # Select workspace slot for double-buffering
        slot_idx = self._current_slot % self.num_workspaces
        self._current_slot += 1

        # Synchronize prior pending work in this slot if still running
        if self._slot_events[slot_idx] is not None:
            self._slot_events[slot_idx].synchronize()
            self._slot_events[slot_idx] = None

        self._ticket_counter += 1
        ticket = Ticket(
            ticket_id=self._ticket_counter,
            block_id=block_id,
            psd_version=psd_plan.psd_version,
            valid_interval=valid_interval,
            slot_idx=slot_idx,
        )

        valid_start, valid_end = valid_interval
        candidate_buffer = self.candidate_buffers[slot_idx]
        candidate_buffer.reset()

        cout_workspace = self.cout_workspaces[slot_idx]
        out_workspace = self.out_workspaces[slot_idx]
        stilde_buf = self.stilde_bufs[slot_idx]
        compute_stream = self._compute_streams[slot_idx]
        transfer_stream = self._transfer_streams[slot_idx]
        transfer_event = self._transfer_events[slot_idx]
        compute_event = self._compute_events[slot_idx]
        pinned_staging = self._pinned_stagings[slot_idx]

        # Prepare overwhitened data: s_tilde(f) / S_n(f)
        if torch is not None and self.device != "numpy":
            dev = torch.device(self.device)

            # Asynchronous pinned H2D transfer if enabled and input is host numpy/tensor
            if (
                self.enable_async_transfers
                and transfer_stream is not None
                and pinned_staging is not None
                and not (
                    isinstance(data_block, torch.Tensor)
                    and data_block.device == dev
                )
            ):
                data_np = (
                    data_block.numpy()
                    if hasattr(data_block, "numpy")
                    else np.asarray(data_block)
                )
                with torch.cuda.stream(transfer_stream):
                    pinned_staging.copy_(
                        torch.as_tensor(
                            data_np[: self.flen], dtype=torch.complex64
                        )
                    )
                    stilde_buf.copy_(pinned_staging, non_blocking=True)
                    transfer_event.record(transfer_stream)

                if compute_stream is not None:
                    compute_stream.wait_event(transfer_event)
                data_tensor = stilde_buf
            elif isinstance(data_block, torch.Tensor):
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

            # Overwhitened data into static stilde_buf
            if compute_stream is not None:
                with torch.cuda.stream(compute_stream):
                    torch.mul(
                        data_tensor[: self.flen],
                        inv_psd[: self.flen],
                        out=stilde_buf[: self.flen],
                    )
                    kmin, kmax = get_cutoff_indices(
                        self.bank_plan.geometry.f_lower,
                        self.bank_plan.geometry.f_upper,
                        self.bank_plan.geometry.delta_f,
                        self.tlen,
                    )
                    stilde_buf[:kmin].zero_()
                    if kmax < len(stilde_buf):
                        stilde_buf[kmax:].zero_()
            else:
                torch.mul(
                    data_tensor[: self.flen],
                    inv_psd[: self.flen],
                    out=stilde_buf[: self.flen],
                )
                kmin, kmax = get_cutoff_indices(
                    self.bank_plan.geometry.f_lower,
                    self.bank_plan.geometry.f_upper,
                    self.bank_plan.geometry.delta_f,
                    self.tlen,
                )
                stilde_buf[:kmin].zero_()
                if kmax < len(stilde_buf):
                    stilde_buf[kmax:].zero_()

            stilde = stilde_buf
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
            stilde_buf[: self.flen] = stilde

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

            if torch is not None and isinstance(cout_workspace, torch.Tensor):
                graph_executed = False
                if self.graph_manager.is_available():
                    graph_entry = self.graph_manager.get_or_capture(
                        tile=tile,
                        cout_workspace=cout_workspace,
                        out_workspace=out_workspace,
                        stilde_dev=stilde_buf,
                        flen=self.flen,
                        tlen=self.tlen,
                        stream=compute_stream,
                    )
                    if graph_entry is not None:
                        graph_executed = self.graph_manager.replay(
                            entry=graph_entry,
                            stilde_input=stilde_buf,
                            stream=compute_stream,
                        )

                if not graph_executed:
                    # Eager execution
                    if compute_stream is not None:
                        with torch.cuda.stream(compute_stream):
                            torch.mul(
                                torch.conj(tile.template_data),
                                stilde.unsqueeze(0),
                                out=cout_workspace[:b, : self.flen],
                            )
                            if self.flen < self.tlen:
                                cout_workspace[:b, self.flen :].zero_()
                            torch.fft.ifft(
                                cout_workspace[:b],
                                n=self.tlen,
                                dim=-1,
                                norm="forward",
                                out=out_workspace[:b],
                            )
                    else:
                        torch.mul(
                            torch.conj(tile.template_data),
                            stilde.unsqueeze(0),
                            out=cout_workspace[:b, : self.flen],
                        )
                        if self.flen < self.tlen:
                            cout_workspace[:b, self.flen :].zero_()
                        torch.fft.ifft(
                            cout_workspace[:b],
                            n=self.tlen,
                            dim=-1,
                            norm="forward",
                            out=out_workspace[:b],
                        )

                # Device candidate selection
                sel = select_tile_candidates(
                    out_workspace[:b],
                    norms,
                    sigmasqs,
                    valid_start,
                    valid_end,
                    self.selection_policy,
                    buffer=candidate_buffer,
                )
            else:
                # Numpy fallback
                cout_workspace[:b, : self.flen] = (
                    np.conj(tile.template_data) * overwhitened_np[np.newaxis, :]
                )
                if self.flen < self.tlen:
                    cout_workspace[:b, self.flen :] = 0.0

                out_workspace[:b] = (
                    np.fft.ifft(cout_workspace[:b], n=self.tlen, axis=-1)
                    * self.tlen
                )

                sel = select_tile_candidates(
                    out_workspace[:b],
                    norms,
                    sigmasqs,
                    valid_start,
                    valid_end,
                    self.selection_policy,
                    buffer=candidate_buffer,
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
                        corr_tile=cout_workspace[:b],
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

        if compute_event is not None and compute_stream is not None:
            compute_event.record(compute_stream)
            self._slot_events[slot_idx] = compute_event
            ticket.event = compute_event

        self._provisional_batches.append(ticket)
        return ticket

    def drain(self) -> List[Ticket]:
        """
        Drain committed results whose boundary and veto dependencies are resolved.
        """
        ready = []
        remaining = []
        for ticket in self._provisional_batches:
            if ticket.event is not None:
                ticket.event.synchronize()
                ticket.event = None
            if ticket.completed:
                ready.append(ticket)
            else:
                remaining.append(ticket)

        self._provisional_batches = remaining
        self._committed_batches.extend(ready)
        return ready

    def flush(self) -> List[Ticket]:
        """
        Flush all outstanding provisional batches and synchronize streams.
        """
        for ev in self._slot_events:
            if ev is not None:
                ev.synchronize()
        self._slot_events = [None] * self.num_workspaces

        for s in self._compute_streams:
            if s is not None:
                s.synchronize()

        batches = list(self._provisional_batches)
        for ticket in batches:
            if ticket.event is not None:
                ticket.event.synchronize()
                ticket.event = None
            ticket.completed = True

        self._provisional_batches.clear()
        self._committed_batches.extend(batches)
        return batches

    def close(self):
        """Release workspaces and clean up resources."""
        self.flush()
        self.graph_manager.invalidate_all()
        self._provisional_batches.clear()
        self._committed_batches.clear()
        self.cout_workspaces.clear()
        self.out_workspaces.clear()
        self.stilde_bufs.clear()
        self.candidate_buffers.clear()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
