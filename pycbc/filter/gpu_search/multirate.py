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
[EXPERIMENTAL PROTOTYPE - Milestone M6]
Multirate matched filtering for the PyCBC persistent GPU search engine.
This module is an experimental research prototype not yet qualified for
production pipeline execution.

Partitions template banks across frequency bands, filtering low-frequency
inspiral components at decimated sample rates to dramatically reduce FFT
complexity and memory bandwidth, followed by coherent full-rate candidate
refinement and veto evaluation.
"""

from dataclasses import dataclass
import logging
from typing import Any, List, Optional, Tuple
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pycbc.types import FrequencySeries
from .plans import BankGeometry, BankPlan, PSDPlan, prepare_bank
from .candidates import SelectionPolicy
from .engine import SearchEngine, Ticket
from .vetoes import VetoManager

logger = logging.getLogger("pycbc.filter.gpu_search.multirate")


@dataclass
class MultirateBand:
    """Specification of a single frequency sub-band in a multirate plan."""

    band_idx: int
    f_lower: float
    f_upper: float
    decimation_factor: int
    sample_rate: float
    delta_f: float
    filter_length: int
    transform_length: int
    geometry: BankGeometry


class MultiratePlan:
    """
    Multirate plan holding full-rate and coarse-rate bank plans and band geometries.
    """

    def __init__(
        self,
        full_bank_plan: BankPlan,
        coarse_bank_plan: BankPlan,
        band: MultirateBand,
    ):
        self.full_bank_plan = full_bank_plan
        self.coarse_bank_plan = coarse_bank_plan
        self.band = band
        self.decimation_factor = int(band.decimation_factor)


def prepare_multirate_plan(
    templates: List[Any],
    decimation_factor: int = 4,
    f_split: Optional[float] = None,
    tile_size: int = 64,
    device: str = "cpu",
) -> MultiratePlan:
    """
    Partition templates into full-rate and decimated coarse-rate bank plans.

    Parameters
    ----------
    templates : List[FrequencySeries]
        List of template frequency series.
    decimation_factor : int, optional
        Integer sub-sampling factor for the low-frequency band (default 4).
    f_split : float, optional
        Frequency boundary separating coarse band and fine band. If None,
        defaults to the Nyquist frequency of the decimated rate.
    tile_size : int, optional
        Batch size of templates per execution tile.
    device : str, optional
        Target execution device ('cpu', 'cuda', etc.).

    Returns
    -------
    MultiratePlan
        Prepared multirate plan with coarse and full bank structures.
    """
    if not templates:
        raise ValueError("templates list cannot be empty")
    if decimation_factor < 1:
        raise ValueError("decimation_factor must be >= 1")

    first_tmpl = templates[0]
    delta_f = float(first_tmpl.delta_f)
    flen_full = len(first_tmpl)
    n_full = (flen_full - 1) * 2

    # Determine coarse transform length and frequency length
    n_coarse = n_full // decimation_factor
    flen_coarse = n_coarse // 2 + 1
    fs_coarse = n_coarse * delta_f
    max_coarse_freq = (flen_coarse - 1) * delta_f

    if f_split is None or f_split > max_coarse_freq:
        f_split = max_coarse_freq

    # Slice templates into coarse band
    coarse_templates = []
    for t in templates:
        data = getattr(t, "data", t)
        if hasattr(data, "numpy"):
            data_np = data.numpy()
        elif hasattr(data, "cpu"):
            data_np = data.cpu().numpy()
        else:
            data_np = np.asarray(data)

        c_data = data_np[:flen_coarse].astype(np.complex64)
        c_series = FrequencySeries(c_data, delta_f=delta_f)
        c_series.id = getattr(t, "id", None)
        if hasattr(t, "params"):
            c_series.params = t.params
        coarse_templates.append(c_series)

    full_bank_plan = prepare_bank(
        templates,
        tile_size=tile_size,
        device=device,
    )
    coarse_bank_plan = prepare_bank(
        coarse_templates,
        tile_size=tile_size,
        device=device,
    )

    coarse_band = MultirateBand(
        band_idx=0,
        f_lower=0.0,
        f_upper=f_split,
        decimation_factor=decimation_factor,
        sample_rate=fs_coarse,
        delta_f=delta_f,
        filter_length=flen_coarse,
        transform_length=n_coarse,
        geometry=coarse_bank_plan.geometry,
    )

    return MultiratePlan(
        full_bank_plan=full_bank_plan,
        coarse_bank_plan=coarse_bank_plan,
        band=coarse_band,
    )


class MultirateSearchEngine:
    """
    Search engine employing multirate proposal and full-resolution refinement.
    """

    def __init__(
        self,
        multirate_plan: MultiratePlan,
        selection_policy: SelectionPolicy,
        veto_manager: Optional[VetoManager] = None,
        proposal_margin: float = 1.0,
        refinement_window: Optional[int] = None,
        device: str = "cpu",
        use_cuda_graphs: bool = False,
        num_workspaces: int = 1,
    ):
        self.multirate_plan = multirate_plan
        self.selection_policy = selection_policy
        self.veto_manager = veto_manager
        self.device = str(device)
        self.d = multirate_plan.decimation_factor
        self.refinement_window = (
            int(refinement_window)
            if refinement_window is not None
            else 2 * self.d
        )

        # 1. Coarse search engine (runs at decimated rate with lowered proposal threshold)
        coarse_thresh = max(0.0, selection_policy.snr_threshold - proposal_margin)
        self.coarse_policy = SelectionPolicy(
            snr_threshold=coarse_thresh,
            snr_abort_threshold=selection_policy.snr_abort_threshold,
            cluster_policy="live_peak",
            max_triggers_in_batch=selection_policy.max_triggers_in_batch,
        )
        self.coarse_engine = SearchEngine(
            bank_plan=multirate_plan.coarse_bank_plan,
            selection_policy=self.coarse_policy,
            device=self.device,
            use_cuda_graphs=use_cuda_graphs,
            num_workspaces=num_workspaces,
        )

        # 2. Full-rate search engine for refinement and vetoes
        self.full_engine = SearchEngine(
            bank_plan=multirate_plan.full_bank_plan,
            selection_policy=self.selection_policy,
            veto_manager=self.veto_manager,
            device=self.device,
            use_cuda_graphs=use_cuda_graphs,
            num_workspaces=num_workspaces,
        )

        self._ticket_counter = 0
        self._provisional_batches: List[Ticket] = []
        self._committed_batches: List[Ticket] = []

    def submit(
        self,
        data_block: Any,
        full_psd_plan: PSDPlan,
        coarse_psd_plan: PSDPlan,
        valid_interval: Tuple[int, int],
        block_id: int = 0,
    ) -> Ticket:
        """
        Submit a data block for multirate filtering.

        1. Filters coarse band at decimated rate.
        2. If candidate proposals exceed threshold, refines them at full rate.
        3. Applies veto manager to surviving refined triggers.
        """
        self._ticket_counter += 1
        ticket = Ticket(
            ticket_id=self._ticket_counter,
            block_id=block_id,
            psd_version=full_psd_plan.psd_version,
            valid_interval=valid_interval,
        )

        valid_start, valid_end = valid_interval
        coarse_start = valid_start // self.d
        coarse_end = valid_end // self.d
        flen_coarse = self.multirate_plan.band.filter_length

        # 1. Prepare coarse data slice
        if hasattr(data_block, "data"):
            arr = data_block.data
        elif hasattr(data_block, "numpy"):
            arr = data_block.numpy()
        else:
            arr = data_block

        coarse_data = arr[:flen_coarse]

        # 2. Submit coarse filtering
        coarse_ticket = self.coarse_engine.submit(
            coarse_data,
            coarse_psd_plan,
            valid_interval=(coarse_start, coarse_end),
            block_id=block_id,
        )
        if coarse_ticket.aborted:
            ticket.aborted = True
            ticket.completed = True
            self._provisional_batches.append(ticket)
            return ticket

        ready_coarse = self.coarse_engine.drain()
        if not ready_coarse or not ready_coarse[0].results:
            ticket.completed = True
            self._provisional_batches.append(ticket)
            return ticket

        # Check if any candidates proposed
        coarse_cands = ready_coarse[0].results
        total_cands = sum(len(r.get("template_id", [])) for r in coarse_cands)
        if total_cands == 0:
            ticket.completed = True
            self._provisional_batches.append(ticket)
            return ticket

        # 3. Full-rate refinement for proposed candidate tiles
        # Submit full data to full engine to obtain refined candidate values
        full_ticket = self.full_engine.submit(
            data_block,
            full_psd_plan,
            valid_interval=valid_interval,
            block_id=block_id,
        )
        if full_ticket.aborted:
            ticket.aborted = True
            ticket.completed = True
            self._provisional_batches.append(ticket)
            return ticket

        ready_full = self.full_engine.drain()
        if ready_full and ready_full[0].results:
            ticket.results = ready_full[0].results

        ticket.completed = True
        self._provisional_batches.append(ticket)
        return ticket

    def drain(self) -> List[Ticket]:
        """Drain completed multirate tickets."""
        ready = list(self._provisional_batches)
        self._provisional_batches.clear()
        self._committed_batches.extend(ready)
        return ready

    def flush(self) -> List[Ticket]:
        """Flush pending batches and synchronize streams."""
        self.coarse_engine.flush()
        self.full_engine.flush()
        return self.drain()

    def close(self):
        """Clean up engines and device memory."""
        self.flush()
        self.coarse_engine.close()
        self.full_engine.close()
        self._provisional_batches.clear()
        self._committed_batches.clear()
