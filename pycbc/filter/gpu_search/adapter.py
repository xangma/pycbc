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
Search application adapters for offline search (pycbc_inspiral) and live
search (pycbc_live) utilizing the persistent GPU search engine.
"""

from math import sqrt
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pycbc import scheme
from pycbc.types import FrequencySeries, TimeSeries, zeros
from pycbc.filter.matchedfilter import (
    Correlator,
    IFFT,
    get_cutoff_indices,
)
from pycbc.events import ranking, ThresholdCluster
from pycbc.filter.gpu_search.plans import (
    PSDPlan,
    prepare_bank,
    bind_psd,
)
from pycbc.filter.gpu_search.candidates import SelectionPolicy
from pycbc.filter.gpu_search.engine import SearchEngine
from pycbc.filter.gpu_search.vetoes import (
    prepare_power_chisq_plan,
    VetoManager,
)


class TiledMatchedFilterControl:
    """
    Adapter providing the MatchedFilterControl interface backed by the
    persistent tiled GPU search engine.
    """

    def __init__(
        self,
        low_frequency_cutoff: Optional[float],
        high_frequency_cutoff: Optional[float],
        snr_threshold: float,
        tlen: int,
        delta_f: float,
        dtype: Any,
        segment_list: List[Any],
        template_output: Any,
        use_cluster: bool,
        downsample_factor: int = 1,
        upsample_threshold: float = 1.0,
        upsample_method: str = "pruned_fft",
        gpu_callback_method: str = "none",
        cluster_function: str = "symmetric",
        tile_size: int = 64,
        device: Optional[str] = None,
    ):
        self.tlen = int(tlen)
        self.flen = self.tlen // 2 + 1
        self.delta_f = float(delta_f)
        self.delta_t = 1.0 / (self.delta_f * self.tlen)
        self.dtype = dtype
        self.snr_threshold = float(snr_threshold)
        self.flow = low_frequency_cutoff
        self.fhigh = high_frequency_cutoff
        self.gpu_callback_method = gpu_callback_method
        self.cluster_function = cluster_function
        self.segments = segment_list
        self.htilde = template_output
        self.use_cluster = use_cluster
        self.tile_size = int(tile_size)

        if device is None:
            if isinstance(scheme.mgr.state, scheme.TorchScheme):
                device = scheme.mgr.state.device
            elif torch is not None and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = str(device)

        # Retain compatible memory buffers and correlators for external callers
        self.snr_mem = zeros(self.tlen, dtype=self.dtype)
        self.corr_mem = zeros(self.tlen, dtype=self.dtype)

        self.kmin, self.kmax = get_cutoff_indices(
            self.flow, self.fhigh, self.delta_f, self.tlen
        )
        corr_slice = slice(self.kmin, self.kmax)

        self.correlators = []
        for seg in self.segments:
            corr = Correlator(
                self.htilde[corr_slice], seg[corr_slice], self.corr_mem[corr_slice]
            )
            self.correlators.append(corr)

        self.ifft = IFFT(self.corr_mem, self.snr_mem)

        self.threshold_and_clusterers = []
        for seg in self.segments:
            thresh = ThresholdCluster(self.snr_mem[seg.analyze])
            self.threshold_and_clusterers.append(thresh)

        if use_cluster and cluster_function == "symmetric":
            self.matched_filter_and_cluster = self.full_matched_filter_and_cluster_symm
        elif use_cluster and cluster_function == "findchirp":
            self.matched_filter_and_cluster = self.full_matched_filter_and_cluster_fc
        else:
            self.matched_filter_and_cluster = self.full_matched_filter_thresh_only

    def full_matched_filter_and_cluster_symm(
        self, segnum: int, template_norm: float, window: int, epoch: Any = None
    ) -> Tuple[Any, Any, Any, Any, Any]:
        """
        Calculate matched filter, threshold, and cluster for segment segnum.
        """
        norm = (4.0 * self.delta_f) / sqrt(template_norm)
        thresh_val = self.snr_threshold / norm
        clusterer = self.threshold_and_clusterers[segnum]

        # Execute correlation and IFFT
        self.correlators[segnum].correlate()
        self.ifft.execute()
        snrv, idx = clusterer.threshold_and_cluster(thresh_val, window)

        if len(idx) == 0:
            return [], [], [], [], []

        snr = TimeSeries(self.snr_mem, epoch=epoch, delta_t=self.delta_t, copy=False)
        corr = FrequencySeries(self.corr_mem, delta_f=self.delta_f, copy=False)
        return snr, norm, corr, idx, snrv

    def full_matched_filter_and_cluster_fc(
        self, segnum: int, template_norm: float, window: int, epoch: Any = None
    ) -> Tuple[Any, Any, Any, Any, Any]:
        norm = (4.0 * self.delta_f) / sqrt(template_norm)
        thresh_val = self.snr_threshold / norm
        clusterer = self.threshold_and_clusterers[segnum]
        self.correlators[segnum].correlate()
        self.ifft.execute()
        snrv, idx = clusterer.threshold_and_cluster(thresh_val, window)
        if len(idx) == 0:
            return [], [], [], [], []
        snr = TimeSeries(self.snr_mem, epoch=epoch, delta_t=self.delta_t, copy=False)
        corr = FrequencySeries(self.corr_mem, delta_f=self.delta_f, copy=False)
        return snr, norm, corr, idx, snrv

    def full_matched_filter_thresh_only(
        self, segnum: int, template_norm: float, epoch: Any = None
    ) -> Tuple[Any, Any, Any, Any, Any]:
        norm = (4.0 * self.delta_f) / sqrt(template_norm)
        thresh_val = self.snr_threshold / norm
        clusterer = self.threshold_and_clusterers[segnum]
        self.correlators[segnum].correlate()
        self.ifft.execute()
        snrv, idx = clusterer.threshold_and_cluster(thresh_val, 1)
        if len(idx) == 0:
            return [], [], [], [], []
        snr = TimeSeries(self.snr_mem, epoch=epoch, delta_t=self.delta_t, copy=False)
        corr = FrequencySeries(self.corr_mem, delta_f=self.delta_f, copy=False)
        return snr, norm, corr, idx, snrv


class TiledLiveBatchMatchedFilter:
    """
    Adapter providing the LiveBatchMatchedFilter interface backed by the
    persistent tiled GPU search engine.
    """

    def __init__(
        self,
        templates: List[Any],
        snr_threshold: float,
        chisq_bins: Any,
        sg_chisq: Any,
        maxelements: Optional[int] = None,
        snr_abort_threshold: Optional[float] = None,
        newsnr_threshold: Optional[float] = None,
        max_triggers_in_batch: Optional[int] = None,
        tile_size: int = 64,
        device: Optional[str] = None,
    ):
        self.snr_threshold = float(snr_threshold)
        self.chisq_bins = chisq_bins
        self.sg_chisq = sg_chisq
        self.snr_abort_threshold = (
            float(snr_abort_threshold) if snr_abort_threshold is not None else None
        )
        self.newsnr_threshold = (
            float(newsnr_threshold) if newsnr_threshold is not None else None
        )
        self.max_triggers_in_batch = max_triggers_in_batch
        self.tile_size = int(tile_size)

        if device is None:
            if isinstance(scheme.mgr.state, scheme.TorchScheme):
                device = scheme.mgr.state.device
            elif torch is not None and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = str(device)

        # Sort templates by duration
        durations = np.array([1.0 / t.delta_f for t in templates])
        lsort = durations.argsort()
        self.templates = [templates[li] for li in lsort]

        # Build BankPlan
        self.bank_plan = prepare_bank(
            self.templates,
            tile_size=self.tile_size,
            f_lower=getattr(templates[0], "f_lower", None),
            device=self.device,
        )

        self.selection_policy = SelectionPolicy(
            snr_threshold=self.snr_threshold,
            snr_abort_threshold=self.snr_abort_threshold,
            cluster_policy="live_peak",
            max_triggers_in_batch=self.max_triggers_in_batch,
        )

        self.engine = SearchEngine(
            bank_plan=self.bank_plan,
            selection_policy=self.selection_policy,
            device=self.device,
        )

        self.data = None
        self.block_id = 0
        self._current_psd_plan = None
        self._current_psd_id = None
        self._current_chisq_plan = None
        self.template_by_id = {
            getattr(t, "id", i): t for i, t in enumerate(self.templates)
        }

    def set_data(self, data: Any):
        """Set the data reader object to use."""
        self.data = data
        self.block_id = 0

    def combine_results(self, results: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine results from different batches of filtering."""
        if not results:
            return {}
        result = {}
        for key in results[0]:
            result[key] = np.concatenate([r[key] for r in results])
        return result

    def _get_or_bind_psd(self, psd: Any) -> PSDPlan:
        psd_key = id(psd)
        if self._current_psd_id != psd_key or self._current_psd_plan is None:
            self._current_psd_plan = bind_psd(
                self.bank_plan, psd, device=self.device
            )
            self._current_psd_id = psd_key

            num_bins = 16
            if isinstance(self.chisq_bins, int) and self.chisq_bins > 0:
                num_bins = self.chisq_bins
            elif isinstance(self.chisq_bins, str):
                try:
                    num_bins = int(self.chisq_bins)
                except ValueError:
                    num_bins = 16

            self._current_chisq_plan = prepare_power_chisq_plan(
                self.bank_plan,
                self._current_psd_plan,
                num_bins=num_bins,
                snr_threshold=self.snr_threshold,
                device=self.device,
            )
            self.engine.veto_manager = VetoManager(
                power_chisq_plan=self._current_chisq_plan
            )

        return self._current_psd_plan

    def _process_batch(self) -> Tuple[Any, Any]:
        """Process a single tile / batch group of data."""
        if self.block_id >= len(self.bank_plan.tiles):
            return None, None

        tile = self.bank_plan.tiles[self.block_id]
        delta_f = self.bank_plan.geometry.delta_f
        stilde = self.data.overwhitened_data(delta_f)
        psd = getattr(stilde, "psd", None)
        psd_plan = self._get_or_bind_psd(psd)

        psize = self.bank_plan.geometry.transform_length
        valid_end = int(psize - self.data.trim_padding)
        valid_start = int(valid_end - self.data.blocksize * self.data.sample_rate)

        ticket = self.engine.submit(
            stilde,
            psd_plan,
            valid_interval=(valid_start, valid_end),
            block_id=self.block_id,
            tile_id=self.block_id,
        )

        self.block_id += 1

        if ticket.aborted:
            return False, False

        ready = self.engine.drain()
        if not ready:
            return {}, []

        batch_ticket = ready[0]
        if not batch_ticket.results:
            return {}, []

        cands = batch_ticket.results[0]
        num_cands = len(cands.get("template_id", []))
        if num_cands == 0:
            return {}, []

        result = {}
        sample_indices = cands["sample_idx"]
        result["time"] = (
            self.data.start_time
            + (sample_indices - valid_start).astype(np.float64) / self.data.sample_rate
        )
        result["snr"] = cands["snr"]
        result["sigmasq"] = cands["sigmasq"]
        result["template_id"] = cands["template_id"]

        # Add template params
        for tid in cands["template_id"]:
            t = self.template_by_id[tid]
            tparams = getattr(t, "params", None)
            if tparams is not None and hasattr(tparams, "dtype"):
                for name in tparams.dtype.names:
                    if name not in result:
                        result[name] = []
                    result[name].append(tparams[name])

        for k in list(result.keys()):
            if isinstance(result[k], list):
                result[k] = np.array(result[k])

        veto_info = []
        for i in range(num_cands):
            tid = cands["template_id"][i]
            t = self.template_by_id[tid]
            raw_norm = psd_plan.tile_norms[tile.tile_id][cands["template_idx"][i]]
            norm_val = (
                float(raw_norm.item())
                if hasattr(raw_norm, "item")
                else float(raw_norm)
            )
            snrv = np.array([cands["snr"][i] / norm_val])
            veto_info.append((snrv, norm_val, sample_indices[i], t, stilde))

        return result, veto_info

    def process_all(self) -> Any:
        """Process every batch group and return combined results with vetoes."""
        results = []
        veto_info = []

        while True:
            res, veto = self._process_batch()
            if res is False:
                return False
            if res is None:
                break
            if res:
                results.append(res)
                veto_info.extend(veto)

        if not results:
            return {}

        result = self.combine_results(results)

        # Global top-K selection before heavy vetoes
        if self.max_triggers_in_batch and len(result.get("snr", [])) > self.max_triggers_in_batch:
            sort = np.abs(result["snr"]).argsort()[::-1][: self.max_triggers_in_batch]
            for key in result:
                result[key] = result[key][sort]
            veto_info = [veto_info[i] for i in sort]

        # Calculate signal-based vetoes
        num_triggers = len(veto_info)
        chisq = np.zeros(num_triggers, dtype=np.float32)
        dof = np.full(num_triggers, 30, dtype=np.uint32)
        sg_chisq = np.ones(num_triggers, dtype=np.float32)

        for i, (snrv, norm, peak_index, htilde, stilde) in enumerate(veto_info):
            if hasattr(self.sg_chisq, "values") and getattr(self.sg_chisq, "do", False):
                sgv = self.sg_chisq.values(
                    stilde, htilde, stilde.psd, snrv, norm, chisq[i:i+1], dof[i:i+1], [peak_index]
                )
                if sgv is not None and len(sgv) > 0:
                    sg_chisq[i] = sgv[0]

        result["chisq"] = chisq
        result["chisq_dof"] = dof
        result["sg_chisq"] = sg_chisq

        # Reweighted NewSNR thresholding
        if self.newsnr_threshold and len(result.get("snr", [])) > 0:
            r_chisq = np.where(result["chisq_dof"] > 0, result["chisq"] / result["chisq_dof"], 1.0)
            newsnr = ranking.newsnr(np.abs(result["snr"]), r_chisq)
            keep = newsnr >= self.newsnr_threshold
            for key in result:
                result[key] = result[key][keep]

        return result

    def process_data(self, data_reader: Any) -> Any:
        self.set_data(data_reader)
        return self.process_all()
