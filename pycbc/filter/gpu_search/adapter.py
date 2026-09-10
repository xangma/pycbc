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
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
    BankPlan,
    PSDPlan,
    prepare_bank,
    bind_psd,
)
from pycbc.filter.gpu_search.candidates import (
    SelectionPolicy,
    select_tile_candidates,
)
from pycbc.filter.gpu_search.core import FilteringWorkspace, correlate_and_ifft
from pycbc.filter.gpu_search.engine import SearchEngine
from pycbc.filter.gpu_search.vetoes import (
    prepare_power_chisq_plan,
    VetoManager,
    _resolve_num_bins,
)
from pycbc.types.backend import wrap_backend_array, backend_array, torch_module_for


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
        num_threads: Optional[int] = None,
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
                device = getattr(scheme.mgr.state, "device_spec", "cpu")
                if num_threads is None:
                    num_threads = getattr(scheme.mgr.state, "num_threads", None)
            elif isinstance(scheme.mgr.state, scheme.CPUScheme):
                device = "numpy"
            elif torch is not None and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = str(device)
        self.num_threads = int(num_threads) if num_threads is not None else None

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

        # Preallocated batched workspaces for tiled/batched filtering
        self.workspace = FilteringWorkspace(
            max_batch_size=self.tile_size if self.tile_size > 1 else 0,
            tlen=self.tlen,
            flen=self.flen,
            device=self.device,
            dtype=self.dtype,
            num_slots=1,
            allocate_stilde=False,
        )

    @property
    def cout_workspace(self) -> Any:
        return self.workspace.cout_workspace

    @cout_workspace.setter
    def cout_workspace(self, val: Any):
        self.workspace.cout_workspace = val

    @property
    def out_workspace(self) -> Any:
        return self.workspace.out_workspace

    @out_workspace.setter
    def out_workspace(self, val: Any):
        self.workspace.out_workspace = val

    def _allocate_batched_workspaces(self, batch_size: Optional[int] = None):
        b = int(batch_size or self.tile_size)
        self.workspace.ensure_capacity(b)

    def prepare_template_batch(self, templates: Any) -> Any:
        """
        Normalize and pack a sequence of templates into a 2D batch tensor or array
        matching the adapter's device and filter length (flen).
        """
        is_torch_dev = torch is not None and self.device != "numpy"
        dev = torch.device(self.device) if is_torch_dev else None

        if isinstance(templates, (list, tuple)):
            if len(templates) == 0:
                if is_torch_dev:
                    return torch.empty(
                        (0, self.flen), device=dev, dtype=torch.complex64
                    )
                return np.empty((0, self.flen), dtype=np.complex64)

            for t in templates:
                t_arr = backend_array(t)
                if torch_module_for(t_arr) is None and not isinstance(
                    t_arr, np.ndarray
                ):
                    t_arr = np.asarray(t_arr)
                if t_arr.shape[-1] < self.flen:
                    raise ValueError(
                        f"template length ({t_arr.shape[-1]}) < filter length ({self.flen})"
                    )

            if is_torch_dev:
                first_storage = backend_array(templates[0])
                if torch_module_for(first_storage) is not None:
                    return torch.stack(
                        [backend_array(t)[: self.flen] for t in templates],
                        dim=0,
                    ).to(device=dev, dtype=torch.complex64)
                else:
                    np_stack = np.stack(
                        [np.asarray(t)[: self.flen] for t in templates], axis=0
                    )
                    return torch.from_numpy(np_stack).to(
                        device=dev, dtype=torch.complex64
                    )
            else:
                return np.stack(
                    [
                        (
                            backend_array(t)
                            .detach()
                            .cpu()
                            .numpy()[: self.flen]
                            if torch_module_for(backend_array(t)) is not None
                            else np.asarray(t)[: self.flen]
                        )
                        for t in templates
                    ],
                    axis=0,
                ).astype(np.complex64)

        if torch is not None and isinstance(templates, torch.Tensor):
            if templates.ndim != 2:
                raise ValueError(
                    f"templates must be 2D, got shape {templates.shape}"
                )
            if templates.shape[1] < self.flen:
                raise ValueError(
                    f"templates width ({templates.shape[1]}) < filter length ({self.flen})"
                )
            res = (
                templates[:, : self.flen]
                if templates.shape[1] > self.flen
                else templates
            )
            if is_torch_dev:
                if res.device == dev and res.dtype == torch.complex64:
                    return res
                return res.to(device=dev, dtype=torch.complex64)
            else:
                return res.detach().cpu().numpy().astype(np.complex64)

        if isinstance(templates, np.ndarray):
            if templates.ndim != 2:
                raise ValueError(
                    f"templates must be 2D, got shape {templates.shape}"
                )
            if templates.shape[1] < self.flen:
                raise ValueError(
                    f"templates width ({templates.shape[1]}) < filter length ({self.flen})"
                )
            res = (
                templates[:, : self.flen]
                if templates.shape[1] > self.flen
                else templates
            )
            if is_torch_dev:
                return torch.from_numpy(res).to(
                    device=dev, dtype=torch.complex64
                )
            else:
                if res.dtype == np.complex64:
                    return res
                return np.asarray(res, dtype=np.complex64)

        raise TypeError(
            f"Unsupported template container type: {type(templates)}"
        )

    def batched_matched_filter_and_cluster(
        self,
        segnum: int,
        templates: Any,
        sigmasqs: Sequence[float],
        window: int,
        epoch: Any = None,
    ) -> List[Tuple[Any, float, Any, np.ndarray, np.ndarray]]:
        """
        Calculate matched filter, threshold, and cluster for a batch of templates
        against segment segnum in a single vectorised execution pass.

        Parameters
        ----------
        segnum : int
            Index of the analysis segment to filter against.
        templates : torch.Tensor, np.ndarray, or sequence of FrequencySeries
            Batch of frequency-domain templates. If tensor or ndarray, shape is
            (B, filter_length). If sequence, each element has length >= flen.
        sigmasqs : sequence of float
            Normalization factors (template inner products) for each template.
        window : int
            Clustering window size in sample points.
        epoch : optional
            GPS start epoch for output TimeSeries.

        Returns
        -------
        results : list of tuples
            For each template in the batch, returns (snr, norm, corr, idx, snrv)
            matching the MatchedFilterControl contract.
        """
        b = len(sigmasqs)
        if b == 0:
            return []

        if self.cout_workspace is None or self.cout_workspace.shape[0] < b:
            self._allocate_batched_workspaces(batch_size=max(b, self.tile_size))

        seg = self.segments[segnum]
        valid_start = seg.analyze.start
        valid_end = seg.analyze.stop

        norms_np = np.asarray(
            [
                (4.0 * self.delta_f) / sqrt(s) if s > 0 else 0.0
                for s in sigmasqs
            ],
            dtype=np.float32,
        )

        if torch is not None and self.device != "numpy":
            dev = torch.device(self.device)
            if hasattr(seg, "_data") and hasattr(seg._data, "tensor"):
                seg_tensor = seg._data.tensor
            elif isinstance(seg, torch.Tensor):
                seg_tensor = seg
            else:
                seg_tensor = torch.as_tensor(
                    np.asarray(seg), device=dev, dtype=torch.complex64
                )
            if seg_tensor.device != dev:
                seg_tensor = seg_tensor.to(device=dev)

            # Templates to device tensor
            if (
                isinstance(templates, torch.Tensor)
                and templates.ndim == 2
                and templates.shape[1] >= self.flen
                and templates.device == dev
                and templates.dtype == torch.complex64
            ):
                tile_tensor = (
                    templates[:, :self.flen]
                    if templates.shape[1] > self.flen
                    else templates
                )
            else:
                tile_tensor = self.prepare_template_batch(templates)

            # Batched correlation and inverse FFT via shared core
            active_cout, active_out = correlate_and_ifft(
                templates=tile_tensor,
                data=seg_tensor,
                cout_workspace=self.cout_workspace,
                out_workspace=self.out_workspace,
                tlen=self.tlen,
                flen=self.flen,
                kmin=self.kmin,
                kmax=self.kmax,
                batch_size=b,
            )

            if self.use_cluster and self.cluster_function == "symmetric":
                norms_t = torch.as_tensor(
                    norms_np, device=dev, dtype=torch.float32
                )
                sigmasqs_t = torch.as_tensor(
                    sigmasqs, device=dev, dtype=torch.float32
                )

                policy = SelectionPolicy(
                    snr_threshold=self.snr_threshold,
                    cluster_policy="symmetric",
                    cluster_window=window,
                )

                sel = select_tile_candidates(
                    active_out,
                    norms_t,
                    sigmasqs_t,
                    valid_start,
                    valid_end,
                    policy,
                )

                cands = sel.get("candidates", {})
                t_indices = cands.get(
                    "template_idx", np.empty(0, dtype=np.int64)
                )
                s_indices = cands.get(
                    "sample_idx", np.empty(0, dtype=np.int64)
                )
                snr_vals = cands.get("snr", np.empty(0, dtype=np.complex64))

                results = []
                for i in range(b):
                    mask = (t_indices == i)
                    norm_i = float(norms_np[i])
                    if not np.any(mask):
                        results.append((
                            [],
                            norm_i,
                            [],
                            np.empty(0, dtype=np.uint32),
                            np.empty(0, dtype=np.complex64),
                        ))
                    else:
                        tmpl_idx_shifted = s_indices[mask]
                        tmpl_idx = (tmpl_idx_shifted - valid_start).astype(
                            np.uint32
                        )
                        tmpl_snrv = (snr_vals[mask] / norm_i).astype(
                            np.complex64
                        )

                        corr_row = active_cout[i]
                        out_row = active_out[i]
                        if isinstance(scheme.mgr.state, scheme.CPUScheme) or not isinstance(scheme.mgr.state, scheme.TorchScheme):
                            corr_arr = corr_row.detach().cpu().numpy() if hasattr(corr_row, "detach") else corr_row
                            out_arr = out_row.detach().cpu().numpy() if hasattr(out_row, "detach") else out_row
                            corr_i = FrequencySeries(
                                corr_arr,
                                delta_f=self.delta_f,
                                copy=False,
                            )
                            snr_i = TimeSeries(
                                out_arr,
                                delta_t=self.delta_t,
                                epoch=epoch,
                                copy=False,
                            )
                        else:
                            corr_i = FrequencySeries(
                                wrap_backend_array(corr_row),
                                delta_f=self.delta_f,
                                copy=False,
                            )
                            snr_i = TimeSeries(
                                wrap_backend_array(out_row),
                                delta_t=self.delta_t,
                                epoch=epoch,
                                copy=False,
                            )

                        results.append(
                            (snr_i, norm_i, corr_i, tmpl_idx, tmpl_snrv)
                        )
                return results

            from pycbc import events
            results = []
            for i in range(b):
                norm_i = float(norms_np[i])
                thresh_val = self.snr_threshold / norm_i
                corr_row = active_cout[i]
                out_row = active_out[i]
                if isinstance(scheme.mgr.state, scheme.CPUScheme) or not isinstance(scheme.mgr.state, scheme.TorchScheme):
                    corr_arr = corr_row.detach().cpu().numpy() if hasattr(corr_row, "detach") else corr_row
                    out_arr = out_row.detach().cpu().numpy() if hasattr(out_row, "detach") else out_row
                    corr_i = FrequencySeries(
                        corr_arr,
                        delta_f=self.delta_f,
                        copy=False,
                    )
                    snr_i = TimeSeries(
                        out_arr,
                        delta_t=self.delta_t,
                        epoch=epoch,
                        copy=False,
                    )
                else:
                    corr_i = FrequencySeries(
                        wrap_backend_array(corr_row),
                        delta_f=self.delta_f,
                        copy=False,
                    )
                    snr_i = TimeSeries(
                        wrap_backend_array(out_row),
                        delta_t=self.delta_t,
                        epoch=epoch,
                        copy=False,
                    )
                ana_snr = snr_i[seg.analyze]

                if self.use_cluster and self.cluster_function == "findchirp":
                    idx, snrv = events.threshold_and_cluster_findchirp(
                        ana_snr, thresh_val, window
                    )
                else:
                    clusterer = ThresholdCluster(ana_snr)
                    snrv, idx = clusterer.threshold_and_cluster(thresh_val, 1)

                if len(idx) == 0:
                    results.append((
                        [],
                        norm_i,
                        [],
                        np.empty(0, dtype=np.uint32),
                        np.empty(0, dtype=np.complex64),
                    ))
                else:
                    results.append((
                        snr_i,
                        norm_i,
                        corr_i,
                        np.asarray(idx, dtype=np.uint32),
                        np.asarray(snrv, dtype=np.complex64),
                    ))
            return results

        else:
            # NumPy execution
            seg_np = np.asarray(seg)[:self.flen]
            if (
                isinstance(templates, np.ndarray)
                and templates.ndim == 2
                and templates.shape[1] >= self.flen
                and templates.dtype == np.complex64
            ):
                tile_np = (
                    templates[:, :self.flen]
                    if templates.shape[1] > self.flen
                    else templates
                )
            else:
                tile_np = self.prepare_template_batch(templates)

            # Batched correlation and inverse FFT via shared core
            active_cout, active_out = correlate_and_ifft(
                templates=tile_np,
                data=seg_np,
                cout_workspace=self.cout_workspace,
                out_workspace=self.out_workspace,
                tlen=self.tlen,
                flen=self.flen,
                kmin=self.kmin,
                kmax=self.kmax,
                batch_size=b,
            )

            if self.use_cluster and self.cluster_function == "symmetric":
                policy = SelectionPolicy(
                    snr_threshold=self.snr_threshold,
                    cluster_policy="symmetric",
                    cluster_window=window,
                )
                sel = select_tile_candidates(
                    active_out,
                    norms_np,
                    np.asarray(sigmasqs, dtype=np.float32),
                    valid_start,
                    valid_end,
                    policy,
                )
                cands = sel.get("candidates", {})
                t_indices = cands.get(
                    "template_idx", np.empty(0, dtype=np.int64)
                )
                s_indices = cands.get(
                    "sample_idx", np.empty(0, dtype=np.int64)
                )
                snr_vals = cands.get("snr", np.empty(0, dtype=np.complex64))

                results = []
                for i in range(b):
                    mask = (t_indices == i)
                    norm_i = float(norms_np[i])
                    if not np.any(mask):
                        results.append((
                            [],
                            norm_i,
                            [],
                            np.empty(0, dtype=np.uint32),
                            np.empty(0, dtype=np.complex64),
                        ))
                    else:
                        tmpl_idx_shifted = s_indices[mask]
                        tmpl_idx = (tmpl_idx_shifted - valid_start).astype(
                            np.uint32
                        )
                        tmpl_snrv = (snr_vals[mask] / norm_i).astype(
                            np.complex64
                        )
                        corr_i = FrequencySeries(
                            active_cout[i],
                            delta_f=self.delta_f,
                            copy=False,
                        )
                        snr_i = TimeSeries(
                            active_out[i],
                            delta_t=self.delta_t,
                            epoch=epoch,
                            copy=False,
                        )
                        results.append(
                            (snr_i, norm_i, corr_i, tmpl_idx, tmpl_snrv)
                        )
                return results

            from pycbc import events
            results = []
            for i in range(b):
                norm_i = float(norms_np[i])
                thresh_val = self.snr_threshold / norm_i
                corr_i = FrequencySeries(
                    active_cout[i],
                    delta_f=self.delta_f,
                    copy=False,
                )
                snr_i = TimeSeries(
                    active_out[i],
                    delta_t=self.delta_t,
                    epoch=epoch,
                    copy=False,
                )
                ana_snr = snr_i[seg.analyze]

                if self.use_cluster and self.cluster_function == "findchirp":
                    idx, snrv = events.threshold_and_cluster_findchirp(
                        ana_snr, thresh_val, window
                    )
                else:
                    clusterer = ThresholdCluster(ana_snr)
                    snrv, idx = clusterer.threshold_and_cluster(thresh_val, 1)

                if len(idx) == 0:
                    results.append((
                        [],
                        norm_i,
                        [],
                        np.empty(0, dtype=np.uint32),
                        np.empty(0, dtype=np.complex64),
                    ))
                else:
                    results.append((
                        snr_i,
                        norm_i,
                        corr_i,
                        np.asarray(idx, dtype=np.uint32),
                        np.asarray(snrv, dtype=np.complex64),
                    ))
            return results

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
        use_cuda_graphs: bool = False,
        num_threads: Optional[int] = None,
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
        self.use_cuda_graphs = bool(use_cuda_graphs)
        self.maxelements = maxelements

        if device is None:
            if isinstance(scheme.mgr.state, scheme.TorchScheme):
                device = getattr(scheme.mgr.state, "device_spec", "cpu")
                if num_threads is None:
                    num_threads = getattr(scheme.mgr.state, "num_threads", None)
            elif torch is not None and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = str(device)
        self.num_threads = int(num_threads) if num_threads is not None else None

        # Sort templates by duration and length
        durations = np.array([1.0 / float(t.delta_f) for t in templates])
        lengths = np.array([len(t) for t in templates])
        lsort = np.lexsort((lengths, durations))
        self.templates = [templates[li] for li in lsort]

        # Group templates by uniform (delta_f, len(t))
        group_keys = [(float(t.delta_f), len(t)) for t in self.templates]
        self.tgroups = []
        cur_group = []
        cur_key = None
        for t, k in zip(self.templates, group_keys):
            if cur_key is None or k == cur_key:
                cur_group.append(t)
                cur_key = k
            else:
                self.tgroups.append(cur_group)
                cur_group = [t]
                cur_key = k
        if cur_group:
            self.tgroups.append(cur_group)

        self.selection_policy = SelectionPolicy(
            snr_threshold=self.snr_threshold,
            snr_abort_threshold=self.snr_abort_threshold,
            cluster_policy="live_peak",
            max_triggers_in_batch=self.max_triggers_in_batch,
        )

        from pycbc import vetoes

        self.power_chisq = vetoes.SingleDetPowerChisq(self.chisq_bins, None)

        self.bank_plans = []
        self.engines = []
        for g_idx, tgroup in enumerate(self.tgroups):
            tlen = (len(tgroup[0]) - 1) * 2
            eff_tile_size = self.tile_size
            if self.maxelements is not None and tlen > 0:
                calc_rows = max(1, self.maxelements // tlen)
                eff_tile_size = min(self.tile_size, calc_rows)

            bp = prepare_bank(
                tgroup,
                tile_size=eff_tile_size,
                f_lower=getattr(tgroup[0], "f_lower", None),
                device=self.device,
            )
            engine = SearchEngine(
                bank_plan=bp,
                selection_policy=self.selection_policy,
                device=self.device,
                use_cuda_graphs=self.use_cuda_graphs,
                num_threads=self.num_threads,
            )
            self.bank_plans.append(bp)
            self.engines.append(engine)

        self.batch_blocks = []
        for g_idx, bp in enumerate(self.bank_plans):
            for tile in bp.tiles:
                self.batch_blocks.append((g_idx, bp, self.engines[g_idx], tile))

        self.data = None
        self.block_id = 0
        self._psd_plans = {}
        self._veto_managers = {}
        self.template_by_id = {}
        for i, t in enumerate(templates):
            self.template_by_id[getattr(t, "id", i)] = t
        for t in self.templates:
            if hasattr(t, "id"):
                self.template_by_id[t.id] = t

    def _empty_result(self) -> Dict[str, np.ndarray]:
        res = {
            "end_time": np.empty(0, dtype=np.float64),
            "snr": np.empty(0, dtype=np.float32),
            "coa_phase": np.empty(0, dtype=np.float32),
            "template_id": np.empty(0, dtype=np.uint64),
            "sigmasq": np.empty(0, dtype=np.float32),
            "chisq": np.empty(0, dtype=np.float32),
            "chisq_dof": np.empty(0, dtype=np.uint32),
            "sg_chisq": np.empty(0, dtype=np.float32),
        }
        if self.templates:
            t0 = self.templates[0]
            tparams = getattr(t0, "params", None)
            if tparams is not None and hasattr(tparams, "dtype") and tparams.dtype.names:
                for name in tparams.dtype.names:
                    res[name] = np.empty(0, dtype=tparams.dtype[name])
        return res

    def set_data(self, data: Any):
        """Set the data reader object to use."""
        self.data = data
        self.block_id = 0

    def combine_results(
        self, results: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Combine results from different batches of filtering."""
        non_empty = [
            r for r in results if r and len(r.get("end_time", [])) > 0
        ]
        if not non_empty:
            return self._empty_result()

        result = {}
        for key in non_empty[0].keys():
            result[key] = np.concatenate([r[key] for r in non_empty if key in r])
        return result

    _combine_results = combine_results

    def _get_or_bind_psd(
        self, g_idx: int, bp: BankPlan, psd: Any, engine: SearchEngine
    ) -> PSDPlan:
        psd_key = id(psd)
        cache_key = (g_idx, psd_key)
        if cache_key not in self._psd_plans:
            psd_plan = bind_psd(bp, psd, device=self.device)
            self._psd_plans[cache_key] = psd_plan

            # Evaluate num_bins per template to handle template-dependent expressions (e.g. params.mass1)
            num_bins_map = {}
            for tile in bp.tiles:
                for t_id in tile.template_ids:
                    tmpl = None
                    if getattr(bp, "template_by_id", None) and t_id in bp.template_by_id:
                        tmpl = bp.template_by_id[t_id]
                    elif t_id in self.template_by_id:
                        tmpl = self.template_by_id[t_id]
                    elif getattr(tile, "template_by_id", None) and t_id in tile.template_by_id:
                        tmpl = tile.template_by_id[t_id]
                    else:
                        for t in getattr(bp, "templates", []) or []:
                            if getattr(t, "id", None) == t_id:
                                tmpl = t
                                break

                    nb = _resolve_num_bins(self.chisq_bins, t_id, tmpl)
                    num_bins_map[t_id] = nb

            chisq_plan = prepare_power_chisq_plan(
                bp,
                psd_plan,
                num_bins=num_bins_map,
                snr_threshold=self.snr_threshold,
                device=self.device,
            )
            self._veto_managers[cache_key] = VetoManager(
                power_chisq_plan=chisq_plan
            )

        # Reactivate the cached VetoManager corresponding to this PSD plan on the engine
        engine.veto_manager = self._veto_managers[cache_key]
        return self._psd_plans[cache_key]

    def _process_batch(self) -> Tuple[Any, Any]:
        """Process a single tile / batch group of data."""
        if self.block_id >= len(self.batch_blocks):
            return None, None

        g_idx, bp, engine, tile = self.batch_blocks[self.block_id]
        delta_f = bp.geometry.delta_f
        stilde = self.data.overwhitened_data(delta_f)
        psd = getattr(stilde, "psd", None)
        psd_plan = self._get_or_bind_psd(g_idx, bp, psd, engine)

        psize = bp.geometry.transform_length
        valid_end = int(psize - self.data.trim_padding)
        valid_start = int(
            valid_end - self.data.blocksize * self.data.sample_rate
        )

        ticket = engine.submit(
            stilde,
            psd_plan,
            valid_interval=(valid_start, valid_end),
            block_id=self.block_id,
            tile_id=tile.tile_id,
            already_overwhitened=True,
        )

        self.block_id += 1

        if ticket.aborted:
            return False, False

        if getattr(ticket, "overflow", False):
            raise RuntimeError(
                "Candidate buffer overflow: maximum capacity exceeded"
            )

        ready = engine.drain()
        if not ready:
            return self._empty_result(), []

        batch_ticket = ready[0]
        if getattr(batch_ticket, "overflow", False):
            raise RuntimeError(
                "Candidate buffer overflow: maximum capacity exceeded"
            )
        if not batch_ticket.results:
            return self._empty_result(), []

        cands = batch_ticket.results[0]
        num_cands = len(cands.get("template_id", []))
        if num_cands == 0:
            return self._empty_result(), []

        result = {}
        sample_indices = cands["sample_idx"]
        complex_snrs = np.asarray(cands["snr"], dtype=np.complex64)
        result["end_time"] = (
            self.data.start_time
            + (sample_indices - valid_start).astype(np.float64) / self.data.sample_rate
        )
        result["snr"] = np.abs(complex_snrs).astype(np.float32)
        result["coa_phase"] = np.angle(complex_snrs).astype(np.float32)
        result["sigmasq"] = np.asarray(cands["sigmasq"], dtype=np.float32)
        result["template_id"] = np.asarray(cands["template_id"], dtype=np.uint64)
        if "chisq" in cands:
            result["chisq"] = np.asarray(cands["chisq"], dtype=np.float32)
        if "chisq_dof" in cands:
            result["chisq_dof"] = np.asarray(cands["chisq_dof"], dtype=np.uint32)
        if "sg_chisq" in cands:
            result["sg_chisq"] = np.asarray(cands["sg_chisq"], dtype=np.float32)

        # Add template params
        tparams_keys = []
        for tid in cands["template_id"]:
            t = self.template_by_id[tid]
            tparams = getattr(t, "params", None)
            if tparams is not None and hasattr(tparams, "dtype") and tparams.dtype.names:
                for name in tparams.dtype.names:
                    if name not in result:
                        result[name] = []
                        tparams_keys.append(name)
                    result[name].append(tparams[name])

        for k in tparams_keys:
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
            snrv = np.array([complex_snrs[i] / norm_val])
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
            if res and len(res.get("snr", [])) > 0:
                results.append(res)
                veto_info.extend(veto)

        if not results:
            return self._empty_result()

        result = self.combine_results(results)
        if len(result.get("snr", [])) == 0:
            return self._empty_result()

        # Global top-K selection before heavy vetoes
        if self.max_triggers_in_batch and len(result.get("snr", [])) > self.max_triggers_in_batch:
            sort = result["snr"].argsort()[::-1][: self.max_triggers_in_batch]
            for key in result:
                result[key] = result[key][sort]
            veto_info = [veto_info[i] for i in sort]

        # Calculate signal-based vetoes
        num_triggers = len(veto_info)
        if "chisq" in result and len(result["chisq"]) == num_triggers:
            chisq = np.asarray(result["chisq"], dtype=np.float32)
            dof = np.asarray(
                result.get("chisq_dof", np.full(num_triggers, 30, dtype=np.uint32)),
                dtype=np.uint32,
            )
        else:
            chisq = np.zeros(num_triggers, dtype=np.float32)
            dof = np.full(num_triggers, 30, dtype=np.uint32)
            for i, (snrv, norm, peak_index, htilde, stilde) in enumerate(veto_info):
                if hasattr(self.power_chisq, "values") and getattr(self.power_chisq, "do", False):
                    cout = getattr(htilde, "cout", None)
                    c, d = self.power_chisq.values(
                        cout, snrv, norm, stilde.psd, [peak_index], htilde
                    )
                    if c is not None and d is not None:
                        chisq[i] = c[0]
                        dof[i] = d[0]

        sg_chisq = np.ones(num_triggers, dtype=np.float32)

        for i, (snrv, norm, peak_index, htilde, stilde) in enumerate(veto_info):
            if hasattr(self.sg_chisq, "values") and getattr(self.sg_chisq, "do", False):
                sgv = self.sg_chisq.values(
                    stilde,
                    htilde,
                    stilde.psd,
                    snrv,
                    norm,
                    chisq[i : i + 1],
                    dof[i : i + 1],
                    [peak_index],
                )
                if sgv is not None and len(sgv) > 0:
                    sg_chisq[i] = sgv[0]

        # Store reduced chi-square to match LiveBatchMatchedFilter and coinc.py contract
        result["chisq"] = np.where(dof > 0, chisq / dof, 1.0).astype(np.float32)
        result["chisq_dof"] = dof
        result["sg_chisq"] = sg_chisq

        # Reweighted NewSNR thresholding
        if self.newsnr_threshold and len(result.get("snr", [])) > 0:
            newsnr = ranking.newsnr(result["snr"], result["chisq"])
            keep = newsnr >= self.newsnr_threshold
            for key in result:
                result[key] = result[key][keep]
            if len(result["snr"]) == 0:
                return self._empty_result()

        return result

    def process_data(self, data_reader: Any) -> Any:
        self.set_data(data_reader)
        return self.process_all()
