"""Opt-in CUDA graphs for a fixed, sequential offline filtering controller.

Only ordinary contiguous complex64 tensors are eligible. Payloads and the
threshold may change; storage, analysis geometry, stream and thread may not.
"""

import os
import threading
import weakref
from math import sqrt

import torch

_failed_cleanup = []


def _tensor_binding(tensor, dtype=torch.complex64, ndim=1):
    if not (
        type(tensor) is torch.Tensor
        and tensor.layout == torch.strided
        and tensor.device.type == "cuda"
        and tensor.dtype == dtype
        and tensor.ndim == ndim
        and tensor.numel() > 0
        and tensor.is_contiguous()
        and not tensor.requires_grad
        and not torch.is_inference(tensor)
        and not tensor.is_conj()
        and not tensor.is_neg()
        and torch.autograd.forward_ad.unpack_dual(tensor).tangent is None
    ):
        raise ValueError("Unsupported CUDA graph tensor")
    return (
        id(tensor),
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
        tensor.untyped_storage().data_ptr(),
        tensor.untyped_storage().nbytes(),
    )


def _binding(control, segnum, window):
    """Return guarded metadata and the owners of all captured operand storage."""
    from pycbc.events.threshold_torch import TorchThresholdCluster
    from pycbc.fft.torchfft import IFFT

    from .matchedfilter import MatchedFilterControl
    from .matchedfilter_torch import TorchCorrelator

    try:
        if (
            type(control) is not MatchedFilterControl
            or type(window) is not int
            or window <= 0
            or type(segnum) is not int
            or not 0 <= segnum < len(control.segments)
            or control.cluster_function != "symmetric"
            or type(control.ifft) is not IFFT
            or control.ifft.nbatch != 1
            or torch.is_inference_mode_enabled()
            or torch.is_autocast_enabled("cuda")
        ):
            return None
        segment = control.segments[segnum]
        correlator = control.correlators[segnum]
        clusterer = control.threshold_and_clusterers[segnum]
        if (
            type(correlator) is not TorchCorrelator
            or type(clusterer) is not TorchThresholdCluster
        ):
            return None
        template, data, corr, snr = (
            value._data.tensor
            for value in (control.htilde, segment, control.corr_mem, control.snr_mem)
        )
        fft_input = control.ifft.invec._data.tensor
        fft_output = control.ifft.outvec._data.tensor
        tensors = (
            template,
            data,
            corr,
            snr,
            fft_input,
            fft_output,
            correlator.x,
            correlator.y,
            correlator.z,
            clusterer.series,
            clusterer._source,
        )
        signatures = tuple(_tensor_binding(tensor) for tensor in tensors)
        device = snr.device
        if (
            any(value.device != device for value in tensors)
            or torch.cuda.current_device() != device.index
            or corr.numel() != control.tlen
            or snr.numel() != control.tlen
            or fft_input.data_ptr() != corr.data_ptr()
            or fft_output.data_ptr() != snr.data_ptr()
            or fft_input.shape != corr.shape
            or fft_output.shape != snr.shape
            or max(corr.data_ptr(), snr.data_ptr())
            < min(
                corr.data_ptr() + corr.numel() * corr.element_size(),
                snr.data_ptr() + snr.numel() * snr.element_size(),
            )
        ):
            return None
        for source in (template, data):
            for target in (corr, snr):
                if max(source.data_ptr(), target.data_ptr()) < min(
                    source.data_ptr() + source.numel() * source.element_size(),
                    target.data_ptr() + target.numel() * target.element_size(),
                ):
                    return None
        for value, original in (
            (correlator.x, template),
            (correlator.y, data),
            (correlator.z, corr),
        ):
            if (
                not 0 <= control.kmin < control.kmax <= original.numel()
                or value.data_ptr()
                != original.data_ptr() + control.kmin * original.element_size()
                or value.numel() != control.kmax - control.kmin
            ):
                return None
        analysis = segment.analyze
        if (
            analysis.step not in (None, 1)
            or not 0 <= analysis.start < analysis.stop <= control.tlen
            or clusterer.series is not clusterer._source
            or clusterer.series.data_ptr()
            != snr.data_ptr() + analysis.start * snr.element_size()
            or clusterer.series.numel() != analysis.stop - analysis.start
        ):
            return None
        methods = (
            control.ifft.execute,
            correlator.correlate,
            clusterer.symmetric_cuda_graph_step,
        )
        method_ids = tuple(
            (
                id(getattr(method, "__self__", None)),
                id(getattr(method, "__func__", method)),
            )
            for method in methods
        )
        signature = (
            signatures,
            method_ids,
            id(control.ifft),
            id(correlator),
            id(clusterer),
            control.tlen,
            control.kmin,
            control.kmax,
            control.delta_f,
            analysis.start,
            analysis.stop,
            window,
            os.getpid(),
            threading.get_ident(),
            torch.cuda.current_stream().cuda_stream,
        )
        return signature, tensors, (control.ifft, correlator, clusterer)
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
        return None


def _scratch_binding(clusterer, window):
    count = (clusterer.series.numel() + window - 1) // window
    if clusterer._triton_scratch_nb != count:
        raise ValueError("CUDA graph scratch size changed")
    tensors = (
        clusterer._triton_block_max,
        clusterer._triton_block_idx,
        clusterer._triton_keep,
    )
    signatures = tuple(
        _tensor_binding(value, dtype=dtype)
        for value, dtype in zip(
            tensors, (torch.float32, torch.int64, torch.bool), strict=True
        )
    )
    if any(
        tuple(value.shape) != (count,) or value.device != clusterer.series.device
        for value in tensors
    ):
        raise ValueError("CUDA graph scratch binding changed")
    return signatures, tensors


def _release(resources):
    if os.getpid() != resources["pid"]:
        return
    try:
        with torch.cuda.device(resources["device"]):
            torch.cuda.synchronize(resources["device"])
            if resources.get("graph") is not None:
                resources["graph"].reset()
    except BaseException:
        # Retain every captured operand if pending GPU work cannot be ruled out.
        _failed_cleanup.append(resources)
        raise
    resources.clear()


class _GraphEntry:
    def __init__(self, binding, scratch, raw, squared, stream):
        self.signature, tensors, owners = binding
        self.scratch_signature, scratch_tensors = scratch
        self.raw, self.squared = raw, squared
        self.threshold_signature = (
            _tensor_binding(raw, torch.float32, 0),
            _tensor_binding(squared, torch.float32, 0),
        )
        captured = tensors + scratch_tensors + (raw, squared)
        self.resources = dict(
            pid=os.getpid(),
            device=raw.device,
            stream=stream,
            tensors=captured,
            # Tensor.data/set_ can rebind the same Tensor object. Keep the
            # original storage alive until synchronization and graph reset.
            storages=tuple(value.untyped_storage() for value in captured),
            owners=owners,
        )
        self.close = weakref.finalize(self, _release, self.resources)

    def record_stream(self, stream):
        # Also protect allocations if resize_ changes an existing Storage.
        for tensor in self.resources["tensors"]:
            tensor.record_stream(stream)

    def matches(self, binding, clusterer, window):
        if binding is None or binding[0] != self.signature:
            return False
        try:
            return self.scratch_signature == _scratch_binding(clusterer, window)[
                0
            ] and self.threshold_signature == (
                _tensor_binding(self.raw, torch.float32, 0),
                _tensor_binding(self.squared, torch.float32, 0),
            )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False


def clear_cuda_graphs(control):
    """Synchronize and release captured graphs, then allow a fresh opt-in."""
    entries = getattr(control, "_cuda_graphs", {})
    if any(entry.resources.get("pid") != os.getpid() for entry in entries.values()):
        raise RuntimeError("Cannot release CUDA graphs inherited across fork")
    control._cuda_graphs = {}
    control._cuda_graph_enabled = False
    control._cuda_graph_rejected = False
    error = None
    for entry in entries.values():
        try:
            entry.close()
        except BaseException as exc:
            error = exc
    if error is not None:
        raise error


def _reject(control):
    clear_cuda_graphs(control)
    control._cuda_graph_rejected = True
    return False


def capture_symmetric_cuda_graph(control, segnum, window, template_norm=1.0):
    """Capture one eligible segment; unsupported bindings use eager filtering."""
    if getattr(control, "_cuda_graph_rejected", False):
        return False
    binding = _binding(control, segnum, window)
    entries = getattr(control, "_cuda_graphs", {})
    if binding is None or any(key[1] != window for key in entries):
        return _reject(control)
    clusterer = control.threshold_and_clusterers[segnum]
    key = (segnum, window)
    if key in entries:
        if entries[key].matches(binding, clusterer, window):
            control._cuda_graph_enabled = True
            return True
        return _reject(control)
    if not clusterer.prepare_symmetric_cuda_graph(window):
        return _reject(control)
    norm = (4.0 * control.delta_f) / sqrt(template_norm)
    raw = torch.tensor(
        control.snr_threshold / norm,
        device=clusterer.series.device,
        dtype=torch.float32,
    )
    squared = torch.empty_like(raw)
    stream = torch.cuda.Stream()
    entry = _GraphEntry(
        binding, _scratch_binding(clusterer, window), raw, squared, stream
    )
    stream.wait_stream(torch.cuda.current_stream())
    try:
        entry.record_stream(stream)
        with torch.cuda.stream(stream):
            for _ in range(3):
                control.correlators[segnum].correlate()
                control.ifft.execute()
                torch.mul(raw, raw, out=squared)
                clusterer.symmetric_cuda_graph_step(window, squared)
            graph = torch.cuda.CUDAGraph()
            entry.resources["graph"] = graph
            with torch.cuda.graph(graph, stream=stream):
                control.correlators[segnum].correlate()
                control.ifft.execute()
                torch.mul(raw, raw, out=squared)
                clusterer.symmetric_cuda_graph_step(window, squared)
        torch.cuda.current_stream().wait_stream(stream)
    except BaseException:
        control._cuda_graph_rejected = True
        try:
            entry.close()
        except BaseException:
            pass  # Preserve the capture error; failed cleanup retains resources.
        raise
    entries[key] = entry
    control._cuda_graphs = entries
    control._cuda_graph_enabled = True
    return True


def replay_symmetric_cuda_graph(control, segnum, window, template_norm, threshold):
    """Replay after checking bindings, or return None for eager filtering."""
    if not capture_symmetric_cuda_graph(control, segnum, window, template_norm):
        return None
    entry = control._cuda_graphs[(segnum, window)]
    # Match eager conversion to float32 *before* squaring the threshold.
    try:
        entry.record_stream(torch.cuda.current_stream())
        entry.raw.fill_(float(threshold))
        entry.resources["graph"].replay()
    except BaseException:
        try:
            _reject(control)
        except BaseException:
            pass
        raise
    torch.autograd.graph.increment_version(
        (control.corr_mem._data.tensor, control.snr_mem._data.tensor)
    )
    return control.threshold_and_clusterers[segnum].symmetric_cuda_graph_result()
