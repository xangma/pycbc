# Copyright (C) 2025
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Torch backend for thresholding and clustering event triggers.
The implementations mirror the CUDA/CuPy backends but use Torch tensor
operations and keep scheme plumbing scheme‑agnostic.
"""

from __future__ import division

import numpy
import torch

from .eventmgr import _BaseThresholdCluster


def _as_tensor(series):
    """Return a 1D torch tensor view of the input series data.

    Accepts pycbc Array/TimeSeries/FrequencySeries with TorchArrayData,
    a raw torch.Tensor, or anything convertible via torch.as_tensor.
    """
    data = getattr(series, "_data", getattr(series, "data", series))
    if hasattr(data, "tensor"):
        return data.tensor.reshape(-1)
    if isinstance(data, torch.Tensor):
        return data.reshape(-1)
    return torch.as_tensor(data).reshape(-1)


def _threshold_mask(tensor, threshold):
    """Boolean mask of samples whose magnitude exceeds threshold."""
    thresh_sq = torch.as_tensor(threshold, device=tensor.device,
                                dtype=tensor.real.dtype)
    thresh_sq = thresh_sq * thresh_sq
    mag_sq = torch.abs(tensor) ** 2
    return mag_sq > thresh_sq, mag_sq


def threshold(series, value):
    """Return locations and values above the given threshold."""
    tensor = _as_tensor(series)
    mask, _ = _threshold_mask(tensor, value)
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    vals = tensor[mask]
    return idx.cpu().numpy(), vals.cpu().numpy()


def threshold_only(series, value):
    """Alias for threshold; kept for API parity."""
    return threshold(series, value)


def _cluster_candidates(mag_sq, window):
    """Return per-window maxima and indices (before neighbor clustering)."""
    length = mag_sq.numel()
    window = int(window)
    if window <= 0:
        raise ValueError("window must be positive")

    num_blocks = (length + window - 1) // window
    if num_blocks == 0:
        return (torch.empty(0, device=mag_sq.device, dtype=mag_sq.dtype),
                torch.empty(0, device=mag_sq.device, dtype=torch.long))

    pad = num_blocks * window - length
    if pad:
        pad_val = torch.full((pad,), float("-inf"), device=mag_sq.device,
                             dtype=mag_sq.dtype)
        mag_padded = torch.cat((mag_sq, pad_val))
    else:
        mag_padded = mag_sq

    mag_blocks = mag_padded.view(num_blocks, window)
    block_max, block_idx = torch.max(mag_blocks, dim=1)
    global_idx = block_idx + torch.arange(num_blocks, device=mag_sq.device) * window
    return block_max, global_idx


def _symmetric_cluster(max_vals, max_idx, threshold_sq):
    """Apply symmetric neighbor clustering (keep local maxima)."""
    nb = max_vals.numel()
    if nb == 0:
        return (torch.empty(0, device=max_vals.device, dtype=max_vals.dtype),
                torch.empty(0, device=max_idx.device, dtype=max_idx.dtype))

    mask_thresh = max_vals > threshold_sq
    # Use -inf for invalid entries so comparisons behave as expected
    comp_vals = torch.full((nb,), float("-inf"), device=max_vals.device,
                           dtype=max_vals.dtype)
    comp_vals[mask_thresh] = max_vals[mask_thresh]

    left_better = torch.zeros(nb, dtype=torch.bool, device=max_vals.device)
    right_better = torch.zeros(nb, dtype=torch.bool, device=max_vals.device)
    if nb > 1:
        left_better[1:] = comp_vals[:-1] > comp_vals[1:]
        right_better[:-1] = comp_vals[1:] > comp_vals[:-1]

    keep = mask_thresh & ~left_better & ~right_better
    return max_vals[keep], max_idx[keep]


def threshold_and_cluster(series, threshold, window):
    """Return clustered values and indices above threshold."""
    tensor = _as_tensor(series)
    mask, mag_sq = _threshold_mask(tensor, threshold)
    block_max, block_idx = _cluster_candidates(mag_sq, window)

    thresh_sq = torch.as_tensor(threshold, device=mag_sq.device,
                                dtype=mag_sq.dtype)
    thresh_sq = thresh_sq * thresh_sq

    kept_vals_mag, kept_idx = _symmetric_cluster(block_max, block_idx, thresh_sq)
    if kept_idx.numel() == 0:
        return (numpy.array([], dtype=numpy.complex64),
                numpy.array([], dtype=numpy.uint32))

    flat_series = tensor.reshape(-1)
    kept_vals = flat_series[kept_idx]
    return kept_vals.cpu().numpy(), kept_idx.cpu().numpy()


class TorchThresholdCluster(_BaseThresholdCluster):
    """Torch implementation of ThresholdCluster using tensor ops."""
    def __init__(self, series):
        self.series = _as_tensor(series)

    def threshold_and_cluster(self, threshold, window):
        mask, mag_sq = _threshold_mask(self.series, threshold)
        block_max, block_idx = _cluster_candidates(mag_sq, window)

        thresh_sq = torch.as_tensor(threshold, device=mag_sq.device,
                                    dtype=mag_sq.dtype)
        thresh_sq = thresh_sq * thresh_sq

        kept_vals_mag, kept_idx = _symmetric_cluster(block_max, block_idx, thresh_sq)
        if kept_idx.numel() == 0:
            return (numpy.array([], dtype=numpy.complex64),
                    numpy.array([], dtype=numpy.uint32))

        flat_series = self.series.reshape(-1)
        kept_vals = flat_series[kept_idx]
        return kept_vals.cpu().numpy(), kept_idx.cpu().numpy()


def _threshold_cluster_factory(series):
    return TorchThresholdCluster

