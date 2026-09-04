# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch backend for coincidence construction and clustering."""

import numpy
import torch

from pycbc.types import Array
from pycbc.types.array_torch import TorchArrayData


def _as_torch_tensor(value):
    """Return the tensor stored by a PyCBC array or a raw Torch tensor."""
    data = getattr(value, "_data", value)
    tensor = getattr(data, "tensor", None)
    if tensor is not None:
        return tensor
    return value if isinstance(value, torch.Tensor) else None


def _wrap_coincidence_result(t1, t2, *values):
    """Preserve PyCBC Array inputs while leaving raw tensors as tensors."""
    if not isinstance(t1, Array) and not isinstance(t2, Array):
        return values
    return tuple(
        Array(TorchArrayData(value), copy=False) for value in values
    )


def _wrap_cluster_result(inputs, value):
    """Return an Array when any public clustering input was an Array."""
    if not any(isinstance(item, Array) for item in inputs):
        return value
    return Array(TorchArrayData(value), copy=False)


def _cluster_vectors(values):
    """Move mixed clustering vectors beside the first Torch input."""
    tensors = [_as_torch_tensor(value) for value in values]
    reference = next((value for value in tensors if value is not None), None)
    if reference is None:
        raise TypeError("the Torch backend requires a Torch-backed input")

    result = []
    for value, tensor in zip(values, tensors):
        if tensor is None:
            tensor = torch.as_tensor(value, device=reference.device)
        if tensor.device != reference.device:
            raise ValueError("cluster arrays must use one device")
        if tensor.ndim != 1:
            raise TypeError("cluster arrays must be one-dimensional")
        result.append(tensor)
    return result


def time_coincidence(t1, t2, window, slide_step=0):
    """Find time coincidences without moving Torch inputs off-device."""
    tensor1 = _as_torch_tensor(t1)
    tensor2 = _as_torch_tensor(t2)
    reference = tensor1 if tensor1 is not None else tensor2
    if reference is None:
        raise TypeError("the Torch backend requires a Torch-backed input")

    def _as_time_tensor(value, tensor):
        if tensor is None:
            tensor = torch.as_tensor(
                value, dtype=reference.dtype, device=reference.device
            )
        if tensor.device != reference.device:
            raise ValueError("coincidence time arrays must use one device")
        if tensor.dtype != reference.dtype:
            raise TypeError("coincidence time arrays must use one dtype")
        if tensor.ndim != 1 or not tensor.is_floating_point():
            raise TypeError(
                "coincidence time arrays must be one-dimensional floating "
                "point values"
            )
        return tensor

    tensor1 = _as_time_tensor(t1, tensor1)
    tensor2 = _as_time_tensor(t2, tensor2)

    if slide_step:
        # Cython's cdivision path uses C fmod rather than Python remainder.
        fold1 = torch.fmod(tensor1, slide_step)
        fold2 = torch.fmod(tensor2, slide_step)
    else:
        fold1 = tensor1
        fold2 = tensor2

    sort1 = torch.argsort(fold1)
    sort2 = torch.argsort(fold2)
    fold1 = fold1[sort1]
    fold2 = fold2[sort2]
    if slide_step:
        fold2 = torch.cat((fold2 - slide_step, fold2, fold2 + slide_step))

    left = torch.searchsorted(fold2, fold1 - window)
    right = torch.searchsorted(fold2, fold1 + window)
    counts = right - left
    idx1 = torch.repeat_interleave(sort1, counts)

    if idx1.numel() and sort2.numel():
        repeated_left = torch.repeat_interleave(left, counts)
        group_starts = torch.repeat_interleave(
            torch.cumsum(counts, dim=0) - counts, counts
        )
        flat_positions = torch.arange(
            idx1.numel(), dtype=torch.int64, device=reference.device
        )
        folded_positions = repeated_left + flat_positions - group_starts
        idx2 = sort2[torch.remainder(folded_positions, sort2.numel())]
    else:
        idx2 = torch.empty(0, dtype=torch.int64, device=reference.device)

    if slide_step:
        difference = (tensor1[idx1] - tensor2[idx2]) / slide_step
        # C round(), used by the Cython path, rounds halfway away from zero.
        slide = torch.where(
            difference >= 0,
            torch.floor(difference + 0.5),
            torch.ceil(difference - 0.5),
        ).to(torch.int32)
    else:
        slide = torch.zeros_like(idx1, dtype=torch.int32)

    return _wrap_coincidence_result(t1, t2, idx1, idx2, slide)


def cluster_coincs(
        stat, time1, time2, timeslide_id, slide, window, **kwargs):
    """Cluster two-detector coincidences on the input device."""
    inputs = (stat, time1, time2, timeslide_id)
    tensors = _cluster_vectors(inputs)
    stat_tensor, time1_tensor, time2_tensor, slide_tensor = tensors
    if time1_tensor.numel() == 0 or time2_tensor.numel() == 0:
        empty = torch.empty(0, dtype=torch.int64, device=stat_tensor.device)
        return _wrap_cluster_result(inputs, empty)

    length = stat_tensor.numel()
    if any(value.numel() != length for value in tensors[1:]):
        raise ValueError("coincidence arrays must be equal length")
    if stat_tensor.is_complex() or slide_tensor.is_complex():
        raise TypeError("coincidence statistics and slide IDs must be real")
    if not time1_tensor.is_floating_point() \
            or not time2_tensor.is_floating_point():
        raise TypeError("coincidence times must be floating point")

    # Removing a common anchor preserves every window and ordering decision,
    # while retaining sub-second precision for GPS-scale input times.
    anchor = time1_tensor[:1]
    time = (time1_tensor - anchor) + (time2_tensor - anchor)
    if numpy.isfinite(slide):
        time = time + slide_tensor.to(time.dtype) * slide
    time = time * 0.5

    tslide = slide_tensor.to(time.dtype)
    span = (time.max() - time.min()) + window * 10
    time = time + span * tslide
    result = cluster_over_time(stat_tensor, time, window, **kwargs)
    return _wrap_cluster_result(inputs, result)


def cluster_coincs_multiifo(
        stat, time_coincs, timeslide_id, slide, window, **kwargs):
    """Cluster multi-detector coincidences on the input device."""
    inputs = (stat, *time_coincs, timeslide_id)
    tensors = _cluster_vectors(inputs)
    stat_tensor = tensors[0]
    time_tensors = tensors[1:-1]
    slide_tensor = tensors[-1]
    if not time_tensors or any(value.numel() == 0 for value in time_tensors):
        empty = torch.empty(0, dtype=torch.int64, device=stat_tensor.device)
        return _wrap_cluster_result(inputs, empty)

    length = stat_tensor.numel()
    if any(value.numel() != length for value in tensors[1:]):
        raise ValueError("coincidence arrays must be equal length")
    if stat_tensor.is_complex() or slide_tensor.is_complex():
        raise TypeError("coincidence statistics and slide IDs must be real")
    if any(not value.is_floating_point() for value in time_tensors):
        raise TypeError("coincidence times must be floating point")
    if any(value.dtype != time_tensors[0].dtype for value in time_tensors):
        raise TypeError("coincidence times must use one dtype")

    times = torch.stack(time_tensors)
    participating = times > 0
    num_ifos = participating.sum(dim=0)
    first_ifo = participating.to(torch.int64).argmax(dim=0)
    event_anchor = times.gather(0, first_ifo.unsqueeze(0)).squeeze(0)
    relative_sum = torch.where(
        participating, times - event_anchor, torch.zeros_like(times)
    ).sum(dim=0)
    time_avg = (
        event_anchor - event_anchor[:1]
        + relative_sum / num_ifos.to(times.dtype)
    )

    if numpy.isfinite(slide):
        time_avg = time_avg + (
            (num_ifos - 1).to(times.dtype)
            * slide_tensor.to(times.dtype)
            * slide
            / num_ifos.to(times.dtype)
        )

    tslide = slide_tensor.to(times.dtype)
    span = (time_avg.max() - time_avg.min()) + window * 10
    time_avg = time_avg + span * tslide
    result = cluster_over_time(stat_tensor, time_avg, window, **kwargs)
    return _wrap_cluster_result(inputs, result)


def _cluster_better(first, second, stat, sentinel, nan_high=True):
    """Select the earlier maximum represented by two index tensors."""
    first_valid = first != sentinel
    second_valid = second != sentinel
    first_value = stat[first.clamp_max(sentinel)]
    second_value = stat[second.clamp_max(sentinel)]

    if stat.is_floating_point():
        first_nan = torch.isnan(first_value) & first_valid
        second_nan = torch.isnan(second_value) & second_valid
        both_numeric = ~first_nan & ~second_nan
        both_nan = first_nan & second_nan
        if nan_high:
            second_better = second_nan & ~first_nan
        else:
            second_better = ~second_nan & first_nan
        second_better |= both_nan & (second < first)
        second_better |= both_numeric & (
            (second_value > first_value)
            | ((second_value == first_value) & (second < first))
        )
    else:
        second_better = (second_value > first_value) | (
            (second_value == first_value) & (second < first)
        )

    second_better = second_valid & (~first_valid | second_better)
    return torch.where(second_better, second, first)


def _cluster_window_maxima(stat, left, right, method):
    """Return first-maximum indices for many half-open ranges on-device."""
    length = stat.numel()
    sentinel = length
    stat_with_sentinel = torch.cat((stat, torch.zeros_like(stat[:1])))
    tree_size = 1 << (length - 1).bit_length()
    tree = torch.full(
        (2 * tree_size,), sentinel, dtype=torch.int64, device=stat.device
    )
    tree[tree_size:tree_size + length] = torch.arange(
        length, dtype=torch.int64, device=stat.device
    )

    # NumPy argmax treats the first NaN as the maximum. The Cython loop only
    # retains a NaN when it is the first value in the queried interval.
    nan_high = method == 'python'
    width = tree_size
    while width > 1:
        tree[width // 2:width] = _cluster_better(
            tree[width:2 * width:2],
            tree[width + 1:2 * width:2],
            stat_with_sentinel,
            sentinel,
            nan_high=nan_high,
        )
        width //= 2

    query_left = left + tree_size
    query_right = right + tree_size
    best = torch.full(
        (length,), sentinel, dtype=torch.int64, device=stat.device
    )
    absent = torch.full_like(best, sentinel)
    last_tree_index = 2 * tree_size - 1
    for _ in range(tree_size.bit_length()):
        take = (query_left < query_right) & ((query_left & 1) == 1)
        candidate = tree[query_left.clamp_max(last_tree_index)]
        best = _cluster_better(
            best,
            torch.where(take, candidate, absent),
            stat_with_sentinel,
            sentinel,
            nan_high=nan_high,
        )
        query_left = query_left + take.to(query_left.dtype)

        take = (query_left < query_right) & ((query_right & 1) == 1)
        query_right = query_right - take.to(query_right.dtype)
        candidate = tree[query_right.clamp_max(last_tree_index)]
        best = _cluster_better(
            best,
            torch.where(take, candidate, absent),
            stat_with_sentinel,
            sentinel,
            nan_high=nan_high,
        )
        query_left //= 2
        query_right //= 2

    if method == 'cython' and stat.is_floating_point():
        first_is_nan = torch.isnan(stat[left])
        best = torch.where(first_is_nan, left, best)
    return best


def cluster_over_time(stat, time, window, method='python',
                      argmax=numpy.argmax):
    """Cluster generalized transient events entirely on a Torch device."""
    stat_tensor = _as_torch_tensor(stat)
    time_tensor = _as_torch_tensor(time)
    reference = stat_tensor if stat_tensor is not None else time_tensor
    if reference is None:
        raise TypeError("the Torch backend requires a Torch-backed input")

    if method not in ('python', 'cython'):
        raise ValueError(f'Do not recognize method {method}')

    def _as_tensor(value, tensor):
        if tensor is None:
            tensor = torch.as_tensor(value, device=reference.device)
        if tensor.device != reference.device:
            raise ValueError("cluster arrays must use one device")
        if tensor.ndim != 1:
            raise TypeError("cluster arrays must be one-dimensional")
        return tensor

    stat_tensor = _as_tensor(stat, stat_tensor)
    time_tensor = _as_tensor(time, time_tensor)
    if stat_tensor.numel() != time_tensor.numel():
        raise ValueError(
            "cluster statistic and time arrays must be equal length"
        )
    if stat_tensor.is_complex() or not time_tensor.is_floating_point():
        raise TypeError(
            "cluster statistics must be real and times floating point"
        )

    length = time_tensor.numel()
    if length == 0:
        empty = torch.empty(0, dtype=torch.int64, device=reference.device)
        return _wrap_coincidence_result(stat, time, empty)[0]
    if method == 'python' and argmax not in (numpy.argmax, torch.argmax):
        raise NotImplementedError(
            "Torch clustering supports numpy.argmax or torch.argmax"
        )
    if window <= 0:
        raise ValueError("cluster window must be positive")

    time_sorting = torch.argsort(time_tensor)
    sorted_stat = stat_tensor[time_sorting]
    sorted_time = time_tensor[time_sorting]
    left = torch.searchsorted(sorted_time, sorted_time - window)
    right = torch.searchsorted(sorted_time, sorted_time + window)
    maxima = _cluster_window_maxima(sorted_stat, left, right, method)

    # The Python/Cython loop follows a strictly increasing successor graph.
    # Binary lifting evaluates every f**k(0) together, retaining the exact
    # greedy path without extracting device scalars for Python control flow.
    positions = torch.arange(
        length, dtype=torch.int64, device=reference.device
    )
    successor = torch.where(
        maxima == positions,
        right,
        torch.where(maxima > positions, maxima, positions + 1),
    )
    jump = torch.cat((successor, successor.new_tensor([length])))
    steps = positions
    visited_nodes = torch.zeros_like(steps)
    bit = 1
    while bit < length:
        visited_nodes = torch.where(
            (steps & bit) != 0, jump[visited_nodes], visited_nodes
        )
        jump = jump[jump]
        bit <<= 1

    visited = torch.zeros(
        length + 1, dtype=torch.bool, device=reference.device
    )
    visited.scatter_(0, visited_nodes, True)
    keep = torch.nonzero(
        visited[:length] & (maxima == positions), as_tuple=False
    ).flatten()
    result = time_sorting[keep]
    return _wrap_coincidence_result(stat, time, result)[0]
