# Copyright (C) 2017 Christopher M. Biwer
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
""" Functions for computing the Geweke convergence statistic.
"""

import numpy


def _torch_chain(x):
    """Return a floating Torch chain without importing Torch eagerly."""
    if type(x).__module__.split(".", 1)[0] != "torch":
        return None

    import torch

    if not isinstance(x, torch.Tensor):
        return None
    if not (x.is_floating_point() or x.is_complex()):
        return x.to(dtype=torch.get_default_dtype())
    return x


def geweke(x, seg_length, seg_stride, end_idx, ref_start,
           ref_end=None, seg_start=0):
    """ Calculates Geweke conervergence statistic for a chain of data.
    This function will advance along the chain and calculate the
    statistic for each step.

    Parameters
    ----------
    x : numpy.array or torch.Tensor
        A one-dimensional array of data. Torch inputs remain on their current
        device and produce Torch outputs.
    seg_length : int
        Number of samples to use for each Geweke calculation.
    seg_stride : int
        Number of samples to advance before next Geweke calculation.
    end_idx : int
        Index of last start.
    ref_start : int
        Index of beginning of end reference segment.
    ref_end : int
        Index of end of end reference segment. Default is None which
        will go to the end of the data array.
    seg_start : int
        What index to start computing the statistic. Default is 0 which
        will go to the beginning of the data array.

    Returns
    -------
    starts : numpy.array or torch.Tensor
        The start index of the first segment in the chain.
    ends : numpy.array or torch.Tensor
        The end index of the first segment in the chain.
    stats : numpy.array or torch.Tensor
        The Geweke convergence diagnostic statistic for the segment.
    """

    tensor = _torch_chain(x)
    if tensor is not None:
        import torch

        start_values = range(seg_start, end_idx, seg_stride)
        starts = torch.arange(
            seg_start, end_idx, seg_stride,
            device=tensor.device, dtype=torch.int64,
        )
        x_end = tensor[ref_start:ref_end]
        x_end_mean = x_end.mean()
        x_end_variance = x_end.var(correction=0)
        stats = []
        ends = []
        for start in start_values:
            x_start_end = start + seg_length
            x_start = tensor[start:x_start_end]
            stats.append(
                (x_start.mean() - x_end_mean) / torch.sqrt(
                    x_start.var(correction=0) + x_end_variance
                )
            )
            ends.append(x_start_end)
        stats = (
            torch.stack(stats)
            if stats else tensor.new_empty((0,))
        )
        ends = torch.tensor(
            ends, device=tensor.device, dtype=torch.int64
        )
        return starts, ends, stats

    # lists to hold statistic and end index
    stats = []
    ends = []

    # get the beginning of all segments
    starts = numpy.arange(seg_start, end_idx, seg_stride)

    # get second segment of data at the end to compare
    x_end = x[ref_start:ref_end]

    # loop over all segments
    for start in starts:

        # find the end of the first segment
        x_start_end = int(start + seg_length)

        # get first segment
        x_start = x[start:x_start_end]

        # compute statistic
        stats.append((x_start.mean() - x_end.mean()) / numpy.sqrt(
            x_start.var() + x_end.var()))

        # store end of first segment
        ends.append(x_start_end)

    return numpy.array(starts), numpy.array(ends), numpy.array(stats)
