# Copyright (C) 2025
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
Torch backend for matched filtering primitives.
"""

import torch
from .matchedfilter import _BaseCorrelator


def correlate(x, y, z):
    """Elementwise z = conj(x) * y."""
    z._data.tensor.copy_(torch.conj(x._data.tensor) * y._data.tensor)


class TorchCorrelator(_BaseCorrelator):
    def __init__(self, x, y, z):
        self.x = x._data.tensor
        self.y = y._data.tensor
        self.z = z._data.tensor

    def correlate(self):
        self.z.copy_(torch.conj(self.x) * self.y)


def _correlate_factory(x, y, z):
    return TorchCorrelator


def batch_correlate_execute(self, y):
    """Vectorised batch correlation for BatchCorrelator."""
    y_t = y._data.tensor
    for x, z in zip(self.xs, self.zs):
        z._data.tensor.copy_(torch.conj(x._data.tensor) * y_t)
    return self
