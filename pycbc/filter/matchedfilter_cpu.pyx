# Copyright (C) 2018  Alex Nitz, Josh Willis
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
# cython: embedsignature=True
import numpy
from .matchedfilter import _BaseCorrelator
cimport numpy, cython
from cython.parallel import prange

ctypedef fused COMPLEXTYPE:
    float complex
    double complex


cdef inline void _abs_arg_max_complex64_row(
        float complex* values,
        numpy.int64_t* indices,
        float complex* peaks,
        unsigned int row,
        unsigned int row_size,
        unsigned int start,
        unsigned int stop) noexcept nogil:
    """Store the legacy complex abs-arg-max result for one row segment."""
    cdef unsigned int offset = row * row_size + start
    cdef unsigned int length = stop - start
    cdef unsigned int index
    cdef unsigned int best = 0
    cdef float complex value
    cdef double mag
    cdef double magmax = 0

    for index in range(length):
        value = values[offset + index]
        mag = (
            value.real * value.real
            + value.imag * value.imag
        )
        if mag > magmax:
            magmax = mag
            best = index

    indices[row] = best
    peaks[row] = values[offset + best]


@cython.boundscheck(False)
@cython.wraparound(False)
def _batch_abs_arg_max_complex64(
        numpy.ndarray [float complex, ndim=1, mode="c"] values,
        numpy.ndarray [numpy.int64_t, ndim=1, mode="c"] indices,
        numpy.ndarray [float complex, ndim=1, mode="c"] peaks,
        row_size,
        start,
        stop,
        num_vectors):
    """Find exact legacy peak indices and values for complex64 rows.

    Indices are relative to ``[start:stop]``, matching the scalar
    ``Array.abs_arg_max`` call used by ``LiveBatchMatchedFilter``.
    """
    row_size = int(row_size)
    start = int(start)
    stop = int(stop)
    num_vectors = int(num_vectors)
    if (
        num_vectors < 1
        or row_size < 1
        or start < 0
        or start >= stop
        or stop > row_size
        or num_vectors > 0xffffffff
        or row_size > 0xffffffff
        or num_vectors * row_size > 0xffffffff
        or values.shape[0] != num_vectors * row_size
        or indices.shape[0] != num_vectors
        or peaks.shape[0] != num_vectors
    ):
        raise ValueError("invalid batched peak geometry")

    cdef unsigned int nvec = num_vectors
    cdef unsigned int vsize = row_size
    cdef unsigned int begin = start
    cdef unsigned int end = stop
    cdef unsigned int row
    cdef float complex* value_ptr = <float complex*> values.data
    cdef numpy.int64_t* index_ptr = <numpy.int64_t*> indices.data
    cdef float complex* peak_ptr = <float complex*> peaks.data

    for row in prange(nvec, nogil=True, schedule="static"):
        _abs_arg_max_complex64_row(
            value_ptr,
            index_ptr,
            peak_ptr,
            row,
            vsize,
            begin,
            end,
        )

@cython.boundscheck(False)
@cython.wraparound(False)
def _batch_correlate(numpy.ndarray [long, ndim=1] x,
                     numpy.ndarray [float complex, ndim=1] y,
                     numpy.ndarray [long, ndim=1] z,
                     size, num_vectors):
    cdef unsigned int nvec = num_vectors
    cdef unsigned int vsize = size

    cdef float complex* xp
    cdef float complex* zp

    cdef unsigned int i, j

    for i in prange(nvec, nogil=True):
        xp = <float complex*> x[i]
        zp = <float complex*> z[i]
        for j in range(vsize):
            zp[j] = xp[j].conjugate() * y[j]

def batch_correlate_execute(self, y):
    num_vectors = self.num_vectors # pylint:disable=unused-variable
    size = self.size # pylint:disable=unused-variable
    _batch_correlate(self.x.data, y.data, self.z.data, size, num_vectors)

def correlate_numpy(x, y, z):
    z.data[:] = numpy.conjugate(x.data)[:]
    z *= y

@cython.boundscheck(False)
@cython.wraparound(False)
def _correlate(COMPLEXTYPE[:] x,
               COMPLEXTYPE[:] y,
               COMPLEXTYPE[:] z):
    cdef unsigned int xmax = x.shape[0]
    cdef unsigned int i
    for i in prange(xmax, nogil=True):
        z[i] = x[i].conjugate() * y[i]

def correlate(x, y, z):
    _correlate(x.data, y.data, z.data)

class CPUCorrelator(_BaseCorrelator):
    def __init__(self, x, y, z):
        self.x = numpy.array(x.data, copy=False)
        self.y = numpy.array(y.data, copy=False)
        self.z = numpy.array(z.data, copy=False)

    def correlate(self):
        _correlate(self.x, self.y, self.z)

def _correlate_factory(x, y, z):
    return CPUCorrelator
