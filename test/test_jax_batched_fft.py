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

"""Unit tests for PyCBC JAX batched FFT plans."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.fft import FFT, IFFT
from pycbc.types import Array

pytest.importorskip("jax")


@pytest.mark.parametrize("nbatch", [2, 4, 8])
def test_batched_c2c_fft(nbatch):
    """Verify batched complex-to-complex FFT executes accurately."""
    size = 64
    total_len = nbatch * size
    cdata = (
        np.random.randn(total_len) + 1j * np.random.randn(total_len)
    ).astype(np.complex128)

    invec = Array(cdata.copy())
    outvec = Array(np.zeros(total_len, dtype=np.complex128))
    recvec = Array(np.zeros(total_len, dtype=np.complex128))

    with scheme.JAXScheme():
        plan_f = FFT(invec, outvec, nbatch=nbatch, size=size)
        plan_i = IFFT(outvec, recvec, nbatch=nbatch, size=size)

        plan_f.execute()

        # Check each batch row against single numpy FFT
        in_matrix = cdata.reshape(nbatch, size)
        out_matrix = outvec.numpy().reshape(nbatch, size)
        for i in range(nbatch):
            ref = np.fft.fft(in_matrix[i])
            assert np.allclose(out_matrix[i], ref)

        plan_i.execute()
        rec_matrix = recvec.numpy().reshape(nbatch, size)
        for i in range(nbatch):
            assert np.allclose(rec_matrix[i], in_matrix[i] * size)


@pytest.mark.parametrize("nbatch", [2, 4])
def test_batched_r2c_and_c2r_fft(nbatch):
    """Verify batched real-to-complex FFT and complex-to-real IFFT."""
    size = 128
    flen = size // 2 + 1
    rdata = np.random.randn(nbatch * size).astype(np.float64)

    invec = Array(rdata.copy())
    outvec = Array(np.zeros(nbatch * flen, dtype=np.complex128))
    recvec = Array(np.zeros(nbatch * size, dtype=np.float64))

    with scheme.JAXScheme():
        plan_f = FFT(invec, outvec, nbatch=nbatch, size=size)
        plan_i = IFFT(outvec, recvec, nbatch=nbatch, size=size)

        plan_f.execute()

        in_matrix = rdata.reshape(nbatch, size)
        out_matrix = outvec.numpy().reshape(nbatch, flen)
        for i in range(nbatch):
            ref = np.fft.rfft(in_matrix[i])
            assert np.allclose(out_matrix[i], ref)

        plan_i.execute()
        rec_matrix = recvec.numpy().reshape(nbatch, size)
        for i in range(nbatch):
            assert np.allclose(rec_matrix[i], in_matrix[i] * size)
