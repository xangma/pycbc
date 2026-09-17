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

import numpy as np
import pytest
from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.types.array_jax import empty, from_jax, is_jax_array, to_jax, zeros

jax = pytest.importorskip("jax")


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_to_jax_and_from_jax_dtypes(dtype):
    """Verify to_jax and from_jax preserve precision across dtypes."""
    if np.issubdtype(dtype, np.complexfloating):
        raw = np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=dtype)
    else:
        raw = np.array([1.5, -2.5, 3.25], dtype=dtype)

    jarr = to_jax(raw)
    assert is_jax_array(jarr)
    assert jarr.dtype == raw.dtype

    recovered = from_jax(jarr)
    assert isinstance(recovered, np.ndarray)
    assert recovered.dtype == raw.dtype
    assert np.array_equal(recovered, raw)


def test_to_jax_from_pycbc_types():
    """Verify to_jax converts PyCBC Array, TimeSeries, and FrequencySeries."""
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    arr = Array(data)
    ts = TimeSeries(data, delta_t=1.0 / 2048)
    fs = FrequencySeries(np.array([1.0 + 0j, 2.0 - 1.0j]), delta_f=1.0)

    j_arr = to_jax(arr)
    j_ts = to_jax(ts)
    j_fs = to_jax(fs)

    assert is_jax_array(j_arr)
    assert is_jax_array(j_ts)
    assert is_jax_array(j_fs)

    assert np.allclose(from_jax(j_arr), data)
    assert np.allclose(from_jax(j_ts), data)


def test_from_jax_target_type():
    """Verify from_jax can directly reconstruct PyCBC types."""
    jnp = jax.numpy
    jarr = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float64)

    pycbc_arr = from_jax(jarr, target_type=Array)
    assert isinstance(pycbc_arr, Array)
    assert np.allclose(pycbc_arr.numpy(), [10.0, 20.0, 30.0])


def test_zeros_and_empty():
    """Verify zeros and empty create valid JAX arrays."""
    z = zeros((4, 4), dtype=np.float64)
    assert is_jax_array(z)
    assert z.shape == (4, 4)
    assert z.dtype == np.float64
    assert np.all(from_jax(z) == 0.0)

    e = empty(8, dtype=np.complex128)
    assert is_jax_array(e)
    assert e.shape == (8,)
    assert e.dtype == np.complex128
