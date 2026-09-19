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
from pycbc.types.backend import (
    backend_array,
    backend_name,
    coerce_jax_values,
    is_backend,
)

jax = pytest.importorskip("jax")


def test_backend_name_numpy():
    """Verify numpy and PyCBC arrays report numpy backend."""
    arr = np.array([1.0, 2.0, 3.0])
    assert backend_name(arr) == "numpy"
    assert is_backend(arr, "numpy") is True
    assert is_backend(arr, "jax") is False

    pycbc_arr = Array(arr)
    assert backend_name(pycbc_arr) == "numpy"


def test_backend_name_jax():
    """Verify JAX arrays report jax backend."""
    jnp = jax.numpy
    jarr = jnp.array([1.0, 2.0, 3.0])
    assert backend_name(jarr) == "jax"
    assert is_backend(jarr, "jax") is True
    assert is_backend(jarr, "numpy") is False


def test_backend_array_unwrapping():
    """Verify backend_array unwraps PyCBC containers correctly."""
    jnp = jax.numpy
    data = np.array([1.0, 2.0, 3.0])
    ts = TimeSeries(data, delta_t=1.0 / 4096)
    fs = FrequencySeries(data, delta_f=1.0)

    assert isinstance(backend_array(ts), np.ndarray)
    assert isinstance(backend_array(fs), np.ndarray)

    jarr = jnp.array([1.0, 2.0, 3.0])
    assert backend_array(jarr) is jarr
    assert backend_array(jarr, name="jax") is jarr
    assert backend_array(jarr, name="torch") is None


def test_coerce_jax_values():
    """Verify coerce_jax_values converts mixed inputs to JAX arrays."""
    jnp = jax.numpy
    jarr = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    narr = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    scalar = 2.5

    mod, (out_j, out_n, out_s) = coerce_jax_values(jarr, narr, scalar)
    assert mod is jax
    assert isinstance(out_j, jax.Array)
    assert isinstance(out_n, jax.Array)
    assert isinstance(out_s, jax.Array)
    assert np.allclose(np.asarray(out_n), narr)
    assert np.allclose(np.asarray(out_s), scalar)


def test_coerce_jax_values_no_jax():
    """Verify coerce_jax_values returns None when no JAX inputs are passed."""
    narr = np.array([1.0, 2.0, 3.0])
    scalar = 5.0
    mod, vals = coerce_jax_values(narr, scalar)
    assert mod is None
    assert vals == (narr, scalar)
