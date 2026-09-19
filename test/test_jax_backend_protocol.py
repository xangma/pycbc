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
from pycbc.scheme import JAXScheme
from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.types.backend import (
    backend_array,
    backend_name,
    coerce_jax_values,
    is_backend,
)

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


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
    assert backend_array(jarr, name="other") is None


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


def test_coerce_jax_values_uses_reference_device():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("multiple JAX CPU devices unavailable")
    reference = jax.device_put(np.asarray([1, 2], np.float32), devices[1])
    other = jax.device_put(np.asarray([3, 4], np.float64), devices[0])
    _, converted = coerce_jax_values(reference, other, np.asarray([5, 6]))
    for value in converted:
        assert value.device == reference.device
        assert value.dtype == reference.dtype


def test_coerce_jax_values_accepts_tracers():
    @jax.jit
    def mixed(values):
        _, (left, right) = coerce_jax_values(values, np.asarray([3, 4]))
        return left + right

    np.testing.assert_array_equal(mixed(jax.numpy.asarray([1, 2])), [4, 6])


@pytest.mark.parametrize("dtype, complex_dtype", [(np.float32, np.complex64),
                                                  (np.float64, np.complex128)])
@pytest.mark.parametrize("complex_first", [False, True])
@pytest.mark.parametrize("host_operand", [False, True])
def test_mixed_complex_coercion_preserves_imaginary_components(
        dtype, complex_dtype, complex_first, host_operand):
    with JAXScheme("cpu") as context:
        real = jax.device_put(np.array([2., 3.], dtype=dtype), context.jax_device)
        complex_values = np.array([4.+1.j, 5.-2.j], dtype=complex_dtype)
        other = (complex_values if host_operand else
                 jax.device_put(complex_values, context.jax_device))
        values = (other, real) if complex_first else (real, other)
        _, converted = coerce_jax_values(*values)
        for value in converted:
            assert value.dtype == np.dtype(complex_dtype)
            assert value.devices() == {context.jax_device}
        actual = converted[0] + converted[1]
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(real) + complex_values)
        compiled = jax.jit(lambda left, right: sum(coerce_jax_values(left, right)[1]))
        np.testing.assert_array_equal(np.asarray(compiled(*values)), np.asarray(actual))
