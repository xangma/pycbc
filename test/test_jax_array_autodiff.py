# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Array conversion retains derivatives under an active processing scheme."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pycbc.scheme import JAXScheme
from pycbc.types import TimeSeries
from pycbc.types.array_jax import to_jax
from pycbc.types.backend import wrap_backend_array


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("series", [False, True])
def test_active_scheme_conversion_preserves_eager_and_compiled_gradients(
        device, dtype, series):
    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    with JAXScheme(device) as context:
        values = np.array([-2., 0., .5, 3.], dtype=dtype)
        if np.issubdtype(dtype, np.complexfloating):
            values += np.array([.5j, 0j, -2j, 1j], dtype=dtype)
        inputs = jax.device_put(values, context.jax_device)

        def energy(value):
            if series:
                data = TimeSeries(wrap_backend_array(value), delta_t=.25)
                return data.squared_norm().sum()
            data = to_jax(value)
            return jnp.sum(jnp.real(data * jnp.conj(data)))

        expected = 2 * np.conj(values)
        for gradient in (jax.grad(energy), jax.jit(jax.grad(energy))):
            actual = gradient(inputs)
            assert actual.dtype == values.dtype
            assert actual.devices() == {context.jax_device}
            np.testing.assert_array_equal(np.asarray(actual), expected)
