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

"""Unit tests for PyCBC JAX FFT backend."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pycbc import scheme  # noqa: E402
from pycbc.fft import FFT, IFFT, fft, ifft  # noqa: E402
from pycbc.fft.jaxfft import jax_fft, jax_ifft, jax_irfft, jax_rfft  # noqa: E402
from pycbc.types import Array, FrequencySeries, TimeSeries  # noqa: E402
from pycbc.types.array_jax import JAXArrayData  # noqa: E402


def test_jax_fft_backend_registration():
    """Verify jax scheme selects jaxfft backend."""
    with scheme.JAXScheme():
        from pycbc.fft.backend_support import get_backend

        backend = get_backend()
        assert backend.__name__.endswith("jaxfft")


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_functional_r2c_and_c2r_fft(dtype):
    """Verify functional fft and ifft for real-to-complex and back."""
    cdtype = np.complex128 if dtype == np.float64 else np.complex64
    n = 128
    data = np.sin(np.linspace(0, 4 * np.pi, n)).astype(dtype)
    invec = TimeSeries(data, delta_t=1.0 / n)
    outvec = FrequencySeries(np.zeros(n // 2 + 1, dtype=cdtype), delta_f=1.0)

    with scheme.JAXScheme():
        fft(invec, outvec)
        ref_f = np.fft.rfft(data) * (1.0 / n)
        assert np.allclose(outvec.numpy(), ref_f, atol=1e-5)

        # Invert back to TimeSeries
        rec = TimeSeries(np.zeros(n, dtype=dtype), delta_t=1.0 / n)
        ifft(outvec, rec)
        assert np.allclose(rec.numpy(), data, atol=1e-5)


@pytest.mark.parametrize("cdtype", [np.complex128, np.complex64])
def test_functional_c2c_fft(cdtype):
    """Verify functional complex-to-complex FFT and IFFT."""
    n = 64
    cdata = (
        np.cos(np.linspace(0, 2 * np.pi, n))
        + 1j * np.sin(np.linspace(0, 2 * np.pi, n))
    ).astype(cdtype)
    invec = Array(cdata)
    outvec = Array(np.zeros(n, dtype=cdtype))

    with scheme.JAXScheme():
        fft(invec, outvec)
        ref_f = np.fft.fft(cdata)
        assert np.allclose(outvec.numpy(), ref_f, atol=1e-5)

        rec = Array(np.zeros(n, dtype=cdtype))
        ifft(outvec, rec)
        # Functional API multiplies by len(outvec), so rec matches cdata * n
        assert np.allclose(rec.numpy(), cdata * n, atol=1e-4)


def test_class_based_c2c_fft_roundtrip():
    """Verify class-based FFT and IFFT plans."""
    n = 256
    cdata = (
        np.random.randn(n) + 1j * np.random.randn(n)
    ).astype(np.complex128)
    invec = Array(cdata.copy())
    outvec = Array(np.zeros(n, dtype=np.complex128))
    rec = Array(np.zeros(n, dtype=np.complex128))

    with scheme.JAXScheme():
        plan_f = FFT(invec, outvec)
        plan_i = IFFT(outvec, rec)

        plan_f.execute()
        ref_fft = np.fft.fft(cdata)
        assert np.allclose(outvec.numpy(), ref_fft)

        plan_i.execute()
        # PyCBC class-based IFFT is unnormalized: scaled by n
        assert np.allclose(rec.numpy(), cdata * n)


def test_class_based_r2c_fft_roundtrip():
    """Verify class-based R2C FFT and C2R IFFT plans."""
    n = 512
    rdata = np.random.randn(n).astype(np.float64)
    invec = Array(rdata.copy())
    outvec = Array(np.zeros(n // 2 + 1, dtype=np.complex128))
    rec = Array(np.zeros(n, dtype=np.float64))

    with scheme.JAXScheme():
        plan_f = FFT(invec, outvec)
        plan_i = IFFT(outvec, rec)

        plan_f.execute()
        ref_rfft = np.fft.rfft(rdata)
        assert np.allclose(outvec.numpy(), ref_rfft)

        plan_i.execute()
        # Unnormalized roundtrip: scaled by n
        assert np.allclose(rec.numpy(), rdata * n)


def test_native_jax_fft_primitives():
    """Verify native JAX primitives without PyCBC wrapper containers."""
    jnp = jax.numpy
    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)

    f = jax_rfft(x)
    assert isinstance(f, jax.Array)
    assert np.allclose(np.asarray(f), np.fft.rfft(np.asarray(x)))

    inv = jax_irfft(f, n=len(x), unnormalized=True)
    assert isinstance(inv, jax.Array)
    assert np.allclose(np.asarray(inv), np.asarray(x) * len(x))

    xc = jnp.array([1.0 + 1j, 2.0 - 2j, 3.0 + 3j, 4.0 - 4j])
    fc = jax_fft(xc)
    assert np.allclose(np.asarray(fc), np.fft.fft(np.asarray(xc)))

    inv_c = jax_ifft(fc, unnormalized=False)
    assert np.allclose(np.asarray(inv_c), np.asarray(xc))


@pytest.mark.parametrize("planned", [False, True])
def test_fft_distinguishes_copy_storage_and_updates_slice(planned):
    raw = np.arange(16, dtype=np.complex64)
    with scheme.JAXScheme("cpu"):
        source = Array(raw)
        copied = source.copy()
        original = source.numpy().copy()
        if planned:
            FFT(source, copied).execute()
        else:
            fft(source, copied)
        np.testing.assert_array_equal(source.numpy(), original)
        np.testing.assert_allclose(copied.numpy(), np.fft.fft(raw), atol=1e-5)

        parent = Array(np.full(20, -7, dtype=np.complex64))
        target = parent[2:18]
        if planned:
            FFT(source[:], target).execute()
        else:
            fft(source[:], target)
        np.testing.assert_allclose(parent.numpy()[2:18], np.fft.fft(raw), atol=1e-5)
        np.testing.assert_array_equal(parent.numpy()[[0, 1, 18, 19]], -7)


@pytest.mark.parametrize("primitive", [jax_fft, jax_ifft, jax_rfft, jax_irfft])
def test_wrapped_fft_primitives_remain_jittable(primitive, monkeypatch):
    values = jax.numpy.arange(16, dtype=jax.numpy.float32)
    if primitive is not jax_rfft:
        values = values.astype(jax.numpy.complex64)
    expected = primitive(values)

    def reject_host_conversion(*args, **kwargs):
        raise AssertionError("FFT primitive converted its input to NumPy")

    monkeypatch.setattr(JAXArrayData, "__array__", reject_host_conversion)
    actual = jax.jit(lambda data: primitive(JAXArrayData(data)))(values)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('operation,plan_class', [('fft', FFT), ('ifft', IFFT)])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_native_plan_reuse_preserves_previous_result(operation, plan_class, dtype):
    """A reusable host plan must not mutate published JAX snapshots."""
    values = np.arange(64, dtype=dtype) * (1. + .5j)
    with scheme.JAXScheme('cpu', reference_operations=[operation]):
        source = Array(values)
        output = Array(np.zeros(64, dtype=dtype))
        plan = plan_class(source, output)
        plan.execute()
        retained = output.copy()
        first = output.numpy().copy()
        source[:] = Array((values[::-1] * (2. - .25j)).astype(dtype))
        plan.execute()
        assert not np.array_equal(output.numpy(), first)
        np.testing.assert_array_equal(retained.numpy(), first)
