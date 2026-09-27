# Copyright (C) 2026 The PyCBC Collaboration

"""Regression tests for the JAX LiveBatch correlation update."""

import numpy as np
import pytest
from types import SimpleNamespace

from pycbc import scheme
from pycbc.filter.matchedfilter import BatchCorrelator
from pycbc.types import Array

jax = pytest.importorskip("jax")
jnp = jax.numpy
from pycbc.filter.matchedfilter_jax import batch_correlate_execute


def test_batch_correlator_updates_sibling_views_and_preserves_tails():
    """One batched update must preserve parent storage outside output views."""
    batch, size, stride, base, tail = 4, 9, 16, 3, 7
    rng = np.random.default_rng(1234)
    xs_data = (
        rng.normal(size=(batch, size + 4)) +
        1j * rng.normal(size=(batch, size + 4))
    ).astype(np.complex64)
    y_data = (rng.normal(size=size + 5) + 1j * rng.normal(size=size + 5)).astype(
        np.complex64
    )

    with scheme.JAXScheme(device="cpu"):
        xs = [Array(row) for row in xs_data]
        y = Array(y_data)
        parent = Array(np.full(base + batch * stride + tail, 7 + 3j,
                               dtype=np.complex64))
        zs = [
            parent[base + i * stride:base + (i + 1) * stride]
            for i in range(batch)
        ]

        correlator = BatchCorrelator(xs, zs, size)
        assert correlator.x is None
        assert correlator.z is None
        assert correlator._jax_template_matrix is None
        correlator.batch_correlate_execute(y)
        expected = np.conj(xs_data[:, :size]) * y_data[None, :size]
        result = np.asarray(parent)
        np.testing.assert_allclose(
            result[base:base + batch * stride].reshape(batch, stride)[:, :size],
            expected,
                                   rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(
            result[:base], 7 + 3j)
        np.testing.assert_allclose(
            result[base:base + batch * stride].reshape(batch, stride)[:, size:],
            7 + 3j)
        np.testing.assert_allclose(result[-tail:], 7 + 3j)

        # A second call must use the current inputs rather than stale output.
        xs_data *= np.complex64(0.25 - 0.5j)
        for x, row in zip(xs, xs_data):
            x._data.set_array(jnp.asarray(row))
        y._data.set_array(jnp.asarray(y_data * (0.75 + 0.2j)))
        correlator.batch_correlate_execute(y)
        np.testing.assert_allclose(
            np.asarray(parent)[base:base + batch * stride].reshape(batch, stride)[:, :size],
            np.conj(xs_data[:, :size]) * (y_data[None, :size] * (0.75 + 0.2j)),
            rtol=2e-6, atol=2e-6,
        )


def test_immutable_live_correlator_packs_ordinary_jax_rows_once():
    """The live opt-in caches ordinary rows while generic callers stay mutable."""
    batch, size = 2, 8
    rng = np.random.default_rng(9753)
    xs_data = (rng.normal(size=(batch, size + 1)) +
               1j * rng.normal(size=(batch, size + 1))).astype(np.complex64)
    y_data = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(np.complex64)

    with scheme.JAXScheme(device="cpu"):
        xs = [Array(row) for row in xs_data]
        y = Array(y_data)
        zs = [Array(np.full(size + 2, 3 + 4j, dtype=np.complex64))
              for _ in range(batch)]
        correlator = BatchCorrelator(xs, zs, size, immutable_templates=True)
        assert correlator.x is None
        assert correlator.z is None
        assert correlator._jax_template_matrix is not None
        correlator.batch_correlate_execute(y)
        for z, row in zip(zs, xs_data):
            np.testing.assert_allclose(
                np.asarray(z)[:size], np.conj(row[:size]) * y_data,
                rtol=2e-6, atol=2e-6,
            )


def test_batch_correlator_falls_back_for_unrelated_outputs():
    """Unrelated output arrays retain the general BatchCorrelator behavior."""
    batch, size = 3, 16
    rng = np.random.default_rng(5678)
    xs_data = (
        rng.normal(size=(batch, size)) + 1j * rng.normal(size=(batch, size))
    ).astype(np.complex64)
    y_data = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(
        np.complex64
    )
    with scheme.JAXScheme(device="cpu"):
        xs = [Array(row) for row in xs_data]
        ys = Array(y_data)
        zs = [Array(np.full(size + 3, 5 + 2j, dtype=np.complex64))
              for _ in range(batch)]
        BatchCorrelator(xs, zs, size).batch_correlate_execute(ys)
        for z, row in zip(zs, xs_data):
            np.testing.assert_allclose(np.asarray(z)[:size], np.conj(row[:size]) * y_data[:size],
                                       rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(np.asarray(z)[size:], 5 + 2j)


def test_numpy_output_prefix_fallback_preserves_tail():
    """The general fallback must preserve NumPy output backing arrays."""
    class NumpyHolder:
        def __init__(self, data):
            self._data = data

        def __getitem__(self, item):
            return NumpyHolder(self._data[item])

    batch, size = 2, 5
    xs = [np.arange(size + 2, dtype=np.complex64) + 1j * (i + 1)
          for i in range(batch)]
    y = np.arange(size + 3, dtype=np.complex64) + 2j
    backing = [np.full(size + 3, 4 + 1j, dtype=np.complex64)
               for _ in range(batch)]
    zs = [NumpyHolder(z) for z in backing]
    identities = [id(z._data) for z in zs]
    control = SimpleNamespace(xs=xs, zs=zs, size=size)
    batch_correlate_execute(control, y)
    for i, (z, x) in enumerate(zip(zs, xs)):
        np.testing.assert_allclose(z._data[:size], np.conj(x[:size]) * y[:size])
        np.testing.assert_array_equal(z._data[size:], 4 + 1j)
        assert id(z._data) == identities[i]


def test_batch_correlator_full_parent_block_uses_exact_storage():
    """A parent consisting exactly of the sibling block is updated in place."""
    batch, size, stride = 2, 7, 10
    rng = np.random.default_rng(4321)
    xs_data = (rng.normal(size=(batch, size + 2)) +
               1j * rng.normal(size=(batch, size + 2))).astype(np.complex64)
    y_data = (rng.normal(size=size + 3) +
              1j * rng.normal(size=size + 3)).astype(np.complex64)
    with scheme.JAXScheme(device="cpu"):
        xs = [Array(row) for row in xs_data]
        y = Array(y_data)
        parent = Array(np.full(batch * stride, 9 + 4j, dtype=np.complex64))
        zs = [parent[i * stride:(i + 1) * stride] for i in range(batch)]
        BatchCorrelator(xs, zs, size).batch_correlate_execute(y)
        result = np.asarray(parent)[:batch * stride].reshape(batch, stride)
        np.testing.assert_allclose(result[:, :size],
                                   np.conj(xs_data[:, :size]) * y_data[:size],
                                   rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(result[:, size:], 9 + 4j)


def test_batch_correlator_uses_contiguous_bank_tensor_without_row_materialization():
    """Aligned lazy bank rows are consumed from the shared 2-D tensor."""
    batch, size, stride = 3, 6, 9
    rng = np.random.default_rng(2468)
    bank = (rng.normal(size=(batch, size + 3)) +
            1j * rng.normal(size=(batch, size + 3))).astype(np.complex64)
    y_data = (rng.normal(size=size + 2) + 1j * rng.normal(size=size + 2)).astype(
        np.complex64
    )

    class LazyRow:
        dtype = np.dtype(np.complex64)

        def __init__(self, tensor, pos):
            self._batch_tensor = tensor
            self._batch_pos = pos
            self.ptr = id(self)

        @property
        def _data(self):
            raise AssertionError("contiguous bank path materialized a row")

    with scheme.JAXScheme(device="cpu"):
        bank_jax = jnp.asarray(bank)
        xs = [LazyRow(bank_jax, i) for i in range(batch)]
        y = Array(y_data)
        parent = Array(np.full(batch * stride + 4, 5 - 2j, dtype=np.complex64))
        zs = [parent[i * stride:(i + 1) * stride] for i in range(batch)]

        BatchCorrelator(xs, zs, size).batch_correlate_execute(y)
        result = np.asarray(parent)[:batch * stride].reshape(batch, stride)
        np.testing.assert_allclose(
            result[:, :size], np.conj(bank[:, :size]) * y_data[:size],
            rtol=2e-6, atol=2e-6,
        )
        np.testing.assert_array_equal(result[:, size:], 5 - 2j)
