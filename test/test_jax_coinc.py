# Copyright (C) 2026  The PyCBC Collaboration
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

"""Tests for JAX coincidence construction and clustering."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import pycbc
from pycbc import scheme
from pycbc.events import coinc
from pycbc.types import Array
from pycbc.types.array_jax import JAXArrayData, is_jax_array


if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


@pytest.mark.parametrize("method", ("python", "cython"))
def test_cluster_over_time_stays_on_jax_device(monkeypatch, method):
    """Verify clustering stays on JAX device without host fallbacks."""
    host_stat = np.array([5.0, 2.0, 7.0, 7.0, -1.0, 4.0, 8.0, 3.0])
    host_time = np.array([4.0, 0.0, 1.0, 1.25, 8.0, 4.25, 7.75, 12.0])
    expected = coinc.cluster_over_time(
        host_stat, host_time, window=0.5, method=method
    )

    def reject_host_path(*_args, **_kwargs):
        raise AssertionError("clustering used the NumPy/Cython host path")

    with scheme.JAXScheme():
        jax_stat = Array(host_stat)
        jax_time = Array(host_time)
        monkeypatch.setattr(coinc, "timecluster_cython", reject_host_path)

        actual = coinc.cluster_over_time(
            jax_stat, jax_time, window=0.5, method=method
        )

    assert isinstance(actual, Array)
    assert isinstance(actual._data, JAXArrayData)
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_cluster_over_time_raw_jax_nan_and_validation():
    """Verify NaN handling and tie-breaking matching PyCBC specifications."""
    times = jnp.array([0.0, 0.1, 0.2, 2.0], dtype=jnp.float64)
    cases = (
        ([1.0, np.nan, 3.0, 4.0], [1, 3], [2, 3]),
        ([np.nan, 4.0, 3.0, 2.0], [0, 3], [0, 3]),
        ([1.0, 2.0, np.nan, np.nan], [2, 3], [1, 3]),
    )
    for statistics, python_expected, cython_expected in cases:
        stat = jnp.array(statistics, dtype=jnp.float64)
        for method, expected in (
            ("python", python_expected),
            ("cython", cython_expected),
        ):
            actual = coinc.cluster_over_time(
                stat, times, window=0.5, method=method
            )
            assert is_jax_array(actual)
            np.testing.assert_array_equal(np.asarray(actual), expected)

    empty = coinc.cluster_over_time(
        jnp.empty(0, dtype=jnp.float64),
        jnp.empty(0, dtype=jnp.float64),
        window=0.5,
    )
    assert is_jax_array(empty)
    assert empty.size == 0


def test_time_coincidence_jax_parity():
    """Verify time coincidence matching Cython/NumPy with and without slides."""
    time1 = np.array([0.0, 0.6, 1.2, 2.5, 4.0, 6.1], dtype=np.float64)
    time2 = np.array([0.1, 0.8, 1.1, 3.0, 3.9, 6.05], dtype=np.float64)

    # Without time slides
    expected = coinc.time_coincidence(time1, time2, 0.25)
    actual = coinc.time_coincidence(
        jnp.asarray(time1), jnp.asarray(time2), 0.25
    )
    for jax_val, numpy_val in zip(actual, expected):
        assert is_jax_array(jax_val)
        np.testing.assert_array_equal(np.asarray(jax_val), numpy_val)

    # With time slides
    expected_slide = coinc.time_coincidence(time1, time2, 0.25, slide_step=1.0)
    actual_slide = coinc.time_coincidence(
        jnp.asarray(time1), jnp.asarray(time2), 0.25, slide_step=1.0
    )
    for jax_val, numpy_val in zip(actual_slide, expected_slide):
        assert is_jax_array(jax_val)
        np.testing.assert_array_equal(np.asarray(jax_val), numpy_val)


def test_cluster_coincs_jax_parity():
    """Verify 2-detector and multi-detector coincidence clustering."""
    time1 = np.array([0.0, 0.6, 1.2, 2.5], dtype=np.float64)
    time2 = np.array([0.1, 0.8, 1.1, 3.0], dtype=np.float64)
    stat = np.array([2.0, 7.0, 4.0, 8.0], dtype=np.float64)
    slide_ids = np.array([0, 0, 1, 1], dtype=np.int32)

    expected = coinc.cluster_coincs(stat, time1, time2, slide_ids, 1.0, 0.5)
    actual = coinc.cluster_coincs(
        jnp.asarray(stat),
        jnp.asarray(time1),
        jnp.asarray(time2),
        jnp.asarray(slide_ids),
        1.0,
        0.5,
    )
    assert is_jax_array(actual)
    np.testing.assert_array_equal(np.asarray(actual), expected)

    # Multi-detector
    detector_times = (
        time1 + 10.0,
        time2 + 10.0,
        np.array([-1.0, 10.7, 11.0, 12.9], dtype=np.float64),
    )
    expected_multi = coinc.cluster_coincs_multiifo(
        stat, detector_times, slide_ids, 1.0, 0.5
    )
    actual_multi = coinc.cluster_coincs_multiifo(
        jnp.asarray(stat),
        tuple(jnp.asarray(v) for v in detector_times),
        jnp.asarray(slide_ids),
        1.0,
        0.5,
    )
    assert is_jax_array(actual_multi)
    np.testing.assert_array_equal(np.asarray(actual_multi), expected_multi)
