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
try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

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


@pytest.mark.parametrize("method", ("python", "cython"))
@pytest.mark.parametrize("jax_stat", (False, True))
def test_cluster_over_time_preserves_gps_time_precision(method, jax_stat):
    """A float32 ranking statistic must not round float64 GPS times."""
    stat = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    time = np.array([1e9, 1e9 + 0.2, 1e9 + 1.0], dtype=np.float64)
    expected = coinc.cluster_over_time(stat, time, 0.5, method=method)
    np.testing.assert_array_equal(expected, [1, 2])

    with scheme.JAXScheme():
        actual = coinc.cluster_over_time(
            jnp.asarray(stat) if jax_stat else stat,
            time, 0.5, method=method,
        )

    assert is_jax_array(actual)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("multiifo", (False, True))
@pytest.mark.parametrize("jax_stat", (False, True))
def test_cluster_coincs_preserves_mixed_input_dtypes(multiifo, jax_stat):
    """Float32 rankings must not round distinct float64 GPS times together."""
    stat = np.array([2.0, 3.0], dtype=np.float32)
    time = np.array([1e9, 1e9 + 1.0], dtype=np.float64)
    slides = np.zeros(2, dtype=np.int32)

    def cluster(ranking):
        if multiifo:
            absent = np.full(2, -1.0, dtype=np.float64)
            return coinc.cluster_coincs_multiifo(
                ranking, (time, time, absent), slides, 0.0, 0.1
            )
        return coinc.cluster_coincs(ranking, time, time, slides, 0.0, 0.1)

    expected = cluster(stat)
    np.testing.assert_array_equal(expected, [0, 1])
    with scheme.JAXScheme("cpu"):
        actual = cluster(jnp.asarray(stat) if jax_stat else stat)

    assert is_jax_array(actual)
    np.testing.assert_array_equal(np.asarray(actual), expected)


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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cluster_over_time_reuses_numeric_geometry(monkeypatch, device):
    """A changed window/data reuses the fused core; methods retain their ties."""
    from pycbc.events import coinc_jax

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    cases = [(np.array([1., np.nan, 3., 4.]), .5, "python"),
             (np.array([4., 3., 3., 2.]), .15, "python"),
             (np.array([1., 3., 3., 4.]), .5, "cython")]
    times = np.array([1e9, 1e9 + .1, 1e9 + .2, 1e9 + 2.])
    expected = [coinc.cluster_over_time(stat, times, window, method=method)
                for stat, window, method in cases]
    traces = []
    maxima = coinc_jax._cluster_window_maxima

    def counted(stat, left, right, method):
        traces.append(method)
        return maxima(stat, left, right, method)

    coinc_jax._cluster_over_time_core.clear_cache()
    monkeypatch.setattr(coinc_jax, "_cluster_window_maxima", counted)
    try:
        with scheme.JAXScheme(device):
            for (stat, window, method), want in zip(cases, expected):
                actual = coinc.cluster_over_time(jnp.asarray(stat),
                                                jnp.asarray(times), window,
                                                method=method)
                assert is_jax_array(actual)
                np.testing.assert_array_equal(np.asarray(actual), want)
        assert traces == ["python", "cython"]
    finally:
        coinc_jax._cluster_over_time_core.clear_cache()


def test_cluster_over_time_keeps_wrapper_validation(monkeypatch):
    """Validation and data-dependent empty handling precede the compiled core."""
    from pycbc.events import coinc_jax

    def reject_core(*args, **kwargs):
        raise AssertionError("invalid/empty input reached clustering core")

    monkeypatch.setattr(coinc_jax, "_cluster_over_time_core", reject_core)
    with scheme.JAXScheme("cpu"):
        stat, times = jnp.ones(2), jnp.arange(2, dtype=jnp.float64)
        for values, clock, kwargs, error in (
                (stat, times, {"method": "unknown"}, ValueError),
                (stat[None, :], times, {}, TypeError),
                (stat, times[:1], {}, ValueError),
                (stat.astype(jnp.complex64), times, {}, TypeError),
                (stat, times.astype(jnp.int64), {}, TypeError),
                (stat, times, {"argmax": np.argmin}, NotImplementedError),
                (stat, times, {"window": 0.}, ValueError)):
            with pytest.raises(error):
                coinc.cluster_over_time(values, clock,
                                        **({"window": .5} | kwargs))
        empty = coinc.cluster_over_time(stat[:0], times[:0], window=0.,
                                       argmax=np.argmin)
        assert empty.shape == (0,) and empty.dtype == jnp.int64


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


def test_time_coincidence_preserves_mixed_gps_precision():
    """A float32 JAX detector must not narrow the other detector's GPS times."""
    time1 = np.array([1e9, 1e9 + 1.0], dtype=np.float64)
    time2 = np.array([1e9], dtype=np.float32)
    expected = coinc.time_coincidence(time1, time2, 0.1)
    np.testing.assert_array_equal(expected[0], [0])

    with scheme.JAXScheme("cpu"):
        actual = coinc.time_coincidence(time1, jnp.asarray(time2), 0.1)

    for jax_val, numpy_val in zip(actual, expected):
        assert is_jax_array(jax_val)
        np.testing.assert_array_equal(np.asarray(jax_val), numpy_val)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_time_coincidence_single_matches_general_order_and_endpoints(device):
    """Singleton fusion retains folding, duplicate pairs and rounding bytes."""
    from pycbc.events import coinc_jax

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    cases = [
        ([1e9 + .75, 1e9 + .25], [1e9 + .125], .2, 1.),
        ([-.8, -.2], [-.1], .2, 1.),
        ([.2, .1], [.1], 1., 1.),
        ([.1, .1], [.1], 2., 1.),
        ([0., .5], [.25], .25, 0.),
        ([np.nextafter(0., 1.), np.nextafter(.5, 0.)], [.25], .25, 0.),
        ([np.nextafter(0., -1.), np.nextafter(.5, 1.)], [.25], .25, 0.),
        ([-.5, .5], [0.], .75, 1.),
        ([np.nextafter(.5, 0.)], [0.], 1., 1.),
        ([.2, .1], [.1], 0., 1.),
        ([.2, .1], [.9], .01, 0.),
        ([np.nan, .2], [.1], .3, 1.),
        ([np.inf, .2], [.1], .3, 1.),
        ([.2, .1], [np.nan], .3, 1.),
        (np.array([1e9, 1e9 + 1.]), np.array([1e9], "f4"), .1, 0.),
        (np.array([.5, -.5], "f4"), np.array([0.], "f4"), .75, 1.),
        (np.arange(64)[::-1] / 64., [.5], .1, 1.),
    ]
    # Preserve the independent standard CPU oracle where its double-precision
    # Cython input contract and half-away-from-zero rounding match the backend.
    cpu_expected = [coinc.time_coincidence(
        np.asarray(t1, dtype="f8"), np.asarray(t2, dtype="f8"), window, step)
        for t1, t2, window, step in cases[:8]]
    with scheme.JAXScheme(device):
        for pos, (t1, t2, window, step) in enumerate(cases):
            arr1, arr2 = jnp.asarray(t1), jnp.asarray(t2)
            dtype = jnp.result_type(arr1.dtype, arr2.dtype)
            expected = coinc_jax._time_coincidence_general(
                arr1.astype(dtype), arr2.astype(dtype), window, step)
            actual = coinc.time_coincidence(arr1, arr2, window, step)
            for got, want, output_dtype in zip(
                    actual, expected, (np.int64, np.int64, np.int32)):
                assert is_jax_array(got)
                assert got.dtype == output_dtype
                assert got.devices() == arr1.devices()
                assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
            if pos < len(cpu_expected):
                for got, want in zip(actual, cpu_expected[pos]):
                    np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("wrapped", ["none", "first", "second", "both"])
def test_time_coincidence_single_preserves_mixed_wrappers(device, wrapped):
    """A host/device mixture preserves GPS precision, placement and wrappers."""
    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    times = np.array([1e9 + 1., 1e9], dtype="f8")
    fixed = np.array([1e9], dtype="f4")
    expected = coinc.time_coincidence(times, fixed, .1)
    with scheme.JAXScheme(device):
        first = Array(times) if wrapped in ("first", "both") else times
        second = (Array(fixed) if wrapped in ("second", "both") else
                  jnp.asarray(fixed))
        actual = coinc.time_coincidence(first, second, .1)
        for got, want in zip(actual, expected):
            if wrapped != "none":
                assert isinstance(got, Array)
                assert isinstance(got._data, JAXArrayData)
                got = got._data._array
            else:
                assert is_jax_array(got)
            assert all(d.platform == ("gpu" if device == "cuda" else "cpu")
                       for d in got.devices())
            np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_time_coincidence_single_reuses_core_and_compacts_once(monkeypatch,
                                                            device):
    """Changing numeric times/windows/slides avoids eager dynamic repeats."""
    from pycbc.events import coinc_jax

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    traces, compactions = [], []
    nonzero = coinc_jax.jnp.nonzero
    compact = coinc_jax._compact_time_coincidence

    def counted_nonzero(mask, **kwargs):
        traces.append(mask.shape)
        return nonzero(mask, **kwargs)

    def counted_compact(*args, count):
        compactions.append(count)
        return compact(*args, count=count)

    def reject_repeat(*args, **kwargs):
        raise AssertionError("singleton coincidence dispatched dynamic repeat")

    coinc_jax._time_coincidence_single_core.clear_cache()
    monkeypatch.setattr(coinc_jax.jnp, "nonzero", counted_nonzero)
    monkeypatch.setattr(coinc_jax.jnp, "repeat", reject_repeat)
    monkeypatch.setattr(coinc_jax, "_compact_time_coincidence", counted_compact)
    try:
        with scheme.JAXScheme(device):
            times, fixed = jnp.array([.2, .1]), jnp.array([.1])
            for window, step in ((.01, 1.), (.2, 2.), (2., 1.),
                                 (0., 1.), (.01, 0.)):
                outputs = coinc.time_coincidence(times, fixed, window, step)
                assert outputs[0].size == compactions[-1]
                assert all(is_jax_array(value) for value in outputs)
        assert traces == [(2, 3), (2, 1)]
        assert compactions == [1, 2, 6, 0, 1]
    finally:
        coinc_jax._time_coincidence_single_core.clear_cache()


def test_time_coincidence_single_keeps_general_fallback_and_validation(
        monkeypatch):
    """The fused path leaves larger, empty and unusual inputs unchanged."""
    from pycbc.events import coinc_jax

    def reject_core(*args, **kwargs):
        raise AssertionError("ineligible coincidence reached singleton core")

    monkeypatch.setattr(coinc_jax, "_time_coincidence_single_core", reject_core)
    with scheme.JAXScheme("cpu"):
        cases = [(np.arange(65, dtype="f8"), [.1], .01, 1.),
                 ([.2, .1], [.1, .2], .1, 1.),
                 ([], [.1], .1, 1.), ([.1], [], .1, 1.),
                 ([.2, .1], [.1], .1, -1.),
                 ([.2, .1], [.1], np.nan, 1.),
                 ([.2, .1], [.1], np.inf, 1.)]
        for times, fixed, window, step in cases:
            times = jnp.asarray(times, dtype=jnp.float64)
            fixed = jnp.asarray(fixed, dtype=jnp.float64)
            expected = coinc_jax._time_coincidence_general(
                times, fixed, window, step)
            actual = coinc.time_coincidence(times, fixed, window, step)
            for got, want in zip(actual, expected):
                assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
        with pytest.raises(TypeError):
            coinc.time_coincidence(jnp.array([.2, .1]), jnp.array([.1]),
                                   -.1, 1.)
        for times, fixed in ((jnp.ones((1, 1)), jnp.ones(1)),
                             (jnp.ones(1, dtype=jnp.int64), jnp.ones(1)),
                             (jnp.ones(1), jnp.ones(1, dtype=jnp.int64))):
            with pytest.raises(TypeError):
                coinc.time_coincidence(times, fixed, .1, 1.)


def test_numpy_coincidence_inputs_use_jax_under_jax_scheme():
    """Live host buffers enter the JAX backend at the scheme boundary."""
    time1 = np.array([0.0, 0.6, 1.2, 2.5, 4.0, 6.1], dtype=np.float64)
    time2 = np.array([0.1, 0.8, 1.1, 3.0, 3.9, 6.05], dtype=np.float64)
    stat = np.array([2.0, 7.0, 4.0, 8.0, 3.0, 6.0], dtype=np.float64)
    slide_ids = np.array([0, 0, 1, 1, -1, 2], dtype=np.int32)

    expected_coinc = coinc.time_coincidence(
        time1, time2, 0.25, slide_step=1.0
    )
    expected_cluster = coinc.cluster_coincs(
        stat, time1, time2, slide_ids, 1.0, 0.5
    )
    expected_multi = coinc.cluster_coincs_multiifo(
        stat,
        (time1, time2, time1 + 10.0),
        slide_ids,
        1.0,
        0.5,
    )
    expected_over_time = coinc.cluster_over_time(
        stat, time1, 0.5, method="python"
    )

    with scheme.JAXScheme():
        actual_coinc = coinc.time_coincidence(
            time1, time2, 0.25, slide_step=1.0
        )
        actual_cluster = coinc.cluster_coincs(
            stat, time1, time2, slide_ids, 1.0, 0.5
        )
        actual_multi = coinc.cluster_coincs_multiifo(
            stat,
            (time1, time2, time1 + 10.0),
            slide_ids,
            1.0,
            0.5,
        )
        actual_over_time = coinc.cluster_over_time(
            stat, time1, 0.5, method="python"
        )

    for actual, expected in zip(actual_coinc, expected_coinc):
        assert is_jax_array(actual)
        np.testing.assert_array_equal(np.asarray(actual), expected)
    for actual, expected in (
        (actual_cluster, expected_cluster),
        (actual_multi, expected_multi),
        (actual_over_time, expected_over_time),
    ):
        assert is_jax_array(actual)
        np.testing.assert_array_equal(np.asarray(actual), expected)


def test_jax_expiring_background_buffer_is_device_resident():
    """The live background statistic buffer performs updates with JAX arrays."""
    from pycbc.events.coinc_jax import JAXCoincExpireBuffer

    with scheme.JAXScheme():
        buffer = JAXCoincExpireBuffer(4, ["H1", "L1"], initial_size=2)
        buffer.add(
            np.array([2.0, 5.0]),
            {"H1": np.array([0, 0]), "L1": np.array([0, 0])},
            ["H1", "L1"],
        )
        assert is_jax_array(buffer.data)
    np.testing.assert_array_equal(np.asarray(buffer.data), [2.0, 5.0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ifos", [("H1", "L1"), ("H1", "L1", "V1")])
def test_jax_background_fused_updates_preserve_complete_eager_state(
        monkeypatch, device, dtype, ifos):
    """The unchanged eager path is the oracle for growth, pruning and tails."""
    from pycbc.events import coinc_jax

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    tiny = np.nextafter(dtype(0), dtype(1))
    # Selection/copy operations must retain signed zero and NaN payload bits.
    bits = np.array([0x7fc12345], dtype=np.uint32).view(np.float32)[0]
    seed = np.array([0., -0., tiny, bits], dtype=dtype)
    with scheme.JAXScheme(device):
        actual = coinc_jax.JAXCoincExpireBuffer(2, ifos, initial_size=4,
                                               dtype=dtype)
        expected = coinc_jax.JAXCoincExpireBuffer(2, ifos, initial_size=4,
                                                 dtype=dtype)
        compiled = coinc_jax._append_expire_coinc_buffer
        calls = []

        def recorded(*args, **kwargs):
            calls.append(kwargs["work_size"])
            return compiled(*args, **kwargs)

        monkeypatch.setattr(coinc_jax, "_append_expire_coinc_buffer", recorded)

        def assert_state():
            assert actual.index == expected.index
            assert actual.time == expected.time
            assert actual.nbytes == expected.nbytes
            for got, want in ((actual.buffer, expected.buffer),
                              *((actual.timer[ifo], expected.timer[ifo])
                                for ifo in ifos)):
                got, want = np.asarray(got), np.asarray(want)
                assert got.shape == want.shape and got.dtype == want.dtype
                assert got.tobytes() == want.tobytes()

        def add(values, clocks, active):
            values = jnp.asarray(values, dtype=dtype)
            times = {ifo: jnp.asarray(clocks, dtype=jnp.int64) for ifo in ifos}
            snapshot = (actual.buffer, actual.data, *actual.timer.values())
            before = tuple(np.asarray(value).tobytes() for value in snapshot)
            with monkeypatch.context() as reference:
                reference.setattr(coinc_jax, "_can_append_expire_coinc_buffer",
                                  lambda *_args: False)
                expected.add(values, times, active)
            actual.add(values, times, active)
            assert_state()
            assert tuple(np.asarray(value).tobytes() for value in snapshot) == before

        add(seed, [0, 0, 0, 0], [])
        assert actual.buffer.size == 4  # Exact capacity does not grow on JAX.
        actual.remove(2)
        expected.remove(2)
        assert_state()
        add([tiny], [1], ["H1"])
        add([], [], ["H1"])
        assert actual.index == 3  # Expiration equality is inclusive.
        add([], [], ["H1"])
        assert actual.index == 1
        add([7., 8., -0., tiny], [-1, 1, 2, 3], list(ifos))
        assert actual.buffer.size == 8
        add([10., 11.], [4, 5], ["H1", "H1"])
        add([], [], ["L1"])
        add([12.], 10, [])  # Scalar timer broadcasting, without expiration.
        actual.expiration = expected.expiration = -1
        add([], [], list(ifos))
        actual.remove(0)
        expected.remove(0)
        with monkeypatch.context() as reference:
            reference.setattr(coinc_jax, "_can_append_expire_coinc_buffer",
                              lambda *_args: False)
            expected.increment(list(ifos))
        actual.increment(list(ifos))
        assert_state()
        assert calls and max(calls) <= 8


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_background_fused_updates_match_cpu_expiration(device):
    """Compare active outputs and clocks with the independent CPU reference."""
    from pycbc.events import coinc_jax

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    native = coinc.CoincExpireBuffer(2, ["H1", "L1"], initial_size=64)
    with scheme.JAXScheme(device):
        buffer = coinc_jax.JAXCoincExpireBuffer(2, ["H1", "L1"], initial_size=64)
        for block in range(8):
            values = np.array([block, block + .5], dtype=np.float32)
            clocks = {"H1": np.array([block - 2, block], np.int32),
                      "L1": np.array([block, block - 1], np.int32)}
            active = ["H1"] if block % 2 else ["H1", "L1"]
            native.add(values, clocks, active)
            buffer.add(jnp.asarray(values),
                       {ifo: jnp.asarray(value) for ifo, value in clocks.items()},
                       active)
            assert buffer.index == native.index and buffer.time == native.time
            assert np.asarray(buffer.data).tobytes() == native.data.tobytes()
            for ifo in native.ifos:
                np.testing.assert_array_equal(
                    np.asarray(buffer.timer[ifo][:buffer.index]),
                    native.timer[ifo][:native.index])
            assert buffer.num_greater(3.) == native.num_greater(3.)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_background_fused_updates_keep_statistics_resident(monkeypatch,
                                                              device):
    """A production-capacity update reads only the scalar survivor count."""
    from pycbc.events import coinc_jax

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    with scheme.JAXScheme(device):
        buffer = coinc_jax.JAXCoincExpireBuffer(2, ["H1", "L1"])
        values = jnp.arange(6, dtype=jnp.float32)
        times = {ifo: jnp.arange(6, dtype=jnp.int32) for ifo in buffer.ifos}
        array_type = type(values)
        original = array_type.__array__

        def reject_vector_copy(value, *args, **kwargs):
            if value.ndim:
                raise AssertionError("scientific column copied to host")
            return original(value, *args, **kwargs)

        with monkeypatch.context() as device_only:
            device_only.setattr(array_type, "__array__", reject_vector_copy)
            assert coinc_jax._can_append_expire_coinc_buffer(
                buffer, values, times, ["H1", "L1"])
            buffer.add(values, times, ["H1", "L1"])
            buffer.increment(["H1"])
            buffer.increment(["H1", "L1"])
        assert buffer.index == 5
        assert buffer.buffer.size == 2**20
        assert buffer.timer["H1"].dtype == jnp.int32
        assert buffer.buffer.devices() == values.devices()
        np.testing.assert_array_equal(np.asarray(buffer.data), [1, 2, 3, 4, 5])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_background_prefix_core_reuses_dynamic_clock_and_index(monkeypatch,
                                                                  device):
    """Changing append positions and survivor counts reuses one executable."""
    from pycbc.events import coinc_jax

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    traces = []
    nonzero = coinc_jax.jnp.nonzero

    def counted_nonzero(mask, **kwargs):
        traces.append(mask.shape)
        return nonzero(mask, **kwargs)

    coinc_jax._append_expire_coinc_buffer.clear_cache()
    monkeypatch.setattr(coinc_jax.jnp, "nonzero", counted_nonzero)
    try:
        with scheme.JAXScheme(device):
            buffer = coinc_jax.JAXCoincExpireBuffer(2, ["H1", "L1"],
                                                   initial_size=64)
            values = jnp.array([3., 5.], jnp.float32)
            counts = []
            for index, clock in ((3, 0), (4, 1), (5, 4), (3, 8)):
                buffer.index = index
                buffer.time = {ifo: clock for ifo in buffer.ifos}
                times = {ifo: jnp.array([clock, clock + 1], jnp.int32)
                         for ifo in buffer.ifos}
                buffer.add(values, times, ["H1", "L1"])
                counts.append(buffer.index)
        assert traces == [(8,)]
        assert len(set(counts)) > 1
    finally:
        coinc_jax._append_expire_coinc_buffer.clear_cache()


@pytest.mark.parametrize("case", ["large", "negative_index", "float_expiration",
                                 "large_threshold", "host_times", "missing_times",
                                 "fractional_clock", "unconfigured_ifo",
                                 "generator", "array_ifos", "matrix_values"])
def test_jax_background_unusual_inputs_retain_eager_behavior(monkeypatch, case):
    """Fast-path admission preserves existing fallback output/error state."""
    from pycbc.events import coinc_jax

    with scheme.JAXScheme():
        buffers = [coinc_jax.JAXCoincExpireBuffer(2, ["H1", "L1"],
                                                 initial_size=8192)
                   for _ in range(2)]
        values = jnp.array([1., 2.], jnp.float32)
        times = {ifo: jnp.array([0, 1], jnp.int32) for ifo in buffers[0].ifos}
        if case == "large":
            for buffer in buffers:
                buffer.index = 4096
        elif case == "negative_index":
            for buffer in buffers:
                buffer.remove(1)
        elif case == "float_expiration":
            for buffer in buffers:
                buffer.expiration = .5
        elif case == "large_threshold":
            for buffer in buffers:
                buffer.expiration = 2**40
        elif case == "host_times":
            times = {ifo: np.asarray(value) for ifo, value in times.items()}
        elif case == "missing_times":
            del times["L1"]
        elif case == "fractional_clock":
            for buffer in buffers:
                buffer.time["H1"] = 2.5
        elif case == "unconfigured_ifo":
            for buffer in buffers:
                buffer.time["V1"] = 0
        elif case == "matrix_values":
            values = values.reshape(1, 2)

        def active():
            if case == "unconfigured_ifo":
                return ["V1"]
            if case == "generator":
                return iter(["H1", "L1"])
            if case == "array_ifos":
                return np.array(["H1", "L1"])
            return ["H1", "L1"]

        assert not coinc_jax._can_append_expire_coinc_buffer(
            buffers[0], values, times, active())
        errors = []
        for index, buffer in enumerate(buffers):
            with monkeypatch.context() as reference:
                if not index:
                    reference.setattr(coinc_jax, "_can_append_expire_coinc_buffer",
                                      lambda *_args: False)
                try:
                    buffer.add(values, times, active())
                except Exception as error:  # Compare legacy failure state too.
                    errors.append((type(error), str(error)))
                else:
                    errors.append(None)
        assert errors[0] == errors[1]
        assert buffers[0].time == buffers[1].time
        assert buffers[0].index == buffers[1].index
        for got, want in ((buffers[0].buffer, buffers[1].buffer),
                          *((buffers[0].timer[ifo], buffers[1].timer[ifo])
                            for ifo in buffers[0].ifos)):
            assert np.asarray(got).tobytes() == np.asarray(want).tobytes()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_singles_buffer_keeps_approximant_metadata_host_side(device):
    """Live trigger metadata may be strings beside device numeric columns."""
    from pycbc.events.coinc_jax import JAXMultiRingBuffer

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    values = {"snr": np.array([3.0, 4.0]),
              "template_id": np.array([0, 1], dtype=np.int32),
              "approximant": np.array(["TaylorF2", "TaylorF2"])}
    with scheme.JAXScheme(device):
        buffer = JAXMultiRingBuffer(2, 2)
        buffer.add(values["template_id"], values)
        buffer.add(np.array([0]), {"snr": np.array([5.0]),
                                   "template_id": np.array([0], dtype=np.int32),
                                   "approximant": np.array(["TaylorF2"])})
        stored = buffer.data(0)
        # An empty block advances expiration without converting metadata.
        buffer.add(np.array([], dtype=np.int32),
                   {key: value[:0] for key, value in values.items()})
        expired = buffer.data(0)
        np.testing.assert_array_equal(np.asarray(expired["snr"]), [5.0])
        np.testing.assert_array_equal(expired["approximant"], ["TaylorF2"])
        # Expiration compacts the backing columns as well as the public view.
        assert buffer.buffer[0]["snr"].size == 1
        assert buffer.buffer_expire[0].size == 1
        assert buffer.valid_ends[0] == 1
        assert all(d.platform == ("gpu" if device == "cuda" else "cpu")
                   for d in stored["snr"].devices())
    assert is_jax_array(stored["snr"])
    assert is_jax_array(stored["template_id"])
    assert np.asarray(stored["snr"]).tolist() == [3.0, 5.0]
    assert np.asarray(stored["approximant"]).tolist() == ["TaylorF2", "TaylorF2"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("max_time", [0, 1, 2])
def test_jax_singles_grouped_updates_match_native_rings(device, max_time):
    """Duplicates, backout and post-add expiration preserve ordered columns."""
    from pycbc.events.coinc_jax import JAXMultiRingBuffer

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    dtype = np.dtype([("snr", "f4"), ("end_time", "f8"),
                      ("event_id", "u8"), ("metadata", object)])
    native = coinc.MultiRingBuffer(3, max_time, dtype)
    with scheme.JAXScheme(device):
        actual = JAXMultiRingBuffer(3, max_time)
        for block, ids in enumerate(([1, 0, 1, 2, 1, 0], [2, 1, 2, 0],
                                     [], [1, 2, 1])):
            ids = np.asarray(ids, dtype=np.int32)
            rows = np.empty(len(ids), dtype=dtype)
            rows["snr"] = np.resize(np.array([-0., 0., np.nan], "f4"), len(ids))
            rows["end_time"] = 1e9 + (block * 10 + np.arange(len(ids))) / 1024
            rows["event_id"] = 2**53 + 1 + np.arange(len(ids), dtype=np.uint64)
            rows["metadata"] = [np.nan if i % 3 == 2 else f"block{block}/row{i}"
                                for i in range(len(ids))]
            values = {key: rows[key] for key in dtype.names}
            # Both native host columns and already resident columns are accepted.
            if block % 2:
                values["snr"] = jnp.asarray(values["snr"])
            native.add(ids, rows)
            actual.add(ids, values)
            if block == 3:
                # Repeated indices remove repeated rows, without rewinding time.
                native.discard_last(ids)
                actual.discard_last(ids)
            assert actual.time == native.time
            assert actual.filled_time == native.filled_time
            for ring in range(3):
                expected = native.data(ring)
                got = actual.data(ring)
                assert actual.valid_ends[ring] == len(expected)
                assert got.keys() == set(dtype.names)
                for key in dtype.names:
                    value = np.asarray(got[key])
                    assert value.dtype == expected[key].dtype
                    assert value.shape == expected[key].shape
                    if key == "metadata":
                        np.testing.assert_equal(value.tolist(),
                                                expected[key].tolist())
                        assert isinstance(got[key], np.ndarray)
                    else:
                        # Byte comparison also checks signed zero and large IDs.
                        assert value.tobytes() == expected[key].tobytes()
                        assert is_jax_array(got[key])
                np.testing.assert_array_equal(
                    np.asarray(actual.expire_vector(ring)),
                    native.expire_vector(ring))


def test_jax_singles_grouped_dispatch_and_host_expiration(monkeypatch):
    """Appending resident rings shares dispatch; expiry reads no device mask."""
    from pycbc.events import coinc_jax

    with scheme.JAXScheme("cpu"):
        buffer = coinc_jax.JAXMultiRingBuffer(1024, 2)
        assert all(value is buffer.buffer_expire[0]
                   for value in buffer.buffer_expire)
        columns = {"snr": np.arange(9, dtype=np.float32),
                   "end_time": np.arange(9, dtype=np.float64),
                   "metadata": np.array(["TaylorF2"] * 9)}
        ids = np.tile(np.arange(3), 3)
        buffer.add(ids, columns)
        calls = []
        append = coinc_jax._append_selected_singles_groups

        def counted(selected, previous, expiries, clock):
            calls.append((len(selected), tuple(len(row) for row in previous),
                          tuple(row[0].shape for row in selected)))
            return append(selected, previous, expiries, clock)

        def reject_serial(*_args):
            raise AssertionError("eligible ring group dispatched per-ring append")

        def reject_device_mask(*args, **kwargs):
            raise AssertionError("expiration read a device mask")

        monkeypatch.setattr(coinc_jax, "_append_selected_singles_groups", counted)
        monkeypatch.setattr(coinc_jax, "_append_singles_columns", reject_serial)
        monkeypatch.setattr(coinc_jax.jnp, "all", reject_device_mask)
        buffer.add(ids, columns)
        # Both numeric columns and expiry share one call across all three rings.
        assert calls == [(3, (2, 2, 2), ((3,), (3,), (3,)))]
        buffer.add([], {})
        for ring in range(3):
            np.testing.assert_array_equal(np.asarray(buffer.data(ring)["snr"]),
                                          columns["snr"][ids == ring])
        buffer.add([], {})
        assert all(len(buffer.data(ring)["snr"]) == 0 for ring in range(3))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("x64", [False, True])
def test_jax_singles_appends_numeric_columns_together(monkeypatch, device, x64):
    """One resident group append retains typed rows and metadata order."""
    from pycbc.events import coinc_jax

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    tiny = np.nextafter(np.float32(0), np.float32(1))
    host = {"snr": np.array([-0., tiny, -tiny, np.nan], dtype=np.float32),
            "metadata": np.array(["a", np.nan, "b", "c"], dtype=object),
            "end_time": np.array([1e9 + i / 1024 for i in range(4)]),
            "event_id": np.array([2**53 + i for i in range(4)], dtype=np.uint64),
            "vector": np.arange(8, dtype=np.int32).reshape(4, 2)}
    ids = np.array([2, 0, 2, 1])
    calls = []
    append = coinc_jax._append_selected_singles_groups

    def counted(selected, previous, expiries, clock):
        calls.append((len(selected), tuple(len(row) for row in selected),
                      tuple(row[0].shape for row in selected)))
        return append(selected, previous, expiries, clock)

    def reject_serial(*_args):
        raise AssertionError("eligible ring group dispatched per-ring append")

    with scheme.JAXScheme(device), enable_x64(x64):
        columns = {key: (value if key == "metadata" else jnp.asarray(value))
                   for key, value in host.items()}
        expected = {
            ring: {key: (value[np.flatnonzero(ids == ring)]
                         if key == "metadata" else
                         value[jnp.asarray(np.flatnonzero(ids == ring))])
                   for key, value in columns.items()}
            for ring in (2, 0, 1)
        }
        monkeypatch.setattr(coinc_jax, "_append_selected_singles_groups", counted)
        monkeypatch.setattr(coinc_jax, "_append_singles_columns", reject_serial)
        actual = coinc_jax.JAXMultiRingBuffer(3, 2)
        actual.add(ids, columns)
        assert calls == [(3, (4, 4, 4), ((2,), (1,), (1,)))]
        for ring, want in expected.items():
            row = actual.data(ring)
            assert list(row) == list(columns)
            for key, value in row.items():
                if key == "metadata":
                    assert isinstance(value, np.ndarray)
                    np.testing.assert_equal(value.tolist(), want[key].tolist())
                else:
                    got, reference = np.asarray(value), np.asarray(want[key])
                    assert got.shape == reference.shape
                    assert got.dtype == reference.dtype
                    assert got.tobytes() == reference.tobytes()
                    assert value.devices() == columns[key].devices()


def test_jax_singles_malformed_columns_keep_gather_errors(monkeypatch):
    """Ineligible geometry retains the original per-column error ordering."""
    from pycbc.events import coinc_jax

    def reject_gather(*args, **kwargs):
        raise AssertionError("malformed columns reached fused selection")

    monkeypatch.setattr(coinc_jax, "_gather_singles_columns", reject_gather)
    with scheme.JAXScheme("cpu"):
        for columns in ({"metadata": np.array("TaylorF2"), "snr": jnp.ones(2)},
                        {"snr": jnp.array(1.), "metadata": np.array(["a", "b"])},
                        {"metadata": np.array(["a"]), "snr": jnp.ones(2)}):
            indices = jnp.asarray(np.array([0, 1]))
            with pytest.raises(IndexError) as reference:
                {key: value[indices] if is_jax_array(value) else value[[0, 1]]
                 for key, value in columns.items()}
            with pytest.raises(type(reference.value)):
                coinc_jax.JAXMultiRingBuffer(2, 2).add([0, 0], columns)


def test_jax_singles_mixed_devices_align_to_selected_device(monkeypatch):
    """Mixed-device columns are admitted together without changing inputs."""
    from pycbc.events import coinc_jax

    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("CUDA device unavailable")
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]

    gather = coinc_jax._gather_singles_columns
    gather_calls = []

    def record_gather(*args, **kwargs):
        gather_calls.append(True)
        return gather(*args, **kwargs)

    with scheme.JAXScheme("cuda"):
        expected = {"snr": np.array([-0., np.nan], "f4"),
                    "end_time": np.array([1e9, 1e9 + .5])}
        values = {"snr": jax.device_put(expected["snr"], gpu),
                  "end_time": jax.device_put(expected["end_time"], cpu)}
        original_devices = {key: value.devices() for key, value in values.items()}
        monkeypatch.setattr(coinc_jax, "_gather_singles_columns", record_gather)
        actual = coinc_jax.JAXMultiRingBuffer(1, 2)
        actual.add([0, 0], values)
        assert gather_calls
        for key, value in actual.data(0).items():
            assert value.devices() == {scheme.mgr.state.jax_device}
            got, want = np.asarray(value), expected[key]
            assert got.dtype == want.dtype and got.shape == want.shape
            assert got.tobytes() == want.tobytes()
            assert values[key].devices() == original_devices[key]
            assert np.asarray(values[key]).tobytes() == want.tobytes()


def test_jax_singles_traced_columns_keep_original_selection(monkeypatch):
    """Traced inputs bypass the concrete-device selection guard."""
    from pycbc.events import coinc_jax

    def reject_gather(*args, **kwargs):
        raise AssertionError("traced columns reached concrete-device selection")

    monkeypatch.setattr(coinc_jax, "_gather_singles_columns", reject_gather)

    @jax.jit
    def select(values):
        buffer = coinc_jax.JAXMultiRingBuffer(2, 2)
        buffer.add([1, 0, 1], {"snr": values})
        return buffer.data(1)["snr"], buffer.data(0)["snr"]

    with scheme.JAXScheme("cpu"):
        values = jnp.asarray(np.array([-0., np.nan, 3.], "f4"))
        got = select(values)
        expected = (values[jnp.array([0, 2])], values[jnp.array([1])])
        for value, want in zip(got, expected):
            assert np.asarray(value).tobytes() == np.asarray(want).tobytes()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("x64", [False, True])
def test_jax_singles_grouped_dtype_promotion_matches_scalar_append(device, x64):
    """Grouping retains prior JAX casts, including mixed widening/subnormals."""
    from pycbc.events.coinc_jax import JAXMultiRingBuffer

    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    tiny = np.nextafter(np.float32(0), np.float32(1))
    blocks = [{"value": np.array([-0., np.nan], dtype=np.float64),
               "id": np.array([2**53 + 1, 2**53 + 3], dtype=np.int64)},
              {"value": np.array([tiny, -tiny], dtype=np.float32),
               "id": np.array([2**53 + 5, 2**53 + 7], dtype=np.uint64)}]
    with scheme.JAXScheme(device), enable_x64(x64):
        buffer = JAXMultiRingBuffer(1, 10)
        expected = {}
        for values in blocks:
            # Prior ring insertion cast and concatenated each row separately.
            for pos in range(2):
                for key, value in values.items():
                    new = jnp.asarray(value[pos:pos + 1])
                    expected[key] = (jnp.concatenate((expected[key], new))
                                     if key in expected else new)
            buffer.add(np.array([0, 0]), values)
        for key, value in buffer.data(0).items():
            got, want = np.asarray(value), np.asarray(expected[key])
            assert got.shape == want.shape
            assert got.dtype == want.dtype
            assert got.tobytes() == want.tobytes()


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


