# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

"""Exact compatibility of live singles preparation and chirp omission."""

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import pycbc
from pycbc import conversions, scheme
from pycbc.events import coinc, coinc_jax
from pycbc.events.stat import QuadratureSumStatistic
from pycbc.types import Array


if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not any(
            value.platform == "gpu" for value in jax.devices()):
        pytest.skip("CUDA device unavailable")
    with scheme.JAXScheme(request.param):
        yield request.param


def _assert_bytes(actual, expected):
    assert actual.weak_type == expected.weak_type
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def _float_edges(dtype):
    dtype = np.dtype(dtype).type
    limits = np.finfo(dtype)
    tiny = np.nextafter(dtype(0), dtype(1))
    unsigned = np.uint32 if dtype == np.float32 else np.uint64
    payload = 0x7fc01234 if dtype == np.float32 else 0x7ff8000000001234
    nan = np.array([payload], dtype=unsigned).view(dtype)[0]
    return np.array([0., -0., tiny, -tiny, limits.tiny,
                     np.nextafter(dtype(limits.tiny), dtype(0)),
                     np.nextafter(dtype(1), dtype(0)), 1.,
                     np.nextafter(dtype(1), dtype(2)),
                     np.nextafter(dtype(-2), dtype(-3)), -2.,
                     np.nextafter(dtype(-2), dtype(-1)), 2., limits.max,
                     -limits.max, np.inf, -np.inf, nan], dtype=dtype)


def _inputs(chisq_dtype, dof_dtype, size):
    rng = np.random.default_rng(8219)
    chisq = rng.uniform(.1, 30., size).astype(chisq_dtype)
    chisq[:min(size, 18)] = _float_edges(chisq_dtype)[:min(size, 18)]
    if dof_dtype == np.uint32:
        dof = rng.integers(1, 32, size, dtype=np.uint32)
        # Addition wraps in uint32 before the original JAX division promotion.
        edges = np.array([0, 1, 2, 2**24 - 1, 2**24, 2**31,
                          2**32 - 3, 2**32 - 2, 2**32 - 1], np.uint32)
    else:
        dof = rng.uniform(.1, 30., size).astype(dof_dtype)
        edges = _float_edges(dof_dtype)[::-1]
    dof[:min(size, len(edges))] = edges[:min(size, len(edges))]
    return jnp.asarray(chisq), jnp.asarray(dof)


def _eager_chisq(chisq, dof):
    """Frozen preparation primitives, before the new compiled boundary."""
    return (jnp.asarray(chisq) * jnp.asarray(dof),
            (jnp.asarray(dof) + 2) / 2)


@pytest.mark.parametrize("chisq_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dof_dtype", [np.float32, np.float64, np.uint32])
def test_live_chisq_matches_eager_dtypes_and_bits(device, chisq_dtype, dof_dtype):
    """Mixed precisions, exceptional floats and uint32 wrap retain old bits."""
    for size in (0, 1, 17, 132, 4096):
        chisq, dof = _inputs(chisq_dtype, dof_dtype, size)
        expected = _eager_chisq(chisq, dof)
        actual = coinc_jax._prepare_live_chisq(chisq, dof)
        assert coinc_jax._can_prepare_live_chisq(chisq, dof)
        for got, want in zip(actual, expected):
            _assert_bytes(got, want)


def test_live_chisq_reuses_shape_with_changed_values(device, monkeypatch):
    """Numerical values do not specialize the executable or become constants."""
    traces = []
    original = coinc_jax._prepare_live_chisq.__wrapped__

    def counted(chisq, dof):
        traces.append((chisq.shape, dof.shape))
        return original(chisq, dof)

    monkeypatch.setattr(coinc_jax, "_prepare_live_chisq", jax.jit(counted))
    for offset in (0, 1, 7):
        chisq = jnp.arange(17, dtype=jnp.float32) + offset
        dof = jnp.arange(17, dtype=jnp.uint32) + np.uint32(offset)
        for got, want in zip(coinc_jax._prepare_live_chisq(chisq, dof),
                             _eager_chisq(chisq, dof)):
            _assert_bytes(got, want)
    assert traces == [((17,), (17,))]


def test_live_chisq_guard_retains_unusual_input_fallback(device):
    """Admission inspects storage, shape and dtype without reading samples."""
    chisq, dof = _inputs(np.float32, np.uint32, 17)
    cases = [(np.asarray(chisq), dof), (Array(np.asarray(chisq)), dof),
             (chisq[0], dof), (chisq[None, :], dof), (chisq, dof[:1]),
             (chisq, dof.astype(jnp.int32)),
             (chisq.astype(jnp.complex64), dof),
             (jnp.ones(4097, jnp.float32), jnp.ones(4097, jnp.uint32))]
    for values in cases:
        assert not coinc_jax._can_prepare_live_chisq(*values)
    # Traced inputs have no concrete resident placement and must fall back.
    result = jax.jit(lambda c, d: coinc_jax._can_prepare_live_chisq(c, d))(
        chisq, dof)
    assert not bool(result)


@pytest.mark.parametrize("storage", ["resident", "host", "signed", "scalar"])
def test_custom_single_observes_exact_preparation_without_input_mutation(
        device, monkeypatch, storage):
    """Custom rankers still receive the old columns and metadata exactly once."""
    chisq, dof = _inputs(np.float32, np.uint32, 17)
    if storage == "host":
        chisq, dof = np.asarray(chisq), np.asarray(dof)
    elif storage == "signed":
        dof = dof.astype(jnp.int32)
    elif storage == "scalar":
        chisq = chisq[12]
    expected = _eager_chisq(chisq, dof)
    observations, stored, calls = [], [], []
    original_core = coinc_jax._prepare_live_chisq

    def recorded_core(*args):
        calls.append(1)
        return original_core(*args)

    monkeypatch.setattr(coinc_jax, "_prepare_live_chisq", recorded_core)
    metadata = np.array(["TaylorF2"] * 17)
    trigs = {"snr": jnp.arange(17, dtype=jnp.float32), "chisq": chisq,
             "chisq_dof": dof, "template_id": np.zeros(17, np.int32),
             "approximant": metadata}
    before = {key: (value, np.asarray(value).tobytes())
              for key, value in trigs.items()}

    def single(values):
        observations.append(values)
        assert values["ifo"] == "H1"
        assert values["approximant"] is metadata
        for key, want in zip(("chisq", "chisq_dof"), expected):
            _assert_bytes(values[key], want)
        return jnp.abs(values["snr"]).astype(jnp.float32)

    estimator = SimpleNamespace(
        singles={"H1": SimpleNamespace(add=lambda ids, values:
                                       stored.append((ids, values)))},
        stat_calculator=SimpleNamespace(single=single, single_dtype=np.float32))
    indices = coinc_jax.add_singles_to_buffer_jax(
        estimator, {"H1": trigs}, ["H1"],
        SimpleNamespace(info=lambda *args: None))
    assert calls == ([1] if storage == "resident" else [])
    assert len(observations) == len(stored) == 1
    assert indices["H1"] is trigs["template_id"]
    assert trigs.keys() == before.keys() | {"stat"}
    for key, (value, bits) in before.items():
        assert trigs[key] is value
        assert np.asarray(trigs[key]).tobytes() == bits
    _assert_bytes(trigs["stat"], jnp.abs(trigs["snr"]).astype(jnp.float32))
    for key in ("chisq", "chisq_dof"):
        _assert_bytes(stored[0][1][key], jnp.asarray(trigs[key]))


@pytest.mark.parametrize("ranking", ["snr", "newsnr"])
def test_builtin_single_ranking_keeps_preparation_rounding(device, ranking):
    """The compiled preparation leaves downstream ranking primitives unchanged."""
    chisq, dof = _inputs(np.float32, np.uint32, 132)
    snr = jnp.linspace(3., 30., 132, dtype=jnp.float32)
    ranker = QuadratureSumStatistic(ranking)
    prepared = coinc_jax._prepare_live_chisq(chisq, dof)
    eager = _eager_chisq(chisq, dof)
    actual = ranker.single({"snr": snr, "chisq": prepared[0],
                            "chisq_dof": prepared[1]})
    expected = ranker.single({"snr": snr, "chisq": eager[0],
                              "chisq_dof": eager[1]})
    _assert_bytes(actual, expected)


def test_live_preparation_empty_and_invalid_columns_keep_state(device,
                                                             monkeypatch):
    """Empty blocks bypass ranking; invalid broadcasting fails before writes."""
    writes = []

    def reject(*args, **kwargs):
        raise AssertionError("empty/invalid input reached compiled preparation")

    monkeypatch.setattr(coinc_jax, "_prepare_live_chisq", reject)
    estimator = SimpleNamespace(
        singles={"H1": SimpleNamespace(add=lambda ids, values:
                                       writes.append((ids, values)))},
        stat_calculator=SimpleNamespace(single=reject, single_dtype=np.float32))
    logger = SimpleNamespace(info=lambda *args: None)
    empty = {key: jnp.empty(0, dtype=dtype) for key, dtype in
             (("snr", jnp.float32), ("chisq", jnp.float32),
              ("chisq_dof", jnp.uint32), ("template_id", jnp.int32))}
    coinc_jax.add_singles_to_buffer_jax(estimator, {"H1": empty}, ["H1"], logger)
    assert len(writes) == 1
    assert empty["stat"].shape == (0,) and empty["stat"].dtype == jnp.float32
    invalid = {"snr": jnp.ones(3, jnp.float32),
               "chisq": jnp.ones(3, jnp.float32),
               "chisq_dof": jnp.ones(2, jnp.uint32),
               "template_id": jnp.zeros(3, jnp.int32)}
    before = {key: value for key, value in invalid.items()}
    with pytest.raises((TypeError, ValueError)):
        coinc_jax.add_singles_to_buffer_jax(estimator, {"H1": invalid},
                                           ["H1"], logger)
    assert invalid.keys() == before.keys()
    assert all(invalid[key] is value for key, value in before.items())
    assert len(writes) == 1


@pytest.mark.parametrize("fallback", ["custom", "subclass", "conversion", "eta"])
def test_chirp_fallback_preserves_scalar_kwargs_and_order(device, monkeypatch,
                                                         fallback):
    """Observers and overridden conversions retain per-trigger scalar chirps."""
    observed = []
    native = QuadratureSumStatistic("snr")
    original_rank = native.rank_stat_coinc

    def record(singles, *args, **kwargs):
        observed.append(kwargs["mchirp"])
        return original_rank(singles, *args, **kwargs)

    if fallback == "subclass":
        class ObservedStatistic(QuadratureSumStatistic):
            def rank_stat_coinc(self, *args, **kwargs):
                return record(*args, **kwargs)

        ranker = ObservedStatistic("snr")
    else:
        ranker = native
        # Custom rankers observe the scalar contract. For monkeypatched
        # conversions, retain the native ranker and observe conversion calls.
        if fallback == "custom":
            monkeypatch.setattr(ranker, "rank_stat_coinc", record)

    mass1 = jnp.array([10., np.nextafter(20., 21.), 40.], jnp.float64)
    mass2 = jnp.array([9., 12., 3.], jnp.float32)
    expected_chirps = conversions.mchirp_from_mass1_mass2(mass1, mass2)
    conversion_calls, prepared_chirps = [], []
    name = ("eta_from_mass1_mass2" if fallback == "eta" else
            "mchirp_from_mass1_mass2")
    if fallback in ("conversion", "eta"):
        original_conversion = getattr(conversions, name)

        def conversion(*args):
            value = original_conversion(*args)
            conversion_calls.append(value)
            return value

        monkeypatch.setattr(conversions, name, conversion)
        original_prepare = coinc_jax._prepare_live_match

        def record_preparation(*args):
            result = original_prepare(*args)
            prepared_chirps.append(result[3])
            return result

        monkeypatch.setattr(coinc_jax, "_prepare_live_match", record_preparation)

    estimator, trigs = _chirp_workflow(ranker, mass1, mass2, monkeypatch)
    assert not coinc_jax._can_skip_live_mchirp(ranker, trigs)
    count, output = coinc_jax.find_coincs_jax(estimator, {"H1": trigs}, ["H1"])
    assert int(count) == 6 and output["background/count"] == 6
    if fallback in ("custom", "subclass"):
        assert len(observed) == 3
        for i, value in enumerate(observed):
            assert value.ndim == 0
            _assert_bytes(value, expected_chirps[i])
    else:
        assert len(conversion_calls) == 1
        assert len(prepared_chirps) == 3
        for i, value in enumerate(prepared_chirps):
            assert value.ndim == 0
            _assert_bytes(value, expected_chirps[i])


def _chirp_workflow(ranker, mass1, mass2, monkeypatch):
    """Exercise ordered assembly without clustering obscuring the payloads."""
    trigs = {"template_id": np.array([0, 0, 0], np.int32),
             "end_time": jnp.array([1e9, 1e9 + .1, 1e9 + .2]),
             "stat": jnp.array([7., 8., 9.], jnp.float32),
             "mass1": mass1, "mass2": mass2}
    shifted = {"end_time": jnp.array([1e9 + .01, 1e9 + .02]),
               "stat": jnp.array([5., 6.], jnp.float32)}
    ring = SimpleNamespace(time=4, data=lambda template: shifted,
                           expire_vector=lambda template:
                           jnp.array([2, 3], jnp.int32))
    estimator = SimpleNamespace(
        ifos=["H1", "L1"], singles={"H1": ring, "L1": ring},
        stat_calculator=ranker, trig_stat_memory=None,
        time_window=.03, timeslide_interval=.1, coinc_window_pad=.002,
        analysis_block=1., dets={}, background_time=6.,
        return_background=True,
        coincs=coinc_jax.JAXCoincExpireBuffer(4, ["H1", "L1"], initial_size=8))
    monkeypatch.setattr(coinc, "time_coincidence", lambda *args:
                        (jnp.array([1, 0], jnp.int64),
                         jnp.zeros(2, jnp.int64), jnp.ones(2, jnp.int32)))
    monkeypatch.setattr(coinc, "cluster_coincs", lambda values, *args, **kwargs:
                        jnp.arange(values.size, dtype=jnp.int64))
    return estimator, trigs


@pytest.mark.parametrize("storage", ["resident", "host"])
def test_native_find_omits_unused_chirps_without_changing_results(device,
                                                               monkeypatch,
                                                               storage):
    """Native quadrature consumes identical ordered payloads with no chirp."""
    ranker = QuadratureSumStatistic("snr")
    masses = (np.array([10., 20., 40.]), np.array([9., 12., 3.], np.float32))
    if storage == "resident":
        masses = tuple(jnp.asarray(value) for value in masses)
    estimator, trigs = _chirp_workflow(ranker, *masses, monkeypatch)
    assert coinc_jax._can_skip_live_mchirp(ranker, trigs)
    original_prepare = coinc_jax._prepare_live_match
    chirps = []

    def recorded(*args):
        result = original_prepare(*args)
        chirps.append((args[2], result[3]))
        return result

    monkeypatch.setattr(coinc_jax, "_prepare_live_match", recorded)
    count, output = coinc_jax.find_coincs_jax(estimator, {"H1": trigs}, ["H1"])
    assert int(count) == 6 and chirps == [(None, None)] * 3
    # Run the previous eager conversion route against fresh estimator state.
    reference, ref_trigs = _chirp_workflow(ranker, *masses, monkeypatch)
    monkeypatch.setattr(coinc_jax, "_can_skip_live_mchirp", lambda *args: False)
    _, expected = coinc_jax.find_coincs_jax(reference, {"H1": ref_trigs}, ["H1"])
    assert output.keys() == expected.keys()
    for key in output:
        actual, want = np.asarray(output[key]), np.asarray(expected[key])
        assert actual.dtype == want.dtype and actual.shape == want.shape
        assert actual.tobytes() == want.tobytes(), key
    _assert_bytes(estimator.trig_stat_memory, reference.trig_stat_memory)
    _assert_bytes(estimator.coincs.buffer, reference.coincs.buffer)
    for ifo in estimator.ifos:
        _assert_bytes(estimator.coincs.timer[ifo], reference.coincs.timer[ifo])
    assert estimator.coincs.index == reference.coincs.index
    assert estimator.coincs.time == reference.coincs.time


def test_chirp_omission_guard_keeps_unusual_mass_inputs(device, monkeypatch):
    """Only native ranking and ordinary shape/type-preserving columns qualify."""
    ranker = QuadratureSumStatistic("snr")
    trigs = {"mass1": jnp.ones(3, jnp.float64),
             "mass2": jnp.ones(3, jnp.float32),
             "stat": jnp.ones(3, jnp.float32),
             "end_time": jnp.ones(3, jnp.float64)}
    assert coinc_jax._can_skip_live_mchirp(ranker, trigs)
    assert coinc_jax._can_skip_live_mchirp(
        ranker, dict(trigs, mass1=np.ones(3), mass2=np.ones(3, np.float32)))
    for mass in (np.ones(3), jnp.ones(3, jnp.int32),
                 jnp.ones(3, jnp.complex64), jnp.ones((1, 3)),
                 jnp.ones(2), Array(np.ones(3))):
        candidate = dict(trigs, mass1=mass)
        assert not coinc_jax._can_skip_live_mchirp(ranker, candidate)
    if device == "cuda":
        cpu_mass = jax.device_put(np.ones(3), jax.devices("cpu")[0])
        assert not coinc_jax._can_skip_live_mchirp(
            ranker, dict(trigs, mass1=cpu_mass))
    monkeypatch.setattr(ranker, "rank_stat_coinc", lambda *args, **kwargs: None)
    assert not coinc_jax._can_skip_live_mchirp(ranker, trigs)
