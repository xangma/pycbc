# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under the GNU General Public License, version 3 or later.

"""Batched PSD refresh decisions against the existing StrainBuffer oracle."""

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pycbc import scheme
from pycbc.psd import estimate_jax
from pycbc.strain import live_psd_jax as batch
from pycbc.strain import strain_jax
from pycbc.strain.strain import StrainBuffer
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.types.array_jax import _ensure_x64, to_jax


def _devices():
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("cuda:0")
    except RuntimeError:
        pass
    return devices


def _buffer(old=100.0, skip=0.01, abort=0.2, detector="H1", dtype=np.float32):
    strain = TimeSeries(np.arange(512, dtype=dtype), delta_t=1.0 / 64)
    psd = None if old is None else FrequencySeries(np.ones(33, np.float32), delta_f=1.0)
    if psd is not None:
        psd.dist = old
    return SimpleNamespace(strain=strain, sample_rate=64, psd_segment_length=1.0,
                           psd_samples=5, low_frequency_cutoff=5.0, detector=detector,
                           psd=psd, psd_recalculate_difference=skip,
                           psd_abort_difference=abort, psds={"cached": object()},
                           segments={"cached": object()})


def _controlled(monkeypatch, horizons):
    supplied = iter(horizons)
    ledger = {"welch": [], "horizon": []}

    def welch(values, **kwargs):
        ledger["welch"].append((values, kwargs))
        return FrequencySeries(np.linspace(1, 2, 33, dtype=np.float64), delta_f=1.0)

    def horizon(psd, lower_frequency_cutoff):
        ledger["horizon"].append(psd)
        distance, negative = next(supplied)
        return jnp.asarray(distance, jnp.float64), jnp.asarray(negative)

    monkeypatch.setattr(estimate_jax, "welch_jax", welch)
    monkeypatch.setattr(batch, "welch_jax", welch)
    monkeypatch.setattr(strain_jax, "psd_horizon_payload_jax", horizon)
    monkeypatch.setattr(batch, "psd_horizon_payload_jax", horizon)
    return ledger


def _outcome(function):
    try:
        return function(), None
    except Exception as error:
        return None, (type(error), str(error))


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("old,new,skip,abort,negative", [
    (None, 100.0, 0.01, 0.2, False),
    (100.0, 100.5, 0.01, 0.2, False),
    (100.0, 101.0, 0.01, 0.2, False),
    (100.0, 110.0, 0.01, 0.2, False),
    (100.0, 130.0, 0.01, 0.2, False),
    (100.0, 130.0, 0.5, 0.2, False),
    (100.0, 130.0, 0.0, 0.0, False),
    (100.0, np.nan, 0.01, 0.2, False),
    (100.0, np.inf, 0.01, 0.2, False),
    (np.nan, 100.0, 0.01, 0.2, False),
    (np.inf, np.inf, 0.01, 0.2, False),
    (0.0, 100.0, 0.01, 0.2, False),
    (np.float64(0), 100.0, 0.01, 0.2, False),
    (100.0, np.nan, 0.01, 0.2, True),
])
def test_batch_matches_original_decisions_and_cache_identities(
        monkeypatch, dev_name, old, new, skip, abort, negative):
    _ensure_x64()
    with scheme.JAXScheme(dev_name), np.errstate(divide="ignore", invalid="ignore"):
        oracle = _buffer(old, skip, abort)
        original = (oracle.psd, oracle.psds, oracle.segments)
        expected_ledger = _controlled(monkeypatch, [(new, negative)])
        expected = _outcome(lambda: StrainBuffer.recalculate_psd(oracle))
        changed = tuple(after is not before for after, before in zip(
            (oracle.psd, oracle.psds, oracle.segments), original))

        actual = _buffer(old, skip, abort)
        original = (actual.psd, actual.psds, actual.segments)
        actual_ledger = _controlled(monkeypatch, [(new, negative)])
        outcome = _outcome(lambda: batch.recalculate_psds_jax([actual])[0])
        assert outcome == expected
        assert tuple(after is not before for after, before in zip(
            (actual.psd, actual.psds, actual.segments), original)) == changed
        if changed[0]:
            assert actual.psd is actual_ledger["horizon"][0]
            assert actual.psds == {} and actual.segments == {}
        assert actual_ledger["horizon"][0].dtype == np.float32
        for ledger in (actual_ledger, expected_ledger):
            values, kwargs = ledger["welch"][0]
            assert len(values) == 192
            assert kwargs["seg_len"] == 64 and kwargs["seg_stride"] == 32
            assert not kwargs.get("wide_fft", False)
        if outcome[1] is None:
            for bounds in ((0.0, 200.0), (new, new), (0.0, np.inf)):
                assert bool(StrainBuffer.check_psd_dist(actual, *bounds)) == bool(
                    StrainBuffer.check_psd_dist(oracle, *bounds))


@pytest.mark.parametrize("dev_name", _devices())
def test_all_submissions_precede_one_collection_and_lazy_decisions(monkeypatch, dev_name):
    _ensure_x64()
    with scheme.JAXScheme(dev_name):
        buffers = [_buffer(detector=ifo) for ifo in ("H1", "L1", "V1")]
        original = [(value.psd, value.psds, value.segments) for value in buffers]
        ledger = _controlled(monkeypatch, [(100.5, False), (130.0, False), (110.0, False)])
        device_get = jax.device_get
        collecting = False
        calls = []
        array_type = type(to_jax(buffers[0].strain))
        original_array = array_type.__array__

        def reject(*args, **kwargs):
            raise AssertionError("PSD preparation collected a scientific device value")

        def guarded_array(self, *args, **kwargs):
            if not collecting:
                reject()
            return original_array(self, *args, **kwargs)

        def get(payloads):
            nonlocal collecting
            assert len(ledger["welch"]) == len(ledger["horizon"]) == 3
            assert len(payloads) == 3
            calls.append(payloads)
            collecting = True
            try:
                return device_get(payloads)
            finally:
                collecting = False

        with monkeypatch.context() as patch:
            patch.setattr(jax, "device_get", get)
            patch.setattr(array_type, "__array__", guarded_array)
            for name in ("__float__", "__int__", "__bool__"):
                patch.setattr(array_type, name, reject)
            pending = batch.prepare_psd_updates_jax(buffers)
        assert len(calls) == 1
        for buffer, saved in zip(buffers, original):
            assert all(left is right for left, right in zip(
                (buffer.psd, buffer.psds, buffer.segments), saved))
        assert tuple(update.result() for update in pending) == (True, False, True)
        caches = buffers[1].psds
        assert pending[1].result() is False and buffers[1].psds is caches
        assert buffers[0].psd is original[0][0]


def test_preparation_and_domain_errors_keep_detector_turn_priority(monkeypatch):
    _ensure_x64()
    with scheme.JAXScheme("cpu"):
        buffers = [_buffer(detector=ifo) for ifo in ("H1", "L1", "V1")]
        ledger = _controlled(monkeypatch, [(np.nan, True), (110.0, False)])
        welch = batch.welch_jax
        count = 0

        def fail_second(values, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("L1 PSD preparation failed")
            return welch(values, **kwargs)

        monkeypatch.setattr(batch, "welch_jax", fail_second)
        pending = batch.prepare_psd_updates_jax(buffers)
        assert len(ledger["welch"]) == 2
        # No error is exposed before the original detector/filter turn.
        with pytest.raises(ValueError, match="math domain error"):
            pending[0].result()
        with pytest.raises(RuntimeError, match="L1 PSD preparation failed"):
            pending[1].result()
        assert pending[2].result() is True


def test_grouped_readback_failure_is_identified_per_detector(monkeypatch):
    _ensure_x64()
    with scheme.JAXScheme("cpu"):
        buffers = [_buffer(detector=ifo) for ifo in ("H1", "L1")]
        _controlled(monkeypatch, [(110.0, False), (110.0, False)])
        device_get = jax.device_get
        calls = []

        def get(payload):
            calls.append(payload)
            if isinstance(payload, list) or len(calls) == 3:
                raise RuntimeError("L1 horizon collection failed")
            return device_get(payload)

        monkeypatch.setattr(jax, "device_get", get)
        pending = batch.prepare_psd_updates_jax(buffers)
        assert len(calls) == 3
        assert pending[0].result() is True
        with pytest.raises(RuntimeError, match="L1 horizon collection failed"):
            pending[1].result()


def test_empty_batch_has_no_readback_and_cpu_scheme_is_unchanged(monkeypatch):
    with scheme.JAXScheme("cpu"):
        monkeypatch.setattr(jax, "device_get", lambda _: pytest.fail("empty refresh collected"))
        assert batch.recalculate_psds_jax([]) == ()
    with scheme.DefaultScheme(), pytest.raises(TypeError, match="requires JAXScheme"):
        batch.recalculate_psds_jax([])


@pytest.mark.parametrize("dev_name", _devices())
def test_real_welch_batch_matches_existing_recalculation(monkeypatch, dev_name):
    _ensure_x64()
    with scheme.JAXScheme(dev_name):
        oracle = [_buffer(old=None, dtype=dtype) for dtype in (np.float32, np.float64)]
        actual = [_buffer(old=None, dtype=dtype) for dtype in (np.float32, np.float64)]
        assert all(StrainBuffer.recalculate_psd(value) is True for value in oracle)
        welch, device_get = batch.welch_jax, jax.device_get
        submissions = []
        readbacks = []

        def enqueue(*args, **kwargs):
            submissions.append(kwargs)
            return welch(*args, **kwargs)

        def collect(payloads):
            assert len(submissions) == 2
            readbacks.append(payloads)
            return device_get(payloads)

        with monkeypatch.context() as patch:
            patch.setattr(batch, "welch_jax", enqueue)
            patch.setattr(jax, "device_get", collect)
            assert batch.recalculate_psds_jax(actual) == (True, True)
        assert len(readbacks) == 1
        for observed, expected in zip(actual, oracle):
            assert observed.psd.dtype == observed.strain.dtype
            np.testing.assert_array_equal(to_jax(observed.psd), to_jax(expected.psd))
            assert np.float64(observed.psd.dist).tobytes() == np.float64(expected.psd.dist).tobytes()
