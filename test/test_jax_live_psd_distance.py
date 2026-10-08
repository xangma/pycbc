# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under the GNU General Public License, version 3 or later.

"""Live PSD horizon parity and its explicit scientific control boundary."""

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import pycbc
from pycbc import scheme
from pycbc.types import FrequencySeries
from pycbc.types.array_jax import _ensure_x64, to_jax
from pycbc.strain import strain_jax
from pycbc.strain.strain import StrainBuffer
from pycbc.waveform.spa_tmplt import spa_distance
from pycbc.vetoes import chisq_jax


def _devices():
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("cuda:0")
    except RuntimeError:
        pass
    return devices


def _psd(length, delta_f, dtype):
    rng = np.random.default_rng(834)
    values = np.exp(rng.uniform(-3, 3, size=length)).astype(dtype)
    return FrequencySeries(values, delta_f=delta_f)


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("length,delta_f", [(257, 0.5), (1025, 4.0), (4097, 0.25)])
def test_horizon_matches_legacy_scalar_bytes_and_changed_psd(dev_name, dtype,
                                                           length, delta_f):
    _ensure_x64()
    with scheme.CPUScheme():
        reference = _psd(length, delta_f, dtype)
        expected = spa_distance(reference, 1.4, 1.4, 20.0) * pycbc.DYN_RANGE_FAC
        expected_changed = spa_distance(reference * 4.0, 1.4, 1.4, 20.0) * pycbc.DYN_RANGE_FAC
    with scheme.JAXScheme(dev_name):
        psd = _psd(length, delta_f, dtype)
        actual = strain_jax.psd_horizon_distance_jax(psd, 20.0)
        assert type(actual) is float
        assert np.float64(actual).tobytes() == np.float64(expected).tobytes()
        changed = psd * 4.0
        actual_changed = strain_jax.psd_horizon_distance_jax(changed, 20.0)
        assert np.float64(actual_changed).tobytes() == np.float64(expected_changed).tobytes()
        assert actual_changed == actual / 2.0


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_full_length_horizon_preserves_every_weighted_prefix(
        monkeypatch, dev_name, dtype):
    """The Live 4 s PSD grid retains all 3977 ordered additions."""
    _ensure_x64()
    scan = chisq_jax._ordered_cumsum_rows
    calls = []

    def checked(values, use_gpu_scan):
        assert values.shape == (1, 3977)
        assert values.dtype == np.dtype(dtype)
        result = scan(values, use_gpu_scan)
        expected = chisq_jax._ordered_cumsum(values[0])
        assert np.asarray(result[0]).tobytes() == np.asarray(expected).tobytes()
        calls.append(bool(use_gpu_scan))
        return result

    with scheme.JAXScheme(dev_name):
        psd = _psd(4097, 0.25, dtype)
        expected_gpu_scan = chisq_jax._use_gpu_ordered_scan(to_jax(psd))
        monkeypatch.setattr(chisq_jax, "_ordered_cumsum_rows", checked)
        first = strain_jax.psd_horizon_distance_jax(psd, 30)
        changed = strain_jax.psd_horizon_distance_jax(psd * 4.0, 30)
        assert changed == first / 2.0
    assert calls == [expected_gpu_scan, expected_gpu_scan]


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("case", ["cancellation", "zeros", "nonfinite"])
def test_horizon_scan_reuse_preserves_order_and_exceptional_values(dev_name, case):
    _ensure_x64()
    values = np.zeros(3977, dtype=np.float32)
    if case == "cancellation":
        values[:5] = [2**24, 1, -2**24, 0, 0]
    elif case == "nonfinite":
        values[:5] = [1, np.inf, -np.inf, np.nan, 1]
    with scheme.JAXScheme(dev_name):
        values = to_jax(values)
        result = np.asarray(chisq_jax._ordered_cumsum_rows(
            values[None, :], chisq_jax._use_gpu_ordered_scan(values))[0])
        expected = np.asarray(chisq_jax._ordered_cumsum(values))
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    # Invalid PSDs retain NaN behavior; finite values and infinity signs
    # retain exact payloads, without prescribing backend NaN payload bits.
    valid = ~np.isnan(expected)
    assert result[valid].tobytes() == expected[valid].tobytes()


@pytest.mark.parametrize("dev_name", _devices())
def test_horizon_collects_only_completed_scalar_and_domain_status(monkeypatch, dev_name):
    _ensure_x64()
    with scheme.JAXScheme(dev_name):
        psd = _psd(2049, 1.0, np.float32)
        device_get = jax.device_get
        array_type = type(to_jax(psd))
        original_array = array_type.__array__
        collecting = False
        collected = []

        def reject(*args, **kwargs):
            raise AssertionError("PSD horizon performed an implicit device readback")

        def guarded_array(self, *args, **kwargs):
            if not collecting:
                reject()
            return original_array(self, *args, **kwargs)

        def collect(tree):
            nonlocal collecting
            assert isinstance(tree, tuple) and len(tree) == 2
            assert [leaf.shape for leaf in tree] == [(), ()]
            assert [leaf.dtype for leaf in tree] == [np.dtype(np.float64), np.dtype(bool)]
            collected.append(tree)
            collecting = True
            try:
                return device_get(tree)
            finally:
                collecting = False

        with monkeypatch.context() as patch:
            patch.setattr(jax, "device_get", collect)
            patch.setattr(array_type, "__array__", guarded_array)
            for name in ("__int__", "__float__", "__bool__"):
                patch.setattr(array_type, name, reject)
            distance = strain_jax.psd_horizon_distance_jax(psd, 30.0)
        assert len(collected) == 1
        assert np.isfinite(distance) and distance > 0.0


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("spectrum", [-1.0, 0.0, np.inf, np.nan])
def test_horizon_keeps_legacy_invalid_psd_behavior(dev_name, spectrum):
    _ensure_x64()
    with scheme.JAXScheme(dev_name):
        psd = FrequencySeries(np.full(257, spectrum, np.float32), delta_f=1.0)
        if spectrum < 0.0:
            with pytest.raises(ValueError, match="math domain error"):
                strain_jax.psd_horizon_distance_jax(psd, 20.0)
        else:
            expected = spa_distance(psd, 1.4, 1.4, 20.0) * pycbc.DYN_RANGE_FAC
            actual = strain_jax.psd_horizon_distance_jax(psd, 20.0)
            if np.isnan(expected):
                assert np.isnan(actual)
            else:
                assert actual == expected


@pytest.mark.parametrize("distance,accepted", [(1.0, True), (2.0, True),
                                               (0.99, False), (2.01, False),
                                               (np.inf, False), (np.nan, False)])
def test_cached_host_distance_keeps_required_range_decisions(distance, accepted):
    buffer = SimpleNamespace(psd=SimpleNamespace(dist=float(distance)), detector="H1")
    assert bool(StrainBuffer.check_psd_dist(buffer, 1.0, 2.0)) is accepted


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("operations", [
    ("psd_horizon",), ("cumsum",), ("psd_horizon_amplitude",), ("divide",),
    ("psd_horizon_amplitude", "divide", "cumsum"),
])
def test_horizon_original_whole_and_independent_stage_routes(
        monkeypatch, dev_name, dtype, operations):
    with scheme.CPUScheme():
        psd = _psd(257, .5, dtype)
        expected = spa_distance(psd, 1.4, 1.4, 20) * pycbc.DYN_RANGE_FAC
        values = psd.numpy().copy()
    if set(operations) & {"psd_horizon", "cumsum"}:
        monkeypatch.setattr(chisq_jax, "_ordered_cumsum_rows", lambda *args: (
            pytest.fail("original horizon/cumsum control used the JAX recurrence")))
    with scheme.JAXScheme(dev_name, reference_operations=operations):
        actual = strain_jax.psd_horizon_distance_jax(
            FrequencySeries(values, delta_f=.5), 20)
    assert np.float64(actual).tobytes() == np.float64(expected).tobytes()
