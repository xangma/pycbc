# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under the GNU General Public License, version 3 or later.

"""Scientific and residency checks for fixed-shape Live autogating."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pycbc import scheme
from pycbc.types import TimeSeries
from pycbc.types.array_jax import JAXArrayData, _ensure_x64, to_jax
from pycbc.strain import strain_jax


def _devices():
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("cuda:0")
    except RuntimeError:
        pass
    return devices


def _peaks(magnitude, threshold, window):
    selected = []
    for index in np.flatnonzero(magnitude > threshold):
        if not selected or index - selected[-1] > window:
            selected.append(index)
        elif magnitude[index] > magnitude[selected[-1]]:
            selected[-1] = index
    return np.asarray(selected, dtype=np.int32)


@pytest.mark.parametrize("case", ["none", "ties", "overlap", "edges", "dense"])
@pytest.mark.parametrize("width,taper", [(0.15, 0.25), (0.2, 0.0), (0.0, 0.0)])
@pytest.mark.parametrize("dev_name", _devices())
def test_resident_gate_selection_and_order_match_legacy(case, width, taper, dev_name):
    _ensure_x64()
    with scheme.JAXScheme(dev_name):
        rate, window, size = 20, 2, 128
        epoch = 1187007104.0001
        detection_epoch = epoch - (0.25 if case == "edges" else 0.0)
        rng = np.random.default_rng(9108)
        values = rng.normal(size=size).astype(np.float32)
        magnitude = np.zeros(size, np.float32)
        if case == "ties":
            magnitude[[3, 5, 6, 9, 11]] = [8, 8, 9, 7, 7]
        elif case == "overlap":
            magnitude[[15, 18, 21, 24, 27]] = [8, 9, 7, 8, 7]
        elif case == "edges":
            magnitude[[0, 3, 6, 125, 127]] = [8, 9, 7, 8, 8]
        elif case == "dense":
            magnitude[:] = 8
        selected = _peaks(magnitude, 5, window)
        times = detection_epoch + selected * (1.0 / rate)
        expected = TimeSeries(values.copy(), delta_t=1.0 / rate, epoch=epoch)
        strain_jax.gate_data_jax(expected, [(time, width, taper) for time in times])
        actual, indices, count = strain_jax._resident_autogate_core(
            jnp.asarray(values), jnp.asarray(magnitude), 5.0,
            detection_epoch, 1.0 / rate, epoch, width, taper,
            cluster_samples=window, window_samples=int(2 * rate * (width + taper)),
            pad_samples=int(rate * taper))
        np.testing.assert_array_equal(np.asarray(indices)[:int(count)], selected)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(to_jax(expected)))


def test_empty_magnitude_retains_values():
    _ensure_x64()
    values = jnp.arange(8, dtype=jnp.float32)
    output, indices, count = strain_jax._resident_autogate_core(
        values, jnp.empty(0, jnp.float32), 5.0, 0.0, 0.25, 0.0, 0.25, 0.25,
        cluster_samples=2, window_samples=4, pad_samples=1)
    np.testing.assert_array_equal(np.asarray(output), np.asarray(values))
    assert indices.shape == (0,)
    assert int(count) == 0


def _buffer():
    rate = 64
    rng = np.random.default_rng(127)
    strain = TimeSeries(rng.normal(size=1024).astype(np.float32),
                        delta_t=1.0 / rate, epoch=1187007104.0)
    return SimpleNamespace(
        strain=strain, sample_rate=rate, corruption=8,
        autogating_duration=8, autogating_threshold=5.0,
        autogating_cluster=0.25, autogating_width=0.1, autogating_taper=0.1,
        autogating_pad=0.125, autogating_psd_segment_length=1.0,
        autogating_psd_stride=0.5, highpass_frequency=5.0,
        gate_params=[])


@pytest.mark.parametrize("dev_name", _devices())
def test_live_gate_helper_defers_all_numeric_collection(monkeypatch, dev_name):
    _ensure_x64()
    with scheme.JAXScheme(dev_name):
        buffer = _buffer()
        original = to_jax(buffer.strain)
        magnitude = jnp.zeros(504, jnp.float32).at[50].set(8).at[90].set(9)
        if dev_name == "cpu":
            monkeypatch.setattr(strain_jax, "_is_cuda_autogate_array", lambda _: True)
        monkeypatch.setattr(strain_jax, "_autogate_magnitude_jax",
                            lambda detection, **kwargs: (detection, magnitude))

        def reject(*args, **kwargs):
            raise AssertionError("live gating collected a scientific device value")

        array_type = type(original)
        with monkeypatch.context() as patch:
            patch.setattr(jax, "device_get", reject)
            for name in ("__array__", "__int__", "__float__", "__bool__"):
                patch.setattr(array_type, name, reject)
            assert strain_jax.autogate_strain_buffer_jax(buffer)
            frozen = buffer.gate_params.snapshot()
            assert frozen is buffer.gate_params
        rows = frozen.materialize()
        assert len(rows) == 2
        epoch = frozen.epoch
        buffer.strain.start_time += 8
        buffer.autogating_width = 20
        assert frozen.epoch == epoch
        assert all(row[1:] == (0.1, 0.1) for row in frozen.materialize())
        with pytest.raises(FrozenInstanceError):
            frozen.width = 7
        assert not np.array_equal(np.asarray(original), np.asarray(to_jax(buffer.strain)))


def test_live_gpu_helper_keeps_default_cpu_path(monkeypatch):
    with scheme.JAXScheme("cpu"):
        buffer = _buffer()
        original = buffer.strain._data
        assert not strain_jax.autogate_strain_buffer_jax(buffer)
        assert buffer.strain._data is original
        assert buffer.gate_params == []


@pytest.mark.parametrize("dev_name", _devices())
def test_real_whitening_path_does_not_collect_values(monkeypatch, dev_name):
    _ensure_x64()
    with scheme.JAXScheme(dev_name):
        buffer = _buffer()
        buffer.autogating_threshold = 1000
        # Exercise a first-use whitening geometry and a nontrivial edge taper.
        buffer.autogating_duration = 7.75
        buffer.autogating_pad = 9.0 / buffer.sample_rate
        before = np.asarray(to_jax(buffer.strain))
        if dev_name == "cpu":
            monkeypatch.setattr(strain_jax, "_is_cuda_autogate_array", lambda _: True)

        def reject(*args, **kwargs):
            raise AssertionError("whitening/gating collected a scientific value")

        original_init = strain_jax.pycbc.types.Array.__init__
        tapers = []

        def resident_init(self, initial_array, *args, **kwargs):
            # Raw JAX inputs fall through the standard constructor's NumPy copy
            # on CUDA. Check this boundary on CPU too, so the regression is local.
            assert not isinstance(initial_array, jax.Array)
            if (type(self) is strain_jax.pycbc.types.Array and
                    isinstance(initial_array, JAXArrayData) and len(initial_array) == 9):
                assert kwargs.get("copy") is False
                tapers.append(initial_array.array)
            return original_init(self, initial_array, *args, **kwargs)

        array_type = type(to_jax(buffer.strain))
        with monkeypatch.context() as patch:
            patch.setattr(strain_jax.pycbc.types.Array, "__init__", resident_init)
            patch.setattr(jax, "device_get", reject)
            for name in ("__array__", "__int__", "__float__", "__bool__"):
                patch.setattr(array_type, name, reject)
            assert strain_jax.autogate_strain_buffer_jax(buffer)
        assert len(tapers) == 2
        expected_taper = (np.arange(9) / 9.0).astype(before.dtype)
        np.testing.assert_array_equal(np.asarray(tapers[0]), expected_taper)
        np.testing.assert_array_equal(np.asarray(tapers[1]), expected_taper[::-1])
        assert buffer.gate_params.materialize() == []
        np.testing.assert_array_equal(np.asarray(to_jax(buffer.strain)), before)
