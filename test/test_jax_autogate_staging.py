# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Tests for the JAX autogate padding and CPU parity contract."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.strain import strain as strain_cpu
import pycbc.strain.strain_jax as strain_jax
from pycbc.types import TimeSeries
from pycbc.types.array_jax import JAXArrayData, from_jax, is_jax_array

jax = pytest.importorskip("jax")


_jax_devices = ["cpu"]
if any(device.platform in ("gpu", "cuda") for device in jax.devices()):
    _jax_devices.append("gpu")


@pytest.fixture(params=_jax_devices, ids=lambda device: f"jax-{device}")
def jax_device(request):
    return request.param


def test_zero_pad_on_device_does_not_use_numpy_padding(
    monkeypatch, jax_device
):
    """The padded autogate input is allocated and sliced on the JAX device."""
    data = np.arange(5, dtype=np.float64)

    def reject_host_padding(*args, **kwargs):
        raise AssertionError("autogate padding was allocated on the host")

    monkeypatch.setattr(strain_jax.np, "zeros", reject_host_padding)
    with scheme.JAXScheme(device=jax_device):
        strain = TimeSeries(data, delta_t=1.0)

        def reject_host_transfer(*args, **kwargs):
            raise AssertionError("JAX strain data was copied to the host")

        monkeypatch.setattr(JAXArrayData, "numpy", reject_host_transfer)
        monkeypatch.setattr(JAXArrayData, "__array__", reject_host_transfer)
        padded = strain_jax._zero_pad_on_device(
            strain, 8, 1, 6
        )

    assert is_jax_array(padded)
    np.testing.assert_array_equal(
        from_jax(padded), np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0])
    )
    assert padded.dtype == np.float64


def test_jax_autogate_matches_cpu_peak_times(jax_device):
    """Device-side padding preserves the CPU autogate peak locations."""
    rng = np.random.default_rng(1)
    strain = TimeSeries(
        rng.normal(size=512).astype(np.float64), delta_t=1.0 / 64.0
    )
    kwargs = dict(
        psd_duration=2.0,
        psd_stride=1.0,
        low_freq_cutoff=4.0,
        threshold=3.0,
        cluster_window=0.5,
        corrupt_time=1.0,
    )

    cpu_times = strain_cpu.detect_loud_glitches(strain, **kwargs)
    original = strain.numpy().copy()
    with scheme.JAXScheme(device=jax_device):
        jax_strain = TimeSeries(original, delta_t=1.0 / 64.0)
        jax_times, snrs = strain_jax.detect_loud_glitches_jax(
            jax_strain, **kwargs
        )

    np.testing.assert_array_equal(
        jax_times, np.asarray(cpu_times, dtype=float)
    )
    assert len(snrs) == len(jax_times)
    assert all(snr > kwargs["threshold"] for snr in snrs)
    np.testing.assert_array_equal(jax_strain.numpy(), original)
