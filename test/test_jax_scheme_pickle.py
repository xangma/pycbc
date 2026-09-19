# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""JAX configuration and Series serialize without retaining active contexts."""
import pickle

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pycbc import scheme
from pycbc.types import TimeSeries
from pycbc.types.backend import backend_name


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("active", [False, True])
def test_scheme_pickle_preserves_configuration_without_entering_context(device, active):
    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    context = scheme.JAXScheme(device, num_threads=2, chisq_mode="direct-phase",
                              reference_operations=("inner", "sum"))
    previous = scheme.mgr.state

    def roundtrip():
        current = scheme.mgr.state
        restored = pickle.loads(pickle.dumps(context))
        assert scheme.mgr.state is current
        assert restored._prev_default_device is None
        assert restored.device_spec == context.device_spec
        assert restored.jax_device == context.jax_device
        assert restored.num_threads == context.num_threads
        assert restored.jax_chisq_mode == context.jax_chisq_mode
        assert restored.jax_highpass_mode == context.jax_highpass_mode
        assert restored.jax_reference_operations == context.jax_reference_operations

    if active:
        with context:
            roundtrip()
    else:
        roundtrip()
    assert scheme.mgr.state is previous


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_series_pickle_preserves_data_metadata_and_original_context(device):
    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    values = np.array([1., -2., 3.5], dtype=np.float64)
    previous = scheme.mgr.state
    with scheme.JAXScheme(device) as context:
        series = TimeSeries(values, delta_t=.25, epoch=1234)
        restored = pickle.loads(pickle.dumps(series))
        assert scheme.mgr.state is context
        assert restored.delta_t == series.delta_t
        assert restored.start_time == series.start_time
        assert restored.dtype == series.dtype
        assert backend_name(restored) == "jax"
        np.testing.assert_array_equal(restored.numpy(), values)
        np.testing.assert_array_equal((restored * 2).numpy(), values * 2)
    assert scheme.mgr.state is previous
