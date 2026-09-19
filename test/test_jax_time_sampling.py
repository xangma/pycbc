# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Concrete Series sampling retains native indexing and extrapolation."""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
from pycbc.scheme import CPUScheme, JAXScheme
from pycbc.types import TimeSeries


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not any(d.platform == 'gpu' for d in jax.devices()):
        pytest.skip('CUDA device unavailable')
    return request.param


@pytest.mark.parametrize('interpolate', [None, 'linear', 'quadratic'])
@pytest.mark.parametrize('time', [1002., 999., [1000.2, 1002.], 1000.5, 999.5])
def test_time_sampling_preserves_native_errors(device, interpolate, time):
    values = np.arange(8, dtype=np.float64)
    with CPUScheme():
        original = TimeSeries(values, delta_t=.0625, epoch=1000)
        try:
            expected = original.at_time(time, interpolate=interpolate)
        except IndexError as error:
            expected_error = str(error)
        else:
            expected_error = None
    with JAXScheme(device):
        candidate = TimeSeries(values, delta_t=.0625, epoch=1000)
        if expected_error is not None:
            with pytest.raises(IndexError) as error:
                candidate.at_time(time, interpolate=interpolate)
            assert str(error.value) == expected_error
        else:
            np.testing.assert_array_equal(
                candidate.at_time(time, interpolate=interpolate), expected)


@pytest.mark.parametrize('interpolate', [None, 'linear', 'quadratic'])
@pytest.mark.parametrize('nearest', [False, True])
def test_time_sampling_preserves_negative_indices_and_fill(device, interpolate, nearest):
    values = np.arange(8, dtype=np.float64)
    times = np.array([999.8, 1000., 1000.2, 1000.45, 1002.])
    kwargs = dict(interpolate=interpolate, nearest_sample=nearest, extrapolate=-9.)
    with CPUScheme():
        original = TimeSeries(values, delta_t=.0625, epoch=1000)
        expected = original.at_time(times, **kwargs)
        wrapped = original.at_time(999.8, interpolate=interpolate)
    with JAXScheme(device):
        candidate = TimeSeries(values, delta_t=.0625, epoch=1000)
        np.testing.assert_array_equal(candidate.at_time(times, **kwargs), expected)
        np.testing.assert_array_equal(candidate.at_time(999.8, interpolate=interpolate), wrapped)
