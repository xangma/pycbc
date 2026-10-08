# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent original decompression and taper controls preserve contracts."""

import os

import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.types.array_jax import to_jax
from pycbc.waveform import utils, utils_jax
from pycbc.waveform.compress import fd_decompress
from pycbc.waveform.decompress_jax import stage_batched_device_interp_jax

pytest.importorskip('jax')
DEVICE = os.environ.get('PYCBC_TEST_SCHEME', 'jax:cpu').removeprefix('jax:')


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('method', ['inline_linear', 'nearest'])
def test_original_decompression_preserves_output_view(dtype, method):
    frequencies = np.array([0., 1.25, 3.5, 7.25], dtype=dtype)
    amplitude = np.array([.4, .8, .3, .9], dtype=dtype)
    phase = np.array([-.4, .3, -.7, 1.1], dtype=dtype)
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
    values = np.full(20, 5 + 2j, dtype=complex_dtype)
    with scheme.CPUScheme():
        expected = FrequencySeries(values[2:18], delta_f=.5, epoch=12)
        fd_decompress(amplitude, phase, frequencies, out=expected,
                      f_lower=1.25, interpolation=method)
    with scheme.JAXScheme(DEVICE, reference_operations=('decompress',)) as ctx:
        parent = FrequencySeries(values, delta_f=.5, epoch=12)
        output = parent[2:18]
        result = fd_decompress(amplitude, phase, frequencies, out=output,
                               f_lower=1.25, interpolation=method)
        assert result is output
        assert to_jax(result).devices() == {ctx.jax_device}
        assert result.numpy().tobytes() == expected.numpy().tobytes()
        assert result._epoch == expected._epoch
        assert result.delta_f == expected.delta_f
        np.testing.assert_array_equal(parent[:2].numpy(), values[:2])
        np.testing.assert_array_equal(parent[18:].numpy(), values[18:])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_device_decompression_reference_calls_original_kernel(dtype):
    frequencies = np.array([0., 1.25, 3.5, 7.25], dtype=dtype)
    amplitude = np.array([.4, .8, .3, .9], dtype=dtype)
    phase = np.array([-.4, .3, -.7, 1.1], dtype=dtype)
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
    with scheme.CPUScheme():
        expected = fd_decompress(amplitude, phase, frequencies,
                                 df=.5, f_lower=1.5)
    with scheme.JAXScheme(DEVICE, reference_operations=('decompress',)) as ctx:
        _, result = stage_batched_device_interp_jax(
            [amplitude], [phase], [frequencies], [3], [len(expected)], [4],
            .5, len(expected), dtype=complex_dtype)
        assert result.devices() == {ctx.jax_device}
        assert np.asarray(result[0]).tobytes() == expected.numpy().tobytes()


@pytest.mark.parametrize('operation', ['td_taper', 'fd_taper'])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_original_taper_is_exact_copy(operation, side, monkeypatch):
    values = np.linspace(1., 2., 64, dtype=np.float64)
    def source():
        if operation == 'td_taper':
            return TimeSeries(values, delta_t=.25, epoch=0)
        return FrequencySeries(values.astype(np.complex128), delta_f=.25, epoch=0)
    with scheme.CPUScheme():
        expected = getattr(utils, operation)(source(), 2, 6, side=side)
    monkeypatch.setattr(utils_jax, '_jax_kaiser_window',
                        lambda *a: pytest.fail('JAX window was selected'))
    with scheme.JAXScheme(DEVICE, reference_operations=(operation,)) as ctx:
        original = source()
        result = getattr(utils, operation)(original, 2, 6, side=side)
        assert result is not original
        assert to_jax(result).devices() == {ctx.jax_device}
        assert result.numpy().tobytes() == expected.numpy().tobytes()
        np.testing.assert_array_equal(original.numpy(), values)
        assert result._epoch == expected._epoch
