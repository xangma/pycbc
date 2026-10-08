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

"""Tests for batched JAX on-device template decompression."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import pycbc
from pycbc.types.array_jax import _ensure_x64
from pycbc.types.backend import backend_array
from pycbc.scheme import CPUScheme, JAXScheme
from pycbc.types import FrequencySeries, zeros
from pycbc.waveform.compress import fd_decompress
from pycbc.waveform.decompress_jax import batched_inline_linear_interp_jax


if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


def test_batched_inline_linear_interp_parity():
    """Verify batched JAX linear interpolation matches single-template CPU interpolation."""
    _ensure_x64()
    b = 4
    flen = 8192
    df = 0.5
    f_lower = 30.0

    amps_list = []
    phases_list = []
    freqs_list = []
    imins = []
    starts = []
    ends = []
    counts = []

    np.random.seed(42)
    for i in range(b):
        k = 100 + i * 20
        counts.append(k)
        freq = np.linspace(25.0, 1000.0 + i * 50.0, k)
        amp = np.random.uniform(0.1, 1.0, k)
        phase = np.random.uniform(-np.pi, np.pi, k)

        imin = int(np.searchsorted(freq, f_lower, side="right")) - 1
        imins.append(imin)
        s_idx = int(np.ceil(f_lower / df))
        starts.append(s_idx)
        last_idx = int(freq[-1] / df)
        e_idx = min(flen, last_idx + 1)
        ends.append(e_idx)

        amps_list.append(amp)
        phases_list.append(phase)
        freqs_list.append(freq)

    # Compute batched JAX result
    res_batch = batched_inline_linear_interp_jax(
        amps_list, phases_list, freqs_list,
        imins, starts, ends, counts,
        df, flen, dtype=jnp.complex64
    )
    res_np = np.asarray(res_batch)

    # Compute reference single-template results
    for i in range(b):
        res_single = batched_inline_linear_interp_jax(
            [amps_list[i]], [phases_list[i]], [freqs_list[i]],
            [imins[i]], [starts[i]], [ends[i]], [counts[i]],
            df, flen, dtype=jnp.complex64
        )[0]
        single_np = np.asarray(res_single)

        max_diff = np.max(np.abs(res_np[i] - single_np))
        assert max_diff < 1e-6, f"Batch vs single mismatch on template {i}: {max_diff}"


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_jax_path_matches_cpu_reference(dtype, device):
    """Host-staged batches retain exact CPU semantics on each JAX device."""
    if device == "cuda":
        try:
            jax.devices("cuda")
        except RuntimeError:
            pytest.skip("CUDA JAX device unavailable")

    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    freq = np.array([10.25, 31.1, 57.7, 91.3, 140.2], dtype=real_dtype)
    amp = np.array([.4, .8, .3, .9, .2], dtype=real_dtype)
    phase = np.array([-.2, .4, 1.1, -.7, .9], dtype=real_dtype)
    f_lower = 14.7
    df = .7
    out_len = 128
    imin = int(np.searchsorted(freq, f_lower, side="right")) - 1
    start = int(np.ceil(f_lower / df))
    args = ([amp], [phase], [freq], [imin], [start], [out_len], [5], df, out_len)
    with JAXScheme(device=device):
        result = batched_inline_linear_interp_jax(*args, dtype=dtype)
        raw = backend_array(result, "jax")
        assert raw is not None
        assert raw.dtype == dtype
        platform = next(iter(raw.devices())).platform
        assert platform in ("cuda", "gpu") if device == "cuda" else platform == "cpu"
        result_np = np.asarray(result)
        assert result_np.shape == (1, 128)
        assert np.any(np.abs(result_np[:, start:]) > 0)
        assert np.all(result_np[:, :start] == 0)

    with CPUScheme():
        expected = FrequencySeries(
            zeros(out_len, dtype=dtype), delta_f=df, copy=False,
        )
        fd_decompress(
            amp, phase, freq, out=expected, f_lower=f_lower,
            interpolation="inline_linear",
        )
    np.testing.assert_array_equal(result_np[0], expected.numpy())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_jax_path_matches_cpu_reference(dtype):
    """Single-template JAX dispatch preserves native interpolation exactly."""
    frequencies = np.array([0.0, .43, 1.17, 2.05, 3.4, 4.8], dtype=dtype)
    amplitude = np.exp(-.08 * frequencies).astype(dtype)
    phase = (.13 * frequencies).astype(dtype)
    with JAXScheme():
        result = fd_decompress(
            amplitude, phase, frequencies, df=.2, f_lower=.71,
            interpolation="inline_linear",
        )
        raw = backend_array(result, "jax")
        assert raw is not None
        assert raw.dtype == (np.complex64 if dtype == np.float32 else np.complex128)
        assert next(iter(raw.devices())).platform == "cpu"
        result_np = np.asarray(result)
        assert result.shape[0] > 0
        assert np.any(np.abs(result_np) > 0)
    with CPUScheme():
        expected = fd_decompress(
            amplitude, phase, frequencies, df=.2, f_lower=.71,
            interpolation="inline_linear",
        )
    np.testing.assert_array_equal(result_np, expected.numpy())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_jax_short_rows_are_zero(device):
    """Rows with fewer than two samples preserve native zero-output semantics."""
    if device == "cuda":
        try:
            jax.devices("cuda")
        except RuntimeError:
            pytest.skip("CUDA JAX device unavailable")

    with JAXScheme(device=device):
        result = batched_inline_linear_interp_jax(
            [np.array([]), np.array([2.0]), np.array([1.0, 2.0])],
            [np.array([]), np.array([.3]), np.array([0.0, 0.0])],
            [np.array([]), np.array([10.0]), np.array([1.0, 2.0])],
            [0, 0, 0], [0, 0, 0], [0, 8, 8], [0, 1, 2],
            .5, 8, dtype=jnp.complex64,
        )
        raw = backend_array(result, "jax")
        assert raw is not None
        platform = next(iter(raw.devices())).platform
        assert platform == "cpu" if device == "cpu" else platform in ("cuda", "gpu")
        result = np.asarray(result)
        assert np.all(result[:2] == 0)
        assert np.any(np.abs(result[2]) > 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "method",
    ["inline_linear", "inline_quadratic", "inline_cubic", "inline_quartic"],
)
def test_grid_boundary_parity(dtype, method):
    """Near-grid sample points retain the native last-bin decisions."""
    df = dtype(.2)
    grid = dtype(24) * df
    frequencies = np.array(
        [0, .43, 1.17, 2.05, 3.4, np.nextafter(grid, dtype(0))],
        dtype=dtype,
    )
    amplitude = np.exp(-.08 * frequencies).astype(dtype)
    phase = (.13 * frequencies).astype(dtype)
    f_lower = float(np.nextafter(frequencies[1], frequencies[2]))

    with JAXScheme():
        actual = fd_decompress(
            amplitude, phase, frequencies, df=float(df), f_lower=f_lower,
            interpolation=method,
        )
        actual = np.asarray(actual)
    with CPUScheme():
        expected = fd_decompress(
            amplitude, phase, frequencies, df=float(df), f_lower=f_lower,
            interpolation=method,
        )
    np.testing.assert_array_equal(actual, expected.numpy())


def test_stage_batched_device_interp_jax():
    """Verify on-device batched linear interpolation produces expected output."""
    from pycbc.waveform.decompress_jax import stage_batched_device_interp_jax

    b = 4
    flen = 1024
    df = 0.5
    f_lower = 30.0

    amps_list = []
    phases_list = []
    freqs_list = []
    starts = []
    ends = []
    counts = []

    np.random.seed(42)
    for i in range(b):
        k = 50 + i * 10
        counts.append(k)
        freq = np.linspace(25.0, 400.0, k, dtype=np.float32)
        amp = np.random.uniform(0.1, 1.0, k).astype(np.float32)
        phase = np.random.uniform(-np.pi, np.pi, k).astype(np.float32)
        s_idx = int(np.ceil(f_lower / df))
        starts.append(s_idx)
        ends.append(flen)
        amps_list.append(amp)
        phases_list.append(phase)
        freqs_list.append(freq)

    with JAXScheme():
        _, device_waveforms = stage_batched_device_interp_jax(
            amps_list, phases_list, freqs_list,
            starts, ends, counts,
            df, flen, dtype=jnp.complex64,
        )
        assert device_waveforms is not None
        assert device_waveforms.shape == (b, flen)
        res_np = np.asarray(device_waveforms)
        assert np.all(res_np[:, :int(np.ceil(f_lower / df))] == 0)
        assert np.any(np.abs(res_np[:, int(np.ceil(f_lower / df)):]) > 0)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_decompression_direct_interpolation_oracle(dtype, device):
    """Check direct arithmetic and support without requiring host recurrence parity."""
    from pycbc.waveform.decompress_jax import stage_batched_device_interp_jax

    if device == "cuda":
        try:
            jax.devices("cuda")
        except RuntimeError:
            pytest.skip("CUDA JAX device unavailable")

    real_dtype = np.empty((), dtype=dtype).real.dtype
    frequencies = [
        np.array([0., 1.25, 3.5, 7.25], dtype=real_dtype),
        np.array([0., 1.25, 3.5, 7.25], dtype=real_dtype),
        np.array([2.], dtype=real_dtype),
        np.array([], dtype=real_dtype),
        np.array([0., 4.], dtype=real_dtype),
        np.array([0., 4.], dtype=real_dtype),
    ]
    amplitudes = [
        np.linspace(.3, .9, len(row), dtype=real_dtype) for row in frequencies
    ]
    phases = [
        np.linspace(-.4, 1.1, len(row), dtype=real_dtype) for row in frequencies
    ]
    starts = [-1, 3, 0, 0, 2, 5]
    ends = [7, 20, 16, 16, 6, 5]
    counts = [4, 4, 1, 0, 99, 2]
    df, out_len = .5, 16
    with JAXScheme(device=device):
        host, result = stage_batched_device_interp_jax(
            amplitudes, phases, frequencies, starts, ends, counts,
            df, out_len, dtype=dtype,
        )
        assert host is None
        assert result.dtype == dtype
        platform = next(iter(result.devices())).platform
        assert platform in ("cuda", "gpu") if device == "cuda" else platform == "cpu"
        actual = np.asarray(result)

    expected = np.zeros((len(frequencies), out_len), dtype=dtype)
    bins = np.arange(out_len)
    evaluation_frequencies = bins * df
    for row, (amp, phase, freq, start, end, count) in enumerate(zip(
        amplitudes, phases, frequencies, starts, ends, counts
    )):
        count = min(count, len(amp), len(phase), len(freq))
        if count < 2:
            continue
        valid = ((bins >= max(0, start)) & (bins < min(end, out_len))
                 & (evaluation_frequencies <= freq[count - 1]))
        a = np.interp(evaluation_frequencies[valid], freq[:count], amp[:count])
        p = np.interp(evaluation_frequencies[valid], freq[:count], phase[:count])
        expected[row, valid] = a * (np.cos(p) + 1j * np.sin(p))

    np.testing.assert_array_equal(actual[expected == 0], 0)
    tolerance = 2e-6 if dtype == np.complex64 else 1e-13
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_device_decompression_matches_native_support_boundaries(dtype):
    """Constant phase isolates end-exclusive and short-row support semantics."""
    from pycbc.waveform.decompress_jax import (
        stage_batched_device_interp_jax, stage_batched_inline_interp_jax,
    )

    real_dtype = np.empty((), dtype=dtype).real.dtype
    frequencies = [np.array(row, dtype=real_dtype) for row in ([], [2], [0, 5])]
    amplitudes = [np.ones(len(row), dtype=real_dtype) for row in frequencies]
    phases = [np.zeros(len(row), dtype=real_dtype) for row in frequencies]
    starts, ends, counts = [0, 0, 0], [8, 8, 3], [0, 1, 2]
    with JAXScheme(device="cpu"):
        expected, _ = stage_batched_inline_interp_jax(
            amplitudes, phases, frequencies, [0, 0, 0], starts, ends, counts,
            ["inline_linear"] * 3, 1., 8, dtype=dtype,
        )
        _, actual = stage_batched_device_interp_jax(
            amplitudes, phases, frequencies, starts, ends, counts,
            1., 8, dtype=dtype,
        )
        np.testing.assert_array_equal(np.asarray(actual), expected)


