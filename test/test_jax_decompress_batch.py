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
import h5py
from types import SimpleNamespace

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import pycbc
from pycbc.types.array_jax import _ensure_x64
from pycbc.types.backend import backend_array
from pycbc.scheme import CPUScheme, JAXScheme
from pycbc.types import FrequencySeries, zeros
from pycbc.waveform.bank import FilterBank
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


def _compressed_bank(filehandler, methods, override=None):
    """Build a small bank around real compressed waveform HDF groups."""
    frequencies = np.array([0.0, .43, 1.17, 2.05, 3.4, 4.8, 6.35, 8.0])
    amplitude = np.exp(-.08 * frequencies) * (1 + .03 * np.cos(.7 * frequencies))
    phase = .13 * frequencies + .011 * frequencies ** 2
    rows = []
    for index, method in enumerate(methods):
        group = filehandler.create_group(f"compressed_waveforms/{index}")
        group["sample_points"] = frequencies
        group["amplitude"] = amplitude
        group["phase"] = phase
        group.attrs["interpolation"] = method
        rows.append(SimpleNamespace(approximant="TaylorF2", template_duration=1.0))

    class Table:
        template_hash = np.arange(len(rows))

        def __getitem__(self, index):
            return rows[index]

    bank = object.__new__(FilterBank)
    bank.table = Table()
    bank.filehandler = filehandler
    bank.filter_length = 48
    bank.delta_f = .2
    bank.dtype = np.complex128
    bank.f_lower = .71
    bank.min_f_lower = .71
    bank.max_template_length = None
    bank.extra_args = {}
    bank.waveform_decompression_method = override
    bank._template_cache = {}
    bank.approximant = lambda index: "TaylorF2"
    bank.end_frequency = lambda index: None
    return bank, frequencies, amplitude, phase


@pytest.mark.parametrize("methods,override", [
    (("inline_linear", "inline_quadratic", "inline_cubic", "inline_quartic"), None),
    (("inline_linear", "inline_cubic"), "inline_quartic"),
    (("inline_quadratic", "inline_quadratic"), "inline_linear"),
])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bank_batch_honors_interpolation_metadata_and_override(
    tmp_path, methods, override, dtype, device
):
    if device == "cuda":
        try:
            jax.devices("cuda")
        except RuntimeError:
            pytest.skip("CUDA JAX device unavailable")

    with h5py.File(tmp_path / "compressed.hdf", "w") as filehandler:
        bank, frequencies, amplitude, phase = _compressed_bank(
            filehandler, methods, override
        )
        bank.dtype = dtype
        with JAXScheme(device=device):
            bank._decompress_batch_jax(list(range(len(methods))))
            actual = np.asarray(bank._last_batch_tensor)
            assert backend_array(bank._template_cache[0], "jax") is not None

        with CPUScheme():
            for index, method in enumerate(methods):
                output = FrequencySeries(
                    zeros(bank.filter_length, dtype=bank.dtype),
                    delta_f=bank.delta_f, copy=False,
                )
                fd_decompress(
                    amplitude, phase, frequencies, out=output,
                    f_lower=bank.f_lower, interpolation=override or method,
                )
                np.testing.assert_array_equal(actual[index], output.numpy())


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


@pytest.mark.parametrize("method,override", [
    ("nearest", None),
    ("inline_linear", "nearest"),
])
def test_bank_batch_rejects_unsupported_interpolation(
    tmp_path, method, override
):
    with h5py.File(tmp_path / "compressed.hdf", "w") as filehandler:
        bank, _, _, _ = _compressed_bank(filehandler, (method,), override)
        with JAXScheme(), pytest.raises(NotImplementedError, match="nearest"):
            bank._decompress_batch_jax([0])
        assert not bank._template_cache
