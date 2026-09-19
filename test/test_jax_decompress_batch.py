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
from pycbc.waveform.decompress_jax import batched_inline_linear_interp_jax, _grid_indices


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
        last_idx = int(_grid_indices(freq[-1], df))
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
