# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Regression and parity tests for frequency-domain waveform decompression under JAX."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pycbc.scheme import CPUScheme, JAXScheme
from pycbc.types import FrequencySeries, zeros
from pycbc.types.backend import backend_array
from pycbc.waveform.compress import fd_decompress

INTERPOLATIONS = (
    "inline_linear",
    "inline_quadratic",
    "inline_cubic",
    "inline_quartic",
)


@pytest.fixture
def sample_data():
    sample_f = np.array([0.0, 0.43, 1.17, 2.05, 3.4, 4.8, 6.35, 8.0])
    df = 0.2
    f_lower = 0.71
    epoch = 1234567890.125
    return sample_f, df, f_lower, epoch


@pytest.mark.parametrize("interp", INTERPOLATIONS)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_jax_decompress_matches_cpu(sample_data, interp, dtype):
    sample_f, df, f_lower, epoch = sample_data
    sample_f = sample_f.astype(dtype)
    amp = (
        np.exp(-0.08 * sample_f) * (1.0 + 0.03 * np.cos(0.7 * sample_f))
    ).astype(dtype)
    phase = (0.13 * sample_f + 0.011 * sample_f**2).astype(dtype)

    complex_dtype = np.complex64 if dtype is np.float32 else np.complex128
    tolerance = (
        dict(rtol=5e-5, atol=5e-6)
        if dtype is np.float32
        else dict(rtol=5e-12, atol=5e-13)
    )

    # Reference on CPU
    with CPUScheme():
        ref_auto = fd_decompress(
            amp,
            phase,
            sample_f,
            df=df,
            f_lower=f_lower,
            interpolation=interp,
        )
        out_len = len(ref_auto) + 7
        ref_out = FrequencySeries(
            np.full(out_len, 17.0 - 4.0j, dtype=complex_dtype),
            delta_f=df,
            epoch=epoch,
            copy=False,
        )
        fd_decompress(
            amp,
            phase,
            sample_f,
            out=ref_out,
            f_lower=f_lower,
            interpolation=interp,
        )

    # Evaluation under JAXScheme
    with JAXScheme():
        actual_auto = fd_decompress(
            amp,
            phase,
            sample_f,
            df=df,
            f_lower=f_lower,
            interpolation=interp,
        )
        assert backend_array(actual_auto, "jax") is not None
        assert actual_auto.delta_f == df
        assert float(actual_auto.epoch) == 0.0

        actual_out = FrequencySeries(
            zeros(len(ref_out), dtype=complex_dtype),
            delta_f=df,
            epoch=epoch,
            copy=False,
        )
        actual_out.data[:] = 17.0 - 4.0j

        ret = fd_decompress(
            amp,
            phase,
            sample_f,
            out=actual_out,
            f_lower=f_lower,
            interpolation=interp,
        )
        assert ret is actual_out
        assert backend_array(actual_out, "jax") is not None
        assert actual_out.delta_f == df
        assert float(actual_out.epoch) == epoch

        np.testing.assert_allclose(
            actual_auto.numpy(),
            ref_auto.numpy(),
            **tolerance,
        )
        np.testing.assert_allclose(
            actual_out.numpy(),
            ref_out.numpy(),
            **tolerance,
        )

        start_index = int(np.ceil(f_lower / df))
        last_index = int(np.floor(sample_f[-1] / df))
        out_vals = actual_out.numpy()
        assert np.all(out_vals[:start_index] == 0)
        assert np.all(out_vals[last_index + 1 :] == 0)
