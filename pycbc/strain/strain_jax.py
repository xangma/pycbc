# Copyright (C) 2026 The PyCBC Collaboration
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

"""JAX backend for strain conditioning and autogating routines in PyCBC."""

import numpy as np

import pycbc.events
import pycbc.psd
import pycbc.types
from pycbc.filter.resample import resample_to_delta_t
from pycbc.strain.strain import next_power_of_2


def detect_loud_glitches_jax(
    strain,
    psd_duration=4.0,
    psd_stride=2.0,
    psd_avg_method="median",
    low_freq_cutoff=30.0,
    threshold=50.0,
    cluster_window=5.0,
    corrupt_time=4.0,
    high_freq_cutoff=None,
    output_intermediates=False,
):
    """Automatic identification of loud transients for gating purposes in JAX.

    Performs conditioning, zero-padding, Welch PSD estimation, whitening,
    and FindChirp peak clustering on the active JAXScheme accelerator in
    double precision (float64/complex128). Operates on an isolated data copy
    to ensure bit-exact scientific parity with reference implementations.
    """
    if high_freq_cutoff:
        s = resample_to_delta_t(strain, 0.5 / high_freq_cutoff, method="ldas")
    else:
        s = strain.copy()
        if s.dtype != np.float64:
            s = s.astype(np.float64)

    # Taper strain ends
    corrupt_length = int(corrupt_time * s.sample_rate)
    w = np.arange(corrupt_length) / float(corrupt_length)
    s[0:corrupt_length] *= pycbc.types.Array(w, dtype=s.dtype)
    s[(len(s) - corrupt_length):] *= pycbc.types.Array(w[::-1], dtype=s.dtype)

    if output_intermediates:
        s.save_to_wav("strain_conditioned.wav")

    # Zero-pad strain to a power-of-2 length
    strain_pad_length = next_power_of_2(len(s))
    pad_start = int(strain_pad_length / 2 - len(s) / 2)
    pad_end = pad_start + len(s)
    pad_epoch = s.start_time - pad_start / float(s.sample_rate)
    strain_pad = pycbc.types.TimeSeries(
        pycbc.types.zeros(strain_pad_length, dtype=np.float64),
        delta_t=s.delta_t,
        copy=False,
        epoch=pad_epoch,
    )
    strain_pad[pad_start:pad_end] = s[:]

    # Estimate PSD via Welch method in double precision
    psd = pycbc.psd.welch(
        s[corrupt_length:(len(s) - corrupt_length)],
        seg_len=int(psd_duration * s.sample_rate),
        seg_stride=int(psd_stride * s.sample_rate),
        avg_method=psd_avg_method,
    )
    psd = pycbc.psd.interpolate(psd, strain_pad.delta_f)
    psd = pycbc.psd.inverse_spectrum_truncation(
        psd,
        int(psd_duration * strain_pad.sample_rate),
        low_frequency_cutoff=low_freq_cutoff,
        trunc_method="hann",
    )

    # Whiten the padded strain
    white_strain = (
        strain_pad.to_frequencyseries() / psd ** 0.5
    ).to_timeseries()

    # Unpad and trim corrupted edges
    white_strain = white_strain[pad_start:pad_end]
    white_strain = white_strain[corrupt_length:(len(s) - corrupt_length)]

    mag = np.abs(white_strain.numpy())
    indices = np.where(mag > threshold)[0]
    if len(indices) == 0:
        return [], []

    cluster_window_samples = int(cluster_window * s.sample_rate)
    cluster_idx = pycbc.events.findchirp_cluster_over_window(
        indices.astype(np.int32),
        np.ascontiguousarray(mag[indices], dtype=np.float64),
        cluster_window_samples,
    )
    peak_indices = indices[cluster_idx]
    times = [
        float(white_strain.start_time + idx * white_strain.delta_t)
        for idx in peak_indices
    ]
    snrs = [float(mag[idx]) for idx in peak_indices]

    return times, snrs
