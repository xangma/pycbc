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
"""GPU-accelerated strain conditioning and autogating routines for PyCBC."""

import numpy
import pycbc.events
import pycbc.psd
import pycbc.types
from pycbc.filter.resample import resample_to_delta_t
from pycbc.strain.strain import _linear_tapers_for_series, next_power_of_2
from pycbc.types import TimeSeries


def detect_loud_glitches_torch(
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
    """Automatic identification of loud transients for gating purposes on GPU.

    Performs conditioning, zero-padding, Welch PSD estimation, whitening,
    and FindChirp peak clustering on the active TorchScheme accelerator in
    double precision (float64/complex128). Operates on an isolated data copy to
    ensure the caller's strain TimeSeries is not mutated in place, guaranteeing
    bit-exact scientific parity with the reference CPU implementation.
    """
    if high_freq_cutoff:
        s = resample_to_delta_t(strain, 0.5 / high_freq_cutoff, method="ldas")
    else:
        # Create an isolated float64 TimeSeries on host without calling
        # methods decorated with @_convert (which mutate caller's strain)
        raw_data = strain._data
        if hasattr(raw_data, "numpy"):
            data_np = raw_data.numpy()
        elif hasattr(raw_data, "cpu"):
            data_np = raw_data.cpu().numpy()
        else:
            data_np = numpy.asarray(raw_data)
        s = TimeSeries(
            numpy.array(data_np, dtype=numpy.float64, copy=True),
            delta_t=strain.delta_t,
            epoch=strain.start_time,
        )

    # taper strain
    corrupt_length = int(corrupt_time * s.sample_rate)
    rising_taper, falling_taper = _linear_tapers_for_series(s, corrupt_length)
    s[0:corrupt_length] *= rising_taper
    s[(len(s) - corrupt_length):] *= falling_taper

    if output_intermediates:
        s.save_to_wav("strain_conditioned.wav")

    # zero-pad strain to a power-of-2 length on GPU
    strain_pad_length = next_power_of_2(len(s))
    pad_start = int(strain_pad_length / 2 - len(s) / 2)
    pad_end = pad_start + len(s)
    pad_epoch = s.start_time - pad_start / float(s.sample_rate)
    strain_pad = pycbc.types.TimeSeries(
        pycbc.types.zeros(strain_pad_length, dtype=numpy.float64),
        delta_t=s.delta_t,
        copy=False,
        epoch=pad_epoch,
    )
    strain_pad[pad_start:pad_end] = s[:]

    # estimate the PSD on GPU in double precision
    psd = pycbc.psd.welch(
        s[corrupt_length:(len(s) - corrupt_length)],
        seg_len=int(psd_duration * s.sample_rate),
        seg_stride=int(psd_stride * s.sample_rate),
        avg_method=psd_avg_method,
        require_exact_data_fit=False,
    )
    psd = pycbc.psd.interpolate(psd, 1.0 / strain_pad.duration)
    psd = pycbc.psd.inverse_spectrum_truncation(
        psd,
        int(psd_duration * s.sample_rate),
        low_frequency_cutoff=low_freq_cutoff,
        trunc_method="hann",
    )
    kmin = int(low_freq_cutoff / psd.delta_f)
    psd[0:kmin] = numpy.inf
    if high_freq_cutoff:
        kmax = int(high_freq_cutoff / psd.delta_f)
        psd[kmax:] = numpy.inf

    # whiten on GPU
    strain_tilde = strain_pad.to_frequencyseries()

    if high_freq_cutoff:
        norm = high_freq_cutoff - low_freq_cutoff
    else:
        norm = s.sample_rate / 2.0 - low_freq_cutoff
    strain_tilde *= (psd * norm) ** (-0.5)

    strain_pad = strain_tilde.to_timeseries()

    if output_intermediates:
        strain_pad[pad_start:pad_end].save_to_wav("strain_whitened.wav")

    mag = abs(strain_pad[pad_start:pad_end])

    if output_intermediates:
        mag.save("strain_whitened_mag.npy")

    # remove strain corrupted by filters at the ends
    if corrupt_length:
        mag[0:corrupt_length] = 0
        mag[len(mag) - corrupt_length:] = 0

    # find peaks and their times
    cluster_samples = int(cluster_window * s.sample_rate)
    indices, _ = pycbc.events.threshold_real_and_cluster_findchirp(
        mag, threshold, cluster_samples
    )
    times = [idx * s.delta_t + s.start_time for idx in indices]

    return times
