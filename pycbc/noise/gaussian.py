# Copyright (C) 2012  Alex Nitz
#
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""This module contains functions to generate gaussian noise colored with a
noise spectrum.
"""

import lal
from pycbc import libutils
from pycbc.types import TimeSeries, zeros
from pycbc.types import complex_same_precision_as, FrequencySeries
import pycbc
from pycbc import lal_compat as lal
import numpy.random
import pycbc
from pycbc import libutils
from pycbc.types import FrequencySeries, TimeSeries, complex_same_precision_as, zeros

try:
    import torch
    from pycbc.types.array_torch import TorchArrayData
    _HAVE_TORCH = pycbc.HAVE_TORCH
except Exception:  # pragma: no cover - torch optional
    torch = None
    TorchArrayData = None
    _HAVE_TORCH = False

lalsimulation = libutils.import_optional(
    'lalsimulation', defer=libutils.defer_lalsimulation_import()
)

def frequency_noise_from_psd(psd, seed=None):
    """Create noise with a given psd.

    Return noise coloured with the given psd. The returned noise
    FrequencySeries has the same length and frequency step as the given psd.
    Note that if unique noise is desired a unique seed should be provided.

    Parameters
    ----------
    psd : FrequencySeries
        The noise weighting to color the noise.
    seed : {0, int} or None
        The seed to generate the noise. If None specified,
        the seed will not be reset.

    Returns
    --------
    noise : FrequencySeriesSeries
        A FrequencySeries containing gaussian noise colored by the given psd.
    """
    sigma = 0.5 * (psd / psd.delta_f) ** (0.5)
    if _HAVE_TORCH and isinstance(getattr(psd, "_data", None), TorchArrayData):
        sigma_t = sigma._data.tensor
        device = sigma_t.device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        noise_re = torch.zeros_like(sigma_t)
        noise_im = torch.zeros_like(sigma_t)
        mask = sigma_t != 0
        sigma_red = sigma_t[mask]
        noise_re_red = torch.randn(
            sigma_red.shape, dtype=sigma_red.dtype,
            device=device, generator=generator,
        ) * sigma_red
        noise_im_red = torch.randn(
            sigma_red.shape, dtype=sigma_red.dtype,
            device=device, generator=generator,
        ) * sigma_red
        noise_re[mask] = noise_re_red
        noise_im[mask] = noise_im_red
        noise = torch.complex(noise_re, noise_im)
        return FrequencySeries(TorchArrayData(noise), delta_f=psd.delta_f,
                               copy=False)

    if seed is not None:
        numpy.random.seed(seed)
    sigma_np = sigma.numpy()
    dtype = complex_same_precision_as(psd)

    not_zero = (sigma_np != 0)

    sigma_red = sigma_np[not_zero]
    noise_re = numpy.random.normal(0, sigma_red)
    noise_co = numpy.random.normal(0, sigma_red)
    noise_red = noise_re + 1j * noise_co

    noise = numpy.zeros(len(sigma_np), dtype=dtype)
    noise[not_zero] = noise_red

    return FrequencySeries(noise, delta_f=psd.delta_f, dtype=dtype)



def _torch_noise_segment(sigma, length, delta_f, generator):
    """Generate one periodic colored-noise segment on a Torch device."""
    noise_re = torch.randn(
        sigma.shape, dtype=sigma.dtype, device=sigma.device,
        generator=generator,
    )
    noise_im = torch.randn(
        sigma.shape, dtype=sigma.dtype, device=sigma.device,
        generator=generator,
    )
    stilde = torch.complex(noise_re * sigma, noise_im * sigma)

    # LAL's frequency-to-time transform is unnormalised and then multiplied
    # by delta_f. torch.fft.irfft includes a 1 / length normalisation.
    return torch.fft.irfft(stilde, n=length) * (length * delta_f)


def _torch_noise_from_psd(length, delta_t, psd, seed):
    """Torch implementation of LALSimNoise's overlap-and-feather method."""
    length = int(length)
    samples_per_segment = 1.0 / delta_t / psd.delta_f
    segment_length = int(samples_per_segment)
    frequency_length = segment_length // 2 + 1
    stride = segment_length // 2

    if segment_length < 2:
        raise ValueError("PSD and delta_t must define at least two samples")
    if (int(samples_per_segment + 0.5) != segment_length
            or frequency_length > len(psd)):
        raise ValueError("PSD not compatible with requested delta_t")

    psd_tensor = psd._data.tensor
    generator = torch.Generator(device=psd_tensor.device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed))

    sigma = 0.5 * torch.sqrt(
        psd_tensor[:frequency_length] / psd.delta_f
    )
    sigma = sigma.clone()
    sigma[0] = 0
    sigma[-1] = 0

    segment = _torch_noise_segment(
        sigma, segment_length, psd.delta_f, generator
    )
    output = torch.empty(
        length, dtype=psd_tensor.dtype, device=psd_tensor.device
    )

    overlap_length = segment_length - stride
    overlap_index = torch.arange(
        overlap_length, dtype=psd_tensor.dtype, device=psd_tensor.device
    )
    phase = overlap_index * (torch.pi / (2.0 * overlap_length))
    old_weight = torch.cos(phase)
    new_weight = torch.sin(phase)

    length_generated = 0
    while length_generated < length:
        copy_length = min(stride, length - length_generated)
        output[
            length_generated:length_generated + copy_length
        ] = segment[:copy_length]
        length_generated += stride

        if length_generated < length:
            new_segment = _torch_noise_segment(
                sigma, segment_length, psd.delta_f, generator
            )
            new_segment[:overlap_length] = (
                old_weight * segment[stride:]
                + new_weight * new_segment[:overlap_length]
            )
            segment = new_segment

    return TimeSeries(
        TorchArrayData(output), delta_t=delta_t, copy=False
    )


def noise_from_psd(length, delta_t, psd, seed=None):
    """Create noise with a given psd.

    Return noise with a given psd. Note that if unique noise is desired
    a unique seed should be provided.

    Parameters
    ----------
    length : int
        The length of noise to generate in samples.
    delta_t : float
        The time step of the noise.
    psd : FrequencySeries
        The noise weighting to color the noise.
    seed : {None, int}
        The seed to generate the noise. If ``None``, a random seed is used.
        Torch and LAL use different random number generators, so their seeded
        realizations are not identical.

    Returns
    --------
    noise : TimeSeries
        A TimeSeries containing gaussian noise colored by the given psd.
    """
    use_torch = _HAVE_TORCH and isinstance(
        getattr(psd, "_data", None), TorchArrayData
    )
    if use_torch:
        return _torch_noise_from_psd(length, delta_t, psd, seed)

    try:
        sim_noise = lalsimulation.SimNoise
    except ImportError as exc:
        raise ImportError(
            "CPU noise_from_psd requires lalsimulation; pass a Torch-backed "
            "PSD under TorchScheme to use the native Torch noise generator."
        ) from exc

    noise_ts = TimeSeries(zeros(length, dtype=psd.dtype), delta_t=delta_t)

    if seed is None:
        seed = numpy.random.randint(2**32)

    randomness = lal.gsl_rng("ranlux", seed)

    N = round(1.0 / delta_t / psd.delta_f)
    n = N // 2 + 1
    stride = N // 2

    if n > len(psd):
        raise ValueError("PSD not compatible with requested delta_t")

    psd_lal = (psd[0:n]).lal()
    psd_lal.data.data[n - 1] = 0
    psd_lal.data.data[0] = 0

    segment = TimeSeries(zeros(N), delta_t=delta_t).lal()
    length_generated = 0

    sim_noise(segment, 0, psd_lal, randomness)
    while length_generated < length:
        if (length_generated + stride) < length:
            noise_ts.data[length_generated : length_generated + stride] = (
                segment.data.data[0:stride]
            )
        else:
            noise_ts.data[length_generated:length] = segment.data.data[
                0 : length - length_generated
            ]

        length_generated += stride
        sim_noise(segment, stride, psd_lal, randomness)

    return noise_ts


def noise_from_string(psd_name, length, delta_t, seed=None, low_frequency_cutoff=10.0):
    """Create noise from an analytic PSD

    Return noise from the chosen PSD. Note that if unique noise is desired
    a unique seed should be provided.

    Parameters
    ----------
    psd_name : str
        Name of the analytic PSD to use.
    low_fr
    length : int
        The length of noise to generate in samples.
    delta_t : float
        The time step of the noise.
    seed : {None, int}
        The seed to generate the noise.
    low_frequency_cutof : {10.0, float}
        The low frequency cutoff to pass to the PSD generation.

    Returns
    --------
    noise : TimeSeries
        A TimeSeries containing gaussian noise colored by the given psd.
    """
    import pycbc.psd

    # We just need enough resolution to resolve lines
    delta_f = 1.0 / 8
    flen = round(0.5 / delta_t / delta_f) + 1
    psd = pycbc.psd.from_string(psd_name, flen, delta_f, low_frequency_cutoff)
    return noise_from_psd(int(length), delta_t, psd, seed=seed)
