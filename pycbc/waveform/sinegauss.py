""" Generation of sine-Gaussian bursty type things
"""

import functools
import math
from math import pi

import numpy

import pycbc.scheme as _scheme
import pycbc.types


@functools.lru_cache(maxsize=128)
def cached_arange(kmax, delta_f):
    return numpy.arange(0, kmax) * delta_f


@functools.lru_cache(maxsize=128)
def _cached_torch_arange(kmax, delta_f, device, dtype):
    import torch

    return torch.arange(kmax, device=device, dtype=dtype) * delta_f


@functools.lru_cache(maxsize=128)
def _cached_torch_time_grid(length, delta_t, device, dtype):
    import torch

    midpoint = (length - 1) // 2
    return (
        torch.arange(length, device=device, dtype=dtype) - midpoint
    ) * delta_t


def _validate_td_parameters(
    quality,
    central_frequency,
    hrss,
    eccentricity,
    phase,
    delta_t,
):
    parameters = {
        "quality": quality,
        "central_frequency": central_frequency,
        "hrss": hrss,
        "eccentricity": eccentricity,
        "phase": phase,
        "delta_t": delta_t,
    }
    parameters = {name: float(value) for name, value in parameters.items()}

    if not all(math.isfinite(value) for value in parameters.values()):
        raise ValueError("sine-Gaussian parameters must be finite")
    if parameters["quality"] <= 0:
        raise ValueError("quality must be positive")
    if parameters["central_frequency"] <= 0:
        raise ValueError("central_frequency must be positive")
    if parameters["hrss"] < 0:
        raise ValueError("hrss must be non-negative")
    if not 0 <= parameters["eccentricity"] <= 1:
        raise ValueError("eccentricity must be between 0 and 1")
    if parameters["delta_t"] <= 0:
        raise ValueError("delta_t must be positive")

    return parameters


def _td_normalizations(quality, central_frequency, hrss, eccentricity, phase):
    exp_q = math.exp(-(quality * quality))
    scale = quality / (4 * central_frequency * math.sqrt(pi))
    cosine_sq = scale * (1 + exp_q)
    sine_sq = scale * (1 - exp_q)

    semimajor = 1 / math.sqrt(2 - eccentricity * eccentricity)
    semiminor = semimajor * math.sqrt(1 - eccentricity * eccentricity)
    cos_phase = math.cos(phase)
    sin_phase = math.sin(phase)
    h0_plus = hrss * semimajor / math.sqrt(
        cosine_sq * cos_phase * cos_phase
        + sine_sq * sin_phase * sin_phase
    )
    h0_cross = hrss * semiminor / math.sqrt(
        cosine_sq * sin_phase * sin_phase
        + sine_sq * cos_phase * cos_phase
    )
    return h0_plus, h0_cross


def td_sine_gaussian(
    quality,
    central_frequency,
    hrss,
    delta_t,
    eccentricity=0,
    phase=0,
):
    """Generate an elliptically polarized time-domain sine-Gaussian.

    The normalization, 21-sigma sample window, and Tukey taper match
    ``lalsimulation.SimBurstSineGaussian``. Under a Torch scheme all bulk
    waveform construction stays on the selected device.
    """
    p = _validate_td_parameters(
        quality,
        central_frequency,
        hrss,
        eccentricity,
        phase,
        delta_t,
    )
    quality = p["quality"]
    central_frequency = p["central_frequency"]
    hrss = p["hrss"]
    eccentricity = p["eccentricity"]
    phase = p["phase"]
    delta_t = p["delta_t"]

    duration = quality / (2 * pi * central_frequency)
    half_length = math.floor(21 * duration / delta_t / 2)
    length = 2 * half_length + 1
    epoch = -half_length * delta_t
    h0_plus, h0_cross = _td_normalizations(
        quality,
        central_frequency,
        hrss,
        eccentricity,
        phase,
    )

    using_torch = isinstance(_scheme.mgr.state, _scheme.TorchScheme)
    if using_torch:
        import torch

        device = _scheme.mgr.state.torch_device
        dtype = torch.float32 if device.type == "mps" else torch.float64
        time = _cached_torch_time_grid(length, delta_t, device, dtype)
        if length == 1:
            window = torch.ones(1, device=device, dtype=dtype)
        else:
            coordinate = (time / (half_length * delta_t)).abs()
            tapered = torch.sin(pi * (coordinate - 1)).square()
            window = torch.where(coordinate >= 0.5, tapered, 1.0)
        angle = 2 * pi * central_frequency * time
        envelope = torch.exp(-(angle * angle) / (2 * quality * quality))
        envelope *= window
        hplus = h0_plus * envelope * torch.cos(angle - phase)
        hcross = h0_cross * envelope * torch.sin(angle - phase)

        from pycbc.types.array_torch import TorchArrayData

        hplus = TorchArrayData(hplus)
        hcross = TorchArrayData(hcross)
    else:
        sample = numpy.arange(length)
        time = (sample - half_length) * delta_t
        if length == 1:
            window = numpy.ones(1)
        else:
            coordinate = numpy.abs(2 * sample / (length - 1) - 1)
            window = numpy.where(
                coordinate >= 0.5,
                numpy.sin(pi * (coordinate - 1)) ** 2,
                1.0,
            )
        angle = 2 * pi * central_frequency * time
        envelope = numpy.exp(-(angle * angle) / (2 * quality * quality))
        envelope *= window
        hplus = h0_plus * envelope * numpy.cos(angle - phase)
        hcross = h0_cross * envelope * numpy.sin(angle - phase)

    return (
        pycbc.types.TimeSeries(
            hplus, delta_t=delta_t, epoch=epoch, copy=False
        ),
        pycbc.types.TimeSeries(
            hcross, delta_t=delta_t, epoch=epoch, copy=False
        ),
    )


def fd_sine_gaussian(amp, quality, central_frequency, fmin, fmax, delta_f):
    """ Generate a Fourier domain sine-Gaussian

    Parameters
    ----------
    amp: float
        Amplitude of the sine-Gaussian
    quality: float
        The quality factor
    central_frequency: float
        The central frequency of the sine-Gaussian
    fmin: float
        The minimum frequency to generate the sine-Gaussian. This determines
        the length of the output vector.
    fmax: float
        The maximum frequency to generate the sine-Gaussian
    delta_f: float
        The size of the frequency step

    Returns
    -------
    sg: pycbc.types.Frequencyseries
        A Fourier domain sine-Gaussian
    """
    # Optimization note: Ian has profiled and done optimization on this
    # function. If further speed up is needed caching the v vector and
    # avoiding a numpy.zeros call would be the next thing to speed up.
    # After that the numpy.exp call dominates.
    kmin = int(round(fmin / delta_f))
    kmax = int(round(fmax / delta_f))

    tau = (quality / (2 * pi * central_frequency))
    quality_sq = quality**2

    using_torch = isinstance(_scheme.mgr.state, _scheme.TorchScheme)
    if using_torch:
        import torch

        device = _scheme.mgr.state.torch_device
        dtype = torch.float32 if device.type == "mps" else torch.float64
        f = _cached_torch_arange(
            kmax,
            float(delta_f),
            device,
            dtype,
        )
        complex_dtype = (
            torch.complex64 if dtype == torch.float32 else torch.complex128
        )
        v = torch.zeros(kmax, dtype=complex_dtype, device=device)
        exponential = torch.exp
    else:
        f = cached_arange(kmax, delta_f)
        v = numpy.zeros(kmax, dtype=numpy.complex128)
        exponential = numpy.exp

    # exp(exp_term1) and exp(exp_term2) are often 0 (at double-precision
    # level) but still slow to compute. Want to shortcut this by not
    # computing terms at values where we don't need to. Use e**(-50) ~ 0 as
    # the point at which we no longer compute np.exp. Given that the maximum
    # value is O(1), e**(-50) / e**(-1) is 0 at double precision and this is
    # safe.

    # We first figure out which points we need to compute amplitudes for
    exp_term_cutoff = -50

    # Find frequencies at which first term is equal to exp_term_cutoff
    low_freq_first_term = (
        central_frequency - (-exp_term_cutoff)**0.5 / (tau * pi)
    )
    high_freq_first_term = (
        central_frequency + (-exp_term_cutoff)**0.5 / (tau * pi)
    )
    low_freq_first_idx = max(kmin, int(low_freq_first_term//delta_f))
    high_freq_first_idx = min(kmax, int(high_freq_first_term//delta_f))
    # Find frequency at which second term drops to exp_term_cutoff
    high_freq_second_idx = (
        int(-exp_term_cutoff / quality_sq * central_frequency // delta_f)
    )

    exp_term_1 = -(
        tau * pi *
        (f[low_freq_first_idx:high_freq_first_idx] - central_frequency)
    )**2.0

    A_term = amp * (pi**0.5) / 2 * tau

    v[low_freq_first_idx:high_freq_first_idx] = (
        A_term * exponential(exp_term_1)
    )
    # If the first term is already less than e**50 don't need the second
    # term at all ... It's often the case that the second term is not needed.
    if high_freq_second_idx > kmin:
        exp_term_2 = (
            -quality_sq * f[kmin:high_freq_second_idx] / central_frequency
        )
        v[kmin:high_freq_second_idx] *= (1 + exponential(exp_term_2))

    if using_torch:
        from pycbc.types.array_torch import TorchArrayData

        v = TorchArrayData(v)

    return pycbc.types.FrequencySeries(v, delta_f=delta_f, copy=False)
