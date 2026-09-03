"""Utilities for introducing nonlinear tidal effects into waveforms."""

import numpy

import pycbc.conversions
from pycbc.constants import PI


def _torch_module(value):
    """Return torch for a tensor without importing it on NumPy-only paths."""
    if not (hasattr(value, "device") and hasattr(value, "new_ones")):
        return None
    import torch

    return torch if isinstance(value, torch.Tensor) else None

def nltides_fourier_phase_difference(f, delta_f, f0, amplitude, n, m1, m2):
    r"""Calculate the change to the Fourier phase change due
    to non-linear tides. Note that the Fourier phase Psi(f)
    is not the same as the gravitational-wave phase phi(f) and
    is computed by
    Delta Psi(f) = 2 \pi f Delta t(f) - Delta phi(f)

    Parameters
    ----------
    f: numpy.ndarray or torch.Tensor
        Array of frequency values to calculate the fourier phase difference
    delta_f: float
        Frequency resolution of f array
    f0: float
        Frequency that NL effects switch on
    amplitude: float
        Amplitude of effect
    n: float
        Growth dependence of effect
    m1: float
        Mass of component 1
    m2: float
        Mass of component 2

    Returns
    -------
    delta_psi: numpy.ndarray or torch.Tensor
        Fourier phase as a function of frequency, on the same backend as ``f``
    """

    kmin = int(f0/delta_f)
    kmax = len(f)

    f_ref, t_of_f_factor, phi_of_f_factor = \
        pycbc.conversions.nltides_coefs(amplitude, n, m1, m2)

    torch = _torch_module(f)

    # Fourier phase shift below f0 from \Delta \phi(f)
    if torch is None:
        delta_psi_f_le_f0 = numpy.ones(kmin)
    else:
        delta_psi_f_le_f0 = f.new_ones(kmin)
    delta_psi_f_le_f0 *= - phi_of_f_factor * (f0/f_ref)**(n-3.)

    # Fourier phase shift above f0 from \Delta \phi(f)
    delta_psi_f_gt_f0 = - phi_of_f_factor * (f[kmin:kmax]/f_ref)**(n-3.)

    # Fourier phase shift below f0 from 2 pi f \Delta t(f)
    delta_psi_f_le_f0 += 2.0 * PI * f[0:kmin] * t_of_f_factor * \
        (f0/f_ref)**(n-4.)

    # Fourier phase shift above f0 from 2 pi f \Delta t(f)
    delta_psi_f_gt_f0 += 2.0 * PI * f[kmin:kmax] * t_of_f_factor * \
        (f[kmin:kmax]/f_ref)**(n-4.)

    # Return the shift to the Fourier phase
    phase_segments = (delta_psi_f_le_f0, delta_psi_f_gt_f0)
    if torch is None:
        return numpy.concatenate(phase_segments, axis=0)
    return torch.cat(phase_segments, dim=0)


def nonlinear_tidal_spa(**kwds):
    """Generates a frequency-domain waveform that implements the
    TaylorF2+NL tide model described in https://arxiv.org/abs/1808.07013
    """

    from pycbc import waveform
    from pycbc.types import Array

    # We start with the standard TaylorF2 based waveform
    kwds.pop('approximant')
    hp, hc = waveform.get_fd_waveform(approximant="TaylorF2", **kwds)

    # Add the phasing difference from the nonlinear tides. Build the frequency
    # grid and correction on the waveform device when TaylorF2 returned Torch
    # storage, including when the base model used the LAL fallback.
    tensor = getattr(hp._data, "tensor", None)
    if tensor is None:
        f = numpy.arange(len(hp)) * hp.delta_f
        phase_difference = nltides_fourier_phase_difference(
            f, hp.delta_f, kwds['f0'], kwds['amplitude'], kwds['n'],
            kwds['mass1'], kwds['mass2'])
        pd = Array(numpy.exp(-1.0j * phase_difference), dtype=hp.dtype)
    else:
        import torch
        from pycbc.types.array_torch import TorchArrayData

        f = torch.arange(
            len(hp), dtype=tensor.real.dtype, device=tensor.device
        ) * hp.delta_f
        phase_difference = nltides_fourier_phase_difference(
            f, hp.delta_f, kwds['f0'], kwds['amplitude'], kwds['n'],
            kwds['mass1'], kwds['mass2'])
        correction = torch.polar(
            torch.ones_like(phase_difference), -phase_difference
        ).to(tensor.dtype)
        pd = Array(TorchArrayData(correction), copy=False)
    hp *= pd
    hc *= pd
    return hp, hc
