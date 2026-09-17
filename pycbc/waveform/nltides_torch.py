# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch backend for the legacy ``TaylorF2NL`` nonlinear-tide model."""

import pycbc.conversions
from pycbc.constants import PI


def nltides_fourier_phase_difference(frequencies, delta_f, f0, amplitude, n, m1, m2):
    """Calculate the nonlinear-tide Fourier phase with Torch."""
    import torch

    kmin = int(f0 / delta_f)
    kmax = len(frequencies)
    f_ref, t_of_f_factor, phi_of_f_factor = pycbc.conversions.nltides_coefs(
        amplitude, n, m1, m2
    )

    below = frequencies.new_ones(kmin)
    below *= -phi_of_f_factor * (f0 / f_ref) ** (n - 3.0)
    above_frequencies = frequencies[kmin:kmax]
    above = -phi_of_f_factor * (above_frequencies / f_ref) ** (n - 3.0)

    below += 2.0 * PI * frequencies[:kmin] * t_of_f_factor * (f0 / f_ref) ** (n - 4.0)
    above += (
        2.0
        * PI
        * above_frequencies
        * t_of_f_factor
        * (above_frequencies / f_ref) ** (n - 4.0)
    )
    return torch.cat((below, above), dim=0)


def nonlinear_tidal_spa(**kwds):
    """Generate ``TaylorF2NL`` with a device-native Torch correction."""
    import torch

    from pycbc import waveform
    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData
    from pycbc.types.backend import backend_array

    kwds.pop("approximant")
    hp, hc = waveform.get_fd_waveform(approximant="TaylorF2", **kwds)
    tensor = backend_array(hp, "torch")
    frequencies = (
        torch.arange(len(hp), dtype=tensor.real.dtype, device=tensor.device)
        * hp.delta_f
    )
    phase_difference = nltides_fourier_phase_difference(
        frequencies,
        hp.delta_f,
        kwds["f0"],
        kwds["amplitude"],
        kwds["n"],
        kwds["mass1"],
        kwds["mass2"],
    )
    correction = torch.polar(torch.ones_like(phase_difference), -phase_difference).to(
        tensor.dtype
    )
    phase = Array(TorchArrayData(correction), copy=False)
    hp *= phase
    hc *= phase
    return hp, hc


__all__ = (
    "nltides_fourier_phase_difference",
    "nonlinear_tidal_spa",
)
