# Copyright (C) 2025
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Pure torch implementation of the SPA template engine.

This mirrors the CUDA and CPU implementations (fall-through switch structure for
phase terms and f^-7/6 amplitude) but keeps computation entirely on the active
torch device. CPU routes remain available via other scheme backends.
"""

import torch
import numpy as _np
import pycbc.scheme as _scheme
from pycbc.waveform import spa_tmplt as _spa_cpu


def spa_tmplt_engine(htilde, kmin, phase_order, delta_f, piM, pfaN,
                     pfa2, pfa3, pfa4, pfa5, pfl5, pfa6, pfl6, pfa7,
                     amp_factor):
    """
    Torch implementation of the SPA template kernel.

    Parameters mirror the CPU/CUDA backends; ``htilde`` is a complex Array
    allocated by the caller. The function fills it in-place.
    """
    tensor = htilde._data.tensor
    device = tensor.device
    dtype = tensor.dtype

    piM13 = _np.cbrt(piM)
    logpiM13 = _np.log(piM13)
    log4 = _np.log(4.0)
    two_pi = 2.0 * _np.pi

    idx = torch.arange(tensor.numel(), device=device, dtype=torch.float64)
    f_phase = torch.clamp((idx + float(kmin)) * float(delta_f), min=1e-12)
    f_amp = torch.clamp((idx + float(kmin) + 1.0) * float(delta_f), min=1e-12)

    v = piM13 * torch.pow(f_phase, 1.0 / 3.0)
    logv = torch.log(f_phase) * (1.0 / 3.0) + logpiM13
    amp = amp_factor * torch.pow(f_amp, -7.0 / 6.0)
    amp = amp.to(dtype=torch.float32 if dtype == torch.complex64 else torch.float64)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3

    # Follow the same nested phasing as Cython (phase_order ignored in CPU path).
    phasing = pfa7 * v
    phasing = (phasing + pfa6 + pfl6 * (logv + log4)) * v
    phasing = (phasing + pfa5 + pfl5 * logv) * v
    phasing = (phasing + pfa4) * v
    phasing = (phasing + pfa3) * v
    phasing = (phasing + pfa2) * v2 + 1.0

    phasing = phasing * (pfaN / (v5 + 1e-30)) - (_np.pi / 4.0)
    phasing = phasing - torch.floor(phasing / two_pi) * two_pi

    phasing = torch.where(phasing < -_np.pi, phasing + two_pi, phasing)
    phasing = torch.where(phasing > _np.pi, phasing - two_pi, phasing)

    # Polynomial sin/cos approximations used by CPU implementation
    sinp = 1.273239545 * phasing - 0.405284735 * phasing * torch.abs(phasing)
    sinp = 0.225 * (sinp * torch.abs(sinp) - sinp) + sinp

    phasing_shift = phasing + (_np.pi / 2.0)
    phasing_shift = torch.where(phasing_shift > _np.pi, phasing_shift - two_pi, phasing_shift)

    cosp = 1.273239545 * phasing_shift - 0.405284735 * phasing_shift * torch.abs(phasing_shift)
    cosp = 0.225 * (cosp * torch.abs(cosp) - cosp) + cosp

    real = cosp * amp
    imag = -sinp * amp
    real = real.to(dtype=torch.float32 if dtype == torch.complex64 else torch.float64)
    imag = imag.to(dtype=real.dtype)
    out = torch.complex(real, imag).to(dtype=dtype)
    tensor.copy_(out)

    return None
