# Torch-native SpinTaylorF2 generator (current: aligned-spin).
# Precession port in progress: see _precession_factors stub. Once complete,
# this module will fully mirror the CUDA kernel (alpha/zeta evolution and
# sideband modulation) on torch without CPU/PyCUDA.

import numpy as _np
import torch

from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData


def _pn_coeffs(eta):
    pfa2 = 0.0
    pfa3 = -16.0 * _np.pi
    pfa4 = 10.0 * (3058673.0 / 1016064.0 + 5429.0 * eta / 1008.0 + 617.0 * eta * eta / 144.0)
    pfa5 = -10.0 * (7729.0 / 1016064.0 + 3.0 * eta * eta / 8.0) * _np.pi
    pfl5 = 0.0
    pfa6 = 10.0 * (
        11583231236531.0 / 4694215680.0
        + 64969.0 * eta / 708.0
        + 64.0 * _np.pi * _np.pi / 3.0
        - 6848.0 * _np.pi / 21.0
    )
    pfa6 -= 10.0 * (15737765635.0 / 3048192.0 + 2255.0 * _np.pi * _np.pi / 12.0) * eta
    pfl6 = -6848.0 / 21.0
    pfa7 = 10.0 * _np.pi * (77096675.0 / 254016.0 + 378515.0 * eta / 1512.0 - 74045.0 * eta * eta / 756.0)
    return pfa2, pfa3, pfa4, pfa5, pfl5, pfa6, pfl6, pfa7


def _precession_factors(*args, **kwargs):
    """Placeholder for full precession (alpha/zeta, sidebands) torch port.

    TODO:
    - Implement alpha/zeta evolution using gamma0/kappa as in the CUDA kernel.
    - Compute RE/IM_SBfac* sideband factors and apply modulation to hplus/hcross.
    - Validate against CPU/PyCUDA across a grid of generic spin directions.
    """
    raise NotImplementedError("Precession factors not yet implemented in torch")


def spintaylorf2_torch(**kwds):
    f_lower = kwds["f_lower"]
    delta_f = kwds["delta_f"]
    distance = kwds["distance"]
    mass1 = kwds["mass1"]
    mass2 = kwds["mass2"]
    phi0 = kwds["coa_phase"]
    phase_order = int(kwds["phase_order"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    tC = -1.0 / delta_f
    M = mass1 + mass2
    eta = mass1 * mass2 / (M * M)
    m_sec = M * 4.92549095e-6  # MTSUN_SI
    piM = _np.pi * m_sec

    vISCO = 1.0 / _np.sqrt(6.0)
    fISCO = vISCO**3 / piM
    n = int(_np.ceil(fISCO / delta_f) + 1)
    kmax = int(fISCO / delta_f)
    kmin = int(_np.ceil(f_lower / delta_f))
    kmax = kmax if kmax < n else n

    idx = torch.arange(kmax - kmin, device=device, dtype=dtype)
    freqs = (idx + kmin) * delta_f

    v = torch.pow(piM * freqs, 1.0 / 3.0)
    v2 = v * v
    v3 = v * v2
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3
    v7 = v3 * v4

    pfa2, pfa3, pfa4, pfa5, pfl5, pfa6, pfl6, pfa7 = _pn_coeffs(eta)

    phasing = torch.zeros_like(v)
    phasing += pfa7 * v7
    phasing += (pfa6 + pfl6 * torch.log(4.0 * v)) * v6
    v0 = torch.tensor(piM * kmin * delta_f, device=device, dtype=dtype)
    phasing += (pfa5 + pfl5 * torch.log(v / torch.pow(v0, 1.0 / 3.0))) * v5
    phasing += pfa4 * v4
    phasing += pfa3 * v3
    phasing += pfa2 * v2
    phasing += 1.0

    phasing = phasing / torch.clamp(v5, min=1e-20)
    phasing += phi0 - 2.0 * _np.pi * freqs * tC

    amp = torch.pow(freqs, -7.0 / 6.0) / distance
    h = amp * torch.exp(-1j * phasing)

    fs = FrequencySeries(TorchArrayData(h.to(torch.complex128)), delta_f=delta_f, copy=False)
    return fs, fs
