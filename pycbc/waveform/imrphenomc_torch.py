# Copyright (C) 2025
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
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
Torch-native IMRPhenomC frequency-domain waveform.

This is a direct port of the legacy PyCUDA implementation
(``pycbc.waveform.pycbc_phenomC_tmplt``) rewritten in pure Python/Torch so it
can run entirely within ``TorchScheme`` without calling into lalsimulation or
CPU-only kernels.  Physics and coefficients follow the original PhenomC paper
(arXiv:1005.3306) and the existing PyCBC implementation.

Activation
----------
- Supported calls are native by default on CPU and CUDA.
- Per-model flag: ``PYCBC_IMRPHENOMC_NATIVE=1``
- Global flag   : ``PYCBC_TORCH_NATIVE_PORTS``

Apple MPS remains opt-in because float32 cancellation among the fitted phase
terms can materially reduce accuracy, especially at low dimensionless
frequencies. Explicit flags override the device-aware default. Unsupported
regular-grid requests fall back to lalsimulation; lalsimulation has no
IMRPhenomC sequence implementation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from pycbc import lal_compat as lal
import torch

import pycbc.scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.waveform.imrphenomd_torch import (
    _DEFAULT_ONLY_ORDER_KEYS,
    _NON_GR_KEYS,
    _TIDAL_EXTENSION_KEYS,
    _TRANSVERSE_SPIN_KEYS,
    _is_default_order,
    _is_nonzero,
)

_PI = lal.PI
_MSUN_SI = lal.MSUN_SI
_EULER_GAMMA = lal.GAMMA
_PC_SI = lal.PC_SI
_MRSUN_SI = lal.MRSUN_SI
_MTSUN_SI = lal.MTSUN_SI
_C_SI = lal.C_SI
_G_SI = lal.G_SI


def imrphenomc_native_supported(params) -> bool:
    """Return whether the native implementation covers ``params``.

    Unsupported waveform modifications deliberately retain the public
    lalsimulation path so that they are never accepted and silently ignored.
    """
    if params.get("approximant", "IMRPhenomC") != "IMRPhenomC":
        return False
    if any(
        not _is_default_order(params.get(key, -1))
        for key in (*_DEFAULT_ONLY_ORDER_KEYS, "phase_order", "amplitude_order")
    ):
        return False
    if any(
        _is_nonzero(params.get(key, 0.0))
        for key in (
            _TRANSVERSE_SPIN_KEYS
            + _TIDAL_EXTENSION_KEYS
            + _NON_GR_KEYS
            + (
                "lambda1",
                "lambda2",
                "eccentricity",
                "mean_per_ano",
                "frame_axis",
                "modes_choice",
                "side_bands",
            )
        )
    ):
        return False
    if params.get("mode_array") is not None or params.get("numrel_data", ""):
        return False
    return True


def _final_spin(xi: float, eta: float) -> float:
    """Final BH spin (Eq. 5-6 of arXiv:0710.3345)."""
    s4 = -0.129
    s5 = -0.384
    t0 = -2.686
    t2 = -3.454
    t3 = 2.353
    eta_xi = eta * xi
    eta2 = eta * eta
    finspin = (
        xi
        + s4 * xi * eta_xi
        + s5 * eta_xi * eta
        + t0 * eta_xi
        + 2.0 * (3.0**0.5) * eta
        + t2 * eta2
        + t3 * eta2 * eta
    )
    if abs(finspin) > 1.0:
        raise ValueError("Absolute value of final spin > 1.0. Aborting")
    return finspin


def _f_rd(a: float, mass_msun: float) -> float:
    """Ring-down frequency for the final Kerr BH (Eq. 5.5)."""
    return (_C_SI**3.0 / (2.0 * _PI * _G_SI * mass_msun * _MSUN_SI)) * (
        1.5251 - 1.1568 * (1.0 - a) ** 0.1292
    )


def _q(a: float) -> float:
    """Quality factor of ring-down (Eq. 5.6)."""
    return 0.7 + 1.4187 * (1.0 - a) ** -0.4990


def _distance_scale(m_total_msun: float) -> float:
    """Scale distance from Mpc to the dimensionless factor used in PhenomC."""
    return (
        1.0e6
        * _PC_SI
        / (
            2.0
            * math.sqrt(5.0 / (64.0 * _PI))
            * m_total_msun
            * _MRSUN_SI
            * m_total_msun
            * _MTSUN_SI
        )
    )


def _dst_type_one(values):
    """Return the unnormalised type-I discrete sine transform."""
    size = values.shape[0]
    extended = torch.zeros(
        2 * (size + 1), dtype=values.dtype, device=values.device
    )
    extended[1 : size + 1] = values
    extended[size + 2 :] = -values.flip(0)
    return -torch.fft.fft(extended).imag[1 : size + 1]


def _natural_cubic_derivative(values, spacing, sample_position):
    """Evaluate the derivative of GSL's natural cubic spline.

    ``sample_position`` is expressed in grid intervals from ``values[0]``.
    The spline system is Toeplitz on this uniform frequency grid, so a pair
    of sine transforms solves it without moving phase samples off-device.
    """
    count = values.shape[0]
    if count < 2:
        raise ValueError("IMRPhenomC needs at least two active frequency bins")
    if count == 2:
        return (values[1] - values[0]) / spacing

    rhs = (
        6.0
        * (values[2:] - 2.0 * values[1:-1] + values[:-2])
        / (spacing * spacing)
    )
    modes = torch.arange(
        1, count - 1, dtype=values.dtype, device=values.device
    )
    eigenvalues = 4.0 + 2.0 * torch.cos(_PI * modes / (count - 1))
    transformed = _dst_type_one(rhs) / eigenvalues
    second_internal = _dst_type_one(transformed) / (2.0 * (count - 1))
    second = torch.zeros_like(values)
    second[1:-1] = second_internal

    interval = min(max(math.floor(sample_position), 0), count - 2)
    fraction = min(max(sample_position - interval, 0.0), 1.0)
    left = fraction * spacing
    right = spacing - left
    return (
        (values[interval + 1] - values[interval]) / spacing
        + spacing * (second[interval] - second[interval + 1]) / 6.0
        - second[interval] * right * right / (2.0 * spacing)
        + second[interval + 1] * left * left / (2.0 * spacing)
    )


def imrphenomc_sequence_native_supported(params):
    """Return whether arbitrary-frequency IMRPhenomC is Torch-native."""
    return imrphenomc_native_supported(params)


def imrphenomc_default_native_supported(_params) -> bool:
    """Return whether unflagged native use is accurate on this device.

    Apple MPS evaluates the legacy phase fits in float32. Cancellation among
    their large terms can materially reduce accuracy at low dimensionless
    frequencies, so the native MPS path remains an explicit opt-in.
    """
    state = _scheme.mgr.state
    return not (
        isinstance(state, _scheme.TorchScheme)
        and state.torch_device.type == "mps"
    )


@dataclass(frozen=True)
class _IMRPhenomCInputs:
    """Validated scalar inputs shared by regular and sequence generation."""

    distance: float
    inclination: float
    coa_phase: float
    long_asc_nodes: float
    total_mass: float
    eta: float
    xi: float
    total_mass_seconds: float
    f_cut: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype


@dataclass(frozen=True)
class _IMRPhenomCCoefficients:
    """Frequency-independent PhenomC fit coefficients."""

    a1: float
    a2: float
    a3: float
    a4: float
    a5: float
    a6: float
    g1: float
    del1: float
    del2: float
    Q: float
    f_rd: float
    Mfrd: float
    f1: float
    Mf1: float
    Mf2: float
    d1: float
    d2: float
    Mf0: float
    d0: float
    b1: float
    b2: float
    pfaN: float
    pfa2: float
    pfa3: float
    pfa4: float
    pfa5: float
    pfa6: float
    pfa6log: float
    pfa7: float
    xdotaN: float
    xdota2: float
    xdota3: float
    xdota4: float
    xdota5: float
    xdota6: float
    xdota6log: float
    xdota7: float
    AN: float
    A2: float
    A3: float
    A4: float
    A5: float
    A5imag: float
    A6: float
    A6log: float
    A6imag: float


def _imrphenomc_inputs(params, *, sequence=False):
    """Validate scalar inputs and select the active Torch precision."""
    if not imrphenomc_native_supported(params):
        raise ValueError(
            "IMRPhenomC parameters are not supported by the native Torch path"
        )

    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomC requires TorchScheme")
    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )

    f_ref = float(params.get("f_ref", 0.0))
    distance = float(params["distance"])
    mass1 = float(params["mass1"])
    mass2 = float(params["mass2"])
    spin1z = float(params.get("spin1z", 0.0))
    spin2z = float(params.get("spin2z", 0.0))
    inclination = float(params.get("inclination", 0.0))
    coa_phase = float(params.get("coa_phase", 0.0))
    # The LAL sequence API has no ascending-node argument. Match the common
    # PyCBC sequence convention even though this native extension has no LAL
    # IMRPhenomC sequence implementation to call.
    long_asc_nodes = (
        0.0 if sequence else float(params.get("long_asc_nodes", 0.0))
    )

    if not all(
        math.isfinite(value)
        for value in (
            mass1,
            mass2,
            spin1z,
            spin2z,
            inclination,
            coa_phase,
            long_asc_nodes,
            f_ref,
            distance,
        )
    ):
        raise ValueError("IMRPhenomC parameters must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("IMRPhenomC component masses must be positive")
    if abs(spin1z) > 1.0 or abs(spin2z) > 1.0:
        raise ValueError("IMRPhenomC aligned spins must be between -1 and 1")
    if f_ref < 0.0:
        raise ValueError("IMRPhenomC f_ref must be non-negative")
    if distance <= 0.0:
        raise ValueError("IMRPhenomC distance must be positive")

    mass_ratio = max(mass1, mass2) / min(mass1, mass2)
    if mass_ratio > 20.0:
        raise ValueError("IMRPhenomC mass ratio must not exceed 20")
    if mass_ratio > 4.0:
        warnings.warn(
            "IMRPhenomC is calibrated only for mass ratios up to 4",
            RuntimeWarning,
            stacklevel=3,
        )

    M = mass1 + mass2
    eta = mass1 * mass2 / (M * M)
    xi = (mass1 * spin1z / M) + (mass2 * spin2z / M)
    if abs(xi) > 0.9:
        raise ValueError("IMRPhenomC effective spin must be between -0.9 and 0.9")
    m_sec = M * _MTSUN_SI
    return _IMRPhenomCInputs(
        distance=distance * _distance_scale(M),
        inclination=inclination,
        coa_phase=coa_phase,
        long_asc_nodes=long_asc_nodes,
        total_mass=M,
        eta=eta,
        xi=xi,
        total_mass_seconds=m_sec,
        f_cut=0.15 / m_sec,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )


def _imrphenomc_coefficients(inputs):
    """Assemble the scalar PhenomC fits shared by both sampling APIs."""
    M = inputs.total_mass
    eta = inputs.eta
    xi = inputs.xi
    xisum = 2.0 * xi
    xiprod = xi * xi
    xi2 = xi * xi
    eta2 = eta * eta
    m_sec = inputs.total_mass_seconds

    # Lambda parameter fits (Table II and Eq. 5.14 of PhenomC)
    z101 = -2.417e-03
    z102 = -1.093e-03
    z111 = -1.917e-02
    z110 = 7.267e-02
    z120 = -2.504e-01

    z201 = 5.962e-01
    z202 = -5.600e-02
    z211 = 1.520e-01
    z210 = -2.970e00
    z220 = 1.312e01

    z301 = -3.283e01
    z302 = 8.859e00
    z311 = 2.931e01
    z310 = 7.954e01
    z320 = -4.349e02

    z401 = 1.619e02
    z402 = -4.702e01
    z411 = -1.751e02
    z410 = -3.225e02
    z420 = 1.587e03

    z501 = -6.320e02
    z502 = 2.463e02
    z511 = 1.048e03
    z510 = 3.355e02
    z520 = -5.115e03

    z601 = -4.809e01
    z602 = -3.643e02
    z611 = -5.215e02
    z610 = 1.870e03
    z620 = 7.354e02

    z701 = 4.149e00
    z702 = -4.070e00
    z711 = -8.752e01
    z710 = -4.897e01
    z720 = 6.665e02

    z801 = -5.472e-02
    z802 = 2.094e-02
    z811 = 3.554e-01
    z810 = 1.151e-01
    z820 = 9.640e-01

    z901 = -1.235e00
    z902 = 3.423e-01
    z911 = 6.062e00
    z910 = 5.949e00
    z920 = -1.069e01

    a1 = z101 * xi + z102 * xi2 + z111 * eta * xi + z110 * eta + z120 * eta2
    a2 = z201 * xi + z202 * xi2 + z211 * eta * xi + z210 * eta + z220 * eta2
    a3 = z301 * xi + z302 * xi2 + z311 * eta * xi + z310 * eta + z320 * eta2
    a4 = z401 * xi + z402 * xi2 + z411 * eta * xi + z410 * eta + z420 * eta2
    a5 = z501 * xi + z502 * xi2 + z511 * eta * xi + z510 * eta + z520 * eta2
    a6 = z601 * xi + z602 * xi2 + z611 * eta * xi + z610 * eta + z620 * eta2

    g1 = z701 * xi + z702 * xi2 + z711 * eta * xi + z710 * eta + z720 * eta2

    del1 = z801 * xi + z802 * xi2 + z811 * eta * xi + z810 * eta + z820 * eta2
    del2 = z901 * xi + z902 * xi2 + z911 * eta * xi + z910 * eta + z920 * eta2

    a_fin = _final_spin(xi, eta)
    Q = _q(abs(a_fin))
    f_rd = _f_rd(abs(a_fin), M)
    Mfrd = f_rd * m_sec

    f1 = 0.1 * f_rd
    Mf1 = m_sec * f1
    f2 = f_rd
    Mf2 = m_sec * f2
    d1 = 0.005
    d2 = 0.005
    f0 = 0.98 * f_rd
    Mf0 = m_sec * f0
    d0 = 0.015

    b2 = (
        (-5.0 / 3.0) * a1 * pow(Mfrd, -8.0 / 3.0)
        - a2 / (Mfrd * Mfrd)
        - (a3 / 3.0) * pow(Mfrd, -4.0 / 3.0)
        + (2.0 / 3.0) * a5 * pow(Mfrd, -1.0 / 3.0)
        + a6
    ) / eta
    psiPMrd = (
        a1 * pow(Mfrd, -5.0 / 3.0)
        + a2 / Mfrd
        + a3 * pow(Mfrd, -1.0 / 3.0)
        + a4
        + a5 * pow(Mfrd, 2.0 / 3.0)
        + a6 * Mfrd
    ) / eta
    b1 = psiPMrd - b2 * Mfrd

    # PN coefficients (Eq. A3-A5)
    pfaN = 3.0 / (128.0 * eta)
    pfa2 = (3715.0 / 756.0) + (55.0 * eta / 9.0)
    pfa3 = -16.0 * _PI + (113.0 / 3.0) * xi - 38.0 * eta * xisum / 3.0
    pfa4 = (
        (152.93365 / 5.08032)
        - 50.0 * xi2
        + eta * (271.45 / 5.04 + 1.25 * xiprod)
        + 3085.0 * eta2 / 72.0
    )
    pfa5 = (
        _PI * (386.45 / 7.56 - 65.0 * eta / 9.0)
        - xi * (735.505 / 2.268 + 130.0 * eta / 9.0)
        + xisum * (1285.0 * eta / 8.1 + 170.0 * eta2 / 9.0)
        - 10.0 * xi2 * xi / 3.0
        + 10.0 * eta * xi * xiprod
    )
    pfa6 = (
        11583.231236531 / 4.694215680
        - 640.0 * _PI * _PI / 3.0
        - 6848.0 * _EULER_GAMMA / 21.0
        - 684.8 * math.log(64.0) / 6.3
        + eta * (2255.0 * _PI * _PI / 12.0 - 15737.765635 / 3.048192)
        + 76.055 * eta2 / 1.728
        - (127.825 * eta2 * eta / 1.296)
        + 2920.0 * _PI * xi / 3.0
        - (175.0 - 1490.0 * eta) * xi2 / 3.0
        - (1120.0 * _PI / 3.0 - 1085.0 * xi / 3.0) * eta * xisum
        + (269.45 * eta / 3.36 - 2365.0 * eta2 / 6.0) * xiprod
    )
    pfa6log = -6848.0 / 63.0
    pfa7 = (
        _PI * (770.96675 / 2.54016 + 378.515 * eta / 1.512 - 740.45 * eta2 / 7.56)
        - xi * (20373.952415 / 3.048192 + 1509.35 * eta / 2.24 - 5786.95 * eta2 / 4.32)
        + xisum
        * (
            4862.041225 * eta / 1.524096
            + 1189.775 * eta2 / 1.008
            - 717.05 * eta2 * eta / 2.16
            - 830.0 * eta * xi2 / 3.0
            + 35.0 * eta2 * xiprod / 3.0
        )
        - 560.0 * _PI * xi2
        + 20.0 * _PI * eta * xiprod
        + xi2 * xi * (945.55 / 1.68 - 85.0 * eta)
        + xi * xiprod * (396.65 * eta / 1.68 + 255.0 * eta2)
    )

    xdotaN = 64.0 * eta / 5.0
    xdota2 = -7.43 / 3.36 - 11.0 * eta / 4.0
    xdota3 = 4.0 * _PI - 11.3 * xi / 1.2 + 19.0 * eta * xisum / 6.0
    xdota4 = (
        3.4103 / 1.8144
        + 5.0 * xi2
        + eta * (13.661 / 2.016 - xiprod / 8.0)
        + 5.9 * eta2 / 1.8
    )
    xdota5 = (
        -_PI * (41.59 / 6.72 + 189.0 * eta / 8.0)
        - xi * (31.571 / 1.008 - 116.5 * eta / 2.4)
        + xisum * (21.863 * eta / 1.008 - 79.0 * eta2 / 6.0)
        - 3.0 * xi * xi2 / 4.0
        + 9.0 * eta * xi * xiprod / 4.0
    )
    xdota6 = (
        164.47322263 / 1.39708800
        - 17.12 * _EULER_GAMMA / 1.05
        + 16.0 * _PI * _PI / 3.0
        - 8.56 * math.log(16.0) / 1.05
        + eta * (45.1 * _PI * _PI / 4.8 - 561.98689 / 2.17728)
        + 5.41 * eta2 / 8.96
        - 5.605 * eta * eta2 / 2.592
        - 80.0 * _PI * xi / 3.0
        + eta * xisum * (20.0 * _PI / 3.0 - 113.5 * xi / 3.6)
        + xi2 * (64.153 / 1.008 - 45.7 * eta / 3.6)
        - xiprod * (7.87 * eta / 1.44 - 30.37 * eta2 / 1.44)
    )
    xdota6log = -856.0 / 105.0
    xdota7 = (
        -_PI * (4.415 / 4.032 - 358.675 * eta / 6.048 - 91.495 * eta2 / 1.512)
        - xi * (252.9407 / 2.7216 - 845.827 * eta / 6.048 + 415.51 * eta2 / 8.64)
        + xisum
        * (
            158.0239 * eta / 5.4432
            - 451.597 * eta2 / 6.048
            + 20.45 * eta2 * eta / 4.32
            + 107.0 * eta * xi2 / 6.0
            - 5.0 * eta2 * xiprod / 24.0
        )
        + 12.0 * _PI * xi2
        - xi2 * xi * (150.5 / 2.4 + eta / 8.0)
        + xi * xiprod * (10.1 * eta / 2.4 + 3.0 * eta2 / 8.0)
    )

    AN = 8.0 * eta * math.sqrt(_PI / 5.0)
    A2 = (-107.0 + 55.0 * eta) / 42.0
    A3 = 2.0 * _PI - 4.0 * xi / 3.0 + 2.0 * eta * xisum / 3.0
    A4 = -2.173 / 1.512 - eta * (10.69 / 2.16 - 2.0 * xiprod) + 2.047 * eta2 / 1.512
    A5 = -10.7 * _PI / 2.1 + eta * (3.4 * _PI / 2.1)
    A5imag = -24.0 * eta
    A6 = (
        270.27409 / 6.46800
        - 8.56 * _EULER_GAMMA / 1.05
        + 2.0 * _PI * _PI / 3.0
        + eta * (4.1 * _PI * _PI / 9.6 - 27.8185 / 3.3264)
        - 20.261 * eta2 / 2.772
        + 11.4635 * eta * eta2 / 9.9792
        - 4.28 * math.log(16.0) / 1.05
    )
    A6log = -428.0 / 105.0
    A6imag = 4.28 * _PI / 1.05

    return _IMRPhenomCCoefficients(
        a1=a1,
        a2=a2,
        a3=a3,
        a4=a4,
        a5=a5,
        a6=a6,
        g1=g1,
        del1=del1,
        del2=del2,
        Q=Q,
        f_rd=f_rd,
        Mfrd=Mfrd,
        f1=f1,
        Mf1=Mf1,
        Mf2=Mf2,
        d1=d1,
        d2=d2,
        Mf0=Mf0,
        d0=d0,
        b1=b1,
        b2=b2,
        pfaN=pfaN,
        pfa2=pfa2,
        pfa3=pfa3,
        pfa4=pfa4,
        pfa5=pfa5,
        pfa6=pfa6,
        pfa6log=pfa6log,
        pfa7=pfa7,
        xdotaN=xdotaN,
        xdota2=xdota2,
        xdota3=xdota3,
        xdota4=xdota4,
        xdota5=xdota5,
        xdota6=xdota6,
        xdota6log=xdota6log,
        xdota7=xdota7,
        AN=AN,
        A2=A2,
        A3=A3,
        A4=A4,
        A5=A5,
        A5imag=A5imag,
        A6=A6,
        A6log=A6log,
        A6imag=A6imag,
    )


def _imrphenomc_spa_amplitude(coefficients, frequencies, m_sec):
    """Evaluate the dimensionless PhenomC inspiral SPA amplitude."""

    v = torch.pow(_PI * m_sec * frequencies, 1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3
    v7 = v3 * v4
    xdot = (
        1.0
        + coefficients.xdota2 * v2
        + coefficients.xdota3 * v3
        + coefficients.xdota4 * v4
        + coefficients.xdota5 * v5
        + (
            coefficients.xdota6
            + coefficients.xdota6log * torch.log(v2)
        )
        * v6
        + coefficients.xdota7 * v7
    )
    xdot = xdot * (coefficients.xdotaN * v5 * v5)

    if torch.any((xdot < 0.0) & (frequencies < coefficients.f1)):
        raise ValueError("IMRPhenomC xdot is negative below the transition")
    amplitude_factor = torch.sqrt(torch.abs(_PI / (1.5 * v * xdot)))
    real = (
        amplitude_factor
        * coefficients.AN
        * v2
        * (
            1.0
            + coefficients.A2 * v2
            + coefficients.A3 * v3
            + coefficients.A4 * v4
            + coefficients.A5 * v5
            + (
                coefficients.A6
                + coefficients.A6log * torch.log(v2)
            )
            * v6
        )
    )
    imag = (
        amplitude_factor
        * coefficients.AN
        * v2
        * (coefficients.A5imag * v5 + coefficients.A6imag * v6)
    )
    return torch.sqrt(real * real + imag * imag)


def _imrphenomc_components(inputs, coefficients, frequencies):
    """Evaluate the fitted amplitude and phase at device frequencies."""
    m_sec = inputs.total_mass_seconds
    piM = _PI * m_sec
    eta = inputs.eta
    distance = inputs.distance
    (
        a1,
        a2,
        a3,
        a4,
        a5,
        a6,
    ) = (
        coefficients.a1,
        coefficients.a2,
        coefficients.a3,
        coefficients.a4,
        coefficients.a5,
        coefficients.a6,
    )
    (
        pfaN,
        pfa2,
        pfa3,
        pfa4,
        pfa5,
        pfa6,
        pfa6log,
        pfa7,
    ) = (
        coefficients.pfaN,
        coefficients.pfa2,
        coefficients.pfa3,
        coefficients.pfa4,
        coefficients.pfa5,
        coefficients.pfa6,
        coefficients.pfa6log,
        coefficients.pfa7,
    )
    fd = frequencies * m_sec
    v = torch.pow(piM * frequencies, 1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3
    v7 = v3 * v4
    w = torch.pow(m_sec * frequencies, 1.0 / 3.0)
    w2 = w * w
    w3 = w2 * w

    logv3 = torch.log(v3)
    phSPA = (
        1.0
        + pfa2 * v2
        + pfa3 * v3
        + pfa4 * v4
        + (1.0 + logv3) * pfa5 * v5
        + (pfa6 + pfa6log * logv3) * v6
        + pfa7 * v7
    )
    phSPA = phSPA * (pfaN / v5) - (_PI / 4.0)

    phPM = (
        a1 / (w3 * w2)
        + a2 / w3
        + a3 / w
        + a4
        + a5 * w2
        + a6 * w3
    ) / eta
    phRD = coefficients.b1 + coefficients.b2 * fd

    wPlusf1 = 0.5 * (
        1.0
        + torch.tanh(4.0 * (fd - coefficients.Mf1) / coefficients.d1)
    )
    wMinusf1 = 0.5 * (
        1.0
        - torch.tanh(4.0 * (fd - coefficients.Mf1) / coefficients.d1)
    )
    wPlusf2 = 0.5 * (
        1.0
        + torch.tanh(4.0 * (fd - coefficients.Mf2) / coefficients.d2)
    )
    wMinusf2 = 0.5 * (
        1.0
        - torch.tanh(4.0 * (fd - coefficients.Mf2) / coefficients.d2)
    )
    phasing = phSPA * wMinusf1 + phPM * wPlusf1 * wMinusf2 + phRD * wPlusf2

    ampSPA = _imrphenomc_spa_amplitude(coefficients, frequencies, m_sec)

    ampPM = ampSPA + coefficients.g1 * torch.pow(fd, 5.0 / 6.0)

    sig = coefficients.Mfrd * coefficients.del2 / coefficients.Q
    sig2 = sig * sig
    L = sig2 / (
        (fd - coefficients.Mfrd) * (fd - coefficients.Mfrd)
        + sig2 / 4.0
    )
    ampRD = coefficients.del1 * L * torch.pow(fd, -7.0 / 6.0)

    wPlusf0 = 0.5 * (
        1.0
        + torch.tanh(4.0 * (fd - coefficients.Mf0) / coefficients.d0)
    )
    wMinusf0 = 0.5 * (
        1.0
        - torch.tanh(4.0 * (fd - coefficients.Mf0) / coefficients.d0)
    )
    amplitude = -(ampPM * wMinusf0 + ampRD * wPlusf0) / distance
    return amplitude, phasing


def _imrphenomc_ringdown_time_correction(inputs, coefficients):
    """Return the continuous-phase coalescence-time correction.

    The regular LAL implementation differentiates a natural cubic spline of
    ``-phase`` at ringdown. An arbitrary sequence has no grid on which to build
    that spline, so this evaluates the analytic derivative of the same phase
    ansatz. The result is independent of sequence density and ordering.
    """
    f = coefficients.f_rd
    m_sec = inputs.total_mass_seconds
    eta = inputs.eta
    fd = f * m_sec
    v = pow(_PI * m_sec * f, 1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v2 * v3
    v6 = v3 * v3
    v7 = v3 * v4
    dv_df = v / (3.0 * f)
    logv3 = math.log(v3)

    spa_bracket = (
        1.0
        + coefficients.pfa2 * v2
        + coefficients.pfa3 * v3
        + coefficients.pfa4 * v4
        + (1.0 + logv3) * coefficients.pfa5 * v5
        + (coefficients.pfa6 + coefficients.pfa6log * logv3) * v6
        + coefficients.pfa7 * v7
    )
    spa_bracket_derivative = (
        2.0 * coefficients.pfa2 * v * dv_df
        + 3.0 * coefficients.pfa3 * v2 * dv_df
        + 4.0 * coefficients.pfa4 * v3 * dv_df
        + coefficients.pfa5
        * (v5 / f + 5.0 * (1.0 + logv3) * v4 * dv_df)
        + coefficients.pfa6log * v6 / f
        + (coefficients.pfa6 + coefficients.pfa6log * logv3)
        * 6.0
        * v5
        * dv_df
        + 7.0 * coefficients.pfa7 * v6 * dv_df
    )
    phSPA = coefficients.pfaN * spa_bracket / v5 - _PI / 4.0
    dphSPA_df = coefficients.pfaN * (
        spa_bracket_derivative / v5
        - 5.0 * spa_bracket * dv_df / v6
    )

    w = pow(m_sec * f, 1.0 / 3.0)
    w2 = w * w
    w3 = w2 * w
    w4 = w2 * w2
    w5 = w3 * w2
    w6 = w3 * w3
    dw_df = w / (3.0 * f)
    phPM = (
        coefficients.a1 / w5
        + coefficients.a2 / w3
        + coefficients.a3 / w
        + coefficients.a4
        + coefficients.a5 * w2
        + coefficients.a6 * w3
    ) / eta
    dphPM_df = (
        -5.0 * coefficients.a1 / w6
        - 3.0 * coefficients.a2 / w4
        - coefficients.a3 / w2
        + 2.0 * coefficients.a5 * w
        + 3.0 * coefficients.a6 * w2
    ) * dw_df / eta
    phRD = coefficients.b1 + coefficients.b2 * fd
    dphRD_df = coefficients.b2 * m_sec

    tanh1 = math.tanh(
        4.0 * (fd - coefficients.Mf1) / coefficients.d1
    )
    tanh2 = math.tanh(
        4.0 * (fd - coefficients.Mf2) / coefficients.d2
    )
    wPlusf1 = 0.5 * (1.0 + tanh1)
    wMinusf1 = 1.0 - wPlusf1
    wPlusf2 = 0.5 * (1.0 + tanh2)
    wMinusf2 = 1.0 - wPlusf2
    dwPlusf1_df = (
        2.0 * m_sec / coefficients.d1 * (1.0 - tanh1 * tanh1)
    )
    dwPlusf2_df = (
        2.0 * m_sec / coefficients.d2 * (1.0 - tanh2 * tanh2)
    )
    dphasing_df = (
        dphSPA_df * wMinusf1
        - phSPA * dwPlusf1_df
        + dphPM_df * wPlusf1 * wMinusf2
        + phPM * dwPlusf1_df * wMinusf2
        - phPM * wPlusf1 * dwPlusf2_df
        + dphRD_df * wPlusf2
        + phRD * dwPlusf2_df
    )
    return -dphasing_df / (2.0 * _PI)


def _imrphenomc_polarizations(
    inputs,
    frequencies,
    amplitude,
    phasing,
    time_correction,
):
    """Apply phase/time conventions and project into polarizations."""
    complex_phase = (
        -phasing
        + 2.0 * inputs.coa_phase
        - 2.0 * _PI * frequencies * time_correction
    )
    # ``torch.polar`` requires a non-negative magnitude, whereas PhenomC's
    # fitted amplitude carries an overall minus sign.
    samples = torch.complex(
        amplitude * torch.cos(complex_phase),
        amplitude * torch.sin(complex_phase),
    )

    cosi = math.cos(inputs.inclination)
    plus0 = 0.5 * (1.0 + cosi * cosi) * samples
    cross0 = samples * complex(0.0, -cosi)
    cos_nodes = math.cos(2.0 * inputs.long_asc_nodes)
    sin_nodes = math.sin(2.0 * inputs.long_asc_nodes)
    return (
        (cos_nodes * plus0 + sin_nodes * cross0).to(
            inputs.complex_dtype
        ),
        (cos_nodes * cross0 - sin_nodes * plus0).to(
            inputs.complex_dtype
        ),
    )


def imrphenomc_fd_torch(**params):
    """Torch-native regular-grid IMRPhenomC waveform (hp, hc)."""
    inputs = _imrphenomc_inputs(params)
    delta_f = float(params["delta_f"])
    f_min = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))
    if not all(math.isfinite(value) for value in (delta_f, f_min, f_final)):
        raise ValueError("IMRPhenomC frequencies must be finite")
    if delta_f <= 0.0 or f_min <= 0.0:
        raise ValueError("IMRPhenomC delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomC f_final must be non-negative")

    # LAL evaluates only through its fixed Mf=0.15 cutoff while retaining an
    # explicitly larger f_final as zero padding in the output layout.
    layout_f_max = f_final if f_final > 0.0 else inputs.f_cut
    active_f_max = min(layout_f_max, inputs.f_cut)
    if inputs.f_cut <= f_min:
        raise ValueError("IMRPhenomC f_cut is <= f_lower")
    if active_f_max <= f_min:
        raise ValueError("IMRPhenomC f_final is <= f_lower")

    # Frequency grid and derived powers (power-of-two layout like LAL).
    # LAL truncates f_max / delta_f to size_t before selecting the next
    # power-of-two FFT length. Preserve that order at power-of-two boundaries.
    layout_bins = int(layout_f_max / delta_f)
    fft_length = (
        1 if layout_bins <= 1 else 1 << (layout_bins - 1).bit_length()
    )
    nfreq = fft_length + 1
    bins = torch.arange(nfreq, device=inputs.device)
    freqs = bins.to(dtype=inputs.real_dtype) * delta_f
    kmin = int(math.floor(f_min / delta_f))
    kmax = int(math.floor(active_f_max / delta_f))
    mask = (bins >= kmin) & (bins < kmax)
    if kmax - kmin < 2:
        raise ValueError("IMRPhenomC needs at least two active frequency bins")
    active_frequencies = freqs[mask]

    coefficients = _imrphenomc_coefficients(inputs)
    amplitude, phasing = _imrphenomc_components(
        inputs,
        coefficients,
        active_frequencies,
    )

    # LAL shifts the signal so that coalescence is at t=0. It obtains the
    # shift from the derivative of a natural cubic spline of -phase at the
    # ringdown frequency (clipped to the final active sample).
    correction_frequency = min(coefficients.f_rd, (kmax - 1) * delta_f)
    if correction_frequency < kmin * delta_f:
        raise ValueError("IMRPhenomC ringdown frequency is <= f_lower")
    sample_position = (correction_frequency - kmin * delta_f) / delta_f
    time_correction = _natural_cubic_derivative(
        -phasing, delta_f, sample_position
    ) / (2.0 * _PI)
    hp_segment, hc_segment = _imrphenomc_polarizations(
        inputs,
        active_frequencies,
        amplitude,
        phasing,
        time_correction,
    )

    # Assemble the full spectrum with zeros outside the active band
    hp_data = torch.zeros(
        nfreq,
        device=inputs.device,
        dtype=inputs.complex_dtype,
    )
    hc_data = torch.zeros_like(hp_data)
    hp_data[mask] = hp_segment
    hc_data[mask] = hc_segment

    epoch = -1.0 / delta_f
    hp = FrequencySeries(
        TorchArrayData(hp_data), delta_f=delta_f, epoch=epoch, copy=False
    )
    hc = FrequencySeries(
        TorchArrayData(hc_data), delta_f=delta_f, epoch=epoch, copy=False
    )
    return hp, hc


def _imrphenomc_sequence_frequencies(sample_points, inputs):
    """Return validated arbitrary frequencies on the active Torch device."""
    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError(
            "IMRPhenomC sample_points must be a non-empty vector"
        )
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("IMRPhenomC sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("IMRPhenomC sample_points must be positive")
    return frequencies


def imrphenomc_fd_sequence_torch(**params):
    """Evaluate IMRPhenomC at arbitrary frequencies with Torch.

    LAL does not expose IMRPhenomC through its sequence API. This native
    extension uses the continuous phase derivative at ringdown in place of
    the regular generator's grid-dependent spline derivative. Frequencies at
    and above the model's fixed ``Mf=0.15`` cutoff are returned as zero.
    """
    if not imrphenomc_sequence_native_supported(params):
        raise ValueError(
            "IMRPhenomC sequence parameters are not supported by the "
            "native Torch path"
        )
    inputs = _imrphenomc_inputs(params, sequence=True)
    frequencies = _imrphenomc_sequence_frequencies(
        params["sample_points"],
        inputs,
    )
    coefficients = _imrphenomc_coefficients(inputs)
    plus = torch.zeros(
        frequencies.shape,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    cross = torch.zeros_like(plus)
    active = frequencies < inputs.f_cut
    if bool(torch.any(active)):
        active_frequencies = frequencies[active]
        amplitude, phasing = _imrphenomc_components(
            inputs,
            coefficients,
            active_frequencies,
        )
        time_correction = _imrphenomc_ringdown_time_correction(
            inputs,
            coefficients,
        )
        plus_segment, cross_segment = _imrphenomc_polarizations(
            inputs,
            active_frequencies,
            amplitude,
            phasing,
            time_correction,
        )
        plus[active] = plus_segment
        cross[active] = cross_segment
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


__all__ = [
    "imrphenomc_default_native_supported",
    "imrphenomc_fd_sequence_torch",
    "imrphenomc_fd_torch",
    "imrphenomc_native_supported",
    "imrphenomc_sequence_native_supported",
]
