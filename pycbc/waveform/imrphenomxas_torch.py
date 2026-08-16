# Copyright 2022 Adam Coogan and Thomas Edwards
# Copyright 2025 GW JAX Team
# Copyright 2026 PyCBC contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Ruff cannot parse jaxtyping's symbolic shape strings as forward annotations.
# ruff: noqa: F722

"""Torch-native IMRPhenomXAS frequency-domain waveforms.

The coefficient equations are adapted from ripple v0.2.1
(https://github.com/GW-JAX-Team/ripple/tree/v0.2.1) and reproduce the installed
LALSuite IMRPhenomXAS implementation.  Scalar matching derivatives use Torch
autograd; frequency-dependent amplitude, phase, masking, and polarization work
remains on the active Torch device. Both equal-spaced and arbitrary-frequency
XAS generation are supported. The NRTidalv2 and NRTidalv3 variants add their
matter phase, amplitude, alignment, and taper corrections there as well. The
public PyCBC path is opt-in through
``PYCBC_IMRPHENOMXAS_NATIVE=1`` or ``PYCBC_TORCH_NATIVE_PORTS=1``.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import lal
import torch

from pycbc import scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData

from ._torch_jax import jax, jnp, torch_context
from . import imrphenomx_utils_torch as IMRPhenomX_utils
from .nrtidal_torch import (
    nrtidal_amplitude,
    nrtidal_higher_order_spin_phase,
    nrtidal_merger_frequency,
    nrtidal_merger_frequency_v3,
    nrtidal_phase,
    nrtidal_quadrupole_from_lambda,
    nrtidal_self_spin_phase,
    nrtidal_taper,
    nrtidal_version,
)

Array = Any
Float = Any
FloatLike = Any

EULERGAMMA = 0.577215664901532860606512090082402431
MTSUN = lal.MTSUN_SI
MPC = 1.0e6 * lal.PC_SI
C = lal.C_SI
PI = lal.PI

eqspin_indx = 10
uneqspin_indx = 39

amp_eqspin_indx = 8
amp_uneqspin_indx = 36


class _NRTidalParams(NamedTuple):
    mass1: float
    mass2: float
    spin1z: float
    spin2z: float
    lambda1: float
    lambda2: float
    quadrupole1: float
    quadrupole2: float
    merger_frequency: float
    alignment_frequency: float
    version: int


class _IMRPhenomXASCore(NamedTuple):
    """Active XAS samples and the metadata for their full frequency series."""

    polarization: torch.Tensor
    npts: int
    first_bin: int
    stop_bin: int
    delta_f: float
    epoch: float


class _IMRPhenomXASInputs(NamedTuple):
    """Validated scalar inputs shared by uniform and sequence generation."""

    tidal_version: int | None
    mass1: float
    mass2: float
    spin1z: float
    spin2z: float
    lambda1: float
    lambda2: float
    dquad1: float
    dquad2: float
    f_ref: float
    distance: float
    inclination: float
    coa_phase: float
    long_asc_nodes: float
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype


_XAS_MODE_POLARIZATION_FACTOR = math.sqrt(5.0 / (16.0 * PI))


def get_inspiral_phase(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
) -> Float[Array, " n_freq"] | FloatLike:
    """
    Calculate the inspiral phase for the IMRPhenomD waveform.
    """
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    eta2 = eta * eta
    eta3 = eta2 * eta
    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))

    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    chi_eff = mm1 * chi1 + mm2 * chi2
    S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))

    # Spin variables
    chia = chi1 - chi2

    chi1L2L = chi1 * chi2
    chi1L2 = chi1 * chi1
    chi1L3 = chi1 * chi1 * chi1
    chi2L2 = chi2 * chi2
    chi2L3 = chi2 * chi2 * chi2

    # These are the TaylorF2 terms used in IMRPhenomXAS
    phi0 = 1.0
    phi1 = 0.0
    phi2 = (3715.0 / 756.0 + (55.0 * eta) / 9.0) * PI ** (2.0 / 3.0)
    phi3 = (
        -16.0 * PI**2
        + (
            (
                113.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
                - 76.0 * (chi1 + chi2) * eta
            )
            / 6.0
        )
        * PI
    )
    phi4 = (
        15293365.0 / 508032.0 + (27145.0 * eta) / 504.0 + (3085.0 * eta2) / 72.0
    ) * PI ** (4.0 / 3.0) + (
        (
            -5.0
            * (
                81.0 * chi1L2 * (1 + delta - 2 * eta)
                + 316.0 * chi1L2L * eta
                - 81.0 * chi2L2 * (-1 + delta + 2 * eta)
            )
        )
        / 16.0
    ) * PI ** (4.0 / 3.0)
    phi5 = 0.0
    phi5L = ((5.0 * (46374.0 - 6552.0 * eta) * PI) / 4536.0) * PI ** (5.0 / 3.0) + (
        (
            -732985.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
            - 560.0 * (-1213.0 * (chi1 + chi2) + 63.0 * (chi1 - chi2) * delta) * eta
            + 85680.0 * (chi1 + chi2) * eta2
        )
        / 4536.0
    ) * PI ** (5.0 / 3.0)
    phi6L = (-6848.0 / 63.0) * PI**2.0
    phi6 = (
        (
            11583231236531.0 / 4.69421568e9
            - (5.0 * eta * (3147553127.0 + 588.0 * eta * (-45633.0 + 102260.0 * eta)))
            / 3.048192e6
            - (6848.0 * EULERGAMMA) / 21.0
            - (640.0 * PI**2.0) / 3.0
            + (2255.0 * eta * PI**2.0) / 12.0
            - (13696.0 * jnp.log(2.0)) / 21.0
            - (6848.0 * jnp.log(PI)) / 63.0
        )
        * PI**2.0
        + (
            (
                5
                * (
                    227.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
                    - 156.0 * (chi1 + chi2) * eta
                )
                * PI
            )
            / 3.0
        )
        * PI**2.0
        + (
            (
                5.0
                * (
                    20.0 * chi1L2L * eta * (11763.0 + 12488.0 * eta)
                    + 7.0
                    * chi2L2
                    * (
                        -15103.0 * (-1 + delta)
                        + 2.0 * (-21683.0 + 6580.0 * delta) * eta
                        - 9808.0 * eta2
                    )
                    - 7.0
                    * chi1L2
                    * (
                        -15103.0 * (1 + delta)
                        + 2.0 * (21683.0 + 6580.0 * delta) * eta
                        + 9808.0 * eta2
                    )
                )
            )
            / 4032.0
        )
        * PI**2.0
    )
    phi7 = (
        ((5.0 * (15419335.0 + 168.0 * (75703.0 - 29618.0 * eta) * eta) * PI) / 254016.0)
        * PI ** (7.0 / 3.0)
        + (
            (
                5.0
                * (
                    -5030016755.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
                    + 4.0
                    * (
                        2113331119.0 * (chi1 + chi2)
                        + 675484362.0 * (chi1 - chi2) * delta
                    )
                    * eta
                    - 1008.0
                    * (208433.0 * (chi1 + chi2) + 25011.0 * (chi1 - chi2) * delta)
                    * eta2
                    + 90514368.0 * (chi1 + chi2) * eta3
                )
            )
            / 6.096384e6
        )
        * PI ** (7.0 / 3.0)
        + (
            -5.0
            * (
                57.0 * chi1L2 * (1 + delta - 2 * eta)
                + 220.0 * chi1L2L * eta
                - 57.0 * chi2L2 * (-1 + delta + 2 * eta)
            )
            * PI
        )
        * PI ** (7.0 / 3.0)
        + (
            (
                14585.0 * (-(chi2L3 * (-1 + delta)) + chi1L3 * (1 + delta))
                - 5.0
                * (
                    chi2L3 * (8819.0 - 2985.0 * delta)
                    + 8439.0 * chi1 * chi2L2 * (-1.0 + delta)
                    - 8439.0 * chi1L2 * chi2 * (1.0 + delta)
                    + chi1L3 * (8819.0 + 2985.0 * delta)
                )
                * eta
                + 40.0
                * (chi1 + chi2)
                * (17.0 * chi1L2 - 14.0 * chi1L2L + 17.0 * chi2L2)
                * eta2
            )
            / 48.0
        )
        * PI ** (7.0 / 3.0)
    )
    phi8 = (
        (
            -5.0
            * (
                1263141.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
                - 2.0
                * (794075.0 * (chi1 + chi2) + 178533.0 * (chi1 - chi2) * delta)
                * eta
                + 94344.0 * (chi1 + chi2) * eta2
            )
            * PI
            * (-1.0 + jnp.log(PI))
        )
        / 9072.0
    ) * PI ** (8.0 / 3.0)
    phi8L = (
        (
            -5.0
            * (
                1263141.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta)
                - 2.0
                * (794075.0 * (chi1 + chi2) + 178533.0 * (chi1 - chi2) * delta)
                * eta
                + 94344.0 * (chi1 + chi2) * eta2
            )
            * PI
        )
        / 9072.0
    ) * PI ** (8.0 / 3.0)

    gpoints4 = jnp.array([0.0, 1.0 / 4.0, 3.0 / 4.0, 1.0])
    # Note that they do not use 4.1 from 2001.11412, they actually use
    # (Cos(i PI / 3) + 1)/2

    _, _, fMs_MECO, _ = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)

    fMs_PhaseInsMin = 0.0026
    fMs_PhaseInsMax = 1.020 * fMs_MECO

    deltax = fMs_PhaseInsMax - fMs_PhaseInsMin
    xmin = fMs_PhaseInsMin

    CP_phase_Ins0 = gpoints4[0] * deltax + xmin
    CP_phase_Ins1 = gpoints4[1] * deltax + xmin
    CP_phase_Ins2 = gpoints4[2] * deltax + xmin
    CP_phase_Ins3 = gpoints4[3] * deltax + xmin

    CV_phase_Ins0 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[0, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(phase_coeffs[0, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CV(phase_coeffs[0, uneqspin_indx:], eta, S, chia)
    )
    CV_phase_Ins1 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[1, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(phase_coeffs[1, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CV(phase_coeffs[1, uneqspin_indx:], eta, S, chia)
    )
    CV_phase_Ins2 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[2, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(phase_coeffs[2, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CV(phase_coeffs[2, uneqspin_indx:], eta, S, chia)
    )

    # NOTE: This CV_phase_Ins3 disagrees slightly with the value in WF4py at non-zero spin
    CV_phase_Ins3 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[3, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(phase_coeffs[3, eqspin_indx:uneqspin_indx], eta, S)
        + IMRPhenomX_utils.Uneqspin_CV(phase_coeffs[3, uneqspin_indx:], eta, S, chia)
    )

    # See line 1322 of https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__internals_8c_source.html
    CV_phase_Ins0 = CV_phase_Ins0 + CV_phase_Ins2
    CV_phase_Ins1 = CV_phase_Ins1 + CV_phase_Ins2
    CV_phase_Ins3 = CV_phase_Ins3 + CV_phase_Ins2

    A0 = jnp.array(
        [
            jnp.ones(CP_phase_Ins0.shape),
            CP_phase_Ins0 ** (1.0 / 3.0),
            CP_phase_Ins0 ** (2.0 / 3.0),
            CP_phase_Ins0,
        ]
    )
    A1 = jnp.array(
        [
            jnp.ones(CP_phase_Ins1.shape),
            CP_phase_Ins1 ** (1.0 / 3.0),
            CP_phase_Ins1 ** (2.0 / 3.0),
            CP_phase_Ins1,
        ]
    )
    A2 = jnp.array(
        [
            jnp.ones(CP_phase_Ins2.shape),
            CP_phase_Ins2 ** (1.0 / 3.0),
            CP_phase_Ins2 ** (2.0 / 3.0),
            CP_phase_Ins2,
        ]
    )
    A3 = jnp.array(
        [
            jnp.ones(CP_phase_Ins3.shape),
            CP_phase_Ins3 ** (1.0 / 3.0),
            CP_phase_Ins3 ** (2.0 / 3.0),
            CP_phase_Ins3,
        ]
    )

    A = jnp.array([A0, A1, A2, A3])
    b = jnp.array(
        [
            CV_phase_Ins0,
            CV_phase_Ins1,
            CV_phase_Ins2,
            CV_phase_Ins3,
        ]
    )

    coeffs_Ins = jnp.linalg.solve(A, b)

    sigma1 = (-5.0 / 3.0) * coeffs_Ins[0]
    sigma2 = (-5.0 / 4.0) * coeffs_Ins[1]
    sigma3 = (-5.0 / 5.0) * coeffs_Ins[2]
    sigma4 = (-5.0 / 6.0) * coeffs_Ins[3]

    f13 = fM_s ** (1.0 / 3.0)
    f23 = f13 * f13
    f43 = fM_s * f13
    f53 = fM_s * f23
    f2 = fM_s * fM_s
    f73 = f2 * f13
    f83 = f2 * f23
    f3 = f2 * fM_s
    f103 = f3 * f13
    f113 = f3 * f23
    log_f = jnp.log(fM_s)

    phi_TF2 = (
        phi0
        + phi1 * f13
        + phi2 * f23
        + phi3 * fM_s
        + phi4 * f43
        + phi5 * f53
        + phi5L * f53 * log_f
        + phi6 * f2
        + phi6L * f2 * log_f
        + phi7 * f73
        + phi8 * f83
        + phi8L * f83 * log_f
    )

    phi_Ins = phi_TF2 + (sigma1 * f83 + sigma2 * f3 + sigma3 * f103 + sigma4 * f113)

    phiN = -(3.0 * PI ** (-5.0 / 3.0)) / 128.0
    return phi_Ins * phiN / f53


def get_intermediate_raw_phase(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    dPhaseIN: FloatLike,
    dPhaseRD: FloatLike,
    cL: FloatLike,
    chip: FloatLike = 0.0,
) -> Float[Array, " n_freq"] | FloatLike:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))

    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    StotR = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)
    chia = chi1 - chi2

    fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(
        m1, m2, chi1, chi2, chip=chip
    )

    gpoints5 = jnp.array(
        [
            0.0,
            1.0 / 2.0 - 1.0 / (2.0 * jnp.sqrt(2.0)),
            1.0 / 2.0,
            1.0 / 2 + 1.0 / (2.0 * jnp.sqrt(2.0)),
            1.0,
        ]
    )

    fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
    fMs_INmatch = fMs_MECO
    deltafMs = (fMs_IMmatch - fMs_INmatch) * 0.03
    fMs_PhaseMatchIN = fMs_INmatch - 1.0 * deltafMs
    fPhaseMatchIM = fMs_IMmatch + 0.5 * deltafMs

    deltax = fPhaseMatchIM - fMs_PhaseMatchIN
    xmin = fMs_PhaseMatchIN

    CP_phase_Int0 = gpoints5[0] * deltax + xmin
    CP_phase_Int1 = gpoints5[1] * deltax + xmin
    CP_phase_Int2 = gpoints5[2] * deltax + xmin
    CP_phase_Int3 = gpoints5[3] * deltax + xmin
    CP_phase_Int4 = gpoints5[4] * deltax + xmin

    CV_phase_Int0 = dPhaseIN
    CV_phase_Int4 = dPhaseRD

    # NOTE: This is different to WF4py and driving the difference in CV_phase_Int1
    v2IMmRDv4 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[4, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[4, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[4, uneqspin_indx:], eta, StotR, chia
        )
    )

    v3IMmRDv4 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[5, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[5, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[5, uneqspin_indx:], eta, StotR, chia
        )
    )
    v2IM = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[6, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[6, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[6, uneqspin_indx:], eta, StotR, chia
        )
    )

    # NOTE: This is different to WF4py and driving the difference in CV_phase_Int3
    d43 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[7, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[7, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[7, uneqspin_indx:], eta, StotR, chia
        )
    )

    CV_phase_RD3 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[11, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[11, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[11, uneqspin_indx:], eta, StotR, chia
        )
    )

    CV_phase_Int1 = 0.75 * (v2IMmRDv4 + CV_phase_RD3) + 0.25 * v2IM
    CV_phase_Int2 = v3IMmRDv4 + CV_phase_RD3
    CV_phase_Int3 = d43 + CV_phase_Int2

    A0 = jnp.array(
        [
            jnp.ones(CP_phase_Int0.shape),
            fMs_RD / CP_phase_Int0,
            (fMs_RD / CP_phase_Int0) * (fMs_RD / CP_phase_Int0),
            (fMs_RD / CP_phase_Int0) ** 3,
            (fMs_RD / CP_phase_Int0) ** 4,
        ]
    )
    A1 = jnp.array(
        [
            jnp.ones(CP_phase_Int1.shape),
            fMs_RD / CP_phase_Int1,
            (fMs_RD / CP_phase_Int1) * (fMs_RD / CP_phase_Int1),
            (fMs_RD / CP_phase_Int1) ** 3,
            (fMs_RD / CP_phase_Int1) ** 4,
        ]
    )
    A2 = jnp.array(
        [
            jnp.ones(CP_phase_Int2.shape),
            fMs_RD / CP_phase_Int2,
            (fMs_RD / CP_phase_Int2) * (fMs_RD / CP_phase_Int2),
            (fMs_RD / CP_phase_Int2) ** 3,
            (fMs_RD / CP_phase_Int2) ** 4,
        ]
    )
    A3 = jnp.array(
        [
            jnp.ones(CP_phase_Int3.shape),
            fMs_RD / CP_phase_Int3,
            (fMs_RD / CP_phase_Int3) * (fMs_RD / CP_phase_Int3),
            (fMs_RD / CP_phase_Int3) ** 3,
            (fMs_RD / CP_phase_Int3) ** 4,
        ]
    )
    A4 = jnp.array(
        [
            jnp.ones(CP_phase_Int4.shape),
            fMs_RD / CP_phase_Int4,
            (fMs_RD / CP_phase_Int4) * (fMs_RD / CP_phase_Int4),
            (fMs_RD / CP_phase_Int4) ** 3,
            (fMs_RD / CP_phase_Int4) ** 4,
        ]
    )

    A = jnp.array([A0, A1, A2, A3, A4])
    b = jnp.array(
        [
            CV_phase_Int0
            - (
                (4.0 * cL)
                / (
                    (4.0 * fMs_damp * fMs_damp)
                    + (CP_phase_Int0 - fMs_RD) * (CP_phase_Int0 - fMs_RD)
                )
            ),
            CV_phase_Int1
            - (
                (4.0 * cL)
                / (
                    (4.0 * fMs_damp * fMs_damp)
                    + (CP_phase_Int1 - fMs_RD) * (CP_phase_Int1 - fMs_RD)
                )
            ),
            CV_phase_Int2
            - (
                (4.0 * cL)
                / (
                    (4.0 * fMs_damp * fMs_damp)
                    + (CP_phase_Int2 - fMs_RD) * (CP_phase_Int2 - fMs_RD)
                )
            ),
            CV_phase_Int3
            - (
                (4.0 * cL)
                / (
                    (4.0 * fMs_damp * fMs_damp)
                    + (CP_phase_Int3 - fMs_RD) * (CP_phase_Int3 - fMs_RD)
                )
            ),
            CV_phase_Int4
            - (
                (4.0 * cL)
                / (
                    (4.0 * fMs_damp * fMs_damp)
                    + (CP_phase_Int4 - fMs_RD) * (CP_phase_Int4 - fMs_RD)
                )
            ),
        ]
    )

    coeffs_Int = jnp.linalg.solve(A, b)

    b0 = coeffs_Int[0]
    b1 = coeffs_Int[1] * fMs_RD
    b2 = coeffs_Int[2] * fMs_RD**2
    b3 = coeffs_Int[3] * fMs_RD**3
    b4 = coeffs_Int[4] * fMs_RD**4

    return (
        b0 * fM_s
        + b1 * jnp.log(fM_s)
        - b2 * (fM_s**-1.0)
        - b3 * (fM_s**-2.0) / 2.0
        - (b4 * (fM_s**-3.0) / 3.0)
        + (2.0 * cL * jnp.arctan((fM_s - fMs_RD) / (2.0 * fMs_damp))) / fMs_damp
    )


def get_mergerringdown_raw_phase(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    chip: FloatLike = 0.0,
) -> tuple[Float[Array, " n_freq"], tuple[FloatLike, FloatLike]]:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    # chi_eff = mm1 * chi1 + mm2 * chi2
    # S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))
    chia = chi1 - chi2
    StotR = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)

    fMs_RD, fMs_damp, _, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(
        m1, m2, chi1, chi2, chip=chip
    )
    fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
    fMs_PhaseRDMin = fMs_IMmatch
    fMs_PhaseRDMax = fMs_RD + 1.25 * fMs_damp
    dphase0 = 5.0 / (128.0 * (PI ** (5.0 / 3.0)))

    gpoints5 = jnp.array(
        [
            0.0,
            1.0 / 2.0 - 1.0 / (2.0 * jnp.sqrt(2.0)),
            1.0 / 2.0,
            1.0 / 2 + 1.0 / (2.0 * jnp.sqrt(2.0)),
            1.0,
        ]
    )

    # Ringdown phase collocation points:
    # Default is to use 5 pseudo-PN coefficients and hence 5 collocation points.
    deltax = fMs_PhaseRDMax - fMs_PhaseRDMin
    xmin = fMs_PhaseRDMin

    CP_phase_RD0 = gpoints5[0] * deltax + xmin
    CP_phase_RD1 = gpoints5[1] * deltax + xmin
    CP_phase_RD2 = gpoints5[2] * deltax + xmin
    CP_phase_RD3 = jnp.asarray(fMs_RD)
    CP_phase_RD4 = gpoints5[4] * deltax + xmin

    CV_phase_RD0 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[8, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[8, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[8, uneqspin_indx:], eta, StotR, chia
        )
    )
    CV_phase_RD1 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[9, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[9, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[9, uneqspin_indx:], eta, StotR, chia
        )
    )
    CV_phase_RD2 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[10, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[10, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[10, uneqspin_indx:], eta, StotR, chia
        )
    )
    CV_phase_RD3 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[11, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[11, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[11, uneqspin_indx:], eta, StotR, chia
        )
    )
    CV_phase_RD4 = (
        IMRPhenomX_utils.nospin_CV(phase_coeffs[12, 0:eqspin_indx], eta)
        + IMRPhenomX_utils.Eqspin_CV(
            phase_coeffs[12, eqspin_indx:uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Uneqspin_CV(
            phase_coeffs[12, uneqspin_indx:], eta, StotR, chia
        )
    )

    CV_phase_RD4 = CV_phase_RD4 + CV_phase_RD3
    CV_phase_RD2 = CV_phase_RD2 + CV_phase_RD3
    CV_phase_RD1 = CV_phase_RD1 + CV_phase_RD3
    CV_phase_RD0 = CV_phase_RD0 + CV_phase_RD1

    A0 = jnp.array(
        [
            jnp.ones(CP_phase_RD0.shape),
            CP_phase_RD0 ** (-1.0 / 3.0),
            CP_phase_RD0 ** (-2),
            CP_phase_RD0 ** (-4),
            -(dphase0)
            / (fMs_damp * fMs_damp + (CP_phase_RD0 - fMs_RD) * (CP_phase_RD0 - fMs_RD)),
        ]
    )
    A1 = jnp.array(
        [
            jnp.ones(CP_phase_RD1.shape),
            CP_phase_RD1 ** (-1.0 / 3.0),
            CP_phase_RD1 ** (-2),
            CP_phase_RD1 ** (-4),
            -(dphase0)
            / (fMs_damp * fMs_damp + (CP_phase_RD1 - fMs_RD) * (CP_phase_RD1 - fMs_RD)),
        ]
    )
    A2 = jnp.array(
        [
            jnp.ones(CP_phase_RD2.shape),
            CP_phase_RD2 ** (-1.0 / 3.0),
            CP_phase_RD2 ** (-2),
            CP_phase_RD2 ** (-4),
            -(dphase0)
            / (fMs_damp * fMs_damp + (CP_phase_RD2 - fMs_RD) * (CP_phase_RD2 - fMs_RD)),
        ]
    )
    A3 = jnp.array(
        [
            jnp.ones(CP_phase_RD3.shape),
            CP_phase_RD3 ** (-1.0 / 3.0),
            CP_phase_RD3 ** (-2),
            CP_phase_RD3 ** (-4),
            -(dphase0)
            / (fMs_damp * fMs_damp + (CP_phase_RD3 - fMs_RD) * (CP_phase_RD3 - fMs_RD)),
        ]
    )
    A4 = jnp.array(
        [
            jnp.ones(CP_phase_RD4.shape),
            CP_phase_RD4 ** (-1.0 / 3.0),
            CP_phase_RD4 ** (-2),
            CP_phase_RD4 ** (-4),
            -(dphase0)
            / (fMs_damp * fMs_damp + (CP_phase_RD4 - fMs_RD) * (CP_phase_RD4 - fMs_RD)),
        ]
    )

    A = jnp.array([A0, A1, A2, A3, A4])
    b = jnp.array(
        [
            CV_phase_RD0,
            CV_phase_RD1,
            CV_phase_RD2,
            CV_phase_RD3,
            CV_phase_RD4,
        ]
    )

    coeffs_RD = jnp.linalg.solve(A, b)
    c0, c1, c2, c4, cRD = coeffs_RD
    cL = -(dphase0 * cRD)
    c4ov3 = c4 / 3.0
    cLovfda = cL / fMs_damp

    phiRD = (
        c0 * fM_s
        + 1.5 * c1 * (fM_s ** (2.0 / 3.0))
        - c2 * (fM_s**-1.0)
        - c4ov3 * (fM_s**-3.0)
        + (cLovfda * jnp.arctan((fM_s - fMs_RD) / fMs_damp))
    )

    return phiRD, (cL, CV_phase_RD0)


def Phase(
    f: Float[Array, " n_freq"] | float,
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    chip: FloatLike = 0.0,
) -> Float[Array, " n_freq"]:
    """
    Computes the phase of the PhenomD waveform following 1508.07253.
    Sets time and phase of coealence to be zero.

    Returns:
        phase (array): Phase of the GW as a function of frequency
    """
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    fM_s = f * M_s
    fMs_RD, _, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(
        m1, m2, chi1, chi2, chip=chip
    )
    fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
    fMs_INmatch = fMs_MECO
    deltafMs = (fMs_IMmatch - fMs_INmatch) * 0.03
    f1_Ms = fMs_INmatch - 1.0 * deltafMs
    f2_Ms = fMs_IMmatch + 0.5 * deltafMs

    # Calculate the inspiral and raw merger phase (required for the intemediate phase)
    phi_Ins = get_inspiral_phase(fM_s, theta, phase_coeffs)
    phi_MRD, (cL, CV_phase_RD0) = get_mergerringdown_raw_phase(
        fM_s, theta, phase_coeffs, chip
    )

    # Get matching points
    # Here we want to evaluate the gradient and the phase of the raw phase functions
    # in order to enforce C1 continuity at the transition frequencies.
    # This procedure is identical to IMRPhenomD, see IMRPhenomD.py for more details
    phi_Ins_match_f1, dphi_Ins_match_f1 = jax.value_and_grad(get_inspiral_phase)(
        f1_Ms, theta, phase_coeffs
    )
    phi_MRD_match_f2, dphi_MRD_match_f2 = jax.value_and_grad(
        lambda f_: get_mergerringdown_raw_phase(f_, theta, phase_coeffs, chip),
        has_aux=True,
    )(f2_Ms)
    phi_MRD_match_f2, _ = get_mergerringdown_raw_phase(f2_Ms, theta, phase_coeffs, chip)

    # Now find the intermediate phase
    phi_Int_match_f1, dphi_Int_match_f1 = jax.value_and_grad(
        get_intermediate_raw_phase
    )(f1_Ms, theta, phase_coeffs, dphi_Ins_match_f1, CV_phase_RD0, cL, chip)
    alpha1 = dphi_Ins_match_f1 - dphi_Int_match_f1
    alpha0 = phi_Ins_match_f1 - phi_Int_match_f1 - alpha1 * f1_Ms

    def phi_Int_func(fM_s_):
        return (
            get_intermediate_raw_phase(
                fM_s_,
                theta,
                phase_coeffs,
                dphi_Ins_match_f1,
                CV_phase_RD0,
                cL,
                chip,
            )
            + alpha1 * fM_s_
            + alpha0
        )

    phi_Int_match_f2, dphi_Int_match_f2 = jax.value_and_grad(phi_Int_func)(f2_Ms)

    beta1 = dphi_Int_match_f2 - dphi_MRD_match_f2
    beta0 = phi_Int_match_f2 - phi_MRD_match_f2 - beta1 * f2_Ms

    phi_Int_corrected = phi_Int_func(fM_s)
    phi_MRD_corrected = phi_MRD + beta0 + beta1 * fM_s

    phase = (1 / eta) * (
        phi_Ins * jnp.heaviside(f1_Ms - fM_s, 0.5)
        + jnp.heaviside(fM_s - f1_Ms, 0.5)
        * phi_Int_corrected
        * jnp.heaviside(f2_Ms - fM_s, 0.5)
        + phi_MRD_corrected
        * jnp.heaviside(fM_s - f2_Ms, 0.5)
        * jnp.heaviside(IMRPhenomX_utils.fM_CUT - fM_s, 0.5)
    )

    return phase


def PhaseDerivative(
    f: Float[Array, " n_freq"],
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    chip: float = 0.0,
) -> Float[Array, " n_freq"]:
    """
    Compute d Phase / d f for IMRPhenomXAS using the same piecewise construction
    as Phase(), but without differentiating through the final Heaviside assembly.
    """

    m1, m2, chi1, chi2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    fM_s = f * M_s
    fMs_RD, _, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(
        m1, m2, chi1, chi2, chip
    )
    fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
    fMs_INmatch = fMs_MECO
    deltafMs = (fMs_IMmatch - fMs_INmatch) * 0.03
    f1_Ms = fMs_INmatch - 1.0 * deltafMs
    f2_Ms = fMs_IMmatch + 0.5 * deltafMs

    phi_Ins_match_f1, dphi_Ins_match_f1 = jax.value_and_grad(get_inspiral_phase)(
        f1_Ms, theta, phase_coeffs
    )
    _phi_MRD_match_f2, dphi_MRD_match_f2 = jax.value_and_grad(
        get_mergerringdown_raw_phase, has_aux=True
    )(f2_Ms, theta, phase_coeffs, chip)
    _phi_MRD_match_f2, (cL, CV_phase_RD0) = get_mergerringdown_raw_phase(
        f2_Ms, theta, phase_coeffs, chip
    )

    phi_Int_match_f1, dphi_Int_match_f1 = jax.value_and_grad(
        get_intermediate_raw_phase
    )(f1_Ms, theta, phase_coeffs, dphi_Ins_match_f1, CV_phase_RD0, cL, chip)
    alpha1 = dphi_Ins_match_f1 - dphi_Int_match_f1
    alpha0 = phi_Ins_match_f1 - phi_Int_match_f1 - alpha1 * f1_Ms

    def phi_Int_func(fM_s_):
        return (
            get_intermediate_raw_phase(
                fM_s_,
                theta,
                phase_coeffs,
                dphi_Ins_match_f1,
                CV_phase_RD0,
                cL,
                chip,
            )
            + alpha1 * fM_s_
            + alpha0
        )

    _phi_Int_match_f2, dphi_Int_match_f2 = jax.value_and_grad(phi_Int_func)(f2_Ms)
    beta1 = dphi_Int_match_f2 - dphi_MRD_match_f2

    dphi_Ins = jax.grad(get_inspiral_phase)(fM_s, theta, phase_coeffs)
    dphi_Int = jax.grad(phi_Int_func)(fM_s)
    dphi_MRD = (
        jax.grad(
            lambda x: get_mergerringdown_raw_phase(x, theta, phase_coeffs, chip)[0]
        )(fM_s)
        + beta1
    )

    dphase_dMf = jnp.where(
        fM_s < f1_Ms,
        dphi_Ins / eta,
        jnp.where(fM_s < f2_Ms, dphi_Int / eta, dphi_MRD / eta),
    )
    return dphase_dMf * M_s


def get_inspiral_Amp(
    fM_s: Float[Array, " n_freq"],
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    chip: float = 0.0,
) -> Float[Array, " n_freq"]:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    eta2 = eta * eta
    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))

    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    chi_eff = mm1 * chi1 + mm2 * chi2
    S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))
    chia = chi1 - chi2

    _, _, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)
    fMs_AmpInsMax = fMs_MECO + 0.25 * (fMs_ISCO - fMs_MECO)
    fMs_AmpMatchIN = fMs_AmpInsMax

    A0 = 1.0
    # A1 = 0.0
    A2 = ((-969.0 + 1804.0 * eta) / 672.0) * (PI ** (2.0 / 3.0))
    A3 = (
        (
            81.0 * (chi1 + chi2)
            + 81.0 * chi1 * delta
            - 81.0 * chi2 * delta
            - 44.0 * (chi1 + chi2) * eta
        )
        / 48.0
    ) * PI
    A4 = (
        (
            -27312085.0
            - 10287648.0 * chi1**2.0 * (1.0 + delta)
            + 24.0
            * (
                428652.0 * chi2**2 * (-1 + delta)
                + (
                    -1975055.0
                    + 10584.0 * (81.0 * chi1**2.0 - 94.0 * chi1 * chi2 + 81.0 * chi2**2)
                )
                * eta
                + 1473794.0 * eta2
            )
        )
        / 8.128512e6
    ) * (PI ** (4.0 / 3.0))
    A5 = (
        (
            -6048.0 * chi1**2.0 * chi1 * (-1.0 - delta + (3.0 + delta) * eta)
            + chi2
            * (
                -((287213.0 + 6048.0 * chi2**2) * (-1.0 + delta))
                + 4
                * (-93414.0 + 1512.0 * chi2**2.0 * (-3.0 + delta) + 2083.0 * delta)
                * eta
                - 35632.0 * eta2
            )
            + chi1
            * (
                287213.0 * (1.0 + delta)
                - 4.0 * eta * (93414.0 + 2083.0 * delta + 8908.0 * eta)
            )
            + 42840.0 * (-1.0 + 4.0 * eta) * PI
        )
        / 32256.0
    ) * (PI ** (5.0 / 3.0))
    A6 = (
        (
            -1242641879927.0
            + 12.0
            * (
                28.0
                * (
                    -3248849057.0
                    + 11088.0
                    * (
                        163199.0 * chi1**2.0
                        - 266498.0 * chi1 * chi2
                        + 163199.0 * chi2**2.0
                    )
                )
                * eta2
                + 27026893936.0 * eta2 * eta
                - 116424.0
                * (
                    147117.0
                    * (-(chi2**2.0 * (-1.0 + delta)) + chi1**2.0 * (1.0 + delta))
                    + 60928.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta) * PI
                )
                + eta
                * (
                    545384828789.0
                    - 77616.0
                    * (
                        638642.0 * chi1 * chi2
                        + chi1**2.0 * (-158633.0 + 282718.0 * delta)
                        - chi2**2.0 * (158633.0 + 282718.0 * delta)
                        - 107520.0 * (chi1 + chi2) * PI
                        + 275520.0 * PI**2
                    )
                )
            )
        )
        / 6.0085960704e10
    ) * PI**2

    # Now we need to get the higher order components

    CV_Amp_Ins0 = (
        IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[0, 0:amp_eqspin_indx], eta)
        + IMRPhenomX_utils.Amp_Eqspin_CV(
            amp_coeffs[0, amp_eqspin_indx:amp_uneqspin_indx], eta, S
        )
        + IMRPhenomX_utils.Amp_Uneqspin_CV(
            amp_coeffs[0, amp_uneqspin_indx:], eta, S, chia
        )
    )
    CV_Amp_Ins1 = (
        IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[1, 0:amp_eqspin_indx], eta)
        + IMRPhenomX_utils.Amp_Eqspin_CV(
            amp_coeffs[1, amp_eqspin_indx:amp_uneqspin_indx], eta, S
        )
        + IMRPhenomX_utils.Amp_Uneqspin_CV(
            amp_coeffs[1, amp_uneqspin_indx:], eta, S, chia
        )
    )
    CV_Amp_Ins2 = (
        IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[2, 0:amp_eqspin_indx], eta)
        + IMRPhenomX_utils.Amp_Eqspin_CV(
            amp_coeffs[2, amp_eqspin_indx:amp_uneqspin_indx], eta, S
        )
        + IMRPhenomX_utils.Amp_Uneqspin_CV(
            amp_coeffs[2, amp_uneqspin_indx:], eta, S, chia
        )
    )

    CP_Amp_Ins0 = 0.50 * fMs_AmpMatchIN
    CP_Amp_Ins1 = 0.75 * fMs_AmpMatchIN
    CP_Amp_Ins2 = 1.00 * fMs_AmpMatchIN

    rho1 = (
        -((CP_Amp_Ins1 ** (8.0 / 3.0)) * (CP_Amp_Ins2**3) * CV_Amp_Ins0)
        + CP_Amp_Ins1**3 * (CP_Amp_Ins2 ** (8.0 / 3.0)) * CV_Amp_Ins0
        + (CP_Amp_Ins0 ** (8.0 / 3.0)) * (CP_Amp_Ins2**3) * CV_Amp_Ins1
        - CP_Amp_Ins0**3 * (CP_Amp_Ins2 ** (8.0 / 3.0)) * CV_Amp_Ins1
        - (CP_Amp_Ins0 ** (8.0 / 3.0)) * (CP_Amp_Ins1**3) * CV_Amp_Ins2
        + CP_Amp_Ins0**3 * (CP_Amp_Ins1 ** (8.0 / 3.0)) * CV_Amp_Ins2
    ) / (
        (CP_Amp_Ins0 ** (7.0 / 3.0))
        * (jnp.cbrt(CP_Amp_Ins0) - jnp.cbrt(CP_Amp_Ins1))
        * (CP_Amp_Ins1 ** (7.0 / 3.0))
        * (jnp.cbrt(CP_Amp_Ins0) - jnp.cbrt(CP_Amp_Ins2))
        * (jnp.cbrt(CP_Amp_Ins1) - jnp.cbrt(CP_Amp_Ins2))
        * (CP_Amp_Ins2 ** (7.0 / 3.0))
    )
    rho2 = (
        (CP_Amp_Ins1 ** (7.0 / 3.0)) * (CP_Amp_Ins2**3) * CV_Amp_Ins0
        - CP_Amp_Ins1**3 * (CP_Amp_Ins2 ** (7.0 / 3.0)) * CV_Amp_Ins0
        - (CP_Amp_Ins0 ** (7.0 / 3.0)) * (CP_Amp_Ins2**3) * CV_Amp_Ins1
        + CP_Amp_Ins0**3 * (CP_Amp_Ins2 ** (7.0 / 3.0)) * CV_Amp_Ins1
        + (CP_Amp_Ins0 ** (7.0 / 3.0)) * (CP_Amp_Ins1**3) * CV_Amp_Ins2
        - CP_Amp_Ins0**3 * (CP_Amp_Ins1 ** (7.0 / 3.0)) * CV_Amp_Ins2
    ) / (
        (CP_Amp_Ins0 ** (7.0 / 3.0))
        * (jnp.cbrt(CP_Amp_Ins0) - jnp.cbrt(CP_Amp_Ins1))
        * (CP_Amp_Ins1 ** (7.0 / 3.0))
        * (jnp.cbrt(CP_Amp_Ins0) - jnp.cbrt(CP_Amp_Ins2))
        * (jnp.cbrt(CP_Amp_Ins1) - jnp.cbrt(CP_Amp_Ins2))
        * (CP_Amp_Ins2 ** (7.0 / 3.0))
    )
    rho3 = (
        (CP_Amp_Ins1 ** (8.0 / 3.0)) * (CP_Amp_Ins2 ** (7.0 / 3.0)) * CV_Amp_Ins0
        - (CP_Amp_Ins1 ** (7.0 / 3.0)) * (CP_Amp_Ins2 ** (8.0 / 3.0)) * CV_Amp_Ins0
        - (CP_Amp_Ins0 ** (8.0 / 3.0)) * (CP_Amp_Ins2 ** (7.0 / 3.0)) * CV_Amp_Ins1
        + (CP_Amp_Ins0 ** (7.0 / 3.0)) * (CP_Amp_Ins2 ** (8.0 / 3.0)) * CV_Amp_Ins1
        + (CP_Amp_Ins0 ** (8.0 / 3.0)) * (CP_Amp_Ins1 ** (7.0 / 3.0)) * CV_Amp_Ins2
        - (CP_Amp_Ins0 ** (7.0 / 3.0)) * (CP_Amp_Ins1 ** (8.0 / 3.0)) * CV_Amp_Ins2
    ) / (
        (CP_Amp_Ins0 ** (7.0 / 3.0))
        * (jnp.cbrt(CP_Amp_Ins0) - jnp.cbrt(CP_Amp_Ins1))
        * (CP_Amp_Ins1 ** (7.0 / 3.0))
        * (jnp.cbrt(CP_Amp_Ins0) - jnp.cbrt(CP_Amp_Ins2))
        * (jnp.cbrt(CP_Amp_Ins1) - jnp.cbrt(CP_Amp_Ins2))
        * (CP_Amp_Ins2 ** (7.0 / 3.0))
    )

    Amp_Ins = (
        A0
        # A1 is missed since its zero
        + A2 * (fM_s ** (2.0 / 3.0))
        + A3 * fM_s
        + A4 * (fM_s ** (4.0 / 3.0))
        + A5 * (fM_s ** (5.0 / 3.0))
        + A6 * (fM_s**2.0)
        # # Now we add the coefficient terms
        + rho1 * (fM_s ** (7.0 / 3.0))
        + rho2 * (fM_s ** (8.0 / 3.0))
        + rho3 * (fM_s**3.0)
    )

    return Amp_Ins


def get_intermediate_Amp(
    fM_s: Float[Array, " n_freq"],
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    fMs_AmpRDMin: FloatLike,
    chip: float = 0.0,
) -> Float[Array, " n_freq"]:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    # eta2 = eta * eta
    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))

    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    StotR = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)

    # Spin variables
    chia = chi1 - chi2

    # Now the intermediate region
    _, _, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)
    fMs_AmpInsMax = fMs_MECO + 0.25 * (fMs_ISCO - fMs_MECO)
    fMs_AmpMatchIN = fMs_AmpInsMax
    FMs1 = fMs_AmpMatchIN
    # This needs to come from outside
    FMs4 = fMs_AmpRDMin

    inspFMs1, d1 = jax.value_and_grad(get_inspiral_Amp)(FMs1, theta, amp_coeffs, chip)
    rdFMs4, d4 = jax.value_and_grad(get_mergerringdown_Amp, has_aux=True)(
        FMs4, theta, amp_coeffs, chip
    )
    rdFMs4 = rdFMs4[0]

    # Use d1 and d4 calculated above to get the derivative of the amplitude on the boundaries
    d1 = ((7.0 / 6.0) * (FMs1 ** (1.0 / 6.0)) / inspFMs1) - (
        (FMs1 ** (7.0 / 6.0)) * d1 / (inspFMs1 * inspFMs1)
    )
    d4 = ((7.0 / 6.0) * (FMs4 ** (1.0 / 6.0)) / rdFMs4) - (
        (FMs4 ** (7.0 / 6.0)) * d4 / (rdFMs4 * rdFMs4)
    )

    # Use a 4th order polynomial in intermediate - good extrapolation, recommended default fit
    FMs2 = FMs1 + (1.0 / 2.0) * (FMs4 - FMs1)

    V1 = (FMs1 ** (-7.0 / 6)) * inspFMs1

    V2 = (
        IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[3, 0:amp_eqspin_indx], eta)
        + IMRPhenomX_utils.Amp_Eqspin_CV(
            amp_coeffs[3, amp_eqspin_indx:amp_uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Amp_Uneqspin_CV(
            amp_coeffs[3, amp_uneqspin_indx:], eta, StotR, chia
        )
    )
    V4 = (FMs4 ** (-7.0 / 6)) * rdFMs4

    V1 = 1.0 / V1
    V2 = 1.0 / V2
    V4 = 1.0 / V4

    # Reconstruct the phenomenological coefficients for the intermediate ansatz
    F12 = FMs1 * FMs1
    F13 = F12 * FMs1
    F14 = F13 * FMs1
    F15 = F14 * FMs1

    F22 = FMs2 * FMs2
    F23 = F22 * FMs2
    F24 = F23 * FMs2

    F42 = FMs4 * FMs4
    F43 = F42 * FMs4
    F44 = F43 * FMs4
    F45 = F44 * FMs4

    F1mF2 = FMs1 - FMs2
    F1mF4 = FMs1 - FMs4
    F2mF4 = FMs2 - FMs4

    F1mF22 = F1mF2 * F1mF2
    F2mF42 = F2mF4 * F2mF4
    F1mF43 = F1mF4 * F1mF4 * F1mF4

    delta0 = (
        -(d4 * F12 * F1mF22 * F1mF4 * FMs2 * F2mF4 * FMs4)
        + d1 * FMs1 * F1mF2 * F1mF4 * FMs2 * F2mF42 * F42
        + F42
        * (
            FMs2
            * F2mF42
            * (-4 * F12 + 3 * FMs1 * FMs2 + 2 * FMs1 * FMs4 - FMs2 * FMs4)
            * V1
            + F12 * F1mF43 * V2
        )
        + F12
        * F1mF22
        * FMs2
        * (FMs1 * FMs2 - 2 * FMs1 * FMs4 - 3 * FMs2 * FMs4 + 4 * F42)
        * V4
    ) / (F1mF22 * F1mF43 * F2mF42)

    delta1 = (
        d4 * FMs1 * F1mF22 * F1mF4 * F2mF4 * (2 * FMs2 * FMs4 + FMs1 * (FMs2 + FMs4))
        + FMs4
        * (
            -(d1 * F1mF2 * F1mF4 * F2mF42 * (2 * FMs1 * FMs2 + (FMs1 + FMs2) * FMs4))
            - 2
            * FMs1
            * (
                F44 * (V1 - V2)
                + 3 * F24 * (V1 - V4)
                + F14 * (V2 - V4)
                + 4 * F23 * FMs4 * (-V1 + V4)
                + 2 * F13 * FMs4 * (-V2 + V4)
                + FMs1
                * (
                    2 * F43 * (-V1 + V2)
                    + 6 * F22 * FMs4 * (V1 - V4)
                    + 4 * F23 * (-V1 + V4)
                )
            )
        )
    ) / (F1mF22 * F1mF43 * F2mF42)

    delta2 = (
        -(d4 * F1mF22 * F1mF4 * F2mF4 * (F12 + FMs2 * FMs4 + 2 * FMs1 * (FMs2 + FMs4)))
        + d1 * F1mF2 * F1mF4 * F2mF42 * (FMs1 * FMs2 + 2 * (FMs1 + FMs2) * FMs4 + F42)
        - 4 * F12 * F23 * V1
        + 3 * FMs1 * F24 * V1
        - 4 * FMs1 * F23 * FMs4 * V1
        + 3 * F24 * FMs4 * V1
        + 12 * F12 * FMs2 * F42 * V1
        - 4 * F23 * F42 * V1
        - 8 * F12 * F43 * V1
        + FMs1 * F44 * V1
        + F45 * V1
        + F15 * V2
        + F14 * FMs4 * V2
        - 8 * F13 * F42 * V2
        + 8 * F12 * F43 * V2
        - FMs1 * F44 * V2
        - F45 * V2
        - F1mF22
        * (
            F13
            + FMs2 * (3 * FMs2 - 4 * FMs4) * FMs4
            + F12 * (2 * FMs2 + FMs4)
            + FMs1 * (3 * FMs2 - 4 * FMs4) * (FMs2 + 2 * FMs4)
        )
        * V4
    ) / (F1mF22 * F1mF43 * F2mF42)

    delta3 = (
        d4 * F1mF22 * F1mF4 * F2mF4 * (2 * FMs1 + FMs2 + FMs4)
        - d1 * F1mF2 * F1mF4 * F2mF42 * (FMs1 + FMs2 + 2 * FMs4)
        + 2
        * (
            F44 * (-V1 + V2)
            + 2 * F12 * F2mF42 * (V1 - V4)
            + 2 * F22 * F42 * (V1 - V4)
            + 2 * F13 * FMs4 * (V2 - V4)
            + F24 * (-V1 + V4)
            + F14 * (-V2 + V4)
            + 2
            * FMs1
            * FMs4
            * (F42 * (V1 - V2) + F22 * (V1 - V4) + 2 * FMs2 * FMs4 * (-V1 + V4))
        )
    ) / (F1mF22 * F1mF43 * F2mF42)

    delta4 = (
        -(d4 * F1mF22 * F1mF4 * F2mF4)
        + d1 * F1mF2 * F1mF4 * F2mF42
        - 3 * FMs1 * F22 * V1
        + 2 * F23 * V1
        + 6 * FMs1 * FMs2 * FMs4 * V1
        - 3 * F22 * FMs4 * V1
        - 3 * FMs1 * F42 * V1
        + F43 * V1
        + F13 * V2
        - 3 * F12 * FMs4 * V2
        + 3 * FMs1 * F42 * V2
        - F43 * V2
        - F1mF22 * (FMs1 + 2 * FMs2 - 3 * FMs4) * V4
    ) / (F1mF22 * F1mF43 * F2mF42)

    Amp_Int = (fM_s ** (7.0 / 6.0)) / (
        delta0 + fM_s * (delta1 + fM_s * (delta2 + fM_s * (delta3 + fM_s * delta4)))
    )

    return Amp_Int


def get_mergerringdown_Amp(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    chip: FloatLike = 0.0,
) -> tuple[Float[Array, " n_freq"], FloatLike]:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))

    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    StotR = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)
    chia = chi1 - chi2

    fMs_RD, fMs_damp, _, _ = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2, chip)

    gamma2 = (
        IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[4, 0:amp_eqspin_indx], eta)
        + IMRPhenomX_utils.Amp_Eqspin_CV(
            amp_coeffs[4, amp_eqspin_indx:amp_uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Amp_Uneqspin_CV(
            amp_coeffs[4, amp_uneqspin_indx:], eta, StotR, chia
        )
    )
    gamma3 = (
        IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[5, 0:amp_eqspin_indx], eta)
        + IMRPhenomX_utils.Amp_Eqspin_CV(
            amp_coeffs[5, amp_eqspin_indx:amp_uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Amp_Uneqspin_CV(
            amp_coeffs[5, amp_uneqspin_indx:], eta, StotR, chia
        )
    )
    fMs_AmpRDMin = jnp.where(
        gamma2 <= 1.0,
        jnp.fabs(
            fMs_RD
            + fMs_damp * gamma3 * (jnp.sqrt(1.0 - gamma2 * gamma2) - 1.0) / gamma2
        ),
        jnp.fabs(fMs_RD + fMs_damp * (-1.0) * gamma3 / gamma2),
    )
    v1RD = (
        IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[6, 0:amp_eqspin_indx], eta)
        + IMRPhenomX_utils.Amp_Eqspin_CV(
            amp_coeffs[6, amp_eqspin_indx:amp_uneqspin_indx], eta, StotR
        )
        + IMRPhenomX_utils.Amp_Uneqspin_CV(
            amp_coeffs[6, amp_uneqspin_indx:], eta, StotR, chia
        )
    )
    FMs1 = fMs_AmpRDMin

    gamma1 = (
        (v1RD / (fMs_damp * gamma3))
        * (
            FMs1 * FMs1
            - 2.0 * FMs1 * fMs_RD
            + fMs_RD * fMs_RD
            + fMs_damp * fMs_damp * gamma3 * gamma3
        )
        * jnp.exp(((FMs1 - fMs_RD) * gamma2) / (fMs_damp * gamma3))
    )
    gammaR = gamma2 / (fMs_damp * gamma3)
    gammaD2 = (gamma3 * fMs_damp) * (gamma3 * fMs_damp)
    gammaD13 = fMs_damp * gamma1 * gamma3

    Amp_RD = (
        jnp.exp(-(fM_s - fMs_RD) * gammaR)
        * (gammaD13)
        / ((fM_s - fMs_RD) * (fM_s - fMs_RD) + gammaD2)
    )

    return Amp_RD, fMs_AmpRDMin


def Amp(
    f: Float[Array, " n_freq"],
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    D: FloatLike = 1.0,
    chip: float = 0.0,
) -> Float[Array, " n_freq"]:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    fM_s = f * M_s
    _, _, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)
    amp0 = 2.0 * jnp.sqrt(5.0 / (64.0 * PI)) * M_s**2 / ((D * MPC) / C)
    ampNorm = jnp.sqrt(2.0 * eta / 3.0) * (PI ** (-1.0 / 6.0))

    fMs_AmpInsMax = fMs_MECO + 0.25 * (fMs_ISCO - fMs_MECO)
    fMs_AmpMatchIN = fMs_AmpInsMax

    # Below
    Overallamp = amp0 * ampNorm

    Amp_Ins = get_inspiral_Amp(fM_s, theta, amp_coeffs, chip)
    Amp_RD, fMs_AmpRDMin = get_mergerringdown_Amp(fM_s, theta, amp_coeffs, chip)
    Amp_Int = get_intermediate_Amp(fM_s, theta, amp_coeffs, fMs_AmpRDMin, chip)

    Amp = (
        Amp_Ins * jnp.heaviside(fMs_AmpMatchIN - fM_s, 0.5)
        + jnp.heaviside(fM_s - fMs_AmpMatchIN, 0.5)
        * Amp_Int
        * jnp.heaviside(fMs_AmpRDMin - fM_s, 0.5)
        + Amp_RD
        * jnp.heaviside(fM_s - fMs_AmpRDMin, 0.5)
        * jnp.heaviside(IMRPhenomX_utils.fM_CUT - fM_s, 0.5)
    )

    return Overallamp * Amp * (fM_s ** (-7.0 / 6.0))


def _nrtidal_phase(
    frequencies: torch.Tensor,
    params: _NRTidalParams,
    *,
    frequency_series: bool,
) -> torch.Tensor:
    """Return all NRTidal and matter-spin phase contributions."""

    return (
        nrtidal_phase(
            frequencies,
            params.mass1,
            params.mass2,
            params.lambda1,
            params.lambda2,
            params.version,
            params.spin1z,
            params.spin2z,
            frequency_series=frequency_series,
        )
        + nrtidal_self_spin_phase(
            frequencies,
            params.mass1,
            params.mass2,
            params.spin1z,
            params.spin2z,
            params.quadrupole1 - 1.0,
            params.quadrupole2 - 1.0,
        )
        + nrtidal_higher_order_spin_phase(
            frequencies,
            params.mass1,
            params.mass2,
            params.spin1z,
            params.spin2z,
            params.quadrupole1,
            params.quadrupole2,
        )
    )


def _nrtidal_phase_derivative(
    params: _NRTidalParams,
    like: torch.Tensor,
) -> torch.Tensor:
    """Differentiate the tidal phase with respect to dimensionless Mf."""

    with torch.enable_grad():
        mf_alignment = torch.tensor(
            params.alignment_frequency
            * (params.mass1 + params.mass2)
            * MTSUN,
            dtype=like.dtype,
            device=like.device,
            requires_grad=True,
        )
        phase = _nrtidal_phase(
            mf_alignment / ((params.mass1 + params.mass2) * MTSUN),
            params,
            frequency_series=False,
        )
        derivative = torch.autograd.grad(phase, mf_alignment)[0]
    return derivative.detach()


def _gen_IMRPhenomXAS(
    f: Float[Array, " n_freq"],
    theta_intrinsic: Float[Array, "4"],
    theta_extrinsic: Float[Array, "3"],
    phase_coeffs: Float[Array, "13 49"],
    amp_coeffs: Float[Array, "7 42"],
    f_ref: float,
    nrtidal: _NRTidalParams | None = None,
) -> torch.Tensor:
    m1, m2, chi1, chi2 = theta_intrinsic
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN

    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)

    StotR = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)
    chia = chi1 - chi2

    fM_s = f * M_s
    fMs_RD, fMs_damp, _, _ = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)
    Psi = Phase(f, theta_intrinsic, phase_coeffs)

    # Generate the linear in f and constant contribution to the phase in order
    # to roll the waveform such that the peak is at the input tc and phic
    lina, linb, psi4tostrain = IMRPhenomX_utils.calc_phaseatpeak(
        eta, StotR, chia, delta
    )
    dphi22Ref = (
        PhaseDerivative(
            (fMs_RD - fMs_damp) / M_s,
            theta_intrinsic,
            phase_coeffs,
        )
        / M_s
    )
    linb = linb - dphi22Ref - 2.0 * PI * (500.0 + psi4tostrain)
    phifRef = (
        -(Phase(f_ref, theta_intrinsic, phase_coeffs) + linb * (f_ref * M_s) + lina)
        + PI / 4.0
    )
    ext_phase_contrib = 2.0 * PI * f * theta_extrinsic[1] + 2 * theta_extrinsic[2]
    Psi = Psi + (linb * fM_s) + lina + phifRef - 2 * PI + ext_phase_contrib

    A = Amp(f, theta_intrinsic, amp_coeffs, D=theta_extrinsic[0])
    if nrtidal is not None:
        phase_tidal = _nrtidal_phase(
            f,
            nrtidal,
            frequency_series=True,
        )
        reference = torch.as_tensor(
            f_ref,
            dtype=f.dtype,
            device=f.device,
        )
        phase_tidal_ref = _nrtidal_phase(
            reference,
            nrtidal,
            frequency_series=False,
        )

        # NRTidal shifts the waveform so the derivative of the complete phase
        # vanishes at min(f_max, f_merger). Derivatives here are with respect
        # to the dimensionless frequency Mf, matching LAL's convention.
        alignment_frequency = torch.as_tensor(
            nrtidal.alignment_frequency,
            dtype=f.dtype,
            device=f.device,
        )
        base_derivative = (
            PhaseDerivative(
                alignment_frequency,
                theta_intrinsic,
                phase_coeffs,
            )
            / M_s
        )
        tidal_derivative = _nrtidal_phase_derivative(nrtidal, f)
        tidal_linb = -base_derivative + tidal_derivative
        Psi = Psi + (tidal_linb - linb) * (fM_s - reference * M_s)
        Psi = Psi + phase_tidal_ref - phase_tidal

        amp0 = (
            2.0
            * jnp.sqrt(5.0 / (64.0 * PI))
            * M_s**2
            / ((theta_extrinsic[0] * MPC) / C)
        )
        A = A + amp0 * 2.0 * math.sqrt(PI / 5.0) * nrtidal_amplitude(
            f,
            nrtidal.mass1,
            nrtidal.mass2,
            nrtidal.lambda1,
            nrtidal.lambda2,
        )
        A = A * nrtidal_taper(f, nrtidal.merger_frequency)

    h0 = A * jnp.exp(1j * Psi)
    return h0


_DEFAULT_ONLY_ORDER_KEYS = (
    "phase_order",
    "amplitude_order",
    "spin_order",
    "tidal_order",
    "eccentricity_order",
)
_ZERO_ONLY_KEYS = (
    "spin1x",
    "spin1y",
    "spin2x",
    "spin2y",
    "eccentricity",
    "mean_per_ano",
    "lambda_octu1",
    "lambda_octu2",
    "quadfmode1",
    "quadfmode2",
    "octufmode1",
    "octufmode2",
    "dchi0",
    "dchi1",
    "dchi2",
    "dchi3",
    "dchi4",
    "dchi5",
    "dchi5l",
    "dchi6",
    "dchi6l",
    "dchi7",
    "dalpha1",
    "dalpha2",
    "dalpha3",
    "dalpha4",
    "dalpha5",
    "dbeta1",
    "dbeta2",
    "dbeta3",
    "frame_axis",
    "modes_choice",
    "side_bands",
)


def _is_nonzero(value):
    if value is None:
        return False
    try:
        return float(value) != 0.0
    except (TypeError, ValueError, OverflowError):
        return True


def _is_default_order(value):
    try:
        return float(value) == -1.0 and int(value) == -1
    except (TypeError, ValueError, OverflowError):
        return False


def _is_nonnegative_finite(value):
    if value is None:
        return True
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(value) and value >= 0.0


def _quadrupole_from_params(lambda_value, dquad_value):
    lambda_value = float(lambda_value or 0.0)
    dquad_value = float(dquad_value or 0.0)
    if lambda_value > 0.0 and dquad_value == 0.0:
        return nrtidal_quadrupole_from_lambda(lambda_value)
    return 1.0 + dquad_value


def imrphenomxas_native_supported(params):
    """Return whether ``params`` preserve the native XAS model semantics."""

    approximant = params.get("approximant", "IMRPhenomXAS")
    if approximant not in {
        "IMRPhenomXAS",
        "IMRPhenomXAS_NRTidalv2",
        "IMRPhenomXAS_NRTidalv3",
    }:
        return False
    if any(
        not _is_default_order(params.get(key, -1))
        for key in _DEFAULT_ONLY_ORDER_KEYS
    ):
        return False
    if any(_is_nonzero(params.get(key, 0.0)) for key in _ZERO_ONLY_KEYS):
        return False
    matter = (
        params.get("lambda1", 0.0),
        params.get("lambda2", 0.0),
        params.get("dquad_mon1", 0.0),
        params.get("dquad_mon2", 0.0),
    )
    if approximant == "IMRPhenomXAS":
        if any(_is_nonzero(value) for value in matter):
            return False
    else:
        lambdas = matter[:2]
        if not all(_is_nonnegative_finite(value) for value in lambdas):
            return False
        try:
            quadrupoles = (
                _quadrupole_from_params(matter[0], matter[2]),
                _quadrupole_from_params(matter[1], matter[3]),
            )
        except (TypeError, ValueError, OverflowError):
            return False
        if not all(math.isfinite(value) and value > 0.0 for value in quadrupoles):
            return False
    if params.get("mode_array") is not None or params.get("numrel_data", ""):
        return False
    return True


def imrphenomxas_sequence_native_supported(params):
    """Return whether arbitrary-frequency XAS generation is native."""

    return imrphenomxas_native_supported(params)


def _imrphenomxas_inputs(p, *, sequence=False):
    """Validate and normalize scalar inputs shared by both public APIs."""

    if not imrphenomxas_native_supported(p):
        raise ValueError(
            "IMRPhenomXAS parameters are not supported by the native Torch path"
        )
    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("native Torch IMRPhenomXAS requires TorchScheme")

    approximant = p.get("approximant", "IMRPhenomXAS")
    tidal_version = nrtidal_version(approximant)
    mass1 = float(p["mass1"])
    mass2 = float(p["mass2"])
    spin1z = float(p.get("spin1z", 0.0))
    spin2z = float(p.get("spin2z", 0.0))
    lambda1 = float(p.get("lambda1") or 0.0)
    lambda2 = float(p.get("lambda2") or 0.0)
    dquad1 = float(p.get("dquad_mon1") or 0.0)
    dquad2 = float(p.get("dquad_mon2") or 0.0)
    if mass2 > mass1:
        mass1, mass2 = mass2, mass1
        spin1z, spin2z = spin2z, spin1z
        lambda1, lambda2 = lambda2, lambda1
        dquad1, dquad2 = dquad2, dquad1

    f_ref = float(p.get("f_ref", 0.0))
    distance = float(p["distance"])
    inclination = float(p.get("inclination", 0.0))
    coa_phase = float(p.get("coa_phase", 0.0))
    # SimInspiralChooseFDWaveformSequence has no ascending-node argument and
    # ignores the corresponding PyCBC parameter.
    long_asc_nodes = 0.0 if sequence else float(p.get("long_asc_nodes", 0.0))

    if not all(math.isfinite(value) for value in (mass1, mass2)):
        raise ValueError("IMRPhenomXAS component masses must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("IMRPhenomXAS component masses must be positive")
    if mass1 / mass2 > 1000.0 + 1.0e-12:
        raise ValueError("IMRPhenomXAS is not valid beyond mass ratio 1000")
    if not all(
        math.isfinite(value)
        for value in (
            spin1z,
            spin2z,
            inclination,
            coa_phase,
            long_asc_nodes,
        )
    ):
        raise ValueError("IMRPhenomXAS spins and angles must be finite")
    if abs(spin1z) > 1.0 or abs(spin2z) > 1.0:
        raise ValueError("IMRPhenomXAS aligned spins must be between -1 and 1")
    if tidal_version is not None and not all(
        math.isfinite(value) and value >= 0.0
        for value in (lambda1, lambda2)
    ):
        raise ValueError("NRTidal deformabilities must be finite and non-negative")
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("IMRPhenomXAS distance must be finite and positive")
    if not math.isfinite(f_ref) or f_ref < 0.0:
        raise ValueError("IMRPhenomXAS f_ref must be finite and non-negative")

    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )
    return _IMRPhenomXASInputs(
        tidal_version=tidal_version,
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        lambda1=lambda1,
        lambda2=lambda2,
        dquad1=dquad1,
        dquad2=dquad2,
        f_ref=f_ref,
        distance=distance,
        inclination=inclination,
        coa_phase=coa_phase,
        long_asc_nodes=long_asc_nodes,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )


def _imrphenomxas_samples(
    inputs,
    frequencies,
    reference_frequency,
    active_f_max=None,
):
    """Evaluate the inclination-independent waveform at device frequencies."""

    intrinsic = torch.tensor(
        [inputs.mass1, inputs.mass2, inputs.spin1z, inputs.spin2z],
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    extrinsic = torch.tensor(
        [inputs.distance, 0.0, inputs.coa_phase],
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    phase_coeffs = IMRPhenomX_utils.PhenomX_phase_coeff_table.to(
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    amp_coeffs = IMRPhenomX_utils.PhenomX_amp_coeff_table.to(
        device=inputs.device,
        dtype=inputs.real_dtype,
    )

    nrtidal = None
    if inputs.tidal_version is not None:
        if active_f_max is None:
            raise ValueError("NRTidal generation requires an active maximum frequency")
        if isinstance(active_f_max, torch.Tensor):
            active_f_max = active_f_max.detach().item()
        quadrupole1 = _quadrupole_from_params(inputs.lambda1, inputs.dquad1)
        quadrupole2 = _quadrupole_from_params(inputs.lambda2, inputs.dquad2)
        if inputs.tidal_version == 3:
            merger_frequency = nrtidal_merger_frequency_v3(
                inputs.mass1,
                inputs.mass2,
                inputs.lambda1,
                inputs.lambda2,
                inputs.spin1z,
                inputs.spin2z,
            )
        else:
            merger_frequency = nrtidal_merger_frequency(
                inputs.mass1,
                inputs.mass2,
                inputs.lambda1,
                inputs.lambda2,
            )
        nrtidal = _NRTidalParams(
            mass1=inputs.mass1,
            mass2=inputs.mass2,
            spin1z=inputs.spin1z,
            spin2z=inputs.spin2z,
            lambda1=inputs.lambda1,
            lambda2=inputs.lambda2,
            quadrupole1=quadrupole1,
            quadrupole2=quadrupole2,
            merger_frequency=merger_frequency,
            alignment_frequency=min(active_f_max, merger_frequency),
            version=inputs.tidal_version,
        )

    with torch_context(frequencies):
        samples = _gen_IMRPhenomXAS(
            frequencies,
            intrinsic,
            extrinsic,
            phase_coeffs,
            amp_coeffs,
            reference_frequency,
            nrtidal,
        )
    return samples.to(inputs.complex_dtype)


def _polarizations_from_samples(samples, inclination, long_asc_nodes):
    """Project inclination-independent XAS samples into plus and cross."""

    cosi = math.cos(inclination)
    plus0 = -0.5 * (1.0 + cosi * cosi) * samples
    cross0 = complex(0.0, 1.0) * cosi * samples
    cos_nodes = math.cos(2.0 * long_asc_nodes)
    sin_nodes = math.sin(2.0 * long_asc_nodes)
    return (
        cos_nodes * plus0 + sin_nodes * cross0,
        cos_nodes * cross0 - sin_nodes * plus0,
    )


def _next_power_of_two(value):
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _imrphenomxas_core_torch(p):
    """Generate the active, inclination-independent XAS samples."""

    inputs = _imrphenomxas_inputs(p)
    delta_f = float(p["delta_f"])
    f_lower = float(p["f_lower"])
    f_final = float(p.get("f_final", 0.0))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomXAS frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomXAS delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomXAS f_final must be non-negative")

    total_mass_seconds = (inputs.mass1 + inputs.mass2) * MTSUN
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / total_mass_seconds
    layout_f_max = f_final if f_final > 0.0 else cutoff_frequency
    active_f_max = min(layout_f_max, cutoff_frequency)
    if active_f_max <= f_lower:
        raise ValueError("f_final (or the IMRPhenomXAS cutoff) is <= f_lower")

    npts = _next_power_of_two(layout_f_max / delta_f) + 1
    first_bin = int(f_lower / delta_f)
    # The direct XAS generator includes its upper bin. The public tidal model
    # is routed through XHM's interpolation, whose upper index is exclusive.
    stop_bin = int(active_f_max / delta_f) + (
        0 if inputs.tidal_version is not None else 1
    )

    frequencies = (
        torch.arange(
            first_bin,
            stop_bin,
            device=inputs.device,
            dtype=inputs.real_dtype,
        )
        * delta_f
    )
    reference_frequency = inputs.f_ref if inputs.f_ref > 0.0 else f_lower
    h22 = _imrphenomxas_samples(
        inputs,
        frequencies,
        reference_frequency,
        active_f_max,
    )

    epoch = -1.0 / delta_f
    return _IMRPhenomXASCore(
        polarization=h22,
        npts=npts,
        first_bin=first_bin,
        stop_bin=stop_bin,
        delta_f=delta_f,
        epoch=epoch,
    )


def _series_from_active_samples(core, samples):
    data = torch.zeros(
        core.npts,
        device=core.polarization.device,
        dtype=core.polarization.dtype,
    )
    data[core.first_bin : core.stop_bin] = samples
    return FrequencySeries(
        TorchArrayData(data),
        delta_f=core.delta_f,
        epoch=core.epoch,
        copy=False,
    )


def imrphenomxas_h2m2_torch(**p):
    r"""Generate LAL's positive-frequency :math:`h_{2,-2}` mode with Torch.

    The internal XAS kernel includes the inclination-independent polarization
    normalization used by ``SimInspiralChooseFDWaveform``. The mode-by-mode
    XHM interface instead returns the unnormalized spherical-harmonic mode;
    remove that factor here so both public interfaces can share one waveform
    evaluation.
    """

    core = _imrphenomxas_core_torch(p)
    return _series_from_active_samples(
        core,
        core.polarization / _XAS_MODE_POLARIZATION_FACTOR,
    )


def imrphenomxas_fd_torch(**p):
    """Generate aligned-spin IMRPhenomXAS polarizations with Torch."""

    core = _imrphenomxas_core_torch(p)
    inclination = float(p.get("inclination", 0.0))
    long_asc_nodes = float(p.get("long_asc_nodes", 0.0))
    plus, cross = _polarizations_from_samples(
        core.polarization,
        inclination,
        long_asc_nodes,
    )

    return (
        _series_from_active_samples(core, plus),
        _series_from_active_samples(core, cross),
    )


def _sequence_frequencies(sample_points, inputs):
    """Return validated sequence frequencies on the active Torch device."""

    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("IMRPhenomXAS sample_points must be a non-empty vector")
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("IMRPhenomXAS sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("IMRPhenomXAS sample_points must be positive")
    return frequencies


def imrphenomxas_fd_sequence_torch(**p):
    """Evaluate IMRPhenomXAS at arbitrary frequencies with Torch."""

    if not imrphenomxas_sequence_native_supported(p):
        raise ValueError(
            "IMRPhenomXAS sequence parameters are not supported by the "
            "native Torch path"
        )
    inputs = _imrphenomxas_inputs(p, sequence=True)
    frequencies = _sequence_frequencies(p["sample_points"], inputs)

    # LAL's sequence API treats the last sample as f_max, even if the input
    # is not sorted, and also applies the calibrated model cutoff.
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / (
        (inputs.mass1 + inputs.mass2) * MTSUN
    )
    active_f_max = torch.minimum(
        frequencies[-1],
        frequencies.new_tensor(cutoff_frequency),
    )
    active = frequencies <= active_f_max
    samples = torch.zeros(
        frequencies.shape,
        device=inputs.device,
        dtype=inputs.complex_dtype,
    )
    if bool(torch.any(active)):
        reference_frequency = (
            inputs.f_ref if inputs.f_ref > 0.0 else frequencies[0]
        )
        samples[active] = _imrphenomxas_samples(
            inputs,
            frequencies[active],
            reference_frequency,
            active_f_max,
        )

    plus, cross = _polarizations_from_samples(
        samples,
        inputs.inclination,
        inputs.long_asc_nodes,
    )
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )
