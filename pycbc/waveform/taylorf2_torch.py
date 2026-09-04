# Copyright (C) 2025
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Native Torch implementation of the aligned-spin TaylorF2 waveform.

The coefficient construction mirrors ``XLALSimInspiralPNPhasing_F2`` and the
waveform generator mirrors ``XLALSimInspiralTaylorF2Core``.  No lalsimulation
calls are made here.  Scalar coefficients are assembled as small NumPy arrays;
broadcast Torch inputs retain tensor coefficient storage.  All
frequency-dependent work for regular and arbitrary sampling is performed on
the active Torch device.
"""

import math
from dataclasses import dataclass

import numpy as _np

import pycbc.scheme as _scheme
from pycbc.constants import C_SI, GAMMA, MRSUN_SI, MTSUN_SI, PC_SI, PI
from pycbc.types.backend import torch_module_for

# Keep the same maximum order used by PNPhasingSeries in LAL (0..15)
_MAX_PN_ORDER = 16

# Apple MPS evaluates the frequency-dependent phase in float32.  Below this
# mass-ratio-scaled dimensionless frequency, roundoff in the leading inspiral
# phase can produce a measurable waveform mismatch.
_MPS_MIN_EQUAL_MASS_PHASE_MF = 1.0e-4


def _minimum_sequence_frequency(sample_points) -> float:
    """Return the lowest sample without moving Torch data through NumPy."""

    import torch

    from pycbc.types.array_torch import TorchArrayData

    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError("sample_points must be a non-empty vector")
        if not bool(torch.all(torch.isfinite(values))):
            raise ValueError("sample_points must be finite")
        return float(torch.min(values).item())

    frequencies = tuple(float(value) for value in sample_points)
    if not frequencies:
        raise ValueError("sample_points must be non-empty")
    if not all(math.isfinite(value) for value in frequencies):
        raise ValueError("sample_points must be finite")
    return min(frequencies)


class _PNPhasingSeries:
    """Lightweight stand-in for LAL PNPhasingSeries."""

    def __init__(self, like=None):
        torch = torch_module_for(like)
        if torch is not None:
            # Assemble tensor coefficients as independent values. Mutating
            # views of one stacked tensor while later terms still reference
            # those views invalidates Torch's saved autograd versions.
            zero = torch.zeros_like(like)
            self.v = [zero for _ in range(_MAX_PN_ORDER)]
            self.vlogv = [zero for _ in range(_MAX_PN_ORDER)]
            self.vlogvsq = [zero for _ in range(_MAX_PN_ORDER)]
        else:
            self.v = _np.zeros(_MAX_PN_ORDER, dtype=_np.float64)
            self.vlogv = _np.zeros(_MAX_PN_ORDER, dtype=_np.float64)
            self.vlogvsq = _np.zeros(_MAX_PN_ORDER, dtype=_np.float64)


@dataclass(frozen=True)
class _TaylorF2Inputs:
    """Normalized scalar inputs shared by regular and sequence sampling."""

    mass1: float
    mass2: float
    distance: float
    inclination: float
    coa_phase: float
    long_asc_nodes: float
    f_ref: float
    tidal_order: int
    lambda1: float
    lambda2: float
    phasing: _PNPhasingSeries
    device: object
    real_dtype: object
    complex_dtype: object


@dataclass(frozen=True)
class TaylorF2FDBatch:
    """Padded TaylorF2 frequency-domain polarizations and row metadata.

    ``hplus`` and ``hcross`` have shape ``(batch, frequency)``.  Samples in
    row ``i`` are non-zero only in ``[first_bins[i], end_bins[i])``; padding
    outside that interval is exactly zero.  Iteration yields the two
    polarizations so callers may continue to use ``hplus, hcross = result``.
    """

    hplus: object
    hcross: object
    delta_f: float
    epoch: float
    first_bins: object
    end_bins: object

    def __iter__(self):
        yield self.hplus
        yield self.hcross


# ---- Non-spinning phasing pieces (numeric coefficients copied verbatim) ----
# LAL reference: lalsimulation/lib/LALSimInspiralPNCoefficients.c lines 692-749
def _f2_2pn(eta):
    return 5.0 * (74.3 / 8.4 + 11.0 * eta) / 9.0


def _f2_3pn(_eta):
    return -16.0 * PI


def _f2_4pn(eta):
    return 5.0 * (3058.673 / 7.056 + 5429.0 / 7.0 * eta + 617.0 * eta * eta) / 72.0


def _f2_5pn(eta):
    return 5.0 / 9.0 * (772.9 / 8.4 - 13.0 * eta) * PI


def _f2_5pn_log(eta):
    return 5.0 / 3.0 * (772.9 / 8.4 - 13.0 * eta) * PI


def _f2_6pn_log(_eta):
    return -684.8 / 2.1


def _f2_6pn(eta):
    return (
        11583.231236531 / 4.694215680
        - 640.0 / 3.0 * PI * PI
        - 684.8 / 2.1 * GAMMA
        + eta * (-15737.765635 / 3.048192 + 225.5 / 1.2 * PI * PI)
        + eta * eta * 76.055 / 1.728
        - eta * eta * eta * 127.825 / 1.296
        + _f2_6pn_log(eta) * math.log(4.0)
    )


def _f2_7pn(eta):
    return PI * (
        770.96675 / 2.54016 + 378.515 / 1.512 * eta - 740.45 / 7.56 * eta * eta
    )


# ---- Spin-orbit pieces (LALSimInspiralPNCoefficients.c lines 756-805) ----
def _f2_3pn_so(m_by_m):
    return m_by_m * (25.0 + 38.0 / 3.0 * m_by_m)


def _f2_5pn_so(m_by_m):
    return -m_by_m * (
        1391.5 / 8.4
        - m_by_m * (1.0 - m_by_m) * 10.0 / 3.0
        + m_by_m * (1276.0 / 8.1 + m_by_m * (1.0 - m_by_m) * 170.0 / 9.0)
    )


def _f2_6pn_so(m_by_m):
    return PI * m_by_m * (1490.0 / 3.0 + m_by_m * 260.0)


def _f2_7pn_so(m_by_m, eta):
    return m_by_m * (
        -17097.8035 / 4.8384
        + eta * 28764.25 / 6.72
        + eta * eta * 47.35 / 1.44
        + m_by_m
        * (-7189.233785 / 1.524096 + eta * 458.555 / 3.024 - eta * eta * 534.5 / 7.2)
    )


# ---- Spin-squared / quadrupole-monopole pieces (lines 807-858) ----
def _f2_4pn_s1s2(eta):
    return 247.0 / 4.8 * eta


def _f2_4pn_s1s2_o(eta):
    return -721.0 / 4.8 * eta


def _f2_4pn_qm2_so(m_by_m):
    return -720.0 / 9.6 * m_by_m * m_by_m


def _f2_4pn_self2_so(m_by_m):
    return 1.0 / 9.6 * m_by_m * m_by_m


def _f2_4pn_qm2_s(m_by_m):
    return 240.0 / 9.6 * m_by_m * m_by_m


def _f2_4pn_self2_s(m_by_m):
    return -7.0 / 9.6 * m_by_m * m_by_m


def _f2_6pn_s1s2_o(eta):
    return (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta


def _f2_6pn_self2_s(m_by_m):
    return (
        (-4108.25 / 6.72 - 108.5 / 1.2 * m_by_m + 125.5 / 3.6 * m_by_m * m_by_m)
        * m_by_m
        * m_by_m
    )


def _f2_6pn_qm2_s(m_by_m):
    return (
        (4703.5 / 8.4 + 2935.0 / 6.0 * m_by_m - 120.0 * m_by_m * m_by_m)
        * m_by_m
        * m_by_m
    )


# ---- Tidal pieces (lines 860-915) ----
def _f2_10pn_tidal(m_by_m):
    return (-288.0 + 264.0 * m_by_m) * m_by_m**4


def _f2_12pn_tidal(m_by_m):
    return (
        -15895.0 / 28.0
        + 4595.0 / 28.0 * m_by_m
        + 5715.0 / 14.0 * m_by_m * m_by_m
        - 325.0 / 7.0 * m_by_m**3
    ) * m_by_m**4


def _f2_13pn_tidal(m_by_m):
    m4 = m_by_m**4
    return m4 * 24.0 * (12.0 - 11.0 * m_by_m) * PI


def _f2_14pn_tidal(m_by_m):
    m3 = m_by_m * m_by_m * m_by_m
    m4 = m3 * m_by_m
    return (
        -m4
        * 5.0
        * (
            193986935.0 / 571536.0
            - 14415613.0 / 381024.0 * m_by_m
            - 57859.0 / 378.0 * m_by_m * m_by_m
            - 209495.0 / 1512.0 * m3
            + 965.0 / 54.0 * m4
            - 4.0 * m4 * m_by_m
        )
    )


def _f2_15pn_tidal(m_by_m):
    m2 = m_by_m * m_by_m
    m3 = m2 * m_by_m
    m4 = m3 * m_by_m
    return (
        m4
        * (1.0 / 28.0)
        * PI
        * (27719.0 - 22415.0 * m_by_m + 7598.0 * m2 - 10520.0 * m3)
    )


def _apply_spin_terms(
    pfa,
    m1,
    m2,
    chi1,
    chi2,
    chi1sq,
    chi2sq,
    chi1dotchi2,
    spin_order,
    qm_def1,
    qm_def2,
):
    """Apply spin contributions with fallthrough matching LAL switch."""
    mtot = m1 + m2
    eta = m1 * m2 / (mtot * mtot)
    m1m = m1 / mtot
    m2m = m2 / mtot
    qm1 = 1.0 + qm_def1
    qm2 = 1.0 + qm_def2

    if spin_order in (-1, 7):  # ALL or 3.5PN
        pfa.v[7] = pfa.v[7] + _f2_7pn_so(m1m, eta) * chi1 + _f2_7pn_so(m2m, eta) * chi2
    if spin_order in (-1, 6, 7):
        pfa.v[6] = pfa.v[6] + (
            _f2_6pn_so(m1m) * chi1
            + _f2_6pn_so(m2m) * chi2
            + _f2_6pn_s1s2_o(eta) * chi1 * chi2
            + (_f2_6pn_qm2_s(m1m) * qm1 + _f2_6pn_self2_s(m1m)) * chi1sq
            + (_f2_6pn_qm2_s(m2m) * qm2 + _f2_6pn_self2_s(m2m)) * chi2sq
        )
    if spin_order in (-1, 5, 6, 7):
        so1 = _f2_5pn_so(m1m) * chi1
        so2 = _f2_5pn_so(m2m) * chi2
        pfa.v[5] = pfa.v[5] + so1 + so2
        pfa.vlogv[5] = pfa.vlogv[5] + 3.0 * (so1 + so2)
    if spin_order in (-1, 4, 5, 6, 7):
        pfa.v[4] = pfa.v[4] + (
            _f2_4pn_s1s2(eta) * chi1dotchi2
            + _f2_4pn_s1s2_o(eta) * chi1 * chi2
            + (_f2_4pn_qm2_so(m1m) * qm1 + _f2_4pn_self2_so(m1m)) * chi1 * chi1
            + (_f2_4pn_qm2_so(m2m) * qm2 + _f2_4pn_self2_so(m2m)) * chi2 * chi2
            + (_f2_4pn_qm2_s(m1m) * qm1 + _f2_4pn_self2_s(m1m)) * chi1sq
            + (_f2_4pn_qm2_s(m2m) * qm2 + _f2_4pn_self2_s(m2m)) * chi2sq
        )
    if spin_order in (-1, 3, 4, 5, 6, 7):
        pfa.v[3] = pfa.v[3] + _f2_3pn_so(m1m) * chi1 + _f2_3pn_so(m2m) * chi2


def _apply_tidal_terms(pfa, m1, m2, lambda1, lambda2, tidal_order):
    """Apply tidal terms following LAL fallthrough (lines 1040-1109)."""
    mtot = m1 + m2
    m1m = m1 / mtot
    m2m = m2 / mtot

    # LAL's default stops at 7PN. The 7.5PN term must be requested explicitly.
    if tidal_order == 15:
        pfa.v[15] = lambda1 * _f2_15pn_tidal(m1m) + lambda2 * _f2_15pn_tidal(m2m)
    if tidal_order in (-1, 14, 15):
        pfa.v[14] = lambda1 * _f2_14pn_tidal(m1m) + lambda2 * _f2_14pn_tidal(m2m)
    if tidal_order in (-1, 13, 14, 15):
        pfa.v[13] = lambda1 * _f2_13pn_tidal(m1m) + lambda2 * _f2_13pn_tidal(m2m)
    if tidal_order in (-1, 12, 13, 14, 15):
        pfa.v[12] = lambda1 * _f2_12pn_tidal(m1m) + lambda2 * _f2_12pn_tidal(m2m)
    if tidal_order in (-1, 10, 12, 13, 14, 15):
        pfa.v[10] = lambda1 * _f2_10pn_tidal(m1m) + lambda2 * _f2_10pn_tidal(m2m)


def taylorf2_aligned_phasing(
    mass1,
    mass2,
    chi1,
    chi2,
    *,
    spin_order=-1,
    tidal_order=-1,
    dchi=None,
    qm_def1=0.0,
    qm_def2=0.0,
    chi1sq=None,
    chi2sq=None,
    chi1dotchi2=None,
    lambda1=0.0,
    lambda2=0.0,
):
    """
    Return PN phasing coefficients for aligned-spin TaylorF2 without using LAL.

    Parameters
    ----------
    mass1, mass2 : float
        Component masses in solar masses.
    chi1, chi2 : float
        Dimensionless aligned spins.
    spin_order : int, optional
        PN spin order flag (use -1 for all; matches LAL enums).
    tidal_order : int, optional
        PN tidal order flag. LAL's default, ``-1``, includes terms through
        7PN; 7.5PN is selected explicitly with ``15``.
    dchi : dict, optional
        Non-GR fractional phase deviations keyed by 'dchi0'...'dchi7','dchi5l',
        'dchi6l'. Missing keys default to 0.
    qm_def1, qm_def2 : float, optional
        Spin-induced quadrupole deformation deltas (defaults to 0).
    chi1sq, chi2sq, chi1dotchi2 : float, optional
        Precomputed spin magnitudes / dot-product. If None, they are inferred
        from chi1/chi2 assuming alignment (i.e., chi1sq=chi1**2 etc.).

    Returns
    -------
    _PNPhasingSeries
        Object with arrays ``v``, ``vlogv``, ``vlogvsq`` identical to LAL.
    """
    dchi = dchi or {}
    chi1sq = chi1sq if chi1sq is not None else chi1 * chi1
    chi2sq = chi2sq if chi2sq is not None else chi2 * chi2
    chi1dotchi2 = chi1dotchi2 if chi1dotchi2 is not None else chi1 * chi2

    mtot = mass1 + mass2
    eta = (mass1 * mass2) / (mtot * mtot)
    pfaN = 3.0 / (128.0 * eta)

    pfa = _PNPhasingSeries(like=mass1)
    # Base non-spinning terms
    pfa.v[0] = 1.0
    pfa.v[1] = 0.0
    pfa.v[2] = _f2_2pn(eta)
    pfa.v[3] = _f2_3pn(eta)
    pfa.v[4] = _f2_4pn(eta)
    pfa.v[5] = _f2_5pn(eta)
    pfa.vlogv[5] = _f2_5pn_log(eta)
    pfa.v[6] = _f2_6pn(eta)
    pfa.vlogv[6] = _f2_6pn_log(eta)
    pfa.v[7] = _f2_7pn(eta)

    # Non-GR tweaks (LALSimInspiralPNCoefficients.c lines 972-980)
    pfa.v[0] = pfa.v[0] * (1.0 + dchi.get("dchi0", 0.0))
    pfa.v[1] = dchi.get("dchi1", 0.0)
    pfa.v[2] = pfa.v[2] * (1.0 + dchi.get("dchi2", 0.0))
    pfa.v[3] = pfa.v[3] * (1.0 + dchi.get("dchi3", 0.0))
    pfa.v[4] = pfa.v[4] * (1.0 + dchi.get("dchi4", 0.0))
    pfa.v[5] = pfa.v[5] * (1.0 + dchi.get("dchi5", 0.0))
    pfa.vlogv[5] = pfa.vlogv[5] * (1.0 + dchi.get("dchi5l", 0.0))
    pfa.v[6] = pfa.v[6] * (1.0 + dchi.get("dchi6", 0.0))
    pfa.vlogv[6] = pfa.vlogv[6] * (1.0 + dchi.get("dchi6l", 0.0))
    pfa.v[7] = pfa.v[7] * (1.0 + dchi.get("dchi7", 0.0))

    # Spin / tidal contributions
    _apply_spin_terms(
        pfa,
        mass1,
        mass2,
        chi1,
        chi2,
        chi1sq,
        chi2sq,
        chi1dotchi2,
        spin_order,
        qm_def1,
        qm_def2,
    )
    _apply_tidal_terms(pfa, mass1, mass2, lambda1, lambda2, tidal_order)

    # Stack tensor-valued coefficient lists only after their functional
    # assembly, preserving autograd through all physical parameters.
    if isinstance(pfa.v, list):
        import torch

        def stack_coefficients(values):
            return torch.stack(
                [
                    torch.broadcast_to(
                        torch.as_tensor(
                            value,
                            device=mass1.device,
                            dtype=mass1.dtype,
                        ),
                        mass1.shape,
                    )
                    for value in values
                ]
            )

        pfa.v = stack_coefficients(pfa.v)
        pfa.vlogv = stack_coefficients(pfa.vlogv)
        pfa.vlogvsq = stack_coefficients(pfa.vlogvsq)

    # Final global scaling
    pfa.v = pfa.v * pfaN
    pfa.vlogv = pfa.vlogv * pfaN
    pfa.vlogvsq = pfa.vlogvsq * pfaN
    return pfa


_PHASE_ORDERS = frozenset((-1, 0, 1, 2, 3, 4, 5, 6, 7))
_SPIN_ORDERS = frozenset((-1, 0, 1, 2, 3, 4, 5, 6, 7))
_TIDAL_ORDERS = frozenset((-1, 0, 10, 12, 13, 14, 15))
_DCHI_KEYS = (
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
)
_UNSUPPORTED_TIDAL_KEYS = (
    "lambda_octu1",
    "lambda_octu2",
    "quadfmode1",
    "quadfmode2",
    "octufmode1",
    "octufmode2",
)
_UNSUPPORTED_NON_GR_KEYS = (
    "dalpha1",
    "dalpha2",
    "dalpha3",
    "dalpha4",
    "dalpha5",
    "dbeta1",
    "dbeta2",
    "dbeta3",
)


def _as_order(value):
    """Normalize the integer-valued LAL order flags without raising."""
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _is_nonzero(value):
    """Return whether an optional scalar is set to a non-zero value."""
    if value is None:
        return False
    try:
        return float(value) != 0.0
    except (TypeError, ValueError, OverflowError):
        return True


def _taylorf2_features_supported(params):
    """Return whether ``params`` use features implemented by this port.

    Unsupported options deliberately fall back to lalsimulation in the public
    dispatcher.  The native path currently covers aligned spins, all TaylorF2
    phase/spin/tidal orders, testing-GR ``dchi`` terms, tidal deformability,
    spin-induced quadrupoles, reference phase, and polarization rotation.  PN
    amplitude corrections above Newtonian order are not yet implemented.
    """
    if _as_order(params.get("phase_order", -1)) not in _PHASE_ORDERS:
        return False
    if _as_order(params.get("spin_order", -1)) not in _SPIN_ORDERS:
        return False
    if _as_order(params.get("tidal_order", -1)) not in _TIDAL_ORDERS:
        return False
    if _as_order(params.get("amplitude_order", -1)) not in (-1, 0):
        return False
    if _as_order(params.get("eccentricity_order", -1)) != -1:
        return False

    if any(
        _is_nonzero(params.get(key, 0.0))
        for key in (
            "spin1x",
            "spin1y",
            "spin2x",
            "spin2y",
            "eccentricity",
            "mean_per_ano",
            "frame_axis",
            "modes_choice",
            "side_bands",
        )
    ):
        return False
    if params.get("mode_array") is not None or params.get("numrel_data", ""):
        return False
    if any(params.get(key) is not None for key in _UNSUPPORTED_TIDAL_KEYS):
        return False
    if any(_is_nonzero(params.get(key, 0.0)) for key in _UNSUPPORTED_NON_GR_KEYS):
        return False

    # Negative tidal deformabilities are rejected by LAL's universal-relation
    # and contact-frequency helpers. Let the fallback preserve that behavior.
    for key in ("lambda1", "lambda2"):
        value = params.get(key)
        if value is not None:
            try:
                value = float(value)
                if not math.isfinite(value) or value < 0.0:
                    return False
            except (TypeError, ValueError, OverflowError):
                return False
    return True


def _native_device_supported(params, *, sequence):
    """Bound single-precision inspiral-phase error on Apple MPS."""

    state = _scheme.mgr.state
    if not (
        isinstance(state, _scheme.TorchScheme) and state.torch_device.type == "mps"
    ):
        return True

    try:
        mass1 = float(params["mass1"])
        mass2 = float(params["mass2"])
        total_mass = mass1 + mass2
        symmetric_mass_ratio = mass1 * mass2 / total_mass**2
        if sequence:
            start_frequency = _minimum_sequence_frequency(params["sample_points"])
        else:
            start_frequency = float(params["f_lower"])
        reference_frequency = float(params.get("f_ref", 0.0) or 0.0)
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
        ZeroDivisionError,
    ):
        return False

    if not all(
        math.isfinite(value) and value > 0.0
        for value in (total_mass, symmetric_mass_ratio, start_frequency)
    ):
        return False
    if not math.isfinite(reference_frequency) or reference_frequency < 0.0:
        return False

    # A nonzero reference phase is evaluated independently before subtraction,
    # so its frequency must satisfy the same roundoff boundary as the first
    # waveform sample.
    phase_frequency = start_frequency
    if reference_frequency > 0.0:
        phase_frequency = min(phase_frequency, reference_frequency)
    minimum_phase_mf = _MPS_MIN_EQUAL_MASS_PHASE_MF * (0.25 / symmetric_mass_ratio) ** (
        3.0 / 5.0
    )
    phase_mf = total_mass * MTSUN_SI * phase_frequency
    return math.isfinite(phase_mf) and phase_mf >= minimum_phase_mf


def taylorf2_native_supported(params):
    """Return whether regular-grid TaylorF2 generation is native."""

    return _taylorf2_features_supported(params) and _native_device_supported(
        params,
        sequence=False,
    )


def _eos_q_from_lambda(lambda_tidal):
    """Mirror ``XLALSimInspiralEOSQfromLambda``.

    The native phasing helper consumes the deformation relative to the
    black-hole quadrupole value, so callers subtract one from this result.
    """
    if lambda_tidal < 0.0:
        raise ValueError("tidal deformability must be non-negative")
    if lambda_tidal < 0.5:
        return 1.0

    x = math.log(lambda_tidal)
    polynomial = 0.1940 + x * (0.0936 + x * (0.0474 + x * (-0.00421 + x * 0.000123)))
    return math.exp(polynomial)


def _radius_from_lambda(mass, lambda_tidal):
    """Mirror LAL's compactness fit and return the radius in metres."""
    if lambda_tidal < 0.0:
        raise ValueError("tidal deformability must be non-negative")
    if lambda_tidal <= 1.0e-15:
        compactness = 0.5
    else:
        log_lambda = math.log(lambda_tidal)
        compactness = 0.371 - 0.0391 * log_lambda + 0.001056 * log_lambda**2
        compactness = min(compactness, 0.5)
        if compactness < 0.0:
            raise ValueError("tidal deformability gives an invalid compactness")
    return MRSUN_SI * mass / compactness


def _contact_frequency(mass1, mass2, lambda1, lambda2):
    """Return the LAL TaylorF2 contact frequency in hertz."""
    radius1 = _radius_from_lambda(mass1, lambda1)
    radius2 = _radius_from_lambda(mass2, lambda2)
    radius_seconds = (radius1 + radius2) / C_SI
    total_mass_seconds = (mass1 + mass2) * MTSUN_SI
    return math.sqrt(total_mass_seconds / radius_seconds**3) / math.pi


def _evaluate_phase_polynomial(v, coeff, coeff_log, coeff_log_sq):
    """Evaluate the PN polynomial at one or many Torch ``v`` values."""
    import torch

    log_v = torch.log(v)
    log_v_sq = log_v * log_v
    result = coeff[15] + coeff_log[15] * log_v + coeff_log_sq[15] * log_v_sq
    for index in range(_MAX_PN_ORDER - 2, -1, -1):
        term = coeff[index] + coeff_log[index] * log_v + coeff_log_sq[index] * log_v_sq
        result = torch.addcmul(term, result, v)
    v2 = v * v
    return result / (v2 * v2 * v)


def taylorf2_sequence_native_supported(params):
    """Return whether arbitrary-frequency TaylorF2 generation is native."""

    return _taylorf2_features_supported(params) and _native_device_supported(
        params,
        sequence=True,
    )


def _taylorf2_inputs(
    params,
    *,
    sequence=False,
    infer_tidal_quadrupoles=True,
):
    """Validate scalars and construct phasing shared by both public APIs."""
    import torch

    supported = (
        taylorf2_sequence_native_supported(params)
        if sequence
        else taylorf2_native_supported(params)
    )
    if not supported:
        raise ValueError(
            "TaylorF2 parameters are not supported by the native Torch path"
        )

    mass1 = float(params["mass1"])
    mass2 = float(params["mass2"])
    spin1z = float(params.get("spin1z", 0.0))
    spin2z = float(params.get("spin2z", 0.0))
    distance = float(params.get("distance", 1.0))
    inclination = float(params.get("inclination", 0.0))
    coa_phase = float(params.get("coa_phase", 0.0))
    # SimInspiralChooseFDWaveformSequence has no ascending-node argument and
    # ignores the corresponding PyCBC parameter.
    long_asc_nodes = 0.0 if sequence else float(params.get("long_asc_nodes", 0.0))
    f_ref = float(params.get("f_ref", 0.0))
    phase_order = _as_order(params.get("phase_order", -1))
    spin_order = _as_order(params.get("spin_order", -1))
    tidal_order = _as_order(params.get("tidal_order", -1))

    if not math.isfinite(mass1) or not math.isfinite(mass2):
        raise ValueError("TaylorF2 component masses must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("TaylorF2 component masses must be positive")
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
        raise ValueError("TaylorF2 spins and angles must be finite")
    if not math.isfinite(distance):
        raise ValueError("TaylorF2 distance must be finite")
    if distance <= 0.0:
        raise ValueError("TaylorF2 distance must be positive")
    if not math.isfinite(f_ref) or f_ref < 0.0:
        raise ValueError("TaylorF2 f_ref must be finite and non-negative")

    lambda1 = float(params.get("lambda1") or 0.0)
    lambda2 = float(params.get("lambda2") or 0.0)

    dquad1 = float(params.get("dquad_mon1") or 0.0)
    if infer_tidal_quadrupoles and lambda1 > 0.0 and dquad1 == 0.0:
        dquad1 = _eos_q_from_lambda(lambda1) - 1.0
    dquad2 = float(params.get("dquad_mon2") or 0.0)
    if infer_tidal_quadrupoles and lambda2 > 0.0 and dquad2 == 0.0:
        dquad2 = _eos_q_from_lambda(lambda2) - 1.0

    dchi = {key: float(params.get(key) or 0.0) for key in _DCHI_KEYS}
    phasing = taylorf2_aligned_phasing(
        mass1,
        mass2,
        spin1z,
        spin2z,
        spin_order=spin_order,
        tidal_order=tidal_order,
        dchi=dchi,
        qm_def1=float(dquad1),
        qm_def2=float(dquad2),
        lambda1=lambda1,
        lambda2=lambda2,
    )

    # The orbital phase order truncates only the point-particle/spin series.
    # Tidal terms at indices 10--15 remain controlled by tidal_order.
    if phase_order != -1:
        phasing.v[phase_order + 1 : 8] = 0.0
        phasing.vlogv[phase_order + 1 : 8] = 0.0
        phasing.vlogvsq[phase_order + 1 : 8] = 0.0

    state = _scheme.mgr.state
    device = getattr(state, "torch_device", torch.device("cpu"))
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    return _TaylorF2Inputs(
        mass1=mass1,
        mass2=mass2,
        distance=distance,
        inclination=inclination,
        coa_phase=coa_phase,
        long_asc_nodes=long_asc_nodes,
        f_ref=f_ref,
        tidal_order=tidal_order,
        lambda1=lambda1,
        lambda2=lambda2,
        phasing=phasing,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )


def _taylorf2_samples(inputs, frequencies, time_shift=0.0):
    """Evaluate the inclination-independent waveform at device frequencies."""
    import torch

    mass1 = inputs.mass1
    mass2 = inputs.mass2
    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    pi_mass = math.pi * total_mass * MTSUN_SI
    phasing = inputs.phasing
    real_dtype = inputs.real_dtype
    device = inputs.device
    coeff = torch.as_tensor(phasing.v, dtype=real_dtype, device=device)
    coeff_log = torch.as_tensor(phasing.vlogv, dtype=real_dtype, device=device)
    coeff_log_sq = torch.as_tensor(
        phasing.vlogvsq,
        dtype=real_dtype,
        device=device,
    )
    velocity = torch.pow(pi_mass * frequencies, 1.0 / 3.0)
    phase = _evaluate_phase_polynomial(
        velocity,
        coeff,
        coeff_log,
        coeff_log_sq,
    )

    if inputs.f_ref == 0.0:
        reference_phase = torch.zeros((), dtype=real_dtype, device=device)
    else:
        reference_velocity = torch.pow(
            torch.as_tensor(
                pi_mass * inputs.f_ref,
                dtype=real_dtype,
                device=device,
            ),
            1.0 / 3.0,
        )
        reference_phase = _evaluate_phase_polynomial(
            reference_velocity,
            coeff,
            coeff_log,
            coeff_log_sq,
        )

    phase = (
        phase
        + 2.0 * math.pi * time_shift * frequencies
        - 2.0 * inputs.coa_phase
        - reference_phase
    )
    distance_metres = inputs.distance * 1.0e6 * PC_SI
    amplitude0 = (
        -4.0
        * mass1
        * mass2
        / distance_metres
        * MRSUN_SI
        * MTSUN_SI
        * math.sqrt(math.pi / 12.0)
    )
    amplitude = amplitude0 * math.sqrt(5.0 / (32.0 * eta)) * torch.pow(velocity, -3.5)
    return torch.polar(amplitude, -(phase - math.pi / 4.0)).to(inputs.complex_dtype)


def _taylorf2_polarizations(samples, inputs):
    """Project inclination-independent samples into plus and cross."""

    cos_inclination = math.cos(inputs.inclination)
    plus0 = samples * (0.5 * (1.0 + cos_inclination**2))
    cross0 = samples * complex(0.0, -cos_inclination)
    cos_nodes = math.cos(2.0 * inputs.long_asc_nodes)
    sin_nodes = math.sin(2.0 * inputs.long_asc_nodes)
    return (
        cos_nodes * plus0 + sin_nodes * cross0,
        cos_nodes * cross0 - sin_nodes * plus0,
    )


def taylorf2_fd_torch(**params):
    """Generate aligned-spin TaylorF2 polarizations on the active Torch device.

    Callers should first use :func:`taylorf2_native_supported`; the public
    waveform dispatcher does so and routes unsupported options to LAL.
    """
    import torch

    from pycbc.types import FrequencySeries
    from pycbc.types.array_torch import TorchArrayData

    inputs = _taylorf2_inputs(params)
    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("TaylorF2 frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("TaylorF2 delta_f and f_lower must be positive")

    mass1 = inputs.mass1
    mass2 = inputs.mass2
    pi_mass = math.pi * (mass1 + mass2) * MTSUN_SI
    f_isco = 1.0 / (6.0**1.5 * pi_mass)
    if f_final == 0.0:
        if inputs.tidal_order == 0:
            f_max = f_isco
        else:
            f_max = min(
                f_isco,
                _contact_frequency(
                    mass1,
                    mass2,
                    inputs.lambda1,
                    inputs.lambda2,
                ),
            )
    else:
        f_max = f_final
    if f_max <= f_lower:
        raise ValueError("TaylorF2 ending frequency must exceed f_lower")

    length = int(f_max / delta_f + 1.0)
    first_bin = int(math.ceil(f_lower / delta_f))
    if first_bin >= length:
        raise ValueError("TaylorF2 frequency range contains no sampled bins")
    device = inputs.device
    real_dtype = inputs.real_dtype
    raw = torch.zeros(
        length,
        dtype=inputs.complex_dtype,
        device=device,
    )
    if first_bin < length:
        frequencies = (
            torch.arange(first_bin, length, dtype=real_dtype, device=device) * delta_f
        )
        epoch = -1.0 / delta_f
        raw[first_bin:] = _taylorf2_samples(
            inputs,
            frequencies,
            time_shift=epoch,
        )
    plus, cross = _taylorf2_polarizations(raw, inputs)
    epoch = -1.0 / delta_f
    return (
        FrequencySeries(TorchArrayData(plus), delta_f=delta_f, epoch=epoch, copy=False),
        FrequencySeries(
            TorchArrayData(cross), delta_f=delta_f, epoch=epoch, copy=False
        ),
    )


def _taylorf2_sequence_frequencies(sample_points, inputs):
    """Move arbitrary sample points directly onto the active Torch device."""
    import torch

    from pycbc.types.array_torch import TorchArrayData

    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError("TaylorF2 sample_points must be a non-empty vector")
    return frequencies


def taylorf2_fd_sequence_torch(**params):
    """Evaluate aligned-spin TaylorF2 at arbitrary frequencies with Torch."""
    from pycbc.types import Array as PyCBCArray
    from pycbc.types.array_torch import TorchArrayData

    if not taylorf2_sequence_native_supported(params):
        raise ValueError(
            "TaylorF2 sequence parameters are not supported by the native Torch path"
        )
    inputs = _taylorf2_inputs(params, sequence=True)
    frequencies = _taylorf2_sequence_frequencies(
        params["sample_points"],
        inputs,
    )
    samples = _taylorf2_samples(inputs, frequencies)
    plus, cross = _taylorf2_polarizations(samples, inputs)
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


_BATCH_PARAMETER_DEFAULTS = {
    "spin1z": 0.0,
    "spin2z": 0.0,
    "distance": 1.0,
    "inclination": 0.0,
    "coa_phase": 0.0,
    "long_asc_nodes": 0.0,
    "f_ref": 0.0,
    "f_final": 0.0,
    "lambda1": 0.0,
    "lambda2": 0.0,
    "dquad_mon1": 0.0,
    "dquad_mon2": 0.0,
    **dict.fromkeys(_DCHI_KEYS, 0.0),
}

_BATCH_ZERO_ONLY_PARAMETERS = (
    "spin1x",
    "spin1y",
    "spin2x",
    "spin2y",
    "eccentricity",
    "mean_per_ano",
    "frame_axis",
    "modes_choice",
    "side_bands",
    *_UNSUPPORTED_NON_GR_KEYS,
)

_BATCH_DISCRETE_PARAMETERS = {
    "phase_order": (-1, _PHASE_ORDERS),
    "spin_order": (-1, _SPIN_ORDERS),
    "tidal_order": (-1, _TIDAL_ORDERS),
    "amplitude_order": (-1, frozenset((-1, 0))),
    "eccentricity_order": (-1, frozenset((-1,))),
}


def _batch_unwrap(value):
    """Return the underlying numeric object without a NumPy round trip."""

    value = getattr(value, "_data", value)
    return getattr(value, "tensor", value)


def _batch_tensor(value, name, torch, device, dtype):
    """Convert one scalar-or-vector batch argument to the active device."""

    value = _batch_unwrap(value)
    if isinstance(value, torch.Tensor):
        source = value
        nonreal = source.dtype == torch.bool or source.is_complex()
    else:
        try:
            source = _np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TaylorF2 batch parameter {name!r} must be numeric"
            ) from exc
        nonreal = source.dtype.kind in ("b", "c")
    if nonreal:
        raise ValueError(f"TaylorF2 batch parameter {name!r} must be real-valued")
    try:
        result = torch.as_tensor(source, device=device, dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"TaylorF2 batch parameter {name!r} must be numeric") from exc
    if result.ndim > 1:
        raise ValueError(f"TaylorF2 batch parameter {name!r} must be scalar or 1-D")
    if result.ndim == 1 and result.numel() == 0:
        raise ValueError(f"TaylorF2 batch parameter {name!r} must not be empty")
    return result


def _batch_shared_scalar(params, name, default, torch):
    """Read a shared scalar without accepting singleton vectors."""

    value = _batch_unwrap(params.get(name, default))
    if isinstance(value, torch.Tensor):
        source = value
        nonreal = source.dtype == torch.bool or source.is_complex()
        if source.ndim != 0 or nonreal:
            raise ValueError(f"TaylorF2 batch parameter {name!r} must be a real scalar")
        if source.requires_grad:
            raise ValueError(
                f"TaylorF2 batch parameter {name!r} cannot require gradients"
            )
        numeric = float(source.item())
    else:
        try:
            source = _np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TaylorF2 batch parameter {name!r} must be a scalar"
            ) from exc
        nonreal = source.dtype.kind in ("b", "c")
        if source.ndim != 0 or nonreal:
            raise ValueError(f"TaylorF2 batch parameter {name!r} must be a real scalar")
        try:
            numeric = float(source)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TaylorF2 batch parameter {name!r} must be a scalar"
            ) from exc
    if not math.isfinite(numeric):
        raise ValueError(f"TaylorF2 batch parameter {name!r} must be finite")
    return numeric


def _batch_shared_order(params, name, default, allowed, torch):
    """Read a shared, exactly integer-valued PN order flag."""

    numeric = _batch_shared_scalar(
        params,
        name,
        default,
        torch,
    )
    order = int(numeric)
    if numeric != order or order not in allowed:
        choices = ", ".join(str(item) for item in sorted(allowed))
        raise ValueError(f"TaylorF2 batch parameter {name!r} must be one of {choices}")
    return order


def _batch_validate(checks):
    """Validate row predicates with one device-to-host synchronization."""

    import torch

    checks = tuple(checks)
    invalid_locations = (
        torch.nonzero(
            torch.stack([~condition for condition, _message in checks]),
            as_tuple=False,
        )
        .detach()
        .cpu()
        .tolist()
    )
    if invalid_locations:
        first_check = min(location[0] for location in invalid_locations)
        bad_rows = [row for check, row in invalid_locations if check == first_check]
        raise ValueError(f"{checks[first_check][1]}; invalid rows: {bad_rows}")


def _batch_require(condition, message):
    """Raise a row-oriented validation error when ``condition`` is false."""

    _batch_validate(((condition, message),))


def _batch_eos_q_from_lambda(lambda_tidal, torch):
    """Tensor form of :func:`_eos_q_from_lambda`."""

    # Clamp before taking the logarithm so inactive black-hole rows do not
    # inject NaNs into the autograd graph evaluated by ``torch.where``.
    x = torch.log(torch.clamp(lambda_tidal, min=0.5))
    polynomial = 0.1940 + x * (0.0936 + x * (0.0474 + x * (-0.00421 + x * 0.000123)))
    return torch.where(
        lambda_tidal < 0.5,
        torch.ones_like(lambda_tidal),
        torch.exp(polynomial),
    )


def _batch_contact_frequency(
    mass1,
    mass2,
    lambda1,
    lambda2,
    torch,
):
    """Tensor form of :func:`_contact_frequency` plus a validity mask."""

    def compactness(lambda_tidal):
        positive = lambda_tidal > 1.0e-15
        log_lambda = torch.log(torch.clamp(lambda_tidal, min=1.0e-15))
        fitted = 0.371 - 0.0391 * log_lambda + 0.001056 * log_lambda**2
        fitted = torch.minimum(fitted, torch.full_like(fitted, 0.5))
        value = torch.where(positive, fitted, torch.full_like(fitted, 0.5))
        return value, (~positive) | (fitted > 0.0)

    compactness1, valid1 = compactness(lambda1)
    compactness2, valid2 = compactness(lambda2)
    # The clamped denominator is used only to keep invalid rows numerically
    # defined until the caller can report their row indices.
    radius1 = MRSUN_SI * mass1 / torch.clamp(compactness1, min=1.0e-30)
    radius2 = MRSUN_SI * mass2 / torch.clamp(compactness2, min=1.0e-30)
    radius_seconds = (radius1 + radius2) / C_SI
    total_mass_seconds = (mass1 + mass2) * MTSUN_SI
    frequency = torch.sqrt(total_mass_seconds / radius_seconds**3) / math.pi
    return frequency, valid1 & valid2 & torch.isfinite(frequency)


def taylorf2_fd_batch(**params):
    """Generate an explicit batch of aligned-spin TaylorF2 waveforms.

    Physical inputs may be scalars or one-dimensional values. Scalars and
    length-one vectors broadcast to the common batch length; other vector
    lengths must agree. ``delta_f`` and all PN order flags are shared scalars.
    Per-row ``f_lower`` and ``f_final`` determine the exact zero-padded output
    support. Continuous Torch inputs retain their autograd connection.

    Unlike the scalar waveform dispatcher, this API never falls back to LAL
    and never interprets a vector as an implicit request for generic batching.
    """

    import torch

    state = _scheme.mgr.state
    if not isinstance(state, _scheme.TorchScheme):
        raise RuntimeError("TaylorF2 batch generation requires an active TorchScheme")
    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128

    allowed_keys = {
        "mass1",
        "mass2",
        "delta_f",
        "f_lower",
        "approximant",
        "mode_array",
        "numrel_data",
        *_BATCH_PARAMETER_DEFAULTS,
        *_BATCH_ZERO_ONLY_PARAMETERS,
        *_BATCH_DISCRETE_PARAMETERS,
        *_UNSUPPORTED_TIDAL_KEYS,
    }
    unknown = sorted(set(params) - allowed_keys)
    if unknown:
        raise ValueError(
            "TaylorF2 batch parameters are not supported: " + ", ".join(unknown)
        )
    approximant = params.get("approximant")
    if approximant not in (None, "TaylorF2"):
        raise ValueError("TaylorF2 batch generation supports only TaylorF2")
    required = ("mass1", "mass2", "delta_f", "f_lower")
    missing = [name for name in required if name not in params]
    if missing:
        raise ValueError("TaylorF2 batch generation requires: " + ", ".join(missing))
    if params.get("mode_array") is not None:
        raise ValueError("TaylorF2 batch mode_array is not supported")
    if params.get("numrel_data", ""):
        raise ValueError("TaylorF2 batch numrel_data is not supported")
    unsupported_tidal = [
        key for key in _UNSUPPORTED_TIDAL_KEYS if params.get(key) is not None
    ]
    if unsupported_tidal:
        raise ValueError(
            "TaylorF2 batch tidal parameters are not supported: "
            + ", ".join(unsupported_tidal)
        )

    delta_f = _batch_shared_scalar(
        params,
        "delta_f",
        None,
        torch,
    )
    if delta_f <= 0.0:
        raise ValueError("TaylorF2 batch delta_f must be positive")
    orders = {
        name: _batch_shared_order(
            params,
            name,
            default,
            allowed,
            torch,
        )
        for name, (default, allowed) in _BATCH_DISCRETE_PARAMETERS.items()
    }

    numeric = {}
    default_tensors = {}
    defaults = {
        "mass1": None,
        "mass2": None,
        "f_lower": None,
        **_BATCH_PARAMETER_DEFAULTS,
        **dict.fromkeys(_BATCH_ZERO_ONLY_PARAMETERS, 0.0),
    }
    for name, default in defaults.items():
        if name in params:
            numeric[name] = _batch_tensor(
                params[name],
                name,
                torch,
                device,
                real_dtype,
            )
        else:
            if default not in default_tensors:
                default_tensors[default] = _batch_tensor(
                    default,
                    name,
                    torch,
                    device,
                    real_dtype,
                )
            numeric[name] = default_tensors[default]

    lengths = {
        name: value.numel() for name, value in numeric.items() if value.ndim == 1
    }
    batch_size = max(lengths.values(), default=1)
    mismatched = {
        name: length
        for name, length in lengths.items()
        if length not in (1, batch_size)
    }
    if mismatched:
        detail = ", ".join(
            f"{name}={length}" for name, length in sorted(mismatched.items())
        )
        raise ValueError(
            "TaylorF2 batch vector lengths must be one or the common batch "
            f"length {batch_size}; got {detail}"
        )
    for name, value in numeric.items():
        if value.ndim == 0:
            numeric[name] = value.expand(batch_size)
        elif value.numel() == 1 and batch_size != 1:
            numeric[name] = value.expand(batch_size)

    mass1 = numeric["mass1"]
    mass2 = numeric["mass2"]
    f_lower = numeric["f_lower"]
    f_final = numeric["f_final"]
    input_checks = [
        (
            torch.isfinite(value),
            f"TaylorF2 batch parameter {name!r} must be finite",
        )
        for name, value in numeric.items()
    ]
    input_checks.extend(
        (
            numeric[name] == 0.0,
            f"TaylorF2 batch parameter {name!r} is not supported",
        )
        for name in _BATCH_ZERO_ONLY_PARAMETERS
    )
    input_checks.extend(
        (
            condition,
            message,
        )
        for condition, message in (
            (mass1 > 0.0, "TaylorF2 batch mass1 must be positive"),
            (mass2 > 0.0, "TaylorF2 batch mass2 must be positive"),
            (
                numeric["distance"] > 0.0,
                "TaylorF2 batch distance must be positive",
            ),
            (f_lower > 0.0, "TaylorF2 batch f_lower must be positive"),
            (
                f_final >= 0.0,
                "TaylorF2 batch f_final must be non-negative",
            ),
        )
    )
    for name in ("f_ref", "lambda1", "lambda2"):
        input_checks.append(
            (
                numeric[name] >= 0.0,
                f"TaylorF2 batch {name} must be non-negative",
            )
        )
    _batch_validate(input_checks)

    for name in ("f_lower", "f_final"):
        if numeric[name].requires_grad:
            raise ValueError(
                f"TaylorF2 batch {name} defines discrete support and cannot "
                "require gradients"
            )

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    derived_checks = [
        (
            torch.isfinite(total_mass) & torch.isfinite(eta) & (eta > 0.0),
            "TaylorF2 batch masses give invalid total mass or mass ratio",
        )
    ]
    if device.type == "mps":
        minimum_phase_mf = _MPS_MIN_EQUAL_MASS_PHASE_MF * (0.25 / eta) ** (3.0 / 5.0)
        phase_frequency = torch.where(
            numeric["f_ref"] > 0.0,
            torch.minimum(f_lower, numeric["f_ref"]),
            f_lower,
        )
        phase_mf = total_mass * MTSUN_SI * phase_frequency
        derived_checks.append(
            (
                phase_mf >= minimum_phase_mf,
                "TaylorF2 batch frequencies are below the accurate MPS range",
            )
        )
    _batch_validate(derived_checks)

    lambda1 = numeric["lambda1"]
    lambda2 = numeric["lambda2"]
    dquad1 = numeric["dquad_mon1"]
    dquad2 = numeric["dquad_mon2"]
    dquad1 = torch.where(
        (lambda1 > 0.0) & (dquad1 == 0.0),
        _batch_eos_q_from_lambda(lambda1, torch) - 1.0,
        dquad1,
    )
    dquad2 = torch.where(
        (lambda2 > 0.0) & (dquad2 == 0.0),
        _batch_eos_q_from_lambda(lambda2, torch) - 1.0,
        dquad2,
    )
    _batch_require(
        torch.isfinite(dquad1) & torch.isfinite(dquad2),
        "TaylorF2 batch inferred quadrupoles must be finite",
    )

    phasing = taylorf2_aligned_phasing(
        mass1,
        mass2,
        numeric["spin1z"],
        numeric["spin2z"],
        spin_order=orders["spin_order"],
        tidal_order=orders["tidal_order"],
        dchi={key: numeric[key] for key in _DCHI_KEYS},
        qm_def1=dquad1,
        qm_def2=dquad2,
        lambda1=lambda1,
        lambda2=lambda2,
    )
    phase_order = orders["phase_order"]
    if phase_order != -1:
        order_mask = torch.ones(
            _MAX_PN_ORDER,
            1,
            dtype=real_dtype,
            device=device,
        )
        order_mask[phase_order + 1 : 8] = 0.0
        phasing.v = phasing.v * order_mask
        phasing.vlogv = phasing.vlogv * order_mask
        phasing.vlogvsq = phasing.vlogvsq * order_mask

    pi_mass = math.pi * total_mass * MTSUN_SI
    f_isco = 1.0 / (6.0**1.5 * pi_mass)
    use_default_cutoff = f_final == 0.0
    if orders["tidal_order"] == 0:
        default_cutoff = f_isco
    else:
        contact, valid_contact = _batch_contact_frequency(
            mass1,
            mass2,
            lambda1,
            lambda2,
            torch,
        )
        _batch_require(
            (~use_default_cutoff) | valid_contact,
            "TaylorF2 batch tidal deformability gives invalid compactness",
        )
        default_cutoff = torch.minimum(f_isco, contact)
    f_max = torch.where(use_default_cutoff, default_cutoff, f_final)
    _batch_require(
        torch.isfinite(f_max) & (f_max > f_lower),
        "TaylorF2 batch ending frequency must exceed f_lower",
    )

    first_bins = torch.ceil(f_lower / delta_f).to(torch.int64)
    end_bins = torch.floor(f_max / delta_f).to(torch.int64) + 1
    _batch_require(
        first_bins < end_bins,
        "TaylorF2 batch frequency range contains no sampled bins",
    )
    first_active = int(torch.min(first_bins).item())
    output_length = int(torch.max(end_bins).item())
    bin_numbers = torch.arange(
        first_active,
        output_length,
        dtype=real_dtype,
        device=device,
    )
    frequencies = bin_numbers[None, :] * delta_f

    coeff = phasing.v.unsqueeze(-1)
    coeff_log = phasing.vlogv.unsqueeze(-1)
    coeff_log_sq = phasing.vlogvsq.unsqueeze(-1)
    velocity = torch.pow(pi_mass[:, None] * frequencies, 1.0 / 3.0)
    phase = _evaluate_phase_polynomial(
        velocity,
        coeff,
        coeff_log,
        coeff_log_sq,
    )
    reference_velocity = torch.pow(
        pi_mass
        * torch.where(
            numeric["f_ref"] > 0.0,
            numeric["f_ref"],
            torch.ones_like(numeric["f_ref"]),
        ),
        1.0 / 3.0,
    )
    reference_phase = _evaluate_phase_polynomial(
        reference_velocity[:, None],
        coeff,
        coeff_log,
        coeff_log_sq,
    ).squeeze(-1)
    reference_phase = torch.where(
        numeric["f_ref"] > 0.0,
        reference_phase,
        torch.zeros_like(reference_phase),
    )
    epoch = -1.0 / delta_f
    phase = (
        phase
        + 2.0 * math.pi * epoch * frequencies
        - 2.0 * numeric["coa_phase"][:, None]
        - reference_phase[:, None]
    )
    distance_metres = numeric["distance"] * 1.0e6 * PC_SI
    amplitude0 = (
        -4.0
        * mass1
        * mass2
        / distance_metres
        * MRSUN_SI
        * MTSUN_SI
        * math.sqrt(math.pi / 12.0)
    )
    amplitude = (
        amplitude0[:, None]
        * torch.sqrt(5.0 / (32.0 * eta))[:, None]
        * torch.pow(velocity, -3.5)
    )
    phase_factor = torch.exp(-(phase - math.pi / 4.0).to(complex_dtype) * 1j)
    samples = amplitude.to(complex_dtype) * phase_factor

    cos_inclination = torch.cos(numeric["inclination"])
    plus0 = samples * (0.5 * (1.0 + cos_inclination**2))[:, None]
    cross0 = samples * (-1j * cos_inclination)[:, None]
    cos_nodes = torch.cos(2.0 * numeric["long_asc_nodes"])[:, None]
    sin_nodes = torch.sin(2.0 * numeric["long_asc_nodes"])[:, None]
    plus = cos_nodes * plus0 + sin_nodes * cross0
    cross = cos_nodes * cross0 - sin_nodes * plus0

    support = (bin_numbers[None, :] >= first_bins[:, None]) & (
        bin_numbers[None, :] < end_bins[:, None]
    )
    plus = torch.where(support, plus, torch.zeros_like(plus))
    cross = torch.where(support, cross, torch.zeros_like(cross))
    if first_active:
        prefix = torch.zeros(
            batch_size,
            first_active,
            dtype=complex_dtype,
            device=device,
        )
        plus = torch.cat((prefix, plus), dim=1)
        cross = torch.cat((prefix, cross), dim=1)

    return TaylorF2FDBatch(
        hplus=plus,
        hcross=cross,
        delta_f=delta_f,
        epoch=epoch,
        first_bins=first_bins,
        end_bins=end_bins,
    )
