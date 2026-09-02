"""Torch-native implementation of LAL's single-spin ``SpinTaylorF2`` model.

The equations are a vectorized translation of
``LALSimInspiralSpinTaylorF2.c`` and the associated PN-coefficient routines.
Public PyCBC parameter conventions are applied here; unsupported options are
left to the waveform dispatcher to generate with lalsimulation.
"""

from dataclasses import dataclass
import math

from pycbc import lal_compat as lal
import torch

from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData


_EULER_GAMMA = 0.5772156649015329


# Non-spinning PN phasing coefficients used as the base of SpinTaylorF2.
def _nonspin_phasing_coeffs(eta):
    pfaN = 3.0 / (128.0 * eta)
    pfa2 = 5.0 * (743.0 / 84.0 + 11.0 * eta) / 9.0
    pfa3 = -16.0 * math.pi
    pfa4 = 5.0 * (3058.673 / 7.056 + 5429.0 / 7.0 * eta + 617.0 * eta * eta) / 72.0
    pfa5 = 5.0 / 9.0 * (7729.0 / 84.0 - 13.0 * eta) * math.pi
    pfl5 = 5.0 / 3.0 * (7729.0 / 84.0 - 13.0 * eta) * math.pi
    pfa6 = (
        11583.231236531 / 4.694215680
        - 640.0 / 3.0 * math.pi * math.pi
        - 6848.0 / 21.0 * _EULER_GAMMA
        + eta * (-15737.765635 / 3.048192 + 2255.0 / 12.0 * math.pi * math.pi)
        + eta * eta * 76055.0 / 1728.0
        - eta * eta * eta * 127825.0 / 1296.0
        + (-6848.0 / 21.0) * math.log(4.0)
    )
    pfl6 = -6848.0 / 21.0
    pfa7 = (
        math.pi
        * 5.0
        / 756.0
        * (15419335.0 / 336.0 + 75703.0 / 2.0 * eta - 14809.0 * eta * eta)
    )

    return dict(
        pfaN=pfaN,
        pfa2=pfa2,
        pfa3=pfa3,
        pfa4=pfa4,
        pfa5=pfa5,
        pfl5=pfl5,
        pfa6=pfa6,
        pfl6=pfl6,
        pfa7=pfa7,
    )


# ---- Helper routines mirroring the LAL SpinTaylorF2 implementation (single-spin) ----


def _safe_atan2(y, x):
    """Match LAL's safe_atan2 helper, including its zero/zero convention."""
    return torch.where(
        (y == 0.0) & (x == 0.0),
        torch.zeros_like(x),
        torch.atan2(y, x),
    )


def _orientation(m1, m2, v_ref, lnhatx, lnhaty, lnhatz, s1x, s1y, s1z):
    """Replicate XLALSimInspiralSF2CalculateOrientation."""
    chi = torch.sqrt(s1x * s1x + s1y * s1y + s1z * s1z)
    kappa = (
        (lnhatx * s1x + lnhaty * s1y + lnhatz * s1z) / chi
        if chi.item() != 0.0
        else torch.ones_like(chi)
    )
    kappa = torch.clamp(kappa, -1.0, 1.0)
    Jx0 = m1 * m2 * lnhatx / v_ref + m1 * m1 * s1x
    Jy0 = m1 * m2 * lnhaty / v_ref + m1 * m1 * s1y
    Jz0 = m1 * m2 * lnhatz / v_ref + m1 * m1 * s1z
    Jnorm = torch.sqrt(Jx0 * Jx0 + Jy0 * Jy0 + Jz0 * Jz0)
    thetaJ = torch.acos(torch.clamp(Jz0 / Jnorm, -1.0, 1.0))
    # LAL sign convention for polarization phase
    psiJ = _safe_atan2(Jy0, -Jx0)

    rotLx = (
        lnhatx * torch.cos(thetaJ) * torch.cos(psiJ)
        - lnhaty * torch.cos(thetaJ) * torch.sin(psiJ)
        + lnhatz * torch.sin(thetaJ)
    )
    rotLy = lnhatx * torch.sin(psiJ) + lnhaty * torch.cos(psiJ)
    alpha0 = _safe_atan2(rotLy, rotLx)

    return dict(chi=chi, kappa=kappa, thetaJ=thetaJ, psiJ=psiJ, alpha0=alpha0)


def _coeffs(m1, m2, chi, kappa):
    """Replicate XLALSimInspiralSF2CalculateCoeffs."""
    mtot = m1 + m2
    eta = m1 * m2 / (mtot * mtot)
    gamma0 = m1 * chi / m2
    kappa_perp = torch.sqrt(torch.clamp(1.0 - kappa * kappa, min=0.0))

    pn_beta = (113.0 * m1 / (12.0 * mtot) - 19.0 * eta / 6.0) * chi * kappa
    pn_sigma = (
        (5.0 * (3.0 * kappa * kappa - 1.0) / 2.0) + (7.0 - kappa * kappa) / 96.0
    ) * (m1 * m1 * chi * chi / (mtot * mtot))
    pn_gamma = (
        (
            5.0 * (146597.0 + 7056.0 * eta) * m1 / (2268.0 * mtot)
            - 10.0 * eta * (1276.0 + 153.0 * eta) / 81.0
        )
        * chi
        * kappa
    )

    prec_fac = 5.0 * (4.0 + 3.0 * m2 / m1) / 64.0
    dtdv2 = 743.0 / 336.0 + 11.0 * eta / 4.0
    dtdv3 = -4.0 * math.pi + pn_beta
    dtdv4 = (
        3058673.0 / 1016064.0
        + 5429.0 * eta / 1008.0
        + 617.0 * eta * eta / 144.0
        - pn_sigma
    )
    dtdv5 = (-7729.0 / 672.0 + 13.0 * eta / 8.0) * math.pi + 9.0 * pn_gamma / 40.0

    aclog1 = (
        kappa * (1.0 - kappa * kappa) * gamma0 * gamma0 * gamma0 / 2.0
        - dtdv2 * kappa * gamma0
        - dtdv3
    )
    aclog2 = (
        dtdv2 * gamma0
        + dtdv3 * kappa
        + (1.0 - kappa * kappa) * (dtdv4 - dtdv5 * kappa / gamma0) / (2.0 * gamma0)
    )
    ac0 = -1.0 / 3.0
    ac1 = -gamma0 * kappa / 6.0
    ac2 = gamma0 * gamma0 * (-1.0 / 3.0 + kappa * kappa / 2.0) - dtdv2
    ac3 = (
        dtdv3
        + dtdv4 * kappa / (2.0 * gamma0)
        + dtdv5 * (1.0 / 3.0 - kappa * kappa / 2.0) / (gamma0 * gamma0)
    )
    ac4 = dtdv4 / 2.0 + dtdv5 * kappa / (6.0 * gamma0)
    ac5 = dtdv5 / 3.0

    zc0 = -1.0 / 3.0
    zc1 = -gamma0 * kappa / 2.0
    zc2 = -dtdv2
    zc3 = dtdv2 * gamma0 * kappa + dtdv3
    zc4 = dtdv3 * gamma0 * kappa + dtdv4
    zc5 = (dtdv4 * gamma0 * kappa + dtdv5) / 2.0
    zc6 = dtdv5 * gamma0 * kappa / 3.0

    return dict(
        gamma0=gamma0,
        kappa=kappa,
        kappa_perp=kappa_perp,
        prec_fac=prec_fac,
        aclog1=aclog1,
        aclog2=aclog2,
        ac=(ac0, ac1, ac2, ac3, ac4, ac5),
        zc=(zc0, zc1, zc2, zc3, zc4, zc5, zc6),
    )


def _alpha(v, coeffs):
    """Evaluate XLALSimInspiralSF2Alpha with vectorized Torch operations."""
    gam = coeffs["gamma0"] * v
    kappa = coeffs["kappa"]
    sqrtfac = torch.sqrt(1.0 + 2.0 * kappa * gam + gam * gam)
    logfac1 = torch.log((1.0 + kappa * gam + sqrtfac) / v)
    logfac2 = torch.log(kappa + gam + sqrtfac)

    ac0, ac1, ac2, ac3, ac4, ac5 = coeffs["ac"]
    aclog1 = coeffs["aclog1"]
    aclog2 = coeffs["aclog2"]
    prec_fac = coeffs["prec_fac"]

    poly = (((ac0 / v + ac1) / v + ac2) / v + ac3 + (ac4 + ac5 * v) * v) * sqrtfac
    return prec_fac * (aclog1 * logfac1 + aclog2 * logfac2 + poly)


def _emission(v, coeffs):
    """Return the five XLALSimInspiralSF2Emission tensors."""
    gam = coeffs["gamma0"] * v
    kappa = coeffs["kappa"]
    kappa_perp = coeffs["kappa_perp"]
    sqrtfac = torch.sqrt(1.0 + 2.0 * kappa * gam + gam * gam)
    cosbeta = (1.0 + kappa * gam) / sqrtfac
    sinbeta = (kappa_perp * gam) / sqrtfac

    em0 = (1.0 + cosbeta) * (1.0 + cosbeta) / 4.0
    em1 = (1.0 + cosbeta) * sinbeta / 4.0
    em2 = sinbeta * sinbeta / 4.0
    em3 = (1.0 - cosbeta) * sinbeta / 4.0
    em4 = (1.0 - cosbeta) * (1.0 - cosbeta) / 4.0
    return em0, em1, em2, em3, em4


def _polarization(thetaJ, psiJ, mm):
    """Evaluate the complex XLALSimInspiralSF2Polarization factor."""
    ct = torch.cos(thetaJ)
    st = torch.sin(thetaJ)
    if mm == 2:
        plus_fac = (1.0 + ct * ct) / 2.0
        cross_fac = -1j * ct
    elif mm == 1:
        plus_fac = torch.sin(2.0 * thetaJ)
        cross_fac = -2j * st
    elif mm == 0:
        plus_fac = 3.0 * st * st
        cross_fac = 0.0j
    elif mm == -1:
        plus_fac = -torch.sin(2.0 * thetaJ)
        cross_fac = -2j * st
    elif mm == -2:
        plus_fac = (1.0 + ct * ct) / 2.0
        cross_fac = 1j * ct
    else:
        raise ValueError(f"Invalid SpinTaylorF2 sideband mode {mm}")
    return plus_fac * torch.cos(2.0 * psiJ) + cross_fac * torch.sin(2.0 * psiJ)


def _pfa_coeffs(
    m1,
    m2,
    eta,
    chi1L,
    chi1sq,
    coeffs,
    phase_order,
    enable_prec=True,
    pn_spin_order=-1,
    qm_def1=0.0,
    non_gr=None,
):
    """Compute the 3.5PN phasing series used by SpinTaylorF2.

    The coefficients follow the LAL PNPhasingSeries: base (non-spin) terms are
    scaled by pfaN = 3/(128*eta), single-spin contributions are added via the
    SpinTaylorF2 SO/SS coefficients, and the precession zeta-series pieces are
    added afterwards (unscaled by pfaN) when precession is enabled.
    """

    mtot = m1 + m2
    m1M = m1 / mtot
    chi1L2 = chi1L * chi1L
    qm_def1 = 1.0 + qm_def1

    # Non-spinning TaylorF2 phasing terms (use pn_beta/sigma/gamma = 0 to avoid
    # double-counting spin contributions that are added explicitly below).
    base = _nonspin_phasing_coeffs(eta)

    # Apply non-GR scaling to the non-spin pieces (matches PNPhasingSeries)
    if non_gr is None:
        non_gr = {}
    # pfa1 carries the non-GR 0.5PN phase term (PNPhasingSeries v[1]); LAL
    # sets it before the global pfaN scaling.
    base_dchi1 = non_gr.get("dchi1", 0.0)
    base["pfa2"] *= 1.0 + non_gr.get("dchi2", 0.0)
    base["pfa3"] *= 1.0 + non_gr.get("dchi3", 0.0)
    base["pfa4"] *= 1.0 + non_gr.get("dchi4", 0.0)
    base["pfa5"] *= 1.0 + non_gr.get("dchi5", 0.0)
    base["pfl5"] *= 1.0 + non_gr.get("dchi5l", 0.0)
    base["pfa6"] *= 1.0 + non_gr.get("dchi6", 0.0)
    base["pfl6"] *= 1.0 + non_gr.get("dchi6l", 0.0)
    base["pfa7"] *= 1.0 + non_gr.get("dchi7", 0.0)

    # Spin-orbit and spin-squared pieces (single-spin, chi2 = 0)
    def so3(mbym):
        return mbym * (25.0 + 38.0 / 3.0 * mbym)

    def so5(mbym):
        return -mbym * (
            1391.5 / 8.4
            - mbym * (1.0 - mbym) * 10.0 / 3.0
            + mbym * (1276.0 / 8.1 + mbym * (1.0 - mbym) * 170.0 / 9.0)
        )

    def so6(mbym):
        return math.pi * mbym * (1490.0 / 3.0 + mbym * 260.0)

    def so7(mbym):
        eta_loc = mbym * (1.0 - mbym)
        return mbym * (
            -17097.8035 / 4.8384
            + eta_loc * 28764.25 / 6.72
            + eta_loc * eta_loc * 47.35 / 1.44
            + mbym
            * (
                -7189.233785 / 1.524096
                + eta_loc * 458.555 / 3.024
                - eta_loc * eta_loc * 534.5 / 7.2
            )
        )

    def qm2so4(mbym):
        return -720.0 / 9.6 * mbym * mbym

    def self2so4(mbym):
        return 1.0 / 9.6 * mbym * mbym

    def qm2s4(mbym):
        return 240.0 / 9.6 * mbym * mbym

    def self2s4(mbym):
        return -7.0 / 9.6 * mbym * mbym

    def qm2s6(mbym):
        return (4703.5 / 8.4 + 2935.0 / 6.0 * mbym - 120.0 * mbym * mbym) * mbym * mbym

    def self2s6(mbym):
        return (
            (-4108.25 / 6.72 - 108.5 / 1.2 * mbym + 125.5 / 3.6 * mbym * mbym)
            * mbym
            * mbym
        )

    pfaN_base = base["pfaN"]
    pfa1_base = base_dchi1
    pfa2_base = base["pfa2"]
    include_so7 = pn_spin_order in (-1, 7)
    include_so6 = pn_spin_order in (-1, 7, 6)
    include_so5 = pn_spin_order in (-1, 7, 6, 5)
    include_ss2pn = pn_spin_order in (-1, 7, 6, 5, 4)
    include_so3 = pn_spin_order in (-1, 7, 6, 5, 4, 3)

    so3_term = so3(m1M) * chi1L if include_so3 else 0.0
    so5_term = so5(m1M) * chi1L if include_so5 else 0.0
    so6_term = so6(m1M) * chi1L if include_so6 else 0.0
    so7_term = so7(m1M) * chi1L if include_so7 else 0.0
    ss2pn_term = (
        (
            (qm2so4(m1M) * qm_def1 + self2so4(m1M)) * chi1L2
            + (qm2s4(m1M) * qm_def1 + self2s4(m1M)) * chi1sq
        )
        if include_ss2pn
        else 0.0
    )
    ss3pn_term = (
        ((qm2s6(m1M) * qm_def1 + self2s6(m1M)) * chi1sq) if include_so6 else 0.0
    )

    pfa3_base = base["pfa3"] + so3_term
    pfa4_base = base["pfa4"] + ss2pn_term
    pfa5_base = base["pfa5"] + so5_term
    pfl5_base = base["pfl5"] + (3.0 * so5_term if include_so5 else 0.0)
    pfa6_base = base["pfa6"] + so6_term + ss3pn_term
    pfl6_base = base["pfl6"]
    pfa7_base = base["pfa7"] + so7_term
    # Scale by pfaN (as in PNPhasingSeries)
    pfaN_full = pfaN_base * (1.0 + non_gr.get("dchi0", 0.0))
    pfa1_full = pfaN_base * pfa1_base
    pfa2_full = pfaN_base * pfa2_base
    pfa3_full = pfaN_base * pfa3_base
    pfa4_full = pfaN_base * pfa4_base
    pfa5_full = pfaN_base * pfa5_base
    pfl5_full = pfaN_base * pfl5_base
    pfa6_full = pfaN_base * pfa6_base
    pfl6_full = pfaN_base * pfl6_base
    pfa7_full = pfaN_base * pfa7_base
    # Apply phase_order selection (mirrors LAL fall-through)
    if phase_order not in (-1, 7, 6, 5, 4, 3, 2, 1, 0):
        raise ValueError(f"Invalid phase_order {phase_order}")

    pfa_sel = dict(
        pfaN=0.0,
        pfa1=0.0,
        pfa2=0.0,
        pfa3=0.0,
        pfa4=0.0,
        pfa5=0.0,
        pfl5=0.0,
        pfa6=0.0,
        pfl6=0.0,
        pfa7=0.0,
        pfa8=0.0,
    )

    if phase_order in (-1, 7):
        pfa_sel["pfa7"] = pfa7_full
    if phase_order in (-1, 7, 6):
        pfa_sel["pfa6"] = pfa6_full
        pfa_sel["pfl6"] = pfl6_full
    if phase_order in (-1, 7, 6, 5):
        pfa_sel["pfa5"] = pfa5_full
        pfa_sel["pfl5"] = pfl5_full
    if phase_order in (-1, 7, 6, 5, 4):
        pfa_sel["pfa4"] = pfa4_full
    if phase_order in (-1, 7, 6, 5, 4, 3):
        pfa_sel["pfa3"] = pfa3_full
    if phase_order in (-1, 7, 6, 5, 4, 3, 2):
        pfa_sel["pfa2"] = pfa2_full
    if phase_order in (-1, 7, 6, 5, 4, 3, 2, 1):
        pfa_sel["pfa1"] = pfa1_full
    if phase_order in (-1, 7, 6, 5, 4, 3, 2, 1, 0):
        pfa_sel["pfaN"] = pfaN_full

    # Add zeta PN pieces after selection (unscaled by pfaN)
    if enable_prec:
        zc = coeffs["zc"]
        prec_fac = coeffs["prec_fac"]
        pfa_sel["pfa2"] += 2.0 * prec_fac * zc[0]
        pfa_sel["pfa3"] += 2.0 * prec_fac * zc[1]
        pfa_sel["pfa4"] += 2.0 * prec_fac * zc[2]
        pfa_sel["pfl5"] += 2.0 * prec_fac * zc[3]
        pfa_sel["pfa6"] += 2.0 * prec_fac * zc[4]
        pfa_sel["pfa7"] += 2.0 * prec_fac * zc[5]
        pfa_sel["pfa8"] += 2.0 * prec_fac * zc[6]

    return pfa_sel


_PHASE_ORDERS = frozenset((-1, 0, 1, 2, 3, 4, 5, 6, 7))
_SPIN_ORDERS = frozenset((-1, 0, 1, 2, 3, 4, 5, 6, 7))
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
    """Normalize an integer-valued LAL order flag without raising."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _is_nonzero(value):
    """Return whether an optional scalar is set to a non-zero value."""
    if value is None:
        return False
    try:
        return float(value) != 0.0
    except (TypeError, ValueError, OverflowError):
        return True


def spintaylorf2_native_supported(params):
    """Return whether params are covered by the native Torch generator.

    SpinTaylorF2 itself is a single-spin model. The native path implements its
    public LAL parameter contract, including spin-frame and polarization
    rotations, PN phase/spin orders, testing-GR dchi terms, the primary's
    quadrupole deformation, and sideband selection. Options outside that
    contract deliberately remain on the lalsimulation path.
    """
    if _as_order(params.get("phase_order", -1)) not in _PHASE_ORDERS:
        return False
    if _as_order(params.get("spin_order", -1)) not in _SPIN_ORDERS:
        return False
    if _as_order(params.get("amplitude_order", -1)) is None:
        return False
    if _as_order(params.get("tidal_order", -1)) != -1:
        return False
    if _as_order(params.get("eccentricity_order", -1)) != -1:
        return False
    if _as_order(params.get("side_bands", 0)) is None:
        return False

    if any(
        _is_nonzero(params.get(key, 0.0))
        for key in (
            "spin2x",
            "spin2y",
            "spin2z",
            "eccentricity",
            "mean_per_ano",
            "frame_axis",
            "modes_choice",
            "lambda1",
            "lambda2",
            "dquad_mon2",
        )
    ):
        return False
    if params.get("mode_array") is not None or params.get("numrel_data", ""):
        return False
    if any(params.get(key) is not None for key in _UNSUPPORTED_TIDAL_KEYS):
        return False
    if any(_is_nonzero(params.get(key, 0.0)) for key in _UNSUPPORTED_NON_GR_KEYS):
        return False
    return True


def _evaluate_phasing(velocity, pfa):
    """Evaluate the SpinTaylorF2 PN phasing series."""
    velocity2 = velocity * velocity
    velocity3 = velocity2 * velocity
    velocity4 = velocity2 * velocity2
    velocity5 = velocity3 * velocity2
    log_velocity = torch.log(velocity)
    numerator = (
        pfa["pfaN"]
        + pfa["pfa1"] * velocity
        + pfa["pfa2"] * velocity2
        + pfa["pfa3"] * velocity3
        + pfa["pfa4"] * velocity4
    )
    return (
        numerator / velocity5
        + pfa["pfa5"]
        + pfa["pfl5"] * log_velocity
        + (pfa["pfa6"] + pfa["pfl6"] * log_velocity) * velocity
        + pfa["pfa7"] * velocity2
        + pfa["pfa8"] * velocity3
    )


@dataclass(frozen=True)
class _SpinTaylorF2Inputs:
    """Validated scalar and coefficient state shared by both public APIs."""

    mass1: float
    mass2: float
    distance: float
    coa_phase: float
    long_asc_nodes: float
    eta: float
    pi_mass: float
    sideband: int
    device: torch.device
    real_dtype: torch.dtype
    complex_dtype: torch.dtype
    orientation: dict
    coeffs: dict
    enable_precession: bool
    alpha_reference: torch.Tensor
    pfa: dict
    reference_phasing: torch.Tensor


def spintaylorf2_sequence_native_supported(params):
    """Return whether arbitrary-frequency SpinTaylorF2 is native."""

    return spintaylorf2_native_supported(params)


def spintaylorf2_default_native_supported(_params):
    """Return whether unflagged native use is accurate on this device.

    Apple MPS evaluates the scalar precession and inspiral phase setup in
    float32.  Cancellation in that setup can produce a waveform-wide phase
    offset, so MPS retains the LAL fallback unless native execution is
    requested explicitly.
    """
    from pycbc import scheme as _scheme

    state = _scheme.mgr.state
    return not (
        isinstance(state, _scheme.TorchScheme)
        and state.torch_device.type == "mps"
    )


def _spintaylorf2_inputs(
    params,
    *,
    default_reference_frequency,
    sequence=False,
):
    """Validate scalars and construct model state shared by both samplers."""
    from pycbc import scheme as _scheme

    if not spintaylorf2_native_supported(params):
        raise ValueError(
            "SpinTaylorF2 parameters are not supported by the native Torch path"
        )

    mass1 = float(params["mass1"])
    mass2 = float(params["mass2"])
    spin1x = float(params.get("spin1x", 0.0))
    spin1y = float(params.get("spin1y", 0.0))
    spin1z = float(params.get("spin1z", 0.0))
    distance = float(params.get("distance", 1.0))
    inclination = float(params.get("inclination", 0.0))
    coa_phase = float(params.get("coa_phase", 0.0))
    # SimInspiralChooseFDWaveformSequence has no ascending-node argument and
    # ignores the corresponding PyCBC parameter.
    long_asc_nodes = (
        0.0 if sequence else float(params.get("long_asc_nodes", 0.0))
    )
    f_ref = float(params.get("f_ref", 0.0))
    phase_order = _as_order(params.get("phase_order", -1))
    spin_order = _as_order(params.get("spin_order", -1))
    sideband = _as_order(params.get("side_bands", 0))
    dquad_mon1 = float(params.get("dquad_mon1") or 0.0)
    non_gr = {key: float(params.get(key) or 0.0) for key in _DCHI_KEYS}

    if not math.isfinite(mass1) or not math.isfinite(mass2):
        raise ValueError("SpinTaylorF2 component masses must be finite")
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("SpinTaylorF2 component masses must be positive")
    if not all(
        math.isfinite(value)
        for value in (
            spin1x,
            spin1y,
            spin1z,
            inclination,
            coa_phase,
            long_asc_nodes,
            dquad_mon1,
            *non_gr.values(),
        )
    ):
        raise ValueError("SpinTaylorF2 spins, angles, and PN terms must be finite")
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("SpinTaylorF2 distance must be finite and positive")
    if not math.isfinite(f_ref) or f_ref < 0.0:
        raise ValueError("SpinTaylorF2 f_ref must be finite and non-negative")
    default_reference_frequency = float(default_reference_frequency)
    if (
        not math.isfinite(default_reference_frequency)
        or default_reference_frequency <= 0.0
    ):
        raise ValueError(
            "SpinTaylorF2 default reference frequency must be finite and positive"
        )

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / total_mass**2
    pi_mass = math.pi * total_mass * lal.MTSUN_SI

    state = _scheme.mgr.state
    device = getattr(state, "torch_device", torch.device("cpu"))
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128

    pi_mass_tensor = torch.as_tensor(pi_mass, dtype=real_dtype, device=device)

    # The public LAL wrapper rotates the source-frame spin around +y before
    # invoking the core generator, then supplies LNhat from the inclination.
    sin_inclination = math.sin(inclination)
    cos_inclination = math.cos(inclination)
    rotated_spin1x = spin1x * cos_inclination + spin1z * sin_inclination
    rotated_spin1y = spin1y
    rotated_spin1z = -spin1x * sin_inclination + spin1z * cos_inclination
    spin1x_tensor = torch.as_tensor(rotated_spin1x, dtype=real_dtype, device=device)
    spin1y_tensor = torch.as_tensor(rotated_spin1y, dtype=real_dtype, device=device)
    spin1z_tensor = torch.as_tensor(rotated_spin1z, dtype=real_dtype, device=device)
    lnhatx = torch.as_tensor(sin_inclination, dtype=real_dtype, device=device)
    lnhaty = torch.zeros((), dtype=real_dtype, device=device)
    lnhatz = torch.as_tensor(cos_inclination, dtype=real_dtype, device=device)

    # Regular generation maps f_ref=0 to f_lower. Sequence generation follows
    # the other LAL sequence interfaces and uses the first supplied frequency.
    effective_f_ref = (
        f_ref if f_ref > 0.0 else default_reference_frequency
    )
    reference_velocity = torch.pow(
        pi_mass_tensor
        * torch.as_tensor(effective_f_ref, dtype=real_dtype, device=device),
        1.0 / 3.0,
    )
    orientation = _orientation(
        mass1,
        mass2,
        reference_velocity,
        lnhatx,
        lnhaty,
        lnhatz,
        spin1x_tensor,
        spin1y_tensor,
        spin1z_tensor,
    )
    coeffs = _coeffs(
        mass1,
        mass2,
        orientation["chi"],
        orientation["kappa"],
    )
    enable_precession = (
        orientation["chi"].item() != 0.0 and abs(orientation["kappa"].item()) < 1.0
    )
    alpha_reference = (
        _alpha(reference_velocity, coeffs) - orientation["alpha0"]
        if enable_precession
        else torch.zeros((), dtype=real_dtype, device=device)
    )

    pfa = _pfa_coeffs(
        mass1,
        mass2,
        eta,
        orientation["chi"] * orientation["kappa"],
        orientation["chi"] ** 2,
        coeffs,
        phase_order,
        enable_prec=enable_precession,
        pn_spin_order=spin_order,
        qm_def1=dquad_mon1,
        non_gr=non_gr,
    )
    reference_phasing = _evaluate_phasing(reference_velocity, pfa)
    return _SpinTaylorF2Inputs(
        mass1=mass1,
        mass2=mass2,
        distance=distance,
        coa_phase=coa_phase,
        long_asc_nodes=long_asc_nodes,
        eta=eta,
        pi_mass=pi_mass,
        sideband=sideband,
        device=device,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        orientation=orientation,
        coeffs=coeffs,
        enable_precession=enable_precession,
        alpha_reference=alpha_reference,
        pfa=pfa,
        reference_phasing=reference_phasing,
    )


def _spintaylorf2_samples(inputs, frequencies, *, time_shift=0.0):
    """Evaluate SpinTaylorF2 polarizations at device-resident frequencies."""
    velocity = torch.pow(inputs.pi_mass * frequencies, 1.0 / 3.0)
    velocity2 = velocity * velocity
    velocity3 = velocity2 * velocity
    phasing = (
        _evaluate_phasing(velocity, inputs.pfa)
        + 2.0 * math.pi * time_shift * frequencies
        - 2.0 * inputs.coa_phase
        - inputs.reference_phasing
        - math.pi / 4.0
    )
    alpha = (
        _alpha(velocity, inputs.coeffs) - inputs.alpha_reference
        if inputs.enable_precession
        else torch.zeros_like(velocity)
    )
    precession_phase = torch.complex(torch.cos(alpha), torch.sin(alpha))
    inverse_precession_phase = 1.0 / precession_phase
    em0, em1, em2, em3, em4 = _emission(velocity, inputs.coeffs)

    sideband_plus = torch.zeros(
        5,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    sideband_cross = torch.zeros_like(sideband_plus)
    # PyCBC only inserts a non-zero side_bands value into the LAL dictionary.
    # LAL's default selects m=0; any inserted value activates all five terms.
    sideband_modes = (
        (0,) if inputs.sideband == 0 else (-2, -1, 0, 1, 2)
    )
    quarter_turn = torch.as_tensor(
        math.pi / 4.0,
        dtype=inputs.real_dtype,
        device=inputs.device,
    )
    for mode in sideband_modes:
        index = 2 - mode
        sideband_plus[index] = _polarization(
            inputs.orientation["thetaJ"],
            inputs.orientation["psiJ"],
            mode,
        )
        sideband_cross[index] = _polarization(
            inputs.orientation["thetaJ"],
            inputs.orientation["psiJ"] + quarter_turn,
            mode,
        )

    precession_plus = (
        sideband_plus[0] * em0 * precession_phase**2
        + sideband_plus[1] * em1 * precession_phase
        + sideband_plus[2] * em2
        + sideband_plus[3] * em3 * inverse_precession_phase
        + sideband_plus[4] * em4 * inverse_precession_phase**2
    )
    precession_cross = (
        sideband_cross[0] * em0 * precession_phase**2
        + sideband_cross[1] * em1 * precession_phase
        + sideband_cross[2] * em2
        + sideband_cross[3] * em3 * inverse_precession_phase
        + sideband_cross[4] * em4 * inverse_precession_phase**2
    )

    distance_metres = inputs.distance * 1.0e6 * lal.PC_SI
    amplitude0 = (
        -4.0
        * inputs.mass1
        * inputs.mass2
        / distance_metres
        * lal.MRSUN_SI
        * lal.MTSUN_SI
        * math.sqrt(math.pi / 12.0)
        * math.sqrt(5.0 / (32.0 * inputs.eta))
    )
    amplitude = amplitude0 / (velocity3 * torch.sqrt(velocity))
    carrier = torch.exp(-1j * phasing)
    plus = precession_plus * carrier * amplitude
    cross = precession_cross * carrier * amplitude

    # The legacy public wrapper applies the longitude-of-ascending-nodes
    # polarization rotation after waveform generation.
    cos_nodes = math.cos(2.0 * inputs.long_asc_nodes)
    sin_nodes = math.sin(2.0 * inputs.long_asc_nodes)
    rotated_plus = cos_nodes * plus + sin_nodes * cross
    rotated_cross = cos_nodes * cross - sin_nodes * plus
    return rotated_plus, rotated_cross


def spintaylorf2_torch(**params):
    """Generate regular-grid SpinTaylorF2 on the active Torch device."""
    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))
    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("SpinTaylorF2 frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("SpinTaylorF2 delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("SpinTaylorF2 f_final must be non-negative")

    inputs = _spintaylorf2_inputs(
        params,
        default_reference_frequency=f_lower,
    )
    f_isco = 1.0 / (6.0**1.5 * inputs.pi_mass)
    f_max = f_final if f_final > 0.0 else f_isco
    if f_max <= f_lower:
        raise ValueError("SpinTaylorF2 ending frequency must exceed f_lower")

    length = int(f_max / delta_f + 1.0)
    first_bin = int(math.ceil(f_lower / delta_f))
    if first_bin >= length:
        raise ValueError("SpinTaylorF2 frequency range contains no sampled bins")
    frequencies = (
        torch.arange(
            first_bin,
            length,
            dtype=inputs.real_dtype,
            device=inputs.device,
        )
        * delta_f
    )
    epoch = -1.0 / delta_f
    rotated_plus, rotated_cross = _spintaylorf2_samples(
        inputs,
        frequencies,
        time_shift=epoch,
    )

    plus_full = torch.zeros(
        length,
        dtype=inputs.complex_dtype,
        device=inputs.device,
    )
    cross_full = torch.zeros_like(plus_full)
    plus_full[first_bin:] = rotated_plus
    cross_full[first_bin:] = rotated_cross
    return (
        FrequencySeries(
            TorchArrayData(plus_full),
            delta_f=delta_f,
            epoch=epoch,
            copy=False,
        ),
        FrequencySeries(
            TorchArrayData(cross_full),
            delta_f=delta_f,
            epoch=epoch,
            copy=False,
        ),
    )


def _sequence_frequencies(sample_points):
    """Validate and move arbitrary frequencies to the active Torch device."""
    from pycbc import scheme as _scheme

    state = _scheme.mgr.state
    device = getattr(state, "torch_device", torch.device("cpu"))
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    values = getattr(sample_points, "_data", sample_points)
    if isinstance(values, TorchArrayData):
        values = values.tensor
    frequencies = torch.as_tensor(
        values,
        dtype=real_dtype,
        device=device,
    )
    if frequencies.ndim != 1 or frequencies.numel() == 0:
        raise ValueError(
            "SpinTaylorF2 sample_points must be a non-empty vector"
        )
    if not bool(torch.all(torch.isfinite(frequencies))):
        raise ValueError("SpinTaylorF2 sample_points must be finite")
    if bool(torch.any(frequencies <= 0.0)):
        raise ValueError("SpinTaylorF2 sample_points must be positive")
    return frequencies


def spintaylorf2_fd_sequence_torch(**params):
    """Evaluate SpinTaylorF2 at arbitrary frequencies with Torch.

    LAL does not expose SpinTaylorF2 through its sequence API. This native
    extension follows the common sequence conventions: no epoch time shift,
    ``long_asc_nodes`` is ignored, and ``f_ref=0`` uses the first sample.
    """
    from pycbc.types import Array as PyCBCArray

    if not spintaylorf2_sequence_native_supported(params):
        raise ValueError(
            "SpinTaylorF2 sequence parameters are not supported by the "
            "native Torch path"
        )
    frequencies = _sequence_frequencies(params["sample_points"])
    inputs = _spintaylorf2_inputs(
        params,
        default_reference_frequency=float(frequencies[0].item()),
        sequence=True,
    )
    plus, cross = _spintaylorf2_samples(inputs, frequencies)
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


__all__ = [
    "spintaylorf2_default_native_supported",
    "spintaylorf2_fd_sequence_torch",
    "spintaylorf2_native_supported",
    "spintaylorf2_sequence_native_supported",
    "spintaylorf2_torch",
]
