# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch-native IMRPhenomXHM (3, -2) mode with ringdown mixing.

This ports the default LALSuite 7.26 mixed-mode path. Parameter-space fits are
evaluated once per waveform; matching systems, frequency-dependent evaluation,
the spheroidal-to-spherical rotation, and waveform assembly use Torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import lal
import torch

from . import imrphenomx_utils_torch as _xutils
from ._torch_jax import torch_context
from .imrphenomxas_torch import (
    Phase,
    PhaseDerivative,
    get_inspiral_phase,
    get_mergerringdown_Amp,
)
from .imrphenomxhm_mode21_torch import (
    _as_float,
    _mode21_state,
    _solve,
    _tensor,
    _value_and_derivative,
)
from .imrphenomxhm_mode33_torch import _inspiral_boundary


_PI = lal.PI
_FALSE_ZERO = 1.0e-15
_PN_EXPONENTS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 4.0 / 3.0, 5.0 / 3.0, 2.0)


@dataclass(frozen=True)
class _Mode32State:
    base: object
    f_ring_32: float
    f_damp_32: float
    mixing_322: complex
    mixing_323: complex

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def chi1z(self):
        return self.chi1

    @property
    def chi2z(self):
        return self.chi2

    @property
    def final_mass(self):
        return 1.0 - self.radiated_energy

    @property
    def f_meco_32(self):
        return self.f_meco_22


@dataclass(frozen=True)
class _Transitions:
    f_amp_in: float
    f_amp_rd: float
    f_phase_in: float
    f_phase_rd: float
    phase_intermediate_points: tuple[float, ...]
    phase_ringdown_points: tuple[float, ...]


@dataclass
class _Phase32:
    transitions: _Transitions
    linb: torch.Tensor
    phiref22: torch.Tensor
    alpha0_s: torch.Tensor
    alpha_l_s: torch.Tensor
    alpha2_s: torch.Tensor
    alpha4_s: torch.Tensor
    phi0_s: torch.Tensor
    c0: torch.Tensor | None = None
    c_l: torch.Tensor | None = None
    c1: torch.Tensor | None = None
    c2: torch.Tensor | None = None
    c3: torch.Tensor | None = None
    c4: torch.Tensor | None = None
    c1_insp: torch.Tensor | None = None
    c_insp: torch.Tensor | None = None
    c1_rd: torch.Tensor | None = None
    c_rd: torch.Tensor | None = None
    delta_phi: torch.Tensor | None = None


@dataclass
class _Amplitude32:
    amp_norm: float
    pn_global_factor: float
    pn_coefficients: torch.Tensor
    pseudo_coefficients: torch.Tensor
    rd_alambda: float
    rd_lambda: float
    rd_sigma: float
    f_rd_aux: float
    f_falloff: float
    tail_amplitude: torch.Tensor
    tail_decay: torch.Tensor
    rd_aux_coefficients: torch.Tensor
    intermediate_coefficients: torch.Tensor | None = None


def _chi_pn_hat(state):
    return state.chi_pn_hat


def _xhm32_delta_t(state):
    _, _, psi4_to_strain = _xutils.calc_phaseatpeak(
        state.eta, state.s_tot_r, state.dchi, state.delta
    )
    return -2.0 * _PI * (500.0 + _as_float(psi4_to_strain))


def _check_final_spin(final_spin):
    spin = float(final_spin)
    if abs(spin) > 1.0:
        raise ValueError('XHM QNM fits require |final_spin| <= 1.')
    return spin

def qnm_fring32_fit(final_spin):
    a = _check_final_spin(final_spin)
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    return float((0.09540436245212061 - 0.13628306966373951 * a + 0.030099881830507727 * x2 - 0.000673589757007597 * x3 + 0.0118277880067919 * x4 + 0.0020533816327907334 * x5 - 0.0015206141948469621 * x6) / (1.0 - 1.6531854335715193 * a + 0.5634705514193629 * x2 + 0.12256204148002939 * x4 - 0.027297817699401976 * x6))

def qnm_fdamp32_fit(final_spin):
    a = _check_final_spin(final_spin)
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    return float((0.014754148319335946 - 0.03445752346074498 * a + 0.02168855041940869 * x2 + 0.0014945908223317514 * x3 - 0.0034761714223258693 * x4) / (1.0 - 2.320722660848874 * a + 1.5096146036915865 * x2 - 0.18791187563554512 * x4))

def evaluate_qnmfit_re_l3m2lp2(final_spin):
    a = _check_final_spin(final_spin)
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    return float(a * (0.47513455283841244 - 0.9016636384605536 * a + 0.3844811236426182 * x2 + 0.0855565148647794 * x3 - 0.03620067426672167 * x4 - 0.006557249133752502 * x5) / (-6.76894063440646 + 15.170831931186493 * a - 9.406169787571082 * x2 + x4))

def evaluate_qnmfit_im_l3m2lp2(final_spin):
    a = _check_final_spin(final_spin)
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    return float(a * (-2.8704762147145533 + 4.436434016918535 * a - 1.0115343326360486 * x2 - 0.08965314412106505 * x3 - 0.4236810894599512 * x4 - 0.041787576033810676 * x5) / (-171.80908957903395 + 272.362882450877 * a - 76.68544453077854 * x2 - 25.14197656531123 * x4 + x6))

def evaluate_qnmfit_re_l3m2lp3(final_spin):
    a = _check_final_spin(final_spin)
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    return float((1.0 - 2.107852425643677 * a + 1.1906393634562715 * x2 + 0.02244848864087732 * x3 - 0.09593447799423722 * x4 - 0.0021343381708933025 * x5 - 0.005319515989331159 * x6) / (1.0 - 2.1078515887706324 * a + 1.2043484690080966 * x2 - 0.08910191596778137 * x4 - 0.005471749827809503 * x6))

def evaluate_qnmfit_im_l3m2lp3(final_spin):
    a = _check_final_spin(final_spin)
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    return float(a * (12.45701482868677 - 29.398484595717147 * a + 18.26221675782779 * x2 + 1.9308599142669403 * x3 - 3.159763242921214 * x4 - 0.0910871567367674 * x5) / (345.52914639836257 - 815.4349339779621 * a + 538.3888932415709 * x2 - 69.3840921447381 * x4 + x6))

def xhm32_inspiral_phase_lambda_fit(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    s = state.s_tot_r
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    no_spin = (9.913819875501506 + 18.424900617803107 * eta - 574.8672384388947 * eta2 + 2671.7813055097877 * eta3 - 6244.001932443913 * eta4) / (1.0 - 0.9103118343073325 * eta)
    eq_spin = (-4.367632806613781 + 245.06757304950986 * eta - 2233.9319708029775 * eta2 + 5894.355429022858 * eta3) * s + (-1.375112297530783 - 1876.760129419146 * eta + 17608.172965575013 * eta2 - 40928.07304790013 * eta3) * s2 + (-1.28324755577382 - 138.36970336658558 * eta + 708.1455154504333 * eta2 - 273.23750933544176 * eta3) * s3 + (1.8403161863444328 + 2009.7361967331492 * eta - 18636.271414571278 * eta2 + 42379.205045791656 * eta3) * s4
    uneq_spin = state.dchi * state.delta * eta2 * (-105.34550407768225 - 1566.1242344157668 * state.chi1z * eta + 1566.1242344157668 * state.chi2z * eta + 2155.472229664981 * eta * s)
    return float(no_spin + eq_spin + uneq_spin)

def _xhm32_intermediate_phase_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    eta7 = eta6 * eta
    s = state.s_tot_r
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    dchi = state.dchi
    chi1 = state.chi1z
    chi2 = state.chi2z
    root = math.sqrt(max(1.0 - 4.0 * eta, 0.0))
    p1 = 4414.11 + 4.21564 / eta - 10687.8 * eta + 58234.6 * eta2 - 64068.4 * eta3 - 704442.0 * eta4 + 2863930.0 * eta5 - 3263620.0 * eta6 + ((6.39833 - 610.267 * eta + 2095.72 * eta2 - 3970.89 * eta3) * s + (22.956700000000005 - 99.1551 * eta + 331.593 * eta2 - 794.79 * eta3) * s2 + (10.4333 + 43.8812 * eta - 541.261 * eta2 + 294.289 * eta3) * s3 + eta * (106.047 - 1569.0299999999997 * eta + 4810.61 * eta2) * s4) / eta + 132.244 * root * eta * (chi1 * (6.227738120444028 - eta) + chi2 * (-6.227738120444028 + eta))
    p2 = 3980.7 + 0.956703 / eta - 6202.38 * eta + 29218.1 * eta2 + 24484.2 * eta3 - 807629.0 * eta4 + 2863930.0 * eta5 - 3263620.0 * eta6 + ((1.92692 - 226.825 * eta + 75.246 * eta2 + 1291.56 * eta3) * s + (15.328700000000001 - 99.1551 * eta + 608.328 * eta2 - 2402.94 * eta3) * s2 + (10.4333 + 43.8812 * eta - 541.261 * eta2 + 294.289 * eta3) * s3 + eta * (106.047 - 1569.0299999999997 * eta + 4810.61 * eta2) * s4) / eta + 132.244 * root * eta * (chi1 * (2.5769789177580837 - eta) + chi2 * (-2.5769789177580837 + eta))
    p3 = 3416.57 + 2308.63 * eta - 84042.9 * eta2 + 1019360.0 * eta3 - 6064400.0 * eta4 + 17639900.0 * eta5 - 20065000.0 * eta6 + (24.6295 - 282.354 * eta - 2582.55 * eta2 + 12750.0 * eta3) * s + (433.675 - 8775.86 * eta + 56407.8 * eta2 - 114798.0 * eta3) * s2 + (559.705 - 10627.4 * eta + 61581.0 * eta2 - 114029.0 * eta3) * s3 + (106.047 - 1569.03 * eta + 4810.61 * eta2) * s4 + 63.9466 * dchi * root * eta2
    p4 = 3307.49 - 476.909 * eta - 5980.37 * eta2 + 127610.0 * eta3 - 919108.0 * eta4 + 2863930.0 * eta5 - 3263620.0 * eta6 + (-5.02553 - 282.354 * eta + 1291.56 * eta2) * s + (-43.8823 + 740.123 * eta - 2402.94 * eta2) * s2 + (43.8812 - 370.362 * eta + 294.289 * eta2) * s3 + (106.047 - 1569.03 * eta + 4810.61 * eta2) * s4 - 132.244 * dchi * root * eta2
    p56 = 3259.03 - 3967.58 * eta + 111203.0 * eta2 - 1818830.0 * eta3 + 17381100.0 * eta4 - 95698800.0 * eta5 + 275056000.0 * eta6 - 315866000.0 * eta7 + (19.7509 - 1104.53 * eta + 3810.18 * eta2) * s + (-230.07 + 2314.51 * eta - 5944.49 * eta2) * s2 + (-201.633 + 2183.43 * eta - 6233.99 * eta2) * s3 + (106.047 - 1569.03 * eta + 4810.61 * eta2) * s4 + 112.714 * dchi * root * eta2
    delta_t = _xhm32_delta_t(state)
    return tuple((float(value + delta_t) for value in (p1, p2, p3, p4, p56, p56)))

def _xhm32_inspiral_amp_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    root = math.sqrt(eta)
    delta = state.delta
    cd = state.chi_a
    cd2 = cd * cd
    s = _chi_pn_hat(state)
    iv1 = (cd * delta * (-0.739317114582042 * eta - 47.473246070362634 * eta2 + 278.9717709112207 * eta3 - 566.6420939162068 * eta4) + cd2 * (-0.5873680378268906 * eta + 6.692187014925888 * eta2 - 24.37776782232888 * eta3 + 23.783684827838247 * eta4)) * root + (3.2940434453819694 + 4.94285331708559 * eta - 343.3143244815765 * eta2 + 3585.9269057886418 * eta3 - 19279.186145681153 * eta4 + 51904.91007211022 * eta5 - 55436.68857586653 * eta6) * root + cd * delta * (12.488240781993923 * eta - 209.32038774208385 * eta2 + 1160.9833883184604 * eta3 - 2069.5349737049073 * eta4) * s * root + s * (0.6343034651912586 * (-2.5844888818001737 + 78.98200041834092 * eta - 1087.6241783616488 * eta2 + 7616.234910399297 * eta3 - 24776.529123239357 * eta4 + 30602.210950069973 * eta5) - 0.062088720220899465 * (6.5586380356588565 + 36.01386705325694 * eta - 3124.4712274775407 * eta2 + 33822.437731298516 * eta3 - 138572.93700180828 * eta4 + 198366.10615196894 * eta5) * s) * root
    iv2 = (cd2 * (-0.03940151060321499 * eta + 1.9034209537174116 * eta2 - 8.78587250202154 * eta3) + cd * delta * (-1.704299788495861 * eta - 4.923510922214181 * eta2 + 0.36790005839460627 * eta3)) * root + (2.2911849711339123 - 5.1846950040514335 * eta + 60.10368251688146 * eta2 - 1139.110227749627 * eta3 + 7970.929280907627 * eta4 - 25472.73682092519 * eta5 + 30950.67053883646 * eta6) * root + s * (0.7718201508695763 * (-1.3012906461000349 + 26.432880113146012 * eta - 186.5001124789369 * eta2 + 712.9101229418721 * eta3 - 970.2126139442341 * eta4) + 0.04832734931068797 * (-5.9999628512498315 + 78.98681284391004 * eta + 1.8360177574514709 * eta2 - 2537.636347529708 * eta3 + 6858.003573909322 * eta4) * s) * root
    iv3 = (cd2 * (-0.6358511175987503 * eta + 5.555088747533164 * eta2 - 14.078156877577733 * eta3) + cd * delta * (0.23205448591711159 * eta - 19.46049432345157 * eta2 + 36.20685853857613 * eta3)) * root + (1.1525594672495008 + 7.380126197972549 * eta - 17.51265776660515 * eta2 - 976.9940395257111 * eta3 + 8880.536804741967 * eta4 - 30849.228936891763 * eta5 + 38785.53683146884 * eta6) * root + cd * delta * (1.904350804857431 * eta - 25.565242391371093 * eta2 + 80.67120303906654 * eta3) * s * root + s * (0.785171689871352 * (-0.4634745514643032 + 18.70856733065619 * eta - 167.9231114864569 * eta2 + 744.7699462372949 * eta3 - 1115.008825153004 * eta4) + 0.13469300326662165 * (-2.7311391326835133 + 72.17373498208947 * eta - 483.7040402103785 * eta2 + 1136.8367114738041 * eta3 - 472.02962341590774 * eta4) * s) * root
    return (float(iv1), float(iv2), float(iv3))

def xhm32_rd_amp_aux_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    eta7 = eta6 * eta
    delta = state.delta
    cd = state.chi_a
    cd2 = cd * cd
    s = state.s_tot_r
    aux1 = cd2 * (-4.188795724777721 * eta2 + 53.39200466700963 * eta3 - 131.19660856923554 * eta4) + cd * delta * (14.284921364132623 * eta2 - 321.26423637658746 * eta3 + 1242.865584938088 * eta4) + s * (-0.022968727462555794 * (83.66854837403105 * eta - 3330.6261333413177 * eta2 + 77424.12614733395 * eta3 - 710313.3016672594 * eta4 + 2693491.7075009225 * eta5 - 3572465.179268999 * eta6) + 0.0014795114305436387 * (-1672.7273629876313 * eta + 90877.38260964208 * eta2 - 1669016.9155105734 * eta3 + 13705532.554135624 * eta4 - 51161109.98398143 * eta5 + 70606676.6311127 * eta6) * s) + (4.45156488896258 * eta - 77.39303992494544 * eta2 + 522.5070635563092 * eta3 - 1642.3057499049708 * eta4 + 2048.333892310575 * eta5) / (1.0 - 9.611489164758915 * eta + 24.249594730050312 * eta2)
    spn = _chi_pn_hat(state)
    aux2 = cd2 * (-18.550171209458394 * eta2 + 188.99161055445936 * eta3 - 440.26516625611 * eta4) + cd * delta * (13.132625215315063 * eta2 - 340.5204040505528 * eta3 + 1327.1224176812448 * eta4) + spn * (-0.16707403272774676 * (6.678916447469937 * eta + 1331.480396625797 * eta2 - 41908.45179140144 * eta3 + 520786.0225074669 * eta4 - 3189462.4909922685 * eta5 + 9515538.23212259 * eta6 - 11006903.622406831 * eta7) + 0.015205286051218441 * (108.10032279461095 * eta - 16084.215590200103 * eta2 + 462957.5593513407 * eta3 - 5635028.227588545 * eta4 + 33799252.77713386 * eta5 - 98658152.75452062 * eta6 + 112013079.79786257 * eta7) * spn) + (3.902154247490771 * eta - 55.77521071924907 * eta2 + 294.9496843041973 * eta3 - 693.6803787318279 * eta4 + 636.0141528226893 * eta5) / (1.0 - 8.56699762573719 * eta + 19.119341007236955 * eta2)
    return (float(abs(aux1)), float(abs(aux2)))

def _xhm32_intermediate_amp_fit_values(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    delta = state.delta
    cd = state.chi_a
    cd2 = cd * cd
    root = math.sqrt(eta)
    spn = _chi_pn_hat(state)
    st = state.s_tot_r
    int1 = (cd2 * (-0.2341404256829785 * eta + 2.606326837996192 * eta2 - 8.68296921440857 * eta3) + cd * delta * (0.5454562486736877 * eta - 25.19759222940851 * eta2 + 73.40268975811729 * eta3)) * root + cd * delta * (0.4422257616009941 * eta - 8.490112284851655 * eta2 + 32.22238925527844 * eta3) * spn * root + spn * (0.7067243321652764 * (0.12885110296881636 + 9.608999847549535 * eta - 85.46581740280585 * eta2 + 325.71940024255775 * eta3 + 175.4194342269804 * eta4 - 1929.9084724384807 * eta5) + 0.1540566313813899 * (-0.3261041495083288 + 45.55785402900492 * eta - 827.591235943271 * eta2 + 7184.647314370326 * eta3 - 28804.241518798244 * eta4 + 43309.69769878964 * eta5) * spn) * root + (480.0434256230109 * eta + 25346.341240810478 * eta2 - 99873.4707358776 * eta3 + 106683.98302194536 * eta4) * root / (1.0 + 1082.6574834474493 * eta + 10083.297670051445 * eta2)
    int2 = eta * (cd2 * (-4.175680729484314 * eta + 47.54281549129226 * eta2 - 128.88334273588077 * eta3) + cd * delta * (-0.18274358639599947 * eta - 71.01128541687838 * eta2 + 208.07105580635888 * eta3)) + eta * (4.760999387359598 - 38.57900689641654 * eta + 456.2188780552874 * eta2 - 4544.076411013166 * eta3 + 24956.9592553473 * eta4 - 69430.10468748478 * eta5 + 77839.74180254337 * eta6) + cd * delta * eta * (1.2198776533959694 * eta - 26.816651899746475 * eta2 + 68.72798751937934 * eta3) * st + eta * st * (1.5098291294292217 * (0.4844667556328104 + 9.848766999273414 * eta - 143.66427232396376 * eta2 + 856.9917885742416 * eta3 - 1633.3295758142904 * eta4) + 0.32413108737204144 * (2.835358206961064 - 62.37317183581803 * eta + 761.6103793011912 * eta2 - 3811.5047139343505 * eta3 + 6660.304740652403 * eta4) * st)
    int3 = 3.881450518842405 * eta - 12.580316392558837 * eta2 + 1.7262466525848588 * eta3 + cd2 * (-7.065118823041031 * eta2 + 77.97950589523865 * eta3 - 203.65975422378446 * eta4) - 58.408542930248046 * eta4 + cd * delta * (1.924723094787216 * eta2 - 90.92716917757797 * eta3 + 387.00162600306226 * eta4) + 403.5748987560612 * eta5 + cd * delta * (-0.2566958540737833 * eta2 + 14.488550203412675 * eta3 - 26.46699529970884 * eta4) * spn + spn * (0.3650871458400108 * (71.57390929624825 * eta2 - 994.5272351916166 * eta3 + 6734.058809060536 * eta4 - 18580.859291282686 * eta5 + 16001.318492586077 * eta6) + 0.0960146077440495 * (451.74917589707513 * eta2 - 9719.470997418284 * eta3 + 83403.5743434538 * eta4 - 318877.43061174755 * eta5 + 451546.88775684836 * eta6) * spn - 0.03985156529181297 * (-304.92981902871617 * eta2 + 3614.518459296278 * eta3 - 7859.4784979916085 * eta4 - 46454.57664737511 * eta5 + 162398.81483375572 * eta6) * spn * spn)
    int4 = eta * (cd2 * (-8.572797326909152 * eta + 92.95723645687826 * eta2 - 236.2438921965621 * eta3) + cd * delta * (6.674358856924571 * eta - 171.4826985994883 * eta2 + 645.2760206304703 * eta3)) + eta * (3.921660532875504 - 16.57299637423352 * eta + 25.254017911686333 * eta2 - 143.41033155133266 * eta3 + 692.926425981414 * eta4) + cd * delta * eta * (-3.582040878719185 * eta + 57.75888914133383 * eta2 - 144.21651114700492 * eta3) * st + eta * st * (1.242750265695504 * (-0.522172424518215 + 25.168480118950065 * eta - 303.5223688400309 * eta2 + 1858.1518762309654 * eta3 - 3797.3561904195085 * eta4) + 0.2927045241764365 * (0.5056957789079993 - 15.488754837330958 * eta + 471.64047356915603 * eta2 - 3131.5783196211587 * eta3 + 6097.887891566872 * eta4) * st)
    return tuple((float(abs(v)) for v in (int1, int2, int3, int4)))

def xhm32_rd_phase_spheroidal_time_shift_fit(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    eta5 = eta4 * eta
    s = state.s_tot_r
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    no_spin = 11.851438981981772 + 167.95086712701223 * eta - 4565.033758777737 * eta2 + 61559.132976189896 * eta3 - 364129.24735853914 * eta4 + 739270.8814129328 * eta5
    eq_spin = (9.506768471271634 + 434.31707030999445 * eta - 8046.364492927503 * eta2 + 26929.677144312944 * eta3) * s + (-5.949655484033632 - 307.67253970367034 * eta + 1334.1062451631644 * eta2 + 3575.347142399199 * eta3) * s2 + (3.4881615575084797 - 2244.4613237912527 * eta + 24145.932943269272 * eta2 - 60929.87465551446 * eta3) * s3 + (15.585154698977842 - 2292.778112523392 * eta + 24793.809334683185 * eta2 - 65993.84497923202 * eta3) * s4
    uneq_spin = 465.7904934097202 * state.dchi * state.delta * eta2
    return float(no_spin + eq_spin + uneq_spin)

def xhm32_rd_phase_spheroidal_phase_shift_fit(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    eta7 = eta6 * eta
    s = state.s_tot_r
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    denom = -2.950271397057221 + s
    no_spin = -1.3328895897490733 - 22.209549522908667 * eta + 1056.2426481245027 * eta2 - 21256.376324666326 * eta3 + 246313.12887984765 * eta4 - 1631296.8467540336 * eta5 + 5614617.173188322 * eta6 - 7612233.821752137 * eta7
    eq_spin = s * (-1.622727240110213 + 0.9960210841611344 * s - 1.1239505323267036 * s2 - 1.9586085340429995 * s3 + eta2 * (196.7055281997748 + 135.25216875394943 * s + 1086.7504825459278 * s2 + 546.6246807461155 * s3 - 312.1010566468068 * s4) + 0.7638287749489343 * s4 + eta * (-47.475568056234245 - 35.074072557604445 * s - 97.16014978329918 * s2 - 34.498125910065156 * s3 + 24.02858084544326 * s4) + eta3 * (62.632493533037625 - 22.59781899512552 * s - 2683.947280170815 * s2 - 1493.177074873678 * s3 + 805.0266029288334 * s4)) / denom
    uneq_spin = state.delta * (state.chi2z * eta ** 2.5 * (88.56162028006072 - 30.01812659282717 * s) + state.chi2z * eta2 * (43.126266433486435 - 14.617728550838805 * s) + state.chi1z * eta2 * (-43.126266433486435 + 14.617728550838805 * s) + state.chi1z * eta ** 2.5 * (-88.56162028006072 + 30.01812659282717 * s)) / denom
    return float(no_spin + eq_spin + uneq_spin)

def _xhm32_rd_phase_fit_values_122019(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    eta5 = eta4 * eta
    s = state.s_tot_r
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    s5 = s4 * s
    d = state.dchi * state.delta
    p1 = 3169.372056189274 + 426.8372805022653 * eta - 12569.748101922158 * eta2 + 149846.7281073725 * eta3 - 817182.2896823225 * eta4 + 1567405.3633767858 * eta5 + (19.23408352151287 - 1762.6573670619173 * eta + 7855.316419853637 * eta2 - 3785.49764771212 * eta3) * s + (-42.88446003698396 + 336.8340966473415 * eta - 5615.908682338113 * eta2 + 20497.5021807654 * eta3) * s2 + (13.918237996338371 + 10145.53174542332 * eta - 91664.12621864353 * eta2 + 201204.5096556517 * eta3) * s3 + (-24.72321125342808 - 4901.068176970293 * eta + 53893.9479532688 * eta2 - 139322.02687945773 * eta3) * s4 + (-61.01931672442576 - 16556.65370439302 * eta + 162941.8009556697 * eta2 - 384336.57477596396 * eta3) * s5 + state.dchi * state.delta * eta2 * (641.2473192044652 - 1600.240100295189 * state.chi1z * eta + 1600.240100295189 * state.chi2z * eta + 13275.623692212472 * eta * s)
    p2 = 3131.0260952676376 + 206.09687819102305 * eta - 2636.4344627081873 * eta2 + 7475.062269742079 * eta3 + (49.90874152040307 - 691.9815135740145 * eta - 434.60154548208334 * eta2 + 10514.68111669422 * eta3) * s + (97.3078084654917 - 3458.2579971189534 * eta + 26748.805404989867 * eta2 - 56142.13736008524 * eta3) * s2 + (-132.49105074500454 + 429.0787542102207 * eta + 7269.262546204149 * eta2 - 27654.067482558712 * eta3) * s3 + (-227.8023564332453 + 5119.138772157134 * eta - 34444.2579678986 * eta2 + 69666.01833764123 * eta3) * s4 + 477.51566939885424 * d * eta2
    p3 = 3082.803556599222 + 76.94679795837645 * eta - 586.2469821978381 * eta2 + 977.6115755788503 * eta3 + (45.08944710349874 - 807.7353772747749 * eta + 1775.4343704616288 * eta2 + 2472.6476419567534 * eta3) * s + (95.57355060136699 - 2224.9613131172046 * eta + 13821.251641893134 * eta2 - 25583.314298758105 * eta3) * s2 + (-144.96370424517866 + 2268.4693587493093 * eta - 10971.864789147161 * eta2 + 16259.911572457446 * eta3) * s3 + (-227.8023564332453 + 5119.138772157134 * eta - 34444.2579678986 * eta2 + 69666.01833764123 * eta3) * s4 + 378.2359918274837 * d * eta2
    p4 = 3077.0657367004565 + 64.99844502520415 * eta - 357.38692756785395 * eta2 + (34.793450080444714 - 986.7751755509875 * eta - 9490.641676924794 * eta3 + 5700.682624203565 * eta2) * s + (57.38106384558743 - 1644.6690499868596 * eta - 19906.416384606226 * eta3 + 11008.881935880598 * eta2) * s2 + (-126.02362949830213 + 3169.3397351803583 * eta + 62863.79877094988 * eta3 - 26766.730897942085 * eta2) * s3 + (-169.30909412804587 + 4900.706039920717 * eta + 95314.99988114933 * eta3 - 41414.05689348732 * eta2) * s4 + 390.5443469721231 * d * eta2
    return (float(p1), float(p2), float(p3), float(p4))

def xhm32_rd_amp_alambda_fit(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    s = state.s_tot_r
    s2 = s * s
    chi = state.chi_a
    chi2 = chi * chi
    return float(chi2 * (-3.4614418482110163 * eta3 + 35.464117772624164 * eta4 - 85.19723511005235 * eta5) + chi * state.delta * (2.0328561081997463 * eta3 - 46.18751757691501 * eta4 + 170.9266105597438 * eta5) + chi2 * (-0.4600401291210382 * eta3 + 12.23450117663151 * eta4 - 42.74689906831975 * eta5) * s + chi * state.delta * (5.786292428422767 * eta3 - 53.60467819078566 * eta4 + 117.66195692191727 * eta5) * s + s * (-0.0013330716557843666 * (56.35538385647113 * eta - 1218.1550992423377 * eta2 + 16509.69605686402 * eta3 - 102969.88022112886 * eta4 + 252228.94931931415 * eta5 - 150504.2927996263 * eta6) + 0.0010126460331462495 * (-33.87083889060834 * eta + 502.6221651850776 * eta2 - 1304.9210590188136 * eta3 - 36980.079328277505 * eta4 + 295469.28617550555 * eta5 - 597155.7619486618 * eta6) * s - 0.00043088431510840695 * (-30.014415072587354 * eta - 1900.5495690280086 * eta2 + 76517.21042363928 * eta3 - 870035.1394696251 * eta4 + 3907267.4134789007 * eta5 - 6094089.675611567 * eta6) * s2) + (0.08408469319155859 * eta - 1.223794846617597 * eta2 + 6.5972460654253515 * eta3 - 15.707327897569396 * eta4 + 14.163264397061505 * eta5) / (1.0 - 8.612447115134758 * eta + 18.93655612952139 * eta2))

def xhm32_rd_amp_lambda_fit(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    eta5 = eta4 * eta
    s = state.s_tot_r
    chi = state.chi_a
    chi2 = chi * chi
    return float(0.978510781593996 + 0.36457571743142897 * eta - 12.259851752618998 * eta2 + 49.19719473681921 * eta3 + chi * state.delta * (-188.37119473865533 * eta3 + 2151.8731700399308 * eta4 - 6328.182823770599 * eta5) + chi2 * (115.3689949926392 * eta3 - 1159.8596972989067 * eta4 + 2657.6998831179444 * eta5) + s * (0.22358643406992756 * (0.48943645614341924 - 32.06682257944444 * eta + 365.2485484044132 * eta2 - 915.2489655397206 * eta3) + 0.0792473022309144 * (1.877251717679991 - 103.65639889587327 * eta + 1202.174780792418 * eta2 - 3206.340850767219 * eta3) * s))

def xhm32_rd_amp_sigma_fit(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    eta5 = eta4 * eta
    s = state.s_tot_r
    chi = state.chi_a
    chi2 = chi * chi
    return float(1.3353917551819414 + 0.13401718687342024 * eta + chi * state.delta * (144.37065005786636 * eta3 - 754.4085447486738 * eta4 + 123.86194078913776 * eta5) + chi2 * (209.09202210427972 * eta3 - 1769.4658099037918 * eta4 + 3592.287297392387 * eta5) + s * (-0.012086025709597246 * (-6.230497473791485 + 600.5968613752918 * eta - 6606.1009717965735 * eta2 + 17277.60594350428 * eta3) - 0.06066548829900489 * (-0.9208054306316676 + 142.0346574366267 * eta - 1567.249168668069 * eta2 + 4119.373703246675 * eta3) * s))

def _mode32_state(params):
    base = _mode21_state(params)
    final_mass = 1.0 - base.radiated_energy
    final_spin = base.final_spin
    mixing_322 = -complex(
        evaluate_qnmfit_re_l3m2lp2(final_spin),
        evaluate_qnmfit_im_l3m2lp2(final_spin),
    )
    mixing_323 = -complex(
        evaluate_qnmfit_re_l3m2lp3(final_spin),
        evaluate_qnmfit_im_l3m2lp3(final_spin),
    )
    return _Mode32State(
        base=base,
        f_ring_32=qnm_fring32_fit(final_spin) / final_mass,
        f_damp_32=qnm_fdamp32_fit(final_spin) / final_mass,
        mixing_322=mixing_322,
        mixing_323=mixing_323,
    )


def _transition_frequencies(state):
    if state.q < 20.0:
        f_amp_in = state.f_meco_32
    else:
        blend = 0.5 + 0.5 * math.tanh((state.eta - 0.0192234) / 0.004)
        fcut_emr = (
            2.5
            * (
                0.011671068725758493
                - 0.0000858396080377194 * state.chi1z
                + 0.000316707064291237 * state.chi1z**2
            )
            * (0.8447212540381764 + 6.2873167352395125 * state.eta)
            / (1.2857082764038923 - 0.9977728883419751 * state.chi1z)
        )
        f_amp_in = blend * state.f_meco_32 + (1.0 - blend) * fcut_emr

    f_end = state.f_ring_22 - 0.5 * state.f_damp_22
    f_phase_in = (1.0 + 0.001 * (0.25 / state.eta - 1.0)) * state.f_meco_32
    return _Transitions(
        f_amp_in=f_amp_in,
        f_amp_rd=f_end,
        f_phase_in=f_phase_in,
        f_phase_rd=f_end,
        phase_intermediate_points=(
            f_phase_in,
            (
                math.sqrt(3.0) * (f_phase_in - f_end)
                + 2.0 * (f_phase_in + f_end)
            )
            / 4.0,
            (3.0 * f_phase_in + f_end) / 4.0,
            (f_phase_in + f_end) / 2.0,
            f_end,
            f_end,
        ),
        phase_ringdown_points=(
            state.f_ring_22,
            state.f_ring_32 - 1.5 * state.f_damp_32,
            state.f_ring_32 - 0.5 * state.f_damp_32,
            state.f_ring_32 + 0.5 * state.f_damp_32,
        ),
    )


def _lambda_pn(state):
    if state.eta > 0.01:
        return (
            2376.0
            * _PI
            * (-5.0 + 22.0 * state.eta)
            / (-3960.0 + 11880.0 * state.eta)
        )
    return xhm32_inspiral_phase_lambda_fit(state)


def _pn_amplitude_coefficients(state):
    eta = state.eta
    eta2 = eta * eta
    chi_s = state.chi_s
    chi_a = state.chi_a
    delta = state.delta
    return (
        0.0,
        0.0,
        (-1.0 + 3.0 * eta) * _PI ** (2.0 / 3.0),
        -4.0 * chi_s * eta * _PI,
        (10471.0 - 61625.0 * eta + 82460.0 * eta2)
        / 10080.0
        * _PI ** (4.0 / 3.0),
        (
            2520.0j
            - 3955.0 * chi_s
            - 3955.0 * chi_a * delta
            - 11088.0j * eta
            + 10810.0 * chi_s * eta
            + 11865.0 * chi_a * delta * eta
            - 12600.0 * chi_s * eta2
        )
        / 840.0
        * _PI ** (5.0 / 3.0),
        (
            824173699.0
            + 2263282560.0 * chi_a * chi_s * delta
            - 26069649.0 * eta
            - 15209631360.0 * chi_a * chi_s * delta * eta
            + 3576545280.0 * chi_s * eta * _PI
            + 1131641280.0 * chi_a**2
            - 7865605440.0 * eta * chi_a**2
            + 1131641280.0 * chi_s**2
            - 11870591040.0 * eta * chi_s**2
            - 13202119896.0 * eta2
            + 13412044800.0 * chi_a**2 * eta2
            + 5830513920.0 * chi_s**2 * eta2
            + 5907445488.0 * eta**3
        )
        / 447068160.0
        * _PI**2,
    )


def _pn_amplitude(frequency, coefficients, amp_norm, pn_global_factor):
    series = torch.zeros_like(frequency, dtype=coefficients.dtype)
    for coefficient, exponent in zip(coefficients, _PN_EXPONENTS):
        series = series + coefficient * frequency**exponent
    return (
        torch.abs(series)
        * pn_global_factor
        * amp_norm
        * frequency ** (-7.0 / 6.0)
    )


def _ringdown_spheroidal_phase(frequency, state, phase):
    return (
        phase.phi0_s
        + phase.alpha0_s * frequency
        - phase.alpha2_s / frequency
        - phase.alpha4_s / (3.0 * frequency**3)
        + phase.alpha_l_s
        * torch.atan((frequency - state.f_ring_32) / state.f_damp_32)
    )


def _ringdown_spheroidal_phase_derivative(frequency, state, phase):
    return (
        phase.alpha0_s
        + phase.alpha2_s / frequency**2
        + phase.alpha4_s / frequency**4
        + phase.alpha_l_s
        * state.f_damp_32
        / (state.f_damp_32**2 + (frequency - state.f_ring_32) ** 2)
    )


def _ringdown_spheroidal_phase_without_constant(frequency, state, coefficients):
    alpha0_s, alpha_l_s, alpha2_s, alpha4_s = coefficients
    return (
        alpha0_s * frequency
        - alpha2_s / frequency
        - alpha4_s / (3.0 * frequency**3)
        + alpha_l_s
        * torch.atan((frequency - state.f_ring_32) / state.f_damp_32)
    )


def _partial_phase(
    mf,
    state,
    transitions,
    intrinsic,
    phase_table,
    reference_frequency,
    coa_phase,
):
    rows = [
        [
            1.0,
            state.f_damp_32
            / (
                state.f_damp_32**2
                + (frequency - state.f_ring_32) ** 2
            ),
            frequency**-2,
            frequency**-4,
        ]
        for frequency in transitions.phase_ringdown_points
    ]
    coefficients = _solve(
        rows,
        list(_xhm32_rd_phase_fit_values_122019(state)),
        mf,
    )
    alpha0_s, alpha_l_s, alpha2_s, alpha4_s = coefficients.unbind()

    _, linb_fit, psi4_to_strain = _xutils.calc_phaseatpeak(
        state.eta, state.s_tot_r, state.dchi, state.delta
    )
    linb_fit = _as_float(linb_fit)
    psi4_to_strain = _as_float(psi4_to_strain)
    derivative_frequency = (
        state.f_ring_22 - state.f_damp_22
    ) / state.total_mass_seconds
    dphi22_ref = (
        PhaseDerivative(
            _tensor(derivative_frequency, mf),
            intrinsic,
            phase_table,
        )
        / state.total_mass_seconds
    )
    linb = (
        linb_fit
        - dphi22_ref
        - 2.0 * _PI * (500.0 + psi4_to_strain)
    )

    derivative_match = _tensor(
        state.f_ring_22 + state.f_damp_22,
        mf,
    )
    dphi22_match = (
        PhaseDerivative(
            derivative_match / state.total_mass_seconds,
            intrinsic,
            phase_table,
        )
        / state.total_mass_seconds
        + linb
    )
    raw_derivative = (
        alpha0_s
        + alpha2_s / derivative_match**2
        + alpha4_s / derivative_match**4
        + alpha_l_s
        * state.f_damp_32
        / (
            state.f_damp_32**2
            + (derivative_match - state.f_ring_32) ** 2
        )
    )
    alpha0_s = (
        alpha0_s
        + dphi22_match
        + xhm32_rd_phase_spheroidal_time_shift_fit(state)
        - raw_derivative
    )

    mf_ref = reference_frequency * state.total_mass_seconds
    phiref22 = (
        -Phase(_tensor(reference_frequency, mf), intrinsic, phase_table)
        - linb * mf_ref
        + 2.0 * coa_phase
        + _PI / 4.0
    )
    phase_match = _tensor(state.f_ring_22, mf)
    phase22_match = (
        Phase(
            phase_match / state.total_mass_seconds,
            intrinsic,
            phase_table,
        )
        + linb * phase_match
        + phiref22
    )
    phi0_s = (
        phase22_match
        - _ringdown_spheroidal_phase_without_constant(
            phase_match,
            state,
            (alpha0_s, alpha_l_s, alpha2_s, alpha4_s),
        )
        + xhm32_rd_phase_spheroidal_phase_shift_fit(state)
    )
    return _Phase32(
        transitions=transitions,
        linb=linb,
        phiref22=phiref22,
        alpha0_s=alpha0_s,
        alpha_l_s=alpha_l_s,
        alpha2_s=alpha2_s,
        alpha4_s=alpha4_s,
        phi0_s=phi0_s,
    )


def _ringdown_lorentzian(frequency, state, amplitude):
    offset = frequency - state.f_ring_32
    width = state.f_damp_32 * amplitude.rd_sigma
    return (
        amplitude.rd_alambda
        * state.f_damp_32
        / (
            torch.exp(amplitude.rd_lambda * offset / width)
            * (offset * offset + width * width)
        )
    )


def _ringdown_lorentzian_derivative(frequency, state, amplitude):
    offset = frequency - state.f_ring_32
    fdamp = state.f_damp_32
    sigma = amplitude.rd_sigma
    numerator = amplitude.rd_alambda * (
        offset * offset * amplitude.rd_lambda
        + 2.0 * fdamp * offset * sigma
        + fdamp * fdamp * amplitude.rd_lambda * sigma * sigma
    )
    denominator = (
        sigma
        * (offset * offset + fdamp * fdamp * sigma * sigma) ** 2
        * torch.exp(offset * amplitude.rd_lambda / (fdamp * sigma))
    )
    return -numerator / denominator


def _ringdown_spheroidal_amplitude(frequency, state, amplitude):
    auxiliary = torch.zeros_like(frequency)
    frequency_power = torch.ones_like(frequency)
    for coefficient in amplitude.rd_aux_coefficients:
        auxiliary = auxiliary + coefficient * frequency_power
        frequency_power = frequency_power * frequency

    central = _ringdown_lorentzian(frequency, state, amplitude)
    tail = amplitude.tail_amplitude * torch.exp(
        -amplitude.tail_decay * (frequency - amplitude.f_falloff)
    )
    return torch.where(
        frequency < amplitude.f_rd_aux,
        auxiliary,
        torch.where(frequency < amplitude.f_falloff, central, tail),
    )


def _h22_ringdown_component(
    frequency,
    state,
    phase,
    amplitude,
    intrinsic,
    phase_table,
    amp_table,
):
    amp22, _ = get_mergerringdown_Amp(frequency, intrinsic, amp_table)
    phase22 = (
        Phase(
            frequency / state.total_mass_seconds,
            intrinsic,
            phase_table,
        )
        + phase.linb * frequency
        + phase.phiref22
    )
    return (
        amp22
        * amplitude.amp_norm
        * frequency ** (-7.0 / 6.0)
        * torch.exp(1j * phase22)
    )


def _mixed_ringdown_component(
    frequency,
    state,
    phase,
    amplitude,
    intrinsic,
    phase_table,
    amp_table,
):
    h22 = _h22_ringdown_component(
        frequency,
        state,
        phase,
        amplitude,
        intrinsic,
        phase_table,
        amp_table,
    )
    h32_spheroidal = _ringdown_spheroidal_amplitude(
        frequency,
        state,
        amplitude,
    ) * torch.exp(1j * _ringdown_spheroidal_phase(frequency, state, phase))
    mixing_322 = _tensor(
        state.mixing_322,
        frequency,
        complex_value=True,
    ).conj()
    mixing_323 = _tensor(
        state.mixing_323,
        frequency,
        complex_value=True,
    ).conj()
    return mixing_322 * h22 + mixing_323 * h32_spheroidal


def _inspiral_amplitude(frequency, state, transitions, amplitude):
    ratio = frequency / transitions.f_amp_in
    pseudo = (
        amplitude.pseudo_coefficients[0] * ratio ** (7.0 / 3.0)
        + amplitude.pseudo_coefficients[1] * ratio ** (8.0 / 3.0)
        + amplitude.pseudo_coefficients[2] * ratio**3
    )
    return _pn_amplitude(
        frequency,
        amplitude.pn_coefficients,
        amplitude.amp_norm,
        amplitude.pn_global_factor,
    ) + amplitude.amp_norm * frequency ** (-7.0 / 6.0) * pseudo


def _intermediate_amplitude(frequency, amplitude):
    polynomial = torch.zeros_like(frequency)
    frequency_power = torch.ones_like(frequency)
    for coefficient in amplitude.intermediate_coefficients:
        polynomial = polynomial + coefficient * frequency_power
        frequency_power = frequency_power * frequency
    return polynomial * frequency ** (-7.0 / 6.0)


def _build_amplitude(
    mf,
    state,
    transitions,
    phase,
    intrinsic,
    phase_table,
    amp_table,
):
    amp_norm = math.sqrt(2.0 * state.eta / 3.0) * _PI ** (-1.0 / 6.0)
    pn_global_factor = math.sqrt(5.0 / 7.0) / 3.0
    pn_coefficients = _tensor(
        _pn_amplitude_coefficients(state),
        mf,
        complex_value=True,
    )

    collocation_frequencies = (
        0.5 * transitions.f_amp_in,
        0.75 * transitions.f_amp_in,
        transitions.f_amp_in,
    )
    fit_values = _xhm32_inspiral_amp_fit_values(state)
    pseudo_targets = []
    pseudo_rows = []
    for frequency, fit in zip(collocation_frequencies, fit_values):
        frequency_tensor = _tensor(frequency, mf)
        pn_value = _pn_amplitude(
            frequency_tensor,
            pn_coefficients,
            amp_norm,
            pn_global_factor,
        )
        pseudo_targets.append(
            (abs(fit) - pn_value)
            / (amp_norm * frequency_tensor ** (-7.0 / 6.0))
        )
        ratio = frequency / transitions.f_amp_in
        pseudo_rows.append(
            [ratio ** (7.0 / 3.0), ratio ** (8.0 / 3.0), ratio**3]
        )
    pseudo_coefficients = _solve(pseudo_rows, pseudo_targets, mf)

    alambda = abs(xhm32_rd_amp_alambda_fit(state))
    rd_lambda = xhm32_rd_amp_lambda_fit(state)
    rd_sigma = xhm32_rd_amp_sigma_fit(state)
    f_rd_aux = state.f_ring_32 - state.f_damp_32
    f_falloff = state.f_ring_32 + 2.0 * state.f_damp_32
    placeholder = _tensor((0.0, 0.0, 0.0, 0.0), mf)
    amplitude = _Amplitude32(
        amp_norm=amp_norm,
        pn_global_factor=pn_global_factor,
        pn_coefficients=pn_coefficients,
        pseudo_coefficients=pseudo_coefficients,
        rd_alambda=alambda,
        rd_lambda=rd_lambda,
        rd_sigma=rd_sigma,
        f_rd_aux=f_rd_aux,
        f_falloff=f_falloff,
        tail_amplitude=_tensor(0.0, mf),
        tail_decay=_tensor(0.0, mf),
        rd_aux_coefficients=placeholder,
    )
    falloff_tensor = _tensor(f_falloff, mf)
    amplitude.tail_amplitude = _ringdown_lorentzian(
        falloff_tensor,
        state,
        amplitude,
    )
    amplitude.tail_decay = (
        -_ringdown_lorentzian_derivative(
            falloff_tensor,
            state,
            amplitude,
        )
        / amplitude.tail_amplitude
    )

    aux_values = xhm32_rd_amp_aux_fit_values(state)
    auxiliary_frequencies = (
        transitions.f_amp_rd,
        0.5 * (transitions.f_amp_rd + f_rd_aux),
        f_rd_aux,
        f_rd_aux,
    )
    f0, f1, f2, f3 = auxiliary_frequencies
    auxiliary_rows = (
        (1.0, f0, f0**2, f0**3),
        (1.0, f1, f1**2, f1**3),
        (1.0, f2, f2**2, f2**3),
        (0.0, 1.0, 2.0 * f3, 3.0 * f3**2),
    )
    auxiliary_targets = (
        aux_values[0],
        aux_values[1],
        _ringdown_lorentzian(_tensor(f_rd_aux, mf), state, amplitude),
        _ringdown_lorentzian_derivative(
            _tensor(f_rd_aux, mf),
            state,
            amplitude,
        ),
    )
    amplitude.rd_aux_coefficients = _solve(
        auxiliary_rows,
        auxiliary_targets,
        mf,
    )

    def inspiral(frequency):
        return _inspiral_amplitude(
            frequency,
            state,
            transitions,
            amplitude,
        )

    def mixed(frequency):
        return torch.abs(
            _mixed_ringdown_component(
                frequency,
                state,
                phase,
                amplitude,
                intrinsic,
                phase_table,
                amp_table,
            )
        )

    spacing = (transitions.f_amp_rd - transitions.f_amp_in) / 5.0
    frequencies = (
        transitions.f_amp_in,
        transitions.f_amp_in,
        transitions.f_amp_in + spacing,
        transitions.f_amp_in + 2.0 * spacing,
        transitions.f_amp_in + 3.0 * spacing,
        transitions.f_amp_in + 4.0 * spacing,
        transitions.f_amp_rd,
        transitions.f_amp_rd,
    )
    left_value, left_derivative = _inspiral_boundary(
        inspiral,
        transitions.f_amp_in,
        mf,
    )
    if mf.dtype == torch.float64:
        right_value, right_derivative = _inspiral_boundary(
            mixed,
            transitions.f_amp_rd,
            mf,
        )
    else:
        # MPS does not implement complex-double autograd. A centered
        # float32-scale difference keeps this boundary calculation on-device.
        point = transitions.f_amp_rd
        step = 1.0e-4
        right_value = mixed(_tensor(point, mf))
        right_derivative = (
            mixed(_tensor(point + step, mf))
            - mixed(_tensor(point - step, mf))
        ) / (2.0 * step)
    intermediate_targets = (
        left_value,
        left_derivative,
        *(_tensor(value, mf) for value in _xhm32_intermediate_amp_fit_values(state)),
        right_value,
        right_derivative,
    )
    intermediate_rows = []
    for index, frequency in enumerate(frequencies):
        if index in (1, 7):
            intermediate_rows.append(
                [
                    (power - 7.0 / 6.0)
                    * frequency ** (power - 1.0 - 7.0 / 6.0)
                    for power in range(8)
                ]
            )
        else:
            intermediate_rows.append(
                [
                    frequency ** (power - 7.0 / 6.0)
                    for power in range(8)
                ]
            )
    amplitude.intermediate_coefficients = _solve(
        intermediate_rows,
        intermediate_targets,
        mf,
    )
    return amplitude


def _mixed_phase_at(
    frequency,
    state,
    phase,
    amplitude,
    intrinsic,
    phase_table,
    amp_table,
):
    angle = torch.fmod(
        torch.angle(
            _mixed_ringdown_component(
                frequency,
                state,
                phase,
                amplitude,
                intrinsic,
                phase_table,
                amp_table,
            )
        ),
        2.0 * _PI,
    )
    return torch.where(angle > 0.0, angle - 2.0 * _PI, angle)


def _complete_phase(
    mf,
    state,
    phase,
    amplitude,
    intrinsic,
    phase_table,
    amp_table,
):
    transitions = phase.transitions
    cutoff = transitions.f_phase_rd
    phase_zero = _mixed_phase_at(
        _tensor(cutoff, mf),
        state,
        phase,
        amplitude,
        intrinsic,
        phase_table,
        amp_table,
    )
    if mf.dtype == torch.float64:
        with torch.enable_grad():
            frequency = _tensor(cutoff, mf).detach().requires_grad_(True)
            angle = torch.angle(
                _mixed_ringdown_component(
                    frequency,
                    state,
                    phase,
                    amplitude,
                    intrinsic,
                    phase_table,
                    amp_table,
                )
            )
            derivative_zero = torch.autograd.grad(
                angle, frequency, create_graph=True
            )[0]
            second_derivative_zero = torch.autograd.grad(
                derivative_zero, frequency
            )[0]
        derivative_zero = derivative_zero.detach()
        second_derivative_zero = second_derivative_zero.detach()
    else:
        step = 1.0e-4
        phase_minus = _mixed_phase_at(
            _tensor(cutoff - step, mf),
            state,
            phase,
            amplitude,
            intrinsic,
            phase_table,
            amp_table,
        )
        phase_plus = _mixed_phase_at(
            _tensor(cutoff + step, mf),
            state,
            phase,
            amplitude,
            intrinsic,
            phase_table,
            amp_table,
        )
        derivative_zero = (phase_plus - phase_minus) / (2.0 * step)
        second_derivative_zero = (
            phase_plus - 2.0 * phase_zero + phase_minus
        ) / step**2

    frequencies = list(transitions.phase_intermediate_points)
    fit_values = list(_xhm32_intermediate_phase_fit_values(state))
    if state.eta > 0.05:
        frequencies[-2] = cutoff
        fit_values[-2] = derivative_zero
    frequencies[-1] = cutoff
    fit_values[-1] = second_derivative_zero

    rows = []
    for index, frequency in enumerate(frequencies):
        if index == len(frequencies) - 1:
            offset = frequency - state.f_ring_32
            denominator = state.f_damp_32**2 + offset**2
            rows.append(
                [
                    0.0,
                    -2.0 * state.f_damp_32 * offset / denominator**2,
                    -frequency**-2,
                    -2.0 * frequency**-3,
                    -4.0 * frequency**-5,
                    -3.0 * frequency**-4,
                ]
            )
        else:
            rows.append(
                [
                    1.0,
                    state.f_damp_32
                    / (
                        state.f_damp_32**2
                        + (frequency - state.f_ring_32) ** 2
                    ),
                    1.0 / frequency,
                    1.0 / frequency**2,
                    1.0 / frequency**4,
                    1.0 / frequency**3,
                ]
            )
    c0, c_l, c1, c2, c4, c3 = _solve(rows, fit_values, mf).unbind()

    def intermediate_raw(frequency):
        return (
            c0 * frequency
            + c1 * torch.log(frequency)
            - c2 / frequency
            - c4 / (3.0 * frequency**3)
            - 0.5 * c3 / frequency**2
            + c_l
            * torch.atan(
                (frequency - state.f_ring_32) / state.f_damp_32
            )
        )

    def intermediate_derivative(frequency):
        return (
            c0
            + c_l
            * state.f_damp_32
            / (
                state.f_damp_32**2
                + (frequency - state.f_ring_32) ** 2
            )
            + c1 / frequency
            + c2 / frequency**2
            + c4 / frequency**4
            + c3 / frequency**3
        )

    if state.eta < 0.05:
        c0 = c0 + derivative_zero - intermediate_derivative(
            _tensor(cutoff, mf)
        )

    lambda_pn = _lambda_pn(state)

    def inspiral_raw(frequency):
        return (
            get_inspiral_phase(frequency, intrinsic, phase_table) / state.eta
            + lambda_pn * frequency
        )

    frequency_in = _tensor(transitions.f_phase_in, mf)
    inspiral_value, inspiral_derivative = _value_and_derivative(
        inspiral_raw,
        transitions.f_phase_in,
        mf,
    )
    intermediate_value = intermediate_raw(frequency_in)
    intermediate_slope = intermediate_derivative(frequency_in)
    c1_insp = intermediate_slope - inspiral_derivative
    c_insp = (
        -c1_insp * frequency_in
        + intermediate_value
        - inspiral_value
    )

    frequency_rd = _tensor(cutoff, mf)
    intermediate_rd = intermediate_raw(frequency_rd)
    intermediate_slope_rd = intermediate_derivative(frequency_rd)
    c1_rd = intermediate_slope_rd - derivative_zero
    c_rd = -c1_rd * frequency_rd + intermediate_rd - phase_zero

    f_align = state.f_meco_22
    if state.eta > 0.05:
        f_align *= 0.6
    align_tensor = _tensor(f_align, mf)
    xas_align = Phase(
        align_tensor / state.total_mass_seconds,
        intrinsic,
        phase_table,
    )
    mode_align = (
        inspiral_raw(align_tensor)
        + c1_insp * align_tensor
        + c_insp
    )
    delta_phi = torch.fmod(
        xas_align
        + phase.phiref22
        + phase.linb * align_tensor
        - mode_align,
        2.0 * _PI,
    )

    phase.c0 = c0
    phase.c_l = c_l
    phase.c1 = c1
    phase.c2 = c2
    phase.c3 = c3
    phase.c4 = c4
    phase.c1_insp = c1_insp
    phase.c_insp = c_insp
    phase.c1_rd = c1_rd
    phase.c_rd = c_rd
    phase.delta_phi = delta_phi
    return phase


def _evaluate_phase(
    mf,
    state,
    phase,
    mixed_ringdown,
    intrinsic,
    phase_table,
):
    inspiral_raw = (
        get_inspiral_phase(mf, intrinsic, phase_table) / state.eta
        + _lambda_pn(state) * mf
    )
    inspiral = (
        inspiral_raw
        + phase.c1_insp * mf
        + phase.c_insp
        + phase.delta_phi
    )
    intermediate = (
        phase.c0 * mf
        + phase.c1 * torch.log(mf)
        - phase.c2 / mf
        - phase.c4 / (3.0 * mf**3)
        - 0.5 * phase.c3 / mf**2
        + phase.c_l
        * torch.atan((mf - state.f_ring_32) / state.f_damp_32)
        + phase.delta_phi
    )
    ringdown = (
        torch.angle(mixed_ringdown)
        + phase.c1_rd * mf
        + phase.c_rd
        + phase.delta_phi
    )
    return torch.where(
        mf < phase.transitions.f_phase_in,
        inspiral,
        torch.where(
            mf < phase.transitions.f_phase_rd,
            intermediate,
            ringdown,
        ),
    )


def _evaluate_amplitude(mf, state, transitions, amplitude, mixed_ringdown):
    inspiral = _inspiral_amplitude(
        mf,
        state,
        transitions,
        amplitude,
    )
    intermediate = _intermediate_amplitude(mf, amplitude)
    ringdown = torch.abs(mixed_ringdown)
    result = torch.where(
        mf < transitions.f_amp_in,
        inspiral,
        torch.where(mf < transitions.f_amp_rd, intermediate, ringdown),
    )
    return torch.where(result < 0.0, _FALSE_ZERO, result)


def _move_coefficient_tensors(coefficients, like):
    for name, value in vars(coefficients).items():
        if isinstance(value, torch.Tensor):
            dtype = (
                torch.complex64
                if value.is_complex() and like.dtype == torch.float32
                else torch.complex128
                if value.is_complex()
                else like.dtype
            )
            setattr(
                coefficients,
                name,
                value.to(device=like.device, dtype=dtype),
            )


def imrphenomxhm_h3m2_samples(
    core,
    params,
    *,
    frequencies=None,
    reference_frequency=None,
):
    r"""Return active positive-frequency samples of LAL's h_(3,-2)."""

    state = _mode32_state(params)
    if frequencies is None:
        frequencies = (
            torch.arange(
                core.first_bin,
                core.stop_bin,
                device=core.polarization.device,
                dtype=core.polarization.real.dtype,
            )
            * core.delta_f
        )
    mf = frequencies * state.total_mass_seconds
    intrinsic = torch.tensor(
        [state.mass1, state.mass2, state.chi1, state.chi2],
        device=frequencies.device,
        dtype=frequencies.dtype,
    )
    phase_table = _xutils.PhenomX_phase_coeff_table.to(
        device=frequencies.device,
        dtype=frequencies.dtype,
    )
    amp_table = _xutils.PhenomX_amp_coeff_table.to(
        device=frequencies.device,
        dtype=frequencies.dtype,
    )
    if reference_frequency is None:
        reference_frequency = float(params.get("f_ref", 0.0))
        if reference_frequency <= 0.0:
            reference_frequency = float(params["f_lower"])
    coa_phase = float(params.get("coa_phase", 0.0))

    # MPS only supports float32 waveforms. Build the small, ill-conditioned
    # matching systems in CPU float64, then move their coefficients back; the
    # frequency-dependent work remains on the requested device.
    setup = frequencies
    setup_intrinsic = intrinsic
    setup_phase_table = phase_table
    setup_amp_table = amp_table
    if frequencies.dtype != torch.float64:
        setup = torch.empty((), dtype=torch.float64)
        setup_intrinsic = torch.tensor(
            [state.mass1, state.mass2, state.chi1, state.chi2],
            dtype=torch.float64,
        )
        setup_phase_table = _xutils.PhenomX_phase_coeff_table.to(
            device="cpu", dtype=torch.float64
        )
        setup_amp_table = _xutils.PhenomX_amp_coeff_table.to(
            device="cpu", dtype=torch.float64
        )

    with torch_context(setup):
        transitions = _transition_frequencies(state)
        phase = _partial_phase(
            setup,
            state,
            transitions,
            setup_intrinsic,
            setup_phase_table,
            reference_frequency,
            coa_phase,
        )
        amplitude = _build_amplitude(
            setup,
            state,
            transitions,
            phase,
            setup_intrinsic,
            setup_phase_table,
            setup_amp_table,
        )
        phase = _complete_phase(
            setup,
            state,
            phase,
            amplitude,
            setup_intrinsic,
            setup_phase_table,
            setup_amp_table,
        )

    if setup is not frequencies:
        _move_coefficient_tensors(phase, frequencies)
        _move_coefficient_tensors(amplitude, frequencies)

    with torch_context(frequencies):
        mixed_ringdown = _mixed_ringdown_component(
            mf,
            state,
            phase,
            amplitude,
            intrinsic,
            phase_table,
            amp_table,
        )
        amplitude_values = _evaluate_amplitude(
            mf,
            state,
            transitions,
            amplitude,
            mixed_ringdown,
        )
        phase_values = _evaluate_phase(
            mf,
            state,
            phase,
            mixed_ringdown,
            intrinsic,
            phase_table,
        )
        samples = (
            -state.amp0
            * amplitude_values
            * torch.exp(1j * phase_values)
        )
    return samples.to(core.polarization.dtype)
