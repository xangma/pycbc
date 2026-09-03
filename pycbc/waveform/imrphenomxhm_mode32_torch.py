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
import os
import struct
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass

from pycbc import lal_compat as lal
import torch

from . import imrphenomx_utils_torch as _xutils
from ._torch_jax import torch_context
from .imrphenomxas_torch import (
    Phase,
    PhaseDerivative,
    _IMRPhenomXASCore,
    _IMRPhenomXASPhasePlan,
    _InspiralPhasePlan,
    _IntermediatePhasePlan,
    _MergerRingdownAmpPlan,
    _MergerRingdownPhasePlan,
    _evaluate_aligned_region,
    _evaluate_mergerringdown_phase,
    _imrphenomxas_phase_plan_type_supported,
    _imrphenomxas_ringdown_amp_plan_type_supported,
    _prepare_amp_fit_rows,
    _prepare_mergerringdown_amp_plan,
    _prepare_phase_plan,
    get_inspiral_phase,
    get_mergerringdown_Amp,
)
from .imrphenomxhm_mode21_torch import (
    _CARRIER_ALIGNMENT_PHASE,
    _CARRIER_REFERENCE_PHASE,
    _CARRIER_RINGDOWN_START_DERIVATIVE,
    _amplitude_release,
    _as_float,
    _carrier_phase_anchor,
    _mode21_state,
    _ringdown_frequency,
    _solve,
    _tensor,
    _value_and_derivative,
)
from .imrphenomxhm_mode33_torch import (
    _inspiral_boundary,
    _qualified_uniform_grid_common,
    _uniform_grid_region_indices,
)
from .torch_switches import _parse_switch


_PI = lal.PI
_FALSE_ZERO = 1.0e-15
_PN_EXPONENTS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 4.0 / 3.0, 5.0 / 3.0, 2.0)
_AMP_PLAN_ENV = "PYCBC_IMRPHENOMXHM_MODE32_AMP_PLAN"
_DERIVATIVE_GRAPH_ENV = "PYCBC_IMRPHENOMXHM_MODE32_DERIVATIVE_GRAPH"
_DERIVATIVE_REGION_SPECIALIZATION_ENV = (
    "PYCBC_IMRPHENOMXHM_MODE32_DERIVATIVE_REGION_SPECIALIZATION"
)
_ANALYTIC_PHASE_DERIVATIVES_ENV = (
    "PYCBC_IMRPHENOMXHM_MODE32_ANALYTIC_PHASE_DERIVATIVES"
)
_SCRIPTED_ANALYTIC_PHASE_TAIL_ENV = (
    "PYCBC_IMRPHENOMXHM_MODE32_SCRIPTED_ANALYTIC_PHASE_TAIL"
)
_SCRIPTED_ANALYTIC_PHASE_TAIL_EXECUTOR = None
_SCRIPTED_ANALYTIC_PHASE_TAIL_FAILED = False
_SCRIPTED_ANALYTIC_PHASE_TAIL_PID = os.getpid()
_SCRIPTED_ANALYTIC_PHASE_TAIL_LOCK = threading.Lock()
_RINGDOWN_BOUNDARY_REUSE_ENV = (
    "PYCBC_IMRPHENOMXHM_MODE32_RINGDOWN_BOUNDARY_REUSE"
)
_REGION_PRUNING_ENV = "PYCBC_IMRPHENOMXHM_MODE32_REGION_PRUNING"
_SCRIPTED_BOUNDARY_LANE_ENV = (
    "PYCBC_IMRPHENOMXHM_MODE32_SCRIPTED_BOUNDARY_LANE"
)
_SCRIPTED_BOUNDARY_LANE_EXECUTOR = None
_SCRIPTED_BOUNDARY_LANE_FAILED = False
_SCRIPTED_BOUNDARY_LANE_LOCK = threading.Lock()
_SCRIPTED_MIXED_BOUNDARY_LANE_ENV = (
    "PYCBC_IMRPHENOMXHM_MODE32_SCRIPTED_MIXED_BOUNDARY_LANE"
)
_NATIVE_CPU_BOUNDARY_ENV = (
    "PYCBC_IMRPHENOMXHM_MODE32_NATIVE_CPU_BOUNDARY"
)
_SCRIPTED_MIXED_BOUNDARY_LANE_EXECUTOR = None
_SCRIPTED_MIXED_BOUNDARY_LANE_FAILED = False
_SCRIPTED_MIXED_BOUNDARY_LANE_PID = os.getpid()
_SCRIPTED_MIXED_BOUNDARY_LANE_LOCK = threading.Lock()
_CUDA_GRAPH_MIXED_BOUNDARY_LANE_ENV = (
    "PYCBC_IMRPHENOMXHM_MODE32_CUDA_GRAPH_MIXED_BOUNDARY_LANE"
)
# A graph takes about 112 ms to capture and has retained 2--4 MiB of CUDA
# allocator reserve in qualification.  Exact physical states must therefore
# recur before this path pays off, and the process may retain only one by
# default.  Keep this a constant like the other waveform graph caches so tests
# can qualify a different bound without adding another public switch.
_CUDA_GRAPH_MIXED_BOUNDARY_LANE_MAX_ENTRIES = 1
_CUDA_GRAPH_MIXED_BOUNDARY_LANE_CACHE = OrderedDict()
_CUDA_GRAPH_MIXED_BOUNDARY_LANE_FAILURES = OrderedDict()
_CUDA_GRAPH_MIXED_BOUNDARY_LANE_PID = os.getpid()
_CUDA_GRAPH_MIXED_BOUNDARY_LANE_LOCK = threading.Lock()
_REGION_PRUNING_ALIGNMENT = 64
_REGION_PRUNING_MIN_SAMPLES = 256
_DERIVATIVE_GRAPH_STATE_FIELDS = (
    "total_mass_seconds",
    "final_spin",
    "f_ring_32",
    "f_damp_32",
    "mixing_322",
    "mixing_323",
)
_DERIVATIVE_GRAPH_PHASE_FIELDS = (
    "linb",
    "phiref22",
    "alpha0_s",
    "alpha_l_s",
    "alpha2_s",
    "alpha4_s",
    "phi0_s",
)
_DERIVATIVE_GRAPH_AMPLITUDE_FIELDS = (
    "amp_norm",
    "rd_alambda",
    "rd_lambda",
    "rd_sigma",
    "f_rd_aux",
    "f_falloff",
    "tail_amplitude",
    "tail_decay",
    "rd_aux_coefficients",
)
_DERIVATIVE_GRAPH_PHASE_PLAN_FIELDS = (
    "total_mass_seconds",
    "eta",
    "f1_Ms",
    "f2_Ms",
)
_SCRIPTED_ANALYTIC_PHASE_TAIL_HOST_REAL_INDICES = (
    0,
    1,
    2,
    3,
    13,
    14,
    15,
    16,
    17,
    18,
    26,
    27,
    31,
    34,
)
_SCRIPTED_ANALYTIC_PHASE_TAIL_HOST_COMPLEX_INDICES = (4, 5)
_SCRIPTED_ANALYTIC_PHASE_TAIL_HOST_INDICES = frozenset(
    _SCRIPTED_ANALYTIC_PHASE_TAIL_HOST_REAL_INDICES
    + _SCRIPTED_ANALYTIC_PHASE_TAIL_HOST_COMPLEX_INDICES
)
_SCRIPTED_ANALYTIC_PHASE_TAIL_TENSOR_INDICES = tuple(
    index
    for index in range(68)
    if index not in _SCRIPTED_ANALYTIC_PHASE_TAIL_HOST_INDICES
)
_SCRIPTED_ANALYTIC_PHASE_TAIL_SCALAR_SHAPE = torch.Size(())
_SCRIPTED_ANALYTIC_PHASE_TAIL_AUXILIARY_SHAPE = torch.Size((4,))
_SCRIPTED_ANALYTIC_PHASE_TAIL_FIELDS = (
    "c0",
    "c_l",
    "c1",
    "c2",
    "c3",
    "c4",
    "c1_insp",
    "c_insp",
    "c1_rd",
    "c_rd",
    "delta_phi",
)
_DERIVATIVE_GRAPH_CACHE = {}
_DERIVATIVE_GRAPH_FAILURES = set()
_DERIVATIVE_GRAPH_LOCK = threading.Lock()


def _amp_plan_enabled():
    """Return the strict mode-32 amplitude-plan switch."""

    value = os.environ.get(_AMP_PLAN_ENV)
    return True if value is None else _parse_switch(_AMP_PLAN_ENV, value)


def _derivative_graph_enabled():
    """Return the strict, off-by-default mode-32 derivative-graph switch."""

    value = os.environ.get(_DERIVATIVE_GRAPH_ENV)
    return False if value is None else _parse_switch(_DERIVATIVE_GRAPH_ENV, value)


def _derivative_region_specialization_enabled():
    """Return the strict scalar-region specialization switch."""

    value = os.environ.get(_DERIVATIVE_REGION_SPECIALIZATION_ENV)
    return (
        True
        if value is None
        else _parse_switch(_DERIVATIVE_REGION_SPECIALIZATION_ENV, value)
    )


def _analytic_phase_derivatives_enabled():
    """Return the strict analytic-derivative switch."""

    value = os.environ.get(_ANALYTIC_PHASE_DERIVATIVES_ENV)
    return (
        True
        if value is None
        else _parse_switch(_ANALYTIC_PHASE_DERIVATIVES_ENV, value)
    )


def _scripted_analytic_phase_tail_enabled():
    """Return the strict analytic-tail switch."""

    value = os.environ.get(_SCRIPTED_ANALYTIC_PHASE_TAIL_ENV)
    return (
        True
        if value is None
        else _parse_switch(_SCRIPTED_ANALYTIC_PHASE_TAIL_ENV, value)
    )


def _ringdown_boundary_reuse_enabled():
    """Return the strict boundary-result reuse switch."""

    value = os.environ.get(_RINGDOWN_BOUNDARY_REUSE_ENV)
    return (
        True
        if value is None
        else _parse_switch(_RINGDOWN_BOUNDARY_REUSE_ENV, value)
    )


def _region_pruning_enabled():
    """Return the strict mode-32 piecewise switch."""

    value = os.environ.get(_REGION_PRUNING_ENV)
    return True if value is None else _parse_switch(_REGION_PRUNING_ENV, value)


def _ringdown_boundary_runtime_boolean(function, *args):
    """Call a Torch runtime predicate, failing closed on uncertainty."""

    if function is None:
        return None
    try:
        return bool(function(*args))
    except Exception:
        return None


def _ringdown_boundary_reuse_runtime_supported(mf):
    """Reject observable transforms while permitting ordinary inference."""

    if _ringdown_boundary_runtime_boolean(
        getattr(torch.jit, "is_scripting", None)
    ) is not False:
        return False
    if _ringdown_boundary_runtime_boolean(
        getattr(torch.jit, "is_tracing", None)
    ) is not False:
        return False
    tracing_state = getattr(getattr(torch, "_C", None), "_get_tracing_state", None)
    if tracing_state is None:
        return False
    try:
        if tracing_state() is not None:
            return False
    except Exception:
        return False

    compiler = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if _ringdown_boundary_runtime_boolean(compiler) is not False:
        return False
    dynamo = getattr(getattr(torch, "_dynamo", None), "is_compiling", None)
    if _ringdown_boundary_runtime_boolean(dynamo) is not False:
        return False

    if getattr(torch.autograd.forward_ad, "_current_level", None) != -1:
        return False
    functorch = getattr(getattr(torch, "_C", None), "_functorch", None)
    dynamic_depth = getattr(functorch, "get_dynamic_layer_stack_depth", None)
    if dynamic_depth is None:
        return False
    try:
        if dynamic_depth() != 0:
            return False
    except Exception:
        return False

    torch_c = getattr(torch, "_C", None)
    for name in ("_len_torch_dispatch_stack", "_len_torch_function_stack"):
        stack_length = getattr(torch_c, name, None)
        if stack_length is None:
            return False
        try:
            if stack_length() != 0:
                return False
        except Exception:
            return False

    autocast_enabled = getattr(torch, "is_autocast_enabled", None)
    if autocast_enabled is None:
        return False
    try:
        if autocast_enabled("cpu") or autocast_enabled("cuda"):
            return False
    except (RuntimeError, TypeError):
        legacy_cpu = getattr(torch, "is_autocast_cpu_enabled", None)
        if (
            _ringdown_boundary_runtime_boolean(autocast_enabled) is not False
            or _ringdown_boundary_runtime_boolean(legacy_cpu) is not False
        ):
            return False
    except Exception:
        return False

    if mf.device.type == "cuda":
        capture = getattr(torch.cuda, "is_current_stream_capturing", None)
        if _ringdown_boundary_runtime_boolean(capture) is not False:
            return False
    return True


def _derivative_region_specialization_runtime_supported(mf):
    """Accept only an ordinary runtime for the exact derivative executors."""

    if mf.device.type == "cuda":
        bad_fork = getattr(torch.cuda, "_is_in_bad_fork", None)
        if _ringdown_boundary_runtime_boolean(bad_fork) is not False:
            return False
    return _ringdown_boundary_reuse_runtime_supported(mf)


def _amp_plan_inputs_have_autograd(intrinsic, amp_table, final_spin, deviations):
    """Return whether preparing a reusable amplitude plan could capture AD state."""

    values = [intrinsic, amp_table, final_spin]
    if deviations is not None:
        values.append(deviations.strength)
        values.extend(vars(deviations.fits).values())
    return _xutils._tree_has_autograd(values)


def _carrier_amp_plan_supported(plan, like, deviations=None):
    """Accept only an exact same-device scalar carrier amplitude plan."""

    return (
        deviations is None
        and _imrphenomxas_ringdown_amp_plan_type_supported(plan)
        and type(like) is torch.Tensor
        and like.layout == torch.strided
        and like.dtype in (torch.float32, torch.float64)
        and like.device.type in ("cpu", "cuda")
        and all(
            type(value) is torch.Tensor
            and value.layout == torch.strided
            and value.ndim == 0
            and value.dtype == like.dtype
            and value.device == like.device
            and not value.is_conj()
            and not value.is_neg()
            for value in plan
        )
        and not _xutils._tree_has_autograd(plan)
    )


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


@dataclass(frozen=True)
class _Mode32RegionPlan:
    """Validated active-region indices on the standard uniform grid."""

    amplitude: tuple[int, int]
    phase: tuple[int, int]
    ringdown_start: int


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
    carrier_coprecessing_deviations: object | None = None
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
    release: int
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
    phase_boundary_component: torch.Tensor | None = None


@dataclass(frozen=True)
class _DerivativeGraphState:
    total_mass_seconds: torch.Tensor
    final_spin: torch.Tensor
    f_ring_32: torch.Tensor
    f_damp_32: torch.Tensor
    mixing_322: torch.Tensor
    mixing_323: torch.Tensor


@dataclass(frozen=True)
class _DerivativeGraphPhase:
    linb: torch.Tensor
    phiref22: torch.Tensor
    alpha0_s: torch.Tensor
    alpha_l_s: torch.Tensor
    alpha2_s: torch.Tensor
    alpha4_s: torch.Tensor
    phi0_s: torch.Tensor
    carrier_coprecessing_deviations: object | None = None


@dataclass(frozen=True)
class _DerivativeGraphAmplitude:
    release: int
    amp_norm: torch.Tensor
    rd_alambda: torch.Tensor
    rd_lambda: torch.Tensor
    rd_sigma: torch.Tensor
    f_rd_aux: torch.Tensor
    f_falloff: torch.Tensor
    tail_amplitude: torch.Tensor
    tail_decay: torch.Tensor
    rd_aux_coefficients: torch.Tensor


def _chi_pn_hat(state):
    return state.chi_pn_hat


def _xhm32_delta_t(state):
    _, _, psi4_to_strain = _xutils.calc_phaseatpeak(
        state.eta, state.s_tot_r, state.dchi, state.delta
    )
    return -2.0 * _PI * (500.0 + _as_float(psi4_to_strain))


def _check_final_spin(final_spin):
    spin = _as_float(final_spin)
    if abs(spin) > 1.0:
        raise ValueError("XHM QNM fits require |final_spin| <= 1.")
    return final_spin


def qnm_fring32_fit(final_spin):
    a = _check_final_spin(final_spin)
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    return (
        0.09540436245212061
        - 0.13628306966373951 * a
        + 0.030099881830507727 * x2
        - 0.000673589757007597 * x3
        + 0.0118277880067919 * x4
        + 0.0020533816327907334 * x5
        - 0.0015206141948469621 * x6
    ) / (
        1.0
        - 1.6531854335715193 * a
        + 0.5634705514193629 * x2
        + 0.12256204148002939 * x4
        - 0.027297817699401976 * x6
    )


def qnm_fdamp32_fit(final_spin):
    a = _check_final_spin(final_spin)
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    return (
        0.014754148319335946
        - 0.03445752346074498 * a
        + 0.02168855041940869 * x2
        + 0.0014945908223317514 * x3
        - 0.0034761714223258693 * x4
    ) / (
        1.0 - 2.320722660848874 * a + 1.5096146036915865 * x2 - 0.18791187563554512 * x4
    )


def evaluate_qnmfit_re_l3m2lp2(final_spin):
    a = _check_final_spin(final_spin)
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    return float(
        a
        * (
            0.47513455283841244
            - 0.9016636384605536 * a
            + 0.3844811236426182 * x2
            + 0.0855565148647794 * x3
            - 0.03620067426672167 * x4
            - 0.006557249133752502 * x5
        )
        / (-6.76894063440646 + 15.170831931186493 * a - 9.406169787571082 * x2 + x4)
    )


def evaluate_qnmfit_im_l3m2lp2(final_spin):
    a = _check_final_spin(final_spin)
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    return float(
        a
        * (
            -2.8704762147145533
            + 4.436434016918535 * a
            - 1.0115343326360486 * x2
            - 0.08965314412106505 * x3
            - 0.4236810894599512 * x4
            - 0.041787576033810676 * x5
        )
        / (
            -171.80908957903395
            + 272.362882450877 * a
            - 76.68544453077854 * x2
            - 25.14197656531123 * x4
            + x6
        )
    )


def evaluate_qnmfit_re_l3m2lp3(final_spin):
    a = _check_final_spin(final_spin)
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    return float(
        (
            1.0
            - 2.107852425643677 * a
            + 1.1906393634562715 * x2
            + 0.02244848864087732 * x3
            - 0.09593447799423722 * x4
            - 0.0021343381708933025 * x5
            - 0.005319515989331159 * x6
        )
        / (
            1.0
            - 2.1078515887706324 * a
            + 1.2043484690080966 * x2
            - 0.08910191596778137 * x4
            - 0.005471749827809503 * x6
        )
    )


def evaluate_qnmfit_im_l3m2lp3(final_spin):
    a = _check_final_spin(final_spin)
    x2 = a * a
    x3 = x2 * a
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    return float(
        a
        * (
            12.45701482868677
            - 29.398484595717147 * a
            + 18.26221675782779 * x2
            + 1.9308599142669403 * x3
            - 3.159763242921214 * x4
            - 0.0910871567367674 * x5
        )
        / (
            345.52914639836257
            - 815.4349339779621 * a
            + 538.3888932415709 * x2
            - 69.3840921447381 * x4
            + x6
        )
    )


def xhm32_inspiral_phase_lambda_fit(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    s = state.s_tot_r
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    no_spin = (
        9.913819875501506
        + 18.424900617803107 * eta
        - 574.8672384388947 * eta2
        + 2671.7813055097877 * eta3
        - 6244.001932443913 * eta4
    ) / (1.0 - 0.9103118343073325 * eta)
    eq_spin = (
        (
            -4.367632806613781
            + 245.06757304950986 * eta
            - 2233.9319708029775 * eta2
            + 5894.355429022858 * eta3
        )
        * s
        + (
            -1.375112297530783
            - 1876.760129419146 * eta
            + 17608.172965575013 * eta2
            - 40928.07304790013 * eta3
        )
        * s2
        + (
            -1.28324755577382
            - 138.36970336658558 * eta
            + 708.1455154504333 * eta2
            - 273.23750933544176 * eta3
        )
        * s3
        + (
            1.8403161863444328
            + 2009.7361967331492 * eta
            - 18636.271414571278 * eta2
            + 42379.205045791656 * eta3
        )
        * s4
    )
    uneq_spin = (
        state.dchi
        * state.delta
        * eta2
        * (
            -105.34550407768225
            - 1566.1242344157668 * state.chi1z * eta
            + 1566.1242344157668 * state.chi2z * eta
            + 2155.472229664981 * eta * s
        )
    )
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
    p1 = (
        4414.11
        + 4.21564 / eta
        - 10687.8 * eta
        + 58234.6 * eta2
        - 64068.4 * eta3
        - 704442.0 * eta4
        + 2863930.0 * eta5
        - 3263620.0 * eta6
        + (
            (6.39833 - 610.267 * eta + 2095.72 * eta2 - 3970.89 * eta3) * s
            + (22.956700000000005 - 99.1551 * eta + 331.593 * eta2 - 794.79 * eta3) * s2
            + (10.4333 + 43.8812 * eta - 541.261 * eta2 + 294.289 * eta3) * s3
            + eta * (106.047 - 1569.0299999999997 * eta + 4810.61 * eta2) * s4
        )
        / eta
        + 132.244
        * root
        * eta
        * (chi1 * (6.227738120444028 - eta) + chi2 * (-6.227738120444028 + eta))
    )
    p2 = (
        3980.7
        + 0.956703 / eta
        - 6202.38 * eta
        + 29218.1 * eta2
        + 24484.2 * eta3
        - 807629.0 * eta4
        + 2863930.0 * eta5
        - 3263620.0 * eta6
        + (
            (1.92692 - 226.825 * eta + 75.246 * eta2 + 1291.56 * eta3) * s
            + (15.328700000000001 - 99.1551 * eta + 608.328 * eta2 - 2402.94 * eta3)
            * s2
            + (10.4333 + 43.8812 * eta - 541.261 * eta2 + 294.289 * eta3) * s3
            + eta * (106.047 - 1569.0299999999997 * eta + 4810.61 * eta2) * s4
        )
        / eta
        + 132.244
        * root
        * eta
        * (chi1 * (2.5769789177580837 - eta) + chi2 * (-2.5769789177580837 + eta))
    )
    p3 = (
        3416.57
        + 2308.63 * eta
        - 84042.9 * eta2
        + 1019360.0 * eta3
        - 6064400.0 * eta4
        + 17639900.0 * eta5
        - 20065000.0 * eta6
        + (24.6295 - 282.354 * eta - 2582.55 * eta2 + 12750.0 * eta3) * s
        + (433.675 - 8775.86 * eta + 56407.8 * eta2 - 114798.0 * eta3) * s2
        + (559.705 - 10627.4 * eta + 61581.0 * eta2 - 114029.0 * eta3) * s3
        + (106.047 - 1569.03 * eta + 4810.61 * eta2) * s4
        + 63.9466 * dchi * root * eta2
    )
    p4 = (
        3307.49
        - 476.909 * eta
        - 5980.37 * eta2
        + 127610.0 * eta3
        - 919108.0 * eta4
        + 2863930.0 * eta5
        - 3263620.0 * eta6
        + (-5.02553 - 282.354 * eta + 1291.56 * eta2) * s
        + (-43.8823 + 740.123 * eta - 2402.94 * eta2) * s2
        + (43.8812 - 370.362 * eta + 294.289 * eta2) * s3
        + (106.047 - 1569.03 * eta + 4810.61 * eta2) * s4
        - 132.244 * dchi * root * eta2
    )
    p56 = (
        3259.03
        - 3967.58 * eta
        + 111203.0 * eta2
        - 1818830.0 * eta3
        + 17381100.0 * eta4
        - 95698800.0 * eta5
        + 275056000.0 * eta6
        - 315866000.0 * eta7
        + (19.7509 - 1104.53 * eta + 3810.18 * eta2) * s
        + (-230.07 + 2314.51 * eta - 5944.49 * eta2) * s2
        + (-201.633 + 2183.43 * eta - 6233.99 * eta2) * s3
        + (106.047 - 1569.03 * eta + 4810.61 * eta2) * s4
        + 112.714 * dchi * root * eta2
    )
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
    iv1 = (
        (
            cd
            * delta
            * (
                -0.739317114582042 * eta
                - 47.473246070362634 * eta2
                + 278.9717709112207 * eta3
                - 566.6420939162068 * eta4
            )
            + cd2
            * (
                -0.5873680378268906 * eta
                + 6.692187014925888 * eta2
                - 24.37776782232888 * eta3
                + 23.783684827838247 * eta4
            )
        )
        * root
        + (
            3.2940434453819694
            + 4.94285331708559 * eta
            - 343.3143244815765 * eta2
            + 3585.9269057886418 * eta3
            - 19279.186145681153 * eta4
            + 51904.91007211022 * eta5
            - 55436.68857586653 * eta6
        )
        * root
        + cd
        * delta
        * (
            12.488240781993923 * eta
            - 209.32038774208385 * eta2
            + 1160.9833883184604 * eta3
            - 2069.5349737049073 * eta4
        )
        * s
        * root
        + s
        * (
            0.6343034651912586
            * (
                -2.5844888818001737
                + 78.98200041834092 * eta
                - 1087.6241783616488 * eta2
                + 7616.234910399297 * eta3
                - 24776.529123239357 * eta4
                + 30602.210950069973 * eta5
            )
            - 0.062088720220899465
            * (
                6.5586380356588565
                + 36.01386705325694 * eta
                - 3124.4712274775407 * eta2
                + 33822.437731298516 * eta3
                - 138572.93700180828 * eta4
                + 198366.10615196894 * eta5
            )
            * s
        )
        * root
    )
    iv2 = (
        (
            cd2
            * (
                -0.03940151060321499 * eta
                + 1.9034209537174116 * eta2
                - 8.78587250202154 * eta3
            )
            + cd
            * delta
            * (
                -1.704299788495861 * eta
                - 4.923510922214181 * eta2
                + 0.36790005839460627 * eta3
            )
        )
        * root
        + (
            2.2911849711339123
            - 5.1846950040514335 * eta
            + 60.10368251688146 * eta2
            - 1139.110227749627 * eta3
            + 7970.929280907627 * eta4
            - 25472.73682092519 * eta5
            + 30950.67053883646 * eta6
        )
        * root
        + s
        * (
            0.7718201508695763
            * (
                -1.3012906461000349
                + 26.432880113146012 * eta
                - 186.5001124789369 * eta2
                + 712.9101229418721 * eta3
                - 970.2126139442341 * eta4
            )
            + 0.04832734931068797
            * (
                -5.9999628512498315
                + 78.98681284391004 * eta
                + 1.8360177574514709 * eta2
                - 2537.636347529708 * eta3
                + 6858.003573909322 * eta4
            )
            * s
        )
        * root
    )
    iv3 = (
        (
            cd2
            * (
                -0.6358511175987503 * eta
                + 5.555088747533164 * eta2
                - 14.078156877577733 * eta3
            )
            + cd
            * delta
            * (
                0.23205448591711159 * eta
                - 19.46049432345157 * eta2
                + 36.20685853857613 * eta3
            )
        )
        * root
        + (
            1.1525594672495008
            + 7.380126197972549 * eta
            - 17.51265776660515 * eta2
            - 976.9940395257111 * eta3
            + 8880.536804741967 * eta4
            - 30849.228936891763 * eta5
            + 38785.53683146884 * eta6
        )
        * root
        + cd
        * delta
        * (
            1.904350804857431 * eta
            - 25.565242391371093 * eta2
            + 80.67120303906654 * eta3
        )
        * s
        * root
        + s
        * (
            0.785171689871352
            * (
                -0.4634745514643032
                + 18.70856733065619 * eta
                - 167.9231114864569 * eta2
                + 744.7699462372949 * eta3
                - 1115.008825153004 * eta4
            )
            + 0.13469300326662165
            * (
                -2.7311391326835133
                + 72.17373498208947 * eta
                - 483.7040402103785 * eta2
                + 1136.8367114738041 * eta3
                - 472.02962341590774 * eta4
            )
            * s
        )
        * root
    )
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
    aux1 = (
        cd2
        * (
            -4.188795724777721 * eta2
            + 53.39200466700963 * eta3
            - 131.19660856923554 * eta4
        )
        + cd
        * delta
        * (
            14.284921364132623 * eta2
            - 321.26423637658746 * eta3
            + 1242.865584938088 * eta4
        )
        + s
        * (
            -0.022968727462555794
            * (
                83.66854837403105 * eta
                - 3330.6261333413177 * eta2
                + 77424.12614733395 * eta3
                - 710313.3016672594 * eta4
                + 2693491.7075009225 * eta5
                - 3572465.179268999 * eta6
            )
            + 0.0014795114305436387
            * (
                -1672.7273629876313 * eta
                + 90877.38260964208 * eta2
                - 1669016.9155105734 * eta3
                + 13705532.554135624 * eta4
                - 51161109.98398143 * eta5
                + 70606676.6311127 * eta6
            )
            * s
        )
        + (
            4.45156488896258 * eta
            - 77.39303992494544 * eta2
            + 522.5070635563092 * eta3
            - 1642.3057499049708 * eta4
            + 2048.333892310575 * eta5
        )
        / (1.0 - 9.611489164758915 * eta + 24.249594730050312 * eta2)
    )
    spn = _chi_pn_hat(state)
    aux2 = (
        cd2
        * (
            -18.550171209458394 * eta2
            + 188.99161055445936 * eta3
            - 440.26516625611 * eta4
        )
        + cd
        * delta
        * (
            13.132625215315063 * eta2
            - 340.5204040505528 * eta3
            + 1327.1224176812448 * eta4
        )
        + spn
        * (
            -0.16707403272774676
            * (
                6.678916447469937 * eta
                + 1331.480396625797 * eta2
                - 41908.45179140144 * eta3
                + 520786.0225074669 * eta4
                - 3189462.4909922685 * eta5
                + 9515538.23212259 * eta6
                - 11006903.622406831 * eta7
            )
            + 0.015205286051218441
            * (
                108.10032279461095 * eta
                - 16084.215590200103 * eta2
                + 462957.5593513407 * eta3
                - 5635028.227588545 * eta4
                + 33799252.77713386 * eta5
                - 98658152.75452062 * eta6
                + 112013079.79786257 * eta7
            )
            * spn
        )
        + (
            3.902154247490771 * eta
            - 55.77521071924907 * eta2
            + 294.9496843041973 * eta3
            - 693.6803787318279 * eta4
            + 636.0141528226893 * eta5
        )
        / (1.0 - 8.56699762573719 * eta + 19.119341007236955 * eta2)
    )
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
    int1 = (
        (
            cd2
            * (
                -0.2341404256829785 * eta
                + 2.606326837996192 * eta2
                - 8.68296921440857 * eta3
            )
            + cd
            * delta
            * (
                0.5454562486736877 * eta
                - 25.19759222940851 * eta2
                + 73.40268975811729 * eta3
            )
        )
        * root
        + cd
        * delta
        * (
            0.4422257616009941 * eta
            - 8.490112284851655 * eta2
            + 32.22238925527844 * eta3
        )
        * spn
        * root
        + spn
        * (
            0.7067243321652764
            * (
                0.12885110296881636
                + 9.608999847549535 * eta
                - 85.46581740280585 * eta2
                + 325.71940024255775 * eta3
                + 175.4194342269804 * eta4
                - 1929.9084724384807 * eta5
            )
            + 0.1540566313813899
            * (
                -0.3261041495083288
                + 45.55785402900492 * eta
                - 827.591235943271 * eta2
                + 7184.647314370326 * eta3
                - 28804.241518798244 * eta4
                + 43309.69769878964 * eta5
            )
            * spn
        )
        * root
        + (
            480.0434256230109 * eta
            + 25346.341240810478 * eta2
            - 99873.4707358776 * eta3
            + 106683.98302194536 * eta4
        )
        * root
        / (1.0 + 1082.6574834474493 * eta + 10083.297670051445 * eta2)
    )
    int2 = (
        eta
        * (
            cd2
            * (
                -4.175680729484314 * eta
                + 47.54281549129226 * eta2
                - 128.88334273588077 * eta3
            )
            + cd
            * delta
            * (
                -0.18274358639599947 * eta
                - 71.01128541687838 * eta2
                + 208.07105580635888 * eta3
            )
        )
        + eta
        * (
            4.760999387359598
            - 38.57900689641654 * eta
            + 456.2188780552874 * eta2
            - 4544.076411013166 * eta3
            + 24956.9592553473 * eta4
            - 69430.10468748478 * eta5
            + 77839.74180254337 * eta6
        )
        + cd
        * delta
        * eta
        * (
            1.2198776533959694 * eta
            - 26.816651899746475 * eta2
            + 68.72798751937934 * eta3
        )
        * st
        + eta
        * st
        * (
            1.5098291294292217
            * (
                0.4844667556328104
                + 9.848766999273414 * eta
                - 143.66427232396376 * eta2
                + 856.9917885742416 * eta3
                - 1633.3295758142904 * eta4
            )
            + 0.32413108737204144
            * (
                2.835358206961064
                - 62.37317183581803 * eta
                + 761.6103793011912 * eta2
                - 3811.5047139343505 * eta3
                + 6660.304740652403 * eta4
            )
            * st
        )
    )
    int3 = (
        3.881450518842405 * eta
        - 12.580316392558837 * eta2
        + 1.7262466525848588 * eta3
        + cd2
        * (
            -7.065118823041031 * eta2
            + 77.97950589523865 * eta3
            - 203.65975422378446 * eta4
        )
        - 58.408542930248046 * eta4
        + cd
        * delta
        * (
            1.924723094787216 * eta2
            - 90.92716917757797 * eta3
            + 387.00162600306226 * eta4
        )
        + 403.5748987560612 * eta5
        + cd
        * delta
        * (
            -0.2566958540737833 * eta2
            + 14.488550203412675 * eta3
            - 26.46699529970884 * eta4
        )
        * spn
        + spn
        * (
            0.3650871458400108
            * (
                71.57390929624825 * eta2
                - 994.5272351916166 * eta3
                + 6734.058809060536 * eta4
                - 18580.859291282686 * eta5
                + 16001.318492586077 * eta6
            )
            + 0.0960146077440495
            * (
                451.74917589707513 * eta2
                - 9719.470997418284 * eta3
                + 83403.5743434538 * eta4
                - 318877.43061174755 * eta5
                + 451546.88775684836 * eta6
            )
            * spn
            - 0.03985156529181297
            * (
                -304.92981902871617 * eta2
                + 3614.518459296278 * eta3
                - 7859.4784979916085 * eta4
                - 46454.57664737511 * eta5
                + 162398.81483375572 * eta6
            )
            * spn
            * spn
        )
    )
    int4 = (
        eta
        * (
            cd2
            * (
                -8.572797326909152 * eta
                + 92.95723645687826 * eta2
                - 236.2438921965621 * eta3
            )
            + cd
            * delta
            * (
                6.674358856924571 * eta
                - 171.4826985994883 * eta2
                + 645.2760206304703 * eta3
            )
        )
        + eta
        * (
            3.921660532875504
            - 16.57299637423352 * eta
            + 25.254017911686333 * eta2
            - 143.41033155133266 * eta3
            + 692.926425981414 * eta4
        )
        + cd
        * delta
        * eta
        * (
            -3.582040878719185 * eta
            + 57.75888914133383 * eta2
            - 144.21651114700492 * eta3
        )
        * st
        + eta
        * st
        * (
            1.242750265695504
            * (
                -0.522172424518215
                + 25.168480118950065 * eta
                - 303.5223688400309 * eta2
                + 1858.1518762309654 * eta3
                - 3797.3561904195085 * eta4
            )
            + 0.2927045241764365
            * (
                0.5056957789079993
                - 15.488754837330958 * eta
                + 471.64047356915603 * eta2
                - 3131.5783196211587 * eta3
                + 6097.887891566872 * eta4
            )
            * st
        )
    )
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
    no_spin = (
        11.851438981981772
        + 167.95086712701223 * eta
        - 4565.033758777737 * eta2
        + 61559.132976189896 * eta3
        - 364129.24735853914 * eta4
        + 739270.8814129328 * eta5
    )
    eq_spin = (
        (
            9.506768471271634
            + 434.31707030999445 * eta
            - 8046.364492927503 * eta2
            + 26929.677144312944 * eta3
        )
        * s
        + (
            -5.949655484033632
            - 307.67253970367034 * eta
            + 1334.1062451631644 * eta2
            + 3575.347142399199 * eta3
        )
        * s2
        + (
            3.4881615575084797
            - 2244.4613237912527 * eta
            + 24145.932943269272 * eta2
            - 60929.87465551446 * eta3
        )
        * s3
        + (
            15.585154698977842
            - 2292.778112523392 * eta
            + 24793.809334683185 * eta2
            - 65993.84497923202 * eta3
        )
        * s4
    )
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
    no_spin = (
        -1.3328895897490733
        - 22.209549522908667 * eta
        + 1056.2426481245027 * eta2
        - 21256.376324666326 * eta3
        + 246313.12887984765 * eta4
        - 1631296.8467540336 * eta5
        + 5614617.173188322 * eta6
        - 7612233.821752137 * eta7
    )
    eq_spin = (
        s
        * (
            -1.622727240110213
            + 0.9960210841611344 * s
            - 1.1239505323267036 * s2
            - 1.9586085340429995 * s3
            + eta2
            * (
                196.7055281997748
                + 135.25216875394943 * s
                + 1086.7504825459278 * s2
                + 546.6246807461155 * s3
                - 312.1010566468068 * s4
            )
            + 0.7638287749489343 * s4
            + eta
            * (
                -47.475568056234245
                - 35.074072557604445 * s
                - 97.16014978329918 * s2
                - 34.498125910065156 * s3
                + 24.02858084544326 * s4
            )
            + eta3
            * (
                62.632493533037625
                - 22.59781899512552 * s
                - 2683.947280170815 * s2
                - 1493.177074873678 * s3
                + 805.0266029288334 * s4
            )
        )
        / denom
    )
    uneq_spin = (
        state.delta
        * (
            state.chi2z * eta**2.5 * (88.56162028006072 - 30.01812659282717 * s)
            + state.chi2z * eta2 * (43.126266433486435 - 14.617728550838805 * s)
            + state.chi1z * eta2 * (-43.126266433486435 + 14.617728550838805 * s)
            + state.chi1z * eta**2.5 * (-88.56162028006072 + 30.01812659282717 * s)
        )
        / denom
    )
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
    p1 = (
        3169.372056189274
        + 426.8372805022653 * eta
        - 12569.748101922158 * eta2
        + 149846.7281073725 * eta3
        - 817182.2896823225 * eta4
        + 1567405.3633767858 * eta5
        + (
            19.23408352151287
            - 1762.6573670619173 * eta
            + 7855.316419853637 * eta2
            - 3785.49764771212 * eta3
        )
        * s
        + (
            -42.88446003698396
            + 336.8340966473415 * eta
            - 5615.908682338113 * eta2
            + 20497.5021807654 * eta3
        )
        * s2
        + (
            13.918237996338371
            + 10145.53174542332 * eta
            - 91664.12621864353 * eta2
            + 201204.5096556517 * eta3
        )
        * s3
        + (
            -24.72321125342808
            - 4901.068176970293 * eta
            + 53893.9479532688 * eta2
            - 139322.02687945773 * eta3
        )
        * s4
        + (
            -61.01931672442576
            - 16556.65370439302 * eta
            + 162941.8009556697 * eta2
            - 384336.57477596396 * eta3
        )
        * s5
        + state.dchi
        * state.delta
        * eta2
        * (
            641.2473192044652
            - 1600.240100295189 * state.chi1z * eta
            + 1600.240100295189 * state.chi2z * eta
            + 13275.623692212472 * eta * s
        )
    )
    p2 = (
        3131.0260952676376
        + 206.09687819102305 * eta
        - 2636.4344627081873 * eta2
        + 7475.062269742079 * eta3
        + (
            49.90874152040307
            - 691.9815135740145 * eta
            - 434.60154548208334 * eta2
            + 10514.68111669422 * eta3
        )
        * s
        + (
            97.3078084654917
            - 3458.2579971189534 * eta
            + 26748.805404989867 * eta2
            - 56142.13736008524 * eta3
        )
        * s2
        + (
            -132.49105074500454
            + 429.0787542102207 * eta
            + 7269.262546204149 * eta2
            - 27654.067482558712 * eta3
        )
        * s3
        + (
            -227.8023564332453
            + 5119.138772157134 * eta
            - 34444.2579678986 * eta2
            + 69666.01833764123 * eta3
        )
        * s4
        + 477.51566939885424 * d * eta2
    )
    p3 = (
        3082.803556599222
        + 76.94679795837645 * eta
        - 586.2469821978381 * eta2
        + 977.6115755788503 * eta3
        + (
            45.08944710349874
            - 807.7353772747749 * eta
            + 1775.4343704616288 * eta2
            + 2472.6476419567534 * eta3
        )
        * s
        + (
            95.57355060136699
            - 2224.9613131172046 * eta
            + 13821.251641893134 * eta2
            - 25583.314298758105 * eta3
        )
        * s2
        + (
            -144.96370424517866
            + 2268.4693587493093 * eta
            - 10971.864789147161 * eta2
            + 16259.911572457446 * eta3
        )
        * s3
        + (
            -227.8023564332453
            + 5119.138772157134 * eta
            - 34444.2579678986 * eta2
            + 69666.01833764123 * eta3
        )
        * s4
        + 378.2359918274837 * d * eta2
    )
    p4 = (
        3077.0657367004565
        + 64.99844502520415 * eta
        - 357.38692756785395 * eta2
        + (
            34.793450080444714
            - 986.7751755509875 * eta
            - 9490.641676924794 * eta3
            + 5700.682624203565 * eta2
        )
        * s
        + (
            57.38106384558743
            - 1644.6690499868596 * eta
            - 19906.416384606226 * eta3
            + 11008.881935880598 * eta2
        )
        * s2
        + (
            -126.02362949830213
            + 3169.3397351803583 * eta
            + 62863.79877094988 * eta3
            - 26766.730897942085 * eta2
        )
        * s3
        + (
            -169.30909412804587
            + 4900.706039920717 * eta
            + 95314.99988114933 * eta3
            - 41414.05689348732 * eta2
        )
        * s4
        + 390.5443469721231 * d * eta2
    )
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
    return float(
        chi2
        * (
            -3.4614418482110163 * eta3
            + 35.464117772624164 * eta4
            - 85.19723511005235 * eta5
        )
        + chi
        * state.delta
        * (
            2.0328561081997463 * eta3
            - 46.18751757691501 * eta4
            + 170.9266105597438 * eta5
        )
        + chi2
        * (
            -0.4600401291210382 * eta3
            + 12.23450117663151 * eta4
            - 42.74689906831975 * eta5
        )
        * s
        + chi
        * state.delta
        * (
            5.786292428422767 * eta3
            - 53.60467819078566 * eta4
            + 117.66195692191727 * eta5
        )
        * s
        + s
        * (
            -0.0013330716557843666
            * (
                56.35538385647113 * eta
                - 1218.1550992423377 * eta2
                + 16509.69605686402 * eta3
                - 102969.88022112886 * eta4
                + 252228.94931931415 * eta5
                - 150504.2927996263 * eta6
            )
            + 0.0010126460331462495
            * (
                -33.87083889060834 * eta
                + 502.6221651850776 * eta2
                - 1304.9210590188136 * eta3
                - 36980.079328277505 * eta4
                + 295469.28617550555 * eta5
                - 597155.7619486618 * eta6
            )
            * s
            - 0.00043088431510840695
            * (
                -30.014415072587354 * eta
                - 1900.5495690280086 * eta2
                + 76517.21042363928 * eta3
                - 870035.1394696251 * eta4
                + 3907267.4134789007 * eta5
                - 6094089.675611567 * eta6
            )
            * s2
        )
        + (
            0.08408469319155859 * eta
            - 1.223794846617597 * eta2
            + 6.5972460654253515 * eta3
            - 15.707327897569396 * eta4
            + 14.163264397061505 * eta5
        )
        / (1.0 - 8.612447115134758 * eta + 18.93655612952139 * eta2)
    )


def xhm32_rd_amp_lambda_fit(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    eta5 = eta4 * eta
    s = state.s_tot_r
    chi = state.chi_a
    chi2 = chi * chi
    return float(
        0.978510781593996
        + 0.36457571743142897 * eta
        - 12.259851752618998 * eta2
        + 49.19719473681921 * eta3
        + chi
        * state.delta
        * (
            -188.37119473865533 * eta3
            + 2151.8731700399308 * eta4
            - 6328.182823770599 * eta5
        )
        + chi2
        * (
            115.3689949926392 * eta3
            - 1159.8596972989067 * eta4
            + 2657.6998831179444 * eta5
        )
        + s
        * (
            0.22358643406992756
            * (
                0.48943645614341924
                - 32.06682257944444 * eta
                + 365.2485484044132 * eta2
                - 915.2489655397206 * eta3
            )
            + 0.0792473022309144
            * (
                1.877251717679991
                - 103.65639889587327 * eta
                + 1202.174780792418 * eta2
                - 3206.340850767219 * eta3
            )
            * s
        )
    )


def xhm32_rd_amp_sigma_fit(state):
    eta = state.eta
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    eta5 = eta4 * eta
    s = state.s_tot_r
    chi = state.chi_a
    chi2 = chi * chi
    return float(
        1.3353917551819414
        + 0.13401718687342024 * eta
        + chi
        * state.delta
        * (
            144.37065005786636 * eta3
            - 754.4085447486738 * eta4
            + 123.86194078913776 * eta5
        )
        + chi2
        * (
            209.09202210427972 * eta3
            - 1769.4658099037918 * eta4
            + 3592.287297392387 * eta5
        )
        + s
        * (
            -0.012086025709597246
            * (
                -6.230497473791485
                + 600.5968613752918 * eta
                - 6606.1009717965735 * eta2
                + 17277.60594350428 * eta3
            )
            - 0.06066548829900489
            * (
                -0.9208054306316676
                + 142.0346574366267 * eta
                - 1567.249168668069 * eta2
                + 4119.373703246675 * eta3
            )
            * s
        )
    )


def _mode32_state(
    params,
    *,
    final_spin=None,
    ringdown_frequency=None,
    damping_frequency=None,
    carrier_ringdown_frequency=None,
    carrier_damping_frequency=None,
    _base_state=None,
    _remnant=None,
):
    base = _base_state
    if base is None:
        base = _mode21_state(
            params,
            final_spin=final_spin,
            _remnant=_remnant,
            carrier_ringdown_frequency=carrier_ringdown_frequency,
            carrier_damping_frequency=carrier_damping_frequency,
        )
    final_mass = 1.0 - base.radiated_energy
    final_spin = base.final_spin
    ringdown_frequency = _ringdown_frequency(
        ringdown_frequency,
        name="mode ringdown frequency",
    )
    damping_frequency = _ringdown_frequency(
        damping_frequency,
        name="mode damping frequency",
    )
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
        f_ring_32=(
            qnm_fring32_fit(final_spin) / final_mass
            if ringdown_frequency is None
            else ringdown_frequency
        ),
        f_damp_32=(
            qnm_fdamp32_fit(final_spin) / final_mass
            if damping_frequency is None
            else damping_frequency
        ),
        mixing_322=mixing_322,
        mixing_323=mixing_323,
    )


def _transition_frequencies(state, amplitude_release=122022):
    amplitude_release = _amplitude_release(amplitude_release)
    if amplitude_release == 122019:
        from .imrphenomxhm_mode32_2019_torch import (
            inspiral_configuration_32_2019,
            ringdown_match_32_2019,
        )

        _, f_amp_in, _, _, _ = inspiral_configuration_32_2019(state)
        f_amp_rd = ringdown_match_32_2019(state)
    else:
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
        f_amp_rd = state.f_ring_22 - 0.5 * state.f_damp_22

    f_end = state.f_ring_22 - 0.5 * state.f_damp_22
    f_phase_in = (1.0 + 0.001 * (0.25 / state.eta - 1.0)) * state.f_meco_32
    f_phase_rd = f_end
    if amplitude_release == 122019 and state.eta < 0.01 and state.chi1 < 0.0:
        f_phase_rd *= 1.2 - 0.25 * state.chi1
    return _Transitions(
        f_amp_in=f_amp_in,
        f_amp_rd=f_amp_rd,
        f_phase_in=f_phase_in,
        f_phase_rd=f_phase_rd,
        phase_intermediate_points=(
            f_phase_in,
            (math.sqrt(3.0) * (f_phase_in - f_end) + 2.0 * (f_phase_in + f_end)) / 4.0,
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


def _prepare_uniform_region_plan(
    core,
    frequencies,
    mf,
    state,
    transitions,
    amplitude_release,
    *,
    uniform_grid_metadata=None,
):
    """Return exact active-region indices for the qualified CPU lane."""

    if (
        amplitude_release != 122022
        or transitions.f_amp_rd != transitions.f_phase_rd
        or not _ringdown_boundary_reuse_runtime_supported(mf)
    ):
        return None
    if uniform_grid_metadata is None and type(core) is _IMRPhenomXASCore:
        uniform_grid_metadata = (
            core.first_bin,
            core.stop_bin,
            core.delta_f,
        )
    common = _qualified_uniform_grid_common(
        frequencies,
        mf,
        state,
        uniform_grid_metadata,
        enabled=_region_pruning_enabled(),
    )
    if common is None:
        return None

    amplitude = _uniform_grid_region_indices(
        *common,
        transitions.f_amp_in,
        transitions.f_amp_rd,
        min_samples=_REGION_PRUNING_MIN_SAMPLES,
    )
    phase = _uniform_grid_region_indices(
        *common,
        transitions.f_phase_in,
        transitions.f_phase_rd,
        min_samples=_REGION_PRUNING_MIN_SAMPLES,
    )
    if amplitude is None or phase is None or amplitude[1] != phase[1]:
        return None
    return _Mode32RegionPlan(
        amplitude=amplitude,
        phase=phase,
        ringdown_start=amplitude[1],
    )


def _region_pruning_value_supported(value, like, *, allow_views):
    """Validate plain request-local leaves used by the pruned evaluators."""

    if type(value) is torch.Tensor:
        return (
            value.layout is torch.strided
            and value.device == like.device
            and value.dtype in (torch.float64, torch.complex128)
            and value.is_contiguous()
            and (allow_views or (value._base is None and value.storage_offset() == 0))
            and not value.is_conj()
            and not value.is_neg()
            and not _xutils._tree_has_autograd_untrusted(value)
        )
    if isinstance(value, torch.Tensor):
        return False
    if type(value) in (type(None), bool, int, float, complex):
        return True
    if isinstance(value, tuple):
        return all(
            _region_pruning_value_supported(item, like, allow_views=allow_views)
            for item in value
        )
    if type(value) is list:
        return all(
            _region_pruning_value_supported(item, like, allow_views=allow_views)
            for item in value
        )
    if type(value) is dict:
        return all(
            _region_pruning_value_supported(item, like, allow_views=allow_views)
            for item in value.values()
        )
    return False


def _region_pruning_values_supported(
    mf,
    phase,
    amplitude,
    intrinsic,
    phase_table,
    amp_table,
    carrier_phase_plan,
    carrier_amp_plan,
    carrier_inspiral_phase,
):
    """Reject differentiated, overridden, or externally viewed inputs."""

    internal_values = (
        tuple(
            value
            for name, value in vars(phase).items()
            if name != "transitions"
        ),
        tuple(vars(amplitude).values()),
        carrier_phase_plan,
        carrier_amp_plan,
    )
    external_values = (
        intrinsic,
        phase_table,
        amp_table,
        carrier_inspiral_phase,
    )
    return _region_pruning_value_supported(
        internal_values,
        mf,
        allow_views=True,
    ) and _region_pruning_value_supported(
        external_values,
        mf,
        allow_views=False,
    )


def _lambda_pn(state):
    if state.eta > 0.01:
        return (
            2376.0 * _PI * (-5.0 + 22.0 * state.eta) / (-3960.0 + 11880.0 * state.eta)
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
        (10471.0 - 61625.0 * eta + 82460.0 * eta2) / 10080.0 * _PI ** (4.0 / 3.0),
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
        + phase.alpha_l_s * torch.atan((frequency - state.f_ring_32) / state.f_damp_32)
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
        + alpha_l_s * torch.atan((frequency - state.f_ring_32) / state.f_damp_32)
    )


def _partial_phase(
    mf,
    state,
    transitions,
    intrinsic,
    phase_table,
    reference_frequency,
    coa_phase,
    carrier_coprecessing_deviations=None,
    carrier_phase_plan=None,
    carrier_phase_anchors=None,
):
    rows = [
        [
            1.0,
            state.f_damp_32 / (state.f_damp_32**2 + (frequency - state.f_ring_32) ** 2),
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
        _carrier_phase_anchor(
            carrier_phase_anchors,
            _CARRIER_RINGDOWN_START_DERIVATIVE,
            mf,
            lambda: PhaseDerivative(
                _tensor(derivative_frequency, mf),
                intrinsic,
                phase_table,
                final_spin=state.final_spin,
                coprecessing_deviations=carrier_coprecessing_deviations,
                _phase_plan=carrier_phase_plan,
            ),
        )
        / state.total_mass_seconds
    )
    linb = linb_fit - dphi22_ref - 2.0 * _PI * (500.0 + psi4_to_strain)

    derivative_match = _tensor(
        state.f_ring_22 + state.f_damp_22,
        mf,
    )
    dphi22_match = (
        PhaseDerivative(
            derivative_match / state.total_mass_seconds,
            intrinsic,
            phase_table,
            final_spin=state.final_spin,
            coprecessing_deviations=carrier_coprecessing_deviations,
            _phase_plan=carrier_phase_plan,
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
        / (state.f_damp_32**2 + (derivative_match - state.f_ring_32) ** 2)
    )
    alpha0_s = (
        alpha0_s
        + dphi22_match
        + xhm32_rd_phase_spheroidal_time_shift_fit(state)
        - raw_derivative
    )

    mf_ref = reference_frequency * state.total_mass_seconds
    phiref22 = (
        -_carrier_phase_anchor(
            carrier_phase_anchors,
            _CARRIER_REFERENCE_PHASE,
            mf,
            lambda: Phase(
                _tensor(reference_frequency, mf),
                intrinsic,
                phase_table,
                final_spin=state.final_spin,
                coprecessing_deviations=carrier_coprecessing_deviations,
                _phase_plan=carrier_phase_plan,
            ),
        )
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
            final_spin=state.final_spin,
            coprecessing_deviations=carrier_coprecessing_deviations,
            _phase_plan=carrier_phase_plan,
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
        carrier_coprecessing_deviations=(carrier_coprecessing_deviations),
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


def _ringdown_auxiliary_amplitude(frequency, amplitude):
    """Evaluate the 2022 auxiliary polynomial in its original operation order."""

    auxiliary = torch.zeros_like(frequency)
    frequency_power = torch.ones_like(frequency)
    for coefficient in amplitude.rd_aux_coefficients:
        auxiliary = auxiliary + coefficient * frequency_power
        frequency_power = frequency_power * frequency
    return auxiliary


def _ringdown_spheroidal_amplitude(frequency, state, amplitude):
    if amplitude.release == 122019:
        offset = frequency - state.f_ring_32
        width = state.f_damp_32 * amplitude.rd_sigma
        rescaled = (
            state.f_damp_32
            * amplitude.rd_alambda
            * amplitude.rd_sigma
            * torch.exp(-offset * amplitude.rd_lambda / width)
            / (offset * offset + width * width)
            * frequency ** (-1.0 / 12.0)
        )
        return amplitude.amp_norm * frequency ** (-7.0 / 6.0) * rescaled

    auxiliary = _ringdown_auxiliary_amplitude(frequency, amplitude)

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
    carrier_phase_plan=None,
    carrier_amp_plan=None,
    *,
    derivative_region_specialized=False,
):
    amp22, _ = get_mergerringdown_Amp(
        frequency,
        intrinsic,
        amp_table,
        _amp_plan=carrier_amp_plan,
        final_spin=state.final_spin,
        coprecessing_deviations=(phase.carrier_coprecessing_deviations),
    )
    carrier_frequency = frequency / state.total_mass_seconds
    if derivative_region_specialized:
        carrier_fMs = carrier_frequency * carrier_phase_plan.total_mass_seconds
        carrier_phase, _ = _evaluate_mergerringdown_phase(
            carrier_fMs,
            carrier_phase_plan.mergerringdown,
        )
        carrier_phase = (
            carrier_phase
            + carrier_phase_plan.beta0
            + carrier_phase_plan.beta1 * carrier_fMs
        )
        carrier_phase = (1 / carrier_phase_plan.eta) * carrier_phase
    else:
        carrier_phase = Phase(
            carrier_frequency,
            intrinsic,
            phase_table,
            final_spin=state.final_spin,
            coprecessing_deviations=(phase.carrier_coprecessing_deviations),
            _phase_plan=carrier_phase_plan,
        )
    phase22 = carrier_phase + phase.linb * frequency + phase.phiref22
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
    carrier_phase_plan=None,
    carrier_amp_plan=None,
    *,
    derivative_region_specialized=False,
):
    h22 = _h22_ringdown_component(
        frequency,
        state,
        phase,
        amplitude,
        intrinsic,
        phase_table,
        amp_table,
        carrier_phase_plan,
        carrier_amp_plan,
        derivative_region_specialized=derivative_region_specialized,
    )
    if derivative_region_specialized:
        spheroidal_amplitude = _ringdown_auxiliary_amplitude(frequency, amplitude)
    else:
        spheroidal_amplitude = _ringdown_spheroidal_amplitude(
            frequency,
            state,
            amplitude,
        )
    h32_spheroidal = spheroidal_amplitude * torch.exp(
        1j * _ringdown_spheroidal_phase(frequency, state, phase)
    )
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


def _derivative_graph_input(value, like):
    """Return one detached dynamic input on the derivative graph's device."""

    complex_value = (
        value.is_complex()
        if isinstance(value, torch.Tensor)
        else isinstance(value, complex)
    )
    dtype = (
        torch.complex64
        if complex_value and like.dtype == torch.float32
        else torch.complex128
        if complex_value
        else like.dtype
    )
    return torch.as_tensor(value, device=like.device, dtype=dtype).detach()


def _ordered_derivative_graph_values(
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Return derivative parameters in the fixed traced-program order."""

    values = [getattr(state, name) for name in _DERIVATIVE_GRAPH_STATE_FIELDS]
    values.extend(getattr(phase, name) for name in _DERIVATIVE_GRAPH_PHASE_FIELDS)
    values.extend(
        getattr(amplitude, name) for name in _DERIVATIVE_GRAPH_AMPLITUDE_FIELDS
    )
    values.extend(
        getattr(carrier_phase_plan, name)
        for name in _DERIVATIVE_GRAPH_PHASE_PLAN_FIELDS
    )
    values.extend(carrier_phase_plan.inspiral)
    values.extend(carrier_phase_plan.intermediate)
    values.extend(carrier_phase_plan.mergerringdown)
    values.extend(
        getattr(carrier_phase_plan, name)
        for name in ("alpha0", "alpha1", "beta0", "beta1")
    )
    values.extend(carrier_amp_plan)
    return values


def _derivative_graph_inputs(
    frequency,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Pack every parameter-dependent value used by the differentiated graph."""

    if carrier_phase_plan is None or carrier_amp_plan is None:
        return None
    try:
        values = _ordered_derivative_graph_values(
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    except (AttributeError, TypeError):
        return None
    if any(value is None for value in values) or _xutils._tree_has_autograd(values):
        return None

    inputs = (frequency.detach(),) + tuple(
        _derivative_graph_input(value, frequency) for value in values
    )
    # All graph inputs are scalar except the four-term 2022 auxiliary
    # ringdown polynomial. Keep the shape check explicit so a changed ansatz
    # cannot accidentally reuse an incompatible trace.
    auxiliary_index = (
        1
        + len(_DERIVATIVE_GRAPH_STATE_FIELDS)
        + len(_DERIVATIVE_GRAPH_PHASE_FIELDS)
        + len(_DERIVATIVE_GRAPH_AMPLITUDE_FIELDS)
        - 1
    )
    for index, value in enumerate(inputs):
        if index == auxiliary_index:
            if value.ndim != 1 or value.numel() != 4:
                return None
        elif value.ndim != 0:
            return None
    return inputs


def _derivative_graph_component(release, *, region_specialized=False):
    """Build a pure dynamic-input component for one static amplitude release."""

    def component(*values):
        values = iter(values)
        frequency = next(values)
        state = _DerivativeGraphState(
            *(next(values) for _ in _DERIVATIVE_GRAPH_STATE_FIELDS)
        )
        phase = _DerivativeGraphPhase(
            *(next(values) for _ in _DERIVATIVE_GRAPH_PHASE_FIELDS)
        )
        amplitude = _DerivativeGraphAmplitude(
            release,
            *(next(values) for _ in _DERIVATIVE_GRAPH_AMPLITUDE_FIELDS),
        )
        phase_plan_top = [next(values) for _ in _DERIVATIVE_GRAPH_PHASE_PLAN_FIELDS]
        inspiral_plan = _InspiralPhasePlan(
            *(next(values) for _ in _InspiralPhasePlan._fields)
        )
        intermediate_plan = _IntermediatePhasePlan(
            *(next(values) for _ in _IntermediatePhasePlan._fields)
        )
        ringdown_plan = _MergerRingdownPhasePlan(
            *(next(values) for _ in _MergerRingdownPhasePlan._fields)
        )
        phase_alignment = [next(values) for _ in range(4)]
        phase_plan = _IMRPhenomXASPhasePlan(
            *phase_plan_top,
            inspiral_plan,
            intermediate_plan,
            ringdown_plan,
            inspiral_plan,
            intermediate_plan,
            ringdown_plan,
            *phase_alignment,
            False,
        )
        amp_plan = _MergerRingdownAmpPlan(
            *(next(values) for _ in _MergerRingdownAmpPlan._fields)
        )
        try:
            next(values)
        except StopIteration:
            pass
        else:
            raise AssertionError("unconsumed mode-32 derivative graph inputs")

        return _mixed_ringdown_component(
            frequency,
            state,
            phase,
            amplitude,
            None,
            None,
            None,
            phase_plan,
            amp_plan,
            derivative_region_specialized=region_specialized,
        )

    return component


def _build_derivative_graph(inputs, release, *, region_specialized=False):
    """Materialize and trace the exact reverse-over-reverse scalar program."""

    from torch.fx.experimental.proxy_tensor import make_fx

    component = _derivative_graph_component(
        release,
        region_specialized=region_specialized,
    )

    def angle(frequency, *parameters):
        return torch.angle(component(frequency, *parameters))

    def differentiated(frequency, *parameters):
        def first_with_value(inner_frequency, *inner_parameters):
            first, value = torch.func.grad_and_value(angle, argnums=0)(
                inner_frequency,
                *inner_parameters,
            )
            return first, value

        second, (first, _value) = torch.func.grad_and_value(
            first_with_value,
            argnums=0,
            has_aux=True,
        )(frequency, *parameters)
        return first, second

    graph = make_fx(differentiated)(*inputs)
    # TorchScript remains the only exact, low-overhead executor for this
    # complex reverse-over-reverse graph in the supported Torch release;
    # Inductor currently fails its complex fallback.
    return _call_derivative_graph(torch.jit.trace, graph, inputs, check_trace=False)


def _call_derivative_graph(function, *args, **kwargs):
    """Keep TorchScript's deprecation internal to this opt-in bridge."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"`torch\.jit\."
                r"(freeze|trace|trace_method|script)` is deprecated.*"
            ),
            category=DeprecationWarning,
            module=r"torch\.jit\..*",
        )
        return function(*args, **kwargs)


def _derivative_graph_key(inputs, release, *, region_specialized=False):
    frequency = inputs[0]
    return (
        frequency.device.type,
        frequency.device.index,
        release,
        region_specialized,
        tuple((value.dtype, tuple(value.shape)) for value in inputs),
    )


def _cached_derivative_graph(inputs, release, *, region_specialized=False):
    """Return one process-local trace, or ``None`` after a safe build failure."""

    key = _derivative_graph_key(
        inputs,
        release,
        region_specialized=region_specialized,
    )
    with _DERIVATIVE_GRAPH_LOCK:
        graph = _DERIVATIVE_GRAPH_CACHE.get(key)
        if graph is not None:
            return key, graph
        if key in _DERIVATIVE_GRAPH_FAILURES:
            return key, None
        try:
            graph = _build_derivative_graph(
                inputs,
                release,
                region_specialized=region_specialized,
            )
        except Exception:
            _DERIVATIVE_GRAPH_FAILURES.add(key)
            return key, None
        _DERIVATIVE_GRAPH_CACHE[key] = graph
        return key, graph


def _invalidate_derivative_graph(key):
    with _DERIVATIVE_GRAPH_LOCK:
        _DERIVATIVE_GRAPH_CACHE.pop(key, None)
        _DERIVATIVE_GRAPH_FAILURES.add(key)


def _derivative_region_value_supported(value, like):
    """Fail closed on tensor wrappers, cross-device values, or user AD state."""

    if isinstance(value, torch.Tensor):
        expected_dtype = torch.complex128 if value.is_complex() else like.dtype
        return (
            type(value) is torch.Tensor
            and value.layout == torch.strided
            and value.device == like.device
            and value.dtype == expected_dtype
            and not value.is_conj()
            and not value.is_neg()
            and not _xutils._tree_has_autograd(value)
        )
    if isinstance(value, (tuple, list)):
        return all(_derivative_region_value_supported(item, like) for item in value)
    return type(value) in (int, float, complex)


def _derivative_region_specialization_controls_supported(
    mf,
    cutoff,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Prove the host/runtime controls for the exact MRD/auxiliary program.

    The mode-32 matching frequency and 2022 auxiliary boundary are Python
    floats.  The carrier phase boundary is reconstructed from the same host
    remnant state used by the internal carrier plan.  Co-precessing deviations
    and exact boundaries retain the legacy piecewise program.
    """

    if (
        type(mf) is not torch.Tensor
        or mf.layout != torch.strided
        or mf.ndim != 1
        or mf.dtype != torch.float64
        or mf.device.type not in ("cpu", "cuda")
        or mf.is_conj()
        or mf.is_neg()
        or _xutils._tree_has_autograd(mf)
        or not _derivative_region_specialization_runtime_supported(mf)
        or type(state) is not _Mode32State
        or type(phase) is not _Phase32
        or type(amplitude) is not _Amplitude32
        or not _imrphenomxas_phase_plan_type_supported(carrier_phase_plan)
        or not _imrphenomxas_ringdown_amp_plan_type_supported(
            carrier_amp_plan
        )
        or amplitude.release != 122022
        or phase.carrier_coprecessing_deviations is not None
    ):
        return False

    host_values = (
        cutoff,
        state.f_ring_22,
        state.f_isco_22,
        state.f_meco_22,
        amplitude.f_rd_aux,
    )
    if not all(type(value) is float and math.isfinite(value) for value in host_values):
        return False
    fMs_IMmatch = 0.6 * (0.5 * state.f_ring_22 + state.f_isco_22)
    deltafMs = (fMs_IMmatch - state.f_meco_22) * 0.03
    f2_Ms = fMs_IMmatch + 0.5 * deltafMs
    if not (cutoff > f2_Ms and cutoff < amplitude.f_rd_aux and cutoff < _xutils.fM_CUT):
        return False

    return True


def _derivative_region_specialization_supported(
    mf,
    cutoff,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Select the exact MRD/auxiliary scalar program without device reads."""

    if not _derivative_region_specialization_controls_supported(
        mf,
        cutoff,
        state,
        phase,
        amplitude,
        carrier_phase_plan,
        carrier_amp_plan,
    ):
        return False
    values = _ordered_derivative_graph_values(
        state,
        phase,
        amplitude,
        carrier_phase_plan,
        carrier_amp_plan,
    )
    if not _derivative_region_value_supported(values, mf):
        return False

    # CPU can cheaply verify that the internal tensor plan and host remnant
    # reconstruction agree.  Avoid introducing a CUDA scalar synchronization.
    if mf.device.type == "cpu":
        fMs_IMmatch = 0.6 * (0.5 * state.f_ring_22 + state.f_isco_22)
        deltafMs = (fMs_IMmatch - state.f_meco_22) * 0.03
        if float(carrier_phase_plan.f2_Ms) != fMs_IMmatch + 0.5 * deltafMs:
            return False
    return True


def _specialized_ringdown_phase_derivatives(
    mf,
    cutoff,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Differentiate only the proven active scalar branches, exactly eagerly."""

    with torch.enable_grad():
        frequency = _tensor(cutoff, mf).detach().requires_grad_(True)
        angle = torch.angle(
            _mixed_ringdown_component(
                frequency,
                state,
                phase,
                amplitude,
                None,
                None,
                None,
                carrier_phase_plan,
                carrier_amp_plan,
                derivative_region_specialized=True,
            )
        )
        first = torch.autograd.grad(angle, frequency, create_graph=True)[0]
        second = torch.autograd.grad(first, frequency)[0]
    return first.detach(), second.detach()


def _analytic_ringdown_phase_derivatives(
    mf,
    cutoff,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Differentiate the proven carrier-MRD/auxiliary mixed component.

    For each component ``g = A exp(i phi)``, evaluate ``g'`` and ``g''``
    analytically.  The phase derivatives of their complex sum ``h`` are
    ``Im(h'/h)`` and ``Im(h''/h - (h'/h)^2)``.
    """

    frequency = _tensor(cutoff, mf).detach()

    # Carrier (2, 2): the prepared MRD amplitude is
    # exp(-gammaR*u) * gammaD13 / D.  Logarithmic derivatives avoid
    # differentiating its products and quotient through an autograd graph.
    carrier_offset = frequency - carrier_amp_plan.fMs_RD
    carrier_denominator = (
        carrier_offset * carrier_offset + carrier_amp_plan.gammaD2
    )
    carrier_base_amplitude = (
        torch.exp(-carrier_offset * carrier_amp_plan.gammaR)
        * carrier_amp_plan.gammaD13
        / carrier_denominator
    )
    carrier_power = -7.0 / 6.0
    carrier_amplitude = (
        carrier_base_amplitude
        * amplitude.amp_norm
        * frequency**carrier_power
    )
    carrier_log_first = (
        -carrier_amp_plan.gammaR
        - 2.0 * carrier_offset / carrier_denominator
        + carrier_power / frequency
    )
    carrier_log_second = (
        2.0
        * (
            carrier_offset * carrier_offset
            - carrier_amp_plan.gammaD2
        )
        / (carrier_denominator * carrier_denominator)
        - carrier_power / (frequency * frequency)
    )
    carrier_amplitude_first = carrier_amplitude * carrier_log_first
    carrier_amplitude_second = carrier_amplitude * (
        carrier_log_first * carrier_log_first + carrier_log_second
    )

    carrier_frequency = frequency / state.total_mass_seconds
    carrier_fMs = carrier_frequency * carrier_phase_plan.total_mass_seconds
    carrier_mrd = carrier_phase_plan.mergerringdown
    (
        c0,
        c1,
        c2,
        c4_over_3,
        c_l_over_fd,
        f_rd,
        f_damp,
        _,
        _,
    ) = carrier_mrd
    scaled = (carrier_fMs - f_rd) / f_damp
    scaled_denominator = 1.0 + scaled * scaled
    carrier_mrd_first = (
        c0
        + c1 * carrier_fMs ** (-1.0 / 3.0)
        + c2 * carrier_fMs**-2.0
        + 3.0 * c4_over_3 * carrier_fMs**-4.0
        + c_l_over_fd / (f_damp * scaled_denominator)
    )
    carrier_mrd_second = (
        -(c1 / 3.0) * carrier_fMs ** (-4.0 / 3.0)
        - 2.0 * c2 * carrier_fMs**-3.0
        - 12.0 * c4_over_3 * carrier_fMs**-5.0
        - 2.0
        * c_l_over_fd
        * scaled
        / (f_damp * f_damp * scaled_denominator * scaled_denominator)
    )
    carrier_scale = (
        torch.ones_like(frequency) / state.total_mass_seconds
    ) * carrier_phase_plan.total_mass_seconds
    carrier_phase_first = (
        (1.0 / carrier_phase_plan.eta)
        * (carrier_mrd_first + carrier_phase_plan.beta1)
        * carrier_scale
        + phase.linb
    )
    carrier_phase_second = (
        (1.0 / carrier_phase_plan.eta)
        * carrier_mrd_second
        * carrier_scale
        * carrier_scale
    )
    carrier_phase, _ = _evaluate_mergerringdown_phase(
        carrier_fMs,
        carrier_mrd,
    )
    carrier_phase = (
        carrier_phase
        + carrier_phase_plan.beta0
        + carrier_phase_plan.beta1 * carrier_fMs
    )
    carrier_phase = (1 / carrier_phase_plan.eta) * carrier_phase
    carrier_phase = carrier_phase + phase.linb * frequency + phase.phiref22
    carrier_exponential = torch.exp(1j * carrier_phase)
    h22 = carrier_amplitude * carrier_exponential
    h22_first = (
        carrier_amplitude_first
        + 1j * carrier_amplitude * carrier_phase_first
    ) * carrier_exponential
    h22_second = (
        carrier_amplitude_second
        + 2j * carrier_amplitude_first * carrier_phase_first
        + 1j * carrier_amplitude * carrier_phase_second
        - carrier_amplitude * carrier_phase_first * carrier_phase_first
    ) * carrier_exponential

    # The 2022 auxiliary branch is a cubic in frequency.
    _coefficient0, coefficient1, coefficient2, coefficient3 = (
        amplitude.rd_aux_coefficients.unbind()
    )
    spheroidal_amplitude = _ringdown_auxiliary_amplitude(frequency, amplitude)
    spheroidal_amplitude_first = (
        coefficient1
        + 2.0 * coefficient2 * frequency
        + 3.0 * coefficient3 * frequency * frequency
    )
    spheroidal_amplitude_second = (
        2.0 * coefficient2 + 6.0 * coefficient3 * frequency
    )
    spheroidal_phase = _ringdown_spheroidal_phase(frequency, state, phase)
    spheroidal_phase_first = _ringdown_spheroidal_phase_derivative(
        frequency,
        state,
        phase,
    )
    spheroidal_offset = frequency - state.f_ring_32
    spheroidal_denominator = (
        state.f_damp_32**2 + spheroidal_offset * spheroidal_offset
    )
    spheroidal_phase_second = (
        -2.0 * phase.alpha2_s / frequency**3
        - 4.0 * phase.alpha4_s / frequency**5
        - 2.0
        * phase.alpha_l_s
        * state.f_damp_32
        * spheroidal_offset
        / (spheroidal_denominator * spheroidal_denominator)
    )
    spheroidal_exponential = torch.exp(1j * spheroidal_phase)
    h32_spheroidal = spheroidal_amplitude * spheroidal_exponential
    h32_spheroidal_first = (
        spheroidal_amplitude_first
        + 1j * spheroidal_amplitude * spheroidal_phase_first
    ) * spheroidal_exponential
    h32_spheroidal_second = (
        spheroidal_amplitude_second
        + 2j * spheroidal_amplitude_first * spheroidal_phase_first
        + 1j * spheroidal_amplitude * spheroidal_phase_second
        - spheroidal_amplitude
        * spheroidal_phase_first
        * spheroidal_phase_first
    ) * spheroidal_exponential

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
    mixed = mixing_322 * h22 + mixing_323 * h32_spheroidal
    mixed_first = (
        mixing_322 * h22_first + mixing_323 * h32_spheroidal_first
    )
    mixed_second = (
        mixing_322 * h22_second + mixing_323 * h32_spheroidal_second
    )
    first_ratio = mixed_first / mixed
    first = torch.imag(first_ratio)
    second = torch.imag(mixed_second / mixed - first_ratio * first_ratio)
    return first, second


def _scripted_analytic_phase_tail_values(
    real_pack,
    complex_pack,
    phase_zero,
    inspiral_value,
    inspiral_derivative,
    xas_align,
    mode_align,
    *tensor_values,
):
    """Evaluate the qualified high-eta analytic matching tail."""

    matrix = real_pack[:36].view(6, 6)
    fitted = real_pack[36:40]
    host_real = iter(real_pack[40:54].unbind())
    host_complex = iter(complex_pack.unbind())
    tensors = iter(tensor_values)
    values = []
    for index in range(68):
        if index in _SCRIPTED_ANALYTIC_PHASE_TAIL_HOST_REAL_INDICES:
            values.append(next(host_real))
        elif index in _SCRIPTED_ANALYTIC_PHASE_TAIL_HOST_COMPLEX_INDICES:
            values.append(next(host_complex))
        else:
            values.append(next(tensors))

    values = iter(values)
    state = _DerivativeGraphState(
        *(next(values) for _ in _DERIVATIVE_GRAPH_STATE_FIELDS)
    )
    phase = _DerivativeGraphPhase(
        *(next(values) for _ in _DERIVATIVE_GRAPH_PHASE_FIELDS)
    )
    amplitude = _DerivativeGraphAmplitude(
        122022,
        *(next(values) for _ in _DERIVATIVE_GRAPH_AMPLITUDE_FIELDS),
    )
    phase_plan_top = [
        next(values) for _ in _DERIVATIVE_GRAPH_PHASE_PLAN_FIELDS
    ]
    inspiral_plan = _InspiralPhasePlan(
        *(next(values) for _ in _InspiralPhasePlan._fields)
    )
    intermediate_plan = _IntermediatePhasePlan(
        *(next(values) for _ in _IntermediatePhasePlan._fields)
    )
    ringdown_plan = _MergerRingdownPhasePlan(
        *(next(values) for _ in _MergerRingdownPhasePlan._fields)
    )
    phase_alignment = [next(values) for _ in range(4)]
    phase_plan = _IMRPhenomXASPhasePlan(
        *phase_plan_top,
        inspiral_plan,
        intermediate_plan,
        ringdown_plan,
        inspiral_plan,
        intermediate_plan,
        ringdown_plan,
        *phase_alignment,
        False,
    )
    amp_plan = _MergerRingdownAmpPlan(
        *(next(values) for _ in _MergerRingdownAmpPlan._fields)
    )
    try:
        next(values)
    except StopIteration:
        pass
    else:
        raise AssertionError("unconsumed scripted analytic phase-tail inputs")

    cutoff = real_pack[54]
    derivative_zero, second_derivative_zero = (
        _analytic_ringdown_phase_derivatives(
            cutoff,
            cutoff,
            state,
            phase,
            amplitude,
            phase_plan,
            amp_plan,
        )
    )
    rhs = torch.stack(
        (*fitted.unbind(), derivative_zero, second_derivative_zero)
    )
    c0, c_l, c1, c2, c4, c3 = torch.linalg.solve(matrix, rhs).unbind()

    def intermediate_raw(frequency):
        return (
            c0 * frequency
            + c1 * torch.log(frequency)
            - c2 / frequency
            - c4 / (3.0 * frequency**3)
            - 0.5 * c3 / frequency**2
            + c_l * torch.atan((frequency - state.f_ring_32) / state.f_damp_32)
        )

    def intermediate_derivative(frequency):
        return (
            c0
            + c_l
            * state.f_damp_32
            / (state.f_damp_32**2 + (frequency - state.f_ring_32) ** 2)
            + c1 / frequency
            + c2 / frequency**2
            + c4 / frequency**4
            + c3 / frequency**3
        )

    frequency_in = real_pack[55]
    intermediate_value = intermediate_raw(frequency_in)
    intermediate_slope = intermediate_derivative(frequency_in)
    c1_insp = intermediate_slope - inspiral_derivative
    c_insp = (
        -c1_insp * frequency_in + intermediate_value - inspiral_value
    )

    intermediate_rd = intermediate_raw(cutoff)
    intermediate_slope_rd = intermediate_derivative(cutoff)
    c1_rd = intermediate_slope_rd - derivative_zero
    c_rd = -c1_rd * cutoff + intermediate_rd - phase_zero

    align = real_pack[56]
    mode_align = mode_align + c1_insp * align + c_insp
    delta_phi = torch.fmod(
        xas_align + phase.phiref22 + phase.linb * align - mode_align,
        2.0 * _PI,
    )
    return (
        c0,
        c_l,
        c1,
        c2,
        c3,
        c4,
        c1_insp,
        c_insp,
        c1_rd,
        c_rd,
        delta_phi,
    )


def _scripted_analytic_phase_tail_tensor_supported(
    value,
    *,
    shape,
    dtype,
    check_autograd=True,
):
    """Accept a plain CPU input with the frozen trace's exact schema."""

    return (
        type(value) is torch.Tensor
        and value.layout == torch.strided
        and value.device.type == "cpu"
        and value.dtype == dtype
        and value.shape == shape
        and value.is_contiguous()
        and not value.is_conj()
        and not value.is_neg()
        and (
            not check_autograd
            or (
                not value.requires_grad
                and value.grad_fn is None
                and not _xutils._tensor_has_forward_ad(value)
            )
        )
    )


def _scripted_analytic_phase_tail_supported(
    mf,
    cutoff,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Prove the ordinary CPU/high-eta domain sealed by qualification."""

    for predicate in (
        getattr(torch, "is_anomaly_enabled", None),
        getattr(torch, "are_deterministic_algorithms_enabled", None),
    ):
        if (
            predicate is None
            or _ringdown_boundary_runtime_boolean(predicate) is not False
        ):
            return False
    return (
        _analytic_phase_derivatives_enabled()
        and type(mf) is torch.Tensor
        and mf.layout == torch.strided
        and mf.device.type == "cpu"
        and mf.dtype == torch.float64
        and mf.ndim == 1
        and mf.is_contiguous()
        and mf.storage_offset() == 0
        and mf._base is None
        and mf._version == 0
        and not mf.is_conj()
        and not mf.is_neg()
        and not _xutils._tree_has_autograd_untrusted(mf)
        and type(state.eta) is float
        and math.isfinite(state.eta)
        and state.eta > 0.05
        and type(phase.transitions) is _Transitions
        and len(phase.transitions.phase_intermediate_points) == 6
        and _derivative_region_specialization_controls_supported(
            mf,
            cutoff,
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    )


def _scripted_analytic_phase_tail_rows_and_fits(state, phase):
    """Build the unchanged high-eta matrix and four fitted RHS values."""

    cutoff = phase.transitions.f_phase_rd
    frequencies = list(phase.transitions.phase_intermediate_points)
    fit_values = list(_xhm32_intermediate_phase_fit_values(state))
    if len(frequencies) != 6 or len(fit_values) != 6:
        return None
    frequencies[-2] = cutoff
    frequencies[-1] = cutoff
    rows = []
    for index, frequency in enumerate(frequencies):
        if index == len(frequencies) - 1:
            offset = frequency - state.f_ring_32
            denominator = state.f_damp_32**2 + offset**2
            rows.append(
                (
                    0.0,
                    -2.0 * state.f_damp_32 * offset / denominator**2,
                    -(frequency**-2),
                    -2.0 * frequency**-3,
                    -4.0 * frequency**-5,
                    -3.0 * frequency**-4,
                )
            )
        else:
            rows.append(
                (
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
                )
            )
    flat_rows = tuple(value for row in rows for value in row)
    return flat_rows, tuple(fit_values[:4])


def _scripted_analytic_phase_tail_inputs(
    mf,
    state,
    phase,
    amplitude,
    intrinsic,
    phase_table,
    carrier_phase_plan,
    carrier_amp_plan,
    carrier_phase_anchors,
    carrier_inspiral_align,
    phase_zero,
):
    """Pack one generic trace invocation after proving its finite schema."""

    cutoff = phase.transitions.f_phase_rd
    try:
        supported = _scripted_analytic_phase_tail_supported(
            mf,
            cutoff,
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    except Exception:
        supported = False
    if not supported:
        return None

    try:
        ordered = _ordered_derivative_graph_values(
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
        if len(ordered) != 68:
            return None
        host_real = tuple(
            ordered[index]
            for index in _SCRIPTED_ANALYTIC_PHASE_TAIL_HOST_REAL_INDICES
        )
        host_complex = tuple(
            ordered[index]
            for index in _SCRIPTED_ANALYTIC_PHASE_TAIL_HOST_COMPLEX_INDICES
        )
        if not all(
            type(value) is float and math.isfinite(value)
            for value in host_real
        ):
            return None
        if not all(
            type(value) is complex
            and math.isfinite(value.real)
            and math.isfinite(value.imag)
            for value in host_complex
        ):
            return None

        tensor_values = tuple(
            ordered[index]
            for index in _SCRIPTED_ANALYTIC_PHASE_TAIL_TENSOR_INDICES
        )
        for index, value in zip(
            _SCRIPTED_ANALYTIC_PHASE_TAIL_TENSOR_INDICES,
            tensor_values,
        ):
            shape = (
                _SCRIPTED_ANALYTIC_PHASE_TAIL_AUXILIARY_SHAPE
                if index == 21
                else _SCRIPTED_ANALYTIC_PHASE_TAIL_SCALAR_SHAPE
            )
            if not _scripted_analytic_phase_tail_tensor_supported(
                value,
                shape=shape,
                dtype=torch.float64,
            ):
                return None

        # Retain the CPU-only equality proof from the derivative-specialized
        # gate after validating the plan tensor's exact plain schema.
        fMs_IMmatch = 0.6 * (0.5 * state.f_ring_22 + state.f_isco_22)
        deltafMs = (fMs_IMmatch - state.f_meco_22) * 0.03
        if float(carrier_phase_plan.f2_Ms) != fMs_IMmatch + 0.5 * deltafMs:
            return None

        rows_and_fits = _scripted_analytic_phase_tail_rows_and_fits(
            state,
            phase,
        )
        if rows_and_fits is None:
            return None
        rows, fit_values = rows_and_fits

        lambda_pn = _lambda_pn(state)

        def inspiral_raw(frequency):
            return (
                get_inspiral_phase(
                    frequency,
                    intrinsic,
                    phase_table,
                    _phase_plan=carrier_phase_plan,
                )
                / state.eta
                + lambda_pn * frequency
            )

        inspiral_value, inspiral_derivative = _value_and_derivative(
            inspiral_raw,
            phase.transitions.f_phase_in,
            mf,
            carrier_phase_plan=carrier_phase_plan,
            state_eta=state.eta,
            lambda_pn=lambda_pn,
        )
        f_align = state.f_meco_22 * 0.6
        align_tensor = _tensor(f_align, mf)
        xas_align = _carrier_phase_anchor(
            carrier_phase_anchors,
            _CARRIER_ALIGNMENT_PHASE,
            mf,
            lambda: Phase(
                align_tensor / state.total_mass_seconds,
                intrinsic,
                phase_table,
                final_spin=state.final_spin,
                coprecessing_deviations=(
                    phase.carrier_coprecessing_deviations
                ),
                _phase_plan=carrier_phase_plan,
            ),
        )
        if carrier_inspiral_align is None:
            mode_align = inspiral_raw(align_tensor)
        else:
            mode_align = (
                carrier_inspiral_align / state.eta
                + lambda_pn * align_tensor
            )

        precomputed = (
            phase_zero,
            inspiral_value,
            inspiral_derivative,
            xas_align,
            mode_align,
        )
        if not all(
            _scripted_analytic_phase_tail_tensor_supported(
                value,
                shape=_SCRIPTED_ANALYTIC_PHASE_TAIL_SCALAR_SHAPE,
                dtype=torch.float64,
            )
            for value in precomputed
        ):
            return None

        real_pack = torch.tensor(
            (
                *rows,
                *fit_values,
                *host_real,
                cutoff,
                phase.transitions.f_phase_in,
                f_align,
            ),
            dtype=torch.float64,
            device=mf.device,
        )
        complex_pack = torch.tensor(
            host_complex,
            dtype=torch.complex128,
            device=mf.device,
        )
        if not (
            _scripted_analytic_phase_tail_tensor_supported(
                real_pack,
                shape=torch.Size((57,)),
                dtype=torch.float64,
                check_autograd=False,
            )
            and _scripted_analytic_phase_tail_tensor_supported(
                complex_pack,
                shape=torch.Size((2,)),
                dtype=torch.complex128,
                check_autograd=False,
            )
            and bool(torch.isfinite(real_pack).all().item())
            and bool(torch.isfinite(complex_pack).all().item())
        ):
            return None
    except Exception:
        return None
    return (real_pack, complex_pack, *precomputed, *tensor_values)


def _scripted_analytic_phase_tail_output_supported(values):
    """Validate all eleven scalar outputs before mutating the phase."""

    return (
        type(values) is tuple
        and len(values) == len(_SCRIPTED_ANALYTIC_PHASE_TAIL_FIELDS)
        and all(
            type(value) is torch.Tensor
            and value.layout == torch.strided
            and value.device.type == "cpu"
            and value.dtype == torch.float64
            and value.shape == torch.Size(())
            and not value.is_conj()
            and not value.is_neg()
            and not _xutils._tree_has_autograd_untrusted(value)
            for value in values
        )
    )


def _scripted_analytic_phase_tail_raw_equal(reference, candidate):
    """Compare every qualified coefficient as unconverted raw bytes."""

    if not (
        _scripted_analytic_phase_tail_output_supported(reference)
        and _scripted_analytic_phase_tail_output_supported(candidate)
    ):
        return False
    return all(
        torch.equal(
            left.detach().reshape(-1).view(torch.uint8),
            right.detach().reshape(-1).view(torch.uint8),
        )
        for left, right in zip(reference, candidate)
    )


def _trace_scripted_analytic_phase_tail(inputs):
    """Create the exact frozen executor for the fixed generic schema."""

    from torch.fx.experimental.proxy_tensor import make_fx

    graph = make_fx(_scripted_analytic_phase_tail_values)(*inputs)
    traced = _call_derivative_graph(
        torch.jit.trace,
        graph,
        inputs,
        check_trace=False,
        strict=True,
    )
    return _call_derivative_graph(torch.jit.freeze, traced.eval())


def _reset_scripted_analytic_phase_tail_after_fork():
    """Drop process-local JIT state and replace an inherited lock."""

    global _SCRIPTED_ANALYTIC_PHASE_TAIL_EXECUTOR
    global _SCRIPTED_ANALYTIC_PHASE_TAIL_FAILED
    global _SCRIPTED_ANALYTIC_PHASE_TAIL_PID
    global _SCRIPTED_ANALYTIC_PHASE_TAIL_LOCK

    _SCRIPTED_ANALYTIC_PHASE_TAIL_EXECUTOR = None
    _SCRIPTED_ANALYTIC_PHASE_TAIL_FAILED = False
    _SCRIPTED_ANALYTIC_PHASE_TAIL_PID = os.getpid()
    _SCRIPTED_ANALYTIC_PHASE_TAIL_LOCK = threading.Lock()


def _ensure_scripted_analytic_phase_tail_process():
    """Reset inherited state where at-fork hooks are unavailable."""

    if _SCRIPTED_ANALYTIC_PHASE_TAIL_PID != os.getpid():
        _reset_scripted_analytic_phase_tail_after_fork()


def _clear_scripted_analytic_phase_tail_cache():
    """Release the one executor and sticky failure for debugging."""

    global _SCRIPTED_ANALYTIC_PHASE_TAIL_EXECUTOR
    global _SCRIPTED_ANALYTIC_PHASE_TAIL_FAILED

    _ensure_scripted_analytic_phase_tail_process()
    with _SCRIPTED_ANALYTIC_PHASE_TAIL_LOCK:
        _SCRIPTED_ANALYTIC_PHASE_TAIL_EXECUTOR = None
        _SCRIPTED_ANALYTIC_PHASE_TAIL_FAILED = False


def _scripted_analytic_phase_tail_cache_state():
    """Return the bounded process-local executor and failure state."""

    _ensure_scripted_analytic_phase_tail_process()
    return (
        _SCRIPTED_ANALYTIC_PHASE_TAIL_EXECUTOR,
        _SCRIPTED_ANALYTIC_PHASE_TAIL_FAILED,
    )


def _mark_scripted_analytic_phase_tail_failed():
    """Fail subsequent calls closed until an explicit cache clear."""

    global _SCRIPTED_ANALYTIC_PHASE_TAIL_EXECUTOR
    global _SCRIPTED_ANALYTIC_PHASE_TAIL_FAILED

    _ensure_scripted_analytic_phase_tail_process()
    with _SCRIPTED_ANALYTIC_PHASE_TAIL_LOCK:
        _SCRIPTED_ANALYTIC_PHASE_TAIL_EXECUTOR = None
        _SCRIPTED_ANALYTIC_PHASE_TAIL_FAILED = True


def _get_or_build_scripted_analytic_phase_tail(inputs, reference):
    """Publish only a trace whose first replay is byte-identical."""

    global _SCRIPTED_ANALYTIC_PHASE_TAIL_EXECUTOR
    global _SCRIPTED_ANALYTIC_PHASE_TAIL_FAILED

    _ensure_scripted_analytic_phase_tail_process()
    executor = _SCRIPTED_ANALYTIC_PHASE_TAIL_EXECUTOR
    if executor is not None:
        return executor
    if _SCRIPTED_ANALYTIC_PHASE_TAIL_FAILED:
        return None
    with _SCRIPTED_ANALYTIC_PHASE_TAIL_LOCK:
        executor = _SCRIPTED_ANALYTIC_PHASE_TAIL_EXECUTOR
        if executor is not None:
            return executor
        if _SCRIPTED_ANALYTIC_PHASE_TAIL_FAILED:
            return None
        try:
            executor = _trace_scripted_analytic_phase_tail(inputs)
            replay = executor(*inputs)
            if not _scripted_analytic_phase_tail_raw_equal(reference, replay):
                raise RuntimeError("scripted analytic phase tail changed bytes")
        except Exception:
            _SCRIPTED_ANALYTIC_PHASE_TAIL_EXECUTOR = None
            _SCRIPTED_ANALYTIC_PHASE_TAIL_FAILED = True
            return None
        _SCRIPTED_ANALYTIC_PHASE_TAIL_EXECUTOR = executor
        return executor


def _scripted_analytic_phase_tail_candidate(
    mf,
    state,
    phase,
    amplitude,
    intrinsic,
    phase_table,
    carrier_phase_plan,
    carrier_amp_plan,
    carrier_phase_anchors,
    carrier_inspiral_align,
    phase_zero,
):
    """Return canary inputs or one validated warm replay."""

    if not _scripted_analytic_phase_tail_enabled():
        return None, None
    executor, failed = _scripted_analytic_phase_tail_cache_state()
    if failed:
        return None, None
    inputs = _scripted_analytic_phase_tail_inputs(
        mf,
        state,
        phase,
        amplitude,
        intrinsic,
        phase_table,
        carrier_phase_plan,
        carrier_amp_plan,
        carrier_phase_anchors,
        carrier_inspiral_align,
        phase_zero,
    )
    if inputs is None or executor is None:
        return inputs, None
    try:
        values = executor(*inputs)
        if not _scripted_analytic_phase_tail_output_supported(values):
            raise RuntimeError("invalid scripted analytic phase-tail output")
    except Exception:
        _mark_scripted_analytic_phase_tail_failed()
        return None, None
    return None, values


def _assign_scripted_analytic_phase_tail(phase, values):
    """Assign a completely validated matching result."""

    for name, value in zip(_SCRIPTED_ANALYTIC_PHASE_TAIL_FIELDS, values):
        setattr(phase, name, value)
    return phase


def _ringdown_phase_derivatives_valid(first, second, mf):
    """Return whether a fast derivative result preserves the scalar contract."""

    return (
        type(first) is torch.Tensor
        and type(second) is torch.Tensor
        and first.ndim == 0
        and second.ndim == 0
        and first.dtype == mf.dtype
        and second.dtype == mf.dtype
        and first.device == mf.device
        and second.device == mf.device
    )


def _cached_ringdown_phase_derivatives(
    mf,
    cutoff,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Return gated fast d/dMf and d2/dMf2, or select the legacy path.

    Keep the traced executor on CPU.  CUDA traces have produced one-ULP-scale
    differences from the eager reverse-over-reverse program on supported
    PyTorch releases, so CUDA uses the exact eager specialized expression.
    The analytic path is algebraically identical but can differ in low bits
    because it does not reproduce autograd's reverse-operation order.
    """

    analytic_enabled = _analytic_phase_derivatives_enabled()
    graph_enabled = _derivative_graph_enabled()
    specialization_enabled = _derivative_region_specialization_enabled()
    if (
        not (analytic_enabled or graph_enabled or specialization_enabled)
        or type(mf) is not torch.Tensor
        or mf.layout != torch.strided
        or mf.ndim != 1
        or mf.dtype != torch.float64
        or mf.device.type not in ("cpu", "cuda")
        or mf.is_conj()
        or mf.is_neg()
        or _xutils._tree_has_autograd(mf)
        or amplitude.release != 122022
    ):
        return None

    analytic_supported = (
        analytic_enabled
        and mf._base is None
        and amplitude.rd_aux_coefficients.ndim == 1
        and amplitude.rd_aux_coefficients.numel() == 4
        and _derivative_region_specialization_supported(
            mf,
            cutoff,
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    )
    if analytic_supported:
        try:
            with torch.no_grad():
                first, second = _analytic_ringdown_phase_derivatives(
                    mf,
                    cutoff,
                    state,
                    phase,
                    amplitude,
                    carrier_phase_plan,
                    carrier_amp_plan,
                )
        except Exception:
            pass
        else:
            if _ringdown_phase_derivatives_valid(first, second, mf):
                return first.detach(), second.detach()

    if not (graph_enabled or specialization_enabled):
        return None
    if (
        graph_enabled
        and not _derivative_region_specialization_runtime_supported(mf)
    ):
        return None
    frequency = _tensor(cutoff, mf).detach()
    inputs = _derivative_graph_inputs(
        frequency,
        state,
        phase,
        amplitude,
        carrier_phase_plan,
        carrier_amp_plan,
    )
    if inputs is None:
        return None
    region_specialized = (
        specialization_enabled
        and _derivative_region_specialization_supported(
            mf,
            cutoff,
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    )
    key = None
    if graph_enabled and mf.device.type == "cpu":
        key, graph = _cached_derivative_graph(
            inputs,
            amplitude.release,
            region_specialized=region_specialized,
        )
        if graph is None:
            return None
        try:
            first, second = graph(*inputs)
        except Exception:
            _invalidate_derivative_graph(key)
            return None
    elif region_specialized:
        try:
            first, second = _specialized_ringdown_phase_derivatives(
                mf,
                cutoff,
                state,
                phase,
                amplitude,
                carrier_phase_plan,
                carrier_amp_plan,
            )
        except Exception:
            return None
    else:
        return None
    if not _ringdown_phase_derivatives_valid(first, second, mf):
        if key is not None:
            _invalidate_derivative_graph(key)
        return None
    return first.detach(), second.detach()


def _inspiral_amplitude(frequency, state, transitions, amplitude):
    ratio = frequency / transitions.f_amp_in
    pseudo = torch.zeros_like(frequency)
    for power, coefficient in enumerate(amplitude.pseudo_coefficients):
        pseudo = pseudo + coefficient * ratio ** ((7.0 + power) / 3.0)
    return (
        _pn_amplitude(
            frequency,
            amplitude.pn_coefficients,
            amplitude.amp_norm,
            amplitude.pn_global_factor,
        )
        + amplitude.amp_norm * frequency ** (-7.0 / 6.0) * pseudo
    )


def _scripted_boundary_lane_enabled():
    """Return the strict mode-32 boundary switch."""

    value = os.environ.get(_SCRIPTED_BOUNDARY_LANE_ENV)
    return (
        True
        if value is None
        else _parse_switch(_SCRIPTED_BOUNDARY_LANE_ENV, value)
    )


def _scripted_mixed_boundary_lane_enabled():
    """Return the strict mixed-boundary switch."""

    value = os.environ.get(_SCRIPTED_MIXED_BOUNDARY_LANE_ENV)
    return (
        True
        if value is None
        else _parse_switch(_SCRIPTED_MIXED_BOUNDARY_LANE_ENV, value)
    )


def _native_cpu_boundary_enabled():
    """Return the independent, strict, off-by-default native CPU switch."""

    value = os.environ.get(_NATIVE_CPU_BOUNDARY_ENV)
    return (
        False
        if value is None
        else _parse_switch(_NATIVE_CPU_BOUNDARY_ENV, value)
    )


def _cuda_graph_mixed_boundary_lane_enabled():
    """Return the independent, strict, off-by-default CUDA graph switch."""

    value = os.environ.get(_CUDA_GRAPH_MIXED_BOUNDARY_LANE_ENV)
    return (
        False
        if value is None
        else _parse_switch(_CUDA_GRAPH_MIXED_BOUNDARY_LANE_ENV, value)
    )


def _scripted_boundary_lane_runtime_supported():
    """Reject tracing and graph-capture runtimes before crossing into JIT."""

    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return False
    try:
        if torch._C._get_tracing_state() is not None:
            return False
    except Exception:
        return False
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if is_compiling is not None:
        try:
            if is_compiling():
                return False
        except Exception:
            return False
    return True


def _scripted_boundary_tensor_supported(value, *, shape, dtype):
    """Return whether ``value`` has the qualified plain-CPU contract."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.device.type == "cpu"
        and value.dtype == dtype
        and (shape is None or value.shape == shape)
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and not value.is_conj()
        and not value.is_neg()
    )


def _scripted_boundary_lane_supported(
    point,
    like,
    value,
    amp_norm,
    pn_global_factor,
    cutoff,
    coefficients,
    pseudo,
):
    """Accept only the byte-qualified release-2022 CPU float64 domain."""

    scalars = (amp_norm, pn_global_factor, cutoff)
    return (
        _scripted_boundary_lane_runtime_supported()
        and type(point) is float
        and math.isfinite(point)
        and all(type(scalar) is float and math.isfinite(scalar) for scalar in scalars)
        and cutoff != 0.0
        and _scripted_boundary_tensor_supported(
            like,
            shape=None,
            dtype=torch.float64,
        )
        and like.ndim == 1
        and _scripted_boundary_tensor_supported(
            value,
            shape=torch.Size(()),
            dtype=torch.float64,
        )
        and value.device == like.device
        and _scripted_boundary_tensor_supported(
            coefficients,
            shape=torch.Size((7,)),
            dtype=torch.complex128,
        )
        and coefficients.device == like.device
        and _scripted_boundary_tensor_supported(
            pseudo,
            shape=torch.Size((3,)),
            dtype=torch.float64,
        )
        and pseudo.device == like.device
        and not _xutils._tree_has_autograd(
            (like, value, coefficients, pseudo)
        )
    )


def _scripted_scalar_power_four(
    frequencies: torch.Tensor,
    exponent: float,
) -> torch.Tensor:
    """Pack four scalar powers without selecting CPU vector libm."""

    return torch.stack(
        (
            frequencies[0] ** exponent,
            frequencies[1] ** exponent,
            frequencies[2] ** exponent,
            frequencies[3] ** exponent,
        )
    )


def _scripted_inspiral_vector_lane(
    frequencies: torch.Tensor,
    coefficients: torch.Tensor,
    amp_norm: float,
    pn_global_factor: float,
    cutoff: float,
    pseudo_coefficients: torch.Tensor,
) -> torch.Tensor:
    """Evaluate four exact mode-32 stencil points on one native axis."""

    series = torch.zeros_like(frequencies, dtype=coefficients.dtype)
    series = series + coefficients[0] * frequencies**0.0
    series = series + coefficients[1] * _scripted_scalar_power_four(
        frequencies,
        1.0 / 3.0,
    )
    series = series + coefficients[2] * _scripted_scalar_power_four(
        frequencies,
        2.0 / 3.0,
    )
    series = series + coefficients[3] * frequencies**1.0
    series = series + coefficients[4] * _scripted_scalar_power_four(
        frequencies,
        4.0 / 3.0,
    )
    series = series + coefficients[5] * _scripted_scalar_power_four(
        frequencies,
        5.0 / 3.0,
    )
    series = series + coefficients[6] * frequencies**2.0

    # The eager closure computes this identical power twice per point. Reuse
    # the byte-identical scalar result for the PN and pseudo-PN terms.
    leading_power = _scripted_scalar_power_four(
        frequencies,
        -7.0 / 6.0,
    )
    pn = (
        torch.abs(series)
        * pn_global_factor
        * amp_norm
        * leading_power
    )
    ratio = frequencies / cutoff
    pseudo = torch.zeros_like(frequencies)
    pseudo = pseudo + pseudo_coefficients[0] * _scripted_scalar_power_four(
        ratio,
        7.0 / 3.0,
    )
    pseudo = pseudo + pseudo_coefficients[1] * _scripted_scalar_power_four(
        ratio,
        8.0 / 3.0,
    )
    pseudo = pseudo + pseudo_coefficients[2] * ratio**3.0
    return pn + amp_norm * leading_power * pseudo


def _get_scripted_boundary_lane_executor():
    """Compile the static mode-32 lane, remembering any failure."""

    global _SCRIPTED_BOUNDARY_LANE_EXECUTOR, _SCRIPTED_BOUNDARY_LANE_FAILED

    if _SCRIPTED_BOUNDARY_LANE_EXECUTOR is not None:
        return _SCRIPTED_BOUNDARY_LANE_EXECUTOR
    if _SCRIPTED_BOUNDARY_LANE_FAILED:
        return None
    with _SCRIPTED_BOUNDARY_LANE_LOCK:
        if _SCRIPTED_BOUNDARY_LANE_EXECUTOR is not None:
            return _SCRIPTED_BOUNDARY_LANE_EXECUTOR
        if _SCRIPTED_BOUNDARY_LANE_FAILED:
            return None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"`torch\.jit\.(trace|trace_method|script)` "
                        r"is deprecated.*"
                    ),
                    category=DeprecationWarning,
                    module=r"torch\.jit\..*",
                )
                executor = torch.jit.script(_scripted_inspiral_vector_lane)
        except Exception:
            _SCRIPTED_BOUNDARY_LANE_FAILED = True
            return None
        _SCRIPTED_BOUNDARY_LANE_EXECUTOR = executor
        return executor


def _clear_scripted_boundary_lane_cache():
    """Release the compiled lane and its remembered failure state."""

    global _SCRIPTED_BOUNDARY_LANE_EXECUTOR, _SCRIPTED_BOUNDARY_LANE_FAILED

    with _SCRIPTED_BOUNDARY_LANE_LOCK:
        _SCRIPTED_BOUNDARY_LANE_EXECUTOR = None
        _SCRIPTED_BOUNDARY_LANE_FAILED = False


def _mark_scripted_boundary_lane_failed():
    """Remember a runtime failure until the cache is explicitly cleared."""

    global _SCRIPTED_BOUNDARY_LANE_EXECUTOR, _SCRIPTED_BOUNDARY_LANE_FAILED

    with _SCRIPTED_BOUNDARY_LANE_LOCK:
        _SCRIPTED_BOUNDARY_LANE_EXECUTOR = None
        _SCRIPTED_BOUNDARY_LANE_FAILED = True


def _mode32_inspiral_boundary(
    function,
    point,
    like,
    *,
    scripted_lane_parameters=None,
):
    """Use the exact fixed-width lane or fail closed to scalar evaluation."""

    if not _scripted_boundary_lane_enabled() or like.dtype != torch.float64:
        return _inspiral_boundary(function, point, like)

    frequency = _tensor(point, like)
    value = function(frequency)
    step = 1.0e-9
    if (
        type(scripted_lane_parameters) is tuple
        and len(scripted_lane_parameters) == 5
    ):
        (
            amp_norm,
            pn_global_factor,
            cutoff,
            coefficients,
            pseudo,
        ) = scripted_lane_parameters
        if _scripted_boundary_lane_supported(
            point,
            like,
            value,
            amp_norm,
            pn_global_factor,
            cutoff,
            coefficients,
            pseudo,
        ):
            executor = _get_scripted_boundary_lane_executor()
            if executor is not None:
                stencil = _tensor(
                    (
                        point + 2.0 * step,
                        point + step,
                        point - step,
                        point - 2.0 * step,
                    ),
                    like,
                )
                try:
                    values = executor(
                        stencil,
                        coefficients,
                        amp_norm,
                        pn_global_factor,
                        cutoff,
                        pseudo,
                    )
                    plus_two, plus_one, minus_one, minus_two = values.unbind()
                except Exception:
                    _mark_scripted_boundary_lane_failed()
                else:
                    derivative = (
                        -plus_two
                        + 8.0 * plus_one
                        - 8.0 * minus_one
                        + minus_two
                    ) / (12.0 * step)
                    return value, derivative

    # An enabled mode-32 lane never falls through to a separately enabled
    # packed-stencil experiment. Preserve the established scalar call order.
    derivative = (
        -function(_tensor(point + 2.0 * step, like))
        + 8.0 * function(_tensor(point + step, like))
        - 8.0 * function(_tensor(point - step, like))
        + function(_tensor(point - 2.0 * step, like))
    ) / (12.0 * step)
    return value, derivative


def _scripted_mixed_boundary_lane_values(
    stencil: torch.Tensor,
    tensor_real: torch.Tensor,
    host_real: torch.Tensor,
    host_complex: torch.Tensor,
    auxiliary: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the four established mixed-ringdown scalar trees."""

    # Keep the fixed 68-field release-122022 schema explicit. The placeholder
    # fields are dead only after the caller has proved the active carrier-MRD
    # and mode-32 auxiliary regions for all four stencil points.
    zero = tensor_real[0]
    values = (
        # Mode-32 state.
        host_real[0],
        zero,
        host_real[1],
        host_real[2],
        host_complex[0],
        host_complex[1],
        # Mode-32 phase.
        tensor_real[0],
        tensor_real[1],
        tensor_real[2],
        tensor_real[3],
        tensor_real[4],
        tensor_real[5],
        tensor_real[6],
        # Mode-32 amplitude.
        host_real[3],
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        auxiliary,
        # Carrier phase-plan header.
        tensor_real[7],
        tensor_real[8],
        zero,
        zero,
        # Carrier inspiral phase (inactive).
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        # Carrier intermediate phase (inactive).
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        # Carrier merger-ringdown phase.
        tensor_real[9],
        tensor_real[10],
        tensor_real[11],
        tensor_real[12],
        tensor_real[13],
        tensor_real[14],
        tensor_real[15],
        zero,
        zero,
        # Carrier phase alignment.
        zero,
        zero,
        tensor_real[16],
        tensor_real[17],
        # Carrier merger-ringdown amplitude.
        tensor_real[18],
        tensor_real[19],
        tensor_real[20],
        tensor_real[21],
        zero,
    )
    component = _derivative_graph_component(
        122022,
        region_specialized=True,
    )
    return torch.stack(
        (
            torch.abs(component(stencil[0], *values)),
            torch.abs(component(stencil[1], *values)),
            torch.abs(component(stencil[2], *values)),
            torch.abs(component(stencil[3], *values)),
        )
    )


def _scripted_mixed_boundary_tensor_supported(value, *, shape, dtype):
    """Accept one owned, immutable-by-contract packed CPU input."""

    return (
        type(value) is torch.Tensor
        and value.layout == torch.strided
        and value.device.type == "cpu"
        and value.dtype == dtype
        and value.shape == shape
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and value._version == 0
        and not value.is_conj()
        and not value.is_neg()
        and not _xutils._tree_has_autograd(value)
    )


def _scripted_mixed_boundary_lane_supported(
    point,
    like,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Prove the exact ordinary-runtime, CPU, and active-region contract."""

    inference_enabled = getattr(torch, "is_inference_mode_enabled", None)
    if (
        not torch.is_grad_enabled()
        or _ringdown_boundary_runtime_boolean(inference_enabled) is not False
        or type(like) is not torch.Tensor
        or like.layout != torch.strided
        or like.device.type != "cpu"
        or like.dtype != torch.float64
        or like.ndim != 1
        or not like.is_contiguous()
        or like.storage_offset() != 0
        or like._base is not None
        or like._version != 0
        or like.is_conj()
        or like.is_neg()
        or type(point) is not float
        or not math.isfinite(point)
        or type(state.total_mass_seconds) is not float
        or type(state.f_ring_32) is not float
        or type(state.f_damp_32) is not float
        or type(amplitude.amp_norm) is not float
        or type(state.mixing_322) is not complex
        or type(state.mixing_323) is not complex
        or not _ringdown_boundary_reuse_runtime_supported(like)
        or not _derivative_region_specialization_supported(
            like,
            point,
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    ):
        return False

    host_values = (
        state.total_mass_seconds,
        state.f_ring_32,
        state.f_damp_32,
        amplitude.amp_norm,
        state.mixing_322.real,
        state.mixing_322.imag,
        state.mixing_323.real,
        state.mixing_323.imag,
    )
    if not all(math.isfinite(value) for value in host_values):
        return False
    if not _scripted_mixed_boundary_tensor_supported(
        amplitude.rd_aux_coefficients,
        shape=torch.Size((4,)),
        dtype=torch.float64,
    ):
        return False

    # Prove that every explicit stencil point selects the same branches used
    # by the region-specialized component. Preserve the host expression order
    # from the carrier plan builder.
    step = 1.0e-9
    lower = point - 2.0 * step
    upper = point + 2.0 * step
    fMs_IMmatch = 0.6 * (0.5 * state.f_ring_22 + state.f_isco_22)
    deltafMs = (fMs_IMmatch - state.f_meco_22) * 0.03
    carrier_f2_Ms = fMs_IMmatch + 0.5 * deltafMs
    return (
        lower > 0.0
        and lower > carrier_f2_Ms
        and upper < amplitude.f_rd_aux
        and upper < _xutils.fM_CUT
    )


def _scripted_mixed_boundary_lane_inputs(
    stencil,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Pack the finite dynamic schema for the global generic executor."""

    tensor_real = torch.stack(
        (
            phase.linb,
            phase.phiref22,
            phase.alpha0_s,
            phase.alpha_l_s,
            phase.alpha2_s,
            phase.alpha4_s,
            phase.phi0_s,
            carrier_phase_plan.total_mass_seconds,
            carrier_phase_plan.eta,
            carrier_phase_plan.mergerringdown.c0,
            carrier_phase_plan.mergerringdown.c1,
            carrier_phase_plan.mergerringdown.c2,
            carrier_phase_plan.mergerringdown.c4ov3,
            carrier_phase_plan.mergerringdown.cLovfda,
            carrier_phase_plan.mergerringdown.fMs_RD,
            carrier_phase_plan.mergerringdown.fMs_damp,
            carrier_phase_plan.beta0,
            carrier_phase_plan.beta1,
            carrier_amp_plan.fMs_RD,
            carrier_amp_plan.gammaR,
            carrier_amp_plan.gammaD2,
            carrier_amp_plan.gammaD13,
        )
    )
    host_real = torch.tensor(
        (
            state.total_mass_seconds,
            state.f_ring_32,
            state.f_damp_32,
            amplitude.amp_norm,
        ),
        dtype=torch.float64,
        device=stencil.device,
    )
    host_complex = torch.tensor(
        (state.mixing_322, state.mixing_323),
        dtype=torch.complex128,
        device=stencil.device,
    )
    inputs = (
        stencil,
        tensor_real,
        host_real,
        host_complex,
        amplitude.rd_aux_coefficients,
    )
    contracts = (
        (torch.Size((4,)), torch.float64),
        (torch.Size((22,)), torch.float64),
        (torch.Size((4,)), torch.float64),
        (torch.Size((2,)), torch.complex128),
        (torch.Size((4,)), torch.float64),
    )
    if not all(
        _scripted_mixed_boundary_tensor_supported(
            value,
            shape=shape,
            dtype=dtype,
        )
        for value, (shape, dtype) in zip(inputs, contracts)
    ):
        return None
    return inputs


def _native_cpu_boundary_scalar_values(
    phase,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Return the exact fixed-width real schema consumed by native code."""

    return (
        phase.linb,
        phase.phiref22,
        phase.alpha0_s,
        phase.alpha_l_s,
        phase.alpha2_s,
        phase.alpha4_s,
        phase.phi0_s,
        carrier_phase_plan.total_mass_seconds,
        carrier_phase_plan.eta,
        carrier_phase_plan.mergerringdown.c0,
        carrier_phase_plan.mergerringdown.c1,
        carrier_phase_plan.mergerringdown.c2,
        carrier_phase_plan.mergerringdown.c4ov3,
        carrier_phase_plan.mergerringdown.cLovfda,
        carrier_phase_plan.mergerringdown.fMs_RD,
        carrier_phase_plan.mergerringdown.fMs_damp,
        carrier_phase_plan.beta0,
        carrier_phase_plan.beta1,
        carrier_amp_plan.fMs_RD,
        carrier_amp_plan.gammaR,
        carrier_amp_plan.gammaD2,
        carrier_amp_plan.gammaD13,
    )


def _native_cpu_boundary_plain_tensor(value, *, shape):
    """Accept only an owned, version-zero, non-AD CPU float64 tensor."""

    try:
        return (
            type(value) is torch.Tensor
            and value.layout == torch.strided
            and value.device.type == "cpu"
            and value.dtype == torch.float64
            and value.shape == shape
            and value.is_contiguous()
            and value.storage_offset() == 0
            and value._base is None
            and not value._is_view()
            and value._version == 0
            and not value.is_conj()
            and not value.is_neg()
            and value.is_leaf
            and not _xutils._tree_has_autograd_untrusted(value)
            and value.untyped_storage().nbytes()
            == value.numel() * value.element_size()
        )
    except (AttributeError, NotImplementedError, RuntimeError, TypeError):
        return False


def _native_cpu_boundary_lane_supported(
    point,
    like,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Prove the native lane's ordinary eager CPU float64 contract."""

    # Boundary-component reuse depends on the centre closure's side effect.
    # Keep that independently gated experiment on its established path.
    if _ringdown_boundary_reuse_enabled():
        return False
    for predicate in (
        getattr(torch, "is_anomaly_enabled", None),
        getattr(torch, "are_deterministic_algorithms_enabled", None),
    ):
        if (
            predicate is None
            or _ringdown_boundary_runtime_boolean(predicate) is not False
        ):
            return False
    if not _scripted_mixed_boundary_lane_supported(
        point,
        like,
        state,
        phase,
        amplitude,
        carrier_phase_plan,
        carrier_amp_plan,
    ):
        return False
    if (
        not _native_cpu_boundary_plain_tensor(like, shape=like.shape)
        or not _native_cpu_boundary_plain_tensor(
            amplitude.rd_aux_coefficients,
            shape=torch.Size((4,)),
        )
    ):
        return False

    # The generated scalar fields can be internal zero-offset views. They are
    # never passed to native code: torch.stack copies them into the separately
    # qualified owned tensor below. Reject subclasses, versions, and AD here.
    return all(
        type(value) is torch.Tensor
        and value.layout == torch.strided
        and value.device.type == "cpu"
        and value.dtype == torch.float64
        and value.shape == torch.Size(())
        and value.is_contiguous()
        and value._version == 0
        and not value.is_conj()
        and not value.is_neg()
        and not _xutils._tree_has_autograd_untrusted(value)
        for value in _native_cpu_boundary_scalar_values(
            phase,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    )


def _native_cpu_boundary_inputs(
    point,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Pack and finitely qualify the one-call native argument schema."""

    tensor_real = torch.stack(
        _native_cpu_boundary_scalar_values(
            phase,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    )
    auxiliary = amplitude.rd_aux_coefficients
    if not (
        _native_cpu_boundary_plain_tensor(
            tensor_real,
            shape=torch.Size((22,)),
        )
        and _native_cpu_boundary_plain_tensor(
            auxiliary,
            shape=torch.Size((4,)),
        )
        and bool(torch.isfinite(tensor_real).all().item())
        and bool(torch.isfinite(auxiliary).all().item())
    ):
        return None
    return (
        point,
        tensor_real,
        state.total_mass_seconds,
        state.f_ring_32,
        state.f_damp_32,
        amplitude.amp_norm,
        state.mixing_322,
        state.mixing_323,
        auxiliary,
    )


def _native_cpu_boundary_module():
    """Import build machinery only after the gate and contract qualify."""

    from . import _imrphenomxhm_mode32_native

    return _imrphenomxhm_mode32_native


def _native_mode32_mixed_boundary(
    function,
    point,
    like,
    *,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Return a qualified native/eager result, or ``None`` to fall through."""

    try:
        supported = _native_cpu_boundary_lane_supported(
            point,
            like,
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    except Exception:
        supported = False
    if not supported:
        return None

    try:
        inputs = _native_cpu_boundary_inputs(
            point,
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    except Exception:
        inputs = None
    if inputs is None:
        return None

    # Do not even import build machinery until the finite packed contract is
    # proved. This keeps unsupported requests entirely on the eager path.
    try:
        native = _native_cpu_boundary_module()
        executor, failed = native.cache_state()
    except Exception:
        return None
    if failed:
        return None

    if executor is not None:
        try:
            packed = executor.evaluate_packed(*inputs)
            if not _native_cpu_boundary_plain_tensor(
                packed,
                shape=torch.Size((2,)),
            ):
                raise RuntimeError("invalid native mode-32 boundary output")
        except Exception:
            native.mark_failed()
            return None
        return packed.unbind()

    # The first eligible request is the platform canary. Evaluate and return
    # the unchanged eager program; native is published only for later calls if
    # the result made from these same inputs is identical as raw bytes.
    eager = _inspiral_boundary(function, point, like)
    try:
        reference = torch.stack(eager)
        if not _native_cpu_boundary_plain_tensor(
            reference,
            shape=torch.Size((2,)),
        ):
            raise RuntimeError("invalid eager mode-32 boundary reference")
        native.get_or_build_qualified(inputs, reference)
    except Exception:
        native.mark_failed()
    return eager


def _scripted_mixed_boundary_raw_equal(reference, candidate):
    """Compare qualified outputs byte-for-byte without dtype conversion."""

    if not (
        _scripted_mixed_boundary_tensor_supported(
            reference,
            shape=torch.Size((4,)),
            dtype=torch.float64,
        )
        and _scripted_mixed_boundary_tensor_supported(
            candidate,
            shape=torch.Size((4,)),
            dtype=torch.float64,
        )
    ):
        return False
    return torch.equal(
        reference.detach().view(torch.uint8),
        candidate.detach().view(torch.uint8),
    )


def _trace_scripted_mixed_boundary_lane(inputs):
    """Trace the generic fixed-schema lane without request-local captures."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings(
            "ignore",
            message=(
                r"`torch\.jit\.(trace|trace_method|script|freeze)` "
                r"is deprecated.*"
            ),
            category=DeprecationWarning,
        )
        return torch.jit.trace(
            _scripted_mixed_boundary_lane_values,
            inputs,
            check_trace=False,
            strict=True,
        )


def _reset_scripted_mixed_boundary_lane_after_fork():
    """Drop process-local JIT state and replace a potentially held lock."""

    global _SCRIPTED_MIXED_BOUNDARY_LANE_EXECUTOR
    global _SCRIPTED_MIXED_BOUNDARY_LANE_FAILED
    global _SCRIPTED_MIXED_BOUNDARY_LANE_PID
    global _SCRIPTED_MIXED_BOUNDARY_LANE_LOCK

    _SCRIPTED_MIXED_BOUNDARY_LANE_EXECUTOR = None
    _SCRIPTED_MIXED_BOUNDARY_LANE_FAILED = False
    _SCRIPTED_MIXED_BOUNDARY_LANE_PID = os.getpid()
    _SCRIPTED_MIXED_BOUNDARY_LANE_LOCK = threading.Lock()


def _ensure_scripted_mixed_boundary_lane_process():
    """Reset an inherited cache even where at-fork hooks are unavailable."""

    if _SCRIPTED_MIXED_BOUNDARY_LANE_PID != os.getpid():
        _reset_scripted_mixed_boundary_lane_after_fork()


def _clear_scripted_mixed_boundary_lane_cache():
    """Release the executor and sticky failure for deterministic debugging."""

    global _SCRIPTED_MIXED_BOUNDARY_LANE_EXECUTOR
    global _SCRIPTED_MIXED_BOUNDARY_LANE_FAILED

    _ensure_scripted_mixed_boundary_lane_process()
    with _SCRIPTED_MIXED_BOUNDARY_LANE_LOCK:
        _SCRIPTED_MIXED_BOUNDARY_LANE_EXECUTOR = None
        _SCRIPTED_MIXED_BOUNDARY_LANE_FAILED = False


def _scripted_mixed_boundary_lane_cache_state():
    """Return the immutable warm executor and sticky-failure state."""

    _ensure_scripted_mixed_boundary_lane_process()
    return (
        _SCRIPTED_MIXED_BOUNDARY_LANE_EXECUTOR,
        _SCRIPTED_MIXED_BOUNDARY_LANE_FAILED,
    )


def _mark_scripted_mixed_boundary_lane_failed():
    """Fail subsequent calls closed until the cache is explicitly cleared."""

    global _SCRIPTED_MIXED_BOUNDARY_LANE_EXECUTOR
    global _SCRIPTED_MIXED_BOUNDARY_LANE_FAILED

    _ensure_scripted_mixed_boundary_lane_process()
    with _SCRIPTED_MIXED_BOUNDARY_LANE_LOCK:
        _SCRIPTED_MIXED_BOUNDARY_LANE_EXECUTOR = None
        _SCRIPTED_MIXED_BOUNDARY_LANE_FAILED = True


def _get_or_build_scripted_mixed_boundary_lane(inputs, reference):
    """Return one self-qualified generic executor and its first replay."""

    global _SCRIPTED_MIXED_BOUNDARY_LANE_EXECUTOR
    global _SCRIPTED_MIXED_BOUNDARY_LANE_FAILED

    _ensure_scripted_mixed_boundary_lane_process()
    executor = _SCRIPTED_MIXED_BOUNDARY_LANE_EXECUTOR
    if executor is not None:
        return executor, None
    if _SCRIPTED_MIXED_BOUNDARY_LANE_FAILED:
        return None, None
    with _SCRIPTED_MIXED_BOUNDARY_LANE_LOCK:
        executor = _SCRIPTED_MIXED_BOUNDARY_LANE_EXECUTOR
        if executor is not None:
            return executor, None
        if _SCRIPTED_MIXED_BOUNDARY_LANE_FAILED:
            return None, None
        try:
            executor = _trace_scripted_mixed_boundary_lane(inputs)
            replay = executor(*inputs)
            if not _scripted_mixed_boundary_raw_equal(reference, replay):
                raise RuntimeError("mixed-boundary trace changed output bytes")
        except Exception:
            _SCRIPTED_MIXED_BOUNDARY_LANE_EXECUTOR = None
            _SCRIPTED_MIXED_BOUNDARY_LANE_FAILED = True
            return None, None
        _SCRIPTED_MIXED_BOUNDARY_LANE_EXECUTOR = executor
        return executor, replay


def _scripted_mixed_boundary_eager_stencil(function, point, like):
    """Evaluate the established four calls in their original order."""

    step = 1.0e-9
    plus_two = function(_tensor(point + 2.0 * step, like))
    plus_one = function(_tensor(point + step, like))
    minus_one = function(_tensor(point - step, like))
    minus_two = function(_tensor(point - 2.0 * step, like))
    return plus_two, plus_one, minus_one, minus_two


def _scripted_mixed_boundary_derivative(values):
    """Apply the unchanged ordered finite-difference reduction."""

    if type(values) is tuple:
        plus_two, plus_one, minus_one, minus_two = values
    else:
        plus_two, plus_one, minus_one, minus_two = values.unbind()
    return (
        -plus_two
        + 8.0 * plus_one
        - 8.0 * minus_one
        + minus_two
    ) / (12.0 * 1.0e-9)


@dataclass(frozen=True)
class _CudaGraphMixedBoundaryLaneState:
    """Owned buffers and executor for one exact host-scalar CUDA key."""

    static_inputs: tuple[torch.Tensor, ...]
    topology: tuple
    scalar_key: tuple[bytes, ...]
    replay_stream: object
    capture_stream: object
    graph: object
    output: torch.Tensor


def _cuda_graph_mixed_boundary_lane_environment():
    """Return every PyCBC switch which can affect captured eager kernels."""

    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("PYCBC_")
        )
    )


def _cuda_graph_mixed_boundary_lane_runtime_supported(like):
    """Accept only an ordinary, observable-mode-free CUDA runtime."""

    _ensure_cuda_graph_mixed_boundary_lane_process()
    if (
        os.getpid() != _CUDA_GRAPH_MIXED_BOUNDARY_LANE_PID
        or type(like) is not torch.Tensor
        or like.device.type != "cuda"
        or like.device.index is None
        or not torch.cuda.is_available()
        or not callable(getattr(torch.cuda, "CUDAGraph", None))
        or not callable(getattr(torch.cuda, "Stream", None))
        or not callable(getattr(torch.cuda, "graph", None))
        or _ringdown_boundary_runtime_boolean(
            getattr(torch.jit, "is_scripting", None)
        )
        is not False
        or _ringdown_boundary_runtime_boolean(
            getattr(torch.jit, "is_tracing", None)
        )
        is not False
    ):
        return False

    tracing_state = getattr(getattr(torch, "_C", None), "_get_tracing_state", None)
    if tracing_state is None:
        return False
    try:
        if tracing_state() is not None:
            return False
    except Exception:
        return False

    compiler = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if compiler is not None and _ringdown_boundary_runtime_boolean(compiler) is not False:
        return False
    dynamo = getattr(getattr(torch, "_dynamo", None), "is_compiling", None)
    if _ringdown_boundary_runtime_boolean(dynamo) is not False:
        return False
    if getattr(torch.autograd.forward_ad, "_current_level", None) != -1:
        return False

    torch_c = getattr(torch, "_C", None)
    functorch = getattr(torch_c, "_functorch", None)
    dynamic_depth = getattr(functorch, "get_dynamic_layer_stack_depth", None)
    transforms_active = getattr(torch_c, "_are_functorch_transforms_active", None)
    try:
        if dynamic_depth is not None:
            if dynamic_depth() != 0:
                return False
        elif (
            transforms_active is None
            or bool(transforms_active())
        ):
            return False
        for name in ("_len_torch_dispatch_stack", "_len_torch_function_stack"):
            stack_length = getattr(torch_c, name, None)
            if stack_length is None or stack_length() != 0:
                return False
    except Exception:
        return False

    autocast_enabled = getattr(torch, "is_autocast_enabled", None)
    if autocast_enabled is None:
        return False
    try:
        if autocast_enabled("cpu") or autocast_enabled("cuda"):
            return False
    except (RuntimeError, TypeError):
        legacy_cpu = getattr(torch, "is_autocast_cpu_enabled", None)
        if (
            _ringdown_boundary_runtime_boolean(autocast_enabled) is not False
            or _ringdown_boundary_runtime_boolean(legacy_cpu) is not False
        ):
            return False
    except Exception:
        return False

    # Debug modes can make execution observable or select different kernels.
    # Do not capture under them, and do not silently reuse a graph after they
    # change.
    if (
        _ringdown_boundary_runtime_boolean(
            getattr(torch, "is_anomaly_enabled", None)
        )
        is not False
        or _ringdown_boundary_runtime_boolean(
            getattr(torch, "are_deterministic_algorithms_enabled", None)
        )
        is not False
    ):
        return False
    deterministic_debug = getattr(torch, "get_deterministic_debug_mode", None)
    sync_debug = getattr(torch.cuda, "get_sync_debug_mode", None)
    if deterministic_debug is None or sync_debug is None:
        return False
    try:
        if deterministic_debug() != 0 or sync_debug() != 0:
            return False
        bad_fork = getattr(torch.cuda, "_is_in_bad_fork", None)
        if bad_fork is not None and bad_fork():
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        stream = torch.cuda.current_stream(like.device)
        if stream.device != like.device or type(stream.cuda_stream) is not int:
            return False
    except Exception:
        return False
    return True


def _cuda_graph_mixed_boundary_lane_tensor_supported(
    value,
    *,
    shape,
    dtype,
    device,
):
    """Accept one plain, owned, binary64 CUDA input or output."""

    return (
        type(value) is torch.Tensor
        and value.layout == torch.strided
        and value.device == device
        and value.dtype == dtype
        and value.shape == shape
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and value._version == 0
        and not value.is_conj()
        and not value.is_neg()
        and not _xutils._tree_has_autograd_untrusted(value)
    )


def _cuda_graph_mixed_boundary_lane_supported(
    point,
    like,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Prove the exact CUDA, release-122022, and active-region contract."""

    inference_enabled = getattr(torch, "is_inference_mode_enabled", None)
    if (
        not torch.is_grad_enabled()
        or _ringdown_boundary_runtime_boolean(inference_enabled) is not False
        or type(like) is not torch.Tensor
        or like.layout != torch.strided
        or like.device.type != "cuda"
        or like.dtype != torch.float64
        or like.ndim != 1
        or not like.is_contiguous()
        or like.storage_offset() != 0
        or like._base is not None
        or like._version != 0
        or like.is_conj()
        or like.is_neg()
        or _xutils._tree_has_autograd_untrusted(like)
        or type(point) is not float
        or not math.isfinite(point)
        or type(state.total_mass_seconds) is not float
        or type(state.f_ring_32) is not float
        or type(state.f_damp_32) is not float
        or type(amplitude.amp_norm) is not float
        or type(state.mixing_322) is not complex
        or type(state.mixing_323) is not complex
        or not _cuda_graph_mixed_boundary_lane_runtime_supported(like)
        or not _derivative_region_specialization_supported(
            like,
            point,
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    ):
        return False

    host_values = (
        state.total_mass_seconds,
        state.f_ring_32,
        state.f_damp_32,
        amplitude.amp_norm,
        state.mixing_322.real,
        state.mixing_322.imag,
        state.mixing_323.real,
        state.mixing_323.imag,
    )
    if not all(math.isfinite(value) for value in host_values):
        return False
    if not _cuda_graph_mixed_boundary_lane_tensor_supported(
        amplitude.rd_aux_coefficients,
        shape=torch.Size((4,)),
        dtype=torch.float64,
        device=like.device,
    ):
        return False

    step = 1.0e-9
    lower = point - 2.0 * step
    upper = point + 2.0 * step
    fMs_IMmatch = 0.6 * (0.5 * state.f_ring_22 + state.f_isco_22)
    deltafMs = (fMs_IMmatch - state.f_meco_22) * 0.03
    carrier_f2_Ms = fMs_IMmatch + 0.5 * deltafMs
    return (
        lower > 0.0
        and lower > carrier_f2_Ms
        and upper < amplitude.f_rd_aux
        and upper < _xutils.fM_CUT
    )


def _cuda_graph_mixed_boundary_lane_inputs(
    stencil,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Pack only dynamic tensors; byte-sensitive host reals remain Scalars."""

    tensor_real = torch.stack(
        (
            phase.linb,
            phase.phiref22,
            phase.alpha0_s,
            phase.alpha_l_s,
            phase.alpha2_s,
            phase.alpha4_s,
            phase.phi0_s,
            carrier_phase_plan.total_mass_seconds,
            carrier_phase_plan.eta,
            carrier_phase_plan.mergerringdown.c0,
            carrier_phase_plan.mergerringdown.c1,
            carrier_phase_plan.mergerringdown.c2,
            carrier_phase_plan.mergerringdown.c4ov3,
            carrier_phase_plan.mergerringdown.cLovfda,
            carrier_phase_plan.mergerringdown.fMs_RD,
            carrier_phase_plan.mergerringdown.fMs_damp,
            carrier_phase_plan.beta0,
            carrier_phase_plan.beta1,
            carrier_amp_plan.fMs_RD,
            carrier_amp_plan.gammaR,
            carrier_amp_plan.gammaD2,
            carrier_amp_plan.gammaD13,
        )
    )
    host_complex = torch.tensor(
        (state.mixing_322, state.mixing_323),
        dtype=torch.complex128,
        device=stencil.device,
    )
    inputs = (
        stencil,
        tensor_real,
        host_complex,
        amplitude.rd_aux_coefficients,
    )
    contracts = (
        (torch.Size((4,)), torch.float64),
        (torch.Size((22,)), torch.float64),
        (torch.Size((2,)), torch.complex128),
        (torch.Size((4,)), torch.float64),
    )
    if not all(
        _cuda_graph_mixed_boundary_lane_tensor_supported(
            value,
            shape=shape,
            dtype=dtype,
            device=stencil.device,
        )
        for value, (shape, dtype) in zip(inputs, contracts)
    ):
        return None
    return inputs


def _cuda_graph_mixed_boundary_lane_topology(inputs):
    """Return fixed input view metadata for the graph cache key."""

    try:
        return tuple(
            (
                value.dtype,
                tuple(value.shape),
                tuple(value.stride()),
                int(value.storage_offset()),
                value.layout,
                value.device,
            )
            for value in inputs
        )
    except Exception:
        return None


def _cuda_graph_mixed_boundary_lane_host_values(state, amplitude):
    """Return the six Python Scalars captured by one exact graph."""

    return (
        state.total_mass_seconds,
        state.f_ring_32,
        state.f_damp_32,
        amplitude.amp_norm,
        state.mixing_322,
        state.mixing_323,
    )


def _cuda_graph_mixed_boundary_lane_scalar_key(host_values):
    """Preserve every IEEE bit, including signed zero, in a hashable key."""

    packed = []
    for value in host_values:
        if type(value) is float and math.isfinite(value):
            packed.append(struct.pack(">d", value))
        elif (
            type(value) is complex
            and math.isfinite(value.real)
            and math.isfinite(value.imag)
        ):
            packed.append(struct.pack(">dd", value.real, value.imag))
        else:
            return None
    return tuple(packed)


def _cuda_graph_mixed_boundary_lane_key(like, topology, scalar_key):
    """Return an exact process/thread/device/stream/runtime/cache key."""

    stream = torch.cuda.current_stream(like.device)
    return (
        os.getpid(),
        threading.get_ident(),
        like.device,
        stream.cuda_stream,
        topology,
        scalar_key,
        bool(torch.is_grad_enabled()),
        bool(torch.is_inference_mode_enabled()),
        torch.get_default_dtype(),
        _cuda_graph_mixed_boundary_lane_environment(),
    )


def _cuda_graph_mixed_boundary_lane_values(host_values):
    """Build the exact scalar-host executor for one physical cache key."""

    (
        total_mass_seconds,
        f_ring_32,
        f_damp_32,
        amp_norm,
        _mixing_322,
        _mixing_323,
    ) = host_values

    def lane(stencil, tensor_real, host_complex, auxiliary):
        zero = tensor_real[0]
        values = (
            total_mass_seconds,
            zero,
            f_ring_32,
            f_damp_32,
            host_complex[0],
            host_complex[1],
            tensor_real[0],
            tensor_real[1],
            tensor_real[2],
            tensor_real[3],
            tensor_real[4],
            tensor_real[5],
            tensor_real[6],
            amp_norm,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            auxiliary,
            tensor_real[7],
            tensor_real[8],
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            tensor_real[9],
            tensor_real[10],
            tensor_real[11],
            tensor_real[12],
            tensor_real[13],
            tensor_real[14],
            tensor_real[15],
            zero,
            zero,
            zero,
            zero,
            tensor_real[16],
            tensor_real[17],
            tensor_real[18],
            tensor_real[19],
            tensor_real[20],
            tensor_real[21],
            zero,
        )
        component = _derivative_graph_component(
            122022,
            region_specialized=True,
        )
        return torch.stack(
            (
                torch.abs(component(stencil[0], *values)),
                torch.abs(component(stencil[1], *values)),
                torch.abs(component(stencil[2], *values)),
                torch.abs(component(stencil[3], *values)),
            )
        )

    return lane


def _cuda_graph_mixed_boundary_lane_raw_equal(reference, candidate):
    """Compare qualified CUDA outputs byte-for-byte without conversion."""

    if not (
        type(reference) is torch.Tensor
        and type(candidate) is torch.Tensor
        and reference.layout == torch.strided
        and candidate.layout == torch.strided
        and reference.device.type == "cuda"
        and candidate.device == reference.device
        and reference.dtype == torch.float64
        and candidate.dtype == torch.float64
        and reference.shape == torch.Size((4,))
        and candidate.shape == torch.Size((4,))
        and not _xutils._tree_has_autograd_untrusted((reference, candidate))
    ):
        return False
    return torch.equal(
        reference.detach().contiguous().view(torch.uint8),
        candidate.detach().contiguous().view(torch.uint8),
    )


def _reset_cuda_graph_mixed_boundary_lane_after_fork():
    """Drop process-local CUDA state and replace a potentially held lock."""

    global _CUDA_GRAPH_MIXED_BOUNDARY_LANE_CACHE
    global _CUDA_GRAPH_MIXED_BOUNDARY_LANE_FAILURES
    global _CUDA_GRAPH_MIXED_BOUNDARY_LANE_PID
    global _CUDA_GRAPH_MIXED_BOUNDARY_LANE_LOCK

    _CUDA_GRAPH_MIXED_BOUNDARY_LANE_CACHE = OrderedDict()
    _CUDA_GRAPH_MIXED_BOUNDARY_LANE_FAILURES = OrderedDict()
    _CUDA_GRAPH_MIXED_BOUNDARY_LANE_PID = os.getpid()
    _CUDA_GRAPH_MIXED_BOUNDARY_LANE_LOCK = threading.Lock()


def _ensure_cuda_graph_mixed_boundary_lane_process():
    """Reset inherited graph state where at-fork hooks are unavailable."""

    if _CUDA_GRAPH_MIXED_BOUNDARY_LANE_PID != os.getpid():
        _reset_cuda_graph_mixed_boundary_lane_after_fork()


def _clear_cuda_graph_mixed_boundary_lane_cache():
    """Release graphs/static buffers and all bounded failure records."""

    _ensure_cuda_graph_mixed_boundary_lane_process()
    with _CUDA_GRAPH_MIXED_BOUNDARY_LANE_LOCK:
        _CUDA_GRAPH_MIXED_BOUNDARY_LANE_CACHE.clear()
        _CUDA_GRAPH_MIXED_BOUNDARY_LANE_FAILURES.clear()


def _cuda_graph_mixed_boundary_lane_cache_state(key=None):
    """Return an LRU hit/failure or an immutable complete cache snapshot."""

    _ensure_cuda_graph_mixed_boundary_lane_process()
    with _CUDA_GRAPH_MIXED_BOUNDARY_LANE_LOCK:
        if key is None:
            return (
                tuple(_CUDA_GRAPH_MIXED_BOUNDARY_LANE_CACHE.items()),
                tuple(_CUDA_GRAPH_MIXED_BOUNDARY_LANE_FAILURES),
            )
        state = _CUDA_GRAPH_MIXED_BOUNDARY_LANE_CACHE.pop(key, None)
        if state is not None:
            _CUDA_GRAPH_MIXED_BOUNDARY_LANE_CACHE[key] = state
        return state, key in _CUDA_GRAPH_MIXED_BOUNDARY_LANE_FAILURES


def _cuda_graph_mixed_boundary_lane_remember_failure(key):
    """Bound and remember one exact capture/copy/replay failure key."""

    _CUDA_GRAPH_MIXED_BOUNDARY_LANE_CACHE.pop(key, None)
    _CUDA_GRAPH_MIXED_BOUNDARY_LANE_FAILURES.pop(key, None)
    _CUDA_GRAPH_MIXED_BOUNDARY_LANE_FAILURES[key] = None
    while (
        len(_CUDA_GRAPH_MIXED_BOUNDARY_LANE_FAILURES)
        > _CUDA_GRAPH_MIXED_BOUNDARY_LANE_MAX_ENTRIES
    ):
        _CUDA_GRAPH_MIXED_BOUNDARY_LANE_FAILURES.popitem(last=False)


def _cuda_graph_mixed_boundary_lane_store(key, state):
    """Insert one graph into the tightly bounded least-recently-used cache."""

    _CUDA_GRAPH_MIXED_BOUNDARY_LANE_CACHE.pop(key, None)
    _CUDA_GRAPH_MIXED_BOUNDARY_LANE_CACHE[key] = state
    while (
        len(_CUDA_GRAPH_MIXED_BOUNDARY_LANE_CACHE)
        > _CUDA_GRAPH_MIXED_BOUNDARY_LANE_MAX_ENTRIES
    ):
        _CUDA_GRAPH_MIXED_BOUNDARY_LANE_CACHE.popitem(last=False)


def _build_cuda_graph_mixed_boundary_lane(
    inputs,
    host_values,
    topology,
    scalar_key,
):
    """Warm and capture the unchanged four-call scalar component region."""

    lane = _cuda_graph_mixed_boundary_lane_values(host_values)
    static_inputs = tuple(torch.empty_like(value) for value in inputs)
    for target, source in zip(static_inputs, inputs):
        target.copy_(source)

    device = inputs[0].device
    replay_stream = torch.cuda.current_stream(device)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(replay_stream)
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            lane(*static_inputs)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        output = lane(*static_inputs)
    replay_stream.wait_stream(capture_stream)
    torch.cuda.synchronize(device)
    if not _cuda_graph_mixed_boundary_lane_tensor_supported(
        output,
        shape=torch.Size((4,)),
        dtype=torch.float64,
        device=device,
    ):
        raise RuntimeError("invalid mode-32 mixed-boundary graph output")
    return _CudaGraphMixedBoundaryLaneState(
        static_inputs=static_inputs,
        topology=topology,
        scalar_key=scalar_key,
        replay_stream=replay_stream,
        capture_stream=capture_stream,
        graph=graph,
        output=output,
    )


def _replay_cuda_graph_mixed_boundary_lane(state, inputs, topology):
    """Copy current inputs and replay on the exact captured CUDA stream."""

    device = inputs[0].device
    current_stream = torch.cuda.current_stream(device)
    if (
        state.topology != topology
        or current_stream.cuda_stream != state.replay_stream.cuda_stream
    ):
        raise RuntimeError("mode-32 mixed-boundary graph key changed")
    for target, source in zip(state.static_inputs, inputs):
        target.copy_(source)
    state.graph.replay()
    if not _cuda_graph_mixed_boundary_lane_tensor_supported(
        state.output,
        shape=torch.Size((4,)),
        dtype=torch.float64,
        device=device,
    ):
        raise RuntimeError("invalid mode-32 mixed-boundary graph replay")
    return state.output


def _get_or_build_cuda_graph_mixed_boundary_lane(
    key,
    inputs,
    host_values,
    topology,
    scalar_key,
    reference,
):
    """Return one byte-qualified state and its first post-capture replay."""

    _ensure_cuda_graph_mixed_boundary_lane_process()
    with _CUDA_GRAPH_MIXED_BOUNDARY_LANE_LOCK:
        state = _CUDA_GRAPH_MIXED_BOUNDARY_LANE_CACHE.pop(key, None)
        if state is not None:
            _CUDA_GRAPH_MIXED_BOUNDARY_LANE_CACHE[key] = state
            return state, None
        if key in _CUDA_GRAPH_MIXED_BOUNDARY_LANE_FAILURES:
            return None, None
        try:
            state = _build_cuda_graph_mixed_boundary_lane(
                inputs,
                host_values,
                topology,
                scalar_key,
            )
            replay = _replay_cuda_graph_mixed_boundary_lane(
                state,
                inputs,
                topology,
            )
            if not _cuda_graph_mixed_boundary_lane_raw_equal(reference, replay):
                raise RuntimeError("mode-32 CUDA graph changed output bytes")
        except Exception:
            _cuda_graph_mixed_boundary_lane_remember_failure(key)
            return None, None
        _cuda_graph_mixed_boundary_lane_store(key, state)
        return state, replay


def _cuda_graph_mode32_mixed_boundary(
    function,
    point,
    like,
    *,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Use one exact keyed CUDA graph, or preserve scalar eager fallback."""

    try:
        supported = _cuda_graph_mixed_boundary_lane_supported(
            point,
            like,
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    except Exception:
        supported = False
    if not supported:
        return _inspiral_boundary(function, point, like)

    step = 1.0e-9
    try:
        stencil = _tensor(
            (
                point + 2.0 * step,
                point + step,
                point - step,
                point - 2.0 * step,
            ),
            like,
        )
        inputs = _cuda_graph_mixed_boundary_lane_inputs(
            stencil,
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
        if inputs is None:
            raise RuntimeError("mode-32 CUDA graph input packing declined")
        topology = _cuda_graph_mixed_boundary_lane_topology(inputs)
        if topology is None:
            raise RuntimeError("mode-32 CUDA graph topology declined")
        host_values = _cuda_graph_mixed_boundary_lane_host_values(
            state,
            amplitude,
        )
        scalar_key = _cuda_graph_mixed_boundary_lane_scalar_key(host_values)
        if scalar_key is None:
            raise RuntimeError("mode-32 CUDA graph host Scalars declined")
        key = _cuda_graph_mixed_boundary_lane_key(like, topology, scalar_key)
    except Exception:
        return _inspiral_boundary(function, point, like)

    graph_state, failed = _cuda_graph_mixed_boundary_lane_cache_state(key)
    value = function(_tensor(point, like))
    if failed:
        eager_values = _scripted_mixed_boundary_eager_stencil(
            function,
            point,
            like,
        )
        return value, _scripted_mixed_boundary_derivative(eager_values)
    if graph_state is not None:
        try:
            replay = _replay_cuda_graph_mixed_boundary_lane(
                graph_state,
                inputs,
                topology,
            )
        except Exception:
            with _CUDA_GRAPH_MIXED_BOUNDARY_LANE_LOCK:
                _cuda_graph_mixed_boundary_lane_remember_failure(key)
            eager_values = _scripted_mixed_boundary_eager_stencil(
                function,
                point,
                like,
            )
            return value, _scripted_mixed_boundary_derivative(eager_values)
        return value, _scripted_mixed_boundary_derivative(replay)

    eager_values = _scripted_mixed_boundary_eager_stencil(
        function,
        point,
        like,
    )
    reference = torch.stack(eager_values)
    graph_state, replay = _get_or_build_cuda_graph_mixed_boundary_lane(
        key,
        inputs,
        host_values,
        topology,
        scalar_key,
        reference,
    )
    if graph_state is None:
        return value, _scripted_mixed_boundary_derivative(eager_values)
    if replay is None:
        try:
            replay = _replay_cuda_graph_mixed_boundary_lane(
                graph_state,
                inputs,
                topology,
            )
        except Exception:
            with _CUDA_GRAPH_MIXED_BOUNDARY_LANE_LOCK:
                _cuda_graph_mixed_boundary_lane_remember_failure(key)
            return value, _scripted_mixed_boundary_derivative(eager_values)
    return value, _scripted_mixed_boundary_derivative(replay)


def _mode32_mixed_boundary(
    function,
    point,
    like,
    *,
    state,
    phase,
    amplitude,
    carrier_phase_plan,
    carrier_amp_plan,
):
    """Use an exact mixed lane or fail closed to scalar eager."""

    if _cuda_graph_mixed_boundary_lane_enabled():
        if type(like) is torch.Tensor and like.device.type == "cuda":
            return _cuda_graph_mode32_mixed_boundary(
                function,
                point,
                like,
                state=state,
                phase=phase,
                amplitude=amplitude,
                carrier_phase_plan=carrier_phase_plan,
                carrier_amp_plan=carrier_amp_plan,
            )

    if _native_cpu_boundary_enabled():
        if type(like) is torch.Tensor and like.device.type == "cpu":
            native_result = _native_mode32_mixed_boundary(
                function,
                point,
                like,
                state=state,
                phase=phase,
                amplitude=amplitude,
                carrier_phase_plan=carrier_phase_plan,
                carrier_amp_plan=carrier_amp_plan,
            )
            if native_result is not None:
                return native_result

    if not _scripted_mixed_boundary_lane_enabled():
        return _inspiral_boundary(function, point, like)
    try:
        supported = _scripted_mixed_boundary_lane_supported(
            point,
            like,
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    except Exception:
        supported = False
    if not supported:
        return _inspiral_boundary(function, point, like)

    executor, failed = _scripted_mixed_boundary_lane_cache_state()
    if failed:
        return _inspiral_boundary(function, point, like)
    step = 1.0e-9
    try:
        stencil = _tensor(
            (
                point + 2.0 * step,
                point + step,
                point - step,
                point - 2.0 * step,
            ),
            like,
        )
        inputs = _scripted_mixed_boundary_lane_inputs(
            stencil,
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    except Exception:
        inputs = None
    if inputs is None:
        return _inspiral_boundary(function, point, like)

    # Preserve the centre-first closure call which records the reusable phase
    # component. Only the subsequent four fixed scalar trees move into JIT.
    value = function(_tensor(point, like))
    if executor is not None:
        try:
            values = executor(*inputs)
            if not _scripted_mixed_boundary_tensor_supported(
                values,
                shape=torch.Size((4,)),
                dtype=torch.float64,
            ):
                raise RuntimeError("invalid mixed-boundary executor output")
        except Exception:
            _mark_scripted_mixed_boundary_lane_failed()
            eager_values = _scripted_mixed_boundary_eager_stencil(
                function,
                point,
                like,
            )
            return value, _scripted_mixed_boundary_derivative(eager_values)
        return value, _scripted_mixed_boundary_derivative(values)

    eager_values = _scripted_mixed_boundary_eager_stencil(
        function,
        point,
        like,
    )
    reference = torch.stack(eager_values)
    executor, replay = _get_or_build_scripted_mixed_boundary_lane(
        inputs,
        reference,
    )
    if executor is None:
        return value, _scripted_mixed_boundary_derivative(eager_values)
    if replay is None:
        try:
            replay = executor(*inputs)
        except Exception:
            _mark_scripted_mixed_boundary_lane_failed()
            return value, _scripted_mixed_boundary_derivative(eager_values)
    return value, _scripted_mixed_boundary_derivative(replay)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        after_in_child=_reset_scripted_analytic_phase_tail_after_fork
    )
    os.register_at_fork(after_in_child=_reset_scripted_mixed_boundary_lane_after_fork)
    os.register_at_fork(
        after_in_child=_reset_cuda_graph_mixed_boundary_lane_after_fork
    )


def _intermediate_amplitude(frequency, amplitude):
    polynomial = torch.zeros_like(frequency)
    frequency_power = torch.ones_like(frequency)
    for coefficient in amplitude.intermediate_coefficients:
        polynomial = polynomial + coefficient * frequency_power
        frequency_power = frequency_power * frequency
    if amplitude.release == 122019:
        return amplitude.amp_norm / polynomial
    return polynomial * frequency ** (-7.0 / 6.0)


def _ringdown_boundary_reuse_tree_supported(value, mf):
    """Accept only ordinary same-device tensors and built-in scalar trees."""

    if type(value) is torch.Tensor:
        return (
            value.layout == torch.strided
            and value.device == mf.device
            and value.dtype in (torch.float64, torch.complex128)
            and not value.is_conj()
            and not value.is_neg()
        )
    if isinstance(value, torch.Tensor):
        return False
    if type(value) in (type(None), bool, int, float, complex, str):
        return True
    if isinstance(value, tuple):
        return all(
            _ringdown_boundary_reuse_tree_supported(item, mf) for item in value
        )
    if type(value) is list:
        return all(
            _ringdown_boundary_reuse_tree_supported(item, mf) for item in value
        )
    if type(value) is dict:
        return all(
            _ringdown_boundary_reuse_tree_supported(item, mf)
            for item in value.values()
        )
    return False


def _ringdown_boundary_component_supported(component, mf):
    """Require an owned ordinary complex scalar before retaining it."""

    return (
        type(component) is torch.Tensor
        and component.layout == torch.strided
        and component.device == mf.device
        and component.dtype == torch.complex128
        and component.ndim == 0
        and component._base is None
        and component.storage_offset() == 0
        and not component.is_conj()
        and not component.is_neg()
        and not _xutils._tree_has_autograd_untrusted(component)
    )


def _ringdown_boundary_reuse_supported(
    mf,
    transitions,
    amplitude,
    values,
):
    """Accept the exact release-122022 plain binary64 boundary contract."""

    trusted_plain_request = _xutils._TRUSTED_PLAIN_REQUEST.get()
    return (
        type(mf) is torch.Tensor
        and mf.layout == torch.strided
        and mf.ndim == 1
        and mf.dtype == torch.float64
        and mf.device.type in ("cpu", "cuda")
        and mf._base is None
        and mf.storage_offset() == 0
        and not mf.is_conj()
        and not mf.is_neg()
        and type(transitions) is _Transitions
        and type(transitions.f_amp_rd) is float
        and type(transitions.f_phase_rd) is float
        and math.isfinite(transitions.f_amp_rd)
        and transitions.f_amp_rd == transitions.f_phase_rd
        and type(amplitude) is _Amplitude32
        and amplitude.release == 122022
        and (
            trusted_plain_request
            or (
                _ringdown_boundary_reuse_runtime_supported(mf)
                and _ringdown_boundary_reuse_tree_supported(values, mf)
                and not _xutils._tree_has_autograd_untrusted(values)
            )
        )
    )


def _build_amplitude_2022(
    mf,
    state,
    transitions,
    phase,
    intrinsic,
    phase_table,
    amp_table,
    carrier_phase_plan=None,
    carrier_amp_plan=None,
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
    collocation_tensors = _tensor(collocation_frequencies, mf)
    pn_values = _pn_amplitude(
        collocation_tensors,
        pn_coefficients,
        amp_norm,
        pn_global_factor,
    )
    pseudo_targets = []
    pseudo_rows = []
    for frequency, frequency_tensor, fit, pn_value in zip(
        collocation_frequencies,
        collocation_tensors,
        fit_values,
        pn_values,
    ):
        pseudo_targets.append(
            (abs(fit) - pn_value) / (amp_norm * frequency_tensor ** (-7.0 / 6.0))
        )
        ratio = frequency / transitions.f_amp_in
        pseudo_rows.append([ratio ** (7.0 / 3.0), ratio ** (8.0 / 3.0), ratio**3])
    pseudo_coefficients = _solve(pseudo_rows, pseudo_targets, mf)

    alambda = abs(xhm32_rd_amp_alambda_fit(state))
    rd_lambda = xhm32_rd_amp_lambda_fit(state)
    rd_sigma = xhm32_rd_amp_sigma_fit(state)
    f_rd_aux = state.f_ring_32 - state.f_damp_32
    f_falloff = state.f_ring_32 + 2.0 * state.f_damp_32
    placeholder = _tensor((0.0, 0.0, 0.0, 0.0), mf)
    amplitude = _Amplitude32(
        release=122022,
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
    reuse_phase_boundary = (
        _ringdown_boundary_reuse_enabled()
        and phase.carrier_coprecessing_deviations is None
        and _ringdown_boundary_reuse_supported(
            mf,
            transitions,
            amplitude,
            (
                mf,
                intrinsic,
                phase_table,
                amp_table,
                tuple(vars(state.base).values()),
                (
                    state.f_ring_32,
                    state.f_damp_32,
                    state.mixing_322,
                    state.mixing_323,
                ),
                tuple(
                    value
                    for name, value in vars(phase).items()
                    if name != "transitions"
                ),
                tuple(vars(amplitude).values()),
                carrier_phase_plan,
                carrier_amp_plan,
            ),
        )
    )
    phase_boundary_component = None

    def inspiral(frequency):
        return _inspiral_amplitude(
            frequency,
            state,
            transitions,
            amplitude,
        )

    def mixed(frequency):
        nonlocal phase_boundary_component
        component = _mixed_ringdown_component(
            frequency,
            state,
            phase,
            amplitude,
            intrinsic,
            phase_table,
            amp_table,
            carrier_phase_plan,
            carrier_amp_plan,
        )
        # _inspiral_boundary evaluates the centre before the four-point
        # stencil.  Retain only that first result and leave every subsequent
        # call and the finite-difference reduction in its established order.
        if (
            reuse_phase_boundary
            and phase_boundary_component is None
            and _ringdown_boundary_component_supported(component, mf)
        ):
            phase_boundary_component = component
        return torch.abs(component)

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
    left_value, left_derivative = _mode32_inspiral_boundary(
        inspiral,
        transitions.f_amp_in,
        mf,
        scripted_lane_parameters=(
            amplitude.amp_norm,
            amplitude.pn_global_factor,
            transitions.f_amp_in,
            amplitude.pn_coefficients,
            amplitude.pseudo_coefficients,
        ),
    )
    if mf.dtype == torch.float64:
        right_value, right_derivative = _mode32_mixed_boundary(
            mixed,
            transitions.f_amp_rd,
            mf,
            state=state,
            phase=phase,
            amplitude=amplitude,
            carrier_phase_plan=carrier_phase_plan,
            carrier_amp_plan=carrier_amp_plan,
        )
    else:
        # MPS does not implement complex-double autograd. A centered
        # float32-scale difference keeps this boundary calculation on-device.
        point = transitions.f_amp_rd
        step = 1.0e-4
        right_value = mixed(_tensor(point, mf))
        right_derivative = (
            mixed(_tensor(point + step, mf)) - mixed(_tensor(point - step, mf))
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
                    (power - 7.0 / 6.0) * frequency ** (power - 1.0 - 7.0 / 6.0)
                    for power in range(8)
                ]
            )
        else:
            intermediate_rows.append(
                [frequency ** (power - 7.0 / 6.0) for power in range(8)]
            )
    amplitude.intermediate_coefficients = _solve(
        intermediate_rows,
        intermediate_targets,
        mf,
    )
    amplitude.phase_boundary_component = phase_boundary_component
    return amplitude


def _build_amplitude_2019(
    mf,
    state,
    transitions,
    phase,
    intrinsic,
    phase_table,
    amp_table,
    carrier_phase_plan=None,
    carrier_amp_plan=None,
):
    from .imrphenomxhm_mode21_2019_torch import (
        _safe_intermediate_coefficients,
    )
    from .imrphenomxhm_mode32_2019_torch import (
        inspiral_model_32_2019,
        intermediate_fit_values_32_2019,
        reject_two_region_32_2019,
        ringdown_parameters_32_2019,
    )

    reject_two_region_32_2019(state)
    amp_norm = math.sqrt(2.0 * state.eta / 3.0) * _PI ** (-1.0 / 6.0)
    pn_global_factor = math.sqrt(5.0 / 7.0) / 3.0
    pn_coefficients = _tensor(
        _pn_amplitude_coefficients(state),
        mf,
        complex_value=True,
    )
    original_cutoff, cutoff, pseudo_coefficients, _ = inspiral_model_32_2019(mf, state)
    if not math.isclose(cutoff, transitions.f_amp_in, rel_tol=0.0, abs_tol=1.0e-15):
        raise RuntimeError("legacy (3, 2) inspiral cutoff changed during setup")

    alambda, rd_lambda, rd_sigma = ringdown_parameters_32_2019(state)
    amplitude = _Amplitude32(
        release=122019,
        amp_norm=amp_norm,
        pn_global_factor=pn_global_factor,
        pn_coefficients=pn_coefficients,
        pseudo_coefficients=pseudo_coefficients,
        rd_alambda=alambda,
        rd_lambda=rd_lambda,
        rd_sigma=rd_sigma,
        f_rd_aux=0.0,
        f_falloff=0.0,
        tail_amplitude=_tensor(0.0, mf),
        tail_decay=_tensor(0.0, mf),
        rd_aux_coefficients=_tensor((), mf),
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
                carrier_phase_plan,
                carrier_amp_plan,
            )
        )

    f1 = cutoff
    f4 = transitions.f_amp_rd
    original_width = f4 - original_cutoff
    f2 = original_cutoff + original_width / 3.0
    f3 = original_cutoff + 2.0 * original_width / 3.0
    left_value, left_derivative = _inspiral_boundary(inspiral, f1, mf)
    right_value, right_derivative = _inspiral_boundary(mixed, f4, mf)
    v1 = amp_norm / left_value
    v4 = amp_norm / right_value
    d1 = -amp_norm * left_derivative / left_value**2
    d4 = -amp_norm * right_derivative / right_value**2
    fit2, fit3 = intermediate_fit_values_32_2019(state)
    v2 = 1.0 / fit2
    v3 = 1.0 / fit3
    version = 105

    if 1.0 / v4 < 0.1 / amp_norm:
        v2 = v3 = 1.0
        version = 101

    corner_veto = (state.q > 2.5 and state.chi1 < -0.6 and state.chi2 > 0.0) or (
        state.chi1 < -0.9 and state.chi2 < -0.9
    )
    high_spin_veto = state.q > 40.0 and state.chi1 > 0.9 and v2 != 1.0 and v3 != 1.0
    if version != 101 and (corner_veto or high_spin_veto):
        v2 = v3 = 1.0
        version = 1032

    if v3 == 1.0:
        v3, f3 = v2, f2
        v2 = 1.0

    amplitude.intermediate_coefficients = _safe_intermediate_coefficients(
        version,
        (v1, v2, v3, v4),
        (f1, f2, f3, f4),
        (d1, d4),
        mf,
    )
    return amplitude


def _build_amplitude(
    mf,
    state,
    transitions,
    phase,
    intrinsic,
    phase_table,
    amp_table,
    amplitude_release=122022,
    carrier_phase_plan=None,
    carrier_amp_plan=None,
):
    amplitude_release = _amplitude_release(amplitude_release)
    if amplitude_release == 122019:
        return _build_amplitude_2019(
            mf,
            state,
            transitions,
            phase,
            intrinsic,
            phase_table,
            amp_table,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    return _build_amplitude_2022(
        mf,
        state,
        transitions,
        phase,
        intrinsic,
        phase_table,
        amp_table,
        carrier_phase_plan,
        carrier_amp_plan,
    )


def _mixed_phase_from_component(component):
    """Apply the established phase-origin operations to a mixed component."""

    angle = torch.fmod(
        torch.angle(component),
        2.0 * _PI,
    )
    return torch.where(angle > 0.0, angle - 2.0 * _PI, angle)


def _mixed_phase_at(
    frequency,
    state,
    phase,
    amplitude,
    intrinsic,
    phase_table,
    amp_table,
    carrier_phase_plan=None,
    carrier_amp_plan=None,
):
    return _mixed_phase_from_component(
        _mixed_ringdown_component(
            frequency,
            state,
            phase,
            amplitude,
            intrinsic,
            phase_table,
            amp_table,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    )


def _complete_phase(
    mf,
    state,
    phase,
    amplitude,
    intrinsic,
    phase_table,
    amp_table,
    carrier_phase_plan=None,
    carrier_amp_plan=None,
    carrier_phase_anchors=None,
    carrier_inspiral_align=None,
):
    transitions = phase.transitions
    cutoff = transitions.f_phase_rd
    phase_boundary_component = amplitude.phase_boundary_component
    amplitude.phase_boundary_component = None
    if phase_boundary_component is None:
        phase_zero = _mixed_phase_at(
            _tensor(cutoff, mf),
            state,
            phase,
            amplitude,
            intrinsic,
            phase_table,
            amp_table,
            carrier_phase_plan,
            carrier_amp_plan,
        )
    else:
        phase_zero = _mixed_phase_from_component(phase_boundary_component)
    scripted_phase_tail_inputs, scripted_phase_tail_values = (
        _scripted_analytic_phase_tail_candidate(
            mf,
            state,
            phase,
            amplitude,
            intrinsic,
            phase_table,
            carrier_phase_plan,
            carrier_amp_plan,
            carrier_phase_anchors,
            carrier_inspiral_align,
            phase_zero,
        )
    )
    if scripted_phase_tail_values is not None:
        return _assign_scripted_analytic_phase_tail(
            phase,
            scripted_phase_tail_values,
        )
    if mf.dtype == torch.float64:
        derivatives = _cached_ringdown_phase_derivatives(
            mf,
            cutoff,
            state,
            phase,
            amplitude,
            carrier_phase_plan,
            carrier_amp_plan,
        )
        if derivatives is None:
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
                        carrier_phase_plan,
                        carrier_amp_plan,
                    )
                )
                derivative_zero = torch.autograd.grad(
                    angle,
                    frequency,
                    create_graph=True,
                )[0]
                second_derivative_zero = torch.autograd.grad(
                    derivative_zero,
                    frequency,
                )[0]
            derivative_zero = derivative_zero.detach()
            second_derivative_zero = second_derivative_zero.detach()
        else:
            derivative_zero, second_derivative_zero = derivatives
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
            carrier_phase_plan,
            carrier_amp_plan,
        )
        phase_plus = _mixed_phase_at(
            _tensor(cutoff + step, mf),
            state,
            phase,
            amplitude,
            intrinsic,
            phase_table,
            amp_table,
            carrier_phase_plan,
            carrier_amp_plan,
        )
        derivative_zero = (phase_plus - phase_minus) / (2.0 * step)
        second_derivative_zero = (phase_plus - 2.0 * phase_zero + phase_minus) / step**2

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
                    -(frequency**-2),
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
                    / (state.f_damp_32**2 + (frequency - state.f_ring_32) ** 2),
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
            + c_l * torch.atan((frequency - state.f_ring_32) / state.f_damp_32)
        )

    def intermediate_derivative(frequency):
        return (
            c0
            + c_l
            * state.f_damp_32
            / (state.f_damp_32**2 + (frequency - state.f_ring_32) ** 2)
            + c1 / frequency
            + c2 / frequency**2
            + c4 / frequency**4
            + c3 / frequency**3
        )

    if state.eta < 0.05:
        c0 = c0 + derivative_zero - intermediate_derivative(_tensor(cutoff, mf))

    lambda_pn = _lambda_pn(state)

    def inspiral_raw(frequency):
        return (
            get_inspiral_phase(
                frequency,
                intrinsic,
                phase_table,
                _phase_plan=carrier_phase_plan,
            )
            / state.eta
            + lambda_pn * frequency
        )

    frequency_in = _tensor(transitions.f_phase_in, mf)
    inspiral_value, inspiral_derivative = _value_and_derivative(
        inspiral_raw,
        transitions.f_phase_in,
        mf,
        carrier_phase_plan=carrier_phase_plan,
        state_eta=state.eta,
        lambda_pn=lambda_pn,
    )
    intermediate_value = intermediate_raw(frequency_in)
    intermediate_slope = intermediate_derivative(frequency_in)
    c1_insp = intermediate_slope - inspiral_derivative
    c_insp = -c1_insp * frequency_in + intermediate_value - inspiral_value

    frequency_rd = _tensor(cutoff, mf)
    intermediate_rd = intermediate_raw(frequency_rd)
    intermediate_slope_rd = intermediate_derivative(frequency_rd)
    c1_rd = intermediate_slope_rd - derivative_zero
    c_rd = -c1_rd * frequency_rd + intermediate_rd - phase_zero

    f_align = state.f_meco_22
    if state.eta > 0.05:
        f_align *= 0.6
    align_tensor = _tensor(f_align, mf)
    xas_align = _carrier_phase_anchor(
        carrier_phase_anchors,
        _CARRIER_ALIGNMENT_PHASE,
        mf,
        lambda: Phase(
            align_tensor / state.total_mass_seconds,
            intrinsic,
            phase_table,
            final_spin=state.final_spin,
            coprecessing_deviations=(phase.carrier_coprecessing_deviations),
            _phase_plan=carrier_phase_plan,
        ),
    )
    if carrier_inspiral_align is None:
        mode_align = inspiral_raw(align_tensor)
    else:
        mode_align = (
            carrier_inspiral_align / state.eta
            + lambda_pn * align_tensor
        )
    mode_align = mode_align + c1_insp * align_tensor + c_insp
    delta_phi = torch.fmod(
        xas_align + phase.phiref22 + phase.linb * align_tensor - mode_align,
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
    if scripted_phase_tail_inputs is not None:
        reference = tuple(
            getattr(phase, name)
            for name in _SCRIPTED_ANALYTIC_PHASE_TAIL_FIELDS
        )
        _get_or_build_scripted_analytic_phase_tail(
            scripted_phase_tail_inputs,
            reference,
        )
    return phase


def _evaluate_phase(
    mf,
    state,
    phase,
    mixed_ringdown,
    intrinsic,
    phase_table,
    carrier_phase_plan=None,
    carrier_inspiral_phase=None,
):
    if carrier_inspiral_phase is None:
        carrier_inspiral_phase = get_inspiral_phase(
            mf,
            intrinsic,
            phase_table,
            _phase_plan=carrier_phase_plan,
        )
    inspiral_raw = (
        carrier_inspiral_phase / state.eta
        + _lambda_pn(state) * mf
    )
    inspiral = inspiral_raw + phase.c1_insp * mf + phase.c_insp + phase.delta_phi
    intermediate = (
        phase.c0 * mf
        + phase.c1 * torch.log(mf)
        - phase.c2 / mf
        - phase.c4 / (3.0 * mf**3)
        - 0.5 * phase.c3 / mf**2
        + phase.c_l * torch.atan((mf - state.f_ring_32) / state.f_damp_32)
        + phase.delta_phi
    )
    ringdown = (
        torch.angle(mixed_ringdown) + phase.c1_rd * mf + phase.c_rd + phase.delta_phi
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


def _evaluate_pruned_phase(
    mf,
    state,
    phase,
    mixed_ringdown,
    intrinsic,
    phase_table,
    indices,
    carrier_phase_plan=None,
    carrier_inspiral_phase=None,
):
    """Evaluate only active mode-32 phase regions in the original order."""

    first, second = indices

    def inspiral_region(frequency):
        if carrier_inspiral_phase is None:
            carrier = get_inspiral_phase(
                frequency,
                intrinsic,
                phase_table,
                _phase_plan=carrier_phase_plan,
            )
        else:
            # The inspiral region always begins at the first grid sample, so
            # the aligned evaluator requests the matching carrier prefix.
            carrier = carrier_inspiral_phase[: frequency.numel()]
        inspiral_raw = carrier / state.eta + _lambda_pn(state) * frequency
        return (
            inspiral_raw
            + phase.c1_insp * frequency
            + phase.c_insp
            + phase.delta_phi
        )

    def intermediate_region(frequency):
        return (
            phase.c0 * frequency
            + phase.c1 * torch.log(frequency)
            - phase.c2 / frequency
            - phase.c4 / (3.0 * frequency**3)
            - 0.5 * phase.c3 / frequency**2
            + phase.c_l
            * torch.atan((frequency - state.f_ring_32) / state.f_damp_32)
            + phase.delta_phi
        )

    values = torch.empty_like(mf)
    values[:first] = _evaluate_aligned_region(
        mf,
        0,
        first,
        inspiral_region,
    )
    values[first:second] = _evaluate_aligned_region(
        mf,
        first,
        second,
        intermediate_region,
    )
    if second < mf.numel():
        padded_start, component = mixed_ringdown
        frequency = mf[padded_start:]
        ringdown = (
            torch.angle(component)
            + phase.c1_rd * frequency
            + phase.c_rd
            + phase.delta_phi
        )
        values[second:] = ringdown[second - padded_start :]
    return values


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
    if amplitude.release == 122019:
        return result
    return torch.where(result < 0.0, _FALSE_ZERO, result)


def _evaluate_pruned_amplitude(
    mf,
    state,
    transitions,
    amplitude,
    mixed_ringdown,
    indices,
):
    """Evaluate only active mode-32 amplitude regions in eager order."""

    first, second = indices
    result = torch.empty_like(mf)
    result[:first] = _evaluate_aligned_region(
        mf,
        0,
        first,
        lambda frequency: _inspiral_amplitude(
            frequency,
            state,
            transitions,
            amplitude,
        ),
    )
    result[first:second] = _evaluate_aligned_region(
        mf,
        first,
        second,
        lambda frequency: _intermediate_amplitude(frequency, amplitude),
    )
    if second < mf.numel():
        padded_start, component = mixed_ringdown
        ringdown = torch.abs(component)
        result[second:] = ringdown[second - padded_start :]
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
    final_spin=None,
    _remnant=None,
    ringdown_frequency=None,
    damping_frequency=None,
    carrier_ringdown_frequency=None,
    carrier_damping_frequency=None,
    carrier_coprecessing_deviations=None,
    carrier_phase_plan=None,
    carrier_amp_plan=None,
    carrier_phase_anchors=None,
    amplitude_release=122022,
    _shared_mode_inputs=None,
    _carrier_inspiral_align=None,
    _shared_carrier_inspiral_phase=None,
    _uniform_grid_metadata=None,
):
    r"""Return active positive-frequency samples of LAL's h_(3,-2)."""

    amplitude_release = _amplitude_release(amplitude_release)
    state = _mode32_state(
        params,
        final_spin=final_spin,
        ringdown_frequency=ringdown_frequency,
        damping_frequency=damping_frequency,
        carrier_ringdown_frequency=carrier_ringdown_frequency,
        carrier_damping_frequency=carrier_damping_frequency,
        _base_state=(
            None if _shared_mode_inputs is None else _shared_mode_inputs.state
        ),
        _remnant=_remnant,
    )
    cutoff_fMs = None
    if _remnant is not None and carrier_coprecessing_deviations is None:
        cutoff_fMs = (
            _remnant.ringdown_frequency,
            _remnant.damping_frequency,
            _remnant.meco_frequency,
            _remnant.isco_frequency,
        )
    if _shared_mode_inputs is None:
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
        phase_table = _xutils._get_phenomx_phase_coeff_table_cached_master(
            device=frequencies.device,
            dtype=frequencies.dtype,
        )
        if reference_frequency is None:
            reference_frequency = float(params.get("f_ref", 0.0))
            if reference_frequency <= 0.0:
                reference_frequency = float(params["f_lower"])
        coa_phase = float(params.get("coa_phase", 0.0))
    else:
        frequencies = _shared_mode_inputs.frequencies
        mf = _shared_mode_inputs.mf
        intrinsic = _shared_mode_inputs.intrinsic
        phase_table = _shared_mode_inputs.phase_coeffs
        reference_frequency = _shared_mode_inputs.reference_frequency
        coa_phase = _shared_mode_inputs.coa_phase
    amp_table = _xutils._get_phenomx_amp_coeff_table_cached_master(
        device=frequencies.device,
        dtype=frequencies.dtype,
    )

    # MPS only supports float32 waveforms. Build the small, ill-conditioned
    # matching systems in CPU float64, then move their coefficients back; the
    # frequency-dependent work remains on the requested device.
    setup = frequencies
    setup_intrinsic = intrinsic
    setup_phase_table = phase_table
    setup_amp_table = amp_table
    setup_phase_plan = carrier_phase_plan
    reuse_amp_plan = _amp_plan_enabled()
    prepare_analytic_derivatives = (
        _analytic_phase_derivatives_enabled() and frequencies.dtype == torch.float64
    )
    prepare_derivative_graph = _derivative_graph_enabled()
    prepare_derivative_specialization = _derivative_region_specialization_enabled()
    prepare_amp_plan = reuse_amp_plan
    setup_amp_plan = None
    if frequencies.dtype != torch.float64:
        setup = torch.empty((), dtype=torch.float64)
        setup_intrinsic = torch.tensor(
            [state.mass1, state.mass2, state.chi1, state.chi2],
            dtype=torch.float64,
        )
        setup_phase_table = (
            _xutils._get_phenomx_phase_coeff_table_cached_master(
            device="cpu", dtype=torch.float64
            )
        )
        setup_amp_table = _xutils._get_phenomx_amp_coeff_table_cached_master(
            device="cpu", dtype=torch.float64
        )
        setup_phase_plan = None
    elif reuse_amp_plan and _carrier_amp_plan_supported(
        carrier_amp_plan,
        setup,
        carrier_coprecessing_deviations,
    ):
        setup_amp_plan = carrier_amp_plan

    with torch_context(setup):
        if (
            prepare_amp_plan
            and setup_amp_plan is None
            and not _amp_plan_inputs_have_autograd(
                setup_intrinsic,
                setup_amp_table,
                state.final_spin,
                carrier_coprecessing_deviations,
            )
        ):
            setup_amp_plan = _prepare_mergerringdown_amp_plan(
                setup_intrinsic,
                setup_amp_table,
                fit_rows=_prepare_amp_fit_rows(
                    setup_intrinsic,
                    setup_amp_table,
                ),
                final_spin=state.final_spin,
                coprecessing_deviations=carrier_coprecessing_deviations,
                _cutoff_fMs=cutoff_fMs,
            )
        if (
            setup_phase_plan is None
            and (
                carrier_phase_plan is not None
                or prepare_analytic_derivatives
                or prepare_derivative_graph
                or prepare_derivative_specialization
            )
            and not _xutils._tree_has_autograd(
                (
                    setup_intrinsic,
                    setup_phase_table,
                    state.final_spin,
                    carrier_coprecessing_deviations,
                )
            )
        ):
            setup_phase_plan = _prepare_phase_plan(
                setup_intrinsic,
                setup_phase_table,
                final_spin=state.final_spin,
                coprecessing_deviations=carrier_coprecessing_deviations,
                _cutoff_fMs=cutoff_fMs,
            )
        transitions = _transition_frequencies(state, amplitude_release)
        phase = _partial_phase(
            setup,
            state,
            transitions,
            setup_intrinsic,
            setup_phase_table,
            reference_frequency,
            coa_phase,
            carrier_coprecessing_deviations,
            setup_phase_plan,
            carrier_phase_anchors,
        )
        amplitude = _build_amplitude(
            setup,
            state,
            transitions,
            phase,
            setup_intrinsic,
            setup_phase_table,
            setup_amp_table,
            amplitude_release,
            setup_phase_plan,
            setup_amp_plan,
        )
        phase = _complete_phase(
            setup,
            state,
            phase,
            amplitude,
            setup_intrinsic,
            setup_phase_table,
            setup_amp_table,
            setup_phase_plan,
            setup_amp_plan,
            carrier_phase_anchors,
            _carrier_inspiral_align,
        )

    if setup is not frequencies:
        _move_coefficient_tensors(phase, frequencies)
        _move_coefficient_tensors(amplitude, frequencies)

    region_plan = _prepare_uniform_region_plan(
        core,
        frequencies,
        mf,
        state,
        transitions,
        amplitude_release,
        uniform_grid_metadata=_uniform_grid_metadata,
    )
    if region_plan is not None and not _region_pruning_values_supported(
        mf,
        phase,
        amplitude,
        intrinsic,
        phase_table,
        amp_table,
        carrier_phase_plan,
        setup_amp_plan if reuse_amp_plan and setup is frequencies else None,
        _shared_carrier_inspiral_phase,
    ):
        region_plan = None

    with torch_context(frequencies):
        amp_plan = setup_amp_plan if reuse_amp_plan and setup is frequencies else None
        if (
            reuse_amp_plan
            and amp_plan is None
            and not _amp_plan_inputs_have_autograd(
                intrinsic,
                amp_table,
                state.final_spin,
                carrier_coprecessing_deviations,
            )
        ):
            amp_plan = _prepare_mergerringdown_amp_plan(
                intrinsic,
                amp_table,
                fit_rows=_prepare_amp_fit_rows(intrinsic, amp_table),
                final_spin=state.final_spin,
                coprecessing_deviations=carrier_coprecessing_deviations,
                _cutoff_fMs=cutoff_fMs,
            )
        if region_plan is None:
            mixed_ringdown = _mixed_ringdown_component(
                mf,
                state,
                phase,
                amplitude,
                intrinsic,
                phase_table,
                amp_table,
                carrier_phase_plan,
                amp_plan,
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
                carrier_phase_plan,
                _shared_carrier_inspiral_phase,
            )
            samples = None
        else:
            mixed_ringdown = None
            if region_plan.ringdown_start < mf.numel():
                padded_start = (
                    region_plan.ringdown_start // _REGION_PRUNING_ALIGNMENT
                ) * _REGION_PRUNING_ALIGNMENT
                mixed_ringdown = (
                    padded_start,
                    _mixed_ringdown_component(
                        mf[padded_start:],
                        state,
                        phase,
                        amplitude,
                        intrinsic,
                        phase_table,
                        amp_table,
                        carrier_phase_plan,
                        amp_plan,
                    ),
                )
            amplitude_values = _evaluate_pruned_amplitude(
                mf,
                state,
                transitions,
                amplitude,
                mixed_ringdown,
                region_plan.amplitude,
            )
            phase_values = _evaluate_pruned_phase(
                mf,
                state,
                phase,
                mixed_ringdown,
                intrinsic,
                phase_table,
                region_plan.phase,
                carrier_phase_plan,
                _shared_carrier_inspiral_phase,
            )
            samples = None
        if samples is None:
            samples = -state.amp0 * amplitude_values * torch.exp(1j * phase_values)
    return samples.to(core.polarization.dtype)
