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

from collections import OrderedDict
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
import itertools
import math
import operator
import os
import struct
import sys
import threading
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple

from pycbc import lal_compat as lal
import numpy as _np
import torch

from pycbc import scheme as _scheme
from pycbc.types import Array as PyCBCArray
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData

from . import _torch_jax as _torch_jax_compat
from ._torch_jax import jax, jnp, torch_context
from . import imrphenomx_utils_torch as IMRPhenomX_utils
from .torch_switches import _parse_switch
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

if TYPE_CHECKING:
    from .imrphenomxpnr_torch import PNRCoprecessingDeviations

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

_PHASE_PLAN_ENV = "PYCBC_IMRPHENOMX_PHASE_PLAN"
_FIXED_SCHEMA_PHASE_PLAN_ENV = (
    "PYCBC_IMRPHENOMXAS_FIXED_SCHEMA_PHASE_PLAN"
)
_PHASE_PLAN_CUDA_SOLVE_GRAPH_ENV = (
    "PYCBC_IMRPHENOMX_PHASE_PLAN_CUDA_SOLVE_GRAPH"
)
_PHASE_PLAN_TORCHSCRIPT_TRACE_ENV = (
    "PYCBC_IMRPHENOMXAS_PHASE_PLAN_TORCHSCRIPT_TRACE"
)
_PHASE_PLAN_TORCHSCRIPT_TRACE_PYTHON_MIN = (3, 11)
_PHASE_PLAN_TORCHSCRIPT_TRACE_PYTHON_MAX = (3, 13)
_PHASE_PLAN_TORCHSCRIPT_TRACE_TORCH_MIN = (2, 9)
_PHASE_PLAN_TORCHSCRIPT_TRACE_TORCH_MAX = (2, 13)
_SCALAR_REGION_DISPATCH_ENV = "PYCBC_IMRPHENOMX_SCALAR_REGION_DISPATCH"
_AMP_PLAN_ENV = "PYCBC_IMRPHENOMX_AMP_PLAN"
_PYTHON_INTERMEDIATE_AMP_ENV = (
    "PYCBC_IMRPHENOMXAS_PYTHON_INTERMEDIATE_AMP"
)
_PYTHON_INSPIRAL_DERIVATIVE_ENV = (
    "PYCBC_IMRPHENOMXAS_PYTHON_INSPIRAL_DERIVATIVE"
)
_PYTHON_INSPIRAL_DERIVATIVE_BULK_IO_ENV = (
    "PYCBC_IMRPHENOMXAS_PYTHON_INSPIRAL_DERIVATIVE_BULK_IO"
)
_INSPIRAL_PHASE_HOST_SCALARS_ENV = (
    "PYCBC_IMRPHENOMXAS_INSPIRAL_PHASE_HOST_SCALARS"
)
_INSPIRAL_AMP_HOST_SCALARS_ENV = (
    "PYCBC_IMRPHENOMXAS_INSPIRAL_AMP_HOST_SCALARS"
)
_REGION_PRUNING_ENV = "PYCBC_IMRPHENOMX_REGION_PRUNING"
_EXACT_SCALAR_DERIVATIVES_ENV = "PYCBC_IMRPHENOMX_EXACT_SCALAR_DERIVATIVES"
_SCALAR_DERIVATIVE_PLAN_CSE_ENV = (
    "PYCBC_IMRPHENOMXAS_SCALAR_DERIVATIVE_PLAN_CSE"
)
_EXACT_SCALAR_AMP_DERIVATIVES_ENV = (
    "PYCBC_IMRPHENOMX_EXACT_SCALAR_AMP_DERIVATIVES"
)
_DERIVED_POWER_REUSE_ENV = "PYCBC_IMRPHENOMX_DERIVED_POWER_REUSE"
_PHASE_PLAN_BULK_COLLOCATION_ENV = (
    "PYCBC_IMRPHENOMX_PHASE_PLAN_BULK_COLLOCATION"
)
_PHASE_FIT_PYTHON_SCALARS_ENV = (
    "PYCBC_IMRPHENOMX_PHASE_FIT_PYTHON_SCALARS"
)
_PHASE_FIT_NATIVE_ITERATOR_ENV = (
    "PYCBC_IMRPHENOMXAS_PHASE_FIT_NATIVE_ITERATOR"
)
_AMP_FIT_PYTHON_SCALARS_ENV = "PYCBC_IMRPHENOMX_AMP_FIT_PYTHON_SCALARS"
_CUDA_AMP_HOST_PACK_ENV = "PYCBC_IMRPHENOMXAS_CUDA_AMP_HOST_PACK"
_AMP_FIT_NATIVE_ITERATOR_ENV = (
    "PYCBC_IMRPHENOMXAS_AMP_FIT_NATIVE_ITERATOR"
)
_PACKED_HEAVISIDE_MASKS_ENV = (
    "PYCBC_IMRPHENOMXAS_PACKED_HEAVISIDE_MASKS"
)
_PACKED_FREQUENCY_PLAN_ENV = (
    "PYCBC_IMRPHENOMXAS_PACKED_FREQUENCY_PLAN"
)
_PACKED_FREQUENCY_PLAN_TORCHSCRIPT_TRACE_ENV = (
    "PYCBC_IMRPHENOMXAS_PACKED_FREQUENCY_PLAN_TORCHSCRIPT_TRACE"
)
_SCRIPTED_PHASE_ANSATZ_CPU_ENV = (
    "PYCBC_IMRPHENOMXAS_SCRIPTED_PHASE_ANSATZ_CPU"
)
_CUDA_GRAPH_PHASE_ANSATZ_ENV = (
    "PYCBC_IMRPHENOMXAS_CUDA_GRAPH_PHASE_ANSATZ"
)
_REQUEST_PROOF_PLAN_ENV = "PYCBC_IMRPHENOMXAS_REQUEST_PROOF_PLAN"
_INTRINSIC_CONTROL_REUSE_ENV = (
    "PYCBC_IMRPHENOMXAS_INTRINSIC_CONTROL_REUSE"
)
_INTRINSIC_PLAN_CACHE_ENV = "PYCBC_IMRPHENOMXAS_INTRINSIC_PLAN_CACHE"
_INTRINSIC_PLAN_CACHE_FAST_HIT_ENV = (
    "PYCBC_IMRPHENOMXAS_INTRINSIC_PLAN_CACHE_FAST_HIT"
)
_PACKED_CUTOFF_REUSE_ENV = "PYCBC_IMRPHENOMXAS_PACKED_CUTOFF_REUSE"
# These private import-time CPU masters are immutable by module contract;
# public coefficient-table accessors return clones.  Cache admission therefore
# requires these exact objects at version zero.  Mutating a master through
# ``.data`` or its NumPy storage is an unsupported internal contract violation.
_PHASE_FIT_COEFFICIENT_SOURCE = (
    IMRPhenomX_utils._PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER
)
_AMP_FIT_COEFFICIENT_SOURCE = (
    IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER
)
_PHASE_FIT_COEFFICIENT_ROWS_PYTHON = None
_PHASE_FIT_COEFFICIENT_ROWS_SOURCE = None
_PHASE_FIT_COEFFICIENT_ROWS_VERSION = None
_AMP_FIT_COEFFICIENT_ROWS_CACHE = None
_AMP_FIT_COEFFICIENT_ROWS_LOCK = threading.Lock()

_REGION_PRUNING_ALIGNMENT = 64
# Paired full-wave timings put the CPU crossover between 493 and 985 samples.
_REGION_PRUNING_MIN_SAMPLES = 512


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


class _IMRPhenomXASSequence(NamedTuple):
    """Validated frequencies and inclination-independent sequence samples."""

    inputs: _IMRPhenomXASInputs
    frequencies: torch.Tensor
    polarization: torch.Tensor
    reference_frequency: float | torch.Tensor


class _InspiralPhasePlan(NamedTuple):
    """Frequency-independent coefficients for the inspiral phase."""

    phi0: FloatLike
    phi1: FloatLike
    phi2: FloatLike
    phi3: FloatLike
    phi4: FloatLike
    phi5: FloatLike
    phi5L: FloatLike
    phi6: FloatLike
    phi6L: FloatLike
    phi7: FloatLike
    phi8: FloatLike
    phi8L: FloatLike
    sigma1: FloatLike
    sigma2: FloatLike
    sigma3: FloatLike
    sigma4: FloatLike


class _IntermediatePhasePlan(NamedTuple):
    """Frequency-independent coefficients for the intermediate phase."""

    b0: FloatLike
    b1: FloatLike
    b2: FloatLike
    b3: FloatLike
    b4: FloatLike
    cL: FloatLike
    fMs_RD: FloatLike
    fMs_damp: FloatLike


class _MergerRingdownPhasePlan(NamedTuple):
    """Frequency-independent coefficients for the merger-ringdown phase."""

    c0: FloatLike
    c1: FloatLike
    c2: FloatLike
    c4ov3: FloatLike
    cLovfda: FloatLike
    fMs_RD: FloatLike
    fMs_damp: FloatLike
    cL: FloatLike
    CV_phase_RD0: FloatLike


class _PrequalifiedInspiralPhasePlan(_InspiralPhasePlan):
    """Request-local proof token for one immutable scalar plan."""

    __slots__ = ()


class _PrequalifiedIntermediatePhasePlan(_IntermediatePhasePlan):
    """Request-local proof token for one immutable scalar plan."""

    __slots__ = ()


class _PrequalifiedMergerRingdownPhasePlan(_MergerRingdownPhasePlan):
    """Request-local proof token for one immutable scalar plan."""

    __slots__ = ()


class _ScriptedInspiralPhasePlan(_InspiralPhasePlan):
    """Request-local proof of the scripted CPU inspiral schema."""

    __slots__ = ()


class _ScriptedIntermediatePhasePlan(_IntermediatePhasePlan):
    """Request-local proof of the scripted CPU intermediate schema."""

    __slots__ = ()


class _ScriptedMergerRingdownPhasePlan(_MergerRingdownPhasePlan):
    """Request-local proof of the scripted CPU ringdown schema."""

    __slots__ = ()


class _CudaGraphInspiralPhasePlan(_InspiralPhasePlan):
    """Request-local proof of the exact CUDA-graph inspiral schema."""

    __slots__ = ()


class _CudaGraphIntermediatePhasePlan(_IntermediatePhasePlan):
    """Request-local proof of the exact CUDA-graph intermediate schema."""

    __slots__ = ()


class _CudaGraphMergerRingdownPhasePlan(_MergerRingdownPhasePlan):
    """Request-local proof of the exact CUDA-graph ringdown schema."""

    __slots__ = ()


class _MergerRingdownAmpPlan(NamedTuple):
    """Frequency-independent coefficients for merger-ringdown amplitude."""

    fMs_RD: FloatLike
    gammaR: FloatLike
    gammaD2: FloatLike
    gammaD13: FloatLike
    fMs_AmpRDMin: FloatLike


class _InspiralAmpPlan(NamedTuple):
    """Frequency-independent coefficients for inspiral amplitude."""

    A0: FloatLike
    A2: FloatLike
    A3: FloatLike
    A4: FloatLike
    A5: FloatLike
    A6: FloatLike
    rho1: FloatLike
    rho2: FloatLike
    rho3: FloatLike


class _IntermediateAmpPlan(NamedTuple):
    """Frequency-independent coefficients for intermediate amplitude."""

    delta0: FloatLike
    delta1: FloatLike
    delta2: FloatLike
    delta3: FloatLike
    delta4: FloatLike


class _IMRPhenomXASAmpPlan(NamedTuple):
    """One request's frequency-independent piecewise amplitude state."""

    inspiral: _InspiralAmpPlan
    intermediate: _IntermediateAmpPlan
    mergerringdown: _MergerRingdownAmpPlan


class _IMRPhenomXASPhasePlan(NamedTuple):
    """One request's frequency-independent piecewise phase state."""

    total_mass_seconds: FloatLike
    eta: FloatLike
    f1_Ms: FloatLike
    f2_Ms: FloatLike
    inspiral: _InspiralPhasePlan
    intermediate: _IntermediatePhasePlan
    mergerringdown: _MergerRingdownPhasePlan
    scalar_inspiral: _InspiralPhasePlan
    scalar_intermediate: _IntermediatePhasePlan
    scalar_mergerringdown: _MergerRingdownPhasePlan
    alpha0: FloatLike
    alpha1: FloatLike
    beta0: FloatLike
    beta1: FloatLike
    scalar_region_dispatch: bool


class _CarrierAlignmentResult(NamedTuple):
    """Two exact XAS carrier results handed to one XPHM mode request."""

    phase_plan: object
    reference_frequency: object
    reference_phase: torch.Tensor
    ringdown_start_derivative: torch.Tensor


def _make_request_proof_plan_primitives():
    """Create sealed request scopes and same-arity phase-plan markers."""

    token = object()
    barrier = object()
    marker_owners = {}
    xas_target = None
    xphm_supported = None
    xphm_target = None
    active_proof = ContextVar(
        "pycbc_imrphenomxas_request_proof",
        default=None,
    )

    class _Proof:
        """One active, closure-private public-request execution scope."""

        __slots__ = (
            "token",
            "process_id",
            "thread_id",
            "region_pruning",
            "active",
        )

        def __init__(self, process_id, thread_id, region_pruning):
            self.token = token
            self.process_id = process_id
            self.thread_id = thread_id
            self.region_pruning = region_pruning
            self.active = True

        def __setattr__(self, name, value):
            if hasattr(self, name):
                if name == "active" and value is False:
                    object.__setattr__(self, name, False)
                    return
                raise AttributeError("request proof state is write-once")
            object.__setattr__(self, name, value)

    class _RequestScalarInspiralPhasePlan(_PrequalifiedInspiralPhasePlan):
        __slots__ = ()

    class _RequestScalarIntermediatePhasePlan(
        _PrequalifiedIntermediatePhasePlan
    ):
        __slots__ = ()

    class _RequestScalarMergerRingdownPhasePlan(
        _PrequalifiedMergerRingdownPhasePlan
    ):
        __slots__ = ()

    class _RequestScriptedInspiralPhasePlan(_ScriptedInspiralPhasePlan):
        __slots__ = ()

    class _RequestScriptedIntermediatePhasePlan(
        _ScriptedIntermediatePhasePlan
    ):
        __slots__ = ()

    class _RequestScriptedMergerRingdownPhasePlan(
        _ScriptedMergerRingdownPhasePlan
    ):
        __slots__ = ()

    class _RequestPhasePlanPruned(_IMRPhenomXASPhasePlan):
        __slots__ = ()

    class _RequestPhasePlanDense(_IMRPhenomXASPhasePlan):
        __slots__ = ()

    request_scalar_phase_types = (
        _RequestScalarInspiralPhasePlan,
        _RequestScalarIntermediatePhasePlan,
        _RequestScalarMergerRingdownPhasePlan,
    )
    request_top_phase_types = (
        _RequestPhasePlanPruned,
        _RequestPhasePlanDense,
    )

    def resolve(proof=None):
        candidate = active_proof.get() if proof is None else proof
        if (
            type(candidate) is _Proof
            and candidate.token is token
            and candidate.active
            and candidate.process_id == os.getpid()
            and candidate.thread_id == threading.get_ident()
            and active_proof.get() is candidate
        ):
            return candidate
        return None

    def current(proof=None):
        """Recognize only the active proof in this exact synchronous frame."""

        return resolve(proof) is not None

    def owned(plan, proof=None):
        candidate = resolve(proof)
        registry = marker_owners.get(candidate)
        return registry is not None and registry.get(id(plan)) is plan

    def register(plan, proof=None):
        candidate = resolve(proof)
        registry = marker_owners.get(candidate)
        if registry is None:
            return plan
        registry[id(plan)] = plan
        return plan

    def phase_scope_supported(params):
        """Sample every phase-only proof prerequisite at request entry."""

        if not _request_proof_plan_enabled():
            return False
        n_batch = params.get("n_batch")
        if (
            not _scripted_phase_ansatz_cpu_plain_request_parameters_supported(
                params
            )
            or (
                n_batch is not None
                and (type(n_batch) is not int or n_batch != 1)
            )
            or not _request_proof_lifecycle_supported()
        ):
            return False
        return (
            _phase_plan_enabled()
            and _scalar_derivative_plan_cse_enabled()
            and _exact_scalar_derivatives_enabled()
            and _python_inspiral_derivative_enabled()
            and _python_inspiral_derivative_bulk_io_enabled()
            and _scripted_phase_ansatz_cpu_enabled()
            and _inspiral_phase_host_scalars_enabled()
            and _packed_heaviside_masks_enabled()
        )

    def run_fixed_request(
        params,
        target,
        supported=None,
        fallback_target=None,
        fallback_supported=None,
    ):
        """Run one hard-coded production target behind a nested barrier."""

        frame = active_proof.set(barrier)
        proof = None
        try:
            proof_supported = phase_scope_supported(params)
            request_supported = (
                supported
                if proof_supported or fallback_supported is None
                else fallback_supported
            )
            if request_supported is not None and not request_supported(params):
                raise ValueError(
                    "unsupported parameters for native Torch IMRPhenomXPHM"
                )
            if proof_supported:
                proof = _Proof(
                    os.getpid(),
                    threading.get_ident(),
                    _region_pruning_enabled(),
                )
            if proof is None:
                return (
                    target(params)
                    if fallback_target is None
                    else fallback_target(params)
                )
            marker_owners[proof] = {}
            proof_frame = active_proof.set(proof)
            try:
                # Seal the public entry, not every nested module lookup: the
                # ordinary module implementation is the integrity boundary
                # and deliberately stays inspectable/debuggable.  The proof
                # object never enters that call graph; phase-only helpers
                # resolve this closure-private scope and recognize only
                # identity-owned marker instances.
                return target(params)
            finally:
                registry = marker_owners.pop(proof, None)
                if registry is not None:
                    registry.clear()
                proof.active = False
                active_proof.reset(proof_frame)
        finally:
            active_proof.reset(frame)

    def bind_xas_target(target):
        """Bind the exact XAS implementation once during module import."""

        nonlocal xas_target
        if xas_target is not None:
            raise RuntimeError("XAS request-proof target is already bound")
        if (
            type(target) is not type(run_fixed_request)
            or target.__module__ != __name__
            or target.__name__ != "_imrphenomxas_fd_torch_impl"
        ):
            raise TypeError("invalid XAS request-proof target")
        xas_target = target

    def bind_xphm_target(supported, target):
        """Bind the exact XPHM predicates and implementation once."""

        nonlocal xphm_supported, xphm_target
        if xphm_target is not None:
            raise RuntimeError("XPHM request-proof target is already bound")
        module = f"{__package__}.imrphenomxphm_torch"
        if (
            type(supported) is not type(run_fixed_request)
            or supported.__module__ != module
            or supported.__name__ != "imrphenomxphm_native_supported"
            or type(target) is not type(run_fixed_request)
            or target.__module__ != module
            or target.__name__ != "_imrphenomxphm_fd_torch"
        ):
            raise TypeError("invalid XPHM request-proof target")
        xphm_supported = supported
        xphm_target = target
        globals().pop("_bind_xphm_request_proof_target", None)

    def run_xas_request(params, fallback_target=None):
        """Run only the production XAS public implementation."""

        if xas_target is None:
            raise RuntimeError("XAS request-proof target is not bound")
        return run_fixed_request(
            params,
            xas_target,
            fallback_target=fallback_target,
        )

    def run_xphm_request(
        params,
        fallback_supported=None,
        fallback_target=None,
    ):
        """Run only the production XPHM public implementation."""

        if xphm_target is None or xphm_supported is None:
            raise RuntimeError("XPHM request-proof target is not bound")
        return run_fixed_request(
            params,
            xphm_target,
            supported=xphm_supported,
            fallback_target=fallback_target,
            fallback_supported=fallback_supported,
        )

    def phase_ready(proof):
        return current(proof)

    def amplitude_ready(proof):
        return False

    def qualify_scalar_phase(plan, proof):
        proof = resolve(proof)
        if proof is None:
            return plan
        if type(plan) is _InspiralPhasePlan:
            return register(_RequestScalarInspiralPhasePlan._make(plan), proof)
        if type(plan) is _IntermediatePhasePlan:
            return register(
                _RequestScalarIntermediatePhasePlan._make(plan), proof
            )
        if type(plan) is _MergerRingdownPhasePlan:
            return register(
                _RequestScalarMergerRingdownPhasePlan._make(plan), proof
            )
        return plan

    def qualify_phase_ansatzes(inspiral, intermediate, ringdown, proof):
        proof = resolve(proof)
        if proof is None:
            return None
        if (
            type(inspiral) is not _InspiralPhasePlan
            or type(intermediate) is not _IntermediatePhasePlan
            or type(ringdown) is not _MergerRingdownPhasePlan
        ):
            return None
        return (
            register(
                _RequestScriptedInspiralPhasePlan._make(inspiral), proof
            ),
            register(
                _RequestScriptedIntermediatePhasePlan._make(intermediate),
                proof,
            ),
            register(
                _RequestScriptedMergerRingdownPhasePlan._make(ringdown),
                proof,
            ),
        )

    def qualify_amp_regions(inspiral, ringdown, proof):
        return None

    def qualify_top(plan, proof):
        proof = resolve(proof)
        if proof is None:
            return plan
        if type(plan) is _IMRPhenomXASPhasePlan:
            plan_type = (
                _RequestPhasePlanPruned
                if proof.region_pruning
                else _RequestPhasePlanDense
            )
            return register(plan_type._make(plan), proof)
        return plan

    def scalar_phase_supported(plan, proof=None):
        return owned(plan, proof) and type(plan) in request_scalar_phase_types

    def inspiral_phase_supported(plan, proof=None):
        return (
            owned(plan, proof)
            and type(plan) is _RequestScalarInspiralPhasePlan
        )

    def amp_region_supported(plan, proof=None):
        return False

    def phase_plan_supported(plan, proof=None):
        return type(plan) is _IMRPhenomXASPhasePlan or (
            owned(plan, proof) and type(plan) in request_top_phase_types
        )

    def amp_plan_supported(plan, proof=None):
        return type(plan) is _IMRPhenomXASAmpPlan

    def ringdown_amp_plan_supported(plan, proof=None):
        return type(plan) is _MergerRingdownAmpPlan

    def top_qualified(plan, proof=None):
        return owned(plan, proof) and type(plan) in request_top_phase_types

    def phase_top_qualified(plan, proof=None):
        return owned(plan, proof) and type(plan) in request_top_phase_types

    def region_pruning_qualified(plan, proof=None):
        return owned(plan, proof) and type(plan) is _RequestPhasePlanPruned

    def unqualify_top(plan, proof=None):
        """Return ordinary same-schema carrier plans before scope exit."""

        if not owned(plan, proof):
            return plan

        def plain_phase_region(region):
            if type(region) is _RequestScriptedInspiralPhasePlan:
                return _ScriptedInspiralPhasePlan._make(region)
            if type(region) is _RequestScriptedIntermediatePhasePlan:
                return _ScriptedIntermediatePhasePlan._make(region)
            if type(region) is _RequestScriptedMergerRingdownPhasePlan:
                return _ScriptedMergerRingdownPhasePlan._make(region)
            if type(region) is _RequestScalarInspiralPhasePlan:
                return _PrequalifiedInspiralPhasePlan._make(region)
            if type(region) is _RequestScalarIntermediatePhasePlan:
                return _PrequalifiedIntermediatePhasePlan._make(region)
            if type(region) is _RequestScalarMergerRingdownPhasePlan:
                return _PrequalifiedMergerRingdownPhasePlan._make(region)
            return region

        if type(plan) in (_RequestPhasePlanPruned, _RequestPhasePlanDense):
            return _IMRPhenomXASPhasePlan(
                plan.total_mass_seconds,
                plan.eta,
                plan.f1_Ms,
                plan.f2_Ms,
                plain_phase_region(plan.inspiral),
                plain_phase_region(plan.intermediate),
                plain_phase_region(plan.mergerringdown),
                plain_phase_region(plan.scalar_inspiral),
                plain_phase_region(plan.scalar_intermediate),
                plain_phase_region(plan.scalar_mergerringdown),
                plan.alpha0,
                plan.alpha1,
                plan.beta0,
                plan.beta1,
                plan.scalar_region_dispatch,
            )
        return plan

    return (
        bind_xas_target,
        bind_xphm_target,
        run_xas_request,
        run_xphm_request,
        current,
        phase_ready,
        amplitude_ready,
        qualify_scalar_phase,
        qualify_phase_ansatzes,
        qualify_amp_regions,
        qualify_top,
        scalar_phase_supported,
        inspiral_phase_supported,
        amp_region_supported,
        phase_plan_supported,
        amp_plan_supported,
        ringdown_amp_plan_supported,
        top_qualified,
        phase_top_qualified,
        region_pruning_qualified,
        unqualify_top,
    )


(
    _bind_xas_request_proof_target,
    _bind_xphm_request_proof_target,
    _run_xas_request_proof_plan,
    _run_xphm_request_proof_plan,
    _request_proof_plan_current,
    _request_proof_phase_ready,
    _request_proof_amplitude_ready,
    _request_qualify_scalar_phase_plan,
    _request_qualify_phase_ansatzes,
    _request_qualify_amp_regions,
    _request_qualify_top_plan,
    _request_scalar_phase_plan_supported,
    _request_inspiral_phase_plan_supported,
    _request_amp_region_plan_supported,
    _imrphenomxas_phase_plan_type_supported,
    _imrphenomxas_amp_plan_type_supported,
    _imrphenomxas_ringdown_amp_plan_type_supported,
    _request_top_plan_qualified,
    _request_phase_plan_qualified,
    _request_region_pruning_qualified,
    _request_unqualify_top_plan,
) = _make_request_proof_plan_primitives()
del _make_request_proof_plan_primitives


class _IMRPhenomXASIntrinsicControls(NamedTuple):
    """One exact request-local copy of XAS intrinsic scalar controls."""

    mass1: torch.Tensor
    mass2: torch.Tensor
    spin1: torch.Tensor
    spin2: torch.Tensor
    mass1_seconds: torch.Tensor
    mass2_seconds: torch.Tensor
    total_mass_seconds: torch.Tensor
    eta: torch.Tensor
    delta: torch.Tensor
    mass_fraction1: torch.Tensor
    mass_fraction2: torch.Tensor
    effective_spin: torch.Tensor
    inspiral_spin: torch.Tensor
    merger_spin: torch.Tensor
    spin_difference: torch.Tensor
    theta_values: tuple[float, float, float, float] | None
    fit_values: tuple[float, float, float, float, float] | None


class _IMRPhenomXASInspiralPhaseHostScalars(NamedTuple):
    """Provenance for one exact request-local host-scalar phase build."""

    theta: torch.Tensor
    phase_coeffs: torch.Tensor
    phase_fit_rows: torch.Tensor
    meco_frequency: torch.Tensor
    intrinsic_controls: _IMRPhenomXASIntrinsicControls
    builder_values: tuple[float, float, float, float, float, float]


class _IMRPhenomXASInspiralAmpHostScalars(NamedTuple):
    """Provenance for one exact request-local host-scalar amp build."""

    theta: torch.Tensor
    amp_coeffs: torch.Tensor
    amp_fit_rows: torch.Tensor
    meco_frequency: torch.Tensor
    isco_frequency: torch.Tensor
    intrinsic_controls: _IMRPhenomXASIntrinsicControls | None
    chip: float
    builder_values: tuple[float, ...]


class _InspiralAmpHostIntrinsicScalars(NamedTuple):
    """Minimal host controls consumed by the unchanged amp expression."""

    mass1: float
    mass2: float
    spin1: float
    spin2: float
    eta: float
    delta: float
    inspiral_spin: float
    spin_difference: float


_INSPIRAL_PHASE_HOST_TENSOR_POSITIONS = (
    2,
    3,
    4,
    6,
    7,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
)
_INSPIRAL_PHASE_HOST_PLAN_ITEMS = operator.itemgetter(
    *_INSPIRAL_PHASE_HOST_TENSOR_POSITIONS
)


class _IMRPhenomXASCarrierPlans(NamedTuple):
    """Exact plans already prepared while evaluating one XAS carrier."""

    phase: _IMRPhenomXASPhasePlan | None
    amplitude: _IMRPhenomXASAmpPlan | None


class _IMRPhenomXASIntrinsicPlanBundle(NamedTuple):
    """Private frequency-independent XAS state owned by one cache entry."""

    intrinsic_bits: tuple[bytes, ...]
    phase_rows: torch.Tensor
    amp_rows: torch.Tensor
    aligned_cutoff: tuple[torch.Tensor, ...]
    cutoff: tuple[torch.Tensor, ...]
    phase_plan: _IMRPhenomXASPhasePlan
    amp_plan: _IMRPhenomXASAmpPlan


class _IMRPhenomXASIntrinsicPlanCacheEntry(NamedTuple):
    """Sealed key, tensor provenance, and value for one LRU entry."""

    key: tuple[Any, ...]
    phase_coeffs: torch.Tensor
    amp_coeffs: torch.Tensor
    tensor_provenance: tuple[tuple[Any, ...], ...]
    byte_size: int
    bundle: _IMRPhenomXASIntrinsicPlanBundle


class _IMRPhenomXASIntrinsicPlanFastCacheEntry(NamedTuple):
    """Opt-in entry with the validated tensor traversal preflattened."""

    key: tuple[Any, ...]
    phase_coeffs: torch.Tensor
    amp_coeffs: torch.Tensor
    tensor_leaves: tuple[torch.Tensor, ...]
    tensor_provenance: tuple[tuple[Any, ...], ...]
    byte_size: int
    bundle: _IMRPhenomXASIntrinsicPlanBundle
    bundle_identity: int
    tensor_leaves_identity: int
    tensor_provenance_identity: int


class _IMRPhenomXASIntrinsicPlanValidatedHit(NamedTuple):
    """One cache hit plus its short-lived consumer-validation token."""

    bundle: _IMRPhenomXASIntrinsicPlanBundle
    token: object


def _make_intrinsic_plan_cache_fast_hit_primitives():
    """Create unforgeable, same-request cache-hit validation tokens."""

    owner = object()

    class _ValidatedHit:
        """Bind one fully checked LRU hit to its immediate consumer."""

        __slots__ = (
            "owner",
            "process_id",
            "thread_id",
            "entry",
            "key",
            "bundle",
            "inputs",
            "theta_intrinsic",
            "theta_bits",
            "theta_provenance",
            "tensor_leaves",
            "phase_coeffs",
            "amp_coeffs",
            "phase_table_key",
            "amp_table_key",
            "runtime_key",
            "gate_profile",
            "active",
        )

        def __init__(
            self,
            entry,
            key,
            inputs,
            theta_intrinsic,
            theta_bits,
            theta_provenance,
            phase_coeffs,
            amp_coeffs,
        ):
            object.__setattr__(self, "owner", owner)
            object.__setattr__(self, "process_id", os.getpid())
            object.__setattr__(self, "thread_id", threading.get_ident())
            object.__setattr__(self, "entry", entry)
            object.__setattr__(self, "key", entry.key)
            object.__setattr__(self, "bundle", entry.bundle)
            object.__setattr__(self, "inputs", inputs)
            object.__setattr__(self, "theta_intrinsic", theta_intrinsic)
            object.__setattr__(self, "theta_bits", theta_bits)
            object.__setattr__(self, "theta_provenance", theta_provenance)
            object.__setattr__(
                self,
                "tensor_leaves",
                entry.tensor_leaves,
            )
            object.__setattr__(self, "phase_coeffs", phase_coeffs)
            object.__setattr__(self, "amp_coeffs", amp_coeffs)
            object.__setattr__(self, "phase_table_key", key[9])
            object.__setattr__(self, "amp_table_key", key[10])
            object.__setattr__(self, "runtime_key", key[11])
            object.__setattr__(self, "gate_profile", key[12])
            object.__setattr__(self, "active", True)

        def __setattr__(self, name, value):
            if name == "active" and value is False and self.active:
                object.__setattr__(self, name, False)
                return
            raise AttributeError("validated cache-hit state is immutable")

    def issue(
        entry,
        key,
        inputs,
        theta_intrinsic,
        theta_bits,
        phase_coeffs,
        amp_coeffs,
    ):
        """Issue only after the ordinary cache-hit validator succeeds."""

        try:
            theta_provenance = _intrinsic_plan_cache_tensor_provenance(
                theta_intrinsic
            )
            if (
                not _intrinsic_plan_cache_fast_hit_enabled()
                or type(entry)
                is not _IMRPhenomXASIntrinsicPlanFastCacheEntry
                or type(entry.key) is not tuple
                or entry.key != key
                or entry.bundle.intrinsic_bits[:4] != theta_bits
                or theta_provenance is None
            ):
                return None
            return _ValidatedHit(
                entry,
                key,
                inputs,
                theta_intrinsic,
                theta_bits,
                theta_provenance,
                phase_coeffs,
                amp_coeffs,
            )
        except Exception:
            return None

    def request_binding_current(token, phase_coeffs, amp_coeffs) -> bool:
        """Refresh mutable runtime/table/gate parts of the validated key."""

        try:
            key = token.key
            return (
                key[0] == _INTRINSIC_PLAN_CACHE_MODEL_RELEASE
                and key[1] == torch.__version__
                and key[2] == getattr(torch.version, "cuda", None)
                and key[3] == getattr(torch.version, "hip", None)
                and key[5] == token.inputs.device.type
                and key[6] == token.inputs.device.index
                and key[7] is token.inputs.real_dtype
                and key[8] is token.inputs.complex_dtype
                and _intrinsic_plan_cache_table_key(
                    phase_coeffs,
                    (13, 49),
                    source=_PHASE_FIT_COEFFICIENT_SOURCE,
                    current=(
                        IMRPhenomX_utils._PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER
                    ),
                )
                == token.phase_table_key
                and _intrinsic_plan_cache_table_key(
                    amp_coeffs,
                    (7, 42),
                    source=_AMP_FIT_COEFFICIENT_SOURCE,
                    current=(
                        IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER
                    ),
                )
                == token.amp_table_key
                and _intrinsic_plan_cache_runtime_key(token.inputs)
                == token.runtime_key
                and _intrinsic_plan_cache_gate_profile()
                == token.gate_profile
            )
        except Exception:
            return False

    def current(
        token,
        bundle,
        theta_intrinsic,
        phase_coeffs,
        amp_coeffs,
    ) -> bool:
        """Recheck cheap request bindings without rescanning the plan tree."""

        try:
            if (
                type(token) is not _ValidatedHit
                or token.owner is not owner
                or not token.active
                or token.process_id != os.getpid()
                or token.thread_id != threading.get_ident()
                or token.entry.key is not token.key
                or token.entry.bundle is not token.bundle
                or token.bundle is not bundle
                or token.inputs is None
                or token.theta_intrinsic is not theta_intrinsic
                or token.phase_coeffs is not phase_coeffs
                or token.amp_coeffs is not amp_coeffs
                or token.entry.phase_coeffs is not phase_coeffs
                or token.entry.amp_coeffs is not amp_coeffs
                or token.phase_table_key != token.key[9]
                or token.amp_table_key != token.key[10]
                or token.runtime_key != token.key[11]
                or token.gate_profile != token.key[12]
                or token.theta_provenance
                != _intrinsic_plan_cache_tensor_provenance(theta_intrinsic)
                or token.theta_bits
                != _intrinsic_plan_cache_theta_bits(theta_intrinsic)
                or token.bundle.intrinsic_bits[:4] != token.theta_bits
                or not _intrinsic_plan_cache_flat_leaves_supported(
                    token.tensor_leaves,
                    token.entry.tensor_provenance,
                    token.entry.byte_size,
                )
            ):
                return False
            # This host-only key refresh detects table rebinding, runtime-state
            # changes, gate changes, and free-threaded/transform entry.  The
            # expensive immutable-bundle tensor traversal already succeeded in
            # the cache lookup immediately above this consumer.
            return request_binding_current(token, phase_coeffs, amp_coeffs)
        except Exception:
            return False

    def retire(token) -> None:
        """Make a token unusable at the end of its one public request."""

        if type(token) is _ValidatedHit and token.owner is owner:
            try:
                token.active = False
            except Exception:
                pass

    return issue, current, retire


(
    _issue_intrinsic_plan_cache_validated_hit,
    _intrinsic_plan_cache_validated_hit_current,
    _retire_intrinsic_plan_cache_validated_hit,
) = _make_intrinsic_plan_cache_fast_hit_primitives()
del _make_intrinsic_plan_cache_fast_hit_primitives


class _PhasePlanCudaSolveGraphState(NamedTuple):
    """Owned inputs and outputs for one CUDA phase-plan graph."""

    static_theta: torch.Tensor
    phase_coeffs: torch.Tensor
    static_chip: torch.Tensor
    static_final_spin: torch.Tensor
    static_fit_rows: torch.Tensor | None
    capture_stream: Any
    graph: Any
    packed_plan: torch.Tensor
    graph_info: torch.Tensor
    template: _IMRPhenomXASPhasePlan


class _CudaGraphPhaseAnsatzState(NamedTuple):
    """Owned static buffers for one exact phase-ansatz CUDA graph."""

    graph: Any
    static_arguments: tuple[Any, ...]
    copy_positions: tuple[int, ...]
    static_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    replay_stream_pointer: int
    capture_stream: Any


_PHASE_PLAN_CUDA_SOLVE_GRAPH_CACHE = {}
_PHASE_PLAN_CUDA_SOLVE_GRAPH_FAILURES = set()
_PHASE_PLAN_CUDA_SOLVE_GRAPH_LOCK = threading.Lock()
_PHASE_PLAN_CUDA_SOLVE_GRAPH_MAX_ENTRIES = 16

_PHASE_PLAN_TORCHSCRIPT_TRACE_CACHE = OrderedDict()
_PHASE_PLAN_TORCHSCRIPT_TRACE_MISSING = object()
_PHASE_PLAN_TORCHSCRIPT_TRACE_PID = os.getpid()
_PHASE_PLAN_TORCHSCRIPT_TRACE_LOCK = threading.Lock()
_PHASE_PLAN_TORCHSCRIPT_TRACE_MAX_ENTRIES = 4

_PACKED_FREQUENCY_PLAN_WIDTH = 66
_PACKED_FREQUENCY_PLAN_TRACE_CACHE = OrderedDict()
_PACKED_FREQUENCY_PLAN_TRACE_FAILURES = OrderedDict()
_PACKED_FREQUENCY_PLAN_TRACE_LOCK = threading.Lock()
_PACKED_FREQUENCY_PLAN_TRACE_PID = os.getpid()
_PACKED_FREQUENCY_PLAN_TRACE_MAX_ENTRIES = 8

_INTRINSIC_PLAN_CACHE = OrderedDict()
_INTRINSIC_PLAN_CACHE_LOCK = threading.Lock()
_INTRINSIC_PLAN_CACHE_PID = os.getpid()
_INTRINSIC_PLAN_CACHE_MAX_ENTRIES = 8
_INTRINSIC_PLAN_CACHE_MAX_BYTES = 2 * 1024 * 1024
_INTRINSIC_PLAN_CACHE_BYTES = 0
_INTRINSIC_PLAN_CACHE_HITS = 0
_INTRINSIC_PLAN_CACHE_MISSES = 0
_INTRINSIC_PLAN_CACHE_EVICTIONS = 0
_INTRINSIC_PLAN_CACHE_OVERSIZED = 0
_INTRINSIC_PLAN_CACHE_MODEL_RELEASE = ("IMRPhenomXAS", 1)

_SCRIPTED_PHASE_ANSATZ_CPU_EXECUTOR = None
_SCRIPTED_PHASE_ANSATZ_CPU_FAILED = False
_SCRIPTED_PHASE_ANSATZ_CPU_PID = os.getpid()
_SCRIPTED_PHASE_ANSATZ_CPU_LOCK = threading.Lock()

_CUDA_GRAPH_PHASE_ANSATZ_CACHE = OrderedDict()
_CUDA_GRAPH_PHASE_ANSATZ_FAILURES = OrderedDict()
_CUDA_GRAPH_PHASE_ANSATZ_PID = os.getpid()
_CUDA_GRAPH_PHASE_ANSATZ_LOCK = threading.RLock()
_CUDA_GRAPH_PHASE_ANSATZ_MAX_ENTRIES = 8

# The static CPython lane consumes these fractional powers in this exact order.
# Keeping all 16 sites, including numerically equal-looking exponents, preserves
# the float value produced by each original Python exponent expression.
_PYTHON_INTERMEDIATE_AMP_POWER_BASE_ARGUMENTS = (
    0,  # FMs1 ** (2 / 3)
    0,  # FMs1 ** (4 / 3)
    0,  # FMs1 ** (5 / 3)
    0,  # FMs1 ** (7 / 3)
    0,  # FMs1 ** (8 / 3)
    0,  # FMs1 ** ((2 / 3) - 1)
    0,  # FMs1 ** ((4 / 3) - 1)
    0,  # FMs1 ** ((5 / 3) - 1)
    0,  # FMs1 ** ((7 / 3) - 1)
    0,  # FMs1 ** ((8 / 3) - 1)
    0,  # FMs1 ** (1 / 6)
    0,  # FMs1 ** (7 / 6)
    1,  # FMs4 ** (1 / 6)
    1,  # FMs4 ** (7 / 6)
    0,  # FMs1 ** (-7 / 6)
    1,  # FMs4 ** (-7 / 6)
)
_PYTHON_INTERMEDIATE_AMP_POWER_EXPONENTS = (
    2.0 / 3.0,
    4.0 / 3.0,
    5.0 / 3.0,
    7.0 / 3.0,
    8.0 / 3.0,
    (2.0 / 3.0) - 1.0,
    (4.0 / 3.0) - 1.0,
    (5.0 / 3.0) - 1.0,
    (7.0 / 3.0) - 1.0,
    (8.0 / 3.0) - 1.0,
    1.0 / 6.0,
    7.0 / 6.0,
    1.0 / 6.0,
    7.0 / 6.0,
    -7.0 / 6,
    -7.0 / 6,
)
_PYTHON_INTERMEDIATE_AMP_EXECUTOR = None
_PYTHON_INTERMEDIATE_AMP_CALIBRATED = False
_PYTHON_INTERMEDIATE_AMP_FAILED = False
_PYTHON_INTERMEDIATE_AMP_PID = None
_PYTHON_INTERMEDIATE_AMP_LOCK = threading.Lock()
_PYTHON_INSPIRAL_DERIVATIVE_EXECUTOR = None
_PYTHON_INSPIRAL_DERIVATIVE_CALIBRATED = False
_PYTHON_INSPIRAL_DERIVATIVE_FAILED = False
_PYTHON_INSPIRAL_DERIVATIVE_PID = None
_PYTHON_INSPIRAL_DERIVATIVE_LOCK = threading.Lock()


def _reset_amp_fit_coefficient_rows_after_fork() -> None:
    """Discard an inherited row cache and its possibly locked mutex."""

    global _AMP_FIT_COEFFICIENT_ROWS_CACHE
    global _AMP_FIT_COEFFICIENT_ROWS_LOCK

    _AMP_FIT_COEFFICIENT_ROWS_CACHE = None
    _AMP_FIT_COEFFICIENT_ROWS_LOCK = threading.Lock()


def _reset_python_intermediate_amp_after_fork() -> None:
    """Discard inherited calibration and its possibly locked mutex in a child."""

    global _PYTHON_INTERMEDIATE_AMP_EXECUTOR
    global _PYTHON_INTERMEDIATE_AMP_CALIBRATED
    global _PYTHON_INTERMEDIATE_AMP_FAILED
    global _PYTHON_INTERMEDIATE_AMP_PID
    global _PYTHON_INTERMEDIATE_AMP_LOCK

    _PYTHON_INTERMEDIATE_AMP_EXECUTOR = None
    _PYTHON_INTERMEDIATE_AMP_CALIBRATED = False
    _PYTHON_INTERMEDIATE_AMP_FAILED = False
    _PYTHON_INTERMEDIATE_AMP_PID = os.getpid()
    _PYTHON_INTERMEDIATE_AMP_LOCK = threading.Lock()


def _reset_python_inspiral_derivative_after_fork() -> None:
    """Discard inherited scalar-lane state and mutex in a child process."""

    global _PYTHON_INSPIRAL_DERIVATIVE_EXECUTOR
    global _PYTHON_INSPIRAL_DERIVATIVE_CALIBRATED
    global _PYTHON_INSPIRAL_DERIVATIVE_FAILED
    global _PYTHON_INSPIRAL_DERIVATIVE_PID
    global _PYTHON_INSPIRAL_DERIVATIVE_LOCK

    _PYTHON_INSPIRAL_DERIVATIVE_EXECUTOR = None
    _PYTHON_INSPIRAL_DERIVATIVE_CALIBRATED = False
    _PYTHON_INSPIRAL_DERIVATIVE_FAILED = False
    _PYTHON_INSPIRAL_DERIVATIVE_PID = os.getpid()
    _PYTHON_INSPIRAL_DERIVATIVE_LOCK = threading.Lock()


def _reset_packed_frequency_plan_trace_after_fork() -> None:
    """Discard inherited TorchScript modules and their possibly locked mutex."""

    global _PACKED_FREQUENCY_PLAN_TRACE_CACHE
    global _PACKED_FREQUENCY_PLAN_TRACE_FAILURES
    global _PACKED_FREQUENCY_PLAN_TRACE_LOCK
    global _PACKED_FREQUENCY_PLAN_TRACE_PID

    _PACKED_FREQUENCY_PLAN_TRACE_CACHE = OrderedDict()
    _PACKED_FREQUENCY_PLAN_TRACE_FAILURES = OrderedDict()
    _PACKED_FREQUENCY_PLAN_TRACE_LOCK = threading.Lock()
    _PACKED_FREQUENCY_PLAN_TRACE_PID = os.getpid()


def _reset_intrinsic_plan_cache_after_fork() -> None:
    """Discard inherited tensors and replace the possibly held cache mutex."""

    global _INTRINSIC_PLAN_CACHE
    global _INTRINSIC_PLAN_CACHE_BYTES
    global _INTRINSIC_PLAN_CACHE_EVICTIONS
    global _INTRINSIC_PLAN_CACHE_HITS
    global _INTRINSIC_PLAN_CACHE_LOCK
    global _INTRINSIC_PLAN_CACHE_MISSES
    global _INTRINSIC_PLAN_CACHE_OVERSIZED
    global _INTRINSIC_PLAN_CACHE_PID

    _INTRINSIC_PLAN_CACHE = OrderedDict()
    _INTRINSIC_PLAN_CACHE_BYTES = 0
    _INTRINSIC_PLAN_CACHE_HITS = 0
    _INTRINSIC_PLAN_CACHE_MISSES = 0
    _INTRINSIC_PLAN_CACHE_EVICTIONS = 0
    _INTRINSIC_PLAN_CACHE_OVERSIZED = 0
    _INTRINSIC_PLAN_CACHE_LOCK = threading.Lock()
    _INTRINSIC_PLAN_CACHE_PID = os.getpid()


def _reset_phase_plan_torchscript_trace_after_fork() -> None:
    """Discard inherited phase-plan programs and their possibly held lock."""

    global _PHASE_PLAN_TORCHSCRIPT_TRACE_CACHE
    global _PHASE_PLAN_TORCHSCRIPT_TRACE_PID
    global _PHASE_PLAN_TORCHSCRIPT_TRACE_LOCK

    _PHASE_PLAN_TORCHSCRIPT_TRACE_CACHE = OrderedDict()
    _PHASE_PLAN_TORCHSCRIPT_TRACE_PID = os.getpid()
    _PHASE_PLAN_TORCHSCRIPT_TRACE_LOCK = threading.Lock()


def _reset_scripted_phase_ansatz_cpu_after_fork() -> None:
    """Discard inherited TorchScript functions and their mutex in a child."""

    global _SCRIPTED_PHASE_ANSATZ_CPU_EXECUTOR
    global _SCRIPTED_PHASE_ANSATZ_CPU_FAILED
    global _SCRIPTED_PHASE_ANSATZ_CPU_PID
    global _SCRIPTED_PHASE_ANSATZ_CPU_LOCK

    _SCRIPTED_PHASE_ANSATZ_CPU_EXECUTOR = None
    _SCRIPTED_PHASE_ANSATZ_CPU_FAILED = False
    _SCRIPTED_PHASE_ANSATZ_CPU_PID = os.getpid()
    _SCRIPTED_PHASE_ANSATZ_CPU_LOCK = threading.Lock()


def _reset_cuda_graph_phase_ansatz_after_fork() -> None:
    """Discard inherited CUDA graphs and their possibly locked mutex."""

    global _CUDA_GRAPH_PHASE_ANSATZ_CACHE
    global _CUDA_GRAPH_PHASE_ANSATZ_FAILURES
    global _CUDA_GRAPH_PHASE_ANSATZ_PID
    global _CUDA_GRAPH_PHASE_ANSATZ_LOCK

    _CUDA_GRAPH_PHASE_ANSATZ_CACHE = OrderedDict()
    _CUDA_GRAPH_PHASE_ANSATZ_FAILURES = OrderedDict()
    _CUDA_GRAPH_PHASE_ANSATZ_PID = os.getpid()
    _CUDA_GRAPH_PHASE_ANSATZ_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        after_in_child=_reset_amp_fit_coefficient_rows_after_fork
    )
    os.register_at_fork(after_in_child=_reset_python_intermediate_amp_after_fork)
    os.register_at_fork(
        after_in_child=_reset_python_inspiral_derivative_after_fork
    )
    os.register_at_fork(
        after_in_child=_reset_packed_frequency_plan_trace_after_fork
    )
    os.register_at_fork(after_in_child=_reset_intrinsic_plan_cache_after_fork)
    os.register_at_fork(
        after_in_child=_reset_phase_plan_torchscript_trace_after_fork
    )
    os.register_at_fork(
        after_in_child=_reset_scripted_phase_ansatz_cpu_after_fork
    )
    os.register_at_fork(
        after_in_child=_reset_cuda_graph_phase_ansatz_after_fork
    )


_XAS_MODE_POLARIZATION_FACTOR = math.sqrt(5.0 / (16.0 * PI))
_INSPIRAL_PHASE_NORMALIZATION = -(3.0 * PI ** (-5.0 / 3.0)) / 128.0


def _phase_plan_enabled() -> bool:
    """Return the strict phase-plan execution switch."""

    value = os.environ.get(_PHASE_PLAN_ENV)
    return False if value is None else _parse_switch(_PHASE_PLAN_ENV, value)


def _fixed_schema_phase_plan_enabled() -> bool:
    """Return the independent fixed-schema phase-plan debug switch."""

    value = os.environ.get(_FIXED_SCHEMA_PHASE_PLAN_ENV)
    return (
        False
        if value is None
        else _parse_switch(_FIXED_SCHEMA_PHASE_PLAN_ENV, value)
    )


def _phase_plan_cuda_solve_graph_enabled() -> bool:
    """Return the strict, off-by-default CUDA solve-graph switch."""

    value = os.environ.get(_PHASE_PLAN_CUDA_SOLVE_GRAPH_ENV)
    return (
        False
        if value is None
        else _parse_switch(_PHASE_PLAN_CUDA_SOLVE_GRAPH_ENV, value)
    )


def _phase_plan_torchscript_trace_enabled() -> bool:
    """Return the independent, strict, off-by-default trace switch."""

    value = os.environ.get(_PHASE_PLAN_TORCHSCRIPT_TRACE_ENV)
    return (
        False
        if value is None
        else _parse_switch(_PHASE_PLAN_TORCHSCRIPT_TRACE_ENV, value)
    )


def _scalar_region_dispatch_enabled() -> bool:
    """Return the strict scalar phase dispatch switch."""

    value = os.environ.get(_SCALAR_REGION_DISPATCH_ENV)
    return False if value is None else _parse_switch(_SCALAR_REGION_DISPATCH_ENV, value)


def _amp_plan_enabled() -> bool:
    """Return the strict amplitude-plan execution switch."""

    value = os.environ.get(_AMP_PLAN_ENV)
    return False if value is None else _parse_switch(_AMP_PLAN_ENV, value)


def _python_intermediate_amp_enabled() -> bool:
    """Return the strict, off-by-default binary64-executor switch."""

    value = os.environ.get(_PYTHON_INTERMEDIATE_AMP_ENV)
    return (
        False
        if value is None
        else _parse_switch(_PYTHON_INTERMEDIATE_AMP_ENV, value)
    )


def _python_inspiral_derivative_enabled() -> bool:
    """Return the strict, off-by-default scalar reverse-pass switch."""

    value = os.environ.get(_PYTHON_INSPIRAL_DERIVATIVE_ENV)
    return (
        False
        if value is None
        else _parse_switch(_PYTHON_INSPIRAL_DERIVATIVE_ENV, value)
    )


def _python_inspiral_derivative_bulk_io_enabled() -> bool:
    """Return the strict fixed-schema scalar-I/O switch."""

    value = os.environ.get(_PYTHON_INSPIRAL_DERIVATIVE_BULK_IO_ENV)
    return (
        True
        if value is None
        else _parse_switch(_PYTHON_INSPIRAL_DERIVATIVE_BULK_IO_ENV, value)
    )


def _inspiral_phase_host_scalars_enabled() -> bool:
    """Return the strict host-scalar phase-plan switch."""

    value = os.environ.get(_INSPIRAL_PHASE_HOST_SCALARS_ENV)
    return (
        True
        if value is None
        else _parse_switch(_INSPIRAL_PHASE_HOST_SCALARS_ENV, value)
    )


def _inspiral_amp_host_scalars_enabled() -> bool:
    """Return the strict host-scalar amp-plan switch."""

    value = os.environ.get(_INSPIRAL_AMP_HOST_SCALARS_ENV)
    return (
        True
        if value is None
        else _parse_switch(_INSPIRAL_AMP_HOST_SCALARS_ENV, value)
    )


def _derived_power_reuse_enabled() -> bool:
    """Return the strict switch for exact request-local power reuse."""

    value = os.environ.get(_DERIVED_POWER_REUSE_ENV)
    return (
        True
        if value is None
        else _parse_switch(_DERIVED_POWER_REUSE_ENV, value)
    )


def _region_pruning_enabled() -> bool:
    """Return the strict piecewise-evaluation switch."""

    value = os.environ.get(_REGION_PRUNING_ENV)
    return False if value is None else _parse_switch(_REGION_PRUNING_ENV, value)


def _exact_scalar_derivatives_enabled() -> bool:
    """Return the strict exact-derivative switch."""

    value = os.environ.get(_EXACT_SCALAR_DERIVATIVES_ENV)
    return (
        True if value is None else _parse_switch(_EXACT_SCALAR_DERIVATIVES_ENV, value)
    )


def _scalar_derivative_plan_cse_enabled() -> bool:
    """Return the strict scalar-plan CSE switch."""

    value = os.environ.get(_SCALAR_DERIVATIVE_PLAN_CSE_ENV)
    return (
        True
        if value is None
        else _parse_switch(_SCALAR_DERIVATIVE_PLAN_CSE_ENV, value)
    )


def _exact_scalar_amp_derivatives_enabled() -> bool:
    """Return the strict switch for exact amplitude-boundary derivatives."""

    value = os.environ.get(_EXACT_SCALAR_AMP_DERIVATIVES_ENV)
    return (
        True
        if value is None
        else _parse_switch(_EXACT_SCALAR_AMP_DERIVATIVES_ENV, value)
    )


def _phase_plan_bulk_collocation_enabled() -> bool:
    """Return the strict switch for packed phase collocation algebra."""

    value = os.environ.get(_PHASE_PLAN_BULK_COLLOCATION_ENV)
    return (
        True
        if value is None
        else _parse_switch(_PHASE_PLAN_BULK_COLLOCATION_ENV, value)
    )


def _phase_fit_python_scalars_enabled() -> bool:
    """Return the strict switch for CPU scalar phase-fit evaluation."""

    value = os.environ.get(_PHASE_FIT_PYTHON_SCALARS_ENV)
    return (
        True
        if value is None
        else _parse_switch(_PHASE_FIT_PYTHON_SCALARS_ENV, value)
    )


def _phase_fit_native_iterator_enabled() -> bool:
    """Return the strict switch for native fixed-row phase-fit iteration."""

    value = os.environ.get(_PHASE_FIT_NATIVE_ITERATOR_ENV)
    return (
        True
        if value is None
        else _parse_switch(_PHASE_FIT_NATIVE_ITERATOR_ENV, value)
    )


def _amp_fit_python_scalars_enabled() -> bool:
    """Return the strict switch for CPU scalar amplitude-fit evaluation."""

    value = os.environ.get(_AMP_FIT_PYTHON_SCALARS_ENV)
    return (
        True
        if value is None
        else _parse_switch(_AMP_FIT_PYTHON_SCALARS_ENV, value)
    )


def _cuda_amp_host_pack_enabled() -> bool:
    """Return the independent, strict CUDA amplitude host-pack switch."""

    value = os.environ.get(_CUDA_AMP_HOST_PACK_ENV)
    return (
        False
        if value is None
        else _parse_switch(_CUDA_AMP_HOST_PACK_ENV, value)
    )


def _amp_fit_native_iterator_enabled() -> bool:
    """Return the strict switch for native fixed-row amplitude-fit iteration."""

    value = os.environ.get(_AMP_FIT_NATIVE_ITERATOR_ENV)
    return (
        True
        if value is None
        else _parse_switch(_AMP_FIT_NATIVE_ITERATOR_ENV, value)
    )


def _packed_heaviside_masks_enabled() -> bool:
    """Return the strict switch for native five-mask piecewise lanes."""

    value = os.environ.get(_PACKED_HEAVISIDE_MASKS_ENV)
    return (
        True
        if value is None
        else _parse_switch(_PACKED_HEAVISIDE_MASKS_ENV, value)
    )


def _packed_frequency_plan_enabled() -> bool:
    """Return the strict packed frequency-dataflow switch."""

    value = os.environ.get(_PACKED_FREQUENCY_PLAN_ENV)
    return (
        True
        if value is None
        else _parse_switch(_PACKED_FREQUENCY_PLAN_ENV, value)
    )


def _packed_frequency_plan_torchscript_trace_enabled() -> bool:
    """Return the strict, off-by-default cached TorchScript trace switch."""

    value = os.environ.get(_PACKED_FREQUENCY_PLAN_TORCHSCRIPT_TRACE_ENV)
    return (
        False
        if value is None
        else _parse_switch(
            _PACKED_FREQUENCY_PLAN_TORCHSCRIPT_TRACE_ENV,
            value,
        )
    )


def _scripted_phase_ansatz_cpu_enabled() -> bool:
    """Return the strict scripted phase-ansatz switch."""

    value = os.environ.get(_SCRIPTED_PHASE_ANSATZ_CPU_ENV)
    return (
        True
        if value is None
        else _parse_switch(_SCRIPTED_PHASE_ANSATZ_CPU_ENV, value)
    )


def _cuda_graph_phase_ansatz_enabled() -> bool:
    """Return the independent, strict, off-by-default CUDA-graph switch."""

    value = os.environ.get(_CUDA_GRAPH_PHASE_ANSATZ_ENV)
    return (
        False
        if value is None
        else _parse_switch(_CUDA_GRAPH_PHASE_ANSATZ_ENV, value)
    )


def _request_proof_plan_enabled() -> bool:
    """Return the independent, strict request-proof debug switch."""

    value = os.environ.get(_REQUEST_PROOF_PLAN_ENV)
    return (
        False
        if value is None
        else _parse_switch(_REQUEST_PROOF_PLAN_ENV, value)
    )


def _intrinsic_control_reuse_enabled() -> bool:
    """Return the strict intrinsic-control reuse switch."""

    value = os.environ.get(_INTRINSIC_CONTROL_REUSE_ENV)
    return (
        True
        if value is None
        else _parse_switch(_INTRINSIC_CONTROL_REUSE_ENV, value)
    )


def _intrinsic_plan_cache_enabled() -> bool:
    """Return the strict cross-request plan-cache switch."""

    value = os.environ.get(_INTRINSIC_PLAN_CACHE_ENV)
    return (
        True
        if value is None
        else _parse_switch(_INTRINSIC_PLAN_CACHE_ENV, value)
    )


def _intrinsic_plan_cache_fast_hit_enabled() -> bool:
    """Return the strict switch for immediate validated-hit consumption."""

    value = os.environ.get(_INTRINSIC_PLAN_CACHE_FAST_HIT_ENV)
    return (
        True
        if value is None
        else _parse_switch(_INTRINSIC_PLAN_CACHE_FAST_HIT_ENV, value)
    )


def _packed_cutoff_reuse_enabled() -> bool:
    """Return the strict packed cutoff-reuse switch."""

    value = os.environ.get(_PACKED_CUTOFF_REUSE_ENV)
    return (
        True
        if value is None
        else _parse_switch(_PACKED_CUTOFF_REUSE_ENV, value)
    )


def _packed_heaviside_masks_runtime_supported() -> bool:
    """Reject transforms that can observe replacing five calls with one lane."""

    for function in (
        getattr(torch.jit, "is_scripting", None),
        getattr(torch.jit, "is_tracing", None),
        getattr(getattr(torch, "compiler", None), "is_compiling", None),
        getattr(getattr(torch, "_dynamo", None), "is_compiling", None),
    ):
        if function is None:
            return False
        try:
            if function():
                return False
        except Exception:
            return False

    tracing_state = getattr(getattr(torch, "_C", None), "_get_tracing_state", None)
    if tracing_state is None:
        return False
    try:
        if tracing_state() is not None:
            return False
    except Exception:
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
        try:
            legacy_cpu = getattr(torch, "is_autocast_cpu_enabled", None)
            if autocast_enabled() or legacy_cpu is None or legacy_cpu():
                return False
        except Exception:
            return False
    except Exception:
        return False
    return True


def _fit_tensor_version_is_zero(value) -> bool:
    """Read a tensor mutation version without trusting it to be available."""

    try:
        return value._version == 0
    except (AttributeError, RuntimeError, TypeError):
        return False


def _fit_theta_values_supported(theta_values) -> bool:
    """Accept one physical four-float public intrinsic tuple."""

    return (
        type(theta_values) is tuple
        and len(theta_values) == 4
        and all(type(value) is float for value in theta_values)
        and all(math.isfinite(value) for value in theta_values)
        and theta_values[0] > 0.0
        and theta_values[1] > 0.0
        and -1.0 <= theta_values[2] <= 1.0
        and -1.0 <= theta_values[3] <= 1.0
    )


def _cuda_fit_runtime_supported(device) -> bool:
    """Accept only ordinary eager execution on the current CUDA device."""

    if (
        type(device) is not torch.device
        or device.type != "cuda"
        or not torch.is_grad_enabled()
        or torch.is_inference_mode_enabled()
        or not _packed_heaviside_masks_runtime_supported()
    ):
        return False
    try:
        current_index = torch.cuda.current_device()
        target_index = current_index if device.index is None else device.index
        if target_index != current_index:
            return False
        capture_check = getattr(
            torch.cuda,
            "is_current_stream_capturing",
            None,
        )
        if not callable(capture_check) or capture_check():
            return False
    except Exception:
        return False
    return True


def _canonical_fit_coefficient_table_supported(
    table,
    current,
    pinned,
    shape,
) -> bool:
    """Validate one immutable canonical CPU coefficient-table source."""

    return (
        table is current
        and current is pinned
        and type(table) is torch.Tensor
        and table.layout is torch.strided
        and table.device.type == "cpu"
        and table.dtype is torch.float64
        and table.shape == shape
        and table.is_contiguous()
        and table.storage_offset() == 0
        and table._base is None
        and _fit_tensor_version_is_zero(table)
        and not table.is_conj()
        and not table.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(table)
    )


def _packed_heaviside_boundary_supported(value, frequency) -> bool:
    """Accept one ordinary scalar bound compatible with ``frequency``."""

    if type(value) in (int, float):
        return True
    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.ndim == 0
        and value.dtype is frequency.dtype
        and value.device == frequency.device
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and not value.is_conj()
        and not value.is_neg()
    )


def _packed_heaviside_masks_dynamic_inputs_supported(
    frequency,
    lower,
    upper,
) -> bool:
    """Validate operands not covered by a request's static-plan proof."""

    return (
        type(frequency) is torch.Tensor
        and frequency.layout is torch.strided
        and frequency.device.type in ("cpu", "cuda")
        and frequency.dtype is torch.float64
        and frequency.ndim in (0, 1)
        and frequency.numel() != 0
        and frequency.is_contiguous()
        and frequency.storage_offset() == 0
        and frequency._base is None
        and not frequency.is_conj()
        and not frequency.is_neg()
        and _packed_heaviside_boundary_supported(lower, frequency)
        and _packed_heaviside_boundary_supported(upper, frequency)
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(
            (frequency, lower, upper)
        )
    )


def _packed_heaviside_masks_supported(frequency, lower, upper) -> bool:
    """Accept only exact plain binary64 CPU/CUDA piecewise inputs."""

    return (
        _packed_heaviside_masks_enabled()
        and _packed_heaviside_masks_runtime_supported()
        and _packed_heaviside_masks_dynamic_inputs_supported(
            frequency,
            lower,
            upper,
        )
    )


def _native_packed_heaviside_masks(frequency, lower, upper):
    """Evaluate five independent eager Heavisides in one leading tensor lane."""

    predicates = torch.stack(
        (
            lower - frequency,
            frequency - lower,
            upper - frequency,
            frequency - upper,
            IMRPhenomX_utils.fM_CUT - frequency,
        )
    )
    at_zero = torch.as_tensor(
        0.5,
        dtype=predicates.dtype,
        device=predicates.device,
    )
    masks = torch.where(
        predicates > 0.0,
        torch.ones_like(predicates),
        torch.where(
            predicates < 0.0,
            torch.zeros_like(predicates),
            at_zero,
        ),
    )
    return masks.unbind(0)


def _maybe_packed_heaviside_masks(
    frequency,
    lower,
    upper,
    *,
    _request_plan=None,
    _request_proof=None,
):
    """Return five packed masks, or ``None`` for the unchanged eager path."""

    request_qualified = _request_top_plan_qualified(
        _request_plan,
        _request_proof,
    )
    if request_qualified:
        supported = _packed_heaviside_masks_dynamic_inputs_supported(
            frequency,
            lower,
            upper,
        )
    else:
        supported = _packed_heaviside_masks_supported(
            frequency,
            lower,
            upper,
        )
    if not supported:
        return None
    return _native_packed_heaviside_masks(frequency, lower, upper)


def _phase_fit_python_scalars_supported(theta, phase_coeffs) -> bool:
    """Accept only plain, independent CPU float64 tensors without AD."""

    return (
        _phase_fit_python_scalars_enabled()
        and type(theta) is torch.Tensor
        and theta.layout is torch.strided
        and theta.device.type == "cpu"
        and theta.dtype == torch.float64
        and theta.shape == (4,)
        and theta.is_contiguous()
        and theta.storage_offset() == 0
        and theta._base is None
        and not theta.is_conj()
        and not theta.is_neg()
        and type(phase_coeffs) is torch.Tensor
        and phase_coeffs.layout is torch.strided
        and phase_coeffs.device == theta.device
        and phase_coeffs.dtype == theta.dtype
        and phase_coeffs.shape == (13, 49)
        and phase_coeffs.is_contiguous()
        and phase_coeffs.storage_offset() == 0
        and phase_coeffs._base is None
        and not phase_coeffs.is_conj()
        and not phase_coeffs.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd((theta, phase_coeffs))
    )


def _phase_fit_native_iterator_supported(theta, phase_coeffs) -> bool:
    """Accept only canonical plain binary64 CPU inputs outside transforms."""

    return (
        _phase_fit_native_iterator_enabled()
        and _packed_heaviside_masks_runtime_supported()
        and type(theta) is torch.Tensor
        and theta.layout is torch.strided
        and theta.device.type == "cpu"
        and theta.dtype is torch.float64
        and theta.shape == (4,)
        and theta.is_contiguous()
        and theta.storage_offset() == 0
        and theta._base is None
        and not theta.is_conj()
        and not theta.is_neg()
        and type(phase_coeffs) is torch.Tensor
        and phase_coeffs
        is IMRPhenomX_utils._PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER
        and phase_coeffs.layout is torch.strided
        and phase_coeffs.device == theta.device
        and phase_coeffs.dtype is theta.dtype
        and phase_coeffs.shape == (13, 49)
        and phase_coeffs.is_contiguous()
        and phase_coeffs.storage_offset() == 0
        and phase_coeffs._base is None
        and phase_coeffs._version == 0
        and not phase_coeffs.is_conj()
        and not phase_coeffs.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(
            (theta, phase_coeffs)
        )
    )


def _amp_fit_python_scalars_supported(theta, amp_coeffs) -> bool:
    """Accept only plain, independent CPU float64 tensors without AD."""

    return (
        _amp_fit_python_scalars_enabled()
        and type(theta) is torch.Tensor
        and theta.layout is torch.strided
        and theta.device.type == "cpu"
        and theta.dtype == torch.float64
        and theta.shape == (4,)
        and theta.is_contiguous()
        and theta.storage_offset() == 0
        and theta._base is None
        and not theta.is_conj()
        and not theta.is_neg()
        and type(amp_coeffs) is torch.Tensor
        and amp_coeffs
        is IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER
        and amp_coeffs.layout is torch.strided
        and amp_coeffs.device == theta.device
        and amp_coeffs.dtype == theta.dtype
        and amp_coeffs.shape == (7, 42)
        and amp_coeffs.is_contiguous()
        and amp_coeffs.storage_offset() == 0
        and amp_coeffs._base is None
        and amp_coeffs._version == 0
        and not amp_coeffs.is_conj()
        and not amp_coeffs.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd((theta, amp_coeffs))
    )


def _amp_fit_native_iterator_supported(theta, amp_coeffs) -> bool:
    """Accept only canonical plain binary64 CPU inputs outside transforms."""

    return (
        _amp_fit_native_iterator_enabled()
        and _packed_heaviside_masks_runtime_supported()
        and type(theta) is torch.Tensor
        and theta.layout is torch.strided
        and theta.device.type == "cpu"
        and theta.dtype is torch.float64
        and theta.shape == (4,)
        and theta.is_contiguous()
        and theta.storage_offset() == 0
        and theta._base is None
        and not theta.is_conj()
        and not theta.is_neg()
        and type(amp_coeffs) is torch.Tensor
        and amp_coeffs
        is IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER
        and amp_coeffs.layout is torch.strided
        and amp_coeffs.device == theta.device
        and amp_coeffs.dtype is theta.dtype
        and amp_coeffs.shape == (7, 42)
        and amp_coeffs.is_contiguous()
        and amp_coeffs.storage_offset() == 0
        and amp_coeffs._base is None
        and amp_coeffs._version == 0
        and not amp_coeffs.is_conj()
        and not amp_coeffs.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(
            (theta, amp_coeffs)
        )
    )


def _phase_fit_python_scalars_host_cuda_supported(
    theta_values,
    phase_coeffs,
    *,
    device,
    dtype,
) -> bool:
    """Accept only canonical host inputs for one exact CUDA upload."""

    return (
        _phase_fit_python_scalars_enabled()
        and _cuda_fit_runtime_supported(device)
        and dtype is torch.float64
        and _fit_theta_values_supported(theta_values)
        and _canonical_fit_coefficient_table_supported(
            phase_coeffs,
            IMRPhenomX_utils._PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER,
            _PHASE_FIT_COEFFICIENT_SOURCE,
            (13, 49),
        )
    )


def _amp_fit_python_scalars_host_cuda_supported(
    theta_values,
    amp_coeffs,
    *,
    device,
    dtype,
) -> bool:
    """Accept only canonical host inputs for one exact CUDA upload."""

    return (
        _cuda_amp_host_pack_enabled()
        and _cuda_fit_runtime_supported(device)
        and dtype is torch.float64
        and _fit_theta_values_supported(theta_values)
        and _canonical_fit_coefficient_table_supported(
            amp_coeffs,
            IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER,
            _AMP_FIT_COEFFICIENT_SOURCE,
            (7, 42),
        )
    )


def _precomputed_cuda_fit_rows_supported(fit_rows, theta, shape) -> bool:
    """Validate rows and their exact consuming CUDA intrinsic tensor."""

    return (
        type(theta) is torch.Tensor
        and _cuda_fit_runtime_supported(theta.device)
        and type(fit_rows) is torch.Tensor
        and fit_rows.layout is torch.strided
        and fit_rows.device == theta.device
        and fit_rows.dtype is torch.float64
        and fit_rows.shape == shape
        and fit_rows.is_contiguous()
        and fit_rows.storage_offset() == 0
        and fit_rows._base is None
        and _fit_tensor_version_is_zero(fit_rows)
        and not fit_rows.is_conj()
        and not fit_rows.is_neg()
        and theta.layout is torch.strided
        and theta.device.type == "cuda"
        and theta.dtype is torch.float64
        and theta.shape == (4,)
        and theta.is_contiguous()
        and theta.storage_offset() == 0
        and theta._base is None
        and _fit_tensor_version_is_zero(theta)
        and not theta.is_conj()
        and not theta.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(
            (fit_rows, theta)
        )
    )


def _precomputed_phase_fit_rows_supported(fit_rows, theta) -> bool:
    """Validate private precomputed rows before reusing them in a plan."""

    if isinstance(theta, torch.Tensor) and theta.device.type == "cuda":
        return _precomputed_cuda_fit_rows_supported(
            fit_rows,
            theta,
            (13,),
        )
    return (
        type(fit_rows) is torch.Tensor
        and fit_rows.layout is torch.strided
        and fit_rows.device == theta.device
        and fit_rows.dtype == theta.dtype
        and fit_rows.shape == (13,)
        and fit_rows.is_contiguous()
        and fit_rows.storage_offset() == 0
        and fit_rows._base is None
        and not fit_rows.is_conj()
        and not fit_rows.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd((fit_rows, theta))
    )


def _precomputed_amp_fit_rows_supported(fit_rows, theta) -> bool:
    """Validate private precomputed amplitude rows before reusing them."""

    if isinstance(theta, torch.Tensor) and theta.device.type == "cuda":
        return _precomputed_cuda_fit_rows_supported(
            fit_rows,
            theta,
            (7,),
        )
    return (
        type(fit_rows) is torch.Tensor
        and fit_rows.layout is torch.strided
        and fit_rows.device == theta.device
        and fit_rows.dtype == theta.dtype
        and fit_rows.shape == (7,)
        and fit_rows.is_contiguous()
        and fit_rows.storage_offset() == 0
        and fit_rows._base is None
        and not fit_rows.is_conj()
        and not fit_rows.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd((fit_rows, theta))
    )


def _canonical_phase_fit_coefficient_rows_python():
    """Cache immutable Python rows from the guarded canonical CPU table."""

    global _PHASE_FIT_COEFFICIENT_ROWS_PYTHON
    global _PHASE_FIT_COEFFICIENT_ROWS_SOURCE
    global _PHASE_FIT_COEFFICIENT_ROWS_VERSION
    source = IMRPhenomX_utils._PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER
    try:
        version = source._version
    except (AttributeError, RuntimeError, TypeError):
        return None
    if (
        _PHASE_FIT_COEFFICIENT_ROWS_PYTHON is None
        or _PHASE_FIT_COEFFICIENT_ROWS_SOURCE is not source
        or _PHASE_FIT_COEFFICIENT_ROWS_VERSION != version
    ):
        try:
            rows = source.tolist()
        except (RuntimeError, TypeError, ValueError):
            return None
        _PHASE_FIT_COEFFICIENT_ROWS_PYTHON = tuple(
            tuple(row) for row in rows
        )
        _PHASE_FIT_COEFFICIENT_ROWS_SOURCE = source
        _PHASE_FIT_COEFFICIENT_ROWS_VERSION = version
    return _PHASE_FIT_COEFFICIENT_ROWS_PYTHON


def _canonical_amp_fit_coefficient_rows_python(source=None):
    """Cache immutable rows from one already-validated canonical table."""

    global _AMP_FIT_COEFFICIENT_ROWS_CACHE

    if source is None:
        source = IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER
    try:
        version = source._version
    except (AttributeError, RuntimeError, TypeError):
        return None
    if (
        source
        is not IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER
        or source is not _AMP_FIT_COEFFICIENT_SOURCE
        or version != 0
    ):
        return None

    cache = _AMP_FIT_COEFFICIENT_ROWS_CACHE
    if (
        cache is not None
        and cache[0] is source
        and cache[1] == version
    ):
        try:
            if (
                source
                is IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER
                and source is _AMP_FIT_COEFFICIENT_SOURCE
                and source._version == version
            ):
                return cache[2]
        except (AttributeError, RuntimeError, TypeError):
            return None

    try:
        rows = tuple(tuple(row) for row in source.tolist())
        source_is_current = (
            source
            is IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER
            and source is _AMP_FIT_COEFFICIENT_SOURCE
            and source._version == version
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if not source_is_current:
        return None

    # A single immutable cache record makes the unlocked hit coherent under
    # free-threaded Python.  Serialize only first publication or replacement.
    with _AMP_FIT_COEFFICIENT_ROWS_LOCK:
        try:
            if (
                source
                is not IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER
                or source is not _AMP_FIT_COEFFICIENT_SOURCE
                or source._version != version
            ):
                return None
        except (AttributeError, RuntimeError, TypeError):
            return None
        cache = _AMP_FIT_COEFFICIENT_ROWS_CACHE
        if (
            cache is not None
            and cache[0] is source
            and cache[1] == version
        ):
            return cache[2]
        _AMP_FIT_COEFFICIENT_ROWS_CACHE = (source, version, rows)
        return rows


def _phase_plan_bulk_tree_supported(value, *, device, dtype) -> bool:
    """Fail closed unless every leaf has the packed path's exact semantics."""

    if isinstance(value, torch.Tensor):
        return (
            type(value) is torch.Tensor
            and value.layout is torch.strided
            and value.device == device
            and value.dtype == dtype
            and value._base is None
            and not value.is_conj()
            and not value.is_neg()
            and not IMRPhenomX_utils._tree_has_autograd(value)
        )
    if isinstance(value, (tuple, list)):
        return all(
            _phase_plan_bulk_tree_supported(
                item,
                device=device,
                dtype=dtype,
            )
            for item in value
        )
    if isinstance(value, dict):
        return all(
            _phase_plan_bulk_tree_supported(
                item,
                device=device,
                dtype=dtype,
            )
            for item in value.values()
        )
    if is_dataclass(value) and not isinstance(value, type):
        return all(
            _phase_plan_bulk_tree_supported(
                getattr(value, field.name),
                device=device,
                dtype=dtype,
            )
            for field in fields(value)
        )
    return value is None or type(value) in (int, float)


def _phase_plan_bulk_collocation_supported(
    theta,
    phase_coeffs,
    chip,
    final_spin,
    coprecessing_deviations,
) -> bool:
    """Accept only plain non-AD float64 plans proven lane-byte-exact."""

    return (
        _phase_plan_bulk_collocation_enabled()
        and type(theta) is torch.Tensor
        and theta.layout is torch.strided
        and theta.device.type in ("cpu", "cuda")
        and theta.dtype == torch.float64
        and theta.ndim == 1
        and theta.shape[0] == 4
        and theta.is_contiguous()
        and theta.storage_offset() == 0
        and theta._base is None
        and not theta.is_conj()
        and not theta.is_neg()
        and type(phase_coeffs) is torch.Tensor
        and phase_coeffs.layout is torch.strided
        and phase_coeffs.device == theta.device
        and phase_coeffs.dtype == theta.dtype
        and phase_coeffs.shape == (13, 49)
        and phase_coeffs.is_contiguous()
        and phase_coeffs.storage_offset() == 0
        and phase_coeffs._base is None
        and not phase_coeffs.is_conj()
        and not phase_coeffs.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd((theta, phase_coeffs))
        and _phase_plan_bulk_tree_supported(
            (chip, final_spin, coprecessing_deviations),
            device=theta.device,
            dtype=theta.dtype,
        )
    )


def _exact_derivative_tree_supported(value, like) -> bool:
    """Reject tensor semantics outside the exact scalar formulas' proof."""

    if isinstance(value, torch.Tensor):
        return (
            type(value) is torch.Tensor
            and value.layout == torch.strided
            and value.ndim == 0
            and value.dtype == like.dtype
            and value.device == like.device
            and not value.is_conj()
            and not value.is_neg()
            and not IMRPhenomX_utils._tree_has_autograd_untrusted(value)
        )
    if isinstance(value, (tuple, list)):
        return all(_exact_derivative_tree_supported(item, like) for item in value)
    return True


def _exact_scalar_derivative_inputs_supported(
    frequency,
    plan,
    *extra,
    _request_proof=None,
) -> bool:
    """Return whether a local reverse pass is byte-exact and AD-safe."""

    if _request_scalar_phase_plan_supported(plan, _request_proof) or type(
        plan
    ) in (
        _PrequalifiedInspiralPhasePlan,
        _PrequalifiedIntermediatePhasePlan,
        _PrequalifiedMergerRingdownPhasePlan,
    ):
        return _exact_scalar_derivative_dynamic_inputs_supported(
            frequency,
            plan,
            *extra,
            _request_proof=_request_proof,
        )
    return (
        type(frequency) is torch.Tensor
        and frequency.layout == torch.strided
        and frequency.ndim == 0
        and frequency.dtype == torch.float64
        and frequency.device.type in ("cpu", "cuda")
        and frequency.is_contiguous()
        and not frequency.is_conj()
        and not frequency.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(frequency)
        and _exact_derivative_tree_supported(plan, frequency)
        and _exact_derivative_tree_supported(extra, frequency)
    )


def _exact_scalar_derivative_plan_like(plan, *, _request_proof=None):
    """Return one validated leaf representing a fixed scalar-plan schema."""

    if _request_scalar_phase_plan_supported(plan, _request_proof):
        return plan.phi2 if len(plan) == 16 else plan[0]
    if type(plan) is _PrequalifiedInspiralPhasePlan:
        return plan.phi2
    if type(plan) in (
        _PrequalifiedIntermediatePhasePlan,
        _PrequalifiedMergerRingdownPhasePlan,
    ):
        return plan[0]
    return None


def _exact_scalar_derivative_dynamic_inputs_supported(
    frequency,
    plan,
    *extra,
    _request_proof=None,
) -> bool:
    """Revalidate dynamic inputs after one request-local plan proof."""

    plan_like = _exact_scalar_derivative_plan_like(
        plan,
        _request_proof=_request_proof,
    )
    return (
        type(frequency) is torch.Tensor
        and frequency.layout == torch.strided
        and frequency.ndim == 0
        and frequency.dtype == torch.float64
        and frequency.device.type in ("cpu", "cuda")
        and frequency.is_contiguous()
        and frequency.storage_offset() == 0
        and frequency._base is None
        and not frequency.is_conj()
        and not frequency.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(frequency)
        and type(plan_like) is torch.Tensor
        and plan_like.layout is torch.strided
        and plan_like.ndim == 0
        and plan_like.dtype == frequency.dtype
        and plan_like.device == frequency.device
        and not plan_like.is_conj()
        and not plan_like.is_neg()
        and _exact_derivative_tree_supported(extra, frequency)
    )


def _maybe_prequalify_scalar_derivative_plan(
    plan,
    like,
    *,
    enabled,
    _request_proof=None,
):
    """Mark one fixed tuple schema after its sole request-local proof."""

    qualified = _request_qualify_scalar_phase_plan(plan, _request_proof)
    if qualified is not plan:
        return qualified
    if not enabled or not _exact_derivative_tree_supported(plan, like):
        return plan
    if type(plan) is _InspiralPhasePlan:
        return _PrequalifiedInspiralPhasePlan._make(plan)
    if type(plan) is _IntermediatePhasePlan:
        return _PrequalifiedIntermediatePhasePlan._make(plan)
    if type(plan) is _MergerRingdownPhasePlan:
        return _PrequalifiedMergerRingdownPhasePlan._make(plan)
    return plan


def _exact_scalar_derivative_supported(
    frequency,
    plan,
    *extra,
    _request_proof=None,
) -> bool:
    """Select the guarded exact phase reverse pass."""

    request_qualified = _request_scalar_phase_plan_supported(
        plan,
        _request_proof,
    )
    return (
        request_qualified or _exact_scalar_derivatives_enabled()
    ) and _exact_scalar_derivative_inputs_supported(
        frequency,
        plan,
        *extra,
        _request_proof=_request_proof,
    )


def _exact_scalar_amp_derivative_supported(
    frequency,
    plan,
    *extra,
    _request_proof=None,
) -> bool:
    """Select the guarded exact amplitude reverse pass."""

    return (
        _request_amp_region_plan_supported(plan, _request_proof)
        or _exact_scalar_amp_derivatives_enabled()
    ) and _exact_scalar_derivative_inputs_supported(
        frequency,
        plan,
        *extra,
        _request_proof=_request_proof,
    )


def _python_intermediate_amp_runtime_supported() -> bool:
    """Require CPython IEEE binary64 and reject observable tensor modes."""

    if (
        sys.implementation.name != "cpython"
        or sys.float_info.radix != 2
        or sys.float_info.mant_dig != 53
        or sys.float_info.rounds != 1
        or not float.__getformat__("double").startswith("IEEE")
        or not callable(getattr(torch, "_foreach_pow", None))
    ):
        return False

    for function in (
        getattr(torch.jit, "is_scripting", None),
        getattr(torch.jit, "is_tracing", None),
        getattr(getattr(torch, "compiler", None), "is_compiling", None),
        getattr(getattr(torch, "_dynamo", None), "is_compiling", None),
    ):
        if function is None:
            return False
        try:
            if function():
                return False
        except Exception:
            return False

    tracing_state = getattr(getattr(torch, "_C", None), "_get_tracing_state", None)
    if tracing_state is None:
        return False
    try:
        if tracing_state() is not None:
            return False
    except Exception:
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
        try:
            legacy_cpu = getattr(torch, "is_autocast_cpu_enabled", None)
            if (
                autocast_enabled()
                or legacy_cpu is None
                or legacy_cpu()
            ):
                return False
        except Exception:
            return False
    except Exception:
        return False
    return True


def _python_intermediate_amp_plain_scalar(value) -> bool:
    """Accept one owned, ordinary CPU binary64 scalar tensor."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.device.type == "cpu"
        and value.dtype is torch.float64
        and value.ndim == 0
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and not value.is_conj()
        and not value.is_neg()
    )


def _python_intermediate_amp_packed_fMs_RD_view(value) -> bool:
    """Accept only the carrier ringdown view made by the packed XPHM plan.

    ``_pack_remnant_plan`` evaluates the aligned and carrier ringdown
    frequencies in one two-element lane, then returns the carrier value as
    ``ringdown_lane.unbind()[1]``.  Attest that exact request-local topology;
    other scalar views keep using the eager path.  The caller clones the
    accepted value so the existing executor still owns every scalar argument.
    """

    try:
        if (
            type(value) is not torch.Tensor
            or value.layout is not torch.strided
            or value.device.type != "cpu"
            or value.dtype is not torch.float64
            or value.ndim != 0
            or not value.is_contiguous()
            or value.storage_offset() != 1
            or value.is_conj()
            or value.is_neg()
            or value.requires_grad
            or value.grad_fn is not None
            or not value.is_leaf
            or value._version != 0
        ):
            return False
        base = value._base
        if (
            type(base) is not torch.Tensor
            or base.layout is not torch.strided
            or base.device.type != "cpu"
            or base.dtype is not torch.float64
            or base.shape != (2,)
            or base.stride() != (1,)
            or not base.is_contiguous()
            or base.storage_offset() != 0
            or base._base is not None
            or base.is_conj()
            or base.is_neg()
            or base.requires_grad
            or base.grad_fn is not None
            or not base.is_leaf
            or base._version != 0
            or base.untyped_storage().nbytes() != 2 * base.element_size()
            or value.data_ptr() != base.data_ptr() + base.element_size()
            or IMRPhenomX_utils._tree_has_autograd_untrusted((value, base))
        ):
            return False
    except Exception:
        return False
    return True


def _python_intermediate_amp_executor_arguments(
    FMs1,
    FMs4,
    fit_rows,
    inspiral_plan,
    mergerringdown_plan,
    coprecessing_deviations,
    *,
    _request_proof=None,
):
    """Return proven executor arguments, or ``None`` to use eager code.

    The executor receives ``fit_rows[3]`` as a scalar view.  Its ordinary
    binary64 semantics are established by validating the exact, owned parent
    vector here before creating that view.
    """

    request_qualified = (
        _request_amp_region_plan_supported(inspiral_plan, _request_proof)
        and _request_amp_region_plan_supported(
            mergerringdown_plan,
            _request_proof,
        )
    )
    if not request_qualified and (
        not _python_intermediate_amp_runtime_supported()
        or coprecessing_deviations is not None
        or type(fit_rows) is not torch.Tensor
        or fit_rows.layout is not torch.strided
        or fit_rows.device.type != "cpu"
        or fit_rows.dtype is not torch.float64
        or fit_rows.shape != (7,)
        or not fit_rows.is_contiguous()
        or fit_rows.storage_offset() != 0
        or fit_rows._base is not None
        or fit_rows.is_conj()
        or fit_rows.is_neg()
        or type(inspiral_plan) is not _InspiralAmpPlan
        or type(inspiral_plan.A0) is not float
        or inspiral_plan.A0 != 1.0
        or type(mergerringdown_plan) is not _MergerRingdownAmpPlan
    ):
        return None

    fMs_RD = mergerringdown_plan.fMs_RD
    if request_qualified:
        executor_fMs_RD = (
            fMs_RD if fMs_RD._base is None else fMs_RD.clone()
        )
        return (
            FMs1,
            FMs4,
            fit_rows[3],
            inspiral_plan.A2,
            inspiral_plan.A3,
            inspiral_plan.A4,
            inspiral_plan.A5,
            inspiral_plan.A6,
            inspiral_plan.rho1,
            inspiral_plan.rho2,
            inspiral_plan.rho3,
            executor_fMs_RD,
            mergerringdown_plan.gammaR,
            mergerringdown_plan.gammaD2,
            mergerringdown_plan.gammaD13,
        )
    owned_tensor_values = (
        FMs1,
        FMs4,
        *inspiral_plan[1:],
        *mergerringdown_plan[1:],
    )
    if not all(
        _python_intermediate_amp_plain_scalar(value)
        for value in owned_tensor_values
    ):
        return None
    if _python_intermediate_amp_plain_scalar(fMs_RD):
        executor_fMs_RD = fMs_RD
    elif _python_intermediate_amp_packed_fMs_RD_view(fMs_RD):
        try:
            executor_fMs_RD = fMs_RD.clone()
        except Exception:
            return None
        if not _python_intermediate_amp_plain_scalar(executor_fMs_RD):
            return None
    else:
        return None
    if IMRPhenomX_utils._tree_has_autograd_untrusted(
        (fit_rows, owned_tensor_values, fMs_RD, executor_fMs_RD)
    ):
        return None
    return (
        FMs1,
        FMs4,
        fit_rows[3],
        inspiral_plan.A2,
        inspiral_plan.A3,
        inspiral_plan.A4,
        inspiral_plan.A5,
        inspiral_plan.A6,
        inspiral_plan.rho1,
        inspiral_plan.rho2,
        inspiral_plan.rho3,
        executor_fMs_RD,
        mergerringdown_plan.gammaR,
        mergerringdown_plan.gammaD2,
        mergerringdown_plan.gammaD13,
    )


def _load_python_intermediate_amp_executor():
    """Import the static lane only after its gate and contract pass."""

    from ._imrphenomxas_python_intermediate import intermediate_amp_lane

    return intermediate_amp_lane


def _get_python_intermediate_amp_executor():
    """Load and cache the fixed binary64 lane once per process."""

    global _PYTHON_INTERMEDIATE_AMP_EXECUTOR
    global _PYTHON_INTERMEDIATE_AMP_CALIBRATED
    global _PYTHON_INTERMEDIATE_AMP_FAILED
    global _PYTHON_INTERMEDIATE_AMP_PID

    process_id = os.getpid()
    if _PYTHON_INTERMEDIATE_AMP_PID != process_id:
        # A child must never acquire a possibly locked mutex inherited from a
        # parent thread. ``register_at_fork`` normally performs this reset;
        # the explicit check also covers embedders with unusual fork hooks.
        _reset_python_intermediate_amp_after_fork()
    if _PYTHON_INTERMEDIATE_AMP_EXECUTOR is not None:
        return _PYTHON_INTERMEDIATE_AMP_EXECUTOR
    if _PYTHON_INTERMEDIATE_AMP_FAILED:
        return None
    with _PYTHON_INTERMEDIATE_AMP_LOCK:
        if _PYTHON_INTERMEDIATE_AMP_EXECUTOR is not None:
            return _PYTHON_INTERMEDIATE_AMP_EXECUTOR
        if _PYTHON_INTERMEDIATE_AMP_FAILED:
            return None
        try:
            executor = _load_python_intermediate_amp_executor()
        except Exception:
            _PYTHON_INTERMEDIATE_AMP_FAILED = True
            return None
        _PYTHON_INTERMEDIATE_AMP_EXECUTOR = executor
        _PYTHON_INTERMEDIATE_AMP_CALIBRATED = False
        _PYTHON_INTERMEDIATE_AMP_PID = process_id
        return executor


def _clear_python_intermediate_amp_cache() -> None:
    """Release this process's lane, calibration, and remembered failure."""

    global _PYTHON_INTERMEDIATE_AMP_EXECUTOR
    global _PYTHON_INTERMEDIATE_AMP_CALIBRATED
    global _PYTHON_INTERMEDIATE_AMP_FAILED
    global _PYTHON_INTERMEDIATE_AMP_PID

    with _PYTHON_INTERMEDIATE_AMP_LOCK:
        _PYTHON_INTERMEDIATE_AMP_EXECUTOR = None
        _PYTHON_INTERMEDIATE_AMP_CALIBRATED = False
        _PYTHON_INTERMEDIATE_AMP_FAILED = False
        _PYTHON_INTERMEDIATE_AMP_PID = os.getpid()


def _mark_python_intermediate_amp_failed() -> None:
    """Fail closed after a load, execution, or calibration error."""

    global _PYTHON_INTERMEDIATE_AMP_EXECUTOR
    global _PYTHON_INTERMEDIATE_AMP_CALIBRATED
    global _PYTHON_INTERMEDIATE_AMP_FAILED
    global _PYTHON_INTERMEDIATE_AMP_PID

    with _PYTHON_INTERMEDIATE_AMP_LOCK:
        _PYTHON_INTERMEDIATE_AMP_EXECUTOR = None
        _PYTHON_INTERMEDIATE_AMP_CALIBRATED = False
        _PYTHON_INTERMEDIATE_AMP_FAILED = True
        _PYTHON_INTERMEDIATE_AMP_PID = os.getpid()


def _mark_python_intermediate_amp_calibrated() -> None:
    """Remember that this process/runtime reproduced eager output exactly."""

    global _PYTHON_INTERMEDIATE_AMP_CALIBRATED

    with _PYTHON_INTERMEDIATE_AMP_LOCK:
        if not _PYTHON_INTERMEDIATE_AMP_FAILED:
            _PYTHON_INTERMEDIATE_AMP_CALIBRATED = True


def _python_intermediate_amp_exact_nonlinear_values(arguments):
    """Evaluate fractional powers with one exact Tensor-Scalar foreach call."""

    bases = tuple(
        arguments[index]
        for index in _PYTHON_INTERMEDIATE_AMP_POWER_BASE_ARGUMENTS
    )
    # The CPU foreach scalar-list kernel invokes ``tensor.pow(scalar)`` for
    # each pair.  This removes repeated Python dispatch without changing the
    # overload used by the original eager expressions.
    power_tensors = torch._foreach_pow(
        bases,
        _PYTHON_INTERMEDIATE_AMP_POWER_EXPONENTS,
    )
    exact_powers = tuple(map(torch.Tensor.item, power_tensors))
    exact_exponential = torch.exp(
        (-(arguments[1] - arguments[11])) * arguments[12]
    ).item()
    return exact_powers, exact_exponential


def _execute_python_intermediate_amp(executor, arguments):
    """Run Torch nonlinear primitives, then the ordered binary64 lane."""

    try:
        scalar_arguments = tuple(map(torch.Tensor.item, arguments))
        if (
            scalar_arguments[0] <= 0.0
            or scalar_arguments[1] <= 0.0
            or not all(math.isfinite(value) for value in scalar_arguments)
        ):
            return None
        exact_nonlinear_values = (
            _python_intermediate_amp_exact_nonlinear_values(arguments)
        )
        if exact_nonlinear_values is None:
            return None
        exact_powers, exact_exponential = exact_nonlinear_values
        if (
            not all(math.isfinite(value) for value in exact_powers)
            or not math.isfinite(exact_exponential)
        ):
            return None
        values = executor(
            *scalar_arguments,
            exact_powers,
            exact_exponential,
        )
        if (
            type(values) is not tuple
            or len(values) != 5
            or not all(type(value) is float for value in values)
        ):
            raise TypeError("invalid binary64 intermediate-amplitude result")
        if not all(math.isfinite(value) for value in values):
            return None
        like = arguments[0]
        return tuple(
            torch.scalar_tensor(
                value,
                dtype=like.dtype,
                device=like.device,
            )
            for value in values
        )
    except (ArithmeticError, ValueError):
        # Python scalar arithmetic raises on some domains where tensor
        # arithmetic deliberately returns IEEE infinities or NaNs. Preserve
        # the eager request semantics without poisoning later valid requests.
        return None
    except Exception:
        _mark_python_intermediate_amp_failed()
        return None


def _python_intermediate_amp_raw_equal(left, right) -> bool:
    """Compare scalar plans by payload bytes, including signed zero and NaN."""

    return all(
        type(left_value) is torch.Tensor
        and type(right_value) is torch.Tensor
        and left_value.dtype is right_value.dtype
        and left_value.device == right_value.device
        and torch.equal(
            left_value.detach().contiguous().reshape(-1).view(torch.uint8),
            right_value.detach().contiguous().reshape(-1).view(torch.uint8),
        )
        for left_value, right_value in zip(left, right)
    )


def _python_inspiral_derivative_runtime_supported() -> bool:
    """Require the same plain CPython/Torch runtime as the scalar amp lane."""

    return _python_intermediate_amp_runtime_supported()


def _python_inspiral_derivative_executor_arguments(
    frequency,
    plan,
    output_adjoint,
    initial_gradient,
    *,
    exact_inputs_prequalified=False,
    _request_proof=None,
):
    """Return qualified scalar-lane inputs, or ``None`` for eager fallback."""

    request_qualified = _request_inspiral_phase_plan_supported(
        plan,
        _request_proof,
    )
    if request_qualified:
        # The marker proves only the immutable phase-plan schema. Frequency
        # and reverse-pass operands remain request-dynamic and must retain the
        # exact scalar lane's complete layout/AD contract.
        if not _exact_scalar_derivative_dynamic_inputs_supported(
            frequency,
            plan,
            output_adjoint,
            initial_gradient,
            _request_proof=_request_proof,
        ):
            return None
    elif (
        not _python_inspiral_derivative_runtime_supported()
        or type(plan)
        not in (_InspiralPhasePlan, _PrequalifiedInspiralPhasePlan)
    ):
        return None

    if exact_inputs_prequalified or request_qualified:
        # The caller has just proven the exact-derivative tensor contract.
        # Check only the additional ownership/CPU restrictions needed before
        # exposing storage to NumPy; retain the untrusted AD scan below.
        if (
            frequency.device.type != "cpu"
            or frequency.storage_offset() != 0
            or frequency._base is not None
        ):
            return None
        if not request_qualified:
            for value in plan:
                if type(value) is float:
                    continue
                if (
                    type(value) is not torch.Tensor
                    or value.storage_offset() != 0
                    or value._base is not None
                ):
                    return None
        for value in (output_adjoint, initial_gradient):
            if value is not None and (
                type(value) is not torch.Tensor
                or value.storage_offset() != 0
                or value._base is not None
            ):
                return None
    else:
        if not _python_intermediate_amp_plain_scalar(frequency):
            return None
        for value in plan:
            if type(value) is float:
                continue
            if not _python_intermediate_amp_plain_scalar(value):
                return None
        for value in (output_adjoint, initial_gradient):
            if value is not None and not _python_intermediate_amp_plain_scalar(
                value
            ):
                return None
    dynamic_inputs = (frequency, output_adjoint, initial_gradient)
    if not request_qualified:
        dynamic_inputs = (frequency, plan, output_adjoint, initial_gradient)
    if IMRPhenomX_utils._tree_has_autograd_untrusted(dynamic_inputs):
        return None
    return frequency, plan, output_adjoint, initial_gradient


def _load_python_inspiral_derivative_executor():
    """Import the static ordered lane only after its contract passes."""

    from ._imrphenomxas_python_inspiral_derivative import (
        inspiral_phase_value_and_derivative_lane,
    )

    return inspiral_phase_value_and_derivative_lane


def _get_python_inspiral_derivative_executor():
    """Load the fixed executor once per process; never cache request data."""

    global _PYTHON_INSPIRAL_DERIVATIVE_EXECUTOR
    global _PYTHON_INSPIRAL_DERIVATIVE_CALIBRATED
    global _PYTHON_INSPIRAL_DERIVATIVE_FAILED
    global _PYTHON_INSPIRAL_DERIVATIVE_PID

    process_id = os.getpid()
    if _PYTHON_INSPIRAL_DERIVATIVE_PID != process_id:
        _reset_python_inspiral_derivative_after_fork()
    if _PYTHON_INSPIRAL_DERIVATIVE_EXECUTOR is not None:
        return _PYTHON_INSPIRAL_DERIVATIVE_EXECUTOR
    if _PYTHON_INSPIRAL_DERIVATIVE_FAILED:
        return None
    with _PYTHON_INSPIRAL_DERIVATIVE_LOCK:
        if _PYTHON_INSPIRAL_DERIVATIVE_EXECUTOR is not None:
            return _PYTHON_INSPIRAL_DERIVATIVE_EXECUTOR
        if _PYTHON_INSPIRAL_DERIVATIVE_FAILED:
            return None
        try:
            executor = _load_python_inspiral_derivative_executor()
        except Exception:
            _PYTHON_INSPIRAL_DERIVATIVE_FAILED = True
            return None
        _PYTHON_INSPIRAL_DERIVATIVE_EXECUTOR = executor
        _PYTHON_INSPIRAL_DERIVATIVE_CALIBRATED = False
        _PYTHON_INSPIRAL_DERIVATIVE_PID = process_id
        return executor


def _clear_python_inspiral_derivative_cache() -> None:
    """Release the process-local executor and calibration state."""

    global _PYTHON_INSPIRAL_DERIVATIVE_EXECUTOR
    global _PYTHON_INSPIRAL_DERIVATIVE_CALIBRATED
    global _PYTHON_INSPIRAL_DERIVATIVE_FAILED
    global _PYTHON_INSPIRAL_DERIVATIVE_PID

    with _PYTHON_INSPIRAL_DERIVATIVE_LOCK:
        _PYTHON_INSPIRAL_DERIVATIVE_EXECUTOR = None
        _PYTHON_INSPIRAL_DERIVATIVE_CALIBRATED = False
        _PYTHON_INSPIRAL_DERIVATIVE_FAILED = False
        _PYTHON_INSPIRAL_DERIVATIVE_PID = os.getpid()


def _mark_python_inspiral_derivative_failed() -> None:
    """Fail closed after a load, execution, or calibration error."""

    global _PYTHON_INSPIRAL_DERIVATIVE_EXECUTOR
    global _PYTHON_INSPIRAL_DERIVATIVE_CALIBRATED
    global _PYTHON_INSPIRAL_DERIVATIVE_FAILED
    global _PYTHON_INSPIRAL_DERIVATIVE_PID

    with _PYTHON_INSPIRAL_DERIVATIVE_LOCK:
        _PYTHON_INSPIRAL_DERIVATIVE_EXECUTOR = None
        _PYTHON_INSPIRAL_DERIVATIVE_CALIBRATED = False
        _PYTHON_INSPIRAL_DERIVATIVE_FAILED = True
        _PYTHON_INSPIRAL_DERIVATIVE_PID = os.getpid()


def _mark_python_inspiral_derivative_calibrated() -> None:
    """Remember that this runtime reproduced the eager payload exactly."""

    global _PYTHON_INSPIRAL_DERIVATIVE_CALIBRATED

    with _PYTHON_INSPIRAL_DERIVATIVE_LOCK:
        if not _PYTHON_INSPIRAL_DERIVATIVE_FAILED:
            _PYTHON_INSPIRAL_DERIVATIVE_CALIBRATED = True


def _python_inspiral_derivative_exact_nonlinear_values(frequency):
    """Evaluate the two powers and logarithm with their eager Torch overloads."""

    powers = torch._foreach_pow(
        (frequency, frequency),
        (1.0 / 3.0, (1.0 / 3.0) - 1.0),
    )
    return powers[0].item(), powers[1].item(), torch.log(frequency).item()


def _python_inspiral_derivative_bulk_io_plan_supported(
    plan,
    *,
    _request_proof=None,
) -> bool:
    """Recognize the fixed scalar/tensor schema qualified by the outer lane."""

    return _request_inspiral_phase_plan_supported(plan, _request_proof) or (
        _python_inspiral_derivative_bulk_io_enabled()
        and type(plan) in (_InspiralPhasePlan, _PrequalifiedInspiralPhasePlan)
        and type(plan[0]) is float
        and type(plan[1]) is float
        and type(plan[2]) is torch.Tensor
        and type(plan[3]) is torch.Tensor
        and type(plan[4]) is torch.Tensor
        and type(plan[5]) is float
        and type(plan[6]) is torch.Tensor
        and type(plan[7]) is torch.Tensor
        and type(plan[8]) is float
        and type(plan[9]) is torch.Tensor
        and type(plan[10]) is torch.Tensor
        and type(plan[11]) is torch.Tensor
        and type(plan[12]) is torch.Tensor
        and type(plan[13]) is torch.Tensor
        and type(plan[14]) is torch.Tensor
        and type(plan[15]) is torch.Tensor
    )


def _execute_python_inspiral_derivative(
    executor,
    arguments,
    *,
    _request_proof=None,
):
    """Pack 0-D inputs once, then run the ordered binary64 executor."""

    frequency, plan, output_adjoint, initial_gradient = arguments
    bulk_io = _python_inspiral_derivative_bulk_io_plan_supported(
        plan,
        _request_proof=_request_proof,
    )
    try:
        if bulk_io:
            packed = [
                frequency,
                plan[2],
                plan[3],
                plan[4],
                plan[6],
                plan[7],
                plan[9],
                plan[10],
                plan[11],
                plan[12],
                plan[13],
                plan[14],
                plan[15],
            ]
            if output_adjoint is not None:
                packed.append(output_adjoint)
            if initial_gradient is not None:
                packed.append(initial_gradient)
            scalar_values = torch.stack(packed).tolist()
            frequency_value = scalar_values[0]
            plan_values = (
                plan[0],
                plan[1],
                scalar_values[1],
                scalar_values[2],
                scalar_values[3],
                plan[5],
                scalar_values[4],
                scalar_values[5],
                plan[8],
                scalar_values[6],
                scalar_values[7],
                scalar_values[8],
                scalar_values[9],
                scalar_values[10],
                scalar_values[11],
                scalar_values[12],
            )
            next_index = 13
            output_value = None
            if output_adjoint is not None:
                output_value = scalar_values[next_index]
                next_index += 1
            initial_value = (
                None
                if initial_gradient is None
                else scalar_values[next_index]
            )
        else:
            tensor_indices = tuple(
                index
                for index, value in enumerate(plan)
                if type(value) is torch.Tensor
            )
            packed = [frequency, *(plan[index] for index in tensor_indices)]
            output_index = initial_index = None
            if output_adjoint is not None:
                output_index = len(packed)
                packed.append(output_adjoint)
            if initial_gradient is not None:
                initial_index = len(packed)
                packed.append(initial_gradient)

            scalar_values = torch.stack(packed).numpy()
            frequency_value = float(scalar_values[0])
            plan_values = list(plan)
            for packed_index, plan_index in enumerate(tensor_indices, 1):
                plan_values[plan_index] = float(scalar_values[packed_index])
            output_value = (
                None
                if output_index is None
                else float(scalar_values[output_index])
            )
            initial_value = (
                None
                if initial_index is None
                else float(scalar_values[initial_index])
            )
        all_values = (
            frequency_value,
            *plan_values,
            *(() if output_value is None else (output_value,)),
            *(() if initial_value is None else (initial_value,)),
        )
        if frequency_value <= 0.0 or not all(
            math.isfinite(value) for value in all_values
        ):
            return None

        f13, fminus23, log_f = (
            _python_inspiral_derivative_exact_nonlinear_values(frequency)
        )
        nonlinear_values = f13, fminus23, log_f
        if not all(math.isfinite(value) for value in nonlinear_values):
            return None
        phase_normalization = -(3.0 * PI ** (-5.0 / 3.0)) / 128.0
        values = executor(
            frequency_value,
            tuple(plan_values),
            output_value,
            initial_value,
            f13,
            fminus23,
            log_f,
            phase_normalization,
        )
        if (
            type(values) is not tuple
            or len(values) != 2
            or not all(type(value) is float for value in values)
        ):
            raise TypeError("invalid binary64 inspiral-derivative result")
        if not all(math.isfinite(value) for value in values):
            return None
        return tuple(
            torch.scalar_tensor(
                value,
                dtype=frequency.dtype,
                device=frequency.device,
            )
            for value in values
        )
    except (ArithmeticError, ValueError):
        return None
    except Exception:
        _mark_python_inspiral_derivative_failed()
        return None


def _python_inspiral_derivative_raw_equal(left, right) -> bool:
    """Compare both scalar payloads, including signed zero and NaN bits."""

    return (
        type(left) is tuple
        and type(right) is tuple
        and len(left) == len(right) == 2
        and _python_intermediate_amp_raw_equal(left, right)
    )


def _amp_plan_tree_supported(value, *, device, dtype) -> bool:
    """Fail closed unless every leaf is safe for exact plan reuse."""

    if isinstance(value, torch.Tensor):
        return (
            type(value) is torch.Tensor
            and value.layout is torch.strided
            and value.device == device
            and value.dtype == dtype
            and not value.is_conj()
            and not value.is_neg()
            and not IMRPhenomX_utils._tree_has_autograd(value)
        )
    if isinstance(value, (tuple, list)):
        return all(
            _amp_plan_tree_supported(item, device=device, dtype=dtype) for item in value
        )
    if isinstance(value, dict):
        return all(
            _amp_plan_tree_supported(item, device=device, dtype=dtype)
            for item in value.values()
        )
    if is_dataclass(value) and not isinstance(value, type):
        return all(
            _amp_plan_tree_supported(
                getattr(value, field.name),
                device=device,
                dtype=dtype,
            )
            for field in fields(value)
        )
    return value is None or type(value) in (int, float)


def _amp_plan_inputs_supported(f, theta, amp_coeffs, inputs) -> bool:
    """Return whether one exact non-autograd waveform can reuse a plan."""

    return (
        type(f) is torch.Tensor
        and f.layout is torch.strided
        and f.device.type in ("cpu", "cuda")
        and f.dtype in (torch.float32, torch.float64)
        and f.ndim == 1
        and type(theta) is torch.Tensor
        and theta.layout is torch.strided
        and theta.device == f.device
        and theta.dtype == f.dtype
        and theta.ndim == 1
        and theta.shape[0] == 4
        and type(amp_coeffs) is torch.Tensor
        and amp_coeffs.layout is torch.strided
        and amp_coeffs.device == f.device
        and amp_coeffs.dtype == f.dtype
        and amp_coeffs.shape == (7, 42)
        and _amp_plan_tree_supported(
            inputs,
            device=f.device,
            dtype=f.dtype,
        )
    )


def _derived_scalar_powers_supported(*values) -> bool:
    """Fail closed unless scalar power results are safe to share exactly."""

    if (
        not _derived_power_reuse_enabled()
        or not values
        or not IMRPhenomX_utils._remnant_python_scalars_runtime_supported()
    ):
        return False
    first = values[0]
    if (
        type(first) is not torch.Tensor
        or first.layout is not torch.strided
        or first.ndim != 0
        or first.dtype not in (torch.float32, torch.float64)
        or first.device.type not in ("cpu", "cuda")
    ):
        return False
    return all(
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.ndim == 0
        and value.dtype == first.dtype
        and value.device == first.device
        and not value.is_conj()
        and not value.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(value)
        for value in values
    )


def _region_pruning_vector_supported(value, *, dtype=None) -> bool:
    """Accept only owned, contiguous, one-dimensional CPU float tensors."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.device.type == "cpu"
        and value.dtype in (torch.float32, torch.float64)
        and (dtype is None or value.dtype == dtype)
        and value.ndim == 1
        and value.numel() >= _REGION_PRUNING_MIN_SAMPLES
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and not value.is_conj()
        and not value.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(value)
    )


def _region_pruning_scalar_supported(value, *, like) -> bool:
    """Accept only plain scalar boundaries matching the frequency tensor."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.device == like.device
        and value.dtype == like.dtype
        and value.ndim == 0
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and not value.is_conj()
        and not value.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(value)
    )


def _piecewise_region_indices(
    source_frequency,
    frequency,
    first,
    second,
    plan,
    *,
    _request_proof=None,
):
    """Return exact region starts, or ``None`` for the legacy dense path.

    TensorIterator's CPU vector/scalar partition depends on the input length.
    Exact pruning is therefore restricted to one Torch thread and callers pad
    every evaluated region to a global 64-sample boundary. All device checks
    precede reductions and scalar extraction so accelerator fallbacks cannot
    introduce a synchronization.
    """

    request_qualified = _request_top_plan_qualified(plan, _request_proof)
    if request_qualified:
        if not _request_region_pruning_qualified(plan, _request_proof):
            return None
    elif not _region_pruning_enabled():
        return None
    if not _region_pruning_vector_supported(source_frequency):
        return None
    if not _region_pruning_vector_supported(
        frequency,
        dtype=source_frequency.dtype,
    ):
        return None
    if torch.get_num_threads() != 1:
        return None
    if not (
        _imrphenomxas_phase_plan_type_supported(plan, _request_proof)
        or _imrphenomxas_amp_plan_type_supported(plan, _request_proof)
    ):
        return None
    if not _region_pruning_scalar_supported(first, like=frequency):
        return None
    if not _region_pruning_scalar_supported(second, like=frequency):
        return None
    # Both exact plan types are private immutable containers built from the
    # already device/dtype-validated waveform inputs. Recheck AD here because
    # a caller can still pass a manually prepared graph-bearing plan.
    if (
        not request_qualified
        and IMRPhenomX_utils._tree_has_autograd_untrusted(plan)
    ):
        return None

    boundaries = torch.stack(
        (first, second, frequency.new_tensor(IMRPhenomX_utils.fM_CUT))
    )
    if not bool(torch.all(torch.isfinite(boundaries))):
        return None
    if not bool(torch.all(boundaries[1:] > boundaries[:-1])):
        return None
    if not bool(torch.all(torch.isfinite(frequency) & (frequency > 0.0))):
        return None
    if not bool(torch.all(frequency[1:] >= frequency[:-1])):
        return None

    left = torch.searchsorted(frequency, boundaries, right=False).tolist()
    right = torch.searchsorted(frequency, boundaries, right=True).tolist()
    if right[2] != frequency.numel() or any(
        left_index != right_index
        for left_index, right_index in zip(left, right)
    ):
        # Preserve the legacy 0.5 blend at exact transition/cutoff samples and
        # its behavior for samples above the model cutoff.
        return None
    return tuple(left)


def _evaluate_aligned_region(frequency, left, right, evaluator):
    """Evaluate one region with the dense TensorIterator lane partition."""

    if left == right:
        return frequency[left:right]
    alignment = _REGION_PRUNING_ALIGNMENT
    padded_left = (left // alignment) * alignment
    padded_right = min(
        frequency.numel(),
        ((right + alignment - 1) // alignment) * alignment,
    )
    offset = left - padded_left
    values = evaluator(frequency[padded_left:padded_right])
    return values[offset : offset + right - left]


def _detach_phase_plan(plan):
    """Detach coefficients used only for scalar frequency derivatives."""

    def detached(value):
        if not isinstance(value, torch.Tensor):
            return value
        result = value.detach()
        # Detaching a nonzero-offset view hides its base but preserves the
        # offset. Clone only those leaves so the fixed-schema derivative plan
        # can be proven once without changing any value bytes.
        if result.storage_offset() != 0:
            result = result.clone()
        return result

    return type(plan)(
        *(detached(value) for value in plan)
    )


def _coprecessing_deviation(
    deviations: PNRCoprecessingDeviations | None,
    name: str,
    *,
    like: torch.Tensor | None = None,
):
    """Return one scaled PNR co-precessing carrier correction."""

    if deviations is None:
        return 0.0
    correction = deviations.strength * getattr(deviations.fits, name)
    if like is not None:
        if correction.device != like.device:
            if like.device.type == "mps":
                correction = correction.to(dtype=like.dtype)
                correction = correction.to(device=like.device)
            else:
                correction = correction.to(device=like.device)
                correction = correction.to(dtype=like.dtype)
        else:
            correction = correction.to(dtype=like.dtype)
    return correction


def _get_cutoff_fMs(
    m1: FloatLike,
    m2: FloatLike,
    chi1: FloatLike,
    chi2: FloatLike,
    chip: FloatLike = 0.0,
    *,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
):
    """Return PhenomX cutoffs after the PNR ringdown-frequency shifts."""

    fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(
        m1,
        m2,
        chi1,
        chi2,
        chip=chip,
        final_spin=final_spin,
    )
    fMs_RD = fMs_RD - _coprecessing_deviation(
        coprecessing_deviations,
        "nu5",
        like=fMs_RD,
    )
    fMs_damp = fMs_damp + _coprecessing_deviation(
        coprecessing_deviations,
        "nu6",
        like=fMs_damp,
    )
    return fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO


def _phase_fit_nospin_python(coeffs, eta):
    """Mirror ``nospin_CV`` with one Python float per scalar operation."""

    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    return (
        coeffs[0]
        + coeffs[1] * eta
        + coeffs[2] * eta2
        + coeffs[3] * eta3
        + coeffs[4] * eta4
        + coeffs[5] * eta5
    ) / (
        coeffs[6]
        + coeffs[7] * eta
        + coeffs[8] * eta2
        + coeffs[9] * eta3
    )


def _phase_fit_equal_spin_python(coeffs, eta, spin):
    """Mirror ``Eqspin_CV`` without changing its expression tree."""

    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    spin2 = spin * spin
    spin3 = spin2 * spin
    spin4 = spin3 * spin
    numerator = spin * (
        coeffs[0]
        + coeffs[1] * spin
        + coeffs[2] * spin2
        + coeffs[3] * spin3
        + coeffs[4] * spin4
        + eta
        * (
            coeffs[5]
            + coeffs[6] * spin
            + coeffs[7] * spin2
            + coeffs[8] * spin3
            + coeffs[9] * spin4
        )
        + eta2
        * (
            coeffs[10]
            + coeffs[11] * spin
            + coeffs[12] * spin2
            + coeffs[13] * spin3
            + coeffs[14] * spin4
        )
        + eta3
        * (
            coeffs[15]
            + coeffs[16] * spin
            + coeffs[17] * spin2
            + coeffs[18] * spin3
            + coeffs[19] * spin4
        )
        + eta4
        * (
            coeffs[20]
            + coeffs[21] * spin
            + coeffs[22] * spin2
            + coeffs[23] * spin3
            + coeffs[24] * spin4
        )
    )
    denominator = (
        coeffs[25]
        + coeffs[26] * spin
        + coeffs[27] * spin2
        + coeffs[28] * spin3
    )
    return numerator / denominator


def _phase_fit_unequal_spin_python(coeffs, eta, spin, chia):
    """Mirror ``Uneqspin_CV`` without changing its expression tree."""

    chia2 = chia * chia
    # Some Torch CPU builds round TensorIterator ``sqrt`` one ULP away from
    # C/Python libm here.  The host path intentionally retains the latter,
    # which is also the scalar arithmetic used by the LAL reference model.
    delta = math.sqrt(max(1.0 - 4.0 * eta, 0.0))
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    return (
        chia
        * delta
        * eta
        * (
            coeffs[0]
            + coeffs[1] * eta
            + coeffs[2] * eta2
            + coeffs[3] * eta3
            + coeffs[4] * eta4
            + coeffs[5] * eta5
            + coeffs[6] * spin
            + coeffs[7] * spin * eta2
            + coeffs[8] * spin * eta3
        )
        + coeffs[9] * chia2 * eta
    )


def _phase_fit_rows_python_values(
    theta_values,
    coefficient_rows,
    *,
    cuda_scalar_semantics=False,
):
    """Evaluate the exact scalar fit expression tree on host values."""

    if not all(math.isfinite(value) for value in theta_values):
        return None
    if not all(
        math.isfinite(value)
        for coefficient_row in coefficient_rows
        for value in coefficient_row
    ):
        return None

    m1, m2, chi1, chi2 = theta_values
    if (
        m1 <= 0.0
        or m2 <= 0.0
        or chi1 < -1.0
        or chi1 > 1.0
        or chi2 < -1.0
        or chi2 > 1.0
    ):
        return None

    try:
        m1_s = m1 * MTSUN
        m2_s = m2 * MTSUN
        M_s = m1_s + m2_s
        # Torch specializes this scalar square to multiplication on CPU/CUDA.
        eta = m1_s * m2_s / (M_s * M_s)
        delta = math.sqrt(max(1.0 - 4.0 * eta, 0.0))
        mm1 = 0.5 * (1.0 + delta)
        mm2 = 0.5 * (1.0 - delta)
        chi_eff = mm1 * chi1 + mm2 * chi2
        spin_denominator_term = 76.0 * eta
        if cuda_scalar_semantics:
            # CUDA lowers division by this scalar constant to multiplication
            # by its rounded reciprocal. Mirror the operation the kernel
            # actually executes so the host-built rows retain CUDA bytes.
            spin_denominator_term = spin_denominator_term * (1.0 / 113.0)
        else:
            spin_denominator_term = spin_denominator_term / 113.0
        spin = (
            chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)
        ) / (1.0 - spin_denominator_term)
        mm1_2 = mm1 * mm1
        mm2_2 = mm2 * mm2
        total_spin = (mm1_2 * chi1 + mm2_2 * chi2) / (mm1_2 + mm2_2)
        chia = chi1 - chi2

        intermediates = (
            m1_s,
            m2_s,
            M_s,
            eta,
            delta,
            mm1,
            mm2,
            chi_eff,
            spin_denominator_term,
            spin,
            mm1_2,
            mm2_2,
            total_spin,
            chia,
        )
        if not all(math.isfinite(value) for value in intermediates):
            return None
        if eta <= 0.0 or eta > 0.25:
            return None

        fit_rows = []
        for row_index, coefficient_row in enumerate(coefficient_rows):
            row_spin = spin if row_index < 4 else total_spin
            no_spin = _phase_fit_nospin_python(
                coefficient_row[0:eqspin_indx],
                eta,
            )
            equal_spin = _phase_fit_equal_spin_python(
                coefficient_row[eqspin_indx:uneqspin_indx],
                eta,
                row_spin,
            )
            unequal_spin = _phase_fit_unequal_spin_python(
                coefficient_row[uneqspin_indx:],
                eta,
                row_spin,
                chia,
            )
            fit_rows.append((no_spin + equal_spin) + unequal_spin)
    except (ArithmeticError, ValueError):
        return None

    if not all(math.isfinite(value) for value in fit_rows):
        return None
    return fit_rows


def _prepare_phase_fit_rows_python(theta, phase_coeffs):
    """Return exact CPU scalar fit rows, or ``None`` to fail closed."""

    try:
        theta_values = theta.tolist()
        coefficient_rows = phase_coeffs.tolist()
    except (RuntimeError, TypeError, ValueError):
        return None

    fit_rows = _phase_fit_rows_python_values(
        theta_values,
        coefficient_rows,
    )
    if fit_rows is None:
        return None
    return torch.tensor(fit_rows, device=theta.device, dtype=theta.dtype)


def _prepare_phase_fit_rows_from_host(
    theta_values,
    phase_coeffs,
    *,
    device,
    dtype,
):
    """Build exact phase-fit rows on the host for one CUDA upload."""

    if not _phase_fit_python_scalars_host_cuda_supported(
        theta_values,
        phase_coeffs,
        device=device,
        dtype=dtype,
    ):
        return None
    coefficient_rows = _canonical_phase_fit_coefficient_rows_python()
    if coefficient_rows is None:
        return None
    fit_rows = _phase_fit_rows_python_values(
        theta_values,
        coefficient_rows,
        cuda_scalar_semantics=True,
    )
    if fit_rows is None:
        return None
    try:
        return torch.tensor(fit_rows, device=device, dtype=dtype)
    except (RuntimeError, TypeError, ValueError):
        return None


def _phase_fit_native_iterator_row(coefficient_row, powers):
    """Evaluate one independent phase-fit row with its original tree."""

    (
        eta,
        eta2,
        eta3,
        eta4,
        eta5,
        spin,
        spin2,
        spin3,
        spin4,
        chia,
        chia2,
        delta,
    ) = powers
    no_spin = (
        coefficient_row[0]
        + coefficient_row[1] * eta
        + coefficient_row[2] * eta2
        + coefficient_row[3] * eta3
        + coefficient_row[4] * eta4
        + coefficient_row[5] * eta5
    ) / (
        coefficient_row[6]
        + coefficient_row[7] * eta
        + coefficient_row[8] * eta2
        + coefficient_row[9] * eta3
    )
    equal_spin = spin * (
        coefficient_row[10]
        + coefficient_row[11] * spin
        + coefficient_row[12] * spin2
        + coefficient_row[13] * spin3
        + coefficient_row[14] * spin4
        + eta
        * (
            coefficient_row[15]
            + coefficient_row[16] * spin
            + coefficient_row[17] * spin2
            + coefficient_row[18] * spin3
            + coefficient_row[19] * spin4
        )
        + eta2
        * (
            coefficient_row[20]
            + coefficient_row[21] * spin
            + coefficient_row[22] * spin2
            + coefficient_row[23] * spin3
            + coefficient_row[24] * spin4
        )
        + eta3
        * (
            coefficient_row[25]
            + coefficient_row[26] * spin
            + coefficient_row[27] * spin2
            + coefficient_row[28] * spin3
            + coefficient_row[29] * spin4
        )
        + eta4
        * (
            coefficient_row[30]
            + coefficient_row[31] * spin
            + coefficient_row[32] * spin2
            + coefficient_row[33] * spin3
            + coefficient_row[34] * spin4
        )
    ) / (
        coefficient_row[35]
        + coefficient_row[36] * spin
        + coefficient_row[37] * spin2
        + coefficient_row[38] * spin3
    )
    unequal_spin = (
        chia
        * delta
        * eta
        * (
            coefficient_row[39]
            + coefficient_row[40] * eta
            + coefficient_row[41] * eta2
            + coefficient_row[42] * eta3
            + coefficient_row[43] * eta4
            + coefficient_row[44] * eta5
            + coefficient_row[45] * spin
            + coefficient_row[46] * spin * eta2
            + coefficient_row[47] * spin * eta3
        )
        + coefficient_row[48] * chia2 * eta
    )
    return (no_spin + equal_spin) + unequal_spin


def _prepare_phase_fit_rows_native_iterator(
    theta,
    phase_coeffs,
    *,
    _intrinsic_controls=None,
):
    """Build 13 exact CPU rows through native fixed-length iterators."""

    if not _phase_fit_native_iterator_supported(theta, phase_coeffs):
        return None
    try:
        theta_values = (
            theta.tolist()
            if _intrinsic_controls is None
            else _intrinsic_controls.theta_values
        )
        coefficient_rows = phase_coeffs.tolist()
    except (RuntimeError, TypeError, ValueError):
        return None
    if not all(map(math.isfinite, theta_values)):
        return None
    if not all(
        map(math.isfinite, itertools.chain.from_iterable(coefficient_rows))
    ):
        return None

    mass1, mass2, chi1_value, chi2_value = theta_values
    if (
        mass1 <= 0.0
        or mass2 <= 0.0
        or chi1_value < -1.0
        or chi1_value > 1.0
        or chi2_value < -1.0
        or chi2_value > 1.0
    ):
        return None

    if _intrinsic_controls is None:
        m1, m2, chi1, chi2 = theta
        m1_s = m1 * MTSUN
        m2_s = m2 * MTSUN
        M_s = m1_s + m2_s
        eta = m1_s * m2_s / (M_s**2.0)
        delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
        mm1 = 0.5 * (1.0 + delta)
        mm2 = 0.5 * (1.0 - delta)
        chi_eff = mm1 * chi1 + mm2 * chi2
        inspiral_spin = (
            chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)
        ) / (1.0 - (76.0 * eta / 113.0))
        merger_spin = (mm1**2 * chi1 + mm2**2 * chi2) / (
            mm1**2 + mm2**2
        )
        chia = chi1 - chi2

        try:
            (
                eta_value,
                inspiral_spin_value,
                merger_spin_value,
                chia_value,
                delta_value,
            ) = torch.stack(
                (eta, inspiral_spin, merger_spin, chia, delta)
            ).tolist()
        except (RuntimeError, TypeError, ValueError):
            return None
    else:
        (
            eta_value,
            inspiral_spin_value,
            merger_spin_value,
            chia_value,
            delta_value,
        ) = _intrinsic_controls.fit_values
    controls = (
        eta_value,
        inspiral_spin_value,
        merger_spin_value,
        chia_value,
        delta_value,
    )
    if (
        not all(map(math.isfinite, controls))
        or eta_value <= 0.0
        or eta_value > 0.25
    ):
        return None

    eta2 = eta_value * eta_value
    eta3 = eta2 * eta_value
    eta4 = eta3 * eta_value
    eta5 = eta4 * eta_value
    chia2 = chia_value * chia_value

    def fit_powers(spin):
        spin2 = spin * spin
        spin3 = spin2 * spin
        spin4 = spin3 * spin
        return (
            eta_value,
            eta2,
            eta3,
            eta4,
            eta5,
            spin,
            spin2,
            spin3,
            spin4,
            chia_value,
            chia2,
            delta_value,
        )

    inspiral_powers = fit_powers(inspiral_spin_value)
    merger_powers = fit_powers(merger_spin_value)
    power_rows = itertools.chain(
        itertools.repeat(inspiral_powers, 4),
        itertools.repeat(merger_powers, 9),
    )
    try:
        fit_rows = list(
            map(
                _phase_fit_native_iterator_row,
                coefficient_rows,
                power_rows,
            )
        )
    except (ArithmeticError, ValueError):
        return None
    if not all(map(math.isfinite, fit_rows)):
        return None
    return torch.tensor(fit_rows, device=theta.device, dtype=theta.dtype)


def _prepare_phase_fit_rows_torch(
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    *,
    _intrinsic_controls=None,
) -> torch.Tensor:
    """Evaluate all 13 phase collocation fits in three bulk operations."""

    native_rows = _prepare_phase_fit_rows_native_iterator(
        theta,
        phase_coeffs,
        _intrinsic_controls=_intrinsic_controls,
    )
    if native_rows is not None:
        return native_rows

    if _intrinsic_controls is None:
        m1, m2, chi1, chi2 = theta
        m1_s = m1 * MTSUN
        m2_s = m2 * MTSUN
        M_s = m1_s + m2_s
        eta = m1_s * m2_s / (M_s**2.0)
        delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
        mm1 = 0.5 * (1.0 + delta)
        mm2 = 0.5 * (1.0 - delta)
        chi_eff = mm1 * chi1 + mm2 * chi2
        S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))
        StotR = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)
        chia = chi1 - chi2
    else:
        eta = _intrinsic_controls.eta
        S = _intrinsic_controls.inspiral_spin
        StotR = _intrinsic_controls.merger_spin
        chia = _intrinsic_controls.spin_difference

    # Rows 0:4 use S; the intermediate and merger-ringdown rows use StotR.
    # Expanding a scalar is a view, and the fit helpers preserve the exact
    # per-row arithmetic tree while Torch performs the fixed row loop in C.
    spin_rows = torch.cat((S.expand(4), StotR.expand(9)))
    no_spin = IMRPhenomX_utils.nospin_CV(
        phase_coeffs[:, 0:eqspin_indx],
        eta,
    )
    equal_spin = IMRPhenomX_utils.Eqspin_CV(
        phase_coeffs[:, eqspin_indx:uneqspin_indx],
        eta,
        spin_rows,
    )
    unequal_spin = IMRPhenomX_utils.Uneqspin_CV(
        phase_coeffs[:, uneqspin_indx:],
        eta,
        spin_rows,
        chia,
    )
    return (no_spin + equal_spin) + unequal_spin


def _prepare_phase_fit_rows(
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    *,
    _intrinsic_controls=None,
) -> torch.Tensor:
    """Evaluate all phase fits with an optional exact CPU scalar path."""

    if _phase_fit_python_scalars_supported(theta, phase_coeffs):
        fit_rows = _prepare_phase_fit_rows_python(theta, phase_coeffs)
        if fit_rows is not None:
            return fit_rows
    return _prepare_phase_fit_rows_torch(
        theta,
        phase_coeffs,
        _intrinsic_controls=_intrinsic_controls,
    )


def _amp_fit_native_iterator_row(coefficient_row, powers):
    """Evaluate one independent amplitude-fit row with its original tree."""

    (
        eta,
        eta2,
        eta3,
        eta4,
        eta5,
        spin,
        spin2,
        spin3,
        spin4,
        spin5,
        chia,
        delta,
    ) = powers
    no_spin = (
        coefficient_row[0]
        + coefficient_row[1] * eta
        + coefficient_row[2] * eta2
        + coefficient_row[3] * eta3
        + coefficient_row[4] * eta4
    ) / (
        coefficient_row[5]
        + coefficient_row[6] * eta
        + coefficient_row[7] * eta2
    )
    numerator_spin0 = (
        coefficient_row[8]
        + coefficient_row[9] * eta
        + coefficient_row[10] * eta2
        + coefficient_row[11] * eta3
    )
    numerator_spin1 = (
        coefficient_row[12]
        + coefficient_row[13] * eta
        + coefficient_row[14] * eta2
        + coefficient_row[15] * eta3
    )
    numerator_spin2 = (
        coefficient_row[16]
        + coefficient_row[17] * eta
        + coefficient_row[18] * eta2
        + coefficient_row[19] * eta3
    )
    numerator_spin3 = (
        coefficient_row[20]
        + coefficient_row[21] * eta
        + coefficient_row[22] * eta2
        + coefficient_row[23] * eta3
    )
    numerator_spin4 = (
        coefficient_row[24]
        + coefficient_row[25] * eta
        + coefficient_row[26] * eta2
        + coefficient_row[27] * eta3
    )
    numerator_spin5 = (
        coefficient_row[28]
        + coefficient_row[29] * eta
        + coefficient_row[30] * eta2
        + coefficient_row[31] * eta3
    )
    equal_spin = (
        numerator_spin0
        + numerator_spin1 * spin
        + numerator_spin2 * spin2
        + numerator_spin3 * spin3
        + numerator_spin4 * spin4
        + numerator_spin5 * spin5
    ) / (
        coefficient_row[32]
        + coefficient_row[33] * spin
        + coefficient_row[34] * eta
        + coefficient_row[35] * spin2
    )
    unequal_spin = (
        chia
        * delta
        * (
            coefficient_row[36]
            + coefficient_row[37] * eta
            + coefficient_row[38] * eta2
            + coefficient_row[39] * eta3
            + coefficient_row[40] * eta4
            + coefficient_row[41] * eta5
        )
    )
    return (no_spin + equal_spin) + unequal_spin


def _amp_fit_nospin_python(coeffs, eta):
    """Mirror ``Amp_Nospin_CV`` without zero-dimensional dispatches."""

    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta**4
    numerator = (
        coeffs[0]
        + coeffs[1] * eta
        + coeffs[2] * eta2
        + coeffs[3] * eta3
        + coeffs[4] * eta4
    )
    denominator = coeffs[5] + coeffs[6] * eta + coeffs[7] * eta2
    return numerator / denominator


def _amp_fit_equal_spin_python(coeffs, eta, spin):
    """Mirror ``Amp_Eqspin_CV`` with its original expression tree."""

    eta2 = eta * eta
    eta3 = eta2 * eta
    spin2 = spin * spin
    spin3 = spin2 * spin
    spin4 = spin**4
    spin5 = spin**5
    numerator_spin0 = (
        coeffs[0] + coeffs[1] * eta + coeffs[2] * eta2 + coeffs[3] * eta3
    )
    numerator_spin1 = (
        coeffs[4] + coeffs[5] * eta + coeffs[6] * eta2 + coeffs[7] * eta3
    )
    numerator_spin2 = (
        coeffs[8] + coeffs[9] * eta + coeffs[10] * eta2 + coeffs[11] * eta3
    )
    numerator_spin3 = (
        coeffs[12]
        + coeffs[13] * eta
        + coeffs[14] * eta2
        + coeffs[15] * eta3
    )
    numerator_spin4 = (
        coeffs[16]
        + coeffs[17] * eta
        + coeffs[18] * eta2
        + coeffs[19] * eta3
    )
    numerator_spin5 = (
        coeffs[20]
        + coeffs[21] * eta
        + coeffs[22] * eta2
        + coeffs[23] * eta3
    )
    denominator = (
        coeffs[24]
        + coeffs[25] * spin
        + coeffs[26] * eta
        + coeffs[27] * spin2
    )
    return (
        numerator_spin0
        + numerator_spin1 * spin
        + numerator_spin2 * spin2
        + numerator_spin3 * spin3
        + numerator_spin4 * spin4
        + numerator_spin5 * spin5
    ) / denominator


def _amp_fit_unequal_spin_python(coeffs, eta, chia):
    """Mirror ``Amp_Uneqspin_CV`` with its original expression tree."""

    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta**4
    eta5 = eta**5
    delta = math.sqrt(max(1.0 - 4.0 * eta, 0.0))
    return (
        chia
        * delta
        * (
            coeffs[0]
            + coeffs[1] * eta
            + coeffs[2] * eta2
            + coeffs[3] * eta3
            + coeffs[4] * eta4
            + coeffs[5] * eta5
        )
    )


def _amp_fit_rows_python_values(
    theta_values,
    coefficient_rows,
):
    """Evaluate the exact ordered amplitude-fit expressions on host values."""

    if len(theta_values) != 4 or len(coefficient_rows) != 7:
        return None
    if not all(math.isfinite(value) for value in theta_values):
        return None
    if not all(
        len(coefficient_row) == 42
        and all(math.isfinite(value) for value in coefficient_row)
        for coefficient_row in coefficient_rows
    ):
        return None

    m1, m2, chi1, chi2 = theta_values
    if (
        m1 <= 0.0
        or m2 <= 0.0
        or chi1 < -1.0
        or chi1 > 1.0
        or chi2 < -1.0
        or chi2 > 1.0
    ):
        return None

    try:
        m1_s = m1 * MTSUN
        m2_s = m2 * MTSUN
        M_s = m1_s + m2_s
        # Both CPU and CUDA specialize ``tensor**2.0`` to multiplication.
        eta = m1_s * m2_s / (M_s * M_s)
        delta = math.sqrt(max(1.0 - 4.0 * eta, 0.0))
        mm1 = 0.5 * (1.0 + delta)
        mm2 = 0.5 * (1.0 - delta)
        chi_eff = mm1 * chi1 + mm2 * chi2
        spin_denominator_term = 76.0 * eta / 113.0
        spin = (
            chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)
        ) / (1.0 - spin_denominator_term)
        mm1_2 = mm1 * mm1
        mm2_2 = mm2 * mm2
        total_spin = (mm1_2 * chi1 + mm2_2 * chi2) / (mm1_2 + mm2_2)
        chia = chi1 - chi2

        intermediates = (
            m1_s,
            m2_s,
            M_s,
            eta,
            delta,
            mm1,
            mm2,
            chi_eff,
            spin_denominator_term,
            spin,
            mm1_2,
            mm2_2,
            total_spin,
            chia,
        )
        if not all(math.isfinite(value) for value in intermediates):
            return None
        if eta <= 0.0 or eta > 0.25:
            return None

        fit_rows = []
        for row_index, coefficient_row in enumerate(coefficient_rows):
            row_spin = spin if row_index < 3 else total_spin
            no_spin = _amp_fit_nospin_python(
                coefficient_row[0:amp_eqspin_indx],
                eta,
            )
            equal_spin = _amp_fit_equal_spin_python(
                coefficient_row[amp_eqspin_indx:amp_uneqspin_indx],
                eta,
                row_spin,
            )
            unequal_spin = _amp_fit_unequal_spin_python(
                coefficient_row[amp_uneqspin_indx:],
                eta,
                chia,
            )
            fit_rows.append((no_spin + equal_spin) + unequal_spin)
    except (ArithmeticError, TypeError, ValueError):
        return None

    if not all(math.isfinite(value) for value in fit_rows):
        return None
    return fit_rows


def _prepare_amp_fit_rows_python(theta, amp_coeffs):
    """Return exact CPU scalar amplitude rows, or ``None`` to fail closed."""

    if (
        amp_coeffs
        is not IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER
        or amp_coeffs._version != 0
    ):
        return None
    try:
        theta_values = theta.tolist()
    except (RuntimeError, TypeError, ValueError):
        return None
    coefficient_rows = _canonical_amp_fit_coefficient_rows_python(amp_coeffs)
    if coefficient_rows is None:
        return None

    fit_rows = _amp_fit_rows_python_values(theta_values, coefficient_rows)
    if fit_rows is None:
        return None
    return torch.tensor(fit_rows, device=theta.device, dtype=theta.dtype)


def _prepare_amp_fit_rows_from_host(
    theta_values,
    amp_coeffs,
    *,
    device,
    dtype,
):
    """Build exact amplitude-fit rows with one fixed hybrid CUDA lane."""

    if not _amp_fit_python_scalars_host_cuda_supported(
        theta_values,
        amp_coeffs,
        device=device,
        dtype=dtype,
    ):
        return None
    coefficient_rows = _canonical_amp_fit_coefficient_rows_python(amp_coeffs)
    if coefficient_rows is None:
        return None

    try:
        mass1, mass2, spin1, spin2 = theta_values
        mass1_seconds = mass1 * MTSUN
        mass2_seconds = mass2 * MTSUN
        total_mass_seconds = mass1_seconds + mass2_seconds
        eta = (
            mass1_seconds
            * mass2_seconds
            / (total_mass_seconds * total_mass_seconds)
        )
        delta = math.sqrt(max(1.0 - 4.0 * eta, 0.0))
        mass_fraction1 = 0.5 * (1.0 + delta)
        mass_fraction2 = 0.5 * (1.0 - delta)
        effective_spin = mass_fraction1 * spin1 + mass_fraction2 * spin2
        spin_denominator_term = (76.0 * eta) * (1.0 / 113.0)
        inspiral_spin = (
            effective_spin
            - (38.0 / 113.0) * eta * (spin1 + spin2)
        ) / (1.0 - spin_denominator_term)
        mass_fraction1_squared = mass_fraction1 * mass_fraction1
        mass_fraction2_squared = mass_fraction2 * mass_fraction2
        merger_spin = (
            mass_fraction1_squared * spin1
            + mass_fraction2_squared * spin2
        ) / (mass_fraction1_squared + mass_fraction2_squared)
        spin_difference = spin1 - spin2
        controls = (
            mass1_seconds,
            mass2_seconds,
            total_mass_seconds,
            eta,
            delta,
            mass_fraction1,
            mass_fraction2,
            effective_spin,
            spin_denominator_term,
            inspiral_spin,
            mass_fraction1_squared,
            mass_fraction2_squared,
            merger_spin,
            spin_difference,
        )
        if (
            not all(math.isfinite(value) for value in controls)
            or eta <= 0.0
            or eta > 0.25
        ):
            return None

        power_bases = torch.tensor(
            (eta, inspiral_spin, merger_spin),
            device=device,
            dtype=dtype,
        )
        power_rows = torch.stack(
            tuple(power_bases**exponent for exponent in (2, 3, 4, 5)),
            dim=1,
        ).tolist()
        eta_squared, eta_cubed, eta_fourth, eta_fifth = power_rows[0]
        fit_rows = []
        for row_index, coefficient_row in enumerate(coefficient_rows):
            use_inspiral_spin = row_index < 3
            row_spin = inspiral_spin if use_inspiral_spin else merger_spin
            (
                spin_squared,
                spin_cubed,
                spin_fourth,
                spin_fifth,
            ) = power_rows[1 if use_inspiral_spin else 2]
            fit_rows.append(
                _amp_fit_native_iterator_row(
                    coefficient_row,
                    (
                        eta,
                        eta_squared,
                        eta_cubed,
                        eta_fourth,
                        eta_fifth,
                        row_spin,
                        spin_squared,
                        spin_cubed,
                        spin_fourth,
                        spin_fifth,
                        spin_difference,
                        delta,
                    ),
                )
            )
        if not all(math.isfinite(value) for value in fit_rows):
            return None
        return torch.tensor(fit_rows, device=device, dtype=dtype)
    except (ArithmeticError, RuntimeError, TypeError, ValueError):
        return None


def _prepare_amp_fit_rows_native_iterator(
    theta,
    amp_coeffs,
    *,
    _intrinsic_controls=None,
):
    """Build seven exact CPU rows through native fixed-length iterators."""

    if not _amp_fit_native_iterator_supported(theta, amp_coeffs):
        return None
    try:
        theta_values = (
            theta.tolist()
            if _intrinsic_controls is None
            else _intrinsic_controls.theta_values
        )
        coefficient_rows = amp_coeffs.tolist()
    except (RuntimeError, TypeError, ValueError):
        return None
    if not all(map(math.isfinite, theta_values)):
        return None
    if not all(
        map(math.isfinite, itertools.chain.from_iterable(coefficient_rows))
    ):
        return None

    mass1, mass2, chi1_value, chi2_value = theta_values
    if (
        mass1 <= 0.0
        or mass2 <= 0.0
        or chi1_value < -1.0
        or chi1_value > 1.0
        or chi2_value < -1.0
        or chi2_value > 1.0
    ):
        return None

    if _intrinsic_controls is None:
        m1, m2, chi1, chi2 = theta
        m1_s = m1 * MTSUN
        m2_s = m2 * MTSUN
        M_s = m1_s + m2_s
        eta = m1_s * m2_s / (M_s**2.0)
        delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
        mm1 = 0.5 * (1.0 + delta)
        mm2 = 0.5 * (1.0 - delta)
        chi_eff = mm1 * chi1 + mm2 * chi2
        inspiral_spin = (
            chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)
        ) / (1.0 - (76.0 * eta / 113.0))
        merger_spin = (mm1**2 * chi1 + mm2**2 * chi2) / (
            mm1**2 + mm2**2
        )
        chia = chi1 - chi2

        try:
            (
                eta_value,
                inspiral_spin_value,
                merger_spin_value,
                chia_value,
                delta_value,
            ) = torch.stack(
                (eta, inspiral_spin, merger_spin, chia, delta)
            ).tolist()
        except (RuntimeError, TypeError, ValueError):
            return None
    else:
        (
            eta_value,
            inspiral_spin_value,
            merger_spin_value,
            chia_value,
            delta_value,
        ) = _intrinsic_controls.fit_values
    controls = (
        eta_value,
        inspiral_spin_value,
        merger_spin_value,
        chia_value,
        delta_value,
    )
    if (
        not all(map(math.isfinite, controls))
        or eta_value <= 0.0
        or eta_value > 0.25
    ):
        return None

    eta2 = eta_value * eta_value
    eta3 = eta2 * eta_value
    eta4 = eta_value**4
    eta5 = eta_value**5

    def fit_powers(spin):
        spin2 = spin * spin
        spin3 = spin2 * spin
        return (
            eta_value,
            eta2,
            eta3,
            eta4,
            eta5,
            spin,
            spin2,
            spin3,
            spin**4,
            spin**5,
            chia_value,
            delta_value,
        )

    inspiral_powers = fit_powers(inspiral_spin_value)
    merger_powers = fit_powers(merger_spin_value)
    power_rows = itertools.chain(
        itertools.repeat(inspiral_powers, 3),
        itertools.repeat(merger_powers, 4),
    )
    try:
        fit_rows = list(
            map(
                _amp_fit_native_iterator_row,
                coefficient_rows,
                power_rows,
            )
        )
    except (ArithmeticError, ValueError):
        return None
    if not all(map(math.isfinite, fit_rows)):
        return None
    return torch.tensor(fit_rows, device=theta.device, dtype=theta.dtype)


def _prepare_amp_fit_rows_torch(
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    *,
    _intrinsic_controls=None,
) -> torch.Tensor:
    """Evaluate all seven amplitude collocation fits in three bulk operations."""

    native_rows = _prepare_amp_fit_rows_native_iterator(
        theta,
        amp_coeffs,
        _intrinsic_controls=_intrinsic_controls,
    )
    if native_rows is not None:
        return native_rows

    if _intrinsic_controls is None:
        m1, m2, chi1, chi2 = theta
        m1_s = m1 * MTSUN
        m2_s = m2 * MTSUN
        M_s = m1_s + m2_s
        eta = m1_s * m2_s / (M_s**2.0)
        delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
        mm1 = 0.5 * (1.0 + delta)
        mm2 = 0.5 * (1.0 - delta)
        chi_eff = mm1 * chi1 + mm2 * chi2
        S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))
        StotR = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)
        chia = chi1 - chi2
    else:
        eta = _intrinsic_controls.eta
        S = _intrinsic_controls.inspiral_spin
        StotR = _intrinsic_controls.merger_spin
        chia = _intrinsic_controls.spin_difference

    spin_rows = torch.cat((S.expand(3), StotR.expand(4)))
    fit_powers = IMRPhenomX_utils._prepare_amp_fit_powers(eta, spin_rows)
    no_spin = IMRPhenomX_utils.Amp_Nospin_CV(
        amp_coeffs[:, 0:amp_eqspin_indx],
        eta,
        powers=fit_powers,
    )
    equal_spin = IMRPhenomX_utils.Amp_Eqspin_CV(
        amp_coeffs[:, amp_eqspin_indx:amp_uneqspin_indx],
        eta,
        spin_rows,
        powers=fit_powers,
    )
    unequal_spin = IMRPhenomX_utils.Amp_Uneqspin_CV(
        amp_coeffs[:, amp_uneqspin_indx:],
        eta,
        spin_rows,
        chia,
        powers=fit_powers,
    )
    return (no_spin + equal_spin) + unequal_spin


def _prepare_amp_fit_rows(
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    *,
    _intrinsic_controls=None,
) -> torch.Tensor:
    """Evaluate amplitude fits with an optional exact CPU scalar path."""

    if _amp_fit_python_scalars_supported(theta, amp_coeffs):
        fit_rows = _prepare_amp_fit_rows_python(theta, amp_coeffs)
        if fit_rows is not None:
            return fit_rows
    return _prepare_amp_fit_rows_torch(
        theta,
        amp_coeffs,
        _intrinsic_controls=_intrinsic_controls,
    )


def _prepare_inspiral_phase(
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    fit_rows: torch.Tensor | None = None,
    *,
    bulk_collocation: bool = False,
    _solve_info: list[torch.Tensor] | None = None,
    _cutoff_fMs=None,
    _intrinsic_controls=None,
    _host_scalars=None,
) -> _InspiralPhasePlan:
    """Prepare the frequency-independent inspiral phase coefficients."""
    host_values = (
        None
        if _host_scalars is None
        else _inspiral_phase_host_scalar_values(
            theta,
            phase_coeffs,
            fit_rows,
            bulk_collocation=bulk_collocation,
            _solve_info=_solve_info,
            _cutoff_fMs=_cutoff_fMs,
            _intrinsic_controls=_intrinsic_controls,
            _host_scalars=_host_scalars,
        )
    )
    if host_values is not None:
        (
            m1,
            m2,
            chi1,
            chi2,
            eta,
            S,
            _,
            chia,
            delta,
            CV_phase_Ins0,
            CV_phase_Ins1,
            CV_phase_Ins2,
            CV_phase_Ins3,
            host_log_two,
            host_log_pi,
        ) = host_values
        eta2 = eta * eta
        eta3 = eta2 * eta
    elif _intrinsic_controls is None:
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
    else:
        m1 = _intrinsic_controls.mass1
        m2 = _intrinsic_controls.mass2
        chi1 = _intrinsic_controls.spin1
        chi2 = _intrinsic_controls.spin2
        eta = _intrinsic_controls.eta
        eta2 = eta * eta
        eta3 = eta2 * eta
        delta = _intrinsic_controls.delta
        S = _intrinsic_controls.inspiral_spin
        chia = _intrinsic_controls.spin_difference

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
            - (
                13696.0
                * (
                    host_log_two
                    if host_values is not None
                    else jnp.log(2.0)
                )
            )
            / 21.0
            - (
                6848.0
                * (
                    host_log_pi
                    if host_values is not None
                    else jnp.log(PI)
                )
            )
            / 63.0
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
            * (
                -1.0
                + (
                    host_log_pi
                    if host_values is not None
                    else jnp.log(PI)
                )
            )
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

    gpoints4 = torch.as_tensor(
        [0.0, 0.25, 0.75, 1.0],
        dtype=theta.dtype,
        device=theta.device,
    )
    # Note that they do not use 4.1 from 2001.11412, they actually use
    # (Cos(i PI / 3) + 1)/2

    if _cutoff_fMs is None:
        _, _, fMs_MECO, _ = IMRPhenomX_utils.get_cutoff_fMs(
            m1,
            m2,
            chi1,
            chi2,
        )
    else:
        fMs_MECO = _cutoff_fMs[2]

    fMs_PhaseInsMin = 0.0026
    fMs_PhaseInsMax = 1.020 * fMs_MECO

    deltax = fMs_PhaseInsMax - fMs_PhaseInsMin
    xmin = fMs_PhaseInsMin

    if bulk_collocation:
        CP_phase_Ins = gpoints4 * deltax + xmin
    else:
        CP_phase_Ins0 = gpoints4[0] * deltax + xmin
        CP_phase_Ins1 = gpoints4[1] * deltax + xmin
        CP_phase_Ins2 = gpoints4[2] * deltax + xmin
        CP_phase_Ins3 = gpoints4[3] * deltax + xmin

    if host_values is not None:
        pass
    elif fit_rows is None:
        CV_phase_Ins0 = (
            IMRPhenomX_utils.nospin_CV(phase_coeffs[0, 0:eqspin_indx], eta)
            + IMRPhenomX_utils.Eqspin_CV(
                phase_coeffs[0, eqspin_indx:uneqspin_indx], eta, S
            )
            + IMRPhenomX_utils.Uneqspin_CV(
                phase_coeffs[0, uneqspin_indx:], eta, S, chia
            )
        )
        CV_phase_Ins1 = (
            IMRPhenomX_utils.nospin_CV(phase_coeffs[1, 0:eqspin_indx], eta)
            + IMRPhenomX_utils.Eqspin_CV(
                phase_coeffs[1, eqspin_indx:uneqspin_indx], eta, S
            )
            + IMRPhenomX_utils.Uneqspin_CV(
                phase_coeffs[1, uneqspin_indx:], eta, S, chia
            )
        )
        CV_phase_Ins2 = (
            IMRPhenomX_utils.nospin_CV(phase_coeffs[2, 0:eqspin_indx], eta)
            + IMRPhenomX_utils.Eqspin_CV(
                phase_coeffs[2, eqspin_indx:uneqspin_indx], eta, S
            )
            + IMRPhenomX_utils.Uneqspin_CV(
                phase_coeffs[2, uneqspin_indx:], eta, S, chia
            )
        )

        # This fit disagrees slightly with WF4py at non-zero spin.
        CV_phase_Ins3 = (
            IMRPhenomX_utils.nospin_CV(phase_coeffs[3, 0:eqspin_indx], eta)
            + IMRPhenomX_utils.Eqspin_CV(
                phase_coeffs[3, eqspin_indx:uneqspin_indx], eta, S
            )
            + IMRPhenomX_utils.Uneqspin_CV(
                phase_coeffs[3, uneqspin_indx:], eta, S, chia
            )
        )
    else:
        CV_phase_Ins0, CV_phase_Ins1, CV_phase_Ins2, CV_phase_Ins3 = fit_rows[0:4]

    # See line 1322 of https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__internals_8c_source.html
    CV_phase_Ins0 = CV_phase_Ins0 + CV_phase_Ins2
    CV_phase_Ins1 = CV_phase_Ins1 + CV_phase_Ins2
    CV_phase_Ins3 = CV_phase_Ins3 + CV_phase_Ins2

    if bulk_collocation:
        A = torch.stack(
            (
                torch.ones_like(CP_phase_Ins),
                CP_phase_Ins ** (1.0 / 3.0),
                CP_phase_Ins ** (2.0 / 3.0),
                CP_phase_Ins,
            ),
            dim=1,
        )
    else:
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
    if isinstance(CV_phase_Ins0, torch.Tensor):
        b = torch.stack(
            (
                CV_phase_Ins0,
                CV_phase_Ins1,
                CV_phase_Ins2,
                CV_phase_Ins3,
            )
        ).to(device=A.device, dtype=A.dtype)
    else:
        b = torch.tensor(
            [
                CV_phase_Ins0,
                CV_phase_Ins1,
                CV_phase_Ins2,
                CV_phase_Ins3,
            ],
            device=A.device,
            dtype=A.dtype,
        )

    if _solve_info is None:
        coeffs_Ins = jnp.linalg.solve(A, b)
    else:
        coeffs_Ins, info = torch.linalg.solve_ex(
            A,
            b,
            check_errors=False,
        )
        _solve_info.append(info)

    sigma1 = (-5.0 / 3.0) * coeffs_Ins[0]
    sigma2 = (-5.0 / 4.0) * coeffs_Ins[1]
    sigma3 = (-5.0 / 5.0) * coeffs_Ins[2]
    sigma4 = (-5.0 / 6.0) * coeffs_Ins[3]

    plan = _InspiralPhasePlan(
        phi0,
        phi1,
        phi2,
        phi3,
        phi4,
        phi5,
        phi5L,
        phi6,
        phi6L,
        phi7,
        phi8,
        phi8L,
        sigma1,
        sigma2,
        sigma3,
        sigma4,
    )
    if host_values is None:
        return plan
    owned_plan = _owned_inspiral_phase_host_plan(plan, theta)
    if owned_plan is not None:
        return owned_plan
    return _prepare_inspiral_phase(
        theta,
        phase_coeffs,
        fit_rows,
        bulk_collocation=bulk_collocation,
        _solve_info=_solve_info,
        _cutoff_fMs=_cutoff_fMs,
        _intrinsic_controls=_intrinsic_controls,
        _host_scalars=None,
    )


def _scripted_phase_ansatz_cpu_runtime_supported() -> bool:
    """Accept only ordinary eager execution observable as plain Torch ops."""

    if IMRPhenomX_utils._TRUSTED_PLAIN_REQUEST.get():
        return True

    for function in (
        getattr(torch.jit, "is_scripting", None),
        getattr(torch.jit, "is_tracing", None),
        getattr(getattr(torch, "compiler", None), "is_compiling", None),
        getattr(getattr(torch, "_dynamo", None), "is_compiling", None),
    ):
        if function is None:
            return False
        try:
            if function():
                return False
        except Exception:
            return False

    torch_c = getattr(torch, "_C", None)
    tracing_state = getattr(torch_c, "_get_tracing_state", None)
    if tracing_state is None:
        return False
    try:
        if tracing_state() is not None:
            return False
    except Exception:
        return False

    if getattr(torch.autograd.forward_ad, "_current_level", None) != -1:
        return False
    functorch = getattr(torch_c, "_functorch", None)
    dynamic_depth = getattr(functorch, "get_dynamic_layer_stack_depth", None)
    if dynamic_depth is None:
        return False
    try:
        if dynamic_depth() != 0:
            return False
    except Exception:
        return False

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
        if autocast_enabled("cpu"):
            return False
    except Exception:
        return False
    return True


def _scripted_phase_ansatz_cpu_plain_request_tree_supported(value) -> bool:
    """Accept only recursively immutable built-ins below a request dict."""

    if type(value) in (type(None), bool, int, float, str):
        return True
    if type(value) is tuple:
        return all(
            _scripted_phase_ansatz_cpu_plain_request_tree_supported(item)
            for item in value
        )
    return False


def _scripted_phase_ansatz_cpu_plain_request_parameters_supported(
    params,
) -> bool:
    """Accept one exact top dict without shared mutable descendants."""

    return type(params) is dict and all(
        type(key) is str
        and _scripted_phase_ansatz_cpu_plain_request_tree_supported(value)
        for key, value in params.items()
    )


def _scripted_phase_ansatz_cpu_plain_request_supported(params) -> bool:
    """Prove a public XAS request once before its internal tensor creation."""

    return (
        _scripted_phase_ansatz_cpu_enabled()
        and _scripted_phase_ansatz_cpu_plain_request_parameters_supported(
            params
        )
        and _scripted_phase_ansatz_cpu_runtime_supported()
    )


def _request_proof_lifecycle_supported() -> bool:
    """Revalidate the execution state captured by one private proof."""

    state = _scheme.mgr.state
    return (
        type(state) is _scheme.TorchScheme
        and state.torch_device.type == "cpu"
        and torch.get_num_threads() == 1
        and torch.is_grad_enabled()
        and not torch.is_inference_mode_enabled()
        and sys.gettrace() is None
        and sys.getprofile() is None
        and _python_intermediate_amp_runtime_supported()
    )


def _scripted_phase_ansatz_cpu_tensor_supported(value, like) -> bool:
    """Validate one plain scalar coefficient without excluding reverse AD."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.device == like.device
        and value.dtype is like.dtype
        and value.ndim == 0
        and not value.is_conj()
        and not value.is_neg()
    )


def _scripted_phase_ansatz_cpu_frequency_supported(frequency) -> bool:
    """Validate one plain binary64 CPU scalar or frequency vector."""

    return (
        type(frequency) is torch.Tensor
        and frequency.layout is torch.strided
        and frequency.device.type == "cpu"
        and frequency.dtype is torch.float64
        and frequency.ndim in (0, 1)
        and frequency.is_contiguous()
        and frequency.storage_offset() == 0
        and frequency._base is None
        and not frequency.is_conj()
        and not frequency.is_neg()
        and not IMRPhenomX_utils._tensor_has_forward_ad(frequency)
    )


def _maybe_prequalify_scripted_phase_ansatz_cpu_plans(
    inspiral,
    intermediate,
    mergerringdown,
    like,
    *,
    _request_proof=None,
):
    """Validate three fixed schemas once for one request-local phase plan."""

    qualified = _request_qualify_phase_ansatzes(
        inspiral,
        intermediate,
        mergerringdown,
        _request_proof,
    )
    if qualified is not None:
        return qualified
    if (
        not _scripted_phase_ansatz_cpu_enabled()
        or not _scripted_phase_ansatz_cpu_runtime_supported()
        or not _scripted_phase_ansatz_cpu_frequency_supported(like)
        or type(inspiral) is not _InspiralPhasePlan
        or type(intermediate) is not _IntermediatePhasePlan
        or type(mergerringdown) is not _MergerRingdownPhasePlan
    ):
        return inspiral, intermediate, mergerringdown
    if IMRPhenomX_utils._TRUSTED_PLAIN_REQUEST.get():
        # These exact plan classes were just constructed by a public request
        # whose complete parameter tree and runtime were validated once.  The
        # named-tuple constructors still enforce the fixed arities here.
        return (
            _ScriptedInspiralPhasePlan._make(inspiral),
            _ScriptedIntermediatePhasePlan._make(intermediate),
            _ScriptedMergerRingdownPhasePlan._make(mergerringdown),
        )
    scalar_positions = (0, 1, 5, 8)
    if not (
        all(type(inspiral[index]) is float for index in scalar_positions)
        and all(
            _scripted_phase_ansatz_cpu_tensor_supported(
                inspiral[index],
                like,
            )
            for index in _INSPIRAL_PHASE_HOST_TENSOR_POSITIONS
        )
        and all(
            _scripted_phase_ansatz_cpu_tensor_supported(value, like)
            for value in intermediate
        )
        and all(
            _scripted_phase_ansatz_cpu_tensor_supported(value, like)
            for value in mergerringdown
        )
    ):
        return inspiral, intermediate, mergerringdown
    return (
        _ScriptedInspiralPhasePlan._make(inspiral),
        _ScriptedIntermediatePhasePlan._make(intermediate),
        _ScriptedMergerRingdownPhasePlan._make(mergerringdown),
    )


def _cuda_graph_phase_ansatz_scalar_tensor_supported(value, like) -> bool:
    """Accept one exact plain CUDA binary64 scalar plan coefficient."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.device == like.device
        and value.dtype is torch.float64
        and value.ndim == 0
        and not value.is_conj()
        and not value.is_neg()
        and not value.requires_grad
        and value.grad_fn is None
    )


def _maybe_prequalify_cuda_graph_phase_ansatz_plans(
    inspiral,
    intermediate,
    mergerringdown,
    like,
):
    """Mark only the fixed byte-qualified CUDA triplet schemas."""

    if (
        type(inspiral) is not _InspiralPhasePlan
        or type(intermediate) is not _IntermediatePhasePlan
        or type(mergerringdown) is not _MergerRingdownPhasePlan
        or not _cuda_graph_phase_ansatz_enabled()
        or not torch.cuda.is_available()
        or type(like) is not torch.Tensor
        or like.layout is not torch.strided
        or like.device.type != "cuda"
        or like.device.index is None
        or like.dtype is not torch.float64
        or like.shape != (4,)
        or not like.is_contiguous()
        or like.storage_offset() != 0
        or like._base is not None
        or like.is_conj()
        or like.is_neg()
        or like.requires_grad
        or like.grad_fn is not None
    ):
        return inspiral, intermediate, mergerringdown
    try:
        if torch.cuda.is_current_stream_capturing():
            return inspiral, intermediate, mergerringdown
        torch.cuda.current_stream(like.device)
    except Exception:
        return inspiral, intermediate, mergerringdown
    scalar_positions = (0, 1, 5, 8)
    if not (
        all(type(inspiral[index]) is float for index in scalar_positions)
        and all(
            _cuda_graph_phase_ansatz_scalar_tensor_supported(
                inspiral[index],
                like,
            )
            for index in _INSPIRAL_PHASE_HOST_TENSOR_POSITIONS
        )
        and all(
            _cuda_graph_phase_ansatz_scalar_tensor_supported(value, like)
            for value in intermediate
        )
        and all(
            _cuda_graph_phase_ansatz_scalar_tensor_supported(value, like)
            for value in mergerringdown
        )
    ):
        return inspiral, intermediate, mergerringdown
    return (
        _CudaGraphInspiralPhasePlan._make(inspiral),
        _CudaGraphIntermediatePhasePlan._make(intermediate),
        _CudaGraphMergerRingdownPhasePlan._make(mergerringdown),
    )


def _scripted_inspiral_phase_ansatz_source(
    frequency: torch.Tensor,
    phi0: float,
    phi1: float,
    phi2: torch.Tensor,
    phi3: torch.Tensor,
    phi4: torch.Tensor,
    phi5: float,
    phi5_l: torch.Tensor,
    phi6: torch.Tensor,
    phi6_l: float,
    phi7: torch.Tensor,
    phi8: torch.Tensor,
    phi8_l: torch.Tensor,
    sigma1: torch.Tensor,
    sigma2: torch.Tensor,
    sigma3: torch.Tensor,
    sigma4: torch.Tensor,
    normalization: float,
) -> torch.Tensor:
    """Straight-line inspiral expression compiled without reassociation."""

    f13 = frequency ** (1.0 / 3.0)
    f23 = f13 * f13
    f43 = frequency * f13
    f53 = frequency * f23
    f2 = frequency * frequency
    f73 = f2 * f13
    f83 = f2 * f23
    f3 = f2 * frequency
    f103 = f3 * f13
    f113 = f3 * f23
    log_f = torch.log(frequency)
    phase_tf2 = (
        phi0
        + phi1 * f13
        + phi2 * f23
        + phi3 * frequency
        + phi4 * f43
        + phi5 * f53
        + phi5_l * f53 * log_f
        + phi6 * f2
        + phi6_l * f2 * log_f
        + phi7 * f73
        + phi8 * f83
        + phi8_l * f83 * log_f
    )
    phase_inspiral = phase_tf2 + (
        sigma1 * f83 + sigma2 * f3 + sigma3 * f103 + sigma4 * f113
    )
    return phase_inspiral * normalization / f53


def _scripted_intermediate_phase_ansatz_source(
    frequency: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    b3: torch.Tensor,
    b4: torch.Tensor,
    c_l: torch.Tensor,
    f_rd: torch.Tensor,
    f_damp: torch.Tensor,
) -> torch.Tensor:
    """Straight-line intermediate expression compiled without reassociation."""

    return (
        b0 * frequency
        + b1 * torch.log(frequency)
        - b2 * (frequency**-1.0)
        - b3 * (frequency**-2.0) / 2.0
        - (b4 * (frequency**-3.0) / 3.0)
        + (2.0 * c_l * torch.atan((frequency - f_rd) / (2.0 * f_damp)))
        / f_damp
    )


def _scripted_mergerringdown_phase_ansatz_source(
    frequency: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c4_over_3: torch.Tensor,
    c_l_over_f_damp: torch.Tensor,
    f_rd: torch.Tensor,
    f_damp: torch.Tensor,
    c_l: torch.Tensor,
    cv_phase_rd0: torch.Tensor,
) -> torch.Tensor:
    """Straight-line ringdown expression compiled without reassociation."""

    del c_l, cv_phase_rd0
    return (
        c0 * frequency
        + 1.5 * c1 * (frequency ** (2.0 / 3.0))
        - c2 * (frequency**-1.0)
        - c4_over_3 * (frequency**-3.0)
        + c_l_over_f_damp * torch.atan((frequency - f_rd) / f_damp)
    )


def _scripted_phase_ansatz_triplet_source(
    inspiral_frequency: torch.Tensor,
    intermediate_frequency: torch.Tensor,
    ringdown_frequency: torch.Tensor,
    phi0: float,
    phi1: float,
    phi2: torch.Tensor,
    phi3: torch.Tensor,
    phi4: torch.Tensor,
    phi5: float,
    phi5_l: torch.Tensor,
    phi6: torch.Tensor,
    phi6_l: float,
    phi7: torch.Tensor,
    phi8: torch.Tensor,
    phi8_l: torch.Tensor,
    sigma1: torch.Tensor,
    sigma2: torch.Tensor,
    sigma3: torch.Tensor,
    sigma4: torch.Tensor,
    normalization: float,
    b0: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    b3: torch.Tensor,
    b4: torch.Tensor,
    intermediate_c_l: torch.Tensor,
    intermediate_f_rd: torch.Tensor,
    intermediate_f_damp: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c4_over_3: torch.Tensor,
    c_l_over_f_damp: torch.Tensor,
    ringdown_f_rd: torch.Tensor,
    ringdown_f_damp: torch.Tensor,
    ringdown_c_l: torch.Tensor,
    cv_phase_rd0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the three independent phase regions through one boundary."""

    inspiral = _scripted_inspiral_phase_ansatz_source(
        inspiral_frequency,
        phi0,
        phi1,
        phi2,
        phi3,
        phi4,
        phi5,
        phi5_l,
        phi6,
        phi6_l,
        phi7,
        phi8,
        phi8_l,
        sigma1,
        sigma2,
        sigma3,
        sigma4,
        normalization,
    )
    intermediate = _scripted_intermediate_phase_ansatz_source(
        intermediate_frequency,
        b0,
        b1,
        b2,
        b3,
        b4,
        intermediate_c_l,
        intermediate_f_rd,
        intermediate_f_damp,
    )
    ringdown = _scripted_mergerringdown_phase_ansatz_source(
        ringdown_frequency,
        c0,
        c1,
        c2,
        c4_over_3,
        c_l_over_f_damp,
        ringdown_f_rd,
        ringdown_f_damp,
        ringdown_c_l,
        cv_phase_rd0,
    )
    return inspiral, intermediate, ringdown


def _cuda_graph_phase_ansatz_arguments(frequency, plan):
    """Flatten one qualified plan in the captured source's exact order."""

    return (
        frequency,
        frequency,
        frequency,
        *plan.inspiral,
        _INSPIRAL_PHASE_NORMALIZATION,
        *plan.intermediate,
        *plan.mergerringdown,
    )


def _cuda_graph_phase_ansatz_runtime_supported(arguments) -> bool:
    """Fail closed outside the byte-qualified CUDA replay lifecycle."""

    if (
        os.getpid() <= 0
        or not torch.cuda.is_available()
        or not torch.is_grad_enabled()
        or torch.is_inference_mode_enabled()
        or not callable(getattr(torch.cuda, "CUDAGraph", None))
        or not callable(getattr(torch.cuda, "Stream", None))
        or not callable(getattr(torch.cuda, "graph", None))
    ):
        return False
    for function in (
        getattr(torch.jit, "is_scripting", None),
        getattr(torch.jit, "is_tracing", None),
        getattr(getattr(torch, "compiler", None), "is_compiling", None),
        getattr(getattr(torch, "_dynamo", None), "is_compiling", None),
    ):
        if function is None:
            return False
        try:
            if function():
                return False
        except Exception:
            return False

    torch_c = getattr(torch, "_C", None)
    tracing_state = getattr(torch_c, "_get_tracing_state", None)
    if tracing_state is None:
        return False
    try:
        if tracing_state() is not None:
            return False
        if getattr(torch.autograd.forward_ad, "_current_level", None) != -1:
            return False
        functorch = getattr(torch_c, "_functorch", None)
        dynamic_depth = getattr(
            functorch,
            "get_dynamic_layer_stack_depth",
            None,
        )
        if dynamic_depth is None or dynamic_depth() != 0:
            return False
        for name in (
            "_len_torch_dispatch_stack",
            "_len_torch_function_stack",
        ):
            stack_length = getattr(torch_c, name, None)
            if stack_length is None or stack_length() != 0:
                return False
        if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled(
            "cuda"
        ):
            return False
        if torch.is_anomaly_enabled():
            return False
        if torch.are_deterministic_algorithms_enabled():
            return False
        if torch.get_deterministic_debug_mode() != 0:
            return False
        if torch.cuda.get_sync_debug_mode() != 0:
            return False
        bad_fork = getattr(torch.cuda, "_is_in_bad_fork", None)
        if bad_fork is not None and bad_fork():
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
    except Exception:
        return False

    like = arguments[0]
    if not (
        type(like) is torch.Tensor
        and like.layout is torch.strided
        and like.device.type == "cuda"
        and like.device.index is not None
        and like.dtype is torch.float64
        and like.ndim == 1
        and not like.is_conj()
        and not like.is_neg()
        and arguments[1] is like
        and arguments[2] is like
    ):
        return False
    for position, value in enumerate(arguments):
        if type(value) is torch.Tensor:
            if not (
                value.layout is torch.strided
                and value.device == like.device
                and value.dtype is torch.float64
                and value.ndim == (1 if position < 3 else 0)
                and not value.is_conj()
                and not value.is_neg()
                and not value.requires_grad
                and value.grad_fn is None
            ):
                return False
        elif type(value) is not float:
            return False
    try:
        stream = torch.cuda.current_stream(like.device)
        if stream.device != like.device or type(stream.cuda_stream) is not int:
            return False
    except Exception:
        return False
    return True


def _cuda_graph_phase_ansatz_key(arguments):
    """Return an exact process/thread/device/stream/schema cache key."""

    tensor_aliases = {}
    descriptors = []
    for value in arguments:
        if type(value) is torch.Tensor:
            object_id = id(value)
            alias = tensor_aliases.setdefault(object_id, len(tensor_aliases))
            descriptors.append(
                (
                    "tensor",
                    alias,
                    value.device.index,
                    value.dtype,
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.storage_offset(),
                )
            )
        elif type(value) is float:
            descriptors.append(("float", struct.pack(">d", value)))
        else:
            return None
    stream = torch.cuda.current_stream(arguments[0].device)
    return (
        os.getpid(),
        threading.get_ident(),
        arguments[0].device,
        stream.cuda_stream,
        tuple(descriptors),
        bool(torch.is_grad_enabled()),
        bool(torch.is_inference_mode_enabled()),
        torch.get_default_dtype(),
        _phase_plan_cuda_solve_graph_environment(),
    )


def _cuda_graph_phase_ansatz_static_arguments(arguments):
    """Clone tensor aliases once and retain all Python scalar constants."""

    clones = {}
    static_arguments = []
    copy_positions = []
    for position, value in enumerate(arguments):
        if type(value) is not torch.Tensor:
            static_arguments.append(value)
            continue
        object_id = id(value)
        static = clones.get(object_id)
        if static is None:
            static = value.detach().clone()
            clones[object_id] = static
            copy_positions.append(position)
        static_arguments.append(static)
    return tuple(static_arguments), tuple(copy_positions)


def _build_cuda_graph_phase_ansatz(arguments):
    """Warm and capture the unchanged eager triplet on a private stream."""

    static_arguments, copy_positions = (
        _cuda_graph_phase_ansatz_static_arguments(arguments)
    )
    device = arguments[0].device
    origin = torch.cuda.current_stream(device)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(origin)
    with torch.cuda.stream(capture_stream), torch.no_grad():
        for _ in range(3):
            _scripted_phase_ansatz_triplet_source(*static_arguments)
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream), torch.no_grad():
        static_outputs = _scripted_phase_ansatz_triplet_source(
            *static_arguments
        )
    origin.wait_stream(capture_stream)
    return _CudaGraphPhaseAnsatzState(
        graph=graph,
        static_arguments=static_arguments,
        copy_positions=copy_positions,
        static_outputs=static_outputs,
        replay_stream_pointer=origin.cuda_stream,
        capture_stream=capture_stream,
    )


def _replay_cuda_graph_phase_ansatz(state, arguments):
    """Copy inputs, replay on the keyed stream, and return owned outputs."""

    current_stream = torch.cuda.current_stream(arguments[0].device)
    if current_stream.cuda_stream != state.replay_stream_pointer:
        raise RuntimeError("CUDA phase-ansatz replay stream changed")
    with torch.no_grad():
        for position in state.copy_positions:
            state.static_arguments[position].copy_(arguments[position])
        state.graph.replay()
        outputs = tuple(value.clone() for value in state.static_outputs)
        # A different stream key may evict this entry immediately after the
        # lock is released. Keep graph storage alive until queued work ends.
        for value in state.static_arguments:
            if type(value) is torch.Tensor:
                value.record_stream(current_stream)
        for value in state.static_outputs:
            value.record_stream(current_stream)
    return outputs


def _cuda_graph_phase_ansatz_raw_equal(reference, candidate) -> bool:
    """Compare all public binary64 output bytes during cold calibration."""

    return all(
        torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
        for left, right in zip(reference, candidate, strict=True)
    )


def _cuda_graph_phase_ansatz_store(key, state) -> None:
    """Insert one graph while bounding process-local device-memory use."""

    _CUDA_GRAPH_PHASE_ANSATZ_CACHE.pop(key, None)
    _CUDA_GRAPH_PHASE_ANSATZ_CACHE[key] = state
    while (
        len(_CUDA_GRAPH_PHASE_ANSATZ_CACHE)
        > _CUDA_GRAPH_PHASE_ANSATZ_MAX_ENTRIES
    ):
        _CUDA_GRAPH_PHASE_ANSATZ_CACHE.popitem(last=False)


def _cuda_graph_phase_ansatz_remember_failure(key) -> None:
    """Fail closed for one key until its bounded cache is cleared."""

    _CUDA_GRAPH_PHASE_ANSATZ_CACHE.pop(key, None)
    _CUDA_GRAPH_PHASE_ANSATZ_FAILURES.pop(key, None)
    _CUDA_GRAPH_PHASE_ANSATZ_FAILURES[key] = None
    while (
        len(_CUDA_GRAPH_PHASE_ANSATZ_FAILURES)
        > _CUDA_GRAPH_PHASE_ANSATZ_MAX_ENTRIES
    ):
        _CUDA_GRAPH_PHASE_ANSATZ_FAILURES.popitem(last=False)


def _ensure_cuda_graph_phase_ansatz_process() -> None:
    """Discard inherited CUDA state if an at-fork callback was unavailable."""

    if _CUDA_GRAPH_PHASE_ANSATZ_PID != os.getpid():
        _reset_cuda_graph_phase_ansatz_after_fork()


def _clear_cuda_graph_phase_ansatz_cache() -> None:
    """Release cached graphs and remembered failures in this process."""

    _ensure_cuda_graph_phase_ansatz_process()
    with _CUDA_GRAPH_PHASE_ANSATZ_LOCK:
        _CUDA_GRAPH_PHASE_ANSATZ_CACHE.clear()
        _CUDA_GRAPH_PHASE_ANSATZ_FAILURES.clear()


def _cuda_graph_phase_ansatz_cache_state():
    """Expose only bounded lifecycle counts for focused debug tests."""

    _ensure_cuda_graph_phase_ansatz_process()
    with _CUDA_GRAPH_PHASE_ANSATZ_LOCK:
        return {
            "pid": _CUDA_GRAPH_PHASE_ANSATZ_PID,
            "entries": len(_CUDA_GRAPH_PHASE_ANSATZ_CACHE),
            "failures": len(_CUDA_GRAPH_PHASE_ANSATZ_FAILURES),
            "max_entries": _CUDA_GRAPH_PHASE_ANSATZ_MAX_ENTRIES,
        }


def _evaluate_cuda_graph_phase_ansatz_triplet(frequency, plan):
    """Return exact owned captured regions, or decline to ordinary eager."""

    if not (
        type(plan.inspiral) is _CudaGraphInspiralPhasePlan
        and type(plan.intermediate) is _CudaGraphIntermediatePhasePlan
        and type(plan.mergerringdown) is _CudaGraphMergerRingdownPhasePlan
        and _cuda_graph_phase_ansatz_enabled()
    ):
        return None
    arguments = _cuda_graph_phase_ansatz_arguments(frequency, plan)
    if not _cuda_graph_phase_ansatz_runtime_supported(arguments):
        return None
    _ensure_cuda_graph_phase_ansatz_process()
    key = None
    try:
        with _CUDA_GRAPH_PHASE_ANSATZ_LOCK:
            key = _cuda_graph_phase_ansatz_key(arguments)
            if key is None:
                return None
            state = _CUDA_GRAPH_PHASE_ANSATZ_CACHE.pop(key, None)
            if state is not None:
                _CUDA_GRAPH_PHASE_ANSATZ_CACHE[key] = state
                return _replay_cuda_graph_phase_ansatz(state, arguments)
            if key in _CUDA_GRAPH_PHASE_ANSATZ_FAILURES:
                return None

            reference = _scripted_phase_ansatz_triplet_source(*arguments)
            state = _build_cuda_graph_phase_ansatz(arguments)
            replay = _replay_cuda_graph_phase_ansatz(state, arguments)
            if not _cuda_graph_phase_ansatz_raw_equal(reference, replay):
                raise RuntimeError("CUDA phase-ansatz graph changed bytes")
            _cuda_graph_phase_ansatz_store(key, state)
            return reference
    except Exception:
        if key is not None:
            with _CUDA_GRAPH_PHASE_ANSATZ_LOCK:
                _cuda_graph_phase_ansatz_remember_failure(key)
        return None


def _load_scripted_phase_ansatz_cpu_executor():
    """Compile the fixed schemas only after their debug gate is selected."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.jit.script(_scripted_phase_ansatz_triplet_source)


def _get_scripted_phase_ansatz_cpu_executor():
    """Return one process-local lazy TorchScript cache, failing closed."""

    global _SCRIPTED_PHASE_ANSATZ_CPU_EXECUTOR
    global _SCRIPTED_PHASE_ANSATZ_CPU_FAILED
    global _SCRIPTED_PHASE_ANSATZ_CPU_PID

    process_id = os.getpid()
    if _SCRIPTED_PHASE_ANSATZ_CPU_PID != process_id:
        _reset_scripted_phase_ansatz_cpu_after_fork()
    if _SCRIPTED_PHASE_ANSATZ_CPU_EXECUTOR is not None:
        return _SCRIPTED_PHASE_ANSATZ_CPU_EXECUTOR
    if _SCRIPTED_PHASE_ANSATZ_CPU_FAILED:
        return None
    with _SCRIPTED_PHASE_ANSATZ_CPU_LOCK:
        if _SCRIPTED_PHASE_ANSATZ_CPU_EXECUTOR is not None:
            return _SCRIPTED_PHASE_ANSATZ_CPU_EXECUTOR
        if _SCRIPTED_PHASE_ANSATZ_CPU_FAILED:
            return None
        try:
            executor = _load_scripted_phase_ansatz_cpu_executor()
        except Exception:
            _SCRIPTED_PHASE_ANSATZ_CPU_FAILED = True
            return None
        _SCRIPTED_PHASE_ANSATZ_CPU_EXECUTOR = executor
        _SCRIPTED_PHASE_ANSATZ_CPU_PID = process_id
        return executor


def _clear_scripted_phase_ansatz_cpu_cache() -> None:
    """Release compiled functions and a remembered failure for this process."""

    global _SCRIPTED_PHASE_ANSATZ_CPU_EXECUTOR
    global _SCRIPTED_PHASE_ANSATZ_CPU_FAILED
    global _SCRIPTED_PHASE_ANSATZ_CPU_PID

    with _SCRIPTED_PHASE_ANSATZ_CPU_LOCK:
        _SCRIPTED_PHASE_ANSATZ_CPU_EXECUTOR = None
        _SCRIPTED_PHASE_ANSATZ_CPU_FAILED = False
        _SCRIPTED_PHASE_ANSATZ_CPU_PID = os.getpid()


def _mark_scripted_phase_ansatz_cpu_failed() -> None:
    """Disable this optional process-local lane after an execution failure."""

    global _SCRIPTED_PHASE_ANSATZ_CPU_EXECUTOR
    global _SCRIPTED_PHASE_ANSATZ_CPU_FAILED

    with _SCRIPTED_PHASE_ANSATZ_CPU_LOCK:
        _SCRIPTED_PHASE_ANSATZ_CPU_EXECUTOR = None
        _SCRIPTED_PHASE_ANSATZ_CPU_FAILED = True


def _evaluate_scripted_phase_ansatz_triplet(
    frequency,
    plan,
    *,
    _request_proof=None,
):
    """Return three compiled region values for one prequalified request."""

    request_qualified = _request_phase_plan_qualified(
        plan,
        _request_proof,
    )
    if not _scripted_phase_ansatz_cpu_frequency_supported(frequency):
        return None
    if request_qualified and IMRPhenomX_utils._tree_has_autograd_untrusted(
        frequency
    ):
        # A request marker proves only the immutable phase-plan schema. Keep
        # the public request's dynamic frequency strictly graph-free even
        # though the ordinary scripted evaluator supports reverse AD.
        return None
    if not request_qualified and not (
        _scripted_phase_ansatz_cpu_enabled()
        and _scripted_phase_ansatz_cpu_runtime_supported()
        and type(plan.inspiral) is _ScriptedInspiralPhasePlan
        and type(plan.intermediate) is _ScriptedIntermediatePhasePlan
        and type(plan.mergerringdown) is _ScriptedMergerRingdownPhasePlan
    ):
        return None
    executor = _get_scripted_phase_ansatz_cpu_executor()
    if executor is None:
        return None
    try:
        return executor(
            frequency,
            frequency,
            frequency,
            *plan.inspiral,
            _INSPIRAL_PHASE_NORMALIZATION,
            *plan.intermediate,
            *plan.mergerringdown,
        )
    except Exception:
        _mark_scripted_phase_ansatz_cpu_failed()
        return None


def _evaluate_inspiral_phase(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    plan: _InspiralPhasePlan,
) -> Float[Array, " n_freq"] | FloatLike:
    """Evaluate an already-prepared inspiral phase."""

    (
        phi0,
        phi1,
        phi2,
        phi3,
        phi4,
        phi5,
        phi5L,
        phi6,
        phi6L,
        phi7,
        phi8,
        phi8L,
        sigma1,
        sigma2,
        sigma3,
        sigma4,
    ) = plan

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


def get_inspiral_phase(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    *,
    _phase_plan: _IMRPhenomXASPhasePlan | None = None,
) -> Float[Array, " n_freq"] | FloatLike:
    """Calculate the inspiral phase for the IMRPhenomD waveform."""

    if _phase_plan is not None:
        return _evaluate_inspiral_phase(fM_s, _phase_plan.inspiral)
    return _evaluate_inspiral_phase(
        fM_s,
        _prepare_inspiral_phase(theta, phase_coeffs),
    )


def _prepare_intermediate_phase(
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    dPhaseIN: FloatLike,
    dPhaseRD: FloatLike,
    cL: FloatLike,
    chip: FloatLike = 0.0,
    *,
    fit_rows: torch.Tensor | None = None,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
    bulk_collocation: bool = False,
    _solve_info: list[torch.Tensor] | None = None,
    _cutoff_fMs=None,
    _intrinsic_controls=None,
) -> _IntermediatePhasePlan:
    """Prepare the frequency-independent intermediate phase coefficients."""

    if _intrinsic_controls is None:
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
    else:
        m1 = _intrinsic_controls.mass1
        m2 = _intrinsic_controls.mass2
        chi1 = _intrinsic_controls.spin1
        chi2 = _intrinsic_controls.spin2
        eta = _intrinsic_controls.eta
        StotR = _intrinsic_controls.merger_spin
        chia = _intrinsic_controls.spin_difference

    if _cutoff_fMs is None:
        fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO = _get_cutoff_fMs(
            m1,
            m2,
            chi1,
            chi2,
            chip=chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
        )
    else:
        fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO = _cutoff_fMs

    gpoints5 = torch.as_tensor(
        [
            0.0,
            0.5 - 0.5 / math.sqrt(2.0),
            0.5,
            0.5 + 0.5 / math.sqrt(2.0),
            1.0,
        ],
        dtype=theta.dtype,
        device=theta.device,
    )

    fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
    fMs_INmatch = fMs_MECO
    deltafMs = (fMs_IMmatch - fMs_INmatch) * 0.03
    fMs_PhaseMatchIN = fMs_INmatch - 1.0 * deltafMs
    fPhaseMatchIM = fMs_IMmatch + 0.5 * deltafMs

    deltax = fPhaseMatchIM - fMs_PhaseMatchIN
    xmin = fMs_PhaseMatchIN

    if bulk_collocation:
        CP_phase_Int = gpoints5 * deltax + xmin
    else:
        CP_phase_Int0 = gpoints5[0] * deltax + xmin
        CP_phase_Int1 = gpoints5[1] * deltax + xmin
        CP_phase_Int2 = gpoints5[2] * deltax + xmin
        CP_phase_Int3 = gpoints5[3] * deltax + xmin
        CP_phase_Int4 = gpoints5[4] * deltax + xmin

    CV_phase_Int0 = dPhaseIN
    CV_phase_Int4 = dPhaseRD

    if fit_rows is None:
        # These fits differ from WF4py and drive its intermediate differences.
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
    else:
        v2IMmRDv4 = fit_rows[4]
        v3IMmRDv4 = fit_rows[5]
        v2IM = fit_rows[6]
        d43 = fit_rows[7]
        CV_phase_RD3 = fit_rows[11]

    CV_phase_Int1 = 0.75 * (v2IMmRDv4 + CV_phase_RD3) + 0.25 * v2IM
    CV_phase_Int2 = v3IMmRDv4 + CV_phase_RD3
    CV_phase_Int3 = d43 + CV_phase_Int2

    if bulk_collocation:
        ratio = fMs_RD / CP_phase_Int
        A = torch.stack(
            (
                torch.ones_like(CP_phase_Int),
                ratio,
                ratio * ratio,
                ratio**3,
                ratio**4,
            ),
            dim=1,
        )
        collocation_values = torch.stack(
            (
                CV_phase_Int0,
                CV_phase_Int1,
                CV_phase_Int2,
                CV_phase_Int3,
                CV_phase_Int4,
            )
        )
        offset = CP_phase_Int - fMs_RD
        b = collocation_values - (4.0 * cL) / (
            (4.0 * fMs_damp * fMs_damp) + offset * offset
        )
    else:
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

    b = b.to(device=A.device, dtype=A.dtype)
    if _solve_info is None:
        coeffs_Int = jnp.linalg.solve(A, b)
    else:
        coeffs_Int, info = torch.linalg.solve_ex(
            A,
            b,
            check_errors=False,
        )
        _solve_info.append(info)

    b0 = coeffs_Int[0]
    b1 = coeffs_Int[1] * fMs_RD
    b2 = coeffs_Int[2] * fMs_RD**2
    b3 = coeffs_Int[3] * fMs_RD**3
    b4 = coeffs_Int[4] * fMs_RD**4
    b1 = b1 + _coprecessing_deviation(coprecessing_deviations, "zeta2", like=b1)
    b4 = b4 + _coprecessing_deviation(coprecessing_deviations, "zeta1", like=b4)

    return _IntermediatePhasePlan(
        b0,
        b1,
        b2,
        b3,
        b4,
        cL,
        fMs_RD,
        fMs_damp,
    )


def _evaluate_intermediate_phase(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    plan: _IntermediatePhasePlan,
) -> Float[Array, " n_freq"] | FloatLike:
    """Evaluate an already-prepared intermediate phase."""

    b0, b1, b2, b3, b4, cL, fMs_RD, fMs_damp = plan

    return (
        b0 * fM_s
        + b1 * jnp.log(fM_s)
        - b2 * (fM_s**-1.0)
        - b3 * (fM_s**-2.0) / 2.0
        - (b4 * (fM_s**-3.0) / 3.0)
        + (2.0 * cL * jnp.arctan((fM_s - fMs_RD) / (2.0 * fMs_damp))) / fMs_damp
    )


def get_intermediate_raw_phase(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    dPhaseIN: FloatLike,
    dPhaseRD: FloatLike,
    cL: FloatLike,
    chip: FloatLike = 0.0,
    *,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
) -> Float[Array, " n_freq"] | FloatLike:
    """Calculate the raw intermediate phase."""

    return _evaluate_intermediate_phase(
        fM_s,
        _prepare_intermediate_phase(
            theta,
            phase_coeffs,
            dPhaseIN,
            dPhaseRD,
            cL,
            chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
        ),
    )


def _prepare_mergerringdown_phase(
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    chip: FloatLike = 0.0,
    *,
    fit_rows: torch.Tensor | None = None,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
    bulk_collocation: bool = False,
    _solve_info: list[torch.Tensor] | None = None,
    _cutoff_fMs=None,
    _intrinsic_controls=None,
) -> _MergerRingdownPhasePlan:
    """Prepare the frequency-independent merger-ringdown coefficients."""

    if _intrinsic_controls is None:
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
    else:
        m1 = _intrinsic_controls.mass1
        m2 = _intrinsic_controls.mass2
        chi1 = _intrinsic_controls.spin1
        chi2 = _intrinsic_controls.spin2
        eta = _intrinsic_controls.eta
        chia = _intrinsic_controls.spin_difference
        StotR = _intrinsic_controls.merger_spin

    if _cutoff_fMs is None:
        fMs_RD, fMs_damp, _, fMs_ISCO = _get_cutoff_fMs(
            m1,
            m2,
            chi1,
            chi2,
            chip=chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
        )
    else:
        fMs_RD, fMs_damp, _, fMs_ISCO = _cutoff_fMs
    fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
    fMs_PhaseRDMin = fMs_IMmatch
    fMs_PhaseRDMax = fMs_RD + 1.25 * fMs_damp
    dphase0 = 5.0 / (128.0 * (PI ** (5.0 / 3.0)))

    gpoints5 = torch.as_tensor(
        [
            0.0,
            0.5 - 0.5 / math.sqrt(2.0),
            0.5,
            0.5 + 0.5 / math.sqrt(2.0),
            1.0,
        ],
        dtype=theta.dtype,
        device=theta.device,
    )

    # Ringdown phase collocation points:
    # Default is to use 5 pseudo-PN coefficients and hence 5 collocation points.
    deltax = fMs_PhaseRDMax - fMs_PhaseRDMin
    xmin = fMs_PhaseRDMin

    if bulk_collocation:
        CP_phase_RD = gpoints5 * deltax + xmin
        CP_phase_RD = torch.cat(
            (
                CP_phase_RD[:3],
                fMs_RD.reshape(1),
                CP_phase_RD[4:],
            )
        )
    else:
        CP_phase_RD0 = gpoints5[0] * deltax + xmin
        CP_phase_RD1 = gpoints5[1] * deltax + xmin
        CP_phase_RD2 = gpoints5[2] * deltax + xmin
        CP_phase_RD3 = jnp.asarray(fMs_RD)
        CP_phase_RD4 = gpoints5[4] * deltax + xmin

    if fit_rows is None:
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
    else:
        (
            CV_phase_RD0,
            CV_phase_RD1,
            CV_phase_RD2,
            CV_phase_RD3,
            CV_phase_RD4,
        ) = fit_rows[8:13]

    CV_phase_RD4 = CV_phase_RD4 + CV_phase_RD3
    CV_phase_RD2 = CV_phase_RD2 + CV_phase_RD3
    CV_phase_RD1 = CV_phase_RD1 + CV_phase_RD3
    CV_phase_RD0 = CV_phase_RD0 + CV_phase_RD1

    if bulk_collocation:
        offset = CP_phase_RD - fMs_RD
        A = torch.stack(
            (
                torch.ones_like(CP_phase_RD),
                CP_phase_RD ** (-1.0 / 3.0),
                CP_phase_RD ** (-2),
                CP_phase_RD ** (-4),
                -(dphase0)
                / (fMs_damp * fMs_damp + offset * offset),
            ),
            dim=1,
        )
    else:
        A0 = jnp.array(
            [
                jnp.ones(CP_phase_RD0.shape),
                CP_phase_RD0 ** (-1.0 / 3.0),
                CP_phase_RD0 ** (-2),
                CP_phase_RD0 ** (-4),
                -(dphase0)
                / (
                    fMs_damp * fMs_damp
                    + (CP_phase_RD0 - fMs_RD) * (CP_phase_RD0 - fMs_RD)
                ),
            ]
        )
        A1 = jnp.array(
            [
                jnp.ones(CP_phase_RD1.shape),
                CP_phase_RD1 ** (-1.0 / 3.0),
                CP_phase_RD1 ** (-2),
                CP_phase_RD1 ** (-4),
                -(dphase0)
                / (
                    fMs_damp * fMs_damp
                    + (CP_phase_RD1 - fMs_RD) * (CP_phase_RD1 - fMs_RD)
                ),
            ]
        )
        A2 = jnp.array(
            [
                jnp.ones(CP_phase_RD2.shape),
                CP_phase_RD2 ** (-1.0 / 3.0),
                CP_phase_RD2 ** (-2),
                CP_phase_RD2 ** (-4),
                -(dphase0)
                / (
                    fMs_damp * fMs_damp
                    + (CP_phase_RD2 - fMs_RD) * (CP_phase_RD2 - fMs_RD)
                ),
            ]
        )
        A3 = jnp.array(
            [
                jnp.ones(CP_phase_RD3.shape),
                CP_phase_RD3 ** (-1.0 / 3.0),
                CP_phase_RD3 ** (-2),
                CP_phase_RD3 ** (-4),
                -(dphase0)
                / (
                    fMs_damp * fMs_damp
                    + (CP_phase_RD3 - fMs_RD) * (CP_phase_RD3 - fMs_RD)
                ),
            ]
        )
        A4 = jnp.array(
            [
                jnp.ones(CP_phase_RD4.shape),
                CP_phase_RD4 ** (-1.0 / 3.0),
                CP_phase_RD4 ** (-2),
                CP_phase_RD4 ** (-4),
                -(dphase0)
                / (
                    fMs_damp * fMs_damp
                    + (CP_phase_RD4 - fMs_RD) * (CP_phase_RD4 - fMs_RD)
                ),
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
    b = b.to(device=A.device, dtype=A.dtype)

    if _solve_info is None:
        coeffs_RD = jnp.linalg.solve(A, b)
    else:
        coeffs_RD, info = torch.linalg.solve_ex(
            A,
            b,
            check_errors=False,
        )
        _solve_info.append(info)
    c0, c1, c2, c4, cRD = coeffs_RD
    cL = -(dphase0 * cRD)
    cL = cL + _coprecessing_deviation(coprecessing_deviations, "nu4", like=cL)
    c4ov3 = c4 / 3.0
    cLovfda = cL / fMs_damp

    return _MergerRingdownPhasePlan(
        c0,
        c1,
        c2,
        c4ov3,
        cLovfda,
        fMs_RD,
        fMs_damp,
        cL,
        CV_phase_RD0,
    )


def _evaluate_mergerringdown_phase(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    plan: _MergerRingdownPhasePlan,
) -> tuple[Float[Array, " n_freq"] | FloatLike, tuple[FloatLike, FloatLike]]:
    """Evaluate an already-prepared merger-ringdown phase."""

    c0, c1, c2, c4ov3, cLovfda, fMs_RD, fMs_damp, cL, CV_phase_RD0 = plan

    phiRD = (
        c0 * fM_s
        + 1.5 * c1 * (fM_s ** (2.0 / 3.0))
        - c2 * (fM_s**-1.0)
        - c4ov3 * (fM_s**-3.0)
        + (cLovfda * jnp.arctan((fM_s - fMs_RD) / fMs_damp))
    )

    return phiRD, (cL, CV_phase_RD0)


def _exact_inspiral_phase_value_and_derivative(
    frequency,
    plan,
    *,
    output_adjoint=None,
    initial_gradient=None,
):
    """Evaluate the inspiral phase and Torch's exact local reverse pass.

    The arithmetic order mirrors the eager expression and its reverse-mode
    graph. It is intentionally not algebraically simplified: regrouping these
    operations changes float64 low bits at higher-mode matching seams.
    """

    (
        phi0,
        phi1,
        phi2,
        phi3,
        phi4,
        phi5,
        phi5_l,
        phi6,
        phi6_l,
        phi7,
        phi8,
        phi8_l,
        sigma1,
        sigma2,
        sigma3,
        sigma4,
    ) = plan

    f13 = frequency ** (1.0 / 3.0)
    f23 = f13 * f13
    f43 = frequency * f13
    f53 = frequency * f23
    f2 = frequency * frequency
    f73 = f2 * f13
    f83 = f2 * f23
    f3 = f2 * frequency
    f103 = f3 * f13
    f113 = f3 * f23
    log_f = torch.log(frequency)

    term1 = phi1 * f13
    phase_tf2 = phi0 + term1
    term2 = phi2 * f23
    phase_tf2 = phase_tf2 + term2
    term3 = phi3 * frequency
    phase_tf2 = phase_tf2 + term3
    term4 = phi4 * f43
    phase_tf2 = phase_tf2 + term4
    term5 = phi5 * f53
    phase_tf2 = phase_tf2 + term5
    term5_l_pre = phi5_l * f53
    term5_l = term5_l_pre * log_f
    phase_tf2 = phase_tf2 + term5_l
    term6 = phi6 * f2
    phase_tf2 = phase_tf2 + term6
    term6_l_pre = phi6_l * f2
    term6_l = term6_l_pre * log_f
    phase_tf2 = phase_tf2 + term6_l
    term7 = phi7 * f73
    phase_tf2 = phase_tf2 + term7
    term8 = phi8 * f83
    phase_tf2 = phase_tf2 + term8
    term8_l_pre = phi8_l * f83
    term8_l = term8_l_pre * log_f
    phase_tf2 = phase_tf2 + term8_l

    sigma_term1 = sigma1 * f83
    sigma_term2 = sigma2 * f3
    sigma_phase = sigma_term1 + sigma_term2
    sigma_term3 = sigma3 * f103
    sigma_phase = sigma_phase + sigma_term3
    sigma_term4 = sigma4 * f113
    sigma_phase = sigma_phase + sigma_term4
    phase_inspiral = phase_tf2 + sigma_phase
    phase_normalization = -(3.0 * PI ** (-5.0 / 3.0)) / 128.0
    scaled_phase = phase_inspiral * phase_normalization
    value = scaled_phase / f53

    one = torch.ones_like(value) if output_adjoint is None else output_adjoint
    gradient_scaled_phase = one / f53
    gradient_f53 = -one * ((scaled_phase / f53) / f53)
    gradient_phase = gradient_scaled_phase * phase_normalization

    gradient_f113 = gradient_phase * sigma4
    gradient_f103 = gradient_phase * sigma3
    gradient_f3 = gradient_phase * sigma2
    gradient_f83 = gradient_phase * sigma1

    gradient_pre = gradient_phase * log_f
    gradient_log = gradient_phase * term8_l_pre
    gradient_f83 = gradient_f83 + gradient_pre * phi8_l
    gradient_f83 = gradient_f83 + gradient_phase * phi8
    gradient_f73 = gradient_phase * phi7
    gradient_pre = gradient_phase * log_f
    gradient_log = gradient_log + gradient_phase * term6_l_pre
    gradient_f2 = gradient_pre * phi6_l
    gradient_f2 = gradient_f2 + gradient_phase * phi6
    gradient_pre = gradient_phase * log_f
    gradient_log = gradient_log + gradient_phase * term5_l_pre
    gradient_f53 = gradient_f53 + gradient_pre * phi5_l
    gradient_f53 = gradient_f53 + gradient_phase * phi5
    gradient_f43 = gradient_phase * phi4
    direct_gradient = gradient_phase * phi3
    gradient_frequency = (
        direct_gradient
        if initial_gradient is None
        else initial_gradient + direct_gradient
    )
    gradient_f23 = gradient_phase * phi2
    gradient_f13 = gradient_phase * phi1

    gradient_frequency = gradient_frequency + gradient_log / frequency
    gradient_f3 = gradient_f3 + gradient_f113 * f23
    gradient_f23 = gradient_f23 + gradient_f113 * f3
    gradient_f3 = gradient_f3 + gradient_f103 * f13
    gradient_f13 = gradient_f13 + gradient_f103 * f3
    gradient_f2 = gradient_f2 + gradient_f3 * frequency
    gradient_frequency = gradient_frequency + gradient_f3 * f2
    gradient_f2 = gradient_f2 + gradient_f83 * f23
    gradient_f23 = gradient_f23 + gradient_f83 * f2
    gradient_f2 = gradient_f2 + gradient_f73 * f13
    gradient_f13 = gradient_f13 + gradient_f73 * f2
    gradient_frequency = gradient_frequency + gradient_f2 * frequency
    gradient_frequency = gradient_frequency + gradient_f2 * frequency
    gradient_frequency = gradient_frequency + gradient_f53 * f23
    gradient_f23 = gradient_f23 + gradient_f53 * frequency
    gradient_frequency = gradient_frequency + gradient_f43 * f13
    gradient_f13 = gradient_f13 + gradient_f43 * frequency
    gradient_f13 = gradient_f13 + gradient_f23 * f13
    gradient_f13 = gradient_f13 + gradient_f23 * f13
    gradient_frequency = gradient_frequency + gradient_f13 * (
        (1.0 / 3.0) * frequency.pow((1.0 / 3.0) - 1.0)
    )
    return value, gradient_frequency


def _maybe_python_inspiral_phase_value_and_derivative(
    frequency,
    plan,
    *,
    output_adjoint=None,
    initial_gradient=None,
    exact_inputs_prequalified=False,
    _request_proof=None,
):
    """Return the qualified binary64 lane result, or select eager fallback."""

    if (
        not _request_inspiral_phase_plan_supported(plan, _request_proof)
        and not _python_inspiral_derivative_enabled()
    ):
        return None
    arguments = _python_inspiral_derivative_executor_arguments(
        frequency,
        plan,
        output_adjoint,
        initial_gradient,
        exact_inputs_prequalified=exact_inputs_prequalified,
        _request_proof=_request_proof,
    )
    if arguments is None:
        return None
    executor = _get_python_inspiral_derivative_executor()
    if executor is None:
        return None
    python_result = _execute_python_inspiral_derivative(
        executor,
        arguments,
        _request_proof=_request_proof,
    )
    if python_result is None:
        return None
    if _PYTHON_INSPIRAL_DERIVATIVE_CALIBRATED:
        return python_result

    eager_result = _exact_inspiral_phase_value_and_derivative(
        frequency,
        plan,
        output_adjoint=output_adjoint,
        initial_gradient=initial_gradient,
    )
    if _python_inspiral_derivative_raw_equal(python_result, eager_result):
        _mark_python_inspiral_derivative_calibrated()
    else:
        _mark_python_inspiral_derivative_failed()
    return eager_result


def _exact_intermediate_phase_derivative(
    frequency,
    plan,
    *,
    initial_gradient=None,
):
    """Return Torch's exact reverse derivative for the intermediate ansatz."""

    b0, b1, b2, b3, b4, c_l, f_rd, f_damp = plan
    one = torch.ones_like(frequency)
    scaled = (frequency - f_rd) / (2.0 * f_damp)
    contributions = (
        b0,
        b1 / frequency,
        ((-one) * b2) * ((-1.0) * frequency.pow(-2.0)),
        (((-one) / 2.0) * b3) * ((-2.0) * frequency.pow(-3.0)),
        (((-one) / 3.0) * b4) * ((-3.0) * frequency.pow(-4.0)),
        (((one / f_damp) * (2.0 * c_l)) / (1.0 + scaled * scaled)) / (2.0 * f_damp),
    )
    if initial_gradient is None:
        derivative = contributions[5] + contributions[4]
        remaining = (3, 2, 1, 0)
    else:
        derivative = initial_gradient + contributions[5]
        remaining = (4, 3, 2, 1, 0)
    for index in remaining:
        derivative = derivative + contributions[index]
    return derivative


def _exact_mergerringdown_phase_derivative(frequency, plan):
    """Return Torch's exact reverse derivative for the ringdown ansatz."""

    c0, c1, c2, c4_over_3, c_l_over_fd, f_rd, f_damp, _, _ = plan
    one = torch.ones_like(frequency)
    scaled = (frequency - f_rd) / f_damp
    contributions = (
        c0,
        (one * (1.5 * c1)) * ((2.0 / 3.0) * frequency.pow((2.0 / 3.0) - 1.0)),
        ((-one) * c2) * ((-1.0) * frequency.pow(-2.0)),
        ((-one) * c4_over_3) * ((-3.0) * frequency.pow(-4.0)),
        ((one * c_l_over_fd) / (1.0 + scaled * scaled)) / f_damp,
    )
    derivative = contributions[4] + contributions[3]
    for index in (2, 1, 0):
        derivative = derivative + contributions[index]
    return derivative


def _maybe_exact_inspiral_phase_value_and_derivative(
    frequency,
    plan,
    *,
    output_adjoint=None,
    initial_gradient=None,
    _request_proof=None,
):
    """Return an exact local reverse pass, or select caller fallback."""

    if not _exact_scalar_derivative_supported(
        frequency,
        plan,
        output_adjoint,
        initial_gradient,
        _request_proof=_request_proof,
    ):
        return None
    python_result = _maybe_python_inspiral_phase_value_and_derivative(
        frequency,
        plan,
        output_adjoint=output_adjoint,
        initial_gradient=initial_gradient,
        exact_inputs_prequalified=True,
        _request_proof=_request_proof,
    )
    if python_result is not None:
        return python_result
    return _exact_inspiral_phase_value_and_derivative(
        frequency,
        plan,
        output_adjoint=output_adjoint,
        initial_gradient=initial_gradient,
    )


def _inspiral_phase_value_and_derivative(
    frequency,
    plan,
    *,
    _request_proof=None,
):
    """Use the guarded exact reverse pass or the legacy autograd seam."""

    result = _maybe_exact_inspiral_phase_value_and_derivative(
        frequency,
        plan,
        _request_proof=_request_proof,
    )
    if result is not None:
        return result
    if _exact_scalar_derivative_supported(
        frequency,
        plan,
        _request_proof=_request_proof,
    ):
        return _exact_inspiral_phase_value_and_derivative(frequency, plan)
    return jax.value_and_grad(_evaluate_inspiral_phase)(frequency, plan)


def _intermediate_phase_value_and_derivative(
    frequency,
    plan,
    *,
    correction=None,
    _request_proof=None,
):
    """Evaluate one intermediate seam with an exact guarded derivative."""

    if correction is None:
        alpha0 = alpha1 = None

        def evaluate(value, intermediate_plan):
            return _evaluate_intermediate_phase(value, intermediate_plan)

    else:
        alpha0, alpha1 = correction

        def evaluate(value, intermediate_plan):
            return (
                _evaluate_intermediate_phase(value, intermediate_plan)
                + alpha1 * value
                + alpha0
            )

    if _exact_scalar_derivative_supported(
        frequency,
        plan,
        alpha0,
        alpha1,
        _request_proof=_request_proof,
    ):
        value = evaluate(frequency, plan)
        derivative = _exact_intermediate_phase_derivative(
            frequency,
            plan,
            initial_gradient=alpha1,
        )
        return value, derivative
    return jax.value_and_grad(evaluate)(frequency, plan)


def _mergerringdown_phase_value_and_derivative(
    frequency,
    plan,
    *,
    _request_proof=None,
):
    """Evaluate one ringdown seam with an exact guarded derivative."""

    if _exact_scalar_derivative_supported(
        frequency,
        plan,
        _request_proof=_request_proof,
    ):
        value = _evaluate_mergerringdown_phase(frequency, plan)
        derivative = _exact_mergerringdown_phase_derivative(frequency, plan)
        return value, derivative
    return jax.value_and_grad(
        _evaluate_mergerringdown_phase,
        has_aux=True,
    )(frequency, plan)


def _inspiral_phase_derivative(frequency, plan, *, _request_proof=None):
    result = _maybe_exact_inspiral_phase_value_and_derivative(
        frequency,
        plan,
        _request_proof=_request_proof,
    )
    if result is not None:
        return result[1]
    if _exact_scalar_derivative_supported(
        frequency,
        plan,
        _request_proof=_request_proof,
    ):
        return _exact_inspiral_phase_value_and_derivative(frequency, plan)[1]
    return jax.grad(_evaluate_inspiral_phase)(frequency, plan)


def _intermediate_phase_derivative(
    frequency,
    plan,
    *,
    correction=None,
    _request_proof=None,
):
    alpha0, alpha1 = (None, None) if correction is None else correction
    if _exact_scalar_derivative_supported(
        frequency,
        plan,
        alpha0,
        alpha1,
        _request_proof=_request_proof,
    ):
        return _exact_intermediate_phase_derivative(
            frequency,
            plan,
            initial_gradient=alpha1,
        )

    def evaluate(value, intermediate_plan):
        phase = _evaluate_intermediate_phase(value, intermediate_plan)
        if correction is not None:
            phase = phase + alpha1 * value + alpha0
        return phase

    return jax.grad(evaluate)(frequency, plan)


def _mergerringdown_phase_derivative(
    frequency,
    plan,
    *,
    _request_proof=None,
):
    if _exact_scalar_derivative_supported(
        frequency,
        plan,
        _request_proof=_request_proof,
    ):
        return _exact_mergerringdown_phase_derivative(frequency, plan)
    return jax.grad(
        lambda value, phase_plan: _evaluate_mergerringdown_phase(
            value,
            phase_plan,
        )[0]
    )(frequency, plan)


def get_mergerringdown_raw_phase(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    chip: FloatLike = 0.0,
    *,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
) -> tuple[Float[Array, " n_freq"] | FloatLike, tuple[FloatLike, FloatLike]]:
    """Calculate the raw merger-ringdown phase."""

    return _evaluate_mergerringdown_phase(
        fM_s,
        _prepare_mergerringdown_phase(
            theta,
            phase_coeffs,
            chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
        ),
    )


def _prepare_phase_plan_eager(
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    chip: FloatLike = 0.0,
    *,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
    _phase_fit_rows: torch.Tensor | None = None,
    _solve_info: list[torch.Tensor] | None = None,
    _reuse_cutoff_fMs: bool = False,
    _cutoff_fMs=None,
    _intrinsic_controls=None,
    _inspiral_phase_host_scalars=None,
    _request_proof=None,
) -> _IMRPhenomXASPhasePlan:
    """Prepare the exact piecewise phase once for one waveform request."""

    bulk_collocation = _phase_plan_bulk_collocation_supported(
        theta,
        phase_coeffs,
        chip,
        final_spin,
        coprecessing_deviations,
    )
    scalar_region_dispatch = (
        _scalar_region_dispatch_enabled()
        and not IMRPhenomX_utils._tree_has_autograd(
            (
                theta,
                phase_coeffs,
                chip,
                final_spin,
                coprecessing_deviations,
                _phase_fit_rows,
                _cutoff_fMs,
            )
        )
    )
    if _intrinsic_controls is None:
        m1, m2, chi1, chi2 = theta
        m1_s = m1 * MTSUN
        m2_s = m2 * MTSUN
        M_s = m1_s + m2_s
        eta = m1_s * m2_s / (M_s**2.0)
    else:
        m1 = _intrinsic_controls.mass1
        m2 = _intrinsic_controls.mass2
        chi1 = _intrinsic_controls.spin1
        chi2 = _intrinsic_controls.spin2
        M_s = _intrinsic_controls.total_mass_seconds
        eta = _intrinsic_controls.eta

    if _cutoff_fMs is None:
        fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO = _get_cutoff_fMs(
            m1,
            m2,
            chi1,
            chi2,
            chip=chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
        )
    else:
        fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO = _cutoff_fMs
    fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
    fMs_INmatch = fMs_MECO
    deltafMs = (fMs_IMmatch - fMs_INmatch) * 0.03
    f1_Ms = fMs_INmatch - 1.0 * deltafMs
    f2_Ms = fMs_IMmatch + 0.5 * deltafMs
    cutoff_fMs = (
        (fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO)
        if _reuse_cutoff_fMs or _cutoff_fMs is not None
        else None
    )

    fit_rows = (
        _phase_fit_rows
        if _precomputed_phase_fit_rows_supported(_phase_fit_rows, theta)
        else None
    )
    if fit_rows is None and not IMRPhenomX_utils._tree_has_autograd(
        (theta, phase_coeffs)
    ):
        fit_rows = _prepare_phase_fit_rows(
            theta,
            phase_coeffs,
            _intrinsic_controls=_intrinsic_controls,
        )
    if (
        _inspiral_phase_host_scalars is None
        and _phase_fit_rows is None
        and _cutoff_fMs is not None
    ):
        # XPHM supplies a canonical request-local cutoff but prepares its
        # canonical fit rows here.  Qualify only after that exact eager work
        # exists, matching the established inspiral-builder call seam.
        _inspiral_phase_host_scalars = (
            _maybe_inspiral_phase_host_scalars(
                theta,
                phase_coeffs,
                chip,
                final_spin,
                coprecessing_deviations,
                fit_rows,
                cutoff_fMs,
                _intrinsic_controls,
            )
        )
        if _inspiral_phase_host_scalars is not None:
            _intrinsic_controls = (
                _inspiral_phase_host_scalars.intrinsic_controls
            )
    inspiral_plan = _prepare_inspiral_phase(
        theta,
        phase_coeffs,
        fit_rows,
        bulk_collocation=bulk_collocation,
        _solve_info=_solve_info,
        _cutoff_fMs=cutoff_fMs,
        _intrinsic_controls=_intrinsic_controls,
        _host_scalars=_inspiral_phase_host_scalars,
    )
    mergerringdown_plan = _prepare_mergerringdown_phase(
        theta,
        phase_coeffs,
        chip,
        fit_rows=fit_rows,
        final_spin=final_spin,
        coprecessing_deviations=coprecessing_deviations,
        bulk_collocation=bulk_collocation,
        _solve_info=_solve_info,
        _cutoff_fMs=cutoff_fMs,
        _intrinsic_controls=_intrinsic_controls,
    )

    prequalify_scalar_derivative_plans = (
        _scalar_derivative_plan_cse_enabled()
        and _exact_scalar_derivatives_enabled()
    )
    scalar_inspiral_plan = _maybe_prequalify_scalar_derivative_plan(
        _detach_phase_plan(inspiral_plan),
        f1_Ms,
        enabled=prequalify_scalar_derivative_plans,
        _request_proof=_request_proof,
    )
    phi_Ins_match_f1, dphi_Ins_match_f1 = _inspiral_phase_value_and_derivative(
        f1_Ms,
        scalar_inspiral_plan,
        _request_proof=_request_proof,
    )
    scalar_mergerringdown_plan = _maybe_prequalify_scalar_derivative_plan(
        _detach_phase_plan(mergerringdown_plan),
        f2_Ms,
        enabled=prequalify_scalar_derivative_plans,
        _request_proof=_request_proof,
    )
    (_phi_MRD_match_f2, (cL, CV_phase_RD0)), dphi_MRD_match_f2 = (
        _mergerringdown_phase_value_and_derivative(
            f2_Ms,
            scalar_mergerringdown_plan,
            _request_proof=_request_proof,
        )
    )
    # Preserve the graph-bearing value used by the existing phase assembly.
    phi_MRD_match_f2, _ = _evaluate_mergerringdown_phase(
        f2_Ms,
        mergerringdown_plan,
    )

    intermediate_plan = _prepare_intermediate_phase(
        theta,
        phase_coeffs,
        dphi_Ins_match_f1,
        CV_phase_RD0,
        cL,
        chip,
        fit_rows=fit_rows,
        final_spin=final_spin,
        coprecessing_deviations=coprecessing_deviations,
        bulk_collocation=bulk_collocation,
        _solve_info=_solve_info,
        _cutoff_fMs=cutoff_fMs,
        _intrinsic_controls=_intrinsic_controls,
    )
    scalar_intermediate_plan = _maybe_prequalify_scalar_derivative_plan(
        _detach_phase_plan(intermediate_plan),
        f1_Ms,
        enabled=prequalify_scalar_derivative_plans,
        _request_proof=_request_proof,
    )
    phi_Int_match_f1, dphi_Int_match_f1 = _intermediate_phase_value_and_derivative(
        f1_Ms,
        scalar_intermediate_plan,
        _request_proof=_request_proof,
    )
    alpha1 = dphi_Ins_match_f1 - dphi_Int_match_f1
    alpha0 = phi_Ins_match_f1 - phi_Int_match_f1 - alpha1 * f1_Ms

    phi_Int_match_f2, dphi_Int_match_f2 = _intermediate_phase_value_and_derivative(
        f2_Ms,
        scalar_intermediate_plan,
        correction=(alpha0, alpha1),
        _request_proof=_request_proof,
    )
    beta1 = dphi_Int_match_f2 - dphi_MRD_match_f2
    beta0 = phi_Int_match_f2 - phi_MRD_match_f2 - beta1 * f2_Ms

    (
        inspiral_plan,
        intermediate_plan,
        mergerringdown_plan,
    ) = _maybe_prequalify_scripted_phase_ansatz_cpu_plans(
        inspiral_plan,
        intermediate_plan,
        mergerringdown_plan,
        theta,
        _request_proof=_request_proof,
    )
    (
        inspiral_plan,
        intermediate_plan,
        mergerringdown_plan,
    ) = _maybe_prequalify_cuda_graph_phase_ansatz_plans(
        inspiral_plan,
        intermediate_plan,
        mergerringdown_plan,
        theta,
    )

    plan = _IMRPhenomXASPhasePlan(
        total_mass_seconds=M_s,
        eta=eta,
        f1_Ms=f1_Ms,
        f2_Ms=f2_Ms,
        inspiral=inspiral_plan,
        intermediate=intermediate_plan,
        mergerringdown=mergerringdown_plan,
        scalar_inspiral=scalar_inspiral_plan,
        scalar_intermediate=scalar_intermediate_plan,
        scalar_mergerringdown=scalar_mergerringdown_plan,
        alpha0=alpha0,
        alpha1=alpha1,
        beta0=beta0,
        beta1=beta1,
        scalar_region_dispatch=scalar_region_dispatch,
    )
    return _request_qualify_top_plan(plan, _request_proof)


def _phase_plan_torchscript_trace_python_supported() -> bool:
    """Accept only qualified ordinary-GIL CPython interpreter families."""

    version = sys.version_info[:2]
    if (
        getattr(sys.implementation, "name", None) != "cpython"
        or not (
            _PHASE_PLAN_TORCHSCRIPT_TRACE_PYTHON_MIN
            <= version
            <= _PHASE_PLAN_TORCHSCRIPT_TRACE_PYTHON_MAX
        )
    ):
        return False

    gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if gil_enabled is None:
        # Official CPython 3.11/3.12 builds always have the GIL and do not
        # expose the runtime query introduced with free threading in 3.13.
        return version < (3, 13)
    if not callable(gil_enabled):
        return False
    try:
        return gil_enabled() is True
    except Exception:
        return False


def _phase_plan_torchscript_trace_runtime_supported(device) -> bool:
    """Accept supported CPU Torch under ordinary observable eager semantics."""

    version = torch.__version__.split("+", 1)[0].split(".")
    try:
        version = tuple(int(value) for value in version[:2])
    except (TypeError, ValueError):
        return False
    if (
        type(device) is not torch.device
        or device.type != "cpu"
        or len(version) != 2
        or not (
            _PHASE_PLAN_TORCHSCRIPT_TRACE_TORCH_MIN
            <= version
            <= _PHASE_PLAN_TORCHSCRIPT_TRACE_TORCH_MAX
        )
        or not _phase_plan_torchscript_trace_python_supported()
        or not callable(getattr(torch.jit, "trace", None))
        or not callable(getattr(torch.jit, "freeze", None))
        or not callable(getattr(torch._C, "_is_alias_of", None))
        or not _packed_heaviside_masks_runtime_supported()
    ):
        return False

    try:
        from torch.fx.experimental.proxy_tensor import make_fx

        if not callable(make_fx):
            return False
    except Exception:
        return False

    try:
        anomaly = getattr(torch, "is_anomaly_enabled", None)
        deterministic = getattr(
            torch,
            "are_deterministic_algorithms_enabled",
            None,
        )
        if (
            not torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
            or anomaly is None
            or anomaly()
            or deterministic is None
            or deterministic()
            or _phase_plan_torchscript_trace_thread_settings() is None
        ):
            return False
    except Exception:
        return False
    return True


def _phase_plan_torchscript_trace_thread_settings():
    """Return the exact intra/inter-op CPU topology or fail closed."""

    try:
        settings = (
            int(torch.get_num_threads()),
            int(torch.get_num_interop_threads()),
        )
    except Exception:
        return None
    return settings if all(value > 0 for value in settings) else None


def _phase_plan_torchscript_trace_tensor_supported(
    value,
    *,
    device,
    shape,
    owned=True,
) -> bool:
    """Accept one plain binary64 tensor without observable AD state."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.device == device
        and value.dtype is torch.float64
        and value.shape == shape
        and value.is_contiguous()
        and (
            not owned
            or (value.storage_offset() == 0 and value._base is None)
        )
        and _fit_tensor_version_is_zero(value)
        and not value.is_conj()
        and not value.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(value)
    )


def _phase_plan_torchscript_trace_supported(
    theta,
    phase_coeffs,
    chip,
    final_spin,
    coprecessing_deviations,
    phase_fit_rows,
    cutoff_fMs,
    intrinsic_controls,
    request_proof,
) -> bool:
    """Accept only the calibrated scalar, batch-one XAS tensor contract."""

    if (
        not _phase_plan_torchscript_trace_enabled()
        or type(theta) is not torch.Tensor
        or not _phase_plan_torchscript_trace_runtime_supported(theta.device)
        or not _phase_plan_torchscript_trace_tensor_supported(
            theta,
            device=theta.device,
            shape=torch.Size((4,)),
        )
        or not _phase_plan_torchscript_trace_tensor_supported(
            phase_coeffs,
            device=theta.device,
            shape=torch.Size((13, 49)),
        )
        or type(chip) is not float
        or not math.isfinite(chip)
        or type(final_spin) is not float
        or not math.isfinite(final_spin)
        or coprecessing_deviations is not None
        or intrinsic_controls is not None
        or request_proof is not None
        or _phase_plan_bulk_collocation_enabled()
        or type(cutoff_fMs) is not tuple
        or len(cutoff_fMs) != 4
        or not all(
            _phase_plan_torchscript_trace_tensor_supported(
                value,
                device=theta.device,
                shape=torch.Size(()),
                owned=False,
            )
            for value in cutoff_fMs
        )
    ):
        return False
    return phase_fit_rows is None or (
        _phase_plan_torchscript_trace_tensor_supported(
            phase_fit_rows,
            device=theta.device,
            shape=torch.Size((13,)),
        )
    )


def _phase_plan_torchscript_trace_environment():
    """Return switches which can affect the captured eager expression tree."""

    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("PYCBC_")
        )
    )


def _phase_plan_torchscript_trace_key(theta, phase_coeffs, fit_rows, cutoff_fMs):
    """Key one generic graph by runtime topology, never physical values."""

    thread_settings = _phase_plan_torchscript_trace_thread_settings()
    if thread_settings is None:
        raise RuntimeError("Torch CPU thread topology is unavailable")
    return (
        torch.__version__,
        sys.version_info[:3],
        theta.device.type,
        theta.device.index,
        thread_settings,
        torch.get_default_dtype(),
        tuple(
            (
                value.dtype,
                tuple(value.shape),
                value.stride(),
                value.storage_offset(),
                value._base is not None,
            )
            for value in (theta, phase_coeffs, fit_rows, *cutoff_fMs)
        ),
        _phase_plan_torchscript_trace_environment(),
    )


def _ensure_phase_plan_torchscript_trace_process() -> None:
    """Reset inherited state where an at-fork callback was unavailable."""

    if _PHASE_PLAN_TORCHSCRIPT_TRACE_PID != os.getpid():
        _reset_phase_plan_torchscript_trace_after_fork()


def _clear_phase_plan_torchscript_trace_cache() -> None:
    """Clear warm programs and sticky failures for testing and debugging."""

    _ensure_phase_plan_torchscript_trace_process()
    with _PHASE_PLAN_TORCHSCRIPT_TRACE_LOCK:
        _PHASE_PLAN_TORCHSCRIPT_TRACE_CACHE.clear()


def _phase_plan_torchscript_trace_cache_state():
    """Return immutable cache keys and failure keys for instrumentation."""

    _ensure_phase_plan_torchscript_trace_process()
    with _PHASE_PLAN_TORCHSCRIPT_TRACE_LOCK:
        ready = tuple(
            key
            for key, executor in _PHASE_PLAN_TORCHSCRIPT_TRACE_CACHE.items()
            if executor is not None
        )
        failed = tuple(
            key
            for key, executor in _PHASE_PLAN_TORCHSCRIPT_TRACE_CACHE.items()
            if executor is None
        )
        return ready, failed


def _phase_plan_torchscript_trace_store(key, executor) -> None:
    """Store a ready executor or sticky failure in one bounded cache."""

    _PHASE_PLAN_TORCHSCRIPT_TRACE_CACHE.pop(key, None)
    _PHASE_PLAN_TORCHSCRIPT_TRACE_CACHE[key] = executor
    while (
        len(_PHASE_PLAN_TORCHSCRIPT_TRACE_CACHE)
        > _PHASE_PLAN_TORCHSCRIPT_TRACE_MAX_ENTRIES
    ):
        _PHASE_PLAN_TORCHSCRIPT_TRACE_CACHE.popitem(last=False)


def _phase_plan_torchscript_primary_values(plan):
    """Return the fixed-schema primary plan leaves in eager field order."""

    return (
        plan.total_mass_seconds,
        plan.eta,
        plan.f1_Ms,
        plan.f2_Ms,
        *plan.inspiral,
        *plan.intermediate,
        *plan.mergerringdown,
        plan.alpha0,
        plan.alpha1,
        plan.beta0,
        plan.beta1,
    )


def _phase_plan_torchscript_numeric(
    theta,
    phase_coeffs,
    fit_rows,
    fMs_RD,
    fMs_damp,
    fMs_MECO,
    fMs_ISCO,
):
    """Trace only parameter-dependent tensor leaves of the eager scalar plan."""

    plan = _prepare_phase_plan_eager(
        theta,
        phase_coeffs,
        0.0,
        final_spin=None,
        _phase_fit_rows=fit_rows,
        _cutoff_fMs=(fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO),
        _request_proof=False,
    )
    return tuple(
        value
        for value in _phase_plan_torchscript_primary_values(plan)
        if type(value) is torch.Tensor
    )


def _phase_plan_torchscript_rebuild_plain(tensor_values):
    """Rebuild the fixed plan schema from 37 traced tensor leaves."""

    tensors = iter(tensor_values)
    top = tuple(next(tensors) for _ in range(4))
    inspiral = _InspiralPhasePlan(
        1.0,
        0.0,
        next(tensors),
        next(tensors),
        next(tensors),
        0.0,
        next(tensors),
        next(tensors),
        (-6848.0 / 63.0) * PI**2.0,
        next(tensors),
        next(tensors),
        next(tensors),
        next(tensors),
        next(tensors),
        next(tensors),
        next(tensors),
    )
    intermediate = _IntermediatePhasePlan(
        *(next(tensors) for _ in range(8))
    )
    ringdown_values = tuple(next(tensors) for _ in range(9))
    # TorchScript CSE returns the two cL outputs as one Tensor object.  Eager
    # returns distinct Tensor wrappers over the same storage, which is
    # observable through both identity and in-place mutation.  Detach restores
    # that exact wrapper/storage relationship without changing any bytes.
    ringdown_values = (
        *ringdown_values[:7],
        ringdown_values[7].detach(),
        ringdown_values[8],
    )
    ringdown = _MergerRingdownPhasePlan(*ringdown_values)
    corrections = tuple(next(tensors) for _ in range(4))
    try:
        next(tensors)
    except StopIteration:
        pass
    else:
        raise RuntimeError("phase-plan trace returned extra tensor leaves")
    return _IMRPhenomXASPhasePlan(
        *top,
        inspiral,
        intermediate,
        ringdown,
        inspiral,
        intermediate,
        ringdown,
        *corrections,
        False,
    )


def _phase_plan_torchscript_finalize(
    plan,
    theta,
    scalar_region_dispatch,
    request_proof,
):
    """Reapply ordinary request-local proof and executor plan semantics."""

    prequalify_scalar_derivative_plans = (
        _scalar_derivative_plan_cse_enabled()
        and _exact_scalar_derivatives_enabled()
    )
    scalar_inspiral = _maybe_prequalify_scalar_derivative_plan(
        _detach_phase_plan(plan.inspiral),
        plan.f1_Ms,
        enabled=prequalify_scalar_derivative_plans,
        _request_proof=request_proof,
    )
    scalar_intermediate = _maybe_prequalify_scalar_derivative_plan(
        _detach_phase_plan(plan.intermediate),
        plan.f1_Ms,
        enabled=prequalify_scalar_derivative_plans,
        _request_proof=request_proof,
    )
    scalar_ringdown_source = _detach_phase_plan(plan.mergerringdown)
    if prequalify_scalar_derivative_plans:
        # The eager exact scalar derivative returns cL itself while assembling
        # the intermediate plan.  Consequently intermediate.cL and the scalar
        # ringdown cL are the same Tensor wrapper, not merely storage aliases.
        scalar_ringdown_source = scalar_ringdown_source._replace(
            cL=plan.intermediate.cL,
        )
    scalar_ringdown = _maybe_prequalify_scalar_derivative_plan(
        scalar_ringdown_source,
        plan.f2_Ms,
        enabled=prequalify_scalar_derivative_plans,
        _request_proof=request_proof,
    )
    inspiral, intermediate, ringdown = (
        _maybe_prequalify_scripted_phase_ansatz_cpu_plans(
            plan.inspiral,
            plan.intermediate,
            plan.mergerringdown,
            theta,
            _request_proof=request_proof,
        )
    )
    inspiral, intermediate, ringdown = (
        _maybe_prequalify_cuda_graph_phase_ansatz_plans(
            inspiral,
            intermediate,
            ringdown,
            theta,
        )
    )
    result = _IMRPhenomXASPhasePlan(
        *plan[:4],
        inspiral,
        intermediate,
        ringdown,
        scalar_inspiral,
        scalar_intermediate,
        scalar_ringdown,
        *plan[10:14],
        scalar_region_dispatch,
    )
    return _request_qualify_top_plan(result, request_proof)


def _phase_plan_torchscript_tree_raw_equal(reference, candidate) -> bool:
    """Compare plan bytes, metadata, Python leaves, and proof subtypes exactly."""

    if type(reference) is torch.Tensor or type(candidate) is torch.Tensor:
        if not (
            type(reference) is torch.Tensor
            and type(candidate) is torch.Tensor
            and reference.layout == candidate.layout == torch.strided
            and reference.device == candidate.device
            and reference.dtype == candidate.dtype
            and reference.shape == candidate.shape
            and reference.stride() == candidate.stride()
            and reference.storage_offset() == candidate.storage_offset()
            and reference.is_conj() == candidate.is_conj()
            and reference.is_neg() == candidate.is_neg()
        ):
            return False
        return torch.equal(
            reference.detach().contiguous().reshape(-1).view(torch.uint8),
            candidate.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    if isinstance(reference, tuple) or isinstance(candidate, tuple):
        return (
            type(reference) is type(candidate)
            and len(reference) == len(candidate)
            and all(
                _phase_plan_torchscript_tree_raw_equal(left, right)
                for left, right in zip(reference, candidate)
            )
        )
    return type(reference) is type(candidate) and reference == candidate


def _phase_plan_torchscript_tree_alias_equal(reference, candidate) -> bool:
    """Compare Tensor identity and storage-alias topology exactly."""

    reference_leaves = _phase_plan_tensor_leaves(reference)
    candidate_leaves = _phase_plan_tensor_leaves(candidate)
    alias_of = getattr(torch._C, "_is_alias_of", None)
    if len(reference_leaves) != len(candidate_leaves) or not callable(alias_of):
        return False
    try:
        for index, (reference_left, candidate_left) in enumerate(
            zip(reference_leaves, candidate_leaves)
        ):
            for reference_right, candidate_right in zip(
                reference_leaves[index + 1 :],
                candidate_leaves[index + 1 :],
            ):
                if (reference_left is reference_right) != (
                    candidate_left is candidate_right
                ) or bool(alias_of(reference_left, reference_right)) != bool(
                    alias_of(candidate_left, candidate_right)
                ):
                    return False
    except Exception:
        return False
    return True


def _build_phase_plan_torchscript_trace(inputs):
    """Build the exact make_fx-to-TorchScript executor for one topology."""

    from torch.fx.experimental.proxy_tensor import make_fx

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.(trace|trace_method|script|freeze)` is deprecated.*",
            category=DeprecationWarning,
        )
        graph = make_fx(
            _phase_plan_torchscript_numeric,
            tracing_mode="real",
        )(*inputs)
        executor = torch.jit.trace(
            graph,
            inputs,
            check_trace=False,
            strict=True,
        )
        executor = torch.jit.freeze(executor.eval())
    return executor


def _maybe_prepare_phase_plan_torchscript_trace(
    theta,
    phase_coeffs,
    chip,
    final_spin,
    coprecessing_deviations,
    phase_fit_rows,
    cutoff_fMs,
    intrinsic_controls,
    request_proof,
):
    """Return an exact warm traced plan, a cold eager reference, or ``None``."""

    if not _phase_plan_torchscript_trace_supported(
        theta,
        phase_coeffs,
        chip,
        final_spin,
        coprecessing_deviations,
        phase_fit_rows,
        cutoff_fMs,
        intrinsic_controls,
        request_proof,
    ):
        return None
    _ensure_phase_plan_torchscript_trace_process()

    if phase_fit_rows is None:
        try:
            with torch_context(theta):
                fit_rows = _prepare_phase_fit_rows(theta, phase_coeffs)
        except Exception:
            return None
    else:
        fit_rows = phase_fit_rows
    if not _phase_plan_torchscript_trace_tensor_supported(
        fit_rows,
        device=theta.device,
        shape=torch.Size((13,)),
    ):
        return None

    try:
        key = _phase_plan_torchscript_trace_key(
            theta,
            phase_coeffs,
            fit_rows,
            cutoff_fMs,
        )
    except Exception:
        return None
    inputs = (theta, phase_coeffs, fit_rows, *cutoff_fMs)
    scalar_region_dispatch = (
        _scalar_region_dispatch_enabled()
        and not IMRPhenomX_utils._tree_has_autograd(
            (
                theta,
                phase_coeffs,
                chip,
                final_spin,
                coprecessing_deviations,
                phase_fit_rows,
                cutoff_fMs,
            )
        )
    )

    with _PHASE_PLAN_TORCHSCRIPT_TRACE_LOCK:
        executor = _PHASE_PLAN_TORCHSCRIPT_TRACE_CACHE.get(
            key,
            _PHASE_PLAN_TORCHSCRIPT_TRACE_MISSING,
        )
    if executor is None:
        return None
    if executor is not _PHASE_PLAN_TORCHSCRIPT_TRACE_MISSING:
        try:
            values = executor(*inputs)
            plain = _phase_plan_torchscript_rebuild_plain(values)
            return _phase_plan_torchscript_finalize(
                plain,
                theta,
                scalar_region_dispatch,
                request_proof,
            )
        except Exception:
            with _PHASE_PLAN_TORCHSCRIPT_TRACE_LOCK:
                _phase_plan_torchscript_trace_store(key, None)
            return None

    inspiral_host_scalars = _maybe_inspiral_phase_host_scalars(
        theta,
        phase_coeffs,
        chip,
        final_spin,
        coprecessing_deviations,
        phase_fit_rows,
        cutoff_fMs,
        intrinsic_controls,
    )
    reference_controls = intrinsic_controls
    if inspiral_host_scalars is not None:
        reference_controls = inspiral_host_scalars.intrinsic_controls
    with torch_context(theta):
        reference = _prepare_phase_plan_eager(
            theta,
            phase_coeffs,
            chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
            _phase_fit_rows=phase_fit_rows,
            _cutoff_fMs=cutoff_fMs,
            _intrinsic_controls=reference_controls,
            _inspiral_phase_host_scalars=inspiral_host_scalars,
            _request_proof=request_proof,
        )

    with _PHASE_PLAN_TORCHSCRIPT_TRACE_LOCK:
        if key in _PHASE_PLAN_TORCHSCRIPT_TRACE_CACHE:
            return reference
        try:
            with torch_context(theta):
                executor = _build_phase_plan_torchscript_trace(inputs)
            values = executor(*inputs)
            candidate = _phase_plan_torchscript_finalize(
                _phase_plan_torchscript_rebuild_plain(values),
                theta,
                scalar_region_dispatch,
                request_proof,
            )
            if not _phase_plan_torchscript_tree_raw_equal(
                reference,
                candidate,
            ) or not _phase_plan_torchscript_tree_alias_equal(
                reference,
                candidate,
            ):
                raise RuntimeError(
                    "phase-plan trace changed output bytes, metadata, or aliases"
                )
        except Exception:
            _phase_plan_torchscript_trace_store(key, None)
        else:
            _phase_plan_torchscript_trace_store(key, executor)
    # The first request always observes the unchanged eager result.  Only a
    # byte-qualified executor may serve later requests with the same topology.
    return reference


def _phase_plan_cuda_solve_graph_environment():
    """Return every PyCBC switch that can affect captured eager kernels."""

    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("PYCBC_")
        )
    )


def _phase_plan_cuda_solve_graph_tensor_supported(
    value,
    *,
    shape,
    device,
    dtype,
):
    """Accept one plain, contiguous, independent CUDA graph input."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.device == device
        and value.dtype == dtype
        and value.shape == shape
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and not value.is_conj()
        and not value.is_neg()
        and not IMRPhenomX_utils._tree_has_autograd(value)
    )


def _phase_plan_cuda_solve_graph_supported(
    theta,
    phase_coeffs,
    chip,
    final_spin,
    coprecessing_deviations,
    phase_fit_rows,
):
    """Fail closed outside the byte-qualified CUDA phase-plan inputs."""

    if (
        type(theta) is not torch.Tensor
        or theta.device.type != "cuda"
        or theta.dtype not in (torch.float32, torch.float64)
        or not torch.cuda.is_available()
        or not _torch_jax_compat._cuda_device_scalars_enabled()
        or coprecessing_deviations is not None
        or type(chip) is not float
        or not math.isfinite(chip)
        or type(final_spin) is not float
        or not math.isfinite(final_spin)
    ):
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
    except RuntimeError:
        return False
    device = theta.device
    dtype = theta.dtype
    if not _phase_plan_cuda_solve_graph_tensor_supported(
        theta,
        shape=(4,),
        device=device,
        dtype=dtype,
    ):
        return False
    if not _phase_plan_cuda_solve_graph_tensor_supported(
        phase_coeffs,
        shape=(13, 49),
        device=device,
        dtype=dtype,
    ):
        return False
    canonical_coeffs = (
        IMRPhenomX_utils._get_phenomx_phase_coeff_table_cached_master(
            device=torch.device("cuda"),
            dtype=dtype,
        ),
        IMRPhenomX_utils._get_phenomx_phase_coeff_table_cached_master(
            device=device,
            dtype=dtype,
        ),
    )
    if (
        not any(phase_coeffs is candidate for candidate in canonical_coeffs)
        or phase_coeffs._version != 0
    ):
        return False
    if phase_fit_rows is not None and not (
        _phase_plan_cuda_solve_graph_tensor_supported(
            phase_fit_rows,
            shape=(13,),
            device=device,
            dtype=dtype,
        )
    ):
        return False
    return True


def _phase_plan_cuda_solve_graph_key(
    theta,
    phase_coeffs,
    phase_fit_rows,
):
    """Return a process/thread/device/stream/environment-local key."""

    device = theta.device
    stream = torch.cuda.current_stream(device)
    return (
        os.getpid(),
        threading.get_ident(),
        theta.dtype,
        device,
        stream.cuda_stream,
        phase_coeffs.data_ptr(),
        phase_coeffs._version,
        phase_fit_rows is not None,
        _phase_plan_cuda_solve_graph_environment(),
    )


def _phase_plan_tensor_leaves(value):
    """Return the tensor leaves of a phase plan in named-tuple order."""

    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, tuple):
        return tuple(
            leaf
            for item in value
            for leaf in _phase_plan_tensor_leaves(item)
        )
    return ()


def _phase_plan_from_owned_packed(template, packed):
    """Rebuild a phase plan from views of one request-owned packed clone."""

    tensor_index = 0

    def rebuild(value):
        nonlocal tensor_index
        if isinstance(value, torch.Tensor):
            result = packed[tensor_index]
            tensor_index += 1
            return result
        if isinstance(value, tuple):
            items = tuple(rebuild(item) for item in value)
            if hasattr(type(value), "_fields"):
                return type(value)(*items)
            return items
        return value

    result = rebuild(template)
    if tensor_index != packed.shape[0]:
        raise RuntimeError("packed CUDA phase plan has an invalid schema")
    return result


def _phase_plan_cuda_solve_graph_remember_failure(key):
    """Bound and remember one capture/replay failure."""

    _PHASE_PLAN_CUDA_SOLVE_GRAPH_CACHE.pop(key, None)
    if (
        len(_PHASE_PLAN_CUDA_SOLVE_GRAPH_FAILURES)
        >= _PHASE_PLAN_CUDA_SOLVE_GRAPH_MAX_ENTRIES
    ):
        _PHASE_PLAN_CUDA_SOLVE_GRAPH_FAILURES.pop()
    _PHASE_PLAN_CUDA_SOLVE_GRAPH_FAILURES.add(key)


def _phase_plan_cuda_solve_graph_store(key, state):
    """Insert one graph while bounding device-memory lifetime."""

    if key not in _PHASE_PLAN_CUDA_SOLVE_GRAPH_CACHE and len(
        _PHASE_PLAN_CUDA_SOLVE_GRAPH_CACHE
    ) >= (
        _PHASE_PLAN_CUDA_SOLVE_GRAPH_MAX_ENTRIES
    ):
        oldest = next(iter(_PHASE_PLAN_CUDA_SOLVE_GRAPH_CACHE))
        del _PHASE_PLAN_CUDA_SOLVE_GRAPH_CACHE[oldest]
    _PHASE_PLAN_CUDA_SOLVE_GRAPH_CACHE[key] = state


def _clear_phase_plan_cuda_solve_graph_cache():
    """Release cached phase graphs and remembered failures."""

    with _PHASE_PLAN_CUDA_SOLVE_GRAPH_LOCK:
        _PHASE_PLAN_CUDA_SOLVE_GRAPH_CACHE.clear()
        _PHASE_PLAN_CUDA_SOLVE_GRAPH_FAILURES.clear()


def _build_phase_plan_cuda_solve_graph(
    theta,
    phase_coeffs,
    chip,
    final_spin,
    phase_fit_rows,
):
    """Capture unchanged phase algebra with asynchronous ``solve_ex``."""

    device = theta.device
    static_theta = theta.clone()
    static_chip = theta.new_tensor(chip)
    static_final_spin = theta.new_tensor(final_spin)
    static_fit_rows = (
        None if phase_fit_rows is None else phase_fit_rows.clone()
    )

    def plan_call(solve_info):
        with torch_context(static_theta), IMRPhenomX_utils.remnant_cache_context(
            enabled=False
        ):
            return _prepare_phase_plan_eager(
                static_theta,
                phase_coeffs,
                static_chip,
                final_spin=static_final_spin,
                _phase_fit_rows=static_fit_rows,
                _solve_info=solve_info,
                _reuse_cutoff_fMs=True,
            )

    current_stream = torch.cuda.current_stream(device)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(current_stream)
    warmup_info = []
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            one_call_info = []
            plan_call(one_call_info)
            if len(one_call_info) != 3:
                raise RuntimeError("CUDA phase plan did not execute three solves")
            warmup_info.extend(one_call_info)
        warmup_status = torch.stack(tuple(warmup_info)).abs().sum()
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize(device)
    if int(warmup_status.item()) != 0:
        raise RuntimeError("CUDA phase-plan warmup solve failed")

    graph = torch.cuda.CUDAGraph()
    capture_info = []
    with torch.cuda.graph(graph, stream=capture_stream):
        template = plan_call(capture_info)
        if len(capture_info) != 3:
            raise RuntimeError("CUDA phase plan did not capture three solves")
        packed_plan = torch.stack(_phase_plan_tensor_leaves(template))
        graph_info = torch.stack(tuple(capture_info)).abs().sum()
    torch.cuda.synchronize(device)
    if int(graph_info.item()) != 0:
        raise RuntimeError("CUDA phase-plan capture solve failed")
    return _PhasePlanCudaSolveGraphState(
        static_theta=static_theta,
        phase_coeffs=phase_coeffs,
        static_chip=static_chip,
        static_final_spin=static_final_spin,
        static_fit_rows=static_fit_rows,
        capture_stream=capture_stream,
        graph=graph,
        packed_plan=packed_plan,
        graph_info=graph_info,
        template=template,
    )


def _replay_phase_plan_cuda_solve_graph(
    state,
    theta,
    chip,
    final_spin,
    phase_fit_rows,
):
    """Replay one graph, synchronously validate all solves, and own output."""

    state.static_theta.copy_(theta)
    state.static_chip.fill_(chip)
    state.static_final_spin.fill_(final_spin)
    if state.static_fit_rows is not None:
        state.static_fit_rows.copy_(phase_fit_rows)
    state.graph.replay()
    # solve_ex(check_errors=False) avoids the per-solve CUDA synchronizations
    # that make ordinary solve uncapturable. One aggregate status check retains
    # ordinary solve's fail-closed behavior before any result is returned.
    if int(state.graph_info.item()) != 0:
        return None
    owned = state.packed_plan.clone()
    # Capture allocates ``packed_plan`` on a private capture stream, while
    # replay and this clone use the caller's stream.  Cache eviction (including
    # a clear from another thread) can drop the last graph-state reference as
    # soon as this function returns.  Record the clone's stream so the CUDA
    # allocator cannot recycle the packed source before that clone completes.
    state.packed_plan.record_stream(torch.cuda.current_stream(theta.device))
    return _phase_plan_from_owned_packed(state.template, owned)


def _prepare_phase_plan_cuda_solve_graph(
    theta,
    phase_coeffs,
    chip,
    final_spin,
    coprecessing_deviations,
    phase_fit_rows,
):
    """Return an owned captured plan, or ``None`` for ordinary eager solve."""

    if not _phase_plan_cuda_solve_graph_supported(
        theta,
        phase_coeffs,
        chip,
        final_spin,
        coprecessing_deviations,
        phase_fit_rows,
    ):
        return None
    key = _phase_plan_cuda_solve_graph_key(
        theta,
        phase_coeffs,
        phase_fit_rows,
    )
    with _PHASE_PLAN_CUDA_SOLVE_GRAPH_LOCK:
        state = _PHASE_PLAN_CUDA_SOLVE_GRAPH_CACHE.get(key)
        if (
            state is None
            and key not in _PHASE_PLAN_CUDA_SOLVE_GRAPH_FAILURES
        ):
            try:
                state = _build_phase_plan_cuda_solve_graph(
                    theta,
                    phase_coeffs,
                    chip,
                    final_spin,
                    phase_fit_rows,
                )
            except Exception:
                _phase_plan_cuda_solve_graph_remember_failure(key)
            else:
                _phase_plan_cuda_solve_graph_store(key, state)
    if state is None:
        return None
    try:
        return _replay_phase_plan_cuda_solve_graph(
            state,
            theta,
            chip,
            final_spin,
            phase_fit_rows,
        )
    except Exception:
        with _PHASE_PLAN_CUDA_SOLVE_GRAPH_LOCK:
            _phase_plan_cuda_solve_graph_remember_failure(key)
        return None


def _prepare_phase_plan(
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    chip: FloatLike = 0.0,
    *,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
    _phase_fit_rows: torch.Tensor | None = None,
    _cutoff_fMs=None,
    _intrinsic_controls=None,
    _request_proof=None,
) -> _IMRPhenomXASPhasePlan:
    """Prepare the exact piecewise phase once for one waveform request."""

    if (
        _intrinsic_controls is None
        and _cutoff_fMs is None
        and _phase_plan_cuda_solve_graph_enabled()
    ):
        captured = _prepare_phase_plan_cuda_solve_graph(
            theta,
            phase_coeffs,
            chip,
            final_spin,
            coprecessing_deviations,
            _phase_fit_rows,
        )
        if captured is not None:
            return captured
    traced = _maybe_prepare_phase_plan_torchscript_trace(
        theta,
        phase_coeffs,
        chip,
        final_spin,
        coprecessing_deviations,
        _phase_fit_rows,
        _cutoff_fMs,
        _intrinsic_controls,
        _request_proof,
    )
    if traced is not None:
        return traced
    if _FIXED_SCHEMA_PHASE_PLAN_ENV in os.environ:
        from ._imrphenomxas_fixed_schema_phase_plan import (
            _maybe_prepare_fixed_schema_phase_plan,
        )

        fixed_schema = _maybe_prepare_fixed_schema_phase_plan(
            theta,
            phase_coeffs,
            chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
            _phase_fit_rows=_phase_fit_rows,
            _cutoff_fMs=_cutoff_fMs,
            _intrinsic_controls=_intrinsic_controls,
            _request_proof=_request_proof,
        )
        if fixed_schema is not None:
            return fixed_schema
    inspiral_phase_host_scalars = _maybe_inspiral_phase_host_scalars(
        theta,
        phase_coeffs,
        chip,
        final_spin,
        coprecessing_deviations,
        _phase_fit_rows,
        _cutoff_fMs,
        _intrinsic_controls,
    )
    if inspiral_phase_host_scalars is not None:
        _intrinsic_controls = inspiral_phase_host_scalars.intrinsic_controls
    return _prepare_phase_plan_eager(
        theta,
        phase_coeffs,
        chip,
        final_spin=final_spin,
        coprecessing_deviations=coprecessing_deviations,
        _phase_fit_rows=_phase_fit_rows,
        _cutoff_fMs=_cutoff_fMs,
        _intrinsic_controls=_intrinsic_controls,
        _inspiral_phase_host_scalars=inspiral_phase_host_scalars,
        _request_proof=_request_proof,
    )


def _scalar_phase_region(f, plan):
    """Select one safe scalar phase region, or preserve the dense path.

    Higher-mode setup uses CPU float64 scalar anchors even for CUDA waveform
    evaluation. Dispatching only those immutable, non-autograd requests avoids
    evaluating two unused ansatz regions without introducing a device scalar
    synchronization or changing differentiable execution. Exact joins retain
    the original piecewise assembly and its boundary convention.
    """

    if (
        not plan.scalar_region_dispatch
        or type(f) is not torch.Tensor
        or f.layout != torch.strided
        or f.ndim != 0
        or f.device.type != "cpu"
        or f.is_conj()
        or f.is_neg()
        or IMRPhenomX_utils._tree_has_autograd(f)
    ):
        return None

    fM_s = f * plan.total_mass_seconds
    if (
        type(fM_s) is not torch.Tensor
        or fM_s.layout != torch.strided
        or fM_s.ndim != 0
        or fM_s.device.type != "cpu"
        or fM_s.is_conj()
        or fM_s.is_neg()
        or IMRPhenomX_utils._tree_has_autograd(fM_s)
    ):
        return None

    value = float(fM_s)
    f1_Ms = float(plan.f1_Ms)
    f2_Ms = float(plan.f2_Ms)
    cutoff = float(fM_s.new_tensor(IMRPhenomX_utils.fM_CUT))
    if (
        not math.isfinite(value)
        or value <= 0.0
        or value >= cutoff
        or value == f1_Ms
        or value == f2_Ms
    ):
        return None
    if value < f1_Ms:
        region = 0
    elif value < f2_Ms:
        region = 1
    else:
        region = 2
    return region, fM_s


def _evaluate_pruned_phase(frequency, plan, indices):
    """Evaluate only the active phase spans with the eager formulas unchanged."""

    first, second, cutoff = indices
    phase = torch.zeros_like(frequency)
    phase[:first] = _evaluate_aligned_region(
        frequency,
        0,
        first,
        lambda value: _evaluate_inspiral_phase(value, plan.inspiral),
    )
    phase[first:second] = _evaluate_aligned_region(
        frequency,
        first,
        second,
        lambda value: (
            _evaluate_intermediate_phase(value, plan.intermediate)
            + plan.alpha1 * value
            + plan.alpha0
        ),
    )
    phase[second:cutoff] = _evaluate_aligned_region(
        frequency,
        second,
        cutoff,
        lambda value: (
            _evaluate_mergerringdown_phase(value, plan.mergerringdown)[0]
            + plan.beta0
            + plan.beta1 * value
        ),
    )
    return (1 / plan.eta) * phase


def _evaluate_phase(
    f: Float[Array, " n_freq"] | float,
    plan: _IMRPhenomXASPhasePlan,
    *,
    _request_proof=None,
) -> Float[Array, " n_freq"]:
    """Evaluate an already prepared exact piecewise phase."""

    scalar_region = _scalar_phase_region(f, plan)
    if scalar_region is not None:
        region, fM_s = scalar_region
        if region == 0:
            phase = _evaluate_inspiral_phase(fM_s, plan.inspiral)
        elif region == 1:
            phase = (
                _evaluate_intermediate_phase(fM_s, plan.intermediate)
                + plan.alpha1 * fM_s
                + plan.alpha0
            )
        else:
            phase, _ = _evaluate_mergerringdown_phase(
                fM_s,
                plan.mergerringdown,
            )
            phase = phase + plan.beta0 + plan.beta1 * fM_s
        return (1 / plan.eta) * phase

    fM_s = f * plan.total_mass_seconds
    region_indices = _piecewise_region_indices(
        f,
        fM_s,
        plan.f1_Ms,
        plan.f2_Ms,
        plan,
        _request_proof=_request_proof,
    )
    if region_indices is not None:
        return _evaluate_pruned_phase(fM_s, plan, region_indices)

    scripted_ansatzes = _evaluate_cuda_graph_phase_ansatz_triplet(fM_s, plan)
    if scripted_ansatzes is None:
        scripted_ansatzes = _evaluate_scripted_phase_ansatz_triplet(
            fM_s,
            plan,
            _request_proof=_request_proof,
        )
    if scripted_ansatzes is None:
        phi_Ins = _evaluate_inspiral_phase(fM_s, plan.inspiral)
        phi_Int_corrected = (
            _evaluate_intermediate_phase(fM_s, plan.intermediate)
            + plan.alpha1 * fM_s
            + plan.alpha0
        )
        phi_MRD, _ = _evaluate_mergerringdown_phase(
            fM_s,
            plan.mergerringdown,
        )
    else:
        phi_Ins, phi_Int, phi_MRD = scripted_ansatzes
        phi_Int_corrected = phi_Int + plan.alpha1 * fM_s + plan.alpha0
    phi_MRD_corrected = phi_MRD + plan.beta0 + plan.beta1 * fM_s

    packed_masks = _maybe_packed_heaviside_masks(
        fM_s,
        plan.f1_Ms,
        plan.f2_Ms,
        _request_plan=plan,
        _request_proof=_request_proof,
    )
    if packed_masks is not None:
        inspiral_mask, intermediate_mask, intermediate_end, ringdown_mask, cutoff = (
            packed_masks
        )
        return (1 / plan.eta) * (
            phi_Ins * inspiral_mask
            + intermediate_mask * phi_Int_corrected * intermediate_end
            + phi_MRD_corrected * ringdown_mask * cutoff
        )

    return (1 / plan.eta) * (
        phi_Ins * jnp.heaviside(plan.f1_Ms - fM_s, 0.5)
        + jnp.heaviside(fM_s - plan.f1_Ms, 0.5)
        * phi_Int_corrected
        * jnp.heaviside(plan.f2_Ms - fM_s, 0.5)
        + phi_MRD_corrected
        * jnp.heaviside(fM_s - plan.f2_Ms, 0.5)
        * jnp.heaviside(IMRPhenomX_utils.fM_CUT - fM_s, 0.5)
    )


def _evaluate_phase_derivative(
    f: Float[Array, " n_freq"],
    plan: _IMRPhenomXASPhasePlan,
    *,
    _request_proof=None,
) -> Float[Array, " n_freq"]:
    """Evaluate dPhase/df from an already prepared phase plan."""

    fM_s = f * plan.total_mass_seconds

    scalar_region = _scalar_phase_region(f, plan)
    if scalar_region is not None:
        region, fM_s = scalar_region
        if region == 0:
            derivative = _inspiral_phase_derivative(
                fM_s,
                plan.scalar_inspiral,
                _request_proof=_request_proof,
            )
        elif region == 1:
            derivative = _intermediate_phase_derivative(
                fM_s,
                plan.scalar_intermediate,
                correction=(plan.alpha0, plan.alpha1),
                _request_proof=_request_proof,
            )
        else:
            derivative = (
                _mergerringdown_phase_derivative(
                    fM_s,
                    plan.scalar_mergerringdown,
                    _request_proof=_request_proof,
                )
                + plan.beta1
            )
        return (derivative / plan.eta) * plan.total_mass_seconds

    dphi_Ins = _inspiral_phase_derivative(
        fM_s,
        plan.scalar_inspiral,
        _request_proof=_request_proof,
    )
    dphi_Int = _intermediate_phase_derivative(
        fM_s,
        plan.scalar_intermediate,
        correction=(plan.alpha0, plan.alpha1),
        _request_proof=_request_proof,
    )
    dphi_MRD = (
        _mergerringdown_phase_derivative(
            fM_s,
            plan.scalar_mergerringdown,
            _request_proof=_request_proof,
        )
        + plan.beta1
    )
    dphase_dMf = jnp.where(
        fM_s < plan.f1_Ms,
        dphi_Ins / plan.eta,
        jnp.where(
            fM_s < plan.f2_Ms,
            dphi_Int / plan.eta,
            dphi_MRD / plan.eta,
        ),
    )
    return dphase_dMf * plan.total_mass_seconds


def Phase(
    f: Float[Array, " n_freq"] | float,
    theta: Float[Array, "4"],
    phase_coeffs: Float[Array, "13 49"],
    chip: FloatLike = 0.0,
    *,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
    _phase_plan: _IMRPhenomXASPhasePlan | None = None,
    _cutoff_fMs=None,
    _request_proof=None,
) -> Float[Array, " n_freq"]:
    """
    Computes the phase of the PhenomD waveform following 1508.07253.
    Sets time and phase of coealence to be zero.

    Returns:
        phase (array): Phase of the GW as a function of frequency
    """
    if _phase_plan is not None:
        return _evaluate_phase(
            f,
            _phase_plan,
            _request_proof=_request_proof,
        )

    m1, m2, chi1, chi2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    fM_s = f * M_s
    if _cutoff_fMs is None:
        fMs_RD, _, fMs_MECO, fMs_ISCO = _get_cutoff_fMs(
            m1,
            m2,
            chi1,
            chi2,
            chip=chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
        )
    else:
        fMs_RD, _, fMs_MECO, fMs_ISCO = _cutoff_fMs
    fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
    fMs_INmatch = fMs_MECO
    deltafMs = (fMs_IMmatch - fMs_INmatch) * 0.03
    f1_Ms = fMs_INmatch - 1.0 * deltafMs
    f2_Ms = fMs_IMmatch + 0.5 * deltafMs

    # Prepare each region once. The same frequency-independent collocation
    # systems are used for both the dense phase and the scalar C1 matching
    # evaluations below.
    # Bulk row evaluation is forward-bitwise exact. Reverse-mode accumulation
    # across rows can use a different order, so retain the scalar fit path when
    # callers request derivatives with respect to intrinsic inputs or tables.
    fit_rows = None
    if not IMRPhenomX_utils._tree_has_autograd((theta, phase_coeffs)):
        fit_rows = _prepare_phase_fit_rows(theta, phase_coeffs)
    inspiral_plan = _prepare_inspiral_phase(
        theta,
        phase_coeffs,
        fit_rows,
        _cutoff_fMs=_cutoff_fMs,
    )
    phi_Ins = _evaluate_inspiral_phase(fM_s, inspiral_plan)
    mergerringdown_plan = _prepare_mergerringdown_phase(
        theta,
        phase_coeffs,
        chip,
        fit_rows=fit_rows,
        final_spin=final_spin,
        coprecessing_deviations=coprecessing_deviations,
        _cutoff_fMs=_cutoff_fMs,
    )
    phi_MRD, (cL, CV_phase_RD0) = _evaluate_mergerringdown_phase(
        fM_s,
        mergerringdown_plan,
    )

    # Get matching points
    # Here we want to evaluate the gradient and the phase of the raw phase functions
    # in order to enforce C1 continuity at the transition frequencies.
    # This procedure is identical to IMRPhenomD, see IMRPhenomD.py for more details
    # Matching values are detached by ``value_and_grad``. Detaching their plan
    # as well prevents this internal derivative from consuming a parameter-
    # gradient graph shared with the dense waveform evaluation.
    scalar_inspiral_plan = _detach_phase_plan(inspiral_plan)
    phi_Ins_match_f1, dphi_Ins_match_f1 = _inspiral_phase_value_and_derivative(
        f1_Ms,
        scalar_inspiral_plan,
    )
    (phi_MRD_match_f2, _), dphi_MRD_match_f2 = (
        _mergerringdown_phase_value_and_derivative(
            f2_Ms,
            _detach_phase_plan(mergerringdown_plan),
        )
    )
    # ``value_and_grad`` deliberately detaches its value. Re-evaluate the
    # inexpensive ansatz with the graph-bearing plan so derivatives of the
    # assembled phase with respect to intrinsic parameters retain the original
    # merger-ringdown matching contribution. This reuses the fit and solve.
    phi_MRD_match_f2, _ = _evaluate_mergerringdown_phase(
        f2_Ms,
        mergerringdown_plan,
    )

    # Now find the intermediate phase
    intermediate_plan = _prepare_intermediate_phase(
        theta,
        phase_coeffs,
        dphi_Ins_match_f1,
        CV_phase_RD0,
        cL,
        chip,
        fit_rows=fit_rows,
        final_spin=final_spin,
        coprecessing_deviations=coprecessing_deviations,
        _cutoff_fMs=_cutoff_fMs,
    )
    scalar_intermediate_plan = _detach_phase_plan(intermediate_plan)
    phi_Int_match_f1, dphi_Int_match_f1 = _intermediate_phase_value_and_derivative(
        f1_Ms,
        scalar_intermediate_plan,
    )
    alpha1 = dphi_Ins_match_f1 - dphi_Int_match_f1
    alpha0 = phi_Ins_match_f1 - phi_Int_match_f1 - alpha1 * f1_Ms

    def phi_Int_func(fM_s_, plan):
        return _evaluate_intermediate_phase(fM_s_, plan) + alpha1 * fM_s_ + alpha0

    phi_Int_match_f2, dphi_Int_match_f2 = _intermediate_phase_value_and_derivative(
        f2_Ms,
        scalar_intermediate_plan,
        correction=(alpha0, alpha1),
    )

    beta1 = dphi_Int_match_f2 - dphi_MRD_match_f2
    beta0 = phi_Int_match_f2 - phi_MRD_match_f2 - beta1 * f2_Ms

    phi_Int_corrected = phi_Int_func(fM_s, intermediate_plan)
    phi_MRD_corrected = phi_MRD + beta0 + beta1 * fM_s

    packed_masks = _maybe_packed_heaviside_masks(fM_s, f1_Ms, f2_Ms)
    if packed_masks is not None:
        inspiral_mask, intermediate_mask, intermediate_end, ringdown_mask, cutoff = (
            packed_masks
        )
        phase = (1 / eta) * (
            phi_Ins * inspiral_mask
            + intermediate_mask * phi_Int_corrected * intermediate_end
            + phi_MRD_corrected * ringdown_mask * cutoff
        )
        return phase

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
    *,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
    _phase_plan: _IMRPhenomXASPhasePlan | None = None,
    _cutoff_fMs=None,
    _request_proof=None,
) -> Float[Array, " n_freq"]:
    """
    Compute d Phase / d f for IMRPhenomXAS using the same piecewise construction
    as Phase(), but without differentiating through the final Heaviside assembly.
    """

    if _phase_plan is not None:
        return _evaluate_phase_derivative(
            f,
            _phase_plan,
            _request_proof=_request_proof,
        )

    m1, m2, chi1, chi2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    fM_s = f * M_s
    if _cutoff_fMs is None:
        fMs_RD, _, fMs_MECO, fMs_ISCO = _get_cutoff_fMs(
            m1,
            m2,
            chi1,
            chi2,
            chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
        )
    else:
        fMs_RD, _, fMs_MECO, fMs_ISCO = _cutoff_fMs
    fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
    fMs_INmatch = fMs_MECO
    deltafMs = (fMs_IMmatch - fMs_INmatch) * 0.03
    f1_Ms = fMs_INmatch - 1.0 * deltafMs
    f2_Ms = fMs_IMmatch + 0.5 * deltafMs

    fit_rows = None
    if not IMRPhenomX_utils._tree_has_autograd((theta, phase_coeffs)):
        fit_rows = _prepare_phase_fit_rows(theta, phase_coeffs)
    inspiral_plan = _prepare_inspiral_phase(
        theta,
        phase_coeffs,
        fit_rows,
        _cutoff_fMs=_cutoff_fMs,
    )
    scalar_inspiral_plan = _detach_phase_plan(inspiral_plan)
    phi_Ins_match_f1, dphi_Ins_match_f1 = _inspiral_phase_value_and_derivative(
        f1_Ms,
        scalar_inspiral_plan,
    )
    mergerringdown_plan = _prepare_mergerringdown_phase(
        theta,
        phase_coeffs,
        chip,
        fit_rows=fit_rows,
        final_spin=final_spin,
        coprecessing_deviations=coprecessing_deviations,
        _cutoff_fMs=_cutoff_fMs,
    )
    scalar_mergerringdown_plan = _detach_phase_plan(mergerringdown_plan)
    (_phi_MRD_match_f2, (cL, CV_phase_RD0)), dphi_MRD_match_f2 = (
        _mergerringdown_phase_value_and_derivative(
            f2_Ms,
            scalar_mergerringdown_plan,
        )
    )

    intermediate_plan = _prepare_intermediate_phase(
        theta,
        phase_coeffs,
        dphi_Ins_match_f1,
        CV_phase_RD0,
        cL,
        chip,
        fit_rows=fit_rows,
        final_spin=final_spin,
        coprecessing_deviations=coprecessing_deviations,
        _cutoff_fMs=_cutoff_fMs,
    )
    scalar_intermediate_plan = _detach_phase_plan(intermediate_plan)
    phi_Int_match_f1, dphi_Int_match_f1 = _intermediate_phase_value_and_derivative(
        f1_Ms,
        scalar_intermediate_plan,
    )
    alpha1 = dphi_Ins_match_f1 - dphi_Int_match_f1
    alpha0 = phi_Ins_match_f1 - phi_Int_match_f1 - alpha1 * f1_Ms

    def phi_Int_func(fM_s_, plan):
        return _evaluate_intermediate_phase(fM_s_, plan) + alpha1 * fM_s_ + alpha0

    _phi_Int_match_f2, dphi_Int_match_f2 = _intermediate_phase_value_and_derivative(
        f2_Ms,
        scalar_intermediate_plan,
        correction=(alpha0, alpha1),
    )
    beta1 = dphi_Int_match_f2 - dphi_MRD_match_f2

    dphi_Ins = _inspiral_phase_derivative(fM_s, scalar_inspiral_plan)
    dphi_Int = _intermediate_phase_derivative(
        fM_s,
        scalar_intermediate_plan,
        correction=(alpha0, alpha1),
    )
    dphi_MRD = (
        _mergerringdown_phase_derivative(
            fM_s,
            scalar_mergerringdown_plan,
        )
        + beta1
    )

    dphase_dMf = jnp.where(
        fM_s < f1_Ms,
        dphi_Ins / eta,
        jnp.where(fM_s < f2_Ms, dphi_Int / eta, dphi_MRD / eta),
    )
    return dphase_dMf * M_s


def _inspiral_amp_rhos_from_shared_powers(
    cp0,
    cp1,
    cp2,
    cv0,
    cv1,
    cv2,
    *,
    _host_powers=None,
):
    """Evaluate the legacy rho tree while sharing identical scalar powers."""

    if _host_powers is None:
        cp0_7o3 = cp0 ** (7.0 / 3.0)
        cp1_7o3 = cp1 ** (7.0 / 3.0)
        cp2_7o3 = cp2 ** (7.0 / 3.0)
        cp0_8o3 = cp0 ** (8.0 / 3.0)
        cp1_8o3 = cp1 ** (8.0 / 3.0)
        cp2_8o3 = cp2 ** (8.0 / 3.0)
        cp0_3 = cp0**3
        cp1_3 = cp1**3
        cp2_3 = cp2**3
        cp0_cbrt = jnp.cbrt(cp0)
        cp1_cbrt = jnp.cbrt(cp1)
        cp2_cbrt = jnp.cbrt(cp2)
    else:
        (
            (cp0_7o3, cp1_7o3, cp2_7o3),
            (cp0_8o3, cp1_8o3, cp2_8o3),
            (cp0_3, cp1_3, cp2_3),
            (cp0_cbrt, cp1_cbrt, cp2_cbrt),
        ) = _host_powers

    denominator = (
        cp0_7o3
        * (cp0_cbrt - cp1_cbrt)
        * cp1_7o3
        * (cp0_cbrt - cp2_cbrt)
        * (cp1_cbrt - cp2_cbrt)
        * cp2_7o3
    )
    rho1 = (
        -((cp1_8o3) * (cp2_3) * cv0)
        + cp1_3 * (cp2_8o3) * cv0
        + (cp0_8o3) * (cp2_3) * cv1
        - cp0_3 * (cp2_8o3) * cv1
        - (cp0_8o3) * (cp1_3) * cv2
        + cp0_3 * (cp1_8o3) * cv2
    ) / denominator
    rho2 = (
        (cp1_7o3) * (cp2_3) * cv0
        - cp1_3 * (cp2_7o3) * cv0
        - (cp0_7o3) * (cp2_3) * cv1
        + cp0_3 * (cp2_7o3) * cv1
        + (cp0_7o3) * (cp1_3) * cv2
        - cp0_3 * (cp1_7o3) * cv2
    ) / denominator
    rho3 = (
        (cp1_8o3) * (cp2_7o3) * cv0
        - (cp1_7o3) * (cp2_8o3) * cv0
        - (cp0_8o3) * (cp2_7o3) * cv1
        + (cp0_7o3) * (cp2_8o3) * cv1
        + (cp0_8o3) * (cp1_7o3) * cv2
        - (cp0_7o3) * (cp1_8o3) * cv2
    ) / denominator
    return rho1, rho2, rho3


def _prepare_inspiral_amp_eager(
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    chip: float = 0.0,
    *,
    fit_rows: torch.Tensor | None = None,
    _cutoff_fMs=None,
    _intrinsic_controls=None,
    _host_rho_powers=None,
) -> _InspiralAmpPlan:
    """Prepare the frequency-independent inspiral amplitude coefficients."""
    if _intrinsic_controls is None:
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
    else:
        m1 = _intrinsic_controls.mass1
        m2 = _intrinsic_controls.mass2
        chi1 = _intrinsic_controls.spin1
        chi2 = _intrinsic_controls.spin2
        eta = _intrinsic_controls.eta
        eta2 = eta * eta
        delta = _intrinsic_controls.delta
        S = _intrinsic_controls.inspiral_spin
        chia = _intrinsic_controls.spin_difference

    if _cutoff_fMs is None:
        _, _, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(
            m1,
            m2,
            chi1,
            chi2,
        )
    else:
        _, _, fMs_MECO, fMs_ISCO = _cutoff_fMs
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

    if fit_rows is None:
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
    else:
        CV_Amp_Ins0, CV_Amp_Ins1, CV_Amp_Ins2 = fit_rows[0:3]

    CP_Amp_Ins0 = 0.50 * fMs_AmpMatchIN
    CP_Amp_Ins1 = 0.75 * fMs_AmpMatchIN
    CP_Amp_Ins2 = 1.00 * fMs_AmpMatchIN

    if _host_rho_powers is not None:
        rho1, rho2, rho3 = _inspiral_amp_rhos_from_shared_powers(
            CP_Amp_Ins0,
            CP_Amp_Ins1,
            CP_Amp_Ins2,
            CV_Amp_Ins0,
            CV_Amp_Ins1,
            CV_Amp_Ins2,
            _host_powers=_host_rho_powers,
        )
        return _InspiralAmpPlan(
            A0,
            A2,
            A3,
            A4,
            A5,
            A6,
            rho1,
            rho2,
            rho3,
        )

    if _derived_scalar_powers_supported(
        CP_Amp_Ins0,
        CP_Amp_Ins1,
        CP_Amp_Ins2,
        CV_Amp_Ins0,
        CV_Amp_Ins1,
        CV_Amp_Ins2,
    ):
        rho1, rho2, rho3 = _inspiral_amp_rhos_from_shared_powers(
            CP_Amp_Ins0,
            CP_Amp_Ins1,
            CP_Amp_Ins2,
            CV_Amp_Ins0,
            CV_Amp_Ins1,
            CV_Amp_Ins2,
        )
        return _InspiralAmpPlan(
            A0,
            A2,
            A3,
            A4,
            A5,
            A6,
            rho1,
            rho2,
            rho3,
        )

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

    return _InspiralAmpPlan(
        A0,
        A2,
        A3,
        A4,
        A5,
        A6,
        rho1,
        rho2,
        rho3,
    )


def _prepare_inspiral_amp(
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    chip: float = 0.0,
    *,
    fit_rows: torch.Tensor | None = None,
    _cutoff_fMs=None,
    _intrinsic_controls=None,
    _host_scalars=None,
) -> _InspiralAmpPlan:
    """Prepare inspiral amplitude, using the guarded host lane if proven."""

    host_plan = _inspiral_amp_host_plan(
        theta,
        amp_coeffs,
        chip,
        fit_rows=fit_rows,
        _cutoff_fMs=_cutoff_fMs,
        _intrinsic_controls=_intrinsic_controls,
        _host_scalars=_host_scalars,
    )
    if host_plan is not None:
        return host_plan
    return _prepare_inspiral_amp_eager(
        theta,
        amp_coeffs,
        chip,
        fit_rows=fit_rows,
        _cutoff_fMs=_cutoff_fMs,
        _intrinsic_controls=_intrinsic_controls,
    )


def _evaluate_inspiral_amp(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    plan: _InspiralAmpPlan,
) -> Float[Array, " n_freq"] | FloatLike:
    """Evaluate a prepared plan with the eager expression tree unchanged."""

    return (
        plan.A0
        # A1 is missed since its zero
        + plan.A2 * (fM_s ** (2.0 / 3.0))
        + plan.A3 * fM_s
        + plan.A4 * (fM_s ** (4.0 / 3.0))
        + plan.A5 * (fM_s ** (5.0 / 3.0))
        + plan.A6 * (fM_s**2.0)
        # # Now we add the coefficient terms
        + plan.rho1 * (fM_s ** (7.0 / 3.0))
        + plan.rho2 * (fM_s ** (8.0 / 3.0))
        + plan.rho3 * (fM_s**3.0)
    )


def _exact_inspiral_amp_value_and_derivative(frequency, plan):
    """Replay Torch's scalar reverse pass without constructing an AD tape."""

    value = _evaluate_inspiral_amp(frequency, plan)
    contributions = (
        plan.A2 * ((2.0 / 3.0) * frequency.pow((2.0 / 3.0) - 1.0)),
        plan.A3,
        plan.A4 * ((4.0 / 3.0) * frequency.pow((4.0 / 3.0) - 1.0)),
        plan.A5 * ((5.0 / 3.0) * frequency.pow((5.0 / 3.0) - 1.0)),
        plan.A6 * (2.0 * frequency.pow(1.0)),
        plan.rho1 * ((7.0 / 3.0) * frequency.pow((7.0 / 3.0) - 1.0)),
        plan.rho2 * ((8.0 / 3.0) * frequency.pow((8.0 / 3.0) - 1.0)),
        plan.rho3 * (3.0 * frequency.pow(2.0)),
    )
    derivative = contributions[-1] + contributions[-2]
    for index in range(len(contributions) - 3, -1, -1):
        derivative = derivative + contributions[index]
    return value, derivative


def _exact_mergerringdown_amp_value_and_derivative(frequency, plan):
    """Replay Torch's scalar reverse pass without constructing an AD tape."""

    numerator_offset = frequency - plan.fMs_RD
    left_offset = frequency - plan.fMs_RD
    right_offset = frequency - plan.fMs_RD
    exponential = torch.exp((-numerator_offset) * plan.gammaR)
    numerator = exponential * plan.gammaD13
    denominator = left_offset * right_offset + plan.gammaD2
    value = numerator / denominator

    one = torch.ones_like(frequency)
    numerator_derivative = -(
        (((one / denominator) * plan.gammaD13) * exponential) * plan.gammaR
    )
    denominator_adjoint = (-(numerator / denominator)) / denominator
    denominator_factor_derivative = denominator_adjoint * left_offset
    derivative = (
        denominator_factor_derivative + denominator_factor_derivative
    ) + numerator_derivative
    return (value, plan.fMs_AmpRDMin), derivative


def get_inspiral_Amp(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    chip: float = 0.0,
    *,
    fit_rows: torch.Tensor | None = None,
    _amp_plan: _InspiralAmpPlan | None = None,
    _cutoff_fMs=None,
) -> Float[Array, " n_freq"] | FloatLike:
    """Evaluate the inspiral amplitude, optionally reusing a prepared plan."""

    if _amp_plan is None:
        _amp_plan = _prepare_inspiral_amp(
            theta,
            amp_coeffs,
            chip,
            fit_rows=fit_rows,
            _cutoff_fMs=_cutoff_fMs,
        )
    return _evaluate_inspiral_amp(fM_s, _amp_plan)


def _prepare_intermediate_amp(
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    fMs_AmpRDMin: FloatLike,
    chip: float = 0.0,
    *,
    fit_rows: torch.Tensor | None = None,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
    inspiral_plan: _InspiralAmpPlan | None = None,
    mergerringdown_plan: _MergerRingdownAmpPlan | None = None,
    _aligned_cutoff_fMs=None,
    _cutoff_fMs=None,
    _intrinsic_controls=None,
    _request_proof=None,
) -> _IntermediateAmpPlan:
    """Prepare the frequency-independent intermediate amplitude coefficients."""

    if _intrinsic_controls is None:
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
    else:
        m1 = _intrinsic_controls.mass1
        m2 = _intrinsic_controls.mass2
        chi1 = _intrinsic_controls.spin1
        chi2 = _intrinsic_controls.spin2
        eta = _intrinsic_controls.eta
        StotR = _intrinsic_controls.merger_spin
        chia = _intrinsic_controls.spin_difference

    # Now the intermediate region
    if _aligned_cutoff_fMs is None:
        _, _, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(
            m1,
            m2,
            chi1,
            chi2,
        )
    else:
        _, _, fMs_MECO, fMs_ISCO = _aligned_cutoff_fMs
    fMs_AmpInsMax = fMs_MECO + 0.25 * (fMs_ISCO - fMs_MECO)
    fMs_AmpMatchIN = fMs_AmpInsMax
    FMs1 = fMs_AmpMatchIN
    # This needs to come from outside
    FMs4 = fMs_AmpRDMin

    python_coefficients = None
    if (
        _request_amp_region_plan_supported(inspiral_plan, _request_proof)
        and _request_amp_region_plan_supported(
            mergerringdown_plan,
            _request_proof,
        )
    ) or _python_intermediate_amp_enabled():
        python_arguments = _python_intermediate_amp_executor_arguments(
            FMs1,
            FMs4,
            fit_rows,
            inspiral_plan,
            mergerringdown_plan,
            coprecessing_deviations,
            _request_proof=_request_proof,
        )
        if python_arguments is not None:
            executor = _get_python_intermediate_amp_executor()
            if executor is not None:
                python_coefficients = _execute_python_intermediate_amp(
                    executor,
                    python_arguments,
                )
                if (
                    python_coefficients is not None
                    and _PYTHON_INTERMEDIATE_AMP_CALIBRATED
                ):
                    return _IntermediateAmpPlan(*python_coefficients)

    if inspiral_plan is None:
        inspiral_plan = _prepare_inspiral_amp(
            theta,
            amp_coeffs,
            chip,
            fit_rows=fit_rows,
            _cutoff_fMs=_aligned_cutoff_fMs,
        )
    if _exact_scalar_amp_derivative_supported(
        FMs1,
        inspiral_plan,
        _request_proof=_request_proof,
    ):
        inspFMs1, d1 = _exact_inspiral_amp_value_and_derivative(
            FMs1,
            inspiral_plan,
        )
    else:
        inspFMs1, d1 = jax.value_and_grad(_evaluate_inspiral_amp)(
            FMs1,
            inspiral_plan,
        )
    if mergerringdown_plan is None:
        mergerringdown_plan = _prepare_mergerringdown_amp_plan(
            theta,
            amp_coeffs,
            chip,
            fit_rows=fit_rows,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
            _cutoff_fMs=_cutoff_fMs,
        )
    if _exact_scalar_amp_derivative_supported(
        FMs4,
        mergerringdown_plan,
        _request_proof=_request_proof,
    ):
        rdFMs4, d4 = _exact_mergerringdown_amp_value_and_derivative(
            FMs4,
            mergerringdown_plan,
        )
    else:
        rdFMs4, d4 = jax.value_and_grad(
            _evaluate_mergerringdown_amp,
            has_aux=True,
        )(
            FMs4,
            mergerringdown_plan,
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

    if fit_rows is None:
        V2 = (
            IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[3, 0:amp_eqspin_indx], eta)
            + IMRPhenomX_utils.Amp_Eqspin_CV(
                amp_coeffs[3, amp_eqspin_indx:amp_uneqspin_indx], eta, StotR
            )
            + IMRPhenomX_utils.Amp_Uneqspin_CV(
                amp_coeffs[3, amp_uneqspin_indx:], eta, StotR, chia
            )
        )
    else:
        V2 = fit_rows[3]
    V4 = (FMs4 ** (-7.0 / 6)) * rdFMs4

    V1 = 1.0 / V1
    V2 = 1.0 / V2
    V4 = 1.0 / V4
    V2 = V2 + _coprecessing_deviation(coprecessing_deviations, "mu3", like=V2)

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

    eager_coefficients = _IntermediateAmpPlan(
        delta0,
        delta1,
        delta2,
        delta3,
        delta4,
    )
    if python_coefficients is not None:
        if _python_intermediate_amp_raw_equal(
            python_coefficients,
            eager_coefficients,
        ):
            _mark_python_intermediate_amp_calibrated()
        else:
            _mark_python_intermediate_amp_failed()
    return eager_coefficients


def _evaluate_intermediate_amp(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    plan: _IntermediateAmpPlan,
) -> Float[Array, " n_freq"] | FloatLike:
    """Evaluate a prepared plan with the eager expression tree unchanged."""

    return (fM_s ** (7.0 / 6.0)) / (
        plan.delta0
        + fM_s
        * (
            plan.delta1
            + fM_s * (plan.delta2 + fM_s * (plan.delta3 + fM_s * plan.delta4))
        )
    )


def get_intermediate_Amp(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    fMs_AmpRDMin: FloatLike,
    chip: float = 0.0,
    *,
    fit_rows: torch.Tensor | None = None,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
    _amp_plan: _IntermediateAmpPlan | None = None,
    _inspiral_plan: _InspiralAmpPlan | None = None,
    _mergerringdown_plan: _MergerRingdownAmpPlan | None = None,
    _aligned_cutoff_fMs=None,
    _cutoff_fMs=None,
) -> Float[Array, " n_freq"] | FloatLike:
    """Evaluate intermediate amplitude, optionally reusing prepared plans."""

    if _amp_plan is None:
        _amp_plan = _prepare_intermediate_amp(
            theta,
            amp_coeffs,
            fMs_AmpRDMin,
            chip,
            fit_rows=fit_rows,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
            inspiral_plan=_inspiral_plan,
            mergerringdown_plan=_mergerringdown_plan,
            _aligned_cutoff_fMs=_aligned_cutoff_fMs,
            _cutoff_fMs=_cutoff_fMs,
        )
    return _evaluate_intermediate_amp(fM_s, _amp_plan)


def _prepare_mergerringdown_amp_plan(
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    chip: FloatLike = 0.0,
    *,
    fit_rows: torch.Tensor | None = None,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
    _cutoff_fMs=None,
    _intrinsic_controls=None,
) -> _MergerRingdownAmpPlan:
    """Prepare one request's exact merger-ringdown amplitude constants."""

    if _intrinsic_controls is None:
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
    else:
        m1 = _intrinsic_controls.mass1
        m2 = _intrinsic_controls.mass2
        chi1 = _intrinsic_controls.spin1
        chi2 = _intrinsic_controls.spin2
        eta = _intrinsic_controls.eta
        StotR = _intrinsic_controls.merger_spin
        chia = _intrinsic_controls.spin_difference

    if _cutoff_fMs is None:
        fMs_RD, fMs_damp, _, _ = _get_cutoff_fMs(
            m1,
            m2,
            chi1,
            chi2,
            chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
        )
    else:
        fMs_RD, fMs_damp, _, _ = _cutoff_fMs

    if fit_rows is None:
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
        v1RD = (
            IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[6, 0:amp_eqspin_indx], eta)
            + IMRPhenomX_utils.Amp_Eqspin_CV(
                amp_coeffs[6, amp_eqspin_indx:amp_uneqspin_indx], eta, StotR
            )
            + IMRPhenomX_utils.Amp_Uneqspin_CV(
                amp_coeffs[6, amp_uneqspin_indx:], eta, StotR, chia
            )
        )
    else:
        gamma2, gamma3, v1RD = fit_rows[4:7]

    gamma3 = gamma3 + _coprecessing_deviation(
        coprecessing_deviations, "mu2", like=gamma3
    )
    fMs_AmpRDMin = jnp.where(
        gamma2 <= 1.0,
        jnp.fabs(
            fMs_RD
            + fMs_damp * gamma3 * (jnp.sqrt(1.0 - gamma2 * gamma2) - 1.0) / gamma2
        ),
        jnp.fabs(fMs_RD + fMs_damp * (-1.0) * gamma3 / gamma2),
    )
    v1RD = v1RD + _coprecessing_deviation(coprecessing_deviations, "mu1", like=v1RD)
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

    return _MergerRingdownAmpPlan(
        fMs_RD,
        gammaR,
        gammaD2,
        gammaD13,
        fMs_AmpRDMin,
    )


def _evaluate_mergerringdown_amp(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    plan: _MergerRingdownAmpPlan,
) -> tuple[Float[Array, " n_freq"], FloatLike]:
    """Evaluate a prepared plan with the eager expression tree unchanged."""

    Amp_RD = (
        jnp.exp(-(fM_s - plan.fMs_RD) * plan.gammaR)
        * (plan.gammaD13)
        / ((fM_s - plan.fMs_RD) * (fM_s - plan.fMs_RD) + plan.gammaD2)
    )
    return Amp_RD, plan.fMs_AmpRDMin


def get_mergerringdown_Amp(
    fM_s: Float[Array, " n_freq"] | FloatLike,
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    chip: FloatLike = 0.0,
    *,
    fit_rows: torch.Tensor | None = None,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
    _amp_plan: _MergerRingdownAmpPlan | None = None,
    _cutoff_fMs=None,
) -> tuple[Float[Array, " n_freq"], FloatLike]:
    if _amp_plan is not None:
        return _evaluate_mergerringdown_amp(fM_s, _amp_plan)

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

    if _cutoff_fMs is None:
        fMs_RD, fMs_damp, _, _ = _get_cutoff_fMs(
            m1,
            m2,
            chi1,
            chi2,
            chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
        )
    else:
        fMs_RD, fMs_damp, _, _ = _cutoff_fMs

    if fit_rows is None:
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
        v1RD = (
            IMRPhenomX_utils.Amp_Nospin_CV(amp_coeffs[6, 0:amp_eqspin_indx], eta)
            + IMRPhenomX_utils.Amp_Eqspin_CV(
                amp_coeffs[6, amp_eqspin_indx:amp_uneqspin_indx], eta, StotR
            )
            + IMRPhenomX_utils.Amp_Uneqspin_CV(
                amp_coeffs[6, amp_uneqspin_indx:], eta, StotR, chia
            )
        )
    else:
        gamma2, gamma3, v1RD = fit_rows[4:7]

    gamma3 = gamma3 + _coprecessing_deviation(
        coprecessing_deviations, "mu2", like=gamma3
    )
    fMs_AmpRDMin = jnp.where(
        gamma2 <= 1.0,
        jnp.fabs(
            fMs_RD
            + fMs_damp * gamma3 * (jnp.sqrt(1.0 - gamma2 * gamma2) - 1.0) / gamma2
        ),
        jnp.fabs(fMs_RD + fMs_damp * (-1.0) * gamma3 / gamma2),
    )
    v1RD = v1RD + _coprecessing_deviation(coprecessing_deviations, "mu1", like=v1RD)
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


def _prepare_amp_plan(
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    chip: FloatLike = 0.0,
    *,
    fit_rows: torch.Tensor,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
    _aligned_cutoff_fMs=None,
    _cutoff_fMs=None,
    _intrinsic_controls=None,
    _request_proof=None,
) -> _IMRPhenomXASAmpPlan:
    """Prepare each amplitude region once for a non-autograd request."""

    inspiral_host_scalars = _maybe_inspiral_amp_host_scalars(
        theta,
        amp_coeffs,
        chip,
        final_spin,
        coprecessing_deviations,
        fit_rows,
        _aligned_cutoff_fMs,
        _intrinsic_controls,
    )
    inspiral = _prepare_inspiral_amp(
        theta,
        amp_coeffs,
        chip,
        fit_rows=fit_rows,
        _cutoff_fMs=_aligned_cutoff_fMs,
        _intrinsic_controls=_intrinsic_controls,
        _host_scalars=inspiral_host_scalars,
    )
    mergerringdown = _prepare_mergerringdown_amp_plan(
        theta,
        amp_coeffs,
        chip,
        fit_rows=fit_rows,
        final_spin=final_spin,
        coprecessing_deviations=coprecessing_deviations,
        _cutoff_fMs=_cutoff_fMs,
        _intrinsic_controls=_intrinsic_controls,
    )
    request_regions = _request_qualify_amp_regions(
        inspiral,
        mergerringdown,
        _request_proof,
    )
    if request_regions is not None:
        inspiral, mergerringdown = request_regions
    intermediate = _prepare_intermediate_amp(
        theta,
        amp_coeffs,
        mergerringdown.fMs_AmpRDMin,
        chip,
        fit_rows=fit_rows,
        final_spin=final_spin,
        coprecessing_deviations=coprecessing_deviations,
        inspiral_plan=inspiral,
        mergerringdown_plan=mergerringdown,
        _aligned_cutoff_fMs=_aligned_cutoff_fMs,
        _cutoff_fMs=_cutoff_fMs,
        _intrinsic_controls=_intrinsic_controls,
        _request_proof=_request_proof,
    )
    plan = _IMRPhenomXASAmpPlan(inspiral, intermediate, mergerringdown)
    return _request_qualify_top_plan(plan, _request_proof)


def _evaluate_pruned_amp(frequency, plan, indices):
    """Evaluate only the active amplitude spans with unchanged formulas."""

    first, second, cutoff = indices
    amplitude = torch.zeros_like(frequency)
    amplitude[:first] = _evaluate_aligned_region(
        frequency,
        0,
        first,
        lambda value: _evaluate_inspiral_amp(value, plan.inspiral),
    )
    amplitude[first:second] = _evaluate_aligned_region(
        frequency,
        first,
        second,
        lambda value: _evaluate_intermediate_amp(value, plan.intermediate),
    )
    amplitude[second:cutoff] = _evaluate_aligned_region(
        frequency,
        second,
        cutoff,
        lambda value: _evaluate_mergerringdown_amp(
            value,
            plan.mergerringdown,
        )[0],
    )
    return amplitude


def Amp(
    f: Float[Array, " n_freq"],
    theta: Float[Array, "4"],
    amp_coeffs: Float[Array, "7 42"],
    D: FloatLike = 1.0,
    chip: float = 0.0,
    *,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
    _amp_fit_rows: torch.Tensor | None = None,
    _return_amp_plan: bool = False,
    _aligned_cutoff_fMs=None,
    _cutoff_fMs=None,
    _prepared_amp_plan=None,
    _request_proof=None,
) -> (
    Float[Array, " n_freq"]
    | tuple[Float[Array, " n_freq"], _IMRPhenomXASAmpPlan | None]
):
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)

    fM_s = f * M_s
    if _aligned_cutoff_fMs is None:
        _, _, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(
            m1,
            m2,
            chi1,
            chi2,
        )
    else:
        _, _, fMs_MECO, fMs_ISCO = _aligned_cutoff_fMs
    amp0 = 2.0 * jnp.sqrt(5.0 / (64.0 * PI)) * M_s**2 / ((D * MPC) / C)
    ampNorm = jnp.sqrt(2.0 * eta / 3.0) * (PI ** (-1.0 / 6.0))

    fMs_AmpInsMax = fMs_MECO + 0.25 * (fMs_ISCO - fMs_MECO)
    fMs_AmpMatchIN = fMs_AmpInsMax

    # Below
    Overallamp = amp0 * ampNorm

    amp_plan = (
        _prepared_amp_plan
        if (
            type(_prepared_amp_plan) is _IMRPhenomXASAmpPlan
            and not _request_proof_plan_current(_request_proof)
        )
        else None
    )
    fit_rows = None
    if amp_plan is None:
        fit_rows = (
            _amp_fit_rows
            if _precomputed_amp_fit_rows_supported(_amp_fit_rows, theta)
            else None
        )
        if fit_rows is None and not IMRPhenomX_utils._tree_has_autograd(
            (theta, amp_coeffs, _amp_fit_rows)
        ):
            fit_rows = _prepare_amp_fit_rows(theta, amp_coeffs)

    reuse_amp_plan = amp_plan is not None or (
        _request_proof_amplitude_ready(_request_proof)
        or (
            _amp_plan_enabled()
            and _amp_plan_inputs_supported(
                f,
                theta,
                amp_coeffs,
                (
                    f,
                    theta,
                    amp_coeffs,
                    D,
                    chip,
                    final_spin,
                    coprecessing_deviations,
                ),
            )
        )
    )
    if reuse_amp_plan:
        if amp_plan is None:
            amp_plan = _prepare_amp_plan(
                theta,
                amp_coeffs,
                chip,
                fit_rows=fit_rows,
                final_spin=final_spin,
                coprecessing_deviations=coprecessing_deviations,
                _aligned_cutoff_fMs=_aligned_cutoff_fMs,
                _cutoff_fMs=_cutoff_fMs,
                _request_proof=_request_proof,
            )
        fMs_AmpRDMin = amp_plan.mergerringdown.fMs_AmpRDMin
        region_indices = _piecewise_region_indices(
            f,
            fM_s,
            fMs_AmpMatchIN,
            fMs_AmpRDMin,
            amp_plan,
            _request_proof=_request_proof,
        )
        if region_indices is not None:
            amplitude = _evaluate_pruned_amp(fM_s, amp_plan, region_indices)
            result = Overallamp * amplitude * (fM_s ** (-7.0 / 6.0))
            return (result, amp_plan) if _return_amp_plan else result

        Amp_Ins = _evaluate_inspiral_amp(fM_s, amp_plan.inspiral)
        Amp_RD, _ = _evaluate_mergerringdown_amp(
            fM_s,
            amp_plan.mergerringdown,
        )
        Amp_Int = _evaluate_intermediate_amp(fM_s, amp_plan.intermediate)
    else:
        Amp_Ins = get_inspiral_Amp(
            fM_s,
            theta,
            amp_coeffs,
            chip,
            fit_rows=fit_rows,
            _cutoff_fMs=_aligned_cutoff_fMs,
        )
        Amp_RD, fMs_AmpRDMin = get_mergerringdown_Amp(
            fM_s,
            theta,
            amp_coeffs,
            chip,
            fit_rows=fit_rows,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
            _cutoff_fMs=_cutoff_fMs,
        )
        Amp_Int = get_intermediate_Amp(
            fM_s,
            theta,
            amp_coeffs,
            fMs_AmpRDMin,
            chip,
            fit_rows=fit_rows,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
            _aligned_cutoff_fMs=_aligned_cutoff_fMs,
            _cutoff_fMs=_cutoff_fMs,
        )

    packed_masks = _maybe_packed_heaviside_masks(
        fM_s,
        fMs_AmpMatchIN,
        fMs_AmpRDMin,
        _request_plan=amp_plan,
        _request_proof=_request_proof,
    )
    if packed_masks is not None:
        inspiral_mask, intermediate_mask, intermediate_end, ringdown_mask, cutoff = (
            packed_masks
        )
        Amp = (
            Amp_Ins * inspiral_mask
            + intermediate_mask * Amp_Int * intermediate_end
            + Amp_RD * ringdown_mask * cutoff
        )
        result = Overallamp * Amp * (fM_s ** (-7.0 / 6.0))
        return (result, amp_plan) if _return_amp_plan else result

    Amp = (
        Amp_Ins * jnp.heaviside(fMs_AmpMatchIN - fM_s, 0.5)
        + jnp.heaviside(fM_s - fMs_AmpMatchIN, 0.5)
        * Amp_Int
        * jnp.heaviside(fMs_AmpRDMin - fM_s, 0.5)
        + Amp_RD
        * jnp.heaviside(fM_s - fMs_AmpRDMin, 0.5)
        * jnp.heaviside(IMRPhenomX_utils.fM_CUT - fM_s, 0.5)
    )

    result = Overallamp * Amp * (fM_s ** (-7.0 / 6.0))
    return (result, amp_plan) if _return_amp_plan else result


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
            params.alignment_frequency * (params.mass1 + params.mass2) * MTSUN,
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


def _carrier_alignment_result_tensor_supported(value, *, shape):
    """Accept one owned, non-AD binary64 CPU tensor from this XAS call."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.shape == shape
        and value.dtype == torch.float64
        and value.device == torch.device("cpu")
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and value._version == 0
        and not value.is_conj()
        and not value.is_neg()
        and not value.requires_grad
        and value.grad_fn is None
        and not IMRPhenomX_utils._tensor_has_forward_ad(value)
    )


def _carrier_alignment_phase_scalar_supported(value):
    """Accept only ``theta_extrinsic[2]`` from one owned XAS input tensor."""

    if (
        type(value) is not torch.Tensor
        or value.layout is not torch.strided
        or value.shape != torch.Size(())
        or value.dtype != torch.float64
        or value.device != torch.device("cpu")
        or value.storage_offset() != 2
        or value.is_conj()
        or value.is_neg()
        or value.requires_grad
        or value.grad_fn is not None
        or IMRPhenomX_utils._tensor_has_forward_ad(value)
    ):
        return False
    base = value._base
    return _carrier_alignment_result_tensor_supported(
        base,
        shape=torch.Size((3,)),
    )


def _carrier_alignment_result_inputs_supported(
    theta_intrinsic,
    phase_coeffs,
    f_ref,
    carrier_phase,
    chip,
    final_spin,
    coprecessing_deviations,
    phase_plan,
    cutoff_fMs,
    request_proof,
):
    """Recognize the exact plain XPHM carrier-alignment input schema."""

    return (
        request_proof is None
        and coprecessing_deviations is None
        and type(f_ref) is float
        and math.isfinite(f_ref)
        and _carrier_alignment_phase_scalar_supported(carrier_phase)
        and type(chip) is float
        and math.isfinite(chip)
        and type(final_spin) is float
        and math.isfinite(final_spin)
        and _carrier_alignment_result_tensor_supported(
            theta_intrinsic,
            shape=torch.Size((4,)),
        )
        and _carrier_alignment_result_tensor_supported(
            phase_coeffs,
            shape=torch.Size((13, 49)),
        )
        and _imrphenomxas_phase_plan_type_supported(phase_plan)
        and type(cutoff_fMs) is tuple
        and len(cutoff_fMs) == 4
    )


def _phase_alignment_terms_with_result(
    theta_intrinsic,
    phase_coeffs,
    f_ref,
    carrier_phase,
    *,
    chip,
    final_spin,
    coprecessing_deviations,
    _phase_plan,
    _cutoff_fMs,
    _request_proof,
):
    """Evaluate the eager alignment tree and return two exact subresults."""

    # Keep this arithmetic tree in lockstep with the ordinary implementation
    # below.  Only the two existing subresults are named before their original
    # consumers use them; no Torch operation is inserted, removed, or moved.
    m1, m2, chi1, chi2 = theta_intrinsic
    total_mass_seconds = (m1 + m2) * MTSUN
    eta = m1 * m2 / ((m1 + m2) ** 2.0)
    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
    mass_fraction1 = 0.5 * (1.0 + delta)
    mass_fraction2 = 0.5 * (1.0 - delta)
    total_spin = (mass_fraction1**2 * chi1 + mass_fraction2**2 * chi2) / (
        mass_fraction1**2 + mass_fraction2**2
    )
    spin_difference = chi1 - chi2
    ringdown_frequency, damping_frequency, _, _ = _cutoff_fMs
    linear_a, linear_b, psi4_to_strain = IMRPhenomX_utils.calc_phaseatpeak(
        eta,
        total_spin,
        spin_difference,
        delta,
    )
    ringdown_start_derivative = PhaseDerivative(
        (ringdown_frequency - damping_frequency) / total_mass_seconds,
        theta_intrinsic,
        phase_coeffs,
        chip,
        final_spin=final_spin,
        coprecessing_deviations=coprecessing_deviations,
        _phase_plan=_phase_plan,
        _cutoff_fMs=_cutoff_fMs,
        _request_proof=_request_proof,
    )
    phase_derivative = ringdown_start_derivative / total_mass_seconds
    linear_b = linear_b - phase_derivative - 2.0 * PI * (500.0 + psi4_to_strain)
    reference_phase = Phase(
        f_ref,
        theta_intrinsic,
        phase_coeffs,
        chip,
        final_spin=final_spin,
        coprecessing_deviations=coprecessing_deviations,
        _phase_plan=_phase_plan,
        _cutoff_fMs=_cutoff_fMs,
        _request_proof=_request_proof,
    )
    phase_offset = (
        -(
            reference_phase
            + linear_b * (f_ref * total_mass_seconds)
            + linear_a
        )
        + 2.0 * carrier_phase
        + PI / 4.0
    )
    carrier_alignment_result = None
    if (
        _carrier_alignment_result_tensor_supported(
            reference_phase,
            shape=torch.Size(()),
        )
        and _carrier_alignment_result_tensor_supported(
            ringdown_start_derivative,
            shape=torch.Size(()),
        )
    ):
        carrier_alignment_result = _CarrierAlignmentResult(
            _phase_plan,
            f_ref,
            reference_phase,
            ringdown_start_derivative,
        )
    return linear_a, linear_b, phase_offset, carrier_alignment_result


def _phase_alignment_terms(
    theta_intrinsic,
    phase_coeffs,
    f_ref,
    carrier_phase=0.0,
    *,
    chip=0.0,
    final_spin=None,
    coprecessing_deviations=None,
    _phase_plan=None,
    _cutoff_fMs=None,
    _request_proof=None,
    _return_carrier_alignment_result=False,
):
    """Return the constant and linear terms that align the XAS phase."""

    if _return_carrier_alignment_result and (
        _carrier_alignment_result_inputs_supported(
            theta_intrinsic,
            phase_coeffs,
            f_ref,
            carrier_phase,
            chip,
            final_spin,
            coprecessing_deviations,
            _phase_plan,
            _cutoff_fMs,
            _request_proof,
        )
    ):
        return _phase_alignment_terms_with_result(
            theta_intrinsic,
            phase_coeffs,
            f_ref,
            carrier_phase,
            chip=chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
            _phase_plan=_phase_plan,
            _cutoff_fMs=_cutoff_fMs,
            _request_proof=_request_proof,
        )

    m1, m2, chi1, chi2 = theta_intrinsic
    total_mass_seconds = (m1 + m2) * MTSUN
    eta = m1 * m2 / ((m1 + m2) ** 2.0)
    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
    mass_fraction1 = 0.5 * (1.0 + delta)
    mass_fraction2 = 0.5 * (1.0 - delta)
    total_spin = (mass_fraction1**2 * chi1 + mass_fraction2**2 * chi2) / (
        mass_fraction1**2 + mass_fraction2**2
    )
    spin_difference = chi1 - chi2
    if _cutoff_fMs is None:
        ringdown_frequency, damping_frequency, _, _ = _get_cutoff_fMs(
            m1,
            m2,
            chi1,
            chi2,
            chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
        )
    else:
        ringdown_frequency, damping_frequency, _, _ = _cutoff_fMs
    linear_a, linear_b, psi4_to_strain = IMRPhenomX_utils.calc_phaseatpeak(
        eta,
        total_spin,
        spin_difference,
        delta,
    )
    phase_derivative = (
        PhaseDerivative(
            (ringdown_frequency - damping_frequency) / total_mass_seconds,
            theta_intrinsic,
            phase_coeffs,
            chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
            _phase_plan=_phase_plan,
            _cutoff_fMs=_cutoff_fMs,
            _request_proof=_request_proof,
        )
        / total_mass_seconds
    )
    linear_b = linear_b - phase_derivative - 2.0 * PI * (500.0 + psi4_to_strain)
    phase_offset = (
        -(
            Phase(
                f_ref,
                theta_intrinsic,
                phase_coeffs,
                chip,
                final_spin=final_spin,
                coprecessing_deviations=coprecessing_deviations,
                _phase_plan=_phase_plan,
                _cutoff_fMs=_cutoff_fMs,
                _request_proof=_request_proof,
            )
            + linear_b * (f_ref * total_mass_seconds)
            + linear_a
        )
        + 2.0 * carrier_phase
        + PI / 4.0
    )
    if _return_carrier_alignment_result:
        return linear_a, linear_b, phase_offset, None
    return linear_a, linear_b, phase_offset


def _packed_frequency_plan_plain_tensor(value, shape, *, device=None) -> bool:
    """Accept one owned, plain binary64 tensor in the fixed XAS schema."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.device.type in ("cpu", "cuda")
        and (device is None or value.device == device)
        and value.dtype is torch.float64
        and value.shape == shape
        and value.numel() != 0
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and not value.is_conj()
        and not value.is_neg()
    )


def _packed_frequency_plan_supported(
    frequency,
    theta_intrinsic,
    theta_extrinsic,
    phase_coeffs,
    amp_coeffs,
    f_ref,
) -> bool:
    """Keep the packed lane on its raw-qualified plain XAS contract."""

    if not _packed_heaviside_masks_runtime_supported():
        return False
    if not _packed_frequency_plan_plain_tensor(
        frequency,
        frequency.shape,
    ):
        return False
    if frequency.ndim != 1:
        return False
    device = frequency.device
    tensors = (
        (theta_intrinsic, (4,)),
        (theta_extrinsic, (3,)),
        (phase_coeffs, (13, 49)),
        (amp_coeffs, (7, 42)),
    )
    if not all(
        _packed_frequency_plan_plain_tensor(value, shape, device=device)
        for value, shape in tensors
    ):
        return False
    if type(f_ref) is not float or not math.isfinite(f_ref):
        return False
    return not IMRPhenomX_utils._tree_has_autograd_untrusted(
        (
            frequency,
            theta_intrinsic,
            theta_extrinsic,
            phase_coeffs,
            amp_coeffs,
        )
    )


def _intrinsic_control_reuse_supported(theta) -> bool:
    """Accept the exact request shape qualified for shared eager controls."""

    return (
        torch.is_grad_enabled()
        and not torch.is_inference_mode_enabled()
        and _packed_heaviside_masks_runtime_supported()
        and _packed_frequency_plan_plain_tensor(theta, (4,))
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(theta)
    )


def _build_intrinsic_controls(theta, *, host_values):
    """Evaluate the existing full intrinsic-control DAG exactly once."""

    mass1, mass2, spin1, spin2 = theta
    mass1_seconds = mass1 * MTSUN
    mass2_seconds = mass2 * MTSUN
    total_mass_seconds = mass1_seconds + mass2_seconds
    eta = mass1_seconds * mass2_seconds / (total_mass_seconds**2.0)
    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
    mass_fraction1 = 0.5 * (1.0 + delta)
    mass_fraction2 = 0.5 * (1.0 - delta)
    effective_spin = mass_fraction1 * spin1 + mass_fraction2 * spin2
    inspiral_spin = (
        effective_spin - (38.0 / 113.0) * eta * (spin1 + spin2)
    ) / (1.0 - (76.0 * eta / 113.0))
    merger_spin = (
        mass_fraction1**2 * spin1 + mass_fraction2**2 * spin2
    ) / (mass_fraction1**2 + mass_fraction2**2)
    spin_difference = spin1 - spin2

    theta_values = None
    fit_values = None
    if host_values:
        values = tuple(
            torch.stack(
                (
                    mass1,
                    mass2,
                    spin1,
                    spin2,
                    eta,
                    inspiral_spin,
                    merger_spin,
                    spin_difference,
                    delta,
                )
            ).tolist()
        )
        theta_values = values[0:4]
        fit_values = values[4:9]
    return _IMRPhenomXASIntrinsicControls(
        mass1,
        mass2,
        spin1,
        spin2,
        mass1_seconds,
        mass2_seconds,
        total_mass_seconds,
        eta,
        delta,
        mass_fraction1,
        mass_fraction2,
        effective_spin,
        inspiral_spin,
        merger_spin,
        spin_difference,
        theta_values,
        fit_values,
    )


def _intrinsic_controls_valid(controls, theta, *, host_values) -> bool:
    """Validate a producer result before any consumer may trust its leaves."""

    if type(controls) is not _IMRPhenomXASIntrinsicControls:
        return False
    tensor_leaves = controls[:15]
    if not all(
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.device == theta.device
        and value.dtype is theta.dtype
        and value.shape == ()
        and value.is_contiguous()
        and not value.is_conj()
        and not value.is_neg()
        for value in tensor_leaves
    ):
        return False
    if host_values:
        return (
            type(controls.theta_values) is tuple
            and len(controls.theta_values) == 4
            and all(type(value) is float for value in controls.theta_values)
            and type(controls.fit_values) is tuple
            and len(controls.fit_values) == 5
            and all(type(value) is float for value in controls.fit_values)
        )
    return controls.theta_values is None and controls.fit_values is None


def _inspiral_phase_host_plain_tensor(value, shape, *, device) -> bool:
    """Accept one plain owned CPU binary64 tensor in the qualified schema."""

    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.device == device
        and value.dtype is torch.float64
        and value.shape == shape
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value._base is None
        and value._version == 0
        and not value.is_conj()
        and not value.is_neg()
    )


def _inspiral_phase_host_controls_have_provenance(
    controls,
    theta,
) -> bool:
    """Verify that host values came from this exact intrinsic tensor."""

    return _intrinsic_controls_valid(
        controls,
        theta,
        host_values=True,
    ) and all(
        type(value) is torch.Tensor
        and value._base is theta
        and value.storage_offset() == index
        for index, value in enumerate(controls[0:4])
    )


def _inspiral_phase_host_values_supported(controls) -> bool:
    """Reject nonfinite and nonphysical host inputs before scalar replay."""

    values = (*controls.theta_values, *controls.fit_values)
    return (
        all(type(value) is float and math.isfinite(value) for value in values)
        and controls.theta_values[0] > 0.0
        and controls.theta_values[1] > 0.0
    )


def _maybe_inspiral_phase_host_scalars(
    theta,
    phase_coeffs,
    chip,
    final_spin,
    coprecessing_deviations,
    phase_fit_rows,
    cutoff_fMs,
    intrinsic_controls,
):
    """Build one request-local host provenance pack, or fail closed."""

    cpu = torch.device("cpu")
    if (
        not _inspiral_phase_host_scalars_enabled()
        or not torch.is_grad_enabled()
        or torch.is_inference_mode_enabled()
        or not _packed_heaviside_masks_runtime_supported()
        or _phase_plan_bulk_collocation_enabled()
        or not callable(getattr(torch, "unbind_copy", None))
        or not _inspiral_phase_host_plain_tensor(
            theta,
            (4,),
            device=cpu,
        )
        or not _inspiral_phase_host_plain_tensor(
            phase_coeffs,
            (13, 49),
            device=cpu,
        )
        or phase_coeffs
        is not IMRPhenomX_utils._PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER
        or not _inspiral_phase_host_plain_tensor(
            phase_fit_rows,
            (13,),
            device=cpu,
        )
        or type(cutoff_fMs) is not tuple
        or len(cutoff_fMs) != 4
        or not _inspiral_phase_host_plain_tensor(
            cutoff_fMs[2],
            (),
            device=cpu,
        )
        or type(chip) not in (int, float)
        or not math.isfinite(chip)
        or IMRPhenomX_utils._tree_has_autograd_untrusted(
            (
                theta,
                phase_coeffs,
                chip,
                final_spin,
                coprecessing_deviations,
                phase_fit_rows,
                cutoff_fMs,
                intrinsic_controls,
            )
        )
    ):
        return None

    controls = intrinsic_controls
    if controls is None:
        try:
            controls = _build_intrinsic_controls(theta, host_values=True)
        except Exception:
            return None
    if (
        not _inspiral_phase_host_controls_have_provenance(controls, theta)
        or not _inspiral_phase_host_values_supported(controls)
    ):
        return None
    try:
        builder_values = tuple(
            torch.cat(
                (
                    phase_fit_rows[0:4],
                    torch.stack((jnp.log(2.0), jnp.log(PI))),
                )
            ).tolist()
        )
    except Exception:
        return None
    if (
        len(builder_values) != 6
        or not all(
            type(value) is float and math.isfinite(value)
            for value in builder_values
        )
    ):
        return None
    return _IMRPhenomXASInspiralPhaseHostScalars(
        theta,
        phase_coeffs,
        phase_fit_rows,
        cutoff_fMs[2],
        controls,
        builder_values,
    )


def _inspiral_phase_host_execution_supported(
    theta,
    phase_coeffs,
    fit_rows,
    *,
    bulk_collocation,
    _solve_info,
    _cutoff_fMs,
    _intrinsic_controls,
    _host_scalars,
) -> bool:
    """Revalidate the complete fixed schema at its arithmetic consumer."""

    if type(_host_scalars) is not _IMRPhenomXASInspiralPhaseHostScalars:
        return False
    controls = _host_scalars.intrinsic_controls
    cpu = torch.device("cpu")
    return (
        _inspiral_phase_host_scalars_enabled()
        and torch.is_grad_enabled()
        and not torch.is_inference_mode_enabled()
        and _packed_heaviside_masks_runtime_supported()
        and not _phase_plan_bulk_collocation_enabled()
        and callable(getattr(torch, "unbind_copy", None))
        and bulk_collocation is False
        and _solve_info is None
        and _host_scalars.theta is theta
        and _host_scalars.phase_coeffs is phase_coeffs
        and _host_scalars.phase_fit_rows is fit_rows
        and controls is _intrinsic_controls
        and _inspiral_phase_host_plain_tensor(theta, (4,), device=cpu)
        and _inspiral_phase_host_plain_tensor(
            phase_coeffs,
            (13, 49),
            device=cpu,
        )
        and phase_coeffs
        is IMRPhenomX_utils._PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER
        and _inspiral_phase_host_plain_tensor(
            fit_rows,
            (13,),
            device=cpu,
        )
        and type(_cutoff_fMs) is tuple
        and len(_cutoff_fMs) == 4
        and _inspiral_phase_host_plain_tensor(
            _cutoff_fMs[2],
            (),
            device=cpu,
        )
        and _host_scalars.meco_frequency is _cutoff_fMs[2]
        and type(_host_scalars.builder_values) is tuple
        and len(_host_scalars.builder_values) == 6
        and all(
            type(value) is float and math.isfinite(value)
            for value in _host_scalars.builder_values
        )
        and _inspiral_phase_host_controls_have_provenance(controls, theta)
        and _inspiral_phase_host_values_supported(controls)
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(
            (
                theta,
                phase_coeffs,
                fit_rows,
                _cutoff_fMs,
                controls,
            )
        )
    )


def _inspiral_phase_host_scalar_values(
    theta,
    phase_coeffs,
    fit_rows,
    *,
    bulk_collocation,
    _solve_info,
    _cutoff_fMs,
    _intrinsic_controls,
    _host_scalars,
):
    """Return the validated fixed host inputs without tensor dispatch."""

    if not _inspiral_phase_host_execution_supported(
        theta,
        phase_coeffs,
        fit_rows,
        bulk_collocation=bulk_collocation,
        _solve_info=_solve_info,
        _cutoff_fMs=_cutoff_fMs,
        _intrinsic_controls=_intrinsic_controls,
        _host_scalars=_host_scalars,
    ):
        return None
    controls = _host_scalars.intrinsic_controls
    return (
        *controls.theta_values,
        *controls.fit_values,
        *_host_scalars.builder_values,
    )


def _owned_inspiral_phase_host_plan(plan, theta):
    """Copy the fixed tensor leaves into owned scalars with two native calls."""

    try:
        packed = torch.tensor(
            _INSPIRAL_PHASE_HOST_PLAN_ITEMS(plan),
            dtype=theta.dtype,
            device=theta.device,
        )
        owned = torch.unbind_copy(packed)
    except Exception:
        return None
    if (
        type(owned) is not tuple
        or len(owned) != len(_INSPIRAL_PHASE_HOST_TENSOR_POSITIONS)
        or not all(
            _inspiral_phase_host_plain_tensor(
                value,
                (),
                device=theta.device,
            )
            for value in owned
        )
    ):
        return None
    values = list(plan)
    for index, value in zip(
        _INSPIRAL_PHASE_HOST_TENSOR_POSITIONS,
        owned,
        strict=True,
    ):
        values[index] = value
    return _InspiralPhasePlan(*values)


class _InspiralAmpExactSquareFloat(float):
    """Use one retained Torch square inside an otherwise binary64 DAG."""

    def __new__(cls, value, exact_square):
        result = float.__new__(cls, value)
        result._exact_square = exact_square
        return result

    def __pow__(self, exponent):
        if exponent == 2.0:
            return self._exact_square
        return float.__pow__(self, exponent)

    def __mul__(self, other):
        if other is self:
            return self._exact_square
        return float(self) * other


def _inspiral_amp_host_plain_tensor(value, shape) -> bool:
    """Accept one exact, owned CPU binary64 tensor in the amp schema."""

    return _inspiral_phase_host_plain_tensor(
        value,
        shape,
        device=torch.device("cpu"),
    )


def _inspiral_amp_host_controls_have_provenance(controls, theta) -> bool:
    """Accept no shared controls or controls tied to this exact theta."""

    return controls is None or _inspiral_phase_host_controls_have_provenance(
        controls,
        theta,
    )


def _maybe_inspiral_amp_host_scalars(
    theta,
    amp_coeffs,
    chip,
    final_spin,
    coprecessing_deviations,
    amp_fit_rows,
    aligned_cutoff_fMs,
    intrinsic_controls,
):
    """Build one exact request-local amp host pack, or fail closed."""

    if (
        not _inspiral_amp_host_scalars_enabled()
        or not torch.is_grad_enabled()
        or torch.is_inference_mode_enabled()
        or not _python_intermediate_amp_runtime_supported()
        or not callable(getattr(torch, "unbind_copy", None))
        or not callable(getattr(jnp, "foreach_cbrt", None))
        or (
            final_spin is not None
            and (
                type(final_spin) not in (int, float)
                or not math.isfinite(final_spin)
            )
        )
        or coprecessing_deviations is not None
        or type(chip) not in (int, float)
        or not math.isfinite(chip)
        or not _inspiral_amp_host_plain_tensor(theta, (4,))
        or not _inspiral_amp_host_plain_tensor(amp_coeffs, (7, 42))
        or amp_coeffs
        is not IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER
        or not _inspiral_amp_host_plain_tensor(amp_fit_rows, (7,))
        or type(aligned_cutoff_fMs) is not tuple
        or len(aligned_cutoff_fMs) != 4
        or not all(
            _inspiral_amp_host_plain_tensor(value, ())
            for value in aligned_cutoff_fMs[2:4]
        )
        or not _inspiral_amp_host_controls_have_provenance(
            intrinsic_controls,
            theta,
        )
        or IMRPhenomX_utils._tree_has_autograd_untrusted(
            (
                theta,
                amp_coeffs,
                chip,
                amp_fit_rows,
                aligned_cutoff_fMs,
                intrinsic_controls,
            )
        )
    ):
        return None

    try:
        mass1, mass2, spin1, spin2 = theta
        if intrinsic_controls is None:
            mass1_seconds = mass1 * MTSUN
            mass2_seconds = mass2 * MTSUN
            total_mass_seconds = mass1_seconds + mass2_seconds
            total_mass_square, spin1_square, spin2_square = torch._foreach_pow(
                (total_mass_seconds, spin1, spin2),
                (2.0, 2.0, 2.0),
            )
            eta = mass1_seconds * mass2_seconds / total_mass_square
            delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
        else:
            eta = intrinsic_controls.eta
            delta = intrinsic_controls.delta
            spin1_square, spin2_square = torch._foreach_pow(
                (spin1, spin2),
                (2.0, 2.0),
            )
        eta_square = eta * eta

        meco_frequency, isco_frequency = aligned_cutoff_fMs[2:4]
        match_frequency = meco_frequency + 0.25 * (
            isco_frequency - meco_frequency
        )
        control_points = (
            0.50 * match_frequency,
            0.75 * match_frequency,
            1.00 * match_frequency,
        )
        power_values = torch._foreach_pow(
            control_points * 3,
            (
                7.0 / 3.0,
                7.0 / 3.0,
                7.0 / 3.0,
                8.0 / 3.0,
                8.0 / 3.0,
                8.0 / 3.0,
                3,
                3,
                3,
            ),
        )
        cube_roots = jnp.foreach_cbrt(
            control_points,
            prevalidated=True,
        )
        exact_leaves = torch.stack(
            (
                meco_frequency,
                isco_frequency,
                eta,
                eta_square,
                delta,
                spin1_square,
                spin2_square,
                *power_values,
                *cube_roots,
            )
        )
        builder_values = tuple(
            torch.cat((theta, amp_fit_rows[:3], exact_leaves)).tolist()
        )
    except Exception:
        return None
    if (
        len(builder_values) != 26
        or not all(
            type(value) is float and math.isfinite(value)
            for value in builder_values
        )
        or builder_values[0] <= 0.0
        or builder_values[1] <= 0.0
    ):
        return None
    return _IMRPhenomXASInspiralAmpHostScalars(
        theta,
        amp_coeffs,
        amp_fit_rows,
        meco_frequency,
        isco_frequency,
        intrinsic_controls,
        chip,
        builder_values,
    )


def _inspiral_amp_host_execution_supported(
    theta,
    amp_coeffs,
    chip,
    *,
    fit_rows,
    _cutoff_fMs,
    _intrinsic_controls,
    _host_scalars,
) -> bool:
    """Revalidate the fixed amp schema at its arithmetic consumer."""

    return (
        type(_host_scalars) is _IMRPhenomXASInspiralAmpHostScalars
        and _inspiral_amp_host_scalars_enabled()
        and torch.is_grad_enabled()
        and not torch.is_inference_mode_enabled()
        and _python_intermediate_amp_runtime_supported()
        and callable(getattr(torch, "unbind_copy", None))
        and _host_scalars.theta is theta
        and _host_scalars.amp_coeffs is amp_coeffs
        and _host_scalars.amp_fit_rows is fit_rows
        and _host_scalars.intrinsic_controls is _intrinsic_controls
        and type(chip) is type(_host_scalars.chip)
        and chip == _host_scalars.chip
        and _inspiral_amp_host_plain_tensor(theta, (4,))
        and _inspiral_amp_host_plain_tensor(amp_coeffs, (7, 42))
        and amp_coeffs
        is IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER
        and _inspiral_amp_host_plain_tensor(fit_rows, (7,))
        and type(_cutoff_fMs) is tuple
        and len(_cutoff_fMs) == 4
        and _host_scalars.meco_frequency is _cutoff_fMs[2]
        and _host_scalars.isco_frequency is _cutoff_fMs[3]
        and all(
            _inspiral_amp_host_plain_tensor(value, ())
            for value in _cutoff_fMs[2:4]
        )
        and _inspiral_amp_host_controls_have_provenance(
            _intrinsic_controls,
            theta,
        )
        and type(_host_scalars.builder_values) is tuple
        and len(_host_scalars.builder_values) == 26
        and all(
            type(value) is float and math.isfinite(value)
            for value in _host_scalars.builder_values
        )
        and _host_scalars.builder_values[0] > 0.0
        and _host_scalars.builder_values[1] > 0.0
        and not IMRPhenomX_utils._tree_has_autograd_untrusted(
            (
                theta,
                amp_coeffs,
                chip,
                fit_rows,
                _cutoff_fMs,
                _intrinsic_controls,
            )
        )
    )


def _owned_inspiral_amp_host_plan(plan, theta):
    """Copy eight host results into plain, owned scalar tensors."""

    if (
        type(plan) is not _InspiralAmpPlan
        or type(plan.A0) is not float
        or plan.A0 != 1.0
        or not all(type(value) is float and math.isfinite(value) for value in plan[1:])
    ):
        return None
    try:
        owned = torch.unbind_copy(
            torch.tensor(
                plan[1:],
                dtype=theta.dtype,
                device=theta.device,
            )
        )
    except Exception:
        return None
    if (
        type(owned) is not tuple
        or len(owned) != 8
        or not all(_inspiral_amp_host_plain_tensor(value, ()) for value in owned)
    ):
        return None
    return _InspiralAmpPlan(plan.A0, *owned)


def _inspiral_amp_host_plan(
    theta,
    amp_coeffs,
    chip,
    *,
    fit_rows,
    _cutoff_fMs,
    _intrinsic_controls,
    _host_scalars,
):
    """Replay the eager coefficient DAG in CPython binary64 if proven."""

    if not _inspiral_amp_host_execution_supported(
        theta,
        amp_coeffs,
        chip,
        fit_rows=fit_rows,
        _cutoff_fMs=_cutoff_fMs,
        _intrinsic_controls=_intrinsic_controls,
        _host_scalars=_host_scalars,
    ):
        return None
    (
        mass1,
        mass2,
        spin1,
        spin2,
        cv0,
        cv1,
        cv2,
        meco_frequency,
        isco_frequency,
        eta,
        eta_square,
        delta,
        spin1_square,
        spin2_square,
        cp0_7o3,
        cp1_7o3,
        cp2_7o3,
        cp0_8o3,
        cp1_8o3,
        cp2_8o3,
        cp0_3,
        cp1_3,
        cp2_3,
        cp0_cbrt,
        cp1_cbrt,
        cp2_cbrt,
    ) = _host_scalars.builder_values
    spin1 = _InspiralAmpExactSquareFloat(spin1, spin1_square)
    spin2 = _InspiralAmpExactSquareFloat(spin2, spin2_square)
    eta = _InspiralAmpExactSquareFloat(eta, eta_square)
    host_controls = _InspiralAmpHostIntrinsicScalars(
        mass1,
        mass2,
        spin1,
        spin2,
        eta,
        delta,
        0.0,
        0.0,
    )
    rho_powers = (
        (cp0_7o3, cp1_7o3, cp2_7o3),
        (cp0_8o3, cp1_8o3, cp2_8o3),
        (cp0_3, cp1_3, cp2_3),
        (cp0_cbrt, cp1_cbrt, cp2_cbrt),
    )
    try:
        host_plan = _prepare_inspiral_amp_eager(
            (mass1, mass2, spin1, spin2),
            amp_coeffs,
            chip,
            fit_rows=(cv0, cv1, cv2),
            _cutoff_fMs=(0.0, 0.0, meco_frequency, isco_frequency),
            _intrinsic_controls=host_controls,
            _host_rho_powers=rho_powers,
        )
        return _owned_inspiral_amp_host_plan(host_plan, theta)
    except Exception:
        return None


def _maybe_intrinsic_controls(theta, phase_coeffs, amp_coeffs):
    """Return one shared exact control bundle, or preserve repeated builders."""

    if not _intrinsic_control_reuse_enabled():
        return None
    if not _intrinsic_control_reuse_supported(theta):
        return None
    if (
        _phase_fit_python_scalars_enabled()
        or _amp_fit_python_scalars_enabled()
        or _phase_plan_cuda_solve_graph_enabled()
    ):
        return None
    host_values = _phase_fit_native_iterator_supported(
        theta,
        phase_coeffs,
    ) or _amp_fit_native_iterator_supported(theta, amp_coeffs)
    try:
        controls = _build_intrinsic_controls(
            theta,
            host_values=host_values,
        )
    except Exception:
        return None
    if not _intrinsic_controls_valid(
        controls,
        theta,
        host_values=host_values,
    ):
        return None
    return controls


def _intrinsic_plan_cache_float_bits(value) -> bytes:
    """Return the exact IEEE-754 binary64 representation of a public scalar."""

    return struct.pack("!d", value)


def _intrinsic_plan_cache_public_bits(inputs):
    """Return every public frequency-independent physical scalar bitwise."""

    values = (
        inputs.mass1,
        inputs.mass2,
        inputs.spin1z,
        inputs.spin2z,
        inputs.lambda1,
        inputs.lambda2,
        inputs.dquad1,
        inputs.dquad2,
    )
    if not all(type(value) is float and math.isfinite(value) for value in values):
        return None
    return tuple(_intrinsic_plan_cache_float_bits(value) for value in values)


def _intrinsic_plan_cache_theta_bits(theta_intrinsic):
    """Read exact binary64 CPU theta bits for public-scalar provenance."""

    try:
        if (
            not _packed_frequency_plan_plain_tensor(theta_intrinsic, (4,))
            or theta_intrinsic.device.type != "cpu"
            or IMRPhenomX_utils._tree_has_autograd_untrusted(theta_intrinsic)
        ):
            return None
        values = theta_intrinsic.tolist()
        if type(values) is not list or len(values) != 4:
            return None
        return tuple(_intrinsic_plan_cache_float_bits(value) for value in values)
    except Exception:
        return None


def _intrinsic_plan_cache_table_key(value, shape, *, source, current):
    """Describe one live canonical CPU master without scalar extraction."""

    if (
        value is not source
        or value is not current
        or not _packed_frequency_plan_plain_tensor(
            value,
            shape,
            device=torch.device("cpu"),
        )
    ):
        return None
    if (
        value.dtype is not torch.float64
        or value._base is not None
        or IMRPhenomX_utils._tree_has_autograd_untrusted(value)
    ):
        return None
    try:
        if value._version != 0:
            return None
        return (
            id(value),
            value.data_ptr(),
            value._version,
            value.device.type,
            value.device.index,
            value.dtype,
            tuple(value.shape),
            tuple(value.stride()),
        )
    except (AttributeError, RuntimeError, TypeError):
        return None


def _intrinsic_plan_cache_gate_profile():
    """Snapshot every PyCBC execution switch that can alter a plan DAG."""

    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith("PYCBC_")
        )
    )


def _intrinsic_plan_cache_runtime_key(inputs):
    """Qualify and describe one stable ordinary eager CPU runtime."""

    try:
        state = _scheme.mgr.state
        active_tensor = _torch_jax_compat._ACTIVE_TENSOR.get()
        if (
            type(state) is not _scheme.TorchScheme
            or type(state.torch_device) is not torch.device
            or state.torch_device != inputs.device
            or type(active_tensor) is not torch.Tensor
            or active_tensor.layout is not torch.strided
            or active_tensor.device != inputs.device
            or active_tensor.dtype is not inputs.real_dtype
            or active_tensor._base is not None
            or active_tensor.storage_offset() != 0
            or not active_tensor.is_contiguous()
            or IMRPhenomX_utils._tree_has_autograd_untrusted(active_tensor)
            or not torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
            or sys.gettrace() is not None
            or sys.getprofile() is not None
            or torch.get_num_threads() != 1
            or torch.get_default_dtype() is not torch.float32
            or not _packed_heaviside_masks_runtime_supported()
        ):
            return None
        gil_enabled = getattr(sys, "_is_gil_enabled", None)
        return (
            getattr(sys.implementation, "name", None),
            sys.version_info[:3],
            True if gil_enabled is None else bool(gil_enabled()),
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
            torch.get_default_dtype(),
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.is_anomaly_enabled(),
            torch.get_float32_matmul_precision(),
            bool(torch.backends.mkldnn.enabled),
            bool(IMRPhenomX_utils._TRUSTED_PLAIN_REQUEST.get()),
            bool(
                _torch_jax_compat._IDENTITY_TENSOR_FASTPATH_ACTIVE.get()
            ),
        )
    except Exception:
        return None


def _intrinsic_plan_cache_key(inputs, phase_coeffs, amp_coeffs):
    """Build a host-only exact key for one public XAS intrinsic state."""

    if (
        type(inputs) is not _IMRPhenomXASInputs
        or inputs.tidal_version is not None
        or type(inputs.device) is not torch.device
        or inputs.device.type != "cpu"
        or inputs.device.index is not None
        or inputs.real_dtype is not torch.float64
        or inputs.complex_dtype is not torch.complex128
        or not _phase_plan_enabled()
        or not _amp_plan_enabled()
    ):
        return None
    runtime_key = _intrinsic_plan_cache_runtime_key(inputs)
    if runtime_key is None:
        return None
    intrinsic_bits = _intrinsic_plan_cache_public_bits(inputs)
    if intrinsic_bits is None:
        return None
    phase_table = _intrinsic_plan_cache_table_key(
        phase_coeffs,
        (13, 49),
        source=_PHASE_FIT_COEFFICIENT_SOURCE,
        current=IMRPhenomX_utils._PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER,
    )
    amp_table = _intrinsic_plan_cache_table_key(
        amp_coeffs,
        (7, 42),
        source=_AMP_FIT_COEFFICIENT_SOURCE,
        current=IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER,
    )
    if phase_table is None or amp_table is None:
        return None
    return (
        _INTRINSIC_PLAN_CACHE_MODEL_RELEASE,
        torch.__version__,
        getattr(torch.version, "cuda", None),
        getattr(torch.version, "hip", None),
        intrinsic_bits,
        inputs.device.type,
        inputs.device.index,
        inputs.real_dtype,
        inputs.complex_dtype,
        phase_table,
        amp_table,
        runtime_key,
        _intrinsic_plan_cache_gate_profile(),
    )


def _intrinsic_plan_cache_tensors(value):
    """Yield tensor leaves in a stable private-plan traversal order."""

    if type(value) is torch.Tensor:
        yield value
        return
    if isinstance(value, tuple):
        for item in value:
            yield from _intrinsic_plan_cache_tensors(item)


def _intrinsic_plan_cache_seal(value):
    """Deep-own every tensor occurrence while retaining exact schemas."""

    if type(value) is torch.Tensor:
        return value.detach().clone(memory_format=torch.contiguous_format)
    if type(value) is tuple:
        return tuple(_intrinsic_plan_cache_seal(item) for item in value)
    if isinstance(value, tuple):
        return type(value)(
            *(_intrinsic_plan_cache_seal(item) for item in value)
        )
    if value is None or type(value) in (bool, int, float, bytes):
        return value
    raise TypeError("unsupported intrinsic-plan cache leaf")


def _intrinsic_plan_cache_tensor_provenance(value):
    """Capture structural and mutation provenance without device extraction."""

    try:
        return (
            id(value),
            value.data_ptr(),
            value._version,
            value.layout,
            value.device.type,
            value.device.index,
            value.dtype,
            tuple(value.shape),
            tuple(value.stride()),
            value.storage_offset(),
            id(value._base) if value._base is not None else None,
            value.requires_grad,
            value.grad_fn is None,
            value.is_contiguous(),
            value.is_conj(),
            value.is_neg(),
        )
    except (AttributeError, RuntimeError, TypeError):
        return None


def _intrinsic_plan_cache_storage_bytes(value):
    """Return exact uniquely-owned storage bytes, or reject any alias."""

    try:
        seen = set()
        byte_size = 0
        for tensor in _intrinsic_plan_cache_tensors(value):
            storage = tensor.untyped_storage()
            identity = (
                tensor.device.type,
                tensor.device.index,
                storage._cdata,
            )
            if identity in seen:
                return None
            seen.add(identity)
            byte_size += storage.nbytes()
        return byte_size if seen and byte_size > 0 else None
    except (AttributeError, RuntimeError, TypeError):
        return None


def _intrinsic_plan_cache_flat_leaves_supported(
    leaves,
    provenance,
    expected_byte_size,
) -> bool:
    """Revalidate preflattened leaves without walking the plan structure."""

    try:
        if (
            type(leaves) is not tuple
            or type(provenance) is not tuple
            or type(expected_byte_size) is not int
            or expected_byte_size <= 0
            or len(leaves) != len(provenance)
            or not leaves
        ):
            return False
        seen = set()
        byte_size = 0
        for value, expected in zip(leaves, provenance, strict=True):
            storage = value.untyped_storage()
            storage_identity = (
                value.device.type,
                value.device.index,
                storage._cdata,
            )
            if (
                expected is None
                or id(value) != expected[0]
                or value.data_ptr() != expected[1]
                or value._version != expected[2]
                or value.layout is not expected[3]
                or value.device.type != expected[4]
                or value.device.index != expected[5]
                or value.dtype is not expected[6]
                or value.shape != expected[7]
                or value.stride() != expected[8]
                or value.storage_offset() != expected[9]
                or (
                    id(value._base) if value._base is not None else None
                )
                != expected[10]
                or value.requires_grad != expected[11]
                or (value.grad_fn is None) != expected[12]
                or value.is_contiguous() != expected[13]
                or value.is_conj() != expected[14]
                or value.is_neg() != expected[15]
                or storage_identity in seen
            ):
                return False
            seen.add(storage_identity)
            byte_size += storage.nbytes()
        return byte_size == expected_byte_size
    except Exception:
        return False


def _intrinsic_plan_cache_bundle_supported(bundle, *, device, dtype) -> bool:
    """Validate one private immutable bundle before insertion or every hit."""

    try:
        phase = bundle.phase_plan
        amp = bundle.amp_plan
        scripted_phase = (
            type(phase.inspiral) is _ScriptedInspiralPhasePlan
            or type(phase.intermediate) is _ScriptedIntermediatePhasePlan
            or type(phase.mergerringdown)
            is _ScriptedMergerRingdownPhasePlan
        )
        prequalified_scalar = (
            type(phase.scalar_inspiral) is _PrequalifiedInspiralPhasePlan
            or type(phase.scalar_intermediate)
            is _PrequalifiedIntermediatePhasePlan
            or type(phase.scalar_mergerringdown)
            is _PrequalifiedMergerRingdownPhasePlan
        )
        if (
            type(bundle) is not _IMRPhenomXASIntrinsicPlanBundle
            or type(phase) is not _IMRPhenomXASPhasePlan
            or type(amp) is not _IMRPhenomXASAmpPlan
            or type(bundle.intrinsic_bits) is not tuple
            or len(bundle.intrinsic_bits) != 8
            or not all(
                type(value) is bytes and len(value) == 8
                for value in bundle.intrinsic_bits
            )
            or type(phase.inspiral)
            not in (_InspiralPhasePlan, _ScriptedInspiralPhasePlan)
            or type(phase.intermediate)
            not in (_IntermediatePhasePlan, _ScriptedIntermediatePhasePlan)
            or type(phase.mergerringdown)
            not in (
                _MergerRingdownPhasePlan,
                _ScriptedMergerRingdownPhasePlan,
            )
            or type(phase.scalar_inspiral)
            not in (_InspiralPhasePlan, _PrequalifiedInspiralPhasePlan)
            or type(phase.scalar_intermediate)
            not in (
                _IntermediatePhasePlan,
                _PrequalifiedIntermediatePhasePlan,
            )
            or type(phase.scalar_mergerringdown)
            not in (
                _MergerRingdownPhasePlan,
                _PrequalifiedMergerRingdownPhasePlan,
            )
            or type(amp.inspiral) is not _InspiralAmpPlan
            or type(amp.intermediate) is not _IntermediateAmpPlan
            or type(amp.mergerringdown) is not _MergerRingdownAmpPlan
            or type(bundle.aligned_cutoff) is not tuple
            or len(bundle.aligned_cutoff) != 4
            or type(bundle.cutoff) is not tuple
            or len(bundle.cutoff) != 4
            or type(bundle.phase_rows) is not torch.Tensor
            or bundle.phase_rows.shape != (13,)
            or type(bundle.amp_rows) is not torch.Tensor
            or bundle.amp_rows.shape != (7,)
            or (
                scripted_phase
                and (
                    not _scripted_phase_ansatz_cpu_enabled()
                    or not _scripted_phase_ansatz_cpu_runtime_supported()
                )
            )
            or (
                prequalified_scalar
                and not (
                    _scalar_derivative_plan_cse_enabled()
                    and _exact_scalar_derivatives_enabled()
                )
            )
            or IMRPhenomX_utils._tree_has_autograd_untrusted(bundle)
        ):
            return False
        tensors = tuple(_intrinsic_plan_cache_tensors(bundle))
        return (
            bool(tensors)
            and _intrinsic_plan_cache_storage_bytes(bundle) is not None
            and all(
                type(value) is torch.Tensor
                and value.layout is torch.strided
                and value.device == device
                and value.dtype is dtype
                and value.is_contiguous()
                and value.storage_offset() == 0
                and value._base is None
                and value._version == 0
                and not value.requires_grad
                and value.grad_fn is None
                and not value.is_conj()
                and not value.is_neg()
                and _intrinsic_plan_cache_tensor_provenance(value) is not None
                for value in tensors
            )
        )
    except Exception:
        return False


def _intrinsic_plan_cache_bundle_matches_theta(
    bundle,
    theta_intrinsic,
) -> bool:
    """Bind one sealed bundle to the exact four request tensor scalars."""

    try:
        theta_bits = _intrinsic_plan_cache_theta_bits(theta_intrinsic)
        return (
            theta_bits is not None
            and _intrinsic_plan_cache_bundle_supported(
                bundle,
                device=theta_intrinsic.device,
                dtype=theta_intrinsic.dtype,
            )
            and bundle.intrinsic_bits[:4] == theta_bits
        )
    except Exception:
        return False


def _intrinsic_plan_cache_entry_supported(
    entry,
    key,
    phase_coeffs,
    amp_coeffs,
    *,
    device,
    dtype,
) -> bool:
    """Reject rebinding, mutation, schema drift, and stale cache entries."""

    try:
        if (
            type(entry) is not _IMRPhenomXASIntrinsicPlanCacheEntry
            or type(key) is not tuple
            or len(key) != 13
            or type(key[4]) is not tuple
            or entry.key != key
            or entry.bundle.intrinsic_bits != key[4]
            or entry.phase_coeffs is not phase_coeffs
            or entry.amp_coeffs is not amp_coeffs
            or _intrinsic_plan_cache_table_key(
                entry.phase_coeffs,
                (13, 49),
                source=_PHASE_FIT_COEFFICIENT_SOURCE,
                current=(
                    IMRPhenomX_utils._PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER
                ),
            )
            is None
            or _intrinsic_plan_cache_table_key(
                entry.amp_coeffs,
                (7, 42),
                source=_AMP_FIT_COEFFICIENT_SOURCE,
                current=(
                    IMRPhenomX_utils._PHENOMX_AMP_COEFF_TABLE_CPU_MASTER
                ),
            )
            is None
            or type(entry.tensor_provenance) is not tuple
            or type(entry.byte_size) is not int
            or entry.byte_size <= 0
            or not _intrinsic_plan_cache_bundle_supported(
                entry.bundle,
                device=device,
                dtype=dtype,
            )
        ):
            return False
        current = tuple(
            _intrinsic_plan_cache_tensor_provenance(value)
            for value in _intrinsic_plan_cache_tensors(entry.bundle)
        )
        return (
            current == entry.tensor_provenance
            and all(value is not None for value in current)
            and _intrinsic_plan_cache_storage_bytes(entry.bundle)
            == entry.byte_size
        )
    except Exception:
        return False


def _intrinsic_plan_cache_fast_entry_supported(
    entry,
    key,
    phase_coeffs,
    amp_coeffs,
) -> bool:
    """Validate an opt-in entry through its immutable schema and flat leaves."""

    try:
        bundle = entry.bundle
        phase = bundle.phase_plan
        amp = bundle.amp_plan
        scripted_phase = (
            type(phase.inspiral) is _ScriptedInspiralPhasePlan
            or type(phase.intermediate) is _ScriptedIntermediatePhasePlan
            or type(phase.mergerringdown)
            is _ScriptedMergerRingdownPhasePlan
        )
        prequalified_scalar = (
            type(phase.scalar_inspiral) is _PrequalifiedInspiralPhasePlan
            or type(phase.scalar_intermediate)
            is _PrequalifiedIntermediatePhasePlan
            or type(phase.scalar_mergerringdown)
            is _PrequalifiedMergerRingdownPhasePlan
        )
        if (
            type(entry)
            is not _IMRPhenomXASIntrinsicPlanFastCacheEntry
            or type(key) is not tuple
            or len(key) != 13
            or type(key[4]) is not tuple
            or entry.key != key
            or entry.phase_coeffs is not phase_coeffs
            or entry.amp_coeffs is not amp_coeffs
            or type(bundle) is not _IMRPhenomXASIntrinsicPlanBundle
            or id(bundle) != entry.bundle_identity
            or id(entry.tensor_leaves) != entry.tensor_leaves_identity
            or id(entry.tensor_provenance)
            != entry.tensor_provenance_identity
            or bundle.intrinsic_bits != key[4]
            or type(bundle.intrinsic_bits) is not tuple
            or len(bundle.intrinsic_bits) != 8
            or not all(
                type(value) is bytes and len(value) == 8
                for value in bundle.intrinsic_bits
            )
            or type(phase) is not _IMRPhenomXASPhasePlan
            or type(amp) is not _IMRPhenomXASAmpPlan
            or type(phase.inspiral)
            not in (_InspiralPhasePlan, _ScriptedInspiralPhasePlan)
            or type(phase.intermediate)
            not in (_IntermediatePhasePlan, _ScriptedIntermediatePhasePlan)
            or type(phase.mergerringdown)
            not in (
                _MergerRingdownPhasePlan,
                _ScriptedMergerRingdownPhasePlan,
            )
            or type(phase.scalar_inspiral)
            not in (_InspiralPhasePlan, _PrequalifiedInspiralPhasePlan)
            or type(phase.scalar_intermediate)
            not in (
                _IntermediatePhasePlan,
                _PrequalifiedIntermediatePhasePlan,
            )
            or type(phase.scalar_mergerringdown)
            not in (
                _MergerRingdownPhasePlan,
                _PrequalifiedMergerRingdownPhasePlan,
            )
            or type(amp.inspiral) is not _InspiralAmpPlan
            or type(amp.intermediate) is not _IntermediateAmpPlan
            or type(amp.mergerringdown) is not _MergerRingdownAmpPlan
            or type(bundle.aligned_cutoff) is not tuple
            or len(bundle.aligned_cutoff) != 4
            or type(bundle.cutoff) is not tuple
            or len(bundle.cutoff) != 4
            or type(bundle.phase_rows) is not torch.Tensor
            or bundle.phase_rows.shape != (13,)
            or type(bundle.amp_rows) is not torch.Tensor
            or bundle.amp_rows.shape != (7,)
            or (
                scripted_phase
                and (
                    not _scripted_phase_ansatz_cpu_enabled()
                    or not _scripted_phase_ansatz_cpu_runtime_supported()
                )
            )
            or (
                prequalified_scalar
                and not (
                    _scalar_derivative_plan_cse_enabled()
                    and _exact_scalar_derivatives_enabled()
                )
            )
        ):
            return False
        return _intrinsic_plan_cache_flat_leaves_supported(
            entry.tensor_leaves,
            entry.tensor_provenance,
            entry.byte_size,
        )
    except Exception:
        return False


def _build_intrinsic_plan_bundle(
    theta_intrinsic,
    phase_coeffs,
    amp_coeffs,
    intrinsic_bits,
):
    """Build only canonical frequency-independent XAS intrinsic state."""

    intrinsic_controls = _maybe_intrinsic_controls(
        theta_intrinsic,
        phase_coeffs,
        amp_coeffs,
    )
    if intrinsic_controls is None:
        m1, m2, chi1, chi2 = theta_intrinsic
    else:
        m1 = intrinsic_controls.mass1
        m2 = intrinsic_controls.mass2
        chi1 = intrinsic_controls.spin1
        chi2 = intrinsic_controls.spin2
    phase_rows = _prepare_phase_fit_rows(
        theta_intrinsic,
        phase_coeffs,
        _intrinsic_controls=intrinsic_controls,
    )
    aligned_cutoff = IMRPhenomX_utils.get_cutoff_fMs(m1, m2, chi1, chi2)
    if _packed_cutoff_reuse_enabled():
        cutoff = (
            aligned_cutoff[0] - 0.0,
            aligned_cutoff[1] + 0.0,
            aligned_cutoff[2],
            aligned_cutoff[3],
        )
    else:
        cutoff = _get_cutoff_fMs(m1, m2, chi1, chi2)
    phase_plan = _prepare_phase_plan(
        theta_intrinsic,
        phase_coeffs,
        _phase_fit_rows=phase_rows,
        _cutoff_fMs=cutoff,
        _intrinsic_controls=intrinsic_controls,
    )
    amp_rows = _prepare_amp_fit_rows(
        theta_intrinsic,
        amp_coeffs,
        _intrinsic_controls=intrinsic_controls,
    )
    amp_plan = _prepare_amp_plan(
        theta_intrinsic,
        amp_coeffs,
        fit_rows=amp_rows,
        _aligned_cutoff_fMs=aligned_cutoff,
        _cutoff_fMs=cutoff,
        _intrinsic_controls=intrinsic_controls,
    )
    unsealed = _IMRPhenomXASIntrinsicPlanBundle(
        intrinsic_bits,
        phase_rows,
        amp_rows,
        tuple(aligned_cutoff),
        tuple(cutoff),
        phase_plan,
        amp_plan,
    )
    return _intrinsic_plan_cache_seal(unsealed)


def _clear_intrinsic_plan_cache() -> None:
    """Release every private intrinsic plan under the cache mutex."""

    global _INTRINSIC_PLAN_CACHE_BYTES
    global _INTRINSIC_PLAN_CACHE_EVICTIONS
    global _INTRINSIC_PLAN_CACHE_HITS
    global _INTRINSIC_PLAN_CACHE_MISSES
    global _INTRINSIC_PLAN_CACHE_OVERSIZED
    global _INTRINSIC_PLAN_CACHE_PID

    with _INTRINSIC_PLAN_CACHE_LOCK:
        _INTRINSIC_PLAN_CACHE.clear()
        _INTRINSIC_PLAN_CACHE_BYTES = 0
        _INTRINSIC_PLAN_CACHE_HITS = 0
        _INTRINSIC_PLAN_CACHE_MISSES = 0
        _INTRINSIC_PLAN_CACHE_EVICTIONS = 0
        _INTRINSIC_PLAN_CACHE_OVERSIZED = 0
        _INTRINSIC_PLAN_CACHE_PID = os.getpid()


def _intrinsic_plan_cache_stats():
    """Return a consistent private diagnostic snapshot for qualification."""

    with _INTRINSIC_PLAN_CACHE_LOCK:
        return {
            "entries": len(_INTRINSIC_PLAN_CACHE),
            "bytes": _INTRINSIC_PLAN_CACHE_BYTES,
            "hits": _INTRINSIC_PLAN_CACHE_HITS,
            "misses": _INTRINSIC_PLAN_CACHE_MISSES,
            "evictions": _INTRINSIC_PLAN_CACHE_EVICTIONS,
            "oversized": _INTRINSIC_PLAN_CACHE_OVERSIZED,
        }


def _maybe_cached_intrinsic_plan_bundle(
    inputs,
    theta_intrinsic,
    phase_coeffs,
    amp_coeffs,
    *,
    _request_proof=None,
    _return_validated_hit=False,
):
    """Return a validated LRU hit, populate one miss, or fail closed."""

    global _INTRINSIC_PLAN_CACHE_BYTES
    global _INTRINSIC_PLAN_CACHE_EVICTIONS
    global _INTRINSIC_PLAN_CACHE_HITS
    global _INTRINSIC_PLAN_CACHE_MISSES
    global _INTRINSIC_PLAN_CACHE_OVERSIZED

    if not _intrinsic_plan_cache_enabled():
        return None
    if _request_proof_plan_current(_request_proof):
        return None
    fast_hit_requested = False
    if _return_validated_hit:
        fast_hit_requested = _intrinsic_plan_cache_fast_hit_enabled()
    if _INTRINSIC_PLAN_CACHE_PID != os.getpid():
        _reset_intrinsic_plan_cache_after_fork()
    try:
        intrinsic_bits = _intrinsic_plan_cache_public_bits(inputs)
        theta_bits = _intrinsic_plan_cache_theta_bits(theta_intrinsic)
        if (
            not _packed_frequency_plan_plain_tensor(theta_intrinsic, (4,))
            or theta_intrinsic.device != inputs.device
            or theta_intrinsic.dtype is not inputs.real_dtype
            or IMRPhenomX_utils._tree_has_autograd_untrusted(theta_intrinsic)
            or intrinsic_bits is None
            or theta_bits is None
            or theta_bits != intrinsic_bits[:4]
        ):
            return None
        key = _intrinsic_plan_cache_key(inputs, phase_coeffs, amp_coeffs)
    except Exception:
        return None
    if key is None:
        return None

    with _INTRINSIC_PLAN_CACHE_LOCK:
        entry = _INTRINSIC_PLAN_CACHE.get(key)
        if entry is not None:
            entry_supported = (
                _intrinsic_plan_cache_fast_entry_supported(
                    entry,
                    key,
                    phase_coeffs,
                    amp_coeffs,
                )
                if fast_hit_requested
                else _intrinsic_plan_cache_entry_supported(
                    entry,
                    key,
                    phase_coeffs,
                    amp_coeffs,
                    device=inputs.device,
                    dtype=inputs.real_dtype,
                )
            )
            if entry_supported:
                _INTRINSIC_PLAN_CACHE.move_to_end(key)
                _INTRINSIC_PLAN_CACHE_HITS += 1
                if fast_hit_requested:
                    validated_hit = _issue_intrinsic_plan_cache_validated_hit(
                        entry,
                        key,
                        inputs,
                        theta_intrinsic,
                        theta_bits,
                        phase_coeffs,
                        amp_coeffs,
                    )
                    if validated_hit is not None:
                        return _IMRPhenomXASIntrinsicPlanValidatedHit(
                            entry.bundle,
                            validated_hit,
                        )
                return entry.bundle
            _INTRINSIC_PLAN_CACHE.pop(key, None)
            if type(entry.byte_size) is int and entry.byte_size > 0:
                _INTRINSIC_PLAN_CACHE_BYTES = max(
                    0,
                    _INTRINSIC_PLAN_CACHE_BYTES - entry.byte_size,
                )
        _INTRINSIC_PLAN_CACHE_MISSES += 1
        try:
            bundle = _build_intrinsic_plan_bundle(
                theta_intrinsic,
                phase_coeffs,
                amp_coeffs,
                intrinsic_bits,
            )
            if (
                _intrinsic_plan_cache_key(inputs, phase_coeffs, amp_coeffs)
                != key
                or not _intrinsic_plan_cache_bundle_supported(
                    bundle,
                    device=inputs.device,
                    dtype=inputs.real_dtype,
                )
            ):
                return None
            if fast_hit_requested:
                tensor_leaves = tuple(
                    _intrinsic_plan_cache_tensors(bundle)
                )
                provenance = tuple(
                    _intrinsic_plan_cache_tensor_provenance(value)
                    for value in tensor_leaves
                )
            else:
                provenance = tuple(
                    _intrinsic_plan_cache_tensor_provenance(value)
                    for value in _intrinsic_plan_cache_tensors(bundle)
                )
            if not provenance or any(value is None for value in provenance):
                return None
            byte_size = _intrinsic_plan_cache_storage_bytes(bundle)
            if byte_size is None:
                return None
            if byte_size > _INTRINSIC_PLAN_CACHE_MAX_BYTES:
                _INTRINSIC_PLAN_CACHE_OVERSIZED += 1
                return bundle
            if fast_hit_requested:
                entry = _IMRPhenomXASIntrinsicPlanFastCacheEntry(
                    key,
                    phase_coeffs,
                    amp_coeffs,
                    tensor_leaves,
                    provenance,
                    byte_size,
                    bundle,
                    id(bundle),
                    id(tensor_leaves),
                    id(provenance),
                )
            else:
                entry = _IMRPhenomXASIntrinsicPlanCacheEntry(
                    key,
                    phase_coeffs,
                    amp_coeffs,
                    provenance,
                    byte_size,
                    bundle,
                )
            _INTRINSIC_PLAN_CACHE[key] = entry
            _INTRINSIC_PLAN_CACHE_BYTES += byte_size
            while (
                len(_INTRINSIC_PLAN_CACHE)
                > _INTRINSIC_PLAN_CACHE_MAX_ENTRIES
                or _INTRINSIC_PLAN_CACHE_BYTES
                > _INTRINSIC_PLAN_CACHE_MAX_BYTES
            ):
                _, evicted = _INTRINSIC_PLAN_CACHE.popitem(last=False)
                _INTRINSIC_PLAN_CACHE_BYTES -= evicted.byte_size
                _INTRINSIC_PLAN_CACHE_EVICTIONS += 1
            return bundle
        except Exception:
            failed = _INTRINSIC_PLAN_CACHE.pop(key, None)
            if failed is not None and type(failed.byte_size) is int:
                _INTRINSIC_PLAN_CACHE_BYTES = max(
                    0,
                    _INTRINSIC_PLAN_CACHE_BYTES - failed.byte_size,
                )
            return None


def _packed_frequency_plan_scalar(value, like):
    """Return one scalar in the plan's common device and dtype."""

    if type(value) is torch.Tensor:
        if (
            value.layout is torch.strided
            and value.ndim == 0
            and value._base is None
            and value.storage_offset() == 0
            and value.device == like.device
            and value.dtype == like.dtype
        ):
            # ``reshape(())`` only creates a new view for this exact owned
            # scalar contract.  ``torch.stack`` consumes the same value and
            # copies it into the packed plan, so keep the original tensor and
            # avoid two dispatcher events per fixed-schema scalar.  Views and
            # every other tensor retain the legacy normalization below.
            return value
        return value.to(device=like.device, dtype=like.dtype).reshape(())
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _build_packed_frequency_plan(
    theta_intrinsic,
    theta_extrinsic,
    phase_coeffs,
    amp_coeffs,
    f_ref,
    *,
    _intrinsic_plan_bundle=None,
    _intrinsic_plan_cache_validated_hit=None,
    _request_proof=None,
):
    """Construct the exact frequency-independent state as one scalar lane."""

    if _intrinsic_plan_bundle is None:
        intrinsic_controls = _maybe_intrinsic_controls(
            theta_intrinsic,
            phase_coeffs,
            amp_coeffs,
        )
        if intrinsic_controls is None:
            m1, m2, chi1, chi2 = theta_intrinsic
        else:
            m1 = intrinsic_controls.mass1
            m2 = intrinsic_controls.mass2
            chi1 = intrinsic_controls.spin1
            chi2 = intrinsic_controls.spin2
        phase_rows = _prepare_phase_fit_rows(
            theta_intrinsic,
            phase_coeffs,
            _intrinsic_controls=intrinsic_controls,
        )
        aligned_cutoff = IMRPhenomX_utils.get_cutoff_fMs(
            m1,
            m2,
            chi1,
            chi2,
        )
        if _packed_cutoff_reuse_enabled():
            # Packed-plan eligibility proves the otherwise-default arguments:
            # chip=0, final_spin=None, and no co-precessing deviations.
            # Preserve the wrapper's arithmetic while reusing the physical fit.
            cutoff = (
                aligned_cutoff[0] - 0.0,
                aligned_cutoff[1] + 0.0,
                aligned_cutoff[2],
                aligned_cutoff[3],
            )
        else:
            cutoff = _get_cutoff_fMs(m1, m2, chi1, chi2)
        phase_plan = _prepare_phase_plan(
            theta_intrinsic,
            phase_coeffs,
            _phase_fit_rows=phase_rows,
            _cutoff_fMs=cutoff,
            _intrinsic_controls=intrinsic_controls,
            _request_proof=_request_proof,
        )
    else:
        if (
            _request_proof_plan_current(_request_proof)
            or (
                (
                    _intrinsic_plan_cache_validated_hit is None
                    or not _intrinsic_plan_cache_validated_hit_current(
                        _intrinsic_plan_cache_validated_hit,
                        _intrinsic_plan_bundle,
                        theta_intrinsic,
                        phase_coeffs,
                        amp_coeffs,
                    )
                )
                and not _intrinsic_plan_cache_bundle_matches_theta(
                    _intrinsic_plan_bundle,
                    theta_intrinsic,
                )
            )
        ):
            raise ValueError("invalid cached intrinsic frequency plan")
        intrinsic_controls = None
        m1, m2, chi1, chi2 = theta_intrinsic
        aligned_cutoff = _intrinsic_plan_bundle.aligned_cutoff
        cutoff = _intrinsic_plan_bundle.cutoff
        phase_plan = _intrinsic_plan_bundle.phase_plan
    linear_a, linear_b, phase_at_reference = _phase_alignment_terms(
        theta_intrinsic,
        phase_coeffs,
        f_ref,
        theta_extrinsic[2],
        _phase_plan=phase_plan,
        _cutoff_fMs=cutoff,
        _request_proof=_request_proof,
    )
    if _intrinsic_plan_bundle is None:
        amp_rows = _prepare_amp_fit_rows(
            theta_intrinsic,
            amp_coeffs,
            _intrinsic_controls=intrinsic_controls,
        )
        amp_plan = _prepare_amp_plan(
            theta_intrinsic,
            amp_coeffs,
            fit_rows=amp_rows,
            _aligned_cutoff_fMs=aligned_cutoff,
            _cutoff_fMs=cutoff,
            _intrinsic_controls=intrinsic_controls,
            _request_proof=_request_proof,
        )
    else:
        amp_plan = _intrinsic_plan_bundle.amp_plan

    if _intrinsic_plan_bundle is not None:
        total_mass_seconds = phase_plan.total_mass_seconds
        eta = phase_plan.eta
    elif intrinsic_controls is None:
        m1_seconds = m1 * MTSUN
        m2_seconds = m2 * MTSUN
        total_mass_seconds = m1_seconds + m2_seconds
        eta = m1_seconds * m2_seconds / (total_mass_seconds**2.0)
    else:
        total_mass_seconds = intrinsic_controls.total_mass_seconds
        eta = intrinsic_controls.eta
    _, _, meco, isco = aligned_cutoff
    amp_match = meco + 0.25 * (isco - meco)
    amp0 = (
        2.0
        * torch.sqrt(theta_intrinsic.new_tensor(5.0 / (64.0 * PI)))
        * total_mass_seconds**2
        / ((theta_extrinsic[0] * MPC) / C)
    )
    amp_norm = torch.sqrt(2.0 * eta / 3.0) * (PI ** (-1.0 / 6.0))
    overall_amp = amp0 * amp_norm

    values = (
        phase_plan.total_mass_seconds,
        phase_plan.eta,
        phase_plan.f1_Ms,
        phase_plan.f2_Ms,
        *phase_plan.inspiral,
        *phase_plan.intermediate,
        *phase_plan.mergerringdown,
        phase_plan.alpha0,
        phase_plan.alpha1,
        phase_plan.beta0,
        phase_plan.beta1,
        *amp_plan.inspiral,
        *amp_plan.intermediate,
        *amp_plan.mergerringdown,
        linear_a,
        linear_b,
        phase_at_reference,
        theta_extrinsic[1],
        overall_amp,
        amp_match,
    )
    if len(values) != _PACKED_FREQUENCY_PLAN_WIDTH:
        raise RuntimeError("packed XAS frequency-plan schema changed")
    return torch.stack(
        tuple(_packed_frequency_plan_scalar(value, theta_intrinsic) for value in values)
    )


def _evaluate_packed_frequency_inspiral_phase(frequency, plan):
    (
        phi0,
        phi1,
        phi2,
        phi3,
        phi4,
        phi5,
        phi5_l,
        phi6,
        phi6_l,
        phi7,
        phi8,
        phi8_l,
        sigma1,
        sigma2,
        sigma3,
        sigma4,
    ) = plan
    f13 = frequency ** (1.0 / 3.0)
    f23 = f13 * f13
    f43 = frequency * f13
    f53 = frequency * f23
    f2 = frequency * frequency
    f73 = f2 * f13
    f83 = f2 * f23
    f3 = f2 * frequency
    f103 = f3 * f13
    f113 = f3 * f23
    log_f = torch.log(frequency)
    phase_tf2 = (
        phi0
        + phi1 * f13
        + phi2 * f23
        + phi3 * frequency
        + phi4 * f43
        + phi5 * f53
        + phi5_l * f53 * log_f
        + phi6 * f2
        + phi6_l * f2 * log_f
        + phi7 * f73
        + phi8 * f83
        + phi8_l * f83 * log_f
    )
    phase_inspiral = phase_tf2 + (
        sigma1 * f83 + sigma2 * f3 + sigma3 * f103 + sigma4 * f113
    )
    normalization = -(3.0 * PI ** (-5.0 / 3.0)) / 128.0
    return phase_inspiral * normalization / f53


def _evaluate_packed_frequency_intermediate_phase(frequency, plan):
    b0, b1, b2, b3, b4, c_l, f_rd, f_damp = plan
    return (
        b0 * frequency
        + b1 * torch.log(frequency)
        - b2 * (frequency**-1.0)
        - b3 * (frequency**-2.0) / 2.0
        - (b4 * (frequency**-3.0) / 3.0)
        + (2.0 * c_l * torch.atan((frequency - f_rd) / (2.0 * f_damp))) / f_damp
    )


def _evaluate_packed_frequency_ringdown_phase(frequency, plan):
    c0, c1, c2, c4_over_3, c_l_over_f_damp, f_rd, f_damp, _, _ = plan
    return (
        c0 * frequency
        + 1.5 * c1 * (frequency ** (2.0 / 3.0))
        - c2 * (frequency**-1.0)
        - c4_over_3 * (frequency**-3.0)
        + c_l_over_f_damp * torch.atan((frequency - f_rd) / f_damp)
    )


def _evaluate_packed_frequency_inspiral_amp(frequency, plan):
    a0, a2, a3, a4, a5, a6, rho1, rho2, rho3 = plan
    return (
        a0
        + a2 * (frequency ** (2.0 / 3.0))
        + a3 * frequency
        + a4 * (frequency ** (4.0 / 3.0))
        + a5 * (frequency ** (5.0 / 3.0))
        + a6 * (frequency**2.0)
        + rho1 * (frequency ** (7.0 / 3.0))
        + rho2 * (frequency ** (8.0 / 3.0))
        + rho3 * (frequency**3.0)
    )


def _evaluate_packed_frequency_intermediate_amp(frequency, plan):
    delta0, delta1, delta2, delta3, delta4 = plan
    return (frequency ** (7.0 / 6.0)) / (
        delta0
        + frequency
        * (delta1 + frequency * (delta2 + frequency * (delta3 + frequency * delta4)))
    )


def _evaluate_packed_frequency_ringdown_amp(frequency, plan):
    f_rd, gamma_r, gamma_d2, gamma_d13, _ = plan
    return (
        torch.exp(-(frequency - f_rd) * gamma_r)
        * gamma_d13
        / ((frequency - f_rd) * (frequency - f_rd) + gamma_d2)
    )


def _evaluate_packed_frequency_plan(frequency, packed):
    """Evaluate one XAS frequency vector from its fixed 66-scalar plan."""

    values = packed.unbind(0)
    total_mass_seconds, eta, phase_lower, phase_upper = values[:4]
    phase_inspiral = values[4:20]
    phase_intermediate = values[20:28]
    phase_ringdown = values[28:37]
    alpha0, alpha1, beta0, beta1 = values[37:41]
    amp_inspiral = values[41:50]
    amp_intermediate = values[50:55]
    amp_ringdown = values[55:60]
    linear_a, linear_b, phase_at_reference = values[60:63]
    time_shift, overall_amp, amp_match = values[63:66]

    dimensionless_frequency = frequency * total_mass_seconds
    phase_ins = _evaluate_packed_frequency_inspiral_phase(
        dimensionless_frequency,
        phase_inspiral,
    )
    phase_int = (
        _evaluate_packed_frequency_intermediate_phase(
            dimensionless_frequency,
            phase_intermediate,
        )
        + alpha1 * dimensionless_frequency
        + alpha0
    )
    phase_rd = (
        _evaluate_packed_frequency_ringdown_phase(
            dimensionless_frequency,
            phase_ringdown,
        )
        + beta0
        + beta1 * dimensionless_frequency
    )
    phase_masks = _native_packed_heaviside_masks(
        dimensionless_frequency,
        phase_lower,
        phase_upper,
    )
    phase = (1 / eta) * (
        phase_ins * phase_masks[0]
        + phase_masks[1] * phase_int * phase_masks[2]
        + phase_rd * phase_masks[3] * phase_masks[4]
    )
    extrinsic_phase = 2.0 * PI * frequency * time_shift
    phase = (
        phase
        + (linear_b * dimensionless_frequency)
        + linear_a
        + phase_at_reference
        - 2 * PI
        + extrinsic_phase
    )

    amplitude_ins = _evaluate_packed_frequency_inspiral_amp(
        dimensionless_frequency,
        amp_inspiral,
    )
    amplitude_int = _evaluate_packed_frequency_intermediate_amp(
        dimensionless_frequency,
        amp_intermediate,
    )
    amplitude_rd = _evaluate_packed_frequency_ringdown_amp(
        dimensionless_frequency,
        amp_ringdown,
    )
    amp_upper = amp_ringdown[4]
    amp_masks = _native_packed_heaviside_masks(
        dimensionless_frequency,
        amp_match,
        amp_upper,
    )
    amplitude = (
        amplitude_ins * amp_masks[0]
        + amp_masks[1] * amplitude_int * amp_masks[2]
        + amplitude_rd * amp_masks[3] * amp_masks[4]
    )
    amplitude = overall_amp * amplitude * (dimensionless_frequency ** (-7.0 / 6.0))
    return amplitude * torch.exp(1j * phase)


def _packed_frequency_plan_raw_equal(left, right) -> bool:
    """Compare shape, dtype, device, and every output byte exactly."""

    if (
        type(left) is not torch.Tensor
        or type(right) is not torch.Tensor
        or left.shape != right.shape
        or left.dtype is not right.dtype
        or left.device != right.device
    ):
        return False
    try:
        left_bytes = left.detach().contiguous().view(torch.uint8)
        right_bytes = right.detach().contiguous().view(torch.uint8)
        return bool(torch.equal(left_bytes, right_bytes))
    except Exception:
        return False


def _packed_frequency_plan_trace_key(frequency):
    """Key one shape-polymorphic fixed-schema TorchScript evaluator."""

    return (
        frequency.device.type,
        frequency.device.index,
        frequency.dtype,
        _PACKED_FREQUENCY_PLAN_WIDTH,
    )


def _packed_frequency_plan_trace_remember_failure_locked(key) -> None:
    """Drop a trace and remember its exact bounded runtime key."""

    _PACKED_FREQUENCY_PLAN_TRACE_CACHE.pop(key, None)
    _PACKED_FREQUENCY_PLAN_TRACE_FAILURES.pop(key, None)
    _PACKED_FREQUENCY_PLAN_TRACE_FAILURES[key] = None
    while len(_PACKED_FREQUENCY_PLAN_TRACE_FAILURES) > (
        _PACKED_FREQUENCY_PLAN_TRACE_MAX_ENTRIES
    ):
        _PACKED_FREQUENCY_PLAN_TRACE_FAILURES.popitem(last=False)


def _clear_packed_frequency_plan_trace_cache() -> None:
    """Release cached TorchScript modules and remembered failures."""

    global _PACKED_FREQUENCY_PLAN_TRACE_PID

    with _PACKED_FREQUENCY_PLAN_TRACE_LOCK:
        _PACKED_FREQUENCY_PLAN_TRACE_CACHE.clear()
        _PACKED_FREQUENCY_PLAN_TRACE_FAILURES.clear()
        _PACKED_FREQUENCY_PLAN_TRACE_PID = os.getpid()


def _packed_frequency_plan_cached_trace_executor(frequency):
    """Return one validated warm executor without evaluating its eager lane."""

    if _PACKED_FREQUENCY_PLAN_TRACE_PID != os.getpid():
        _reset_packed_frequency_plan_trace_after_fork()
    key = _packed_frequency_plan_trace_key(frequency)
    with _PACKED_FREQUENCY_PLAN_TRACE_LOCK:
        executor = _PACKED_FREQUENCY_PLAN_TRACE_CACHE.get(key)
        if executor is not None:
            _PACKED_FREQUENCY_PLAN_TRACE_CACHE.move_to_end(key)
        return executor


def _packed_frequency_plan_trace_executor(frequency, packed, eager):
    """Return a raw-self-checked cached trace, or ``None`` to stay eager."""

    if _PACKED_FREQUENCY_PLAN_TRACE_PID != os.getpid():
        _reset_packed_frequency_plan_trace_after_fork()
    key = _packed_frequency_plan_trace_key(frequency)
    with _PACKED_FREQUENCY_PLAN_TRACE_LOCK:
        executor = _PACKED_FREQUENCY_PLAN_TRACE_CACHE.get(key)
        if executor is not None:
            _PACKED_FREQUENCY_PLAN_TRACE_CACHE.move_to_end(key)
            return executor
        if key in _PACKED_FREQUENCY_PLAN_TRACE_FAILURES:
            _PACKED_FREQUENCY_PLAN_TRACE_FAILURES.move_to_end(key)
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", torch.jit.TracerWarning)
                executor = torch.jit.trace(
                    _evaluate_packed_frequency_plan,
                    (frequency, packed),
                    check_trace=False,
                    strict=True,
                )
            traced = executor(frequency, packed)
        except Exception:
            _packed_frequency_plan_trace_remember_failure_locked(key)
            return None
        if not _packed_frequency_plan_raw_equal(eager, traced):
            _packed_frequency_plan_trace_remember_failure_locked(key)
            return None
        _PACKED_FREQUENCY_PLAN_TRACE_CACHE[key] = executor
        while len(_PACKED_FREQUENCY_PLAN_TRACE_CACHE) > (
            _PACKED_FREQUENCY_PLAN_TRACE_MAX_ENTRIES
        ):
            _PACKED_FREQUENCY_PLAN_TRACE_CACHE.popitem(last=False)
        return executor


def _packed_frequency_plan_trace_failed(executor, frequency) -> None:
    """Evict and fail closed after a cached trace execution error."""

    if _PACKED_FREQUENCY_PLAN_TRACE_PID != os.getpid():
        _reset_packed_frequency_plan_trace_after_fork()
    key = _packed_frequency_plan_trace_key(frequency)
    with _PACKED_FREQUENCY_PLAN_TRACE_LOCK:
        if _PACKED_FREQUENCY_PLAN_TRACE_CACHE.get(key) is executor:
            _packed_frequency_plan_trace_remember_failure_locked(key)


def _packed_frequency_plan_output_supported(output, frequency) -> bool:
    """Validate the inexpensive public contract of one traced output."""

    return (
        type(output) is torch.Tensor
        and output.layout is torch.strided
        and output.device == frequency.device
        and output.dtype is torch.complex128
        and output.shape == frequency.shape
        and output.is_contiguous()
        and output.storage_offset() == 0
        and output._base is None
        and not output.is_conj()
        and not output.is_neg()
    )


def _maybe_packed_frequency_plan(
    frequency,
    theta_intrinsic,
    theta_extrinsic,
    phase_coeffs,
    amp_coeffs,
    f_ref,
    nrtidal,
    chip,
    final_spin,
    coprecessing_deviations,
    return_phase_plan,
    return_amp_plan,
    phase_fit_rows,
    amp_fit_rows,
    aligned_cutoff,
    cutoff,
    return_carrier_alignment_result,
    *,
    _intrinsic_plan_bundle=None,
    _intrinsic_plan_cache_validated_hit=None,
    _request_proof=None,
):
    """Return the opted-in packed result, or preserve the legacy XAS body."""

    packed_enabled = _packed_frequency_plan_enabled()
    trace_enabled = _packed_frequency_plan_torchscript_trace_enabled()
    if not packed_enabled and not trace_enabled:
        return None
    if (
        nrtidal is not None
        or type(chip) is not float
        or chip != 0.0
        or final_spin is not None
        or coprecessing_deviations is not None
        or return_phase_plan
        or return_amp_plan
        or phase_fit_rows is not None
        or amp_fit_rows is not None
        or aligned_cutoff is not None
        or cutoff is not None
        or return_carrier_alignment_result
        or not _packed_frequency_plan_supported(
            frequency,
            theta_intrinsic,
            theta_extrinsic,
            phase_coeffs,
            amp_coeffs,
            f_ref,
        )
    ):
        return None

    try:
        packed = _build_packed_frequency_plan(
            theta_intrinsic,
            theta_extrinsic,
            phase_coeffs,
            amp_coeffs,
            f_ref,
            _intrinsic_plan_bundle=_intrinsic_plan_bundle,
            _intrinsic_plan_cache_validated_hit=(
                _intrinsic_plan_cache_validated_hit
            ),
            _request_proof=_request_proof,
        )
    except Exception:
        return None

    lookup_failed = False
    if trace_enabled:
        try:
            executor = _packed_frequency_plan_cached_trace_executor(frequency)
        except Exception:
            executor = None
            lookup_failed = True
        if executor is not None:
            try:
                traced = executor(frequency, packed)
                if _packed_frequency_plan_output_supported(traced, frequency):
                    return traced
            except Exception:
                pass
            try:
                _packed_frequency_plan_trace_failed(executor, frequency)
            except Exception:
                pass
            lookup_failed = True

    try:
        eager = _evaluate_packed_frequency_plan(frequency, packed)
    except Exception:
        return None
    if not _packed_frequency_plan_output_supported(eager, frequency):
        return None
    if not trace_enabled or lookup_failed:
        return eager

    executor = _packed_frequency_plan_trace_executor(
        frequency,
        packed,
        eager,
    )
    if executor is None:
        return eager
    try:
        traced = executor(frequency, packed)
    except Exception:
        _packed_frequency_plan_trace_failed(executor, frequency)
        return eager
    if not _packed_frequency_plan_output_supported(traced, frequency):
        _packed_frequency_plan_trace_failed(executor, frequency)
        return eager
    return traced


def _gen_IMRPhenomXAS(
    f: Float[Array, " n_freq"],
    theta_intrinsic: Float[Array, "4"],
    theta_extrinsic: Float[Array, "3"],
    phase_coeffs: Float[Array, "13 49"],
    amp_coeffs: Float[Array, "7 42"],
    f_ref: float,
    nrtidal: _NRTidalParams | None = None,
    *,
    chip: float = 0.0,
    final_spin: FloatLike | None = None,
    coprecessing_deviations: PNRCoprecessingDeviations | None = None,
    return_phase_plan: bool = False,
    return_amp_plan: bool = False,
    _phase_fit_rows: torch.Tensor | None = None,
    _amp_fit_rows: torch.Tensor | None = None,
    _aligned_cutoff_fMs=None,
    _cutoff_fMs=None,
    _intrinsic_plan_bundle=None,
    _intrinsic_plan_cache_validated_hit=None,
    _request_proof=None,
    _return_carrier_alignment_result: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, _IMRPhenomXASPhasePlan | None]
    | tuple[torch.Tensor, _IMRPhenomXASAmpPlan | None]
    | tuple[
        torch.Tensor,
        _IMRPhenomXASPhasePlan | None,
        _IMRPhenomXASAmpPlan | None,
    ]
):
    if (
        f.device.type == "cpu"
        and (not torch.is_grad_enabled() or not (f.requires_grad or theta_intrinsic.requires_grad))
        and nrtidal is None
        and type(chip) is float
        and chip == 0.0
        and final_spin is None
        and coprecessing_deviations is None
        and not return_phase_plan
        and not return_amp_plan
        and _phase_fit_rows is None
        and _amp_fit_rows is None
        and not _return_carrier_alignment_result
        and float(theta_extrinsic[1]) == 0.0
        and _request_proof is None
    ):
        try:
            from ._imrphenomxp_msa_native import evaluate_xas_native
            m1_val = float(theta_intrinsic[0])
            m2_val = float(theta_intrinsic[1])
            s1z_val = float(theta_intrinsic[2])
            s2z_val = float(theta_intrinsic[3])
            m1_s_val = m1_val * MTSUN
            m2_s_val = m2_val * MTSUN
            M_s_val = m1_s_val + m2_s_val
            eta_val = m1_s_val * m2_s_val / (M_s_val ** 2.0)
            amp0_val = 2.0 * math.sqrt(5.0 / (64.0 * math.pi)) * (M_s_val ** 2) / ((float(theta_extrinsic[0]) * MPC) / C)
            ampNorm_val = math.sqrt(2.0 * eta_val / 3.0) * (math.pi ** (-1.0 / 6.0))
            overall_amp_val = amp0_val * ampNorm_val

            phase_plan = _prepare_phase_plan(theta_intrinsic, phase_coeffs)
            amp_plan = _prepare_amp_plan(theta_intrinsic, amp_coeffs, chip=0.0, fit_rows=None)

            _, _, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(m1_val, m2_val, s1z_val, s2z_val)
            fMs_AmpMatchIN_val = float(fMs_MECO + 0.25 * (fMs_ISCO - fMs_MECO))
            fMs_AmpRDMin_val = float(amp_plan.mergerringdown.fMs_AmpRDMin)

            insp_p = torch.as_tensor([float(x) for x in phase_plan.inspiral], dtype=torch.float64)
            int_p = torch.as_tensor([
                float(phase_plan.intermediate.b0), float(phase_plan.intermediate.b1), float(phase_plan.intermediate.b2),
                float(phase_plan.intermediate.b3), float(phase_plan.intermediate.b4), float(phase_plan.intermediate.cL),
                float(phase_plan.intermediate.fMs_RD), float(phase_plan.intermediate.fMs_damp),
                float(phase_plan.alpha0), float(phase_plan.alpha1)
            ], dtype=torch.float64)
            mrd_p = torch.as_tensor([
                float(phase_plan.mergerringdown.c0), float(phase_plan.mergerringdown.c1), float(phase_plan.mergerringdown.c2),
                float(phase_plan.mergerringdown.c4ov3), float(phase_plan.mergerringdown.cLovfda),
                float(phase_plan.mergerringdown.fMs_RD), float(phase_plan.mergerringdown.fMs_damp),
                float(phase_plan.beta0), float(phase_plan.beta1)
            ], dtype=torch.float64)

            insp_a = torch.as_tensor([float(x) for x in amp_plan.inspiral], dtype=torch.float64)
            int_a = torch.as_tensor([float(x) for x in amp_plan.intermediate], dtype=torch.float64)
            mrd_a = torch.as_tensor([
                float(amp_plan.mergerringdown.fMs_RD), float(amp_plan.mergerringdown.gammaR),
                float(amp_plan.mergerringdown.gammaD2), float(amp_plan.mergerringdown.gammaD13)
            ], dtype=torch.float64)

            phase_alignment = _phase_alignment_terms(
                theta_intrinsic, phase_coeffs, f_ref, theta_extrinsic[2], _phase_plan=phase_plan
            )
            lina, linb, phifRef = phase_alignment
            lin_phase_val = float(linb * M_s_val)
            const_phase_val = float(lina + phifRef - 2.0 * math.pi)

            res = evaluate_xas_native(
                f, M_s_val, eta_val, overall_amp_val,
                float(phase_plan.f1_Ms), float(phase_plan.f2_Ms),
                fMs_AmpMatchIN_val, fMs_AmpRDMin_val,
                insp_p, int_p, mrd_p, insp_a, int_a, mrd_a,
                lin_phase_val, const_phase_val, -999.0, 0.0
            )
            if res is not None:
                return res[0]
        except Exception:
            pass

    if not _request_proof_plan_current(_request_proof):
        _request_proof = None
    intrinsic_plan_cache_validated_hit = (
        _intrinsic_plan_cache_validated_hit is not None
        and type(_intrinsic_plan_bundle)
        is _IMRPhenomXASIntrinsicPlanBundle
        and _intrinsic_plan_cache_validated_hit_current(
            _intrinsic_plan_cache_validated_hit,
            _intrinsic_plan_bundle,
            theta_intrinsic,
            phase_coeffs,
            amp_coeffs,
        )
    )
    if (
        type(_intrinsic_plan_bundle) is not _IMRPhenomXASIntrinsicPlanBundle
        or (
            not intrinsic_plan_cache_validated_hit
            and not _intrinsic_plan_cache_bundle_matches_theta(
                _intrinsic_plan_bundle,
                theta_intrinsic,
            )
        )
        or _request_proof is not None
        or nrtidal is not None
        or type(chip) is not float
        or chip != 0.0
        or final_spin is not None
        or coprecessing_deviations is not None
        or return_phase_plan
        or return_amp_plan
        or _phase_fit_rows is not None
        or _amp_fit_rows is not None
        or _aligned_cutoff_fMs is not None
        or _cutoff_fMs is not None
        or _return_carrier_alignment_result
    ):
        _intrinsic_plan_bundle = None
        _intrinsic_plan_cache_validated_hit = None

    packed_result = _maybe_packed_frequency_plan(
        f,
        theta_intrinsic,
        theta_extrinsic,
        phase_coeffs,
        amp_coeffs,
        f_ref,
        nrtidal,
        chip,
        final_spin,
        coprecessing_deviations,
        return_phase_plan,
        return_amp_plan,
        _phase_fit_rows,
        _amp_fit_rows,
        _aligned_cutoff_fMs,
        _cutoff_fMs,
        _return_carrier_alignment_result,
        _intrinsic_plan_bundle=_intrinsic_plan_bundle,
        _intrinsic_plan_cache_validated_hit=(
            _intrinsic_plan_cache_validated_hit
        ),
        _request_proof=_request_proof,
    )
    if packed_result is not None:
        return packed_result

    m1, m2, _, _ = theta_intrinsic
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN

    M_s = m1_s + m2_s
    fM_s = f * M_s
    phase_plan = (
        None
        if _intrinsic_plan_bundle is None
        else _intrinsic_plan_bundle.phase_plan
    )
    active_aligned_cutoff = (
        _aligned_cutoff_fMs
        if _intrinsic_plan_bundle is None
        else _intrinsic_plan_bundle.aligned_cutoff
    )
    active_cutoff = (
        _cutoff_fMs
        if _intrinsic_plan_bundle is None
        else _intrinsic_plan_bundle.cutoff
    )
    # Private precomputed rows are only useful through the phase plan.  Treat
    # validated rows as an explicit request for that exact reuse path so XP's
    # backend-specific fallback does not re-evaluate the fits internally.
    reuse_precomputed_phase_rows = (
        _phase_plan_enabled()
        and _precomputed_phase_fit_rows_supported(
            _phase_fit_rows,
            theta_intrinsic,
        )
    )
    request_phase_ready = _request_proof_phase_ready(_request_proof)
    if (
        phase_plan is None
        and nrtidal is None
        and _phase_plan_enabled()
        and _fixed_schema_phase_plan_enabled()
    ):
        from ._imrphenomxas_fixed_schema_phase_plan import (
            _maybe_prepare_fixed_schema_public_default_phase_plan,
        )

        phase_plan = _maybe_prepare_fixed_schema_public_default_phase_plan(
            theta_intrinsic,
            phase_coeffs,
            chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
            _phase_fit_rows=_phase_fit_rows,
            _cutoff_fMs=active_cutoff,
            _intrinsic_controls=None,
            _request_proof=_request_proof,
        )
    if phase_plan is None and (
        request_phase_ready
        or (
            (_phase_plan_enabled() or reuse_precomputed_phase_rows)
            and not IMRPhenomX_utils._tree_has_autograd(
                (
                    theta_intrinsic,
                    phase_coeffs,
                    chip,
                    final_spin,
                    coprecessing_deviations,
                    _phase_fit_rows,
                    active_cutoff,
                )
            )
        )
    ):
        phase_plan = _prepare_phase_plan(
            theta_intrinsic,
            phase_coeffs,
            chip,
            final_spin=final_spin,
            coprecessing_deviations=coprecessing_deviations,
            _phase_fit_rows=_phase_fit_rows,
            _cutoff_fMs=active_cutoff,
            _request_proof=_request_proof,
        )
    Psi = Phase(
        f,
        theta_intrinsic,
        phase_coeffs,
        chip,
        final_spin=final_spin,
        coprecessing_deviations=coprecessing_deviations,
        _phase_plan=phase_plan,
        _cutoff_fMs=active_cutoff,
        _request_proof=_request_proof,
    )

    # Generate the linear in f and constant contribution to the phase in order
    # to roll the waveform such that the peak is at the input tc and phic
    phase_alignment = _phase_alignment_terms(
        theta_intrinsic,
        phase_coeffs,
        f_ref,
        theta_extrinsic[2],
        chip=chip,
        final_spin=final_spin,
        coprecessing_deviations=coprecessing_deviations,
        _phase_plan=phase_plan,
        _cutoff_fMs=active_cutoff,
        _request_proof=_request_proof,
        _return_carrier_alignment_result=(
            _return_carrier_alignment_result
        ),
    )
    if _return_carrier_alignment_result:
        lina, linb, phifRef, carrier_alignment_result = phase_alignment
    else:
        lina, linb, phifRef = phase_alignment
        carrier_alignment_result = None
    ext_phase_contrib = 2.0 * PI * f * theta_extrinsic[1]

    amplitude = Amp(
        f,
        theta_intrinsic,
        amp_coeffs,
        D=theta_extrinsic[0],
        chip=chip,
        final_spin=final_spin,
        coprecessing_deviations=coprecessing_deviations,
        _amp_fit_rows=_amp_fit_rows,
        _return_amp_plan=return_amp_plan,
        _aligned_cutoff_fMs=active_aligned_cutoff,
        _cutoff_fMs=active_cutoff,
        _prepared_amp_plan=(
            None
            if _intrinsic_plan_bundle is None
            else _intrinsic_plan_bundle.amp_plan
        ),
        _request_proof=_request_proof,
    )
    if return_amp_plan:
        A, amp_plan = amplitude
    else:
        A = amplitude
        amp_plan = None

    Psi = Psi + (linb * fM_s) + lina + phifRef - 2 * PI + ext_phase_contrib
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
                chip,
                final_spin=final_spin,
                coprecessing_deviations=coprecessing_deviations,
                _phase_plan=phase_plan,
                _cutoff_fMs=_cutoff_fMs,
                _request_proof=_request_proof,
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
    if return_phase_plan and not _request_proof_plan_current(_request_proof):
        phase_plan = _request_unqualify_top_plan(
            phase_plan,
            _request_proof,
        )
    if return_amp_plan:
        amp_plan = _request_unqualify_top_plan(
            amp_plan,
            _request_proof,
        )
    if (
        _return_carrier_alignment_result
        and return_phase_plan
        and return_amp_plan
    ):
        return h0, phase_plan, amp_plan, carrier_alignment_result
    if _return_carrier_alignment_result and return_phase_plan:
        return h0, phase_plan, carrier_alignment_result
    if _return_carrier_alignment_result and return_amp_plan:
        return h0, amp_plan, carrier_alignment_result
    if _return_carrier_alignment_result:
        return h0, carrier_alignment_result
    if return_phase_plan and return_amp_plan:
        return h0, phase_plan, amp_plan
    if return_phase_plan:
        return h0, phase_plan
    if return_amp_plan:
        return h0, amp_plan
    return h0


_INVALID_INT4_ORDER = object()
_XAS_IGNORED_COERCED_ORDER_KEYS = ("phase_order", "amplitude_order")
_XAS_IGNORED_EXACT_ORDER_KEYS = ("eccentricity_order",)
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


def _resolve_lal_int4_order(value, *, coerce):
    """Resolve an order using the same conversion as ``_check_lal_pars``."""

    try:
        # PyCBC does not call the SWIG inserter when the supplied value compares
        # equal to the default. In particular, ``-1.0`` is therefore valid for
        # the otherwise exact-integer order fields.
        if value == -1:
            return -1
        value = int(value) if coerce else operator.index(value)
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return _INVALID_INT4_ORDER
    if -(1 << 31) <= value < (1 << 31):
        return value
    return _INVALID_INT4_ORDER


def _is_lal_int4_order(value, *, coerce, allowed=None):
    """Return whether an order survives PyCBC's signed-INT4 LAL boundary."""

    order = _resolve_lal_int4_order(value, coerce=coerce)
    return order is not _INVALID_INT4_ORDER and (allowed is None or order in allowed)


def _is_nonnegative_finite(value):
    if value is None:
        return True
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(value) and value >= 0.0


def _quadrupole_from_params(lambda_value, dquad_value):
    if isinstance(lambda_value, torch.Tensor) or isinstance(dquad_value, torch.Tensor):
        l_val = torch.as_tensor(0.0 if lambda_value is None else lambda_value)
        dq_val = torch.as_tensor(0.0 if dquad_value is None else dquad_value)
        return torch.where(
            (l_val > 0.0) & (dq_val == 0.0),
            nrtidal_quadrupole_from_lambda(l_val),
            1.0 + dq_val,
        )
    lambda_value = float(lambda_value or 0.0)
    dquad_value = float(dquad_value or 0.0)
    if lambda_value > 0.0 and dquad_value == 0.0:
        return nrtidal_quadrupole_from_lambda(lambda_value)
    return 1.0 + dquad_value


def _build_nrtidal_params(
    *,
    tidal_version,
    mass1,
    mass2,
    spin1z,
    spin2z,
    lambda1,
    lambda2,
    dquad1,
    dquad2,
    active_f_max,
):
    """Build the shared scalar matter state for PhenomX carriers."""

    if tidal_version is None:
        return None
    if active_f_max is None:
        raise ValueError("NRTidal generation requires an active maximum frequency")
    if isinstance(active_f_max, torch.Tensor) and active_f_max.ndim == 0:
        active_f_max = active_f_max.detach().item()
    quadrupole1 = _quadrupole_from_params(lambda1, dquad1)
    quadrupole2 = _quadrupole_from_params(lambda2, dquad2)
    if tidal_version == 3:
        merger_frequency = nrtidal_merger_frequency_v3(
            mass1,
            mass2,
            lambda1,
            lambda2,
            spin1z,
            spin2z,
        )
    else:
        merger_frequency = nrtidal_merger_frequency(
            mass1,
            mass2,
            lambda1,
            lambda2,
        )
    if isinstance(active_f_max, torch.Tensor) or isinstance(merger_frequency, torch.Tensor):
        alignment_frequency = torch.minimum(
            torch.as_tensor(active_f_max),
            torch.as_tensor(merger_frequency),
        )
    else:
        alignment_frequency = min(active_f_max, merger_frequency)
    return _NRTidalParams(
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        lambda1=lambda1,
        lambda2=lambda2,
        quadrupole1=quadrupole1,
        quadrupole2=quadrupole2,
        merger_frequency=merger_frequency,
        alignment_frequency=alignment_frequency,
        version=tidal_version,
    )


def imrphenomxas_native_supported(params):
    """Return whether ``params`` preserve the native XAS model semantics."""

    approximant = params.get("approximant", "IMRPhenomXAS")
    if approximant not in {
        "IMRPhenomXAS",
        "IMRPhenomXAS_NRTidalv2",
        "IMRPhenomXAS_NRTidalv3",
    }:
        return False
    # XAS accepts phase, amplitude, and eccentricity PN orders but does not use
    # them. Spin and tidal orders are rejected by both regular and sequence LAL
    # entry points unless they resolve to the default value.
    if any(
        not _is_lal_int4_order(params.get(key, -1), coerce=True)
        for key in _XAS_IGNORED_COERCED_ORDER_KEYS
    ) or any(
        not _is_lal_int4_order(params.get(key, -1), coerce=False)
        for key in _XAS_IGNORED_EXACT_ORDER_KEYS
    ):
        return False
    if not _is_lal_int4_order(
        params.get("spin_order", -1),
        coerce=True,
        allowed=(-1,),
    ) or not _is_lal_int4_order(
        params.get("tidal_order", -1),
        coerce=False,
        allowed=(-1,),
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
        math.isfinite(value) and value >= 0.0 for value in (lambda1, lambda2)
    ):
        raise ValueError("NRTidal deformabilities must be finite and non-negative")
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("IMRPhenomXAS distance must be finite and positive")
    if not math.isfinite(f_ref) or f_ref < 0.0:
        raise ValueError("IMRPhenomXAS f_ref must be finite and non-negative")

    device = state.torch_device
    real_dtype = torch.float32 if device.type == "mps" else torch.float64
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
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
    *,
    return_carrier_plans=False,
    _request_proof=None,
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
    phase_coeffs = (
        IMRPhenomX_utils._get_phenomx_phase_coeff_table_cached_master(
        device=inputs.device,
        dtype=inputs.real_dtype,
        )
    )
    amp_coeffs = IMRPhenomX_utils._get_phenomx_amp_coeff_table_cached_master(
        device=inputs.device,
        dtype=inputs.real_dtype,
    )
    nrtidal = _build_nrtidal_params(
        tidal_version=inputs.tidal_version,
        mass1=inputs.mass1,
        mass2=inputs.mass2,
        spin1z=inputs.spin1z,
        spin2z=inputs.spin2z,
        lambda1=inputs.lambda1,
        lambda2=inputs.lambda2,
        dquad1=inputs.dquad1,
        dquad2=inputs.dquad2,
        active_f_max=active_f_max,
    )
    with torch_context(frequencies):
        intrinsic_plan_bundle = None
        intrinsic_plan_cache_validated_hit = None
        if not return_carrier_plans and nrtidal is None:
            if _intrinsic_plan_cache_fast_hit_enabled():
                cached = _maybe_cached_intrinsic_plan_bundle(
                    inputs,
                    intrinsic,
                    phase_coeffs,
                    amp_coeffs,
                    _request_proof=_request_proof,
                    _return_validated_hit=True,
                )
                if type(cached) is _IMRPhenomXASIntrinsicPlanValidatedHit:
                    intrinsic_plan_bundle = cached.bundle
                    intrinsic_plan_cache_validated_hit = cached.token
                else:
                    intrinsic_plan_bundle = cached
            else:
                intrinsic_plan_bundle = _maybe_cached_intrinsic_plan_bundle(
                    inputs,
                    intrinsic,
                    phase_coeffs,
                    amp_coeffs,
                    _request_proof=_request_proof,
                )
        if intrinsic_plan_cache_validated_hit is None:
            generated = _gen_IMRPhenomXAS(
                frequencies,
                intrinsic,
                extrinsic,
                phase_coeffs,
                amp_coeffs,
                reference_frequency,
                nrtidal,
                return_phase_plan=return_carrier_plans,
                return_amp_plan=return_carrier_plans,
                _intrinsic_plan_bundle=intrinsic_plan_bundle,
                _request_proof=_request_proof,
            )
        else:
            try:
                generated = _gen_IMRPhenomXAS(
                    frequencies,
                    intrinsic,
                    extrinsic,
                    phase_coeffs,
                    amp_coeffs,
                    reference_frequency,
                    nrtidal,
                    return_phase_plan=return_carrier_plans,
                    return_amp_plan=return_carrier_plans,
                    _intrinsic_plan_bundle=intrinsic_plan_bundle,
                    _intrinsic_plan_cache_validated_hit=(
                        intrinsic_plan_cache_validated_hit
                    ),
                    _request_proof=_request_proof,
                )
            finally:
                _retire_intrinsic_plan_cache_validated_hit(
                    intrinsic_plan_cache_validated_hit
                )
    if return_carrier_plans:
        samples, phase_plan, amp_plan = generated
        return (
            samples.to(inputs.complex_dtype),
            _IMRPhenomXASCarrierPlans(phase_plan, amp_plan),
        )
    return generated.to(inputs.complex_dtype)


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


def _imrphenomxas_core_torch(
    p,
    *,
    return_carrier_plans=False,
    _manage_remnant_cache=True,
    _request_proof=None,
):
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
    cache_context = (
        IMRPhenomX_utils.remnant_cache_context()
        if _manage_remnant_cache
        else nullcontext()
    )
    with cache_context:
        generated = _imrphenomxas_samples(
            inputs,
            frequencies,
            reference_frequency,
            active_f_max,
            return_carrier_plans=return_carrier_plans,
            _request_proof=_request_proof,
        )
    if return_carrier_plans:
        h22, carrier_plans = generated
    else:
        h22 = generated

    epoch = -1.0 / delta_f
    core = _IMRPhenomXASCore(
        polarization=h22,
        npts=npts,
        first_bin=first_bin,
        stop_bin=stop_bin,
        delta_f=delta_f,
        epoch=epoch,
    )
    if return_carrier_plans:
        return core, carrier_plans
    return core


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


def _imrphenomxas_fd_torch_impl(p, *, _request_proof=None):
    """Generate polarizations inside an optional validated request scope."""

    core = _imrphenomxas_core_torch(p, _request_proof=_request_proof)
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


def _imrphenomxas_fd_torch_request(p, *, _request_proof=None):
    """Generate one waveform with an optional live request proof."""

    if _request_proof is None:
        return _imrphenomxas_fd_torch_impl(p)
    return _imrphenomxas_fd_torch_impl(
        p,
        _request_proof=_request_proof,
    )


_bind_xas_request_proof_target(_imrphenomxas_fd_torch_impl)
del _bind_xas_request_proof_target


def imrphenomxas_fd_torch(**p):
    """Generate aligned-spin IMRPhenomXAS polarizations with Torch."""

    return _run_xas_request_proof_plan(
        p,
        _imrphenomxas_fd_torch_impl,
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


def _imrphenomxas_sequence_samples(
    p,
    *,
    return_carrier_plans=False,
    _manage_remnant_cache=True,
):
    """Return native XAS samples and metadata for the sequence interface."""

    inputs = _imrphenomxas_inputs(p, sequence=True)
    frequencies = _sequence_frequencies(p["sample_points"], inputs)

    # LAL's sequence API treats the last sample as f_max, even if the input
    # is not sorted, and also applies the calibrated model cutoff.
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / ((inputs.mass1 + inputs.mass2) * MTSUN)
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
    reference_frequency = inputs.f_ref if inputs.f_ref > 0.0 else frequencies[0]
    carrier_plans = _IMRPhenomXASCarrierPlans(None, None)
    if bool(torch.any(active)):
        cache_context = (
            IMRPhenomX_utils.remnant_cache_context()
            if _manage_remnant_cache
            else nullcontext()
        )
        with cache_context:
            generated = _imrphenomxas_samples(
                inputs,
                frequencies[active],
                reference_frequency,
                active_f_max,
                return_carrier_plans=return_carrier_plans,
            )
        if return_carrier_plans:
            active_samples, carrier_plans = generated
        else:
            active_samples = generated
        samples[active] = active_samples

    sequence = _IMRPhenomXASSequence(
        inputs=inputs,
        frequencies=frequencies,
        polarization=samples,
        reference_frequency=reference_frequency,
    )
    if return_carrier_plans:
        return sequence, carrier_plans
    return sequence


def imrphenomxas_fd_sequence_torch(**p):
    """Evaluate IMRPhenomXAS at arbitrary frequencies with Torch."""

    if not imrphenomxas_sequence_native_supported(p):
        raise ValueError(
            "IMRPhenomXAS sequence parameters are not supported by the "
            "native Torch path"
        )
    sequence = _imrphenomxas_sequence_samples(p)
    plus, cross = _polarizations_from_samples(
        sequence.polarization,
        sequence.inputs.inclination,
        sequence.inputs.long_asc_nodes,
    )
    return (
        PyCBCArray(TorchArrayData(plus), copy=False),
        PyCBCArray(TorchArrayData(cross), copy=False),
    )


def _compute_phase_fit_rows_batch(eta, S, StotR, chia, delta, phase_coeffs):
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    chia2 = chia * chia

    spin = torch.cat((S.unsqueeze(1).expand(-1, 4), StotR.unsqueeze(1).expand(-1, 9)), dim=1)
    spin2 = spin * spin
    spin3 = spin2 * spin
    spin4 = spin3 * spin

    c = phase_coeffs.unsqueeze(0)
    e = eta.unsqueeze(1)
    e2 = eta2.unsqueeze(1)
    e3 = eta3.unsqueeze(1)
    e4 = eta4.unsqueeze(1)
    e5 = eta5.unsqueeze(1)
    ca = chia.unsqueeze(1)
    ca2 = chia2.unsqueeze(1)
    d = delta.unsqueeze(1)

    no_spin = (c[..., 0] + c[..., 1]*e + c[..., 2]*e2 + c[..., 3]*e3 + c[..., 4]*e4 + c[..., 5]*e5) / (
        c[..., 6] + c[..., 7]*e + c[..., 8]*e2 + c[..., 9]*e3
    )

    num = spin * (
        c[..., 10] + c[..., 11]*spin + c[..., 12]*spin2 + c[..., 13]*spin3 + c[..., 14]*spin4
        + e * (c[..., 15] + c[..., 16]*spin + c[..., 17]*spin2 + c[..., 18]*spin3 + c[..., 19]*spin4)
        + e2 * (c[..., 20] + c[..., 21]*spin + c[..., 22]*spin2 + c[..., 23]*spin3 + c[..., 24]*spin4)
        + e3 * (c[..., 25] + c[..., 26]*spin + c[..., 27]*spin2 + c[..., 28]*spin3 + c[..., 29]*spin4)
        + e4 * (c[..., 30] + c[..., 31]*spin + c[..., 32]*spin2 + c[..., 33]*spin3 + c[..., 34]*spin4)
    )
    den = c[..., 35] + c[..., 36]*spin + c[..., 37]*spin2 + c[..., 38]*spin3
    equal_spin = num / den

    unequal_spin = (
        ca * d * e * (
            c[..., 39] + c[..., 40]*e + c[..., 41]*e2 + c[..., 42]*e3 + c[..., 43]*e4 + c[..., 44]*e5
            + c[..., 45]*spin + c[..., 46]*spin*e2 + c[..., 47]*spin*e3
        )
        + c[..., 48] * ca2 * e
    )

    return no_spin + equal_spin + unequal_spin


def _compute_amp_fit_rows_batch(eta, S, StotR, chia, delta, amp_coeffs):
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta

    spin = torch.cat((S.unsqueeze(1).expand(-1, 3), StotR.unsqueeze(1).expand(-1, 4)), dim=1)
    spin2 = spin * spin
    spin3 = spin2 * spin
    spin4 = spin3 * spin
    spin5 = spin4 * spin

    c = amp_coeffs.unsqueeze(0)
    e = eta.unsqueeze(1)
    e2 = eta2.unsqueeze(1)
    e3 = eta3.unsqueeze(1)
    e4 = eta4.unsqueeze(1)
    e5 = eta5.unsqueeze(1)
    ca = chia.unsqueeze(1)
    d = delta.unsqueeze(1)

    no_spin = (c[..., 0] + c[..., 1]*e + c[..., 2]*e2 + c[..., 3]*e3 + c[..., 4]*e4) / (
        c[..., 5] + c[..., 6]*e + c[..., 7]*e2
    )

    num_s0 = c[..., 8] + c[..., 9]*e + c[..., 10]*e2 + c[..., 11]*e3
    num_s1 = c[..., 12] + c[..., 13]*e + c[..., 14]*e2 + c[..., 15]*e3
    num_s2 = c[..., 16] + c[..., 17]*e + c[..., 18]*e2 + c[..., 19]*e3
    num_s3 = c[..., 20] + c[..., 21]*e + c[..., 22]*e2 + c[..., 23]*e3
    num_s4 = c[..., 24] + c[..., 25]*e + c[..., 26]*e2 + c[..., 27]*e3
    num_s5 = c[..., 28] + c[..., 29]*e + c[..., 30]*e2 + c[..., 31]*e3

    equal_spin = (num_s0 + num_s1*spin + num_s2*spin2 + num_s3*spin3 + num_s4*spin4 + num_s5*spin5) / (
        c[..., 32] + c[..., 33]*spin + c[..., 34]*e + c[..., 35]*spin2
    )

    unequal_spin = ca * d * (c[..., 36] + c[..., 37]*e + c[..., 38]*e2 + c[..., 39]*e3 + c[..., 40]*e4 + c[..., 41]*e5)

    return no_spin + equal_spin + unequal_spin


def _build_packed_frequency_plans_batch(
    m1_eff, m2_eff, s1z_eff, s2z_eff, dist, coa_phase, f_ref, f_lower, phase_coeffs, amp_coeffs,
    chip=None, final_spin=None
):
    m1_eff = torch.atleast_1d(m1_eff)
    m2_eff = torch.atleast_1d(m2_eff)
    s1z_eff = torch.atleast_1d(s1z_eff)
    s2z_eff = torch.atleast_1d(s2z_eff)
    dist = torch.atleast_1d(dist)
    coa_phase = torch.atleast_1d(coa_phase)
    f_ref = torch.atleast_1d(f_ref)
    if chip is not None:
        chip = torch.atleast_1d(chip)
    else:
        chip = torch.zeros_like(s1z_eff)
    if final_spin is not None:
        final_spin = torch.atleast_1d(final_spin)

    device = m1_eff.device
    dtype = m1_eff.dtype

    m1_s = m1_eff * MTSUN
    m2_s = m2_eff * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s * M_s)
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    delta = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=0.0))

    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)
    chi_eff = mm1 * s1z_eff + mm2 * s2z_eff
    S = (chi_eff - (38.0 / 113.0) * eta * (s1z_eff + s2z_eff)) / (1.0 - (76.0 * eta / 113.0))
    StotR = (mm1**2 * s1z_eff + mm2**2 * s2z_eff) / (mm1**2 + mm2**2)
    chia = s1z_eff - s2z_eff

    fMs_RD_al, fMs_damp_al, fMs_MECO_al, fMs_ISCO_al = IMRPhenomX_utils.get_cutoff_fMs(
        m1_eff, m2_eff, s1z_eff, s2z_eff
    )
    fMs_RD_al = torch.atleast_1d(torch.as_tensor(fMs_RD_al, dtype=dtype, device=device))
    fMs_damp_al = torch.atleast_1d(torch.as_tensor(fMs_damp_al, dtype=dtype, device=device))
    fMs_MECO_al = torch.atleast_1d(torch.as_tensor(fMs_MECO_al, dtype=dtype, device=device))
    fMs_ISCO_al = torch.atleast_1d(torch.as_tensor(fMs_ISCO_al, dtype=dtype, device=device))

    fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO = IMRPhenomX_utils.get_cutoff_fMs(
        m1_eff, m2_eff, s1z_eff, s2z_eff, chip=chip, final_spin=final_spin
    )
    fMs_RD = torch.atleast_1d(torch.as_tensor(fMs_RD, dtype=dtype, device=device))
    fMs_damp = torch.atleast_1d(torch.as_tensor(fMs_damp, dtype=dtype, device=device))
    fMs_MECO = torch.atleast_1d(torch.as_tensor(fMs_MECO, dtype=dtype, device=device))
    fMs_ISCO = torch.atleast_1d(torch.as_tensor(fMs_ISCO, dtype=dtype, device=device))

    fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
    fMs_INmatch = fMs_MECO
    deltafMs = (fMs_IMmatch - fMs_INmatch) * 0.03
    f1_Ms = fMs_INmatch - 1.0 * deltafMs
    f2_Ms = fMs_IMmatch + 0.5 * deltafMs

    phase_rows = _compute_phase_fit_rows_batch(eta, S, StotR, chia, delta, phase_coeffs)
    amp_rows = _compute_amp_fit_rows_batch(eta, S, StotR, chia, delta, amp_coeffs)

    # 1. Inspiral Phase (TF2 PN + 4x4 solve)
    chi1 = s1z_eff
    chi2 = s2z_eff
    chi1L2L = chi1 * chi2
    chi1L2 = chi1 * chi1
    chi1L3 = chi1L2 * chi1
    chi2L2 = chi2 * chi2
    chi2L3 = chi2L2 * chi2

    phi0 = torch.ones_like(eta)
    phi1 = torch.zeros_like(eta)
    phi2 = (3715.0 / 756.0 + (55.0 * eta) / 9.0) * (PI ** (2.0 / 3.0))
    phi3 = -16.0 * (PI**2) + ((113.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta) - 76.0 * (chi1 + chi2) * eta) / 6.0) * PI
    phi4 = (15293365.0 / 508032.0 + (27145.0 * eta) / 504.0 + (3085.0 * eta2) / 72.0) * (PI ** (4.0 / 3.0)) + (
        (-5.0 * (81.0 * chi1L2 * (1 + delta - 2 * eta) + 316.0 * chi1L2L * eta - 81.0 * chi2L2 * (-1 + delta + 2 * eta))) / 16.0
    ) * (PI ** (4.0 / 3.0))
    phi5 = torch.zeros_like(eta)
    phi5L = ((5.0 * (46374.0 - 6552.0 * eta) * PI) / 4536.0) * (PI ** (5.0 / 3.0)) + (
        (-732985.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta) - 560.0 * (-1213.0 * (chi1 + chi2) + 63.0 * (chi1 - chi2) * delta) * eta + 85680.0 * (chi1 + chi2) * eta2) / 4536.0
    ) * (PI ** (5.0 / 3.0))
    phi6L = torch.full_like(eta, (-6848.0 / 63.0) * (PI**2.0))
    phi6 = (
        (11583231236531.0 / 4.69421568e9 - (5.0 * eta * (3147553127.0 + 588.0 * eta * (-45633.0 + 102260.0 * eta))) / 3.048192e6 - (6848.0 * EULERGAMMA) / 21.0 - (640.0 * PI**2.0) / 3.0 + (2255.0 * eta * PI**2.0) / 12.0 - (13696.0 * math.log(2.0)) / 21.0 - (6848.0 * math.log(PI)) / 63.0) * (PI**2.0)
        + ((5 * (227.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta) - 156.0 * (chi1 + chi2) * eta) * PI) / 3.0) * (PI**2.0)
        + ((5.0 * (20.0 * chi1L2L * eta * (11763.0 + 12488.0 * eta) + 7.0 * chi2L2 * (-15103.0 * (-1 + delta) + 2.0 * (-21683.0 + 6580.0 * delta) * eta - 9808.0 * eta2) - 7.0 * chi1L2 * (-15103.0 * (1 + delta) + 2.0 * (21683.0 + 6580.0 * delta) * eta + 9808.0 * eta2))) / 4032.0) * (PI**2.0)
    )
    phi7 = (
        ((5.0 * (15419335.0 + 168.0 * (75703.0 - 29618.0 * eta) * eta) * PI) / 254016.0) * (PI ** (7.0 / 3.0))
        + ((5.0 * (-5030016755.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta) + 4.0 * (2113331119.0 * (chi1 + chi2) + 675484362.0 * (chi1 - chi2) * delta) * eta - 1008.0 * (208433.0 * (chi1 + chi2) + 25011.0 * (chi1 - chi2) * delta) * eta2 + 90514368.0 * (chi1 + chi2) * eta3)) / 6.096384e6) * (PI ** (7.0 / 3.0))
        + (-5.0 * (57.0 * chi1L2 * (1 + delta - 2 * eta) + 220.0 * chi1L2L * eta - 57.0 * chi2L2 * (-1 + delta + 2 * eta)) * PI) * (PI ** (7.0 / 3.0))
        + ((14585.0 * (-(chi2L3 * (-1.0 + delta)) + chi1L3 * (1.0 + delta)) - 5.0 * (chi2L3 * (8819.0 - 2985.0 * delta) + 8439.0 * chi1 * chi2L2 * (-1.0 + delta) - 8439.0 * chi1L2 * chi2 * (1.0 + delta) + chi1L3 * (8819.0 + 2985.0 * delta)) * eta + 40.0 * (chi1 + chi2) * (17.0 * chi1L2 - 14.0 * chi1L2L + 17.0 * chi2L2) * eta2) / 48.0) * (PI ** (7.0 / 3.0))
    )
    phi8 = (
        (-5.0 * (1263141.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta) - 2.0 * (794075.0 * (chi1 + chi2) + 178533.0 * (chi1 - chi2) * delta) * eta + 94344.0 * (chi1 + chi2) * eta2) * PI * (-1.0 + math.log(PI)) / 9072.0) * (PI ** (8.0 / 3.0))
    )
    phi8L = (
        (-5.0 * (1263141.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta) - 2.0 * (794075.0 * (chi1 + chi2) + 178533.0 * (chi1 - chi2) * delta) * eta + 94344.0 * (chi1 + chi2) * eta2) * PI / 9072.0) * (PI ** (8.0 / 3.0))
    )

    fMs_PhaseInsMin = 0.0026
    fMs_PhaseInsMax = 1.020 * fMs_MECO
    deltax_ins = fMs_PhaseInsMax - fMs_PhaseInsMin
    xmin_ins = fMs_PhaseInsMin
    gpoints4 = torch.as_tensor([0.0, 0.25, 0.75, 1.0], dtype=dtype, device=device).unsqueeze(0)
    CP_phase_Ins = gpoints4 * deltax_ins.unsqueeze(1) + xmin_ins

    CV_phase_Ins0 = phase_rows[:, 0] + phase_rows[:, 2]
    CV_phase_Ins1 = phase_rows[:, 1] + phase_rows[:, 2]
    CV_phase_Ins2 = phase_rows[:, 2]
    CV_phase_Ins3 = phase_rows[:, 3] + phase_rows[:, 2]

    A_ins = torch.stack((
        torch.ones_like(CP_phase_Ins),
        CP_phase_Ins ** (1.0 / 3.0),
        CP_phase_Ins ** (2.0 / 3.0),
        CP_phase_Ins,
    ), dim=2)
    b_ins = torch.stack((CV_phase_Ins0, CV_phase_Ins1, CV_phase_Ins2, CV_phase_Ins3), dim=1)
    coeffs_Ins = torch.linalg.solve(A_ins, b_ins)

    sigma1 = (-5.0 / 3.0) * coeffs_Ins[:, 0]
    sigma2 = (-5.0 / 4.0) * coeffs_Ins[:, 1]
    sigma3 = (-5.0 / 5.0) * coeffs_Ins[:, 2]
    sigma4 = (-5.0 / 6.0) * coeffs_Ins[:, 3]

    inspiral_phase = torch.stack((
        phi0, phi1, phi2, phi3, phi4, phi5, phi5L, phi6, phi6L, phi7, phi8, phi8L,
        sigma1, sigma2, sigma3, sigma4
    ), dim=1)

    # 2. Ringdown Phase solve (5x5)
    fMs_PhaseRDMin = fMs_IMmatch
    fMs_PhaseRDMax = fMs_RD + 1.25 * fMs_damp
    dphase0 = 5.0 / (128.0 * (PI ** (5.0 / 3.0)))

    gpoints5 = torch.as_tensor(
        [0.0, 0.5 - 0.5 / math.sqrt(2.0), 0.5, 0.5 + 0.5 / math.sqrt(2.0), 1.0],
        dtype=dtype, device=device
    ).unsqueeze(0)

    deltax_rd = fMs_PhaseRDMax - fMs_PhaseRDMin
    xmin_rd = fMs_PhaseRDMin
    CP_phase_RD_base = gpoints5 * deltax_rd.unsqueeze(1) + xmin_rd.unsqueeze(1)
    CP_phase_RD = torch.stack((
        CP_phase_RD_base[:, 0],
        CP_phase_RD_base[:, 1],
        CP_phase_RD_base[:, 2],
        fMs_RD,
        CP_phase_RD_base[:, 4]
    ), dim=1)

    CV_phase_RD0 = phase_rows[:, 8]
    CV_phase_RD1 = phase_rows[:, 9]
    CV_phase_RD2 = phase_rows[:, 10]
    CV_phase_RD3 = phase_rows[:, 11]
    CV_phase_RD4 = phase_rows[:, 12]

    CV_phase_RD4 = CV_phase_RD4 + CV_phase_RD3
    CV_phase_RD2 = CV_phase_RD2 + CV_phase_RD3
    CV_phase_RD1 = CV_phase_RD1 + CV_phase_RD3
    CV_phase_RD0 = CV_phase_RD0 + CV_phase_RD1

    offset_rd = CP_phase_RD - fMs_RD.unsqueeze(1)
    A_rd = torch.stack((
        torch.ones_like(CP_phase_RD),
        CP_phase_RD ** (-1.0 / 3.0),
        CP_phase_RD ** (-2.0),
        CP_phase_RD ** (-4.0),
        -(dphase0) / (fMs_damp.unsqueeze(1) * fMs_damp.unsqueeze(1) + offset_rd * offset_rd),
    ), dim=2)
    b_rd = torch.stack((CV_phase_RD0, CV_phase_RD1, CV_phase_RD2, CV_phase_RD3, CV_phase_RD4), dim=1)
    coeffs_RD = torch.linalg.solve(A_rd, b_rd)

    c0 = coeffs_RD[:, 0]
    c1 = coeffs_RD[:, 1]
    c2 = coeffs_RD[:, 2]
    c4 = coeffs_RD[:, 3]
    cRD = coeffs_RD[:, 4]
    cL = -(dphase0 * cRD)
    c4ov3 = c4 / 3.0
    cLovfda = cL / fMs_damp

    mergerringdown_phase = torch.stack((
        c0, c1, c2, c4ov3, cLovfda, fMs_RD, fMs_damp, cL, CV_phase_RD0
    ), dim=1)

    # 3. Intermediate Phase solve (5x5)
    def eval_ins_phase_and_deriv(f_val):
        f13 = f_val ** (1.0 / 3.0)
        f23 = f13 * f13
        f43 = f_val * f13
        f53 = f_val * f23
        f2 = f_val * f_val
        f73 = f2 * f13
        f83 = f2 * f23
        f3 = f2 * f_val
        f103 = f3 * f13
        f113 = f3 * f23
        log_f = torch.log(f_val)
        ph_tf2 = (
            phi0 + phi1*f13 + phi2*f23 + phi3*f_val + phi4*f43 + phi5*f53 + phi5L*f53*log_f
            + phi6*f2 + phi6L*f2*log_f + phi7*f73 + phi8*f83 + phi8L*f83*log_f
        )
        ph_ins = ph_tf2 + (sigma1*f83 + sigma2*f3 + sigma3*f103 + sigma4*f113)
        norm = -(3.0 * (PI ** (-5.0 / 3.0))) / 128.0
        val = ph_ins * norm / f53

        dph_tf2 = (
            phi1*(1.0/3.0)*(f13**-2) + phi2*(2.0/3.0)*(f13**-1) + phi3 + phi4*(4.0/3.0)*f13
            + phi5*(5.0/3.0)*f23 + phi5L*(5.0/3.0)*f23*log_f + phi5L*f23
            + phi6*2.0*f_val + phi6L*2.0*f_val*log_f + phi6L*f_val
            + phi7*(7.0/3.0)*f43 + phi8*(8.0/3.0)*f53 + phi8L*(8.0/3.0)*f53*log_f + phi8L*f53
        )
        dph_ins = dph_tf2 + (sigma1*(8.0/3.0)*f53 + sigma2*3.0*f2 + sigma3*(10.0/3.0)*f73 + sigma4*(11.0/3.0)*f83)
        dval = norm * (dph_ins / f53 - (5.0 / 3.0) * ph_ins / (f53 * f_val))
        return val, dval

    phi_Ins_match_f1, dphi_Ins_match_f1 = eval_ins_phase_and_deriv(f1_Ms)

    def eval_mrd_phase_and_deriv(f_val):
        offset = f_val - fMs_RD
        val = (
            c0 * f_val + 1.5 * c1 * (f_val ** (2.0 / 3.0)) - c2 * (f_val ** -1.0)
            - c4ov3 * (f_val ** -3.0) + cLovfda * torch.atan(offset / fMs_damp)
        )
        dval = (
            c0 + c1 * (f_val ** (-1.0 / 3.0)) + c2 * (f_val ** -2.0)
            + 3.0 * c4ov3 * (f_val ** -4.0) + cL / (fMs_damp * fMs_damp + offset * offset)
        )
        return val, dval

    phi_MRD_match_f2, dphi_MRD_match_f2 = eval_mrd_phase_and_deriv(f2_Ms)

    fMs_PhaseMatchIN = f1_Ms
    fPhaseMatchIM = f2_Ms
    deltax_int = fPhaseMatchIM - fMs_PhaseMatchIN
    xmin_int = fMs_PhaseMatchIN
    CP_phase_Int = gpoints5 * deltax_int.unsqueeze(1) + xmin_int.unsqueeze(1)

    v2IMmRDv4 = phase_rows[:, 4]
    v3IMmRDv4 = phase_rows[:, 5]
    v2IM = phase_rows[:, 6]
    d43 = phase_rows[:, 7]
    CV_phase_RD3 = phase_rows[:, 11]

    CV_phase_Int0 = dphi_Ins_match_f1
    CV_phase_Int1 = 0.75 * (v2IMmRDv4 + CV_phase_RD3) + 0.25 * v2IM
    CV_phase_Int2 = v3IMmRDv4 + CV_phase_RD3
    CV_phase_Int3 = d43 + CV_phase_Int2
    CV_phase_Int4 = CV_phase_RD0

    ratio_int = fMs_RD.unsqueeze(1) / CP_phase_Int
    A_int = torch.stack((
        torch.ones_like(CP_phase_Int),
        ratio_int,
        ratio_int * ratio_int,
        ratio_int ** 3,
        ratio_int ** 4,
    ), dim=2)

    collocation_int = torch.stack((
        CV_phase_Int0, CV_phase_Int1, CV_phase_Int2, CV_phase_Int3, CV_phase_Int4
    ), dim=1)
    offset_int = CP_phase_Int - fMs_RD.unsqueeze(1)
    b_int = collocation_int - (4.0 * cL.unsqueeze(1)) / (
        4.0 * (fMs_damp.unsqueeze(1)**2) + offset_int * offset_int
    )
    coeffs_Int = torch.linalg.solve(A_int, b_int)

    a0_int = coeffs_Int[:, 0]
    a1_int = coeffs_Int[:, 1]
    a2_int = coeffs_Int[:, 2]
    a3_int = coeffs_Int[:, 3]
    a4_int = coeffs_Int[:, 4]

    b0 = a0_int
    b1 = a1_int * fMs_RD
    b2 = a2_int * (fMs_RD**2)
    b3 = a3_int * (fMs_RD**3)
    b4 = a4_int * (fMs_RD**4)

    intermediate_phase = torch.stack((
        b0, b1, b2, b3, b4, cL, fMs_RD, fMs_damp
    ), dim=1)

    def eval_int_phase_and_deriv(f_val):
        offset = (f_val - fMs_RD) / (2.0 * fMs_damp)
        val = (
            b0 * f_val + b1 * torch.log(f_val) - b2 * (f_val ** -1.0)
            - 0.5 * b3 * (f_val ** -2.0) - (b4 / 3.0) * (f_val ** -3.0)
            + (2.0 * cL * torch.atan(offset)) / fMs_damp
        )
        dval = (
            b0 + b1 * (f_val ** -1.0) + b2 * (f_val ** -2.0)
            + b3 * (f_val ** -3.0) + b4 * (f_val ** -4.0)
            + (4.0 * cL) / (4.0 * fMs_damp * fMs_damp + (f_val - fMs_RD)**2)
        )
        return val, dval

    phi_Int_match_f1, dphi_Int_match_f1 = eval_int_phase_and_deriv(f1_Ms)
    alpha1 = dphi_Ins_match_f1 - dphi_Int_match_f1
    alpha0 = phi_Ins_match_f1 - phi_Int_match_f1 - alpha1 * f1_Ms

    phi_Int_match_f2, dphi_Int_match_f2 = eval_int_phase_and_deriv(f2_Ms)
    beta1 = dphi_Int_match_f2 + alpha1 - dphi_MRD_match_f2
    beta0 = phi_Int_match_f2 + alpha1 * f2_Ms + alpha0 - phi_MRD_match_f2 - beta1 * f2_Ms

    # 4. Inspiral Amplitude
    A0_amp = torch.ones_like(eta)
    A2_amp = ((-969.0 + 1804.0 * eta) / 672.0) * (PI ** (2.0 / 3.0))
    A3_amp = ((81.0 * (chi1 + chi2 + chi1*delta - chi2*delta) - 44.0 * (chi1 + chi2) * eta) / 48.0) * PI
    A4_amp = (
        (-27312085.0 - 10287648.0 * chi1**2 * (1.0 + delta) + 24.0 * (428652.0 * chi2**2 * (-1 + delta) + (-1975055.0 + 10584.0 * (81.0 * chi1**2 - 94.0 * chi1 * chi2 + 81.0 * chi2**2)) * eta + 1473794.0 * eta2))
        / 8.128512e6
    ) * (PI ** (4.0 / 3.0))
    A5_amp = (
        (-6048.0 * chi1**3 * (-1.0 - delta + (3.0 + delta) * eta) + chi2 * (-((287213.0 + 6048.0 * chi2**2) * (-1.0 + delta)) + 4 * (-93414.0 + 1512.0 * chi2**2 * (-3.0 + delta) + 2083.0 * delta) * eta - 35632.0 * eta2) + chi1 * (287213.0 * (1.0 + delta) - 4.0 * eta * (93414.0 + 2083.0 * delta + 8908.0 * eta)) + 42840.0 * (-1.0 + 4.0 * eta) * PI)
        / 32256.0
    ) * (PI ** (5.0 / 3.0))
    A6_amp = (
        (-1242641879927.0 + 12.0 * (28.0 * (-3248849057.0 + 11088.0 * (163199.0 * chi1**2 - 266498.0 * chi1 * chi2 + 163199.0 * chi2**2)) * eta2 + 27026893936.0 * eta3 - 116424.0 * (147117.0 * (-(chi2**2 * (-1.0 + delta)) + chi1**2 * (1.0 + delta)) + 60928.0 * (chi1 + chi2 + chi1 * delta - chi2 * delta) * PI) + eta * (545384828789.0 - 77616.0 * (638642.0 * chi1 * chi2 + chi1**2 * (-158633.0 + 282718.0 * delta) - chi2**2 * (158633.0 + 282718.0 * delta) - 107520.0 * (chi1 + chi2) * PI + 275520.0 * PI**2))))
        / 6.0085960704e10
    ) * (PI**2)

    fMs_AmpMatchIN = fMs_MECO_al + 0.25 * (fMs_ISCO_al - fMs_MECO_al)
    CP_Amp_Ins0 = 0.50 * fMs_AmpMatchIN
    CP_Amp_Ins1 = 0.75 * fMs_AmpMatchIN
    CP_Amp_Ins2 = 1.00 * fMs_AmpMatchIN

    CV_Amp_Ins0 = amp_rows[:, 0]
    CV_Amp_Ins1 = amp_rows[:, 1]
    CV_Amp_Ins2 = amp_rows[:, 2]

    cbrt0 = CP_Amp_Ins0 ** (1.0 / 3.0)
    cbrt1 = CP_Amp_Ins1 ** (1.0 / 3.0)
    cbrt2 = CP_Amp_Ins2 ** (1.0 / 3.0)

    den_rho = (CP_Amp_Ins0 ** (7.0 / 3.0)) * (cbrt0 - cbrt1) * (CP_Amp_Ins1 ** (7.0 / 3.0)) * (cbrt0 - cbrt2) * (cbrt1 - cbrt2) * (CP_Amp_Ins2 ** (7.0 / 3.0))
    rho1 = (
        -(CP_Amp_Ins1 ** (8.0 / 3.0)) * (CP_Amp_Ins2**3) * CV_Amp_Ins0 + (CP_Amp_Ins1**3) * (CP_Amp_Ins2 ** (8.0 / 3.0)) * CV_Amp_Ins0
        + (CP_Amp_Ins0 ** (8.0 / 3.0)) * (CP_Amp_Ins2**3) * CV_Amp_Ins1 - (CP_Amp_Ins0**3) * (CP_Amp_Ins2 ** (8.0 / 3.0)) * CV_Amp_Ins1
        - (CP_Amp_Ins0 ** (8.0 / 3.0)) * (CP_Amp_Ins1**3) * CV_Amp_Ins2 + (CP_Amp_Ins0**3) * (CP_Amp_Ins1 ** (8.0 / 3.0)) * CV_Amp_Ins2
    ) / den_rho
    rho2 = (
        (CP_Amp_Ins1 ** (7.0 / 3.0)) * (CP_Amp_Ins2**3) * CV_Amp_Ins0 - (CP_Amp_Ins1**3) * (CP_Amp_Ins2 ** (7.0 / 3.0)) * CV_Amp_Ins0
        - (CP_Amp_Ins0 ** (7.0 / 3.0)) * (CP_Amp_Ins2**3) * CV_Amp_Ins1 + (CP_Amp_Ins0**3) * (CP_Amp_Ins2 ** (7.0 / 3.0)) * CV_Amp_Ins1
        + (CP_Amp_Ins0 ** (7.0 / 3.0)) * (CP_Amp_Ins1**3) * CV_Amp_Ins2 - (CP_Amp_Ins0**3) * (CP_Amp_Ins1 ** (7.0 / 3.0)) * CV_Amp_Ins2
    ) / den_rho
    rho3 = (
        (CP_Amp_Ins1 ** (8.0 / 3.0)) * (CP_Amp_Ins2 ** (7.0 / 3.0)) * CV_Amp_Ins0 - (CP_Amp_Ins1 ** (7.0 / 3.0)) * (CP_Amp_Ins2 ** (8.0 / 3.0)) * CV_Amp_Ins0
        - (CP_Amp_Ins0 ** (8.0 / 3.0)) * (CP_Amp_Ins2 ** (7.0 / 3.0)) * CV_Amp_Ins1 + (CP_Amp_Ins0 ** (7.0 / 3.0)) * (CP_Amp_Ins2 ** (8.0 / 3.0)) * CV_Amp_Ins1
        + (CP_Amp_Ins0 ** (8.0 / 3.0)) * (CP_Amp_Ins1 ** (7.0 / 3.0)) * CV_Amp_Ins2 - (CP_Amp_Ins0 ** (7.0 / 3.0)) * (CP_Amp_Ins1 ** (8.0 / 3.0)) * CV_Amp_Ins2
    ) / den_rho

    amp_inspiral = torch.stack((
        A0_amp, A2_amp, A3_amp, A4_amp, A5_amp, A6_amp, rho1, rho2, rho3
    ), dim=1)

    # 5. Ringdown Amplitude
    gamma2 = amp_rows[:, 4]
    gamma3 = amp_rows[:, 5]
    v1RD = amp_rows[:, 6]

    fMs_AmpRDMin = torch.where(
        gamma2 <= 1.0,
        torch.abs(fMs_RD + fMs_damp * gamma3 * (torch.sqrt(torch.clamp(1.0 - gamma2 * gamma2, min=0.0)) - 1.0) / gamma2),
        torch.abs(fMs_RD + fMs_damp * (-1.0) * gamma3 / gamma2),
    )
    FMs1_amp = fMs_AmpMatchIN
    FMs4_amp = fMs_AmpRDMin

    gamma1 = (
        (v1RD / (fMs_damp * gamma3))
        * (FMs4_amp * FMs4_amp - 2.0 * FMs4_amp * fMs_RD + fMs_RD * fMs_RD + fMs_damp * fMs_damp * gamma3 * gamma3)
        * torch.exp(((FMs4_amp - fMs_RD) * gamma2) / (fMs_damp * gamma3))
    )
    gammaR = gamma2 / (fMs_damp * gamma3)
    gammaD2 = (gamma3 * fMs_damp) * (gamma3 * fMs_damp)
    gammaD13 = fMs_damp * gamma1 * gamma3

    amp_ringdown = torch.stack((
        fMs_RD, gammaR, gammaD2, gammaD13, fMs_AmpRDMin
    ), dim=1)

    # 6. Intermediate Amplitude (delta0..delta4)
    def eval_ins_amp_and_deriv(f_val):
        val = (
            A0_amp + A2_amp * (f_val ** (2.0 / 3.0)) + A3_amp * f_val + A4_amp * (f_val ** (4.0 / 3.0))
            + A5_amp * (f_val ** (5.0 / 3.0)) + A6_amp * (f_val ** 2.0)
            + rho1 * (f_val ** (7.0 / 3.0)) + rho2 * (f_val ** (8.0 / 3.0)) + rho3 * (f_val ** 3.0)
        )
        dval = (
            A2_amp * (2.0 / 3.0) * (f_val ** (-1.0 / 3.0)) + A3_amp + A4_amp * (4.0 / 3.0) * (f_val ** (1.0 / 3.0))
            + A5_amp * (5.0 / 3.0) * (f_val ** (2.0 / 3.0)) + A6_amp * 2.0 * f_val
            + rho1 * (7.0 / 3.0) * (f_val ** (4.0 / 3.0)) + rho2 * (8.0 / 3.0) * (f_val ** (5.0 / 3.0)) + rho3 * 3.0 * (f_val ** 2.0)
        )
        return val, dval

    inspFMs1, d1_raw = eval_ins_amp_and_deriv(FMs1_amp)

    def eval_mrd_amp_and_deriv(f_val):
        offset = f_val - fMs_RD
        exp_factor = torch.exp((-offset) * gammaR)
        num = exp_factor * gammaD13
        den = offset * offset + gammaD2
        val = num / den
        dnum = -(gammaR * exp_factor * gammaD13)
        dden = 2.0 * offset
        dval = (dnum * den - num * dden) / (den * den)
        return val, dval

    rdFMs4, d4_raw = eval_mrd_amp_and_deriv(FMs4_amp)

    d1 = ((7.0 / 6.0) * (FMs1_amp ** (1.0 / 6.0)) / inspFMs1) - ((FMs1_amp ** (7.0 / 6.0)) * d1_raw / (inspFMs1 * inspFMs1))
    d4 = ((7.0 / 6.0) * (FMs4_amp ** (1.0 / 6.0)) / rdFMs4) - ((FMs4_amp ** (7.0 / 6.0)) * d4_raw / (rdFMs4 * rdFMs4))

    FMs2_amp = FMs1_amp + 0.5 * (FMs4_amp - FMs1_amp)
    V1 = 1.0 / ((FMs1_amp ** (-7.0 / 6.0)) * inspFMs1)
    V2 = 1.0 / amp_rows[:, 3]
    V4 = 1.0 / ((FMs4_amp ** (-7.0 / 6.0)) * rdFMs4)

    F12 = FMs1_amp * FMs1_amp
    F13 = F12 * FMs1_amp
    F14 = F13 * FMs1_amp
    F15 = F14 * FMs1_amp

    F22 = FMs2_amp * FMs2_amp
    F23 = F22 * FMs2_amp
    F24 = F23 * FMs2_amp

    F42 = FMs4_amp * FMs4_amp
    F43 = F42 * FMs4_amp
    F44 = F43 * FMs4_amp
    F45 = F44 * FMs4_amp

    F1mF2 = FMs1_amp - FMs2_amp
    F1mF4 = FMs1_amp - FMs4_amp
    F2mF4 = FMs2_amp - FMs4_amp

    F1mF22 = F1mF2 * F1mF2
    F2mF42 = F2mF4 * F2mF4
    F1mF43 = F1mF4 * F1mF4 * F1mF4
    den_delta = F1mF22 * F1mF43 * F2mF42

    delta0 = (
        -(d4 * F12 * F1mF22 * F1mF4 * FMs2_amp * F2mF4 * FMs4_amp)
        + d1 * FMs1_amp * F1mF2 * F1mF4 * FMs2_amp * F2mF42 * F42
        + F42 * (FMs2_amp * F2mF42 * (-4 * F12 + 3 * FMs1_amp * FMs2_amp + 2 * FMs1_amp * FMs4_amp - FMs2_amp * FMs4_amp) * V1 + F12 * F1mF43 * V2)
        + F12 * F1mF22 * FMs2_amp * (FMs1_amp * FMs2_amp - 2 * FMs1_amp * FMs4_amp - 3 * FMs2_amp * FMs4_amp + 4 * F42) * V4
    ) / den_delta

    delta1 = (
        d4 * FMs1_amp * F1mF22 * F1mF4 * F2mF4 * (2 * FMs2_amp * FMs4_amp + FMs1_amp * (FMs2_amp + FMs4_amp))
        + FMs4_amp * (
            -(d1 * F1mF2 * F1mF4 * F2mF42 * (2 * FMs1_amp * FMs2_amp + (FMs1_amp + FMs2_amp) * FMs4_amp))
            - 2 * FMs1_amp * (F44 * (V1 - V2) + 3 * F24 * (V1 - V4) + F14 * (V2 - V4) + 4 * F23 * FMs4_amp * (-V1 + V4) + 2 * F13 * FMs4_amp * (-V2 + V4) + FMs1_amp * (2 * F43 * (-V1 + V2) + 6 * F22 * FMs4_amp * (V1 - V4) + 4 * F23 * (-V1 + V4)))
        )
    ) / den_delta

    delta2 = (
        -(d4 * F1mF22 * F1mF4 * F2mF4 * (F12 + FMs2_amp * FMs4_amp + 2 * FMs1_amp * (FMs2_amp + FMs4_amp)))
        + d1 * F1mF2 * F1mF4 * F2mF42 * (FMs1_amp * FMs2_amp + 2 * (FMs1_amp + FMs2_amp) * FMs4_amp + F42)
        - 4 * F12 * F23 * V1 + 3 * FMs1_amp * F24 * V1 - 4 * FMs1_amp * F23 * FMs4_amp * V1 + 3 * F24 * FMs4_amp * V1 + 12 * F12 * FMs2_amp * F42 * V1 - 4 * F23 * F42 * V1 - 8 * F12 * F43 * V1 + FMs1_amp * F44 * V1 + F45 * V1
        + F15 * V2 + F14 * FMs4_amp * V2 - 8 * F13 * F42 * V2 + 8 * F12 * F43 * V2 - FMs1_amp * F44 * V2 - F45 * V2
        - F1mF22 * (F13 + FMs2_amp * (3 * FMs2_amp - 4 * FMs4_amp) * FMs4_amp + F12 * (2 * FMs2_amp + FMs4_amp) + FMs1_amp * (3 * FMs2_amp - 4 * FMs4_amp) * (FMs2_amp + 2 * FMs4_amp)) * V4
    ) / den_delta

    delta3 = (
        d4 * F1mF22 * F1mF4 * F2mF4 * (2 * FMs1_amp + FMs2_amp + FMs4_amp)
        - d1 * F1mF2 * F1mF4 * F2mF42 * (FMs1_amp + FMs2_amp + 2 * FMs4_amp)
        + 2 * (F44 * (-V1 + V2) + 2 * F12 * F2mF42 * (V1 - V4) + 2 * F22 * F42 * (V1 - V4) + 2 * F13 * FMs4_amp * (V2 - V4) + F24 * (-V1 + V4) + F14 * (-V2 + V4) + 2 * FMs1_amp * FMs4_amp * (F42 * (V1 - V2) + F22 * (V1 - V4) + 2 * FMs2_amp * FMs4_amp * (-V1 + V4)))
    ) / den_delta

    delta4 = (
        -(d4 * F1mF22 * F1mF4 * F2mF4)
        + d1 * F1mF2 * F1mF4 * F2mF42
        - 3 * FMs1_amp * F22 * V1 + 2 * F23 * V1 + 6 * FMs1_amp * FMs2_amp * FMs4_amp * V1 - 3 * F22 * FMs4_amp * V1 - 3 * FMs1_amp * F42 * V1 + F43 * V1
        + F13 * V2 - 3 * F12 * FMs4_amp * V2 + 3 * FMs1_amp * F42 * V2 - F43 * V2
        - F1mF22 * (FMs1_amp + 2 * FMs2_amp - 3 * FMs4_amp) * V4
    ) / den_delta

    amp_intermediate = torch.stack((delta0, delta1, delta2, delta3, delta4), dim=1)

    # 7. Alignment terms & reference phase
    total_spin = (mm1**2 * chi1 + mm2**2 * chi2) / (mm1**2 + mm2**2)
    spin_diff = chi1 - chi2
    _, linear_b_raw, psi4_to_strain = IMRPhenomX_utils.calc_phaseatpeak(eta, total_spin, spin_diff, delta)
    linear_a = torch.zeros_like(eta)

    f_rd_start = fMs_RD - fMs_damp
    _, dphi_mrd_start = eval_mrd_phase_and_deriv(f_rd_start)
    ringdown_start_derivative = (dphi_mrd_start + beta1) / eta

    linear_b = linear_b_raw - ringdown_start_derivative - 2.0 * PI * (500.0 + psi4_to_strain)

    f_ref_eff = torch.where(f_ref > 0.0, f_ref, torch.full_like(f_ref, f_lower))
    fM_ref = f_ref_eff * M_s

    phi_ins_ref, _ = eval_ins_phase_and_deriv(fM_ref)
    phi_int_ref, _ = eval_int_phase_and_deriv(fM_ref)
    phi_mrd_ref, _ = eval_mrd_phase_and_deriv(fM_ref)

    phi_ref_val = torch.where(
        fM_ref < f1_Ms,
        phi_ins_ref / eta,
        torch.where(
            fM_ref < f2_Ms,
            (phi_int_ref + alpha1 * fM_ref + alpha0) / eta,
            (phi_mrd_ref + beta1 * fM_ref + beta0) / eta,
        ),
    )

    phase_offset = -(phi_ref_val + linear_b * fM_ref + linear_a) + 2.0 * coa_phase + PI / 4.0

    amp_match = FMs1_amp
    amp0 = (2.0 * torch.sqrt(torch.as_tensor(5.0 / (64.0 * PI), dtype=dtype, device=device)) * (M_s**2)) / ((dist * MPC) / C)
    amp_norm = torch.sqrt(2.0 * eta / 3.0) * (PI ** (-1.0 / 6.0))
    overall_amp = amp0 * amp_norm

    time_shift = torch.zeros_like(eta)

    packed_plans = torch.cat((
        M_s.unsqueeze(1),  # 0
        eta.unsqueeze(1),  # 1
        f1_Ms.unsqueeze(1),  # 2
        f2_Ms.unsqueeze(1),  # 3
        inspiral_phase,  # 4..19 (16)
        intermediate_phase,  # 20..27 (8)
        mergerringdown_phase,  # 28..36 (9)
        alpha0.unsqueeze(1),  # 37
        alpha1.unsqueeze(1),  # 38
        beta0.unsqueeze(1),  # 39
        beta1.unsqueeze(1),  # 40
        amp_inspiral,  # 41..49 (9)
        amp_intermediate,  # 50..54 (5)
        amp_ringdown,  # 55..59 (5)
        linear_a.unsqueeze(1),  # 60
        linear_b.unsqueeze(1),  # 61
        phase_offset.unsqueeze(1),  # 62
        time_shift.unsqueeze(1),  # 63
        overall_amp.unsqueeze(1),  # 64
        amp_match.unsqueeze(1),  # 65
    ), dim=1)

    return packed_plans


def imrphenomxas_fd_batch(**params):
    """Generate a batch of IMRPhenomXAS frequency-domain waveforms directly as 2D PyTorch tensors.

    Parameters
    ----------
    mass1 : float or Tensor
        Primary mass in solar masses (shape (B,) or scalar).
    mass2 : float or Tensor
        Secondary mass in solar masses (shape (B,) or scalar).
    spin1z : float or Tensor, optional
        Dimensionless aligned spin of primary (shape (B,) or scalar, default 0.0).
    spin2z : float or Tensor, optional
        Dimensionless aligned spin of secondary (shape (B,) or scalar, default 0.0).
    distance : float or Tensor, optional
        Luminosity distance in Mpc (shape (B,) or scalar, default 1.0).
    inclination : float or Tensor, optional
        Inclination angle in radians (shape (B,) or scalar, default 0.0).
    coa_phase : float or Tensor, optional
        Coalescence phase in radians (shape (B,) or scalar, default 0.0).
    long_asc_nodes : float or Tensor, optional
        Longitude of ascending nodes in radians (shape (B,) or scalar, default 0.0).
    f_ref : float or Tensor, optional
        Reference frequency in Hertz (shape (B,) or scalar, default 0.0, uses f_lower).
    delta_f : float
        Frequency resolution in Hertz.
    f_lower : float
        Lower frequency cutoff in Hertz.
    f_final : float, optional
        Upper frequency cutoff in Hertz. If 0.0 or not provided, uses the maximum
        cutoff frequency across the batch.

    Returns
    -------
    hp : torch.Tensor
        Batch of plus polarizations of shape (B, length) on target device.
    hc : torch.Tensor
        Batch of cross polarizations of shape (B, length) on target device.
    """
    state = _scheme.mgr.state
    device = getattr(state, "torch_device", torch.device("cpu"))
    real_dtype = torch.float32 if getattr(device, "type", "cpu") == "mps" else torch.float64
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128

    batch_size = 1
    for k in (
        "mass1",
        "mass2",
        "spin1z",
        "spin2z",
        "distance",
        "inclination",
        "coa_phase",
        "long_asc_nodes",
        "f_ref",
        "lambda1",
        "lambda2",
        "dquad_mon1",
        "dquad_mon2",
    ):
        v = params.get(k)
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            batch_size = max(batch_size, v.shape[0])
        elif isinstance(v, (list, tuple, _np.ndarray)) and len(v) > 1:
            batch_size = max(batch_size, len(v))

    def _to_tensor(val, default=0.0):
        if val is None:
            val = default
        if isinstance(val, torch.Tensor):
            t = val.to(device=device, dtype=real_dtype)
            if t.ndim == 0:
                t = t.repeat(batch_size)
            return t
        elif isinstance(val, (list, tuple, _np.ndarray)):
            t = torch.as_tensor(val, device=device, dtype=real_dtype)
            if t.ndim == 0:
                t = t.repeat(batch_size)
            elif t.ndim == 1 and t.shape[0] == 1 and batch_size > 1:
                t = t.repeat(batch_size)
            return t
        else:
            return torch.full((batch_size,), float(val), device=device, dtype=real_dtype)

    approximant = params.get("approximant", "IMRPhenomXAS")
    tidal_version = nrtidal_version(approximant)

    m1 = _to_tensor(params["mass1"])
    m2 = _to_tensor(params["mass2"])
    s1z = _to_tensor(params.get("spin1z", 0.0))
    s2z = _to_tensor(params.get("spin2z", 0.0))
    dist = _to_tensor(params.get("distance", 1.0), 1.0)
    incl = _to_tensor(params.get("inclination", 0.0))
    coa_phase = _to_tensor(params.get("coa_phase", 0.0))
    long_asc_nodes = _to_tensor(params.get("long_asc_nodes", 0.0))
    f_ref = _to_tensor(params.get("f_ref", 0.0))
    l1 = _to_tensor(params.get("lambda1", 0.0))
    l2 = _to_tensor(params.get("lambda2", 0.0))
    dquad1 = _to_tensor(params.get("dquad_mon1", 0.0))
    dquad2 = _to_tensor(params.get("dquad_mon2", 0.0))

    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params.get("f_final", 0.0))

    if not all(math.isfinite(value) for value in (delta_f, f_lower, f_final)):
        raise ValueError("IMRPhenomXAS frequencies must be finite")
    if delta_f <= 0.0 or f_lower <= 0.0:
        raise ValueError("IMRPhenomXAS delta_f and f_lower must be positive")
    if f_final < 0.0:
        raise ValueError("IMRPhenomXAS f_final must be non-negative")

    # Swap masses and spins where m2 > m1
    swap_mask = m2 > m1
    m1_eff = torch.where(swap_mask, m2, m1)
    m2_eff = torch.where(swap_mask, m1, m2)
    s1z_eff = torch.where(swap_mask, s2z, s1z)
    s2z_eff = torch.where(swap_mask, s1z, s2z)
    l1_eff = torch.where(swap_mask, l2, l1)
    l2_eff = torch.where(swap_mask, l1, l2)
    dquad1_eff = torch.where(swap_mask, dquad2, dquad1)
    dquad2_eff = torch.where(swap_mask, dquad1, dquad2)

    q1_eff = torch.where((l1_eff > 0.0) & (dquad1_eff == 0.0), nrtidal_quadrupole_from_lambda(l1_eff), 1.0 + dquad1_eff)
    q2_eff = torch.where((l2_eff > 0.0) & (dquad2_eff == 0.0), nrtidal_quadrupole_from_lambda(l2_eff), 1.0 + dquad2_eff)

    phase_coeffs = IMRPhenomX_utils._get_phenomx_phase_coeff_table_cached_master(
        device=device, dtype=real_dtype
    )
    amp_coeffs = IMRPhenomX_utils._get_phenomx_amp_coeff_table_cached_master(
        device=device, dtype=real_dtype
    )

    packed_plans = _build_packed_frequency_plans_batch(
        m1_eff,
        m2_eff,
        s1z_eff,
        s2z_eff,
        dist,
        coa_phase,
        f_ref,
        f_lower,
        phase_coeffs,
        amp_coeffs,
    )

    total_mass_seconds = (m1_eff + m2_eff) * MTSUN
    cutoff_frequency = IMRPhenomX_utils.fM_CUT / total_mass_seconds
    if f_final > 0.0:
        layout_f_max = f_final
        active_f_max = torch.clamp(cutoff_frequency, max=f_final)
    else:
        layout_f_max = float(torch.max(cutoff_frequency).item())
        active_f_max = cutoff_frequency

    if layout_f_max <= f_lower:
        raise ValueError("f_final (or default f_cut) is <= f_lower")

    npts = _next_power_of_two(layout_f_max / delta_f) + 1
    first_bin = int(f_lower / delta_f)
    max_stop_bin = int(layout_f_max / delta_f) + 1

    if first_bin >= max_stop_bin:
        raise ValueError("f_final (or default f_cut) is <= f_lower")

    frequencies = torch.arange(first_bin, max_stop_bin, device=device, dtype=real_dtype) * delta_f
    frequencies_2d = frequencies.unsqueeze(0)

    total_mass_seconds_col = packed_plans[:, 0:1]
    eta_col = packed_plans[:, 1:2]
    phase_lower_col = packed_plans[:, 2:3]
    phase_upper_col = packed_plans[:, 3:4]
    phase_inspiral = packed_plans[:, 4:20]
    phase_intermediate = packed_plans[:, 20:28]
    phase_ringdown = packed_plans[:, 28:37]
    alpha0_col = packed_plans[:, 37:38]
    alpha1_col = packed_plans[:, 38:39]
    beta0_col = packed_plans[:, 39:40]
    beta1_col = packed_plans[:, 40:41]
    amp_inspiral = packed_plans[:, 41:50]
    amp_intermediate = packed_plans[:, 50:55]
    amp_ringdown = packed_plans[:, 55:60]
    linear_a_col = packed_plans[:, 60:61]
    linear_b_col = packed_plans[:, 61:62]
    phase_at_ref_col = packed_plans[:, 62:63]
    time_shift_col = packed_plans[:, 63:64]
    overall_amp_col = packed_plans[:, 64:65]
    amp_match_col = packed_plans[:, 65:66]

    Mf = total_mass_seconds_col * frequencies_2d

    # Inspiral Phase
    f13 = Mf ** (1.0 / 3.0)
    f23 = f13 * f13
    f43 = Mf * f13
    f53 = Mf * f23
    f2 = Mf * Mf
    f73 = f2 * f13
    f83 = f2 * f23
    f3 = f2 * Mf
    f103 = f3 * f13
    f113 = f3 * f23
    log_f = torch.log(Mf)

    phase_tf2 = (
        phase_inspiral[:, 0:1]
        + phase_inspiral[:, 1:2] * f13
        + phase_inspiral[:, 2:3] * f23
        + phase_inspiral[:, 3:4] * Mf
        + phase_inspiral[:, 4:5] * f43
        + phase_inspiral[:, 5:6] * f53
        + phase_inspiral[:, 6:7] * f53 * log_f
        + phase_inspiral[:, 7:8] * f2
        + phase_inspiral[:, 8:9] * f2 * log_f
        + phase_inspiral[:, 9:10] * f73
        + phase_inspiral[:, 10:11] * f83
        + phase_inspiral[:, 11:12] * f83 * log_f
    )
    phase_inspiral_val = phase_tf2 + (
        phase_inspiral[:, 12:13] * f83
        + phase_inspiral[:, 13:14] * f3
        + phase_inspiral[:, 14:15] * f103
        + phase_inspiral[:, 15:16] * f113
    )
    normalization = -(3.0 * PI ** (-5.0 / 3.0)) / 128.0
    phase_ins = phase_inspiral_val * normalization / f53

    # Intermediate Phase
    b0 = phase_intermediate[:, 0:1]
    b1 = phase_intermediate[:, 1:2]
    b2 = phase_intermediate[:, 2:3]
    b3 = phase_intermediate[:, 3:4]
    b4 = phase_intermediate[:, 4:5]
    c_l_int = phase_intermediate[:, 5:6]
    f_rd_int = phase_intermediate[:, 6:7]
    f_damp_int = phase_intermediate[:, 7:8]
    phase_int_raw = (
        b0 * Mf
        + b1 * torch.log(Mf)
        - b2 * (Mf ** -1.0)
        - b3 * (Mf ** -2.0) / 2.0
        - (b4 * (Mf ** -3.0) / 3.0)
        + (2.0 * c_l_int * torch.atan((Mf - f_rd_int) / (2.0 * f_damp_int))) / f_damp_int
    )
    phase_int = phase_int_raw + alpha1_col * Mf + alpha0_col

    # Ringdown Phase
    c0 = phase_ringdown[:, 0:1]
    c1 = phase_ringdown[:, 1:2]
    c2 = phase_ringdown[:, 2:3]
    c4_over_3 = phase_ringdown[:, 3:4]
    c_l_over_f_damp = phase_ringdown[:, 4:5]
    f_rd_rd = phase_ringdown[:, 5:6]
    f_damp_rd = phase_ringdown[:, 6:7]
    phase_rd_raw = (
        c0 * Mf
        + 1.5 * c1 * (Mf ** (2.0 / 3.0))
        - c2 * (Mf ** -1.0)
        - c4_over_3 * (Mf ** -3.0)
        + c_l_over_f_damp * torch.atan((Mf - f_rd_rd) / f_damp_rd)
    )
    phase_rd = phase_rd_raw + beta0_col + beta1_col * Mf

    # 3-region phase piecewise assembly
    half = torch.tensor(0.5, dtype=real_dtype, device=device)
    fM_cut = torch.tensor(IMRPhenomX_utils.fM_CUT, dtype=real_dtype, device=device)

    p_ins_mask = torch.heaviside(phase_lower_col - Mf, half)
    p_int_mask1 = torch.heaviside(Mf - phase_lower_col, half)
    p_int_mask2 = torch.heaviside(phase_upper_col - Mf, half)
    p_rd_mask1 = torch.heaviside(Mf - phase_upper_col, half)
    p_rd_mask2 = torch.heaviside(fM_cut - Mf, half)

    phase = (1.0 / eta_col) * (
        phase_ins * p_ins_mask
        + p_int_mask1 * phase_int * p_int_mask2
        + phase_rd * p_rd_mask1 * p_rd_mask2
    )
    extrinsic_phase = 2.0 * PI * frequencies_2d * time_shift_col
    phase = (
        phase
        + (linear_b_col * Mf)
        + linear_a_col
        + phase_at_ref_col
        - 2.0 * PI
        + extrinsic_phase
    )

    # Inspiral Amplitude
    a0 = amp_inspiral[:, 0:1]
    a2 = amp_inspiral[:, 1:2]
    a3 = amp_inspiral[:, 2:3]
    a4 = amp_inspiral[:, 3:4]
    a5 = amp_inspiral[:, 4:5]
    a6 = amp_inspiral[:, 5:6]
    rho1 = amp_inspiral[:, 6:7]
    rho2 = amp_inspiral[:, 7:8]
    rho3 = amp_inspiral[:, 8:9]
    amplitude_ins = (
        a0
        + a2 * (Mf ** (2.0 / 3.0))
        + a3 * Mf
        + a4 * (Mf ** (4.0 / 3.0))
        + a5 * (Mf ** (5.0 / 3.0))
        + a6 * (Mf ** 2.0)
        + rho1 * (Mf ** (7.0 / 3.0))
        + rho2 * (Mf ** (8.0 / 3.0))
        + rho3 * (Mf ** 3.0)
    )

    # Intermediate Amplitude
    d0 = amp_intermediate[:, 0:1]
    d1 = amp_intermediate[:, 1:2]
    d2 = amp_intermediate[:, 2:3]
    d3 = amp_intermediate[:, 3:4]
    d4 = amp_intermediate[:, 4:5]
    amplitude_int = (Mf ** (7.0 / 6.0)) / (
        d0 + Mf * (d1 + Mf * (d2 + Mf * (d3 + Mf * d4)))
    )

    # Ringdown Amplitude
    f_rd_amp = amp_ringdown[:, 0:1]
    gamma_r = amp_ringdown[:, 1:2]
    gamma_d2 = amp_ringdown[:, 2:3]
    gamma_d13 = amp_ringdown[:, 3:4]
    amp_upper = amp_ringdown[:, 4:5]
    amplitude_rd = (
        torch.exp(-(Mf - f_rd_amp) * gamma_r)
        * gamma_d13
        / ((Mf - f_rd_amp) * (Mf - f_rd_amp) + gamma_d2)
    )

    # 3-region amplitude piecewise assembly
    a_ins_mask = torch.heaviside(amp_match_col - Mf, half)
    a_int_mask1 = torch.heaviside(Mf - amp_match_col, half)
    a_int_mask2 = torch.heaviside(amp_upper - Mf, half)
    a_rd_mask1 = torch.heaviside(Mf - amp_upper, half)
    a_rd_mask2 = torch.heaviside(fM_cut - Mf, half)

    amplitude = (
        amplitude_ins * a_ins_mask
        + a_int_mask1 * amplitude_int * a_int_mask2
        + amplitude_rd * a_rd_mask1 * a_rd_mask2
    )
    amplitude = overall_amp_col * amplitude * (Mf ** (-7.0 / 6.0))

    if tidal_version is not None:
        f_ref_eff = torch.where(f_ref > 0.0, f_ref, torch.full_like(f_ref, f_lower))
        if tidal_version == 3:
            f_merger = nrtidal_merger_frequency_v3(
                m1_eff,
                m2_eff,
                l1_eff,
                l2_eff,
                s1z_eff,
                s2z_eff,
            )
        else:
            f_merger = nrtidal_merger_frequency(
                m1_eff,
                m2_eff,
                l1_eff,
                l2_eff,
            )

        f_align = torch.minimum(active_f_max, f_merger)
        mf_align = total_mass_seconds * f_align

        # Matter Fourier-phase corrections
        phase_tidal = (
            nrtidal_phase(
                frequencies_2d,
                m1_eff,
                m2_eff,
                l1_eff,
                l2_eff,
                tidal_version,
                s1z_eff,
                s2z_eff,
                frequency_series=True,
            )
            + nrtidal_self_spin_phase(
                frequencies_2d,
                m1_eff,
                m2_eff,
                s1z_eff,
                s2z_eff,
                q1_eff - 1.0,
                q2_eff - 1.0,
            )
            + nrtidal_higher_order_spin_phase(
                frequencies_2d,
                m1_eff,
                m2_eff,
                s1z_eff,
                s2z_eff,
                q1_eff,
                q2_eff,
            )
        )

        phase_tidal_ref = (
            nrtidal_phase(
                f_ref_eff,
                m1_eff,
                m2_eff,
                l1_eff,
                l2_eff,
                tidal_version,
                s1z_eff,
                s2z_eff,
                frequency_series=False,
            )
            + nrtidal_self_spin_phase(
                f_ref_eff,
                m1_eff,
                m2_eff,
                s1z_eff,
                s2z_eff,
                q1_eff - 1.0,
                q2_eff - 1.0,
            )
            + nrtidal_higher_order_spin_phase(
                f_ref_eff,
                m1_eff,
                m2_eff,
                s1z_eff,
                s2z_eff,
                q1_eff,
                q2_eff,
            )
        )

        with torch.enable_grad():
            mf_align_grad = mf_align.clone().detach().requires_grad_(True)
            f_eval = mf_align_grad / total_mass_seconds
            phase_eval = (
                nrtidal_phase(
                    f_eval,
                    m1_eff,
                    m2_eff,
                    l1_eff,
                    l2_eff,
                    tidal_version,
                    s1z_eff,
                    s2z_eff,
                    frequency_series=False,
                )
                + nrtidal_self_spin_phase(
                    f_eval,
                    m1_eff,
                    m2_eff,
                    s1z_eff,
                    s2z_eff,
                    q1_eff - 1.0,
                    q2_eff - 1.0,
                )
                + nrtidal_higher_order_spin_phase(
                    f_eval,
                    m1_eff,
                    m2_eff,
                    s1z_eff,
                    s2z_eff,
                    q1_eff,
                    q2_eff,
                )
            )
            tidal_derivative = torch.autograd.grad(phase_eval.sum(), mf_align_grad)[0].detach()

        ins_plan_tuple = tuple(phase_inspiral[:, i] for i in range(16))
        int_plan_tuple = tuple(phase_intermediate[:, i] for i in range(8))
        mrd_plan_tuple = tuple(phase_ringdown[:, i] for i in range(9))

        dphi_ins = _exact_inspiral_phase_value_and_derivative(mf_align, ins_plan_tuple)[1]
        dphi_int = _exact_intermediate_phase_derivative(mf_align, int_plan_tuple, initial_gradient=alpha1_col.squeeze(-1))
        dphi_mrd = _exact_mergerringdown_phase_derivative(mf_align, mrd_plan_tuple) + beta1_col.squeeze(-1)

        base_derivative = torch.where(
            mf_align < phase_lower_col.squeeze(-1),
            dphi_ins / eta_col.squeeze(-1),
            torch.where(
                mf_align < phase_upper_col.squeeze(-1),
                dphi_int / eta_col.squeeze(-1),
                dphi_mrd / eta_col.squeeze(-1),
            ),
        )

        tidal_linb = -base_derivative + tidal_derivative
        linb = linear_b_col.squeeze(-1)
        fM_ref = f_ref_eff * total_mass_seconds

        phase = phase + (tidal_linb - linb)[:, None] * (Mf - fM_ref[:, None])
        phase = phase + phase_tidal_ref[:, None] - phase_tidal

        amp0 = (
            2.0
            * math.sqrt(5.0 / (64.0 * PI))
            * (total_mass_seconds**2)
            / ((dist * MPC) / C)
        )
        amplitude = amplitude + amp0[:, None] * 2.0 * math.sqrt(PI / 5.0) * nrtidal_amplitude(
            frequencies_2d,
            m1_eff,
            m2_eff,
            l1_eff,
            l2_eff,
        )
        amplitude = amplitude * nrtidal_taper(frequencies_2d, f_merger[:, None])

    active_samples = amplitude * torch.exp(1j * phase)

    # Zero out bins beyond active cutoff for each batch item
    stop_bins = (torch.floor(active_f_max / delta_f).to(torch.int64) + (0 if tidal_version is not None else 1)).unsqueeze(1)
    bin_indices = torch.arange(first_bin, max_stop_bin, device=device).unsqueeze(0)
    valid_mask = bin_indices < stop_bins
    active_samples = torch.where(valid_mask, active_samples, torch.zeros_like(active_samples))

    # Polarizations
    cosi = torch.cos(incl).unsqueeze(1)
    plus0 = -0.5 * (1.0 + cosi * cosi) * active_samples
    cross0 = torch.complex(torch.zeros_like(cosi), cosi) * active_samples

    cos_nodes = torch.cos(2.0 * long_asc_nodes).unsqueeze(1)
    sin_nodes = torch.sin(2.0 * long_asc_nodes).unsqueeze(1)

    hp_active = cos_nodes * plus0 + sin_nodes * cross0
    hc_active = cos_nodes * cross0 - sin_nodes * plus0

    hp = torch.zeros((batch_size, npts), dtype=complex_dtype, device=device)
    hc = torch.zeros((batch_size, npts), dtype=complex_dtype, device=device)
    hp[:, first_bin:max_stop_bin] = hp_active
    hc[:, first_bin:max_stop_bin] = hc_active

    return hp, hc
