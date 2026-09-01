# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Optional exact fixed-schema CPU executor for the XAS phase plan.

The generated core replaces rank-zero ATen arithmetic with ordered CPython
binary64 expressions.  It deliberately retains the three tiny ATen linear
solves and bulk-materializes the 66 scalar Tensor leaves.  The route is
default-off, admits only the canonical batch-one CPU schema, performs a cold
raw-byte canary against the eager implementation, and returns ``None`` for the
caller to use its unchanged fallback on any unsupported state or failure.
"""

from __future__ import annotations

from collections import OrderedDict
import math
import os
import platform
import struct
import sys
import threading
import types
from typing import NamedTuple

import torch

from . import _imrphenomxas_fixed_schema_phase_plan_generated as _generated
from .torch_switches import _parse_switch


TENSOR_OUTPUT_COUNT = _generated.TENSOR_OUTPUT_COUNT
SCALAR_OUTPUT_COUNT = _generated.SCALAR_OUTPUT_COUNT
HIDDEN_STORAGE_SPECS = _generated.HIDDEN_STORAGE_SPECS
fixed_schema_core = _generated.fixed_schema_core


_FIXED_SCHEMA_PHASE_PLAN_ENV = (
    "PYCBC_IMRPHENOMXAS_FIXED_SCHEMA_PHASE_PLAN"
)
_FIXED_SCHEMA_PHASE_PLAN_ENVIRONMENT_NAMES = (
    _FIXED_SCHEMA_PHASE_PLAN_ENV,
    "PYCBC_IMRPHENOMX_PHASE_PLAN_BULK_COLLOCATION",
    "PYCBC_IMRPHENOMX_SCALAR_REGION_DISPATCH",
    "PYCBC_IMRPHENOMX_EXACT_SCALAR_DERIVATIVES",
    "PYCBC_IMRPHENOMXAS_SCALAR_DERIVATIVE_PLAN_CSE",
    "PYCBC_IMRPHENOMXAS_INSPIRAL_PHASE_HOST_SCALARS",
    "PYCBC_IMRPHENOMXAS_SCRIPTED_PHASE_ANSATZ_CPU",
    "PYCBC_IMRPHENOMX_PHASE_FIT_PYTHON_SCALARS",
    "PYCBC_IMRPHENOMXAS_PHASE_FIT_NATIVE_ITERATOR",
)
_FIXED_SCHEMA_PHASE_PLAN_CACHE_LIMIT = 4
_SUPPORTED_TORCH_VERSION = (2, 9, 1)
_SUPPORTED_PYTHON_VERSION = (3, 13)
_SUPPORTED_PLATFORM = ("Darwin", "arm64")
_EXPLICIT_CUTOFF_TOPOLOGY_MODE = "explicit-cutoff"
_PUBLIC_DEFAULT_TOPOLOGY_MODE = "public-default"
_FIXED_SCHEMA_PHASE_PLAN_TOPOLOGY_MODES = (
    _EXPLICIT_CUTOFF_TOPOLOGY_MODE,
    _PUBLIC_DEFAULT_TOPOLOGY_MODE,
)


class _FixedSchemaTensorObjectSpec(NamedTuple):
    storage_index: int
    external_index: int
    external_identity: bool
    shape: tuple
    stride: tuple
    storage_offset: int
    base_is_none: bool


class _FixedSchemaTensorTopology(NamedTuple):
    storage_sizes: tuple
    storage_output_indices: tuple
    equivalent_output_pairs: tuple
    external_output_indices: tuple
    objects: tuple
    occurrences: tuple


class _FixedSchemaPhasePlanState(NamedTuple):
    template: object
    phase_coefficients: tuple
    topology: _FixedSchemaTensorTopology
    class_dependencies: tuple
    dependency_owners: tuple


_FIXED_SCHEMA_PHASE_PLAN_LOCK = threading.RLock()
_FIXED_SCHEMA_PHASE_PLAN_CACHE = OrderedDict()
_FIXED_SCHEMA_PHASE_PLAN_FAILURES = OrderedDict()
_FIXED_SCHEMA_PHASE_PLAN_PID = os.getpid()
_FIXED_SCHEMA_PHASE_PLAN_TENSOR_SLOT = object()
_FIXED_SCHEMA_PHASE_PLAN_MISSING = object()
_FIXED_SCHEMA_PHASE_PLAN_DEPENDENCY_SCHEMAS = {}


def _retain_dependency_identity(value, owners):
    """Record a strong owner beside each identity used in a cache token."""

    if owners is not None:
        owners.append(value)
    return id(value)


def _fixed_schema_phase_plan_dependency_owners(values):
    """Deduplicate strong owners without invoking user equality methods."""

    owners = []
    identities = set()
    for value in values:
        identity = id(value)
        if identity not in identities:
            identities.add(identity)
            owners.append(value)
    return tuple(owners)


def _fixed_schema_phase_plan_enabled() -> bool:
    """Return the strict execution switch."""

    value = os.environ.get(_FIXED_SCHEMA_PHASE_PLAN_ENV)
    return (
        False
        if value is None
        else _parse_switch(_FIXED_SCHEMA_PHASE_PLAN_ENV, value)
    )


def _fixed_schema_phase_plan_python_supported() -> bool:
    """Accept only the qualified ordinary GIL-enabled CPython runtime."""

    try:
        version = tuple(sys.version_info[:2])
        if (
            sys.implementation.name != "cpython"
            or version != _SUPPORTED_PYTHON_VERSION
            or (platform.system(), platform.machine()) != _SUPPORTED_PLATFORM
        ):
            return False
        gil_enabled = getattr(sys, "_is_gil_enabled", None)
        return callable(gil_enabled) and gil_enabled() is True
    except Exception:
        return False


def _fixed_schema_phase_plan_runtime_supported(device) -> bool:
    """Accept the exact eager CPU runtime used to generate this core."""

    try:
        version = torch.__version__.split("+", 1)[0].split(".")
        version = tuple(int(value) for value in version[:3])
        if (
            type(device) is not torch.device
            or device.type != "cpu"
            or version != _SUPPORTED_TORCH_VERSION
            or not _fixed_schema_phase_plan_python_supported()
            or torch.get_default_dtype() is not torch.float32
            or not torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
            or torch.jit.is_scripting()
            or torch.jit.is_tracing()
            or torch.is_anomaly_enabled()
            or torch.are_deterministic_algorithms_enabled()
            or torch.get_num_threads() <= 0
            or torch.get_num_interop_threads() <= 0
            or getattr(torch.autograd.forward_ad, "_current_level", None) != -1
        ):
            return False
        functorch = getattr(getattr(torch, "_C", None), "_functorch", None)
        dynamic_depth = getattr(functorch, "get_dynamic_layer_stack_depth", None)
        if not callable(dynamic_depth) or dynamic_depth() != 0:
            return False
        torch_c = getattr(torch, "_C", None)
        for name in ("_len_torch_dispatch_stack", "_len_torch_function_stack"):
            stack_length = getattr(torch_c, name, None)
            if not callable(stack_length) or stack_length() != 0:
                return False
        compiling = getattr(torch.compiler, "is_compiling", None)
        if callable(compiling) and compiling():
            return False
        dynamo = getattr(torch, "_dynamo", None)
        dynamo_compiling = getattr(dynamo, "is_compiling", None)
        if callable(dynamo_compiling) and dynamo_compiling():
            return False
        autocast_enabled = getattr(torch, "is_autocast_enabled", None)
        if not callable(autocast_enabled):
            return False
        if autocast_enabled() or autocast_enabled("cpu"):
            return False
    except Exception:
        return False
    return True


def _fixed_schema_phase_plan_switch_bundle_supported(xas) -> bool:
    """Seal the optional branches represented by the generated FX graph."""

    try:
        return (
            not xas._phase_plan_bulk_collocation_enabled()
            and not xas._scalar_region_dispatch_enabled()
            and xas._exact_scalar_derivatives_enabled()
            and xas._scalar_derivative_plan_cse_enabled()
            and xas._inspiral_phase_host_scalars_enabled()
            and xas._scripted_phase_ansatz_cpu_enabled()
            and not xas._phase_fit_python_scalars_enabled()
            and xas._phase_fit_native_iterator_enabled()
        )
    except Exception:
        return False


def _fixed_schema_phase_plan_tensor_supported(
    xas,
    value,
    *,
    shape,
    owned,
) -> bool:
    """Accept one plain immutable CPU binary64 Tensor."""

    try:
        return (
            type(value) is torch.Tensor
            and value.layout is torch.strided
            and value.device.type == "cpu"
            and value.device.index is None
            and value.dtype is torch.float64
            and value.shape == torch.Size(shape)
            and value.is_contiguous()
            and (
                not owned
                or (value.storage_offset() == 0 and value._base is None)
            )
            and xas._fit_tensor_version_is_zero(value)
            and not value.is_conj()
            and not value.is_neg()
            and not xas.IMRPhenomX_utils._tree_has_autograd_untrusted(value)
        )
    except Exception:
        return False


def _fixed_schema_phase_plan_cutoffs_independent(cutoff_fMs) -> bool:
    """Require four distinct caller storages so input aliases are unambiguous."""

    try:
        alias_of = torch._C._is_alias_of
        return all(
            left is not right and not alias_of(left, right)
            for left_index, left in enumerate(cutoff_fMs)
            for right in cutoff_fMs[left_index + 1 :]
        )
    except Exception:
        return False


def _fixed_schema_phase_plan_values(
    xas,
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
    """Return the ten binary64 inputs, or ``None`` on contract mismatch."""

    try:
        if (
            not _fixed_schema_phase_plan_enabled()
            or not _fixed_schema_phase_plan_runtime_supported(theta.device)
            or not _fixed_schema_phase_plan_switch_bundle_supported(xas)
            or not _fixed_schema_phase_plan_tensor_supported(
                xas,
                theta,
                shape=(4,),
                owned=True,
            )
            or not xas._canonical_fit_coefficient_table_supported(
                phase_coeffs,
                xas.IMRPhenomX_utils._PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER,
                xas._PHASE_FIT_COEFFICIENT_SOURCE,
                (13, 49),
            )
            or type(chip) is not float
            or not math.isfinite(chip)
            or not 0.0 <= chip <= 1.0
            or type(final_spin) is not float
            or not math.isfinite(final_spin)
            or not -1.0 < final_spin < 1.0
            or coprecessing_deviations is not None
            or phase_fit_rows is not None
            or intrinsic_controls is not None
            or request_proof is not None
            or type(cutoff_fMs) is not tuple
            or len(cutoff_fMs) != 4
            or not all(
                _fixed_schema_phase_plan_tensor_supported(
                    xas,
                    value,
                    shape=(),
                    owned=False,
                )
                for value in cutoff_fMs
            )
            or not _fixed_schema_phase_plan_cutoffs_independent(cutoff_fMs)
        ):
            return None
        theta_values = tuple(theta.tolist())
        if not xas._fit_theta_values_supported(theta_values):
            return None
        cutoff_values = []
        for value in cutoff_fMs:
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    return None
                cutoff_values.append(float(value.item()))
            elif type(value) in (float, int):
                cutoff_values.append(float(value))
            else:
                return None
        cutoff_values = tuple(cutoff_values)
        if not all(
            math.isfinite(value) and value > 0.0 for value in cutoff_values
        ):
            return None
        return theta_values + (chip, final_spin) + cutoff_values
    except Exception:
        return None


def _fixed_schema_public_default_inputs_supported(
    xas,
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
    """Accept only the standalone aligned public request used by XAS."""

    try:
        return (
            _fixed_schema_phase_plan_runtime_supported(theta.device)
            and _fixed_schema_phase_plan_switch_bundle_supported(xas)
            and _fixed_schema_phase_plan_tensor_supported(
                xas,
                theta,
                shape=(4,),
                owned=True,
            )
            and xas._canonical_fit_coefficient_table_supported(
                phase_coeffs,
                xas.IMRPhenomX_utils._PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER,
                xas._PHASE_FIT_COEFFICIENT_SOURCE,
                (13, 49),
            )
            and type(chip) is float
            and struct.pack("=d", chip) == struct.pack("=d", 0.0)
            and final_spin is None
            and coprecessing_deviations is None
            and phase_fit_rows is None
            and cutoff_fMs is None
            and intrinsic_controls is None
            and request_proof is None
            and xas._fit_theta_values_supported(tuple(theta.tolist()))
        )
    except Exception:
        return False


def _fixed_schema_public_default_values(
    xas,
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
    """Capture the first legacy remnant evaluation and its ten inputs."""

    if not _fixed_schema_public_default_inputs_supported(
        xas,
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
    try:
        m1, m2, chi1, chi2 = theta
        remnant = xas.IMRPhenomX_utils.get_remnant_fMs(
            m1,
            m2,
            chi1,
            chi2,
            chip=chip,
            final_spin=None,
        )
        # Preserve the exact add/subtract-zero operations performed by
        # ``xas._get_cutoff_fMs`` when no coprecessing deviations are present.
        derived_cutoff_fMs = (
            remnant.ringdown_frequency - 0.0,
            remnant.damping_frequency + 0.0,
            remnant.meco_frequency,
            remnant.isco_frequency,
        )
        if isinstance(remnant.final_spin, torch.Tensor):
            if remnant.final_spin.numel() != 1:
                return None
            derived_final_spin = float(remnant.final_spin.item())
        elif type(remnant.final_spin) in (float, int):
            derived_final_spin = float(remnant.final_spin)
        else:
            return None
        values = _fixed_schema_phase_plan_values(
            xas,
            theta,
            phase_coeffs,
            chip,
            derived_final_spin,
            None,
            None,
            derived_cutoff_fMs,
            None,
            None,
        )
        if values is None:
            return None
        return values, derived_final_spin, derived_cutoff_fMs
    except Exception:
        return None


def _immutable_dependency_token(value, owners=None):
    """Return an exact token for one generated-code scalar dependency."""

    if value is None or type(value) in (bool, int, str, bytes):
        return (type(value), value)
    if type(value) is float:
        return (float, struct.pack("=d", value))
    if type(value) is tuple:
        # The admitted tuples contain only immutable values. Rebinding changes
        # identity, so the warm key need not walk their contents per call.
        return (tuple, _retain_dependency_identity(value, owners))
    if type(value) is frozenset:
        return (frozenset, _retain_dependency_identity(value, owners))
    if type(value) is torch.Tensor:
        return (
            torch.Tensor,
            _retain_dependency_identity(value, owners),
            value.data_ptr(),
            value._version,
            tuple(value.shape),
            value.dtype,
            value.device,
        )
    return (type(value), _retain_dependency_identity(value, owners))


def _dependency_entry_kind(value):
    if isinstance(value, types.FunctionType):
        return "function"
    if isinstance(value, type):
        return "type"
    if isinstance(value, types.ModuleType):
        return "module"
    return "constant"


def _changed_dependency_token(value, owners=None):
    """Describe a rebound dependency without transient method identities."""

    if isinstance(value, types.MethodType):
        function = value.__func__
        return (
            "method",
            _retain_dependency_identity(value.__self__, owners),
            _retain_dependency_identity(function, owners),
            _retain_dependency_identity(function.__code__, owners),
            _retain_dependency_identity(function.__defaults__, owners),
            _retain_dependency_identity(function.__kwdefaults__, owners),
        )
    return (type(value), _retain_dependency_identity(value, owners))


def _class_dependency_schema(value):
    members = []
    for name, member in vars(value).items():
        if isinstance(member, (staticmethod, classmethod, types.FunctionType)):
            members.append((name, "function"))
        elif name.isupper():
            members.append((name, "constant"))
    return (tuple(vars(value)), tuple(members))


def _class_dependency_manifest(value, schema, owners=None):
    """Validate a pre-indexed class without rescanning all of its members."""

    if not isinstance(value, type):
        return (
            "changed",
            type(value),
            _retain_dependency_identity(value, owners),
        )
    value_identity = _retain_dependency_identity(value, owners)
    expected_names, member_schema = schema
    namespace = vars(value)
    members = []
    for name, kind in member_schema:
        member = namespace.get(name, _FIXED_SCHEMA_PHASE_PLAN_MISSING)
        function = None
        if isinstance(member, (staticmethod, classmethod)):
            function = member.__func__
        elif isinstance(member, types.FunctionType):
            function = member
        if kind == "function" and function is not None:
            members.append(
                (
                    name,
                    _retain_dependency_identity(function, owners),
                    _retain_dependency_identity(function.__code__, owners),
                    _retain_dependency_identity(function.__defaults__, owners),
                    _retain_dependency_identity(function.__kwdefaults__, owners),
                )
            )
        elif kind == "constant":
            members.append(
                (name, _immutable_dependency_token(member, owners))
            )
        else:
            members.append(
                (
                    name,
                    "changed",
                    _changed_dependency_token(member, owners),
                )
            )
    return (
        value_identity,
        tuple(namespace) == expected_names,
        tuple(members),
    )


def _module_dependency_schema(module):
    key = id(module)
    cached = _FIXED_SCHEMA_PHASE_PLAN_DEPENDENCY_SCHEMAS.get(key)
    if cached is not None:
        owner, schema = cached
        if owner is module:
            return schema
    entries = []
    for name, value in vars(module).items():
        if name.startswith("__"):
            continue
        kind = _dependency_entry_kind(value)
        if (
            kind == "type"
            and getattr(value, "__module__", None) != module.__name__
        ):
            # Rebinding an imported type still changes its identity token;
            # its own module is responsible for mutations to its namespace.
            kind = "constant"
        if kind != "constant" or name.lstrip("_").isupper():
            class_schema = None
            if (
                kind == "type"
                and module.__name__
                != "pycbc.waveform.imrphenomxas_torch"
            ):
                class_schema = _class_dependency_schema(value)
            entries.append((name, kind, class_schema))
    schema = (tuple(vars(module)), tuple(entries))
    cached = _FIXED_SCHEMA_PHASE_PLAN_DEPENDENCY_SCHEMAS.setdefault(
        key,
        (module, schema),
    )
    if cached[0] is not module:
        raise RuntimeError("dependency-schema module identity collision")
    return cached[1]


def _module_dependency_manifest(module, owners=None):
    """Snapshot functions, schemas, imports, and declared constants."""

    expected_names, schema = _module_dependency_schema(module)
    namespace = vars(module)
    entries = []
    for name, kind, class_schema in schema:
        value = namespace.get(name, _FIXED_SCHEMA_PHASE_PLAN_MISSING)
        if kind == "function" and isinstance(value, types.FunctionType):
            entries.append(
                (
                    name,
                    "function",
                    _retain_dependency_identity(value, owners),
                    _retain_dependency_identity(value.__code__, owners),
                    _retain_dependency_identity(value.__defaults__, owners),
                    _retain_dependency_identity(value.__kwdefaults__, owners),
                )
            )
        elif kind == "type" and isinstance(value, type):
            entries.append(
                (
                    name,
                    "type",
                    (
                        (_retain_dependency_identity(value, owners),)
                        if class_schema is None
                        else _class_dependency_manifest(
                            value,
                            class_schema,
                            owners,
                        )
                    ),
                )
            )
        elif kind == "module" and isinstance(value, types.ModuleType):
            entries.append(
                (
                    name,
                    "module",
                    _retain_dependency_identity(value, owners),
                )
            )
        elif kind == "constant":
            entries.append(
                (
                    name,
                    "constant",
                    _immutable_dependency_token(value, owners),
                )
            )
        else:
            entries.append(
                (
                    name,
                    "changed",
                    _changed_dependency_token(value, owners),
                )
            )
    return (tuple(namespace) == expected_names, tuple(entries))


def _generated_callable_dependency_manifest(owners=None):
    solve = torch.ops.aten._linalg_solve_ex.default
    return tuple(
        _retain_dependency_identity(value, owners)
        for value in (
            math.atan,
            math.copysign,
            math.exp,
            math.log,
            math.pow,
            math.sqrt,
            torch.as_strided,
            torch.inference_mode,
            torch.no_grad,
            torch.tensor,
            torch.unbind_copy,
            torch.Tensor.detach,
            solve,
        )
    )


def _fixed_schema_phase_plan_environment_manifest():
    """Snapshot each admitted switch that can alter eager object topology."""

    return tuple(
        (name, os.environ.get(name))
        for name in _FIXED_SCHEMA_PHASE_PLAN_ENVIRONMENT_NAMES
    )


def _fixed_schema_phase_plan_tensor_topology_token(value):
    """Describe Tensor/view topology without retaining an allocation identity."""

    base = value._base
    base_token = None
    if base is not None:
        base_token = (
            type(base),
            base.layout,
            base.device,
            base.dtype,
            tuple(base.shape),
            tuple(base.stride()),
            base.storage_offset(),
            base.untyped_storage().nbytes(),
            base.untyped_storage().resizable(),
            base._base is None,
            base.requires_grad,
            base.is_conj(),
            base.is_neg(),
        )
    return (
        type(value),
        value.layout,
        value.device,
        value.dtype,
        tuple(value.shape),
        tuple(value.stride()),
        value.storage_offset(),
        value.untyped_storage().nbytes(),
        value.untyped_storage().resizable(),
        value._base is None,
        value.requires_grad,
        value.is_conj(),
        value.is_neg(),
        base_token,
    )


def _fixed_schema_phase_plan_cutoff_topology_token(cutoff_fMs):
    """Key distinct caller ownership/view contracts independently."""

    alias_of = getattr(torch._C, "_is_alias_of", None)
    if not callable(alias_of):
        raise RuntimeError("Tensor alias inspection is unavailable")
    relations = tuple(
        (left is right, alias_of(left, right))
        for left_index, left in enumerate(cutoff_fMs)
        for right in cutoff_fMs[left_index + 1 :]
    )
    return (
        tuple(
            _fixed_schema_phase_plan_tensor_topology_token(value)
            for value in cutoff_fMs
        ),
        relations,
    )


def _fixed_schema_phase_plan_cache_key(
    xas,
    phase_coeffs,
    cutoff_fMs,
    topology_mode,
    dependency_owners=None,
):
    """Key admission by all runtime state that may affect exact arithmetic."""

    if topology_mode not in _FIXED_SCHEMA_PHASE_PLAN_TOPOLOGY_MODES:
        raise RuntimeError("unknown fixed-schema phase-plan topology mode")
    xutils = xas.IMRPhenomX_utils
    wrapper = sys.modules[__name__]
    return (
        topology_mode,
        int(torch.get_num_threads()),
        int(torch.get_num_interop_threads()),
        _fixed_schema_phase_plan_environment_manifest(),
        _fixed_schema_phase_plan_cutoff_topology_token(cutoff_fMs),
        phase_coeffs.data_ptr(),
        phase_coeffs._version,
        _retain_dependency_identity(phase_coeffs, dependency_owners),
        _retain_dependency_identity(fixed_schema_core, dependency_owners),
        _generated_callable_dependency_manifest(dependency_owners),
        _module_dependency_manifest(wrapper, dependency_owners),
        _module_dependency_manifest(_generated, dependency_owners),
        _module_dependency_manifest(xas, dependency_owners),
        _module_dependency_manifest(xutils, dependency_owners),
    )


def _fixed_schema_phase_plan_sealed_cache_key(
    xas,
    phase_coeffs,
    cutoff_fMs,
    topology_mode,
):
    """Return one cache key together with strong owners for its identities."""

    dependency_owners = []
    key = _fixed_schema_phase_plan_cache_key(
        xas,
        phase_coeffs,
        cutoff_fMs,
        topology_mode,
        dependency_owners,
    )
    return (
        key,
        _fixed_schema_phase_plan_dependency_owners(dependency_owners),
    )


def _tuple_from_nested_lists(value):
    if isinstance(value, list):
        return tuple(_tuple_from_nested_lists(item) for item in value)
    return value


def _fixed_schema_phase_plan_tensor_leaves(value):
    if type(value) is torch.Tensor:
        return (value,)
    if isinstance(value, tuple):
        return tuple(
            leaf
            for item in value
            for leaf in _fixed_schema_phase_plan_tensor_leaves(item)
        )
    return ()


def _fixed_schema_phase_plan_template(value):
    """Retain only result structure, never a cold caller Tensor allocation."""

    if type(value) is torch.Tensor:
        return _FIXED_SCHEMA_PHASE_PLAN_TENSOR_SLOT
    if type(value) is tuple:
        return tuple(_fixed_schema_phase_plan_template(item) for item in value)
    if isinstance(value, tuple):
        return type(value)(
            *(_fixed_schema_phase_plan_template(item) for item in value)
        )
    return value


def _fixed_schema_phase_plan_template_class_dependencies(value):
    """Seal only tuple subclasses that the warm result reconstructs."""

    dependencies = []
    seen = set()

    def visit(item):
        if not isinstance(item, tuple):
            return
        value_type = type(item)
        if value_type is not tuple and value_type not in seen:
            seen.add(value_type)
            schema = _class_dependency_schema(value_type)
            owners = []
            dependencies.append(
                (
                    value_type,
                    schema,
                    _class_dependency_manifest(
                        value_type,
                        schema,
                        owners,
                    ),
                    _fixed_schema_phase_plan_dependency_owners(owners),
                )
            )
        for child in item:
            visit(child)

    visit(value)
    return tuple(dependencies)


def _fixed_schema_phase_plan_class_dependencies_supported(dependencies):
    return all(
        _class_dependency_manifest(value_type, schema) == expected
        for value_type, schema, expected, _owners in dependencies
    )


def _fixed_schema_phase_plan_rebuild(template, tensors):
    iterator = iter(tensors)

    def rebuild(value):
        if value is _FIXED_SCHEMA_PHASE_PLAN_TENSOR_SLOT:
            return next(iterator)
        if type(value) is tuple:
            return tuple(rebuild(item) for item in value)
        if isinstance(value, tuple):
            return type(value)(*(rebuild(item) for item in value))
        return value

    result = rebuild(template)
    try:
        next(iterator)
    except StopIteration:
        return result
    raise RuntimeError("generated phase plan left unused Tensor outputs")


def _fixed_schema_phase_plan_build_topology(reference, external_tensors):
    """Capture eager identities and caller-owned storage relationships."""

    leaves = _fixed_schema_phase_plan_tensor_leaves(reference)
    if len(leaves) != TENSOR_OUTPUT_COUNT:
        raise RuntimeError("eager phase-plan Tensor schema changed")

    external_storage_indices = {}
    for external_index, value in enumerate(external_tensors):
        storage = value.untyped_storage()
        storage_key = (storage.data_ptr(), storage.nbytes())
        if storage_key in external_storage_indices:
            raise RuntimeError("caller cutoff Tensors unexpectedly alias")
        external_storage_indices[storage_key] = external_index

    storage_indices = {}
    storage_sizes = []
    storage_slots = []
    object_indices = {}
    objects = []
    occurrences = []
    leaf_storage_indices = []
    for leaf in leaves:
        storage = leaf.untyped_storage()
        storage_key = (storage.data_ptr(), storage.nbytes())
        external_index = external_storage_indices.get(storage_key, -1)
        external_identity = False
        if external_index >= 0:
            external = external_tensors[external_index]
            storage_index = -1
            storage_offset = leaf.storage_offset() - external.storage_offset()
            if storage_offset != 0:
                raise RuntimeError(
                    "phase-plan output aliases a hidden caller storage slot"
                )
            external_identity = leaf is external
        else:
            storage_index = storage_indices.get(storage_key)
            if storage_index is None:
                if storage.nbytes() % leaf.element_size() != 0:
                    raise RuntimeError("unaligned eager phase-plan storage")
                storage_index = len(storage_sizes)
                storage_indices[storage_key] = storage_index
                storage_size = storage.nbytes() // leaf.element_size()
                storage_sizes.append(storage_size)
                storage_slots.append([None] * storage_size)
            storage_offset = leaf.storage_offset()
        leaf_storage_indices.append(storage_index)

        object_index = object_indices.get(id(leaf))
        spec = _FixedSchemaTensorObjectSpec(
            storage_index,
            external_index,
            external_identity,
            tuple(leaf.shape),
            tuple(leaf.stride()),
            storage_offset,
            leaf._base is None,
        )
        if object_index is None:
            object_index = len(objects)
            object_indices[id(leaf)] = object_index
            objects.append(spec)
        elif objects[object_index] != spec:
            raise RuntimeError("eager Tensor identity changed metadata")
        occurrences.append(object_index)

    equivalent = []
    external_outputs = []

    def record(output_index, storage_index, storage_offset):
        slots = storage_slots[storage_index]
        if not 0 <= storage_offset < len(slots):
            raise RuntimeError("phase-plan output exceeds its storage")
        prior = slots[storage_offset]
        if prior is None:
            slots[storage_offset] = output_index
        else:
            equivalent.append((output_index, prior))

    for output_index, (leaf, storage_index) in enumerate(
        zip(leaves, leaf_storage_indices)
    ):
        if storage_index < 0:
            external_outputs.append(
                (output_index, objects[occurrences[output_index]].external_index)
            )
        else:
            record(output_index, storage_index, leaf.storage_offset())
    for hidden_index, (leaf_index, storage_offset) in enumerate(
        HIDDEN_STORAGE_SPECS,
        start=TENSOR_OUTPUT_COUNT,
    ):
        if leaf_storage_indices[leaf_index] < 0:
            raise RuntimeError("generated hidden output aliases caller storage")
        record(hidden_index, leaf_storage_indices[leaf_index], storage_offset)
    if (
        SCALAR_OUTPUT_COUNT
        != TENSOR_OUTPUT_COUNT + len(HIDDEN_STORAGE_SPECS)
        or any(slot is None for slots in storage_slots for slot in slots)
    ):
        raise RuntimeError("generated core does not cover eager storage")
    return _FixedSchemaTensorTopology(
        tuple(storage_sizes),
        tuple(tuple(slots) for slots in storage_slots),
        tuple(equivalent),
        tuple(external_outputs),
        tuple(objects),
        tuple(occurrences),
    )


def _fixed_schema_phase_plan_scalar_raw_equal(left, right):
    return (
        type(left) is float
        and type(right) is float
        and struct.pack("=d", left) == struct.pack("=d", right)
    )


def _fixed_schema_phase_plan_object_supported(
    value,
    spec,
    topology,
    external_tensors,
):
    """Validate one reconstructed object against its sealed ownership class."""

    if spec.external_index >= 0:
        external = external_tensors[spec.external_index]
        expected_offset = external.storage_offset() + spec.storage_offset
        expected_storage_nbytes = external.untyped_storage().nbytes()
        expected_version = external._version
        external_relation = (
            torch._C._is_alias_of(value, external)
            and (value is external) == spec.external_identity
        )
    else:
        expected_offset = spec.storage_offset
        expected_storage_nbytes = (
            topology.storage_sizes[spec.storage_index] * value.element_size()
        )
        expected_version = 0
        external_relation = all(
            not torch._C._is_alias_of(value, external)
            for external in external_tensors
        )
    return (
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.shape == torch.Size(spec.shape)
        and value.stride() == spec.stride
        and value.dtype is torch.float64
        and value.device.type == "cpu"
        and value.device.index is None
        and value.storage_offset() == expected_offset
        and (value._base is None) == spec.base_is_none
        and value.untyped_storage().nbytes() == expected_storage_nbytes
        and value._version == expected_version
        and not value.requires_grad
        and value.grad_fn is None
        and not value.is_conj()
        and not value.is_neg()
        and external_relation
    )


def _fixed_schema_phase_plan_materialize(
    state,
    values,
    external_tensors,
    *,
    validate_schema=True,
):
    outputs = fixed_schema_core(values, state.phase_coefficients)
    if type(outputs) is not tuple or len(outputs) != SCALAR_OUTPUT_COUNT:
        raise RuntimeError("generated phase-plan output schema changed")
    if not all(type(value) is float for value in outputs):
        raise RuntimeError("generated phase-plan output type changed")
    topology = state.topology
    if not all(
        _fixed_schema_phase_plan_scalar_raw_equal(outputs[left], outputs[right])
        for left, right in topology.equivalent_output_pairs
    ):
        raise RuntimeError("generated phase-plan equivalent outputs diverged")
    if not all(
        _fixed_schema_phase_plan_scalar_raw_equal(
            outputs[output_index],
            values[6 + external_index],
        )
        for output_index, external_index in topology.external_output_indices
    ):
        raise RuntimeError("generated phase plan diverged from caller storage")

    size_groups = {}
    for storage_index, storage_size in enumerate(topology.storage_sizes):
        size_groups.setdefault(storage_size, []).append(storage_index)
    storage_bases = [None] * len(topology.storage_sizes)
    with torch.inference_mode(False), torch.no_grad():
        for storage_size, storage_indices in size_groups.items():
            if storage_size == 1:
                packed_values = tuple(
                    outputs[
                        topology.storage_output_indices[storage_index][0]
                    ]
                    for storage_index in storage_indices
                )
            else:
                packed_values = tuple(
                    tuple(
                        outputs[output_index]
                        for output_index in topology.storage_output_indices[
                            storage_index
                        ]
                    )
                    for storage_index in storage_indices
                )
            packed = torch.tensor(
                packed_values,
                dtype=torch.float64,
                device="cpu",
            )
            bases = torch.unbind_copy(packed, dim=0)
            if len(bases) != len(storage_indices):
                raise RuntimeError("generated storage materialization changed")
            for storage_index, base in zip(storage_indices, bases):
                expected_shape = (
                    torch.Size(())
                    if storage_size == 1
                    else torch.Size((storage_size,))
                )
                if (
                    base.shape != expected_shape
                    or base.untyped_storage().nbytes()
                    != storage_size * base.element_size()
                    or base._base is not None
                ):
                    raise RuntimeError("generated storage owner schema changed")
                storage_bases[storage_index] = base

        tensor_objects = []
        claimed_storage_bases = set()
        for spec in topology.objects:
            if spec.external_index >= 0:
                external = external_tensors[spec.external_index]
                if spec.external_identity:
                    tensor = external
                else:
                    tensor = torch.as_strided(
                        external,
                        spec.shape,
                        spec.stride,
                        external.storage_offset() + spec.storage_offset,
                    )
                    if spec.base_is_none:
                        tensor = tensor.detach()
            else:
                storage_index = spec.storage_index
                if (
                    topology.storage_sizes[storage_index] == 1
                    and storage_index not in claimed_storage_bases
                    and spec.shape == torch.Size(())
                    and spec.stride == ()
                    and spec.storage_offset == 0
                    and spec.base_is_none
                ):
                    tensor = storage_bases[storage_index]
                    claimed_storage_bases.add(storage_index)
                else:
                    tensor = torch.as_strided(
                        storage_bases[storage_index],
                        spec.shape,
                        spec.stride,
                        spec.storage_offset,
                    )
                    if spec.base_is_none:
                        tensor = tensor.detach()
            tensor_objects.append(tensor)

    tensors = tuple(tensor_objects[index] for index in topology.occurrences)
    result = _fixed_schema_phase_plan_rebuild(state.template, tensors)
    if validate_schema:
        leaves = _fixed_schema_phase_plan_tensor_leaves(result)
        if (
            len(leaves) != TENSOR_OUTPUT_COUNT
            or not all(
                _fixed_schema_phase_plan_object_supported(
                    value,
                    spec,
                    topology,
                    external_tensors,
                )
                for value, object_index in zip(leaves, topology.occurrences)
                for spec in (topology.objects[object_index],)
            )
        ):
            raise RuntimeError(
                "materialized phase plan failed its schema seal"
            )
    return result


def _fixed_schema_phase_plan_raw_equal(reference, candidate) -> bool:
    """Compare values and public semantic metadata exactly."""

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
            and reference.requires_grad == candidate.requires_grad
            and reference.is_leaf == candidate.is_leaf
            and (reference.grad_fn is None) == (candidate.grad_fn is None)
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
                _fixed_schema_phase_plan_raw_equal(left, right)
                for left, right in zip(reference, candidate)
            )
        )
    return type(reference) is type(candidate) and reference == candidate


def _fixed_schema_phase_plan_storage_equal(
    reference,
    candidate,
    topology,
    external_tensors,
) -> bool:
    """Compare hidden bytes, bases, and intentional caller-storage aliases."""

    reference_leaves = _fixed_schema_phase_plan_tensor_leaves(reference)
    candidate_leaves = _fixed_schema_phase_plan_tensor_leaves(candidate)
    alias_of = getattr(torch._C, "_is_alias_of", None)
    if (
        len(reference_leaves) != TENSOR_OUTPUT_COUNT
        or len(candidate_leaves) != TENSOR_OUTPUT_COUNT
        or not callable(alias_of)
    ):
        return False
    try:
        representatives = [None] * len(topology.storage_sizes)
        candidate_storage_pointers = set()
        for leaf_index, object_index in enumerate(topology.occurrences):
            spec = topology.objects[object_index]
            reference_leaf = reference_leaves[leaf_index]
            candidate_leaf = candidate_leaves[leaf_index]
            if spec.external_index < 0 and representatives[spec.storage_index] is None:
                representatives[spec.storage_index] = leaf_index
                pointer = candidate_leaf.untyped_storage().data_ptr()
                if pointer in candidate_storage_pointers:
                    return False
                candidate_storage_pointers.add(pointer)
            if (
                reference_leaf._version != candidate_leaf._version
                or reference_leaf.untyped_storage().resizable()
                != candidate_leaf.untyped_storage().resizable()
            ):
                return False
            reference_base = reference_leaf._base
            candidate_base = candidate_leaf._base
            if (reference_base is None) != (candidate_base is None):
                return False
            if reference_base is not None and not (
                type(reference_base) is type(candidate_base) is torch.Tensor
                and reference_base.layout
                == candidate_base.layout
                == torch.strided
                and reference_base.device == candidate_base.device
                and reference_base.dtype == candidate_base.dtype
                and reference_base.shape == candidate_base.shape
                and reference_base.stride() == candidate_base.stride()
                and reference_base.storage_offset()
                == candidate_base.storage_offset()
                and reference_base.untyped_storage().nbytes()
                == candidate_base.untyped_storage().nbytes()
                and reference_base.untyped_storage().resizable()
                == candidate_base.untyped_storage().resizable()
                and reference_base._version == candidate_base._version
                and reference_base.requires_grad == candidate_base.requires_grad
                and reference_base.is_conj() == candidate_base.is_conj()
                and reference_base.is_neg() == candidate_base.is_neg()
            ):
                return False
            for external_index, external in enumerate(external_tensors):
                expected_alias = spec.external_index == external_index
                expected_identity = (
                    expected_alias and spec.external_identity
                )
                if (
                    alias_of(reference_leaf, external) != expected_alias
                    or alias_of(candidate_leaf, external) != expected_alias
                    or (reference_leaf is external) != expected_identity
                    or (candidate_leaf is external) != expected_identity
                ):
                    return False
        for storage_index, leaf_index in enumerate(representatives):
            if leaf_index is None:
                return False
            storage_size = topology.storage_sizes[storage_index]
            reference_storage = torch.as_strided(
                reference_leaves[leaf_index],
                (storage_size,),
                (1,),
                0,
            )
            candidate_storage = torch.as_strided(
                candidate_leaves[leaf_index],
                (storage_size,),
                (1,),
                0,
            )
            if not torch.equal(
                reference_storage.view(torch.uint8),
                candidate_storage.view(torch.uint8),
            ):
                return False
        for reference_index, reference_leaf in enumerate(reference_leaves):
            reference_spec = topology.objects[
                topology.occurrences[reference_index]
            ]
            for candidate_index, candidate_leaf in enumerate(candidate_leaves):
                candidate_spec = topology.objects[
                    topology.occurrences[candidate_index]
                ]
                expected_alias = (
                    reference_spec.external_index >= 0
                    and reference_spec.external_index
                    == candidate_spec.external_index
                )
                if alias_of(reference_leaf, candidate_leaf) != expected_alias:
                    return False
    except Exception:
        return False
    return True


def _after_fixed_schema_phase_plan_fork():
    global _FIXED_SCHEMA_PHASE_PLAN_LOCK
    global _FIXED_SCHEMA_PHASE_PLAN_PID

    # A user-created lock may have been held by a vanished thread at fork.
    # Replace it before touching child state instead of trying to acquire it.
    _FIXED_SCHEMA_PHASE_PLAN_LOCK = threading.RLock()
    _FIXED_SCHEMA_PHASE_PLAN_CACHE.clear()
    _FIXED_SCHEMA_PHASE_PLAN_FAILURES.clear()
    _FIXED_SCHEMA_PHASE_PLAN_PID = os.getpid()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fixed_schema_phase_plan_fork)


def _fixed_schema_phase_plan_cache_state():
    """Return cache/failure keys for focused tests and profiling."""

    if _FIXED_SCHEMA_PHASE_PLAN_PID != os.getpid():
        _after_fixed_schema_phase_plan_fork()
    with _FIXED_SCHEMA_PHASE_PLAN_LOCK:
        return (
            tuple(_FIXED_SCHEMA_PHASE_PLAN_CACHE),
            tuple(_FIXED_SCHEMA_PHASE_PLAN_FAILURES),
        )


def _clear_fixed_schema_phase_plan_cache():
    """Drop all process-local admissions and sticky failures."""

    global _FIXED_SCHEMA_PHASE_PLAN_PID

    with _FIXED_SCHEMA_PHASE_PLAN_LOCK:
        _FIXED_SCHEMA_PHASE_PLAN_CACHE.clear()
        _FIXED_SCHEMA_PHASE_PLAN_FAILURES.clear()
        _FIXED_SCHEMA_PHASE_PLAN_PID = os.getpid()


def _fixed_schema_phase_plan_remember_failure(key, dependency_owners):
    with _FIXED_SCHEMA_PHASE_PLAN_LOCK:
        _FIXED_SCHEMA_PHASE_PLAN_CACHE.pop(key, None)
        _FIXED_SCHEMA_PHASE_PLAN_FAILURES[key] = dependency_owners
        _FIXED_SCHEMA_PHASE_PLAN_FAILURES.move_to_end(key)
        while (
            len(_FIXED_SCHEMA_PHASE_PLAN_FAILURES)
            > _FIXED_SCHEMA_PHASE_PLAN_CACHE_LIMIT
        ):
            _FIXED_SCHEMA_PHASE_PLAN_FAILURES.popitem(last=False)


def _fixed_schema_phase_plan_build_state(
    xas,
    values,
    theta,
    phase_coeffs,
    chip,
    final_spin,
    cutoff_fMs,
    topology_mode,
    dependency_owners,
):
    if topology_mode == _EXPLICIT_CUTOFF_TOPOLOGY_MODE:
        reference_final_spin = final_spin
        reference_cutoff_fMs = cutoff_fMs
        topology_external_tensors = cutoff_fMs
    elif topology_mode == _PUBLIC_DEFAULT_TOPOLOGY_MODE:
        # Preserve the exact standalone public eager ownership graph.  Its
        # repeated cutoff calculations do not alias the captured inputs.
        reference_final_spin = None
        reference_cutoff_fMs = None
        topology_external_tensors = ()
    else:
        raise RuntimeError("unknown fixed-schema phase-plan topology mode")
    reference = xas._prepare_phase_plan_eager(
        theta,
        phase_coeffs,
        chip,
        final_spin=reference_final_spin,
        coprecessing_deviations=None,
        _phase_fit_rows=None,
        _cutoff_fMs=reference_cutoff_fMs,
        _intrinsic_controls=None,
        _request_proof=None,
    )
    phase_coefficients = _tuple_from_nested_lists(
        phase_coeffs.detach().cpu().tolist()
    )
    topology = _fixed_schema_phase_plan_build_topology(
        reference,
        topology_external_tensors,
    )
    state = _FixedSchemaPhasePlanState(
        _fixed_schema_phase_plan_template(reference),
        phase_coefficients,
        topology,
        _fixed_schema_phase_plan_template_class_dependencies(reference),
        dependency_owners,
    )
    candidate = _fixed_schema_phase_plan_materialize(
        state,
        values,
        cutoff_fMs,
    )
    if not (
        _fixed_schema_phase_plan_raw_equal(reference, candidate)
        and xas._phase_plan_torchscript_tree_raw_equal(reference, candidate)
        and xas._phase_plan_torchscript_tree_alias_equal(reference, candidate)
        and _fixed_schema_phase_plan_storage_equal(
            reference,
            candidate,
            topology,
            cutoff_fMs,
        )
    ):
        raise RuntimeError(
            "generated phase plan failed its cold semantic canary"
        )
    return state, candidate


def _maybe_prepare_fixed_schema_phase_plan_values(
    xas,
    values,
    theta,
    phase_coeffs,
    chip,
    final_spin,
    cutoff_fMs,
    topology_mode,
):
    """Cache and materialize one already-qualified fixed-schema request."""

    try:
        key = _fixed_schema_phase_plan_cache_key(
            xas,
            phase_coeffs,
            cutoff_fMs,
            topology_mode,
        )
    except Exception:
        return None
    if _FIXED_SCHEMA_PHASE_PLAN_PID != os.getpid():
        _after_fixed_schema_phase_plan_fork()
    state = _FIXED_SCHEMA_PHASE_PLAN_CACHE.get(key)
    if state is not None and not (
        _fixed_schema_phase_plan_class_dependencies_supported(
            state.class_dependencies
        )
    ):
        state = None
    if state is None:
        with _FIXED_SCHEMA_PHASE_PLAN_LOCK:
            state = _FIXED_SCHEMA_PHASE_PLAN_CACHE.get(key)
            if state is not None and not (
                _fixed_schema_phase_plan_class_dependencies_supported(
                    state.class_dependencies
                )
            ):
                _FIXED_SCHEMA_PHASE_PLAN_CACHE.pop(key, None)
                state = None
            if state is None and key not in _FIXED_SCHEMA_PHASE_PLAN_FAILURES:
                dependency_owners = ()
                failure_key = key
                try:
                    sealed_key, dependency_owners = (
                        _fixed_schema_phase_plan_sealed_cache_key(
                            xas,
                            phase_coeffs,
                            cutoff_fMs,
                            topology_mode,
                        )
                    )
                    if sealed_key != key:
                        return None
                    for build_index in range(2):
                        failure_key = sealed_key
                        state, result = _fixed_schema_phase_plan_build_state(
                            xas,
                            values,
                            theta,
                            phase_coeffs,
                            chip,
                            final_spin,
                            cutoff_fMs,
                            topology_mode,
                            dependency_owners,
                        )
                        post_build_key, post_build_dependency_owners = (
                            _fixed_schema_phase_plan_sealed_cache_key(
                                xas,
                                phase_coeffs,
                                cutoff_fMs,
                                topology_mode,
                            )
                        )
                        if post_build_key == sealed_key:
                            break
                        if build_index != 0:
                            _fixed_schema_phase_plan_remember_failure(
                                post_build_key,
                                post_build_dependency_owners,
                            )
                            return None
                        # A qualified eager branch may initialize one of its
                        # own process-local optional executors.  Re-canary from
                        # that new dependency identity, retaining its owners,
                        # and never publish the state built under the stale
                        # pre-initialization key.
                        sealed_key = post_build_key
                        dependency_owners = post_build_dependency_owners
                except Exception:
                    _fixed_schema_phase_plan_remember_failure(
                        failure_key,
                        dependency_owners,
                    )
                    return None
                _FIXED_SCHEMA_PHASE_PLAN_CACHE[sealed_key] = state
                _FIXED_SCHEMA_PHASE_PLAN_CACHE.move_to_end(sealed_key)
                while (
                    len(_FIXED_SCHEMA_PHASE_PLAN_CACHE)
                    > _FIXED_SCHEMA_PHASE_PLAN_CACHE_LIMIT
                ):
                    _FIXED_SCHEMA_PHASE_PLAN_CACHE.popitem(last=False)
                return result
    if state is None:
        return None
    try:
        return _fixed_schema_phase_plan_materialize(
            state,
            values,
            cutoff_fMs,
            validate_schema=False,
        )
    except Exception:
        _fixed_schema_phase_plan_remember_failure(
            key,
            state.dependency_owners,
        )
        return None


def _maybe_prepare_fixed_schema_phase_plan(
    theta,
    phase_coeffs,
    chip=0.0,
    *,
    final_spin=None,
    coprecessing_deviations=None,
    _phase_fit_rows=None,
    _cutoff_fMs=None,
    _intrinsic_controls=None,
    _request_proof=None,
):
    """Return one exact generated plan, or ``None`` for unchanged fallback."""

    if not _fixed_schema_phase_plan_enabled():
        return None
    # Lazy import keeps this module safe for a top-level import by XAS.
    from . import imrphenomxas_torch as xas

    values = _fixed_schema_phase_plan_values(
        xas,
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
    if values is None:
        return None
    return _maybe_prepare_fixed_schema_phase_plan_values(
        xas,
        values,
        theta,
        phase_coeffs,
        chip,
        final_spin,
        _cutoff_fMs,
        _EXPLICIT_CUTOFF_TOPOLOGY_MODE,
    )


def _maybe_prepare_fixed_schema_public_default_phase_plan(
    theta,
    phase_coeffs,
    chip=0.0,
    *,
    final_spin=None,
    coprecessing_deviations=None,
    _phase_fit_rows=None,
    _cutoff_fMs=None,
    _intrinsic_controls=None,
    _request_proof=None,
):
    """Return an exact generated plan for standalone public XAS defaults."""

    if not _fixed_schema_phase_plan_enabled():
        return None
    # Lazy import keeps this module safe for a top-level import by XAS.
    from . import imrphenomxas_torch as xas

    prepared = _fixed_schema_public_default_values(
        xas,
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
    if prepared is None:
        return None
    try:
        # The public waveform evaluates this plan through XAS's optional
        # TorchScript ansatz later in the same request.  Initialize that
        # process-local dependency before sealing the phase-plan cache key;
        # otherwise the first waveform stores a pre-initialization identity
        # that can never be reused.  A failed optional executor keeps this
        # generated path fail-closed on the unchanged eager fallback.
        if (
            xas._SCRIPTED_PHASE_ANSATZ_CPU_EXECUTOR is None
            and xas._get_scripted_phase_ansatz_cpu_executor() is None
        ):
            return None
    except Exception:
        return None
    values, derived_final_spin, derived_cutoff_fMs = prepared
    return _maybe_prepare_fixed_schema_phase_plan_values(
        xas,
        values,
        theta,
        phase_coeffs,
        chip,
        derived_final_spin,
        derived_cutoff_fMs,
        _PUBLIC_DEFAULT_TOPOLOGY_MODE,
    )
