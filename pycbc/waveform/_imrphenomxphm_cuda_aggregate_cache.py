"""Bounded exact CUDA cache for pre-terminal XPHM polarizations.

The cache owns detached complex128 copies of the two polarization tensors
immediately before XPHM's request-local terminal rotations.  It is strictly
opt in, process/thread/device/stream local, and depends on the independently
gated CUDA co-precessing-plan cache.  Ordinary warm hits inspect tensor
metadata only; the optional debug fingerprint performs a synchronizing
device-to-host integrity check.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib
import math
import os
import sys
import threading
from types import FunctionType
from typing import NamedTuple

import torch


CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_ENV = (
    "PYCBC_IMRPHENOMXPHM_CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE"
)
CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_BYTES_ENV = (
    "PYCBC_IMRPHENOMXPHM_CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_BYTES"
)
CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_DEBUG_FINGERPRINT_ENV = (
    "PYCBC_IMRPHENOMXPHM_CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_"
    "DEBUG_FINGERPRINT"
)
CUDA_AGGREGATE_PRETERMINAL_TWIST_PUBLIC_FASTPATH_ENV = (
    "PYCBC_IMRPHENOMXPHM_CUDA_AGGREGATE_PRETERMINAL_TWIST_PUBLIC_FASTPATH"
)
CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_DEFAULT_BYTES = 16 * 1024 * 1024
CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_MAX_ENTRIES = 32
CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_ENTRY_OVERHEAD = 1024
CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_IMPLEMENTATION = 2


class _TensorDescriptor(NamedTuple):
    identity: int
    storage_identity: int
    storage_pointer: int
    storage_bytes: int
    data_pointer: int
    version: int
    dtype: torch.dtype
    device: torch.device
    shape: tuple
    stride: tuple
    storage_offset: int
    base_identity: int | None
    requires_grad: bool
    grad_fn_identity: int | None
    inference: bool
    conjugated: bool
    negated: bool


class _Request(NamedTuple):
    key: tuple
    semantic_key: tuple
    key_bytes: int
    budget: int
    plan: object
    plan_descriptors: tuple
    device_index: int
    thread_id: int
    stream_id: int
    proof: object
    params: dict
    public_key: tuple
    public_layout: object
    modes: tuple


class _PublicLayout(NamedTuple):
    npoints: int
    first_bin: int
    stop_bin: int
    delta_f: float
    polarization_rotation: float
    complex_dtype: torch.dtype
    device: torch.device


class _PublicRequest(NamedTuple):
    key: tuple
    budget: int
    device_index: int
    thread_id: int
    stream_id: int
    proof: object
    params: dict
    long_asc_nodes: float
    long_asc_nodes_key: tuple


class _Entry(NamedTuple):
    plus: torch.Tensor
    cross: torch.Tensor
    descriptors: tuple
    fingerprint: tuple
    nbytes: int
    plan: object
    semantic_key: tuple
    device_index: int
    thread_id: int
    stream_id: int
    public_key: tuple
    public_layout: _PublicLayout
    modes: tuple
    plan_descriptors: tuple


class _PublicHit(NamedTuple):
    entry_key: tuple
    entry: _Entry
    request: _PublicRequest


def _new_counters():
    return {
        "hits": 0,
        "misses": 0,
        "stores": 0,
        "evictions": 0,
        "invalidations": 0,
        "ineligible": 0,
        "canary_failures": 0,
        "oversize": 0,
        "races": 0,
        "debug_fingerprint_checks": 0,
        "debug_fingerprint_failures": 0,
        "public_hits": 0,
        "public_misses": 0,
        "public_invalidations": 0,
        "public_races": 0,
        "public_materialization_failures": 0,
    }


@lru_cache(maxsize=None)
def _device_identity(index):
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    return (
        index,
        None if uuid is None else str(uuid),
        properties.name,
        int(properties.major),
        int(properties.minor),
        int(properties.total_memory),
        torch.version.cuda,
    )


class CudaAggregatePreterminalTwistCache:
    """One bounded, exact, same-stream CUDA aggregate LRU."""

    gate_env = CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_ENV
    budget_env = CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_BYTES_ENV
    debug_fingerprint_env = (
        CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_DEBUG_FINGERPRINT_ENV
    )
    public_fastpath_env = CUDA_AGGREGATE_PRETERMINAL_TWIST_PUBLIC_FASTPATH_ENV

    def __init__(self, owner, proof_current):
        self._owner = owner
        self._proof_current = proof_current
        self._sealed_control_roots = (
            owner._run_xphm_request_proof_plan,
            owner._request_proof_plan_current,
        )
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._size = 0
        self._counters = _new_counters()

    def enabled(self):
        value = os.environ.get(self.gate_env)
        return (
            False
            if value is None
            else self._owner._parse_switch(self.gate_env, value)
        )

    def debug_fingerprint_enabled(self):
        value = os.environ.get(self.debug_fingerprint_env)
        return (
            False
            if value is None
            else self._owner._parse_switch(self.debug_fingerprint_env, value)
        )

    def public_fastpath_enabled(self):
        value = os.environ.get(self.public_fastpath_env)
        return (
            False
            if value is None
            else self._owner._parse_switch(self.public_fastpath_env, value)
        )

    def budget(self):
        value = os.environ.get(self.budget_env)
        if value is None:
            return CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_DEFAULT_BYTES
        if not value.isascii() or not value.isdecimal():
            return None
        try:
            budget = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return budget if budget > 0 else None

    def _reset_if_forked_locked(self):
        process_id = os.getpid()
        if process_id == self._pid:
            return
        self._entries = OrderedDict()
        self._size = 0
        self._counters = _new_counters()
        self._pid = process_id

    def after_fork(self):
        """Drop inherited CUDA tensors and replace the inherited lock."""

        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._size = 0
        self._counters = _new_counters()
        _device_identity.cache_clear()

    def clear(self):
        with self._lock:
            self._reset_if_forked_locked()
            self._entries.clear()
            self._size = 0
            self._counters = _new_counters()

    def invalidate_all(self):
        with self._lock:
            self._reset_if_forked_locked()
            invalidated = len(self._entries)
            self._entries.clear()
            self._size = 0
            self._counters["invalidations"] += invalidated

    def stats(self):
        with self._lock:
            self._reset_if_forked_locked()
            result = dict(self._counters)
            result.update(
                pid=self._pid,
                entries=len(self._entries),
                bytes=self._size,
                budget_bytes=self.budget() or 0,
                max_entries=CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_MAX_ENTRIES,
            )
            return result

    def _note(self, name, amount=1):
        try:
            with self._lock:
                self._reset_if_forked_locked()
                self._counters[name] += amount
        except Exception:
            pass

    @staticmethod
    def _resolved_device_index(device):
        if type(device) is not torch.device or device.type != "cuda":
            return None
        return torch.cuda.current_device() if device.index is None else device.index

    def _plan_cache(self):
        return getattr(self._owner, "_CUDA_COPRECESSING_PLAN_CACHE", None)

    def _request_supported(
        self,
        model,
        frequencies,
        params,
        modes,
        active_f_max,
        uniform_grid_metadata,
        plan,
    ):
        owner = self._owner
        plan_cache = self._plan_cache()
        if plan_cache is None:
            return False
        try:
            if (
                not owner._cuda_coprecessing_plan_cache_enabled()
                or model.msa_reference_angles_deferred is not False
                or plan.reference_angle_core is not None
                or not plan_cache._request_supported(
                    model,
                    frequencies,
                    params,
                    modes,
                    active_f_max,
                    uniform_grid_metadata,
                )
                or not owner._coprecessing_plan_coefficient_tables_supported()
                or not plan_cache._schema_supported(
                    plan,
                    frequencies,
                    modes,
                    privately_owned=False,
                )
            ):
                return False
            plan_descriptors = plan_cache._descriptors(plan)
        except Exception:
            return False
        return bool(plan_descriptors)

    @staticmethod
    def _terminal_free_params(params):
        values = dict(params)
        values.pop("long_asc_nodes", None)
        return values

    @staticmethod
    def _producer_module_function_identity(root):
        """Fingerprint every Python producer in ``root``'s defining module."""

        try:
            namespace = root.__globals__
            module_name = root.__module__
            if type(namespace) is not dict or type(module_name) is not str:
                return None
            functions = tuple(
                sorted(
                    (
                        name,
                        id(value),
                        id(value.__code__),
                    )
                    for name, value in namespace.items()
                    if type(name) is str
                    and type(value) is FunctionType
                    and value.__module__ == module_name
                )
            )
            return module_name, id(namespace), functions
        except Exception:
            return None

    def _producer_roots(self):
        """Return every producer whose live identity governs an aggregate."""

        owner = self._owner
        return (
            owner._run_scoped_xphm_request,
            owner._dispatch_imrphenomxphm_request,
            owner._run_xphm_request_proof_plan,
            owner._imrphenomxphm_fd_torch,
            owner._cuda_aggregate_public_fastpath,
            owner._request_proof_plan_current,
            owner._validated_inputs,
            owner._request_bulk_twist_harmonics,
            owner._bulk_twist_harmonics,
            owner._build_model,
            owner._twist_coprecessing_modes,
            owner._bulk_mode_angles,
            owner._mode_angles,
            owner._reference_and_mode_msa_angles,
            owner.msa_angles,
            owner._twist_mode,
            owner._twist_reuse_supported,
            owner._packed_twist_harmonics,
            owner._stacked_twist_request_device,
            owner._stacked_twist_mode,
            owner._grouped_outer_twist_request_device,
            owner._qualified_grouped_outer_twist_calls,
            owner._grouped_outer_twist_modes,
            owner._grouped_outer_twist_cuda_graph,
            owner._build_grouped_outer_twist_cuda_graph,
            owner._replay_grouped_outer_twist_cuda_graph,
            owner._grouped_outer_twist_mprime_coefficients,
            owner._wigner_columns,
            owner._ordered_stacked_twist_sum,
            owner._bulk_twist_exponentials,
            owner._packed_twist_exponentials,
            owner._twist_exponential_recurrence,
            owner.spin_weighted_spherical_harmonic,
            owner.spin_minus_two_spherical_harmonics_phi_zero,
            owner.cudagraphed_spin_minus_two_spherical_harmonics_phi_zero,
            owner.scripted_spin_minus_two_spherical_harmonics_phi_zero,
            owner.vectorized_spin_minus_two_spherical_harmonics_phi_zero,
            owner._series_from_active_samples,
        )

    def _public_producers_supported(self):
        """Reject mutable closure state that an early lookup cannot key."""

        owner = self._owner
        try:
            if self._sealed_control_roots != (
                owner._run_xphm_request_proof_plan,
                owner._request_proof_plan_current,
            ):
                return False
            sealed_control_roots = set(self._sealed_control_roots)
            for root in self._producer_roots():
                if root in sealed_control_roots:
                    continue
                if type(root) is FunctionType:
                    module_name = root.__module__
                    namespace = root.__globals__
                    if (
                        type(module_name) is not str
                        or not module_name.startswith("pycbc.waveform.")
                        or type(namespace) is not dict
                        or getattr(sys.modules.get(module_name), "__dict__", None)
                        is not namespace
                        or namespace.get(root.__name__) is not root
                        or root.__closure__ is not None
                        or root.__dict__
                    ):
                        return False
                    continue
                wrapped = getattr(root, "__wrapped__", None)
                if not (
                    root is owner._grouped_outer_twist_mprime_coefficients
                    and type(wrapped) is FunctionType
                    and root.__module__ == wrapped.__module__
                    and root.__module__.startswith("pycbc.waveform.")
                    and root.__module__ in sys.modules
                    and wrapped.__globals__
                    is sys.modules[root.__module__].__dict__
                    and wrapped.__globals__.get(wrapped.__name__) is root
                    and wrapped.__closure__ is None
                    and not wrapped.__dict__
                ):
                    return False
            return True
        except Exception:
            return False

    def _implementation_identity(self, device_index):
        owner = self._owner
        plan_cache = self._plan_cache()
        roots = (self._proof_current, *self._producer_roots())
        ignored_environment = {
            self.gate_env,
            self.budget_env,
            self.debug_fingerprint_env,
            self.public_fastpath_env,
        }
        environment = tuple(
            sorted(
                (name, value)
                for name, value in os.environ.items()
                if name not in ignored_environment
                and name.startswith(("PYCBC_IMRPHENOMX", "PYCBC_TORCH_"))
            )
        )
        return (
            CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_IMPLEMENTATION,
            (
                id(type(self)),
                tuple(
                    (
                        name,
                        id(getattr(type(self), name, None)),
                        id(
                            getattr(
                                getattr(type(self), name, None),
                                "__code__",
                                None,
                            )
                        ),
                    )
                    for name in (
                        "prepare_public",
                        "lookup_public",
                        "finish_public_hit",
                        "prepare",
                        "lookup",
                        "store",
                    )
                ),
            ),
            (
                id(type(plan_cache)),
                id(getattr(type(plan_cache), "_key", None)),
                id(
                    getattr(
                        getattr(type(plan_cache), "_key", None),
                        "__code__",
                        None,
                    )
                ),
                id(getattr(type(plan_cache), "_schema_supported", None)),
                id(
                    getattr(
                        getattr(type(plan_cache), "_schema_supported", None),
                        "__code__",
                        None,
                    )
                ),
                plan_cache._implementation_identity(device_index),
            ),
            sys.implementation.name,
            tuple(sys.version_info[:3]),
            torch.__version__,
            getattr(torch.version, "git_version", None),
            torch.get_default_dtype(),
            bool(torch.are_deterministic_algorithms_enabled()),
            torch.get_float32_matmul_precision(),
            tuple(
                (
                    id(root),
                    id(getattr(root, "__code__", None)),
                    id(getattr(root, "__wrapped__", None)),
                    id(
                        getattr(
                            getattr(root, "__wrapped__", None),
                            "__code__",
                            None,
                        )
                    ),
                )
                for root in roots
            ),
            self._producer_module_function_identity(owner.msa_angles),
            tuple(
                (
                    name,
                    id(owner._series_from_active_samples.__globals__.get(name)),
                    id(
                        getattr(
                            owner._series_from_active_samples.__globals__.get(
                                name
                            ),
                            "__init__",
                            None,
                        )
                    ),
                )
                for name in ("FrequencySeries", "TorchArrayData")
            ),
            _device_identity(device_index),
            environment,
        )

    def _public_key(
        self,
        params,
        device_index,
        stream_id,
        *,
        implementation_identity=None,
    ):
        """Freeze the exact request before model/grid construction."""

        if type(params) is not dict or not all(
            type(name) is str for name in params
        ):
            raise TypeError("CUDA aggregate public parameters are not plain")
        if implementation_identity is None:
            implementation_identity = self._implementation_identity(
                device_index
            )
        return (
            implementation_identity,
            self._owner._freeze_coprecessing_plan_value(
                self._terminal_free_params(params)
            ),
            (os.getpid(), threading.get_ident(), device_index, stream_id),
        )

    @staticmethod
    def _public_layout(
        model,
        frequencies,
        modes,
        uniform_grid_metadata,
        public_layout_metadata,
    ):
        """Validate and freeze everything needed after an aggregate hit."""

        inputs = model.inputs
        if (
            type(uniform_grid_metadata) is not tuple
            or len(uniform_grid_metadata) != 3
            or type(public_layout_metadata) is not tuple
            or len(public_layout_metadata) != 4
        ):
            return None
        first_bin, stop_bin, delta_f = uniform_grid_metadata
        npoints, public_first, public_stop, public_delta = (
            public_layout_metadata
        )
        if (
            type(npoints) is not int
            or type(first_bin) is not int
            or type(stop_bin) is not int
            or type(delta_f) is not float
            or public_first != first_bin
            or public_stop != stop_bin
            or type(public_delta) is not float
            or public_delta != delta_f
            or not math.isfinite(delta_f)
            or delta_f <= 0.0
            or first_bin < 0
            or stop_bin <= first_bin
            or npoints < stop_bin
            or type(modes) is not tuple
            or not modes
            or type(frequencies) is not torch.Tensor
            or frequencies.ndim != 1
            or frequencies.numel() != stop_bin - first_bin
            or type(inputs.polarization_rotation) is not float
            or not math.isfinite(inputs.polarization_rotation)
            or type(inputs.device) is not torch.device
            or inputs.device.type != "cuda"
            or inputs.real_dtype is not torch.float64
            or inputs.complex_dtype is not torch.complex128
            or frequencies.device != inputs.device
            or frequencies.dtype is not torch.float64
        ):
            return None
        return _PublicLayout(
            npoints,
            first_bin,
            stop_bin,
            delta_f,
            inputs.polarization_rotation,
            inputs.complex_dtype,
            inputs.device,
        )

    def _key(
        self,
        model,
        frequencies,
        params,
        modes,
        active_f_max,
        uniform_grid_metadata,
        plan,
        public_layout_metadata,
    ):
        owner = self._owner
        plan_cache = self._plan_cache()
        inputs = model.inputs
        device_index = self._resolved_device_index(frequencies.device)
        stream_id = int(torch.cuda.current_stream(device_index).cuda_stream)
        plan_descriptors = plan_cache._descriptors(plan)
        implementation_identity = self._implementation_identity(device_index)
        public_key = self._public_key(
            params,
            device_index,
            stream_id,
            implementation_identity=implementation_identity,
        )
        frozen_modes = tuple(modes)
        public_layout = self._public_layout(
            model,
            frequencies,
            frozen_modes,
            uniform_grid_metadata,
            public_layout_metadata,
        )
        if public_layout is None:
            raise TypeError("CUDA aggregate public layout is unsupported")
        model_scalars = (
            inputs.theta_jn,
            inputs.alpha0,
            inputs.epsilon0,
            model.final_spin,
            bool(model.msa_reference_angles_deferred is True),
        )
        # The MSA reference offsets are deterministic products of the frozen
        # terminal-free parameters and guarded producer implementation.  On
        # CUDA they are zero-dimensional device tensors; reading their bytes
        # here would synchronize every warm hit.  The admitted model/plan
        # schema above fixes their production route instead.
        if not all(
            value is None or type(value) in (bool, float)
            for value in model_scalars
        ):
            raise TypeError("CUDA aggregate model scalars are not plain")
        semantic_key = (
            implementation_identity,
            owner._freeze_coprecessing_plan_value(
                self._terminal_free_params(params)
            ),
            owner._freeze_coprecessing_plan_value(model_scalars),
            frozen_modes,
            owner._freeze_coprecessing_plan_value(active_f_max),
            owner._freeze_coprecessing_plan_value(uniform_grid_metadata),
            (
                frequencies.dtype,
                device_index,
                tuple(frequencies.shape),
                tuple(frequencies.stride()),
                int(frequencies.storage_offset()),
                int(frequencies._version),
            ),
            (os.getpid(), threading.get_ident(), device_index, stream_id),
        )
        key = semantic_key, id(plan), plan_descriptors
        return (
            key,
            semantic_key,
            owner._coprecessing_plan_deep_size(key),
            plan_descriptors,
            device_index,
            stream_id,
            public_key,
            public_layout,
        )

    def prepare(
        self,
        model,
        frequencies,
        params,
        modes,
        active_f_max,
        uniform_grid_metadata,
        plan,
        proof,
        *,
        public_layout_metadata=None,
        note=True,
    ):
        """Build a metadata-only exact request token, or fail closed."""

        try:
            if not self.enabled():
                return None
            if self._proof_current(proof, params) is not True:
                if note:
                    self._note("ineligible")
                return None
            budget = self.budget()
            supported = self._request_supported(
                model,
                frequencies,
                params,
                modes,
                active_f_max,
                uniform_grid_metadata,
                plan,
            )
            if budget is None or not supported:
                if note:
                    self._note("ineligible")
                return None
            (
                key,
                semantic_key,
                key_bytes,
                plan_descriptors,
                device_index,
                stream_id,
                public_key,
                public_layout,
            ) = self._key(
                model,
                frequencies,
                params,
                modes,
                active_f_max,
                uniform_grid_metadata,
                plan,
                public_layout_metadata,
            )
            return _Request(
                key,
                semantic_key,
                key_bytes,
                budget,
                plan,
                plan_descriptors,
                device_index,
                threading.get_ident(),
                stream_id,
                proof,
                params,
                public_key,
                public_layout,
                tuple(modes),
            )
        except Exception:
            if note:
                self._note("ineligible")
            return None

    def prepare_public(self, params, proof, *, note=True):
        """Build an exact pre-model public lookup token, or fail closed."""

        try:
            if not self.public_fastpath_enabled():
                return None
            if (
                not self.enabled()
                or not self._plan_cache().enabled()
                or self._proof_current(proof, params) is not True
                or not self._public_producers_supported()
                or not self._owner._coprecessing_plan_coefficient_tables_supported()
            ):
                if note:
                    self._note("ineligible")
                return None
            budget = self.budget()
            if budget is None:
                if note:
                    self._note("ineligible")
                return None
            device_index = proof.device_index
            stream_id = proof.stream_id
            if (
                type(device_index) is not int
                or type(stream_id) is not int
                or torch.cuda.current_device() != device_index
                or int(torch.cuda.current_stream(device_index).cuda_stream)
                != stream_id
            ):
                if note:
                    self._note("ineligible")
                return None
            key = self._public_key(params, device_index, stream_id)
            long_asc_nodes = float(
                0.0
                if params.get("long_asc_nodes") is None
                else params["long_asc_nodes"]
            )
            if not math.isfinite(long_asc_nodes):
                raise ValueError("non-finite terminal longitude")
            return _PublicRequest(
                key,
                budget,
                device_index,
                threading.get_ident(),
                stream_id,
                proof,
                params,
                long_asc_nodes,
                self._owner._freeze_coprecessing_plan_value(
                    long_asc_nodes
                ),
            )
        except Exception:
            if note:
                self._note("ineligible")
            return None

    @staticmethod
    def _same_public_request(current, request):
        return (
            type(current) is _PublicRequest
            and current.key == request.key
            and current.budget == request.budget
            and current.device_index == request.device_index
            and current.thread_id == request.thread_id
            and current.stream_id == request.stream_id
            and current.proof is request.proof
            and current.params is request.params
            and current.long_asc_nodes_key == request.long_asc_nodes_key
        )

    def _public_entry_valid(self, request, entry, *, debug_fingerprint):
        owner = self._owner
        plan_cache = self._plan_cache()
        if (
            type(request) is not _PublicRequest
            or type(entry) is not _Entry
            or entry.public_key != request.key
            or entry.device_index != request.device_index
            or entry.thread_id != request.thread_id
            or entry.stream_id != request.stream_id
            or type(entry.public_layout) is not _PublicLayout
            or type(entry.modes) is not tuple
            or not entry.modes
            or type(entry.plan) is not owner._CoprecessingPlan
            or type(entry.plan.active_modes) is not tuple
            or entry.plan.reference_angle_core is not None
            or len(entry.plan.active_modes) != len(entry.modes)
        ):
            return False
        layout = entry.public_layout
        active_size = layout.stop_bin - layout.first_bin
        if (
            type(layout.npoints) is not int
            or type(layout.first_bin) is not int
            or type(layout.stop_bin) is not int
            or type(layout.delta_f) is not float
            or not math.isfinite(layout.delta_f)
            or layout.delta_f <= 0.0
            or layout.first_bin < 0
            or layout.stop_bin <= layout.first_bin
            or layout.npoints < layout.stop_bin
            or type(layout.polarization_rotation) is not float
            or not math.isfinite(layout.polarization_rotation)
            or layout.complex_dtype is not torch.complex128
            or type(layout.device) is not torch.device
            or self._resolved_device_index(layout.device)
            != request.device_index
        ):
            return False
        try:
            if any(
                type(item) is not tuple
                or len(item) != 2
                or item[0] != expected_mode
                for expected_mode, item in zip(
                    entry.modes,
                    entry.plan.active_modes,
                )
            ):
                return False
            aggregate = self._aggregate_tensors(entry.plus, entry.cross)
            aggregate_pointers = []
            for value in aggregate:
                storage = value.untyped_storage()
                if (
                    type(value) is not torch.Tensor
                    or value.layout is not torch.strided
                    or value.device != layout.device
                    or value.dtype is not torch.complex128
                    or tuple(value.shape) != (active_size,)
                    or tuple(value.stride()) != (1,)
                    or not value.is_contiguous()
                    or value.storage_offset() != 0
                    or value._base is not None
                    or value._version != 0
                    or value.is_inference()
                    or value.is_conj()
                    or value.is_neg()
                    or value.requires_grad
                    or value.grad_fn is not None
                    or owner.IMRPhenomX_utils._tensor_has_forward_ad(value)
                    or storage.data_ptr() == 0
                    or storage.nbytes() != value.numel() * value.element_size()
                ):
                    return False
                aggregate_pointers.append(storage.data_ptr())
            if len(set(aggregate_pointers)) != 2:
                return False
            if self._descriptors(entry.plus, entry.cross) != entry.descriptors:
                return False
            plan_tensors = tuple(plan_cache._plan_tensors(entry.plan))
            if (
                not plan_tensors
                or plan_cache._descriptors(entry.plan)
                != entry.plan_descriptors
            ):
                return False
            for value in plan_tensors:
                if (
                    type(value) is not torch.Tensor
                    or value.layout is not torch.strided
                    or value.device != layout.device
                    or value.dtype is not torch.complex128
                    or tuple(value.shape) != (active_size,)
                    or not value.is_contiguous()
                    or value.is_conj()
                    or value.is_neg()
                    or value.requires_grad
                    or value.grad_fn is not None
                    or owner.IMRPhenomX_utils._tensor_has_forward_ad(value)
                    or value.untyped_storage().data_ptr() == 0
                ):
                    return False
        except Exception:
            return False
        valid = True
        if debug_fingerprint:
            self._counters["debug_fingerprint_checks"] += 1
            try:
                valid = (
                    self._fingerprint(entry.plus, entry.cross)
                    == entry.fingerprint
                )
            except Exception:
                valid = False
            if not valid:
                self._counters["debug_fingerprint_failures"] += 1
        return valid

    def lookup_public(self, request):
        """Return a validated pre-model hit without counting it prematurely."""

        if type(request) is not _PublicRequest:
            return None
        current = self.prepare_public(
            request.params,
            request.proof,
            note=False,
        )
        if not self._same_public_request(current, request):
            self._note("races")
            self._note("public_races")
            return None
        with self._lock:
            self._reset_if_forked_locked()
            self._trim_locked(request.budget)
            match = next(
                (
                    (entry_key, entry)
                    for entry_key, entry in reversed(self._entries.items())
                    if entry.public_key == request.key
                ),
                None,
            )
            if match is None:
                self._counters["public_misses"] += 1
                return None
            entry_key, entry = match
            if not self._public_entry_valid(
                request,
                entry,
                debug_fingerprint=self.debug_fingerprint_enabled(),
            ):
                self._entries.pop(entry_key, None)
                self._size -= entry.nbytes
                self._counters["invalidations"] += 1
                self._counters["public_invalidations"] += 1
                self._counters["public_misses"] += 1
                return None
            return _PublicHit(entry_key, entry, request)

    def finish_public_hit(self, hit):
        """Commit hit accounting only after fresh outputs were materialized."""

        if type(hit) is not _PublicHit:
            return False
        request = hit.request
        current = self.prepare_public(
            request.params,
            request.proof,
            note=False,
        )
        if not self._same_public_request(current, request):
            self._note("races")
            self._note("public_races")
            return False
        with self._lock:
            self._reset_if_forked_locked()
            entry = self._entries.get(hit.entry_key)
            if entry is not hit.entry or not self._public_entry_valid(
                request,
                hit.entry,
                debug_fingerprint=False,
            ):
                self._counters["races"] += 1
                self._counters["public_races"] += 1
                return False
            self._entries.move_to_end(hit.entry_key)
            self._counters["hits"] += 1
            self._counters["public_hits"] += 1
            return True

    @staticmethod
    def _aggregate_tensors(plus, cross):
        return plus, cross

    def _schema_supported(self, plus, cross, frequencies, *, privately_owned):
        owner = self._owner
        try:
            frequency_pointer = frequencies.untyped_storage().data_ptr()
            pointers = []
            for value in self._aggregate_tensors(plus, cross):
                if (
                    type(value) is not torch.Tensor
                    or value.layout is not torch.strided
                    or value.device != frequencies.device
                    or value.dtype is not torch.complex128
                    or tuple(value.shape) != tuple(frequencies.shape)
                    or not value.is_contiguous()
                    or value.is_conj()
                    or value.is_neg()
                    or value.requires_grad
                    or value.grad_fn is not None
                    or owner.IMRPhenomX_utils._tensor_has_forward_ad(value)
                ):
                    return False
                storage = value.untyped_storage()
                pointer = storage.data_ptr()
                if pointer == 0 or pointer == frequency_pointer:
                    return False
                if privately_owned and (
                    value.is_inference()
                    or value._version != 0
                    or value.storage_offset() != 0
                    or value._base is not None
                    or storage.nbytes() != value.numel() * value.element_size()
                ):
                    return False
                pointers.append(pointer)
        except Exception:
            return False
        return not privately_owned or len(pointers) == len(set(pointers))

    @staticmethod
    def _copy(plus, cross):
        def owned(value):
            with torch.inference_mode(False), torch.no_grad():
                return value.detach().clone(memory_format=torch.contiguous_format)

        return owned(plus), owned(cross)

    @staticmethod
    def _descriptor(value):
        storage = value.untyped_storage()
        return _TensorDescriptor(
            id(value),
            int(storage._cdata),
            int(storage.data_ptr()),
            int(storage.nbytes()),
            int(value.data_ptr()),
            int(value._version),
            value.dtype,
            value.device,
            tuple(value.shape),
            tuple(value.stride()),
            int(value.storage_offset()),
            id(value._base) if value._base is not None else None,
            bool(value.requires_grad),
            id(value.grad_fn) if value.grad_fn is not None else None,
            bool(value.is_inference()),
            bool(value.is_conj()),
            bool(value.is_neg()),
        )

    def _descriptors(self, plus, cross):
        return tuple(
            self._descriptor(value)
            for value in self._aggregate_tensors(plus, cross)
        )

    def _fingerprint(self, plus, cross):
        """Synchronize and hash device bytes; cold/debug use only."""

        hashes = []
        for value in self._aggregate_tensors(plus, cross):
            raw = (
                value.detach()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
            hashes.append(
                (value.dtype, tuple(value.shape), hashlib.sha256(raw).digest())
            )
        return tuple(hashes)

    def _entry_bytes(
        self,
        request,
        plus,
        cross,
        descriptors,
        fingerprint,
    ):
        owner = self._owner
        plan_cache = self._plan_cache()
        storage_bytes = 0
        seen = set()
        tensors = self._aggregate_tensors(plus, cross) + tuple(
            plan_cache._plan_tensors(request.plan)
        )
        for value in tensors:
            storage = value.untyped_storage()
            identity = int(storage._cdata)
            if identity not in seen:
                seen.add(identity)
                storage_bytes += int(storage.nbytes())
        return (
            request.key_bytes
            + owner._coprecessing_plan_deep_size(descriptors)
            + owner._coprecessing_plan_deep_size(fingerprint)
            + owner._coprecessing_plan_deep_size(request.plan_descriptors)
            + storage_bytes
            + CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_ENTRY_OVERHEAD
        )

    def _trim_locked(self, budget):
        while self._entries and (
            self._size > budget
            or len(self._entries)
            > CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_MAX_ENTRIES
        ):
            _, entry = self._entries.popitem(last=False)
            self._size -= entry.nbytes
            self._counters["evictions"] += 1

    @staticmethod
    def _same_request(current, request):
        return (
            type(current) is _Request
            and current.key == request.key
            and current.budget == request.budget
            and current.plan is request.plan
            and current.plan_descriptors == request.plan_descriptors
            and current.device_index == request.device_index
            and current.thread_id == request.thread_id
            and current.stream_id == request.stream_id
            and current.proof is request.proof
            and current.params is request.params
        )

    def lookup(self, request, frequencies, modes, revalidate):
        """Return a validated private aggregate without a default D2H read."""

        if type(request) is not _Request:
            return None
        try:
            current = revalidate()
            proof_current = (
                self._proof_current(request.proof, request.params) is True
            )
        except Exception:
            current = None
            proof_current = False
        if not proof_current or not self._same_request(current, request):
            self._note("races")
            return None
        plan_cache = self._plan_cache()
        with self._lock:
            self._reset_if_forked_locked()
            self._trim_locked(request.budget)
            entry = self._entries.get(request.key)
            if entry is None:
                self._counters["misses"] += 1
                return None
            valid = (
                entry.plan is request.plan
                and entry.semantic_key == request.semantic_key
                and entry.device_index == request.device_index
                and entry.thread_id == request.thread_id
                and entry.stream_id == request.stream_id
                and entry.public_key == request.public_key
                and entry.modes == request.modes
                and entry.plan_descriptors == request.plan_descriptors
                and plan_cache._schema_supported(
                    entry.plan,
                    frequencies,
                    modes,
                    privately_owned=False,
                )
                and plan_cache._descriptors(entry.plan)
                == request.plan_descriptors
                and self._schema_supported(
                    entry.plus,
                    entry.cross,
                    frequencies,
                    privately_owned=True,
                )
            )
            if valid:
                try:
                    valid = (
                        self._descriptors(entry.plus, entry.cross)
                        == entry.descriptors
                    )
                except Exception:
                    valid = False
            if valid and self.debug_fingerprint_enabled():
                self._counters["debug_fingerprint_checks"] += 1
                try:
                    valid = (
                        self._fingerprint(entry.plus, entry.cross)
                        == entry.fingerprint
                    )
                except Exception:
                    valid = False
                if not valid:
                    self._counters["debug_fingerprint_failures"] += 1
            if not valid:
                self._entries.pop(request.key, None)
                self._size -= entry.nbytes
                self._counters["invalidations"] += 1
                self._counters["misses"] += 1
                return None
            self._entries.move_to_end(request.key)
            self._counters["hits"] += 1
            return entry.plus, entry.cross

    def store(self, request, plus, cross, frequencies, modes, revalidate):
        """Own, byte-check, revalidate, and insert one exact aggregate."""

        if type(request) is not _Request or not self._schema_supported(
            plus,
            cross,
            frequencies,
            privately_owned=False,
        ):
            return
        try:
            candidate_plus, candidate_cross = self._copy(plus, cross)
            if not self._schema_supported(
                candidate_plus,
                candidate_cross,
                frequencies,
                privately_owned=True,
            ):
                self._note("canary_failures")
                return
            source_fingerprint = self._fingerprint(plus, cross)
            fingerprint = self._fingerprint(candidate_plus, candidate_cross)
            if source_fingerprint != fingerprint:
                self._note("canary_failures")
                return
            descriptors = self._descriptors(candidate_plus, candidate_cross)
            entry = _Entry(
                candidate_plus,
                candidate_cross,
                descriptors,
                fingerprint,
                self._entry_bytes(
                    request,
                    candidate_plus,
                    candidate_cross,
                    descriptors,
                    fingerprint,
                ),
                request.plan,
                request.semantic_key,
                request.device_index,
                request.thread_id,
                request.stream_id,
                request.public_key,
                request.public_layout,
                request.modes,
                request.plan_descriptors,
            )
        except Exception:
            self._note("canary_failures")
            return

        with self._lock:
            self._reset_if_forked_locked()
            # Revalidation calls only the plan cache's lock-free admission,
            # schema, descriptor, and key helpers.  It never performs a plan
            # lookup/store while this aggregate lock is held.
            try:
                current = revalidate()
                proof_current = (
                    self._proof_current(request.proof, request.params) is True
                )
            except Exception:
                current = None
                proof_current = False
            if not proof_current or not self._same_request(current, request):
                self._counters["races"] += 1
                return
            if entry.nbytes > request.budget:
                self._counters["oversize"] += 1
                return
            if request.key in self._entries:
                self._counters["races"] += 1
                return
            stale_keys = tuple(
                key
                for key, existing in self._entries.items()
                if existing.semantic_key == request.semantic_key
            )
            for key in stale_keys:
                stale = self._entries.pop(key)
                self._size -= stale.nbytes
                self._counters["evictions"] += 1
            self._entries[request.key] = entry
            self._size += entry.nbytes
            self._counters["stores"] += 1
            self._trim_locked(request.budget)


def make_cuda_aggregate_preterminal_twist_cache(
    owner,
    proof_current,
):
    """Construct and fork-register one cache owned by the XPHM module."""

    cache = CudaAggregatePreterminalTwistCache(
        owner,
        proof_current,
    )
    register_at_fork = getattr(os, "register_at_fork", None)
    if register_at_fork is not None:
        register_at_fork(after_in_child=cache.after_fork)
    return cache
