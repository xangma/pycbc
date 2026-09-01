"""Bounded, opt-in CUDA cache for exact XPHM co-precessing plans.

This module is deliberately private and owns no waveform arithmetic.  It
retains detached, unaliased copies of the carrier and active co-precessing mode
tensors produced by one exact regular-grid ``n_batch == 1`` request.  Cache
hits perform metadata/version checks only.  The optional debug fingerprint
gate performs the expensive device-to-host integrity check.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib
import math
import os
import threading
from typing import NamedTuple

import torch


CUDA_COPRECESSING_PLAN_CACHE_ENV = (
    "PYCBC_IMRPHENOMXPHM_CUDA_COPRECESSING_PLAN_CACHE"
)
CUDA_COPRECESSING_PLAN_CACHE_BYTES_ENV = (
    "PYCBC_IMRPHENOMXPHM_CUDA_COPRECESSING_PLAN_CACHE_BYTES"
)
CUDA_COPRECESSING_PLAN_CACHE_DEBUG_FINGERPRINT_ENV = (
    "PYCBC_IMRPHENOMXPHM_CUDA_COPRECESSING_PLAN_CACHE_DEBUG_FINGERPRINT"
)
CUDA_COPRECESSING_PLAN_CACHE_DEFAULT_BYTES = 64 * 1024 * 1024
CUDA_COPRECESSING_PLAN_CACHE_MAX_ENTRIES = 32
CUDA_COPRECESSING_PLAN_CACHE_ENTRY_OVERHEAD = 1024
CUDA_COPRECESSING_PLAN_CACHE_IMPLEMENTATION = 1


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


class _Entry(NamedTuple):
    plan: object
    descriptors: tuple
    fingerprint: tuple
    nbytes: int
    device_index: int
    thread_id: int
    stream_id: int


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


class CudaCoprecessingPlanCache:
    """One process-local, same-stream CUDA plan LRU."""

    gate_env = CUDA_COPRECESSING_PLAN_CACHE_ENV
    budget_env = CUDA_COPRECESSING_PLAN_CACHE_BYTES_ENV
    debug_fingerprint_env = (
        CUDA_COPRECESSING_PLAN_CACHE_DEBUG_FINGERPRINT_ENV
    )

    def __init__(self, owner):
        self._owner = owner
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

    def budget(self):
        value = os.environ.get(self.budget_env)
        if value is None:
            return CUDA_COPRECESSING_PLAN_CACHE_DEFAULT_BYTES
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
        """Drop inherited device tensors and replace an inherited lock."""

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
                max_entries=CUDA_COPRECESSING_PLAN_CACHE_MAX_ENTRIES,
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

    @staticmethod
    def _compile_requested(owner):
        name = "PYCBC_TORCH_COMPILE"
        value = os.environ.get(name)
        return False if value is None else owner._parse_switch(name, value)

    def _request_supported(
        self,
        model,
        frequencies,
        params,
        modes,
        active_f_max,
        uniform_grid_metadata,
    ):
        owner = self._owner
        if (
            not owner._plain_request_supported(params)
            or not owner._manual_exact_coverage_supported(params)
            or self._compile_requested(owner)
            or torch.is_inference_mode_enabled()
            or type(uniform_grid_metadata) is not tuple
            or len(uniform_grid_metadata) != 3
            or type(active_f_max) is not float
            or not math.isfinite(active_f_max)
        ):
            return False

        first_bin, stop_bin, delta_f = uniform_grid_metadata
        n_batch = params.get("n_batch")
        inputs = model.inputs
        state = owner._scheme.mgr.state
        if (
            (n_batch is not None and (type(n_batch) is not int or n_batch != 1))
            or type(first_bin) is not int
            or type(stop_bin) is not int
            or type(delta_f) is not float
            or first_bin < 0
            or stop_bin <= first_bin
            or not math.isfinite(delta_f)
            or delta_f <= 0.0
            or type(state) is not owner._scheme.TorchScheme
            or state.torch_device.type != "cuda"
            or type(inputs.device) is not torch.device
            or inputs.device.type != "cuda"
            or inputs.real_dtype is not torch.float64
            or inputs.complex_dtype is not torch.complex128
            or type(frequencies) is not torch.Tensor
            or frequencies.layout is not torch.strided
            or frequencies.device.type != "cuda"
            or frequencies.dtype is not torch.float64
            or frequencies.ndim != 1
            or frequencies.numel() != stop_bin - first_bin
            or frequencies.numel() == 0
            or frequencies.stride() != (1,)
            or not frequencies.is_contiguous()
            or frequencies.storage_offset() != 0
            or frequencies._base is not None
            or frequencies.is_inference()
            or frequencies._version != 0
            or frequencies.is_conj()
            or frequencies.is_neg()
            or frequencies.requires_grad
            or frequencies.grad_fn is not None
            or owner.IMRPhenomX_utils._tensor_has_forward_ad(frequencies)
            or type(modes) is not list
            or not modes
        ):
            return False

        device_index = self._resolved_device_index(frequencies.device)
        input_device_index = self._resolved_device_index(inputs.device)
        scheme_device_index = self._resolved_device_index(state.torch_device)
        if (
            device_index is None
            or input_device_index != device_index
            or scheme_device_index != device_index
        ):
            return False

        try:
            capture = torch.cuda.is_current_stream_capturing()
            storage = frequencies.untyped_storage()
            canonical_modes = tuple(
                mode for mode in owner._COPRECESSING_MODES if mode in modes
            )
            expected_first = int(float(params["f_lower"]) / delta_f)
            expected_stop = int(active_f_max / delta_f) + 1
        except Exception:
            return False
        return (
            capture is False
            and tuple(modes) == canonical_modes
            and all(
                type(mode) is tuple
                and len(mode) == 2
                and all(type(value) is int for value in mode)
                for mode in modes
            )
            and first_bin == expected_first
            and stop_bin == expected_stop
            and float(params["delta_f"]) == delta_f
            and (stop_bin - 1) * delta_f <= active_f_max
            and storage.data_ptr() != 0
            and storage.nbytes()
            == frequencies.numel() * frequencies.element_size()
        )

    @staticmethod
    def _table_identity(owner, value):
        if type(value) is not torch.Tensor:
            return id(value), None
        try:
            return (
                id(value),
                value.data_ptr(),
                value._version,
                value.dtype,
                value.device,
                tuple(value.shape),
                tuple(value.stride()),
                value.storage_offset(),
                id(value._base) if value._base is not None else None,
                bool(value.requires_grad),
                id(value.grad_fn) if value.grad_fn is not None else None,
                bool(value.is_inference()),
                bool(owner.IMRPhenomX_utils._tensor_has_forward_ad(value)),
            )
        except Exception:
            return id(value), None

    def _implementation_identity(self, device_index):
        owner = self._owner
        try:
            phase, amp, phase_current, amp_current = (
                owner._coprecessing_plan_coefficient_tables()
            )
        except Exception:
            phase = amp = phase_current = amp_current = None
        roots = (
            owner._imrphenomxphm_fd_torch,
            owner._build_model,
            owner._build_coprecessing_plan,
            owner._xas_samples,
            owner._active_mode_samples,
            owner._coprecessing_params,
            owner._coprecessing_final_spin,
        )
        ignored_environment = {
            self.budget_env,
            self.debug_fingerprint_env,
            getattr(owner, "_COPRECESSING_PLAN_CACHE_BYTES_ENV", ""),
            getattr(
                owner,
                "_CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_ENV",
                "",
            ),
            getattr(
                owner,
                "_CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_BYTES_ENV",
                "",
            ),
            getattr(
                owner,
                "_CUDA_AGGREGATE_PRETERMINAL_TWIST_CACHE_DEBUG_FINGERPRINT_ENV",
                "",
            ),
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
            CUDA_COPRECESSING_PLAN_CACHE_IMPLEMENTATION,
            getattr(owner, "_COPRECESSING_PLAN_CACHE_IMPLEMENTATION", None),
            torch.__version__,
            getattr(torch.version, "git_version", None),
            torch.get_default_dtype(),
            bool(torch.are_deterministic_algorithms_enabled()),
            torch.get_float32_matmul_precision(),
            tuple(
                (id(root), id(getattr(root, "__code__", None))) for root in roots
            ),
            tuple(
                (
                    id(value),
                    id(getattr(value, "__new__", None)),
                    id(getattr(getattr(value, "__new__", None), "__code__", None)),
                )
                for value in (owner._SequenceCore, owner._CoprecessingPlan)
            ),
            owner._freeze_coprecessing_plan_value(
                owner._XAS_MODE_POLARIZATION_FACTOR
            ),
            self._table_identity(owner, owner._COPRECESSING_PLAN_PHASE_TABLE_SOURCE),
            self._table_identity(owner, owner._COPRECESSING_PLAN_AMP_TABLE_SOURCE),
            self._table_identity(owner, phase),
            self._table_identity(owner, amp),
            self._table_identity(owner, phase_current),
            self._table_identity(owner, amp_current),
            _device_identity(device_index),
            environment,
        )

    def _key(
        self,
        model,
        frequencies,
        params,
        modes,
        active_f_max,
        uniform_grid_metadata,
        request_proof,
    ):
        owner = self._owner
        inputs = model.inputs
        # Only fields consumed by the carrier/mode plan belong here.  Viewing
        # geometry remains request-local, so inclination/polarization changes
        # can reuse the same exact intrinsic plan.
        input_fields = (
            "tidal_version",
            "mass1",
            "mass2",
            "chi1_l",
            "chi2_l",
            "chip",
            "spin_aligned",
            "spin_perp",
            "spin1",
            "spin2",
            "lambda1",
            "lambda2",
            "dquad1",
            "dquad2",
            "prec_version",
            "final_spin_mod",
            "distance",
            "carrier_phase",
            "f_ref",
            "total_mass",
            "total_mass_seconds",
            "eta",
            "device",
            "real_dtype",
            "complex_dtype",
        )
        device_index = self._resolved_device_index(frequencies.device)
        stream_id = int(torch.cuda.current_stream(device_index).cuda_stream)
        aligned_params = owner._coprecessing_params(params, inputs)
        key = (
            self._implementation_identity(device_index),
            tuple(
                (
                    name,
                    owner._freeze_coprecessing_plan_value(getattr(inputs, name)),
                )
                for name in input_fields
            ),
            owner._freeze_coprecessing_plan_value(aligned_params),
            bool(model.msa_reference_angles_deferred is True),
            owner._freeze_coprecessing_plan_value(model.final_spin),
            tuple(modes),
            owner._freeze_coprecessing_plan_value(active_f_max),
            owner._freeze_coprecessing_plan_value(uniform_grid_metadata),
            bool(owner._request_proof_plan_current(request_proof)),
            (
                frequencies.dtype,
                device_index,
                tuple(frequencies.shape),
                tuple(frequencies.stride()),
                frequencies.storage_offset(),
                frequencies._version,
            ),
            (os.getpid(), threading.get_ident(), device_index, stream_id),
        )
        return key, owner._coprecessing_plan_deep_size(key)

    @staticmethod
    def _plan_tensors(plan):
        return (plan.carrier,) + tuple(
            samples for _, samples in plan.active_modes
        )

    def _schema_supported(self, plan, frequencies, modes, *, privately_owned):
        owner = self._owner
        if (
            type(plan) is not owner._CoprecessingPlan
            or type(plan.active_modes) is not tuple
            or len(plan.active_modes) != len(modes)
            or plan.reference_angle_core is not None
        ):
            return False
        for expected_mode, item in zip(modes, plan.active_modes):
            if type(item) is not tuple or len(item) != 2 or item[0] != expected_mode:
                return False

        try:
            frequency_pointer = frequencies.untyped_storage().data_ptr()
            pointers = []
            for value in self._plan_tensors(plan):
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

    def _copy(self, plan):
        owner = self._owner

        def owned(value):
            with torch.inference_mode(False), torch.no_grad():
                return value.detach().clone(memory_format=torch.contiguous_format)

        return owner._CoprecessingPlan(
            owned(plan.carrier),
            tuple((mode, owned(samples)) for mode, samples in plan.active_modes),
            None,
        )

    def _descriptor(self, value):
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

    def _descriptors(self, plan):
        return tuple(self._descriptor(value) for value in self._plan_tensors(plan))

    def _fingerprint(self, plan):
        """Synchronize and hash device bytes; cold/debug use only."""

        hashes = []
        for value in self._plan_tensors(plan):
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
                (
                    value.dtype,
                    tuple(value.shape),
                    hashlib.sha256(raw).digest(),
                )
            )
        return tuple(hashes)

    def _entry_bytes(self, plan, descriptors, fingerprint, key_bytes):
        owner = self._owner
        storage_bytes = 0
        seen = set()
        for value in self._plan_tensors(plan):
            storage = value.untyped_storage()
            identity = storage._cdata
            if identity not in seen:
                seen.add(identity)
                storage_bytes += storage.nbytes()
        return (
            key_bytes
            + owner._coprecessing_plan_deep_size(plan)
            + owner._coprecessing_plan_deep_size(descriptors)
            + owner._coprecessing_plan_deep_size(fingerprint)
            + storage_bytes
            + CUDA_COPRECESSING_PLAN_CACHE_ENTRY_OVERHEAD
        )

    def _trim_locked(self, budget):
        while self._entries and (
            self._size > budget
            or len(self._entries) > CUDA_COPRECESSING_PLAN_CACHE_MAX_ENTRIES
        ):
            _, entry = self._entries.popitem(last=False)
            self._size -= entry.nbytes
            self._counters["evictions"] += 1

    def _lookup(self, key, frequencies, modes, budget):
        with self._lock:
            self._reset_if_forked_locked()
            self._trim_locked(budget)
            entry = self._entries.get(key)
            if entry is None:
                self._counters["misses"] += 1
                return None
            valid = self._schema_supported(
                entry.plan,
                frequencies,
                modes,
                privately_owned=True,
            )
            if valid:
                try:
                    valid = self._descriptors(entry.plan) == entry.descriptors
                except Exception:
                    valid = False
            if valid and self.debug_fingerprint_enabled():
                self._counters["debug_fingerprint_checks"] += 1
                try:
                    valid = self._fingerprint(entry.plan) == entry.fingerprint
                except Exception:
                    valid = False
                if not valid:
                    self._counters["debug_fingerprint_failures"] += 1
            if not valid:
                self._entries.pop(key, None)
                self._size -= entry.nbytes
                self._counters["invalidations"] += 1
                self._counters["misses"] += 1
                return None
            self._entries.move_to_end(key)
            self._counters["hits"] += 1
            return entry.plan

    def _store(
        self,
        key,
        key_bytes,
        plan,
        frequencies,
        modes,
        budget,
        revalidate,
    ):
        if not self._schema_supported(
            plan,
            frequencies,
            modes,
            privately_owned=False,
        ):
            return
        try:
            candidate = self._copy(plan)
            if not self._schema_supported(
                candidate,
                frequencies,
                modes,
                privately_owned=True,
            ):
                self._note("canary_failures")
                return
            source_fingerprint = self._fingerprint(plan)
            fingerprint = self._fingerprint(candidate)
            if source_fingerprint != fingerprint:
                self._note("canary_failures")
                return
            descriptors = self._descriptors(candidate)
            device_index = self._resolved_device_index(frequencies.device)
            stream_id = int(torch.cuda.current_stream(device_index).cuda_stream)
            entry = _Entry(
                candidate,
                descriptors,
                fingerprint,
                self._entry_bytes(
                    candidate,
                    descriptors,
                    fingerprint,
                    key_bytes,
                ),
                device_index,
                threading.get_ident(),
                stream_id,
            )
        except Exception:
            self._note("canary_failures")
            return

        with self._lock:
            self._reset_if_forked_locked()
            try:
                current = revalidate()
            except Exception:
                self._counters["races"] += 1
                return
            if current != (key, key_bytes, budget):
                self._counters["races"] += 1
                return
            if entry.nbytes > budget:
                self._counters["oversize"] += 1
                return
            if key in self._entries:
                self._counters["races"] += 1
                return
            self._entries[key] = entry
            self._size += entry.nbytes
            self._counters["stores"] += 1
            self._trim_locked(budget)

    def get_or_build(
        self,
        model,
        frequencies,
        params,
        modes,
        active_f_max,
        *,
        uniform_grid_metadata,
        request_proof,
        build,
    ):
        """Return a same-stream warm plan, otherwise preserve eager behavior."""

        try:
            enabled = self.enabled()
        except Exception:
            return build()
        if not enabled:
            return build()
        try:
            budget = self.budget()
            supported = self._request_supported(
                model,
                frequencies,
                params,
                modes,
                active_f_max,
                uniform_grid_metadata,
            )
            if budget is None or not supported:
                self._note("ineligible")
                return build()
            if not self._owner._coprecessing_plan_coefficient_tables_supported():
                self.invalidate_all()
                self._note("ineligible")
                return build()
            key, key_bytes = self._key(
                model,
                frequencies,
                params,
                modes,
                active_f_max,
                uniform_grid_metadata,
                request_proof,
            )
            cached = self._lookup(key, frequencies, modes, budget)
        except Exception:
            self._note("ineligible")
            return build()
        if cached is not None:
            return cached

        plan = build()

        def revalidate():
            if not self.enabled():
                return None
            current_budget = self.budget()
            if (
                current_budget is None
                or current_budget != budget
                or not self._request_supported(
                    model,
                    frequencies,
                    params,
                    modes,
                    active_f_max,
                    uniform_grid_metadata,
                )
                or not self._owner._coprecessing_plan_coefficient_tables_supported()
            ):
                return None
            current_key, current_key_bytes = self._key(
                model,
                frequencies,
                params,
                modes,
                active_f_max,
                uniform_grid_metadata,
                request_proof,
            )
            return current_key, current_key_bytes, current_budget

        try:
            current = revalidate()
        except Exception:
            self._note("races")
            return plan
        if current != (key, key_bytes, budget):
            self._note("races")
            return plan
        try:
            self._store(
                key,
                key_bytes,
                plan,
                frequencies,
                modes,
                budget,
                revalidate,
            )
        except Exception:
            pass
        return plan


def make_cuda_coprecessing_plan_cache(owner):
    """Construct and fork-register one cache owned by the XPHM module."""

    cache = CudaCoprecessingPlanCache(owner)
    register_at_fork = getattr(os, "register_at_fork", None)
    if register_at_fork is not None:
        register_at_fork(after_in_child=cache.after_fork)
    return cache
