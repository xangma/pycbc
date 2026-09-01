"""Exact opt-in cache for identical public CPU XPHM results.

The cache sits behind the normalized public regular-grid entry point.  It owns
detached complex128 tensors and reconstructs fresh ``FrequencySeries`` objects
for every hit.  Admission is deliberately narrow: one plain, immutable,
binary64 CPU request, ordinary eager execution, and the already-qualified
exact inference route only.

Warm hits validate ownership, tensor schema, storage identity, and version
without scanning result bytes.  An optional strict-debug gate retains the
expensive SHA integrity check for hostile mutation diagnostics.
"""

from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar
import hashlib
import math
import os
import struct
import sys
import threading
from typing import NamedTuple

import numpy as _np
import torch

from pycbc import lal_compat as _lal
from pycbc import scheme as _scheme
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData


PUBLIC_RESULT_CACHE_ENV = "PYCBC_IMRPHENOMXPHM_PUBLIC_RESULT_CACHE"
PUBLIC_RESULT_CACHE_BYTES_ENV = (
    "PYCBC_IMRPHENOMXPHM_PUBLIC_RESULT_CACHE_BYTES"
)
PUBLIC_RESULT_CACHE_DEBUG_FINGERPRINT_ENV = (
    "PYCBC_IMRPHENOMXPHM_PUBLIC_RESULT_CACHE_DEBUG_FINGERPRINT"
)
PUBLIC_RESULT_CACHE_DEFAULT_BYTES = 16 * 1024 * 1024
PUBLIC_RESULT_CACHE_MAX_ENTRIES = 32
PUBLIC_RESULT_CACHE_METADATA_ENTRIES = 32
PUBLIC_RESULT_CACHE_ENTRY_OVERHEAD = 1024
PUBLIC_RESULT_CACHE_IMPLEMENTATION = 6

_OWNER_IMPLEMENTATION_ROOT_NAMES = (
    "_run_scoped_xphm_request",
    "_run_xphm_request",
    "_dispatch_imrphenomxphm_request",
    "_run_xphm_request_proof_plan",
    "_imrphenomxphm_fd_torch",
    "imrphenomxphm_native_supported",
    "_request_proof_plan_current",
    "_normalize_public_request_parameter_keys",
    "_plain_request_supported",
    "_manual_exact_coverage_supported",
    "_validated_inputs",
    "_xp_params",
    "_requested_coprecessing_modes",
    "_request_bulk_twist_harmonics",
    "_bulk_twist_harmonics",
    "_build_model",
    "_build_coprecessing_plan",
    "_build_reference_mode_msa_angle_core",
    "_bulk_mode_angles",
    "_reference_and_mode_msa_angles",
    "_xas_samples",
    "_active_mode_samples",
    "_carrier_amp_plan_reuse_enabled",
    "_carrier_alignment_result_reuse_supported",
    "_coprecessing_params",
    "_coprecessing_final_spin",
    "_twist_coprecessing_modes",
    "_series_from_active_samples",
    "_phenomx_torch_environment_items",
    "_coprecessing_plan_environment_identity",
    "_SequenceCore",
    "_ReferenceModeMSAAngleCore",
    "_CoprecessingPlan",
)

_OWNER_IMPLEMENTATION_TABLE_NAMES = (
    "_COPRECESSING_PLAN_PHASE_TABLE_SOURCE",
    "_COPRECESSING_PLAN_AMP_TABLE_SOURCE",
)

_CACHE_IMPLEMENTATION_ROOT_NAMES = (
    "_environment_items",
    "_expected_metadata",
    "_tensor_descriptor",
    "_fingerprint",
    "_owned_tensor",
    "_tensor_schema_supported",
    "_series_metadata",
    "_result_supported",
    "_metadata_supported",
    "_descriptor_admission_supported",
    "_entry_valid",
    "_reconstructed_tensor",
    "_reconstruct",
    "_lookup",
    "_store",
)


class _Request(NamedTuple):
    key: tuple
    budget: int
    pid: int
    thread_id: int
    debug_fingerprint: bool
    expected_metadata: tuple


class _TensorDescriptor(NamedTuple):
    identity: int
    storage_identity: int
    storage_pointer: int
    storage_bytes: int
    storage_resizable: bool
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


class _SeriesMetadata(NamedTuple):
    series_type: type
    data_type: type
    delta_f_bytes: bytes
    epoch_type: type
    epoch_seconds: int
    epoch_nanoseconds: int


class _Entry(NamedTuple):
    request_key: tuple
    plus: torch.Tensor
    cross: torch.Tensor
    descriptors: tuple
    fingerprints: tuple
    metadata: tuple
    nbytes: int


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
        "reentrant": 0,
        "debug_fingerprint_checks": 0,
        "debug_fingerprint_failures": 0,
    }


class _IdentityReferences:
    """Keep identity-key producers alive without comparing their values."""

    __slots__ = ("values",)

    def __init__(self, values):
        self.values = values

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return type(other) is _IdentityReferences


def _function_identity(value):
    identity = (
        id(value),
        id(getattr(value, "__code__", None)),
    )
    if isinstance(value, type):
        constructor = getattr(value, "__new__", None)
        identity += (
            id(constructor),
            id(getattr(constructor, "__code__", None)),
        )
    return identity


def _identities_and_references(values):
    identities = []
    references = []
    for value in values:
        code = getattr(value, "__code__", None)
        identity = [id(value), id(code)]
        references.extend((value, code))
        if isinstance(value, type):
            constructor = getattr(value, "__new__", None)
            constructor_code = getattr(constructor, "__code__", None)
            identity.extend((id(constructor), id(constructor_code)))
            references.extend((constructor, constructor_code))
        identities.append(tuple(identity))
    return tuple(identities), tuple(references)


class _PublicResultCache:
    """One bounded, exact, same-process/thread CPU result LRU."""

    gate_env = PUBLIC_RESULT_CACHE_ENV
    budget_env = PUBLIC_RESULT_CACHE_BYTES_ENV
    debug_fingerprint_env = PUBLIC_RESULT_CACHE_DEBUG_FINGERPRINT_ENV

    def __init__(self, owner, dispatch, supported, target):
        self._owner = owner
        self._dispatch = dispatch
        self._supported = supported
        self._target = target
        self._public_entry = None
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._size = 0
        self._counters = _new_counters()
        self._metadata = OrderedDict()
        self._hot_metadata = None
        self._active = ContextVar(
            "imrphenomxphm_public_result_cache_active",
            default=False,
        )
        self._validate_bound_function(
            dispatch,
            "_dispatch_imrphenomxphm_request",
        )
        self._validate_bound_function(
            supported,
            "imrphenomxphm_native_supported",
        )
        self._validate_bound_function(target, "_imrphenomxphm_fd_torch")

    def _validate_bound_function(self, function, name):
        if not (
            type(function) is type(_function_identity)
            and function.__module__ == self._owner.__name__
            and function.__name__ == name
            and function.__qualname__ == name
            and function.__globals__ is self._owner.__dict__
        ):
            raise TypeError("invalid XPHM public-result cache target")

    def bind_public_entry(self, public_entry):
        """Bind the sole normalized public entry exactly once."""

        if self._public_entry is not None:
            raise RuntimeError("XPHM public-result entry is already bound")
        self._validate_bound_function(
            public_entry,
            "imrphenomxphm_fd_torch",
        )
        self._public_entry = public_entry

    def _bindings_current(self):
        owner = self._owner
        entry = self._public_entry
        return (
            entry is not None
            and owner.__dict__.get("imrphenomxphm_fd_torch") is entry
            and owner.__dict__.get("_dispatch_imrphenomxphm_request")
            is self._dispatch
            and owner.__dict__.get("imrphenomxphm_native_supported")
            is self._supported
            and owner.__dict__.get("_imrphenomxphm_fd_torch") is self._target
        )

    def enabled(self):
        value = os.environ.get(self.gate_env)
        return (
            False
            if value is None
            else self._owner._parse_switch(self.gate_env, value)
        )

    def budget(self):
        value = os.environ.get(self.budget_env)
        if value is None:
            return PUBLIC_RESULT_CACHE_DEFAULT_BYTES
        if not value.isascii() or not value.isdecimal():
            return None
        try:
            budget = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return budget if budget > 0 else None

    def debug_fingerprint_enabled(self):
        value = os.environ.get(self.debug_fingerprint_env)
        return (
            False
            if value is None
            else self._owner._parse_switch(self.debug_fingerprint_env, value)
        )

    def _reset_if_forked_locked(self):
        process_id = os.getpid()
        if process_id == self._pid:
            return
        self._entries = OrderedDict()
        self._size = 0
        self._counters = _new_counters()
        self._metadata = OrderedDict()
        self._hot_metadata = None
        self._pid = process_id

    def after_fork(self):
        """Drop inherited tensors and replace the inherited lock."""

        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._size = 0
        self._counters = _new_counters()
        self._metadata = OrderedDict()
        self._hot_metadata = None
        self._active = ContextVar(
            "imrphenomxphm_public_result_cache_active",
            default=False,
        )

    def clear(self):
        with self._lock:
            self._reset_if_forked_locked()
            self._entries.clear()
            self._size = 0
            self._counters = _new_counters()
            self._metadata.clear()
            self._hot_metadata = None

    def stats(self):
        with self._lock:
            self._reset_if_forked_locked()
            result = dict(self._counters)
            result.update(
                pid=self._pid,
                entries=len(self._entries),
                bytes=self._size,
                budget_bytes=self.budget() or 0,
                max_entries=PUBLIC_RESULT_CACHE_MAX_ENTRIES,
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
    def _freeze_leaf(value):
        value_type = type(value)
        if value_type is type(None):
            return ("none",)
        if value_type is bool:
            return "bool", value
        if value_type is int:
            return "int", value
        if value_type is float:
            return "float64", struct.pack("!d", value)
        if value_type is str:
            return "str", value
        if value_type is tuple:
            return "tuple", tuple(
                _PublicResultCache._freeze_leaf(item) for item in value
            )
        raise TypeError("mutable or non-plain public-result key value")

    @classmethod
    def _freeze_params(cls, params):
        if type(params) is not dict:
            raise TypeError("public-result request must be an exact dict")
        if not all(type(key) is str for key in params):
            raise TypeError("public-result request keys must be exact strings")
        return tuple(
            (key, cls._freeze_leaf(params[key])) for key in sorted(params)
        )

    def _environment_items(self):
        """Snapshot relevant environment state without decoding all entries.

        CPython's POSIX ``_Environ`` stores the authoritative bytes mapping in
        ``_data``.  Filtering that mapping first avoids decoding every process
        variable twice per cache hit.  Unsupported implementations retain the
        public owner's ordinary snapshot routine.
        """

        environment = os.environ
        data = getattr(environment, "_data", None)
        if type(data) is dict:
            sample = next(iter(data), None)
            if sample is None or type(sample) is str:
                try:
                    return tuple(
                        sorted(
                            (name, value)
                            for name, value in data.items()
                            if type(name) is str
                            and type(value) is str
                            and name.startswith(
                                ("PYCBC_IMRPHENOMX", "PYCBC_TORCH_")
                            )
                        )
                    )
                except Exception:
                    pass
            elif type(sample) is bytes:
                try:
                    raw = tuple(
                        sorted(
                            (name, value)
                            for name, value in data.items()
                            if type(name) is bytes
                            and type(value) is bytes
                            and name.startswith(
                                (b"PYCBC_IMRPHENOMX", b"PYCBC_TORCH_")
                            )
                        )
                    )
                    return tuple(
                        (os.fsdecode(name), os.fsdecode(value))
                        for name, value in raw
                    )
                except Exception:
                    pass
        return self._owner._phenomx_torch_environment_items()

    def _implementation_identity(self):
        owner = self._owner
        owner_namespace = owner.__dict__
        owner_roots = tuple(
            owner_namespace.get(name)
            for name in _OWNER_IMPLEMENTATION_ROOT_NAMES
        )
        owner_tables = tuple(
            owner_namespace.get(name)
            for name in _OWNER_IMPLEMENTATION_TABLE_NAMES
        )
        owner_utils = owner_namespace.get("IMRPhenomX_utils")
        owner_utility_roots = (
            owner_utils,
            getattr(
                owner_utils,
                "_get_phenomx_phase_coeff_table_cached_master",
                None,
            ),
            getattr(
                owner_utils,
                "_get_phenomx_amp_coeff_table_cached_master",
                None,
            ),
        )
        external_roots = (
            FrequencySeries.__init__,
            FrequencySeries.copy,
            TorchArrayData.__init__,
            TorchArrayData.copy,
            _lal.LIGOTimeGPS,
        )
        cache_namespace = vars(type(self))
        cache_roots = tuple(
            cache_namespace.get(name)
            for name in _CACHE_IMPLEMENTATION_ROOT_NAMES
        )
        owner_identity_roots = (
            *owner_roots,
            *owner_utility_roots,
            *external_roots,
        )
        cache_identity_roots = tuple(
            getattr(root, "__func__", root) for root in cache_roots
        )
        identity_roots = (
            *owner_identity_roots,
            *cache_identity_roots,
        )
        identities, references = _identities_and_references(identity_roots)
        owner_identity_count = len(owner_identity_roots)
        owner_identities = identities[:owner_identity_count]
        cache_identities = identities[owner_identity_count:]
        identity_references = _IdentityReferences(references + owner_tables)
        ignored_environment = {
            self.gate_env,
            self.budget_env,
            self.debug_fingerprint_env,
        }
        environment_items = self._environment_items()
        environment = tuple(
            (name, value)
            for name, value in environment_items
            if name not in ignored_environment
        )
        return (
            PUBLIC_RESULT_CACHE_IMPLEMENTATION,
            sys.implementation.name,
            tuple(sys.version_info[:3]),
            torch.__version__,
            getattr(torch.version, "git_version", None),
            torch.get_default_dtype(),
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
            bool(torch.are_deterministic_algorithms_enabled()),
            torch.get_float32_matmul_precision(),
            owner_identities,
            tuple(id(root) for root in owner_tables),
            cache_identities,
            owner._coprecessing_plan_implementation_identity(
                environment_items
            ),
            owner._aggregate_preterminal_twist_implementation_identity(),
            environment,
            identity_references,
        )

    def _request_supported(self, params, generator):
        owner = self._owner
        state = _scheme.mgr.state
        n_batch = params.get("n_batch")
        if (
            generator is not self._dispatch
            or not self._bindings_current()
            or type(state) is not _scheme.TorchScheme
            or type(state.torch_device) is not torch.device
            or state.torch_device != torch.device("cpu")
            or not torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
            or sys.gettrace() is not None
            or sys.getprofile() is not None
            or not owner._plain_request_supported(params)
            or not owner._manual_exact_coverage_supported(params)
            or not owner._trusted_plain_request_enabled()
            or not owner._inference_mode_enabled()
            or not owner._inference_exact_coverage_supported(
                params,
                manual_coverage=True,
            )
            or not owner._coprecessing_plan_coefficient_tables_supported()
            or not (n_batch is None or (type(n_batch) is int and n_batch == 1))
        ):
            return False
        compile_value = os.environ.get("PYCBC_TORCH_COMPILE")
        if compile_value is not None and owner._parse_switch(
            "PYCBC_TORCH_COMPILE",
            compile_value,
        ):
            return False
        return self._supported(params) is True

    def _prepare(self, params, generator, *, note):
        try:
            if not self.enabled():
                return None
            budget = self.budget()
            debug_fingerprint = self.debug_fingerprint_enabled()
            if budget is None or not self._request_supported(params, generator):
                if note:
                    self._note("ineligible")
                return None
            expected_metadata = self._expected_metadata(params)
            key = (
                self._implementation_identity(),
                self._freeze_params(params),
                os.getpid(),
                threading.get_ident(),
            )
            return _Request(
                key,
                budget,
                os.getpid(),
                threading.get_ident(),
                debug_fingerprint,
                expected_metadata,
            )
        except Exception:
            if note:
                self._note("ineligible")
            return None

    def _expected_metadata(self, params):
        delta_f = float(params["delta_f"])
        if not math.isfinite(delta_f) or delta_f <= 0.0:
            raise ValueError("invalid public-result delta_f")
        delta_f_bytes = struct.pack("!d", delta_f)
        if _lal.LAL_AVAILABLE:
            constructor = _lal.LIGOTimeGPS
            constructor_new = getattr(constructor, "__new__", None)
            key = (
                delta_f_bytes,
                constructor,
                constructor_new,
                getattr(constructor_new, "__code__", None),
            )
            hot = self._hot_metadata
            if hot is not None and hot[0] == key:
                return hot[1]
            with self._lock:
                self._reset_if_forked_locked()
                metadata = self._metadata.get(key)
                if metadata is not None:
                    self._metadata.move_to_end(key)
                    self._hot_metadata = key, metadata
                    return metadata
            epoch = _lal.LIGOTimeGPS(-1.0 / delta_f)
            sec = int(epoch.gpsSeconds)
            nsec = int(epoch.gpsNanoSeconds)
            epoch_type = type(epoch)
        else:
            constructor = _np.float64
            constructor_new = getattr(constructor, "__new__", None)
            key = (
                delta_f_bytes,
                constructor,
                constructor_new,
                getattr(constructor_new, "__code__", None),
            )
            hot = self._hot_metadata
            if hot is not None and hot[0] == key:
                return hot[1]
            with self._lock:
                self._reset_if_forked_locked()
                metadata = self._metadata.get(key)
                if metadata is not None:
                    self._metadata.move_to_end(key)
                    self._hot_metadata = key, metadata
                    return metadata
            epoch = _np.float64(-1.0 / delta_f)
            sec = int(epoch)
            nsec = int(round((float(epoch) - sec) * 1e9))
            epoch_type = type(epoch)

        metadata = _SeriesMetadata(
            FrequencySeries,
            TorchArrayData,
            delta_f_bytes,
            epoch_type,
            sec,
            nsec,
        )
        result = metadata, metadata
        with self._lock:
            self._reset_if_forked_locked()
            self._metadata[key] = result
            self._metadata.move_to_end(key)
            while len(self._metadata) > PUBLIC_RESULT_CACHE_METADATA_ENTRIES:
                self._metadata.popitem(last=False)
            self._hot_metadata = key, result
        return result

    @staticmethod
    def _tensor_descriptor(value):
        storage = value.untyped_storage()
        return _TensorDescriptor(
            id(value),
            int(storage._cdata),
            int(storage.data_ptr()),
            int(storage.nbytes()),
            bool(storage.resizable()),
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

    @staticmethod
    def _fingerprint(value):
        # NumPy exposure makes the backing Torch storage non-resizable.  Hash a
        # temporary clone so admission and debug checks cannot mutate either a
        # public cold result or the privately owned cache tensor.
        raw = (
            value.detach()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .clone()
            .numpy()
            .tobytes()
        )
        return value.dtype, tuple(value.shape), hashlib.sha256(raw).digest()

    @staticmethod
    def _owned_tensor(value):
        with torch.inference_mode(False), torch.no_grad():
            return value.detach().clone(memory_format=torch.contiguous_format)

    def _tensor_schema_supported(self, value, *, privately_owned):
        try:
            storage = value.untyped_storage()
            return (
                type(value) is torch.Tensor
                and value.layout is torch.strided
                and value.device == torch.device("cpu")
                and value.dtype is torch.complex128
                and value.ndim == 1
                and value.numel() > 0
                and value.is_contiguous()
                and not value.is_conj()
                and not value.is_neg()
                and not value.requires_grad
                and value.grad_fn is None
                and not value.is_inference()
                and not self._owner.IMRPhenomX_utils._tensor_has_forward_ad(
                    value
                )
                and storage.data_ptr() != 0
                and (
                    not privately_owned
                    or (
                        value._version == 0
                        and value.storage_offset() == 0
                        and value._base is None
                        and storage.nbytes()
                        == value.numel() * value.element_size()
                    )
                )
            )
        except Exception:
            return False

    @staticmethod
    def _epoch_type_supported(epoch_type):
        ltg = getattr(_lal, "LIGOTimeGPS", None)
        return (ltg is not None and epoch_type is ltg) or epoch_type in (float, int) or issubclass(epoch_type, (float, int))

    @classmethod
    def _series_metadata(cls, series):
        epoch = series.epoch
        sec = int(getattr(epoch, "gpsSeconds", int(epoch)))
        nsec = int(getattr(epoch, "gpsNanoSeconds", int(round((float(epoch) - int(epoch)) * 1e9))))
        return _SeriesMetadata(
            type(series),
            type(series._data),
            struct.pack("!d", series.delta_f),
            type(epoch),
            sec,
            nsec,
        )

    def _result_supported(self, result, *, privately_owned):
        try:
            if type(result) is not tuple or len(result) != 2:
                return False
            plus, cross = result
            if type(plus) is not FrequencySeries or type(cross) is not FrequencySeries:
                return False
            if type(plus._data) is not TorchArrayData or type(cross._data) is not TorchArrayData:
                return False
            if not (self._epoch_type_supported(type(plus.epoch)) and self._epoch_type_supported(type(cross.epoch))):
                return False
            if not (
                type(plus.delta_f) is float
                and type(cross.delta_f) is float
                and struct.pack("!d", plus.delta_f)
                == struct.pack("!d", cross.delta_f)
            ):
                return False
            tensors = plus._data.tensor, cross._data.tensor
            if tuple(tensors[0].shape) != tuple(tensors[1].shape):
                return False
            if not all(
                self._tensor_schema_supported(
                    value,
                    privately_owned=privately_owned,
                )
                for value in tensors
            ):
                return False
            pointers = tuple(
                value.untyped_storage().data_ptr() for value in tensors
            )
            return pointers[0] != pointers[1]
        except Exception:
            return False

    @classmethod
    def _metadata_supported(cls, metadata):
        if type(metadata) is not tuple or len(metadata) != 2:
            return False
        try:
            deltas = tuple(
                struct.unpack("!d", value.delta_f_bytes)[0]
                for value in metadata
                if type(value) is _SeriesMetadata
                and value.series_type is FrequencySeries
                and value.data_type is TorchArrayData
                and type(value.delta_f_bytes) is bytes
                and len(value.delta_f_bytes) == 8
                and cls._epoch_type_supported(value.epoch_type)
                and type(value.epoch_seconds) is int
                and type(value.epoch_nanoseconds) is int
            )
            return (
                len(deltas) == 2
                and all(math.isfinite(value) and value > 0.0 for value in deltas)
                and metadata[0].delta_f_bytes == metadata[1].delta_f_bytes
            )
        except Exception:
            return False

    @staticmethod
    def _descriptor_admission_supported(descriptors):
        """Recheck the immutable schema proof recorded at admission."""

        if type(descriptors) is not tuple or len(descriptors) != 2:
            return False
        plus, cross = descriptors
        try:
            valid = all(
                type(value) is _TensorDescriptor
                and type(value.identity) is int
                and type(value.storage_identity) is int
                and type(value.storage_pointer) is int
                and value.storage_pointer != 0
                and type(value.storage_bytes) is int
                and type(value.storage_resizable) is bool
                and value.storage_resizable is True
                and type(value.data_pointer) is int
                and value.data_pointer == value.storage_pointer
                and type(value.version) is int
                and value.version == 0
                and value.dtype is torch.complex128
                and value.device == torch.device("cpu")
                and type(value.shape) is tuple
                and len(value.shape) == 1
                and type(value.shape[0]) is int
                and value.shape[0] > 0
                and value.stride == (1,)
                and value.storage_bytes == value.shape[0] * 16
                and value.storage_offset == 0
                and value.base_identity is None
                and value.requires_grad is False
                and value.grad_fn_identity is None
                and value.inference is False
                and value.conjugated is False
                and value.negated is False
                for value in descriptors
            )
            return (
                valid
                and plus.identity != cross.identity
                and plus.storage_identity != cross.storage_identity
                and plus.storage_pointer != cross.storage_pointer
                and plus.shape == cross.shape
            )
        except Exception:
            return False

    def _entry_valid(self, request, entry, *, debug_fingerprint):
        if (
            type(request) is not _Request
            or type(entry) is not _Entry
            or entry.request_key != request.key
            or entry.metadata != request.expected_metadata
            or not self._metadata_supported(entry.metadata)
        ):
            return False
        tensors = entry.plus, entry.cross
        try:
            # Metadata, schema, and ownership were proven before the private
            # immutable entry was admitted.  Recheck that proof and the full
            # live tensor descriptors without constructing throwaway public
            # wrappers or GPS objects on every hit.
            valid = (
                self._descriptor_admission_supported(entry.descriptors)
                and tuple(
                    self._tensor_descriptor(value) for value in tensors
                )
                == entry.descriptors
            )
        except Exception:
            valid = False
        if valid and debug_fingerprint:
            self._counters["debug_fingerprint_checks"] += 1
            try:
                valid = (
                    tuple(self._fingerprint(value) for value in tensors)
                    == entry.fingerprints
                )
            except Exception:
                valid = False
            if not valid:
                self._counters["debug_fingerprint_failures"] += 1
        return valid

    @staticmethod
    def _reconstructed_tensor(value):
        result = torch.empty_like(
            value,
            memory_format=torch.contiguous_format,
        )
        result.copy_(value)
        return result

    def _reconstruct(self, entry):
        with torch.inference_mode(False), torch.no_grad():
            tensors = tuple(
                self._reconstructed_tensor(value)
                for value in (entry.plus, entry.cross)
            )
        series_list = []
        for value, metadata in zip(tensors, entry.metadata):
            if _lal.LAL_AVAILABLE:
                epoch = _lal.LIGOTimeGPS(
                    metadata.epoch_seconds,
                    metadata.epoch_nanoseconds,
                )
            else:
                epoch = float(metadata.epoch_seconds) + float(metadata.epoch_nanoseconds) * 1e-9
            series_list.append(
                FrequencySeries(
                    TorchArrayData(value),
                    delta_f=struct.unpack("!d", metadata.delta_f_bytes)[0],
                    epoch=epoch,
                    copy=False,
                )
            )
        return tuple(series_list)

    @staticmethod
    def _deep_size(value, seen=None):
        if seen is None:
            seen = set()
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(value)
        if type(value) in (tuple, list):
            size += sum(_PublicResultCache._deep_size(item, seen) for item in value)
        elif type(value) is dict:
            size += sum(
                _PublicResultCache._deep_size(key, seen)
                + _PublicResultCache._deep_size(item, seen)
                for key, item in value.items()
            )
        return size

    def _entry_bytes(self, request, entry):
        storage_bytes = sum(
            value.untyped_storage().nbytes()
            for value in (entry.plus, entry.cross)
        )
        return (
            self._deep_size(request.key)
            + self._deep_size(entry.descriptors)
            + self._deep_size(entry.fingerprints)
            + self._deep_size(entry.metadata)
            + storage_bytes
            + PUBLIC_RESULT_CACHE_ENTRY_OVERHEAD
        )

    def _trim_locked(self, budget):
        while self._entries and (
            self._size > budget
            or len(self._entries) > PUBLIC_RESULT_CACHE_MAX_ENTRIES
        ):
            _, entry = self._entries.popitem(last=False)
            self._size -= entry.nbytes
            self._counters["evictions"] += 1

    def _lookup(self, request, revalidate):
        with self._lock:
            self._reset_if_forked_locked()
            self._trim_locked(request.budget)
            entry = self._entries.get(request.key)
            if entry is None:
                self._counters["misses"] += 1
                return None
            if not self._entry_valid(
                request,
                entry,
                debug_fingerprint=request.debug_fingerprint,
            ):
                self._entries.pop(request.key, None)
                self._size -= entry.nbytes
                self._counters["invalidations"] += 1
                self._counters["misses"] += 1
                return None
            self._entries.move_to_end(request.key)

        try:
            result = self._reconstruct(entry)
            current = revalidate()
        except Exception:
            current = None
            result = None
        if type(current) is not _Request or current != request:
            self._note("races")
            return None
        with self._lock:
            self._reset_if_forked_locked()
            current_entry = self._entries.get(request.key)
            if current_entry is not entry:
                self._counters["races"] += 1
                return None
            self._counters["hits"] += 1
        return result

    def _store(self, request, result, revalidate):
        if not self._result_supported(result, privately_owned=False):
            self._note("canary_failures")
            return
        try:
            source_tensors = tuple(series._data.tensor for series in result)
            tensors = tuple(self._owned_tensor(value) for value in source_tensors)
            metadata = tuple(self._series_metadata(series) for series in result)
            if metadata != request.expected_metadata:
                self._note("canary_failures")
                return
            if _lal.LAL_AVAILABLE:
                candidate_epoch = lambda item: _lal.LIGOTimeGPS(
                    item.epoch_seconds,
                    item.epoch_nanoseconds,
                )
            else:
                candidate_epoch = lambda item: float(item.epoch_seconds) + float(item.epoch_nanoseconds) * 1e-9
            candidate = tuple(
                FrequencySeries(
                    TorchArrayData(value),
                    delta_f=struct.unpack("!d", item.delta_f_bytes)[0],
                    epoch=candidate_epoch(item),
                    copy=False,
                )
                for value, item in zip(tensors, metadata)
            )
            if not self._result_supported(candidate, privately_owned=True):
                self._note("canary_failures")
                return
            source_fingerprints = tuple(
                self._fingerprint(value) for value in source_tensors
            )
            fingerprints = tuple(self._fingerprint(value) for value in tensors)
            if source_fingerprints != fingerprints:
                self._note("canary_failures")
                return
            descriptors = tuple(self._tensor_descriptor(value) for value in tensors)
            provisional = _Entry(
                request.key,
                tensors[0],
                tensors[1],
                descriptors,
                fingerprints,
                metadata,
                0,
            )
            entry = provisional._replace(
                nbytes=self._entry_bytes(request, provisional)
            )
            current = revalidate()
        except Exception:
            self._note("canary_failures")
            return
        if (
            type(current) is not _Request
            or current != request
        ):
            self._note("races")
            return

        with self._lock:
            self._reset_if_forked_locked()
            if entry.nbytes > request.budget:
                self._counters["oversize"] += 1
                return
            existing = self._entries.pop(request.key, None)
            if existing is not None:
                self._size -= existing.nbytes
                self._counters["races"] += 1
            self._entries[request.key] = entry
            self._size += entry.nbytes
            self._counters["stores"] += 1
            self._trim_locked(request.budget)

    def run(self, params, generator, invoke):
        """Run one normalized public request through the exact result LRU."""

        if self._active.get() is True:
            self._note("reentrant")
            return invoke()
        token = self._active.set(True)
        try:
            request = self._prepare(params, generator, note=True)
            if request is None:
                return invoke()

            def revalidate():
                return self._prepare(params, generator, note=False)

            cached = self._lookup(request, revalidate)
            if cached is not None:
                return cached
            result = invoke()
            self._store(request, result, revalidate)
            return result
        finally:
            self._active.reset(token)


def make_public_result_cache(owner, dispatch, supported, target):
    """Construct, fork-register, and hide one public-result cache instance."""

    cache = _PublicResultCache(owner, dispatch, supported, target)
    register_at_fork = getattr(os, "register_at_fork", None)
    if register_at_fork is not None:
        register_at_fork(after_in_child=cache.after_fork)

    def bind_public_entry(public_entry):
        return cache.bind_public_entry(public_entry)

    def run(params, generator, invoke):
        return cache.run(params, generator, invoke)

    def clear():
        return cache.clear()

    def stats():
        return cache.stats()

    return bind_public_entry, run, clear, stats
