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

"""Torch IMRPhenomX coefficient fits and remnant-frequency utilities.

Adapted from ripple v0.2.1's independently validated IMRPhenomX utilities
(https://github.com/GW-JAX-Team/ripple/tree/v0.2.1).  The numerical coefficient
tables and equations match that tagged source.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import math
import os
from typing import Any, NamedTuple

from pycbc import lal_compat as lal
import torch

from ._torch_jax import _ACTIVE_TENSOR, jnp
from .imrphenomt_fits_torch import qnm_fdamp_22, qnm_fring_22
from .torch_switches import _parse_switch

Array = Any
Float = Any
FloatLike = Any

MTSUN = lal.MTSUN_SI
_DERIVED_POWER_REUSE_ENV = "PYCBC_IMRPHENOMX_DERIVED_POWER_REUSE"
_REMNANT_FINAL_SPIN_REUSE_ENV = (
    "PYCBC_IMRPHENOMX_REMNANT_FINAL_SPIN_REUSE"
)
_REMNANT_ALIGNED_BASE_REUSE_ENV = (
    "PYCBC_IMRPHENOMX_REMNANT_ALIGNED_BASE_REUSE"
)
_REMNANT_PYTHON_SCALARS_ENV = "PYCBC_IMRPHENOMX_REMNANT_PYTHON_SCALARS"
_FOREACH_CBRT_ENV = "PYCBC_IMRPHENOMX_FOREACH_CBRT"


def _derived_power_reuse_enabled():
    """Return the strict switch for exact scalar-power sharing."""

    value = os.environ.get(_DERIVED_POWER_REUSE_ENV)
    return (
        True
        if value is None
        else _parse_switch(_DERIVED_POWER_REUSE_ENV, value)
    )


def _remnant_final_spin_reuse_enabled():
    """Return the strict switch for exact aligned-final-spin sharing."""

    value = os.environ.get(_REMNANT_FINAL_SPIN_REUSE_ENV)
    return (
        True
        if value is None
        else _parse_switch(_REMNANT_FINAL_SPIN_REUSE_ENV, value)
    )


def _remnant_aligned_base_reuse_enabled():
    """Return the strict switch for request-local aligned-fit sharing."""

    value = os.environ.get(_REMNANT_ALIGNED_BASE_REUSE_ENV)
    return (
        True
        if value is None
        else _parse_switch(_REMNANT_ALIGNED_BASE_REUSE_ENV, value)
    )


def _remnant_python_scalars_enabled():
    """Return the strict switch for exact CPU scalar remnant fits."""

    value = os.environ.get(_REMNANT_PYTHON_SCALARS_ENV)
    return (
        True
        if value is None
        else _parse_switch(_REMNANT_PYTHON_SCALARS_ENV, value)
    )


def _foreach_cbrt_enabled():
    """Return the strict switch for the exact remnant cube-root lane."""

    value = os.environ.get(_FOREACH_CBRT_ENV)
    return (
        True
        if value is None
        else _parse_switch(_FOREACH_CBRT_ENV, value)
    )


def _remnant_foreach_cbrt_supported(value):
    """Fail closed before changing evaluation order for unsupported inputs."""

    return (
        _foreach_cbrt_enabled()
        and type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.ndim == 0
        and value.dtype in (torch.float32, torch.float64)
        and value.device.type in ("cpu", "cuda")
        and not value.is_conj()
        and not value.is_neg()
    )


def qnm_fring_21(final_spin: FloatLike) -> FloatLike:
    """Return the dimensionless :math:`(2,1)` QNM ringdown frequency."""

    numerator = (
        0.059471695665734674
        - 0.07585416297991414 * final_spin
        + 0.021967909664591865 * final_spin**2
        - 0.0018964744613388146 * final_spin**3
        + 0.001164879406179587 * final_spin**4
        - 0.0003387374454044957 * final_spin**5
    )
    denominator = (
        1.0
        - 1.4437415542456158 * final_spin
        + 0.49246920313191234 * final_spin**2
    )
    return numerator / denominator


def qnm_fdamp_21(final_spin: FloatLike) -> FloatLike:
    """Return the dimensionless :math:`(2,1)` QNM damping frequency."""

    numerator = (
        2.0696914454467294
        - 3.1358071947583093 * final_spin
        + 0.14456081596393977 * final_spin**2
        + 1.2194717985037946 * final_spin**3
        - 0.2947372598589144 * final_spin**4
        + 0.002943057145913646 * final_spin**5
    )
    denominator = (
        146.1779212636481
        - 219.81790388304876 * final_spin
        + 17.7141194900164 * final_spin**2
        + 75.90115083917898 * final_spin**3
        - 18.975287709794745 * final_spin**4
    )
    return numerator / denominator
PI = lal.PI


# Dimensionless cutoff frequency for PhenomXAS
fM_CUT = 0.3


class IMRPhenomXRemnant(NamedTuple):
    """Dimensionless remnant quantities shared by the PhenomX models."""

    final_spin: FloatLike
    radiated_energy: FloatLike
    ringdown_frequency: FloatLike
    damping_frequency: FloatLike
    meco_frequency: FloatLike
    isco_frequency: FloatLike


class _PackedRemnantPlan(NamedTuple):
    """Two exact remnant variants owned by one XPHM request."""

    aligned: IMRPhenomXRemnant
    carrier: IMRPhenomXRemnant


class _AlignedRemnantBase(NamedTuple):
    """Precession-independent results retained within one waveform call."""

    eta: FloatLike
    aligned_spin: FloatLike
    radiated_energy: FloatLike
    meco_frequency: FloatLike
    isco_frequency: FloatLike


def _tensor_has_forward_ad(value):
    """Return whether ``value`` carries a forward-mode AD tangent."""

    # Avoid unpacking every tensor in the normal waveform path. A forward dual
    # cannot exist outside an active dual level.
    current_level = getattr(torch.autograd.forward_ad, "_current_level", None)
    if current_level == -1:
        return False
    try:
        return torch.autograd.forward_ad.unpack_dual(value).tangent is not None
    except (AttributeError, RuntimeError):
        # If the installed Torch cannot safely expose the tangent, fail closed
        # and leave evaluation to its autograd-aware path.
        return True


_TRUSTED_PLAIN_REQUEST = ContextVar(
    "pycbc_imrphenomx_trusted_plain_request",
    default=False,
)


@contextmanager
def trusted_plain_request_context(*, enabled=False):
    """Skip repeated AD tree walks inside one prevalidated request.

    Callers must validate the complete public input tree and active Torch
    runtime before enabling this context.  A :class:`ContextVar` keeps the
    promise local to the current synchronous/thread context and the token is
    always restored, including for nested contexts and exceptions.
    """

    token = _TRUSTED_PLAIN_REQUEST.set(bool(enabled))
    try:
        yield
    finally:
        _TRUSTED_PLAIN_REQUEST.reset(token)


def _tree_has_autograd_untrusted(value):
    """Scan ``value`` for AD without consulting the request-local promise."""

    if isinstance(value, torch.Tensor):
        return (
            value.requires_grad
            or value.grad_fn is not None
            or _tensor_has_forward_ad(value)
        )
    if isinstance(value, (tuple, list)):
        return any(_tree_has_autograd_untrusted(item) for item in value)
    if isinstance(value, dict):
        return any(
            _tree_has_autograd_untrusted(item) for item in value.values()
        )
    return False


def _tree_has_autograd(value):
    """Return whether a result contains a reverse- or forward-AD tensor."""

    if _TRUSTED_PLAIN_REQUEST.get():
        return False
    return _tree_has_autograd_untrusted(value)


def _remnant_argument_key(value):
    """Return a synchronization-free remnant-cache key, or ``None``."""

    if isinstance(value, torch.Tensor):
        # Tensor subclasses may override operators used by the remnant fits.
        # Their Python type is not otherwise represented in the storage key,
        # so keep them on the normal eager path.
        if type(value) is not torch.Tensor:
            return None
        if (
            value.requires_grad
            or value.grad_fn is not None
            or _tensor_has_forward_ad(value)
            or value.layout is not torch.strided
            or value.numel() == 0
        ):
            return None
        if (
            value.device.type == "cpu"
            and value.dtype == torch.float64
            and value.ndim == 0
            and not value.is_conj()
            and not value.is_neg()
        ):
            scalar = value.item()
            if math.isfinite(scalar):
                # Match an equivalent Python float while preserving signed
                # zero. Value extraction is deliberately CPU-only so cache
                # lookup can never synchronize an accelerator.
                return ("float", scalar.hex())
        try:
            version = value._version
            storage_pointer = value.untyped_storage().data_ptr()
        except (NotImplementedError, RuntimeError):
            return None
        if storage_pointer == 0:
            return None
        return (
            "tensor",
            value.device,
            value.dtype,
            value.layout,
            value.is_conj(),
            value.is_neg(),
            storage_pointer,
            value.storage_offset(),
            tuple(value.shape),
            tuple(value.stride()),
            version,
        )
    # Python numeric equality collapses signed zero. Preserve its bit-relevant
    # representation because the remnant fits contain copysign operations.
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, complex):
        return ("complex", value.real.hex(), value.imag.hex())
    try:
        hash(value)
    except TypeError:
        return None
    return ("value", type(value), value)


def _aligned_remnant_base_inputs_supported(values):
    """Fail closed outside exact scalar CPU/CUDA remnant inputs."""

    for value in values:
        if not isinstance(value, torch.Tensor):
            if not isinstance(value, (int, float)):
                return False
            continue
        if (
            type(value) is not torch.Tensor
            or value.layout is not torch.strided
            or value.ndim != 0
            or value.dtype not in (torch.float32, torch.float64)
            or value.device.type not in ("cpu", "cuda")
            or value.is_conj()
            or value.is_neg()
            or _tree_has_autograd(value)
        ):
            return False
    return True


class _PerWaveformRemnantCache:
    """Cache remnant fits for one top-level native waveform invocation."""

    def __init__(self):
        self._entries = {}
        self._aligned_entries = {}
        self.calls = 0
        self.hits = 0
        self.underlying_evaluations = 0
        self.bypasses = 0
        self.peak_entries = 0
        self.aligned_base_evaluations = 0
        self.aligned_base_hits = 0
        self.aligned_base_bypasses = 0

    @staticmethod
    def _key(args, kwargs):
        positional = tuple(_remnant_argument_key(value) for value in args)
        keyword = tuple(
            (name, _remnant_argument_key(value))
            for name, value in sorted(kwargs.items())
        )
        if any(item is None for item in positional):
            return None
        if any(item is None for _, item in keyword):
            return None
        return positional, keyword

    def evaluate(self, function, args, kwargs):
        self.calls += 1
        key = self._key(args, kwargs)
        if key is None:
            self.bypasses += 1
            self.underlying_evaluations += 1
            return function(*args, **kwargs)
        if key in self._entries:
            self.hits += 1
            return self._entries[key][1]

        self.underlying_evaluations += 1
        result = function(*args, **kwargs)
        if _tree_has_autograd(result):
            self.bypasses += 1
            return result

        # Retaining tensor arguments prevents allocator pointer reuse while
        # this one-waveform cache is active. Input mutation changes _version
        # and therefore cannot hit the old entry.
        guards = tuple(
            value
            for value in (*args, *kwargs.values())
            if isinstance(value, torch.Tensor)
        )
        self._entries[key] = (guards, result)
        self.peak_entries = max(self.peak_entries, len(self._entries))
        return result

    def evaluate_with_aligned_base(self, function, args, kwargs):
        """Reuse exact precession-independent fits within this request."""

        self.calls += 1
        key = self._key(args, kwargs)
        if key is None:
            self.bypasses += 1
            self.aligned_base_bypasses += 1
            self.underlying_evaluations += 1
            return function(*args, **kwargs)
        if key in self._entries:
            self.hits += 1
            return self._entries[key][1]

        normalized = _normalize_remnant_inputs(
            *args,
            kwargs.get("final_spin"),
        )
        aligned_inputs = normalized[:4]
        aligned_key = (
            self._key(aligned_inputs, {})
            if _aligned_remnant_base_inputs_supported(aligned_inputs)
            else None
        )
        if aligned_key is None:
            self.aligned_base_bypasses += 1
            self.underlying_evaluations += 1
            result = function(*args, **kwargs)
            if not _tree_has_autograd(result):
                guards = tuple(
                    value
                    for value in (*args, *kwargs.values())
                    if isinstance(value, torch.Tensor)
                )
                self._entries[key] = (guards, result)
                self.peak_entries = max(
                    self.peak_entries,
                    len(self._entries),
                )
            else:
                self.bypasses += 1
            return result

        capture = []
        aligned_entry = self._aligned_entries.get(aligned_key)
        self.underlying_evaluations += 1
        if aligned_entry is None:
            self.aligned_base_evaluations += 1
            result = function(
                *args,
                **kwargs,
                _aligned_base_capture=capture,
            )
        else:
            self.aligned_base_hits += 1
            result = function(
                *args,
                **kwargs,
                _aligned_base=aligned_entry[1],
            )

        if _tree_has_autograd(result):
            self.bypasses += 1
            return result

        guards = tuple(
            value
            for value in (*args, *kwargs.values())
            if isinstance(value, torch.Tensor)
        )
        self._entries[key] = (guards, result)
        self.peak_entries = max(self.peak_entries, len(self._entries))
        if capture and not _tree_has_autograd(capture[0]):
            aligned_guards = tuple(
                value
                for value in aligned_inputs
                if isinstance(value, torch.Tensor)
            )
            self._aligned_entries[aligned_key] = (
                aligned_guards,
                capture[0],
            )
        return result

    def clear(self):
        """Release all input guards and cached results."""

        self._entries.clear()
        self._aligned_entries.clear()

    def report(self):
        """Return counters used by focused regression tests and profiling."""

        return {
            "calls": self.calls,
            "hits": self.hits,
            "underlying_evaluations": self.underlying_evaluations,
            "bypasses": self.bypasses,
            "peak_entries": self.peak_entries,
            "active_entries": len(self._entries),
            "aligned_base_evaluations": self.aligned_base_evaluations,
            "aligned_base_hits": self.aligned_base_hits,
            "aligned_base_bypasses": self.aligned_base_bypasses,
            "active_aligned_base_entries": len(self._aligned_entries),
        }


_ACTIVE_REMNANT_CACHE = ContextVar(
    "pycbc_imrphenomx_per_waveform_remnant_cache",
    default=None,
)


@contextmanager
def remnant_cache_context(*, enabled=True):
    """Bound remnant caching to one synchronous waveform invocation.

    A new cache is installed even when another invocation is nested in the
    current context. Disabling the context temporarily suppresses any outer
    cache. Entries and allocator guards are cleared before the previous
    context is restored, including when waveform construction raises.
    """

    cache = _PerWaveformRemnantCache() if enabled else None
    token = _ACTIVE_REMNANT_CACHE.set(cache)
    try:
        yield cache
    finally:
        if cache is not None:
            cache.clear()
        _ACTIVE_REMNANT_CACHE.reset(token)


def final_spin_2017(
    eta: FloatLike,
    chi1: FloatLike,
    chi2: FloatLike,
) -> FloatLike:
    """Return the aligned-spin PhenomX remnant-spin fit."""

    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
    mass1 = 0.5 * (1.0 + delta)
    mass2 = 0.5 * (1.0 - delta)
    mass1_sq = mass1 * mass1
    mass2_sq = mass2 * mass2
    eta2 = eta * eta
    eta3 = eta2 * eta
    total_spin = (mass1_sq * chi1 + mass2_sq * chi2) / (mass1_sq + mass2_sq)
    total_spin2 = total_spin * total_spin
    total_spin3 = total_spin2 * total_spin
    delta_chi = chi1 - chi2

    nonspinning = (
        3.4641016151377544 * eta + 20.0830030082033 * eta2 - 12.333573402277912 * eta3
    ) / (1.0 + 7.2388440419467335 * eta)
    equal_spin = (mass1_sq + mass2_sq) * total_spin + (
        (
            -0.8561951310209386 * eta
            - 0.09939065676370885 * eta2
            + 1.668810429851045 * eta3
        )
        * total_spin
        + (
            0.5881660363307388 * eta
            - 2.149269067519131 * eta2
            + 3.4768263932898678 * eta3
        )
        * total_spin2
        + (
            0.142443244743048 * eta
            - 0.9598353840147513 * eta2
            + 1.9595643107593743 * eta3
        )
        * total_spin3
    ) / (
        1.0
        + (-0.9142232693081653 + 2.3191363426522633 * eta - 9.710576749140989 * eta3)
        * total_spin
    )
    unequal_spin = (
        0.3223660562764661 * delta_chi * delta * (1.0 + 9.332575956437443 * eta) * eta2
        - 0.059808322561702126 * delta_chi * delta_chi * eta3
        + 2.3170397514509933
        * delta_chi
        * delta
        * (1.0 - 3.2624649875884852 * eta)
        * eta3
        * total_spin
    )
    return nonspinning + equal_spin + unequal_spin


def precessing_final_spin_2017(
    eta: FloatLike,
    chi1: FloatLike,
    chi2: FloatLike,
    chi_inplane: FloatLike,
) -> FloatLike:
    """Return LAL's PhenomX precessing remnant-spin extension."""

    aligned_spin = final_spin_2017(eta, chi1, chi2)
    return _precessing_final_spin_from_aligned(
        eta,
        chi_inplane,
        aligned_spin,
    )


def _precessing_final_spin_from_aligned(
    eta: FloatLike,
    chi_inplane: FloatLike,
    aligned_spin: FloatLike,
) -> FloatLike:
    """Extend an already-evaluated aligned spin without changing fit order."""

    larger_mass_fraction = 0.5 * (
        1.0 + jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
    )
    perpendicular_spin = chi_inplane * larger_mass_fraction**2
    return jnp.copysign(
        jnp.sqrt(aligned_spin**2 + perpendicular_spin**2),
        aligned_spin,
    )


def _remnant_final_spin_reuse_supported(aligned_spin):
    """Return whether an exact scalar aligned-spin result may be shared."""

    return (
        _remnant_final_spin_reuse_enabled()
        and type(aligned_spin) is torch.Tensor
        and aligned_spin.layout is torch.strided
        and aligned_spin.ndim == 0
        and aligned_spin.dtype in (torch.float32, torch.float64)
        and aligned_spin.device.type in ("cpu", "cuda")
        and not aligned_spin.is_conj()
        and not aligned_spin.is_neg()
        and not _tree_has_autograd(aligned_spin)
    )


def _normalize_remnant_inputs(m1, m2, chi1, chi2, chip, final_spin):
    """Place mixed scalar/tensor remnant inputs on one Torch device."""

    values = (final_spin, m1, m2, chi1, chi2, chip)
    reference = next(
        (value for value in values if isinstance(value, torch.Tensor)),
        None,
    )
    if reference is None:
        return m1, m2, chi1, chi2, chip, final_spin
    if not reference.dtype.is_floating_point:
        raise TypeError("IMRPhenomX remnant tensor inputs must be real floating point")

    def match(value):
        if value is None:
            return None
        return torch.as_tensor(
            value,
            device=reference.device,
            dtype=reference.dtype,
        )

    return tuple(match(value) for value in (m1, m2, chi1, chi2, chip, final_spin))


def _get_remnant_fMs_uncached_eager(
    m1: FloatLike,
    m2: FloatLike,
    chi1: FloatLike,
    chi2: FloatLike,
    chip: float | FloatLike = 0.0,
    *,
    final_spin: FloatLike | None = None,
    _aligned_base: _AlignedRemnantBase | None = None,
    _aligned_base_capture: list[_AlignedRemnantBase] | None = None,
) -> IMRPhenomXRemnant:
    # This function returns a variety of frequencies needed for computing IMRPhenomXAS
    # In particular, we have fRD, fdamp, fMECO, FISCO
    # chip: effective precession spin parameter. When non-zero, fRD/fdamp are computed
    # from the precessing final spin afinal_prec (matching LAL's pWF->afinal = afinal_prec).
    # final_spin: optional already-computed precessing final spin. This lets XP use
    # its MSA prescription while preserving the aligned spin for fISCO and Erad.
    m1, m2, chi1, chi2, chip, final_spin = _normalize_remnant_inputs(
        m1,
        m2,
        chi1,
        chi2,
        chip,
        final_spin,
    )
    if _aligned_base is not None:
        if final_spin is None:
            a_prec = _precessing_final_spin_from_aligned(
                _aligned_base.eta,
                chip,
                _aligned_base.aligned_spin,
            )
        else:
            a_prec = jnp.asarray(final_spin)
        fRD = qnm_fring_22(a_prec) / (
            1.0 - _aligned_base.radiated_energy
        )
        fdamp = qnm_fdamp_22(a_prec) / (
            1.0 - _aligned_base.radiated_energy
        )
        return IMRPhenomXRemnant(
            a_prec,
            _aligned_base.radiated_energy,
            fRD,
            fdamp,
            _aligned_base.meco_frequency,
            _aligned_base.isco_frequency,
        )
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta_s = m1_s * m2_s / (M_s**2.0)
    # m1Sq = m1_s * m1_s
    # m2Sq = m2_s * m2_s

    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta_s, 0.0))
    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)

    chi_eff = mm1 * chi1 + mm2 * chi2

    eta2 = eta_s * eta_s
    eta3 = eta2 * eta_s
    eta4 = eta3 * eta_s
    S = (chi_eff - (38.0 / 113.0) * eta_s * (chi1 + chi2)) / (
        1.0 - (76.0 * eta_s / 113.0)
    )
    S2 = S * S
    S3 = S2 * S

    dchi = chi1 - chi2
    dchi2 = dchi * dchi

    StotR = (mm1**2.0 * chi1 + mm2**2.0 * chi2) / (mm1**2.0 + mm2**2.0)
    StotR2 = StotR * StotR
    StotR3 = StotR2 * StotR

    # First we need to calculate the dimensionless final spin and radiated energy.
    a = final_spin_2017(eta_s, chi1, chi2)

    Erad = (
        (
            (
                0.057190958417936644 * eta_s
                + 0.5609904135313374 * eta2
                - 0.84667563764404 * eta3
                + 3.145145224278187 * eta4
            )
            * (
                1
                + (
                    -0.13084389181783257
                    - 1.1387311580238488 * eta_s
                    + 5.49074464410971 * eta2
                )
                * StotR
                + (-0.17762802148331427 + 2.176667900182948 * eta2) * StotR2
                + (
                    -0.6320191645391563
                    + 4.952698546796005 * eta_s
                    - 10.023747993978121 * eta2
                )
                * StotR3
            )
        )
        / (
            1
            + (
                -0.9919475346968611
                + 0.367620218664352 * eta_s
                + 4.274567337924067 * eta2
            )
            * StotR
        )
    ) + (
        -0.09803730445895877 * dchi * delta * (1 - 3.2283713377939134 * eta_s) * eta2
        + 0.01118530335431078 * dchi2 * eta3
        - 0.01978238971523653
        * dchi
        * delta
        * (1 - 4.91667749015812 * eta_s)
        * eta_s
        * StotR
    )

    # Taken from https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_t_h_m__fits_8c_source.html

    # Precessing final spin (= a when chip=0): LAL sets pWF->afinal = afinal_prec,
    # so fRD/fdamp use a_prec. fISCO keeps using the aligned-spin a.
    if final_spin is None:
        if _remnant_final_spin_reuse_supported(a):
            a_prec = _precessing_final_spin_from_aligned(
                eta_s,
                chip,
                a,
            )
        else:
            a_prec = precessing_final_spin_2017(
                eta_s,
                chi1,
                chi2,
                chip,
            )
    else:
        a_prec = jnp.asarray(final_spin)

    # First the ringdown frequency (uses a_prec = afinal_prec)
    fRD = qnm_fring_22(a_prec) / (1.0 - Erad)

    # Then the damping frequency (uses a_prec = afinal_prec)
    fdamp = qnm_fdamp_22(a_prec) / (1.0 - Erad)

    # fISCO uses aligned-spin afinal (a), not afinal_prec — LAL sets fISCO before
    # the precessing final spin override.
    a_isco2 = a * a
    if _remnant_foreach_cbrt_supported(a):
        cbrt_one_minus_a2, cbrt_one_plus_a, cbrt_one_minus_a = (
            jnp.foreach_cbrt(
                (1.0 - a_isco2, 1 + a, 1 - a),
                prevalidated=True,
            )
        )
    else:
        cbrt_one_minus_a2 = jnp.cbrt(1.0 - a_isco2)
        cbrt_one_plus_a = jnp.cbrt(1 + a)
        cbrt_one_minus_a = jnp.cbrt(1 - a)
    Z1 = 1.0 + cbrt_one_minus_a2 * (
        cbrt_one_plus_a + cbrt_one_minus_a
    )
    Z1 = jnp.where(Z1 > 3.0, 3.0, Z1)
    Z2 = jnp.sqrt(3.0 * a_isco2 + Z1 * Z1)
    rISCO = 3.0 + Z2 - jnp.sign(a) * jnp.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
    rISCOsq = jnp.sqrt(rISCO)
    rISCO3o2 = rISCOsq * rISCOsq * rISCOsq
    OmegaISCO = 1.0 / (rISCO3o2 + a)
    fISCO = OmegaISCO / PI

    fMECO = (
        (
            0.018744340279608845
            + 0.0077903147004616865 * eta_s
            + 0.003940354686136861 * eta2
            - 0.00006693930988501673 * eta3
        )
        / (1.0 - 0.10423384680638834 * eta_s)
        + (
            (
                S
                * (
                    0.00027180386951683135
                    - 0.00002585252361022052 * S
                    + eta4
                    * (
                        -0.0006807631931297156
                        + 0.022386313074011715 * S
                        - 0.0230825153005985 * S2
                    )
                    + eta2
                    * (
                        0.00036556167661117023
                        - 0.000010021140796150737 * S
                        - 0.00038216081981505285 * S2
                    )
                    + eta_s
                    * (
                        0.00024422562796266645
                        - 0.00001049013062611254 * S
                        - 0.00035182990586857726 * S2
                    )
                    + eta3
                    * (
                        -0.0005418851224505745
                        + 0.000030679548774047616 * S
                        + 4.038390455349854e-6 * S2
                    )
                    - 0.00007547517256664526 * S2
                )
            )
            / (
                0.026666543809890402
                + (
                    -0.014590539285641243
                    - 0.012429476486138982 * eta_s
                    + 1.4861197211952053 * eta4
                    + 0.025066696514373803 * eta2
                    + 0.005146809717492324 * eta3
                )
                * S
                + (
                    -0.0058684526275074025
                    - 0.02876774751921441 * eta_s
                    - 2.551566872093786 * eta4
                    - 0.019641378027236502 * eta2
                    - 0.001956646166089053 * eta3
                )
                * S2
                + (
                    0.003507640638496499
                    + 0.014176504653145768 * eta_s
                    + 1.0 * eta4
                    + 0.012622225233586283 * eta2
                    - 0.00767768214056772 * eta3
                )
                * S3
            )
        )
        + (
            dchi2 * (0.00034375176678815234 + 0.000016343732281057392 * eta_s) * eta2
            + dchi
            * delta
            * eta_s
            * (
                0.08064665214195679 * eta2
                + eta_s * (-0.028476219509487793 - 0.005746537021035632 * S)
                - 0.0011713735642446144 * S
            )
        )
    )

    # NOTE: These are dimensionless frequencies (i.e. M in seconds * f in Hz)
    result = IMRPhenomXRemnant(
        a_prec,
        Erad,
        fRD,
        fdamp,
        fMECO,
        fISCO,
    )
    if _aligned_base_capture is not None:
        _aligned_base_capture.append(
            _AlignedRemnantBase(
                eta_s,
                a,
                Erad,
                fMECO,
                fISCO,
            )
        )
    return result


def _final_spin_2017_python_scalar(eta, chi1, chi2):
    """Evaluate the aligned remnant-spin DAG with C/Python doubles."""

    delta = math.sqrt(max(1.0 - 4.0 * eta, 0.0))
    mass1 = 0.5 * (1.0 + delta)
    mass2 = 0.5 * (1.0 - delta)
    mass1_sq = mass1 * mass1
    mass2_sq = mass2 * mass2
    eta2 = eta * eta
    eta3 = eta2 * eta
    total_spin = (mass1_sq * chi1 + mass2_sq * chi2) / (
        mass1_sq + mass2_sq
    )
    total_spin2 = total_spin * total_spin
    total_spin3 = total_spin2 * total_spin
    delta_chi = chi1 - chi2

    nonspinning = (
        3.4641016151377544 * eta
        + 20.0830030082033 * eta2
        - 12.333573402277912 * eta3
    ) / (1.0 + 7.2388440419467335 * eta)
    equal_spin = (mass1_sq + mass2_sq) * total_spin + (
        (
            -0.8561951310209386 * eta
            - 0.09939065676370885 * eta2
            + 1.668810429851045 * eta3
        )
        * total_spin
        + (
            0.5881660363307388 * eta
            - 2.149269067519131 * eta2
            + 3.4768263932898678 * eta3
        )
        * total_spin2
        + (
            0.142443244743048 * eta
            - 0.9598353840147513 * eta2
            + 1.9595643107593743 * eta3
        )
        * total_spin3
    ) / (
        1.0
        + (
            -0.9142232693081653
            + 2.3191363426522633 * eta
            - 9.710576749140989 * eta3
        )
        * total_spin
    )
    unequal_spin = (
        0.3223660562764661
        * delta_chi
        * delta
        * (1.0 + 9.332575956437443 * eta)
        * eta2
        - 0.059808322561702126 * delta_chi * delta_chi * eta3
        + 2.3170397514509933
        * delta_chi
        * delta
        * (1.0 - 3.2624649875884852 * eta)
        * eta3
        * total_spin
    )
    return nonspinning + equal_spin + unequal_spin


def _cbrt_python_scalar(value):
    """Match the sign/absolute-power sequence used by ``jnp.cbrt``."""

    sign = 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)
    return sign * abs(value) ** (1.0 / 3.0)


def _get_remnant_fMs_python_scalars(m1, m2, chi1, chi2, chip):
    """Evaluate the full scalar remnant fit without 0-D ATen dispatches.

    The expression tree intentionally mirrors :func:`_get_remnant_fMs_uncached_eager`.
    Tensor exponent-two operations are written as multiplication because that
    is the operation executed by ATen for a float64 scalar exponent of two;
    Python's generic ``pow`` differs by one ULP for some inputs.
    """

    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta_s = m1_s * m2_s / (M_s**2.0)

    delta = math.sqrt(max(1.0 - 4.0 * eta_s, 0.0))
    mm1 = 0.5 * (1.0 + delta)
    mm2 = 0.5 * (1.0 - delta)

    chi_eff = mm1 * chi1 + mm2 * chi2

    eta2 = eta_s * eta_s
    eta3 = eta2 * eta_s
    eta4 = eta3 * eta_s
    S = (chi_eff - (38.0 / 113.0) * eta_s * (chi1 + chi2)) / (
        1.0 - (76.0 * eta_s / 113.0)
    )
    S2 = S * S
    S3 = S2 * S

    dchi = chi1 - chi2
    dchi2 = dchi * dchi

    mm1_sq = mm1 * mm1
    mm2_sq = mm2 * mm2
    StotR = (mm1_sq * chi1 + mm2_sq * chi2) / (mm1_sq + mm2_sq)
    StotR2 = StotR * StotR
    StotR3 = StotR2 * StotR

    a = _final_spin_2017_python_scalar(eta_s, chi1, chi2)

    Erad = (
        (
            (
                0.057190958417936644 * eta_s
                + 0.5609904135313374 * eta2
                - 0.84667563764404 * eta3
                + 3.145145224278187 * eta4
            )
            * (
                1
                + (
                    -0.13084389181783257
                    - 1.1387311580238488 * eta_s
                    + 5.49074464410971 * eta2
                )
                * StotR
                + (-0.17762802148331427 + 2.176667900182948 * eta2)
                * StotR2
                + (
                    -0.6320191645391563
                    + 4.952698546796005 * eta_s
                    - 10.023747993978121 * eta2
                )
                * StotR3
            )
        )
        / (
            1
            + (
                -0.9919475346968611
                + 0.367620218664352 * eta_s
                + 4.274567337924067 * eta2
            )
            * StotR
        )
    ) + (
        -0.09803730445895877
        * dchi
        * delta
        * (1 - 3.2283713377939134 * eta_s)
        * eta2
        + 0.01118530335431078 * dchi2 * eta3
        - 0.01978238971523653
        * dchi
        * delta
        * (1 - 4.91667749015812 * eta_s)
        * eta_s
        * StotR
    )

    larger_mass_fraction = 0.5 * (
        1.0 + math.sqrt(max(1.0 - 4.0 * eta_s, 0.0))
    )
    larger_mass_fraction_sq = larger_mass_fraction * larger_mass_fraction
    perpendicular_spin = chip * larger_mass_fraction_sq
    a_prec = math.copysign(
        math.sqrt(a * a + perpendicular_spin * perpendicular_spin),
        a,
    )

    fRD = qnm_fring_22(a_prec) / (1.0 - Erad)
    fdamp = qnm_fdamp_22(a_prec) / (1.0 - Erad)

    a_isco2 = a * a
    Z1 = 1.0 + _cbrt_python_scalar(1.0 - a_isco2) * (
        _cbrt_python_scalar(1 + a) + _cbrt_python_scalar(1 - a)
    )
    Z1 = 3.0 if Z1 > 3.0 else Z1
    Z2 = math.sqrt(3.0 * a_isco2 + Z1 * Z1)
    sign_a = 1.0 if a > 0.0 else (-1.0 if a < 0.0 else 0.0)
    rISCO = 3.0 + Z2 - sign_a * math.sqrt(
        (3 - Z1) * (3 + Z1 + 2 * Z2)
    )
    rISCOsq = math.sqrt(rISCO)
    rISCO3o2 = rISCOsq * rISCOsq * rISCOsq
    OmegaISCO = 1.0 / (rISCO3o2 + a)
    fISCO = OmegaISCO / PI

    fMECO = (
        (
            0.018744340279608845
            + 0.0077903147004616865 * eta_s
            + 0.003940354686136861 * eta2
            - 0.00006693930988501673 * eta3
        )
        / (1.0 - 0.10423384680638834 * eta_s)
        + (
            (
                S
                * (
                    0.00027180386951683135
                    - 0.00002585252361022052 * S
                    + eta4
                    * (
                        -0.0006807631931297156
                        + 0.022386313074011715 * S
                        - 0.0230825153005985 * S2
                    )
                    + eta2
                    * (
                        0.00036556167661117023
                        - 0.000010021140796150737 * S
                        - 0.00038216081981505285 * S2
                    )
                    + eta_s
                    * (
                        0.00024422562796266645
                        - 0.00001049013062611254 * S
                        - 0.00035182990586857726 * S2
                    )
                    + eta3
                    * (
                        -0.0005418851224505745
                        + 0.000030679548774047616 * S
                        + 4.038390455349854e-6 * S2
                    )
                    - 0.00007547517256664526 * S2
                )
            )
            / (
                0.026666543809890402
                + (
                    -0.014590539285641243
                    - 0.012429476486138982 * eta_s
                    + 1.4861197211952053 * eta4
                    + 0.025066696514373803 * eta2
                    + 0.005146809717492324 * eta3
                )
                * S
                + (
                    -0.0058684526275074025
                    - 0.02876774751921441 * eta_s
                    - 2.551566872093786 * eta4
                    - 0.019641378027236502 * eta2
                    - 0.001956646166089053 * eta3
                )
                * S2
                + (
                    0.003507640638496499
                    + 0.014176504653145768 * eta_s
                    + 1.0 * eta4
                    + 0.012622225233586283 * eta2
                    - 0.00767768214056772 * eta3
                )
                * S3
            )
        )
        + (
            dchi2
            * (0.00034375176678815234 + 0.000016343732281057392 * eta_s)
            * eta2
            + dchi
            * delta
            * eta_s
            * (
                0.08064665214195679 * eta2
                + eta_s
                * (-0.028476219509487793 - 0.005746537021035632 * S)
                - 0.0011713735642446144 * S
            )
        )
    )

    return (
        (a_prec, Erad, fRD, fdamp, fMECO, fISCO),
        (eta_s, a, Erad, fMECO, fISCO),
    )


def _remnant_python_scalars_runtime_supported():
    """Reject transforms, tensor modes, AD, and autocast."""

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


def _remnant_python_scalars_supported(
    m1,
    m2,
    chi1,
    chi2,
    chip,
    *,
    final_spin,
    aligned_base,
    aligned_base_capture,
):
    """Accept only the calibrated plain-float CPU float64 path."""

    if not _remnant_python_scalars_enabled():
        return False
    reference = _ACTIVE_TENSOR.get()
    values = (m1, m2, chi1, chi2, chip)
    return (
        _remnant_final_spin_reuse_enabled()
        and final_spin is None
        and aligned_base is None
        and (
            aligned_base_capture is None
            or type(aligned_base_capture) is list
        )
        and all(type(value) is float for value in values)
        and all(math.isfinite(value) for value in values)
        and m1 > 0.0
        and m2 > 0.0
        and -1.0 <= chi1 <= 1.0
        and -1.0 <= chi2 <= 1.0
        and 0.0 <= chip <= 1.0
        and (
            reference is None
            or (
                type(reference) is torch.Tensor
                and reference.layout is torch.strided
                and reference.device.type == "cpu"
                and reference.dtype == torch.float64
                and not reference.is_conj()
                and not reference.is_neg()
                and not _tree_has_autograd_untrusted(reference)
            )
        )
        and _remnant_python_scalars_runtime_supported()
    )


def _raw_scalar_sequence_matches(left, right):
    """Return whether scalar sequences have identical binary64 values."""

    try:
        return len(left) == len(right) and all(
            math.isfinite(float(left_value))
            and float(left_value).hex() == float(right_value).hex()
            for left_value, right_value in zip(left, right)
        )
    except (RuntimeError, TypeError, ValueError):
        return False


@lru_cache(maxsize=1)
def _remnant_python_scalars_calibrated():
    """Verify this Python/libm/Torch build is raw-byte compatible once."""

    sentinels = (
        (35.0, 12.0, -0.3, 0.4, 0.0),
        (30.0, 30.0, 0.99, -0.99, 0.99),
        (
            763.7800496707056,
            71.96588066610528,
            0.6525381219784064,
            -0.1218984489598034,
            0.1348269379900036,
        ),
        (
            2.1909855759424017,
            16.099243804637066,
            -0.514810978779217,
            -0.38025048592303456,
            0.9106746315492824,
        ),
    )
    try:
        for sentinel in sentinels:
            captured = []
            eager = _get_remnant_fMs_uncached_eager(
                *sentinel,
                _aligned_base_capture=captured,
            )
            candidate, base = _get_remnant_fMs_python_scalars(*sentinel)
            if len(captured) != 1:
                return False
            if not _raw_scalar_sequence_matches(eager, candidate):
                return False
            if not _raw_scalar_sequence_matches(captured[0], base):
                return False
    except Exception:
        return False
    return True


def _materialize_remnant_python_scalars(result_values, base_values):
    """Materialize calibrated binary64 values in the active Torch context."""

    a_prec, Erad, fRD, fdamp, fMECO, fISCO = (
        jnp.asarray(value) for value in result_values
    )
    eta_s, aligned_spin_value, _, _, _ = base_values
    aligned_spin = jnp.asarray(aligned_spin_value)
    tensors = (a_prec, Erad, fRD, fdamp, fMECO, fISCO, aligned_spin)
    if not all(
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.device.type == "cpu"
        and value.dtype == torch.float64
        and value.ndim == 0
        and not value.is_conj()
        and not value.is_neg()
        for value in tensors
    ):
        raise RuntimeError("scalar remnant materialization changed semantics")
    result = IMRPhenomXRemnant(a_prec, Erad, fRD, fdamp, fMECO, fISCO)
    base = _AlignedRemnantBase(eta_s, aligned_spin, Erad, fMECO, fISCO)
    return result, base


def _get_remnant_fMs_uncached(
    m1: FloatLike,
    m2: FloatLike,
    chi1: FloatLike,
    chi2: FloatLike,
    chip: float | FloatLike = 0.0,
    *,
    final_spin: FloatLike | None = None,
    _aligned_base: _AlignedRemnantBase | None = None,
    _aligned_base_capture: list[_AlignedRemnantBase] | None = None,
) -> IMRPhenomXRemnant:
    """Use the exact host-scalar fit when its strict gate is supported."""

    if not _remnant_python_scalars_supported(
        m1,
        m2,
        chi1,
        chi2,
        chip,
        final_spin=final_spin,
        aligned_base=_aligned_base,
        aligned_base_capture=_aligned_base_capture,
    ):
        return _get_remnant_fMs_uncached_eager(
            m1,
            m2,
            chi1,
            chi2,
            chip,
            final_spin=final_spin,
            _aligned_base=_aligned_base,
            _aligned_base_capture=_aligned_base_capture,
        )
    try:
        if not _remnant_python_scalars_calibrated():
            raise RuntimeError("scalar remnant calibration failed")
        result_values, base_values = _get_remnant_fMs_python_scalars(
            m1,
            m2,
            chi1,
            chi2,
            chip,
        )
        if not all(
            math.isfinite(value) for value in (*result_values, *base_values)
        ):
            raise ArithmeticError("non-finite scalar remnant result")
        result, base = _materialize_remnant_python_scalars(
            result_values,
            base_values,
        )
    except Exception:
        return _get_remnant_fMs_uncached_eager(
            m1,
            m2,
            chi1,
            chi2,
            chip,
            final_spin=final_spin,
            _aligned_base=_aligned_base,
            _aligned_base_capture=_aligned_base_capture,
        )
    if _aligned_base_capture is not None:
        _aligned_base_capture.append(base)
    return result


def _get_aligned_remnant_for_packed_plan(
    m1: FloatLike,
    m2: FloatLike,
    chi1: FloatLike,
    chi2: FloatLike,
) -> tuple[IMRPhenomXRemnant, _AlignedRemnantBase]:
    """Evaluate one aligned fit while retaining its request-local base."""

    captured = []
    aligned = _get_remnant_fMs_uncached(
        m1,
        m2,
        chi1,
        chi2,
        0.0,
        _aligned_base_capture=captured,
    )
    if len(captured) != 1:
        raise RuntimeError("aligned remnant fit did not produce one base")
    return aligned, captured[0]


def _pack_remnant_plan(
    aligned: IMRPhenomXRemnant,
    base: _AlignedRemnantBase,
    final_spin: FloatLike,
) -> _PackedRemnantPlan | None:
    """Evaluate aligned and carrier QNM fits on one fixed two-value lane.

    The caller owns ``aligned`` and ``base`` for this waveform request.  This
    helper deliberately fails closed unless both requested final spins are
    ordinary scalar binary64 CPU tensors without AD state.
    """

    carrier_spin = jnp.asarray(final_spin)
    spin_values = (aligned.final_spin, carrier_spin)
    if not all(
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.device.type == "cpu"
        and value.dtype == torch.float64
        and value.ndim == 0
        and not value.is_conj()
        and not value.is_neg()
        and not _tree_has_autograd_untrusted(value)
        for value in spin_values
    ):
        return None
    shared_values = (
        base.radiated_energy,
        base.meco_frequency,
        base.isco_frequency,
    )
    if not all(
        type(value) is torch.Tensor
        and value.layout is torch.strided
        and value.device == carrier_spin.device
        and value.dtype == carrier_spin.dtype
        and value.ndim == 0
        and not value.is_conj()
        and not value.is_neg()
        and not _tree_has_autograd_untrusted(value)
        for value in shared_values
    ):
        return None

    spin_lane = torch.stack(spin_values)
    denominator = 1.0 - base.radiated_energy
    ringdown_lane = qnm_fring_22(spin_lane) / denominator
    damping_lane = qnm_fdamp_22(spin_lane) / denominator
    aligned_ringdown, carrier_ringdown = ringdown_lane.unbind()
    aligned_damping, carrier_damping = damping_lane.unbind()
    aligned_result = IMRPhenomXRemnant(
        aligned.final_spin,
        base.radiated_energy,
        aligned_ringdown,
        aligned_damping,
        base.meco_frequency,
        base.isco_frequency,
    )
    carrier_result = IMRPhenomXRemnant(
        carrier_spin,
        base.radiated_energy,
        carrier_ringdown,
        carrier_damping,
        base.meco_frequency,
        base.isco_frequency,
    )
    return _PackedRemnantPlan(aligned_result, carrier_result)


@lru_cache(maxsize=256)
def _cached_remnant_fMs_scalar(
    m1_f: float,
    m2_f: float,
    chi1_f: float,
    chi2_f: float,
    chip_f: float,
    final_spin_f: float | None,
) -> IMRPhenomXRemnant:
    return _get_remnant_fMs_uncached(
        m1_f, m2_f, chi1_f, chi2_f, chip_f, final_spin=final_spin_f
    )


def get_remnant_fMs(
    m1: FloatLike,
    m2: FloatLike,
    chi1: FloatLike,
    chi2: FloatLike,
    chip: float | FloatLike = 0.0,
    *,
    final_spin: FloatLike | None = None,
) -> IMRPhenomXRemnant:
    """Return remnant fits, using an active per-waveform cache if present."""

    args = (m1, m2, chi1, chi2, chip)
    kwargs = {"final_spin": final_spin}
    cache = _ACTIVE_REMNANT_CACHE.get()
    if cache is not None:
        if (
            _remnant_aligned_base_reuse_enabled()
            and _remnant_final_spin_reuse_enabled()
        ):
            return cache.evaluate_with_aligned_base(
                _get_remnant_fMs_uncached,
                args,
                kwargs,
            )
        return cache.evaluate(_get_remnant_fMs_uncached, args, kwargs)

    if (
        type(m1) in (float, int)
        and type(m2) in (float, int)
        and type(chi1) in (float, int)
        and type(chi2) in (float, int)
        and type(chip) in (float, int)
        and (final_spin is None or type(final_spin) in (float, int))
    ):
        return _cached_remnant_fMs_scalar(
            float(m1),
            float(m2),
            float(chi1),
            float(chi2),
            float(chip),
            float(final_spin) if final_spin is not None else None,
        )

    return _get_remnant_fMs_uncached(*args, **kwargs)


def get_cutoff_fMs(
    m1: FloatLike,
    m2: FloatLike,
    chi1: FloatLike,
    chi2: FloatLike,
    chip: float | FloatLike = 0.0,
    *,
    final_spin: FloatLike | None = None,
) -> tuple[FloatLike, FloatLike, FloatLike, FloatLike]:
    """Return the legacy PhenomX frequency tuple.

    New code that also needs remnant properties should use
    :func:`get_remnant_fMs` so the shared final-spin and radiated-energy fits
    are evaluated only once.
    """

    remnant = get_remnant_fMs(
        m1,
        m2,
        chi1,
        chi2,
        chip,
        final_spin=final_spin,
    )
    return (
        remnant.ringdown_frequency,
        remnant.damping_frequency,
        remnant.meco_frequency,
        remnant.isco_frequency,
    )


def _shared_scalar_powers_supported(*values):
    """Return whether identical scalar power results may be shared safely."""

    if (
        not _derived_power_reuse_enabled()
        or not values
        or not _remnant_python_scalars_runtime_supported()
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
        and value._base is None
        and not value.is_conj()
        and not value.is_neg()
        and not _tree_has_autograd_untrusted(value)
        for value in values
    )


def _calc_phaseatpeak_shared_powers(eta, S, chia, delta):
    """Evaluate the legacy fits while sharing identical scalar powers."""

    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4
    eta5 = eta**5
    eta6 = eta**6
    S2 = S**2
    S3 = S**3
    S4 = S**4
    linb = (
        (
            3155.1635543201924
            + 1257.9949740608242 * eta
            - 32243.28428870599 * eta2
            + 347213.65466875216 * eta3
            - 1.9223851649491738e6 * eta4
            + 5.3035911346921865e6 * eta5
            - 5.789128656876938e6 * eta6
        )
        + (
            (-24.181508118588667 + 115.49264174560281 * eta - 380.19778216022763 * eta2)
            * S
            + (24.72585609641552 - 328.3762360751952 * eta + 725.6024119989094 * eta2)
            * S2
            + (23.404604124552 - 646.3410199799737 * eta + 1941.8836639529036 * eta2)
            * S3
            + (-12.814828278938885 - 325.92980012408367 * eta + 1320.102640190539 * eta2)
            * S4
        )
        + (-148.17317525117338 * chia * delta * eta2)
    )
    psi4tostrain = (
        (
            13.39320482758057
            - 175.42481512989315 * eta
            + 2097.425116152503 * eta2
            - 9862.84178637907 * eta3
            + 16026.897939722587 * eta4
        )
        + (
            (4.7895602776763 - 163.04871764530466 * eta + 609.5575850476959 * eta2)
            * S
            + (1.3934428041390161 - 97.51812681228478 * eta + 376.9200932531847 * eta2)
            * S2
            + (15.649521097877374 + 137.33317057388916 * eta - 755.9566456906406 * eta2)
            * S3
            + (13.097315867845788 + 149.30405703643288 * eta - 764.5242164872267 * eta2)
            * S4
        )
        + (105.37711654943146 * chia * delta * eta2)
    )
    return 0.0, linb, psi4tostrain


def calc_phaseatpeak(
    eta: FloatLike, S: FloatLike, chia: FloatLike, delta: FloatLike
) -> tuple[FloatLike, FloatLike, FloatLike]:
    if _shared_scalar_powers_supported(eta, S, chia, delta):
        return _calc_phaseatpeak_shared_powers(eta, S, chia, delta)
    lina = 0.0

    linb = (
        (
            3155.1635543201924
            + 1257.9949740608242 * eta
            - 32243.28428870599 * eta**2
            + 347213.65466875216 * eta**3
            - 1.9223851649491738e6 * eta**4
            + 5.3035911346921865e6 * eta**5
            - 5.789128656876938e6 * eta**6
        )
        + (
            (
                -24.181508118588667
                + 115.49264174560281 * eta
                - 380.19778216022763 * eta**2
            )
            * S
            + (24.72585609641552 - 328.3762360751952 * eta + 725.6024119989094 * eta**2)
            * S**2
            + (23.404604124552 - 646.3410199799737 * eta + 1941.8836639529036 * eta**2)
            * S**3
            + (
                -12.814828278938885
                - 325.92980012408367 * eta
                + 1320.102640190539 * eta**2
            )
            * S**4
        )
        + (-148.17317525117338 * chia * delta * eta**2)
    )

    psi4tostrain = (
        (
            13.39320482758057
            - 175.42481512989315 * eta
            + 2097.425116152503 * eta**2
            - 9862.84178637907 * eta**3
            + 16026.897939722587 * eta**4
        )
        + (
            (4.7895602776763 - 163.04871764530466 * eta + 609.5575850476959 * eta**2)
            * S
            + (
                1.3934428041390161
                - 97.51812681228478 * eta
                + 376.9200932531847 * eta**2
            )
            * S**2
            + (
                15.649521097877374
                + 137.33317057388916 * eta
                - 755.9566456906406 * eta**2
            )
            * S**3
            + (
                13.097315867845788
                + 149.30405703643288 * eta
                - 764.5242164872267 * eta**2
            )
            * S**4
        )
        + (105.37711654943146 * chia * delta * eta**2)
    )
    return lina, linb, psi4tostrain


def nospin_CV(NoSpin_coeffs: Float[Array, " n_coeffs"], eta: FloatLike) -> FloatLike:
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    return (
        NoSpin_coeffs[..., 0]
        + NoSpin_coeffs[..., 1] * eta
        + NoSpin_coeffs[..., 2] * eta2
        + NoSpin_coeffs[..., 3] * eta3
        + NoSpin_coeffs[..., 4] * eta4
        + NoSpin_coeffs[..., 5] * eta5
    ) / (
        NoSpin_coeffs[..., 6]
        + NoSpin_coeffs[..., 7] * eta
        + NoSpin_coeffs[..., 8] * eta2
        + NoSpin_coeffs[..., 9] * eta3
    )


def Eqspin_CV(
    EqSpin_coeffs: Float[Array, " n_coeffs"], eta: FloatLike, S: FloatLike
) -> FloatLike:
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S
    numerator = S * (
        EqSpin_coeffs[..., 0]
        + EqSpin_coeffs[..., 1] * S
        + EqSpin_coeffs[..., 2] * S2
        + EqSpin_coeffs[..., 3] * S3
        + EqSpin_coeffs[..., 4] * S4
        + eta
        * (
            EqSpin_coeffs[..., 5]
            + EqSpin_coeffs[..., 6] * S
            + EqSpin_coeffs[..., 7] * S2
            + EqSpin_coeffs[..., 8] * S3
            + EqSpin_coeffs[..., 9] * S4
        )
        + eta2
        * (
            EqSpin_coeffs[..., 10]
            + EqSpin_coeffs[..., 11] * S
            + EqSpin_coeffs[..., 12] * S2
            + EqSpin_coeffs[..., 13] * S3
            + EqSpin_coeffs[..., 14] * S4
        )
        + eta3
        * (
            EqSpin_coeffs[..., 15]
            + EqSpin_coeffs[..., 16] * S
            + EqSpin_coeffs[..., 17] * S2
            + EqSpin_coeffs[..., 18] * S3
            + EqSpin_coeffs[..., 19] * S4
        )
        + eta4
        * (
            EqSpin_coeffs[..., 20]
            + EqSpin_coeffs[..., 21] * S
            + EqSpin_coeffs[..., 22] * S2
            + EqSpin_coeffs[..., 23] * S3
            + EqSpin_coeffs[..., 24] * S4
        )
    )
    denominator = (
        EqSpin_coeffs[..., 25]
        + EqSpin_coeffs[..., 26] * S
        + EqSpin_coeffs[..., 27] * S2
        + EqSpin_coeffs[..., 28] * S3
    )
    return numerator / denominator


def Uneqspin_CV(
    EqSpin_coeffs: Float[Array, " n_coeffs"],
    eta: FloatLike,
    S: FloatLike,
    chia: FloatLike,
) -> FloatLike:
    chia2 = chia * chia
    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    return (
        chia
        * delta
        * eta
        * (
            EqSpin_coeffs[..., 0]
            + EqSpin_coeffs[..., 1] * eta
            + EqSpin_coeffs[..., 2] * eta2
            + EqSpin_coeffs[..., 3] * eta3
            + EqSpin_coeffs[..., 4] * eta4
            + EqSpin_coeffs[..., 5] * eta5
            + EqSpin_coeffs[..., 6] * S
            + EqSpin_coeffs[..., 7] * S * eta2
            + EqSpin_coeffs[..., 8] * S * eta3
        )
        + EqSpin_coeffs[..., 9] * chia2 * eta
    )


class _AmpFitPowers(NamedTuple):
    """Exact powers shared by the three amplitude collocation fits."""

    eta2: FloatLike
    eta3: FloatLike
    eta4: FloatLike
    eta5: FloatLike
    spin2: FloatLike
    spin3: FloatLike
    spin4: FloatLike
    spin5: FloatLike


def _prepare_amp_fit_powers(eta, spin):
    """Return an AD-safe exact power plan, or ``None`` for eager fallback."""

    if (
        not _derived_power_reuse_enabled()
        or not _remnant_python_scalars_runtime_supported()
    ):
        return None
    if (
        type(eta) is not torch.Tensor
        or type(spin) is not torch.Tensor
        or eta.layout is not torch.strided
        or spin.layout is not torch.strided
        or eta.ndim != 0
        or spin.ndim != 1
        or eta.dtype not in (torch.float32, torch.float64)
        or spin.dtype != eta.dtype
        or eta.device != spin.device
        or eta.device.type not in ("cpu", "cuda")
        or eta._base is not None
        or spin._base is not None
        or eta.is_conj()
        or eta.is_neg()
        or spin.is_conj()
        or spin.is_neg()
        or _tree_has_autograd_untrusted((eta, spin))
    ):
        return None
    return _AmpFitPowers(
        eta**2,
        eta**3,
        eta**4,
        eta**5,
        spin**2,
        spin**3,
        spin**4,
        spin**5,
    )


def Amp_Nospin_CV(
    NoSpin_coeffs: Float[Array, " n_coeffs"],
    eta: FloatLike,
    *,
    powers: _AmpFitPowers | None = None,
) -> FloatLike:
    if powers is None:
        numerator = (
            NoSpin_coeffs[..., 0]
            + NoSpin_coeffs[..., 1] * eta
            + NoSpin_coeffs[..., 2] * eta**2
            + NoSpin_coeffs[..., 3] * eta**3
            + NoSpin_coeffs[..., 4] * eta**4
        )
        denominator = (
            NoSpin_coeffs[..., 5]
            + NoSpin_coeffs[..., 6] * eta
            + NoSpin_coeffs[..., 7] * eta**2
        )
        return numerator / denominator

    numerator = (
        NoSpin_coeffs[..., 0]
        + NoSpin_coeffs[..., 1] * eta
        + NoSpin_coeffs[..., 2] * powers.eta2
        + NoSpin_coeffs[..., 3] * powers.eta3
        + NoSpin_coeffs[..., 4] * powers.eta4
    )
    denominator = (
        NoSpin_coeffs[..., 5]
        + NoSpin_coeffs[..., 6] * eta
        + NoSpin_coeffs[..., 7] * powers.eta2
    )
    return numerator / denominator


def Amp_Eqspin_CV(
    EqSpin_coeffs: Float[Array, " n_coeffs"],
    eta: FloatLike,
    S: FloatLike,
    *,
    powers: _AmpFitPowers | None = None,
) -> FloatLike:
    if powers is None:
        numeratorS0 = (
            EqSpin_coeffs[..., 0]
            + EqSpin_coeffs[..., 1] * eta
            + EqSpin_coeffs[..., 2] * eta**2
            + EqSpin_coeffs[..., 3] * eta**3
        )
        numeratorS1 = (
            EqSpin_coeffs[..., 4]
            + EqSpin_coeffs[..., 5] * eta
            + EqSpin_coeffs[..., 6] * eta**2
            + EqSpin_coeffs[..., 7] * eta**3
        )
        numeratorS2 = (
            EqSpin_coeffs[..., 8]
            + EqSpin_coeffs[..., 9] * eta
            + EqSpin_coeffs[..., 10] * eta**2
            + EqSpin_coeffs[..., 11] * eta**3
        )
        numeratorS3 = (
            EqSpin_coeffs[..., 12]
            + EqSpin_coeffs[..., 13] * eta
            + EqSpin_coeffs[..., 14] * eta**2
            + EqSpin_coeffs[..., 15] * eta**3
        )
        numeratorS4 = (
            EqSpin_coeffs[..., 16]
            + EqSpin_coeffs[..., 17] * eta
            + EqSpin_coeffs[..., 18] * eta**2
            + EqSpin_coeffs[..., 19] * eta**3
        )
        numeratorS5 = (
            EqSpin_coeffs[..., 20]
            + EqSpin_coeffs[..., 21] * eta
            + EqSpin_coeffs[..., 22] * eta**2
            + EqSpin_coeffs[..., 23] * eta**3
        )
        denominator = (
            EqSpin_coeffs[..., 24]
            + EqSpin_coeffs[..., 25] * S
            + EqSpin_coeffs[..., 26] * eta
            + EqSpin_coeffs[..., 27] * S**2
        )
        return (
            numeratorS0
            + numeratorS1 * S
            + numeratorS2 * S**2
            + numeratorS3 * S**3
            + numeratorS4 * S**4
            + numeratorS5 * S**5
        ) / denominator

    numeratorS0 = (
        EqSpin_coeffs[..., 0]
        + EqSpin_coeffs[..., 1] * eta
        + EqSpin_coeffs[..., 2] * powers.eta2
        + EqSpin_coeffs[..., 3] * powers.eta3
    )
    numeratorS1 = (
        EqSpin_coeffs[..., 4]
        + EqSpin_coeffs[..., 5] * eta
        + EqSpin_coeffs[..., 6] * powers.eta2
        + EqSpin_coeffs[..., 7] * powers.eta3
    )
    numeratorS2 = (
        EqSpin_coeffs[..., 8]
        + EqSpin_coeffs[..., 9] * eta
        + EqSpin_coeffs[..., 10] * powers.eta2
        + EqSpin_coeffs[..., 11] * powers.eta3
    )
    numeratorS3 = (
        EqSpin_coeffs[..., 12]
        + EqSpin_coeffs[..., 13] * eta
        + EqSpin_coeffs[..., 14] * powers.eta2
        + EqSpin_coeffs[..., 15] * powers.eta3
    )
    numeratorS4 = (
        EqSpin_coeffs[..., 16]
        + EqSpin_coeffs[..., 17] * eta
        + EqSpin_coeffs[..., 18] * powers.eta2
        + EqSpin_coeffs[..., 19] * powers.eta3
    )
    numeratorS5 = (
        EqSpin_coeffs[..., 20]
        + EqSpin_coeffs[..., 21] * eta
        + EqSpin_coeffs[..., 22] * powers.eta2
        + EqSpin_coeffs[..., 23] * powers.eta3
    )
    denominator = (
        EqSpin_coeffs[..., 24]
        + EqSpin_coeffs[..., 25] * S
        + EqSpin_coeffs[..., 26] * eta
        + EqSpin_coeffs[..., 27] * powers.spin2
    )
    return (
        numeratorS0
        + numeratorS1 * S
        + numeratorS2 * powers.spin2
        + numeratorS3 * powers.spin3
        + numeratorS4 * powers.spin4
        + numeratorS5 * powers.spin5
    ) / denominator


def Amp_Uneqspin_CV(
    UneqSpin_coeffs: Float[Array, " n_coeffs"],
    eta: FloatLike,
    S: FloatLike,
    chia: FloatLike,
    *,
    powers: _AmpFitPowers | None = None,
) -> FloatLike:
    delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
    if powers is None:
        return (
            chia
            * delta
            * (
                UneqSpin_coeffs[..., 0]
                + UneqSpin_coeffs[..., 1] * eta
                + UneqSpin_coeffs[..., 2] * eta**2
                + UneqSpin_coeffs[..., 3] * eta**3
                + UneqSpin_coeffs[..., 4] * eta**4
                + UneqSpin_coeffs[..., 5] * eta**5
            )
        )

    return (
        chia
        * delta
        * (
            UneqSpin_coeffs[..., 0]
            + UneqSpin_coeffs[..., 1] * eta
            + UneqSpin_coeffs[..., 2] * powers.eta2
            + UneqSpin_coeffs[..., 3] * powers.eta3
            + UneqSpin_coeffs[..., 4] * powers.eta4
            + UneqSpin_coeffs[..., 5] * powers.eta5
        )
    )


PhenomX_amp_coeff_table = jnp.array(
    [
        [  # Coeffs for CV_Amp_Ins0 (ind 0)
            -0.015178276424448592,  # No spin
            -0.06098548699809163,
            0.4845148547154606,
            0.0,
            0.0,
            1.0,
            0.09799277215675059,
            0.0,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            0.02300153747158323,
            0.0,
            0.10495263104245876,
            0.0,
            0.01761591799745109,
            0.0,
            -0.14404522791467844,
            0.0,
            0.0,
            0.04834642258922544,
            -0.14189350657140673,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -0.7340448493183307,
            0.0,
            0.0,
            0.0,  # Uneq spin
            0.0,
            0.0,
            0.0,
            0.0018724905795891192,
            34.90874132485147,
        ],
        [  # Coeffs for CV_Amp_Ins1 (ind 1)
            -0.058572000924124644,  # No spin
            -1.1970535595488723,
            8.4630293045015,
            0.0,
            0.0,
            1.0,
            15.430818840453686,
            0.0,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            -0.08746408292050666,
            -0.20646621646484237,
            0.788717372588848,
            0.0,
            -0.018924013869130434,
            -0.21291764491897636,
            +0.8282888482429105,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.332123330797879,
            1.0,
            0.0,
            0.0,
            0.0,  # Uneq spin
            0.0,
            0.0,
            0.0,
            0.004389995099201855,
            105.84553997647659,
        ],
        [  # Coeffs for CV_Amp_Ins2 (ind 2)
            -0.16212854591357853,  # No spin
            1.617404703616985,
            -3.186012733446088,
            5.629598195000046,
            0.0,
            1.0,
            0.04507019231274476,
            0.0,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            1.0055835408962206,
            -4.127597118865669,
            18.353433894421833,
            -41.0378120175805,
            -0.31443470118113853,
            5.215501942120774,
            -18.80590889704093,
            19.099315016873643,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            5.852706459485663,
            -5.717874483424523,
            0.0,
            1.0,
            0.0,  # Uneq spin
            0.0,
            0.0,
            0.0,
            0.05575955418803233,
            208.92352600701068,
        ],
        [  # Coeffs for V2 (ind 3)
            1.4873184918202145,  # No spin
            1974.6112656679577,
            27563.641024162127,
            -19837.908020966777,
            0.0,
            1.0,
            143.29004876335128,
            458.4097306093354,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            27.952730865904343,
            -365.55631765202895,
            0.0,
            1612.2681322644232,
            3.2646808851249016,
            -260.3494489873286,
            3011.446602208493,
            -6962.675551371755,
            -19.38970173389662,
            0.0,
            0.0,
            1486.4658089990298,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            12.647425554323242,
            -10.540154508599963,
            0.0,
            1.0,
            0.0,  # Uneq spin
            0.0,
            -0.016404056649860943,
            -296.473359655246,
            0.0,
            0.0,
        ],
        [  # Coeffs for gamma2 (ind 4)
            0.8312293675316895,  # No spin
            7.480371544268765,
            -18.256121237800397,
            0.0,
            0.0,
            1.0,
            10.915453595496611,
            -30.578409433912874,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            0.5869408584532747,
            -0.1467158405070222,
            0.25295441250444334,
            0.0,
            0.031852563636196894,
            -2.8489481072076472,
            4.6849496672664594,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.8775263105069953,
            -3.41755361841226,
            0.0,
            1.0,
            0.0,  # Uneq spin
            -0.00548054788508203,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs for gamma3 (ind 5)
            1.3666000000000007,  # No spin
            -4.091333144596439,
            2.109081209912545,
            -4.222259944408823,
            0.0,
            1.0,
            -2.7440263888207594,
            0.0,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            0.07179105336478316,
            -0.8752427297525086,
            2.331724812782498,
            0.0,
            -0.05633734476062242,
            0.4168560229353532,
            -0.6330998412809531,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,  # Uneq spin
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs for v1RD (ind 6)
            0.03689164742964719,  # No spin
            25.417967754401182,
            162.52904393600332,
            0.0,
            0.0,
            1.0,
            61.19874463331437,
            -29.628854485544874,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            -0.14352506969368556,
            -4.805034453745424,
            27.675454081988036,
            -48.31945248941757,
            0.026356911108320547,
            1.11147906765112,
            -2.398327419614959,
            -3.751501972663298,
            0.19967405175523437,
            6.176053843938542,
            -47.99096500250743,
            81.9290740950083,
            -0.05292913111731128,
            -0.2874540719094058,
            -5.104257870393138,
            30.491948143930266,
            -0.18147275151697131,
            -8.990840289951514,
            72.08174136362386,
            -132.77982622925845,
            -1.4160870461211452,
            1.0,
            0.0,
            0.0,
            0.0,  # Uneq spin
            0.0,
            -0.04426571511345366,
            0.0,
            0.0,
            0.0,
        ],
    ]
)

PhenomX_phase_coeff_table = jnp.array(
    [
        [  # Coeffs collocation point 0 of the inspiral phase (ind 0)
            -17294.000000000007,  # No spin
            -19943.076428555978,
            483033.0998073767,
            0.0,
            0.0,
            0.0,
            1.0,
            4.460294035404433,
            0.0,
            0.0,
            68384.62786426462,  # Eq spin
            67663.42759836042,
            -2179.3505885609297,
            19703.894135534803,
            32614.091002011017,
            -58475.33302037833,
            62190.404951852535,
            18298.307770807573,
            -303141.1945565486,
            0.0,
            -148368.4954044637,
            -758386.5685734496,
            -137991.37032619823,
            1.0765877367729193e6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0412979553629143,
            1.0,
            0.0,
            0.0,
            12017.062595934838,  # UnEq Spin
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 1 of the inspiral phase (ind 1)
            -7579.300000000004,  # No spin
            -120297.86185566607,
            1.1694356931282217e6,
            -557253.0066989232,
            0.0,
            0.0,
            1.0,
            18.53018618227582,
            0.0,
            0.0,
            -27089.36915061857,  # Eq spin
            -66228.9369155027,
            -44331.41741405198,
            0.0,
            0.0,
            50644.13475990821,
            157036.45676788126,
            126736.43159783827,
            0.0,
            0.0,
            150022.21343386435,
            -50166.382087278434,
            -399712.22891153296,
            0.0,
            0.0,
            -593633.5370110178,
            -325423.99477314285,
            +847483.2999508682,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.5232497464826662,
            -3.062957826830017,
            -1.130185486082531,
            1.0,
            3843.083992827935,  # UnEq Spin
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 2 of the inspiral phase (ind 2)
            15415.000000000007,  # No spin
            873401.6255736464,
            376665.64637025696,
            -3.9719980569125614e6,
            8.913612508054944e6,
            0.0,
            1.0,
            46.83697749859996,
            0.0,
            0.0,
            397951.95299014193,  # Eq spin
            -207180.42746987,
            -130668.37221912303,
            0.0,
            0.0,
            -1.0053073129700898e6,
            1.235279439281927e6,
            -174952.69161683554,
            0.0,
            0.0,
            -1.9826323844247842e6,
            208349.45742548333,
            895372.155565861,
            0.0,
            0.0,
            4.662143741417853e6,
            -584728.050612325,
            -1.6894189124921719e6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -9.675704197652225,
            3.5804521763363075,
            2.5298346636273306,
            1.0,
            -24708.109411857182,  # UnEq Spin
            24703.28267342699,
            0.0,
            0.0,
            0.0,
            0.0,
            47752.17032707405,
            0.0,
            0.0,
            -1296.9289110696955,
        ],
        [  # Coeffs collocation point 3 of the inspiral phase (ind 3)
            2439.000000000001,  # No spin
            -31133.52170083207,
            28867.73328134167,
            0.0,
            0.0,
            0.0,
            1.0,
            0.41143032589262585,
            0.0,
            0.0,
            16116.057657391262,  # Eq spin
            9861.635308837876,
            0.0,
            0.0,
            0.0,
            -82355.86732027541,
            -25843.06175439942,
            0.0,
            0.0,
            0.0,
            229284.04542668918,
            117410.37432997991,
            0.0,
            0.0,
            0.0,
            -375818.0132734753,
            -386247.80765802023,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -3.7385208695213668,
            0.25294420589064653,
            1.0,
            0.0,
            194.5554531509207,  # UnEq Spin
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 0 of the intermediate phase (ind 4)
            0.0,  # No Spin
            0.9951733419499662,
            101.21991715215253,
            632.4731389009143,
            0.0,
            0.0,
            0.00016803066316882238,
            0.11412314719189287,
            1.8413983770369362,
            1.0,
            18.694178521101332,  # Eq spin
            16.89845522539974,
            0.3612417066833153,
            0.0,
            0.0,
            -697.6773920613674,
            0.0,
            -147.53381808989846,
            0.0,
            0.0,
            0.0,
            4941.31613710257,
            0.0,
            0.0,
            0.0,
            3531.552143264721,
            -14302.70838220423,
            178.85850322465944,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.965640445745779,
            -2.7706595614504725,
            1.0,
            0.0,
            0.0,  # UnEq Spin
            356.74395864902294,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1693.326644293169,
            0.0,
        ],
        [  # Coeffs collocation point 1 of the intermediate phase (ind 5)
            0.0,  # No Spin
            -5.126358906504587,
            -227.46830225846668,
            688.3609087244353,
            -751.4184178636324,
            0.0,
            -0.004551938711031158,
            -0.7811680872741462,
            1.0,
            0.0,
            0.1549280856660919,  # Eq spin
            -0.9539250460041732,
            -2.84311102369862,
            0.0,
            0.0,
            73.79645135116367,
            0.0,
            -8.13494176717772,
            0.0,
            0.0,
            0.0,
            -539.4071941841604,
            0.0,
            0.0,
            0.0,
            -936.3740515136005,
            1862.9097047992134,
            224.77581754671272,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.5308507364054487,
            1.0,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            0.0,
            0.0,
            0.0,
            2993.3598520496153,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 1 of the intermediate phase (ind 6)
            -82.54500000000004,  # No Spin
            -5.58197349185435e6,
            -3.5225742421184325e8,
            1.4667258334378073e9,
            0.0,  #
            0.0,  #
            1.0,
            66757.12830903867,
            5.385164380400193e6,
            2.5176585751772933e6,
            19.416719811164853,  # Eq spin
            -36.066611959079935,
            -0.8612656616290079,
            5.95010003393006,
            4.984750041013893,
            207.69898051583655,
            -132.88417400679026,
            -17.671713040498304,
            29.071788188638315,
            37.462217031512786,
            170.97203068800542,
            -107.41099349364234,
            0.0,
            -647.8103976942541,
            0.0,
            -1365.1499998427248,
            1152.425940764218,
            415.7134909564443,
            1897.5444343138167,
            -866.283566780576,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.1492259468169692,
            1.0,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            7343.130973149263,
            -20486.813161100774,
            0.0,
            0.0,
            0.0,
            515.9898508588834,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 2 of the intermediate phase (ind 7)
            0.4248820426833804,  # No Spin
            -906.746595921514,
            -282820.39946006844,
            -967049.2793750163,
            670077.5414916876,
            0.0,
            1.0,
            1670.9440812294847,
            19783.077247023448,
            0.0,
            0.22814271667259703,  # Eq spin
            1.1366593671801855,
            0.4818323187946999,
            0.0,
            0.0,
            12.840649528989287,
            0.0,
            -61.17248283184154,
            0.0,
            0.0,
            -711.8532052499075,
            269.9234918621958,
            941.6974723887743,
            0.0,
            0.0,
            3499.432393555856,
            -877.8811492839261,
            -4974.189172654984,
            0.0,
            0.0,
            -4939.642457025497,
            -227.7672020783411,
            8745.201037897836,
            0.0,
            0.0,
            -1.2442293719740283,
            1.0,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            -514.8494071830514,
            1493.3851099678195,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 0 of the merger ringdown phase (ind 8)
            0.0,  # No spin
            0.7207992174994245,
            -1.237332073800276,
            6.086871214811216,
            0.0,
            0.0,
            0.006851189888541745,
            0.06099184229137391,
            -0.15500218299268662,
            1.0,
            0.06519048552628343,  # Eq spin
            0.0,
            0.20035146870472367,
            0.0,
            -0.2697933899920511,
            -25.25397971063995,
            -5.215945111216946,
            -0.28745205203100666,
            5.7756520242745735,
            +4.917070939324979,
            +58.59408241189781,
            +153.95945758807616,
            0.0,
            -43.97332874253772,
            -11.61488280763592,
            +160.14971486043524,
            -693.0504179144295,
            0.0,
            0.0,
            0.0,
            -308.62513664956975,
            +835.1725103648205,
            -47.56042058800358,
            +338.7263666984089,
            -22.384949087140086,
            1.0,
            -0.6628745847248266,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            -23.504907495268824,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 1 of the merger ringdown phase (ind 9)
            0.0,  # No spin
            -9.460253118496386,
            +9.429314399633007,
            +64.69109972468395,
            0.0,
            0.0,
            -0.0670554310666559,
            -0.09987544893382533,
            1.0,
            0.0,
            0.0,  # Eq spin
            0.0,
            0.04497628581617564,
            0.0,
            0.0,
            17.36495157980372,
            0.0,
            0.0,
            0.0,
            0.0,
            -191.00932194869588,
            -62.997389062600035,
            +64.42947340363101,
            0.0,
            0.0,
            930.3458437154668,
            +808.457330742532,
            0.0,
            0.0,
            0.0,
            -774.3633787391745,
            -2177.554979351284,
            -1031.846477275069,
            0.0,
            0.0,
            1.0,
            -0.7267610313751913,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            -36.66374091965371,
            91.60477826830407,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 2 of the merger ringdown phase (ind 10)
            0.0,  # No spin
            -8.506898502692536,
            +13.936621412517798,
            0.0,
            0.0,
            0.0,
            -0.40919671232073945,
            1.0,
            0.0,
            0.0,
            0.0,  # Eq spin
            0.046849371468156265,
            0.0,
            0.0,
            0.0,
            1.7280582989361533,
            0.0,
            18.41570325463385,
            -13.743271480938104,
            0.0,
            73.8367329022058,
            0.0,
            -95.57802408341716,
            +215.78111099820157,
            0.0,
            -27.976989112929353,
            +6.404060932334562,
            +109.04824706217418,
            -633.1966645925428,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -0.6862449113932192,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            0.0,
            0.0,
            641.8965762829259,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 3 of the merger ringdown phase (ind 11)
            -85.86062966719405,  # No spin
            -4616.740713893726,
            -4925.756920247186,
            +7732.064464348168,
            +12828.269960300782,
            -39783.51698102803,
            1.0,
            +50.206318806624004,
            0.0,
            0.0,
            33.335857451144356,  # Eq spin
            -36.49019206094966,
            -3.835967351280833,
            2.302712009652155,
            1.6533417657003922,
            -69.19412903018717,
            26.580344399838758,
            -15.399770764623746,
            31.231253209893488,
            97.69027029734173,
            93.64156367505917,
            -18.184492163348665,
            423.48863373726243,
            -104.36120236420928,
            -719.8775484010988,
            0.0,
            1497.3545918387515,
            -101.72731770500685,
            0.0,
            0.0,
            1075.8686153198323,
            -3443.0233614187396,
            -4253.974688619423,
            -608.2901586790335,
            5064.173605639933,
            -1.3705601055555852,
            1.0,
            0.0,
            0.0,
            22.363215261437862,  # UnEq Spin
            156.08206945239374,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [  # Coeffs collocation point 4 of the merger ringdown phase (ind 12)
            0.0,  # No spin
            7.05731400277692,
            22.455288821807095,
            119.43820622871043,
            0.0,
            0.0,
            0.26026709603623255,
            1.0,
            0.0,
            0.0,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            0.0,
            -7.9407123129681425,
            9.486783128047414,
            0.0,
            0.0,
            0.0,
            134.88158268621922,
            -56.05992404859163,
            0.0,
            0.0,
            0.0,
            -316.26970506215554,
            90.31815139272628,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -0.7162058321905909,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            43.82713604567481,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    ]
)

# Retain the exact tensors historically exposed above as private production
# masters. Public names and getters return independent storage so external
# mutation or autograd metadata can never poison an identity-qualified cache.
_PHENOMX_AMP_COEFF_TABLE_CPU_MASTER = PhenomX_amp_coeff_table
_PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER = PhenomX_phase_coeff_table
PhenomX_amp_coeff_table = (
    _PHENOMX_AMP_COEFF_TABLE_CPU_MASTER.detach().clone(
        memory_format=torch.preserve_format
    )
)
PhenomX_phase_coeff_table = (
    _PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER.detach().clone(
        memory_format=torch.preserve_format
    )
)


@lru_cache(maxsize=None)
def _cached_phenomx_phase_coeff_table(
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the immutable phase-fit table on one device and dtype."""

    return _PHENOMX_PHASE_COEFF_TABLE_CPU_MASTER.to(
        device=device,
        dtype=dtype,
    )


@lru_cache(maxsize=None)
def _cached_phenomx_amp_coeff_table(
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the immutable amplitude-fit table on one device and dtype."""

    return _PHENOMX_AMP_COEFF_TABLE_CPU_MASTER.to(
        device=device,
        dtype=dtype,
    )


def _get_phenomx_phase_coeff_table_cached_master(
    *, device, dtype
) -> torch.Tensor:
    """Return the private cached phase table used by production kernels."""

    return _cached_phenomx_phase_coeff_table(torch.device(device), dtype)


def _get_phenomx_amp_coeff_table_cached_master(
    *, device, dtype
) -> torch.Tensor:
    """Return the private cached amplitude table used by production kernels."""

    return _cached_phenomx_amp_coeff_table(torch.device(device), dtype)


def get_phenomx_phase_coeff_table(*, device, dtype) -> torch.Tensor:
    """Return an independent phase-fit table for ``device`` and ``dtype``."""

    return _get_phenomx_phase_coeff_table_cached_master(
        device=device,
        dtype=dtype,
    ).detach().clone(memory_format=torch.preserve_format)


def get_phenomx_amp_coeff_table(*, device, dtype) -> torch.Tensor:
    """Return an independent amplitude-fit table for ``device`` and ``dtype``."""

    return _get_phenomx_amp_coeff_table_cached_master(
        device=device,
        dtype=dtype,
    ).detach().clone(memory_format=torch.preserve_format)


def _clear_phenomx_coeff_table_cache() -> None:
    """Release cached coefficient tensors, primarily for device-level tests."""

    _cached_phenomx_phase_coeff_table.cache_clear()
    _cached_phenomx_amp_coeff_table.cache_clear()
