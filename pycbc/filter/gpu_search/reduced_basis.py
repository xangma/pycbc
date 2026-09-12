# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
[EXPERIMENTAL PROTOTYPE - Milestone M7]
Reduced-basis matched filtering for the PyCBC persistent GPU search engine.
This module is an experimental research prototype not yet qualified for
production pipeline execution.

Approximates groups of physically similar templates using a low-rank orthonormal
basis via Singular Value Decomposition (SVD). Filters only R basis waveforms
via IFFT, then reconstructs full template responses using high-throughput
matrix multiplication (GEMM). This lossy route requires explicit opt-in;
default search delegates to the dense engine using the retained originals.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import logging
from typing import Any, List, Optional, Tuple
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pycbc.filter.matchedfilter import get_cutoff_indices
from pycbc.types import FrequencySeries
from .plans import BankGeometry, prepare_bank, bind_psd
from .candidates import SelectionPolicy, CandidateBuffer, select_tile_candidates
from .engine import SearchEngine, Ticket
from .vetoes import VetoManager

logger = logging.getLogger("pycbc.filter.gpu_search.reduced_basis")


def _as_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _move(value, device):
    if torch is not None and str(device) != "numpy":
        return torch.as_tensor(value, device=device)
    return _as_numpy(value).copy()


def _fingerprint(*values):
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, str):
            digest.update(value.encode("utf8"))
        else:
            arr = np.ascontiguousarray(_as_numpy(value))
            digest.update(str((arr.dtype.str, arr.shape)).encode("ascii"))
            digest.update(arr.tobytes())
    return digest.hexdigest()


@dataclass
class ReducedBasisPlan:
    """Basis and owned original complex64 filter samples, before truncation.

    ``tolerance`` selects aggregate unweighted SVD energy only. It does not
    certify individual PSD-weighted errors or floating-point decisions.
    Original samples and the experimental full response workspace cost O(M N).
    """

    rank: int
    num_templates: int
    basis_data: Any
    coefficients: Any
    singular_values: Any
    template_ids: np.ndarray
    geometry: BankGeometry
    tolerance: float
    device: str
    original_data: Any
    template_metadata: Any
    unweighted_relative_error: float
    svd_tolerance_met: bool

    @property
    def content_hash(self):
        """Fingerprint current samples, approximation and numerical settings."""
        return _fingerprint(
            self.original_data, self.basis_data, self.coefficients,
            self.template_ids, repr(self.geometry), repr(self.tolerance),
        )

    def to(self, device: str) -> "ReducedBasisPlan":
        """Move all retained samples and factors, including NumPy transitions."""
        return replace(
            self, device=str(device),
            original_data=_move(self.original_data, device),
            basis_data=_move(self.basis_data, device),
            coefficients=_move(self.coefficients, device),
            singular_values=_move(self.singular_values, device),
            template_ids=self.template_ids.copy(),
            template_metadata=deepcopy(self.template_metadata),
        )


def _original_bank(plan, device):
    templates = []
    for row, tid, metadata in zip(
        _as_numpy(plan.original_data), plan.template_ids, plan.template_metadata,
    ):
        template = FrequencySeries(row.copy(), delta_f=plan.geometry.delta_f)
        for key, value in metadata.items():
            setattr(template, key, deepcopy(value))
        template.id = int(tid)
        templates.append(template)
    return prepare_bank(
        templates, tile_size=plan.num_templates, device=device,
        f_lower=plan.geometry.f_lower, f_upper=plan.geometry.f_upper,
    )


@dataclass
class ReducedBasisPSDPlan:
    """Original normalization and PSD-bound residual diagnostics.

    ``residual_sigmasqs`` and ``relative_error`` use float64 arithmetic.
    The exact-arithmetic SNR error estimate is D * relative_error, where
    D is the PSD-weighted data norm. Rounding and reference-backend errors
    are not included: ``certifies_reference_decisions`` is always False.
    """

    psd_data: Any
    tile_norms: Any
    tile_sigmasqs: Any
    psd_version: str
    device: str
    residual_sigmasqs: Any
    relative_error: Any
    tolerance_met: Any
    plan_content_hash: str
    psd_content_hash: str
    version_hash: str
    reference_psd: Any
    certifies_reference_decisions: bool = False

    def validate(self, plan):
        """Reject stale bindings, including reused user version identifiers."""
        if plan.content_hash != self.plan_content_hash:
            raise ValueError("Reduced-basis plan changed; bind the PSD again")
        if _fingerprint(self.psd_data) != self.psd_content_hash:
            raise ValueError("Reduced-basis PSD changed; bind the PSD again")

    def to(self, device: str) -> "ReducedBasisPSDPlan":
        norms = _move(self.tile_norms, device)
        sigmasqs = _move(self.tile_sigmasqs, device)
        psd_data = _move(self.psd_data, device)
        reference_psd = replace(
            self.reference_psd, psd_data=psd_data,
            tile_norms={0: norms}, tile_sigmasqs={0: sigmasqs},
        )
        return replace(
            self, device=str(device), psd_data=psd_data,
            tile_norms=norms, tile_sigmasqs=sigmasqs,
            residual_sigmasqs=_move(self.residual_sigmasqs, device),
            relative_error=_move(self.relative_error, device),
            tolerance_met=_move(self.tolerance_met, device),
            reference_psd=reference_psd,
        )


def compute_reduced_basis(
    templates: List[Any],
    max_rank: Optional[int] = None,
    tolerance: float = 1e-3,
    f_lower: Optional[float] = None,
    f_upper: Optional[float] = None,
    device: str = "cpu",
) -> ReducedBasisPlan:
    """
    Compute an orthonormal reduced basis for a group of templates via SVD.

    Parameters
    ----------
    templates : List[FrequencySeries]
        Templates to compress into a reduced basis.
    max_rank : int, optional
        Maximum number of basis vectors to retain.
    tolerance : float, optional
        Target fractional energy truncation tolerance (default 1e-3).
    f_lower : float, optional
        Low-frequency cutoff (Hz).
    f_upper : float, optional
        High-frequency cutoff (Hz).
    device : str, optional
        Target device ('cpu', 'cuda', etc.).

    Returns
    -------
    ReducedBasisPlan
        Reduced basis with orthonormal filters and expansion coefficients.
    """
    if not templates:
        raise ValueError("templates list cannot be empty")
    if not np.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("tolerance must be finite and in [0, 1)")
    if max_rank is not None and (int(max_rank) != max_rank or max_rank < 1):
        raise ValueError("max_rank must be a positive integer")

    num_templates = len(templates)
    first = templates[0]
    delta_f = float(first.delta_f)
    flen = len(first)
    tlen = (flen - 1) * 2
    fs = tlen * delta_f
    if flen < 2 or not np.isfinite(delta_f) or delta_f <= 0:
        raise ValueError("Templates require a positive delta_f and length >= 2")
    delta_t = 1.0 / fs

    # Own the same complex64 filter samples as prepare_bank, before truncation.
    h_rows = []
    t_ids = []
    for i, t in enumerate(templates):
        if len(t) != flen or float(t.delta_f) != delta_f:
            raise ValueError("Templates must share frequency length and delta_f")
        data_np = _as_numpy(t)
        if not np.all(np.isfinite(data_np)):
            raise ValueError("Templates must contain finite samples")
        h_rows.append(data_np.astype(np.complex64))
        t_ids.append(getattr(t, "id", i))

    h_matrix_np = np.stack(h_rows, axis=0)
    template_ids = np.array(t_ids, dtype=np.int64)

    if torch is not None and str(device) != "numpy":
        dev = torch.device(device)
        h_tensor = torch.from_numpy(h_matrix_np).to(device=dev)
        u, s, vh = torch.linalg.svd(h_tensor, full_matrices=False)

        singular_np = s.detach().cpu().numpy().astype(np.float64)
    else:
        u, s, vh = np.linalg.svd(h_matrix_np, full_matrices=False)
        singular_np = s.astype(np.float64)

    # Reverse summation avoids losing a small tail by subtracting near one.
    energy = singular_np ** 2
    total_energy = energy.sum()
    tail = np.r_[np.cumsum(energy[::-1])[::-1][1:], 0.0]
    targets = np.flatnonzero(tail <= tolerance ** 2 * total_energy)
    target_r = int(targets[0] + 1) if len(targets) else len(singular_np)
    if max_rank is not None:
        target_r = min(target_r, int(max_rank))
    basis_data = vh[:target_r, :]
    coeffs = u[:, :target_r] * s[:target_r][None, :]
    sing_vals = s
    relative_error = np.sqrt(tail[target_r - 1] / total_energy) if total_energy else 0.0

    geom = BankGeometry(
        delta_f=delta_f,
        filter_length=flen,
        transform_length=tlen,
        sample_rate=fs,
        delta_t=delta_t,
        f_lower=f_lower if f_lower is not None else getattr(first, "f_lower", None),
        f_upper=f_upper if f_upper is not None else getattr(first, "f_upper", None),
    )

    return ReducedBasisPlan(
        rank=target_r,
        num_templates=num_templates,
        basis_data=basis_data,
        coefficients=coeffs,
        singular_values=sing_vals,
        template_ids=template_ids,
        geometry=geom,
        tolerance=float(tolerance),
        device=str(device),
        original_data=_move(h_matrix_np.copy(), device),
        template_metadata=tuple(
            {key: deepcopy(value) for key, value in vars(t).items()
             if not key.startswith("_") or key == "_epoch"}
            for t in templates
        ),
        unweighted_relative_error=float(relative_error),
        svd_tolerance_met=bool(relative_error <= tolerance),
    )


def bind_reduced_basis_psd(
    plan: ReducedBasisPlan,
    psd: Any,
    psd_version: str = "psd-v1",
    device: Optional[str] = None,
) -> ReducedBasisPSDPlan:
    """Bind original norms and per-template PSD-weighted residual estimates.

    The basis is chosen without PSD weighting; a max-rank cap or changed PSD
    can violate the requested tolerance. Report every violation. Reconstruct
    one template at a time to avoid an extra full-bank residual allocation.
    """
    device = plan.device if device is None else str(device)
    if hasattr(psd, "delta_f") and float(psd.delta_f) != plan.geometry.delta_f:
        raise ValueError("PSD delta_f does not match reduced-basis plan")
    flen = plan.geometry.filter_length
    raw_psd = _as_numpy(psd)
    if raw_psd.ndim != 1 or len(raw_psd) < flen:
        raise ValueError("PSD must cover the full frequency grid")
    if np.any(~np.isfinite(raw_psd[:flen])) or np.any(raw_psd[:flen] < 0):
        raise ValueError("PSD must contain finite, nonnegative samples")

    # Share the dense engine's original-template normalization convention,
    # including dynamic range handling, before evaluating truncation error.
    bank = _original_bank(plan, device)
    psd_snapshot = FrequencySeries(raw_psd.copy(), delta_f=plan.geometry.delta_f)
    if hasattr(psd, "dyn_range_factor"):
        psd_snapshot.dyn_range_factor = psd.dyn_range_factor
    reference_psd = bind_psd(
        bank, psd_snapshot, psd_version=psd_version, device=device,
    )
    psd_np = _as_numpy(reference_psd.psd_data)
    kmin, kmax = get_cutoff_indices(
        plan.geometry.f_lower, plan.geometry.f_upper,
        plan.geometry.delta_f, plan.geometry.transform_length,
    )
    active_psd = psd_np[kmin:kmax].astype(np.float64)
    weights = np.zeros_like(active_psd)
    np.divide(4 * plan.geometry.delta_f, active_psd, out=weights,
              where=active_psd > 0)
    basis = _as_numpy(plan.basis_data)[:, kmin:kmax].astype(np.complex128)
    coeffs = _as_numpy(plan.coefficients).astype(np.complex128)
    originals = _as_numpy(plan.original_data)
    residuals = np.empty(plan.num_templates, dtype=np.float64)
    original_norms = np.empty_like(residuals)
    for i, original in enumerate(originals):
        original = original[kmin:kmax].astype(np.complex128)
        error = original - coeffs[i] @ basis
        residuals[i] = np.sum(np.abs(error) ** 2 * weights)
        original_norms[i] = np.sum(np.abs(original) ** 2 * weights)
    ratios = np.full_like(residuals, np.inf)
    np.divide(residuals, original_norms, out=ratios, where=original_norms > 0)
    ratios = np.sqrt(ratios)
    tolerance_met = np.isfinite(ratios) & (ratios <= plan.tolerance)
    plan_hash = plan.content_hash
    psd_hash = _fingerprint(reference_psd.psd_data)
    return ReducedBasisPSDPlan(
        psd_data=reference_psd.psd_data,
        tile_norms=reference_psd.tile_norms[0],
        tile_sigmasqs=reference_psd.tile_sigmasqs[0],
        psd_version=psd_version, device=device,
        residual_sigmasqs=_move(residuals, device),
        relative_error=_move(ratios, device),
        tolerance_met=_move(tolerance_met, device),
        plan_content_hash=plan_hash, psd_content_hash=psd_hash,
        version_hash=_fingerprint(plan_hash, psd_hash),
        reference_psd=reference_psd,
    )


class ReducedBasisSearchEngine:
    """
    Dense original-template search by default; opt-in low-rank experiments.

    No pre-discard bound currently certifies floating-point reference decisions.
    Default submissions therefore use SearchEngine on retained originals.
    experimental_approximation=True permits lossy GEMM reconstruction and may
    change thresholds, clustering, aborts, vetoes and accepted identities.
    """

    def __init__(
        self,
        basis_plan: ReducedBasisPlan,
        selection_policy: SelectionPolicy,
        veto_manager: Optional[Any] = None,
        candidate_capacity: int = 65536,
        device: str = "cpu",
        experimental_approximation: bool = False,
    ):
        self.basis_plan = basis_plan.to(device)
        basis_plan = self.basis_plan
        self._plan_content_hash = basis_plan.content_hash
        self.experimental_approximation = bool(experimental_approximation)
        self.fallback_reason = (
            None if self.experimental_approximation
            else "Reduced-basis reference decisions are uncertified"
        )
        self.selection_policy = selection_policy
        if veto_manager is not None and hasattr(veto_manager, "tile_bin_edges"):
            self.veto_manager = VetoManager(power_chisq_plan=veto_manager)
        else:
            self.veto_manager = veto_manager
        self.candidate_capacity = int(candidate_capacity)
        self.device = str(device)

        self.r = basis_plan.rank
        self.m = basis_plan.num_templates
        self.flen = basis_plan.geometry.filter_length
        self.tlen = basis_plan.geometry.transform_length

        self._reference_engine = None
        if self.experimental_approximation:
            self._allocate_workspaces()
        else:
            self._reference_engine = SearchEngine(
                _original_bank(basis_plan, self.device), selection_policy,
                veto_manager=self.veto_manager,
                candidate_capacity=self.candidate_capacity, device=self.device,
            )
        self._ticket_counter = 0

    @property
    def cout_workspace(self):
        if self._reference_engine is not None:
            return self._reference_engine.cout_workspace
        return self.basis_cout

    @property
    def out_workspace(self):
        if self._reference_engine is not None:
            return self._reference_engine.out_workspace
        return self.basis_out

    def _allocate_workspaces(self):
        """Allocate bounded basis and reconstructed correlation workspaces."""
        if torch is not None and self.device != "numpy":
            dev = torch.device(self.device)
            # Basis workspaces: shape (R, tlen)
            self.basis_cout = torch.zeros((self.r, self.tlen), dtype=torch.complex64, device=dev)
            self.basis_out = torch.zeros((self.r, self.tlen), dtype=torch.complex64, device=dev)
            # Reconstructed workspace: shape (M, tlen)
            self.q_reconstructed = torch.zeros((self.m, self.tlen), dtype=torch.complex64, device=dev)
        else:
            self.basis_cout = np.zeros((self.r, self.tlen), dtype=np.complex64)
            self.basis_out = np.zeros((self.r, self.tlen), dtype=np.complex64)
            self.q_reconstructed = np.zeros((self.m, self.tlen), dtype=np.complex64)

        self.candidate_buffer = CandidateBuffer(
            capacity=self.candidate_capacity, device=self.device
        )

    def submit(
        self,
        data_block: Any,
        psd_plan: ReducedBasisPSDPlan,
        valid_interval: Tuple[int, int],
        block_id: int = 0,
    ) -> Ticket:
        """
        Submit a data block for reduced-basis filtering and reconstruction.
        """
        psd_plan.validate(self.basis_plan)
        if self.basis_plan.content_hash != self._plan_content_hash:
            raise ValueError("Reduced-basis plan changed; create a new engine")
        if self._reference_engine is not None:
            return self._reference_engine.submit(
                data_block, psd_plan.reference_psd, valid_interval, block_id,
            )
        self._ticket_counter += 1
        ticket = Ticket(
            ticket_id=self._ticket_counter,
            block_id=block_id,
            psd_version=psd_plan.psd_version,
            valid_interval=valid_interval,
        )

        valid_start, valid_end = valid_interval
        self.candidate_buffer.reset()

        # Prepare overwhitened data
        if torch is not None and self.device != "numpy":
            dev = torch.device(self.device)
            if isinstance(data_block, torch.Tensor):
                data_tensor = data_block.to(device=dev, dtype=torch.complex64)
            elif hasattr(data_block, "data") and isinstance(data_block.data, torch.Tensor):
                data_tensor = data_block.data.to(device=dev, dtype=torch.complex64)
            else:
                data_np = data_block.numpy() if hasattr(data_block, "numpy") else np.asarray(data_block)
                data_tensor = torch.as_tensor(data_np[:self.flen], device=dev, dtype=torch.complex64)

            psd_tensor = torch.as_tensor(psd_plan.psd_data[:self.flen], device=dev, dtype=torch.float32)
            inv_psd = torch.where(psd_tensor > 0, 1.0 / psd_tensor, torch.zeros_like(psd_tensor))
            stilde = data_tensor[:self.flen] * inv_psd[:self.flen]

            kmin, kmax = get_cutoff_indices(
                self.basis_plan.geometry.f_lower,
                self.basis_plan.geometry.f_upper,
                self.basis_plan.geometry.delta_f,
                self.tlen,
            )
            stilde[:kmin].zero_()
            if kmax < len(stilde):
                stilde[kmax:].zero_()

            # 1. Filter the R basis waveforms
            torch.mul(
                torch.conj(self.basis_plan.basis_data),
                stilde.unsqueeze(0),
                out=self.basis_cout[:, :self.flen],
            )
            if self.flen < self.tlen:
                self.basis_cout[:, self.flen:].zero_()

            torch.fft.ifft(
                self.basis_cout,
                n=self.tlen,
                dim=-1,
                norm="forward",
                out=self.basis_out,
            )

            # 2. Reconstruct M template responses via GEMM: Q = A.conj() @ y
            torch.matmul(
                torch.conj(self.basis_plan.coefficients),
                self.basis_out,
                out=self.q_reconstructed,
            )

            # 3. Candidate selection on reconstructed responses
            sel = select_tile_candidates(
                self.q_reconstructed,
                psd_plan.tile_norms,
                psd_plan.tile_sigmasqs,
                valid_start,
                valid_end,
                self.selection_policy,
                buffer=self.candidate_buffer,
            )
        else:
            data_np = data_block.numpy() if hasattr(data_block, "numpy") else np.asarray(data_block)
            psd_np = psd_plan.psd_data
            with np.errstate(divide="ignore", invalid="ignore"):
                stilde_np = np.where(psd_np > 0, data_np[:self.flen] / psd_np[:self.flen], 0.0).astype(np.complex64)

            kmin, kmax = get_cutoff_indices(
                self.basis_plan.geometry.f_lower,
                self.basis_plan.geometry.f_upper,
                self.basis_plan.geometry.delta_f,
                self.tlen,
            )
            stilde_np[:kmin] = 0.0
            if kmax < len(stilde_np):
                stilde_np[kmax:] = 0.0

            self.basis_cout[:, :self.flen] = (
                np.conj(self.basis_plan.basis_data) * stilde_np[np.newaxis, :]
            )
            if self.flen < self.tlen:
                self.basis_cout[:, self.flen:] = 0.0

            self.basis_out = np.fft.ifft(self.basis_cout, n=self.tlen, axis=-1) * self.tlen

            # GEMM
            self.q_reconstructed = np.matmul(np.conj(self.basis_plan.coefficients), self.basis_out)

            sel = select_tile_candidates(
                self.q_reconstructed,
                psd_plan.tile_norms,
                psd_plan.tile_sigmasqs,
                valid_start,
                valid_end,
                self.selection_policy,
                buffer=self.candidate_buffer,
            )

        if sel.get("aborted", False):
            ticket.aborted = True
            ticket.completed = True
            return ticket

        if sel.get("overflow", False):
            ticket.overflow = True
            ticket.completed = True
            return ticket

        cands = sel.get("candidates", {})
        if len(cands.get("template_idx", [])) > 0:
            if self.veto_manager is not None:
                if torch is not None and self.device != "numpy":
                    corr_freq = torch.matmul(
                        torch.conj(self.basis_plan.coefficients),
                        self.basis_cout[:, : self.flen],
                    )
                    corr_tile = torch.zeros(
                        (self.m, self.tlen),
                        dtype=torch.complex64,
                        device=corr_freq.device,
                    )
                    corr_tile[:, : self.flen] = corr_freq
                else:
                    corr_freq = np.matmul(
                        np.conj(self.basis_plan.coefficients),
                        self.basis_cout[:, : self.flen],
                    )
                    corr_tile = np.zeros(
                        (self.m, self.tlen), dtype=np.complex64
                    )
                    corr_tile[:, : self.flen] = corr_freq

                cands = self.veto_manager.evaluate(
                    corr_tile=corr_tile,
                    candidates=cands,
                    tile_id=0,
                    tile_norms=psd_plan.tile_norms,
                    transform_length=self.tlen,
                )

            global_tmplt_ids = self.basis_plan.template_ids[cands["template_idx"]]
            cands["template_id"] = global_tmplt_ids
            ticket.results = [cands]

        ticket.completed = True
        return ticket

    def drain(self) -> List[Ticket]:
        """Drain completed reference tickets."""
        if self._reference_engine is not None:
            return self._reference_engine.drain()
        return []

    def close(self):
        """Clean up buffers."""
        if self._reference_engine is not None:
            self._reference_engine.close()
        if hasattr(self, "basis_cout"):
            del self.basis_cout
        if hasattr(self, "basis_out"):
            del self.basis_out
        if hasattr(self, "q_reconstructed"):
            del self.q_reconstructed
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
