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
Bank, PSD, geometry and capability plans for the GPU search engine.
"""

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from pycbc.filter.matchedfilter import get_cutoff_indices

try:
    import torch
except ImportError:
    torch = None


@dataclass(frozen=True)
class BankGeometry:
    """Frequency and time grid geometry for a bank tile."""

    delta_f: float
    filter_length: int
    transform_length: int
    sample_rate: float
    delta_t: float
    f_lower: Optional[float] = None
    f_upper: Optional[float] = None
    dtype: str = "complex64"

    @classmethod
    def from_delta_f_and_length(
        cls,
        delta_f: float,
        filter_length: int,
        f_lower: Optional[float] = None,
        f_upper: Optional[float] = None,
        dtype: str = "complex64",
    ) -> "BankGeometry":
        transform_length = (filter_length - 1) * 2
        sample_rate = float(transform_length * delta_f)
        delta_t = 1.0 / sample_rate
        return cls(
            delta_f=float(delta_f),
            filter_length=int(filter_length),
            transform_length=int(transform_length),
            sample_rate=sample_rate,
            delta_t=delta_t,
            f_lower=f_lower,
            f_upper=f_upper,
            dtype=dtype,
        )


@dataclass
class TemplateTile:
    """An immutable, owned batch of templates with uniform geometry."""

    tile_id: int
    template_ids: Sequence[int]
    template_data: Any  # torch.Tensor or np.ndarray, shape: (B, filter_length)
    power_matrix: Any  # torch.Tensor or np.ndarray, shape: (B, filter_length)
    params: Dict[str, Any] = field(default_factory=dict)
    version_hash: str = ""
    templates: Optional[Sequence[Any]] = None
    template_by_id: Dict[Any, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return len(self.template_ids)


@dataclass
class BankPlan:
    """Versioned plan describing a partitioned template bank."""

    geometry: BankGeometry
    tiles: List[TemplateTile]
    num_templates: int
    version_hash: str
    approximant: Optional[str] = None
    waveform_policy: Optional[str] = None
    templates: Optional[Sequence[Any]] = None
    template_by_id: Dict[Any, Any] = field(default_factory=dict)

    def estimate_workspace_bytes(self, max_batch_size: Optional[int] = None) -> int:
        """Estimate workspace requirements in bytes."""
        b = max_batch_size or max(t.batch_size for t in self.tiles)
        n = self.geometry.transform_length
        return WorkspaceBudget.calculate_workspace_bytes(
            b, n, self.geometry.filter_length
        )


@dataclass
class PSDPlan:
    """Versioned binding of a Power Spectral Density to a BankPlan."""

    bank_version_hash: str
    psd_version: str
    psd_data: Any
    delta_f: float
    tile_sigmasqs: Dict[int, Any] = field(default_factory=dict)
    tile_norms: Dict[int, Any] = field(default_factory=dict)
    tile_chisq_bins: Optional[Dict[int, Any]] = None
    detector: Optional[str] = None
    psd: Optional[Any] = None
    dyn_range_factor: Optional[float] = None
    version_hash: str = ""


class WorkspaceBudget:
    """Calculator and manager for engine VRAM workspace budgets."""

    def __init__(self, max_workspace_bytes: Optional[int] = None):
        self.max_workspace_bytes = (
            int(max_workspace_bytes) if max_workspace_bytes is not None else None
        )

    @staticmethod
    def calculate_workspace_bytes(
        batch_size: int,
        transform_length: int,
        filter_length: int,
        candidate_capacity: int = 0,
        dtype_bytes: int = 8,  # complex64 = 8 bytes
    ) -> int:
        """
        Calculates required buffer memory.
        Correlation array: batch_size * transform_length * dtype_bytes
        Time-domain SNR / IFFT output: batch_size * transform_length * dtype_bytes
        Overwhitened data buffer: filter_length * dtype_bytes
        Candidate SoA scratch: 28 bytes per slot * candidate_capacity
        """
        corr_bytes = batch_size * transform_length * dtype_bytes
        snr_bytes = batch_size * transform_length * dtype_bytes
        overwhitened_bytes = filter_length * dtype_bytes
        candidate_bytes = candidate_capacity * 28
        return corr_bytes + snr_bytes + overwhitened_bytes + candidate_bytes

    @staticmethod
    def recommend_batch_size(
        available_vram_bytes: int,
        transform_length: int,
        filter_length: int,
        headroom_ratio: float = 0.5,
    ) -> int:
        """Recommend a safe batch size fitting within available VRAM."""
        budget = int(available_vram_bytes * headroom_ratio)
        bytes_per_template = 2 * transform_length * 8
        if bytes_per_template <= 0:
            return 1
        return max(1, budget // bytes_per_template)


def _compute_hash(data_bytes: bytes) -> str:
    return hashlib.sha256(data_bytes).hexdigest()[:16]


def prepare_bank(
    templates: Sequence[Any],
    tile_size: int = 64,
    f_lower: Optional[float] = None,
    f_upper: Optional[float] = None,
    device: str = "cpu",
    approximant: Optional[str] = None,
) -> BankPlan:
    """
    Partition templates into immutable tiles with uniform geometry and power matrices.
    """
    if not templates:
        raise ValueError("templates list cannot be empty")

    first_tmpl = templates[0]
    delta_f = float(first_tmpl.delta_f)
    flen = len(first_tmpl)

    geometry = BankGeometry.from_delta_f_and_length(
        delta_f=delta_f,
        filter_length=flen,
        f_lower=f_lower,
        f_upper=f_upper,
    )

    tiles = []
    tile_hashes = []
    num_templates = len(templates)

    bp_template_by_id = {getattr(t, "id", i): t for i, t in enumerate(templates)}

    for tile_idx, start_idx in enumerate(range(0, num_templates, tile_size)):
        chunk = templates[start_idx : start_idx + tile_size]
        t_ids = [getattr(t, "id", start_idx + i) for i, t in enumerate(chunk)]
        tile_template_by_id = {tid: t for tid, t in zip(t_ids, chunk)}

        data_list = []
        power_list = []
        for t in chunk:
            arr = getattr(t, "data", t)
            if hasattr(arr, "numpy"):
                np_arr = arr.numpy()
            elif hasattr(arr, "cpu"):
                np_arr = arr.cpu().numpy()
            else:
                np_arr = np.asarray(arr)

            data_list.append(np_arr.astype(np.complex64))
            pwr = (np_arr.real**2 + np_arr.imag**2).astype(np.float32)
            power_list.append(pwr)

        tile_data_np = np.stack(data_list, axis=0)
        power_matrix_np = np.stack(power_list, axis=0)

        geom_header = (
            f"{geometry.delta_f}:{geometry.filter_length}:"
            f"{geometry.transform_length}:{geometry.f_lower}:{geometry.f_upper}"
        ).encode("ascii")
        t_hash = _compute_hash(geom_header + b":" + tile_data_np.tobytes())
        tile_hashes.append(t_hash)

        if torch is not None and device != "numpy":
            tile_data = torch.from_numpy(tile_data_np).to(device=device)
            power_matrix = torch.from_numpy(power_matrix_np).to(device=device)
        else:
            tile_data = tile_data_np
            power_matrix = power_matrix_np

        tile = TemplateTile(
            tile_id=tile_idx,
            template_ids=t_ids,
            template_data=tile_data,
            power_matrix=power_matrix,
            version_hash=t_hash,
            templates=chunk,
            template_by_id=tile_template_by_id,
        )
        tiles.append(tile)

    geom_header = (
        f"{geometry.delta_f}:{geometry.filter_length}:"
        f"{geometry.transform_length}:{geometry.f_lower}:{geometry.f_upper}"
    ).encode("ascii")
    composite_hash = _compute_hash(geom_header + b":" + "".join(tile_hashes).encode("ascii"))

    return BankPlan(
        geometry=geometry,
        tiles=tiles,
        num_templates=num_templates,
        version_hash=composite_hash,
        approximant=approximant,
        templates=templates,
        template_by_id=bp_template_by_id,
    )


def bind_psd(
    bank_plan: BankPlan,
    psd: Any,
    psd_version: Optional[str] = None,
    detector: Optional[str] = None,
    device: str = "cpu",
    dyn_range_factor: Optional[float] = None,
) -> PSDPlan:
    """
    Bind a PSD to a BankPlan, computing sigmasqs and normalization factors.
    """
    if hasattr(psd, "numpy"):
        psd_np = psd.numpy()
    elif hasattr(psd, "cpu"):
        psd_np = psd.cpu().numpy()
    else:
        psd_np = np.asarray(psd)

    # Dynamic range scaling to avoid float32 underflow with physical PSDs
    max_val = float(np.max(psd_np)) if len(psd_np) > 0 else 0.0
    effective_dyn_range = dyn_range_factor or getattr(psd, "dyn_range_factor", None)

    if 0 < max_val < 1e-25:
        from pycbc import DYN_RANGE_FAC

        effective_dyn_range = effective_dyn_range or float(DYN_RANGE_FAC)
        psd_np = psd_np * (float(effective_dyn_range) ** 2)

    dyn_range_factor = effective_dyn_range

    psd_np = psd_np.astype(np.float32)
    flen = bank_plan.geometry.filter_length
    delta_f = bank_plan.geometry.delta_f

    if len(psd_np) > flen:
        psd_np = psd_np[:flen]
    elif len(psd_np) < flen:
        raise ValueError(
            f"PSD length {len(psd_np)} is shorter than filter length {flen}"
        )

    if psd_version is None:
        psd_version = _compute_hash(psd_np.tobytes())

    # inv_psd = (4 * delta_f) / psd
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_psd_np = np.where(psd_np > 0, (4.0 * delta_f) / psd_np, 0.0).astype(
            np.float32
        )

    # Zero out outside frequency cutoffs (DC is index 0, Nyquist is flen - 1)
    kmin, kmax = get_cutoff_indices(
        bank_plan.geometry.f_lower,
        bank_plan.geometry.f_upper,
        delta_f,
        bank_plan.geometry.transform_length,
    )
    inv_psd_np[:kmin] = 0.0
    if kmax < len(inv_psd_np):
        inv_psd_np[kmax:] = 0.0

    tile_sigmasqs = {}
    tile_norms = {}

    for tile in bank_plan.tiles:
        pm = tile.power_matrix
        if torch is not None and isinstance(pm, torch.Tensor):
            pm_np = pm.detach().cpu().numpy()
        else:
            pm_np = pm

        sigmasqs_np = pm_np.dot(inv_psd_np)
        with np.errstate(divide="ignore", invalid="ignore"):
            norms_np = np.where(
                sigmasqs_np > 0, (4.0 * delta_f) / np.sqrt(sigmasqs_np), 0.0
            )

        if torch is not None and device != "numpy":
            tile_sigmasqs[tile.tile_id] = torch.from_numpy(sigmasqs_np).to(
                device=device
            )
            tile_norms[tile.tile_id] = torch.from_numpy(norms_np).to(device=device)
        else:
            tile_sigmasqs[tile.tile_id] = sigmasqs_np
            tile_norms[tile.tile_id] = norms_np

    composite_version = _compute_hash(
        f"{bank_plan.version_hash}:{psd_version}:{detector}".encode("ascii")
    )

    if torch is not None and device != "numpy":
        psd_tensor = torch.from_numpy(psd_np).to(device=device)
    else:
        psd_tensor = psd_np

    return PSDPlan(
        bank_version_hash=bank_plan.version_hash,
        psd_version=psd_version,
        psd_data=psd_tensor,
        delta_f=delta_f,
        tile_sigmasqs=tile_sigmasqs,
        tile_norms=tile_norms,
        detector=detector,
        psd=psd,
        dyn_range_factor=effective_dyn_range,
        version_hash=composite_version,
    )
