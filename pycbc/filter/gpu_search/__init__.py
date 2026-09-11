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
PyCBC GPU Search Engine.

A persistent, tiled search engine for gravitational-wave template filtering,
candidate extraction, batched veto evaluation, and search pipeline adapters:
- Production Qualified (M1-M5): plans, candidates, vetoes, engine, graphs, adapter.
- Experimental Prototypes (M6-M7): multirate, reduced_basis, screening (research prototypes).
"""

from .plans import (
    BankGeometry,
    TemplateTile,
    BankPlan,
    PSDPlan,
    WorkspaceBudget,
    prepare_bank,
    bind_psd,
)
from .candidates import (
    CandidateBuffer,
    SelectionPolicy,
)
from .vetoes import (
    PowerChisqPlan,
    prepare_power_chisq_plan,
    batched_power_chisq,
    SineGaussianPlan,
    VetoManager,
)
from .engine import (
    SearchEngine,
    Ticket,
)
from .graphs import (
    CUDAGraphEntry,
    CUDAGraphManager,
)
from .adapter import (
    TiledMatchedFilterControl,
    TiledLiveBatchMatchedFilter,
)
from .multirate import (
    MultirateBand,
    MultiratePlan,
    prepare_multirate_plan,
    MultirateSearchEngine,
)
from .reduced_basis import (
    ReducedBasisPlan,
    ReducedBasisPSDPlan,
    compute_reduced_basis,
    bind_reduced_basis_psd,
    ReducedBasisSearchEngine,
)
from .core import (
    FilteringWorkspaceSlot,
    FilteringWorkspace,
    correlate_and_ifft,
)
from .screening import ConsistencyScreen

__all__ = [
    "FilteringWorkspaceSlot",
    "FilteringWorkspace",
    "correlate_and_ifft",
    "BankGeometry",
    "TemplateTile",
    "BankPlan",
    "PSDPlan",
    "WorkspaceBudget",
    "prepare_bank",
    "bind_psd",
    "CandidateBuffer",
    "SelectionPolicy",
    "PowerChisqPlan",
    "prepare_power_chisq_plan",
    "batched_power_chisq",
    "SineGaussianPlan",
    "VetoManager",
    "SearchEngine",
    "Ticket",
    "CUDAGraphEntry",
    "CUDAGraphManager",
    "TiledMatchedFilterControl",
    "TiledLiveBatchMatchedFilter",
    "MultirateBand",
    "MultiratePlan",
    "prepare_multirate_plan",
    "MultirateSearchEngine",
    "ReducedBasisPlan",
    "ReducedBasisPSDPlan",
    "compute_reduced_basis",
    "bind_reduced_basis_psd",
    "ReducedBasisSearchEngine",
    "ConsistencyScreen",
]
