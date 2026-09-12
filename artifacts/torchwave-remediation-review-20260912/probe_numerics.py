"""Small CPU reproducers exercising the existing experimental functions."""
import json

import numpy as np
import torch

from pycbc.events.ranking import newsnr
from pycbc.filter.gpu_search.reduced_basis import (
    bind_reduced_basis_psd,
    compute_reduced_basis,
)
from pycbc.filter.gpu_search.screening import ConsistencyScreen
from pycbc.filter.gpu_search.vetoes import (
    PowerChisqPlan,
    VetoManager,
    batched_power_chisq,
)
from pycbc.types import FrequencySeries


results = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
           "screening": [], "reduced_basis": []}
n = 64
z = np.r_[np.full(8, (10 + np.sqrt(60)) / 16),
          np.full(8, (10 - np.sqrt(60)) / 16)].astype(np.complex64)
for device in ["numpy", "cpu"]:
    corr = np.zeros((1, n), dtype=np.complex64)
    corr[0, 1:17] = z
    edges = np.arange(1, 18, dtype=np.int64)[None, :]
    norms = np.ones(1, dtype=np.float32)
    if device == "cpu":
        corr, edges, norms = map(torch.tensor, (corr, edges, norms))
    candidates = {"template_idx": np.array([0]), "sample_idx": np.array([0]),
                  "snr": np.array([10 + 0j], dtype=np.complex64)}
    screen = ConsistencyScreen(device=device)
    score = screen.evaluate_screening_chisq(corr, candidates, norms, n, edges)
    chi, dof = batched_power_chisq(corr, candidates, edges, norms,
                                 transform_length=n)
    manager = VetoManager(
        power_chisq_plan=PowerChisqPlan(tile_bin_edges={0: edges}),
        consistency_screen=screen)
    filtered = manager.evaluate(corr, candidates, 0, norms, n)
    results["screening"].append({
        "device": device, "screen": score.tolist(), "chisq": chi.tolist(),
        "dof": dof.tolist(), "newsnr": np.asarray(newsnr(10, chi / dof)).tolist(),
        "survivors": len(filtered["sample_idx"]),
    })

h1, h2 = np.zeros((2, 17), dtype=np.complex64)
h1[1], h2[2] = 1, 0.5
templates = [FrequencySeries(h1, delta_f=1), FrequencySeries(h2, delta_f=1)]
for device in ["numpy", "cpu"]:
    plan = compute_reduced_basis(templates, max_rank=1, tolerance=1e-6,
                                 f_lower=1, f_upper=16, device=device)
    bound = bind_reduced_basis_psd(
        plan, FrequencySeries(np.ones(17, dtype=np.float32), delta_f=1),
        device=device)
    original = [float(4 * np.sum(np.abs(t.numpy()[1:16]) ** 2))
                for t in templates]
    results["reduced_basis"].append({
        "device": device, "rank": plan.rank, "stored_tolerance": plan.tolerance,
        "reported_sigmasqs": np.asarray(bound.tile_sigmasqs).tolist(),
        "original_sigmasqs": original,
    })
print(json.dumps(results, indent=2))
