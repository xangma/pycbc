"""Read frozen B1 evidence and compare selected rows to a complex128 DFT.

This diagnostic does not change campaign tolerances or qualify timings.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ROWS = [141, 144, 8, 0]
spec = importlib.util.spec_from_file_location("worker", ROOT / "batch-worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
worker.np = np
reference = json.loads((ROOT / "runs/qual-b1-branch_standard/result.json").read_text())
candidate = json.loads((ROOT / "runs/qual-b1-torch_cpu/result.json").read_text())
args = SimpleNamespace(**reference["arguments"])
bank, psd, sigma, blocks, geometry, injections, identity = worker.workload(args)
mismatched = {key: {"local": value, "frozen": reference["inputs"].get(key)}
              for key, value in identity.items() if value != reference["inputs"].get(key)}
assert not mismatched, mismatched
saved = np.load(ROOT / "validation/selected-reference-rows.npy", allow_pickle=False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
results = []
for index, row in enumerate(ROWS):
    original = saved[index]
    assert worker.digest_bytes(original) == reference["pointwise_blocks"][0]["rows"][row]["sha256"]
    exact_corr = np.zeros(args.size, np.complex128)
    exact_corr[:bank.shape[1]] = bank[row].astype(np.complex128).conj() * blocks[0].astype(np.complex128)
    exact = np.fft.ifft(exact_corr) * args.size
    corr = torch.zeros(args.size, dtype=torch.complex64)
    torch.mul(torch.from_numpy(bank[row]).conj(), torch.from_numpy(blocks[0]), out=corr[:bank.shape[1]])
    local = (torch.fft.ifft(corr) * args.size).numpy()
    fft_only = np.fft.ifft(corr.numpy().astype(np.complex128)) * args.size
    tol = worker.POLICY["pointwise_complex"]
    failing = np.abs(local.astype(np.complex128) - original) > tol["atol"] + tol["rtol"] * np.abs(original.astype(np.complex128))
    details = []
    for i in np.flatnonzero(failing)[:8]:
        details.append({"sample": int(i), "reference_magnitude": float(abs(original[i])),
                        "pairwise_error": float(abs(local[i] - original[i])),
                        "cpu_reference_error_vs_complex128": float(abs(original[i] - exact[i])),
                        "local_torch_error_vs_complex128": float(abs(local[i] - exact[i]))})
    results.append({"template_id": row + 1000,
                    "local_torch_matches_frozen_linux_sha256": worker.digest_bytes(local) == candidate["pointwise_blocks"][0]["rows"][row]["sha256"],
                    "cpu_reference_vs_complex128": worker.row_metrics(original, exact),
                    "local_torch_vs_complex128": worker.row_metrics(local, exact),
                    "local_torch_vs_same_float32_correlation_complex128_fft": worker.row_metrics(local, fft_only),
                    "local_torch_vs_cpu_reference": worker.row_metrics(local, original),
                    "selected_pairwise_failures": details})
out = {"purpose": __doc__, "source": worker.EXPECTED_HEAD,
       "worker_sha256": hashlib.sha256((ROOT / "batch-worker.py").read_bytes()).hexdigest(),
       "numpy": np.__version__, "torch": torch.__version__, "rows": results,
       "scope": "Selected block-zero rows only; local Torch may differ from Linux. No campaign pass or timing authorization."}
(ROOT / "validation/fft-precision-diagnostic.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"rows": [{"template_id": r["template_id"],
                            "linux_hash_match": r["local_torch_matches_frozen_linux_sha256"],
                            "reference_failures_vs_complex128": r["cpu_reference_vs_complex128"]["failed_samples"],
                            "local_torch_failures_vs_complex128": r["local_torch_vs_complex128"]["failed_samples"],
                            "pairwise_failures": r["selected_pairwise_failures"]} for r in results]}, indent=2))
