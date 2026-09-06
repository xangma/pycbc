#!/usr/bin/env python3
"""Untimed repeatability diagnostic for CUDA cumulative chi-squared bins."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    assert 1 <= args.repeats <= 1000
    assert not args.output.exists()
    status_path = args.capture / "status.json"
    status = json.loads(status_path.read_text())
    assert status["state"] == "complete"
    row = status["captures"][0]
    inputs = {str(p.resolve()): digest(p) for p in [Path(__file__), status_path]}
    arrays = {}
    for name in ["template", "psd", "bins"]:
        meta = row["arrays"][name]
        path = args.capture / Path(meta["path"]).name
        inputs[str(path.resolve())] = digest(path)
        assert inputs[str(path.resolve())] == meta["sha256"]
        arrays[name] = np.load(path, allow_pickle=False)
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    low = row["segment"]["filter_bin_start"]
    high = row["segment"]["filter_bin_stop"]
    h = torch.tensor(arrays["template"], device="cuda")
    psd = torch.tensor(arrays["psd"], device="cuda")
    power = ((h.real.square() + h.imag.square()) / psd)[low:high]
    truth = np.cumsum(power.cpu().numpy(), dtype=np.float64)
    num_bins = len(arrays["bins"]) - 1
    truth_bins = np.append(
        np.searchsorted(truth, np.arange(num_bins) * truth[-1] / num_bins, side="right")
        + low,
        high,
    )
    result = {
        "schema_version": 1,
        "scope": "Repeatability diagnostic only; same fixed CUDA float32 power vector in both routes.",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "input_sha256": inputs,
        "repeats": args.repeats,
        "truth_bins_float64": truth_bins.tolist(),
        "recorded_bins": arrays["bins"].tolist(),
        "routes": {},
    }
    for dtype in [torch.float32, torch.float64]:
        outcomes = Counter()
        totals = []
        for _ in range(args.repeats):
            cumulative = torch.cumsum(power, dim=0, dtype=dtype).to(torch.float32)
            thresholds = (
                torch.arange(num_bins, device="cuda", dtype=torch.float32)
                * cumulative[-1]
                / num_bins
            )
            bins = torch.searchsorted(cumulative, thresholds, right=True) + low
            edges = tuple(bins.cpu().tolist()) + (high,)
            outcomes[edges] += 1
            totals.append(float(cumulative[-1].item()))
        result["routes"][str(dtype)] = {
            "unique_edge_vectors": len(outcomes),
            "outcomes": [
                {"bins": list(edges), "count": count}
                for edges, count in outcomes.items()
            ],
            "cumulative_total_range": [min(totals), max(totals)],
        }
    result["input_sha256_after"] = {path: digest(path) for path in inputs}
    assert inputs == result["input_sha256_after"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["routes"], indent=2))


if __name__ == "__main__":
    main()
