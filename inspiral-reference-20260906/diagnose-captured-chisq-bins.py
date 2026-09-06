#!/usr/bin/env python3
"""Untimed causal bin-swap diagnostic for captured CPU and CUDA search data."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    helper = args.root / "diagnose-chisq-inputs.py"
    spec = importlib.util.spec_from_file_location("chisq_diagnostic", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = {str(p.resolve()): digest(p) for p in [helper, Path(__file__)]}
    arrays, rows, statuses = {}, {}, {}
    for name, serial in [("cpu", 3), ("cuda", 0)]:
        root = args.root / f"chisq-input-capture-{name}"
        path = root / "status.json"
        inputs[str(path.resolve())] = digest(path)
        status = json.loads(path.read_text())
        assert status["source_info"]["status"] == ""
        assert status["source_status_after"] == ""
        assert status["input_sha256"] == status["input_sha256_after"]
        statuses[name] = {
            key: status.get(key)
            for key in ["state", "returncode", "traceback", "source_info"]
        }
        row = status["captures"][serial]
        rows[name] = row
        arrays[name] = {}
        for key, meta in row["arrays"].items():
            path = root / Path(meta["path"]).name
            actual_hash = digest(path)
            assert actual_hash == meta["sha256"], path
            inputs[str(path.resolve())] = actual_hash
            arrays[name][key] = np.load(path, allow_pickle=False)
    assert statuses["cpu"]["source_info"] == statuses["cuda"]["source_info"]
    assert rows["cpu"]["template_hash"] == rows["cuda"]["template_hash"]
    assert rows["cpu"]["segment"] == rows["cuda"]["segment"]
    assert np.array_equal(arrays["cpu"]["indices"], arrays["cuda"]["indices"])
    output = {
        "schema_version": 1,
        "scope": "Causal diagnostic on one captured trigger; no qualification tolerances changed.",
        "capture_statuses": statuses,
        "input_sha256": inputs,
        "template_hash": rows["cpu"]["template_hash"],
        "segment": rows["cpu"]["segment"],
        "points": arrays["cpu"]["indices"].tolist(),
        "recorded_chisq": {
            name: value["chisq"].tolist() for name, value in arrays.items()
        },
        "direct_chisq_bin_swap": {},
        "binning": {},
        "active_band_input_differences": {},
    }
    low = rows["cpu"]["segment"]["filter_bin_start"]
    high = rows["cpu"]["segment"]["filter_bin_stop"]
    for input_name, data in arrays.items():
        norm = rows[input_name]["snr_norm"]
        num_bins = len(data["bins"]) - 1
        for bin_name, bin_data in arrays.items():
            sums = module.direct_bin_sums(
                data["corr"],
                data["indices"].astype(np.float64),
                bin_data["bins"],
            )
            chisq = (
                num_bins * np.sum(abs(sums) ** 2, axis=1)
                - abs(data["snrv"].astype(np.complex128)) ** 2
            ) * norm**2
            output["direct_chisq_bin_swap"][f"{input_name}_inputs_{bin_name}_bins"] = (
                chisq.tolist()
            )
        h = data["template"]
        power = ((h.real**2 + h.imag**2) / data["psd"])[low:high]
        assert np.all(np.isfinite(power)) and np.all(power >= 0)
        cumulative64 = np.cumsum(power, dtype=np.float64)
        methods = {
            "numpy_float32": np.cumsum(power),
            "numpy_float64": cumulative64,
            "numpy_float64_cast_float32": cumulative64.astype(np.float32),
        }
        output["binning"][input_name] = {
            "recorded_bins": data["bins"].tolist(),
            "methods": {},
        }
        for method, cumulative in methods.items():
            thresholds = np.arange(num_bins) * cumulative[-1] / num_bins
            bins = np.append(
                np.searchsorted(cumulative, thresholds, side="right") + low, high
            )
            # Measure each resulting bin against the independent double prefix.
            edges = np.append([0.0], cumulative64)[bins - low]
            bin_power = np.diff(edges)
            output["binning"][input_name]["methods"][method] = {
                "cumulative_total": float(cumulative[-1]),
                "total_relative_error": float(cumulative[-1] / cumulative64[-1] - 1),
                "bins": bins.tolist(),
                "equals_recorded": bool(np.array_equal(bins, data["bins"])),
                "relative_bin_power_vs_equal": (
                    bin_power / (cumulative64[-1] / num_bins) - 1
                ).tolist(),
            }
    for name in ["corr", "template", "psd"]:
        x = arrays["cpu"][name][low:high].astype(np.complex128)
        y = arrays["cuda"][name][low:high].astype(np.complex128)
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        nz = abs(x) > 0
        output["active_band_input_differences"][name] = {
            "relative_l2": float(np.linalg.norm(y - x) / np.linalg.norm(x)),
            "maximum_relative_error_nonzero_cpu": float(np.max(abs(y[nz] / x[nz] - 1))),
        }
    output["input_sha256_after"] = {path: digest(path) for path in inputs}
    assert inputs == output["input_sha256_after"]
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["direct_chisq_bin_swap"], indent=2))


if __name__ == "__main__":
    main()
