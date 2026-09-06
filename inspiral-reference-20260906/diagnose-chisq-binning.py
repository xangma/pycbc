#!/usr/bin/env python3
"""Compare chi-squared algorithms on identical, recorded search inputs."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direct_bin_sums(corr, points, bins):
    result = np.zeros((len(points), len(bins) - 1), dtype=np.complex128)
    for column, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
        for first in range(int(start), int(end), 32768):
            last = min(first + 32768, int(end))
            frequency = np.arange(first, last, dtype=np.float64)
            phase = np.exp(
                2j * np.pi * points[:, None] * frequency[None, :] / len(corr)
            )
            result[:, column] += np.sum(corr[None, first:last] * phase, axis=1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-extension", type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    status_path = args.capture / "status.json"
    status = json.loads(status_path.read_text())
    inputs = {str(status_path.resolve()): digest(status_path)}
    if args.native_extension:
        spec = importlib.util.spec_from_file_location(
            "pycbc.vetoes.chisq_cpu", args.native_extension
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[spec.name] = module
    from pycbc.vetoes import chisq_cpu, chisq_torch
    from pycbc import scheme
    from pycbc.types import FrequencySeries

    for filename in [__file__, chisq_cpu.__file__, chisq_torch.__file__]:
        inputs[str(Path(filename).resolve())] = digest(filename)
    output = {
        "schema_version": 1,
        "scope": "Diagnostic algorithms on identical captured inputs; no qualification tolerances changed.",
        "capture_state": status["state"],
        "capture_returncode": status.get("returncode"),
        "capture_traceback": status.get("traceback"),
        "capture_source_info": status["source_info"],
        "input_sha256": inputs,
        "captures": [],
    }
    for row in status["captures"]:
        data = {}
        for name, meta in row["arrays"].items():
            path = args.capture / Path(meta["path"]).name
            actual_hash = digest(path)
            assert actual_hash == meta["sha256"], path
            inputs[str(path.resolve())] = actual_hash
            data[name] = np.load(path, allow_pickle=False)
        corr, points, bins = data["corr"], data["indices"], data["bins"]
        snr, norm = data["snrv"], row["snr_norm"]
        assert corr.dtype == np.complex64 and corr.ndim == 1
        assert points.ndim == 1 and np.all(points == np.rint(points))
        assert np.all((points >= 0) & (points < len(corr)))
        assert 0 <= bins[0] <= bins[-1] <= len(corr)
        assert np.all(np.diff(bins) >= 0)
        points = points.astype(np.float64)
        num_bins = len(bins) - 1
        direct = direct_bin_sums(corr, points, bins)
        expected_sum = np.sum(abs(direct) ** 2, axis=1)
        snr_power = abs(snr.astype(np.complex128)) ** 2

        def normalize(value):
            result = (value.astype(np.float64) * num_bins - snr_power) * norm**2
            threshold = row["snr_threshold"]
            if threshold:
                result[abs(snr * norm) <= threshold] = 0
            return result

        variants = {"direct_complex128": normalize(expected_sum)}
        for name, dtype, scale in [
            ("native_float32", np.float32, 1.0),
            ("native_float64_original_pi", np.float64, 1.0),
            ("native_float64_full_pi", np.float64, np.pi / 3.141592653),
        ]:
            result = np.zeros(len(points), dtype=dtype)
            chisq_cpu.point_chisq_code(
                result,
                corr,
                len(points),
                len(corr),
                (points * scale).astype(dtype),
                bins.astype(np.uint32),
                num_bins,
            )
            variants[name] = normalize(result)
        with scheme.TorchScheme("cpu", num_threads=1):
            actual = chisq_torch.power_chisq_at_points_from_precomputed(
                FrequencySeries(corr, delta_f=1.0),
                snr,
                norm,
                bins,
                points,
            )
            variants["torch_cpu_same_inputs"] = actual.numpy()
        low = row["segment"]["filter_bin_start"]
        high = row["segment"]["filter_bin_stop"]
        template = data["template"]
        power = (template.real**2 + template.imag**2) / data["psd"]
        power = power[low:high]
        cumulative = {
            "numpy_float32": np.cumsum(power),
            "numpy_float64": np.cumsum(power, dtype=np.float64),
            "torch_float32": torch.cumsum(torch.tensor(power), dim=0).numpy(),
        }
        binning = {}
        for name, cumsum in cumulative.items():
            thresholds = np.arange(num_bins) * cumsum[-1] / num_bins
            new_bins = np.append(
                np.searchsorted(cumsum, thresholds, side="right") + low, high
            )
            new_sums = direct_bin_sums(corr, points, new_bins)
            binning[name] = {
                "accumulated_power": float(cumsum[-1]),
                "relative_power_error_vs_float64": float(
                    cumsum[-1] / cumulative["numpy_float64"][-1] - 1
                ),
                "bins": new_bins.tolist(),
                "bins_equal_recorded": bool(np.array_equal(new_bins, bins)),
                "edge_difference_from_recorded": (new_bins - bins).tolist(),
                "direct_chisq_with_only_bins_changed": normalize(
                    np.sum(abs(new_sums) ** 2, axis=1)
                ).tolist(),
            }
        for value in variants.values():
            assert np.all(np.isfinite(value))
        output["captures"].append(
            {
                "template_hash": row["template_hash"],
                "segment": row["segment"],
                "points": points.astype(np.int64).tolist(),
                "bins": bins.tolist(),
                "snr_norm": norm,
                "snr_magnitude": (abs(snr) * norm).tolist(),
                "recorded_chisq": data["chisq"].tolist(),
                "variants": {key: val.tolist() for key, val in variants.items()},
                "binning_from_same_captured_template_and_psd": binning,
                "max_absolute_error_vs_direct": {
                    key: float(np.max(abs(val - variants["direct_complex128"])))
                    for key, val in variants.items()
                },
            }
        )
    output["input_sha256_after"] = {path: digest(path) for path in inputs}
    assert inputs == output["input_sha256_after"]
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["captures"], indent=2))


if __name__ == "__main__":
    main()
