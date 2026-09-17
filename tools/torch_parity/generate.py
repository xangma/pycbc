#!/usr/bin/env python3
"""Generate deterministic PyCBC parity artifacts in one isolated process."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time

import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scheme", choices=("cpu", "torch"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-revision")
    return parser.parse_args()


def _distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_revision(pycbc_module):
    source_root = Path(pycbc_module.__file__).resolve().parent.parent
    process = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = process.stdout.strip() if process.returncode == 0 else None
    return source_root, revision


def _epoch_metadata(value):
    epoch = getattr(value, "epoch", getattr(value, "_epoch", None))
    if epoch is None:
        return None
    seconds = getattr(epoch, "gpsSeconds", None)
    nanoseconds = getattr(epoch, "gpsNanoSeconds", None)
    if seconds is not None and nanoseconds is not None:
        return {
            "gpsSeconds": int(seconds),
            "gpsNanoSeconds": int(nanoseconds),
        }
    return {"value": float(epoch)}


def _tensor_from(value):
    tensor = getattr(getattr(value, "_data", None), "tensor", None)
    if tensor is not None:
        return tensor
    try:
        import torch
    except ImportError:
        return None
    return value if isinstance(value, torch.Tensor) else None


def _to_numpy(value):
    tensor = _tensor_from(value)
    if tensor is not None:
        return tensor.detach().cpu().numpy().copy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy()).copy()
    return np.asarray(value).copy()


def _scalar(value):
    tensor = _tensor_from(value)
    if tensor is not None:
        return float(tensor.detach().cpu().item())
    return float(value)


def _capture(name, value, arrays, records, expected_device=None):
    tensor = _tensor_from(value)
    if tensor is None:
        if expected_device is not None:
            raise AssertionError(
                f"{name} uses host storage, expected Torch on "
                f"{expected_device}"
            )
        storage = "numpy"
    else:
        storage = f"torch:{tensor.device}"
        if expected_device is not None:
            import torch
            expected = torch.device(expected_device)
            if expected.type == "cuda" and expected.index is None:
                expected = torch.device("cuda", torch.cuda.current_device())
            elif expected.type == "mps" and expected.index is None:
                expected = torch.device("mps", 0)
            elif expected.type == "cpu":
                expected = torch.device("cpu")
            if tensor.device != expected:
                raise AssertionError(
                    f"{name} is on {tensor.device}, expected {expected_device}"
                )

    array = _to_numpy(value)
    arrays[name] = array
    metadata = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "storage": storage,
    }
    for attribute in ("delta_t", "delta_f"):
        if hasattr(value, attribute):
            metadata[attribute] = float(getattr(value, attribute))
    epoch = _epoch_metadata(value)
    if epoch is not None:
        metadata["epoch"] = epoch
    if hasattr(value, "kind"):
        metadata["kind"] = str(value.kind)
    if hasattr(value, "corrupted_samples"):
        metadata["corrupted_samples"] = int(value.corrupted_samples)
    records[name] = metadata


def _run_corpus(expected_device):
    from pycbc.filter import match, matched_filter, resample_to_delta_t
    from pycbc.psd import analytical, welch
    from pycbc.types import Array, FrequencySeries, TimeSeries
    from pycbc.waveform import get_fd_waveform

    arrays = {}
    records = {}
    timings = {}

    def timed(name, function):
        started = time.perf_counter()
        result = function()
        timings[name] = time.perf_counter() - started
        return result

    base = Array(np.linspace(-2.0, 2.0, 4097, dtype=np.float64))
    polynomial = timed("array", lambda: (base * 1.5 - 0.25) ** 2 + base)
    _capture(
        "array_polynomial_f64",
        polynomial,
        arrays,
        records,
        expected_device,
    )

    sample_rate = 2048
    delta_t = 1.0 / sample_rate
    sample_times = np.arange(4099, dtype=np.float64) * delta_t
    samples = (
        np.sin(2.0 * np.pi * 31.0 * sample_times)
        + 0.2 * np.cos(2.0 * np.pi * 173.0 * sample_times)
        + 0.05 * np.sin(2.0 * np.pi * 401.0 * sample_times + 0.3)
    )
    series = TimeSeries(samples, delta_t=delta_t, epoch=1126259462.125)

    spectrum = timed(
        "fft", lambda: series[:4096].to_frequencyseries(delta_f=0.5)
    )
    _capture(
        "fft_timeseries_to_frequencyseries_f64",
        spectrum,
        arrays,
        records,
        expected_device,
    )

    shifted = timed("frequency_shift", lambda: spectrum.cyclic_time_shift(0.0137))
    _capture(
        "frequencyseries_cyclic_time_shift_c128",
        shifted,
        arrays,
        records,
        expected_device,
    )

    highpassed = timed(
        "fir_highpass",
        lambda: series.highpass_fir(55.0, 32, remove_corrupted=False),
    )
    _capture(
        "fir_highpass_timeseries_f64",
        highpassed,
        arrays,
        records,
        expected_device,
    )

    resampled = timed(
        "butterworth_resample",
        lambda: resample_to_delta_t(series, 4.0 * delta_t),
    )
    _capture(
        "resample_butterworth_timeseries_f64",
        resampled,
        arrays,
        records,
        expected_device,
    )

    estimated_psd = timed(
        "welch_psd",
        lambda: welch(
            series[:4096],
            seg_len=1024,
            seg_stride=512,
            avg_method="median",
        ),
    )
    _capture(
        "psd_welch_median_f64",
        estimated_psd,
        arrays,
        records,
        expected_device,
    )

    model_psd = timed(
        "analytical_psd",
        lambda: analytical.from_string(
            "aLIGOZeroDetHighPower",
            length=2049,
            delta_f=0.25,
            low_freq_cutoff=20.0,
        ),
    )
    _capture(
        "psd_aligo_zero_det_high_power_f64",
        model_psd,
        arrays,
        records,
        expected_device,
    )

    common_waveform = {
        "mass1": 35.0,
        "mass2": 28.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "delta_f": 0.25,
        "f_lower": 20.0,
        "f_final": 512.0,
        "f_ref": 20.0,
        "distance": 500.0,
        "inclination": 0.4,
        "coa_phase": 1.1,
    }
    for approximant, artifact_name in (
        ("TaylorF2", "waveform_taylorf2"),
    ):
        hp, hc = timed(
            artifact_name,
            lambda approximant=approximant: get_fd_waveform(
                approximant=approximant,
                **common_waveform,
            ),
        )
        _capture(
            artifact_name + "_plus_c128",
            hp,
            arrays,
            records,
            expected_device,
        )
        _capture(
            artifact_name + "_cross_c128",
            hc,
            arrays,
            records,
            expected_device,
        )

    frequencies = np.arange(2049, dtype=np.float64) * 0.25
    amplitude = np.exp(-0.5 * ((frequencies - 180.0) / 70.0) ** 2)
    amplitude[frequencies < 20.0] = 0.0
    phase = 0.009 * frequencies + 2.0e-5 * frequencies**2
    template = FrequencySeries(
        amplitude * np.exp(1j * phase),
        delta_f=0.25,
        epoch=1126259462.125,
    )
    data = template.cyclic_time_shift(0.00137) * np.exp(0.73j)
    weighting = FrequencySeries(
        1.0 + (frequencies / 240.0) ** 2,
        delta_f=template.delta_f,
    )

    snr = timed(
        "matched_filter",
        lambda: matched_filter(
            template,
            data,
            psd=weighting,
            low_frequency_cutoff=20.0,
            high_frequency_cutoff=500.0,
        ),
    )
    _capture(
        "matched_filter_snr_c128",
        snr,
        arrays,
        records,
        expected_device,
    )

    match_result = timed(
        "match",
        lambda: match(
            template,
            data,
            psd=weighting,
            low_frequency_cutoff=20.0,
            high_frequency_cutoff=500.0,
            subsample_interpolation=True,
            return_phase=True,
        ),
    )
    match_values = np.asarray([_scalar(value) for value in match_result])
    _capture("matched_filter_match_f64", match_values, arrays, records)

    return arrays, records, timings


def main():
    args = _parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        raise ValueError("label may contain only letters, numbers, '.', '_' and '-'")
    if args.scheme == "cpu" and args.device != "cpu":
        raise ValueError("CPU scheme only supports --device=cpu")
    if args.scheme == "torch":
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1"
    else:
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"

    import pycbc
    from pycbc import scheme

    source_root, revision = _source_revision(pycbc)
    if args.expected_revision and revision != args.expected_revision:
        raise RuntimeError(
            f"loaded PyCBC revision {revision!r}, expected "
            f"{args.expected_revision!r}; source is {source_root}"
        )

    if args.scheme == "torch":
        context = scheme.TorchScheme(args.device)
        expected_device = args.device
    else:
        context = scheme.CPUScheme(num_threads=1)
        expected_device = None

    started = time.time()
    with context:
        arrays, records, timings = _run_corpus(expected_device)
    elapsed = time.time() - started

    lalsimulation_loaded = any(
        name == "lalsimulation" or name.startswith("lalsimulation.")
        for name in sys.modules
    )
    runtime = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "source_root": str(source_root),
        "source_revision": revision,
        "pycbc": _distribution_version("PyCBC"),
        "numpy": _distribution_version("numpy"),
        "scipy": _distribution_version("scipy"),
        "torch": _distribution_version("torch"),
        "lalsuite": _distribution_version("lalsuite"),
        "scheme": args.scheme,
        "device": args.device,
        "lalsimulation_loaded": lalsimulation_loaded,
        "elapsed_seconds": elapsed,
    }

    manifest = {
        "schema_version": 1,
        "label": args.label,
        "runtime": runtime,
        "timings_seconds": timings,
        "records": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    array_path = args.output_dir / f"{args.label}.npz"
    manifest_path = args.output_dir / f"{args.label}.json"
    array_temp = array_path.with_suffix(".npz.tmp")
    manifest_temp = manifest_path.with_suffix(".json.tmp")
    with array_temp.open("wb") as output:
        np.savez_compressed(output, **arrays)
    manifest_temp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(array_temp, array_path)
    os.replace(manifest_temp, manifest_path)
    print(
        f"wrote {len(arrays)} records for {args.label} "
        f"({args.scheme}:{args.device}) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
