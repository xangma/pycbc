#!/usr/bin/env python3
"""Generate deterministic PyCBC JAX parity artifacts in one isolated process."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scheme", choices=("cpu", "jax"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-revision")
    return parser.parse_args()


def _distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    if hasattr(value, "_data") and hasattr(value._data, "array"):
        return np.asarray(value._data.array)
    return np.asarray(value)


def _capture(name, value, arrays, records, expected_device):
    from pycbc.types.array_jax import JAXArrayData

    if not hasattr(value, "_data") or not isinstance(value._data, JAXArrayData):
        storage = "numpy"
    else:
        storage = f"jax:{value._data.array.device}"

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
    if hasattr(value, "epoch") and value.epoch is not None:
        metadata["epoch"] = float(value.epoch)
    records[name] = metadata


def _run_corpus(expected_device):
    from pycbc.filter import match, matched_filter
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
    _capture("array_polynomial_f64", polynomial, arrays, records, expected_device)

    sample_rate = 2048
    delta_t = 1.0 / sample_rate
    sample_times = np.arange(4099, dtype=np.float64) * delta_t
    samples = (
        np.sin(2.0 * np.pi * 31.0 * sample_times)
        + 0.2 * np.cos(2.0 * np.pi * 173.0 * sample_times)
        + 0.05 * np.sin(2.0 * np.pi * 401.0 * sample_times + 0.3)
    )
    series = TimeSeries(samples, delta_t=delta_t, epoch=1126259462.125)

    spectrum = timed("fft", lambda: series[:4096].to_frequencyseries(delta_f=0.5))
    _capture("fft_timeseries_to_frequencyseries_f64", spectrum, arrays, records, expected_device)

    shifted = timed("frequency_shift", lambda: spectrum.cyclic_time_shift(0.0137))
    _capture("frequencyseries_cyclic_time_shift_c128", shifted, arrays, records, expected_device)

    estimated_psd = timed(
        "welch_psd",
        lambda: welch(series[:4096], seg_len=1024, seg_stride=512, avg_method="median"),
    )
    _capture("psd_welch_median_f64", estimated_psd, arrays, records, expected_device)

    model_psd = timed(
        "analytical_psd",
        lambda: analytical.from_string(
            "aLIGOZeroDetHighPower", length=2049, delta_f=0.25, low_freq_cutoff=20.0
        ),
    )
    _capture("psd_aligo_zero_det_high_power_f64", model_psd, arrays, records, expected_device)

    common_waveform = {
        "approximant": "TaylorF2",
        "mass1": 1.4,
        "mass2": 1.4,
        "f_lower": 20.0,
        "delta_f": 0.25,
    }
    hp, hc = timed("waveform", lambda: get_fd_waveform(**common_waveform))
    _capture("waveform_taylorf2_hp_c128", hp, arrays, records, expected_device)
    _capture("waveform_taylorf2_hc_c128", hc, arrays, records, expected_device)

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
        "filter",
        lambda: matched_filter(
            template,
            data,
            psd=weighting,
            low_frequency_cutoff=20.0,
            high_frequency_cutoff=500.0,
        ),
    )
    _capture("matched_filter_snr_c128", snr, arrays, records, expected_device)

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
    _capture(
        "matched_filter_match_value_f64",
        Array([match_result[0]]),
        arrays,
        records,
        expected_device,
    )

    return arrays, records, timings


def main():
    args = _parse_args()
    import pycbc
    from pycbc import scheme

    source_root = Path(pycbc.__file__).resolve().parent.parent
    process = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = process.stdout.strip() if process.returncode == 0 else None

    if args.scheme == "jax":
        import jax
        jax.config.update("jax_enable_x64", True)
        context = scheme.JAXScheme(args.device)
    else:
        context = scheme.CPUScheme()

    started = time.time()
    with context:
        arrays, records, timings = _run_corpus(args.device if args.scheme == "jax" else None)
    elapsed = time.time() - started

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
        "jax": _distribution_version("jax"),
        "lalsuite": _distribution_version("lalsuite"),
        "scheme": args.scheme,
        "device": args.device,
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
    np.savez_compressed(args.output_dir / f"{args.label}.npz", **arrays)
    with (args.output_dir / f"{args.label}.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated parity artifacts in {args.output_dir}")


if __name__ == "__main__":
    main()
