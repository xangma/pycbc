#!/usr/bin/env python3
"""Diagnostic precision matrix for one captured template and strain segment.

Run in the frozen Linux source968 environment. This companion preserves v1.
--forward-data additionally tests the actual functional strain FFT, native
overwhitening, Correlator and class IFFT. No qualification or timing is produced.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_comparison(reference, candidate):
    """Apply frozen SNR/phase budgets at identical sample indices."""
    a, b = np.asarray(reference, np.complex128), np.asarray(candidate, np.complex128)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid comparison samples")
    amplitude_difference = np.abs(b) - np.abs(a)
    phase_difference = np.angle(b * a.conj())
    allowed = 1e-5 + 1e-4 * np.abs(a)
    return {
        "reference_snr": np.abs(a).tolist(), "candidate_snr": np.abs(b).tolist(),
        "snr_difference": amplitude_difference.tolist(),
        "relative_snr_difference": (amplitude_difference / np.abs(a)).tolist(),
        "phase_difference_radians": phase_difference.tolist(),
        "snr_violations": int(np.count_nonzero(np.abs(amplitude_difference) > allowed)),
        "phase_violations": int(np.count_nonzero(np.abs(phase_difference) > 1e-4)),
        "sample_count": a.size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segment", type=int, default=3)
    parser.add_argument("--indices", type=int, nargs="*", default=[1884341])
    parser.add_argument("--forward-data", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    inputs = {str(Path(__file__).resolve()): file_hash(__file__)}

    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = file_hash(path)
        return path

    helper_path = bind(root / "diagnose-snr-psd.py")
    if inputs[str(helper_path)] != "49424d202dfaa121a1929f72f73ce82be6394e7886db2c9af79c138cb20c8beb":
        raise ValueError("v1 helper changed")
    spec = importlib.util.spec_from_file_location("snr_psd_v1", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    metrics, digest = helper.metrics, helper.digest

    def read_json(path):
        return json.loads(bind(path).read_text())

    def load_array(directory, meta, file_key="sha256"):
        path = bind(directory / Path(meta["path"]).name)
        if inputs[str(path)] != meta[file_key]:
            raise ValueError(f"capture file hash mismatch: {path}")
        value = np.load(path, allow_pickle=False)
        if "dtype" in meta and (str(value.dtype) != meta["dtype"] or list(value.shape) != meta["shape"]):
            raise ValueError(f"capture dtype/shape mismatch: {path}")
        if "data_sha256" in meta and digest(value) != meta["data_sha256"]:
            raise ValueError(f"capture data hash mismatch: {path}")
        return value

    names = ("cpu", "torch-cpu", "torch-cuda")
    captures, arrays, statuses, strains = {}, {}, {}, {}
    for name, dirname in (("cpu", "chisq-input-capture-cpu"), ("torch-cuda", "chisq-input-capture-cuda")):
        directory = root / dirname
        status = read_json(directory / "status.json")
        if status["source_info"]["commit"] != "968bcd558117262af0d603710b054174659adb51":
            raise ValueError("capture source is not the frozen source968")
        if status["input_sha256"] != status["input_sha256_after"] or status["source_status_after"]:
            raise ValueError("capture inputs changed")
        selected = [c for c in status["captures"] if c["segment"]["number"] == args.segment]
        if len(selected) != 1:
            raise ValueError("expected one captured template for selected segment")
        captures[name], statuses[name] = selected[0], status
        arrays[name] = {k: load_array(directory, v) for k, v in selected[0]["arrays"].items()}
        strains[name] = load_array(directory, status["conditioned_strain"])
    if not np.array_equal(strains["cpu"], strains["torch-cuda"]):
        raise ValueError("conditioned strain differs across captures")
    cpu = captures["cpu"]
    geometry = cpu["segment"]
    if captures["torch-cuda"]["segment"] != geometry or captures["torch-cuda"]["template_hash"] != cpu["template_hash"]:
        raise ValueError("capture work mismatch")
    strain = strains["cpu"]
    rate, delta_f, length = 4096, 1 / 512, 2097152
    band = slice(geometry["filter_bin_start"], geometry["filter_bin_stop"])
    analyze = slice(geometry["analyze_start"], geometry["analyze_stop"])
    start = geometry["cumulative_index"] - analyze.start
    chunk = strain[start:start + length]
    if strain.dtype != np.float32 or chunk.shape != (length,) or not np.isfinite(chunk).all():
        raise ValueError("invalid exact conditioned strain segment")
    epoch = statuses["cpu"]["conditioned_strain"]["start_time"] + start / rate
    if epoch != geometry["epoch"] or band != slice(15360, 1048576):
        raise ValueError("capture segment epoch/filter mismatch")
    indices = list(args.indices)
    for data in arrays.values():
        raw_indices = data["indices"]
        if not np.equal(raw_indices, np.rint(raw_indices)).all():
            raise ValueError("nonintegral captured trigger index")
        indices.extend(raw_indices.astype(np.int64).tolist())
        if data["corr"].shape != (length,) or data["corr"].dtype != np.complex64:
            raise ValueError("invalid captured correlation")
        if np.any(data["corr"][:band.start]) or np.any(data["corr"][band.stop:]):
            raise ValueError("captured correlation is nonzero outside filter band")
    indices = np.array(sorted(set(indices)), dtype=np.int64)
    if not len(indices) or np.any((indices < analyze.start) | (indices >= analyze.stop)):
        raise ValueError("indices must be in the captured analysis interval")
    quals, saved_psds = {}, {}
    for name in names:
        directory = root / "runs" / f"qual-final-{name}-l512"
        qual = read_json(directory / "qualification.json")
        if qual["status"] != "success":
            raise ValueError(f"unsuccessful qualification: {name}")
        quals[name] = qual["observations"]
        if quals[name]["conditioned_strain"][0]["data_sha256"] != digest(strain):
            raise ValueError("strain differs from qualification")
        meta = quals[name]["psd_arrays"][0]
        value = np.load(bind(directory / meta["relative_path"]), allow_pickle=False)
        if digest(value) != meta["data_sha256"] or value.shape != (length // 2 + 1,) or value.dtype != np.float32:
            raise ValueError("qualification PSD identity mismatch")
        if not np.all(np.isfinite(value[band]) & (value[band] > 0)):
            raise ValueError("invalid qualification PSD in filter band")
        saved_psds[name] = value
        if name in arrays and not np.array_equal(arrays[name]["psd"], value):
            raise ValueError("capture PSD differs from qualification")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    import pycbc
    from pycbc.fft import IFFT
    from pycbc.fft.backend_support import set_backend
    from pycbc.filter.matchedfilter import make_frequency_series, Correlator
    from pycbc.psd import welch, interpolate, inverse_spectrum_truncation
    from pycbc.scheme import CPUScheme, TorchScheme
    from pycbc.types import Array, FrequencySeries, TimeSeries, zeros

    torch.set_num_threads(1)

    def context(name):
        if name == "cpu":
            return CPUScheme(1)
        return TorchScheme("cpu", num_threads=1) if name == "torch-cpu" else TorchScheme("cuda:0")

    def backend(name):
        set_backend(["mkl"] if name == "cpu" else ["torch"])

    def save(label, value):
        path = args.output_dir / (label + ".npy")
        np.save(path, value, allow_pickle=False)
        return {"path": path.name, "sha256": file_hash(path), "data_sha256": digest(value),
                "dtype": str(value.dtype), "shape": list(value.shape)}

    def truncate(value):
        result = inverse_spectrum_truncation(
            value, 16 * rate, low_frequency_cutoff=30,
            trunc_method="hann", which_spectrum="invasd")
        return result.numpy().astype(np.float32)

    report = {
        "status": "diagnostic-running", "host": platform.node(), "pid": os.getpid(),
        "python": sys.executable, "pycbc_source": pycbc.__file__, "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(), "affinity": sorted(os.sched_getaffinity(0)),
        "forward_data": args.forward_data, "template_hash": str(cpu["template_hash"]),
        "segment": geometry, "strain_slice": [start, start + length], "indices": indices.tolist(),
        "capture_states": {name: status["state"] for name, status in statuses.items()},
        "capture_note": "CPU capture harness failed only its expected-capture-count assertion; raw arrays are hash checked.",
        "input_sha256": inputs, "psd": {}, "forward_fft": {}, "matrix": {},
        "tolerances": {"snr_absolute": 1e-5, "snr_relative": 1e-4, "phase_absolute_radians": 1e-4},
        "limitations": [
            "One captured template and segment; no qualification, full trigger parity or timing claim.",
            "All trials retain native complex64 overwhitening, correlation and IFFT after explicit precision/cast changes.",
            "Variant normalization uses direct float64 power summation, which can differ slightly from native template normalization.",
            "PSD output remains float32; native/full-double PSD trials use each backend's own Welch and interpolation.",
            "Comparisons above threshold inspect samples before clustering or chi-square/newSNR vetoes.",
        ],
    }
    psds, spectra, cpu_series = {}, {}, {}
    for name in names:
        print(f"PSD and forward FFT: {name}", flush=True)
        with context(name):
            backend(name)
            psds[name] = {"native32": saved_psds[name]}
            raw32 = welch(TimeSeries(strain.copy(), delta_t=1 / rate), seg_len=32 * rate,
                          seg_stride=16 * rate, num_segments=126, avg_method="median",
                          require_exact_data_fit=False)
            interp32 = interpolate(raw32, delta_f, length // 2 + 1)
            native = truncate(interp32)
            if not np.array_equal(native, saved_psds[name]):
                raise ValueError(f"native PSD replay differs from frozen qualification: {name}")
            psds[name]["trunc64_cast32"] = truncate(interp32.astype(np.float64))
            raw64 = welch(TimeSeries(strain.astype(np.float64), delta_t=1 / rate), seg_len=32 * rate,
                          seg_stride=16 * rate, num_segments=126, avg_method="median",
                          require_exact_data_fit=False)
            psds[name]["all64_cast32"] = truncate(interpolate(raw64, delta_f, length // 2 + 1))
            psds[name]["common_cpu_all64_cast32"] = psds["cpu"]["all64_cast32"]
            report["psd"][name] = {}
            for label, value in psds[name].items():
                if not np.all(np.isfinite(value[band]) & (value[band] > 0)):
                    raise ValueError(f"invalid trial PSD: {name}/{label}")
                report["psd"][name][label] = {
                    "array": save(f"psd-{name}-{label}", value),
                    "vs_frozen_cpu": metrics(saved_psds["cpu"][band], value[band]),
                    "vs_same_variant_cpu": metrics(psds["cpu"][label][band], value[band]),
                }
            if not args.forward_data:
                continue
            spectra[name] = {}
            report["forward_fft"][name] = {}
            for label, dtype in (("native32", np.float32), ("fft64_cast32", np.float64)):
                value = make_frequency_series(TimeSeries(chunk.astype(dtype), delta_t=1 / rate)).numpy().astype(np.complex64)
                spectra[name][label] = value
                report["forward_fft"][name][label] = {
                    "array": save(f"data-fft-{name}-{label}", value),
                    "vs_same_variant_cpu": metrics(spectra["cpu"][label][band], value[band]),
                    "vs_numpy_float64": metrics((np.fft.rfft(chunk.astype(np.float64)) / rate)[band], value[band]),
                }

    if args.forward_data:
        def inverse(correlation):
            buffer = Array(correlation.copy())
            result = zeros(length, dtype=np.complex64)
            engine = IFFT(buffer, result)
            engine.execute()
            route = {"class": type(engine).__module__ + "." + type(engine).__name__}
            for attr in ("_mkl_plan", "_fftw_plan", "_promoted_batch_plan"):
                route[attr + "_present"] = getattr(engine, attr, None) is not None
            return result.numpy().copy(), route

        with context("cpu"):
            backend("cpu")
            raw_cpu, _ = inverse(arrays["cpu"]["corr"])
        frozen_cpu = raw_cpu.astype(np.complex128) * cpu["snr_norm"]
        report["frozen_cpu_selected_snr"] = np.abs(frozen_cpu[indices]).tolist()
        for name in names:
            print(f"Correlation precision matrix: {name}", flush=True)
            report["matrix"][name] = {}
            waveform = arrays["torch-cuda" if name == "torch-cuda" else "cpu"]["template"]
            with context(name):
                backend(name)
                h = FrequencySeries(waveform.copy(), delta_f=delta_f)
                for data_label, data in spectra[name].items():
                    for psd_label, psd in psds[name].items():
                        label = data_label + "/" + psd_label
                        stilde = FrequencySeries(data.copy(), delta_f=delta_f)
                        stilde /= FrequencySeries(psd.copy(), delta_f=delta_f)
                        corr = zeros(length, dtype=np.complex64)
                        correlator = Correlator(h[band], stilde[band], corr[band])
                        correlator.correlate()
                        corr_np = corr.numpy().copy()
                        raw, route = inverse(corr_np)
                        sigma = float(4 * delta_f * np.sum(np.abs(waveform[band].astype(np.complex128))**2 / psd[band]))
                        norm = 4 * delta_f / np.sqrt(sigma)
                        series = raw.astype(np.complex128) * norm
                        if name == "cpu":
                            cpu_series[label] = series
                        cpu_variant = cpu_series[label]
                        active = (np.abs(cpu_variant[analyze]) >= 5.5) | (np.abs(series[analyze]) >= 5.5)
                        result = {
                            "route": route, "direct_double_sigmasq": sigma, "snr_norm": norm,
                            "selected_vs_frozen_cpu": sample_comparison(frozen_cpu[indices], series[indices]),
                            "selected_vs_same_variant_cpu": sample_comparison(cpu_variant[indices], series[indices]),
                            "above_threshold_sample_comparison": sample_comparison(cpu_variant[analyze][active], series[analyze][active]),
                        }
                        if data_label == "native32" and psd_label == "native32" and name in arrays:
                            result["native_correlation_vs_capture"] = metrics(arrays[name]["corr"], corr_np)
                            result["native_correlation_bitwise_equal"] = bool(np.array_equal(arrays[name]["corr"], corr_np))
                            own_indices = arrays[name]["indices"].astype(np.int64)
                            result["native_selected_raw_vs_capture"] = metrics(arrays[name]["snrv"], raw[own_indices])
                            result["direct_norm_relative_to_capture"] = norm / captures[name]["snr_norm"] - 1
                        report["matrix"][name][label] = result
    if any(file_hash(path) != expected for path, expected in inputs.items()):
        raise ValueError("diagnostic input changed during replay")
    report["status"] = "diagnostic-complete"
    with (args.output_dir / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output_dir / "report.json", flush=True)


if __name__ == "__main__":
    main()
