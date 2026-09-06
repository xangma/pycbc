#!/usr/bin/env python3
"""Bounded diagnostic; run with the qualified Linux Python/source environment.

Reads existing qualification PSDs and the compressed bank. Optional strain is
the exact float32 array passed to StrainSegments. Optional correlation is the
CPU full complex64 correlation buffer for --template-hash; --indices refer to
that full buffer's inverse FFT, before the analysis slice. No pass budgets are
changed. All outputs are diagnostic, never qualification or timing evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import h5py
import numpy as np


def digest(data):
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def metrics(reference, candidate):
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    valid = np.isfinite(a) & np.isfinite(b)
    av = a[valid].astype(np.complex128 if np.iscomplexobj(a) else np.float64)
    bv = b[valid].astype(av.dtype)
    diff = np.abs(bv - av)
    nonzero = np.abs(av) > 0
    rel = diff[nonzero] / np.abs(av[nonzero])
    norm = float(np.linalg.norm(av))
    out = {
        "finite_count": int(valid.sum()),
        "nonfinite_disagreement": int(np.count_nonzero(~valid & (a != b))),
        "max_absolute": float(diff.max(initial=0)),
        "relative_l2": float(np.linalg.norm(diff) / norm) if norm else None,
        "relative_abs_quantiles": np.quantile(rel, [0, .5, .99, 1]).tolist()
        if len(rel) else [],
    }
    if np.iscomplexobj(a):
        phase = np.abs(np.angle(bv[nonzero] * av[nonzero].conj()))
        out["max_phase_radians"] = float(phase.max(initial=0))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template-hash", type=int, default=7190896270511713173)
    parser.add_argument("--strain", type=Path)
    parser.add_argument("--correlation", type=Path)
    parser.add_argument("--indices", type=int, nargs="*", default=[])
    parser.add_argument("--schemes", nargs="+", default=["cpu", "torch-cpu", "torch-cuda"],
                        choices=["cpu", "torch-cpu", "torch-cuda"])
    args = parser.parse_args()
    if args.schemes[0] != "cpu":
        parser.error("cpu must be the first scheme")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    import pycbc
    from pycbc.scheme import CPUScheme, TorchScheme
    from pycbc.types import Array, FrequencySeries, TimeSeries, zeros
    from pycbc.waveform.compress import fd_decompress
    from pycbc.fft import IFFT
    from pycbc.fft.backend_support import set_backend
    from pycbc.psd import welch, interpolate, inverse_spectrum_truncation

    torch.set_num_threads(1)
    length, delta_f, rate = 2097152, 1 / 512, 4096
    band = slice(15360, 1048576)
    report = {
        "purpose": "diagnostic only; original scientific tolerances unchanged",
        "host": platform.node(), "pid": os.getpid(), "python": sys.executable,
        "pycbc_source": pycbc.__file__, "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "template_hash": str(args.template_hash), "fft_length": length,
        "delta_f": delta_f, "filter_slice": [band.start, band.stop],
        "schemes": args.schemes, "psd": {}, "waveforms": {}, "fft": {},
    }

    def context(name):
        if name == "cpu":
            return CPUScheme(1)
        return TorchScheme("cpu", num_threads=1) if name == "torch-cpu" else TorchScheme("cuda:0")

    def select_backend(name):
        set_backend(["mkl"] if name == "cpu" else ["torch"])

    def save(name, data):
        np.save(args.output_dir / (name + ".npy"), data, allow_pickle=False)
        return {"relative_path": name + ".npy", "data_sha256": digest(data),
                "dtype": str(data.dtype), "shape": list(data.shape)}

    saved_psds, quals = {}, {}
    for name in args.schemes:
        run = args.root / "runs" / f"qual-final-{name}-l512"
        qual = json.loads((run / "qualification.json").read_text())
        if qual["status"] != "success":
            raise ValueError(f"unsuccessful qualification: {run}")
        quals[name] = qual["observations"]
        meta = quals[name]["psd_arrays"][0]
        value = np.load(run / meta["relative_path"], allow_pickle=False)
        if digest(value) != meta["data_sha256"] or value.shape != (length // 2 + 1,):
            raise ValueError(f"qualification PSD identity/shape mismatch: {name}")
        if not np.all(np.isfinite(value[band]) & (value[band] > 0)):
            raise ValueError(f"invalid filter PSD: {name}")
        saved_psds[name] = value
        report["psd"][name] = {"saved_vs_cpu": metrics(saved_psds["cpu"][band], value[band])}

    bank_path = args.root / "inputs" / "bank-compressed-1e5.hdf"
    report["bank_sha256"] = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    if any(q["banks"][0]["file"]["sha256"] != report["bank_sha256"] for q in quals.values()):
        raise ValueError("bank differs from qualification")
    with h5py.File(bank_path, "r") as bank:
        group = bank["compressed_waveforms"][str(args.template_hash)]
        amp, phase, freq = (group[k][:] for k in ("amplitude", "phase", "sample_points"))
        report["compressed"] = {"samples": len(freq), "phase_range": [float(phase.min()), float(phase.max())]}
    waveforms, sigma = {}, {}
    for name in args.schemes:
        with context(name):
            select_backend(name)
            output = FrequencySeries(zeros(length // 2 + 1, dtype=np.complex64), delta_f=delta_f, copy=False)
            fd_decompress(amp, phase, freq, out=output, f_lower=30, interpolation="inline_linear")
            value = output.numpy().copy()
        waveforms[name] = value
        report["waveforms"][name] = save("waveform-" + name, value)
        report["waveforms"][name]["vs_cpu"] = metrics(waveforms["cpu"][band], value[band])
        sigma[name] = {}
        for psd_name, psd in saved_psds.items():
            sigma[name][psd_name] = float(4 * delta_f * np.sum(np.abs(value[band].astype(np.complex128))**2 / psd[band]))
    report["direct_double_sigmasq_waveform_by_psd"] = sigma

    if args.strain:
        strain = np.load(args.strain, allow_pickle=False)
        for name in args.schemes:
            expected = quals[name]["conditioned_strain"][0]
            if digest(strain) != expected["data_sha256"]:
                raise ValueError("strain differs from exact qualified conditioned strain")
        report["strain_sha256"] = digest(strain)
        cpu_welch = cpu_interp = None
        for name in args.schemes:
            with context(name):
                select_backend(name)
                ts = TimeSeries(strain.copy(), delta_t=1 / rate)
                raw = welch(ts, seg_len=32 * rate, seg_stride=16 * rate, num_segments=126,
                            avg_method="median", require_exact_data_fit=False)
                raw_np = raw.numpy().copy()
                expanded = interpolate(raw, delta_f, length // 2 + 1)
                expanded_np = expanded.numpy().copy()
                if name == "cpu":
                    cpu_welch, cpu_interp = raw_np, expanded_np
                common_raw = FrequencySeries(cpu_welch.copy(), delta_f=1 / 32)
                expanded_common = interpolate(common_raw, delta_f, length // 2 + 1).numpy().copy()
                stages = {"welch_vs_cpu": metrics(cpu_welch[960:-1], raw_np[960:-1]),
                          "interpolate_vs_cpu": metrics(cpu_interp[band], expanded_np[band]),
                          "interpolate_common_input": metrics(cpu_interp[band], expanded_common[band])}
                for label, values in (("native", expanded_np), ("common_cpu_interp", cpu_interp),
                                      ("common_cpu_interp_float64", cpu_interp.astype(np.float64))):
                    trial = FrequencySeries(values.copy(), delta_f=delta_f)
                    final = inverse_spectrum_truncation(trial, 16 * rate, low_frequency_cutoff=30,
                                                        trunc_method="hann", which_spectrum="invasd")
                    final_np = final.numpy().copy()
                    stages[label + "_vs_saved_cpu"] = metrics(saved_psds["cpu"][band], final_np[band])
                    if label == "native":
                        stages["native_vs_own_saved"] = metrics(saved_psds[name][band], final_np[band])
                    save("psd-" + name + "-" + label, final_np)
                for n_window in (32 * rate, 16 * rate):
                    device = "cpu" if name != "torch-cuda" else "cuda:0"
                    window = torch.hann_window(n_window, periodic=False, device=device, dtype=torch.float32).cpu().numpy()
                    stages[f"torch_hann_{n_window}_vs_numpy_float32"] = metrics(np.hanning(n_window).astype(np.float32), window)
                report["psd"][name]["replay_stages"] = stages

    if args.correlation:
        correlation = np.load(args.correlation, allow_pickle=False)
        if correlation.shape != (length,) or correlation.dtype != np.complex64:
            raise ValueError("correlation must be full length complex64")
        if not np.isfinite(correlation).all():
            raise ValueError("nonfinite correlation")
        if np.any(correlation[:band.start]) or np.any(correlation[band.stop:]):
            raise ValueError("correlation is not zero outside the qualified filter band")
        indices = np.asarray(args.indices, dtype=np.int64)
        if np.any((indices < 0) | (indices >= length)):
            raise ValueError("indices must refer to the full IFFT output")
        report["correlation"] = {"path": str(args.correlation), "data_sha256": digest(correlation),
                                 "indices": indices.tolist(),
                                 "assumption": "captured CPU correlation for this template hash and one segment"}
        truth = np.fft.ifft(correlation.astype(np.complex128)) * length
        reference = None
        h0 = waveforms["cpu"][band].astype(np.complex128)
        support = h0 != 0
        if np.any(correlation[band][~support]):
            raise ValueError("correlation has power where reference waveform is zero")

        def fft_trial(name, corr):
            with context(name):
                select_backend(name)
                input_array = Array(corr.copy())
                output_array = zeros(length, dtype=np.complex64)
                engine = IFFT(input_array, output_array)
                engine.execute()
                result = output_array.numpy().copy()
                route = {"class": type(engine).__module__ + "." + type(engine).__name__}
                for attr in ("_mkl_plan", "_fftw_plan", "_promoted_batch_plan"):
                    route[attr + "_present"] = getattr(engine, attr, None) is not None
            return result, route

        for name in args.schemes:
            result, route = fft_trial(name, correlation)
            if name == "cpu":
                reference = result
            entry = {"route": route, "common_input_vs_cpu": metrics(reference, result),
                     "common_input_vs_numpy_complex128": metrics(truth, result), "variants": {}}
            for label, use_waveform, use_psd in (("waveform_only", True, False),
                                                ("psd_only", False, True), ("waveform_and_psd", True, True)):
                modifier = np.ones(len(h0), dtype=np.complex128)
                hn = waveforms[name][band].astype(np.complex128)
                if use_waveform:
                    modifier[support] *= np.conj(hn[support] / h0[support])
                if use_psd:
                    modifier *= saved_psds["cpu"][band].astype(float) / saved_psds[name][band]
                corr = correlation.copy()
                corr[band] = (correlation[band].astype(np.complex128) * modifier).astype(np.complex64)
                variant, _ = fft_trial(name, corr)
                waveform_name = name if use_waveform else "cpu"
                psd_name = name if use_psd else "cpu"
                norm_ratio = np.sqrt(sigma[waveform_name][psd_name] / sigma["cpu"]["cpu"])
                normalized = variant.astype(np.complex128) / norm_ratio
                sample = reference[indices].astype(np.complex128)
                changed = normalized[indices]
                with np.errstate(divide="ignore", invalid="ignore"):
                    amplitude = (np.abs(changed) / np.abs(sample) - 1).tolist()
                entry["variants"][label] = {
                    "relative_sigmasq": norm_ratio**2 - 1,
                    "normalized_series_vs_cpu": metrics(reference, normalized),
                    "selected_relative_snr_change": amplitude,
                    "selected_phase_change": np.angle(changed * sample.conj()).tolist(),
                    "selected_cpu_raw_complex": [[float(x.real), float(x.imag)] for x in sample],
                }
            report["fft"][name] = entry
    report["limitations"] = [
        "PSD substitution holds the CPU data FFT fixed; it does not reproduce backend data-FFT differences.",
        "Correlation substitution uses double ratios then complex64 rounding, not the exact native multiply/divide order.",
        "Direct sigmasq uses double summation; native template norm reductions may differ.",
        "No replacement qualification, scientific pass claim, or timing result is produced.",
    ]
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output_dir / "report.json")


if __name__ == "__main__":
    main()
