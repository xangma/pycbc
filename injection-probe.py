#!/usr/bin/env python3
"""Bounded, fixed-input CPU injection probe; no acquisition or remote execution.

Run in the checked proposed CPU environment. See injection-probe-method.md.
"""

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import time

# Set before importing numerical libraries, including when imported by tests.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"
sys.dont_write_bytecode = True

import numpy as np
from scipy import fft

ORIGINAL = "40e94792b3edf59f39b18b65102b28a4f74433a7"
PROPOSED = "123e1fb3ef1b338cada636e71c3e9c7987002402"
SELECTED = [(0, 0), (107, 0), (256, 0), (283, 0), (348, 0), (370, 4)]
RHOS = [5.5, 8.0, 12.0, 20.0]
LD, CLD = np.longdouble, np.clongdouble


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha(values):
    return hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def save(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Capture:
    def __init__(self, directory, metadata):
        self.root = directory.resolve()
        self.metadata = json.loads(metadata.read_text())
        require(self.metadata.get("status") == "complete", "Incomplete capture")
        self.pins = {str(metadata.resolve()): sha(metadata)}

    def load(self, name):
        path = (self.root / name).resolve()
        require(path.parent == self.root, "Capture array outside capture directory")
        item = self.metadata["arrays"][name]
        require(sha(path) == item["sha256"], f"File hash mismatch: {name}")
        values = np.load(path, allow_pickle=False)
        require(
            list(values.shape) == item["shape"] and str(values.dtype) == item["dtype"],
            f"Array format mismatch: {name}",
        )
        require(array_sha(values) == item["data_sha256"], f"Array hash mismatch: {name}")
        self.pins[str(path)] = item["sha256"]
        return values

    def unchanged(self):
        return all(sha(path) == digest for path, digest in self.pins.items())


def original_functions(repo):
    """Execute exact frozen function ASTs with compatible checked CPU globals."""
    from pycbc.filter import matchedfilter as mf
    from pycbc.vetoes import chisq

    functions, pins = {}, {}
    for path, name, current in (
        ("pycbc/filter/matchedfilter.py", "sigmasq_series", mf.sigmasq_series),
        (
            "pycbc/vetoes/chisq.py",
            "power_chisq_at_points_from_precomputed",
            chisq.power_chisq_at_points_from_precomputed,
        ),
    ):
        source = git(repo, "show", f"{ORIGINAL}:{path}")
        node = next(
            n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        scope = dict(current.__globals__)
        exec(compile(ast.Module(body=[node], type_ignores=[]), f"{ORIGINAL}:{path}", "exec"), scope)
        functions[name] = scope[name]
        pins[path] = dict(
            commit=ORIGINAL,
            function_sha256=hashlib.sha256(
                ast.get_source_segment(source, node).encode()
            ).hexdigest(),
        )
    return functions, pins


def phase(start, stop, index, size, dtype=np.float64):
    # Exact integer range reduction avoids the large k*t phase responsible for
    # the old point evaluator's late-index sensitivity.
    residue = (np.arange(start, stop, dtype=np.int64) * int(index)) % size
    theta = residue.astype(dtype) * (2 * np.arccos(dtype(-1)) / size)
    return np.cos(theta) + 1j * np.sin(theta)


def calibration(template, psd, kmin, kmax, delta_f):
    h = template[kmin:kmax].astype(CLD)
    power = (h.real**2 + h.imag**2) / psd[kmin:kmax].astype(LD)
    q = LD(4) * LD(delta_f) * np.sum(power, dtype=LD)
    require(np.isfinite(q) and q > 0, "Invalid independent template normalization")
    return q, power


def injected_input(template, noise, rho, q, index, kmin, kmax):
    size = (len(template) - 1) * 2
    injected = np.zeros(len(template), dtype=np.complex128)
    injected[kmin:kmax] = (
        float(LD(rho) / np.sqrt(q))
        * template[kmin:kmax].astype(np.complex128)
        * phase(kmin, kmax, index, size).conj()
    )
    # Same rounded array is shared by both implementations and the reference.
    return (injected if noise is None else noise.astype(np.complex128) + injected).astype(
        np.complex64
    )


class NativeFilter:
    def __init__(self, template, psd, info):
        from pycbc.types import Array, FrequencySeries, zeros
        from pycbc.filter.matchedfilter import Correlator
        from pycbc.fft import IFFT

        self.kmin, self.kmax = info["kmin"], info["kmax"]
        self.size = (len(template) - 1) * 2
        self.template = FrequencySeries(template, delta_f=info["delta_f"])
        self.psd = FrequencySeries(psd, delta_f=info["delta_f"])
        self.data = Array(np.zeros_like(template))
        self.corr = zeros(self.size, dtype=np.complex64)
        self.snr = zeros(self.size, dtype=np.complex64)
        band = slice(self.kmin, self.kmax)
        self.correlator = Correlator(self.template[band], self.data[band], self.corr[band])
        self.ifft = IFFT(self.corr, self.snr)

    def apply(self, spectrum):
        np.copyto(self.data.data, spectrum)
        # Reproduce inspiral's division BEFORE multiplication (float32).
        self.data /= self.psd
        self.correlator.correlate()
        self.ifft.execute()
        return self.corr.numpy(), self.snr.numpy()


def bins_for(native, info, legacy):
    from pycbc.vetoes.chisq import power_chisq_bins, power_chisq_bins_from_sigmasq_series

    low = info["kmin"] * info["delta_f"]
    high = info["kmax"] * info["delta_f"]
    args = (native.template, native.psd, low, high)
    old_prefix = legacy["sigmasq_series"](*args)
    original = power_chisq_bins_from_sigmasq_series(
        old_prefix, info["num_bins"], info["kmin"], info["kmax"]
    )
    proposed = power_chisq_bins(native.template, info["num_bins"], native.psd, low, high)
    result = {
        key: np.asarray(value, dtype=np.int64)
        for key, value in [("original", original), ("proposed", proposed)]
    }
    for key, edges in result.items():
        require(
            np.all(np.diff(edges) > 0) and edges[0] == info["kmin"] and edges[-1] == info["kmax"],
            f"Invalid {key} bins",
        )
    return result


def new_snr(rho, chi2, count):
    reduced = float(chi2) / (2 * count - 2)
    return float(rho) * ((0.5 * (1 + reduced**3)) ** (-1 / 6) if reduced > 1 else 1)


def reference_correlation(template, spectrum, psd, kmin, kmax):
    values = np.zeros((len(template) - 1) * 2, dtype=np.complex128)
    values[kmin:kmax] = (
        template[kmin:kmax].astype(np.complex128).conj()
        * spectrum[kmin:kmax].astype(np.complex128)
        / psd[kmin:kmax].astype(np.float64)
    )
    return values


def bin_sums(correlation, edges, phasor):
    first = int(edges[0])
    return np.asarray(
        [
            np.sum(correlation[a:b].astype(CLD) * phasor[a - first : b - first], dtype=CLD)
            for a, b in zip(edges[:-1], edges[1:])
        ]
    )


def reference_statistics(correlation, native_corr, native_snr, bins, index, norm):
    """Stable physical reference and same-native-input evaluator reference."""
    edges = next(iter(bins.values()))
    phasor = phase(edges[0], edges[-1], index, len(correlation), LD)
    stats = {}
    for label, edges in bins.items():
        z = bin_sums(correlation, edges, phasor)
        total = np.sum(z, dtype=CLD)
        p = len(z)
        # Residual form avoids catastrophic subtraction for matched signals.
        chi = LD(p) * np.sum(abs((z - total / p) * LD(norm)) ** 2, dtype=LD)
        zn = bin_sums(native_corr, edges, phasor)
        # Reference to the literal native formula, holding its FFT SNR fixed.
        evaluation = (
            LD(p) * np.sum(abs(zn * LD(norm)) ** 2, dtype=LD) - abs(CLD(native_snr) * LD(norm)) ** 2
        )
        stats[label] = dict(
            rho=float(abs(total) * LD(norm)),
            chisq=float(chi),
            evaluator_chisq=float(evaluation),
            native_fft_subtraction_discrepancy=float(
                abs(np.sum(zn, dtype=CLD) * LD(norm)) ** 2 - abs(CLD(native_snr) * LD(norm)) ** 2
            ),
            newsnr=new_snr(abs(total) * LD(norm), chi, p),
        )
    return stats


def validate_null(capture, point, info, native, bins):
    from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed
    from pycbc.filter.matchedfilter import sigmasq_series

    observed_corr = capture.load(point["correlation_file"])
    observed_snr = capture.load(point["snr_file"])
    indices = capture.load(point["indices_file"])
    observed_chi = capture.load(point["chisq_file"])
    require(len(indices) > 0, "Empty captured trigger list")
    require(np.issubdtype(indices.dtype, np.unsignedinteger), "Expected unsigned capture indices")
    require(np.all(indices < native.size), "Captured indices outside FFT")
    corr, snr = native.apply(capture.load(capture.metadata["segment_spectra"][point["segment"]]))
    current = power_chisq_at_points_from_precomputed(
        native.corr, observed_snr, point["norm"], bins["proposed"], indices
    )
    prefix = sigmasq_series(
        native.template, native.psd, info["kmin"] * info["delta_f"], info["kmax"] * info["delta_f"]
    )
    checks = dict(
        correlation_exact=bool(np.array_equal(corr, observed_corr)),
        snr_exact=bool(np.array_equal(snr[indices], observed_snr)),
        chisq_exact=bool(np.array_equal(current, observed_chi)),
        bins_exact=(bins["proposed"].tolist() == point["bins"] == info["captured_current_bins"]),
        prefix_exact=bool(
            np.array_equal(prefix.numpy(), capture.load(info["current_prefix_file"]))
        ),
    )
    require(all(checks.values()), f'Capture null failed for {point["index"]}: {checks}')
    return indices, checks


def probe(capture, legacy, result, checkpoint, half_window=41):
    from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed
    from pycbc.events.ranking import newsnr

    metadata = capture.metadata
    psd = capture.load("psd-proposed.npy")
    require(np.array_equal(psd, capture.load("psd-output-0.npy")), "PSD identity failed")
    points = {(p["index"], p["segment"]): p for p in metadata["points"]}
    require(
        set(points) == set(SELECTED) and len(metadata["points"]) == len(SELECTED),
        "Unexpected capture point selection",
    )
    result.update(null_checks=[], templates=[], cases=[])
    # Complete every null check before evaluating ANY injections.
    for key in SELECTED:
        point, info = points[key], metadata["templates"][str(key[0])]
        template = capture.load(info["file"])
        require(
            template.dtype == np.complex64
            and psd.dtype == np.float32
            and template.shape == psd.shape,
            "Expected complex64 template/float32 PSD",
        )
        require(np.all(np.isfinite(template)), "Nonfinite template")
        size = (len(template) - 1) * 2
        require(
            0 < info["kmin"] < info["kmax"] == size // 2 and info["kmin"] * info["delta_f"] == 30.0,
            "Unexpected captured analysis band",
        )
        require(
            np.all(np.isfinite(psd[info["kmin"] : info["kmax"]]))
            and np.all(psd[info["kmin"] : info["kmax"]] > 0),
            "Invalid active PSD",
        )
        native = NativeFilter(template, psd, info)
        bins = bins_for(native, info, legacy)
        indices, checks = validate_null(capture, point, info, native, bins)
        segment = metadata["segments"][key[1]]
        valid = indices[
            (indices >= segment["analyze_start"] + half_window)
            & (indices < segment["analyze_stop"] - half_window)
        ]
        require(len(valid) > 0, f"No captured index with a full peak window: {key}")
        target = int(np.max(valid))
        baseline_z = complex(native.snr.numpy()[target] * point["norm"])
        baseline = {}
        for label, function in [
            ("original", legacy["power_chisq_at_points_from_precomputed"]),
            ("proposed", power_chisq_at_points_from_precomputed),
        ]:
            chi = float(
                function(
                    native.corr,
                    native.snr.numpy()[[target]],
                    point["norm"],
                    bins[label],
                    np.asarray([target], dtype=np.uint32),
                )[0]
            )
            baseline[label] = dict(
                chisq=chi, newsnr=new_snr(abs(baseline_z), chi, info["num_bins"])
            )
        result["null_checks"].append(
            dict(
                template=key[0],
                segment=key[1],
                selected_index=target,
                captured_indices=indices.tolist(),
                checks=checks,
                noise_snr_at_target=dict(
                    real=baseline_z.real,
                    imag=baseline_z.imag,
                    magnitude=abs(baseline_z),
                    variants=baseline,
                ),
            )
        )
        print(f"null exact: template={key[0]} segment={key[1]} target={target}", flush=True)
        checkpoint()
    result["null_gate_passed"] = True
    for null in result["null_checks"]:
        key = (null["template"], null["segment"])
        point, info = points[key], metadata["templates"][str(key[0])]
        template = capture.load(info["file"])
        noise = capture.load(metadata["segment_spectra"][key[1]])
        require(
            noise.dtype == np.complex64 and noise.shape == template.shape,
            "Invalid captured spectrum",
        )
        require(np.all(np.isfinite(noise)), "Nonfinite conditioned spectrum")
        native = NativeFilter(template, psd, info)
        bins = bins_for(native, info, legacy)
        target, norm = null["selected_index"], point["norm"]
        q, power = calibration(template, psd, info["kmin"], info["kmax"], info["delta_f"])
        ref_norm = LD(4) * LD(info["delta_f"]) / np.sqrt(q)
        dt = metadata["strain"]["delta_t"]
        require(
            np.isclose(dt * native.size * info["delta_f"], 1, rtol=0, atol=1e-12),
            "Inconsistent frequency/time spacing",
        )
        fractions = {
            label: np.asarray(
                [
                    np.sum(power[a - info["kmin"] : b - info["kmin"]], dtype=LD)
                    for a, b in zip(edges[:-1], edges[1:])
                ]
            )
            / np.sum(power, dtype=LD)
            for label, edges in bins.items()
        }
        result["templates"].append(
            dict(
                template=key[0],
                segment=key[1],
                template_hash=str(info["template_hash"]),
                target=target,
                gps_sample_tick=int(round(float(metadata["strain"]["epoch"]) / dt))
                + metadata["segments"][key[1]]["start"]
                + target,
                q_reference_decimal=str(q),
                captured_norm=norm,
                reference_norm=float(ref_norm),
                normalization_ratio=float(LD(norm) / ref_norm),
                bins={k: v.tolist() for k, v in bins.items()},
                bin_power_fractions={k: [float(x) for x in v] for k, v in fractions.items()},
            )
        )
        for mode in ("signal_only", "captured_noise_plus_signal"):
            for rho in RHOS:
                spectrum = injected_input(
                    template,
                    None if mode == "signal_only" else noise,
                    rho,
                    q,
                    target,
                    info["kmin"],
                    info["kmax"],
                )
                corr, snr = native.apply(spectrum)
                reference = reference_correlation(
                    template, spectrum, psd, info["kmin"], info["kmax"]
                )
                ref_snr = fft.ifft(reference, workers=1, norm="forward")
                first, last = target - half_window, target + half_window + 1
                peak = first + int(np.argmax(abs(snr[first:last])))
                ref_peak = first + int(np.argmax(abs(ref_snr[first:last])))
                selected = np.asarray(sorted({target, peak, ref_peak}), dtype=np.uint32)
                observed = {
                    "original": legacy["power_chisq_at_points_from_precomputed"](
                        native.corr, snr[selected], norm, bins["original"], selected
                    ),
                    "proposed": power_chisq_at_points_from_precomputed(
                        native.corr, snr[selected], norm, bins["proposed"], selected
                    ),
                }
                crossed = {
                    "original": power_chisq_at_points_from_precomputed(
                        native.corr, snr[selected], norm, bins["original"], selected
                    ),
                    "proposed": legacy["power_chisq_at_points_from_precomputed"](
                        native.corr, snr[selected], norm, bins["proposed"], selected
                    ),
                }
                row = dict(
                    template=key[0],
                    segment=key[1],
                    mode=mode,
                    target_rho=rho,
                    input_data_sha256=array_sha(spectrum),
                    injected_index=target,
                    peak_index=peak,
                    reference_peak_index=ref_peak,
                    peak_offset_samples=peak - target,
                    peak_offset_seconds=(peak - target) * dt,
                    reference_peak_offset_samples=ref_peak - target,
                    peak_at_window_edge=peak in (first, last - 1),
                    reference_peak_at_window_edge=ref_peak in (first, last - 1),
                    ideal_signal_only_rho_common_norm=float(LD(rho) * LD(norm) / ref_norm),
                    ideal_signal_only_chisq_at_target={
                        label: float(
                            (LD(rho) * LD(norm) / ref_norm) ** 2
                            * len(f)
                            * np.sum((f - 1 / LD(len(f))) ** 2, dtype=LD)
                        )
                        for label, f in fractions.items()
                    },
                    points=[],
                )
                for i, index in enumerate(selected):
                    refs = reference_statistics(reference, corr, snr[index], bins, int(index), norm)
                    measured_rho = float(abs(snr[index] * norm))
                    values = {}
                    for label, edges in bins.items():
                        chi = float(observed[label][i])
                        count = len(edges) - 1
                        score = float(
                            np.asarray(newsnr(measured_rho, chi / (2 * count - 2))).item()
                        )
                        require(
                            np.isfinite(chi)
                            and np.isclose(
                                score, new_snr(measured_rho, chi, count), rtol=1e-14, atol=1e-14
                            ),
                            "Invalid newSNR",
                        )
                        ref = refs[label]
                        values[label] = dict(
                            chisq=chi,
                            reduced_chisq=chi / (2 * count - 2),
                            negative_chisq=chi < 0,
                            newsnr=score,
                            reference=ref,
                            chisq_minus_reference=chi - ref["chisq"],
                            evaluator_error=chi - ref["evaluator_chisq"],
                            chisq_error_scaled_by_snr_squared=(chi - ref["chisq"])
                            / max(measured_rho**2, 1),
                            other_evaluator=dict(
                                label="proposed" if label == "original" else "original",
                                chisq=float(crossed[label][i]),
                                evaluator_error=float(crossed[label][i]) - ref["evaluator_chisq"],
                                newsnr=new_snr(measured_rho, crossed[label][i], count),
                            ),
                            newsnr_minus_reference=score - ref["newsnr"],
                        )
                    row["points"].append(
                        dict(
                            index=int(index),
                            snr=measured_rho,
                            snr_minus_reference=measured_rho - refs["proposed"]["rho"],
                            reference_snr_independent_norm=refs["proposed"]["rho"]
                            * float(ref_norm / LD(norm)),
                            reference_fft_minus_direct_snr=float(
                                abs(ref_snr[index]) * norm - refs["proposed"]["rho"]
                            ),
                            variants=values,
                        )
                    )
                if mode == "signal_only":
                    at_target = next(value for value in row["points"] if value["index"] == target)
                    row["target_calibration_relative_error"] = abs(
                        at_target["reference_snr_independent_norm"] / rho - 1
                    )
                    row["analytic_target_chisq"] = {
                        label: float(
                            LD(rho) ** 2
                            * (LD(norm) / ref_norm) ** 2
                            * info["num_bins"]
                            * np.sum((fraction - LD(1) / info["num_bins"]) ** 2, dtype=LD)
                        )
                        for label, fraction in fractions.items()
                    }
                if mode == "signal_only":
                    at_target = next(p for p in row["points"] if p["index"] == target)
                    error = at_target["reference_snr_independent_norm"] / rho - 1
                    row["calibrated_rho_fractional_error"] = error
                    require(
                        abs(error) < 5e-6 and ref_peak == target,
                        f"Signal calibration/phase reference failed: {key}, rho={rho}",
                    )
                result["cases"].append(row)
                print(
                    f'case {len(result["cases"])}/48: template={key[0]} {mode} rho={rho} '
                    f"peak_offset={peak-target} reference_offset={ref_peak-target}",
                    flush=True,
                )
                checkpoint()
    result["summary"] = summarize(result)


def summarize(result):
    summary = {}
    for mode in ("signal_only", "captured_noise_plus_signal"):
        rows = [r for r in result["cases"] if r["mode"] == mode]
        peaks = [next(p for p in r["points"] if p["index"] == r["peak_index"]) for r in rows]
        summary[mode] = dict(
            cases=len(rows),
            peaks_at_injected_index=sum(r["peak_index"] == r["injected_index"] for r in rows),
            peaks_at_reference_index=sum(
                r["peak_index"] == r["reference_peak_index"] for r in rows
            ),
            peaks_at_window_edge=sum(r["peak_at_window_edge"] for r in rows),
            max_abs_peak_offset_samples=max(abs(r["peak_offset_samples"]) for r in rows),
            max_abs_peak_snr_error=max(abs(p["snr_minus_reference"]) for p in peaks),
            max_abs_reference_fft_direct_snr_error=max(
                abs(p["reference_fft_minus_direct_snr"]) for p in peaks
            ),
            max_abs_newsnr_proposed_minus_original=max(
                abs(p["variants"]["proposed"]["newsnr"] - p["variants"]["original"]["newsnr"])
                for p in peaks
            ),
            variants={
                label: dict(
                    max_abs_chisq_error_at_peak=max(
                        abs(p["variants"][label]["chisq_minus_reference"]) for p in peaks
                    ),
                    max_abs_evaluator_error_at_peak=max(
                        abs(p["variants"][label]["evaluator_error"]) for p in peaks
                    ),
                    max_abs_newsnr_error_at_peak=max(
                        abs(p["variants"][label]["newsnr_minus_reference"]) for p in peaks
                    ),
                )
                for label in ("original", "proposed")
            },
        )
        if mode == "signal_only":
            summary[mode]["max_target_calibration_relative_error"] = max(
                r["target_calibration_relative_error"] for r in rows
            )
        if mode == "signal_only":
            summary[mode]["max_abs_calibrated_rho_fractional_error"] = max(
                abs(r["calibrated_rho_fractional_error"]) for r in rows
            )
    return summary


def write_report(path, result):
    lines = [
        "# Fixed-input injection probe",
        "",
        f"Status: {result['status']}. Capture null gate: {result['null_gate_passed']}.",
        "",
        "The PSD, conditioned spectra, injection spectrum and captured normalization are shared. "
        "Only bin construction and point chi-squared arithmetic vary. SNR timing is shared by construction.",
        "",
        "| Input | Cases | Peak at injection | Peak agrees with reference | Edge peaks | Max newSNR change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, item in result.get("summary", {}).items():
        lines.append(
            f"| {mode} | {item['cases']} | {item['peaks_at_injected_index']} | "
            f"{item['peaks_at_reference_index']} | {item['peaks_at_window_edge']} | "
            f"{item['max_abs_newsnr_proposed_minus_original']:.6g} |"
        )
    lines += [
        "",
        "All per-case target/peak values, four bin/evaluator combinations, independent "
        "references, source pins, and input hashes are in results.json. Negative raw chi-squared "
        "values are preserved. An edge peak requires wider-window follow-up before interpreting recovery.",
        "",
        "These six locations were selected from existing triggers and are not an unbiased noise sample. "
        "This checks matched-template numerics after conditioning; it does not measure FAR, detection "
        "efficiency, population sensitivity, full trigger acceptance, gating response, PSD response, "
        "or waveform/model mismatch.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="Default: CAPTURE/capture.json")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original-source", type=Path, required=True)
    parser.add_argument(
        "--source-pins",
        type=Path,
        required=True,
        help="Existing capture source-pins.json with tracked/native groups",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument(
        "--max-seconds", type=int, default=300, choices=range(30, 301), metavar="30..300"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    log_path = (args.output / "probe.log").resolve()
    log = log_path.open("x", buffering=1)
    started = time.monotonic()
    result = dict(
        status="running",
        scope="fixed proposed PSD and conditioning; isolated original/proposed "
        "bin and point-evaluator replay; not full search efficiency, FAR, or population inference",
        host=platform.node(),
        cwd=str(Path.cwd()),
        command=shlex.join([sys.executable, *sys.argv]),
        pid=os.getpid(),
        log=str(log_path),
        expected_next_check_seconds=30,
        stop_command=f"kill -TERM {os.getpid()}",
        script_sha256=sha(__file__),
        python=sys.executable,
        numpy=np.__version__,
        longdouble_mantissa_bits=np.finfo(LD).nmant,
        cpu=args.cpu,
        max_seconds=args.max_seconds,
        target_rhos=RHOS,
        source_commits=dict(original=ORIGINAL, proposed=PROPOSED),
        null_gate_passed=False,
        peak_half_window_samples=41,
        injection_phase_radians=0,
        selected_index_rule="largest captured trigger index permitting full analysis window",
    )

    def checkpoint():
        result["elapsed_seconds"] = time.monotonic() - started
        save(args.output / "results.json", result)

    def timeout(*_):
        raise TimeoutError("Probe exceeded its wall-clock bound")

    print(json.dumps(result), flush=True)
    sys.stdout = sys.stderr = log
    checkpoint()
    try:
        require(hasattr(os, "sched_setaffinity"), "Production probe requires Linux CPU affinity")
        require(args.cpu in os.sched_getaffinity(0), "Requested CPU is unavailable")
        os.sched_setaffinity(0, {args.cpu})
        signal.signal(signal.SIGALRM, timeout)
        signal.signal(signal.SIGTERM, timeout)
        signal.signal(signal.SIGINT, timeout)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
        for directory, commit in [(args.source, PROPOSED), (args.original_source, ORIGINAL)]:
            require(
                git(directory, "rev-parse", "HEAD") == commit, f"Wrong source HEAD: {directory}"
            )
            require(git(directory, "status", "--porcelain") == "", f"Dirty source: {directory}")
        source_pins = json.loads(args.source_pins.read_text())
        require(
            source_pins.get("tracked") and source_pins.get("native"),
            "Missing checked source/native pins",
        )

        def source_unchanged():
            return all(
                sha(args.source / path) == digest
                for group in ("tracked", "native")
                for path, digest in source_pins[group].items()
            )

        require(source_unchanged(), "Checked source/native pin mismatch")
        for path in ("pycbc/vetoes/chisq_cpu.pyx", "pycbc/types/array_cpu.pyx"):
            require(
                git(args.original_source, "show", f"{ORIGINAL}:{path}")
                == git(args.source, "show", f"{PROPOSED}:{path}"),
                f"Original replay requires unchanged native source: {path}",
            )
        sys.path.insert(0, str(args.source.resolve()))
        import pycbc
        from pycbc import scheme
        from pycbc.fft.backend_support import get_backend, get_backend_names, set_backend
        from pycbc.vetoes import chisq

        require(
            Path(pycbc.__file__).resolve().is_relative_to(args.source.resolve()),
            "Wrong imported PyCBC",
        )
        result["pycbc_file"] = pycbc.__file__
        result["source_pins_sha256"] = sha(args.source_pins)
        legacy, result["original_replay_sources"] = original_functions(args.original_source)
        capture = Capture(args.capture, args.metadata or args.capture / "capture.json")
        fn = chisq.power_chisq_bins
        observed = capture.metadata["sources"][fn.__module__ + "." + fn.__qualname__][
            "source_sha256"
        ]
        require(
            hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() == observed,
            "Captured bin-function source mismatch",
        )
        with scheme.CPUScheme(1):
            require("mkl" in get_backend_names(), "Checked MKL FFT backend is unavailable")
            set_backend(["mkl"])
            result["fft_backend"] = get_backend().__name__
            require(result["fft_backend"] == "pycbc.fft.mkl", "Wrong active FFT backend")
            result["cpu_affinity"] = sorted(os.sched_getaffinity(0))
            probe(capture, legacy, result, checkpoint)
        require(len(result["cases"]) == 48, "Incomplete injection case grid")
        result["input_sha256"] = capture.pins
        result["inputs_unchanged"] = capture.unchanged()
        result["source_pins_unchanged"] = source_unchanged()
        for directory, commit in [(args.source, PROPOSED), (args.original_source, ORIGINAL)]:
            require(
                git(directory, "rev-parse", "HEAD") == commit
                and git(directory, "status", "--porcelain") == "",
                "Source checkout changed during probe",
            )
        require(
            result["inputs_unchanged"] and result["source_pins_unchanged"],
            "Inputs/source changed during probe",
        )
        result["status"] = "complete"
        write_report(args.output / "report.md", result)
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        checkpoint()


if __name__ == "__main__":
    main()
