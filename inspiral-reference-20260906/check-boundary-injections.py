#!/usr/bin/env python3
"""Compare circular segment filtering with a shared-kernel linear reference.

Diagnostic only: noiseless, sample-aligned injections, with no strain conditioning,
gating, trigger clustering, chi-square calculation, or throughput measurement.
The finite long reference is assessed by its observed edge energy, not claimed
to prove that the physical waveform/filter has no tails beyond that interval.
"""

import argparse
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback


VALIDATOR_SHA256 = "9c761995585d27364d97960fd74f2efb3b64f7e978178f348072901b6f4489c6"
LENGTHS = (256, 512, 1024)
START_PADS = (96, 112)
END_PAD = 16
LONG_LENGTH = 4096


def load_validator():
    import hashlib

    path = Path(__file__).resolve().with_name("validate-waveforms.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != VALIDATOR_SHA256:
        raise ValueError("Validator helper differs from the reviewed version")
    spec = importlib.util.spec_from_file_location("waveform_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fraction(np, values, mask):
    energy = values.real**2 + values.imag**2 if np.iscomplexobj(values) else values**2
    total = float(np.sum(energy, dtype=np.float64))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Zero or nonfinite energy in reference signal/kernel")
    return float(np.sum(energy[mask], dtype=np.float64) / total)


def compare_segment(
    np,
    fft,
    signal,
    transfer,
    norm,
    reference_window,
    reference_lag,
    count,
    start,
    end,
    offset,
    radius,
):
    """Use an exact coarse-grid sample of the common long transfer function."""
    long_count = signal.size
    if long_count % count:
        raise ValueError("Segment length does not divide the long reference")
    # Coalescence stays sample aligned; put the measured linear-reference peak,
    # including any compression-induced shift, at the requested retained sample.
    first = long_count // 2 - offset + reference_lag
    segment = signal[first : first + count]
    if first < 0 or segment.size != count:
        raise ValueError("Injection segment extends beyond the finite long signal")
    circular = (
        4
        * fft.ifft(
            fft.fft(segment, workers=1) * transfer[:: long_count // count], workers=1
        )
        / norm
    )
    retained_start, retained_stop = start, count - end
    left, right = (
        max(-radius, retained_start - offset),
        min(radius + 1, retained_stop - offset),
    )
    if left >= right:
        raise ValueError("No retained samples around the requested peak")
    expected = reference_window[radius + left : radius + right]
    actual = circular[offset + left : offset + right]
    reference_peak = float(np.max(np.abs(reference_window)))
    reference_index = int(np.argmax(np.abs(expected)))
    actual_index = int(np.argmax(np.abs(actual)))
    expected_peak, actual_peak = (
        float(abs(expected[reference_index])),
        float(abs(actual[actual_index])),
    )
    return {
        "injection_coalescence_sample": offset - reference_lag,
        "target_reference_peak_sample": offset,
        "retained_samples": [retained_start, retained_stop],
        "comparison_samples": [offset + left, offset + right],
        "compared_sample_count": right - left,
        "maximum_relative_complex_error": float(
            np.max(np.abs(actual - expected)) / reference_peak
        ),
        "rms_relative_complex_error": float(
            np.sqrt(np.mean(np.abs(actual - expected) ** 2)) / reference_peak
        ),
        "reference_peak_sample": offset + left + reference_index,
        "circular_peak_sample": offset + left + actual_index,
        "peak_location_error_samples": abs(actual_index - reference_index),
        "reference_peak_snr": expected_peak,
        "circular_peak_snr": actual_peak,
        "relative_peak_snr_error": abs(actual_peak / expected_peak - 1),
        "relative_complex_error_at_reference_peak": float(
            abs(actual[reference_index] - expected[reference_index]) / reference_peak
        ),
    }


def build_reference(
    np, fft, scipy_signal, helpers, modules, case, metadata, index, args
):
    pycbc, waveform, bank_module = modules
    observed = case["qualification"]["observations"]
    options = observed["parsed_options"]
    rate = int(case["geometry"]["sample_rate_hz"])
    count, df = LONG_LENGTH * rate, 1 / LONG_LENGTH
    length = count // 2 + 1
    controller = observed["matched_filter_controllers"][0]
    psd_record = case["psds"][0]
    psd = np.load(psd_record["path"], allow_pickle=False)
    descriptor = helpers.array_record(np, psd)
    helpers.require(
        descriptor["data_sha256"] == psd_record["data_sha256"]
        and descriptor["dtype_str"] == psd_record["dtype_str"]
        and psd.shape == (case["geometry"]["frequency_samples"],),
        "PSD array differs from receipt",
    )
    helpers.require(
        psd_record["dyn_range_factor"] == float(pycbc.DYN_RANGE_FAC),
        "PSD scaling differs from waveform scaling",
    )
    first, stop = (
        int(controller["filter_bin_start"]),
        int(controller["filter_bin_stop"]),
    )
    helpers.require(
        not np.isnan(psd).any()
        and not np.isneginf(psd).any()
        and (psd > 0).all()
        and np.isfinite(psd[first:stop]).all(),
        "Invalid measured PSD",
    )
    ratio = LONG_LENGTH * case["geometry"]["delta_f_hz"]
    helpers.require(ratio == int(ratio), "Long and measured PSD grids are not nested")
    low, high = first * int(ratio), stop * int(ratio)
    helpers.require(
        0 <= low < high <= length - 1,
        "Common controller band extends beyond the long frequency grid",
    )
    inverse = np.zeros(length, dtype=np.float64)
    # Interpolate inverse PSD once. All shorter segments use this identical
    # weighting function; their own PSD estimation/truncation is not repeated.
    inverse[low:high] = np.interp(
        np.arange(low, high, dtype=np.float64) / ratio,
        np.arange(psd.size, dtype=np.float64),
        1 / psd.astype(np.float64),
    )
    kwargs = dict(
        filename=str(case["bank"]),
        filter_length=length,
        delta_f=df,
        dtype=np.complex64,
        phase_order=options["order"],
        taper=options["taper_template"],
        approximant=options["approximant"],
        low_frequency_cutoff=options["low_frequency_cutoff"],
        waveform_decompression_method=options["waveform_decompression_method"],
    )
    compressed = bank_module.FilterBank(**kwargs, enable_compressed_waveforms=True)
    regenerated = bank_module.FilterBank(**kwargs, enable_compressed_waveforms=False)
    try:
        template_hash = str(int(compressed.table.template_hash[index]))
        row = next(
            row
            for row in metadata["templates"]
            if str(row["template_hash"]) == template_hash
        )
        helpers.require(
            compressed.approximant(index)
            == row["parameters"]["approximant"]
            == metadata["approximant"],
            "Injection and filter approximants differ",
        )
        for key, value in row["parameters"].items():
            if key != "approximant":
                helpers.require(
                    float(compressed.table[index][key]) == float(value),
                    f"Template parameter changed: {key}",
                )
        with helpers.forbid_generation(waveform):
            hcomp = compressed[index]
        href = regenerated[index]
        helpers.require(
            len(hcomp) == len(href) == length
            and hcomp.delta_f == href.delta_f == df
            and hcomp.f_lower == href.f_lower
            and hcomp.end_frequency == href.end_frequency
            and hcomp.end_idx == href.end_idx,
            "Injection and filter grids or cutoffs differ",
        )
        comp, ref = (
            np.asarray(hcomp.numpy(), dtype=np.complex128),
            np.asarray(href.numpy(), dtype=np.complex128),
        )
        helpers.require(
            np.isfinite(comp).all() and np.isfinite(ref).all(),
            "Nonfinite long waveform",
        )
        # The arrays must cover the controller grid, but a waveform may end
        # below its upper cutoff and legitimately be zero at higher frequencies.
        helpers.require(
            hcomp.f_lower == options["low_frequency_cutoff"]
            and low < min(high, hcomp.end_idx) <= length - 1,
            "Waveform start changed or its band misses the common controller band",
        )
        norm = math.sqrt(
            float(4 * df * np.sum(np.abs(comp) ** 2 * inverse, dtype=np.float64))
        )
        helpers.require(
            math.isfinite(norm) and norm > 0, "Invalid common SNR normalization"
        )
        transfer = np.zeros(count, dtype=np.complex128)
        transfer[:length] = comp.conj() * inverse
        kernel = fft.fftshift(4 * fft.ifft(transfer, workers=1))
        # A real strain injection with the regenerated positive-frequency
        # waveform. irfft / delta_t supplies the continuous-FT normalization.
        injection = fft.fftshift(fft.irfft(ref, n=count, workers=1)) * rate
        lag = np.arange(count, dtype=np.int64) - count // 2
        edge = np.abs(lag) >= count * 7 // 16  # outer 256 s at each end for T=4096
        tail = {
            "edge_region_absolute_lag_seconds_at_least": LONG_LENGTH * 7 / 16,
            "injection_edge_energy_fraction": fraction(np, injection, edge),
            "kernel_edge_energy_fraction": fraction(np, kernel, edge),
            "kernel_energy_outside_padding": {
                str(start): fraction(
                    np, kernel, (lag < -END_PAD * rate) | (lag > start * rate)
                )
                for start in START_PADS
            },
            "interpretation": "Observed energy inside the finite long reference. It cannot bound unobserved tails beyond +/-2048 s.",
        }
        provenance = {
            "bank_index": index,
            "template_hash": template_hash,
            "main_bank_row_index": row["row_index"],
            "parameters": row["parameters"],
            "long_frequency_samples": length,
            "long_samples": count,
            "delta_f_hz": df,
            "filter_frequency_index_slice": [low, high],
            "waveform_cutoffs_hz": [float(hcomp.f_lower), float(hcomp.end_frequency)],
            "filter_snr_norm": norm,
            "regenerated_frequency_waveform": helpers.array_record(np, ref),
            "compressed_frequency_waveform": helpers.array_record(np, comp),
            "common_transfer_function": helpers.array_record(np, transfer),
            "common_inverse_psd": helpers.array_record(np, inverse),
            "tails": tail,
        }
        del ref, comp, inverse, hcomp, href, lag, edge
        # Both arrays have first sample at lag -M/2. Their linear convolution
        # therefore has first sample at lag -M, and lag zero at index M.
        with fft.set_workers(1):
            linear = scipy_signal.fftconvolve(injection, kernel, mode="full") / norm
        del kernel
        peak = int(np.argmax(np.abs(linear)))
        peak_lag = peak - count
        radius = rate  # one second, matching the workload's cluster window
        helpers.require(
            abs(peak_lag) <= radius and peak > radius and peak + radius < linear.size,
            "Long reference peak is not near the expected coalescence time",
        )
        scale = args.target_snr / float(abs(linear[peak]))
        injection *= scale
        reference_window = linear[peak - radius : peak + radius + 1].copy() * scale
        del linear
        provenance.update(
            injection_amplitude_scale=scale,
            reference_peak_lag_samples=peak_lag,
            normalized_injection=helpers.array_record(np, injection),
            reference_peak_window=helpers.array_record(np, reference_window),
            reference_peak_snr=float(abs(reference_window[radius])),
        )
        return injection, transfer, norm, reference_window, peak_lag, provenance
    finally:
        compressed.file.close()
        regenerated.file.close()


def run(args):
    helpers = load_validator()
    helpers.require(
        not args.output.exists(), f"Refusing existing output: {args.output}"
    )
    source, metadata, compression, cases, inputs = helpers.preflight(args)
    helpers.require(
        len(cases) == 1 and len(cases[0]["psds"]) == 1,
        "Supply one successful L1024 qualification with one shared PSD",
    )
    case = cases[0]
    helpers.require(
        case["geometry"]["delta_f_hz"] == 1 / 1024,
        "Common weighting PSD must come from L1024",
    )
    options = case["qualification"]["observations"]["parsed_options"]
    helpers.require(
        not options["max_template_length"]
        and not options["enable_bank_start_frequency"],
        "Boundary test requires the fixed original waveform start frequency",
    )
    inputs[str(Path(__file__).resolve())] = helpers.sha256(__file__)
    source_before = helpers.tracked_source(source)
    selected = case["bank_record"]["templates"]
    bank_indices = {
        record["template_hash"]: int(index) for index, record in selected.items()
    }
    ordered = sorted(
        metadata["templates"], key=lambda row: row["parameters"]["template_duration"]
    )
    chosen = [("longest", ordered[-1]), ("shortest", ordered[0])]
    helpers.require(
        all(str(row["template_hash"]) in bank_indices for _, row in chosen),
        "Qualified bank does not contain both longest and shortest templates",
    )
    limits = {
        "maximum_relative_complex_error": args.max_complex_error,
        "relative_peak_snr_error": args.max_peak_snr_error,
        "peak_location_error_samples": args.max_peak_location_error_samples,
        "edge_energy_fraction": args.max_edge_energy_fraction,
    }
    helpers.require(
        all(math.isfinite(value) and value >= 0 for value in limits.values())
        and math.isfinite(args.target_snr)
        and args.target_snr > 0,
        "Invalid diagnostic tolerances or SNR",
    )
    result = {
        "schema_version": 1,
        "state": "planned" if args.plan_only else "running",
        "purpose": "Noiseless boundary-injection diagnostic with one common long weighting kernel",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "command": [sys.executable, *sys.argv],
        "source_before": source_before,
        "input_sha256": inputs,
        "tolerances": limits,
        "qualification": case["qualification_path"],
        "common_measured_psd": case["psds"][0],
        "compression_command": compression["command"],
        "target_reference_peak_snr": args.target_snr,
        "long_reference_seconds": LONG_LENGTH,
        "segment_lengths_seconds": list(LENGTHS),
        "start_pads_seconds": list(START_PADS),
        "end_pad_seconds": END_PAD,
        "expected_cases": len(chosen) * len(LENGTHS) * len(START_PADS) * 3,
        "completed_cases": 0,
        "templates": [],
        "method": {
            "waveforms": "Same-approximant complex64 FilterBank regeneration for the real injection; compressed FilterBank template with generation fallback forbidden for the filter",
            "arithmetic": "float64/complex128 diagnostic FFTs and linear convolution; common normalization across lengths",
            "weighting": "One inverse PSD, linearly interpolated once from the actual final L1024 PSD onto the 4096-s grid, zero outside the actual controller band",
            "kernel": "4*IFFT(conj(compressed_H)*inverse_PSD), with only the positive-frequency band populated; unwrapped lags [-2048,2048) seconds",
            "circular": "4*IFFT(FFT(injection_segment)*common_transfer[::4096/L])/common_sigma; exact periodization of the same long kernel",
            "linear": "Full zero-padded linear convolution of the finite unwrapped injection and kernel, divided by common_sigma",
            "placement": "Measured linear-reference peak at first valid sample, integer midpoint, and last valid sample; sample-aligned coalescence",
            "comparison": "All retained samples within one second of the reference peak. Complex errors use the common reference peak amplitude as denominator; no fitted phase, time, or segment-dependent amplitude",
            "tails": "Both outer-reference energy fractions must meet the stated budget; padding-tail energies are reported separately, without discarding kernel samples",
            "limits": "Finite-reference diagnostic, not proof of waveform support. Does not test production FFT rounding, L-dependent PSD differences, real-noise interactions, conditioning, gating, chi-square, or trigger clustering",
        },
    }
    if args.plan_only:
        result["selected_templates"] = [
            {
                "kind": kind,
                "template_hash": str(row["template_hash"]),
                "bank_index": bank_indices[str(row["template_hash"])],
            }
            for kind, row in chosen
        ]
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))
        import numpy as np
        import scipy
        from scipy import fft, signal as scipy_signal
        import pycbc
        from pycbc import waveform
        from pycbc.scheme import CPUScheme
        from pycbc.waveform import bank as bank_module

        helpers.require(
            Path(pycbc.__file__).resolve().parent.parent == source,
            "Imported wrong PyCBC tree",
        )
        result["runtime"] = {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pycbc": pycbc.__version__,
            "scheme": "CPUScheme(1)",
            "fft_workers": 1,
        }
        result["source_modules"] = {}
        for name in (
            "pycbc",
            "pycbc.waveform.bank",
            "pycbc.waveform.waveform",
            "pycbc.waveform.compress",
            "pycbc.waveform.decompress_cpu",
            "pycbc.waveform.decompress_cpu_cython",
        ):
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            digest = helpers.sha256(path)
            helpers.require(
                case["qualification"]["source_modules"][name]["sha256"] == digest,
                f"Source module differs from qualification: {name}",
            )
            result["source_modules"][name] = {"path": str(path), "sha256": digest}
            inputs[str(path)] = digest
        for name, module in list(sys.modules.items()):
            if name in ("lal", "lalsimulation") or name.startswith(
                ("lal._", "lalsimulation._")
            ):
                filename = getattr(module, "__file__", None)
                if filename and Path(filename).is_file():
                    path = Path(filename).resolve()
                    digest = helpers.sha256(path)
                    prepared = metadata["provenance"]["modules"].get(name)
                    helpers.require(
                        prepared is None or prepared["sha256"] == digest,
                        f"Waveform dependency changed since preparation: {name}",
                    )
                    result["source_modules"][name] = {
                        "path": str(path),
                        "sha256": digest,
                    }
                    inputs[str(path)] = digest
        rate = int(case["geometry"]["sample_rate_hz"])
        with CPUScheme(1):
            for kind, selected_row in chosen:
                index = bank_indices[str(selected_row["template_hash"])]
                injection, transfer, norm, ref, lag, provenance = build_reference(
                    np,
                    fft,
                    scipy_signal,
                    helpers,
                    (pycbc, waveform, bank_module),
                    case,
                    metadata,
                    index,
                    args,
                )
                row = {"kind": kind, "reference": provenance, "cases": []}
                result["templates"].append(row)
                tails = provenance["tails"]
                row["reference_edge_budget_passed"] = all(
                    tails[key] <= args.max_edge_energy_fraction
                    for key in (
                        "injection_edge_energy_fraction",
                        "kernel_edge_energy_fraction",
                    )
                )
                for length in LENGTHS:
                    for start_pad in START_PADS:
                        count, start, end = (
                            length * rate,
                            start_pad * rate,
                            END_PAD * rate,
                        )
                        offsets = (
                            ("first_valid", start),
                            ("midpoint", (start + count - end - 1) // 2),
                            ("last_valid", count - end - 1),
                        )
                        for placement, offset in offsets:
                            values = compare_segment(
                                np,
                                fft,
                                injection,
                                transfer,
                                norm,
                                ref,
                                lag,
                                count,
                                start,
                                end,
                                offset,
                                rate,
                            )
                            values.update(
                                segment_length_seconds=length,
                                start_pad_seconds=start_pad,
                                end_pad_seconds=END_PAD,
                                placement=placement,
                            )
                            values["passed"] = row[
                                "reference_edge_budget_passed"
                            ] and all(
                                values[key] <= limit
                                for key, limit in limits.items()
                                if key != "edge_energy_fraction"
                            )
                            row["cases"].append(values)
                            result["completed_cases"] += 1
                            helpers.save(args.output, result)
                            print(
                                json.dumps(
                                    {
                                        "template": kind,
                                        "length": length,
                                        "start_pad": start_pad,
                                        "placement": placement,
                                        "passed": values["passed"],
                                        "relative_complex_error": values[
                                            "maximum_relative_complex_error"
                                        ],
                                    }
                                ),
                                flush=True,
                            )
                row["passed"] = all(item["passed"] for item in row["cases"])
                del injection, transfer, ref
        helpers.require(
            result["completed_cases"] == result["expected_cases"],
            "Boundary matrix is incomplete",
        )
        result.update(
            state="complete", passed=all(row["passed"] for row in result["templates"])
        )
    except BaseException as error:
        result.update(
            state="failed",
            passed=False,
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        try:
            result["input_sha256_after"] = {
                path: helpers.sha256(path) for path in inputs
            }
            helpers.require(
                result["input_sha256_after"] == inputs,
                "Inputs changed during boundary diagnostic",
            )
            result["source_after"] = helpers.tracked_source(source)
            helpers.require(
                result["source_after"] == source_before,
                "Source changed during boundary diagnostic",
            )
        except BaseException as error:
            result.update(state="failed", passed=False, provenance_error=str(error))
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        helpers.save(args.output, result)
    return 0 if result.get("passed") and result["state"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, action="append", required=True)
    parser.add_argument("--bank-metadata", type=Path, required=True)
    parser.add_argument("--compression-receipt", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-snr", type=float, default=8.0)
    parser.add_argument("--max-complex-error", type=float, default=1e-3)
    parser.add_argument("--max-peak-snr-error", type=float, default=1e-3)
    parser.add_argument("--max-peak-location-error-samples", type=int, default=1)
    parser.add_argument("--max-edge-energy-fraction", type=float, default=1e-6)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Verify receipts/hashes; no PyCBC import or output",
    )
    args = parser.parse_args()
    for key in ("bank_metadata", "compression_receipt", "source", "output"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    args.qualification = [
        path.expanduser().resolve(strict=True) for path in args.qualification
    ]
    try:
        return run(args)
    except Exception as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
