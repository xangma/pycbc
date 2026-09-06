#!/usr/bin/env python3
"""Validate a compressed bank against regenerated signals under qualified PSDs.

This is a scientific diagnostic, never throughput evidence. Supply one or more
successful qualify-inspiral.py receipts and the original bank/compression
receipts. Every selected template is checked under every PSD used by its run.
The default acceptance limits are declared diagnostic budgets, not a theorem
that the compression model-PSD tolerance also holds under a measured PSD.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def save(path, result):
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def tracked_source(root):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    result = {
        "root": str(root),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "status": git("status", "--porcelain", "--untracked-files=no"),
    }
    require(not result["status"], "Validation source has tracked modifications")
    return result


def checked_file(record, inputs):
    path = Path(record["path"]).resolve(strict=True)
    actual = sha256(path)
    require(actual == record["sha256"], f"Recorded input hash changed: {path}")
    inputs[str(path)] = actual
    return path


def preflight(args):
    inputs = {}
    for path in (args.bank_metadata, args.compression_receipt, Path(__file__)):
        inputs[str(path.resolve(strict=True))] = sha256(path)
    metadata = read_json(args.bank_metadata)
    compression = read_json(args.compression_receipt)
    require(
        compression["state"] == "complete" and compression["returncode"] == 0,
        "Compression did not finish successfully",
    )
    require(
        compression["input_sha256"] == compression["input_sha256_after"],
        "Compression inputs changed while running",
    )
    for path, digest in compression["input_sha256"].items():
        checked_file({"path": path, "sha256": digest}, inputs)
    require(
        compression["input_sha256"].get(str(args.bank_metadata))
        == sha256(args.bank_metadata),
        "Compression receipt does not bind the supplied bank metadata",
    )
    cases = []
    for path in args.qualification:
        require(str(path) not in inputs, f"Duplicate qualification input: {path}")
        inputs[str(path)] = sha256(path)
        qualification = read_json(path)
        require(
            qualification["status"] == "success"
            and qualification["executable_exit_code"] == 0
            and qualification["checks"]
            and all(qualification["checks"].values()),
            f"Unsuccessful qualification: {path}",
        )
        executable = checked_file(qualification["executable"], inputs)
        source = executable.parent.parent
        observed = qualification["observations"]
        require(
            len(observed["banks"]) == len(observed["segment_geometry"]) == 1,
            "Expected one qualified bank and one segment geometry",
        )
        bank_record = observed["banks"][0]
        bank = checked_file(bank_record["file"], inputs)
        require(
            compression["output_sha256"].get(str(bank)) == inputs[str(bank)],
            "Qualified bank is not a recorded output of this compression",
        )
        geometry = observed["segment_geometry"][0]
        psds = observed["psd_arrays"]
        require(psds, "No actual segment PSD arrays were saved")
        require(
            sorted(i for p in psds for i in p["segment_indices"])
            == list(range(len(geometry["segments"]))),
            "PSD mapping omits or duplicates segments",
        )
        for record in psds:
            checked_file(record, inputs)
            require(record["scaling"] == "DYN_RANGE_FAC**2", "Unsupported PSD scaling")
            require(
                record["n_samples"] == geometry["frequency_samples"]
                and record["delta_f_hz"] == geometry["delta_f_hz"],
                "PSD grid differs from run",
            )
        cases.append(
            {
                "qualification_path": str(path),
                "qualification": qualification,
                "source": source,
                "bank": bank,
                "bank_record": bank_record,
                "geometry": geometry,
                "psds": psds,
            }
        )
    require(cases, "At least one qualification is required")
    source = args.source or cases[0]["source"]
    require(
        all(case["source"] == source for case in cases),
        "All qualifications must use the supplied validation source tree",
    )
    return source, metadata, compression, cases, inputs


def array_record(np, value):
    array = np.ascontiguousarray(value)
    return {
        "data_sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
        "dtype_str": array.dtype.str,
        "shape": list(array.shape),
    }


def weighted_metrics(np, reference, decompressed, psd, delta_f):
    """Compute fixed-time metrics in complex128; no fitted amplitude or time shift."""
    ref = np.asarray(reference, dtype=np.complex128)
    comp = np.asarray(decompressed, dtype=np.complex128)
    noise = np.asarray(psd, dtype=np.float64)
    require(
        ref.ndim == 1 and ref.size > 0 and ref.shape == comp.shape == noise.shape,
        "Waveform/PSD bands must have equal nonempty one-dimensional shapes",
    )
    require(
        np.isfinite(ref).all()
        and np.isfinite(comp).all()
        and np.isfinite(noise).all()
        and (noise > 0).all(),
        "Nonfinite waveform or nonpositive/nonfinite PSD in analysis band",
    )
    weight = (4.0 * delta_f) / noise
    norm_ref_sq = float(np.sum((ref.real**2 + ref.imag**2) * weight, dtype=np.float64))
    norm_comp_sq = float(
        np.sum((comp.real**2 + comp.imag**2) * weight, dtype=np.float64)
    )
    require(
        norm_ref_sq > 0
        and norm_comp_sq > 0
        and math.isfinite(norm_ref_sq)
        and math.isfinite(norm_comp_sq),
        "Zero or nonfinite weighted waveform norm",
    )
    inner = complex(np.sum(ref.conj() * comp * weight, dtype=np.complex128))
    overlap = inner / math.sqrt(norm_ref_sq * norm_comp_sq)
    require(abs(overlap) <= 1 + 1e-10, "Weighted overlap violates Cauchy-Schwarz")
    amp_sq = float(np.sum((np.abs(comp) - np.abs(ref)) ** 2 * weight, dtype=np.float64))
    residual = comp - ref
    error_sq = float(
        np.sum((residual.real**2 + residual.imag**2) * weight, dtype=np.float64)
    )
    norm_ratio = math.sqrt(norm_comp_sq / norm_ref_sq)
    return {
        "normalized_overlap_real": overlap.real,
        "normalized_overlap_imag": overlap.imag,
        "normalized_overlap_abs": abs(overlap),
        "fixed_time_phase_optimized_mismatch": max(0.0, 1.0 - abs(overlap)),
        "fixed_time_fixed_phase_mismatch": max(0.0, 1.0 - overlap.real),
        "constant_phase_offset_radians": math.atan2(overlap.imag, overlap.real),
        "relative_weighted_amplitude_error": math.sqrt(amp_sq / norm_ref_sq),
        "relative_weighted_complex_error": math.sqrt(error_sq / norm_ref_sq),
        "reference_snr_norm": math.sqrt(norm_ref_sq),
        "decompressed_snr_norm": math.sqrt(norm_comp_sq),
        "snr_norm_ratio": norm_ratio,
        "relative_snr_norm_error": abs(norm_ratio - 1.0),
        "recovered_reference_signal_snr_fraction": abs(overlap),
    }


@contextmanager
def forbid_generation(waveform):
    original = waveform.get_waveform_filter

    def reject(*args, **kwargs):
        raise RuntimeError("Compressed waveform fell back to waveform generation")

    waveform.get_waveform_filter = reject
    try:
        yield
    finally:
        waveform.get_waveform_filter = original


def validate_case(case, metadata, result, args, modules):
    np, pycbc, waveform, bank_module, get_cutoff_indices = modules
    observed = case["qualification"]["observations"]
    options = observed["parsed_options"]
    geometry = case["geometry"]
    df, length = float(geometry["delta_f_hz"]), int(geometry["frequency_samples"])
    controllers = observed["matched_filter_controllers"]
    require(len(controllers) == 1, "Expected one qualified filtering controller")
    filter_first = int(controllers[0]["filter_bin_start"])
    filter_stop = int(controllers[0]["filter_bin_stop"])
    require(
        0 <= filter_first < filter_stop <= length, "Invalid controller frequency band"
    )
    require(geometry["fft_samples"] == 2 * (length - 1), "Invalid qualified FFT grid")
    require(
        math.isclose(
            df * geometry["fft_samples"], geometry["sample_rate_hz"], rel_tol=1e-13
        ),
        "Sample rate does not match FFT grid",
    )
    require(
        not options["max_template_length"],
        "Validator forbids changing the bank start frequency",
    )
    kwargs = dict(
        filename=str(case["bank"]),
        filter_length=length,
        delta_f=df,
        dtype=np.complex64,
        phase_order=options["order"],
        taper=options["taper_template"],
        approximant=options["approximant"],
        low_frequency_cutoff=None
        if options["enable_bank_start_frequency"]
        else options["low_frequency_cutoff"],
        waveform_decompression_method=options["waveform_decompression_method"],
    )
    compressed = bank_module.FilterBank(**kwargs, enable_compressed_waveforms=True)
    reference = bank_module.FilterBank(**kwargs, enable_compressed_waveforms=False)
    output = {
        "qualification": case["qualification_path"],
        "bank": str(case["bank"]),
        "segment_length_seconds": 1.0 / df,
        "delta_f_hz": df,
        "fft_samples": geometry["fft_samples"],
        "frequency_samples": length,
        "qualified_processing_scheme": options["processing_scheme"],
        "psds": case["psds"],
        "templates": [],
        "state": "running",
    }
    result["cases"].append(output)
    try:
        qualified_templates = case["bank_record"]["templates"]
        require(
            len(compressed) == len(reference) == len(qualified_templates),
            "Bank selection is not the complete qualified bank",
        )
        require(
            set(qualified_templates) == {str(i) for i in range(len(compressed))},
            "Qualified bank indices are not contiguous",
        )
        by_hash = {str(row["template_hash"]): row for row in metadata["templates"]}
        noises = []
        for record in case["psds"]:
            noise = np.load(record["path"], allow_pickle=False)
            descriptor = array_record(np, noise)
            require(
                noise.shape == (length,)
                and noise.dtype.kind == "f"
                and descriptor["data_sha256"] == record["data_sha256"]
                and descriptor["dtype_str"] == record["dtype_str"],
                "Saved PSD contents differ from receipt",
            )
            require(
                not np.isnan(noise).any()
                and not np.isneginf(noise).any()
                and (noise > 0).all(),
                "PSD contains NaN, -inf, or nonpositive bins",
            )
            require(
                np.isfinite(noise[filter_first:filter_stop]).all(),
                "PSD contains nonfinite bins used by the qualified controller",
            )
            require(
                float(pycbc.DYN_RANGE_FAC) == record["dyn_range_factor"],
                "PSD and regenerated waveforms have different dynamic-range factors",
            )
            noises.append(noise)
        for index in range(len(compressed)):
            template_hash = str(int(compressed.table.template_hash[index]))
            require(
                template_hash == qualified_templates[str(index)]["template_hash"],
                "Qualified template hash differs from bank row",
            )
            require(
                template_hash in by_hash,
                "Template is absent from original bank metadata",
            )
            prepared = by_hash[template_hash]
            parameters = prepared["parameters"]
            require(
                compressed.approximant(index)
                == parameters["approximant"]
                == metadata["approximant"],
                "Regeneration approximant differs from bank preparation",
            )
            for key, value in parameters.items():
                if key != "approximant":
                    require(
                        float(compressed.table[index][key]) == float(value),
                        f"Template parameter changed for {template_hash}: {key}",
                    )
            group = compressed.filehandler["compressed_waveforms"][template_hash]
            require(
                set(group) == {"amplitude", "phase", "sample_points"},
                "Compressed waveform is missing amplitude/phase/sample-point data",
            )
            attributes = {
                key: (value.item() if isinstance(value, np.generic) else value)
                for key, value in group.attrs.items()
            }
            attributes = {
                key: (value.decode() if isinstance(value, bytes) else value)
                for key, value in attributes.items()
            }
            with forbid_generation(waveform):
                comp = compressed[index]
            ref = reference[index]
            require(
                comp.dtype == ref.dtype == np.dtype("complex64"),
                "Waveforms are not CLI precision",
            )
            require(
                len(comp) == len(ref) == length
                and comp.f_lower == ref.f_lower
                and comp.end_frequency == ref.end_frequency,
                "Waveform grids or cutoffs differ",
            )
            require(
                float(comp.f_lower) == qualified_templates[str(index)]["f_lower_hz"],
                "Validation frequency cutoff differs from executed template",
            )
            first, stop = get_cutoff_indices(
                comp.f_lower, comp.end_frequency, df, geometry["fft_samples"]
            )
            first, stop = max(first, filter_first), min(stop, filter_stop)
            require(
                first < stop, "Template and controller frequency bands do not overlap"
            )
            comp_values, ref_values = comp.numpy(), ref.numpy()
            require(
                np.isfinite(comp_values).all() and np.isfinite(ref_values).all(),
                "Nonfinite values outside the scored waveform band",
            )
            row = {
                "index": index,
                "template_hash": template_hash,
                "main_bank_row_index": prepared["row_index"],
                "parameters": parameters,
                "compression_attributes": attributes,
                "f_lower_hz": float(comp.f_lower),
                "f_end_hz": float(comp.end_frequency),
                "frequency_index_slice": [first, stop],
                "decompression_succeeded_without_generation": True,
                "reference": array_record(np, ref_values),
                "decompressed": array_record(np, comp_values),
                "psd_results": [],
            }
            for psd_index, noise in enumerate(noises):
                metrics = weighted_metrics(
                    np,
                    ref_values[first:stop],
                    comp_values[first:stop],
                    noise[first:stop],
                    df,
                )
                passed = all(
                    metrics[key] <= limit for key, limit in result["tolerances"].items()
                )
                row["psd_results"].append(
                    {"psd_index": psd_index, "metrics": metrics, "passed": passed}
                )
                result["completed_template_psd_pairs"] += 1
            row["passed"] = all(item["passed"] for item in row["psd_results"])
            output["templates"].append(row)
            save(args.output, result)
            print(
                json.dumps(
                    {
                        "qualification": case["qualification_path"],
                        "template": index,
                        "template_hash": template_hash,
                        "passed": row["passed"],
                    }
                ),
                flush=True,
            )
        output["state"] = "complete"
        output["passed"] = all(row["passed"] for row in output["templates"])
    finally:
        compressed.file.close()
        reference.file.close()


def run(args):
    require(not args.output.exists(), f"Refusing existing output: {args.output}")
    source, metadata, compression, cases, inputs = preflight(args)
    limits = {
        "fixed_time_phase_optimized_mismatch": args.max_mismatch,
        "relative_weighted_amplitude_error": args.max_amplitude_error,
        "relative_snr_norm_error": args.max_snr_norm_error,
    }
    require(
        all(math.isfinite(value) and value >= 0 for value in limits.values()),
        "Acceptance tolerances must be finite and nonnegative",
    )
    source_before = tracked_source(source)
    result = {
        "schema_version": 1,
        "state": "planned" if args.plan_only else "running",
        "purpose": "Same-approximant compressed waveform validation under actual qualified run PSDs",
        "started_utc": utc(),
        "host": platform.node(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "command": [sys.executable, *sys.argv],
        "source_before": source_before,
        "input_sha256": inputs,
        "tolerances": limits,
        "method": {
            "precision": "CLI complex64 waveforms; complex128/float64 accumulation",
            "inner_product": "4*delta_f*sum(conj(reference)*decompressed/PSD) over PyCBC cutoff indices",
            "alignment": "Fixed coalescence time; no time shift or amplitude fit. Absolute overlap optimizes only a constant phase.",
            "amplitude_error": "sqrt(sum((abs(decompressed)-abs(reference))**2/PSD)/sum(abs(reference)**2/PSD))",
            "snr_norm_error": "abs(sqrt(<decompressed,decompressed>/<reference,reference>)-1)",
            "scaling": "Use saved segment PSD and FilterBank DYN_RANGE_FAC scaling without rescaling",
            "acceptance": "Explicit diagnostic budgets, independently applied to every template/PSD pair; not inferred from model-PSD compression tolerance",
            "scope": "No throughput measurement, time-domain edge-containment proof, tidal/disruption physics validation, or bank coverage claim",
        },
        "compression_command": compression["command"],
        "validation_scheme": "CPUScheme(1)",
        "expected_template_psd_pairs": sum(
            len(c["bank_record"]["templates"]) * len(c["psds"]) for c in cases
        ),
        "completed_template_psd_pairs": 0,
        "cases": [],
    }
    if args.plan_only:
        result["planned_cases"] = [
            {
                "qualification": c["qualification_path"],
                "fft_seconds": 1.0 / c["geometry"]["delta_f_hz"],
                "templates": len(c["bank_record"]["templates"]),
                "psds": len(c["psds"]),
            }
            for c in cases
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
        import pycbc
        from pycbc import waveform
        from pycbc.filter.matchedfilter import get_cutoff_indices
        from pycbc.scheme import CPUScheme
        from pycbc.waveform import bank as bank_module

        require(
            Path(pycbc.__file__).resolve().parent.parent == source,
            "Imported wrong PyCBC tree",
        )
        result["runtime"] = {
            "python": sys.version,
            "numpy": np.__version__,
            "pycbc": pycbc.__version__,
            "dynamic_range_factor": float(pycbc.DYN_RANGE_FAC),
            "environment": {
                key: value
                for key, value in os.environ.items()
                if key.startswith(("OMP_", "MKL_", "OPENBLAS_", "PYCBC_"))
            },
        }
        result["validation_modules"] = {}
        for name in (
            "pycbc",
            "pycbc.waveform.bank",
            "pycbc.waveform.compress",
            "pycbc.waveform.waveform",
            "pycbc.waveform.decompress_cpu",
            "pycbc.waveform.decompress_cpu_cython",
            "pycbc.filter.matchedfilter",
        ):
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            record = {"path": str(path), "sha256": sha256(path)}
            for case in cases:
                recorded = case["qualification"]["source_modules"].get(name)
                require(
                    recorded is not None and recorded["sha256"] == record["sha256"],
                    f"Validation module differs from qualified execution: {name}",
                )
            result["validation_modules"][name] = record
            inputs[str(path)] = record["sha256"]
        result["waveform_dependencies"] = {}
        for name, module in list(sys.modules.items()):
            if name in ("lal", "lalsimulation") or name.startswith(
                ("lal._", "lalsimulation._")
            ):
                filename = getattr(module, "__file__", None)
                if filename and Path(filename).is_file():
                    path = Path(filename).resolve()
                    record = {
                        "path": str(path),
                        "sha256": sha256(path),
                        "version": str(getattr(module, "__version__", "unavailable")),
                    }
                    prepared = metadata["provenance"]["modules"].get(name)
                    if prepared is not None:
                        require(
                            prepared["sha256"] == record["sha256"],
                            f"Waveform dependency changed since bank preparation: {name}",
                        )
                    result["waveform_dependencies"][name] = record
                    inputs[str(path)] = record["sha256"]
        with CPUScheme(1):
            for case in cases:
                validate_case(
                    case,
                    metadata,
                    result,
                    args,
                    (np, pycbc, waveform, bank_module, get_cutoff_indices),
                )
        require(
            result["completed_template_psd_pairs"]
            == result["expected_template_psd_pairs"],
            "Not every selected template/PSD pair was validated",
        )
        result["state"] = "complete"
        result["passed"] = all(case["passed"] for case in result["cases"])
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
            result["input_sha256_after"] = {path: sha256(path) for path in inputs}
            require(
                result["input_sha256_after"] == inputs,
                "Validation inputs changed while running",
            )
            result["source_after"] = tracked_source(source)
            require(
                result["source_after"] == source_before,
                "Validation source changed while running",
            )
        except BaseException as error:
            result.update(state="failed", passed=False, provenance_error=str(error))
        result["finished_utc"] = utc()
        save(args.output, result)
    return 0 if result.get("passed") and result["state"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, action="append", required=True)
    parser.add_argument("--bank-metadata", type=Path, required=True)
    parser.add_argument("--compression-receipt", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mismatch", type=float, default=1e-3)
    parser.add_argument("--max-amplitude-error", type=float, default=1e-2)
    parser.add_argument("--max-snr-norm-error", type=float, default=1e-2)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Verify receipts/hashes; do not import PyCBC or write outputs",
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
