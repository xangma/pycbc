#!/usr/bin/env python3
"""Summarize per-run cProfile and perf report files without adding nested time.

Usage: python summarize-profiles.py --output profiles.json RUN_DIR [RUN_DIR ...]
Native export: perf report --stdio --no-children --call-graph none --percent-limit 0.1
  --show-nr-samples --show-total-period --sort comm,dso,symbol -i perf.data
Save that output as RUN_DIR/perf-report.txt, including its comment headers.
"""

import argparse
import hashlib
import json
import math
import pstats
import re
from collections import defaultdict
from pathlib import Path

GROUPS = ("FFT", "correlation", "threshold/clustering", "signal-consistency",
          "decompression/waveforms", "conditioning/PSD", "I/O/import/setup", "remaining")


def identity(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def classify(filename, name):
    """Classify only identifiable code ownership; shared native operations stay opaque."""
    path = "/" + filename.replace("\\", "/").lstrip("/")
    if "pycbc." in name:
        path += " /" + name[name.index("pycbc."):].replace(".", "/")
    if any(part in path for part in ("/pycbc/fft/", "/numpy/fft/", "/scipy/fft/", "/scipy/fftpack/")) or re.search(r"(?:pocketfft|_fft\.|fft_[ric]|fftpack)", name):
        return "FFT", "FFT module or named FFT operation"
    if "/pycbc/filter/matchedfilter" in path and "correlat" in name:
        return "correlation", "Named correlation function in matchedfilter backend"
    if "/pycbc/events/threshold" in path or (
        "/pycbc/events/" in path and any(s in name for s in ("threshold", "cluster"))
    ):
        return "threshold/clustering", "Threshold module or named event threshold/cluster function"
    if "/pycbc/vetoes/" in path:
        return "signal-consistency", "PyCBC vetoes module"
    if "/pycbc/waveform/" in path:
        return "decompression/waveforms", "PyCBC waveform module"
    if any(s in path for s in ("/pycbc/strain/", "/pycbc/psd/", "/pycbc/filter/resample")):
        return "conditioning/PSD", "Strain, PSD, or resampling module"
    if any(s in path for s in ("/pycbc/io/", "/h5py/", "importlib", "/argparse.py", "/logging/")) or "__import__" in name:
        return "I/O/import/setup", "I/O, import, argument parsing, or logging code"
    if "/pycbc/events/eventmgr.py" in path and name in ("write_events", "write_to_hdf", "make_output_dir", "save_state", "restore_state"):
        return "I/O/import/setup", "Named event output/checkpoint function"
    if filename == "~" or name.startswith(("<built-in", "<method")):
        return "remaining", "Opaque/shared native operation; no unique workload attribution"
    return "remaining", "Shared utility or orchestration; no unique workload attribution"


def python_profile(path, top):
    stats = pstats.Stats(str(path))
    grouped, phases = defaultdict(list), []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        group, basis = classify(filename, name)
        row = dict(file=filename, line=line, function=name, primitive_calls=primitive,
                   calls=calls, self_seconds=own, cumulative_seconds=cumulative,
                   classification_basis=basis)
        grouped[group].append(row)
        if filename.replace("\\", "/").endswith("/bin/pycbc_inspiral") and name in ("template_triggers", "<module>"):
            phases.append(row)
    total = math.fsum(row["self_seconds"] for rows in grouped.values() for row in rows)
    if not math.isclose(total, stats.total_tt, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Exclusive self-time groups do not reconcile to pstats total_tt")
    groups = []
    for name in GROUPS:
        rows = sorted(grouped[name], key=lambda row: row["self_seconds"], reverse=True)
        own = math.fsum(row["self_seconds"] for row in rows)
        groups.append(dict(group=name, self_seconds=own, share_of_profile_self_percent=100*own/total if total else 0,
                           function_count=len(rows), top_functions=rows[:top]))
    return dict(input=identity(path), total_profile_self_seconds=total, groups=groups,
                phase_functions=phases, exclusive_groups_reconcile=True,
                accounting="Each pstats tt/self time is counted exactly once. Native callees can be charged to an opaque C-call entry. Groups are code ownership, not inclusive pipeline phases.",
                phase_note="Phase cumulative times include descendants and overlap the exclusive groups. template_triggers includes one template's decompression, all segment filtering and vetoes; it is not a pure correlation timer. Do not sum phase cumulative rows.",
                timing_note="cProfile instrumentation changes elapsed time; this is not an unprofiled benchmark. CUDA launches can be asynchronous and cProfile does not measure device kernel time.")


def native_label(symbol):
    if re.search(r"(?:fftw|Dfti|pocketfft|mkl_dft|mkl_fft)", symbol, re.I):
        return "FFT symbol (symbol-name evidence)"
    if re.search(r"matchedfilter.*correlat", symbol, re.I):
        return "correlation symbol (symbol-name evidence)"
    if re.search(r"threshold.*cluster|cluster.*threshold", symbol, re.I):
        return "threshold/clustering symbol (symbol-name evidence)"
    return "unassigned native symbol; no unique pipeline attribution"


def native_profile(path, top):
    content = path.read_text(errors="replace")
    headers = [line for line in content.splitlines() if line.lstrip().startswith("#")]
    if any(re.search(r"\bChildren\b", line) for line in headers):
        raise ValueError("Export perf with --no-children; inclusive Children percentages are unsupported")
    samples = re.findall(r"^#\s*Samples:\s*(.*?)\s+of event\s+'([^']+)'", content, re.M)
    periods = re.findall(r"^#\s*Event count \(approx\.\):\s*([\d,]+)", content, re.M)
    if len(samples) != 1:
        raise ValueError("Require one perf event and its Samples header; do not merge events")
    columns = next((line for line in headers if "Overhead" in line and "Symbol" in line), "")
    if not columns:
        raise ValueError("Missing perf Overhead/Symbol header")
    has_samples, has_period = "Samples" in columns, "Period" in columns
    rows = []
    for line in content.splitlines():
        match = re.match(r"^\s*(\d+(?:\.\d+)?)%\s+(.*)$", line)
        if not match:
            continue
        rest = match[2]
        row = {"self_event_percent": float(match[1]), "raw_columns": rest}
        for enabled, field in ((has_samples, "samples"), (has_period, "event_period")):
            if enabled:
                count = re.match(r"^([\d,]+)\s+(.*)$", rest)
                if not count:
                    raise ValueError(f"Cannot parse perf {field}: {line}")
                row[field], rest = int(count[1].replace(",", "")), count[2]
        fields = re.split(r"\s{2,}", rest.strip(), maxsplit=2)
        if len(fields) != 3:
            raise ValueError(f"Expected comm,dso,symbol columns: {line}")
        row.update(command=fields[0], shared_object=fields[1], symbol=fields[2],
                   classification=native_label(fields[2]))
        rows.append(row)
    if not rows:
        raise ValueError("No native self-overhead rows found")
    rows.sort(key=lambda row: row["self_event_percent"], reverse=True)
    sample_text, event = samples[0]
    return dict(input=identity(path), event=event, samples_header=sample_text,
                sample_count_exact=int(sample_text.replace(",", "")) if re.fullmatch(r"[\d,]+", sample_text) else None,
                event_period_count_approx=int(periods[0].replace(",", "")) if len(periods) == 1 else None,
                reported_row_count=len(rows), reported_rows_percent_sum=math.fsum(row["self_event_percent"] for row in rows),
                top_native_functions=rows[:top], headers=headers,
                denominator_note="Overhead is perf's reported direct event-period percentage for this event, not seconds or necessarily sample-count share. Samples header may be rounded; exact sample_count is then null. Percent-limit omits small rows; never renormalize visible rows.",
                timing_note="CPU event sampling is separate from cProfile and full-process wall time; it does not measure CUDA device utilization or elapsed kernel time.")


def summarize_run(directory, top):
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("state") != "complete" or receipt.get("returncode") != 0:
        raise ValueError(f"Run is not successfully complete: {directory}")
    row = dict(directory=str(directory.resolve()), receipt=identity(receipt_path),
               run={key: receipt.get(key) for key in ("case", "mode", "scheme", "segment_length", "source_info", "environment")},
               full_process_wall_seconds=receipt.get("elapsed_wall_seconds"),
               wall_note="Parent run-case elapsed time includes command wrappers and profiling overhead; only mode=timing is unprofiled.")
    profile = directory / "profile.pstats"
    native = directory / "perf-report.txt"
    if profile.exists():
        row["cprofile"] = python_profile(profile, top)
    if native.exists():
        row["perf"] = native_profile(native, top)
    if not profile.exists() and not native.exists():
        raise ValueError(f"No profile.pstats or perf-report.txt in {directory}")
    triggers = directory / "triggers.hdf"
    if triggers.exists():
        try:
            import h5py
        except ImportError:
            row["internal_phase_times"] = {"available": False, "reason": "h5py unavailable"}
        else:
            with h5py.File(triggers, "r") as handle:
                cli = receipt.get("executable_cli", [])
                channel = cli[cli.index("--channel-name")+1] if "--channel-name" in cli else None
                detectors = [channel.split(":", 1)[0]] if channel else [key for key in handle if re.fullmatch(r"[A-Z][0-9]", key) and isinstance(handle[key], h5py.Group)]
                if not detectors or any(not re.fullmatch(r"[A-Z][0-9]", detector) for detector in detectors):
                    raise ValueError("Cannot identify valid detector groups for internal phase times")
                phases = []
                for detector in detectors:
                    prefix = f"{detector}/search"
                    elapsed = float(handle[f"{prefix}/run_time"][0])
                    fraction = float(handle[f"{prefix}/setup_time_fraction"][0])
                    phases.append(dict(detector=detector, run_seconds=elapsed, setup_seconds=elapsed*fraction, setup_fraction=fraction))
            row["internal_phase_times"] = dict(input=identity(triggers), detectors=phases,
                source="pycbc_inspiral tstart/tsetup/tstop; eventmgr DETECTOR/search/run_time and DETECTOR/search/setup_time_fraction",
                note="Internal run excludes imports before tstart and final event-file writing. Setup ends before template chunks. Both values are from this profiled execution, not a separate unprofiled measurement.")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    result = dict(schema_version=1, summarizer=identity(Path(__file__)),
                  runs=[summarize_run(path, args.top) for path in args.runs])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
