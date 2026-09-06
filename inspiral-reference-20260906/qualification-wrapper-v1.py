#!/usr/bin/env python3
"""Untimed diagnostics around an unchanged bin/pycbc_inspiral.

Usage: python qualify-inspiral.py --receipt receipt.json -- /path/to/bin/pycbc_inspiral ...
The instrumented executable's own performance fields are not benchmark timings.
Use a fresh output path: resumed or partially filtered banks fail qualification.
"""

import argparse
import functools
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import runpy
import sys
import tempfile
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def qualified_name(value):
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def file_record(filename):
    path = Path(filename).resolve()
    result = {"path": str(path)}
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result.update(sha256=digest.hexdigest(), bytes=path.stat().st_size)
    except OSError as error:
        result["error"] = str(error)
    return result


def slice_record(value):
    return {
        key: None if getattr(value, key) is None else int(getattr(value, key))
        for key in ("start", "stop", "step")
    }


def json_safe(value):
    """Keep diagnostic receipts writable even when options contain NumPy scalars."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if type(value).__module__.startswith("numpy"):
        return json_safe(value.tolist())
    return {"python_type": qualified_name(value), "repr": repr(value)}


def array_values(value):
    import numpy

    if hasattr(value, "numpy"):
        value = value.numpy()
    return numpy.ascontiguousarray(value)


def array_record(value):
    values = array_values(value)
    return values, {
        "data_sha256": hashlib.sha256(memoryview(values).cast("B")).hexdigest(),
        "dtype": str(values.dtype),
        "dtype_str": values.dtype.str,
        "shape": list(values.shape),
        "n_samples": int(values.size),
        "nbytes": int(values.nbytes),
        "byte_order": "C order; dtype_str records endianness",
    }


def scheme_snapshot():
    scheme = sys.modules.get("pycbc.scheme")
    if scheme is None:
        return None
    state = scheme.mgr.state
    return {
        "class": qualified_name(state),
        "num_threads": getattr(state, "num_threads", None),
        "device": str(getattr(state, "device", None)),
    }


def union_intervals(intervals):
    merged = []
    for start, stop in sorted(intervals):
        if stop <= start:
            raise ValueError("Empty or reversed analysis interval")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(stop, merged[-1][1])
        else:
            merged.append([start, stop])
    return merged


def runtime_snapshot():
    result = {"pid": os.getpid()}
    result["environment"] = {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith(("OMP_", "MKL_", "OPENBLAS_", "NUMEXPR_", "PYCBC_", "BLIS_"))
        or name in ("CUDA_VISIBLE_DEVICES", "VECLIB_MAXIMUM_THREADS", "PYTHONHASHSEED")
    }
    try:
        result["affinity"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as error:
        result["affinity"] = None
        result["affinity_error"] = str(error)
    try:
        from threadpoolctl import threadpool_info

        result["threadpools"] = threadpool_info()
    except Exception as error:
        result["threadpools"] = None
        result["threadpool_error"] = f"{type(error).__name__}: {error}"
    try:
        maps = Path("/proc/self/maps").read_text().splitlines()
        result["loaded_shared_libraries"] = sorted(
            {
                line.split(maxsplit=5)[5]
                for line in maps
                if len(line.split(maxsplit=5)) == 6
                and ".so" in line.split(maxsplit=5)[5]
            }
        )
        result["loaded_shared_libraries_source"] = "/proc/self/maps"
    except OSError as error:
        result["loaded_shared_libraries"] = None
        result["loaded_shared_libraries_error"] = str(error)
    torch = sys.modules.get("torch")
    if torch is not None:
        result["torch"] = {
            "version": str(torch.__version__),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
        }
    result["scheme"] = scheme_snapshot()
    return result


class Qualification:
    def __init__(self, receipt, receipt_path):
        self.receipt = receipt
        self.receipt_path = receipt_path
        self.patches = []
        self.banks = []
        self.bank_ids = {}
        self.current_template = None
        self.current_fft = None
        self.fft_ids = {}
        self.active_decompression = None
        self.data = {
            "banks": [],
            "segment_geometry": [],
            "conditioned_strain": [],
            "psd_arrays": [],
            "matched_filter_controllers": [],
            "fft_engines": [],
            "fft_routes": {},
            "decompression_calls": [],
            "waveform_generation_attempts": [],
            "fallback_guard_targets": [],
            "runtime": {},
        }
        receipt["observations"] = self.data

    def patch(self, owner, name, replacement):
        # Restore inherited methods by removing the temporary override.
        owned = name in vars(owner)
        original = getattr(owner, name)
        self.patches.append((owner, name, original, owned))
        setattr(owner, name, replacement)
        return original

    def restore(self):
        for owner, name, original, owned in reversed(self.patches):
            if owned:
                setattr(owner, name, original)
            else:
                delattr(owner, name)

    def bank_record(self, bank):
        key = id(bank)
        if key not in self.bank_ids:
            index = len(self.banks)
            self.bank_ids[key] = index
            self.banks.append(bank)
            self.data["banks"].append(
                {
                    "id": index,
                    "class": qualified_name(bank),
                    "file": file_record(bank.filename),
                    "has_compressed_waveforms": bool(bank.has_compressed_waveforms),
                    "enable_compressed_waveforms": bool(
                        bank.enable_compressed_waveforms
                    ),
                    "templates": {},
                }
            )
        return self.data["banks"][self.bank_ids[key]]

    def template_record(self, bank, index):
        record = self.bank_record(bank)
        key = str(int(index))
        if key not in record["templates"]:
            record["templates"][key] = {
                "index": int(index),
                "template_hash": str(bank.table.template_hash[index]),
                "getitem_attempts": 0,
                "getitem_successes": 0,
                "decompression_successes": 0,
                "filter_attempts_by_segment": {},
                "filter_successes_by_segment": {},
            }
        return record["templates"][key]

    def observe_geometry(self, segments):
        rate = float(segments.sample_rate)
        intervals = []
        rows = []
        for segment, analyze in zip(
            segments.segment_slices, segments.analyze_slices, strict=True
        ):
            if not (0 <= analyze.start < analyze.stop <= segments.time_len):
                raise ValueError("Analysis slice is outside its FFT segment")
            interval = [
                int(segment.start + analyze.start),
                int(segment.start + analyze.stop),
            ]
            intervals.append(interval)
            rows.append(
                {
                    "segment_slice": slice_record(segment),
                    "analyze_slice": slice_record(analyze),
                    "analyzed_sample_interval": interval,
                }
            )
        merged = union_intervals(intervals)
        unique = sum(stop - start for start, stop in merged)
        total = sum(stop - start for start, stop in intervals)
        span = merged[-1][1] - merged[0][0] if merged else 0
        self.data["segment_geometry"].append(
            {
                "sample_rate_hz": rate,
                "strain_start_time": str(segments.strain.start_time),
                "strain_end_time": str(segments.strain.end_time),
                "fft_samples": int(segments.time_len),
                "frequency_samples": int(segments.freq_len),
                "delta_f_hz": float(segments.delta_f),
                "segments": rows,
                "union_analyzed_sample_intervals": merged,
                "unique_analyzed_samples": unique,
                "unique_analyzed_seconds": unique / rate,
                "bounding_span_seconds": span / rate,
                "overlap_samples": total - unique,
                "gap_samples": span - unique,
                "sample_origin": "Offsets relative to strain_start_time; intervals are half-open.",
            }
        )
        _, strain_record = array_record(segments.strain)
        gates = json_safe(getattr(segments.strain, "gating_info", None))
        canonical_gates = json.dumps(
            gates, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        strain_record.update(
            start_time=str(segments.strain.start_time),
            end_time=str(segments.strain.end_time),
            delta_t_seconds=float(segments.strain.delta_t),
            sample_rate_hz=rate,
            capture="Final conditioned strain supplied to StrainSegments before Fourier transforms",
            gating_info={
                "sha256": hashlib.sha256(canonical_gates.encode("utf-8")).hexdigest(),
                "hash_encoding": "UTF-8 JSON, sorted keys, separators (comma, colon), NumPy values converted to lists",
                "summary": {
                    key: {"entries": len(value) if isinstance(value, list) else None}
                    for key, value in (gates.items() if isinstance(gates, dict) else [])
                },
                "values": gates,
            },
        )
        self.data["conditioned_strain"].append(strain_record)

    def observe_psds(self, segments):
        import numpy

        import pycbc

        seen = {}
        objects = {}
        for segment_index, segment in enumerate(segments):
            psd = segment.psd
            if id(psd) in objects:
                objects[id(psd)]["segment_indices"].append(segment_index)
                continue
            values, record = array_record(psd)
            key = (record["data_sha256"], record["dtype_str"], float(psd.delta_f))
            if key in seen:
                seen[key]["segment_indices"].append(segment_index)
                objects[id(psd)] = seen[key]
                continue
            index = len(self.data["psd_arrays"])
            relative_path = (
                Path("arrays") / f"{self.receipt_path.stem}-psd-{index:03d}.npy"
            )
            path = self.receipt_path.parent / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                numpy.save(stream, values, allow_pickle=False)
            record.update(file_record(path))
            record.update(
                relative_path=str(relative_path),
                delta_f_hz=float(psd.delta_f),
                dyn_range_factor=float(pycbc.DYN_RANGE_FAC),
                scaling="DYN_RANGE_FAC**2",
                capture="Exact final segment.psd before overwhitening; no rescaling or dtype conversion",
                finite=bool(numpy.isfinite(values).all()),
                segment_indices=[segment_index],
            )
            self.data["psd_arrays"].append(record)
            seen[key] = record
            objects[id(psd)] = record

    @staticmethod
    def plans(engine):
        return {
            name: qualified_name(getattr(engine, name))
            for name in (
                "_mkl_plan",
                "_fftw_plan",
                "_fftw_batch_plan",
                "_promoted_batch_plan",
            )
            if getattr(engine, name, None) is not None
        }

    def record_fft(self, engine, kind):
        index = len(self.data["fft_engines"])
        self.fft_ids[id(engine)] = index
        row = {
            "id": index,
            "kind": kind,
            "class": qualified_name(engine),
            "size": int(engine.size),
            "nbatch": int(engine.nbatch),
            "input_dtype": str(engine.invec.dtype),
            "output_dtype": str(engine.outvec.dtype),
            "plans_at_construction": self.plans(engine),
            "execute_attempts": 0,
            "execute_successes": 0,
            "executed_routes": {},
        }
        self.data["fft_engines"].append(row)
        execute = engine.execute

        @functools.wraps(execute)
        def observed_execute(*args, **kwargs):
            previous = self.current_fft
            self.current_fft = row
            row["execute_attempts"] += 1
            try:
                result = execute(*args, **kwargs)
                row["execute_successes"] += 1
                row["plans_after_execution"] = self.plans(engine)
                return result
            finally:
                self.current_fft = previous

        self.patch(engine, "execute", observed_execute)
        return engine

    def install_fft(self, class_api):
        for name, kind in (("_fft_factory", "FFT"), ("_ifft_factory", "IFFT")):
            original = getattr(class_api, name)

            def factory(*args, _original=original, _kind=kind, **kwargs):
                cls = _original(*args, **kwargs)

                def construct(*cargs, **ckwargs):
                    return self.record_fft(cls(*cargs, **ckwargs), _kind)

                return construct

            self.patch(class_api, name, factory)
        torchfft = sys.modules.get("pycbc.fft.torchfft")
        if torchfft is None:
            return
        bool_helpers = (
            "_execute_mkl_cpu_ifft_plan",
            "_execute_fftw_cpu_plan",
            "_execute_fftw_cpu_batch_plan",
            "_execute_promoted_torch_batch_plan",
        )
        for name in bool_helpers + ("fft", "ifft", "_execute_batched_ifft_direct"):
            if not hasattr(torchfft, name):
                continue
            original = getattr(torchfft, name)
            route = f"{torchfft.__name__}.{name}"

            def observed(
                *args,
                _original=original,
                _route=route,
                _bool=name in bool_helpers,
                **kwargs,
            ):
                row = self.data["fft_routes"].setdefault(
                    _route, {"calls": 0, "accepted": 0}
                )
                row["calls"] += 1
                result = _original(*args, **kwargs)
                if not _bool or result:
                    row["accepted"] += 1
                    if self.current_fft is not None:
                        counts = self.current_fft["executed_routes"]
                        counts[_route] = counts.get(_route, 0) + 1
                return result

            self.patch(torchfft, name, observed)

    def install(self):
        import pycbc
        from pycbc import scheme, waveform
        from pycbc.fft import class_api
        from pycbc.filter import matchedfilter
        from pycbc.strain import strain
        from pycbc.waveform import bank

        self.receipt["source_root"] = str(Path(pycbc.__file__).resolve().parent.parent)
        self.data["runtime"]["after_imports"] = runtime_snapshot()
        from_cli = scheme.from_cli

        @functools.wraps(from_cli)
        def observed_from_cli(opt):
            self.data["parsed_options"] = vars(opt).copy()
            if not opt.use_compressed_waveforms:
                raise ValueError("Qualification requires --use-compressed-waveforms")
            if opt.multiprocessing_nprocesses:
                raise ValueError(
                    "Qualification requires one process; omit --multiprocessing-nprocesses"
                )
            return from_cli(opt)

        self.patch(scheme, "from_cli", observed_from_cli)

        def reject_generation(*args, **kwargs):
            self.data["waveform_generation_attempts"].append(
                {
                    "template": self.current_template,
                    "approximant": str(kwargs.get("approximant")),
                }
            )
            raise RuntimeError(
                "Waveform generation fallback forbidden during compressed-bank qualification"
            )

        defining_module = importlib.import_module("pycbc.waveform.waveform")
        for module in (waveform, defining_module, bank):
            if hasattr(module, "get_waveform_filter"):
                self.patch(module, "get_waveform_filter", reject_generation)
                self.data["fallback_guard_targets"].append(
                    f"{module.__name__}.get_waveform_filter"
                )

        bank_init = bank.FilterBank.__init__

        @functools.wraps(bank_init)
        def observed_bank_init(obj, *args, **kwargs):
            bank_init(obj, *args, **kwargs)
            self.bank_record(obj)

        self.patch(bank.FilterBank, "__init__", observed_bank_init)
        getitem = bank.FilterBank.__getitem__

        @functools.wraps(getitem)
        def observed_getitem(obj, index):
            row = self.template_record(obj, index)
            self.current_template = [self.bank_ids[id(obj)], int(index)]
            row["getitem_attempts"] += 1
            result = getitem(obj, index)
            row["getitem_successes"] += 1
            row["f_lower_hz"] = float(result.f_lower)
            duration = getattr(result, "length_in_time", None)
            row["template_duration_seconds"] = (
                None if duration is None else float(duration)
            )
            return result

        self.patch(bank.FilterBank, "__getitem__", observed_getitem)
        decompress = bank.FilterBank.get_decompressed_waveform
        signature = inspect.signature(decompress)

        @functools.wraps(decompress)
        def observed_decompress(obj, *args, **kwargs):
            bound = signature.bind(obj, *args, **kwargs)
            index = int(bound.arguments["index"])
            template = self.template_record(obj, index)
            call = {
                "bank_id": self.bank_ids[id(obj)],
                "index": index,
                "template_hash": template["template_hash"],
                "success": False,
                "requested_f_lower_hz": bound.arguments.get("f_lower"),
                "method_override": obj.waveform_decompression_method,
            }
            self.data["decompression_calls"].append(call)
            previous = self.active_decompression
            self.active_decompression = call
            try:
                result = decompress(obj, *args, **kwargs)
            except BaseException as error:
                call["error"] = f"{type(error).__name__}: {error}"
                raise
            else:
                call["success"] = True
                template["decompression_successes"] += 1
                return result
            finally:
                self.active_decompression = previous

        self.patch(bank.FilterBank, "get_decompressed_waveform", observed_decompress)
        fd_decompress = waveform.compress.fd_decompress
        fd_signature = inspect.signature(fd_decompress)

        @functools.wraps(fd_decompress)
        def observed_fd_decompress(*args, **kwargs):
            bound = fd_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if self.active_decompression is not None:
                self.active_decompression["actual_interpolation"] = bound.arguments[
                    "interpolation"
                ]
                self.active_decompression["scheme"] = scheme_snapshot()
            return fd_decompress(*args, **kwargs)

        self.patch(waveform.compress, "fd_decompress", observed_fd_decompress)
        segment_init = strain.StrainSegments.__init__

        @functools.wraps(segment_init)
        def observed_segment_init(obj, *args, **kwargs):
            segment_init(obj, *args, **kwargs)
            self.observe_geometry(obj)

        self.patch(strain.StrainSegments, "__init__", observed_segment_init)
        self.install_fft(class_api)
        controller_init = matchedfilter.MatchedFilterControl.__init__

        @functools.wraps(controller_init)
        def observed_controller_init(obj, *args, **kwargs):
            controller_init(obj, *args, **kwargs)
            self.observe_psds(obj.segments)
            selected = obj.matched_filter_and_cluster
            record = {
                "class": qualified_name(obj),
                "method": f"{selected.__module__}.{selected.__qualname__}",
                "segment_count": len(obj.segments),
                "ifft_engine_id": self.fft_ids.get(id(getattr(obj, "ifft", None))),
                "correlators": [
                    qualified_name(corr) for corr in getattr(obj, "correlators", [])
                ],
            }
            self.data["matched_filter_controllers"].append(record)
            self.data["runtime"]["inside_filter_context"] = runtime_snapshot()

            @functools.wraps(selected)
            def observed_filter(segnum, *fargs, **fkwargs):
                if self.current_template is None:
                    raise RuntimeError(
                        "Filtering started without an observed bank template"
                    )
                bank_id, index = self.current_template
                template = self.template_record(self.banks[bank_id], index)
                key = str(int(segnum))
                attempted = template["filter_attempts_by_segment"]
                attempted[key] = attempted.get(key, 0) + 1
                result = selected(segnum, *fargs, **fkwargs)
                succeeded = template["filter_successes_by_segment"]
                succeeded[key] = succeeded.get(key, 0) + 1
                return result

            self.patch(obj, "matched_filter_and_cluster", observed_filter)

        self.patch(
            matchedfilter.MatchedFilterControl, "__init__", observed_controller_init
        )

    def checks(self):
        checks = {}
        controllers = self.data["matched_filter_controllers"]
        checks["one_scalar_controller"] = len(controllers) == 1
        segments = controllers[0]["segment_count"] if len(controllers) == 1 else 0
        checks["nonempty_segments"] = segments > 0
        geometry = self.data["segment_geometry"]
        checks["recorded_geometry_matches_controller"] = (
            len(geometry) == 1
            and len(geometry[0]["segments"]) == segments
            and geometry[0]["unique_analyzed_samples"] > 0
        )
        checks["conditioned_strain_recorded"] = (
            len(self.data["conditioned_strain"]) == 1
        )
        psds = self.data["psd_arrays"]
        checks["all_segment_psds_saved"] = (
            bool(psds)
            and all(row["finite"] for row in psds)
            and sorted(i for row in psds for i in row["segment_indices"])
            == list(range(segments))
        )
        checks["one_compressed_bank"] = len(self.banks) == 1
        calls = self.data["decompression_calls"]
        checks["all_decompressions_succeeded"] = bool(calls) and all(
            c["success"] for c in calls
        )
        checks["all_interpolations_observed"] = bool(calls) and all(
            "actual_interpolation" in c for c in calls
        )
        checks["no_generation_fallback"] = not self.data["waveform_generation_attempts"]
        expected_segments = {str(i): 1 for i in range(segments)}
        for bank_id, bank in enumerate(self.banks):
            row = self.bank_record(bank)
            count = len(bank)
            row["selected_template_count"] = count
            row["expected_decompressions"] = count
            row["expected_filter_calls"] = count * segments
            for index in range(count):
                template = self.template_record(bank, index)
                template["expected_filter_calls"] = segments
                template["expected_decompression_calls"] = 1
            checks[f"bank_{bank_id}_compression_enabled"] = (
                row["has_compressed_waveforms"] and row["enable_compressed_waveforms"]
            )
            checks[f"bank_{bank_id}_all_templates_once"] = (
                count > 0
                and len(row["templates"]) == count
                and all(
                    t["getitem_attempts"]
                    == t["getitem_successes"]
                    == t["decompression_successes"]
                    == 1
                    and t["filter_attempts_by_segment"]
                    == t["filter_successes_by_segment"]
                    == expected_segments
                    for t in row["templates"].values()
                )
            )
        primary_id = controllers[0]["ifft_engine_id"] if len(controllers) == 1 else None
        primary = None if primary_id is None else self.data["fft_engines"][primary_id]
        expected_filters = sum(len(bank) for bank in self.banks) * segments
        checks["scalar_ifft_executed_for_every_pair"] = bool(primary) and (
            primary["nbatch"] == 1
            and primary["execute_successes"]
            == primary["execute_attempts"]
            == expected_filters
            and expected_filters > 0
        )
        self.data["decompression_success_count"] = sum(c["success"] for c in calls)
        self.data["interpolation_counts"] = dict(
            Counter(c.get("actual_interpolation", "unobserved") for c in calls)
        )
        return checks


def write_receipt(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(json_safe(receipt), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    receipt = {
        "schema_version": 1,
        "purpose": "Untimed pycbc_inspiral qualification; all instrumented timing fields are invalid for benchmarking.",
        "status": "failed",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "cwd": str(Path.cwd()),
        "pid": os.getpid(),
        "python": sys.executable,
        "python_version": sys.version,
        "wrapper": file_record(__file__),
        "argv": command,
    }
    receipt_path = args.receipt.resolve()
    qualification = Qualification(receipt, receipt_path)
    old_argv = sys.argv
    exit_code = 1
    try:
        if not command:
            raise ValueError(
                "Provide -- /exact/path/to/bin/pycbc_inspiral followed by its arguments"
            )
        executable = Path(command[0]).resolve(strict=True)
        if executable.name != "pycbc_inspiral" or executable.parent.name != "bin":
            raise ValueError(
                "The executable must be the exact bin/pycbc_inspiral source file"
            )
        receipt["executable"] = file_record(executable)
        if executable == args.receipt.resolve():
            raise ValueError("Receipt path cannot overwrite the executable")
        qualification.install()
        if Path(receipt["source_root"]) != executable.parent.parent:
            raise ValueError(
                "Imported pycbc must come from the executable's source checkout"
            )
        sys.argv = [str(executable), *command[1:]]
        receipt["executed_argv"] = list(sys.argv)
        try:
            runpy.run_path(str(executable), run_name="__main__")
            exit_code = 0
        except SystemExit as error:
            exit_code = (
                error.code
                if isinstance(error.code, int)
                else (0 if error.code is None else 1)
            )
            if exit_code:
                raise
    except BaseException as error:
        receipt["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        if not isinstance(error, SystemExit):
            exit_code = 1
    finally:
        sys.argv = old_argv
        receipt["executable_exit_code"] = exit_code
        try:
            receipt["checks"] = qualification.checks()
            qualification.data["runtime"]["at_finish"] = runtime_snapshot()
            receipt["source_modules"] = {
                name: file_record(module.__file__)
                for name, module in sorted(sys.modules.items())
                if (name == "pycbc" or name.startswith("pycbc."))
                and getattr(module, "__file__", None)
            }
            if exit_code == 0 and all(receipt["checks"].values()):
                receipt["status"] = "success"
        except BaseException as error:
            receipt["diagnostics_error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        finally:
            qualification.restore()
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_receipt(args.receipt.resolve(), receipt)
    if receipt["status"] != "success":
        print(
            f"Qualification failed; inspect {args.receipt.resolve()}", file=sys.stderr
        )
    return 0 if receipt["status"] == "success" else (exit_code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
