#!/usr/bin/env python3
"""One isolated current-code LiveBatch cell; no scheduling or remote operations.

First run qualify/branch_standard/B1 without --reference-dir. It creates
outputs.npy (complex64, blocks x bank x FFT samples) and result.json. All other
qualification cells require that directory. Timing uses the untouched public
process_data method and never reads/writes full output arrays. Timing requires
a passed reference and checks every call's triggers outside its clock. Full
pointwise qualification remains separate. Exit 0/status pass means successful,
1/status fail means a scientific mismatch, and 2/status error is a setup error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
import types
from datetime import datetime, timezone

EXPECTED_HEAD = "9578a710479b924e882857c4dffab6ed372a634b"
BATCHES = (1, 8, 32, 128, 512, 1024)
ROUTES = ("branch_standard", "torch_cpu", "torch_cuda")
POLICY = {
    "version": 1,
    "pointwise_complex": {"rtol": 1e-4, "atol": 1e-6},
    "snr": {"rtol": 0.0, "atol": 1e-3},
    "coa_phase": {"rtol": 0.0, "atol": 1e-3, "circular": True},
    "sigmasq": {"rtol": 1e-3, "atol": 0.0},
    "chisq": {"rtol": 1e-4, "atol": 1e-4},
    "sg_chisq": {"rtol": 3e-5, "atol": 3e-5},
    "exact_fields": ["template_id", "peak_index", "end_time", "chisq_dof",
                     "mass1", "mass2", "template_hash"],
    "finite": "all complex samples and all numeric trigger fields",
    "comparison": "abs(actual-reference) <= atol + rtol*abs(reference)",
    "sources": {
        "pointwise_complex": "predeclared campaign policy rtol=1e-4, atol=1e-6; not a repository-test tolerance",
        "snr_phase_sigmasq": "tools/bench_production_live_batch.py:_verify_parity",
        "chisq": "test/test_chisq_torch.py:test_power_chisq_cuda_triton_fused (1e-4 envelope applied to reduced chisq)",
        "sg_chisq": "test/test_chisq_torch.py:test_sine_gaussian_chisq_stays_on_torch",
        "exact_indices": "test/test_torch_large_batches.py:test_public_live_filter_full_batch",
    },
}
SCIENCE = {
    "bank": "deterministic synthetic complex64 frequency templates; no waveform generation",
    "strain": "complex64 frequency noise plus coherent injections, divided by PSD",
    "chisq_bins": "16",
    "chisq_output": "reduced power chisq; chisq_dof=30",
    "sg_enabled": True,
    "sg_constructor": "SingleDetSGChisq(bank, '16', 0.0, ['mass1>0:8-20,12-40'])",
    "sg_activation": "two tiles per trigger; execution conditions checked after each call (enabled, matched settings, zero threshold, valid tile support)",
    "snr_threshold": 5.5,
    "newsnr_threshold": None,
    "snr_abort_threshold": None,
    "max_triggers_in_batch": None,
    "scope": "public live matched filter and vetoes, excludes PSD estimation and frame IO",
}
np = None  # Imported only after thread limits and production flags are set.


def digest_bytes(value):
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--route", choices=ROUTES, required=True)
    p.add_argument("--batch", type=positive, choices=BATCHES, required=True)
    p.add_argument("--mode", choices=("qualify", "timing"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path)
    p.add_argument("--size", type=positive, default=131072)
    p.add_argument("--bank-size", type=positive, default=1024)
    p.add_argument("--num-blocks", type=positive, default=3)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--samples", type=positive, default=5)
    p.add_argument("--seed", type=int, default=7101)
    p.add_argument("--cuda-device", type=int, default=0)
    p.add_argument("--threads", type=positive, default=1)
    return p


def source_identity(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args],
                                       text=True).strip()
    revision = git("rev-parse", "HEAD")
    if revision != EXPECTED_HEAD:
        raise ValueError(f"source HEAD {revision} != required {EXPECTED_HEAD}")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError("source has tracked changes: " + dirty)
    paths = ["pycbc/filter/matchedfilter.py", "pycbc/filter/matchedfilter_torch.py",
             "pycbc/fft/mkl.py", "pycbc/fft/torchfft.py", "pycbc/vetoes/chisq.py",
             "pycbc/vetoes/chisq_torch.py", "pycbc/vetoes/sgchisq.py",
             "tools/bench_production_live_batch.py"]
    return {"revision": revision, "root": str(root), "tracked_dirty": False,
            "file_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                            for p in paths}}


def workload(args):
    """No batch-dependent RNG consumption, bank construction or injections."""
    n, count = args.size, args.bank_size
    rate = 2048.0
    df = rate / n
    flen = n // 2 + 1
    freq = np.arange(flen, dtype=np.float64) * df
    psd = np.clip((np.maximum(freq, df) / 100.0) ** -1
                  + (freq / 500.0) ** 2, 0.1, 100.0).astype(np.float32)
    lo, hi = max(1, int(30.0 / df)), min(flen - 1, int(800.0 / df))
    rng = np.random.default_rng(args.seed)
    bank = np.zeros((count, flen), dtype=np.complex64)
    sigma = np.empty(count, dtype=np.float64)
    for i in range(count):
        row = bank[i]
        row[lo:hi] = (rng.normal(size=hi-lo)
                      + 1j * rng.normal(size=hi-lo)).astype(np.complex64)
        power = float(4 * df * np.sum(np.abs(row) ** 2 / psd, dtype=np.float64))
        # Unequal powers expose wrong-group normalization cache reuse.
        row *= (1.0 + 0.125 * (i % 29)) / math.sqrt(power)
        sigma[i] = float(4 * df * np.sum(np.abs(row) ** 2 / psd, dtype=np.float64))
    trim = n // 32
    valid_length = 7 * n // 8
    valid_start, valid_end = n - trim - valid_length, n - trim
    duration = valid_length / rate
    # Match the public filter's duration sort so the first group (when B>1)
    # contains at least two injections despite equal-duration argsort ordering.
    duration_order = np.full(count, 1.0 / df).argsort()
    injected_rows = [int(duration_order[i]) for i in sorted(set(
        [0, min(1, count-1), count // 2, count-1]))]
    blocks, injections = [], []
    for b in range(args.num_blocks):
        strain = ((rng.normal(size=flen) + 1j * rng.normal(size=flen))
                  * 0.001).astype(np.complex64)
        metadata = []
        for j, i in enumerate(injected_rows):
            offset = valid_start + (j+1) * valid_length // (len(injected_rows)+1)
            target_snr = 8.0 + 1.5 * ((b+j) % 4)
            phase = np.exp(-2j * np.pi * np.arange(flen) * offset / n)
            strain += (bank[i] * (target_snr / math.sqrt(sigma[i]))
                       * phase).astype(np.complex64)
            metadata.append({"template_id": 1000+i, "peak_index": offset,
                             "nominal_snr": target_snr})
        blocks.append((strain / psd).astype(np.complex64))
        injections.append(metadata)
    geometry = {"fft_samples": n, "bank_templates": count, "blocks": args.num_blocks,
                "sample_rate": rate, "delta_f": df, "trim_padding": trim,
                "blocksize": duration, "valid_start": valid_start,
                "valid_end": valid_end, "f_lower": 30.0,
                "template_nonzero_stop": hi, "epoch": 0.0,
                "reader_start_time": 1000000000.0}
    identity = {"geometry": geometry, "seed": args.seed,
                "template_bank_sha256": digest_bytes(bank),
                "psd_sha256": digest_bytes(psd),
                "sigmasq_sha256": digest_bytes(sigma),
                "overwhitened_block_sha256": [digest_bytes(s) for s in blocks],
                "injections": injections, "science": SCIENCE}
    return bank, psd, sigma, blocks, geometry, injections, identity


def row_metrics(actual, reference=None):
    """Compare one row once; complex magnitude error, not magnitude-only output."""
    finite = np.isfinite(actual)
    result = {"samples": int(actual.size),
              "nonfinite_actual": int(actual.size - np.count_nonzero(finite)),
              "sha256": digest_bytes(actual), "failed_samples": 0,
              "nonfinite_reference": 0, "max_abs_error": 0.0,
              "max_tolerance_ratio": 0.0, "relative_l2_error": 0.0}
    if reference is None:
        result["failed_samples"] = result["nonfinite_actual"]
        return result
    ref_finite = np.isfinite(reference)
    result["nonfinite_reference"] = int(reference.size - np.count_nonzero(ref_finite))
    result["reference_sha256"] = digest_bytes(reference)
    ok = finite & ref_finite
    error = np.abs(actual.astype(np.complex128) - reference.astype(np.complex128))
    magnitude = np.abs(reference.astype(np.complex128))
    tol = POLICY["pointwise_complex"]
    limit = tol["atol"] + tol["rtol"] * magnitude
    result["failed_samples"] = int(np.count_nonzero(~ok | (error > limit)))
    if np.any(ok):
        result["max_abs_error"] = float(np.max(error[ok]))
        result["max_tolerance_ratio"] = float(np.max(error[ok] / limit[ok]))
        denominator = float(np.linalg.norm(magnitude[ok]))
        numerator = float(np.linalg.norm(error[ok]))
        result["relative_l2_error"] = numerator / denominator if denominator else None
    return result


def reduce_rows(rows):
    return {"rows": len(rows),
            **{key: sum(r[key] for r in rows) for key in
               ("samples", "failed_samples", "nonfinite_actual", "nonfinite_reference")},
            **{key: max((r[key] for r in rows), default=0.0) for key in
               ("max_abs_error", "max_tolerance_ratio")}}


def numeric_list(values):
    # JSON remains valid on a scientific failure containing NaN/Inf.
    return [v if isinstance(v, int) or math.isfinite(v) else None
            for v in values.tolist()]


def trigger_record(result, geometry, block, captured_indices=None):
    if not isinstance(result, dict):
        raise ValueError("public process_data aborted or returned no result dictionary")
    order = np.argsort(result["template_id"])
    fields = {}
    nonfinite = {}
    count = len(order)
    for key, value in result.items():
        value = np.asarray(value)
        if value.ndim != 1 or len(value) != count:
            raise ValueError(f"misaligned trigger field {key}")
        nonfinite[key] = int(np.count_nonzero(~np.isfinite(value)))
        fields[key] = numeric_list(value[order])
    start = geometry["reader_start_time"] + block * geometry["blocksize"]
    if captured_indices is not None:
        fields["peak_index"] = [captured_indices.get(i) for i in fields["template_id"]]
    else:
        fields["peak_index"] = [int(round((t-start) * geometry["sample_rate"]))
                                + geometry["valid_start"] if t is not None else None
                                for t in fields["end_time"]]
    return {"block": block, "count": count, "fields": fields,
            "nonfinite_fields": nonfinite}


def compare_triggers(actual, reference):
    af, rf = actual["fields"], reference["fields"]
    details = {}
    for key in sorted(set(af) | set(rf)):
        a, r = af.get(key), rf.get(key)
        if a is None or r is None or len(a) != len(r):
            details[key] = {"passed": False, "reason": "field/length mismatch"}
            continue
        if key not in POLICY or key in POLICY["exact_fields"]:
            failed = sum(x != y or x is None or y is None for x, y in zip(a, r))
            details[key] = {"passed": failed == 0, "failed_values": failed,
                            "comparison": "exact"}
            continue
        rule = POLICY[key]
        av, rv = np.asarray(a, dtype=float), np.asarray(r, dtype=float)
        delta = av-rv
        if rule.get("circular"):
            delta = (delta + np.pi) % (2*np.pi) - np.pi
        finite = np.isfinite(av) & np.isfinite(rv)
        errors = np.abs(delta)
        failed = int(np.count_nonzero(~finite | (errors > rule["atol"]
                                                + rule["rtol"] * np.abs(rv))))
        details[key] = {"passed": failed == 0, "failed_values": failed,
                        "max_abs_error": float(np.max(errors[finite])) if np.any(finite) else None}
    populated = actual["count"] > 0 and reference["count"] > 0
    return {"passed": populated and bool(details) and all(d["passed"] for d in details.values()),
            "nonempty": populated, "fields": details}


def validate_triggers(record, injections):
    f = record["fields"]
    failures = []
    ids = f["template_id"]
    if len(ids) != len(set(ids)):
        failures.append("duplicate trigger IDs")
    if sum(record["nonfinite_fields"].values()):
        failures.append("nonfinite trigger fields")
    detected = set(zip(ids, f["peak_index"]))
    for injection in injections:
        if (injection["template_id"], injection["peak_index"]) not in detected:
            failures.append(f"injection missing or displaced: {injection['template_id']}")
    if not ids:
        failures.append("no injected triggers")
    if any(value != 30 for value in f["chisq_dof"]):
        failures.append("power chisq not evaluated with 16 bins")
    return failures


def sg_activation(sg, templates_by_id, psd, record):
    """Verify actual SG execution conditions, without replacing any veto call."""
    checks = []
    for template_id in record["fields"]["template_id"]:
        template = templates_by_id[template_id]
        settings = sg.params.get(template.params.template_hash)
        if not sg.do or sg.snr_threshold != 0.0 or not settings:
            raise ValueError("SG configuration would skip a recorded trigger")
        bins = sg.cached_chisq_bins(template, psd)
        fpeak = float((2 * bins[-2] - bins[-3]) * template.delta_f)
        fstop = len(template) * template.delta_f * 0.9
        kmin = int(template.f_lower / psd.delta_f)
        tiles = []
        for setting in settings.split(","):
            q, offset = map(float, setting.split("-"))
            center = fpeak + offset
            low = max(kmin * template.delta_f, center - 50)
            high = center + 50
            kmin, kmax = int(low / template.delta_f), int(high / template.delta_f)
            if q <= 0 or high > fstop or not 0 <= kmin < kmax <= len(template):
                raise ValueError("SG tile would return unity fallback or have invalid support")
            tiles.append({"q": q, "center_hz": center, "low_hz": low, "high_hz": high})
        if len(tiles) != 2:
            raise ValueError("expected exactly two active SG tiles")
        checks.append({"template_id": int(template_id), "tiles": tiles,
                       "chisq_bins_sha256": digest_bytes(np.asarray(bins))})
    return checks


class Capture:
    """Qualification-only wrapper. Capture every group before its mid is reused."""
    def __init__(self, args, filt, output_map, ref_result, result):
        self.args, self.filt = args, filt
        self.output_map, self.reference = output_map, ref_result
        self.result = result
        self.block = 0
        self.indices = {}

    def start_block(self, block):
        self.block = block
        self.indices = {}
        self.counts = np.zeros(self.args.bank_size, dtype=np.uint32)
        self.rows = [None] * self.args.bank_size

    def capture(self, group_index, triggers, veto):
        group = self.filt.tgroups[group_index]
        mid = self.filt.mids[group_index]
        storage = self.filt.out_mem[mid]
        tensor = getattr(getattr(storage, "_data", None), "tensor", None)
        # 16 MiB maximum transfer, with comparison/error scratch limited to a row.
        row_chunk = max(1, (16 * 1024**2) // (8 * self.args.size))
        for start in range(0, len(group), row_chunk):
            end = min(start + row_chunk, len(group))
            if tensor is None:
                values = storage.numpy().reshape(len(group), self.args.size)[start:end]
            else:
                values = tensor.view(len(group), self.args.size)[start:end].detach().cpu().numpy()
            for offset, row in enumerate(values):
                row_id = int(group[start+offset].id) - 1000
                self.counts[row_id] += 1
                if self.reference is None:
                    self.output_map[self.block, row_id] = row
                    metrics = row_metrics(row)
                else:
                    metrics = row_metrics(row, self.output_map[self.block, row_id])
                    expected_hash = self.reference["pointwise_blocks"][self.block]["rows"][row_id]["sha256"]
                    if metrics["reference_sha256"] != expected_hash:
                        self.result["failures"].append(f"reference row hash mismatch {self.block}/{row_id}")
                metrics["template_id"] = 1000 + row_id
                self.rows[row_id] = metrics
        for _snr, _norm, peak, template, _strain in veto:
            if int(template.id) in self.indices:
                self.result["failures"].append(f"duplicate captured trigger {template.id}")
            self.indices[int(template.id)] = int(peak)

    def finish_block(self):
        rows = [r for r in self.rows if r is not None]
        summary = reduce_rows(rows)
        coverage = bool(np.all(self.counts == 1))
        summary.update(block=self.block, processed_once=coverage,
                       processing_counts=self.counts.tolist(), rows=rows)
        self.result["pointwise_blocks"].append(summary)
        if not coverage or summary["failed_samples"]:
            self.result["failures"].append(f"pointwise/coverage mismatch in block {self.block}")


def execute(args, result):
    global np
    if args.size < 256 or args.size % 32:
        raise ValueError("--size must be >=256 and divisible by 32 (valid window and 16 bins)")
    if args.batch > args.bank_size:
        raise ValueError("--batch cannot exceed --bank-size; requested B must actually execute")
    if args.threads != 1:
        raise ValueError("this sweep requires --threads=1")
    if args.warmups < 0 or args.cuda_device < 0 or args.seed < 0:
        raise ValueError("warmups, device and seed must be nonnegative")
    root = args.source_root.resolve()
    result["source"] = source_identity(root)
    sys.path.insert(0, str(root))
    from tools.bench_production_live_batch import route_environment, route_configuration
    configured = route_environment(args.route, dict(os.environ))
    removed = sorted(k for k in os.environ if k not in configured)
    for key in removed:
        os.environ.pop(key)
    os.environ.update(configured)
    thread_vars = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS")
    for key in thread_vars:
        os.environ[key] = str(args.threads)
    result["routing"] = route_configuration(args.route)
    result["removed_environment_flags"] = removed
    result["thread_environment"] = {k: os.environ[k] for k in thread_vars}
    import numpy as numpy_module
    np = numpy_module
    import pycbc
    from pycbc import scheme
    from pycbc.filter import matchedfilter
    from pycbc.fft.backend_support import get_backend, set_backend
    from pycbc.types import FrequencySeries
    from pycbc.io.record import FieldArray
    from pycbc.vetoes.sgchisq import SingleDetSGChisq
    if not Path(pycbc.__file__).resolve().is_relative_to(root):
        raise ValueError(f"imported incorrect pycbc: {pycbc.__file__}")
    torch = None
    if args.route == "branch_standard":
        context = scheme.CPUScheme(num_threads=args.threads)
    else:
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)
        torch.set_grad_enabled(False)
        device = f"cuda:{args.cuda_device}" if args.route == "torch_cuda" else "cpu"
        if args.route == "torch_cuda":
            torch.cuda.set_device(args.cuda_device)
            torch.cuda.reset_peak_memory_stats()
        context = scheme.TorchScheme(device)
    result["runtime"] = {"python": sys.version, "numpy": np.__version__,
                         "pycbc": pycbc.__version__, "pycbc_path": pycbc.__file__,
                         "torch": torch.__version__ if torch else None,
                         "torch_num_threads": torch.get_num_threads() if torch else None,
                         "torch_num_interop_threads": torch.get_num_interop_threads() if torch else None,
                         "cuda": torch.version.cuda if torch else None,
                         "device": (f"cuda:{args.cuda_device}" if args.route == "torch_cuda" else "cpu")}

    def sync():
        if args.route == "torch_cuda":
            torch.cuda.synchronize(args.cuda_device)

    bank_np, psd_np, sigma, data_np, geom, injections, identity = workload(args)
    result["inputs"] = identity
    table = FieldArray.from_kwargs(
        mass1=(10 + np.arange(args.bank_size) * 0.01).astype(np.float32),
        mass2=np.full(args.bank_size, 10, dtype=np.float32),
        template_hash=np.arange(1000, 1000 + args.bank_size, dtype=np.int64))
    identity["bank_parameters_sha256"] = digest_bytes(np.asarray(table))
    result["input_sha256"] = canonical_hash(identity)
    reference = None
    output_map = None
    is_reference = args.mode == "qualify" and args.reference_dir is None
    if is_reference and (args.route != "branch_standard" or args.batch != 1):
        raise ValueError("only qualify/branch_standard/B1 can create a reference")
    if not is_reference:
        if args.reference_dir is None:
            raise ValueError("timing and candidate qualification require --reference-dir")
        reference = json.loads((args.reference_dir / "result.json").read_text())
        required = {"status": "pass", "mode": "qualify", "route": "branch_standard", "batch": 1,
                    "input_sha256": result["input_sha256"], "tolerance_policy": POLICY,
                    "worker_sha256": result["worker_sha256"]}
        for key, value in required.items():
            if reference.get(key) != value:
                raise ValueError(f"reference {key} differs or reference failed")
        if reference["source"]["revision"] != EXPECTED_HEAD:
            raise ValueError("reference source mismatch")
        result["reference_result_sha256"] = hashlib.sha256(
            (args.reference_dir / "result.json").read_bytes()).hexdigest()
        result["reference_directory"] = str(args.reference_dir)
    if args.mode == "qualify":
        shape = (args.num_blocks, args.bank_size, args.size)
        if is_reference:
            output_map = np.lib.format.open_memmap(args.output_dir / "outputs.npy", mode="w+",
                                                  dtype=np.complex64, shape=shape)
        else:
            output_map = np.load(args.reference_dir / "outputs.npy", mmap_mode="r", allow_pickle=False)
            if output_map.shape != shape or output_map.dtype != np.dtype("complex64"):
                raise ValueError("reference shape/dtype mismatch")
        result["reference"] = {"created": is_reference,
                               "directory": str(args.output_dir if is_reference else args.reference_dir),
                               "filename": "outputs.npy", "shape": list(shape), "dtype": "complex64",
                               "quantity": "raw complex IFFT output; SNR = output * 4*delta_f/sqrt(sigmasq)"}

    with context:
        if args.route == "branch_standard":
            set_backend(["mkl"])
            if get_backend().__name__ != "pycbc.fft.mkl":
                raise RuntimeError("explicit MKL CPU FFT backend unavailable; no fallback allowed")
        psd = FrequencySeries(psd_np, delta_f=geom["delta_f"], epoch=0)
        templates = []
        for i, row in enumerate(bank_np):
            template = FrequencySeries(row, delta_f=geom["delta_f"], epoch=0)
            template.id = 1000+i
            template.params = table[i]
            template.f_lower = geom["f_lower"]
            # Same PSD object throughout the fixed workload; host sigma computed once.
            template.sigmasq = lambda p, value=float(sigma[i]): value
            templates.append(template)
        sg = SingleDetSGChisq(types.SimpleNamespace(table=table), "16", 0.0,
                             ["mass1>0:8-20,12-40"])
        templates_by_id = {int(t.id): t for t in templates}
        sync()
        setup_start = time.perf_counter_ns()
        filt = matchedfilter.LiveBatchMatchedFilter(
            templates, snr_threshold=5.5, chisq_bins="16", sg_chisq=sg,
            maxelements=args.batch * args.size)
        sync()
        result["filter_setup_ms"] = (time.perf_counter_ns()-setup_start) / 1e6
        group_ids = [[int(t.id) for t in g] for g in filt.tgroups]
        flat_ids = [i for group in group_ids for i in group]
        if sorted(flat_ids) != list(range(1000, 1000+args.bank_size)):
            raise ValueError("filter group layout misses or duplicates templates")
        if max(map(len, group_ids)) != args.batch:
            raise ValueError("effective group size does not equal requested batch")
        result["group_layout"] = group_ids
        result["fft_plans"] = [{"group_size": len(group),
                                "class": type(filt.ifts[mid]).__module__ + "." + type(filt.ifts[mid]).__name__,
                                "nbatch": int(filt.ifts[mid].nbatch)}
                               for group, mid in zip(filt.tgroups, filt.mids)]
        if args.route == "branch_standard" and any(p["class"] != "pycbc.fft.mkl.IFFT"
                                                   for p in result["fft_plans"]):
            raise RuntimeError("CPU plan did not dispatch to MKL")
        readers = []
        for b, values in enumerate(data_np):
            strain = FrequencySeries(values, delta_f=geom["delta_f"], epoch=0)
            strain.psd = psd
            readers.append(types.SimpleNamespace(
                overwhitened_data=lambda _df, s=strain: s,
                trim_padding=geom["trim_padding"], blocksize=geom["blocksize"],
                sample_rate=geom["sample_rate"],
                start_time=geom["reader_start_time"] + b * geom["blocksize"]))
        result["trigger_blocks"] = []
        result["trigger_comparisons"] = []
        if args.mode == "qualify":
            result["pointwise_blocks"] = []
            capture = Capture(args, filt, output_map, reference, result)
            original = filt._process_batch

            def wrapped():
                group_index = filt.block_id
                triggers, veto = original()
                if triggers is not None and triggers is not False:
                    capture.capture(group_index, triggers, veto)
                return triggers, veto

            filt._process_batch = wrapped
            try:
                for b, reader in enumerate(readers):
                    capture.start_block(b)
                    raw_result = filt.process_data(reader)
                    capture.finish_block()
                    record = trigger_record(raw_result, geom, b, capture.indices)
                    record["sg_activation"] = sg_activation(sg, templates_by_id, psd, record)
                    result["trigger_blocks"].append(record)
                    result["failures"].extend(f"block {b}: {f}" for f in validate_triggers(record, injections[b]))
                    if reference is not None:
                        comparison = compare_triggers(record, reference["trigger_blocks"][b])
                        result["trigger_comparisons"].append(comparison)
                        if not comparison["passed"]:
                            result["failures"].append(f"trigger mismatch in block {b}")
            finally:
                filt._process_batch = original
                if is_reference:
                    output_map.flush()
            result["qualification"] = "reference_created" if is_reference else "compared_all_samples_and_triggers"
        else:
            # All extraction and bookkeeping are outside the timed API boundary.
            latencies = []
            result["timing_trigger_checks"] = []
            cold = []
            for iteration in range(1 + args.warmups + args.samples):
                elapsed, records = [], []
                for b, reader in enumerate(readers):
                    sync()
                    start = time.perf_counter_ns()
                    output = filt.process_data(reader)
                    sync()
                    elapsed.append((time.perf_counter_ns() - start) / 1e6)
                    record = trigger_record(output, geom, b)
                    record["sg_activation"] = sg_activation(sg, templates_by_id, psd, record)
                    records.append(record)
                    result["failures"].extend(
                        f"iteration {iteration}/block {b}: {f}"
                        for f in validate_triggers(record, injections[b]))
                    comparison = compare_triggers(record, reference["trigger_blocks"][b])
                    comparison.update(iteration=iteration, block=b)
                    result["trigger_comparisons"].append(comparison)
                    if not comparison["passed"]:
                        result["failures"].append(f"trigger mismatch in iteration {iteration}/block {b}")
                result["timing_trigger_checks"].append({"iteration": iteration, "blocks": records})
                if iteration == 0:
                    cold = elapsed
                elif iteration > args.warmups:
                    latencies.append(elapsed)
            result["trigger_blocks"] = records
            totals = [sum(values) for values in latencies]
            result["timing"] = {"call_surface": "LiveBatchMatchedFilter.process_data",
                                "instrumented": False, "cold_block_ms": cold,
                                "trigger_validation": "every cold, warmup and measured call outside clock",
                                "warm_block_ms": latencies, "warm_iteration_ms": totals,
                                "median_iteration_ms": statistics.median(totals),
                                "templates_per_iteration": args.bank_size * args.num_blocks,
                                "templates_per_second": [args.bank_size * args.num_blocks * 1000/t for t in totals],
                                "coverage": "all templates exactly once by validated static group layout"}
            result["qualification"] = "triggers_compared_each_call; full pointwise checked separately"
        if args.route == "torch_cuda":
            result["runtime"]["cuda_device_name"] = torch.cuda.get_device_name(args.cuda_device)
            result["cuda_memory"] = {"allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                                     "reserved_peak_bytes": torch.cuda.max_memory_reserved()}
    if source_identity(root) != result["source"]:
        raise ValueError("source changed during worker")
    result["status"] = "fail" if result["failures"] else "pass"


def main(argv=None):
    args = parser().parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if args.reference_dir is not None:
        args.reference_dir = args.reference_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Never overwrite evidence or a reference with another cell's result.
    if (args.output_dir / "result.json").exists() or (args.output_dir / "outputs.npy").exists():
        print("output directory already contains cell evidence; choose a fresh directory", file=sys.stderr)
        return 2
    result = {"schema_version": 1, "status": "running", "failures": [],
              "mode": args.mode, "route": args.route, "batch": args.batch,
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "expected_head": EXPECTED_HEAD, "tolerance_policy": POLICY, "science": SCIENCE,
              "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "host": platform.node(), "pid": os.getpid(), "cwd": os.getcwd(),
              "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    start = time.monotonic()
    try:
        execute(args, result)
        code = 0 if result["status"] == "pass" else 1
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        code = 2
    result["elapsed_seconds"] = time.monotonic()-start
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["exit_code"] = code
    atomic_json(args.output_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "result": str(args.output_dir / "result.json"),
                      "failures": len(result["failures"]), "exit_code": code}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
