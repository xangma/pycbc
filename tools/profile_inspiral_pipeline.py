"""Inspiral pipeline diagnostic profiling tool.

Instruments and profiles pycbc_inspiral across sequential shards in plain,
stage-instrumented, or cProfile modes. Evaluates wall/CPU time, host RSS,
and PyTorch CUDA memory dynamics, generating NVTX ranges for NVIDIA
Nsight Systems (nsys) timeline attribution.

Usage:
  PYTHONPATH=. python tools/profile_inspiral_pipeline.py \
    --manifest /path/to/manifest.json \
    --output /path/to/profile_output \
    --mode stages \
    --shards 2
"""

import time

T0_PROCESS = time.perf_counter()

import argparse  # noqa: E402
import cProfile  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pstats  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import tools.run_inspiral_campaign as campaign  # noqa: E402
from tools.run_inspiral_campaign import (  # noqa: E402
    _validate_shard_argv,
    _worker_cleanup,
    execute_inspiral_shard,
)

torch = None
psutil = None
_PROCESS_HANDLE = None

EXACT_STAGE_ANCHORS: List[Tuple[str, str]] = [
    (
        "gwstrain = strain.from_cli(opt, dyn_range_fac=DYN_RANGE_FAC,\n",
        "strain_conditioning",
    ),
    (
        "strain_segments = strain.StrainSegments.from_cli(opt, gwstrain)\n",
        "segmentation",
    ),
    (
        "with ctx:\n",
        "scheme_setup",
    ),
    (
        "    segments = strain_segments.fourier_segments()\n",
        "Fourier",
    ),
    (
        "    psd.associate_psds_to_segments(opt, segments, gwstrain, flen, "
        "delta_f,\n",
        "PSD",
    ),
    (
        "    out_types = {\n",
        "event_setup",
    ),
    (
        "    template_mem = zeros(tlen, dtype = complex64)\n",
        "buffers_filter_veto_setup",
    ),
    (
        "    logging.info(\"Overwhitening frequency-domain data segments\")\n",
        "overwhitening",
    ),
    (
        "    logging.info(\"Read in template bank\")\n",
        "bank_load_bind",
    ),
    (
        "    for tchunk in tchunks:\n",
        "template_loop",
    ),
    (
        "event_mgr.consolidate_events(opt, gwstrain=gwstrain)\n",
        "event_consolidation",
    ),
    (
        "logging.info(\"Writing out triggers\")\n",
        "hdf_output",
    ),
]


def _ensure_process_handle():
    global psutil, _PROCESS_HANDLE
    if _PROCESS_HANDLE is None:
        if psutil is None:
            import psutil as _psutil
            psutil = _psutil
        _PROCESS_HANDLE = psutil.Process()
    return _PROCESS_HANDLE


def _get_current_rss_bytes() -> int:
    proc = _ensure_process_handle()
    return int(proc.memory_info().rss)


def _get_lifetime_max_rss_bytes() -> int:
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return int(rusage.ru_maxrss)
    return int(rusage.ru_maxrss * 1024)


class RSSSampler(threading.Thread):
    """Background thread sampling process host RSS every 50ms."""

    def __init__(self, interval_s: float = 0.05):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self.peak_rss_bytes = 0
        self.samples_count = 0

    def run(self):
        while not self._stop_event.is_set():
            rss = _get_current_rss_bytes()
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
            self.samples_count += 1
            self._stop_event.wait(self.interval_s)

    def stop(self) -> Tuple[int, int]:
        self._stop_event.set()
        self.join()
        curr_rss = _get_current_rss_bytes()
        peak = max(self.peak_rss_bytes, curr_rss)
        return peak, self.samples_count


class StageRecorder:
    """Tracks sequential pipeline stages, emitting balanced NVTX ranges."""

    def __init__(self):
        self.stages: List[Dict[str, Any]] = []
        self._current_stage: str = "imports_options"
        self._t_wall_stage_start: float = time.perf_counter()
        self._t_cpu_stage_start: float = time.process_time()
        self._nvtx_active: bool = False
        if torch is not None and torch.cuda.is_available():
            torch.cuda.nvtx.range_push(self._current_stage)
            self._nvtx_active = True

    def advance(self, next_stage_name: str) -> None:
        now_wall = time.perf_counter()
        now_cpu = time.process_time()
        wall_dur = now_wall - self._t_wall_stage_start
        cpu_dur = now_cpu - self._t_cpu_stage_start

        if torch is not None and torch.cuda.is_available():
            if self._nvtx_active:
                torch.cuda.nvtx.range_pop()
                self._nvtx_active = False
            alloc = torch.cuda.memory_allocated()
            res = torch.cuda.memory_reserved()
            max_alloc = torch.cuda.max_memory_allocated()
            max_res = torch.cuda.max_memory_reserved()
        else:
            alloc = res = max_alloc = max_res = 0

        self.stages.append({
            "stage": self._current_stage,
            "wall_s": wall_dur,
            "cpu_s": cpu_dur,
            "cuda_allocated_bytes": alloc,
            "cuda_reserved_bytes": res,
            "cuda_max_allocated_bytes": max_alloc,
            "cuda_max_reserved_bytes": max_res,
            "host_rss_bytes": _get_current_rss_bytes(),
        })

        self._current_stage = next_stage_name
        self._t_wall_stage_start = time.perf_counter()
        self._t_cpu_stage_start = time.process_time()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.nvtx.range_push(next_stage_name)
            self._nvtx_active = True

    def close(self) -> None:
        now_wall = time.perf_counter()
        now_cpu = time.process_time()
        wall_dur = now_wall - self._t_wall_stage_start
        cpu_dur = now_cpu - self._t_cpu_stage_start

        if torch is not None and torch.cuda.is_available():
            if self._nvtx_active:
                torch.cuda.nvtx.range_pop()
                self._nvtx_active = False
            alloc = torch.cuda.memory_allocated()
            res = torch.cuda.memory_reserved()
            max_alloc = torch.cuda.max_memory_allocated()
            max_res = torch.cuda.max_memory_reserved()
        else:
            alloc = res = max_alloc = max_res = 0

        self.stages.append({
            "stage": self._current_stage,
            "wall_s": wall_dur,
            "cpu_s": cpu_dur,
            "cuda_allocated_bytes": alloc,
            "cuda_reserved_bytes": res,
            "cuda_max_allocated_bytes": max_alloc,
            "cuda_max_reserved_bytes": max_res,
            "host_rss_bytes": _get_current_rss_bytes(),
        })


def instrument_entrypoint(source_path: str, target_path: str) -> None:
    text = Path(source_path).read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    for target_line, next_stage in EXACT_STAGE_ANCHORS:
        matches = [
            idx for idx, line in enumerate(lines) if line == target_line
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Anchor check failed: expected 1 match for "
                f"{repr(target_line)}, found {len(matches)}"
            )
        idx = matches[0]
        indent_len = len(target_line) - len(target_line.lstrip())
        indent = " " * indent_len
        stmt = f"{indent}_stage_recorder.advance({repr(next_stage)})\n"
        lines.insert(idx, stmt)

    lines.append("_stage_recorder.advance('runner_cleanup')\n")
    Path(target_path).write_text("".join(lines), encoding="utf-8")


def _capture_cuda_stats() -> Dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        return {}
    raw_stats = torch.cuda.memory_stats()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        "inactive_split_bytes": raw_stats.get(
            "inactive_split_bytes.all.current", 0
        ),
        "allocation_retries": raw_stats.get("num_alloc_retries", 0),
        "num_ooms": raw_stats.get("num_ooms", 0),
    }


def run_profiled_shard(
    task_spec: Dict[str, Any],
    mode: str,
    base_entrypoint: str,
    disposable_entrypoint: str,
    output_dir: str,
) -> Dict[str, Any]:
    shard_idx = task_spec["shard_idx"]
    shard_output = task_spec["output_file"]
    orig_run_path = campaign.runpy.run_path
    stage_recorder: Optional[StageRecorder] = None
    sampler: Optional[RSSSampler] = None
    prof: Optional[cProfile.Profile] = None
    error_caught: Optional[str] = None
    shard_result: Dict[str, Any] = {}
    shard_nvtx_active = False

    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    if mode in ("stages", "cprofile"):
        sampler = RSSSampler(interval_s=0.05)
        sampler.start()

    mem_start = _get_current_rss_bytes()
    cuda_before = _capture_cuda_stats()
    t_start_wall = time.perf_counter()
    t_start_cpu = time.process_time()

    def _wrapped_run_path(path_or_file, init_globals=None, run_name=None):
        actual_path = path_or_file
        merged_globals = dict(init_globals or {})
        if (
            mode == "stages"
            and os.path.abspath(str(path_or_file)) == os.path.abspath(
                base_entrypoint
            )
        ):
            actual_path = disposable_entrypoint
            merged_globals["_stage_recorder"] = stage_recorder
        return orig_run_path(
            actual_path, init_globals=merged_globals, run_name=run_name
        )

    campaign.runpy.run_path = _wrapped_run_path
    if mode == "stages":
        if torch is not None and torch.cuda.is_available():
            torch.cuda.nvtx.range_push(f"shard_{shard_idx}")
            shard_nvtx_active = True
        stage_recorder = StageRecorder()
    elif mode == "cprofile":
        prof = cProfile.Profile()
        prof.enable()

    try:
        shard_result = execute_inspiral_shard(task_spec)
    except Exception as exc:
        error_caught = f"{type(exc).__name__}: {str(exc)}"
    finally:
        campaign.runpy.run_path = orig_run_path
        if prof is not None:
            prof.disable()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                if error_caught is None:
                    error_caught = f"CUDA synchronization failed: {exc}"
        if stage_recorder is not None:
            stage_recorder.close()
            if shard_nvtx_active:
                torch.cuda.nvtx.range_pop()
                shard_nvtx_active = False

    t_end_wall = time.perf_counter()
    t_end_cpu = time.process_time()
    cuda_after = _capture_cuda_stats()

    if sampler is not None:
        peak_rss, samples_count = sampler.stop()
        sampled_flag = True
    else:
        peak_rss = None
        samples_count = 0
        sampled_flag = False

    mem_end = _get_current_rss_bytes()

    cprofile_summary = {}
    if prof is not None:
        pstats_path = os.path.join(output_dir, f"shard_{shard_idx}.pstats")
        prof.dump_stats(pstats_path)
        st = pstats.Stats(prof)
        st.sort_stats("cumulative")
        top_cum = []
        for func in st.fcn_list[:80]:
            cc, nc, tt, ct, callers = st.stats[func]
            top_cum.append({
                "file": func[0],
                "line": func[1],
                "name": func[2],
                "ncalls": nc,
                "tottime_s": tt,
                "cumtime_s": ct,
            })
        st.sort_stats("tottime")
        top_self = []
        for func in st.fcn_list[:80]:
            cc, nc, tt, ct, callers = st.stats[func]
            top_self.append({
                "file": func[0],
                "line": func[1],
                "name": func[2],
                "ncalls": nc,
                "tottime_s": tt,
                "cumtime_s": ct,
            })
        cprofile_summary = {
            "pstats_file": pstats_path,
            "top80_cumulative": top_cum,
            "top80_self": top_self,
        }

    after_shard_cleanup_cuda = _capture_cuda_stats()
    after_shard_cleanup_rss = _get_current_rss_bytes()

    return {
        "shard_idx": shard_idx,
        "pid": os.getpid(),
        "status": "completed" if error_caught is None else "failed",
        "error": error_caught,
        "argv": task_spec.get("argv", []),
        "output_file": shard_output,
        "output_exists": os.path.isfile(shard_output),
        "total_wall_s": t_end_wall - t_start_wall,
        "total_cpu_s": t_end_cpu - t_start_cpu,
        "host_rss_start_bytes": mem_start,
        "host_rss_peak_bytes": peak_rss,
        "host_rss_sampled": sampled_flag,
        "host_rss_sample_count": samples_count,
        "host_rss_end_bytes": mem_end,
        "host_rss_after_shard_cleanup_bytes": after_shard_cleanup_rss,
        "host_ru_maxrss_bytes": _get_lifetime_max_rss_bytes(),
        "cuda_memory_before": cuda_before,
        "cuda_memory_peak_and_end": cuda_after,
        "cuda_memory_after_shard_cleanup": after_shard_cleanup_cuda,
        "cache_delta": shard_result.get("cache_delta", {}),
        "cache_stats": shard_result.get("cache_stats", {}),
        "cache_bytes_used": shard_result.get("cache_bytes_used", 0),
        "stages": stage_recorder.stages if stage_recorder is not None else [],
        "cprofile": cprofile_summary,
    }


def main() -> int:
    global torch, psutil, _PROCESS_HANDLE
    t_main_start = time.perf_counter()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="Path to manifest JSON"
    )
    parser.add_argument(
        "--output", required=True, help="Profile output directory"
    )
    parser.add_argument(
        "--shards", type=int, default=2, help="Positive number of shards"
    )
    parser.add_argument(
        "--cache-mib", type=int, default=256, help="Cache budget in MiB"
    )
    parser.add_argument(
        "--mode",
        choices=["plain", "stages", "cprofile"],
        default="stages",
        help="Profiling mode (plain, stages, cprofile)",
    )
    args = parser.parse_args()

    if args.shards <= 0:
        print(
            f"Invalid --shards {args.shards}: must be positive integer",
            file=sys.stderr,
        )
        return 1
    if args.cache_mib < 0:
        print(
            f"Invalid --cache-mib {args.cache_mib}: must be non-negative",
            file=sys.stderr,
        )
        return 1
    if os.path.exists(args.output):
        print(
            f"Output directory already exists: {args.output}", file=sys.stderr
        )
        return 1

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    if not isinstance(manifest_data, dict) or "shards" not in manifest_data:
        print("Manifest missing 'shards' key", file=sys.stderr)
        return 1
    shards_list = manifest_data["shards"]
    if not isinstance(shards_list, list) or len(shards_list) == 0:
        print("Manifest shards list empty", file=sys.stderr)
        return 1

    num_shards = min(len(shards_list), args.shards)
    selected_shards = shards_list[:num_shards]
    for idx, s in enumerate(selected_shards):
        _validate_shard_argv(s.get("argv", []), idx)

    mem_initial_bytes = _get_current_rss_bytes()

    t_import0 = time.perf_counter()
    import torch as _torch
    import psutil as _psutil
    torch = _torch
    psutil = _psutil
    _PROCESS_HANDLE = psutil.Process()
    t_imports_s = time.perf_counter() - t_import0

    t_cuda0 = time.perf_counter()
    cuda_device_name = "none"
    if torch.cuda.is_available():
        torch.cuda.init()
        cuda_device_name = torch.cuda.get_device_name(0)
    t_cuda_init_s = time.perf_counter() - t_cuda0

    os.makedirs(args.output, exist_ok=False)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_entrypoint = os.path.join(repo_root, "bin", "pycbc_inspiral")
    disposable_entrypoint = os.path.join(
        args.output, "instrumented_pycbc_inspiral"
    )

    if args.mode == "stages":
        instrument_entrypoint(base_entrypoint, disposable_entrypoint)

    source_hashes = {}
    for path_rel in [
        "bin/pycbc_inspiral",
        "tools/run_inspiral_campaign.py",
        "tools/profile_inspiral_pipeline.py",
        "pycbc/filter/gpu_search/inspiral_session.py",
    ]:
        p = os.path.join(repo_root, path_rel)
        if os.path.isfile(p):
            source_hashes[path_rel] = hashlib.sha256(
                Path(p).read_bytes()
            ).hexdigest()

    receipt_path = os.path.join(args.output, "profile_receipt.json")
    shard_reports: List[Dict[str, Any]] = []
    overall_success = True
    suite_error = None

    def _write_receipt(
        status: str,
        err: Optional[str] = None,
        session_cleanup_meta: Optional[Dict[str, Any]] = None,
    ):
        rec = {
            "manifest": os.path.abspath(args.manifest),
            "output_dir": os.path.abspath(args.output),
            "mode": args.mode,
            "shards_requested": num_shards,
            "status": status,
            "error": err,
            "cuda_device": cuda_device_name,
            "source_sha256": source_hashes,
            "limitations": [
                "Asynchronous GPU execution: stage wall and CPU times "
                "reflect enqueue durations unless explicitly synchronized; "
                "use NVTX markers with nsys for kernel execution times.",
                "Campaign spawn/subprocess overhead is not measured; "
                "profile runner executes directly in-process.",
                "CUDA initialization and PyTorch imports are benchmarked "
                "during startup and moved ahead of shard execution.",
            ],
            "startup_timings": {
                "process_startup_to_main_s": t_main_start - T0_PROCESS,
                "host_rss_initial_bytes": mem_initial_bytes,
                "heavy_imports_s": t_imports_s,
                "cuda_init_s": t_cuda_init_s,
            },
            "after_session_cleanup": session_cleanup_meta or {},
            "total_process_wall_s": time.perf_counter() - T0_PROCESS,
            "shards": shard_reports,
        }
        tmp_path = receipt_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as rf:
            json.dump(rec, rf, indent=2)
        os.replace(tmp_path, receipt_path)

    cleanup_meta: Dict[str, Any] = {}
    try:
        for idx, shard_info in enumerate(selected_shards):
            shard_out = os.path.abspath(
                os.path.join(args.output, f"shard_{idx}.hdf")
            )
            task_spec = {
                "shard_idx": idx,
                "argv": shard_info["argv"],
                "output_file": shard_out,
                "cache_bytes": args.cache_mib * 1024 * 1024,
                "inspiral_entrypoint": base_entrypoint,
            }
            rep = run_profiled_shard(
                task_spec,
                args.mode,
                base_entrypoint,
                disposable_entrypoint,
                args.output,
            )
            shard_reports.append(rep)
            _write_receipt("in_progress")
            if rep["status"] != "completed":
                overall_success = False
                suite_error = rep.get("error")
                break
    except Exception as exc:
        overall_success = False
        suite_error = f"{type(exc).__name__}: {str(exc)}"
    finally:
        t_clean0 = time.perf_counter()
        _worker_cleanup()
        t_clean_dur = time.perf_counter() - t_clean0
        cleanup_meta = {
            "cleanup_s": t_clean_dur,
            "cuda_memory_after_session_cleanup": _capture_cuda_stats(),
            "host_rss_after_session_cleanup_bytes": _get_current_rss_bytes(),
        }
        final_status = "completed" if overall_success else "failed"
        _write_receipt(
            final_status, err=suite_error, session_cleanup_meta=cleanup_meta
        )

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
