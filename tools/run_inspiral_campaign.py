"""CLI tool and API for executing pycbc_inspiral persistent worker campaigns.

Executes sequential inspiral shards using a single spawned worker process,
retaining in-memory waveform and sigmasq caches across shards.

Manifest schema:
  {
    "shards": [
      {
        "argv": [
          "--bank-file", "/abs/bank.hdf",
          "--sample-rate", "2048",
          "--batch-size", "16",
          "--processing-scheme", "torch:cpu:1"
        ]
      }
    ]
  }

Constraints:
  - Supported schemes: Torch CPU or CUDA (e.g. torch:cpu:1, torch:cuda:0).
  - Batch size must satisfy batch-size >= 2.
  - Multiprocessing and checkpointing options are forbidden.
  - The runner strictly owns shard output locations (--output is rejected).
  - Relative file paths resolve against the working directory.

Usage:
  PYTHONPATH=. python tools/run_inspiral_campaign.py \
    --manifest /path/to/manifest.json \
    --output /path/to/output_dir \
    --cache-mib 256
"""

import argparse
import gc
import json
import os
import runpy
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any, Dict, List, Optional

_WORKER_SESSION: Optional[Any] = None


def _worker_cleanup() -> None:
    global _WORKER_SESSION
    if _WORKER_SESSION is not None:
        try:
            _WORKER_SESSION.close()
        except Exception:
            pass
    _WORKER_SESSION = None
    gc.collect()


def _validate_shard_argv(argv: Any, shard_idx: int = 0) -> List[str]:
    if not isinstance(argv, list) or len(argv) == 0:
        raise ValueError(f"Shard {shard_idx} needs a non-empty argv list")

    for arg in argv:
        if not isinstance(arg, str) or not arg.strip():
            raise ValueError(
                f"Shard {shard_idx} has invalid argument: {arg}"
            )

    final_batch = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--output" or arg.startswith("--output="):
            raise ValueError(
                f"Shard {shard_idx}: --output is managed exclusively by runner"
            )
        if arg in (
            "--checkpoint-interval",
            "--require-valid-checkpoint",
            "--checkpoint-exit-maxtime",
            "--checkpoint-exit-code",
        ) or arg.startswith("--checkpoint-"):
            raise ValueError(
                f"Shard {shard_idx}: checkpoint option '{arg}' is forbidden"
            )
        if arg == "--multiprocessing-nprocesses" or arg.startswith(
            "--multiprocessing-nprocesses="
        ):
            raise ValueError(
                f"Shard {shard_idx}: multiprocessing options are forbidden"
            )
        if arg in ("--help", "-h", "--version"):
            raise ValueError(
                f"Shard {shard_idx}: informational flags cannot be used"
            )

        if arg == "--batch-size" and i + 1 < len(argv):
            try:
                final_batch = int(argv[i + 1])
            except ValueError:
                raise ValueError(f"Shard {shard_idx}: invalid --batch-size")
            i += 1
        elif arg.startswith("--batch-size="):
            try:
                final_batch = int(arg.split("=")[1])
            except ValueError:
                raise ValueError(f"Shard {shard_idx}: invalid --batch-size")
        i += 1

    if final_batch is None or final_batch < 2:
        raise ValueError(
            f"Shard {shard_idx} requires --batch-size >= 2 for campaigns"
        )

    return [str(a) for a in argv]


def execute_inspiral_shard(task_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single inspiral shard inside the spawned worker process."""
    global _WORKER_SESSION
    t_start = time.perf_counter()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    orig_argv = sys.argv
    run_res = None

    try:
        if not isinstance(task_spec, dict):
            raise ValueError("task_spec must be a dictionary")
        shard_idx = task_spec.get("shard_idx", 0)
        argv = _validate_shard_argv(task_spec.get("argv"), shard_idx)
        shard_output = str(task_spec.get("output_file", ""))
        if not shard_output or not os.path.isabs(shard_output):
            raise ValueError(f"Invalid absolute output path: {shard_output}")
        if os.path.exists(shard_output):
            raise FileExistsError(
                f"Output file already exists before run: {shard_output}"
            )

        raw_cache_bytes = task_spec.get("cache_bytes", 268435456)
        if isinstance(raw_cache_bytes, bool) or not isinstance(
            raw_cache_bytes, int
        ) or raw_cache_bytes < 0:
            raise ValueError("cache_bytes must be a non-negative integer")
        cache_bytes = int(raw_cache_bytes)

        inspiral_entrypoint = task_spec.get("inspiral_entrypoint")
        if inspiral_entrypoint is None:
            inspiral_entrypoint = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "bin", "pycbc_inspiral"
                )
            )

        from pycbc.filter.gpu_search.inspiral_session import InspiralSession
        import torch
        torch.set_num_threads(1)

        pid = os.getpid()
        full_argv = ["pycbc_inspiral"] + argv + ["--output", shard_output]

        if (_WORKER_SESSION is None
                or _WORKER_SESSION.cache_bytes != cache_bytes):
            _worker_cleanup()
            _WORKER_SESSION = InspiralSession(cache_bytes=cache_bytes)

        stats_before = dict(_WORKER_SESSION.stats)
        sys.argv = full_argv

        try:
            run_res = runpy.run_path(
                inspiral_entrypoint,
                run_name="__main__",
                init_globals={"_inspiral_session": _WORKER_SESSION},
            )
        except SystemExit as se:
            code = se.code if se.code is not None else 0
            if code != 0:
                _worker_cleanup()
                raise RuntimeError(
                    f"pycbc_inspiral failed with exit code {code}"
                ) from se

        if not os.path.isfile(shard_output):
            _worker_cleanup()
            raise RuntimeError(
                f"Output file was not generated: {shard_output}"
            )

        stats_after = dict(_WORKER_SESSION.stats)
        stats_delta = {
            k: stats_after[k] - stats_before.get(k, 0) for k in stats_after
        }

        result = {
            "pid": pid,
            "output_file": shard_output,
            "elapsed_s": None,
            "argv": full_argv,
            "cache_stats": stats_after,
            "cache_delta": stats_delta,
            "cache_bytes_used": _WORKER_SESSION.current_bytes,
        }

    except Exception:
        _worker_cleanup()
        raise
    finally:
        sys.argv = orig_argv
        if run_res is not None:
            for k in ["bank", "bank_chisq"]:
                obj = run_res.get(k)
                if obj is not None:
                    f_h = getattr(obj, "file", None)
                    if f_h is not None:
                        try:
                            f_h.close()
                        except Exception:
                            pass
            run_res.clear()
            del run_res
        gc.collect()
        if _WORKER_SESSION is not None:
            _WORKER_SESSION.end_shard()

    result["elapsed_s"] = time.perf_counter() - t_start
    return result


def _validate_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict) or "shards" not in manifest:
        raise ValueError("Manifest must be a JSON object containing 'shards'")

    shards = manifest["shards"]
    if not isinstance(shards, list) or len(shards) == 0:
        raise ValueError("Manifest shards must be a non-empty list")

    for idx, s in enumerate(shards):
        if not isinstance(s, dict) or "argv" not in s:
            raise ValueError(f"Shard {idx} missing 'argv' list")
        _validate_shard_argv(s["argv"], idx)

    return shards


def run_campaign(
    manifest_path: str,
    output_path: str,
    fresh_workers: bool = False,
    cache_bytes: int = 268435456,
    inspiral_entrypoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a full pycbc_inspiral campaign across manifest shards."""
    t_camp_start = time.perf_counter()

    if (isinstance(cache_bytes, bool)
            or not isinstance(cache_bytes, int) or cache_bytes < 0):
        raise ValueError("cache_bytes must be a non-negative integer")

    shards = _validate_manifest(manifest_path)

    if os.path.exists(output_path):
        raise FileExistsError(f"Output directory exists: {output_path}")

    os.makedirs(output_path, exist_ok=False)
    mp_ctx = get_context("spawn")

    task_specs = []
    for idx, s in enumerate(shards):
        shard_out = os.path.abspath(
            os.path.join(output_path, f"shard_{idx}.hdf"))
        task_specs.append({
            "argv": s["argv"],
            "output_file": shard_out,
            "cache_bytes": cache_bytes,
            "inspiral_entrypoint": inspiral_entrypoint,
            "shard_idx": idx,
        })

    receipt_path = os.path.join(output_path, "receipt.json")
    shard_receipts: List[Dict[str, Any]] = []
    persistent_executor: Optional[ProcessPoolExecutor] = None
    cleanup_exc: Optional[Exception] = None
    failure_idx: Optional[int] = None
    campaign_exc: Optional[Exception] = None

    def _write_receipt(status, exc=None):
        rec = {
            "manifest": os.path.abspath(manifest_path),
            "output_dir": os.path.abspath(output_path),
            "entrypoint": inspiral_entrypoint,
            "fresh_workers": fresh_workers,
            "cache_bytes": cache_bytes,
            "status": status,
            "failure_index": failure_idx,
            "error": str(exc) if exc else None,
            "total_campaign_s": time.perf_counter() - t_camp_start,
            "shards": shard_receipts,
        }
        tmp_receipt = receipt_path + ".tmp"
        with open(tmp_receipt, "w", encoding="utf-8") as rf:
            json.dump(rec, rf, indent=2)
        os.replace(tmp_receipt, receipt_path)
        return rec

    try:
        if not fresh_workers:
            persistent_executor = ProcessPoolExecutor(
                max_workers=1, mp_context=mp_ctx)

        for spec in task_specs:
            executor = persistent_executor
            if fresh_workers:
                executor = ProcessPoolExecutor(
                    max_workers=1, mp_context=mp_ctx)

            try:
                fut = executor.submit(execute_inspiral_shard, spec)
                res = fut.result()
                shard_receipts.append(res)
                _write_receipt("in_progress")
            except Exception as e:
                failure_idx = spec["shard_idx"]
                campaign_exc = e
                raise
            finally:
                if fresh_workers and executor is not None:
                    try:
                        executor.submit(_worker_cleanup).result()
                    except Exception as ce:
                        if cleanup_exc is None:
                            cleanup_exc = ce
                    finally:
                        executor.shutdown(wait=True)

    except Exception as exc:
        if campaign_exc is None:
            campaign_exc = exc
    finally:
        if persistent_executor is not None:
            try:
                persistent_executor.submit(_worker_cleanup).result()
            except Exception as ce:
                if cleanup_exc is None:
                    cleanup_exc = ce
            finally:
                persistent_executor.shutdown(wait=True)

        primary_err = campaign_exc or cleanup_exc
        final_status = "completed" if primary_err is None else "failed"
        final_rec = _write_receipt(final_status, primary_err)

    if campaign_exc is not None:
        raise campaign_exc
    if cleanup_exc is not None:
        raise cleanup_exc

    return final_rec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument(
        "--output", required=True, help="Path to output directory")
    parser.add_argument(
        "--fresh-workers", action="store_true",
        help="Spawn fresh worker per shard")
    parser.add_argument(
        "--cache-mib", type=int, default=256, help="Cache budget in MiB")
    args = parser.parse_args()

    try:
        run_campaign(
            manifest_path=args.manifest,
            output_path=args.output,
            fresh_workers=args.fresh_workers,
            cache_bytes=args.cache_mib * 1024 * 1024,
        )
        return 0
    except Exception as e:
        print(f"Campaign failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
