"""Persistent real-file SearchEngine campaign runner.

Executes candidate-search campaigns across file-backed shards (NPZ banks,
PSDs, and strain segments) using a single spawned worker process.
Reuses BankPlan, SearchEngine, and bound PSDPlan instances across shards
when cryptographic content hashes and all search/clustering parameters match.

Input NPZ Schema:
  bank.npz:   templates (complex64 [B, flen]), template_ids (int64 [B]),
              delta_f (scalar float > 0)
  psd.npz:    psd (float64 [flen]), delta_f (scalar float > 0)
  strain.npz: strain (complex64 [flen]), delta_f (scalar float > 0)

Output NPZ Schema (block_<block_id>.npz):
  block_id: np.int64
  tile<idx>_template_id:  np.ndarray (native dtype preserved from SearchEngine)
  tile<idx>_template_idx: np.ndarray (native dtype preserved from SearchEngine)
  tile<idx>_sample_idx:   np.ndarray (native dtype preserved from SearchEngine)
  tile<idx>_snr:          np.ndarray (native dtype preserved from SearchEngine)
  tile<idx>_sigmasq:      np.ndarray (native dtype preserved from SearchEngine)

Note: This tool produces candidate event arrays from matched filtering and
clustering; it is not a full pycbc_inspiral replacement and performs no chisq
vetoes or XML table formatting.

Example:
  PYTHONPATH=. python tools/run_torch_search_campaign.py \
    --manifest /path/to/manifest.json \
    --output /path/to/output_dir \
    --device cpu --tile-size 16 --snr-threshold 5.5 --cluster-window 64
"""

import argparse
import atexit
import hashlib
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_WORKER_CACHE: Dict[str, Any] = {
    "bank_key": None,
    "bank_plan": None,
    "engine": None,
    "psd_key": None,
    "psd_plan": None,
}


def _close_state() -> None:
    """Close engine and unconditionally reset cache dictionary."""
    engine = _WORKER_CACHE.get("engine")
    try:
        if engine is not None:
            engine.close()
    finally:
        _WORKER_CACHE.update(
            {
                "bank_key": None,
                "bank_plan": None,
                "engine": None,
                "psd_key": None,
                "psd_plan": None,
            }
        )


def _atexit_cleanup() -> None:
    try:
        _close_state()
    except Exception:
        pass


atexit.register(_atexit_cleanup)


def _read_and_hash(path: str) -> Tuple[bytes, str]:
    data = Path(path).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def execute_shard(shard_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single shard task inside the spawned worker process."""
    import torch
    from pycbc.filter.gpu_search.candidates import SelectionPolicy
    from pycbc.filter.gpu_search.engine import SearchEngine
    from pycbc.filter.gpu_search.plans import bind_psd, prepare_bank
    from pycbc.filter.matchedfilter import get_cutoff_indices
    from pycbc.types import FrequencySeries

    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    t_start = time.perf_counter()
    pid = os.getpid()

    try:
        bank_path = shard_spec["bank"]
        psd_path = shard_spec["psd"]
        strain_path = shard_spec["strain"]
        block_id = shard_spec["block_id"]
        if isinstance(block_id, bool) or not isinstance(block_id, int):
            raise ValueError(f"block_id must be an integer, got {block_id}")
        if not (-0x8000000000000000 <= block_id <= 0x7FFFFFFFFFFFFFFF):
            raise ValueError(f"block_id {block_id} does not fit in int64")

        val_interval = shard_spec["valid_interval"]
        if (
            not isinstance(val_interval, (list, tuple))
            or len(val_interval) != 2
        ):
            raise ValueError(
                "valid_interval must be a list/tuple of 2 integers"
            )
        start_idx, end_idx = val_interval
        if (
            isinstance(start_idx, bool)
            or isinstance(end_idx, bool)
            or not isinstance(start_idx, int)
            or not isinstance(end_idx, int)
        ):
            raise ValueError(
                "valid_interval bounds must be non-boolean integers"
            )

        device = shard_spec["device"]
        if device != "cpu" and not re.match(r"^cuda:\d+$", device):
            raise ValueError(f"Invalid device: {device}")

        tile_size = shard_spec["tile_size"]
        if (
            isinstance(tile_size, bool)
            or not isinstance(tile_size, int)
            or tile_size <= 0
        ):
            raise ValueError(
                "tile_size must be a positive non-boolean integer"
            )

        snr_thresh = float(shard_spec["snr_threshold"])
        if not np.isfinite(snr_thresh) or snr_thresh <= 0.0:
            raise ValueError("snr_threshold must be finite and positive")

        cluster_window = shard_spec["cluster_window"]
        if (
            isinstance(cluster_window, bool)
            or not isinstance(cluster_window, int)
            or cluster_window < 0
        ):
            raise ValueError(
                "cluster_window must be a non-negative non-boolean integer"
            )

        f_lower = float(shard_spec["f_lower"])
        if not np.isfinite(f_lower) or f_lower <= 0.0:
            raise ValueError("f_lower must be finite and positive")
        f_upper = shard_spec.get("f_upper")
        if f_upper is not None:
            f_upper = float(f_upper)
            if not np.isfinite(f_upper) or f_upper <= f_lower:
                raise ValueError(
                    "f_upper must be finite and greater than f_lower"
                )

        bank_bytes, bank_hash = _read_and_hash(bank_path)
        psd_bytes, psd_hash = _read_and_hash(psd_path)
        strain_bytes, strain_hash = _read_and_hash(strain_path)

        bank_key = (
            bank_hash,
            tile_size,
            snr_thresh,
            cluster_window,
            f_lower,
            f_upper,
            device,
        )
        psd_key = (psd_hash, bank_key)

        cache_hit_bank = False
        cache_hit_psd = False
        t_bank_prep = 0.0
        t_psd_prep = 0.0

        # 1. Bank and SearchEngine Management
        if (
            _WORKER_CACHE["bank_key"] != bank_key
            or _WORKER_CACHE["engine"] is None
        ):
            _close_state()
            t0 = time.perf_counter()
            with np.load(io.BytesIO(bank_bytes), allow_pickle=False) as bdata:
                tmps = bdata["templates"]
                t_ids = bdata["template_ids"]
                delta_f = bdata["delta_f"]

            if (
                delta_f.ndim != 0
                or not np.isfinite(delta_f)
                or float(delta_f) <= 0.0
            ):
                raise ValueError(f"Invalid bank delta_f: {delta_f}")
            df = float(delta_f)

            if tmps.dtype != np.complex64 or tmps.ndim != 2:
                raise ValueError("templates must be 2D complex64")
            if tmps.shape[0] == 0 or tmps.shape[1] < 2:
                raise ValueError(f"Bank empty or invalid shape: {tmps.shape}")
            if not np.all(np.isfinite(tmps)):
                raise ValueError("Bank templates contain non-finite values")

            flen = tmps.shape[1]
            nyquist = (flen - 1) * df
            if not (0.0 < f_lower < nyquist):
                raise ValueError(
                    f"f_lower ({f_lower}) must be in (0, Nyquist={nyquist})"
                )
            if f_upper is not None and not (f_lower < f_upper <= nyquist):
                raise ValueError(
                    f"f_upper={f_upper} must be in ({f_lower}, {nyquist}]"
                )

            if t_ids.dtype != np.int64 or t_ids.ndim != 1:
                raise ValueError("template_ids must be 1D int64")
            if len(tmps) != len(t_ids):
                raise ValueError(
                    "Mismatch between templates and template_ids lengths"
                )
            if len(np.unique(t_ids)) != len(t_ids):
                raise ValueError("template_ids must be unique")

            fseries_list = []
            for idx, arr in enumerate(tmps):
                fs = FrequencySeries(arr, delta_f=df)
                fs.id = int(t_ids[idx])
                fseries_list.append(fs)

            bank_plan = prepare_bank(
                fseries_list,
                tile_size=tile_size,
                f_lower=f_lower,
                f_upper=f_upper,
                device=device,
            )
            policy = SelectionPolicy(
                snr_threshold=snr_thresh,
                cluster_policy="symmetric",
                cluster_window=cluster_window,
            )
            engine = SearchEngine(
                bank_plan,
                policy,
                device=device,
                use_cuda_graphs=False,
                num_workspaces=1,
                enable_async_transfers=False,
                num_threads=1,
            )
            t_bank_prep = time.perf_counter() - t0
            _WORKER_CACHE["bank_key"] = bank_key
            _WORKER_CACHE["bank_plan"] = bank_plan
            _WORKER_CACHE["engine"] = engine
        else:
            cache_hit_bank = True

        bank_plan = _WORKER_CACHE["bank_plan"]
        engine = _WORKER_CACHE["engine"]
        flen = bank_plan.geometry.filter_length
        tlen = bank_plan.geometry.transform_length
        df = bank_plan.geometry.delta_f

        if not (0 <= start_idx < end_idx <= tlen):
            raise ValueError(
                f"valid_interval [{start_idx}, {end_idx}] out of [0, {tlen}]"
            )

        # 2. PSD Plan Management
        if (
            _WORKER_CACHE["psd_key"] != psd_key
            or _WORKER_CACHE["psd_plan"] is None
        ):
            t0 = time.perf_counter()
            with np.load(io.BytesIO(psd_bytes), allow_pickle=False) as pdata:
                psd_arr = pdata["psd"]
                psd_df = pdata["delta_f"]

            if psd_df.ndim != 0 or float(psd_df) != df:
                raise ValueError(f"PSD delta_f {psd_df} != bank delta_f {df}")
            if (
                psd_arr.dtype != np.float64
                or psd_arr.ndim != 1
                or len(psd_arr) != flen
            ):
                raise ValueError(f"PSD must be 1D float64 of length {flen}")
            if not np.all(np.isfinite(psd_arr)):
                raise ValueError("PSD contains non-finite values")

            kmin, kmax = get_cutoff_indices(f_lower, f_upper, df, tlen)
            if np.any(psd_arr[kmin:kmax] <= 0.0):
                raise ValueError(
                    "PSD must be strictly positive in active band"
                )

            psd_fs = FrequencySeries(psd_arr, delta_f=df)
            psd_plan = bind_psd(bank_plan, psd_fs, device=device)
            t_psd_prep = time.perf_counter() - t0
            _WORKER_CACHE["psd_key"] = psd_key
            _WORKER_CACHE["psd_plan"] = psd_plan
        else:
            cache_hit_psd = True

        psd_plan = _WORKER_CACHE["psd_plan"]

        # 3. Strain and Search Execution
        with np.load(io.BytesIO(strain_bytes), allow_pickle=False) as sdata:
            strain_arr = sdata["strain"]
            strain_df = sdata["delta_f"]

        if strain_df.ndim != 0 or float(strain_df) != df:
            raise ValueError(
                f"Strain delta_f {strain_df} != bank delta_f {df}"
            )
        if (
            strain_arr.dtype != np.complex64
            or strain_arr.ndim != 1
            or len(strain_arr) != flen
        ):
            raise ValueError(f"Strain must be 1D complex64 of length {flen}")
        if not np.all(np.isfinite(strain_arr)):
            raise ValueError("Strain contains non-finite values")

        strain_fs = FrequencySeries(strain_arr, delta_f=df)
        t_search0 = time.perf_counter()
        engine.submit(
            strain_fs,
            psd_plan,
            valid_interval=(start_idx, end_idx),
            block_id=block_id,
            already_overwhitened=False,
        )
        tickets = engine.drain()
        t_search = time.perf_counter() - t_search0

        if len(tickets) != 1:
            raise RuntimeError(
                f"Expected exactly 1 drained ticket, got {len(tickets)}"
            )
        tkt = tickets[0]
        if not tkt.completed or tkt.aborted or tkt.overflow:
            raise RuntimeError(
                f"Ticket failed: completed={tkt.completed}, "
                f"abort={tkt.aborted}, overflow={tkt.overflow}"
            )

        # Engine tickets omit empty tiles; use globally unique template IDs
        # to retain the original bank tile index in every output file.
        field_dtypes = dict(
            template_id=np.int64,
            template_idx=np.int64,
            sample_idx=np.int64,
            snr=np.complex64,
            sigmasq=np.float32,
        )
        tile_results = [
            {k: np.empty(0, dtype=dtype) for k, dtype in field_dtypes.items()}
            for _ in bank_plan.tiles
        ]
        id_to_tile = {
            tid: index
            for index, tile in enumerate(bank_plan.tiles)
            for tid in tile.template_ids
        }
        seen_tiles = set()
        for cand_dict in tkt.results:
            ids = cand_dict["template_id"]
            if len(ids) == 0:
                continue
            index = id_to_tile[int(ids[0])]
            if index in seen_tiles or any(
                id_to_tile[int(tid)] != index for tid in ids
            ):
                raise RuntimeError(
                    "Unexpected mixed or duplicate tile results"
                )
            seen_tiles.add(index)
            tile_results[index] = {
                key: np.array(cand_dict[key], copy=True)
                for key in field_dtypes
            }

        return {
            "pid": pid,
            "block_id": block_id,
            "bank_hash": bank_hash,
            "psd_hash": psd_hash,
            "strain_hash": strain_hash,
            "cache_hit_bank": cache_hit_bank,
            "cache_hit_psd": cache_hit_psd,
            "t_bank_prep": t_bank_prep,
            "t_psd_prep": t_psd_prep,
            "t_search": t_search,
            "t_total": time.perf_counter() - t_start,
            "tile_results": tile_results,
            "ticket_completed": bool(tkt.completed),
            "ticket_aborted": bool(tkt.aborted),
            "ticket_overflow": bool(tkt.overflow),
        }
    except Exception:
        _close_state()
        raise


def run_campaign(
    manifest_path: str,
    output_path: str,
    device: str = "cpu",
    tile_size: int = 16,
    snr_threshold: float = 5.5,
    cluster_window: int = 64,
    f_lower: float = 20.0,
    f_upper: Optional[float] = None,
    fresh_workers: bool = False,
) -> Dict[str, Any]:
    """Run a full search campaign over manifest shards."""
    t_camp_start = time.perf_counter()

    if os.path.exists(output_path):
        raise FileExistsError(
            f"Output directory already exists: {output_path}"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict) or "shards" not in manifest:
        raise ValueError("Manifest must be a JSON object containing 'shards'")

    shards = manifest["shards"]
    if not isinstance(shards, list) or len(shards) == 0:
        raise ValueError("Manifest shards must be a non-empty list")

    seen_blocks = set()
    for s in shards:
        if not isinstance(s, dict):
            raise ValueError("Each shard entry must be a JSON object (dict)")
        for req in ["bank", "psd", "strain", "block_id", "valid_interval"]:
            if req not in s:
                raise ValueError(f"Missing required shard key '{req}'")
        b_id = s["block_id"]
        if isinstance(b_id, bool) or not isinstance(b_id, int):
            raise ValueError(f"block_id must be int, got {b_id}")
        if not (-0x8000000000000000 <= b_id <= 0x7FFFFFFFFFFFFFFF):
            raise ValueError(f"block_id {b_id} does not fit in int64")
        if b_id in seen_blocks:
            raise ValueError(f"Duplicate block_id {b_id} in manifest")
        seen_blocks.add(b_id)

        val_interval = s["valid_interval"]
        if (
            not isinstance(val_interval, (list, tuple))
            or len(val_interval) != 2
        ):
            raise ValueError(
                "valid_interval must be a list/tuple of 2 integers"
            )
        if (
            isinstance(val_interval[0], bool)
            or isinstance(val_interval[1], bool)
            or not isinstance(val_interval[0], int)
            or not isinstance(val_interval[1], int)
        ):
            raise ValueError(
                "valid_interval bounds must be non-boolean integers"
            )

        for p_key in ["bank", "psd", "strain"]:
            val = s[p_key]
            if not isinstance(val, str):
                raise ValueError(
                    f"Path for '{p_key}' must be a string, got {type(val)}"
                )
            if not os.path.isabs(val):
                raise ValueError(f"Path {val} for '{p_key}' must be absolute")

    os.makedirs(output_path, exist_ok=False)
    mp_ctx = get_context("spawn")

    task_specs = []
    for s in shards:
        spec = dict(s)
        spec.update(
            {
                "device": device,
                "tile_size": tile_size,
                "snr_threshold": snr_threshold,
                "cluster_window": cluster_window,
                "f_lower": f_lower,
                "f_upper": f_upper,
            }
        )
        task_specs.append(spec)

    worker_receipts = []
    persistent_executor: Optional[ProcessPoolExecutor] = None
    cleanup_exc: Optional[Exception] = None
    try:
        if not fresh_workers:
            persistent_executor = ProcessPoolExecutor(
                max_workers=1, mp_context=mp_ctx
            )

        for spec in task_specs:
            executor = persistent_executor
            if fresh_workers:
                executor = ProcessPoolExecutor(
                    max_workers=1, mp_context=mp_ctx
                )
            try:
                fut = executor.submit(execute_shard, spec)
                res = fut.result()
            finally:
                if fresh_workers and executor is not None:
                    try:
                        executor.submit(_close_state).result()
                    except Exception as e:
                        if cleanup_exc is None:
                            cleanup_exc = e
                    finally:
                        executor.shutdown(wait=True)

            block_id = res["block_id"]
            tile_results = res.pop("tile_results")
            shard_save_dict: Dict[str, Any] = {"block_id": np.int64(block_id)}
            for t_idx, t_res in enumerate(tile_results):
                prefix = f"tile{t_idx}_"
                for k, v in t_res.items():
                    shard_save_dict[f"{prefix}{k}"] = v

            shard_npz_path = os.path.join(output_path, f"block_{block_id}.npz")
            np.savez(shard_npz_path, **shard_save_dict)

            res["output_file"] = shard_npz_path
            res["num_tiles"] = len(tile_results)
            res["candidate_counts"] = [
                int(len(t["sample_idx"])) for t in tile_results
            ]
            worker_receipts.append(res)
    finally:
        if persistent_executor is not None:
            try:
                persistent_executor.submit(_close_state).result()
            except Exception as e:
                if cleanup_exc is None:
                    cleanup_exc = e
            finally:
                persistent_executor.shutdown(wait=True)

    if cleanup_exc is not None:
        raise cleanup_exc

    t_camp_elapsed = time.perf_counter() - t_camp_start
    receipt: Dict[str, Any] = {
        "manifest": os.path.abspath(manifest_path),
        "output_dir": os.path.abspath(output_path),
        "device": device,
        "tile_size": tile_size,
        "fresh_workers": fresh_workers,
        "total_campaign_s": t_camp_elapsed,
        "tasks": worker_receipts,
    }

    receipt_path = os.path.join(output_path, "receipt.json")
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    return receipt


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="Path to manifest JSON"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--device", default="cpu", help="Compute device (cpu, cuda:0)"
    )
    parser.add_argument(
        "--tile-size", type=int, default=16, help="Batch tile size"
    )
    parser.add_argument(
        "--snr-threshold", type=float, default=5.5, help="SNR cutoff"
    )
    parser.add_argument(
        "--cluster-window", type=int, default=64, help="Clustering window"
    )
    parser.add_argument(
        "--f-lower", type=float, default=20.0, help="Lower frequency"
    )
    parser.add_argument(
        "--f-upper", type=float, default=None, help="Upper frequency"
    )
    parser.add_argument(
        "--fresh-workers",
        action="store_true",
        help="Spawn fresh worker per shard",
    )

    args = parser.parse_args()
    try:
        run_campaign(
            manifest_path=args.manifest,
            output_path=args.output,
            device=args.device,
            tile_size=args.tile_size,
            snr_threshold=args.snr_threshold,
            cluster_window=args.cluster_window,
            f_lower=args.f_lower,
            f_upper=args.f_upper,
            fresh_workers=args.fresh_workers,
        )
        return 0
    except Exception as e:
        print(f"Campaign execution failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
