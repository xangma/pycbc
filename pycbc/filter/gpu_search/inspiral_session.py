"""Inspiral session manager for persistent pycbc_inspiral worker campaigns.

Provides waveform snapshot caching and scalar sigmasq caching across shards
within a persistent worker process, bound by an explicit LRU memory budget.
"""

import collections
import hashlib
import json
import types
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pycbc.waveform.bank import sigma_cached


SCALAR_OVERHEAD_BYTES = 1024
TEMPLATE_OVERHEAD_BYTES = 2048


def _hash_file_stream(path: str, chunk_size: int = 65536) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _serialize_scalar_item(val: Any) -> Any:
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="surrogateescape")
    if isinstance(val, np.generic):
        item = val.item()
        if isinstance(item, (bytes, bytearray)):
            return item.decode("utf-8", errors="surrogateescape")
        return item
    raise TypeError(f"Unsupported object table value type: {type(val)}")


def _fingerprint_table(table: Any) -> Tuple[str, int]:
    if table is None:
        return "none", 0
    arr = np.asarray(table)
    num_rows = len(arr)
    hasher = hashlib.sha256()
    if arr.dtype.names:
        for name in sorted(arr.dtype.names):
            col = arr[name]
            hasher.update(name.encode("utf-8"))
            hasher.update(str(col.dtype).encode("ascii"))
            if col.dtype == object:
                serialized = [
                    _serialize_scalar_item(x)
                    for x in col.tolist()
                ]
                payload = json.dumps(serialized, sort_keys=True)
                hasher.update(payload.encode("utf-8"))
            else:
                hasher.update(np.ascontiguousarray(col).tobytes())
    else:
        hasher.update(str(arr.dtype).encode("ascii"))
        if arr.dtype == object:
            serialized = [
                _serialize_scalar_item(x)
                for x in arr.tolist()
            ]
            payload = json.dumps(serialized, sort_keys=True)
            hasher.update(payload.encode("utf-8"))
        else:
            hasher.update(np.ascontiguousarray(arr).tobytes())
    return hasher.hexdigest(), num_rows


class InspiralSession:
    """Session managing bounded cross-shard caching for pycbc_inspiral."""

    def __init__(self, cache_bytes: int = 268435456):
        if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int):
            raise ValueError("cache_bytes must be a non-boolean integer")
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be non-negative")
        self.cache_bytes: int = int(cache_bytes)
        self.current_bytes: int = 0
        self.bank_key: Optional[Tuple[Any, ...]] = None

        self.lru_cache: collections.OrderedDict[
            Tuple[Any, ...], Dict[str, Any]
        ] = collections.OrderedDict()
        self._shard_psds: Dict[int, Tuple[Any, str]] = {}

        self.stats: Dict[str, int] = {
            "bank_binds": 0,
            "bank_hits": 0,
            "bank_invalidations": 0,
            "template_hits": 0,
            "template_misses": 0,
            "sigmasq_hits": 0,
            "sigmasq_misses": 0,
            "evictions": 0,
        }

    def clear(self) -> None:
        """Unconditionally clear all cached waveforms and scalar sigmasqs."""
        self.lru_cache.clear()
        self._shard_psds.clear()
        self.current_bytes = 0
        self.bank_key = None

    def close(self) -> None:
        """Public alias for resource disposal."""
        self.clear()

    def end_shard(self) -> None:
        """Release shard-specific object references at boundary."""
        self._shard_psds.clear()

    def bind_bank(self, bank: Any, config_dict: Dict[str, Any]) -> None:
        """Validate the bank content and generation settings before reuse."""
        self.end_shard()
        self.stats["bank_binds"] += 1
        bank_path = getattr(bank, "filename", None)
        if not bank_path or not Path(bank_path).is_file():
            raise ValueError(f"FilterBank requires a file: {bank_path}")

        bank_content_hash = _hash_file_stream(str(bank_path))
        table_hash, num_rows = _fingerprint_table(getattr(bank, "table", None))

        cfg_json = json.dumps(config_dict, sort_keys=True, default=str)
        cfg_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()

        new_key = (
            bank_content_hash,
            table_hash,
            num_rows,
            getattr(bank, "filter_length", None),
            getattr(bank, "delta_f", None),
            str(getattr(bank, "dtype", None)),
            getattr(bank, "f_lower", None),
            getattr(bank, "max_template_length", None),
            getattr(bank, "enable_compressed_waveforms", None),
            getattr(bank, "waveform_decompression_method", None),
            json.dumps(getattr(bank, "extra_args", {}), sort_keys=True),
            cfg_hash,
        )

        if self.bank_key != new_key:
            if self.bank_key is not None:
                self.stats["bank_invalidations"] += 1
            self.clear()
            self.bank_key = new_key
        else:
            self.stats["bank_hits"] += 1

    def _evict_lru(self, required_bytes: int) -> None:
        while (
            self.lru_cache
            and self.current_bytes + required_bytes > self.cache_bytes
        ):
            _, entry = self.lru_cache.popitem(last=False)
            self.current_bytes -= entry["nbytes"]
            self.stats["evictions"] += 1

    def template(self, bank: Any, index: int) -> Any:
        """Retrieve or generate an isolated template snapshot."""
        cache_key = ("tmpl", int(index))
        if self.cache_bytes > 0 and cache_key in self.lru_cache:
            self.lru_cache.move_to_end(cache_key)
            cached = self.lru_cache[cache_key]
            self.stats["template_hits"] += 1

            dur = cached["template_duration"]
            try:
                bank.table[index].template_duration = dur
            except (AttributeError, IndexError):
                pass

            clone = cached["series"].copy()
            for attr, val in cached["attrs"].items():
                setattr(clone, attr, val)

            clone.params = bank.table[index]
            clone._sigmasq = {}
            clone.sigmasq = types.MethodType(sigma_cached, clone)
            return clone

        self.stats["template_misses"] += 1
        tmpl = bank[index]

        if self.cache_bytes > 0:
            est_bytes = tmpl.nbytes + TEMPLATE_OVERHEAD_BYTES
            if est_bytes <= self.cache_bytes:
                try:
                    dur = getattr(bank.table[index], "template_duration", None)
                except (AttributeError, IndexError):
                    dur = getattr(tmpl, "chirp_length", None)

                attrs = {}
                for attr in [
                    "f_lower",
                    "min_f_lower",
                    "end_idx",
                    "chirp_length",
                    "length_in_time",
                    "approximant",
                    "end_frequency",
                ]:
                    if hasattr(tmpl, attr):
                        attrs[attr] = getattr(tmpl, attr)

                self._evict_lru(est_bytes)
                snapshot = tmpl.copy()
                self.lru_cache[cache_key] = {
                    "series": snapshot,
                    "template_duration": dur,
                    "attrs": attrs,
                    "nbytes": est_bytes,
                }
                self.current_bytes += est_bytes

        return tmpl

    def _psd_digest(self, psd_obj: Any) -> str:
        psd_id = id(psd_obj)
        if psd_id in self._shard_psds:
            return self._shard_psds[psd_id][1]

        if hasattr(psd_obj, "numpy"):
            arr = psd_obj.numpy()
        else:
            arr = np.asarray(psd_obj)
        hasher = hashlib.sha256()
        hasher.update(str(arr.dtype).encode("ascii"))
        hasher.update(str(arr.shape).encode("ascii"))
        delta_f = str(getattr(psd_obj, "delta_f", "none"))
        hasher.update(delta_f.encode("ascii"))
        hasher.update(np.ascontiguousarray(arr).tobytes())
        digest = hasher.hexdigest()
        self._shard_psds[psd_id] = (psd_obj, digest)
        return digest

    def sigmasq(self, tmpl: Any, index: int, psd_obj: Any) -> Any:
        """Evaluate or retrieve cached scalar template sigmasq."""
        if self.cache_bytes <= 0:
            self.stats["sigmasq_misses"] += 1
            return tmpl.sigmasq(psd_obj)

        digest = self._psd_digest(psd_obj)
        cache_key = ("sigma", int(index), digest)
        if cache_key in self.lru_cache:
            self.lru_cache.move_to_end(cache_key)
            self.stats["sigmasq_hits"] += 1
            return self.lru_cache[cache_key]["val"]

        self.stats["sigmasq_misses"] += 1
        val = tmpl.sigmasq(psd_obj)
        est_bytes = SCALAR_OVERHEAD_BYTES
        if est_bytes <= self.cache_bytes:
            self._evict_lru(est_bytes)
            self.lru_cache[cache_key] = {
                "val": val,
                "nbytes": est_bytes,
            }
            self.current_bytes += est_bytes
        return val
