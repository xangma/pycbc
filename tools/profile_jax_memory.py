#!/usr/bin/env python3
"""Capture JAX device-memory snapshots during a PyCBC executable run.

The target is executed in this process with :func:`runpy.run_path`, so the
wrapper can observe the target's logging milestones without changing the
timed workload command.  A pprof device-memory profile and a small JSON
inventory are written for selected batches.

Usage::

    python tools/profile_jax_memory.py --output-dir /tmp/memory -- \
        bin/pycbc_inspiral <pycbc_inspiral arguments>

The inventory is intentionally a live-array snapshot.  It does not attempt
to identify equal-valued arrays, and it does not include every internal XLA
temporary or executable. The pprof profile adds allocation stacks and live
executables; compiler buffer assignments are needed for internal temporaries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import runpy
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import psutil


LOGGER = logging.getLogger(__name__)
_BATCH_RE = re.compile(
    r"Filtering template batch (\d+)-(\d+)/(\d+) segment (\d+)/(\d+)"
)


def _json_value(value: Any) -> Any:
    """Convert common JAX/NumPy scalar values to JSON values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return value.item()
    except (AttributeError, TypeError, ValueError):
        return str(value)


def _array_device(array: Any) -> Optional[str]:
    """Return a stable device label without retaining the array."""
    try:
        device = getattr(array, "device", None)
        if callable(device):
            device = device()
        if device is not None:
            return str(device)
    except Exception:
        pass
    try:
        devices = array.devices()
        return ",".join(sorted(str(device) for device in devices))
    except Exception:
        return None


def _array_pointer(array: Any) -> Optional[int]:
    """Read the unsafe pointer when the installed JAX version exposes it."""
    try:
        pointer = getattr(array, "unsafe_buffer_pointer", None)
        if callable(pointer):
            return int(pointer())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        pass
    # Older DeviceArray versions exposed the method on the buffer object.
    try:
        buffer = getattr(array, "device_buffer", None)
        pointer = getattr(buffer, "unsafe_buffer_pointer", None)
        if callable(pointer):
            return int(pointer())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        pass
    return None


def inventory_live_arrays(arrays: Iterable[Any]) -> Dict[str, Any]:
    """Serialize metadata for live arrays and group identical buffer pointers.

    Only scalar metadata is retained in the result.  In particular, the
    iterable of arrays is not stored by this function.  Pointer groups show
    possible views/aliases; equal shapes and contents are not checked.
    """
    records: List[Dict[str, Any]] = []
    pointer_to_indices: Dict[tuple, List[int]] = {}
    for index, array in enumerate(arrays):
        try:
            shape = [int(value) for value in tuple(array.shape)]
        except Exception:
            shape = None
        try:
            dtype = str(array.dtype)
        except Exception:
            dtype = None
        try:
            nbytes = int(array.nbytes)
        except (AttributeError, TypeError, ValueError):
            nbytes = None
        pointer = _array_pointer(array)
        record = {
            "index": index,
            "shape": shape,
            "dtype": dtype,
            "nbytes": nbytes,
            "device": _array_device(array),
            "unsafe_buffer_pointer": pointer,
        }
        records.append(record)
        if pointer is not None:
            key = (record["device"], pointer)
            pointer_to_indices.setdefault(key, []).append(index)

    alias_groups = [
        {
            "device": device,
            "unsafe_buffer_pointer": pointer,
            "array_indices": indices,
            "nbytes": [records[index]["nbytes"] for index in indices],
        }
        for (device, pointer), indices in pointer_to_indices.items()
        if len(indices) > 1
    ]
    return {
        "array_count": len(records),
        "arrays": records,
        "same_pointer_alias_groups": alias_groups,
    }


def device_memory_stats(jax_module: Any) -> List[Dict[str, Any]]:
    """Collect per-device allocator stats, preserving unavailable as null."""
    try:
        devices = list(jax_module.devices())
    except Exception as exc:
        return [{"device": None, "memory_stats": None, "error": str(exc)}]

    result = []
    for device in devices:
        stats = None
        error = None
        try:
            getter = getattr(device, "memory_stats", None)
            stats = getter() if callable(getter) else None
            if stats is not None:
                stats = {
                    str(key): _json_value(value)
                    for key, value in stats.items()
                }
        except Exception as exc:
            error = str(exc)
        record = {"device": str(device), "memory_stats": stats}
        if error is not None:
            record["error"] = error
        result.append(record)
    return result


class JAXMemoryProfiler:
    """Take bounded snapshots without retaining live JAX arrays."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: List[Dict[str, Any]] = []
        self.started = time.monotonic()

    def capture(self, label: str, trigger: str) -> None:
        """Capture one pprof profile and one JSON live-array inventory."""
        try:
            import jax
            import jax.profiler
        except Exception as exc:
            self.snapshots.append(
                self._failure_snapshot(label, trigger, "jax_import", exc)
            )
            return

        profile_path = self.output_dir / f"{_safe_name(label)}.pprof"
        snapshot: Dict[str, Any] = {
            "label": label,
            "trigger": trigger,
            "elapsed_sec": round(time.monotonic() - self.started, 6),
            "profile_path": str(profile_path),
            "process_rss_bytes": self._rss_bytes(),
            "device_memory_stats": None,
            "profile_error": None,
            "inventory_error": None,
        }
        arrays = None
        array = None
        block = None
        synchronized = True
        try:
            try:
                arrays = self._live_arrays(jax)
                # Make outstanding asynchronous work visible before both
                # captures.
                for array in arrays:
                    block = getattr(array, "block_until_ready", None)
                    if callable(block):
                        block()
                if arrays:
                    # Explicitly drop loop locals as well as the list below.
                    # A bound method can keep the last array live.
                    del block
                    del array
            except Exception as exc:
                synchronized = False
                snapshot["inventory_error"] = str(exc)
                snapshot["live_arrays"] = None

            # This is deliberately after synchronization so allocator stats
            # describe the same settled point as the pprof and inventory.
            snapshot["device_memory_stats"] = device_memory_stats(jax)
            try:
                jax.profiler.save_device_memory_profile(str(profile_path))
            except Exception as exc:
                snapshot["profile_error"] = str(exc)
            if synchronized:
                try:
                    snapshot["live_arrays"] = inventory_live_arrays(arrays)
                except Exception as exc:
                    snapshot["inventory_error"] = str(exc)
                    snapshot["live_arrays"] = None
        finally:
            # Do not force collection: the target owns the arrays.  The local
            # reference is released only after scalar metadata is serialized.
            array = None
            block = None
            del arrays
        self.snapshots.append(snapshot)

    @staticmethod
    def _live_arrays(jax_module: Any) -> List[Any]:
        getter = getattr(jax_module, "live_arrays", None)
        if callable(getter):
            return list(getter())
        # JAX 0.4.x may expose live buffers on devices instead.
        arrays: List[Any] = []
        for device in jax_module.devices():
            buffers = getattr(device, "live_buffers", None)
            if callable(buffers):
                arrays.extend(buffers())
        return arrays

    @staticmethod
    def _rss_bytes() -> Optional[int]:
        try:
            return int(psutil.Process().memory_info().rss)
        except Exception:
            return None

    @staticmethod
    def _failure_snapshot(label: str, trigger: str, kind: str, exc: Exception):
        return {
            "label": label,
            "trigger": trigger,
            "elapsed_sec": None,
            "profile_path": None,
            "process_rss_bytes": None,
            "device_memory_stats": None,
            "profile_error": str(exc) if kind == "profile" else None,
            "inventory_error": str(exc) if kind != "profile" else None,
            "live_arrays": None,
        }

    def write_receipt(
        self, command: Sequence[str], returncode: Optional[int] = None
    ):
        receipt = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": list(command),
            "returncode": returncode,
            "snapshots": self.snapshots,
            "limitations": [
                (
                    "live_arrays excludes internal JIT temporaries, allocator "
                    "pools, and executables"
                ),
                (
                    "same-pointer groups identify possible views or aliases; "
                    "equal content is not compared"
                ),
                "missing pointer or allocator statistics are recorded as null",
            ],
        }
        path = self.output_dir / "jax_memory_profile.json"
        with path.open("w") as handle:
            json.dump(receipt, handle, indent=2)
        return path


def _safe_name(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in "-_" else "_" for char in value
    )


class LoggingCaptureHook:
    """Inspect logging.info after forwarding it to the original function."""

    def __init__(self, profiler: JAXMemoryProfiler):
        self.profiler = profiler
        self.original = logging.info
        self.original_logger_info = logging.Logger.info
        self.seen_batches = set()
        self.next_batch_number = 0
        self.captured = set()
        self._capturing = False

    def __call__(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self.original(message, *args, **kwargs)
        self.observe(message, args)

    def logger_info(
        self, logger: logging.Logger, message: Any, *args: Any, **kwargs: Any
    ) -> None:
        """Forward a named logger call, then inspect its rendered message."""
        self.original_logger_info(logger, message, *args, **kwargs)
        self.observe(message, args)

    def observe(self, message: Any, args: Sequence[Any]) -> None:
        try:
            text = message % args if args else str(message)
            match = _BATCH_RE.search(text)
            if match and int(match.group(4)) == 1:
                batch = (int(match.group(1)), int(match.group(2)))
                if batch not in self.seen_batches:
                    self.seen_batches.add(batch)
                    self.next_batch_number += 1
                    number = self.next_batch_number
                    if number in (1, 6, 12) and number not in self.captured:
                        label = f"production_batch_{number}_segment_1"
                        self._capture(label, text)
                        self.captured.add(number)
            elif (
                "We currently have" in text
                and "postcleanup" not in self.captured
            ):
                self._capture("postcleanup", text)
                self.captured.add("postcleanup")
        except Exception as exc:
            # Profiling must never alter the target's logging or computation.
            self.original("JAX memory snapshot skipped: %s", exc)

    def _capture(self, label: str, trigger: str) -> None:
        if self._capturing:
            return
        self._capturing = True
        try:
            self.profiler.capture(label, trigger)
        finally:
            self._capturing = False


def run_target(
    executable: Path, target_args: Sequence[str], output_dir: Path
) -> int:
    """Run a target script and write the memory receipt even on failure."""
    profiler = JAXMemoryProfiler(output_dir)
    hook = LoggingCaptureHook(profiler)
    old_info = logging.info
    old_logger_info = logging.Logger.info
    old_argv = sys.argv
    returncode = 0

    def logger_info(
        logger: logging.Logger, message: Any, *args: Any, **kwargs: Any
    ) -> None:
        hook.logger_info(logger, message, *args, **kwargs)

    try:
        logging.info = hook
        logging.Logger.info = logger_info
        sys.argv = [str(executable), *target_args]
        runpy.run_path(str(executable), run_name="__main__")
    except SystemExit as exc:
        returncode = 0 if exc.code is None else (
            int(exc.code) if isinstance(exc.code, int) else 1
        )
    except BaseException:
        returncode = 1
        raise
    finally:
        logging.info = old_info
        logging.Logger.info = old_logger_info
        sys.argv = old_argv
        profiler.write_receipt([str(executable), *target_args], returncode)
    return returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        separator = values.index("--")
    except ValueError:
        parser.error("target executable and arguments must follow '--'")
    options = parser.parse_args(values[:separator])
    target = values[separator + 1:]
    if not target:
        parser.error("a target executable is required after '--'")
    return run_target(Path(target[0]), target[1:], options.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
