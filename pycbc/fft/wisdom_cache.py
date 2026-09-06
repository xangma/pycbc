# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Automatic FFTW wisdom caching for qualified Torch CPU plans.

The cache is intentionally narrower than FFTW's general wisdom interface.
Only the precision- and performance-qualified sequential search IFFT may
request an automatic plan.  Explicit command-line wisdom options continue to
use :mod:`pycbc.fft.fftw` directly and disable this cache.
"""

import ctypes
import hashlib
import json
import logging
import os
import platform
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on Windows
    fcntl = None


logger = logging.getLogger("pycbc.fft.wisdom_cache")

_CACHE_SCHEMA = 1
_ROUTE_VERSION = 1
_SEARCH_IFFT_SIZE = 131072
_CPUINFO_KEYS = (
    "vendor_id",
    "cpu family",
    "model",
    "model name",
    "stepping",
    "microcode",
    "flags",
)


@dataclass
class _CacheEntry:
    """State for one exact cache fingerprint."""

    path: Path
    ready: bool = False
    dirty: bool = False
    lock_file: object | None = None


@dataclass
class _CacheConfig:
    """Process-local automatic cache configuration."""

    enabled: bool = False
    cache_dir: Path | None = None
    entries: dict[str, _CacheEntry] = field(default_factory=dict)


_config = _CacheConfig()


def _has_explicit_wisdom_options(opt):
    """Return whether the caller selected any manual wisdom operation."""
    return bool(
        getattr(opt, "fftw_import_system_wisdom", False)
        or getattr(opt, "fftw_input_float_wisdom_file", None) is not None
        or getattr(opt, "fftw_input_double_wisdom_file", None) is not None
        or getattr(opt, "fftw_output_float_wisdom_file", None) is not None
        or getattr(opt, "fftw_output_double_wisdom_file", None) is not None
    )


def _default_cache_dir():
    """Return the per-user cache location, following the XDG convention."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache and os.path.isabs(xdg_cache):
        root = Path(xdg_cache)
    else:
        root = Path.home() / ".cache"
    return root / "pycbc" / "fftw"


def configure_from_cli(opt):
    """Reset and configure the automatic cache from parsed CLI options."""
    global _config

    for entry in _config.entries.values():
        _release_lock(entry)
    enabled = bool(getattr(opt, "fftw_wisdom_cache", False))
    enabled = enabled and not _has_explicit_wisdom_options(opt)
    cache_dir = getattr(opt, "fftw_wisdom_cache_dir", None)
    try:
        if cache_dir is None:
            cache_dir = _default_cache_dir()
        else:
            cache_dir = Path(cache_dir).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        logger.warning("FFTW wisdom cache is unavailable: %s", exc)
        enabled = False
        cache_dir = None
    _config = _CacheConfig(enabled=enabled, cache_dir=cache_dir)


def _hash_file(path):
    """Hash a binary or source file without retaining its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path):
    """Return a content-bound identity for a program or shared library."""
    candidate = Path(path).expanduser()
    try:
        candidate = candidate.resolve(strict=True)
        stat = candidate.stat()
        return {
            "path": str(candidate),
            "size": stat.st_size,
            "sha256": _hash_file(candidate),
        }
    except (OSError, RuntimeError):
        return {"path": str(path), "unresolved": True}


def _loaded_library_path(library, fragment):
    """Resolve a ctypes library name, including Linux loader aliases."""
    name = str(getattr(library, "_name", ""))
    if name and Path(name).is_file():
        return name
    requested_name = Path(name).name
    maps = Path("/proc/self/maps")
    try:
        exact_candidates = set()
        fallback_candidates = set()
        for line in maps.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields and fields[-1].startswith("/"):
                path = fields[-1]
                candidate_name = Path(path).name
                if requested_name and (
                    candidate_name == requested_name
                    or candidate_name.startswith(f"{requested_name}.")
                    or requested_name in candidate_name
                ):
                    exact_candidates.add(path)
                elif candidate_name == fragment or candidate_name.startswith(
                    f"{fragment}."
                ):
                    fallback_candidates.add(path)
        if exact_candidates:
            return sorted(exact_candidates)[0]
        if fallback_candidates:
            return sorted(fallback_candidates)[0]
    except OSError:
        pass
    return name


def _fftw_version(library):
    """Read FFTW's exported version string without adding a C binding."""
    try:
        first = ctypes.c_char.in_dll(library, "fftwf_version")
        raw = ctypes.string_at(ctypes.addressof(first), 256)
        return raw.split(b"\0", 1)[0].decode("ascii", "replace")
    except (AttributeError, TypeError, ValueError, OSError):
        return "unknown"


def _cpu_identity():
    """Return stable CPU characteristics relevant to FFTW planning."""
    identity = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    cpuinfo = Path("/proc/cpuinfo")
    try:
        first_cpu = cpuinfo.read_text(encoding="utf-8").split("\n\n", 1)[0]
        values = {}
        for line in first_cpu.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in _CPUINFO_KEYS:
                values[key.strip()] = value.strip()
        identity["linux_cpuinfo"] = values
    except OSError:
        pass
    return identity


def _fingerprint_payload(fftw, *, aligned, nthreads):
    """Build the complete compatibility identity for cached wisdom."""
    import numpy
    import torch

    import pycbc

    source_dir = Path(__file__).resolve().parent
    float_library = _loaded_library_path(fftw.float_lib, "libfftw3f")
    threaded_library = getattr(fftw, "_float_threaded_lib", None)
    if threaded_library is None:
        threaded_identity = None
    else:
        threaded_path = _loaded_library_path(threaded_library, "libfftw3f")
        threaded_identity = _file_identity(threaded_path)

    return {
        "schema": _CACHE_SCHEMA,
        "route_version": _ROUTE_VERSION,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "cpu": _cpu_identity(),
        },
        "program": {
            "python": _file_identity(sys.executable),
            "pycbc_version": pycbc.__version__,
            "pycbc_git_hash": pycbc.git_hash,
            "numpy_version": numpy.__version__,
            "torch_version": torch.__version__,
            "torchfft_source": _file_identity(source_dir / "torchfft.py"),
            "fftw_wrapper_source": _file_identity(source_dir / "fftw.py"),
        },
        "fftw": {
            "version": _fftw_version(fftw.float_lib),
            "float_library": _file_identity(float_library),
            "thread_backend": getattr(fftw, "_fftw_threaded_lib", None),
            "thread_library": threaded_identity,
        },
        "plan": {
            "precision": "float32",
            "kind": "complex-to-complex",
            "direction": "backward",
            "length": _SEARCH_IFFT_SIZE,
            "batch": 1,
            "input_stride": 1,
            "output_stride": 1,
            "input_distance": _SEARCH_IFFT_SIZE,
            "output_distance": _SEARCH_IFFT_SIZE,
            "in_place": False,
            "aligned": aligned,
            "pycbc_alignment": pycbc.PYCBC_ALIGNMENT,
            "nthreads": nthreads,
            "planner_rigor": "measure",
        },
    }


def _cache_entry(fftw, *, aligned, nthreads):
    """Return the cache entry for an exact plan fingerprint."""
    payload = _fingerprint_payload(fftw, aligned=aligned, nthreads=nthreads)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    entry = _config.entries.get(fingerprint)
    if entry is None:
        filename = f"torch-b1-search-ifft-{fingerprint}.wisdom"
        entry = _CacheEntry(_config.cache_dir / filename)
        _config.entries[fingerprint] = entry
    return entry


def _release_lock(entry):
    """Release this process's single-writer lock, if any."""
    lock_file = entry.lock_file
    entry.lock_file = None
    if lock_file is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except (OSError, ValueError) as exc:
        logger.warning("Could not release FFTW wisdom cache lock: %s", exc)
    try:
        lock_file.close()
    except OSError as exc:
        logger.warning("Could not close FFTW wisdom cache lock: %s", exc)


def _acquire_lock(entry):
    """Serialize a cold measurement for this exact fingerprint."""
    if fcntl is None or entry.lock_file is not None:
        return
    lock_path = entry.path.with_suffix(f"{entry.path.suffix}.lock")
    lock_file = lock_path.open("a+b")
    try:
        os.fchmod(lock_file.fileno(), 0o600)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    except BaseException:
        lock_file.close()
        raise
    entry.lock_file = lock_file


def _import_entry(fftw, entry, *, report_failure):
    """Try importing one complete atomic cache file."""
    if not entry.path.is_file():
        return False
    try:
        fftw.import_single_wisdom_from_filename(str(entry.path))
    except (OSError, RuntimeError) as exc:
        if report_failure:
            logger.warning(
                "Ignoring unusable cached FFTW wisdom %s: %s",
                entry.path,
                exc,
            )
        return False
    entry.ready = True
    logger.info("Imported cached FFTW wisdom from %s", entry.path)
    return True


def prepare_plan(
    fftw,
    *,
    size,
    forward,
    direct,
    aligned,
    nthreads,
    batch,
    requested_measure_level,
):
    """Import a compatible plan or request one bounded measurement.

    Returns the effective measure level and an opaque entry token.  The caller
    must pass the token to :func:`record_plan` only after direct-plan creation
    succeeds.
    """
    eligible = (
        _config.enabled
        and size == _SEARCH_IFFT_SIZE
        and not forward
        and direct
        and aligned
        and nthreads == 1
        and batch == 1
        and requested_measure_level in (0, 1)
    )
    if not eligible:
        return requested_measure_level, None

    try:
        _config.cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not fftw._fftw_threaded_set:
            fftw.set_threads_backend()
        entry = _cache_entry(fftw, aligned=aligned, nthreads=nthreads)
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        logger.warning("FFTW wisdom cache is unavailable: %s", exc)
        return requested_measure_level, None

    if entry.ready:
        return requested_measure_level, entry
    # A warm reader never waits for the lock: atomic rename guarantees that it
    # sees either the old complete file or the new complete file.  Only cold
    # writers serialize, then recheck after acquiring the lock in case another
    # process published the plan while this process was waiting.
    if _import_entry(fftw, entry, report_failure=False):
        return requested_measure_level, entry
    try:
        _acquire_lock(entry)
        if _import_entry(fftw, entry, report_failure=True):
            _release_lock(entry)
            return requested_measure_level, entry
    except (OSError, RuntimeError) as exc:
        logger.warning("FFTW wisdom cache locking is unavailable: %s", exc)
        _release_lock(entry)
    return 1, entry


def record_plan(fftw, entry, measure_level):
    """Publish a successfully measured cache-backed direct plan."""
    if entry is None:
        return
    try:
        if not entry.ready and measure_level == 1:
            entry.ready = True
            entry.dirty = True
            _export_entry(fftw, entry)
    finally:
        _release_lock(entry)


def cancel_plan(entry):
    """Release a cold-writer lock after direct-plan construction failed."""
    if entry is not None:
        _release_lock(entry)


def has_pending_export():
    """Return whether a newly measured automatic plan needs persistence."""
    return any(entry.dirty for entry in _config.entries.values())


def _atomic_export(fftw, path):
    """Export single-precision wisdom through a same-directory rename."""
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        fftw.export_single_wisdom_to_filename(temporary)
        os.chmod(temporary, 0o600)
        with open(temporary, "rb") as fp:
            os.fsync(fp.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _export_entry(fftw, entry):
    """Atomically export one entry while preserving application progress."""
    try:
        _atomic_export(fftw, entry.path)
    except (OSError, RuntimeError) as exc:
        logger.warning(
            "Could not update automatic FFTW wisdom cache %s: %s",
            entry.path,
            exc,
        )
        return
    entry.dirty = False
    logger.info("Updated automatic FFTW wisdom cache %s", entry.path)


def export_pending(fftw):
    """Persist all newly measured cache entries without failing the job."""
    for entry in _config.entries.values():
        if not entry.dirty:
            continue
        _export_entry(fftw, entry)
