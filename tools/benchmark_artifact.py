"""Small, dependency-free helpers for reproducible benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = frozenset({2, SCHEMA_VERSION})


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value using a stable serialization."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_content_sha256(payload: Mapping) -> str:
    """Hash an artifact excluding its self-referential content-seal field."""

    material = dict(payload)
    material.pop("content_sha256", None)
    return canonical_sha256(material)


def seal_artifact(payload: Mapping) -> dict:
    """Return an artifact carrying a seal over all other serialized content."""

    sealed = dict(payload)
    sealed.pop("content_sha256", None)
    sealed["content_sha256"] = artifact_content_sha256(sealed)
    return sealed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sample")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be between zero and one")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_median_ci(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a deterministic percentile-bootstrap interval for the median."""

    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if len(samples) == 1:
        return samples[0], samples[0]

    rng = random.Random(seed)
    count = len(samples)
    medians = [
        statistics.median(samples[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    tail = (1.0 - confidence) / 2.0
    point = float(statistics.median(samples))
    lower = percentile(medians, tail)
    upper = percentile(medians, 1.0 - tail)
    return (
        max(min(samples), min(point, lower)),
        min(max(samples), max(point, upper)),
    )


def sample_summary(
    values: Iterable[float],
    *,
    unit: str,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = 2000,
) -> dict:
    """Retain samples and report robust descriptive statistics."""

    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("cannot summarize an empty sample")
    ci_low, ci_high = bootstrap_median_ci(
        samples, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    median = float(statistics.median(samples))
    p95 = percentile(samples, 0.95)
    p99 = percentile(samples, 0.99)
    return {
        "unit": unit,
        "count": len(samples),
        "samples": samples,
        "median": median,
        # Keep the historical ``median`` spelling while making the latency
        # dimensions consumed by plots and reports explicit.
        "p50": median,
        "p95": p95,
        "p99": p99,
        "percentiles": {"p50": median, "p95": p95, "p99": p99},
        "mean": float(statistics.mean(samples)),
        "p25": percentile(samples, 0.25),
        "p75": percentile(samples, 0.75),
        "minimum": min(samples),
        "maximum": max(samples),
        "stddev": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
        "median_ci95": {"low": ci_low, "high": ci_high},
        "bootstrap": {
            "method": "percentile",
            "confidence": 0.95,
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
        },
    }


def summary_statistic(summary: Mapping, statistic: str = "p50") -> float:
    """Read a statistic from current or legacy sample-summary JSON.

    Schema-v2 artifacts used ``median`` and did not always materialize tail
    percentiles.  Raw samples remain the source of truth when a requested
    percentile is absent.
    """

    if not isinstance(summary, Mapping):
        raise TypeError("sample summary must be a mapping")
    normalized = statistic.lower()
    if normalized == "median":
        normalized = "p50"
    if normalized not in {"p50", "p95", "p99"}:
        raise ValueError(f"unsupported summary statistic {statistic!r}")

    candidates = [normalized]
    if normalized == "p50":
        candidates.append("median")
    percentiles = summary.get("percentiles")
    for key in candidates:
        if key in summary:
            value = float(summary[key])
            if math.isfinite(value):
                return value
        if isinstance(percentiles, Mapping) and key in percentiles:
            value = float(percentiles[key])
            if math.isfinite(value):
                return value

    samples = summary.get("samples")
    if isinstance(samples, (list, tuple)) and samples:
        fractions = {"p50": 0.50, "p95": 0.95, "p99": 0.99}
        return percentile(samples, fractions[normalized])
    raise ValueError(f"sample summary has no {normalized} measurement")


def summary_median_ci95(
    summary: Mapping,
    *,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = 2000,
) -> tuple[float, float]:
    """Read or reconstruct a median 95% interval from artifact JSON.

    Older summaries without an interval remain plottable.  If they retained
    raw samples, reconstruct the deterministic bootstrap interval; otherwise
    return a zero-width interval around the only available point estimate.
    """

    median = summary_statistic(summary, "p50")
    interval = summary.get("median_ci95") if isinstance(summary, Mapping) else None
    if isinstance(interval, Mapping) and {"low", "high"} <= interval.keys():
        low = float(interval["low"])
        high = float(interval["high"])
        if all(math.isfinite(value) for value in (low, high)) and low <= high:
            return min(low, median), max(high, median)

    samples = summary.get("samples") if isinstance(summary, Mapping) else None
    if isinstance(samples, (list, tuple)) and samples:
        return bootstrap_median_ci(
            samples,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        )
    return median, median


def _git_output(repo_root: Path, *args: str) -> str | None:
    process = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        return None
    return process.stdout.rstrip("\n")


def _git_bytes(repo_root: Path, *args: str) -> bytes | None:
    process = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
    )
    return process.stdout if process.returncode == 0 else None


def source_identity(repo_root: Path, source_files: Iterable[Path] = ()) -> dict:
    """Describe the exact revision, dirty state, and benchmark source files."""

    repo_root = repo_root.resolve()
    status = _git_output(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    revision = _git_output(repo_root, "rev-parse", "HEAD")
    tracked_diff = _git_bytes(repo_root, "diff", "--binary", "HEAD")
    untracked_output = _git_bytes(
        repo_root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    untracked_files = {}
    if untracked_output is not None:
        for raw_name in untracked_output.split(b"\0"):
            if not raw_name:
                continue
            name = os.fsdecode(raw_name)
            candidate = repo_root / name
            if candidate.is_file():
                untracked_files[name] = file_sha256(candidate)
    files = {}
    for path in source_files:
        resolved = path.resolve()
        try:
            name = str(resolved.relative_to(repo_root))
        except ValueError:
            name = str(resolved)
        files[name] = file_sha256(resolved)
    return {
        "repository": str(repo_root),
        "revision": revision,
        "dirty": status is None or bool(status),
        "status_sha256": canonical_sha256(status) if status is not None else None,
        "tracked_diff_sha256": (
            hashlib.sha256(tracked_diff).hexdigest()
            if tracked_diff is not None
            else None
        ),
        "untracked_files_sha256": dict(sorted(untracked_files.items())),
        "files_sha256": dict(sorted(files.items())),
    }


def runtime_metadata() -> dict:
    """Collect runtime and accelerator identity without requiring Torch."""

    memory_bytes = None
    try:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, TypeError, ValueError):
        pass

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_logical_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        metadata["torch"] = torch.__version__
        metadata["torch_cuda"] = torch.version.cuda
        metadata["cuda_available"] = cuda_available
        metadata["cuda_devices"] = (
            [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ]
            if cuda_available
            else []
        )
    except ImportError:
        metadata.update(
            {
                "torch": None,
                "torch_cuda": None,
                "cuda_available": False,
                "cuda_devices": [],
            }
        )
    metadata["hardware"] = {
        "hostname": metadata["hostname"],
        "platform": metadata["platform"],
        "machine": metadata["machine"],
        "processor": metadata["processor"],
        "cpu_logical_count": metadata["cpu_logical_count"],
        "memory_bytes": metadata["memory_bytes"],
        "accelerators": metadata["cuda_devices"],
    }
    metadata["software"] = {
        "python": {
            "version": metadata["python"],
            "implementation": platform.python_implementation(),
            "executable": metadata["python_executable"],
        },
        "torch": {
            "version": metadata["torch"],
            "cuda_runtime": metadata["torch_cuda"],
        },
    }
    return metadata


def compatibility_record(settings: Mapping[str, object], metadata: Mapping) -> dict:
    """Build the identity used to reject incomparable incremental runs."""

    material = {
        "settings": dict(settings),
        "source": metadata.get("source"),
        "runtime": {
            name: metadata.get(name)
            for name in (
                "platform",
                "machine",
                "python",
                "python_executable",
                "torch",
                "torch_cuda",
                "cuda_devices",
            )
        },
    }
    return {"material": material, "sha256": canonical_sha256(material)}


def load_mergeable_artifact(
    path: Path,
    *,
    artifact_type: str,
    compatibility_sha256: str,
) -> dict:
    """Load an artifact only when its schema and benchmark identity match."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read benchmark artifact {path}: {exc}") from exc
    observed_schema = payload.get("schema_version")
    if observed_schema not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{path} uses a legacy or unknown schema; pass --no-merge to replace it"
        )
    if payload.get("artifact_type") != artifact_type:
        raise ValueError(
            f"{path} contains {payload.get('artifact_type')!r}, not {artifact_type!r}"
        )
    if payload.get("content_sha256") != artifact_content_sha256(payload):
        raise ValueError(
            f"{path} has a missing or invalid benchmark content seal; "
            "use a new output path or pass --no-merge"
        )
    observed = payload.get("compatibility", {}).get("sha256")
    if observed != compatibility_sha256:
        raise ValueError(
            f"{path} was produced by an incompatible source/runtime/configuration; "
            "use a new output path or pass --no-merge"
        )
    # The additive v3 summary fields can be populated by subsequent samples;
    # upgrading in memory lets v2 campaigns continue without discarding their
    # raw measurements.  Validate the old seal before changing this field.
    payload["schema_version"] = SCHEMA_VERSION
    if "schema" in payload:
        payload["schema"] = SCHEMA_VERSION
    return payload


def atomic_write_json(path: Path, payload: Mapping) -> None:
    """Write JSON atomically in the destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
