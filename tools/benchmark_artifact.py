"""Small, dependency-free helpers for reproducible benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
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

SCHEMA_VERSION = 2


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
    return {
        "unit": unit,
        "count": len(samples),
        "samples": samples,
        "median": float(statistics.median(samples)),
        "mean": float(statistics.mean(samples)),
        "p25": percentile(samples, 0.25),
        "p75": percentile(samples, 0.75),
        "minimum": min(samples),
        "maximum": max(samples),
        "stddev": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
        "median_ci95": {"low": ci_low, "high": ci_high},
        "bootstrap": {"method": "percentile", "resamples": bootstrap_resamples},
    }


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

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
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
    if payload.get("schema_version") != SCHEMA_VERSION:
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
