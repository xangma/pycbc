#!/usr/bin/env python3
"""Prepare and verify the provenance consumed by ``run_matrix.sh``.

Preparation observes an already installed pair of worktrees; it never installs
packages, builds extensions, or repairs a mismatch. Hashes detect subsequent
changes, not the trustworthiness of the machine that produced them.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys


EXCLUDED_CACHE_DIRECTORIES = {
    ".hypothesis", ".mypy_cache", ".nox", ".pytest_cache", ".ruff_cache",
    ".tox", "__pycache__",
}
EXCLUDED_CACHE_FILENAMES = {".coverage"}
EXCLUDED_CACHE_SUFFIXES = {".pyc", ".pyo"}
IGNORED_ARTIFACT_POLICY = {
    "scope": "all_git_ignored_regular_files",
    "excluded_directory_names": sorted(EXCLUDED_CACHE_DIRECTORIES),
    "excluded_filenames": sorted(EXCLUDED_CACHE_FILENAMES),
    "excluded_suffixes": sorted(EXCLUDED_CACHE_SUFFIXES),
}
RUNTIME_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OMP_DYNAMIC",
    "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n").encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sealed(value):
    value = dict(value)
    digest = hashlib.sha256(canonical_bytes(value)).hexdigest()
    value["content_sha256"] = digest
    return value


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def clean_environment():
    return {name: value for name, value in os.environ.items()
            if name != "PYTHONPATH" and not name.startswith("PYCBC_")}


def git(source, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(source), *arguments], text=True,
        env=clean_environment(),
    ).strip()


def ignored_artifacts(source):
    output = subprocess.check_output([
        "git", "-C", str(source), "ls-files", "--others", "--ignored",
        "--exclude-standard", "-z",
    ], env=clean_environment())
    artifacts = []
    for encoded in output.split(b"\0"):
        if not encoded:
            continue
        relative = Path(os.fsdecode(encoded))
        if (any(part in EXCLUDED_CACHE_DIRECTORIES for part in relative.parts)
                or relative.name in EXCLUDED_CACHE_FILENAMES
                or relative.suffix.lower() in EXCLUDED_CACHE_SUFFIXES):
            continue
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"ignored artifact is not a regular file: {path}")
        artifacts.append({
            "path": relative.as_posix(), "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    return sorted(artifacts, key=lambda item: item["path"])


def python_json(python, *arguments):
    output = subprocess.check_output(
        [str(python), "-I", *arguments], text=True, env=clean_environment(),
    )
    return json.loads(output.strip().splitlines()[-1])


def import_identity(python):
    return python_json(python, "-c", """
import json
from pathlib import Path
import subprocess
import pycbc
source = Path(pycbc.__file__).resolve().parent.parent
revision = subprocess.check_output(
    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
print(json.dumps({"source": str(source), "revision": revision}))
""")


def installed_packages():
    """Fingerprint selected distributions, including venv shadowing rules."""
    names = {
        re.sub(r"[-_.]+", "-", dist.metadata.get("Name") or "").lower()
        for dist in metadata.distributions()
    }
    # Enumeration includes shadowed system-site distributions. Resolve each
    # name through the normal lookup instead of allowing the last copy seen
    # to overwrite the active distribution's version.
    return {name: metadata.version(name) for name in sorted(names)
            if name and name != "pycbc"}


def packages(python):
    return python_json(python, str(Path(__file__).resolve()), "_packages")


def runtime_probe():
    """Run inside the selected current interpreter."""
    runtime = {
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(), "python": sys.version,
        "executable": sys.executable,
        "environment": {
            key: os.environ.get(key) for key in RUNTIME_ENVIRONMENT
        },
    }
    try:
        import torch
    except ImportError:
        runtime["torch"] = None
    else:
        devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append({
                    "index": index, "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory_bytes": properties.total_memory,
                })
        runtime["torch"] = {
            "version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": devices,
        }
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        runtime["nvidia_smi"] = {
            "available": False, "query": None,
            "error": "nvidia-smi executable not found",
        }
    else:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True,
        )
        runtime["nvidia_smi"] = {
            "available": result.returncode == 0,
            "query": result.stdout.strip() if result.returncode == 0 else None,
            "error": result.stderr.strip() if result.returncode else None,
        }
    return runtime


def current_runtime(python):
    return python_json(python, str(Path(__file__).resolve()), "_runtime")


def source_snapshot(specifications):
    sources, artifacts = {}, {}
    for label, (source, python) in specifications.items():
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError(f"Python is not executable: {python}")
        status = git(
            source, "status", "--porcelain=v1", "--untracked-files=all",
            "--ignore-submodules=none",
        )
        if status:
            raise ValueError(f"{label} source worktree is not clean: {status}")
        revision = git(source, "rev-parse", "HEAD")
        identity = import_identity(python)
        if identity != {"source": str(source), "revision": revision}:
            raise ValueError(f"{label} import identity differs from worktree")
        sources[label] = {
            "path": str(source), "python": str(python), "revision": revision,
            "import_source": identity["source"],
            "import_revision": identity["revision"],
        }
        artifacts[label] = ignored_artifacts(source)
    return sources, artifacts


def dependency_snapshot(specifications):
    fingerprints = {label: packages(python)
                    for label, (_, python) in specifications.items()}
    runtime = current_runtime(specifications["current"][1])
    return fingerprints, runtime


def prepare(specifications, dependency_path, deployment_path):
    for path in (dependency_path, deployment_path):
        if path.exists():
            raise ValueError(f"refusing to replace existing evidence: {path}")
        for source, _ in specifications.values():
            if path.is_relative_to(source):
                raise ValueError("manifest outputs must be outside worktrees")
    if dependency_path == deployment_path:
        raise ValueError("dependency and deployment paths must differ")
    sources, artifacts = source_snapshot(specifications)
    fingerprints, runtime = dependency_snapshot(specifications)
    if fingerprints["original"] != fingerprints["current"]:
        raise ValueError(
            "original/current installed package fingerprints differ"
        )
    dependency = {
        "schema_version": 1, "packages": fingerprints["current"],
        "runtime": runtime,
    }
    write_json(dependency_path, dependency)
    count = sum(len(items) for items in artifacts.values())
    deployment = sealed({
        "schema_version": 2,
        "dependencies": {"path": str(dependency_path),
                         "sha256": sha256_file(dependency_path)},
        "sources": sources, "ignored_artifact_policy": IGNORED_ARTIFACT_POLICY,
        "ignored_artifacts": artifacts, "ignored_artifact_count": count,
        "build_artifacts": artifacts, "build_artifact_count": count,
    })
    write_json(deployment_path, deployment)
    return deployment


def verify(specifications, dependency_path, deployment_path, launch_path=None):
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    unsigned = dict(deployment)
    seal = unsigned.pop("content_sha256", None)
    if seal != hashlib.sha256(canonical_bytes(unsigned)).hexdigest():
        raise ValueError("deployment manifest content seal is invalid")
    if deployment.get("schema_version") != 2:
        raise ValueError("deployment manifest schema_version is not 2")
    dependency_identity = {"path": str(dependency_path),
                           "sha256": sha256_file(dependency_path)}
    if deployment.get("dependencies") != dependency_identity:
        raise ValueError("dependency fingerprint path or SHA256 differs")
    sources, artifacts = source_snapshot(specifications)
    if deployment.get("sources") != sources:
        raise ValueError(
            "source path, interpreter, revision or import changed"
        )
    if deployment.get("ignored_artifact_policy") != IGNORED_ARTIFACT_POLICY:
        raise ValueError("ignored-artifact policy differs from the verifier")
    count = sum(len(items) for items in artifacts.values())
    if any((deployment.get("ignored_artifacts") != artifacts,
            deployment.get("ignored_artifact_count") != count,
            deployment.get("build_artifacts") != artifacts,
            deployment.get("build_artifact_count") != count)):
        raise ValueError("ignored build-artifact inventory/hash mismatch")
    dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
    fingerprints, runtime = dependency_snapshot(specifications)
    errors = []
    if dependency.get("schema_version") != 1:
        errors.append("dependency fingerprint schema_version is not 1")
    for label, fingerprint in fingerprints.items():
        if fingerprint != dependency.get("packages"):
            errors.append(f"{label} installed package fingerprint changed")
    if runtime != dependency.get("runtime"):
        errors.append(
            "current Python/Torch/CUDA/device/driver/environment changed"
        )
    launch = sealed({
        "schema_version": 1, "kind": "parity_matrix_launch",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dependencies": dependency_identity,
        "deployment": {"path": str(deployment_path),
                       "sha256": sha256_file(deployment_path),
                       "content_sha256": seal},
        "package_fingerprints": fingerprints, "runtime": runtime,
        "valid": not errors, "errors": errors,
    })
    if launch_path is not None:
        write_json(launch_path, launch)
    if errors:
        raise ValueError("launch verification failed: " + "; ".join(errors))
    return launch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "verify", "_runtime", "_packages")
    )
    for label in ("original", "current"):
        parser.add_argument(f"--{label}-source", type=Path)
        parser.add_argument(f"--{label}-python", type=Path)
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--launch", type=Path,
                        help="write a new launch receipt during verification")
    args = parser.parse_args(argv)
    if args.command == "_runtime":
        print(json.dumps(runtime_probe(), sort_keys=True))
        return 0
    if args.command == "_packages":
        print(json.dumps(installed_packages(), sort_keys=True))
        return 0
    required = ("original_source", "current_source", "original_python",
                "current_python", "dependencies", "deployment")
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    specifications = {
        label: (getattr(args, f"{label}_source").resolve(),
                getattr(args, f"{label}_python").absolute())
        for label in ("original", "current")
    }
    if (specifications["original"][0] == specifications["current"][0]
            or specifications["original"][1] == specifications["current"][1]):
        parser.error("use separate source worktrees and Python environments")
    try:
        if args.command == "prepare":
            if args.launch is not None:
                parser.error("--launch applies only to verify")
            prepare(specifications, args.dependencies.resolve(),
                    args.deployment.resolve())
        else:
            verify(specifications, args.dependencies.resolve(),
                   args.deployment.resolve(), args.launch)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"{args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(f"{args.command}=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
