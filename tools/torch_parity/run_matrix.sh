#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CURRENT_SOURCE=${CURRENT_SOURCE:-$(cd "$SCRIPT_DIR/../.." && pwd)}
PYCBC_PARITY_ROOT=${PYCBC_PARITY_ROOT:-$(cd "$CURRENT_SOURCE/.." && pwd)}
ORIGINAL_SOURCE=${ORIGINAL_SOURCE:-"$PYCBC_PARITY_ROOT/original"}
ORIGINAL_PYTHON=${ORIGINAL_PYTHON:-"$PYCBC_PARITY_ROOT/venvs/original/bin/python"}
CURRENT_PYTHON=${CURRENT_PYTHON:-"$PYCBC_PARITY_ROOT/venvs/current/bin/python"}
BASELINE_COMMIT=${BASELINE_COMMIT:-$(git -C "$ORIGINAL_SOURCE" rev-parse HEAD)}
CURRENT_COMMIT=${CURRENT_COMMIT:-$(git -C "$CURRENT_SOURCE" rev-parse HEAD)}
RESULTS_ROOT=${RESULTS_ROOT:-"$PYCBC_PARITY_ROOT/results"}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RESULT_DIR="$RESULTS_ROOT/$RUN_ID"
DEPENDENCIES_FILE=${DEPENDENCIES_FILE:-"$PYCBC_PARITY_ROOT/dependencies.json"}
DEPLOYMENT_FILE=${DEPLOYMENT_FILE:-"$PYCBC_PARITY_ROOT/deployment.json"}

scrubbed_environment=()
while IFS= read -r name; do
    if [[ "$name" == PYCBC_* || "$name" == PYTHONPATH ]]; then
        scrubbed_environment+=("$name")
        unset "$name"
    fi
done < <(compgen -e | LC_ALL=C sort)

for executable in "$ORIGINAL_PYTHON" "$CURRENT_PYTHON"; do
    if [[ ! -x "$executable" ]]; then
        echo "Python is not executable: $executable" >&2
        exit 1
    fi
done
for source in "$ORIGINAL_SOURCE" "$CURRENT_SOURCE"; do
    if [[ ! -d "$source/.git" && ! -f "$source/.git" ]]; then
        echo "Source is not a Git worktree: $source" >&2
        exit 1
    fi
    source_status=$(git -C "$source" status --porcelain=v1 \
        --untracked-files=all --ignore-submodules=none)
    if [[ -n "$source_status" ]]; then
        echo "Source worktree is not clean: $source" >&2
        echo "$source_status" >&2
        exit 1
    fi
done
if [[ ! -f "$DEPENDENCIES_FILE" ]]; then
    echo "Dependency fingerprint is missing: $DEPENDENCIES_FILE" >&2
    exit 1
fi
if [[ ! -f "$DEPLOYMENT_FILE" ]]; then
    echo "Deployment manifest is missing: $DEPLOYMENT_FILE" >&2
    exit 1
fi

export ORIGINAL_SOURCE CURRENT_SOURCE ORIGINAL_PYTHON CURRENT_PYTHON
export DEPENDENCIES_FILE DEPLOYMENT_FILE
verify_deployment() {
    "$CURRENT_PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess

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


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def is_excluded_cache(relative):
    return (
        any(part in EXCLUDED_CACHE_DIRECTORIES for part in relative.parts)
        or relative.name in EXCLUDED_CACHE_FILENAMES
        or relative.suffix.lower() in EXCLUDED_CACHE_SUFFIXES
    )


def ignored_artifacts(source):
    output = subprocess.check_output([
        "git", "-C", str(source), "ls-files", "--others", "--ignored",
        "--exclude-standard", "-z",
    ])
    artifacts = []
    for encoded in output.split(b"\0"):
        if not encoded:
            continue
        relative = Path(os.fsdecode(encoded))
        if is_excluded_cache(relative):
            continue
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise SystemExit(
                f"included ignored artifact is not a regular file: {path}"
            )
        artifacts.append({
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    return sorted(artifacts, key=lambda item: item["path"])


def import_identity(python):
    script = r'''
import json
from pathlib import Path
import subprocess
import sys
import pycbc
source = Path(pycbc.__file__).resolve().parent.parent
revision = subprocess.check_output(
    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
).strip()
print(json.dumps({"executable": sys.executable, "source": str(source), "revision": revision}))
'''
    output = subprocess.check_output(
        [str(python), "-I", "-c", script],
        text=True,
    )
    return json.loads(output.strip().splitlines()[-1])


manifest_path = Path(os.environ["DEPLOYMENT_FILE"]).resolve()
dependency_path = Path(os.environ["DEPENDENCIES_FILE"]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
unsigned = dict(manifest)
recorded_seal = unsigned.pop("content_sha256", None)
actual_seal = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
errors = []
if manifest.get("schema_version") != 2:
    errors.append("deployment manifest schema_version is not 2")
if recorded_seal != actual_seal:
    errors.append("deployment manifest content seal is invalid")

dependencies = manifest.get("dependencies", {})
if dependencies.get("path") != str(dependency_path):
    errors.append("dependency fingerprint path differs from deployment manifest")
try:
    dependency_sha256 = sha256_file(dependency_path)
except OSError as exc:
    errors.append(f"cannot hash dependency fingerprint: {exc}")
else:
    if dependencies.get("sha256") != dependency_sha256:
        errors.append("dependency fingerprint SHA256 differs from deployment manifest")

specifications = {
    "original": (
        Path(os.environ["ORIGINAL_SOURCE"]).resolve(),
        Path(os.environ["ORIGINAL_PYTHON"]).absolute(),
    ),
    "current": (
        Path(os.environ["CURRENT_SOURCE"]).resolve(),
        Path(os.environ["CURRENT_PYTHON"]).absolute(),
    ),
}
sources = manifest.get("sources", {})
if manifest.get("ignored_artifact_policy") != IGNORED_ARTIFACT_POLICY:
    errors.append("ignored-artifact policy differs from the verifier")
recorded_artifacts = manifest.get("ignored_artifacts", {})
actual_artifacts_by_source = {}
for label, (source, python) in specifications.items():
    evidence = sources.get(label, {})
    if evidence.get("path") != str(source):
        errors.append(f"{label} source path differs from deployment manifest")
    if evidence.get("python") != str(python):
        errors.append(f"{label} Python path differs from deployment manifest")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if evidence.get("revision") != revision:
        errors.append(f"{label} source revision differs from deployment manifest")
    identity = import_identity(python)
    if (
        identity.get("source") != evidence.get("import_source")
        or identity.get("revision") != evidence.get("import_revision")
        or identity.get("source") != str(source)
        or identity.get("revision") != revision
    ):
        errors.append(f"{label} import identity differs from deployment manifest")
    actual_artifacts = ignored_artifacts(source)
    actual_artifacts_by_source[label] = actual_artifacts
    if recorded_artifacts.get(label) != actual_artifacts:
        errors.append(f"{label} ignored build-artifact inventory/hash mismatch")

recorded_count = manifest.get("ignored_artifact_count")
actual_count = sum(len(items) for items in actual_artifacts_by_source.values())
if recorded_count != actual_count:
    errors.append("ignored artifact count differs from deployment manifest")
if (
    manifest.get("build_artifacts") != recorded_artifacts
    or manifest.get("build_artifact_count") != recorded_count
):
    errors.append("legacy build-artifact aliases differ from canonical inventory")
if errors:
    raise SystemExit("deployment verification failed:\n- " + "\n- ".join(errors))
print(f"deployment_manifest={manifest_path}")
print(f"deployment_content_sha256={recorded_seal}")
print(f"deployment_file_sha256={sha256_file(manifest_path)}")
print(f"dependency_fingerprint_sha256={dependencies['sha256']}")
PY
}

write_launch_evidence() {
    local output_path=$1
    "$CURRENT_PYTHON" - "$output_path" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone


def canonical_bytes(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packages(python):
    script = r'''
import importlib.metadata
import json
packages = {}
for dist in importlib.metadata.distributions():
    name = (dist.metadata.get("Name") or "").lower()
    if name and name != "pycbc":
        packages[name] = dist.version
print(json.dumps(packages, sort_keys=True))
'''
    output = subprocess.check_output([str(python), "-c", script], text=True)
    return json.loads(output)


def runtime_probe():
    runtime = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "executable": sys.executable,
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
                    "index": index,
                    "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory_bytes": properties.total_memory,
                })
        runtime["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": devices,
        }
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        runtime["nvidia_smi"] = {
            "available": False,
            "query": None,
            "error": "nvidia-smi executable not found",
        }
    else:
        result = subprocess.run(
            [
                nvidia_smi_path,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        runtime["nvidia_smi"] = {
            "available": result.returncode == 0,
            "query": result.stdout.strip() if result.returncode == 0 else None,
            "error": result.stderr.strip() if result.returncode else None,
        }
    return runtime


output_path = Path(sys.argv[1]).absolute()
dependency_path = Path(os.environ["DEPENDENCIES_FILE"]).resolve()
deployment_path = Path(os.environ["DEPLOYMENT_FILE"]).resolve()
dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
fingerprints = {
    "original": packages(Path(os.environ["ORIGINAL_PYTHON"])),
    "current": packages(Path(os.environ["CURRENT_PYTHON"])),
}
runtime = runtime_probe()
errors = []
setup_packages = dependency.get("packages")
for label, fingerprint in fingerprints.items():
    if fingerprint != setup_packages:
        errors.append(f"{label} installed package fingerprint differs from setup")
if fingerprints["original"] != fingerprints["current"]:
    errors.append("original/current installed package fingerprints differ")
if runtime != dependency.get("runtime"):
    errors.append("current Python/Torch/CUDA/device/driver runtime differs from setup")
payload = {
    "schema_version": 1,
    "kind": "parity_matrix_launch",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "dependencies": {
        "path": str(dependency_path),
        "sha256": sha256_file(dependency_path),
    },
    "deployment": {
        "path": str(deployment_path),
        "sha256": sha256_file(deployment_path),
        "content_sha256": deployment.get("content_sha256"),
    },
    "package_fingerprints": fingerprints,
    "runtime": runtime,
    "valid": not errors,
    "errors": errors,
}
payload["content_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
with output_path.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
output_path.chmod(0o444)
print(f"launch_evidence={output_path}")
print(f"launch_evidence_content_sha256={payload['content_sha256']}")
if errors:
    raise SystemExit("launch runtime verification failed:\n- " + "\n- ".join(errors))
PY
}

if [[ -e "$RESULT_DIR" ]]; then
    echo "Result directory already exists: $RESULT_DIR" >&2
    exit 1
fi
mkdir -p "$RESULT_DIR"
exec > >(tee "$RESULT_DIR/matrix.log") 2>&1

verify_deployment
write_launch_evidence "$RESULT_DIR/launch.json"

echo "host=$(hostname)"
echo "result_dir=$RESULT_DIR"
echo "baseline_commit=$BASELINE_COMMIT"
echo "current_commit=$CURRENT_COMMIT"
echo "dependencies_file=$DEPENDENCIES_FILE"
echo "deployment_file=$DEPLOYMENT_FILE"
echo "scrubbed_environment=${scrubbed_environment[*]:-none}"
cp "$DEPENDENCIES_FILE" "$RESULT_DIR/dependencies.json"
cp "$DEPLOYMENT_FILE" "$RESULT_DIR/deployment.json"

clean_env=(
    env
    -u PYTHONPATH
    -u PYCBC_SCHEME
)

"${clean_env[@]}" PYCBC_SCHEME=cpu PYCBC_TORCH_NATIVE_PORTS=0 \
    "$ORIGINAL_PYTHON" "$SCRIPT_DIR/generate.py" \
    --label A-original-cpu --output-dir "$RESULT_DIR" \
    --scheme cpu --expected-revision "$BASELINE_COMMIT"

"${clean_env[@]}" PYCBC_SCHEME=cpu PYCBC_TORCH_NATIVE_PORTS=0 \
    "$CURRENT_PYTHON" "$SCRIPT_DIR/generate.py" \
    --label B-current-cpu --output-dir "$RESULT_DIR" \
    --scheme cpu --expected-revision "$CURRENT_COMMIT"

"${clean_env[@]}" PYCBC_SCHEME=cpu PYCBC_TORCH_NATIVE_PORTS=1 \
    "$CURRENT_PYTHON" "$SCRIPT_DIR/generate.py" \
    --label C-current-torch-cpu --output-dir "$RESULT_DIR" \
    --scheme torch --device cpu --block-lalsimulation \
    --expected-revision "$CURRENT_COMMIT"

cuda_available=$(
    "${clean_env[@]}" "$CURRENT_PYTHON" -c \
        'import torch; print(int(torch.cuda.is_available()))'
)
if [[ "$cuda_available" == "1" ]]; then
    "${clean_env[@]}" PYCBC_SCHEME=cpu PYCBC_TORCH_NATIVE_PORTS=1 \
        "$CURRENT_PYTHON" "$SCRIPT_DIR/generate.py" \
        --label D-current-torch-cuda --output-dir "$RESULT_DIR" \
        --scheme torch --device cuda --block-lalsimulation \
        --expected-revision "$CURRENT_COMMIT"
else
    echo "Torch CUDA unavailable; skipping matrix cell D"
fi

status=0
compare() {
    local reference=$1
    local candidate=$2
    local profile=$3
    local report=$4
    "$CURRENT_PYTHON" "$SCRIPT_DIR/compare.py" \
        "$RESULT_DIR/$reference" "$RESULT_DIR/$candidate" \
        --profile "$profile" --report "$RESULT_DIR/$report" || status=1
}

compare A-original-cpu B-current-cpu cpu-regression compare-A-B.json
compare B-current-cpu C-current-torch-cpu torch compare-B-C.json
if [[ "$cuda_available" == "1" ]]; then
    compare B-current-cpu D-current-torch-cuda torch compare-B-D.json
    compare C-current-torch-cpu D-current-torch-cuda torch compare-C-D.json
fi

if [[ "$status" == "0" ]]; then
    echo "matrix_result=PASS"
else
    echo "matrix_result=FAIL"
fi
echo "artifacts=$RESULT_DIR"
exit "$status"
