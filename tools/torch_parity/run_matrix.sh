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

verify_deployment() {
    "$CURRENT_PYTHON" "$SCRIPT_DIR/manifest.py" verify \
        --original-source "$ORIGINAL_SOURCE" \
        --current-source "$CURRENT_SOURCE" \
        --original-python "$ORIGINAL_PYTHON" \
        --current-python "$CURRENT_PYTHON" \
        --dependencies "$DEPENDENCIES_FILE" \
        --deployment "$DEPLOYMENT_FILE" \
        --launch "$RESULT_DIR/launch.json"
}

if [[ -e "$RESULT_DIR" ]]; then
    echo "Result directory already exists: $RESULT_DIR" >&2
    exit 1
fi
mkdir -p "$RESULT_DIR"
exec > >(tee "$RESULT_DIR/matrix.log") 2>&1

verify_deployment

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
    --scheme torch --device cpu \
    --expected-revision "$CURRENT_COMMIT"

cuda_available=$(
    "${clean_env[@]}" "$CURRENT_PYTHON" -c \
        'import torch; print(int(torch.cuda.is_available()))'
)
if [[ "$cuda_available" == "1" ]]; then
    "${clean_env[@]}" PYCBC_SCHEME=cpu PYCBC_TORCH_NATIVE_PORTS=1 \
        "$CURRENT_PYTHON" "$SCRIPT_DIR/generate.py" \
        --label D-current-torch-cuda --output-dir "$RESULT_DIR" \
        --scheme torch --device cuda \
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
    if [[ "$cuda_available" == "1" ]]; then
        echo "matrix_result=PASS"
    else
        echo "matrix_result=PASS_CPU_ONLY"
    fi
else
    echo "matrix_result=FAIL"
fi
echo "artifacts=$RESULT_DIR"
exit "$status"
