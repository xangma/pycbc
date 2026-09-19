#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CURRENT_SOURCE=${CURRENT_SOURCE:-$(cd "$SCRIPT_DIR/../.." && pwd)}
CURRENT_PYTHON=${CURRENT_PYTHON:-python}
OUTPUT_DIR=${OUTPUT_DIR:-"$CURRENT_SOURCE/artifacts/jax_parity"}

mkdir -p "$OUTPUT_DIR"

echo "=== Generating CPU baseline artifacts ==="
"$CURRENT_PYTHON" "$SCRIPT_DIR/generate.py" \
    --label baseline \
    --scheme cpu \
    --output-dir "$OUTPUT_DIR"

echo "=== Generating JAX candidate artifacts ==="
"$CURRENT_PYTHON" "$SCRIPT_DIR/generate.py" \
    --label candidate \
    --scheme jax \
    --device cpu \
    --output-dir "$OUTPUT_DIR"

echo "=== Comparing CPU baseline vs JAX candidate ==="
"$CURRENT_PYTHON" "$SCRIPT_DIR/compare.py" \
    "$OUTPUT_DIR/baseline.json" \
    "$OUTPUT_DIR/candidate.json" \
    --profile jax \
    --policy "$SCRIPT_DIR/policy.json" \
    --report "$OUTPUT_DIR/parity_report.json"

echo "=== Parity check completed successfully ==="
