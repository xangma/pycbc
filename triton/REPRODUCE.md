# Reproduce the TaylorF2 Triton comparison

Use Linux with NVIDIA CUDA and Triton. The exact measured package inventory is in `finalization.json`; primary versions were Python 3.11.9, Torch 2.13.0+cu130 and Triton 3.7.1. Follow the repository's installation instructions, including LALSuite and an editable PyCBC installation. Compile the native extensions at the selected source revision. This supplement does not alter low-level native extensions.

```sh
git clone https://github.com/xangma/pycbc.git pycbc-triton
cd pycbc-triton
git checkout --detach 6829dc9bcba07ca7d3da44de7589cc4e9fb84da5
python -m pip install -e .
python setup.py build_ext --inplace -j 8
git status --porcelain
```

The status output must be empty; keep evidence and compiler caches outside the source checkout. Install dependencies before starting timed work. The exact measured revision remains reachable on `codex/torch-taylorf2-triton-20260906`. The final assembled PR revision `d44ed6e2273488b0fcd52819f3a604ae38b8a79c` has an identical tree.

Set absolute paths for your checkout, interpreter and this artifact directory. Use a new output directory, as the harness preserves existing attempts.

```sh
TF_SOURCE=/absolute/path/pycbc-triton
TF_PYTHON=/absolute/path/venv/bin/python
TF_EVIDENCE=/absolute/path/evidence/triton
TF_OUTPUT=/absolute/path/new-comparison
taskset -c 8-11 "$TF_PYTHON" -B "$TF_EVIDENCE/harness/run.py" \
  --root "$TF_SOURCE" \
  --expected-sha 6829dc9bcba07ca7d3da44de7589cc4e9fb84da5 \
  --python "$TF_PYTHON" --out "$TF_OUTPUT"
```

The controller runs 60 fresh worker processes sequentially, alternating route order: two routes, five batch sizes, two frequency grids and three replicates. Every worker sets one host thread, a fresh Triton cache and the appropriate route flag. It records a separate cold call, two warmups, five synchronized groups lasting at least 50 ms each, and subsequent untimed full-output parity and actual-dispatch checks. [Detailed contracts](harness/README.md) describe the reference chain and timing exclusions.

Run the focused scientific and fallback tests separately from timing:

```sh
cd "$TF_SOURCE"
PYCBC_TEST_SCHEME=torch:cuda PYCBC_TAYLORF2_TRITON=1 \
  "$TF_PYTHON" -m pytest -q -ra -p no:cacheprovider \
  test/waveform/test_taylorf2_batch.py test/waveform/test_taylorf2_torch.py
```

Regenerate the figures with Python, NumPy and Matplotlib. When rerunning the experiment on another day, set `--run-date` to that day's UTC date. To render the archived data, use `full-v3-clean` as the input and `2026-09-06` as the date.

```sh
python "$TF_EVIDENCE/report/render.py" \
  --input-dir "$TF_OUTPUT" --output-dir /absolute/path/new-report \
  --harness-dir "$TF_EVIDENCE/harness" \
  --expected-sha 6829dc9bcba07ca7d3da44de7589cc4e9fb84da5 \
  --run-date 2026-09-06
```

`qualify-restack.py` records the host-specific commands used to validate each final PR prefix. Its saved paths and publication bundle are specific to the original host; use the commits and test commands in `restack-validation.json` for independent prefix validation.
