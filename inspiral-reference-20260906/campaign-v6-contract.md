# Frozen v6 campaign contract

Only the orchestration is new: `campaign-v6.py`, `launch-campaign-v6.py`, and
this contract. No v5 file is rewritten, no source is edited, and preparing this
harness does not launch a workload. Runtime root is the directory containing
the scripts on `len`, using the same Python environment as the v5 receipts.

## Source and reference reuse

Root prepares `source-v6.json` after reviewing and committing the candidate:

```json
{
  "schema_version": 1,
  "source": "<absolute campaign root>/source-v6",
  "commit": "<reviewed 40-character commit>",
  "parent": "837f38d493420043e45fb1ad210a0ccf68bacbaa",
  "changed_paths": [
    "pycbc/fft/torchfft.py",
    "pycbc/waveform/decompress_torch.py",
    "test/test_torch_decompress_cpu.py",
    "test/test_torch_large_ifft.py"
  ],
  "changed_files_sha256": {"<each changed path>": "<sha256>"},
  "native_modules_sha256": {"<each v5 native module path>": "<same v5 sha256>"},
  "normal_cpu_path_audit": {
    "status": "pass",
    "source_commit": "<same reviewed commit>",
    "parent_source_commit": "837f38d493420043e45fb1ad210a0ccf68bacbaa",
    "dispatch": {
      "scheme_prefix": "cpu",
      "fft": "pycbc.fft.mkl",
      "decompression": "pycbc.waveform.decompress_cpu"
    },
    "rationale": "<root's reviewed CPU dispatch and side-effect assessment>"
  }
}
```

Additional setup provenance is retained. If `input_sha256` is present, it must
be nonempty, equal `input_sha256_after`, and every absolute path must still
match. The harness checks the exact direct parent, exact four-path git diff,
changed-file content hashes, clean source, and all eleven native module hashes
in both v5 and v6. It also checks unchanged committed and working-file content
for CPU dispatch/backend paths and parses the CPU prefix, FFT backend mapping,
and waveform interpolation decorator from their ASTs. Root's audit assesses
behavior and side effects of the reviewed Torch module edits; filename and AST
checks alone are not a proof of arbitrary Python behavior.

`config.json`, the compressed bank, original frame, v5 source manifest, v5
tuning decision and existing run/profiling helpers are pinned to their frozen
hashes. All 38 tuning-decision input files are checked. The selected geometry
is exactly 512 seconds, with 112-second start and 16-second end padding. Only
that prior **normal CPU tuning choice** is reused; all nine matched v6 timing
runs, including normal CPU, execute the new source.

## Two stages and prerequisite receipts

`--stage qualifications` requires `unit-tests-v6.json` with the v5 unit receipt
schema: `state=complete`, `passed=true`, `returncode=0`, `finished_utc`,
`source_info` and `source_after` both exactly `{commit: <v6>, status: ""}`,
unchanged nonempty absolute `input_sha256`/`input_sha256_after`, and
`log_sha256` binding `unit-tests-v6.log`. Its input map must bind
`source-v6.json`. It runs, in order:

1. `qual-selected6-cpu-l512`, `cpu:1`.
2. `qual-selected6-torch-cpu-l512`, `torch:cpu:1`.
3. `qual-selected6-torch-cuda-l512`, `torch:cuda:0`.

Root then performs scientific validation using these new qualifier captures
and emits `scientific-validation-v6.json`:

```json
{
  "schema_version": 1,
  "state": "complete",
  "passed": true,
  "returncode": 0,
  "finished_utc": "<UTC timestamp>",
  "source_info": {"commit": "<v6>", "status": ""},
  "source_after": {"commit": "<v6>", "status": ""},
  "checks": {
    "waveform_reference": true,
    "compressed_bank_backend_parity": true,
    "boundary_injections": true,
    "qualification_trigger_parity": true
  },
  "evidence": {
    "waveform_reference": {"path": "<absolute receipt>", "sha256": "<sha256>"},
    "compressed_bank_backend_parity": {"path": "<absolute receipt>", "sha256": "<sha256>"},
    "boundary_injections": {"path": "<absolute receipt>", "sha256": "<sha256>"},
    "qualification_trigger_parity": {"path": "<absolute receipt>", "sha256": "<sha256>"}
  },
  "input_sha256": {"<absolute input path>": "<sha256>"},
  "input_sha256_after": {"<same absolute input paths>": "<same sha256>"}
}
```

The input map must include `source-v6.json`, `unit-tests-v6.json`, the completed
`campaign-v6-qualifications.status.json`, all three `qualification.json` and
`triggers.hdf` files, and the four evidence receipt paths. Root is responsible
for validating each scientific receipt's numerical budgets and exact expected
coverage before asserting these checks. If unchanged normal CPU science is
reused, its evidence remains explicitly attributed to v5; compressed waveform
CPU/Torch CPU/CUDA parity must exercise v6 at the declared geometries.

`--stage measurements` verifies that integration receipt and the complete
immutable qualification-stage output manifest before running:

* Nine unprofiled timings `matched-optimized6-<backend>-l512-r<1..3>`; backend
  order rotates CPU/Torch CPU/CUDA, Torch CPU/CUDA/CPU, CUDA/CPU/Torch CPU.
* Six `profile-optimized6-<backend>-l512-<cprofile|perf>` runs, in backend order
  CPU, Torch CPU, CUDA, then one CUDA `torchprofile` run.
* Native `perf report` exports and `profiles-v6.json` using the original
  summarizer. Profile times are excluded from timing summaries.

The measurement stage reports orchestration completion, not a publication or
final scientific acceptance. Root still compares all final v6 trigger outputs
against the normal CPU reference and reviews performance before accepting v6.

## Inspection, launch and recovery

Both scripts require `--stage <qualifications|measurements>` and
`--source-manifest-sha256 <reviewed source-v6.json hash>`.
`campaign-v6.py --describe` prints the fixed plan without reading source or
writing files. Adding `--check` to a fully specified campaign command performs
all stage preflight checks without writing or executing workloads.

`launch-campaign-v6.py` performs preflight, starts one detached process group,
and writes an exclusive launch receipt with host, cwd, command, PID, log path,
expected next check in 60 seconds, and `kill -TERM -- -<PID>` stop command.
Each stage uses fresh `campaign-v6-<stage>.*` outputs. Existing run directories,
stage outputs or launch receipts are rejected. There is no automatic resume
and no overwrite flag; failed or interrupted evidence is retained for review.
Source, inputs and completed outputs are checked before and after each child.
The status records active command and child PID and is updated atomically.
