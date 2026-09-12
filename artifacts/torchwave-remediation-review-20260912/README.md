# TorchWave remediation review evidence — 2026-09-12

These are source-review artifacts and small CPU reproductions. They do not constitute implementation, a GPU benchmark, or production qualification.

- `input-brief.md`: original supplied brief, preserved for comparison. Its implementation instructions were not executed; the user clarified “Review and revise the brief.”
- `review-provenance.json`: both source revisions, runtime/module paths, reviewed source hashes, actual commands and recorded test result summaries.
- `starting-user-changes.patch` and `starting-status.txt`: pre-existing PyCBC tracked changes and status. The status includes numerous pre-existing unrelated artifacts; those are not claimed as outputs of this review.
- `starting-untracked-source/`: copies of the four relevant pre-existing untracked integration tests/tools. Together with the pinned PyCBC commit and tracked patch, these preserve the reviewed integration source, not every unrelated local artifact.
- `probe_bank_contract.py` and `bank-contract.json`: actual bank option/dtype/default-eligibility reproduction.
- `probe_numerics.py` and `numerics.json`: actual NumPy/Torch CPU screening and reduced-basis reproductions.
- `checksums.json`: SHA-256 manifest for this evidence directory, excluding the manifest itself.

The lead's commands ran from `/Users/xangma/repos/pycbc` with `python` resolving to `/Users/xangma/miniconda3/bin/python`, Torch 2.13.0 and CUDA unavailable. The separate numerical subagent used the `pycbc313` environment, Torch 2.9.1 and CUDA unavailable. Commands and test summaries identify each environment; these runs are not interchangeable performance samples.

## Reproduce the saved probes

Use an environment importing the recorded PyCBC and TorchWave checkouts, including the reviewed dirty integration. From the PyCBC repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python artifacts/torchwave-remediation-review-20260912/probe_bank_contract.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python artifacts/torchwave-remediation-review-20260912/probe_numerics.py
```

The first script creates and removes a temporary HDF bank. Both scripts print measurements rather than asserting them as approved behavior. Preserve the saved JSON when running against future revisions.

Expected reviewed-state observations:

- Changing reference PN phase order changes its waveform, but the TorchWave adapter's output remains identical.
- Requested complex128 output is returned as complex64; the default uncompressed bank is eligible without explicit enable.
- Screening rejects the constructed candidate although its final NewSNR is approximately 7.7827.
- The rank-one plan reports original norms `[4, 0]` instead of `[4, 1]`.

The PN-order relative-L2 number compares reference order 0 against reference order 7. It is not an adapter/reference accuracy measurement. Screening and reduced-basis failures concern experimental code paths.

## Existing tests actually run

```sh
python -m pytest -q test/test_torchwave_integration.py test/test_torchwave_live_and_inspiral.py -k 'not throughput'
# 9 passed, 1 deselected in 6.13s — lead

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/xangma/miniconda3/envs/pycbc313/bin/python -m pytest -q test/test_gpu_search_screening.py test/test_gpu_search_reduced_basis.py test/test_gpu_search_vetoes.py -k 'not large_sample'
# 24 passed, 1 deselected, 2 warnings in 2.70s — numerical subagent
```

These are transcribed summaries of executed tool results, not newly captured raw pytest logs. The lead independently reran the numerical reproducer and saved its JSON. No GPU performance or full executable real-frame campaign was run.

Final deliverables: [review](/Users/xangma/repos/pycbc/docs/torchwave_remediation_review.md) and [revised future implementation brief](/Users/xangma/repos/pycbc/docs/torchwave_remediation_brief.md).

