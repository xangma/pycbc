# TorchWave identity helper relocation

This artifact records a narrow source-helper relocation only. TorchWave commit
`890541e` adds `torchwave.provenance.taylorf2_source_identity()`, preserving
the exact original five-file digest inputs, ordering, and byte algorithm. The
relocation adds no model capability, physics, optimization, or compatibility
promise. The tests' “legacy digest” wording means algorithm byte compatibility.

PyCBC remains the sole cache owner and directly requires the new module; there
is no old-version fallback. If optional TorchWave is absent, the original
PyCBC behavior remains `None`.

The source bases are PyCBC `d0aa34d64` and TorchWave `84ef9b3`, with the exact
four-file overlays used for this restack. PyCBC PRs 6–11 are now integrated
atomically; final PR11 code is `d9687892605bdf96b6c5f906a7155e6dd6460363`.
Existing tracked and user-untracked TorchWave changes remain untouched; only
the two added relocation files are integrated.

For reproduction, copy the committed code and existing qualified compiled
extensions into `/home/xangma/pycbc-remediation-20260912/`, then run
`bash /home/xangma/pycbc-remediation-20260912/identity-relocation/run-tests.sh`;
the script changes into each checkout itself. The final run passed 30 PyCBC
CPU/CUDA tests (6.70s) and 9 TorchWave tests (2.11s); targeted four-file F401
checks passed with exit 0. Initial collection required copying 11 compiled extensions from the
qualified checkout symlinks to their existing target bytes. This does not
mutate the environment or install packages.

No performance remeasurement was performed; earlier receipts retain pinned
scope.

Test result: **39 total passed (30 PyCBC CPU/CUDA, 9 TorchWave); F401 clean.**
