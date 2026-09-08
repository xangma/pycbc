# Torch compatibility with the original CPU — 2026-09-08

The fixes are in the existing Torch stack. Original CPU reference `40e94792b3edf59f39b18b65102b28a4f74433a7` is unchanged. The executable qualification measured `88878b1c38c952e63002b812058a0c7316123f70`; subsequent documentation changes and restacked optional leaves are mapped in final-publication when available.

## Scientific result

**PASS:** original CPU, candidate normal CPU, Torch CPU and Torch CUDA each produce **1988 triggers** on the same 384-template, five-segment, 1904-second H1 workload. All five comparisons pass the unchanged trigger and full-PSD gates. No missing or extra trigger identities and no numerical violations remain. Original versus candidate CPU retains byte-exact identity for all 18 scientific H1 datasets and full PSD arrays. Conditioned-strain identity is verified through full-array hashes and metadata; raw conditioned strain was not archived. Segment geometry is identical.

Only the four recorded elapsed-time-derived H1/search fields (`filter_rate_per_core`, `run_time`, `setup_time_fraction`, `templates_per_core`) are outside scientific byte equality. Raw comparator verdicts retain the separately documented source/executable provenance differences. Scientific comparison normalizes only those two verified provenance fields, with no numerical tolerance changes.

[Qualification summary](remote/summary.json), [raw and adjusted comparisons](remote/comparisons/), [independent HDF/NPY and receipt verification](independent-verification/verification.json), [source preservation](source-preservation.json).

## Implementation and scope

The owning PRs provide a guarded Torch CPU context (#5), original PSD arithmetic for compatible Torch strain (#7), NumPy's serial float32 template-power scan and original compiled point chi-square evaluation (#8), and original CPU strain FFT arithmetic for compatible Torch search (#9). Ordinary CPU behavior is unchanged. The compatibility paths cover ordinary contiguous single-precision CPU/CUDA tensors; their predicates preserve fallback handling for unsupported tensor semantics. CUDA compatibility uses synchronous host transfers. There is **no new performance sample, speedup claim or sustained-capacity claim**.

The optional general FFT (#16) and native CPU optimization (#17) leaves remain separate from this executable qualification. Their own focused tests are recorded separately. PR20 remains withdrawn.

## Validation

The measured main suite passed **535 tests plus 8 subtests, with 156 skipped**. Focused owning-PR suites and the optional FFT leaf passed; see [test receipts](tests/status.json). Earlier no-Torch and CPU preservation checks, completed on the same unchanged CPU paths, are retained. Linux CUDA regressions passed **178 tests, 4 skipped**. Qlty reports only the existing B904 in `pycbc/frame/frame.py:629`, whose file is unchanged from the previously published main. Outputs are in [linux-checks](linux-checks/). CI-scoped unused-import checks and changed-file Ruff formatting passed. Full flake8 output retains existing repository findings; it is not represented as a green gate. Qlty findings are reported from the archived output.

## Provenance

The archive retains HDF triggers, complete PSD arrays, exact source bundle, source/native/input/module hashes, commands, dependencies and runtime observations. Each route ran on core 8 with native thread pools fixed to one and Torch intra/inter-op counts of one. Source, input and harness pins were checked throughout; the downloaded files were hash-verified. Independent verification recomputed comparisons, individual numerical metrics, source Git hashes, event identity sets, PSD comparisons and CPU scientific equality without rerunning analysis.

The initial attempt stopped before scientific analysis because its pinned thread-pool observer was omitted from the upload. Its failure is retained in `attempt-initial`; the completed campaign includes that exact pinned observer. Earlier original-CPU restoration failures elsewhere in this evidence branch remain historical records and are superseded by this result.
