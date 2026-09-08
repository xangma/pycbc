# Torch documentation audit — 8 September 2026

All 14 original Torch guide pages were reviewed against source, tests and their cited evidence. Twelve remain, with distinct user or reviewer purposes; two development-history pages and 13 intermediate plot/manifest assets were removed from the active documentation. The waveform guide's Torch sections, navigation, referenced assets and Sphinx configuration were also checked. This audit does not cover every page of the historical PyCBC manual.

The review is recorded at documentation commit `123e1fb3ef1b338cada636e71c3e9c7987002402`. Its runtime is byte-identical to the previously published `9d4e4f6905d9f109d559284326d62173a854264f`. File age alone was not used to decide whether documentation was stale.

## Measurement conclusion

There is no completed matched timing campaign comparing the existing-code baseline `40e94792b3edf59f39b18b65102b28a4f74433a7` with this proposed runtime. The old executable headline used a development revision for all three backends; it did not measure unchanged PyCBC against the current proposal. Earlier original-baseline comparisons also reported numerical differences and 1988 versus 1991 triggers. Those differences must be assessed before claiming equivalence to the baseline.

The main review documentation now defines four arms: unchanged CPU, proposed normal CPU, proposed Torch CPU and proposed Torch CUDA. It preserves source/input provenance, identical scientific arguments, original-CPU geometry selection, resource accounting, fresh-process timing, repeated samples and qualification requirements. Historical timings and optimization experiments were removed from the active comparison. Existing native regression and executable correctness evidence remains in the testing guide, with its source and backend-reference limits explicit. It supplies neither missing timing nor unchanged-baseline equivalence.

## Page decisions

| Original page | Decision and purpose |
| --- | --- |
| `torch.rst` | Retain as the single entry point. Correct installation and provide an executable CPU example and a single navigation hierarchy. |
| `torch_runtime.rst` | Retain for scheme/device/thread selection, storage and transfer contracts. Clarify CPU thread restoration, accelerator precision limits and host boundaries. |
| `torch_filtering.rst` | Retain as the filtering support map. Narrow residency claims to implemented kernels and document the host work in clustering and filter setup. |
| `torch_search.rst` | Retain for offline/live integration, stream controls and graph ownership. Correct defaults, constructor precedence, explicit graph capture and unsupported fork handling. |
| `torch_workflows.rst` | Retain, reviewed without changes. Explains orchestration and the limits of workflow-level Torch selection; it serves a different purpose from numerical API guides. |
| `torch_parity.rst` | Retain, reviewed without changes. Defines waveform/sequence reference selection and numerical contracts. |
| `torch_testing.rst` | Retain for test selectors, CI coverage and reproducible qualification. Clarify standalone versus pytest selectors and add current integrated correctness evidence formerly in the follow-up page. |
| `torch_optimizations.rst` | Retain as the current control reference. Correct promoted MKL sizes, private workspace semantics, flag precedence, graph capture requirements and shared MKL descriptor lifecycle. |
| `torch_performance.rst` | Rewrite as existing-code versus proposed-code comparison guidance. State the missing current comparison and baseline-output gaps; remove development timing headlines. |
| `torch_benchmark_protocol.rst` | Retain as measurement methodology: qualification, resource controls, full-process clocks, convergence and capacity accounting. Remove experiment history. |
| `torch_reference_campaign.rst` | Retain as the concrete workload/reproduction specification. Preserve bank/frame hashes, scientific settings and original-CPU geometry selection; distinguish archived revisions from the current proposal. |
| `torch_batch_numerics.rst` | Retain for the separate prepared live-filter API and independent normalized complex-SNR oracle. Preserve float64 normalization, oracle precision and absolute error gates; remove historical throughput/results. |
| `torch_profile_attribution.rst` | Remove from active documentation. Development-stage profile interpretation remains in the preserved source/evidence archive. |
| `torch_followups.rst` | Remove from active documentation. Move current correctness evidence to testing and current controls to optimizations; preserve intermediate experiments in the archive. |

The waveform guide now has standalone batch imports, accurate sequence-reference language and evaluated support intervals with an excluded end bin. The top-level index no longer duplicates the Torch hierarchy. The documentation configuration discovers real FFT backends before mocking native library loading, so executable examples cannot select a fictitious MKL backend. An unused theme import was removed; the configured theme still builds.

## Verification

- A fresh Sphinx HTML build with `-E -a -W --keep-going` passed with zero warnings. It covered all 12 retained Torch pages and seven actual waveform/plugin/installation dependencies, using the repository's extensions and theme and executing plot/command directives. Generated index/API context supplied only the bounded build context. Unrelated manual pages, include generators and external intersphinx inventories were excluded.
- Quickstart, CPU thread restoration and standalone waveform batch examples passed on Torch 2.9.1. The matching built editable checkout had 370 byte-identical tracked runtime files. These local checks do not newly qualify CUDA.
- Existing runtime/selector/graph checks: 37 passed, 13 skipped for unavailable CUDA. Existing evidence/plot tests: 115 passed. Counts describe separate selections, not a unique-test total.
- Frozen optimization packages and native production evidence passed independent offline verification. The integrated evidence records 283 native test passes and 52 unavailable-MPS skips, three backend science outputs, five comparisons, 1920 graph replays and two negative controls. Each backend's byte comparisons use its own frozen reference.
- `flake8 --select F401 docs/conf.py` and `git diff --check` passed. Independent source/evidence reviews found no remaining blocker in the final measurement presentation.
- The four updated PR heads preserve the `pycbc`, `bin`, `test`, `tools` and `.github` trees byte for byte relative to their previously published heads. Every restack difference is under `docs`; the existing runtime validation therefore remains applicable.

The pre-commit build log records the then-HEAD runtime revision. `build-source-receipt.json` binds each of its 19 actual input pages to the final committed documentation bytes. Initial build/setup failures and subsequent corrections remain in the receipts. No remote benchmark or native scientific acquisition was run for this documentation audit.

## Preserved evidence

`runtime-review.md` and `evidence-review.md` contain detailed claim/source checks. Their initial page-retention recommendations precede the final consolidation; this audit and the final source are authoritative for the delivered layout. `development-documents-before-consolidation/` preserves that reviewed intermediate text. `original-documents/` and `removed-assets/` preserve the exact previously published documents/assets at `9d4e4f6905d9f109d559284326d62173a854264f`. Hashes and the final restack mapping are included. Existing sealed experiment archives were not modified.

Run `python3 -B -I verify.py` in this directory to check the archive inventory and hashes. This is an integrity check of the audit evidence; it does not rerun measurements or science validation.
