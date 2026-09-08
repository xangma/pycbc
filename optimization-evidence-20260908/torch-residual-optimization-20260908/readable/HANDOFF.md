# Residual optimization handoff

Descriptor reuse is a qualified prototype with measured full-process gains: standard CPU median 66.849562 → 65.370059 s (2.213184% reduction) and CUDA 21.976242 → 20.291608 s (7.665707%). All four paired runs improved for each family. Cached CUDA's median paired speedup over contemporaneous cached standard CPU is 3.222772×. Owner and independent peer replay passed, and all remote jobs have ended.

The separate NumPy-copy CPU candidate was rejected after passing its contracts and precision gates because none of its three workers showed a whole-call benefit. Torch CPU parity remains unmet. No new CPU gap is inferred against descriptor-cached standard CPU. Further CPU investigation is separate from this sealed evidence snapshot.

| Read first | Purpose |
|---|---|
| [Descriptor timing result](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/DESCRIPTOR-REUSE-TIMING-v1.md) | Design, full-wall results, raw evidence, comparisons and scope limits |
| [All timing samples](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/descriptor-reuse-timing-v1-samples.csv) | Six metrics for all 16 timed runs |
| [Owner timing replay](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/descriptor-reuse-timing-v1-summary.json) | Full samples, medians, ranges, paired calculations, lifecycle and comparisons |
| [Independent raw replay](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/peer-reviews/fast-reuse-results-v1-review.json) | Every archive member, source/runtime provenance, native/science gates and closure |
| [Independent summary cross-check](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/peer-reviews/fast-reuse-summary-v1-crosscheck.json) | Exact agreement with all owner metrics and receipts |
| [Residual synthesis](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/peer-reviews/RESIDUAL-ANALYSIS-v3.md) | Completed CPU diagnostics and descriptor results in context |
| [CPU-copy rejection](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/CPU-COPY-RESULT-v1.md) | Native contracts, unchanged numerical budgets and unfavorable measured pairs |
| [Diagnostic reuse qualification](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/DESCRIPTOR-REUSE-QUALIFICATION-v1.md) | Earlier instrumented helper's native and science qualification |
| [Frozen timing protocol](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/descriptor-reuse-timing-v1/PROTOCOL.md) | Four-cell schedule, controls, helper scope and pre-timing gates |

The final timing archive contains 317 regular files and is 29,368,320 bytes. SHA256: `52c50829795096da3683c6b678deb5ddfc69e6f8a90d1ab8b106db84ebee18f3`; results-manifest SHA256: `21dd1a3f2112aba4664bea0b619af0c31fd7c947c6f7d1443658a1645531bb7f`. Four scientific qualifications and 16 timing runs plus a fresh native process completed; all 51 comparators passed, including 38 exact comparisons of all 18 H1 science datasets. Controller group 2936779 exited, and the independent auditor reacquired the lock and verified final pins before sealing.

The local source `/private/tmp/pycbc-torch-fft-optimization-20260908` and qualified remote source remain at clean `ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f`. All reuse implementations and harnesses are experiment artifacts. Local fast contracts and harness tests passed; source was unchanged, so no new project source tests were needed for this artifact-only phase. No low-level C/CUDA or `pycbc/lib` changes, production integration, PR, push or publication occurred.

The final input manifest is `807406093bee08fa621eeb4c327a8c1547f66c85224db9c3d33f11e5156f244b`. The older `c95322f1` timing input snapshot is preserved under `descriptor-reuse-timing-v1-superseded-c95322f1`; only documentation about creation-count semantics changed before the executed freeze. It was not launched. Earlier diagnostics, rejected prototypes and reviews remain preserved as evidence and do not supersede the final result.

[SHA256SUMS](/Users/xangma/repos/pycbc/artifacts/torch-residual-optimization-20260908/SHA256SUMS) seals all regular files in this snapshot except itself and interpreter caches. It includes the result archives, acquired files, scripts, reviews and this handoff. Four repetitions on this single finite workload do not establish statistical significance or general workload behavior. Native equality was checked inside the pinned gate; native arrays themselves are not archived. Production adoption needs its own implementation review and qualification.
