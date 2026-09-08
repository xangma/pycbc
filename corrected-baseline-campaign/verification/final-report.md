# Final corrected-baseline verification

**Complete evidence: PASS_WITH_LIMITATIONS. Scientific equivalence: FAIL. Equal-output speedup eligibility across the campaign: false.** This final report supersedes the provisional qualification handoff in README.md.

All four qualifications and 16 counterbalanced timings validate, including runtime/source/thread/lock receipts, three-clock consistency, and all 16 timing-versus-own-qualification comparisons. All five cross-arm trigger comparisons pass, with 1,991 triggers in every qualification and timing output. The acquired aggregate is `summary.json`; its samples, medians, rates, and status reproduce exactly.

| Arm | Median wall seconds | Observed min–max seconds | Template-seconds / wall second |
|---|---:|---:|---:|
| Corrected CPU | 68.238 | 68.174–68.487 | 10,714.4 |
| Proposed CPU | 65.472 | 65.370–65.672 | 11,167.1 |
| Torch CPU | 102.446 | 102.207–102.728 | 7,136.8 |
| Torch CUDA | 20.324 | 20.298–20.353 | 35,973.9 |

Each rate is `384 templates × 1904 valid seconds / median wall seconds` (731,136 template-seconds). Each arm has four timed repetitions. The timing boundary is fresh checked-process launch through exit after HDF output, runtime receipts, and native-module verification; qualification runs and profiling are excluded. These are descriptive measurements of a finite workload on a shared host.

Corrected CPU versus proposed CPU passes full scientific checks; its median ratio is **1.0423×**, eligible within this measured workload. Every CPU-versus-Torch pair fails the frozen full-PSD budget (`rtol=1e-4`, `atol=0`): 2,375 failing bins per segment for Torch CPU and 3,105 for Torch CUDA. Their wall-time ratios are descriptive only. Exact used PSD bins `[15360:1048576]`, conditioned-strain receipt digests, and geometry do not erase the full-array failures. The original failure and disclosed continuation remain preserved.

The post-run diagnostic validates against this acquisition's dependency and completed-status hashes. It explains why metadata enumeration records Torch 2.1.1 while workers report 2.13.0+cu130 / CUDA 13.0. It establishes later installation identity, not retrospective worker binary identity. Original metadata remains unchanged.

Independent verification matches the supplied tar SHA-256 `3c7cff78082c7b344a38d13d216ae1c647315720eea1b4690b4e969d618b4af6`, all **292 manifested file hashes**, and the manifest itself (293 tar members/extracted files). The verifier consumes 264 evidence files; other transferred auxiliary files are hash-checked only. The required source-review SHA remains `ab4977b38481e4dd133fb3b838a3fc51591c51ca22b43c106f163964101d301a`. The new Git proof was independently reproduced byte-for-byte: 1,081 corrected and 1,230 proposed tracked files, resolving four symlinks per source.

Measurements remain bound to corrected `66789ac4a7468094b0cc3ca1498a1de67e0311f6` and proposed `f582b6fd250d0b82612492979e01e645d5c07afc`. Native binaries, input bytes, and strain samples are supported by acquisition receipts rather than transferred bytes. Later formatter/publication revisions and separate Qlty execution are outside this measurement proof.

Validation: **47 self-tests**, **13 final-evidence rejection tests plus one passing control**, and Flake8 `F401,F821,F822,F823` pass. Acquisition bytes remain unchanged. Default verifier exit 0 means consistent evidence; scientific failure remains explicit. Missing, corrupt, or inconsistent evidence exits 1.

- [Concise machine-readable report](final-report.json)
- [Full independent verification](final-verification-2.json)
- [Transfer verification](transfer-verification.json)
- [Reproduced Git proof](source-proof-reproduced.json)
- [Self-tests](final-self-test-2.json) and [evidence rejection tests](final-evidence-rejection-tests.json)
