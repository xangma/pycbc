# Existing upstream work on CPU pointwise chi-square precision

Search date: **2026-09-22**, approximately 07:52–07:57 UTC. Repository: `gwastro/pycbc`. Read-only GitHub API/`gh` searches; no repository edits or remote mutations.

**Result:** no existing issue or PR specifically fixing CPU recurrent phase drift, the truncated CPU π constant, input-precision-dependent shift casting, or CPU sum precision was found in this search. This is a search result, not proof that no unmentioned or unindexed work exists. The strongest related precedent is the merged CUDA phase-accuracy fix #2950.

## Relevant issues and PRs

| Work | State on search date | Scope and overlap |
| --- | --- | --- |
| [#2950: Change CUDA chisq to use more accurate phase calculation](https://github.com/gwastro/pycbc/pull/2950) | Merged 2019-10-23 | Direct numerical precedent. Reports CPU/GPU χ² differences of several to tens of percent associated with large phase arguments. Reduces the integer frequency-index × time-index product modulo the series length before CUDA intrinsic trigonometry for power-of-two lengths; otherwise uses the more accurate trigonometric function. **Only `chisq_cuda.py` changed.** Related to phase accuracy, but does not fix CPU recurrence, truncated π, float32 shift storage, or accumulation. |
| [#2638: Chisquared -> Cython](https://github.com/gwastro/pycbc/pull/2638) | Merged 2019-04-12 | Introduced `chisq_cpu.pyx`, porting the earlier implementation. Discussion concerns allocation, threading, and runtime; no corrective precision patch. Requests production-like segment lengths, bin counts, single precision, and timings for 1, 2, and 5 selected samples. |
| [#5158: multi_inspiral: avoid repeated power chi^2 calculations in the critical loop](https://github.com/gwastro/pycbc/pull/5158) | Merged 2025-08-19 | Caches single-detector χ² values across time slides and sky positions. Does not change CPU arithmetic. Discussion of fixing PSD inputs to stabilize comparisons is useful methodology, not evidence of this defect. |
| [#5148: Large potential speedup of pycbc_multi_inspiral](https://github.com/gwastro/pycbc/issues/5148) | Closed | Profiling, caching, repeated calculations, trigger storage, and FFT-versus-selected-point strategy. Here “accumulation” includes stored triggers, not the floating-point sum defect. No matching numerical fix. |
| [#5249: DRAFT: Trying to improve pycbc_inspiral performance on GPU](https://github.com/gwastro/pycbc/pull/5249) | Open; GitHub `isDraft=false` despite title | GPU batching and CuPy/CUDA performance work. Changes `chisq.py` and `chisq_cupy.py`, but **not `chisq_cpu.pyx`**. Adjacent active work, not a duplicate CPU correction. |
| [#3753: Time delay and chisq values in pycbc_multi_inspiral](https://github.com/gwastro/pycbc/issues/3753) | Closed | Inconsistent time-delay alignment between a rolled SNR series and unrolled correlation. A semantic indexing bug, distinct from float32 failing to represent adjacent integers above 2²⁴. |
| [#5235: Make changes to constants, use numpy/astropy but use LAL when requested](https://github.com/gwastro/pycbc/pull/5235) | Merged 2026-01-05 | Constants infrastructure and associated replacements. Changes `chisq_cupy.py`, not `chisq_cpu.pyx`; does not remove its truncated π literal. |
| [#5280: Noise triggers from pycbc_multi_inspiral show periodic structures](https://github.com/gwastro/pycbc/issues/5280) | Open | Discussion investigates segmentation and PSD estimation, including overlap bias. Similar-looking periodic behavior is not evidence of the CPU phase-recurrence issue. |

Other screened false positives include [#3514](https://github.com/gwastro/pycbc/pull/3514) (cached template normalization), [#5295](https://github.com/gwastro/pycbc/pull/5295) (template-length rounding), [#5163](https://github.com/gwastro/pycbc/pull/5163) (sine-Gaussian veto performance), and [#5395](https://github.com/gwastro/pycbc/pull/5395) (ratio filtering; no changed `chisq`/`vetoes` file).

## Corroborating source history

`gh api 'repos/gwastro/pycbc/commits?path=pycbc/vetoes/chisq_cpu.pyx&per_page=100'` returned one commit: [a43424da24532e84e9f79c678c9d43c73063ff2c](https://github.com/gwastro/pycbc/commit/a43424da24532e84e9f79c678c9d43c73063ff2c), dated 2019-04-12, “Chisquared -> Cython (#2638)”. Thus the current default-branch history for that path contains no later corrective commit. This does not inspect every unmerged branch or historical filename.

Useful benchmarking discussion: [production-like configuration](https://github.com/gwastro/pycbc/pull/2638#issuecomment-482590709), [1/2/5 selected samples](https://github.com/gwastro/pycbc/pull/2638#issuecomment-482591248), [reported timings](https://github.com/gwastro/pycbc/pull/2638#issuecomment-482606901). Related [FFT versus selected-point discussion](https://github.com/gwastro/pycbc/issues/5148#issuecomment-3058213691).

## Search coverage

Each query used `gh api -X GET search/issues -f q='repo:gwastro/pycbc QUERY' -f per_page=100`, without state or issue/PR restrictions. Therefore both open and closed issues and PRs were searched. Counts below were all below the page limit. The API can tokenize punctuation unexpectedly, so both plain and quoted identifiers were tried. Relevant hits were inspected through bodies/comments and, where needed, changed-file lists or diffs.

| Query after repository qualifier | Hits | Query after repository qualifier | Hits |
| --- | ---: | --- | ---: |
| `chisq precision` | 9 | `chi-square precision` | 2 |
| `chisq_cpu` | 0 | `shift_sum` | 0 |
| `"3.141592653"` | 1 | `chisq rounding` | 3 |
| `chisquared precision` | 1 | `"phase" "drift"` | 3 |
| `"float32" "shift"` | 2 | `"chi" "accumulation"` | 1 |
| `"chisq_cpu.pyx"` | 2 | `"point" "chisq"` | 48 |
| `"point_chisq"` | 0 | `"shift_sum"` | 0 |
| `"chi-square" "roundoff"` | 0 | `"chisq" "float32"` | 5 |
| `"chisq" "complex64"` | 2 | `"chisq" "accuracy"` | 1 |
| `"chi" "recurrence"` | 0 | `"truncated" "pi"` | 1 |
| `"chisq" "double"` | 8 | `"chisq" "numerical"` | 4 |
| `"chisq" "phase"` | 9 | `"pointwise"` | 0 |
| `"chi-square" "single precision"` | 0 | `"chi-squared" "precision"` | 2 |
| `"16777216"` | 0 | `"24" "float32"` | 6 |

## Implications for the proposed fixes

- Cite #2950 as related phase-accuracy precedent, clearly distinguishing its CUDA implementation from the CPU fixes.
- Keep π, shift representation, and internal arithmetic as independently reviewable changes. Phase and sum arithmetic can share one implementation PR while having separate regression examples.
- For π, use phase-sensitive interference between coefficients: a single coefficient's squared magnitude cannot reveal a pure angular error. The existing CPU complex128 path is not an exact reference when it retains truncated π.
- For shifts, demonstrate lost adjacent sample indices near 2²⁴ and preserve any supported fractional-shift behavior. Distinguish representability from time-delay alignment (#3753).
- For internal arithmetic, test recurrence drift separately from addition loss at zero phase, then a combined case against an independent reference. Ordinary per-operation rounding alone is not a separate established defect.
- Benchmark realistic bin widths and 1, 2, and 5 selected samples as well as larger batches; widening arithmetic has a performance tradeoff. These search findings do not establish its cost or superiority over periodic reseeding.

No numerical models, benchmarks, or tests were run for this search.
