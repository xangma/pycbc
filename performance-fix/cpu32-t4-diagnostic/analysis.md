The remaining CPU-native B32/T4 slowdown has no isolated cause in these diagnostics. Overlap admission is substantially cheaper and complete peak-selection time falls in the cProfile sample. The positive difference is concentrated in unchanged FFTW plan execution and other batch-processing work, while the separate Torch profile shows the candidate faster overall.

The original unprofiled report records a paired candidate/baseline template-block rate ratio of **0.941504**, a **5.85% rate loss** (0.934050–0.947121 across three process pairs). That corresponds to 6.21% more time for fixed work. Median capacities are 16,199.1 and 15,283.4 templates/core at real time. These are synthetic component capacities.

The timing candidate is `97bf1614` (tree `d9819f6f`); this diagnostic uses candidate-v2 `0d005812` (tree `fa7a7c09`) against baseline `dfd42bf7`. The original candidate tree equals local commit `450ab3f9`. Source inspection confirms the same CPU chunk arithmetic and unchanged FFTW/correlation source between candidate phases; this remains a diagnostic, not a new candidate-v2 throughput result.

cProfile captured call 7, strain block 0, once per source. These disjoint terms reconcile to total recorded time:

| Work | Baseline ms | Candidate-v2 ms | Difference ms |
|---|---:|---:|---:|
| FFTW plan execution including target version update | 13.985 | 15.841 | +1.856 |
| Native correlation wrapper including admission checks | 3.056 | 2.120 | -0.935 |
| Complete peak helper | 10.482 | 9.943 | -0.539 |
| Batch orchestration own time | 0.291 | 1.819 | +1.529 |
| All other exclusive work | 0.809 | 0.776 | -0.033 |
| Total | 28.623 | 30.501 | +1.878 |

Within correlation, `_batch_outputs_are_disjoint` falls from **1.104 to 0.111 ms**; `_spans_overlap` calls fall from **1,552 to 32**. Source changes replace pairwise output/input and output/output comparisons with a sorted interval pass. These nested times are already included in correlation above. Native correlation wrapper own time is nearly unchanged (1.104 versus 1.132 ms).

The peak helper falls from **10.482 to 9.943 ms** cumulatively, despite its own charged time rising from **0.139 to 3.619 ms**. Caller own time can include expression operations and deallocation work without separate cProfile events; this increase does not establish extra Python instruction cost.

Torch profiling captured the next call, strain block 1. Exact event durations below are reconstructed from the saved trace, subtracting direct same-thread children, and checked against `operators.txt`:

| Operation | Calls baseline → candidate | Self ms baseline → candidate |
|---|---:|---:|
| `aten::pow` | 1 → 8 | 4.072 → 4.331 |
| `aten::sum` | 1 → 0 | 6.474 → 0.000 |
| `aten::fill_` | 1 → 0 | 2.816 → 0.000 |
| `aten::add` | 0 → 4 | 0.000 → 2.064 |
| `aten::argmax` | 1 → 4 | 1.486 → 2.066 |
| All recorded operators | 41 → 152 | 15.019 → 9.001 |

The baseline reduces `[32, 114688, 2]` squared real/imaginary values with `sum`; the candidate performs separate real/imaginary squares and addition in chunks of **9, 9, 9 and 5 templates**. Both square **7,340,032 scalar elements** and search **3,670,016 magnitudes**. Chunking removes the sum/fill operation but increases operator events from 41 to 152 and raises observed `argmax` self time. These are work and chunk-size observations, not measured peak memory.

The Torch trace span is 33.844 → 29.016 ms, opposite to the cProfile direction. Recorded operators cover 15.019 → 9.001 ms; remaining trace spans (18.824 → 20.015 ms) mix opaque native code, Python and profiler overhead and cannot be attributed from this trace.

Both workers ran serially on `len`, affinity CPUs 8–11, four Torch/OMP/MKL threads, FFT size 131072, batch 32 and the public `LiveBatchMatchedFilter.process_data` surface. Native correlation and FFTW are enabled; optional native CPU peak selection is not. Before/after tracked-source and native-binary snapshots match, all diagnostic profile hashes validate, and the six original worker medians reproduce the report.

There is only one sample for each profiler/source combination; the two profilers observe different strain blocks and introduce different overhead. FFTW implementation bytes match across sources, but these records do not resolve scheduling, worker-thread waits, cache effects or memory bandwidth. Retain the original rate loss and this limited attribution; the evidence does not justify another automatic performance-code change.

`analysis.json` contains the complete metrics, source references, accounting checks and **36 SHA-256-bound inputs**.
