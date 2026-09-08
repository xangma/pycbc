# Residual performance review: completed descriptor reuse timing

Descriptor reuse improved both standard and CUDA whole-process performance in the
completed four-repeat experiment. The independent replay passed all 20 executable
runs, the fresh native gate, and all 51 scientific comparisons. The owner summary
agrees exactly with the independently reconstructed raw metrics and paired results.

| Scheme and helper mode | Median wall (s) | Four-run range (s) | Median setup (s) |
| --- | ---: | ---: | ---: |
| standard-original | 66.849562 | 66.791962–67.038664 | 12.107852 |
| standard-cached | 65.370059 | 65.189339–65.505714 | 10.549228 |
| cuda-original | 21.976242 | 21.893677–22.051733 | 10.342075 |
| cuda-cached | 20.291608 | 20.213787–20.307819 | 8.634298 |

Each family improved in all four paired repeats; its cached and original
wall-time ranges do not overlap. Reduction of the wall-time medians and
the median of paired percentage reductions are different summaries:

| Reuse comparison | Reduction of medians | Paired reduction median | Paired reduction range |
| --- | ---: | ---: | ---: |
| standard | 2.21318% | 2.25201% | 2.20901–2.39943% |
| cuda | 7.66571% | 7.66457% | 7.24346–8.33470% |

CUDA with reuse achieves a paired whole-process speedup median of 3.222772×
(3.218219–3.229945×) against the contemporaneous standard with reuse.
The ratio of wall-time medians is 0.310411349.
This supports descriptor reuse as a shared setup optimization. GPU kernel speed
was not the changed mechanism. Setup medians are secondary telemetry; independently
computed medians need not add to the median wall time.

The closed archive contains 317 verified files. Four fresh qualifications capture
conditioned strain and actual PSD arrays; sixteen timed runs retain the same work
and pass persisted H1 science checks without recapturing strain or PSD. All 51
comparisons pass default tolerances, and 38 same-scheme comparisons additionally
match all 18 science datasets exactly. No substitutions were needed.

The native gate exercises nine transforms with 18 distinct retained input/output
pointers, six externally recorded reuse hits and three descriptors freed exactly
once. Cached executable children each own and free three descriptors; original
children install no cache. Timed calls carry no per-call clocks or hit counter.
The original and cached children construct the same helper; full wall time includes
construction, candidate installation, cleanup, restoration, receipts and exit.
The independent terminal audit confirms the entire controller group is absent
and the benchmark lock was reacquired.

The earlier Torch CPU executable gap remains 37.466764 s
against the standard from that earlier campaign. The qualified promoted CPU plan
spent 28.861 ms in the native stage and 4.492 ms in paired conversion, about 13.446%
of a staged call. These observations do not prove any cost unavoidable or partition
the executable gap. The qualified NumPy copy candidate was slower in all three
workers and remains rejected; retain Tensor.copy_. There is no new Torch CPU timing
or inferred CPU gap against the cached standard in this experiment.

The four repetitions cover one fixed 384-template workload, one pinned CPU core
and one CUDA device, with warm filesystem/library state. They establish no
statistical significance, general workload guarantee or saturated multicore result.
Native array equality was checked in the pinned executed gate; those arrays were
not archived. CPU stage diagnostics used inter-op 64 while executable trials use 1.
The prototype result does not certify production integration. A production change
needs its own implementation review and fresh qualification. No additional CPU
precision matrix is justified; first-creation cost remains a lower-priority lead.

[Raw archive replay](/Users/xangma/repos/pycbc/artifacts/torch-profiling-investigation-20260908/fast-reuse-results-v1-review.json),
[owner-summary cross-check](/Users/xangma/repos/pycbc/artifacts/torch-profiling-investigation-20260908/fast-reuse-summary-v1-crosscheck.json),
[previous CPU and profiling context](/Users/xangma/repos/pycbc/artifacts/torch-profiling-investigation-20260908/RESIDUAL-ANALYSIS-v2.md),
[exact values and evidence pins](/Users/xangma/repos/pycbc/artifacts/torch-profiling-investigation-20260908/residual-analysis-v3.json).
