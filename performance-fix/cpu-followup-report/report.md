# Optional CPU native peak follow-up

Host: `len`. Both revisions explicitly set `PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK=1`.

- cpu-candidate: `1514327669fc7be125523b991c847868c3a2a17e` (tree `0bbf15de82c467649cb88fb6378aaf74d829cfca`).
- cpu-baseline: `bd53914be6d2e4324cc867d52b3842b77cc6729a` (tree `ffddeb39a1b90d97736b79164aa620133eec3a1b`).

Each capacity cell is median [minimum, maximum] in templates/core at real time over three separate worker medians. The paired ratio uses after/before within each replicate, then reports its median and full range.

| Configured cores | Batch | Before templates/core at real time | After templates/core at real time | Paired ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 52,659.0 (52,568.5–52,837.7) | 52,569.8 (52,537.9–52,786.7) | 0.999 (0.998–0.999) |
| 1 | 1024 | 46,412.9 (46,177.4–46,519.6) | 53,086.3 (52,998.9–53,318.7) | 1.149 (1.139–1.150) |
| 4 | 32 | 24,036.3 (23,320.8–24,140.8) | 24,569.2 (24,154.7–24,863.4) | 1.030 (1.022–1.036) |
| 4 | 1024 | 25,539.2 (25,464.3–25,575.3) | 35,544.1 (32,555.6–35,618.0) | 1.393 (1.275–1.396) |

Standard CPU controls (cpu-baseline):

| Configured cores | Batch | Templates/core at real time |
| ---: | ---: | ---: |
| 1 | 32 | 66,586.9 (66,508.2–66,706.6) |
| 1 | 1024 | 66,917.1 (66,720.6–67,131.1) |
| 4 | 32 | 38,143.9 (37,640.1–38,263.1) |
| 4 | 1024 | 40,997.1 (38,818.3–41,082.0) |

Validated 36 distinct successful worker processes and 12 parity documents covering 24 native/control comparisons. All stored parity flags passed, and trigger fields and aggregate norm differences were independently recomputed from the exported worker results.

| Parity metric | Maximum across all comparisons |
| --- | ---: |
| `max_snr_diff` | 3.81469727e-06 |
| `max_phase_diff` | 3.77149263e-08 |
| `max_sigmasq_relative_diff` | 1.19209304e-07 |
| `relative_output_l2_diff` | 1.23492611e-07 |

Three separate worker processes per source, route, thread count and batch. Throughput is the median of their three warm throughput medians. Ranges are min–max across workers, not confidence intervals. The paired ratio is the median of candidate/baseline worker ratios matched by replicate; it need not equal the ratio of the two aggregate medians.

Synthetic seeded templates and injected strain; N=131072, three strain blocks per iteration, complex64 inputs/outputs, SNR threshold 5.5. The public LiveBatchMatchedFilter.process_data API is timed with chi-square and sine-Gaussian vetoes disabled. Frame I/O, PSD estimation, bank loading, waveform generation and CLI startup are excluded. The raw unit templates/s counts one template filtered against one strain block. Native labels mean opt-in routing is enabled; admission can select a fallback for a given batch.

Real-time filtering capacity is derived from the measured template–block rate: multiply by 56 seconds of valid strain per block, then divide CPU routes by their configured 1- or 4-core budget. CUDA capacity uses one GPU and includes the timed host work. The CPU divisor is the configured thread count, not measured CPU utilization; affinity CPUs 8–11 are distinct physical cores. This is synthetic matched-filter component capacity with vetoes and I/O excluded, not complete production search capacity. Raw template–block rates and all unchanged ratios remain in JSON. The live harness's legacy throughput_wps_summary unit says waveforms/second, but its numerator is batch size times block count: template–block evaluations, not generated waveforms. That raw label is preserved; standalone waveform generation has a separate waveforms/second metric. The 56 s basis, pinned source trees, full file hashes and source excerpts are recorded in search_capacity_basis.

Each baseline and candidate is checked against the corresponding standard CPU control: trigger counts and template IDs, end times within 1e-4 s, SNR and wrapped phase differences below 1e-3, relative sigma-squared and aggregate output-norm differences below 1e-3. These checks do not establish pointwise equivalence of every matched-filter output sample.

Sequential fresh workers, with three timed public API iterations after one cold iteration and one warmup. Each iteration filters all templates against three strain blocks; throughput is 3*batch/seconds. No profiling instrumentation is included in these worker commands.

Both revisions explicitly enable PYCBC_TORCH_CPU_NATIVE_BATCH_PEAK=1 alongside native correlation and FFTW batch routing. Enabled routing permits fallback when admission requires it; these records do not count native dispatches. The branch_standard control belongs to cpu-baseline. This report summarizes only the optional CPU follow-up, not the other postcheck tests or profiles.

The input manifest records SHA-256 hashes for the source snapshots, completion ledger, plan, worker JSON, command receipts, raw logs, parity documents and local reporting/runner scripts. Before/after source snapshots and native binary hashes are unchanged within each revision. JSON retains every worker median, paired ratio and the separate ratio of aggregate medians.
