# Offline CUDA graph selection v1

The next bounded candidate is the existing offline symmetric CUDA graph, with eager threshold rounding restored. This targets repeated Python dispatch and CUDA launch orchestration in correlation, IFFT and fixed-shape clustering. It does not address the remaining Torch CPU parity gap.

The current source is clean `ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f`; the companion JSON pins seven inspected files. Owner is the sole len operator. This is source review and a proposed qualification contract, not a measured improvement or production approval.

## Evidence and exact seam

- `matchedfilter.py:345–384` enables the path with `PYCBC_TORCH_CUDA_GRAPH=1` and retains the existing empty-trigger branch and return contract.
- `matchedfilter_torch.py:103–157` captures correlation, inverse FFT and fixed-shape clustering per `(segnum, window)`, after three side-stream warmups. Dynamic survivor compaction remains after replay.
- `threshold_torch.py:891–929` allocates block-max/index/mask scratch and returns fresh advanced-indexed sparse outputs. General capture eligibility and later storage/window mutation are insufficiently guarded by current code; a bounded trial must prove its fixed workload preconditions. This is not general API qualification.
- `matchedfilter_torch.py:978–1032` retains correlator input/output tensors; `torchfft.py:1120–1140` uses unchanged CUDA `torch.fft.ifft(..., norm="forward", out=fout)` arithmetic. No precision or reduction-route change is proposed.
- `bank.py:903–943` reuses the supplied template output, clears it and decompresses the next template in place. `pycbc_inspiral:240–310` processes five segments sequentially and consumes each correlation/SNR in veto calculations before the next filter overwrites shared scratch. Preserve this ordering.

The prior frozen CUDA profile contains 1,920 threshold calls, 1.357452 s threshold host range and 0.026126 s threshold kernels. These overlapping instrumented boundaries are motivation, not an additive overhead estimate or speedup prediction. Existing graph code had no exhausted fixed-workload graph experiment in the owner's prior candidate record. Scalar-hoist and live peak-helper experiments are distinct.

## Required numerical correction

Eager thresholding at `threshold_torch.py:1065` converts the raw Python threshold into the series real dtype, then multiplies that float32 tensor by itself. The graph currently multiplies Python floats before converting/filling a float32 tensor. Local arithmetic demonstrates different trigger selection: threshold `1.00000003` gives eager squared threshold `1.0` and old graph `1.0000001192092896`. Complex64 sample `(1, 0.0003452669770922512)` has float32 power `1.0000001192092896`, passing eager strict comparison but failing the old graph comparison. This is a CPU scalar demonstration; a real CUDA regression must exercise the cluster kernel.

Minimal correction: retain a raw float32 threshold tensor, fill the current raw threshold before replay, and capture its float32 multiplication before the existing graph cluster step. Preserve multiplication order, strict threshold behavior, tie/NaN handling, correlation, FFT, compaction and all downstream scientific arithmetic. Fixing only initialization is insufficient: later replay updates must also convert before squaring.

## Qualification before any timing

Use the accepted descriptor-reuse runner on both eager and graph arms. Preserve source/native/input pins, core-8 affinity, all native and Torch pools at one, workload and existing comparator budgets. Establish actual CUDA/Triton support and graph capture/replay success; a skipped or silently eager fallback cannot pass.

Require exact contiguous complex64 CUDA tensors, fixed pointers/shapes/strides/dtypes/devices, stable template/segment/correlation/SNR bindings, fixed per-segment analysis slices, cutoffs and one positive integer window. Exclude AD, inference, tensor subclasses, lazy views and storage rebinding in this artifact trial. Prove current-stream dependencies through capture and replay, with sequential consumers before shared-workspace reuse. Window/key changes that replace cluster scratch cannot reuse an old capture.

Run a bounded kernel-level regression covering the rounding counterexample and nearby strict threshold values, changed norms/thresholds, zero/all survivors, tie and NaN behavior, and exact unchanged selected indices/values. Real filtering checks must change template contents and segment selection, compare full correlation and SNR against eager with unchanged precision budgets, and retain sparse output arrays across later replay and template overwrite. Check default and explicitly selected current-stream execution with proper producer/consumer ordering.

For the frozen complete executable, qualification alone records five successful captures, 1,920 replays and stable storage/geometry throughout, along with capture/warmup cost and allocated/peak device memory. Preserve 384 templates, five 512-second segments, 1,904 valid seconds, 16 chi-square bins, PSD and conditioned-strain identity. Require graph/eager same-scheme equality of all 18 H1 science datasets and the unchanged standard-reference comparator. Every timing output must pass the same scientific checks.

## Timing and decision boundary

Freeze the detailed protocol before running. Compare incremental graph versus eager with accepted descriptor reuse applied equally; the reviewed CUDA cached median is about 20.292 s. A stale uncached control would confound the gain. Use fresh serial workers, balanced fixed order, retain every sample, and include interpreter startup, common helper setup, CUDA initialization, three warmups per graph, capture, filtering, output, cleanup and process exit in full wall. Qualification hooks, capture tracing and per-call counters stay out of timing workers.

Report raw paired full-wall samples, medians/ranges, paired differences/ratios and secondary setup/internal values. The current whole executable includes substantial work outside this captured span; no gain follows merely from warm replay. If native capture, numerical/lifetime gates or the frozen performance decision fails, reject this candidate without relaxed budgets or adaptive resampling. Keep the CPU gap explicitly unresolved.
