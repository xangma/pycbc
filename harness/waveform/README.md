# Public TaylorF2 benchmark harness

Pinned source: `607bce53ead14f12af32552a5b2441d3bc667267`. These external scripts do not modify source files. Run against a clean checkout with that exact HEAD and its built extensions. Results belong outside that checkout. The selected Python must already import PyCBC's dependencies, Torch, and LALSuite.

Recommended campaign on a CUDA host:

```sh
/path/to/python -B /path/to/waveform/run.py \
  --root /path/to/clean/pycbc \
  --python /path/to/python \
  --out /path/to/new-results
```

The default matrix uses batches 1, 8 and 32; CPU threads 1 followed by 4; CUDA with one host thread; three independent worker processes per cell; five samples per worker. All workers run sequentially. Each sample averages enough repeated public API calls to target 50 ms, capped at 64 calls. Raw sample times and repeat counts are retained. Each worker has a 180-second timeout. The orchestrator writes its PID and worker PIDs, full commands, log paths and progress to `manifest.json`, and prints each start/completion. Stop the orchestrator and the active worker PID from the manifest if cancelling a campaign.

A fast initial runtime smoke check:

```sh
/path/to/python -B /path/to/waveform/run.py \
  --root /path/to/clean/pycbc --python /path/to/python \
  --out /path/to/new-smoke-results \
  --routes standard-cpu torch-cpu-scalar torch-cpu-batch torch-cuda-scalar torch-cuda-batch \
  --batches 1 --threads 1 --precisions double --replicates 1 --samples 1 --max-inner 1
```

Routes:

| Request | Actual operation at this SHA |
| --- | --- |
| `standard-cpu` | Public `get_fd_waveform`, standard CPU/LAL, Python loop over rows |
| `torch-cpu-scalar` / `torch-cuda-scalar` | Public `get_fd_waveform`, native Torch, Python loop over rows |
| `torch-cpu-batch` / `torch-cuda-batch` | Public `get_fd_waveform_batch`, one native Torch batch call |
| `torch-cuda-triton-batch` | Explicit unsupported row: no TaylorF2 Triton entry point exists |
| Any CPU/CUDA route with `single` precision | Explicit unsupported row: public API selects complex128 internally |

Single precision is not simulated by casting outputs. MPS has a separate single precision path but is outside this CPU/CUDA campaign. Unsupported hardware is recorded separately from execution failures. Unsupported source capabilities are recorded before scientific imports; package versions are queried without loading them.

The waveform grid has 4097 bins (`delta_f=0.25`, `f_lower=20`, `f_final=1024` Hz). A deterministic sequence of small aligned-spin BNS parameter rows is shared by every route; batches above eight repeat that sequence. The complete physical inputs are saved in every worker result. Both polarizations are generated, and all samples, including zero padding, enter verification. This is a bounded throughput workload, not an astrophysical parameter-space survey.

Cold means the first waveform call after imports, capability discovery and scheme entry; it does not mean fresh operating-system caches or a cold CUDA driver. Import time is recorded separately. Two warmup calls and one calibration call precede the five timed samples. CUDA is synchronized before and after every timed block. Timing includes public API validation, host parameter conversion, output allocation and Python scalar loops where applicable. Output host copies, correctness checks and instrumentation are excluded. No gradients are tracked.

After timing, a separate public API call counts native scalar, native batch and LAL FD generator dispatch. CPU baseline must call LAL once per row. Native scalar must call Torch once per row; native batch exactly once; both must make zero LAL calls. Unexpected dispatch fails the result. Actual output dtype and device are checked and saved. The full waveform checks use existing test tolerances:

- Native scalar against CPU/LAL: complex relative L2 `<1e-11`, exact zero support, matching grid and epoch, both polarizations. This follows `test_taylorf2_public_torch_parity_and_dispatch`.
- Batch against the native scalar on the same device: pointwise complex relative error `<=2e-10`, with zero absolute tolerance, following `test_each_batch_row_matches_scalar_taylorf2`. Those scalar outputs are independently checked against CPU/LAL with the preceding tolerance.
- Direct batch-to-CPU/LAL relative L2, pointwise complex error, maximum relative amplitude error and wrapped phase error are also recorded. A new direct-batch tolerance is not invented. Finite values and exact zero support remain mandatory. Scalar outputs also record those amplitude/phase diagnostics.

Failed dispatch/parity cells retain timing evidence but are excluded from speedups. A summary cell is eligible only when every requested independent replicate passes. Summaries use the median of per-process medians, expose their min/max, and compare equal batch sizes and host-thread settings to the standard CPU baseline. Inner repetitions are not treated as independent statistical observations. Raw records preserve exact source/tree SHA, clean status, source and harness hashes, module origins, runtime versions, hardware, effective Torch thread settings, environment switches, cold timings and sample timings.

Initial validation here covers syntax, CLI parsing and unsupported-row recording only. Run the runtime smoke above before the full campaign; no benchmark numbers or performance conclusions are bundled with these scripts.
