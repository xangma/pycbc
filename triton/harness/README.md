# TaylorF2 Triton comparison

These external tools compare `PYCBC_TAYLORF2_TRITON=0` and `=1` through the
complete public `get_fd_waveform_batch("TaylorF2", ...)` call at the **same
explicit, clean candidate revision**. They do not edit the measured checkout.

Default campaign: batches 1, 8, 32, 128 and 512; 4,097 and 32,769 frequency
bins; complex128 `hplus` and `hcross`; one host thread; three independent
processes per route and workload. Each process records five synchronized
timed groups lasting at least 50 ms. Route order rotates between replicates.
The JSON summary uses the median of worker medians and their observed range,
not pooled calls or a confidence interval. Both faster and slower results
remain visible; failed rows have no speedup.

Set `TF_ROOT` to the prepared checkout, `TF_SHA` to its reviewed commit,
`TF_PYTHON` to the Linux environment interpreter, and `TF_HARNESS` to this
directory on that host. Use fresh output directories for every attempt.

Quick on/off smoke, two processes:

```bash
"$TF_PYTHON" -B "$TF_HARNESS/run.py" --root "$TF_ROOT" \
  --expected-sha "$TF_SHA" --python "$TF_PYTHON" \
  --out "$TF_OUTPUT/smoke" --batches 8 --delta-f 0.25 --replicates 1
```

Full comparison, 60 processes:

```bash
"$TF_PYTHON" -B "$TF_HARNESS/run.py" --root "$TF_ROOT" \
  --expected-sha "$TF_SHA" --python "$TF_PYTHON" \
  --out "$TF_OUTPUT/full"
```

Add `--routes cuda-off cuda-on standard-cpu` to include the complete scalar
LAL CPU loop as a third route (90 processes). It uses the same parameter rows
and one thread. The default compares only the two CUDA batch routes.

Timing includes public argument validation, host-list conversion, coefficient
work, allocation, evaluation and output wrapping. Host output copies and all
instrumentation occur afterward. Autograd is disabled. Every worker has a
fresh Triton cache; the separately reported cold call includes compilation
when needed plus other lazy initialization. It is **not isolated compiler
time**. The CUDA driver cache is not cleared. No cached output is reused.

Qualification checks every output bin and both polarizations of every row:

- Gate on versus gate off: maximum pointwise relative error and relative L2
  at most `2e-10`, exact zero support and exact metadata.
- Gate off versus native scalar: the same full comparison and tolerances.
- Native scalar versus LAL: relative L2 at most `1e-11`, exact zero support
  and exact metadata. This retains the established scalar reference check.
- Direct route versus LAL errors are additionally reported without another
  numerical tolerance. Every comparison still requires finite values and
  exact support and metadata.

All actual host input rows and their hash are saved. The eight distinct
deterministic mass pairs repeat for larger batches, so exact duplicate rows
reuse their complete scalar references; no bins or batch rows are sampled.

After timing, a separate public call counts native and LAL dispatch, the
`evaluate_taylorf2` launcher, and the actual Triton JITFunction's successful
`_taylorf2_kernel.run` launches. A CUDA synchronization follows that call.
Gate on requires one launcher and one actual kernel launch; gate off requires
zero of each. Setting the environment variable alone cannot qualify a row.

`manifest.json` preserves each command, PID, log, return code and status;
each worker records full revision, tree, clean state, source and harness
hashes, package versions, module origins, inputs, errors and raw timings.
Failures write JSON when possible; timeout or abrupt worker exits get an
explicit failed artifact from the controller. SIGTERM to the controller
terminates its active worker process group. GPU availability and competing
jobs should be monitored by the campaign owner; this tool never stops them.

These are fixed-input public waveform-generation timings, not complete
search, inference, or waveform-bank production workflow timings. Scientific
edge cases and autograd fallback need the candidate's unit tests in addition
to this benchmark.
