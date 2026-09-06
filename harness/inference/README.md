# Public inference benchmark

`benchmark.py` is external to PyCBC and accepts a built, clean checkout at exactly
`607bce53ead14f12af32552a5b2441d3bc667267`. It uses full public
`GaussianNoise` and `Relative` models. It does not replace kernels, patch model
internals, or manufacture model objects. Missing dependencies, constructor
failures and parity failures are recorded; they are not benchmark results.

The local import smoke attempt was blocked by the checkout's missing
`pycbc.events.eventmgr_cython` extension. See `smoke/prepare.json`. No local
extension build or remote execution was attempted. The primary runner must
perform the model smoke test in its built environment before the full matrix.

## Run

Pass absolute paths for the built checkout, its Python interpreter, this script,
and a fresh output directory. For example, on the local host (requires built
extensions before it can succeed):

```sh
/Users/xangma/miniconda3/bin/python /Users/xangma/repos/pycbc/artifacts/torch-benchmark-20260906/inference/benchmark.py \
  --mode orchestrate \
  --root /private/tmp/pycbc-torch-restack.5U9D7C/tree \
  --python /Users/xangma/miniconda3/bin/python \
  --output /Users/xangma/repos/pycbc/artifacts/torch-benchmark-20260906/inference/run
```

Substitute the remote environment's absolute paths when running there. The
orchestrator launches workers **serially** in independent subprocesses, exports
`PYTHONPATH=<root>:<existing PYTHONPATH>` and thread limits before process startup,
and uses the checkout as each subprocess's working directory. A direct worker
also prepends the resolved `--root` to `sys.path` before its first PyCBC import,
then verifies that `pycbc.__file__` resolves beneath that root. Built extension
modules must be available in that checkout; unrelated installed PyCBC files are
not substituted. `--python` selects child interpreters in orchestrator mode;
a direct worker uses the interpreter that launched it.

For a bounded smoke test, use `--mode prepare --groups 5 --evaluations 2
--duration 8 --sample-rate 512` with a separate output directory, then run each
worker route/model against that directory. Smoke results describe that smaller
case and must not be combined with the default 32-second case.

The composable sequence is:

```text
PYTHON benchmark.py --mode prepare --root ROOT --output OUTPUT
PYTHON benchmark.py --mode worker --root ROOT --output OUTPUT \
  --model gaussian --route standard_cpu --threads 1 --replicate 0
PYTHON benchmark.py --mode worker --root ROOT --output OUTPUT \
  --model gaussian --route torch_cpu --threads 1 --replicate 0
PYTHON benchmark.py --mode worker --root ROOT --output OUTPUT \
  --model gaussian --route cuda --threads 1 --replicate 0
```

`prepare` must finish successfully before workers. Reuse its exact `case.json`
and `case.npz` across all routes, thread counts and replicates. Do not regenerate
between routes. Use `--model relative` for the full relative-binning model.

## Default matrix

| Model | Routes | CPU threads | CUDA host threads | Replicates | Sample groups | Evaluations/group |
| --- | --- | --- | --- | --- | --- | --- |
| GaussianNoise | standard_cpu, torch_cpu, cuda | 1, 4 | 1 | 3 separate processes | 5 | 16 |
| Relative | standard_cpu, torch_cpu, cuda | 1, 4 | 1 | 3 separate processes | 5 | 16 |

There are 30 workers plus one separate CPU preparation/reference process.
Each worker uses one cold point, twelve correctness points, eight warmup points,
and eighty distinct timed points. The same points are reused across replicates
and routes, allowing paired comparisons. Every point within a worker differs,
including its `(mass1, mass2)` pair, so an identical intrinsic waveform or
cached likelihood cannot stand in for the requested evaluation.

## Scientific case and APIs

- Two detectors, H1 and L1; 32 seconds, 2048 Hz, 32769 complex128 frequency bins,
  delta_f=1/32 Hz; likelihood band 20–1023 Hz.
- Public `FDomainDetFrameGenerator(FDomainCBCGenerator, ...)` generates a
  standard CPU TaylorF2 injection with masses 10 and 8 solar masses, distance
  500 Mpc, zero aligned spins, fixed seeded sky/orientation/time parameters.
- A positive synthetic colored PSD is used, with independent seeded complex
  Gaussian frequency noise added to each detector. `case.json` records the
  exact formula, parameters, seed and every parameter point. This is a
  reproducible synthetic benchmark, not a detector PSD or astrophysical result.
- The twelve CPU reference evaluations are computed with real models before
  any target worker starts. Target parity compares both `loglikelihood` and
  `loglr` for varied masses, distance, inclination, phase, sky position,
  polarization and coalescence time. Default tolerance: rtol=2e-7, atol=2e-5.
  Raw expected/actual values and absolute errors are preserved. A failed parity
  check stops steady-state timing, without changing tolerances automatically.
- Relative uses `epsilon=0.1`, the injection as fiducial parameters, and disables
  phase/distance marginalization and Earth rotation. It performs real fiducial
  waveform generation, bin construction, summary preparation and likelihood
  evaluation. Each route is compared with the standard CPU **same model**;
  this benchmark does not assert that the relative approximation is identical
  to the full Gaussian likelihood.

## Timing and residency

Every timed operation is `model.update(**params)` followed by
`float(model.loglikelihood)`. It includes fresh waveform generation, detector
projection, the likelihood reduction, and returning a host scalar suitable for
a conventional sampler. Input parameters are ordinary Python floats on all
routes. Fixed data and PSDs are constructed on the active backend during setup
and stay there as the model permits. Each result records actual model data
storage types/devices and the pre-conversion likelihood result type/device.
Relative's public implementation may perform CPU work or transfers; those
operations remain in the measurement. A CUDA route label alone is not a claim
that every operation executes on the GPU.

CUDA is explicitly synchronized immediately before and after each timed group;
the required host scalar return also synchronizes each evaluation. These are
end-to-end scalar sampler timings, not asynchronous enqueue timings or batched
GPU throughput. Five group totals and per-evaluation means are retained, plus
every returned likelihood. Do not treat the 15 groups as 15 independent
processes: compare distributions of the three replicate summaries.

Imports, first scheme/context setup, model/data construction, and the first
update/likelihood call have separate nanosecond records. These are first-use
component measurements in a fresh worker. Correctness checks and warmup follow
the cold measurement, and are excluded from the five steady-state groups.
The orchestrator also records whole subprocess wall time, which includes all
work and Python startup and must not be described as cold likelihood latency.

`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS` and `NUMEXPR_NUM_THREADS` are set before numeric imports.
The active scheme receives the thread count; Torch intra-op threads are set to
the same count and inter-op threads to one. Environment values, Torch settings,
and available `threadpoolctl` information are recorded for inspection.

## Evidence and failures

- `case.json` and `case.npz`: shared input, CPU reference and content hashes.
- `MODEL-ROUTE-tN-rN.json`: raw groups, parity, cold components, model/device
  details, exact source SHA/cleanliness, interpreter, host/PID, argv, harness
  hash and case hash. Progress is saved outside timed sections. Exceptions
  preserve earlier samples and their traceback. Nonfinite numeric values are
  serialized as strings so the failure evidence remains valid JSON.
- `commands.json`, `process-receipts.json`, per-process `.log` files and
  `orchestrate.json`: complete command matrix, return codes and process timing.
- `unsupported`: unavailable CUDA; `unsupported_reference`: CPU could not
  construct/evaluate that model; `failed`: exception/dependency failure;
  `parity_failed`: route disagrees with the CPU reference. These statuses are
  not successful performance measurements. Inspect statuses, not just exit
  code, because unsupported routes return zero to allow the matrix to finish.

Model/source cleanliness is verified before and after successful workers.
No source file is modified by the harness. Use a fresh output directory for
each campaign so a deliberate rerun does not overwrite prior evidence.
