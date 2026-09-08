# Production MKL and CUDA graph qualification

Production commit `fad7d8440bfde083f2e94ee62a0017492dfc4013`, based on
`ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f`. The two implementation commits are
`abf2528b367eefe80aafbed6d504796b292df959` (bounded MKL descriptor reuse) and
`fad7d8440bfde083f2e94ee62a0017492dfc4013` (opt-in production CUDA graph).
This archive qualifies those exact source bytes. Later stack formatting and
test-only changes are outside this frozen source snapshot.

## Results

| Check | Result |
| --- | --- |
| Native MKL and CUDA graph contracts | 43 passed, zero skips |
| Native regression suites | 240 passed; 52 skips, all unavailable MPS |
| Standard CPU, Torch CPU, Torch CUDA full executable | All passed; 1,991 triggers each |
| Own pinned reference for each scheme | All 18 scientific H1 datasets byte-exact |
| Conditioned strain, PSD arrays, segment geometry | Exact agreement with frozen scheme-specific scientific reference |
| Torch CPU and CUDA against standard CPU | All frozen numerical budgets passed; identical event identities |
| Actual production graph output | 1,920 full correlation/SNR and sparse output comparisons passed; five captures |
| Sparse output lifetime and actual consumer mutation | All retained-output, version, and expected index-offset checks passed |
| Suppressed graph replay negative control | Detected full-buffer mismatch at first reused graph: six replay calls, five completed comparisons |
| Pending CUDA allocation lifetime control | Pending work preserved with stream recording; omission reproduced allocation reuse and sentinel corruption |

Each scientific run processes the same compressed bank of 384 BNS/NSBH
templates (256/128), five 512-second segments sampled at 4,096 Hz, 1,904 valid
seconds, and 1,920 template/segment pairs. All compressed waveforms were used
without generation fallback. The actual `bin/pycbc_inspiral` executable uses
16 chi-square bins and symmetric clustering. `config.json` contains the full
analysis arguments. The CUDA configuration adds `PYCBC_TORCH_CUDA_GRAPH=1`.

The production graph executes before the independent eager oracle. The
qualifier snapshots complete production correlation/SNR buffers and sparse
outputs before the oracle runs, then checks their bytes, bindings, inputs,
and retained results. Original production sparse objects reach the actual
executable consumer; its in-place index offset is checked on the next call.
The deliberately suppressed replay fails this same comparison. Explicit
graph cleanup also passes.

The native MKL oracle reproduces the former descriptor setup independently.
The production implementation uses the documented complex storage setting,
checks configuration/compute failures, and restricts caching to the qualified
three transforms. Native CUDA contracts also cover default/custom streams,
unsupported binding fallback, tensor storage rebinding/resizing, and cleanup.

## Reproduce the archive checks

With Python 3.11+, NumPy and h5py, from any directory:

```sh
python /absolute/path/to/torch-residual-production-20260908/verify_evidence.py
```

This command performs no native workload or performance run. It verifies
every archived file against `manifest.json`, validates the decompressed
source archive and all tracked source hashes, checks test XML and both
negative controls, and reconstructs all five scientific comparisons from
the archived HDF files and original receipts. It independently checks PSD
array bytes, exact scientific reference records, runtime/thread/affinity
observations, and every graph/consumer comparison record. It checks the
three own-reference comparisons across all 18 scientific datasets.

The four excluded H1 datasets are execution telemetry:
`search/run_time`, `search/filter_rate_per_core`,
`search/setup_time_fraction`, and `search/templates_per_core`.
All scientific identities, arrays, budgets, and event counts remain intact.

The unmodified trigger comparator requires identical source provenance.
`v2-*-raw.json` preserves that original verdict. The companion report
records the only allowed substitutions: clean, independently pinned source
snapshots and the executable path with identical executable SHA-256. Bank
and frame hashes must match, and all other analysis options and numerical
values are unchanged. The offline verifier reproduces both reports.

`source.tar.gz` is the complete tracked production source. Its decompressed
SHA-256 equals the original `source.tar` pin. The redundant uncompressed tar
and Git bundle listed in `source-pins.json` are omitted from this archive;
the verifier needs neither. Reference receipts and scientific arrays are
included under `references/`. Original absolute paths are preserved as
provenance, while the verifier reads archived files by relative location.
Raw gravitational-wave frame/bank files and platform-specific native
extensions are identified by hash in `input-pins.json` and
`native-extension-pins.json`; replaying the native workloads additionally
requires those inputs and a compatible Linux environment.

## Execution and retained harness history

Host `len`; remote directory
`/home/xangma/pycbc-torch-residual-production-20260908/native-v1`.
Native controller PID/PGID 1555597; final scientific controller PID/PGID
2066648. Both finished successfully. Controllers held the shared benchmark
flock; workers inherited it, ran with affinity CPU 8 (SMT siblings 8,72),
and used one thread in each observed native thread pool. Runtime receipts
pin source imports and extension hashes. Source and input hashes were
unchanged before and after the runs. Machine and library versions are in
`machine.json` and the runtime receipts.

The sole recorded Python import outside the source checkout is the generated
`pycbc.version` metadata module. Its exact original path, 850-byte size, and
SHA-256 are explicitly pinned by the verifier; its bytes are included as
`generated-version.py`. Every other recorded PyCBC module must match the
tracked source or native extension pins. The verifier executes comparator
and runtime-checker code directly from manifest-verified source bytes and
refuses Python optimization modes that would remove validation assertions.

The final controller is `science_controller_v3.py`, using
`qualify_production_graph_v2.py`. All earlier scripts and unsuccessful
controller attempts remain in the archive:

1. The first controller completed standard CPU, then rejected its exact
   scientific match solely because source provenance differed. The final
   controller reused that completed run after checking all recorded hashes.
2. Review found the first, unused graph qualifier ran its eager oracle
   before production. Version 2 fixed the ordering and passed a suppressed
   replay negative control. The second controller incorrectly searched
   stderr for that assertion, although the qualifier stored it in JSON.
3. Version 3 validated the pinned negative run's actual JSON error and
   counters, then completed both Torch positive runs. The original false
   reporting verdict is preserved in `negative-result.json`; the validated
   verdict is `negative-result-v3.json`. No negative workload was repeated.

The final qualification contains one completed positive run per scheme.
This is correctness and lifecycle evidence, with no new timing campaign.
Earlier prototype speed measurements must not be assigned to these guarded
production implementations. CPU-only MKL reuse and opt-in offline CUDA
graphs retain their bounded applicability; this evidence does not qualify
additional transforms, devices, or workloads.
