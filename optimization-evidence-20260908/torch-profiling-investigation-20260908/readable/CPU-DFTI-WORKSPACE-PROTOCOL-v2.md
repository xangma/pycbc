# CPU DFTI workspace policy protocol v2

Status: bounded diagnostic specification, 2026-09-08; v2 supersedes v1 only for the owner-selected artifact root and exactly balanced within-worker timing order; freeze implementation and accept this schedule before native execution. The optimizer owns all `len` operations and the benchmark lock. The profiling peer reviews local artifacts. Use a new `torch-cpu-workspace-policy-20260908` artifact root; preserve every sealed earlier campaign. This experiment does not authorize source integration or publication.

## Hypothesis and fixed computation

Test whether requesting `DFTI_WORKSPACE=DFTI_AVOID` improves the existing 2**21 complex128 in-place CPU IFFT. Intel documents default `DFTI_ALLOW` and best-effort `DFTI_AVOID`. A reported value of AVOID proves configuration, not elimination of scratch storage or selection of a different algorithm. Any algorithm change may change rounding, so precision alone is insufficient evidence of parity. [Intel workspace documentation](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-1/dfti-workspace.html).

Targeted prior-source/artifact searches found no previous use beyond constant definitions. Earlier native/complete-call attribution, 28.861/33.409 ms, motivates the test only: it used inter-op 64. Measure fresh baselines with intra-op/inter-op/native pools all 1 and affinity `[8]`.

Use clean source `ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f`, torchfft.py SHA256 `0b9109bbf99a4eb591a4c006b1220ec36c8587e56352352277eb1478cf876c0c`, mkl.py SHA256 `8362642e359f48f17a89e0877757a29124a3c2b231e4debfa77d5520c2bb9fc3`. Preserve original engine/class/constructor, `torch.empty` allocation, complex64 public buffers, private complex128 in-place buffer, placement/scaling/thread limit, two-argument native execution, Tensor.copy_ conversions, eligibility/version/status behavior and finalizer. Rejected precision, NumPy-copy and page-backing routes stay closed.

## Arms and construction seam

| Arm | Workspace setting | Purpose |
| --- | --- | --- |
| A | Untouched default ALLOW (51); no workspace SetValue | Current baseline |
| B | One checked SetValue(WORKSPACE=17, AVOID=52) immediately before original commit | Candidate |

Use the same narrow library/commit proxy in both arms, with getters confined to construction or untimed boundaries. Pass the proxy through the target plan factory; call the exact original constructor on the original class. Intercept only the intended commit. Restore temporary bindings in `finally`. Do not mutate shared ctypes function signatures for the new SetValue/GetValue calls: bind private callables from the actual loaded library address and retain the library owner. Validate their enum/output-pointer and MKL_LONG return types against the installed header/ABI, recording that evidence. DftiGetValue returns a type determined by the queried parameter. [Intel DftiGetValue](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/dftigetvalue.html).

Require a native getter to observe 51 before the optional setter, then 51/52 immediately before commit and after successful commit. Every status must pass the unchanged status checker. Record descriptor identity, call order and counts; exactly one commit per plan, zero/one workspace setters for A/B. Query again before/after timing outside timers. A failed getter/setter/commit or unexpected value stops; no silent fallback. Add an untimed unproxied production-plan control to establish A's unchanged results and settings. Do not time an extra arm.

The proxy must accept only the frozen target geometry and must restore the original factory by identity on success and every failure. Prove original constructor, execute, can_execute and allocator identities, and absence of extra hot-path work. Each plan owns its usual distinct private allocation; record storage bindings/alignment and disjoint public/native buffers. No allocation advice or memory-policy changes. Report only a policy-setting effect unless separate evidence establishes actual scratch use.

## Contracts and native qualification

Reuse the existing fresh-public-buffer guard and lifecycle contracts. Cover object/storage replacement, shape/stride/dtype/device, overlap, autograd/inference/conjugate/negative state, thread count and owner PID without unsafe native dispatch. On success input bytes/version remain unchanged, public output increments once and private storage twice. Native status failure preserves public output while propagating the error. Verify descriptor free once, repeated cleanup, collection, and library/buffer lifetime. Inject failures for the initial getter, setter, precommit getter, commit, postcommit getter, original create/placement/thread-limit/finalizer paths; assert ownership cleanup and restored bindings. Include an existing unrelated plan whose ctypes signatures and execution remain intact.

First run real native contracts and the full 36-case gate under the owner lock in a qualification process. Preserve generator `qualify_runtime.py` SHA256 `e77603961c1f066647065e2e5c86dd2a4676fc891192bab508dc00d926888297`: seeds `(7,91,812,20260906)` × patterns `(dense,banded,impulse)` × scales `(1e-12,1,1e12)`, at 2**21, in that order. Require complete native complex128 bytes and final complex64 bytes equal between B, A and the unproxied reference on identical inputs. Signed zero counts; compare bytes, not approximate tensor equality. Snapshot native results before the next promotion. Keep matching-precision legacy MKL final-output parity and both L2/max-absolute errors bounded by the existing FFTW reference against promoted NumPy truth, without new tolerance floors. Verify input preservation and mutation counts. Any byte/error/contract mismatch rejects B and prevents timing.

## Fixed timing and decision

After the initial gate passes, run exactly three fresh serial workers. Each constructs its own A and B plans in order AB, BA, AB respectively, and qualifies those exact retained plans across all 36 cases before and after timing. Include an unproxied reference in these untimed gates. Do not replace plans after their gate.

Use four paired rounds per worker: worker 1 `AB,BA,BA,AB`; worker 2 `BA,AB,AB,BA`; worker 3 `AB,BA,BA,AB`. Each arm block has 3 warmups and 15 recorded complete-engine calls on seed 7 dense scale 1. Keep all raw samples. Compute an arm's worker median from its 60 recorded calls, and report each paired block median as well. Time the unchanged complete engine, including promotion, native execution, demotion, checks and mutation/status work. No getters, hashes, receipt writes or output snapshots inside the timer.

Run a separate native-attribution pass on the same plans, using complementary arm order in every paired round (swap AB with BA) and the same 3/15 warmup/sample counts. Repromote the fixed original input before every native call outside the native timer; never repeatedly transform the prior in-place output. Preserve the original two-argument call and report exactly which status/version work is included. Measure constructor/setup, first complete call and final cleanup separately. All settings/runtime/bindings and numerical gates must still pass afterward.

Advance only if B has lower complete-call and native worker medians than A in all three workers, with all qualification/runtime/ownership checks passing. Preserve every sample and pair, including contrary blocks. This bounded consistency rule is not a statistical significance claim. Non-improvement or mixed results stop without selective reruns. Contract/numerical failure rejects; runtime/provenance failure is inconclusive. No threshold or schedule changes after measurement.

Before launch freeze protocol/helper/harness/source/runtime/reference hashes, worker order and loop counts. Record actual imports/libraries, thread pools at configuration and worker boundaries, lock device/inode/inheritance, affinity/SMT sibling information and source state before/after. Retain command, host/cwd, PID/process group/start identity, logs, stop command, all exits/failure receipts and terminal group-absence/lock-reacquisition audit. Seal results and manifest for independent local review.

Only a passing microbenchmark warrants a separately frozen executable qualification using the accepted workload and unchanged scientific comparator gates, followed by balanced whole-process timings that include setup and cleanup. No CPU-parity, GPU, multicore or general-workload claim follows from this diagnostic.
