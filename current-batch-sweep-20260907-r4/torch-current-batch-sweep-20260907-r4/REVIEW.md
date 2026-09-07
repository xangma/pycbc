# Complex-SNR qualification campaign

The user adopted the documented version-2 criterion on 2026-09-07. The campaign is being staged in a fresh remote directory; no version-2 result is claimed before acquisition.

The scientific definition and its provenance are in `NUMERICAL-POLICY.md`. The original R3 failures and the completed Linux precision diagnosis remain separate and unchanged. Source revision is pinned to `9578a710479b924e882857c4dffab6ed372a634b`.

`batch-worker.py` is a configurable single-cell CLI. It accepts source and output paths, route, batch size, bank size, FFT length, block count, seed, and timing repetitions. Its accepted batches and source revision are deliberately pinned for this experiment. It constructs an independent complex128 reference from the stored complex64 inputs and captures every route's actual normalization during qualification. Timing uses the uninstrumented public filtering API.

`batch-campaign.py` schedules the fixed experiment on len: 12 smoke cells, 36 full qualification cells using two fresh seeds, then 54 timing workers only after both seed matrices pass. The controller independently validates policy identity, oracle and output hashes, actual normalization, every row's scientific verdict, full coverage, and timing denominators. The controller retains len-specific paths and affinity CPU 8; adapting it for another host requires a new frozen campaign definition.

The controller requires an explicit `--snr-policy-v2` selection and a separate adopted policy decision file. The decision and exact policy-document hash are recorded in `policy-decision.json`. Native/source provenance and helper hashes will be frozen before acquisition in a new remote directory. The existing remote campaign directories will not be overwritten.

Local validation: 91 tests passed (37 controller, 54 worker), including hidden row failures, incorrect normalization and oracle identities, incomplete seed matrices, invalid timings, cancellation, and descendant cleanup. Three worker tests exercise the real public API: Torch CPU at batches 1 and 8, and the standard scalar path using the locally available FFTW backend. Those three integration tests were independently rerun and passed. Input generation and every existing trigger/veto comparison function are structurally unchanged from R3. F401/F821 checks passed for the controller; F401 checks passed for the worker.

The pinned source sets the bulk-peak crossover to zero, so both default Torch CPU integration cells observe actual bulk normalization arrays. Scalar callback observation is also covered by the standard CPU integration test. Profile hooks and callbacks are restored after successful runs and errors. These tests validate the harness; they do not constitute the new full-bank Linux MKL/CUDA qualification.

The preparation changed no runtime source or native extension. `launch-receipt.json` and `batch-status.json` record acquisition state once launched. Frozen inputs and receipts from earlier campaigns remain unchanged.
