# CUDA host-read patch review, 8 September 2026

Reviewed candidate `7e56ac42417dd6f61498f917e04c5a55250ddc26` against parent `c659f4ed25c90f4f8b19237ce0d079cf3fe5e277` directly through the shared Git object database. It changes only `pycbc/filter/matchedfilter_torch.py` and the new `test/test_torch_cuda_peak_host_read.py`. The staged regression SHA256 is `962e64267af1ebcdfe3d7458f09d52660e5a60976bba177c0b20a2fdb654f2dd`.

No source-level blocker found. The helper retains the three CPU tensors produced by asynchronous transfers, completes the current stream for `values.device`, then exposes NumPy views. This establishes completed host storage at the public return boundary. The source GPU tensors and host destinations remain owned through the wait. Empty and abort returns, numerical peak selection, thresholds and native dispatch are unchanged. The optional live caller remains disabled by default.

The eight regression cases cover native/fallback execution, default/custom current stream and a delay before the first/last transfer. All three expected NumPy arrays and completion events are checked immediately after helper return, before the test's `finally` synchronization. This directly addresses the archived probe's weak oracle: that probe could return exit zero after observing three pending events and incorrect immediate arrays because it asserted only settled values.

Execution evidence remains necessary: run the same delayed-copy regression against the unchanged baseline and candidate, retaining baseline failures. Retain the existing peak, tie, NaN, empty and abort tests. This review makes no performance claim and does not authorize enabling the optional caller by default.
