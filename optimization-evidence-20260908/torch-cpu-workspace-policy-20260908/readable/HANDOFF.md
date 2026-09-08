# CPU workspace-policy diagnostic handoff

Outcome: **rejected before timing**. No source patch, integration request or performance claim.

Read [CPU-WORKSPACE-POLICY-v1.md](CPU-WORKSPACE-POLICY-v1.md). The exact first case passed final complex64 bytes and both unchanged FFTW error limits, but `DFTI_AVOID` changed the retained native complex128 bytes. The frozen protocol required both native and final byte parity. Native contracts passed 15 check groups and 14 controlled failures; the separate qualification stopped after its first three arm rows. Zero timing workers ran. No gate was relaxed and no measured rerun occurred.

Accepted source remains clean at `ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f`. CPU affinity was 8; Torch intra/inter-op and all four discovered native pools were one thread. The original constructor, storage allocation, guards and whole-engine execute stayed unchanged.

Sealed remote directory: `/home/xangma/pycbc-torch-cpu-workspace-policy-20260908/diagnostic-v1` on `len`. Controller PID/PGID 3912037, start ticks 65034982; native worker 3912170 and qualification worker 3914216 both exited zero. The explicit rejected result stopped the controller. Every retained plan/native owner was released; independent terminal closure confirmed the whole group absent, the shared lock available and all source/runtime/header/input pins unchanged.

| Artifact | SHA-256 |
|---|---|
| Frozen input manifest, 23 files | `c0cf59a906575a7d89d0be4b5d2f60ee054c8d8f7953b1cec279776e20518c06` |
| Input tar, 24 members | `559214db6f834043642e0d1acdd985aec65c3c765ff93608606032a8ba99734c` |
| Results manifest, 34 files | `04ff0360199bd176c2f58411abf17c106438a1e57f161d12a092286ec855a4a0` |
| Results tar, 35 regular members, 399,360 bytes | `05a828101cce79bcded64363d20f30270d407911da032c14fc4a338ae55ddb55` |

`audit_results.py` recomputes artifact, native receipt, first-case scientific, runtime, ordering and closure checks; its successful receipt is `owner-results-audit.json`. `acquired-diagnostic-v1` is the verified local extraction. Inputs in `diagnostic-v1` are frozen and match the archive exactly. Local fake, synthetic and mocked-child tests are separately identified in the input provenance.

Harness and terminal evidence peer review both passed. The independent receipt is `peer-results-review-v1.json`, SHA-256 `ddd9497d61b5c3700db0cb6e8cb972ac34744e71e68a46eee6ae4ac50c1ba5c5`; no findings remain. The outer `SHA256SUMS` covers the complete handoff.
