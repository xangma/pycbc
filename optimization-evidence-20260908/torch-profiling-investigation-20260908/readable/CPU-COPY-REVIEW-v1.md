# CPU copy diagnostic review

Reject the NumPy copy candidate: it passed the numerical and native contract
checks but did not improve whole-call time in any of three fresh workers.

| Worker | Current engine (ms) | NumPy copies (ms) | Candidate slower |
| --- | ---: | ---: | ---: |
| cpu-copy-1 | 33.145242 | 33.183830 | 0.1164% |
| cpu-copy-2 | 33.450207 | 33.510964 | 0.1816% |
| cpu-copy-3 | 33.522067 | 33.558876 | 0.1098% |

Each value is the median of 60 whole-call samples per path within the same
worker, from four blocks with alternating order. Across workers the median
paired slowdown is 0.1164% (range 0.1098–0.1816%). These small differences
provide no evidence for adoption; no isolated conversion saving or executable
gain is claimed.

The independent review verified all 27 archived files against acquisition,
all 26 result-manifest entries and nine frozen input pins, source and native
identities, library versions, receipt controls, owner summary statistics,
four distinct worker processes, and controller termination/lock reacquisition.
All 14 native contracts ran in a separate fresh process before qualifier tensor
exports. All 432 precision cases passed, match the prior stage diagnostic,
and precede/follow the 360 total timed calls as specified.

The diagnostic ran on len, core 8, Torch intra-op 1 and native pools 1.
Observed Torch inter-op was 64, differing from executable controls of 1.
The 2**21 transform retained the qualified promoted in-place complex128 plan,
native kernel and two-argument execution ABI. Source remains unchanged.
The separate CPU-COPY-PROTOCOL-ERRATUM-v1.md corrects the matrix counts in
frozen prose. No integration, publication, or executable timing is warranted.

Evidence: cpu-copy-v1-results-peer-review.json binds the archive, inputs,
owner summary, contracts, per-worker samples and review script by SHA-256.
CPU-COPY-REVIEW-v1.json binds this report and erratum to that review.
