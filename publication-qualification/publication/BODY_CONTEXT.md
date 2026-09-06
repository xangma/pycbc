# Local PR body generation

After the four candidates, the final passing optimized report and all cited
evidence exist, the primary agent reviews a context JSON and runs from this
directory. Editing this generator does not generate bodies or publish results.

```sh
python3 -B write-bodies.py --context reviewed-body-context.json
```

Defaults read `prs-current.json`, `candidate-stack.json` and the repository's
`.github/PULL_REQUEST_TEMPLATE.md`. Optional `--snapshot`, `--stack`, `--template`
and `--out` select other paths. The default new output directory is `bodies/`;
an existing output is refused. Output contains `pr-N.md`, `pr-N.diff` for all
seven PRs and `manifest.json` with their hashes and unchanged title/base/branch/
draft metadata. No Git, network, PR, label or source-worktree actions occur.

## Required context

Use schema `torch-performance-fix-body-context-v1` with these fields:

- `reviewed`: Boolean `true` after the primary agent reviews the claims against
  the actual final evidence. The generator does not infer numerical claims.
- `snapshot_sha256`, `candidate_stack_sha256`: hashes of the two exact inputs.
- `source_identities`: object keyed by every full commit SHA used in a claim;
  each value has `tree` (full Git tree SHA) and `label` (e.g. the actual recorded
  campaign role). These identify measured/tested sources, which can differ from
  the final publication heads.
- `evidence`: list of distinct evidence entries described below.
- `primary_reference`: the six-record frozen baseline mapping described below.
- `optimized_reference`: the eleven-record final optimized mapping below. Both
  mappings are required, with distinct evidence IDs and explicit source identities.
- `prs`: object with exactly the keys `8`, `9`, `11`, `15`, `19`, `16`, `17`.
  Each value has `testing` (one to four claim objects) and optional `notes`
  (at most two claim objects).

Each evidence entry has `id`, `path`, `sha256`, `url`, `label`, `kind` and
optional `assertions`. Paths are absolute or relative to the context JSON.
URLs must be immutable `https://github.com/xangma/pycbc/{blob,tree}/FULL_SHA/...`
links. Allowed kinds are `report`, `tests`, `qualification`, `science` and
`artifact`. At least one of each non-artifact kind is required globally.

For report, test, qualification and science evidence, provide one or more JSON
assertions.
An assertion is `{"pointer": "/actual/field", "equals": ACTUAL_JSON_VALUE}`;
the pointer uses JSON Pointer escaping and can index arrays. Choose real fields
that qualify completion, source identity and the cited results. The generator
reads these JSON files and checks exact values/types and file hashes. Artifact
entries can reference other file types by hash without JSON assertions.

## Frozen baseline reference evidence

Use this mapping, replacing each descriptive evidence ID with its corresponding
entry in `evidence`:

```json
{
  "source_head": "837f38d493420043e45fb1ad210a0ccf68bacbaa",
  "source_manifest": "final-source-v5",
  "provenance": "final-source-provenance-v5",
  "report": "final-reference-report",
  "unit_tests": "final-unit-tests-v5",
  "waveform": "final-waveform-validation",
  "boundary": "final-boundary-validation"
}
```

The six IDs must be distinct. The files below are under the actual
`torch-inspiral-reference-20260906` artifact directory before archive staging;
use their immutable archive paths/URLs after staging. These are required
evidence identities, not a declaration that unfinished runs have passed.

| Mapping key | Kind | Actual file and required state |
|---|---|---|
| `source_manifest` | `qualification` | `source-v5.json`, with `commit` equal to the final source. |
| `provenance` | `qualification` | `source-precision-provenance-v5.json`, passing, with `new_source_commit` equal to the final source, normal CPU outputs changed and prior science not reused. |
| `report` | `report` | `final-report/report.json` from `build-reference-report-v4.py`, schema 1, passing with every gate true, `source_phases` exactly `{"precision": "837f38d493420043e45fb1ad210a0ccf68bacbaa"}`, final-source unit tests and passing parity. |
| `unit_tests` | `tests` | `unit-tests-v5.json`, complete, passing, return code zero, with identical clean final-source `source_info` and `source_after`. |
| `waveform` | `science` | `waveform-validation-precision5.json`, complete and passing on the unchanged clean final source, with all 288 expected template/PSD pairs completed. |
| `boundary` | `science` | `boundary-injections-precision5.json`, complete and passing on the unchanged clean final source, with all 36 expected cases completed. |

The generator checks these states and source identities in addition to each
entry's reviewed JSON assertions and hash. It also checks the report's provenance
fields for changed normal CPU outputs and no reused prior science. Include the
unchanged `unit-tests-v5.log` as hashed artifact evidence whenever citing its
terminal test summary; the final report binds that log and the test runner.
Use the actual summary and skips, not a predicted count. Review the source
manifest, provenance checks, final report input bindings, scientific budgets and
candidate qualification separately; this body writer does not rerun that work.

Across #15's testing claims and optional evidence notes, cite all six baseline
evidence IDs. The first claim now cites the optimized report, as specified below. This permits, for example,
one claim for reference tuning and matched capacity, one for independent
science and parity, one for source/unit/candidate qualification, and one for
separately measured profiles. Supporting component results can occupy an
evidence note. Do not fill these claims until the final report is complete.

## Optimized reference evidence

Provide `optimized_reference` with `source_head` equal to
`a4d77a6d1863c0515e8dace64c5609b63d40b51e`, `parent_source_head` equal to
`837f38d493420043e45fb1ad210a0ccf68bacbaa`, and these eleven distinct evidence IDs:

| Mapping key | Kind | File under `inspiral-reference-20260906` |
|---|---|---|
| `source_manifest` | `qualification` | `source-v6.json` |
| `report` | `report` | `optimized-report-v6/report.json` |
| `unit_tests` | `tests` | `unit-tests-v6.json` |
| `ifft_matrix` | `qualification` | `large-ifft-v6.json` |
| `ifft_decision` | `qualification` | `large-ifft-v6-decision.json` |
| `science` | `science` | `scientific-validation-v6.json` |
| `waveform` | `science` | `waveform-validation-v6.json` |
| `boundary` | `science` | `boundary-injections-v6.json` |
| `compressed` | `science` | `compressed-bank-v6.json` |
| `qualifications` | `qualification` | `campaign-v6-qualifications.status.json` |
| `measurements` | `qualification` | `campaign-v6-measurements.status.json` |

The report must pass all gates and bind the final source manifest, 19 campaign
cases, 18 strict trigger comparisons, three fresh unprofiled processes for each
backend, and seven separate profiles. Science must complete 288 waveform/PSD,
36 boundary and 576 compressed-bank parity cases. The 108-case inverse-FFT
matrix and its passing decision qualify only 1048576, 2097152 and 4194304
samples on one thread using double-precision MKL workspaces; preserve the
existing numerical budgets and all rejected alternatives in the evidence.
All receipts identify the clean optimized source before and after execution.
The completed 22-file unit run records 470 passed, 69 skipped and 8 subtests
passed; cite the exact unit log as an additional hashed artifact.

The first #15 testing claim must cite `optimized_reference.report`, identify
the optimized source, and lead with the fresh normal-MKL and matched Torch
results. Across its testing claims and notes, cite all eleven optimized and
six frozen baseline records. Use the baseline explicitly as the before source.
Four testing claims and two optional evidence notes provide room for timing,
science/precision, unit/candidate checks, profiles, and supporting evidence.

## Claims and source attribution

Each claim object is:

```json
{
  "text": "Reviewed prose with only results established by the cited evidence.",
  "source_heads": ["FULL_ACTUAL_TESTED_OR_MEASURED_COMMIT_SHA"],
  "evidence": ["EVIDENCE_ID"]
}
```

Every source must exist in `source_identities`; every evidence ID must exist in
`evidence`. The generator appends the full source IDs and immutable links to each
claim. Claim prose has a length limit and cannot introduce template headings or
replace the AI/Code of Conduct attribution. Evidence and input hashes are checked
again before the output directory becomes visible.

Retain the complete eight-commit scope in #15:

| Commit | Change |
|---|---|
| `450ab3f96ccea2783abb943b47f698578a507d59` | Sorted overlap checks, chunked CPU peak reduction and sparse float64 logarithmic phase terms. |
| `0d00581251e642a5d6b56b2497a9adad93069e6b` | Direct accelerator peak-reduction refinement. |
| `fb4b335eeeeaeaa907c1143b45e0191e2d977761` | Safe executable CPU-thread reporting. |
| `968bcd558117262af0d603710b054174659adb51` | Preserve shared Torch processing contexts during array deep copies. |
| `6c82155044d58f3344b281869d87745f71ba2285` | Spectral-power, PSD/strain and chi-square precision fixes with regression tests. |
| `f2c0abe61e787a26f41208f489c62c877bbd5667` | Cumulative-power test-reference and accumulated-rounding-bound correction. |
| `837f38d493420043e45fb1ad210a0ccf68bacbaa` | Preserve the CUDA phase coefficient at the Python launch site; extend the precision regression, leaving the kernel body unchanged. |
| `a4d77a6d1863c0515e8dace64c5609b63d40b51e` | Reuse compiled CPU compressed-waveform interpolation through shared views and add retained double-precision MKL workspaces for qualified single-thread large search inverse FFTs, with regression tests. |

The component v1 measurements remain at `450ab3`; the v2 accelerator follow-on
remains at `0d0058` with its explicitly reused baseline/control attribution.
Earlier executable attempts at `fb4b335`, `968bcd`, `6c8215` and `f2c0abe` are
supplemental, including rejected results. The frozen baseline reference and tuning identify `837f38d`. All optimized
scientific validation, matched timings and profiles identify `a4d77a6`; only the
verified unchanged normal-CPU tuning choice is reused. No old timing is pooled
with new repetitions or relabeled as a measurement of the optimized source.
Supply full SHAs and actual tree identities in `source_identities`; a later
archive commit in an evidence URL identifies storage, not the measured source.

## Content to review

- For #8/#9/#11, distinguish unchanged heads from later fixes/evidence in #15.
  Do not attribute new timing measurements to these original feature heads.
- For #15, lead with the final single-thread normal MKL `pycbc_inspiral`
  reference, its selected geometry and the matched normal CPU, Torch CPU and
  Torch CUDA results. Include actual unit/science/precision provenance and
  applicable candidate qualification. Keep component reports in their supporting
  role, with retained slower cases and reused controls. Never merge distinct
  campaigns or claim a test ran on a later publication head.
- For #19, cite structural/formatting qualification without assigning inherited
  batch fixes to its formatting-only feature diff.
- For #16, cite actual FFT qualification source/test receipts and feature
  preservation; keep historical FFT timing attribution distinct.
- For #17, include the optional CPU before/after qualification and preserved
  native peak gate/kernels; distinguish v1 CPU timing from v2 source checks.

Numerical prose must agree with the final report's measured scope and units:

- The workload contains 96 deterministic compressed BNS/NSBH point-particle,
  aligned-spin templates and 1904 unique valid detector seconds. It does not
  establish search-bank coverage, tidal/disruption accuracy or population
  throughput.
- Reference tuning compares 256, 512 and 1024 second FFT segments with the final
  112/16 second start/end padding. Use the report's actual selected geometry;
  do not assume a winner or claim a global optimum. PSD bins and triggers can
  change with FFT length, so parity comparisons apply within the same geometry.
- Matched schemes are `cpu:1`, `torch:cpu:1` and `torch:cuda:0`, each with one
  allocated host core; CUDA also consumes one GPU. Report the observed dispatch
  and exact CPU/GPU hardware from the evidence.
- Capacity is `96 * 1904 / full-process wall seconds / 1 allocated host core`,
  in templates per core at real time. Report three unprofiled repetitions by
  median and full observed minimum/maximum range. The finite interval does not
  establish steady-state throughput or confidence intervals. Executable internal
  time omits early startup and final HDF writing; distinguish it from full wall
  time and from setup/postsetup accounting.
- Profiles use separate executions. Keep cProfile exclusive seconds, native
  `perf` cycle sample percentages and CUDA event durations with their own
  denominators. Do not add them into one timeline or use them as capacity runs.
- Report validation counts, tolerances, observed errors, parity, test summaries,
  ratios and timing ranges only when established by the final evidence. Planned
  counts and structural source preservation are not completed measurements.

The generator preserves the snapshot's full stack index, source feature
descriptions (except #15's rewritten scope), historical immutable links,
unresolved CUDA copy-race paragraphs, relevant timing-fixture/MPS limitations,
and Additional notes with AI Gareth, `agent-assisted`, the unchecked Code of
Conduct box and exact `@xangma` note. It takes replacement heads/parents only
from `candidate-stack.json`. Review all seven resulting bodies and diffs before
the primary agent publishes them; the body manifest is not publication evidence.
