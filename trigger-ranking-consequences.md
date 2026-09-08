# Frozen CPU trigger ranking consequences

The frozen original/proposed outputs contain 1,988/1,991 triggers: 1,959 exact matches, 29 original-only and 32 proposed-only. Their top 100 overlap in 92 identities.

## Scope and method

Read-only analysis of `qual-original-cpu/triggers.hdf` and `qual-proposed-cpu/triggers.hdf` under `artifacts/torch-baseline-final-20260908/acquisition`. These are the existing thresholded and clustered H1 qualification outputs. They are not the newly split CPU branch's output. The original/proposed source commits and HDF/receipt hashes are recorded in the JSON.

Both receipts specify CPU:1, MKL, 4096 Hz, SNR >= 5.5, newSNR >= 5, and valid GPS interval [1187007160, 1187009064). Analysis options, consumed frame/bank hashes, gating and valid intervals agree. Each HDF matches its completed receipt and remains unchanged after analysis.

Match identity is `(H1, signed template_hash, GPS sample tick)` with unique keys and validated sample-grid times; there is no nearest-time substitution. JSON stores template hashes as decimal strings to preserve all 64 bits.

Stored `chisq_dof` is the bin count p=16, so physical DOF is 2p-2=30. Reconstruct r=chisq/30 and newSNR=snr when r<=1, otherwise snr*[0.5*(1+r^3)]^(-1/6). Arithmetic is float64 from the stored values. The NumPy CPU formula is unchanged between frozen sources; results match extracted original `newsnr` exactly, the float32 `get_newsnr` exactly after casting, and an independent scalar calculation within 2e-15 relative error. The HDF DOF conversion and >= selection are confirmed in frozen `pycbc/events/eventmgr.py` (original lines 296–305 and 549–552).

Ranks descend by reconstructed newSNR; ties use signed template hash then GPS tick. Both full-output ranks and ranks restricted to the 1,959 common identities are reported. A positive rank improvement is original rank minus proposed rank.

## Matched-event changes

| Metric | Result |
| --- | --- |
| Score increases / decreases / exactly unchanged | 990 / 968 / 1 |
| Signed newSNR delta: minimum / median / maximum | -0.36311969 / 6.1988831e-06 / 0.28476911 |
| Absolute newSNR delta: median / 95th percentile / maximum | 0.01854589 / 0.13491808 / 0.36311969 |
| Relative newSNR delta: minimum / maximum | -6.59541% / 5.46038% |
| Absolute relative delta: median / 95th percentile | 0.337014% / 2.44966% |
| Spearman / Kendall tau-b on common-event scores | 0.96633146 / 0.86606820 |
| Common-only absolute rank shift: median / 95th percentile / maximum | 29 / 341.1 / 797 |
| Full-output absolute rank shift: median / 95th percentile / maximum | 30 / 342.1 / 798 |
| Common-event ranks changed: common-only / full-output | 1892 / 1891 |

## Top 100

92 identities remain in both top-100 sets, with 8 exits and 8 entrants. Cutoff newSNR changes from 5.97924017 to 5.97225301. All these entrants/exits are present in both saved outputs; they reflect score ordering within the common events.

| Change | Template hash | GPS time | Original rank → proposed rank | Original newSNR → proposed newSNR |
| --- | --- | --- | --- | --- |
| exit | 1151654211523979201 | 1187008770.557128906 | 49 → 157 | 6.097340 → 5.894152 |
| exit | -968096554619096481 | 1187007905.083496094 | 72 → 166 | 6.046311 → 5.879500 |
| exit | -1731295501536661983 | 1187008356.706054688 | 78 → 138 | 6.027115 → 5.923056 |
| exit | -8703920638929076228 | 1187008368.705322266 | 88 → 113 | 6.003965 → 5.952979 |
| exit | 3965429967274226491 | 1187008044.427734375 | 89 → 112 | 6.002726 → 5.955373 |
| exit | -5477346001834174796 | 1187008066.740722656 | 92 → 129 | 5.993453 → 5.937409 |
| exit | 5461946332450954582 | 1187008818.789062500 | 98 → 149 | 5.983777 → 5.907309 |
| exit | 49837305221464261 | 1187008770.557128906 | 100 → 140 | 5.979240 → 5.918670 |
| entrant | 8392550104231525737 | 1187008369.856933594 | 101 → 48 | 5.975213 → 6.095589 |
| entrant | -8661556660204741982 | 1187008514.905273438 | 110 → 62 | 5.963752 → 6.066267 |
| entrant | -693574943151888723 | 1187008644.976562500 | 151 → 69 | 5.903843 → 6.048431 |
| entrant | -510003994118919637 | 1187008775.791259766 | 162 → 74 | 5.891183 → 6.031078 |
| entrant | -3586887046219042361 | 1187008270.291259766 | 215 → 83 | 5.836359 → 6.015773 |
| entrant | -3190827822836241740 | 1187007553.382080078 | 111 → 84 | 5.962308 → 6.014061 |
| entrant | 1786736971323122058 | 1187007198.704345703 | 132 → 96 | 5.935421 → 5.983538 |
| entrant | -1332058827206336129 | 1187008744.397216797 | 102 → 99 | 5.974416 → 5.974642 |

## Retention and membership

The observed saved-output union has 2,020 identities: 1,959 retained in both, 29 retained only by original, and 32 only by proposed. That is 61 changed memberships and a net +3, with 98.5412% of original identities also present in proposed. These are output-set transitions, not inferred crossings of a particular cut. The absent counterpart's SNR/newSNR is unavailable.

The table below applies post-hoc newSNR cuts to already saved triggers. Only 5.0 is the actual recorded newSNR cut; the remaining levels are descriptive probes. Up/down counts use only exact common identities. Saved counts additionally include that output's unmatched events.

| newSNR cut | Common retained: original / proposed | Common up / down | Saved retained: original / proposed |
| --- | --- | --- | --- |
| 5 (actual) | 1959 / 1959 | 0 / 0 | 1988 / 1991 |
| 5.1 | 1880 / 1887 | 36 / 29 | 1883 / 1892 |
| 5.25 | 1701 / 1693 | 40 / 48 | 1701 / 1694 |
| 5.5 | 1232 / 1231 | 69 / 70 | 1232 / 1232 |
| 5.75 | 356 / 351 | 20 / 25 | 356 / 351 |
| 6 | 89 / 88 | 7 / 8 | 89 / 88 |
| 6.25 | 17 / 19 | 2 / 0 | 17 / 19 |
| 6.5 | 4 / 4 | 0 / 0 | 4 / 4 |

Original-only reconstructed newSNR spans 5.01340753–5.19835085. All unmatched identities and their available scores are in the JSON; missing counterparts remain null.

Proposed-only reconstructed newSNR spans 5.00093044–5.50014067. All unmatched identities and their available scores are in the JSON; missing counterparts remain null.

## Numerical checks and limits

- Original: 0 exact score-tie groups; casting scores to float32 changes 0 full ranks, 0 top-100 memberships, and 0 decisions across the listed cuts. Maximum score-rounding difference is 2.38e-07.
- Proposed: 1 exact score-tie groups; casting scores to float32 changes 0 full ranks, 0 top-100 memberships, and 0 decisions across the listed cuts. Maximum score-rounding difference is 2.38e-07.

The HDF contains rounded output magnitudes, not every internal complex SNR or rejected candidate. Reconstructed scores are sufficient for this saved-output comparison but do not replay the complete threshold/clustering process. The common-event analysis conditions on surviving both pipelines. It cannot measure selection efficiency or attribute unmatched events to SNR, newSNR, or clustering. A high overall rank correlation does not establish unchanged tail membership.

No injection population, background population, coincidence statistic, FAR, or astrophysical sensitivity has been validated. This analysis does not establish ranking equivalence of the proposed CPU corrections or isolate the effects of individual corrections.

Reproduce from the repository with the existing offline environment:

```sh
/private/tmp/pycbc-cpu-precision-env-20260908/bin/python -B artifacts/torch-precision-validation-20260908/trigger-ranking-consequences.py
```

[Analysis script](trigger-ranking-consequences.py) · [Full JSON, event records and provenance](trigger-ranking-consequences.json)
