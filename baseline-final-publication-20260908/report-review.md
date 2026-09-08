# Final bounded report review — 8 September 2026

**Ready: no factual correction required in the reviewed reports or future RST template.** Reviewed current `README.md`, `REPRODUCE.md`, and all report/RST text in `render-results.py` against `verification.json`, with targeted checks of configuration, qualification/runtime receipts, machine observations, the acquisition timer, and supplemental source-proof binding. Only this review note was written.

- All 16 displayed repeat values, four medians, observed ranges and rates agree with the verified receipts. Work is `384 * 1904 = 731136` template-seconds; five 512-second segments give 1920 template/segment pairs. All 16 own-qualification trigger comparisons pass; the four qualification times are excluded. The rotating order is correct.
- Baseline/proposal counts are 1988/1991, with 1959 matched identities, 29 baseline-only and 32 proposal-only; two proposal-only identities require threshold review. Each baseline/proposal comparison has 1951 chi-square, 22 phase and 10 SNR violations, with no sigmasq or degrees-of-freedom violations.
- Proposed PSDs are byte-identical on `[15360:1048576]`, representing 30 Hz through the bin below Nyquist at `delta_f = 1/512` Hz. The reported 2375/3105 finite-bin violations are correct for each of the five segment PSD comparisons against proposed CPU; their last violating bin is 14321, below the filter slice. Full PSD failures remain authoritative. Baseline PSDs also fail within the used slice (7175 finite violations per segment); the reports correctly limit exact used-bin agreement to the proposed arms.
- All five comparison records confirm exact conditioned-strain digests/gating metadata and geometry. Recorded digests support strain equality; the reports correctly disclose that strain samples were not archived. Overall science remains FAIL and equivalent-output speedup eligibility false. The post-qualification timing amendment, shared-host limitations and lack of confidence/capacity claims are disclosed consistently.
- Source commits match `40e94792b3edf59f39b18b65102b28a4f74433a7` and `123e1fb3ef1b338cada636e71c3e9c7987002402`. Supplemental source-proof counts are 1078/1227 and its manifest SHA256 matches acquired `source-pins.json`. Resource receipts confirm CPU 8, sibling 72, one native/Torch numerical thread, RTX 4090, and the stated library versions. The Torch inventory reconciliation retains its retrospective-binary-hash limitation.
- The current README/REPRODUCE wording correctly places parent host sampling concurrently with the child. Parent source hashing and HDF comparisons are outside the measured child interval. The RST template makes no conflicting claim. No duplicate edit is needed.

Reviewed SHA256 identities:

| File | SHA256 |
| --- | --- |
| README.md | `b98d16aa0394b36b355fdbb7f49da863d42e3591af39d5a2139032e6b58d6c8a` |
| REPRODUCE.md | `6e9619958cfe4ad4fee24aa827b36fa159dada2db9b7ce96557f56494eef0b43` |
| render-results.py | `adc7b8100d8c28791210deefafade65415434ee0385f3a6eb194a83c00bdd59b` |
| verification.json | `0dc09137fdd73c381555d0edae1f3bbcf599f6d2103ae34c4653d6936bc1c3da` |

Renderer syntax parsed successfully; it was not executed. This review does not rerun acquisition, scientific verification, external input downloads or Git-source reconstruction. Final archive sealing, immutable-link resolution and Sphinx rendering remain the primary agent's pending publication checks.
