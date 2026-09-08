.. _torch-followup-evidence:

Executable optimization follow-ups
==================================

The latest matched three-backend campaign measures clean
``ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f``. Its results appear in
:ref:`torch-performance`. Later descriptor-reuse and CUDA-graph measurements
use that source plus pinned Python prototype helpers. They establish results
for those prototypes, not for a subsequently integrated implementation.

All executable comparisons use the same 384 compressed templates, five segments
per template and 1904 valid H1 seconds. They run serially on shared ``len``,
with CPU 8 affinity and one numerical thread; CUDA additionally uses one
RTX 4090. The host and its SMT sibling are unreserved. Full wall clocks cover
fresh-process startup, setup, filtering, output and shutdown. Qualification
and controller comparisons are separate. Ranges describe retained observations,
not confidence intervals or full-machine capacity.

Changes in the measured source
------------------------------

The qualified promoted CPU IFFT uses a private complex128 workspace in place,
preserving its complex64 input and converting the result back. Its separate
controlled executable comparison reduced Torch CPU median wall time by 6.16%.
The bounded frame loader skips redundant full-channel metadata discovery when
both bounds are already supplied; actual reads and validation remain. Applying
that change equally to all three backends produced this later comparison:

.. list-table:: Frame-loader campaign; full wall seconds
   :header-rows: 1

   * - Backend
     - Baseline median (range)
     - Candidate median (range)
     - Reduction of median wall time
   * - Standard CPU / MKL
     - 70.001 (69.822--70.571)
     - 66.933 (66.629--67.997)
     - 4.38%
   * - Torch CPU
     - 107.068 (107.062--107.645)
     - 104.400 (104.212--104.995)
     - 2.49%
   * - Torch CUDA
     - 24.786 (24.707--25.537)
     - 21.766 (21.727--22.124)
     - 12.19%

Baseline is ``2f799f0046fc36db4215bd8b8b8a774d40c0e011``; candidate is
``ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f``. Three workers per role run in
forward/reverse/forward order. The candidate's three backend rows form the
current headline comparison. The loader improves common setup, including
CUDA's host work; it is not evidence of a faster GPU kernel. Do not add its
percentage to the earlier CPU workspace result from another campaign.

The source also includes CUDA scalar-normalization scheduling. Its earlier
full-executable comparison showed overlapping ranges and no demonstrated gain
(24.659 versus 24.712 seconds). A separate synchronized warm sparse-API
diagnostic found a 5.76% reduction at 2M samples / 32 points, about 39
microseconds. That API result is not a live-batch or executable speedup.
The CUDA peak-transfer correctness fix is retained independently of timing.

R: descriptor reuse prototype
-----------------------------

R reuses eligible MKL descriptors across recurring function-level transforms.
The measured source is ``ecd5d082`` in all four roles: standard CPU with reuse
off/on and CUDA with reuse off/on. “Original” in its raw role names means reuse
disabled, not an older checkout. CUDA benefits through host-side transforms.
Torch CPU was not timed in R.

.. list-table:: R campaign; four workers per role
   :header-rows: 1

   * - Backend
     - No reuse median (range), seconds
     - Reuse median (range), seconds
     - Reduction of median wall time
   * - Standard CPU / MKL
     - 66.850 (66.792--67.039)
     - 65.370 (65.189--65.506)
     - 2.21%
   * - CUDA
     - 21.976 (21.894--22.052)
     - 20.292 (20.214--20.308)
     - 7.67%

Four role orders balance the repetitions: A-B-D-C, B-C-A-D, C-D-B-A, D-A-C-B,
where A/B are standard off/on and C/D are CUDA off/on. All samples are retained.
Median paired candidate-minus-baseline differences are -1.508019 seconds for
standard CPU and -1.684634 seconds for CUDA. These differ from subtracting the
two role medians. The percentages in the table use ``1 - median(on)/median(off)``.

All 51 scientific comparisons passed; 38 were exact across all 18 H1 science
datasets. The native gate checked full-output byte identity, input preservation,
and nine calls using three descriptors and six hits. Each cached process freed
its three descriptors once. Cache-hit counts were not recorded in the timing
workers. Qualification had already warmed filesystem and library state.

The measured ``descriptor_reuse.py`` SHA256 is
``626e0c123d5ba056475675f7134f83f44cc783b6da22065dc3fb7c383c0d0764``;
``run_reuse.py`` is
``6795f7a605b0525fe0ca2369e96f551b2fd5a019244b714cec3198c065099365``.
These are artifact helper identities, not source commits.

G: offline CUDA graph prototype
--------------------------------

G compares eager CUDA against graph capture/replay in a separate four-pair
campaign. **Both arms already use R's identical descriptor helper** on
``ecd5d082``. Orders are E1-G1, G2-E2, G3-E3, E4-G4. The graph clock includes
warmup, capture, synchronization, resets and cleanup.

.. list-table:: G campaign; four workers per role
   :header-rows: 1

   * - Role
     - Median seconds
     - Observed minimum--maximum seconds
   * - Eager + descriptor reuse
     - 20.257
     - 20.159--20.522
   * - Graph + descriptor reuse
     - 19.945
     - 19.933--19.995

Median wall time decreased by **1.54%**. All four paired differences favored
the graph, from 0.225909 to 0.526734 seconds. This result has its own eager
baseline; it is not a paired comparison with R's 20.292-second CUDA median.
No measured cumulative R-to-G speedup or updated three-backend comparison is
claimed.

All 25 scientific comparisons passed, 15 exactly across the 18 H1 science
datasets, with 1991 triggers throughout. The qualification checked 1920 graph
replays against eager oracles, five captures, full correlation/SNR arrays,
sparse outputs, input preservation, 1309 consumer index transitions and 14125
output-retention checks. Version 1 ran no timings. Version 2 corrected the
consumer-aware qualifier; the native helper, timed wrappers, source, science
comparator, workload, order and acceptance rule stayed byte-identical.

The measured ``offline_graph.py`` SHA256 is
``933e7024c4df4b09ebe9dd093ec99eb1726dcf13a564d6f923f70e25a26ff6de``;
``run_graph.py`` is
``5229f7e3d9d5370114a4ab504cca473016587f65abfdbf5f13ba6433ef25158e``.
These offline scalar-search results do not validate the existing live-batch
graph controls in :ref:`torch-optimizations`.

.. _torch-cpu-precision-cost:

CPU FFT precision and remaining cost
------------------------------------

The latest matched executable takes 104.400 seconds on Torch CPU and 66.933
seconds on standard CPU, a 37.467-second difference. The older native profiles
resolve standard FFT work to MKL ``32fc`` (complex64) and Torch FFT work to
``64fc`` (complex128). The qualified Torch route promotes the input, performs
the transform and converts back. Dtype, FFT implementation and algorithm can
change rounding as well as runtime. Intermediate byte identity, final-output
identity and scientific error tolerances are distinct checks.

A later controlled 2M-point IFFT diagnostic on ``ecd5d082`` measures 33.409 ms
through the engine. The staged path measures 28.861 ms inside the native DFTI
call, 4.492 ms for paired promotion and demotion, and 0.011 ms for version/status
bookkeeping. These are medians of worker medians; separate interval medians
need not sum. All 432 fixed precision cases pass the unchanged error and native
byte gates. Engine and direct-plan ranges overlap, so the result supports no
further wrapper speedup claim.

This establishes native FFT execution and precision conversion as the main
costs of that IFFT. It does **not** causally divide the complete 37.467-second
executable gap or prove that the native time is an unavoidable precision cost.
The microbenchmark records Torch inter-op at its default 64, with intra-op and
native pools at one; the executable explicitly sets inter-op to one. This is
not evidence of 64 active microbenchmark threads, but the settings must not be
described as identical. No timed overhead is subtracted.

Some tested direct complex64 alternatives fail the frozen scientific limits.
A retained out-of-place FFTW complex64 alternative passed its prescribed cases
but did not establish a competitive complete replacement. This does not imply
that every complex64 method is unsuitable or that CPU parity is impossible.
The full candidate inventory and limitations remain in the CPU assessment.

The later ``DFTI_WORKSPACE=DFTI_AVOID`` experiment is separate from the accepted
in-place CPU workspace change above. It stopped on its **first qualification
case**, because native complex128 bytes differed. On that same case, final
complex64 bytes matched exactly, input/version checks passed, and both L2 and
maximum-absolute errors passed their frozen limits. This was rejection under
the experiment's stricter intermediate-byte contract, not a demonstrated
failure of the final scientific output. The remaining cases were unrun and
**zero timing workers ran**. Construction and first-call durations provide no
performance comparison. The frozen rejection remains preserved.

.. _torch-followup-plot-reproduction:

Evidence and figure reproduction
--------------------------------

The publication archive is being prepared; its immutable URL will be supplied
before publication. **Archive URL pending: https://github.com/xangma/pycbc/tree/c4bfea522807742388dc8bcddc86473b9c03b047/optimization-evidence-20260908.**
The archive inventory covers ``torch-fft-optimization-20260908``,
``torch-residual-optimization-20260908``, ``torch-offline-cuda-graph-20260908``,
``torch-cpu-workspace-policy-20260908`` and the profiling investigation.
It preserves rejected candidates and superseded controllers alongside accepted
results. All campaigns are closed, with process-exit and independent lock
reacquisition records; the sealed acquisition bytes are unchanged.

Three byte-identical machine-readable summaries are committed for offline
plotting:

* :download:`Loader campaign <data/torch-20260908/loader-v1-summary.json>`:
  all six baseline/candidate roles and eighteen raw worker samples.
* :download:`Descriptor reuse R <data/torch-20260908/descriptor-reuse-timing-v1-summary.json>`:
  four roles, sixteen raw samples and paired comparisons.
* :download:`CUDA graph G <data/torch-20260908/graph-v2-timing-summary.json>`:
  four eager/graph pairs; the archive name is
  ``acquired-diagnostic-v2/timing-summary.json``.

Using Python 3.11 or later with Matplotlib, render into a new directory:

.. code-block:: console

   python tools/plot_torch_followups.py --output /path/to/new-plots
   python tools/plot_torch_followups.py --output /path/to/new-plots --verify-only

``--data`` can select another directory containing the same three named summary
files. The renderer requires fixed SHA256 pins and recomputes medians, ranges,
rates and paired differences. Its manifest pins the consumed bytes, calculated
values, renderer and PNG/SVG files. ``--verify-only`` uses the Python standard
library and does not render. These checks validate presentation; they do not
rerun benchmarks, repeat HDF comparisons or re-audit raw acquisition receipts.

:download:`Executable SVG <images/torch-executable-20260908/executable-wall.svg>`;
:download:`prototype follow-ups SVG <images/torch-executable-20260908/prototype-followups.svg>`;
:download:`figure manifest <images/torch-executable-20260908/manifest.json>`.
