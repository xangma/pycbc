.. _torch-batch-numerics:

Live-filter benchmark definition and accuracy
=============================================

The completed batch sweep (R4, 7 September 2026) measures the public
``LiveBatchMatchedFilter.process_data`` API at revision
``9578a710479b924e882857c4dffab6ed372a634b``, before the squared-norm change.
:ref:`torch-batch-throughput` reports the rates. This test exercises prepared
filtering and vetoes using synthetic frequency-domain inputs.

Fixed workload and timing
-------------------------

Every full worker uses 1024 deterministic ``complex64`` templates and three
strain blocks. Templates contain seeded Gaussian complex values from 30 to
800 Hz, with unequal powers to exercise group normalization. They are synthetic
arrays, not IMRPhenomD waveforms. The fixture uses a smooth analytic PSD and
complex frequency-domain noise with four coherent injections per block at
nominal SNRs 8, 9.5, 11 and 12.5, then divides strain by that PSD.

The FFT has 131072 samples at 2048 Hz (64 seconds; ``delta_f = 0.015625`` Hz).
Each block advances 56 seconds and searches indices 12288 through 126975.
This differs from the executable's sample rate, FFT length and real frame data.
The fixture hashes the bank, parameters, PSD, normalization and all strain
blocks; changing batch size does not change those inputs within a seed.

The API performs correlation, IFFT, normalization, peak selection and vetoes.
SNR threshold is 5.5. It evaluates 16-bin reduced power chi-square and
sine-Gaussian chi-square with two configured tiles per trigger; activation and
valid tile support are checked. Reweighted-SNR and SNR-abort thresholds are
disabled. This tests their implemented behavior on the fixture, not detection
sensitivity across an astrophysical population.

Execution batches are 1, 8, 32, 128, 512 and 1024 on standard CPU/MKL, Torch CPU
and Torch CUDA. All workers run serially on shared ``len``, pinned to CPU 8
with one numerical thread; CUDA additionally uses one RTX 4090. Each worker
performs one cold iteration, two warmups and five timed iterations. One
iteration calls the prepared API for all three blocks, completing 3072
template-block evaluations. CUDA synchronizes at both timing boundaries.
Input creation, PSD estimation, waveform/bank preparation, executable startup
and trigger validation are outside the clock. The reported cell rate is the
median of three fresh-worker median rates; ranges span those three medians.

Numerical qualification uses two fresh seeds, 7102 and 7103. All 12 small smoke
and 36 full qualifications must pass before the 54 timing workers start.
Timing uses seed 7102 and validates every call's triggers outside the clock.
This establishes finite warm-API behavior under the stated controls; it does
not establish executable or full-machine capacity.

.. _torch-batch-policy-v2:

Adopted complex-SNR criterion
-----------------------------

Policy v2 requires, for every block, template and complex sample:

.. code-block:: text

   abs(actual_complex_SNR - oracle_complex_SNR) <= 0.001
   abs(actual_complex_SNR - MKL_complex_SNR) <= 0.001

Both comparisons are mandatory, with zero relative tolerance. The standard
CPU/MKL batch-1 reference must itself pass the independent oracle gate. The
``0.001`` budget extends the existing trigger-SNR tolerance to the full complex
time series: this is a new engineering acceptance criterion, not a pre-existing
repository guarantee or a proof of unchanged scientific decisions.

The oracle promotes stored complex64 template and overwhitened strain inputs
before multiplication and applies an unnormalized NumPy complex128 inverse FFT.
Its normalization is computed independently from the bank and PSD in float64:

.. code-block:: text

   oracle_sigmasq = 4 * delta_f * sum((real64(bank)**2 + imag64(bank)**2) / PSD64)
   oracle_norm = 4 * delta_f / sqrt(oracle_sigmasq)
   oracle_complex_SNR = oracle_raw_output * oracle_norm

Actual and MKL outputs use their observed route normalizations. Qualification
records the norms and sigmasqs used by the default bulk path, or the actual
scalar sigma callback, for every row; it verifies the original function, group
identity and executed branch. A shared fixture normalization is insufficient.
Every observed sigmasq must agree with the independent value within relative
``1e-3``. Norms and sigmasqs must be finite and positive, and all complex samples
finite. Missing or ambiguous observations, missing or duplicate row coverage,
or inconsistent source, policy, input or reference hashes fail qualification.
There are no percentile exceptions.

Existing trigger and veto comparisons remain unchanged: exact identities,
peak indices, end times and metadata; SNR absolute ``1e-3``; circular phase
absolute ``1e-3`` radians; sigmasq relative ``1e-3``; power chi-square relative
and absolute ``1e-4``; and sine-Gaussian chi-square relative and absolute
``3e-5``. Small complex error alone cannot guarantee unchanged threshold
decisions or peak selection near a tie, so the exact trigger gates still apply.

Raw v1 metrics remain separately named diagnostics alongside relative L2 and
FFT-only residuals. No universal norm threshold is introduced. Normalization
observation and output capture run only during qualification and are restored
on exit; timing uses the uninstrumented public ``process_data`` method.

.. _torch-batch-attribution-summary:

Policy provenance and evidence
------------------------------

The preceding raw-complex policy failed because FFT routes use different
arithmetic, with an additional CUDA correlation contribution. The standard
MKL reference itself exceeded that raw envelope against a complex128 oracle.
The `FFT investigation archive
<https://github.com/xangma/pycbc/tree/b26bbc1d612f92ad54377a2d687800ad207834d6/fft-numerics-20260907>`_
retains the failed R3 verdicts, selected-row replay and its limits. Policy v2
was adopted before R4 acquisition and changes the measured quantity to actual
normalized complex SNR; it does not turn previous failures into passes.

The `immutable R4 archive
<https://github.com/xangma/pycbc/tree/b5cf0acf0eeddf20e6ebd51eab19500d5a83c06b/current-batch-sweep-20260907-r4>`_
contains every worker receipt and result, frozen input construction and
qualification helpers, policy, source/native identities and post-run audit.
Its README provides lossless restoration and checksum verification. Full
``.npy`` arrays and native binaries are omitted; offline JSON verification
checks recorded results and file integrity without replaying scientific arrays.

.. _torch-batch-accuracy:

Completed qualification results
--------------------------------

The largest absolute complex-SNR discrepancy over both full seeds, all six
batches and all three routes is ``3.3868e-6`` against the independent oracle
and ``3.9178e-6`` against MKL. Both are below the adopted ``0.001`` budget.
The standard reference itself passes the oracle gate, with maximum discrepancy
``1.9161e-6``. Actual route normalizations are included in these comparisons.
All required trigger and veto comparisons pass.

.. figure:: images/torch-batch-r4-20260907/batch-accuracy.png
   :alt: Maximum absolute complex SNR errors for every route and batch size across both full seeds, compared with the independent oracle and MKL; all are below the 0.001 budget.
   :width: 100%

   Each point is the maximum over every sample, template and block of seeds
   7102 and 7103. The symlog axis is linear below ``1e-8`` and preserves exact
   zero for standard CPU/MKL compatibility. The dashed line marks the v2 budget.

:download:`Accuracy SVG <images/torch-batch-r4-20260907/batch-accuracy.svg>`;
:download:`plot input and image manifest <images/torch-batch-r4-20260907/manifest.json>`.

Offline plot reproduction
--------------------------

For a completed R4 evidence directory, the repository provides
``tools/plot_torch_batch_sweep.py``. Use Python 3.11 or later and Matplotlib;
local rendering tests used Matplotlib 3.11.1. Write plots outside the evidence
folder:

.. code-block:: bash

   python tools/plot_torch_batch_sweep.py --campaign /path/to/restored-r4 --output /path/to/new-plots
   python tools/plot_torch_batch_sweep.py --campaign /path/to/restored-r4 --output /path/to/new-plots --verify-only

The renderer checks the frozen helper and policy hashes, all 12 smoke and
36 full qualification results, all 54 timing results, and their acquisition
receipts. It recomputes the timing summary, checks that every qualification
finished before timing started, and rejects overlapping worker intervals.
It produces PNG/SVG throughput and accuracy figures plus a manifest of consumed
inputs, renderer and image hashes. Existing output directories are preserved.
Verification checks recorded JSON evidence; it does not replay the large
``.npy`` arrays or execute scientific filtering.

Throughput counts template-block evaluations per second: 1024 templates times
three blocks per iteration. Whiskers show the observed range of three fresh
worker medians, not confidence intervals. Accuracy takes the maximum over both
seed matrices, including actual route normalization. Its symlog axis is linear
below ``1e-8`` so exact-zero MKL compatibility remains visible alongside the
``0.001`` criterion.

The focused plot tests use the same frozen helper directory:

.. code-block:: bash

   PYCBC_BATCH_SWEEP_SCHEMA=/path/to/restored-r4 pytest -q test/test_plot_torch_batch_sweep.py

Fixtures exercise metadata rejection and rendering only; passing these tests
is separate from the measured scientific qualification.
