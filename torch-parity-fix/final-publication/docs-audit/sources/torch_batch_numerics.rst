.. _torch-batch-numerics:

Live-filter benchmark definition and accuracy
=============================================

This method compares the public ``LiveBatchMatchedFilter.process_data`` API
using synthetic frequency-domain inputs. It is a separate experiment from the
complete-search comparison in :ref:`torch-performance`. Apply the same frozen
fixture to unchanged CPU ``40e94792b3edf59f39b18b65102b28a4f74433a7`` and all
restored backend arms; identify the source revisions in every result. Preserve
the original CPU behavior. The executable campaign does not qualify this
separate API experiment. This API benchmark has not been rerun for the Torch
compatibility implementation described in :ref:`torch-current-qualification`
and remains unqualified for that source. It was also not rerun for restored main
``aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0`` and remains unqualified for that
historical source. The `restoration evidence <https://github.com/xangma/pycbc/tree/31039e44d35ece9c6d755bd265c854d2bd8bb6a6/original-cpu-restoration>`_ concerns the historical executable
campaign; the archived development run below provides no current-source
API qualification.

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
and Torch CUDA. Run workers serially on the same recorded host, with one
allocated host core and numerical thread; CUDA additionally uses one recorded
GPU. Rotate route and batch order between repeats. Each worker performs one cold iteration, two warmups and five timed iterations. One
iteration calls the prepared API for all three blocks, completing 3072
template-block evaluations. CUDA synchronizes at both timing boundaries.
Input creation, PSD estimation, waveform/bank preparation, executable startup
and trigger validation are outside the clock. The reported cell rate is the
median of three fresh-worker median rates; ranges span those three medians.

The frozen fixture uses qualification seeds 7102 and 7103 and timing seed
7102. Qualify every source/backend/batch cell against both the independent
oracle and the unchanged CPU reference before timing. Include restored
normal CPU to check preservation separately from Torch agreement. A changed
CPU implementation may be studied only as a separately labeled experiment;
it cannot replace the required original reference.
Validate every timed call's triggers outside the clock.
This establishes finite warm-API behavior under the stated controls; it does
not establish executable or full-machine capacity.

.. _torch-batch-policy-v2:

Adopted complex-SNR criterion
-----------------------------

The fixture's normalized-SNR policy requires, for every block, template and
complex sample:

.. code-block:: text

   abs(actual_complex_SNR - oracle_complex_SNR) <= 0.001
   abs(actual_complex_SNR - MKL_complex_SNR) <= 0.001

Both comparisons are mandatory, with zero relative tolerance. The standard
CPU/MKL batch-1 reference must itself pass the independent oracle gate for
this API timing policy. If unchanged CPU fails, retain the failure and stop
qualification; do not change its arithmetic to make the oracle gate pass. The
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

Relative L2 and FFT-only residuals can be retained as separately named
diagnostics. No universal norm threshold is introduced. Normalization
observation and output capture run only during qualification and are restored
on exit; timing uses the uninstrumented public ``process_data`` method.

Reproducing the numerical method
--------------------------------

The `frozen fixture and qualification policy
<https://github.com/xangma/pycbc/tree/b5cf0acf0eeddf20e6ebd51eab19500d5a83c06b/current-batch-sweep-20260907-r4>`_
retain deterministic input construction, normalization observation, independent
oracle and trigger/veto comparisons. Restore them using the archive's
instructions, then run new acquisition in separate output directories against
the named source revisions. Preserve the original fixture and policy hashes;
record any harness adaptation and verify that it does not alter scientific
inputs, observed outputs or timing boundaries.

**SUPERSEDED for the restored stack:** the archived run qualified an earlier
development revision. Its rates and verdicts do not qualify restored source
or a CPU-reference arm that it did not execute. Source-native identities,
every per-worker receipt, all failures and the exact policy belong with each
new comparison. The archive
omits full scientific arrays; verifying recorded JSON alone does not replay
the scientific calculation.
