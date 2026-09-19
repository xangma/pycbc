.. _jax-batch-numerics:

Streaming batch benchmark definition and accuracy (pycbc_live)
===============================================================

This benchmark evaluates the public :class:`pycbc.filter.matchedfilter.LiveBatchMatchedFilter.process_data`
API used in the low-latency streaming pipeline (``pycbc_live``) with synthetic frequency-domain
inputs. It is a separate experiment from the complete-executable offline search benchmark in
:ref:`jax-reference-campaign`. Use the same frozen fixture for the original CPU reference
and each candidate backend; record the source revisions and qualify every source/backend
combination before collecting timings.

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

Execution batches are 1, 8, 32, 128, 512 and 1024 on standard CPU/MKL, JAX CPU
and JAX CUDA. Run workers serially on the same recorded host, with one
allocated host core and numerical thread; CUDA additionally uses one recorded
GPU. Rotate route and batch order between repeats. Each worker performs one cold iteration, two warmups and five timed iterations. One
iteration calls the prepared API for all three blocks, completing 3072
template-block evaluations. JAX device execution synchronizes via
``.block_until_ready()`` at both timing boundaries.
Input creation, PSD estimation, waveform/bank preparation, executable startup
and trigger validation are outside the clock. The reported cell rate is the
median of three fresh-worker median rates; ranges span those three medians.

Numerical standards and accuracy oracle
---------------------------------------

Higher precision reduces rounding error but does not guarantee bitwise agreement
between implementations. FFT algorithms, reduction order and device arithmetic
can change the result at either precision. Report exact comparisons separately
from tolerance-based comparisons and apply the declared criteria below to the
stored inputs and observed outputs.

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
time series.

The oracle promotes stored complex64 template and overwhitened strain inputs
before multiplication and applies an unnormalized NumPy complex128 inverse FFT.
Its normalization is computed independently from the bank and PSD in float64:

.. code-block:: text

   oracle_sigmasq = 4 * delta_f * sum((real64(bank)**2 + imag64(bank)**2) / PSD64)
   oracle_norm = 4 * delta_f / sqrt(oracle_sigmasq)
   oracle_complex_SNR = oracle_raw_output * oracle_norm

Actual and MKL outputs use their observed route normalizations. Qualification
records the norms and sigmasqs used by the default bulk path for every row.
Every observed sigmasq must agree with the independent value within relative
``1e-3``. Norms and sigmasqs must be finite and positive, and all complex samples
finite.

Existing trigger and veto comparisons remain unchanged: exact identities,
peak indices, end times and metadata; SNR absolute ``1e-3``; circular phase
absolute ``1e-3`` radians; sigmasq relative ``1e-3``; power chi-square relative
and absolute ``1e-4``; and sine-Gaussian chi-square relative and absolute
``3e-5``. Small complex error alone cannot guarantee unchanged threshold
decisions or peak selection near a tie, so the exact trigger gates still apply.

Historical benchmark harness
----------------------------

The one-off streaming benchmark harness is retained in Git history at
``jax-evidence-archive-20260919``. It is no longer distributed with the maintained reproduction
tools. The workload and numerical gates above describe that historical
experiment; use :doc:`jax_reference_campaign` for the maintained executable
benchmark workflow.
