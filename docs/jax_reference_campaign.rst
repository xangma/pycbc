.. _jax-inspiral-reference:

Complete-executable offline benchmark definition (pycbc_inspiral)
==================================================================

This test runs ``pycbc_inspiral`` from process launch through completed HDF
output. It processes real H1 frame data across all pipeline stages:
data conditioning, PSD estimation, template bank preparation, normalization,
scalar matched filtering, power chi-square, clustering, and trigger output.
It does not call ``LiveBatchMatchedFilter.process_data``.
The required reference baseline is unchanged CPU
``40e94792b3edf59f39b18b65102b28a4f74433a7``. Compare it with candidate
normal CPU, JAX CPU and JAX CUDA as separate arms, using the same inputs
and scientific settings.

The benchmark protocol defines two complementary tracks:

1. **Track 1: Compressed Bank Campaign (Inline Linear Decompression)**:
   Evaluates filtering, data conditioning, and linear decompression from stored
   SVD coefficients without dynamic waveform synthesis overhead.
2. **Track 2: Dynamic Generation Campaign (On-Device Batch Generation with ``diffgw``)**:
   Evaluates dynamic on-the-fly waveform generation from physical parameters.
   CPU arms evaluate waveforms sequentially via LALSimulation; JAX CUDA
   evaluates batched waveforms directly on device via ``diffgw`` (or ``jaxwave``).

Track 1: Compressed bank settings
---------------------------------

The bank contains **384 distinct templates: 256 BNS and 128 NSBH**.
Masses are detector-frame solar masses; spins are dimensionless aligned
components. The count ratio is a benchmark choice, not a population weight.

.. list-table:: Parameter bounds
   :header-rows: 1

   * - Region
     - Mass 1
     - Mass 2
     - Spin 1z
     - Spin 2z
   * - BNS
     - 1.2--2
     - 1.2--2
     - -0.05--0.05
     - -0.05--0.05
   * - NSBH
     - 3--8
     - 1.2--2
     - -0.5--0.5
     - -0.05--0.05

Templates use ``IMRPhenomD`` at 30 Hz and 4096 Hz sampling, with
``--use-compressed-waveforms --waveform-decompression-method inline_linear``.
This deterministic set is not a coverage-qualified search bank; it includes
neither tidal physics nor disruption. The frozen compressed bank SHA256 is
``26050d48322a1d71092bb0e024e71a89ace56b3e7b1c5e4cf20c7b769213fb7f``.

The frame is ``H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf``, channel
``H1:LOSC-STRAIN``, SHA256
``580e238054474fd09be900c47217bbcd0497ab84d1756f886647e934352e4865``.
The requested GPS input interval is 1187007048--1187009080. The unique valid
trigger interval is **1187007160--1187009064: 1904 seconds**.

.. list-table:: Fixed search configuration
   :header-rows: 1

   * - Setting
     - Value
   * - FFT geometry
     - 512-second segment; 112-second start pad; 16-second end pad;
       2097152 samples; five segments per template
   * - Strain conditioning
     - 25 Hz high-pass; 8-second data pad; autogating threshold 100,
       cluster 5 s, width/taper 0.25 s, pad 16 s, one iteration
   * - PSD
     - Median estimate; 126 segments of 32 s with 16 s stride;
       16-second inverse spectrum support; Hann truncation of inverse ASD
   * - Trigger selection
     - SNR threshold 5.5; reweighted SNR threshold 5;
       symmetric clustering with a 1-second window
   * - Vetoes
     - 16-bin power chi-square; sine-Gaussian chi-square is not requested
   * - Backend controls
     - Standard CPU ``cpu:1`` uses explicit MKL;
       JAX CPU ``jax:cpu`` and JAX CUDA ``jax:cuda:0`` use native JAX FFTs (with cuFFT on NVIDIA GPU)

Following the geometry selection methodology in :ref:`jax-benchmark-protocol`,
the configuration was selected on unchanged CPU ``40e94792b3`` by
sweeping 256/512/1024 second segments and 96/112 second start padding, with
16 second end padding across three fresh unprofiled processes per candidate.
The rule selected the lowest median, treating settings within 3% as tied, then
preferring more start padding and a smaller FFT. Retain this selected
512/112/16 second geometry for all matched backend comparisons.

Scientific qualification
------------------------

Verify that every arm decompresses the compressed bank without waveform
regeneration, completes all ``384 * 5 = 1920`` template/segment pairs, and
covers the same unique valid interval. Compare trigger identities before
comparing matched fields. Configuration and degrees of freedom must match.

The frozen executable comparator uses relative tolerance ``1e-4`` and absolute
tolerance ``1e-5``, sigmasq relative tolerance ``1e-5``, and circular phase
absolute tolerance ``1e-4`` radians. Compare complete PSD references as well as
SNR, phase, sigmasq and the available veto fields. The full-PSD gate uses
relative tolerance ``1e-4`` with zero absolute floor. Retain all failed verdicts.
Exclude runtime timing metadata fields: ``H1/search/filter_rate_per_core``,
``H1/search/run_time``, ``H1/search/setup_time_fraction`` and
``H1/search/templates_per_core``.

Commands and reproducibility
----------------------------

Prepare the original CPU reference and the candidate source in separate
clean checkouts with matching dependencies. Preserve measured commits and
input hashes.

Execution arms
~~~~~~~~~~~~~~

Each campaign comparison evaluates arms under identical host isolation:

#. **Original Standard CPU (``original_cpu``)**: Baseline checkout ``40e94792b3``
   using ``--processing-scheme cpu:1`` with explicit MKL FFTs.
#. **Candidate Standard CPU (``branch_cpu``)**: Candidate branch checkout
   using ``--processing-scheme cpu:1`` with explicit MKL FFTs.
#. **Candidate Batched CPU (``branch_cpu_batched``)**: Candidate branch checkout
   using ``--processing-scheme cpu:1 --batch-size 128`` with explicit MKL FFTs.
#. **Candidate JAX CPU (``jax_cpu``)**: Candidate branch checkout using
   ``--processing-scheme jax:cpu`` with explicit MKL FFTs.
#. **Candidate JAX Batched CPU (``jax_cpu_batched``)**: Candidate branch checkout using
   ``--processing-scheme jax:cpu --batch-size 16``.
#. **Candidate JAX CUDA (``jax_cuda``)**: Candidate branch checkout using
   ``--processing-scheme jax:cuda:0`` on NVIDIA GPU with native JAX data
   conditioning and on-device linear spline decompression.
#. **Candidate JAX CUDA Batched (``jax_cuda_batched``)**: Candidate branch checkout using
   ``--processing-scheme jax:cuda:0 --batch-size 128`` with zero-transfer GPU-resident filtering
   and on-device Power :math:`\chi^2` vetoes.

Track 1 executable command template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run each arm inside a clean environment with pinned single-thread limits:

.. code-block:: console

   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
   export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1

   taskset -c 8 pycbc_inspiral \
     --verbose \
     --frame-files docs/_include/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf \
     --channel-name H1:LOSC-STRAIN \
     --gps-start-time 1187007048 \
     --gps-end-time 1187009080 \
     --trig-start-time 1187007160 \
     --trig-end-time 1187009064 \
     --sample-rate 4096 \
     --low-frequency-cutoff 30 \
     --strain-high-pass 25 \
     --pad-data 8 \
     --autogating-threshold 100 \
     --autogating-cluster 5 \
     --autogating-width 0.25 \
     --autogating-taper 0.25 \
     --autogating-pad 16 \
     --autogating-max-iterations 1 \
     --psd-estimation median \
     --psd-segment-length 32 \
     --psd-segment-stride 16 \
     --psd-num-segments 126 \
     --psd-inverse-length 16 \
     --invpsd-trunc-method hann \
     --invpsd-trunc-which-spectrum invasd \
     --approximant IMRPhenomD \
     --order -1 \
     --snr-threshold 5.5 \
     --newsnr-threshold 5 \
     --chisq-bins 16 \
     --cluster-window 1 \
     --cluster-function symmetric \
     --fft-backends mkl \
     --bank-file inputs/bank_384_compressed.hdf \
     --use-compressed-waveforms \
     --waveform-decompression-method inline_linear \
     --segment-length 512 \
     --segment-start-pad 112 \
     --segment-end-pad 16 \
     --processing-scheme <SCHEME> \
     --output triggers_track1_<SCHEME>.hdf

Track 2 executable commands (Dynamic Generation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   # Standard CPU baseline (cpu:1):
   taskset -c 8 pycbc_inspiral \
     --verbose \
     --frame-files docs/_include/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf \
     --channel-name H1:LOSC-STRAIN \
     --gps-start-time 1187007048 \
     --gps-end-time 1187009080 \
     --trig-start-time 1187007160 \
     --trig-end-time 1187009064 \
     --sample-rate 4096 \
     --low-frequency-cutoff 30 \
     --strain-high-pass 25 \
     --pad-data 8 \
     --autogating-threshold 100 \
     --autogating-cluster 5 \
     --autogating-width 0.25 \
     --autogating-taper 0.25 \
     --autogating-pad 16 \
     --autogating-max-iterations 1 \
     --psd-estimation median \
     --psd-segment-length 32 \
     --psd-segment-stride 16 \
     --psd-num-segments 126 \
     --psd-inverse-length 16 \
     --invpsd-trunc-method hann \
     --invpsd-trunc-which-spectrum invasd \
     --approximant TaylorF2 \
     --order 7 \
     --snr-threshold 5.5 \
     --newsnr-threshold 5 \
     --chisq-bins 16 \
     --cluster-window 1 \
     --cluster-function symmetric \
     --fft-backends mkl \
     --bank-file inputs/bank_512_taylorf2.hdf \
     --segment-length 512 \
     --segment-start-pad 112 \
     --segment-end-pad 16 \
     --processing-scheme cpu:1 \
     --output triggers_track2_cpu.hdf

   # JAX CPU DiffGW arm (jax_cpu_diffgw):
   taskset -c 8 pycbc_inspiral \
     --verbose \
     --frame-files docs/_include/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf \
     --channel-name H1:LOSC-STRAIN \
     --gps-start-time 1187007048 \
     --gps-end-time 1187009080 \
     --trig-start-time 1187007160 \
     --trig-end-time 1187009064 \
     --sample-rate 4096 \
     --low-frequency-cutoff 30 \
     --strain-high-pass 25 \
     --pad-data 8 \
     --autogating-threshold 100 \
     --autogating-cluster 5 \
     --autogating-width 0.25 \
     --autogating-taper 0.25 \
     --autogating-pad 16 \
     --autogating-max-iterations 1 \
     --psd-estimation median \
     --psd-segment-length 32 \
     --psd-segment-stride 16 \
     --psd-num-segments 126 \
     --psd-inverse-length 16 \
     --invpsd-trunc-method hann \
     --invpsd-trunc-which-spectrum invasd \
     --approximant TaylorF2 \
     --order 7 \
     --snr-threshold 5.5 \
     --newsnr-threshold 5 \
     --chisq-bins 16 \
     --cluster-window 1 \
     --cluster-function symmetric \
     --fft-backends mkl \
     --bank-file inputs/bank_512_taylorf2.hdf \
     --segment-length 512 \
     --segment-start-pad 112 \
     --segment-end-pad 16 \
     --processing-scheme jax:cpu \
     --batch-size 64 \
     --enable-diffgw \
     --output triggers_track2_jax_cpu_diffgw.hdf

   # JAX CUDA DiffGW arm (jax_cuda_diffgw):
   taskset -c 8 pycbc_inspiral \
     --verbose \
     --frame-files docs/_include/H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf \
     --channel-name H1:LOSC-STRAIN \
     --gps-start-time 1187007048 \
     --gps-end-time 1187009080 \
     --trig-start-time 1187007160 \
     --trig-end-time 1187009064 \
     --sample-rate 4096 \
     --low-frequency-cutoff 30 \
     --strain-high-pass 25 \
     --pad-data 8 \
     --autogating-threshold 100 \
     --autogating-cluster 5 \
     --autogating-width 0.25 \
     --autogating-taper 0.25 \
     --autogating-pad 16 \
     --autogating-max-iterations 1 \
     --psd-estimation median \
     --psd-segment-length 32 \
     --psd-segment-stride 16 \
     --psd-num-segments 126 \
     --psd-inverse-length 16 \
     --invpsd-trunc-method hann \
     --invpsd-trunc-which-spectrum invasd \
     --approximant TaylorF2 \
     --order 7 \
     --snr-threshold 5.5 \
     --newsnr-threshold 5 \
     --chisq-bins 16 \
     --cluster-window 1 \
     --cluster-function symmetric \
     --fft-backends mkl \
     --bank-file inputs/bank_512_taylorf2.hdf \
     --segment-length 512 \
     --segment-start-pad 112 \
     --segment-end-pad 16 \
     --processing-scheme jax:cuda:0 \
     --batch-size 64 \
     --enable-diffgw \
     --output triggers_track2_jax_cuda_diffgw.hdf

Six-phase workload decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following the protocol in :ref:`jax-benchmark-protocol`, full execution wall time
is decomposed into six mutually exclusive phases with explicit JAX device synchronization:

#. **Process & Import Overhead**: Process spawn through argument parsing and
   module imports.
#. **Data Conditioning**: Frame reading, high-pass filtering, autogating, and
   126-segment Welch PSD estimation.
#. **Waveform Bank Preparation**: Decompressing stored linear coefficients
   (Track 1) or evaluating on-device batched ``diffgw`` waveforms vs sequential
   LAL CPU generation (Track 2).
#. **Core Matched Filtering**: Template Fourier transforms, frequency-domain
   whitening, correlation, and inverse FFTs (measured by the internal timer
   ``calc_time``).
#. **Vetoes & Clustering**: 16-bin Power :math:`\chi^2` calculation, SNR thresholding,
   and symmetric clustering.
#. **Serialization & I/O**: HDF5 trigger dataset creation, metadata formatting,
   and filesystem flushing.

Unprofiled timing runs (three counterbalanced fresh processes per arm) establish
certified wall-time and ``calc_time`` medians. Attribution percentages are
extracted from separate profiling passes without contaminating unprofiled samples.
For an exact breakdown of which pipeline phases are captured or excluded by each timer,
see :ref:`jax-timing-boundaries`.

Measured campaign results & scaling evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Certified multi-arm campaign measurements, operational throughputs, speedup factors,
and scientific parity validation results for the complete Track 1 offline inspiral search
are published centrally in :ref:`jax-performance` (see :ref:`Table 3: Complete Executable Offline Campaign <jax-performance>`).
