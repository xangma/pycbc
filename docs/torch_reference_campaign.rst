.. _torch-reference-campaign:
.. _torch-inspiral-reference:

Complete-executable offline benchmark definition (pycbc_inspiral)
==================================================================

This test runs ``pycbc_inspiral`` from process launch through completed HDF
output. It processes real H1 frame data across all pipeline stages:
data conditioning, PSD estimation, template bank preparation, normalization,
scalar matched filtering, power chi-square, clustering, and trigger output.
It does not call ``LiveBatchMatchedFilter.process_data``.
The required reference baseline is unchanged CPU
``40e94792b3edf59f39b18b65102b28a4f74433a7``. Compare it with candidate
normal CPU, Torch CPU and Torch CUDA as separate arms, using the same inputs
and scientific settings.

The benchmark protocol defines two complementary tracks:

1. **Track 1: Compressed Bank Campaign (Inline Linear Decompression)**:
   Evaluates filtering, data conditioning, and linear decompression from stored
   SVD coefficients without dynamic waveform synthesis overhead.
2. **Track 2: Dynamic Generation Campaign (On-Device Batch Generation with ``diffgw``)**:
   Evaluates dynamic on-the-fly waveform generation from physical parameters.
   CPU arms evaluate waveforms sequentially via LALSimulation; Torch CUDA
   evaluates batched waveforms directly on device via ``diffgw``.

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
Its parameters and construction records are in the `input archive
<https://github.com/gwastro/pycbc/tree/2fb788fde4c612a827e12b1be42559f408106bba/reference-campaign-20260907/inputs>`_.

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
     - Standard CPU ``cpu:1`` and Torch CPU ``torch:cpu:1`` use explicit MKL;
       Torch CUDA ``torch:cuda:0`` uses one RTX 4090

Following the geometry selection methodology in :ref:`torch-benchmark-protocol`,
the configuration was selected on unchanged CPU ``40e94792b3`` by
sweeping 256/512/1024 second segments and 96/112 second start padding, with
16 second end padding across three fresh unprofiled processes per candidate.
The rule selected the lowest median, treating settings within 3% as tied, then
preferring more start padding and a smaller FFT. Retain this selected
512/112/16 second geometry for all matched backend comparisons.
That finite grid and fixed end padding do not establish global optimality or
fresh boundary-injection validation for every template.

Scientific qualification
------------------------

Verify that every arm decompresses the compressed bank without waveform
regeneration, completes all ``384 * 5 = 1920`` template/segment pairs, and
covers the same unique valid interval. Compare trigger identities before
comparing matched fields. Configuration and degrees of freedom must match;
a stored zero veto field does not prove that the optional veto ran.

The frozen executable comparator uses relative tolerance ``1e-4`` and absolute
tolerance ``1e-5``, sigmasq relative tolerance ``1e-5``, and circular phase
absolute tolerance ``1e-4`` radians. Compare complete PSD references as well as
SNR, phase, sigmasq and the available veto fields. The full-PSD gate uses
relative tolerance ``1e-4`` with zero absolute floor. Retain all failed verdicts.
Any allowed source or executable-path substitution must be declared separately
from scientific fields; it must not alter the numerical tolerances or conceal
missing triggers. Runtime timing metadata is outside scientific comparison:
exclude only explicitly enumerated timing-field paths, retain their raw values,
and continue checking scientific metadata such as sampling, epoch and geometry.
The excluded timing paths are ``H1/search/filter_rate_per_core``,
``H1/search/run_time``, ``H1/search/setup_time_fraction`` and
``H1/search/templates_per_core``.

Commands and reproducibility
----------------------------

Prepare the original CPU reference and the exact candidate source in separate
clean checkouts with matching dependencies. Preserve the measured commits
and record any later publication mapping separately.

The `input archive
<https://github.com/gwastro/pycbc/tree/2fb788fde4c612a827e12b1be42559f408106bba/reference-campaign-20260907/inputs>`_
provides bank construction records; the `geometry sweep
<https://github.com/gwastro/pycbc/blob/2fb788fde4c612a827e12b1be42559f408106bba/reference-campaign-20260907/REPRODUCE.md>`_
records the reference geometry selection.

Build native modules with recorded flags, or verify native-source identity
and copied binary hashes if reusing a frozen build. Restore the frozen bank
and frame by hash, relocate paths into new output directories, and preserve the
scientific arguments above. Verify that each interpreter imports its intended
checkout and that the requested processing scheme actually runs. Do not run
acquisition scripts inside an immutable archive or reuse existing worker output.

The normal CPU selectors are ``--processing-scheme cpu:1`` and
``--fft-backends mkl``; Torch CPU uses ``--processing-scheme torch:cpu:1``.
Torch CUDA uses ``--processing-scheme torch:cuda:0`` with one GPU and one host
thread. Explicitly set and verify Torch intra/inter-op pools and the numerical
library limits. Keep any required runtime adapter inside the timed command
and archive its source. Capture the complete expanded argv for all four arms,
including unchanged data, PSD, veto, clustering and output options.

Four-arm execution architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each campaign comparison evaluates four arms under identical host isolation
(single physical CPU core affinity via ``taskset``, one numerical-library thread):

#. **Original Standard CPU (``original_cpu``)**: Frozen baseline checkout
   ``40e94792b3`` using ``--processing-scheme cpu:1`` with explicit MKL FFTs.
#. **Candidate Standard CPU (``branch_cpu``)**: Candidate PR branch checkout
   using ``--processing-scheme cpu:1`` with explicit MKL FFTs, verifying CPU
   preservation before evaluating Torch routes.
#. **Candidate Torch CPU (``torch_cpu``)**: Candidate PR branch checkout
   using ``--processing-scheme torch:cpu:1`` with explicit MKL FFTs.
#. **Candidate Torch CUDA (``torch_cuda``)**: Candidate PR branch checkout
   using ``--processing-scheme torch:cuda:0`` on NVIDIA GeForce RTX 4090,
   with auto-enabled native GPU data conditioning and inline linear waveform
   decompression.

Track 1 executable command template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
     --use-compressed-waveforms \
     --waveform-decompression-method inline_linear \
     --snr-threshold 5.5 \
     --newsnr-threshold 5 \
     --chisq-bins 16 \
     --cluster-window 1 \
     --cluster-function symmetric \
     --fft-backends mkl \
     --bank-file inputs/bank-compressed.hdf \
     --segment-length 512 \
     --segment-start-pad 112 \
     --segment-end-pad 16 \
     --processing-scheme <SCHEME> \
     --output triggers_track1_<SCHEME>.hdf

Where ``<SCHEME>`` is ``cpu:1`` for the CPU arms, ``torch:cpu:1`` for Torch CPU,
or ``torch:cuda:0`` for Torch CUDA.

Track 2: Dynamic waveform generation campaign with diffgw
---------------------------------------------------------

Track 2 evaluates the complete executable when template banks are uncompressed,
requiring dynamic on-the-fly waveform generation:

* **Bank**: **512 distinct BNS templates** ($1.4 \le m_1 \le 2.5 \, M_\odot$,
  $1.2 \le m_2 \le 1.4 \, M_\odot$, non-spinning).
  Bank SHA256: ``b0dcadc61346fa76b26413ae2f3068dbaf1115fef722e0787abfceaaac072b03``.
* **Approximant**: ``TaylorF2`` at 3.5PN phase order (``--order 7``), 30 Hz cutoff,
  4096 Hz sample rate.
* **Waveform Generation**:
  - **CPU arms** (``original_cpu``, ``branch_cpu``, ``torch_cpu``): Sequential
    evaluation via LALSimulation (``XLALSimInspiralChooseFDWaveform``).
  - **Torch CUDA** (``torch_cuda``): On-device batched evaluation via ``diffgw``
    (with batch size 64).
* **Workload**: ``512 * 5 = 2560`` template/segment pairs.

Track 2 executable command templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
   export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1

   # CPU arms (original_cpu, branch_cpu, torch_cpu):
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
     --processing-scheme <SCHEME> \
     --output triggers_track2_<SCHEME>.hdf

   # Torch CUDA arm (torch_cuda with diffgw):
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
     --processing-scheme torch:cuda:0 \
     --batch-size 64 \
     --enable-diffgw \
     --output triggers_track2_torch_cuda.hdf

Six-phase workload decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following the protocol in :ref:`torch-benchmark-protocol`, full execution wall time
is decomposed into six mutually exclusive phases with explicit CUDA synchronization:

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

Record the hardware, dependency versions and load observations for every
acquisition. The fixed workload measures finite-process cost; sustained or
full-machine capacity needs the additional experiments in
:ref:`torch-benchmark-protocol`.
