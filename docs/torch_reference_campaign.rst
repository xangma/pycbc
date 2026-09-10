.. _torch-reference-campaign:
.. _torch-inspiral-reference:

Complete-executable offline benchmark definition (pycbc_inspiral)
================================================================

This test runs ``pycbc_inspiral`` from process launch through completed HDF
output. It processes real H1 frame data with a fixed compressed low-mass bank,
including data conditioning, PSD estimation, waveform decompression,
normalization, scalar matched filtering, power chi-square, clustering and
trigger output. It does not call ``LiveBatchMatchedFilter.process_data``.
The required reference is unchanged CPU
``40e94792b3edf59f39b18b65102b28a4f74433a7``. Compare it with candidate
normal CPU, Torch CPU and Torch CUDA as separate arms, using the same inputs
and scientific settings.

Inputs and scientific settings
------------------------------

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
<https://github.com/xangma/pycbc/tree/2fb788fde4c612a827e12b1be42559f408106bba/reference-campaign-20260907/inputs>`_.

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

The geometry was selected on unchanged CPU ``40e94792b3`` by
sweeping 256/512/1024 second segments and 96/112 second start padding, with
16 second end padding.
Each setting had three fresh processes. The rule selected the lowest median,
treating settings within 3% as tied, then preferring more start padding and a
smaller FFT. Retain the selected 512/112/16 second geometry for the
backend comparison.
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
<https://github.com/xangma/pycbc/tree/2fb788fde4c612a827e12b1be42559f408106bba/reference-campaign-20260907/inputs>`_
provides bank construction records; the `geometry sweep
<https://github.com/xangma/pycbc/blob/2fb788fde4c612a827e12b1be42559f408106bba/reference-campaign-20260907/REPRODUCE.md>`_
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

Record the hardware, dependency versions and load observations for every
acquisition. The fixed workload measures finite-process cost; sustained or
full-machine capacity needs the additional experiments in
:ref:`torch-benchmark-protocol`.
