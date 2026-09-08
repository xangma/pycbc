.. _torch-reference-campaign:
.. _torch-inspiral-reference:

Executable benchmark definition
===============================

This test runs ``pycbc_inspiral`` from process launch through completed HDF
output. It processes real H1 frame data with a fixed compressed low-mass bank,
including data conditioning, PSD estimation, waveform decompression,
normalization, scalar matched filtering, power chi-square, clustering and
trigger output. It does not call ``LiveBatchMatchedFilter.process_data``.
The measured results and revisions are in :ref:`torch-performance`.

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

The geometry was selected by a prior original-CPU sweep with repeated runs and
a declared rule, retained in the input archive. It is fixed for this backend
comparison. That finite grid and fixed end padding do not establish global
optimality or fresh boundary-injection validation for every template.

What is timed and checked
-------------------------

The latest matched source is clean
``ecd5d08231d8ce0a938bc31cfde27c9a5d6f901f``. The loader campaign also measures
its immediate baseline ``2f799f0046fc36db4215bd8b8b8a774d40c0e011`` on all
three backends. Only the candidate's nine workers contribute to the current
backend figure. This equal application of the loader avoids comparing an
optimized setup on one backend against the older setup on another.

All routes use runtime-verification wrappers with checks appropriate to the
selected backend. Full wall time starts immediately before worker launch and
stops at child exit. It includes verification, imports, frame reading, setup,
filtering and HDF output. Three fresh unprofiled processes per role run in
forward/reverse/forward role order. Report the median and observed range.
The six instrumented qualifications and controller comparisons are separate
from the eighteen timing workers. The earlier full-workload profiles in
:ref:`torch-profile-attribution` measure a different revision.

The qualifications verify compressed-template decompression without generation
fallback, all ``384 * 5 = 1920`` scalar IFFTs and complete valid-time coverage.
All 44 scientific comparisons pass: eight from qualification and 36 from
timing. They preserve all 1991 H1 trigger identities. Configuration and degrees
of freedom must match exactly. The eleven compared fields include SNR, phase,
sigmasq and the available veto fields; a stored zero field does not imply that
its optional veto ran.

The frozen comparator uses relative tolerance ``1e-4`` and absolute tolerance
``1e-5``, sigmasq relative tolerance ``1e-5``, and circular phase absolute
tolerance ``1e-4`` radians. These budgets are unchanged. Explicit revision and
command-path substitutions allow the declared baseline/candidate comparison;
scientific fields are not normalized. Complete per-scheme PSD references stay
pinned, including bins outside the used filter slice. Failed earlier controller
assumptions remain in the evidence. This campaign does not independently
measure the compressed bank's waveform approximation error.

All runs use ``len`` (AMD Ryzen Threadripper PRO 3995WX), CPU 8 affinity, one
allocated host core, and one numerical-library thread. Torch routes set and
verify both intra/inter-op pools to one. CPU 8's SMT sibling is CPU 72; neither
is reserved. The shared lock serializes cooperating benchmark tasks, not other
users of the machine. CUDA also uses one RTX 4090.
Python 3.11.9, Torch 2.13.0+cu130 and NumPy 1.26.4 are pinned in the receipts.
These full-wall measurements cover a finite workload. Sustained and full-machine
capacity require the additional experiments in :ref:`torch-benchmark-protocol`.

Reproduction and archives
-------------------------

The latest loader campaign and plot sources are listed in
:ref:`torch-followup-evidence`. Its ``loader-v1/`` directory contains the frozen
protocol, source/input/runtime pins and acquisition helpers;
``loader-qualification-v1.tar`` and ``loader-timing-v1.tar`` retain the raw
receipts, logs, scientific outputs, comparisons and terminal audits.

The `earlier executable and profile archive
<https://github.com/xangma/pycbc/tree/a742e59004779b35e3caea1a088ad5854042b704/device-profile-20260907-r3>`_
restores ``pycbc-torch-profile-20260907-r3/config.json`` with every scientific argument and
thread setting, the exact source/native snapshot, frozen acquisition helpers,
expanded per-worker commands, input hashes, all 21 HDF outputs and original
comparison receipts. Its README gives checksum verification, omitted original
inputs and runtime requirements, safe restoration and offline replay commands.

To repeat acquisition, restore a clean built checkout at the measured revision
and the pinned input files, then relocate the archived configuration and
helpers to that checkout. Preserve scientific options,
thread limits and hashes; retain new receipts for every attempt. The runtime
adapter is part of the timed command and must be included for a like-for-like
comparison. Archive restoration and figure verification alone do not re-run
scientific filtering. :ref:`torch-followup-plot-reproduction` gives the separate
commands for regenerating the current figures from verified summaries.
