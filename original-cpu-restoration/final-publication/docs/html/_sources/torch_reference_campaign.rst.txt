.. _torch-reference-campaign:
.. _torch-inspiral-reference:

Executable benchmark definition
===============================

This test runs ``pycbc_inspiral`` from process launch through completed HDF
output. It processes real H1 frame data with a fixed compressed low-mass bank,
including data conditioning, PSD estimation, waveform decompression,
normalization, scalar matched filtering, power chi-square, clustering and
trigger output. It does not call ``LiveBatchMatchedFilter.process_data``.
The required reference is unchanged CPU
``40e94792b3edf59f39b18b65102b28a4f74433a7``. Compare it with restored normal
CPU, Torch CPU and Torch CUDA as separate arms; CPU arithmetic changes in
PR #20 are excluded. The replacement source pin is
``aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0``. New results and the explicitly
superseded historical campaigns are distinguished in :ref:`torch-performance`.

The completed restoration campaign used the frozen inputs and geometry
below. The `immutable restoration evidence <https://github.com/xangma/pycbc/tree/31039e44d35ece9c6d755bd265c854d2bd8bb6a6/original-cpu-restoration>`_ contains
their identities and workload checks. Original and restored normal CPU each
produced 1988 triggers with byte-identical scientific output. Both Torch
routes produced 1991 triggers and failed the unchanged strict comparisons;
no performance samples were collected.

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

The geometry was selected on historical unchanged CPU ``40e94792b3`` by
sweeping 256/512/1024 second segments and 96/112 second start padding, with
16 second end padding.
Each setting had three fresh processes. The rule selected the lowest median,
treating settings within 3% as tied, then preferring more start padding and a
smaller FFT. Retain the selected 512/112/16 second geometry for the
original-CPU/restored-stack comparison. The superseded corrected-CPU
campaign also used this geometry without re-tuning it on corrected CPU or Torch.
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
SNR, phase, sigmasq and the available veto fields. Retain all failed verdicts.
Any allowed source or executable-path substitution must be declared separately
from scientific fields; it must not alter the numerical tolerances or conceal
missing triggers. Runtime timing metadata is outside scientific comparison:
exclude only explicitly enumerated timing-field paths, retain their raw values,
and continue checking scientific metadata such as sampling, epoch and geometry.
The first restoration comparison mistakenly included four ``H1/search`` timing
fields; retain both its raw verdict and corrected scientific comparison as
described in :ref:`torch-performance`.

The completed comparison passes original CPU versus restored normal CPU:
all 18 H1 scientific datasets, full PSD arrays, conditioned strain and
geometry are byte-identical. PSD arrays are archived; conditioned-strain
identity uses full-data SHA256 and metadata, with no archived raw strain.
Both Torch routes fail against both CPU controls,
including trigger comparison and the full-PSD gate with relative tolerance
``1e-4`` and zero absolute floor. Conditioned strain and geometry remain exact
for both Torch routes. The evidence retains all five verdicts, arrays and
raw and corrected comparator outputs. The four excluded timing paths are
enumerated in :ref:`torch-performance`. Qualification stopped before any
performance sampling; historical timing permission does not apply.

Commands and reproducibility
----------------------------

The `restoration campaign archive <https://github.com/xangma/pycbc/tree/31039e44d35ece9c6d755bd265c854d2bd8bb6a6/original-cpu-restoration>`_ records all four
expanded source/backend commands, source/native identities, dependencies and
output hashes. Reproduce into a new acquisition directory.

**SUPERSEDED for the restored stack:** corrected-baseline `reproduction instructions
<https://github.com/xangma/pycbc/blob/e1dd5e7164a3e8ae8ee8b58ecd7b27200b8cfb9c/corrected-baseline-campaign/REPRODUCE.md>`_ and `configuration
<https://github.com/xangma/pycbc/blob/e1dd5e7164a3e8ae8ee8b58ecd7b27200b8cfb9c/corrected-baseline-campaign/acquisition/config.json>`_ retain expanded
commands, imported-source and native-build identity, input/output hashes,
exit status, the strict qualification stop and subsequent timing policy.
That historical campaign copied unchanged native binaries from the frozen
build after verifying source and binary hashes in both separate checkouts.
The archive includes the acquired ``sources.bundle`` and hash-verified frozen
bank. The bundle preserves both measured commits after published branches
advance; the reproduction instructions identify its required frozen base.
Those receipts reproduce historical source only; the bundle contains no new
restoration or qualification.

The **superseded combined-proposal** `historical comparison instructions
<https://github.com/xangma/pycbc/blob/bc88a36a225f9b89559e0480e66fac828ee3dd77/baseline-final-20260908/REPRODUCE.md>`_
and `historical configuration
<https://github.com/xangma/pycbc/blob/bc88a36a225f9b89559e0480e66fac828ee3dd77/baseline-final-20260908/acquisition/config.json>`_
describe only the earlier ``40e94792b3``/``123e1fb3ef`` run. The
`original CPU geometry sweep
<https://github.com/xangma/pycbc/blob/2fb788fde4c612a827e12b1be42559f408106bba/reference-campaign-20260907/REPRODUCE.md>`_
records how the frozen segment geometry was selected.

To reproduce the measured comparison, prepare unchanged ``40e94792b3`` and
``aa6b795a63bb18c4e63e4f4c203ca6e7c039d0f0`` in separate clean checkouts with
matching dependencies. Record later publication mapping separately, including
documentation changes and complete-module AST checks for declared formatting
changes. Behavior changes require new qualification; historical formatting
receipts cannot cover them.
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

The superseded corrected-CPU acquisition used shared ``len`` (AMD Ryzen
Threadripper PRO 3995WX), CPU 8 with SMT sibling CPU 72, and an RTX 4090 for CUDA. Affinity did
not reserve either CPU. Reproduction must record its own hardware, dependency
versions and load observations. The fixed workload measures finite-process
cost; sustained or full-machine capacity needs the additional experiments in
:ref:`torch-benchmark-protocol`. Restoring or verifying an archive does
not execute a new baseline/proposed comparison.
