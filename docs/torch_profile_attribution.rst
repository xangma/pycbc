.. _torch-profile-attribution:

Current profiling and optimization evidence
===========================================

The latest profiles measure clean
``d2647addb884ead3249914ebc980f3c132076d93`` on 7 September 2026, including the
integrated CPU squared-norm change. Its ``pycbc/`` and ``bin/`` trees match
``9e6a688a5190d6e1ddc655fbe352cc206085d5c6``; later documentation changes are
outside the frozen acquisition. Every profile uses the complete 384-template,
1904-second workload defined in :ref:`torch-reference-campaign`.

Promoted MKL FFT work and conversion copies dominate Torch CPU. Squared norm
and trigger-result copying are much smaller targets. CUDA chi-square owns the
largest recorded device duration. These are instrumented observations; only
the separate nine unprofiled workers contribute to :ref:`torch-performance`.

Python and native call ownership
--------------------------------

Full-process cProfile includes imports, setup, frame I/O, filtering and output.
The separate filtering profile starts at the first ``FilterBank`` lookup and
ends after final event consolidation, before performance metadata and HDF
writing. CUDA synchronizes at both loop boundaries. Filtering profiles total
51.065916 seconds for standard CPU, 96.890385 for Torch CPU and 9.043454 for
CUDA. They are independent runs, not components of the unprofiled timers.

.. list-table:: Selected filtering-profile observations
   :header-rows: 1

   * - Operation and time basis
     - Standard CPU seconds
     - Torch CPU seconds
     - Calls
   * - FFT execution, self time
     - 28.4065
     - 62.5173
     - 1920
   * - Conversion copies inside FFT, self time
     - Not applicable
     - 8.6989
     - 3840 on Torch CPU
   * - Native point chi-square, self time
     - 7.8446
     - 7.8691
     - 1309
   * - Squared norm, cumulative time
     - Native work is not separately exposed
     - 1.4526
     - 765 on Torch CPU

The Torch FFT entry point totals 71.2541 cumulative seconds, **including** the
8.6989 copy seconds above. The source deliberately promotes complex64 input
into complex128 FFT workspaces and copies the result back. Removing these
passes would require a separate precision-preserving FFT change.

The full Torch CPU profile assigns 1.188421 cumulative seconds to 1309
``deepcopy`` calls from ``template_triggers``. Output snapshots carry array,
metadata, aliasing and mutation-independence contracts. This is an investigation
target, not a validated replacement. Recursive cumulative copy rows overlap
and must not be added. The first enclosing template call starts before filtering
profiling is enabled, so the full profile supplies this caller edge;
qualification receipts establish complete bank coverage.

Standard CPU ``scheme._scheming_function`` self time includes Cython work not
exposed as separate cProfile rows. It does not measure pure Python dispatch.
Torch squared-norm self time also includes native/operator work; its visible
``numel``, ``is_complex`` and ``is_conj`` calls total only about 0.0017 seconds
in the full profile.

Linux ``perf`` samples user cycles at 199 Hz with DWARF call stacks. Separate
full-process and exactly clipped filtering-window reports are retained.
Displayed filtering symbols attribute 57.62% of standard-CPU event periods to
MKL ``32fc`` FFT routines; Torch CPU attributes 66.04% to MKL ``64fc`` routines
and 9.50% to Torch copy kernels. These sums include only symbols above the
0.5% report threshold. They are rounded sampled-cycle shares, not wall-time
fractions. Both filtering reports record zero lost samples. Five symbolization
warnings in the full Torch CPU report remain in the archive.

CUDA ownership and synchronization
-----------------------------------

The instrumented CUDA loop lasts 10.690923 seconds. Its 162006 recorded device
events have 1.864559 seconds of union activity. Explicit external/correlation
IDs and containing semantic ranges assign 1.377272 seconds of kernel duration
to chi-square, 0.217452 to bank lookup, 0.101767 to IFFT and 0.016205 to squared
norm. Unattributed kernels retain 0.040779 seconds and unattributed device
copies 0.008542 seconds; the full category inventory is archived.

The 1309 host chi-square ranges total 3.249361 seconds, including 1955
``cudaStreamSynchronize`` calls totaling 0.993084 seconds. Host ranges and
device durations can overlap. Device-event sums are neither end-to-end runtime
nor GPU occupancy, and the difference from loop duration is not measured idle
time. Unmatched events remain unattributed.

Relating internal and full-process clocks
------------------------------------------

These values come from each backend's actual median-wall worker. Setup plus
the remainder of the internal timer plus the outside-clock residual equals
that worker's full wall time. The residual includes process/wrapper startup,
shutdown and other work outside PyCBC's internal clock; it is not pure startup.
Internal time after setup also differs from the separate filtering scope above.

.. list-table:: Accounting for the median-wall worker, in seconds
   :header-rows: 1

   * - Backend
     - Setup
     - Internal time after setup
     - Outside internal clock
     - Full wall
   * - Standard CPU / MKL
     - 15.323808
     - 50.261342
     - 4.634138
     - 70.219288
   * - Torch CPU
     - 15.143020
     - 94.537085
     - 4.483119
     - 114.163224
   * - Torch CUDA
     - 13.424875
     - 7.011554
     - 4.573399
     - 25.009828

In the CUDA median worker, setup and the outside-clock residual together take
71.96% of full wall time. This finite workload exposes material initialization
and setup costs; sustained capacity needs additional measurements under
:ref:`torch-benchmark-protocol`.

Validation and reproduction
----------------------------

All 21 science outputs pass the frozen trigger and numerical gates described in
:ref:`torch-reference-campaign`. All nine Linux lifecycle controls pass,
including actual ``perf``/``time``/``taskset`` inheritance of the shared lock
after supervisor exit. Runtime controls exercise 18 route/failure combinations;
all 15 frozen comparator fixtures pass. The final audit independently acquires
the lock after all 52 owned science/export process groups exit. Source, native
extension, input and helper pins remain fixed throughout acquisition and copy.

The `current executable and profiling evidence
<https://github.com/xangma/pycbc/tree/a742e59004779b35e3caea1a088ad5854042b704/device-profile-20260907-r3>`_
contains the source/native snapshot, six raw pstats files, two native profiles
with full/window reports, the CUDA trace, all 21 HDF outputs, helpers, receipts
and a checksum-verified restore and replay procedure. The unchanged original
archive SHA256 is
``a86204f5163c40f9ce011406cbc9f025209b48bbc9d95b18d0784bdce4772375``.

After restoring the publication, use Python 3.11 or later with Matplotlib to
regenerate the current executable figure from its pinned summaries:

.. code-block:: console

   python tools/plot_torch_executable.py --evidence /path/to/device-profile-20260907-r3 --output /path/to/new-plots
   python tools/plot_torch_executable.py --evidence /path/to/device-profile-20260907-r3 --output /path/to/new-plots --verify-only

The renderer recomputes the nine samples' medians, ranges and rates and checks
the recorded validation results. Its manifest pins inputs, renderer and PNG/SVG
files. This offline presentation check does not rerun the executable, repeat
HDF comparisons or replay raw profiles. Follow the archive's README for those
separate checks.

:download:`Executable SVG <images/torch-executable/executable-wall.svg>`;
:download:`figure manifest <images/torch-executable/manifest.json>`.

.. _torch-squared-norm-optimization:

Integrated Python optimization
-------------------------------

The measured implementation replaces a two-component squared-norm reduction
with separate real and imaginary squares and their sum, retaining the input's
real precision. It applies to ordinary, non-conjugated complex CPU tensors with
at least 4096 elements. The earlier comparison also observed a small public-API
penalty below that cutoff; the current backend campaign does not remeasure
individual API sizes. The `original optimization comparison and contracts
<https://github.com/xangma/pycbc/tree/cc282328975f6b8b7c224fc585e2a14b1b49d819/python-optimization-20260907-r4>`_
remain available in the immutable archive. Superseded figures and detailed
timing tables are omitted from active documentation.

Latest optimization pass
-------------------------

A second allocation experiment used the fresh real-square result as the
in-place addition destination. Its public-API measurements improved all 14
changed-path cases, while its three baseline and three candidate executable
workers had overlapping wall-time ranges and no demonstrated end-to-end gain.
Both qualifications and all ten scientific comparisons passed, with exact
trigger identities and numerical fields. The candidate remains isolated and
is **not integrated**; the measured source and current backend figure above
remain valid. The `complete second-pass evidence
<https://github.com/xangma/pycbc/tree/5ef0db0adea356b7a1657e92ab14e8fb11ad4761/python-optimization-pass2-20260907>`_
retains the candidate patch, every timing sample, contract checks, original
strict metadata failures and their bounded comparison receipts.
