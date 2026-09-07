Torch profile attribution: 7 September 2026
===========================================

The strongest measured Torch CPU target is the promoted MKL IFFT and its
conversion passes. A smaller, bounded target is the two-component reduction in
``array_torch.squared_norm``. Corrected normal CPU already spends most filtering
cycles in MKL FFT. The new CUDA device trace identifies pointwise chi-square
as the largest recorded kernel cost, with substantial host synchronization
inside the same call path. Instrumented durations are attribution evidence,
not throughput or GPU occupancy measurements.

Evidence and boundaries
-----------------------

The eight cProfile files and four native text reports are from the
`immutable 384-template campaign
<https://github.com/xangma/pycbc/tree/2fb788fde4c612a827e12b1be42559f408106bba/reference-campaign-20260907>`_.
Original CPU source is ``40e94792b3edf59f39b18b65102b28a4f74433a7``;
corrected CPU and Torch source is
``a4d77a6d1863c0515e8dace64c5609b63d40b51e``. All profile hashes match
``summary.json``. Strict original-versus-corrected science comparison fails;
Torch-versus-corrected and each backend's profile/repeat comparisons pass.
Original CPU is therefore a separate scientific baseline, not a parity control.

There are 384 distinct compressed IMRPhenomD templates (256 BNS, 128 NSBH),
1904 unique valid detector seconds, five segments per template, 4096 Hz sampling,
30 Hz cutoff, 512/112/16-second segment/start/end geometry, and 16-bin power
chi-square. This executable configuration does not request sine-Gaussian
chi-square. Every numerical thread limit is one; normal CPU explicitly uses
MKL. CPU 8 and SMT sibling 72 are unreserved on the shared 64-physical-core
``len``; CUDA also uses one RTX 4090. This is finite-workload evidence.

Profiling branch base is ``818331d6ec39ed6f9fb64abcb867a27ee9919785``.
Direct source comparison with the measured revision finds identical ASTs for
the changed FFT, decompression and TaylorF2 files; strain's only change moves an
import. This is a source inspection, not a fresh numerical or timing result at
the later commit. The completed R4 live-API sweep instead measures
``9578a710479b924e882857c4dffab6ed372a634b``. Its runtime changes from the
profiling branch base correct live-batch template-group normalization and
power-chi-square correlation geometry/cache validity. Those changes are
confined to ``LiveBatchMatchedFilter``; the scalar executable paths inspected
for this profile are unchanged. New acquisition uses the measured R4 source
directly, with external profiling helpers, and retains its own provenance.
The R4 source directory resolves to the preserved R3 source checkout at that
same clean revision; executable and import paths in run receipts therefore
contain ``r3/source``. This path alias does not change the failed R3 campaign's
scientific classification.

Startup and filtering
---------------------

Each row below uses the *same median-wall unprofiled process* for its wall
components. The last column comes from a separate observed process; it is not
subtracted from those components. Units are seconds.

.. list-table:: Full executable accounting and independent loop observation
   :header-rows: 1

   * - Backend
     - Full wall
     - Internal setup
     - Internal post-setup
     - Outside internal timer
     - Observed loop
   * - Original CPU
     - 66.503
     - 14.536
     - 49.136
     - 2.831
     - 49.413
   * - Corrected CPU
     - 70.032
     - 15.281
     - 50.231
     - 4.519
     - 50.094
   * - Torch CPU
     - 119.198
     - 15.022
     - 99.591
     - 4.586
     - 99.782
   * - Torch CUDA
     - 25.180
     - 13.408
     - 7.087
     - 4.686
     - 7.174

The outside-timer residual includes early startup and final output; it is not
an isolated import or Python-overhead measurement. The loop starts at the first
``FilterBank`` lookup and ends after event consolidation, before performance
metadata/output. CUDA synchronizes at both boundaries. Setup plus residual
takes 71.86% of CUDA wall time in this selected process. Full-process cProfile
charges roughly 6.7 seconds to frame APIs on each backend. MKL descriptor creation
costs 4.177 seconds on corrected CPU and 2.531 seconds on CUDA (1003 calls on
CUDA), outside the measured filter-loop profile. Planning is a setup target,
not evidence of repeated IFFT planning in the loop.

CPU filter-loop attribution
---------------------------

These are exclusive cProfile seconds unless explicitly marked cumulative.
Separate original/corrected/Torch CPU loop denominators are
48.374959/51.257855/101.804049 seconds; instrumented time is not throughput.

.. list-table:: Exact call ownership in the measured loop
   :header-rows: 1

   * - Operation
     - Corrected CPU
     - Torch CPU
     - Interpretation
   * - FFT entry point
     - 28.6493 s, 55.89%
     - 62.5831 s, 61.47%
     - 1920 IFFTs; native work is charged to Python/ctypes entry points
   * - Copies inside Torch FFT
     - Not separately measured
     - 8.6049 s; 3840 calls
     - Exactly two ``copy_`` calls per promoted IFFT
   * - Native point chi-square
     - 7.9159 s
     - 7.8448 s; 1309 calls
     - Same corrected native precision route
   * - Native threshold wrapper
     - 2.9192 s
     - 3.2322 s; 1920 calls
     - Torch already invokes the native CPU kernel
   * - Native decompression wrapper
     - 1.6841 s
     - 1.7043 s; 384 calls
     - Torch already invokes the native CPU kernel
   * - Correlation
     - 4.58% native cycles
     - 3.2472 s; 1920 ``torch.mul`` calls
     - CPU correlation is not separately exposed by the cProfile grouping
   * - Squared magnitude reduction
     - Not separately measured
     - 5.7998 s; 765 ``Tensor.sum`` calls
     - All charged directly under ``squared_norm``; square adds 0.5888 s

``_MKLCPUDirectIFFTPlan.execute`` has 71.2222 cumulative seconds, of which
62.5831 is exclusive and 8.6049 is tensor copying. These values overlap and
must not be summed. Total loop ``Tensor.copy_`` is 8.7810 seconds/4221 calls:
the remaining 0.1761 seconds/381 calls belongs to ``array_torch._copy``.
The FFT length is 2097152. The source deliberately promotes complex64 input to
complex128 retained input/output workspaces, executes double-precision MKL,
then copies to complex64 output. Native ``64fc`` FFT symbols confirm that
arithmetic route. Precision promotion cannot be removed as a routine copy fix.

The original summary's generic grouping places these Torch FFT, threshold and
decompression wrappers in "Other measured functions". That group is not Python
overhead. The call-level values above recover their actual ownership.

Native loop event-period shares independently support FFT dominance:

.. list-table:: Rounded ``cycles:u`` percentages; no children
   :header-rows: 1

   * - Backend
     - FFT
     - Chi-square
     - Threshold
     - Correlation
     - Decompression
   * - Original CPU
     - 59.94
     - 13.81
     - 5.67
     - 4.31
     - 3.19
   * - Corrected CPU
     - 57.99
     - 15.93
     - 5.78
     - 4.58
     - 3.24
   * - Torch CPU
     - 62.81
     - 7.96
     - 3.41
     - Not separately grouped
     - 1.51

These are sampled cycle-period shares, not seconds, device times, or normalized
sample counts. Preserve rounded and omitted-row remainder. No confidence
interval or expected optimization speedup is inferred from one profile.

CUDA: host calls, transfers and synchronization
-----------------------------------------------

The CUDA loop cProfile denominator is 9.123969 seconds. ``torch.as_tensor``
costs 1.493001 host seconds/14031 calls, including 1.107819 seconds/1309 calls
directly beneath ``power_chisq_at_points_from_precomputed``. The source converts
non-Torch SNR values to the correlation device here. That cost can include
conversion, allocation, transfers and waiting; its split requires device/runtime
events. The chi-square function is 2.4119 cumulative seconds, which already
includes that conversion and other descendants.

Other host observations: threshold is 1.2044 cumulative seconds/1920 calls;
the single-point chi-square bin loop makes 11296 ``torch.sum`` calls (0.1657
exclusive host seconds); ``Tensor.cpu`` takes 0.084828 seconds/5236 calls;
``Tensor.to`` takes 0.217481 seconds/23916 calls. Many ``to`` calls are dtype or
same-device operations. Their count is not a transfer count or transferred-byte
measurement. The 0.1040 seconds in 1920 ``fft_ifft`` calls is host API time,
not CUDA FFT duration. One final explicit synchronization takes 9 microseconds;
that says nothing about implicit synchronizations earlier in the loop.

Native CUDA-run sampling mostly sees Python, CUDA driver, allocation and hashing
activity. It does not sample GPU kernels. The following device measurements
come from a separate trace, not from those CPU samples.

CUDA device trace on the qualified R4 source
--------------------------------------------

The new 384-template acquisition uses PyTorch ``2.13.0+cu130``, CUDA 13.0 and
the RTX 4090. All nine Linux lifecycle controls passed before science runs.
CUDA qualification and unchanged trigger/veto comparisons passed: each run
has 1991 matched triggers and no unmatched triggers. Qualification and profiled
CUDA have zero differences in every compared field against unprofiled CUDA.
Against fresh CPU/MKL, CUDA's maximum absolute SNR error is 3.33786e-6 and
chi-square error is 9.15527e-5, both within the original comparator budgets.
The common Torch bootstrap makes the fresh CPU run a scientific reference;
its startup is not directly comparable to the older unwrapped CPU timings.

The trace covers all 384 ordered templates, 1920 IFFTs, 1920 correlations,
1920 thresholds, 1309 chi-square calls and 765 squared-magnitude calls. Actual
Torch intra/inter-op counts remain one and affinity remains CPU 8. Its host
annotation spans 10.577002 seconds; the adjacent monotonic boundary timers give
10.575218 seconds. Neither is an unprofiled throughput measurement.

There are 162006 recorded device activities, all inside the host loop:
121062 kernels, 39639 memcopies and 1305 memsets. Their summed duration and
interval union both equal 1.867244 seconds; no recorded intervals overlap in
this trace. GPU annotation ranges are excluded. The difference from host-loop
duration is not established device idle time.

.. list-table:: Device kernel attribution by correlated semantic range
   :header-rows: 1

   * - Range
     - Kernels
     - Summed device seconds
   * - Chi-square
     - 41727
     - 1.378349
   * - Bank lookup/materialization
     - 31104
     - 0.217412
   * - IFFT
     - 5760
     - 0.103141
   * - Correlation
     - 3840
     - 0.046434
   * - Unattributed
     - 21811
     - 0.040749
   * - Threshold
     - 14138
     - 0.026027
   * - Squared magnitude
     - 1530
     - 0.016280
   * - Normalization
     - 1152
     - 0.007456

The 603 ``_triton_pointwise_chisq_bin_kernel`` launches alone total 1.187486
device seconds, the largest named kernel cost. Bank lookup includes all work
inside template materialization; its kernel total is not an isolated waveform
decompression measurement. IFFTs produce three kernels per call here.

Memcopies total 0.030948 device seconds: 14418 host-to-device events carry
10673392 recorded bytes; 8689 device-to-host events carry 124332 bytes; 16532
device-to-device events carry 152108 bytes. Memsets add 0.000448 seconds.
These are recorded transfer payloads, not allocator traffic or a bandwidth
benchmark. Small transfers are numerous despite the modest total payload.

The inclusive host chi-square range totals 3.228059 seconds. Within it,
1955 ``cudaStreamSynchronize`` calls total 0.996559 seconds and 40418
``cudaLaunchKernel`` calls total 0.343245 seconds. Its 13251
``cudaMemcpyAsync`` calls take 0.151211 host seconds while their device
events take 0.011413 seconds. Host waits can overlap the kernel work above;
these durations must not be summed. The evidence prioritizes the pointwise
chi-square call path and its synchronization/dispatch pattern, rather than
assuming host conversion time measures transfer bandwidth.

External and runtime IDs agree for 121128 activities; another 4443 link through
runtime IDs and three through external IDs alone. There are no conflicting
links. The remaining 36432 activities retain no semantic attribution:
21811 kernels and 14621 memcopies total 0.049301 seconds. No timing-only
association is used to redistribute that remainder. Host ranges and per-name
runtime totals remain available in ``device-attribution.json``.

Ranked actions and outstanding evidence
---------------------------------------

#. Optimize the measured large CPU ``squared_norm`` reduction first as a bounded
   Python/Torch change. Preserve dtype, elementwise rounding, views and AD;
   retain negative small-input results. The companion optimization task owns it.
#. Investigate promoted IFFT arithmetic and conversion traffic together. This
   is the largest measured cost, but neither float32 substitution nor native
   kernel edits are authorized shortcuts. Evaluate any candidate with the
   existing precision matrix and complete corrected-science workload.
#. Investigate CUDA pointwise chi-square and its host synchronization/dispatch
   path. Preserve its phase-precision and scientific checks. The CUDA
   squared-magnitude kernels account for only 0.016280 seconds here, so the
   measured CPU reduction target does not imply a similar CUDA opportunity.
#. Treat frame I/O and repeated setup descriptors as finite-job startup targets.
   Reassess their importance with the larger convergence workload; do not
   subtract profiled durations from another process's wall time.

The completed
`R4 live-batch sweep
<https://github.com/xangma/pycbc/tree/b5cf0acf0eeddf20e6ebd51eab19500d5a83c06b/current-batch-sweep-20260907-r4>`_
holds 1024 synthetic templates, three blocks and FFT length 131072 fixed across
execution batches 1/8/32/128/512/1024, with real power and sine-Gaussian
chi-square. All 12 smoke and 36 full qualifications across seeds 7102/7103
passed before 54 timing workers. Adopted policy v2 checks every actual normalized
complex-SNR sample against both an independent complex128 oracle and actual MKL
output within absolute 0.001; all trigger/veto budgets are unchanged. R3 remains
failed under its original raw-complex rule.

CUDA median throughput rises from 2,424.4 template-block evaluations/s at batch
1 to 30,392.6 at batch 32, then 34,259.8 at batch 1024. The batch-1024 worker
medians span 34,031.2--34,505.9. Standard CPU stays near 1,200 evaluations/s;
Torch CPU is fastest at batch 1 (710.7), falling to 329.1 at batch 1024.
Each cell uses three fresh workers, two warmups and five timed iterations per
worker after a separate cold iteration. These observations identify a batch
effect but do not attribute it to kernels or transfers without a device trace.
This warm public live-filter API experiment excludes executable
setup/decompression/I/O and cannot establish the 2097152-point executable's
batch response or full-machine capacity.
The live worker supplies ``template.sigmasq`` as a precomputed host value,
whereas the executable profiles include ``squared_norm``. Verify actual call
coverage before using that sweep to assess a normalization optimization.

Reproduction
------------

Run the standard-library extractor against the trusted immutable evidence::

    python tools/summarize_torch_profiles.py /path/to/reference-campaign-20260907 \
        --output /new/path/existing-profile-attribution.json

It verifies every pstats/native-report hash and exclusive denominator against
the campaign summary, emits self/cumulative seconds and caller edges separately,
and retains each native denominator. Complete measurement argv/environment and
input hashes are in ``runs/<case>/receipt.json``. This verifies the profile
inputs, not every scientific input in the full archive.

Acquisition is serialized using Linux ``flock`` at
``/home/xangma/pycbc-torch-performance-coordination-20260907/len-benchmark.lock``.
Both convergence and the reviewed R4 batch campaign must be terminal and their
owned processes gone before new measurements; the older campaigns do not take
this lock. The obsolete failed batch/profiling queue is retained as failed.
Profiling completed in the first new measurement window. Scientific parity,
trace validation and unchanged source/native/input/helper hashes passed; all
owned process groups exited and the shared lock was independently reacquired
before the optimization task was released.

The acquired evidence directory is ``pycbc-torch-device-profile-20260907-r2``
on ``len`` and in the local ``torch-profiling-20260907-r2`` artifact folder.
It includes the complete helper bundle, per-stage and per-run receipts, HDF
outputs, host observations, raw trace, attribution and ``terminal-audit.json``.
The terminal ``profile-status.json`` SHA256 is
``954a1627755d6d0de57443368f7fc1f9947316983bd1f29531a13bade49a98bf``;
the raw trace SHA256 is
``c2378a8bc85f47332fa8ef63426cdf631062bdadd645d18dbe2be0b3e89ba7aa``.
The copied ``remote-evidence.tar.gz`` SHA256 is
``10298718066da7d1266a00cd874e87c7aaa0e7a8221cf1bfdd829196aa108870``.

CUDA acquisition and analysis method
------------------------------------

``tools/profile_torch_filtering.py`` runs the unchanged executable in a fresh
process. It synchronizes once before the first bank lookup, starts PyTorch CPU
and CUDA profiling, then synchronizes and stops at ``save_performance`` after
event consolidation. Semantic ranges cover IFFT, correlation, thresholding,
chi-square, squared magnitude, normalization, NumPy conversion and bank lookup.
There are no per-range synchronizations. The receipt requires every template
index exactly once and an IFFT for every template/segment pair. Failure restores
patches and leaves a failed receipt. Profiling output is ineligible for
throughput comparisons.

The new campaign must reuse the frozen 384-template inputs and geometry, record
source/native/input/helper hashes before and after, and run a separate
unprofiled same-source CUDA reference. Validate the instrumented HDF output
against that reference with the unchanged ``compare-triggers.py`` budgets.
Native acquisition and these parity checks passed in the new campaign.
Synthetic helper tests supplement those checks. The shared lock remains held while
the supervisor waits for and cleans up every measurement child.

Under that supervisor and the frozen single-thread environment, acquisition
and analysis have this form::

    taskset -c 8 python tools/profile_torch_filtering.py \
        --output /new/run/trace -- /pinned/source/bin/pycbc_inspiral \
        <unchanged executable arguments, with a fresh output HDF path>
    python tools/summarize_torch_trace.py /new/run/trace/trace.json \
        --receipt /new/run/trace/receipt.json \
        --output /new/run/device-attribution.json

Both Torch thread pools are initialized to one before the executable runs.
Environment variables alone do not set the inter-op pool. The helper records
and checks actual counts before the executable, at the first bank lookup and at
the loop end. The unprofiled comparison must use the same thread initialization
and record its CPU affinity and SMT topology before, during and after the run.

The analyzer verifies the completed receipt and trace SHA256. It counts only
kernel, GPU memcpy and GPU memset events; synthetic GPU annotation ranges are
excluded. External IDs and CUDA runtime correlation IDs link device events to
host calls. Containing ranges on the host call's own thread provide full
semantic paths such as ``chisq/to_numpy``. Missing or conflicting links remain
unattributed; timestamps alone never establish a host/device association.

Output retains per-device event counts, summed durations, interval unions,
events crossing the loop boundary, kernel/transfer names, recorded bytes where
available, inclusive host ranges and runtime calls. Windowed counts include only
events with positive-duration intersections; zero-duration events and boundary
touches are excluded. Summed durations can overlap;
neither category unions nor host and device durations may be added to obtain
elapsed time. The loop remainder is not proof of device idleness, and recorded
device activity is not an occupancy measurement. A trace without device events
fails analysis instead of producing a zero-time CUDA result.

Nine local tests cover asynchronous links, nesting, overlap, separate devices,
missing/conflicting associations, checksum rejection, absent CUDA events,
complete-loop coverage, thread-setting drift, boundary-only synchronization and
cleanup after failed or incomplete workloads. The APIs and event fields follow the
`PyTorch profiler documentation <https://docs.pytorch.org/docs/stable/profiler>`_
and `Kineto trace writer
<https://github.com/pytorch/kineto/blob/main/libkineto/src/output_json.cpp>`_.
The new native acquisition also verified installed-version trace compatibility.
