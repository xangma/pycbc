.. _torch-parity:

Torch scientific parity
=======================

Torch performance evidence is accepted only after the same route, device,
dtype, shape, and workload pass scientific parity. Comparisons cover public
values and metadata; independent floating-point implementations are not
expected to be byte-identical unless a test explicitly requires that.

Terminology
-----------

Raw-byte exact
   Every compared output byte matches, including signed zero. This is unusual
   for independent floating-point implementations.

Numerically equivalent
   Shapes, public dtypes, metadata, finite/non-finite classification, and values
   pass an explicit, justified tolerance and error metric.

Route-qualified
   Numerical equivalence plus evidence that the expected native, optimized, or
   fallback implementation ran on the expected device, with no prohibited host
   conversion.

Resident
   Bulk numerical inputs, intermediates, and outputs stay on the selected
   device over the claimed region. Small scalar control decisions, I/O, and
   documented boundary conversions are excluded only when stated.

A numerical pass is not evidence of speed, and an output copied to an
accelerator is not evidence that its generator ran there.

Reference and comparison design
-------------------------------

Use the most independent established implementation available. Keep identical
physical inputs, detector ordering, PSDs, frequency support, thresholds,
random seeds, and preprocessing. Do not compare two wrappers around the same
optimized kernel and call the result independent parity.

For every case:

#. record device, dtype, shape, feature flags, route identifier, and reference;
#. compare public shape, dtype, sampling metadata, epoch, support, ordering, and
   output device before comparing values;
#. check NaN, infinity, signed/complex-zero, empty, boundary, duplicate, and
   non-contiguous cases applicable to the API;
#. use the error metric and tolerance encoded by the focused test, and report
   them with the result rather than inventing a shared global tolerance;
#. poison or instrument host conversions when the route claims residency; and
#. force the optional route both on and off, then exercise at least one
   ineligible input to verify the documented fallback or error.

When random input is useful, preserve the seed and serialized input hash.
Property and metamorphic tests supplement, but do not replace, independent
reference cases.

Capability-specific contract
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 31 31 18

   * - Area
     - Scientific comparison
     - Route and metadata checks
     - Required error view
   * - FFT
     - Established FFT result and forward/inverse reconstruction with the same
       normalization, logical length, and input.
     - Direction, transform kind, length, batch shape, dtype, strides,
       output allocation, device, and optimized/generic route.
     - Absolute and relative complex error over FFT length, batch, and dtype.
   * - Filtering and search
     - Correlation/SNR series plus independently selected thresholds, peaks,
       clustering, and trigger records.
     - Sample spacing, epoch, normalization, template/data order, threshold
       semantics, event ordering, route, and residency.
     - Series error and discrete trigger agreement, including boundary events.
   * - Waveforms
     - Independent LAL/existing implementation where available; otherwise an
       analytic or compatible-grid contract.
     - Polarization, frequency support, epoch, delta-f or requested frequency
       order, duplicate frequencies, dtype, device, and native/fallback route.
     - Amplitude and phase/complex error over parameter and frequency grids.
   * - Decompression
     - Established CPU decompressor for supported interpolation methods.
     - For native inline linear/quadratic/cubic/quartic routes, check automatic
       and provided output, zeroed boundaries, dtype, epoch, and output device.
       Treat non-inline SciPy interpolation as a host route.
     - Interpolation error over degree, output precision, sample spacing, and
       device; MPS single precision is a separate comparison class.
   * - Inference
     - Existing scalar/NumPy likelihood, relative-binning, and per-detector
       response calculations using identical bins, PSDs, and detectors.
     - Batch axes, detector order, dtype, device, reductions, waveform route,
       and any supported gradient path.
     - Likelihood/response error and gradient error where gradients are part of
       the public contract.

Waveform boundary
-----------------

The registered native FD and FD-sequence families are ``TaylorF2``,
``TaylorF2NLTides``, ``TaylorF2RedSpin``, ``TaylorF2RedSpinTidal``, and
``TaylorF2Ecc``. The registry support predicate remains authoritative.

Do not classify every sequence port as lacking a LAL reference. Use the
existing dispatcher/reference when the approximant and sequence interface
provide one. If no equivalent independent sequence implementation exists,
compare with analytic identities and a compatible regular-grid call while
testing requested-frequency order, duplicate frequencies, support boundaries,
empty input, dtype, and device. The waveform registry and its focused tests are
the authority for detailed routing.

The ``TaylorF2`` batch API additionally checks scalar and length-one broadcast,
inconsistent lengths, row padding, ``first_bins``/``end_bins``, common grid
metadata, batch ordering, and the output tensors' device.

Fallback and residency
----------------------

Test three states for each optional optimization:

* **disabled:** the established route remains numerically correct;
* **enabled and eligible:** the intended route is observed and passes parity;
* **enabled but ineligible:** the documented fallback or explicit error occurs
  and preserves the public contract.

Fallback-generated output copied to CUDA or MPS is device-correct but not
native-generation evidence. Likewise, MPS precision promotion that stages
through CPU memory is not resident even if its final tensor is on MPS.
Performance labels and plots must keep these routes separate.

Host-conversion guards should cover the bulk region claimed as resident.
Allow only explicit public conversions such as a requested ``.numpy()`` or
``.lal()``, compact event materialization, file I/O, and documented fallback
boundaries.

MPS accuracy
------------

Many MPS operations lack ``float64`` or ``complex128`` support. Use the actual
MPS-compatible public dtype, compare against a reference rounded through the
same dtype where scientifically appropriate, and retain the focused test's
explicit tolerance. Do not silently downcast an API that promises a
double-precision result. If the public contract cannot be met, guard the route
or report it unsupported.

MPS results require real MPS hardware. A conditional skip on a non-MPS runner
is not a parity pass, and a CPU-emulated check is not MPS evidence.

Parity artifacts
----------------

Store machine-readable case rows containing input or input hash, reference and
candidate route, device, dtype, dimensions, all error metrics, tolerance,
metadata checks, residency result, pass/fail/skip reason, source revision, and
test command. Preserve failures and unsupported cells.

The parity/error plot required by :ref:`torch-performance` must be derived from
these rows for every timed route. A plot without the raw parity artifact cannot
qualify a benchmark.

Focused and full validation commands are maintained in
:ref:`torch-testing`. Device skips must be counted and reported separately from
passes.
