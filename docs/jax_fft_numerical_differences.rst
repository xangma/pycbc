.. _jax-fft-numerical-differences:

JAX FFT numerical differences
==============================

PyCBC's CPU and JAX FFT paths evaluate the same discrete Fourier transform
using different implementations. Finite-precision results can differ even
with identical input dtype, length and normalization. Such differences do not
by themselves establish that a downstream search makes the same decisions.

The :download:`FFT comparison notebook <../examples/jax/jax_fft_numerical_differences.ipynb>` compares the selected
original CPU backend with default JAX and native validation routes. It uses
identical deterministic real and complex inputs in single and double precision,
including cancellation patterns and seeded random data. Forward and inverse
native results must match the CPU output dtype and bytes exactly. Versions,
the selected CPU backend and the JAX device are recorded without local paths.

The :download:`standalone fft rounding notebook <../examples/jax/jax_fft_rounding.ipynb>` requires only NumPy and JAX.
It uses small generated inputs to isolate arithmetic and rounding, without
importing PyCBC or LAL. Use the comparison notebook above to validate the
actual PyCBC implementations and original routes. Each notebook records
its own library versions and device; matching arithmetic controls do not
establish complete-search equivalence.

Sources of differences
----------------------

FFT algorithms factor the transform into smaller operations, often described
as butterflies. Different decompositions, ordering and approximations to
complex phase factors can round differently. Cancellation makes small absolute
errors large relative to an individual near-zero output bin. The notebook
therefore reports maximum absolute error and error scaled by the largest CPU
output magnitude, together with exact mismatch counts.

This output comparison does not identify an internal butterfly or phase-factor
calculation responsible for each bin. CPU library choice, library versions,
planning and device execution can change results. The notebook names the
actual CPU backend rather than assuming FFTW or MKL, and keeps the input
bytes fixed when testing the inverse direction.

CPU buffer alignment can also change the result. PyCBC selects FFTW's
unaligned planning flag when a transform buffer misses its required alignment.
That can select a different algorithm, even with the same dtype, input bytes
and planning level. Independent CPU allocations can therefore produce small
rounding differences between two original pipeline runs. See FFTW's
`alignment requirements
<https://www.fftw.org/fftw3_doc/SIMD-alignment-and-fftw_005fmalloc.html>`_
and `planner flags <https://www.fftw.org/fftw3_doc/Planner-Flags.html>`_.
The notebook varies only buffer alignment and reports the observed difference;
some builds produce identical values for both layouts.

For reproducible whole-pipeline byte comparisons, select the original NumPy
FFT backend in both runs with ``--fft-backends numpy``. The validation switches
honor that selection; they do not change the default CPU backend. Controlled
FFTW comparisons require equivalent buffer alignment and plan settings.

Input precision also matters. ``float32``/``complex64`` transforms retain
single-precision outputs; JAX's enabled 64-bit support does not widen them
automatically. A double-precision transform of the same rounded input is a
useful diagnostic but does not replace the original single-precision target.

The held-input inverse example also prints the installed NumPy FFT's return
dtype. Do not assume that every NumPy version automatically widens a
complex64 input to complex128. The original plan casts its library result to
the supplied output dtype before its unnormalized scale. The native ``ifft``
control restores that exact boundary. Holding the correlation bytes fixed
isolates the inverse transform; it does not determine an undocumented
internal accumulator or per-bin instruction sequence.

Normalization is a separate contract. Plain PyCBC ``Array`` transforms are
unnormalized in both directions. The functional series interface applies
``delta_t`` or ``delta_f`` afterward; see :doc:`jax_fft`. The JAX inverse
implementation obtains a normalized JAX inverse and scales by ``N``. That
additional scale can introduce rounding compared with an implementation that
forms an unnormalized inverse directly, especially when ``N`` is not a power
of two. The notebook checks the normalization contract but does not attribute
every observed inverse discrepancy to that scale.

FFTW real inverse transforms may overwrite their CPU input buffer. Save a
forward spectrum before an inverse call when comparing those boundaries.
Native validation uses a scratch copy and preserves the JAX input; this input
ownership difference does not change the inverse output.

Isolating each direction
------------------------

Select ``reference_operations=("fft",)`` or ``("ifft",)`` independently,
keeping the original CPU transform input fixed. Selecting both names restores
the original transform pair while retaining JAX buffers around it. The notebook
asserts exact equality for these native routes and demonstrates the standard
series spacing and epoch behavior.

Native routes use the same installed CPU backend, input bytes and applicable
plan geometry as the reference. Equality across different CPU builds or
complete workflows requires separate evidence. Class-plan batching also
depends on the selected native backend's capabilities. Host transfers and
CPU execution make these routes validation tools, with no differentiation or
JAX tracing across the selected boundary.
