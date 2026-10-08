.. _jax-filtering-numerical-differences:

JAX conditioning and filtering numerical differences
====================================================

Conditioning, correlation, normalization and vetoes combine operations whose
floating-point ordering depends on the implementation. The
:download:`filtering comparison notebook
<../examples/jax/jax_filtering_numerical_differences.ipynb>` fixes input bytes,
precision, grids and calculation options before comparing original CPU and
default JAX results. Its controlled examples distinguish changed mathematical
choices from ordinary rounding.

The :download:`standalone filtering rounding notebook
<../examples/jax/jax_filtering_rounding.ipynb>` requires only NumPy and JAX.
It uses small generated inputs to isolate arithmetic and rounding, without
importing PyCBC or LAL. Use the comparison notebook above to validate the
actual PyCBC implementations and original routes. See
:doc:`jax_numerical_differences` for the shared validation rules and limits of
these fixed-input examples.

.. _jax-strain-numerical-differences:
.. _jax-conditioning-numerical-differences:

Conditioning and whitening
--------------------------

The :download:`strain comparison notebook
<../examples/jax/jax_strain_numerical_differences.ipynb>` generates a temporary
GWF, advances the public strain buffer, and compares conditioning, Welch PSD,
both truncated PSD grids and the whitened spectrum. It fixes sample bytes,
precision, spacing, epochs and options. The temporary frame is removed after
execution; no downloaded data is needed.

Default JAX runs conditioning on the selected device. Different results can
come from FIR coefficient construction, transform ordering, PSD arithmetic,
complex division and Tukey cosine values. The
:ref:`IIR <jax-iir-numerical-differences>` and
:ref:`FIR <jax-fir-numerical-differences>` sections below, together with the
:doc:`FFT <jax_fft_numerical_differences>`,
:doc:`PSD <jax_psd_numerical_differences>` and
:doc:`array <jax_array_numerical_differences>` examples, isolate those boundaries.
Single-precision live input retains single-precision PSD storage and Welch
calculation; x64 availability does not silently widen its transforms.

.. _jax-iir-numerical-differences:

Butterworth state updates
~~~~~~~~~~~~~~~~~~~~~~~~~

An IIR section advances its state as ``state[n] = A * state[n-1] + offset[n]``.
LAL visits samples in order. JAX composes these affine maps in a parallel
prefix scan. Composition is associative in exact arithmetic; finite-precision
addition and multiplication are not. Changing the tree therefore changes
rounding, even with identical coefficients and zero initial state.

Coefficient construction adds trigonometric functions, pole transforms and
gain normalization. Algebraically equivalent coefficient expressions can
round differently before samples enter the recurrence. The filtering notebook
holds a simple recurrence fixed to demonstrate the grouping boundary, separately
from complete highpass/lowpass comparisons. For additive updates
``[1e16, 1, -1e16, 1]``, sequential and parallel scans end at ``1`` and ``0``.
Its 256-sample float64 highpass comparison differs by at most ``2.42e-13``;
float32 highpass agrees in that particular example.

.. _jax-fir-numerical-differences:

FIR design and application
~~~~~~~~~~~~~~~~~~~~~~~~~~

Kaiser-windowed sinc design combines elementary functions with a coefficient
normalization sum. JAX uses a fixed compensated pair tree for that sum;
compensation retains rounding residuals, but does not make different
elementary-function implementations identical. The filtering notebook compares
actual coefficients and a cancellation example with known lost terms.

Short causal filtering and long circular FFT filtering follow the original
API's separate conventions. Fixing one coefficient array removes design
differences from an application comparison. FFT grouping, conjugate products
and inverse normalization can still round differently; see
:doc:`jax_fft_numerical_differences`. LDAS downsampling also inherits these
boundaries before selecting every ``factor``-th sample. The notebook's 41-tap
Kaiser coefficients differ by at most ``5.55e-17``. With identical coefficients
and original forward/inverse transforms, circular filtering still differs by
``4.44e-16``: that control leaves JAX spectral multiplication and scaling in
place, and does not apportion the remaining error between them.

.. _jax-whitening-numerical-differences:

Whitening operation order
~~~~~~~~~~~~~~~~~~~~~~~~~

The original padded path scales its forward transform by ``delta_t``, divides
by the PSD, and scales an unnormalized inverse by ``delta_f``. The fused JAX
path cancels those normalization factors algebraically before the inverse.
Finite-precision arithmetic rounds the original intermediate scale and
division separately, so this cancellation can change the result even when the
real-arithmetic expressions agree.

The strain notebook isolates that cause without an FFT. For float32 value
``0.86845714``, scale ``0.2`` and denominator ``1.142317``, scaling before
division gives ``0.15205187``; dividing before scaling gives ``0.15205185`` on
the displayed NumPy installation. Full whitening can additionally differ
because of the transform and division implementations.

Selecting ``fft``, ``ifft``, ``divide`` and ``gate_data`` uses the original
staged order with the original kernels. Adding ``firwin``, ``fir_zero_filter``,
``resample``, ``welch``, ``interpolate`` and
``inverse_spectrum_truncation`` restores all preceding stages in the notebook's
public GWF-to-spectrum example. Every intermediate dtype and byte, plus series
metadata, must match its original CPU reference exactly.

On the notebook's displayed CPU/FFTW installation, the default conditioned
strain and final whitened spectrum have relative L2 differences of about
``2.38e-7`` and ``4.91e-7``. The selected original stages match every byte.
Those observations describe this input and installation, not a universal
error bound.

.. _jax-gating-numerical-differences:

Gating and decisions
~~~~~~~~~~~~~~~~~~~~

Tukey gating evaluates a cosine taper and multiplies the data in place.
Elementary-function and multiplication rounding can change taper samples.
The strain notebook holds gate times and widths fixed, compares original/default
values, and verifies that gating a slice updates its parent. Its float64 gate
example has ten differing samples, at most ``2.22e-16`` apart, on the displayed
CPU libraries; the original gate control matches every byte.

Glitch detection estimates a PSD, whitens data, applies a strict magnitude
threshold and clusters surviving peaks. A small upstream difference can move
a threshold decision. Equal peak counts do not establish equal whitening
values. ``detect_loud_glitches`` restores the complete original calculation;
``findchirp_cluster`` restores clustering alone. Both controls remain opt-in.

Matched filtering and autocorrelation
-------------------------------------

Matched filtering passes conditioned samples through a segment transform,
PSD division, ``conjugate(template) * data``, an inverse transform and template
normalization. Compare those boundaries in order, holding each operation's
input bytes fixed. Equal templates alone do not fix the conditioned data or
PSD. A changed correlation can inherit preceding differences even when
native and JAX complex multiplication agree exactly on the same inputs.

A held complex frequency array similarly separates inverse-transform
rounding from correlation. See :doc:`jax_fft_numerical_differences` for that
control and the limits of attributing an opaque FFT library's internal
operations. See :doc:`jax_array_numerical_differences` for independent product
precision and reduction controls.

The filtering notebook fixes the norm and replaces correlation and the inverse
transform independently. Selecting both still leaves 16 correlation samples
different because PSD division remains JAX. Adding ``divide`` restores those
bytes for the fixture. Its complete frequency-input comparison selects
``weighted_inner``, ``correlate``, ``divide`` and ``ifft`` and asserts the
original SNR, correlation and normalization exactly.

A separate fixed-input complex64/float32 division control isolates reciprocal
rounding on the recorded CPU libraries. NumPy's result matches
``value * float32(1 / PSD)``; JAX's matches separately rounded real/imaginary
component divisions. Rounding the reciprocal before multiplication differs
from rounding each quotient, despite equal real-arithmetic expressions.
The 65-sample control has 27 differing complex values and an exact original
``divide`` route. This observation does not specify other libraries' lowering.

On a GPU, compare CPU and device component divisions separately. The
additional notebook control freezes both float32 reciprocals before
multiplication. On its recorded CUDA implementation, the device reciprocal
reproduces the device quotient; substituting the CPU reciprocal restores the
original complex quotient exactly. This isolates reciprocal rounding instead
of assuming that componentwise division has identical rounding on both
processors. It identifies a numerical boundary, not a unique vendor
instruction. The original ``divide`` control restores the installed CPU
calculation.

The final Inspiral SNR multiplies complex64 raw values by a Python
normalization scalar before taking their float32 magnitude. The notebook
isolates this finish: distinct binary64 scalars can round to the same float32
value at the multiplication boundary, while a larger step changes the scaled
components. Normalization and inverse-transform changes can therefore mask or
partly cancel one another. Preserve the scalar type and output dtype when
propagating held-stage differences to the final SNR.

Autocorrelation centers the input, transforms its power and normalizes the
result. An unbiased estimate additionally uses variance and lag-dependent
sample counts. Correlation length selects a cumulative-sum crossing, so a
small arithmetic difference can change a discrete window decision. A matched
array length or selected peak alone does not prove numerical equality.

The original convention promotes the zero-padded FFT input to float64 and
includes the time-series spacing in the unbiased normalization. On the
notebook's fixed float32 input, mean, centered bytes and double-precision power
sum agree, while variances are ``0.8751317`` and ``0.8751317858695984``. Holding
that power sum fixed reproduces the zero-lag change from about
``0.125000005235368`` to ``0.124999996721701``. This isolates variance reduction
rounding; full autocorrelation can also include transform differences.
``autocorrelation_mean`` and ``autocorrelation_variance`` select the original
NumPy reductions independently. Combining them with ``fft``, ``ifft``,
``correlate`` and ``divide`` restores the original ACF bytes in the notebook
while exercising the JAX stage composition.

Cached batched normalization
----------------------------

The batched search cache forms template power in the input's real precision,
scales it by ``4 * delta_f``, multiplies by a reciprocal PSD in that precision,
and accumulates in float64. The separate JAX ``weighted_inner`` API instead
widens operands before multiplication. ``squared_norm``, ``divide`` and
``inner`` independently select the cached calculation's original arithmetic.
Hold PSD bytes fixed when isolating these boundaries: an upstream PSD change
also changes normalization when every arithmetic operation is original.

The notebook's portable single-bin example changes a float32 PSD by two
representable steps. The reciprocal changes from ``23839.404296875`` to
``23839.400390625``; the two rounded weighted contributions decrease by
``0.25`` and ``0.125``. It compares each arithmetic control on fixed inputs
and isolates PSD propagation without recreating a full waveform or search.

Both Inspiral paths compute ``norm = 4 * delta_f / sqrt(sigma²)`` with host
``math.sqrt`` and a Python float result. A separate scalar example shows how
changed binary64 ``sigma²`` inputs change this normalization, while their
float32 event-column casts move by one representable step. Storage rounding
is a distinct boundary from normalization and subsequent SNR multiplication.

Magnitude thresholds
--------------------

Clustering uses a strict squared-magnitude threshold. For complex64
``-0.7931225 - 0.13891791j`` and threshold ``0.8051965236663818``, separate
float32 products/addition give power ``0.6483415``. The recorded compiled JAX
expression gives ``0.6483414``, equal to the squared float32 threshold. The
CPU class selects index zero; default JAX selects no index.

A control retaining one product's extra precision until addition reproduces
the JAX value. This demonstrates fused-expression rounding; it does not claim
a particular machine instruction on every compiler or device. The
``threshold_cluster`` control restores the original values and indices.

Power and sine-Gaussian chi-square
----------------------------------

The original compiled point chi-square seeds each frequency-bin rotation
using ``3.141592653``, converts sample indices to the input's real precision,
and advances phases through repeated complex multiplication. Independent
Fourier phases instead form integer frequency/sample products modulo the
transform length before double-precision phase evaluation. The phase seeds,
recurrence drift, complex-product grouping and summation therefore differ.
The notebook isolates seed precision and then compares complete point values.

The default ``chisq_mode="cpu-compatible"`` calculation follows the original
source recurrence on the selected device. Compiler fusion and elementary
functions can still produce different values from the installed CPU binary;
matching the source formula is not an exact-reference guarantee.
The notebook's original recurrence differs by at most ``1.53e-5`` from its JAX
counterpart; ``"direct-phase"`` differs by ``1.07e-4`` on those same inputs.
The isolated pi-only seed control differs by ``3.77e-5`` in complex phase.

The original CPU build can reassociate that recurrence. In the observed
scalar float32 path of one qualified CPU build, the compiler groups two
adjacent frequencies. With
``a = vr*(pr+pi)``, ``b = pr*(vi-vr)`` and ``c = pi*(vr+vi)``, its pair
updates are ``R = ((a1-c1)+(a0-c0))+R`` and
``I = ((I+b0)+(b1+a0))+a1``; the scalar remainder uses ``I = (b+I)+a``.
The observed JAX device loop instead adds ``a-c`` and ``a+b`` to each
complex carry separately at every frequency. The native final total also
adds each real square and imaginary square separately, ``(S+R*R)+I*I``;
JAX first groups a bin's power, ``S+(R*R+I*I)``. Each regrouping changes
float32 rounding even with identical phase seeds, products and bin order.

The notebook's six independently generated samples isolate the two
boundaries with fixed inputs: paired recurrence adds two float32 steps to the
point sum, and native final accumulation removes one. No phase-seed change is needed for that witness.
Other compiler builds or candidate geometries may group the loop differently; the original point
control remains the exact validation reference.

A loud, nearly coherent signal also amplifies preceding rounding. Write the
final statistic in exact arithmetic as ``chi_square = A - B``, with
``A = p * S * norm**2`` and ``B = abs(raw_snr)**2 * norm**2``. The
implementation subtracts the unnormalized terms before multiplying by
``norm**2``. Both terms can greatly exceed their
difference. The notebook's four-bin coherent example has ``A = 55281``,
``B = 55112`` and ``chi_square = 169``. Advancing only ``S`` by one float32
representable step changes the statistic by ``0.00390625``: its relative
change is about 327 times the relative change in ``S``. Advancing one raw
SNR component independently changes the statistic by ``-0.00390625``.
The original point control still restores dtype, shape and bytes.

This example uses sample zero to remove phase evolution and isolate the
subtraction's sensitivity. The CPU-compatible implementations retain the
same pi literal, index precision, three-product recurrence and bin order;
that source correspondence does not identify which compiled operation
produces each differing bit on another device. Upstream and point-kernel
rounding can reinforce or cancel after subtraction. Compare held inputs and
both terms per event; a difference between two maximum-error summaries does
not measure a stage's contribution.

Chi-square bin edges are discrete crossings of cumulative weighted power.
Float32 division followed by cumulative summation can move an edge when a
rounded value crosses a target. Widened intermediate division and ordered sums
reduce particular discrepancies without guaranteeing every edge on every
device. Keep bins fixed when isolating point evaluation.

The sine-Gaussian veto adds exponentials, a frequency-domain time shift,
template normalization and complex sums. Its activation thresholds and tile
placement also depend on upstream SNR and bin values. Compare those inputs
before attributing a veto difference to tile arithmetic.

The original frequency-domain shift rotates phases recursively and refreshes
them every 100 bins. JAX evaluates each bin's angle independently in the
input's real precision. Recurrence drift and product grouping therefore
differ even at identical grid spacing and shift time. The fixed complex64
shift differs by at most ``1.62e-6``; its original ``time_shift`` control is
byte-exact.

For the internal tile, NumPy and JAX evaluate exponentials and their products
with different numerical libraries. The recorded complex128 tile differs by
``1.73e-18``; the comparison alone does not isolate which elementary operation
accounts for every differing bit. The active complete veto differs by
``1.97e-6``. Original tile, shift and reduction controls together restore the
fixture, and ``sgchisq`` separately restores the entire original calculation.

Validation controls
-------------------

See :doc:`jax_filtering` for the operation names and
:doc:`jax_numerical_differences` for shared comparison requirements. The
filtering and strain notebooks assert dtype, shape, bytes and applicable
series metadata against the installed original calculation. Select the
preceding stages as well when validating a composed calculation.
