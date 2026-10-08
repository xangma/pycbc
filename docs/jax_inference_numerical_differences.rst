Inference numerical differences
================================

The companion notebook
:download:`jax_inference_numerical_differences.ipynb <../examples/jax/jax_inference_numerical_differences.ipynb>`
compares the installed original implementations with deterministic JAX inputs.
Its assertions check dtype, shape, and bytes for reference routes. Reported
residuals describe the notebook's device and library versions.

Array products and whitening
----------------------------

Original single-precision Array inner products round each product before
accumulating in double precision. JAX promotes the inputs before multiplication.
For a float32 value ``1.234567``, the self-products therefore differ even with
one element. At physical complex64 amplitudes of order ``1e-23``, rounding a
product in single precision can underflow. ``inference_inner`` restores the
original product and accumulation; ``sum`` alone controls only the reduction
of the already formed JAX products.

Whitening has separate inverse-PSD, square-root, and in-place precision
boundaries. ``divide`` isolates division; ``inference_whitening`` executes the
original PSD setter and template weighting. An original routine's precision
restrictions and errors still apply to its reference route. In particular,
the notebook's complete original-model comparison uses complex128 data;
complex64 evidence is demonstrated at the supported Array/kernel boundaries.

Relative bins
-------------

Subtracting two global prefix sums can erase a small bin after a much larger
preceding contribution. The notebook shows ``1e16, 1`` giving a prefix
difference of zero while the bin-local sum is one. Default JAX summaries reduce
each bin locally, following the original calculation. ``relbin_summary``
selects that complete original stage; ``sum`` independently selects their original ndarray reductions.

Squaring complex components around ``1e-23`` underflows in float32. Complex
division avoids that unnecessary squared denominator; the notebook compares a
safe quotient with the explicitly squared counterexample. ``divide`` isolates
the original division. For the notebook's tiny complex64 quotient, original
NumPy rounds a reciprocal scale before multiplying the components, giving a
value one ULP below the observed JAX quotient. The notebook reproduces that
rounding using the `original NumPy implementation
<https://github.com/numpy/numpy/blob/v2.5.2/numpy/_core/src/umath/loops.c.src#L2032-L2060>`_.
This observation is specific to those inputs and builds. Relative likelihoods use the original linearized
expression ``a0*r0 + a1*(r1-r0)`` and compiled constant ``3.141592653``.
The notebook checks that phase convention against the original compiled kernel,
and shows why replacing it with full-precision pi changes the phase.

Parallel reductions and complex exponential implementations can still round
differently from the original serial compiled loops. The time-grid SNR predictor
also evaluates phases directly, whereas the original advances a complex twiddle
by repeated multiplication. ``relbin_likelihood`` and ``relbin_snr`` select
the corresponding complete original kernels, including their buffering and
iteration order. Those native kernels require double-precision buffers.
Dominant-mode SNR samples have a further magnitude and scalar-normalization
boundary. With identical kernel outputs, the notebook's magnitude and scalar
square root match, while division differs. Multiplication by a rounded reciprocal
reproduces the observed JAX result; the original divides directly.
``relbin_snr_normalization`` independently restores those NumPy operations.
``relbin_likelihood`` also restores the original final dominant-mode products
and accumulation after interpolation.

Relative polarization rotation has another complex-product boundary. The
notebook fixes the original phase and measures the rotation separately from
phase and inclination trigonometry. Separate real/imaginary expressions and
NumPy complex multiplication round differently for these inputs.
``inference_projection`` restores the original rotation and inclination
combination independently. No claim about a particular compiler instruction
is needed to establish this measured boundary.

Timing and marginalization
--------------------------

Adding a small delay to a large absolute GPS timestamp can round before the
epoch is subtracted. Default projection forms the small relative time first;
``time_shift`` restores the original absolute-time ordering and shift kernel. Relative-bin
reference-data preparation separately restores its original NumPy exponential
and multiplication, before the bin summaries are computed.
Antenna patterns are evaluated at the detector arrival time, as in the original
marginalized models. ``inference_time_interpolation`` separately restores the
original TimeSeries interpolation.

JAX's scaled Bessel function, weighted log-sum-exp, and B-spline arithmetic may
differ from SciPy's evaluation and reduction order. The formulas and probability
weights are the same. The notebook measures those residuals without attributing
them to an uninspected library implementation. ``inference_marginalization``
and ``inference_interpolant`` independently restore the original operations.

Weighted sampling uses the caller's NumPy uniform draws. Small differences in
exponentials, cumulative sums, or normalization move CDF boundaries and can
select different indices for the same draw. ``inference_sampling`` restores the
original weighted draw while preserving the RNG stream. It also restores the
phase-grid and log-weight expressions used in parameter reconstruction before
that draw. The phase grid uses float64, including for complex64 likelihood
inputs; the notebook isolates residuals after matching that precision.
Sampling without replacement retains the model's original RNG generator.

The notebook composes the independent controls in physical Gaussian and
marginalized-model calculations and checks exact original results. It also
checks a device likelihood gradient against a finite difference. Reference
controls cross a host boundary and are excluded from that gradient check.
