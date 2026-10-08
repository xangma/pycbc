.. _jax-psd-numerical-differences:

JAX PSD numerical differences
=============================

A PSD combines several numerical boundaries: window construction and
multiplication, Fourier transforms, complex power, averaging, bias correction
and normalization. Inverse-spectrum truncation adds reciprocals, square roots,
another window and two transforms. Identical physical formulas therefore do
not guarantee identical finite-precision outputs.

The :download:`PSD comparison notebook
<../examples/jax/jax_psd_numerical_differences.ipynb>` uses fixed deterministic
inputs. It compares original CPU and default JAX results, and asserts that
selected native routes reproduce CPU dtype, shape, metadata and output bytes.
Its window, fixed-spectrum and transform controls isolate particular boundaries.

The :download:`standalone psd rounding notebook
<../examples/jax/jax_psd_rounding.ipynb>` requires only NumPy and JAX.
It uses small generated inputs to isolate arithmetic and rounding, without
importing PyCBC or LAL. Use the comparison notebook above to validate the
actual PyCBC implementations and original routes. See
:doc:`jax_numerical_differences` for the shared validation rules and limits of
these fixed-input examples.

Welch estimation
-----------------

CPU and JAX construct Hann windows through different numerical libraries.
Casting to the sample dtype can eliminate some window differences and retain
others. Supplying the same explicit window fixes this input boundary, without
fixing subsequent FFT or reduction arithmetic. The notebook compares generated
windows, their energy sums and PSDs obtained with one common window.
In the recorded CPU run, the 64-point float32 windows matched, whereas
float64 windows differed by at most ``3.89e-16``.

The CPU estimator transforms segments separately through its selected FFT
backend. JAX batches the transforms. See :doc:`jax_fft_numerical_differences`
for transform rounding and its limits: the exact internal butterfly causing a
particular bin difference is not established by an output comparison.
With the common float32 window, replacing segment FFTs reduced the mean PSD's
maximum absolute difference from ``2.33e-9`` to ``9.31e-10``. Eleven of 33 bins
still differed, demonstrating a boundary beyond the transforms.

CPU forms ``abs(spectrum * conjugate(spectrum))``; JAX Welch uses the
real part of that product. Fixing the complex spectrum separates this power
boundary from transform differences. For float32 components ``r`` and ``i``,
rounding both squares before addition differs from retaining one square until
the final addition. The notebook supplies finite complex64 values for which
these two expansions differ and reports which matches each displayed
implementation. Matching an expansion describes its rounding; it does not
uniquely identify a machine instruction on every compiler.

The same power boundary occurs after the final inverse-ASD truncation
transform. Taking a float32 reciprocal afterward can preserve or amplify the
power difference. Computing ``abs(spectrum)**2`` instead introduces another
rounding sequence; it is a separate control, not either implementation's
source formula. Native ``welch`` and ``inverse_spectrum_truncation`` controls
restore the original complete stages.

Mean reductions, the averaging of an even median's middle values, division
by median bias and window-energy normalization introduce further rounding.
JAX includes widened float32 median-bias division to avoid known
reciprocal-multiplication differences; this does not guarantee equality of the
complete estimator. The notebook reports controlled mean/median/division
results without attributing every remaining PSD difference to one stage.
The fixed float32 mean differed by ``1.19e-7`` while its median matched.
Multiplying by a rounded reciprocal changed three division results, by at
most ``2.38e-7``; widened division matched the CPU results. Reduction grouping
and the extra reciprocal rounding therefore have separate demonstrated
effects on these inputs.

Explicit ``wide_fft=True`` retains the input-precision window multiplication
but widens the transform and following PSD arithmetic to double precision.
The notebook reports this option's output dtype and differences separately;
it changes the calculation rather than reproducing the original CPU path.

Inverse spectrum and interpolation
-----------------------------------

Inverse-ASD truncation takes a reciprocal and square root; inverse-PSD
truncation omits the square root. Both transform to time, truncate or taper
the response, transform back and take a reciprocal. JAX uses widened
float32 reciprocal and square-root calculations before casting back to the
original precision. Transform rounding, window rounding and grouping of the
final magnitude/power operations remain separate boundaries.
JAX uses normalized ``irfft`` followed by raw ``rfft``. The original
functional pair scales an unnormalized inverse by ``delta_f`` and the forward
by ``1 / (N * delta_f)``. These have equivalent net scaling in exact arithmetic
but place rounding differently. Internal ``fft``/``ifft`` controls retain the
JAX normalization; the whole-stage control restores the original placement.
The recorded float32 inverse-ASD/Hann result differed by at most ``7.15e-7``;
selecting both original transforms retained that maximum difference. This
comparison does not identify a unique cause within the remaining arithmetic.

Interpolation uses double-precision intermediate value arithmetic before
casting back to the input dtype, matching NumPy's precision policy. Frequency
grid construction and interpolation arithmetic can still round differently.
Fix both input values and the requested output grid when comparing it.
The notebook's fixed interpolation examples matched the CPU values exactly
in both precisions.

Analytical models
------------------

JAX analytical models use ported formulas or interpolated reference tables.
Constants, elementary-function implementations and floating-point grouping
can affect agreement with the original model. A representative model check
does not establish agreement for every model; retain model-specific evidence.
Quantum noise is proportional to reduced Planck's constant in the
`original formula
<https://lscsoft.docs.ligo.org/lalsuite/dev/lalsimulation/_l_a_l_sim_noise_p_s_d_8c_source.html#l00272>`_.
JAX uses ``h / (2*pi)``; substituting ``1.054571817e-34`` changes the quantum
PSD by approximately ``-6.13e-10`` relatively. The notebook changes only this
factor and compares the default with original LAL scalar evaluations.

The `original table algorithm
<https://lscsoft.docs.ligo.org/lalsuite/dev/lalsimulation/_l_a_l_sim_noise_p_s_d_8c_source.html#l01093>`_
extrapolates log ASD from its selected interval. At 9 Hz, the installed
version-18 design table gives ``2.617705e-42``; clamping to its first row gives
``3.017420e-42``, approximately 15.27% higher. The notebook computes both
controls and compares the extrapolation with original CPU and default JAX.
The series cutoff
starts at ``int(low_freq_cutoff / delta_f)`` rather than rounding up to the
first frequency above the cutoff. The notebook checks this convention at a
fractional cutoff and reports the model's remaining numerical residual.

.. _jax-live-psd-numerical-differences:
.. _jax-live-psd-horizon:

Live PSD horizon diagnostic
---------------------------

Live uses a 1.4 + 1.4 solar-mass, SNR-8 horizon as a PSD diagnostic. The JAX
calculation follows the original ``spa_distance`` calculation and its
dynamic-range scaling. This diagnostic is independent of diffGW template
generation. Default conditioning and the horizon integral run on the selected
JAX device.

The :download:`horizon comparison notebook
<../examples/jax/jax_live_psd_numerical_differences.ipynb>` compares identical
deterministic float32 and float64 PSD samples. It records library versions,
measures each selected route and asserts exact original scalar bytes for the
whole-stage and combined-stage controls. The shared
:doc:`validation limits <jax_numerical_differences>` apply to these examples.

Grid and rounding boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original amplitude preconditioner uses
``f = (index + 1) * delta_f``. Its cutoff index is
``int(low_frequency_cutoff / delta_f)``. With ``delta_f=1`` and a 20.25 Hz
cutoff, the first retained PSD index is 20 and its amplitude is evaluated at
21 Hz. Substituting the zero-based FFT frequency grid changes this diagnostic;
the notebook holds the PSD fixed to demonstrate the effect. In its recorded
examples, that substitution increases the horizon by approximately 3.805%.

The amplitude ``f**(-7/6)`` is rounded to float32 before squaring, including
for a float64 PSD. Squaring first in float64 and rounding only the power can
produce different float32 values. The notebook isolates this cast boundary;
later accumulation can absorb a particular power difference.
At 22 Hz, the notebook obtains powers ``0.000737361377`` and
``0.000737361435`` from these two rounding policies.

Division promotes float32 powers to float64 when the PSD is float64. With a
float32 PSD, the quotient and original cumulative sum remain float32.
Repeated rounded additions can differ from a float64 cumulative sum cast
once at the end. The notebook changes only this accumulation policy before
applying the original scaling, and plots its effect on cumulative norms.
For the recorded float32 PSD, widening this sum changes the horizon by
approximately ``-4.945e-7`` relatively.
Wider precision changes the calculation; it is not an original-code route.

The cumulative norm is multiplied by 4 and then ``delta_f`` before storage
in a float64 vector. The scalar finish preserves multiplication by the
binary-mass amplitude factor squared, square root, division by 8 and
dynamic-range multiplication. The upper index uses the original Schwarzschild
ISCO cutoff; when that cutoff lies beyond the supplied PSD, it uses
``len(psd) - 2``. Elementary-function implementations and compiler arithmetic
remain device-dependent boundaries that require measured comparisons.

Select original calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each control is optional. An empty selection keeps the on-device default.

=========================  ================================================
Reference operation        Original calculation selected
=========================  ================================================
``psd_horizon``            Complete CPU ``spa_distance`` diagnostic
``psd_horizon_amplitude``  CPU float32 amplitude preconditioner
``divide``                 NumPy division of power by the supplied PSD
``cumsum``                 NumPy cumulative sum in the quotient dtype
=========================  ================================================

For example, retain the JAX surrounding calculation while replacing the
three numerical stages::

    from pycbc.scheme import JAXScheme
    from pycbc.strain.strain_jax import psd_horizon_distance_jax

    with JAXScheme("cpu", reference_operations=(
            "psd_horizon_amplitude", "divide", "cumsum")):
        distance = psd_horizon_distance_jax(psd, 20.25)

Select ``psd_horizon`` alone to compare the complete original implementation.
The shared ``divide`` and ``cumsum`` controls also affect other JAX operations
executed in the same scheme. See :doc:`jax_numerical_differences` for the
shared requirements and costs of original controls.

Selecting boundaries
--------------------

Select ``welch``, ``inverse_spectrum_truncation``, ``interpolate`` or
``analytical_psd`` independently with ``reference_operations``; see
:doc:`jax_fft`. These routes call the original CPU implementation rather than
approximating its rounding in a new JAX expression. The shared
:doc:`validation rules <jax_numerical_differences>` apply to exact comparisons
and the interpretation of remaining default discrepancies.

To isolate transform effects within JAX PSD arithmetic, select ``fft`` for
Welch or ``fft`` and ``ifft`` separately for inverse-spectrum truncation.
The PSD notebook keeps other inputs fixed and reports the resulting residuals;
using original transforms alone does not replace windowing, power or
averaging arithmetic with their CPU implementations.
