.. _jax-live-psd-numerical-differences:

Live PSD horizon numerical conventions
======================================

Live uses a 1.4 + 1.4 solar-mass, SNR-8 horizon as a PSD diagnostic. The JAX
calculation follows the original ``spa_distance`` calculation and its
dynamic-range scaling. This diagnostic is independent of diffGW template
generation. Default conditioning and the horizon integral run on the selected
JAX device.

The :download:`horizon comparison notebook
<../examples/jax/jax_live_psd_numerical_differences.ipynb>` compares identical
deterministic float32 and float64 PSD samples. It records library versions,
measures each selected route and asserts exact original scalar bytes for the
whole-stage and combined-stage controls. Agreement for these examples does
not establish equivalence of a complete search or every device.

Grid and rounding boundaries
----------------------------

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
----------------------------

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
executed in the same scheme. Native routes transfer data to the CPU and can
start a worker process; they are deliberately slow validation controls and
cannot be traced or differentiated through JAX.
