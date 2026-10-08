Detector geometry numerical differences
=======================================

The executed notebook ``examples/jax/jax_detector_numerical_differences.ipynb``
holds inputs fixed, compares the original public CPU methods, and asserts
exact dtype/shape/byte equality for independently selected original routes.
Observed last-bit differences depend on precision, device, and library build.

Clock and trigonometric arithmetic
----------------------------------

The original clock computes ``(gmst_reference + phase_offset) % (2*pi)``.
The default JAX geometry instead uses angle-addition identities to share the
sky basis over a time grid: for example,
``cos(start - ra)*cos(offset) - sin(start - ra)*sin(offset)``. These are equal
in real arithmetic, but change argument reduction and the rounding sequence.
The notebook compares all three expressions using the same NumPy trigonometric
functions before comparing NumPy with JAX. This isolates expression changes
from differences in elementary-function implementations.

The GMST control restores the original clock array. The antenna and delay
controls separately restore their complete original calculations. They also
work through combined/network methods.

Response contraction and distance scaling
-----------------------------------------

The original static response uses NumPy ``dot`` followed by NumPy elementwise
products and reductions. JAX uses XLA contractions and reductions. Their
accumulation and fusion can differ. The notebook freezes the basis vectors to
isolate this boundary; it does not infer a particular compiler instruction
sequence from output differences alone.

Original effective-distance scaling uses NumPy cosine, squared products,
addition, a half-power, and division. The JAX path uses JAX cosine, products,
``sqrt``, and division. The independent scaling control executes the original
method with fixed antenna factors; no replacement CPU formula is used.

Finite-arm transfer function
----------------------------

Write ``p = 2*pi*f*arm_length/c``. The original function forms complex
exponential differences, including ``1 - exp(-i*p*(1-n))``, and divides by
``p``. The JAX expression is the equivalent half-sum of phase factors times
sinc functions. At small nonzero phase, subtracting an exponential close to
one loses its small real component: ``1-cos(theta)`` can round to zero while
``2*sin(theta/2)**2`` remains nonzero. The notebook demonstrates this specific
cancellation and compares both transfer functions at held frequencies.
Phase multiplication/division order also differs.

The standalone response remains undefined at zero frequency and returns NaN,
matching the original function. ``Detector.antenna_pattern(frequency=0)`` uses
the original static response instead. The finite-arm validation control
restores the actual original exponential calculation.

Input precision and shape
-------------------------

Absolute GPS times near ``1e9`` seconds have float32 spacing much larger than
subsecond timing. The notebook shows this loss before any geometry call and
checks original validation against the same rounded inputs. It does not
attribute an input-precision change to JAX arithmetic.

Additional broadcast grids are an API extension. Existing native-supported
dot axes are retained; grids the original array method cannot evaluate use
the original scalar method per point as their validation oracle. This is a
shape contract, rather than a numerical tolerance.
