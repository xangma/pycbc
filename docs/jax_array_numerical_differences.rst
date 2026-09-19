.. _jax-array-numerical-differences:

JAX array numerical differences
===============================

Identical input values do not guarantee identical floating-point results
across the original CPU and JAX implementations. Precision before a product,
the order of a reduction and fused arithmetic all matter. Enabling JAX's
64-bit support does not reproduce the CPU implementation's arithmetic order.

The :download:`array comparison notebook
<../examples/jax/jax_array_numerical_differences.ipynb>` runs the original CPU
operations, the default JAX operations and each selected native validation
route on identical deterministic inputs. It asserts exact CPU/native equality,
including output dtype and bytes, and reports the default JAX differences.
Its recorded outputs apply to the displayed versions and device; rerun it for
another environment. These array examples do not qualify a complete search.

Products and accumulation
-------------------------

The CPU ``inner`` forms products in the input precision, then accumulates in
double precision. JAX converts inputs to double precision before multiplying.
For the single ``float32`` value ``1.234567``, the CPU self-inner product is
``1.5241557359695435`` and the default JAX result is
``1.5241557914777246``. The notebook explicitly calculates both rounding
sequences to establish the cause.

For complex inputs, CPU ``inner`` forms conjugate products in the input
precision. JAX widens the inputs; its self-inner specialization evaluates the
sum of squared real and imaginary parts. CPU ``weighted_inner`` also forms
the product and division before double-precision accumulation, whereas JAX
widens the operands first. The notebook covers real and complex examples of
both operations. Select ``inner`` or ``weighted_inner`` independently to
restore the corresponding original CPU calculation.

Reduction order
----------------

Floating-point addition is not associative. With ``float64`` values,
``(1e16 + -1e16) + 1`` gives one, while ``1e16 + (-1e16 + 1)`` rounds
to zero. Cancellation therefore makes different reduction trees visible.

CPU ``sum`` uses NumPy's double-precision reduction; ``dot`` uses NumPy's
dot implementation, and ``cumsum`` uses NumPy's cumulative sum. The JAX
counterparts use JAX reductions and scans. Their grouping can depend on the
device, library versions and array length. The notebook uses repeated
large/small cancellation patterns, reports actual discrepancies and compares
the sum with ``math.fsum``. A higher-precision diagnostic does not replace
the original CPU validation target. Select ``sum``, ``dot`` and ``cumsum``
separately when isolating these operations.

Multiply and add
-----------------

CPU ``multiply_and_add`` uses BLAS AXPY. Its arithmetic can fuse a multiply
and add, rounding once. The default JAX array implementation evaluates the
multiply and add through separate array operations. For ``float32`` inputs,
``-1 + (1 + 2**-23) * (1 - 2**-23)`` distinguishes the two sequences:
separate rounding gives zero, whereas one final rounding retains ``-2**-46``.
The notebook computes both controls and reports which result each backend
produces. BLAS and device behavior can vary; select ``multiply_and_add`` to
use the installed original BLAS implementation exactly.

Validation boundaries
----------------------

Use :ref:`jax-native-validation` to select operations individually. Native
routes transfer their inputs to the host and call the original CPU helpers;
array outputs return to the JAX device. They can be substantially slower and
cannot be used inside JAX tracing or differentiated through that boundary.

Exact equality is a comparison against the same installed CPU implementation
with the same input bytes. Different CPU builds can use different BLAS,
compiler flags or reduction implementations. In particular, nonfinite values
and overflow can affect magnitude selection; do not infer their behavior from
a finite-input example. The additional notebook controls check the available
norm and extrema routes without claiming a default discrepancy in every case.

These switches select named PyCBC array operations. Internal expressions in
other JAX kernels require their own stage controls and validation. JAX's
`numerical FAQ <https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs>`_
also explains how compiler transformations can change floating-point results.
