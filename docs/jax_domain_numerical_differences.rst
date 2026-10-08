Numerical differences in JAX domain calculations
================================================

The :download:`executed notebook <../examples/jax/jax_domain_numerical_differences.ipynb>`
compares fixed inputs, reports package versions and dtypes, and checks each
selected original route with byte equality. Its measured differences are
examples for the recorded environment, not universal error bounds.

Mass roots and precision
------------------------

The original chirp-mass inverse constructs the cubic
``m2**3 - a*(m2 + m1) = 0``, where ``a = mchirp**5 / m1**3``, and calls
``numpy.roots``. NumPy solves this through companion-matrix eigenvalues.
The JAX path solves the dimensionless cubic using Cardano's formula or its
trigonometric form, then makes one Newton correction in ``log(m2 / m1)``.
It solves the same equation through a different sequence of operations.
The quadratic known-mass inverse likewise uses explicit roots instead of
companion-matrix eigenvalues; its smaller real root uses the product-of-roots
identity to avoid subtraction cancellation.

In the notebook's single-precision example, chirp mass 16 and known mass 30
produce approximately 11.747801099584137 through the original double-precision
root solver and 11.747801780700684 through the JAX single-precision solver.
Double-precision inputs can still differ in their final bits because the
algorithms differ. These are not changes to the mass definitions.
Use ``conversions.mass2_from_mchirp_mass1`` or
``conversions.mass_from_knownmass_eta`` to restore each original solver
independently.

Linear and spline interpolation
-------------------------------

TOV conversion tables use the JAX mass precision. The original NumPy
interpolation uses double precision, so float32 JAX masses also round table
knots and values. For the notebook's table and float32 mass near 1.1, the
original scalar transform gives 799.9999523162842 and the JAX transform gives
800.0. Selecting ``transforms.LambdaFromTOVFile.transform`` replays the original
scalar transform for every vector entry and returns its double-precision
results.

Even with identical double-precision tables, the linear-interpolation kernels
can round differently. NumPy evaluates a precomputed slope times the offset;
JAX computes the offset divided by the interval width before multiplying by
the value difference. The multiply and addition can also be fused by the CPU
compiler or XLA. The notebook isolates both operation orders and checks a
compiled JAX expression against ``jax.numpy.interp``. It also classifies
whether the original CPU build uses fused or separate multiply/add rounding.
Cosmology retains the original double-precision table values, so it has the
kernel-order boundary without TOV's float32 table-rounding boundary.
Select ``cosmology`` or the relevant independent interpolator method to restore
the original interpolation.

QNM helpers reuse the original ``pykerr`` cubic-spline coefficients and knots.
JAX casts those tables to the tensor precision and evaluates them in Horner
form; original SciPy spline evaluation uses double precision. Table precision
and compiled arithmetic can therefore differ. The independent
``conversions.get_lm_f0tau`` control restores the actual original public API,
including an original dependency failure where applicable.

Elementary functions and compositions
-------------------------------------

Coordinate, spin, tidal, log, and custom-expression calculations use JAX
arithmetic and elementary-function kernels. NumPy/SciPy kernels and XLA can
use different implementations of powers, logarithms, and trigonometric
functions, or fuse adjacent operations. The notebook measures these boundaries
on held inputs and verifies original coordinate and transform routes exactly.
There is no fixed tolerance that proves equivalence for every composition.
Choose a function's fully qualified control to isolate it, or select several
categories to replay all of these domain stages together.

Sources
-------

* `NumPy root-solver algorithm <https://numpy.org/doc/stable/reference/generated/numpy.roots.html>`_.
* `NumPy linear-interpolation implementation <https://github.com/numpy/numpy/blob/v2.5.2/numpy/_core/src/multiarray/compiled_base.c>`_.
* `JAX linear-interpolation implementation <https://github.com/jax-ml/jax/blob/jax-v0.11.1/jax/_src/numpy/lax_numpy.py>`_.
