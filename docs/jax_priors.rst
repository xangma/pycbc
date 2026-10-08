JAX prior distributions
=======================

Passing raw JAX parameter arrays to supported distributions evaluates their
PDFs, log PDFs, inverse CDFs and constraints on the input device. NumPy and
Python inputs retain their original implementations. Broadcast parameter grids
are supported by uniform, Gaussian, power-law, angular, mass-ratio and
log-uniform priors, their joint distributions, fitted KDEs, tabulated priors
and convex-hull constraints.

Use floating-point parameter arrays. Their precision determines the JAX
calculation; original validation may return a wider native dtype. Mixed inputs
are placed with the first raw JAX parameter array, including fitted constants.
Creating the arrays inside ``JAXScheme`` selects its default device. Constants
are cached separately by precision and device; fitting KDEs and preparing
interpolation tables still use their original host implementations.

.. code-block:: python

   import jax.numpy as jnp
   from pycbc.distributions import Gaussian
   from pycbc.scheme import JAXScheme

   prior = Gaussian(x=(-2.0, 2.0), x_mean=0.25, x_var=1.5)
   with JAXScheme("cpu"):
       values = jnp.array([-0.5, 0.2, 0.9], dtype=jnp.float64)
       log_probability = prior.logpdf(x=values)

Smooth PDFs, log PDFs and inverse CDFs support JAX differentiation within their
valid domains. Bounds and discrete constraints are predicates. Fixed-sample
inverse CDFs select stored rows, preserve pairing, and are neither smooth nor
supported under ``jax.jit``. Concrete invalid tabulated or mass-ratio inverse
CDF inputs raise errors; traced invalid inputs produce NaNs.

Original implementation controls
--------------------------------

The default selects no original operations. For validation, pass independent
names through ``JAXScheme(reference_operations=...)`` or the command-line
``--jax-reference-operations`` option:

.. code-block:: python

   with JAXScheme(
       "cpu", reference_operations=("priors.Gaussian.logpdf",)
   ):
       original_log_probability = prior.logpdf(x=values)

Selected operations copy inputs to the host, call the original implementation,
and return its dtype and values to the input device. They require concrete
arrays and cannot be differentiated or compiled with JAX. Where the new JAX
API accepts batches and the original accepts only scalars, validation calls
the original scalar API for each broadcast row and reshapes the results.
Original NumPy vector APIs and errors remain unchanged.

.. list-table:: Validation boundaries
   :header-rows: 1
   :widths: 35 65

   * - Selector
     - Original calculation
   * - ``priors``
     - All supported prior validation boundaries.
   * - ``priors.<Family>.pdf``, ``.logpdf``, ``.cdfinv``, ``.contains``
     - One concrete family, such as ``Gaussian``, ``SinAngle``,
       ``UniformPowerLaw``, ``QfromUniformMass1Mass2`` or ``Arbitrary``.
   * - ``priors.BoundedDist.pdf``, ``.logpdf``, ``.cdfinv``, ``.contains``
     - The inherited public API for all bounded families.
   * - ``priors.Gaussian._normalcdf``, ``._normalcdfinv``, ``.cdf``
     - Unbounded error-function helpers or the complete truncated CDF.
   * - ``priors.Arbitrary.kde``
     - Fitted SciPy KDE density, with surrounding JAX coordinate transforms
       and log aggregation left active.
   * - ``priors.JointDistribution.logpdf``, ``.contains``, ``.within_constraints``
     - Joint evaluation, component membership or constraint combination.
   * - ``priors.Constraint.evaluate``, ``priors.SupernovaeConvexHull.evaluate``
     - Original expression evaluation or Delaunay membership.
   * - ``priors.FixedSamples.cdfinv``
     - Original scalar row selection and pairing.
   * - ``priors.DistributionFunctionFromFile.pdf``, ``.logpdf``, ``.cdf``, ``.cdfinv``
     - Original table interpolation and aggregation.

KDEs also use the independently controlled public ``Logit.logit`` and
``Logit.jacobian`` transforms; their selectors are
``transforms.Logit.logit`` and ``transforms.Logit.jacobian``.

Why results can differ
----------------------

Gaussian CDFs use JAX error-function primitives rather than SciPy's special
functions. Angular and logarithmic priors use JAX transcendental operations.
Arithmetic can round differently at each operation and after compiler fusion;
mathematical equivalence does not imply equal bytes. The notebook isolates
primitive differences before comparing complete prior evaluations.

The mass-ratio inverse CDF uses the same 1000-point grid. SciPy's cubic
``interp1d`` evaluates a spline representation; JAX evaluates cached
``CubicSpline`` coefficients in a power basis with Horner's rule. Different
representations and coefficient precision can change final rounding.
See `CubicSpline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html>`_.

KDE evaluation shares the fitted covariance and samples. JAX subtracts sample
coordinates before whitening and aggregates kernel weights with log-sum-exp.
SciPy's original density kernel whitens samples and queries separately,
subtracts them, and accumulates densities before PyCBC takes a logarithm.
These orders round differently; log-sum-exp also avoids density underflow.
See the `SciPy kernel <https://github.com/scipy/scipy/blob/v1.16.3/scipy/stats/_stats.pyx#L711-L794>`_.

Delaunay membership is a boundary decision. JAX evaluates cached barycentric
transforms in the parameter precision; SciPy evaluates its original simplex
search. The JAX tolerance follows SciPy's default ``100 * double eps``;
rounding of a point or transform can still change membership near a face.
See `find_simplex <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.find_simplex.html>`_.

Original validation retains dependency limitations. NumPy 2.4 and later reject
conversion of a non-scalar array to a Python scalar; the original scalar
``Arbitrary.logpdf`` has this limitation. The independent ``Arbitrary.kde``
boundary remains usable. See the
`NumPy release notes <https://numpy.org/doc/stable/release/2.4.0-notes.html#raise-typeerror-on-attempt-to-convert-array-with-ndim-0-to-scalar>`_.

The executed
:download:`numerical notebook <../examples/jax/jax_priors_numerical_differences.ipynb>`
uses synthetic inputs to compare default results, replay individual families
and stages, and prove exact original composition with dtype, shape and byte
assertions. Its measurements apply to its displayed library versions and
hardware.
