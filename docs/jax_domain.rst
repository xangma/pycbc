.. _jax-domain:

JAX model parameters
====================

Supported parameter APIs accept raw JAX arrays. NumPy and scalar calls retain
their original CPU behavior. Mixed inputs use the first JAX array's precision
and device, with promotion to the corresponding complex precision when an
input is complex. Interpolation tables are cached separately for each concrete
array placement. The ordinary JAX path does not select original CPU validation
kernels.

Creating arrays inside ``JAXScheme("cpu")`` or ``JAXScheme("cuda")`` selects
its default device. Raw JAX inputs retain their own device even when a different
device is selected by an enclosing scheme.

.. _jax-conversions:

Conversions and coordinates
---------------------------

Supported mass, spin, tidal and Cartesian/spherical conversions accept JAX
arrays. For example::

    import jax.numpy as jnp
    from pycbc import conversions
    from pycbc.scheme import JAXScheme

    with JAXScheme("cpu"):
        mass1 = jnp.array([30.0, 15.0])
        mass2 = jnp.array([20.0, 10.0])
        chirp_mass = conversions.mchirp_from_mass1_mass2(mass1, mass2)

``mass2_from_mchirp_mass1`` uses a real cubic solution, and
``mass_from_knownmass_eta`` uses quadratic roots. The former requires real,
nonnegative chirp mass and positive known mass. Integer tensor inputs are
converted to floating point. QNM conversions use the installed ``pykerr``
spline data and require scalar integer mode labels. Existing EOS/ISSO routines
remain host operations; this API does not make every scientific helper
device-only.

.. _jax-cosmology:

Cosmological interpolation
--------------------------

For predefined Astropy cosmologies, ``DistToZ`` and
``ComovingVolInterpolator`` interpolate JAX arrays using their existing
Astropy-generated grids. JAX distances must be finite, nonnegative, and within
``default_maxz``. JAX volumes must be finite, positive, and within the grid
from redshift 0.001 to ``default_maxz``. Original validation routes retain the
original interpolation and Astropy fallback rules. Interpolation-table
construction remains a host operation.

.. _jax-transforms:

Parameter transforms
--------------------

``LambdaFromTOVFile`` accepts JAX mass arrays and preserves the original
below-table endpoint and above-maximum zero conventions. Custom transforms
support arithmetic, named conversion functions, and the supported elementary
functions. Expressions outside that supported subset use the existing
``FieldArray`` host evaluator and do not preserve JAX differentiation.

Differentiable formulas support ``jax.grad``. Some APIs perform eager
value-dependent validation and do not accept ``jax.jit`` tracers. Compiled
``Logit`` calls must satisfy their configured bounds; eager calls retain bounds
validation. A fresh TOV transform can be compiled before its first eager call.

.. _jax-domain-validation:

Original validation controls
----------------------------

Select a category (``conversions``, ``coordinates``, ``cosmology``,
``transforms``, or ``boundaries``) or one independent operation::

    with JAXScheme("cpu", reference_operations=(
        "conversions.mass2_from_mchirp_mass1",
    )):
        mass2 = conversions.mass2_from_mchirp_mass1(chirp_mass, mass1)

The controls copy arguments to writable NumPy buffers, execute the original
implementation, and return its numeric result with its original dtype on the
input device. They preserve untouched input arrays and nonnumeric metadata.
They are deliberately slow and require concrete inputs, so they cannot be
used through ``jax.jit`` or ``jax.grad``. Double-precision native results
require ``jax_enable_x64=True``; validation rejects silent precision loss.
Supported names are listed in ``pycbc.domain_jax.DOMAIN_REFERENCE_NAMES``.
The CLI accepts the same names through ``--jax-reference-operations``; the
default selection is empty.

New vector TOV transforms, conditioned-bound membership, and Logit Jacobians
validate by replaying the original scalar API once per entry. Ordinary NumPy
calls retain their original array support and errors. Original dependency
errors also propagate: for example, older ``pykerr`` versions can fail with
NumPy versions that removed ``numpy.float``. A validation control does not
repair that original implementation.

See :doc:`jax_domain_numerical_differences` for controlled conversions and
transform comparisons and their executed notebook.

.. _jax-priors:

Prior distributions
-------------------

Supported distributions evaluate PDFs, log PDFs, inverse CDFs and constraints
on the raw JAX input device. NumPy and Python inputs retain their original
implementations. Broadcast parameter grids are supported by uniform, Gaussian,
power-law, angular, mass-ratio and
log-uniform priors, their joint distributions, fitted KDEs, tabulated priors
and convex-hull constraints.

Use floating-point parameter arrays. Their precision determines the JAX
calculation; original validation may return a wider native dtype. Fitted
constants follow the first JAX parameter array and are cached separately by
precision and device. KDE fitting and interpolation-table preparation retain
their original host implementations.

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

Prior validation controls
~~~~~~~~~~~~~~~~~~~~~~~~~

Select one prior family or calculation independently through
``reference_operations``, for example:

.. code-block:: python

   with JAXScheme(
       "cpu", reference_operations=("priors.Gaussian.logpdf",)
   ):
       original_log_probability = prior.logpdf(x=values)

The same host-transfer and concrete-input restrictions apply as for parameter
validation above. Where the JAX API accepts batches and the original accepts
only scalars, validation calls the original scalar API for each broadcast row
and reshapes the results. Original NumPy vector APIs and errors remain
unchanged.

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

.. _jax-priors-numerical-differences:

Why prior results can differ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
