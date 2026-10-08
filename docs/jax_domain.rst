JAX parameter conversions and transforms
========================================

The supported mass, spin, tidal, Cartesian/spherical, and parameter-transform
APIs accept raw JAX arrays. NumPy and scalar calls retain their original CPU
behavior. Mixed inputs use the first JAX array's precision and device, with
promotion to the corresponding complex precision when an input is complex;
interpolation tables are cached separately for each concrete array placement.
The ordinary JAX path does not select original CPU validation kernels.

For example::

    import jax.numpy as jnp
    from pycbc import conversions
    from pycbc.scheme import JAXScheme

    with JAXScheme("cpu"):
        mass1 = jnp.array([30.0, 15.0])
        mass2 = jnp.array([20.0, 10.0])
        chirp_mass = conversions.mchirp_from_mass1_mass2(mass1, mass2)

Use ``"cuda"`` to select an available CUDA device. Raw JAX inputs retain their
own device even when a different device is selected by an enclosing scheme.
Host interpolation-table construction and existing EOS/ISSO routines remain
host operations; this API does not make every scientific helper device-only.

Scope and constraints
---------------------

``mass2_from_mchirp_mass1`` uses a real cubic solution, and
``mass_from_knownmass_eta`` uses quadratic roots. The former requires real,
nonnegative chirp mass and positive known mass. Integer tensor inputs are
converted to floating point. QNM conversions use the installed ``pykerr``
spline data and require scalar integer mode labels.

For predefined Astropy cosmologies, ``DistToZ`` and
``ComovingVolInterpolator`` interpolate JAX arrays using their existing
Astropy-generated grids. JAX distances must be finite, nonnegative, and within
``default_maxz``. JAX volumes must be finite, positive, and within the grid
from redshift 0.001 to ``default_maxz``. Original validation routes retain the
original interpolation and Astropy fallback rules.

``LambdaFromTOVFile`` accepts JAX mass arrays and preserves the original
below-table endpoint and above-maximum zero conventions. Custom transforms
support arithmetic, named conversion functions, and the supported elementary
functions. Expressions outside that supported subset use the existing
``FieldArray`` host evaluator and do not preserve JAX differentiation.

Differentiable formulas support ``jax.grad``. Some APIs perform eager
value-dependent validation and do not accept ``jax.jit`` tracers. Compiled
``Logit`` calls must satisfy their configured bounds; eager calls retain bounds
validation. A fresh TOV transform can be compiled before its first eager call.

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

New vector TOV transforms, conditioned-bound membership, and Logit Jacobians
validate by replaying the original scalar API once per entry. Ordinary NumPy
calls retain their original array support and errors. Original dependency
errors also propagate: for example, older ``pykerr`` versions can fail with
NumPy versions that removed ``numpy.float``. A validation control does not
repair that original implementation.

.. toctree::
   :maxdepth: 1

   jax_domain_numerical_differences
