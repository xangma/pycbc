.. _jax-waveform:

JAX waveforms and template banks
================================

Inside ``JAXScheme``, astrophysical waveform synthesis uses the diffGW JAX
interface. Ordinary CPU generation retains its original implementations.
The adapter requires diffGW with its JAX extra alongside PyCBC's scientific
requirements. From a diffGW source checkout, install it with::

    python -m pip install '.[jax]'

The original ``waveform`` validation selector works without diffGW.
See :doc:`jax_arrays` for JAX installation and device selection.

Supported generation
---------------------

The adapter supports frequency-domain ``TaylorF2``, ``IMRPhenomD`` and
``IMRPhenomXAS``, including ``get_fd_waveform``,
``get_fd_waveform_sequence`` and the waveform-filter APIs. Inputs use PyCBC
units: solar masses, megaparsecs, radians and hertz. Synthesis uses float64;
output storage supports complex64 and complex128.

Supported inputs have finite values, positive masses and lower frequency,
aligned spins in ``[-1, 1]``, and the standard model options. For TaylorF2,
``phase_order`` and ``spin_order`` accept their default or ``7``;
``amplitude_order`` accepts its default or ``0``. Tidal deformabilities must
be zero. Nonzero reference frequency, transverse spins, eccentricity,
nondefault mode selections and unsupported model/options raise an error.
Time-domain and separate-mode bridging are currently available only through
the original waveform validation route. Availability helpers report these
three FD/filter models and empty TD/mode lists by default; selecting
``waveform`` reports the original CPU provider's available models instead.

``coa_phase`` follows the original convention. TaylorF2 maps it to diffGW's
orbital phase plus ``pi / 2`` to account for the amplitude sign. Phenom models
use the lower frequency as their reference frequency; the sequence API uses
its first requested frequency. This maps the original meaning of
``f_ref=0`` to diffGW's explicit reference-frequency convention.

.. code-block:: python

   from pycbc.scheme import JAXScheme
   from pycbc.waveform import get_fd_waveform

   with JAXScheme("cpu"):
       plus, cross = get_fd_waveform(
           approximant="TaylorF2", mass1=30, mass2=20,
           spin1z=0.1, spin2z=-0.2, distance=100,
           delta_f=2, f_lower=20, f_final=128)

Decompression and tapers
------------------------

Compressed banks use the stored interpolation method. Native host-staged
batch decompression executes the original compiled interpolators and then
transfers the samples to the selected device. ``device_linear`` explicitly
selects direct on-device linear interpolation on CUDA; higher-order stored
interpolation cannot be silently changed to linear. The public
``fd_decompress`` JAX route retains native interpolation and output-buffer
contracts.

``td_taper`` and ``fd_taper`` use an on-device periodic Kaiser window and
return a copy, as their original APIs do. Original single-precision tapering
can raise a precision-mismatch error because its window is float64; the
validation route preserves that original behavior.

Original-implementation validation
----------------------------------

Select each numerical boundary independently, for example::

    with JAXScheme("cpu", reference_operations=("waveform",)):
        plus, cross = get_fd_waveform(
            approximant="TaylorF2", mass1=30, mass2=20,
            delta_f=2, f_lower=20, f_final=128)

``waveform`` runs the corresponding original CPU generation function,
including both polarizations and its metadata. It also permits original
models outside the diffGW adapter's supported set. ``decompress`` selects
original decompression, including the native recurrence when a device-linear
batch was requested. ``td_taper`` and ``fd_taper`` select the corresponding
original taper independently. ``time_shift`` controls frequency-domain time
shifts; see :doc:`jax_filtering` for other filtering boundaries. Dedicated JAX
executables accept the same names through ``--jax-reference-operations``.

Pure original LAL frequency-domain, sequence and FD filter generators run
directly with explicit host parameters, preserving the caller's native-library
environment. Specialized filters, other generation and taper validation run
in an isolated CPU process. Neither
route changes the active JAX scheme; results use the selected device.
These routes are deliberately slow and cannot be traced or differentiated.
The executed examples assert matching original sample bytes, precision, grid
and metadata for the demonstrated models and operations. Native waveform
rounding can depend on library loading; direct FD validation uses the
original wrappers in the caller's environment. Other model/plugin validation
requires the same comparison in the intended runtime environment.

Numerical differences
----------------------

After convention mapping, fixed float64 examples show small residual
waveform differences. TaylorF2 evaluates the same post-Newtonian terms with
separate arithmetic. For example, diffGW expresses its 3PN logarithm as
``log(4*v)``, whereas the original public phasing coefficients include
``log(4)`` in the constant coefficient and use ``log(v)``. These equivalent
expressions and separately rounded coefficients need not give identical
floating-point phase. Polarization construction also evaluates trigonometric
functions in the respective numerical libraries. A complex64 output adds a
final storage rounding step.

Phenom's reference-frequency convention explains the large constant phase
offset in the raw-provider comparison. After that mapping, PhenomD still has
arithmetic differences: its Table V fits use expanded powers in diffGW and
nested products in the original implementation. Its inspiral phase forms
``(pi*Mf)**(1/3)`` directly; the original caches sixth-root products of ``Mf``
and ``pi`` separately. Holding the coefficients fixed isolates the resulting
phase rounding. Intermediate amplitude coefficients use a linear solve in
diffGW and closed-form collocation expressions in the original; their
polynomial evaluation also groups products differently. See the
`original PhenomD source
<https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_d__internals_8c_source.html>`_.

PhenomXAS has additional fit arithmetic boundaries. Its inspiral phase-fit
matrix uses separate fractional powers; the original uses a cube root and its
cached square. Its intermediate phase fit uses inverse powers of ``Mf``;
the original scales these columns by the ringdown frequency before solving
and rescales the coefficients afterwards. The same fitted polynomial gives
different rounded coefficients in these two matrix coordinates. The original
uses GSL LU routines and diffGW uses JAX's linear solver. Intermediate
amplitude evaluates a polynomial in a normalized frequency coordinate in
diffGW and in ``Mf`` in the original. See the `original PhenomX fit source
<https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__internals_8c_source.html>`_
and `amplitude expression
<https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__intermediate_8c_source.html>`_.

These are demonstrated arithmetic contributors, rather than a unique
attribution of every differing waveform bin. The notebook holds coefficients,
fit inputs or polynomial targets fixed to isolate each boundary, and reports
mapped waveform residuals without a time or phase fit. An algebraic rewrite
can also agree exactly at a particular input: the PhenomD peak-slope forms do
so in the example. Original PyCBC/LAL waveforms remain the validation
reference; ``waveform`` restores their sample bytes and metadata. The examples
cover the stated inputs and versions, rather than each model's entire physical
domain.

Device-linear decompression evaluates amplitude and phase interpolation and
``cos``/``sin`` at every output bin in the output precision. Original compiled
linear decompression uses double intermediate slopes and a complex rotation
recurrence between knots, with periodic trigonometric reseeding. Both follow
the same linear amplitude/phase interpolation but round in different
sequences. JAX tapers evaluate their Kaiser expression and Bessel function on
device in the input precision; the original SciPy window is float64.

The executed :download:`waveform comparison notebook
<../examples/jax/jax_waveform.ipynb>` holds inputs and grids fixed, records package
versions and source hashes, isolates arithmetic boundaries, and asserts exact
original waveform, decompression and taper controls.

