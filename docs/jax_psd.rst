.. _jax-psd:

JAX power spectral densities
============================

Use the standard ``pycbc.psd`` functions inside ``JAXScheme`` to select JAX
PSD estimation, inverse-spectrum truncation and interpolation. See
:doc:`jax_arrays` for installation, devices and precision, and :doc:`jax_fft`
for Fourier-transform conventions. Ordinary CPU use retains its existing
implementations.

Estimating and preparing a PSD
------------------------------

``welch`` estimates a one-sided PSD from windowed time segments. It supports
``mean``, ``median`` and ``median-mean`` averaging and a Hann window or an
explicit window array. Segment length and stride are measured in samples;
the output spacing is ``1 / (seg_len * delta_t)``. Default output precision
matches the input. Segmentation, trimming and exact-fit checks follow the
standard API.

The lower-level ``welch_jax(..., wide_fft=True)`` explicitly uses double
precision for the transform and subsequent PSD arithmetic, after multiplying
samples and window in the input precision. This changes the numerical policy;
the standard default retains input precision.

``interpolate`` linearly resamples a frequency series to the requested
``delta_f``. Specify ``length`` when the consuming calculation requires an
exact number of bins. Values outside the supplied frequency range use the
endpoint value; output precision and epoch follow the input.

``inverse_spectrum_truncation`` shortens the time-domain inverse-spectrum
response to ``max_filter_len`` samples. ``which_spectrum="invasd"`` truncates
the inverse ASD; ``"invpsd"`` truncates the inverse PSD. The standard API
defaults to hard truncation; ``trunc_method="hann"`` applies a Hann taper.
Low-frequency cutoff and fill options control the inverse-spectrum entries
before truncation. The returned PSD has epoch zero, following the original
truncation routine; the input epoch is unchanged.

.. code-block:: python

   import numpy as np
   from pycbc.scheme import JAXScheme
   from pycbc.types import TimeSeries
   from pycbc.psd import welch, interpolate, inverse_spectrum_truncation

   values = np.random.default_rng(1729).standard_normal(256).astype(np.float32)
   with JAXScheme("cpu"):
       samples = TimeSeries(values, delta_t=1 / 256)
       psd = welch(samples, seg_len=64, seg_stride=32, avg_method="median")
       resampled = interpolate(psd, delta_f=2, length=65)
       conditioned = inverse_spectrum_truncation(
           resampled, max_filter_len=16, low_frequency_cutoff=8,
           which_spectrum="invasd", trunc_method="hann")

Analytical models
------------------

``from_string`` selects an analytical PSD by name. Models listed by
``get_jax_psd_list()`` have JAX implementations; other models retain their
standard library path. Model formulas, data-table availability and numerical
agreement are model-specific. Table-backed models obtain their reference data
from the installed scientific dependencies.

.. code-block:: python

   from pycbc.psd import from_string, get_jax_psd_list

   print(get_jax_psd_list())
   with JAXScheme("cpu"):
       model = from_string("aLIGOZeroDetHighPower", 33, 4, 8)

Native validation
------------------

The default uses the JAX implementations. Select any of ``welch``,
``inverse_spectrum_truncation``, ``interpolate`` or ``analytical_psd`` with
``JAXScheme(..., reference_operations=(...))`` to restore that original CPU
calculation. For example, ``reference_operations=("welch",)`` keeps other
available JAX operations selected. The CLI equivalent is
``--jax-reference-operations welch`` with a JAX processing scheme.

The ``fft`` and ``ifft`` selections also control transforms inside JAX Welch
estimation and inverse-spectrum truncation. They replace only that transform
direction, retaining the surrounding JAX PSD arithmetic.

PSD native routes execute the original CPU implementation in a separate
process, then return its result to the selected JAX device. Startup and host
transfers can make them substantially slower; tracing and differentiation
cannot cross this boundary. They apply to the PyCBC PSD entry points; raw
``analytical_psd_jax`` frequency-array evaluation remains JAX. See
:doc:`jax_psd_numerical_differences` for controlled examples and numerical
limits.

.. toctree::
   :maxdepth: 1

   jax_psd_numerical_differences
