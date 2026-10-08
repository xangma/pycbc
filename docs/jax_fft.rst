.. _jax-fft:

JAX spectral calculations
=========================

Fourier transforms
------------------

Use the standard PyCBC FFT interfaces inside ``JAXScheme``. See
:doc:`jax_arrays` for installation, device selection and array semantics.
Forward transforms support real-to-complex and complex-to-complex inputs;
inverse transforms support complex-to-real and complex-to-complex outputs.
Input and output precisions must match: ``float32`` pairs with ``complex64``
and ``float64`` with ``complex128``. Input and output buffers must be distinct;
the JAX backend rejects in-place transforms.

Functional interface and normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fft(invec, outvec)`` and ``ifft(invec, outvec)`` write into the supplied
PyCBC output container. For plain ``Array`` buffers, both transforms are
unnormalized: a forward/inverse pair returns ``N * input`` for length ``N``.

The functional interface preserves PyCBC's series normalization and metadata.
A ``TimeSeries`` input supplies a forward scale of ``delta_t``; a
``FrequencySeries`` input supplies an inverse scale of ``delta_f``. The
spacing relation is ``delta_f = 1 / (N * delta_t)``. Consequently, a
time/frequency-series roundtrip reconstructs the input within the transform's
numerical error rather than multiplying it by ``N``. Epochs are preserved.

.. code-block:: python

   import numpy as np
   from pycbc.fft import fft, ifft
   from pycbc.scheme import JAXScheme
   from pycbc.types import TimeSeries, FrequencySeries

   with JAXScheme("cpu"):
       samples = TimeSeries(np.arange(8, dtype=np.float64), delta_t=0.125)
       spectrum = FrequencySeries(np.zeros(5, dtype=np.complex128), delta_f=1)
       restored = TimeSeries(np.zeros(8), delta_t=samples.delta_t)
       fft(samples, spectrum)
       ifft(spectrum, restored)

Plans and batching
~~~~~~~~~~~~~~~~~~

``FFT`` and ``IFFT`` create reusable plans bound to input and output buffers;
``execute()`` performs the transform. Plan execution supplies raw,
unnormalized transforms. Apply any series scale required by the calling
calculation explicitly; the functional interface above applies it for you.

For ``nbatch > 1``, supply the logical transform ``size=N`` and flattened,
consecutive rows. Complex-to-complex input and output each have length
``nbatch * N``. Real forward input has length ``nbatch * N`` and complex
output has length ``nbatch * (N // 2 + 1)``; the inverse uses those lengths
in reverse. Separate transforms retain their own grids and normalization.

.. code-block:: python

   from pycbc.fft import FFT, IFFT
   from pycbc.types import Array

   with JAXScheme("cpu"):
       values = Array(np.arange(16, dtype=np.float64))
       spectra = Array(np.zeros(10, dtype=np.complex128))
       restored = Array(np.zeros(16, dtype=np.float64))
       forward = FFT(values, spectra, nbatch=2, size=8)
       inverse = IFFT(spectra, restored, nbatch=2, size=8)
       forward.execute()
       inverse.execute()  # Each restored row is 8 times its input row.

Native validation
~~~~~~~~~~~~~~~~~

The default uses JAX transforms. Select ``reference_operations=("fft",)``
or ``("ifft",)`` in ``JAXScheme`` to run that direction through PyCBC's
selected original CPU FFT backend. Both names can be selected together.
The CLI equivalent is ``--jax-reference-operations fft,ifft`` with a JAX
processing scheme.

These routes convert the transform buffers to host storage, execute the CPU
backend and return the result to the JAX output buffer. They can be slower and
prevent JAX tracing or differentiation through the selected operation. Native
plans inherit the selected CPU backend's batching capabilities. The switches
apply to PyCBC's functional and plan interfaces; direct ``jax_fft`` and related
raw JAX helpers continue to use JAX. See
:doc:`jax_fft_numerical_differences` for controlled comparisons.

.. _jax-psd:

Power spectral densities
------------------------

Use the standard ``pycbc.psd`` functions inside ``JAXScheme`` to select JAX
PSD estimation, inverse-spectrum truncation and interpolation. Fourier
conventions are described above; ordinary CPU use retains its existing
implementations.

Estimating and preparing a PSD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~

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
