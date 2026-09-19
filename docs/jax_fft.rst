.. _jax-fft:

JAX Fourier transforms
======================

Use the standard PyCBC FFT interfaces inside ``JAXScheme``. See
:doc:`jax_arrays` for installation, device selection and array semantics.
Forward transforms support real-to-complex and complex-to-complex inputs;
inverse transforms support complex-to-real and complex-to-complex outputs.
Input and output precisions must match: ``float32`` pairs with ``complex64``
and ``float64`` with ``complex128``. Input and output buffers must be distinct;
the JAX backend rejects in-place transforms.

Functional interface and normalization
---------------------------------------

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
-------------------

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
------------------

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

