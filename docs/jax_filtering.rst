.. _jax-filtering:

JAX conditioning and filtering
==============================

Use ``JAXScheme`` with the standard PyCBC filtering interfaces. See
:doc:`jax_arrays` for installation and devices, :doc:`jax_fft` for transform
conventions, and :ref:`jax psd <jax-psd>` for PSD preparation. The ordinary CPU scheme
retains its existing implementations.

Conditioning
------------

``highpass`` and ``lowpass`` apply Butterworth filters to real time series.
JAX performs the filter sections and forward/reverse passes on the selected
device. Its parallel prefix scan composes each sample's affine state update;
double-precision state histories are used even for float32 samples. This
preserves the recurrence in exact arithmetic while changing floating-point
grouping. A JAX implementation following LAL's coefficients is still a JAX
calculation, rather than execution of the original LAL routine.

``highpass_mode="lal-coefficients"`` uses that coefficient construction;
``"closed-form"`` uses algebraically simplified coefficients. Both modes
perform associative scans on device. The constructor defaults to
``"lal-coefficients"``; the processing CLI selects it for CPU and
``"closed-form"`` for CUDA.

The FIR interfaces use Kaiser-windowed filter design. ``lfilter`` follows
PyCBC's two conventions: short records use causal filtering with promoted
calculation precision, whereas records of at least 128 samples use circular
FFT filtering in the input precision, following the standard chunk handling
for large records. ``fir_zero_filter`` applies the standard
edge zeroing and delay correction. Filter coefficients, transforms and
normalization each have their own numerical boundaries.

JAX ``resample_to_delta_t`` supports integer downsampling with
``method="ldas"``. Specify the method explicitly: the ordinary API's default
Butterworth resampling is unavailable in the default JAX implementation.

.. code-block:: python

   import numpy as np
   from pycbc.scheme import JAXScheme
   from pycbc.types import TimeSeries
   from pycbc.filter import highpass, resample_to_delta_t

   values = np.random.default_rng(1729).standard_normal(1024).astype(np.float32)
   with JAXScheme("cpu"):
       samples = TimeSeries(values, delta_t=1 / 1024)
       filtered = highpass(samples, 30, filter_order=4)
       downsampled = resample_to_delta_t(filtered, 1 / 512, method="ldas")

Matched filtering
-----------------

``matched_filter`` produces the normalized complex SNR time series.
``matched_filter_core`` exposes the unnormalized time series, frequency-domain
correlation and normalization separately. Template and data grids must agree;
PSD and frequency cutoffs determine the weighting and included bins.
``sigmasq`` computes template power and ``sigmasq_series`` its cumulative
frequency contribution. Supplying a template normalization to the core holds
that boundary fixed when comparing correlation and transform stages.

The following uses synthetic frequency arrays to illustrate the interface;
it is not a physical waveform-generation example.

.. code-block:: python

   from pycbc.types import FrequencySeries
   from pycbc.filter import matched_filter

   rng = np.random.default_rng(1729)
   template_values = (rng.normal(size=513) + 1j * rng.normal(size=513))
   data_values = template_values + 0.1 * (
       rng.normal(size=513) + 1j * rng.normal(size=513))
   with JAXScheme("cpu"):
       template = FrequencySeries(template_values.astype(np.complex64), delta_f=1)
       data = FrequencySeries(data_values.astype(np.complex64), delta_f=1)
       psd = FrequencySeries(np.ones(513, dtype=np.float32), delta_f=1)
       snr = matched_filter(template, data, psd=psd,
                            low_frequency_cutoff=20, high_frequency_cutoff=200)

Autocorrelation and vetoes
--------------------------

JAX time series support ``calculate_acf`` and ``calculate_acl``. Their
centering, transforms, normalization and cumulative sums can affect agreement
with the original CPU values and the selected correlation-length window.
Event threshold and clustering decisions can change when magnitude arithmetic
rounds across a strict threshold.

Power chi-square partitions PSD-weighted template power into frequency bins
and evaluates their contributions at selected samples. Point calculations
retain JAX execution by default; following the original phase recurrence is
distinct from calling its compiled CPU kernel. Independently evaluated
Fourier phases use a different finite-precision calculation.

The sine-Gaussian veto evaluates its usual frequency tiles and activation
criteria. Its internal basis functions support the veto calculation; this
does not select a separate astrophysical waveform provider.

Original-implementation validation
----------------------------------

``reference_operations`` selects original calculations independently; its
default is empty. For example, keep all other filtering stages on JAX while
validating the highpass against the original LAL routine:

.. code-block:: python

   with JAXScheme("cpu", reference_operations=("highpass",)):
       samples = TimeSeries(values, delta_t=1 / 1024)
       filtered = highpass(samples, 30, filter_order=4)

The processing CLI accepts the same comma-separated names through
``--jax-reference-operations``. These controls copy data to the host and
return array results to the selected device. Whole-stage controls run original
CPU routines in an isolated process. They are deliberately slow, and JAX
tracing and differentiation cannot cross a native boundary.

.. list-table:: Filtering validation boundaries
   :header-rows: 1
   :widths: 45 55

   * - Names
     - Original calculation selected
   * - ``highpass``, ``lowpass``
     - Complete LAL Butterworth filter
   * - ``firwin``
     - SciPy FIR coefficients
   * - ``lfilter``, ``fir_zero_filter``, ``resample``
     - Complete corresponding CPU filter or resampling operation
   * - ``correlate``, ``divide``
     - Complex correlation kernel; NumPy array division
   * - ``fft``, ``ifft``, ``weighted_inner``
     - Transform or PSD-weighted template norm; see the FFT and array guides
   * - ``autocorrelation``
     - Complete autocorrelation function or length
   * - ``autocorrelation_mean``, ``autocorrelation_variance``
     - NumPy centering mean or variance, retaining the other ACF stages
   * - ``threshold_cluster``
     - Original CPU clustering class
   * - ``shift_sum``, ``chisq_accum_bin``
     - Compiled point sum or bin accumulation
   * - ``power_chisq_bins``, ``power_chisq_at_points``
     - Complete bin selection or point chi-square calculation
   * - ``sg_basis``, ``time_shift``, ``sgchisq``
     - Internal sine-Gaussian tile, frequency-domain shift or complete veto

Selecting ``fft`` and ``ifft`` inside FIR filtering leaves its surrounding
JAX spectral multiplication and normalization in place. Select the complete
filter to restore that whole stage. For matched filtering with frequency
inputs, the notebook combines ``correlate``, ``divide``, ``ifft`` and
``weighted_inner`` while checking each boundary separately.

See :doc:`jax_filtering_numerical_differences` for deterministic comparisons,
controlled numerical examples and validation boundaries. Small examples do
not establish parity of a complete inspiral or live search.

