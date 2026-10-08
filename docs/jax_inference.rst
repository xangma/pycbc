JAX inference
=============

``JAXScheme`` selects JAX subclasses of the registered Gaussian-noise,
phase/time/polarization-marginalized, and relative-binning models. The ordinary
CPU classes retain their implementation. Explicit custom model subclasses and
custom waveform generators retain their overrides.

CBC waveforms use diffGW by default. Its supported models and parameter limits
apply; an unsupported model raises an error. ``waveform`` selects the original
waveform provider explicitly for validation. Static prior grids, configuration,
NumPy random streams, and sampler output remain host interfaces. Likelihood
arrays, detector projection, weighting, and numerical marginalization use JAX.

Construct the data and model inside the processing context::

    from pycbc.scheme import JAXScheme
    from pycbc.inference.models.gaussian_noise import GaussianNoise

    with JAXScheme():
        model = GaussianNoise(variable_params, data, low_frequency_cutoff,
                              psds=psds, static_params=static_params)
        model.update(**parameters)
        loglr = model.loglr

``batched_loglr`` on Gaussian and phase-marginalized Gaussian models batches
one-dimensional detector-frame extrinsics: right ascension, declination,
polarization, and coalescence time. Intrinsic waveform parameters remain scalar.
All detectors must share sample rate and length. Batched evaluation rejects
sampling/waveform transforms, waveform gates, and recalibration; scalar model
evaluation retains those interfaces.

Absolute GPS values must have double precision. A float32 GPS timestamp has
already lost subsecond information before the likelihood receives it. Default
projection subtracts the epoch before adding a small delay. ``time_shift``
restores the original order of those operations as well as the original shift
kernel. Relative-bin kernels retain the original compiled phase constant.

Validation controls
-------------------

The default is device execution. ``JAXScheme(reference_operations=(...))`` or
``--jax-reference-operations`` opts into original implementations. Reconstruct a
model when changing controls so its PSD weights and relative-bin summaries are
prepared under the requested policy.

.. list-table:: Inference boundaries
   :header-rows: 1
   :widths: 32 68

   * - Control
     - Original calculation
   * - ``inference_projection``
     - Array polarization combination and relative-model polarization/inclination responses.
   * - ``inference_whitening``
     - Gaussian PSD/weight preparation and in-place template weighting.
   * - ``inference_inner`` or ``inner``
     - Compiled Array inner products, including weighting/likelihood fusion.
   * - ``inference_time_interpolation``
     - TimeSeries interpolation used by time-marginalized and dominant-mode relative models.
   * - ``inference_marginalization``
     - Phase, distance, and vector likelihood marginalization; log-weight normalization.
   * - ``inference_interpolant``
     - SciPy evaluation of the static distance-marginalization spline.
   * - ``inference_sampling``
     - Original phase/log-weight reconstruction preparation and weighted sampling with the caller's NumPy RNG stream.
   * - ``relbin_summary``
     - Original per-bin summary products.
   * - ``relbin_likelihood`` / ``relbin_snr``
     - Original compiled relative-bin likelihood / SNR kernels, including the dominant-mode post-kernel likelihood products.
   * - ``relbin_snr_normalization``
     - NumPy magnitude and scalar normalization following the dominant-mode SNR kernel.

Existing ``divide``, ``sum``, ``cumsum``, ``fft``, ``ifft``, ``correlate``,
``squared_norm``, ``waveform``, and ``time_shift`` controls remain independent
within their inference consumers. A reference route copies values to the host
and publishes its result on the input device. These diagnostic routes are slow
and cannot be differentiated or run inside ``jax.jit``.

Default numerical kernels support gradients. Model caches, prior draws,
parameter reconstruction, and sampler APIs are stateful host boundaries; this
is not a promise that an entire sampler can be JIT-compiled.

See :doc:`jax_inference_numerical_differences` for measured examples and the
executed notebook.
