.. _jax-strain:

JAX strain conditioning
=======================

Use the standard strain APIs inside ``pycbc.scheme.JAXScheme`` to keep
conditioning arrays on the selected JAX device::

    from pycbc import scheme
    from pycbc.strain import gate_data

    with scheme.JAXScheme("cpu"):
        gate_data(strain, [(time, half_width, taper_duration)])

``StrainBuffer.advance`` conditions and stitches new samples into its rolling
buffer. Its default JAX path applies the highpass FIR, dynamic-range scaling,
float32 conversion and LDAS downsampling on device. CUDA autogating keeps
whitening, peak selection and gate application resident when supported.
``StrainSegments.fourier_segments`` batches complete segments; padded segments
retain the standard per-segment route.

``StrainBuffer.overwhitened_data(delta_f)`` prepares PSD grids and caches the
whitened spectrum. ``preload_overwhitened_data(delta_fs)`` prepares several
requested resolutions. JAX invalidates its derived PSD and spectrum caches
when the PSD or relevant validation controls change. Live Welch estimation
retains the input precision; single-precision strain produces a float32 PSD.

Original-kernel validation
--------------------------

Select the original implementations independently, for example::

    with scheme.JAXScheme("cpu", reference_operations=("fft", "divide")):
        spectrum = buffer.overwhitened_data(delta_f)

``fft`` and ``ifft`` select the installed original CPU transform backend.
``divide`` selects original array division. ``firwin``, ``fir_zero_filter``
and ``resample`` isolate coefficient design, FIR application and downsampling.
``lfilter`` selects the original causal filter within a staged FIR path.
``welch``, ``interpolate`` and ``inverse_spectrum_truncation`` select complete
PSD stages. ``gate_data`` selects original Tukey gating;
``findchirp_cluster`` selects original peak clustering;
``detect_loud_glitches`` selects the complete original glitch calculation.

Selected operations use explicit host transfers and original CPU routines.
Fused JAX paths run in stages where necessary to reach a selected boundary;
the default remains on device. Recreate the buffer or recalculate its base PSD
when changing the Welch implementation. A PSD already supplied to whitening
is retained as an input.

These controls compare a fixed input against the installed CPU implementation.
They are slow diagnostics and cannot be traced or differentiated through JAX.
See :doc:`jax_strain_numerical_differences` for causes, fixed-input examples and
exact native assertions.

.. toctree::
   :hidden:

   jax_strain_numerical_differences
