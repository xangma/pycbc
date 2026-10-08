.. _jax-strain-numerical-differences:

JAX strain numerical differences
=================================

The :download:`strain comparison notebook
<../examples/jax/jax_strain_numerical_differences.ipynb>` generates a temporary
GWF, advances the public strain buffer, and compares conditioning, Welch PSD,
both truncated PSD grids and the whitened spectrum. It fixes sample bytes,
precision, spacing, epochs and options. The temporary frame is removed after
execution; no downloaded data is needed.

Default JAX runs conditioning on the selected device. Different results can
come from FIR coefficient construction, transform ordering, PSD arithmetic,
complex division and Tukey cosine values. The :doc:`filtering
<jax_filtering_numerical_differences>`, :doc:`FFT
<jax_fft_numerical_differences>`, :doc:`PSD
<jax_psd_numerical_differences>` and :doc:`array
<jax_array_numerical_differences>` examples isolate those boundaries.
Single-precision live input retains single-precision PSD storage and Welch
calculation; x64 availability does not silently widen its transforms.

Whitening operation order
--------------------------

The original padded path scales its forward transform by ``delta_t``, divides
by the PSD, and scales an unnormalized inverse by ``delta_f``. The fused JAX
path cancels those normalization factors algebraically before the inverse.
Finite-precision arithmetic rounds the original intermediate scale and
division separately, so this cancellation can change the result even when the
real-arithmetic expressions agree.

The notebook isolates that cause without an FFT. For float32 value
``0.86845714``, scale ``0.2`` and denominator ``1.142317``, scaling before
division gives ``0.15205187``; dividing before scaling gives ``0.15205185`` on
the displayed NumPy installation. Full whitening can additionally differ
because of the transform and division implementations.

Selecting ``fft``, ``ifft``, ``divide`` and ``gate_data`` uses the original
staged order with the original kernels. Adding ``firwin``, ``fir_zero_filter``,
``resample``, ``welch``, ``interpolate`` and
``inverse_spectrum_truncation`` restores all preceding stages in the notebook's
public GWF-to-spectrum example. Every intermediate dtype and byte, plus series
metadata, must match its original CPU reference exactly.

On the notebook's displayed CPU/FFTW installation, the default conditioned
strain and final whitened spectrum have relative L2 differences of about
``2.38e-7`` and ``4.91e-7``. The selected original stages match every byte.
Those observations describe this input and installation, not a universal
error bound.

Gating and decisions
---------------------

Tukey gating evaluates a cosine taper and multiplies the data in place.
Elementary-function and multiplication rounding can change taper samples.
The notebook holds gate times and widths fixed, compares original/default
values, and verifies that gating a slice updates its parent. Its float64 gate
example has ten differing samples, at most ``2.22e-16`` apart, on the displayed
CPU libraries; the original gate control matches every byte.

Glitch detection estimates a PSD, whitens data, applies a strict magnitude
threshold and clusters surviving peaks. A small upstream difference can move
a threshold decision. Equal peak counts do not establish equal whitening
values. ``detect_loud_glitches`` restores the complete original calculation;
``findchirp_cluster`` restores clustering alone. Both controls remain opt-in.

Exact native assertions apply to the configured CPU backend and fixed inputs.
The notebook reports default differences on its selected device and records
library versions. It does not infer identical search decisions for an entire
observation or exact agreement between different CPU builds.
