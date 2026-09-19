.. _jax-filtering:

JAX Matched Filtering & Signal Processing
=========================================

PyCBC's JAX filtering subsystem executes frequency-domain matched filtering,
correlation, and signal-to-noise ratio (SNR) time series calculation using
pure functional JAX transformations compiled via XLA.

Matched Filtering Formulation
-----------------------------

Given frequency-domain strain data :math:`\tilde{s}(f)` and template waveform
:math:`\tilde{h}(f)` with one-sided power spectral density :math:`S_n(f)`, the
complex matched filter output is given by:

.. math::

   z(t) = 4 \int_{0}^{\infty} \frac{\tilde{s}(f) \tilde{h}^*(f)}{S_n(f)} e^{2\pi i f t} df

In PyCBC JAX:

.. code-block:: python

   from pycbc.scheme import JAXScheme
   from pycbc.filter import matched_filter

   with JAXScheme("cuda:0"):
       snr = matched_filter(template, data, psd=psd, low_frequency_cutoff=20.0)

Chi-Squared Signal Consistency Vetoes
-------------------------------------

JAX implements the standard Bruce Allen power chi-squared test partitioned into
:math:`p` frequency bins of equal expected power. In JAX, the chi-squared time
series is evaluated functionally:

.. code-block:: python

   from pycbc.vetoes import power_chisq

   with JAXScheme("cuda:0"):
       chisq = power_chisq(template, data, num_bins=16, psd=psd,
                           low_frequency_cutoff=20.0)
