###################################################
Waveforms
###################################################

=====================================
What waveforms can I generate?
=====================================

.. literalinclude:: ../examples/waveform/what_waveform.py
.. command-output:: python ../examples/waveform/what_waveform.py


=====================================
Plotting Time Domain Waveforms 
=====================================

.. plot:: ../examples/waveform/plot_waveform.py
   :include-source:

==============================================
Generating one waveform in multiple detectors
==============================================

.. plot:: ../examples/waveform/plot_detwaveform.py
   :include-source:


===============================================
Selecting which modes to include
===============================================
Gravitational waves can be decomposed into a set
of modes. Some approximants only calculate the dominant
l=2, m=2 mode, while others included higher-order modes. These
often, but not always, include 'HM' in the name. The modes
present in the output polarizations can be selected for these waveforms
as demonstrated below. By default, all modes that the waveform model
supports are typically generated.

.. plot:: ../examples/waveform/higher_modes.py
   :include-source:

=======================================
Calculating the match between waveforms
=======================================

.. literalinclude:: ../examples/waveform/match_waveform.py
.. command-output:: python ../examples/waveform/match_waveform.py

================================================
Plotting a TD and FD waveform together in the TD
================================================

.. plot:: ../examples/waveform/plot_fd_td.py
   :include-source:
   
================================================
Plotting GW phase and amplitude of TD waveform
================================================

.. plot:: ../examples/waveform/plot_phase.py
   :include-source:

================================================
Plotting frequency evolution of TD waveform
================================================

.. plot:: ../examples/waveform/plot_freq.py
   :include-source:
   
=====================================
Adding a custom waveform
=====================================

You can also add your own custom waveform and make it available through 
the waveform interface standard. You can directly call the code as below
or if you include it in a python package, :ref:`PyCBC can directly detect it! <waveform_plugin>`

.. plot:: ../examples/waveform/add_waveform.py
   :include-source:
   

===========================================
Torch-native waveform ports (torch scheme)
===========================================
PyCBC includes torch-native implementations for several frequency-domain
approximants. The default remains the LAL/CPU path; enable torch ports with
environment flags:

- Global: ``PYCBC_TORCH_NATIVE_PORTS=1`` (used when a per-approximant flag is unset).
- SPA/TaylorF2: ``PYCBC_SPATPLT_NATIVE``, ``PYCBC_TAYLORF2_NATIVE``. The
  public TaylorF2 port covers aligned spins, Newtonian amplitude, all supported
  phase/spin/tidal orders, testing-GR ``dchi`` terms, tidal quadrupoles, and
  polarization rotation. Precession, higher-PN amplitude corrections, and
  dynamic-tide extras fall back to lalsimulation.
- SpinTaylorF2: ``PYCBC_SPINTAYLORF2_NATIVE``. This port remains experimental;
  the default path is the lalsimulation implementation.
- IMRPhenomD (FD): ``PYCBC_IMRPHENOMD_NATIVE``.
- SEOBNRv4 / SEOBNRv4_ROM (BBH only): ``PYCBC_SEOBNRV4_NATIVE``. NRTidal
  approximants continue to use the lalsimulation implementation.
- SEOBNRv4HM_ROM (FD higher modes): ``PYCBC_SEOBNRV4HM_NATIVE`` (torch-native ROM
  evaluation using ``SEOBNRv4HMROM_v1.0.hdf5``); requires the ROM file in
  ``$LAL_DATA_PATH`` or ``pycbc/waveform``.

These ports provide Torch-device-compatible PyCBC series, but several waveform
implementations reconstruct their models with NumPy/SciPy before transferring
the result. Torch-scheme support therefore does not, by itself, imply an
end-to-end differentiable waveform.

Set a per-approximant flag to ``1`` to request the torch implementation or ``0``
to force the lalsimulation implementation. Unsupported options retain the
lalsimulation fallback. If a per-flag is unset, the global flag decides.
