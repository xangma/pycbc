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
  polarization rotation. ``TaylorF2NL`` also applies its nonlinear-tide phase
  correction on the active Torch device. Precession, higher-PN amplitude
  corrections, and dynamic-tide extras fall back to lalsimulation.
- SpinTaylorF2: ``PYCBC_SPINTAYLORF2_NATIVE``. The native port covers the
  model's single-spin public interface, including precession, reference-phase
  conventions, phase/spin orders, sidebands, testing-GR ``dchi`` terms,
  primary-object quadrupole deformation, and polarization rotation. Tidal,
  eccentric, second-spin, custom-mode, and non-``dchi`` testing-GR options fall
  back to lalsimulation.
- IMRPhenomC (FD): ``PYCBC_IMRPHENOMC_NATIVE``. Frequency-grid construction,
  amplitude and phase evaluation, coalescence-time correction, masking, and
  polarization assembly run on the active Torch device. The native path covers
  the aligned-spin, circular, dominant-mode model; transverse spins, tides,
  testing-GR changes, non-default orders, and custom modes fall back to
  lalsimulation.
- IMRPhenomD (FD): ``PYCBC_IMRPHENOMD_NATIVE``. Frequency-grid construction,
  amplitude/phase evaluation, masking, and polarization assembly run on the
  active Torch device; small coefficient and QNM-table setup remains on CPU.
  ``IMRPhenomD_NRTidal`` and ``IMRPhenomD_NRTidalv2`` also apply their tidal
  phase, amplitude, spin, and taper corrections on-device. Transverse spins,
  testing-GR changes, and unsupported tidal or mode options fall back to
  lalsimulation.
- IMRPhenomXAS (FD): ``PYCBC_IMRPHENOMXAS_NATIVE``. Aligned-spin amplitude and
  phase fits, scalar matching derivatives, frequency-grid construction,
  masking, and polarization assembly use Torch. The coefficient tables are
  moved to the active device for evaluation. ``IMRPhenomXAS_NRTidalv2`` and
  ``IMRPhenomXAS_NRTidalv3`` also apply their tidal amplitude, phase,
  matter-spin, alignment, and taper corrections on-device. Transverse spins,
  unsupported tidal parameters, eccentricity, non-default PN orders,
  testing-GR changes, and custom modes fall back to lalsimulation.
- IMRPhenomXHM modes (FD): ``PYCBC_IMRPHENOMXHM_NATIVE``. Explicit ``(2, 2)``
  and ``(2, -2)`` requests through ``get_fd_waveform_modes`` reuse the native
  XAS quadrupole on the active Torch device. Default mode sets and requests
  containing the ``(2, 1)``, ``(3, 3)``, ``(3, 2)``, or ``(4, 4)`` fits still
  fall back to lalsimulation; the normal polarization interface is not yet
  native.
- IMRPhenomHM (FD higher modes): ``PYCBC_IMRPHENOMHM_NATIVE``. The six modeled
  positive-m modes, their IMRPhenomD frequency maps, and polarization assembly
  run on the active Torch device. Scalar coefficient and spin-weighted
  spherical-harmonic setup remains on CPU. Mode subsets are supported;
  transverse spins, tides, testing-GR changes, and unmodeled modes fall back to
  lalsimulation.
- SEOBNRv4_ROM (FD aligned-spin): ``PYCBC_SEOBNRV4_NATIVE``. ROM interpolation,
  frequency interpolation, and waveform assembly run on the active Torch
  device using ``SEOBNRv4ROM_v3.0.hdf5``. The native path also covers
  ``SEOBNRv4_ROM_NRTidal`` and ``SEOBNRv4_ROM_NRTidalv2`` with on-device tidal
  corrections. The time-domain ``SEOBNRv4`` model, transverse spins, and
  unsupported tidal extensions continue to use lalsimulation.
- SEOBNRv4HM_ROM (FD higher modes): ``PYCBC_SEOBNRV4HM_NATIVE`` (torch-native ROM
  evaluation using ``SEOBNRv4HMROM_v1.0.hdf5``); requires the ROM file in
  ``$LAL_DATA_PATH`` or ``pycbc/waveform``.

These ports provide Torch-device-compatible PyCBC series, but several waveform
implementations still reconstruct their models with NumPy/SciPy before
transferring the result. Even the device-native ports assemble scalar
coefficients from Python/NumPy values, so Torch-scheme support does not, by
itself, imply an end-to-end differentiable waveform.

The ``multiband`` wrapper keeps its overlap windows, FFT interpolation, and
band accumulation on the active Torch device when its base approximant returns
Torch-backed series.

Set a per-approximant flag to ``1`` to request the torch implementation or ``0``
to force the lalsimulation implementation. Unsupported options retain the
lalsimulation fallback. If a per-flag is unset, the global flag decides.
