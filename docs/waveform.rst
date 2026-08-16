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
approximants and for SEOBNRv4PHM in both time and frequency domains. The
default remains the LAL/CPU path; enable torch ports with environment flags:

- Global: ``PYCBC_TORCH_NATIVE_PORTS=1`` (used when a per-approximant flag is unset).
- SPA/TaylorF2: ``PYCBC_SPATPLT_NATIVE``, ``PYCBC_TAYLORF2_NATIVE``, and
  ``PYCBC_TAYLORF2ECC_NATIVE``. The
  public TaylorF2 port covers aligned spins, Newtonian amplitude, all supported
  phase/spin/tidal orders, testing-GR ``dchi`` terms, tidal quadrupoles, and
  polarization rotation. Regular-grid and arbitrary-frequency
  ``get_fd_waveform_sequence`` evaluation both run on the active device.
  ``TaylorF2Ecc`` adds the model's low-eccentricity phase correction through
  relative 3PN order for regular-grid generation, including its eccentricity
  reference-frequency convention. LAL does not expose ``TaylorF2Ecc`` through
  the arbitrary-frequency sequence API.
  ``TaylorF2NL`` also applies its nonlinear-tide phase correction on the active
  Torch device. Precession, higher-PN amplitude corrections, and dynamic-tide
  extras fall back to lalsimulation.
- TaylorF2 reduced spin: ``PYCBC_TAYLORF2REDSPIN_NATIVE`` and
  ``PYCBC_TAYLORF2REDSPINTIDAL_NATIVE``. Both analytic models evaluate their
  phase and amplitude corrections through 3.5PN, frequency masking, and
  polarization assembly on the active device. The tidal variant includes its
  5PN and 6PN tidal phase terms. Arbitrary-frequency evaluation is a native
  extension because lalsimulation does not expose either model through its
  sequence API; it accepts unordered positive frequencies, ignores
  ``long_asc_nodes``, and returns zero above Schwarzschild ISCO. Transverse
  spins, non-default spin/tidal/eccentricity flags, testing-GR changes, and
  custom modes retain the regular-grid lalsimulation path.
- SpinTaylorF2: ``PYCBC_SPINTAYLORF2_NATIVE``. The native port covers the
  model's single-spin public interface, including precession, reference-phase
  conventions, phase/spin orders, sidebands, testing-GR ``dchi`` terms,
  primary-object quadrupole deformation, and polarization rotation.
  Arbitrary-frequency ``get_fd_waveform_sequence`` evaluation also runs on the
  active device. This is a native extension because lalsimulation does not
  expose SpinTaylorF2 through its sequence API; ``long_asc_nodes`` is ignored,
  ``f_ref=0`` uses the first supplied frequency, and unordered positive
  frequencies are supported. Tidal, eccentric, second-spin, custom-mode, and
  non-``dchi`` testing-GR options fall back to lalsimulation for regular-grid
  generation and remain unavailable through the sequence API.
- IMRPhenomA/B (FD): ``PYCBC_IMRPHENOMA_NATIVE`` and
  ``PYCBC_IMRPHENOMB_NATIVE``. The legacy analytic
  amplitude and phase fits, their distinct frequency-bin boundary rules,
  masking, and polarization assembly run on the active Torch device.
  Arbitrary-frequency ``get_fd_waveform_sequence`` evaluation is a native
  extension because lalsimulation does not expose either model through its
  sequence API. It accepts unordered positive frequencies, ignores
  ``long_asc_nodes``, and returns zero at and above the fitted cutoff.
  IMRPhenomA covers nonspinning binaries and IMRPhenomB covers aligned spins;
  transverse spins, tides, testing-GR changes, non-default orders, and custom
  modes retain the regular-grid lalsimulation path.
- IMRPhenomC (FD): ``PYCBC_IMRPHENOMC_NATIVE``. Frequency-grid construction,
  amplitude and phase evaluation, regular-grid coalescence-time spline,
  masking, and polarization assembly run on the active Torch device.
  Arbitrary-frequency ``get_fd_waveform_sequence`` evaluation is also native.
  This is a native extension because lalsimulation does not expose IMRPhenomC
  through its sequence API; it uses the continuous phase derivative at
  ringdown instead of a grid-dependent spline, ignores ``long_asc_nodes``,
  accepts unordered positive frequencies, and returns zero at and above the
  model's fixed ``Mf=0.15`` cutoff. The native path covers the aligned-spin,
  circular, dominant-mode model; transverse spins, tides, testing-GR changes,
  non-default orders, and custom modes fall back to lalsimulation for the
  regular grid and remain unavailable through the sequence API.
- IMRPhenomD (FD): ``PYCBC_IMRPHENOMD_NATIVE``. Frequency-grid construction,
  amplitude/phase evaluation, masking, and polarization assembly run on the
  active Torch device; small coefficient and QNM-table setup remains on CPU.
  Arbitrary-frequency ``get_fd_waveform_sequence`` evaluation is also native.
  ``IMRPhenomD_NRTidal`` and ``IMRPhenomD_NRTidalv2`` also apply their tidal
  phase, amplitude, spin, and taper corrections on-device for both sampling
  interfaces. Transverse spins, testing-GR changes, and unsupported tidal or
  mode options fall back to lalsimulation.
- IMRPhenomP (FD): ``PYCBC_IMRPHENOMP_NATIVE``. The IMRPhenomC baseline,
  NNLO precession angles, small-angle Wigner rotation, frequency masking, and
  polarization assembly run on the active Torch device. Scalar coefficient
  construction and the regular-grid time-alignment spline remain on CPU.
  Regular-grid and strictly increasing arbitrary-frequency sequence
  evaluation are native; as in LAL's sequence interface,
  ``long_asc_nodes`` is ignored and ``f_ref=0`` uses the first supplied
  frequency. Tides, eccentricity, testing-GR changes, non-default orders, and
  custom modes fall back to lalsimulation.
- IMRPhenomPv2 (FD): ``PYCBC_IMRPHENOMPV2_NATIVE``. The PhenomD baseline,
  NNLO precession angles, Wigner rotation, frequency-grid construction, and
  polarization assembly run on the active Torch device. Source-frame model
  mapping, scalar coefficient setup, and the merger-time spline remain on CPU.
  The native path covers regular-grid and arbitrary-frequency BBH evaluation,
  plus the ``IMRPhenomPv2_NRTidal`` and ``IMRPhenomPv2_NRTidalv2`` phase,
  amplitude, matter-spin, and taper corrections on-device. As in LAL's
  sequence interface, ``long_asc_nodes`` is ignored there and ``f_ref=0`` uses
  the first supplied frequency. Testing-GR changes, eccentricity, non-default
  spin/tidal orders, and custom modes fall back to lalsimulation.
- IMRPhenomXAS (FD): ``PYCBC_IMRPHENOMXAS_NATIVE``. Aligned-spin amplitude and
  phase fits, scalar matching derivatives, frequency-grid construction,
  masking, and polarization assembly use Torch. Arbitrary-frequency
  ``get_fd_waveform_sequence`` evaluation is also native. The coefficient
  tables are moved to the active device for evaluation.
  ``IMRPhenomXAS_NRTidalv2`` and ``IMRPhenomXAS_NRTidalv3`` also apply their
  tidal amplitude, phase, matter-spin, alignment, and taper corrections
  on-device for both sampling interfaces. Transverse spins,
  unsupported tidal parameters, eccentricity, non-default PN orders,
  testing-GR changes, and custom modes fall back to lalsimulation.
- IMRPhenomXHM modes (FD): ``PYCBC_IMRPHENOMXHM_NATIVE``. The default mode set
  and explicit ``(2, +/-2)``, ``(2, +/-1)``, ``(3, +/-2)``, ``(3, +/-3)``, and
  ``(4, +/-4)`` requests through ``get_fd_waveform_modes`` run on the active
  Torch device. The quadrupole reuses native XAS; ``(3, +/-2)`` includes its
  spheroidal-to-spherical ringdown mixing, and the other modes use native XHM
  no-mixing kernels. The mixed mode uses analytic phase-boundary derivatives
  instead of LAL's roundoff-sensitive ``1e-7`` finite difference, so measurable
  phase differences can remain near the calibration boundary. Polarization
  assembly and spin-weighted spherical-harmonic evaluation through
  ``get_fd_waveform`` also run on-device. This path performs the full mode
  evaluation rather than LAL's polarization-interface multibanding,
  so sparse mode selections can retain small additional differences.
  Arbitrary-frequency ``get_fd_waveform_sequence`` evaluation reuses the same
  mode kernels on the active device, including supported mode subsets; as in
  LAL's sequence interface, ``long_asc_nodes`` is ignored and ``f_ref=0`` uses
  the first supplied frequency.
- IMRPhenomHM (FD higher modes): ``PYCBC_IMRPHENOMHM_NATIVE``. The six modeled
  positive-m modes, their IMRPhenomD frequency maps, and polarization assembly
  run on the active Torch device, including spin-weighted spherical-harmonic
  evaluation. Scalar model-coefficient setup remains on CPU. Mode subsets are
  supported. Arbitrary-frequency ``get_fd_waveform_sequence`` evaluation is
  also native for strictly increasing positive frequencies; as in LAL,
  ``long_asc_nodes`` is ignored and ``f_ref=0`` uses the first frequency.
  There is no sequence-only high-frequency cutoff;
  transverse spins, tides, testing-GR changes, and unmodeled modes fall back to
  lalsimulation.
- SEOBNRv4_ROM (FD aligned-spin): ``PYCBC_SEOBNRV4_NATIVE``. ROM interpolation,
  frequency interpolation, and waveform assembly run on the active Torch
  device using ``SEOBNRv4ROM_v3.0.hdf5``. The native path also covers
  ``SEOBNRv4_ROM_NRTidal`` and ``SEOBNRv4_ROM_NRTidalv2`` with on-device tidal
  corrections. Regular-grid and arbitrary-frequency
  ``get_fd_waveform_sequence`` evaluation are both native; as in LAL's
  sequence interface, ``long_asc_nodes`` is ignored and ``f_ref=0`` uses the
  first supplied frequency. The time-domain ``SEOBNRv4`` model, transverse
  spins, and unsupported tidal extensions continue to use lalsimulation.
- SEOBNRv4HM_ROM (FD higher modes): ``PYCBC_SEOBNRV4HM_NATIVE``. ROM
  interpolation, low-frequency TaylorF2/ROM hybridization, harmonic evaluation,
  and waveform assembly run on the active Torch device using
  ``SEOBNRv4HMROM_v1.0.hdf5``; the file must be in ``$LAL_DATA_PATH`` or
  ``pycbc/waveform``. Regular-grid and arbitrary-frequency
  ``get_fd_waveform_sequence`` evaluation are native across the model's full
  positive-frequency range. Mode subsets and unordered sequence frequencies are
  supported, provided later samples do not fall below the inspiral spline domain
  set by the first frequency. As in LAL, this model ignores ``f_ref``; the
  sequence interface also ignores ``long_asc_nodes``.
- SEOBNRv4PHM (TD and FD precessing higher modes):
  ``PYCBC_SEOBNRV4PHM_NATIVE``. The adaptive EOB dynamics, factorized modes,
  NQC and ringdown attachment, frame rotations, harmonic projection, and
  polarization assembly run through the native Torch pipeline. The FD entry
  point applies a Torch real FFT to the same native TD result. The default
  ``(2,2)``, ``(2,1)``, ``(3,3)``, ``(4,4)``, and ``(5,5)`` positive-m modes
  and supported subsets are available. Scalar setup, QNM interpolation, and
  some adaptive-control decisions remain Python/NumPy/SciPy work. Apple MPS
  uses ``float32``/``complex64`` because it does not provide the corresponding
  double-precision kernels. Non-default PN orders, eccentricity, matter and
  testing-GR parameters, alternate frame options, unsupported modes, and
  inputs with ``mass1 < mass2`` retain the LAL/CPU path.

These ports provide Torch-device-compatible PyCBC series, but several waveform
implementations still reconstruct their models with NumPy/SciPy before
transferring the result. Even the device-native ports assemble scalar
coefficients from Python/NumPy values, so Torch-scheme support does not, by
itself, imply an end-to-end differentiable waveform.

The ``multiband`` wrapper keeps its overlap windows, FFT interpolation, and
band accumulation on the active Torch device when its base approximant returns
Torch-backed series.

The analytical time- and frequency-domain ringdown generators keep their
spherical-harmonic evaluation and waveform assembly on the active Torch device
for spherical harmonics. Spheroidal harmonics retain their optional CPU
``pykerr`` implementation.

Set a per-approximant flag to ``1`` to request the torch implementation or ``0``
to force the lalsimulation implementation. Unsupported options retain the
lalsimulation fallback. If a per-flag is unset, the global flag decides.
