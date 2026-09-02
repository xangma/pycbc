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
approximants, time-domain approximants, and public mode interfaces. The
registry-derived table below is the source of truth for those public waveform
interfaces. It does not list auxiliary Torch kernels such as ``SPAtmplt``,
``PreTaylorF2``, ``TaylorF2NL``, or the taper helpers described later in this
section.

The Python ``lal`` and ``lalsimulation`` packages are optional for enabled,
supported native ports: the Torch registry and native approximant discovery do
not require either package. PyCBC supplies the scalar constants, GPS metadata,
and spherical harmonics needed by those native closures. Explicit conversion
to LAL series, non-native approximants, and other LAL-backed utilities still
require the corresponding package and raise a descriptive dependency error
when it is absent. When LALSimulation is unavailable under a Torch scheme,
PyCBC automatically attempts compatible native ports, including ports that are
normally opt-in. Predicate-readiness and parameter-support guards remain
authoritative. An explicit native-port opt-out is still honored; disabled ports
and requests outside a native support envelope raise a descriptive error
instead of attempting an unavailable LALSimulation fallback.
Some ROM ports also require external HDF5 model tables distributed with LAL
model-data packages (normally located through ``LAL_DATA_PATH``); native
evaluation removes the LALSuite Python calls, but does not replace those
tables.

In the table, ``attempted by default`` means native selection is enabled,
``attempted by default (predicate guarded)`` adds a default-readiness check
that may retain the fallback, and ``opt-in`` requires a global or component
override. These labels do not bypass each interface's parameter-support
or default-readiness predicate. The automatic opt-in attempt described above
applies only when the LALSimulation fallback is absent. Synthesized
FD-to-TD interfaces are not exposed in that configuration. Synthesized
``*_INTERP`` interfaces remain available only when their duration-estimator
closure is also independent of LALSimulation. Filter-template helpers likewise
raise a descriptive error when a selected native waveform still has only a
LALSimulation-backed duration estimator. Named final-frequency cutoffs backed
by LALSimulation are rejected before waveform generation when that package is
unavailable; analytical cutoffs remain available. Direct use of the FD-to-TD
conversion helper remains possible with an explicit ``rwrap``; inferring that
value uses a LALSimulation remnant fit.

"Eligible Torch targets" are possible output devices for at least one
supported request, not claims that every internal operation is device-resident
or that the hardware is available. The predicate remains authoritative for an
individual request and may apply stricter device-specific accuracy limits. A
"native extension" has no equivalent LAL public interface, so disabling it or
requesting unsupported options preserves LAL's existing unsupported-model
error. ``LAL reference`` identifies the validation baseline, not a blanket
numerical-parity guarantee. ``CPU/LAL adapter`` denotes a model-specific mode
or TD-to-FD fallback rather than the standard dispatcher.

.. include:: _include/torch-waveform-capabilities.rst

Environment flags can override either policy:

- Global: ``PYCBC_TORCH_NATIVE_PORTS=1`` enables remaining ports, while ``=0``
  disables native ports whose per-approximant flag is unset.
- SPA/TaylorF2: ``PYCBC_SPATPLT_NATIVE``, ``PYCBC_TAYLORF2_NATIVE``,
  ``PYCBC_TAYLORF2ECC_NATIVE``, and ``PYCBC_TAYLORF2NLTIDES_NATIVE``. The
  optimized ``SPAtmplt`` filter kernel supports both its regular grid and its
  direct arbitrary-frequency ``sample_points`` interface on the active device.
  Set either SPA or TaylorF2 flag to ``0`` to opt that piece out of its native
  default; per-approximant flags take precedence over the global switch.
  The public TaylorF2 port covers aligned spins, Newtonian amplitude, all
  supported phase/spin/tidal orders, testing-GR ``dchi`` terms, tidal
  quadrupoles, and polarization rotation. Regular-grid and arbitrary-frequency
  ``get_fd_waveform_sequence`` evaluation both run on the active device. On
  Apple MPS, requests below a mass-ratio-scaled dimensionless-frequency
  boundary retain the LAL fallback to bound float32 inspiral-phase error; the
  lowest arbitrary-frequency sample and any lower ``f_ref`` both participate
  in that check.
  ``TaylorF2Ecc`` adds the model's low-eccentricity phase correction through
  relative 3PN order for regular-grid generation, including its eccentricity
  reference-frequency convention. Arbitrary-frequency evaluation is a native
  extension because LAL has no ``TaylorF2Ecc`` sequence implementation; it
  accepts unordered positive frequencies (including duplicates), ignores
  ``long_asc_nodes``, and uses ``f_ref`` as the eccentricity reference when
  nonzero or otherwise the lowest supplied frequency. The supported regular
  and sequence paths are enabled by default under ``TorchScheme``; set
  ``PYCBC_TAYLORF2ECC_NATIVE=0`` to opt out. The TaylorF2 MPS carrier-phase
  guard also applies. In addition, MPS requests fall back when the absolute
  eccentric phase at the starting or reference frequency exceeds ``5e4``
  radians, bounding float32 phase error. LAL handles that fallback for regular
  grids; because LAL has no sequence implementation, an unsafe sequence
  request retains its normal unsupported-model error.
  ``TaylorF2NLTides`` supports regular-grid and arbitrary-frequency generation
  with the six public parameters
  ``nl_tides_a1``, ``nl_tides_n1``, ``nl_tides_f1``, ``nl_tides_a2``,
  ``nl_tides_n2``, and ``nl_tides_f2`` parameters. Its ordinary 5PN/6PN tidal
  phase and both nonlinear-tide corrections run on the active device; LAL
  does not expose this model through its arbitrary-frequency sequence API, so
  the sequence path is a native extension. It accepts unordered positive
  frequencies, preserves duplicates, and ignores ``long_asc_nodes`` like the
  underlying LAL sequence interface. The supported regular and sequence paths
  are enabled by default under ``TorchScheme``; set
  ``PYCBC_TAYLORF2NLTIDES_NATIVE=0`` to opt out.
  ``TaylorF2NL`` also applies its nonlinear-tide phase correction on the active
  Torch device. ``PreTaylorF2`` shares the ``PYCBC_TAYLORF2_NATIVE`` switch;
  its carrier, cyclic shift, frequency masking, and optional final Kaiser
  taper remain on the active device. The shared ``fd_taper`` and ``td_taper``
  helpers likewise construct their Kaiser windows on-device for Torch-backed
  series. Precession, higher-PN amplitude corrections, and dynamic-tide extras
  fall back to lalsimulation.
- TaylorF2 reduced spin: ``PYCBC_TAYLORF2REDSPIN_NATIVE`` and
  ``PYCBC_TAYLORF2REDSPINTIDAL_NATIVE``. Both analytic models evaluate their
  phase and amplitude corrections through 3.5PN, frequency masking, and
  polarization assembly on the active device. The tidal variant includes its
  5PN and 6PN tidal phase terms. Arbitrary-frequency evaluation is a native
  extension because lalsimulation does not expose either model through its
  sequence API; it accepts unordered positive frequencies, ignores
  ``long_asc_nodes``, and returns zero above Schwarzschild ISCO. Transverse
  spins, non-default spin/tidal/eccentricity flags, testing-GR changes, and
  custom modes retain the regular-grid lalsimulation path. Supported regular
  and sequence calls are enabled by default under ``TorchScheme``; set the
  corresponding per-model flag to ``0`` to opt out.
- EccentricFD: ``PYCBC_ECCENTRICFD_NATIVE``. The 3.5PN phase with
  eccentricity corrections through order ``e^8``, ten restricted-amplitude
  harmonics, frequency cutoffs, and polarization assembly run on the active
  Torch device. Scalar analytic coefficient setup remains Python work. The
  implementation mirrors the legacy model's use of ``long_asc_nodes`` both as
  the inclination azimuth and as a final polarization-basis rotation.
  Arbitrary-frequency evaluation is a native extension because lalsimulation
  has no EccentricFD sequence implementation. It accepts unordered positive
  frequencies, including duplicates, and uses their minimum as the
  eccentricity reference frequency; like the regular model, it ignores
  ``f_ref``. Transverse spins, matter and testing-GR parameters, non-default
  auxiliary PN orders, and custom modes retain the lalsimulation path.
  Supported regular-grid and sequence calls are enabled by default on CPU and
  CUDA. Apple MPS retains the regular-grid LAL fallback because float32 phase
  cancellation can materially reduce accuracy; its sequence interface has no
  LAL implementation and therefore remains opt-in there. Set
  ``PYCBC_ECCENTRICFD_NATIVE=1`` to request the native MPS path explicitly, or
  ``=0`` to opt out on every device.
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
  generation and remain unavailable through the sequence API. Supported calls
  are native by default on CPU and CUDA. Apple MPS retains the regular-grid LAL
  fallback because float32 cancellation in the precession and inspiral phase
  setup can produce a waveform-wide phase offset; its sequence interface has
  no LAL implementation and therefore remains opt-in there. Set the component
  flag to ``1`` to request the native MPS path explicitly, or ``=0`` to opt out
  on every device.
- SpinTaylorT4Fourier/SpinTaylorT5Fourier:
  ``PYCBC_SPINTAYLORT4FOURIER_NATIVE`` and
  ``PYCBC_SPINTAYLORT5FOURIER_NATIVE``. These opt-in, CPU-only ports cover the
  public regular-grid FD interface with restricted amplitude
  (``amplitude_order=0``), phase order ``-1``, ``7``, or ``8``, and BBH/default
  matter settings. Both accept spin order ``-1``, ``6``, or ``7``; these orders
  are equivalent in the averaged irregular Fourier evolution. The reference
  frequency is ``f_lower`` when ``f_ref=0`` and otherwise must lie from
  ``f_lower`` up to, but not including, Schwarzschild ISCO. Output must end no
  later than ISCO; ``f_final=0`` and values at or below the snapped first grid
  bin use LAL's full-to-ISCO convention. Arbitrary-frequency, time-domain, and
  mode interfaces are not registered. Unsupported options, non-CPU schemes,
  and disabled ports retain the standard lalsimulation path.
  Numeric validation uses an isolated pointer-corrected LAL build because the
  irregular SpinTaylor Fourier driver in LALSuite 7.26.1 passes an invalid
  coefficient pointer to its ODE integrator.
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
  modes retain the regular-grid lalsimulation path. Supported calls are native
  by default on CPU and CUDA. Apple MPS retains the regular-grid LAL fallback
  because float32 cancellation among the fitted phase terms can materially
  reduce accuracy; its sequence interface has no LAL implementation and
  therefore remains opt-in there. Set the corresponding component flag to
  ``1`` to request the native MPS path explicitly, or ``=0`` to opt out on
  every device.
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
  regular grid and remain unavailable through the sequence API. Supported
  calls are native by default on CPU and CUDA. Apple MPS retains the
  regular-grid LAL fallback because float32 cancellation among the fitted
  phase terms can materially reduce accuracy; its sequence interface has no
  LAL implementation and therefore remains opt-in there. Set the component
  flag to ``1`` to request the native MPS path explicitly, or ``=0`` to opt
  out on every device.
- IMRPhenomD (FD): ``PYCBC_IMRPHENOMD_NATIVE``. Frequency-grid construction,
  amplitude/phase evaluation, masking, and polarization assembly run on the
  active Torch device; small coefficient and QNM-table setup remains on CPU.
  Arbitrary-frequency ``get_fd_waveform_sequence`` evaluation is also native.
  ``IMRPhenomD``, ``IMRPhenomD_NRTidal``, and
  ``IMRPhenomD_NRTidalv2`` are native by default under ``TorchScheme``; set the
  flag to ``0`` to opt out. The NRTidal variants apply their tidal phase,
  amplitude, spin, and taper corrections on-device for both sampling
  interfaces. Transverse spins, testing-GR changes, and unsupported tidal or
  mode options fall back to lalsimulation.
- TaylorT4 (TD): ``PYCBC_TAYLORT4_NATIVE``. The nonspinning PN orbit,
  polarizations, and modes are lalsimulation-free. The strictly sequential
  two-scalar RK4 recurrence uses a compiled host-double kernel (with a
  pure-Python source-tree fallback) and transfers its completed trajectory to
  the active device once; bulk polarization and mode construction then run
  with Torch on that device. This avoids per-step GPU launches and
  synchronization while retaining LAL's sample, epoch, ISCO,
  reference-frequency, and tidal conventions. Supported polarization and mode
  requests are native by default on CPU and CUDA. Apple MPS retains the faster
  LAL path by default; set the component flag to ``1`` to request native MPS
  execution explicitly, or to ``0`` to opt out on every device. Spins,
  eccentricity, testing-GR changes, unsupported PN orders, and custom mode
  arrays retain the lalsimulation path.
- IMRPhenomT (TD): ``PYCBC_IMRPHENOMT_NATIVE``. Dominant-mode phase and
  amplitude construction, frequency-to-time inversion, regular time-grid
  synthesis, reference phase, and polarization assembly use Torch. CPU and
  CUDA evaluate the two scalar frequency-time roots on the active device;
  MPS evaluates those roots with Torch CPU float64 because MPS is limited to
  float32, then keeps bulk waveform synthesis on MPS. The output length and
  epoch are the only scalar host values required by the PyCBC interface. The
  native path is opt-in and covers the aligned-spin, circular, dominant-mode
  model while preserving LAL's ``f_ref=0`` convention and ascending-node
  polarization rotation. Transverse spins, tides, testing-GR changes,
  non-default orders, and custom modes fall back to lalsimulation.
- IMRPhenomTHM (TD): ``PYCBC_IMRPHENOMTHM_NATIVE``. The calibrated
  ``(2,2)``, ``(2,1)``, ``(3,3)``, ``(4,4)``, and ``(5,5)`` mode families,
  aligned-spin negative-mode symmetry, default and custom mode-array
  assembly, spherical-harmonic projection, and polarization rotation run on
  the active Torch device. It shares IMRPhenomT's native time grid, carrier
  phase, frequency inversion, and remnant construction. Empty or unsupported
  mode arrays and waveform modifications outside the aligned-spin, circular,
  default-order model retain the lalsimulation path. Unflagged requests use
  the native port for the default mode projection when a conservative
  chirp-time estimate bounds LAL's relative frequency-root bracket to 0.21
  time samples and the mass ratio is at most 10. This avoids compiler-sensitive
  grid and phase shifts in long waveforms. Setting the component or global
  native switch to ``1`` explicitly retains the broader native subset,
  including custom mode arrays; setting it to ``0`` opts out.
- IMRPhenomTP / IMRPhenomTPHM (TD): ``PYCBC_IMRPHENOMTP_NATIVE`` and
  ``PYCBC_IMRPHENOMTPHM_NATIVE``. The default numerical-precession orbit,
  Euler angles, co-precessing carrier, mode twisting, mode interface, and
  polarization assembly are lalsimulation-free. The small adaptive
  13-scalar orbit state and the sequential natural-cubic recurrences use
  fused host arithmetic, after which their completed tensors are transferred
  once to the active device; carrier, angle quadrature, modes, and waveform
  assembly remain on-device. The aligned binary and remnant setup is prepared
  once and reused by the precessing carrier. The ports cover the default
  version-300, convention-1, final-spin-mode-4 BBH model, including supported
  TPHM mode subsets. They remain opt-in because short-waveform GPU latency is
  still substantially above the compiled LAL implementation. Tides,
  eccentricity, testing-GR changes, alternate precession settings, non-default
  orders, and unsupported modes retain the lalsimulation path.
- IMRPhenomNSBH (FD): ``PYCBC_IMRPHENOMNSBH_NATIVE``. The PhenomC-derived
  amplitude, PhenomD plus NRTidalv2 phase, frequency-grid construction, and
  polarization assembly run on the active Torch device. Remnant, disruption,
  and phenomenological coefficient fits remain scalar CPU work. Regular-grid
  and arbitrary-frequency sequence evaluation are native; as in LAL's
  sequence interface, ``long_asc_nodes`` is ignored and ``f_ref=0`` uses the
  first supplied frequency. The implementation covers the aligned-spin,
  dominant-mode model with a black hole in ``mass1`` and neutron star in
  ``mass2``. It is native by default under ``TorchScheme``; set the flag to
  ``0`` to opt out. On Apple MPS, starts below ``Mf=5e-4`` retain the LAL path
  to bound float32 phase error. Unsupported waveform modifications also fall
  back to lalsimulation.
- IMRPhenomP (FD): ``PYCBC_IMRPHENOMP_NATIVE``. The IMRPhenomC baseline,
  NNLO precession angles, small-angle Wigner rotation, frequency masking, and
  polarization assembly run on the active Torch device. Scalar coefficient
  construction remains on CPU; the regular-grid time-alignment spline runs on
  the active Torch device.
  Regular-grid and strictly increasing arbitrary-frequency sequence
  evaluation are native; as in LAL's sequence interface,
  ``long_asc_nodes`` is ignored and ``f_ref=0`` uses the first supplied
  frequency. Supported requests are native by default under ``TorchScheme``;
  set the model flag to ``0`` to opt out. On Apple MPS, requests retain the LAL
  path unless their first evaluated frequency satisfies
  ``M f >= max(7.5e-4, 1.0e-4 q)``, where ``q`` is the larger-to-smaller mass
  ratio. Tides, eccentricity, testing-GR changes, non-default spin/tidal/
  eccentricity orders, and custom modes fall back to lalsimulation.
- IMRPhenomPv2 (FD): ``PYCBC_IMRPHENOMPV2_NATIVE``. The PhenomD baseline,
  NNLO precession angles, Wigner rotation, frequency-grid construction, and
  polarization assembly run on the active Torch device. Source-frame model
  mapping, scalar coefficient setup, and the merger-time spline remain on CPU.
  The native path covers regular-grid and arbitrary-frequency BBH evaluation,
  plus the ``IMRPhenomPv2_NRTidal`` and ``IMRPhenomPv2_NRTidalv2`` phase,
  amplitude, matter-spin, and taper corrections on-device. As in LAL's
  sequence interface, ``long_asc_nodes`` is ignored there and ``f_ref=0`` uses
  the first supplied frequency. All three variants are native by default under
  ``TorchScheme``; set ``PYCBC_IMRPHENOMPV2_NATIVE=0`` to opt out. Apple MPS
  uses single precision, so requests whose first evaluated frequency is below
  ``M f = 2.5e-4`` retain the LAL path to bound accumulated phase error.
  Testing-GR changes, eccentricity, non-default spin/tidal orders, and custom
  modes fall back to lalsimulation.
- IMRPhenomXAS (FD): ``PYCBC_IMRPHENOMXAS_NATIVE``. Aligned-spin amplitude and
  phase fits, scalar matching derivatives, frequency-grid construction,
  masking, and polarization assembly use Torch. Arbitrary-frequency
  ``get_fd_waveform_sequence`` evaluation is also native. The coefficient
  tables are moved to the active device for evaluation.
  ``IMRPhenomXAS_NRTidalv2`` and ``IMRPhenomXAS_NRTidalv3`` also apply their
  tidal amplitude, phase, matter-spin, alignment, and taper corrections
  on-device for both sampling interfaces. All three variants are native by
  default under ``TorchScheme``; set ``PYCBC_IMRPHENOMXAS_NATIVE=0`` to opt
  out. Transverse spins, unsupported tidal parameters, eccentricity,
  non-default PN orders, testing-GR changes, and custom modes fall back to
  lalsimulation.
- IMRPhenomXP (FD): ``PYCBC_IMRPHENOMXP_NATIVE``. The aligned-spin XAS carrier,
  NNLO or multiple-scale-analysis precession angles, Wigner rotation, frequency
  masking, and polarization assembly run on the active Torch device for regular-
  grid and arbitrary-frequency evaluation. Native NNLO version 102 uses
  convention 0 and final-spin mode 0. Native MSA version 223 (including its 300
  alias) uses convention 1 and supports final-spin modes 0, 3, and 4, so the
  default XP configuration is native. ``IMRPhenomXP_NRTidalv2`` and
  ``IMRPhenomXP_NRTidalv3`` apply their matter phase, amplitude, alignment, and
  taper corrections to the co-precessing carrier on-device before twist-up.
  All three variants are native by default under ``TorchScheme``; set
  ``PYCBC_IMRPHENOMXP_NATIVE=0`` to opt out.
  MSA coefficient, reference-frame, final-spin, and matter-scalar setup remain
  CPU work. Other configurations continue to use lalsimulation.
- IMRPhenomXHM modes (FD): ``PYCBC_IMRPHENOMXHM_NATIVE``. The model is native
  by default under ``TorchScheme``; set the flag to ``0`` to opt out. The
  default mode set and explicit ``(2, +/-2)``, ``(2, +/-1)``, ``(3, +/-2)``,
  ``(3, +/-3)``, and ``(4, +/-4)`` requests through
  ``get_fd_waveform_modes`` run on the active Torch device. The quadrupole
  reuses native XAS; ``(3, +/-2)`` includes its
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
- IMRPhenomXPHM (FD): ``PYCBC_IMRPHENOMXPHM_NATIVE``. The model is native by
  default under ``TorchScheme``; set the flag to ``0`` to opt out. The
  default XHM co-precessing modes, MSA Euler angles, Wigner rotations, and
  polarization assembly run on the active Torch device for regular-grid and
  arbitrary-frequency evaluation. The native scope covers the default mode
  set and explicit subsets of the five positive-m co-precessing families with
  MSA version 223 or its 300 alias, convention 1, and final-spin modes 0, 3,
  and 4. Explicit arrays are deduplicated and evaluated in canonical model
  order, and an empty array returns all-zero polarizations.
  Scalar source-frame, MSA-coefficient, and remnant-spin setup remain on CPU.
  It inherits XHM's analytic ``(3, 2)`` phase-boundary derivative and performs
  full regular-grid mode evaluation, so the XHM calibration-edge and
  multibanding differences described above also apply.
  Mode arrays with negative-m or unmodeled entries, NNLO and SpinTaylor
  angles, PNR options, matter,
  eccentricity, testing-GR changes, and non-default orders fall back to
  lalsimulation. As in LAL's sequence interface, ``long_asc_nodes`` is ignored
  and ``f_ref=0`` uses the first supplied frequency.
- IMRPhenomXO4a and IMRPhenomXPNR (FD tuned precession):
  ``PYCBC_IMRPHENOMXO4A_NATIVE`` and ``PYCBC_IMRPHENOMXPNR_NATIVE``.
  Both ports reuse the native XPHM carrier while applying tuned PNR angles,
  co-precessing-mode corrections, the antisymmetric quadrupole contribution,
  higher-mode angle-frequency maps, and polarization assembly with Torch.
  They support the default positive-m ``(2,2)``, ``(2,1)``,
  ``(3,3)``, ``(3,2)``, and ``(4,4)`` families and supported explicit subsets.
  XO4a uses its version-300 MSA angle prescription and is native by default for
  supported regular grids and arbitrary-frequency sequences; set
  ``PYCBC_IMRPHENOMXO4A_NATIVE=0`` to opt out. XPNR evolves the version-330
  SpinTaylor trajectory and constructs its common fine-grid angle splines
  natively on CPU and CUDA for both regular grids and arbitrary-frequency
  sequences; set ``PYCBC_IMRPHENOMXPNR_NATIVE=0`` to opt out. MPS intentionally
  falls back to lalsimulation because its float32-only version-330 SpinTaylor
  trajectory does not meet the native port's accuracy target. Aligned-spin
  inputs, alternate precession conventions, and final-spin choices outside the
  default version-330 configuration also fall back. Scalar source-frame,
  remnant-fit, interpolation-control, and model-coefficient setup remain host
  work.
- IMRPhenomHM (FD higher modes): ``PYCBC_IMRPHENOMHM_NATIVE``. The six modeled
  positive-m modes, their IMRPhenomD frequency maps, and polarization assembly
  run on the active Torch device, including spin-weighted spherical-harmonic
  evaluation. The supported regular-grid and arbitrary-frequency paths are
  native by default under ``TorchScheme``; set
  ``PYCBC_IMRPHENOMHM_NATIVE=0`` to opt out. Scalar model-coefficient setup
  remains on CPU. Mode subsets are supported. The public
  ``get_fd_waveform_modes`` interface returns signed ``(l, m)`` modes from the
  modeled ``(2, +/-2)``, ``(2, +/-1)``, ``(3, +/-3)``, ``(3, +/-2)``,
  ``(4, +/-4)``, and ``(4, +/-3)`` families using the same native kernels;
  unsupported requests use its model-specific CPU/LAL adapter.
  Arbitrary-frequency
  ``get_fd_waveform_sequence`` evaluation accepts strictly increasing positive
  frequencies; as in LAL, ``long_asc_nodes`` is ignored and ``f_ref=0`` uses
  the first frequency.
  There is no sequence-only high-frequency cutoff;
  transverse spins, tides, testing-GR changes, and unmodeled modes fall back to
  lalsimulation.
- IMRPhenomPv3/Pv3HM (FD): ``PYCBC_IMRPHENOMPV3_NATIVE`` and
  ``PYCBC_IMRPHENOMPV3HM_NATIVE``. Source-frame mapping, MSA angle evolution,
  Wigner rotations, mode assembly, and polarization synthesis support finite
  Cartesian spins with magnitude at most one on CPU and CUDA. Both regular
  grids and strictly increasing arbitrary-frequency sequences run natively by
  default for the supported model configuration. Apple MPS currently keeps
  genuinely precessing requests on the LAL path, while its aligned-spin path
  supports either sign of the aligned spins. IMRPhenomPv3 uses only ``(2, 2)``;
  every Pv3HM mode selection must include ``(2, 2)`` and may otherwise contain
  only ``(2, 1)``, ``(3, 3)``, ``(3, 2)``, ``(4, 4)``, and ``(4, 3)``. Tides,
  eccentricity, testing-GR changes, non-default model settings, and invalid
  mode selections retain the LAL fallback. LAL's generic sequence dispatcher
  does not expose Pv3 or Pv3HM, so unsupported or disabled sequence requests
  retain its existing error rather than silently changing models.
- EOBNRv2_ROM / EOBNRv2HM_ROM (FD nonspinning):
  ``PYCBC_EOBNRV2_NATIVE``. The dominant ``(2, 2)`` model and the full default
  higher-mode set ``(2, 2)``, ``(2, 1)``, ``(3, 3)``, ``(4, 4)``, ``(5, 5)``
  are reconstructed directly from the public ``EOBNRv2HMROM_*.dat`` files.
  Mass-ratio interpolation, reduced-basis reconstruction, frequency
  interpolation, harmonic evaluation, and polarization assembly run on the
  active Torch device; binary data loading remains host-side. The data files
  must share a directory on ``$LAL_DATA_PATH`` or be placed in
  ``pycbc/waveform``. Regular-grid and strictly increasing positive-frequency
  ``get_fd_waveform_sequence`` evaluation are native. ``EOBNRv2HM_ROM`` also
  accepts canonical positive-m mode subsets; duplicates are removed, and an
  empty array returns zero polarizations. The ``(2, 2)`` ROM is still
  reconstructed when it is not selected because its phase fixes the common
  reference calibration. As in LAL's sequence interface,
  ``long_asc_nodes`` is ignored; ``f_ref=0`` retains EOBNRv2's regular-grid
  reference convention. Spins, matter, eccentricity, testing-GR changes, and
  unmodeled modes retain the lalsimulation path. LAL itself ignores EOBNRv2HM
  mode arrays and does not implement EOBNRv2 frequency-sequence generation,
  so mode subsets are meaningful only on the native path and unsupported
  sequence requests still fail in the fallback.
- SEOBNRv4_ROM (FD aligned-spin): ``PYCBC_SEOBNRV4_NATIVE``. ROM interpolation,
  frequency interpolation, and waveform assembly run on the active Torch
  device using ``SEOBNRv4ROM_v3.0.hdf5``. The ROM family is native by default
  under ``TorchScheme``; set ``PYCBC_SEOBNRV4_NATIVE=0`` to opt out. The native
  path also covers
  ``SEOBNRv4_ROM_NRTidal``, ``SEOBNRv4_ROM_NRTidalv2``, and
  ``SEOBNRv4_ROM_NRTidalv2_NSBH`` with on-device tidal corrections. The NSBH
  path includes its multiplicative disruption-amplitude fit; scalar remnant
  fits and the disruption-radius root solve remain on the host. Regular-grid
  and arbitrary-frequency
  ``get_fd_waveform_sequence`` evaluation are both native; as in LAL's
  sequence interface, ``long_asc_nodes`` is ignored and ``f_ref=0`` uses the
  first supplied frequency. The separate time-domain ``SEOBNRv4`` port remains
  opt-in (default-off); transverse spins and unsupported tidal extensions
  continue to use lalsimulation.
- SEOBNRv4T_surrogate (FD aligned-spin tidal):
  ``PYCBC_SEOBNRV4T_SURROGATE_NATIVE``. The public
  ``SEOBNRv4T_surrogate_v2.0.0.hdf5`` data (with the compatible v1 file as a
  fallback) are loaded and validated on the host. Gaussian-process regression,
  corrected TaylorF2 spline construction, and waveform evaluation then run on
  the active Torch device for regular grids and arbitrary-frequency sequences.
  Supported requests are native by default under ``TorchScheme``; set the
  component flag to ``0`` to opt out. The sequence interface follows LAL by
  accepting unordered interior samples, using the first and last samples for
  frequency bounds, and ignoring ``long_asc_nodes``. Transverse spins,
  non-default PN orders, custom modes, and non-GR modifications retain the
  lalsimulation path. Apple MPS also retains the fallback pending a bounded
  single-precision accuracy policy.
- SEOBNRv4HM (regular TD higher modes): ``PYCBC_SEOBNRV4HM_NATIVE``. This
  CPU-only, opt-in selector-41 specialization evolves the aligned four-state
  EOB system, builds the internal ``(2,2)``, ``(2,1)``, ``(3,3)``, ``(4,4)``,
  and ``(5,5)`` modes, and reproduces LAL's direct high-rate decimation,
  zero-tail allocation, fixed ``l=5`` frequency checks, and public
  ``-pi/2`` polarization convention. The initial native envelope requires
  default orders and modes, aligned BBH inputs, and LAL's mass-ratio, spin,
  start-frequency, and Nyquist bounds; explicit modes, matter/non-GR
  extensions, transverse spins, and non-CPU devices retain standard LAL.
  Finite ``f_ref`` and ``f_final`` are ignored like the public TD path.
  Although this row shares the component flag with ``SEOBNRv4HM_ROM``, it is
  default-off: set the component or global Torch-native flag to ``1`` to opt
  in, and set the component flag to ``0`` to force fallback.
- SEOBNRv4HM_ROM (FD higher modes): ``PYCBC_SEOBNRV4HM_NATIVE``. ROM
  interpolation, low-frequency TaylorF2/ROM hybridization, harmonic evaluation,
  and waveform assembly run on the active Torch device using
  ``SEOBNRv4HMROM_v1.0.hdf5``; the file must be in ``$LAL_DATA_PATH`` or
  ``pycbc/waveform``. This ROM row remains enabled by default under
  ``TorchScheme`` despite sharing its flag with the opt-in regular-TD row;
  set ``PYCBC_SEOBNRV4HM_NATIVE=0`` to opt out. Regular-grid
  and arbitrary-frequency
  ``get_fd_waveform_sequence`` evaluation are native across the model's full
  positive-frequency range. Mode subsets and unordered sequence frequencies are
  supported, provided later samples do not fall below the inspiral spline domain
  set by the first frequency. As in LAL, this model ignores ``f_ref``; the
  sequence interface also ignores ``long_asc_nodes``. Apple MPS uses
  ``float32``/``complex64``; requests starting below ``M f_start = 1e-3``
  retain the LAL path to avoid low-frequency phase loss from single precision.
  For sequences, this guard uses the lowest supplied frequency, including for
  unordered inputs.
  Transverse spins, tides, non-GR parameters, and unsupported mode arrays also
  retain the lalsimulation path.
- SEOBNRv5_ROM (FD aligned-spin): ``PYCBC_SEOBNRV5_NATIVE``. The native path
  reconstructs the public ``SEOBNRv5ROM_v1.0.hdf5`` data and performs
  low-frequency TaylorF2 hybridization, ROM interpolation, polarization
  assembly, and the ``SEOBNRv5_ROM_NRTidalv3`` matter corrections on the active
  Torch device for regular grids and arbitrary-frequency sequences. Supported
  requests are native by default under ``TorchScheme``; set
  ``PYCBC_SEOBNRV5_NATIVE=0`` to opt out. Apple MPS uses single precision: BBH
  requests below
  ``M f_start = 0.002 (0.25 / eta)^(3/5)`` and all NRTidalv3 requests retain
  the LAL path to bound phase error. For BBH sequences, this guard uses the
  lowest supplied frequency, including for unordered inputs. Transverse spins,
  unsupported matter or non-GR parameters, and non-dominant mode requests also
  fall back to lalsimulation.
- SEOBNRv5HM_ROM (FD higher modes): ``PYCBC_SEOBNRV5HM_NATIVE``. The public
  ``SEOBNRv5HMROM_v1.0.hdf5`` data are read lazily on the host and the resulting
  model tensors are cached per device, while coefficient interpolation,
  sparse-basis reconstruction, low-frequency TaylorF2 hybridization, harmonic
  evaluation, and polarization summation run on the active Torch device. All
  seven directly modeled modes and their supported subsets are available on
  regular grids and arbitrary frequency sequences. Supported requests are
  native by default under
  ``TorchScheme``; set ``PYCBC_SEOBNRV5HM_NATIVE=0`` to opt out. The model
  ignores ``f_ref``; its sequence interface also ignores ``long_asc_nodes``
  and sets the inspiral spline domain from the first supplied frequency.
  Apple MPS uses single precision and retains the LAL path when the mass ratio
  exceeds 10 or when
  ``M f_start < 0.010 (0.25 / eta)^(3/5)``. The frequency guard uses the
  lowest sequence sample, including for unordered inputs. CPU and CUDA retain
  the model's mass-ratio limit of 100. Transverse spins, tides, non-GR
  parameters, and unsupported mode arrays also fall back to lalsimulation.
- SEOBNRv4P (TD, FD, and TD-mode precessing quadrupole):
  ``PYCBC_SEOBNRV4P_NATIVE``. This opt-in port reuses the native v4PHM
  adaptive EOB engine with v4P's model-specific mode boundary. Its default
  positive co-precessing modes are ``(2,2)`` and ``(2,1)``; an explicit
  ``(2,2)``-only request is also supported, while ``(2,2)`` must always be
  present. Opt-in ``get_td_waveform_modes`` returns the public initial
  inertial-frame convention, which is the negative of the internal modes, in
  descending ``(l,m)`` order with all five ``l=2`` modes materialized. Like
  LAL's mode API, it ignores ``coa_phase``, ``f_ref``, ``f_final``,
  ``ell_max``, inclination, and ``long_asc_nodes``; its Nyquist check always
  uses ``l=2``. The regular FD API retains PyCBC's padded, tapered TD-to-FD
  conversion, and arbitrary-frequency sequence evaluation uses the shared
  native nonuniform transform. Other modes, unsupported physical options,
  disabled ports, and inputs with ``mass1 < mass2`` retain the standard
  LAL/CPU path.
- SEOBNRv4PHM (TD, FD, and TD-mode precessing higher modes):
  ``PYCBC_SEOBNRV4PHM_NATIVE``. The adaptive EOB dynamics, factorized modes,
  NQC and ringdown attachment, frame rotations, harmonic projection, and
  polarization assembly run through the native Torch pipeline. The FD entry
  point applies a Torch real FFT to the same native TD result.
  Arbitrary-frequency ``get_fd_waveform_sequence`` evaluation is a native
  extension because LAL does not implement a sequence interface for this
  model. It starts the TD evolution at the lowest requested frequency, then
  applies the exact chunked nonuniform counterpart of that FFT on the active
  device. Unordered positive frequencies and duplicates are preserved,
  ``long_asc_nodes`` and ``f_final`` are ignored like other sequence paths,
  and samples may not exceed the Nyquist frequency set by ``delta_t``
  (2048 Hz at its default value). The default
  ``(2,2)``, ``(2,1)``, ``(3,3)``, ``(4,4)``, and ``(5,5)`` positive-m modes
  and supported subsets are available. Opt-in ``get_td_waveform_modes`` emits
  the negative internal inertial modes in LAL's descending degree/order, with
  every ``m`` from ``l=2`` through the largest selected degree and exact-zero
  intervening degree blocks. It ignores the same phase, reference-frequency,
  final-frequency, orientation, and ``ell_max`` inputs as LAL's mode API; its
  Nyquist check always uses ``l=5``, even for a ``(2,2)``-only request. Scalar
  setup, QNM interpolation, and some adaptive-control decisions remain
  Python/NumPy/SciPy work. Apple MPS uses ``float32``/``complex64`` because it
  does not provide the corresponding double-precision kernels. Non-default PN
  orders, eccentricity, matter and testing-GR parameters, alternate frame
  options, unsupported modes, disabled ports, and inputs with
  ``mass1 < mass2`` retain the standard LAL/CPU path.

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

The ``CoreCollapseBounce`` time-domain generator loads its principal-component
basis from HDF5 on the host, then caches and evaluates the requested PCA
reconstruction on the active Torch device. This path is selected automatically
under ``TorchScheme`` and does not require an environment flag.

Set a per-approximant flag to ``1`` to request the torch implementation or ``0``
to force the lalsimulation implementation. Unsupported options retain the
lalsimulation fallback. If a per-flag is unset, the global flag decides.
