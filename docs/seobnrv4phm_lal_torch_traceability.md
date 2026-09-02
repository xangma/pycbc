# SEOBNRv4HM/P/PHM LAL-to-torch traceability

This is the working map for the torch-native regular-TD `SEOBNRv4HM`,
`SEOBNRv4P`, and `SEOBNRv4PHM` ports. The first table is exhaustive for
functions defined in LAL's
`lalsimulation/lib/LALSimIMRSpinPrecEOBv4P.c`. The second table lists the
nontrivial external LAL helper families called by those functions. A separate
selector-41 table records the aligned regular-TD specialization rather than
treating it as an alias of the precessing engine.

Status key:

- `DONE`: torch has a direct counterpart and local tests cover the intended behavior.
- `PARTIAL`: torch has a counterpart, but the LAL behavior is not fully reproduced.
- `MISSING`: no torch equivalent yet.
- `STRUCTURAL`: behavior is represented by Python/Torch orchestration rather than a function-for-function port.
- `N/A`: LAL-only allocation, cleanup, or unused helper.

## Direct functions in `LALSimIMRSpinPrecEOBv4P.c`

| LAL function | LAL line | Role in LAL workflow | Torch counterpart | Status | Notes |
| --- | ---: | --- | --- | --- | --- |
| `XLALEOBHighestInitialFreq` | 189 | Validate maximum starting frequency. | `pycbc/waveform/seobnrv4phm_dynamics.py:505` `highest_initial_freq` | DONE | Direct scalar port. |
| `argmax` | 198 | Find the maximum element index for robust peak search. | `torch.argmax` inside `pycbc.waveform.seobnrv4phm_peak.find_peak_time` | STRUCTURAL | LAL-only helper used by `XLALEOBFindRobustPeak`; torch uses native tensor argmax. |
| `XLALEOBFindRobustPeak` | 227 | Locate a robust local maximum near a search guess. | `pycbc.waveform.seobnrv4phm_peak.find_peak_time` | DONE | Shared helper used by the native PHM timing path and covered by focused peak tests. |
| `XLALEOBSpinPrecStopCondition_v4` | 309 | Older precessing stop-condition callback. | None | N/A | Marked `UNUSED` in LAL and not referenced by the waveform pipeline. |
| `XLALEOBSpinPrecStopConditionBasedOnPR` | 351 | Adaptive integrator stop condition. | `pycbc/waveform/seobnrv4phm_torch.py` `_lal_stop_quantities_from_cartesian`, `_lal_stop_quantities_from_reduced`, `_run._stop_fn`; `pycbc/waveform/seobnrv4phm_ode.py:88` `integrate` | DONE | LAL's Cartesian radius, momentum, derivative, omega thresholds, callback state, and AdaS/HiS cadence are represented. Independent Torch and C/GSL arithmetic is not bit-identical. |
| `XLALEOBSpinPrecAlignedStopCondition` | 603 | Aligned-spin adaptive stop condition for the AdaS branch. | `pycbc/waveform/seobnrv4phm_torch.py` `_run._stop_fn` aligned branch | DONE | Peak/radius stop logic and callback state follow LAL. |
| `XLALSpinPrecAlignedHiSRStopCondition` | 630 | Aligned-spin high-sampling-rate stop condition. | `pycbc/waveform/seobnrv4phm_torch.py` `_run._stop_fn` HiS branch | DONE | The HiS peak counter behavior follows LAL. |
| `XLALSetup_EOB__std_mode_array_structure` | 663 | Build the default LAL mode array. | `pycbc/waveform/seobnrv4phm_dynamics.py` `normalize_mode_array`, `LM_DEFAULT` | DONE | Torch carries normalized `(ell, m)` tuples instead of a LAL bit-array structure. |
| `XLALCheck_EOB_mode_array_structure` | 694 | Validate requested LAL modes. | `pycbc/waveform/seobnrv4phm_dynamics.py` `normalize_mode_array` | DONE | Torch now rejects negative/unsupported requested modes and returns LAL traversal order for supported PHM modes. |
| `XLALSimIMRSpinPrecEOBWaveform` | 763 | Public TD wrapper around `WaveformAll`. | `pycbc/waveform/seobnrv4phm_torch.py` `seobnrv4phm_td_torch`, `seobnrv4phm_modes_torch`; `seobnrv4phm_fd_torch` for PyCBC's FD API | STRUCTURAL | The native TD entry point invokes the complete pipeline. The mode entry point exposes its inertial modes in the public sign/layout convention, and the FD entry point conditions the TD result with Torch. With native flags disabled, PyCBC's shared dispatchers retain the CPU/LAL paths. |
| `SEOBGetLMaxInModeArray` | 958 | Determine max requested `l`. | `pycbc/waveform/seobnrv4phm_dynamics.py` `normalize_mode_array`; native waveform `waveform_ell_max` | STRUCTURAL | Torch carries explicit normalized `params.mode_array` and derives needed `l` sets locally. |
| `CAmpPhaseSequence_Init` | 1264 | Allocate complex amplitude/phase time series. | tensors and per-mode dictionaries in `pycbc/waveform/seobnrv4phm_torch.py` | STRUCTURAL | Python containers replace the C allocation helper. |
| `CAmpPhaseSequence_Destroy` | 1311 | Free complex amplitude/phase time series. | Python GC | N/A | No torch behavior needed. |
| `SphHarmListEOBNonQCCoeffs_Destroy` | 1332 | Free NQC coefficient mode list. | Python GC | N/A | No torch behavior needed. |
| `SphHarmListEOBNonQCCoeffs_AddMode` | 1352 | Add NQC coefficient data for one mode. | `params.nqc_a_map`, `params.nqc_b_map` | STRUCTURAL | Torch stores NQC coefficients in keyed maps instead of linked-list nodes. |
| `SphHarmListCAmpPhaseSequence_Destroy` | 1413 | Free spherical-harmonic mode list. | Python GC | N/A | No torch behavior needed. |
| `SphHarmListCAmpPhaseSequence_AddMode` | 1438 | Add one complex amplitude/phase mode to a mode list. | mode dictionaries in `pycbc/waveform/seobnrv4phm_torch.py` | STRUCTURAL | Torch represents mode lists as dictionaries keyed by `(ell, m)`. |
| `SEOBdynamics_Destroy` | 1499 | Free LAL dynamics struct. | Python GC / tensors | N/A | No torch behavior needed. |
| `SEOBdynamics_Init` | 1506 | Allocate LAL dynamics struct. | `_TrajectorySegments`, tensors, dicts | STRUCTURAL | Represented by tensors and small Python containers. |
| `SEOBCalculateChiS` | 1577 | Symmetric spin projection. | `pycbc/waveform/seobnrv4phm_torch.py` `_spin_combos`, `_dynamic_spin_projection_combos`, NQC peak projections | DONE FOR DEFAULT MODES | Static, peak, and per-sample P-frame mode projections are represented. |
| `SEOBCalculateChiA` | 1580 | Antisymmetric spin projection. | `pycbc/waveform/seobnrv4phm_torch.py` `_spin_combos`, `_dynamic_spin_projection_combos`, NQC peak projections | DONE FOR DEFAULT MODES | Static, peak, and per-sample P-frame mode projections are represented. |
| `SEOBCalculatetplspin` | 1588 | Spin combination passed into waveform coefficients. | `pycbc/waveform/seobnrv4phm_torch.py` `_dynamic_spin_projection_combos` | DONE FOR DEFAULT MODES | Main P-frame generation uses LAL's weighted `s1dotZ/s2dotZ` branch; 21/55 residual calibration uses the unweighted `chiS/chiA` branch found in LAL's calibration helper. |
| `SEOBCalculateSigmaKerr` | 1620 | Kerr spin vector. | `pycbc/waveform/seobnrv4phm_dynamics.py:615` `_eob_potentials`, `pycbc/waveform/seobnrv4phm_dynamics.py:551` `_augmented_spin_chi` | DONE | Used in Hamiltonian/potentials path. |
| `SEOBCalculateSigmaStar` | 1638 | Effective test-particle spin vector. | `pycbc/waveform/seobnrv4phm_dynamics.py:617` `_eob_potentials` | DONE | Used in Hamiltonian/potentials path. |
| `SEOBComputeExtendedSEOBdynamics` | 1659 | Derive polar dynamics, omega, spin projections, and Hamiltonian from integrated Cartesian dynamics. | `pycbc/waveform/seobnrv4phm_torch.py` `_series_from_states`, `_dynamic_spin_projection_combos`, `pycbc/waveform/seobnrv4phm_dynamics.py` `_eob_potentials` | STRUCTURAL | Torch derives the complete series consumed downstream directly from tensors; a C allocation-layout equivalent is unnecessary. |
| `SEOBInitialConditions` | 1869 | Precessing initial condition solve. | `pycbc/waveform/seobnrv4phm_dynamics.py` `initial_cartesian_conditions`, `_precessing_spherical_initial_conditions`, `_precessing_ic_radial_momentum_summary`; `pycbc/waveform/seobnrv4phm_multiroot.py` | DONE | Mirrors LAL's scaled GSL `hybrids` solve, derivative ordering, coefficient-cache side effects, radial-momentum balance, spherical/Cartesian round trip, and final tortoise transform. The returned 14-state matches LAL to roundoff-level IC tolerances. |
| `SEOBConvertSpinAlignedDynamicsToGenericSpins` | 1996 | Convert aligned-spin trajectory into generic-spin layout. | `pycbc/waveform/seobnrv4phm_dynamics.py:1826` `reduced_state_to_cartesian_state`, aligned-spin branch handling in `EOBParams.__post_init__` | STRUCTURAL | Torch uses one reduced state layout for both aligned and precessing branches. |
| `SEOBIntegrateDynamics` | 2060 | Configure and run LAL adaptive RK integrator. | `pycbc/waveform/seobnrv4phm_torch.py:2145` `_integrate_traj`, `pycbc/waveform/seobnrv4phm_ode.py:88` `integrate` | DONE | AdaS/HiS split, RKF45 control order, minimum-step behavior, dense sampling, handoff, and stop callbacks follow LAL. Roundoff-level RHS differences can select nearby adaptive steps. |
| `SEOBCalculatehlmAmpPhase` | 2270 | Build one P-frame mode's amp/phase, with optional NQC. | `pycbc/waveform/seobnrv4phm_torch.py` `_build_coprecessing_modes`, `_factorized_residual_power` | DONE | Dynamic coefficients, all supported residual phase powers, 21/55 calibration, and NQC application are represented. |
| `SEOBCalculateSphHarmListhlmAmpPhase` | 2421 | Loop over requested P-frame modes. | `pycbc/waveform/seobnrv4phm_torch.py` `_build_coprecessing_modes`, `_populate_nqc_coeffs_lal_v4` | DONE | Requested positive modes and NQC processing follow LAL traversal. |
| `SEOBLocateTimePeakOmega` | 2469 | Locate peak orbital frequency. | `pycbc/waveform/seobnrv4phm_torch.py:1878` `_phm_merger_timing` | DONE | Uses shared peak helper and fallback argmax. |
| `SEOBLocateTimePeakModeAmp` | 2532 | Locate peak mode amplitude. | None | N/A | Marked `UNUSED` in LAL file. |
| `SEOBInterpolateDynamicsAtTime` | 2582 | Cubic interpolation of dynamics at a requested time. | `pycbc/waveform/seobnrv4phm_torch.py:2085` `_interpolate_traj_at_time`, `pycbc/waveform/seobnrv4phm_torch.py:2066` `_interp_series_cubic_lal_local` | DONE | Local tests cover the LAL-style interpolation window behavior. |
| `SEOBLFrameVectors` | 2644 | Spin vectors in the L-frame. | `pycbc/waveform/seobnrv4phm_torch.py:1548` `_l_frame_spin_vectors_from_state`, `pycbc/waveform/seobnrv4phm_torch.py:1858` `_l_frame_spin_projections` | DONE | Used for final spin/mass and NQC peak projections. |
| `SEOBJfromDynamics` | 2724 | Final total angular momentum vector. | `pycbc/waveform/seobnrv4phm_torch.py:2398` `J_final` calculation | DONE | Uses peak-Omega state. |
| `SEOBLhatfromDynamics` | 2772 | Radiation-frame `Lhat` or `LNhat`. | `pycbc/waveform/seobnrv4phm_torch.py:1858` `_l_frame_spin_projections`, `pycbc/waveform/seobnrv4phm_dynamics.py:1637` `_orbital_basis_from_L_phi` | PARTIAL | L-frame path exists; LN flag behavior is not fully exposed in torch. |
| `SEOBBuildJframeVectors` | 2833 | Build final-J frame basis. | `pycbc/waveform/seobnrv4phm_torch.py:529` `_build_J_frame` | DONE | Recently made source-faithful. |
| `SEOBEulerI2JFromJframeVectors` | 2925 | Constant I-to-J Euler angles. | `pycbc/waveform/seobnrv4phm_torch.py:564` `_euler_from_basis` | DONE | Covered by frame tests. |
| `SEOBCalculateSphHarmListNQCCoefficientsV4` | 2954 | Compute NQC coefficients for the requested modes. | `pycbc/waveform/seobnrv4phm_torch.py` `_populate_nqc_coeffs_lal_v4`, `_solve_nqc_coeffs_lal_series` | DONE | Linear solve, basis series, NR targets, and requested-positive-mode traversal follow LAL. |
| `FindClosestIndex` | 3115 | Find nearest index in an increasing vector. | `pycbc/waveform/seobnrv4phm_torch.py:335` `_nearest_index_increasing` | DONE | Used by the torch final-mass and ringdown-attachment paths where LAL uses this helper. |
| `FindClosestValueInIncreasingVector` | 3139 | Return the closest value in an increasing vector. | `_nearest_index_increasing` plus tensor indexing | STRUCTURAL | Torch uses the index helper directly and indexes the target vector at the call site. |
| `SEOBGetFinalSpinMass` | 3147 | Final mass/spin fit at `r=10M` dynamics time. | `pycbc/waveform/seobnrv4phm_torch.py:1566` `_final_mass_spin_from_adas_10M`, `pycbc/waveform/seobnrv4phm_torch.py:1444` `_final_mass_spin_prec` | DONE | Includes AdaS `r=10M` handling and fit fallback. |
| `SEOBAttachRDToSphHarmListhPlm` | 3198 | Attach ringdown to P-frame modes. | `pycbc/waveform/seobnrv4phm_torch.py:1590` `_attach_ringdown_modes` | DONE | Recently aligned with LAL ringdown buffer behavior. |
| `SEOBJoinTimeVector` | 3441 | Join AdaS prefix with HiS+RD time vector. | `pycbc/waveform/seobnrv4phm_torch.py:2503`-`2509` time-vector construction | DONE | Native path follows LAL AdaS/HiS split. |
| `SEOBJoinDynamics` | 3512 | Join AdaS and HiS dynamics. | `pycbc/waveform/seobnrv4phm_torch.py:2032` `_join_adaptive_his_trajectories`, `pycbc/waveform/seobnrv4phm_torch.py:2512`-`2517` Euler dynamics join | DONE | Used before Euler construction. |
| `SEOBJoinSphHarmListhlm` | 3557 | Join AdaS and HiS/RD P-frame modes. | `pycbc/waveform/seobnrv4phm_torch.py:2127` `_join_mode_dicts` | DONE | Structural dict-based port. |
| `SEOBAmplitudePeakFromAmp22Amp21` | 3633 | Compute final waveform epoch from 22/21 peaks. | `pycbc/waveform/seobnrv4phm_torch.py` `_amplitude_peak_from_22_21` | DONE | Uses LAL's discrete first-maximum rule on `|h22|^2`, adding `|h21|^2` only when mode 21 is requested, then sets the FD epoch to `-M*tPeak`. |
| `SEOBEulerJ2PFromDynamics` | 3706 | J-to-P Euler angles before attachment. | `pycbc/waveform/seobnrv4phm_torch.py:653` `_euler_j2p` | DONE | Recent work fixed initial separation vector and minimal-rotation gamma. |
| `SEOBEulerJ2PPostMergerExtension` | 3912 | Extend Euler angles through ringdown. | `pycbc/waveform/seobnrv4phm_torch.py:755` `_extend_euler_from_attach_times` | DONE | Follows attach-index extension on the P-mode grid. |
| `SEOBWignerDAmp` | 4013 | Wigner-d amplitude helper. | `pycbc/waveform/seobnrv4phm_torch.py:907` `_wigner_d_element` | DONE | Shared by time-varying and constant rotations. |
| `SEOBWignerDPhase` | 4016 | Wigner phase helper. | `pycbc/waveform/seobnrv4phm_torch.py:970` `_rotate_interpolate_modes_jframe`, `pycbc/waveform/seobnrv4phm_torch.py:1031` `_rotate_modes_constant` | DONE | Implemented through complex phase factors. |
| `SEOBRotateInterpolatehJlmReImFromSphHarmListhPlmAmpPhase` | 4031 | Rotate P-frame modes to J-frame and interpolate onto output grid. | `pycbc/waveform/seobnrv4phm_torch.py:970` `_rotate_interpolate_modes_jframe` | DONE | Local tests cover LAL interpolation and rotation ordering. |
| `SEOBRotatehIlmFromhJlm` | 4303 | Constant J-to-I rotation. | `pycbc/waveform/seobnrv4phm_torch.py:1031` `_rotate_modes_constant`; `_public_time_domain_modes` | DONE | Direct mode-rotation counterpart. The public adapter negates the internal inertial modes, emits descending complete degree/order blocks, and zero-fills absent degrees like LAL. |
| `SEOBComputehplushcrossFromhIlm` | 4396 | Project inertial modes to plus/cross. | `pycbc/waveform/seobnrv4phm_torch.py:1074` `_polarizations_from_modes` | DONE | Direct spin-weighted spherical-harmonic projection. |
| `SEOBGetNumberOfModesInModeArray` | 4445 | Count requested modes. | `params.mode_array` / dict key sets | STRUCTURAL | No separate torch function needed. |
| `SEOBGetModeNumbersFromModeArray` | 4472 | Materialize requested mode list. | `pycbc/waveform/seobnrv4phm_torch.py:2338` mode-array normalization | STRUCTURAL | Torch stores the tuple directly in `EOBParams`. |
| `XLALEOBCheckNyquistFrequency` | 4503 | Validate Nyquist constraints. | `pycbc/waveform/seobnrv4phm_torch.py` `_check_nyquist_frequency`, `_ringdown_omega_for_nyquist` | DONE | Native path checks the requested `ellMaxForNyquistCheck` or waveform `ell_max` before integration and rejects sample rates below the checked ringdown frequency's Nyquist requirement. The public TD-mode adapters fix this check at `l=2` for v4P and `l=5` for v4PHM, independent of the selected modes, matching LAL. |
| `XLALSimIMRSpinPrecEOBWaveformAll` | 4571 | Full 14-step SEOBNRv4PHM generation pipeline. | `pycbc/waveform/seobnrv4phm_torch.py` `_seobnrv4phm_td_native`; `_seobnrv4phm_fd_native` for TD-to-FD conditioning | DONE FOR SUPPORTED API | The complete default-L-frame pipeline and all supported PHM modes are native Torch. The remaining difference from LAL is numerical trajectory drift, quantified below. |

## External LAL helper families used by the waveform

| LAL helper | LAL source | Torch counterpart | Status | Notes |
| --- | --- | --- | --- | --- |
| `XLALSpinPrecHcapRvecDerivative`, `XLALSpinPrecHcapRvecDerivative_exact` | `LALSimIMRSpinEOBHamiltonianPrec.c`, exact variant in v3opt source | `pycbc/waveform/seobnrv4phm_dynamics.py` `rhs_cartesian_full`, `rhs_cartesian_projected`, finite-diff/autograd Hamiltonian helpers | DONE | Full Cartesian weighted-spin state, tortoise momentum equations, flux, x/p/spin derivatives, and phase variables are represented. Analytic spin gradients retain the derivative through weighted-state overrides; they agree with the LAL-style finite-difference rates at about `1e-9` relative in the parity fixture. |
| `XLALAdaptiveRungeKutta4*` | LAL support integrator | `pycbc/waveform/seobnrv4phm_ode.py:21` `rk45_step`, `pycbc/waveform/seobnrv4phm_ode.py:115` `integrate` | DONE | Implements LAL/GSL RKF45 error scaling and step adjustment, AdaS minimum-step forcing, HiS sampling, rejection behavior, and callback ordering. Compiler/library arithmetic remains independently rounded. |
| `XLALSimIMRCalculateSpinPrecEOBHCoeffs_v2`, `XLALEOBSpinPrecCalcSEOBHCoeffConstants` | `LALSimIMRCalculateSpinPrecEOBHCoeffs.c`, `LALSimIMRSpinEOB.h` | `pycbc/waveform/seobnrv4phm_constants.py:58` `compute_spin_aligned_hcoeffs`, `pycbc/waveform/seobnrv4phm_dynamics.py:571` `_refresh_hcoeffs` | DONE | Per-state Hamiltonian coefficient refresh and LAL evaluation-order variants are represented. |
| `XLALSimIMREOBCalcSpinPrecFacWaveformCoefficients` | `LALSimIMRSpinEOBFactorizedWaveformCoefficientsPrec.c:41/66` | `pycbc/waveform/seobnrv4phm_dynamics.py:90` `_factorized_rho_aux_delta`, `pycbc/waveform/seobnrv4phm_torch.py` `_dynamic_spin_projection_combos` | DONE FOR DEFAULT MODES | Torch now refreshes coefficients from per-sample `chiS`, `chiA`, and `tplspin`. The main waveform loop uses LAL's weighted `s1dotZ/s2dotZ` `tplspin`; the 21/55 calibration helper uses LAL's unweighted `chiS/chiA` variant. |
| `XLALSimIMREOBCalcCalibCoefficientHigherModesPrec` | `LALSimIMRSpinEOBFactorizedWaveformPrec.c:80/1399` | `pycbc/waveform/seobnrv4phm_torch.py` `_higher_mode_residual_calibration` | DONE | Computes `cal21`/`cal55` from the uncalibrated residual at the mode attachment point and wires them into `_factorized_rho_aux_delta` as `f21v7c`/`f55v5c`. |
| `XLALSimIMRSpinEOBGetPrecSpinFactorizedWaveform` | `LALSimIMRSpinEOBFactorizedWaveformPrec.c:49/546` | `pycbc/waveform/seobnrv4phm_torch.py` `_build_coprecessing_modes`, `pycbc/waveform/seobnrv4phm_dynamics.py` `_factorized_rho_aux_delta`, `_factorized_flux`, `non_keplerian_vphi`, `pycbc/waveform/seobnrv4phm_torch.py` `_tail_factor_complex` | DONE FOR SUPPORTED MODES | Factorized residuals, tail, source, non-Keplerian factor, calibration, and flux are represented for the PHM mode set accepted by LAL's public v4PHM mode array. |
| `XLALSimIMRSpinEOBNonQCCorrection` | `LALSimIMREOBNQCCorrection.c:488` | `pycbc/waveform/seobnrv4phm_torch.py` NQC application in `_build_coprecessing_modes` | DONE | Formula and application order follow LAL. |
| `XLALSimIMRSpinEOBCalculateNQCCoefficientsV4` | `LALSimIMREOBNQCCorrection.c:2359` | `pycbc/waveform/seobnrv4phm_torch.py` `_populate_nqc_coeffs_lal_v4`, `_solve_nqc_coeffs_lal_series` | DONE | Linear solve, basis vectors, NR targets, and requested-mode loop follow LAL. |
| `XLALSimIMREOBGetNRSpinPeakDeltaTv4` | `LALSimIMREOBNQCCorrection.c:1699` | `pycbc.waveform.seobnrv4phm_coeffs.peak_delta_t_v4` | DONE | Used by `_phm_merger_timing` and NQC solve. |
| `XLALSimIMREOBFinalMassSpinPrec` | `LALSimBlackHoleRingdownPrec.c:39` | `pycbc/waveform/seobnrv4phm_torch.py:1444` `_final_mass_spin_prec` | DONE | Direct fit port. |
| `XLALSimIMREOBGenerateQNMFreqV2FromFinalPrec` | `LALSimBlackHoleRingdownPrec.c:1926` | `pycbc/waveform/seobnrv4phm_torch.py` `_qnm_re_im` | DONE | Embedded Cardoso tables cover every supported public mode and are used for ringdown and the Euler post-merger rate. |
| `XLALSimIMREOBAttachFitRingdown` plus RD coefficient helpers | `LALSimIMREOBHybridRingdownPrec.c` | `pycbc/waveform/seobnrv4phm_torch.py:1590` `_attach_ringdown_modes`, `pycbc/waveform/seobnrv4phm_torch.py:1382`-`1394` RD coefficient helpers | DONE | Ringdown buffer and coefficient behavior have parity tests. |

## Aligned selector-41 specialization

| LAL selector-41 behavior | Torch counterpart | Status | Notes |
| --- | --- | --- | --- |
| Nonoptimized circular-orbit and radial-momentum initial conditions | `_seobnrv4hm_nonoptimized_circular_root`, `_seobnrv4hm_nonoptimized_ic_radial_summary` | DONE | Uses the three-variable hybrids root, nested numerical Hamiltonian derivatives, and a deterministic ported solver path; it does not silently dispatch to host GSL. |
| Four-state aligned dynamics and selector-41 stop callbacks | `_seobnrv4hm_aligned_rhs4`, `_integrate_traj(..., selector41=True)` | DONE | Integrates `[r, phi, pr*, pphi]` with fixed Hamiltonian coefficients, aligned factorized flux, distinct AdaS/HiS stop state, and fail-closed exact tolerances. |
| Five internal higher modes, aligned NQC/calibration, and ringdown | `_seobnrv4phm_td_native(..., _aligned_v4hm=True)` | DONE FOR SUPPORTED API | All five positive mode families are retained internally; the 21/55 calibration uses the aligned phase derivative, 55 attaches 10 M early, and `l=5` controls the frequency bounds. |
| Power-of-two HiS grid and public output allocation | `_seobnrv4hm_resample_factor`, `_seobnrv4hm_direct_assemble_modes` | DONE | Direct decimation, integer-floor copy counts, and the exact zero allocation tail replace the precessing path's mode interpolation. |
| Public polarization and dispatch contract | `_seobnrv4hm_public_polarizations`, `seobnrv4hm_td_torch`, `seobnrv4hm_native_supported` | DONE FOR SUPPORTED API | Preserves selector 41's public `-pi/2` convention, remaining node rotation, CPU-only support gate, and standard-LAL fallback. |

## Current parity implication

Regular-TD `SEOBNRv4HM` is a distinct CPU-only, opt-in selector-41 port. It
always constructs all five internal positive modes and currently accepts only
the default public mode/order configuration with aligned BBH inputs; explicit
mode selection and unsupported physical controls retain standard LAL. Its
shared `PYCBC_SEOBNRV4HM_NATIVE` flag does not inherit the ROM row's default-on
policy. The public wrapper follows selector 41's polarization sign and node
rotation, ignores finite `f_ref`/`f_final`, and preserves the fixed `l=5`
start-frequency and Nyquist checks.

The selector-41 regression compares the raw public strain on exact physical
timestamps without a fitted time, phase, or amplitude adjustment. A natural
cubic evaluation of the native result on the LAL timestamps remains below
`2e-3` normalized complex error in the reference fixture. The discrete
start-frequency crossing can differ by one sample, while the residual epoch is
less than one tenth of a sample. Rebuilding the corrected C path with floating
point contraction disabled reproduces the ported circular root and brings the
trajectory into close numerical agreement, so the remaining installed-LAL
drift reflects compiled arithmetic rather than a different dynamics or
projection convention.

The native port is complete for LAL's public/default `L`-frame API and the five
positive P-frame modes accepted by SEOBNRv4PHM: `(2,2)`, `(2,1)`, `(3,3)`,
`(4,4)`, and `(5,5)`. The optional internal `LN`-frame flag is not exposed.
Enabling `PYCBC_SEOBNRV4PHM_NATIVE` (or the global Torch-native switch) under
``TorchScheme`` invokes this implementation through ``get_td_waveform``,
``get_fd_waveform``, and ``get_td_waveform_modes``; no second implementation
gate is required. The mode interface returns the negative of the internal
initial-inertial-frame modes in descending complete `l,m` blocks, with absent
intervening degrees represented by exact zeros. It follows LAL in ignoring
coalescence phase, reference/final frequency, observer orientation, and
`ell_max`, and fixes its Nyquist check at `l=5`. Unsupported options and
reversed component-mass labels retain the standard LAL/CPU path.

`SEOBNRv4P` uses the same LAL engine restricted to its public mode set. Its
default is `(2,2)` plus `(2,1)`, with `(2,2)` required; the native wrapper also
accepts the explicit `(2,2)` subset. Direct LAL comparisons confirm that v4P is
identical, including epoch, to v4PHM evaluated with the corresponding mode
subset. `PYCBC_SEOBNRV4P_NATIVE` exposes that boundary through the TD,
TD-mode, conditioned FD, and arbitrary-frequency APIs. Its TD-mode interface
materializes all five inertial `l=2` modes and fixes the Nyquist check at
`l=2`; the same observer/reference inputs are ignored. Unsupported mode arrays
preserve the existing standard LAL/CPU fallback.

The resolved trajectory discrepancy was in the analytic spin derivative. The
Cartesian state stores weighted spins, and passing those state values as
Hamiltonian overrides had detached the corresponding `chi_i -> S_i/M^2`
gradient. Consequently, the spin-precession part of the Torch RHS was missing
coefficient and Hamiltonian contributions. The override now preserves the exact
state value while retaining that derivative. Analytic spin rates agree with the
LAL-style finite-difference rates to about `1e-9` relative in the regression
fixture; the former full-trajectory discrepancy was about 11%.

The final staged checks separate implementation error from independent
floating-point integration:

| Check | Result |
| --- | ---: |
| Initial 14-state | Roundoff-level agreement with the LAL row |
| Analytic versus LAL-style finite-difference spin rates | `~1e-9` relative |
| Ringdown attachment time | Exact in the boundary fixture |
| Joined-trajectory end time | `-1.24e-2 M` versus LAL |
| Joined-trajectory final radius | `+1.24e-3 M` versus LAL |
| Exact LAL trajectory through Torch modes/RD/conditioning | `2.61e-6` aligned relative error |
| Independent end-to-end Torch waveform | `8.82e-5` aligned relative error |
| Independent end-to-end phase scatter after alignment | `8.56e-5` rad |

The exact-trajectory check bypasses Torch initial conditions and integration;
its `2.61e-6` result bounds the combined P-frame modes, NQC, ringdown, rotations,
polarization projection, and TD-to-FD conditioning error. The difference between
that result and the `8.82e-5` full-waveform result comes from nearby adaptive
step choices after roundoff-level Torch-versus-C/GSL RHS differences. Requiring
bit identity would require executing the same compiled C/GSL arithmetic and
would no longer be an independent Torch port.

The opt-in full-waveform regression therefore uses sub-`1e-3` raw and
`2.5e-4` aligned bounds. These retain platform margin around the measured
result while rejecting the former 11% defect by more than two orders of
magnitude.
