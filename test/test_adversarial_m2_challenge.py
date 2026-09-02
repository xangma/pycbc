# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.

"""Empirical adversarial stress testing and parity suite for Milestone M2:
- Waveform generation under extreme physical parameters
- Multi-detector antenna projections and time-domain geometric response
- Relative-binning likelihood evaluations across execution paths
"""

import math
import unittest
import numpy as np
import scipy.special
import torch

import pycbc.waveform
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.scheme import TorchScheme
from pycbc.detector import Detector
import pycbc.detector.ground_torch as ground_torch
from pycbc.waveform.utils_torch import (
    apply_fseries_time_shift,
    fused_detector_strain_fd_torch,
)
try:
    import lal
    import lalsimulation
except ImportError:
    lal = None
    lalsimulation = None

from pycbc.inference.models import relbin_torch
from pycbc.filter import match


class TestM2AdversarialWaveforms(unittest.TestCase):
    """Adversarial stress-testing of native Torch waveforms vs reference."""

    def setUp(self):
        if lal is None or lalsimulation is None:
            self.skipTest("lal and lalsimulation required for CPU waveform reference")


    def _assert_waveform_parity(self, params, tol=1e-10, check_match=True):
        """Generate waveform with CPU and TorchScheme, verifying parity."""
        # 1. Standard LAL / PyCBC CPU reference
        hp_ref, hc_ref = pycbc.waveform.get_fd_waveform(**params)
        hp_ref_np = hp_ref.numpy()
        hc_ref_np = hc_ref.numpy()

        # 2. Native Torch scheme
        with TorchScheme("cpu"):
            hp_torch, hc_torch = pycbc.waveform.get_fd_waveform(**params)
        hp_torch_np = hp_torch.numpy()
        hc_torch_np = hc_torch.numpy()

        self.assertFalse(np.isnan(hp_torch_np).any(), "NaN in Torch hp")
        self.assertFalse(np.isnan(hc_torch_np).any(), "NaN in Torch hc")
        self.assertFalse(np.isinf(hp_torch_np).any(), "Inf in Torch hp")
        self.assertFalse(np.isinf(hc_torch_np).any(), "Inf in Torch hc")

        # Peak normalization for relative error
        max_hp = np.max(np.abs(hp_ref_np))
        max_hc = np.max(np.abs(hc_ref_np))

        if max_hp > 1e-30:
            rel_err_hp = np.max(np.abs(hp_torch_np - hp_ref_np)) / max_hp
            self.assertLess(
                rel_err_hp,
                tol,
                f"hp rel error {rel_err_hp} > {tol} "
                f"for {params['approximant']}",
            )

        if max_hc > 1e-30:
            rel_err_hc = np.max(np.abs(hc_torch_np - hc_ref_np)) / max_hc
            self.assertLess(
                rel_err_hc,
                tol,
                f"hc rel error {rel_err_hc} > {tol} "
                f"for {params['approximant']}",
            )

        if check_match and max_hp > 1e-30 and len(hp_ref) > 64:
            f_low = params.get("f_lower", 20.0)
            overlap_hp, _ = match(hp_ref, hp_torch, low_frequency_cutoff=f_low)
            self.assertGreater(
                overlap_hp,
                1.0 - tol,
                f"Plus match {overlap_hp} below 1 - {tol}",
            )

    def test_imrphenomxas_extreme_spins(self):
        """Stress-test IMRPhenomXAS at extreme aligned spins (+-0.99)."""
        configs = [
            {"spin1z": 0.99, "spin2z": 0.99},
            {"spin1z": -0.99, "spin2z": -0.99},
            {"spin1z": 0.99, "spin2z": -0.99},
            {"spin1z": 0.0, "spin2z": 0.0},
        ]
        for cfg in configs:
            params = {
                "approximant": "IMRPhenomXAS",
                "mass1": 35.0,
                "mass2": 25.0,
                "delta_f": 0.25,
                "f_lower": 20.0,
                "f_final": 1024.0,
                "distance": 100.0,
                "inclination": 0.4,
                **cfg,
            }
            self._assert_waveform_parity(params, tol=1e-10)

    def test_imrphenomxas_extreme_mass_ratios(self):
        """Stress-test IMRPhenomXAS at extreme mass ratios q=50 and q=80."""
        for q in (10.0, 50.0, 80.0):
            m2 = 2.0
            m1 = m2 * q
            params = {
                "approximant": "IMRPhenomXAS",
                "mass1": m1,
                "mass2": m2,
                "spin1z": 0.6,
                "spin2z": -0.3,
                "delta_f": 0.5,
                "f_lower": 20.0,
                "f_final": 1024.0,
                "distance": 100.0,
            }
            self._assert_waveform_parity(params, tol=1e-10)

    def test_imrphenomxas_extreme_frequencies_and_masses(self):
        """Stress-test IMRPhenomXAS at low (BNS) and high (IMBH) total mass."""
        # BNS: M = 2.8, f_low = 15 Hz, f_final = 2048 Hz
        params_bns = {
            "approximant": "IMRPhenomXAS",
            "mass1": 1.4,
            "mass2": 1.4,
            "spin1z": 0.05,
            "spin2z": -0.02,
            "delta_f": 0.25,
            "f_lower": 15.0,
            "f_final": 2048.0,
            "distance": 50.0,
        }
        self._assert_waveform_parity(params_bns, tol=1e-10)

        # IMBH: M = 400, f_low = 10 Hz
        params_imbh = {
            "approximant": "IMRPhenomXAS",
            "mass1": 250.0,
            "mass2": 150.0,
            "spin1z": 0.85,
            "spin2z": 0.75,
            "delta_f": 0.125,
            "f_lower": 10.0,
            "f_final": 512.0,
            "distance": 500.0,
        }
        self._assert_waveform_parity(params_imbh, tol=1e-10)

    def test_imrphenomd_extreme_parameters(self):
        """Stress-test IMRPhenomD at high spin and high mass ratio."""
        params = {
            "approximant": "IMRPhenomD",
            "mass1": 80.0,
            "mass2": 2.0,  # q = 40
            "spin1z": 0.985,
            "spin2z": -0.985,
            "delta_f": 0.25,
            "f_lower": 20.0,
            "f_final": 1024.0,
            "distance": 100.0,
            "inclination": 0.5,
        }
        self._assert_waveform_parity(params, tol=1e-10)

    def test_imrphenomxhm_higher_modes_and_inclinations(self):
        """Stress-test IMRPhenomXHM with higher modes across inclinations."""
        for incl in (0.0, math.pi / 4.0, math.pi / 2.0, 0.95 * math.pi):
            params = {
                "approximant": "IMRPhenomXHM",
                "mass1": 60.0,
                "mass2": 15.0,  # q = 4
                "spin1z": 0.7,
                "spin2z": -0.5,
                "delta_f": 0.25,
                "f_lower": 20.0,
                "f_final": 1024.0,
                "distance": 100.0,
                "inclination": incl,
            }
            # Tol 5e-3 covers LAL multibanding vs full evaluation
            self._assert_waveform_parity(params, tol=5e-3, check_match=False)

    def test_imrphenomxp_precessing_spins(self):
        """Stress-test IMRPhenomXP with strong precessing spin dynamics."""
        params = {
            "approximant": "IMRPhenomXP",
            "mass1": 45.0,
            "mass2": 15.0,
            "spin1x": 0.6,
            "spin1y": 0.4,
            "spin1z": 0.3,
            "spin2x": -0.3,
            "spin2y": 0.5,
            "spin2z": -0.4,
            "delta_f": 0.25,
            "f_lower": 20.0,
            "f_final": 1024.0,
            "distance": 100.0,
            "inclination": math.pi / 3.0,
        }
        self._assert_waveform_parity(params, tol=1e-10)

    def test_taylorf2_extreme_spins_and_masses(self):
        """Stress-test TaylorF2 under high spins and unequal masses."""
        params = {
            "approximant": "TaylorF2",
            "mass1": 1.9,
            "mass2": 1.1,
            "spin1z": 0.85,
            "spin2z": -0.85,
            "delta_f": 0.5,
            "f_lower": 25.0,
            "f_final": 1024.0,
            "distance": 40.0,
        }
        self._assert_waveform_parity(params, tol=1e-10)


class TestM2AdversarialDetectorAndProjections(unittest.TestCase):
    """Adversarial stress-testing of detector projections and ground_torch."""

    def test_torch_sine_integral_domain_branches_and_boundaries(self):
        """Empirically test _torch_sine_integral across domains and bounds."""
        # 1. Pure small domain (< 4.0)
        x_small = torch.linspace(-3.99, 3.99, 1000, dtype=torch.float64)
        si_small_torch = ground_torch._torch_sine_integral(x_small).numpy()
        si_small_ref = np.array(
            [scipy.special.sici(v)[0] for v in x_small.numpy()]
        )
        np.testing.assert_allclose(
            si_small_torch, si_small_ref, atol=1e-13, rtol=1e-13
        )

        # 2. Pure large domain (> 8.0)
        x_large = torch.linspace(8.01, 500.0, 1000, dtype=torch.float64)
        si_large_torch = ground_torch._torch_sine_integral(x_large).numpy()
        si_large_ref = np.array(
            [scipy.special.sici(v)[0] for v in x_large.numpy()]
        )
        np.testing.assert_allclose(
            si_large_torch, si_large_ref, atol=1e-12, rtol=1e-12
        )

        # 3. Mixed domain spanning -50 to +50
        x_mixed = torch.linspace(-50.0, 50.0, 2000, dtype=torch.float64)
        si_mixed_torch = ground_torch._torch_sine_integral(x_mixed).numpy()
        si_mixed_ref = np.array(
            [scipy.special.sici(v)[0] for v in x_mixed.numpy()]
        )
        np.testing.assert_allclose(
            si_mixed_torch, si_mixed_ref, atol=1e-12, rtol=1e-12
        )

        # 4. Critical transition boundaries
        critical_vals = [
            0.0, 1e-15, -1e-15,
            4.0 - 1e-12, 4.0, 4.0 + 1e-12,
            -4.0 + 1e-12, -4.0, -4.0 - 1e-12,
            8.0 - 1e-12, 8.0, 8.0 + 1e-12,
            -8.0 + 1e-12, -8.0, -8.0 - 1e-12,
            1e4, -1e4,
        ]
        x_crit = torch.tensor(critical_vals, dtype=torch.float64)
        si_crit_torch = ground_torch._torch_sine_integral(x_crit).numpy()
        si_crit_ref = np.array(
            [scipy.special.sici(v)[0] for v in critical_vals]
        )
        np.testing.assert_allclose(
            si_crit_torch, si_crit_ref, atol=1e-12, rtol=1e-12
        )

    def test_fused_detector_strain_fd_torch_parity(self):
        """Verify fused_detector_strain_fd_torch matches un-fused multi-ifo."""
        n_freq = 512
        delta_f = 0.5
        hp = torch.randn(n_freq, dtype=torch.complex128)
        hc = torch.randn(n_freq, dtype=torch.complex128)

        # 5 detectors: H1, L1, V1, K1, I1
        fp = [0.8, -0.4, 0.6, -0.7, 0.2]
        fc = [0.2, 0.5, -0.3, 0.4, -0.8]
        dt = [0.005, -0.012, 0.021, -0.003, 0.015]

        for kmin in (0, 32, 128):
            fused = fused_detector_strain_fd_torch(
                hp, hc, fp, fc, dt, delta_f, kmin=kmin
            )
            self.assertEqual(fused.shape, (5, n_freq))

            # Reference un-fused calculation
            for d in range(5):
                comb = fp[d] * hp + fc[d] * hc
                ref_d = comb.clone()
                freqs = torch.arange(kmin, n_freq, dtype=torch.float64)
                phase = -2.0 * math.pi * delta_f * dt[d] * freqs
                shift = torch.complex(torch.cos(phase), torch.sin(phase))
                ref_d[kmin:] = comb[kmin:] * shift

                diff = torch.max(torch.abs(fused[d] - ref_d)).item()
                self.assertLess(
                    diff,
                    1e-14,
                    f"Fused mismatch for det {d} with kmin={kmin}: {diff}",
                )

    def test_apply_fseries_time_shift_in_place_and_copy(self):
        """Verify apply_fseries_time_shift accuracy and in-place vs copy."""
        with TorchScheme("cpu"):
            n = 256
            delta_f = 0.5
            data = torch.randn(n, dtype=torch.complex128)
            ts = FrequencySeries(
                TorchArrayData(data.clone()), delta_f=delta_f, copy=False
            )

            dt = 0.035
            # Copy=True
            shifted_copy = apply_fseries_time_shift(ts, dt, kmin=10, copy=True)
            # Verify original unaffected
            self.assertTrue(torch.equal(ts._data.tensor, data))

            # Direct mathematical reference
            ref_data = data.clone()
            freqs = torch.arange(10, n, dtype=torch.float64)
            phase = -2.0 * math.pi * dt * delta_f * freqs
            shift = torch.complex(torch.cos(phase), torch.sin(phase))
            ref_data[10:] = ref_data[10:] * shift

            diff = torch.max(
                torch.abs(shifted_copy._data.tensor - ref_data)
            ).item()
            self.assertLess(diff, 1e-14, f"Time shift copy error: {diff}")

            # In-place shift
            shifted_inplace = apply_fseries_time_shift(
                ts, dt, kmin=10, copy=False
            )
            self.assertTrue(
                torch.equal(ts._data.tensor, shifted_inplace._data.tensor)
            )
            diff_ip = torch.max(torch.abs(ts._data.tensor - ref_data)).item()
            self.assertLess(
                diff_ip, 1e-14, f"Time shift in-place error: {diff_ip}"
            )

    def test_ground_torch_project_wave_against_cpu_detector(self):
        """Stress-test ground_torch.project_wave against CPU Detector."""
        if lal is None or lalsimulation is None:
            self.skipTest(
                "lal and lalsimulation required for CPU Detector.project_wave"
            )
        delta_t = 1.0 / 4096.0
        n_pts = 4096
        t = np.arange(n_pts) * delta_t
        epoch = 1126259462.0

        # Create chirp-like test polarizations
        hp_np = np.sin(2.0 * np.pi * (30.0 * t + 50.0 * t**2)) * np.exp(-t)
        hc_np = np.cos(2.0 * np.pi * (30.0 * t + 50.0 * t**2)) * np.exp(-t)

        hp_cpu = TimeSeries(hp_np, delta_t=delta_t, epoch=epoch)
        hc_cpu = TimeSeries(hc_np, delta_t=delta_t, epoch=epoch)

        with TorchScheme("cpu"):
            hp_torch = TimeSeries(
                TorchArrayData(torch.tensor(hp_np, dtype=torch.float64)),
                delta_t=delta_t,
                epoch=epoch,
                copy=False,
            )
            hc_torch = TimeSeries(
                TorchArrayData(torch.tensor(hc_np, dtype=torch.float64)),
                delta_t=delta_t,
                epoch=epoch,
                copy=False,
            )

            # Test across multiple detectors and extreme angles
            sky_cases = [
                ("H1", 0.0, math.pi / 2.0, 0.0),        # North pole
                ("L1", math.pi, -math.pi / 2.0, 0.5),   # South pole
                ("V1", math.pi / 2.0, 0.0, math.pi / 4.0),  # Equator
                ("H1", 2.0 * math.pi, 0.3, math.pi / 2.0),
            ]

            for ifo_name, ra, dec, psi in sky_cases:
                det = Detector(ifo_name)
                # CPU reference
                res_cpu = det.project_wave(hp_cpu, hc_cpu, ra, dec, psi)
                # Torch implementation
                res_torch = ground_torch.project_wave(
                    det, hp_torch, hc_torch, ra, dec, psi
                )

                self.assertEqual(len(res_cpu), len(res_torch))
                self.assertEqual(res_cpu.start_time, res_torch.start_time)
                self.assertEqual(res_cpu.delta_t, res_torch.delta_t)

                res_torch_np = res_torch.numpy()
                res_cpu_np = res_cpu.numpy()

                max_val = np.max(np.abs(res_cpu_np))
                max_diff = np.max(np.abs(res_torch_np - res_cpu_np))
                rel_diff = max_diff / max_val if max_val > 0 else max_diff

                self.assertLess(
                    rel_diff,
                    1e-12,
                    f"project_wave mismatch for {ifo_name}: rel={rel_diff}",
                )


class TestM2AdversarialRelbin(unittest.TestCase):
    """Adversarial stress-testing of relative-binning likelihood."""

    def test_linearized_filter_and_norm_factored_identities(self):
        """Verify optimized factored formulas are mathematically exact."""
        n_bins = 128
        n_samples = 64

        # Random complex ratios and summary data
        ratio = torch.randn(n_samples, n_bins + 1, dtype=torch.complex128)
        a0 = torch.randn(n_bins, dtype=torch.complex128)
        a1 = torch.randn(n_bins, dtype=torch.complex128)
        b0 = torch.randn(n_bins, dtype=torch.float64)
        b1 = torch.randn(n_bins, dtype=torch.float64)

        # 1. Linearized filter
        opt_filter = relbin_torch._linearized_filter(ratio, a0, a1)
        # Un-factored naive reference
        ratio_lo = ratio[..., :-1]
        ratio_hi = ratio[..., 1:]
        ref_filter = (
            (a0 * ratio_lo + a1 * (ratio_hi - ratio_lo)).sum(dim=-1).conj()
        )
        diff_filter = torch.max(torch.abs(opt_filter - ref_filter)).item()
        self.assertLess(
            diff_filter, 1e-12, f"Filter difference: {diff_filter}"
        )

        # 2. Linearized norm
        opt_norm = relbin_torch._linearized_norm(ratio, b0, b1)
        power = ratio.real.square() + ratio.imag.square()
        power_lo = power[..., :-1]
        power_hi = power[..., 1:]
        ref_norm = (
            (b0 * power_lo + b1 * (power_hi - power_lo)).sum(dim=-1).real
        )
        diff_norm = torch.max(torch.abs(opt_norm - ref_norm)).item()
        self.assertLess(diff_norm, 1e-12, f"Norm difference: {diff_norm}")

        # 3. Linearized cross
        ratio2 = torch.randn(n_samples, n_bins + 1, dtype=torch.complex128)
        opt_cross = relbin_torch._linearized_cross(ratio, ratio2, a0, a1)
        cross = ratio * ratio2.conj()
        ref_cross = (
            (a0 * cross[..., :-1] + a1 * (cross[..., 1:] - cross[..., :-1]))
            .sum(dim=-1)
        )
        diff_cross = torch.max(torch.abs(opt_cross - ref_cross)).item()
        self.assertLess(diff_cross, 1e-12, f"Cross difference: {diff_cross}")

    def test_batched_likelihood_parts_parity_and_vmap(self):
        """Stress-test batched_likelihood_parts across batches and vmap."""
        n_bins = 64
        freqs = torch.linspace(20.0, 1024.0, n_bins + 1, dtype=torch.float64)
        hp = torch.randn(n_bins + 1, dtype=torch.complex128)
        hc = torch.randn(n_bins + 1, dtype=torch.complex128)
        h00 = torch.randn(n_bins + 1, dtype=torch.complex128) + 1.0

        a0 = torch.randn(n_bins, dtype=torch.complex128)
        a1 = torch.randn(n_bins, dtype=torch.complex128)
        b0 = torch.randn(n_bins, dtype=torch.float64)
        b1 = torch.randn(n_bins, dtype=torch.float64)

        for N in (1, 16, 128, 512):
            fp = torch.randn(N, dtype=torch.float64)
            fc = torch.randn(N, dtype=torch.float64)
            dtc = torch.tensor(
                [0.0, 1e-5, -1e-5, 0.05, -0.05, 0.5, -0.5, 1.5][:N]
                + [0.01 * (i % 10) for i in range(max(0, N - 8))],
                dtype=torch.float64,
            )[:N]

            # 1. Vectorized broadcast (use_vmap=False)
            d_inn_b, h_norm_b = relbin_torch.batched_likelihood_parts(
                freqs, fp, fc, dtc, hp, hc, h00, a0, a1, b0, b1, use_vmap=False
            )

            # 2. Vectorized vmap (use_vmap=True)
            d_inn_v, h_norm_v = relbin_torch.batched_likelihood_parts(
                freqs, fp, fc, dtc, hp, hc, h00, a0, a1, b0, b1, use_vmap=True
            )

            diff_inn_v = torch.max(torch.abs(d_inn_b - d_inn_v)).item()
            diff_norm_v = torch.max(torch.abs(h_norm_b - h_norm_v)).item()
            self.assertLess(
                diff_inn_v,
                1e-12,
                f"vmap inner mismatch at N={N}: {diff_inn_v}",
            )
            self.assertLess(
                diff_norm_v,
                1e-12,
                f"vmap norm mismatch at N={N}: {diff_norm_v}",
            )

            # 3. Compare elementwise with scalar likelihood_parts
            for i in range(min(N, 16)):
                d_inn_s, h_norm_s = relbin_torch.likelihood_parts(
                    freqs, fp[i], fc[i], dtc[i], hp, hc, h00, a0, a1, b0, b1
                )
                diff_s_inn = torch.abs(d_inn_b[i] - d_inn_s).item()
                diff_s_norm = torch.abs(h_norm_b[i] - h_norm_s).item()
                self.assertLess(
                    diff_s_inn,
                    1e-12,
                    f"Scalar inner mismatch at sample {i}: {diff_s_inn}",
                )
                self.assertLess(
                    diff_s_norm,
                    1e-12,
                    f"Scalar norm mismatch at sample {i}: {diff_s_norm}",
                )

    def test_multi_detector_relbin_evaluations(self):
        """Stress-test likelihood_parts_det and multi-channel evaluations."""
        n_bins = 32
        freqs = torch.linspace(20.0, 512.0, n_bins + 1, dtype=torch.float64)
        hp = torch.randn(n_bins + 1, dtype=torch.complex128)
        hc = torch.randn(n_bins + 1, dtype=torch.complex128)
        h00 = torch.randn(n_bins + 1, dtype=torch.complex128) + 1.0

        a0 = torch.randn(n_bins, dtype=torch.complex128)
        a1 = torch.randn(n_bins, dtype=torch.complex128)
        b0 = torch.randn(n_bins, dtype=torch.float64)
        b1 = torch.randn(n_bins, dtype=torch.float64)

        N = 30
        fp = torch.randn(N, dtype=torch.float64)
        fc = torch.randn(N, dtype=torch.float64)
        dtc = torch.randn(N, dtype=torch.float64) * 0.01

        # Channel combination fp * hp + fc * hc
        channel = (
            fp.unsqueeze(-1) * hp.unsqueeze(0)
            + fc.unsqueeze(-1) * hc.unsqueeze(0)
        )

        # Single detector channel evaluation
        d_inn, h_norm = relbin_torch.likelihood_parts_det(
            freqs, dtc, channel, h00, a0, a1, b0, b1
        )
        self.assertEqual(d_inn.shape, (N,))
        self.assertEqual(h_norm.shape, (N,))

        # Verify against batched_likelihood_parts
        d_inn_ref, h_norm_ref = relbin_torch.batched_likelihood_parts(
            freqs, fp, fc, dtc, hp, hc, h00, a0, a1, b0, b1
        )
        diff_inn = torch.max(torch.abs(d_inn - d_inn_ref)).item()
        diff_norm = torch.max(torch.abs(h_norm - h_norm_ref)).item()
        self.assertLess(diff_inn, 1e-12, f"Channel inner diff: {diff_inn}")
        self.assertLess(diff_norm, 1e-12, f"Channel norm diff: {diff_norm}")

    def test_likelihood_parts_multi_cross_term(self):
        """Stress-test likelihood_parts_multi cross term between waveforms."""
        n_bins = 32
        freqs = torch.linspace(20.0, 512.0, n_bins + 1, dtype=torch.float64)
        hp = torch.randn(n_bins + 1, dtype=torch.complex128)
        hc = torch.randn(n_bins + 1, dtype=torch.complex128)
        h00 = torch.randn(n_bins + 1, dtype=torch.complex128) + 1.0

        hp2 = torch.randn(n_bins + 1, dtype=torch.complex128)
        hc2 = torch.randn(n_bins + 1, dtype=torch.complex128)
        h002 = torch.randn(n_bins + 1, dtype=torch.complex128) + 1.0

        a0 = torch.randn(n_bins, dtype=torch.complex128)
        a1 = torch.randn(n_bins, dtype=torch.complex128)

        N = 25
        fp = torch.randn(N, dtype=torch.float64)
        fc = torch.randn(N, dtype=torch.float64)
        dtc = torch.randn(N, dtype=torch.float64) * 0.01

        fp2 = torch.randn(N, dtype=torch.float64)
        fc2 = torch.randn(N, dtype=torch.float64)
        dtc2 = torch.randn(N, dtype=torch.float64) * 0.01

        cross = relbin_torch.likelihood_parts_multi(
            freqs, fp, fc, dtc, hp, hc, h00,
            fp2, fc2, dtc2, hp2, hc2, h002, a0, a1
        )
        self.assertEqual(cross.shape, (N,))

        # Elementwise check
        for i in range(N):
            single_cross = relbin_torch.likelihood_parts_multi(
                freqs, fp[i], fc[i], dtc[i], hp, hc, h00,
                fp2[i], fc2[i], dtc2[i], hp2, hc2, h002, a0, a1
            )
            diff_i = torch.abs(cross[i] - single_cross).item()
            self.assertLess(
                diff_i, 1e-12, f"Cross term diff at sample {i}: {diff_i}"
            )


if __name__ == "__main__":
    unittest.main()
