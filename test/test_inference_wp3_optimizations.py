# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Unit tests for Work Package 3 optimizations:
1. Fused inner product evaluation (_fused_inner_hd_hh) in inference models.
2. Vectorized multi-detector NetworkGeometry in pycbc.detector.
3. Batched relative-binning summary evaluation in relbin_torch.
"""

import unittest
import numpy as np
import torch

from pycbc.types import FrequencySeries
from pycbc.detector import Detector, NetworkGeometry
from pycbc.inference.models.tools import _fused_inner_hd_hh, _inner, _real_inner
from pycbc.inference.models import relbin_torch


class TestWP3FusedInner(unittest.TestCase):
    def test_numpy_fused_inner(self):
        delta_f = 0.25
        n = 1024
        # Unweighted
        h = FrequencySeries(
            np.random.randn(n) + 1j * np.random.randn(n), delta_f=delta_f
        )
        d = FrequencySeries(
            np.random.randn(n) + 1j * np.random.randn(n), delta_f=delta_f
        )
        cplx_hd_ref = _inner(h, d)
        hh_ref = _real_inner(h, h)

        cplx_hd_fused, hh_fused = _fused_inner_hd_hh(h, d)
        np.testing.assert_allclose(cplx_hd_fused, cplx_hd_ref, rtol=1e-12)
        np.testing.assert_allclose(hh_fused, hh_ref, rtol=1e-12)

        # Weighted
        weight = FrequencySeries(np.random.rand(n) + 0.1, delta_f=delta_f)
        h_whitened = h.copy()
        h_whitened *= weight
        cplx_hd_w_ref = _inner(h_whitened, d)
        hh_w_ref = _real_inner(h_whitened, h_whitened)

        cplx_hd_w_fused, hh_w_fused = _fused_inner_hd_hh(h, d, weight=weight)
        np.testing.assert_allclose(cplx_hd_w_fused, cplx_hd_w_ref, rtol=1e-12)
        np.testing.assert_allclose(hh_w_fused, hh_w_ref, rtol=1e-12)

    def test_torch_fused_inner(self):
        n = 512
        for dtype in (torch.complex64, torch.complex128):
            ht = torch.randn(n, dtype=dtype)
            dt = torch.randn(n, dtype=dtype)
            wt = torch.rand(n, dtype=ht.real.dtype) + 0.1

            cplx_hd_fused, hh_fused = _fused_inner_hd_hh(ht, dt, weight=wt)
            hw = ht * wt
            cplx_hd_ref = _inner(hw, dt)
            hh_ref = _real_inner(hw, hw)

            torch.testing.assert_close(cplx_hd_fused, cplx_hd_ref)
            torch.testing.assert_close(hh_fused, hh_ref)

    def test_empty_slice(self):
        h = np.array([], dtype=np.complex128)
        d = np.array([], dtype=np.complex128)
        cplx_hd, hh = _fused_inner_hd_hh(h, d)
        self.assertEqual(cplx_hd, 0j)
        self.assertEqual(hh, 0.0)

    def test_numpy_batched_fused_inner(self):
        n_samples = 20
        n_freq = 256
        h_batch = np.random.randn(n_samples, n_freq) + 1j * np.random.randn(n_samples, n_freq)
        d_1d = np.random.randn(n_freq) + 1j * np.random.randn(n_freq)
        w_1d = np.random.rand(n_freq) + 0.1

        # Batched call
        cplx_hd_batch, hh_batch = _fused_inner_hd_hh(h_batch, d_1d, weight=w_1d)
        self.assertEqual(cplx_hd_batch.shape, (n_samples,))
        self.assertEqual(hh_batch.shape, (n_samples,))

        # Compare against single 1D calls
        for i in range(n_samples):
            cplx_i, hh_i = _fused_inner_hd_hh(h_batch[i], d_1d, weight=w_1d)
            np.testing.assert_allclose(cplx_hd_batch[i], cplx_i, rtol=1e-12)
            np.testing.assert_allclose(hh_batch[i], hh_i, rtol=1e-12)

        # 3D Batch (B, N, F)
        b_dim = 3
        h_3d = np.random.randn(b_dim, n_samples, n_freq) + 1j * np.random.randn(b_dim, n_samples, n_freq)
        cplx_hd_3d, hh_3d = _fused_inner_hd_hh(h_3d, d_1d, weight=w_1d)
        self.assertEqual(cplx_hd_3d.shape, (b_dim, n_samples))
        self.assertEqual(hh_3d.shape, (b_dim, n_samples))
        for b in range(b_dim):
            for n in range(n_samples):
                cplx_bn, hh_bn = _fused_inner_hd_hh(h_3d[b, n], d_1d, weight=w_1d)
                np.testing.assert_allclose(cplx_hd_3d[b, n], cplx_bn, rtol=1e-12)
                np.testing.assert_allclose(hh_3d[b, n], hh_bn, rtol=1e-12)

    def test_torch_batched_fused_inner(self):
        n_samples = 15
        n_freq = 128
        for dtype in (torch.complex64, torch.complex128):
            ht_batch = torch.randn(n_samples, n_freq, dtype=dtype)
            dt_1d = torch.randn(n_freq, dtype=dtype)
            wt_1d = torch.rand(n_freq, dtype=ht_batch.real.dtype) + 0.1

            cplx_hd_batch, hh_batch = _fused_inner_hd_hh(ht_batch, dt_1d, weight=wt_1d)
            self.assertEqual(cplx_hd_batch.shape, torch.Size([n_samples]))
            self.assertEqual(hh_batch.shape, torch.Size([n_samples]))

            for i in range(n_samples):
                cplx_i, hh_i = _fused_inner_hd_hh(ht_batch[i], dt_1d, weight=wt_1d)
                torch.testing.assert_close(cplx_hd_batch[i], cplx_i)
                torch.testing.assert_close(hh_batch[i], hh_i)

            # Test autograd backwards through batched inner products
            ht_grad = torch.randn(n_samples, n_freq, dtype=dtype, requires_grad=True)
            cplx_grad, hh_grad = _fused_inner_hd_hh(ht_grad, dt_1d, weight=wt_1d)
            loss = cplx_grad.real.sum() + hh_grad.sum()
            loss.backward()
            self.assertIsNotNone(ht_grad.grad)


class TestWP3NetworkGeometry(unittest.TestCase):
    def setUp(self):
        self.ifos = ['H1', 'L1', 'V1', 'K1']
        self.net = NetworkGeometry(self.ifos)
        self.ra = 1.37
        self.dec = -1.26
        self.pol = 2.76
        self.t_gps = 1126259462.0

    def test_scalar_network(self):
        fp_net, fc_net, delay_net = self.net.antenna_pattern_and_time_delay(
            self.ra, self.dec, self.pol, self.t_gps
        )
        for i, ifo in enumerate(self.ifos):
            d = Detector(ifo)
            fp_ref, fc_ref, delay_ref = d.antenna_pattern_and_time_delay(
                self.ra, self.dec, self.pol, self.t_gps
            )
            np.testing.assert_allclose(fp_net[i], fp_ref, rtol=1e-12)
            np.testing.assert_allclose(fc_net[i], fc_ref, rtol=1e-12)
            np.testing.assert_allclose(delay_net[i], delay_ref, rtol=1e-12)

        # test separate antenna_pattern and time_delay_from_earth_center
        fp, fc = self.net.antenna_pattern(
            self.ra, self.dec, self.pol, self.t_gps
        )
        delay = self.net.time_delay_from_earth_center(
            self.ra, self.dec, self.t_gps
        )
        np.testing.assert_allclose(fp, fp_net, rtol=1e-12)
        np.testing.assert_allclose(fc, fc_net, rtol=1e-12)
        np.testing.assert_allclose(delay, delay_net, rtol=1e-12)

    def test_torch_batched_network(self):
        batch_size = 25
        ra_t = torch.linspace(0.1, 6.0, batch_size, dtype=torch.float64)
        dec_t = torch.linspace(-1.4, 1.4, batch_size, dtype=torch.float64)
        pol_t = torch.linspace(0.0, 3.14, batch_size, dtype=torch.float64)
        t_t = torch.linspace(
            1126259460.0, 1126259470.0, batch_size, dtype=torch.float64
        )

        fp_net, fc_net, delay_net = self.net.antenna_pattern_and_time_delay(
            ra_t, dec_t, pol_t, t_t
        )
        self.assertEqual(fp_net.shape, torch.Size([4, batch_size]))

        for i, ifo in enumerate(self.ifos):
            d = Detector(ifo)
            fp_ref, fc_ref, delay_ref = d.antenna_pattern_and_time_delay(
                ra_t, dec_t, pol_t, t_t
            )
            torch.testing.assert_close(fp_net[i], fp_ref)
            torch.testing.assert_close(fc_net[i], fc_ref)
            torch.testing.assert_close(delay_net[i], delay_ref)


class TestWP3RelbinTorchBatched(unittest.TestCase):
    def setUp(self):
        self.num_bins = 16
        self.num_freqs = 129
        self.delta_f = 0.25
        self.full_freqs = torch.linspace(
            20.0, 500.0, self.num_freqs, dtype=torch.float64
        )
        edge_indices = torch.arange(0, self.num_bins + 1, dtype=torch.int64) * 8
        self.edge_indices = edge_indices
        self.bins = torch.stack(
            (edge_indices[:-1], edge_indices[1:]),
            dim=1,
        )
        self.fedges = self.full_freqs[edge_indices]

        self.psd = torch.ones(self.num_freqs, dtype=torch.float64)
        self.full_hp = torch.randn(self.num_freqs, dtype=torch.complex128)
        self.full_hc = torch.randn(self.num_freqs, dtype=torch.complex128)
        self.full_h00 = torch.randn(self.num_freqs, dtype=torch.complex128)

        # Summary data on the full frequency grid
        self.a0, self.a1 = relbin_torch.summary_product(
            self.full_hp, self.full_h00, self.psd, self.full_freqs, self.bins, self.delta_f
        )
        self.b0, self.b1 = relbin_torch.summary_product(
            self.full_h00, self.full_h00, self.psd, self.full_freqs, self.bins, self.delta_f
        )

        # Waveforms evaluated at bin edges (sparse frequency grid)
        self.hp_edge = self.full_hp[edge_indices]
        self.hc_edge = self.full_hc[edge_indices]
        self.h00_edge = self.full_h00[edge_indices]

    def test_batched_summary_product(self):
        # 1D single waveform
        a0_single, a1_single = relbin_torch.summary_product(
            self.full_hp, self.full_h00, self.psd, self.full_freqs, self.bins, self.delta_f
        )
        self.assertEqual(a0_single.shape, torch.Size([self.num_bins]))

        # 2D batch of N waveforms
        batch_n = 10
        hp_batch = self.full_hp.unsqueeze(0).expand(batch_n, -1)
        h00_batch = self.full_h00.unsqueeze(0).expand(batch_n, -1)
        a0_batch, a1_batch = relbin_torch.summary_product(
            hp_batch, h00_batch, self.psd, self.full_freqs, self.bins, self.delta_f
        )
        self.assertEqual(a0_batch.shape, torch.Size([batch_n, self.num_bins]))
        for i in range(batch_n):
            torch.testing.assert_close(a0_batch[i], a0_single)
            torch.testing.assert_close(a1_batch[i], a1_single)

        # 3D tensor (B, N, F)
        b_dim = 3
        hp_3d = self.full_hp.unsqueeze(0).unsqueeze(0).expand(b_dim, batch_n, -1)
        h00_3d = self.full_h00.unsqueeze(0).unsqueeze(0).expand(b_dim, batch_n, -1)
        a0_3d, a1_3d = relbin_torch.summary_product(
            hp_3d, h00_3d, self.psd, self.full_freqs, self.bins, self.delta_f
        )
        self.assertEqual(a0_3d.shape, torch.Size([b_dim, batch_n, self.num_bins]))

    def test_batched_likelihood_parts_broadcasting_and_vmap(self):
        n_samples = 20
        fp = torch.linspace(0.1, 1.0, n_samples, dtype=torch.float64)
        fc = torch.linspace(-0.5, 0.5, n_samples, dtype=torch.float64)
        dtc = torch.linspace(-0.01, 0.01, n_samples, dtype=torch.float64)

        # Evaluate one-by-one as reference
        filt_ref = []
        norm_ref = []
        for i in range(n_samples):
            f_i, n_i = relbin_torch.likelihood_parts(
                self.fedges, fp[i], fc[i], dtc[i],
                self.hp_edge, self.hc_edge, self.h00_edge,
                self.a0, self.a1, self.b0.real, self.b1.real
            )
            filt_ref.append(f_i)
            norm_ref.append(n_i)
        filt_ref = torch.stack(filt_ref)
        norm_ref = torch.stack(norm_ref)

        # Broadcasted 2D evaluation
        filt_bc, norm_bc = relbin_torch.likelihood_parts(
            self.fedges, fp, fc, dtc,
            self.hp_edge, self.hc_edge, self.h00_edge,
            self.a0, self.a1, self.b0.real, self.b1.real
        )
        torch.testing.assert_close(filt_bc, filt_ref)
        torch.testing.assert_close(norm_bc, norm_ref)

        # batched_likelihood_parts (broadcast)
        filt_b, norm_b = relbin_torch.batched_likelihood_parts(
            self.fedges, fp, fc, dtc,
            self.hp_edge, self.hc_edge, self.h00_edge,
            self.a0, self.a1, self.b0.real, self.b1.real,
            use_vmap=False
        )
        torch.testing.assert_close(filt_b, filt_ref)
        torch.testing.assert_close(norm_b, norm_ref)

        # batched_likelihood_parts (torch.vmap)
        filt_vmap, norm_vmap = relbin_torch.batched_likelihood_parts(
            self.fedges, fp, fc, dtc,
            self.hp_edge, self.hc_edge, self.h00_edge,
            self.a0, self.a1, self.b0.real, self.b1.real,
            use_vmap=True
        )
        torch.testing.assert_close(filt_vmap, filt_ref)
        torch.testing.assert_close(norm_vmap, norm_ref)

    def test_batched_3d_likelihood_parts(self):
        b_dim, n_dim = 2, 5
        fp = torch.randn(b_dim, n_dim, dtype=torch.float64)
        fc = torch.randn(b_dim, n_dim, dtype=torch.float64)
        dtc = torch.randn(b_dim, n_dim, dtype=torch.float64) * 0.01

        filt_3d, norm_3d = relbin_torch.likelihood_parts(
            self.fedges, fp, fc, dtc,
            self.hp_edge, self.hc_edge, self.h00_edge,
            self.a0, self.a1, self.b0.real, self.b1.real
        )
        self.assertEqual(filt_3d.shape, torch.Size([b_dim, n_dim]))
        self.assertEqual(norm_3d.shape, torch.Size([b_dim, n_dim]))

        # verify against single point evaluations
        for b in range(b_dim):
            for n in range(n_dim):
                f_single, n_single = relbin_torch.likelihood_parts(
                    self.fedges, fp[b, n], fc[b, n], dtc[b, n],
                    self.hp_edge, self.hc_edge, self.h00_edge,
                    self.a0, self.a1, self.b0.real, self.b1.real
                )
                torch.testing.assert_close(filt_3d[b, n], f_single)
                torch.testing.assert_close(norm_3d[b, n], n_single)


class TestWP3RelbinTorchCompile(unittest.TestCase):
    """Test torch.compile compatibility and zero graph breaks for relbin_torch."""

    def test_torch_compile_linearized_filter_and_summaries(self):
        compiled_filter = torch.compile(relbin_torch._linearized_filter, dynamic=True)
        compiled_summaries = torch.compile(relbin_torch._summaries, dynamic=True)

        for num_bins in (8, 16, 32, 64):
            for batch_shape in ((), (10,), (2, 5)):
                ratio = torch.randn(*batch_shape, num_bins + 1, dtype=torch.complex128)
                a0 = torch.randn(num_bins, dtype=torch.complex128)
                a1 = torch.randn(num_bins, dtype=torch.complex128)
                b0 = torch.randn(num_bins, dtype=torch.float64)
                b1 = torch.randn(num_bins, dtype=torch.float64)

                filt_eager = relbin_torch._linearized_filter(ratio, a0, a1)
                filt_compiled = compiled_filter(ratio, a0, a1)
                torch.testing.assert_close(filt_compiled, filt_eager)

                s_filt_eager, s_norm_eager = relbin_torch._summaries(ratio, a0, a1, b0, b1)
                s_filt_compiled, s_norm_compiled = compiled_summaries(ratio, a0, a1, b0, b1)
                torch.testing.assert_close(s_filt_compiled, s_filt_eager)
                torch.testing.assert_close(s_norm_compiled, s_norm_eager)

        # Verify no graph breaks
        ratio_test = torch.randn(10, 17, dtype=torch.complex128)
        a0_test = torch.randn(16, dtype=torch.complex128)
        a1_test = torch.randn(16, dtype=torch.complex128)
        b0_test = torch.randn(16, dtype=torch.float64)
        b1_test = torch.randn(16, dtype=torch.float64)

        exp_f = torch._dynamo.explain(relbin_torch._linearized_filter)(ratio_test, a0_test, a1_test)
        self.assertEqual(exp_f.graph_count, 1)
        self.assertEqual(len(exp_f.break_reasons), 0)

        exp_s = torch._dynamo.explain(relbin_torch._summaries)(
            ratio_test, a0_test, a1_test, b0_test, b1_test
        )
        self.assertEqual(exp_s.graph_count, 1)
        self.assertEqual(len(exp_s.break_reasons), 0)

    def test_torch_compile_likelihood_parts_dynamic_bins(self):
        compiled_lp = torch.compile(relbin_torch.likelihood_parts, dynamic=True)

        for num_bins in (8, 16, 32):
            num_freqs = num_bins + 1
            fedges = torch.linspace(20.0, 500.0, num_freqs, dtype=torch.float64)
            hp_edge = torch.randn(num_freqs, dtype=torch.complex128)
            hc_edge = torch.randn(num_freqs, dtype=torch.complex128)
            h00_edge = torch.randn(num_freqs, dtype=torch.complex128)
            a0 = torch.randn(num_bins, dtype=torch.complex128)
            a1 = torch.randn(num_bins, dtype=torch.complex128)
            b0 = torch.randn(num_bins, dtype=torch.float64)
            b1 = torch.randn(num_bins, dtype=torch.float64)

            # 1. Scalar parameters
            fp_s = torch.tensor(0.5, dtype=torch.float64)
            fc_s = torch.tensor(-0.3, dtype=torch.float64)
            dtc_s = torch.tensor(0.001, dtype=torch.float64)
            filt_s_ref, norm_s_ref = relbin_torch.likelihood_parts(
                fedges, fp_s, fc_s, dtc_s, hp_edge, hc_edge, h00_edge, a0, a1, b0, b1
            )
            filt_s_comp, norm_s_comp = compiled_lp(
                fedges, fp_s, fc_s, dtc_s, hp_edge, hc_edge, h00_edge, a0, a1, b0, b1
            )
            torch.testing.assert_close(filt_s_comp, filt_s_ref)
            torch.testing.assert_close(norm_s_comp, norm_s_ref)

            # 2. Batched 1D parameters
            batch_n = 15
            fp_b = torch.linspace(0.1, 1.0, batch_n, dtype=torch.float64)
            fc_b = torch.linspace(-0.5, 0.5, batch_n, dtype=torch.float64)
            dtc_b = torch.linspace(-0.01, 0.01, batch_n, dtype=torch.float64)
            filt_b_ref, norm_b_ref = relbin_torch.likelihood_parts(
                fedges, fp_b, fc_b, dtc_b, hp_edge, hc_edge, h00_edge, a0, a1, b0, b1
            )
            filt_b_comp, norm_b_comp = compiled_lp(
                fedges, fp_b, fc_b, dtc_b, hp_edge, hc_edge, h00_edge, a0, a1, b0, b1
            )
            torch.testing.assert_close(filt_b_comp, filt_b_ref)
            torch.testing.assert_close(norm_b_comp, norm_b_ref)

        # Verify no graph breaks
        exp_lp = torch._dynamo.explain(relbin_torch.likelihood_parts)(
            fedges, fp_b, fc_b, dtc_b, hp_edge, hc_edge, h00_edge, a0, a1, b0, b1
        )
        self.assertEqual(exp_lp.graph_count, 1)
        self.assertEqual(len(exp_lp.break_reasons), 0)


class TestWP2MultiDetectorBatchedLikelihood(unittest.TestCase):
    def setUp(self):
        from pycbc.inference.models import GaussianNoise, MarginalizedPhaseGaussianNoise
        from pycbc.psd import aLIGOZeroDetHighPower

        np.random.seed(42)
        self.ifos = ['H1', 'L1', 'V1']
        self.delta_f = 0.25
        self.f_len = 1024 * 4 + 1
        self.f_low = 30.0

        psd = aLIGOZeroDetHighPower(self.f_len, self.delta_f, self.f_low)
        self.psds = {ifo: psd for ifo in self.ifos}
        self.data = {
            ifo: FrequencySeries(
                np.random.randn(self.f_len) + 1j * np.random.randn(self.f_len),
                delta_f=self.delta_f,
                epoch=1126259460.0,
            )
            for ifo in self.ifos
        }
        self.static_params = {
            'mass1': 1.4,
            'mass2': 1.4,
            'approximant': 'TaylorF2',
            'f_lower': self.f_low,
            'distance': 100.0,
            'inclination': 0.0,
        }
        self.variable_params = ['ra', 'dec', 'polarization', 'tc']
        self.low_frequency_cutoff = {ifo: self.f_low for ifo in self.ifos}

        self.model = GaussianNoise(
            self.variable_params,
            self.data,
            psds=self.psds,
            low_frequency_cutoff=self.low_frequency_cutoff,
            static_params=self.static_params,
        )
        self.marg_phase_model = MarginalizedPhaseGaussianNoise(
            self.variable_params,
            self.data,
            psds=self.psds,
            low_frequency_cutoff=self.low_frequency_cutoff,
            static_params=self.static_params,
        )

    def test_network_geometry_integration(self):
        self.assertIsNotNone(self.model.network_geometry)
        self.assertEqual(len(self.model.network_geometry.detector_names), 3)
        self.assertEqual(self.model.network_geometry.detector_names, self.ifos)

        self.assertIsNotNone(self.marg_phase_model.network_geometry)
        self.assertEqual(len(self.marg_phase_model.network_geometry.detector_names), 3)

    def test_gaussian_noise_batched_loglr_numpy(self):
        n_samples = 10
        ra = np.linspace(0.2, 5.8, n_samples)
        dec = np.linspace(-1.2, 1.2, n_samples)
        pol = np.linspace(0.1, 3.0, n_samples)
        tc = np.linspace(1126259460.0, 1126259465.0, n_samples)

        # Batched evaluation
        batched_lr = self.model._batched_loglr(ra=ra, dec=dec, polarization=pol, tc=tc)
        batched_logl = self.model.batched_loglikelihood(ra=ra, dec=dec, polarization=pol, tc=tc)
        self.assertEqual(batched_lr.shape, (n_samples,))
        self.assertEqual(batched_logl.shape, (n_samples,))
        np.testing.assert_allclose(batched_logl, batched_lr + self.model.lognl)

        # Dictionary argument support
        dict_logl = self.model.batched_loglikelihood({'ra': ra, 'dec': dec, 'polarization': pol, 'tc': tc})
        np.testing.assert_allclose(dict_logl, batched_logl)

        # Compare against single point sequential evaluations
        ref_lr = []
        ref_logl = []
        for i in range(n_samples):
            self.model.update(ra=ra[i], dec=dec[i], polarization=pol[i], tc=tc[i])
            ref_lr.append(self.model.loglr)
            ref_logl.append(self.model.loglikelihood)

        np.testing.assert_allclose(batched_lr, ref_lr, rtol=1e-3)
        np.testing.assert_allclose(batched_logl, ref_logl, rtol=1e-3)

    def test_marginalized_phase_batched_loglr_numpy(self):
        n_samples = 10
        ra = np.linspace(0.2, 5.8, n_samples)
        dec = np.linspace(-1.2, 1.2, n_samples)
        pol = np.linspace(0.1, 3.0, n_samples)
        tc = np.linspace(1126259460.0, 1126259465.0, n_samples)

        batched_lr = self.marg_phase_model._batched_loglr(ra=ra, dec=dec, polarization=pol, tc=tc)
        batched_logl = self.marg_phase_model.batched_loglikelihood(ra=ra, dec=dec, polarization=pol, tc=tc)
        self.assertEqual(batched_lr.shape, (n_samples,))

        ref_lr = []
        ref_logl = []
        for i in range(n_samples):
            self.marg_phase_model.update(ra=ra[i], dec=dec[i], polarization=pol[i], tc=tc[i])
            ref_lr.append(self.marg_phase_model.loglr)
            ref_logl.append(self.marg_phase_model.loglikelihood)

        np.testing.assert_allclose(batched_lr, ref_lr, rtol=1e-3)
        np.testing.assert_allclose(batched_logl, ref_logl, rtol=1e-3)

    def test_batched_likelihood_torch_autograd(self):
        n_samples = 6
        ra_t = torch.linspace(0.5, 5.5, n_samples, dtype=torch.float64, requires_grad=True)
        dec_t = torch.linspace(-1.0, 1.0, n_samples, dtype=torch.float64, requires_grad=True)
        pol_t = torch.linspace(0.2, 2.8, n_samples, dtype=torch.float64, requires_grad=True)
        tc_t = torch.linspace(1126259460.0, 1126259464.0, n_samples, dtype=torch.float64, requires_grad=True)

        logl_t = self.model.batched_loglikelihood(ra=ra_t, dec=dec_t, polarization=pol_t, tc=tc_t)
        self.assertEqual(logl_t.shape, torch.Size([n_samples]))

        loss = logl_t.sum()
        loss.backward()

        self.assertIsNotNone(ra_t.grad)
        self.assertIsNotNone(dec_t.grad)
        self.assertIsNotNone(pol_t.grad)
        self.assertIsNotNone(tc_t.grad)
        self.assertTrue(torch.all(torch.isfinite(ra_t.grad)))
        self.assertTrue(torch.all(torch.isfinite(tc_t.grad)))


if __name__ == '__main__':
    unittest.main()
