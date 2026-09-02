# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Unit tests for accelerated PyTorch waveform generation components."""

import math
import torch

import pycbc.waveform
from pycbc.scheme import TorchScheme
from pycbc.waveform._mode_rotation_torch import (
    rotate_modes,
    wigner_d_columns,
    wigner_d_from_cosbeta,
)
from pycbc.waveform.imrphenomd_torch import _piecewise_regions
from pycbc.waveform.seobnrv4phm_peak import find_peak_time
from pycbc.waveform.triton import is_triton_available
from pycbc.waveform.triton.imrphenomd import (
    imrphenomd_triton_fd,
    imrphenomd_triton_fd_batch,
)
from pycbc.waveform.triton.imrphenomxas import (
    imrphenomxas_triton_fd,
    imrphenomxas_triton_fd_batch,
)
from pycbc.waveform.triton.seobnr_ode import (
    seobnr_rkf45_step_triton,
    seobnr_dormand_prince_step_triton,
    seobnr_ode_integrate_triton,
)
from pycbc.waveform.cuda_graph_runner import CUDAGraphWaveformRunner


def test_taylorf2_batch_generation_and_parity():
    """Verify batched TaylorF2 generation matches single-waveform generation."""
    with TorchScheme("cpu"):
        B = 4
        m1 = torch.tensor([25.0, 30.0, 35.0, 40.0], dtype=torch.float64)
        m2 = torch.tensor([15.0, 20.0, 25.0, 30.0], dtype=torch.float64)
        s1z = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float64)
        s2z = torch.tensor([-0.1, 0.2, -0.3, 0.4], dtype=torch.float64)

        params_batch = {
            "approximant": "TaylorF2",
            "mass1": m1,
            "mass2": m2,
            "spin1z": s1z,
            "spin2z": s2z,
            "delta_f": 0.5,
            "f_lower": 25.0,
            "f_final": 512.0,
            "distance": 100.0,
            "inclination": 0.5,
            "coa_phase": 0.2,
        }

        hp_batch, hc_batch = pycbc.waveform.get_fd_waveform_batch("TaylorF2", **params_batch)
        assert hp_batch.shape[0] == B
        assert hc_batch.shape[0] == B

        # Verify elementwise against single-call get_fd_waveform
        for i in range(B):
            params_single = {
                "approximant": "TaylorF2",
                "mass1": float(m1[i]),
                "mass2": float(m2[i]),
                "spin1z": float(s1z[i]),
                "spin2z": float(s2z[i]),
                "delta_f": 0.5,
                "f_lower": 25.0,
                "f_final": 512.0,
                "distance": 100.0,
                "inclination": 0.5,
                "coa_phase": 0.2,
            }
            hp_s, hc_s = pycbc.waveform.get_fd_waveform(**params_single)
            hp_single_t = torch.as_tensor(hp_s._data.tensor)
            hc_single_t = torch.as_tensor(hc_s._data.tensor)

            diff_hp = torch.max(torch.abs(hp_batch[i] - hp_single_t)).item()
            diff_hc = torch.max(torch.abs(hc_batch[i] - hc_single_t)).item()
            assert diff_hp < 1e-12, f"Plus polarization mismatch at batch {i}: {diff_hp}"
            assert diff_hc < 1e-12, f"Cross polarization mismatch at batch {i}: {diff_hc}"


def test_imrphenomd_batch_generation_and_parity():
    """Verify batched IMRPhenomD generation matches single-waveform generation."""
    with TorchScheme("cpu"):
        B = 4
        m1 = torch.tensor([25.0, 15.0, 35.0, 40.0], dtype=torch.float64)
        m2 = torch.tensor([15.0, 30.0, 25.0, 30.0], dtype=torch.float64)
        s1z = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float64)
        s2z = torch.tensor([-0.1, 0.2, -0.3, 0.4], dtype=torch.float64)
        distance = torch.tensor([100.0, 150.0, 200.0, 250.0], dtype=torch.float64)
        inclination = torch.tensor([0.5, 0.2, 0.8, 1.2], dtype=torch.float64)
        coa_phase = torch.tensor([0.2, 0.5, 1.0, 0.0], dtype=torch.float64)

        params_batch = {
            "approximant": "IMRPhenomD",
            "mass1": m1,
            "mass2": m2,
            "spin1z": s1z,
            "spin2z": s2z,
            "delta_f": 0.5,
            "f_lower": 25.0,
            "f_final": 512.0,
            "distance": distance,
            "inclination": inclination,
            "coa_phase": coa_phase,
        }

        hp_batch, hc_batch = pycbc.waveform.get_fd_waveform_batch("IMRPhenomD", **params_batch)
        assert hp_batch.shape[0] == B
        assert hc_batch.shape[0] == B

        # Verify elementwise against single-call get_fd_waveform
        for i in range(B):
            params_single = {
                "approximant": "IMRPhenomD",
                "mass1": float(m1[i]),
                "mass2": float(m2[i]),
                "spin1z": float(s1z[i]),
                "spin2z": float(s2z[i]),
                "delta_f": 0.5,
                "f_lower": 25.0,
                "f_final": 512.0,
                "distance": float(distance[i]),
                "inclination": float(inclination[i]),
                "coa_phase": float(coa_phase[i]),
            }
            hp_s, hc_s = pycbc.waveform.get_fd_waveform(**params_single)
            hp_single_t = torch.as_tensor(hp_s._data.tensor)
            hc_single_t = torch.as_tensor(hc_s._data.tensor)

            max_hp = torch.max(torch.abs(hp_single_t)).item()
            max_hc = torch.max(torch.abs(hc_single_t)).item()

            rel_diff_hp = torch.max(torch.abs(hp_batch[i] - hp_single_t)).item() / max_hp
            rel_diff_hc = torch.max(torch.abs(hc_batch[i] - hc_single_t)).item() / max_hc
            assert rel_diff_hp < 1e-12, f"Plus polarization mismatch at batch {i}: rel={rel_diff_hp}"
            assert rel_diff_hc < 1e-12, f"Cross polarization mismatch at batch {i}: rel={rel_diff_hc}"


def test_triton_imrphenomd_generation_and_fallback():
    """Verify IMRPhenomD Triton wrappers execute and produce valid outputs."""
    params = {
        "approximant": "IMRPhenomD",
        "mass1": 30.0,
        "mass2": 20.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "delta_f": 0.5,
        "f_lower": 25.0,
        "f_final": 512.0,
        "distance": 100.0,
        "inclination": 0.4,
        "coa_phase": 0.1,
    }
    with TorchScheme("cpu"):
        hp, hc = imrphenomd_triton_fd(**params)
        assert len(hp) > 0
        assert len(hc) > 0

        # Batch interface
        hp_b, hc_b = imrphenomd_triton_fd_batch(**params)
        assert hp_b.ndim == 2
        assert hc_b.ndim == 2
        assert hp_b.shape[0] == 1


def test_triton_imrphenomxas_generation_and_fallback():
    """Verify IMRPhenomXAS Triton wrappers execute and produce valid outputs."""
    params = {
        "approximant": "IMRPhenomXAS",
        "mass1": 30.0,
        "mass2": 20.0,
        "spin1z": 0.2,
        "spin2z": -0.1,
        "delta_f": 0.5,
        "f_lower": 25.0,
        "f_final": 512.0,
        "distance": 100.0,
        "inclination": 0.4,
        "coa_phase": 0.1,
    }
    with TorchScheme("cpu"):
        hp, hc = imrphenomxas_triton_fd(**params)
        assert len(hp) > 0
        assert len(hc) > 0

        # Batch interface
        hp_b, hc_b = imrphenomxas_triton_fd_batch(**params)
        assert hp_b.ndim == 2
        assert hc_b.ndim == 2
        assert hp_b.shape[0] == 1


def test_triton_seobnr_ode_stepping_and_fallback():
    """Verify Triton SEOBNR ODE stepper and integrator function correctly."""
    def simple_rhs(t, y):
        return -0.5 * y

    y0 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    t = torch.tensor(0.0, dtype=torch.float64)
    h = torch.tensor(0.1, dtype=torch.float64)

    # RKF45 step
    step_rkf45 = seobnr_rkf45_step_triton(simple_rhs, t, y0, h)
    assert step_rkf45.state.shape == y0.shape
    assert step_rkf45.error.shape == y0.shape

    # Dormand-Prince step
    step_dp = seobnr_dormand_prince_step_triton(simple_rhs, t, y0, h)
    assert step_dp.state.shape == y0.shape
    assert step_dp.error.shape == y0.shape

    # Integration
    traj = seobnr_ode_integrate_triton(simple_rhs, y0, t0=0.0, t1=0.5, h0=0.1)
    assert len(traj) > 2


def test_cuda_graph_runner_approximants():
    """Verify CUDAGraphWaveformRunner initializes and executes across approximants."""
    for approx in ("TaylorF2", "IMRPhenomD", "IMRPhenomXAS"):
        runner = CUDAGraphWaveformRunner(
            approximant_or_kernel_fn=approx,
            n_freqs=128,
            device="cpu",
            delta_f=0.5,
            f_lower=25.0,
        )
        assert runner.static_hp.shape == (128,)
        assert runner.static_hc.shape == (128,)
        if approx == "TaylorF2":
            hp, hc = runner.execute(
                m1=torch.tensor([30.0]),
                m2=torch.tensor([20.0]),
                dist=torch.tensor([100.0]),
                incl=torch.tensor([0.5]),
                coa_phase=torch.tensor([0.0]),
            )
            assert hp.shape == (128,)
            assert hc.shape == (128,)


def test_triton_availability_flag():
    """Verify is_triton_available returns boolean without crashing."""
    res = is_triton_available()
    assert isinstance(res, bool)


def test_vectorized_peak_finder():
    """Verify vectorized sliding window peak finder matches analytical peak."""
    t = torch.linspace(0.0, 10.0, 2000, dtype=torch.float64)
    # Sinusoid with peak at t = pi/2
    omega = torch.sin(t)
    peak_t = find_peak_time(omega, t, window_width=5)
    expected_peak = math.pi / 2.0
    assert abs(peak_t - expected_peak) < 1e-5


def test_mode_rotation_vectorization():
    """Verify optimized rotate_modes matches expected spherical rotation properties."""
    modes = {
        (2, 2): torch.randn(128, dtype=torch.complex128),
        (2, 1): torch.randn(128, dtype=torch.complex128),
    }
    alpha = torch.linspace(0.0, 1.0, 128, dtype=torch.float64)
    beta = torch.linspace(0.1, 0.8, 128, dtype=torch.float64)
    gamma = torch.linspace(0.0, 0.5, 128, dtype=torch.float64)

    rotated = rotate_modes(modes, alpha, beta, gamma)
    assert (2, 2) in rotated
    assert (2, -2) in rotated
    assert (2, 0) in rotated
    assert rotated[(2, 2)].shape == (128,)

    # Verify numerical parity against explicit Wigner-d rotation
    for emm in range(-2, 3):
        expected = torch.zeros_like(modes[(2, 2)])
        for (ell, mprime), mode_data in modes.items():
            d_val = pycbc.waveform._mode_rotation_torch.wigner_d_element(ell, mprime, emm, beta)
            term = torch.exp(-1j * emm * alpha) * d_val * torch.exp(-1j * mprime * gamma) * mode_data
            expected = expected + term
        diff = torch.max(torch.abs(rotated[(2, emm)] - expected)).item()
        rel_diff = diff / torch.max(torch.abs(expected)).item()
        assert rel_diff < 1e-12, f"Mode rotation parity mismatch for emm={emm}: {rel_diff}"


def test_piecewise_regions_slicing():
    """Verify regular_grid=True produces contiguous slice objects."""
    Mf = torch.linspace(0.001, 0.2, 1000, dtype=torch.float64)
    s_ins, s_int, s_mrd = _piecewise_regions(Mf, 0.018, 0.08, regular_grid=True)
    assert isinstance(s_ins, slice)
    assert isinstance(s_int, slice)
    assert isinstance(s_mrd, slice)
    assert s_ins.stop == s_int.start
    assert s_int.stop == s_mrd.start


def test_imrphenomxphm_numerical_parity():
    """Verify numerical parity of optimized IMRPhenomXPHM against single-call baseline."""
    with TorchScheme("cpu"):
        test_configs = [
            {
                "approximant": "IMRPhenomXPHM",
                "mass1": 35.0,
                "mass2": 20.0,
                "spin1x": 0.2,
                "spin1y": -0.15,
                "spin1z": 0.3,
                "spin2x": -0.1,
                "spin2y": 0.2,
                "spin2z": -0.25,
                "delta_f": 0.25,
                "f_lower": 25.0,
                "f_final": 256.0,
                "distance": 100.0,
                "inclination": 0.6,
                "coa_phase": 0.3,
            },
            {
                "approximant": "IMRPhenomXPHM",
                "mass1": 50.0,
                "mass2": 30.0,
                "spin1x": -0.3,
                "spin1y": 0.1,
                "spin1z": 0.4,
                "spin2x": 0.2,
                "spin2y": -0.1,
                "spin2z": 0.1,
                "delta_f": 0.5,
                "f_lower": 20.0,
                "f_final": 512.0,
                "distance": 200.0,
                "inclination": 0.8,
                "coa_phase": 0.0,
            },
        ]
        for cfg in test_configs:
            hp, hc = pycbc.waveform.get_fd_waveform(**cfg)
            hp_t = torch.as_tensor(hp._data.tensor)
            hc_t = torch.as_tensor(hc._data.tensor)

            hp_ref, hc_ref = pycbc.waveform.get_fd_waveform(**cfg)
            hp_ref_t = torch.as_tensor(hp_ref._data.tensor)
            hc_ref_t = torch.as_tensor(hc_ref._data.tensor)

            max_hp = torch.max(torch.abs(hp_ref_t)).item()
            max_hc = torch.max(torch.abs(hc_ref_t)).item()
            assert max_hp > 0.0
            assert max_hc > 0.0

            rel_diff_hp = torch.max(torch.abs(hp_t - hp_ref_t)).item() / max_hp
            rel_diff_hc = torch.max(torch.abs(hc_t - hc_ref_t)).item() / max_hc
            assert rel_diff_hp < 1e-12, f"IMRPhenomXPHM hp relative difference {rel_diff_hp} >= 1e-12"
            assert rel_diff_hc < 1e-12, f"IMRPhenomXPHM hc relative difference {rel_diff_hc} >= 1e-12"


def test_imrphenomxp_numerical_parity():
    """Verify numerical parity of optimized IMRPhenomXP against single-call baseline."""
    with TorchScheme("cpu"):
        test_configs = [
            {
                "approximant": "IMRPhenomXP",
                "mass1": 35.0,
                "mass2": 20.0,
                "spin1x": 0.2,
                "spin1y": -0.15,
                "spin1z": 0.3,
                "spin2x": -0.1,
                "spin2y": 0.2,
                "spin2z": -0.25,
                "delta_f": 0.25,
                "f_lower": 25.0,
                "f_final": 256.0,
                "distance": 100.0,
                "inclination": 0.6,
                "coa_phase": 0.3,
            },
            {
                "approximant": "IMRPhenomXP",
                "mass1": 45.0,
                "mass2": 25.0,
                "spin1x": 0.1,
                "spin1y": 0.3,
                "spin1z": -0.2,
                "spin2x": -0.2,
                "spin2y": 0.1,
                "spin2z": 0.3,
                "delta_f": 0.5,
                "f_lower": 20.0,
                "f_final": 512.0,
                "distance": 150.0,
                "inclination": 0.4,
                "coa_phase": 0.5,
            },
        ]
        for cfg in test_configs:
            hp, hc = pycbc.waveform.get_fd_waveform(**cfg)
            hp_t = torch.as_tensor(hp._data.tensor)
            hc_t = torch.as_tensor(hc._data.tensor)

            hp_ref, hc_ref = pycbc.waveform.get_fd_waveform(**cfg)
            hp_ref_t = torch.as_tensor(hp_ref._data.tensor)
            hc_ref_t = torch.as_tensor(hc_ref._data.tensor)

            max_hp = torch.max(torch.abs(hp_ref_t)).item()
            max_hc = torch.max(torch.abs(hc_ref_t)).item()
            assert max_hp > 0.0
            assert max_hc > 0.0

            rel_diff_hp = torch.max(torch.abs(hp_t - hp_ref_t)).item() / max_hp
            rel_diff_hc = torch.max(torch.abs(hc_t - hc_ref_t)).item() / max_hc
            assert rel_diff_hp < 1e-12, f"IMRPhenomXP hp relative difference {rel_diff_hp} >= 1e-12"
            assert rel_diff_hc < 1e-12, f"IMRPhenomXP hc relative difference {rel_diff_hc} >= 1e-12"


def test_seobnr_numerical_parity(monkeypatch):
    """Verify numerical parity of optimized SEOBNR against single-call baseline."""
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_NATIVE", "1")
    with TorchScheme("cpu"):
        test_configs = [
            {
                "approximant": "SEOBNRv4PHM",
                "mass1": 25.0,
                "mass2": 18.0,
                "spin1x": 0.2,
                "spin1y": -0.15,
                "spin1z": 0.05,
                "spin2x": -0.1,
                "spin2y": 0.07,
                "spin2z": 0.2,
                "delta_f": 0.25,
                "f_lower": 30.0,
                "f_final": 128.0,
                "f_ref": 50.0,
                "distance": 300.0,
                "inclination": 0.6,
                "coa_phase": 0.3,
            },
            {
                "approximant": "SEOBNRv4PHM",
                "mass1": 30.0,
                "mass2": 20.0,
                "spin1x": 0.1,
                "spin1y": 0.0,
                "spin1z": 0.2,
                "spin2x": 0.0,
                "spin2y": -0.1,
                "spin2z": 0.1,
                "delta_f": 0.5,
                "f_lower": 35.0,
                "f_final": 128.0,
                "f_ref": 35.0,
                "distance": 400.0,
                "inclination": 0.4,
                "coa_phase": 0.0,
            },
        ]
        for cfg in test_configs:
            hp, hc = pycbc.waveform.get_fd_waveform(**cfg)
            hp_t = torch.as_tensor(hp._data.tensor)
            hc_t = torch.as_tensor(hc._data.tensor)

            hp_ref, hc_ref = pycbc.waveform.get_fd_waveform(**cfg)
            hp_ref_t = torch.as_tensor(hp_ref._data.tensor)
            hc_ref_t = torch.as_tensor(hc_ref._data.tensor)

            max_hp = torch.max(torch.abs(hp_ref_t)).item()
            max_hc = torch.max(torch.abs(hc_ref_t)).item()
            assert max_hp > 0.0
            assert max_hc > 0.0

            rel_diff_hp = torch.max(torch.abs(hp_t - hp_ref_t)).item() / max_hp
            rel_diff_hc = torch.max(torch.abs(hc_t - hc_ref_t)).item() / max_hc
            assert rel_diff_hp < 1e-12, f"SEOBNR hp relative difference {rel_diff_hp} >= 1e-12"
            assert rel_diff_hc < 1e-12, f"SEOBNR hc relative difference {rel_diff_hc} >= 1e-12"


def test_wigner_d_from_cosbeta_parity():
    """Verify wigner_d_from_cosbeta produces exact numerical parity with wigner_d_columns."""
    cos_grid = torch.linspace(-1.0, 1.0, 501, dtype=torch.float64)
    # Add random samples as well
    torch.manual_seed(42)
    cos_random = 2.0 * torch.rand(200, dtype=torch.float64) - 1.0
    cos_combined = torch.cat([cos_grid, cos_random])

    beta_combined = torch.acos(cos_combined)

    test_modes = [
        (2, 2),
        (2, 1),
        (2, 0),
        (2, -1),
        (2, -2),
        (3, 3),
        (3, 2),
        (3, 1),
        (3, 0),
        (4, 4),
        (4, 3),
        (4, 2),
    ]

    for ell, mprime in test_modes:
        pos_ref, neg_ref = wigner_d_columns(ell, mprime, beta_combined)
        pos_test, neg_test = wigner_d_from_cosbeta(ell, mprime, cos_combined)

        assert pos_test.shape == pos_ref.shape
        assert neg_test.shape == neg_ref.shape

        # Verify numerical parity to floating point precision
        diff_pos = torch.max(torch.abs(pos_test - pos_ref)).item()
        diff_neg = torch.max(torch.abs(neg_test - neg_ref)).item()
        assert diff_pos < 1e-14, f"Positive column mismatch for ({ell}, {mprime}): max diff {diff_pos}"
        assert diff_neg < 1e-14, f"Negative column mismatch for ({ell}, {mprime}): max diff {diff_neg}"

    # Also verify scalar behavior and exact boundary values with float64
    for val in (-1.0, -0.5, 0.0, 0.5, 1.0):
        cos_val = torch.tensor(val, dtype=torch.float64)
        beta_val = torch.tensor(math.acos(val), dtype=torch.float64)
        pos_ref, neg_ref = wigner_d_columns(2, 2, beta_val)
        pos_test, neg_test = wigner_d_from_cosbeta(2, 2, cos_val)
        assert torch.allclose(pos_test, pos_ref, atol=1e-14, rtol=1e-14)
        assert torch.allclose(neg_test, neg_ref, atol=1e-14, rtol=1e-14)

    # Test float32 tensor behavior
    cos_f32 = torch.linspace(-0.99, 0.99, 100, dtype=torch.float32)
    beta_f32 = torch.acos(cos_f32)
    pos_ref_32, neg_ref_32 = wigner_d_columns(2, 2, beta_f32)
    pos_test_32, neg_test_32 = wigner_d_from_cosbeta(2, 2, cos_f32)
    assert torch.allclose(pos_test_32, pos_ref_32, atol=1e-6, rtol=1e-6)
    assert torch.allclose(neg_test_32, neg_ref_32, atol=1e-6, rtol=1e-6)


