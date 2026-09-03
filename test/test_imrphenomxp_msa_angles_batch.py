import math
import pytest

torch = pytest.importorskip("torch")

import pycbc.waveform
from pycbc.scheme import TorchScheme
from pycbc.waveform.imrphenomxp_msa_torch import (
    build_msa_state,
    msa_angles,
    msa_angles_batch,
)
from pycbc.waveform import imrphenomx_utils_torch as xutils


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_msa_angles_batch_matches_scalar_loop(dtype):
    """Verify msa_angles_batch matches elementwise scalar calls."""
    cases = [
        (30.0, 20.0, (0.2, -0.1, 0.3), (-0.1, 0.2, -0.2), 20.0),
        (50.0, 10.0, (0.0, 0.0, 0.5), (0.0, 0.0, -0.3), 25.0),
        (25.0, 25.0, (0.1, 0.3, -0.1), (-0.2, 0.1, 0.4), 15.0),
        (40.0, 15.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 20.0),
    ]

    states = []
    m_sec_list = []
    for m1, m2, s1, s2, f_ref in cases:
        m_sec = (m1 + m2) * xutils.MTSUN
        m_sec_list.append(m_sec)
        states.append(build_msa_state(m1, m2, s1, s2, m_sec, f_ref))

    B = len(states)
    N_f = 64
    f = torch.linspace(20.0, 500.0, N_f, dtype=dtype)
    m_sec_tensor = torch.tensor(m_sec_list, dtype=dtype).view(1, B, 1)
    mprime_factors = (
        2.0 / torch.arange(1, 5, dtype=dtype)
    ).view(4, 1, 1)
    v_3d = torch.pow(
        math.pi * m_sec_tensor * f.view(1, 1, -1) * mprime_factors,
        1.0 / 3.0,
    )

    # Batched call
    phiz_b, zeta_b, cos_beta_b = msa_angles_batch(v_3d, states)
    assert phiz_b.shape == (4, B, N_f)
    assert zeta_b.shape == (4, B, N_f)
    assert cos_beta_b.shape == (4, B, N_f)

    # Scalar loop
    atol = 1e-10 if dtype == torch.float64 else 0.1
    rtol = 1e-10 if dtype == torch.float64 else 1e-4
    for m_idx in range(4):
        for b_idx in range(B):
            v_single = v_3d[m_idx, b_idx]
            pz, zt, cb = msa_angles(v_single, states[b_idx])
            assert torch.allclose(phiz_b[m_idx, b_idx], pz, atol=atol, rtol=rtol)
            assert torch.allclose(zeta_b[m_idx, b_idx], zt, atol=atol, rtol=rtol)
            assert torch.allclose(
                cos_beta_b[m_idx, b_idx], cb, atol=atol, rtol=rtol
            )


def test_msa_angles_batch_single_batch():
    """Verify msa_angles_batch works with B=1."""
    m1, m2 = 35.0, 20.0
    m_sec = (m1 + m2) * xutils.MTSUN
    state = build_msa_state(
        m1, m2, (0.2, 0.1, -0.3), (-0.1, 0.2, 0.1), m_sec, 20.0
    )
    N_f = 32
    f = torch.linspace(20.0, 400.0, N_f, dtype=torch.float64)
    mprime_factors = (
        2.0 / torch.arange(1, 5, dtype=torch.float64)
    ).view(4, 1, 1)
    v_3d = torch.pow(
        math.pi * m_sec * f.view(1, 1, -1) * mprime_factors,
        1.0 / 3.0,
    )

    phiz_b, zeta_b, cos_beta_b = msa_angles_batch(v_3d, [state])
    assert phiz_b.shape == (4, 1, N_f)

    for m_idx in range(4):
        pz, zt, cb = msa_angles(v_3d[m_idx, 0], state)
        assert torch.allclose(phiz_b[m_idx, 0], pz, atol=1e-13, rtol=1e-13)
        assert torch.allclose(zeta_b[m_idx, 0], zt, atol=1e-13, rtol=1e-13)
        assert torch.allclose(cos_beta_b[m_idx, 0], cb, atol=1e-13, rtol=1e-13)


def test_imrphenomxphm_fd_batch_parity():
    """Verify IMRPhenomXPHM batch generation produces identical waveforms."""
    with TorchScheme("cpu"):
        B = 3
        m1 = torch.tensor([30.0, 20.0, 45.0], dtype=torch.float64)
        m2 = torch.tensor([15.0, 20.0, 15.0], dtype=torch.float64)
        s1x = torch.tensor([0.2, 0.0, -0.3], dtype=torch.float64)
        s1y = torch.tensor([0.1, 0.0, 0.2], dtype=torch.float64)
        s1z = torch.tensor([0.3, 0.5, -0.1], dtype=torch.float64)
        s2x = torch.tensor([-0.1, 0.0, 0.2], dtype=torch.float64)
        s2y = torch.tensor([0.2, 0.0, -0.1], dtype=torch.float64)
        s2z = torch.tensor([-0.2, -0.4, 0.3], dtype=torch.float64)
        distance = torch.tensor([100.0, 200.0, 150.0], dtype=torch.float64)
        inclination = torch.tensor([0.6, 0.3, 1.1], dtype=torch.float64)
        coa_phase = torch.tensor([0.4, 0.0, 1.2], dtype=torch.float64)
        long_asc_nodes = torch.tensor([0.2, 0.5, 0.0], dtype=torch.float64)

        params_batch = {
            "approximant": "IMRPhenomXPHM",
            "mass1": m1,
            "mass2": m2,
            "spin1x": s1x,
            "spin1y": s1y,
            "spin1z": s1z,
            "spin2x": s2x,
            "spin2y": s2y,
            "spin2z": s2z,
            "delta_f": 0.5,
            "f_lower": 25.0,
            "f_final": 512.0,
            "distance": distance,
            "inclination": inclination,
            "coa_phase": coa_phase,
            "long_asc_nodes": long_asc_nodes,
        }

        hp_batch, hc_batch = pycbc.waveform.get_fd_waveform_batch(
            "IMRPhenomXPHM", **params_batch
        )
        assert hp_batch.shape[0] == B
        assert hc_batch.shape[0] == B

        for i in range(B):
            params_single = {
                "approximant": "IMRPhenomXPHM",
                "mass1": float(m1[i]),
                "mass2": float(m2[i]),
                "spin1x": float(s1x[i]),
                "spin1y": float(s1y[i]),
                "spin1z": float(s1z[i]),
                "spin2x": float(s2x[i]),
                "spin2y": float(s2y[i]),
                "spin2z": float(s2z[i]),
                "delta_f": 0.5,
                "f_lower": 25.0,
                "f_final": 512.0,
                "distance": float(distance[i]),
                "inclination": float(inclination[i]),
                "coa_phase": float(coa_phase[i]),
                "long_asc_nodes": float(long_asc_nodes[i]),
            }
            hp_s, hc_s = pycbc.waveform.get_fd_waveform(**params_single)
            hp_single_t = torch.as_tensor(hp_s._data.tensor)
            hc_single_t = torch.as_tensor(hc_s._data.tensor)

            max_hp = torch.max(torch.abs(hp_single_t)).item()
            max_hc = torch.max(torch.abs(hc_single_t)).item()

            rel_diff_hp = (
                torch.max(torch.abs(hp_batch[i] - hp_single_t)).item() / max_hp
            )
            rel_diff_hc = (
                torch.max(torch.abs(hc_batch[i] - hc_single_t)).item() / max_hc
            )
            assert rel_diff_hp < 1e-12, f"Mismatch at b={i}: rel={rel_diff_hp}"
            assert rel_diff_hc < 1e-12, f"Mismatch at b={i}: rel={rel_diff_hc}"


def test_msa_angles_batch_dict_input():
    """Verify msa_angles_batch works with pre-batched dictionary."""
    m1, m2 = 30.0, 20.0
    m_sec = (m1 + m2) * xutils.MTSUN
    state = build_msa_state(
        m1, m2, (0.2, -0.1, 0.3), (-0.1, 0.2, -0.2), m_sec, 20.0
    )
    N_f = 16
    f = torch.linspace(20.0, 300.0, N_f, dtype=torch.float64)
    v_3d = torch.pow(
        math.pi * m_sec * f.view(1, 1, -1) * (2.0 / 2.0), 1.0 / 3.0
    ).repeat(4, 2, 1)

    phiz_b, zeta_b, cos_beta_b = msa_angles_batch(v_3d, state)
    assert phiz_b.shape == (4, 2, N_f)
    assert zeta_b.shape == (4, 2, N_f)
    assert cos_beta_b.shape == (4, 2, N_f)
