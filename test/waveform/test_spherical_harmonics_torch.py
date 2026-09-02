import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc.waveform._spherical_harmonics_torch import (  # noqa: E402
    cudagraphed_spin_minus_two_spherical_harmonics_phi_zero,
    scripted_spin_minus_two_spherical_harmonics_phi_zero,
    spin_minus_two_spherical_harmonics_phi_zero,
    spin_weighted_spherical_harmonic,
)


@pytest.mark.parametrize("theta", [0.0, 0.2, 1.1, 2.7, math.pi])
@pytest.mark.parametrize("phi", [-0.3, 0.0, 0.7, math.pi / 2.0])
def test_torch_harmonics_match_lal(theta, phi):
    for ell in range(2, 6):
        for emm in range(-ell, ell + 1):
            expected = lal.SpinWeightedSphericalHarmonic(
                theta, phi, -2, ell, emm
            )
            actual = spin_weighted_spherical_harmonic(
                theta,
                phi,
                -2,
                ell,
                emm,
                dtype=torch.float64,
                device="cpu",
            )
            assert actual.item() == pytest.approx(expected, abs=2.0e-13)


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_torch_harmonics_stay_on_requested_device(device_name):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    dtype = torch.float32 if device_name == "mps" else torch.float64
    theta = torch.tensor(
        [0.2, 0.8, 1.4], dtype=dtype, device=device_name
    )
    actual = spin_weighted_spherical_harmonic(
        theta,
        0.7,
        -2,
        5,
        -3,
        dtype=dtype,
        device=device_name,
    )
    expected = np.array(
        [
            lal.SpinWeightedSphericalHarmonic(
                float(angle), 0.7, -2, 5, -3
            )
            for angle in theta.cpu()
        ]
    )

    assert actual.device.type == device_name
    expected_dtype = (
        torch.complex64 if dtype == torch.float32 else torch.complex128
    )
    assert actual.dtype == expected_dtype
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(),
        expected,
        rtol=2.0e-5 if dtype == torch.float32 else 1.0e-12,
        atol=2.0e-7 if dtype == torch.float32 else 1.0e-14,
    )


def test_torch_harmonics_are_differentiable():
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    harmonic = spin_weighted_spherical_harmonic(
        theta,
        0.2,
        -2,
        4,
        3,
        dtype=torch.float64,
        device="cpu",
    )
    harmonic.real.backward()
    assert torch.isfinite(theta.grad)


@pytest.mark.parametrize(
    ("dtype", "word_dtype"),
    ((torch.float32, torch.int32), (torch.float64, torch.int64)),
)
@pytest.mark.parametrize(
    "theta",
    (-0.0, 0.0, 1.0e-12, 0.2, math.pi / 2.0, math.pi - 1.0e-12, math.pi),
)
def test_bulk_spin_minus_two_phi_zero_harmonics_are_bitwise_exact(
    theta,
    dtype,
    word_dtype,
):
    bulk = spin_minus_two_spherical_harmonics_phi_zero(
        theta,
        dtype=dtype,
        device="cpu",
    )

    assert tuple(bulk) == tuple(
        (ell, emm) for ell in range(2, 5) for emm in range(-ell, ell + 1)
    )
    for (ell, emm), value in bulk.items():
        scalar = spin_weighted_spherical_harmonic(
            theta,
            0.0,
            -2,
            ell,
            emm,
            dtype=dtype,
            device="cpu",
        )
        assert value.shape == scalar.shape == torch.Size([])
        assert value.dtype == scalar.dtype
        assert value.device == scalar.device
        assert torch.equal(
            torch.view_as_real(value).view(word_dtype),
            torch.view_as_real(scalar).view(word_dtype),
        )


def test_scripted_bulk_harmonics_are_bitwise_and_cached_by_dtype(
    monkeypatch,
):
    import pycbc.waveform._spherical_harmonics_torch as harmonics_torch

    monkeypatch.setattr(
        harmonics_torch,
        "_SCRIPTED_SPIN_MINUS_TWO_PHI_ZERO",
        {},
    )
    original_trace = torch.jit.trace
    trace_calls = 0

    def counted_trace(*args, **kwargs):
        nonlocal trace_calls
        trace_calls += 1
        return original_trace(*args, **kwargs)

    monkeypatch.setattr(torch.jit, "trace", counted_trace)
    for dtype, word_dtype in (
        (torch.float32, torch.int32),
        (torch.float64, torch.int64),
    ):
        for theta in (
            -0.0,
            0.0,
            1.0e-12,
            0.2,
            math.pi / 2.0,
            math.pi - 1.0e-12,
            math.pi,
        ):
            eager = spin_minus_two_spherical_harmonics_phi_zero(
                theta,
                dtype=dtype,
                device="cpu",
            )
            scripted = scripted_spin_minus_two_spherical_harmonics_phi_zero(
                theta,
                dtype=dtype,
                device="cpu",
            )
            assert tuple(scripted) == tuple(eager)
            for mode in eager:
                assert torch.equal(
                    torch.view_as_real(scripted[mode]).view(word_dtype),
                    torch.view_as_real(eager[mode]).view(word_dtype),
                )

    assert trace_calls == 2
    with pytest.raises(ValueError, match="scalar theta"):
        scripted_spin_minus_two_spherical_harmonics_phi_zero(
            torch.tensor([0.2], dtype=torch.float64),
            dtype=torch.float64,
            device="cpu",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    ("dtype", "word_dtype"),
    ((torch.float32, torch.int32), (torch.float64, torch.int64)),
)
def test_scripted_bulk_harmonics_are_bitwise_on_cuda(dtype, word_dtype):
    for theta in (-0.0, 0.0, 0.2, math.pi / 2.0, math.pi):
        eager = spin_minus_two_spherical_harmonics_phi_zero(
            theta,
            dtype=dtype,
            device="cuda",
        )
        scripted = scripted_spin_minus_two_spherical_harmonics_phi_zero(
            theta,
            dtype=dtype,
            device="cuda",
        )
        for mode in eager:
            assert torch.equal(
                torch.view_as_real(scripted[mode]).view(word_dtype),
                torch.view_as_real(eager[mode]).view(word_dtype),
            )


def test_cudagraphed_bulk_harmonics_fall_back_bitwise_on_cpu():
    eager = spin_minus_two_spherical_harmonics_phi_zero(
        0.731,
        dtype=torch.float64,
        device="cpu",
    )
    candidate = cudagraphed_spin_minus_two_spherical_harmonics_phi_zero(
        0.731,
        dtype=torch.float64,
        device="cpu",
    )
    for mode in eager:
        assert torch.equal(
            torch.view_as_real(candidate[mode]).view(torch.int64),
            torch.view_as_real(eager[mode]).view(torch.int64),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    ("dtype", "word_dtype"),
    ((torch.float32, torch.int32), (torch.float64, torch.int64)),
)
def test_cudagraphed_bulk_harmonics_are_bitwise_owned_and_cached(
    dtype,
    word_dtype,
    monkeypatch,
):
    import pycbc.waveform._spherical_harmonics_torch as harmonics_torch

    monkeypatch.setattr(
        harmonics_torch,
        "_CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO",
        {},
    )
    monkeypatch.setattr(
        harmonics_torch,
        "_CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO_FAILURES",
        set(),
    )
    retained = None
    retained_bits = None
    for theta in (-0.0, 0.0, 0.137, 0.731, math.pi / 2.0, math.pi):
        eager = spin_minus_two_spherical_harmonics_phi_zero(
            theta,
            dtype=dtype,
            device="cuda",
        )
        candidate = cudagraphed_spin_minus_two_spherical_harmonics_phi_zero(
            theta,
            dtype=dtype,
            device="cuda",
        )
        for mode in eager:
            assert torch.equal(
                torch.view_as_real(candidate[mode]).view(word_dtype),
                torch.view_as_real(eager[mode]).view(word_dtype),
            )
        if retained is None:
            retained = candidate
            retained_bits = {
                mode: torch.view_as_real(value).view(word_dtype).clone()
                for mode, value in candidate.items()
            }

    assert len(harmonics_torch._CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO) == 1
    assert not harmonics_torch._CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO_FAILURES
    for mode, expected in retained_bits.items():
        assert torch.equal(
            torch.view_as_real(retained[mode]).view(word_dtype),
            expected,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cudagraphed_bulk_harmonics_preserve_forward_autograd(monkeypatch):
    import pycbc.waveform._spherical_harmonics_torch as harmonics_torch

    monkeypatch.setattr(
        harmonics_torch,
        "_build_cudagraph_spin_minus_two_phi_zero",
        lambda *args, **kwargs: pytest.fail(
            "forward AD must bypass CUDA Graph capture"
        ),
    )
    primal = torch.tensor(0.731, dtype=torch.float64, device="cuda")
    tangent = torch.ones_like(primal)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(primal, tangent)
        eager = spin_minus_two_spherical_harmonics_phi_zero(
            dual,
            dtype=torch.float64,
            device="cuda",
        )
        candidate = cudagraphed_spin_minus_two_spherical_harmonics_phi_zero(
            dual,
            dtype=torch.float64,
            device="cuda",
        )
        for mode in eager:
            eager_dual = torch.autograd.forward_ad.unpack_dual(eager[mode])
            candidate_dual = torch.autograd.forward_ad.unpack_dual(
                candidate[mode]
            )
            assert torch.equal(candidate_dual.primal, eager_dual.primal)
            assert torch.equal(candidate_dual.tangent, eager_dual.tangent)


@pytest.mark.parametrize(
    "indices, exception",
    [
        ((-2, 2.0, 2), TypeError),
        ((-3, 2, 2), ValueError),
        ((-2, 2, 3), ValueError),
    ],
)
def test_torch_harmonics_validate_indices(indices, exception):
    with pytest.raises(exception):
        spin_weighted_spherical_harmonic(
            0.2,
            0.3,
            *indices,
            dtype=torch.float64,
            device="cpu",
        )
