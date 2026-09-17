import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")

from pycbc.waveform._spherical_harmonics_torch import (  # noqa: E402
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
