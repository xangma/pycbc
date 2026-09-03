import pytest

torch = pytest.importorskip("torch")
lalsimulation = pytest.importorskip("lalsimulation")
from pycbc.waveform.nsbh_torch import nsbh_xi_tide  # noqa: E402


@pytest.mark.parametrize(
    ("mass_ratio", "spin", "mu"),
    [
        (1.0, -0.99, 0.1),
        (5.0, 0.8, 0.8),
        (8.0, -0.9, 1.28),
        (8.0, 0.8, 1.28),
        (100.0, 0.0, 50.0),
        (100.0, 0.99, 50.0),
    ],
)
def test_nsbh_xi_tide_torch_root_matches_lal(mass_ratio, spin, mu):
    expected = lalsimulation.SimNSBH_xi_tide(mass_ratio, spin, mu)
    assert nsbh_xi_tide(mass_ratio, spin, mu) == pytest.approx(
        expected, rel=2.0e-12, abs=2.0e-14
    )


def test_nsbh_xi_tide_uses_float64_torch_companion(monkeypatch):
    original = torch.linalg.eigvals
    seen = {}

    def record_companion(matrix):
        seen["shape"] = matrix.shape
        seen["dtype"] = matrix.dtype
        seen["device"] = matrix.device
        return original(matrix)

    monkeypatch.setattr(torch.linalg, "eigvals", record_companion)
    nsbh_xi_tide(5.0, 0.8, 0.8)

    assert seen == {
        "shape": torch.Size((10, 10)),
        "dtype": torch.float64,
        "device": torch.device("cpu"),
    }
