import sys
import types

import lal
import numpy as np
import pytest
from scipy import stats

try:
    import lal.utils  # noqa: F401
except ModuleNotFoundError:
    lal_utils = types.ModuleType("lal.utils")
    sys.modules["lal.utils"] = lal_utils
    lal.utils = lal_utils

from pycbc.inference import entropy


@pytest.mark.parametrize("base", [None, np.e, 2.0])
def test_torch_pdf_entropy_matches_scipy(base):
    torch = pytest.importorskip("torch")
    probabilities = np.array([[0.0, 2.0], [1.0, 3.0], [3.0, 5.0]])
    tensor = torch.tensor(probabilities, dtype=torch.float64)

    actual = entropy.entropy(tensor, base=base)

    np.testing.assert_allclose(
        actual.numpy(), stats.entropy(probabilities, base=base)
    )


@pytest.mark.parametrize("base", [None, np.e, 2.0])
def test_torch_pdf_kl_matches_scipy(base):
    torch = pytest.importorskip("torch")
    probabilities = np.array([0.0, 2.0, 3.0, 5.0])
    reference = np.array([1.0, 1.0, 4.0, 4.0])
    tensor = torch.tensor(probabilities, dtype=torch.float64)
    reference_tensor = torch.tensor(reference, dtype=torch.float64)

    actual = entropy.kl(
        tensor, reference_tensor, pdf1=True, pdf2=True, base=base
    )

    np.testing.assert_allclose(
        actual.numpy(), stats.entropy(probabilities, reference, base=base)
    )


def test_torch_pdf_kl_preserves_zero_reference_semantics():
    torch = pytest.importorskip("torch")

    actual = entropy.kl(
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        pdf1=True,
        pdf2=True,
    )

    assert torch.isinf(actual)


@pytest.mark.parametrize(
    "values,hist_range",
    [
        ([-1.0, -0.45, -0.1, 0.25, 0.85, 1.0], (-1.0, 1.0)),
        ([1.0, 1.0, 1.0], None),
        ([], None),
    ],
)
def test_torch_histogram_pdf_matches_numpy(values, hist_range):
    torch = pytest.importorskip("torch")
    samples = torch.tensor(values, dtype=torch.float64)
    hist_min, hist_max = hist_range or (None, None)

    actual = entropy.compute_pdf(
        samples, "hist", 5, hist_min, hist_max
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = entropy.compute_pdf(
            np.asarray(values), "hist", 5, hist_min, hist_max
        )

    assert isinstance(actual, torch.Tensor)
    np.testing.assert_allclose(actual.numpy(), expected, equal_nan=True)


def test_torch_kde_evaluation_matches_scipy():
    torch = pytest.importorskip("torch")
    values = np.array([-1.2, -0.4, 0.1, 0.5, 1.7])
    points = np.array([-1.0, -0.2, 0.3, 1.1])
    samples = torch.tensor(values, dtype=torch.float64)
    evaluation_points = torch.tensor(points, dtype=torch.float64)
    bandwidth = (
        samples.std(unbiased=True)
        * samples.new_tensor(samples.numel() ** (-1.0 / 5.0))
    )

    actual = entropy._torch_kde_evaluate(
        samples, evaluation_points, bandwidth
    )
    expected = stats.gaussian_kde(values).evaluate(points)

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12)


def test_torch_kde_pdf_shape_and_values():
    torch = pytest.importorskip("torch")
    samples = torch.tensor(
        [-0.9, -0.4, -0.1, 0.3, 0.8], dtype=torch.float64
    )
    torch.manual_seed(1234)

    pdf = entropy.compute_pdf(samples, "kde", None, None, None)

    assert isinstance(pdf, torch.Tensor)
    assert pdf.shape == (10_000,)
    assert torch.isfinite(pdf).all()
    assert (pdf > 0).all()


def test_torch_histogram_information_metrics_match_scipy():
    torch = pytest.importorskip("torch")
    samples = np.array([-0.9, -0.7, -0.2, 0.1, 0.3, 0.6, 0.8, 0.9])
    reference = np.array([-0.8, -0.4, -0.1, 0.2, 0.4, 0.5, 0.7, 0.95])
    bins = 4
    hist_range = (-1.0, 1.0)
    expected_pdf = np.histogram(
        samples, bins=bins, range=hist_range, density=True
    )[0]
    expected_reference = np.histogram(
        reference, bins=bins, range=hist_range, density=True
    )[0]
    expected_kl = stats.entropy(expected_pdf, expected_reference, base=2.0)
    mixture = 0.5 * (expected_pdf + expected_reference)
    expected_js = 0.5 * (
        stats.entropy(expected_pdf, mixture, base=2.0)
        + stats.entropy(expected_reference, mixture, base=2.0)
    )

    actual_kl = entropy.kl(
        torch.tensor(samples, dtype=torch.float64),
        torch.tensor(reference, dtype=torch.float64),
        bins=bins,
        hist_min=hist_range[0],
        hist_max=hist_range[1],
        base=2.0,
    )
    actual_js = entropy.js(
        torch.tensor(samples, dtype=torch.float64),
        torch.tensor(reference, dtype=torch.float64),
        bins=bins,
        hist_min=hist_range[0],
        hist_max=hist_range[1],
        base=2.0,
    )

    assert isinstance(actual_kl, torch.Tensor)
    assert isinstance(actual_js, torch.Tensor)
    np.testing.assert_allclose(actual_kl.numpy(), expected_kl)
    np.testing.assert_allclose(actual_js.numpy(), expected_js)
