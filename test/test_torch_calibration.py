"""Parity tests for the Torch calibration smoothing spline."""

import numpy as np
import pytest
import scipy.interpolate

from pycbc import scheme
from pycbc.strain import calibration
from pycbc.types import FrequencySeries

torch = pytest.importorskip("torch")


def _scipy_full_knots(spline):
    base_knots = spline.get_knots()
    degree = len(spline.get_coeffs()) - len(base_knots) + 1
    return np.concatenate((
        np.repeat(base_knots[0], degree),
        base_knots,
        np.repeat(base_knots[-1], degree),
    ))


def _torch_spline(x, y, samples):
    reference = torch.empty((), dtype=torch.float64)
    knots, coefficients = calibration._torch_fitpack_spline(
        x, torch.as_tensor(y, dtype=torch.float64), reference
    )
    values = calibration._evaluate_torch_spline(
        knots,
        coefficients,
        torch.as_tensor(samples, dtype=torch.float64),
        calibration._FITPACK_DEGREE,
    )
    return (
        knots.detach().numpy(),
        coefficients.detach().numpy(),
        values.detach().numpy(),
    )


@pytest.mark.parametrize(
    "x,y",
    (
        (
            np.geomspace(10, 1000, 4),
            np.array((0.0, 2.0, -1.0, 3.0)),
        ),
        (
            np.geomspace(10, 1000, 5),
            np.array((0.01, -0.02, 0.03, -0.01, 0.02)),
        ),
        (
            np.geomspace(10, 1024, 8),
            np.array((0.0, 4.0, -3.0, 5.0, -4.0, 3.0, -2.0, 0.0)),
        ),
        (
            np.geomspace(10, 1000, 20),
            np.random.default_rng(125).normal(size=20) * 5,
        ),
    ),
    ids=("interpolating-cubic", "global-cubic", "adaptive", "continued-nest"),
)
def test_torch_fitpack_curated_parity(x, y):
    """Cover global, adaptive, and legacy workspace-continuation paths."""
    scipy_spline = scipy.interpolate.UnivariateSpline(x, y)
    samples = np.linspace(0, 1200, 257)
    knots, coefficients, values = _torch_spline(x, y, samples)

    np.testing.assert_array_equal(knots, _scipy_full_knots(scipy_spline))
    np.testing.assert_allclose(
        coefficients, scipy_spline.get_coeffs(), rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        values, scipy_spline(samples), rtol=2e-11, atol=2e-10
    )

    torch_at_data = _torch_spline(x, y, x)[2]
    np.testing.assert_allclose(
        np.sum((torch_at_data - y) ** 2),
        scipy_spline.get_residual(),
        rtol=2e-11,
        atol=2e-11,
    )


def test_torch_fitpack_legacy_continuation_topology():
    """Lock the topology that requires UnivariateSpline's small-nest resume."""
    x = np.geomspace(10, 1000, 20)
    y = np.random.default_rng(125).normal(size=20) * 5
    knots = _torch_spline(x, y, x)[0]

    assert len(knots) == 18
    assert 183.29807108324357 in knots


@pytest.mark.parametrize("point_count", (5, 8, 12, 20, 33))
def test_torch_fitpack_randomized_adaptive_parity(point_count):
    """Compare randomized knot topology, coefficients, and extrapolation."""
    scales = (0.01, 0.5, 2.0, 5.0, 10.0, 100.0)
    for seed, scale in enumerate(scales):
        rng = np.random.default_rng(7000 + 100 * point_count + seed)
        if seed % 2:
            x = np.geomspace(10, 1000, point_count)
        else:
            x = np.sort(rng.uniform(10, 1000, point_count))
        y = rng.normal(size=point_count) * scale
        samples = np.linspace(0, 1200, 97)

        scipy_spline = scipy.interpolate.UnivariateSpline(x, y)
        knots, coefficients, values = _torch_spline(x, y, samples)

        np.testing.assert_array_equal(knots, _scipy_full_knots(scipy_spline))
        np.testing.assert_allclose(
            coefficients,
            scipy_spline.get_coeffs(),
            rtol=2e-10,
            atol=2e-10,
        )
        np.testing.assert_allclose(
            values,
            scipy_spline(samples),
            rtol=2e-10,
            atol=2e-10,
        )


@pytest.mark.parametrize(
    "x",
    (
        np.array((1.0, 3.0, 2.0, 4.0)),
        np.array((1.0, 2.0, np.nan, 4.0)),
    ),
)
def test_torch_fitpack_rejects_invalid_abscissae_like_scipy(x):
    y = np.arange(4.0)
    with pytest.raises(ValueError, match="x must be increasing"):
        scipy.interpolate.UnivariateSpline(x, y)
    with pytest.raises(ValueError, match="x must be increasing"):
        _torch_spline(x, y, x)


@pytest.mark.parametrize(
    "y",
    (
        np.array((1.0, 2.0, np.nan, 4.0)),
        np.array((1.0, 2.0, np.inf, 4.0)),
    ),
)
def test_torch_fitpack_preserves_nonfinite_result_semantics(y):
    x = np.arange(4.0)
    samples = np.linspace(x[0], x[-1], 9)
    expected = scipy.interpolate.UnivariateSpline(x, y)(samples)
    actual = _torch_spline(x, y, samples)[2]

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))


@pytest.mark.parametrize(
    "x",
    (
        np.array((1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
        np.logspace(
            np.log10(1.0),
            np.log10(np.nextafter(np.nextafter(1.0, np.inf), np.inf)),
            4,
        ),
    ),
)
def test_torch_calibration_degenerate_nodes_keep_scipy_fallback(
        monkeypatch, x):
    """Keep FITPACK's singular repeated-node behavior at the legacy edge."""
    amplitude = np.arange(len(x), dtype=np.float64)
    phase = amplitude[::-1].copy()
    data = np.linspace(1, 2, 17).astype(np.complex128)
    strain = FrequencySeries(data, delta_f=0.25, epoch=123)
    expected = calibration._apply_spline_calibration(
        strain, x, amplitude, phase
    )

    calls = []
    original = scipy.interpolate.UnivariateSpline

    def recording_spline(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy.interpolate, "UnivariateSpline", recording_spline)
    with scheme.TorchScheme("cpu"):
        actual = calibration._apply_spline_calibration(
            FrequencySeries(data, delta_f=0.25, epoch=123),
            x,
            amplitude,
            phase,
        )

    assert len(calls) == 2
    np.testing.assert_allclose(
        actual.numpy(), expected.numpy(), rtol=1e-14, atol=1e-14
    )


def test_torch_fitpack_preserves_maxit_warning():
    x = np.geomspace(0.01, 1e5, 5)
    y = np.random.default_rng(115008).normal(size=5) * 1e5

    with pytest.warns(UserWarning, match="maximal number of iterations"):
        scipy.interpolate.UnivariateSpline(x, y)
    with pytest.warns(UserWarning, match="maximal number of iterations"):
        _torch_spline(x, y, x)


def test_torch_fitpack_adaptive_gradcheck():
    x = np.geomspace(10, 1024, 8)
    y = torch.tensor(
        (0.0, 4.0, -3.0, 5.0, -4.0, 3.0, -2.0, 0.0),
        dtype=torch.float64,
        requires_grad=True,
    )
    samples = torch.linspace(0, 1200, 17, dtype=torch.float64)

    def evaluate(parameters):
        knots, coefficients = calibration._torch_fitpack_spline(
            x, parameters, samples
        )
        return calibration._evaluate_torch_spline(
            knots,
            coefficients,
            samples,
            calibration._FITPACK_DEGREE,
        )

    assert torch.autograd.gradcheck(
        evaluate, (y,), eps=1e-6, atol=2e-4, rtol=2e-3
    )


def _available_fit_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.mark.parametrize("device", _available_fit_devices())
def test_torch_fitpack_preserves_device_dtype_and_graph(device):
    x = np.geomspace(10, 1024, 8)
    y_numpy = np.array((0.0, 4.0, -3.0, 5.0, -4.0, 3.0, -2.0, 0.0))
    y = torch.tensor(
        y_numpy, device=device, dtype=torch.float32, requires_grad=True
    )
    samples = torch.linspace(0, 1200, 67, device=device, dtype=torch.float32)

    knots, coefficients = calibration._torch_fitpack_spline(x, y, samples)
    values = calibration._evaluate_torch_spline(
        knots,
        coefficients,
        samples,
        calibration._FITPACK_DEGREE,
    )
    expected = scipy.interpolate.UnivariateSpline(x, y_numpy)(
        samples.detach().cpu().numpy()
    )

    assert knots.device.type == device
    assert coefficients.device.type == device
    assert values.device.type == device
    expected_dtype = torch.float32 if device == "mps" else torch.float64
    assert coefficients.dtype == expected_dtype
    tolerance = 3e-4 if device == "mps" else 2e-10
    np.testing.assert_allclose(
        values.detach().cpu().numpy(), expected,
        rtol=tolerance, atol=tolerance,
    )

    values.sum().backward()
    assert y.grad is not None
    assert bool(torch.isfinite(y.grad).all().cpu())


def _calibration_model(parameters_as_tensors=False):
    model = calibration.CubicSpline(
        ifo_name="h1", minimum_frequency=10,
        maximum_frequency=1024, n_points=8,
    )
    amplitude = (0.0, 4.0, -3.0, 5.0, -4.0, 3.0, -2.0, 0.0)
    phase = (0.0, -2.0, 3.0, -4.0, 3.0, -2.0, 1.0, 0.0)
    tensors = []
    for index, (amp, pha) in enumerate(zip(amplitude, phase)):
        if parameters_as_tensors:
            amp = torch.tensor(amp, dtype=torch.float64, requires_grad=True)
            pha = torch.tensor(pha, dtype=torch.float64, requires_grad=True)
            tensors.extend((amp, pha))
        model.params[f"amplitude_h1_{index}"] = amp
        model.params[f"phase_h1_{index}"] = pha
    return model, tensors


def test_torch_calibration_avoids_scipy_fit_and_preserves_gradients(
        monkeypatch):
    data = (
        np.linspace(1, 2, 129) + 1j * np.linspace(-0.5, 0.5, 129)
    ).astype(np.complex128)
    expected = _calibration_model()[0].apply_calibration(
        FrequencySeries(data, delta_f=8, epoch=123)
    )

    def reject_scipy_fit(*_args, **_kwargs):
        raise AssertionError("Torch calibration called SciPy/FITPACK")

    monkeypatch.setattr(
        scipy.interpolate, "UnivariateSpline", reject_scipy_fit
    )
    with scheme.TorchScheme("cpu"):
        strain = FrequencySeries(data, delta_f=8, epoch=123)
        model, parameters = _calibration_model(parameters_as_tensors=True)
        actual = model.apply_calibration(strain)
        loss = actual._data.tensor.real.sum()
        loss = loss + 0.37 * actual._data.tensor.imag.sum()
        loss.backward()

    np.testing.assert_allclose(
        actual._data.tensor.detach().numpy(),
        expected.numpy(),
        rtol=2e-11,
        atol=2e-11,
    )
    assert actual._data.tensor.device.type == "cpu"
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad) for parameter in parameters)


def test_cpu_calibration_keeps_scipy_fit(monkeypatch):
    calls = []
    original = scipy.interpolate.UnivariateSpline

    def recording_spline(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy.interpolate, "UnivariateSpline", recording_spline)
    data = np.linspace(1, 2, 129).astype(np.complex128)
    model, _ = _calibration_model()
    model.apply_calibration(FrequencySeries(data, delta_f=8, epoch=123))

    assert len(calls) == 2
