import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme
from pycbc.types import TimeSeries
from pycbc.types.array_torch import TorchArrayData


@pytest.fixture
def torch_cpu_ctx():
    ctx = scheme.TorchScheme("cpu")
    try:
        yield ctx
    finally:
        del ctx
        scheme.Scheme._single = None


def _series(dtype):
    samples = np.arange(64, dtype=np.float64)
    data = np.sin(samples / 7.0)
    if np.issubdtype(dtype, np.complexfloating):
        data = data + 1j * np.cos(samples / 9.0)
    return TimeSeries(
        data.astype(dtype), delta_t=1 / 1024, epoch=1_000_000_000
    )


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("interpolate", (None, "linear", "quadratic"))
@pytest.mark.parametrize(
    "sample_offset", (-0.25, 0.0, 0.25, 17.375, 62.625)
)
def test_torch_at_time_host_scalar_matches_vector_path(
        torch_cpu_ctx, dtype, interpolate, sample_offset):
    with torch_cpu_ctx:
        series = _series(dtype)
        query = float(series.start_time) + sample_offset * series.delta_t
        expected = series.at_time(
            np.asarray([query]), interpolate=interpolate
        )[0]
        actual = series.at_time(query, interpolate=interpolate)

    assert actual.ndim == 0
    assert actual.dtype == series._data.tensor.dtype
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
@pytest.mark.parametrize("interpolate", ("linear", "quadratic"))
@pytest.mark.parametrize(
    "sample_offset", (-0.25, 0.0, 0.25, 17.375, 62.625)
)
def test_torch_at_time_host_scalar_matches_legacy_strictly(
        torch_cpu_ctx, dtype, interpolate, sample_offset):
    reference = _series(dtype)
    query = float(reference.start_time) + sample_offset * reference.delta_t
    expected = reference.at_time(query, interpolate=interpolate)

    with torch_cpu_ctx:
        series = _series(dtype)
        actual = series.at_time(query, interpolate=interpolate)

    actual_array = actual.numpy()
    expected_array = np.asarray(expected)
    real_dtype = np.empty((), dtype=dtype).real.dtype
    if real_dtype == np.dtype(np.float64):
        # NumPy and Torch use the same arithmetic precision here.
        np.testing.assert_array_equal(actual_array, expected_array)
    else:
        # Legacy at_time promotes its float32 coordinate and therefore its
        # result to float64/complex128.  The established Torch API preserves
        # the series dtype.  Its result is exact against the Torch vector path
        # (tested above); the pre-existing Torch calculation differs from the
        # cast legacy value by at most two ULPs for these negative and boundary
        # indices.
        expected_array = expected_array.astype(dtype)
        np.testing.assert_array_max_ulp(
            actual_array.real, expected_array.real, maxulp=2
        )
        if np.iscomplexobj(actual_array):
            np.testing.assert_array_max_ulp(
                actual_array.imag, expected_array.imag, maxulp=2
            )


def test_torch_at_time_host_scalar_returns_owning_tensor(torch_cpu_ctx):
    with torch_cpu_ctx:
        series = _series(np.complex128)
        query = float(series.start_time) + 17.375 * series.delta_t
        actual = series.at_time(query, interpolate="quadratic")
        series_before = series._data.tensor.clone()

        # A NumPy-backed 0-D tensor cannot be resized.  The optimized result
        # retains the ordinary owning-storage behavior of the Torch path.
        actual.resize_(2)

    assert actual.shape == (2,)
    torch.testing.assert_close(
        series._data.tensor, series_before, rtol=0, atol=0
    )


def test_torch_at_time_host_scalar_avoids_coordinate_tensor(
        torch_cpu_ctx, monkeypatch):
    with torch_cpu_ctx:
        series = _series(np.complex128)
        query = float(series.start_time) + 21.625 * series.delta_t
        expected = series.at_time(
            np.asarray([query]), interpolate="quadratic"
        )[0]

        def _reject_as_tensor(*_args, **_kwargs):
            raise AssertionError("scalar lookup created a coordinate tensor")

        monkeypatch.setattr(torch, "as_tensor", _reject_as_tensor)
        actual = series.at_time(query, interpolate="quadratic")

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "coordinate_kind", ("tensor", "vector", "numpy_vector")
)
def test_torch_at_time_non_host_scalar_coordinates_use_torch_path(
        torch_cpu_ctx, monkeypatch, coordinate_kind):
    with torch_cpu_ctx:
        series = _series(np.complex128)
        query = float(series.start_time) + 21.625 * series.delta_t
        if coordinate_kind == "tensor":
            coordinate = torch.tensor(query, dtype=torch.float64)
        elif coordinate_kind == "vector":
            coordinate = torch.tensor([query], dtype=torch.float64)
        else:
            coordinate = np.asarray([query], dtype=np.float64)
        expected = series.at_time(coordinate, interpolate="quadratic")

        with monkeypatch.context() as patch:
            patch.setattr(
                torch.Tensor,
                "numpy",
                lambda _self: pytest.fail(
                    f"{coordinate_kind} coordinate visited NumPy"
                ),
            )
            actual = series.at_time(coordinate, interpolate="quadratic")

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device_name", ("cuda", "mps"))
def test_torch_at_time_accelerator_host_scalar_uses_existing_path(
        monkeypatch, device_name):
    if device_name == "cuda":
        available = torch.cuda.is_available()
    else:
        available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    if not available:
        pytest.skip(f"{device_name} is unavailable")

    device = torch.device(device_name)
    dtype = torch.float32 if device_name == "mps" else torch.float64
    with scheme.TorchScheme(device_name):
        tensor = torch.sin(
            torch.arange(64, device=device, dtype=dtype) / 7.0
        )
        series = TimeSeries(
            TorchArrayData(tensor), delta_t=1 / 1024,
            epoch=1_000_000_000, copy=False,
        )
        query = float(series.start_time) + 17.375 * series.delta_t
        expected = series.at_time(
            np.asarray([query]), interpolate="quadratic"
        )[0]

        with monkeypatch.context() as patch:
            patch.setattr(
                torch.Tensor,
                "numpy",
                lambda _self: pytest.fail(
                    f"{device_name} storage visited NumPy"
                ),
            )
            actual = series.at_time(query, interpolate="quadratic")

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_torch_at_time_nearest_sample_uses_existing_path(
        torch_cpu_ctx, monkeypatch):
    with torch_cpu_ctx:
        series = _series(np.float64)
        query = float(series.start_time) + 17.625 * series.delta_t
        expected = series.at_time(
            np.asarray([query]), nearest_sample=True
        )[0]

        with monkeypatch.context() as patch:
            patch.setattr(
                torch.Tensor,
                "numpy",
                lambda _self: pytest.fail(
                    "nearest-sample lookup visited NumPy"
                ),
            )
            actual = series.at_time(query, nearest_sample=True)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("sample_offset", "interpolate"),
    ((-0.25, None), (0.25, "quadratic"), (7.6, None)),
)
def test_torch_at_time_host_scalar_boundary_behavior(
        torch_cpu_ctx, sample_offset, interpolate):
    with torch_cpu_ctx:
        series = _series(np.complex128)
        query = float(series.start_time) + sample_offset * series.delta_t
        expected = series.at_time(
            np.asarray([query]), interpolate=interpolate,
            nearest_sample=interpolate is None,
        )[0]
        actual = series.at_time(
            query, interpolate=interpolate,
            nearest_sample=interpolate is None,
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("sample_offset", "interpolate", "fill_value"),
    (
        (0.5, "quadratic", -3.0),
        (63.5, "linear", 2.0),
        (-0.25, None, 1.0 + 0.0j),
    ),
)
def test_torch_at_time_host_scalar_extrapolation(
        torch_cpu_ctx, monkeypatch, sample_offset, interpolate, fill_value):
    with torch_cpu_ctx:
        series = _series(np.float64)
        query = float(series.start_time) + sample_offset * series.delta_t

        with monkeypatch.context() as patch:
            patch.setattr(
                torch.Tensor,
                "numpy",
                lambda _self: pytest.fail(
                    "extrapolation used the native scalar fast path"
                ),
            )
            expected = series.at_time(
                np.asarray([query]), interpolate=interpolate,
                extrapolate=fill_value,
            )[0]
            actual = series.at_time(
                query, interpolate=interpolate, extrapolate=fill_value
            )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_torch_at_time_host_scalar_errors_match_vector_path(torch_cpu_ctx):
    with torch_cpu_ctx:
        series = _series(np.float64)
        out_of_bounds = float(series.end_time) + series.delta_t
        with pytest.raises(IndexError):
            series.at_time(out_of_bounds, interpolate="quadratic")
        with pytest.raises(IndexError):
            series.at_time(
                np.asarray([out_of_bounds]), interpolate="quadratic"
            )
        with pytest.raises(ValueError, match="Unsupported extrapolate"):
            series.at_time(
                float(series.start_time), extrapolate="unsupported"
            )


@pytest.mark.parametrize("extrapolate", (None, 0.0))
def test_torch_at_time_host_scalar_preserves_data_gradient(
        torch_cpu_ctx, monkeypatch, extrapolate):
    with torch_cpu_ctx:
        series = _series(np.complex128)
        data = series._data.tensor.requires_grad_()
        query = float(series.start_time) + 19.375 * series.delta_t

        with monkeypatch.context() as patch:
            patch.setattr(
                torch.Tensor,
                "numpy",
                lambda _self: pytest.fail(
                    "gradient-carrying storage visited NumPy"
                ),
            )
            actual = series.at_time(
                query, interpolate="quadratic", extrapolate=extrapolate
            )
            actual_loss = actual.real + 0.25 * actual.imag
            actual_gradient = torch.autograd.grad(
                actual_loss, data, retain_graph=True
            )[0]

            expected = series.at_time(
                np.asarray([query]), interpolate="quadratic",
                extrapolate=extrapolate,
            )[0]
            expected_loss = expected.real + 0.25 * expected.imag
            expected_gradient = torch.autograd.grad(expected_loss, data)[0]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=0, atol=0
    )


def test_torch_at_time_host_scalar_preserves_forward_ad(
        torch_cpu_ctx, monkeypatch):
    with torch_cpu_ctx:
        with torch.autograd.forward_ad.dual_level():
            primal = torch.linspace(-1.0, 1.0, 64, dtype=torch.float64)
            tangent = torch.linspace(0.25, 2.0, 64, dtype=torch.float64)
            dual = torch.autograd.forward_ad.make_dual(primal, tangent)
            assert not dual.requires_grad
            series = TimeSeries(
                TorchArrayData(dual), delta_t=1 / 1024,
                epoch=1_000_000_000, copy=False,
            )
            query = float(series.start_time) + 19.375 * series.delta_t
            expected = series.at_time(
                np.asarray([query]), interpolate="quadratic"
            )[0]

            with monkeypatch.context() as patch:
                patch.setattr(
                    torch.Tensor,
                    "numpy",
                    lambda _self: pytest.fail(
                        "forward-AD storage visited NumPy"
                    ),
                )
                actual = series.at_time(query, interpolate="quadratic")

            actual_primal, actual_tangent = (
                torch.autograd.forward_ad.unpack_dual(actual)
            )
            expected_primal, expected_tangent = (
                torch.autograd.forward_ad.unpack_dual(expected)
            )

    torch.testing.assert_close(
        actual_primal, expected_primal, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_tangent, expected_tangent, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "storage_kind", ("noncontiguous", "conjugate", "negative")
)
def test_torch_at_time_host_scalar_nonstandard_storage_uses_torch_path(
        torch_cpu_ctx, monkeypatch, storage_kind):
    with torch_cpu_ctx:
        if storage_kind == "noncontiguous":
            base = torch.arange(128, dtype=torch.float64)
            tensor = base[::2]
        elif storage_kind == "conjugate":
            base = torch.arange(64, dtype=torch.float64).to(torch.complex128)
            tensor = (base + 0.25j).conj()
        else:
            base = torch.arange(64, dtype=torch.float64)
            tensor = torch._neg_view(base)

        series = TimeSeries(
            TorchArrayData(tensor), delta_t=1 / 1024,
            epoch=1_000_000_000, copy=False,
        )
        assert series._data.tensor is tensor
        query = float(series.start_time) + 17.375 * series.delta_t
        expected = series.at_time(
            torch.tensor(query, dtype=torch.float64),
            interpolate="quadratic",
        )

        with monkeypatch.context() as patch:
            patch.setattr(
                torch.Tensor,
                "numpy",
                lambda _self: pytest.fail(
                    f"{storage_kind} storage visited NumPy"
                ),
            )
            actual = series.at_time(query, interpolate="quadratic")

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_torch_at_time_host_scalar_preserves_extrapolation_gradient(
        torch_cpu_ctx):
    with torch_cpu_ctx:
        series = _series(np.complex128)
        data = series._data.tensor.requires_grad_()
        query = float(series.end_time) + series.delta_t

        actual = series.at_time(query, extrapolate=2.0)
        actual_gradient = torch.autograd.grad(actual.real, data)[0]

        expected = series.at_time(
            np.asarray([query]), extrapolate=2.0
        )[0]
        expected_gradient = torch.autograd.grad(expected.real, data)[0]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=0, atol=0
    )
