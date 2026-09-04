"""Tests for the public PyCBC array-backend protocol."""

import numpy
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme  # noqa: E402
from pycbc import conversions  # noqa: E402
from pycbc.types import Array, TimeSeries  # noqa: E402
from pycbc.types.array_torch import TorchArrayData  # noqa: E402
from pycbc.types.backend import (  # noqa: E402
    backend_array, backend_name, is_backend, wrap_backend_array,
    backend_matches_scheme,
)


def test_torch_storage_uses_public_backend_protocol():
    tensor = torch.arange(3, dtype=torch.float64)
    wrapped = TorchArrayData(tensor)
    with scheme.TorchScheme("cpu"):
        array = Array(wrapped, copy=False)

    assert backend_name(tensor) == "torch"
    assert backend_name(wrapped) == "torch"
    assert backend_name(array) == "torch"
    assert is_backend(array, "torch")
    assert backend_array(array, "torch") is tensor
    assert backend_array(array, "numpy") is None
    assert backend_name(numpy.arange(2)) == "numpy"


def test_conversions_backend_accepts_public_torch_storage():
    tensor = torch.tensor([4.0, 9.0], dtype=torch.float64)
    with scheme.TorchScheme("cpu"):
        array = Array(TorchArrayData(tensor), copy=False)

    result = conversions.primary_mass(array, 5.0)

    assert isinstance(result, torch.Tensor)
    expected = conversions.primary_mass(tensor, 5.0)
    torch.testing.assert_close(result, expected)


def test_hypertriangle_checks_shapes_of_tensor_subclasses():
    class TensorSubclass(torch.Tensor):
        pass

    first = torch.tensor([0.2, 0.8]).as_subclass(TensorSubclass)
    second = torch.tensor([0.4]).as_subclass(TensorSubclass)
    with pytest.raises(AssertionError, match="same number of elements"):
        conversions.hypertriangle(first, second)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_public_storage_constructor_preserves_views_and_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    source = torch.arange(8, dtype=torch.float32, device=device,
                          requires_grad=True)
    tensor = source[::2]
    with scheme.TorchScheme(device):
        assert backend_matches_scheme(tensor)
        series = TimeSeries(wrap_backend_array(tensor), delta_t=0.25,
                            epoch=123, copy=False)
        assert backend_array(series) is tensor
        assert series.delta_t == 0.25
        assert float(series.start_time) == 123
        backend_array(series).sum().backward()
        torch.testing.assert_close(source.grad[::2], torch.ones_like(tensor))
        torch.testing.assert_close(source.grad[1::2], torch.zeros_like(tensor))
        if device != "cpu":
            assert not backend_matches_scheme(torch.zeros(2))
            with pytest.raises(TypeError, match="Cannot avoid a copy"):
                Array(wrap_backend_array(torch.zeros(2)), copy=False)


def test_public_storage_constructor_preserves_numpy_and_rejects_wrong_scheme():
    values = numpy.arange(3, dtype=numpy.float64)
    with scheme.CPUScheme():
        assert wrap_backend_array(values) is values
        assert backend_matches_scheme(values)
        array = Array(wrap_backend_array(values), copy=False)
        assert backend_array(array) is values
        assert not backend_matches_scheme(torch.zeros(2))
        with pytest.raises(TypeError, match="Cannot avoid a copy"):
            Array(wrap_backend_array(torch.zeros(2)), copy=False)
