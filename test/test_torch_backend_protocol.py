"""Tests for the public PyCBC array-backend protocol."""

import numpy
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme  # noqa: E402
from pycbc import conversions  # noqa: E402
from pycbc.types import Array  # noqa: E402
from pycbc.types.array_torch import TorchArrayData  # noqa: E402
from pycbc.types.backend import (  # noqa: E402
    backend_array, backend_name, is_backend,
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
