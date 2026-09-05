import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pycbc
from pycbc import scheme
from pycbc.types import Array


if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


@pytest.fixture
def torch_ctx():
    ctx = scheme.TorchScheme("cpu")
    try:
        yield ctx
    finally:
        del ctx
        scheme.Scheme._single = None


def test_dtype_view_shares_torch_storage(torch_ctx):
    with torch_ctx:
        real = Array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        complex_view = real.view(np.complex64)

        assert complex_view.ptr == real.ptr
        np.testing.assert_array_equal(
            complex_view.numpy(),
            np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64),
        )

        complex_view[0] = 5.0 + 6.0j
        np.testing.assert_array_equal(
            real.numpy(), np.array([5.0, 6.0, 3.0, 4.0], dtype=np.float32)
        )


def test_inplace_integer_operation_does_not_silently_truncate(torch_ctx):
    with torch_ctx:
        values = Array(np.array([1, 2], dtype=np.int32))
        with pytest.raises(RuntimeError, match="can't be cast"):
            values += 0.5
