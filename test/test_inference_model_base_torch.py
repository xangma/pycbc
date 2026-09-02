import sys
import types

import lal
import numpy as np
import pytest

try:
    import lal.utils  # noqa: F401
except ModuleNotFoundError:
    lal_utils = types.ModuleType("lal.utils")
    sys.modules["lal.utils"] = lal_utils
    lal.utils = lal_utils

from pycbc import transforms
from pycbc.inference.models.base import SamplingTransforms


def _log_sampling_transform():
    return SamplingTransforms(
        variable_params=["x"],
        sampling_params=["logx"],
        replace_parameters=["x"],
        sampling_transforms=[transforms.Log("x", "logx")],
    )


def test_sampling_logjacobian_torch_matches_numpy_and_has_gradients():
    torch = pytest.importorskip("torch")
    values = np.array([-1.5, 0.25, 2.0])
    sampling = _log_sampling_transform()
    expected = sampling.logjacobian(logx=values)

    tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    actual = sampling.logjacobian(logx=tensor)

    assert isinstance(actual, torch.Tensor)
    assert actual.device == tensor.device
    np.testing.assert_allclose(actual.detach().numpy(), expected)
    actual.sum().backward()
    torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))


@pytest.mark.parametrize("device_name", ["cpu", "mps"])
def test_real_inner_self_optimization(device_name):
    torch = pytest.importorskip("torch")
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")

    from pycbc.inference.models.tools import _real_inner
    from pycbc.types import Array
    from pycbc.scheme import TorchScheme

    device = torch.device(device_name)
    dtype = torch.complex64 if device_name == "mps" else torch.complex128

    # 1. Direct Torch tensor (complex)
    t = torch.randn(100, dtype=dtype, device=device, requires_grad=True)
    res_self = _real_inner(t, t)
    assert isinstance(res_self, torch.Tensor)
    assert res_self.device.type == device.type
    expected = torch.sum(torch.conj(t) * t).real
    torch.testing.assert_close(res_self, expected)
    res_self.backward()
    assert t.grad is not None

    # 2. PyCBC Array with TorchScheme (testing left is right & _data identity)
    with TorchScheme():
        np_vals = np.random.randn(50) + 1j * np.random.randn(50)
        arr = Array(np_vals)
        slc = slice(5, 25)
        sliced1 = arr[slc]
        sliced2 = arr[slc]

        # left is right
        res_same = _real_inner(sliced1, sliced1)
        # sliced1._data is sliced2._data
        res_shared_data = _real_inner(sliced1, sliced2)
        expected_val = np.vdot(np_vals[slc], np_vals[slc]).real

        np.testing.assert_allclose(
            res_same.detach().cpu().numpy(), expected_val
        )
        np.testing.assert_allclose(
            res_shared_data.detach().cpu().numpy(), expected_val
        )


@pytest.mark.parametrize("device_name", ["cpu", "mps"])
def test_inner_product_complex_and_real(device_name):
    torch = pytest.importorskip("torch")
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")

    from pycbc.inference.models.tools import _inner, _real_inner

    device = torch.device(device_name)
    complex_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    real_dtype = torch.float32 if device_name == "mps" else torch.float64

    # Complex inner product
    a = torch.randn(64, dtype=complex_dtype, device=device, requires_grad=True)
    b = torch.randn(64, dtype=complex_dtype, device=device, requires_grad=True)
    cplx_res = _inner(a, b)
    assert isinstance(cplx_res, torch.Tensor)
    assert cplx_res.device.type == device.type
    expected_cplx = torch.sum(torch.conj(a) * b, dtype=complex_dtype)
    torch.testing.assert_close(cplx_res, expected_cplx)

    (cplx_res.real + cplx_res.imag).backward()
    assert a.grad is not None
    assert b.grad is not None

    # Real inner product cross terms
    ar = torch.randn(64, dtype=real_dtype, device=device, requires_grad=True)
    br = torch.randn(64, dtype=real_dtype, device=device, requires_grad=True)
    real_res = _real_inner(ar, br)
    assert isinstance(real_res, torch.Tensor)
    expected_real = torch.sum(ar * br, dtype=real_dtype)
    torch.testing.assert_close(real_res, expected_real)
