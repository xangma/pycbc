"""Precision, storage and differentiation contracts for squared norms."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme  # noqa: E402
from pycbc.types import Array  # noqa: E402
from pycbc.types.array_torch import TorchArrayData  # noqa: E402


@pytest.fixture
def torch_cpu():
    context = scheme.TorchScheme("cpu")
    try:
        with context:
            yield
    finally:
        del context
        scheme.Scheme._single = None


def _squared_norm(tensor):
    array = Array(TorchArrayData(tensor), copy=False)
    return array.squared_norm()._data.tensor


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("size", (0, 17, 4095, 4096, 8193, 131072, 1048577))
@pytest.mark.parametrize("stride", (1, 3))
def test_squared_norm_preserves_bits_and_input(torch_cpu, dtype, size, stride):
    """Cover the branch boundary, search sizes, and noncontiguous views."""
    generator = torch.Generator().manual_seed(170817)
    storage = torch.randn(size * stride + 1, dtype=dtype, generator=generator)
    original = storage.clone()
    tensor = storage[1::stride]
    expected = torch.view_as_real(tensor).square().sum(dim=-1)

    actual = _squared_norm(tensor)

    assert actual.dtype == tensor.real.dtype
    assert actual.shape == tensor.shape
    integer_dtype = torch.int32 if dtype == torch.complex64 else torch.int64
    assert torch.equal(
        actual.view(integer_dtype), expected.view(integer_dtype)
    )
    assert torch.equal(storage, original)
    if size:
        actual.fill_(0)
        assert torch.equal(storage, original)


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_squared_norm_extreme_values(torch_cpu, dtype):
    """Retain rounding, subnormals, signed zero, overflow, NaN and infinity."""
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    info = torch.finfo(real_dtype)
    values = torch.tensor(
        [0.0, -0.0, info.tiny, info.tiny**0.5, info.eps, 1.0,
         info.max**0.5, info.max, float("inf"), -float("inf"), float("nan")],
        dtype=real_dtype,
    )
    real, imag = torch.meshgrid(values, values, indexing="ij")
    tensor = torch.complex(real.flatten(), imag.flatten()).repeat(40)
    expected = torch.view_as_real(tensor).square().sum(dim=-1)

    actual = _squared_norm(tensor)

    torch.testing.assert_close(
        actual, expected, rtol=0, atol=0, equal_nan=True
    )
    assert torch.equal(torch.signbit(actual[~expected.isnan()]),
                       torch.signbit(expected[~expected.isnan()]))


@pytest.mark.parametrize("stride", (1, 3))
def test_squared_norm_reverse_and_forward_gradients(torch_cpu, stride):
    generator = torch.Generator().manual_seed(2017)
    tensor = torch.randn(4096 * stride, dtype=torch.complex128,
                         generator=generator, requires_grad=True)
    view = tensor[::stride]
    expected = torch.view_as_real(view).square().sum(dim=-1)
    actual = _squared_norm(view)
    expected_gradient, = torch.autograd.grad(expected.sum(), tensor)
    actual_gradient, = torch.autograd.grad(actual.sum(), tensor)
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=0, atol=0
    )

    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(
            view.detach(), torch.ones_like(view)
        )
        reference = torch.autograd.forward_ad.unpack_dual(
            torch.view_as_real(dual).square().sum(dim=-1)
        )
        result = torch.autograd.forward_ad.unpack_dual(_squared_norm(dual))
        torch.testing.assert_close(
            result.primal, reference.primal, rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.tangent, reference.tangent, rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.int64))
def test_squared_norm_real_inputs(torch_cpu, dtype):
    tensor = torch.arange(-4096, 4096, dtype=dtype)
    actual = _squared_norm(tensor)
    assert actual.dtype == dtype
    assert torch.equal(actual, tensor.square())


def test_squared_norm_lazy_conjugate_keeps_existing_error(torch_cpu):
    tensor = torch.ones(4096, dtype=torch.complex64).conj()
    assert tensor.is_conj()
    with pytest.raises(RuntimeError, match="unresolved conjugated"):
        _squared_norm(tensor)
    assert tensor.is_conj()


def test_squared_norm_respects_subclass_reduction(torch_cpu):
    class SpecialReduction(torch.Tensor):
        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            if function is torch.Tensor.sum:
                return torch.full_like(args[0][..., 0], 42)
            return super().__torch_function__(function, types, args, kwargs)

    tensor = torch.ones(4096, dtype=torch.complex64).as_subclass(
        SpecialReduction
    )
    actual = _squared_norm(tensor)
    assert isinstance(actual, SpecialReduction)
    assert torch.all(actual == 42)


@pytest.mark.parametrize("device", ("cuda", "mps"))
def test_squared_norm_accelerator_route(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    values = np.random.default_rng(2017).standard_normal((8193, 2))
    values = values.astype(np.float32)
    context = scheme.TorchScheme(device)
    try:
        with context:
            tensor = torch.view_as_complex(
                torch.as_tensor(values, device=device)
            )
            expected = torch.view_as_real(tensor).square().sum(dim=-1)
            actual = _squared_norm(tensor)
            assert actual.device == tensor.device
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        del context
        scheme.Scheme._single = None
