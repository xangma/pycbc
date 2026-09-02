import numpy as np
import pytest

from pycbc.opt import LimitedSizeDict
from pycbc.types import Array


def test_slice_cache_is_lazy_and_preserves_memoization():
    source = Array(np.arange(64, dtype=np.float32))
    assert source._saved is None

    first = source[1:8]

    assert isinstance(source._saved, LimitedSizeDict)
    assert first._saved is None
    assert source[1:8] is first
    np.testing.assert_array_equal(first.numpy(), np.arange(1, 8))


def test_failed_slice_does_not_allocate_cache():
    source = Array(np.arange(4, dtype=np.float32))

    with pytest.raises(ValueError):
        source[::0]

    assert source._saved is None


def test_slice_cache_retains_fifo_limit():
    source = Array(np.arange(64, dtype=np.float32))
    oldest = source[0:1]
    for start in range(1, 33):
        source[start:start + 1]

    assert len(source._saved) == 32
    assert source[0:1] is not oldest


def test_data_rebind_invalidates_slice_cache():
    source = Array(np.arange(4, dtype=np.float32))
    old = source[:]

    source.data = np.arange(10, 14, dtype=np.float32)

    assert source._saved is None
    new = source[:]
    assert new is not old
    np.testing.assert_array_equal(new.numpy(), np.arange(10, 14))


def test_resize_and_roll_invalidate_slice_cache():
    resized = Array(np.arange(4, dtype=np.float32))
    resized[0:2]
    resized.resize(2)

    assert resized._saved is None
    resized[0] = np.float32(9)
    np.testing.assert_array_equal(resized[:].numpy(), [9, 1])

    rolled = Array(np.arange(4, dtype=np.float32))
    old = rolled[:]
    rolled.roll(1)

    assert rolled._saved is None
    new = rolled[:]
    assert new is not old
    np.testing.assert_array_equal(new.numpy(), [3, 0, 1, 2])


def test_scheme_conversion_invalidates_slice_cache():
    pytest.importorskip("torch")
    from pycbc.scheme import CPUScheme, TorchScheme

    with CPUScheme(num_threads=1):
        source = Array(np.arange(4, dtype=np.float32))
        old = source[:]

    with TorchScheme("cpu"):
        source.numpy()
        assert source._saved is None
        new = source[:]
        assert new is not old
        np.testing.assert_array_equal(new.numpy(), np.arange(4))


def test_torch_multiply_and_add_invalidates_slice_cache():
    pytest.importorskip("torch")
    from pycbc.scheme import TorchScheme

    with TorchScheme("cpu"):
        source = Array(np.asarray([1, 2], dtype=np.float32))
        other = Array(np.asarray([3, 4], dtype=np.float32))
        old = source[:]

        source.multiply_and_add(other, np.float32(2))

        assert source._saved is None
        new = source[:]
        assert new is not old
        np.testing.assert_array_equal(new.numpy(), [7, 10])
