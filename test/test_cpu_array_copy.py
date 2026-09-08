"""Preserve CPU conversion ownership while supporting the NumPy copy keyword."""
import numpy as np
import pytest

from pycbc.types import Array


@pytest.mark.parametrize("dtype", (np.float32, np.float64, np.complex64))
def test_explicit_same_dtype_conversion_copies_cpu_storage(dtype):
    array = Array(np.array([1, 2, 3], dtype=dtype))
    converted = array.__array__(dtype=array.dtype)
    assert not np.shares_memory(converted, array.data)
    converted[0] = 9
    assert array[0] == 1


@pytest.mark.parametrize("copy", (None, False, True))
def test_cpu_conversion_copy_keyword(copy):
    array = Array([1., 2.])
    converted = array.__array__(copy=copy)
    assert np.shares_memory(converted, array.data) == (copy is not True)
    converted = array.__array__(dtype=array.dtype, copy=copy)
    assert np.shares_memory(converted, array.data) == (copy is False)
