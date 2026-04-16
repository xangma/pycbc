import warnings

import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.types import Array


def test_array_numpy_protocol_accepts_copy_kwarg():
    ary = Array([1.0, 2.0, 3.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = np.array(ary, copy=False)
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep == []
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))


@pytest.mark.skipif(not pycbc.HAVE_TORCH, reason="torch support unavailable")
def test_torch_array_data_numpy_protocol_accepts_copy_kwarg():
    old_scheme = scheme.mgr.state
    old_single = scheme.Scheme._single
    try:
        scheme.Scheme._single = None
        scheme.mgr.state = scheme.TorchScheme()
        ary = Array([1.0, 2.0, 3.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = np.array(ary._data, copy=False)
    finally:
        scheme.mgr.state = old_scheme
        scheme.Scheme._single = old_single
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep == []
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))
