# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch runtime transfers regression tests."""

import numpy as np
import pytest
import pycbc


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


def test_zeros_pinned_and_to_cuda_async_pipeline():
    from pycbc.types.array_torch import zeros_pinned, to_cuda_async, TorchArrayData

    pinned = zeros_pinned(64, dtype=np.float32)
    assert isinstance(pinned, TorchArrayData)
    assert pinned.shape == (64,)
    assert pinned.dtype == np.dtype(np.float32)
    if torch.cuda.is_available():
        assert pinned.tensor.is_pinned()

    transferred = to_cuda_async(pinned)
    assert isinstance(transferred, TorchArrayData)
    if torch.cuda.is_available():
        assert transferred.tensor.device.type == "cuda"

    raw_transferred = to_cuda_async(pinned.tensor)
    assert isinstance(raw_transferred, torch.Tensor)
    if torch.cuda.is_available():
        assert raw_transferred.device.type == "cuda"
