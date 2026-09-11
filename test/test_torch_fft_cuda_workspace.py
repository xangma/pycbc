# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Torch fft cuda workspace regression tests."""

import numpy as np
import pytest
from pycbc import scheme
from pycbc.types import zeros
import pycbc


torch = pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


if not torch.cuda.is_available():
    pytest.skip("CUDA device is required", allow_module_level=True)


PROMOTED_ROWS_GATE = "PYCBC_TORCH_CUDA_PROMOTED_ROWS"


def _complex_values(rows, size, seed):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(rows, size)) + 1j * rng.normal(size=(rows, size))).astype(
        np.complex64
    )


def test_cuda_promoted_rows_gate_is_strict_and_default_off(monkeypatch):
    from pycbc.fft import torchfft

    size = 131072
    batch = 32
    with scheme.TorchScheme("cuda"):
        invec = zeros(batch * size, dtype=np.complex64)
        outvec = zeros(batch * size, dtype=np.complex64)

        monkeypatch.setenv("PYCBC_TORCH_DIRECT_BATCH_IFFT", "0")
        monkeypatch.delenv(PROMOTED_ROWS_GATE, raising=False)
        fftobj = torchfft.IFFT(invec, outvec, nbatch=batch, size=size)
        assert fftobj._promoted_batch_plan is not None
        assert fftobj._promoted_batch_plan.rows == 16

        monkeypatch.setenv(PROMOTED_ROWS_GATE, "sometimes")
        with pytest.raises(ValueError, match=PROMOTED_ROWS_GATE):
            torchfft.IFFT(invec, outvec, nbatch=batch, size=size)

        monkeypatch.setenv(PROMOTED_ROWS_GATE, "1")
        fftobj32 = torchfft.IFFT(invec, outvec, nbatch=batch, size=size)
        assert fftobj32._promoted_batch_plan is not None
        assert fftobj32._promoted_batch_plan.rows == 32


def test_cuda_promoted_rows_fft_execution_and_accuracy(monkeypatch):
    from pycbc.fft import torchfft

    size = 131072
    batch = 32
    monkeypatch.setenv("PYCBC_TORCH_DIRECT_BATCH_IFFT", "0")
    monkeypatch.setenv(PROMOTED_ROWS_GATE, "1")

    with scheme.TorchScheme("cuda"):
        values = _complex_values(batch, size, seed=7901)
        invec = zeros(batch * size, dtype=np.complex64)
        outvec = zeros(batch * size, dtype=np.complex64)
        invec._data.tensor.copy_(torch.from_numpy(values.reshape(-1)))

        ifftobj = torchfft.IFFT(invec, outvec, nbatch=batch, size=size)
        assert ifftobj._promoted_batch_plan.rows == 32
        ifftobj.execute()

        # Compare with complex128 torch.fft.ifft reference
        torch_ref = (
            torch.fft.ifft(
                torch.from_numpy(values).to(torch.complex128),
                n=size,
                dim=-1,
                norm="forward",
            )
            .to(torch.complex64)
            .numpy()
        )

        actual = outvec.numpy().reshape(batch, size)
        np.testing.assert_allclose(actual, torch_ref, rtol=1e-5, atol=1e-5)
