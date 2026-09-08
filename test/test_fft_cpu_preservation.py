"""Preserve the existing CPU validation contract when adding Torch FFTs."""
import pytest

from pycbc import scheme
from pycbc.fft.core import _check_fwd_args


class _View:
    ptr = 1

    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


@pytest.mark.parametrize("size", [4, 5, 6, 7])
@pytest.mark.parametrize("nbatch", [1, 2])
def test_cpu_inplace_real_fft_keeps_original_lengths(size, nbatch):
    expected = int(2 * (size / 2 + 1))
    with scheme.CPUScheme():
        _check_fwd_args(_View(nbatch * expected), "real",
                        _View(nbatch * int(size / 2 + 1)), "complex",
                        nbatch, size)
        if size % 2:
            with pytest.raises(ValueError, match="in-place FFT"):
                _check_fwd_args(_View(nbatch * (expected - 1)), "real",
                                _View(nbatch * int(size / 2 + 1)), "complex",
                                nbatch, size)
