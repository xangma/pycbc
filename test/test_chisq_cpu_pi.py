"""Known-answer CPU bin powers sensitive to the precision of pi."""

import unittest

import numpy

from pycbc.types import Array
from pycbc.vetoes.chisq_cpu import shift_sum


class TestCPUChisqPi(unittest.TestCase):
    """Use complex128 to separate pi truncation from single precision."""

    def test_adjacent_coefficients(self):
        # q(t) = 1 + exp(2*pi*i*t/4), so |q(t)|^2 = 2 + 2*cos(pi*t/2).
        # These exact values need neither an FFT nor a numerical phase model.
        corr = Array(numpy.array([1, 1, 0, 0], dtype=numpy.complex128))
        result = shift_sum(corr, [0, 0.5, 1], [0, 2])
        expected = [4, 2 + numpy.sqrt(2), 2]
        numpy.testing.assert_allclose(result, expected, rtol=0, atol=5e-14)
        self.assertEqual(result.dtype, numpy.dtype(numpy.float64))

    def test_wide_bin(self):
        # At t=N/4, each rotation is i. The last occupied frequency is
        # 3 modulo 4, giving q = 1 - i and power 2. A truncated pi creates
        # a systematic angular error that grows with the frequency index.
        n = 2**16
        corr = numpy.zeros(n, dtype=numpy.complex128)
        corr[0] = corr[n // 2 - 1] = 1
        result = shift_sum(Array(corr), [n // 4], [0, n // 2])
        numpy.testing.assert_allclose(result, [2], rtol=0, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
