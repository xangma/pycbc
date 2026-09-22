"""Keep CPU chi-square time coordinates independent of input dtype."""

import unittest

import numpy as np

from pycbc.types import Array
from pycbc.vetoes.chisq_cpu import shift_sum


class TestCPUChisqShifts(unittest.TestCase):
    def test_integer_shift_above_float32_boundary(self):
        # The large transform length makes these valid in-segment sample
        # indices. Only two frequencies are visited, so no long recurrence is
        # involved. A broadcast view avoids allocating a large array; the
        # kernel reads only the first two elements, both equal to i.
        length = 2**25
        correlation = np.broadcast_to(np.array([1j], dtype=np.complex64),
                                      (length,))
        shifts = np.array([2**24 - 1, 2**24, 2**24 + 1], dtype=np.int64)

        result = shift_sum(Array(correlation, copy=False), shifts, [0, 2])

        # |i + i exp(2 pi i t / N)|^2, evaluated without subtracting nearly
        # equal cosines. The central sample cancels, but its neighbours do not.
        expected = 4 * np.sin(np.pi * (shifts - length // 2) / length)**2
        # Allow the independent truncated-pi error still present in this
        # kernel (about 0.63% here), while detecting the lost integer index.
        np.testing.assert_allclose(result, expected, rtol=0.01, atol=1e-18)
        self.assertEqual(result.dtype, np.dtype(np.float32))
        self.assertGreater(result[2], 1e4 * result[1])

    def test_fractional_and_ordinary_shifts_preserve_output_dtype(self):
        shifts = np.array([0, 0.25, 1.5, 9, 15.75], dtype=np.float64)
        bins = np.array([0, 2, 7, 16])
        samples = np.arange(16)
        values = (samples % 5 - 2) + 1j * (samples % 3 - 1)

        for dtype in (np.complex64, np.complex128):
            with self.subTest(dtype=dtype):
                correlation = values.astype(dtype)
                result = shift_sum(Array(correlation, copy=False), shifts,
                                   bins)
                expected = np.zeros(len(shifts))
                for start, stop in zip(bins[:-1], bins[1:]):
                    frequencies = np.arange(start, stop)
                    phases = np.exp(2j * np.pi *
                                    shifts[:, None] * frequencies / 16)
                    sums = phases @ correlation[start:stop].astype(
                        np.complex128)
                    expected += abs(sums)**2

                # The existing recurrence and pi approximation are unchanged.
                tolerance = 3e-6 if dtype == np.complex64 else 2e-8
                np.testing.assert_allclose(result, expected, rtol=tolerance)
                expected_dtype = np.float32 if dtype == np.complex64 else \
                    np.float64
                self.assertEqual(result.dtype, np.dtype(expected_dtype))


if __name__ == '__main__':
    unittest.main()
