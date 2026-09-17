"""
These unit tests are for the pycbc.strain.strain module
"""
import numpy
from pycbc.types import TimeSeries
from pycbc.strain.strain import (
    execute_cached_fft,
    execute_cached_ifft,
)
import unittest

from utils import simple_exit


class TestStrain(unittest.TestCase):

    def setUp(self):

        self.rng = numpy.random.default_rng()
        self.td_data = TimeSeries(
            self.rng.normal(size=100), delta_t=0.2, epoch=1123456789.6,
        )
        self.fd_data = self.td_data.to_frequencyseries()
        # Tolerance for float64
        self.tol = 1e-14

    def test_cached_fft(self):
        fd_data = execute_cached_fft(
            self.td_data,
            uid=87651,
            copy_output=True,
        )
        self.assertTrue(
            fd_data.almost_equal_norm(
                self.fd_data, tol=self.tol, dtol=self.tol
            )
        )

    def test_cached_ifft(self):
        td_data = execute_cached_ifft(
            self.fd_data,
            uid=87652,
            copy_output=True,
        )
        self.assertTrue(
            td_data.almost_equal_norm(
                self.td_data, tol=self.tol, dtol=self.tol
            )
        )


class TestOverwhitenSegments(unittest.TestCase):

    def _create_segments(self, num_samples=32768, delta_t=1.0/512):
        rng = numpy.random.default_rng(42)
        values = rng.normal(size=num_samples).astype(numpy.float32)
        strain = TimeSeries(values, delta_t=delta_t, epoch=1000000000)
        from pycbc.strain.strain import StrainSegments
        segments_obj = StrainSegments(
            strain,
            segment_length=16,
            segment_start_pad=2,
            segment_end_pad=2,
            trigger_start=1000000000,
            trigger_end=1000000064,
            allow_zero_padding=True,
        )
        return segments_obj

    def test_overwhiten_shared_buffer_single_psd_cpu(self):
        from pycbc.strain.strain import overwhiten_segments
        from pycbc.types import FrequencySeries
        from pycbc import scheme
        with scheme.CPUScheme(1):
            segments_obj = self._create_segments()
            segs = segments_obj.fourier_segments()
            self.assertIsNotNone(segments_obj.fourier_buffer)
            self.assertIs(segs[0]._shared_fourier_buffer, segments_obj.fourier_buffer)

            freq_len = len(segs[0])
            psd = FrequencySeries(
                numpy.full(freq_len, 2.5, dtype=numpy.float32),
                delta_f=segs[0].delta_f,
            )
            for s in segs:
                s.psd = psd

            orig_values = [s.numpy().copy() for s in segs]
            overwhiten_segments(segs)

            for s, orig in zip(segs, orig_values):
                numpy.testing.assert_allclose(s.numpy(), orig / 2.5, rtol=1e-6)

    def test_overwhiten_shared_buffer_multi_psd_cpu(self):
        from pycbc.strain.strain import overwhiten_segments
        from pycbc.types import FrequencySeries
        from pycbc import scheme
        with scheme.CPUScheme(1):
            segments_obj = self._create_segments()
            segs = segments_obj.fourier_segments()

            freq_len = len(segs[0])
            psd1 = FrequencySeries(
                numpy.full(freq_len, 2.0, dtype=numpy.float32),
                delta_f=segs[0].delta_f,
            )
            psd2 = FrequencySeries(
                numpy.full(freq_len, 5.0, dtype=numpy.float32),
                delta_f=segs[0].delta_f,
            )
            for i, s in enumerate(segs):
                s.psd = psd1 if i % 2 == 0 else psd2

            orig_values = [s.numpy().copy() for s in segs]
            overwhiten_segments(segs)

            for i, (s, orig) in enumerate(zip(segs, orig_values)):
                divisor = 2.0 if i % 2 == 0 else 5.0
                numpy.testing.assert_allclose(s.numpy(), orig / divisor, rtol=1e-6)

    def test_overwhiten_shared_buffer_torch(self):
        import pycbc
        if not getattr(pycbc, "HAVE_TORCH", False):
            self.skipTest("PyTorch not available")
        from pycbc.types import FrequencySeries
        from pycbc import scheme
        with scheme.TorchScheme("cpu", num_threads=1):
            segments_obj = self._create_segments()
            segs = segments_obj.fourier_segments()
            self.assertIsNotNone(segments_obj.fourier_buffer)
            self.assertIs(segs[0]._shared_fourier_buffer, segments_obj.fourier_buffer)

            freq_len = len(segs[0])
            psd = FrequencySeries(
                numpy.full(freq_len, 3.0, dtype=numpy.float32),
                delta_f=segs[0].delta_f,
            )
            for s in segs:
                s.psd = psd

            orig_values = [s.numpy().copy() for s in segs]
            segments_obj.overwhiten()

            for s, orig in zip(segs, orig_values):
                numpy.testing.assert_allclose(s.numpy(), orig / 3.0, rtol=1e-6)

    def test_overwhiten_unshared_fallback(self):
        from pycbc.strain.strain import overwhiten_segments
        from pycbc.types import FrequencySeries
        from pycbc import scheme
        with scheme.CPUScheme(1):
            s1 = FrequencySeries(numpy.array([2.0, 4.0, 6.0], dtype=numpy.complex64), delta_f=1.0)
            s2 = FrequencySeries(numpy.array([10.0, 20.0, 30.0], dtype=numpy.complex64), delta_f=1.0)
            psd = FrequencySeries(numpy.array([2.0, 2.0, 2.0], dtype=numpy.float32), delta_f=1.0)
            s1.psd = psd
            s2.psd = psd
            segs = [s1, s2]
            overwhiten_segments(segs)
            numpy.testing.assert_allclose(s1.numpy(), [1.0, 2.0, 3.0])
            numpy.testing.assert_allclose(s2.numpy(), [5.0, 10.0, 15.0])


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestStrain))
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestOverwhitenSegments))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
