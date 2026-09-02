"""
These unit tests are for the pycbc.strain.strain module
"""
import numpy
import pycbc
import pycbc.scheme as scheme
from pycbc.types import TimeSeries
from pycbc.strain.strain import (
    create_memory_and_engine_for_class_based_fft,
    execute_cached_fft,
    execute_cached_ifft,
)
import unittest
from unittest import mock

if pycbc.HAVE_TORCH:
    from pycbc.types.array_torch import TorchArrayData

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

    @unittest.skipUnless(pycbc.HAVE_TORCH, "PyTorch is unavailable")
    def test_cached_fft_does_not_cross_torch_scheme(self):
        if scheme.current_prefix() != "cpu":
            self.skipTest("cross-scheme regression requires the CPU default")

        uid = 982451653
        create_memory_and_engine_for_class_based_fft.cache_clear()
        expected = execute_cached_fft(self.td_data, uid=uid)
        raw_data = self.td_data.numpy().copy()

        def reject_host_transfer(_self):
            raise AssertionError("cached FFT copied a Torch buffer to the host")

        torch_context = scheme.TorchScheme("cpu")
        try:
            with mock.patch.object(
                TorchArrayData, "numpy", reject_host_transfer
            ):
                with torch_context:
                    torch_data = TimeSeries(
                        raw_data,
                        delta_t=self.td_data.delta_t,
                        epoch=self.td_data.start_time,
                    )
                    actual = execute_cached_fft(torch_data, uid=uid)
                    self.assertIsInstance(actual._data, TorchArrayData)

                cpu_again = execute_cached_fft(
                    TimeSeries(
                        raw_data,
                        delta_t=self.td_data.delta_t,
                        epoch=self.td_data.start_time,
                    ),
                    uid=uid,
                )
                self.assertIsInstance(cpu_again._data, numpy.ndarray)
        finally:
            del torch_context
            scheme.Scheme._single = None
            create_memory_and_engine_for_class_based_fft.cache_clear()

        numpy.testing.assert_allclose(actual.numpy(), expected.numpy())
        numpy.testing.assert_allclose(cpu_again.numpy(), expected.numpy())


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestStrain))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
