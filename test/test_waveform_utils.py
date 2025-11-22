import unittest
import numpy

from utils import simple_exit, parse_args_all_schemes

from pycbc.waveform.utils import apply_fd_time_shift, fd_to_td
from pycbc.types import TimeSeries


_scheme, _context = parse_args_all_schemes("Waveform Utils")


class TestFDTimeShift(unittest.TestCase):
    """Tests waveform utility helpers across schemes."""
    def setUp(self):
        self.scheme = _scheme
        self.context = _context
        # Build a clean sinusoid with integer cycles to avoid edge effects
        self.freq = 128
        self.sample_rate = 4096
        self.seglen = 1
        ncycles = self.freq * self.seglen
        t = numpy.linspace(0, ncycles*2*numpy.pi,
                           num=self.sample_rate*self.seglen,
                           endpoint=False)
        with self.context:
            self.time_series = TimeSeries(t, delta_t=1./self.sample_rate, epoch=0)
            tdsinx_arr = numpy.sin(t)
            self.tdsinx = TimeSeries(tdsinx_arr, delta_t=1./self.sample_rate, epoch=0)
            self.fdsinx = self.tdsinx.to_frequencyseries()

    def _shift_and_ifft(self, fdsinx, tshift, fseries=None):
        start_time = self.time_series.start_time
        tdshift = apply_fd_time_shift(fdsinx, start_time+tshift,
                                      fseries=fseries)
        return tdshift.to_timeseries()

    def _test_apply_fd_time_shift(self, fdsinx, fseries=None, atol=1e-8):
        # shift by -pi/2: should be cosine
        tshift = 1./(4*self.freq)
        tdshift = self._shift_and_ifft(fdsinx, -tshift, fseries=fseries)
        comp = numpy.cos(self.time_series.numpy())
        if tdshift.precision == 'single':
            comp = comp.astype(numpy.float32)
        self.assertTrue(numpy.isclose(tdshift, comp, atol=atol).all())

        # shift by +pi/2: should be -cosine
        tdshift = self._shift_and_ifft(fdsinx, tshift, fseries=fseries)
        self.assertTrue(numpy.isclose(tdshift.numpy(), -1*comp, atol=atol).all())

        # shift by an arbitrary fraction of period
        tshift = 193 * self.time_series.delta_t / 3.
        tdshift = self._shift_and_ifft(fdsinx, tshift, fseries=fseries)
        comp = numpy.sin(self.time_series.numpy() - 2*numpy.pi*self.freq*tshift)
        if tdshift.precision == 'single':
            comp = comp.astype(numpy.float32)
        self.assertTrue(numpy.isclose(tdshift, comp, atol=atol).all())

        # backward
        tdshift = self._shift_and_ifft(fdsinx, -tshift, fseries=fseries)
        comp = numpy.sin(self.time_series.numpy() + 2*numpy.pi*self.freq*tshift)
        if tdshift.precision == 'single':
            comp = comp.astype(numpy.float32)
        self.assertTrue(numpy.isclose(tdshift.numpy(), comp, atol=atol).all())

    def test_fd_time_shift(self):
        with self.context:
            self._test_apply_fd_time_shift(self.fdsinx)

    def test_fd_time_shift32(self):
        with self.context:
            self._test_apply_fd_time_shift(self.fdsinx.astype(numpy.complex64),
                                           atol=1e-4)

    def test_fseries_time_shift(self):
        if self.scheme != 'cpu':
            self.skipTest("Non-uniform fseries path tested only under CPU scheme")
        fdsinx = self.fdsinx.copy()
        fseries = self.fdsinx.sample_frequencies.numpy()
        with self.context:
            self._test_apply_fd_time_shift(fdsinx, fseries)

    def test_fd_to_td_roundtrip(self):
        """Convert FD back to TD and compare with original sine."""
        with self.context:
            td_round = fd_to_td(self.fdsinx, delta_t=self.time_series.delta_t)
        comp = self.tdsinx[:len(td_round)]
        self.assertTrue(numpy.allclose(td_round, comp, atol=1e-6))


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestFDTimeShift))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
