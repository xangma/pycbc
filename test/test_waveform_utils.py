import unittest
import numpy

from utils import simple_exit, parse_args_all_schemes

from pycbc import cosmology
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.waveform.utils import (
    apply_fd_time_shift,
    fd_to_td,
    redshift_waveform,
)
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



class TestRedshiftWaveform(unittest.TestCase):
    """Tests ``redshift_waveform`` against detector-frame generation.
    
    Specifically, this tests that a waveform generated in the source frame and
    then redshifted using ``redshift_waveform`` matches a waveform generated
    directly in the detector frame with redshifted masses. This is done for
    both TD and FD waveforms.
    """

    def setUp(self):
        self.srcm1 = 30.0
        self.srcm2 = 20.0
        self.distance = 4000.
        self.z = cosmology.redshift(self.distance)

        # Detector-frame settings.
        self.flow = 30.0
        self.sample_rate = 4096.0
        self.seglen = 64.0

        # Source-frame settings.
        self.srcflow = self.flow * (1 + self.z)
        self.srcsr = self.sample_rate * (1 + self.z)
        self.srcseglen = self.seglen / (1 + self.z)

        self.detm1 = self.srcm1 * (1 + self.z)
        self.detm2 = self.srcm2 * (1 + self.z)

    def _relative_l2_error(self, test, ref):
        """Returns relative L2 norm of ``test - ref``."""
        return numpy.linalg.norm(test - ref) / numpy.linalg.norm(ref)


    def _check_epochs(self, redshifted_hp, det_hp, err=0.01):
        """Checks that the epochs of the two waveforms are close enough.

        Small differences in the epochs may arise due to floating point
        errors. That may cause a failure when we compute the relative L2
        error, so we'll check that the epochs are close enough.
        """
        isclose = numpy.isclose(redshifted_hp.start_time, det_hp.start_time,
                                      rtol=0., atol=err*det_hp.delta_t)
        self.assertTrue(isclose,
                        msg=f"Epochs differ by more than {err*det_hp.delta_t}:"
                            f" |redshifted - detector epoch| = "
                            f"{abs(redshifted_hp.start_time - det_hp.start_time)}")

    def test_td_redshift_matches_redshifted_masses(self):
        """Redshifting a source-frame TD waveform matches detector-frame TD."""
        src_hp, _ = get_td_waveform(
            approximant='SEOBNRv4',
            mass1=self.srcm1,
            mass2=self.srcm2,
            distance=self.distance,
            delta_t=1.0 / self.srcsr,
            f_lower=self.srcflow,
        )
        redshifted_hp = redshift_waveform(src_hp, self.z)

        det_hp, _ = get_td_waveform(
            approximant='SEOBNRv4',
            mass1=self.detm1,
            mass2=self.detm2,
            distance=self.distance,
            delta_t=1.0 / self.sample_rate,
            f_lower=self.flow,
        )
        self._check_epochs(redshifted_hp, det_hp)
        # if passed, set the redshifted_hp epoch to be the same as the det_hp
        # epoch, so that we can compare the waveforms directly
        redshifted_hp.start_time = det_hp.start_time
        relerr = self._relative_l2_error(redshifted_hp, det_hp)
        self.assertLess(relerr, 2e-3)


    def test_fd_redshift_matches_redshifted_masses(self):
        """Redshifting a source-frame FD waveform matches detector-frame FD."""
        src_hptilde, _ = get_fd_waveform(
            approximant='IMRPhenomXPHM',
            mass1=self.srcm1,
            mass2=self.srcm2,
            distance=self.distance,
            delta_f=1.0 / self.srcseglen,
            f_lower=self.srcflow,
            f_final=self.srcsr / 2.0,
        )
        redshifted_hptilde = redshift_waveform(src_hptilde, self.z)

        det_hptilde, _ = get_fd_waveform(
            approximant='IMRPhenomXPHM',
            mass1=self.detm1,
            mass2=self.detm2,
            distance=self.distance,
            delta_f=1.0 / self.seglen,
            f_lower=self.flow,
            f_final=self.sample_rate / 2.0,
        )

        redshifted_hp = redshifted_hptilde.to_timeseries()
        det_hp = det_hptilde.to_timeseries()
        self._check_epochs(redshifted_hp, det_hp)
        # if passed, set the redshifted_hp epoch to be the same as the det_hp
        # epoch, so that we can compare the waveforms directly
        redshifted_hp.start_time = det_hp.start_time
        relerr = self._relative_l2_error(redshifted_hp, det_hp)
        self.assertLess(relerr, 2e-3)


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestFDTimeShift))
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestRedshiftWaveform))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
