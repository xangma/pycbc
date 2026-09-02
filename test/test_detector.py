# Copyright (C) 2018 Alex Nitz
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""
These are the unittests for the pycbc.detector module
"""

import pycbc.detector as det
import unittest, numpy
from numpy.random import uniform, seed
seed(0)

# We require lal as a reference comparison
import lal

from utils import simple_exit

class TestDetector(unittest.TestCase):
    def setUp(self):
        self.d = [det.Detector(ifo)
                  for ifo in det.get_available_detectors()]

        # not distributed sanely, but should provide some good coverage
        N = 1000
        self.ra = uniform(0, numpy.pi * 2, size=N)
        self.dec = uniform(-numpy.pi, numpy.pi, size=N)
        self.pol = uniform(0, numpy.pi * 2, size=N)
        self.time = uniform(1126000000.0, 1336096017.0, size=N)

    def test_light_time(self):
        for d1 in self.d:
            for d2 in self.d:
                t1 = lal.LightTravelTime(d1.lal(), d2.lal()) * 1e-9
                t2 = d1.light_travel_time_to_detector(d2)
                self.assertAlmostEqual(t1, t2, 7)

    def test_custom_detector(self):
        det.add_detector_on_earth("TEST", 1.3, 0.5, yangle=0, xaltitude=0.01)
        d = det.Detector("TEST")

        # Check that we can call the new detector response
        fp, fc = d.antenna_pattern(1.5, 1.0, 0, 1000000000)

        # Check it interacts with existing detectors
        d2 = det.Detector("H1")
        t1 = d2.light_travel_time_to_detector(d)

    def test_response_matrix(self):
        import lal
        cached = {d.frDetector.prefix: d for d in lal.CachedDetectors}
        for ifo in ['H1', 'L1', 'V1', 'K1', 'I1']:
            ref_resp = cached[ifo].response
            resp = det.Detector(ifo).response
            self.assertAlmostEqual((ref_resp - resp).max(), 0, places=6)

    def test_antenna_pattern(self):
        vals = list(zip(self.ra, self.dec, self.pol, self.time))
        for ifo in self.d:
            fp = []
            fc = []
            for ra1, dec1, pol1, time1 in vals:
                gmst = lal.GreenwichMeanSiderealTime(time1)
                fp1, fc1 = tuple(lal.ComputeDetAMResponse(ifo.response, ra1, dec1, pol1, gmst))
                fp.append(fp1)
                fc.append(fc1)

            fp2, fc2 = ifo.antenna_pattern(self.ra, self.dec, self.pol, self.time)

            fp = numpy.array(fp)
            fc = numpy.array(fc)

            diff1 = fp - fp2
            diff2 = fc - fc2
            diff = abs(numpy.concatenate([diff1, diff2]))
            tolerance = 2e-4
            print("Max antenna diff:", ifo.name, diff.max())

            self.assertLess(diff.max(), tolerance)

    def test_delay_from_detector(self):
        ra, dec, time = self.ra[0:10], self.dec[0:10], self.time[0:10]
        for d1 in self.d:
            for d2 in self.d:
                time1 = []
                for ra1, dec1, tim1 in zip(ra, dec, time):
                    t1 = lal.ArrivalTimeDiff(d1.location, d2.location,
                                             ra1, dec1, tim1)
                    time1.append(t1)
                time1 = numpy.array(time1)
                time2 = d1.time_delay_from_detector(d2, ra, dec, time)
                self.assertLess(abs(time1 - time2).max(), 1e-3)

    def test_optimal_orientation(self):
        for d1 in self.d:
            ra, dec = d1.optimal_orientation(self.time[0])
            ra1 = d1.longitude + lal.GreenwichMeanSiderealTime(self.time[0]) % (numpy.pi *2)
            dec1 = d1.latitude

            self.assertAlmostEqual(ra, ra1, 3)
            self.assertAlmostEqual(dec, dec1, 7)
            
    def test_det_tc_conversion(self):
        """Test that the convert_tc method functions properly. The same times
        should be returned in all frames regardless of the reference.
        """
        vals = list(zip(self.ra, self.dec, self.time))
        ref_frames = ['geocentric', 'H1', 'L1', 'V1']
        target_frames = ['H1', 'L1', 'V1']
        # convert the times from geocentric using time_delay_from_earth_center
        test_times = {'geocentric': self.time}
        for ifo in target_frames:
            d = det.Detector(ifo)
            det_times = []
            for ra1, dec1, time1 in vals:
                tc = time1 + d.time_delay_from_earth_center(ra1, dec1, time1)
                det_times.append(tc)
            test_times[ifo] = det_times
        # convert the nominal times to each of the other detectors
        for target_ifo in target_frames:
            # set up the target detector
            d = det.Detector(target_ifo)
            target_times = test_times[target_ifo]
            for ref_ifo in ref_frames:
                ref_times = test_times[ref_ifo]
                converted_times = []
                for i in range(len(vals)):
                    ra1, dec1, _ = vals[i]
                    ref_tc = ref_times[i]
                    # convert the reference time to the target detector
                    tc = d.arrival_time(ref_tc, ra1, dec1, ref_frame = ref_ifo)
                    converted_times.append(tc)
                # check that the times converted to target match nominal
                print(f"Testing conversion from {ref_ifo} to {target_ifo}")
                for i in range(len(converted_times)):
                    self.assertAlmostEqual(converted_times[i], target_times[i], 
                                           places=6)

    def test_one_at_a_time_matches_vector(self):
        """Calling one at a time must match calling with a vector.

        The response is applied to the whole set at once, so a mistake in
        lining the components up would show at some positions and not
        others. Covers the vector and scalar polarizations, and
        time_delay_from_earth_center given an array.

        The response is compared to the precision of the arithmetic rather
        than exactly: the matrix product rounds differently over a whole
        set than over one position, in the last bit of a double.
        """
        test_time = 1187008882.0
        polarizations = [{}, {'polarization_type': 'vector'},
                         {'polarization_type': 'scalar'}]
        for detector in self.d:
            delay_vector = detector.time_delay_from_earth_center(
                self.ra, self.dec, test_time)
            response_vectors = [
                detector.antenna_pattern(self.ra, self.dec, self.pol,
                                         test_time, **kwargs)
                for kwargs in polarizations]

            for index in (0, 137, len(self.ra) - 1):
                right_ascension = float(self.ra[index])
                declination = float(self.dec[index])
                polarization = float(self.pol[index])

                self.assertEqual(
                    delay_vector[index],
                    detector.time_delay_from_earth_center(
                        right_ascension, declination, test_time))

                for kwargs, from_vector in zip(polarizations,
                                               response_vectors):
                    one_at_a_time = detector.antenna_pattern(
                        right_ascension, declination, polarization,
                        test_time, **kwargs)
                    for whole, single in zip(from_vector, one_at_a_time):
                        self.assertTrue(
                            numpy.isclose(whole[index], single,
                                          rtol=1e-14, atol=1e-16),
                            "%s at %s: %r against %r"
                            % (kwargs, index, whole[index], single))

    def test_antenna_pattern_and_time_delay_scalar(self):
        from pycbc.detector.ground import _scalar_antenna_pattern_and_time_delay
        for d in self.d:
            ra = 1.234
            dec = 0.567
            time = 1126259462.0
            # polarization = 0
            fp0, fc0, dt0 = _scalar_antenna_pattern_and_time_delay(
                d, ra, dec, time
            )
            fp, fc, dt = d.antenna_pattern_and_time_delay(ra, dec, 0.0, time)
            self.assertEqual(fp, fp0)
            self.assertEqual(fc, fc0)
            self.assertEqual(dt, dt0)

            # polarization != 0
            pol = 0.891
            cos2psi = numpy.cos(2.0 * pol)
            sin2psi = numpy.sin(2.0 * pol)
            expected_fp = cos2psi * fp0 + sin2psi * fc0
            expected_fc = -sin2psi * fp0 + cos2psi * fc0
            fp, fc, dt = d.antenna_pattern_and_time_delay(ra, dec, pol, time)
            self.assertEqual(fp, expected_fp)
            self.assertEqual(fc, expected_fc)
            self.assertEqual(dt, dt0)

    def test_antenna_pattern_and_time_delay_numpy(self):
        for d in self.d:
            # 1D array
            fp, fc, dt = d.antenna_pattern_and_time_delay(
                self.ra, self.dec, self.pol, self.time
            )
            fp_ref, fc_ref = d.antenna_pattern(
                self.ra, self.dec, self.pol, self.time
            )
            dt_ref = d.time_delay_from_earth_center(
                self.ra, self.dec, self.time
            )
            numpy.testing.assert_allclose(fp, fp_ref, rtol=1e-14, atol=1e-16)
            numpy.testing.assert_allclose(fc, fc_ref, rtol=1e-14, atol=1e-16)
            numpy.testing.assert_allclose(dt, dt_ref, rtol=1e-14, atol=1e-16)

            # 2D array and broadcasting
            ra2 = uniform(0, numpy.pi * 2, size=(5, 10))
            dec2 = uniform(-numpy.pi / 2, numpy.pi / 2, size=(5, 10))
            pol2 = uniform(0, numpy.pi * 2, size=(10,))
            t2 = 1126259462.0
            fp2, fc2, dt2 = d.antenna_pattern_and_time_delay(
                ra2, dec2, pol2, t2
            )
            self.assertEqual(fp2.shape, (5, 10))
            self.assertEqual(fc2.shape, (5, 10))
            self.assertEqual(dt2.shape, (5, 10))
            fp2_ref, fc2_ref = d.antenna_pattern(ra2, dec2, pol2, t2)
            dt2_ref = d.time_delay_from_earth_center(ra2, dec2, t2)
            numpy.testing.assert_allclose(fp2, fp2_ref, rtol=1e-14, atol=1e-16)
            numpy.testing.assert_allclose(fc2, fc2_ref, rtol=1e-14, atol=1e-16)
            numpy.testing.assert_allclose(dt2, dt2_ref, rtol=1e-14, atol=1e-16)

    def test_antenna_pattern_and_time_delay_torch(self):
        try:
            import torch
        except ImportError:
            return

        d = det.Detector('H1')
        # 1D tensors
        ra_t = torch.tensor(
            self.ra[:50], dtype=torch.float64, requires_grad=True
        )
        dec_t = torch.tensor(
            self.dec[:50], dtype=torch.float64, requires_grad=True
        )
        pol_t = torch.tensor(
            self.pol[:50], dtype=torch.float64, requires_grad=True
        )
        t_t = torch.tensor(
            self.time[:50], dtype=torch.float64, requires_grad=True
        )

        fp_t, fc_t, dt_t = d.antenna_pattern_and_time_delay(
            ra_t, dec_t, pol_t, t_t
        )
        fp_ref, fc_ref = d.antenna_pattern(ra_t, dec_t, pol_t, t_t)
        dt_ref = d.time_delay_from_earth_center(ra_t, dec_t, t_t)

        torch.testing.assert_close(fp_t, fp_ref)
        torch.testing.assert_close(fc_t, fc_ref)
        torch.testing.assert_close(dt_t, dt_ref)

        loss = (fp_t.square() + fc_t.square() + dt_t.square()).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(ra_t.grad).all())
        self.assertTrue(torch.isfinite(dec_t.grad).all())
        self.assertTrue(torch.isfinite(pol_t.grad).all())
        self.assertTrue(torch.isfinite(t_t.grad).all())

        # 2D tensors and broadcasting
        ra2_t = torch.tensor(self.ra[:20].reshape(4, 5), dtype=torch.float64)
        dec2_t = torch.tensor(self.dec[:20].reshape(4, 5), dtype=torch.float64)
        pol2_t = torch.tensor(0.4, dtype=torch.float64)
        t2_t = torch.tensor(1126259462.0, dtype=torch.float64)

        fp2_t, fc2_t, dt2_t = d.antenna_pattern_and_time_delay(
            ra2_t, dec2_t, pol2_t, t2_t
        )
        self.assertEqual(fp2_t.shape, torch.Size([4, 5]))
        self.assertEqual(fc2_t.shape, torch.Size([4, 5]))
        self.assertEqual(dt2_t.shape, torch.Size([4, 5]))
        fp2_ref, fc2_ref = d.antenna_pattern(ra2_t, dec2_t, pol2_t, t2_t)
        dt2_ref = d.time_delay_from_earth_center(ra2_t, dec2_t, t2_t)
        torch.testing.assert_close(fp2_t, fp2_ref)
        torch.testing.assert_close(fc2_t, fc2_ref)
        torch.testing.assert_close(dt2_t, dt2_ref)

    def test_network_geometry(self):
        ifos = ['H1', 'L1', 'V1', 'K1']
        net = det.NetworkGeometry(ifos)
        self.assertEqual(len(net), 4)
        self.assertEqual(net.detector_names, ifos)
        for name in ifos:
            self.assertEqual(net[name].name, name)

        # 1. Scalar test
        ra = 1.2
        dec = -0.4
        pol = 0.8
        t_gps = 1126259462.0

        fp_net, fc_net, dt_net = net.antenna_pattern_and_time_delay(
            ra, dec, pol, t_gps
        )
        for i, name in enumerate(ifos):
            d = net[name]
            fp_ref, fc_ref, dt_ref = d.antenna_pattern_and_time_delay(
                ra, dec, pol, t_gps
            )
            numpy.testing.assert_allclose(fp_net[i], fp_ref, rtol=1e-12)
            numpy.testing.assert_allclose(fc_net[i], fc_ref, rtol=1e-12)
            numpy.testing.assert_allclose(dt_net[i], dt_ref, rtol=1e-12)

        # to_dict helper
        d_dict = net.to_dict(fp_net)
        for i, name in enumerate(ifos):
            self.assertEqual(d_dict[name], fp_net[i])

        # 2. Vector test (1D array)
        ra_arr = self.ra[:50]
        dec_arr = self.dec[:50]
        pol_arr = self.pol[:50]
        time_arr = self.time[:50]

        fp_net, fc_net, dt_net = net.antenna_pattern_and_time_delay(
            ra_arr, dec_arr, pol_arr, time_arr
        )
        self.assertEqual(fp_net.shape, (4, 50))
        for i, name in enumerate(ifos):
            d = net[name]
            fp_ref, fc_ref, dt_ref = d.antenna_pattern_and_time_delay(
                ra_arr, dec_arr, pol_arr, time_arr
            )
            numpy.testing.assert_allclose(fp_net[i], fp_ref, rtol=1e-12)
            numpy.testing.assert_allclose(fc_net[i], fc_ref, rtol=1e-12)
            numpy.testing.assert_allclose(dt_net[i], dt_ref, rtol=1e-12)

        # 3. Torch test with gradients
        import torch
        ra_t = torch.tensor(ra_arr, dtype=torch.float64, requires_grad=True)
        dec_t = torch.tensor(dec_arr, dtype=torch.float64, requires_grad=True)
        pol_t = torch.tensor(pol_arr, dtype=torch.float64, requires_grad=True)
        t_t = torch.tensor(time_arr, dtype=torch.float64, requires_grad=True)

        fp_net_t, fc_net_t, dt_net_t = net.antenna_pattern_and_time_delay(
            ra_t, dec_t, pol_t, t_t
        )
        self.assertEqual(fp_net_t.shape, torch.Size([4, 50]))
        for i, name in enumerate(ifos):
            d = net[name]
            fp_ref_t, fc_ref_t, dt_ref_t = d.antenna_pattern_and_time_delay(
                ra_t, dec_t, pol_t, t_t
            )
            torch.testing.assert_close(fp_net_t[i], fp_ref_t)
            torch.testing.assert_close(fc_net_t[i], fc_ref_t)
            torch.testing.assert_close(dt_net_t[i], dt_ref_t)

        loss = (fp_net_t.square() + fc_net_t.square() + dt_net_t.square()).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(ra_t.grad).all())
        self.assertTrue(torch.isfinite(dec_t.grad).all())
        self.assertTrue(torch.isfinite(pol_t.grad).all())
        self.assertTrue(torch.isfinite(t_t.grad).all())


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestDetector))

if __name__ == '__main__':
    from astropy.utils import iers
    iers.conf.auto_download = False
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
