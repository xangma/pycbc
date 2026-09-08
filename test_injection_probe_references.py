"""Analytic checks independent of the synthetic capture generator."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('probe', Path(__file__).with_name('injection-probe.py'))
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
np = probe.np


class ReferenceTests(unittest.TestCase):
    def test_residual_chisq_and_fft_subtraction_are_distinct(self):
        correlation = np.zeros(16, dtype=np.complex128)
        correlation[1:5] = [1, 2, 3, 4]
        bins = {'original': np.array([1, 3, 5]), 'proposed': np.array([1, 2, 5])}
        refs = probe.reference_statistics(correlation, correlation.astype(np.complex64),
                                          np.complex64(11), bins, 0, 2)
        # z=(3,7), total=10. Scientific chi=64; supplying FFT SNR 11
        # changes its subtraction by -84, giving a negative literal result.
        self.assertAlmostEqual(refs['original']['chisq'], 64)
        self.assertAlmostEqual(refs['original']['evaluator_chisq'], -20)
        self.assertAlmostEqual(refs['original']['native_fft_subtraction_discrepancy'], -84)
        self.assertAlmostEqual(refs['proposed']['chisq'], 256)

    def test_newsnr_uses_physical_dof_and_only_penalizes_above_one(self):
        self.assertEqual(probe.new_snr(8, -2, 16), 8)
        self.assertEqual(probe.new_snr(8, 30, 16), 8)
        self.assertAlmostEqual(probe.new_snr(8, 60, 16), 8*(.5*(1+2**3))**(-1/6))

    def test_target_calibration_does_not_use_captured_norm(self):
        size, target, df = 8192, 7801, .125
        h = np.ones(size//2+1, dtype=np.complex64)*(2+1j)
        psd = np.full(len(h), 4, dtype=np.float32)
        q, _ = probe.calibration(h, psd, 240, size//2, df)
        injected = probe.injected_input(h, None, 12, q, target, 240, size//2)
        corr = probe.reference_correlation(h, injected, psd, 240, size//2)
        raw = probe.fft.ifft(corr, norm='forward')[target]
        independent_norm = float(4*df/np.sqrt(q))
        self.assertAlmostEqual(abs(raw)*independent_norm, 12, delta=1e-6)
        deliberately_different_norm = independent_norm*1.02
        self.assertAlmostEqual(abs(raw)*deliberately_different_norm, 12.24, delta=1e-6)

    def test_calibration_promotes_before_squaring(self):
        h = np.full(17, np.complex64(1e25+1e25j))
        psd = np.full(17, 1e30, dtype=np.float32)
        q, _ = probe.calibration(h, psd, 1, 16, .5)
        value = complex(h[1])
        expected = 4*.5*15*(value.real**2+value.imag**2)/float(psd[1])
        self.assertTrue(np.isfinite(q))
        self.assertLess(abs(float(q)/expected-1), 1e-14)

    def test_zero_signal_and_single_rounding_preserve_noise(self):
        rng = np.random.default_rng(63)
        h = (rng.normal(size=513)+1j*rng.normal(size=513)).astype(np.complex64)
        noise = h.copy()
        before = noise.copy()
        np.testing.assert_array_equal(probe.injected_input(h, noise, 0, np.longdouble(3),
                                                          975, 16, 512), noise)
        value = probe.injected_input(h, noise, 8, np.longdouble(3), 975, 16, 512)
        manual = noise.astype(np.complex128)
        k = np.arange(16, 512, dtype=np.int64)
        manual[16:512] += 8/np.sqrt(3)*h[16:512].astype(np.complex128)*np.exp(-2j*np.pi*((k*975)%1024)/1024)
        np.testing.assert_array_equal(value, manual.astype(np.complex64))
        np.testing.assert_array_equal(noise, before)


if __name__ == '__main__':
    unittest.main()
