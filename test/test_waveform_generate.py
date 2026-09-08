import unittest

from utils import parse_args_all_schemes, simple_exit
from pycbc.waveform.waveform import get_fd_waveform


_scheme, _context = parse_args_all_schemes("Waveform Generation")


class TestWaveformGeneration(unittest.TestCase):
    def test_fd_waveform_torch_device(self):
        params = {
            "approximant": "TaylorF2",
            "mass1": 10,
            "mass2": 10,
            "delta_f": 0.25,
            "f_lower": 30,
        }
        with _context:
            hp, hc = get_fd_waveform(**params)

        if _scheme == "torch":
            self.assertTrue(hasattr(hp._data, "tensor"))
            self.assertEqual(hp._data.tensor.device, _context.device)
            self.assertEqual(hc._data.tensor.device, _context.device)
        else:
            self.assertFalse(hasattr(hp._data, "tensor"))
            self.assertFalse(hasattr(hc._data, "tensor"))


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestWaveformGeneration))


if __name__ == "__main__":
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
