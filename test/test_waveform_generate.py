import unittest
import numpy

from utils import parse_args_all_schemes, simple_exit
from pycbc.waveform.waveform import get_fd_waveform, get_td_waveform

_scheme, _context = parse_args_all_schemes("Waveform Generation")


class TestWaveformGeneration(unittest.TestCase):
    def setUp(self):
        self.scheme = _scheme
        self.context = _context

    def test_fd_waveform_torch_device(self):
        params = dict(approximant="TaylorF2", mass1=10, mass2=10,
                      delta_f=0.25, f_lower=30)
        with self.context:
            hp, hc = get_fd_waveform(**params)
        if self.scheme == 'torch':
            self.assertTrue(hasattr(hp._data, "tensor"))
            self.assertEqual(hp._data.tensor.device.type,
                             self.context.device.type if hasattr(self.context, "device") else hp._data.tensor.device.type)
        else:
            self.assertFalse(hasattr(hp._data, "tensor"))

    def test_td_waveform_torch_device(self):
        params = dict(approximant="TaylorT2", mass1=5, mass2=5,
                      delta_t=1/4096., f_lower=30)
        with self.context:
            hp, hc = get_td_waveform(**params)
        if self.scheme == 'torch':
            self.assertTrue(hasattr(hp._data, "tensor"))
            self.assertEqual(hp._data.tensor.device.type,
                             self.context.device.type if hasattr(self.context, "device") else hp._data.tensor.device.type)
        else:
            self.assertFalse(hasattr(hp._data, "tensor"))


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestWaveformGeneration))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
