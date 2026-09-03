# Copyright (C) 2025  The PyCBC team
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Minimal regression test for fd_decompress under all schemes.
Compares torch output against CPU reference for inline_linear interpolation.
"""

import unittest
import numpy

from pycbc.waveform.compress import fd_decompress
from pycbc.scheme import CPUScheme
from utils import parse_args_all_schemes, simple_exit

_scheme, _context = parse_args_all_schemes("Decompress")


class TestDecompress(unittest.TestCase):
    def setUp(self):
        self.scheme = _scheme
        self.context = _context
        # Simple monotonic sample grid
        self.sample_f = numpy.linspace(0.0, 8.0, 17, dtype=numpy.float32)
        self.amp = numpy.exp(-0.1 * self.sample_f).astype(numpy.float32)
        self.phase = (0.3 * self.sample_f).astype(numpy.float32)
        self.df = 0.25
        self.f_lower = 0.5
        self.interps = [
            "inline_linear",
            "inline_quadratic",
            "inline_cubic",
            "inline_quartic",
        ]
        self.refs = {}
        with CPUScheme():
            for interp in self.interps:
                self.refs[interp] = fd_decompress(self.amp, self.phase,
                                                  self.sample_f,
                                                  df=self.df,
                                                  f_lower=self.f_lower,
                                                  interpolation=interp)

    def test_fd_decompress_interpolations(self):
        for interp in self.interps:
            with self.context:
                test = fd_decompress(self.amp, self.phase, self.sample_f,
                                     df=self.df, f_lower=self.f_lower,
                                     interpolation=interp)
            ref_np = numpy.array(self.refs[interp])
            test_np = numpy.array(test)
            self.assertEqual(len(test_np), len(ref_np))
            self.assertTrue(numpy.allclose(test_np, ref_np,
                                           rtol=1e-3, atol=5e-3))


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestDecompress))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
