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

"""Regression tests for frequency-domain waveform decompression."""

import unittest
import numpy

import pycbc
from pycbc.scheme import CPUScheme, TorchScheme
from pycbc.types import FrequencySeries, zeros
from pycbc.types.backend import backend_array
from pycbc.waveform.compress import fd_decompress
from utils import parse_args_all_schemes, simple_exit


try:
    import torch
except ImportError:
    torch = None


_scheme, _context = parse_args_all_schemes("Decompress")
_HAVE_TORCH = torch is not None and pycbc.HAVE_TORCH
_HAVE_TORCH_CUDA = _HAVE_TORCH and torch.cuda.is_available()


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
                self.refs[interp] = fd_decompress(
                    self.amp,
                    self.phase,
                    self.sample_f,
                    df=self.df,
                    f_lower=self.f_lower,
                    interpolation=interp,
                )

    def test_fd_decompress_interpolations(self):
        for interp in self.interps:
            with self.context:
                test = fd_decompress(
                    self.amp,
                    self.phase,
                    self.sample_f,
                    df=self.df,
                    f_lower=self.f_lower,
                    interpolation=interp,
                )
            ref_np = numpy.array(self.refs[interp])
            test_np = numpy.array(test)
            self.assertEqual(len(test_np), len(ref_np))
            self.assertTrue(
                numpy.allclose(test_np, ref_np, rtol=1e-3, atol=5e-3)
            )


@unittest.skipUnless(_HAVE_TORCH, "PyTorch is unavailable")
class TestTorchDecompress(unittest.TestCase):
    """Compare each native Torch interpolation route with the CPU backend."""

    def setUp(self):
        self.sample_f = numpy.array(
            [0.0, 0.43, 1.17, 2.05, 3.4, 4.8, 6.35, 8.0]
        )
        self.df = 0.2
        self.f_lower = 0.71
        self.epoch = 1234567890.125
        self.interps = (
            "inline_linear",
            "inline_quadratic",
            "inline_cubic",
            "inline_quartic",
        )

    def _exercise_device(self, device):
        for dtype in (numpy.float32, numpy.float64):
            sample_f = self.sample_f.astype(dtype)
            amp = (
                numpy.exp(-0.08 * sample_f)
                * (1.0 + 0.03 * numpy.cos(0.7 * sample_f))
            ).astype(dtype)
            phase = (0.13 * sample_f + 0.011 * sample_f ** 2).astype(dtype)
            complex_dtype = (
                numpy.complex64 if dtype is numpy.float32
                else numpy.complex128
            )
            torch_dtype = (
                torch.complex64 if dtype is numpy.float32
                else torch.complex128
            )
            tolerance = (
                dict(rtol=5e-5, atol=5e-6) if dtype is numpy.float32
                else dict(rtol=5e-12, atol=5e-13)
            )

            references = {}
            with CPUScheme():
                for interp in self.interps:
                    automatic = fd_decompress(
                        amp,
                        phase,
                        sample_f,
                        df=self.df,
                        f_lower=self.f_lower,
                        interpolation=interp,
                    )
                    output_length = len(automatic) + 7
                    output = FrequencySeries(
                        numpy.full(
                            output_length,
                            17.0 - 4.0j,
                            dtype=complex_dtype,
                        ),
                        delta_f=self.df,
                        epoch=self.epoch,
                        copy=False,
                    )
                    fd_decompress(
                        amp,
                        phase,
                        sample_f,
                        out=output,
                        f_lower=self.f_lower,
                        interpolation=interp,
                    )
                    references[interp] = (
                        numpy.array(automatic),
                        numpy.array(output),
                    )

            with TorchScheme(device):
                for interp in self.interps:
                    automatic = fd_decompress(
                        amp,
                        phase,
                        sample_f,
                        df=self.df,
                        f_lower=self.f_lower,
                        interpolation=interp,
                    )
                    automatic_storage = backend_array(automatic, "torch")
                    self.assertIsNotNone(automatic_storage)
                    self.assertEqual(automatic_storage.device.type, device)
                    self.assertEqual(automatic_storage.dtype, torch_dtype)
                    self.assertEqual(automatic.delta_f, self.df)
                    self.assertEqual(float(automatic.epoch), 0.0)

                    output = FrequencySeries(
                        zeros(
                            len(references[interp][1]),
                            dtype=complex_dtype,
                        ),
                        delta_f=self.df,
                        epoch=self.epoch,
                        copy=False,
                    )
                    backend_array(output, "torch").fill_(17.0 - 4.0j)
                    returned = fd_decompress(
                        amp,
                        phase,
                        sample_f,
                        out=output,
                        f_lower=self.f_lower,
                        interpolation=interp,
                    )
                    self.assertIs(returned, output)
                    self.assertEqual(output.delta_f, self.df)
                    self.assertEqual(float(output.epoch), self.epoch)

                    numpy.testing.assert_allclose(
                        numpy.array(automatic),
                        references[interp][0],
                        **tolerance,
                    )
                    numpy.testing.assert_allclose(
                        numpy.array(output),
                        references[interp][1],
                        **tolerance,
                    )
                    start_index = int(numpy.ceil(self.f_lower / self.df))
                    last_index = int(numpy.floor(sample_f[-1] / self.df))
                    output_values = numpy.array(output)
                    self.assertTrue(
                        numpy.all(output_values[:start_index] == 0)
                    )
                    self.assertTrue(
                        numpy.all(output_values[last_index + 1:] == 0)
                    )

    def test_torch_cpu(self):
        self._exercise_device("cpu")

    @unittest.skipUnless(_HAVE_TORCH_CUDA, "Torch CUDA is unavailable")
    def test_torch_cuda(self):
        self._exercise_device("cuda")


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestDecompress))
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestTorchDecompress))

if __name__ == "__main__":
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
