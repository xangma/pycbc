# Copyright (C) 2012  Andrew Miller, Alex Nitz, Josh Willis
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
'''
These are the unittests for the pycbc frame/cache reading functions
'''


import unittest
from unittest import mock
from pathlib import Path
import tempfile
import lal
import lalframe
import numpy
from pycbc.io import get_file

import pycbc
import pycbc.frame
from pycbc.frame import frame
from pycbc.types import TimeSeries
from utils import parse_args_cpu_only, simple_exit

# Frame tests only need to happen on the CPU
parse_args_cpu_only("Frame I/O")

class FrameTestBase(unittest.TestCase):
    __test__ = False
    def setUp(self):
        numpy.random.seed(1023)
        self.size = pow(2,12)

        self.data1 = numpy.array(numpy.random.rand(self.size), dtype=self.dtype)
        self.data2 = numpy.array(numpy.random.rand(self.size), dtype=self.dtype)

        # If the dtype is complex, we should throw in some complex values as well
        if self.dtype == numpy.complex64 or self.dtype == numpy.complex128:
            self.data1 += numpy.random.rand(self.size) * 1j
            self.data2 += numpy.random.rand(self.size) * 1j

        self.delta_t = .5
        self.epoch = 123456.0
        self.expected_data1 = TimeSeries(self.data1,dtype=self.dtype,
                                         epoch=self.epoch,delta_t=self.delta_t)
        self.expected_data2 = TimeSeries(self.data2,dtype=self.dtype,
                                         epoch=self.epoch,delta_t=self.delta_t)

    def _write_test_frames(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        middle = self.size // 2
        files = []
        for number, section in enumerate((slice(None, middle),
                                          slice(middle, None))):
            path = root / f'frame-{number}.gwf'
            pycbc.frame.write_frame(
                str(path), ['channel1', 'channel2'],
                [self.expected_data1[section], self.expected_data2[section]])
            files.append(str(path))
        cache = root / 'frames.cache'
        frame_duration = int(middle * self.delta_t)
        cache.write_text(''.join(
            f'H TEST {int(self.epoch) + i * frame_duration} '
            f'{frame_duration} file://localhost{path}\n'
            for i, path in enumerate(files)))
        return root, files, str(cache)

    def test_explicit_bounds_skip_duration_metadata(self):
        root, files, cache = self._write_test_frames()
        middle = self.size // 2
        across_frames = slice(middle - 3, middle + 7)
        cases = ((files[0], 'channel1', slice(3, 11)),
                 (list(reversed(files)), ['channel1', 'channel2'],
                  across_frames),
                 (str(root / '*.gwf'), ['channel1', 'channel2'],
                  across_frames),
                 (cache, ['channel2', 'channel1'], across_frames))
        no_metadata = mock.Mock(side_effect=AssertionError(
            'explicit bounds must not query duration metadata'))
        real_get_type = lalframe.FrStreamGetTimeSeriesType
        real_set_mode = lalframe.FrStreamSetMode
        type_map = {key: [*value[:2], no_metadata, no_metadata, *value[4:]]
                    for key, value in frame._fr_type_map.items()}
        for location, channels, section in cases:
            start = lal.LIGOTimeGPS(self.epoch + section.start * self.delta_t)
            end = lal.LIGOTimeGPS(self.epoch + section.stop * self.delta_t)
            with self.subTest(location=location, channels=channels), \
                    mock.patch.object(lalframe, 'FrStreamGetVectorLength',
                                      no_metadata), \
                    mock.patch.dict(frame._fr_type_map, type_map), \
                    mock.patch.object(lalframe, 'FrStreamGetTimeSeriesType',
                                      wraps=real_get_type) as get_type, \
                    mock.patch.object(lalframe, 'FrStreamSetMode',
                                      wraps=real_set_mode) as mode, \
                    mock.patch.object(lal, 'CacheSieve',
                                      wraps=lal.CacheSieve) as sieve:
                result = pycbc.frame.read_frame(
                    location, channels, start_time=start, end_time=end,
                    check_integrity=True, sieve=r'/frame-')
                names = channels if type(channels) is list else [channels]
                results = result if type(channels) is list else [result]
                self.assertEqual(get_type.call_count, len(names))
                self.assertTrue(mode.call_args.args[1] &
                                lalframe.FR_STREAM_CHECKSUM_MODE)
                self.assertEqual(sieve.call_args_list[0].args[-1], r'/frame-')
                self.assertEqual(sieve.call_count, 2)
                for name, actual in zip(names, results):
                    expected = (self.expected_data1 if name == 'channel1'
                                else self.expected_data2)[section]
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertEqual(actual.start_time, expected.start_time)
                    self.assertEqual(actual.delta_t, expected.delta_t)

    def test_omitted_bounds_keep_duration_metadata(self):
        _, files, _ = self._write_test_frames()
        middle = self.size // 2
        cases = (({}, slice(0, middle)),
                 ({'start_time': self.epoch}, slice(0, middle)),
                 ({'end_time': self.epoch + 4}, slice(0, 8)),
                 ({'duration': 4}, slice(0, 8)),
                 ({'start_time': self.epoch + 1, 'duration': 4}, slice(2, 10)))
        for kwargs, section in cases:
            with self.subTest(kwargs=kwargs), mock.patch.object(
                    lalframe, 'FrStreamGetVectorLength',
                    wraps=lalframe.FrStreamGetVectorLength) as metadata:
                actual = pycbc.frame.read_frame(files[0], 'channel1', **kwargs)
                metadata.assert_called_once()
                expected = self.expected_data1[section]
                self.assertEqual(actual, expected)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.start_time, expected.start_time)
                self.assertEqual(actual.delta_t, expected.delta_t)

    def test_bounded_read_validation(self):
        _, files, _ = self._write_test_frames()
        for kwargs in ({'end_time': self.epoch},
                       {'end_time': self.epoch - 1},
                       {'end_time': self.epoch + 1, 'duration': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pycbc.frame.read_frame(files[0], 'channel1',
                                      start_time=self.epoch, **kwargs)
        with self.assertRaises(IndexError):
            pycbc.frame.read_frame(files[0], [], start_time=self.epoch,
                                  end_time=self.epoch + 1)
        with self.assertRaises(RuntimeError):
            pycbc.frame.read_frame(files[0], 'missing', start_time=self.epoch,
                                  end_time=self.epoch + 1)

    def test_frame(self):
        # TODO also test reading a cache

        # This is a file in the temp directory that will be deleted when it is garbage collected
        url = 'https://github.com/gwastro/pycbc-config/raw/master/'
        url += 'test_data_files/frametest{}.gwf'
        filename = get_file(url.format(self.data1.dtype), cache=True)

        # Make sure we can run from one directory higher as well
        import os.path
        if not os.path.exists(filename):
            filename =  "test/" + filename

        # Now we will create a frame file, specifiying that it is a timeseries
        #Fr.frputvect(filename,
        #             [{'name':'channel1', 'data':self.data1, 'start':int(self.epoch),
        #               'dx':self.delta_t,'type':1},
        #              {'name':'channel2', 'data':self.data2, 'start':int(self.epoch),
        #               'dx':self.delta_t,'type':1}])

        # Reading just one channel first
        ts1 = pycbc.frame.read_frame(filename, 'channel1')
        # Checking all values
        self.assertEqual(ts1,self.expected_data1)
        # Now checking the start time
        self.assertEqual(ts1.start_time, self.epoch)
        # And the duration
        self.assertEqual(ts1.end_time - ts1.start_time,self.size * self.delta_t)

        # Now reading multiple channels
        ts2 = pycbc.frame.read_frame(filename, ['channel1','channel2'])
        # We should get back a list
        self.assertTrue(type(ts2) is list)
        self.assertEqual(ts2[0],self.expected_data1)
        self.assertEqual(ts2[1],self.expected_data2)
        self.assertEqual(ts2[0].start_time, self.epoch)
        self.assertEqual(ts2[1].start_time, self.epoch)
        self.assertEqual(ts2[0].end_time - ts2[0].start_time,self.size * self.delta_t)
        self.assertEqual(ts2[1].end_time - ts2[1].start_time,self.size * self.delta_t)

        # These are the times and indices for the segment we will try to read
        start = self.epoch+10
        end = self.epoch+50
        startind = int(10/self.delta_t)
        endind = int(50/self.delta_t)

        # Now reading in a specific segment with an integer
        ts3 = pycbc.frame.read_frame(filename, 'channel1',
                                                start_time=int(start),
                                                end_time=int(end))

        # The same, but with a LIGOTimeGPS for the start and end times
        ts4 = pycbc.frame.read_frame(filename, 'channel1',
                                                start_time=start,
                                                 end_time=end)

        # Now we will check those two TimeSeries
        self.assertEqual(ts3,self.expected_data1[startind:endind])
        self.assertEqual(ts4,self.expected_data1[startind:endind])
        self.assertTrue(40 - (float(ts3.end_time)-float(ts3.start_time)) < self.delta_t)
        self.assertEqual(ts3.start_time, start)
        self.assertTrue(40 - (float(ts4.end_time)-float(ts4.start_time)) < self.delta_t)
        self.assertEqual(ts4.start_time, start)

        # And now some cases that should raise errors

        # There must be a span grater than 0
        self.assertRaises(ValueError, pycbc.frame.read_frame, filename,
                          'channel1', start_time=self.epoch,
                          end_time=self.epoch)
        # The start must be before the end
        self.assertRaises(ValueError, pycbc.frame.read_frame, filename,
                          'channel1', start_time=self.epoch+1,
                          end_time=self.epoch)

# We take a factory approach so we can test all possible dtypes we support
TestClasses = []
types = [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128]

for ty in types:
    klass = type('{0}_Test'.format(ty.__name__),(FrameTestBase,),{'dtype': ty})
    klass.__test__ = True
    vars()[klass.__name__] = klass
    TestClasses.append(klass)
    del klass

if __name__ == '__main__':
    suite = unittest.TestSuite()
    for klass in TestClasses:
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(klass))
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
