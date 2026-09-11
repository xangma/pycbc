# Copyright (C) 2026 The PyCBC Collaboration
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
"""Optimized frame stream reading routines for PyCBC Torch.

Bypasses redundant LALFrame metadata roundtrips when GPS time bounds
and channel types are known.
"""

import math
import lal
import lalframe
import numpy

from pycbc.frame.frame import _fr_type_map, locations_to_cache
from pycbc.types import TimeSeries


def _read_channel_direct(channel, stream, start, duration, dtype=None):
    """Read channel directly from stream without redundant type inspection."""
    if dtype is None or dtype in (
        numpy.float64,
        "float64",
        "double",
        lal.D_TYPE_CODE,
    ):
        try:
            data = lalframe.FrStreamReadREAL8TimeSeries(
                stream, channel, start, duration, 0
            )
            return TimeSeries(
                data.data.data,
                delta_t=data.deltaT,
                epoch=start,
                dtype=numpy.float64,
            )
        except RuntimeError:
            if dtype is not None and dtype not in (
                numpy.float64,
                "float64",
                "double",
                lal.D_TYPE_CODE,
            ):
                raise

    if dtype in (numpy.float32, "float32", "single", lal.S_TYPE_CODE):
        try:
            data = lalframe.FrStreamReadREAL4TimeSeries(
                stream, channel, start, duration, 0
            )
            return TimeSeries(
                data.data.data,
                delta_t=data.deltaT,
                epoch=start,
                dtype=numpy.float32,
            )
        except RuntimeError:
            pass

    # Generic fallback
    channel_type = lalframe.FrStreamGetTimeSeriesType(channel, stream)
    read_func = _fr_type_map[channel_type][0]
    d_type = _fr_type_map[channel_type][1]
    data = read_func(stream, channel, start, duration, 0)
    return TimeSeries(
        data.data.data, delta_t=data.deltaT, epoch=start, dtype=d_type
    )


def read_frame_torch(
    location,
    channels,
    start_time=None,
    end_time=None,
    duration=None,
    check_integrity=False,
    sieve=None,
    dtype=None,
):
    """Read time series from frame data with direct streaming."""
    if end_time and duration:
        raise ValueError("end time and duration are mutually exclusive")

    if isinstance(location, list):
        locations = location
    else:
        locations = [location]

    cum_cache = locations_to_cache(locations)
    if sieve:
        lal.CacheSieve(cum_cache, 0, 0, None, None, sieve)
    if start_time is not None and end_time is not None:
        if (int(math.ceil(end_time)) - int(start_time)) <= 0:
            raise ValueError("Negative or null duration")
        lal.CacheSieve(
            cum_cache,
            int(start_time),
            int(math.ceil(end_time)),
            None,
            None,
            None,
        )

    stream = lalframe.FrStreamCacheOpen(cum_cache)
    stream.mode = lalframe.FR_STREAM_VERBOSE_MODE
    if check_integrity:
        stream.mode = stream.mode | lalframe.FR_STREAM_CHECKSUM_MODE
    lalframe.FrStreamSetMode(stream, stream.mode)

    first_channel = channels[0] if isinstance(channels, list) else channels

    # Fallback to metadata discovery only if bounds were omitted
    if start_time is None or (end_time is None and duration is None):
        data_length = lalframe.FrStreamGetVectorLength(first_channel, stream)
        channel_type = lalframe.FrStreamGetTimeSeriesType(first_channel, stream)
        create_series_func = _fr_type_map[channel_type][2]
        get_series_metadata_func = _fr_type_map[channel_type][3]
        series = create_series_func(
            first_channel, stream.epoch, 0, 0, lal.ADCCountUnit, 0
        )
        get_series_metadata_func(series, stream)
        data_duration = (data_length + 0.5) * series.deltaT
        if start_time is None:
            start_time = stream.epoch * 1
        if end_time is None:
            end_time = start_time + data_duration

    if not isinstance(start_time, lal.LIGOTimeGPS):
        start_time = lal.LIGOTimeGPS(start_time)
    if end_time is not None and not isinstance(end_time, lal.LIGOTimeGPS):
        end_time = lal.LIGOTimeGPS(end_time)

    if duration is None:
        duration = float(end_time - start_time)
    else:
        duration = float(duration)

    if duration <= 0:
        raise ValueError("Negative or null duration")

    if isinstance(channels, list):
        all_data = []
        for channel in channels:
            channel_data = _read_channel_direct(
                channel, stream, start_time, duration, dtype=dtype
            )
            lalframe.FrStreamSeek(stream, start_time)
            all_data.append(channel_data)
        return all_data
    else:
        return _read_channel_direct(
            channels, stream, start_time, duration, dtype=dtype
        )
