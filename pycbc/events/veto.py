"""This module contains utilities to manipulate trigger lists based on
segment.
"""

import logging

import numpy

from pycbc.types.backend import (
    backend_array,
    wrap_backend_array,
)

try:
    from igwn_segments import segment, segmentlist
except ImportError:
    segment = segmentlist = None

from pycbc.types import Array

logger = logging.getLogger("pycbc.events.veto")


def _ligolw_modules(feature=None):
    """Load LIGO-LW table support for XML-backed operations."""
    from igwn_ligolw import ligolw, lsctables
    from igwn_ligolw import utils as ligolw_utils

    return ligolw, lsctables, ligolw_utils


def _torch_veto_tensor(value):
    """Return existing Torch storage without importing the optional backend."""
    return backend_array(value, "torch")


def _wrap_torch_veto_result(inputs, tensor):
    """Preserve PyCBC Array inputs without copying indices to the host."""
    if not any(isinstance(value, Array) for value in inputs):
        return tensor

    return Array(wrap_backend_array(tensor), copy=False)


def _torch_veto_vectors(times, start, end):
    """Move mixed veto vectors beside their first Torch-backed input."""
    values = (times, start, end)
    tensors = [_torch_veto_tensor(value) for value in values]
    reference = next((value for value in tensors if value is not None), None)
    if reference is None:
        return None

    import torch

    result = []
    for value, tensor in zip(values, tensors, strict=True):
        if tensor is None:
            tensor = torch.as_tensor(
                value, dtype=reference.dtype, device=reference.device
            )
        if tensor.device != reference.device:
            raise ValueError("veto time arrays must use one device")
        if tensor.dtype != reference.dtype:
            raise TypeError("veto time arrays must use one dtype")
        if tensor.ndim != 1 or tensor.dtype == torch.bool or tensor.is_complex():
            raise TypeError("veto time arrays must be one-dimensional numeric values")
        result.append(tensor)

    if result[1].shape != result[2].shape:
        raise ValueError("veto segment start and end arrays must match")
    return tuple(result)


def _torch_coalesced_segments(start, end):
    """Return sorted, coalesced half-open segments on a Torch device."""
    import torch

    lo = torch.minimum(start, end)
    hi = torch.maximum(start, end)
    nonempty = lo != hi
    lo = lo[nonempty]
    hi = hi[nonempty]
    if lo.numel() == 0:
        return lo, hi

    order = torch.argsort(lo)
    lo = lo[order]
    hi = hi[order]
    running_end = torch.cummax(hi, dim=0).values
    new_segment = torch.cat(
        (
            torch.ones(1, dtype=torch.bool, device=lo.device),
            lo[1:] > running_end[:-1],
        )
    )
    last_in_segment = torch.cat(
        (
            new_segment[1:],
            torch.ones(1, dtype=torch.bool, device=lo.device),
        )
    )
    return lo[new_segment], running_end[last_in_segment]


def _torch_indices_within_times(times, start, end):
    """Torch implementation backing :func:`indices_within_times`."""
    tensors = _torch_veto_vectors(times, start, end)
    if tensors is None:
        return None

    import torch

    times_t, start_t, end_t = tensors
    start_t, end_t = _torch_coalesced_segments(start_t, end_t)
    if times_t.numel() == 0 or start_t.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=times_t.device)

    order = torch.argsort(times_t)
    sorted_times = times_t[order]
    segment_id = torch.searchsorted(start_t, sorted_times, right=True) - 1
    safe_segment_id = segment_id.clamp_min(0)
    within = (segment_id >= 0) & (sorted_times < end_t[safe_segment_id])
    return order[within]


def _torch_indices_outside_times(times, start, end):
    """Torch implementation backing :func:`indices_outside_times`."""
    exclude = _torch_indices_within_times(times, start, end)
    if exclude is None:
        return None

    import torch

    keep = torch.ones(len(times), dtype=torch.bool, device=exclude.device)
    keep[exclude] = False
    return torch.arange(len(times), dtype=torch.int64, device=exclude.device)[keep]


def _torch_complement_indices(times, exclude):
    """Return the device-resident complement of an index vector."""
    import torch

    times_t = _torch_veto_tensor(times)
    exclude_t = _torch_veto_tensor(exclude)
    reference = times_t if times_t is not None else exclude_t
    if reference is None:
        return None
    if times_t is not None and times_t.device != reference.device:
        raise ValueError("veto time arrays must use one device")
    if exclude_t is None:
        exclude_t = torch.as_tensor(exclude, dtype=torch.int64, device=reference.device)
    elif exclude_t.device != reference.device:
        raise ValueError("veto index arrays must use one device")

    keep = torch.ones(len(times), dtype=torch.bool, device=reference.device)
    keep[exclude_t.to(dtype=torch.int64)] = False
    return torch.arange(len(times), dtype=torch.int64, device=reference.device)[keep]


def start_end_to_segments(start, end):
    return segmentlist([segment(s, e) for s, e in zip(start, end)])


def segments_to_start_end(segs):
    segs.coalesce()
    return (numpy.array([s[0] for s in segs]), numpy.array([s[1] for s in segs]))


def start_end_from_segments(segment_file):
    """
    Return the start and end time arrays from a segment file.

    Parameters
    ----------
    segment_file: xml segment file

    Returns
    -------
    start: numpy.ndarray
    end: numpy.ndarray
    """
    _, lsctables, ligolw_utils = _ligolw_modules("reading LIGO-LW segment files")
    from pycbc.io.ligolw import LIGOLWContentHandler as h

    indoc = ligolw_utils.load_filename(segment_file, False, contenthandler=h)
    segment_table = lsctables.SegmentTable.get_table(indoc)
    start = numpy.array(segment_table.getColumnByName("start_time"))
    start_ns = numpy.array(segment_table.getColumnByName("start_time_ns"))
    end = numpy.array(segment_table.getColumnByName("end_time"))
    end_ns = numpy.array(segment_table.getColumnByName("end_time_ns"))
    return start + start_ns * 1e-9, end + end_ns * 1e-9


def indices_within_times(times, start, end):
    """
    Return an index array into times that lie within the durations defined by start end arrays

    Parameters
    ----------
    times: numpy.ndarray
        Array of times
    start: numpy.ndarray
        Array of duration start times
    end: numpy.ndarray
        Array of duration end times

    Returns
    -------
    indices: numpy.ndarray
        Array of indices into times
    """
    torch_indices = _torch_indices_within_times(times, start, end)
    if torch_indices is not None:
        return _wrap_torch_veto_result((times, start, end), torch_indices)

    # coalesce the start/end segments
    start, end = segments_to_start_end(start_end_to_segments(start, end).coalesce())

    tsort = times.argsort()
    times_sorted = times[tsort]
    left = numpy.searchsorted(times_sorted, start)
    right = numpy.searchsorted(times_sorted, end)

    if len(left) == 0:
        return numpy.array([], dtype=numpy.uint32)

    return tsort[numpy.hstack([numpy.r_[s:e] for s, e in zip(left, right)])]


def indices_outside_times(times, start, end):
    """
    Return an index array into times that like outside the durations defined by start end arrays

    Parameters
    ----------
    times: numpy.ndarray
        Array of times
    start: numpy.ndarray
        Array of duration start times
    end: numpy.ndarray
        Array of duration end times

    Returns
    -------
    indices: numpy.ndarray
        Array of indices into times
    """
    torch_indices = _torch_indices_outside_times(times, start, end)
    if torch_indices is not None:
        return _wrap_torch_veto_result((times, start, end), torch_indices)

    exclude = indices_within_times(times, start, end)
    indices = numpy.arange(0, len(times))
    return numpy.delete(indices, exclude)


def select_segments_by_definer(segment_file, segment_name=None, ifo=None):
    """Return the list of segments that match the segment name

    Parameters
    ----------
    segment_file: str
        path to segment xml file

    segment_name: str
        Name of segment
    ifo: str, optional

    Returns
    -------
    seg: list of segments
    """
    ligolw, _, ligolw_utils = _ligolw_modules("selecting LIGO-LW segment definitions")
    from pycbc.io.ligolw import LIGOLWContentHandler as h

    indoc = ligolw_utils.load_filename(segment_file, False, contenthandler=h)
    segment_table = ligolw.Table.get_table(indoc, "segment")

    seg_def_table = ligolw.Table.get_table(indoc, "segment_definer")
    def_ifos = seg_def_table.getColumnByName("ifos")
    def_names = seg_def_table.getColumnByName("name")
    def_ids = seg_def_table.getColumnByName("segment_def_id")

    valid_id = []
    for def_ifo, def_name, def_id in zip(def_ifos, def_names, def_ids):
        if ifo and ifo != def_ifo:
            continue
        if segment_name and segment_name != def_name:
            continue
        valid_id += [def_id]

    start = numpy.array(segment_table.getColumnByName("start_time"))
    start_ns = numpy.array(segment_table.getColumnByName("start_time_ns"))
    end = numpy.array(segment_table.getColumnByName("end_time"))
    end_ns = numpy.array(segment_table.getColumnByName("end_time_ns"))
    start, end = start + 1e-9 * start_ns, end + 1e-9 * end_ns
    did = segment_table.getColumnByName("segment_def_id")

    keep = numpy.array([d in valid_id for d in did])
    if sum(keep) > 0:
        return start_end_to_segments(start[keep], end[keep])
    else:
        return segmentlist([])


def indices_within_segments(times, segment_files, ifo=None, segment_name=None):
    """Return the list of indices that should be vetoed by the segments in the
    list of veto_files.

    Parameters
    ----------
    times: numpy.ndarray of integer type
        Array of gps start times
    segment_files: string or list of strings
        A string or list of strings that contain the path to xml files that
        contain a segment table
    ifo: string, optional
        The ifo to retrieve segments for from the segment files
    segment_name: str, optional
        name of segment
    Returns
    -------
    indices: numpy.ndarray
        The array of index values within the segments
    segmentlist:
        The segment list corresponding to the selected time.
    """
    veto_segs = segmentlist([])
    for veto_file in segment_files:
        veto_segs += select_segments_by_definer(veto_file, segment_name, ifo)
    veto_segs.coalesce()

    start, end = segments_to_start_end(veto_segs)
    idx = indices_within_times(times, start, end)
    tensor = _torch_veto_tensor(idx)
    if tensor is not None:
        indices = _wrap_torch_veto_result((times, idx), tensor.sort().values)
    else:
        empty = numpy.array([], dtype=numpy.uint32)
        indices = numpy.union1d(empty, idx)

    return indices, veto_segs.coalesce()


def indices_outside_segments(times, segment_files, ifo=None, segment_name=None):
    """Return the list of indices that are outside the segments in the
    list of segment files.

    Parameters
    ----------
    times: numpy.ndarray of integer type
        Array of gps start times
    segment_files: string or list of strings
        A string or list of strings that contain the path to xml files that
        contain a segment table
    ifo: string, optional
        The ifo to retrieve segments for from the segment files
    segment_name: str, optional
        name of segment
    Returns
    --------
    indices: numpy.ndarray
        The array of index values outside the segments
    segmentlist:
        The segment list corresponding to the selected time.
    """
    exclude, segs = indices_within_segments(
        times, segment_files, ifo=ifo, segment_name=segment_name
    )
    torch_indices = _torch_complement_indices(times, exclude)
    if torch_indices is not None:
        indices = _wrap_torch_veto_result((times, exclude), torch_indices)
    else:
        indices = numpy.delete(numpy.arange(0, len(times)), exclude)
    return indices, segs


def get_segment_definer_comments(xml_file, include_version=True):
    """Returns a dict with the comment column as the value for each segment"""

    _, lsctables, ligolw_utils = _ligolw_modules("reading LIGO-LW segment definitions")
    from pycbc.io.ligolw import LIGOLWContentHandler as h

    # read segment definer table
    xmldoc = ligolw_utils.load_fileobj(xml_file, compress="auto", contenthandler=h)
    seg_def_table = lsctables.SegmentDefTable.get_table(xmldoc)

    # put comment column into a dict
    comment_dict = {}
    for seg_def in seg_def_table:
        if include_version:
            full_channel_name = ":".join(
                [str(seg_def.ifos), str(seg_def.name), str(seg_def.version)]
            )
        else:
            full_channel_name = ":".join([str(seg_def.ifos), str(seg_def.name)])

        comment_dict[full_channel_name] = seg_def.comment

    return comment_dict
