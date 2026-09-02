# Copyright (C) 2022 Andrew Williamson
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
""" This module contains functions for calculating and manipulating coherent
triggers.
"""
import logging
import operator
import numpy as np

from .eventmgr_cython import get_coinc_indexes_cython_twodet_twocoinc
from pycbc.types import Array

logger = logging.getLogger('pycbc.events.coherent')


def _torch_coherent_tensors(*values, kinds="f", coerce_host=False):
    """Return compatible Torch tensors for coherent statistics."""
    if not values:
        return None

    from pycbc import scheme
    if scheme.current_prefix() != "torch":
        return None

    import torch
    from pycbc.types.array_torch import (
        TorchArrayData,
        _device_matches_active,
    )

    data = [
        value._data if isinstance(value, Array) else value
        for value in values
    ]
    torch_data = [
        value for value in data if isinstance(value, TorchArrayData)
    ]
    if not torch_data or (
        not coerce_host
        and len(torch_data) != len(data)
    ):
        return None

    first = torch_data[0].tensor
    if not (
        all(_device_matches_active(value.tensor) for value in torch_data)
        and all(value.tensor.device == first.device for value in torch_data)
        and all(value.tensor.shape == first.shape for value in torch_data)
        and all(value.tensor.dtype == first.dtype for value in torch_data)
        and all(value.dtype.kind in kinds for value in torch_data)
    ):
        return None

    tensors = []
    for value in data:
        if isinstance(value, TorchArrayData):
            tensors.append(value.tensor)
            continue
        if isinstance(value, Array):
            return None
        try:
            host = np.asarray(value)
            if host.dtype.kind not in kinds or host.shape != first.shape:
                return None
            tensors.append(
                torch.as_tensor(
                    host, dtype=first.dtype, device=first.device
                )
            )
        except (TypeError, ValueError, RuntimeError):
            return None

    return tuple(tensors)


def _torch_boolean_mask(mask, reference):
    """Return a boolean mask on the same Torch device as ``reference``."""
    import torch
    from pycbc.types.array_torch import TorchArrayData

    data = mask._data if isinstance(mask, Array) else mask
    if isinstance(data, TorchArrayData):
        tensor = data.tensor
        if (
            tensor.device == reference.device
            and tensor.shape == reference.shape
            and tensor.dtype == torch.bool
        ):
            return tensor
        return None

    try:
        tensor = torch.as_tensor(
            data, dtype=torch.bool, device=reference.device
        )
    except (TypeError, ValueError, RuntimeError):
        return None
    return tensor if tensor.shape == reference.shape else None


def _torch_array(tensor):
    """Wrap a Torch tensor as a PyCBC Array without a host copy."""
    from pycbc.types.array_torch import TorchArrayData

    return Array(TorchArrayData(tensor), copy=False)


def _torch_cache_tensors(cache, indices):
    """Return a device cache and compatible integer indices, when possible."""
    from pycbc import scheme

    if scheme.current_prefix() != "torch":
        return None

    import torch
    from pycbc.types.array_torch import (
        TorchArrayData,
        _device_matches_active,
    )

    cache_data = cache._data if isinstance(cache, Array) else cache
    if not isinstance(cache_data, TorchArrayData):
        return None
    cache_tensor = cache_data.tensor
    if not _device_matches_active(cache_tensor):
        return None

    index_data = indices._data if isinstance(indices, Array) else indices
    if isinstance(index_data, TorchArrayData):
        index_tensor = index_data.tensor
    elif isinstance(index_data, torch.Tensor):
        index_tensor = index_data
    else:
        try:
            host_indices = np.asarray(index_data)
        except (TypeError, ValueError):
            return None
        if host_indices.dtype.kind not in "iu" or host_indices.ndim != 1:
            return None
        index_tensor = torch.as_tensor(host_indices)

    valid_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    uint32_t = getattr(torch, "uint32", None)
    if uint32_t is not None:
        valid_dtypes.add(uint32_t)

    if index_tensor.dtype not in valid_dtypes or index_tensor.ndim != 1:
        return None
    return cache_tensor, index_tensor.to(
        device=cache_tensor.device, dtype=torch.int64
    )


def create_coherent_cache(reference, fill_value, dtype):
    """Allocate a coherent-trigger cache beside its reference array."""
    from pycbc import scheme

    data = reference._data if isinstance(reference, Array) else reference
    if scheme.current_prefix() == "torch":
        import torch
        from pycbc.types.array_torch import (
            TorchArrayData,
            _device_matches_active,
        )

        if (
            isinstance(data, TorchArrayData)
            and _device_matches_active(data.tensor)
        ):
            torch_dtype = torch.from_numpy(
                np.empty(0, dtype=np.dtype(dtype))
            ).dtype
            return _torch_array(
                torch.full(
                    (len(reference),),
                    fill_value,
                    dtype=torch_dtype,
                    device=data.tensor.device,
                )
            )

    return np.full(len(reference), fill_value, dtype=dtype)


def unavailable_coherent_indices(cache, indices):
    """Return cache indices whose floating-point values are still NaN."""
    tensors = _torch_cache_tensors(cache, indices)
    if tensors is not None:
        import torch

        cache_tensor, index_tensor = tensors
        if not cache_tensor.is_floating_point():
            raise TypeError("coherent cache availability requires real values")
        return _torch_array(
            index_tensor[torch.isnan(cache_tensor[index_tensor])]
        )

    if isinstance(cache, Array):
        raise TypeError("Torch coherent cache requires device indices")
    if isinstance(indices, Array):
        indices = indices.numpy()
    indices = np.asarray(indices)
    return indices[np.isnan(cache[indices])]


def update_coherent_cache(cache, indices, values):
    """Scatter newly calculated values into a coherent-trigger cache."""
    tensors = _torch_cache_tensors(cache, indices)
    if tensors is not None:
        import torch
        from pycbc.types.array_torch import TorchArrayData

        cache_tensor, index_tensor = tensors
        value_data = values._data if isinstance(values, Array) else values
        if isinstance(value_data, TorchArrayData):
            value_tensor = value_data.tensor.to(
                device=cache_tensor.device, dtype=cache_tensor.dtype
            )
        elif isinstance(value_data, torch.Tensor):
            value_tensor = value_data.to(
                device=cache_tensor.device, dtype=cache_tensor.dtype
            )
        else:
            value_tensor = torch.as_tensor(
                value_data,
                device=cache_tensor.device,
                dtype=cache_tensor.dtype,
            )
        cache_tensor[index_tensor] = value_tensor
        return

    if isinstance(cache, Array):
        raise TypeError("Torch coherent cache requires device indices")
    if isinstance(indices, Array):
        indices = indices.numpy()
    if isinstance(values, Array):
        values = values.numpy()
    cache[indices] = values


def _torch_selection_indices(mask):
    """Return ordered device indices selected by a Torch mask."""
    return mask.nonzero(as_tuple=False).flatten()


def _torch_select(value, device_indices, host_indices=None):
    """Select values with Torch indices, transferring only host positions.

    Device-backed values stay on their existing Torch device. Host values use
    one compact copy of the selected positions, which callers may reuse for
    any other host-resident companions.
    """
    import torch
    from pycbc.types.array_torch import TorchArrayData

    data = value._data if isinstance(value, Array) else value
    if isinstance(data, TorchArrayData):
        return _torch_array(data.tensor[device_indices]), host_indices
    if isinstance(data, torch.Tensor):
        return data[device_indices], host_indices

    if host_indices is None:
        host_indices = device_indices.detach().cpu().numpy()
    return value[host_indices], host_indices


def _torch_coincidence_indexes(idx_dict, time_delay_idx, min_nifos):
    """Build sorted coincidence indices on the active Torch device."""
    if not idx_dict:
        return None

    from pycbc import scheme
    if scheme.current_prefix() != "torch":
        return None

    import torch
    from pycbc.types.array_torch import (
        TorchArrayData,
        _device_matches_active,
    )

    tensors = []
    device = None
    try:
        for ifo, value in idx_dict.items():
            data = value._data if isinstance(value, Array) else value
            if not isinstance(data, TorchArrayData):
                return None
            tensor = data.tensor
            if (
                tensor.ndim != 1
                or data.dtype.kind not in "iu"
                or not _device_matches_active(tensor)
                or (device is not None and tensor.device != device)
            ):
                return None
            device = tensor.device
            delay = operator.index(time_delay_idx[ifo])
            if tensor.numel():
                tensors.append(tensor.to(dtype=torch.int64) - delay)
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None

    if device is None:
        return None
    if tensors:
        coincident = torch.sort(torch.cat(tensors)).values
    else:
        coincident = torch.empty(0, dtype=torch.int64, device=device)

    if not coincident.numel():
        return _torch_array(coincident)

    run_start = torch.ones_like(coincident, dtype=torch.bool)
    run_start[1:] = coincident[1:] != coincident[:-1]
    starts = torch.nonzero(run_start, as_tuple=False).flatten()
    unique = coincident[starts]
    if len(idx_dict) == 1:
        return _torch_array(unique)

    ends = torch.cat(
        (
            starts[1:],
            torch.tensor(
                [coincident.numel()], dtype=starts.dtype, device=device
            ),
        )
    )
    counts = ends - starts
    try:
        selected = counts > (min_nifos - 1)
    except (TypeError, ValueError, RuntimeError):
        return None
    return _torch_array(unique[selected])


def _torch_coincidence_triggers(snrs, idx, t_delay_idx):
    """Gather coincident SNR triggers with device-resident indices."""
    from pycbc import scheme
    if scheme.current_prefix() != "torch":
        return None

    import torch
    from pycbc.types.array_torch import (
        TorchArrayData,
        _device_matches_active,
    )

    idx_data = idx._data if isinstance(idx, Array) else idx
    if not isinstance(idx_data, TorchArrayData):
        return None
    idx_tensor = idx_data.tensor
    if (
        idx_tensor.ndim != 1
        or idx_data.dtype.kind not in "iu"
        or not _device_matches_active(idx_tensor)
    ):
        return None

    gathered = {}
    try:
        for ifo, values in snrs.items():
            data = values._data if isinstance(values, Array) else values
            if not isinstance(data, TorchArrayData):
                return None
            tensor = data.tensor
            if (
                tensor.ndim != 1
                or tensor.device != idx_tensor.device
                or not _device_matches_active(tensor)
                or not tensor.numel()
            ):
                return None
            delay = operator.index(t_delay_idx[ifo])
            indices = torch.remainder(
                idx_tensor.to(dtype=torch.int64) + delay,
                tensor.numel(),
            )
            gathered[ifo] = _torch_array(tensor[indices])
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None
    return gathered


def _torch_network_chisq(chisq, chisq_dof, snr_dict, ifos):
    """Return network chi-squared on Torch for device-resident SNRs."""
    if not ifos:
        return None

    from pycbc import scheme
    if scheme.current_prefix() != "torch":
        return None

    import torch
    from pycbc.types.array_torch import (
        TorchArrayData,
        _device_matches_active,
    )

    data = {
        ifo: (
            snr_dict[ifo]._data
            if isinstance(snr_dict[ifo], Array)
            else snr_dict[ifo]
        )
        for ifo in ifos
    }
    if not all(isinstance(value, TorchArrayData) for value in data.values()):
        return None

    tensors = {ifo: data[ifo].tensor for ifo in ifos}
    first = tensors[ifos[0]]
    if not (
        all(_device_matches_active(tensor) for tensor in tensors.values())
        and all(tensor.device == first.device for tensor in tensors.values())
        and all(tensor.shape == first.shape for tensor in tensors.values())
        and all(tensor.dtype == first.dtype for tensor in tensors.values())
        and all(data[ifo].dtype.kind in "fc" for ifo in ifos)
    ):
        return None

    def summary_tensor(value):
        summary = value._data if isinstance(value, Array) else value
        if isinstance(summary, TorchArrayData):
            tensor = summary.tensor
            if not (
                _device_matches_active(tensor)
                and tensor.device == first.device
                and summary.dtype.kind == "f"
            ):
                return None
            return tensor
        if isinstance(value, Array):
            return None
        try:
            host = np.asarray(value)
            if host.dtype.kind != "f":
                return None
            return torch.as_tensor(host, device=first.device)
        except (TypeError, ValueError, RuntimeError):
            return None

    try:
        snr2 = {
            ifo: (
                tensor.real.square() + tensor.imag.square()
                if tensor.is_complex()
                else tensor.square()
            )
            for ifo, tensor in tensors.items()
        }
        coinc_snr2 = sum(snr2.values())
        net_chisq = None
        for ifo in ifos:
            chisq_t = summary_tensor(chisq[ifo])
            dof_t = summary_tensor(chisq_dof[ifo])
            if chisq_t is None or dof_t is None:
                return None
            term = (chisq_t / dof_t) * (snr2[ifo] / coinc_snr2)
            if term.shape != first.shape:
                return None
            net_chisq = term if net_chisq is None else net_chisq + term
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None

    return Array(TorchArrayData(net_chisq), copy=False)


def get_coinc_indexes(idx_dict, time_delay_idx, min_nifos, wraparound_dict=None):
    """Return the indexes corresponding to coincident triggers. If only one
    detector is available in the network, the list of its unique indexes is
    simply returned.

    Parameters
    ----------
    idx_dict: dict
        Dictionary of indexes of triggers above threshold in each
        detector
    time_delay_idx: dict
        Dictionary giving time delay index (time_delay*sample_rate) for
        each detector
    min_nifos: int
        The minimum number of detectors needed to be above threshold
        for a coincidence to be produced
    wraparound_dict: dict
        The length at which indices (at the detector) must be wrapped around

    Returns
    -------
    coinc_idx: list
        List of indexes for triggers in geocent time that appear in
        multiple detectors
    """
    torch_indexes = _torch_coincidence_indexes(
        idx_dict, time_delay_idx, min_nifos
    )
    if torch_indexes is not None:
        return torch_indexes

    if (
        wraparound_dict is not None
        and min_nifos == 2
        and len(idx_dict) == 2
    ):
        ifos = list(idx_dict.keys())
        # Could cache an output array if needed
        idxarr1 = idx_dict[ifos[0]]
        idxarr2 = idx_dict[ifos[1]]
        # If either detector has no above-threshold triggers, there can be
        # no coincidences. Handle this explicitly to avoid passing a
        # zero-length output array into the Cython helper.
        if len(idxarr1) == 0 or len(idxarr2) == 0:
            return np.array([], dtype=idxarr1.dtype)
        outarr = np.zeros(max(len(idxarr1), len(idxarr2)), dtype=idxarr1.dtype)
        num_idxs = get_coinc_indexes_cython_twodet_twocoinc(
            idxarr1,
            idxarr2,
            time_delay_idx[ifos[0]],
            time_delay_idx[ifos[1]],
            wraparound_dict[ifos[0]],
            wraparound_dict[ifos[1]],
            outarr
        )
        return outarr[:num_idxs]
    coinc_list = np.array([], dtype=int)
    for ifo in idx_dict.keys():
        # Create list of indexes above single detector threshold, in geocent
        # time (-time_delay_idx[ifo] applies the time delay for the specific
        # detector). The periodic boundary condition of time slides is
        # enforced by wrapping around the index list of each detector. This
        # collective list will later be searched for repeating index values as
        # these represent triggers appearing in multiple detectors.
        if len(idx_dict[ifo]) != 0:
            delayed = idx_dict[ifo] - time_delay_idx[ifo]
            if wraparound_dict is not None and ifo in wraparound_dict:
                delayed = delayed % wraparound_dict[ifo]
            coinc_list = np.hstack([coinc_list, delayed])
    # Search through coinc_idx for repeated indexes. These must have been loud
    # in at least min_nifos detectors if the analysis uses more than 1
    # detector.
    counts = np.unique(coinc_list, return_counts=True)
    if len(idx_dict) == 1:
        return counts[0]
    coinc_idx = counts[0][counts[1] > min_nifos - 1]
    return coinc_idx


def get_coinc_triggers(snrs, idx, t_delay_idx):
    """Returns a dictionary, indexed by IFO, that collects the individual
    IFO SNRs of coincident triggers by using the indices of such triggers
    within the complete SNR timeseries of each IFO.

    Parameters
    ----------
    snrs: dict
        Dictionary of single detector SNR time series
    idx: list
        List of geocentric time indexes of coincident triggers
    t_delay_idx: dict
        Dictionary of indexes corresponding to light travel time from
        geocenter for each detector

    Returns
    -------
    coincs: dict
        Dictionary of coincident trigger SNRs in each detector
    """
    torch_triggers = _torch_coincidence_triggers(snrs, idx, t_delay_idx)
    if torch_triggers is not None:
        return torch_triggers

    # loops through snrs
    # %len(snrs[ifo]) was included as part of a wrap-around solution
    coincs = {
        ifo: snrs[ifo][(idx + t_delay_idx[ifo]) % len(snrs[ifo])]
        for ifo in snrs}
    return coincs


def coincident_snr(snr_dict, index, threshold, time_delay_idx):
    """Calculate the coincident SNR for all coincident triggers above
    threshold

    Parameters
    ----------
    snr_dict: dict
        Dictionary of individual detector SNRs
    index: list
        List of indexes (geocentric) for which to calculate coincident
        SNR
    threshold: float
        Coincident SNR threshold. Triggers below this are cut
    time_delay_idx: dict
        Dictionary of time delay from geocenter in indexes for each
        detector

    Returns
    -------
    rho_coinc: numpy.ndarray or Array
        Coincident SNR values for surviving triggers
    index: list
        The subset of input indexes corresponding to triggers that
        survive the cuts
    coinc_triggers: dict
        Dictionary of individual detector SNRs for triggers that
        survive cuts
    """
    # Restrict the snr timeseries to just the interesting points
    coinc_triggers = get_coinc_triggers(snr_dict, index, time_delay_idx)
    torch_tensors = _torch_coherent_tensors(
        *coinc_triggers.values(), kinds="fc"
    )
    if torch_tensors is not None:
        import torch

        snr_tensor = torch.stack(torch_tensors)
        if snr_tensor.is_complex():
            snr2 = snr_tensor.real.square() + snr_tensor.imag.square()
        else:
            snr2 = snr_tensor.square()
        rho_tensor = torch.sqrt(torch.sum(snr2, dim=0))
        above_mask = rho_tensor > threshold
        above_indices = _torch_selection_indices(above_mask)
        index, above = _torch_select(index, above_indices)
        coinc_triggers = {
            ifo: _torch_select(trigger, above_indices)[0]
            for ifo, trigger in coinc_triggers.items()
        }
        rho_coinc = _torch_array(rho_tensor[above_indices])
        return rho_coinc, index, coinc_triggers

    # Calculate the coincident snr
    snr_array = np.array(
        [coinc_triggers[ifo] for ifo in coinc_triggers.keys()]
    )
    rho_coinc = abs(np.sqrt(np.sum(snr_array * snr_array.conj(), axis=0)))
    # Apply threshold
    thresh_indexes = rho_coinc > threshold
    index = index[thresh_indexes]
    coinc_triggers = get_coinc_triggers(snr_dict, index, time_delay_idx)
    rho_coinc = rho_coinc[thresh_indexes]
    return rho_coinc, index, coinc_triggers


def get_projection_matrix(f_plus, f_cross, sigma, projection="standard"):
    """Calculate the matrix that projects the signal onto the network.
    Definitions can be found in Fairhurst (2018) [arXiv:1712.04724].
    For the standard projection see Eq. 8, and for left/right
    circular projections see Eq. 21, with further discussion in
    Appendix A. See also Williamson et al. (2014) [arXiv:1410.6042]
    for discussion in context of the GRB search with restricted
    binary inclination angles.

    Parameters
    ----------
    f_plus: dict
        Dictionary containing the plus antenna response factors for
        each IFO
    f_cross: dict
        Dictionary containing the cross antenna response factors for
        each IFO
    sigma: dict
        Dictionary of the sensitivity weights for each IFO
    projection: optional, {string, 'standard'}
        The signal polarization to project. Choice of 'standard'
        (unrestricted; default), 'right' or 'left' (circular
        polarizations)

    Returns
    -------
    projection_matrix: np.ndarray
        The matrix that projects the signal onto the detector network
    """
    # Calculate the weighted antenna responses
    keys = sorted(sigma.keys())
    w_p = np.array([sigma[ifo] * f_plus[ifo] for ifo in keys])
    w_c = np.array([sigma[ifo] * f_cross[ifo] for ifo in keys])

    # Get the projection matrix associated with the requested projection
    if projection == "standard":
        denom = np.dot(w_p, w_p) * np.dot(w_c, w_c) - np.dot(w_p, w_c) ** 2
        projection_matrix = (
            np.dot(w_c, w_c) * np.outer(w_p, w_p)
            + np.dot(w_p, w_p) * np.outer(w_c, w_c)
            - np.dot(w_p, w_c) * (np.outer(w_p, w_c) + np.outer(w_c, w_p))
        ) / denom
    elif projection == "left":
        projection_matrix = (
            np.outer(w_p, w_p)
            + np.outer(w_c, w_c)
            + (np.outer(w_p, w_c) - np.outer(w_c, w_p)) * 1j
        ) / (np.dot(w_p, w_p) + np.dot(w_c, w_c))
    elif projection == "right":
        projection_matrix = (
            np.outer(w_p, w_p)
            + np.outer(w_c, w_c)
            + (np.outer(w_c, w_p) - np.outer(w_p, w_c)) * 1j
        ) / (np.dot(w_p, w_p) + np.dot(w_c, w_c))
    else:
        raise ValueError(
            f'Unknown projection: {projection}. Allowed values are: '
            '"standard", "left", and "right"')

    return projection_matrix


def coherent_snr(
    snr_triggers, index, threshold, projection_matrix, coinc_snr=None
):
    """Calculate the coherent SNR for a given set of triggers. See
    Eq. 2.26 of Harry & Fairhurst (2011) [arXiv:1012.4939].


    Parameters
    ----------
    snr_triggers: dict
        Dictionary of the normalised complex snr time series for each
        ifo
    index: numpy.ndarray
        Array of the indexes corresponding to triggers
    threshold: float
        Coherent SNR threshold. Triggers below this are cut
    projection_matrix: numpy.ndarray
        Matrix that projects the signal onto the network
    coinc_snr: Optional- The coincident snr for each trigger.

    Returns
    -------
    rho_coh: numpy.ndarray or Array
        Array of coherent SNR for the detector network
    index: numpy.ndarray
        Indexes that survive cuts
    snrv: dict
        Dictionary of individual deector triggers that survive cuts
    coinc_snr: list or None (default: None)
        The coincident SNR values for triggers surviving the coherent
        cut
    """
    ifos = sorted(snr_triggers.keys())
    torch_tensors = _torch_coherent_tensors(
        *(snr_triggers[ifo] for ifo in ifos), kinds="fc"
    )
    projection = np.asarray(projection_matrix)
    if (
        torch_tensors is not None
        and projection.shape == (len(ifos), len(ifos))
        and projection.dtype.kind in "fc"
    ):
        import torch

        snr_tensor = torch.stack(torch_tensors)
        projection_tensor = torch.as_tensor(
            projection, device=snr_tensor.device
        )
        target_dtype = torch.promote_types(
            snr_tensor.dtype, projection_tensor.dtype
        )
        snr_tensor = snr_tensor.to(dtype=target_dtype)
        projection_tensor = projection_tensor.to(dtype=target_dtype)
        snr_proj = torch.matmul(
            snr_tensor.conj().transpose(0, 1),
            projection_tensor.transpose(0, 1),
        )
        rho_coh2 = torch.sum(
            snr_proj.transpose(0, 1) * snr_tensor, dim=0
        )
        rho_tensor = torch.abs(torch.sqrt(rho_coh2))
        above_mask = rho_tensor > threshold
        above_indices = _torch_selection_indices(above_mask)
        index, above = _torch_select(index, above_indices)
        coinc_snr = [] if coinc_snr is None else coinc_snr
        if len(coinc_snr) != 0:
            coinc_snr, above = _torch_select(
                coinc_snr, above_indices, above
            )
        snrv = {
            ifo: _torch_select(snr_triggers[ifo], above_indices)[0]
            for ifo in snr_triggers.keys()
        }
        rho_coh = _torch_array(rho_tensor[above_indices])
        return rho_coh, index, snrv, coinc_snr

    # Calculate rho_coh
    snr_array = np.array(
        [snr_triggers[ifo] for ifo in ifos]
    )
    snr_proj = np.inner(snr_array.conj().transpose(), projection_matrix)
    rho_coh2 = sum(snr_proj.transpose() * snr_array)
    rho_coh = abs(np.sqrt(rho_coh2))
    # Apply thresholds
    above = rho_coh > threshold
    index = index[above]
    coinc_snr = [] if coinc_snr is None else coinc_snr
    if len(coinc_snr) != 0:
        coinc_snr = coinc_snr[above]
    snrv = {
        ifo: snr_triggers[ifo][above]
        for ifo in snr_triggers.keys()
    }
    rho_coh = rho_coh[above]
    return rho_coh, index, snrv, coinc_snr


def select_coherent_values(values_left, values_right, select_left):
    """Select left- or right-polarized values point by point."""
    tensors = _torch_coherent_tensors(
        values_left, values_right, kinds="fciu"
    )
    if tensors is not None:
        import torch

        left_tensor, right_tensor = tensors
        selector = _torch_boolean_mask(select_left, left_tensor)
        if selector is not None:
            return _torch_array(
                torch.where(selector, left_tensor, right_tensor)
            )
    return np.where(select_left, values_left, values_right)


def compare_coherent_values(values_left, values_right, comparison):
    """Compare coherent values without moving Torch masks to the host.

    The return value follows the active backend: a Torch boolean tensor for
    device-backed values and the comparison's normal NumPy result otherwise.
    """
    from pycbc import scheme

    if scheme.current_prefix() == "torch":
        import torch
        from pycbc.types.array_torch import (
            TorchArrayData,
            _device_matches_active,
        )

        operations = {
            np.less: torch.lt,
            np.less_equal: torch.le,
            np.greater: torch.gt,
            np.greater_equal: torch.ge,
        }
        operation = operations.get(comparison)
        left_data = (
            values_left._data
            if isinstance(values_left, Array)
            else values_left
        )
        if operation is not None and isinstance(left_data, TorchArrayData):
            left_tensor = left_data.tensor
            if _device_matches_active(left_tensor):
                right_data = (
                    values_right._data
                    if isinstance(values_right, Array)
                    else values_right
                )
                try:
                    if isinstance(right_data, TorchArrayData):
                        right_tensor = right_data.tensor
                        if (
                            not _device_matches_active(right_tensor)
                            or right_tensor.device != left_tensor.device
                        ):
                            raise ValueError
                    else:
                        right_tensor = torch.as_tensor(
                            right_data,
                            dtype=left_tensor.dtype,
                            device=left_tensor.device,
                        )
                    return operation(left_tensor, right_tensor)
                except (TypeError, ValueError, RuntimeError):
                    pass

    return comparison(values_left, values_right)


def select_coherent_triggers(snrv_left, snrv_right, select_left):
    """Select left- or right-polarized detector triggers point by point.

    The selector is normally derived from the small coherent-SNR summary
    arrays. Torch-backed detector triggers remain on their active device.

    Parameters
    ----------
    snrv_left, snrv_right: dict
        Matching dictionaries of detector trigger arrays.
    select_left: array-like
        Boolean selector. True chooses the left trigger and False chooses the
        right trigger.

    Returns
    -------
    dict
        Selected detector triggers.
    """
    if snrv_left.keys() != snrv_right.keys():
        raise ValueError("Left and right trigger dictionaries must match")

    selected = {}
    for ifo in snrv_left:
        selected[ifo] = select_coherent_values(
            snrv_left[ifo], snrv_right[ifo], select_left
        )
    return selected


def network_chisq(chisq, chisq_dof, snr_dict):
    """Calculate the network chi-squared statistic. This is the sum of
    SNR-weighted individual detector chi-squared values. See Eq. 5.4
    of Dorrington (2019) [http://orca.cardiff.ac.uk/id/eprint/128124].

    Parameters
    ----------
    chisq: dict
        Dictionary of individual detector chi-squared statistics
    chisq_dof: dict
        Dictionary of the number of degrees of freedom of the
        chi-squared statistic
    snr_dict: dict
        Dictionary of complex individual detector SNRs

    Returns
    -------
    net_chisq: numpy.ndarray or Array
        Network chi-squared values
    """
    ifos = sorted(snr_dict.keys())
    torch_chisq = _torch_network_chisq(
        chisq, chisq_dof, snr_dict, ifos
    )
    if torch_chisq is not None:
        return torch_chisq

    chisq_per_dof = dict.fromkeys(ifos)
    for ifo in ifos:
        chisq_per_dof[ifo] = chisq[ifo] / chisq_dof[ifo]
    snr2 = {
        ifo: (
            snr_dict[ifo].squared_norm()
            if isinstance(snr_dict[ifo], Array)
            else np.real(
                np.array(snr_dict[ifo])
                * np.array(snr_dict[ifo]).conj()
            )
        )
        for ifo in ifos
    }
    coinc_snr2 = sum(snr2.values())
    snr2_ratio = {ifo: snr2[ifo] / coinc_snr2 for ifo in ifos}
    net_chisq = sum([chisq_per_dof[ifo] * snr2_ratio[ifo] for ifo in ifos])
    return net_chisq


def null_snr(
    rho_coh, rho_coinc, apply_cut=True, null_min=5.25, null_grad=0.2,
    null_step=20.0, index=None, snrv=None
):
    """Calculate the null SNR and optionally apply threshold cut where
    null SNR > null_min where coherent SNR < null_step
    and null SNR > (null_grad * rho_coh + null_min) elsewhere. See
    Eq. 3.1 of Harry & Fairhurst (2011) [arXiv:1012.4939] or
    Eqs. 11 and 12 of Williamson et al. (2014) [arXiv:1410.6042].
    Note that in Eq. 12 rho_coh should instead be rho_coh-null_step as
    reported in Eq. 4.73 of https://orca.cardiff.ac.uk/id/eprint/128124/.

    Parameters
    ----------
    rho_coh: numpy.ndarray or Array
        Array of coherent snr triggers
    rho_coinc: numpy.ndarray or Array
        Array of coincident snr triggers
    apply_cut: bool
        Apply a cut and downweight on null SNR determined by null_min,
        null_grad, null_step (default True)
    null_min: scalar
        Any trigger with null SNR below this is retained
    null_grad: scalar
        Gradient of null SNR cut where coherent SNR > null_step
    null_step: scalar
        The threshold in coherent SNR rho_coh above which the null SNR
        threshold increases as null_grad * rho_coh
    index: dict or None (optional; default None)
        Indexes of triggers. If given, will remove triggers that fail
        cuts
    snrv: dict of None (optional; default None)
        Individual detector SNRs. If given will remove triggers that
        fail cut

    Returns
    -------
    null: numpy.ndarray or Array
        Null SNR for surviving triggers
    rho_coh: numpy.ndarray or Array
        Coherent SNR for surviving triggers
    rho_coinc: numpy.ndarray or Array
        Coincident SNR for suviving triggers
    index: dict
        Indexes for surviving triggers
    snrv: dict
        Single detector SNRs for surviving triggers
    """
    index = {} if index is None else index
    snrv = {} if snrv is None else snrv
    # Calculate null SNRs
    torch_tensors = _torch_coherent_tensors(
        rho_coh, rho_coinc, coerce_host=True
    )
    if torch_tensors is not None:
        import torch

        rho_coh_t, rho_coinc_t = torch_tensors
        null_t = torch.sqrt(
            torch.clamp_min(rho_coinc_t.square() - rho_coh_t.square(), 0)
        )
        if apply_cut:
            keep_mask = (
                ((null_t < null_min) & (rho_coh_t <= null_step))
                | (
                    (
                        null_t
                        < ((rho_coh_t - null_step) * null_grad + null_min)
                    )
                    & (rho_coh_t > null_step)
                )
            )
            keep_indices = _torch_selection_indices(keep_mask)
            keep = None
            if not isinstance(index, dict) or index:
                index, keep = _torch_select(index, keep_indices)
            selected_snrv = {}
            for ifo, triggers in snrv.items():
                selected_snrv[ifo], keep = _torch_select(
                    triggers, keep_indices, keep
                )
            snrv = selected_snrv
            rho_coh_t = rho_coh_t[keep_indices]
            rho_coinc_t = rho_coinc_t[keep_indices]
            null_t = null_t[keep_indices]
        return (
            _torch_array(null_t),
            _torch_array(rho_coh_t),
            _torch_array(rho_coinc_t),
            index,
            snrv,
        )

    null2 = rho_coinc ** 2 - rho_coh ** 2
    # Numerical errors may make this negative and break the sqrt, so set
    # negative values to 0.
    null2[null2 < 0] = 0
    null = null2 ** 0.5
    if apply_cut:
        # Make cut on null.
        keep = (
            ((null < null_min) & (rho_coh <= null_step))
            | (
                (null < ((rho_coh - null_step) * null_grad + null_min))
                & (rho_coh > null_step)
                )
            )
        index = index[keep]
        rho_coh = rho_coh[keep]
        selected_snrv = {}
        for ifo, triggers in snrv.items():
            selected = triggers[keep]
            if _torch_coherent_tensors(triggers, kinds="fc") is not None:
                selected = Array(selected, copy=False)
            selected_snrv[ifo] = selected
        snrv = selected_snrv
        rho_coinc = rho_coinc[keep]
        null = null[keep]
    return null, rho_coh, rho_coinc, index, snrv


def reweight_snr_by_null(
        network_snr, null, coherent, null_min=5.25, null_grad=0.2,
        null_step=20.0):
    """Re-weight the detection statistic as a function of the null SNR.
    See Eq. 16 of Williamson et al. (2014) [arXiv:1410.6042] and note
    that the 4.25 appearing there is actually linked to the 5.25 of
    Eq. 12, hence the -1 carried out in this function.

    Parameters
    ----------
    network_snr: numpy.ndarray
        Array containing SNR statistic to be re-weighted
    null: numpy.ndarray
        Null SNR array
    coherent:
        Coherent SNR array

    Returns
    -------
    rw_snr: numpy.ndarray or Array
        Re-weighted SNR for each trigger
    """
    tensors = _torch_coherent_tensors(
        network_snr, null, coherent, coerce_host=True
    )
    if tensors is not None:
        import torch
        from pycbc.types.array_torch import TorchArrayData

        network_t, null_t, coherent_t = tensors
        downweight = (
            ((null_t > null_min - 1) & (coherent_t <= null_step))
            | (
                (null_t > (coherent_t * null_grad + null_min - 1))
                & (coherent_t > null_step)
            )
        )
        rw_fac = torch.where(
            coherent_t > null_step,
            1 + null_t - (null_min - 1)
            - (coherent_t - null_step) * null_grad,
            1 + null_t - (null_min - 1),
        )
        values = torch.where(
            downweight, network_t / rw_fac, network_t
        )
        return Array(TorchArrayData(values), copy=False)

    downweight = (
        ((null > null_min - 1) & (coherent <= null_step))
        | (
            (null > (coherent * null_grad + null_min - 1))
            & (coherent > null_step)
            )
        )
    rw_fac = np.where(
        coherent > null_step,
        1 + null - (null_min - 1) - (coherent - null_step) * null_grad,
        1 + null - (null_min - 1)
        )
    rw_snr = np.where(downweight, network_snr / rw_fac, network_snr)
    return rw_snr


def reweightedsnr_cut(rw_snr, rw_snr_threshold):
    """
    Performs a cut on reweighted snr based on a given threshold

    Parameters
    ----------
    rw_snr: array of reweighted snr
    rw_snr_threshhold: any reweighted snr below this threshold is set to 0

    Returns
    -------
    rw_snr: array of reweighted snr with cut values as 0

    """
    if rw_snr_threshold is not None:
        tensors = _torch_coherent_tensors(rw_snr)
        if tensors is not None:
            import torch
            from pycbc.types.array_torch import TorchArrayData

            values = torch.where(
                tensors[0] < rw_snr_threshold,
                torch.zeros_like(tensors[0]),
                tensors[0],
            )
            return Array(TorchArrayData(values), copy=False)

        rw_snr = np.where(rw_snr < rw_snr_threshold, 0, rw_snr)
    return rw_snr
