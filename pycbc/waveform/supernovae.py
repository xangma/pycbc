"""Generate core-collapse supernovae waveform for core bounce and
subsequent postbounce oscillations.
"""

import os

import numpy

import pycbc.scheme as _scheme
from pycbc.types import TimeSeries
from pycbc.io.hdf import HFile

_pc_dict = {}
_torch_pc_dict = {}


def _load_principal_components(filename):
    """Load and cache a principal-component basis by absolute filename."""
    cache_key = os.path.abspath(os.fspath(filename))
    try:
        principal_components = _pc_dict[cache_key]
    except KeyError:
        with HFile(cache_key, 'r') as pc_file:
            principal_components = numpy.array(
                pc_file['principal_components']
            )
        _pc_dict[cache_key] = principal_components
    return cache_key, principal_components


def _torch_tensor(value, device, dtype):
    """Return ``value`` as a tensor without staging device data on the host."""
    import torch

    tensor = value if isinstance(value, torch.Tensor) else None
    if tensor is None:
        tensor = getattr(value, 'tensor', None)
    if tensor is None:
        tensor = getattr(getattr(value, '_data', None), 'tensor', None)
    if tensor is not None:
        return tensor.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _torch_principal_components(cache_key, components, device, dtype):
    """Return a cached principal-component basis on a Torch device."""
    import torch

    key = (cache_key, len(components), str(device), dtype)
    try:
        return _torch_pc_dict[key]
    except KeyError:
        basis = torch.as_tensor(components, device=device, dtype=dtype)
        _torch_pc_dict[key] = basis
        return basis


def get_corecollapse_bounce(**kwargs):
    """Generate core bounce and postbounce waveform by using principal
    component basis vectors from a .hdf file. The waveform parameters are the
    coefficients of the principal components and the distance. The number of
    principal components used can also be varied.
    """

    cache_key, principal_components = _load_principal_components(
        kwargs['principal_components_file']
    )

    if 'coefficients_array' in kwargs:
        coefficients_array = kwargs['coefficients_array']
    else:
        coeffs_keys = sorted(
            (x for x in kwargs if x.startswith('coeff_')),
            key=lambda name: int(name[6:]),
        )
        coefficients_array = [kwargs[x] for x in coeffs_keys]

    no_of_pcs = int(kwargs['no_of_pcs'])
    if not 0 <= no_of_pcs <= len(principal_components):
        raise ValueError(
            f'no_of_pcs must be between 0 and {len(principal_components)}'
        )
    coefficients_array = coefficients_array[:no_of_pcs]
    principal_components = principal_components[:no_of_pcs]

    if len(coefficients_array) != no_of_pcs:
        raise ValueError(
            f'Expected {no_of_pcs} principal-component coefficients, got '
            f'{len(coefficients_array)}'
        )

    distance = kwargs['distance']
    mpc_conversion = 3.08567758128e+22

    state = _scheme.mgr.state
    if isinstance(state, _scheme.TorchScheme):
        import torch
        from pycbc.types.array_torch import TorchArrayData

        device = state.torch_device
        dtype = torch.float32 if device.type == 'mps' else torch.float64
        coefficients_array = _torch_tensor(
            coefficients_array, device, dtype
        )
        distance = _torch_tensor(distance, device, dtype) * mpc_conversion
        principal_components = _torch_principal_components(
            cache_key, principal_components, device, dtype
        )
        strain = torch.matmul(
            coefficients_array, principal_components
        ) / distance
        cross = torch.zeros_like(strain)
        strain = TorchArrayData(strain)
        cross = TorchArrayData(cross)
        copy = False
    else:
        coefficients_array = numpy.asarray(coefficients_array)
        distance *= mpc_conversion
        strain = numpy.dot(
            coefficients_array, principal_components
        ) / distance
        cross = numpy.zeros(len(strain))
        copy = True

    delta_t = kwargs['delta_t']
    outhp = TimeSeries(strain, delta_t=delta_t, copy=copy)
    outhc = TimeSeries(cross, delta_t=delta_t, copy=copy)
    return outhp, outhc


# Approximant names ###########################################################
supernovae_td_approximants = {'CoreCollapseBounce': get_corecollapse_bounce}
