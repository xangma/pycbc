# Copyright (C) 2026 The PyCBC Collaboration
# Licensed under the GNU General Public License, version 3 or later.

"""Pack terminal Live columns into fewer device-to-host transfers."""

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _pack_live_arrays(*arrays):
    return jnp.concatenate(tuple(jnp.ravel(array) for array in arrays))


def _packing_key(value):
    if (
        not isinstance(value, jax.Array)
        or isinstance(value, jax.core.Tracer)
        or not isinstance(value.sharding, jax.sharding.SingleDeviceSharding)
        or not value.is_fully_addressable
    ):
        return None
    try:
        dtype = np.dtype(value.dtype)
    except TypeError:
        # Extended JAX dtypes (for example typed PRNG keys) retain their
        # native device_get protocol rather than becoming NumPy buffers.
        return None
    if dtype.kind not in "biufc" or dtype.hasobject:
        return None
    devices = value.devices()
    if len(devices) != 1:
        return None
    device = next(iter(devices))
    # Host/offloaded memory kinds need their original collection path; packing
    # must not silently bring them onto the accelerator first.
    if value.sharding != jax.sharding.SingleDeviceSharding(device):
        return None
    return dtype.str, value.sharding


def collect_live_arrays(tree):
    """Collect a terminal pytree with byte-preserving dtype/device packing.

    Leaves on different devices or with different dtypes never share a buffer.
    Unsupported leaves pass through the same final ``device_get`` as native
    leaves, retaining its conversion contract. Returned views retain their
    packed NumPy owner; callers may copy them when mutable ownership is needed.
    """
    leaves, structure = jax.tree_util.tree_flatten(tree)
    groups = {}
    for index, value in enumerate(leaves):
        key = _packing_key(value)
        if key is not None:
            groups.setdefault(key, []).append(index)
    groups = [
        (key, indices) for key, indices in groups.items() if len(indices) > 1
    ]
    packed = []
    packed_indices = set()
    for (_, placement), indices in groups:
        device = next(iter(placement.device_set))
        with jax.default_device(device):
            packed.append(
                _pack_live_arrays(*(leaves[index] for index in indices))
            )
        packed_indices.update(indices)
    native_indices = [
        index for index in range(len(leaves)) if index not in packed_indices
    ]
    native, buffers = jax.device_get(
        ([leaves[index] for index in native_indices], packed)
    )
    for index, value in zip(native_indices, native):
        leaves[index] = value
    for (_, indices), buffer in zip(groups, buffers):
        offset = 0
        for index in indices:
            original = leaves[index]
            leaves[index] = buffer[offset:offset + original.size].reshape(
                original.shape
            )
            offset += original.size
    return jax.tree_util.tree_unflatten(structure, leaves)
