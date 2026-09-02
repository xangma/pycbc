"""Small helpers shared by Torch-aware population calculations."""

import numpy as np


def tensors(*values):
    """Return compatible floating tensors under the active Torch scheme."""
    if not values:
        return None

    from pycbc import scheme

    if scheme.current_prefix() != "torch":
        return None

    import torch
    from pycbc.types import Array
    from pycbc.types.array_torch import (
        TorchArrayData,
        _device_matches_active,
    )

    data = [
        value._data if isinstance(value, Array) else value
        for value in values
    ]
    first = data[0]
    if isinstance(first, TorchArrayData):
        first = first.tensor
    if not isinstance(first, torch.Tensor) or not (
        first.is_floating_point() and _device_matches_active(first)
    ):
        return None

    result = []
    for value in data:
        if isinstance(value, TorchArrayData):
            value = value.tensor
        if isinstance(value, torch.Tensor):
            if (
                value.is_complex()
                or value.device != first.device
                or not _device_matches_active(value)
            ):
                return None
            result.append(value.to(dtype=first.dtype))
        else:
            try:
                host = np.asarray(value)
                if host.dtype.kind not in "fiu":
                    return None
                result.append(
                    torch.as_tensor(
                        host,
                        dtype=first.dtype,
                        device=first.device,
                    )
                )
            except (TypeError, ValueError, RuntimeError):
                return None

    return tuple(result)


def result(input_value, tensor):
    """Wrap a tensor when the corresponding input was a PyCBC Array."""
    from pycbc.types import Array

    if isinstance(input_value, Array):
        from pycbc.types.array_torch import TorchArrayData

        return Array(TorchArrayData(tensor), copy=False)
    return tensor
