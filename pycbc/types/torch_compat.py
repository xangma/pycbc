"""Private CPU arithmetic bridges for the ordinary Torch search path.

These bridges copy their inputs and return results to the active Torch device.
Differentiable tensors and Tensor subclasses must retain Torch dispatch.
"""

from contextlib import contextmanager
from functools import lru_cache

from pycbc import scheme
from .backend import backend_array, torch_module_for


def cpu_compatible(value):
    """Whether a single-precision search array can use host arithmetic."""
    if not isinstance(scheme.mgr.state, scheme.TorchScheme):
        return False
    tensor = backend_array(value, "torch")
    torch = torch_module_for(tensor)
    if torch is None or type(tensor) is not torch.Tensor:
        return False
    if (
        tensor.device.type not in ("cpu", "cuda")
        or tensor.dtype not in (torch.float32, torch.complex64)
        or tensor.requires_grad
    ):
        return False
    try:
        if torch._C._functorch.is_functorch_wrapped_tensor(tensor):
            return False
    except AttributeError:
        pass
    try:
        return torch.autograd.forward_ad.unpack_dual(tensor).tangent is None
    except (AttributeError, RuntimeError):
        return False


@lru_cache(maxsize=1)
def _cpu_scheme():
    return scheme.CPUScheme()


@contextmanager
def cpu_context():
    """Dispatch copied host arrays through the unchanged CPU implementation.

    PyCBC's processing scheme is process-global. This private synchronous
    bridge retains its lock and restores its state even on failure. It does
    not enter or exit a public CPUScheme, so thread settings are unchanged.
    No caller-owned PyCBC array may be accessed inside the bridge.
    """
    state, locked = scheme.mgr.state, scheme.mgr._lock
    if not isinstance(state, scheme.TorchScheme):
        raise RuntimeError("CPU compatibility bridges require a Torch scheme")
    scheme.mgr.state = _cpu_scheme()
    scheme.mgr.lock()
    try:
        yield
    finally:
        scheme.mgr.state = state
        scheme.mgr._lock = locked
