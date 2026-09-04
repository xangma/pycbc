"""Backend-neutral access to array storage.

Domain modules should use this protocol instead of inspecting private
``Array._data`` implementations or importing an optional array library merely
to identify its values.
"""

from functools import lru_cache
import sys


@lru_cache(maxsize=32)
def _torch_module_for_type(value_type):
    """Cache type classification, never tensor values or execution state."""
    torch = sys.modules.get("torch")
    if torch is None and value_type.__module__.partition(".")[0] != "torch":
        return None
    if torch is None:
        import torch
    return torch if issubclass(value_type, torch.Tensor) else None


def torch_module_for(value):
    """Return Torch for a raw tensor, including user-defined subclasses.

    Ordinary host values do not cause an import of the optional dependency.
    The bounded type cache keeps repeated boundary operations inexpensive;
    devices, dtypes and autograd state are deliberately not cached.
    """
    try:
        return _torch_module_for_type(type(value))
    except TypeError:
        # A custom metaclass may make its class object unhashable.
        return _torch_module_for_type.__wrapped__(type(value))


def backend_name(value):
    """Return the declared array backend name, if one can be identified."""
    declared = getattr(value, "backend", None)
    if declared is not None:
        return declared

    storage = getattr(value, "_data", value)
    declared = getattr(storage, "backend", None)
    if declared is not None:
        return declared
    module = type(storage).__module__.partition(".")[0]
    if module in ("numpy", "cupy", "torch"):
        return module
    if torch_module_for(storage) is not None:
        return "torch"
    return None


def is_backend(value, name):
    """Return whether ``value`` belongs to the named array backend."""
    return backend_name(value) == name


def backend_array(value, name=None):
    """Return public backend storage without exposing PyCBC internals.

    ``None`` is returned when ``name`` is supplied and the value belongs to a
    different backend. Plain backend arrays are returned unchanged.
    """
    if name is not None and backend_name(value) != name:
        return None

    accessor = getattr(value, "backend_array", None)
    if accessor is not None:
        return accessor() if callable(accessor) else accessor

    storage = getattr(value, "_data", value)
    accessor = getattr(storage, "backend_array", None)
    if accessor is None:
        return storage
    return accessor() if callable(accessor) else accessor


def wrap_backend_array(value):
    """Adapt native storage for a PyCBC Array or Series constructor.

    This is the inverse of :func:`backend_array`: Torch tensors receive the
    backend's NumPy-compatible storage wrapper; other native arrays pass
    through unchanged. No data is copied, moved or detached. Constructors
    still enforce the active scheme, supported dtype and series metadata.
    """
    storage = backend_array(value)
    if torch_module_for(storage) is not None:
        from .array_torch import TorchArrayData
        return TorchArrayData(storage)
    return storage


def backend_matches_scheme(value):
    """Whether storage can be used without a copy in the active scheme."""
    from .array import _scheme_matches_base_array
    return _scheme_matches_base_array(wrap_backend_array(value))


def coerce_torch_values(*values):
    """Coerce mixed inputs to the first tensor's device and dtype.

    Public Torch storage wrappers are accepted. Integer and boolean tensors
    select Torch's default floating dtype for scientific calculations; complex
    and floating dtypes are retained. Tensor conversions preserve autograd.
    Shapes are unchanged: callers choose their own broadcasting policy.

    Return ``(None, values)`` unchanged when there are no tensor inputs,
    without importing Torch. Otherwise return ``(torch, converted_values)``.
    """
    storage = tuple(backend_array(value, "torch")
                    if is_backend(value, "torch") else value
                    for value in values)
    reference = next((value for value in storage
                      if torch_module_for(value) is not None), None)
    if reference is None:
        return None, values

    torch = torch_module_for(reference)
    dtype = reference.dtype
    if not (dtype.is_floating_point or dtype.is_complex):
        dtype = torch.get_default_dtype()
    converted = tuple(
        value.to(device=reference.device, dtype=dtype)
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, device=reference.device, dtype=dtype)
        for value in storage
    )
    return torch, converted
