"""Backend-neutral access to array storage.

Domain modules should use this protocol instead of inspecting private
``Array._data`` implementations or importing an optional array library merely
to identify its values.
"""


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
