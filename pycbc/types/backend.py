# Copyright (C) 2026
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

"""Backend-neutral access to array storage.

Domain modules should use this protocol instead of inspecting private
``Array._data`` implementations or importing an optional array library merely
to identify its values.
"""

import sys
from functools import lru_cache


@lru_cache(maxsize=32)
def _jax_module_for_type(value_type):
    """Cache type classification, never array values or execution state."""
    jax = sys.modules.get("jax")
    mod_prefix = getattr(value_type, "__module__", "").partition(".")[0]
    if jax is None and mod_prefix not in ("jax", "jaxlib"):
        return None
    if jax is None:
        try:
            import jax
        except ImportError:
            return None
    array_cls = getattr(jax, "Array", None)
    tracer_cls = getattr(getattr(jax, "core", None), "Tracer", None)
    valid_classes = tuple(cls for cls in (array_cls, tracer_cls) if cls is not None)
    if valid_classes and issubclass(value_type, valid_classes):
        return jax
    return None


def jax_module_for(value):
    """Return JAX for a raw JAX array, including user-defined subclasses.

    Ordinary host values do not cause an import of the optional dependency.
    The bounded type cache keeps repeated boundary operations inexpensive.
    """
    try:
        return _jax_module_for_type(type(value))
    except TypeError:
        # A custom metaclass may make its class object unhashable.
        return _jax_module_for_type.__wrapped__(type(value))



def backend_name(value):
    """Return the declared array backend name, if one can be identified."""
    declared = getattr(value, "backend", None)
    if declared is not None:
        return declared

    storage = getattr(value, "_data", value)
    declared = getattr(storage, "backend", None)
    if declared is not None:
        return declared

    module = getattr(type(storage), "__module__", "").partition(".")[0]
    if module in ("numpy", "cupy", "jax"):
        return module
    if jax_module_for(storage) is not None:
        return "jax"
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
    """Adapt native storage for a PyCBC Array or Series constructor."""
    storage = backend_array(value)
    if jax_module_for(storage) is not None:
        from .array_jax import JAXArrayData

        return JAXArrayData(storage)
    return storage


def backend_matches_scheme(value):
    """Whether storage can be used without a copy in the active scheme."""
    from .array import _scheme_matches_base_array

    return _scheme_matches_base_array(wrap_backend_array(value))


def coerce_jax_values(*values):
    """Coerce mixed inputs to the first JAX array's device and dtype.

    Return ``(None, values)`` unchanged when there are no JAX inputs,
    without importing JAX. Otherwise return ``(jax, converted_values)``.
    """
    storage = tuple(
        backend_array(value, "jax") if is_backend(value, "jax") else value
        for value in values
    )
    reference = next(
        (value for value in storage if jax_module_for(value) is not None), None
    )
    if reference is None:
        return None, values

    jax = jax_module_for(reference)
    import jax.numpy as jnp

    dtype = reference.dtype
    if not (jnp.issubdtype(dtype, jnp.floating) or jnp.issubdtype(dtype, jnp.complexfloating)):
        dtype = jnp.float64
    converted = tuple(
        value.astype(dtype) if jax_module_for(value) is not None
        else jnp.asarray(value, dtype=dtype)
        for value in storage
    )
    return jax, converted

