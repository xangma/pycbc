# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.

"""Environment flag helper for JAX-native waveform ports.

Allows a single global switch (``PYCBC_JAX_NATIVE_PORTS`` or
``PYCBC_JAX_NATIVE``) to enable/disable all JAX-native ports while still
honouring per-component flags such as ``PYCBC_EXAMPLE_NATIVE``.
Component-specific variables take precedence; if they are unset, the global
flag is used; otherwise the provided default is returned.
"""

import os

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _parse_switch(name, value):
    normalized = str(value).strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    choices = ", ".join(sorted(_TRUE | _FALSE))
    raise ValueError(f"{name} must be one of: {choices}; got {value!r}")


def jax_native_override(component_flag: str):
    """Return the explicit native-port override, or ``None`` if unset."""
    if component_flag in os.environ:
        return _parse_switch(component_flag, os.environ[component_flag])

    for env in ("PYCBC_JAX_NATIVE_PORTS", "PYCBC_JAX_NATIVE"):
        if env in os.environ:
            return _parse_switch(env, os.environ[env])
    return None


def jax_native_enabled(component_flag: str, *, default: bool = False) -> bool:
    """Return True if the JAX-native implementation should be used.

    Parameters
    ----------
    component_flag : str
        Environment variable specific to the component.
    default : bool, optional
        Fallback if neither the component flag nor a global flag is set.
    """
    override = jax_native_override(component_flag)
    return default if override is None else override
