"""Environment flag helper for torch-native waveform ports.

Allows a single global switch (``PYCBC_TORCH_NATIVE_PORTS`` or
``PYCBC_TORCH_NATIVE``) to enable/disable all torch-native ports while still
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


def torch_native_override(component_flag: str):
    """Return the explicit native-port override, or ``None`` if unset."""
    if component_flag in os.environ:
        return _parse_switch(component_flag, os.environ[component_flag])

    for env in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_TORCH_NATIVE"):
        if env in os.environ:
            return _parse_switch(env, os.environ[env])
    return None


def torch_native_enabled(component_flag: str, *, default: bool = False) -> bool:
    """Return True if the torch-native implementation should be used.

    Parameters
    ----------
    component_flag : str
        Environment variable specific to the component, e.g.
        ``PYCBC_EXAMPLE_NATIVE``.
    default : bool, optional
        Fallback if neither the component flag nor a global flag is set.

    Notes
    -----
    Global overrides:
    - ``PYCBC_TORCH_NATIVE_PORTS``
    - ``PYCBC_TORCH_NATIVE`` (alias)

    Per-component flags always win if present. Global flags are consulted only
    when the component flag is unset.
    """

    override = torch_native_override(component_flag)
    return default if override is None else override
