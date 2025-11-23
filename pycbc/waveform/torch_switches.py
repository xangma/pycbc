"""Environment flag helper for torch-native waveform ports.

Allows a single global switch (``PYCBC_TORCH_NATIVE_PORTS`` or
``PYCBC_TORCH_NATIVE``) to enable/disable all torch-native ports while still
honouring per-component flags such as ``PYCBC_SPINTAYLORF2_NATIVE``.
Component-specific variables take precedence; if they are unset, the global
flag is used; otherwise the provided default is returned.
"""

import os

_TRUE = {"1", "true", "yes", "on"}


def torch_native_enabled(component_flag: str, *, default: bool = False) -> bool:
    """Return True if the torch-native implementation should be used.

    Parameters
    ----------
    component_flag : str
        Environment variable specific to the component, e.g.
        ``PYCBC_SPINTAYLORF2_NATIVE``.
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

    def _parse(val):
        return str(val).lower() in _TRUE

    if component_flag in os.environ:
        return _parse(os.environ[component_flag])

    for env in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_TORCH_NATIVE"):
        if env in os.environ:
            return _parse(os.environ[env])

    return default
