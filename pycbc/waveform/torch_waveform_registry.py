# Copyright (C) 2026  PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.

"""Declarative registry for Torch-native waveform implementations.

This module deliberately has no dependency on torch or lalsimulation. Native
implementation modules are imported only after a matching waveform is called
under :class:`~pycbc.scheme.TorchScheme` with its feature switch enabled.
"""

from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from typing import Mapping, Optional

from pycbc.waveform.torch_switches import torch_native_override

_INTERFACE_FIELDS = {
    "td": ("td_generator", "td_supported"),
    "td_modes": ("td_modes_generator", "td_modes_supported"),
    "fd": ("fd_generator", "fd_supported"),
    "fd_modes": ("fd_modes_generator", "fd_modes_supported"),
    "sequence": ("sequence_generator", "sequence_supported"),
}


@dataclass(frozen=True)
class TorchNativeWaveform:
    """Entry points and feature switch for one native approximant."""

    approximant: str
    component_flag: str
    module: str
    default_enabled: bool = False
    default_supported: Optional[str] = None
    td_generator: Optional[str] = None
    td_supported: Optional[str] = None
    td_modes_generator: Optional[str] = None
    td_modes_supported: Optional[str] = None
    fd_generator: Optional[str] = None
    fd_supported: Optional[str] = None
    fd_modes_generator: Optional[str] = None
    fd_modes_supported: Optional[str] = None
    sequence_generator: Optional[str] = None
    sequence_supported: Optional[str] = None


_PORTS = ()


def _declared_interfaces(port):
    return tuple(
        interface
        for interface, (generator_field, _) in _INTERFACE_FIELDS.items()
        if getattr(port, generator_field) is not None
    )


def _validate_ports(ports):
    names = [port.approximant for port in ports]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate Torch-native waveform registry entry")

    for port in ports:
        if not _declared_interfaces(port):
            raise RuntimeError(
                f"Torch-native waveform {port.approximant!r} has no interface"
            )
        for interface, (generator_field, supported_field) in _INTERFACE_FIELDS.items():
            generator = getattr(port, generator_field)
            supported = getattr(port, supported_field)
            if (generator is None) != (supported is None):
                raise RuntimeError(
                    f"Torch-native waveform {port.approximant!r} has an "
                    f"incomplete {interface!r} declaration"
                )


_validate_ports(_PORTS)

TORCH_NATIVE_WAVEFORMS: Mapping[str, TorchNativeWaveform] = MappingProxyType(
    {port.approximant: port for port in _PORTS}
)


def native_approximants(interface):
    """Return names with a native implementation for ``interface``."""
    try:
        generator_field, _ = _INTERFACE_FIELDS[interface]
    except KeyError as exc:
        raise ValueError(f"unknown waveform interface {interface!r}") from exc
    return tuple(
        name
        for name, port in TORCH_NATIVE_WAVEFORMS.items()
        if getattr(port, generator_field) is not None
    )


def try_torch_native_waveform(interface, params):
    """Run an enabled and supported native port, or return ``None``.

    The caller owns scheme selection. Parameter-support and default-readiness
    predicates remain authoritative, and environment overrides control whether
    an implementation is attempted before the standard LAL fallback.
    """
    try:
        generator_field, supported_field = _INTERFACE_FIELDS[interface]
    except KeyError as exc:
        raise ValueError(f"unknown waveform interface {interface!r}") from exc

    approximant = params.get("approximant")
    port = TORCH_NATIVE_WAVEFORMS.get(approximant)
    if port is None:
        return None

    generator_name = getattr(port, generator_field)
    supported_name = getattr(port, supported_field)
    if generator_name is None:
        return None

    override = torch_native_override(port.component_flag)
    enabled = port.default_enabled if override is None else override
    if not enabled:
        return None

    implementation = import_module(f".{port.module}", package=__package__)
    if not getattr(implementation, supported_name)(params):
        return None
    if (
        override is None
        and port.default_supported is not None
        and not getattr(implementation, port.default_supported)(params)
    ):
        return None

    return getattr(implementation, generator_name)(**params)


__all__ = (
    "TORCH_NATIVE_WAVEFORMS",
    "TorchNativeWaveform",
    "native_approximants",
    "try_torch_native_waveform",
)
