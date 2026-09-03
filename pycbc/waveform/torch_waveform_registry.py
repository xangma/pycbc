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


def _fd_sequence_port(
    approximant,
    component_flag,
    module,
    stem,
    *,
    default_enabled=True,
    default_supported=None,
    fd_generator=None,
    fd_supported=None,
    fd_modes_generator=None,
    fd_modes_supported=None,
    sequence_generator=None,
    sequence_supported=None,
    td_generator=None,
    td_supported=None,
    td_modes_generator=None,
    td_modes_supported=None,
):
    """Build the common FD plus arbitrary-frequency registry entry."""
    return TorchNativeWaveform(
        approximant=approximant,
        component_flag=component_flag,
        module=module,
        default_enabled=default_enabled,
        default_supported=default_supported,
        td_generator=td_generator,
        td_supported=td_supported,
        td_modes_generator=td_modes_generator,
        td_modes_supported=td_modes_supported,
        fd_generator=fd_generator or f"{stem}_fd_torch",
        fd_supported=fd_supported or f"{stem}_native_supported",
        fd_modes_generator=fd_modes_generator,
        fd_modes_supported=fd_modes_supported,
        sequence_generator=(
            sequence_generator or f"{stem}_fd_sequence_torch"
        ),
        sequence_supported=(
            sequence_supported or f"{stem}_sequence_native_supported"
        ),
    )


_PORTS = (
    _fd_sequence_port(
        "TaylorF2",
        "PYCBC_TAYLORF2_NATIVE",
        "taylorf2_torch",
        "taylorf2",
    ),
    _fd_sequence_port(
        "TaylorF2NLTides",
        "PYCBC_TAYLORF2NLTIDES_NATIVE",
        "taylorf2nltides_torch",
        "taylorf2nltides",
    ),
    _fd_sequence_port(
        "TaylorF2RedSpin",
        "PYCBC_TAYLORF2REDSPIN_NATIVE",
        "taylorf2redspin_torch",
        "taylorf2redspin",
    ),
    _fd_sequence_port(
        "TaylorF2RedSpinTidal",
        "PYCBC_TAYLORF2REDSPINTIDAL_NATIVE",
        "taylorf2redspin_torch",
        "taylorf2redspin",
    ),
    _fd_sequence_port(
        "TaylorF2Ecc",
        "PYCBC_TAYLORF2ECC_NATIVE",
        "taylorf2ecc_torch",
        "taylorf2ecc",
    ),
    _fd_sequence_port(
        "SpinTaylorF2",
        "PYCBC_SPINTAYLORF2_NATIVE",
        "spintaylorf2_torch",
        "spintaylorf2",
        default_supported="spintaylorf2_default_native_supported",
        fd_generator="spintaylorf2_torch",
    ),
    _fd_sequence_port(
        "IMRPhenomA",
        "PYCBC_IMRPHENOMA_NATIVE",
        "imrphenomab_torch",
        "imrphenomab",
        default_supported="imrphenomab_default_native_supported",
    ),
    _fd_sequence_port(
        "IMRPhenomB",
        "PYCBC_IMRPHENOMB_NATIVE",
        "imrphenomab_torch",
        "imrphenomab",
        default_supported="imrphenomab_default_native_supported",
    ),
    _fd_sequence_port(
        "IMRPhenomC",
        "PYCBC_IMRPHENOMC_NATIVE",
        "imrphenomc_torch",
        "imrphenomc",
        default_supported="imrphenomc_default_native_supported",
    ),
    _fd_sequence_port(
        "IMRPhenomD",
        "PYCBC_IMRPHENOMD_NATIVE",
        "imrphenomd_torch",
        "imrphenomd",
    ),
    _fd_sequence_port(
        "IMRPhenomD_NRTidal",
        "PYCBC_IMRPHENOMD_NATIVE",
        "imrphenomd_torch",
        "imrphenomd",
    ),
    _fd_sequence_port(
        "IMRPhenomD_NRTidalv2",
        "PYCBC_IMRPHENOMD_NATIVE",
        "imrphenomd_torch",
        "imrphenomd",
    ),
    _fd_sequence_port(
        "IMRPhenomNSBH",
        "PYCBC_IMRPHENOMNSBH_NATIVE",
        "imrphenomnsbh_torch",
        "imrphenomnsbh",
    ),
    _fd_sequence_port(
        "IMRPhenomP",
        "PYCBC_IMRPHENOMP_NATIVE",
        "imrphenomp_torch",
        "imrphenomp",
    ),
    _fd_sequence_port(
        "IMRPhenomPv2",
        "PYCBC_IMRPHENOMPV2_NATIVE",
        "imrphenompv2_torch",
        "imrphenompv2",
    ),
    _fd_sequence_port(
        "IMRPhenomPv2_NRTidal",
        "PYCBC_IMRPHENOMPV2_NATIVE",
        "imrphenompv2_torch",
        "imrphenompv2",
    ),
    _fd_sequence_port(
        "IMRPhenomPv2_NRTidalv2",
        "PYCBC_IMRPHENOMPV2_NATIVE",
        "imrphenompv2_torch",
        "imrphenompv2",
    ),
    _fd_sequence_port(
        "IMRPhenomXAS",
        "PYCBC_IMRPHENOMXAS_NATIVE",
        "imrphenomxas_torch",
        "imrphenomxas",
    ),
    _fd_sequence_port(
        "IMRPhenomXAS_NRTidalv2",
        "PYCBC_IMRPHENOMXAS_NATIVE",
        "imrphenomxas_torch",
        "imrphenomxas",
    ),
    _fd_sequence_port(
        "IMRPhenomXAS_NRTidalv3",
        "PYCBC_IMRPHENOMXAS_NATIVE",
        "imrphenomxas_torch",
        "imrphenomxas",
    ),
    _fd_sequence_port(
        "IMRPhenomXP",
        "PYCBC_IMRPHENOMXP_NATIVE",
        "imrphenomxp_torch",
        "imrphenomxp",
    ),
    _fd_sequence_port(
        "IMRPhenomXP_NRTidalv2",
        "PYCBC_IMRPHENOMXP_NATIVE",
        "imrphenomxp_torch",
        "imrphenomxp",
    ),
    _fd_sequence_port(
        "IMRPhenomXP_NRTidalv3",
        "PYCBC_IMRPHENOMXP_NATIVE",
        "imrphenomxp_torch",
        "imrphenomxp",
    ),
    _fd_sequence_port(
        "IMRPhenomXHM",
        "PYCBC_IMRPHENOMXHM_NATIVE",
        "imrphenomxhm_torch",
        "imrphenomxhm",
        fd_supported="imrphenomxhm_fd_native_supported",
        fd_modes_generator="imrphenomxhm_modes_torch",
        fd_modes_supported="imrphenomxhm_modes_native_supported",
    ),
    _fd_sequence_port(
        "IMRPhenomXPHM",
        "PYCBC_IMRPHENOMXPHM_NATIVE",
        "imrphenomxphm_torch",
        "imrphenomxphm",
    ),
    _fd_sequence_port(
        "IMRPhenomXO4a",
        "PYCBC_IMRPHENOMXO4A_NATIVE",
        "imrphenomxo4a_torch",
        "imrphenomxo4a",
    ),
    _fd_sequence_port(
        "IMRPhenomXPNR",
        "PYCBC_IMRPHENOMXPNR_NATIVE",
        "imrphenomxpnr_waveform_torch",
        "imrphenomxpnr",
    ),
    _fd_sequence_port(
        "IMRPhenomHM",
        "PYCBC_IMRPHENOMHM_NATIVE",
        "imrphenomhm_torch",
        "imrphenomhm",
        fd_modes_generator="imrphenomhm_modes_torch",
        fd_modes_supported="imrphenomhm_modes_native_supported",
    ),
    _fd_sequence_port(
        "IMRPhenomPv3",
        "PYCBC_IMRPHENOMPV3_NATIVE",
        "imrphenompv3_torch",
        "imrphenompv3",
    ),
    _fd_sequence_port(
        "IMRPhenomPv3HM",
        "PYCBC_IMRPHENOMPV3HM_NATIVE",
        "imrphenompv3_torch",
        "imrphenompv3hm",
    ),
    TorchNativeWaveform(
        approximant="SpinTaylorT1",
        component_flag="PYCBC_SPINTAYLORT1_NATIVE",
        module="spintaylor_torch",
        td_generator="spintaylor_t1_td_torch",
        td_supported="spintaylor_t1_native_supported",
        td_modes_generator="spintaylor_t1_modes_torch",
        td_modes_supported="spintaylor_t1_modes_native_supported",
    ),
    TorchNativeWaveform(
        approximant="SpinTaylorT4",
        component_flag="PYCBC_SPINTAYLORT4_NATIVE",
        module="spintaylor_torch",
        td_generator="spintaylor_t4_td_torch",
        td_supported="spintaylor_t4_native_supported",
        td_modes_generator="spintaylor_t4_modes_torch",
        td_modes_supported="spintaylor_t4_modes_native_supported",
    ),
    TorchNativeWaveform(
        approximant="SpinTaylorT5",
        component_flag="PYCBC_SPINTAYLORT5_NATIVE",
        module="spintaylor_torch",
        td_generator="spintaylor_t5_td_torch",
        td_supported="spintaylor_t5_native_supported",
        td_modes_generator="spintaylor_t5_modes_torch",
        td_modes_supported="spintaylor_t5_modes_native_supported",
    ),
    TorchNativeWaveform(
        approximant="SpinTaylorT4Fourier",
        component_flag="PYCBC_SPINTAYLORT4FOURIER_NATIVE",
        module="spintaylor_fourier_torch",
        fd_generator="spintaylor_t4_fourier_fd_torch",
        fd_supported="spintaylor_t4_fourier_native_supported",
    ),
    TorchNativeWaveform(
        approximant="SpinTaylorT5Fourier",
        component_flag="PYCBC_SPINTAYLORT5FOURIER_NATIVE",
        module="spintaylor_fourier_torch",
        fd_generator="spintaylor_t5_fourier_fd_torch",
        fd_supported="spintaylor_t5_fourier_native_supported",
    ),
    TorchNativeWaveform(
        approximant="IMRPhenomT",
        component_flag="PYCBC_IMRPHENOMT_NATIVE",
        module="imrphenomt_torch",
        td_generator="imrphenomt_td_torch",
        td_supported="imrphenomt_native_supported",
    ),
    TorchNativeWaveform(
        approximant="IMRPhenomTHM",
        component_flag="PYCBC_IMRPHENOMTHM_NATIVE",
        module="imrphenomthm_torch",
        default_enabled=True,
        default_supported="imrphenomthm_default_native_supported",
        td_generator="imrphenomthm_td_torch",
        td_supported="imrphenomthm_native_supported",
    ),
    TorchNativeWaveform(
        approximant="IMRPhenomTP",
        component_flag="PYCBC_IMRPHENOMTP_NATIVE",
        module="imrphenomtp_waveform_torch",
        td_generator="imrphenomtp_td_torch",
        td_supported="imrphenomtp_native_supported",
    ),
    TorchNativeWaveform(
        approximant="IMRPhenomTPHM",
        component_flag="PYCBC_IMRPHENOMTPHM_NATIVE",
        module="imrphenomtphm_torch",
        td_generator="imrphenomtphm_td_torch",
        td_supported="imrphenomtphm_native_supported",
        td_modes_generator="imrphenomtphm_modes_torch",
        td_modes_supported="imrphenomtphm_modes_native_supported",
    ),
    TorchNativeWaveform(
        approximant="SEOBNRv4",
        component_flag="PYCBC_SEOBNRV4_NATIVE",
        module="seobnrv4phm_torch",
        td_generator="seobnrv4_td_torch",
        td_supported="seobnrv4_native_supported",
        fd_generator="seobnrv4_fd_torch",
        fd_supported="seobnrv4_native_supported",
    ),
    TorchNativeWaveform(
        approximant="SEOBNRv4HM",
        component_flag="PYCBC_SEOBNRV4HM_NATIVE",
        module="seobnrv4phm_torch",
        td_generator="seobnrv4hm_td_torch",
        td_supported="seobnrv4hm_native_supported",
    ),
    _fd_sequence_port(
        "SEOBNRv4P",
        "PYCBC_SEOBNRV4P_NATIVE",
        "seobnrv4phm_torch",
        "seobnrv4p",
        default_enabled=False,
        td_generator="seobnrv4p_td_torch",
        td_supported="seobnrv4p_native_supported",
        td_modes_generator="seobnrv4p_modes_torch",
        td_modes_supported="seobnrv4p_modes_native_supported",
        sequence_supported="seobnrv4p_sequence_native_supported",
    ),
    _fd_sequence_port(
        "SEOBNRv4_ROM",
        "PYCBC_SEOBNRV4_NATIVE",
        "seobnrv4_torch",
        "seobnrv4",
        fd_supported="seobnrv4_rom_native_supported",
        sequence_supported="seobnrv4_rom_sequence_native_supported",
    ),
    _fd_sequence_port(
        "SEOBNRv4_ROM_NRTidal",
        "PYCBC_SEOBNRV4_NATIVE",
        "seobnrv4_torch",
        "seobnrv4",
        fd_supported="seobnrv4_rom_native_supported",
        sequence_supported="seobnrv4_rom_sequence_native_supported",
    ),
    _fd_sequence_port(
        "SEOBNRv4_ROM_NRTidalv2",
        "PYCBC_SEOBNRV4_NATIVE",
        "seobnrv4_torch",
        "seobnrv4",
        fd_supported="seobnrv4_rom_native_supported",
        sequence_supported="seobnrv4_rom_sequence_native_supported",
    ),
    _fd_sequence_port(
        "SEOBNRv4_ROM_NRTidalv2_NSBH",
        "PYCBC_SEOBNRV4_NATIVE",
        "seobnrv4_torch",
        "seobnrv4",
        fd_supported="seobnrv4_rom_native_supported",
        sequence_supported="seobnrv4_rom_sequence_native_supported",
    ),
    _fd_sequence_port(
        "SEOBNRv4T_surrogate",
        "PYCBC_SEOBNRV4T_SURROGATE_NATIVE",
        "seobnrv4t_surrogate_torch",
        "seobnrv4t_surrogate",
    ),
    _fd_sequence_port(
        "SEOBNRv4HM_ROM",
        "PYCBC_SEOBNRV4HM_NATIVE",
        "seobnrv4hm_torch",
        "seobnrv4hm",
    ),
    _fd_sequence_port(
        "SEOBNRv4PHM",
        "PYCBC_SEOBNRV4PHM_NATIVE",
        "seobnrv4phm_torch",
        "seobnrv4phm",
        default_enabled=False,
        td_generator="seobnrv4phm_td_torch",
        td_supported="seobnrv4phm_native_supported",
        td_modes_generator="seobnrv4phm_modes_torch",
        td_modes_supported="seobnrv4phm_modes_native_supported",
    ),
    _fd_sequence_port(
        "SEOBNRv5_ROM",
        "PYCBC_SEOBNRV5_NATIVE",
        "seobnrv5_torch",
        "seobnrv5",
    ),
    _fd_sequence_port(
        "SEOBNRv5HM_ROM",
        "PYCBC_SEOBNRV5HM_NATIVE",
        "seobnrv5hm_torch",
        "seobnrv5hm",
    ),
    _fd_sequence_port(
        "SEOBNRv5_ROM_NRTidalv3",
        "PYCBC_SEOBNRV5_NATIVE",
        "seobnrv5_torch",
        "seobnrv5",
    ),
)


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
        for interface, (generator_field, supported_field) in (
            _INTERFACE_FIELDS.items()
        ):
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
