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
from enum import Enum
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
_DEVICE_ORDER = ("cpu", "cuda", "mps")
_CPU_CUDA = ("cpu", "cuda")


class TorchWaveformReference(str, Enum):
    """Reference status for one native public waveform interface."""

    LAL_REFERENCE = "LAL reference"
    NATIVE_EXTENSION = "native extension"


class TorchWaveformFallback(str, Enum):
    """Fallback route used when a native interface is not selected."""

    STANDARD_LAL = "standard LAL path"
    CPU_LAL_ADAPTER = "CPU/LAL adapter"
    NO_LAL_EQUIVALENT = "no equivalent LAL interface"


class TorchWaveformDefault(str, Enum):
    """Default-selection policy for one native waveform port."""

    DEFAULT_ON = "attempted by default"
    PREDICATE_GUARDED = "attempted by default (predicate guarded)"
    OPT_IN = "opt-in"


class TorchNativeWaveformUnavailable(ImportError):
    """A Torch-native request has no usable LALSimulation fallback."""


@dataclass(frozen=True)
class TorchWaveformInterfaceCapability:
    """Audited metadata for one declared native public interface."""

    interface: str
    reference: TorchWaveformReference
    fallback: TorchWaveformFallback
    eligible_devices: tuple[str, ...]
    default_enabled: Optional[bool] = None


@dataclass(frozen=True)
class TorchWaveformCapability:
    """One deterministic, documentation-ready registry capability row."""

    approximant: str
    interface: str
    default_policy: TorchWaveformDefault
    eligible_devices: tuple[str, ...]
    reference: TorchWaveformReference
    fallback: TorchWaveformFallback
    component_flag: str


@dataclass(frozen=True)
class TorchNativeWaveform:
    """Entry points and feature switch for one native approximant."""

    approximant: str
    component_flag: str
    module: str
    interface_capabilities: tuple[TorchWaveformInterfaceCapability, ...]
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


def _lal_interfaces(
    *interfaces,
    devices=_DEVICE_ORDER,
    default_enabled=None,
):
    return tuple(
        TorchWaveformInterfaceCapability(
            interface=interface,
            reference=TorchWaveformReference.LAL_REFERENCE,
            fallback=TorchWaveformFallback.STANDARD_LAL,
            eligible_devices=devices,
            default_enabled=default_enabled,
        )
        for interface in interfaces
    )


def _lal_adapter_interfaces(*interfaces, devices=_DEVICE_ORDER):
    return tuple(
        TorchWaveformInterfaceCapability(
            interface,
            TorchWaveformReference.LAL_REFERENCE,
            TorchWaveformFallback.CPU_LAL_ADAPTER,
            devices,
        )
        for interface in interfaces
    )


def _native_extension_interfaces(*interfaces, devices=_DEVICE_ORDER):
    return tuple(
        TorchWaveformInterfaceCapability(
            interface,
            TorchWaveformReference.NATIVE_EXTENSION,
            TorchWaveformFallback.NO_LAL_EQUIVALENT,
            devices,
        )
        for interface in interfaces
    )


def _fd_sequence_port(
    approximant,
    component_flag,
    module,
    stem,
    *,
    interface_capabilities,
    default_enabled=False,
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
        interface_capabilities=interface_capabilities,
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
        "EccentricFD",
        "PYCBC_ECCENTRICFD_NATIVE",
        "eccentricfd_torch",
        "eccentricfd",
        interface_capabilities=(
            _lal_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
        default_enabled=True,
        default_supported="eccentricfd_default_native_supported",
    ),
    _fd_sequence_port(
        "TaylorF2",
        "PYCBC_TAYLORF2_NATIVE",
        "taylorf2_torch",
        "taylorf2",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "TaylorF2NLTides",
        "PYCBC_TAYLORF2NLTIDES_NATIVE",
        "taylorf2nltides_torch",
        "taylorf2nltides",
        interface_capabilities=(
            _lal_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "TaylorF2RedSpin",
        "PYCBC_TAYLORF2REDSPIN_NATIVE",
        "taylorf2redspin_torch",
        "taylorf2redspin",
        interface_capabilities=(
            _lal_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "TaylorF2RedSpinTidal",
        "PYCBC_TAYLORF2REDSPINTIDAL_NATIVE",
        "taylorf2redspin_torch",
        "taylorf2redspin",
        interface_capabilities=(
            _lal_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "TaylorF2Ecc",
        "PYCBC_TAYLORF2ECC_NATIVE",
        "taylorf2ecc_torch",
        "taylorf2ecc",
        interface_capabilities=(
            _lal_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "SpinTaylorF2",
        "PYCBC_SPINTAYLORF2_NATIVE",
        "spintaylorf2_torch",
        "spintaylorf2",
        interface_capabilities=(
            _lal_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
        fd_generator="spintaylorf2_torch",
        default_enabled=True,
        default_supported="spintaylorf2_default_native_supported",
    ),
    _fd_sequence_port(
        "IMRPhenomA",
        "PYCBC_IMRPHENOMA_NATIVE",
        "imrphenomab_torch",
        "imrphenomab",
        interface_capabilities=(
            _lal_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
        default_enabled=True,
        default_supported="imrphenomab_default_native_supported",
    ),
    _fd_sequence_port(
        "IMRPhenomB",
        "PYCBC_IMRPHENOMB_NATIVE",
        "imrphenomab_torch",
        "imrphenomab",
        interface_capabilities=(
            _lal_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
        default_enabled=True,
        default_supported="imrphenomab_default_native_supported",
    ),
    _fd_sequence_port(
        "IMRPhenomC",
        "PYCBC_IMRPHENOMC_NATIVE",
        "imrphenomc_torch",
        "imrphenomc",
        interface_capabilities=(
            _lal_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
        default_enabled=True,
        default_supported="imrphenomc_default_native_supported",
    ),
    _fd_sequence_port(
        "IMRPhenomD",
        "PYCBC_IMRPHENOMD_NATIVE",
        "imrphenomd_torch",
        "imrphenomd",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomD_NRTidal",
        "PYCBC_IMRPHENOMD_NATIVE",
        "imrphenomd_torch",
        "imrphenomd",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomD_NRTidalv2",
        "PYCBC_IMRPHENOMD_NATIVE",
        "imrphenomd_torch",
        "imrphenomd",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomNSBH",
        "PYCBC_IMRPHENOMNSBH_NATIVE",
        "imrphenomnsbh_torch",
        "imrphenomnsbh",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomP",
        "PYCBC_IMRPHENOMP_NATIVE",
        "imrphenomp_torch",
        "imrphenomp",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomPv2",
        "PYCBC_IMRPHENOMPV2_NATIVE",
        "imrphenompv2_torch",
        "imrphenompv2",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomPv2_NRTidal",
        "PYCBC_IMRPHENOMPV2_NATIVE",
        "imrphenompv2_torch",
        "imrphenompv2",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomPv2_NRTidalv2",
        "PYCBC_IMRPHENOMPV2_NATIVE",
        "imrphenompv2_torch",
        "imrphenompv2",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomXAS",
        "PYCBC_IMRPHENOMXAS_NATIVE",
        "imrphenomxas_torch",
        "imrphenomxas",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomXAS_NRTidalv2",
        "PYCBC_IMRPHENOMXAS_NATIVE",
        "imrphenomxas_torch",
        "imrphenomxas",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomXAS_NRTidalv3",
        "PYCBC_IMRPHENOMXAS_NATIVE",
        "imrphenomxas_torch",
        "imrphenomxas",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomXP",
        "PYCBC_IMRPHENOMXP_NATIVE",
        "imrphenomxp_torch",
        "imrphenomxp",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomXP_NRTidalv2",
        "PYCBC_IMRPHENOMXP_NATIVE",
        "imrphenomxp_torch",
        "imrphenomxp",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomXP_NRTidalv3",
        "PYCBC_IMRPHENOMXP_NATIVE",
        "imrphenomxp_torch",
        "imrphenomxp",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomXHM",
        "PYCBC_IMRPHENOMXHM_NATIVE",
        "imrphenomxhm_torch",
        "imrphenomxhm",
        interface_capabilities=(
            _lal_interfaces("fd", "sequence")
            + _lal_adapter_interfaces("fd_modes")
        ),
        default_enabled=True,
        fd_supported="imrphenomxhm_fd_native_supported",
        fd_modes_generator="imrphenomxhm_modes_torch",
        fd_modes_supported="imrphenomxhm_modes_native_supported",
    ),
    _fd_sequence_port(
        "IMRPhenomXPHM",
        "PYCBC_IMRPHENOMXPHM_NATIVE",
        "imrphenomxphm_torch",
        "imrphenomxphm",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomXO4a",
        "PYCBC_IMRPHENOMXO4A_NATIVE",
        "imrphenomxo4a_torch",
        "imrphenomxo4a",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomXPNR",
        "PYCBC_IMRPHENOMXPNR_NATIVE",
        "imrphenomxpnr_waveform_torch",
        "imrphenomxpnr",
        interface_capabilities=_lal_interfaces(
            "fd", "sequence", devices=_CPU_CUDA
        ),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomHM",
        "PYCBC_IMRPHENOMHM_NATIVE",
        "imrphenomhm_torch",
        "imrphenomhm",
        interface_capabilities=(
            _lal_interfaces("fd", "sequence")
            + _lal_adapter_interfaces("fd_modes")
        ),
        default_enabled=True,
        fd_modes_generator="imrphenomhm_modes_torch",
        fd_modes_supported="imrphenomhm_modes_native_supported",
    ),
    _fd_sequence_port(
        "IMRPhenomPv3",
        "PYCBC_IMRPHENOMPV3_NATIVE",
        "imrphenompv3_torch",
        "imrphenompv3",
        interface_capabilities=(
            _lal_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "IMRPhenomPv3HM",
        "PYCBC_IMRPHENOMPV3HM_NATIVE",
        "imrphenompv3_torch",
        "imrphenompv3hm",
        interface_capabilities=(
            _lal_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
        default_enabled=True,
    ),
    TorchNativeWaveform(
        approximant="TaylorT1",
        component_flag="PYCBC_TAYLORT1_NATIVE",
        module="taylort1_torch",
        interface_capabilities=_lal_interfaces("td", "td_modes"),
        td_generator="taylort1_td_torch",
        td_supported="taylort1_native_supported",
        td_modes_generator="taylort1_modes_torch",
        td_modes_supported="taylort1_modes_native_supported",
    ),
    TorchNativeWaveform(
        approximant="TaylorT2",
        component_flag="PYCBC_TAYLORT2_NATIVE",
        module="taylort2_torch",
        interface_capabilities=_lal_interfaces("td", "td_modes"),
        td_generator="taylort2_td_torch",
        td_supported="taylort2_native_supported",
        td_modes_generator="taylort2_modes_torch",
        td_modes_supported="taylort2_modes_native_supported",
    ),
    TorchNativeWaveform(
        approximant="TaylorT3",
        component_flag="PYCBC_TAYLORT3_NATIVE",
        module="taylort3_torch",
        interface_capabilities=_lal_interfaces("td", "td_modes"),
        td_generator="taylort3_td_torch",
        td_supported="taylort3_native_supported",
        td_modes_generator="taylort3_modes_torch",
        td_modes_supported="taylort3_modes_native_supported",
    ),
    TorchNativeWaveform(
        approximant="TaylorT4",
        component_flag="PYCBC_TAYLORT4_NATIVE",
        module="taylort4_torch",
        interface_capabilities=_lal_interfaces("td", "td_modes"),
        default_enabled=True,
        default_supported="taylort4_default_native_supported",
        td_generator="taylort4_td_torch",
        td_supported="taylort4_native_supported",
        td_modes_generator="taylort4_modes_torch",
        td_modes_supported="taylort4_modes_native_supported",
    ),
    TorchNativeWaveform(
        approximant="SpinTaylorT1",
        component_flag="PYCBC_SPINTAYLORT1_NATIVE",
        module="spintaylor_torch",
        interface_capabilities=_lal_interfaces("td", "td_modes"),
        td_generator="spintaylor_t1_td_torch",
        td_supported="spintaylor_t1_native_supported",
        td_modes_generator="spintaylor_t1_modes_torch",
        td_modes_supported="spintaylor_t1_modes_native_supported",
    ),
    TorchNativeWaveform(
        approximant="SpinTaylorT4",
        component_flag="PYCBC_SPINTAYLORT4_NATIVE",
        module="spintaylor_torch",
        interface_capabilities=_lal_interfaces("td", "td_modes"),
        td_generator="spintaylor_t4_td_torch",
        td_supported="spintaylor_t4_native_supported",
        td_modes_generator="spintaylor_t4_modes_torch",
        td_modes_supported="spintaylor_t4_modes_native_supported",
    ),
    TorchNativeWaveform(
        approximant="SpinTaylorT5",
        component_flag="PYCBC_SPINTAYLORT5_NATIVE",
        module="spintaylor_torch",
        interface_capabilities=_lal_interfaces("td", "td_modes"),
        td_generator="spintaylor_t5_td_torch",
        td_supported="spintaylor_t5_native_supported",
        td_modes_generator="spintaylor_t5_modes_torch",
        td_modes_supported="spintaylor_t5_modes_native_supported",
    ),
    TorchNativeWaveform(
        approximant="SpinTaylorT4Fourier",
        component_flag="PYCBC_SPINTAYLORT4FOURIER_NATIVE",
        module="spintaylor_fourier_torch",
        interface_capabilities=_lal_interfaces("fd", devices=("cpu",)),
        fd_generator="spintaylor_t4_fourier_fd_torch",
        fd_supported="spintaylor_t4_fourier_native_supported",
    ),
    TorchNativeWaveform(
        approximant="SpinTaylorT5Fourier",
        component_flag="PYCBC_SPINTAYLORT5FOURIER_NATIVE",
        module="spintaylor_fourier_torch",
        interface_capabilities=_lal_interfaces("fd", devices=("cpu",)),
        fd_generator="spintaylor_t5_fourier_fd_torch",
        fd_supported="spintaylor_t5_fourier_native_supported",
    ),
    TorchNativeWaveform(
        approximant="IMRPhenomT",
        component_flag="PYCBC_IMRPHENOMT_NATIVE",
        module="imrphenomt_torch",
        interface_capabilities=_lal_interfaces("td"),
        td_generator="imrphenomt_td_torch",
        td_supported="imrphenomt_native_supported",
    ),
    TorchNativeWaveform(
        approximant="IMRPhenomTHM",
        component_flag="PYCBC_IMRPHENOMTHM_NATIVE",
        module="imrphenomthm_torch",
        interface_capabilities=_lal_interfaces("td"),
        default_enabled=True,
        default_supported="imrphenomthm_default_native_supported",
        td_generator="imrphenomthm_td_torch",
        td_supported="imrphenomthm_native_supported",
    ),
    TorchNativeWaveform(
        approximant="IMRPhenomTP",
        component_flag="PYCBC_IMRPHENOMTP_NATIVE",
        module="imrphenomtp_waveform_torch",
        interface_capabilities=_lal_interfaces("td"),
        td_generator="imrphenomtp_td_torch",
        td_supported="imrphenomtp_native_supported",
    ),
    TorchNativeWaveform(
        approximant="IMRPhenomTPHM",
        component_flag="PYCBC_IMRPHENOMTPHM_NATIVE",
        module="imrphenomtphm_torch",
        interface_capabilities=_lal_interfaces("td", "td_modes"),
        td_generator="imrphenomtphm_td_torch",
        td_supported="imrphenomtphm_native_supported",
        td_modes_generator="imrphenomtphm_modes_torch",
        td_modes_supported="imrphenomtphm_modes_native_supported",
    ),
    _fd_sequence_port(
        "EOBNRv2_ROM",
        "PYCBC_EOBNRV2_NATIVE",
        "eobnrv2_torch",
        "eobnrv2",
        interface_capabilities=(
            _lal_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
    ),
    _fd_sequence_port(
        "EOBNRv2HM_ROM",
        "PYCBC_EOBNRV2_NATIVE",
        "eobnrv2_torch",
        "eobnrv2",
        interface_capabilities=(
            _lal_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
    ),
    TorchNativeWaveform(
        approximant="SEOBNRv4",
        component_flag="PYCBC_SEOBNRV4_NATIVE",
        module="seobnrv4phm_torch",
        interface_capabilities=(
            _lal_interfaces("td") + _lal_adapter_interfaces("fd")
        ),
        td_generator="seobnrv4_td_torch",
        td_supported="seobnrv4_native_supported",
        fd_generator="seobnrv4_fd_torch",
        fd_supported="seobnrv4_native_supported",
    ),
    TorchNativeWaveform(
        approximant="SEOBNRv4HM",
        component_flag="PYCBC_SEOBNRV4HM_NATIVE",
        module="seobnrv4phm_torch",
        interface_capabilities=_lal_interfaces("td", devices=("cpu",)),
        td_generator="seobnrv4hm_td_torch",
        td_supported="seobnrv4hm_native_supported",
    ),
    _fd_sequence_port(
        "SEOBNRv4P",
        "PYCBC_SEOBNRV4P_NATIVE",
        "seobnrv4phm_torch",
        "seobnrv4p",
        interface_capabilities=(
            _lal_interfaces("td", "td_modes")
            + _lal_adapter_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
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
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
        fd_supported="seobnrv4_rom_native_supported",
        sequence_supported="seobnrv4_rom_sequence_native_supported",
    ),
    _fd_sequence_port(
        "SEOBNRv4_ROM_NRTidal",
        "PYCBC_SEOBNRV4_NATIVE",
        "seobnrv4_torch",
        "seobnrv4",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
        fd_supported="seobnrv4_rom_native_supported",
        sequence_supported="seobnrv4_rom_sequence_native_supported",
    ),
    _fd_sequence_port(
        "SEOBNRv4_ROM_NRTidalv2",
        "PYCBC_SEOBNRV4_NATIVE",
        "seobnrv4_torch",
        "seobnrv4",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
        fd_supported="seobnrv4_rom_native_supported",
        sequence_supported="seobnrv4_rom_sequence_native_supported",
    ),
    _fd_sequence_port(
        "SEOBNRv4_ROM_NRTidalv2_NSBH",
        "PYCBC_SEOBNRV4_NATIVE",
        "seobnrv4_torch",
        "seobnrv4",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
        fd_supported="seobnrv4_rom_native_supported",
        sequence_supported="seobnrv4_rom_sequence_native_supported",
    ),
    _fd_sequence_port(
        "SEOBNRv4T_surrogate",
        "PYCBC_SEOBNRV4T_SURROGATE_NATIVE",
        "seobnrv4t_surrogate_torch",
        "seobnrv4t_surrogate",
        interface_capabilities=_lal_interfaces(
            "fd", "sequence", devices=_CPU_CUDA
        ),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "SEOBNRv4HM_ROM",
        "PYCBC_SEOBNRV4HM_NATIVE",
        "seobnrv4hm_torch",
        "seobnrv4hm",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "SEOBNRv5_ROM",
        "PYCBC_SEOBNRV5_NATIVE",
        "seobnrv5_torch",
        "seobnrv5",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "SEOBNRv5_ROM_NRTidalv3",
        "PYCBC_SEOBNRV5_NATIVE",
        "seobnrv5_torch",
        "seobnrv5",
        interface_capabilities=_lal_interfaces(
            "fd", "sequence", devices=_CPU_CUDA
        ),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "SEOBNRv5HM_ROM",
        "PYCBC_SEOBNRV5HM_NATIVE",
        "seobnrv5hm_torch",
        "seobnrv5hm",
        interface_capabilities=_lal_interfaces("fd", "sequence"),
        default_enabled=True,
    ),
    _fd_sequence_port(
        "SEOBNRv4PHM",
        "PYCBC_SEOBNRV4PHM_NATIVE",
        "seobnrv4phm_torch",
        "seobnrv4phm",
        interface_capabilities=(
            _lal_interfaces("td", "td_modes")
            + _lal_adapter_interfaces("fd")
            + _native_extension_interfaces("sequence")
        ),
        td_generator="seobnrv4phm_td_torch",
        td_supported="seobnrv4phm_native_supported",
        td_modes_generator="seobnrv4phm_modes_torch",
        td_modes_supported="seobnrv4phm_modes_native_supported",
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
        declared = set(_declared_interfaces(port))
        if not declared:
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

        capabilities = port.interface_capabilities
        capability_interfaces = [item.interface for item in capabilities]
        if len(set(capability_interfaces)) != len(capability_interfaces):
            raise RuntimeError(
                f"duplicate interface metadata for {port.approximant!r}"
            )
        if set(capability_interfaces) != declared:
            raise RuntimeError(
                f"interface metadata does not cover every declared interface "
                f"for {port.approximant!r}"
            )
        valid_routes = {
            (
                TorchWaveformReference.LAL_REFERENCE,
                TorchWaveformFallback.STANDARD_LAL,
            ),
            (
                TorchWaveformReference.LAL_REFERENCE,
                TorchWaveformFallback.CPU_LAL_ADAPTER,
            ),
            (
                TorchWaveformReference.NATIVE_EXTENSION,
                TorchWaveformFallback.NO_LAL_EQUIVALENT,
            ),
        }
        for item in capabilities:
            if item.default_enabled is not None and not isinstance(
                item.default_enabled, bool
            ):
                raise RuntimeError(
                    f"invalid interface default for "
                    f"{port.approximant!r} {item.interface!r}"
                )
            if (item.reference, item.fallback) not in valid_routes:
                raise RuntimeError(
                    f"invalid reference/fallback metadata for "
                    f"{port.approximant!r} {item.interface!r}"
                )
            devices = item.eligible_devices
            if not devices or tuple(
                device for device in _DEVICE_ORDER if device in devices
            ) != devices:
                raise RuntimeError(
                    f"invalid eligible device scope for "
                    f"{port.approximant!r} {item.interface!r}"
                )


_validate_ports(_PORTS)

TORCH_NATIVE_WAVEFORMS: Mapping[str, TorchNativeWaveform] = MappingProxyType(
    {port.approximant: port for port in _PORTS}
)


def _interface_default_enabled(port, interface):
    """Resolve an interface override, inheriting the port default."""
    for capability in port.interface_capabilities:
        if capability.interface == interface:
            if capability.default_enabled is None:
                return port.default_enabled
            return capability.default_enabled
    raise RuntimeError(
        f"missing interface metadata for {port.approximant!r} {interface!r}"
    )


def torch_waveform_capabilities():
    """Return deterministic capability rows derived from the registry.

    Eligible devices are Torch output targets for at least one supported
    request. They do not assert fully device-resident internals or hardware
    availability. The per-interface support predicate remains authoritative
    for a particular parameter set.
    """
    rows = []
    for approximant in sorted(TORCH_NATIVE_WAVEFORMS):
        port = TORCH_NATIVE_WAVEFORMS[approximant]
        interface_capabilities = {
            item.interface: item for item in port.interface_capabilities
        }
        for interface in _declared_interfaces(port):
            metadata = interface_capabilities[interface]
            if not _interface_default_enabled(port, interface):
                default_policy = TorchWaveformDefault.OPT_IN
            elif port.default_supported is not None:
                default_policy = TorchWaveformDefault.PREDICATE_GUARDED
            else:
                default_policy = TorchWaveformDefault.DEFAULT_ON
            rows.append(
                TorchWaveformCapability(
                    approximant=approximant,
                    interface=interface,
                    default_policy=default_policy,
                    eligible_devices=metadata.eligible_devices,
                    reference=metadata.reference,
                    fallback=metadata.fallback,
                    component_flag=port.component_flag,
                )
            )
    return tuple(rows)


def render_torch_waveform_capabilities():
    """Render the registry capability ledger as deterministic reST."""
    interface_labels = {
        "td": "TD",
        "td_modes": "TD modes",
        "fd": "FD",
        "fd_modes": "FD modes",
        "sequence": "FD sequence",
    }
    lines = [
        ".. list-table:: Registered Torch-native waveform interfaces",
        "   :header-rows: 1",
        "",
        "   * - Approximant",
        "     - Interface",
        "     - Default",
        "     - Eligible Torch targets",
        "     - Reference",
        "     - Fallback",
        "     - Component flag",
    ]
    for row in torch_waveform_capabilities():
        lines.extend(
            (
                f"   * - ``{row.approximant}``",
                f"     - {interface_labels[row.interface]}",
                f"     - {row.default_policy.value}",
                "     - "
                + ", ".join(
                    device.upper() for device in row.eligible_devices
                ),
                f"     - {row.reference.value}",
                f"     - {row.fallback.value}",
                f"     - ``{row.component_flag}``",
            )
        )
    return "\n".join(lines) + "\n"


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


def _raise_native_unavailable(interface, approximant, reason):
    raise TorchNativeWaveformUnavailable(
        "Cannot generate approximant "
        f"{approximant!r} through the Torch-native {interface!r} interface: "
        f"{reason}, and lalsimulation is unavailable."
    )


def try_torch_native_waveform(
    interface,
    params,
    *,
    lalsimulation_available=True,
):
    """Run an enabled and supported native port, or return ``None``.

    The caller owns scheme selection. When ``lalsimulation_available`` is
    false, a compatible native port is attempted even if its normal policy is
    opt-in. Parameter-support and default-readiness predicates remain
    authoritative. Explicit environment opt-outs are also honored, and a
    clear error replaces an impossible LALSimulation fallback.
    """
    try:
        generator_field, supported_field = _INTERFACE_FIELDS[interface]
    except KeyError as exc:
        raise ValueError(f"unknown waveform interface {interface!r}") from exc

    approximant = params.get("approximant")
    port = TORCH_NATIVE_WAVEFORMS.get(approximant)
    if port is None:
        if not lalsimulation_available:
            _raise_native_unavailable(
                interface,
                approximant,
                "no native implementation is registered",
            )
        return None

    generator_name = getattr(port, generator_field)
    supported_name = getattr(port, supported_field)
    if generator_name is None:
        if not lalsimulation_available:
            _raise_native_unavailable(
                interface,
                approximant,
                "no native implementation is registered for this interface",
            )
        return None

    override = torch_native_override(port.component_flag)
    if override is None:
        enabled = (
            _interface_default_enabled(port, interface)
            or not lalsimulation_available
        )
    else:
        enabled = override
    if not enabled:
        if not lalsimulation_available:
            _raise_native_unavailable(
                interface,
                approximant,
                f"the native port is explicitly disabled by "
                f"{port.component_flag} or a global native-port switch",
            )
        return None

    try:
        implementation = import_module(f".{port.module}", package=__package__)
    except (ImportError, ModuleNotFoundError) as exc:
        if not lalsimulation_available:
            _raise_native_unavailable(
                interface,
                approximant,
                f"the native port module {port.module!r} "
                f"is unavailable: {exc}",
            )
        return None
    if not getattr(implementation, supported_name)(params):
        if not lalsimulation_available:
            _raise_native_unavailable(
                interface,
                approximant,
                "the requested parameters are outside its supported envelope",
            )
        return None
    if (
        override is None
        and port.default_supported is not None
        and not getattr(implementation, port.default_supported)(params)
    ):
        if not lalsimulation_available:
            _raise_native_unavailable(
                interface,
                approximant,
                "the request does not satisfy the native port's "
                "default-readiness guard",
            )
        return None

    # When running under TorchScheme on CPU without an explicitly configured
    # thread count, clamp intra-op threads to 1 to avoid OpenMP barrier lock
    # contention during single waveform generation, and restore on exit.
    clamp_threads = False
    prev_threads = None
    try:
        from pycbc import scheme as _scheme
        state = getattr(_scheme.mgr, "state", None)
        if (
            isinstance(state, getattr(_scheme, "TorchScheme", ()))
            and getattr(state, "torch_device", None) is not None
            and state.torch_device.type == "cpu"
            and getattr(state, "num_threads", None) is None
        ):
            import torch
            prev_threads = torch.get_num_threads()
            if prev_threads != 1:
                torch.set_num_threads(1)
                clamp_threads = True
    except Exception:
        pass

    try:
        if interface == "fd" and approximant in (
            "IMRPhenomXAS",
            "IMRPhenomD",
            "IMRPhenomXHM",
            "IMRPhenomXP",
        ):
            try:
                from pycbc import scheme as _scheme
                state = getattr(_scheme.mgr, "state", None)
                if isinstance(state, getattr(_scheme, "TorchScheme", ())):
                    device = getattr(state, "torch_device", None)
                    if (
                        device is not None
                        and getattr(device, "type", None) == "cuda"
                    ):
                        import pycbc.waveform.triton as _triton
                        if getattr(
                            _triton, "is_triton_available", lambda: False
                        )():
                            fn = getattr(
                                _triton,
                                f"{approximant.lower()}_triton_fd",
                                None,
                            )
                            if fn is None:
                                try:
                                    mod = import_module(
                                        f".triton.{approximant.lower()}",
                                        package=__package__,
                                    )
                                    fn = getattr(
                                        mod,
                                        f"{approximant.lower()}_triton_fd",
                                        None,
                                    )
                                except Exception:
                                    fn = None
                            if fn is not None:
                                return fn(**params)

            except Exception:
                pass

        return getattr(implementation, generator_name)(**params)
    finally:
        if clamp_threads and prev_threads is not None:

            try:
                import torch
                torch.set_num_threads(prev_threads)
            except Exception:
                pass


__all__ = (
    "TORCH_NATIVE_WAVEFORMS",
    "TorchNativeWaveform",
    "TorchNativeWaveformUnavailable",
    "TorchWaveformCapability",
    "TorchWaveformDefault",
    "TorchWaveformFallback",
    "TorchWaveformInterfaceCapability",
    "TorchWaveformReference",
    "native_approximants",
    "render_torch_waveform_capabilities",
    "torch_waveform_capabilities",
    "try_torch_native_waveform",
)
