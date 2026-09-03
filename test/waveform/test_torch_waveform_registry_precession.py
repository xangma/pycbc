# Copyright (C) 2026  PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.

"""Focused registry and public-route tests for precessing waveforms."""

from importlib import import_module
import sys
from types import SimpleNamespace

import pytest

from pycbc.waveform import torch_waveform_registry as registry
from pycbc.waveform import waveform


_EXPECTED_PORTS = {
    "SpinTaylorF2": (
        "PYCBC_SPINTAYLORF2_NATIVE",
        "spintaylorf2_torch",
        ("fd", "sequence"),
        True,
    ),
    "IMRPhenomP": (
        "PYCBC_IMRPHENOMP_NATIVE", "imrphenomp_torch", ("fd", "sequence"), True
    ),
    "IMRPhenomPv2": (
        "PYCBC_IMRPHENOMPV2_NATIVE", "imrphenompv2_torch", ("fd", "sequence"), True
    ),
    "IMRPhenomPv2_NRTidal": (
        "PYCBC_IMRPHENOMPV2_NATIVE", "imrphenompv2_torch", ("fd", "sequence"), True
    ),
    "IMRPhenomPv2_NRTidalv2": (
        "PYCBC_IMRPHENOMPV2_NATIVE", "imrphenompv2_torch", ("fd", "sequence"), True
    ),
    "IMRPhenomXP": (
        "PYCBC_IMRPHENOMXP_NATIVE", "imrphenomxp_torch", ("fd", "sequence"), True
    ),
    "IMRPhenomXP_NRTidalv2": (
        "PYCBC_IMRPHENOMXP_NATIVE", "imrphenomxp_torch", ("fd", "sequence"), True
    ),
    "IMRPhenomXP_NRTidalv3": (
        "PYCBC_IMRPHENOMXP_NATIVE", "imrphenomxp_torch", ("fd", "sequence"), True
    ),
    "IMRPhenomXPHM": (
        "PYCBC_IMRPHENOMXPHM_NATIVE", "imrphenomxphm_torch", ("fd", "sequence"), True
    ),
    "IMRPhenomXO4a": (
        "PYCBC_IMRPHENOMXO4A_NATIVE", "imrphenomxo4a_torch", ("fd", "sequence"), True
    ),
    "IMRPhenomXPNR": (
        "PYCBC_IMRPHENOMXPNR_NATIVE",
        "imrphenomxpnr_waveform_torch",
        ("fd", "sequence"),
        True,
    ),
    "IMRPhenomPv3": (
        "PYCBC_IMRPHENOMPV3_NATIVE", "imrphenompv3_torch", ("fd", "sequence"), True
    ),
    "IMRPhenomPv3HM": (
        "PYCBC_IMRPHENOMPV3HM_NATIVE", "imrphenompv3_torch", ("fd", "sequence"), True
    ),
    "SpinTaylorT1": (
        "PYCBC_SPINTAYLORT1_NATIVE", "spintaylor_torch", ("td", "td_modes"), False
    ),
    "SpinTaylorT4": (
        "PYCBC_SPINTAYLORT4_NATIVE", "spintaylor_torch", ("td", "td_modes"), False
    ),
    "SpinTaylorT5": (
        "PYCBC_SPINTAYLORT5_NATIVE", "spintaylor_torch", ("td", "td_modes"), False
    ),
    "SpinTaylorT4Fourier": (
        "PYCBC_SPINTAYLORT4FOURIER_NATIVE", "spintaylor_fourier_torch", ("fd",), False
    ),
    "SpinTaylorT5Fourier": (
        "PYCBC_SPINTAYLORT5FOURIER_NATIVE", "spintaylor_fourier_torch", ("fd",), False
    ),
    "IMRPhenomTP": (
        "PYCBC_IMRPHENOMTP_NATIVE", "imrphenomtp_waveform_torch", ("td",), False
    ),
    "IMRPhenomTPHM": (
        "PYCBC_IMRPHENOMTPHM_NATIVE", "imrphenomtphm_torch", ("td", "td_modes"), False
    ),
    "SEOBNRv4": (
        "PYCBC_SEOBNRV4_NATIVE", "seobnrv4phm_torch", ("td", "fd"), False
    ),
    "SEOBNRv4HM": (
        "PYCBC_SEOBNRV4HM_NATIVE", "seobnrv4phm_torch", ("td",), False
    ),
    "SEOBNRv4P": (
        "PYCBC_SEOBNRV4P_NATIVE",
        "seobnrv4phm_torch",
        ("td", "td_modes", "fd", "sequence"),
        False,
    ),
    "SEOBNRv4PHM": (
        "PYCBC_SEOBNRV4PHM_NATIVE",
        "seobnrv4phm_torch",
        ("td", "td_modes", "fd", "sequence"),
        False,
    ),
}


def test_precession_registry_is_exact_and_importable():
    for approximant, expected in _EXPECTED_PORTS.items():
        component_flag, module, interfaces, default_enabled = expected
        port = registry.TORCH_NATIVE_WAVEFORMS[approximant]
        assert port.component_flag == component_flag
        assert port.module == module
        assert port.default_enabled is default_enabled
        assert registry._declared_interfaces(port) == interfaces
        implementation = import_module(f"pycbc.waveform.{module}")
        for interface in interfaces:
            generator = getattr(port, f"{interface}_generator")
            supported = getattr(port, f"{interface}_supported")
            assert callable(getattr(implementation, generator))
            assert callable(getattr(implementation, supported))


@pytest.mark.parametrize(
    ("approximant", "module", "generator"),
    (
        (
            "IMRPhenomXP",
            "pycbc.waveform.imrphenomxp_torch",
            "imrphenomxp_fd_batch",
        ),
        (
            "IMRPhenomXPHM",
            "pycbc.waveform.imrphenomxphm_torch",
            "imrphenomxphm_fd_batch",
        ),
    ),
)
def test_explicit_batch_routes(monkeypatch, approximant, module, generator):
    sentinel = object()

    def generate(**params):
        assert params == {"mass1": 20.0}
        return sentinel

    monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{generator: generate}))
    assert waveform.get_fd_waveform_batch(approximant, mass1=20.0) is sentinel


def test_seobnrv4phm_frequency_domain_reference_route(monkeypatch):
    sentinel = object()

    def generate(*, use_torch, **params):
        assert not use_torch
        assert params == {"approximant": "SEOBNRv4PHM"}
        return sentinel

    module = "pycbc.waveform.seobnrv4phm_torch"
    monkeypatch.setitem(
        sys.modules,
        module,
        SimpleNamespace(seobnrv4phm_fd_from_td=generate),
    )
    assert waveform._lalsim_fd_waveform(approximant="SEOBNRv4PHM") is sentinel
