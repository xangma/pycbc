# Copyright (C) 2026  PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.

"""Focused registry tests for the Torch-native aligned-spin models."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from pycbc.waveform import torch_waveform_registry as registry


_EXPECTED_PORTS = {
    "TaylorF2": (
        "PYCBC_TAYLORF2_NATIVE",
        "taylorf2_torch",
        ("fd", "sequence"),
    ),
    "TaylorF2NLTides": (
        "PYCBC_TAYLORF2NLTIDES_NATIVE",
        "taylorf2nltides_torch",
        ("fd", "sequence"),
    ),
    "TaylorF2RedSpin": (
        "PYCBC_TAYLORF2REDSPIN_NATIVE",
        "taylorf2redspin_torch",
        ("fd", "sequence"),
    ),
    "TaylorF2RedSpinTidal": (
        "PYCBC_TAYLORF2REDSPINTIDAL_NATIVE",
        "taylorf2redspin_torch",
        ("fd", "sequence"),
    ),
    "TaylorF2Ecc": (
        "PYCBC_TAYLORF2ECC_NATIVE",
        "taylorf2ecc_torch",
        ("fd", "sequence"),
    ),
    "IMRPhenomA": (
        "PYCBC_IMRPHENOMA_NATIVE",
        "imrphenomab_torch",
        ("fd", "sequence"),
    ),
    "IMRPhenomB": (
        "PYCBC_IMRPHENOMB_NATIVE",
        "imrphenomab_torch",
        ("fd", "sequence"),
    ),
    "IMRPhenomC": (
        "PYCBC_IMRPHENOMC_NATIVE",
        "imrphenomc_torch",
        ("fd", "sequence"),
    ),
    "IMRPhenomD": (
        "PYCBC_IMRPHENOMD_NATIVE",
        "imrphenomd_torch",
        ("fd", "sequence"),
    ),
    "IMRPhenomD_NRTidal": (
        "PYCBC_IMRPHENOMD_NATIVE",
        "imrphenomd_torch",
        ("fd", "sequence"),
    ),
    "IMRPhenomD_NRTidalv2": (
        "PYCBC_IMRPHENOMD_NATIVE",
        "imrphenomd_torch",
        ("fd", "sequence"),
    ),
    "IMRPhenomNSBH": (
        "PYCBC_IMRPHENOMNSBH_NATIVE",
        "imrphenomnsbh_torch",
        ("fd", "sequence"),
    ),
    "IMRPhenomXAS": (
        "PYCBC_IMRPHENOMXAS_NATIVE",
        "imrphenomxas_torch",
        ("fd", "sequence"),
    ),
    "IMRPhenomXAS_NRTidalv2": (
        "PYCBC_IMRPHENOMXAS_NATIVE",
        "imrphenomxas_torch",
        ("fd", "sequence"),
    ),
    "IMRPhenomXAS_NRTidalv3": (
        "PYCBC_IMRPHENOMXAS_NATIVE",
        "imrphenomxas_torch",
        ("fd", "sequence"),
    ),
    "IMRPhenomT": (
        "PYCBC_IMRPHENOMT_NATIVE",
        "imrphenomt_torch",
        ("td",),
    ),
    "SEOBNRv4_ROM": (
        "PYCBC_SEOBNRV4_NATIVE",
        "seobnrv4_torch",
        ("fd", "sequence"),
    ),
    "SEOBNRv4_ROM_NRTidal": (
        "PYCBC_SEOBNRV4_NATIVE",
        "seobnrv4_torch",
        ("fd", "sequence"),
    ),
    "SEOBNRv4_ROM_NRTidalv2": (
        "PYCBC_SEOBNRV4_NATIVE",
        "seobnrv4_torch",
        ("fd", "sequence"),
    ),
    "SEOBNRv4_ROM_NRTidalv2_NSBH": (
        "PYCBC_SEOBNRV4_NATIVE",
        "seobnrv4_torch",
        ("fd", "sequence"),
    ),
    "SEOBNRv4T_surrogate": (
        "PYCBC_SEOBNRV4T_SURROGATE_NATIVE",
        "seobnrv4t_surrogate_torch",
        ("fd", "sequence"),
    ),
    "SEOBNRv5_ROM": (
        "PYCBC_SEOBNRV5_NATIVE",
        "seobnrv5_torch",
        ("fd", "sequence"),
    ),
    "SEOBNRv5_ROM_NRTidalv3": (
        "PYCBC_SEOBNRV5_NATIVE",
        "seobnrv5_torch",
        ("fd", "sequence"),
    ),
}


def test_registry_contains_aligned_spin_models():
    expected = set(_EXPECTED_PORTS)
    assert expected <= set(registry.TORCH_NATIVE_WAVEFORMS)
    for approximant, (_, _, interfaces) in _EXPECTED_PORTS.items():
        for interface in interfaces:
            assert approximant in registry.native_approximants(interface)


def test_registered_modules_are_importable():
    """Keep registry additions tied to callable code present in this PR."""
    for approximant in _EXPECTED_PORTS:
        port = registry.TORCH_NATIVE_WAVEFORMS[approximant]
        expected_flag, expected_module, interfaces = _EXPECTED_PORTS[
            approximant
        ]
        assert port.component_flag == expected_flag
        assert port.module == expected_module
        implementation = import_module(f"pycbc.waveform.{port.module}")
        for interface in interfaces:
            generator = getattr(port, f"{interface}_generator")
            supported = getattr(port, f"{interface}_supported")
            assert callable(getattr(implementation, generator))
            assert callable(getattr(implementation, supported))
        if port.default_supported is not None:
            assert callable(getattr(implementation, port.default_supported))


def test_unregistered_dispatch_uses_lal_fallback():
    params = {"approximant": "NotRegistered"}
    assert registry.try_torch_native_waveform("fd", params) is None


def test_registered_dispatch(monkeypatch):
    port = registry.TorchNativeWaveform(
        approximant="TestWaveform",
        component_flag="test",
        module="test_waveform",
        default_enabled=True,
        fd_generator="generate",
        fd_supported="supported",
    )
    implementation = SimpleNamespace(
        supported=lambda params: params["mass1"] > 0,
        generate=lambda **params: params["mass1"],
    )
    monkeypatch.setattr(
        registry, "TORCH_NATIVE_WAVEFORMS", {port.approximant: port}
    )
    monkeypatch.setattr(
        registry, "import_module", lambda *args, **kwargs: implementation
    )

    params = {"approximant": port.approximant, "mass1": 10}
    assert registry.native_approximants("fd") == (port.approximant,)
    assert registry.try_torch_native_waveform("fd", params) == 10


def test_registered_import_error_is_not_hidden(monkeypatch):
    port = registry.TorchNativeWaveform(
        approximant="BrokenWaveform",
        component_flag="broken",
        module="missing",
        default_enabled=True,
        fd_generator="generate",
        fd_supported="supported",
    )
    monkeypatch.setattr(
        registry, "TORCH_NATIVE_WAVEFORMS", {port.approximant: port}
    )

    def fail_import(*args, **kwargs):
        raise ImportError("broken native implementation")

    monkeypatch.setattr(registry, "import_module", fail_import)
    with pytest.raises(ImportError, match="broken native implementation"):
        registry.try_torch_native_waveform(
            "fd", {"approximant": port.approximant}
        )
