# Copyright (C) 2026  PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.

"""Focused registry tests for the Torch-native TaylorF2 family."""

from importlib import import_module
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from pycbc.waveform import torch_waveform_registry as registry


_EXPECTED_PORTS = {
    "TaylorF2": ("PYCBC_TAYLORF2_NATIVE", "taylorf2_torch"),
    "TaylorF2NLTides": (
        "PYCBC_TAYLORF2NLTIDES_NATIVE",
        "taylorf2nltides_torch",
    ),
    "TaylorF2RedSpin": (
        "PYCBC_TAYLORF2REDSPIN_NATIVE",
        "taylorf2redspin_torch",
    ),
    "TaylorF2RedSpinTidal": (
        "PYCBC_TAYLORF2REDSPINTIDAL_NATIVE",
        "taylorf2redspin_torch",
    ),
    "TaylorF2Ecc": ("PYCBC_TAYLORF2ECC_NATIVE", "taylorf2ecc_torch"),
}


def test_registry_contains_taylorf2_family():
    expected = set(_EXPECTED_PORTS)
    assert expected <= set(registry.TORCH_NATIVE_WAVEFORMS)
    assert expected <= set(registry.native_approximants("fd"))
    assert expected <= set(registry.native_approximants("sequence"))


def test_registered_modules_are_importable():
    """Keep registry additions tied to callable code present in this PR."""
    for approximant in _EXPECTED_PORTS:
        port = registry.TORCH_NATIVE_WAVEFORMS[approximant]
        expected_flag, expected_module = _EXPECTED_PORTS[approximant]
        assert port.component_flag == expected_flag
        assert port.module == expected_module
        implementation = import_module(f"pycbc.waveform.{port.module}")
        for interface in ("fd", "sequence"):
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
