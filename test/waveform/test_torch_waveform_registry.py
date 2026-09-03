# Copyright (C) 2026  PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.

"""Foundation tests for the Torch-native waveform registry."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from pycbc.waveform import torch_waveform_registry as registry


def test_registry_starts_empty():
    assert dict(registry.TORCH_NATIVE_WAVEFORMS) == {}
    for interface in ("td", "td_modes", "fd", "fd_modes", "sequence"):
        assert registry.native_approximants(interface) == ()


def test_registered_modules_are_importable():
    """Keep later registry additions tied to code present in the same PR."""
    for port in registry.TORCH_NATIVE_WAVEFORMS.values():
        import_module(f"pycbc.waveform.{port.module}")


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
