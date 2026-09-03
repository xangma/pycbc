# Copyright (C) 2026  PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.

"""Focused registry tests for Torch-native higher-mode waveforms."""

from importlib import import_module

from pycbc.waveform import torch_waveform_registry as registry


_EXPECTED_PORTS = {
    "IMRPhenomXHM": {
        "component_flag": "PYCBC_IMRPHENOMXHM_NATIVE",
        "module": "imrphenomxhm_torch",
        "interfaces": {
            "fd": (
                "imrphenomxhm_fd_torch",
                "imrphenomxhm_fd_native_supported",
            ),
            "fd_modes": (
                "imrphenomxhm_modes_torch",
                "imrphenomxhm_modes_native_supported",
            ),
            "sequence": (
                "imrphenomxhm_fd_sequence_torch",
                "imrphenomxhm_sequence_native_supported",
            ),
        },
    },
    "IMRPhenomHM": {
        "component_flag": "PYCBC_IMRPHENOMHM_NATIVE",
        "module": "imrphenomhm_torch",
        "interfaces": {
            "fd": (
                "imrphenomhm_fd_torch",
                "imrphenomhm_native_supported",
            ),
            "fd_modes": (
                "imrphenomhm_modes_torch",
                "imrphenomhm_modes_native_supported",
            ),
            "sequence": (
                "imrphenomhm_fd_sequence_torch",
                "imrphenomhm_sequence_native_supported",
            ),
        },
    },
    "IMRPhenomTHM": {
        "component_flag": "PYCBC_IMRPHENOMTHM_NATIVE",
        "module": "imrphenomthm_torch",
        "interfaces": {
            "td": (
                "imrphenomthm_td_torch",
                "imrphenomthm_native_supported",
            ),
        },
        "default_supported": "imrphenomthm_default_native_supported",
    },
    "SEOBNRv4HM_ROM": {
        "component_flag": "PYCBC_SEOBNRV4HM_NATIVE",
        "module": "seobnrv4hm_torch",
        "interfaces": {
            "fd": (
                "seobnrv4hm_fd_torch",
                "seobnrv4hm_native_supported",
            ),
            "sequence": (
                "seobnrv4hm_fd_sequence_torch",
                "seobnrv4hm_sequence_native_supported",
            ),
        },
    },
    "SEOBNRv5HM_ROM": {
        "component_flag": "PYCBC_SEOBNRV5HM_NATIVE",
        "module": "seobnrv5hm_torch",
        "interfaces": {
            "fd": (
                "seobnrv5hm_fd_torch",
                "seobnrv5hm_native_supported",
            ),
            "sequence": (
                "seobnrv5hm_fd_sequence_torch",
                "seobnrv5hm_sequence_native_supported",
            ),
        },
    },
}


def test_registry_contains_higher_mode_models():
    for approximant, expected in _EXPECTED_PORTS.items():
        port = registry.TORCH_NATIVE_WAVEFORMS[approximant]
        assert port.component_flag == expected["component_flag"]
        assert port.module == expected["module"]
        assert port.default_enabled
        assert set(registry._declared_interfaces(port)) == set(
            expected["interfaces"]
        )
        for interface, (generator, supported) in expected[
            "interfaces"
        ].items():
            assert getattr(port, f"{interface}_generator") == generator
            assert getattr(port, f"{interface}_supported") == supported
            assert approximant in registry.native_approximants(interface)
        assert port.default_supported == expected.get("default_supported")


def test_registered_higher_mode_modules_are_importable():
    for approximant in _EXPECTED_PORTS:
        port = registry.TORCH_NATIVE_WAVEFORMS[approximant]
        implementation = import_module(f"pycbc.waveform.{port.module}")
        for interface in _EXPECTED_PORTS[approximant]["interfaces"]:
            generator = getattr(port, f"{interface}_generator")
            supported = getattr(port, f"{interface}_supported")
            assert callable(getattr(implementation, generator))
            assert callable(getattr(implementation, supported))
        if port.default_supported is not None:
            assert callable(getattr(implementation, port.default_supported))
