import os
import subprocess
import sys
import textwrap
from importlib import import_module

import pytest


pytest.importorskip("torch")


def test_registry_entries_have_complete_interfaces():
    from pycbc.waveform.torch_waveform_registry import (
        TORCH_NATIVE_WAVEFORMS,
        native_approximants,
        torch_waveform_capabilities,
    )

    interfaces = ("td", "td_modes", "fd", "fd_modes", "sequence")
    rows = torch_waveform_capabilities()
    row_keys = {(row.approximant, row.interface) for row in rows}
    declared_keys = set()

    assert len(TORCH_NATIVE_WAVEFORMS) == 59
    assert len(rows) == 118
    assert "TaylorF2" in native_approximants("fd")
    assert "IMRPhenomXO4a" in native_approximants("fd")
    assert "IMRPhenomXPNR" in native_approximants("fd")
    assert "IMRPhenomXPNR" in native_approximants("sequence")
    assert "IMRPhenomPv3" in native_approximants("sequence")
    assert "IMRPhenomPv3HM" in native_approximants("sequence")
    assert "IMRPhenomD_NRTidalv2" in native_approximants("sequence")
    assert "SEOBNRv4" in native_approximants("fd")
    assert "SEOBNRv4T_surrogate" in native_approximants("sequence")
    assert "SEOBNRv4P" in native_approximants("sequence")
    for approximant in (
        "SpinTaylorT4Fourier",
        "SpinTaylorT5Fourier",
    ):
        assert approximant in native_approximants("fd")
        assert approximant not in native_approximants("sequence")
    assert native_approximants("td") == (
        "TaylorT1",
        "TaylorT2",
        "TaylorT3",
        "TaylorT4",
        "SpinTaylorT1",
        "SpinTaylorT4",
        "SpinTaylorT5",
        "IMRPhenomT",
        "IMRPhenomTHM",
        "IMRPhenomTP",
        "IMRPhenomTPHM",
        "SEOBNRv4",
        "SEOBNRv4HM",
        "SEOBNRv4P",
        "SEOBNRv4PHM",
    )
    assert native_approximants("td_modes") == (
        "TaylorT1",
        "TaylorT2",
        "TaylorT3",
        "TaylorT4",
        "SpinTaylorT1",
        "SpinTaylorT4",
        "SpinTaylorT5",
        "IMRPhenomTPHM",
        "SEOBNRv4P",
        "SEOBNRv4PHM",
    )
    assert native_approximants("fd_modes") == (
        "IMRPhenomXHM",
        "IMRPhenomHM",
    )
    default_enabled = {
        "EccentricFD",
        "TaylorF2",
        "TaylorF2Ecc",
        "TaylorF2NLTides",
        "TaylorF2RedSpin",
        "TaylorF2RedSpinTidal",
        "TaylorT4",
        "SpinTaylorF2",
        "IMRPhenomA",
        "IMRPhenomB",
        "IMRPhenomC",
        "IMRPhenomD",
        "IMRPhenomD_NRTidal",
        "IMRPhenomD_NRTidalv2",
        "IMRPhenomNSBH",
        "IMRPhenomP",
        "IMRPhenomPv2",
        "IMRPhenomPv2_NRTidal",
        "IMRPhenomPv2_NRTidalv2",
        "IMRPhenomXAS",
        "IMRPhenomXAS_NRTidalv2",
        "IMRPhenomXAS_NRTidalv3",
        "IMRPhenomXP",
        "IMRPhenomXP_NRTidalv2",
        "IMRPhenomXP_NRTidalv3",
        "IMRPhenomXHM",
        "IMRPhenomXPHM",
        "IMRPhenomXO4a",
        "IMRPhenomXPNR",
        "IMRPhenomHM",
        "IMRPhenomPv3",
        "IMRPhenomPv3HM",
        "IMRPhenomTHM",
        "SEOBNRv4_ROM",
        "SEOBNRv4_ROM_NRTidal",
        "SEOBNRv4_ROM_NRTidalv2",
        "SEOBNRv4_ROM_NRTidalv2_NSBH",
        "SEOBNRv4T_surrogate",
        "SEOBNRv4HM_ROM",
        "SEOBNRv5_ROM",
        "SEOBNRv5_ROM_NRTidalv3",
        "SEOBNRv5HM_ROM",
    }

    for approximant, port in TORCH_NATIVE_WAVEFORMS.items():
        assert port.approximant == approximant
        assert port.component_flag.startswith("PYCBC_")
        assert port.default_enabled is (approximant in default_enabled)
        if port.default_supported is not None:
            implementation = import_module(f"pycbc.waveform.{port.module}")
            assert callable(getattr(implementation, port.default_supported))
        for interface in interfaces:
            generator = getattr(port, f"{interface}_generator")
            supported = getattr(port, f"{interface}_supported")
            assert (generator is None) == (supported is None)
            if generator is not None:
                declared_keys.add((approximant, interface))
                implementation = import_module(f"pycbc.waveform.{port.module}")
                assert callable(getattr(implementation, generator))
                assert callable(getattr(implementation, supported))
        assert all(
            capability.default_enabled in (None, False, True)
            for capability in port.interface_capabilities
        )

    assert row_keys == declared_keys
    for interface in interfaces:
        expected = tuple(
            approximant
            for approximant, port in TORCH_NATIVE_WAVEFORMS.items()
            if getattr(port, f"{interface}_generator") is not None
        )
        assert native_approximants(interface) == expected


def test_capability_ledger_metadata_and_render_are_deterministic():
    from pycbc.waveform.torch_waveform_registry import (
        TorchWaveformFallback,
        TorchWaveformDefault,
        TorchWaveformReference,
        render_torch_waveform_capabilities,
        torch_waveform_capabilities,
    )

    rows = torch_waveform_capabilities()
    lookup = {(row.approximant, row.interface): row for row in rows}
    extension_keys = {
        (row.approximant, row.interface)
        for row in rows
        if row.reference is TorchWaveformReference.NATIVE_EXTENSION
    }
    assert extension_keys == {
        ("EccentricFD", "sequence"),
        ("EOBNRv2HM_ROM", "sequence"),
        ("EOBNRv2_ROM", "sequence"),
        ("IMRPhenomA", "sequence"),
        ("IMRPhenomB", "sequence"),
        ("IMRPhenomC", "sequence"),
        ("IMRPhenomPv3", "sequence"),
        ("IMRPhenomPv3HM", "sequence"),
        ("SEOBNRv4P", "sequence"),
        ("SEOBNRv4PHM", "sequence"),
        ("SpinTaylorF2", "sequence"),
        ("TaylorF2Ecc", "sequence"),
        ("TaylorF2NLTides", "sequence"),
        ("TaylorF2RedSpin", "sequence"),
        ("TaylorF2RedSpinTidal", "sequence"),
    }
    adapter_keys = {
        (row.approximant, row.interface)
        for row in rows
        if row.fallback is TorchWaveformFallback.CPU_LAL_ADAPTER
    }
    assert adapter_keys == {
        ("IMRPhenomHM", "fd_modes"),
        ("IMRPhenomXHM", "fd_modes"),
        ("SEOBNRv4", "fd"),
        ("SEOBNRv4P", "fd"),
        ("SEOBNRv4PHM", "fd"),
    }
    assert {
        (row.approximant, row.interface)
        for row in rows
        if "mps" not in row.eligible_devices
    } == {
        ("IMRPhenomXPNR", "fd"),
        ("IMRPhenomXPNR", "sequence"),
        ("SEOBNRv4T_surrogate", "fd"),
        ("SEOBNRv4T_surrogate", "sequence"),
        ("SEOBNRv5_ROM_NRTidalv3", "fd"),
        ("SEOBNRv5_ROM_NRTidalv3", "sequence"),
        ("SEOBNRv4HM", "td"),
        ("SpinTaylorT4Fourier", "fd"),
        ("SpinTaylorT5Fourier", "fd"),
    }

    assert lookup[("SEOBNRv4P", "sequence")].reference is (
        TorchWaveformReference.NATIVE_EXTENSION
    )
    assert lookup[("SEOBNRv4P", "sequence")].fallback is (
        TorchWaveformFallback.NO_LAL_EQUIVALENT
    )
    assert lookup[("IMRPhenomHM", "fd_modes")].fallback is (
        TorchWaveformFallback.CPU_LAL_ADAPTER
    )
    assert lookup[("IMRPhenomXPNR", "fd")].eligible_devices == (
        "cpu",
        "cuda",
    )
    assert lookup[("SEOBNRv5_ROM_NRTidalv3", "sequence")].eligible_devices == (
        "cpu",
        "cuda",
    )
    for approximant, interface in (
        ("TaylorF2", "fd"),
        ("IMRPhenomD", "fd"),
        ("IMRPhenomXPHM", "fd"),
    ):
        assert lookup[(approximant, interface)].default_policy is (
            TorchWaveformDefault.DEFAULT_ON
        )
    assert lookup[("TaylorT4", "td")].default_policy is (
        TorchWaveformDefault.PREDICATE_GUARDED
    )
    for approximant in ("TaylorF2", "IMRPhenomD", "IMRPhenomXPHM"):
        assert lookup[(approximant, "sequence")].default_policy is (
            TorchWaveformDefault.DEFAULT_ON
        )
    assert lookup[("TaylorT4", "td_modes")].default_policy is (
        TorchWaveformDefault.PREDICATE_GUARDED
    )
    assert lookup[("TaylorT1", "td")].default_policy is (
        TorchWaveformDefault.OPT_IN
    )
    v4hm_td = lookup[("SEOBNRv4HM", "td")]
    assert v4hm_td.default_policy is TorchWaveformDefault.OPT_IN
    assert v4hm_td.eligible_devices == ("cpu",)
    assert v4hm_td.reference is TorchWaveformReference.LAL_REFERENCE
    assert v4hm_td.fallback is TorchWaveformFallback.STANDARD_LAL
    assert v4hm_td.component_flag == "PYCBC_SEOBNRV4HM_NATIVE"
    for approximant, component_flag in (
        (
            "SpinTaylorT4Fourier",
            "PYCBC_SPINTAYLORT4FOURIER_NATIVE",
        ),
        (
            "SpinTaylorT5Fourier",
            "PYCBC_SPINTAYLORT5FOURIER_NATIVE",
        ),
    ):
        row = lookup[(approximant, "fd")]
        assert row.default_policy is TorchWaveformDefault.OPT_IN
        assert row.eligible_devices == ("cpu",)
        assert row.reference is TorchWaveformReference.LAL_REFERENCE
        assert row.fallback is TorchWaveformFallback.STANDARD_LAL
        assert row.component_flag == component_flag

    for approximant, component_flag in (
        ("SEOBNRv4P", "PYCBC_SEOBNRV4P_NATIVE"),
        ("SEOBNRv4PHM", "PYCBC_SEOBNRV4PHM_NATIVE"),
    ):
        row = lookup[(approximant, "td_modes")]
        assert row.default_policy is TorchWaveformDefault.OPT_IN
        assert row.eligible_devices == ("cpu", "cuda", "mps")
        assert row.reference is TorchWaveformReference.LAL_REFERENCE
        assert row.fallback is TorchWaveformFallback.STANDARD_LAL
        assert row.component_flag == component_flag

    rendered = render_torch_waveform_capabilities()
    assert rendered == render_torch_waveform_capabilities()
    assert rendered.endswith("\n")
    blocks = rendered.split("   * - ``")[1:]
    assert len(blocks) == len(rows)
    interface_labels = {
        "td": "TD",
        "td_modes": "TD modes",
        "fd": "FD",
        "fd_modes": "FD modes",
        "sequence": "FD sequence",
    }
    for row, block in zip(rows, blocks):
        assert block.startswith(f"{row.approximant}``\n")
        assert f"     - {interface_labels[row.interface]}\n" in block
        assert f"     - {row.default_policy.value}\n" in block
        devices = ", ".join(
            device.upper() for device in row.eligible_devices
        )
        assert f"     - {devices}\n" in block
        assert f"     - {row.reference.value}\n" in block
        assert f"     - {row.fallback.value}\n" in block
        assert f"     - ``{row.component_flag}``" in block


def test_native_waveforms_work_without_lal_or_lalsimulation():
    """Every advertised native interface must run without LALSuite."""
    code = textwrap.dedent(
        r"""
        import importlib.abc
        import os
        import sys

        class BlockLALSuite(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if (
                    fullname == "lal"
                    or fullname.startswith("lal.")
                    or fullname == "lalsimulation"
                    or fullname.startswith("lalsimulation.")
                ):
                    raise ModuleNotFoundError(
                        "LALSuite intentionally unavailable"
                    )
                return None

        sys.meta_path.insert(0, BlockLALSuite())

        import torch
        from pycbc import scheme as _scheme
        from pycbc.types import Array, TimeSeries
        from pycbc.waveform import (
            fd_approximants,
            get_fd_waveform,
            get_fd_waveform_modes,
            get_fd_waveform_sequence,
            get_td_waveform,
            get_td_waveform_modes,
            get_waveform_filter_length_in_time,
            td_approximants,
        )
        from pycbc.waveform.waveform_modes import (
            fd_waveform_mode_approximants,
            td_waveform_mode_approximants,
        )
        import pycbc.waveform.waveform as waveform_mod
        from pycbc.waveform.torch_waveform_registry import native_approximants

        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme("cpu")

        fd_names = set(fd_approximants(_scheme.mgr.state))
        td_names = set(td_approximants(_scheme.mgr.state))
        assert set(native_approximants("fd")) <= fd_names
        assert set(native_approximants("td")) <= td_names
        assert set(native_approximants("td_modes")) <= set(
            td_waveform_mode_approximants()
        )
        assert set(native_approximants("fd_modes")) <= set(
            fd_waveform_mode_approximants()
        )
        assert set(native_approximants("sequence")) <= set(
            waveform_mod.fd_sequence
        )
        for name in (
            "EccentricFD",
            "SpinTaylorF2",
            "TaylorF2",
            "IMRPhenomC",
            "IMRPhenomD",
            "IMRPhenomP",
            "IMRPhenomHM",
            "IMRPhenomPv3",
            "IMRPhenomPv3HM",
            "IMRPhenomD_NRTidalv2",
            "SEOBNRv4P",
            "SEOBNRv4PHM",
        ):
            assert name in fd_names
            assert name in waveform_mod.fd_sequence
        assert {
            "TaylorT1",
            "TaylorT2",
            "TaylorT3",
            "TaylorT4",
            "SpinTaylorT1",
            "SpinTaylorT4",
            "SpinTaylorT5",
            "IMRPhenomT",
            "IMRPhenomTHM",
            "IMRPhenomTP",
            "IMRPhenomTPHM",
            "SEOBNRv4",
            "SEOBNRv4P",
            "SEOBNRv4PHM",
        } <= td_names
        assert "TaylorF2_INTERP" in fd_names
        assert "IMRPhenomHM" not in fd_approximants(_scheme.CPUScheme())
        assert get_waveform_filter_length_in_time(
            approximant="TaylorF2_INTERP",
            mass1=10.0,
            mass2=9.0,
            f_lower=30.0,
        ) > 0.0

        hp, hc = get_fd_waveform(
            approximant="TaylorF2",
            mass1=10.0,
            mass2=9.0,
            spin1z=0.2,
            spin2z=-0.1,
            delta_f=0.5,
            f_lower=30.0,
            f_final=128.0,
            distance=500.0,
        )
        assert len(hp) == len(hc) > 0
        assert isinstance(hp._data.tensor, torch.Tensor)

        taylorf2ecc_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_TAYLORF2ECC_NATIVE",
            )
        }
        try:
            eccentric = dict(
                mass1=20.0,
                mass2=15.0,
                spin1z=0.2,
                spin2z=-0.1,
                distance=400.0,
                inclination=0.8,
                coa_phase=0.3,
                eccentricity=0.05,
                f_ref=30.0,
            )
            hp, hc = get_fd_waveform(
                approximant="TaylorF2Ecc",
                delta_f=1.0,
                f_lower=20.0,
                f_final=160.0,
                **eccentric,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="TaylorF2Ecc",
                sample_points=[140.0, 20.0, 60.0, 90.0, 20.0],
                **eccentric,
            )
            assert len(hp) == len(hc) == 5
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in taylorf2ecc_env.items():
                if value is not None:
                    os.environ[name] = value

        eccentricfd_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_ECCENTRICFD_NATIVE",
            )
        }
        try:
            eccentricfd = dict(
                mass1=25.0,
                mass2=10.0,
                distance=275.0,
                inclination=1.1,
                coa_phase=1.3,
                long_asc_nodes=0.8,
                eccentricity=0.15,
            )
            hp, hc = get_fd_waveform(
                approximant="EccentricFD",
                delta_f=1.0,
                f_lower=18.0,
                f_final=160.0,
                **eccentricfd,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="EccentricFD",
                sample_points=[140.0, 18.0, 60.0, 90.0, 18.0],
                **eccentricfd,
            )
            assert len(hp) == len(hc) == 5
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in eccentricfd_env.items():
                if value is not None:
                    os.environ[name] = value

        spintaylorf2_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_SPINTAYLORF2_NATIVE",
            )
        }
        try:
            spintaylorf2 = dict(
                mass1=12.0,
                mass2=7.0,
                spin1x=-0.18,
                spin1y=0.12,
                spin1z=0.35,
                inclination=1.1,
                coa_phase=0.4,
                f_ref=31.0,
                distance=300.0,
                phase_order=6,
                spin_order=4,
                side_bands=2,
            )
            hp, hc = get_fd_waveform(
                approximant="SpinTaylorF2",
                delta_f=0.5,
                f_lower=20.0,
                f_final=160.0,
                **spintaylorf2,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="SpinTaylorF2",
                sample_points=[140.0, 20.0, 60.0, 90.0, 20.0],
                **spintaylorf2,
            )
            assert len(hp) == len(hc) == 5
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in spintaylorf2_env.items():
                if value is not None:
                    os.environ[name] = value

        imrphenomab_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_IMRPHENOMA_NATIVE",
                "PYCBC_IMRPHENOMB_NATIVE",
                "PYCBC_IMRPHENOMC_NATIVE",
            )
        }
        try:
            common = dict(
                mass1=35.0,
                mass2=20.0,
                distance=400.0,
                inclination=0.7,
                coa_phase=0.4,
            )
            for approximant, spins in (
                ("IMRPhenomA", {}),
                ("IMRPhenomB", {"spin1z": 0.4, "spin2z": -0.2}),
                ("IMRPhenomC", {"spin1z": 0.4, "spin2z": -0.2}),
            ):
                hp, hc = get_fd_waveform(
                    approximant=approximant,
                    delta_f=0.5,
                    f_lower=20.0,
                    f_final=160.0,
                    **spins,
                    **common,
                )
                assert len(hp) == len(hc) > 0
                assert isinstance(hp._data.tensor, torch.Tensor)

                hp, hc = get_fd_waveform_sequence(
                    approximant=approximant,
                    sample_points=[140.0, 20.0, 60.0, 90.0],
                    **spins,
                    **common,
                )
                assert len(hp) == len(hc) == 4
                assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in imrphenomab_env.items():
                if value is not None:
                    os.environ[name] = value

        taylorf2nltides_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_TAYLORF2NLTIDES_NATIVE",
            )
        }
        try:
            nltides = dict(
                mass1=1.6,
                mass2=1.3,
                spin1z=0.03,
                spin2z=-0.02,
                distance=120.0,
                inclination=0.7,
                coa_phase=0.4,
                f_ref=30.0,
                lambda1=800.0,
                lambda2=650.0,
                nl_tides_a1=1.0e-8,
                nl_tides_n1=2.5,
                nl_tides_f1=60.0,
                nl_tides_a2=2.0e-8,
                nl_tides_n2=3.0,
                nl_tides_f2=90.0,
            )
            hp, hc = get_fd_waveform(
                approximant="TaylorF2NLTides",
                delta_f=1.0,
                f_lower=20.0,
                f_final=160.0,
                **nltides,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="TaylorF2NLTides",
                sample_points=[140.0, 20.0, 60.0, 90.0, 20.0],
                **nltides,
            )
            assert len(hp) == len(hc) == 5
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in taylorf2nltides_env.items():
                if value is not None:
                    os.environ[name] = value

        taylorf2redspin_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_TAYLORF2REDSPIN_NATIVE",
                "PYCBC_TAYLORF2REDSPINTIDAL_NATIVE",
            )
        }
        try:
            redspin = dict(
                mass1=12.3,
                mass2=7.8,
                spin1z=0.45,
                spin2z=-0.27,
                f_ref=41.7,
                distance=321.0,
                inclination=0.8,
                coa_phase=0.37,
            )
            for approximant, extra in (
                ("TaylorF2RedSpin", {}),
                (
                    "TaylorF2RedSpinTidal",
                    {"lambda1": 333.0, "lambda2": 777.0},
                ),
            ):
                hp, hc = get_fd_waveform(
                    approximant=approximant,
                    delta_f=0.5,
                    f_lower=23.1,
                    f_final=180.0,
                    **extra,
                    **redspin,
                )
                assert len(hp) == len(hc) > 0
                assert isinstance(hp._data.tensor, torch.Tensor)

                hp, hc = get_fd_waveform_sequence(
                    approximant=approximant,
                    sample_points=[150.0, 23.25, 80.0, 23.25],
                    **extra,
                    **redspin,
                )
                assert len(hp) == len(hc) == 4
                assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in taylorf2redspin_env.items():
                if value is not None:
                    os.environ[name] = value

        imrphenomxo4a_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_IMRPHENOMXO4A_NATIVE",
            )
        }
        try:
            xo4a = dict(
                mass1=40.0,
                mass2=20.0,
                spin1x=0.35,
                spin1y=-0.15,
                spin1z=0.2,
                spin2x=-0.1,
                spin2y=0.08,
                spin2z=-0.05,
                distance=400.0,
                inclination=0.9,
                coa_phase=0.3,
                f_ref=30.3,
            )
            hp, hc = get_fd_waveform(
                approximant="IMRPhenomXO4a",
                delta_f=2.0,
                f_lower=20.0,
                f_final=128.0,
                **xo4a,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="IMRPhenomXO4a",
                sample_points=[20.0, 30.3, 80.0, 160.0],
                **xo4a,
            )
            assert len(hp) == len(hc) == 4
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in imrphenomxo4a_env.items():
                if value is not None:
                    os.environ[name] = value

        imrphenomxpnr_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_IMRPHENOMXPNR_NATIVE",
            )
        }
        try:
            xpnr = dict(
                mass1=40.0,
                mass2=20.0,
                spin1x=0.35,
                spin1y=-0.15,
                spin1z=0.2,
                spin2x=-0.1,
                spin2y=0.08,
                spin2z=-0.05,
                distance=400.0,
                inclination=0.9,
                coa_phase=0.3,
                f_ref=30.3,
            )
            hp, hc = get_fd_waveform(
                approximant="IMRPhenomXPNR",
                delta_f=2.0,
                f_lower=20.0,
                f_final=128.0,
                **xpnr,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="IMRPhenomXPNR",
                sample_points=[20.0, 30.3, 80.0, 160.0],
                **xpnr,
            )
            assert len(hp) == len(hc) == 4
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in imrphenomxpnr_env.items():
                if value is not None:
                    os.environ[name] = value

        common = dict(
            mass1=35.0,
            mass2=28.0,
            spin1z=0.2,
            spin2z=-0.1,
            distance=500.0,
            inclination=0.4,
            coa_phase=0.2,
            f_ref=20.0,
            sample_points=[20.0, 31.5, 80.0, 180.0],
        )
        tidal = dict(
            mass1=1.4,
            mass2=1.2,
            spin1z=0.05,
            spin2z=-0.02,
            lambda1=400.0,
            lambda2=800.0,
            distance=100.0,
            inclination=0.4,
            coa_phase=0.7,
            f_ref=0.0,
            sample_points=[20.0, 100.0, 500.0, 1000.0],
        )
        imrphenomd_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_IMRPHENOMD_NATIVE",
            )
        }
        try:
            regular = dict(common)
            sample_points = regular.pop("sample_points")
            hp, hc = get_fd_waveform(
                approximant="IMRPhenomD",
                delta_f=0.5,
                f_lower=20.0,
                f_final=256.0,
                **regular,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="IMRPhenomD",
                sample_points=sample_points,
                **regular,
            )
            assert len(hp) == len(hc) == 4
            assert isinstance(hp._data.tensor, torch.Tensor)

            for approximant in (
                "IMRPhenomD_NRTidal",
                "IMRPhenomD_NRTidalv2",
            ):
                regular = dict(tidal)
                sample_points = regular.pop("sample_points")
                hp, hc = get_fd_waveform(
                    approximant=approximant,
                    delta_f=1.0,
                    f_lower=20.0,
                    f_final=2048.0,
                    **regular,
                )
                assert len(hp) == len(hc) > 0
                assert isinstance(hp._data.tensor, torch.Tensor)

                hp, hc = get_fd_waveform_sequence(
                    approximant=approximant,
                    sample_points=sample_points,
                    **regular,
                )
                assert len(hp) == len(hc) == 4
                assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in imrphenomd_env.items():
                if value is not None:
                    os.environ[name] = value

        imrphenomxas_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_IMRPHENOMXAS_NATIVE",
            )
        }
        try:
            regular = dict(common)
            regular.pop("sample_points")
            hp, hc = get_fd_waveform(
                approximant="IMRPhenomXAS",
                delta_f=0.5,
                f_lower=20.0,
                f_final=256.0,
                **regular,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            regular = dict(tidal)
            sample_points = regular.pop("sample_points")
            hp, hc = get_fd_waveform_sequence(
                approximant="IMRPhenomXAS_NRTidalv3",
                sample_points=sample_points,
                **regular,
            )
            assert len(hp) == len(hc) == 4
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in imrphenomxas_env.items():
                if value is not None:
                    os.environ[name] = value

        imrphenomxp_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_IMRPHENOMXP_NATIVE",
            )
        }
        try:
            hp, hc = get_fd_waveform(
                approximant="IMRPhenomXP",
                mass1=40.0,
                mass2=20.0,
                spin1x=0.35,
                spin1y=-0.15,
                spin1z=0.2,
                spin2x=-0.1,
                spin2y=0.08,
                spin2z=-0.05,
                distance=400.0,
                inclination=0.9,
                coa_phase=0.3,
                delta_f=2.0,
                f_lower=20.0,
                f_final=128.0,
                f_ref=30.3,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="IMRPhenomXP_NRTidalv3",
                mass1=1.2,
                mass2=1.6,
                spin1x=0.015,
                spin1y=-0.02,
                spin1z=-0.04,
                spin2x=0.01,
                spin2y=0.012,
                spin2z=0.05,
                lambda1=800.0,
                lambda2=300.0,
                dquad_mon1=3.0,
                dquad_mon2=4.0,
                distance=130.0,
                inclination=0.8,
                coa_phase=0.6,
                f_ref=0.0,
                sample_points=[19.3, 100.0, 500.0, 1024.0],
            )
            assert len(hp) == len(hc) == 4
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in imrphenomxp_env.items():
                if value is not None:
                    os.environ[name] = value

        imrphenomnsbh_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_IMRPHENOMNSBH_NATIVE",
            )
        }
        try:
            nsbh = dict(
                mass1=7.0,
                mass2=1.4,
                spin1z=0.2,
                spin2z=0.0,
                lambda1=0.0,
                lambda2=500.0,
                distance=100.0,
                inclination=0.4,
                coa_phase=0.7,
                f_ref=20.0,
            )
            hp, hc = get_fd_waveform(
                approximant="IMRPhenomNSBH",
                delta_f=1.0,
                f_lower=20.0,
                f_final=256.0,
                **nsbh,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="IMRPhenomNSBH",
                sample_points=[20.0, 30.0, 100.0],
                **nsbh,
            )
            assert len(hp) == len(hc) == 3
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in imrphenomnsbh_env.items():
                if value is not None:
                    os.environ[name] = value

        imrphenomp_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_IMRPHENOMP_NATIVE",
            )
        }
        try:
            phenomp = dict(
                mass1=40.0,
                mass2=20.0,
                spin1x=0.2,
                spin1y=-0.1,
                spin1z=0.3,
                spin2x=-0.1,
                spin2y=0.05,
                spin2z=-0.2,
                distance=400.0,
                inclination=0.9,
                coa_phase=0.3,
                f_ref=30.0,
            )
            hp, hc = get_fd_waveform(
                approximant="IMRPhenomP",
                delta_f=2.0,
                f_lower=20.0,
                f_final=128.0,
                **phenomp,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="IMRPhenomP",
                sample_points=[20.0, 30.0, 80.0, 120.0],
                **phenomp,
            )
            assert len(hp) == len(hc) == 4
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in imrphenomp_env.items():
                if value is not None:
                    os.environ[name] = value

        imrphenompv2_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_IMRPHENOMPV2_NATIVE",
            )
        }
        try:
            pv2 = dict(
                mass1=40.0,
                mass2=20.0,
                spin1x=0.2,
                spin1y=-0.1,
                spin1z=0.3,
                spin2x=-0.1,
                spin2y=0.05,
                spin2z=-0.2,
                distance=400.0,
                inclination=0.9,
                coa_phase=0.3,
                f_ref=30.0,
            )
            hp, hc = get_fd_waveform(
                approximant="IMRPhenomPv2",
                delta_f=2.0,
                f_lower=20.0,
                f_final=128.0,
                **pv2,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="IMRPhenomPv2_NRTidalv2",
                mass1=1.55,
                mass2=1.15,
                spin1x=-0.02,
                spin1y=0.04,
                spin1z=-0.05,
                spin2x=0.03,
                spin2y=-0.01,
                spin2z=0.08,
                lambda1=300.0,
                lambda2=900.0,
                distance=120.0,
                inclination=1.1,
                coa_phase=0.2,
                f_ref=0.0,
                sample_points=[20.0, 100.0, 500.0, 1000.0],
            )
            assert len(hp) == len(hc) == 4
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in imrphenompv2_env.items():
                if value is not None:
                    os.environ[name] = value

        imrphenomxphm_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_IMRPHENOMXPHM_NATIVE",
            )
        }
        try:
            hp, hc = get_fd_waveform(
                approximant="IMRPhenomXPHM",
                mass1=40.0,
                mass2=20.0,
                spin1x=0.2,
                spin1y=0.1,
                spin1z=0.3,
                spin2x=-0.1,
                spin2y=0.05,
                spin2z=-0.2,
                distance=500.0,
                inclination=0.7,
                coa_phase=1.2,
                long_asc_nodes=0.3,
                delta_f=0.5,
                f_lower=20.0,
                f_final=512.0,
                f_ref=30.0,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="IMRPhenomXPHM",
                mass1=12.0,
                mass2=35.0,
                spin1x=0.15,
                spin1y=-0.25,
                spin1z=0.4,
                spin2x=0.05,
                spin2y=0.2,
                spin2z=-0.3,
                distance=320.0,
                inclination=1.1,
                coa_phase=0.0,
                long_asc_nodes=-0.4,
                f_ref=0.0,
                sample_points=[17.3, 22.0, 50.0, 150.0],
            )
            assert len(hp) == len(hc) == 4
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in imrphenomxphm_env.items():
                if value is not None:
                    os.environ[name] = value

        from pycbc.waveform.seobnrv4_torch import _find_rom_file

        try:
            _find_rom_file()
        except FileNotFoundError:
            seobnrv4_available = False
        else:
            seobnrv4_available = True

        if seobnrv4_available:
            seobnrv4_env = {
                name: os.environ.pop(name, None)
                for name in (
                    "PYCBC_TORCH_NATIVE_PORTS",
                    "PYCBC_TORCH_NATIVE",
                    "PYCBC_SEOBNRV4_NATIVE",
                )
            }
            try:
                hp, hc = get_fd_waveform(
                    approximant="SEOBNRv4_ROM",
                    mass1=35.0,
                    mass2=28.0,
                    spin1z=0.2,
                    spin2z=-0.1,
                    distance=500.0,
                    inclination=0.4,
                    coa_phase=0.2,
                    delta_f=1.0,
                    f_lower=30.0,
                    f_ref=30.0,
                )
                assert len(hp) == len(hc) > 0
                assert isinstance(hp._data.tensor, torch.Tensor)

                hp, hc = get_fd_waveform_sequence(
                    approximant="SEOBNRv4_ROM_NRTidalv2_NSBH",
                    mass1=8.0,
                    mass2=1.4,
                    spin1z=0.8,
                    spin2z=0.0,
                    lambda1=0.0,
                    lambda2=800.0,
                    distance=400.0,
                    inclination=0.7,
                    coa_phase=0.5,
                    f_ref=30.0,
                    sample_points=[30.0, 100.0, 500.0, 1000.0],
                )
                assert len(hp) == len(hc) == 4
                assert isinstance(hp._data.tensor, torch.Tensor)
            finally:
                for name, value in seobnrv4_env.items():
                    if value is not None:
                        os.environ[name] = value

        from pycbc.waveform.seobnrv4hm_torch import (
            _find_rom_file as _find_seobnrv4hm_rom_file,
        )

        try:
            _find_seobnrv4hm_rom_file()
        except FileNotFoundError:
            seobnrv4hm_available = False
        else:
            seobnrv4hm_available = True

        if seobnrv4hm_available:
            seobnrv4hm_env = {
                name: os.environ.pop(name, None)
                for name in (
                    "PYCBC_TORCH_NATIVE_PORTS",
                    "PYCBC_TORCH_NATIVE",
                    "PYCBC_SEOBNRV4HM_NATIVE",
                )
            }
            try:
                seobnrv4hm = dict(
                    mass1=35.0,
                    mass2=25.0,
                    spin1z=0.2,
                    spin2z=-0.1,
                    distance=500.0,
                    inclination=0.7,
                    coa_phase=0.3,
                    f_ref=20.0,
                    mode_array=[(2, -2)],
                )
                hp, hc = get_fd_waveform(
                    approximant="SEOBNRv4HM_ROM",
                    delta_f=2.0,
                    f_lower=20.0,
                    f_final=128.0,
                    **seobnrv4hm,
                )
                assert len(hp) == len(hc) > 0
                assert isinstance(hp._data.tensor, torch.Tensor)

                hp, hc = get_fd_waveform_sequence(
                    approximant="SEOBNRv4HM_ROM",
                    sample_points=[20.0, 40.0, 80.0, 120.0],
                    **seobnrv4hm,
                )
                assert len(hp) == len(hc) == 4
                assert isinstance(hp._data.tensor, torch.Tensor)
            finally:
                for name, value in seobnrv4hm_env.items():
                    if value is not None:
                        os.environ[name] = value

        from pycbc.waveform.seobnrv5_torch import (
            _find_rom_file as _find_seobnrv5_rom_file,
        )

        try:
            _find_seobnrv5_rom_file()
        except FileNotFoundError:
            seobnrv5_available = False
        else:
            seobnrv5_available = True

        if seobnrv5_available:
            seobnrv5_env = {
                name: os.environ.pop(name, None)
                for name in (
                    "PYCBC_TORCH_NATIVE_PORTS",
                    "PYCBC_TORCH_NATIVE",
                    "PYCBC_SEOBNRV5_NATIVE",
                )
            }
            try:
                hp, hc = get_fd_waveform(
                    approximant="SEOBNRv5_ROM",
                    mass1=35.0,
                    mass2=25.0,
                    spin1z=0.2,
                    spin2z=-0.1,
                    distance=500.0,
                    inclination=0.7,
                    coa_phase=0.3,
                    f_ref=20.0,
                    delta_f=2.0,
                    f_lower=20.0,
                    f_final=128.0,
                )
                assert len(hp) == len(hc) > 0
                assert isinstance(hp._data.tensor, torch.Tensor)

                hp, hc = get_fd_waveform_sequence(
                    approximant="SEOBNRv5_ROM_NRTidalv3",
                    mass1=1.2,
                    mass2=1.6,
                    spin1z=-0.04,
                    spin2z=0.08,
                    lambda1=900.0,
                    lambda2=400.0,
                    distance=100.0,
                    inclination=0.7,
                    coa_phase=0.3,
                    f_ref=50.0,
                    sample_points=[30.0, 100.0, 500.0, 1000.0],
                )
                assert len(hp) == len(hc) == 4
                assert isinstance(hp._data.tensor, torch.Tensor)
            finally:
                for name, value in seobnrv5_env.items():
                    if value is not None:
                        os.environ[name] = value

        from pycbc.waveform.seobnrv5hm_torch import (
            _find_rom_file as _find_seobnrv5hm_rom_file,
        )

        try:
            _find_seobnrv5hm_rom_file()
        except FileNotFoundError:
            seobnrv5hm_available = False
        else:
            seobnrv5hm_available = True

        if seobnrv5hm_available:
            seobnrv5hm_env = {
                name: os.environ.pop(name, None)
                for name in (
                    "PYCBC_TORCH_NATIVE_PORTS",
                    "PYCBC_TORCH_NATIVE",
                    "PYCBC_SEOBNRV5HM_NATIVE",
                )
            }
            try:
                seobnrv5hm = dict(
                    mass1=35.0,
                    mass2=25.0,
                    spin1z=0.2,
                    spin2z=-0.1,
                    distance=500.0,
                    inclination=0.7,
                    coa_phase=0.3,
                    f_ref=20.0,
                    mode_array=[(2, -2)],
                )
                hp, hc = get_fd_waveform(
                    approximant="SEOBNRv5HM_ROM",
                    delta_f=2.0,
                    f_lower=20.0,
                    f_final=128.0,
                    **seobnrv5hm,
                )
                assert len(hp) == len(hc) > 0
                assert isinstance(hp._data.tensor, torch.Tensor)

                hp, hc = get_fd_waveform_sequence(
                    approximant="SEOBNRv5HM_ROM",
                    sample_points=[20.0, 40.0, 80.0, 120.0],
                    **seobnrv5hm,
                )
                assert len(hp) == len(hc) == 4
                assert isinstance(hp._data.tensor, torch.Tensor)
            finally:
                for name, value in seobnrv5hm_env.items():
                    if value is not None:
                        os.environ[name] = value

        higher_modes = dict(
            mass1=46.0,
            mass2=19.0,
            spin1z=0.35,
            spin2z=-0.2,
            distance=350.0,
            inclination=0.7,
            coa_phase=0.4,
            f_ref=25.0,
            sample_points=[20.0, 31.5, 80.0, 180.0],
        )
        imrphenomxhm_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_IMRPHENOMXHM_NATIVE",
            )
        }
        try:
            xhm_common = dict(higher_modes)
            sample_points = xhm_common.pop("sample_points")
            inclination = xhm_common.pop("inclination")
            modes = get_fd_waveform_modes(
                approximant="IMRPhenomXHM",
                delta_f=1.0,
                f_lower=20.0,
                f_final=300.0,
                mode_array=[(2, 2), (3, 3)],
                **xhm_common,
            )
            assert set(modes) == {(2, 2), (3, 3)}
            assert all(
                isinstance(series._data.tensor, torch.Tensor)
                for pair in modes.values()
                for series in pair
            )

            hp, hc = get_fd_waveform(
                approximant="IMRPhenomXHM",
                delta_f=1.0,
                f_lower=20.0,
                f_final=300.0,
                inclination=inclination,
                mode_array=[(2, 2), (3, 3)],
                **xhm_common,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="IMRPhenomXHM",
                sample_points=sample_points,
                inclination=inclination,
                mode_array=[(2, 2), (3, 3)],
                **xhm_common,
            )
            assert len(hp) == len(hc) == 4
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in imrphenomxhm_env.items():
                if value is not None:
                    os.environ[name] = value

        imrphenomhm_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_IMRPHENOMHM_NATIVE",
                "PYCBC_IMRPHENOMPV3_NATIVE",
                "PYCBC_IMRPHENOMPV3HM_NATIVE",
            )
        }
        try:
            hm_common = dict(higher_modes)
            sample_points = hm_common.pop("sample_points")
            modes = get_fd_waveform_modes(
                approximant="IMRPhenomHM",
                delta_f=1.0,
                f_lower=20.0,
                f_final=300.0,
                mode_array=[(2, 2), (3, -2)],
                **hm_common,
            )
            assert set(modes) == {(2, 2), (3, -2)}
            assert all(
                isinstance(series._data.tensor, torch.Tensor)
                for pair in modes.values()
                for series in pair
            )

            hp, hc = get_fd_waveform(
                approximant="IMRPhenomHM",
                delta_f=1.0,
                f_lower=20.0,
                f_final=300.0,
                **hm_common,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="IMRPhenomHM",
                sample_points=sample_points,
                **hm_common,
            )
            assert len(hp) == len(hc) == 4
            assert isinstance(hp._data.tensor, torch.Tensor)

            pv3hm_common = {
                **hm_common,
                "spin1x": 0.2,
                "spin1y": -0.1,
                "spin2x": -0.05,
                "spin2y": 0.15,
                "spin2z": 0.2,
            }
            pv3_common = dict(pv3hm_common)
            pv3_common.pop("mode_array", None)
            hp, hc = get_fd_waveform(
                approximant="IMRPhenomPv3",
                delta_f=1.0,
                f_lower=20.0,
                f_final=300.0,
                **pv3_common,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="IMRPhenomPv3",
                sample_points=sample_points,
                **pv3_common,
            )
            assert len(hp) == len(hc) == 4
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform(
                approximant="IMRPhenomPv3HM",
                delta_f=1.0,
                f_lower=20.0,
                f_final=300.0,
                **pv3hm_common,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            hp, hc = get_fd_waveform_sequence(
                approximant="IMRPhenomPv3HM",
                sample_points=sample_points,
                **pv3hm_common,
            )
            assert len(hp) == len(hc) == 4
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in imrphenomhm_env.items():
                if value is not None:
                    os.environ[name] = value

        hp, hc = get_td_waveform(
            approximant="TaylorT1",
            mass1=35.0,
            mass2=25.0,
            distance=300.0,
            inclination=0.7,
            coa_phase=0.2,
            delta_t=1.0 / 1024.0,
            f_lower=50.0,
            f_ref=50.0,
        )
        assert len(hp) == len(hc) > 0
        assert isinstance(hp._data.tensor, torch.Tensor)

        modes = get_td_waveform_modes(
            approximant="TaylorT1",
            mass1=35.0,
            mass2=25.0,
            distance=300.0,
            coa_phase=0.2,
            delta_t=1.0 / 1024.0,
            f_lower=50.0,
            f_ref=50.0,
            ell_max=2,
        )
        assert set(modes) == {(2, m) for m in range(-2, 3)}
        assert all(
            isinstance(series._data.tensor, torch.Tensor)
            for pair in modes.values()
            for series in pair
        )

        hp, hc = get_td_waveform(
            approximant="TaylorT2",
            mass1=35.0,
            mass2=25.0,
            distance=300.0,
            inclination=0.7,
            coa_phase=0.2,
            delta_t=1.0 / 1024.0,
            f_lower=50.0,
            f_ref=50.0,
        )
        assert len(hp) == len(hc) > 0
        assert isinstance(hp._data.tensor, torch.Tensor)

        modes = get_td_waveform_modes(
            approximant="TaylorT2",
            mass1=35.0,
            mass2=25.0,
            distance=300.0,
            coa_phase=0.2,
            delta_t=1.0 / 1024.0,
            f_lower=50.0,
            f_ref=50.0,
            ell_max=2,
        )
        assert set(modes) == {(2, m) for m in range(-2, 3)}
        assert all(
            isinstance(series._data.tensor, torch.Tensor)
            for pair in modes.values()
            for series in pair
        )

        hp, hc = get_td_waveform(
            approximant="TaylorT3",
            mass1=30.0,
            mass2=20.0,
            distance=300.0,
            inclination=0.7,
            coa_phase=0.2,
            delta_t=1.0 / 1024.0,
            f_lower=40.0,
            f_ref=40.0,
        )
        assert len(hp) == len(hc) > 0
        assert isinstance(hp._data.tensor, torch.Tensor)

        modes = get_td_waveform_modes(
            approximant="TaylorT3",
            mass1=30.0,
            mass2=20.0,
            distance=300.0,
            coa_phase=0.2,
            delta_t=1.0 / 1024.0,
            f_lower=40.0,
            f_ref=40.0,
            ell_max=2,
        )
        assert set(modes) == {(2, m) for m in range(-2, 3)}
        assert all(
            isinstance(series._data.tensor, torch.Tensor)
            for pair in modes.values()
            for series in pair
        )

        hp, hc = get_td_waveform(
            approximant="TaylorT4",
            mass1=35.0,
            mass2=25.0,
            distance=300.0,
            inclination=0.7,
            coa_phase=0.2,
            delta_t=1.0 / 1024.0,
            f_lower=50.0,
            f_ref=50.0,
        )
        assert len(hp) == len(hc) > 0
        assert isinstance(hp._data.tensor, torch.Tensor)

        for approximant in ("SpinTaylorT1", "SpinTaylorT4", "SpinTaylorT5"):
            hp, hc = get_td_waveform(
                approximant=approximant,
                mass1=45.0,
                mass2=25.0,
                spin1x=0.2,
                spin1y=-0.1,
                spin1z=0.3,
                spin2x=-0.1,
                spin2y=0.05,
                spin2z=-0.2,
                distance=300.0,
                inclination=0.7,
                coa_phase=0.2,
                delta_t=1.0 / 1024.0,
                f_lower=50.0,
                f_ref=50.0,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)

            modes = get_td_waveform_modes(
                approximant=approximant,
                mass1=45.0,
                mass2=25.0,
                spin1x=0.2,
                spin1y=-0.1,
                spin1z=0.3,
                spin2x=-0.1,
                spin2y=0.05,
                spin2z=-0.2,
                distance=300.0,
                delta_t=1.0 / 1024.0,
                f_lower=50.0,
                f_ref=50.0,
                ell_max=2,
            )
            assert set(modes) == {(2, m) for m in range(-2, 3)}
            assert all(
                isinstance(series._data.tensor, torch.Tensor)
                for pair in modes.values()
                for series in pair
            )

        modes = get_td_waveform_modes(
            approximant="TaylorT4",
            mass1=35.0,
            mass2=25.0,
            distance=300.0,
            coa_phase=0.2,
            delta_t=1.0 / 1024.0,
            f_lower=50.0,
            f_ref=50.0,
            ell_max=2,
        )
        assert set(modes) == {(2, m) for m in range(-2, 3)}
        assert all(
            isinstance(series._data.tensor, torch.Tensor)
            for pair in modes.values()
            for series in pair
        )

        hp, hc = get_td_waveform(
            approximant="IMRPhenomT",
            mass1=50.0,
            mass2=10.0,
            spin1z=0.7,
            spin2z=-0.4,
            distance=300.0,
            inclination=1.1,
            coa_phase=-0.2,
            delta_t=1.0 / 2048.0,
            f_lower=40.0,
            f_ref=40.0,
        )
        assert len(hp) == len(hc) > 0
        assert isinstance(hp._data.tensor, torch.Tensor)

        imrphenomthm_env = {
            name: os.environ.pop(name, None)
            for name in (
                "PYCBC_TORCH_NATIVE_PORTS",
                "PYCBC_TORCH_NATIVE",
                "PYCBC_IMRPHENOMTHM_NATIVE",
            )
        }
        try:
            hp, hc = get_td_waveform(
                approximant="IMRPhenomTHM",
                mass1=30.0,
                mass2=20.0,
                spin1z=0.2,
                spin2z=-0.1,
                distance=400.0,
                inclination=0.7,
                coa_phase=0.3,
                delta_t=1.0 / 4096.0,
                f_lower=30.0,
                f_ref=30.0,
            )
            assert len(hp) == len(hc) > 0
            assert isinstance(hp._data.tensor, torch.Tensor)
        finally:
            for name, value in imrphenomthm_env.items():
                if value is not None:
                    os.environ[name] = value

        hp, hc = get_td_waveform(
            approximant="IMRPhenomTP",
            mass1=80.0,
            mass2=40.0,
            spin1x=0.2,
            spin1y=-0.1,
            spin1z=0.3,
            spin2x=-0.1,
            spin2y=0.2,
            spin2z=-0.2,
            distance=100.0,
            inclination=0.7,
            coa_phase=0.2,
            delta_t=1.0 / 2048.0,
            f_lower=40.0,
            f_ref=40.0,
        )
        assert len(hp) == len(hc) > 0
        assert isinstance(hp._data.tensor, torch.Tensor)

        hp, hc = get_td_waveform(
            approximant="IMRPhenomTPHM",
            mass1=80.0,
            mass2=40.0,
            spin1x=0.2,
            spin1y=-0.1,
            spin1z=0.3,
            spin2x=-0.1,
            spin2y=0.2,
            spin2z=-0.2,
            distance=100.0,
            inclination=0.7,
            coa_phase=0.2,
            delta_t=1.0 / 2048.0,
            f_lower=40.0,
            f_ref=40.0,
            mode_array=[(2, 2), (3, -3)],
        )
        assert len(hp) == len(hc) > 0
        assert isinstance(hp._data.tensor, torch.Tensor)

        modes = get_td_waveform_modes(
            approximant="IMRPhenomTPHM",
            mass1=80.0,
            mass2=40.0,
            spin1x=0.2,
            spin1y=-0.1,
            spin1z=0.3,
            spin2x=-0.1,
            spin2y=0.2,
            spin2z=-0.2,
            distance=100.0,
            inclination=0.7,
            coa_phase=0.2,
            delta_t=1.0 / 2048.0,
            f_lower=40.0,
            f_ref=40.0,
            mode_array=[(2, 2), (3, -3)],
        )
        assert modes
        assert all(
            isinstance(series._data.tensor, torch.Tensor)
            for pair in modes.values()
            for series in pair
        )

        from pycbc.waveform import seobnrv4phm_torch

        def fake_v4_td(**params):
            assert params["approximant"] == "SEOBNRv4"
            delta_t = params["delta_t"]
            return (
                TimeSeries([5.0, 6.0], delta_t=delta_t),
                TimeSeries([7.0, 8.0], delta_t=delta_t),
            )

        def fake_phm_td(**params):
            assert params["approximant"] == "SEOBNRv4PHM"
            delta_t = params["delta_t"]
            return (
                TimeSeries([1.0, 2.0], delta_t=delta_t),
                TimeSeries([3.0, 4.0], delta_t=delta_t),
            )

        def fake_v4p_td(**params):
            assert params["approximant"] == "SEOBNRv4P"
            delta_t = params["delta_t"]
            return (
                TimeSeries([9.0, 10.0], delta_t=delta_t),
                TimeSeries([11.0, 12.0], delta_t=delta_t),
            )

        def fake_v4p_sequence(**params):
            assert params["approximant"] == "SEOBNRv4P"
            return Array([5.0 + 6.0j]), Array([7.0 + 8.0j])

        def fake_phm_sequence(**params):
            assert params["approximant"] == "SEOBNRv4PHM"
            return Array([1.0 + 2.0j]), Array([3.0 + 4.0j])

        seobnrv4phm_torch.seobnrv4_td_torch = fake_v4_td
        seobnrv4phm_torch.seobnrv4p_td_torch = fake_v4p_td
        seobnrv4phm_torch.seobnrv4p_fd_sequence_torch = fake_v4p_sequence
        seobnrv4phm_torch.seobnrv4phm_td_torch = fake_phm_td
        seobnrv4phm_torch.seobnrv4phm_fd_sequence_torch = fake_phm_sequence
        hp, hc = get_td_waveform(
            approximant="SEOBNRv4",
            mass1=50.0,
            mass2=40.0,
            spin1z=0.3,
            spin2z=-0.2,
            distance=400.0,
            inclination=0.4,
            coa_phase=0.2,
            f_lower=20.0,
            f_ref=20.0,
            delta_t=1.0 / 4096.0,
        )
        assert hp[0] == 5.0
        assert hc[0] == 7.0

        hp, hc = get_fd_waveform(
            approximant="SEOBNRv4",
            mass1=50.0,
            mass2=40.0,
            spin1z=0.3,
            spin2z=-0.2,
            distance=400.0,
            inclination=0.4,
            coa_phase=0.2,
            f_lower=20.0,
            f_ref=20.0,
            delta_f=0.25,
        )
        assert len(hp) == len(hc) > 0
        assert isinstance(hp._data.tensor, torch.Tensor)

        hp, hc = get_td_waveform(
            approximant="SEOBNRv4P",
            mass1=25.0,
            mass2=18.0,
            spin1x=0.2,
            spin1y=-0.15,
            spin1z=0.05,
            spin2x=-0.1,
            spin2y=0.07,
            spin2z=0.2,
            distance=300.0,
            inclination=0.6,
            coa_phase=0.3,
            f_lower=20.0,
            f_ref=20.0,
            delta_t=1.0 / 4096.0,
        )
        assert hp[0] == 9.0
        assert hc[0] == 11.0

        hp, hc = get_fd_waveform_sequence(
            approximant="SEOBNRv4P",
            mass1=25.0,
            mass2=18.0,
            spin1x=0.2,
            spin1y=-0.15,
            spin1z=0.05,
            spin2x=-0.1,
            spin2y=0.07,
            spin2z=0.2,
            distance=300.0,
            inclination=0.6,
            coa_phase=0.3,
            f_ref=50.0,
            sample_points=[50.0],
        )
        assert hp[0] == 5.0 + 6.0j
        assert hc[0] == 7.0 + 8.0j

        hp, hc = get_td_waveform(
            approximant="SEOBNRv4PHM",
            mass1=25.0,
            mass2=18.0,
            spin1x=0.2,
            spin1y=-0.15,
            spin1z=0.05,
            spin2x=-0.1,
            spin2y=0.07,
            spin2z=0.2,
            distance=300.0,
            inclination=0.6,
            coa_phase=0.3,
            f_lower=20.0,
            f_ref=20.0,
            delta_t=1.0 / 4096.0,
        )
        assert hp[0] == 1.0
        assert hc[0] == 3.0

        hp, hc = get_fd_waveform_sequence(
            approximant="SEOBNRv4PHM",
            mass1=25.0,
            mass2=18.0,
            spin1x=0.2,
            spin1y=-0.15,
            spin1z=0.05,
            spin2x=-0.1,
            spin2y=0.07,
            spin2z=0.2,
            distance=300.0,
            inclination=0.6,
            coa_phase=0.3,
            f_ref=50.0,
            sample_points=[50.0],
        )
        assert hp[0] == 1.0 + 2.0j
        assert hc[0] == 3.0 + 4.0j

        os.environ["PYCBC_TAYLORF2_NATIVE"] = "0"
        try:
            get_fd_waveform(
                approximant="TaylorF2",
                mass1=10.0,
                mass2=9.0,
                delta_f=0.5,
                f_lower=30.0,
                f_final=128.0,
            )
        except ImportError as exc:
            assert "lalsimulation" in str(exc)
        else:
            raise AssertionError("disabled native port did not use fallback")

        assert "lal" not in sys.modules
        assert "lalsimulation" not in sys.modules
        """
    )
    env = os.environ.copy()
    env["PYCBC_SCHEME"] = "cpu"
    env["PYCBC_TORCH_NATIVE_PORTS"] = "1"
    for component in (
        "TAYLORF2",
        "TAYLORT1",
        "TAYLORT2",
        "TAYLORT3",
        "TAYLORT4",
        "SPINTAYLORT4",
        "IMRPHENOMD",
        "IMRPHENOMHM",
        "IMRPHENOMT",
        "IMRPHENOMTHM",
        "IMRPHENOMTP",
        "IMRPHENOMTPHM",
        "IMRPHENOMXO4A",
        "IMRPHENOMXPNR",
        "SEOBNRV4",
        "SEOBNRV4P",
        "SEOBNRV4PHM",
    ):
        env[f"PYCBC_{component}_NATIVE"] = "1"
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
