"""Import-policy regressions for the strict native Torch path."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def test_deferred_optional_import_loads_only_on_attribute_access():
    script = textwrap.dedent(
        r"""
        import importlib
        import importlib.abc
        import sys

        attempts = []

        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "not_installed_for_pycbc_test":
                    attempts.append(fullname)
                    raise ModuleNotFoundError("blocked", name=fullname)
                return None

        sys.meta_path.insert(0, BlockOptional())
        from pycbc import libutils

        module = libutils.import_optional(
            "not_installed_for_pycbc_test", defer=True
        )
        assert attempts == []
        try:
            module.required_api
        except ImportError as exc:
            message = str(exc)
            assert "required_api" in message
            assert "not_installed_for_pycbc_test" in message
        else:
            raise AssertionError("deferred missing module did not fail")
        assert attempts == ["not_installed_for_pycbc_test"]
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_native_torch_import_surface_never_probes_lalsimulation():
    script = textwrap.dedent(
        r"""
        import importlib
        import importlib.abc
        import sys

        attempts = []

        class BlockLALSimulation(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "lalsimulation" or fullname.startswith(
                    "lalsimulation."
                ):
                    attempts.append(fullname)
                    raise ModuleNotFoundError(
                        "lalsimulation blocked by regression test",
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, BlockLALSimulation())

        from pycbc import conversions, pnutils, psd, waveform
        from pycbc.inject import inject
        from pycbc.noise import gaussian
        from pycbc.tmpltbank import lambda_mapping
        spa_tmplt = importlib.import_module("pycbc.waveform.spa_tmplt")

        assert attempts == []
        assert "lalsimulation" not in sys.modules
        assert waveform.waveform._lalsimulation_available is False
        assert psd.analytical.get_lalsim_psd_list() == []

        deferred_modules = (
            conversions.lalsim,
            pnutils.lalsim,
            lambda_mapping.lalsimulation,
            spa_tmplt.lalsimulation,
            inject.sim,
            gaussian.lalsimulation,
        )
        assert all("deferred optional module" in repr(module)
                   for module in deferred_modules)
        """
    )
    environment = os.environ.copy()
    environment["PYCBC_TORCH_NATIVE_PORTS"] = "1"
    environment.pop("PYCBC_TORCH_NATIVE", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
