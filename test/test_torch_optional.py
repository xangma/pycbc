import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("failure", ("ImportError", "OSError"))
def test_pycbc_defers_broken_torch_import_until_scheme_construction(failure):
    """Base import detects Torch without loading its runtime."""
    code = textwrap.dedent(
        f"""
        import builtins
        import importlib.util
        import sys

        real_import = builtins.__import__
        real_find_spec = importlib.util.find_spec
        torch_imports = []

        def find_spec_without_loading(name, *args, **kwargs):
            if name == "torch":
                return object()
            return real_find_spec(name, *args, **kwargs)

        def import_without_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                torch_imports.append(name)
                raise {failure}("torch intentionally unavailable")
            return real_import(name, *args, **kwargs)

        importlib.util.find_spec = find_spec_without_loading
        builtins.__import__ = import_without_torch

        import pycbc
        assert pycbc.HAVE_TORCH
        assert not torch_imports
        assert "torch" not in sys.modules

        import pycbc.scheme as scheme
        assert not torch_imports

        try:
            scheme.TorchScheme("cpu")
        except RuntimeError as exc:
            assert "could not be imported" in str(exc)
            assert isinstance(exc.__cause__, {failure})
        else:
            raise AssertionError("broken Torch unexpectedly constructed")

        import pycbc.psd.analytical
        """
    )
    env = os.environ.copy()
    env["PYCBC_SCHEME"] = "cpu"
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_torch_scheme_handles_missing_package_without_importing_it():
    code = textwrap.dedent(
        """
        import builtins
        import importlib.util
        import sys

        real_import = builtins.__import__
        real_find_spec = importlib.util.find_spec
        torch_imports = []

        def find_spec_without_torch(name, *args, **kwargs):
            if name == "torch":
                return None
            return real_find_spec(name, *args, **kwargs)

        def fail_on_torch_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                torch_imports.append(name)
                raise AssertionError("missing Torch should not be imported")
            return real_import(name, *args, **kwargs)

        importlib.util.find_spec = find_spec_without_torch
        builtins.__import__ = fail_on_torch_import

        import pycbc
        assert not pycbc.HAVE_TORCH
        assert not torch_imports
        assert "torch" not in sys.modules

        import pycbc.scheme as scheme
        try:
            scheme.TorchScheme("cpu")
        except RuntimeError as exc:
            assert "Install PyTorch" in str(exc)
        else:
            raise AssertionError("missing Torch unexpectedly constructed")

        assert not torch_imports
        """
    )
    env = os.environ.copy()
    env["PYCBC_SCHEME"] = "cpu"
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
