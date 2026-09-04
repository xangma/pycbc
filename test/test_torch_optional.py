import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("failure", ("ImportError", "OSError"))
def test_pycbc_imports_without_torch(failure):
    """Torch must remain optional for base and analytical PSD imports."""
    code = textwrap.dedent(
        f"""
        import builtins

        real_import = builtins.__import__

        def import_without_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise {failure}("torch intentionally unavailable")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = import_without_torch

        import pycbc
        assert not pycbc.HAVE_TORCH
        import pycbc.psd.analytical
        """
    )
    env = os.environ.copy()
    env["PYCBC_SCHEME"] = "cpu"
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
