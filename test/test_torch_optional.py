import os
import subprocess
import sys
import textwrap


def test_common_modules_import_without_torch():
    """Torch backends must remain optional for ordinary PyCBC imports."""
    code = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def import_without_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch intentionally unavailable")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = import_without_torch

        import pycbc
        assert not pycbc.HAVE_TORCH
        import pycbc.strain.strain
        import pycbc.waveform.SpinTaylorF2
        import pycbc.waveform.generator
        import pycbc.waveform.waveform
        """
    )
    env = os.environ.copy()
    env["PYCBC_SCHEME"] = "cpu"
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
