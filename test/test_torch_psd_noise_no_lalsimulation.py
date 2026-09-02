"""Regression tests for Torch PSD/noise use without LALSimulation."""

import subprocess
import sys
import textwrap


def test_torch_psd_and_noise_without_lalsimulation():
    script = textwrap.dedent(
        r"""
        import importlib.abc
        import sys

        class BlockLALSimulation(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "lalsimulation" or fullname.startswith(
                    "lalsimulation."
                ):
                    raise ModuleNotFoundError(
                        "lalsimulation blocked by regression test",
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, BlockLALSimulation())

        import numpy
        import torch
        from pycbc.noise import reproduceable
        from pycbc.noise import (
            frequency_noise_from_psd,
            noise_from_psd,
            noise_from_string,
        )
        from pycbc.psd import analytical
        from pycbc.scheme import TorchScheme
        from pycbc.types import FrequencySeries

        model = "aLIGOZeroDetHighPower"
        table_model = "aLIGODesignSensitivityP1200087"
        assert model in analytical.get_torch_psd_list()
        assert table_model in analytical.get_torch_psd_list()
        assert model in analytical.get_psd_model_list()
        assert model not in analytical.get_lalsim_psd_list()
        assert callable(getattr(analytical, model))

        with TorchScheme("cpu"):
            psd = getattr(analytical, model)(513, 1.0, 10.0)
            table_psd = analytical.from_string(
                table_model, 513, 1.0, 10.0
            )
            frequency_noise = frequency_noise_from_psd(psd, seed=11)
            time_noise = noise_from_psd(
                257, 1.0 / 256.0, psd, seed=12
            )
            string_noise = noise_from_string(
                model,
                257,
                1.0 / 256.0,
                seed=13,
                low_frequency_cutoff=10.0,
            )
            reproduceable.BLOCK_SAMPLES = 512
            reproducible_noise = reproduceable.noise_from_string(
                model,
                0,
                1,
                seed=14,
                sample_rate=64,
                low_frequency_cutoff=2.0,
                filter_duration=2,
            )

        for series in (
            psd,
            table_psd,
            frequency_noise,
            time_noise,
            string_noise,
            reproducible_noise,
        ):
            assert series._data.tensor.device.type == "cpu"
            assert torch.all(torch.isfinite(series._data.tensor))
        assert "lalsimulation" not in sys.modules

        try:
            analytical.from_string(model, 33, 1.0, 10.0)
        except ImportError as exc:
            assert "requires lalsimulation" in str(exc)
        else:
            raise AssertionError("CPU PSD fallback did not require lalsimulation")

        cpu_psd = FrequencySeries(numpy.ones(129), delta_f=1.0)
        try:
            noise_from_psd(32, 1.0 / 256.0, cpu_psd, seed=15)
        except ImportError as exc:
            assert "CPU noise_from_psd requires lalsimulation" in str(exc)
        else:
            raise AssertionError(
                "CPU noise fallback did not require lalsimulation"
            )

        try:
            analytical.from_string("not-a-real-psd", 33, 1.0, 10.0)
        except ValueError as exc:
            assert "not a built-in Torch model" in str(exc)
        else:
            raise AssertionError("unknown PSD model was accepted")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
