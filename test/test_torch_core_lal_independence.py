# Copyright (C) 2026  The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Core-LAL independence tests for the supported Torch-native surface."""

import subprocess
import sys
import textwrap
import random
from decimal import Decimal
from fractions import Fraction

import pytest


def _gps_fields(value):
    return value.gpsSeconds, value.gpsNanoSeconds


def test_fallback_ligotimegps_matches_lal():
    """Keep GPS metadata compatible without pretending to be LAL."""
    lal = pytest.importorskip("lal")
    from pycbc.lal_compat import _FallbackLIGOTimeGPS

    values = (
        0,
        1,
        -1,
        0.1,
        -0.1,
        1.2,
        -1.2,
        -(1 << 31),
        "",
        "\t 1",
        "1234567890.1234567895",
        "0.0000000015",
        "-0.0000000025",
    )
    for value in values:
        expected = lal.LIGOTimeGPS(value)
        actual = _FallbackLIGOTimeGPS(value)
        assert _gps_fields(actual) == _gps_fields(expected)
        assert repr(actual) == repr(expected)
        assert str(actual) == str(expected)
        assert int(actual) == int(expected)
        assert float(actual) == float(expected)
        assert hash(actual) == hash(expected)
        assert bool(actual) is bool(expected)

    for seconds, nanoseconds in (
        (1, 200_000_000),
        (0, -1),
        (-1, 200_000_000),
        (1, -200_000_000),
        (1, -1_000_000_001),
    ):
        expected = lal.LIGOTimeGPS(seconds, nanoseconds)
        actual = _FallbackLIGOTimeGPS(seconds, nanoseconds)
        assert _gps_fields(actual) == _gps_fields(expected)

    for value in (Decimal("1.2"), Fraction(6, 5)):
        with pytest.raises(TypeError):
            lal.LIGOTimeGPS(value)
        with pytest.raises(TypeError):
            _FallbackLIGOTimeGPS(value)

    for value in (
        1 << 31,
        -(1 << 31) - 1,
        float("nan"),
        float("inf"),
        float("-inf"),
        "1 ",
    ):
        with pytest.raises(Exception) as expected_error:
            lal.LIGOTimeGPS(value)
        with pytest.raises(type(expected_error.value)):
            _FallbackLIGOTimeGPS(value)

    for seconds in (1 << 31, -(1 << 31) - 1):
        with pytest.raises(OverflowError):
            lal.LIGOTimeGPS(seconds, 0)
        with pytest.raises(OverflowError):
            _FallbackLIGOTimeGPS(seconds, 0)

    left_lal = lal.LIGOTimeGPS("1.2")
    left_fallback = _FallbackLIGOTimeGPS("1.2")
    right_lal = lal.LIGOTimeGPS("0.35")
    right_fallback = _FallbackLIGOTimeGPS("0.35")
    for expected, actual in (
        (left_lal + right_lal, left_fallback + right_fallback),
        (left_lal - right_lal, left_fallback - right_fallback),
        (right_lal - left_lal, right_fallback - left_fallback),
        (-left_lal, -left_fallback),
        (abs(-left_lal), abs(-left_fallback)),
        (left_lal * 2.5, left_fallback * 2.5),
        (left_lal / 2.5, left_fallback / 2.5),
    ):
        assert _gps_fields(actual) == _gps_fields(expected)
    assert left_fallback == _FallbackLIGOTimeGPS("1.2")
    assert left_fallback >= right_fallback
    assert right_fallback < left_fallback

    for invalid in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(RuntimeError):
            left_lal * invalid
        with pytest.raises(RuntimeError):
            left_fallback * invalid
        with pytest.raises(RuntimeError):
            left_lal / invalid
        with pytest.raises(RuntimeError):
            left_fallback / invalid
    with pytest.raises(RuntimeError):
        left_lal / 0
    with pytest.raises(RuntimeError):
        left_fallback / 0


def test_torch_surface_without_core_lal_or_lalsimulation():
    """Exercise constants, GPS, data resolution, noise, and harmonics."""
    script = textwrap.dedent(
        r"""
        import importlib.abc
        import os
        import sys
        import tempfile

        class BlockLALSuite(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if (
                    fullname == "lal"
                    or fullname.startswith("lal.")
                    or fullname == "lalsimulation"
                    or fullname.startswith("lalsimulation.")
                ):
                    raise ModuleNotFoundError(
                        "LALSuite blocked by regression test",
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, BlockLALSuite())

        import torch
        from pycbc import lal_compat
        from pycbc.noise import frequency_noise_from_psd, noise_from_psd
        from pycbc.psd import analytical
        from pycbc.scheme import TorchScheme
        from pycbc.types import Array, FrequencySeries, TimeSeries
        from pycbc.waveform import get_fd_waveform
        from pycbc.waveform.ringdown import spher_harms
        from pycbc.waveform.waveform_modes import get_glm, sum_modes

        assert not lal_compat.LAL_AVAILABLE
        assert lal_compat.LIGOTimeGPS(-0.1).gpsNanoSeconds == -100_000_000

        with tempfile.TemporaryDirectory() as data_path:
            filename = "LIGO-P1200087-v18-aLIGO_DESIGN.txt"
            with open(
                os.path.join(data_path, filename), "w", encoding="ascii"
            ) as table:
                table.write(
                    "1 1e-22\n10 2e-22\n100 3e-22\n1000 4e-22\n"
                )
            os.environ["LAL_DATA_PATH"] = data_path

            with TorchScheme("cpu"):
                array = Array([1.0, 2.0])
                times = TimeSeries(
                    [1.0, 2.0, 3.0], delta_t=0.25, epoch=-0.1
                )
                frequencies = FrequencySeries(
                    [0.0, 1.0, 2.0], delta_f=1.0, epoch="1.2"
                )
                psd = analytical.from_string(
                    "aLIGOZeroDetHighPower", 129, 1.0, 10.0
                )
                table_psd = analytical.from_string(
                    "aLIGODesignSensitivityP1200087", 129, 1.0, 10.0
                )
                frequency_noise = frequency_noise_from_psd(psd, seed=11)
                time_noise = noise_from_psd(
                    65, 1.0 / 256.0, psd, seed=12
                )
                hp, hc = get_fd_waveform(
                    approximant="TaylorF2",
                    mass1=10.0,
                    mass2=9.0,
                    spin1z=0.1,
                    delta_f=1.0,
                    f_lower=30.0,
                    f_final=128.0,
                )

                mode = TimeSeries(
                    [1.0 + 2.0j, 3.0 + 4.0j], delta_t=0.25
                )
                mode_sum = sum_modes({(2, 2): mode}, 0.7, 0.2)

        assert float(times.start_time) == -0.1
        assert float(frequencies.epoch) == 1.2
        for value in (
            array,
            times,
            frequencies,
            psd,
            table_psd,
            frequency_noise,
            time_noise,
            hp,
            hc,
            mode_sum,
        ):
            assert isinstance(value._data.tensor, torch.Tensor)
            assert value._data.tensor.device.type == "cpu"
        assert torch.all(torch.isfinite(psd._data.tensor))
        assert torch.all(torch.isfinite(table_psd._data.tensor))
        assert torch.count_nonzero(table_psd._data.tensor) > 0
        assert torch.all(torch.isfinite(frequency_noise._data.tensor))
        assert torch.all(torch.isfinite(time_noise._data.tensor))

        glm = get_glm(2, 2, 0.7)
        positive, negative = spher_harms(
            l=2, m=2, inclination=0.7, azimuthal=0.2
        )
        assert isinstance(glm, float)
        assert isinstance(positive, complex)
        assert isinstance(negative, complex)

        for value in (array, times, frequencies):
            try:
                value.lal()
            except lal_compat.LALDependencyError as exc:
                assert "requires the core 'lal' Python package" in str(exc)
            else:
                raise AssertionError("a real-LAL conversion was accepted")

        assert "lal" not in sys.modules
        assert "lalsimulation" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_torch_event_thresholds_without_lalsuite():
    """Keep Torch thresholding separate from detector/LIGO-LW features."""
    script = textwrap.dedent(
        r"""
        import importlib.abc
        import sys

        class BlockLALSuite(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"lal", "lalsimulation", "lalframe"} or \
                        fullname.startswith(
                            ("lal.", "lalsimulation.", "lalframe.")
                        ):
                    raise ModuleNotFoundError(
                        "LALSuite blocked by regression test",
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, BlockLALSuite())

        import numpy
        import torch

        try:
            from pycbc import events
        except ModuleNotFoundError as exc:
            if exc.name == "pycbc.events.eventmgr_cython":
                raise SystemExit(77)
            raise

        from pycbc import lal_compat
        from pycbc.scheme import TorchScheme
        from pycbc.types import Array
        from pycbc.types.array_torch import TorchArrayData

        with TorchScheme("cpu"):
            series = Array(
                TorchArrayData(
                    torch.tensor(
                        [0, 1, 3, 2, 0, 4, 0], dtype=torch.complex64
                    )
                ),
                copy=False,
            )
            indices, values = events.threshold_and_cluster_findchirp(
                series, 0.5, 2
            )
            numpy.testing.assert_array_equal(indices, [2, 5])
            numpy.testing.assert_array_equal(values, [3 + 0j, 4 + 0j])

        try:
            events.time_multi_coincidence(
                {"H1": numpy.array([0.0]), "L1": numpy.array([0.0])}
            )
        except lal_compat.LALDependencyError as exc:
            assert "multi-detector coincidence timing" in str(exc)
        else:
            raise AssertionError("detector timing was accepted without LAL")

        assert "pycbc.detector" not in sys.modules
        assert "lal" not in sys.modules
        assert "lalsimulation" not in sys.modules
        assert "lalframe" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 77:
        pytest.skip("eventmgr extension is unavailable for this interpreter")
    assert result.returncode == 0, result.stderr


def test_torch_gating_and_supernovae_without_lalsuite(tmp_path):
    """Keep gate and supernova imports clear of frame and LAL I/O."""
    script = textwrap.dedent(
        r"""
        import importlib.abc
        import os
        import sys

        class BlockLALSuite(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"lal", "lalsimulation", "lalframe"} or \
                        fullname.startswith(
                            ("lal.", "lalsimulation.", "lalframe.")
                        ):
                    raise ModuleNotFoundError(
                        "LALSuite blocked by regression test",
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, BlockLALSuite())

        import h5py
        import numpy
        import torch

        from pycbc.scheme import TorchScheme
        from pycbc.strain import apply_gates_to_fd
        from pycbc.types import TimeSeries
        from pycbc.types.array_torch import TorchArrayData
        from pycbc.waveform.generator import (
            TDomainSupernovaeGenerator,
            get_td_generator,
        )

        try:
            get_td_generator("__not_an_approximant__")
        except ValueError as exc:
            assert "No time-domain generator found" in str(exc)
        else:
            raise AssertionError("unknown approximant was accepted")

        assert (
            get_td_generator("CoreCollapseBounce")
            is TDomainSupernovaeGenerator
        )
        filename = os.environ["PYCBC_TEST_SUPERNOVA_FILE"]
        with h5py.File(filename, "w") as output:
            output["principal_components"] = numpy.array(
                [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
            )

        with TorchScheme("cpu"):
            times = TimeSeries(
                TorchArrayData(torch.ones(64)),
                delta_t=1.0 / 64.0,
                epoch=0,
                copy=False,
            )
            gated = apply_gates_to_fd(
                {"RF": times.to_frequencyseries()},
                {"RF": [(0.5, 0.1, 0.05)]},
            )["RF"]
            assert torch.all(torch.isfinite(gated._data.tensor))

            generator = TDomainSupernovaeGenerator(
                principal_components_file=filename,
                no_of_pcs=2,
                coefficients_array=torch.tensor([1.0, 2.0]),
                distance=10.0,
                delta_t=1.0 / 1024.0,
            )
            hp, hc = generator.generate()
            assert torch.all(torch.isfinite(hp._data.tensor))
            assert torch.count_nonzero(hc._data.tensor) == 0

        for name in (
            "lal",
            "lalsimulation",
            "lalframe",
            "pycbc.frame",
            "pycbc.io.hdf",
            "pycbc.strain.strain",
        ):
            assert name not in sys.modules, name
        """
    )
    env = dict(__import__("os").environ)
    env["PYCBC_TEST_SUPERNOVA_FILE"] = str(tmp_path / "supernova.hdf")
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_fallback_ligotimegps_gps_scale_precision_and_errors():
    """Keep GPS-scale floats precise and reject invalid source ranges."""
    from pycbc import lal_compat

    fallback = lal_compat._FallbackLIGOTimeGPS
    value = fallback(-493897162.0081482)
    assert _gps_fields(value) == (-493897162, -8148193)

    for invalid in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(RuntimeError):
            fallback(invalid)
        with pytest.raises(RuntimeError):
            value * invalid
        with pytest.raises(RuntimeError):
            value / invalid

    for invalid in (1 << 31, -(1 << 31) - 1):
        with pytest.raises(RuntimeError):
            fallback(invalid)
    for invalid in ((1 << 31, 0), (-(1 << 31) - 1, 0)):
        with pytest.raises(OverflowError):
            fallback(*invalid)

    assert _gps_fields(fallback(-(1 << 31))) == (-(1 << 31), 0)
    assert _gps_fields(fallback(-(1 << 31), 0)) == (-(1 << 31), 0)
    assert _gps_fields(fallback((1 << 31) - 1, 0)) == (
        (1 << 31) - 1,
        0,
    )
    assert _gps_fields(fallback("\t 1")) == (1, 0)
    with pytest.raises(RuntimeError):
        fallback("1 ")
    with pytest.raises(RuntimeError):
        value / 0

    # This input cycles in XLALGPSDivide. The fallback must fail instead of
    # hanging a LAL-free application indefinitely.
    with pytest.raises(RuntimeError, match="did not converge"):
        fallback(-231572134, -443382263) / 0.6649662727182819


def test_fallback_ligotimegps_gps_scale_differential():
    """Compare constructor, scalar arithmetic, and multiplication to LAL."""
    lal = pytest.importorskip("lal")
    from pycbc import lal_compat

    fallback = lal_compat._FallbackLIGOTimeGPS
    rng = random.Random(20260824)
    values = [-493897162.0081482]
    values.extend(rng.uniform(-2.0e9, 2.0e9) for _ in range(250))

    for value in values:
        expected = lal.LIGOTimeGPS(value)
        actual = fallback(value)
        assert _gps_fields(actual) == _gps_fields(expected)

        offset = rng.uniform(-1.0e6, 1.0e6)
        factor = rng.uniform(0.25, 1.0)
        for lal_result, fallback_result in (
            (expected + offset, actual + offset),
            (expected - offset, actual - offset),
            (expected * factor, actual * factor),
        ):
            assert _gps_fields(fallback_result) == _gps_fields(lal_result)


def test_fallback_ligotimegps_division_differential_is_timeout_safe():
    """Compare safe GPS-scale divisions without risking a LAL cycle hang."""
    pytest.importorskip("lal")
    script = textwrap.dedent(
        r"""
        import random

        import lal

        from pycbc import lal_compat

        fallback = lal_compat._FallbackLIGOTimeGPS
        rng = random.Random(20260824)
        for _ in range(100):
            seconds = rng.randint(-500_000_000, 500_000_000)
            nanoseconds = rng.randint(-999_999_999, 999_999_999)
            divisor = rng.uniform(1.0, 2.0)
            expected = lal.LIGOTimeGPS(seconds, nanoseconds) / divisor
            actual = fallback(seconds, nanoseconds) / divisor
            assert (
                actual.gpsSeconds,
                actual.gpsNanoSeconds,
            ) == (
                expected.gpsSeconds,
                expected.gpsNanoSeconds,
            )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_waveform_generator_native_path_without_lalsuite():
    """Keep native radiation-frame generators independent of LALSuite."""
    script = textwrap.dedent(
        r"""
        import importlib.abc
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
                        "LALSuite blocked by regression test",
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, BlockLALSuite())

        import torch

        from pycbc import lal_compat
        from pycbc.scheme import TorchScheme
        from pycbc.waveform.generator import (
            FDomainCBCGenerator,
            FDomainDetFrameGenerator,
            get_fd_generator,
        )

        lazy_modules = (
            "pycbc.detector",
            "pycbc.strain",
            "pycbc.waveform.supernovae",
        )
        assert all(name not in sys.modules for name in lazy_modules)

        parameters = {
            "approximant": "TaylorF2",
            "mass1": 10.0,
            "mass2": 9.0,
            "spin1z": 0.1,
            "delta_f": 1.0,
            "f_lower": 30.0,
            "f_final": 128.0,
        }
        with TorchScheme("cpu"):
            generator_class = get_fd_generator("TaylorF2")
            assert generator_class is FDomainCBCGenerator

            radiation_generator = generator_class(**parameters)
            hp, hc = radiation_generator.generate()
            assert len(hp) == 129
            assert len(hc) == 129
            assert torch.all(torch.isfinite(hp._data.tensor))
            assert torch.all(torch.isfinite(hc._data.tensor))

            frame_generator = FDomainDetFrameGenerator(
                generator_class,
                epoch=0,
                detectors=None,
                **parameters,
            )
            radiation_frame = frame_generator.generate()
            assert set(radiation_frame) == {"RF"}
            assert isinstance(
                radiation_frame["RF"]._data.tensor, torch.Tensor
            )

            det_generator = FDomainDetFrameGenerator(
                generator_class,
                epoch=0,
                detectors=("H1",),
                tc=0.0,
                polarization=0.0,
                dec=0.0,
                ra=0.0,
                **parameters,
            )
            det_frame = det_generator.generate()
            assert "H1" in det_frame
            assert isinstance(det_frame["H1"]._data.tensor, torch.Tensor)

        assert "pycbc.waveform.supernovae" not in sys.modules
        assert "lal" not in sys.modules
        assert "lalsimulation" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_public_optimized_match_and_torch_filters_without_lalsuite():
    """Keep public optimized matching and Torch filters LAL-independent."""
    script = textwrap.dedent(
        r"""
        import importlib.abc
        import math
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
                        "LALSuite blocked by regression test",
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, BlockLALSuite())

        import torch

        from pycbc import lal_compat
        from pycbc.filter import (
            highpass,
            lowpass,
            optimized_match,
            resample_to_delta_t,
        )
        from pycbc.scheme import TorchScheme
        from pycbc.types import FrequencySeries, TimeSeries

        event_modules = ("pycbc.events", "pycbc.events.ranking")
        assert all(name not in sys.modules for name in event_modules)

        with TorchScheme("cpu"):
            series = FrequencySeries(
                [0, 1 + 2j, 3 - 1j, 2 + 0.5j, 0],
                delta_f=1.0,
                dtype="complex128",
            )
            shifted = series.cyclic_time_shift(0.03125)
            match, index, phase = optimized_match(
                series, shifted, return_phase=True
            )
            assert abs(match - 1.0) < 1e-12
            assert abs(index - 0.25) < 1e-5
            assert abs(phase) < 1e-5

            times = TimeSeries(
                [math.sin(index * 0.1) for index in range(256)],
                delta_t=1.0 / 256.0,
                dtype="float64",
            )
            outputs = (
                resample_to_delta_t(times, 1.0 / 128.0),
                highpass(times, 16.0),
                lowpass(times, 32.0),
            )
            for output in outputs:
                assert isinstance(output._data.tensor, torch.Tensor)
                assert torch.all(torch.isfinite(output._data.tensor))

        cpu_times = TimeSeries(
            [math.sin(index * 0.1) for index in range(256)],
            delta_t=1.0 / 256.0,
            dtype="float64",
        )
        cpu_calls = (
            (resample_to_delta_t, (cpu_times, 1.0 / 128.0)),
            (highpass, (cpu_times, 16.0)),
            (lowpass, (cpu_times, 32.0)),
        )
        for function, arguments in cpu_calls:
            try:
                function(*arguments)
            except lal_compat.LALDependencyError as exc:
                assert "requires the core 'lal' Python package" in str(exc)
            else:
                raise AssertionError("a CPU LAL operation was accepted")

        assert all(name not in sys.modules for name in event_modules)
        assert "lal" not in sys.modules
        assert "lalsimulation" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
