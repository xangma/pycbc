"""JAX integration must preserve existing native CPU execution paths."""

import builtins

import numpy as np
import pytest

from pycbc import scheme
from pycbc.filter.matchedfilter import MatchedFilterControl
from pycbc.types import FrequencySeries
from pycbc.vetoes.chisq import power_chisq_at_points_from_precomputed


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_native_detector_projection_keeps_absolute_time_arithmetic(dtype):
    from pycbc.detector import Detector
    from pycbc.waveform.generator import FDomainDetFrameGenerator
    from pycbc.waveform.utils import apply_fd_time_shift

    epoch = 1126259460.0
    reference_time = epoch + 2.125

    class StaticGenerator:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            hp = FrequencySeries(np.ones(1025, dtype=dtype), delta_f=1.0)
            hc = FrequencySeries(1j * np.ones(1025, dtype=dtype), delta_f=1.0)
            return hp, hc

    with scheme.CPUScheme():
        generator = FDomainDetFrameGenerator(
            StaticGenerator, epoch=epoch, detectors=["H1"],
            ra=1.1, dec=-0.4, tc=reference_time, polarization=0.3,
        )
        actual = generator.generate()["H1"]
        detector = Detector("H1")
        arrival = detector.arrival_time(reference_time, 1.1, -0.4, "geocentric")
        fp, fc = detector.antenna_pattern(1.1, -0.4, 0.3, arrival)
        hp, hc = StaticGenerator().generate()
        hp._epoch = hc._epoch = epoch
        expected = apply_fd_time_shift(fp * hp + fc * hc, arrival)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())


@pytest.mark.parametrize("derived", [False, True])
def test_native_chisq_keeps_compiled_dispatch(derived, monkeypatch):
    from pycbc.vetoes.chisq_cpu import shift_sum

    class DerivedCPU(scheme.CPUScheme):
        pass

    if derived:
        monkeypatch.setitem(scheme.scheme_prefix, DerivedCPU, "cpu")
    context = DerivedCPU() if derived else scheme.CPUScheme()
    rng = np.random.default_rng(810)
    ntime = 2097152
    corr = np.zeros(ntime, dtype=np.complex64)
    corr[15360:146432] = (
        rng.normal(size=131072) + 1j * rng.normal(size=131072)
    ).astype(np.complex64)
    points = np.array([1276244, 1500001], dtype=np.uint32)
    bins = np.array([15360, 115363, 146432], dtype=np.uint32)
    snrv = np.array([2+3j, 1-2j], dtype=np.complex64)
    with context:
        series = FrequencySeries(corr, delta_f=0.25)
        expected = (2 * shift_sum(series, points, bins) - abs(snrv)**2) * 0.25
        actual = power_chisq_at_points_from_precomputed(
            series, snrv, 0.5, bins, points
        )
    np.testing.assert_array_equal(actual, expected)


def test_native_cache_clear_does_not_import_jax_backend(monkeypatch):
    original_import = builtins.__import__

    def reject_jax(name, *args, **kwargs):
        if name == "pycbc.filter.matchedfilter_jax":
            raise AssertionError("CPU cache clearing imported the JAX backend")
        return original_import(name, *args, **kwargs)

    control = object.__new__(MatchedFilterControl)
    with scheme.CPUScheme():
        monkeypatch.setattr(builtins, "__import__", reject_jax)
        control.clear_batch_cache()
    assert control._cached_templates_2d is None


def test_jax_cache_clear_keeps_backend_cache_state():
    pytest.importorskip("jax")
    control = object.__new__(MatchedFilterControl)
    with scheme.JAXScheme("cpu"):
        control.clear_batch_cache()
    assert control._cached_templates_2d is None
