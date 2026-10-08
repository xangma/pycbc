"""JAX integration must preserve existing native CPU execution paths."""


import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import FrequencySeries


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
