# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Unit tests for GPU strain autogating parity against reference CPU."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.strain.strain import detect_loud_glitches
from pycbc.types import TimeSeries


@pytest.fixture(params=["torch-cpu", "torch-cuda"])
def context(request):
    torch = pytest.importorskip("torch")
    device = request.param.removeprefix("torch-")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return scheme.TorchScheme(device, num_threads=1)


def test_detect_loud_glitches_parity(context):
    rate = 4096
    duration = 64
    n_samples = rate * duration
    rng = np.random.default_rng(12345)
    noise = rng.normal(0, 1.0, n_samples)
    strain = TimeSeries(
        noise.astype(np.float32), delta_t=1.0 / rate, epoch=1000000000
    )

    # 1. Baseline: no glitches
    with scheme.CPUScheme(1):
        cpu_glitches = detect_loud_glitches(
            strain, threshold=50.0, corrupt_time=4.0
        )
    with context:
        gpu_glitches = detect_loud_glitches(
            strain, threshold=50.0, corrupt_time=4.0
        )
    assert cpu_glitches == gpu_glitches == []
    assert isinstance(strain._data, np.ndarray)

    # 2. Injected transient
    strain_glitch = strain.copy()
    spike_idx = int(32.0 * rate)
    strain_glitch[spike_idx:spike_idx + 10] += 500.0

    with scheme.CPUScheme(1):
        cpu_glitches = detect_loud_glitches(
            strain_glitch, threshold=50.0, corrupt_time=4.0
        )
    with context:
        gpu_glitches = detect_loud_glitches(
            strain_glitch, threshold=50.0, corrupt_time=4.0
        )

    assert len(cpu_glitches) == len(gpu_glitches) > 0
    for c, g in zip(cpu_glitches, gpu_glitches):
        assert abs(c - g) == 0.0
    assert isinstance(strain_glitch._data, np.ndarray)
