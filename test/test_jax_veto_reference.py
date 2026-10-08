"""Original CPU controls for JAX veto and phase calculation boundaries."""

import jax
import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import Array, FrequencySeries
from pycbc.vetoes import chisq
from pycbc.vetoes import chisq_jax
@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        try:
            jax.devices("cuda")
        except RuntimeError:
            pytest.skip("CUDA JAX device unavailable")
    return request.param


def _exact(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("cropped", [False, True])
def test_original_point_controls(device, dtype, cropped):
    rng = np.random.default_rng(71)
    row = (rng.normal(size=256) + 1j * rng.normal(size=256)).astype(dtype)
    row[:17] = 0
    row[201:] = 0
    bins = np.array([17, 61, 113, 201], dtype=np.uint32)
    points = np.array([3, 31, 125], dtype=np.uint32)
    snrv = np.array([1 + 2j, 2 - 3j, 4 + 1j], dtype=dtype)
    with scheme.CPUScheme():
        corr = FrequencySeries(row, delta_f=.5)
        expected_sum = chisq.shift_sum(corr, points, bins)
        expected_chisq = chisq.power_chisq_at_points_from_precomputed(
            corr, snrv, .123456789, bins, points)
    for operation, expected in [("shift_sum", expected_sum),
                                ("power_chisq_at_points", expected_chisq)]:
        with scheme.JAXScheme(device, reference_operations=[operation]):
            corr = FrequencySeries(row[17:201] if cropped else row, delta_f=.5)
            if cropped:
                corr._kmin, corr._tlen = 17, 256
            actual = (
                chisq.shift_sum(corr, points, bins)
                if operation == "shift_sum"
                else chisq.power_chisq_at_points_from_precomputed(
                    corr, snrv, 0.123456789, bins, points
                )
            )
            _exact(actual, expected)
            assert actual.device == scheme.mgr.state.jax_device


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_original_accumulation_and_bins(device, dtype):
    rng = np.random.default_rng(3)
    row = (rng.normal(size=129) + 1j * rng.normal(size=129)).astype(dtype)
    real_dtype = row.real.dtype
    weight = rng.uniform(.5, 1.5, size=129).astype(real_dtype)
    initial = rng.uniform(size=129).astype(real_dtype)
    with scheme.CPUScheme():
        output = Array(initial.copy())
        chisq.chisq_accum_bin(output, Array(row))
        expected_accum = output.numpy().copy()
        expected_bins = chisq.power_chisq_bins(
            FrequencySeries(row, delta_f=.5), 4,
            FrequencySeries(weight, delta_f=.5), 3, 50)
    with scheme.JAXScheme(device, reference_operations=["chisq_accum_bin"]):
        output = Array(initial.copy())
        chisq.chisq_accum_bin(output, Array(row))
        _exact(output.numpy(), expected_accum)
    with scheme.JAXScheme(device, reference_operations=["power_chisq_bins"]):
        actual = chisq.power_chisq_bins(
            FrequencySeries(row, delta_f=.5), 4,
            FrequencySeries(weight, delta_f=.5), 3, 50)
        _exact(actual, expected_bins)


def test_default_point_sum_stays_on_device(device, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("default called native CPU point calculation")

    monkeypatch.setattr(chisq_jax, "_point_chisq_cpu_compatible", forbidden)
    with scheme.JAXScheme(device):
        corr = FrequencySeries(np.ones(64, np.complex64), delta_f=1)
        result = chisq.shift_sum(corr, [3], [1, 16, 32])
        assert result.device == scheme.mgr.state.jax_device
