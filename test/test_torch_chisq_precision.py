"""Check point chi-squared phases across long correlation vectors."""

import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import Array, FrequencySeries


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("array_indices", [False, True])
def test_long_shift_sum_matches_direct_phases(
    dtype, array_indices, monkeypatch
):
    triton_launches = []
    torch = pytest.importorskip("torch")
    from pycbc.vetoes import chisq_torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if dtype == np.complex64:
        if not chisq_torch._HAS_TRITON:
            pytest.skip("Triton not available")
        kernel = chisq_torch._triton_pointwise_chisq_bin_kernel
        original_launch = kernel.run

        def record_launch(*args, **kwargs):
            result = original_launch(*args, **kwargs)
            if not kwargs.get("warmup", False):
                triton_launches.append(True)
            return result

        monkeypatch.setattr(kernel, "run", record_launch)
    processing_scheme = scheme.TorchScheme("cuda")

    size = 2**21
    rng = np.random.default_rng(782)
    values = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(dtype)
    points = np.array([137, 1884341, size - 3], dtype=np.int64)
    bins = np.array([15360, 65536, 262144, 1048576], dtype=np.uint32)
    # Evaluate direct complex128 phases independently of the recurrence. The
    # late sample indices and long final bin expose accumulated phase error.
    bin_sums = np.zeros((len(points), len(bins) - 1), dtype=np.complex128)
    for column, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
        for first in range(int(start), int(end), 32768):
            last = min(first + 32768, int(end))
            frequencies = np.arange(first, last, dtype=np.float64)
            phases = np.exp(2j * np.pi * points[:, None] * frequencies[None, :] / size)
            bin_sums[:, column] += np.sum(values[None, first:last] * phases, axis=1)

    # The mathematical shift-sum API keeps full-pi direct phases. Ordinary
    # power-chi-square searches separately preserve the historical CPU method.
    expected = np.sum(abs(bin_sums) ** 2, axis=1)

    with processing_scheme:
        correlation = FrequencySeries(values, delta_f=1 / 512)
        indices = Array(points) if array_indices else points
        actual = chisq_torch.shift_sum(correlation, indices, bins)
        assert actual._data.tensor.device.type == "cuda"
        actual = actual.numpy()

    if dtype == np.complex64:
        # Multiple points must exercise the fused kernel, not a fallback.
        assert triton_launches

    assert actual.dtype == (np.float32 if dtype == np.complex64 else np.float64)
    tolerance = 3e-6 if dtype == np.complex64 else 3e-8
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=1e-9)
