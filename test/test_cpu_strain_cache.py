"""Preserve shared CPU plans while keeping Torch resources separate."""
import numpy as np
import pytest

from pycbc import scheme
from pycbc.strain.strain import create_memory_and_engine_for_class_based_fft


@pytest.fixture
def cache():
    factory = create_memory_and_engine_for_class_based_fft
    factory.cache_clear()
    yield factory
    factory.cache_clear()


@pytest.mark.parametrize("ifft", (False, True))
def test_fft_cache_reuses_cpu_buffers_across_contexts(cache, monkeypatch, ifft):
    monkeypatch.setattr(scheme.Scheme, "_single", None)
    original = cache(16, np.float32, ifft=ifft)
    context = scheme.CPUScheme(1)
    for threads in (1, 2):
        context.num_threads = threads
        with context:
            assert cache(16, np.float32, ifft=ifft) is original
    assert cache(16, np.float32, ifft=ifft, uid=1) is not original


def test_fft_cache_keeps_torch_buffers_separate(cache):
    pytest.importorskip("torch")
    original = cache(16, np.float32)
    with scheme.TorchScheme("cpu", num_threads=1):
        torch_result = cache(16, np.float32)
        assert torch_result is not original
        assert cache(16, np.float32) is torch_result
        assert torch_result[0].backend == "torch"
    with scheme.TorchScheme("cpu", num_threads=2):
        assert cache(16, np.float32) is not torch_result
    assert cache(16, np.float32) is original
