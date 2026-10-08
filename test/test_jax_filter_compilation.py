"""Warmed bank kernels retain original normalization precision."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from jax_test_helpers import CompilationGuard  # noqa: E402
from pycbc import scheme  # noqa: E402
from pycbc.filter import sigmasq  # noqa: E402
from pycbc.filter.matchedfilter_jax import (  # noqa: E402
    batch_matched_filter_bank)
from pycbc.types import FrequencySeries  # noqa: E402


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_bank_reuses_compilation_with_original_normalization_dtype(device):
    if device == "cuda:0" and not any(
            item.platform == "gpu" for item in jax.devices()):
        pytest.skip("CUDA JAX device unavailable")
    templates = np.ones((2, 65), dtype=np.complex64)
    changed = templates * np.complex64(2)
    strain = np.ones(65, dtype=np.complex64)
    psd = np.ones(65, dtype=np.float32)
    with scheme.CPUScheme():
        native_weight = FrequencySeries(psd, delta_f=16)
        expected = np.asarray([
            sigmasq(FrequencySeries(row, delta_f=16), native_weight,
                    low_frequency_cutoff=32, high_frequency_cutoff=512)
            for row in changed])

    with scheme.JAXScheme(device) as context:
        original_inputs = tuple(jax.device_put(value, context.jax_device)
                                for value in (templates, strain, psd))
        changed_inputs = (jax.device_put(changed, context.jax_device),
                          *original_inputs[1:])
        traces = []

        @jax.jit
        def kernel(bank, data, weight):
            traces.append(bank.shape)
            return batch_matched_filter_bank(
                bank, data, weight, delta_f=16,
                low_frequency_cutoff=32, high_frequency_cutoff=512)

        guard = CompilationGuard()
        try:
            jax.block_until_ready(kernel(*original_inputs))
            before = guard.snapshot()
            with guard.timed_guard():
                snr, normalization = kernel(*changed_inputs)
                jax.block_until_ready((snr, normalization))
            assert guard.snapshot() == before
            assert traces == [(2, 65)]
            assert snr.dtype == templates.dtype
            assert normalization.dtype == expected.dtype
            assert normalization.shape == expected.shape
            assert np.asarray(normalization).tobytes() == expected.tobytes()
            assert snr.devices() == normalization.devices() == {
                context.jax_device}
        finally:
            guard.close()
