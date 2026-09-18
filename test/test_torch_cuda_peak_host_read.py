"""CUDA peak arrays must be ready for immediate NumPy consumers."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc.filter import matchedfilter_torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("custom_stream", (False, True))
@pytest.mark.parametrize("delayed_copy", (0, 2))
def test_cuda_peak_host_arrays_ready_at_return(
    monkeypatch, native, custom_stream, delayed_copy
):
    if native and not matchedfilter_torch._HAS_TRITON:
        pytest.skip("Triton required for native peak route")
    monkeypatch.setenv("PYCBC_TORCH_CUDA_NATIVE_BATCH_PEAK", str(int(native)))
    stream = torch.cuda.Stream() if custom_stream else torch.cuda.default_stream()
    transfers = []
    original_to = torch.Tensor.to

    def delayed_to(tensor, *args, **kwargs):
        if tensor.is_cuda and kwargs.get("device") == "cpu":
            if len(transfers) == delayed_copy:
                torch.cuda._sleep(250_000_000)
            result = original_to(tensor, *args, **kwargs)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(tensor.device))
            transfers.append(event)
            return result
        return original_to(tensor, *args, **kwargs)

    with torch.cuda.stream(stream):
        values = torch.zeros((64, 128), dtype=torch.complex64, device="cuda")
        values[5, 42] = 8 + 6j
        values[20, 99] = -12j
        monkeypatch.setattr(torch.Tensor, "to", delayed_to)
        try:
            result = matchedfilter_torch._torch_batch_peak_and_threshold_gpu(
                values, np.ones(64, dtype=np.float64), 9.0
            )
            # Check before any test-side synchronization. The archived race
            # probe checked settled values only and could exit successfully.
            np.testing.assert_array_equal(result[0], [5, 20])
            np.testing.assert_array_equal(result[1], [42, 99])
            np.testing.assert_array_equal(result[2], [8 + 6j, -12j])
            assert result[3] is False
            assert len(transfers) == 3
            assert all(event.query() for event in transfers)
        finally:
            stream.synchronize()
