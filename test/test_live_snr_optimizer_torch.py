import numpy as np
import pytest

from pycbc import scheme
from pycbc.live import snr_optimizer
from pycbc.types import TimeSeries

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("device", ("cpu", "cuda", "mps"))
def test_live_snr_window_reduction_stays_on_torch_device(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")

    ctx = scheme.TorchScheme(device)
    try:
        with ctx:
            from pycbc.types.array_torch import TorchArrayData

            series = TimeSeries(
                np.array(
                    [1.0 + 2.0j, -3.0 + 4.0j, 0.5 - 0.25j],
                    dtype=np.complex64,
                ),
                delta_t=1.0 / 2048,
            )

            def reject_host_transfer(*args, **kwargs):
                raise AssertionError("SNR window moved to NumPy")

            def reject_element_iteration(*args, **kwargs):
                raise AssertionError("SNR window was reduced in Python")

            monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
            monkeypatch.setattr(TorchArrayData, "__getitem__", reject_element_iteration)

            actual = snr_optimizer._max_abs_snr(series)

            assert actual == pytest.approx(5.0)
            assert series._data.tensor.device.type == device
    finally:
        del ctx
        scheme.Scheme._single = None


def test_zeros_pinned_and_to_cuda_async():
    from pycbc.types.array_torch import zeros_pinned, to_cuda_async, TorchArrayData

    pinned = zeros_pinned(128, dtype=np.float32)
    assert isinstance(pinned, TorchArrayData)
    assert pinned.shape == (128,)
    assert pinned.dtype == np.dtype(np.float32)
    if torch.cuda.is_available():
        assert pinned.tensor.is_pinned()

    pinned_2d = zeros_pinned((4, 32), dtype=np.complex64)
    assert pinned_2d.shape == (4, 32)
    assert pinned_2d.dtype == np.dtype(np.complex64)

    # Test to_cuda_async on TorchArrayData
    out_arr = to_cuda_async(pinned)
    assert isinstance(out_arr, TorchArrayData)
    if torch.cuda.is_available():
        assert out_arr.tensor.device.type == "cuda"

    # Test to_cuda_async on raw torch.Tensor
    raw_tensor = pinned.tensor
    out_t = to_cuda_async(raw_tensor)
    assert isinstance(out_t, torch.Tensor)
    if torch.cuda.is_available():
        assert out_t.device.type == "cuda"

    # Test method on TorchArrayData
    out_m = pinned.to_cuda_async()
    assert isinstance(out_m, TorchArrayData)


def test_live_batch_matched_filter_cuda_graph_initialization(monkeypatch):
    from pycbc.filter import matchedfilter
    from pycbc.types import FrequencySeries

    t1 = FrequencySeries(np.ones(33, dtype=np.complex64), delta_f=0.25)
    t1.id = 1
    t1.params = np.array([(10.0,)], dtype=[("mass1", np.float32)])[0]

    # Default without env
    monkeypatch.delenv("PYCBC_ENABLE_CUDA_GRAPHS", raising=False)
    batch = matchedfilter.LiveBatchMatchedFilter(
        [t1], snr_threshold=5.0, chisq_bins=None, sg_chisq=None, maxelements=64
    )
    assert batch.enable_cuda_graphs is False
    assert batch._cuda_graphs == {}

    # Explicit True
    batch_true = matchedfilter.LiveBatchMatchedFilter(
        [t1],
        snr_threshold=5.0,
        chisq_bins=None,
        sg_chisq=None,
        maxelements=64,
        enable_cuda_graphs=True,
    )
    assert batch_true.enable_cuda_graphs is True

    # With env var = 1
    monkeypatch.setenv("PYCBC_ENABLE_CUDA_GRAPHS", "1")
    batch_env = matchedfilter.LiveBatchMatchedFilter(
        [t1], snr_threshold=5.0, chisq_bins=None, sg_chisq=None, maxelements=64
    )
    assert batch_env.enable_cuda_graphs is True


def test_live_batch_matched_filter_cuda_graph_lifecycle(monkeypatch):
    import types
    from pycbc.filter import matchedfilter
    from pycbc.types import FrequencySeries

    with scheme.TorchScheme("cpu"):
        size = 32
        fsize = size // 2 + 1
        rng = np.random.default_rng(1234)
        t_data = (
            rng.normal(size=fsize) + 1j * rng.normal(size=fsize)
        ).astype(np.complex64)
        t1 = FrequencySeries(t_data, delta_f=1.0 / size)
        t1.id = 1
        t1.params = np.array([(10.0,)], dtype=[("mass1", np.float32)])[0]
        t1.sigmasq = lambda _psd: 1.0

        batch = matchedfilter.LiveBatchMatchedFilter(
            [t1],
            snr_threshold=0.0,
            chisq_bins=0,
            sg_chisq=types.SimpleNamespace(),
            maxelements=size,
            enable_cuda_graphs=True,
        )

        class MockStream:
            def wait_stream(self, s):
                pass

            def synchronize(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockGraphContext:
            def __init__(self, graph, stream):
                self.graph = graph
                self.stream = stream

            def __enter__(self):
                self.graph._active = True
                return self

            def __exit__(self, *args):
                self.graph._active = False

        class MockCUDAGraph:
            def __init__(self):
                self.replay_count = 0
                self._active = False

            def replay(self):
                self.replay_count += 1

        orig_zeros = torch.zeros
        orig_zeros_like = torch.zeros_like
        orig_arange = torch.arange

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "current_stream", lambda device=None: MockStream()
        )
        monkeypatch.setattr(
            torch.cuda, "Stream", lambda device=None: MockStream()
        )
        monkeypatch.setattr(torch.cuda, "stream", lambda s: s)
        monkeypatch.setattr(
            torch.cuda,
            "graph",
            lambda g, stream=None: MockGraphContext(g, stream),
        )
        monkeypatch.setattr(torch.cuda, "CUDAGraph", MockCUDAGraph)
        monkeypatch.setattr(
            torch,
            "zeros",
            lambda *args, **kwargs: orig_zeros(
                *args, **{k: v for k, v in kwargs.items() if k != "device"}
            ),
        )
        monkeypatch.setattr(
            torch,
            "zeros_like",
            lambda tensor, **kwargs: orig_zeros_like(
                tensor, **{k: v for k, v in kwargs.items() if k != "device"}
            ),
        )
        monkeypatch.setattr(
            torch,
            "arange",
            lambda *args, **kwargs: orig_arange(
                *args, **{k: v for k, v in kwargs.items() if k != "device"}
            ),
        )

        class MockCudaTensor(torch.Tensor):
            @property
            def device(self):
                return torch.device("cuda:0")

        mid = batch.mids[0]
        batch.out_mem[mid]._data.tensor = (
            batch.out_mem[mid]._data.tensor.as_subclass(MockCudaTensor)
        )
        batch.cout_mem[mid]._data.tensor = (
            batch.cout_mem[mid]._data.tensor.as_subclass(MockCudaTensor)
        )

        d1 = FrequencySeries(
            (rng.normal(size=fsize) + 1j * rng.normal(size=fsize)).astype(
                np.complex64
            ),
            delta_f=1.0 / size,
        )
        d1.psd = object()
        d1._data.tensor = d1._data.tensor.as_subclass(MockCudaTensor)

        tgroup = batch.tgroups[0]
        psize = batch.chunk_tsamples[0]
        seg = slice(0, psize)

        # First call: warmup & capture
        peaks1 = batch._try_cuda_graph_batch(0, mid, tgroup, psize, seg, d1)
        assert peaks1 is not None
        assert 0 in batch._cuda_graphs
        graph_state = batch._cuda_graphs[0]
        graph = graph_state["graph"]
        assert graph.replay_count == 0
        assert graph_state["input_shape"] == tuple(d1._data.tensor.shape)

        # Second call: replay on same shape
        d2 = FrequencySeries(
            (rng.normal(size=fsize) + 1j * rng.normal(size=fsize)).astype(
                np.complex64
            ),
            delta_f=1.0 / size,
        )
        d2.psd = object()
        d2._data.tensor = d2._data.tensor.as_subclass(MockCudaTensor)

        peaks2 = batch._try_cuda_graph_batch(0, mid, tgroup, psize, seg, d2)
        assert peaks2 is not None
        assert graph.replay_count == 1

        # Third call: different size invalidates and recaptures
        d3 = FrequencySeries(
            (
                rng.normal(size=fsize * 2) + 1j * rng.normal(size=fsize * 2)
            ).astype(np.complex64),
            delta_f=1.0 / (size * 2),
        )
        d3.psd = object()
        d3._data.tensor = d3._data.tensor.as_subclass(MockCudaTensor)

        peaks3 = batch._try_cuda_graph_batch(0, mid, tgroup, psize, seg, d3)
        assert peaks3 is not None
        assert 0 in batch._cuda_graphs
        assert batch._cuda_graphs[0]["input_shape"] == tuple(
            d3._data.tensor.shape
        )
