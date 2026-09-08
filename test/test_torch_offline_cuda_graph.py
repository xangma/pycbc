"""Offline graph fallback/lifetime contracts and real CUDA parity checks."""
from contextlib import nullcontext
import os
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from pycbc.filter import _torch_cuda_graph as graphs


def test_cpu_tensor_is_ineligible():
    with pytest.raises(ValueError, match='Unsupported'):
        graphs._tensor_binding(torch.ones(32, dtype=torch.complex64))


def test_binding_change_releases_before_fallback_and_requires_new_opt_in(monkeypatch):
    entry = SimpleNamespace(resources={'pid': os.getpid()}, close=mock.Mock(),
                            matches=mock.Mock(return_value=False))
    control = SimpleNamespace(_cuda_graphs={(0, 64): entry},
                              threshold_and_clusterers=[object()])
    monkeypatch.setattr(graphs, '_binding', lambda *a: ('changed', (), ()))
    assert graphs.replay_symmetric_cuda_graph(control, 0, 64, 1.0, 4.0) is None
    entry.close.assert_called_once()
    assert not control._cuda_graphs
    assert control._cuda_graph_rejected
    assert graphs.replay_symmetric_cuda_graph(control, 0, 64, 1.0, 4.0) is None
    entry.close.assert_called_once()
    graphs.clear_cuda_graphs(control)
    assert not control._cuda_graph_rejected


def test_clear_finishes_all_entries_even_when_one_close_fails():
    first = SimpleNamespace(resources={'pid': os.getpid()},
                            close=mock.Mock(side_effect=RuntimeError('cleanup')))
    second = SimpleNamespace(resources={'pid': os.getpid()}, close=mock.Mock())
    control = SimpleNamespace(_cuda_graphs={(0, 64): first, (1, 64): second})
    with pytest.raises(RuntimeError, match='cleanup'):
        graphs.clear_cuda_graphs(control)
    first.close.assert_called_once()
    second.close.assert_called_once()
    assert control._cuda_graphs == {}


def test_inherited_graph_cannot_be_released():
    entry = SimpleNamespace(resources={'pid': -1}, close=mock.Mock())
    control = SimpleNamespace(_cuda_graphs={(0, 64): entry})
    with pytest.raises(RuntimeError, match='fork'):
        graphs.clear_cuda_graphs(control)
    entry.close.assert_not_called()
    assert control._cuda_graphs[(0, 64)] is entry


def test_failed_synchronization_preserves_resources(monkeypatch):
    graph = mock.Mock()
    resources = dict(pid=os.getpid(), device='cuda:0', graph=graph,
                     tensors=(torch.ones(1),))
    monkeypatch.setattr(torch.cuda, 'device', lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, 'synchronize', mock.Mock(
        side_effect=RuntimeError('device failure')))
    with pytest.raises(RuntimeError, match='device failure'):
        graphs._release(resources)
    assert graphs._failed_cleanup[-1] is resources
    graph.reset.assert_not_called()
    graphs._failed_cleanup.pop()


def test_release_synchronizes_before_reset_and_drops_owners(monkeypatch):
    order = []
    graph = SimpleNamespace(reset=lambda: order.append('reset'))
    resources = dict(pid=os.getpid(), device='cuda:0', graph=graph,
                     tensors=(torch.ones(1),))
    monkeypatch.setattr(torch.cuda, 'device', lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: order.append('sync'))
    graphs._release(resources)
    assert order == ['sync', 'reset']
    assert not resources


def test_replay_failure_releases_entry_and_preserves_primary_exception(monkeypatch):
    graph = mock.Mock()
    graph.replay.side_effect = RuntimeError('replay failed')
    entry = SimpleNamespace(raw=torch.ones(()), resources={'pid': os.getpid(),
                            'graph': graph}, close=mock.Mock(
                                side_effect=RuntimeError('cleanup failed')),
                            record_stream=mock.Mock())
    control = SimpleNamespace(_cuda_graphs={(0, 64): entry})
    monkeypatch.setattr(graphs, 'capture_symmetric_cuda_graph', lambda *a: True)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: None)
    with pytest.raises(RuntimeError, match='replay failed'):
        graphs.replay_symmetric_cuda_graph(control, 0, 64, 1.0, 4.0)
    entry.close.assert_called_once()
    assert not control._cuda_graphs


@pytest.fixture
def cuda_control(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    from pycbc.events import threshold_torch
    if not threshold_torch._TRITON_AVAILABLE:
        pytest.skip('Triton is unavailable')
    from pycbc.filter.matchedfilter import MatchedFilterControl
    from pycbc.scheme import TorchScheme
    from pycbc.types import FrequencySeries, zeros
    monkeypatch.setenv('PYCBC_TORCH_CUDA_GRAPH', '0')
    rng = np.random.default_rng(70908)
    with TorchScheme('cuda'):
        n = 4096
        segments = []
        for analysis in (slice(256, 3800), slice(384, 3600)):
            values = (rng.normal(size=n // 2 + 1)
                      + 1j * rng.normal(size=n // 2 + 1)).astype(np.complex64)
            segment = FrequencySeries(values, delta_f=1.0)
            segment.analyze = analysis
            segments.append(segment)
        template = zeros(n, dtype=np.complex64)
        template._data.tensor.copy_(torch.from_numpy(
            (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
        ).to('cuda'))
        control = MatchedFilterControl(16.0, 1400.0, 4.00000012, n, 1.0,
                                       np.complex64, segments, template, True)
        try:
            yield control
        finally:
            control.clear_cuda_graphs()


def bits(value):
    if hasattr(value, '_data'):
        value = value._data.tensor
    return value.detach().cpu().contiguous().numpy().tobytes()


@pytest.mark.parametrize('custom_stream', [False, True])
def test_native_graph_matches_eager_with_changing_payloads_and_owned_results(
        cuda_control, custom_stream):
    control = cuda_control
    retained = []
    replay_probes = {}
    context = torch.cuda.stream(torch.cuda.Stream()) if custom_stream else nullcontext()
    with context:
        for variant, norm in enumerate((1.0, 2.0, 4.0)):
            if variant == 1:
                control.htilde._data.tensor.mul_(0.75 + 0.125j)
            elif variant == 2:
                control.htilde._data.tensor.zero_()
            for segment in range(2):
                inputs = tuple(bits(value) for value in
                               (control.htilde, control.segments[segment]))
                control._cuda_graph_enabled = False
                expected = control.full_matched_filter_and_cluster_symm(segment, norm, 64)
                expected_full = (bits(control.corr_mem), bits(control.snr_mem))
                expected_sparse = tuple(bits(value) for value in expected[3:]) if len(expected[3]) else ()
                assert control.capture_cuda_graph_symm(segment, 64, norm)
                entry = control._cuda_graphs[(segment, 64)]
                if segment not in replay_probes:
                    assert isinstance(entry.resources['graph'], torch.cuda.CUDAGraph)
                    replay_probes[segment] = mock.Mock(wraps=entry.resources['graph'])
                    entry.resources['graph'] = replay_probes[segment]
                versions = tuple(value._data.tensor._version for value in
                                 (control.corr_mem, control.snr_mem))
                actual = control.full_matched_filter_and_cluster_symm(segment, norm, 64)
                assert tuple(value._data.tensor._version for value in
                             (control.corr_mem, control.snr_mem)) == tuple(v + 1 for v in versions)
                assert (bits(control.corr_mem), bits(control.snr_mem)) == expected_full
                assert tuple(bits(value) for value in
                             (control.htilde, control.segments[segment])) == inputs
                if expected_sparse:
                    assert actual[1] == expected[1]
                    assert tuple(bits(value) for value in actual[3:]) == expected_sparse
                    retained.extend((value, bits(value)) for value in actual[3:])
                else:
                    assert actual == ([], [], [], [], []) or actual == [[], [], [], [], []]
                assert all(bits(value) == original for value, original in retained)
        assert len(control._cuda_graphs) == 2
        assert all(probe.replay.call_count == 3 for probe in replay_probes.values())


@pytest.mark.parametrize('change', ['window', 'stream', 'scratch', 'requires_grad',
                                   'template_storage', 'analysis', 'execute'])
def test_native_binding_changes_fall_back_before_replay(cuda_control, change):
    control = cuda_control
    assert control.capture_cuda_graph_symm(0, 64, 1.0)
    entry = control._cuda_graphs[(0, 64)]
    actual_graph = entry.resources['graph']
    graph = mock.Mock(wraps=actual_graph)
    entry.resources['graph'] = graph
    window = 64
    context = nullcontext()
    if change == 'window':
        window = 128
    elif change == 'stream':
        context = torch.cuda.stream(torch.cuda.Stream())
    elif change == 'scratch':
        clusterer = control.threshold_and_clusterers[0]
        clusterer._triton_keep = torch.empty_like(clusterer._triton_keep)
    elif change == 'requires_grad':
        control.htilde._data.tensor.requires_grad_(True)
    elif change == 'template_storage':
        control.htilde._data.tensor.data = control.htilde._data.tensor.clone()
    elif change == 'analysis':
        control.segments[0].analyze = slice(257, 3800)
    elif change == 'execute':
        control.ifft.execute = mock.Mock(wraps=control.ifft.execute)
    with context:
        assert graphs.replay_symmetric_cuda_graph(control, 0, window, 1.0, 4.0) is None
    graph.replay.assert_not_called()
    graph.reset.assert_called_once()
    assert control._cuda_graph_rejected


@pytest.mark.parametrize('change', ['data', 'resize'])
def test_native_same_scratch_object_storage_change_retains_pending_allocations(
        cuda_control, monkeypatch, change):
    control = cuda_control
    clusterer = control.threshold_and_clusterers[0]
    allocation_stream = torch.cuda.Stream()
    allocation_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(allocation_stream):
        assert clusterer.prepare_symmetric_cuda_graph(64)
    torch.cuda.current_stream().wait_stream(allocation_stream)
    scratch = clusterer._triton_keep
    original_pointer = scratch.untyped_storage().data_ptr()
    records = []
    original_record = torch.Tensor.record_stream

    def observed_record(tensor, stream):
        records.append((tensor.untyped_storage().data_ptr(), stream.cuda_stream))
        original_record(tensor, stream)

    monkeypatch.setattr(torch.Tensor, 'record_stream', observed_record)
    assert control.capture_cuda_graph_symm(0, 64, 1.0)
    entry = control._cuda_graphs[(0, 64)]
    assert original_pointer in [storage.data_ptr() for storage in entry.resources['storages']]
    assert (original_pointer, entry.resources['stream'].cuda_stream) in records
    assert graphs.replay_symmetric_cuda_graph(control, 0, 64, 1.0, 4.0) is not None
    assert (original_pointer, torch.cuda.current_stream().cuda_stream) in records
    with torch.cuda.stream(allocation_stream):
        if change == 'data':
            scratch.data = torch.empty_like(scratch)
            assert original_pointer in [storage.data_ptr() for storage in entry.resources['storages']]
        else:
            scratch.resize_(scratch.numel() + 4096)
    assert clusterer._triton_keep is scratch
    assert graphs.replay_symmetric_cuda_graph(control, 0, 64, 1.0, 4.0) is None
    assert not entry.resources


def test_native_threshold_rounding_before_square(cuda_control):
    # An isolated candidate lies between the two possible rounded squares.
    clusterer = cuda_control.threshold_and_clusterers[0]
    series = clusterer.series
    series.zero_()
    series[128] = complex(1.0, 0.0003452669770922512)
    threshold = 1.00000003
    expected = clusterer.threshold_and_cluster(threshold, 64)
    assert expected[1].numpy().tolist() == [128]
    raw = torch.tensor(threshold, device='cuda', dtype=torch.float32)
    squared = torch.empty_like(raw)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                torch.mul(raw, raw, out=squared)
                clusterer.symmetric_cuda_graph_step(64, squared)
            with torch.cuda.graph(graph, stream=stream):
                torch.mul(raw, raw, out=squared)
                clusterer.symmetric_cuda_graph_step(64, squared)
        torch.cuda.current_stream().wait_stream(stream)
        for value in (threshold, 2.0, 0.0, threshold):
            expected = clusterer.threshold_and_cluster(value, 64)
            raw.fill_(value)
            graph.replay()
            actual = clusterer.symmetric_cuda_graph_result()
            assert tuple(map(bits, actual)) == tuple(map(bits, expected))
        old_square = torch.tensor(threshold * threshold, device='cuda', dtype=torch.float32)
        clusterer.symmetric_cuda_graph_step(64, old_square)
        assert len(clusterer.symmetric_cuda_graph_result()[1]) == 0
    finally:
        torch.cuda.synchronize()
        graph.reset()
