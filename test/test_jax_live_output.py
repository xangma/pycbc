"""Live output parity, batched readback, ownership and writer completion."""

import ast
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip('jax')
h5py = pytest.importorskip('h5py')

# The writer is intentionally independently testable without importing the
# events package's unrelated filtering and LAL extension dependencies.
_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    'live_output_jax_test_module', _ROOT / 'pycbc/events/live_output_jax.py'
)
output = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = output
_SPEC.loader.exec_module(output)


def _source_function(path, name, class_name=None, **namespace):
    tree = ast.parse(path.read_text())
    if class_name is not None:
        tree = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


def _psd(values, epoch, delta_f):
    class PSD:
        def numpy(self):
            return np.asarray(self._data)

    psd = PSD()
    psd._data, psd.epoch, psd.delta_f = values, epoch, delta_f
    psd.save = _source_function(
        _ROOT / 'pycbc/types/frequencyseries.py',
        'save',
        class_name='FrequencySeries',
        _numpy=np,
        h5py=h5py,
        _os=__import__('os'),
    ).__get__(psd)
    return psd


def _dump_reference(manager, results, name, **kwargs):
    from pycbc import scheme
    from pycbc.events.ranking import newsnr

    source = _ROOT / 'pycbc/live/cli.py'
    manager.serialize_value = _source_function(
        source, 'serialize_value', class_name='LiveEventManager'
    )
    manager.loudest_indices = _source_function(
        source,
        'loudest_indices',
        class_name='LiveEventManager',
        numpy=np,
        newsnr=newsnr,
    )
    dump = _source_function(
        source,
        'dump',
        class_name='LiveEventManager',
        os=__import__('os'),
        h5py=h5py,
        numpy=np,
        sys=sys,
        version=SimpleNamespace(git_verbose_msg='test-version'),
    )
    with scheme.CPUScheme():
        dump(manager, results, name, **kwargs)


def _assert_hdf_equal(first, second):
    with h5py.File(first) as left, h5py.File(second) as right:

        def inspect(key, item):
            other = right[key]
            assert type(item) is type(other)
            assert set(item.attrs) == set(other.attrs)
            for attr in item.attrs:
                np.testing.assert_array_equal(
                    item.attrs[attr], other.attrs[attr]
                )
            if isinstance(item, h5py.Dataset):
                assert item.dtype == other.dtype
                assert item.shape == other.shape
                assert item.compression == other.compression
                assert item.compression_opts == other.compression_opts
                assert item.shuffle == other.shuffle
                np.testing.assert_array_equal(item[()], other[()])

        assert set(left.attrs) == set(right.attrs)
        for attr in left.attrs:
            np.testing.assert_array_equal(left.attrs[attr], right.attrs[attr])
        left_names, right_names = [], []
        left.visit(left_names.append)
        right.visit(right_names.append)
        assert left_names == right_names
        left.visititems(inspect)


@pytest.mark.parametrize('epoch', [None, 1234567890.25])
def test_batched_snapshot_matches_native_dump_and_psd_encoding(
    tmp_path, monkeypatch, epoch
):
    import jax.numpy as jnp

    results = {
        'H1': {
            'snr': np.array([4, 9, 6], np.float32),
            'chisq': np.array([1, 3, 1], np.float32),
            'template_id': np.array([7, 2, 4], np.uint32),
        },
        'L1': {
            'snr': np.array([], np.float32),
            'chisq': np.array([], np.float32),
        },
    }
    raw = {
        'foreground/stat': np.float32(7.25),
        'foreground/ifos': np.array(['H1', 'L1']),
        'foreground/name': np.str_('HL'),
        'foreground/description': 'H1-L1',
        'foreground/empty': np.array([], np.int32),
    }
    gates = {'H1': [(1234567890.25, 0.125, 0.5)], 'L1': []}
    values = np.linspace(1, 10, 31).astype(np.float32)
    psds = {'H1': _psd(values, epoch, 0.25), 'L1': None}
    manager = SimpleNamespace(
        live_detectors={'H1', 'L1'}, get_out_dir_path=lambda _: tmp_path
    )
    _dump_reference(
        manager,
        results,
        'native',
        store_psd=psds,
        raw_results=raw,
        gates=gates,
        store_loudest_index=2,
    )

    def loudest(snr, chisq, count):
        # Exercise asynchronous device leaves while isolating this output
        # contract from the separately tested loudest-event kernel.
        assert isinstance(snr, jax.Array)
        # Supply the reference's index dtype; the writer must preserve the
        # dtype selected by its caller rather than normalizing index widths.
        with jax.enable_x64():
            return jnp.asarray(
                np.array([1, 2], np.int64)
                if len(snr)
                else np.array([], np.int64)
            )

    monkeypatch.setattr(output, '_loudest_indices', loudest)
    device_results = jax.tree.map(jnp.asarray, results)
    psds['H1']._data = SimpleNamespace(array=jnp.asarray(values))
    readbacks = []
    original_device_get = jax.device_get

    def device_get(tree):
        readbacks.append(tree)
        return original_device_get(tree)

    monkeypatch.setattr(jax, 'device_get', device_get)
    snapshot = output.snapshot_live_output(
        tmp_path / 'jax.hdf',
        {
            'pycbc_version': 'test-version',
            'command_line': sys.argv,
            'num_live_detectors': 2,
        },
        device_results,
        raw_results=raw,
        gates=gates,
        store_psd=psds,
        store_loudest_index=2,
    )
    output.write_live_output(snapshot)
    assert len(readbacks) == 1
    # Eight columns share three dtype buffers at the terminal readback.
    transferred = jax.tree_util.tree_leaves(readbacks[0])
    assert len(transferred) == 3
    assert sum(value.size for value in transferred) == 42
    assert all(isinstance(value, jax.Array) for value in transferred)
    _assert_hdf_equal(tmp_path / 'native.hdf', tmp_path / 'jax.hdf')


@pytest.mark.parametrize('deferred', [False, True])
def test_snapshot_owns_mutable_inputs_until_async_write(tmp_path, deferred):
    values = np.array([2, 3, 5], np.float32)
    psd_values = np.linspace(1, 2, 9).astype(np.float32)
    argv = ['pycbc_live', '--store-psd']
    gates = {'H1': [(100, 0.25, 0.5)]}
    snapshot = output.snapshot_live_output(
        tmp_path / 'owned.hdf',
        {'command_line': argv},
        {'H1': {'snr': values}},
        raw_results={'foreground/stat': values},
        store_psd={'H1': _psd(psd_values, 100, 0.5)},
        gates=gates,
        defer_readback=deferred,
    )
    values[:] = -1
    psd_values[:] = -2
    argv[0] = 'changed'
    gates['H1'][0] = (-100, 0, 0)
    assert not snapshot.results['H1']['snr'].flags.writeable
    writer = output.JAXLiveOutputWriter()
    writer.submit(snapshot)
    writer.close()
    with h5py.File(snapshot.filename) as data:
        np.testing.assert_array_equal(data['H1/snr'], [2, 3, 5])
        np.testing.assert_array_equal(data['foreground/stat'], [2, 3, 5])
        np.testing.assert_array_equal(data['H1/psd'], np.linspace(1, 2, 9))
        assert data.attrs['command_line'][0] == 'pycbc_live'
        assert data['H1/gates'][0]['center_time'] == 100
    assert not writer._thread.is_alive()


def test_deferred_device_snapshot_collects_only_in_terminal_writer(
    tmp_path, monkeypatch
):
    """Submission proceeds while terminal output readback waits."""
    import jax.numpy as jnp

    entered, release = threading.Event(), threading.Event()
    threads = []
    original = jax.device_get

    def readback(values):
        threads.append(threading.current_thread().name)
        assert threads[-1] == 'pycbc-jax-output'
        entered.set()
        assert release.wait(5), 'terminal readback was not released'
        return original(values)

    device = jnp.array([2.0, 3.0, 5.0], dtype=jnp.float32)
    host = np.array([7, 11, 13], np.int32)
    monkeypatch.setattr(jax, 'device_get', readback)
    snapshot = output.snapshot_live_output(
        tmp_path / 'deferred.hdf',
        {'command_line': ['original']},
        {'H1': {'snr': device, 'template_id': host}},
        raw_results={'foreground/stat': device},
        defer_readback=True,
    )
    assert snapshot.device_pending and not threads
    assert snapshot.results['H1']['snr'] is device
    host[:] = -1
    writer = output.JAXLiveOutputWriter()
    try:
        writer.submit(snapshot)
        assert entered.wait(5)
        assert not Path(snapshot.filename).exists()
        # Array ownership is immutable: advancing a caller's device array
        # reference cannot change the already queued snapshot.
        device = device + 100
        release.set()
        writer.drain()
    finally:
        release.set()
        writer.close()
    assert threads == ['pycbc-jax-output']
    with h5py.File(snapshot.filename) as data:
        np.testing.assert_array_equal(data['H1/snr'], [2, 3, 5])
        np.testing.assert_array_equal(data['foreground/stat'], [2, 3, 5])
        np.testing.assert_array_equal(data['H1/template_id'], [7, 11, 13])


def test_deferred_readback_failure_surfaces_at_drain_and_joins(
    tmp_path, monkeypatch
):
    import jax.numpy as jnp

    snapshot = output.snapshot_live_output(
        tmp_path / 'failed-readback.hdf',
        {},
        {'H1': {'snr': jnp.ones(1)}},
        defer_readback=True,
    )

    def fail(_values):
        raise RuntimeError('device output unavailable')

    monkeypatch.setattr(jax, 'device_get', fail)
    writer = output.JAXLiveOutputWriter()
    try:
        writer.submit(snapshot)
    except RuntimeError:
        pass
    with pytest.raises(RuntimeError, match='failed-readback') as error:
        writer.drain()
    assert str(error.value.__cause__) == 'device output unavailable'
    with pytest.raises(RuntimeError, match='failed-readback'):
        writer.close()
    assert not writer._thread.is_alive()
    assert not Path(snapshot.filename).exists()


def test_resident_gate_metadata_is_materialized_only_at_writer_boundary(
    tmp_path,
):
    from dataclasses import dataclass

    called = []

    @dataclass(frozen=True)
    class GateSnapshot:
        epoch: float

        def snapshot(self):
            return self

        def materialize(self):
            called.append(threading.current_thread().name)
            return [(self.epoch, 0.25, 0.5)]

        def __array__(self, *_args, **_kwargs):
            raise AssertionError('pipeline coerced resident gate metadata')

    gate = GateSnapshot(100.0)
    snapshot = output.snapshot_live_output(
        tmp_path / 'gates.hdf', {}, {}, gates={'H1': gate}, defer_readback=True
    )
    assert snapshot.gates['H1'] is gate and not called
    writer = output.JAXLiveOutputWriter()
    writer.submit(snapshot)
    writer.close()
    assert called == ['pycbc-jax-output']
    with h5py.File(snapshot.filename) as data:
        np.testing.assert_array_equal(
            data['H1/gates'],
            np.array([(100.0, 0.25, 0.5)], output._GATE_DTYPE),
        )


def test_writer_bounds_queue_and_preserves_order():
    entered, release, third_done = (threading.Event() for _ in range(3))
    seen = []

    def write(snapshot):
        seen.append(snapshot.filename)
        if len(seen) == 1:
            entered.set()
            assert release.wait(5)

    writer = output.JAXLiveOutputWriter(queue_depth=1, write=write)
    writer.submit(SimpleNamespace(filename='first'))
    assert entered.wait(5)
    writer.submit(SimpleNamespace(filename='second'))
    assert writer._queue.qsize() == 1

    def submit_third():
        writer.submit(SimpleNamespace(filename='third'))
        third_done.set()

    producer = threading.Thread(target=submit_third)
    producer.start()
    assert not third_done.wait(0.05)
    release.set()
    assert third_done.wait(5)
    producer.join()
    writer.drain()
    writer.close()
    assert seen == ['first', 'second', 'third']
    with pytest.raises(RuntimeError, match='closed'):
        writer.submit(SimpleNamespace(filename='fourth'))


def test_writer_failure_is_raised_and_shutdown_joins_thread():
    failure = OSError('disk full')

    def write(snapshot):
        raise failure

    writer = output.JAXLiveOutputWriter(write=write)
    try:
        writer.submit(SimpleNamespace(filename='failed.hdf'))
    except RuntimeError as error:
        # A fast writer may expose the same failure before submit returns.
        assert error.__cause__ is failure
    with pytest.raises(RuntimeError, match='failed.hdf') as caught:
        writer.drain()
    assert caught.value.__cause__ is failure
    with pytest.raises(RuntimeError, match='failed.hdf'):
        writer.submit(SimpleNamespace(filename='next.hdf'))
    with pytest.raises(RuntimeError, match='failed.hdf'):
        writer.close()
    assert not writer._thread.is_alive()


@pytest.mark.parametrize('raise_in_loop', [False, True])
def test_output_scope_drains_on_completion_and_loop_exception(
    tmp_path, raise_in_loop
):
    manager = SimpleNamespace(rank=0)
    snapshot = output.snapshot_live_output(
        tmp_path / 'complete.hdf', {}, {'H1': {'snr': np.array([7])}}
    )

    def run():
        with output.live_output_scope(manager, enabled=True):
            writer = manager._jax_output_writer
            writer.submit(snapshot)
            if raise_in_loop:
                raise ValueError('pipeline failed')
            output.drain_live_output_jax(manager)
            assert Path(snapshot.filename).is_file()
        assert not writer._thread.is_alive()

    if raise_in_loop:
        with pytest.raises(ValueError, match='pipeline failed'):
            run()
    else:
        run()
    assert Path(snapshot.filename).is_file()
    assert not hasattr(manager, '_jax_output_writer')


def test_worker_rank_has_no_output_thread(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('worker started an output thread')

    monkeypatch.setattr(output, 'JAXLiveOutputWriter', forbidden)
    manager = SimpleNamespace(rank=1)
    with output.live_output_scope(manager, enabled=True):
        assert manager._jax_output_writer is None
        output.drain_live_output_jax(manager)
    assert not hasattr(manager, '_jax_output_writer')


def test_loop_exception_is_preserved_when_writer_also_fails(monkeypatch):
    original = output.JAXLiveOutputWriter
    writers = []
    release = threading.Event()

    def write(snapshot):
        assert release.wait(5)
        raise OSError('disk full')

    def make_writer(depth):
        writer = original(depth, write=write)
        writers.append(writer)
        return writer

    monkeypatch.setattr(output, 'JAXLiveOutputWriter', make_writer)
    manager = SimpleNamespace(rank=0)
    with pytest.raises(ValueError, match='pipeline failed') as caught:
        with output.live_output_scope(manager, enabled=True):
            manager._jax_output_writer.submit(
                SimpleNamespace(filename='failed.hdf')
            )
            release.set()
            raise ValueError('pipeline failed')
    assert any('disk full' in note for note in caught.value.__notes__)
    assert not writers[0]._thread.is_alive()
    assert not hasattr(manager, '_jax_output_writer')


@pytest.mark.parametrize('depth', [0, -1, 1.5, True])
def test_invalid_queue_depth_fails_before_starting_writer(depth):
    with pytest.raises(ValueError, match='positive'):
        output.JAXLiveOutputWriter(queue_depth=depth)


def test_native_output_selection_restores_tied_loudest_indices():
    from pycbc import scheme
    from pycbc.events import ranking
    import jax.numpy as jnp

    snr = np.tile(np.array([4.0, 9.0, 6.0], np.float32), 12)
    chisq = np.ones_like(snr)
    count = 5
    with scheme.CPUScheme():
        nsnr = ranking.newsnr(snr, chisq)
        expected = np.union1d(
            np.argsort(nsnr)[::-1][:count], np.argsort(snr)[::-1][:count]
        )
    with scheme.JAXScheme('cpu'):
        default = np.asarray(
            output._loudest_indices(
                jnp.asarray(snr), jnp.asarray(chisq), count
            )
        )
    assert not np.array_equal(default, expected)
    with scheme.JAXScheme(
        'cpu', reference_operations=('live_output_selection', 'newsnr')
    ):
        restored = output._loudest_indices(
            jnp.asarray(snr), jnp.asarray(chisq), count
        )
    assert restored.dtype == expected.dtype
    assert restored.tobytes() == expected.tobytes()


def test_native_background_selection_restores_partition_order(monkeypatch):
    from pycbc import scheme
    import jax.numpy as jnp

    values = np.random.default_rng(812).normal(size=64).astype(np.float32)
    expected = -np.partition(-values, 7)[:7]
    partition_calls = []
    original_partition = np.partition

    def native_partition(values, count):
        partition_calls.append((values.copy(), count))
        return original_partition(values, count)

    monkeypatch.setattr(output.np, 'partition', native_partition)
    with scheme.JAXScheme('cpu'):
        default = output.collect_live_background_jax(jnp.asarray(values), 7)
    np.testing.assert_array_equal(np.sort(default), np.sort(expected))
    # Both partition contracts permit this order to coincide on some builds.
    # Verify the selected implementation instead of requiring a mismatch.
    assert not partition_calls
    with scheme.JAXScheme(
        'cpu', reference_operations=('live_output_selection',)
    ):
        restored = output.collect_live_background_jax(jnp.asarray(values), 7)
        full = output.collect_live_background_jax(jnp.asarray(values))
    assert (
        restored.shape == expected.shape and restored.dtype == expected.dtype
    )
    assert restored.tobytes() == expected.tobytes()
    assert full.tobytes() == values.tobytes()
    assert len(partition_calls) == 1
    np.testing.assert_array_equal(partition_calls[0][0], -values)
    assert partition_calls[0][1] == 7
