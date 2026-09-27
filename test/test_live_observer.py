"""Observation must preserve upstream calculations and evidence contents."""
import ast
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from tools.observe_pycbc_live import Observer, instrument_source

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '40e94792b3edf59f39b18b65102b28a4f74433a7'
SOURCE = '''
while data_end() < args.end_time:
    results = {}
    for ifo in data_reader:
        status = data_reader[ifo].advance(args.analysis_chunk)
        results[ifo] = mf.process_data(data_reader[ifo])
    if evnt.rank > 0:
        evnt.commit_results(results)
    else:
        evnt.dump(results)
'''


def test_original_statements_are_unchanged():
    source = subprocess.check_output(
        ['git', 'show', f'{BASELINE}:bin/pycbc_live'], cwd=ROOT, text=True)
    observed = instrument_source(source)
    compile(observed, 'pycbc_live', 'exec')

    class RemoveObserver(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == '_benchmark_observer'):
                return None
            return node

    restored = RemoveObserver().visit(observed)
    assert ast.dump(restored) == ast.dump(ast.parse(source))


def test_observed_main_preserves_pool_function_pickling(tmp_path, monkeypatch):
    import multiprocessing
    import sys
    from tools import observe_pycbc_live as observer

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("upstream coincidence pool requires fork")
    script = tmp_path / "pool_script.py"
    output = tmp_path / "result.json"
    script.write_text('''
import multiprocessing
import json
import sys
def calculate(value):
    return value * 2
with multiprocessing.get_context("fork").Pool(1) as pool:
    result = pool.map(calculate, [3, 5])
with open(sys.argv[1], "w") as stream:
    json.dump(result, stream)
''')
    monkeypatch.setattr(observer, "instrument_source", ast.parse)
    original = sys.modules["__main__"]
    original_argv, original_path = sys.argv, sys.path[0]
    observer.main([str(script), str(output)])
    assert json.loads(output.read_text()) == [6, 10]
    assert sys.modules["__main__"] is original
    assert sys.argv is original_argv
    assert sys.path[0] == original_path


@pytest.mark.parametrize('rank', [0, 1])
def test_observed_execution_preserves_calculation_calls_and_results(rank):
    def run(observe):
        calls, hooks = [], []
        reader = SimpleNamespace(end=10, value=np.array([1, 2, 3], dtype=np.float32))

        def advance(chunk):
            calls.append(('advance', chunk))
            reader.end += chunk
            reader.value += 1
            return True

        def process_data(data):
            calls.append(('filter', data.value.tolist()))
            return (data.value * 2).tolist()

        def write(results):
            calls.append(('output', results))

        class RecordObserver:
            def __getattr__(self, name):
                return lambda scope, *args: hooks.append((name, args))

        reader.advance = advance
        scope = dict(data_reader={'H1': reader}, data_end=lambda: reader.end,
                     args=SimpleNamespace(end_time=14, analysis_chunk=2),
                     mf=SimpleNamespace(process_data=process_data),
                     evnt=SimpleNamespace(rank=rank, commit_results=write, dump=write),
                     _benchmark_observer=RecordObserver())
        tree = instrument_source(SOURCE) if observe else ast.parse(SOURCE)
        exec(compile(tree, 'synthetic_live', 'exec'), scope)
        return calls, hooks

    native, _ = run(False)
    observed, hooks = run(True)
    assert observed == native
    assert sum(name == 'start_block' for name, _ in hooks) == 2
    assert sum(name == 'end_block' for name, _ in hooks) == 2


@pytest.mark.parametrize('source', [
    SOURCE.replace('mf.process_data', 'mf.other'),
    SOURCE + '\nstatus = data_reader[ifo].advance(2)\n',
    SOURCE + '\n_benchmark_observer = None\n',
])
def test_unknown_source_fails_closed(source):
    with pytest.raises(ValueError):
        instrument_source(source)


def test_evidence_matches_candidate_writer(tmp_path, monkeypatch):
    # Load only the writer, avoiding executable imports and initialization.
    tree = ast.parse((ROOT / 'bin/pycbc_live').read_text())
    writer = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_write_benchmark_evidence')
    namespace = dict(os=os, h5py=h5py, numpy=np,
                     mpi=SimpleNamespace(COMM_WORLD=SimpleNamespace(Get_rank=lambda: 0)))
    exec(compile(ast.Module(body=[writer], type_ignores=[]), 'writer', 'exec'), namespace)

    class Series(np.ndarray):
        pass

    strain = np.array([1, 2, 3], dtype=np.float32).view(Series)
    strain.start_time, strain.delta_t = 8, 1 / 2048
    psd = np.array([np.inf, 2, 3], dtype=np.float64).view(Series)
    psd.delta_f = 0.25
    readers = {'H1': SimpleNamespace(strain=strain)}
    psds = {'H1': psd}
    monkeypatch.setenv('PYCBC_SCIENCE_CONFIG', '{"sample_rate":2048}')
    monkeypatch.setenv('PYCBC_OBSERVER_EVIDENCE', str(tmp_path / 'reference'))
    observer = Observer()
    observer.block_start = 10
    observer.save_evidence(dict(evnt=SimpleNamespace(rank=0), data_reader=readers,
                                psds=psds, data_end=lambda: 12))
    namespace['_write_benchmark_evidence'](str(tmp_path / 'candidate'), readers, psds, 10, 12)
    from tools.benchmark_science import read_hdf, exact
    a, aa = read_hdf(next((tmp_path / 'reference').glob('*.hdf')))
    b, ba = read_hdf(next((tmp_path / 'candidate').glob('*.hdf')))
    for left, right in ((a, b), (aa, ba)):
        assert left.keys() == right.keys()
        assert all(exact(left[key], right[key]) for key in left)


def test_block_events_measure_completed_interval(monkeypatch, capsys):
    monkeypatch.setenv('PYCBC_OBSERVER_REPLAY_MODE', 'unpaced')
    observer = Observer()
    scope = dict(args=SimpleNamespace(analysis_chunk=2),
                 evnt=SimpleNamespace(rank=0, live_detectors={'H1'}),
                 data_end=lambda: 10)
    observer.start_block(scope)
    scope['data_end'] = lambda: 12
    observer.end_block(scope)
    events = [json.loads(line.split(' ', 1)[1])
              for line in capsys.readouterr().err.splitlines()]
    assert events[0]['data_start'] == 10
    assert events[1]['data_end'] == 12
    assert events[1]['live_detectors'] == ['H1']
    assert events[1]['elapsed_sec'] >= 0


def test_filter_event_records_observed_buffer_dtypes(capsys):
    scope = dict(ifo='H1', evnt=SimpleNamespace(rank=1),
                 mf=SimpleNamespace(out_mem={1: np.zeros(3, dtype=np.complex64)},
                                    cout_mem={1: np.zeros(3, dtype=np.complex128)}),
                 data_reader={'H1': SimpleNamespace(strain=SimpleNamespace(sample_rate=2048))})
    Observer().stage(scope, 'filter', 'end')
    event = json.loads(capsys.readouterr().err.split(' ', 1)[1])
    assert event['signal_dtypes'] == ['complex128', 'complex64']
    assert event['sample_rate'] == 2048
