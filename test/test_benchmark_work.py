"""Completed work excludes overlaps and opt-in markers have no default output."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    'benchmark_hooks', Path(__file__).parents[1] / 'pycbc' / 'benchmark.py')
HOOKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOOKS)


def test_work_uses_unique_valid_samples(tmp_path, monkeypatch):
    path = tmp_path / 'work.json'
    monkeypatch.setenv('PYCBC_BENCHMARK_WORK', str(path))
    segments = [SimpleNamespace(cumulative_index=start, analyze=slice(20, 120),
                                dtype='complex64')
                for start in (0, 50, 300)]
    HOOKS.save_inspiral_work(segments, 10, 10)
    data = json.loads(path.read_text())
    assert data['valid_sample_intervals'] == [[0, 150], [300, 400]]
    assert data['valid_detector_seconds'] == 25
    assert data['completed_template_seconds'] == 250
    assert data['signal_dtypes'] == ['complex64']


def test_markers_opt_in_and_use_monotonic_clock(monkeypatch, capsys):
    monkeypatch.delenv('PYCBC_BENCHMARK_STAGES', raising=False)
    HOOKS.stage_event('search', 'start')
    assert capsys.readouterr().err == ''
    monkeypatch.setenv('PYCBC_BENCHMARK_STAGES', '1')
    HOOKS.stage_event('search', 'start', templates=10)
    marker = json.loads(capsys.readouterr().err.split(' ', 1)[1])
    assert marker['monotonic_ns'] > 0
    assert marker['pid'] > 0
    assert marker['templates'] == 10


def test_marker_records_rank_and_profile_synchronization(monkeypatch, capsys):
    monkeypatch.setenv('PYCBC_BENCHMARK_STAGES', '1')
    monkeypatch.setenv('OMPI_COMM_WORLD_RANK', '3')
    monkeypatch.setattr(HOOKS, 'synchronize_jax_stage', lambda: True)
    HOOKS.stage_event('filter_ifft', 'end')
    marker = json.loads(capsys.readouterr().err.split(' ', 1)[1])
    assert marker['rank'] == 3
    assert marker['synchronized'] is True


def test_work_receipt_records_jax_chisq_mode(monkeypatch, tmp_path):
    path = tmp_path / 'work.json'
    monkeypatch.setenv('PYCBC_BENCHMARK_WORK', str(path))
    monkeypatch.setenv('PYCBC_BENCHMARK_SCIENCE_CONFIG',
                       json.dumps({'--jax-chisq-mode': 'direct-phase'}))
    segment = SimpleNamespace(cumulative_index=0, analyze=slice(0, 1),
                              dtype='complex64')
    HOOKS.save_inspiral_work([segment], 1, 1)
    assert json.loads(path.read_text())['jax_chisq_mode'] == 'direct-phase'
