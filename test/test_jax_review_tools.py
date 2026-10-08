# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scientific comparison rejects changed values and unqualified inventories."""
from pathlib import Path
import runpy

import h5py
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
compare = runpy.run_path(str(TOOLS / 'compare_jax_search.py'))
benchmark = runpy.run_path(str(TOOLS / 'benchmark_jax_fft.py'))


def write(path, values, *, compression=None, epoch=1000):
    with h5py.File(path, 'w') as file:
        file.create_dataset('H1/snr', data=values, compression=compression)
        file.attrs['epoch'] = epoch
        file.attrs['command_line'] = 'analysis'
        file.create_dataset('labels', data=np.array(['H1', 'L1'], dtype=h5py.string_dtype()))


@pytest.mark.parametrize('change', ['dtype', 'signed_zero', 'nan_payload', 'compression', 'metadata'])
def test_comparison_rejects_non_equivalent_results(tmp_path, change):
    a, b = tmp_path / 'reference.hdf', tmp_path / 'candidate.hdf'
    values = np.array([0., 2., np.nan], np.float64)
    write(a, values)
    changed = values.copy()
    kwargs = {}
    if change == 'dtype':
        changed = changed.astype(np.float32)
    elif change == 'signed_zero':
        changed[0] = -0.
    elif change == 'nan_payload':
        changed.view(np.uint64)[2] = 0x7ff8000000000001
    elif change == 'compression':
        kwargs['compression'] = 'gzip'
    elif change == 'metadata':
        kwargs['epoch'] = 1001
    write(b, changed, **kwargs)
    result = compare['compare_files'](a, b)
    assert not result['passed'] and result['errors']


def test_comparison_records_explicit_provenance_exclusion(tmp_path):
    a, b = tmp_path / 'a.hdf', tmp_path / 'b.hdf'
    write(a, np.array([1., 2.]))
    write(b, np.array([1., 2.]))
    with h5py.File(b, 'r+') as file:
        file.attrs['command_line'] = 'jax_analysis'
    assert not compare['compare_files'](a, b)['passed']
    result = compare['compare_files'](a, b, ignore_attributes=['command_line'])
    assert result['passed']
    assert result['excluded'] == [dict(path='/@command_line', kind='attribute')]


def test_comparison_rejects_empty_or_missing_output_sets(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    with pytest.raises(ValueError, match='nonempty'):
        compare['compare_outputs'](a, b)
    write(a / 'output.hdf', np.array([1., 2.]))
    with pytest.raises(ValueError, match='identical'):
        compare['compare_outputs'](a, b)


@pytest.mark.parametrize('device', ['cpu', 'jax:cpu', 'jax:cuda:0'])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_benchmark_finishes_actual_fft_work_and_restores_backend(device, dtype):
    if device.startswith('jax:'):
        jax = pytest.importorskip('jax')
        if ':cuda' in device and not any(d.platform == 'gpu' for d in jax.devices()):
            pytest.skip('CUDA device unavailable')
    from pycbc.fft import backend_cpu
    previous = backend_cpu.cpu_backend
    record = benchmark['benchmark'](device, dtype, length=64, repeats=2)
    assert backend_cpu.cpu_backend == previous
    assert len(record['samples_seconds']) == 2
    assert record['first_seconds'] > 0 and record['median_seconds'] > 0
    assert record['max_roundtrip_error'] < (1e-5 if dtype == 'complex64' else 1e-12)


def test_benchmark_original_fft_controls_and_profile_complete(tmp_path):
    pytest.importorskip('jax')
    record = benchmark['benchmark']('jax:cpu', 'complex128', length=64, repeats=2,
                                    reference_operations=('fft', 'ifft'), profile=tmp_path)
    assert record['reference_operations'] == ['fft', 'ifft']
    assert record['max_roundtrip_error'] < 1e-12
    assert list(tmp_path.rglob('*.trace.json.gz'))
