#!/usr/bin/env python3
"""Bind the reviewed promoted-MKL dispatch to the final precision matrix."""
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
COMMIT = 'a4d77a6d1863c0515e8dace64c5609b63d40b51e'
SIZES = [1048576, 2097152, 4194304]
ROUTE = 'mkl_double_workspace'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    output = ROOT / 'large-ifft-v6-decision.json'
    require(not output.exists(), 'Decision already exists')
    paths = [Path(__file__).resolve(), ROOT / 'source-v6.json',
             ROOT / 'qualify-large-ifft-v6b.py', ROOT / 'large-ifft-v6.json',
             ROOT / 'large-ifft-v6.log', ROOT / 'large-ifft-v6.status.json']
    inputs = {str(path): digest(path) for path in paths}
    source = json.loads((ROOT / 'source-v6.json').read_text())
    matrix = json.loads((ROOT / 'large-ifft-v6.json').read_text())
    runner = json.loads((ROOT / 'large-ifft-v6.status.json').read_text())
    require(source['commit'] == COMMIT, 'Wrong final source')
    state = {name: subprocess.check_output(['git', '-C', source['source'], *args], text=True).strip()
             for name, args in [('commit', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain'])]}
    require(state == dict(commit=COMMIT, status=''), 'Final source changed')
    for receipt in (source, runner):
        require(receipt['input_sha256'] == receipt['input_sha256_after'], 'Recorded input mutation')
        for path, expected in receipt['input_sha256'].items():
            require(digest(path) == expected, f'Input changed: {path}')
            inputs[path] = expected
    require(runner['state'] == 'complete' and runner['returncode'] == 0
            and runner['source_commit'] == COMMIT
            and runner['output_sha256'] == inputs[str(ROOT / 'large-ifft-v6.json')]
            and runner['log_sha256'] == inputs[str(ROOT / 'large-ifft-v6.log')], 'Matrix runner failed')
    require(matrix['state'] == 'complete' and matrix['schema'] == 'torch-large-ifft-qualification-v6b'
            and matrix['source_before'] == matrix['source_after']
            and matrix['source_before']['head'] == COMMIT and matrix['source_before']['status'] == ''
            and matrix['source_before']['root'] == source['source']
            and matrix['source_before']['harness_sha256'] == inputs[str(ROOT / 'qualify-large-ifft-v6b.py')]
            and matrix['threads'] == 1 and matrix['cpu_affinity'] == [8], 'Matrix source or execution differs')
    for name, expected in {**source['changed_files_sha256'], **source['native_modules_sha256']}.items():
        path = str(Path(source['source']) / name)
        require(digest(path) == expected, f'Final source input changed: {name}')
        inputs[path] = expected
    require(matrix['source_before']['torchfft_sha256'] == source['changed_files_sha256']['pycbc/fft/torchfft.py'],
            'Matrix FFT source differs')
    require([row['size'] for row in matrix['sizes']] == SIZES, 'Matrix sizes differ')
    expected_grid = set(itertools.product((7, 91, 812, 20260906), ('dense', 'banded', 'impulse'), (1e-12, 1., 1e12)))
    evidence = {}
    for row in matrix['sizes']:
        require(len(row['cases']) == 36 and {(case['seed'], case['pattern'], case['scale'])
                for case in row['cases']} == expected_grid, 'Precision coverage differs')
        for case in row['cases']:
            gate = case['gates'][ROUTE]
            require(gate['passed'] is True and gate['bitwise_mkl_parity'] is True, 'Promoted precision/parity failed')
            for metric in ('l2', 'max_abs'):
                actual, legacy = case['errors'][ROUTE][metric], case['errors']['legacy_fftw_single'][metric]
                require(math.isfinite(actual) and math.isfinite(legacy) and 0 <= actual <= legacy,
                        'Promoted result exceeds strict legacy FFTW error')
            require(case['errors']['current_torch_dispatch'] == case['errors'][ROUTE], 'Framework dispatch differs')
        for result in row['timings'].values():
            require(result['bound_buffers_verified'] is True and result['input_storage_preserved'] is True,
                    'Invalid timing buffer ownership')
        timing = row['timings']
        previous = timing['fftw_double_workspace_one_native_thread']['steady_median_seconds']
        chosen = timing[ROUTE]['steady_median_seconds']
        require(math.isfinite(previous) and math.isfinite(chosen) and 0 < chosen < previous, 'No measured gain')
        evidence[str(row['size'])] = dict(cases=36, legacy_workspace_seconds=previous,
                                        promoted_mkl_seconds=chosen, speedup=previous / chosen)
    result = dict(schema_version=1, state='complete', passed=True, returncode=0,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_info=state, source_after=state,
                  matrix=dict(path=str(ROOT / 'large-ifft-v6.json'), sha256=inputs[str(ROOT / 'large-ifft-v6.json')]),
                  dispatch_by_size={str(size): ROUTE for size in SIZES}, supported_threads=[1],
                  promoted_native_dtype='complex128', direct_single_sizes_unchanged=[32768],
                  rationale='All 108 final-source cases pass the unchanged strict legacy-FFTW error gate and bitwise '
                            'matching-precision standard-MKL parity. Promoted MKL is faster at all three sizes. '
                            'Larger direct-single routes remain disabled; other thread counts retain their prior paths.',
                  measurements=evidence, input_sha256=inputs,
                  input_sha256_after={path: digest(path) for path in inputs})
    require(result['input_sha256'] == result['input_sha256_after'], 'Decision inputs changed')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(state='complete', output=str(output), measurements=evidence)))


if __name__ == '__main__':
    main()
