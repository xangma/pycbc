#!/usr/bin/env python3
"""Accept complete v6 scientific evidence and strict selected-trigger parity."""
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source-v6'
OUTPUT = ROOT / 'scientific-validation-v6.json'
PARITY = ROOT / 'trigger-parity-qualifications-v6.json'
WAVE_BUDGETS = dict(fixed_time_phase_optimized_mismatch=0.001,
                    relative_snr_norm_error=0.01, relative_weighted_amplitude_error=0.01)
BOUNDARY_BUDGETS = dict(edge_energy_fraction=1e-6, maximum_relative_complex_error=0.001,
                        peak_location_error_samples=1, relative_peak_snr_error=0.001)
TRIGGER_BUDGETS = dict(rtol=1e-4, atol=1e-5, sigmasq_rtol=1e-5, phase_atol=1e-4, max_examples=12)
inputs = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind(path, expected=None):
    path = Path(path)
    require(path.is_absolute(), f'Expected absolute path: {path}')
    observed = digest(path)
    require(expected is None or observed == expected, f'Changed input: {path}')
    require(str(path) not in inputs or inputs[str(path)] == observed, f'Input changed while reading: {path}')
    inputs[str(path)] = observed
    return path


def read(name):
    path = bind(ROOT / name)
    value = json.loads(path.read_text())
    bind(path, inputs[str(path)])
    return value


def manifest(value):
    hashes = value['input_sha256']
    require(hashes and hashes == value['input_sha256_after'], 'Missing or changed input manifest')
    for path, expected in hashes.items():
        bind(path, expected)


def source_info():
    return {name: subprocess.check_output(['git', '-C', str(SOURCE), *args], text=True).strip()
            for name, args in [('commit', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain'])]}


def budget(values, limits):
    for name, maximum in limits.items():
        actual = values[name]
        require(isinstance(actual, (int, float)) and math.isfinite(actual) and 0 <= actual <= maximum,
                f'Exceeded {name} budget: {actual} > {maximum}')


def main():
    require(not OUTPUT.exists() and not PARITY.exists(), 'Acceptance outputs already exist')
    bind(Path(__file__).resolve())
    source = read('source-v6.json')
    before = source_info()
    require(before == dict(commit=source['commit'], status=''), 'Wrong final source')
    manifest(source)
    for name, expected in {**source['changed_files_sha256'], **source['native_modules_sha256']}.items():
        bind(SOURCE / name, expected)
    units = read('unit-tests-v6.json')
    require(units['state'] == 'complete' and units['passed'] is True and units['returncode'] == 0
            and units['source_info'] == units['source_after'] == before, 'Final unit tests failed')
    manifest(units)
    bind(ROOT / 'unit-tests-v6.log', units['log_sha256'])
    selected = [f'qual-selected6-{backend}-l512' for backend in ('cpu', 'torch-cpu', 'torch-cuda')]
    stage = read('campaign-v6-qualifications.status.json')
    require(stage['state'] == 'complete' and stage['returncode'] == 0
            and stage['source_info'] == stage['source_after'] == before
            and [row['case'] for row in stage['completed']] == selected, 'Selected qualifications incomplete')
    manifest(stage)
    for name, expected in stage['output_sha256'].items():
        bind(name, expected)
    runner = read('scientific-checks-v6.status.json')
    require(runner['state'] == 'complete' and runner['passed'] is True and runner['returncode'] == 0
            and runner['source_info'] == runner['source_after'] == before, 'Scientific runner failed')
    manifest(runner)
    require(runner['output_sha256'], 'Scientific runner output manifest is missing')
    for name, expected in runner['output_sha256'].items():
        bind(name, expected)
    normal = ['qual-science6-cpu-l256', selected[0], 'qual-science6-cpu-l1024']
    for case in dict.fromkeys(normal + selected):
        run = read(f'runs/{case}/receipt.json')
        q = read(f'runs/{case}/qualification.json')
        require(run['source_info'] == dict(**before, tracked_diff='') and run['source_status_after'] == ''
                and run['state'] == 'complete' and run['returncode'] == 0
                and q['status'] == 'success' and q['executable_exit_code'] == 0
                and q['source_root'] == str(SOURCE) and q['checks']
                and all(value is True for value in q['checks'].values()), 'Failed scientific qualification')
        manifest(run)
        bind(ROOT / 'runs' / case / 'triggers.hdf', run['trigger_sha256'])
    metadata = read('inputs/bank-metadata.json')
    identities = {str(row['template_hash']) for row in metadata['templates']}
    require(len(identities) == 96, 'Wrong bank identity count')
    wave = read('waveform-validation-v6.json')
    boundary = read('boundary-injections-v6.json')
    for value, count_key, count in ((wave, 'template_psd_pairs', 288), (boundary, 'cases', 36)):
        require(value['state'] == 'complete' and value['passed'] is True
                and value['completed_' + count_key] == value['expected_' + count_key] == count
                and value['source_before'] == value['source_after']
                and value['source_before']['commit'] == before['commit']
                and value['source_before']['status'] == '', 'Incomplete scientific validation')
        manifest(value)
    require(wave['tolerances'] == WAVE_BUDGETS and boundary['tolerances'] == BOUNDARY_BUDGETS,
            'Scientific budgets changed')
    require(len(wave['cases']) == 3 and {row['qualification'] for row in wave['cases']}
            == {str(ROOT / 'runs' / case / 'qualification.json') for case in normal}, 'Wrong waveform qualifiers')
    pairs = []
    for case in wave['cases']:
        require(case['state'] == 'complete' and case['passed'] is True and len(case['templates']) == 96
                and {str(row['template_hash']) for row in case['templates']} == identities,
                'Waveform bank coverage differs')
        for template in case['templates']:
            require(template['passed'] is True and template['decompression_succeeded_without_generation'] is True
                    and len(template['psd_results']) == 1,
                    'Waveform generation fallback or mismatch')
            for row in template['psd_results']:
                require(row['passed'] is True, 'Waveform pair failed')
                budget(row['metrics'], WAVE_BUDGETS)
                pairs.append(row)
    require(len(pairs) == 288, 'Wrong waveform-pair count')
    require(boundary['target_reference_peak_snr'] == 8 and boundary['long_reference_seconds'] == 4096
            and boundary['qualification'] == str(ROOT / 'runs' / normal[2] / 'qualification.json')
            and len(boundary['templates']) == 2
            and {t['kind'] for t in boundary['templates']} == {'longest', 'shortest'}, 'Wrong boundary references')
    grid = {(length, pad, 16, placement) for length in (256, 512, 1024) for pad in (96, 112)
            for placement in ('first_valid', 'midpoint', 'last_valid')}
    for template in boundary['templates']:
        require(template['passed'] is True and template['reference_edge_budget_passed'] is True
                and str(template['reference']['template_hash']) in identities
                and len(template['cases']) == 18
                and {(row['segment_length_seconds'], row['start_pad_seconds'], row['end_pad_seconds'], row['placement'])
                     for row in template['cases']} == grid, 'Boundary grid coverage differs')
        for name in ('injection_edge_energy_fraction', 'kernel_edge_energy_fraction'):
            budget({name: template['reference']['tails'][name]}, {name: BOUNDARY_BUDGETS['edge_energy_fraction']})
        for row in template['cases']:
            require(row['passed'] is True, 'Boundary case failed')
            budget(row, {key: limit for key, limit in BOUNDARY_BUDGETS.items() if key != 'edge_energy_fraction'})
    bank = read('compressed-bank-v6.json')
    require(bank['state'] == 'complete' and bank['passed'] is True and bank['returncode'] == 0
            and bank['source_info'] == bank['source_after'] == before
            and bank['completed_cases'] == bank['expected_cases'] == len(bank['cases']) == 576
            and bank['native_calls'] == bank['cuda_comparisons'] == 288, 'Incomplete compressed-bank parity')
    manifest(bank)
    require({(row['device'], row['length_seconds'], row['template_hash']) for row in bank['cases']}
            == {(device, length, identity) for device in ('cpu', 'cuda') for length in (256, 512, 1024)
                for identity in identities}, 'Compressed-bank parity grid differs')
    require(all(row['bitwise_equal'] is True and
                (row['device'] != 'cpu' or row['native_used'] is True) for row in bank['cases']),
            'Compressed-bank output mismatch')
    helper = bind(ROOT / 'compare-triggers.py')
    spec = importlib.util.spec_from_file_location('trigger_comparator', helper)
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    require(comparator.DEFAULTS == TRIGGER_BUDGETS, 'Trigger parity budgets changed')
    trigger_data = [comparator.load(ROOT / 'runs' / case / 'triggers.hdf') for case in selected]
    comparisons = [comparator.compare(trigger_data[0], data, TRIGGER_BUDGETS) for data in trigger_data[1:]]
    require(len(comparisons) == 2 and all(row['status'] == 'pass' for row in comparisons),
            'Strict qualification trigger parity failed or needs review')
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    parity = dict(schema_version=1, state='complete', passed=True, returncode=0, finished_utc=now,
                  source_info=before, source_after=source_info(), tolerances=TRIGGER_BUDGETS,
                  comparisons=comparisons, input_sha256=dict(inputs),
                  input_sha256_after={name: digest(name) for name in inputs})
    require(parity['source_after'] == before and parity['input_sha256'] == parity['input_sha256_after'],
            'Inputs changed during scientific acceptance')
    with PARITY.open('x') as stream:
        json.dump(parity, stream, indent=2, allow_nan=False)
        stream.write('\n')
    bind(PARITY)
    evidence_names = dict(waveform_reference='waveform-validation-v6.json',
                          boundary_injections='boundary-injections-v6.json',
                          compressed_bank_backend_parity='compressed-bank-v6.json',
                          qualification_trigger_parity=PARITY.name)
    result = dict(schema_version=1, state='complete', passed=True, returncode=0, finished_utc=now,
                  source_info=before, source_after=source_info(), checks={key: True for key in evidence_names},
                  evidence={key: dict(path=str(ROOT / name), sha256=inputs[str(ROOT / name)])
                            for key, name in evidence_names.items()}, input_sha256=inputs,
                  input_sha256_after={name: digest(name) for name in inputs})
    require(result['source_after'] == before and result['input_sha256'] == result['input_sha256_after'],
            'Inputs changed before final acceptance')
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(state='complete', passed=True, output=str(OUTPUT), source=before,
                         waveform_pairs=288, boundary_cases=36, bank_parity_cases=576, trigger_comparisons=2)))


if __name__ == '__main__':
    main()
