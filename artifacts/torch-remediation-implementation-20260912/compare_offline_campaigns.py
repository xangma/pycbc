#!/usr/bin/env python3
"""Compare actual offline campaign receipts without inferred tolerances.

Run on the compute host, with h5py and numpy installed. See --help. Exit 0
means every gate passed, 1 means a failed integrity/identity/numeric gate, and
2 means differing floating fields have no supplied acceptance threshold.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

import h5py
import numpy as np


LABELS = ('native_cuda', 'reference_cuda', 'reference_cpu')
IGNORED = {'--processing-scheme', '--enable-torchwave',
           '--disable-torchwave', '--output'}
EXACT = {'template_hash', 'template_id', 'end_time', 'template_duration',
         'chisq_dof', 'bank_chisq_dof', 'cont_chisq_dof'}
INPUTS = {'--bank-file', '--frame-files'}
PYCBC_CONTRACT = {
    'source': 'Existing PyCBC adopted complex-SNR and trigger/veto criteria',
    'sources': [
        {'commit': 'f9f565030baf631237fa29d7fda06dc3e178d484',
         'path': 'docs/torch_batch_numerics.rst',
         'section': 'Adopted complex-SNR criterion',
         'criteria': 'complex SNR absolute 0.001, zero relative tolerance; '
                     'existing circular phase absolute 0.001 radians'},
        {'commit': 'a6511d24d5801b7e9c291159d98f7a1eb2fda945',
         'path': 'test/test_gpu_search_vetoes.py', 'line': 179,
         'criteria': 'assert_allclose(chisq, ref_chisq_vals, rtol=1e-4, atol=1e-4)'}],
    'fields': {
        'chisq': {'atol': 1e-4, 'rtol': 1e-4},
        'coa_phase': {'atol': 1e-3, 'rtol': 0., 'metric': 'wrapped_phase'},
        'derived_complex_snr': {'atol': 1e-3, 'rtol': 0.}},
    'scope': 'Apply these existing criteria to accepted stored triggers only. '
             'Other nonidentical floating fields remain unassessed. Derived '
             'complex SNR is snr * exp(i * coa_phase), evaluated in float64. '
             'This does not reproduce the full-time-series independent-oracle '
             'qualification required by the source document.'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='backslashreplace')
    if isinstance(value, complex):
        return {'real': scalar(value.real), 'imag': scalar(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def argv_options(argv):
    """Preserve option order and repeated flags, including negative values."""
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise ValueError('argv must be a list of strings')
    tokens = list(argv)
    if tokens and not tokens[0].startswith('--'):
        tokens.pop(0)
    options = []
    for token in tokens:
        if token.startswith('--'):
            key, separator, value = token.partition('=')
            options.append([key, [value] if separator else []])
        elif options:
            options[-1][1].append(token)
        else:
            raise ValueError(f'unassociated argv token: {token}')
    return options


def option_value(options, name):
    found = [values for key, values in options if key == name]
    if len(found) != 1 or len(found[0]) != 1:
        raise ValueError(f'exactly one {name} value is required')
    return found[0][0]


def resolve_input(value, input_cwd):
    path = Path(value)
    if not path.is_absolute():
        if input_cwd is None:
            raise ValueError('relative input path requires --input-cwd: '
                             + value)
        path = input_cwd / path
    return path.resolve(strict=True)


def normalize_argv(argv, input_cwd):
    options = argv_options(argv)
    normalized, inputs, ignored = [], [], []
    for key, values in options:
        if key in IGNORED:
            ignored.append([key, values])
            continue
        if key in INPUTS:
            if not values:
                raise ValueError(f'{key} needs explicit input files')
            hashes = []
            for value in values:
                path = resolve_input(value, input_cwd)
                digest = sha256(path)
                hashes.append(digest)
                inputs.append({'option': key, 'argv_path': value,
                               'resolved_path': str(path), 'sha256': digest,
                               'bytes': path.stat().st_size})
            normalized.append([key, hashes])
        else:
            normalized.append([key, values])
    if not INPUTS.issubset({entry['option'] for entry in inputs}):
        raise ValueError('bank-file and explicit frame-files are required '
                         'to establish input content identity')
    return normalized, inputs, ignored


def read_output(path, options):
    sample_rate = float(option_value(options, '--sample-rate'))
    gps_start = float(option_value(options, '--gps-start-time'))
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError('sample rate must be finite and positive')
    report = {'path': str(path), 'sha256': sha256(path),
              'bytes': path.stat().st_size, 'detectors': {}, 'issues': []}
    arrays = {}
    with h5py.File(path, 'r') as file:
        for ifo, group in file.items():
            if not isinstance(group, h5py.Group):
                continue
            if 'search' not in group and 'template_hash' not in group:
                continue
            fields = {name: dataset[()] for name, dataset in group.items()
                      if isinstance(dataset, h5py.Dataset)}
            record = {'fields': {}, 'duplicates': [], 'search': {}}
            for name, value in fields.items():
                record['fields'][name] = {
                    'dtype': str(value.dtype), 'shape': list(value.shape)}
            if 'search' in group:
                for name, dataset in group['search'].items():
                    if isinstance(dataset, h5py.Dataset):
                        record['search'][name] = [scalar(x) for x in
                                                  np.ravel(dataset[()])]
            if fields and not {'template_hash', 'end_time'} <= fields.keys():
                raise ValueError(f'{path}: {ifo} lacks trigger identity fields')
            count = len(fields['end_time']) if fields else 0
            if any(value.ndim != 1 or len(value) != count
                   for value in fields.values()):
                raise ValueError(f'{path}: {ifo} has inconsistent field shapes')
            identities = []
            if count:
                if fields['template_hash'].dtype.kind not in 'iu':
                    raise ValueError('template_hash must have integer dtype')
                times = fields['end_time']
                if not np.all(np.isfinite(times)):
                    raise ValueError(f'{path}: nonfinite end_time')
                if np.any(np.spacing(np.abs(times)) * sample_rate >= .5):
                    raise ValueError('stored time precision cannot uniquely '
                                     'identify the nearest sample')
                positions = np.rint((times - gps_start) * sample_rate)
                if np.any(np.abs(positions) >= 2**63):
                    raise ValueError('sample index is outside int64 range')
                positions = positions.astype(np.int64)
                on_grid = times == positions / sample_rate + gps_start
                if not np.all(on_grid):
                    report['issues'].append(f'{ifo}: end_time off sample grid')
                for template_hash, position in zip(fields['template_hash'],
                                                   positions):
                    identities.append((ifo, int(template_hash), int(position)))
            counts = Counter(identities)
            record['trigger_count'] = count
            record['identities'] = [list(identity) for identity in identities]
            record['duplicates'] = [{'identity': list(identity), 'count': n}
                                    for identity, n in sorted(counts.items())
                                    if n > 1]
            if record['duplicates']:
                report['issues'].append(f'{ifo}: duplicate trigger identities')
            report['detectors'][ifo] = record
            arrays[ifo] = (fields, identities)
    if not report['detectors']:
        raise ValueError(f'{path}: no detector trigger/search group found')
    return report, arrays


def load_campaign(label, path, input_cwd):
    receipt = json.loads(path.read_text())
    manifest_path = Path(receipt['manifest']).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    issues = []
    if receipt.get('status') != 'completed' or receipt.get('error'):
        issues.append('campaign receipt does not record successful completion')
    if len(receipt['shards']) != 2 or len(manifest['shards']) != 2:
        issues.append('exactly two completed and manifested shards required')
    report = {'receipt': str(path), 'receipt_sha256': sha256(path),
              'manifest': str(manifest_path),
              'manifest_sha256': sha256(manifest_path), 'issues': issues,
              'status': receipt.get('status'), 'shards': [],
              'fresh_workers': receipt.get('fresh_workers'),
              'cache_bytes': receipt.get('cache_bytes'),
              'total_campaign_s': receipt.get('total_campaign_s')}
    arrays = []
    for index, shard in enumerate(receipt['shards']):
        normalized, inputs, ignored = normalize_argv(shard['argv'], input_cwd)
        if index >= len(manifest['shards']):
            raise ValueError('receipt contains a shard absent from manifest')
        manifest_argv = manifest['shards'][index]['argv']
        manifested, _, manifest_ignored = normalize_argv(manifest_argv, input_cwd)
        if normalized != manifested:
            issues.append(f'shard {index}: receipt differs from manifest')
        receipt_config = [x for x in ignored if x[0] != '--output']
        manifest_config = [x for x in manifest_ignored if x[0] != '--output']
        if receipt_config != manifest_config:
            issues.append(f'shard {index}: scheme/provider differs from manifest')
        options = argv_options(shard['argv'])
        device = option_value(options, '--processing-scheme')
        flags = [key for key, _ in options]
        expected_device = 'torch:cpu' if label == 'reference_cpu' else 'torch:cuda'
        expected_flag = ('--enable-torchwave' if label == 'native_cuda'
                         else '--disable-torchwave')
        conflicting = ('--disable-torchwave' if label == 'native_cuda'
                       else '--enable-torchwave')
        if (not (device == expected_device or device.startswith(expected_device + ':'))
                or expected_flag not in flags or conflicting in flags):
            issues.append(f'shard {index}: scheme/provider contradicts label')
        output = Path(shard['output_file']).resolve(strict=True)
        if Path(option_value(options, '--output')).resolve() != output:
            issues.append(f'shard {index}: argv output differs from receipt')
        output_record, data = read_output(output, options)
        issues.extend(f'shard {index}: {issue}' for issue in output_record['issues'])
        if not any(d['trigger_count'] for d in output_record['detectors'].values()):
            issues.append(f'shard {index}: no accepted triggers; scientific '
                          'trigger parity is unexercised')
        report['shards'].append({
            'index': index, 'argv': shard['argv'], 'manifest_argv': manifest_argv,
            'normalized_argv': normalized, 'ignored_deliberate_options': ignored,
            'inputs': inputs, 'output': output_record,
            'pid': shard.get('pid'), 'elapsed_s': shard.get('elapsed_s'),
            'cache_stats': shard.get('cache_stats'),
            'cache_delta': shard.get('cache_delta'),
            'cache_bytes_used': shard.get('cache_bytes_used')})
        arrays.append(data)
    pids = [shard.get('pid') for shard in receipt['shards']]
    report['persistent_worker_observed'] = bool(pids) and len(set(pids)) == 1
    report['second_shard_batch_hits'] = (receipt['shards'][1].get(
        'cache_delta', {}).get('batch_hits') if len(pids) > 1 else None)
    report['second_shard_template_hits'] = (receipt['shards'][1].get(
        'cache_delta', {}).get('template_hits') if len(pids) > 1 else None)
    if label == 'native_cuda':
        native_reuse = (report['persistent_worker_observed'] and
                        (report['second_shard_batch_hits'] or 0) > 0)
        report['native_batch_cache_reuse_gate'] = 'pass' if native_reuse else 'fail'
        if not native_reuse:
            issues.append('native second shard lacks same-worker batch cache-hit evidence')
    return report, arrays


def field_errors(actual, reference, name, threshold):
    result = {'count': int(actual.size), 'actual_dtype': str(actual.dtype),
              'reference_dtype': str(reference.dtype),
              'dtype_equal': actual.dtype == reference.dtype,
              'exact_equal': bool(np.array_equal(actual, reference)),
              'gate': None}
    numeric = actual.dtype.kind in 'biufc' and reference.dtype.kind in 'biufc'
    exact_required = (name in EXACT or name.endswith('_dof') or
                      actual.dtype.kind in 'biuSUO' or not numeric)
    if not numeric:
        result['gate'] = ('pass' if result['exact_equal'] and
                          result['dtype_equal'] else 'fail')
        return result
    a = actual.astype(np.complex128 if actual.dtype.kind == 'c' else np.float64)
    b = reference.astype(np.complex128 if reference.dtype.kind == 'c' else np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    result['actual_nonfinite'] = int(np.count_nonzero(~np.isfinite(a)))
    result['reference_nonfinite'] = int(np.count_nonzero(~np.isfinite(b)))
    result['unequal_count'] = int(np.count_nonzero(actual != reference))
    delta = np.abs(a[finite] - b[finite])
    denominator = np.abs(b[finite])
    nonzero = denominator != 0
    result['max_abs'] = float(delta.max()) if delta.size else None
    result['mean_abs'] = float(delta.mean()) if delta.size else None
    result['rms_abs'] = float(np.sqrt(np.mean(delta**2))) if delta.size else None
    result['max_rel_nonzero_reference'] = (
        float(np.max(delta[nonzero] / denominator[nonzero]))
        if np.any(nonzero) else None)
    result['zero_reference_nonzero_actual'] = int(np.count_nonzero(
        (denominator == 0) & (delta != 0)))
    norm = np.linalg.norm(b[finite])
    result['relative_l2'] = float(np.linalg.norm(delta) / norm) if norm else None
    if delta.size:
        worst = np.flatnonzero(finite)[int(delta.argmax())]
        result['worst_aligned_index'] = int(worst)
        result['worst_actual'] = scalar(actual[worst])
        result['worst_reference'] = scalar(reference[worst])
    if name == 'coa_phase':
        wrapped = np.abs(np.angle(np.exp(1j * (a[finite] - b[finite]))))
        result['max_abs_wrapped_phase'] = float(wrapped.max()) if wrapped.size else None
    if not np.all(finite):
        result['gate'] = 'fail'
    elif exact_required:
        result['gate'] = 'pass' if result['exact_equal'] else 'fail'
        result['gate_rule'] = 'exact scientific metadata/discrete value'
    elif threshold is None:
        result['gate'] = 'pass' if result['exact_equal'] else 'unassessed'
        result['gate_rule'] = 'exact equality; no tolerance supplied'
    else:
        metric = threshold.get('metric', 'absolute_difference')
        differences = np.abs(a - b)
        if metric == 'wrapped_phase':
            if name != 'coa_phase':
                raise ValueError('wrapped_phase gate applies only to coa_phase')
            differences = np.abs(np.angle(np.exp(1j * (a - b))))
        elif metric != 'absolute_difference':
            raise ValueError(f'unsupported error metric: {metric}')
        result['gate'] = ('pass' if np.all(differences <= threshold['atol'] +
                          threshold['rtol'] * np.abs(b)) else 'fail')
        result['gate_rule'] = threshold
    if not result['dtype_equal']:
        result['gate'] = 'fail'
        result['dtype_issue'] = 'trigger field dtype differs'
    for key, value in result.items():
        result[key] = scalar(value)
    return result


def compare_shards(actual, reference, a_data, b_data, thresholds):
    result = {'argv_and_input_hashes_equal': actual['normalized_argv'] ==
              reference['normalized_argv'], 'issues': [], 'detectors': {}}
    if not result['argv_and_input_hashes_equal']:
        result['issues'].append('non-deliberate argv or input content differs')
    if set(a_data) != set(b_data):
        result['issues'].append('detector group sets differ')
    for ifo in sorted(set(a_data) | set(b_data)):
        a_fields, a_ids = a_data.get(ifo, ({}, []))
        b_fields, b_ids = b_data.get(ifo, ({}, []))
        ac, bc = Counter(a_ids), Counter(b_ids)
        missing, extra = bc - ac, ac - bc
        common = sorted(key for key in ac.keys() & bc.keys()
                        if ac[key] == bc[key] == 1)
        ai, bi = ({key: i for i, key in enumerate(ids)} for ids in (a_ids, b_ids))
        a_order, b_order = [ai[key] for key in common], [bi[key] for key in common]
        record = {'actual_count': len(a_ids), 'reference_count': len(b_ids),
                  'missing': [{'identity': list(k), 'count': n}
                              for k, n in sorted(missing.items())],
                  'extra': [{'identity': list(k), 'count': n}
                            for k, n in sorted(extra.items())],
                  'duplicate_actual': any(n > 1 for n in ac.values()),
                  'duplicate_reference': any(n > 1 for n in bc.values()),
                  'aligned_unique_identities': [list(k) for k in common],
                  'field_names_equal': set(a_fields) == set(b_fields),
                  'actual_field_names': sorted(a_fields),
                  'reference_field_names': sorted(b_fields), 'fields': {}}
        record['identity_gate'] = (
            'pass' if not missing and not extra and
            not record['duplicate_actual'] and
            not record['duplicate_reference'] else 'fail')
        for field in sorted(a_fields.keys() & b_fields.keys()):
            threshold = thresholds.get(f'{ifo}/{field}', thresholds.get(field))
            record['fields'][field] = field_errors(
                a_fields[field][a_order], b_fields[field][b_order], field, threshold)
            worst = record['fields'][field].get('worst_aligned_index')
            if worst is not None:
                record['fields'][field]['worst_identity'] = list(common[worst])
        required = {'snr', 'coa_phase'}
        if required <= a_fields.keys() and required <= b_fields.keys():
            a_complex = (a_fields['snr'][a_order].astype(np.float64) *
                         np.exp(1j * a_fields['coa_phase'][a_order].astype(np.float64)))
            b_complex = (b_fields['snr'][b_order].astype(np.float64) *
                         np.exp(1j * b_fields['coa_phase'][b_order].astype(np.float64)))
            field = 'derived_complex_snr'
            threshold = thresholds.get(f'{ifo}/{field}', thresholds.get(field))
            derived = field_errors(a_complex, b_complex, field, threshold)
            derived['derived'] = True
            derived['definition'] = 'snr * exp(i * coa_phase), evaluated in float64'
            derived['source_fields'] = ['snr', 'coa_phase']
            worst = derived.get('worst_aligned_index')
            if worst is not None:
                derived['worst_identity'] = list(common[worst])
            record['fields'][field] = derived
        elif 'derived_complex_snr' in thresholds:
            result['issues'].append(f'{ifo}: complex-SNR gate lacks snr/coa_phase')
        for field in ('start_time', 'end_time'):
            ar = actual['output']['detectors'].get(ifo, {}).get('search', {})
            br = reference['output']['detectors'].get(ifo, {}).get('search', {})
            if field not in ar or field not in br or ar[field] != br[field]:
                result['issues'].append(f'{ifo}/search/{field} differs or missing')
        result['detectors'][ifo] = record
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in LABELS:
        parser.add_argument('--' + label.replace('_', '-'), required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='JSON report path; sibling .md summary is also written')
    parser.add_argument('--input-cwd', type=Path,
                        help='original campaign cwd, required for relative inputs')
    gates = parser.add_mutually_exclusive_group()
    gates.add_argument('--tolerances', type=Path, help='JSON with source and fields: '
                       '{"source":"approved criterion", "fields":'
                       '{"snr":{"atol":0,"rtol":0}}}; no implicit tolerances')
    gates.add_argument('--pycbc-contract', action='store_true', help='explicitly '
                       'apply existing adopted complex-SNR absolute 0.001, '
                       'circular phase absolute 0.001 radians, and power chi-square '
                       'rtol=atol=0.0001 to stored accepted triggers')
    args = parser.parse_args()
    report = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'comparator_sha256': sha256(__file__), 'argv': sys.argv,
              'campaigns': {}, 'comparisons': [], 'fatal_errors': [],
              'identity_definition': ['detector', 'template_hash',
                                      'integer sample index from end_time'],
              'exact_fields': sorted(EXACT),
              'limitations': ['Input files are hashed at comparison time; receipts '
                              'do not attest their content at execution time.',
                              'Receipt flags describe requested provider; batch '
                              'cache hits support native batch-path execution, '
                              'but receipts do not enumerate per-row providers.',
                              'Two repeated fixture shards do not establish '
                              'general scientific or performance qualification.']}
    thresholds, data = {}, {}
    try:
        if args.tolerances or args.pycbc_contract:
            policy = (PYCBC_CONTRACT if args.pycbc_contract else
                      json.loads(args.tolerances.read_text()))
            if not policy.get('source') or not isinstance(policy.get('fields'), dict):
                raise ValueError('tolerances need a source and fields object')
            thresholds = policy['fields']
            for field, value in thresholds.items():
                if field.rsplit('/', 1)[-1] in EXACT:
                    raise ValueError('exact field cannot receive tolerance: ' + field)
                for key in ('atol', 'rtol'):
                    if (key not in value or not isinstance(value[key], (int, float))
                            or not np.isfinite(value[key]) or value[key] < 0):
                        raise ValueError('each field needs finite nonnegative atol/rtol')
            report['tolerance_policy'] = policy
            report['tolerance_mode'] = ('pycbc_contract' if args.pycbc_contract
                                        else 'explicit_file')
            if args.tolerances:
                report['tolerance_file_sha256'] = sha256(args.tolerances)
        else:
            report['tolerance_policy'] = None
        for label in LABELS:
            path = getattr(args, label).resolve(strict=True)
            report['campaigns'][label], data[label] = load_campaign(
                label, path, args.input_cwd)
        pairs = list(itertools.combinations(LABELS, 2))
        for a, b in pairs:
            for index in range(min(len(data[a]), len(data[b]))):
                comparison = compare_shards(
                    report['campaigns'][a]['shards'][index],
                    report['campaigns'][b]['shards'][index], data[a][index],
                    data[b][index], thresholds)
                comparison.update(actual=a, reference=b, actual_shard=index,
                                  reference_shard=index)
                report['comparisons'].append(comparison)
        for label in LABELS:
            if len(data[label]) == 2:
                shards = report['campaigns'][label]['shards']
                comparison = compare_shards(shards[1], shards[0], data[label][1],
                                            data[label][0], {})
                comparison.update(actual=label, reference=label, actual_shard=1,
                                  reference_shard=0, purpose='repeat/cache reuse')
                report['comparisons'].append(comparison)
    except Exception as exc:
        report['fatal_errors'].append(f'{type(exc).__name__}: {exc}')
    failed = bool(report['fatal_errors']) or any(
        campaign['issues'] for campaign in report['campaigns'].values())
    unassessed = []
    for comparison in report['comparisons']:
        failed |= bool(comparison['issues'])
        for ifo, detector in comparison['detectors'].items():
            failed |= (detector['identity_gate'] != 'pass' or
                       not detector['field_names_equal'])
            for field, values in detector['fields'].items():
                failed |= values['gate'] == 'fail'
                if values['gate'] == 'unassessed':
                    unassessed.append(
                        f"{comparison['actual']}:{comparison['actual_shard']}"
                        f" vs {comparison['reference']}:{comparison['reference_shard']} "
                        f'{ifo}/{field}')
    report['unassessed_numeric_fields'] = unassessed
    report['status'] = 'fail' if failed else 'unassessed' if unassessed else 'pass'
    lines = [f"Offline campaign comparison: **{report['status'].upper()}**", '',
             'Exact trigger identity, schema, scientific metadata, and input '
             'content checks are separate from floating-point error gates.', '']
    if args.pycbc_contract:
        lines += ['Explicit PyCBC contract mode: derived complex-SNR absolute '
                  'error ≤ 0.001; existing circular phase absolute error '
                  '≤ 0.001 radians; power chi-square atol=rtol=0.0001. '
                  'Raw phase differences remain diagnostic. Source revisions '
                  'are recorded in the JSON.', '', PYCBC_CONTRACT['scope'], '']
    for label, campaign in report['campaigns'].items():
        counts = [sum(d['trigger_count'] for d in shard['output']['detectors'].values())
                  for shard in campaign['shards']]
        lines.append(f"- {label}: triggers per shard {counts}; persistent worker "
                     f"{campaign['persistent_worker_observed']}; second-shard "
                     f"batch hits {campaign['second_shard_batch_hits']}.")
        lines.extend('- ' + issue for issue in campaign['issues'])
    lines += ['', f'Unassessed numeric field comparisons: {len(unassessed)}.']
    lines += ['', '| Comparison | Identity | Field failures | Unassessed |',
              '| --- | --- | ---: | ---: |']
    metrics = {}
    for comparison in report['comparisons']:
        detectors = comparison['detectors'].values()
        identity = ('PASS' if all(d['identity_gate'] == 'pass' for d in detectors)
                    else 'FAIL')
        gates = [v['gate'] for d in detectors for v in d['fields'].values()]
        description = (f"{comparison['actual']}:{comparison['actual_shard']} vs "
                       f"{comparison['reference']}:{comparison['reference_shard']}")
        lines.append(f"| {description} | {identity} | {gates.count('fail')} | "
                     f"{gates.count('unassessed')} |")
        lines.extend('- ' + issue for issue in comparison['issues'])
        for ifo, detector in comparison['detectors'].items():
            for field, values in detector['fields'].items():
                metrics.setdefault(f'{ifo}/{field}', []).append(values)
    lines += ['', 'Maximum raw error across all compared shards:', '',
              '| Field | Maximum absolute error | Maximum relative error* |',
              '| --- | ---: | ---: |']
    for field, values in sorted(metrics.items()):
        maxima = []
        for key in ('max_abs', 'max_rel_nonzero_reference'):
            available = [float(v[key]) for v in values if v.get(key) is not None]
            maxima.append(f'{max(available):.9g}' if available else 'n/a')
        lines.append(f'| {field} | {maxima[0]} | {maxima[1]} |')
    lines += ['', '*Relative error excludes zero-valued reference elements; '
              'the JSON reports their mismatches separately. Phase errors '
              'above use direct subtraction; wrapped phase diagnostics are '
              'also recorded.']
    lines.extend('- ' + error for error in report['fatal_errors'])
    lines += ['', 'Numeric errors, missing/extra identities, duplicates, per-field '
              'gate rules, cache counters, argv, and file SHA256 values are in '
              'the JSON report. No acceptance tolerance is inferred.', '']
    lines.extend('- ' + limitation for limitation in report['limitations'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 1 if failed else 2 if unassessed else 0


if __name__ == '__main__':
    raise SystemExit(main())
