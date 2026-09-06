#!/usr/bin/env python3
"""Summarize completed inspiral runs, binding timing rows to qualified work."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


ROW_FIELDS = (
    'case', 'scheme', 'mode', 'segment_length', 'start_pad', 'end_pad',
    'templates', 'segments_per_template', 'fft_samples', 'valid_data_seconds',
    'total_template_seconds', 'qualification', 'qualification_signature',
    'allocated_cpu_cores', 'wall_seconds', 'user_cpu_seconds',
    'system_cpu_seconds', 'peak_rss_mib', 'triggers',
    'templates_per_core_at_real_time', 'internal_runtime_seconds',
    'internal_setup_seconds', 'internal_postsetup_seconds', 'receipt',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_cli(argv):
    """Retain every executable option except the per-run output destination."""
    if not isinstance(argv, list) or not argv or not all(
            isinstance(arg, str) for arg in argv):
        raise ValueError('Receipt lacks executable_cli')
    options, key = {}, None
    for arg in argv[1:]:
        if arg.startswith('--'):
            key, separator, value = arg.partition('=')
            if key in options:
                raise ValueError(f'Repeated executable option: {key}')
            options[key] = [value] if separator else []
        elif key is None:
            raise ValueError('Unexpected positional executable argument')
        else:
            options[key].append(arg)
    if len(options.pop('--output', [])) != 1:
        raise ValueError('Expected one executable output destination')
    return argv[0], tuple(sorted((k, tuple(v)) for k, v in options.items()))


def signature(record):
    # The qualification instrumenter is absent from ordinary timing runs.
    # Retain the runner, config and every data/executable input, regardless of
    # filename, so a new bank or wrapper revision cannot inherit qualification.
    instrumenter = str(Path(record['cwd']) / 'qualify-inspiral.py')
    hashes = dict(record['input_sha256'])
    if record['mode'] == 'qualify' and instrumenter in record['command']:
        hashes.pop(instrumenter, None)
    return (record['source_info']['commit'], record['hostname'], record['scheme'],
            record['segment_length'], record['start_pad'], record['end_pad'],
            tuple(sorted(record['environment'].items())),
            normalized_cli(record['executable_cli']), tuple(sorted(hashes.items())))


def signature_id(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def complete(record):
    return (record['state'] == 'complete' and record['returncode'] == 0
            and record['input_sha256'] == record['input_sha256_after']
            and not record['source_status_after']
            and not record['source_info']['status'])


def summarize(root):
    import h5py

    inputs, qualified, receipts, excluded = {}, {}, [], []
    for path in sorted((root / 'runs').glob('*/receipt.json')):
        r = json.loads(path.read_text())
        inputs[str(path.relative_to(root))] = sha(path)
        receipts.append((path, r))
        if r['mode'] != 'qualify' or not complete(r):
            continue
        qpath = path.parent / 'qualification.json'
        inputs[str(qpath.relative_to(root))] = sha(qpath)
        q = json.loads(qpath.read_text())
        assert q['status'] == 'success' and q['executable_exit_code'] == 0
        assert q['checks'] and all(q['checks'].values())
        o = q['observations']
        assert len(o['banks']) == len(o['segment_geometry']) == 1
        g, bank = o['segment_geometry'][0], o['banks'][0]
        n, segments = len(bank['templates']), len(g['segments'])
        assert sum(e['execute_successes'] for e in o['fft_engines']) == n * segments
        assert g['gap_samples'] == g['overlap_samples'] == 0
        work = dict(templates=n, segments_per_template=segments,
                    fft_samples=g['fft_samples'],
                    valid_data_seconds=g['unique_analyzed_seconds'],
                    total_template_seconds=n * g['unique_analyzed_seconds'],
                    qualification=str(qpath.relative_to(root)))
        old = qualified.get(signature(r))
        if old:
            assert {k: v for k, v in old.items() if k != 'qualification'} == {
                k: v for k, v in work.items() if k != 'qualification'}
        qualified[signature(r)] = work
    rows = []
    for path, r in receipts:
        if not complete(r) or signature(r) not in qualified:
            excluded.append(dict(case=r['case'], state=r['state'], mode=r['mode'],
                                 reason='Incomplete, failed, or no matching successful qualification'))
            continue
        trigger = path.parent / 'triggers.hdf'
        assert sha(trigger) == r['trigger_sha256']
        inputs[str(trigger.relative_to(root))] = r['trigger_sha256']
        work = qualified[signature(r)]
        with h5py.File(trigger) as f:
            s = f['H1/search']
            runtime = float(s['run_time'][0])
            setup_fraction = float(s['setup_time_fraction'][0])
            assert sum(s['end_time'][:] - s['start_time'][:]) == work['valid_data_seconds']
            detector = f['H1']
            if 'end_time' in detector:
                triggers = len(detector['end_time'])
            else:
                # EventManager omits all event columns when no events survive.
                assert not any(isinstance(v, h5py.Dataset) for v in detector.values())
                triggers = 0
        wall = r['elapsed_wall_seconds']
        row = dict(case=r['case'], scheme=r['scheme'], mode=r['mode'],
                   segment_length=r['segment_length'], start_pad=r['start_pad'],
                   end_pad=r['end_pad'], **work,
                   qualification_signature=signature_id(signature(r)),
                   allocated_cpu_cores=1,
                   wall_seconds=wall,
                   user_cpu_seconds=r['child_user_cpu_seconds'],
                   system_cpu_seconds=r['child_system_cpu_seconds'],
                   peak_rss_mib=r['peak_child_rss_kib'] / 1024,
                   triggers=triggers,
                   templates_per_core_at_real_time=work['total_template_seconds'] / wall,
                   internal_runtime_seconds=runtime,
                   internal_setup_seconds=runtime * setup_fraction,
                   internal_postsetup_seconds=runtime * (1 - setup_fraction),
                   receipt=str(path.relative_to(root)))
        rows.append(row)
    groups = {}
    for row in rows:
        if row['mode'] != 'timing' or row['case'].startswith('smoke-'):
            continue
        key = (row['scheme'], row['segment_length'], row['start_pad'], row['end_pad'],
               row['qualification_signature'])
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, items in sorted(groups.items()):
        item = dict(scheme=key[0], segment_length=key[1], start_pad=key[2], end_pad=key[3],
                    qualification_signature=key[4],
                    n=len(items), cases=[r['case'] for r in items])
        for metric in ('wall_seconds', 'templates_per_core_at_real_time',
                       'internal_runtime_seconds', 'internal_setup_seconds',
                       'internal_postsetup_seconds', 'peak_rss_mib'):
            values = [r[metric] for r in items]
            item[metric] = dict(median=statistics.median(values), min=min(values), max=max(values))
        item.update({k: items[0][k] for k in ('templates', 'segments_per_template',
                     'valid_data_seconds', 'total_template_seconds', 'fft_samples')})
        summaries.append(item)
    return dict(schema_version=1, input_sha256=inputs, rows=rows, groups=summaries,
                excluded=excluded, limitations=[
                    'Timing eligibility binds completed run receipts to a successful matching qualification.',
                    'Scientific compression, boundary and cross-backend trigger validation are separate gates.',
                    'Only mode=timing contributes to groups. Profiles and qualification are separate rows.',
                    'Internal runtime excludes interpreter/import startup and final trigger-file writing.',
                    'Internal postsetup interval is derived from executable metadata; it includes analysis housekeeping.',
                    'Rates count qualified templates times unique valid detector seconds, divided by full wall time and one host core.',
                    'Settings are compared within the declared grid; this is not a global optimum claim.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    csv_path = args.output.with_suffix('.csv')
    assert csv_path != args.output, 'JSON and CSV output paths must differ'
    assert not args.output.exists(), 'Use a new output path to retain earlier summaries'
    assert not csv_path.exists(), 'Use a new CSV output path to retain earlier summaries'
    result = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    with csv_path.open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(result['rows'])
    for r in result['groups']:
        print(r['scheme'], r['segment_length'], r['start_pad'], r['end_pad'],
              'n', r['n'], 'wall', round(r['wall_seconds']['median'], 3),
              'capacity', round(r['templates_per_core_at_real_time']['median'], 1))


if __name__ == '__main__':
    main()
