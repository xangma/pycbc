#!/usr/bin/env python3
"""Extract checked, exclusive profile evidence without importing PyCBC.

Read only trusted pstats files: Python's marshal format is not a safe format
for untrusted input. Cumulative seconds and caller edges are diagnostic fields,
never additive categories or estimates of GPU device time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pstats


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symbol(key):
    filename, line, function = key
    return dict(file=filename, line=line, function=function)


def summarize(root):
    source = root / 'summary.json'
    campaign = json.loads(source.read_text())
    if campaign['status'] != 'complete':
        raise ValueError('Campaign is not complete')
    result = dict(summary_sha256=digest(source), profiles=[], native_profiles=[],
                  wall_accounting=[], intervals={}, parity=campaign['parity'])
    for entry in campaign['profiles']:
        path = root / entry['source_path']
        if digest(path) != entry['sha256']:
            raise ValueError(f'Profile checksum mismatch: {path}')
        stats = pstats.Stats(str(path)).stats
        total = sum(value[2] for value in stats.values())
        if abs(total - entry['total_exclusive_seconds']) > 1e-8:
            raise ValueError(f'Exclusive denominator mismatch: {path}')
        rows = []
        for key, value in sorted(stats.items(),
                                 key=lambda item: item[1][2], reverse=True):
            if value[2] < .01:
                continue
            callers = []
            for caller, edge in sorted(value[4].items(),
                                       key=lambda item: item[1][2], reverse=True):
                callers.append(dict(**symbol(caller), calls=edge[1],
                                    exclusive_seconds=edge[2],
                                    cumulative_seconds=edge[3]))
            rows.append(dict(**symbol(key), calls=value[1],
                             exclusive_seconds=value[2],
                             cumulative_seconds=value[3],
                             exclusive_percent=100 * value[2] / total,
                             callers=callers))
        result['profiles'].append(dict(case=entry['case'], scope=entry['scope'],
                                       source_path=entry['source_path'],
                                       sha256=entry['sha256'],
                                       total_exclusive_seconds=total, rows=rows))
    for entry in campaign['native_profiles']:
        if digest(root / entry['source_path']) != entry['sha256']:
            raise ValueError('Native report checksum mismatch')
        result['native_profiles'].append(entry)
    for entry in campaign['matched']:
        run = sorted(entry['runs'], key=lambda r: r['wall_seconds'])[1]
        result['wall_accounting'].append(dict(
            backend=entry['backend'], case=run['case'],
            **{key: run[key] for key in (
                'wall_seconds', 'internal_setup_seconds',
                'internal_postsetup_seconds', 'outside_internal_timer_seconds')}))
    for backend, interval in campaign['intervals'].items():
        result['intervals'][backend] = {
            key: interval[key] for key in (
                'filtering_window_seconds', 'completed_templates',
                'segments_per_template', 'clock')}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.campaign.resolve())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
