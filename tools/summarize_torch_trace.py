#!/usr/bin/env python3
"""Summarize Kineto traces, preserving overlap and unattributed activity.

Only kernel/memcpy/memset events count as device activity. GPU annotation
ranges do not. Semantic ownership follows External id or CUDA correlation to
host events, then containing same-thread pycbc:: ranges. Missing/conflicting
links remain unattributed. No timing-only CPU-to-GPU matching is attempted.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path


DEVICE_CATEGORIES = {'kernel', 'gpu_memcpy', 'gpu_memset'}
RUNTIME_CATEGORIES = {'cuda_runtime', 'cuda_driver'}


def interval(event):
    return event['ts'], event['ts'] + event['dur']


def union_us(intervals):
    total = 0.0
    end = -math.inf
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def metrics(events, window=None):
    """Count all events, or only positive intersections with a window."""
    spans = [interval(e) for e in events]
    if window is not None:
        spans = [(max(a, window[0]), min(b, window[1]))
                 for a, b in spans
                 if b > a and b > window[0] and a < window[1]]
    return dict(event_count=len(spans),
                summed_seconds=sum(b - a for a, b in spans) / 1e6,
                union_seconds=union_us(spans) / 1e6)


def semantic_paths(host_events, ranges):
    """Assign full nested semantic paths using a sweep per host thread."""
    by_thread = defaultdict(list)
    for e in ranges:
        by_thread[e['pid'], e['tid']].append(e)
    events_by_thread = defaultdict(list)
    for i, e in enumerate(host_events):
        events_by_thread[e['pid'], e['tid']].append((i, e))
    paths = {}
    for thread, events in events_by_thread.items():
        ordered = sorted(by_thread[thread], key=lambda e: e['ts'])
        next_range = 0
        active = {}
        ends = []
        for i, e in sorted(events, key=lambda pair: pair[1]['ts']):
            start, stop = interval(e)
            while next_range < len(
                    ordered) and ordered[next_range]['ts'] <= start:
                r = ordered[next_range]
                active[next_range] = r
                heapq.heappush(ends, (interval(r)[1], next_range))
                next_range += 1
            while ends and ends[0][0] < start:
                _, expired = heapq.heappop(ends)
                del active[expired]
            containing = [r for r in active.values() if interval(r)[1] >= stop]
            containing.sort(key=lambda r: (r['ts'], -r['dur']))
            names = [r['name'][7:] for r in containing
                     if r['name'] != 'pycbc::filter_loop']
            paths[i] = '/'.join(names) if names else 'unattributed'
    return paths


def link_id(event, field):
    value = event.get('args', {}).get(field)
    # Zero and negative sentinel IDs are not evidence of an association.
    try:
        return str(int(value)) if int(value) > 0 else None
    except (TypeError, ValueError):
        return None


def summarize(trace):
    events = [e for e in trace['traceEvents'] if e.get('ph') == 'X']
    for e in events:
        if not (math.isfinite(e['ts']) and math.isfinite(
                e['dur']) and e['dur'] >= 0):
            raise ValueError('Nonfinite timestamp or invalid event duration')
    ranges = [e for e in events if e.get('cat') == 'user_annotation'
              and e.get('name', '').startswith('pycbc::')]
    loops = [e for e in ranges if e['name'] == 'pycbc::filter_loop']
    if len(loops) != 1 or loops[0]['dur'] <= 0:
        raise ValueError(
            'Exactly one positive-duration pycbc::filter_loop is required')
    loop = loops[0]
    window = interval(loop)
    # Other host processes cannot share this single-process ownership map.
    host = [e for e in events
            if e.get('cat') in RUNTIME_CATEGORIES | {'cpu_op'}
            and e['pid'] == loop['pid']]
    paths = semantic_paths(host, ranges)
    external = defaultdict(set)
    correlation = defaultdict(set)
    for i, e in enumerate(host):
        if e.get('cat') == 'cpu_op':
            key = link_id(e, 'External id')
            if key is not None:
                external[key].add(paths[i])
        else:
            key = link_id(e, 'correlation')
            if key is not None:
                correlation[key].add(paths[i])

    by_device = defaultdict(list)
    by_semantic = defaultdict(list)
    by_name = defaultdict(list)
    linkage = Counter()
    outside = []
    device_events = [e for e in events if e.get('cat') in DEVICE_CATEGORIES]
    for e in device_events:
        direct = external.get(link_id(e, 'External id'), set())
        runtime = correlation.get(link_id(e, 'correlation'), set())
        candidates = direct | runtime
        if len(candidates) == 1 and 'unattributed' not in candidates:
            path = next(iter(candidates))
            reason = 'external_and_runtime' if direct and runtime else (
                'external' if direct else 'runtime')
        else:
            path = 'unattributed'
            reason = 'conflicting' if len(
                candidates) > 1 else 'missing_semantic_or_link'
        linkage[reason] += 1
        device = str(e.get('args', {}).get('device', e['pid']))
        category = e['cat']
        by_device[device].append(e)
        by_semantic[device, category, path].append(e)
        by_name[device, category, e['name']].append(e)
        if e['ts'] < window[0] or interval(e)[1] > window[1]:
            outside.append(e)

    result = dict(
        schema_version=1,
        measurement='instrumented; not eligible for throughput',
        loop_seconds=loop['dur'] / 1e6,
        units='Chrome timestamps/durations are microseconds; output seconds',
        categories=dict(Counter(e.get('cat', '') for e in events)),
        linkage_counts=dict(linkage),
        device_events_outside_or_crossing_loop=metrics(outside),
        devices={device: dict(all_trace=metrics(es),
                              inside_loop=metrics(es, window))
                 for device, es in sorted(by_device.items())},
        by_semantic_path=[dict(device=d, category=c, path=p, **metrics(es))
                          for (d, c, p), es in sorted(by_semantic.items())],
        by_device_event_name=[],
        host_semantic_ranges=[],
        host_runtime=[],
        caveats=[
            'Sums can overlap; category/path unions cannot be added together.',
            'Device union measures recorded activity, not utilization '
            'or occupancy.',
            'The gap to loop time is not proven GPU idle time.',
            'Host ranges/runtime durations are inclusive and may overlap '
            'device work.',
            'No transfer bytes are inferred when the trace omits them.',
        ])
    for (d, c, name), es in sorted(by_name.items()):
        byte_values = [e.get('args', {}).get('bytes') for e in es]
        known = [v for v in byte_values if isinstance(
            v, (int, float)) and v >= 0]
        result['by_device_event_name'].append(dict(
            device=d, category=c, name=name, **metrics(es),
            recorded_bytes=sum(known) if known else None,
            events_with_recorded_bytes=len(known)))
    for name in sorted({r['name'] for r in ranges}):
        result['host_semantic_ranges'].append(dict(
            name=name, **metrics([r for r in ranges if r['name'] == name])))
    runtime_groups = defaultdict(list)
    for i, e in enumerate(host):
        if e.get('cat') in RUNTIME_CATEGORIES:
            runtime_groups[e['name'], paths[i]].append(e)
    result['host_runtime'] = [dict(name=n, path=p, **metrics(es))
                              for (n, p), es in sorted(runtime_groups.items())]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    receipt = json.loads(args.receipt.read_text())
    if receipt.get('state') != 'complete' or receipt.get(
            'trace_sha256') != sha256:
        raise ValueError(
            'Trace does not match a completed acquisition receipt')
    result = summarize(json.loads(raw))
    if not result['devices']:
        raise ValueError(
            'No device activities captured; CUDA attribution unavailable')
    result['trace_sha256'] = sha256
    result['receipt_sha256'] = hashlib.sha256(
        args.receipt.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
