#!/usr/bin/env python3
"""Validate paired bin-audit reports and produce JSON, CSV, and a human report.

Consumes metrics.jsonl and summary.json only; never reads or changes bulk data.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

SCHEMA = 'pycbc-bin-accuracy-v1'
MODULE_SHA = '91860412bce0c3b4c5e6fac74d49f901e3d79b01d75929a18b8b5a39e0197c39'
LABELS = ('original', 'proposed')
REFERENCES = ('input_power', 'rounded_power_scan_only')
PATHS = ('old', 'current', 'ideal_public_rounding')
SNR_VALUES = (5.5, 8, 12, 20, 50, 100)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(text):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(text, parse_constant=invalid)


def distribution(values):
    values = sorted(values)
    require(values and all(math.isfinite(x) for x in values), 'Invalid distribution')
    position = (len(values)-1)*.95
    lo = int(position)
    p95 = values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(position-lo)
    return dict(min=values[0], median=statistics.median(values), p95=p95, max=values[-1])


def classify(old, current, low, high):
    if old == current:
        return 'same_edge'
    old_min, current_min = max(low-old, old-high, 0), max(low-current, current-high, 0)
    old_max = max(abs(old-low), abs(old-high))
    current_max = max(abs(current-low), abs(current-high))
    if old_max < current_min:
        return 'old_closer'
    if current_max < old_min:
        return 'current_closer'
    return 'unresolved'


def edge_stats(rows, reference, boundary_rows):
    counts = Counter(old_closer=0, current_closer=0, same_edge=0, unresolved=0)
    stable_distances = {p: [] for p in PATHS}
    all_distances = {p: [] for p in PATHS}
    internal_count = stable_count = 0
    for row in rows:
        oracle = row['references'][reference]
        edges = {p: row['paths'][p]['edges'] for p in PATHS}
        own_counts = Counter(old_closer=0, current_closer=0, same_edge=0, unresolved=0)
        number = row['inputs']['num_bins']
        require(len(oracle['edges']) == number+1 and len(oracle['boundaries']) == number,
                'Reference edge geometry mismatch')
        for j, boundary in enumerate(oracle['boundaries'][1:], 1):
            require(boundary['threshold_index'] == j and boundary['edge'] == oracle['edges'][j],
                    'Reference boundary identity mismatch')
            low, high = boundary['possible_edge_min'], boundary['possible_edge_max']
            require(low <= boundary['edge'] <= high and boundary['stable'] == (low == high),
                    'Inconsistent reference uncertainty interval')
            verdict = classify(edges['old'][j], edges['current'][j], low, high)
            own_counts[verdict] += 1
            internal_count += 1
            stable_count += bool(boundary['stable'])
            record = dict(index=row['metadata']['index'], psd=row['metadata']['audit_psd'],
                          reference=reference, boundary=j, stable=boundary['stable'],
                          reference_edge=boundary['edge'], possible_min=low, possible_max=high,
                          uncertainty_aware_comparison=verdict)
            for path in PATHS:
                distance = abs(edges[path][j]-boundary['edge'])
                all_distances[path].append(distance)
                if boundary['stable']:
                    stable_distances[path].append(distance)
                record[path+'_edge'] = edges[path][j]
                record[path+'_point_estimate_shift'] = edges[path][j]-boundary['edge']
            boundary_rows.append(record)
        require(own_counts == Counter(oracle['old_current_edge_comparison']['internal_uncertainty_aware_counts']),
                'Per-row winner counts disagree with recomputed intervals')
        require(oracle['unstable_internal_boundaries'] == sum(not b['stable'] for b in oracle['boundaries'][1:]),
                'Unstable boundary count mismatch')
        require(oracle['own_bin_quality']['beyond_discrete_cell_and_uncertainty'] == 0,
                'Reference partition exceeds its analytic cell budget')
        counts.update(own_counts)
    return dict(internal_boundaries=internal_count, stable=stable_count,
                ambiguous=internal_count-stable_count, uncertainty_aware_counts=dict(counts),
                paths={p: dict(stable_exact=sum(d == 0 for d in stable_distances[p]),
                               stable_one_cell_off=sum(d == 1 for d in stable_distances[p]),
                               stable_more_than_one_cell_off=sum(d > 1 for d in stable_distances[p]),
                               max_stable_edge_distance=max(stable_distances[p], default=None),
                               max_point_estimate_edge_distance=max(all_distances[p], default=None))
                       for p in PATHS})


def signal_imbalance(rows):
    """Analytical bin-imbalance lambda; this performs no signal injection."""
    output = {}
    for path in ('old', 'current', 'oracle'):
        coefficients = []
        for row in rows:
            quality = (row['references']['input_power']['own_bin_quality'] if path == 'oracle'
                       else row['paths'][path]['input_reference_bin_quality'])
            relative = quality['relative_to_target']
            require(len(relative) == row['inputs']['num_bins'], 'Bin power count mismatch')
            coefficient = math.fsum(x*x for x in relative)/len(relative)
            coefficients.append((row['metadata']['index'], coefficient))
        worst_index, worst_coefficient = max(coefficients, key=lambda value: value[1])
        output[path] = dict(
            coefficient_distribution=distribution([value for _, value in coefficients]),
            maximum_template_index=worst_index,
            by_snr={str(rho): distribution([rho*rho*value for _, value in coefficients])
                    for rho in SNR_VALUES},
            maximum_coefficient=worst_coefficient)
    return output


def path_stats(rows, path):
    values = [r['paths'][path] for r in rows]
    quality = [v['input_reference_bin_quality'] for v in values]
    result = dict(
        public_total_absolute_relative_error=distribution([abs(v['public_error_vs_input_power']['total_relative']) for v in values]),
        public_prefix_max_absolute_over_total=distribution([v['public_error_vs_input_power']['max_absolute_over_total'] for v in values]),
        true_bin_max_relative_error=distribution([q['max_absolute_relative_to_target'] for q in quality]),
        true_bin_rms_relative_error=distribution([q['rms_relative_to_target'] for q in quality]),
        bins_beyond_cell_and_uncertainty=sum(q['beyond_discrete_cell_and_uncertainty'] for q in quality),
        templates_beyond_cell_and_uncertainty=sum(q['beyond_discrete_cell_and_uncertainty'] > 0 for q in quality),
        empty_bins=sum(q['empty_bins'] for q in quality))
    if path != 'ideal_public_rounding':
        result.update(
            scan_total_absolute_relative_error_input=distribution([abs(v['scan_error_vs_input_power']['total_relative']) for v in values]),
            scan_total_absolute_relative_error_rounded=distribution([abs(v['scan_error_vs_rounded_power_scan_only']['total_relative']) for v in values]),
            scan_prefix_max_absolute_over_total_rounded=distribution([v['scan_error_vs_rounded_power_scan_only']['max_absolute_over_total'] for v in values]),
            stagnated_positive_cells=sum(v['stagnated_positive_cells'] for v in values),
            templates_with_stagnated_positive_cells=sum(v['stagnated_positive_cells'] > 0 for v in values),
            final_plateau_rounded_power_fraction=distribution([
                v['final_plateau_rounded_power']/r['references']['rounded_power_scan_only']['total']
                for r, v in zip(rows, values)]))
    return result


def aggregate(rows, summary, expected_templates=384, expected_module_sha=MODULE_SHA):
    require(summary['schema'] == SCHEMA and summary['status'] == 'complete', 'Incomplete/wrong summary')
    require(summary['inputs_unchanged_after_analysis'] is True and not summary['changed_inputs'],
            'Remote input integrity failed')
    require(not summary['capture_mismatches'], 'Captured replay mismatches')
    require(summary['module_sha256'] == expected_module_sha, 'Unexpected analyzer revision')
    require(summary['rows'] == len(rows) == 2*expected_templates, 'Incorrect row coverage')
    require(summary['template_count'] == expected_templates, 'Incorrect template count')
    grouped = {p: {} for p in LABELS}
    for row in rows:
        require(row['schema'] == SCHEMA and row['status'] == 'ok' and not row['capture_validation_failures'],
                'Invalid/failed metric row')
        meta, inputs, observed = row['metadata'], row['inputs'], row['observed']
        label, index = meta['audit_psd'], meta['index']
        require(label in grouped and index not in grouped[label], 'Unknown PSD or duplicate row')
        require(row['runtime']['longdouble_wider_than_float64'] is True and
                row['runtime']['longdouble_significand_bits'] > 53, 'Reference precision not extended')
        require(inputs['htilde_dtype'] == 'complex64' and inputs['psd_dtype'] == 'float32', 'Unexpected input precision')
        for name in ('captured_current_bins', 'captured_current_prefix'):
            exact = name+'_exact' if name.endswith('bins') else name+'_byte_exact'
            if observed[name+'_supplied']:
                require(observed.get(exact) is True, 'Captured replay equality failed')
        for path in PATHS:
            edges = row['paths'][path]['edges']
            require(len(edges) == inputs['num_bins']+1 and edges[-1] == inputs['kmax'] and
                    inputs['kmin'] <= edges[0] and all(a <= b for a, b in zip(edges[:-1], edges[1:])),
                    'Invalid candidate edge geometry')
        require(row['old_current_max_edge_difference'] == max(
            abs(a-b) for a, b in zip(row['paths']['old']['edges'], row['paths']['current']['edges'])),
            'Per-row old/current edge shift mismatch')
        grouped[label][index] = row
    require(set(grouped['original']) == set(grouped['proposed']) == set(range(expected_templates)),
            'Template index coverage must be contiguous and paired')
    boundary_rows, template_rows = [], []
    output = dict(schema='pycbc-bin-aggregate-v1', template_count=expected_templates, rows=len(rows), by_psd={})
    for label in LABELS:
        selected = [grouped[label][i] for i in range(expected_templates)]
        require(len({r['inputs']['active_psd_sha256'] for r in selected}) == 1, 'PSD changed across templates')
        metrics = dict(
            references={ref: edge_stats(selected, ref, boundary_rows) for ref in REFERENCES},
            paths={p: path_stats(selected, p) for p in PATHS},
            coverage=dict(native_power_templates=sum(r['observed']['native_mag_supplied'] for r in selected),
                          numpy_native_power_mismatch_templates=[r['metadata']['index'] for r in selected if not r['observed']['native_mag_equal_numpy']],
                          bins_captured_templates=sum(r['observed']['captured_current_bins_supplied'] for r in selected),
                          prefix_captured_templates=sum(r['observed']['captured_current_prefix_supplied'] for r in selected),
                          emulated_power_templates=[r['metadata']['index'] for r in selected if not r['observed']['native_mag_supplied']]),
            input_rounding_absolute_relative_total_error=distribution([abs(r['input_power_rounding']['total_relative_error']) for r in selected]),
            max_old_current_edge_difference=max(r['old_current_max_edge_difference'] for r in selected))
        public_differences = [sum(a != b for a, b in zip(r['paths']['current']['edges'][1:-1], r['paths']['ideal_public_rounding']['edges'][1:-1])) for r in selected]
        metrics['current_vs_ideal_public_rounding'] = dict(templates_with_different_edges=sum(n > 0 for n in public_differences),
                                                        different_internal_edges=sum(public_differences))
        total_counts = Counter(old=0, current=0, tied=0)
        for row in selected:
            old = abs(row['paths']['old']['scan_error_vs_input_power']['total_signed'])
            current = abs(row['paths']['current']['scan_error_vs_input_power']['total_signed'])
            total_counts['old' if old < current else 'current' if current < old else 'tied'] += 1
            record = dict(index=row['metadata']['index'], template_hash=row['metadata']['template_hash'], psd=label,
                          native_power=row['observed']['native_mag_supplied'], captured_bins=row['observed']['captured_current_bins_supplied'],
                          old_current_max_edge_difference=row['old_current_max_edge_difference'])
            for path in ('old', 'current'):
                value = row['paths'][path]
                record.update({path+'_scan_total_relative_error_input': value['scan_error_vs_input_power']['total_relative'],
                               path+'_scan_total_relative_error_rounded': value['scan_error_vs_rounded_power_scan_only']['total_relative'],
                               path+'_public_total_relative_error': value['public_error_vs_input_power']['total_relative'],
                               path+'_true_bin_max_relative_error': value['input_reference_bin_quality']['max_absolute_relative_to_target'],
                               path+'_stagnated_positive_cells': value['stagnated_positive_cells']})
            for path in ('old', 'current', 'oracle'):
                quality = (row['references']['input_power']['own_bin_quality'] if path == 'oracle'
                           else row['paths'][path]['input_reference_bin_quality'])
                coefficient = math.fsum(x*x for x in quality['relative_to_target'])/row['inputs']['num_bins']
                record[path+'_lambda_over_rho_squared'] = coefficient
                for rho in SNR_VALUES:
                    record[path+'_lambda_rho_'+str(rho)] = rho*rho*coefficient
            template_rows.append(record)
        metrics['total_error_closer_point_estimates'] = dict(total_counts)
        metrics['analytical_signal_bin_imbalance'] = signal_imbalance(selected)
        expected = summary['by_psd'][label]
        counts = metrics['references']['input_power']['uncertainty_aware_counts']
        for key, suffix in [('old_closer', 'old_closer'), ('current_closer', 'current_closer'), ('same_edge', 'same'), ('unresolved', 'unresolved')]:
            require(counts[key] == expected['input_reference_'+suffix+'_edges'], 'Summary boundary count mismatch')
        for key in ('old', 'current', 'tied'):
            source = key+'_total_error_closer_templates' if key != 'tied' else 'tied_total_error_templates'
            require(total_counts[key] == expected[source], 'Summary total-error count mismatch')
        require(metrics['max_old_current_edge_difference'] == expected['max_old_current_edge_shift'], 'Summary maximum edge shift mismatch')
        output['by_psd'][label] = metrics
    effects = {}
    for path in PATHS:
        shifts = []
        changed_templates = 0
        for index in range(expected_templates):
            a, b = grouped['original'][index], grouped['proposed'][index]
            require(a['metadata']['template_hash'] == b['metadata']['template_hash'] and
                    all(a['inputs'][k] == b['inputs'][k] for k in
                        ('active_htilde_sha256', 'kmin', 'kmax', 'num_bins', 'delta_f')),
                    'Paired PSD comparison does not hold waveform/geometry fixed')
            local = [abs(x-y) for x, y in zip(a['paths'][path]['edges'][1:-1], b['paths'][path]['edges'][1:-1])]
            shifts.extend(local)
            changed_templates += any(local)
        effects[path] = dict(changed_templates=changed_templates, changed_internal_edges=sum(d != 0 for d in shifts), max_edge_difference=max(shifts))
    output['paired_psd_effects_fixed_waveform'] = effects
    output['runtime'] = list({json.dumps(r['runtime'], sort_keys=True): r['runtime'] for r in rows}.values())
    output['normalization'] = list({json.dumps(r['normalization'], sort_keys=True): r['normalization'] for r in rows}.values())
    output['input_geometry'] = sorted({(r['inputs']['full_length'], r['inputs']['kmin'], r['inputs']['kmax'], r['inputs']['num_bins'], r['inputs']['delta_f']) for r in rows})
    output['remote_summary_recomputed_consistently'] = True
    output['reference_scope'] = 'Arithmetic on captured rounded complex64 waveforms and float32 PSDs; stable-edge classifications use reported longdouble uncertainty intervals.'
    output['verification_scope'] = 'Aggregate independently recomputes coverage, boundary classifications, and summary counts from delivered metrics. It does not independently recompute the bulk waveform reference arrays.'
    return output, template_rows, boundary_rows


def human_report(report):
    def f(value):
        return f'{value:.6g}'
    lines = ['# Equal-power bin accuracy on captured workload inputs', '',
             f"Validated {report['rows']} metric rows: {report['template_count']} captured waveforms, each evaluated with both frozen PSDs. Summary counts were independently recomputed and agree with the analyzer. Input-file integrity is attested by the remote analyzer; this aggregation independently pins the delivered reports.", '',
             'The mathematical reference uses the captured rounded complex64 waveform and float32 PSD. The scan-only reference starts with the rounded float32 power array. The public-rounding diagnostic publishes that latter longdouble prefix through float32 storage and normalization, then constructs float64 thresholds. These answer different accuracy questions.', '',
             '## Boundary locations', '',
             'All counts below refer to internal boundaries. Only stable reference boundaries support the exact/one-cell/larger-shift classification. An equal distance or ambiguous reference does not establish which path is better.', '',
             '| PSD | Reference | Stable / total | Old closer | Current closer | Same selected edge | Unresolved |',
             '|---|---|---:|---:|---:|---:|---:|']
    for label in LABELS:
        stats = report['by_psd'][label]
        for ref in REFERENCES:
            edges = stats['references'][ref]
            c = edges['uncertainty_aware_counts']
            lines.append(f"| {label} | {ref} | {edges['stable']} / {edges['internal_boundaries']} | {c['old_closer']} | {c['current_closer']} | {c['same_edge']} | {c['unresolved']} |")
    lines += ['', '| PSD | Current versus input-power reference: exact | One cell off | More than one cell off | Maximum stable shift |',
              '|---|---:|---:|---:|---:|']
    for label in LABELS:
        e = report['by_psd'][label]['references']['input_power']['paths']['current']
        lines.append(f"| {label} | {e['stable_exact']} | {e['stable_one_cell_off']} | {e['stable_more_than_one_cell_off']} | {e['max_stable_edge_distance']} |")
    lines += ['', 'A more accurate cumulative sum does not guarantee every public edge equals the mathematical reference. Input-power rounding and float32 prefix publication can still move a boundary. The following check compares current edges with the diagnostic that retains that same public rounding:', '']
    for label in LABELS:
        d = report['by_psd'][label]['current_vs_ideal_public_rounding']
        lines.append(f"- {label} PSD: {d['different_internal_edges']} internal edges differ across {d['templates_with_different_edges']} templates from the ideal-scan/public-rounding diagnostic.")
    lines += ['', '## Accumulation and final public values', '',
              'The values below are maximum absolute relative total errors across templates. Scan errors use the rounded-power reference; public errors use the input-power reference with exact normalization. They must not be treated as interchangeable.', '',
              '| PSD | Old scan | Current scan | Old public | Current public |', '|---|---:|---:|---:|---:|']
    for label in LABELS:
        p = report['by_psd'][label]['paths']
        values = [p[x]['scan_total_absolute_relative_error_rounded']['max'] for x in ('old', 'current')]
        values += [p[x]['public_total_absolute_relative_error']['max'] for x in ('old', 'current')]
        lines.append('| '+label+' | '+' | '.join(f(x) for x in values)+' |')
    lines += ['', '## Actual per-bin powers and tail loss', '',
              'Bin power is independently reduced over each selected interval using the input-power reference. Error is relative to total power divided by the bin count. The discrete-cell budget is the larger adjacent boundary-cell power, plus reference uncertainty; it is an analytic quantization check, not a search acceptance threshold.', '',
              '| PSD | Path | Median of per-template maximum bin error | Largest bin error | Bins beyond cell budget + uncertainty | Stagnated positive scan cells |',
              '|---|---|---:|---:|---:|---:|']
    for label in LABELS:
        for path in ('old', 'current'):
            s = report['by_psd'][label]['paths'][path]
            lines.append(f"| {label} | {path} | {f(s['true_bin_max_relative_error']['median'])} | {f(s['true_bin_max_relative_error']['max'])} | {s['bins_beyond_cell_and_uncertainty']} | {s['stagnated_positive_cells']} |")
    lines += ['', 'Stagnated-cell power is not automatically the net accumulation error: later rounding can compensate. The JSON separately reports the rounded power remaining after the final scan increment, plus full-prefix and total errors.', '',
              '## Analytical signal consequence of bin imbalance', '',
              'For a perfectly matched noiseless signal with fixed PSD and bins, the analytical bin-imbalance noncentrality diagnostic is `lambda(rho) = rho^2 / M * sum_j(relative_to_target_j^2)`, with M=16 for this capture. Relative errors come from each partition’s true input-reference bin powers. Oracle means the mathematical input-power reference partition, including its unavoidable discrete-cell imbalance.', '',
              'This calculation is not an injected-strain recovery result. It holds the waveform, PSD, partition, and SNR definition fixed; it does not simulate noise, template mismatch, trigger selection, sensitivity, or false-alarm rates.', '',
              'The table gives the maximum lambda across templates for each PSD and partition; the JSON includes the median and 95th percentile, and the template CSV contains every value.', '',
              '| PSD | Partition | SNR 5.5 | SNR 8 | SNR 12 | SNR 20 | SNR 50 | SNR 100 | Maximum template |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in LABELS:
        for path in ('old', 'current', 'oracle'):
            metric = report['by_psd'][label]['analytical_signal_bin_imbalance'][path]
            values = ' | '.join(f(metric['by_snr'][str(rho)]['max']) for rho in SNR_VALUES)
            lines.append(f"| {label} | {path} | {values} | {metric['maximum_template_index']} |")
    lines += ['', '## Coverage, precision, and PSD substitution', '']
    for label in LABELS:
        c = report['by_psd'][label]['coverage']
        lines.append(f"- {label} PSD: {c['native_power_templates']} native power captures, {c['bins_captured_templates']} captured bin checks, {c['prefix_captured_templates']} byte-exact public-prefix checks; {len(c['emulated_power_templates'])} templates use source emulation.")
        if 0 < len(c['emulated_power_templates']) <= 12:
            lines.append('  Emulation indices: '+', '.join(map(str, c['emulated_power_templates']))+'.')
        if c['numpy_native_power_mismatch_templates']:
            lines.append('  Native/NumPy magnitude discrepancies occur at indices: '+', '.join(map(str, c['numpy_native_power_mismatch_templates']))+'. Native values remain authoritative for reconstructed CPU paths.')
    for runtime in report['runtime']:
        lines.append(f"- Reference runtime: NumPy {runtime['numpy_version']}, Python {runtime['python_version']}, {runtime['machine']}; longdouble has {runtime['longdouble_significand_bits']} significand bits ({runtime['longdouble_storage_bits']} storage bits).")
    lines += ['', 'Holding each captured waveform fixed while switching PSDs changes the following selected boundaries. This isolates PSD substitution effects; it does not establish which physical PSD is correct.', '',
              '| Path | Templates with changed edges | Changed internal edges | Maximum shift |', '|---|---:|---:|---:|']
    for path, values in report['paired_psd_effects_fixed_waveform'].items():
        lines.append(f"| {path} | {values['changed_templates']} | {values['changed_internal_edges']} | {values['max_edge_difference']} |")
    lines += ['', 'The findings concern numerical accuracy on these saved inputs. They do not establish end-to-end scientific equivalence, sensitivity, or false-alarm behavior. No frozen comparison tolerance has been changed. Full intermediate numbers are in bin-aggregate.json; per-template and boundary records are in the companion CSV files.', '',
              'Reference boundary uncertainty assumes IEEE round-to-nearest and finite normal longdouble arithmetic. Total-error rankings and floating scalar summaries are descriptive point estimates. The aggregator checks reports; selected waveform fixtures require separate independent recomputation.', '',
              '## Provenance', '', '```json', json.dumps({key: value for key, value in report.get('provenance', {}).items() if key != 'remote_input_sha256'}, indent=2), '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis_dir', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-templates', type=int, default=384)
    parser.add_argument('--expected-module-sha256', default=MODULE_SHA)
    args = parser.parse_args()
    root, out = args.analysis_dir.resolve(), args.output_dir.resolve()
    require(not out.exists() and root != out and not root.is_relative_to(out) and not out.is_relative_to(root),
            'Output must be a fresh directory separate from analysis input')
    paths = {name: root/name for name in ('metrics.jsonl', 'summary.json')}
    pins = {name: sha(path) for name, path in paths.items()}
    rows = [strict_json(line) for line in paths['metrics.jsonl'].read_text().splitlines() if line.strip()]
    summary = strict_json(paths['summary.json'].read_text())
    report, templates, boundaries = aggregate(rows, summary, args.expected_templates, args.expected_module_sha256)
    require(pins == {name: sha(path) for name, path in paths.items()}, 'Delivered reports changed during aggregation')
    report['provenance'] = dict(analysis_directory=str(root), report_sha256=pins,
                               analyzer_sha256=summary['module_sha256'], aggregate_script_sha256=sha(Path(__file__).resolve()),
                               remote_input_file_count=len(summary['input_sha256']),
                               remote_input_sha256=summary['input_sha256'])
    out.mkdir(parents=True)
    (out/'bin-aggregate.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for filename, records in [('bin-templates.csv', templates), ('bin-boundaries.csv', boundaries)]:
        with (out/filename).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (out/'bin-report.md').write_text(human_report(report))
    print(json.dumps(dict(status='complete', rows=len(rows), report=str(out/'bin-report.md'))))


if __name__ == '__main__':
    main()
