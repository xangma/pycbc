#!/usr/bin/env python3
"""Report input equivalence for the six normal-CPU length/padding qualifications.

Read-only except for a new --output JSON. Requires NumPy. PSD statistics use
exact saved arrays, with float64 arithmetic on common nested controller bins.
Conditioned strain was hashed at capture, not saved: its equality is established
from bound capture records; only PSD files/data and gating hashes are recomputed.
"""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np


LENGTHS = (256, 512, 1024)
PADS = (112, 96)
VARYING_OPTIONS = {'--output', '--segment-length', '--segment-start-pad'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def valid_hash(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value),
            'Missing or malformed SHA256')
    return value


def read_json(path, root, hashes):
    raw = path.read_bytes()
    hashes[str(path.relative_to(root))] = sha(raw)
    value = json.loads(raw)
    canonical(value)  # Reject nonfinite JSON numbers.
    return value


def options(argv):
    require(isinstance(argv, list) and argv and
            all(isinstance(arg, str) for arg in argv), 'Malformed executable argv')
    result, key = {}, None
    for arg in argv[1:]:
        if arg.startswith('--'):
            key, separator, value = arg.partition('=')
            require(key not in result, f'Repeated option: {key}')
            result[key] = [value] if separator else []
        else:
            require(key is not None, 'Unexpected positional argument')
            result[key].append(arg)
    return result


def load_case(root, length, pad, hashes):
    case = f'qual-v2-cpu-l{length}' if pad == 112 else f'qual-cpu-l{length}-s96'
    directory = root / 'runs' / case
    receipt = read_json(directory / 'receipt.json', root, hashes)
    q = read_json(directory / 'qualification.json', root, hashes)
    require(receipt['case'] == case and receipt['mode'] == 'qualify' and
            receipt['scheme'] == 'cpu:1' and receipt['state'] == 'complete' and
            receipt['returncode'] == 0, f'{case}: incomplete or wrong run')
    require(receipt['segment_length'] == length and receipt['start_pad'] == pad and
            receipt['end_pad'] == 16, f'{case}: unexpected geometry')
    inputs = receipt['input_sha256']
    require(inputs and inputs == receipt['input_sha256_after'] and
            not receipt['source_info']['status'] and
            not receipt['source_info']['tracked_diff'] and
            not receipt['source_status_after'], f'{case}: changed inputs/source')
    for value in inputs.values():
        valid_hash(value)
    require(q['status'] == 'success' and q['executable_exit_code'] == 0 and
            q['checks'] and all(v is True for v in q['checks'].values()),
            f'{case}: qualification failed')
    argv = receipt['executable_cli']
    require(q['executed_argv'] == q['argv'] == argv and
            q['host'] == receipt['hostname'] and q['source_root'] == receipt['source'],
            f'{case}: qualification/run provenance mismatch')
    for record in (q['executable'], q['wrapper']):
        require(inputs.get(record['path']) == valid_hash(record['sha256']),
                f'{case}: executable/wrapper hash is unbound')
    opts = options(argv)
    for key, expected in (('--segment-length', length), ('--segment-start-pad', pad),
                          ('--segment-end-pad', 16), ('--processing-scheme', 'cpu:1')):
        require(opts[key] == [str(expected)], f'{case}: inconsistent {key}')
    observed = q['observations']
    for key in ('conditioned_strain', 'segment_geometry',
                'matched_filter_controllers', 'psd_arrays'):
        require(len(observed[key]) == 1, f'{case}: expected one {key} record')
    strain = observed['conditioned_strain'][0]
    gate = strain['gating_info']
    require(sha(canonical(gate['values']).encode()) == valid_hash(gate['sha256']),
            f'{case}: gating hash mismatch')
    valid_hash(strain['data_sha256'])
    dtype = np.dtype(strain['dtype_str'])
    rate = strain['sample_rate_hz']
    require(dtype.kind == 'f' and strain['dtype'] == str(dtype) and
            strain['shape'] == [strain['n_samples']] and strain['n_samples'] > 0 and
            strain['nbytes'] == strain['n_samples'] * dtype.itemsize and rate > 0 and
            strain['delta_t_seconds'] == 1 / rate and
            (float(strain['end_time']) - float(strain['start_time'])) * rate ==
            strain['n_samples'], f'{case}: malformed strain capture metadata')
    geometry = observed['segment_geometry'][0]
    controller = observed['matched_filter_controllers'][0]
    psd = observed['psd_arrays'][0]
    count = len(geometry['segments'])
    require(count == controller['segment_count'] and count > 0 and
            psd['segment_indices'] == list(range(count)),
            f'{case}: saved PSD does not cover every segment')
    require(geometry['sample_rate_hz'] == rate and
            geometry['fft_samples'] == length * rate and
            geometry['frequency_samples'] == geometry['fft_samples'] // 2 + 1 and
            geometry['delta_f_hz'] == psd['delta_f_hz'] == 1 / length,
            f'{case}: inconsistent frequency grid')
    # Map the recorded remote path into this local receipt directory. Never read
    # the recorded absolute path or flatten arrays from different cases together.
    relative = Path('arrays') / Path(psd['path']).name
    require(psd['relative_path'] == str(relative) and
            Path(psd['path']).parts[-4:] == ('runs', case, *relative.parts),
            f'{case}: malformed saved PSD path')
    path = directory / relative
    raw = path.read_bytes()
    require(len(raw) == psd['bytes'] and sha(raw) == valid_hash(psd['sha256']),
            f'{case}: PSD file hash/size mismatch')
    hashes[str(path.relative_to(root))] = sha(raw)
    with path.open('rb') as stream:
        values = np.load(stream, allow_pickle=False)
        require(not stream.read(1), f'{case}: trailing PSD file data')
    require(values.ndim == 1 and values.dtype.kind == 'f' and
            values.dtype.str == psd['dtype_str'] and str(values.dtype) == psd['dtype'] and
            list(values.shape) == psd['shape'] == [geometry['frequency_samples']] and
            values.size == psd['n_samples'] and values.nbytes == psd['nbytes'] and
            sha(np.ascontiguousarray(values).tobytes()) == valid_hash(psd['data_sha256']),
            f'{case}: PSD data hash/metadata mismatch')
    lo, hi = controller['filter_bin_start'], controller['filter_bin_stop']
    require(type(lo) is int and type(hi) is int and 0 <= lo < hi <= values.size,
            f'{case}: invalid controller band')
    require(psd['validity']['filter_bin_start'] == lo and
            psd['validity']['filter_bin_stop'] == hi and
            psd['validity']['filter_bins_are_half_open'] is True and
            psd['validity']['valid_for_filter'] is True and
            np.all(np.isfinite(values[lo:hi]) & (values[lo:hi] > 0)) and
            not np.any(np.isnan(values) | np.isneginf(values) | (values <= 0)),
            f'{case}: invalid PSD values/controller metadata')
    require(psd['scaling'] == 'DYN_RANGE_FAC**2' and
            np.isfinite(psd['dyn_range_factor']) and psd['dyn_range_factor'] > 0,
            f'{case}: invalid PSD scaling')
    shared = dict(source=receipt['source_info'], host=receipt['hostname'],
                  environment=receipt['environment'], input_sha256=inputs,
                  executable=argv[0], source_modules=q['source_modules'],
                  options={k: v for k, v in opts.items() if k not in VARYING_OPTIONS})
    metadata = dict(case=case, segment_length=length, start_pad=pad, end_pad=16,
                    segment_count=count, conditioned_strain=strain,
                    psd_file=str(path.relative_to(root)), psd_file_sha256=sha(raw),
                    psd_data_sha256=psd['data_sha256'], psd_dtype=psd['dtype_str'],
                    delta_f_hz=psd['delta_f_hz'], scaling=psd['scaling'],
                    dyn_range_factor=psd['dyn_range_factor'],
                    controller_bin_slice=[lo, hi],
                    controller_band_hz_half_open=[lo / length, hi / length],
                    excluded_positive_infinity_bins=int(np.isposinf(values).sum()))
    return dict(metadata=metadata, shared=shared, values=values)


def difference_stats(reference, candidate):
    delta = candidate - reference
    relative = delta / reference
    absolute_relative = np.abs(relative)
    return dict(exact_equal=bool(np.array_equal(reference, candidate)),
                max_absolute_difference=float(np.max(np.abs(delta))),
                relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(reference)),
                mean_signed_relative=float(np.mean(relative)),
                rms_relative=float(np.sqrt(np.mean(relative ** 2))),
                absolute_relative_quantiles=dict(zip(
                    ('p50', 'p95', 'p99', 'maximum'),
                    map(float, np.quantile(absolute_relative, [0.5, 0.95, 0.99, 1])))))


def compare_lengths(coarse, fine):
    a, b = coarse['metadata'], fine['metadata']
    ratio = a['delta_f_hz'] / b['delta_f_hz']
    require(ratio >= 1 and ratio == int(ratio), 'Frequency grids are not nested')
    stride = int(ratio)
    require((len(coarse['values']) - 1) * stride == len(fine['values']) - 1 and
            (a['psd_dtype'], a['scaling'], a['dyn_range_factor']) ==
            (b['psd_dtype'], b['scaling'], b['dyn_range_factor']),
            'Incompatible PSD extent, precision or scaling')
    lo = max(a['controller_bin_slice'][0],
             (b['controller_bin_slice'][0] + stride - 1) // stride)
    hi = min(a['controller_bin_slice'][1],
             (b['controller_bin_slice'][1] + stride - 1) // stride)
    require(lo < hi, 'No common controller bins')
    x = coarse['values'][lo:hi].astype(np.float64)
    y = fine['values'][lo * stride:hi * stride:stride].astype(np.float64)
    require(x.shape == y.shape and np.all(np.isfinite(x) & (x > 0)) and
            np.all(np.isfinite(y) & (y > 0)), 'Malformed common-bin PSD values')
    return dict(reference_case=a['case'], candidate_case=b['case'], bins=int(x.size),
                coarse_bin_slice=[lo, hi], fine_bin_stride=stride,
                first_frequency_hz=lo * a['delta_f_hz'],
                last_frequency_hz=(hi - 1) * a['delta_f_hz'],
                delta_f_hz=a['delta_f_hz'], psd=difference_stats(x, y),
                inverse_psd=difference_stats(1 / x, 1 / y))


def summarize(root):
    hashes = {}
    runs = {(length, pad): load_case(root, length, pad, hashes)
            for length in LENGTHS for pad in PADS}
    baseline = runs[(LENGTHS[0], PADS[0])]
    require(all(r['shared'] == baseline['shared'] for r in runs.values()),
            'Source, environment, material inputs or other executable options differ')
    strain_equal = all(r['metadata']['conditioned_strain'] ==
                       baseline['metadata']['conditioned_strain'] for r in runs.values())
    padding = []
    for length in LENGTHS:
        a, b = (runs[(length, pad)] for pad in PADS)
        require(all(a['metadata'][key] == b['metadata'][key] for key in
                    ('delta_f_hz', 'psd_dtype', 'scaling', 'dyn_range_factor',
                     'controller_bin_slice')), 'Padding changes PSD grid/scaling/band')
        padding.append(dict(segment_length=length, cases=[a['metadata']['case'],
                                                         b['metadata']['case']],
                            exact_array_equal=bool(np.array_equal(a['values'], b['values'])),
                            data_hash_equal=a['metadata']['psd_data_sha256'] ==
                            b['metadata']['psd_data_sha256'],
                            file_hash_equal=a['metadata']['psd_file_sha256'] ==
                            b['metadata']['psd_file_sha256']))
    equivalent = strain_equal and all(p['exact_array_equal'] and p['data_hash_equal']
                                     for p in padding)
    return dict(schema_version=1, status='pass' if equivalent else 'different',
                reporter_sha256=sha(Path(__file__).read_bytes()), input_sha256=hashes,
                shared_provenance=baseline['shared'],
                final_conditioned_strain_and_gating_capture_equal=strain_equal,
                gating_capture_hashes_recomputed=True,
                runs=[r['metadata'] for r in runs.values()], padding_comparisons=padding,
                length_comparisons=[compare_lengths(runs[(a, pad)], runs[(b, pad)])
                                    for pad in PADS for a, b in combinations(LENGTHS, 2)],
                definitions={
                    'relative_difference': '(candidate - reference) / reference',
                    'relative_l2': 'norm(candidate - reference) / norm(reference)',
                    'inverse_psd': 'float64 reciprocal of each exact saved scaled PSD',
                    'absolute_difference_units': 'PSD: saved scaled units; inverse PSD: reciprocal scaled units',
                    'frequency_selection': 'Exact nested bins inside both actual half-open controller slices; no interpolation',
                    'status': 'Requires identical strain/gating captures and PSD data across padding; cross-length PSD differences are descriptive',
                    'strain_evidence': 'Capture hashes and metadata only; conditioned strain samples were not saved',
                    'input_evidence': 'Receipt input hashes must agree before/after and across cases; remote original inputs are not reread',
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a new output path; existing reports are retained')
    result = summarize(args.root.resolve())
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with args.output.open('x') as stream:
        stream.write(encoded)
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
