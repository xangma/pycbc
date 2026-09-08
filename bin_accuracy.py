#!/usr/bin/env python3
"""Independent, read-only equal-power bin audit for captured complex64 templates.

No PyCBC imports, source mutation, remote access, or frozen-comparator changes.
See method.md for the numerical model, uncertainty bound, and capture contract.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

LD = np.longdouble
SCHEMA = 'pycbc-bin-accuracy-v1'


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _floats(array):
    return [float(x) for x in array]


def _error(value, reference):
    error = np.asarray(value, dtype=LD) - reference
    total = reference[-1] if np.ndim(reference) else reference
    return dict(total_signed=float(error[-1] if error.ndim else error),
                total_relative=float((error[-1] if error.ndim else error) / total),
                max_absolute=float(np.max(np.abs(error))),
                max_absolute_over_total=float(np.max(np.abs(error)) / total))


def _bound(total, operations):
    """Conservative absolute forward bound, expressed using the computed total."""
    unit = LD(np.finfo(LD).eps) / 2
    product = LD(operations) * unit
    if product >= LD('0.01'):
        raise ValueError('Reference error bound is too large for this audit')
    gamma = product / (1 - product)
    return gamma / (1 - gamma) * total


def _bisect_right(values, threshold):
    lo, hi = 0, len(values)
    while lo < hi:
        middle = (lo + hi) // 2
        if values[middle] <= threshold:
            lo = middle + 1
        else:
            hi = middle
    return lo


def _edges(prefix, thresholds):
    edges = np.searchsorted(prefix, thresholds, side='right')
    if np.any(prefix[1:] < prefix[:-1]):
        raise ValueError('Nonmonotone prefix')
    for edge, threshold in zip(edges, thresholds):
        assert int(edge) == _bisect_right(prefix, threshold)
        assert edge == 0 or prefix[edge - 1] <= threshold
        assert edge == len(prefix) or prefix[edge] > threshold
    return np.append(edges, len(prefix)).astype(np.int64)


def _oracle(weights, num_bins, kmin, power_operations):
    prefix = np.cumsum(weights, dtype=LD)
    total = prefix[-1]
    thresholds = np.arange(num_bins, dtype=LD) * total / LD(num_bins)
    edges = _edges(prefix, thresholds)
    total_bound = _bound(total, len(weights) + power_operations + 8)
    rows = []
    for i, (edge, threshold) in enumerate(zip(edges[:-1], thresholds)):
        # A common absolute prefix bound covers every prefix. Threshold error
        # includes uncertainty in total and its multiplication/division.
        threshold_bound = LD(i) / num_bins * total_bound + _bound(abs(threshold), 2)
        combined = total_bound + threshold_bound
        earliest = int(np.searchsorted(prefix, threshold - combined, side='right'))
        latest = int(np.searchsorted(prefix, threshold + combined, side='right'))
        if i == 0:
            # With nonnegative weights, a zero prefix is exactly characterized
            # by leading zero terms; no uncertain subtraction is needed.
            earliest = latest = int(edge)
        # Independently reduce the selected left interval instead of reusing
        # the sequential cumulative value. Check both adjacent inequalities.
        before = np.sum(weights[:edge], dtype=LD)
        after = before + weights[edge] if edge < len(weights) else before
        check_bound = 2 * total_bound + threshold_bound
        consistent = before <= threshold + check_bound and (
            edge == len(weights) or after >= threshold - check_bound)
        if not consistent:
            raise ArithmeticError('Independent boundary reduction contradicts oracle')
        left = prefix[edge - 1] if edge else LD(0)
        right = prefix[edge] if edge < len(prefix) else None
        tie_left = int(np.searchsorted(prefix, threshold, side='left'))
        rows.append(dict(
            threshold_index=i, edge=int(edge) + kmin,
            possible_edge_min=earliest + kmin, possible_edge_max=latest + kmin,
            stable=earliest == latest, threshold_decimal=str(threshold),
            uncertainty_absolute_decimal=str(combined),
            previous_prefix_margin=float(threshold - left),
            next_prefix_margin=None if right is None else float(right - threshold),
            computed_equal_prefix_cells=int(edge) - tie_left,
            independent_boundary_reduction_consistent=bool(consistent),
            independent_previous_margin=float(threshold - before),
            independent_next_margin=None if edge == len(weights) else float(after - threshold)))
    return dict(prefix=prefix, total=total, edges=edges, thresholds=thresholds,
                total_bound=total_bound, boundaries=rows)


def _bin_quality(edges, weights, oracle):
    total = oracle['total']
    number = len(edges) - 1
    target = total / number
    actual = np.asarray([np.sum(weights[a:b], dtype=LD)
                         for a, b in zip(edges[:-1], edges[1:])], dtype=LD)
    error = actual - target
    # At a correct right-search boundary e, exclusive P[e] <= target < P[e+1].
    # Each boundary deficit lies in [0, w[e]); a bin error is the difference
    # of its two boundary deficits, bounded by the larger adjacent cell.
    boundary_cell = np.asarray([weights[e] if e < len(weights) else LD(0)
                                for e in edges], dtype=LD)
    budget = np.maximum(boundary_cell[:-1], boundary_cell[1:])
    uncertainty = 4 * oracle['total_bound']
    boundary_targets = np.arange(number + 1, dtype=LD) * total / number
    exclusive = np.concatenate((np.zeros(1, dtype=LD), oracle['prefix']))
    deficits = boundary_targets - exclusive[edges]
    return dict(
        powers=_floats(actual), target_power=float(target),
        signed_errors=_floats(error), relative_to_target=_floats(error / target),
        rms_relative_to_target=float(np.sqrt(np.mean((error / target)**2))),
        max_absolute_relative_to_target=float(np.max(abs(error)) / target),
        discrete_cell_budgets=_floats(budget),
        reference_uncertainty_absolute=float(uncertainty),
        beyond_discrete_cell_and_uncertainty=int(np.count_nonzero(abs(error) > budget + uncertainty)),
        boundary_deficits=_floats(deficits),
        excluded_head_power=float(np.sum(weights[:edges[0]], dtype=LD)),
        excluded_tail_power=float(np.sum(weights[edges[-1]:], dtype=LD)),
        covered_power_relative_error=float((np.sum(actual, dtype=LD) - total) / total),
        empty_bins=int(np.count_nonzero(np.diff(edges) == 0)))


def _edge_comparison(old_edges, current_edges, oracle, kmin):
    old_distance = abs(old_edges - oracle['edges'])
    current_distance = abs(current_edges - oracle['edges'])
    counts = dict(old_closer=0, current_closer=0, same_edge=0, unresolved=0)
    for old, current, boundary in zip(old_edges[1:-1], current_edges[1:-1], oracle['boundaries'][1:]):
        lo = boundary['possible_edge_min'] - kmin
        hi = boundary['possible_edge_max'] - kmin
        old_min = max(lo - old, old - hi, 0)
        current_min = max(lo - current, current - hi, 0)
        old_max = max(abs(old - lo), abs(old - hi))
        current_max = max(abs(current - lo), abs(current - hi))
        if old == current:
            counts['same_edge'] += 1
        elif old_max < current_min:
            counts['old_closer'] += 1
        elif current_max < old_min:
            counts['current_closer'] += 1
        else:
            counts['unresolved'] += 1
    return dict(
        max_old_edge_shift=int(old_distance.max()), max_current_edge_shift=int(current_distance.max()),
        internal_point_estimate_counts=dict(
            old_closer=int(np.count_nonzero(old_distance[1:-1] < current_distance[1:-1])),
            current_closer=int(np.count_nonzero(current_distance[1:-1] < old_distance[1:-1])),
            tied_distance=int(np.count_nonzero(current_distance[1:-1] == old_distance[1:-1]))),
        internal_uncertainty_aware_counts=counts)


def _active(array, full_length, kmin, kmax, name, dtype):
    array = np.asarray(array)
    if array.dtype != np.dtype(dtype) or array.ndim != 1:
        raise ValueError(f'{name} must be a one-dimensional {np.dtype(dtype)} array')
    if len(array) == full_length:
        return array[kmin:kmax]
    if len(array) == kmax - kmin:
        return array
    raise ValueError(f'{name} must contain either the full vector or the active slice')


def audit(htilde, psd, delta_f, kmin, kmax, num_bins=16, metadata=None, *,
          mag_numpy=None, captured_current_bins=None, captured_current_prefix=None, arrays=None):
    """Return JSON-safe metrics; optionally populate arrays without writing files.

    htilde is the original full complex64 vector; psd is full float32 or None.
    mag_numpy, when provided, is native squared_norm()/psd before the CPU cast.
    Prefix and magnitude validation accepts full vectors or active slices.
    """
    h = np.asarray(htilde)
    if h.ndim != 1 or h.dtype != np.dtype(np.complex64):
        raise ValueError('htilde must be a one-dimensional original complex64 array')
    if any(isinstance(x, (bool, np.bool_)) or int(x) != x for x in (kmin, kmax, num_bins)):
        raise ValueError('kmin, kmax and num_bins must be integers')
    kmin, kmax, num_bins = int(kmin), int(kmax), int(num_bins)
    if not 0 <= kmin < kmax <= len(h) or num_bins < 1:
        raise ValueError('Invalid active slice or bin count')
    if not np.isfinite(delta_f) or delta_f <= 0:
        raise ValueError('delta_f must be finite and positive')
    hs = h[kmin:kmax]
    if not np.isfinite(hs).all():
        raise ValueError('Nonfinite active waveform')
    p = None
    if psd is not None:
        p = np.asarray(psd)
        if p.ndim != 1 or p.dtype != np.dtype(np.float32) or len(p) != len(h):
            raise ValueError('psd must be a full float32 vector matching htilde')
        p = p[kmin:kmax]
        if not np.all(np.isfinite(p) & (p > 0)):
            raise ValueError('Active PSD must be finite and strictly positive')
    # Match array_cpu.squared_norm and its subsequent in-place division.
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        numpy_mag = hs.real**2 + hs.imag**2
        if p is not None:
            numpy_mag /= p
    mag = numpy_mag if mag_numpy is None else _active(mag_numpy, len(h), kmin, kmax, 'mag_numpy', np.float32)
    if not np.all(np.isfinite(mag) & (mag >= 0)):
        raise ValueError('Invalid rounded native power')
    # Cast real and imaginary parts before any power operation. These are
    # references to the already rounded complex64 inputs, not latent waveforms.
    real, imag = hs.real.astype(LD), hs.imag.astype(LD)
    power = real * real + imag * imag
    if p is not None:
        power /= p.astype(LD)
    if not np.all(np.isfinite(power) & (power >= 0)) or np.sum(power, dtype=LD) <= 0:
        raise ValueError('Reference total power must be finite and positive')
    if np.sum(mag, dtype=LD) <= 0:
        raise ValueError('Rounded CPU power is all zero; CPU binning is degenerate')
    references = {'input_power': _oracle(power, num_bins, kmin, 4),
                  'rounded_power_scan_only': _oracle(mag.astype(LD), num_bins, kmin, 0)}
    old_scan = np.cumsum(mag, dtype=np.float32)
    current_scan = np.cumsum(mag, dtype=np.float64)
    norm64 = np.float64(4.0) * np.float64(delta_f)
    norm32 = np.float32(norm64)
    if not np.isfinite(norm32) or norm32 <= 0:
        raise ValueError('CPU public normalization must be finite and positive in float32')
    public = {'old': old_scan * norm32,
              'current': current_scan.astype(np.float32) * norm32,
              'ideal_public_rounding': references['rounded_power_scan_only']['prefix'].astype(np.float32) * norm32}
    paths, edge_vectors, thresholds = {}, {}, {}
    for name, vector in public.items():
        if not np.isfinite(vector).all() or vector[-1] <= 0:
            raise ValueError('Nonfinite/zero CPU public prefix')
        # np.arange(integer) * scalar(float32) promotes thresholds to float64.
        threshold = np.arange(num_bins, dtype=np.int64) * vector[-1] / num_bins
        if threshold.dtype != np.dtype(np.float64):
            raise TypeError('Unexpected NumPy threshold promotion; refusing silent emulation drift')
        edges = _edges(vector, threshold)
        edge_vectors[name], thresholds[name] = edges, threshold
        paths[name] = dict(
            edges=(edges + kmin).tolist(), public_prefix_sha256=array_sha(vector),
            public_total=float(vector[-1]), thresholds_float64=threshold.tolist(),
            public_error_vs_input_power=_error(vector, references['input_power']['prefix'] * (LD(4) * LD(delta_f))),
            public_error_vs_rounded_power_scan_only=_error(vector, references['rounded_power_scan_only']['prefix'] * LD(norm32)),
            input_reference_bin_quality=_bin_quality(edges, power, references['input_power']))
    for name, scan in (('old', old_scan), ('current', current_scan)):
        previous = np.concatenate((np.zeros(1, dtype=scan.dtype), scan[:-1]))
        stagnant = (scan == previous) & (mag > 0)
        changes = np.flatnonzero(scan != previous)
        tail_start = int(changes[-1] + 1) if len(changes) else 0
        last_tenth = 9 * len(mag) // 10
        before_tail = LD(scan[last_tenth - 1]) if last_tenth else LD(0)
        rounded_tail = np.sum(mag[last_tenth:].astype(LD), dtype=LD)
        paths[name].update(
            scan_error_vs_rounded_power_scan_only=_error(scan, references['rounded_power_scan_only']['prefix']),
            scan_error_vs_input_power=_error(scan, references['input_power']['prefix']),
            stagnated_positive_cells=int(stagnant.sum()),
            stagnated_cells_rounded_power=float(np.sum(mag[stagnant].astype(LD), dtype=LD)),
            stagnated_cells_input_reference_power=float(np.sum(power[stagnant], dtype=LD)),
            final_plateau_start=kmin + tail_start,
            final_plateau_positive_cells=int(np.count_nonzero(mag[tail_start:] > 0)),
            final_plateau_rounded_power=float(np.sum(mag[tail_start:].astype(LD), dtype=LD)),
            final_plateau_input_reference_power=float(np.sum(power[tail_start:], dtype=LD)),
            last_tenth_tail=dict(start=kmin + last_tenth,
                                 input_reference_power=float(np.sum(power[last_tenth:], dtype=LD)),
                                 rounded_power=float(rounded_tail),
                                 represented_scan_increment=float(LD(scan[-1]) - before_tail),
                                 signed_scan_error=float(LD(scan[-1]) - before_tail - rounded_tail)))
    observed = dict(native_mag_supplied=mag_numpy is not None,
                    native_mag_equal_numpy=mag.tobytes() == numpy_mag.tobytes(),
                    native_mag_numpy_differing_cells=int(np.count_nonzero(mag != numpy_mag)),
                    captured_current_bins_supplied=captured_current_bins is not None,
                    captured_current_prefix_supplied=captured_current_prefix is not None)
    if captured_current_bins is not None:
        captured = np.asarray(captured_current_bins)
        expected = edge_vectors['current'] + kmin
        observed['captured_current_bins_exact'] = bool(captured.shape == expected.shape and np.array_equal(captured, expected))
    if captured_current_prefix is not None:
        captured = _active(captured_current_prefix, len(h), kmin, kmax, 'captured_current_prefix', np.float32)
        observed['captured_current_prefix_byte_exact'] = captured.tobytes() == public['current'].tobytes()
    failures = [key for key in ('captured_current_bins_exact', 'captured_current_prefix_byte_exact') if observed.get(key) is False]
    oracle_rows = {}
    for name, oracle in references.items():
        oracle_weights = power if name == 'input_power' else mag.astype(LD)
        oracle_rows[name] = dict(
            total=float(oracle['total']), total_decimal=str(oracle['total']),
            total_uncertainty_absolute_decimal=str(oracle['total_bound']),
            edges=(oracle['edges'] + kmin).tolist(), boundaries=oracle['boundaries'],
            unstable_internal_boundaries=sum(not row['stable'] for row in oracle['boundaries'][1:]),
            own_bin_quality=_bin_quality(oracle['edges'], oracle_weights, oracle),
            old_current_edge_comparison=_edge_comparison(edge_vectors['old'], edge_vectors['current'], oracle, kmin))
    finfo = np.finfo(LD)
    result = dict(
        schema=SCHEMA, status='capture_mismatch' if failures else 'ok', capture_validation_failures=failures,
        metadata={} if metadata is None else metadata,
        runtime=dict(numpy_version=np.__version__, python_version=platform.python_version(), machine=platform.machine(),
                     longdouble_dtype=str(np.dtype(LD)), longdouble_storage_bits=finfo.bits,
                     longdouble_significand_bits=finfo.nmant + 1, longdouble_epsilon=str(finfo.eps),
                     longdouble_wider_than_float64=bool(finfo.nmant > np.finfo(np.float64).nmant)),
        inputs=dict(full_length=len(h), active_length=kmax-kmin, kmin=kmin, kmax=kmax, num_bins=num_bins,
                    delta_f=float(delta_f), htilde_dtype=str(h.dtype), psd_dtype=None if p is None else str(p.dtype),
                    active_htilde_sha256=array_sha(hs), active_psd_sha256=None if p is None else array_sha(p),
                    rounded_mag_sha256=array_sha(mag), rounded_mag_origin='captured_native' if mag_numpy is not None else 'numpy_source_emulation'),
        normalization=dict(computed_float64=float(norm64), applied_float32=float(norm32),
                           applied_relative_error=float((LD(norm32) - LD(4)*LD(delta_f)) / (LD(4)*LD(delta_f))),
                           public_dtype='float32', threshold_dtype='float64'),
        observed=observed, references=oracle_rows, paths=paths,
        input_power_rounding=dict(
            total_signed_error=float(np.sum(mag.astype(LD)-power, dtype=LD)),
            total_relative_error=float(np.sum(mag.astype(LD)-power, dtype=LD) / references['input_power']['total']),
            max_absolute_cell_error=float(np.max(abs(mag.astype(LD)-power))),
            zeroed_positive_reference_cells=int(np.count_nonzero((power > 0) & (mag == 0)))),
        old_current_max_edge_difference=int(np.max(abs(edge_vectors['old']-edge_vectors['current']))),
        limitation='Reference is power of captured rounded inputs. Error bounds assume IEEE round-to-nearest and finite normal reference arithmetic. Ambiguous boundaries do not establish a winner. No scientific equivalence or accuracy verdict for unsaved data is implied.')
    if arrays is not None:
        arrays.update(htilde=hs.copy(), psd=np.empty(0, np.float32) if p is None else p.copy(),
                      rounded_mag=mag.copy(), input_reference_power=power,
                      input_reference_prefix=references['input_power']['prefix'],
                      rounded_power_reference_prefix=references['rounded_power_scan_only']['prefix'],
                      old_scan=old_scan, current_scan=current_scan,
                      **{name + '_public_prefix': value for name, value in public.items()},
                      **{name + '_edges': value+kmin for name, value in edge_vectors.items()})
    # Fail now if caller metadata or a newly introduced metric is not JSON-safe.
    json.dumps(result, allow_nan=False)
    return result


def _file_sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture_dir', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--fixtures', default='', help='Comma-separated template indices; saves inputs and compact edge arrays')
    parser.add_argument('--indices', default='', help='Optional comma-separated indices for a bounded smoke run')
    parser.add_argument('--require-extended', action='store_true', help='Require longdouble wider than float64')
    args = parser.parse_args()
    root, out = args.capture_dir.resolve(), args.output_dir.resolve()
    if root == out or root.is_relative_to(out) or out.is_relative_to(root):
        parser.error('Capture and output must be separate, non-nested directories')
    if args.require_extended and np.finfo(LD).nmant <= np.finfo(np.float64).nmant:
        parser.error('This platform longdouble is not wider than float64')
    out.mkdir(parents=True, exist_ok=False)
    selected = {int(x) for x in args.fixtures.split(',') if x}
    hashes = {}
    capture = None

    def pin(path):
        hashes[str(path)] = _file_sha(path)

    def load(path):
        pin(path)
        result = np.load(path, allow_pickle=False, mmap_mode='r')
        if capture is not None:
            expected = capture['arrays'][path.name]
            if hashes[str(path)] != expected['sha256'] or array_sha(result) != expected['data_sha256']:
                raise ValueError('Capture array hash mismatch: ' + path.name)
            if list(result.shape) != expected['shape'] or str(result.dtype) != expected['dtype']:
                raise ValueError('Capture array geometry mismatch: ' + path.name)
        return result

    capture_path = root / 'capture.json'
    if capture_path.exists():
        pin(capture_path)
        capture = json.loads(capture_path.read_text())
        if capture.get('status') != 'complete':
            raise ValueError('Capture is not complete')

    def child(name):
        path = (root / name).resolve()
        if path.parent != root:
            raise ValueError('Capture filename must identify a direct child')
        return path

    manifest = root / 'templates.json'
    pin(manifest)
    document = json.loads(manifest.read_text())
    templates = document['templates'] if isinstance(document, dict) else document
    if not isinstance(templates, list) or not templates:
        raise ValueError('templates.json must contain a nonempty list or {"templates": [...]}')
    if len({int(row['index']) for row in templates}) != len(templates):
        raise ValueError('Duplicate template index')
    if args.indices:
        indices = {int(x) for x in args.indices.split(',')}
        templates = [row for row in templates if int(row['index']) in indices]
        if len(templates) != len(indices):
            raise ValueError('Requested template indices not all present')
    psds = {name: load(root / ('psd-' + name + '.npy')) for name in ('original', 'proposed')}
    summary = dict(schema=SCHEMA, template_count=len(templates), rows=0, capture_mismatches=[],
                   by_psd={name: dict(max_old_current_edge_shift=0, input_reference_old_closer_edges=0,
                                     input_reference_current_closer_edges=0, input_reference_same_edges=0,
                                     input_reference_unresolved_edges=0, old_total_error_closer_templates=0,
                                     current_total_error_closer_templates=0, tied_total_error_templates=0)
                           for name in psds})
    with (out / 'metrics.jsonl').open('x') as stream:
        for row in templates:
            index = int(row['index'])
            h = load(child(row.get('file', f'template-{index:03d}.npy')))
            for psd_name, psd in psds.items():
                kwargs = {}
                def captured_file(key, fallback):
                    name = row.get(key)
                    if isinstance(name, dict):
                        name = name.get(psd_name)
                    elif row.get('captured_psd', 'proposed') != psd_name:
                        name = None
                    return child(name or fallback)

                mag_path = captured_file('native_power_file', f'mag-{psd_name}-{index:03d}.npy')
                prefix_path = captured_file('current_prefix_file', f'prefix-{psd_name}-{index:03d}.npy')
                if mag_path.exists():
                    kwargs['mag_numpy'] = load(mag_path)
                if prefix_path.exists():
                    kwargs['captured_current_prefix'] = load(prefix_path)
                bins = row.get('captured_current_bins')
                if isinstance(bins, dict):
                    bins = bins.get(psd_name)
                elif row.get('captured_psd', 'proposed') != psd_name:
                    bins = None
                kwargs['captured_current_bins'] = bins
                metadata = dict(row, audit_psd=psd_name)
                result = audit(h, psd, row['delta_f'], row['kmin'], row['kmax'], row.get('num_bins', 16), metadata, **kwargs)
                stream.write(json.dumps(result, allow_nan=False) + '\n')
                stream.flush()
                summary['rows'] += 1
                if result['status'] != 'ok':
                    summary['capture_mismatches'].append(dict(index=index, psd=psd_name, failures=result['capture_validation_failures']))
                stats = summary['by_psd'][psd_name]
                stats['max_old_current_edge_shift'] = max(stats['max_old_current_edge_shift'], result['old_current_max_edge_difference'])
                counts = result['references']['input_power']['old_current_edge_comparison']['internal_uncertainty_aware_counts']
                for source, dest in (('old_closer', 'old_closer'), ('current_closer', 'current_closer'), ('same_edge', 'same'), ('unresolved', 'unresolved')):
                    stats['input_reference_' + dest + '_edges'] += counts[source]
                old = abs(result['paths']['old']['scan_error_vs_input_power']['total_signed'])
                current = abs(result['paths']['current']['scan_error_vs_input_power']['total_signed'])
                winner = 'old_total_error_closer_templates' if old < current else 'current_total_error_closer_templates' if current < old else 'tied_total_error_templates'
                stats[winner] += 1
                if index in selected:
                    np.savez(out / f'fixture-{psd_name}-{index:03d}.npz', htilde=np.asarray(h), psd=np.asarray(psd),
                             mag=np.asarray(kwargs['mag_numpy']) if 'mag_numpy' in kwargs else np.empty(0, np.float32),
                             old_edges=result['paths']['old']['edges'], current_edges=result['paths']['current']['edges'],
                             reference_edges=result['references']['input_power']['edges'])
                print(json.dumps(dict(index=index, psd=psd_name, status=result['status'], edge_shift=result['old_current_max_edge_difference'])), flush=True)
    summary['input_sha256'] = hashes
    changed = [path for path, digest in hashes.items() if _file_sha(Path(path)) != digest]
    summary['inputs_unchanged_after_analysis'] = not changed
    summary['changed_inputs'] = changed
    summary['module_sha256'] = _file_sha(Path(__file__).resolve())
    summary['status'] = 'fail' if changed or summary['capture_mismatches'] else 'complete'
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if summary['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
