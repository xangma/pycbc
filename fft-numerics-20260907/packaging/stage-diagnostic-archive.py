#!/usr/bin/env python3
"""Prepare/review/copy immutable FFT evidence locally; never run acquisition.

Python >=3.11; NumPy is imported only to validate captured NPZ arrays.
inspect is read-only and deliberately does not read NPZs still downloading.
prepare requires confirmed download completion and writes a local reviewed plan
and lossless transport cache. stage requires that exact plan and a new target.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path('/Users/xangma/repos/pycbc/artifacts')
REVISION = '9578a710479b924e882857c4dffab6ed372a634b'
COMMIT_URL = 'https://github.com/xangma/pycbc/commit/' + REVISION
ATTR = 'torch-fft-attribution-20260907-r2'
FAILED = 'torch-fft-attribution-20260907'
PREC = 'torch-fft-precision-20260907'
R3 = 'torch-current-batch-sweep-20260907-r3'
PUBLICATION = 'torch-benchmark-controls-publication-20260907'
DESTINATION = 'fft-numerics-20260907'
THRESHOLD = 1 * 1024 ** 2
TRANSPORT = 'torch-performance-fix-20260906/publication/archive_transport.py'
TRANSPORT_SHA256 = 'fbad8fc04e36788d4cf2ed9d42c6556568d4a44c028def25b5cb6260ffdbf11e'
CELLS = {'b1-torch_cpu': 36, 'b8-torch_cpu': 36,
         'b1-torch_cuda': 33, 'b8-torch_cuda': 33}
BATCHES = (1, 8, 32, 128, 512, 1024)
ROUTES = ('branch_standard', 'torch_cpu', 'torch_cuda')
STAGES = {f'{kind}-b{batch}-{route}'
          for kind, batches in [('smoke', (1, 8)), ('qual', BATCHES)]
          for batch in batches for route in ROUTES}
TOP = {
    ATTR: 'REPORT.md launch.json launch.log replay.json replay.log replay.py run.py status.json summary.json'.split(),
    FAILED: 'launch.json launch.log replay.json replay.log replay.py run.py status.json'.split(),
    PREC: ('RUN.md assess-normalized-error.py current-batch-accuracy.png current-batch-accuracy.svg '
           'diagnose.py launch.json launch.log normalized-error-audit.json plot-accuracy.py run.py status.json').split(),
    R3: ('PREPARATION.md RUN.md batch-campaign.py batch-plan.json batch-status.json batch-worker.py '
         'campaign_controls.py capacity.json dependency-receipt.json launch-receipt.json launch.log '
         'native-provenance.json provenance.json qualifications.json queued-inputs.json smoke-qualifications.json '
         'stage-source.py staged-files.json staging-receipt.json test_batch_campaign.py test_batch_worker.py').split(),
}
VALIDATION = ('check_fft_precision.py fft-precision-diagnostic.json fft-precision-diagnostic.log '
              'helper-tests.log observed-smoke-status.json runtime-tests.log').split()
OMISSIONS = {
    'all .npy arrays': 'No full R3 output array archive; outputs.npy is omitted, as is validation/selected-reference-rows.npy.',
    'source/ and source.bundle': 'No source checkout, Git data/bundles, or compiled natives; source/native identities and staged-file hashes remain in original receipts.',
    '.git/ .agents/ __pycache__/ and bytecode': 'Local checkout, agent and interpreter state are excluded without traversal.',
    'precision/TRANSFER.md, transfer.log, transfer-receipt.json': 'Ancillary local download records; not frozen diagnostic acquisition.',
    'packaging/': 'Local review plans and transport cache are not original scientific evidence.',
    'R4': 'Separately adopted policy and ongoing campaign; no R4 inputs, results or timing are included.',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def valid_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def finite_json(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite JSON number')
    elif isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'Duplicate JSON key {key}: {path}')
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=pairs)
    finite_json(value)
    return value


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def load_transport(root):
    path = root / TRANSPORT
    require(path.is_file() and not path.is_symlink(), 'Missing plain transport helper')
    require(digest(path) == TRANSPORT_SHA256, 'Reviewed archive transport helper changed')
    spec = importlib.util.spec_from_file_location('diagnostic_archive_transport', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree_inventory(root, transport, include_npz):
    """Explicit records only; unknown entries stop packaging for inspection."""
    files = {}

    def add(name, source=None):
        source = source or transport.safe_path(root, name)
        require(name not in files, f'Duplicate logical path: {name}')
        item = transport.file_record(source)
        if source.suffix == '.json':
            read(source)
        files[name] = dict(source=str(source), **item)

    def children(path):
        require(path.is_dir() and not path.is_symlink(), f'Not a plain directory: {path}')
        return {p.name for p in path.iterdir()}

    for folder, names in TOP.items():
        present = children(transport.safe_path(root, folder))
        allowed = set(names) | {'.git', '.agents', '__pycache__'}
        if folder == ATTR:
            allowed |= {'stage-diagnostic-archive.py', 'ARCHIVE-README.md', 'packaging'}
        if folder == PREC:
            allowed |= set(CELLS) | {cell + '.log' for cell in CELLS} | {'TRANSFER.md', 'transfer.log', 'transfer-receipt.json'}
        if folder == R3:
            allowed |= {'source', 'source.bundle', 'runs', 'stages', 'validation'}
        require(not present - allowed, f'Unreviewed entries in {folder}: {sorted(present - allowed)}')
        for name in names:
            add(folder + '/' + name)
    for cell in CELLS:
        add(PREC + '/' + cell + '.log')
        path = transport.safe_path(root, PREC + '/' + cell)
        names = children(path)
        require(all(n == 'diagnostic.json' or n.endswith('.npz') for n in names),
                f'Unreviewed capture directory entry: {cell}')
        add(PREC + '/' + cell + '/diagnostic.json')
        if include_npz:
            for name in sorted(names - {'diagnostic.json'}):
                add(PREC + '/' + cell + '/' + name)
    require(children(root / R3 / 'runs') == STAGES, 'R3 run directory coverage differs')
    stage_names = children(root / R3 / 'stages')
    required = {name + suffix for name in STAGES for suffix in ('.json', '.log')}
    require(required <= stage_names <= required | {name + '.stderr' for name in STAGES},
            'R3 stage receipt/log coverage differs')
    for name in sorted(stage_names):
        add(R3 + '/stages/' + name)
    for name in sorted(STAGES):
        present = children(root / R3 / 'runs' / name)
        require({'result.json', 'acquisition.json'} <= present
                and all(n in {'result.json', 'acquisition.json'} or n.endswith('.npy') for n in present),
                f'Unreviewed/missing run evidence: {name}')
        for file in ('result.json', 'acquisition.json'):
            add(R3 + '/runs/' + name + '/' + file)
    present = children(root / R3 / 'validation')
    require(set(VALIDATION) <= present and all(n in VALIDATION or n.endswith('.npy') for n in present),
            'Unreviewed/missing R3 validation evidence')
    for name in VALIDATION:
        add(R3 + '/validation/' + name)
    add('README.md', root / ATTR / 'ARCHIVE-README.md')
    add('packaging/stage-diagnostic-archive.py', root / ATTR / 'stage-diagnostic-archive.py')
    add('packaging/archive_transport.py', root / TRANSPORT)
    add('packaging/ARCHIVE_TRANSPORT.md', (root / TRANSPORT).with_name('ARCHIVE_TRANSPORT.md'))
    return dict(sorted(files.items()))


def check_records(root):
    """Cross-check recorded provenance/counts; do not re-evaluate science."""
    r3 = root / R3
    replay = read(root / ATTR / 'replay.json')
    summary = read(root / ATTR / 'summary.json')
    require(replay['state'] == 'complete' and replay['input_hashes_verified'] is True,
            'Attribution replay incomplete')
    require(summary['replay_json_sha256'] == digest(root / ATTR / 'replay.json')
            and summary['source_revision'] == REVISION, 'Attribution summary identity differs')
    rows = replay['rows']
    keys = [(r['cell'], r['block'], r['template_id']) for r in rows]
    require(len(keys) == len(set(keys)) == summary['rows'] == 138, 'Expected 138 distinct captures')
    require(Counter(r['cell'] for r in rows) == CELLS, 'Capture cell counts differ')
    require(len({k[1:] for k in keys}) == summary['unique_row_block_pairs']
            == replay['unique_standard_rows'] == 65, 'Expected 65 distinct reference pairs')
    for folder, state in [(ATTR, 'complete'), (FAILED, 'failed'), (PREC, 'complete')]:
        status = read(root / folder / 'status.json')
        require(status['state'] == state and 'finished' in status, f'Unexpected state: {folder}')
        for name, expected in read(root / folder / 'launch.json')['sha256'].items():
            require(Path(name).name == name and digest(root / folder / name) == expected,
                    f'Frozen acquisition script changed: {folder}/{name}')
    require(replay['script_sha256'] == digest(root / ATTR / 'replay.py'), 'Replay script hash differs')
    status = read(r3 / 'batch-status.json')
    require(status['state'] == 'failed' and len(status['completed']) == 24
            and set(status['completed']) == STAGES and 'finished' in status,
            'Original R3 failed/completed state differs')
    require(read(r3 / 'batch-plan.json')['revision'] == REVISION, 'R3 revision differs')
    staged = read(r3 / 'staged-files.json')
    for name, expected in staged.items():
        require(Path(name).name == name and valid_hash(expected), 'Unsafe staged-file manifest')
        if name != 'source.bundle':  # Git bundle omitted; its recorded hash stays intact.
            require(digest(r3 / name) == expected, f'Frozen R3 helper changed: {name}')
    native = read(r3 / 'native-provenance.json')
    require(native['target_revision'] == REVISION and native['native_build_inputs_diff'] == ''
            and len(native['extensions']) == 11, 'Native provenance differs')
    require(len({r['relative_path'] for r in native['extensions']}) == 11
            and all(valid_hash(r['sha256']) and r['bytes'] > 0 for r in native['extensions']),
            'Malformed native identity records')
    provenance = read(r3 / 'provenance.json')
    require(provenance['source']['commit'] == REVISION and provenance['source']['status'] == ''
            and provenance['native'] == native, 'R3 source/native provenance differs')
    for name, expected in provenance['helper_sha256'].items():
        require(digest(r3 / Path(name).name) == expected, 'R3 provenance helper hash differs')
    results = {}
    aggregates = {kind: read(r3 / file) for kind, file in
                  [('smoke', 'smoke-qualifications.json'), ('qual', 'qualifications.json')]}
    for kind, aggregate in aggregates.items():
        require(set(aggregate) == {name for name in STAGES if name.startswith(kind + '-')},
                f'Aggregate coverage differs: {kind}')
    for name in sorted(STAGES):
        result = read(r3 / 'runs' / name / 'result.json')
        stage = read(r3 / 'stages' / (name + '.json'))
        acquisition = read(r3 / 'runs' / name / 'acquisition.json')
        expected_status = 'pass' if name.startswith('smoke-') or name.endswith('branch_standard') else 'fail'
        require(stage['state'] == 'complete' and 'finished' in stage
                and stage['returncode'] == result['exit_code']
                and result['status'] == expected_status, f'Original verdict differs: {name}')
        require(acquisition['result_sha256'] == digest(r3 / 'runs' / name / 'result.json')
                and aggregates[name.split('-')[0]][name] == result,
                f'Result/acquisition/aggregate mismatch: {name}')
        require(result['source'] == replay['source'] and result['expected_head'] == REVISION
                and result['worker_sha256'] == staged['batch-worker.py']
                and result['tolerance_policy']['version'] == 1, f'Result provenance differs: {name}')
        args = result['arguments']
        blocks = result['pointwise_blocks']
        require(len(blocks) == args['num_blocks'], f'Block coverage differs: {name}')
        for block_id, block in enumerate(blocks):
            observed = block['rows']
            require(block['block'] == block_id and block['processed_once'] is True
                    and block['processing_counts'] == [1] * args['bank_size']
                    and len(observed) == args['bank_size']
                    and {r['template_id'] for r in observed} == set(range(1000, 1000 + args['bank_size'])),
                    f'Row coverage differs: {name}/{block_id}')
            require(all(r['samples'] == args['size'] and r['nonfinite_actual'] == 0
                        and r['nonfinite_reference'] == 0 and valid_hash(r['sha256']) for r in observed),
                    f'Invalid/nonfinite recorded row: {name}/{block_id}')
        require(all(c['passed'] is True for c in result['trigger_comparisons']),
                f'Original trigger comparison differs: {name}')
        results[name] = result
    reference = results['qual-b1-branch_standard']
    require(reference['source']['revision'] == REVISION and reference['source']['tracked_dirty'] is False,
            'Unclean/wrong recorded source')
    for name, result in results.items():
        ref_name = name.split('-')[0] + '-b1-branch_standard'
        require(result['inputs'] == results[ref_name]['inputs']
                and result['input_sha256'] == results[ref_name]['input_sha256'],
                f'Input identity differs across batch/route: {name}')
        if name != ref_name:
            require(result['reference_result_sha256'] == digest(r3 / 'runs' / ref_name / 'result.json'),
                    f'Reference result hash differs: {name}')
    for cell, count in CELLS.items():
        diagnostic = read(root / PREC / cell / 'diagnostic.json')
        name = 'qual-' + cell
        observations = diagnostic['observations']
        require(diagnostic['status'] == 'diagnostic_complete'
                and diagnostic['all_actual_hashes_match'] is True
                and diagnostic['all_reference_hashes_match'] is True
                and diagnostic['frozen_result_sha256'] == digest(r3 / 'runs' / name / 'result.json')
                and diagnostic['diagnostic_sha256'] == digest(root / PREC / 'diagnose.py')
                and diagnostic['campaign_source'] == REVISION
                and diagnostic['input_sha256'] == reference['input_sha256'],
                f'Diagnostic provenance differs: {cell}')
        expected = {(r['block'], r['template_id']) for r in rows if r['cell'] == cell}
        require(len(observations) == count
                and {(r['block'], r['template_id']) for r in observations} == expected,
                f'Diagnostic selected-row coverage differs: {cell}')
    return replay, results


def validate_capture(path, row, actual_row, reference_row, size, np):
    require(digest(path) == row['capture_sha256'], f'NPZ file hash differs: {path}')
    hashes = {}
    with np.load(path, allow_pickle=False) as arrays:
        types = {'actual': np.complex64, 'reference': np.complex64,
                 'correlation': np.complex64, 'exact_correlation': np.complex128}
        require(len(arrays.files) == 4 and set(arrays.files) == set(types), f'NPZ members differ: {path}')
        for name, dtype in types.items():
            value = arrays[name]
            require(value.shape == (size,) and value.dtype == np.dtype(dtype)
                    and np.isfinite(value).all(), f'Invalid/nonfinite captured array: {path}/{name}')
            hashes[name] = hashlib.sha256(value.tobytes(order='C')).hexdigest()
    require(hashes['actual'] == actual_row['sha256']
            and hashes['reference'] == reference_row['sha256']
            and actual_row['reference_sha256'] == reference_row['sha256'],
            f'Captured actual/reference differs from frozen R3 rows: {path}')
    return hashes


def check_captures(root, replay, results, transport):
    import numpy as np
    expected = {PREC + '/' + r['cell'] + f'/block{r["block"]}-template{r["template_id"]}.npz': r
                for r in replay['rows']}
    present = {p.relative_to(root).as_posix()
               for cell in CELLS for p in (root / PREC / cell).glob('*.npz')}
    require(present == set(expected),
            f'Capture inventory differs: missing={len(set(expected) - present)}, extra={len(present - set(expected))}')
    verified = {}
    for name, row in sorted(expected.items()):
        require(all(row[k] is True for k in ('actual_hash_verified', 'reference_hash_verified',
                    'actual_replay_bitwise_equal', 'standard_replay_bitwise_equal')),
                f'Recorded replay validation failed: {name}')
        result = results['qual-' + row['cell']]
        block, tid = row['block'], row['template_id']
        actual_rows = {r['template_id']: r for r in result['pointwise_blocks'][block]['rows']}
        ref_rows = {r['template_id']: r for r in results['qual-b1-branch_standard']['pointwise_blocks'][block]['rows']}
        path = transport.safe_path(root, name)
        transport.file_record(path)
        verified[name] = validate_capture(path, row, actual_rows[tid], ref_rows[tid], result['arguments']['size'], np)
    return {'captures': len(verified), 'unique_reference_pairs': 65, 'array_hashes': verified,
            'arrays_checked_finite': 4 * len(verified), 'R3_array_hash_comparisons': 2 * len(verified),
            'numpy': np.__version__, 'scope': 'Offline identity/shape/finiteness checks; no FFT, input regeneration or acquisition.'}


def publication_target(root):
    publication = root / PUBLICATION
    require(publication.is_dir() and not publication.is_symlink(), 'Missing plain publication checkout')
    status = subprocess.check_output(['git', '-C', str(publication), 'status', '--porcelain',
                                      '--untracked-files=all'], text=True)
    require(status == '', 'Publication checkout must be clean before prepare/stage')
    destination = publication / DESTINATION
    require(not destination.exists() and not destination.is_symlink(), 'Immutable destination already exists')
    return destination


def copy_archive(destination, actual, transport):
    """Never overwrite; a failed partial directory is left for inspection."""
    destination.mkdir()
    for name, item in sorted(actual.items()):
        target = transport.safe_path(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with Path(item['source']).open('rb') as source, target.open('xb') as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        target.chmod(item['mode'])
        require(transport.file_record(target) == transport.metadata(item), f'Archive copy differs: {name}')
    with (destination / 'SHA256SUMS').open('x') as stream:
        stream.write(''.join(f'{item["sha256"]}  {name}\n' for name, item in sorted(actual.items())))
    transport.verify_archive(destination)
    require({p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
            == set(actual) | {'SHA256SUMS'}, 'Actual archive inventory differs')


def self_test(transport):
    """Small synthetic packaging checks only; no campaign input is read."""
    import numpy as np
    count = 0

    def rejects(function):
        nonlocal count
        try:
            function()
        except (ValueError, FileExistsError):
            count += 1
        else:
            raise AssertionError('Negative packaging test unexpectedly passed')

    with tempfile.TemporaryDirectory(prefix='fft-archive-test-') as temporary:
        base = Path(temporary)
        path = base / 'row.npz'
        arrays = {k: np.arange(8, dtype=np.float64).astype(dtype) for k, dtype in
                  [('actual', 'complex64'), ('reference', 'complex64'),
                   ('correlation', 'complex64'), ('exact_correlation', 'complex128')]}
        np.savez(path, **arrays)
        sha = hashlib.sha256(arrays['actual'].tobytes()).hexdigest()
        actual = {'sha256': sha, 'reference_sha256': sha}
        reference = {'sha256': sha}
        row = {'capture_sha256': digest(path)}
        require(len(validate_capture(path, row, actual, reference, 8, np)) == 4, 'Valid capture rejected')
        count += 1
        rejects(lambda: validate_capture(path, {'capture_sha256': '0' * 64}, actual, reference, 8, np))
        rejects(lambda: validate_capture(path, row, dict(actual, sha256='0' * 64), reference, 8, np))
        rejects(lambda: validate_capture(path, row, actual, reference, 9, np))
        arrays['correlation'][0] = complex(float('nan'), 0)
        np.savez(path, **arrays)
        rejects(lambda: validate_capture(path, {'capture_sha256': digest(path)}, actual, reference, 8, np))
        bad = base / 'bad.json'
        for value in ('{"x": NaN}', '{"x": 1e999}', '{"x": 1, "x": 2}'):
            bad.write_text(value)
            rejects(lambda: read(bad))
        source = base / 'large.json'
        source.write_bytes(b' ' * (THRESHOLD + 1))
        logical = {'large.json': dict(source=str(source), **transport.file_record(source))}
        cache = base / 'cache'
        actual_files, mapping = transport.prepare_transport(logical, cache, threshold=THRESHOLD)
        require(mapping['compressed_files'] == 1, '1 MiB compression threshold not applied')
        require(transport.prepare_transport(logical, cache, create=False, threshold=THRESHOLD)
                == (actual_files, mapping), 'Frozen transport not repeatable')
        destination = base / 'archive'
        copy_archive(destination, actual_files, transport)
        restored = base / 'restored'
        transport.restore_archive(destination, restored)
        require(digest(restored / 'large.json') == digest(source), 'Restored bytes differ')
        count += 1
        rejects(lambda: copy_archive(destination, actual_files, transport))
        (destination / 'archive-transport' / (logical['large.json']['sha256'] + '.gz')).write_bytes(b'bad')
        rejects(lambda: transport.verify_archive(destination))
        (base / 'link').symlink_to(source)
        rejects(lambda: transport.safe_path(base, 'link'))
    return {'self_test_checks': count, 'status': 'pass', 'scope': 'Synthetic temporary files only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('inspect', 'validate', 'prepare', 'stage', 'self-test'))
    parser.add_argument('--artifacts-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--work', type=Path, help='New/reviewed plan directory inside attribution-r2/packaging/')
    parser.add_argument('--download-complete', action='store_true', help='All capture transfers have finished')
    args = parser.parse_args()
    root = args.artifacts_root.absolute()
    require(root.is_dir() and not root.is_symlink(), 'Invalid artifacts root')
    transport = load_transport(root)
    if args.mode == 'self-test':
        print(json.dumps(self_test(transport)))
        return
    full = args.mode != 'inspect'
    require(not full or args.download_complete, 'Download must be complete before validate/prepare/stage')
    destination = publication_target(root) if args.mode in ('prepare', 'stage') else None
    inputs = tree_inventory(root, transport, include_npz=full)
    replay, results = check_records(root)
    if not full:
        available = sum(len(list((root / PREC / cell).glob('*.npz'))) for cell in CELLS)
        print(json.dumps({'mode': 'metadata-inspected', 'R3_completed_stages': len(results),
                          'expected_captures': len(replay['rows']), 'NPZ_filenames_present': available,
                          'NPZ_bytes_validated': False, 'destination_written': False,
                          'logical_metadata_files': len(inputs)}))
        return
    verified = check_captures(root, replay, results, transport)
    require(tree_inventory(root, transport, include_npz=True) == inputs, 'Inputs changed during validation')
    if args.mode == 'validate':
        print(json.dumps({k: v for k, v in verified.items() if k != 'array_hashes'}))
        return
    work = (args.work or root / ATTR / 'packaging' / 'v1').absolute()
    packaging = root / ATTR / 'packaging'
    require(work.is_relative_to(packaging) and work != packaging, 'Plan/cache must be inside attribution-r2/packaging/')
    transport.safe_path(root, work.relative_to(root).as_posix())
    plan_path = work / 'plan.json'
    if args.mode == 'prepare':
        require(not work.exists(), 'Plan/cache already exists; inspect it and choose a new --work directory')
        work.mkdir(parents=True)
        manifest = {'schema': 'fft-numerics-evidence-v1', 'source_revision': REVISION,
                    'source_commit_url': COMMIT_URL, 'omissions': OMISSIONS,
                    'original_R3_state': 'failed', 'R3_completed_stages': len(results),
                    'offline_capture_validation': verified,
                    'logical_inputs': {name: transport.metadata(item) for name, item in inputs.items()}}
        (work / 'PACKAGING-MANIFEST.json').write_bytes(canonical(manifest))
    else:
        plan = read(plan_path)
        require(plan['inputs'] == inputs and plan['validation'] == verified
                and plan['destination'] == str(destination), 'Reviewed plan differs from current inputs/validation')
    logical = dict(inputs)
    manifest_path = work / 'PACKAGING-MANIFEST.json'
    logical['PACKAGING-MANIFEST.json'] = dict(source=str(manifest_path), **transport.file_record(manifest_path))
    actual, mapping = transport.prepare_transport(logical, work / 'transport',
                    create=args.mode == 'prepare', threshold=THRESHOLD)
    computed = {'schema': 'fft-numerics-stage-v1', 'inputs': inputs, 'logical_files': logical,
                'files': actual, 'transport': mapping, 'validation': verified,
                'destination': str(destination)}
    require(tree_inventory(root, transport, include_npz=True) == inputs, 'Inputs changed while preparing transport')
    if args.mode == 'prepare':
        with plan_path.open('xb') as stream:
            stream.write(canonical(computed))
    else:
        require(plan == computed, 'Frozen plan/transport cache differs')
        publication_target(root)
        copy_archive(destination, actual, transport)
        require(tree_inventory(root, transport, include_npz=True) == inputs, 'Inputs changed while staging')
    print(json.dumps({'mode': args.mode, 'plan': str(plan_path), 'destination': str(destination),
                      'destination_written': args.mode == 'stage', 'captures_verified': verified['captures'],
                      'logical_files': len(logical), 'archive_files': len(actual) + 1,
                      'transport': mapping}, allow_nan=False))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
